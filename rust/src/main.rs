//! Accuracy and speed comparison of the rotation estimators.
//!
//! Usage (all optional):
//!   estimator --name=out.txt --plane=[x|y|z] --n=1000 --noise=[true|false]
//!             --norm=<double> --axis=<x,y,z> --angle=<double> --seed=<u64>

use std::f64::consts::PI;
use std::fs::File;
use std::hint::black_box;
use std::io::{BufWriter, Write};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use rotor_estimation::common::{angle_axis, transform_vector};
use rotor_estimation::*;

const BENCHMARK_ITERATIONS: usize = 1_000_000;

/// C++ `std::ostream` output with `precision(15)`, i.e. printf's `%.15g`.
fn fmt_g(x: f64) -> String {
    const P: i32 = 15;
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.into();
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0" } else { "0" }.into();
    }
    fn strip(s: &str) -> &str {
        if s.contains('.') {
            s.trim_end_matches('0').trim_end_matches('.')
        } else {
            s
        }
    }
    let sci = format!("{:.*e}", (P - 1) as usize, x);
    let (mantissa, exp) = sci.split_once('e').expect("exponent");
    let exp: i32 = exp.parse().expect("integer exponent");
    if !(-4..P).contains(&exp) {
        format!(
            "{}e{}{:02}",
            strip(mantissa),
            if exp < 0 { '-' } else { '+' },
            exp.abs()
        )
    } else {
        strip(&format!("{:.*}", (P - 1 - exp) as usize, x)).to_string()
    }
}

/// Writes every line both to the log file and to stdout.
struct Log {
    file: BufWriter<File>,
}

impl Log {
    fn new(filename: &str) -> Self {
        let file = File::create(filename).unwrap_or_else(|e| panic!("cannot create {filename}: {e}"));
        Self {
            file: BufWriter::new(file),
        }
    }

    fn line(&mut self, s: &str) {
        println!("{s}");
        writeln!(self.file, "{s}").expect("write log");
        self.file.flush().expect("flush log");
    }
}

#[derive(Clone, Copy, PartialEq)]
enum DataPlane {
    None,
    X,
    Y,
    Z,
}

struct Params {
    filename: String,
    data_plane: DataPlane,
    n: usize,
    noise: bool,
    norm: Option<f64>,
    axis: Option<Vec3>,
    angle: Option<f64>,
    seed: Option<u64>,
}

impl Params {
    fn parse(args: impl Iterator<Item = String>) -> Self {
        let mut p = Params {
            filename: "out.txt".into(),
            data_plane: DataPlane::None,
            n: 1000,
            noise: true,
            norm: None,
            axis: None,
            angle: None,
            seed: None,
        };
        let num = |key: &str, v: &str| -> f64 { v.parse().unwrap_or_else(|_| panic!("{key}: '{v}' is not a number")) };
        for arg in args {
            let (key, raw_value) = match arg.split_once('=') {
                Some((k, v)) if !k.is_empty() => (k.to_lowercase(), v.to_string()),
                _ => (arg.clone(), String::new()),
            };
            // Values are case-insensitive, except the output file name, which the
            // C++ version lowercased too (breaking paths on case-sensitive filesystems).
            let value = raw_value.to_lowercase();
            match key.as_str() {
                "--name" => p.filename = raw_value,
                "--plane" => match value.as_str() {
                    "x" => p.data_plane = DataPlane::X,
                    "y" => p.data_plane = DataPlane::Y,
                    "z" => p.data_plane = DataPlane::Z,
                    _ => {}
                },
                "--n" => {
                    p.n = value
                        .parse()
                        .unwrap_or_else(|_| panic!("--n: '{value}' is not a count"))
                }
                "--noise" => p.noise = value == "true",
                "--norm" => p.norm = Some(num("--norm", &value)),
                "--axis" => {
                    let c: Vec<f64> = value.split(',').map(|s| num("--axis", s.trim())).collect();
                    assert!(c.len() == 3, "--axis needs three comma-separated numbers");
                    p.axis = Some(Vec3::new(c[0], c[1], c[2]));
                }
                "--angle" => p.angle = Some(num("--angle", &value)),
                "--seed" => {
                    p.seed = Some(
                        value
                            .parse()
                            .unwrap_or_else(|_| panic!("--seed: '{value}' is not a u64")),
                    )
                }
                _ => {}
            }
        }
        p
    }

    fn show_usage(&self, log: &mut Log) {
        let plane = match self.data_plane {
            DataPlane::X => "x",
            DataPlane::Y => "y",
            DataPlane::Z => "z",
            DataPlane::None => "[x|y|z]",
        };
        let mut s = format!(
            "       --name={} --plane={} --n={} --noise={}",
            self.filename, plane, self.n, self.noise
        );
        match self.norm {
            Some(v) => s += &format!(" --norm={}", fmt_g(v)),
            None => s += " --norm=[double]",
        }
        match self.axis {
            Some(a) => s += &format!(" --axis={},{},{}", fmt_g(a.x), fmt_g(a.y), fmt_g(a.z)),
            None => s += " --axis=[double,double,double]",
        }
        match self.angle {
            Some(v) => s += &format!(" --angle={}", fmt_g(v)),
            None => s += " --angle=[double]",
        }
        match self.seed {
            Some(v) => s += &format!(" --seed={v}"),
            None => s += " --seed=[u64]",
        }
        log.line("Params: ");
        log.line(&s);
        log.line("");
    }
}

/// Normalized vector with components uniform in [0, 1).
fn random_vector(rng: &mut StdRng) -> Vec3 {
    Vec3::new(rng.random(), rng.random(), rng.random()).normalize()
}

type Estimator<R> = fn(&[Vec3], &[Vec3], &[f64]) -> R;

/// Seconds spent in `BENCHMARK_ITERATIONS` calls of `estimator`.
fn benchmark<R>(p: &[Vec3], q: &[Vec3], w: &[f64], estimator: Estimator<R>) -> f64 {
    let start = Instant::now();
    for _ in 0..BENCHMARK_ITERATIONS {
        black_box(estimator(black_box(p), black_box(q), black_box(w)));
    }
    start.elapsed().as_secs_f64()
}

fn main() {
    let params = Params::parse(std::env::args().skip(1));
    let seed = params
        .seed
        .unwrap_or_else(|| SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs()));
    let mut rng = StdRng::seed_from_u64(seed);

    let mut log = Log::new(&params.filename);
    params.show_usage(&mut log);

    let n = params.n;
    let mut points_original = Vec::with_capacity(n);
    let mut weights = Vec::with_capacity(n);
    let mut total_norms = 0.0;
    for _ in 0..n {
        let norm = params.norm.unwrap_or_else(|| rng.random_range(0..1000) as f64 + 1.0); // 1 to 1000
        let mut v = random_vector(&mut rng) * norm;
        match params.data_plane {
            DataPlane::X => v[0] = 0.0,
            DataPlane::Y => v[1] = 0.0,
            DataPlane::Z => v[2] = 0.0,
            DataPlane::None => {}
        }
        points_original.push(v);
        weights.push(norm);
        total_norms += norm;
    }
    for w in &mut weights {
        *w /= total_norms;
    }

    let axis = params.axis.unwrap_or_else(|| random_vector(&mut rng));
    let angle = params.angle.unwrap_or_else(|| 4.0 * PI * rng.random::<f64>());
    let rotation = angle_axis(angle, &axis);

    let points_transformed: Vec<Vec3> = points_original
        .iter()
        .map(|p| {
            if params.noise {
                let noise_angle = 2.0 * PI * rng.random::<f64>();
                let noise_axis = random_vector(&mut rng);
                transform_vector(&(rotation * angle_axis(noise_angle, &noise_axis)), p)
            } else {
                transform_vector(&rotation, p)
            }
        })
        .collect();

    let (p, q, w) = (&points_original[..], &points_transformed[..], &weights[..]);

    let errors = [
        ("FLAE", wahba_error_quat(p, q, &flae(p, q, w))),
        ("FLAE Symbolic", wahba_error_quat(p, q, &flae_symbolic(p, q, w))),
        ("FLAE Newton", wahba_error_quat(p, q, &flae_newton(p, q, w))),
        ("FA3R Double", wahba_error_mat(p, q, &fa3r_double(p, q, w))),
        ("FA3R Int", wahba_error_mat(p, q, &fa3r_int(p, q, w))),
        (
            "GA Rotor Estimator",
            wahba_error_quat(p, q, &ga_rotor_estimator(p, q, w)),
        ),
        (
            "LA Rotor Estimator",
            wahba_error_quat(p, q, &la_rotor_estimator(p, q, w)),
        ),
        ("Davenport", wahba_error_quat(p, q, &davenport(p, q, w))),
        ("Quest", wahba_error_quat(p, q, &quest(p, q, w))),
        ("Foam", wahba_error_mat(p, q, &foam(p, q, w))),
        ("GA Valkenburg", wahba_error_quat(p, q, &ga_valkenburg(p, q, w))),
        ("SVD McAdams", wahba_error_mat(p, q, &svd_mcadams(p, q, w))),
        ("SVD", wahba_error_mat(p, q, &svd_eigen(p, q, w))),
        ("Horn", wahba_error_quat(p, q, &horn(p, q, w))),
        ("Quaternion Direct", wahba_error_quat(p, q, &quaternion_direct(p, q, w))),
    ];

    log.line(&format!(
        "Axis                                      : {}, {}, {}",
        fmt_g(axis.x),
        fmt_g(axis.y),
        fmt_g(axis.z)
    ));
    log.line(&format!("Angle                                     : {}", fmt_g(angle)));
    log.line("Comparisons: ");
    log.line("");
    for (name, error) in errors {
        log.line(&format!("{:<42}{}", format!("{name} error"), fmt_g(error)));
    }

    let timed = |name: &str, secs: f64, log: &mut Log| {
        log.line(&format!("{:<42}{} sec.", format!("Exec time {name}:"), fmt_g(secs)));
    };
    timed("FLAE", benchmark(p, q, w, flae), &mut log);
    timed("FLAE Symbolic", benchmark(p, q, w, flae_symbolic), &mut log);
    timed("FLAE Newton", benchmark(p, q, w, flae_newton), &mut log);
    timed("FA3R Double", benchmark(p, q, w, fa3r_double), &mut log);
    timed("FA3R Int", benchmark(p, q, w, fa3r_int), &mut log);
    timed("LARotorEstimator", benchmark(p, q, w, la_rotor_estimator), &mut log);
    timed("Davenport Q-Method", benchmark(p, q, w, davenport), &mut log);
    timed("QUEST", benchmark(p, q, w, quest), &mut log);
    timed("FOAM", benchmark(p, q, w, foam), &mut log);
    timed("GA Valkenburg", benchmark(p, q, w, ga_valkenburg), &mut log);
    timed("SVD McAdams", benchmark(p, q, w, svd_mcadams), &mut log);
    timed("SVD", benchmark(p, q, w, svd_eigen), &mut log);
    timed("Horn", benchmark(p, q, w, horn), &mut log);

    log.line("FINISHED...");
}

#[cfg(test)]
mod tests {
    use super::fmt_g;

    #[test]
    fn fmt_g_matches_printf() {
        let cases = [
            (0.1, "0.1"),
            (1e-7, "1e-07"),
            (1.2345678901234567e17, "1.23456789012346e+17"),
            (std::f64::consts::PI, "3.14159265358979"),
            (226903773.0, "226903773"),
            (669.5302660123456, "669.530266012346"),
            (1e15, "1e+15"),
            (999999999999999.9, "1e+15"),
            (1e-5, "1e-05"),
            (0.0001, "0.0001"),
            (123456.0, "123456"),
            (-2.5e-300, "-2.5e-300"),
            (5.0, "5"),
            (0.0, "0"),
        ];
        for (x, expected) in cases {
            assert_eq!(fmt_g(x), expected, "fmt_g({x:e})");
        }
    }
}
