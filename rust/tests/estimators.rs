use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rotor_estimation::common::{Mat4, angle_axis, det4, transform_vector};
use rotor_estimation::multivector::{Multivector, scp};
use rotor_estimation::*;

/// Noise-free correspondences `q_i = R p_i` with unit-norm `p_i` and uniform weights.
fn noise_free(seed: u64, n: usize, angle: f64) -> (Vec<Vec3>, Vec<Vec3>, Vec<f64>, Mat3) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut unit = || {
        Vec3::new(
            rng.random::<f64>() - 0.5,
            rng.random::<f64>() - 0.5,
            rng.random::<f64>() - 0.5,
        )
        .normalize()
    };
    let axis = unit();
    let r = angle_axis(angle, &axis);
    let p: Vec<Vec3> = (0..n).map(|_| unit()).collect();
    let q = p.iter().map(|pi| transform_vector(&r, pi)).collect();
    let rm = nalgebra::UnitQuaternion::from_quaternion(r)
        .to_rotation_matrix()
        .into_inner();
    (p, q, vec![1.0 / n as f64; n], rm)
}

fn quat_matrix(q: &Quat) -> Mat3 {
    nalgebra::UnitQuaternion::from_quaternion(*q)
        .to_rotation_matrix()
        .into_inner()
}

#[test]
fn every_estimator_recovers_a_noise_free_rotation() {
    for (seed, angle) in [(1, 0.3), (2, 1.7), (3, 2.9), (4, 5.5)] {
        let (p, q, w, truth) = noise_free(seed, 200, angle);
        let (p, q, w) = (&p[..], &q[..], &w[..]);
        let quats: [(&str, Quat); 10] = [
            ("flae", flae(p, q, w)),
            ("flae_symbolic", flae_symbolic(p, q, w)),
            ("flae_newton", flae_newton(p, q, w)),
            ("ga_rotor_estimator", ga_rotor_estimator(p, q, w)),
            ("la_rotor_estimator", la_rotor_estimator(p, q, w)),
            ("davenport", davenport(p, q, w)),
            ("quest", quest(p, q, w)),
            ("ga_valkenburg", ga_valkenburg(p, q, w)),
            ("horn", horn(p, q, w)),
            ("quaternion_direct", quaternion_direct(p, q, w)),
        ];
        for (name, r) in quats {
            let diff = (quat_matrix(&r) - truth).abs().max();
            assert!(diff < 1e-6, "{name}, angle {angle}: rotation off by {diff:e}");
            assert!(wahba_error_quat(p, q, &r) < 1e-9, "{name}, angle {angle}");
        }
        let mats: [(&str, Mat3, f64); 5] = [
            ("fa3r_double", fa3r_double(p, q, w), 1e-6),
            ("fa3r_int", fa3r_int(p, q, w), 1e-4), // fixed point with 20 fractional bits
            ("svd_mcadams", svd_mcadams(p, q, w), 1e-4),
            ("svd_eigen", svd_eigen(p, q, w), 1e-6),
            ("foam", foam(p, q, w), 1e-6),
        ];
        for (name, r, tol) in mats {
            let diff = (r - truth).abs().max();
            assert!(diff < tol, "{name}, angle {angle}: rotation off by {diff:e}");
        }
    }
}

#[test]
fn det4_matches_lu_determinant() {
    let mut rng = StdRng::seed_from_u64(7);
    for _ in 0..100 {
        let m = Mat4::from_fn(|_, _| rng.random::<f64>() * 2.0 - 1.0);
        assert!((det4(&m) - m.determinant()).abs() < 1e-12);
    }
}

#[test]
fn svd3_reconstructs_the_matrix() {
    let mut rng = StdRng::seed_from_u64(11);
    for _ in 0..100 {
        let a: [f64; 9] = std::array::from_fn(|_| rng.random::<f64>() * 2.0 - 1.0);
        let (u, s, v) = svd3::svd(&a);
        let usvt = svd3::mult_ab(&svd3::mult_ab(&u, &s), &svd3::transpose(&v));
        for (x, y) in usvt.iter().zip(&a) {
            assert!((x - y).abs() < 1e-5, "{usvt:?} != {a:?}");
        }
    }
}

#[test]
fn multivector_basics() {
    let e1 = Multivector::basis_vector(0);
    let e2 = Multivector::basis_vector(1);
    let e3 = Multivector::basis_vector(2);
    assert_eq!((&e1 * &e1).scalar_part(), 1.0);
    assert_eq!(&e1 ^ &e2, -1.0 * (&e2 ^ &e1));
    assert!((&e1 ^ &e1).blades.is_empty());
    let e12 = &e1 ^ &e2;
    assert_eq!(scp(&e12, &e12), -1.0);
    // dual(a) = a _| I^-1 with I^-1 = -e1^e2^e3 in 3D, so dual(e3) = -e1^e2
    assert_eq!(e3.dual(3), -1.0 * &e12);
    // in 4D I^-1 = I
    let e4 = Multivector::basis_vector(3);
    let i4 = &e1 ^ &e2 ^ &e3 ^ &e4;
    assert_eq!(i4.versor_inverse(), i4);
    assert_eq!((&e1 + &e2 - &e1), e2);
}

/// Like `main.rs`: random norms in [1, 1000], weights proportional to the norm, and every
/// target point also rotated by its own random "noise" rotation.
fn noisy_large_norms(seed: u64, n: usize, angle: f64) -> (Vec<Vec3>, Vec<Vec3>, Vec<f64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let unit = |rng: &mut StdRng| Vec3::new(rng.random::<f64>(), rng.random::<f64>(), rng.random::<f64>()).normalize();
    let axis = unit(&mut rng);
    let r = angle_axis(angle, &axis);
    let (mut p, mut q, mut w) = (vec![], vec![], vec![]);
    for _ in 0..n {
        let norm = rng.random_range(0..1000) as f64 + 1.0;
        let pi = unit(&mut rng) * norm;
        let noise_angle = 2.0 * std::f64::consts::PI * rng.random::<f64>();
        let noise = angle_axis(noise_angle, &unit(&mut rng));
        q.push(transform_vector(&(r * noise), &pi));
        p.push(pi);
        w.push(norm);
    }
    let total: f64 = w.iter().sum();
    w.iter_mut().for_each(|x| *x /= total);
    (p, q, w)
}

/// Each estimator's Wahba error, which must match the optimum (Horn's exact eigen-solution).
fn errors(p: &[Vec3], q: &[Vec3], w: &[f64]) -> Vec<(&'static str, f64)> {
    vec![
        ("flae", wahba_error_quat(p, q, &flae(p, q, w))),
        ("flae_symbolic", wahba_error_quat(p, q, &flae_symbolic(p, q, w))),
        ("flae_newton", wahba_error_quat(p, q, &flae_newton(p, q, w))),
        ("fa3r_double", wahba_error_mat(p, q, &fa3r_double(p, q, w))),
        (
            "la_rotor_estimator",
            wahba_error_quat(p, q, &la_rotor_estimator(p, q, w)),
        ),
        ("davenport", wahba_error_quat(p, q, &davenport(p, q, w))),
        ("quest", wahba_error_quat(p, q, &quest(p, q, w))),
        ("foam", wahba_error_mat(p, q, &foam(p, q, w))),
        ("ga_valkenburg", wahba_error_quat(p, q, &ga_valkenburg(p, q, w))),
        ("svd_mcadams", wahba_error_mat(p, q, &svd_mcadams(p, q, w))),
        ("svd_eigen", wahba_error_mat(p, q, &svd_eigen(p, q, w))),
        ("quaternion_direct", wahba_error_quat(p, q, &quaternion_direct(p, q, w))),
    ]
}

fn assert_optimal(p: &[Vec3], q: &[Vec3], w: &[f64], what: &str) {
    let optimum = wahba_error_quat(p, q, &horn(p, q, w));
    for (name, e) in errors(p, q, w) {
        assert!(
            (e - optimum).abs() <= 1e-7 * optimum + 1e-9,
            "{what}: {name} error {e:e}, optimum {optimum:e}"
        );
    }
}

#[test]
fn newton_based_estimators_find_the_largest_root_with_large_norms() {
    // FlaeNewton, QUEST and FOAM used to start Newton at 1 and stop at a smaller root;
    // FA3R used to iterate on the unnormalized matrix.
    for seed in 1..9 {
        let angle = 4.0 * std::f64::consts::PI * seed as f64 / 9.0;
        let (p, q, w) = noisy_large_norms(seed, 1000, angle);
        assert_optimal(&p, &q, &w, &format!("seed {seed}"));
    }
}

#[test]
fn fa3r_int_does_not_overflow_with_large_norms() {
    let (p, q, w, truth) = noise_free(5, 200, 0.8);
    let scaled: Vec<Vec3> = p.iter().map(|x| x * 1000.0).collect();
    let target: Vec<Vec3> = q.iter().map(|x| x * 1000.0).collect();
    let diff = (fa3r_int(&scaled, &target, &w) - truth).abs().max();
    assert!(diff < 1e-4, "fa3r_int off by {diff:e}");
}

#[test]
fn svd_returns_a_rotation_for_planar_data() {
    // With all points in a plane the third singular direction is arbitrary, and
    // V U^T is a reflection for one of its two signs.
    for seed in 0..20 {
        let (p, _, w, _) = noise_free(100 + seed, 50, 0.0);
        let planar: Vec<Vec3> = p.iter().map(|x| Vec3::new(x.x, x.y, 0.0)).collect();
        let r = angle_axis(0.3 * seed as f64, &Vec3::new(0.3, -0.5, 0.8).normalize());
        let target: Vec<Vec3> = planar.iter().map(|x| transform_vector(&r, x)).collect();
        for (name, m) in [
            ("svd_eigen", svd_eigen(&planar, &target, &w)),
            ("svd_mcadams", svd_mcadams(&planar, &target, &w)),
        ] {
            assert!(
                (m.determinant() - 1.0).abs() < 1e-9,
                "{name}, seed {seed}: det = {}",
                m.determinant()
            );
        }
    }
}

#[test]
fn rotations_of_180_degrees() {
    // QUEST's (gamma, x) vanishes at 180 degrees (method of sequential rotations).
    for (seed, axis) in [
        (7, Vec3::new(0.0, 0.0, 1.0)),
        (8, Vec3::new(1.0, 0.0, 0.0)),
        (9, Vec3::new(0.2, 0.7, -0.4)),
    ] {
        let (p, _, w, _) = noise_free(seed, 100, 0.0);
        let r = angle_axis(std::f64::consts::PI, &axis.normalize());
        let q: Vec<Vec3> = p.iter().map(|x| transform_vector(&r, x)).collect();
        assert_optimal(&p, &q, &w, &format!("axis {axis:?}"));
    }
}

#[test]
fn flae_with_a_rotation_axis_in_the_xy_plane() {
    // q_z = 0, where FLAE's elimination (which fixes q_z = -1) breaks down.
    for (seed, angle) in [(10, 1.0), (11, 2.0), (12, 0.3)] {
        let (p, _, w, _) = noise_free(seed, 100, 0.0);
        let r = angle_axis(angle, &Vec3::new(0.6, 0.8, 0.0));
        let q: Vec<Vec3> = p.iter().map(|x| transform_vector(&r, x)).collect();
        assert_optimal(&p, &q, &w, &format!("angle {angle}"));
    }
}

#[test]
fn rotor_estimators_on_rotations_with_alternating_component_sum_zero() {
    // The sum of the four triple wedges is c q (q_w - q_x + q_y - q_z), which vanishes for
    // these rotations (e.g. 90 degrees about z); the estimators keep the best single wedge.
    let qs: [(f64, f64, f64, f64); 5] = [
        (1.0, 0.0, 0.0, 1.0),
        (1.0, 0.0, -1.0, 0.0),
        (1.0, 1.0, 0.0, 0.0),
        (1.0, 1.0, 1.0, 1.0),
        (0.3, 0.9, 0.2, -0.4),
    ];
    for (i, &(w0, x0, y0, z0)) in qs.iter().enumerate() {
        assert!((w0 - x0 + y0 - z0).abs() < 1e-12);
        let r = rotor_estimation::common::normalize_quat(rotor_estimation::common::quat(w0, x0, y0, z0));
        let (p, _, w, _) = noise_free(300 + i as u64, 100, 0.0);
        let q: Vec<Vec3> = p.iter().map(|x| transform_vector(&r, x)).collect();
        for (name, est) in [
            ("la_rotor_estimator", la_rotor_estimator(&p, &q, &w)),
            ("ga_rotor_estimator", ga_rotor_estimator(&p, &q, &w)),
        ] {
            let e = wahba_error_quat(&p, &q, &est);
            assert!(e < 1e-12, "{name}, q = {:?}: error {e:e}", (w0, x0, y0, z0));
        }
    }
}

/// `sum w q.(R p R~) - beta |R - prior|^2` for a unit quaternion `r`, with the same sign as `prior`.
fn prior_objective(p: &[Vec3], q: &[Vec3], w: &[f64], prior: &Quat, beta: f64, r: &Quat) -> f64 {
    let gain: f64 = p.iter().zip(q).zip(w).map(|((pi, qi), &wi)| wi * qi.dot(&transform_vector(r, pi))).sum();
    let d = r.coords - prior.coords;
    gain - beta * d.norm_squared()
}

/// The maximum over the unit sphere by a dense start grid and projected-gradient refinement.
fn prior_brute_force(p: &[Vec3], q: &[Vec3], w: &[f64], prior: &Quat, beta: f64) -> f64 {
    let f = |c: &nalgebra::Vector4<f64>| {
        let c = c.normalize();
        prior_objective(p, q, w, prior, beta, &Quat::new(c[3], c[0], c[1], c[2]))
    };
    let mut best = f64::MIN;
    let mut best_c = nalgebra::Vector4::zeros();
    let n = 16;
    for i in 0..=n {
        for j in 0..=n {
            for k in 0..=2 * n {
                let (a, b, c) = (
                    std::f64::consts::PI * i as f64 / n as f64,
                    std::f64::consts::PI * j as f64 / n as f64,
                    std::f64::consts::PI * k as f64 / n as f64,
                );
                let v = nalgebra::Vector4::new(a.cos(), a.sin() * b.cos(), a.sin() * b.sin() * c.cos(), a.sin() * b.sin() * c.sin());
                let fv = f(&v);
                if fv > best {
                    best = fv;
                    best_c = v;
                }
            }
        }
    }
    let mut c = best_c;
    for _ in 0..4000 {
        let eps = 1e-6;
        let g = nalgebra::Vector4::from_fn(|i, _| {
            let mut d = nalgebra::Vector4::zeros();
            d[i] = eps;
            (f(&(c + d)) - f(&(c - d))) / (2.0 * eps)
        });
        let t = g - c * c.dot(&g);
        c = (c + t * 0.05).normalize();
    }
    best.max(f(&c))
}

#[test]
fn prior_rotor_estimator_reduces_to_the_estimator_and_to_the_prior() {
    let (p, q, w, _) = noise_free(11, 50, 2.0);
    let prior = angle_axis(1.0, &Vec3::new(0.3, -0.5, 0.8).normalize());
    let plain = la_rotor_estimator(&p, &q, &w);
    let zero = prior_rotor_estimator(&p, &q, &w, &prior, 0.0);
    assert!(nalgebra::UnitQuaternion::from_quaternion(zero).angle_to(&nalgebra::UnitQuaternion::from_quaternion(plain)) < 1e-8);
    let strong = prior_rotor_estimator(&p, &q, &w, &prior, 1e9);
    assert!((strong.coords - prior.coords).norm() < 1e-6, "follows the prior, with its sign");
    let flipped = Quat::from_vector(-prior.coords);
    let strong = prior_rotor_estimator(&p, &q, &w, &flipped, 1e9);
    assert!((strong.coords + prior.coords).norm() < 1e-6, "and with the opposite sign");
}

#[test]
fn prior_rotor_estimator_is_the_global_maximum() {
    let mut rng = StdRng::seed_from_u64(5);
    for seed in 0..40 {
        let (p, q, w) = noisy_large_norms(seed, 20, 1.0 + seed as f64 * 0.13);
        let axis = Vec3::new(rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5).normalize();
        let prior = angle_axis(rng.random::<f64>() * 6.0, &axis);
        let s: f64 = p.iter().zip(&q).zip(&w).map(|((a, b), &c)| c * (a.norm_squared() + b.norm_squared())).sum();
        for beta in [1e-3 * s, 0.1 * s, 2.0 * s] {
            let r = prior_rotor_estimator(&p, &q, &w, &prior, beta);
            assert!((r.norm() - 1.0).abs() < 1e-12);
            let f = prior_objective(&p, &q, &w, &prior, beta, &r);
            let brute = prior_brute_force(&p, &q, &w, &prior, beta);
            assert!(f >= brute - 1e-9 * (1.0 + s), "seed {seed}, beta {beta}: {f} < {brute}");
        }
    }
}

#[test]
fn prior_rotor_estimator_with_a_prior_half_a_turn_away() {
    // the prior's quaternion orthogonal to the data's best rotor: the hard case, weak and strong
    for seed in 0..20 {
        let (p, q, w) = noisy_large_norms(seed, 20, 0.7 + 0.2 * seed as f64);
        let data = la_rotor_estimator(&p, &q, &w);
        let mut rng = StdRng::seed_from_u64(seed + 100);
        let axis = Vec3::new(rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5, rng.random::<f64>() - 0.5).normalize();
        let prior = data * angle_axis(std::f64::consts::PI, &axis);
        let s: f64 = p.iter().zip(&q).zip(&w).map(|((a, b), &c)| c * (a.norm_squared() + b.norm_squared())).sum();
        for beta in [0.005 * s, 0.05 * s, 0.5 * s] {
            let r = prior_rotor_estimator(&p, &q, &w, &prior, beta);
            assert!((r.norm() - 1.0).abs() < 1e-12 && r.coords.iter().all(|c| c.is_finite()));
            let f = prior_objective(&p, &q, &w, &prior, beta, &r);
            let brute = prior_brute_force(&p, &q, &w, &prior, beta);
            assert!(f >= brute - 1e-9 * (1.0 + s), "seed {seed}, beta {beta}: {f} < {brute}");
        }
    }
}
