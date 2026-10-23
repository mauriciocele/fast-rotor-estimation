//! FLAE: Fast Linear Attitude Estimator (closed-form, symbolic and Newton variants).

use nalgebra::Complex;

use crate::common::*;

/// FLAE fixes the last quaternion component (q_z = -1) and eliminates the first three
/// rows of G = W - lambda I. That breaks down when q_z ~ 0 (e.g. any rotation whose axis
/// lies in the xy-plane): the leading 3x3 block of G becomes singular and the result is
/// rounding noise. Since adj(G) = -p'(lambda) q q^T for the characteristic polynomial p,
/// that block's determinant is -p'(lambda) q_z^2, which tells when q_z^2 < 0.01. Then the
/// eigenvector is read from the column of adj(G) with the largest diagonal entry instead,
/// the best conditioned one ([`adjugate_null_vector`]).
#[inline]
fn needs_adjugate(leading_minor: f64, dp: f64) -> bool {
    leading_minor.is_nan() || dp.is_nan() || leading_minor.abs() < 0.01 * dp.abs()
}

/// p'(lambda) for p(lambda) = lambda^4 + a lambda^2 + b lambda + c
#[inline]
fn dp(a: f64, b: f64, lambda: f64) -> f64 {
    4.0 * lambda * lambda * lambda + 2.0 * a * lambda + b
}

pub fn flae_symbolic(p: &[Vec3], q: &[Vec3], w: &[f64]) -> Quat {
    let sigma = weighted_outer_sum(p, q, w);

    let a_ = sigma - sigma.transpose();
    let mut tmp = sigma + sigma.transpose();
    let d = Vec3::new(a_[(1, 2)], a_[(2, 0)], a_[(0, 1)]);
    let mut qq = Mat4::zeros();
    qq[(0, 0)] = sigma[(0, 0)] + sigma[(1, 1)] + sigma[(2, 2)];
    tmp[(0, 0)] -= qq[(0, 0)];
    tmp[(1, 1)] -= qq[(0, 0)];
    tmp[(2, 2)] -= qq[(0, 0)];
    qq[(0, 1)] = d.x;
    qq[(0, 2)] = d.y;
    qq[(0, 3)] = d.z;
    qq[(1, 0)] = d.x;
    qq[(2, 0)] = d.y;
    qq[(3, 0)] = d.z;
    for i in 0..3 {
        for j in 0..3 {
            qq[(i + 1, j + 1)] = tmp[(i, j)];
        }
    }

    let c = det4(&qq);
    let b = -8.0 * sigma.determinant();
    let a = -2.0 * sigma.norm_squared();

    let t0 = 2.0 * a * a * a + 27.0 * b * b - 72.0 * a * c;
    let tt = a * a + 12.0 * c;
    let theta = (4.0 * tt * tt * tt - t0 * t0).sqrt().atan2(t0);
    let a_t1 = 1.259921049894873 * tt.sqrt() * (theta * 0.333333333333333333).cos();
    let t2 = (-4.0 * a + 3.174802103936399 * a_t1).sqrt();
    let lambda = 0.204124145231932 * (t2 + (-t2 * t2 - 12.0 * a - 29.393876913398135 * b / t2).sqrt());

    let g11 = qq[(0, 0)] - lambda;
    let (g12, g13, g14) = (qq[(0, 1)], qq[(0, 2)], qq[(0, 3)]);
    let g22 = qq[(1, 1)] - lambda;
    let (g23, g24) = (qq[(1, 2)], qq[(1, 3)]);
    let g33 = qq[(2, 2)] - lambda;
    let g34 = qq[(2, 3)];

    let q_res = quat(
        g14 * g23 * g23 - g13 * g23 * g24 - g14 * g22 * g33 + g12 * g24 * g33 + g13 * g22 * g34 - g12 * g23 * g34,
        g13 * g13 * g24 + g12 * g14 * g33 - g11 * g24 * g33 + g11 * g23 * g34 - g13 * g14 * g23 - g13 * g12 * g34,
        g13 * g14 * g22 - g12 * g14 * g23 - g12 * g13 * g24 + g11 * g23 * g24 + g12 * g12 * g34 - g11 * g22 * g34,
        -(g13 * g13 * g22 - 2.0 * g12 * g13 * g23 + g11 * g23 * g23 + g12 * g12 * g33 - g11 * g22 * g33),
    );
    // q_res.k is the leading 3x3 minor of G
    let q_res = if needs_adjugate(q_res.k, dp(a, b, lambda)) {
        let mut g = qq;
        for i in 0..4 {
            g[(i, i)] -= lambda;
        }
        let v = adjugate_null_vector(&g);
        quat(v[0], v[1], v[2], v[3])
    } else {
        q_res
    };
    normalize_quat(q_res)
}

/// The 4x4 FLAE matrix W and the coefficients (a, b, c) of its characteristic
/// polynomial `lambda^4 + a lambda^2 + b lambda + c`, from `H = sum w_j Q_j P_j^T` (`mm`).
fn flae_matrix(mm: &Mat3) -> (Mat4, f64, f64, f64) {
    let (hx1, hx2, hx3) = (mm[(0, 0)], mm[(0, 1)], mm[(0, 2)]);
    let (hy1, hy2, hy3) = (mm[(1, 0)], mm[(1, 1)], mm[(1, 2)]);
    let (hz1, hz2, hz3) = (mm[(2, 0)], mm[(2, 1)], mm[(2, 2)]);

    let mut wm = Mat4::zeros();
    wm[(0, 0)] = hx1 + hy2 + hz3;
    wm[(0, 1)] = -hy3 + hz2;
    wm[(0, 2)] = -hz1 + hx3;
    wm[(0, 3)] = -hx2 + hy1;
    wm[(1, 1)] = hx1 - hy2 - hz3;
    wm[(1, 2)] = hx2 + hy1;
    wm[(1, 3)] = hx3 + hz1;
    wm[(2, 2)] = hy2 - hx1 - hz3;
    wm[(2, 3)] = hy3 + hz2;
    wm[(3, 3)] = hz3 - hy2 - hx1;
    symmetrize_from_upper(&mut wm);

    let c = det4(&wm);
    let b = 8.0 * hx3 * hy2 * hz1 - 8.0 * hx2 * hy3 * hz1 - 8.0 * hx3 * hy1 * hz2
        + 8.0 * hx1 * hy3 * hz2
        + 8.0 * hx2 * hy1 * hz3
        - 8.0 * hx1 * hy2 * hz3;
    let a = -2.0 * hx1 * hx1
        - 2.0 * hx2 * hx2
        - 2.0 * hx3 * hx3
        - 2.0 * hy1 * hy1
        - 2.0 * hy2 * hy2
        - 2.0 * hy3 * hy3
        - 2.0 * hz1 * hz1
        - 2.0 * hz2 * hz2
        - 2.0 * hz3 * hz3;
    (wm, a, b, c)
}

/// Gauss-Jordan elimination of the first three rows of `W - lambda I`;
/// the eigenvector is read from the last column. `dp` is p'(lambda) (see `needs_adjugate`).
fn flae_eigenvector(mut wm: Mat4, lambda: f64, dp: f64) -> Quat {
    for i in 0..4 {
        wm[(i, i)] -= lambda;
    }
    if needs_adjugate(minor3(&wm, 3, 3), dp) {
        let v = adjugate_null_vector(&wm);
        return normalize_quat(quat(v[0], v[1], v[2], v[3]));
    }
    let row = |i: usize| -> [f64; 4] { [wm[(i, 0)], wm[(i, 1)], wm[(i, 2)], wm[(i, 3)]] };
    let (mut w0, mut w1, mut w2) = (row(0), row(1), row(2));
    let scale = |v: &mut [f64; 4], s: f64| v.iter_mut().for_each(|x| *x /= s);
    let axpy = |v: &mut [f64; 4], k: f64, u: &[f64; 4]| {
        for (x, y) in v.iter_mut().zip(u) {
            *x -= k * y;
        }
    };

    let s = w0[0];
    scale(&mut w0, s);
    let k = w1[0];
    axpy(&mut w1, k, &w0);
    let k = w2[0];
    axpy(&mut w2, k, &w0);

    let s = w1[1];
    scale(&mut w1, s);
    let k = w0[1];
    axpy(&mut w0, k, &w1);
    let k = w2[1];
    axpy(&mut w2, k, &w1);

    let s = w2[2];
    scale(&mut w2, s);
    let k = w0[2];
    axpy(&mut w0, k, &w2);
    let k = w1[2];
    axpy(&mut w1, k, &w2);

    normalize_quat(quat(w0[3], w1[3], w2[3], -1.0))
}

pub fn flae(p: &[Vec3], q: &[Vec3], w: &[f64]) -> Quat {
    const POW243: f64 = 2.5198420997897463295344212145565; // pow(2, 4.0 / 3.0)
    const POW223: f64 = 1.5874010519681994747517056392723; // pow(2, 2.0 / 3.0)
    const K: f64 = 0.20412414523193150818310700622549;

    let (wm, a, b, c) = flae_matrix(&weighted_outer_sum(q, p, w));

    let a2 = a * a;
    let t0 = 2.0 * a2 * a + 27.0 * b * b - 72.0 * a * c;
    let s = a2 + 12.0 * c;
    let t1 = (t0 + Complex::new(-4.0 * s * s * s + t0 * t0, 0.0).sqrt()).powf(0.33333333333333333333333333333333);
    let t22 = -4.0 * a + POW243 * (a2 + 12.0 * c) / t1 + POW223 * t1;
    let t2 = t22.sqrt();

    let u = -t22 - 12.0 * a;
    let v = 12.0 * 2.4494897427831780981972840747059 * b / t2;
    let suv = (u + v).sqrt();
    let duv = (u - v).sqrt();
    let lambda1 = K * (t2 - duv);
    let lambda2 = K * (t2 + duv);
    let lambda3 = -K * (t2 + suv);
    let lambda4 = -K * (t2 - suv);
    let lambda = lambda1.re.max(lambda2.re.max(lambda3.re.max(lambda4.re)));

    flae_eigenvector(wm, lambda, dp(a, b, lambda))
}

pub fn flae_newton(p: &[Vec3], q: &[Vec3], w: &[f64]) -> Quat {
    let (wm, a, b, c) = flae_matrix(&weighted_outer_sum(q, p, w));

    // Start from an upper bound of the largest eigenvalue, so that Newton converges to
    // the largest root (starting at 1 it can stop at a smaller one).
    let mut lambda = eigenvalue_bound(p, q, w);
    let mut old_lambda = lambda + 1.0;
    while (old_lambda - lambda).abs() > 1e-5 {
        old_lambda = lambda;
        let l2 = lambda * lambda;
        let l3 = l2 * lambda;
        lambda -= (l3 * lambda + a * l2 + b * lambda + c) / (4.0 * l3 + 2.0 * a * lambda + b);
    }

    flae_eigenvector(wm, lambda, dp(a, b, lambda))
}
