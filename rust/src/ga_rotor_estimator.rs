//! Rotor estimation in geometric algebra.
//!
//! `ga_rotor_estimator` is the reference version built on the generic
//! [`Multivector`] class; `la_rotor_estimator` is the same method with the
//! GA expressions expanded (GAALOP) into plain arithmetic.

use crate::common::*;
use crate::multivector::{Multivector, scp};

/// Blades of the 4D algebra with basis e1..e4, for the eigenvector step of
/// `la_rotor_estimator`: coefficients on e12, e13, e14, e23, e24, e34.
struct Bivector4 {
    e12: f64,
    e13: f64,
    e14: f64,
    e23: f64,
    e24: f64,
    e34: f64,
}

/// Coefficients on e123, e124, e134, e234.
struct Trivector4 {
    e123: f64,
    e124: f64,
    e134: f64,
    e234: f64,
}

/// `a ^ b`
#[inline]
fn wedge2(a: &Vec4, b: &Vec4) -> Bivector4 {
    Bivector4 {
        e12: a[0] * b[1] - a[1] * b[0],
        e13: a[0] * b[2] - a[2] * b[0],
        e14: a[0] * b[3] - a[3] * b[0],
        e23: a[1] * b[2] - a[2] * b[1],
        e24: a[1] * b[3] - a[3] * b[1],
        e34: a[2] * b[3] - a[3] * b[2],
    }
}

/// `B ^ c` (= `c ^ B`)
#[inline]
fn wedge3(b: &Bivector4, c: &Vec4) -> Trivector4 {
    Trivector4 {
        e123: b.e12 * c[2] - b.e13 * c[1] + b.e23 * c[0],
        e124: b.e12 * c[3] - b.e14 * c[1] + b.e24 * c[0],
        e134: b.e13 * c[3] - b.e14 * c[2] + b.e34 * c[0],
        e234: b.e23 * c[3] - b.e24 * c[2] + b.e34 * c[1],
    }
}

/// `T . (e1^e2^e3^e4)^-1`, the vector dual to T
#[inline]
fn dual3(t: &Trivector4) -> Vec4 {
    Vec4::new(t.e234, -t.e134, t.e124, -t.e123)
}

/// `(A ^ C) . (e1^e2^e3^e4)^-1`, the scalar of the 4-vector `A ^ C`
#[inline]
fn wedge4(a: &Bivector4, c: &Bivector4) -> f64 {
    a.e12 * c.e34 - a.e13 * c.e24 + a.e14 * c.e23 + a.e23 * c.e14 - a.e24 * c.e13 + a.e34 * c.e12
}

fn multivector_derivative(p: &[Vec3], q: &[Vec3], w: &[f64], r: &Multivector) -> Vec4 {
    let e1 = Multivector::basis_vector(0);
    let e2 = Multivector::basis_vector(1);
    let e3 = Multivector::basis_vector(2);
    let e12 = &e1 ^ &e2;
    let e31 = &e3 ^ &e1;
    let e23 = &e2 ^ &e3;
    let mut derivative_accum = Multivector::default();
    for (j, ((pj, qj), &wj)) in p.iter().zip(q).zip(w).enumerate() {
        let q_bivector = qj.z * &e12 + qj.y * &e31 + qj.x * &e23;
        let p_bivector = pj.z * &e12 + pj.y * &e31 + pj.x * &e23;
        //q q ~R + 2 q ~R p + ~R p p
        let derivative = -1.0 * (&q_bivector * &q_bivector * r)
            + -2.0 * (&q_bivector * r * &p_bivector)
            + -1.0 * (r * &p_bivector * &p_bivector);
        derivative_accum = if j == 0 {
            wj * derivative
        } else {
            derivative_accum + wj * derivative
        };
    }
    Vec4::new(
        scp(&e23, &derivative_accum),
        scp(&e31, &derivative_accum),
        scp(&e12, &derivative_accum),
        derivative_accum.scalar_part(),
    )
}

/*
max | p + R* q R |^2

p + R* q R = 0

p R* + R* q = 0

p (w - L) + (w - L) q = 0

(Remember: inner product of bivectors is negative and commutator anticommute)
w (p + q) + L . (p + q) + (q - p) x L = 0

| 0     s^T  | | w | = | s^T L       |
| s    [d]_x | | L |   | w s + d x L |

| 0     s^T  | | 0    s^T  | = | s^T s    s^T [d]_x       |
| s   -[d]_x | | s   [d]_x |   |-[d]_x s  s s^T - [d]^2_x |

| 0     s^T  | | 0    s^T  | = | s^T s    s x d         | = | s^T s    s x d                 |
| s   -[d]_x | | s   [d]_x |   | s x d  s s^T - [d]^2_x |   | s x d  s s^T - d d^t + d^T d I |
*/
pub fn ga_rotor_estimator(p: &[Vec3], q: &[Vec3], w: &[f64]) -> Quat {
    let e0 = Multivector::basis_vector(0);
    let e1 = Multivector::basis_vector(1);
    let e2 = Multivector::basis_vector(2);
    let e3 = Multivector::basis_vector(3);
    let e0123 = &e0 ^ &e1 ^ &e2 ^ &e3;
    let e12 = &e0 ^ &e1;
    let e31 = &e2 ^ &e0;
    let e23 = &e1 ^ &e2;
    let one = Multivector::scalar(1.0);

    let h_0 = multivector_derivative(p, q, w, &one);
    let h_1 = multivector_derivative(p, q, w, &(-1.0 * &e12));
    let h_2 = multivector_derivative(p, q, w, &(-1.0 * &e31));
    let h_3 = multivector_derivative(p, q, w, &(-1.0 * &e23));

    let column = |h: &Vec4| h[0] * &e0 + h[1] * &e1 + h[2] * &e2 + h[3] * &e3;
    let a = column(&h_3);
    let b = column(&h_2);
    let c = column(&h_1);
    let d = column(&h_0);

    let mut lambda = h_3[0] + h_2[1] + h_1[2] + h_0[3]; // trace of matrix
    loop {
        let lambda_prev = lambda;
        let xa = &a - lambda * &e0;
        let xb = &b - lambda * &e1;
        let xc = &c - lambda * &e2;
        let xd = &d - lambda * &e3;
        let det = scp(&e0123, &(&xa ^ &xb ^ &xc ^ &xd));
        let ddet = scp(
            &e0123,
            &(((-1.0 * &e0) ^ &xb ^ &xc ^ &xd)
                + (&xa ^ (-1.0 * &e1) ^ &xc ^ &xd)
                + (&xa ^ &xb ^ (-1.0 * &e2) ^ &xd)
                + (&xa ^ &xb ^ &xc ^ (-1.0 * &e3))),
        );
        lambda -= det / ddet;
        if (lambda_prev - lambda).abs() <= 1e-5 {
            break;
        }
    }

    // The dual of each triple wedge is one column of adj(M - lambda I), parallel to the
    // eigenrotor and scaled by its component along the missing column. Summing them
    // cancels for a whole family of rotations (see `adjugate_null_vector`), so keep the one
    // with the largest component along its own missing column.
    let x = [
        &a - lambda * &e0,
        &b - lambda * &e1,
        &c - lambda * &e2,
        &d - lambda * &e3,
    ];
    let basis = [&e0, &e1, &e2, &e3];
    let r3 = (0..4)
        .map(|k| {
            let candidate = (&x[(k + 1) % 4] ^ &x[(k + 2) % 4] ^ &x[(k + 3) % 4]).dual(4);
            (scp(basis[k], &candidate).abs(), candidate)
        })
        .fold(
            (-1.0, Multivector::default()),
            |best, c| if c.0 > best.0 { c } else { best },
        )
        .1;
    let rr = Vec4::new(scp(&e0, &r3), scp(&e1, &r3), scp(&e2, &r3), scp(&e3, &r3));
    normalize_quat(quat_from_xyzw(&rr))
}

pub fn la_rotor_estimator(p: &[Vec3], q: &[Vec3], w: &[f64]) -> Quat {
    // S = sum w (|p|^2 + |q|^2) and Sx = sum w p q^T
    let mut sx = Mat3::zeros();
    let mut s = 0.0;
    for ((pj, qj), &wj) in p.iter().zip(q).zip(w) {
        s += wj * (pj.norm_squared() + qj.norm_squared());
        sx += (wj * pj) * qj.transpose();
    }

    // M = K + (S / 2) I, where K is Davenport's matrix of Sx, with columns k1..k4 (basis
    // e1..e4). The shift does not change the eigenvectors, so we work with K and
    // mu = lambda - S / 2.
    let tr = sx.trace();
    let k1 = Vec4::new(
        2.0 * sx[(0, 0)] - tr,
        sx[(0, 1)] + sx[(1, 0)],
        sx[(2, 0)] + sx[(0, 2)],
        sx[(1, 2)] - sx[(2, 1)],
    );
    let k2 = Vec4::new(
        sx[(0, 1)] + sx[(1, 0)],
        2.0 * sx[(1, 1)] - tr,
        sx[(1, 2)] + sx[(2, 1)],
        sx[(2, 0)] - sx[(0, 2)],
    );
    let k3 = Vec4::new(
        sx[(2, 0)] + sx[(0, 2)],
        sx[(1, 2)] + sx[(2, 1)],
        2.0 * sx[(2, 2)] - tr,
        sx[(0, 1)] - sx[(1, 0)],
    );
    let k4 = Vec4::new(
        sx[(1, 2)] - sx[(2, 1)],
        sx[(2, 0)] - sx[(0, 2)],
        sx[(0, 1)] - sx[(1, 0)],
        tr,
    );

    // Multivector derivative is:
    //   d_R |q + ~R p R|^2 = d_R (R q + p R)~(R q + p R)
    //   d_R |q + ~R p R|^2 = -2 q q ~R - 4 q ~R p - 2 ~R p p
    //
    //   Which implies the following eigen-rotor problem
    //
    //   2 q ~R p  = lambda ~R
    //
    //   where lambda = -(q q + p p)
    //
    //   NOTE: lambda is positive value since square of bivectors p and q is negative
    //
    //   Lambda is the exact eigenvalue when input bivectors Ps and Qs don't have noise
    //   In general, in presence of noise, Lambda is the best initial guess to the newton iteration.
    //
    // The characteristic outer product (x1 ^ x2 ^ x3 ^ x4) . (e1^e2^e3^e4)^-1 = det(K - mu I)
    // is the quartic mu^4 + a mu^2 + b mu + c, since K is traceless: a = -2 |Sx|^2,
    // b = -8 det(Sx) and c = det K = (k1 ^ k2) ^ (k3 ^ k4). With the coefficients computed
    // once, each Newton step is a few operations. lambda_0 = S, i.e. mu_0 = S / 2.
    let a = -2.0 * sx.norm_squared();
    let b = -8.0 * sx.determinant();
    let c = wedge4(&wedge2(&k1, &k2), &wedge2(&k3, &k4));
    let mut mu = 0.5 * s;
    loop {
        let mu_prev = mu;
        let mu2 = mu * mu;
        mu -= ((mu2 + a) * mu2 + b * mu + c) / ((4.0 * mu2 + 2.0 * a) * mu + b);
        if (mu_prev - mu).abs() <= 1e-5 {
            break;
        }
    }

    // Eigenrotor, as in `ga_rotor_estimator`: the dual of each triple wedge of the columns
    // x1..x4 of K - mu I (= M - lambda I) is parallel to the eigenrotor, scaled by its component along
    // the missing column. Keep the one where that component is largest (summing them
    // cancels for a family of rotations).
    //
    //   A = x1 ^ x2;  C = x3 ^ x4;
    //   T1 = x2 ^ C;  T2 = x1 ^ C;  T3 = A ^ x4;  T4 = A ^ x3;   (Tk misses xk)
    //   R = dual(Tk) for the k with the largest |dual(Tk) . ek|
    let x1 = k1 - mu * Vec4::x();
    let x2 = k2 - mu * Vec4::y();
    let x3 = k3 - mu * Vec4::z();
    let x4 = k4 - mu * Vec4::w();
    let a = wedge2(&x1, &x2);
    let c = wedge2(&x3, &x4);

    // dual(Tk) . ek is the coefficient of Tk on the blade without ek
    let own = [
        c.e23 * x2[3] - c.e24 * x2[2] + c.e34 * x2[1], // (x2 ^ C) . e234
        c.e13 * x1[3] - c.e14 * x1[2] + c.e34 * x1[0], // (x1 ^ C) . e134
        a.e12 * x4[3] - a.e14 * x4[1] + a.e24 * x4[0], // (A ^ x4) . e124
        a.e12 * x3[2] - a.e13 * x3[1] + a.e23 * x3[0], // (A ^ x3) . e123
    ];
    // Branch-free selection: the largest meet is T = B ^ v with (B, v) = (C, x2), (C, x1),
    // (A, x4) or (A, x3); select B and v, then compute a single trivector. The winner depends
    // on the rotation, so a branch here would often be mispredicted.
    let pick = |m: bool, a: f64, b: f64| if m { a } else { b };
    let pick4 = |m: bool, a: &Vec4, b: &Vec4| Vec4::from_fn(|i, _| pick(m, a[i], b[i]));
    let pick_b = |m: bool, a: &Bivector4, b: &Bivector4| Bivector4 {
        e12: pick(m, a.e12, b.e12),
        e13: pick(m, a.e13, b.e13),
        e14: pick(m, a.e14, b.e14),
        e23: pick(m, a.e23, b.e23),
        e24: pick(m, a.e24, b.e24),
        e34: pick(m, a.e34, b.e34),
    };
    let m1 = own[1].abs() > own[0].abs();
    let mut best = pick(m1, own[1].abs(), own[0].abs());
    let mut bs = Bivector4 { ..c }; // T_0 and T_1 both use C
    let mut v = pick4(m1, &x1, &x2);
    let m2 = own[2].abs() > best;
    best = pick(m2, own[2].abs(), best);
    bs = pick_b(m2, &a, &bs);
    v = pick4(m2, &x4, &v);
    let m3 = own[3].abs() > best;
    bs = pick_b(m3, &a, &bs);
    v = pick4(m3, &x3, &v);
    let t = wedge3(&bs, &v);

    // basis order e1, e2, e3, e4, read as x, y, z, w
    normalize_quat(quat_from_xyzw(&dual3(&t)))
}
