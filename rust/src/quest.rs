//! QUEST and FOAM.
//!
//! Davenport Matrix
//! ```text
//!     K = | p^T q            p x q            |
//!         | p x q    p q^T + q p^T - (p^T q)I |
//!
//!     K q = lambda q
//!     K q - lambda q = 0
//!     (K - lambda I) q = 0
//!
//!     det(K - lambda I) q = 0
//!     lambda^4 + C1 lambda^3 + C2 lambda^2 + C3 lambda + C4 = 0
//! ```
//!
//! How to Estimate Attitude From Vector Observations
//! <http://www.malcolmdshuster.com/FC_MarkleyMortari_Girdwood_1999_AAS.pdf>
//! An Improvement to the QUEST Algorithm
//! <https://pdfs.semanticscholar.org/657f/afa702198881e4f9584e84b8f26c7bf83185.pdf>

use crate::common::*;

/// Adjugate is the transpose of its cofactor matrix.
#[inline]
fn adjugate(m: &Mat3) -> Mat3 {
    adjugate_transposed(m).transpose()
}

/// Adjugate transposed is the cofactor matrix.
#[inline]
fn adjugate_transposed(m: &Mat3) -> Mat3 {
    let mut c = Mat3::zeros();
    c[(0, 0)] = m[(1, 1)] * m[(2, 2)] - m[(1, 2)] * m[(2, 1)];
    c[(0, 1)] = -m[(1, 0)] * m[(2, 2)] + m[(1, 2)] * m[(2, 0)];
    c[(0, 2)] = m[(1, 0)] * m[(2, 1)] - m[(1, 1)] * m[(2, 0)];
    c[(1, 0)] = -m[(0, 1)] * m[(2, 2)] + m[(0, 2)] * m[(2, 1)];
    c[(1, 1)] = m[(0, 0)] * m[(2, 2)] - m[(0, 2)] * m[(2, 0)];
    c[(1, 2)] = -m[(0, 0)] * m[(2, 1)] + m[(0, 1)] * m[(2, 0)];
    c[(2, 0)] = m[(0, 1)] * m[(1, 2)] - m[(0, 2)] * m[(1, 1)];
    c[(2, 1)] = -m[(0, 0)] * m[(1, 2)] + m[(0, 2)] * m[(1, 0)];
    c[(2, 2)] = m[(0, 0)] * m[(1, 1)] - m[(0, 1)] * m[(1, 0)];
    c
}

/// Unnormalized QUEST quaternion (gamma, x) of B for the eigenvalue lambda:
/// gamma = det((lambda + s) I - S) and x = adj((lambda + s) I - S) Z.
fn quest_vector(b: &Mat3, lambda: f64) -> (f64, Vec3) {
    let s = b.trace();
    let sm = b + b.transpose();
    let z = Vec3::new(b[(1, 2)] - b[(2, 1)], b[(2, 0)] - b[(0, 2)], b[(0, 1)] - b[(1, 0)]);
    let kappa = sm[(1, 1)] * sm[(2, 2)] - sm[(1, 2)] * sm[(2, 1)] + sm[(0, 0)] * sm[(2, 2)] - sm[(0, 2)] * sm[(2, 0)]
        + sm[(0, 0)] * sm[(1, 1)]
        - sm[(0, 1)] * sm[(1, 0)];
    let alpha = lambda * lambda - (s * s - kappa);
    let beta = lambda - s;
    let gamma = alpha * (lambda + s) - sm.determinant();
    let sz = sm * z;
    (gamma, beta * sz + sm * sz + alpha * z)
}

pub fn quest(p: &[Vec3], q: &[Vec3], w: &[f64]) -> Quat {
    let b = weighted_outer_sum(p, q, w);
    let s = b.trace();
    let sm = b + b.transpose();
    let z = Vec3::new(b[(1, 2)] - b[(2, 1)], b[(2, 0)] - b[(0, 2)], b[(0, 1)] - b[(1, 0)]);
    // Adjugate(S).trace()
    let kappa = sm[(1, 1)] * sm[(2, 2)] - sm[(1, 2)] * sm[(2, 1)] + sm[(0, 0)] * sm[(2, 2)] - sm[(0, 2)] * sm[(2, 0)]
        + sm[(0, 0)] * sm[(1, 1)]
        - sm[(0, 1)] * sm[(1, 0)];
    let s2 = s * s;
    let sdet = sm.determinant();
    let sz = sm * z;
    let ssz = sm * sz;
    let a = s2 - kappa;
    let bb = s2 + z.dot(&z);
    let c = sdet + z.dot(&sz);
    let d = z.dot(&ssz);
    // Newton's method: lambda_{n+1} = lambda_n - f(lambda) / f'(lambda)
    // f(lambda) = lambda^4 - (a + b) lambda^2 - c lambda + (a b + c s - d)
    // f'(lambda) = 4 lambda^3 - 2 (a + b) lambda + c
    // Start from an upper bound of the largest eigenvalue, so that Newton converges to
    // the largest root (starting at 1 it can stop at a smaller one).
    let mut lambda = eigenvalue_bound(p, q, w);
    let mut old_lambda = lambda + 1.0;
    while (old_lambda - lambda).abs() > 1e-5 {
        old_lambda = lambda;
        let l2 = lambda * lambda;
        let l3 = l2 * lambda;
        lambda -= (l3 * lambda - (a + bb) * l2 - c * lambda + (a * bb + c * s - d))
            / (4.0 * l3 - 2.0 * (a + bb) * lambda - c);
    }
    let alpha = lambda * lambda - a;
    let beta = lambda - s;
    let gamma = alpha * (lambda + s) - sdet;
    let x = beta * sz + ssz + alpha * z;
    let mut result = quat(gamma, x.x, x.y, x.z);

    // Method of sequential rotations (Shuster & Oh 1981). (gamma, x) is column 0 of
    // adj(lambda I - K) = f'(lambda) q q^T, so q_w^2 = gamma / f'(lambda). Near 180 degrees
    // q_w -> 0 and (gamma, x) vanishes. Then solve for P rotated by 180 degrees about the
    // axis k with the largest |q_k| (B_k = A_k B), and undo that rotation: q = q_k (0, e_k).
    let df = 4.0 * lambda * lambda * lambda - 2.0 * (a + bb) * lambda - c;
    if gamma.abs() < 0.1 * df.abs() {
        let mut best: Option<(usize, f64, Vec3)> = None;
        let mut best_gamma = gamma;
        for k in 0..3 {
            let mut bk = b;
            for r in (0..3).filter(|&r| r != k) {
                bk.row_mut(r).neg_mut();
            }
            let (gk, xk) = quest_vector(&bk, lambda);
            if gk.abs() > best_gamma.abs() {
                best_gamma = gk;
                best = Some((k, gk, xk));
            }
        }
        if let Some((k, gk, xk)) = best {
            let mut axis = quat(0.0, 0.0, 0.0, 0.0);
            axis.coords[k] = 1.0; // coords are (x, y, z, w)
            result = quat(gk, xk.x, xk.y, xk.z) * axis;
        }
    }
    normalize_quat(result)
}

pub fn foam(p: &[Vec3], q: &[Vec3], w: &[f64]) -> Mat3 {
    let b = weighted_outer_sum(q, p, w);
    let bnorm = b.norm_squared();
    let bdet = 8.0 * b.determinant();
    let badj = 4.0 * adjugate(&b).norm_squared();
    // Newton's method: lambda_{n+1} = lambda_n - f(lambda) / f'(lambda)
    // f(lambda) = (lambda^2 - |B|^2_f)^2 - 8 lambda det(B) - 4 |Adjugate(B)|^2_F
    // f'(lambda) = 4 lambda (lambda^2 - |B|^2_f) - 8 det(B)
    // Start from an upper bound of the largest eigenvalue (see `quest`).
    let mut lambda = eigenvalue_bound(p, q, w);
    let mut old_lambda = lambda + 1.0;
    while (old_lambda - lambda).abs() > 1e-5 {
        old_lambda = lambda;
        let kappa = lambda * lambda - bnorm;
        lambda -= (kappa * kappa - lambda * bdet - badj) / (4.0 * lambda * kappa - bdet);
    }
    let kappa = 0.5 * (lambda * lambda - bnorm);
    ((kappa + bnorm) * b + lambda * adjugate_transposed(&b) - b * b.transpose() * b) / (lambda * kappa - bdet / 8.0)
}
