//! Arun's SVD method, with nalgebra's SVD and with the McAdams 3x3 SVD.

use crate::common::*;
use crate::svd3;

pub fn svd_eigen(p: &[Vec3], q: &[Vec3], w: &[f64]) -> Mat3 {
    let m = weighted_outer_sum(p, q, w);
    let svd = m.svd(true, true);
    let u = svd.u.expect("U requested");
    let mut v = svd.v_t.expect("V requested").transpose();
    // V U^T is a reflection when det(V U^T) = -1 (e.g. planar or very noisy data);
    // flipping the singular vector of the smallest singular value gives the
    // closest rotation instead (Arun et al. 1987, Umeyama 1991).
    if (v * u.transpose()).determinant() < 0.0 {
        let k = svd.singular_values.imin();
        v.column_mut(k).neg_mut();
    }
    v * u.transpose()
}

pub fn svd_mcadams(p: &[Vec3], q: &[Vec3], w: &[f64]) -> Mat3 {
    let a = weighted_outer_sum(q, p, w);
    let a = [
        a[(0, 0)],
        a[(0, 1)],
        a[(0, 2)],
        a[(1, 0)],
        a[(1, 1)],
        a[(1, 2)],
        a[(2, 0)],
        a[(2, 1)],
        a[(2, 2)],
    ];

    let (u, _s, v) = svd3::svd(&a);
    // A = sum w Q P^T = U S V^T, so the rotation taking P to Q is R = U V^T.
    // U and V are proper rotations (the sign of det(A) is kept in s33), so R is too.
    let r = svd3::mult_ab(&u, &svd3::transpose(&v));
    Mat3::from_row_slice(&r)
}
