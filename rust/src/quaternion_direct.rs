//! Direct quaternion least squares: smallest eigenvector of `sum w (R(q) - L(p))^T (R(q) - L(p))`.

use crate::common::*;

/// For product L*X
#[inline]
fn quaternion_left(q: &Vec3) -> Mat4 {
    let mut l = Mat4::zeros();
    l[(0, 1)] = -q.x;
    l[(1, 0)] = q.x;
    l[(0, 2)] = -q.y;
    l[(2, 0)] = q.y;
    l[(0, 3)] = -q.z;
    l[(3, 0)] = q.z;
    l[(1, 2)] = q.z;
    l[(2, 1)] = -q.z;
    l[(1, 3)] = -q.y;
    l[(3, 1)] = q.y;
    l[(2, 3)] = q.x;
    l[(3, 2)] = -q.x;
    l
}

/// For product X*R
#[inline]
fn quaternion_right(q: &Vec3) -> Mat4 {
    let mut r = Mat4::zeros();
    r[(0, 1)] = -q.x;
    r[(1, 0)] = q.x;
    r[(0, 2)] = -q.y;
    r[(2, 0)] = q.y;
    r[(0, 3)] = -q.z;
    r[(3, 0)] = q.z;
    r[(1, 2)] = -q.z;
    r[(2, 1)] = q.z;
    r[(1, 3)] = q.y;
    r[(3, 1)] = -q.y;
    r[(2, 3)] = -q.x;
    r[(3, 2)] = q.x;
    r
}

pub fn quaternion_direct(p: &[Vec3], q: &[Vec3], w: &[f64]) -> Quat {
    let mut m = Mat4::zeros();
    for ((pj, qj), &wj) in p.iter().zip(q).zip(w) {
        let a = quaternion_right(qj) - quaternion_left(pj);
        m += wj * a.transpose() * a;
    }
    let v = eigenvector_min(m);
    quat(v[0], v[1], v[2], v[3])
}
