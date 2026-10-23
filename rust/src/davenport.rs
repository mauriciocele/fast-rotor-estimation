//! Davenport's q-method.

use crate::common::*;

pub fn davenport(p: &[Vec3], q: &[Vec3], w: &[f64]) -> Quat {
    let b = weighted_outer_sum(p, q, w);
    let trace = b.trace();
    let mut k = Mat4::zeros();
    k[(0, 0)] = trace;
    k[(1, 0)] = b[(1, 2)] - b[(2, 1)];
    k[(2, 0)] = b[(2, 0)] - b[(0, 2)];
    k[(3, 0)] = b[(0, 1)] - b[(1, 0)];
    k[(1, 1)] = 2.0 * b[(0, 0)] - trace;
    k[(2, 1)] = b[(0, 1)] + b[(1, 0)];
    k[(3, 1)] = b[(2, 0)] + b[(0, 2)];
    k[(2, 2)] = 2.0 * b[(1, 1)] - trace;
    k[(3, 2)] = b[(1, 2)] + b[(2, 1)];
    k[(3, 3)] = 2.0 * b[(2, 2)] - trace;
    symmetrize_from_lower(&mut k);

    let v = eigenvector_max(k);
    quat(v[0], v[1], v[2], v[3])
}
