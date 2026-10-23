//! Horn's closed-form quaternion method.

use crate::common::*;

pub fn horn(p: &[Vec3], q: &[Vec3], w: &[f64]) -> Quat {
    let m = weighted_outer_sum(p, q, w);
    let mut n = Mat4::zeros();

    // on-diagonal elements
    n[(0, 0)] = m[(0, 0)] + m[(1, 1)] + m[(2, 2)];
    n[(1, 1)] = m[(0, 0)] - m[(1, 1)] - m[(2, 2)];
    n[(2, 2)] = -m[(0, 0)] + m[(1, 1)] - m[(2, 2)];
    n[(3, 3)] = -m[(0, 0)] - m[(1, 1)] + m[(2, 2)];

    // off-diagonal elements
    n[(0, 1)] = m[(1, 2)] - m[(2, 1)];
    n[(0, 2)] = m[(2, 0)] - m[(0, 2)];
    n[(0, 3)] = m[(0, 1)] - m[(1, 0)];
    n[(1, 2)] = m[(0, 1)] + m[(1, 0)];
    n[(1, 3)] = m[(2, 0)] + m[(0, 2)];
    n[(2, 3)] = m[(1, 2)] + m[(2, 1)];
    symmetrize_from_upper(&mut n);

    let v = eigenvector_max(n);
    quat(v[0], v[1], v[2], v[3])
}
