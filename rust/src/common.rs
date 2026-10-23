//! Shared types and the few Eigen behaviours the estimators rely on.

use nalgebra::{Matrix3, Matrix4, Quaternion, Vector3, Vector4};

pub type Vec3 = Vector3<f64>;
pub type Vec4 = Vector4<f64>;
pub type Mat3 = Matrix3<f64>;
pub type Mat4 = Matrix4<f64>;
pub type Quat = Quaternion<f64>;

/// Eigen `Quaterniond(w, x, y, z)`.
#[inline]
pub fn quat(w: f64, x: f64, y: f64, z: f64) -> Quat {
    Quaternion::new(w, x, y, z)
}

/// Eigen `Quaterniond(const Vector4d&)`: the vector holds the coefficients
/// in storage order (x, y, z, w), not (w, x, y, z).
#[inline]
pub fn quat_from_xyzw(v: &Vec4) -> Quat {
    Quaternion::new(v[3], v[0], v[1], v[2])
}

/// Eigen `Quaterniond::normalize()`: leaves a zero quaternion untouched.
#[inline]
pub fn normalize_quat(q: Quat) -> Quat {
    let n2 = q.norm_squared();
    if n2 > 0.0 { q / n2.sqrt() } else { q }
}

/// Eigen `Vector3d::normalize()`: leaves a zero vector untouched.
#[inline]
pub fn normalize_vec(v: &mut Vec3) {
    let n2 = v.norm_squared();
    if n2 > 0.0 {
        *v /= n2.sqrt();
    }
}

/// Eigen `AngleAxisd(angle, axis)` converted to a quaternion (axis used as given).
#[inline]
pub fn angle_axis(angle: f64, axis: &Vec3) -> Quat {
    let (s, c) = (0.5 * angle).sin_cos();
    quat(c, s * axis.x, s * axis.y, s * axis.z)
}

/// Eigen `Quaterniond::_transformVector`: rotates `v`, assuming `q` is a unit quaternion.
#[inline]
pub fn transform_vector(q: &Quat, v: &Vec3) -> Vec3 {
    let qv = q.imag();
    let mut uv = qv.cross(v);
    uv += uv;
    v + q.w * uv + qv.cross(&uv)
}

/// `sum_i w_i a_i b_i^T`
#[inline]
pub fn weighted_outer_sum(a: &[Vec3], b: &[Vec3], w: &[f64]) -> Mat3 {
    let mut m = Mat3::zeros();
    for ((ai, bi), &wi) in a.iter().zip(b).zip(w) {
        m += (wi * ai) * bi.transpose();
    }
    m
}

/// `sum_i w_i (|p_i|^2 + |q_i|^2) / 2`, an upper bound of the largest eigenvalue
/// `lambda_max = max_R sum_i w_i q_i . (R p_i) <= sum_i w_i |p_i| |q_i|`. Starting Newton's
/// iteration on the characteristic polynomial there makes it converge monotonically to the
/// largest root. Without noise |p| = |q| and the bound is exact (the start of
/// `la_rotor_estimator`).
#[inline]
pub fn eigenvalue_bound(p: &[Vec3], q: &[Vec3], w: &[f64]) -> f64 {
    let s: f64 = p
        .iter()
        .zip(q)
        .zip(w)
        .map(|((pi, qi), &wi)| wi * (pi.norm_squared() + qi.norm_squared()))
        .sum();
    0.5 * s
}

/// Closed-form 4x4 determinant, the cofactor expansion Eigen uses for fixed 4x4 matrices.
#[inline]
pub fn det4(m: &Mat4) -> f64 {
    let s0 = m[(0, 0)] * m[(1, 1)] - m[(1, 0)] * m[(0, 1)];
    let s1 = m[(0, 0)] * m[(1, 2)] - m[(1, 0)] * m[(0, 2)];
    let s2 = m[(0, 0)] * m[(1, 3)] - m[(1, 0)] * m[(0, 3)];
    let s3 = m[(0, 1)] * m[(1, 2)] - m[(1, 1)] * m[(0, 2)];
    let s4 = m[(0, 1)] * m[(1, 3)] - m[(1, 1)] * m[(0, 3)];
    let s5 = m[(0, 2)] * m[(1, 3)] - m[(1, 2)] * m[(0, 3)];
    let c5 = m[(2, 2)] * m[(3, 3)] - m[(3, 2)] * m[(2, 3)];
    let c4 = m[(2, 1)] * m[(3, 3)] - m[(3, 1)] * m[(2, 3)];
    let c3 = m[(2, 1)] * m[(3, 2)] - m[(3, 1)] * m[(2, 2)];
    let c2 = m[(2, 0)] * m[(3, 3)] - m[(3, 0)] * m[(2, 3)];
    let c1 = m[(2, 0)] * m[(3, 2)] - m[(3, 0)] * m[(2, 2)];
    let c0 = m[(2, 0)] * m[(3, 1)] - m[(3, 0)] * m[(2, 1)];
    s0 * c5 - s1 * c4 + s2 * c3 + s3 * c2 - s4 * c1 + s5 * c0
}

/// Determinant of `g` without row `row` and column `col`.
pub fn minor3(g: &Mat4, row: usize, col: usize) -> f64 {
    let mut r = [0; 3];
    let mut c = [0; 3];
    for (k, i) in (0..4).filter(|&i| i != row).enumerate() {
        r[k] = i;
    }
    for (k, j) in (0..4).filter(|&j| j != col).enumerate() {
        c[k] = j;
    }
    g[(r[0], c[0])] * (g[(r[1], c[1])] * g[(r[2], c[2])] - g[(r[1], c[2])] * g[(r[2], c[1])])
        - g[(r[0], c[1])] * (g[(r[1], c[0])] * g[(r[2], c[2])] - g[(r[1], c[2])] * g[(r[2], c[0])])
        + g[(r[0], c[2])] * (g[(r[1], c[0])] * g[(r[2], c[1])] - g[(r[1], c[1])] * g[(r[2], c[0])])
}

/// Null vector v of a symmetric 4x4 matrix G of rank 3, such as M - lambda I at an
/// eigenvalue lambda of M. Since adj(G) = c v v^T, every column k of adj(G) is v scaled by
/// c v_k, and its diagonal entry c v_k^2 is the principal minor of G without row and
/// column k. The column with the largest diagonal entry is the best conditioned one.
/// Combining columns instead can cancel: their sum with alternating signs, which is what
/// the sum of the four triple wedges a_i ^ a_j ^ a_k computes, vanishes for every v with
/// v_1 - v_2 + v_3 - v_4 = 0.
pub fn adjugate_null_vector(g: &Mat4) -> Vec4 {
    // 2x2 minors of rows 0-1 (s) and rows 2-3 (c), shared by all 3x3 cofactors
    let s0 = g[(0, 0)] * g[(1, 1)] - g[(1, 0)] * g[(0, 1)];
    let s1 = g[(0, 0)] * g[(1, 2)] - g[(1, 0)] * g[(0, 2)];
    let s2 = g[(0, 0)] * g[(1, 3)] - g[(1, 0)] * g[(0, 3)];
    let s3 = g[(0, 1)] * g[(1, 2)] - g[(1, 1)] * g[(0, 2)];
    let s4 = g[(0, 1)] * g[(1, 3)] - g[(1, 1)] * g[(0, 3)];
    let s5 = g[(0, 2)] * g[(1, 3)] - g[(1, 2)] * g[(0, 3)];
    let c0 = g[(2, 0)] * g[(3, 1)] - g[(3, 0)] * g[(2, 1)];
    let c1 = g[(2, 0)] * g[(3, 2)] - g[(3, 0)] * g[(2, 2)];
    let c2 = g[(2, 0)] * g[(3, 3)] - g[(3, 0)] * g[(2, 3)];
    let c3 = g[(2, 1)] * g[(3, 2)] - g[(3, 1)] * g[(2, 2)];
    let c4 = g[(2, 1)] * g[(3, 3)] - g[(3, 1)] * g[(2, 3)];
    let c5 = g[(2, 2)] * g[(3, 3)] - g[(3, 2)] * g[(2, 3)];

    // diagonal of adj(G): the principal minors
    let d = [
        g[(1, 1)] * c5 - g[(1, 2)] * c4 + g[(1, 3)] * c3,
        g[(0, 0)] * c5 - g[(0, 2)] * c2 + g[(0, 3)] * c1,
        g[(3, 0)] * s4 - g[(3, 1)] * s2 + g[(3, 3)] * s0,
        g[(2, 0)] * s3 - g[(2, 1)] * s1 + g[(2, 2)] * s0,
    ];
    let mut best = 0;
    for k in 1..4 {
        if d[k].abs() > d[best].abs() {
            best = k;
        }
    }

    // row `best` of adj(G), which equals its column since G is symmetric
    match best {
        0 => Vec4::new(
            d[0],
            -g[(0, 1)] * c5 + g[(0, 2)] * c4 - g[(0, 3)] * c3,
            g[(3, 1)] * s5 - g[(3, 2)] * s4 + g[(3, 3)] * s3,
            -g[(2, 1)] * s5 + g[(2, 2)] * s4 - g[(2, 3)] * s3,
        ),
        1 => Vec4::new(
            -g[(1, 0)] * c5 + g[(1, 2)] * c2 - g[(1, 3)] * c1,
            d[1],
            -g[(3, 0)] * s5 + g[(3, 2)] * s2 - g[(3, 3)] * s1,
            g[(2, 0)] * s5 - g[(2, 2)] * s2 + g[(2, 3)] * s1,
        ),
        2 => Vec4::new(
            g[(1, 0)] * c4 - g[(1, 1)] * c2 + g[(1, 3)] * c0,
            -g[(0, 0)] * c4 + g[(0, 1)] * c2 - g[(0, 3)] * c0,
            d[2],
            -g[(2, 0)] * s4 + g[(2, 1)] * s2 - g[(2, 3)] * s0,
        ),
        _ => Vec4::new(
            -g[(1, 0)] * c3 + g[(1, 1)] * c1 - g[(1, 2)] * c0,
            g[(0, 0)] * c3 - g[(0, 1)] * c1 + g[(0, 2)] * c0,
            -g[(3, 0)] * s3 + g[(3, 1)] * s1 - g[(3, 2)] * s0,
            d[3],
        ),
    }
}

/// Eigenvector of the largest eigenvalue of a symmetric matrix
/// (Eigen's `SelfAdjointEigenSolver::eigenvectors().col(3)`).
pub fn eigenvector_max(m: Mat4) -> Vec4 {
    let e = m.symmetric_eigen();
    let i = e.eigenvalues.imax();
    e.eigenvectors.column(i).into_owned()
}

/// Eigenvector of the smallest eigenvalue of a symmetric matrix
/// (Eigen's `SelfAdjointEigenSolver::eigenvectors().col(0)`).
pub fn eigenvector_min(m: Mat4) -> Vec4 {
    let e = m.symmetric_eigen();
    let i = e.eigenvalues.imin();
    e.eigenvectors.column(i).into_owned()
}

/// Copies the upper triangle onto the lower one (Eigen `selfadjointView<Upper>().evalTo`).
#[inline]
pub fn symmetrize_from_upper(m: &mut Mat4) {
    for i in 0..4 {
        for j in 0..i {
            m[(i, j)] = m[(j, i)];
        }
    }
}

/// Copies the lower triangle onto the upper one (Eigen `selfadjointView<Lower>().evalTo`).
#[inline]
pub fn symmetrize_from_lower(m: &mut Mat4) {
    for i in 0..4 {
        for j in 0..i {
            m[(j, i)] = m[(i, j)];
        }
    }
}
