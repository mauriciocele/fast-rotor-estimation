//! FA3R: Fast Analytical 3D Registration, floating-point and fixed-point variants.

use crate::common::*;

const SHIFT: u32 = 20;
const D2L: f64 = (1u64 << SHIFT) as f64;
const SHIFTED1: i64 = 2i64 << SHIFT;
const SHIFTED2: u32 = SHIFT << 1;
const SHIFTED3: i64 = 2i64 << (3 * SHIFT + 1);
const L2D: f64 = 1.0 / D2L;

/// `z = k * (z + x × y)`
#[inline]
fn cross_f(x: &Vec3, y: &Vec3, k: f64, z: &mut Vec3) {
    z[0] = k * (z[0] + x[1] * y[2] - x[2] * y[1]);
    z[1] = k * (z[1] + x[2] * y[0] - x[0] * y[2]);
    z[2] = k * (z[2] + x[0] * y[1] - x[1] * y[0]);
}

/// Fixed-point `z = k * (z + x × y)`.
///
/// The input is normalized, so this does not overflow; wrapping arithmetic keeps
/// the C++ `int64_t` semantics in any case instead of panicking in debug builds.
#[inline]
fn cross_i(x: [i64; 3], y: [i64; 3], k: i64, z: &mut [i64; 3]) {
    let c = |a1: i64, b1: i64, a2: i64, b2: i64| a1.wrapping_mul(b1).wrapping_sub(a2.wrapping_mul(b2)) >> SHIFT;
    z[0] = k.wrapping_mul(z[0].wrapping_add(c(x[1], y[2], x[2], y[1]))) >> SHIFTED2;
    z[1] = k.wrapping_mul(z[1].wrapping_add(c(x[2], y[0], x[0], y[2]))) >> SHIFTED2;
    z[2] = k.wrapping_mul(z[2].wrapping_add(c(x[0], y[1], x[1], y[0]))) >> SHIFTED2;
}

pub fn fa3r_int(p: &[Vec3], q: &[Vec3], w: &[f64]) -> Mat3 {
    let mut sigma = weighted_outer_sum(q, p, w);
    // The iteration is tuned for |H| ~ 1 (k = 2 / (|H|^2 + 1)), and in fixed point a
    // large |H| overflows i64. The rotation does not depend on the scale of H.
    let sigma_norm = sigma.norm();
    if sigma_norm > 0.0 {
        sigma /= sigma_norm;
    }
    let fix = |v: f64| (v * D2L) as i64;
    // hx = row 0 of sigma, hy = row 1, hz = row 2 (as in the original)
    let mut hx = [fix(sigma[(0, 0)]), fix(sigma[(0, 1)]), fix(sigma[(0, 2)])];
    let mut hy = [fix(sigma[(1, 0)]), fix(sigma[(1, 1)]), fix(sigma[(1, 2)])];
    let mut hz = [fix(sigma[(2, 0)]), fix(sigma[(2, 1)]), fix(sigma[(2, 2)])];

    const STEPS: usize = 12;
    for _ in 0..STEPS {
        let (hx_, hy_, hz_) = (hx, hy, hz);
        let sq = hx_
            .iter()
            .chain(&hy_)
            .chain(&hz_)
            .fold(0i64, |acc, &v| acc.wrapping_add(v.wrapping_mul(v)));
        let k = SHIFTED3.wrapping_div((sq >> SHIFT).wrapping_add(SHIFTED1));

        cross_i(hx_, hy_, k, &mut hz);
        cross_i(hz_, hx_, k, &mut hy);
        cross_i(hy_, hz_, k, &mut hx);
    }

    let mut cx = Vec3::new(hx[0] as f64 * L2D, hy[0] as f64 * L2D, hz[0] as f64 * L2D);
    let mut cy = Vec3::new(hx[1] as f64 * L2D, hy[1] as f64 * L2D, hz[1] as f64 * L2D);
    let mut cz = Vec3::new(hx[2] as f64 * L2D, hy[2] as f64 * L2D, hz[2] as f64 * L2D);
    normalize_vec(&mut cx);
    normalize_vec(&mut cy);
    normalize_vec(&mut cz);
    Mat3::from_columns(&[cx, cy, cz])
}

pub fn fa3r_double(p: &[Vec3], q: &[Vec3], w: &[f64]) -> Mat3 {
    let mut sigma = weighted_outer_sum(q, p, w);
    // The iteration is tuned for |H| ~ 1 (k = 2 / (|H|^2 + 1)), and in fixed point a
    // large |H| overflows i64. The rotation does not depend on the scale of H.
    let sigma_norm = sigma.norm();
    if sigma_norm > 0.0 {
        sigma /= sigma_norm;
    }
    let mut hx: Vec3 = sigma.column(0).into_owned();
    let mut hy: Vec3 = sigma.column(1).into_owned();
    let mut hz: Vec3 = sigma.column(2).into_owned();

    const STEPS: usize = 12;
    for _ in 0..STEPS {
        let k = 2.0 / (hx.norm_squared() + hy.norm_squared() + hz.norm_squared() + 1.0);
        let (hx_, hy_, hz_) = (hx, hy, hz);
        cross_f(&hx_, &hy_, k, &mut hz);
        cross_f(&hz_, &hx_, k, &mut hy);
        cross_f(&hy_, &hz_, k, &mut hx);
    }
    normalize_vec(&mut hx);
    normalize_vec(&mut hy);
    normalize_vec(&mut hz);
    Mat3::from_columns(&[hx, hy, hz])
}
