//! Quick 3x3 SVD as described by A. McAdams, A. Selle, R. Tamstorf, J. Teran and
//! E. Sifakis, "Computing the Singular Value Decomposition of 3x3 matrices with
//! minimal branching and elementary floating point operations", University of
//! Wisconsin - Madison technical report TR1690, May 2011.
//!
//! Port of Eric Jang's optimized CPU implementation (svd3.h). Matrices are
//! row-major `[f64; 9]`: index 0 = m11, 1 = m12, ..., 8 = m33.

const GAMMA: f64 = 5.8284271247461900976033774484194; // FOUR_GAMMA_SQUARED = sqrt(8)+3;
const CSTAR: f64 = 0.92387953251128675612818318939679; // cos(pi/8)
const SSTAR: f64 = 0.3826834323650897717284599840304; // sin(pi/8)
const EPSILON: f64 = 1e-6;

pub type M3 = [f64; 9];

#[inline]
fn rsqrt(x: f64) -> f64 {
    1.0 / x.sqrt()
}

#[inline]
fn accurate_sqrt(x: f64) -> f64 {
    x * rsqrt(x)
}

#[inline]
fn cond_swap(c: bool, x: &mut f64, y: &mut f64) {
    let z = *x;
    if c {
        *x = *y;
        *y = z;
    }
}

#[inline]
fn cond_neg_swap(c: bool, m: &mut M3, i: usize, j: usize) {
    let z = -m[i];
    if c {
        m[i] = m[j];
        m[j] = z;
    }
}

/// M = A * B
#[inline]
pub fn mult_ab(a: &M3, b: &M3) -> M3 {
    let mut m = [0.0; 9];
    for r in 0..3 {
        for c in 0..3 {
            m[3 * r + c] = a[3 * r] * b[c] + a[3 * r + 1] * b[3 + c] + a[3 * r + 2] * b[6 + c];
        }
    }
    m
}

/// M = Transpose[A] * B
#[inline]
fn mult_atb(a: &M3, b: &M3) -> M3 {
    let mut m = [0.0; 9];
    for r in 0..3 {
        for c in 0..3 {
            m[3 * r + c] = a[r] * b[c] + a[3 + r] * b[3 + c] + a[6 + r] * b[6 + c];
        }
    }
    m
}

#[inline]
pub fn transpose(a: &M3) -> M3 {
    [a[0], a[3], a[6], a[1], a[4], a[7], a[2], a[5], a[8]]
}

/// Quaternion stored as (x, y, z, w), following GLM.
fn quat_to_mat3(qv: &[f64; 4]) -> M3 {
    let (x, y, z, w) = (qv[0], qv[1], qv[2], qv[3]);
    let (qxx, qyy, qzz) = (x * x, y * y, z * z);
    let (qxz, qxy, qyz) = (x * z, x * y, y * z);
    let (qwx, qwy, qwz) = (w * x, w * y, w * z);
    [
        1.0 - 2.0 * (qyy + qzz),
        2.0 * (qxy - qwz),
        2.0 * (qxz + qwy),
        2.0 * (qxy + qwz),
        1.0 - 2.0 * (qxx + qzz),
        2.0 * (qyz - qwx),
        2.0 * (qxz - qwy),
        2.0 * (qyz + qwx),
        1.0 - 2.0 * (qxx + qyy),
    ]
}

/// Given givens angle computed by approximateGivensAngles,
/// compute the corresponding rotation quaternion.
#[inline]
fn approximate_givens_quaternion(a11: f64, a12: f64, a22: f64) -> (f64, f64) {
    let ch = 2.0 * (a11 - a22);
    let sh = a12;
    let b = GAMMA * sh * sh < ch * ch;
    let w = rsqrt(ch * ch + sh * sh);
    if b { (w * ch, w * sh) } else { (CSTAR, SSTAR) }
}

/// Lower triangle of a symmetric 3x3 matrix.
struct Sym {
    s11: f64,
    s21: f64,
    s22: f64,
    s31: f64,
    s32: f64,
    s33: f64,
}

fn jacobi_conjugation(x: usize, y: usize, z: usize, s: &mut Sym, qv: &mut [f64; 4]) {
    let (ch, mut sh) = approximate_givens_quaternion(s.s11, s.s21, s.s22);

    let scale = ch * ch + sh * sh;
    let a = (ch * ch - sh * sh) / scale;
    let b = (2.0 * sh * ch) / scale;

    // make temp copy of S
    let (s11, s21, s22, s31, s32, s33) = (s.s11, s.s21, s.s22, s.s31, s.s32, s.s33);

    // perform conjugation S = Q'*S*Q
    // Q already implicitly solved from a, b
    let n11 = a * (a * s11 + b * s21) + b * (a * s21 + b * s22);
    let n21 = a * (-b * s11 + a * s21) + b * (-b * s21 + a * s22);
    let n22 = -b * (-b * s11 + a * s21) + a * (-b * s21 + a * s22);
    let n31 = a * s31 + b * s32;
    let n32 = -b * s31 + a * s32;
    let n33 = s33;

    // update cumulative rotation qV
    let tmp = [qv[0] * sh, qv[1] * sh, qv[2] * sh];
    sh *= qv[3];

    qv[0] *= ch;
    qv[1] *= ch;
    qv[2] *= ch;
    qv[3] *= ch;

    // (x,y,z) corresponds to ((0,1,2),(1,2,0),(2,0,1))
    // for (p,q) = ((0,1),(1,2),(0,2))
    qv[z] += sh;
    qv[3] -= tmp[z]; // w
    qv[x] += tmp[y];
    qv[y] -= tmp[x];

    // re-arrange matrix for next iteration
    *s = Sym {
        s11: n22,
        s21: n32,
        s22: n33,
        s31: n21,
        s32: n31,
        s33: n11,
    };
}

/// Finds the transformation that diagonalizes a symmetric matrix;
/// returns the quaternion representation of V.
fn jacobi_eigenanalysis(s: &mut Sym) -> [f64; 4] {
    let mut qv = [0.0, 0.0, 0.0, 1.0]; // follow same indexing convention as GLM
    // The paper's 4 sweeps are tuned for single precision; doubles need 6 to
    // reach full precision on ill-conditioned matrices.
    for _ in 0..6 {
        // we wish to eliminate the maximum off-diagonal element
        // on every iteration, but cycling over all 3 possible rotations
        // in fixed order (p,q) = (1,2) , (2,3), (1,3) still retains
        //  asymptotic convergence
        jacobi_conjugation(0, 1, 2, s, &mut qv); // p,q = 0,1
        jacobi_conjugation(1, 2, 0, s, &mut qv); // p,q = 1,2
        jacobi_conjugation(2, 0, 1, s, &mut qv); // p,q = 0,2
    }
    qv
}

#[inline]
fn dist2(x: f64, y: f64, z: f64) -> f64 {
    x * x + y * y + z * z
}

fn sort_singular_values(b: &mut M3, v: &mut M3) {
    let mut rho1 = dist2(b[0], b[3], b[6]);
    let mut rho2 = dist2(b[1], b[4], b[7]);
    let mut rho3 = dist2(b[2], b[5], b[8]);

    let mut swap_cols = |c: bool, i: usize, j: usize| {
        for r in 0..3 {
            cond_neg_swap(c, b, 3 * r + i, 3 * r + j);
            cond_neg_swap(c, v, 3 * r + i, 3 * r + j);
        }
    };

    let c = rho1 < rho2;
    swap_cols(c, 0, 1);
    cond_swap(c, &mut rho1, &mut rho2);
    let c = rho1 < rho3;
    swap_cols(c, 0, 2);
    cond_swap(c, &mut rho1, &mut rho3);
    let c = rho2 < rho3;
    swap_cols(c, 1, 2);
}

/// a1 = pivot point on diagonal, a2 = lower triangular entry we want to annihilate
#[inline]
fn qr_givens_quaternion(a1: f64, a2: f64) -> (f64, f64) {
    let rho = accurate_sqrt(a1 * a1 + a2 * a2);

    let mut sh = if rho > EPSILON { a2 } else { 0.0 };
    let mut ch = a1.abs() + rho.max(EPSILON);
    let b = a1 < 0.0;
    cond_swap(b, &mut sh, &mut ch);
    let w = rsqrt(ch * ch + sh * sh);
    ch *= w;
    sh *= w;
    (ch, sh)
}

/// Returns (Q, R).
fn qr_decomposition(bm: &M3) -> (M3, M3) {
    let [b11, b12, b13, b21, b22, b23, b31, b32, b33] = *bm;

    // first givens rotation (ch,0,0,sh)
    let (ch1, sh1) = qr_givens_quaternion(b11, b21);
    let a = 1.0 - 2.0 * sh1 * sh1;
    let b = 2.0 * ch1 * sh1;
    // apply B = Q' * B
    let (r11, r12, r13) = (a * b11 + b * b21, a * b12 + b * b22, a * b13 + b * b23);
    let (r21, r22, r23) = (-b * b11 + a * b21, -b * b12 + a * b22, -b * b13 + a * b23);
    let (r31, r32, r33) = (b31, b32, b33);

    // second givens rotation (ch,0,-sh,0)
    let (ch2, sh2) = qr_givens_quaternion(r11, r31);
    let a = 1.0 - 2.0 * sh2 * sh2;
    let b = 2.0 * ch2 * sh2;
    // apply B = Q' * B;
    let (b11, b12, b13) = (a * r11 + b * r31, a * r12 + b * r32, a * r13 + b * r33);
    let (b21, b22, b23) = (r21, r22, r23);
    let (b31, b32, b33) = (-b * r11 + a * r31, -b * r12 + a * r32, -b * r13 + a * r33);

    // third givens rotation (ch,sh,0,0)
    let (ch3, sh3) = qr_givens_quaternion(b22, b32);
    let a = 1.0 - 2.0 * sh3 * sh3;
    let b = 2.0 * ch3 * sh3;
    // R is now set to desired value
    let r = [
        b11,
        b12,
        b13,
        a * b21 + b * b31,
        a * b22 + b * b32,
        a * b23 + b * b33,
        -b * b21 + a * b31,
        -b * b22 + a * b32,
        -b * b23 + a * b33,
    ];

    // construct the cumulative rotation Q=Q1 * Q2 * Q3
    let sh12 = sh1 * sh1;
    let sh22 = sh2 * sh2;
    let sh32 = sh3 * sh3;
    let q = [
        (-1.0 + 2.0 * sh12) * (-1.0 + 2.0 * sh22),
        4.0 * ch2 * ch3 * (-1.0 + 2.0 * sh12) * sh2 * sh3 + 2.0 * ch1 * sh1 * (-1.0 + 2.0 * sh32),
        4.0 * ch1 * ch3 * sh1 * sh3 - 2.0 * ch2 * (-1.0 + 2.0 * sh12) * sh2 * (-1.0 + 2.0 * sh32),
        2.0 * ch1 * sh1 * (1.0 - 2.0 * sh22),
        -8.0 * ch1 * ch2 * ch3 * sh1 * sh2 * sh3 + (-1.0 + 2.0 * sh12) * (-1.0 + 2.0 * sh32),
        -2.0 * ch3 * sh3 + 4.0 * sh1 * (ch3 * sh1 * sh3 + ch1 * ch2 * sh2 * (-1.0 + 2.0 * sh32)),
        2.0 * ch2 * sh2,
        2.0 * ch3 * (1.0 - 2.0 * sh22) * sh3,
        (-1.0 + 2.0 * sh22) * (-1.0 + 2.0 * sh32),
    ];
    (q, r)
}

/// A = U S V^T. Returns (U, S, V).
pub fn svd(a: &M3) -> (M3, M3, M3) {
    // normal equations matrix
    let ata = mult_atb(a, a);

    // symmetric eigenalysis
    let mut s = Sym {
        s11: ata[0],
        s21: ata[3],
        s22: ata[4],
        s31: ata[6],
        s32: ata[7],
        s33: ata[8],
    };
    let qv = jacobi_eigenanalysis(&mut s);
    let mut v = quat_to_mat3(&qv);

    let mut b = mult_ab(a, &v);

    // sort singular values and find V
    sort_singular_values(&mut b, &mut v);

    // QR decomposition
    let (u, s) = qr_decomposition(&b);
    (u, s, v)
}
