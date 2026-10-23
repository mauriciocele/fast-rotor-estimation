//! Valkenburg & Dorst's geometric-algebra rotor estimation, with the products written out.

use crate::common::*;

fn lmap_x(x: &Vec3, ps: &[Vec3], qs: &[Vec3], w: &[f64]) -> Vec4 {
    //X = X1*e1*e2 + X2*e1*e3 + X3*e2*e3;
    //P = P1*e1 + P2*e2 + P3*e3;
    //Q = Q1*e1 + Q2*e2 + Q3*e3;
    //?QX = Q*X;
    //?Y = QX*P;
    let mut r = Vec4::zeros();
    for ((p, q), &wk) in ps.iter().zip(qs).zip(w) {
        let qx = [
            (-(q[2] * x[1])) - q[1] * x[0],          // e1
            q[0] * x[0] - q[2] * x[2],               // e2
            q[1] * x[2] + q[0] * x[1],               // e3
            q[0] * x[2] - q[1] * x[1] + q[2] * x[0], // e1^e2^e3
        ];
        let y = Vec4::new(
            p[2] * qx[2] + p[1] * qx[1] + p[0] * qx[0],      // 1
            p[2] * qx[3] - p[0] * qx[1] + p[1] * qx[0],      // e1^e2
            (-(p[1] * qx[3])) - p[0] * qx[2] + p[2] * qx[0], // e1^e3
            p[0] * qx[3] - p[1] * qx[2] + p[2] * qx[1],      // e2^e3
        );
        r += wk * y;
    }
    r
}

fn lmap(ps: &[Vec3], qs: &[Vec3], w: &[f64]) -> Vec4 {
    //P = P1*e1 + P2*e2 + P3*e3;
    //Q = Q1*e1 + Q2*e2 + Q3*e3;
    //?Y = Q*P;
    let mut r = Vec4::zeros();
    for ((p, q), &wk) in ps.iter().zip(qs).zip(w) {
        let y = Vec4::new(
            p[2] * q[2] + p[1] * q[1] + p[0] * q[0], // 1.0
            p[1] * q[0] - p[0] * q[1],               // e1 ^ e2
            p[2] * q[0] - p[0] * q[2],               // e1 ^ e3
            p[2] * q[1] - p[1] * q[2],               // e2 ^ e3
        );
        r += wk * y;
    }
    r
}

/// Scalar part of X * R, for the bivector
/// X = X1*e1*e2 + X2*e1*e3 + X3*e2*e3 and the rotor R = R1 + R2*e1*e2 + R3*e1*e3 + R4*e2*e3.
#[inline]
fn bivector_x_rotor(x: &Vec3, r: &Vec4) -> f64 {
    (-(r[3] * x[2])) - r[2] * x[1] - r[1] * x[0] // 1.0
}

pub fn ga_valkenburg(p: &[Vec3], q: &[Vec3], w: &[f64]) -> Quat {
    let im = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
    ];
    let imi = [
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, -1.0),
    ];

    let mut l = Mat4::zeros();
    for j in 0..4 {
        let lj = if j == 0 { lmap(p, q, w) } else { lmap_x(&im[j], p, q, w) };
        l[(j, j)] = if j == 0 { lj[0] } else { bivector_x_rotor(&imi[j], &lj) };
        for i in j + 1..4 {
            l[(i, j)] = bivector_x_rotor(&imi[i], &lj);
            l[(j, i)] = l[(i, j)];
        }
    }

    let rot = eigenvector_max(l);
    quat(rot[0], -rot[3], -rot[2], -rot[1])
}
