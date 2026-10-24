//! Rotation estimation from weighted vector correspondences (Wahba's problem):
//! find the rotation R minimizing `sum_i w_i |R p_i - q_i|^2`.
//!
//! Every estimator takes the same arguments `(P, Q, w)` and returns either a
//! quaternion or a rotation matrix.

// Constants are kept digit-for-digit as in the C++ original.
#![allow(clippy::excessive_precision)]

pub mod arun_svd;
pub mod common;
pub mod davenport;
pub mod fa3r;
pub mod flae;
pub mod ga_rotor_estimator;
pub mod ga_valkenburg;
pub mod horn;
pub mod multivector;
pub mod quaternion_direct;
pub mod quest;
pub mod svd3;

pub use arun_svd::{svd_eigen, svd_mcadams};
pub use common::{Mat3, Quat, Vec3};
pub use davenport::davenport;
pub use fa3r::{fa3r_double, fa3r_int};
pub use flae::{flae, flae_newton, flae_symbolic};
pub use ga_rotor_estimator::{ga_rotor_estimator, la_rotor_estimator, prior_rotor_estimator};
pub use ga_valkenburg::ga_valkenburg;
pub use horn::horn;
pub use quaternion_direct::quaternion_direct;
pub use quest::{foam, quest};

/// `sum_i |R(q) p_i - q_i|^2` for a quaternion estimate.
pub fn wahba_error_quat(p: &[Vec3], q: &[Vec3], r: &Quat) -> f64 {
    p.iter()
        .zip(q)
        .map(|(pi, qi)| (common::transform_vector(r, pi) - qi).norm_squared())
        .sum()
}

/// `sum_i |R p_i - q_i|^2` for a matrix estimate.
pub fn wahba_error_mat(p: &[Vec3], q: &[Vec3], r: &Mat3) -> f64 {
    p.iter().zip(q).map(|(pi, qi)| (r * pi - qi).norm_squared()).sum()
}
