//! The Float type used throughout the ecosystem

#![cfg_attr(feature = "fast-float", feature(core_intrinsics))]

/// Type used for floats, configure with the `f32` feature
#[cfg(feature = "f32")]
pub type Float = f32;
#[cfg(not(feature = "f32"))]
/// Type used for floats, configure with the `f32` feature
pub type Float = f64;

/// Associated constants for [`Float`]
pub mod consts {
    #[cfg(feature = "f32")]
    pub use std::f32::consts::*;
    #[cfg(not(feature = "f32"))]
    pub use std::f64::consts::*;
}

#[cfg(feature = "fast-float")]
mod fastfloat;
#[cfg(feature = "fast-float")]
pub use fastfloat::FastFloat;
