//! The Float type used throughout the ecosystem
#![no_std]
#![forbid(unsafe_code)]

/// Type used for floats, configure with the `f32` feature
#[cfg(feature = "f32")]
pub type Float = f32;
#[cfg(not(feature = "f32"))]
/// Type used for floats, configure with the `f32` feature
pub type Float = f64;

/// Associated constants for [`Float`]
pub mod consts {
    #[cfg(feature = "f32")]
    pub use core::f32::consts::*;
    #[cfg(not(feature = "f32"))]
    pub use core::f64::consts::*;
}
