#![feature(core_intrinsics)]
#![feature(array_windows)]
#![feature(array_chunks)]

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

/// Grid and grid metrics
pub mod grid;
/// RK operators and methods for implicit integration
pub mod integrate;
/// SBP and interpolation operators
pub mod operators;
/// General utilities
pub mod utils;
