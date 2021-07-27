#![cfg_attr(feature = "fast-float", feature(core_intrinsics))]
#![feature(array_windows)]
#![feature(array_chunks)]
#![feature(const_fn_floating_point_arithmetic)]
#![feature(portable_simd)]

pub use float::{consts, Float};

/// Grid and grid metrics
pub mod grid;
/// SBP and interpolation operators
pub mod operators;
/// General utilities
pub mod utils;
