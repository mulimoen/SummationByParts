#![cfg_attr(feature = "fast-float", feature(core_intrinsics))]
#![feature(array_windows)]
#![feature(array_chunks)]
#![feature(const_fn_floating_point_arithmetic)]

pub use float::{consts, Float};

/// Grid and grid metrics
pub mod grid;
/// RK operators and methods for implicit integration
pub mod integrate;
/// SBP and interpolation operators
pub mod operators;
/// General utilities
pub mod utils;
