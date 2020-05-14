#![feature(str_strip)]
#![feature(specialization)]
#![feature(core_intrinsics)]

#[cfg(feature = "f32")]
pub type Float = f32;
#[cfg(not(feature = "f32"))]
pub type Float = f64;

pub mod consts {
    #[cfg(feature = "f32")]
    pub use std::f32::consts::*;
    #[cfg(not(feature = "f32"))]
    pub use std::f64::consts::*;
}

pub mod grid;
pub mod integrate;
pub mod operators;
pub mod utils;
