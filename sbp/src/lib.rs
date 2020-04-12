#![feature(str_strip)]

#[cfg(feature = "f32")]
pub type Float = f32;
#[cfg(not(feature = "f32"))]
pub type Float = f64;

pub(crate) mod consts {
    #[cfg(feature = "f32")]
    pub(crate) use std::f32::consts::*;
    #[cfg(not(feature = "f32"))]
    pub(crate) use std::f64::consts::*;
}

pub mod euler;
pub mod grid;
pub mod integrate;
pub mod maxwell;
pub mod operators;
pub mod utils;
