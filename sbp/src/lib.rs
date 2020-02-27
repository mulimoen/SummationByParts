#[cfg(feature = "f32")]
pub type Float = f32;
#[cfg(not(feature = "f32"))]
pub type Float = f64;

pub mod euler;
pub mod grid;
pub mod integrate;
pub mod maxwell;
pub mod operators;
pub mod utils;
