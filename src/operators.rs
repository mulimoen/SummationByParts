use ndarray::{ArrayView2, ArrayViewMut2};

pub trait SbpOperator {
    fn diffxi(prev: ArrayView2<f32>, fut: ArrayViewMut2<f32>);
    fn diffeta(prev: ArrayView2<f32>, fut: ArrayViewMut2<f32>);
    fn h() -> &'static [f32];
}

pub trait UpwindOperator: SbpOperator {
    fn dissxi(prev: ArrayView2<f32>, fut: ArrayViewMut2<f32>);
    fn disseta(prev: ArrayView2<f32>, fut: ArrayViewMut2<f32>);
}

mod upwind4;
pub use upwind4::Upwind4;
