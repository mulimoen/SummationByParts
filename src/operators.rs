use ndarray::{ArrayView2, ArrayViewMut2};

pub trait SbpOperator {
    fn diffx(prev: ArrayView2<f32>, fut: ArrayViewMut2<f32>);
    fn diffy(prev: ArrayView2<f32>, fut: ArrayViewMut2<f32>);
    fn h() -> &'static [f32];
}

mod upwind4;
pub use upwind4::Upwind4;
