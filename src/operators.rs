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

#[macro_export]
macro_rules! diff_op_1d {
    ($self: ty, $name: ident, $BLOCK: expr, $DIAG: expr, $symmetric: expr) => {
        impl $self {
            fn $name(prev: ArrayView1<f32>, mut fut: ArrayViewMut1<f32>) {
                assert_eq!(prev.shape(), fut.shape());
                let nx = prev.shape()[0];
                assert!(nx >= 2 * $BLOCK.len());

                let dx = 1.0 / (nx - 1) as f32;
                let idx = 1.0 / dx;

                let block = ::ndarray::arr2($BLOCK);
                let diag = ::ndarray::arr1($DIAG);


                let first_elems = prev.slice(::ndarray::s!(..block.len_of(::ndarray::Axis(1))));
                for (bl, f) in block.outer_iter().zip(&mut fut) {
                    let diff = first_elems.dot(&bl);
                    *f = diff * idx;
                }

                // The window needs to be aligned to the diagonal elements,
                // based on the block size
                let window_elems_to_skip =
                    block.len_of(::ndarray::Axis(0)) - ((diag.len() - 1) / 2);

                for (window, f) in prev
                    .windows(diag.len())
                    .into_iter()
                    .skip(window_elems_to_skip)
                    .zip(fut.iter_mut().skip(block.len_of(::ndarray::Axis(0))))
                    .take(nx - 2 * block.len_of(::ndarray::Axis(0)))
                {
                    let diff = diag.dot(&window);
                    *f = diff * idx;
                }

                let last_elems = prev.slice(::ndarray::s!(nx - block.len_of(::ndarray::Axis(1))..;-1));
                for (bl, f) in block.outer_iter()
                    .zip(&mut fut.slice_mut(s![nx - block.len_of(::ndarray::Axis(0))..;-1]))
                {
                    let diff = if $symmetric {
                        bl.dot(&last_elems)
                    } else {
                        -bl.dot(&last_elems)
                    };
                    *f = diff * idx;
                }
            }
        }
    };
}

mod upwind4;
pub use upwind4::Upwind4;
mod traditional4;
pub use traditional4::SBP4;
