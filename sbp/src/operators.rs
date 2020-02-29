#![allow(clippy::excessive_precision)]
#![allow(clippy::unreadable_literal)]

use crate::Float;

use ndarray::{ArrayView2, ArrayViewMut2};

pub trait SbpOperator {
    fn diffxi(prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>);
    fn diffeta(prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>);
    fn h() -> &'static [Float];
}

pub trait UpwindOperator: SbpOperator {
    fn dissxi(prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>);
    fn disseta(prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>);
}

#[macro_export]
macro_rules! diff_op_1d {
    ($self: ty, $name: ident, $BLOCK: expr, $DIAG: expr, $symmetric: expr) => {
        impl $self {
            fn $name(prev: ArrayView1<Float>, mut fut: ArrayViewMut1<Float>) {
                assert_eq!(prev.shape(), fut.shape());
                let nx = prev.shape()[0];
                assert!(nx >= 2 * $BLOCK.len());

                let dx = 1.0 / (nx - 1) as Float;
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
mod upwind9;
pub use upwind9::Upwind9;
mod traditional4;
pub use traditional4::SBP4;
mod traditional8;
pub use traditional8::SBP8;

#[cfg(test)]
pub(crate) mod testing {
    use super::*;
    use ndarray::prelude::*;
    pub(crate) fn grid_eval<F: Fn(Float, Float) -> Float>(
        n: (usize, usize),
        f: F,
    ) -> Array2<Float> {
        let nx = n.1;
        let dx = 1.0 / (nx - 1) as Float;
        let ny = n.0;
        let dy = 1.0 / (ny - 1) as Float;
        Array2::from_shape_fn(n, |(j, i)| {
            let x = dx * i as Float;
            let y = dy * j as Float;
            f(x, y)
        })
    }

    pub(crate) fn check_operator_on<SBP, F, FX, FY>(
        n: (usize, usize),
        f: F,
        dfdx: FX,
        dfdy: FY,
        eps: Float,
    ) where
        SBP: SbpOperator,
        F: Fn(Float, Float) -> Float,
        FX: Fn(Float, Float) -> Float,
        FY: Fn(Float, Float) -> Float,
    {
        let mut y = Array2::zeros(n);
        let x = grid_eval(n, f);

        y.fill(0.0);
        SBP::diffxi(x.view(), y.view_mut());
        approx::assert_abs_diff_eq!(&y, &grid_eval(n, dfdx), epsilon = eps);

        y.fill(0.0);
        SBP::diffeta(x.view(), y.view_mut());
        approx::assert_abs_diff_eq!(&y, &grid_eval(n, dfdy), epsilon = eps);
    }
}
