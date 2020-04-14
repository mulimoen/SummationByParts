#![allow(clippy::excessive_precision)]
#![allow(clippy::unreadable_literal)]

use crate::Float;
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};

pub trait SbpOperator1d: Copy + Clone + core::fmt::Debug {
    fn diff(&self, prev: ArrayView1<Float>, fut: ArrayViewMut1<Float>);

    fn h(&self) -> &'static [Float];
    fn is_h2(&self) -> bool {
        false
    }
}

pub trait SbpOperator2d: Copy + Clone {
    fn diffxi(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>);
    fn diffeta(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>);

    fn hxi(&self) -> &'static [Float];
    fn heta(&self) -> &'static [Float];

    fn is_h2xi(&self) -> bool;
    fn is_h2eta(&self) -> bool;
}

impl<SBPeta: SbpOperator1d, SBPxi: SbpOperator1d> SbpOperator2d for (SBPeta, SBPxi) {
    fn diffxi(&self, prev: ArrayView2<Float>, mut fut: ArrayViewMut2<Float>) {
        assert_eq!(prev.shape(), fut.shape());
        for (r0, r1) in prev.outer_iter().zip(fut.outer_iter_mut()) {
            self.1.diff(r0, r1)
        }
    }
    fn diffeta(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>) {
        let ba = (self.1, self.0);
        ba.diffxi(prev.reversed_axes(), fut.reversed_axes())
    }
    fn hxi(&self) -> &'static [Float] {
        self.1.h()
    }
    fn heta(&self) -> &'static [Float] {
        self.0.h()
    }
    fn is_h2xi(&self) -> bool {
        self.1.is_h2()
    }
    fn is_h2eta(&self) -> bool {
        self.0.is_h2()
    }
}

impl<SBP: SbpOperator1d> SbpOperator2d for SBP {
    fn diffxi(&self, prev: ArrayView2<Float>, mut fut: ArrayViewMut2<Float>) {
        assert_eq!(prev.shape(), fut.shape());
        for (r0, r1) in prev.outer_iter().zip(fut.outer_iter_mut()) {
            self.diff(r0, r1)
        }
    }
    fn diffeta(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>) {
        self.diffxi(prev.reversed_axes(), fut.reversed_axes())
    }
    fn hxi(&self) -> &'static [Float] {
        self.h()
    }
    fn heta(&self) -> &'static [Float] {
        self.h()
    }
    fn is_h2xi(&self) -> bool {
        self.is_h2()
    }
    fn is_h2eta(&self) -> bool {
        self.is_h2()
    }
}

pub trait UpwindOperator1d: SbpOperator1d + Copy + Clone {
    fn diss(&self, prev: ArrayView1<Float>, fut: ArrayViewMut1<Float>);
}

pub trait UpwindOperator2d: SbpOperator2d + Copy + Clone {
    fn dissxi(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>);
    fn disseta(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>);
}

impl<UOeta: UpwindOperator1d, UOxi: UpwindOperator1d> UpwindOperator2d for (UOeta, UOxi) {
    fn dissxi(&self, prev: ArrayView2<Float>, mut fut: ArrayViewMut2<Float>) {
        assert_eq!(prev.shape(), fut.shape());
        for (r0, r1) in prev.outer_iter().zip(fut.outer_iter_mut()) {
            self.1.diss(r0, r1);
        }
    }
    fn disseta(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>) {
        let ba = (self.1, self.0);
        ba.dissxi(prev.reversed_axes(), fut.reversed_axes())
    }
}

impl<UO: UpwindOperator1d> UpwindOperator2d for UO {
    fn dissxi(&self, prev: ArrayView2<Float>, mut fut: ArrayViewMut2<Float>) {
        assert_eq!(prev.shape(), fut.shape());
        for (r0, r1) in prev.outer_iter().zip(fut.outer_iter_mut()) {
            self.diss(r0, r1);
        }
    }
    fn disseta(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>) {
        self.dissxi(prev.reversed_axes(), fut.reversed_axes())
    }
}

pub trait InterpolationOperator: Send + Sync {
    fn fine2coarse(&self, fine: ArrayView1<Float>, coarse: ArrayViewMut1<Float>);
    fn coarse2fine(&self, coarse: ArrayView1<Float>, fine: ArrayViewMut1<Float>);
}

#[inline(always)]
pub(crate) fn diff_op_1d(
    block: ndarray::ArrayView2<Float>,
    diag: ndarray::ArrayView1<Float>,
    symmetric: bool,
    is_h2: bool,
    prev: ArrayView1<Float>,
    mut fut: ArrayViewMut1<Float>,
) {
    assert_eq!(prev.shape(), fut.shape());
    let nx = prev.shape()[0];
    assert!(nx >= 2 * block.len_of(ndarray::Axis(0)));

    let dx = if is_h2 {
        1.0 / (nx - 2) as Float
    } else {
        1.0 / (nx - 1) as Float
    };
    let idx = 1.0 / dx;

    let first_elems = prev.slice(::ndarray::s!(..block.len_of(::ndarray::Axis(1))));
    for (bl, f) in block.outer_iter().zip(&mut fut) {
        let diff = first_elems.dot(&bl);
        *f = diff * idx;
    }

    // The window needs to be aligned to the diagonal elements,
    // based on the block size
    let window_elems_to_skip = block.len_of(::ndarray::Axis(0)) - ((diag.len() - 1) / 2);

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
    for (bl, f) in block
        .outer_iter()
        .zip(&mut fut.slice_mut(::ndarray::s![nx - block.len_of(::ndarray::Axis(0))..;-1]))
    {
        let diff = if symmetric {
            bl.dot(&last_elems)
        } else {
            -bl.dot(&last_elems)
        };
        *f = diff * idx;
    }
}

mod upwind4;
pub use upwind4::Upwind4;
mod upwind9;
pub use upwind9::Upwind9;

mod upwind4h2;
pub use upwind4h2::Upwind4h2;
mod upwind9h2;
pub use upwind9h2::Upwind9h2;

mod traditional4;
pub use traditional4::SBP4;
mod traditional8;
pub use traditional8::SBP8;

mod interpolation;
pub use interpolation::Interpolation4;

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
        op: SBP,
        n: (usize, usize),
        f: F,
        dfdx: FX,
        dfdy: FY,
        eps: Float,
    ) where
        SBP: SbpOperator2d,
        F: Fn(Float, Float) -> Float,
        FX: Fn(Float, Float) -> Float,
        FY: Fn(Float, Float) -> Float,
    {
        let mut y = Array2::zeros(n);
        let x = grid_eval(n, f);

        y.fill(0.0);
        op.diffxi(x.view(), y.view_mut());
        approx::assert_abs_diff_eq!(&y, &grid_eval(n, dfdx), epsilon = eps);

        y.fill(0.0);
        op.diffeta(x.view(), y.view_mut());
        approx::assert_abs_diff_eq!(&y, &grid_eval(n, dfdy), epsilon = eps);
    }
}
