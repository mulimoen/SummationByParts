#![allow(clippy::excessive_precision)]
#![allow(clippy::unreadable_literal)]

mod algos;
pub(crate) use algos::*;

use crate::Float;
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};

/// One-dimensional Summation By Parts operator
pub trait SbpOperator1d: Send + Sync {
    /// Differentiate on unit grid
    fn diff(&self, prev: ArrayView1<Float>, fut: ArrayViewMut1<Float>);

    /// block component of `H`, with diagonal component 1
    fn h(&self) -> &'static [Float];
    /// Whether the operator acts on a `h2` spaced computational grid
    fn is_h2(&self) -> bool {
        false
    }
    #[cfg(feature = "sparse")]
    fn diff_matrix(&self, n: usize) -> sprs::CsMat<Float>;
    #[cfg(feature = "sparse")]
    fn h_matrix(&self, n: usize) -> sprs::CsMat<Float>;

    fn upwind(&self) -> Option<&dyn UpwindOperator1d> {
        None
    }
    fn d2(&self) -> Option<&dyn SbpOperator1d2> {
        None
    }
}

pub trait SbpOperator1d2: SbpOperator1d {
    /// Double differentiation on unit grid
    fn diff2(&self, prev: ArrayView1<Float>, fut: ArrayViewMut1<Float>);
    /// Result of H^-1 * d1, without the 1/h scaling
    fn d1(&self) -> &[Float];

    #[cfg(feature = "sparse")]
    fn diff2_matrix(&self, n: usize) -> sprs::CsMat<Float>;
    #[cfg(feature = "sparse")]
    fn d1_vec(&self, n: usize, front: bool) -> sprs::CsMat<Float>;
}

pub trait UpwindOperator1d: SbpOperator1d + Send + Sync {
    /// Dissipation operator
    fn diss(&self, prev: ArrayView1<Float>, fut: ArrayViewMut1<Float>);
    fn as_sbp(&self) -> &dyn SbpOperator1d;

    #[cfg(feature = "sparse")]
    fn diss_matrix(&self, n: usize) -> sprs::CsMat<Float>;
}

pub trait SbpOperator2d: Send + Sync {
    fn diffxi(&self, prev: ArrayView2<Float>, mut fut: ArrayViewMut2<Float>) {
        assert_eq!(prev.shape(), fut.shape());
        for (p, f) in prev.outer_iter().zip(fut.outer_iter_mut()) {
            self.op_xi().diff(p, f);
        }
    }
    fn diffeta(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>) {
        self.diffxi(prev.reversed_axes(), fut.reversed_axes());
    }

    fn hxi(&self) -> &'static [Float] {
        self.op_xi().h()
    }
    fn heta(&self) -> &'static [Float] {
        self.op_eta().h()
    }

    fn is_h2xi(&self) -> bool {
        self.op_xi().is_h2()
    }
    fn is_h2eta(&self) -> bool {
        self.op_eta().is_h2()
    }

    fn op_xi(&self) -> &dyn SbpOperator1d;
    fn op_eta(&self) -> &dyn SbpOperator1d {
        self.op_xi()
    }

    fn upwind(&self) -> Option<Box<dyn UpwindOperator2d>> {
        None
    }
}

pub trait UpwindOperator2d: Send + Sync {
    fn dissxi(&self, prev: ArrayView2<Float>, mut fut: ArrayViewMut2<Float>) {
        assert_eq!(prev.shape(), fut.shape());
        for (p, f) in prev.outer_iter().zip(fut.outer_iter_mut()) {
            UpwindOperator2d::op_xi(self).diss(p, f);
        }
    }
    // Assuming operator is symmetrical x/y
    fn disseta(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>) {
        self.dissxi(prev.reversed_axes(), fut.reversed_axes());
    }

    fn op_xi(&self) -> &dyn UpwindOperator1d;
    fn op_eta(&self) -> &dyn UpwindOperator1d {
        self.op_xi()
    }
}

pub trait InterpolationOperator: Send + Sync {
    /// Interpolation from a grid with twice resolution
    fn fine2coarse(&self, fine: ArrayView1<Float>, coarse: ArrayViewMut1<Float>);
    /// Interpolation from a grid with half resolution
    fn coarse2fine(&self, coarse: ArrayView1<Float>, fine: ArrayViewMut1<Float>);
}

impl SbpOperator2d for (Box<dyn SbpOperator2d>, Box<dyn SbpOperator2d>) {
    fn diffxi(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>) {
        self.1.diffxi(prev, fut);
    }
    fn diffeta(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>) {
        self.0.diffeta(prev, fut);
    }

    fn op_xi(&self) -> &dyn SbpOperator1d {
        self.1.op_xi()
    }
    fn op_eta(&self) -> &dyn SbpOperator1d {
        self.0.op_eta()
    }
    fn upwind(&self) -> Option<Box<dyn UpwindOperator2d>> {
        match (self.0.upwind(), self.1.upwind()) {
            (Some(u), Some(v)) => Some(Box::new((u, v))),
            _ => None,
        }
    }
}

impl UpwindOperator2d for (Box<dyn UpwindOperator2d>, Box<dyn UpwindOperator2d>) {
    fn dissxi(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>) {
        self.1.dissxi(prev, fut);
    }
    fn disseta(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>) {
        self.0.disseta(prev, fut);
    }
    fn op_xi(&self) -> &dyn UpwindOperator1d {
        self.1.op_xi()
    }
    fn op_eta(&self) -> &dyn UpwindOperator1d {
        self.0.op_eta()
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
pub use interpolation::{Interpolation4, Interpolation8, Interpolation9, Interpolation9h2};

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
