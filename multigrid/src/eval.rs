use ndarray::{ArrayView, ArrayViewMut, Dimension};
use sbp::Float;

pub mod evalexpr;

#[derive(Clone, Debug)]
pub enum Evaluator {
    EvalExpr(evalexpr::Evaluator),
}

impl<D: Dimension> euler::eval::Evaluator<D> for Evaluator {
    fn evaluate(
        &self,
        t: Float,
        x: ArrayView<Float, D>,
        y: ArrayView<Float, D>,
        rho: ArrayViewMut<Float, D>,
        rhou: ArrayViewMut<Float, D>,
        rhov: ArrayViewMut<Float, D>,
        e: ArrayViewMut<Float, D>,
    ) {
        match self {
            Self::EvalExpr(c) => c.evaluate(t, x, y, rho, rhou, rhov, e),
        }
    }
}
