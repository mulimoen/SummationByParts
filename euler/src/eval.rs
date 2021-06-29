//! Traits for evaluating initial conditions, exact solutions, or boundary conditions

use super::Float;
use super::GAMMA;
use ndarray::{azip, ArrayView, ArrayViewMut, Dimension};

pub trait Evaluator<D: Dimension>: Send + Sync {
    fn evaluate(
        &self,
        t: Float,
        x: ArrayView<Float, D>,
        y: ArrayView<Float, D>,
        rho: ArrayViewMut<Float, D>,
        rhou: ArrayViewMut<Float, D>,
        rhov: ArrayViewMut<Float, D>,
        e: ArrayViewMut<Float, D>,
    );
}

/// Necessary to avoid specialisation of `Evaluator` trait for items
/// that could implement both this and `Evaluator`
#[derive(Clone, Debug)]
pub struct EvaluatorPressureWrapper<'a, D: Dimension, E: EvaluatorPressure<D>>(
    &'a E,
    std::marker::PhantomData<D>,
);

impl<'a, D: Dimension, E: EvaluatorPressure<D>> EvaluatorPressureWrapper<'a, D, E> {
    pub fn new(e: &'a E) -> Self {
        Self(e, std::marker::PhantomData)
    }
}

pub trait EvaluatorPressure<D: Dimension>: Send + Sync {
    fn rho(
        &self,
        t: Float,
        x: ArrayView<Float, D>,
        y: ArrayView<Float, D>,
        out: ArrayViewMut<Float, D>,
    );
    fn u(
        &self,
        t: Float,
        x: ArrayView<Float, D>,
        y: ArrayView<Float, D>,
        rho: ArrayView<Float, D>,
        out: ArrayViewMut<Float, D>,
    );
    fn v(
        &self,
        t: Float,
        x: ArrayView<Float, D>,
        y: ArrayView<Float, D>,
        rho: ArrayView<Float, D>,
        out: ArrayViewMut<Float, D>,
    );
    fn p(
        &self,
        t: Float,
        x: ArrayView<Float, D>,
        y: ArrayView<Float, D>,
        rho: ArrayView<Float, D>,
        u: ArrayView<Float, D>,
        v: ArrayView<Float, D>,
        out: ArrayViewMut<Float, D>,
    );
}

impl<'a, D: Dimension, BP: EvaluatorPressure<D>> Evaluator<D>
    for EvaluatorPressureWrapper<'a, D, BP>
{
    fn evaluate(
        &self,
        t: Float,
        x: ArrayView<Float, D>,
        y: ArrayView<Float, D>,
        mut rho: ArrayViewMut<Float, D>,
        mut rhou: ArrayViewMut<Float, D>,
        mut rhov: ArrayViewMut<Float, D>,
        mut e: ArrayViewMut<Float, D>,
    ) {
        let eva = &self.0;
        eva.rho(t, x.view(), y.view(), rho.view_mut());
        eva.u(t, x.view(), y.view(), rho.view(), rhou.view_mut());
        eva.v(t, x.view(), y.view(), rho.view(), rhov.view_mut());
        eva.p(t, x, y, rho.view(), rhou.view(), rhov.view(), e.view_mut());

        azip!((rho in &rho, u in &rhou, v in &rhov, e in &mut e) {
            let p = *e;
            *e = p / (GAMMA - 1.0) + rho * (u*u + v*v) / 2.0;

        });
        azip!((rho in &rho, rhou in &mut rhou) *rhou *= rho);
        azip!((rho in &rho, rhov in &mut rhov) *rhov *= rho);
    }
}
