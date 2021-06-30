use euler::GAMMA;
use evalexpr::*;
use ndarray::{azip, ArrayView, ArrayViewMut, Dimension};
use sbp::Float;

#[derive(Clone, Debug)]
pub enum Evaluator {
    Pressure(EvaluatorPressure),
    Conservation(EvaluatorConservation),
}

#[derive(Debug, Clone)]
pub struct EvaluatorConservation {
    pub(crate) ctx: HashMapContext,
    pub(crate) rho: Node,
    pub(crate) rhou: Node,
    pub(crate) rhov: Node,
    pub(crate) e: Node,
}

impl<D: ndarray::Dimension> euler::eval::Evaluator<D> for Evaluator {
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
            Self::Conservation(c) => c.evaluate(t, x, y, rho, rhou, rhov, e),
            Self::Pressure(p) => {
                euler::eval::EvaluatorPressureWrapper::new(p).evaluate(t, x, y, rho, rhou, rhov, e)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct EvaluatorPressure {
    pub(crate) ctx: HashMapContext,
    pub(crate) rho: Node,
    pub(crate) u: Node,
    pub(crate) v: Node,
    pub(crate) p: Node,
}

struct ContextWrapper<'a> {
    ctx: &'a HashMapContext,
    x: Option<Value>,
    y: Option<Value>,
    t: Value,
    rho: Option<Value>,
    u: Option<Value>,
    v: Option<Value>,
    rhou: Option<Value>,
    rhov: Option<Value>,
    id: std::collections::HashMap<String, Value>,
}

impl<'a> ContextWrapper<'a> {
    fn wrap(ctx: &'a HashMapContext, t: Value) -> Self {
        Self {
            ctx,
            t,
            x: None,
            y: None,
            rho: None,
            rhou: None,
            rhov: None,
            u: None,
            v: None,
            id: std::collections::HashMap::new(),
        }
    }
}

impl Context for ContextWrapper<'_> {
    fn get_value(&self, identifier: &str) -> Option<&Value> {
        match identifier {
            "t" => Some(&self.t),
            "x" => self.x.as_ref(),
            "y" => self.y.as_ref(),
            "rho" => self.rho.as_ref(),
            "rhou" => self.rhou.as_ref(),
            "rhov" => self.rhov.as_ref(),
            "u" => self.u.as_ref(),
            "v" => self.v.as_ref(),
            id => self.id.get(id).or_else(|| self.ctx.get_value(id)),
        }
    }

    fn call_function(&self, identifier: &str, argument: &Value) -> EvalexprResult<Value> {
        self.ctx.call_function(identifier, argument)
    }
}

impl ContextWithMutableVariables for ContextWrapper<'_> {
    fn set_value(&mut self, identifier: String, value: Value) -> EvalexprResult<()> {
        self.id.insert(identifier, value);
        Ok(())
    }
}

impl<D: Dimension> euler::eval::Evaluator<D> for EvaluatorConservation {
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
        let mut ctx = ContextWrapper::wrap(&self.ctx, t.into());

        azip!((&x in &x.view(), &y in &y, rho in &mut rho) {
            ctx.x = Some(x.into());
            ctx.y = Some(y.into());

            *rho = self.rho.eval_number_with_context_mut(&mut ctx).unwrap();
        });

        azip!((&x in &x, &y in &y, &rho in &rho, rhou in &mut rhou) {
            ctx.x = Some(x.into());
            ctx.y = Some(y.into());
            ctx.rho = Some(rho.into());

            *rhou = self.rhou.eval_number_with_context_mut(&mut ctx).unwrap();
        });

        azip!((&x in &x, &y in &y, &rho in &rho, rhov in &mut rhov) {
            ctx.x = Some(x.into());
            ctx.y = Some(y.into());
            ctx.rho = Some(rho.into());

            *rhov = self.rhov.eval_number_with_context_mut(&mut ctx).unwrap();
        });

        azip!((&x in &x, &y in &y, &rho in &rho, &rhou in &rhou, &rhov in &rhov, e in &mut e) {
            ctx.x = Some(x.into());
            ctx.y = Some(y.into());
            ctx.rho = Some(rho.into());
            ctx.rhou = Some(rhou.into());
            ctx.rhov = Some(rhov.into());

            *e = self.e.eval_number_with_context_mut(&mut ctx).unwrap();
        });
    }
}

impl<D: Dimension> euler::eval::EvaluatorPressure<D> for EvaluatorPressure {
    fn rho(
        &self,
        t: Float,
        x: ArrayView<Float, D>,
        y: ArrayView<Float, D>,
        mut rho: ArrayViewMut<Float, D>,
    ) {
        let mut ctx = ContextWrapper::wrap(&self.ctx, t.into());

        azip!((&x in &x, &y in &y, rho in &mut rho) {
            ctx.x = Some(x.into());
            ctx.y = Some(y.into());

            *rho = self.rho.eval_number_with_context_mut(&mut ctx).unwrap();
        })
    }

    fn u(
        &self,
        t: Float,
        x: ArrayView<Float, D>,
        y: ArrayView<Float, D>,
        rho: ArrayView<Float, D>,
        mut u: ArrayViewMut<Float, D>,
    ) {
        let mut ctx = ContextWrapper::wrap(&self.ctx, t.into());

        azip!((&x in &x, &y in &y, &rho in &rho, u in &mut u) {
            ctx.x = Some(x.into());
            ctx.y = Some(y.into());
            ctx.rho = Some(rho.into());

            *u = self.u.eval_number_with_context_mut(&mut ctx).unwrap();
        })
    }
    fn v(
        &self,
        t: Float,
        x: ArrayView<Float, D>,
        y: ArrayView<Float, D>,
        rho: ArrayView<Float, D>,
        mut v: ArrayViewMut<Float, D>,
    ) {
        let mut ctx = ContextWrapper::wrap(&self.ctx, t.into());

        azip!((&x in &x, &y in &y, &rho in &rho, v in &mut v) {
            ctx.x = Some(x.into());
            ctx.y = Some(y.into());
            ctx.rho = Some(rho.into());

            *v = self.v.eval_number_with_context_mut(&mut ctx).unwrap();
        })
    }

    fn p(
        &self,
        t: Float,
        x: ArrayView<Float, D>,
        y: ArrayView<Float, D>,
        rho: ArrayView<Float, D>,
        u: ArrayView<Float, D>,
        v: ArrayView<Float, D>,
        mut p: ArrayViewMut<Float, D>,
    ) {
        let mut ctx = ContextWrapper::wrap(&self.ctx, t.into());

        azip!((&x in &x, &y in &y, &rho in &rho, &u in &u, &v in &v, p in &mut p) {
            ctx.x = Some(x.into());
            ctx.y = Some(y.into());
            ctx.rho = Some(rho.into());
            ctx.u = Some(u.into());
            ctx.v = Some(v.into());

            *p = self.p.eval_number_with_context_mut(&mut ctx).unwrap();
        })
    }
}

#[test]
fn append_default_context() {
    let basic_ctx = context_map! {
        "a" => 2,
    }
    .unwrap();

    let mut ctx = ContextWrapper::wrap(&basic_ctx, Value::from(4));
    ctx.x = Some(3.into());

    let expr = "a + x + t";

    let node = build_operator_tree(expr).unwrap();

    assert_eq!(node.eval_with_context(&ctx).unwrap().as_int().unwrap(), 9);
}

pub fn default_context() -> HashMapContext {
    let mut context = math_consts_context! {}.unwrap();
    let gamma = *GAMMA.get().expect("GAMMA is not defined");
    context.set_value("GAMMA".into(), gamma.into()).unwrap();

    context
        .set_function(
            "if".into(),
            Function::new(|arg| {
                let arg = arg.as_tuple()?;
                if arg.len() != 3 {
                    return Err(error::EvalexprError::WrongFunctionArgumentAmount {
                        expected: 3,
                        actual: arg.len(),
                    });
                }
                let b = arg[0].as_boolean()?;
                if b {
                    Ok(arg[1].clone())
                } else {
                    Ok(arg[2].clone())
                }
            }),
        )
        .unwrap();

    context
        .set_function(
            "case".into(),
            Function::new(|arg| {
                let mut arg = arg.as_tuple()?;
                if arg.len() % 2 == 0 {
                    return Err(error::EvalexprError::WrongFunctionArgumentAmount {
                        expected: arg.len() + 1,
                        actual: arg.len(),
                    });
                }
                //let mut arg = arg.as_slice();
                while arg.len() > 2 {
                    let boolean = arg.remove(0);
                    let value = arg.remove(0);

                    if boolean.as_boolean()? {
                        return Ok(value);
                    }
                }
                Ok(arg.pop().unwrap())
            }),
        )
        .unwrap();

    context
        .set_function(
            "math::atan2".into(),
            Function::new(|arg| {
                let arg = arg.as_tuple()?;
                if arg.len() != 2 {
                    return Err(error::EvalexprError::WrongFunctionArgumentAmount {
                        expected: 2,
                        actual: arg.len(),
                    });
                }
                let s = arg[0].as_number()?;
                let o = arg[1].as_number()?;
                Ok(s.atan2(o).into())
            }),
        )
        .unwrap();

    context
        .set_function(
            "math::hypot".into(),
            Function::new(|arg| {
                let arg = arg.as_tuple()?;
                if arg.len() != 2 {
                    return Err(error::EvalexprError::WrongFunctionArgumentAmount {
                        expected: 2,
                        actual: arg.len(),
                    });
                }
                let s = arg[0].as_number()?;
                let o = arg[1].as_number()?;
                Ok(s.hypot(o).into())
            }),
        )
        .unwrap();

    context
        .set_function(
            "math::exp".into(),
            Function::new(|arg| {
                let arg = arg.as_number()?;
                Ok(arg.exp().into())
            }),
        )
        .unwrap();

    context
        .set_function(
            "math::pow".into(),
            Function::new(|arg| {
                let arg = arg.as_tuple()?;
                if arg.len() != 2 {
                    return Err(error::EvalexprError::WrongFunctionArgumentAmount {
                        expected: 2,
                        actual: arg.len(),
                    });
                }
                let s = arg[0].as_number()?;
                let o = arg[1].as_number()?;
                Ok(s.powf(o).into())
            }),
        )
        .unwrap();

    context
}
