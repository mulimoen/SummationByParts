use super::*;

#[derive(Copy, Clone, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct Vortice {
    pub x0: Float,
    pub y0: Float,
    pub rstar: Float,
    pub eps: Float,
}

#[cfg(feature = "serde1")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde1", derive(Serialize, Deserialize))]
pub struct VortexParameters {
    // The limit of 5 vortices can be bumped if needed...
    pub vortices: ArrayVec<Vortice, 5>,
    pub mach: Float,
}

impl<D: Dimension> eval::Evaluator<D> for VortexParameters {
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::many_single_char_names)]
    fn evaluate(
        &self,
        time: Float,
        x: ArrayView<Float, D>,
        y: ArrayView<Float, D>,
        rho: ArrayViewMut<Float, D>,
        rhou: ArrayViewMut<Float, D>,
        rhov: ArrayViewMut<Float, D>,
        e: ArrayViewMut<Float, D>,
    ) {
        let gamma = *GAMMA.get().expect("GAMMA is not defined");
        let m = self.mach;
        let p_inf = 1.0 / (gamma * m * m);

        let rho_inf: Float = 1.0;
        let u_inf: Float = 1.0;
        let v_inf: Float = 0.0;
        let e_inf = p_inf / (gamma - 1.0) + rho_inf * (u_inf.powi(2) + v_inf.powi(2)) / 2.0;

        azip!((rho in rho,
               rhou in rhou,
               rhov in rhov,
               e in e,
               x in x,
               y in y)
        {

            let mut iterator = self.vortices.iter();

            match iterator.next() {
                None => {
                    *rho = rho_inf;
                    *rhou = rho_inf*u_inf;
                    *rhou = rho_inf*v_inf;
                    *e = e_inf;
                    return;
                },
                Some(vortice) => {
                    use sbp::consts::PI;

                    let rstar = vortice.rstar;
                    let eps = vortice.eps;

                    let dx = (x - vortice.x0) - time;
                    let dy = y - vortice.y0;
                    let f = (1.0 - (dx*dx + dy*dy))/(rstar*rstar);

                    *rho = Float::powf(1.0 - eps*eps*(gamma - 1.0)*m*m/(8.0*PI*PI*p_inf*rstar*rstar)*f.exp(), 1.0/(gamma - 1.0));
                    assert!(*rho > 0.0);
                    let p = Float::powf(*rho, gamma)*p_inf;
                    let u = 1.0 - eps*dy/(2.0*PI*p_inf.sqrt()*rstar*rstar)*(f/2.0).exp();
                    let v =       eps*dx/(2.0*PI*p_inf.sqrt()*rstar*rstar)*(f/2.0).exp();

                    assert!(p > 0.0);

                    assert!(*rho > 0.0);
                    *rhou = *rho*u;
                    *rhov = *rho*v;
                    *e = p/(gamma - 1.0) + *rho*(u*u + v*v)/2.0;
                }
            }

            for vortice in iterator {
                use sbp::consts::PI;

                let rstar = vortice.rstar;
                let eps = vortice.eps;

                let dx = (x - vortice.x0) - time;
                let dy = y - vortice.y0;
                let f = (1.0 - (dx*dx + dy*dy))/(rstar*rstar);

                let rho_vortice = Float::powf(1.0 - eps*eps*(gamma - 1.0)*m*m/(8.0*PI*PI*p_inf*rstar*rstar)*f.exp(), 1.0/(gamma - 1.0));
                let p = Float::powf(rho_vortice, gamma)*p_inf;
                let u = 1.0 - eps*dy/(2.0*PI*p_inf.sqrt()*rstar*rstar)*(f/2.0).exp();
                let v =       eps*dx/(2.0*PI*p_inf.sqrt()*rstar*rstar)*(f/2.0).exp();

                assert!(rho_vortice > 0.0);
                assert!(p > 0.0);

                *rho += rho_vortice - rho_inf;
                assert!(*rho > 0.0);
                *rhou += rho_vortice*u - rho_inf*u_inf;
                *rhov += rho_vortice*v - rho_inf*v_inf;
                *e += (p/(gamma - 1.0) + rho_vortice*(u*u + v*v)/2.0) - e_inf;
            }
        });
    }
}
