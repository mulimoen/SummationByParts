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

#[allow(clippy::too_many_arguments)]
#[allow(clippy::many_single_char_names)]
pub fn vortex(
    rho: ArrayViewMut1<Float>,
    rhou: ArrayViewMut1<Float>,
    rhov: ArrayViewMut1<Float>,
    e: ArrayViewMut1<Float>,
    x: ArrayView1<Float>,
    y: ArrayView1<Float>,
    time: Float,
    vortex_param: &VortexParameters,
) {
    assert_eq!(rho.len(), rhou.len());
    assert_eq!(rho.len(), rhov.len());
    assert_eq!(rho.len(), e.len());
    assert_eq!(rho.len(), x.len());
    assert_eq!(rho.len(), y.len());
    assert_eq!(x.shape(), y.shape());

    let m = vortex_param.mach;
    let p_inf = 1.0 / (GAMMA * m * m);

    let rho_inf: Float = 1.0;
    let u_inf: Float = 1.0;
    let v_inf: Float = 0.0;
    let e_inf = p_inf / (GAMMA - 1.0) + rho_inf * (u_inf.powi(2) + v_inf.powi(2)) / 2.0;

    azip!((rho in rho,
           rhou in rhou,
           rhov in rhov,
           e in e,
           x in x,
           y in y)
    {

        let mut iterator = vortex_param.vortices.iter();

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

                *rho = Float::powf(1.0 - eps*eps*(GAMMA - 1.0)*m*m/(8.0*PI*PI*p_inf*rstar*rstar)*f.exp(), 1.0/(GAMMA - 1.0));
                assert!(*rho > 0.0);
                let p = Float::powf(*rho, GAMMA)*p_inf;
                let u = 1.0 - eps*dy/(2.0*PI*p_inf.sqrt()*rstar*rstar)*(f/2.0).exp();
                let v =       eps*dx/(2.0*PI*p_inf.sqrt()*rstar*rstar)*(f/2.0).exp();

                assert!(p > 0.0);

                assert!(*rho > 0.0);
                *rhou = *rho*u;
                *rhov = *rho*v;
                *e = p/(GAMMA - 1.0) + *rho*(u*u + v*v)/2.0;
            }
        }

        for vortice in iterator {
            use sbp::consts::PI;

            let rstar = vortice.rstar;
            let eps = vortice.eps;

            let dx = (x - vortice.x0) - time;
            let dy = y - vortice.y0;
            let f = (1.0 - (dx*dx + dy*dy))/(rstar*rstar);

            let rho_vortice = Float::powf(1.0 - eps*eps*(GAMMA - 1.0)*m*m/(8.0*PI*PI*p_inf*rstar*rstar)*f.exp(), 1.0/(GAMMA - 1.0));
            let p = Float::powf(rho_vortice, GAMMA)*p_inf;
            let u = 1.0 - eps*dy/(2.0*PI*p_inf.sqrt()*rstar*rstar)*(f/2.0).exp();
            let v =       eps*dx/(2.0*PI*p_inf.sqrt()*rstar*rstar)*(f/2.0).exp();

            assert!(rho_vortice > 0.0);
            assert!(p > 0.0);

            *rho += rho_vortice - rho_inf;
            assert!(*rho > 0.0);
            *rhou += rho_vortice*u - rho_inf*u_inf;
            *rhov += rho_vortice*v - rho_inf*v_inf;
            *e += (p/(GAMMA - 1.0) + rho_vortice*(u*u + v*v)/2.0) - e_inf;
        }
    });
}
