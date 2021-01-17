pub use arrayvec::ArrayVec;
use ndarray::azip;
use ndarray::prelude::*;
use sbp::grid::{Grid, Metrics};
use sbp::integrate;
use sbp::operators::{InterpolationOperator, SbpOperator2d, UpwindOperator2d};
use sbp::utils::Direction;
use sbp::Float;

pub const GAMMA: Float = 1.4;

// A collection of buffers that allows one to efficiently
// move to the next state
#[derive(Debug)]
pub struct System<SBP: SbpOperator2d> {
    sys: (Field, Field),
    k: [Field; 4],
    wb: WorkBuffers,
    grid: (Grid, Metrics),
    op: SBP,
}

impl<SBP: SbpOperator2d> System<SBP> {
    pub fn new(x: ndarray::Array2<Float>, y: ndarray::Array2<Float>, op: SBP) -> Self {
        let grid = Grid::new(x, y).expect(
            "Could not create grid. Different number of elements compared to width*height?",
        );
        let metrics = grid.metrics(&op).unwrap();
        let nx = grid.nx();
        let ny = grid.ny();
        Self {
            sys: (Field::new(ny, nx), Field::new(ny, nx)),
            grid: (grid, metrics),
            k: [
                Field::new(ny, nx),
                Field::new(ny, nx),
                Field::new(ny, nx),
                Field::new(ny, nx),
            ],
            wb: WorkBuffers::new(ny, nx),
            op,
        }
    }

    pub fn advance(&mut self, dt: Float) {
        let bc = BoundaryCharacteristics {
            north: BoundaryCharacteristic::This,
            south: BoundaryCharacteristic::This,
            east: BoundaryCharacteristic::This,
            west: BoundaryCharacteristic::This,
        };
        let op = &self.op;
        let wb = &mut self.wb.0;
        let grid = &self.grid.0;
        let metrics = &self.grid.1;
        let rhs_trad = |k: &mut Field, y: &Field, _time: Float| {
            let boundaries = boundary_extractor(y, grid, &bc);
            RHS_trad(op, k, y, metrics, &boundaries, wb)
        };
        integrate::integrate::<integrate::Rk4, _, _, _>(
            rhs_trad,
            &self.sys.0,
            &mut self.sys.1,
            &mut 0.0,
            dt,
            &mut self.k,
        );
        std::mem::swap(&mut self.sys.0, &mut self.sys.1);
    }

    pub fn vortex(&mut self, t: Float, vortex_parameters: VortexParameters) {
        self.sys
            .0
            .vortex(self.grid.0.x(), self.grid.0.y(), t, &vortex_parameters);
    }

    #[allow(clippy::many_single_char_names)]
    pub fn init_with_vortex(&mut self, x0: Float, y0: Float) {
        // Should parametrise such that we have radius, drop in pressure at center, etc
        let vortex_parameters = VortexParameters {
            vortices: {
                let mut v = ArrayVec::new();
                v.push(Vortice {
                    x0,
                    y0,
                    rstar: 1.0,
                    eps: 3.0,
                });
                v
            },
            mach: 0.5,
        };

        self.sys
            .0
            .vortex(self.grid.0.x(), self.grid.0.y(), 0.0, &vortex_parameters)
    }

    pub fn field(&self) -> &Field {
        &self.sys.0
    }

    pub fn x(&self) -> ArrayView2<Float> {
        self.grid.0.x()
    }
    pub fn y(&self) -> ArrayView2<Float> {
        self.grid.0.y()
    }

    pub fn nx(&self) -> usize {
        self.grid.0.nx()
    }
    pub fn ny(&self) -> usize {
        self.grid.0.ny()
    }
}

impl<UO: UpwindOperator2d + SbpOperator2d> System<UO> {
    pub fn advance_upwind(&mut self, dt: Float) {
        let bc = BoundaryCharacteristics {
            north: BoundaryCharacteristic::This,
            south: BoundaryCharacteristic::This,
            east: BoundaryCharacteristic::This,
            west: BoundaryCharacteristic::This,
        };
        let op = &self.op;
        let grid = &self.grid;
        let wb = &mut self.wb.0;
        let rhs_upwind = |k: &mut Field, y: &Field, _time: Float| {
            let (grid, metrics) = grid;
            let boundaries = boundary_extractor(y, grid, &bc);
            RHS_upwind(op, k, y, metrics, &boundaries, wb)
        };
        integrate::integrate::<integrate::Rk4, _, _, _>(
            rhs_upwind,
            &self.sys.0,
            &mut self.sys.1,
            &mut 0.0,
            dt,
            &mut self.k,
        );
        std::mem::swap(&mut self.sys.0, &mut self.sys.1);
    }
    pub fn advance_adaptive(&mut self, dt: Float, guess_dt: &mut Float, maxerr: Float) {
        let bc = BoundaryCharacteristics {
            north: BoundaryCharacteristic::This,
            south: BoundaryCharacteristic::This,
            east: BoundaryCharacteristic::This,
            west: BoundaryCharacteristic::This,
        };
        let op = &self.op;
        let grid = &self.grid;
        let wb = &mut self.wb.0;
        let mut rhs_upwind = |k: &mut Field, y: &Field, _time: Float| {
            let (grid, metrics) = grid;
            let boundaries = boundary_extractor(y, grid, &bc);
            RHS_upwind(op, k, y, metrics, &boundaries, wb)
        };
        let mut time = 0.0;
        let mut sys2 = self.sys.0.clone();
        while time < dt {
            integrate::integrate_embedded_rk::<integrate::BogackiShampine, _, _, _>(
                &mut rhs_upwind,
                &self.sys.0,
                &mut self.sys.1,
                &mut sys2,
                &mut time,
                *guess_dt,
                &mut self.k,
            );
            let err = self.sys.0.h2_err(&sys2, &self.op);
            if err < maxerr {
                time += *guess_dt;
                std::mem::swap(&mut self.sys.0, &mut self.sys.1);
                *guess_dt *= 1.05;
            } else {
                *guess_dt *= 0.8;
            }
        }
    }
}

#[derive(Clone, Debug)]
/// A 4 x ny x nx array
pub struct Field(pub(crate) Array3<Float>);

impl std::ops::Deref for Field {
    type Target = Array3<Float>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Field {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a> std::convert::From<&'a Field> for ArrayView3<'a, Float> {
    fn from(f: &'a Field) -> Self {
        f.0.view()
    }
}
impl<'a> std::convert::From<&'a mut Field> for ArrayViewMut3<'a, Float> {
    fn from(f: &'a mut Field) -> Self {
        f.0.view_mut()
    }
}

impl Field {
    pub fn new(ny: usize, nx: usize) -> Self {
        let field = Array3::zeros((4, ny, nx));

        Self(field)
    }

    pub fn nx(&self) -> usize {
        self.0.shape()[2]
    }
    pub fn ny(&self) -> usize {
        self.0.shape()[1]
    }

    pub fn rho(&self) -> ArrayView2<Float> {
        self.slice(s![0, .., ..])
    }
    pub fn rhou(&self) -> ArrayView2<Float> {
        self.slice(s![1, .., ..])
    }
    pub fn rhov(&self) -> ArrayView2<Float> {
        self.slice(s![2, .., ..])
    }
    pub fn e(&self) -> ArrayView2<Float> {
        self.slice(s![3, .., ..])
    }

    pub fn rho_mut(&mut self) -> ArrayViewMut2<Float> {
        self.slice_mut(s![0, .., ..])
    }
    pub fn rhou_mut(&mut self) -> ArrayViewMut2<Float> {
        self.slice_mut(s![1, .., ..])
    }
    pub fn rhov_mut(&mut self) -> ArrayViewMut2<Float> {
        self.slice_mut(s![2, .., ..])
    }
    pub fn e_mut(&mut self) -> ArrayViewMut2<Float> {
        self.slice_mut(s![3, .., ..])
    }

    #[allow(unused)]
    pub fn components(
        &self,
    ) -> (
        ArrayView2<Float>,
        ArrayView2<Float>,
        ArrayView2<Float>,
        ArrayView2<Float>,
    ) {
        (self.rho(), self.rhou(), self.rhov(), self.e())
    }
    #[allow(unused)]
    pub fn components_mut(
        &mut self,
    ) -> (
        ArrayViewMut2<Float>,
        ArrayViewMut2<Float>,
        ArrayViewMut2<Float>,
        ArrayViewMut2<Float>,
    ) {
        self.0
            .multi_slice_mut((s![0, .., ..], s![1, .., ..], s![2, .., ..], s![3, .., ..]))
    }

    pub fn north(&self) -> ArrayView2<Float> {
        self.slice(s![.., self.ny() - 1, ..])
    }
    pub fn south(&self) -> ArrayView2<Float> {
        self.slice(s![.., 0, ..])
    }
    pub fn east(&self) -> ArrayView2<Float> {
        self.slice(s![.., .., self.nx() - 1])
    }
    pub fn west(&self) -> ArrayView2<Float> {
        self.slice(s![.., .., 0])
    }
    fn north_mut(&mut self) -> ArrayViewMut2<Float> {
        let ny = self.ny();
        self.slice_mut(s![.., ny - 1, ..])
    }
    fn south_mut(&mut self) -> ArrayViewMut2<Float> {
        self.slice_mut(s![.., 0, ..])
    }
    fn east_mut(&mut self) -> ArrayViewMut2<Float> {
        let nx = self.nx();
        self.slice_mut(s![.., .., nx - 1])
    }
    fn west_mut(&mut self) -> ArrayViewMut2<Float> {
        self.slice_mut(s![.., .., 0])
    }

    pub fn vortex(
        &mut self,
        x: ArrayView2<Float>,
        y: ArrayView2<Float>,
        time: Float,
        vortex_param: &VortexParameters,
    ) {
        assert_eq!(x.shape(), y.shape());
        assert_eq!(x.shape()[1], self.nx());
        assert_eq!(x.shape()[0], self.ny());

        let (rho, rhou, rhov, e) = self.components_mut();
        let n = rho.len();

        vortex(
            rho.into_shape((n,)).unwrap(),
            rhou.into_shape((n,)).unwrap(),
            rhov.into_shape((n,)).unwrap(),
            e.into_shape((n,)).unwrap(),
            x.into_shape((n,)).unwrap(),
            y.into_shape((n,)).unwrap(),
            time,
            &vortex_param,
        )
    }
}

impl Field {
    /// sqrt((self-other)^T*H*(self-other))
    pub fn h2_err(&self, other: &Self, op: &dyn SbpOperator2d) -> Float {
        assert_eq!(self.nx(), other.nx());
        assert_eq!(self.ny(), other.ny());

        // Resulting structure should be
        // serialized(F0 - F1)^T (Hx kron Hy) serialized(F0 - F1)
        //
        // We accomplish this by serializing along x as fastest dimension
        // Since h is diagonal, it can be iterated with the following iterators

        // This chains the h block into the form [h, 1, 1, 1, rev(h)],
        // and multiplies with a factor
        let itermaker = move |h: &'static [Float], n: usize, factor: Float| {
            h.iter()
                .copied()
                .chain(std::iter::repeat(1.0).take(n - 2 * h.len()))
                .chain(h.iter().copied().rev())
                .map(move |x| x * factor)
        };

        let hxiterator = itermaker(
            op.hxi(),
            self.nx(),
            if op.is_h2xi() {
                1.0 / (self.nx() - 2) as Float
            } else {
                1.0 / (self.nx() - 1) as Float
            },
        );
        // Repeating to get the form
        // [[hx0, hx1, ..., hxn], [hx0, hx1, ..., hxn], ..., [hx0, hx1, ..., hxn]]
        let hxiterator = hxiterator.cycle().take(self.nx() * self.ny());

        let hyiterator = itermaker(
            op.heta(),
            self.ny(),
            1.0 / if op.is_h2eta() {
                (self.ny() - 2) as Float
            } else {
                (self.ny() - 1) as Float
            },
        );
        // Repeating to get the form
        // [[hy0, hy0, ..., hy0], [hy1, hy1, ..., hy1], ..., [hym, hym, ..., hym]]
        let hyiterator = hyiterator.flat_map(|x| std::iter::repeat(x).take(self.nx()));

        let diagiterator = hxiterator.zip(hyiterator).cycle();

        diagiterator
            .zip(self.0.iter())
            .zip(other.0.iter())
            .map(|(((hx, hy), r0), r1)| (*r0 - *r1).powi(2) * hx * hy)
            .sum::<Float>()
            .sqrt()
    }
}

#[test]
fn h2_diff() {
    let mut field0 = Field::new(20, 21);
    for f in field0.0.iter_mut() {
        *f = 1.0
    }
    let field1 = Field::new(20, 21);

    use sbp::operators::{Upwind4, Upwind9, SBP4, SBP8};

    assert!((field0.h2_err(&field1, &Upwind4).powi(2) - 4.0).abs() < 1e-3);
    assert!((field0.h2_err(&field1, &Upwind9).powi(2) - 4.0).abs() < 1e-3);
    assert!((field0.h2_err(&field1, &SBP4).powi(2) - 4.0).abs() < 1e-3);
    assert!((field0.h2_err(&field1, &SBP8).powi(2) - 4.0).abs() < 1e-3);
}

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
    pub vortices: ArrayVec<[Vortice; 5]>,
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

fn pressure(gamma: Float, rho: Float, rhou: Float, rhov: Float, e: Float) -> Float {
    (gamma - 1.0) * (e - (rhou * rhou + rhov * rhov) / (2.0 * rho))
}

#[allow(non_snake_case)]
pub fn RHS_trad(
    op: &dyn SbpOperator2d,
    k: &mut Field,
    y: &Field,
    metrics: &Metrics,
    boundaries: &BoundaryTerms,
    tmp: &mut (Field, Field, Field, Field, Field, Field),
) {
    let ehat = &mut tmp.0;
    let fhat = &mut tmp.1;
    fluxes((ehat, fhat), y, metrics, &mut tmp.2);
    let dE = &mut tmp.2;
    let dF = &mut tmp.3;

    op.diffxi(ehat.rho(), dE.rho_mut());
    op.diffxi(ehat.rhou(), dE.rhou_mut());
    op.diffxi(ehat.rhov(), dE.rhov_mut());
    op.diffxi(ehat.e(), dE.e_mut());

    op.diffeta(fhat.rho(), dF.rho_mut());
    op.diffeta(fhat.rhou(), dF.rhou_mut());
    op.diffeta(fhat.rhov(), dF.rhov_mut());
    op.diffeta(fhat.e(), dF.e_mut());

    azip!((out in &mut k.0,
                    eflux in &dE.0,
                    fflux in &dF.0,
                    detj in &metrics.detj().broadcast((4, y.ny(), y.nx())).unwrap()) {
        *out = (-eflux - fflux)/detj
    });

    SAT_characteristics(op, k, y, metrics, boundaries);
}

#[allow(non_snake_case)]
pub fn RHS_upwind(
    op: &dyn SbpOperator2d,
    k: &mut Field,
    y: &Field,
    metrics: &Metrics,
    boundaries: &BoundaryTerms,
    tmp: &mut (Field, Field, Field, Field, Field, Field),
) {
    let ehat = &mut tmp.0;
    let fhat = &mut tmp.1;
    fluxes((ehat, fhat), y, metrics, &mut tmp.2);
    let dE = &mut tmp.2;
    let dF = &mut tmp.3;

    op.diffxi(ehat.rho(), dE.rho_mut());
    op.diffxi(ehat.rhou(), dE.rhou_mut());
    op.diffxi(ehat.rhov(), dE.rhov_mut());
    op.diffxi(ehat.e(), dE.e_mut());

    op.diffeta(fhat.rho(), dF.rho_mut());
    op.diffeta(fhat.rhou(), dF.rhou_mut());
    op.diffeta(fhat.rhov(), dF.rhov_mut());
    op.diffeta(fhat.e(), dF.e_mut());

    let ad_xi = &mut tmp.4;
    let ad_eta = &mut tmp.5;
    let diss_op = op.upwind().expect("This is not an upwind operator");
    upwind_dissipation(
        &*diss_op,
        (ad_xi, ad_eta),
        y,
        metrics,
        (&mut tmp.0, &mut tmp.1),
    );

    azip!((out in &mut k.0,
                    eflux in &dE.0,
                    fflux in &dF.0,
                    ad_xi in &ad_xi.0,
                    ad_eta in &ad_eta.0,
                    detj in &metrics.detj().broadcast((4, y.ny(), y.nx())).unwrap()) {
        *out = (-eflux - fflux + ad_xi + ad_eta)/detj
    });

    SAT_characteristics(op, k, y, metrics, boundaries);
}

#[allow(clippy::many_single_char_names)]
fn upwind_dissipation(
    op: &dyn UpwindOperator2d,
    k: (&mut Field, &mut Field),
    y: &Field,
    metrics: &Metrics,
    tmp: (&mut Field, &mut Field),
) {
    let n = y.nx() * y.ny();
    let yview = y.view().into_shape((4, n)).unwrap();
    let mut tmp0 = tmp.0.view_mut().into_shape((4, n)).unwrap();
    let mut tmp1 = tmp.1.view_mut().into_shape((4, n)).unwrap();

    for ((((((y, mut tmp0), mut tmp1), detj_dxi_dx), detj_dxi_dy), detj_deta_dx), detj_deta_dy) in
        yview
            .axis_iter(ndarray::Axis(1))
            .zip(tmp0.axis_iter_mut(ndarray::Axis(1)))
            .zip(tmp1.axis_iter_mut(ndarray::Axis(1)))
            .zip(metrics.detj_dxi_dx().iter())
            .zip(metrics.detj_dxi_dy().iter())
            .zip(metrics.detj_deta_dx().iter())
            .zip(metrics.detj_deta_dy().iter())
    {
        let rho = y[0];
        assert!(rho > 0.0);
        let rhou = y[1];
        let rhov = y[2];
        let e = y[3];

        let u = rhou / rho;
        let v = rhov / rho;

        let uhat = detj_dxi_dx * u + detj_dxi_dy * v;
        let vhat = detj_deta_dx * u + detj_deta_dy * v;

        let p = pressure(GAMMA, rho, rhou, rhov, e);
        assert!(p > 0.0);
        let c = (GAMMA * p / rho).sqrt();

        let alpha_u = uhat.abs() + c * Float::hypot(*detj_dxi_dx, *detj_dxi_dy);
        let alpha_v = vhat.abs() + c * Float::hypot(*detj_deta_dx, *detj_deta_dy);

        tmp0[0] = alpha_u * rho;
        tmp1[0] = alpha_v * rho;

        tmp0[1] = alpha_u * rhou;
        tmp1[1] = alpha_v * rhou;

        tmp0[2] = alpha_u * rhov;
        tmp1[2] = alpha_v * rhov;

        tmp0[3] = alpha_u * e;
        tmp1[3] = alpha_v * e;
    }

    op.dissxi(tmp.0.rho(), k.0.rho_mut());
    op.dissxi(tmp.0.rhou(), k.0.rhou_mut());
    op.dissxi(tmp.0.rhov(), k.0.rhov_mut());
    op.dissxi(tmp.0.e(), k.0.e_mut());

    op.disseta(tmp.1.rho(), k.1.rho_mut());
    op.disseta(tmp.1.rhou(), k.1.rhou_mut());
    op.disseta(tmp.1.rhov(), k.1.rhov_mut());
    op.disseta(tmp.1.e(), k.1.e_mut());
}

/// Computes the fluxes
///
/// eflux = [rhou, rhou*rhou/rho + p, rhou*rhov/rho, rhou*(e+p)/rho]
/// fflux = [rhov, rhou*rhov/rho, rhov*rhov/rho + p, rhov*(e+p)/rho]
fn fluxes(k: (&mut Field, &mut Field), y: &Field, metrics: &Metrics, wb: &mut Field) {
    let rho = y.rho();
    let rhou = y.rhou();
    let rhov = y.rhov();
    let e = y.e();

    let mut p = wb.rho_mut();
    azip!((p in &mut p, &rho in &rho, &rhou in &rhou, &rhov in &rhov, &e in &e) {
        *p = pressure(GAMMA, rho, rhou, rhov, e)
    });

    k.0.rho_mut().assign(&rhou);
    azip!((eflux in k.0.rhou_mut(), rho in &rho, rhou in &rhou, p in &p) {
        *eflux = rhou*rhou/rho + p;
    });
    azip!((eflux in k.0.rhov_mut(), rho in &rho, rhou in &rhou, rhov in &rhov) {
        *eflux = rhou*rhov/rho;
    });
    azip!((eflux in k.0.e_mut(), rho in &rho, rhou in &rhou, e in &e, p in &p) {
        *eflux = rhou*(e + p)/rho;
    });

    k.1.rho_mut().assign(&rhov);
    k.1.rhou_mut().assign(&k.0.rhov_mut());
    azip!((fflux in k.1.rhov_mut(), rho in &rho, rhov in &rhov, p in &p) {
        *fflux = rhov*rhov/rho + p;
    });
    azip!((fflux in k.1.e_mut(), rho in &rho, rhov in &rhov, e in &e, p in &p) {
        *fflux = rhov*(e + p)/rho;
    });

    let j_dxi_dx = metrics.detj_dxi_dx();
    let j_dxi_dy = metrics.detj_dxi_dy();
    let j_deta_dx = metrics.detj_deta_dx();
    let j_deta_dy = metrics.detj_deta_dy();
    // Let grid metrics modify the fluxes
    for comp in 0..4 {
        azip!((ef in k.0.slice_mut(s![comp, .., ..]),
               ff in k.1.slice_mut(s![comp, .., ..]),
               j_dxi_dx in &j_dxi_dx,
               j_dxi_dy in &j_dxi_dy,
               j_deta_dx in &j_deta_dx,
               j_deta_dy in &j_deta_dy) {

            let eflux = *ef;
            let fflux = *ff;
            *ef = j_dxi_dx * eflux + j_dxi_dy * fflux;
            *ff = j_deta_dx * eflux + j_deta_dy * fflux;
        })
    }
}

pub enum BoundaryCharacteristic {
    This,
    Grid(usize),
    Vortex(VortexParameters),
    // Vortices(Vec<VortexParameters>),
    Interpolate(usize, Box<dyn InterpolationOperator>),
    MultiGrid(Vec<(usize, usize, usize)>),
}

pub type BoundaryTerms<'a> = Direction<ArrayView2<'a, Float>>;
pub type BoundaryCharacteristics = Direction<BoundaryCharacteristic>;

fn boundary_extractor<'a>(
    field: &'a Field,
    _grid: &Grid,
    bc: &BoundaryCharacteristics,
) -> BoundaryTerms<'a> {
    BoundaryTerms {
        north: match &bc.north {
            BoundaryCharacteristic::This => field.south(),
            BoundaryCharacteristic::Vortex(_params) => todo!(),
            BoundaryCharacteristic::Grid(_)
            | BoundaryCharacteristic::Interpolate(_, _)
            | BoundaryCharacteristic::MultiGrid(_) => panic!("Only working on self grid"),
        },
        south: match &bc.south {
            BoundaryCharacteristic::This => field.north(),
            BoundaryCharacteristic::Vortex(_params) => todo!(),
            BoundaryCharacteristic::Grid(_)
            | BoundaryCharacteristic::Interpolate(_, _)
            | BoundaryCharacteristic::MultiGrid(_) => panic!("Only working on self grid"),
        },
        west: match &bc.west {
            BoundaryCharacteristic::This => field.east(),
            BoundaryCharacteristic::Vortex(_params) => todo!(),
            BoundaryCharacteristic::Grid(_)
            | BoundaryCharacteristic::Interpolate(_, _)
            | BoundaryCharacteristic::MultiGrid(_) => panic!("Only working on self grid"),
        },
        east: match &bc.east {
            BoundaryCharacteristic::This => field.west(),
            BoundaryCharacteristic::Vortex(_params) => todo!(),
            BoundaryCharacteristic::Grid(_)
            | BoundaryCharacteristic::Interpolate(_, _)
            | BoundaryCharacteristic::MultiGrid(_) => panic!("Only working on self grid"),
        },
    }
}

fn boundary_extract<'a>(
    fields: &'a [Field],
    bc: &BoundaryCharacteristic,
    field: &'a Field,
    grid: (ArrayView1<Float>, ArrayView1<Float>),
    seldir: impl Fn(&Field) -> ArrayView2<Float>,
    eb: Option<&'a mut Array2<Float>>,
    time: Float,
) -> ArrayView2<'a, Float> {
    match bc {
        BoundaryCharacteristic::This => seldir(field),
        BoundaryCharacteristic::Grid(g) => seldir(&fields[*g]),
        BoundaryCharacteristic::Vortex(v) => {
            let field = eb.unwrap();
            vortexify(field.view_mut(), grid, v, time);
            field.view()
        }
        BoundaryCharacteristic::Interpolate(g, operator) => {
            let to = eb.unwrap();
            let fine2coarse = field.nx() < fields[*g].nx();

            for (mut to, from) in to.outer_iter_mut().zip(seldir(&fields[*g]).outer_iter()) {
                if fine2coarse {
                    operator.fine2coarse(from.view(), to.view_mut());
                } else {
                    operator.coarse2fine(from.view(), to.view_mut());
                }
            }
            to.view()
        }
        BoundaryCharacteristic::MultiGrid(grids) => {
            let to = eb.unwrap();
            let mut i = 0;
            let mut remaining = grids.len();
            for &(g, start, end) in grids.iter() {
                let n: usize = end - start;
                to.slice_mut(s![.., i..i + n])
                    .assign(&seldir(&fields[g]).slice(s![.., start..end]));
                remaining -= 1;
                if remaining != 0 {
                    to.slice_mut(s![.., i]).iter_mut().for_each(|x| *x /= 2.0);
                    i += n - 1;
                } else {
                    i += n;
                    assert_eq!(i, to.len_of(Axis(1)));
                }
            }
            to.view()
        }
    }
}

pub fn boundary_extracts<'a>(
    fields: &'a [Field],
    bt: &BoundaryCharacteristics,
    field: &'a Field,
    grid: &Grid,
    eb: &'a mut BoundaryStorage,
    time: Float,
) -> BoundaryTerms<'a> {
    BoundaryTerms {
        north: boundary_extract(
            fields,
            &bt.north,
            field,
            grid.north(),
            |f| f.south(),
            eb.north.as_mut(),
            time,
        ),
        south: boundary_extract(
            fields,
            &bt.south,
            field,
            grid.south(),
            |f| f.north(),
            eb.south.as_mut(),
            time,
        ),
        east: boundary_extract(
            fields,
            &bt.east,
            field,
            grid.east(),
            |f| f.west(),
            eb.east.as_mut(),
            time,
        ),
        west: boundary_extract(
            fields,
            &bt.west,
            field,
            grid.west(),
            |f| f.east(),
            eb.west.as_mut(),
            time,
        ),
    }
}

pub fn extract_boundaries<'a>(
    fields: &'a [Field],
    bt: &[BoundaryCharacteristics],
    eb: &'a mut [BoundaryStorage],
    grids: &[Grid],
    time: Float,
) -> Vec<BoundaryTerms<'a>> {
    bt.iter()
        .zip(eb)
        .zip(grids)
        .zip(fields)
        .map(|(((bt, eb), grid), field)| boundary_extracts(fields, bt, field, grid, eb, time))
        .collect()
}

/// Used for storing boundary elements
pub struct BoundaryStorage {
    north: Option<ndarray::Array2<Float>>,
    south: Option<ndarray::Array2<Float>>,
    east: Option<ndarray::Array2<Float>>,
    west: Option<ndarray::Array2<Float>>,
}

impl BoundaryStorage {
    pub fn new(bt: &BoundaryCharacteristics, grid: &Grid) -> Self {
        Self {
            north: match bt.north() {
                BoundaryCharacteristic::Vortex(_)
                | BoundaryCharacteristic::Interpolate(_, _)
                | BoundaryCharacteristic::MultiGrid(_) => {
                    Some(ndarray::Array2::zeros((4, grid.nx())))
                }
                _ => None,
            },
            south: match bt.south() {
                BoundaryCharacteristic::Vortex(_)
                | BoundaryCharacteristic::Interpolate(_, _)
                | BoundaryCharacteristic::MultiGrid(_) => {
                    Some(ndarray::Array2::zeros((4, grid.nx())))
                }
                _ => None,
            },
            east: match bt.east() {
                BoundaryCharacteristic::Vortex(_)
                | BoundaryCharacteristic::Interpolate(_, _)
                | BoundaryCharacteristic::MultiGrid(_) => {
                    Some(ndarray::Array2::zeros((4, grid.ny())))
                }
                _ => None,
            },
            west: match bt.west() {
                BoundaryCharacteristic::Vortex(_)
                | BoundaryCharacteristic::Interpolate(_, _)
                | BoundaryCharacteristic::MultiGrid(_) => {
                    Some(ndarray::Array2::zeros((4, grid.ny())))
                }
                _ => None,
            },
        }
    }
}

fn vortexify(
    mut field: ndarray::ArrayViewMut2<Float>,
    yx: (ndarray::ArrayView1<Float>, ndarray::ArrayView1<Float>),
    vparams: &VortexParameters,
    time: Float,
) {
    let mut fiter = field.outer_iter_mut();
    let (rho, rhou, rhov, e) = (
        fiter.next().unwrap(),
        fiter.next().unwrap(),
        fiter.next().unwrap(),
        fiter.next().unwrap(),
    );
    let (y, x) = yx;
    vortex(rho, rhou, rhov, e, x, y, time, &vparams);
}

#[allow(non_snake_case)]
/// Boundary conditions (SAT)
fn SAT_characteristics(
    op: &dyn SbpOperator2d,
    k: &mut Field,
    y: &Field,
    metrics: &Metrics,
    boundaries: &BoundaryTerms,
) {
    // North boundary
    {
        let hi = if op.is_h2eta() {
            (k.ny() - 2) as Float / op.heta()[0]
        } else {
            (k.ny() - 1) as Float / op.heta()[0]
        };
        let sign = -1.0;
        let tau = 1.0;
        let slice = s![y.ny() - 1, ..];
        SAT_characteristic(
            k.north_mut(),
            y.north(),
            boundaries.north,
            hi,
            sign,
            tau,
            metrics.detj().slice(slice),
            metrics.detj_deta_dx().slice(slice),
            metrics.detj_deta_dy().slice(slice),
        );
    }
    // South boundary
    {
        let hi = if op.is_h2eta() {
            (k.ny() - 2) as Float / op.heta()[0]
        } else {
            (k.ny() - 1) as Float / op.heta()[0]
        };
        let sign = 1.0;
        let tau = -1.0;
        let slice = s![0, ..];
        SAT_characteristic(
            k.south_mut(),
            y.south(),
            boundaries.south,
            hi,
            sign,
            tau,
            metrics.detj().slice(slice),
            metrics.detj_deta_dx().slice(slice),
            metrics.detj_deta_dy().slice(slice),
        );
    }
    // West Boundary
    {
        let hi = if op.is_h2xi() {
            (k.nx() - 2) as Float / op.hxi()[0]
        } else {
            (k.nx() - 1) as Float / op.hxi()[0]
        };
        let sign = 1.0;
        let tau = -1.0;
        let slice = s![.., 0];
        SAT_characteristic(
            k.west_mut(),
            y.west(),
            boundaries.west,
            hi,
            sign,
            tau,
            metrics.detj().slice(slice),
            metrics.detj_dxi_dx().slice(slice),
            metrics.detj_dxi_dy().slice(slice),
        );
    }
    // East Boundary
    {
        let hi = if op.is_h2xi() {
            (k.nx() - 2) as Float / op.hxi()[0]
        } else {
            (k.nx() - 1) as Float / op.hxi()[0]
        };
        let sign = -1.0;
        let tau = 1.0;
        let slice = s![.., y.nx() - 1];
        SAT_characteristic(
            k.east_mut(),
            y.east(),
            boundaries.east,
            hi,
            sign,
            tau,
            metrics.detj().slice(slice),
            metrics.detj_dxi_dx().slice(slice),
            metrics.detj_dxi_dy().slice(slice),
        );
    }
}

#[allow(non_snake_case)]
#[allow(clippy::many_single_char_names)]
#[allow(clippy::too_many_arguments)]
/// Boundary conditions (SAT)
fn SAT_characteristic(
    mut k: ArrayViewMut2<Float>,
    y: ArrayView2<Float>,
    z: ArrayView2<Float>, // Size 4 x n (all components in line)
    hi: Float,
    sign: Float,
    tau: Float,
    detj: ArrayView1<Float>,
    detj_d_dx: ArrayView1<Float>,
    detj_d_dy: ArrayView1<Float>,
) {
    assert_eq!(detj.shape(), detj_d_dx.shape());
    assert_eq!(detj.shape(), detj_d_dy.shape());
    assert_eq!(y.shape(), z.shape());
    assert_eq!(y.shape()[0], 4);
    assert_eq!(y.shape()[1], detj.shape()[0]);

    for (((((mut k, y), z), detj), detj_d_dx), detj_d_dy) in k
        .axis_iter_mut(ndarray::Axis(1))
        .zip(y.axis_iter(ndarray::Axis(1)))
        .zip(z.axis_iter(ndarray::Axis(1)))
        .zip(detj.iter())
        .zip(detj_d_dx.iter())
        .zip(detj_d_dy.iter())
    {
        let rho = y[0];
        let rhou = y[1];
        let rhov = y[2];
        let e = y[3];

        let kx_ = detj_d_dx / detj;
        let ky_ = detj_d_dy / detj;

        let (kx, ky) = {
            let r = Float::hypot(kx_, ky_);
            (kx_ / r, ky_ / r)
        };

        let u = rhou / rho;
        let v = rhov / rho;

        let theta = kx * u + ky * v;

        let p = pressure(GAMMA, rho, rhou, rhov, e);
        let c = (GAMMA * p / rho).sqrt();
        let phi2 = (GAMMA - 1.0) * (u * u + v * v) / 2.0;
        let alpha = rho / (sbp::consts::SQRT_2 * c);

        let phi2_c2 = (phi2 + c * c) / (GAMMA - 1.0);

        #[rustfmt::skip]
        let T = [
            [                 1.0,                   0.0,                       alpha,                       alpha],
            [                   u,                    ky,          alpha*(u + kx * c),          alpha*(u - kx * c)],
            [                   v,                   -kx,          alpha*(v + ky * c),          alpha*(v - ky * c)],
            [phi2 / (GAMMA - 1.0), rho*(ky * u - kx * v), alpha*(phi2_c2 + c * theta), alpha*(phi2_c2 - c * theta)],
        ];
        let U = kx_ * u + ky_ * v;
        let L = [
            U,
            U,
            U + c * Float::hypot(kx_, ky_),
            U - c * Float::hypot(kx_, ky_),
        ];
        let beta = 1.0 / (2.0 * c * c);
        #[rustfmt::skip]
        let TI = [
            [     1.0 - phi2 / (c * c),          (GAMMA - 1.0) * u / (c * c),          (GAMMA - 1.0) * v / (c * c), -(GAMMA - 1.0) / (c * c)],
            [   -(ky * u - kx * v)/rho,                               ky/rho,                              -kx/rho,                      0.0],
            [beta * (phi2 - c * theta),  beta * (kx * c - (GAMMA - 1.0) * u),  beta * (ky * c - (GAMMA - 1.0) * v),     beta * (GAMMA - 1.0)],
            [beta * (phi2 + c * theta), -beta * (kx * c + (GAMMA - 1.0) * u), -beta * (ky * c + (GAMMA - 1.0) * v),     beta * (GAMMA - 1.0)],
        ];

        let res = [rho - z[0], rhou - z[1], rhov - z[2], e - z[3]];
        let mut TIres = [0.0; 4];
        #[allow(clippy::needless_range_loop)]
        for row in 0..4 {
            for col in 0..4 {
                TIres[row] += TI[row][col] * res[col];
            }
        }

        // L + sign(abs(L)) * TIres
        let mut LTIres = [0.0; 4];
        for row in 0..4 {
            LTIres[row] = (L[row] + sign * L[row].abs()) * TIres[row];
        }

        // T*LTIres
        let mut TLTIres = [0.0; 4];
        #[allow(clippy::needless_range_loop)]
        for row in 0..4 {
            for col in 0..4 {
                TLTIres[row] += T[row][col] * LTIres[col];
            }
        }

        for comp in 0..4 {
            k[comp] += hi * tau * TLTIres[comp];
        }
    }
}

#[derive(Debug)]
pub struct WorkBuffers(pub (Field, Field, Field, Field, Field, Field));

impl WorkBuffers {
    pub fn new(nx: usize, ny: usize) -> Self {
        let arr3 = Field::new(nx, ny);
        Self((
            arr3.clone(),
            arr3.clone(),
            arr3.clone(),
            arr3.clone(),
            arr3.clone(),
            arr3,
        ))
    }
}
