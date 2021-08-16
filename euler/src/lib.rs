pub use arrayvec::ArrayVec;
use ndarray::azip;
use ndarray::prelude::*;
use once_cell::sync::OnceCell;
use sbp::grid::{Grid, Metrics};
use sbp::operators::{InterpolationOperator, SbpOperator2d, UpwindOperator2d};
use sbp::utils::Direction;
use sbp::Float;

pub mod eval;
use eval::Evaluator;
mod vortex;
pub use vortex::{VortexParameters, Vortice};

pub static GAMMA: OnceCell<Float> = OnceCell::new();

// A collection of buffers that allows one to efficiently
// move to the next state
#[derive(Debug)]
pub struct System<SBP: SbpOperator2d> {
    sys: (Field, Field),
    k: [Diff; 4],
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
                Diff::zeros((ny, nx)),
                Diff::zeros((ny, nx)),
                Diff::zeros((ny, nx)),
                Diff::zeros((ny, nx)),
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
        let rhs_trad = |k: &mut Diff, y: &Field, _time: Float| {
            let boundaries = boundary_extractor(y, grid, &bc);
            RHS_trad(op, k, y, metrics, &boundaries, wb)
        };
        integrate::integrate::<integrate::Rk4, Field, _>(
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
        let rhs_upwind = |k: &mut Diff, y: &Field, _time: Float| {
            let (grid, metrics) = grid;
            let boundaries = boundary_extractor(y, grid, &bc);
            RHS_upwind(op, k, y, metrics, &boundaries, wb)
        };
        integrate::integrate::<integrate::Rk4, Field, _>(
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
        let mut rhs_upwind = |k: &mut Diff, y: &Field, _time: Float| {
            let (grid, metrics) = grid;
            let boundaries = boundary_extractor(y, grid, &bc);
            RHS_upwind(op, k, y, metrics, &boundaries, wb)
        };
        let mut time = 0.0;
        let mut sys2 = self.sys.0.clone();
        while time < dt {
            integrate::integrate_embedded_rk::<integrate::BogackiShampine, Field, _>(
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

#[derive(Debug)]
/// A 4 x ny x nx array
pub struct Field(pub(crate) Array3<Float>);

impl Clone for Field {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
    fn clone_from(&mut self, source: &Self) {
        self.0.clone_from(&source.0)
    }
}

#[derive(Clone, Debug)]
/// A 4 x ny x nx array
pub struct Diff(pub(crate) Array3<Float>);

impl integrate::Integrable for Field {
    type State = Field;
    type Diff = Diff;

    fn scaled_add(s: &mut Self::State, o: &Self::Diff, scale: Float) {
        s.0.scaled_add(scale, &o.0);
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

    pub(crate) fn slice<Do: Dimension>(
        &self,
        info: &ndarray::SliceInfo<[ndarray::SliceOrIndex; 3], Do>,
    ) -> ArrayView<Float, Do> {
        self.0.slice(info)
    }

    pub(crate) fn slice_mut<Do: Dimension>(
        &mut self,
        info: &ndarray::SliceInfo<[ndarray::SliceOrIndex; 3], Do>,
    ) -> ArrayViewMut<Float, Do> {
        self.0.slice_mut(info)
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
    #[allow(unused)]
    fn north_mut(&mut self) -> ArrayViewMut2<Float> {
        let ny = self.ny();
        self.slice_mut(s![.., ny - 1, ..])
    }
    #[allow(unused)]
    fn south_mut(&mut self) -> ArrayViewMut2<Float> {
        self.slice_mut(s![.., 0, ..])
    }
    #[allow(unused)]
    fn east_mut(&mut self) -> ArrayViewMut2<Float> {
        let nx = self.nx();
        self.slice_mut(s![.., .., nx - 1])
    }
    #[allow(unused)]
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
        let (rho, rhou, rhov, e) = self.components_mut();
        vortex_param.evaluate(time, x, y, rho, rhou, rhov, e)
    }
    #[allow(clippy::erasing_op, clippy::identity_op)]
    fn iter(&self) -> impl ExactSizeIterator<Item = FieldValue> + '_ {
        let n = self.nx() * self.ny();
        let slice = self.0.as_slice().unwrap();
        let rho = &slice[0 * n..1 * n];
        let rhou = &slice[1 * n..2 * n];
        let rhov = &slice[2 * n..3 * n];
        let e = &slice[3 * n..4 * n];

        rho.iter()
            .zip(rhou)
            .zip(rhov)
            .zip(e)
            .map(|(((&rho, &rhou), &rhov), &e)| FieldValue { rho, rhou, rhov, e })
    }
    fn iter_mut(&mut self) -> impl ExactSizeIterator<Item = FieldValueMut<'_>> + '_ {
        let n = self.nx() * self.ny();
        let slice = self.0.as_slice_mut().unwrap();
        let (rho, slice) = slice.split_at_mut(n);
        let (rhou, slice) = slice.split_at_mut(n);
        let (rhov, slice) = slice.split_at_mut(n);
        let (e, slice) = slice.split_at_mut(n);
        assert_eq!(slice.len(), 0);

        rho.iter_mut()
            .zip(rhou.iter_mut())
            .zip(rhov.iter_mut())
            .zip(e.iter_mut())
            .map(|(((rho, rhou), rhov), e)| FieldValueMut { rho, rhou, rhov, e })
    }
}

struct FieldValue {
    rho: Float,
    rhou: Float,
    rhov: Float,
    e: Float,
}

struct FieldValueMut<'a> {
    rho: &'a mut Float,
    rhou: &'a mut Float,
    rhov: &'a mut Float,
    e: &'a mut Float,
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

// Lazy inheritance...
impl core::ops::Deref for Diff {
    type Target = Array3<Float>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Diff {
    pub fn zeros((ny, nx): (usize, usize)) -> Self {
        Self(Array3::zeros((4, ny, nx)))
    }
    fn north_mut(&mut self) -> ArrayViewMut2<Float> {
        let ny = self.shape()[1];
        self.0.slice_mut(s![.., ny - 1, ..])
    }
    fn south_mut(&mut self) -> ArrayViewMut2<Float> {
        self.0.slice_mut(s![.., 0, ..])
    }
    fn east_mut(&mut self) -> ArrayViewMut2<Float> {
        let nx = self.shape()[2];
        self.0.slice_mut(s![.., .., nx - 1])
    }
    fn west_mut(&mut self) -> ArrayViewMut2<Float> {
        self.0.slice_mut(s![.., .., 0])
    }
}

fn pressure(gamma: Float, rho: Float, rhou: Float, rhov: Float, e: Float) -> Float {
    (gamma - 1.0) * (e - (rhou * rhou + rhov * rhov) / (2.0 * rho))
}

#[allow(non_snake_case)]
pub fn RHS_trad(
    op: &dyn SbpOperator2d,
    k: &mut Diff,
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
pub fn RHS_no_SAT(
    op: &dyn SbpOperator2d,
    k: &mut Diff,
    y: &Field,
    metrics: &Metrics,
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

    if let Some(diss_op) = op.upwind() {
        let ad_xi = &mut tmp.4;
        let ad_eta = &mut tmp.5;
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
    } else {
        azip!((out in &mut k.0,
                        eflux in &dE.0,
                        fflux in &dF.0,
                        detj in &metrics.detj().broadcast((4, y.ny(), y.nx())).unwrap()) {
            *out = (-eflux - fflux )/detj
        });
    }
}

#[allow(non_snake_case)]
pub fn RHS_upwind(
    op: &dyn SbpOperator2d,
    k: &mut Diff,
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
    let gamma = *GAMMA.get().expect("GAMMA is not defined");
    for (((FieldValue { rho, rhou, rhov, e }, tmp0), tmp1), metric) in y
        .iter()
        .zip(tmp.0.iter_mut())
        .zip(tmp.1.iter_mut())
        .zip(metrics.iter())
    {
        assert!(rho > 0.0);

        let u = rhou / rho;
        let v = rhov / rho;

        let uhat = metric.detj_dxi_dx * u + metric.detj_dxi_dy * v;
        let vhat = metric.detj_deta_dx * u + metric.detj_deta_dy * v;

        let p = pressure(gamma, rho, rhou, rhov, e);
        assert!(p > 0.0);
        let c = (gamma * p / rho).sqrt();

        // The accurate hypot is very slow, and the accuracy is
        // not that important in this case
        let hypot = |x: Float, y: Float| Float::sqrt(x * x + y * y);

        let alpha_u = uhat.abs() + c * hypot(metric.detj_dxi_dx, metric.detj_dxi_dy);
        let alpha_v = vhat.abs() + c * hypot(metric.detj_deta_dx, metric.detj_deta_dy);

        *tmp0.rho = alpha_u * rho;
        *tmp1.rho = alpha_v * rho;

        *tmp0.rhou = alpha_u * rhou;
        *tmp1.rhou = alpha_v * rhou;

        *tmp0.rhov = alpha_u * rhov;
        *tmp1.rhov = alpha_v * rhov;

        *tmp0.e = alpha_u * e;
        *tmp1.e = alpha_v * e;
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

    let gamma = *GAMMA.get().expect("GAMMA is not defined");

    let mut p = wb.rho_mut();
    azip!((p in &mut p, &rho in &rho, &rhou in &rhou, &rhov in &rhov, &e in &e) {
        *p = pressure(gamma, rho, rhou, rhov, e)
    });

    let (mut c0, c1, mut c2, c3) = k.0.components_mut();
    c0.assign(&rhou);
    azip!((eflux in c1, rho in &rho, rhou in &rhou, p in &p) {
        *eflux = rhou*rhou/rho + p;
    });
    azip!((eflux in c2.view_mut(), rho in &rho, rhou in &rhou, rhov in &rhov) {
        *eflux = rhou*rhov/rho;
    });
    azip!((eflux in c3, rho in &rho, rhou in &rhou, e in &e, p in &p) {
        *eflux = rhou*(e + p)/rho;
    });

    let rhouv = &c2;
    let (mut c0, mut c1, c2, c3) = k.1.components_mut();
    c0.assign(&rhov);
    c1.assign(rhouv);
    azip!((fflux in c2, rho in &rho, rhov in &rhov, p in &p) {
        *fflux = rhov*rhov/rho + p;
    });
    azip!((fflux in c3, rho in &rho, rhov in &rhov, e in &e, p in &p) {
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
    Eval(std::sync::Arc<dyn eval::Evaluator<ndarray::Ix1>>),
    // Vortices(Vec<VortexParameters>),
    Interpolate(usize, Box<dyn InterpolationOperator>),
    MultiGrid(Vec<(usize, usize, usize)>),
}

impl std::fmt::Debug for BoundaryCharacteristic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::This => write!(f, "This"),
            Self::Grid(g) => write!(f, "Grid({})", g),
            Self::Vortex(vp) => write!(f, "{:?}", vp),
            Self::Eval(_) => write!(f, "Eval"),
            Self::Interpolate(_, _) => write!(f, "Interpolate"),
            Self::MultiGrid(m) => write!(f, "Multigrid: {:?}", m),
        }
    }
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
            BoundaryCharacteristic::Eval(_) => todo!(),
            BoundaryCharacteristic::Grid(_)
            | BoundaryCharacteristic::Interpolate(_, _)
            | BoundaryCharacteristic::MultiGrid(_) => panic!("Only working on self grid"),
        },
        south: match &bc.south {
            BoundaryCharacteristic::This => field.north(),
            BoundaryCharacteristic::Vortex(_params) => todo!(),
            BoundaryCharacteristic::Eval(_) => todo!(),
            BoundaryCharacteristic::Grid(_)
            | BoundaryCharacteristic::Interpolate(_, _)
            | BoundaryCharacteristic::MultiGrid(_) => panic!("Only working on self grid"),
        },
        west: match &bc.west {
            BoundaryCharacteristic::This => field.east(),
            BoundaryCharacteristic::Vortex(_params) => todo!(),
            BoundaryCharacteristic::Eval(_) => todo!(),
            BoundaryCharacteristic::Grid(_)
            | BoundaryCharacteristic::Interpolate(_, _)
            | BoundaryCharacteristic::MultiGrid(_) => panic!("Only working on self grid"),
        },
        east: match &bc.east {
            BoundaryCharacteristic::This => field.west(),
            BoundaryCharacteristic::Vortex(_params) => todo!(),
            BoundaryCharacteristic::Eval(_) => todo!(),
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
        BoundaryCharacteristic::Vortex(v) => {
            let field = eb.unwrap();
            vortexify(field.view_mut(), grid, v, time);
            field.view()
        }
        BoundaryCharacteristic::Eval(expr) => {
            let field = eb.unwrap();
            let (x, y) = grid;
            let mut fiter = field.outer_iter_mut();
            let (rho, rhou, rhov, e) = (
                fiter.next().unwrap(),
                fiter.next().unwrap(),
                fiter.next().unwrap(),
                fiter.next().unwrap(),
            );
            expr.evaluate(time, x, y, rho, rhou, rhov, e);
            field.view()
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
                | BoundaryCharacteristic::Eval(_)
                | BoundaryCharacteristic::Interpolate(_, _)
                | BoundaryCharacteristic::MultiGrid(_) => {
                    Some(ndarray::Array2::zeros((4, grid.nx())))
                }
                _ => None,
            },
            south: match bt.south() {
                BoundaryCharacteristic::Vortex(_)
                | BoundaryCharacteristic::Eval(_)
                | BoundaryCharacteristic::Interpolate(_, _)
                | BoundaryCharacteristic::MultiGrid(_) => {
                    Some(ndarray::Array2::zeros((4, grid.nx())))
                }
                _ => None,
            },
            east: match bt.east() {
                BoundaryCharacteristic::Vortex(_)
                | BoundaryCharacteristic::Eval(_)
                | BoundaryCharacteristic::Interpolate(_, _)
                | BoundaryCharacteristic::MultiGrid(_) => {
                    Some(ndarray::Array2::zeros((4, grid.ny())))
                }
                _ => None,
            },
            west: match bt.west() {
                BoundaryCharacteristic::Vortex(_)
                | BoundaryCharacteristic::Eval(_)
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
    vparams.evaluate(time, x, y, rho, rhou, rhov, e)
}

#[allow(non_snake_case)]
/// Boundary conditions (SAT)
pub fn SAT_characteristics(
    op: &dyn SbpOperator2d,
    k: &mut Diff,
    y: &Field,
    metrics: &Metrics,
    boundaries: &BoundaryTerms,
) {
    SAT_north(op, k, y, metrics, boundaries.north);
    SAT_south(op, k, y, metrics, boundaries.south);
    SAT_east(op, k, y, metrics, boundaries.east);
    SAT_west(op, k, y, metrics, boundaries.west);
}

#[allow(non_snake_case)]
pub fn SAT_north(
    op: &dyn SbpOperator2d,
    k: &mut Diff,
    y: &Field,
    metrics: &Metrics,
    boundary: ArrayView2<Float>,
) {
    let ny = y.ny();

    let hi = if op.is_h2eta() {
        (ny - 2) as Float / op.heta()[0]
    } else {
        (ny - 1) as Float / op.heta()[0]
    };
    let sign = -1.0;
    let tau = 1.0;
    let slice = s![y.ny() - 1, ..];
    SAT_characteristic(
        k.north_mut(),
        y.north(),
        boundary,
        hi,
        sign,
        tau,
        metrics.detj().slice(slice),
        metrics.detj_deta_dx().slice(slice),
        metrics.detj_deta_dy().slice(slice),
    );
}

#[allow(non_snake_case)]
pub fn SAT_south(
    op: &dyn SbpOperator2d,
    k: &mut Diff,
    y: &Field,
    metrics: &Metrics,
    boundary: ArrayView2<Float>,
) {
    let ny = y.ny();
    let hi = if op.is_h2eta() {
        (ny - 2) as Float / op.heta()[0]
    } else {
        (ny - 1) as Float / op.heta()[0]
    };
    let sign = 1.0;
    let tau = -1.0;
    let slice = s![0, ..];
    SAT_characteristic(
        k.south_mut(),
        y.south(),
        boundary,
        hi,
        sign,
        tau,
        metrics.detj().slice(slice),
        metrics.detj_deta_dx().slice(slice),
        metrics.detj_deta_dy().slice(slice),
    );
}

#[allow(non_snake_case)]
pub fn SAT_west(
    op: &dyn SbpOperator2d,
    k: &mut Diff,
    y: &Field,
    metrics: &Metrics,
    boundary: ArrayView2<Float>,
) {
    let nx = y.nx();

    let hi = if op.is_h2xi() {
        (nx - 2) as Float / op.hxi()[0]
    } else {
        (nx - 1) as Float / op.hxi()[0]
    };
    let sign = 1.0;
    let tau = -1.0;
    let slice = s![.., 0];
    SAT_characteristic(
        k.west_mut(),
        y.west(),
        boundary,
        hi,
        sign,
        tau,
        metrics.detj().slice(slice),
        metrics.detj_dxi_dx().slice(slice),
        metrics.detj_dxi_dy().slice(slice),
    );
}

#[allow(non_snake_case)]
pub fn SAT_east(
    op: &dyn SbpOperator2d,
    k: &mut Diff,
    y: &Field,
    metrics: &Metrics,
    boundary: ArrayView2<Float>,
) {
    let nx = y.nx();
    let hi = if op.is_h2xi() {
        (nx - 2) as Float / op.hxi()[0]
    } else {
        (nx - 1) as Float / op.hxi()[0]
    };
    let sign = -1.0;
    let tau = 1.0;
    let slice = s![.., y.nx() - 1];
    SAT_characteristic(
        k.east_mut(),
        y.east(),
        boundary,
        hi,
        sign,
        tau,
        metrics.detj().slice(slice),
        metrics.detj_dxi_dx().slice(slice),
        metrics.detj_dxi_dy().slice(slice),
    );
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
    let gamma = *GAMMA.get().expect("GAMMA is not defined");

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

        let p = pressure(gamma, rho, rhou, rhov, e);
        let c = (gamma * p / rho).sqrt();
        let phi2 = (gamma - 1.0) * (u * u + v * v) / 2.0;
        let alpha = rho / (sbp::consts::SQRT_2 * c);

        let phi2_c2 = (phi2 + c * c) / (gamma - 1.0);

        #[rustfmt::skip]
        let T = [
            [                 1.0,                   0.0,                       alpha,                       alpha],
            [                   u,                    ky,          alpha*(u + kx * c),          alpha*(u - kx * c)],
            [                   v,                   -kx,          alpha*(v + ky * c),          alpha*(v - ky * c)],
            [phi2 / (gamma - 1.0), rho*(ky * u - kx * v), alpha*(phi2_c2 + c * theta), alpha*(phi2_c2 - c * theta)],
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
            [     1.0 - phi2 / (c * c),          (gamma - 1.0) * u / (c * c),          (gamma - 1.0) * v / (c * c), -(gamma - 1.0) / (c * c)],
            [   -(ky * u - kx * v)/rho,                               ky/rho,                              -kx/rho,                      0.0],
            [beta * (phi2 - c * theta),  beta * (kx * c - (gamma - 1.0) * u),  beta * (ky * c - (gamma - 1.0) * v),     beta * (gamma - 1.0)],
            [beta * (phi2 + c * theta), -beta * (kx * c + (gamma - 1.0) * u), -beta * (ky * c + (gamma - 1.0) * v),     beta * (gamma - 1.0)],
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
