use super::grid::Grid;
use super::integrate;
use super::operators::{SbpOperator, UpwindOperator};
use super::Float;
use ndarray::azip;
use ndarray::prelude::*;

pub const GAMMA: Float = 1.4;

// A collection of buffers that allows one to efficiently
// move to the next state
#[derive(Debug)]
pub struct System<SBP: SbpOperator> {
    sys: (Field, Field),
    wb: WorkBuffers,
    grid: Grid<SBP>,
}

impl<SBP: SbpOperator> System<SBP> {
    pub fn new(x: ndarray::Array2<Float>, y: ndarray::Array2<Float>) -> Self {
        let grid = Grid::new(x, y).expect(
            "Could not create grid. Different number of elements compared to width*height?",
        );
        let nx = grid.nx();
        let ny = grid.ny();
        Self {
            sys: (Field::new(ny, nx), Field::new(ny, nx)),
            grid,
            wb: WorkBuffers::new(ny, nx),
        }
    }

    pub fn advance(&mut self, dt: Float) {
        let rhs_trad = |k: &mut Field, y: &Field, grid: &_, wb: &mut _| {
            let boundaries = BoundaryTerms {
                north: y.south(),
                south: y.north(),
                west: y.east(),
                east: y.west(),
            };
            RHS_trad(k, y, grid, &boundaries, wb)
        };
        integrate::rk4(
            rhs_trad,
            &self.sys.0,
            &mut self.sys.1,
            dt,
            &self.grid,
            &mut self.wb.k,
            &mut self.wb.tmp,
        );
        std::mem::swap(&mut self.sys.0, &mut self.sys.1);
    }

    pub fn vortex(&mut self, t: Float, vortex_parameters: VortexParameters) {
        self.sys
            .0
            .vortex(self.grid.x.view(), self.grid.y.view(), t, vortex_parameters);
    }

    #[allow(clippy::many_single_char_names)]
    pub fn init_with_vortex(&mut self, x0: Float, y0: Float) {
        // Should parametrise such that we have radius, drop in pressure at center, etc
        let vortex_parameters = VortexParameters {
            x0,
            y0,
            rstar: 1.0,
            eps: 3.0,
            mach: 0.5,
        };

        self.sys.0.vortex(
            self.grid.x.view(),
            self.grid.y.view(),
            0.0,
            vortex_parameters,
        )
    }

    pub fn field(&self) -> &Field {
        &self.sys.0
    }

    pub fn x(&self) -> ArrayView2<Float> {
        self.grid.x.view()
    }
    pub fn y(&self) -> ArrayView2<Float> {
        self.grid.y.view()
    }
}

impl<UO: UpwindOperator> System<UO> {
    pub fn advance_upwind(&mut self, dt: Float) {
        let rhs_upwind = |k: &mut Field, y: &Field, grid: &_, wb: &mut _| {
            let boundaries = BoundaryTerms {
                north: y.south(),
                south: y.north(),
                west: y.east(),
                east: y.west(),
            };
            RHS_upwind(k, y, grid, &boundaries, wb)
        };
        integrate::rk4(
            rhs_upwind,
            &self.sys.0,
            &mut self.sys.1,
            dt,
            &self.grid,
            &mut self.wb.k,
            &mut self.wb.tmp,
        );
        std::mem::swap(&mut self.sys.0, &mut self.sys.1);
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
        let mut iter = self.0.outer_iter_mut();

        let rho = iter.next().unwrap();
        let rhou = iter.next().unwrap();
        let rhov = iter.next().unwrap();
        let e = iter.next().unwrap();
        assert_eq!(iter.next(), None);

        (rho, rhou, rhov, e)
    }

    fn north(&self) -> ArrayView2<Float> {
        self.slice(s![.., self.ny() - 1, ..])
    }
    fn south(&self) -> ArrayView2<Float> {
        self.slice(s![.., 0, ..])
    }
    fn east(&self) -> ArrayView2<Float> {
        self.slice(s![.., .., self.nx() - 1])
    }
    fn west(&self) -> ArrayView2<Float> {
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
        t: Float,
        vortex_param: VortexParameters,
    ) {
        assert_eq!(x.shape(), y.shape());
        assert_eq!(x.shape()[1], self.nx());
        assert_eq!(x.shape()[0], self.ny());

        let (rho, rhou, rhov, e) = self.components_mut();

        let eps = vortex_param.eps;
        let m = vortex_param.mach;
        let rstar = vortex_param.rstar;
        let p_inf = 1.0 / (GAMMA * m * m);
        azip!((rho in rho,
               rhou in rhou,
               rhov in rhov,
               e in e,
               x in x,
               y in y)
        {
            use crate::consts::PI;

            let dx = (x - vortex_param.x0) - t;
            let dy = y - vortex_param.y0;
            let f = (1.0 - (dx*dx + dy*dy))/(rstar*rstar);

            *rho = Float::powf(1.0 - eps*eps*(GAMMA - 1.0)*m*m/(8.0*PI*PI*p_inf*rstar*rstar)*f.exp(), 1.0/(GAMMA - 1.0));
            assert!(*rho > 0.0);
            let p = Float::powf(*rho, GAMMA)*p_inf;
            assert!(p > 0.0);
            let u = 1.0 - eps*dy/(2.0*PI*p_inf.sqrt()*rstar*rstar)*(f/2.0).exp();
            let v =       eps*dx/(2.0*PI*p_inf.sqrt()*rstar*rstar)*(f/2.0).exp();
            *rhou = *rho*u;
            *rhov = *rho*v;
            *e = p/(GAMMA - 1.0) + *rho*(u*u + v*v)/2.0;
        });
    }
}

impl Field {
    /// sqrt((self-other)^T*H*(self-other))
    pub fn h2_err<SBP: SbpOperator>(&self, other: &Self) -> Float {
        assert_eq!(self.nx(), other.nx());
        assert_eq!(self.ny(), other.ny());

        let h = SBP::h();

        // Resulting structure should be
        // serialized(F0 - F1)^T (Hx kron Hy) serialized(F0 - F1)
        //
        // We accomplish this by serializing along x as fastest dimension
        // Since h is diagonal, it can be iterated with the following iterators

        // This chains the h block into the form [h, 1, 1, 1, rev(h)],
        // and multiplies with a factor
        let itermaker = move |n: usize, factor: Float| {
            h.iter()
                .copied()
                .chain(std::iter::repeat(1.0).take(n - 2 * h.len()))
                .chain(h.iter().copied().rev())
                .map(move |x| x * factor)
        };

        let hxiterator = itermaker(self.nx(), 1.0 / (self.nx() - 1) as Float);
        // Repeating to get the form
        // [[hx0, hx1, ..., hxn], [hx0, hx1, ..., hxn], ..., [hx0, hx1, ..., hxn]]
        let hxiterator = hxiterator.into_iter().cycle().take(self.nx() * self.ny());

        let hyiterator = itermaker(self.ny(), 1.0 / (self.ny() - 1) as Float);
        // Repeating to get the form
        // [[hy0, hy0, ..., hy0], [hy1, hy1, ..., hy1], ..., [hym, hym, ..., hym]]
        let hyiterator = hyiterator
            .into_iter()
            .flat_map(|x| std::iter::repeat(x).take(self.nx()));

        let diagiterator = hxiterator.into_iter().zip(hyiterator).cycle();

        diagiterator
            .into_iter()
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

    assert!((field0.h2_err::<super::operators::Upwind4>(&field1).powi(2) - 4.0).abs() < 1e-3);
    assert!((field0.h2_err::<super::operators::Upwind9>(&field1).powi(2) - 4.0).abs() < 1e-3);
    assert!((field0.h2_err::<super::operators::SBP4>(&field1).powi(2) - 4.0).abs() < 1e-3);
    assert!((field0.h2_err::<super::operators::SBP8>(&field1).powi(2) - 4.0).abs() < 1e-3);
}

#[derive(Copy, Clone)]
pub struct VortexParameters {
    pub x0: Float,
    pub y0: Float,
    pub rstar: Float,
    pub eps: Float,
    pub mach: Float,
}

fn pressure(gamma: Float, rho: Float, rhou: Float, rhov: Float, e: Float) -> Float {
    (gamma - 1.0) * (e - (rhou * rhou + rhov * rhov) / (2.0 * rho))
}

#[allow(non_snake_case)]
pub(crate) fn RHS_trad<SBP: SbpOperator>(
    k: &mut Field,
    y: &Field,
    grid: &Grid<SBP>,
    boundaries: &BoundaryTerms,
    tmp: &mut (Field, Field, Field, Field, Field, Field),
) {
    let ehat = &mut tmp.0;
    let fhat = &mut tmp.1;
    fluxes((ehat, fhat), y, grid);
    let dE = &mut tmp.2;
    let dF = &mut tmp.3;

    SBP::diffxi(ehat.rho(), dE.rho_mut());
    SBP::diffxi(ehat.rhou(), dE.rhou_mut());
    SBP::diffxi(ehat.rhov(), dE.rhov_mut());
    SBP::diffxi(ehat.e(), dE.e_mut());

    SBP::diffeta(fhat.rho(), dF.rho_mut());
    SBP::diffeta(fhat.rhou(), dF.rhou_mut());
    SBP::diffeta(fhat.rhov(), dF.rhov_mut());
    SBP::diffeta(fhat.e(), dF.e_mut());

    azip!((out in &mut k.0,
                    eflux in &dE.0,
                    fflux in &dF.0,
                    detj in &grid.detj.broadcast((4, y.ny(), y.nx())).unwrap()) {
        *out = (-eflux - fflux)/detj
    });

    SAT_characteristics(k, y, grid, boundaries);
}

#[allow(non_snake_case)]
pub(crate) fn RHS_upwind<UO: UpwindOperator>(
    k: &mut Field,
    y: &Field,
    grid: &Grid<UO>,
    boundaries: &BoundaryTerms,
    tmp: &mut (Field, Field, Field, Field, Field, Field),
) {
    let ehat = &mut tmp.0;
    let fhat = &mut tmp.1;
    fluxes((ehat, fhat), y, grid);
    let dE = &mut tmp.2;
    let dF = &mut tmp.3;

    UO::diffxi(ehat.rho(), dE.rho_mut());
    UO::diffxi(ehat.rhou(), dE.rhou_mut());
    UO::diffxi(ehat.rhov(), dE.rhov_mut());
    UO::diffxi(ehat.e(), dE.e_mut());

    UO::diffeta(fhat.rho(), dF.rho_mut());
    UO::diffeta(fhat.rhou(), dF.rhou_mut());
    UO::diffeta(fhat.rhov(), dF.rhov_mut());
    UO::diffeta(fhat.e(), dF.e_mut());

    let ad_xi = &mut tmp.4;
    let ad_eta = &mut tmp.5;
    upwind_dissipation((ad_xi, ad_eta), y, grid, (&mut tmp.0, &mut tmp.1));

    azip!((out in &mut k.0,
                    eflux in &dE.0,
                    fflux in &dF.0,
                    ad_xi in &ad_xi.0,
                    ad_eta in &ad_eta.0,
                    detj in &grid.detj.broadcast((4, y.ny(), y.nx())).unwrap()) {
        *out = (-eflux - fflux + ad_xi + ad_eta)/detj
    });

    SAT_characteristics(k, y, grid, boundaries);
}

#[allow(clippy::many_single_char_names)]
fn upwind_dissipation<UO: UpwindOperator>(
    k: (&mut Field, &mut Field),
    y: &Field,
    grid: &Grid<UO>,
    tmp: (&mut Field, &mut Field),
) {
    let n = y.nx() * y.ny();
    let yview = y.view().into_shape((4, n)).unwrap();
    let mut tmp0 = tmp.0.view_mut().into_shape((4, n)).unwrap();
    let mut tmp1 = tmp.1.view_mut().into_shape((4, n)).unwrap();

    for (
        ((((((y, mut tmp0), mut tmp1), detj), detj_dxi_dx), detj_dxi_dy), detj_deta_dx),
        detj_deta_dy,
    ) in yview
        .axis_iter(ndarray::Axis(1))
        .zip(tmp0.axis_iter_mut(ndarray::Axis(1)))
        .zip(tmp1.axis_iter_mut(ndarray::Axis(1)))
        .zip(grid.detj.iter())
        .zip(grid.detj_dxi_dx.iter())
        .zip(grid.detj_dxi_dy.iter())
        .zip(grid.detj_deta_dx.iter())
        .zip(grid.detj_deta_dy.iter())
    {
        let rho = y[0];
        assert!(rho > 0.0);
        let rhou = y[1];
        let rhov = y[2];
        let e = y[3];

        let u = rhou / rho;
        let v = rhov / rho;

        let uhat = detj_dxi_dx / detj * u + detj_dxi_dy / detj * v;
        let vhat = detj_deta_dx / detj * u + detj_deta_dy / detj * v;

        let p = pressure(GAMMA, rho, rhou, rhov, e);
        assert!(p > 0.0);
        let c = (GAMMA * p / rho).sqrt();

        let alpha_u = uhat.abs() + c;
        let alpha_v = vhat.abs() + c;

        tmp0[0] = alpha_u * rho * detj;
        tmp1[0] = alpha_v * rho * detj;

        tmp0[1] = alpha_u * rhou * detj;
        tmp1[1] = alpha_v * rhou * detj;

        tmp0[2] = alpha_u * rhov * detj;
        tmp1[2] = alpha_v * rhov * detj;

        tmp0[3] = alpha_u * e * detj;
        tmp1[3] = alpha_v * e * detj;
    }

    UO::dissxi(tmp.0.rho(), k.0.rho_mut());
    UO::dissxi(tmp.0.rhou(), k.0.rhou_mut());
    UO::dissxi(tmp.0.rhov(), k.0.rhov_mut());
    UO::dissxi(tmp.0.e(), k.0.e_mut());

    UO::disseta(tmp.1.rho(), k.1.rho_mut());
    UO::disseta(tmp.1.rhou(), k.1.rhou_mut());
    UO::disseta(tmp.1.rhov(), k.1.rhov_mut());
    UO::disseta(tmp.1.e(), k.1.e_mut());
}

fn fluxes<SBP: SbpOperator>(k: (&mut Field, &mut Field), y: &Field, grid: &Grid<SBP>) {
    let j_dxi_dx = grid.detj_dxi_dx.view();
    let j_dxi_dy = grid.detj_dxi_dy.view();
    let j_deta_dx = grid.detj_deta_dx.view();
    let j_deta_dy = grid.detj_deta_dy.view();

    let rho = y.rho();
    let rhou = y.rhou();
    let rhov = y.rhov();
    let e = y.e();

    for j in 0..y.ny() {
        for i in 0..y.nx() {
            let rho = rho[(j, i)];
            assert!(rho > 0.0);
            let rhou = rhou[(j, i)];
            let rhov = rhov[(j, i)];
            let e = e[(j, i)];
            let p = pressure(GAMMA, rho, rhou, rhov, e);

            assert!(p > 0.0);

            let ef = [
                rhou,
                rhou * rhou / rho + p,
                rhou * rhov / rho,
                rhou * (e + p) / rho,
            ];
            let ff = [
                rhov,
                rhou * rhov / rho,
                rhov * rhov / rho + p,
                rhov * (e + p) / rho,
            ];

            for comp in 0..4 {
                let eflux = j_dxi_dx[(j, i)] * ef[comp] + j_dxi_dy[(j, i)] * ff[comp];
                let fflux = j_deta_dx[(j, i)] * ef[comp] + j_deta_dy[(j, i)] * ff[comp];

                k.0[(comp, j, i)] = eflux;
                k.1[(comp, j, i)] = fflux;
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct BoundaryTerms<'a> {
    pub north: ArrayView2<'a, Float>,
    pub south: ArrayView2<'a, Float>,
    pub east: ArrayView2<'a, Float>,
    pub west: ArrayView2<'a, Float>,
}

#[allow(non_snake_case)]
/// Boundary conditions (SAT)
fn SAT_characteristics<SBP: SbpOperator>(
    k: &mut Field,
    y: &Field,
    grid: &Grid<SBP>,
    boundaries: &BoundaryTerms,
) {
    // North boundary
    {
        let hi = (k.ny() - 1) as Float * SBP::h()[0];
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
            grid.detj.slice(slice),
            grid.detj_deta_dx.slice(slice),
            grid.detj_deta_dy.slice(slice),
        );
    }
    // South boundary
    {
        let hi = (k.ny() - 1) as Float * SBP::h()[0];
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
            grid.detj.slice(slice),
            grid.detj_deta_dx.slice(slice),
            grid.detj_deta_dy.slice(slice),
        );
    }
    // West Boundary
    {
        let hi = (k.nx() - 1) as Float * SBP::h()[0];
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
            grid.detj.slice(slice),
            grid.detj_dxi_dx.slice(slice),
            grid.detj_dxi_dy.slice(slice),
        );
    }
    // East Boundary
    {
        let hi = (k.nx() - 1) as Float * SBP::h()[0];
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
            grid.detj.slice(slice),
            grid.detj_dxi_dx.slice(slice),
            grid.detj_dxi_dy.slice(slice),
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

        let phi2_c2 = (phi2 + c * c) / (GAMMA - 1.0);

        let T = [
            [1.0, 0.0, 1.0, 1.0],
            [u, ky, u + kx * c, u - kx * c],
            [v, -kx, v + ky * c, v - ky * c],
            [
                phi2 / (GAMMA - 1.0),
                ky * u - kx * v,
                phi2_c2 + c * theta,
                phi2_c2 - c * theta,
            ],
        ];
        let U = kx_ * u + ky_ * v;
        let L = [
            U,
            U,
            U + c * Float::hypot(kx_, ky_),
            U - c * Float::hypot(kx_, ky_),
        ];
        let beta = 1.0 / (2.0 * c * c);
        let TI = [
            [
                1.0 - phi2 / (c * c),
                (GAMMA - 1.0) * u / (c * c),
                (GAMMA - 1.0) * v / (c * c),
                -(GAMMA - 1.0) / (c * c),
            ],
            [-(ky * u - kx * v), ky, -kx, 0.0],
            [
                beta * (phi2 - c * theta),
                beta * (kx * c - (GAMMA - 1.0) * u),
                beta * (ky * c - (GAMMA - 1.0) * v),
                beta * (GAMMA - 1.0),
            ],
            [
                beta * (phi2 + c * theta),
                -beta * (kx * c + (GAMMA - 1.0) * u),
                -beta * (ky * c + (GAMMA - 1.0) * v),
                beta * (GAMMA - 1.0),
            ],
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
pub struct WorkBuffers {
    k: [Field; 4],
    tmp: (Field, Field, Field, Field, Field, Field),
}

impl WorkBuffers {
    pub fn new(nx: usize, ny: usize) -> Self {
        let arr3 = Field::new(nx, ny);
        Self {
            k: [arr3.clone(), arr3.clone(), arr3.clone(), arr3.clone()],
            tmp: (
                arr3.clone(),
                arr3.clone(),
                arr3.clone(),
                arr3.clone(),
                arr3.clone(),
                arr3,
            ),
        }
    }
}
