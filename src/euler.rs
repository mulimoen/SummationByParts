use super::operators::{SbpOperator, UpwindOperator};
use super::Grid;
use ndarray::prelude::*;
use ndarray::{azip, Zip};

pub const GAMMA: f32 = 1.4;

#[derive(Clone, Debug)]
/// A 4 x ny x nx array
pub struct Field(pub(crate) Array3<f32>);

impl std::ops::Deref for Field {
    type Target = Array3<f32>;
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

    pub fn rho(&self) -> ArrayView2<f32> {
        self.slice(s![0, .., ..])
    }
    pub fn rhou(&self) -> ArrayView2<f32> {
        self.slice(s![1, .., ..])
    }
    pub fn rhov(&self) -> ArrayView2<f32> {
        self.slice(s![2, .., ..])
    }
    pub fn e(&self) -> ArrayView2<f32> {
        self.slice(s![3, .., ..])
    }

    pub fn rho_mut(&mut self) -> ArrayViewMut2<f32> {
        self.slice_mut(s![0, .., ..])
    }
    pub fn rhou_mut(&mut self) -> ArrayViewMut2<f32> {
        self.slice_mut(s![1, .., ..])
    }
    pub fn rhov_mut(&mut self) -> ArrayViewMut2<f32> {
        self.slice_mut(s![2, .., ..])
    }
    pub fn e_mut(&mut self) -> ArrayViewMut2<f32> {
        self.slice_mut(s![3, .., ..])
    }

    #[allow(unused)]
    pub fn components(
        &self,
    ) -> (
        ArrayView2<f32>,
        ArrayView2<f32>,
        ArrayView2<f32>,
        ArrayView2<f32>,
    ) {
        (self.rho(), self.rhou(), self.rhov(), self.e())
    }
    #[allow(unused)]
    pub fn components_mut(
        &mut self,
    ) -> (
        ArrayViewMut2<f32>,
        ArrayViewMut2<f32>,
        ArrayViewMut2<f32>,
        ArrayViewMut2<f32>,
    ) {
        let mut iter = self.0.outer_iter_mut();

        let rho = iter.next().unwrap();
        let rhou = iter.next().unwrap();
        let rhov = iter.next().unwrap();
        let e = iter.next().unwrap();
        assert_eq!(iter.next(), None);

        (rho, rhou, rhov, e)
    }

    fn north(&self) -> ArrayView2<f32> {
        self.slice(s![.., self.ny() - 1, ..])
    }
    fn south(&self) -> ArrayView2<f32> {
        self.slice(s![.., 0, ..])
    }
    fn east(&self) -> ArrayView2<f32> {
        self.slice(s![.., .., self.nx() - 1])
    }
    fn west(&self) -> ArrayView2<f32> {
        self.slice(s![.., .., 0])
    }
    fn north_mut(&mut self) -> ArrayViewMut2<f32> {
        let ny = self.ny();
        self.slice_mut(s![.., ny - 1, ..])
    }
    fn south_mut(&mut self) -> ArrayViewMut2<f32> {
        self.slice_mut(s![.., 0, ..])
    }
    fn east_mut(&mut self) -> ArrayViewMut2<f32> {
        let nx = self.nx();
        self.slice_mut(s![.., .., nx - 1])
    }
    fn west_mut(&mut self) -> ArrayViewMut2<f32> {
        self.slice_mut(s![.., .., 0])
    }
}

#[allow(unused)]
pub(crate) fn advance_upwind<UO>(
    prev: &Field,
    fut: &mut Field,
    dt: f32,
    grid: &Grid<UO>,
    work_buffers: Option<&mut WorkBuffers>,
) where
    UO: UpwindOperator,
{
    assert_eq!(prev.0.shape(), fut.0.shape());

    let mut wb: WorkBuffers;
    let (y, k, tmp) = if let Some(x) = work_buffers {
        (&mut x.y, &mut x.buf, &mut x.tmp)
    } else {
        wb = WorkBuffers::new(prev.nx(), prev.ny());
        (&mut wb.y, &mut wb.buf, &mut wb.tmp)
    };

    let boundaries = BoundaryTerms {
        north: Boundary::This,
        south: Boundary::This,
        west: Boundary::This,
        east: Boundary::This,
    };

    for i in 0..4 {
        // y = y0 + c*kn
        y.assign(&prev);
        match i {
            0 => {}
            1 | 2 => {
                y.scaled_add(1.0 / 2.0 * dt, &k[i - 1]);
            }
            3 => {
                y.scaled_add(dt, &k[i - 1]);
            }
            _ => {
                unreachable!();
            }
        };

        RHS_upwind(&mut k[i], &y, grid, &boundaries, tmp);
    }

    Zip::from(&mut fut.0)
        .and(&prev.0)
        .and(&*k[0])
        .and(&*k[1])
        .and(&*k[2])
        .and(&*k[3])
        .apply(|y1, &y0, &k1, &k2, &k3, &k4| *y1 = y0 + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4));
}

/// Solving (Au)_x + (Bu)_y
/// with:
///        A               B
///  [ 0,  0,  0]    [ 0,  1,  0]
///  [ 0,  0, -1]    [ 1,  0,  0]
///  [ 0, -1,  0]    [ 0,  0,  0]
pub(crate) fn advance<SBP>(
    prev: &Field,
    fut: &mut Field,
    dt: f32,
    grid: &Grid<SBP>,
    work_buffers: Option<&mut WorkBuffers>,
) where
    SBP: SbpOperator,
{
    assert_eq!(prev.0.shape(), fut.0.shape());

    let mut wb: WorkBuffers;
    let (y, k, tmp) = if let Some(x) = work_buffers {
        (&mut x.y, &mut x.buf, &mut x.tmp)
    } else {
        wb = WorkBuffers::new(prev.nx(), prev.ny());
        (&mut wb.y, &mut wb.buf, &mut wb.tmp)
    };

    let boundaries = BoundaryTerms {
        north: Boundary::This,
        south: Boundary::This,
        west: Boundary::This,
        east: Boundary::This,
    };

    for i in 0..4 {
        // y = y0 + c*kn
        y.assign(&prev);
        match i {
            0 => {}
            1 | 2 => {
                y.scaled_add(1.0 / 2.0 * dt, &k[i - 1]);
            }
            3 => {
                y.scaled_add(dt, &k[i - 1]);
            }
            _ => {
                unreachable!();
            }
        };

        RHS(&mut k[i], &y, grid, &boundaries, tmp);
    }

    Zip::from(&mut fut.0)
        .and(&prev.0)
        .and(&*k[0])
        .and(&*k[1])
        .and(&*k[2])
        .and(&*k[3])
        .apply(|y1, &y0, &k1, &k2, &k3, &k4| *y1 = y0 + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4));
}

fn pressure(gamma: f32, rho: f32, rhou: f32, rhov: f32, e: f32) -> f32 {
    (gamma - 1.0) * (e - (rhou * rhou + rhov * rhov) / (2.0 * rho))
}

#[allow(non_snake_case)]
/// This flux is rotated by the grid metrics
/// (Au)_x + (Bu)_y = 1/J [
///          (J xi_x Au)_xi + (J eta_x Au)_eta
///          (J xi_y Bu)_xi + (J eta_y Bu)_eta
///      ]
/// where J is the grid determinant
///
/// This is used both in fluxes and SAT terms
fn RHS<SBP: SbpOperator>(
    k: &mut Field,
    y: &Field,
    grid: &Grid<SBP>,
    boundaries: &BoundaryTerms,
    tmp: &mut (Field, Field, Field, Field),
) {
    let ehat = &mut tmp.0;
    let fhat = &mut tmp.1;
    fluxes([ehat, fhat], y, grid);
    let de = &mut tmp.2;
    let df = &mut tmp.3;

    SBP::diffxi(ehat.rho(), de.rho_mut());
    SBP::diffxi(ehat.rhou(), de.rhou_mut());
    SBP::diffxi(ehat.rhov(), de.rhov_mut());
    SBP::diffxi(ehat.e(), de.e_mut());

    SBP::diffeta(fhat.rho(), df.rho_mut());
    SBP::diffeta(fhat.rhou(), df.rhou_mut());
    SBP::diffeta(fhat.rhov(), df.rhov_mut());
    SBP::diffeta(fhat.e(), df.e_mut());

    // And dissipation...

    ndarray::azip!((out in &mut k.0,
                    eflux in &de.0,
                    fflux in &df.0,
                    detj in &grid.detj.broadcast((4, y.ny(), y.nx())).unwrap()) {
        *out = (-eflux - fflux)/detj
    });

    SAT_characteristics(k, y, grid, boundaries);
}

#[allow(non_snake_case)]
#[allow(unused)]
fn RHS_upwind<UO: UpwindOperator>(
    k: &mut Field,
    y: &Field,
    grid: &Grid<UO>,
    boundaries: &BoundaryTerms,
    tmp: &mut (Field, Field, Field, Field),
) {
    // fluxes(k, y, grid, tmp);
    // dissipation(k, y, grid, tmp);

    SAT_characteristics(k, y, grid, boundaries);

    azip!((k in &mut k.0,
                    &detj in &grid.detj.broadcast((3, y.ny(), y.nx())).unwrap()) {
        *k /= detj;
    });
}

fn fluxes<SBP: SbpOperator>(k: [&mut Field; 2], y: &Field, grid: &Grid<SBP>) {
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
            let rhou = rhou[(j, i)];
            let rhov = rhov[(j, i)];
            let e = e[(j, i)];
            let p = pressure(GAMMA, rho, rhou, rhov, e);

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

                k[0][(comp, j, i)] = eflux;
                k[1][(comp, j, i)] = fflux;
            }
        }
    }
}

#[allow(unused)]
fn dissipation<UO: UpwindOperator>(
    k: &mut Field,
    y: &Field,
    grid: &Grid<UO>,
    tmp: &mut (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>),
) {
    todo!()
}

#[derive(Clone, Debug)]
pub enum Boundary {
    This,
}

#[derive(Clone, Debug)]
pub struct BoundaryTerms {
    pub north: Boundary,
    pub south: Boundary,
    pub east: Boundary,
    pub west: Boundary,
}

#[allow(non_snake_case)]
/// Boundary conditions (SAT)
fn SAT_characteristics<SBP: SbpOperator>(
    k: &mut Field,
    y: &Field,
    grid: &Grid<SBP>,
    _boundaries: &BoundaryTerms,
) {
    // North boundary
    {
        let hi = (k.ny() - 1) as f32 * SBP::h()[0];
        let sign = -1.0;
        let tau = 1.0;
        let slice = s![y.nx() - 1, ..];
        SAT_characteristic(
            k.north_mut(),
            y.north(),
            y.south(), // Self South
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
        let hi = (k.ny() - 1) as f32 * SBP::h()[0];
        let sign = 1.0;
        let tau = -1.0;
        let slice = s![0, ..];
        SAT_characteristic(
            k.south_mut(),
            y.south(),
            y.north(), // Self North
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
        let hi = (k.nx() - 1) as f32 * SBP::h()[0];
        let sign = 1.0;
        let tau = -1.0;
        let slice = s![.., 0];
        println!("{:?}", slice);
        SAT_characteristic(
            k.west_mut(),
            y.west(),
            y.east(), // Self East
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
        let hi = (k.nx() - 1) as f32 * SBP::h()[0];
        let sign = -1.0;
        let tau = 1.0;
        let slice = s![.., y.nx() - 1];
        SAT_characteristic(
            k.east_mut(),
            y.east(),
            y.west(), // Self West
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
/// Boundary conditions (SAT)
fn SAT_characteristic(
    mut k: ArrayViewMut2<f32>,
    y: ArrayView2<f32>,
    z: ArrayView2<f32>, // Size 4 x n (all components in line)
    hi: f32,
    sign: f32,
    tau: f32,
    detj: ArrayView1<f32>,
    detj_d_dx: ArrayView1<f32>,
    detj_d_dy: ArrayView1<f32>,
) {
    assert_eq!(detj.shape(), detj_d_dx.shape());
    assert_eq!(detj.shape(), detj_d_dy.shape());
    assert_eq!(y.shape(), z.shape());
    assert_eq!(y.shape()[0], 4);
    assert_eq!(y.shape()[1], detj.shape()[0]);

    for i in 0..z.shape()[1] {
        let rho = y[(0, i)];
        let rhou = y[(1, i)];
        let rhov = y[(2, i)];
        let e = y[(3, i)];

        let kx_ = detj_d_dx[i] / detj[i];
        let ky_ = detj_d_dy[i] / detj[i];

        let (kx, ky) = {
            let r = f32::hypot(kx_, ky_);
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
            U + c * f32::hypot(kx_, ky_),
            U - c * f32::hypot(kx_, ky_),
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

        let res = [
            rho - z[(0, i)],
            rhou - z[(1, i)],
            rhov - z[(2, i)],
            e - z[(3, i)],
        ];
        let mut TIres = [0.0; 4];
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
        for row in 0..4 {
            for col in 0..4 {
                TLTIres[row] += T[row][col] * LTIres[col];
            }
        }

        for comp in 0..4 {
            k[(comp, i)] += hi * tau * TLTIres[comp];
        }
    }
}

pub struct WorkBuffers {
    y: Field,
    buf: [Field; 4],
    tmp: (Field, Field, Field, Field),
}

impl WorkBuffers {
    pub fn new(nx: usize, ny: usize) -> Self {
        let arr3 = Field::new(nx, ny);
        Self {
            y: arr3.clone(),
            buf: [arr3.clone(), arr3.clone(), arr3.clone(), arr3.clone()],
            tmp: (arr3.clone(), arr3.clone(), arr3.clone(), arr3),
        }
    }
}
