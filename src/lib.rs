use wasm_bindgen::prelude::*;

mod euler;
mod grid;
mod maxwell;
pub mod operators;
pub use crate::maxwell::{Field, WorkBuffers};
pub(crate) use grid::Grid;

#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
pub fn set_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct Universe(System<operators::Upwind4>);

#[wasm_bindgen]
impl Universe {
    #[wasm_bindgen(constructor)]
    pub fn new(width: usize, height: usize, x: &[f32], y: &[f32]) -> Self {
        Self(System::new(width as usize, height as usize, x, y))
    }

    pub fn init(&mut self, x0: f32, y0: f32) {
        self.0.set_gaussian(x0, y0);
    }

    pub fn advance(&mut self, dt: f32) {
        self.0.advance(dt)
    }

    pub fn advance_upwind(&mut self, dt: f32) {
        self.0.advance_upwind(dt)
    }

    pub fn get_ex_ptr(&self) -> *const u8 {
        self.0.sys.0.ex().as_ptr() as *const u8
    }

    pub fn get_ey_ptr(&self) -> *const u8 {
        self.0.sys.0.ey().as_ptr() as *const u8
    }

    pub fn get_hz_ptr(&self) -> *const u8 {
        self.0.sys.0.hz().as_ptr() as *const u8
    }
}

pub struct System<SBP: operators::SbpOperator> {
    sys: (Field, Field),
    wb: WorkBuffers,
    grid: Grid<SBP>,
}

impl<SBP: operators::SbpOperator> System<SBP> {
    pub fn new(width: usize, height: usize, x: &[f32], y: &[f32]) -> Self {
        assert_eq!((width * height), x.len());
        assert_eq!((width * height), y.len());

        let grid = Grid::new_from_slice(height, width, x, y).expect(
            "Could not create grid. Different number of elements compared to width*height?",
        );
        Self {
            sys: (Field::new(width, height), Field::new(width, height)),
            grid,
            wb: WorkBuffers::new(width, height),
        }
    }

    pub fn set_gaussian(&mut self, x0: f32, y0: f32) {
        let (ex, hz, ey) = self.sys.0.components_mut();
        ndarray::azip!(
            (ex in ex, hz in hz, ey in ey,
             &x in &self.grid.x, &y in &self.grid.y)
        {
            *ex = 0.0;
            *ey = 0.0;
            *hz = gaussian(x, x0, y, y0)/32.0;
        });
    }

    pub fn advance(&mut self, dt: f32) {
        maxwell::advance(
            &self.sys.0,
            &mut self.sys.1,
            dt,
            &self.grid,
            Some(&mut self.wb),
        );
        std::mem::swap(&mut self.sys.0, &mut self.sys.1);
    }
}

impl<UO: operators::UpwindOperator> System<UO> {
    /// Using artificial dissipation with the upwind operator
    pub fn advance_upwind(&mut self, dt: f32) {
        maxwell::advance_upwind(
            &self.sys.0,
            &mut self.sys.1,
            dt,
            &self.grid,
            Some(&mut self.wb),
        );
        std::mem::swap(&mut self.sys.0, &mut self.sys.1);
    }
}

fn gaussian(x: f32, x0: f32, y: f32, y0: f32) -> f32 {
    use std::f32;
    let x = x - x0;
    let y = y - y0;

    let sigma = 0.05;

    1.0 / (2.0 * f32::consts::PI * sigma * sigma) * (-(x * x + y * y) / (2.0 * sigma * sigma)).exp()
}

pub struct EulerSystem<SBP: operators::SbpOperator> {
    sys: (euler::Field, euler::Field),
    wb: euler::WorkBuffers,
    grid: Grid<SBP>,
}

#[wasm_bindgen]
pub struct EulerUniverse(EulerSystem<operators::Upwind4>);

impl EulerUniverse {
    pub fn new(x: ndarray::Array2<f32>, y: ndarray::Array2<f32>) -> Self {
        Self(EulerSystem::new(x, y))
    }
}

#[wasm_bindgen]
impl EulerUniverse {
    #[wasm_bindgen(constructor)]
    pub fn new_with_slice(height: usize, width: usize, x: &[f32], y: &[f32]) -> Self {
        let x = ndarray::Array2::from_shape_vec((height, width), x.to_vec()).unwrap();
        let y = ndarray::Array2::from_shape_vec((height, width), y.to_vec()).unwrap();
        Self(EulerSystem::new(x, y))
    }

    pub fn init(&mut self, x0: f32, y0: f32) {
        self.0.vortex(x0, y0)
    }

    pub fn advance(&mut self, dt: f32) {
        self.0.advance(dt)
    }

    pub fn advance_upwind(&mut self, dt: f32) {
        self.0.advance_upwind(dt)
    }

    pub fn get_rho_ptr(&self) -> *const u8 {
        self.0.sys.0.rho().as_ptr() as *const u8
    }
    pub fn get_rhou_ptr(&self) -> *const u8 {
        self.0.sys.0.rhou().as_ptr() as *const u8
    }
    pub fn get_rhov_ptr(&self) -> *const u8 {
        self.0.sys.0.rhov().as_ptr() as *const u8
    }
    pub fn get_e_ptr(&self) -> *const u8 {
        self.0.sys.0.e().as_ptr() as *const u8
    }
}

impl<SBP: operators::SbpOperator> EulerSystem<SBP> {
    pub fn new(x: ndarray::Array2<f32>, y: ndarray::Array2<f32>) -> Self {
        let grid = Grid::new(x, y).expect(
            "Could not create grid. Different number of elements compared to width*height?",
        );
        let nx = grid.nx();
        let ny = grid.ny();
        Self {
            sys: (euler::Field::new(ny, nx), euler::Field::new(ny, nx)),
            grid,
            wb: euler::WorkBuffers::new(ny, nx),
        }
    }

    pub fn advance(&mut self, dt: f32) {
        euler::advance(
            &self.sys.0,
            &mut self.sys.1,
            dt,
            &self.grid,
            Some(&mut self.wb),
        );
        std::mem::swap(&mut self.sys.0, &mut self.sys.1);
    }

    pub fn vortex(&mut self, x0: f32, y0: f32) {
        // Should parametrise such that we have radius, drop in pressure at center, etc
        let rstar = 1.0;
        let eps = 3.0;
        #[allow(non_snake_case)]
        let M = 0.5;

        let p_inf = 1.0 / (euler::GAMMA * M * M);
        let t = 0.0;

        let nx = self.grid.nx();
        let ny = self.grid.ny();

        for j in 0..ny {
            for i in 0..nx {
                let x = self.grid.x[(j, i)];
                let y = self.grid.y[(j, i)];

                let dx = (x - x0) - t;
                let dy = y - y0;
                let f = (1.0 - (dx * dx + dy * dy)) / (rstar * rstar);

                use euler::GAMMA;
                use std::f32::consts::PI;
                let u =
                    1.0 - eps * dy / (2.0 * PI * p_inf.sqrt() * rstar * rstar) * (f / 2.0).exp();
                let v =
                    0.0 + eps * dx / (2.0 * PI * p_inf.sqrt() * rstar * rstar) * (f / 2.0).exp();
                let rho = f32::powf(
                    1.0 - eps * eps * (GAMMA - 1.0) * M * M
                        / (8.0 * PI * PI * p_inf * rstar * rstar)
                        * f.exp(),
                    1.0 / (GAMMA - 1.0),
                );
                assert!(rho > 0.0);
                let p = p_inf * rho.powf(GAMMA);
                assert!(p > 0.0);
                let e = p / (GAMMA - 1.0) + rho * (u * u + v * v) / 2.0;
                assert!(e > 0.0);

                self.sys.0[(0, j, i)] = rho;
                self.sys.0[(1, j, i)] = rho * u;
                self.sys.0[(2, j, i)] = rho * v;
                self.sys.0[(3, j, i)] = e;
            }
        }
    }
}

impl<SBP: operators::UpwindOperator> EulerSystem<SBP> {
    pub fn advance_upwind(&mut self, dt: f32) {
        euler::advance_upwind(
            &self.sys.0,
            &mut self.sys.1,
            dt,
            &self.grid,
            Some(&mut self.wb),
        );
        std::mem::swap(&mut self.sys.0, &mut self.sys.1);
    }
}

#[test]
fn start_and_advance_euler() {
    let x = ndarray::Array2::from_shape_fn((20, 20), |(_j, i)| {
        5.0 * 2.0 * ((i as f32 / (20 - 1) as f32) - 0.5)
    });
    let y = ndarray::Array2::from_shape_fn((20, 20), |(j, _i)| {
        5.0 * 2.0 * ((j as f32 / (20 - 1) as f32) - 0.5)
    });
    let mut universe = EulerUniverse::new(x, y);
    universe.init(-1.0, 0.0);
    for _ in 0..50 {
        universe.advance(0.01);
    }
}

#[test]
fn start_and_advance_upwind_euler() {
    let x = ndarray::Array2::from_shape_fn((20, 10), |(_j, i)| i as f32 / (10 - 1) as f32);
    let y = ndarray::Array2::from_shape_fn((20, 10), |(j, _i)| j as f32 / (20 - 1) as f32);
    let mut universe = EulerUniverse::new(x, y);
    universe.init(0.5, 0.5);
    for _ in 0..50 {
        universe.advance_upwind(0.01);
    }
}
