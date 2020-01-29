use wasm_bindgen::prelude::*;

pub mod euler;
mod grid;
pub(crate) mod integrate;
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
pub struct MaxwellUniverse(MaxwellSystem<operators::Upwind4>);

#[wasm_bindgen]
impl MaxwellUniverse {
    #[wasm_bindgen(constructor)]
    pub fn new(width: usize, height: usize, x: &[f32], y: &[f32]) -> Self {
        Self(MaxwellSystem::new(width as usize, height as usize, x, y))
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

pub struct MaxwellSystem<SBP: operators::SbpOperator> {
    sys: (Field, Field),
    wb: WorkBuffers,
    grid: Grid<SBP>,
}

impl<SBP: operators::SbpOperator> MaxwellSystem<SBP> {
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

impl<UO: operators::UpwindOperator> MaxwellSystem<UO> {
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

#[wasm_bindgen]
pub struct EulerUniverse(euler::System<operators::Upwind4>);

impl EulerUniverse {
    pub fn new(x: ndarray::Array2<f32>, y: ndarray::Array2<f32>) -> Self {
        Self(euler::System::new(x, y))
    }
}

#[wasm_bindgen]
impl EulerUniverse {
    #[wasm_bindgen(constructor)]
    pub fn new_with_slice(height: usize, width: usize, x: &[f32], y: &[f32]) -> Self {
        let x = ndarray::Array2::from_shape_vec((height, width), x.to_vec()).unwrap();
        let y = ndarray::Array2::from_shape_vec((height, width), y.to_vec()).unwrap();
        Self(euler::System::new(x, y))
    }

    pub fn init(&mut self, x0: f32, y0: f32) {
        self.0.init_with_vortex(x0, y0)
    }

    pub fn advance(&mut self, dt: f32) {
        self.0.advance(dt)
    }

    pub fn advance_upwind(&mut self, dt: f32) {
        self.0.advance_upwind(dt)
    }

    pub fn get_rho_ptr(&self) -> *const u8 {
        self.0.field().rho().as_ptr() as *const u8
    }
    pub fn get_rhou_ptr(&self) -> *const u8 {
        self.0.field().rhou().as_ptr() as *const u8
    }
    pub fn get_rhov_ptr(&self) -> *const u8 {
        self.0.field().rhov().as_ptr() as *const u8
    }
    pub fn get_e_ptr(&self) -> *const u8 {
        self.0.field().e().as_ptr() as *const u8
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
