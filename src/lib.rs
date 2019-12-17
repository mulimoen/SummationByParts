use wasm_bindgen::prelude::*;

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

        let grid = Grid::new(width, height, x, y).expect(
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
