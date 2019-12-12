use wasm_bindgen::prelude::*;

mod grid;
mod maxwell;
mod operators;
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
pub struct Universe {
    sys: (Field, Field),
    wb: WorkBuffers,
    grid: Grid<operators::Upwind4>,
}

#[wasm_bindgen]
impl Universe {
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32, x: &[f32], y: &[f32]) -> Self {
        assert_eq!((width * height) as usize, x.len());
        assert_eq!((width * height) as usize, y.len());

        let grid = Grid::new(width, height, x, y).expect(
            "Could not create grid. Different number of elements compared to width*height?",
        );
        Self {
            sys: (
                Field::new(width as usize, height as usize),
                Field::new(width as usize, height as usize),
            ),
            grid,
            wb: WorkBuffers::new(width as usize, height as usize),
        }
    }

    pub fn init(&mut self, x0: f32, y0: f32) {
        self.sys.0.set_gaussian(x0, y0);
    }

    /// Using artifical dissipation with the upwind operator
    pub fn advance_upwind(&mut self, dt: f32) {
        Field::advance_upwind::<operators::Upwind4>(
            &self.sys.0,
            &mut self.sys.1,
            dt,
            &self.grid,
            Some(&mut self.wb),
        );
        std::mem::swap(&mut self.sys.0, &mut self.sys.1);
    }

    pub fn advance(&mut self, dt: f32) {
        Field::advance::<operators::Upwind4>(
            &self.sys.0,
            &mut self.sys.1,
            dt,
            &self.grid,
            Some(&mut self.wb),
        );
        std::mem::swap(&mut self.sys.0, &mut self.sys.1);
    }

    pub fn get_ex_ptr(&self) -> *const u8 {
        self.sys.0.ex().as_ptr() as *const u8
    }

    pub fn get_ey_ptr(&self) -> *const u8 {
        self.sys.0.ey().as_ptr() as *const u8
    }

    pub fn get_hz_ptr(&self) -> *const u8 {
        self.sys.0.hz().as_ptr() as *const u8
    }
}
