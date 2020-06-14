use sbp::operators;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct MaxwellUniverse(maxwell::System<operators::Upwind4>);

#[wasm_bindgen]
impl MaxwellUniverse {
    #[wasm_bindgen(constructor)]
    pub fn new(height: usize, width: usize, x: &[f32], y: &[f32]) -> Self {
        let x = ndarray::Array2::from_shape_vec((height, width), x.to_vec()).unwrap();
        let y = ndarray::Array2::from_shape_vec((height, width), y.to_vec()).unwrap();
        Self(maxwell::System::new(x, y, operators::Upwind4))
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
        self.0.field().ex().as_ptr() as *const u8
    }

    pub fn get_ey_ptr(&self) -> *const u8 {
        self.0.field().ey().as_ptr() as *const u8
    }

    pub fn get_hz_ptr(&self) -> *const u8 {
        self.0.field().hz().as_ptr() as *const u8
    }
}
