use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct ShallowWaterUniverse(shallow_water::System);

#[wasm_bindgen]
impl ShallowWaterUniverse {
    #[wasm_bindgen(constructor)]
    pub fn new(height: usize, width: usize) -> Self {
        let x = (-0.5, 0.5, width);
        let y = (-0.5, 0.5, height);
        Self(shallow_water::System::new(x, y))
    }

    pub fn init(&mut self, x0: f32, y0: f32) {
        let nx = self.0.nx();
        let ny = self.0.ny();
        let x = ndarray::Array1::linspace(-0.5, 0.5, nx);
        let y = ndarray::Array1::linspace(-0.5, 0.5, ny);

        let (mut eta, mut etau, mut etav) = self.0.components_mut();

        let sigma = 0.1;

        for j in 0..ny {
            let y = y[j];
            for i in 0..nx {
                let x = x[i];
                let r = f32::hypot(x - x0, y - y0);

                let f = 1.0 / (sigma * (2.0 * sbp::consts::PI).sqrt())
                    * f32::exp(-0.5 * (r / sigma).powi(2));
                eta[(j, i)] = 1.0 - 0.1 * f;
                etau[(j, i)] = eta[(j, i)] * 0.0;
                etav[(j, i)] = eta[(j, i)] * 0.0;
            }
        }
    }

    pub fn advance(&mut self) {
        self.0.advance();
    }

    pub fn get_eta_ptr(&self) -> *const u8 {
        self.0.eta().as_ptr().cast()
    }

    pub fn get_etau_ptr(&self) -> *const u8 {
        self.0.etau().as_ptr().cast()
    }

    pub fn get_etav_ptr(&self) -> *const u8 {
        self.0.etav().as_ptr().cast()
    }
}
