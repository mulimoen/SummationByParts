use wasm_bindgen::prelude::*;

use euler;
use maxwell;
use sbp::operators;

#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
pub fn set_panic_hook() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub fn set_console_logger() {
    use std::sync::Once;
    static LOG_INIT: Once = Once::new();

    LOG_INIT.call_once(|| console_log::init_with_level(log::Level::Trace).unwrap());
}

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

#[wasm_bindgen]
pub struct EulerUniverse(euler::System<operators::Upwind4>);

impl EulerUniverse {
    pub fn new(x: ndarray::Array2<f32>, y: ndarray::Array2<f32>) -> Self {
        Self(euler::System::new(x, y, operators::Upwind4))
    }
}

#[wasm_bindgen]
impl EulerUniverse {
    #[wasm_bindgen(constructor)]
    pub fn new_with_slice(height: usize, width: usize, x: &[f32], y: &[f32]) -> Self {
        let x = ndarray::Array2::from_shape_vec((height, width), x.to_vec()).unwrap();
        let y = ndarray::Array2::from_shape_vec((height, width), y.to_vec()).unwrap();
        Self(euler::System::new(x, y, operators::Upwind4))
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

#[wasm_bindgen]
pub struct ShallowWaterUniverse(shallow_water::System);

#[wasm_bindgen]
impl ShallowWaterUniverse {
    #[wasm_bindgen(constructor)]
    pub fn new(height: usize, width: usize) -> Self {
        let x = (0.0, 1.0, width);
        let y = (0.0, 1.0, height);
        Self(shallow_water::System::new(x, y))
    }

    pub fn init(&mut self, x0: f32, y0: f32) {
        let nx = self.0.nx();
        let ny = self.0.ny();
        let x = ndarray::Array1::linspace(0.0, 1.0, nx);
        let y = ndarray::Array1::linspace(0.0, 1.0, ny);

        let (mut eta, mut etau, mut etav) = self.0.components_mut();

        let sigma = 0.1;

        for j in 0..ny {
            for i in 0..nx {
                let r = f32::hypot(x[i] - x0, y[j] - y0);

                let f = 1.0 / (sigma * (2.0 * sbp::consts::PI).sqrt())
                    * f32::exp(-0.5 * (r / sigma).powi(2));
                eta[(j, i)] = 1.0 - 0.1 * f;
                etau[(j, i)] = eta[(j, i)] * 0.0;
                etav[(j, i)] = eta[(j, i)] * 0.0;
            }
        }
    }

    pub fn advance(&mut self) {
        self.0.advance()
    }

    pub fn get_eta_ptr(&self) -> *const u8 {
        self.0.eta().as_ptr() as *const u8
    }

    pub fn get_etau_ptr(&self) -> *const u8 {
        self.0.etau().as_ptr() as *const u8
    }

    pub fn get_etav_ptr(&self) -> *const u8 {
        self.0.etav().as_ptr() as *const u8
    }
}
