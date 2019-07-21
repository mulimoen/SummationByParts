use ndarray::{Array2, ArrayView2, ArrayViewMut2};
use wasm_bindgen::prelude::*;

#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
pub struct Universe {
    width: u32,
    height: u32,
    e_x: Array2<f32>,
    // e_y: Array2<f32>,
    // h_z: Array2<f32>,
}

const WAVESPEED: f32 = 1.0;

#[wasm_bindgen]
pub fn set_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

fn func(x: f32, y: f32, t: f32) -> f32 {
    use std::f32;
    (2.0 * f32::consts::PI * (x + y) - WAVESPEED * t).sin()
}

#[wasm_bindgen]
impl Universe {
    pub fn new(width: u32, height: u32) -> Self {
        let field = Array2::zeros((height as usize, width as usize));
        let e_x = field.clone();
        // let e_y = field.clone();
        // let h_z = field.clone();

        Universe {
            width,
            height,
            e_x,
            // e_y,
            // h_z,
        }
    }

    pub fn set_initial(&mut self, t: f32) {
        for j in 0..self.height {
            for i in 0..self.width {
                let x = i as f32 / self.width as f32;
                let y = j as f32 / self.height as f32;
                self.e_x[(j as usize, i as usize)] = func(x, y, t);
                // self.e_y[(j as usize, i as usize)] = func(x, y, t);
                // self.h_z[(j as usize, i as usize)] = func(x, y, t);
            }
        }
    }

    pub fn advance(&self, fut: &mut Universe, dt: f32) {
        assert_eq!(self.width, fut.width);
        assert_eq!(self.height, fut.height);

        fut.e_x.assign(&self.e_x);

        traditional4(self.e_x.view(), fut.e_x.view_mut(), dt);
    }

    pub fn get_ptr(&mut self) -> *mut u8 {
        self.e_x.as_mut_ptr() as *mut u8
    }
}

fn traditional4(prev: ArrayView2<f32>, mut fut: ArrayViewMut2<f32>, dt: f32) {
    assert_eq!(prev.shape(), fut.shape());
    let nx = prev.shape()[1];
    let ny = prev.shape()[0];

    let dx = 1.0 / (nx - 1) as f32;

    let diag = [1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0];

    for j in 0..ny {
        let diff = diag[0] * prev[(j, nx - 2)]
            + diag[1] * prev[(j, nx - 1)]
            + diag[2] * prev[(j, 0)]
            + diag[3] * prev[(j, 1)]
            + diag[4] * prev[(j, 2)];
        fut[(j, 0)] += dt / dx * diff;
        let diff = diag[0] * prev[(j, nx - 1)]
            + diag[1] * prev[(j, 0)]
            + diag[2] * prev[(j, 1)]
            + diag[3] * prev[(j, 2)]
            + diag[4] * prev[(j, 3)];
        fut[(j, 1)] += dt / dx * diff;
        for i in 2..nx - 2 {
            let diff = diag[0] * prev[(j, i - 2)]
                + diag[1] * prev[(j, i - 1)]
                + diag[2] * prev[(j, i)]
                + diag[3] * prev[(j, i + 1)]
                + diag[4] * prev[(j, i + 2)];
            fut[(j, i)] += dt / dx * diff;
        }
        let diff = diag[0] * prev[(j, nx - 4)]
            + diag[1] * prev[(j, nx - 3)]
            + diag[2] * prev[(j, nx - 2)]
            + diag[3] * prev[(j, nx - 1)]
            + diag[4] * prev[(j, 0)];
        fut[(j, nx - 2)] += dt / dx * diff;
        let diff = diag[0] * prev[(j, nx - 3)]
            + diag[1] * prev[(j, nx - 2)]
            + diag[2] * prev[(j, nx - 1)]
            + diag[3] * prev[(j, 0)]
            + diag[4] * prev[(j, 1)];
        fut[(j, nx - 1)] += dt / dx * diff;
    }
}
