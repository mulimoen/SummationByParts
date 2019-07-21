use ndarray::Array2;
use wasm_bindgen::prelude::*;

#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

pub fn set_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub struct Universe {
    width: u32,
    height: u32,
    field: Array2<f32>,
}

const WAVESPEED: f32 = 1.0;

fn func(x: f32, y: f32, t: f32) -> f32 {
    use std::f32;
    (2.0 * f32::consts::PI * (x + y) - WAVESPEED * t).sin()
}

#[wasm_bindgen]
impl Universe {
    pub fn new(width: u32, height: u32) -> Self {
        set_panic_hook();

        let field = Array2::zeros((height as usize, width as usize));

        Universe {
            width,
            height,
            field,
        }
    }

    pub fn set_initial(&mut self, t: f32) {
        for j in 0..self.height {
            for i in 0..self.width {
                let x = i as f32 / self.width as f32;
                let y = j as f32 / self.height as f32;
                self.field[(j as usize, i as usize)] = func(x, y, t);
            }
        }
    }

    pub fn get_ptr(&mut self) -> *mut u8 {
        self.field.as_mut_ptr() as *mut u8
    }
}
