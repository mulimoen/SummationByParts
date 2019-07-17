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
    field: Vec<f32>,
}

#[wasm_bindgen]
pub struct Drawable {
    width: u32,
    height: u32,
    field: Vec<u8>,
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

        Universe {
            width,
            height,
            field: vec![0.0; width as usize * height as usize],
        }
    }

    pub fn set_initial(&mut self, t: f32) {
        for j in 0..self.height {
            for i in 0..self.width {
                let x = i as f32 / self.width as f32;
                let y = j as f32 / self.height as f32;
                self.field[(self.width * j + i) as usize] = func(x, y, t);
            }
        }
    }

    pub fn get_drawable(&self) -> Drawable {
        // let fmin = self.field.iter().fold(0.0f32, |acc, x| acc.min(*x));
        let fmin = -1.0;
        // let fmax = self.field.iter().fold(0.0f32, |acc, x| acc.max(*x));
        let fmax = 1.0;

        let field = self
            .field
            .iter()
            .map(|x| (u8::max_value() as f32 * (x - fmin) / (fmax - fmin)) as u8)
            .collect::<Vec<_>>();

        Drawable {
            width: self.width,
            height: self.height,
            field,
        }
    }
}

#[wasm_bindgen]
impl Drawable {
    pub fn get_pointer(&mut self) -> *mut u8 {
        self.field.as_mut_ptr()
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }
}
