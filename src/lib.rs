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
    field: Vec<u8>,
}

#[wasm_bindgen]
impl Universe {
    pub fn new(width: u32, height: u32) -> Self {
        set_panic_hook();

        Universe {
            width,
            height,
            field: vec![0u8; width as usize * height as usize],
        }
    }

    pub fn set_something(&mut self) {
        for j in 0..self.height {
            for i in 0..self.width {
                self.field[(self.width * j + i) as usize] =
                    ((10 * i + 100 * j) % (u8::max_value() as u32)) as u8;
            }
        }
    }

    pub fn get_field(&mut self) -> *mut u8 {
        self.field.as_mut_ptr()
    }
}
