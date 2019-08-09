use ndarray::{Array2, Zip};
use wasm_bindgen::prelude::*;

mod operators;
use operators::diffx_periodic;

#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
pub struct Universe {
    ex: Array2<f32>,
    // e_y: Array2<f32>,
    hz: Array2<f32>,
}

#[wasm_bindgen]
pub fn set_panic_hook() {
    #[cfg(feature = "console_error_panic_hook")]
    console_error_panic_hook::set_once();
}

fn sin_plus_cos(x: f32, y: f32, t: f32) -> f32 {
    use std::f32;
    (2.0 * f32::consts::PI * (x + y) - t).sin()
}

fn exponential(x: f32, y: f32, _t: f32) -> f32 {
    use std::f32;
    let x = x - 0.5;
    let y = y - 0.5;

    let sigma = 0.05;

    1.0 / (2.0 * f32::consts::PI * sigma * sigma) * (-(x * x + y * y) / (2.0 * sigma * sigma)).exp()
}

fn gaussian(x: f32, y: f32, t: f32) -> f32 {
    use std::f32;
    let x = x - 0.5;
    let y = y - 0.5;

    let sigma = 0.05;

    1.0 / (2.0 * f32::consts::PI * sigma * sigma)
        * (-(x + y - t) * (x + y - t) / (2.0 * sigma * sigma)).exp()
}

#[wasm_bindgen]
impl Universe {
    pub fn new(width: u32, height: u32) -> Self {
        let field = Array2::zeros((height as usize, width as usize));
        let ex = field.clone();
        // let e_y = field.clone();
        let hz = field.clone();

        Universe {
            ex,
            // e_y,
            hz,
        }
    }

    pub fn set_initial(&mut self, t: f32, selected_function: &str) {
        let func = match selected_function {
            "sin+cos" => sin_plus_cos,
            "exp" => exponential,
            "gauss" => gaussian,
            _ => |_x: f32, _y: f32, _t: f32| 0.0,
        };
        let nx = self.ex.shape()[1];
        let ny = self.ex.shape()[0];
        for j in 0..ny {
            for i in 0..nx {
                // Must divice interval on nx/ny instead of nx - 1/ny-1
                // due to periodic conditions [0, 1)
                let x = i as f32 / nx as f32;
                let y = j as f32 / ny as f32;
                self.ex[(j, i)] = func(x, y, t);
                self.hz[(j, i)] = -func(x, y, t);
                // self.e_y[(j as usize, i as usize)] = func(x, y, t);
            }
        }
    }

    pub fn advance(&self, fut: &mut Universe, dt: f32, work_buffers: Option<WorkBuffers>) {
        assert_eq!(self.ex.shape(), fut.ex.shape());

        let mut buffers = work_buffers
            .unwrap_or_else(|| WorkBuffers::new(self.ex.shape()[1], self.ex.shape()[0]));

        let mut y = buffers.y;
        let mut k = buffers.buf;

        for i in 0..4 {
            // y = y0 + c*kn
            y.0.assign(&self.ex);
            y.1.assign(&self.hz);
            match i {
                0 => {}
                1 => {
                    y.0.scaled_add(1.0 / 2.0 * dt, &k[i - 1].0);
                    y.1.scaled_add(1.0 / 2.0 * dt, &k[i - 1].1);
                }
                2 => {
                    y.0.scaled_add(1.0 / 2.0 * dt, &k[i - 1].0);
                    y.1.scaled_add(1.0 / 2.0 * dt, &k[i - 1].1);
                }
                3 => {
                    y.0.scaled_add(dt, &k[i - 1].0);
                    y.1.scaled_add(dt, &k[i - 1].1);
                }
                _ => {
                    unreachable!();
                }
            };

            k[i].0.fill(0.0);
            k[i].1.fill(0.0);

            diffx_periodic(y.0.view(), k[i].1.view_mut());
            diffx_periodic(y.1.view(), k[i].0.view_mut());
        }

        Zip::from(&mut fut.ex)
            .and(&self.ex)
            .and(&k[0].0)
            .and(&k[1].0)
            .and(&k[2].0)
            .and(&k[3].0)
            .apply(|y1, &y0, &k1, &k2, &k3, &k4| {
                *y1 = y0 + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            });
        Zip::from(&mut fut.hz)
            .and(&self.hz)
            .and(&k[0].1)
            .and(&k[1].1)
            .and(&k[2].1)
            .and(&k[3].1)
            .apply(|y1, &y0, &k1, &k2, &k3, &k4| {
                *y1 = y0 + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            });
    }

    pub fn get_ex_ptr(&mut self) -> *mut u8 {
        self.ex.as_mut_ptr() as *mut u8
    }

    pub fn get_hz_ptr(&mut self) -> *mut u8 {
        self.hz.as_mut_ptr() as *mut u8
    }
}

#[wasm_bindgen]
pub struct WorkBuffers {
    y: (Array2<f32>, Array2<f32>),
    buf: [(Array2<f32>, Array2<f32>); 4],
}

#[wasm_bindgen]
impl WorkBuffers {
    pub fn new(nx: usize, ny: usize) -> Self {
        let arr = Array2::zeros((ny, nx));
        Self {
            y: (arr.clone(), arr.clone()),
            buf: [
                (arr.clone(), arr.clone()),
                (arr.clone(), arr.clone()),
                (arr.clone(), arr.clone()),
                (arr.clone(), arr),
            ],
        }
    }
}
