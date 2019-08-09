use ndarray::{Array2, Zip};
use wasm_bindgen::prelude::*;

mod operators;
use operators::{diffx, diffx_periodic, diffy, diffy_periodic, dissx, dissy};

#[cfg(feature = "wee_alloc")]
#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
pub struct Universe {
    e_x: Array2<f32>,
    // e_y: Array2<f32>,
    // h_z: Array2<f32>,
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

#[wasm_bindgen]
impl Universe {
    pub fn new(width: u32, height: u32) -> Self {
        let field = Array2::zeros((height as usize, width as usize));
        let e_x = field.clone();
        // let e_y = field.clone();
        // let h_z = field.clone();

        Universe {
            e_x,
            // e_y,
            // h_z,
        }
    }

    pub fn set_initial(&mut self, t: f32, selected_function: &str) {
        let func = match selected_function {
            "sin+cos" => sin_plus_cos,
            "exp" => exponential,
            _ => |_x: f32, _y: f32, _t: f32| 0.0,
        };
        let nx = self.e_x.shape()[1];
        let ny = self.e_x.shape()[0];
        for j in 0..ny {
            for i in 0..nx {
                // Must divice interval on nx/ny instead of nx - 1/ny-1
                // due to periodic conditions [0, 1)
                let x = i as f32 / nx as f32;
                let y = j as f32 / ny as f32;
                self.e_x[(j, i)] = func(x, y, t);
                // self.e_y[(j as usize, i as usize)] = func(x, y, t);
                // self.h_z[(j as usize, i as usize)] = func(x, y, t);
            }
        }
    }

    pub fn advance(&self, fut: &mut Universe, dt: f32, work_buffers: Option<WorkBuffers>) {
        assert_eq!(self.e_x.shape(), fut.e_x.shape());

        let mut y = self.e_x.clone();

        let mut buffers = work_buffers
            .unwrap_or_else(|| WorkBuffers::new(self.e_x.shape()[1], self.e_x.shape()[0]));

        // y.assign(&self.e_x)
        let mut k0 = &mut buffers.k1; // Only need k0 for first step in RK
        k0.fill(0.0);
        diffx_periodic(y.view(), k0.view_mut());
        diffy_periodic(y.view(), k0.view_mut());

        // y.assign(&self.e_x);
        y.scaled_add(-1.0 / 2.0 * dt, &k0);
        let mut k1 = buffers.k1;
        k1.fill(0.0);
        diffx_periodic(y.view(), k1.view_mut());
        diffy_periodic(y.view(), k1.view_mut());

        y.assign(&self.e_x);
        y.scaled_add(-1.0 / 2.0 * dt, &k1);
        let mut k2 = buffers.k2;
        k2.fill(0.0);
        diffx_periodic(y.view(), k2.view_mut());
        diffy_periodic(y.view(), k2.view_mut());

        y.assign(&self.e_x);
        y.scaled_add(-1.0 / 2.0 * dt, &k2);
        let mut k3 = buffers.k3;
        k3.fill(0.0);
        diffx_periodic(y.view(), k3.view_mut());
        diffy_periodic(y.view(), k3.view_mut());

        y.assign(&self.e_x);
        y.scaled_add(-dt, &k2);
        let mut k4 = buffers.k4;
        k4.fill(0.0);
        diffx_periodic(y.view(), k4.view_mut());
        diffy_periodic(y.view(), k4.view_mut());

        Zip::from(&mut fut.e_x)
            .and(&self.e_x)
            .and(&k1)
            .and(&k2)
            .and(&k3)
            .and(&k4)
            .apply(|y1, &y0, &k1, &k2, &k3, &k4| {
                *y1 = y0 + -dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            });
    }

    pub fn advance_upwind(&self, fut: &mut Universe, dt: f32) {
        assert_eq!(self.e_x.dim(), fut.e_x.dim());

        let mut ad = Array2::zeros(fut.e_x.dim());
        let mut y = self.e_x.clone();

        // y.assign(&self.e_x)
        // ad.fill(0.0);
        let mut k0 = Array2::zeros(self.e_x.dim());
        diffx(y.view(), k0.view_mut());
        diffy(y.view(), k0.view_mut());
        dissx(y.view(), ad.view_mut());
        dissy(y.view(), ad.view_mut());
        k0.scaled_add(-0.5, &ad);

        // y.assign(&self.e_x);
        ad.fill(0.0);
        y.scaled_add(-1.0 / 2.0 * dt, &k0);
        let mut k1 = Array2::zeros(self.e_x.dim());
        diffx(y.view(), k1.view_mut());
        diffy(y.view(), k1.view_mut());
        dissx(y.view(), ad.view_mut());
        dissy(y.view(), ad.view_mut());
        k1.scaled_add(-0.5, &ad);

        y.assign(&self.e_x);
        ad.fill(0.0);
        y.scaled_add(-1.0 / 2.0 * dt, &k1);
        let mut k2 = Array2::zeros(self.e_x.dim());
        diffx(y.view(), k2.view_mut());
        diffy(y.view(), k2.view_mut());
        dissx(y.view(), ad.view_mut());
        dissy(y.view(), ad.view_mut());
        k2.scaled_add(-0.5, &ad);

        y.assign(&self.e_x);
        ad.fill(0.0);
        y.scaled_add(-1.0 / 2.0 * dt, &k2);
        let mut k3 = Array2::zeros(self.e_x.dim());
        diffx(y.view(), k3.view_mut());
        diffy(y.view(), k3.view_mut());
        dissx(y.view(), ad.view_mut());
        dissy(y.view(), ad.view_mut());
        k3.scaled_add(-0.5, &ad);

        y.assign(&self.e_x);
        ad.fill(0.0);
        y.scaled_add(-dt, &k2);
        let mut k4 = Array2::zeros(self.e_x.dim());
        diffx(y.view(), k4.view_mut());
        diffy(y.view(), k4.view_mut());
        dissx(y.view(), ad.view_mut());
        dissy(y.view(), ad.view_mut());
        k4.scaled_add(-0.5, &ad);

        Zip::from(&mut fut.e_x)
            .and(&self.e_x)
            .and(&k1)
            .and(&k2)
            .and(&k3)
            .and(&k4)
            .apply(|y1, &y0, &k1, &k2, &k3, &k4| {
                *y1 = y0 + -dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            });
    }

    pub fn get_ptr(&mut self) -> *mut u8 {
        self.e_x.as_mut_ptr() as *mut u8
    }
}

#[wasm_bindgen]
pub struct WorkBuffers {
    k1: Array2<f32>,
    k2: Array2<f32>,
    k3: Array2<f32>,
    k4: Array2<f32>,
}

#[wasm_bindgen]
impl WorkBuffers {
    pub fn new(nx: usize, ny: usize) -> Self {
        Self {
            k1: Array2::zeros((ny, nx)),
            k2: Array2::zeros((ny, nx)),
            k3: Array2::zeros((ny, nx)),
            k4: Array2::zeros((ny, nx)),
        }
    }
}
