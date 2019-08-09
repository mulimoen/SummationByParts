use ndarray::{Array2, Zip};
use wasm_bindgen::prelude::*;

mod operators;
use operators::{diffx, diffx_periodic, diffy, diffy_periodic, dissx, dissy};

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

        let mut y_ex = self.ex.clone();
        let mut y_hz = self.hz.clone();

        let mut buffers = work_buffers
            .unwrap_or_else(|| WorkBuffers::new(self.ex.shape()[1], self.ex.shape()[0]));

        let mut k0_ex = &mut buffers.k1_ex;
        let mut k0_hz = &mut buffers.k1_hz;
        k0_ex.fill(0.0);
        k0_hz.fill(0.0);
        // u_x
        diffx_periodic(y_ex.view(), k0_ex.view_mut());
        diffx_periodic(y_hz.view(), k0_hz.view_mut());

        // y = y + 1/2*dt Au_x
        y_ex.scaled_add(1.0 / 2.0 * dt, &k0_hz);
        y_hz.scaled_add(1.0 / 2.0 * dt, &k0_ex);
        let mut k1_ex = buffers.k1_ex;
        let mut k1_hz = buffers.k1_hz;
        k1_ex.fill(0.0);
        k1_hz.fill(0.0);
        diffx_periodic(y_ex.view(), k1_ex.view_mut());
        diffx_periodic(y_hz.view(), k1_hz.view_mut());

        y_ex.assign(&self.ex);
        y_hz.assign(&self.hz);
        y_ex.scaled_add(1.0 / 2.0 * dt, &k1_hz);
        y_hz.scaled_add(1.0 / 2.0 * dt, &k1_ex);
        let mut k2_ex = buffers.k2_ex;
        let mut k2_hz = buffers.k2_hz;
        k2_ex.fill(0.0);
        k2_hz.fill(0.0);
        diffx_periodic(y_ex.view(), k2_ex.view_mut());
        diffx_periodic(y_hz.view(), k2_hz.view_mut());

        // y = y0
        y_ex.assign(&self.ex);
        y_hz.assign(&self.hz);
        y_ex.scaled_add(1.0 / 2.0 * dt, &k2_hz);
        y_hz.scaled_add(1.0 / 2.0 * dt, &k2_ex);
        let mut k3_ex = buffers.k3_ex;
        let mut k3_hz = buffers.k3_hz;
        k3_ex.fill(0.0);
        k3_hz.fill(0.0);
        diffx_periodic(y_ex.view(), k3_ex.view_mut());
        diffx_periodic(y_hz.view(), k3_hz.view_mut());

        y_ex.assign(&self.ex);
        y_hz.assign(&self.hz);
        y_ex.scaled_add(dt, &k2_hz);
        y_hz.scaled_add(dt, &k2_ex);
        let mut k4_ex = buffers.k4_ex;
        let mut k4_hz = buffers.k4_hz;
        k4_ex.fill(0.0);
        k4_hz.fill(0.0);
        diffx_periodic(y_ex.view(), k4_ex.view_mut());
        diffx_periodic(y_hz.view(), k4_hz.view_mut());

        Zip::from(&mut fut.ex)
            .and(&self.ex)
            .and(&k1_hz)
            .and(&k2_hz)
            .and(&k3_hz)
            .and(&k4_hz)
            .apply(|y1, &y0, &k1, &k2, &k3, &k4| {
                *y1 = y0 + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            });
        Zip::from(&mut fut.hz)
            .and(&self.hz)
            .and(&k1_ex)
            .and(&k2_ex)
            .and(&k3_ex)
            .and(&k4_ex)
            .apply(|y1, &y0, &k1, &k2, &k3, &k4| {
                *y1 = y0 + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            });
    }

    pub fn advance_upwind(&self, fut: &mut Universe, dt: f32) {
        assert_eq!(self.ex.dim(), fut.ex.dim());

        let mut ad = Array2::zeros(fut.ex.dim());
        let mut y = self.ex.clone();

        // y.assign(&self.ex)
        // ad.fill(0.0);
        let mut k0 = Array2::zeros(self.ex.dim());
        diffx(y.view(), k0.view_mut());
        diffy(y.view(), k0.view_mut());
        dissx(y.view(), ad.view_mut());
        dissy(y.view(), ad.view_mut());
        k0.scaled_add(-0.5, &ad);

        // y.assign(&self.ex);
        ad.fill(0.0);
        y.scaled_add(-1.0 / 2.0 * dt, &k0);
        let mut k1 = Array2::zeros(self.ex.dim());
        diffx(y.view(), k1.view_mut());
        diffy(y.view(), k1.view_mut());
        dissx(y.view(), ad.view_mut());
        dissy(y.view(), ad.view_mut());
        k1.scaled_add(-0.5, &ad);

        y.assign(&self.ex);
        ad.fill(0.0);
        y.scaled_add(-1.0 / 2.0 * dt, &k1);
        let mut k2 = Array2::zeros(self.ex.dim());
        diffx(y.view(), k2.view_mut());
        diffy(y.view(), k2.view_mut());
        dissx(y.view(), ad.view_mut());
        dissy(y.view(), ad.view_mut());
        k2.scaled_add(-0.5, &ad);

        y.assign(&self.ex);
        ad.fill(0.0);
        y.scaled_add(-1.0 / 2.0 * dt, &k2);
        let mut k3 = Array2::zeros(self.ex.dim());
        diffx(y.view(), k3.view_mut());
        diffy(y.view(), k3.view_mut());
        dissx(y.view(), ad.view_mut());
        dissy(y.view(), ad.view_mut());
        k3.scaled_add(-0.5, &ad);

        y.assign(&self.ex);
        ad.fill(0.0);
        y.scaled_add(-dt, &k2);
        let mut k4 = Array2::zeros(self.ex.dim());
        diffx(y.view(), k4.view_mut());
        diffy(y.view(), k4.view_mut());
        dissx(y.view(), ad.view_mut());
        dissy(y.view(), ad.view_mut());
        k4.scaled_add(-0.5, &ad);

        Zip::from(&mut fut.ex)
            .and(&self.ex)
            .and(&k1)
            .and(&k2)
            .and(&k3)
            .and(&k4)
            .apply(|y1, &y0, &k1, &k2, &k3, &k4| {
                *y1 = y0 + -dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            });
    }

    pub fn get_ptr(&mut self) -> *mut u8 {
        self.ex.as_mut_ptr() as *mut u8
    }
}

#[wasm_bindgen]
pub struct WorkBuffers {
    k1_ex: Array2<f32>,
    k1_hz: Array2<f32>,
    k2_ex: Array2<f32>,
    k2_hz: Array2<f32>,
    k3_ex: Array2<f32>,
    k3_hz: Array2<f32>,
    k4_ex: Array2<f32>,
    k4_hz: Array2<f32>,
}

#[wasm_bindgen]
impl WorkBuffers {
    pub fn new(nx: usize, ny: usize) -> Self {
        Self {
            k1_ex: Array2::zeros((ny, nx)),
            k1_hz: Array2::zeros((ny, nx)),
            k2_ex: Array2::zeros((ny, nx)),
            k2_hz: Array2::zeros((ny, nx)),
            k3_ex: Array2::zeros((ny, nx)),
            k3_hz: Array2::zeros((ny, nx)),
            k4_ex: Array2::zeros((ny, nx)),
            k4_hz: Array2::zeros((ny, nx)),
        }
    }
}
