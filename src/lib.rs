use ndarray::{s, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Zip};
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
            width,
            height,
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

        let mut y = self.e_x.clone();

        // y.assign(&self.e_x)
        let mut k0 = Array2::zeros((self.height as usize, self.width as usize));
        diffx(y.view(), k0.view_mut());
        diffy(y.view(), k0.view_mut());

        // y.assign(&self.e_x);
        y.scaled_add(-1.0 / 2.0 * dt, &k0);
        let mut k1 = Array2::zeros((self.height as usize, self.width as usize));
        diffx(y.view(), k1.view_mut());
        diffy(y.view(), k1.view_mut());

        y.assign(&self.e_x);
        y.scaled_add(-1.0 / 2.0 * dt, &k1);
        let mut k2 = Array2::zeros((self.height as usize, self.width as usize));
        diffx(y.view(), k2.view_mut());
        diffy(y.view(), k2.view_mut());

        y.assign(&self.e_x);
        y.scaled_add(-1.0 / 2.0 * dt, &k2);
        let mut k3 = Array2::zeros((self.height as usize, self.width as usize));
        diffx(y.view(), k3.view_mut());
        diffy(y.view(), k3.view_mut());

        y.assign(&self.e_x);
        y.scaled_add(-dt, &k2);
        let mut k4 = Array2::zeros((self.height as usize, self.width as usize));
        diffx(y.view(), k4.view_mut());
        diffy(y.view(), k4.view_mut());

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
        assert_eq!(self.width, fut.width);
        assert_eq!(self.height, fut.height);

        let mut ad = Array2::zeros(fut.e_x.dim());
        let mut y = self.e_x.clone();

        // y.assign(&self.e_x)
        // ad.fill(0.0);
        let mut k0 = Array2::zeros((self.height as usize, self.width as usize));
        diffx(y.view(), k0.view_mut());
        diffy(y.view(), k0.view_mut());
        dissx(y.view(), ad.view_mut());
        dissy(y.view(), ad.view_mut());
        k0.scaled_add(-0.5, &ad);

        // y.assign(&self.e_x);
        ad.fill(0.0);
        y.scaled_add(-1.0 / 2.0 * dt, &k0);
        let mut k1 = Array2::zeros((self.height as usize, self.width as usize));
        diffx(y.view(), k1.view_mut());
        diffy(y.view(), k1.view_mut());
        dissx(y.view(), ad.view_mut());
        dissy(y.view(), ad.view_mut());
        k1.scaled_add(-0.5, &ad);

        y.assign(&self.e_x);
        ad.fill(0.0);
        y.scaled_add(-1.0 / 2.0 * dt, &k1);
        let mut k2 = Array2::zeros((self.height as usize, self.width as usize));
        diffx(y.view(), k2.view_mut());
        diffy(y.view(), k2.view_mut());
        dissx(y.view(), ad.view_mut());
        dissy(y.view(), ad.view_mut());
        k2.scaled_add(-0.5, &ad);

        y.assign(&self.e_x);
        ad.fill(0.0);
        y.scaled_add(-1.0 / 2.0 * dt, &k2);
        let mut k3 = Array2::zeros((self.height as usize, self.width as usize));
        diffx(y.view(), k3.view_mut());
        diffy(y.view(), k3.view_mut());
        dissx(y.view(), ad.view_mut());
        dissy(y.view(), ad.view_mut());
        k3.scaled_add(-0.5, &ad);

        y.assign(&self.e_x);
        ad.fill(0.0);
        y.scaled_add(-dt, &k2);
        let mut k4 = Array2::zeros((self.height as usize, self.width as usize));
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

fn diffx(prev: ArrayView2<f32>, mut fut: ArrayViewMut2<f32>) {
    for j in 0..prev.shape()[0] {
        upwind4(prev.slice(s!(j, ..)), fut.slice_mut(s!(j, ..)));
    }
}

fn diffy(prev: ArrayView2<f32>, mut fut: ArrayViewMut2<f32>) {
    for i in 0..prev.shape()[1] {
        upwind4(prev.slice(s!(.., i)), fut.slice_mut(s!(.., i)));
    }
}

fn dissx(prev: ArrayView2<f32>, mut fut: ArrayViewMut2<f32>) {
    for j in 0..prev.shape()[0] {
        upwind4_diss(prev.slice(s!(j, ..)), fut.slice_mut(s!(j, ..)));
    }
}
fn dissy(prev: ArrayView2<f32>, mut fut: ArrayViewMut2<f32>) {
    for i in 0..prev.shape()[1] {
        upwind4_diss(prev.slice(s!(.., i)), fut.slice_mut(s!(.., i)));
    }
}

fn trad4(prev: ArrayView1<f32>, mut fut: ArrayViewMut1<f32>) {
    assert_eq!(prev.shape(), fut.shape());
    let nx = prev.shape()[0];

    let dx = 1.0 / (nx - 1) as f32;

    let diag = [1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0];

    let diff = diag[0] * prev[(nx - 2)]
        + diag[1] * prev[(nx - 1)]
        + diag[2] * prev[(0)]
        + diag[3] * prev[(1)]
        + diag[4] * prev[(2)];
    fut[(0)] += diff / dx;
    let diff = diag[0] * prev[(nx - 1)]
        + diag[1] * prev[(0)]
        + diag[2] * prev[(1)]
        + diag[3] * prev[(2)]
        + diag[4] * prev[(3)];
    fut[(1)] += diff / dx;
    for i in 2..nx - 2 {
        let diff = diag[0] * prev[(i - 2)]
            + diag[1] * prev[(i - 1)]
            + diag[2] * prev[(i)]
            + diag[3] * prev[(i + 1)]
            + diag[4] * prev[(i + 2)];
        fut[(i)] += diff / dx;
    }
    let diff = diag[0] * prev[(nx - 4)]
        + diag[1] * prev[(nx - 3)]
        + diag[2] * prev[(nx - 2)]
        + diag[3] * prev[(nx - 1)]
        + diag[4] * prev[(0)];
    fut[(nx - 2)] += diff / dx;
    let diff = diag[0] * prev[(nx - 3)]
        + diag[1] * prev[(nx - 2)]
        + diag[2] * prev[(nx - 1)]
        + diag[3] * prev[(0)]
        + diag[4] * prev[(1)];
    fut[(nx - 1)] += diff / dx;
}

fn upwind4(prev: ArrayView1<f32>, mut fut: ArrayViewMut1<f32>) {
    assert_eq!(prev.shape(), fut.shape());
    let nx = prev.shape()[0];

    let dx = 1.0 / (nx - 1) as f32;

    let diag = [
        -1.0 / 24.0,
        1.0 / 4.0,
        -7.0 / 8.0,
        0.0,
        7.0 / 8.0,
        -1.0 / 4.0,
        1.0 / 24.0,
    ];

    let diff = diag[0] * prev[(nx - 3)]
        + diag[1] * prev[(nx - 2)]
        + diag[2] * prev[(nx - 1)]
        + diag[3] * prev[(0)]
        + diag[4] * prev[(1)]
        + diag[5] * prev[(2)]
        + diag[6] * prev[(3)];
    fut[0] += diff / dx;
    let diff = diag[0] * prev[(nx - 2)]
        + diag[1] * prev[(nx - 1)]
        + diag[2] * prev[(0)]
        + diag[3] * prev[(1)]
        + diag[4] * prev[(2)]
        + diag[5] * prev[(3)]
        + diag[6] * prev[(4)];
    fut[1] += diff / dx;
    let diff = diag[0] * prev[(nx - 1)]
        + diag[1] * prev[(0)]
        + diag[2] * prev[(1)]
        + diag[3] * prev[(2)]
        + diag[4] * prev[(3)]
        + diag[5] * prev[(4)]
        + diag[6] * prev[(5)];
    fut[2] += diff / dx;

    for i in 3..nx - 3 {
        let diff = diag[0] * prev[(i - 3)]
            + diag[1] * prev[(i - 2)]
            + diag[2] * prev[(i - 1)]
            + diag[3] * prev[(i)]
            + diag[4] * prev[(i + 1)]
            + diag[5] * prev[(i + 2)]
            + diag[6] * prev[(i + 3)];
        fut[(i)] += diff / dx;
    }
    let diff = diag[0] * prev[(nx - 6)]
        + diag[1] * prev[(nx - 5)]
        + diag[2] * prev[(nx - 4)]
        + diag[3] * prev[(nx - 3)]
        + diag[4] * prev[(nx - 2)]
        + diag[5] * prev[(nx - 1)]
        + diag[6] * prev[(0)];
    fut[(nx - 3)] += diff / dx;
    let diff = diag[0] * prev[(nx - 5)]
        + diag[1] * prev[(nx - 4)]
        + diag[2] * prev[(nx - 3)]
        + diag[3] * prev[(nx - 2)]
        + diag[4] * prev[(nx - 1)]
        + diag[5] * prev[(0)]
        + diag[6] * prev[(1)];
    fut[(nx - 2)] += diff / dx;
    let diff = diag[0] * prev[(nx - 4)]
        + diag[1] * prev[(nx - 3)]
        + diag[2] * prev[(nx - 2)]
        + diag[3] * prev[(nx - 1)]
        + diag[4] * prev[(0)]
        + diag[5] * prev[(1)]
        + diag[6] * prev[(2)];
    fut[(nx - 1)] += diff / dx;
}

fn upwind4_diss(prev: ArrayView1<f32>, mut fut: ArrayViewMut1<f32>) {
    assert_eq!(prev.shape(), fut.shape());
    let nx = prev.shape()[0];

    let dx = 1.0 / (nx - 1) as f32;

    let diag = [
        1.0 / 24.0,
        -1.0 / 4.0,
        5.0 / 8.0,
        -5.0 / 6.0,
        5.0 / 8.0,
        -1.0 / 4.0,
        1.0 / 24.0,
    ];

    let diff = diag[0] * prev[(nx - 3)]
        + diag[1] * prev[(nx - 2)]
        + diag[2] * prev[(nx - 1)]
        + diag[3] * prev[(0)]
        + diag[4] * prev[(1)]
        + diag[5] * prev[(2)]
        + diag[6] * prev[(3)];
    fut[0] += diff / dx;
    let diff = diag[0] * prev[(nx - 2)]
        + diag[1] * prev[(nx - 1)]
        + diag[2] * prev[(0)]
        + diag[3] * prev[(1)]
        + diag[4] * prev[(2)]
        + diag[5] * prev[(3)]
        + diag[6] * prev[(4)];
    fut[1] += diff / dx;
    let diff = diag[0] * prev[(nx - 1)]
        + diag[1] * prev[(0)]
        + diag[2] * prev[(1)]
        + diag[3] * prev[(2)]
        + diag[4] * prev[(3)]
        + diag[5] * prev[(4)]
        + diag[6] * prev[(5)];
    fut[2] += diff / dx;

    for i in 3..nx - 3 {
        let diff = diag[0] * prev[(i - 3)]
            + diag[1] * prev[(i - 2)]
            + diag[2] * prev[(i - 1)]
            + diag[3] * prev[(i)]
            + diag[4] * prev[(i + 1)]
            + diag[5] * prev[(i + 2)]
            + diag[6] * prev[(i + 3)];
        fut[(i)] += diff / dx;
    }
    let diff = diag[0] * prev[(nx - 6)]
        + diag[1] * prev[(nx - 5)]
        + diag[2] * prev[(nx - 4)]
        + diag[3] * prev[(nx - 3)]
        + diag[4] * prev[(nx - 2)]
        + diag[5] * prev[(nx - 1)]
        + diag[6] * prev[(0)];
    fut[(nx - 3)] += diff / dx;
    let diff = diag[0] * prev[(nx - 5)]
        + diag[1] * prev[(nx - 4)]
        + diag[2] * prev[(nx - 3)]
        + diag[3] * prev[(nx - 2)]
        + diag[4] * prev[(nx - 1)]
        + diag[5] * prev[(0)]
        + diag[6] * prev[(1)];
    fut[(nx - 2)] += diff / dx;
    let diff = diag[0] * prev[(nx - 4)]
        + diag[1] * prev[(nx - 3)]
        + diag[2] * prev[(nx - 2)]
        + diag[3] * prev[(nx - 1)]
        + diag[4] * prev[(0)]
        + diag[5] * prev[(1)]
        + diag[6] * prev[(2)];
    fut[(nx - 1)] += diff / dx;
}
