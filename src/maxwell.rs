use super::operators::{diffx, diffy};
use ndarray::{Array2, Zip};

pub struct System {
    pub(crate) ex: Array2<f32>,
    pub(crate) ey: Array2<f32>,
    pub(crate) hz: Array2<f32>,
}

fn gaussian(x: f32, x0: f32, y: f32, y0: f32) -> f32 {
    use std::f32;
    let x = x - x0;
    let y = y - y0;

    let sigma = 0.05;

    1.0 / (2.0 * f32::consts::PI * sigma * sigma) * (-(x * x + y * y) / (2.0 * sigma * sigma)).exp()
}

impl System {
    pub fn new(width: u32, height: u32) -> Self {
        let field = Array2::zeros((height as usize, width as usize));
        let ex = field.clone();
        let ey = field.clone();
        let hz = field;

        Self { ex, ey, hz }
    }

    pub fn set_gaussian(&mut self, x0: f32, y0: f32) {
        let nx = self.ex.shape()[1];
        let ny = self.ex.shape()[0];
        for j in 0..ny {
            for i in 0..nx {
                // Must divice interval on nx/ny instead of nx - 1/ny-1
                // due to periodic conditions [0, 1)
                let x = i as f32 / nx as f32;
                let y = j as f32 / ny as f32;
                self.ex[(j, i)] = 0.0;
                self.ey[(j, i)] = 0.0;
                self.hz[(j, i)] = gaussian(x, x0, y, y0) / 32.0;
            }
        }
    }

    pub fn advance(&self, fut: &mut System, dt: f32, work_buffers: Option<&mut WorkBuffers>) {
        assert_eq!(self.ex.shape(), fut.ex.shape());

        let mut wb: WorkBuffers;
        let (y, k) = match work_buffers {
            Some(x) => (&mut x.y, &mut x.buf),
            None => {
                wb = WorkBuffers::new(self.ex.shape()[1], self.ex.shape()[0]);
                (&mut wb.y, &mut wb.buf)
            }
        };

        for i in 0..4 {
            // y = y0 + c*kn
            y.0.assign(&self.ex);
            y.1.assign(&self.hz);
            y.2.assign(&self.ey);
            match i {
                0 => {}
                1 => {
                    y.0.scaled_add(1.0 / 2.0 * dt, &k[i - 1].0);
                    y.1.scaled_add(1.0 / 2.0 * dt, &k[i - 1].1);
                    y.2.scaled_add(1.0 / 2.0 * dt, &k[i - 1].2);
                }
                2 => {
                    y.0.scaled_add(1.0 / 2.0 * dt, &k[i - 1].0);
                    y.1.scaled_add(1.0 / 2.0 * dt, &k[i - 1].1);
                    y.2.scaled_add(1.0 / 2.0 * dt, &k[i - 1].2);
                }
                3 => {
                    y.0.scaled_add(dt, &k[i - 1].0);
                    y.1.scaled_add(dt, &k[i - 1].1);
                    y.2.scaled_add(dt, &k[i - 1].2);
                }
                _ => {
                    unreachable!();
                }
            };

            // ex = hz_y
            k[i].0.fill(0.0);
            diffy(y.1.view(), k[i].0.view_mut());

            // ey = -hz_x
            k[i].2.fill(0.0);
            diffx(y.1.view(), k[i].2.view_mut());
            k[i].2.mapv_inplace(|v| -v);

            // hz = -ey_x + ex_y
            k[i].1.fill(0.0);
            diffx(y.2.view(), k[i].1.view_mut());
            k[i].1.mapv_inplace(|v| -v);
            diffy(y.0.view(), k[i].1.view_mut());

            // Boundary conditions (SAT)
            let ny = y.0.shape()[0];
            let nx = y.0.shape()[1];

            let h = 49.0 / 144.0 / (nx - 1) as f32; // TODO: Get from schema
            let hinv = 1.0 / h;

            // East boundary
            for j in 0..ny {
                let tau = -1.0;
                let g = (y.0[(j, 0)], y.1[(j, 0)], y.2[(j, 0)]);
                let v = (y.0[(j, nx - 1)], y.1[(j, nx - 1)], y.2[(j, nx - 1)]);

                // A+ = (0, 0, 0; 0, 1/2, -1/2; 0, -1/2, 1/2);
                k[i].0[(j, nx - 1)] += 0.0;
                k[i].1[(j, nx - 1)] += tau * hinv * (0.5 * (v.1 - g.1) - 0.5 * (v.2 - g.2));
                k[i].2[(j, nx - 1)] += tau * hinv * (-0.5 * (v.1 - g.1) + 0.5 * (v.2 - g.2));
            }
            // West boundary
            for j in 0..ny {
                let tau = 1.0;
                let g = (y.0[(j, nx - 1)], y.1[(j, nx - 1)], y.2[(j, nx - 1)]);
                let v = (y.0[(j, 0)], y.1[(j, 0)], y.2[(j, 0)]);

                // A- = (0, 0, 0; 0, -1/2, -1/2; 0, -1/2, -1/2);
                k[i].0[(j, 0)] += 0.0;
                k[i].1[(j, 0)] += tau * hinv * (-0.5 * (v.1 - g.1) - 0.5 * (v.2 - g.2));
                k[i].2[(j, 0)] += tau * hinv * (-0.5 * (v.1 - g.1) - 0.5 * (v.2 - g.2));
            }

            let h = 49.0 / 144.0 / (ny - 1) as f32; // TODO: same as above
            let hinv = 1.0 / h;

            // North boundary
            for j in 0..nx {
                let tau = -1.0;
                let g = (y.0[(0, j)], y.1[(0, j)], y.2[(0, j)]);
                let v = (y.0[(ny - 1, j)], y.1[(ny - 1, j)], y.2[(ny - 1, j)]);

                // B+ = (1/2, 1/2, 0; 1/2, 1/2, 0; 0, 0, 0)
                k[i].0[(ny - 1, j)] += tau * hinv * (0.5 * (v.0 - g.0) + 0.5 * (v.1 - g.1));
                k[i].1[(ny - 1, j)] += tau * hinv * (0.5 * (v.0 - g.0) + 0.5 * (v.1 - g.1));
                k[i].2[(ny - 1, j)] += 0.0;
            }

            // South boundary
            for j in 0..nx {
                let tau = 1.0;
                let g = (y.0[(ny - 1, j)], y.1[(ny - 1, j)], y.2[(ny - 1, j)]);
                let v = (y.0[(0, j)], y.1[(0, j)], y.2[(0, j)]);

                // B- = (-1/2, 1/2, 0; 1/2, -1/2, 0; 0, 0, 0);
                k[i].0[(0, j)] += tau * hinv * (-0.5 * (v.0 - g.0) + 0.5 * (v.1 - g.1));
                k[i].1[(0, j)] += tau * hinv * (0.5 * (v.0 - g.0) - 0.5 * (v.1 - g.1));
                k[i].2[(0, j)] += 0.0;
            }
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
        Zip::from(&mut fut.ey)
            .and(&self.ey)
            .and(&k[0].2)
            .and(&k[1].2)
            .and(&k[2].2)
            .and(&k[3].2)
            .apply(|y1, &y0, &k1, &k2, &k3, &k4| {
                *y1 = y0 + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            });
    }
}

pub struct WorkBuffers {
    y: (Array2<f32>, Array2<f32>, Array2<f32>),
    buf: [(Array2<f32>, Array2<f32>, Array2<f32>); 4],
}

impl WorkBuffers {
    pub fn new(nx: usize, ny: usize) -> Self {
        let arr = Array2::zeros((ny, nx));
        Self {
            y: (arr.clone(), arr.clone(), arr.clone()),
            buf: [
                (arr.clone(), arr.clone(), arr.clone()),
                (arr.clone(), arr.clone(), arr.clone()),
                (arr.clone(), arr.clone(), arr.clone()),
                (arr.clone(), arr.clone(), arr),
            ],
        }
    }
}
