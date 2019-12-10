use super::operators::SbpOperator;
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

    pub(crate) fn advance<SBP>(
        &self,
        fut: &mut Self,
        dt: f32,
        grid: &super::Grid<SBP>,
        work_buffers: Option<&mut WorkBuffers>,
    ) where
        SBP: SbpOperator,
    {
        assert_eq!(self.ex.shape(), fut.ex.shape());

        let mut wb: WorkBuffers;
        let (y, k, tmp) = if let Some(x) = work_buffers {
            (&mut x.y, &mut x.buf, &mut x.tmp)
        } else {
            wb = WorkBuffers::new(self.ex.shape()[1], self.ex.shape()[0]);
            (&mut wb.y, &mut wb.buf, &mut wb.tmp)
        };

        for i in 0..4 {
            // y = y0 + c*kn
            y.0.assign(&self.ex);
            y.1.assign(&self.hz);
            y.2.assign(&self.ey);
            match i {
                0 => {}
                1 | 2 => {
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

            // Solving (Au)_x + (Bu)_y
            // with:
            //        A               B
            //  [ 0,  0,  0]    [ 0,  1,  0]
            //  [ 0,  0, -1]    [ 1,  0,  0]
            //  [ 0, -1,  0]    [ 0,  0,  0]

            // This flux is rotated by the grid metrics
            // (Au)_x + (Bu)_y = 1/J [
            //          (J xi_x Au)_xi + (J eta_x Au)_eta
            //          (J xi_y Bu)_xi + (J eta_y Bu)_eta
            //      ]
            // where J is the grid determinant

            // ex = hz_y
            {
                ndarray::azip!((a in &mut tmp.0,
                                &dxi_dy in &grid.detj_dxi_dy,
                                &hz in &y.1)
                    *a = dxi_dy * hz
                );
                SBP::diffxi(tmp.0.view(), tmp.1.view_mut());

                ndarray::azip!((b in &mut tmp.2,
                                &deta_dy in &grid.detj_deta_dy,
                                &hz in &y.1)
                    *b = deta_dy * hz
                );
                SBP::diffeta(tmp.2.view(), tmp.3.view_mut());

                ndarray::azip!((flux in &mut k[i].0, &ax in &tmp.1, &by in &tmp.3)
                    *flux = ax + by
                );
            }

            {
                // hz = -ey_x + ex_y
                ndarray::azip!((a in &mut tmp.0,
                                &dxi_dx in &grid.detj_dxi_dx,
                                &dxi_dy in &grid.detj_dxi_dy,
                                &ex in &y.0,
                                &ey in &y.2)
                    *a = dxi_dx * -ey + dxi_dy * ex
                );
                SBP::diffxi(tmp.0.view(), tmp.1.view_mut());

                ndarray::azip!((b in &mut tmp.2,
                                &deta_dx in &grid.detj_deta_dx,
                                &deta_dy in &grid.detj_deta_dy,
                                &ex in &y.0,
                                &ey in &y.2)
                    *b = deta_dx * -ey + deta_dy * ex
                );
                SBP::diffeta(tmp.2.view(), tmp.3.view_mut());

                ndarray::azip!((flux in &mut k[i].1, &ax in &tmp.1, &by in &tmp.3)
                    *flux = ax + by
                );
            }

            // ey = -hz_x
            {
                ndarray::azip!((a in &mut tmp.0,
                                &dxi_dx in &grid.detj_dxi_dx,
                                &hz in &y.1)
                    *a = dxi_dx * -hz
                );
                SBP::diffxi(tmp.0.view(), tmp.1.view_mut());

                ndarray::azip!((b in &mut tmp.2,
                                &deta_dx in &grid.detj_deta_dx,
                                &hz in &y.1)
                    *b = deta_dx * -hz
                );
                SBP::diffeta(tmp.2.view(), tmp.3.view_mut());

                ndarray::azip!((flux in &mut k[i].2, &ax in &tmp.1, &by in &tmp.3)
                    *flux = ax + by
                );
            }

            // Boundary conditions (SAT)
            let ny = y.0.shape()[0];
            let nx = y.0.shape()[1];

            let hinv = 1.0 / (SBP::h()[0] / (nx - 1) as f32);

            fn positive_flux(kx: f32, ky: f32) -> [[f32; 3]; 3] {
                let r = (kx * kx + ky * ky).sqrt();
                [
                    [ky * ky / r / 2.0, ky / 2.0, -kx * ky / r / 2.0],
                    [ky / 2.0, r / 2.0, -kx / 2.0],
                    [-kx * ky / r / 2.0, -kx / 2.0, kx * kx / r / 2.0],
                ]
            }
            fn negative_flux(kx: f32, ky: f32) -> [[f32; 3]; 3] {
                let r = (kx * kx + ky * ky).sqrt();
                [
                    [-ky * ky / r / 2.0, ky / 2.0, kx * ky / r / 2.0],
                    [ky / 2.0, -r / 2.0, -kx / 2.0],
                    [kx * ky / r / 2.0, -kx / 2.0, -kx * kx / r / 2.0],
                ]
            }

            for j in 0..ny {
                // East boundary, positive flux
                let tau = -1.0;
                let g = (y.0[(j, 0)], y.1[(j, 0)], y.2[(j, 0)]);
                let v = (y.0[(j, nx - 1)], y.1[(j, nx - 1)], y.2[(j, nx - 1)]);

                let kx = grid.detj_dxi_dx[(j, nx - 1)];
                let ky = grid.detj_dxi_dy[(j, nx - 1)];

                let plus = positive_flux(kx, ky);

                k[i].0[(j, nx - 1)] += tau
                    * hinv
                    * (plus[0][0] * (v.0 - g.0)
                        + plus[0][1] * (v.1 - g.1)
                        + plus[0][2] * (v.2 - g.2));
                k[i].1[(j, nx - 1)] += tau
                    * hinv
                    * (plus[1][0] * (v.0 - g.0)
                        + plus[1][1] * (v.1 - g.1)
                        + plus[1][2] * (v.2 - g.2));
                k[i].2[(j, nx - 1)] += tau
                    * hinv
                    * (plus[2][0] * (v.0 - g.0)
                        + plus[2][1] * (v.1 - g.1)
                        + plus[2][2] * (v.2 - g.2));

                // West boundary, negative flux
                let tau = 1.0;
                let (v, g) = (g, v);

                let kx = grid.detj_dxi_dx[(j, 0)];
                let ky = grid.detj_dxi_dy[(j, 0)];

                let minus = negative_flux(kx, ky);

                k[i].0[(j, 0)] += tau
                    * hinv
                    * (minus[0][0] * (v.0 - g.0)
                        + minus[0][1] * (v.1 - g.1)
                        + minus[0][2] * (v.2 - g.2));
                k[i].1[(j, 0)] += tau
                    * hinv
                    * (minus[1][0] * (v.0 - g.0)
                        + minus[1][1] * (v.1 - g.1)
                        + minus[1][2] * (v.2 - g.2));
                k[i].2[(j, 0)] += tau
                    * hinv
                    * (minus[2][0] * (v.0 - g.0)
                        + minus[2][1] * (v.1 - g.1)
                        + minus[2][2] * (v.2 - g.2));
            }

            let hinv = 1.0 / (SBP::h()[0] / (ny - 1) as f32);

            for j in 0..nx {
                // North boundary, positive flux
                let tau = -1.0;
                let g = (y.0[(0, j)], y.1[(0, j)], y.2[(0, j)]);
                let v = (y.0[(ny - 1, j)], y.1[(ny - 1, j)], y.2[(ny - 1, j)]);

                let kx = grid.detj_deta_dx[(ny - 1, j)];
                let ky = grid.detj_deta_dy[(ny - 1, j)];

                let plus = positive_flux(kx, ky);

                k[i].0[(ny - 1, j)] += tau
                    * hinv
                    * (plus[0][0] * (v.0 - g.0)
                        + plus[0][1] * (v.1 - g.1)
                        + plus[0][2] * (v.2 - g.2));
                k[i].1[(ny - 1, j)] += tau
                    * hinv
                    * (plus[1][0] * (v.0 - g.0)
                        + plus[1][1] * (v.1 - g.1)
                        + plus[1][2] * (v.2 - g.2));
                k[i].2[(ny - 1, j)] += tau
                    * hinv
                    * (plus[2][0] * (v.0 - g.0)
                        + plus[2][1] * (v.1 - g.1)
                        + plus[2][2] * (v.2 - g.2));

                // South boundary, negative flux
                let tau = 1.0;
                let (g, v) = (v, g);

                let kx = grid.detj_deta_dx[(0, j)];
                let ky = grid.detj_deta_dy[(0, j)];

                let minus = negative_flux(kx, ky);

                k[i].0[(0, j)] += tau
                    * hinv
                    * (minus[0][0] * (v.0 - g.0)
                        + minus[0][1] * (v.1 - g.1)
                        + minus[0][2] * (v.2 - g.2));
                k[i].1[(0, j)] += tau
                    * hinv
                    * (minus[1][0] * (v.0 - g.0)
                        + minus[1][1] * (v.1 - g.1)
                        + minus[1][2] * (v.2 - g.2));
                k[i].2[(0, j)] += tau
                    * hinv
                    * (minus[2][0] * (v.0 - g.0)
                        + minus[2][1] * (v.1 - g.1)
                        + minus[2][2] * (v.2 - g.2));
            }

            ndarray::azip!((k0 in &mut k[i].0,
                            k1 in &mut k[i].1,
                            k2 in &mut k[i].2,
                            &detj in &grid.detj) {
                *k0 /= detj;
                *k1 /= detj;
                *k2 /= detj;
            });
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
    tmp: (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>),
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
                (arr.clone(), arr.clone(), arr.clone()),
            ],
            tmp: (arr.clone(), arr.clone(), arr.clone(), arr),
        }
    }
}
