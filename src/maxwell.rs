use super::operators::SbpOperator;
use ndarray::prelude::*;
use ndarray::{azip, Zip};

#[derive(Clone, Debug)]
pub struct Field(pub(crate) Array3<f32>);

impl std::ops::Deref for Field {
    type Target = Array3<f32>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for Field {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

fn gaussian(x: f32, x0: f32, y: f32, y0: f32) -> f32 {
    use std::f32;
    let x = x - x0;
    let y = y - y0;

    let sigma = 0.05;

    1.0 / (2.0 * f32::consts::PI * sigma * sigma) * (-(x * x + y * y) / (2.0 * sigma * sigma)).exp()
}

impl Field {
    pub fn new(width: usize, height: usize) -> Self {
        let field = Array3::zeros((3, height, width));

        Self(field)
    }

    pub fn nx(&self) -> usize {
        self.0.shape()[2]
    }
    pub fn ny(&self) -> usize {
        self.0.shape()[1]
    }

    pub fn ex(&self) -> ArrayView2<f32> {
        self.slice(s![0, .., ..])
    }
    pub fn hz(&self) -> ArrayView2<f32> {
        self.slice(s![1, .., ..])
    }
    pub fn ey(&self) -> ArrayView2<f32> {
        self.slice(s![2, .., ..])
    }

    pub fn ex_mut(&mut self) -> ArrayViewMut2<f32> {
        self.slice_mut(s![0, .., ..])
    }
    pub fn hz_mut(&mut self) -> ArrayViewMut2<f32> {
        self.slice_mut(s![1, .., ..])
    }
    pub fn ey_mut(&mut self) -> ArrayViewMut2<f32> {
        self.slice_mut(s![2, .., ..])
    }

    pub fn components_mut(
        &mut self,
    ) -> (ArrayViewMut2<f32>, ArrayViewMut2<f32>, ArrayViewMut2<f32>) {
        let nx = self.nx();
        let ny = self.ny();

        let (ex, f) = self.0.view_mut().split_at(Axis(0), 1);
        let (hz, ey) = f.split_at(Axis(0), 1);

        (
            ex.into_shape((ny, nx)).unwrap(),
            hz.into_shape((ny, nx)).unwrap(),
            ey.into_shape((ny, nx)).unwrap(),
        )
    }

    pub fn set_gaussian(&mut self, x0: f32, y0: f32) {
        let nx = self.nx();
        let ny = self.ny();

        let (mut ex, mut hz, mut ey) = self.components_mut();
        for j in 0..ny {
            for i in 0..nx {
                // Must divice interval on nx/ny instead of nx - 1/ny-1
                // due to periodic conditions [0, 1)
                let x = i as f32 / nx as f32;
                let y = j as f32 / ny as f32;
                ex[(j, i)] = 0.0;
                ey[(j, i)] = 0.0;
                hz[(j, i)] = gaussian(x, x0, y, y0) / 32.0;
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
        assert_eq!(self.0.shape(), fut.0.shape());

        let mut wb: WorkBuffers;
        let (y, k, tmp) = if let Some(x) = work_buffers {
            (&mut x.y, &mut x.buf, &mut x.tmp)
        } else {
            wb = WorkBuffers::new(self.nx(), self.ny());
            (&mut wb.y, &mut wb.buf, &mut wb.tmp)
        };

        for i in 0..4 {
            // y = y0 + c*kn
            y.assign(&self);
            match i {
                0 => {}
                1 | 2 => {
                    y.scaled_add(1.0 / 2.0 * dt, &k[i - 1]);
                }
                3 => {
                    y.scaled_add(dt, &k[i - 1]);
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
                                &hz in &y.hz())
                    *a = dxi_dy * hz
                );
                SBP::diffxi(tmp.0.view(), tmp.1.view_mut());

                ndarray::azip!((b in &mut tmp.2,
                                &deta_dy in &grid.detj_deta_dy,
                                &hz in &y.hz())
                    *b = deta_dy * hz
                );
                SBP::diffeta(tmp.2.view(), tmp.3.view_mut());

                ndarray::azip!((flux in &mut k[i].ex_mut(), &ax in &tmp.1, &by in &tmp.3)
                    *flux = ax + by
                );
            }

            {
                // hz = -ey_x + ex_y
                ndarray::azip!((a in &mut tmp.0,
                                &dxi_dx in &grid.detj_dxi_dx,
                                &dxi_dy in &grid.detj_dxi_dy,
                                &ex in &y.ex(),
                                &ey in &y.ey())
                    *a = dxi_dx * -ey + dxi_dy * ex
                );
                SBP::diffxi(tmp.0.view(), tmp.1.view_mut());

                ndarray::azip!((b in &mut tmp.2,
                                &deta_dx in &grid.detj_deta_dx,
                                &deta_dy in &grid.detj_deta_dy,
                                &ex in &y.ex(),
                                &ey in &y.ey())
                    *b = deta_dx * -ey + deta_dy * ex
                );
                SBP::diffeta(tmp.2.view(), tmp.3.view_mut());

                ndarray::azip!((flux in &mut k[i].hz_mut(), &ax in &tmp.1, &by in &tmp.3)
                    *flux = ax + by
                );
            }

            // ey = -hz_x
            {
                ndarray::azip!((a in &mut tmp.0,
                                &dxi_dx in &grid.detj_dxi_dx,
                                &hz in &y.hz())
                    *a = dxi_dx * -hz
                );
                SBP::diffxi(tmp.0.view(), tmp.1.view_mut());

                azip!((b in &mut tmp.2,
                                &deta_dx in &grid.detj_deta_dx,
                                &hz in &y.hz())
                    *b = deta_dx * -hz
                );
                SBP::diffeta(tmp.2.view(), tmp.3.view_mut());

                azip!((flux in &mut k[i].ey_mut(), &ax in &tmp.1, &by in &tmp.3)
                    *flux = ax + by
                );
            }

            // Boundary conditions (SAT)
            let ny = self.ny();
            let nx = self.nx();

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
                let g = (y.ex()[(j, 0)], y.hz()[(j, 0)], y.ey()[(j, 0)]);
                let v = (
                    y.ex()[(j, nx - 1)],
                    y.hz()[(j, nx - 1)],
                    y.ey()[(j, nx - 1)],
                );

                let kx = grid.detj_dxi_dx[(j, nx - 1)];
                let ky = grid.detj_dxi_dy[(j, nx - 1)];

                let plus = positive_flux(kx, ky);

                k[i].ex_mut()[(j, nx - 1)] += tau
                    * hinv
                    * (plus[0][0] * (v.0 - g.0)
                        + plus[0][1] * (v.1 - g.1)
                        + plus[0][2] * (v.2 - g.2));
                k[i].hz_mut()[(j, nx - 1)] += tau
                    * hinv
                    * (plus[1][0] * (v.0 - g.0)
                        + plus[1][1] * (v.1 - g.1)
                        + plus[1][2] * (v.2 - g.2));
                k[i].ey_mut()[(j, nx - 1)] += tau
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

                k[i].ex_mut()[(j, 0)] += tau
                    * hinv
                    * (minus[0][0] * (v.0 - g.0)
                        + minus[0][1] * (v.1 - g.1)
                        + minus[0][2] * (v.2 - g.2));
                k[i].hz_mut()[(j, 0)] += tau
                    * hinv
                    * (minus[1][0] * (v.0 - g.0)
                        + minus[1][1] * (v.1 - g.1)
                        + minus[1][2] * (v.2 - g.2));
                k[i].ey_mut()[(j, 0)] += tau
                    * hinv
                    * (minus[2][0] * (v.0 - g.0)
                        + minus[2][1] * (v.1 - g.1)
                        + minus[2][2] * (v.2 - g.2));
            }

            let hinv = 1.0 / (SBP::h()[0] / (ny - 1) as f32);

            for j in 0..nx {
                // North boundary, positive flux
                let tau = -1.0;
                let g = (y.ex()[(0, j)], y.hz()[(0, j)], y.ey()[(0, j)]);
                let v = (
                    y.ex()[(ny - 1, j)],
                    y.hz()[(ny - 1, j)],
                    y.ey()[(ny - 1, j)],
                );

                let kx = grid.detj_deta_dx[(ny - 1, j)];
                let ky = grid.detj_deta_dy[(ny - 1, j)];

                let plus = positive_flux(kx, ky);

                k[i].ex_mut()[(ny - 1, j)] += tau
                    * hinv
                    * (plus[0][0] * (v.0 - g.0)
                        + plus[0][1] * (v.1 - g.1)
                        + plus[0][2] * (v.2 - g.2));
                k[i].hz_mut()[(ny - 1, j)] += tau
                    * hinv
                    * (plus[1][0] * (v.0 - g.0)
                        + plus[1][1] * (v.1 - g.1)
                        + plus[1][2] * (v.2 - g.2));
                k[i].ey_mut()[(ny - 1, j)] += tau
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

                k[i].ex_mut()[(0, j)] += tau
                    * hinv
                    * (minus[0][0] * (v.0 - g.0)
                        + minus[0][1] * (v.1 - g.1)
                        + minus[0][2] * (v.2 - g.2));
                k[i].hz_mut()[(0, j)] += tau
                    * hinv
                    * (minus[1][0] * (v.0 - g.0)
                        + minus[1][1] * (v.1 - g.1)
                        + minus[1][2] * (v.2 - g.2));
                k[i].ey_mut()[(0, j)] += tau
                    * hinv
                    * (minus[2][0] * (v.0 - g.0)
                        + minus[2][1] * (v.1 - g.1)
                        + minus[2][2] * (v.2 - g.2));
            }

            azip!((k in &mut k[i].0,
                            &detj in &grid.detj.broadcast((3, ny, nx)).unwrap()) {
                *k /= detj;
            });
        }

        Zip::from(&mut fut.0)
            .and(&self.0)
            .and(&*k[0])
            .and(&*k[1])
            .and(&*k[2])
            .and(&*k[3])
            .apply(|y1, &y0, &k1, &k2, &k3, &k4| {
                *y1 = y0 + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            });
    }
}

pub struct WorkBuffers {
    y: Field,
    buf: [Field; 4],
    tmp: (Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>),
}

impl WorkBuffers {
    pub fn new(nx: usize, ny: usize) -> Self {
        let arr2 = Array2::zeros((ny, nx));
        let arr3 = Field::new(nx, ny);
        Self {
            y: arr3.clone(),
            buf: [arr3.clone(), arr3.clone(), arr3.clone(), arr3],
            tmp: (arr2.clone(), arr2.clone(), arr2.clone(), arr2),
        }
    }
}
