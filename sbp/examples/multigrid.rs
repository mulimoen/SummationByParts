use ndarray::prelude::*;
use sbp::*;

/*
 * A 2D grid divided in four parts, spanning the rectangle [-5, 5] x [-5, 5]
 *
 * / \ y
 *  |
 *  | 5.0   000     333
 *  |       000     333
 *  |       000     333
 *  | 0.0
 *  |       111     222
 *  |       111     222
 *  |-5.0   111     222
 *  |
 *  |     -5.0  0.0  5.0       x
 *    ------------------------->
 */

struct System<T: operators::UpwindOperator> {
    fnow: Vec<euler::Field>,
    fnext: Vec<euler::Field>,
    wb: Vec<(
        euler::Field,
        euler::Field,
        euler::Field,
        euler::Field,
        euler::Field,
        euler::Field,
    )>,
    k: [Vec<euler::Field>; 4],
    grids: Vec<grid::Grid<T>>,
}

impl<T: operators::UpwindOperator> System<T> {
    fn new(grids: Vec<grid::Grid<T>>) -> Self {
        let fnow = grids
            .iter()
            .map(|g| euler::Field::new(g.ny(), g.nx()))
            .collect::<Vec<_>>();
        let fnext = fnow.clone();
        let wb = grids
            .iter()
            .map(|g| {
                let f = euler::Field::new(g.ny(), g.nx());
                (f.clone(), f.clone(), f.clone(), f.clone(), f.clone(), f)
            })
            .collect();
        let k = [fnow.clone(), fnow.clone(), fnow.clone(), fnow.clone()];

        Self {
            fnow,
            fnext,
            k,
            wb,
            grids,
        }
    }

    fn vortex(&mut self, t: Float, vortex_params: euler::VortexParameters) {
        for (f, g) in self.fnow.iter_mut().zip(&self.grids) {
            f.vortex(g.x(), g.y(), t, vortex_params);
        }
    }

    fn advance(&mut self, dt: Float) {
        for i in 0.. {
            let fnext;
            match i {
                0 => {
                    for (prev, fut) in self.fnow.iter().zip(self.fnext.iter_mut()) {
                        fut.assign(prev);
                    }
                    fnext = &mut self.k[i];
                }
                1 | 2 => {
                    for ((prev, fut), k) in self
                        .fnow
                        .iter()
                        .zip(self.fnext.iter_mut())
                        .zip(&self.k[i - 1])
                    {
                        fut.assign(prev);
                        fut.scaled_add(1.0 / 2.0 * dt, k);
                    }
                    fnext = &mut self.k[i];
                }
                3 => {
                    for ((prev, fut), k) in self
                        .fnow
                        .iter()
                        .zip(self.fnext.iter_mut())
                        .zip(&self.k[i - 1])
                    {
                        fut.assign(prev);
                        fut.scaled_add(dt, k);
                    }
                    fnext = &mut self.k[i];
                }
                4 => {
                    for (((((prev, fut), k0), k1), k2), k3) in self
                        .fnow
                        .iter()
                        .zip(self.fnext.iter_mut())
                        .zip(&self.k[0])
                        .zip(&self.k[1])
                        .zip(&self.k[2])
                        .zip(&self.k[3])
                    {
                        ndarray::Zip::from(&mut **fut)
                            .and(&**prev)
                            .and(&**k0)
                            .and(&**k1)
                            .and(&**k2)
                            .and(&**k3)
                            .apply(|y1, &y0, &k1, &k2, &k3, &k4| {
                                *y1 = y0 + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                            });
                    }
                    return;
                }
                _ => {
                    unreachable!();
                }
            }

            let bt = vec![
                euler::BoundaryTerms {
                    north: self.fnext[1].north(),
                    south: self.fnext[1].south(),
                    east: self.fnext[3].west(),
                    west: self.fnext[3].east(),
                },
                euler::BoundaryTerms {
                    north: self.fnext[0].north(),
                    south: self.fnext[0].south(),
                    east: self.fnext[2].west(),
                    west: self.fnext[2].east(),
                },
                euler::BoundaryTerms {
                    north: self.fnext[3].north(),
                    south: self.fnext[3].south(),
                    east: self.fnext[0].west(),
                    west: self.fnext[0].east(),
                },
                euler::BoundaryTerms {
                    north: self.fnext[2].north(),
                    south: self.fnext[2].south(),
                    east: self.fnext[1].west(),
                    west: self.fnext[1].east(),
                },
            ];

            for ((((prev, fut), grid), wb), bt) in self
                .fnext
                .iter()
                .zip(fnext)
                .zip(&self.grids)
                .zip(&mut self.wb)
                .zip(bt)
            {
                euler::RHS_upwind(fut, prev, grid, &bt, wb)
            }
        }
    }
}

fn mesh(x: (f64, f64, usize), y: (f64, f64, usize)) -> (Array2<f64>, Array2<f64>) {
    let arrx = Array1::linspace(x.0, x.1, x.2);
    let arry = Array1::linspace(y.0, y.1, y.2);

    let gx = arrx.broadcast((y.2, x.2)).unwrap();
    let mut gy = arry.broadcast((x.2, y.2)).unwrap();
    gy.swap_axes(0, 1);

    (gx.into_owned(), gy.into_owned())
}

fn main() {
    let n = 20;

    let mut grids = Vec::with_capacity(4);

    let (x0, y0) = mesh((-5.0, 0.0, n), (0.0, 5.0, n));
    grids.push(grid::Grid::<operators::Upwind4>::new(x0, y0).unwrap());

    let (x1, y1) = mesh((-5.0, 0.0, n), (-5.0, 0.0, n));
    grids.push(grid::Grid::<operators::Upwind4>::new(x1, y1).unwrap());

    let (x2, y2) = mesh((0.0, 5.0, n), (-5.0, 0.0, n));
    grids.push(grid::Grid::<operators::Upwind4>::new(x2, y2).unwrap());

    let (x3, y3) = mesh((0.0, 5.0, n), (0.0, 5.0, n));
    grids.push(grid::Grid::<operators::Upwind4>::new(x3, y3).unwrap());

    let mut sys = System::new(grids);
    sys.vortex(
        0.0,
        euler::VortexParameters {
            x0: 0.0,
            y0: 0.0,
            mach: 0.5,
            rstar: 0.5,
            eps: 1.0,
        },
    );
    sys.advance(0.05);

    /*
    let bt0 = euler::BoundaryTerms {
        north: sys1.field().north(),
        south: sys1.field().south(),
        east: sys3.field().west(),
        west: sys3.field().east(),
    };
    let bt1 = euler::BoundaryTerms {
        north: sys0.field().north(),
        south: sys0.field().south(),
        east: sys2.field().west(),
        west: sys2.field().east(),
    };
    let bt2 = euler::BoundaryTerms {
        north: sys3.field().north(),
        south: sys3.field().south(),
        east: sys0.field().west(),
        west: sys0.field().east(),
    };
    let bt3 = euler::BoundaryTerms {
        north: sys2.field().north(),
        south: sys2.field().south(),
        east: sys1.field().west(),
        west: sys1.field().east(),
    };
    */
}
