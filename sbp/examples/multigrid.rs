use sbp::utils::json_to_grids;
use sbp::*;
use structopt::StructOpt;

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
    bt: Vec<euler::BoundaryCharacteristics>,
}

impl<T: operators::UpwindOperator> System<T> {
    fn new(grids: Vec<grid::Grid<T>>, bt: Vec<euler::BoundaryCharacteristics>) -> Self {
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
            bt,
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
                    std::mem::swap(&mut self.fnext, &mut self.fnow);
                    return;
                }
                _ => {
                    unreachable!();
                }
            }

            let fields = &self.fnext;

            let bt = self
                .bt
                .iter()
                .enumerate()
                .map(|(i, bt)| euler::BoundaryTerms {
                    north: match bt.north {
                        euler::BoundaryCharacteristic::This => fields[i].south(),
                        euler::BoundaryCharacteristic::Grid(g) => fields[g].south(),
                        euler::BoundaryCharacteristic::Vortex(_) => todo!(),
                    },
                    south: match bt.south {
                        euler::BoundaryCharacteristic::This => fields[i].north(),
                        euler::BoundaryCharacteristic::Grid(g) => fields[g].north(),
                        euler::BoundaryCharacteristic::Vortex(_) => todo!(),
                    },
                    west: match bt.west {
                        euler::BoundaryCharacteristic::This => fields[i].east(),
                        euler::BoundaryCharacteristic::Grid(g) => fields[g].east(),
                        euler::BoundaryCharacteristic::Vortex(_) => todo!(),
                    },
                    east: match bt.east {
                        euler::BoundaryCharacteristic::This => fields[i].west(),
                        euler::BoundaryCharacteristic::Grid(g) => fields[g].west(),
                        euler::BoundaryCharacteristic::Vortex(_) => todo!(),
                    },
                })
                .collect::<Vec<_>>();

            for ((((prev, fut), grid), wb), bt) in fields
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

#[derive(Debug, StructOpt)]
struct Options {
    json: std::path::PathBuf,
    #[structopt(long, help = "Disable the progressbar")]
    no_progressbar: bool,
}

fn main() {
    let opt = Options::from_args();
    let filecontents = std::fs::read_to_string(&opt.json).unwrap();

    let json = json::parse(&filecontents).unwrap();
    let jgrids = json_to_grids(json["grids"].clone()).unwrap();

    let mut bt = Vec::with_capacity(jgrids.len());
    let determine_bc = |dir| match dir {
        Some(dir) => euler::BoundaryCharacteristic::Grid(
            jgrids
                .iter()
                .position(|other| other.name.as_ref().map_or(false, |name| name == dir))
                .unwrap(),
        ),
        None => euler::BoundaryCharacteristic::This,
    };
    for grid in &jgrids {
        bt.push(euler::BoundaryCharacteristics {
            north: determine_bc(grid.dirn.as_ref()),
            south: determine_bc(grid.dirs.as_ref()),
            east: determine_bc(grid.dire.as_ref()),
            west: determine_bc(grid.dirw.as_ref()),
        });
    }
    let mut grids: Vec<grid::Grid<operators::Upwind4>> = Vec::with_capacity(jgrids.len());
    for grid in jgrids {
        grids.push(grid::Grid::new(grid.x, grid.y).unwrap());
    }
    let integration_time: f64 = json["integration_time"].as_number().unwrap().into();

    let vortexparams = utils::json_to_vortex(json["vortex"].clone());

    let mut sys = System::new(grids, bt);
    sys.vortex(0.0, vortexparams);

    let max_n = {
        let max_nx = sys.grids.iter().map(|g| g.nx()).max().unwrap();
        let max_ny = sys.grids.iter().map(|g| g.ny()).max().unwrap();
        std::cmp::max(max_nx, max_ny)
    };
    let dt = 0.2 / (max_n as Float);

    let ntime = (integration_time / dt).round() as usize;

    let bar = if opt.no_progressbar {
        indicatif::ProgressBar::hidden()
    } else {
        let bar = indicatif::ProgressBar::new(ntime as _);
        bar.with_style(
            indicatif::ProgressStyle::default_bar()
                .template("{wide_bar:.cyan/blue} {pos}/{len} ({eta})"),
        )
    };
    for _ in 0..ntime {
        bar.inc(1);
        sys.advance(dt);
    }
    bar.finish();

    dump_to_file(&sys);
}

fn dump_to_file<T: sbp::operators::UpwindOperator>(sys: &System<T>) {
    use std::io::prelude::*;
    let file = std::fs::File::create("output").unwrap();
    let mut file = std::io::BufWriter::new(file);
    let ngrids = sys.grids.len();
    file.write_all(&(ngrids as u32).to_le_bytes()).unwrap();
    for (grid, s) in sys.grids.iter().zip(&sys.fnow) {
        file.write_all(&(grid.ny() as u32).to_le_bytes()).unwrap();
        file.write_all(&(grid.nx() as u32).to_le_bytes()).unwrap();
        for x in grid.x().as_slice().unwrap() {
            file.write_all(&(x.to_le_bytes())).unwrap();
        }
        for y in grid.y().as_slice().unwrap() {
            file.write_all(&(y.to_le_bytes())).unwrap();
        }
        for rho in s.rho().as_slice().unwrap() {
            file.write_all(&(rho.to_le_bytes())).unwrap();
        }
        for rhou in s.rhou().as_slice().unwrap() {
            file.write_all(&(rhou.to_le_bytes())).unwrap();
        }
        for rhov in s.rhov().as_slice().unwrap() {
            file.write_all(&(rhov.to_le_bytes())).unwrap();
        }
        for e in s.e().as_slice().unwrap() {
            file.write_all(&(e.to_le_bytes())).unwrap();
        }
    }
}
