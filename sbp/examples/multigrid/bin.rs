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
    grids: Vec<grid::Grid>,
    metrics: Vec<grid::Metrics<T>>,
    bt: Vec<euler::BoundaryCharacteristics>,
    eb: Vec<MaybeBoundary>,
    time: Float,
}

impl<T: operators::UpwindOperator> System<T> {
    fn new(grids: Vec<grid::Grid>, bt: Vec<euler::BoundaryCharacteristics>) -> Self {
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
        let metrics = grids.iter().map(|g| g.metrics().unwrap()).collect();
        let eb = bt
            .iter()
            .zip(&grids)
            .map(|(bt, grid)| MaybeBoundary {
                n: match bt.north {
                    euler::BoundaryCharacteristic::Vortex(_) => {
                        Some(ndarray::Array2::zeros((4, grid.nx())))
                    }
                    _ => None,
                },
                s: match bt.north {
                    euler::BoundaryCharacteristic::Vortex(_) => {
                        Some(ndarray::Array2::zeros((4, grid.nx())))
                    }
                    _ => None,
                },
                e: match bt.north {
                    euler::BoundaryCharacteristic::Vortex(_) => {
                        Some(ndarray::Array2::zeros((4, grid.ny())))
                    }
                    _ => None,
                },
                w: match bt.north {
                    euler::BoundaryCharacteristic::Vortex(_) => {
                        Some(ndarray::Array2::zeros((4, grid.ny())))
                    }
                    _ => None,
                },
            })
            .collect();

        Self {
            fnow,
            fnext,
            k,
            wb,
            grids,
            metrics,
            bt,
            eb,
            time: 0.0,
        }
    }

    fn vortex(&mut self, t: Float, vortex_params: euler::VortexParameters) {
        for (f, g) in self.fnow.iter_mut().zip(&self.grids) {
            f.vortex(g.x(), g.y(), t, vortex_params);
        }
    }

    fn advance(&mut self, dt: Float, s: &rayon::ThreadPool) {
        for i in 0.. {
            let time;
            match i {
                0 => {
                    s.scope(|s| {
                        for (prev, fut) in self.fnow.iter().zip(self.fnext.iter_mut()) {
                            s.spawn(move |_| {
                                fut.assign(prev);
                            });
                        }
                    });
                    time = self.time;
                }
                1 | 2 => {
                    s.scope(|s| {
                        for ((prev, fut), k) in self
                            .fnow
                            .iter()
                            .zip(self.fnext.iter_mut())
                            .zip(&self.k[i - 1])
                        {
                            s.spawn(move |_| {
                                fut.assign(prev);
                                fut.scaled_add(1.0 / 2.0 * dt, k);
                            });
                        }
                    });
                    time = self.time + dt / 2.0;
                }
                3 => {
                    s.scope(|s| {
                        for ((prev, fut), k) in self
                            .fnow
                            .iter()
                            .zip(self.fnext.iter_mut())
                            .zip(&self.k[i - 1])
                        {
                            s.spawn(move |_| {
                                fut.assign(prev);
                                fut.scaled_add(dt, k);
                            });
                        }
                    });
                    time = self.time + dt;
                }
                4 => {
                    s.scope(|s| {
                        for (((((prev, fut), k0), k1), k2), k3) in self
                            .fnow
                            .iter()
                            .zip(self.fnext.iter_mut())
                            .zip(&self.k[0])
                            .zip(&self.k[1])
                            .zip(&self.k[2])
                            .zip(&self.k[3])
                        {
                            s.spawn(move |_| {
                                ndarray::Zip::from(&mut **fut)
                                    .and(&**prev)
                                    .and(&**k0)
                                    .and(&**k1)
                                    .and(&**k2)
                                    .and(&**k3)
                                    .apply(|y1, &y0, &k1, &k2, &k3, &k4| {
                                        *y1 = y0 + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                                    });
                            });
                        }
                    });
                    std::mem::swap(&mut self.fnext, &mut self.fnow);
                    self.time += dt;
                    return;
                }
                _ => {
                    unreachable!();
                }
            }

            s.scope(|s| {
                let fields = &self.fnext;
                let bt = extract_boundaries(&fields, &mut self.bt, &mut self.eb, &self.grids, time);
                for ((((prev, fut), metrics), wb), bt) in fields
                    .iter()
                    .zip(&mut self.k[i])
                    .zip(&self.metrics)
                    .zip(&mut self.wb)
                    .zip(bt)
                {
                    s.spawn(move |_| euler::RHS_upwind(fut, prev, metrics, &bt, wb));
                }
            });
        }
    }
}

fn extract_boundaries<'a>(
    fields: &'a [euler::Field],
    bt: &[euler::BoundaryCharacteristics],
    eb: &'a mut [MaybeBoundary],
    grids: &[grid::Grid],
    time: Float,
) -> Vec<euler::BoundaryTerms<'a>> {
    bt.iter()
        .zip(eb)
        .zip(grids)
        .zip(fields)
        .map(|(((bt, eb), grid), field)| euler::BoundaryTerms {
            north: match bt.north {
                euler::BoundaryCharacteristic::This => field.south(),
                euler::BoundaryCharacteristic::Grid(g) => fields[g].south(),
                euler::BoundaryCharacteristic::Vortex(v) => {
                    let field = eb.n.as_mut().unwrap();
                    vortexify(field.view_mut(), grid.north(), v, time);
                    field.view()
                }
            },
            south: match bt.south {
                euler::BoundaryCharacteristic::This => field.north(),
                euler::BoundaryCharacteristic::Grid(g) => fields[g].north(),
                euler::BoundaryCharacteristic::Vortex(v) => {
                    let field = eb.s.as_mut().unwrap();
                    vortexify(field.view_mut(), grid.south(), v, time);
                    field.view()
                }
            },
            west: match bt.west {
                euler::BoundaryCharacteristic::This => field.east(),
                euler::BoundaryCharacteristic::Grid(g) => fields[g].east(),
                euler::BoundaryCharacteristic::Vortex(v) => {
                    let field = eb.w.as_mut().unwrap();
                    vortexify(field.view_mut(), grid.west(), v, time);
                    field.view()
                }
            },
            east: match bt.east {
                euler::BoundaryCharacteristic::This => field.west(),
                euler::BoundaryCharacteristic::Grid(g) => fields[g].west(),
                euler::BoundaryCharacteristic::Vortex(v) => {
                    let field = eb.e.as_mut().unwrap();
                    vortexify(field.view_mut(), grid.east(), v, time);
                    field.view()
                }
            },
        })
        .collect()
}

#[derive(Debug, Clone)]
struct MaybeBoundary {
    n: Option<ndarray::Array2<Float>>,
    s: Option<ndarray::Array2<Float>>,
    e: Option<ndarray::Array2<Float>>,
    w: Option<ndarray::Array2<Float>>,
}

fn vortexify(
    mut field: ndarray::ArrayViewMut2<Float>,
    yx: (ndarray::ArrayView1<Float>, ndarray::ArrayView1<Float>),
    v: euler::VortexParameters,
    t: Float,
) {
    let mut fiter = field.outer_iter_mut();
    let (rho, rhou, rhov, e) = (
        fiter.next().unwrap(),
        fiter.next().unwrap(),
        fiter.next().unwrap(),
        fiter.next().unwrap(),
    );
    let (y, x) = yx;
    euler::vortex(rho, rhou, rhov, e, x, y, t, v);
}

#[derive(Debug, StructOpt)]
struct Options {
    json: std::path::PathBuf,
    /// Disable the progressbar
    #[structopt(long)]
    no_progressbar: bool,
    /// Number of simultaneous threads
    #[structopt(short, long)]
    jobs: Option<Option<usize>>,
    /// Name of output file
    #[structopt(default_value = "output.hdf")]
    output: std::path::PathBuf,
    /// Number of outputs to save
    #[structopt(long, short)]
    number_of_outputs: Option<u64>,
    /// Print the time to complete, taken in the compute loop
    #[structopt(long)]
    timings: bool,
}

fn main() {
    let opt = Options::from_args();
    let filecontents = std::fs::read_to_string(&opt.json).unwrap();

    let json = json::parse(&filecontents).unwrap();
    let jgrids = json_to_grids(json["grids"].clone()).unwrap();
    let vortexparams = utils::json_to_vortex(json["vortex"].clone());

    let mut bt = Vec::with_capacity(jgrids.len());
    let determine_bc = |dir| match dir {
        Some(dir) => {
            if dir == "vortex" {
                euler::BoundaryCharacteristic::Vortex(vortexparams)
            } else {
                euler::BoundaryCharacteristic::Grid(
                    jgrids
                        .iter()
                        .position(|other| other.name.as_ref().map_or(false, |name| name == dir))
                        .unwrap(),
                )
            }
        }
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
    let grids = jgrids.into_iter().map(|egrid| egrid.grid).collect();

    let integration_time: Float = json["integration_time"].as_number().unwrap().into();

    let mut sys = System::<sbp::operators::Upwind4>::new(grids, bt);
    sys.vortex(0.0, vortexparams);

    let max_n = {
        let max_nx = sys.grids.iter().map(|g| g.nx()).max().unwrap();
        let max_ny = sys.grids.iter().map(|g| g.ny()).max().unwrap();
        std::cmp::max(max_nx, max_ny)
    };
    let dt = 0.2 / (max_n as Float);

    let ntime = (integration_time / dt).round() as u64;

    let pool = {
        let builder = rayon::ThreadPoolBuilder::new();
        if let Some(j) = opt.jobs {
            if let Some(j) = j {
                builder.num_threads(j)
            } else {
                builder
            }
        } else {
            builder.num_threads(1)
        }
        .build()
        .unwrap()
    };

    let should_output = |itime| {
        opt.number_of_outputs.map_or(false, |num_out| {
            if num_out == 0 {
                false
            } else {
                itime % (std::cmp::max(ntime / (num_out - 1), 1)) == 0
            }
        })
    };

    let output = File::create(&opt.output, sys.grids.as_slice()).unwrap();
    let mut output = OutputThread::new(output);

    let bar = progressbar(opt.no_progressbar, ntime);

    let timer = if opt.timings {
        Some(std::time::Instant::now())
    } else {
        None
    };

    for itime in 0..ntime {
        if should_output(itime) {
            output.add_timestep(itime, &sys.fnow);
        }
        bar.inc(1);
        sys.advance(dt, &pool);
    }
    bar.finish();

    if let Some(timer) = timer {
        let duration = timer.elapsed();
        println!("Time elapsed: {} seconds", duration.as_secs_f64());
    }

    output.add_timestep(ntime, &sys.fnow);
}

fn progressbar(dummy: bool, ntime: u64) -> indicatif::ProgressBar {
    if dummy {
        indicatif::ProgressBar::hidden()
    } else {
        let bar = indicatif::ProgressBar::new(ntime);
        bar.with_style(
            indicatif::ProgressStyle::default_bar()
                .template("{wide_bar:.cyan/blue} {pos}/{len} ({eta})"),
        )
    }
}

struct OutputThread {
    rx: Option<std::sync::mpsc::Receiver<Vec<euler::Field>>>,
    tx: Option<std::sync::mpsc::SyncSender<(u64, Vec<euler::Field>)>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl OutputThread {
    fn new(file: File) -> Self {
        // Pingpong back and forth a number of Vec<Field> to be used for the
        // output. The sync_channel applies some backpressure
        let (tx_thread, rx) = std::sync::mpsc::channel::<Vec<euler::Field>>();
        let (tx, rx_thread) = std::sync::mpsc::sync_channel::<(u64, Vec<euler::Field>)>(3);
        let thread = std::thread::Builder::new()
            .name("multigrid_output".to_owned())
            .spawn(move || {
                let mut times = Vec::<u64>::new();

                for (ntime, fields) in rx_thread.iter() {
                    if !times.contains(&ntime) {
                        file.add_timestep(ntime, fields.as_slice()).unwrap();
                        times.push(ntime);
                    }
                    tx_thread.send(fields).unwrap();
                }
            })
            .unwrap();

        Self {
            tx: Some(tx),
            rx: Some(rx),
            thread: Some(thread),
        }
    }

    fn add_timestep(&mut self, ntime: u64, fields: &[euler::Field]) {
        match self.rx.as_ref().unwrap().try_recv() {
            Ok(mut copy_fields) => {
                for (from, to) in fields.iter().zip(copy_fields.iter_mut()) {
                    to.assign(&from);
                }
                self.tx
                    .as_ref()
                    .unwrap()
                    .send((ntime, copy_fields))
                    .unwrap();
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => {
                let fields = fields.to_vec();
                self.tx.as_ref().unwrap().send((ntime, fields)).unwrap();
            }
            Err(e) => panic!("{:?}", e),
        };
    }
}

impl Drop for OutputThread {
    fn drop(&mut self) {
        let tx = self.tx.take();
        std::mem::drop(tx);
        let thread = self.thread.take().unwrap();
        thread.join().unwrap();
    }
}

#[derive(Debug, Clone)]
struct File(hdf5::File);

impl File {
    fn create<P: AsRef<std::path::Path>>(
        path: P,
        grids: &[sbp::grid::Grid],
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let file = hdf5::File::create(path.as_ref())?;
        let _tds = file
            .new_dataset::<u64>()
            .resizable(true)
            .chunk((1,))
            .create("t", (0,))?;

        for (i, grid) in grids.iter().enumerate() {
            let g = file.create_group(&i.to_string())?;
            g.link_soft("/t", "t").unwrap();

            let add_dim = |name| {
                g.new_dataset::<Float>()
                    .chunk((grid.ny(), grid.nx()))
                    .gzip(9)
                    .create(name, (grid.ny(), grid.nx()))
            };
            let xds = add_dim("x")?;
            xds.write(grid.x())?;
            let yds = add_dim("y")?;
            yds.write(grid.y())?;

            let add_var = |name| {
                g.new_dataset::<Float>()
                    .gzip(3)
                    .shuffle(true)
                    .chunk((1, grid.ny(), grid.nx()))
                    .resizable_idx(&[true, false, false])
                    .create(name, (0, grid.ny(), grid.nx()))
            };
            add_var("rho")?;
            add_var("rhou")?;
            add_var("rhov")?;
            add_var("e")?;
        }

        Ok(Self(file))
    }

    fn add_timestep(
        &self,
        t: u64,
        fields: &[euler::Field],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let file = &self.0;
        let tds = file.dataset("t")?;
        let tpos = tds.size();
        tds.resize((tpos + 1,))?;
        tds.write_slice(&[t], ndarray::s![tpos..tpos + 1])?;

        for (i, fnow) in fields.iter().enumerate() {
            let g = file.group(&i.to_string())?;
            let (tpos, ny, nx) = {
                let ds = g.dataset("rho")?;
                let shape = ds.shape();
                (shape[0], shape[1], shape[2])
            };

            let rhods = g.dataset("rho")?;
            let rhouds = g.dataset("rhou")?;
            let rhovds = g.dataset("rhov")?;
            let eds = g.dataset("e")?;

            let (rho, rhou, rhov, e) = fnow.components();
            rhods.resize((tpos + 1, ny, nx))?;
            rhods.write_slice(rho, ndarray::s![tpos, .., ..])?;

            rhouds.resize((tpos + 1, ny, nx))?;
            rhouds.write_slice(rhou, ndarray::s![tpos, .., ..])?;

            rhovds.resize((tpos + 1, ny, nx))?;
            rhovds.write_slice(rhov, ndarray::s![tpos, .., ..])?;

            eds.resize((tpos + 1, ny, nx))?;
            eds.write_slice(e, ndarray::s![tpos, .., ..])?;
        }
        Ok(())
    }
}
