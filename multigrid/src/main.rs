#![feature(str_strip)]
use sbp::utils::json_to_grids;
use sbp::*;
use structopt::StructOpt;

struct System {
    fnow: Vec<euler::Field>,
    fnext: Vec<euler::Field>,
    wb: Vec<euler::WorkBuffers>,
    k: [Vec<euler::Field>; 4],
    grids: Vec<grid::Grid>,
    metrics: Vec<Metrics>,
    bt: Vec<euler::BoundaryCharacteristics>,
    eb: Vec<euler::BoundaryStorage>,
    time: Float,
}

enum Metrics {
    Upwind4(grid::Metrics<operators::Upwind4, operators::Upwind4>),
    Upwind9(grid::Metrics<operators::Upwind9, operators::Upwind9>),
    Upwind4h2(grid::Metrics<operators::Upwind4h2, operators::Upwind4h2>),
    Trad4(grid::Metrics<operators::SBP4, operators::SBP4>),
    Trad8(grid::Metrics<operators::SBP8, operators::SBP8>),

    Upwind4Upwind4h2(grid::Metrics<operators::Upwind4, operators::Upwind4h2>),
    Upwind4h2Upwind4(grid::Metrics<operators::Upwind4h2, operators::Upwind4>),
}

impl System {
    fn new(
        grids: Vec<grid::Grid>,
        bt: Vec<euler::BoundaryCharacteristics>,
        operatorx: &str,
        operatory: &str,
    ) -> Self {
        let fnow = grids
            .iter()
            .map(|g| euler::Field::new(g.ny(), g.nx()))
            .collect::<Vec<_>>();
        let fnext = fnow.clone();
        let wb = grids
            .iter()
            .map(|g| euler::WorkBuffers::new(g.ny(), g.nx()))
            .collect();
        let k = [fnow.clone(), fnow.clone(), fnow.clone(), fnow.clone()];
        let metrics = grids
            .iter()
            .map(|g| match (operatorx, operatory) {
                ("upwind4", "upwind4") => Metrics::Upwind4(
                    g.metrics::<operators::Upwind4, operators::Upwind4>()
                        .unwrap(),
                ),
                ("upwind9", "upwind9") => Metrics::Upwind9(
                    g.metrics::<operators::Upwind9, operators::Upwind9>()
                        .unwrap(),
                ),
                ("upwind4h2", "upwind4h2") => Metrics::Upwind4h2(
                    g.metrics::<operators::Upwind4h2, operators::Upwind4h2>()
                        .unwrap(),
                ),
                ("trad4", "trad4") => {
                    Metrics::Trad4(g.metrics::<operators::SBP4, operators::SBP4>().unwrap())
                }
                ("trad8", "trad8") => {
                    Metrics::Trad8(g.metrics::<operators::SBP8, operators::SBP8>().unwrap())
                }
                ("upwind4", "upwind4h2") => Metrics::Upwind4Upwind4h2(
                    g.metrics::<operators::Upwind4, operators::Upwind4h2>()
                        .unwrap(),
                ),
                ("upwind4h2", "upwind4") => Metrics::Upwind4h2Upwind4(
                    g.metrics::<operators::Upwind4h2, operators::Upwind4>()
                        .unwrap(),
                ),
                (opx, opy) => panic!("operator combination {}x{} not known", opx, opy),
            })
            .collect::<Vec<_>>();

        let eb = bt
            .iter()
            .zip(&grids)
            .map(|(bt, grid)| euler::BoundaryStorage::new(bt, grid))
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

    fn advance(&mut self, dt: Float, pool: &rayon::ThreadPool) {
        type MT<'a> = (
            &'a mut [euler::WorkBuffers],
            &'a mut [euler::BoundaryStorage],
        );
        let metrics = &self.metrics;
        let rhs = move |fut: &mut [euler::Field],
                        prev: &[euler::Field],
                        time: Float,
                        c: &(&[grid::Grid], &[euler::BoundaryCharacteristics]),
                        mt: &mut MT| {
            let (grids, bt) = c;
            let (wb, eb) = mt;

            let bc = euler::extract_boundaries::<operators::Interpolation4>(
                prev, *bt, *eb, *grids, time,
            );
            pool.scope(|s| {
                for ((((fut, prev), bc), wb), metrics) in fut
                    .iter_mut()
                    .zip(prev.iter())
                    .zip(bc)
                    .zip(wb.iter_mut())
                    .zip(metrics.iter())
                {
                    s.spawn(move |_| match metrics {
                        Metrics::Upwind4(metrics) => {
                            euler::RHS_upwind(fut, prev, metrics, &bc, &mut wb.0)
                        }
                        Metrics::Upwind9(metrics) => {
                            euler::RHS_upwind(fut, prev, metrics, &bc, &mut wb.0)
                        }
                        Metrics::Upwind4h2(metrics) => {
                            euler::RHS_upwind(fut, prev, metrics, &bc, &mut wb.0)
                        }
                        Metrics::Trad4(metrics) => {
                            euler::RHS_trad(fut, prev, metrics, &bc, &mut wb.0)
                        }
                        Metrics::Trad8(metrics) => {
                            euler::RHS_trad(fut, prev, metrics, &bc, &mut wb.0)
                        }
                        Metrics::Upwind4Upwind4h2(metrics) => {
                            euler::RHS_trad(fut, prev, metrics, &bc, &mut wb.0)
                        }
                        Metrics::Upwind4h2Upwind4(metrics) => {
                            euler::RHS_trad(fut, prev, metrics, &bc, &mut wb.0)
                        }
                    });
                }
            });
        };

        let mut k = self
            .k
            .iter_mut()
            .map(|k| k.as_mut_slice())
            .collect::<Vec<_>>();
        sbp::integrate::integrate_multigrid::<sbp::integrate::Rk4, _, _, _, _>(
            rhs,
            &self.fnow,
            &mut self.fnext,
            &mut self.time,
            dt,
            &mut k,
            &(&self.grids, &self.bt),
            &mut (&mut self.wb, &mut self.eb),
            pool,
        );

        std::mem::swap(&mut self.fnow, &mut self.fnext);
    }
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
    #[structopt(default_value = "output.hdf", long, short)]
    output: std::path::PathBuf,
    /// Number of outputs to save
    #[structopt(long, short)]
    number_of_outputs: Option<u64>,
    /// Print the time to complete, taken in the compute loop
    #[structopt(long)]
    timings: bool,
    /// Print error at the end of the run
    #[structopt(long)]
    error: bool,
}

fn main() {
    type SBP = operators::Upwind4;
    let opt = Options::from_args();
    let filecontents = std::fs::read_to_string(&opt.json).unwrap();

    let json = json::parse(&filecontents).unwrap();
    let jgrids = json_to_grids(json["grids"].clone()).unwrap();
    let vortexparams = utils::json_to_vortex(json["vortex"].clone());

    let mut bt = Vec::with_capacity(jgrids.len());
    let determine_bc = |dir: Option<&String>| match dir {
        Some(dir) => {
            if dir == "vortex" {
                euler::BoundaryCharacteristic::Vortex(vortexparams)
            } else if let Some(grid) = dir.strip_prefix("interpolate:") {
                euler::BoundaryCharacteristic::Interpolate(
                    jgrids
                        .iter()
                        .position(|other| other.name.as_ref().map_or(false, |name| name == grid))
                        .unwrap(),
                )
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

    let (operatorx, operatory) = {
        if json["operator"].is_object() {
            (
                json["operator"]["x"].as_str().unwrap(),
                json["operator"]["y"].as_str().unwrap(),
            )
        } else {
            let op = json["operator"].as_str().unwrap_or("upwind4");
            (op, op)
        }
    };

    let mut sys = System::new(grids, bt, operatorx, operatory);
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

    let progressbar = progressbar(opt.no_progressbar, ntime);

    let timer = if opt.timings {
        Some(std::time::Instant::now())
    } else {
        None
    };

    for itime in 0..ntime {
        if should_output(itime) {
            output.add_timestep(itime, &sys.fnow);
        }
        progressbar.inc(1);
        sys.advance(dt, &pool);
    }
    progressbar.finish_and_clear();

    if let Some(timer) = timer {
        let duration = timer.elapsed();
        println!("Time elapsed: {} seconds", duration.as_secs_f64());
    }

    output.add_timestep(ntime, &sys.fnow);
    if opt.error {
        let time = ntime as Float * dt;
        let mut e = 0.0;
        for (fmod, grid) in sys.fnow.iter().zip(&sys.grids) {
            let mut fvort = fmod.clone();
            fvort.vortex(grid.x(), grid.y(), time, vortexparams);
            e += fmod.h2_err::<SBP>(&fvort);
        }
        println!("Total error: {:e}", e);
    }
}

fn progressbar(dummy: bool, ntime: u64) -> indicatif::ProgressBar {
    if dummy {
        indicatif::ProgressBar::hidden()
    } else {
        let progressbar = indicatif::ProgressBar::new(ntime);
        progressbar.with_style(
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
