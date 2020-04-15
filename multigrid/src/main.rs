#![feature(str_strip)]
use either::*;
use sbp::operators::{SbpOperator2d, UpwindOperator2d};
use sbp::utils::json_to_grids;
use sbp::*;
use structopt::StructOpt;

mod file;
use file::*;

struct System {
    fnow: Vec<euler::Field>,
    fnext: Vec<euler::Field>,
    wb: Vec<euler::WorkBuffers>,
    k: [Vec<euler::Field>; 4],
    grids: Vec<grid::Grid>,
    metrics: Vec<grid::Metrics>,
    bt: Vec<euler::BoundaryCharacteristics>,
    eb: Vec<euler::BoundaryStorage>,
    time: Float,
    operators: Vec<Either<Box<dyn SbpOperator2d>, Box<dyn UpwindOperator2d>>>,
    interpolation_operators: Vec<euler::InterpolationOperators>,
}

impl System {
    fn new(
        grids: Vec<grid::Grid>,
        bt: Vec<euler::BoundaryCharacteristics>,
        interpolation_operators: Vec<euler::InterpolationOperators>,
        operators: Vec<Either<Box<dyn SbpOperator2d>, Box<dyn UpwindOperator2d>>>,
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
            .zip(&operators)
            .map(|(g, op)| {
                let sbpop: &dyn SbpOperator2d = op.as_ref().either(|op| &**op, |uo| uo.as_sbp());
                g.metrics(sbpop).unwrap()
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
            interpolation_operators,
            operators,
        }
    }

    fn vortex(&mut self, t: Float, vortex_params: euler::VortexParameters) {
        for (f, g) in self.fnow.iter_mut().zip(&self.grids) {
            f.vortex(g.x(), g.y(), t, vortex_params);
        }
    }

    fn advance(&mut self, dt: Float, pool: &rayon::ThreadPool) {
        let metrics = &self.metrics;
        let grids = &self.grids;
        let bt = &self.bt;
        let wb = &mut self.wb;
        let mut eb = &mut self.eb;
        let intops = &self.interpolation_operators;
        let operators = &self.operators;

        let rhs = move |fut: &mut [euler::Field],
                        prev: &[euler::Field],
                        time: Float,
                        _c: (),
                        _mt: &mut ()| {
            let bc = euler::extract_boundaries(prev, &bt, &mut eb, &grids, time, Some(intops));
            pool.scope(|s| {
                for (((((fut, prev), bc), wb), metrics), op) in fut
                    .iter_mut()
                    .zip(prev.iter())
                    .zip(bc)
                    .zip(wb.iter_mut())
                    .zip(metrics.iter())
                    .zip(operators.iter())
                {
                    s.spawn(move |_| match op.as_ref() {
                        Left(sbp) => {
                            euler::RHS_trad(&**sbp, fut, prev, metrics, &bc, &mut wb.0);
                        }
                        Right(uo) => {
                            euler::RHS_upwind(&**uo, fut, prev, metrics, &bc, &mut wb.0);
                        }
                    })
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
            (),
            &mut (),
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
    let interpolation_operators = jgrids
        .iter()
        .map(|_g| euler::InterpolationOperators {
            north: Some(Box::new(operators::Interpolation4)),
            south: Some(Box::new(operators::Interpolation4)),
            east: Some(Box::new(operators::Interpolation4)),
            west: Some(Box::new(operators::Interpolation4)),
        })
        .collect::<Vec<_>>();

    let grids = jgrids
        .into_iter()
        .map(|egrid| egrid.grid)
        .collect::<Vec<grid::Grid>>();

    let integration_time: Float = json["integration_time"].as_number().unwrap().into();

    let operators = grids
        .iter()
        .map(|_| Right(Box::new(operators::Upwind4) as Box<dyn UpwindOperator2d>))
        .collect::<Vec<_>>();

    let mut sys = System::new(grids, bt, interpolation_operators, operators);
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
        for ((fmod, grid), op) in sys.fnow.iter().zip(&sys.grids).zip(&sys.operators) {
            let mut fvort = fmod.clone();
            fvort.vortex(grid.x(), grid.y(), time, vortexparams);
            let sbpop: &dyn SbpOperator2d = op.as_ref().either(|op| &**op, |uo| uo.as_sbp());
            e += fmod.h2_err(&fvort, sbpop);
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
