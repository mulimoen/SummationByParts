use argh::FromArgs;
use rayon::prelude::*;

use sbp::operators::SbpOperator2d;
use sbp::*;

mod file;
mod parsing;
use file::*;

struct System {
    fnow: Vec<euler::Field>,
    fnext: Vec<euler::Field>,
    wb: Vec<euler::WorkBuffers>,
    k: [Vec<euler::Diff>; 4],
    grids: Vec<grid::Grid>,
    metrics: Vec<grid::Metrics>,
    bt: Vec<euler::BoundaryCharacteristics>,
    eb: Vec<euler::BoundaryStorage>,
    time: Float,
    operators: Vec<Box<dyn SbpOperator2d>>,
}

use std::sync::atomic::{AtomicBool, Ordering};
pub(crate) static MULTITHREAD: AtomicBool = AtomicBool::new(false);

impl integrate::Integrable for System {
    type State = Vec<euler::Field>;
    type Diff = Vec<euler::Diff>;
    fn assign(s: &mut Self::State, o: &Self::State) {
        if MULTITHREAD.load(Ordering::Acquire) {
            s.par_iter_mut()
                .zip(o.par_iter())
                .for_each(|(s, o)| euler::Field::assign(s, o))
        } else {
            s.iter_mut()
                .zip(o.iter())
                .for_each(|(s, o)| euler::Field::assign(s, o))
        }
    }
    fn scaled_add(s: &mut Self::State, o: &Self::Diff, scale: Float) {
        if MULTITHREAD.load(Ordering::Acquire) {
            s.par_iter_mut()
                .zip(o.par_iter())
                .for_each(|(s, o)| euler::Field::scaled_add(s, o, scale))
        } else {
            s.iter_mut()
                .zip(o.iter())
                .for_each(|(s, o)| euler::Field::scaled_add(s, o, scale))
        }
    }
}

impl System {
    fn new(
        grids: Vec<grid::Grid>,
        bt: Vec<euler::BoundaryCharacteristics>,
        operators: Vec<Box<dyn SbpOperator2d>>,
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
        let k = grids
            .iter()
            .map(|g| euler::Diff::zeros((g.ny(), g.nx())))
            .collect::<Vec<_>>();
        let k = [k.clone(), k.clone(), k.clone(), k];
        let metrics = grids
            .iter()
            .zip(&operators)
            .map(|(g, op)| g.metrics(&**op).unwrap())
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
            operators,
        }
    }

    fn vortex(&mut self, t: Float, vortex_params: &euler::VortexParameters) {
        for (f, g) in self.fnow.iter_mut().zip(&self.grids) {
            f.vortex(g.x(), g.y(), t, &vortex_params);
        }
    }

    fn advance(&mut self, dt: Float) {
        let metrics = &self.metrics;
        let grids = &self.grids;
        let bt = &self.bt;
        let wb = &mut self.wb;
        let eb = &mut self.eb;
        let operators = &self.operators;

        let rhs = move |fut: &mut Vec<euler::Diff>, prev: &Vec<euler::Field>, time: Float| {
            let prev_all = &prev;
            if MULTITHREAD.load(Ordering::Acquire) {
                rayon::scope(|s| {
                    for (((((((fut, prev), wb), grid), metrics), op), bt), eb) in fut
                        .iter_mut()
                        .zip(prev.iter())
                        .zip(wb.iter_mut())
                        .zip(grids)
                        .zip(metrics.iter())
                        .zip(operators.iter())
                        .zip(bt.iter())
                        .zip(eb.iter_mut())
                    {
                        s.spawn(move |_| {
                            let bc = euler::boundary_extracts(prev_all, bt, prev, grid, eb, time);
                            if op.upwind().is_some() {
                                euler::RHS_upwind(&**op, fut, prev, metrics, &bc, &mut wb.0);
                            } else {
                                euler::RHS_trad(&**op, fut, prev, metrics, &bc, &mut wb.0);
                            }
                        })
                    }
                });
            } else {
                for (((((((fut, prev), wb), grid), metrics), op), bt), eb) in fut
                    .iter_mut()
                    .zip(prev.iter())
                    .zip(wb.iter_mut())
                    .zip(grids)
                    .zip(metrics.iter())
                    .zip(operators.iter())
                    .zip(bt.iter())
                    .zip(eb.iter_mut())
                {
                    let bc = euler::boundary_extracts(prev_all, bt, prev, grid, eb, time);
                    if op.upwind().is_some() {
                        euler::RHS_upwind(&**op, fut, prev, metrics, &bc, &mut wb.0);
                    } else {
                        euler::RHS_trad(&**op, fut, prev, metrics, &bc, &mut wb.0);
                    }
                }
            }
        };

        integrate::integrate::<integrate::Rk4, System, _>(
            rhs,
            &self.fnow,
            &mut self.fnext,
            &mut self.time,
            dt,
            &mut self.k,
        );

        std::mem::swap(&mut self.fnow, &mut self.fnext);
    }

    /// Suggested maximum dt for this problem
    fn max_dt(&self) -> Float {
        let is_h2 = self
            .operators
            .iter()
            .any(|op| op.is_h2xi() || op.is_h2eta());
        let c_max = if is_h2 { 0.5 } else { 1.0 };
        let mut max_dt: Float = Float::INFINITY;

        for (field, metrics) in self.fnow.iter().zip(self.metrics.iter()) {
            let nx = field.nx();
            let ny = field.ny();

            let rho = field.rho();
            let rhou = field.rhou();
            let rhov = field.rhov();

            let mut max_u: Float = 0.0;
            let mut max_v: Float = 0.0;

            for ((((((rho, rhou), rhov), detj_dxi_dx), detj_dxi_dy), detj_deta_dx), detj_deta_dy) in
                rho.iter()
                    .zip(rhou.iter())
                    .zip(rhov.iter())
                    .zip(metrics.detj_dxi_dx())
                    .zip(metrics.detj_dxi_dy())
                    .zip(metrics.detj_deta_dx())
                    .zip(metrics.detj_deta_dy())
            {
                let u = rhou / rho;
                let v = rhov / rho;

                let uhat: Float = detj_dxi_dx * u + detj_dxi_dy * v;
                let vhat: Float = detj_deta_dx * u + detj_deta_dy * v;

                max_u = max_u.max(uhat.abs());
                max_v = max_v.max(vhat.abs());
            }

            let dx = 1.0 / nx as Float;
            let dy = 1.0 / ny as Float;

            let c_dt = Float::max(max_u / dx, max_v / dy);

            max_dt = Float::min(max_dt, c_max / c_dt);
        }

        max_dt
    }
}

#[derive(Debug, FromArgs)]
/// Options for configuring and running the solver
struct CliOptions {
    #[argh(positional)]
    json: std::path::PathBuf,
    /// number of simultaneous threads
    #[argh(option, short = 'j')]
    jobs: Option<usize>,
    /// name of output file
    #[argh(
        option,
        short = 'o',
        default = "std::path::PathBuf::from(\"output.hdf\")"
    )]
    output: std::path::PathBuf,
    /// number of outputs to save
    #[argh(option, short = 'n')]
    number_of_outputs: Option<u64>,
    /// print the time to complete, taken in the compute loop
    #[argh(switch)]
    timings: bool,
    /// print error at the end of the run
    #[argh(switch)]
    error: bool,
    /// disable the progressbar
    #[argh(switch)]
    no_progressbar: bool,
    /// output information regarding time elapsed and error
    /// in json format
    #[argh(switch)]
    output_json: bool,
}

#[derive(Default, serde::Serialize)]
struct OutputInformation {
    filename: std::path::PathBuf,
    #[serde(skip_serializing_if = "Option::is_none")]
    time_elapsed: Option<std::time::Duration>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<Float>,
}

fn main() {
    let opt: CliOptions = argh::from_env();
    let filecontents = std::fs::read_to_string(&opt.json).unwrap();

    let config: parsing::Configuration = match json5::from_str(&filecontents) {
        Ok(config) => config,
        Err(e) => {
            eprintln!("Configuration could not be read: {}", e);
            if let json5::Error::Message {
                location: Some(location),
                ..
            } = e
            {
                eprintln!("\t{:?}", location);
            }
            return;
        }
    };

    let parsing::RuntimeConfiguration {
        names,
        grids,
        bc: bt,
        op: operators,
        integration_time,
        vortex: vortexparams,
    } = config.into_runtime();

    let mut sys = System::new(grids, bt, operators);
    sys.vortex(0.0, &vortexparams);

    let dt = sys.max_dt();

    let ntime = (integration_time / dt).round() as u64;

    {
        let nthreads = opt.jobs.unwrap_or(1);
        if nthreads > 1 {
            MULTITHREAD.store(true, Ordering::Release);
            rayon::ThreadPoolBuilder::new()
                .num_threads(nthreads)
                .build_global()
                .unwrap();
        }
    }

    let should_output = |itime| {
        opt.number_of_outputs.map_or(false, |num_out| {
            if num_out == 0 {
                false
            } else {
                itime % (std::cmp::max(ntime / (num_out - 1), 1)) == 0
            }
        })
    };

    let output = File::create(&opt.output, sys.grids.as_slice(), names).unwrap();
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
        sys.advance(dt);
    }
    progressbar.finish_and_clear();

    let mut outinfo = OutputInformation {
        filename: opt.output,
        ..Default::default()
    };

    if let Some(timer) = timer {
        let duration = timer.elapsed();
        outinfo.time_elapsed = Some(duration);
    }

    output.add_timestep(ntime, &sys.fnow);
    if opt.error {
        let time = ntime as Float * dt;
        let mut e = 0.0;
        for ((fmod, grid), op) in sys.fnow.iter().zip(&sys.grids).zip(&sys.operators) {
            let mut fvort = fmod.clone();
            fvort.vortex(grid.x(), grid.y(), time, &vortexparams);
            e += fmod.h2_err(&fvort, &**op);
        }
        outinfo.error = Some(e);
    }

    if opt.output_json {
        println!("{}", json5::to_string(&outinfo).unwrap());
    } else {
        if let Some(duration) = outinfo.time_elapsed {
            println!("Time elapsed: {} seconds", duration.as_secs_f64());
        }
        if let Some(error) = outinfo.error {
            println!("Total error: {:e}", error);
        }
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
