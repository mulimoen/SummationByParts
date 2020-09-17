use either::*;
use structopt::StructOpt;

use sbp::operators::{SbpOperator2d, UpwindOperator2d};
use sbp::*;

mod file;
mod parsing;
use file::*;

pub(crate) type DiffOp = Either<Box<dyn SbpOperator2d>, Box<dyn UpwindOperator2d>>;

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
    operators: Vec<DiffOp>,
}

impl System {
    fn new(
        grids: Vec<grid::Grid>,
        bt: Vec<euler::BoundaryCharacteristics>,
        operators: Vec<DiffOp>,
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
            operators,
        }
    }

    fn vortex(&mut self, t: Float, vortex_params: &euler::VortexParameters) {
        for (f, g) in self.fnow.iter_mut().zip(&self.grids) {
            f.vortex(g.x(), g.y(), t, &vortex_params);
        }
    }

    fn advance(&mut self, dt: Float, pool: &rayon::ThreadPool) {
        let metrics = &self.metrics;
        let grids = &self.grids;
        let bt = &self.bt;
        let wb = &mut self.wb;
        let eb = &mut self.eb;
        let operators = &self.operators;

        let rhs = move |fut: &mut [euler::Field], prev: &[euler::Field], time: Float| {
            let prev_all = &prev;
            pool.scope(|s| {
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
                        match op.as_ref() {
                            Left(sbp) => {
                                euler::RHS_trad(&**sbp, fut, prev, metrics, &bc, &mut wb.0);
                            }
                            Right(uo) => {
                                euler::RHS_upwind(&**uo, fut, prev, metrics, &bc, &mut wb.0);
                            }
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
        sbp::integrate::integrate_multigrid::<sbp::integrate::Rk4, _, _, _>(
            rhs,
            &self.fnow,
            &mut self.fnext,
            &mut self.time,
            dt,
            &mut k,
            pool,
        );

        std::mem::swap(&mut self.fnow, &mut self.fnext);
    }

    /// Suggested maximum dt for this problem
    fn max_dt(&self) -> Float {
        let is_h2 = self.operators.iter().any(|op| {
            op.as_ref().either(
                |op| op.is_h2xi() || op.is_h2eta(),
                |op| op.is_h2xi() || op.is_h2eta(),
            )
        });
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

    let config: parsing::Configuration = json5::from_str(&filecontents).unwrap();

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
            fvort.vortex(grid.x(), grid.y(), time, &vortexparams);
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
