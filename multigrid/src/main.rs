use argh::FromArgs;

use euler::eval::Evaluator;
use sbp::*;

mod file;
mod input;
mod parsing;
use file::*;
mod eval;
mod system;
use system::*;

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
    /// distribute the computation on multiple threads
    #[argh(switch)]
    distribute: bool,
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

    let config: input::Configuration = match json5::from_str(&filecontents) {
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
        grid_connections,
        op: operators,
        integration_time,
        initial_conditions,
        boundary_conditions: _,
    } = config.into_runtime();

    let mut sys = System::new(grids, grid_connections, operators);
    match &initial_conditions {
        /*
        parsing::InitialConditions::File(f) => {
            for grid in &sys.grids {
                // Copy initial conditions from file, requires name of field
                todo!()
            }
        }
        */
        parsing::InitialConditions::Vortex(vortexparams) => sys.vortex(0.0, &vortexparams),
        parsing::InitialConditions::Expressions(expr) => {
            let t = 0.0;
            for (grid, field) in sys.grids.iter().zip(sys.fnow.iter_mut()) {
                // Evaluate the expressions on all variables
                let x = grid.x();
                let y = grid.y();
                let (rho, rhou, rhov, e) = field.components_mut();
                (*expr).evaluate(t, x, y, rho, rhou, rhov, e);
            }
        }
    }

    let dt = sys.max_dt();

    let ntime = (integration_time / dt).round() as u64;

    if opt.distribute {
        let sys = sys.distribute(ntime);
        let timer = if opt.timings {
            Some(std::time::Instant::now())
        } else {
            None
        };
        sys.run();
        if let Some(timer) = timer {
            let duration = timer.elapsed();
            println!("Duration: {:?}", duration);
        }
        return;
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
    output.add_timestep(0, &sys.fnow);

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
            match &initial_conditions {
                parsing::InitialConditions::Vortex(vortexparams) => {
                    fvort.vortex(grid.x(), grid.y(), time, &vortexparams);
                }
                parsing::InitialConditions::Expressions(expr) => {
                    let (rho, rhou, rhov, e) = fvort.components_mut();
                    expr.as_ref()
                        .evaluate(time, grid.x(), grid.y(), rho, rhou, rhov, e)
                }
            }
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
