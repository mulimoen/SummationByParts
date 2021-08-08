use argh::FromArgs;

use sbp::*;

mod eval;
mod input;
mod parsing;
mod system;

#[derive(Debug, FromArgs)]
/// Options for configuring and running the solver
struct CliOptions {
    #[argh(positional)]
    json: std::path::PathBuf,
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
        boundary_conditions,
        op: operators,
        integration_time,
        initial_conditions,
    } = config.into_runtime();

    let basesystem = system::BaseSystem::new(
        names.clone(),
        grids,
        0.0,
        operators,
        boundary_conditions,
        initial_conditions.clone(),
        opt.output.clone(),
    );
    // System::new(grids, grid_connections, operators);

    let mut sys = if opt.distribute {
        basesystem.create_distributed()
    } else {
        basesystem.create()
    };

    let dt = sys.max_dt();
    sys.set_dt(dt);
    let ntime = (integration_time / dt).round() as u64;
    let steps_between_outputs = if let Some(n) = opt.number_of_outputs {
        std::cmp::max(n / ntime, 1)
    } else {
        ntime
    };

    sys.output(0);
    //let output = File::create(&opt.output, sys.grids.as_slice(), names).unwrap();
    //let mut output = OutputThread::new(output);
    //output.add_timestep(0, &sys.fnow);

    let progressbar = progressbar(opt.no_progressbar, ntime);

    let timer = if opt.timings {
        Some(std::time::Instant::now())
    } else {
        None
    };

    let mut itime = 0;
    while itime < ntime {
        sys.advance(steps_between_outputs);
        progressbar.inc(1);

        itime += steps_between_outputs;
        sys.output(itime);
    }

    /*
    for itime in 0..ntime {
        if should_output(itime) {
            output.add_timestep(itime, &sys.fnow);
        }
        progressbar.inc(1);
        sys.advance(dt);
    }
    */
    progressbar.finish_and_clear();

    let mut outinfo = OutputInformation {
        filename: opt.output,
        ..Default::default()
    };

    if let Some(timer) = timer {
        let duration = timer.elapsed();
        outinfo.time_elapsed = Some(duration);
    }

    //output.add_timestep(ntime, &sys.fnow);

    /*
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
    */

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
