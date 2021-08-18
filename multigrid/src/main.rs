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
        names,
        grids,
        0.0,
        operators,
        boundary_conditions,
        initial_conditions,
        opt.output.clone(),
    );

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

    if !opt.no_progressbar {
        sys.add_progressbar(ntime)
    }

    let timer = if opt.timings {
        Some(std::time::Instant::now())
    } else {
        None
    };

    let mut itime = 0;
    while itime < ntime {
        let nexttime = (itime + steps_between_outputs).max(ntime);
        sys.advance(nexttime - itime);

        itime = nexttime;
        sys.output(itime);
    }

    if !opt.no_progressbar {
        sys.finish_progressbar();
    }

    let mut outinfo = OutputInformation {
        filename: opt.output,
        ..Default::default()
    };

    if let Some(timer) = timer {
        let duration = timer.elapsed();
        outinfo.time_elapsed = Some(duration);
    }

    //output.add_timestep(ntime, &sys.fnow);

    if opt.error {
        outinfo.error = Some(sys.error())
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

fn progressbar(ntime: u64) -> indicatif::ProgressBar {
    let progressbar = indicatif::ProgressBar::new(ntime);
    progressbar.with_style(
        indicatif::ProgressStyle::default_bar()
            .template("{wide_bar:.cyan/blue} {pos}/{len} ({eta})"),
    )
}
