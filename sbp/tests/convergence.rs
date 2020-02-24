use ndarray::prelude::*;
use sbp::euler::*;

fn run_with_size<SBP: sbp::operators::UpwindOperator>(size: usize) -> f32 {
    let nx = 4 * size;
    let ny = 4 * size;
    let x = Array1::linspace(-5.0, 5.0, nx);
    let y = Array1::linspace(-5.0, 5.0, ny);

    let x = x.broadcast((ny, nx)).unwrap().to_owned();
    let y = y
        .reversed_axes()
        .broadcast((nx, ny))
        .unwrap()
        .reversed_axes()
        .to_owned();

    let vortex_params = VortexParameters {
        x0: -1.0,
        y0: 0.0,
        mach: 0.5,
        rstar: 0.5,
        eps: 1.0,
    };

    let mut sys = System::<SBP>::new(x, y);
    sys.vortex(0.0, vortex_params);

    let time = 0.1;
    let dt = 0.2 * f32::min(1.0 / (nx - 1) as f32, 1.0 / (ny - 1) as f32);

    let nsteps = (time / dt) as usize;
    for _ in 0..nsteps {
        sys.advance_upwind(dt);
    }

    let mut verifield = Field::new(ny, nx);
    verifield.vortex(sys.x(), sys.y(), nsteps as f32 * dt, vortex_params);

    verifield.err_diff::<SBP>(sys.field())
}

#[test]
fn test() {
    let sizes = [25, 35, 50, 71, 100];
    let mut errs = Vec::with_capacity(sizes.len());
    for size in &sizes {
        println!("Size: {}", size);
        let e = run_with_size::<sbp::operators::Upwind9>(*size);
        errs.push(e);
    }
    panic!("{:?}", errs);
}
