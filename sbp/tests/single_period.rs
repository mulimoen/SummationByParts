#![cfg(feature = "expensive_tests")]
use ndarray::prelude::*;
use sbp::euler::*;
use sbp::Float;

#[test]
#[ignore]
fn single_period_upwind4() {
    let nx = 100;
    let ny = 100;

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
        vortices: {
            let mut v = ArrayVec::new();
            v.push(Vortice {
                x0: -1.0,
                y0: 0.0,
                rstar: 0.5,
                eps: 1.0,
            });
            v
        },
        mach: 0.5,
    };

    let mut sys = System::new(x, y, sbp::operators::Upwind4);
    sys.vortex(0.0, vortex_params.clone());

    let time = 10.0;
    let dt = 0.2 * Float::min(1.0 / (nx - 1) as Float, 1.0 / (ny - 1) as Float);

    let nsteps = (time / dt) as usize;
    for _ in 0..nsteps {
        sys.advance_upwind(dt);
    }

    let mut verifield = Field::new(ny, nx);
    verifield.vortex(
        sys.x(),
        sys.y(),
        nsteps as Float * dt - 10.0,
        &vortex_params,
    );

    let err = verifield.h2_err(sys.field(), &sbp::operators::Upwind4);
    panic!("{}", err);
}
