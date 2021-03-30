#![cfg(feature = "expensive_tests")]
use euler::*;
use ndarray::prelude::*;
use sbp::{
    operators::{SbpOperator2d, UpwindOperator2d},
    Float,
};

fn run_with_size(size: usize, op: impl SbpOperator2d + UpwindOperator2d + Copy) -> Float {
    let nx = size;
    let ny = size;
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

    let mut sys = System::new(x, y, op);
    sys.vortex(0.0, vortex_params.clone());

    let time = 0.2;
    let dt = 0.2 * Float::min(1.0 / (nx - 1) as Float, 1.0 / (ny - 1) as Float);

    let nsteps = (time / dt) as usize;
    for _ in 0..nsteps {
        sys.advance_upwind(dt);
    }

    let mut verifield = Field::new(ny, nx);
    verifield.vortex(sys.x(), sys.y(), nsteps as Float * dt, &vortex_params);

    verifield.h2_err(sys.field(), &op)
}

fn convergence(op: impl SbpOperator2d + UpwindOperator2d + Copy, expected_q: Float) {
    let sizes = [25, 35, 50, 71, 100, 150, 200];
    let mut prev: Option<(usize, Float)> = None;
    let mut q_last = None;
    println!("Size\tError(h2)\tq");
    for size in &sizes {
        print!("{:3}x{:3}", size, size);
        let e = run_with_size(*size, op);
        print!("\t{:.10}", e);
        if let Some(prev) = prev.take() {
            let m0 = size * size;
            let e0 = e;

            let (size1, e1) = prev;
            let m1 = size1 * size1;

            let q =
                Float::log10(e0 / e1) / Float::log10((m0 as Float / m1 as Float).powf(1.0 / 2.0));
            print!("\t{}", q);
            q_last = Some(q);
        }
        println!();
        prev = Some((*size, e));
    }
    let actual_q = q_last.unwrap();
    assert!(actual_q < expected_q);
}

#[test]
fn convergence_upwind4() {
    convergence(sbp::operators::Upwind4, -4.0);
}

#[test]
fn convergence_upwind9() {
    convergence(sbp::operators::Upwind9, -8.5);
}
