//! Integration of explicit PDEs using different Butcher Tableaus
//!
//! Integration can be performed on all systems that can be represented
//! as using a transform into an [`ndarray::ArrayView`] for both the state
//! and the state difference.
//!
//! The integration functions are memory efficient, and relies
//! on the `k` parameter to hold the system state differences.
//! This parameter is tied to the Butcher Tableau
#![forbid(unsafe_code)]
#![no_std]

use float::Float;

/// The Butcher Tableau, with the state transitions described as
/// [on wikipedia](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#Explicit_Runge%E2%80%93Kutta_methods).
pub trait ButcherTableau {
    /// This bound should not be overridden
    const S: usize = Self::B.len();
    /// Only the lower triangle will be used (explicit integration)
    const A: &'static [&'static [Float]];
    const B: &'static [Float];
    const C: &'static [Float];
}

pub trait EmbeddedButcherTableau: ButcherTableau {
    const BSTAR: &'static [Float];
}

pub struct Rk4;
impl ButcherTableau for Rk4 {
    const A: &'static [&'static [Float]] = &[&[0.5], &[0.0, 0.5], &[0.0, 0.0, 1.0]];
    const B: &'static [Float] = &[1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0];
    const C: &'static [Float] = &[0.5, 0.5, 1.0];
}

pub struct Rk4_38;
impl ButcherTableau for Rk4_38 {
    const A: &'static [&'static [Float]] = &[&[1.0 / 3.0], &[-1.0 / 3.0, 1.0], &[1.0, -1.0, 1.0]];
    const B: &'static [Float] = &[1.0 / 8.0, 3.0 / 8.0, 3.0 / 8.0, 1.0 / 8.0];
    const C: &'static [Float] = &[1.0 / 3.0, 2.0 / 3.0, 1.0];
}

pub struct EulerMethod;
impl ButcherTableau for EulerMethod {
    const A: &'static [&'static [Float]] = &[&[]];
    const B: &'static [Float] = &[1.0];
    const C: &'static [Float] = &[];
}

pub struct MidpointMethod;
impl ButcherTableau for MidpointMethod {
    const A: &'static [&'static [Float]] = &[&[0.5]];
    const B: &'static [Float] = &[0.0, 1.0];
    const C: &'static [Float] = &[1.0 / 2.0];
}

/// Bit excessive...
#[allow(clippy::excessive_precision)]
#[allow(clippy::unreadable_literal)]
const SQRT_5: Float = 2.236067977499789696409173668731276235440618359611525724270897245410520925637804899414414408378782275;
pub struct Rk6;
impl ButcherTableau for Rk6 {
    const A: &'static [&'static [Float]] = &[
        &[4.0 / 7.0],
        &[115.0 / 112.0, -5.0 / 16.0],
        &[589.0 / 630.0, 5.0 / 18.0, -16.0 / 45.0],
        &[
            229.0 / 1200.0 - 29.0 / 6000.0 * SQRT_5,
            119.0 / 240.0 - 187.0 / 1200.0 * SQRT_5,
            -14.0 / 75.0 + 34.0 / 375.0 * SQRT_5,
            -3.0 / 100.0 * SQRT_5,
        ],
        &[
            71.0 / 2400.0 - 587.0 / 12000.0 * SQRT_5,
            187.0 / 480.0 - 391.0 / 2400.0 * SQRT_5,
            -38.0 / 75.0 + 26.0 / 375.0 * SQRT_5,
            27.0 / 80.0 - 3.0 / 400.0 * SQRT_5,
            (1.0 + SQRT_5) / 4.0,
        ],
        &[
            -49.0 / 480.0 + 43.0 / 160.0 * SQRT_5,
            -425.0 / 96.0 + 51.0 / 32.0 * SQRT_5,
            52.0 / 15.0 - 4.0 / 5.0 * SQRT_5,
            -27.0 / 16.0 + 3.0 / 16.0 * SQRT_5,
            5.0 / 4.0 - 3.0 / 4.0 * SQRT_5,
            5.0 / 2.0 - 1.0 / 2.0 * SQRT_5,
        ],
    ];
    const B: &'static [Float] = &[
        1.0 / 12.0,
        0.0,
        0.0,
        0.0,
        5.0 / 12.0,
        5.0 / 12.0,
        1.0 / 12.0,
    ];
    const C: &'static [Float] = &[
        4.0 / 7.0,
        5.0 / 7.0,
        6.0 / 7.0,
        (5.0 - SQRT_5) / 10.0,
        (5.0 + SQRT_5) / 10.0,
        1.0,
    ];
}

pub struct Fehlberg;

impl ButcherTableau for Fehlberg {
    #[rustfmt::skip]
    const A: &'static [&'static [Float]] = &[
        &[1.0 / 4.0],
        &[3.0 / 32.0, 9.0 / 32.0],
        &[1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0],
        &[439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0],
        &[-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0],
    ];
    #[rustfmt::skip]
    const B: &'static [Float] = &[
        16.0 / 135.0, 0.0, 6656.0 / 12825.0, 28561.0 / 56430.0, -9.0 / 50.0, 2.0 / 55.0,
    ];
    const C: &'static [Float] = &[0.0, 1.0 / 4.0, 3.0 / 8.0, 12.0 / 13.0, 1.0, 1.0 / 2.0];
}

impl EmbeddedButcherTableau for Fehlberg {
    const BSTAR: &'static [Float] = &[
        25.0 / 216.0,
        0.0,
        1408.0 / 2565.0,
        2197.0 / 4104.0,
        -1.0 / 5.0,
        0.0,
    ];
}

pub struct BogackiShampine;

impl ButcherTableau for BogackiShampine {
    const A: &'static [&'static [Float]] = &[
        &[1.0 / 2.0],
        &[0.0, 3.0 / 4.0],
        &[2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0],
    ];
    const B: &'static [Float] = &[2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0];
    const C: &'static [Float] = &[0.0, 1.0 / 2.0, 3.0 / 4.0, 1.0];
}

impl EmbeddedButcherTableau for BogackiShampine {
    const BSTAR: &'static [Float] = &[7.0 / 24.0, 1.0 / 4.0, 1.0 / 3.0, 1.0 / 8.0];
}

pub trait Integrable {
    /// This type should support `clone_from`
    type State: Clone;
    type Diff;

    fn scaled_add(s: &mut Self::State, o: &Self::Diff, scale: Float);
}

#[allow(clippy::too_many_arguments)]
/// Integrates using the [`ButcherTableau`] specified. `rhs` should be the result
/// of the right hand side of $u_t = rhs$
///
/// rhs takes the old state and the current time, and outputs the state difference
/// in the first parameter
///
/// Should be called as
/// ```rust,ignore
/// integrate::<Rk4, System, _>(...)
/// ```
pub fn integrate<BTableau: ButcherTableau, F: Integrable, RHS>(
    mut rhs: RHS,
    prev: &F::State,
    fut: &mut F::State,
    time: &mut Float,
    dt: Float,
    k: &mut [F::Diff],
) where
    RHS: FnMut(&mut F::Diff, &F::State, Float),
{
    assert!(k.len() >= BTableau::S);

    for i in 0.. {
        let simtime;
        match i {
            0 => {
                fut.clone_from(prev);
                simtime = *time;
            }
            i if i < BTableau::S => {
                fut.clone_from(prev);
                for (&a, k) in BTableau::A[i - 1].iter().zip(k.iter()) {
                    if a == 0.0 {
                        continue;
                    }
                    F::scaled_add(fut, k, a * dt);
                }
                simtime = *time + dt * BTableau::C[i - 1];
            }
            _ if i == BTableau::S => {
                fut.clone_from(prev);
                for (&b, k) in BTableau::B.iter().zip(k.iter()) {
                    if b == 0.0 {
                        continue;
                    }
                    F::scaled_add(fut, k, b * dt);
                }
                *time += dt;
                return;
            }
            _ => {
                unreachable!();
            }
        };

        rhs(&mut k[i], &fut, simtime);
    }
}

#[allow(clippy::too_many_arguments)]
/// Integrate using an [`EmbeddedButcherTableau`], else similar to [`integrate`]
///
/// This produces two results, the most accurate result in `fut`, and the less accurate
/// result in `fut2`. This can be used for convergence testing and adaptive timesteps.
pub fn integrate_embedded_rk<BTableau: EmbeddedButcherTableau, F: Integrable, RHS>(
    rhs: RHS,
    prev: &F::State,
    fut: &mut F::State,
    fut2: &mut F::State,
    time: &mut Float,
    dt: Float,
    k: &mut [F::Diff],
) where
    RHS: FnMut(&mut F::Diff, &F::State, Float),
{
    integrate::<BTableau, F, RHS>(rhs, prev, fut, time, dt, k);
    fut2.clone_from(prev);
    for (&b, k) in BTableau::BSTAR.iter().zip(k.iter()) {
        if b == 0.0 {
            continue;
        }
        F::scaled_add(fut2, k, b * dt);
    }
}

#[test]
/// Solving a second order PDE
fn ballistic() {
    #[derive(Clone, Debug)]
    struct Ball {
        z: Float,
        v: Float,
    }
    impl Integrable for Ball {
        type State = Ball;
        type Diff = (Float, Float);
        fn scaled_add(s: &mut Self::State, o: &Self::Diff, sc: Float) {
            s.z += o.0 * sc;
            s.v += o.1 * sc;
        }
    }

    let mut t = 0.0;
    let dt = 0.001;
    let initial = Ball { z: 0.0, v: 10.0 };
    let g = -9.81;

    let mut k = [(0.0, 0.0); 4];
    let gravity = |d: &mut (Float, Float), s: &Ball, _time: Float| {
        d.1 = g;
        d.0 = s.v
    };
    let mut next = initial.clone();
    //while next.z >= 0.0 {
    while t < 1.0 {
        let mut next2 = next.clone();
        integrate::<EulerMethod, Ball, _>(gravity, &next, &mut next2, &mut t, dt, &mut k);
        core::mem::swap(&mut next, &mut next2);
    }
    let expected_vel = initial.v + g * t;
    assert!((next.v - expected_vel).abs() < 1e-3);
    let expected_pos = initial.z + initial.v * t + g / 2.0 * t.powi(2);
    assert!((next.z - expected_pos).abs() < 1e-2);
}
