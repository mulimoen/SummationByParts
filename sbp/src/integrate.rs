//! Integration of explicit PDEs using different Butcher Tableaus
//!
//! Integration can be performed on all systems that can be represented
//! as using a transform into an [`ndarray::ArrayView`] for both the state
//! and the state difference.
//!
//! The integration functions are memory efficient, and relies
//! on the `k` parameter to hold the system state differences.
//! This parameter is tied to the Butcher Tableau

use super::Float;
use ndarray::{ArrayView, ArrayViewMut};

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

#[allow(clippy::too_many_arguments)]
/// Integrates using the [`ButcherTableau`] specified. `rhs` should be the result
/// of the right hand side of $u_t = rhs$
///
/// rhs takes the old state and the current time, and outputs the state difference
/// in the first parameter
///
/// Should be called as
/// ```rust,ignore
/// integrate::<Rk4, _, _, _, _>(...)
/// ```
pub fn integrate<BTableau: ButcherTableau, F, RHS, D>(
    mut rhs: RHS,
    prev: &F,
    fut: &mut F,
    time: &mut Float,
    dt: Float,
    k: &mut [F],
) where
    for<'r> &'r F: std::convert::Into<ArrayView<'r, Float, D>>,
    for<'r> &'r mut F: std::convert::Into<ArrayViewMut<'r, Float, D>>,
    D: ndarray::Dimension,
    RHS: FnMut(&mut F, &F, Float),
{
    assert_eq!(prev.into().shape(), fut.into().shape());
    assert!(k.len() >= BTableau::S);

    for i in 0.. {
        let simtime;
        match i {
            0 => {
                fut.into().assign(&prev.into());
                simtime = *time;
            }
            i if i < BTableau::S => {
                fut.into().assign(&prev.into());
                for (&a, k) in BTableau::A[i - 1].iter().zip(k.iter()) {
                    if a == 0.0 {
                        continue;
                    }
                    fut.into().scaled_add(a * dt, &k.into());
                }
                simtime = *time + dt * BTableau::C[i - 1];
            }
            _ if i == BTableau::S => {
                fut.into().assign(&prev.into());
                for (&b, k) in BTableau::B.iter().zip(k.iter()) {
                    if b == 0.0 {
                        continue;
                    }
                    fut.into().scaled_add(b * dt, &k.into());
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
pub fn integrate_embedded_rk<BTableau: EmbeddedButcherTableau, F, RHS, D>(
    rhs: RHS,
    prev: &F,
    fut: &mut F,
    fut2: &mut F,
    time: &mut Float,
    dt: Float,
    k: &mut [F],
) where
    for<'r> &'r F: std::convert::Into<ArrayView<'r, Float, D>>,
    for<'r> &'r mut F: std::convert::Into<ArrayViewMut<'r, Float, D>>,
    RHS: FnMut(&mut F, &F, Float),
    D: ndarray::Dimension,
{
    integrate::<BTableau, F, RHS, D>(rhs, prev, fut, time, dt, k);
    fut2.into().assign(&prev.into());
    for (&b, k) in BTableau::BSTAR.iter().zip(k.iter()) {
        if b == 0.0 {
            continue;
        }
        fut2.into().scaled_add(b * dt, &k.into());
    }
}

#[cfg(feature = "rayon")]
#[allow(clippy::too_many_arguments)]
/// Integrates a multigrid problem, much the same as [`integrate`],
/// using a `rayon` threadpool for parallelisation.
///
/// note that `rhs` accepts the full system state, and is responsible
/// for computing the full state difference.
/// `rhs` can be a mutable closure, so buffers can be used
/// and mutated inside the closure.
///
/// This function requires the `rayon` feature, and is not callable in
/// a `wasm` context.
pub fn integrate_multigrid<BTableau: ButcherTableau, F, RHS, D>(
    mut rhs: RHS,
    prev: &[F],
    fut: &mut [F],
    time: &mut Float,
    dt: Float,
    k: &mut [&mut [F]],

    pool: &rayon::ThreadPool,
) where
    for<'r> &'r F: std::convert::Into<ArrayView<'r, Float, D>>,
    for<'r> &'r mut F: std::convert::Into<ArrayViewMut<'r, Float, D>>,
    RHS: FnMut(&mut [F], &[F], Float),
    F: Send + Sync,
    D: ndarray::Dimension,
{
    for i in 0.. {
        let simtime;
        match i {
            0 => {
                pool.scope(|s| {
                    assert!(k.len() >= BTableau::S);
                    for (prev, fut) in prev.iter().zip(fut.iter_mut()) {
                        s.spawn(move |_| {
                            assert_eq!(prev.into().shape(), fut.into().shape());
                            fut.into().assign(&prev.into());
                        });
                    }
                });
                simtime = *time;
            }
            i if i < BTableau::S => {
                pool.scope(|s| {
                    for (ig, (prev, fut)) in prev.iter().zip(fut.iter_mut()).enumerate() {
                        let k = &k;
                        s.spawn(move |_| {
                            fut.into().assign(&prev.into());
                            for (ik, &a) in BTableau::A[i - 1].iter().enumerate() {
                                if a == 0.0 {
                                    continue;
                                }
                                fut.into().scaled_add(a * dt, &(&k[ik][ig]).into());
                            }
                        });
                    }
                });
                simtime = *time + dt * BTableau::C[i - 1];
            }
            _ if i == BTableau::S => {
                pool.scope(|s| {
                    for (ig, (prev, fut)) in prev.iter().zip(fut.iter_mut()).enumerate() {
                        let k = &k;
                        s.spawn(move |_| {
                            fut.into().assign(&prev.into());
                            for (ik, &b) in BTableau::B.iter().enumerate() {
                                if b == 0.0 {
                                    continue;
                                }
                                fut.into().scaled_add(b * dt, &(&k[ik][ig]).into());
                            }
                        });
                    }
                });
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
