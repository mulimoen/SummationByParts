use super::Float;
use ndarray::{ArrayView3, ArrayViewMut3};

pub trait ButcherTableau {
    const S: usize = Self::B.len();
    const A: &'static [&'static [Float]];
    const B: &'static [Float];
    const C: &'static [Float];
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

#[allow(clippy::too_many_arguments)]
pub fn integrate<BTableau: ButcherTableau, F, RHS>(
    mut rhs: RHS,
    prev: &F,
    fut: &mut F,
    time: &mut Float,
    dt: Float,
    k: &mut [F],
) where
    for<'r> &'r F: std::convert::Into<ArrayView3<'r, Float>>,
    for<'r> &'r mut F: std::convert::Into<ArrayViewMut3<'r, Float>>,
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

#[cfg(feature = "rayon")]
#[allow(clippy::too_many_arguments)]
pub fn integrate_multigrid<BTableau: ButcherTableau, F, RHS>(
    mut rhs: RHS,
    prev: &[F],
    fut: &mut [F],
    time: &mut Float,
    dt: Float,
    k: &mut [&mut [F]],

    pool: &rayon::ThreadPool,
) where
    for<'r> &'r F: std::convert::Into<ArrayView3<'r, Float>>,
    for<'r> &'r mut F: std::convert::Into<ArrayViewMut3<'r, Float>>,
    RHS: FnMut(&mut [F], &[F], Float),
    F: Send + Sync,
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
