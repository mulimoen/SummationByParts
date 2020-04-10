use super::Float;
use ndarray::{Array3, Zip};

pub(crate) fn rk4<'a, F: 'a, RHS, MT, C>(
    rhs: RHS,
    prev: &F,
    fut: &mut F,
    time: &mut Float,
    dt: Float,
    k: &mut [F; 4],

    constants: C,
    mut mutables: &mut MT,
) where
    C: Copy,
    F: std::ops::Deref<Target = Array3<Float>> + std::ops::DerefMut<Target = Array3<Float>>,
    RHS: Fn(&mut F, &F, Float, C, &mut MT),
{
    assert_eq!(prev.shape(), fut.shape());

    for i in 0.. {
        let simtime;
        match i {
            0 => {
                fut.assign(prev);
                simtime = *time;
            }
            1 | 2 => {
                fut.assign(prev);
                fut.scaled_add(1.0 / 2.0 * dt, &k[i - 1]);
                simtime = *time + dt / 2.0;
            }
            3 => {
                fut.assign(prev);
                fut.scaled_add(dt, &k[i - 1]);
                simtime = *time + dt;
            }
            4 => {
                Zip::from(&mut **fut)
                    .and(&**prev)
                    .and(&*k[0])
                    .and(&*k[1])
                    .and(&*k[2])
                    .and(&*k[3])
                    .apply(|y1, &y0, &k1, &k2, &k3, &k4| {
                        *y1 = y0 + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                    });
                *time += dt;
                return;
            }
            _ => {
                unreachable!();
            }
        };

        rhs(&mut k[i], &fut, simtime, constants, &mut mutables);
    }
}
