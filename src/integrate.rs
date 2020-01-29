use super::grid::Grid;
use super::operators::SbpOperator;
use ndarray::{Array3, Zip};

pub(crate) fn integrate_rk4<'a, F: 'a, SBP, RHS, WB>(
    rhs: RHS,
    prev: &F,
    fut: &mut F,
    dt: f32,
    grid: &Grid<SBP>,
    k: &mut [F; 4],
    mut wb: &mut WB,
) where
    F: std::ops::Deref<Target = Array3<f32>> + std::ops::DerefMut<Target = Array3<f32>>,
    SBP: SbpOperator,
    RHS: Fn(&mut F, &F, &Grid<SBP>, &mut WB),
{
    assert_eq!(prev.shape(), fut.shape());

    for i in 0..4 {
        // y = y0 + c*kn
        fut.assign(prev);
        match i {
            0 => {}
            1 | 2 => {
                fut.scaled_add(1.0 / 2.0 * dt, &k[i - 1]);
            }
            3 => {
                fut.scaled_add(dt, &k[i - 1]);
            }
            _ => {
                unreachable!();
            }
        };

        rhs(&mut k[i], &fut, grid, &mut wb);
    }

    Zip::from(&mut **fut)
        .and(&**prev)
        .and(&*k[0])
        .and(&*k[1])
        .and(&*k[2])
        .and(&*k[3])
        .apply(|y1, &y0, &k1, &k2, &k3, &k4| *y1 = y0 + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4));
}
