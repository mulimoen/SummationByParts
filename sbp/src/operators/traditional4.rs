use super::SbpOperator;
use crate::diff_op_1d;
use crate::Float;
use ndarray::{s, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};

#[derive(Debug)]
pub struct SBP4 {}

diff_op_1d!(SBP4, diff_1d, SBP4::BLOCK, SBP4::DIAG, false);

impl SBP4 {
    #[rustfmt::skip]
    const HBLOCK: &'static [Float] = &[
        17.0 / 48.0, 59.0 / 48.0, 43.0 / 48.0, 49.0 / 48.0,
    ];
    #[rustfmt::skip]
    const DIAG: &'static [Float] = &[
        1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0,
    ];
    #[rustfmt::skip]
    const BLOCK: &'static [[Float; 6]] = &[
        [-1.41176470588235e+00, 1.73529411764706e+00, -2.35294117647059e-01, -8.82352941176471e-02, 0.00000000000000e+00, 0.00000000000000e+00],
        [-5.00000000000000e-01, 0.00000000000000e+00, 5.00000000000000e-01, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],
        [9.30232558139535e-02, -6.86046511627907e-01, 0.00000000000000e+00, 6.86046511627907e-01, -9.30232558139535e-02, 0.00000000000000e+00],
        [3.06122448979592e-02, 0.00000000000000e+00, -6.02040816326531e-01, 0.00000000000000e+00, 6.53061224489796e-01, -8.16326530612245e-02],
    ];
}

impl SbpOperator for SBP4 {
    fn diffxi(prev: ArrayView2<Float>, mut fut: ArrayViewMut2<Float>) {
        assert_eq!(prev.shape(), fut.shape());
        assert!(prev.shape()[1] >= 2 * Self::BLOCK.len());

        for (r0, r1) in prev.outer_iter().zip(fut.outer_iter_mut()) {
            Self::diff_1d(r0, r1);
        }
    }

    fn diffeta(prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>) {
        // transpose then use diffxi
        Self::diffxi(prev.reversed_axes(), fut.reversed_axes());
    }

    fn h() -> &'static [Float] {
        Self::HBLOCK
    }
}

#[test]
fn test_trad4() {
    use super::testing::*;
    use super::*;
    let nx = 20;
    let ny = 13;

    check_operator_on::<SBP4, _, _, _>((ny, nx), |x, y| x + 2.0 * y, |_, _| 1.0, |_, _| 2.0, 1e-4);
    check_operator_on::<SBP4, _, _, _>(
        (ny, nx),
        |x, y| x * x + 2.0 * x * y + 3.0 * y * y,
        |x, y| 2.0 * x + 2.0 * y,
        |x, y| 2.0 * x + 6.0 * y,
        1e-3,
    );
    check_operator_on::<SBP4, _, _, _>(
        (ny, nx),
        |x, y| x.powi(3) + 2.0 * x.powi(2) * y + 3.0 * x * y.powi(2) + 4.0 * y.powi(3),
        |x, y| 3.0 * x.powi(2) + 4.0 * x * y + 3.0 * y.powi(2),
        |x, y| 2.0 * x.powi(2) + 6.0 * x * y + 12.0 * y.powi(2),
        1e-1,
    );
}
