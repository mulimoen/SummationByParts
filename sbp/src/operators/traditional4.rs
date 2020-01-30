use super::SbpOperator;
use crate::diff_op_1d;
use ndarray::{s, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};

pub struct SBP4 {}

diff_op_1d!(SBP4, diff_1d, SBP4::BLOCK, SBP4::DIAG, false);

impl SBP4 {
    #[rustfmt::skip]
    const HBLOCK: &'static [f32] = &[
        17.0 / 48.0, 59.0 / 48.0, 43.0 / 48.0, 49.0 / 48.0,
    ];
    #[rustfmt::skip]
    const DIAG: &'static [f32] = &[
        1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0,
    ];
    #[rustfmt::skip]
    const BLOCK: &'static [[f32; 6]] = &[
        [-1.41176470588235e+00, 1.73529411764706e+00, -2.35294117647059e-01, -8.82352941176471e-02, 0.00000000000000e+00, 0.00000000000000e+00],
        [-5.00000000000000e-01, 0.00000000000000e+00, 5.00000000000000e-01, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],
        [9.30232558139535e-02, -6.86046511627907e-01, 0.00000000000000e+00, 6.86046511627907e-01, -9.30232558139535e-02, 0.00000000000000e+00],
        [3.06122448979592e-02, 0.00000000000000e+00, -6.02040816326531e-01, 0.00000000000000e+00, 6.53061224489796e-01, -8.16326530612245e-02],
    ];
}

impl SbpOperator for SBP4 {
    fn diffxi(prev: ArrayView2<f32>, mut fut: ArrayViewMut2<f32>) {
        assert_eq!(prev.shape(), fut.shape());
        assert!(prev.shape()[1] >= 2 * Self::BLOCK.len());

        for (r0, r1) in prev.outer_iter().zip(fut.outer_iter_mut()) {
            Self::diff_1d(r0, r1);
        }
    }

    fn diffeta(prev: ArrayView2<f32>, fut: ArrayViewMut2<f32>) {
        // transpose then use diffxi
        Self::diffxi(prev.reversed_axes(), fut.reversed_axes());
    }

    fn h() -> &'static [f32] {
        Self::HBLOCK
    }
}
