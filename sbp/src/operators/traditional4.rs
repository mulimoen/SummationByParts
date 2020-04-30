use super::{diff_op_col, diff_op_row, SbpOperator1d, SbpOperator2d};
use crate::Float;
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};

#[derive(Debug, Copy, Clone)]
pub struct SBP4;

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
    const BLOCK: &'static [&'static [Float]] = &[
        &[-1.41176470588235e+00, 1.73529411764706e+00, -2.35294117647059e-01, -8.82352941176471e-02],
        &[-5.00000000000000e-01, 0.00000000000000e+00, 5.00000000000000e-01],
        &[9.30232558139535e-02, -6.86046511627907e-01, 0.00000000000000e+00, 6.86046511627907e-01, -9.30232558139535e-02],
        &[3.06122448979592e-02, 0.00000000000000e+00, -6.02040816326531e-01, 0.00000000000000e+00, 6.53061224489796e-01, -8.16326530612245e-02],
    ];
}

impl SbpOperator1d for SBP4 {
    fn diff(&self, prev: ArrayView1<Float>, fut: ArrayViewMut1<Float>) {
        super::diff_op_1d(
            Self::BLOCK,
            Self::DIAG,
            super::Symmetry::AntiSymmetric,
            super::OperatorType::Normal,
            prev,
            fut,
        )
    }

    fn h(&self) -> &'static [Float] {
        Self::HBLOCK
    }
}

impl<SBP: SbpOperator1d> SbpOperator2d for (&SBP, &SBP4) {
    fn diffxi(&self, prev: ArrayView2<Float>, mut fut: ArrayViewMut2<Float>) {
        assert_eq!(prev.shape(), fut.shape());
        assert!(prev.shape()[1] >= 2 * SBP4::BLOCK.len());

        let symmetry = super::Symmetry::AntiSymmetric;
        let optype = super::OperatorType::Normal;

        match (prev.strides(), fut.strides()) {
            ([_, 1], [_, 1]) => {
                diff_op_row(SBP4::BLOCK, SBP4::DIAG, symmetry, optype)(prev, fut);
            }
            ([1, _], [1, _]) => {
                diff_op_col(SBP4::BLOCK, SBP4::DIAG, symmetry, optype)(prev, fut);
            }
            ([_, _], [_, _]) => {
                // Fallback, work row by row
                for (r0, r1) in prev.outer_iter().zip(fut.outer_iter_mut()) {
                    SBP4.diff(r0, r1);
                }
            }
            _ => unreachable!("Should only be two elements in the strides vectors"),
        }
    }
}

#[test]
fn test_trad4() {
    use super::testing::*;
    use super::*;
    let nx = 20;
    let ny = 13;

    check_operator_on(
        SBP4,
        (ny, nx),
        |x, y| x + 2.0 * y,
        |_, _| 1.0,
        |_, _| 2.0,
        1e-4,
    );
    check_operator_on(
        SBP4,
        (ny, nx),
        |x, y| x * x + 2.0 * x * y + 3.0 * y * y,
        |x, y| 2.0 * x + 2.0 * y,
        |x, y| 2.0 * x + 6.0 * y,
        1e-3,
    );
    check_operator_on(
        SBP4,
        (ny, nx),
        |x, y| x.powi(3) + 2.0 * x.powi(2) * y + 3.0 * x * y.powi(2) + 4.0 * y.powi(3),
        |x, y| 3.0 * x.powi(2) + 4.0 * x * y + 3.0 * y.powi(2),
        |x, y| 2.0 * x.powi(2) + 6.0 * x * y + 12.0 * y.powi(2),
        1e-1,
    );
}
