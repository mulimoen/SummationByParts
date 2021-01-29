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

    const DIAG_MATRIX: super::RowVector<Float, 5> =
        super::RowVector::new([[1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0]]);
    #[rustfmt::skip]
    const BLOCK: &'static [&'static [Float]] = &[
        &[-24.0/17.0, 59.0/34.0, -4.0/17.0, -3.0/34.0],
        &[-1.0/2.0, 0.0, 1.0/2.0],
        &[4.0/43.0, -59.0/86.0, 0.0, 59.0/86.0, -4.0/43.0],
        &[3.0/98.0, 0.0, -59.0/98.0, 0.0, 32.0/49.0, -4.0/49.0]
    ];
    #[rustfmt::skip]
    const BLOCK_MATRIX: super::Matrix<Float, 4, 6> = super::Matrix::new([
        [-24.0/17.0, 59.0/34.0, -4.0/17.0, -3.0/34.0, 0.0, 0.0],
        [-1.0/2.0, 0.0, 1.0/2.0, 0.0, 0.0, 0.0],
        [4.0/43.0, -59.0/86.0, 0.0, 59.0/86.0, -4.0/43.0, 0.0],
        [3.0/98.0, 0.0, -59.0/98.0, 0.0, 32.0/49.0, -4.0/49.0]
    ]);
    #[rustfmt::skip]
    const BLOCKEND_MATRIX: super::Matrix<Float, 4, 6> = super::Matrix::new([
        [4.0/49.0, -32.0/49.0, 0.0, 59.0/98.0, 0.0, -3.0/98.0 ],
        [0.0,      4.0/43.0, -59.0/86.0, 0.0, 59.0/86.0, -4.0/43.0],
        [0.0, 0.0, 0.0, -1.0/2.0, 0.0, 1.0/2.0],
        [0.0, 0.0, 3.0/34.0, 4.0/17.0, -59.0/34.0, 24.0/17.0],
    ]);

    #[rustfmt::skip]
    const D2DIAG: &'static [Float] = &[
        -1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0
    ];
    #[rustfmt::skip]
    const D2BLOCK: &'static [&'static [Float]] = &[
        &[2.0, -5.0, 4.0, -1.0],
        &[1.0, -2.0, 1.0],
        &[-4.0/43.0, 59.0/43.0, -110.0/43.0, 59.0/43.0, -4.0/43.0],
        &[-1.0/49.0, 0.0, 59.0/49.0, -118.0/49.0, 64.0/49.0, -4.0/49.0]
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

    #[cfg(feature = "sparse")]
    fn diff_matrix(&self, n: usize) -> sprs::CsMat<Float> {
        super::sparse_from_block(
            Self::BLOCK,
            Self::DIAG,
            super::Symmetry::AntiSymmetric,
            super::OperatorType::Normal,
            n,
        )
    }
    #[cfg(feature = "sparse")]
    fn h_matrix(&self, n: usize) -> sprs::CsMat<Float> {
        super::h_matrix(Self::HBLOCK, n, self.is_h2())
    }

    fn d2(&self) -> Option<&dyn super::SbpOperator1d2> {
        Some(&Self)
    }
}

fn diff_op_row_local(prev: ndarray::ArrayView2<Float>, mut fut: ndarray::ArrayViewMut2<Float>) {
    for (p, mut f) in prev
        .axis_iter(ndarray::Axis(0))
        .zip(fut.axis_iter_mut(ndarray::Axis(0)))
    {
        super::diff_op_1d_slice_matrix(
            &SBP4::BLOCK_MATRIX,
            &SBP4::BLOCKEND_MATRIX,
            &SBP4::DIAG_MATRIX,
            super::OperatorType::Normal,
            p.as_slice().unwrap(),
            f.as_slice_mut().unwrap(),
        )
    }
}

impl SbpOperator2d for SBP4 {
    fn diffxi(&self, prev: ArrayView2<Float>, mut fut: ArrayViewMut2<Float>) {
        assert_eq!(prev.shape(), fut.shape());
        assert!(prev.shape()[1] >= 2 * SBP4::BLOCK.len());

        let symmetry = super::Symmetry::AntiSymmetric;
        let optype = super::OperatorType::Normal;

        match (prev.strides(), fut.strides()) {
            ([_, 1], [_, 1]) => {
                //diff_op_row(SBP4::BLOCK, SBP4::DIAG, symmetry, optype)(prev, fut);
                diff_op_row_local(prev, fut)
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
    fn op_xi(&self) -> &dyn SbpOperator1d {
        &Self
    }
    fn op_eta(&self) -> &dyn SbpOperator1d {
        &Self
    }
}

impl super::SbpOperator1d2 for SBP4 {
    fn diff2(&self, prev: ArrayView1<Float>, mut fut: ArrayViewMut1<Float>) {
        super::diff_op_1d(
            Self::D2BLOCK,
            Self::D2DIAG,
            super::Symmetry::Symmetric,
            super::OperatorType::Normal,
            prev,
            fut.view_mut(),
        );
        let hi = (prev.len() - 1) as Float;
        fut.map_inplace(|x| *x *= hi)
    }
    fn d1(&self) -> &[Float] {
        &[-88.0 / 17.0, 144.0 / 17.0, -72.0 / 17.0, 16.0 / 17.0]
    }
    #[cfg(feature = "sparse")]
    fn diff2_matrix(&self, n: usize) -> sprs::CsMat<Float> {
        let mut m = super::sparse_from_block(
            Self::D2BLOCK,
            Self::D2DIAG,
            super::Symmetry::Symmetric,
            super::OperatorType::Normal,
            n,
        );
        let hi = (n - 1) as Float;
        m.map_inplace(|v| v * hi);
        m
    }
    #[cfg(feature = "sparse")]
    fn d1_vec(&self, n: usize, front: bool) -> sprs::CsMat<Float> {
        let d1 = &[-11.0 / 6.0, 3.0, -3.0 / 2.0, 1.0 / 3.0];
        assert!(n >= d1.len());
        let mut d1 = d1.to_vec();
        let hi = (n - 1) as Float;
        d1.iter_mut().for_each(|v| *v *= hi);

        if front {
            sprs::CsMat::new((1, n), vec![0, d1.len()], (0..d1.len()).collect(), d1)
        } else {
            // d_x => -d_x when reversed
            d1.iter_mut().for_each(|v| *v *= -1.0);
            // Indices are now in reverse order
            sprs::CsMat::new(
                (1, n),
                vec![0, d1.len()],
                (0..n).rev().take(d1.len()).collect(),
                d1,
            )
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

#[test]
fn block_equality() {
    let mut flipped_inverted = SBP4::BLOCK_MATRIX.flip();
    flipped_inverted *= &-1.0;

    assert!(flipped_inverted
        .iter()
        .zip(SBP4::BLOCKEND_MATRIX.iter())
        .all(|(x, y)| (x - y).abs() < 1e-3))
}
