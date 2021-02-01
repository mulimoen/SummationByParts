use super::{
    BlockMatrix, DiagonalMatrix, Matrix, OperatorType, RowVector, SbpOperator1d, SbpOperator2d,
};
use crate::Float;
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};

#[derive(Debug, Copy, Clone)]
pub struct SBP4;

impl SBP4 {
    #[rustfmt::skip]
    const H: DiagonalMatrix<4> = DiagonalMatrix::new([
        17.0 / 48.0, 59.0 / 48.0, 43.0 / 48.0, 49.0 / 48.0,
    ]);

    #[rustfmt::skip]
    const DIFF_DIAG: RowVector<Float, 5> = RowVector::new([[
        1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0
    ]]);
    #[rustfmt::skip]
    const DIFF_BLOCK: Matrix<Float, 4, 6> = Matrix::new([
        [-24.0/17.0, 59.0/34.0, -4.0/17.0, -3.0/34.0, 0.0, 0.0],
        [-1.0/2.0, 0.0, 1.0/2.0, 0.0, 0.0, 0.0],
        [4.0/43.0, -59.0/86.0, 0.0, 59.0/86.0, -4.0/43.0, 0.0],
        [3.0/98.0, 0.0, -59.0/98.0, 0.0, 32.0/49.0, -4.0/49.0]
    ]);
    const DIFF_BLOCKEND: super::Matrix<Float, 4, 6> =
        super::flip_sign(super::flip_ud(super::flip_lr(Self::DIFF_BLOCK)));

    const DIFF: BlockMatrix<4, 6, 5> =
        BlockMatrix::new(Self::DIFF_BLOCK, Self::DIFF_DIAG, Self::DIFF_BLOCKEND);

    #[rustfmt::skip]
    const D2DIAG: RowVector<Float, 5> = RowVector::new([[
        -1.0 / 12.0, 4.0 / 3.0, -5.0 / 2.0, 4.0 / 3.0, -1.0 / 12.0
    ]]);
    #[rustfmt::skip]
    const D2BLOCK: Matrix<Float, 4, 6> = Matrix::new([
        [2.0, -5.0, 4.0, -1.0, 0.0, 0.0],
        [1.0, -2.0, 1.0, 0.0, 0.0, 0.0],
        [-4.0/43.0, 59.0/43.0, -110.0/43.0, 59.0/43.0, -4.0/43.0, 0.0],
        [-1.0/49.0, 0.0, 59.0/49.0, -118.0/49.0, 64.0/49.0, -4.0/49.0]
    ]);
    const D2: BlockMatrix<4, 6, 5> = BlockMatrix::new(
        Self::D2BLOCK,
        Self::D2DIAG,
        super::flip_ud(super::flip_lr(Self::D2BLOCK)),
    );
}

impl SbpOperator1d for SBP4 {
    fn diff(&self, prev: ArrayView1<Float>, fut: ArrayViewMut1<Float>) {
        super::diff_op_1d(&Self::DIFF, super::OperatorType::Normal, prev, fut)
    }

    fn h(&self) -> &'static [Float] {
        &Self::H.start
    }

    #[cfg(feature = "sparse")]
    fn diff_matrix(&self, n: usize) -> sprs::CsMat<Float> {
        super::sparse_from_block(&Self::DIFF, super::OperatorType::Normal, n)
    }
    #[cfg(feature = "sparse")]
    fn h_matrix(&self, n: usize) -> sprs::CsMat<Float> {
        super::h_matrix(&Self::H, n, self.is_h2())
    }

    fn d2(&self) -> Option<&dyn super::SbpOperator1d2> {
        Some(&Self)
    }
}

impl SbpOperator2d for SBP4 {
    fn diffxi(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>) {
        assert_eq!(prev.shape(), fut.shape());
        super::diff_op_2d(&Self::DIFF, OperatorType::Normal, prev, fut)
    }
    fn op_xi(&self) -> &dyn SbpOperator1d {
        &Self
    }
}

impl super::SbpOperator1d2 for SBP4 {
    fn diff2(&self, prev: ArrayView1<Float>, mut fut: ArrayViewMut1<Float>) {
        super::diff_op_1d(&Self::D2, OperatorType::Normal, prev, fut.view_mut());
        let hi = (prev.len() - 1) as Float;
        fut.map_inplace(|x| *x *= hi)
    }
    fn d1(&self) -> &[Float] {
        &[-88.0 / 17.0, 144.0 / 17.0, -72.0 / 17.0, 16.0 / 17.0]
    }
    #[cfg(feature = "sparse")]
    fn diff2_matrix(&self, n: usize) -> sprs::CsMat<Float> {
        let mut m = super::sparse_from_block(&Self::D2, OperatorType::Normal, n);
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
