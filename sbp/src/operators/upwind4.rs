use super::{
    BlockMatrix, DiagonalMatrix, Matrix, OperatorType, RowVector, SbpOperator1d, SbpOperator2d,
    UpwindOperator1d, UpwindOperator2d,
};
use crate::Float;
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};

#[derive(Debug, Copy, Clone)]
pub struct Upwind4;

impl Upwind4 {
    #[rustfmt::skip]
    const H: DiagonalMatrix<4> = DiagonalMatrix::new([
        49.0 / 144.0, 61.0 / 48.0, 41.0 / 48.0, 149.0 / 144.0
    ]);
    #[rustfmt::skip]
    const DIFF_BLOCK: Matrix<Float, 4, 7> = Matrix::new([
        [  -72.0 / 49.0, 187.0 / 98.0,   -20.0 / 49.0,   -3.0 / 98.0,           0.0,           0.0,         0.0],
        [-187.0 / 366.0,          0.0,   69.0 / 122.0, -16.0 / 183.0,    2.0 / 61.0,           0.0,         0.0],
        [  20.0 / 123.0, -69.0 / 82.0,            0.0, 227.0 / 246.0,  -12.0 / 41.0,    2.0 / 41.0,         0.0],
        [   3.0 / 298.0, 16.0 / 149.0, -227.0 / 298.0,           0.0, 126.0 / 149.0, -36.0 / 149.0, 6.0 / 149.0],
    ]);
    #[rustfmt::skip]
    const DIFF_DIAG: RowVector<Float, 7> = RowVector::new([[
        -1.0 / 24.0, 1.0 / 4.0, -7.0 / 8.0, 0.0, 7.0 / 8.0, -1.0 / 4.0, 1.0 / 24.0
    ]]);
    const DIFF_BLOCKEND: Matrix<Float, 4, 7> = Self::DIFF_BLOCK.flip_lr().flip_ud().flip_sign();

    const DIFF: BlockMatrix<Float, 4, 7, 7> =
        BlockMatrix::new(Self::DIFF_BLOCK, Self::DIFF_DIAG, Self::DIFF_BLOCKEND);

    #[rustfmt::skip]
    const DISS_BLOCK: Matrix<Float, 4, 7> = Matrix::new([
        [-3.0 / 49.0,    9.0 / 49.0,  -9.0 / 49.0,     3.0 / 49.0,          0.0,           0.0,         0.0],
        [ 3.0 / 61.0,  -11.0 / 61.0,  15.0 / 61.0,    -9.0 / 61.0,   2.0 / 61.0,           0.0,         0.0],
        [-3.0 / 41.0,   15.0 / 41.0, -29.0 / 41.0,    27.0 / 41.0, -12.0 / 41.0,    2.0 / 41.0,         0.0],
        [3.0 / 149.0, -27.0 / 149.0, 81.0 / 149.0, -117.0 / 149.0, 90.0 / 149.0, -36.0 / 149.0, 6.0 / 149.0],
    ]);
    #[rustfmt::skip]
    const DISS_DIAG: RowVector<Float, 7> = Matrix::new([[
        1.0 / 24.0, -1.0 / 4.0, 5.0 / 8.0, -5.0 / 6.0, 5.0 / 8.0, -1.0 / 4.0, 1.0 / 24.0
    ]]);
    const DISS_BLOCKEND: Matrix<Float, 4, 7> = Self::DISS_BLOCK.flip_lr().flip_ud();

    const DISS: BlockMatrix<Float, 4, 7, 7> =
        BlockMatrix::new(Self::DISS_BLOCK, Self::DISS_DIAG, Self::DISS_BLOCKEND);
}

impl SbpOperator1d for Upwind4 {
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

    fn upwind(&self) -> Option<&dyn UpwindOperator1d> {
        Some(&Self)
    }
}

impl SbpOperator2d for Upwind4 {
    fn diffxi(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>) {
        assert_eq!(prev.shape(), fut.shape());
        super::diff_op_2d(&Self::DIFF, OperatorType::Normal, prev, fut);
    }
    fn op_xi(&self) -> &dyn SbpOperator1d {
        &Self
    }
    fn upwind(&self) -> Option<Box<dyn UpwindOperator2d>> {
        Some(Box::new(Self))
    }
}

#[test]
fn upwind4_test() {
    use ndarray::prelude::*;
    let nx = 20;
    let dx = 1.0 / (nx - 1) as Float;
    let mut source: ndarray::Array1<Float> = ndarray::Array1::zeros(nx);
    let mut res = ndarray::Array1::zeros(nx);
    let mut target = ndarray::Array1::zeros(nx);

    for i in 0..nx {
        source[i] = i as Float * dx;
        target[i] = 1.0;
    }
    res.fill(0.0);
    Upwind4.diff(source.view(), res.view_mut());
    approx::assert_abs_diff_eq!(&res, &target, epsilon = 1e-4);
    {
        let source = source.to_owned().insert_axis(ndarray::Axis(0));
        let mut res = res.to_owned().insert_axis(ndarray::Axis(0));
        let target = target.to_owned().insert_axis(ndarray::Axis(0));
        res.fill(0.0);
        Upwind4.diffxi(source.view(), res.view_mut());
        approx::assert_abs_diff_eq!(&res, &target, epsilon = 1e-2);
    }

    {
        let source = Array2::from_shape_fn((nx, 8), |(i, _)| source[i]);
        let target = Array2::from_shape_fn((nx, 8), |(i, _)| target[i]);
        let mut res = Array2::zeros((nx, 8));
        res.fill(0.0);
        Upwind4.diffeta(source.view(), res.view_mut());
        approx::assert_abs_diff_eq!(&res.to_owned(), &target.to_owned(), epsilon = 1e-2);
    }

    for i in 0..nx {
        let x = i as Float * dx;
        source[i] = x * x;
        target[i] = 2.0 * x;
    }
    res.fill(0.0);
    Upwind4.diff(source.view(), res.view_mut());
    approx::assert_abs_diff_eq!(&res, &target, epsilon = 1e-4);
    {
        let source = source.to_owned().insert_axis(ndarray::Axis(0));
        let mut res = res.to_owned().insert_axis(ndarray::Axis(0));
        let target = target.to_owned().insert_axis(ndarray::Axis(0));
        res.fill(0.0);
        Upwind4.diffxi(source.view(), res.view_mut());
        approx::assert_abs_diff_eq!(&res, &target, epsilon = 1e-2);
    }

    {
        let source = Array2::from_shape_fn((nx, 8), |(i, _)| source[i]);
        let target = Array2::from_shape_fn((nx, 8), |(i, _)| target[i]);
        let mut res = Array2::zeros((nx, 8));
        res.fill(0.0);
        Upwind4.diffeta(source.view(), res.view_mut());
        approx::assert_abs_diff_eq!(&res.to_owned(), &target.to_owned(), epsilon = 1e-2);
    }

    for i in 0..nx {
        let x = i as Float * dx;
        source[i] = x * x * x;
        target[i] = 3.0 * x * x;
    }
    res.fill(0.0);
    Upwind4.diff(source.view(), res.view_mut());
    approx::assert_abs_diff_eq!(&res, &target, epsilon = 1e-2);

    {
        let source = source.to_owned().insert_axis(ndarray::Axis(0));
        let mut res = res.to_owned().insert_axis(ndarray::Axis(0));
        let target = target.to_owned().insert_axis(ndarray::Axis(0));
        res.fill(0.0);
        Upwind4.diffxi(source.view(), res.view_mut());
        approx::assert_abs_diff_eq!(&res, &target, epsilon = 1e-2);
    }

    {
        let source = Array2::from_shape_fn((nx, 8), |(i, _)| source[i]);
        let target = Array2::from_shape_fn((nx, 8), |(i, _)| target[i]);
        let mut res = Array2::zeros((nx, 8));
        res.fill(0.0);
        Upwind4.diffeta(source.view(), res.view_mut());
        approx::assert_abs_diff_eq!(&res.to_owned(), &target.to_owned(), epsilon = 1e-2);
    }
}

impl UpwindOperator1d for Upwind4 {
    fn diss(&self, prev: ArrayView1<Float>, fut: ArrayViewMut1<Float>) {
        super::diff_op_1d(&Self::DISS, super::OperatorType::Normal, prev, fut)
    }

    fn as_sbp(&self) -> &dyn SbpOperator1d {
        self
    }

    #[cfg(feature = "sparse")]
    fn diss_matrix(&self, n: usize) -> sprs::CsMat<Float> {
        super::sparse_from_block(&Self::DISS, super::OperatorType::Normal, n)
    }
}

impl UpwindOperator2d for Upwind4 {
    fn dissxi(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>) {
        assert_eq!(prev.shape(), fut.shape());

        super::diff_op_2d(&Self::DISS, OperatorType::Normal, prev, fut)
    }

    fn op_xi(&self) -> &dyn UpwindOperator1d {
        &Self
    }
}

#[test]
fn upwind4_test2() {
    use super::testing::*;
    use super::*;
    let nx = 32;
    let ny = 16;

    check_operator_on(
        Upwind4,
        (ny, nx),
        |x, y| x + 2.0 * y,
        |_, _| 1.0,
        |_, _| 2.0,
        1e-4,
    );
    check_operator_on(
        Upwind4,
        (ny, nx),
        |x, y| x * x + 2.0 * x * y + 3.0 * y * y,
        |x, y| 2.0 * x + 2.0 * y,
        |x, y| 2.0 * x + 6.0 * y,
        1e-3,
    );
    check_operator_on(
        Upwind4,
        (ny, nx),
        |x, y| x.powi(3) + 2.0 * x.powi(2) * y + 3.0 * x * y.powi(2) + 4.0 * y.powi(3),
        |x, y| 3.0 * x.powi(2) + 4.0 * x * y + 3.0 * y.powi(2),
        |x, y| 2.0 * x.powi(2) + 6.0 * x * y + 12.0 * y.powi(2),
        1e-1,
    );
    check_operator_on(
        Upwind4,
        (32, 32),
        |x, y| x.powi(3) + 2.0 * x.powi(2) * y + 3.0 * x * y.powi(2) + 4.0 * y.powi(3),
        |x, y| 3.0 * x.powi(2) + 4.0 * x * y + 3.0 * y.powi(2),
        |x, y| 2.0 * x.powi(2) + 6.0 * x * y + 12.0 * y.powi(2),
        1e-1,
    );
}
