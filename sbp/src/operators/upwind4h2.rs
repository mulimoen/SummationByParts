use super::{
    BlockMatrix, DiagonalMatrix, Matrix, OperatorType, RowVector, SbpOperator1d, SbpOperator2d,
    UpwindOperator1d, UpwindOperator2d,
};
use crate::Float;
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};

#[derive(Debug, Copy, Clone)]
pub struct Upwind4h2;

impl Upwind4h2 {
    #[rustfmt::skip]
    const H: DiagonalMatrix<4> = DiagonalMatrix::new([
        0.91e2/0.720e3, 0.325e3/0.384e3, 0.595e3/0.576e3, 0.1909e4/0.1920e4,
    ]);
    #[rustfmt::skip]
    const DIFF_BLOCK: Matrix<Float, 4, 7> = Matrix::new([
        [-3.95604395604396e+00, 5.41758241758242e+00, -1.94505494505495e+00, 4.83516483516484e-01, 0.0, 0.0, 0.0],
        [-8.09025641025641e-01, 0.00000000000000e+00, 1.03948717948718e+00, -2.47384615384615e-01, 1.69230769230769e-02, 0.0, 0.0],
        [2.37983193277311e-01, -8.51680672268908e-01, 0.00000000000000e+00, 7.35966386554622e-01, -1.36134453781513e-01, 1.38655462184874e-02, 0.0],
        [-6.14632442814737e-02, 2.10581456259822e-01, -7.64623712240265e-01, 0.00000000000000e+00, 7.42535358826611e-01, -1.41435306443164e-01, 1.44054478784704e-02],
    ]);
    #[rustfmt::skip]
    const DIFF_DIAG: RowVector<Float, 7> = RowVector::new([[
        -1.43229166666667e-02, 1.40625000000000e-01, -7.38281250000000e-01, 0.00000000000000e+00, 7.38281250000000e-01, -1.40625000000000e-01, 1.43229166666667e-02
    ]]);
    const DIFF: BlockMatrix<4, 7, 7> = BlockMatrix::new(
        Self::DIFF_BLOCK,
        Self::DIFF_DIAG,
        super::flip_sign(super::flip_ud(super::flip_lr(Self::DIFF_BLOCK))),
    );

    #[rustfmt::skip]
    const DISS_BLOCK: Matrix<Float, 4, 7> = Matrix::new([
        [-2.76989342315976e-01, 5.19355016842454e-01, -3.46236677894969e-01, 1.03871003368491e-01, 0.0, 0.0, 0.0],
        [7.75570158484731e-02, -1.62342481638964e-01, 1.47715500579822e-01, -7.98531117124082e-02, 1.69230769230769e-02, 0.0, 0.0],
        [-4.23630758836198e-02, 1.21027405937249e-01, -1.91609307039399e-01, 1.82272708078206e-01, -8.31932773109244e-02, 1.38655462184874e-02, 0.0],
        [1.32037874021759e-02, -6.79734450144910e-02, 1.89370108794365e-01, -2.78654929966754e-01, 2.16081718177056e-01, -8.64326872708224e-02, 1.44054478784704e-02],
    ]);

    #[rustfmt::skip]
    const DISS_DIAG: RowVector<Float, 7> = RowVector::new([[
        1.43229166666667e-02, -8.59375000000000e-02, 2.14843750000000e-01, -2.86458333333333e-01, 2.14843750000000e-01, -8.59375000000000e-02, 1.43229166666667e-02,
    ]]);
    const DISS: BlockMatrix<4, 7, 7> = BlockMatrix::new(
        Self::DISS_BLOCK,
        Self::DISS_DIAG,
        super::flip_ud(super::flip_lr(Self::DISS_BLOCK)),
    );
}

impl SbpOperator1d for Upwind4h2 {
    fn diff(&self, prev: ArrayView1<Float>, fut: ArrayViewMut1<Float>) {
        super::diff_op_1d(&Self::DIFF, OperatorType::H2, prev, fut)
    }

    fn h(&self) -> &'static [Float] {
        &Self::H.start
    }
    fn is_h2(&self) -> bool {
        true
    }

    #[cfg(feature = "sparse")]
    fn diff_matrix(&self, n: usize) -> sprs::CsMat<Float> {
        super::sparse_from_block(&Self::DIFF, OperatorType::H2, n)
    }
    #[cfg(feature = "sparse")]
    fn h_matrix(&self, n: usize) -> sprs::CsMat<Float> {
        super::h_matrix(&Self::H, n, self.is_h2())
    }

    fn upwind(&self) -> Option<&dyn UpwindOperator1d> {
        Some(&Self)
    }
}

impl SbpOperator2d for Upwind4h2 {
    fn diffxi(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>) {
        assert_eq!(prev.shape(), fut.shape());

        super::diff_op_2d(&Self::DIFF, OperatorType::H2, prev, fut)
    }
    fn op_xi(&self) -> &dyn SbpOperator1d {
        &Self
    }
    fn upwind(&self) -> Option<Box<dyn UpwindOperator2d>> {
        Some(Box::new(Self))
    }
}

impl UpwindOperator2d for Upwind4h2 {
    fn dissxi(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>) {
        assert_eq!(prev.shape(), fut.shape());

        super::diff_op_2d(&Self::DISS, OperatorType::H2, prev, fut)
    }
    fn op_xi(&self) -> &dyn UpwindOperator1d {
        &Self
    }
}

#[test]
fn upwind4h2_test() {
    let nx = 20;

    let x = crate::utils::h2linspace(0.0, 1.0, nx);

    let mut res = ndarray::Array1::zeros(nx);

    Upwind4h2.diff(x.view(), res.view_mut());
    let ans = &x * 0.0 + 1.0;
    approx::assert_abs_diff_eq!(&res, &ans, epsilon = 1e-4);

    res.fill(0.0);
    let y = &x * &x / 2.0;
    Upwind4h2.diff(y.view(), res.view_mut());
    let ans = &x;
    approx::assert_abs_diff_eq!(&res, &ans, epsilon = 1e-4);

    res.fill(0.0);
    let y = &x * &x * &x / 3.0;
    Upwind4h2.diff(y.view(), res.view_mut());
    let ans = &x * &x;
    approx::assert_abs_diff_eq!(&res, &ans, epsilon = 1e-2);
}

impl UpwindOperator1d for Upwind4h2 {
    fn diss(&self, prev: ArrayView1<Float>, fut: ArrayViewMut1<Float>) {
        super::diff_op_1d(&Self::DISS, super::OperatorType::H2, prev, fut)
    }

    fn as_sbp(&self) -> &dyn SbpOperator1d {
        self
    }

    #[cfg(feature = "sparse")]
    fn diss_matrix(&self, n: usize) -> sprs::CsMat<Float> {
        super::sparse_from_block(&Self::DISS, super::OperatorType::H2, n)
    }
}
