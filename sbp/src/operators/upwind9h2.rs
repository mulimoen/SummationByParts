use super::{
    BlockMatrix, DiagonalMatrix, Matrix, OperatorType, RowVector, SbpOperator1d, SbpOperator2d,
    UpwindOperator1d, UpwindOperator2d,
};
use crate::Float;
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};

#[derive(Debug, Copy, Clone)]
pub struct Upwind9h2;

impl Upwind9h2 {
    #[rustfmt::skip]
    const H: DiagonalMatrix<8> = DiagonalMatrix::new([
        0.13528301e8/0.97297200e8, 0.189103813e9/0.232243200e9, 0.125336387e9/0.116121600e9, 0.24412231e8/0.25804800e8, 0.11976149e8/0.11612160e8, 0.27510113e8/0.27869184e8, 0.142384289e9/0.141926400e9, 0.3018054133e10/0.3019161600e10
    ]);
    #[rustfmt::skip]
    const DIFF_DIAG: RowVector<Float, 11> = RowVector::new([[
        -7.93650793650794e-04, 9.92063492063492e-03, -5.95238095238095e-02, 2.38095238095238e-01, -8.33333333333333e-01, 0.00000000000000e+00, 8.33333333333333e-01, -2.38095238095238e-01, 5.95238095238095e-02, -9.92063492063492e-03, 7.93650793650794e-04
    ]]);
    #[rustfmt::skip]
    const DIFF_BLOCK: Matrix<Float, 8, 13> = Matrix::new([
        [-3.59606132359119e+00, 4.89833536312447e+00, -1.88949383693789e+00, 5.47742030095865e-01, 1.98323891961440e-01, -1.81919774755160e-01, 9.75536252790282e-03, 1.33182875745630e-02, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-8.36438764514407e-01, 0.00000000000000e+00, 1.13444534067338e+00, -2.98589126517177e-01, -5.64158604558766e-02, 6.76718494319307e-02, -7.11434545077044e-03, -3.55909316707803e-03, 0.0, 0.0, 0.0, 0.0, 0.0],
        [2.43402051702672e-01, -8.55808694890095e-01, 0.00000000000000e+00, 7.10023434316597e-01, -8.34706840757189e-02, -2.34098045062645e-02, 9.87675724649922e-03, -6.13059793689851e-04, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-8.05029896098980e-02, 2.56994775122436e-01, -8.10083655220697e-01, 0.00000000000000e+00, 7.86933049074895e-01, -1.76732206311904e-01, 2.52232469669152e-02, -2.67114375632805e-03, 8.38923734582063e-04, 0.0, 0.0, 0.0, 0.0],
        [-2.67370674924706e-02, 4.45404208225958e-02, 8.73562441688813e-02, -7.21839392529084e-01, 0.00000000000000e+00, 7.74603212862825e-01, -2.00161722480411e-01, 5.10878939438559e-02, -9.61911880020865e-03, 7.69529504016692e-04, 0.0, 0.0, 0.0],
        [2.56244589039153e-02, -5.58209474181657e-02, 2.55972803937942e-02, 1.69377044771715e-01, -8.09310829929974e-01, 0.00000000000000e+00, 8.33417128898551e-01, -2.39938756878570e-01, 6.03007337701593e-02, -1.00501222950266e-02, 8.04009783602125e-04, 0.0, 0.0],
        [-1.35203347497451e-03, 5.77422023528126e-03, -1.06262408660325e-02, -2.37853245409338e-02, 2.05772021953463e-01, -8.20033622612654e-01, 0.00000000000000e+00, 8.31345778788959e-01, -2.37329555369694e-01, 5.93323888424235e-02, -9.88873147373725e-03, 7.91098517898980e-04, 0.0],
        [-1.85246767034256e-03, 2.89905176240824e-03, 6.61951741227212e-04, 2.52792141498726e-03, -5.27086038822990e-02, 2.36934258275492e-01, -8.34333946306806e-01, 0.00000000000000e+00, 8.33639122800983e-01, -2.38182606514566e-01, 5.95456516286416e-02, -9.92427527144027e-03, 7.93942021715222e-04]
    ]);
    const DIFF: BlockMatrix<Float, 8, 13, 11> = BlockMatrix::new(
        Self::DIFF_BLOCK,
        Self::DIFF_DIAG,
        Self::DIFF_BLOCK.flip_lr().flip_ud().flip_sign(),
    );

    #[rustfmt::skip]
    const DISS_BLOCK: Matrix<Float, 8, 13>  = Matrix::new([
        [-9.67684432055993e-03, 2.46709290489728e-02, -3.59750445192099e-02, 3.68391266633410e-02, -2.15642540957128e-02, 6.31810399460400e-03, -5.50815969583136e-04, -6.12008018523449e-05, 0.0, 0.0, 0.0, 0.0, 0.0],
        [4.21280289799980e-03, -1.11651391003720e-02, 1.78139108670387e-02, -2.04287733194061e-02, 1.39229889120575e-02, -5.16220078628933e-03, 8.08589544070426e-04, -2.17901509897605e-06, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-4.63425679136442e-03, 1.34385494509234e-02, -2.60541352669482e-02, 3.74164435120410e-02, -3.36394001786921e-02, 1.82199905150995e-02, -5.42550576921788e-03, 6.78314528158575e-04, 0.0, 0.0, 0.0, 0.0, 0.0],
        [5.41433680102586e-03, -1.75829845731036e-02, 4.26893646894443e-02, -7.75663958780684e-02, 9.06986463369770e-02, -6.74666360933304e-02, 3.07997352073168e-02, -7.82499022484376e-03, 8.38923734582063e-04, 0.0, 0.0, 0.0, 0.0],
        [-2.90718839510249e-03, 1.09922241766815e-02, -3.52053141560315e-02, 8.31962208882433e-02, -1.27244413706703e-01, 1.27078393791315e-01, -8.23947157593937e-02, 3.34105586971409e-02, -7.69529504016692e-03, 7.69529504016692e-04, 0.0, 0.0, 0.0],
        [8.89941714023592e-04, -4.25818033750240e-03, 1.99225160493104e-02, -6.46588399513875e-02, 1.32772390609268e-01, -1.78508501143865e-01, 1.59999344843642e-01, -9.51030239931668e-02, 3.61804402620956e-02, -8.04009783602125e-03, 8.04009783602125e-04, 0.0, 0.0],
        [-7.63397185185940e-05, 6.56275990492236e-04, -5.83721252683346e-03, 2.90439093207059e-02, -8.47041434795334e-02, 1.57429980520300e-01, -1.95480070025582e-01, 1.65419875422483e-01, -9.49318221478776e-02, 3.55994333054541e-02, -7.91098517898980e-03, 7.91098517898980e-04, 0.0],
        [-8.51254383837181e-06, -1.77491211003807e-06, 7.32410586432031e-04, -7.40542710012764e-03, 3.44704736857857e-02, -9.39121496781828e-02, 1.66014456295034e-01, -1.99926171069111e-01, 1.66727824560197e-01, -9.52730426058266e-02, 3.57273909771850e-02, -7.93942021715222e-03, 7.93942021715222e-04]
    ]);

    #[rustfmt::skip]
    const DISS_DIAG: RowVector<Float, 11> = RowVector::new([[
        7.93650793650794e-04, -7.93650793650794e-03, 3.57142857142857e-02, -9.52380952380952e-02, 1.66666666666667e-01, -2.00000000000000e-01, 1.66666666666667e-01, -9.52380952380952e-02, 3.57142857142857e-02, -7.93650793650794e-03, 7.93650793650794e-04
    ]]);
    const DISS: BlockMatrix<Float, 8, 13, 11> = BlockMatrix::new(
        Self::DISS_BLOCK,
        Self::DISS_DIAG,
        Self::DISS_BLOCK.flip_lr().flip_ud(),
    );
}

impl SbpOperator1d for Upwind9h2 {
    fn diff(&self, prev: ArrayView1<Float>, fut: ArrayViewMut1<Float>) {
        super::diff_op_1d(&Self::DIFF, super::OperatorType::H2, prev, fut);
    }

    fn h(&self) -> &'static [Float] {
        &Self::H.start
    }
    fn is_h2(&self) -> bool {
        true
    }

    #[cfg(feature = "sparse")]
    fn diff_matrix(&self, n: usize) -> sprs::CsMat<Float> {
        super::sparse_from_block(&Self::DIFF, super::OperatorType::H2, n)
    }
    #[cfg(feature = "sparse")]
    fn h_matrix(&self, n: usize) -> sprs::CsMat<Float> {
        super::h_matrix(&Self::H, n, self.is_h2())
    }

    fn upwind(&self) -> Option<&dyn UpwindOperator1d> {
        Some(&Self)
    }
}

impl SbpOperator2d for Upwind9h2 {
    fn diffxi(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>) {
        assert_eq!(prev.shape(), fut.shape());

        super::diff_op_2d(&Self::DIFF, OperatorType::H2, prev, fut);
    }
    fn op_xi(&self) -> &dyn SbpOperator1d {
        &Self
    }
    fn upwind(&self) -> Option<Box<dyn UpwindOperator2d>> {
        Some(Box::new(Self))
    }
}

#[test]
fn upwind9h2_test() {
    let nx = 30;

    let x = crate::utils::h2linspace(0.0, 1.0, nx);

    let mut res = ndarray::Array1::zeros(nx);

    Upwind9h2.diff(x.view(), res.view_mut());
    let ans = &x * 0.0 + 1.0;
    approx::assert_abs_diff_eq!(&res, &ans, epsilon = 1e-4);

    res.fill(0.0);
    let y = &x * &x / 2.0;
    Upwind9h2.diff(y.view(), res.view_mut());
    let ans = &x;
    approx::assert_abs_diff_eq!(&res, &ans, epsilon = 1e-4);

    res.fill(0.0);
    let y = &x * &x * &x / 3.0;
    Upwind9h2.diff(y.view(), res.view_mut());
    let ans = &x * &x;
    approx::assert_abs_diff_eq!(&res, &ans, epsilon = 1e-2);
}

impl UpwindOperator1d for Upwind9h2 {
    fn diss(&self, prev: ArrayView1<Float>, fut: ArrayViewMut1<Float>) {
        super::diff_op_1d(&Self::DISS, super::OperatorType::H2, prev, fut);
    }
    fn as_sbp(&self) -> &dyn SbpOperator1d {
        self
    }

    #[cfg(feature = "sparse")]
    fn diss_matrix(&self, n: usize) -> sprs::CsMat<Float> {
        super::sparse_from_block(&Self::DISS, OperatorType::H2, n)
    }
}

impl UpwindOperator2d for Upwind9h2 {
    fn dissxi(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>) {
        assert_eq!(prev.shape(), fut.shape());

        super::diff_op_2d(&Self::DISS, OperatorType::H2, prev, fut);
    }
    fn op_xi(&self) -> &dyn UpwindOperator1d {
        &Self
    }
}
