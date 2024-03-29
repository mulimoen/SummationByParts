use super::{
    BlockMatrix, DiagonalMatrix, Matrix, OperatorType, RowVector, SbpOperator1d, SbpOperator2d,
    UpwindOperator1d, UpwindOperator2d,
};
use crate::Float;
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};

#[derive(Debug, Copy, Clone)]
pub struct Upwind9;

impl Upwind9 {
    #[rustfmt::skip]
    const H: DiagonalMatrix<8> = DiagonalMatrix::new([
        1070017.0/3628800.0, 5537111.0/3628800.0, 103613.0/403200.0, 261115.0/145152.0, 298951.0/725760.0, 515677.0/403200.0, 3349879.0/3628800.0, 3662753.0/3628800.0
    ]);
    #[rustfmt::skip]
    const DIFF_DIAG: RowVector<Float, 11> = RowVector::new([[
        -1.0/1260.0, 5.0/504.0, -5.0/84.0, 5.0/21.0, -5.0/6.0, 0.0, 5.0/6.0, -5.0/21.0, 5.0/84.0, -5.0/504.0, 1.0/1260.0,
    ]]);
    #[rustfmt::skip]
    const DIFF_BLOCK: Matrix<Float, 8, 13> = Matrix::new([
        [-1.69567399396458e+00, 2.29023358159400e+00, -2.16473500425698e-01, -5.05879766354449e-01, -1.01161106778154e-01, 2.59147072064383e-01, 1.93922119400659e-02, -4.95844980755642e-02, 0.0, 0.0, 0.0, 0.0, 0.0],
        [-4.42575354959737e-01, 0.00000000000000e+00, 1.91582959381899e-01, 2.82222626681305e-01, 1.12083989713257e-01, -1.51334868892111e-01, -2.23600502721044e-02, 3.03806983474913e-02, 0.0, 0.0, 0.0, 0.0,0.0],
        [2.48392603571843e-01, -1.13758367065272e+00, 0.00000000000000e+00, 1.95334726810969e+00, -1.58879011773212e+00, 3.93797129320378e-01, 2.52140821030291e-01, -1.21304033647356e-01, 0.0, 0.0, 0.0, 0.0, 0.0],
        [8.29213105268236e-02, -2.39388470313226e-01, -2.79038666398460e-01, 0.00000000000000e+00, 3.43018053395471e-01, 1.10370852514749e-01, 1.72029988649808e-03, -2.00445645303789e-02, 4.41184918522490e-04, 0.0, 0.0, 0.0, 0.0],
        [7.24159504343116e-02, -4.15199475743626e-01, 9.91181694804303e-01, -1.49802407438608e+00, 0.00000000000000e+00, 1.30188867830442e+00, -6.03535071819214e-01, 1.73429775718218e-01, -2.40842144699299e-02, 1.92673715759439e-03, 0.0, 0.0, 0.0],
        [-5.97470838462221e-02, 1.80551858630298e-01, -7.91241454636765e-02, -1.55240829877729e-01, -4.19298775383066e-01, 0.00000000000000e+00, 6.42287612546289e-01, -1.48833147569152e-01, 4.65407609802260e-02, -7.75679349670433e-03, 6.20543479736347e-04, 0.0, 0.0],
        [-6.19425252179959e-03, 3.69595678895333e-02, -7.01892820620398e-02, -3.35233082197107e-03, 2.69304373763091e-01, -8.89857974743355e-01, 0.00000000000000e+00, 8.66656645522330e-01, -2.57919763669076e-01, 6.44799409172690e-02, -1.07466568195448e-02, 8.59732545563586e-04, 0.0],
        [1.44853491014330e-02, -4.59275574977554e-02, 3.08833474560615e-02, 3.57240610228828e-02, -7.07760049349999e-02, 1.88587240076292e-01, -7.92626447113877e-01, 0.00000000000000e+00, 8.25608497215073e-01, -2.35888142061449e-01, 5.89720355153623e-02, -9.82867258589373e-03, 7.86293806871498e-04],
    ]);
    const DIFF: BlockMatrix<Float, 8, 13, 11> = BlockMatrix::new(
        Self::DIFF_BLOCK,
        Self::DIFF_DIAG,
        Self::DIFF_BLOCK.flip_lr().flip_ud().flip_sign(),
    );

    #[rustfmt::skip]
    const DISS_BLOCK: Matrix<Float, 8, 13> = Matrix::new([
        [-3.99020778658945e-04, 2.05394169917502e-03, -4.24493243399805e-03, 4.38126393542801e-03, -2.18883813216888e-03, 2.98565988131608e-04, 1.38484104084115e-04, -3.94643819928825e-05, 0.0, 0.0, 0.0, 0.0, 0.0],
        [3.96913216138553e-04, -2.28230530115522e-03, 5.43069719436758e-03, -6.81086901935894e-03, 4.69064759201504e-03, -1.61429862514855e-03, 1.62083873811316e-04, 2.71310693302277e-05, 0.0, 0.0, 0.0, 0.0, 0.0 ],
        [-4.87084939816571e-03, 3.22464611075207e-02, -9.06094757860846e-02, 1.39830191253413e-01, -1.27675500367419e-01, 6.87310321912961e-02, -2.00917702215270e-02, 2.43991122096699e-03, 0.0, 0.0, 0.0, 0.0, 0.0],
        [7.18155125886276e-04, -5.77715378536685e-03, 1.99749582302141e-02, -3.87940986951101e-02, 4.62756436981388e-02, -3.46770570075288e-02, 1.59058082995305e-02, -4.06744078428648e-03, 4.41184918522490e-04, 0.0, 0.0, 0.0, 0.0],
        [-1.56687484682703e-03, 1.73758484693946e-02, -7.96515646886111e-02, 2.02094401829054e-01, -3.16098733124618e-01, 3.17999240131250e-01, -2.06522928911140e-01, 8.37112455598470e-02, -1.92673715759439e-02, 1.92673715759439e-03, 0.0, 0.0, 0.0],
        [6.88352254356072e-05, -1.92595810396278e-03, 1.38098624496279e-02, -4.87746083763075e-02, 1.02417890394006e-01, -1.38292226669620e-01, 1.23829022892659e-01, -7.34723830823462e-02, 2.79244565881356e-02, -6.20543479736347e-03, 6.20543479736347e-04, 0.0, 0.0],
        [4.42345367100640e-05, 2.67913080025652e-04, -5.59301314813691e-03, 3.09954862110834e-02, -9.21529346596015e-02, 1.71559035817103e-01, -2.12738289547735e-01, 1.79835101537893e-01, -1.03167905467630e-01, 3.86879645503614e-02, -8.59732545563586e-03, 8.59732545563586e-04, 0.0],
        [-1.15289127131636e-05, 4.10149803795578e-05, 6.21188131452618e-04, -7.24912245235322e-03, 3.41622279353287e-02, -9.30972311856124e-02, 1.64473506705108e-01, -1.98013074867399e-01, 1.65121699443015e-01, -9.43552568245798e-02, 3.53832213092174e-02, -7.86293806871498e-03, 7.86293806871498e-04]
    ]);

    #[rustfmt::skip]
    const DISS_DIAG: RowVector<Float, 11> = RowVector::new([[
        1.0/1260.0, -1.0/126.0, 1.0/28.0, -2.0/21.0, 1.0/6.0, -1.0/5.0, 1.0/6.0, -2.0/21.0, 1.0/28.0, -1.0/126.0, 1.0/1260.0,
    ]]);
    const DISS: BlockMatrix<Float, 8, 13, 11> = BlockMatrix::new(
        Self::DISS_BLOCK,
        Self::DISS_DIAG,
        Self::DISS_BLOCK.flip_lr().flip_ud(),
    );
}

impl SbpOperator1d for Upwind9 {
    fn diff(&self, prev: ArrayView1<Float>, fut: ArrayViewMut1<Float>) {
        super::diff_op_1d(&Self::DIFF, super::OperatorType::Normal, prev, fut);
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

impl SbpOperator2d for Upwind9 {
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

impl UpwindOperator1d for Upwind9 {
    fn diss(&self, prev: ArrayView1<Float>, fut: ArrayViewMut1<Float>) {
        super::diff_op_1d(&Self::DISS, super::OperatorType::Normal, prev, fut);
    }

    fn as_sbp(&self) -> &dyn SbpOperator1d {
        self
    }

    #[cfg(feature = "sparse")]
    fn diss_matrix(&self, n: usize) -> sprs::CsMat<Float> {
        super::sparse_from_block(&Self::DISS, super::OperatorType::Normal, n)
    }
}

impl UpwindOperator2d for Upwind9 {
    fn dissxi(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>) {
        assert_eq!(prev.shape(), fut.shape());
        super::diff_op_2d(&Self::DISS, OperatorType::Normal, prev, fut);
    }
    fn op_xi(&self) -> &dyn UpwindOperator1d {
        &Self
    }
}

#[test]
fn test_upwind9() {
    use super::testing::*;
    let nx = 32;
    let ny = 16;

    // Order one polynomial
    check_operator_on(
        Upwind9,
        (ny, nx),
        |x, y| x + 2.0 * y,
        |_x, _y| 1.0,
        |_x, _y| 2.0,
        1e-4,
    );

    // Order two polynomial
    check_operator_on(
        Upwind9,
        (ny, nx),
        |x, y| x * x + 0.5 * y * y,
        |x, _y| 2.0 * x,
        |_x, y| y,
        1e-4,
    );
    check_operator_on(Upwind9, (ny, nx), |x, y| x * y, |_x, y| y, |x, _y| x, 1e-4);

    // Order three polynomials
    check_operator_on(
        Upwind9,
        (ny, nx),
        |x, y| x * x * x + y * y * y / 6.0,
        |x, _y| 3.0 * x * x,
        |_x, y| y * y / 2.0,
        1e-4,
    );
    check_operator_on(
        Upwind9,
        (ny, nx),
        |x, y| x * x * y + x * y * y / 2.0,
        |x, y| 2.0 * x * y + y * y / 2.0,
        |x, y| x * x + x * y,
        1e-4,
    );
    check_operator_on(
        Upwind9,
        (ny, nx),
        |x, y| x.powi(3) + 2.0 * x.powi(2) * y + 3.0 * x * y.powi(2) + 4.0 * y.powi(3),
        |x, y| 3.0 * x.powi(2) + 4.0 * x * y + 3.0 * y.powi(2),
        |x, y| 2.0 * x.powi(2) + 6.0 * x * y + 12.0 * y.powi(2),
        1e-4,
    );

    // Order four polynomials
    check_operator_on(
        Upwind9,
        (ny, nx),
        |x, y| x.powi(4) + x.powi(3) * y + x.powi(2) * y.powi(2) + x * y.powi(3) + y.powi(4),
        |x, y| 4.0 * x.powi(3) + 3.0 * x.powi(2) * y + 2.0 * x * y.powi(2) + y.powi(3),
        |x, y| x.powi(3) + 2.0 * x.powi(2) * y + 3.0 * x * y.powi(2) + 4.0 * y.powi(3),
        1e-4,
    );
}
