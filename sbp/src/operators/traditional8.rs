use super::{diff_op_col, diff_op_row, Matrix, RowVector, SbpOperator1d, SbpOperator2d};
use crate::Float;
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};

#[derive(Debug, Copy, Clone)]
pub struct SBP8;

impl SBP8 {
    #[rustfmt::skip]
    const HBLOCK: &'static [Float] = &[
        2.94890676177879e-01, 1.52572062389771e+00, 2.57452876984127e-01, 1.79811370149912e+00, 4.12708057760141e-01, 1.27848462301587e+00, 9.23295579805997e-01, 1.00933386085916e+00
    ];
    #[rustfmt::skip]
    const DIAG: &'static [Float] = &[
        3.57142857142857e-03, -3.80952380952381e-02, 2.00000000000000e-01, -8.00000000000000e-01, -0.00000000000000e+00, 8.00000000000000e-01, -2.00000000000000e-01, 3.80952380952381e-02, -3.57142857142857e-03
    ];
    #[rustfmt::skip]
    const DIAG_MATRIX: RowVector<Float, 9> = Matrix::new([[
        3.57142857142857e-03, -3.80952380952381e-02, 2.00000000000000e-01, -8.00000000000000e-01, -0.00000000000000e+00, 8.00000000000000e-01, -2.00000000000000e-01, 3.80952380952381e-02, -3.57142857142857e-03
    ]]);
    #[rustfmt::skip]
    const BLOCK: &'static [&'static [Float]] = &[
        &[-1.69554360443190e+00, 2.24741246341404e+00, -3.38931922601500e-02, -7.81028168126749e-01, 2.54881486107905e-02, 3.43865227388873e-01, -8.62858162633335e-02, -2.00150583315761e-02],
        &[-4.34378988266985e-01, 0.00000000000000e+00, 9.18511925072956e-02, 4.94008626807984e-01, -2.46151762937235e-02, -1.86759403432935e-01, 5.27267838475813e-02, 7.16696483080115e-03],
        &[3.88218088704253e-02, -5.44329744454984e-01, 0.00000000000000e+00, 3.89516189693211e-01, 1.36433486528546e-01, 1.03290582800845e-01, -1.79720579323281e-01, 5.59882558852296e-02],
        &[1.28088632226564e-01, -4.19172130036008e-01, -5.57707021445779e-02, 0.00000000000000e+00, 1.24714160903055e-01, 2.81285212519100e-01, -3.94470423942641e-02, -1.96981310738430e-02],
        &[-1.82119472519009e-02, 9.09986646154550e-02, -8.51090570277506e-02, -5.43362886365301e-01, 0.00000000000000e+00, 6.37392455438558e-01, -1.02950081118829e-01, 2.98964956216039e-02, -8.65364391190110e-03],
        &[-7.93147196245203e-02, 2.22875323171502e-01, -2.07999824391436e-02, -3.95611167748401e-01, -2.05756876210586e-01, 0.00000000000000e+00, 5.45876519966127e-01, -9.42727926638298e-02, 2.97971812952850e-02, -2.79348574643297e-03],
        &[2.75587615266177e-02, -8.71295642560637e-02, 5.01135077563584e-02, 7.68229253600969e-02, 4.60181213406519e-02, -7.55873581663580e-01, 0.00000000000000e+00, 8.21713248844682e-01, -2.16615355227872e-01, 4.12600676624518e-02, -3.86813134335486e-03],
        &[5.84767272160451e-03, -1.08336661209337e-02, -1.42810403117803e-02, 3.50919361287023e-02, -1.22244235731112e-02, 1.19411743193552e-01, -7.51668243727123e-01, 0.00000000000000e+00, 7.92601963555477e-01, -1.98150490888869e-01, 3.77429506454989e-02, -3.53840162301552e-03],
    ];
    #[rustfmt::skip]
    const BLOCK_MATRIX: Matrix<Float, 8, 12> = Matrix::new([
        [-1.69554360443190e+00, 2.24741246341404e+00, -3.38931922601500e-02, -7.81028168126749e-01, 2.54881486107905e-02, 3.43865227388873e-01, -8.62858162633335e-02, -2.00150583315761e-02, 0.0, 0.0, 0.0, 0.0],
        [-4.34378988266985e-01, 0.00000000000000e+00, 9.18511925072956e-02, 4.94008626807984e-01, -2.46151762937235e-02, -1.86759403432935e-01, 5.27267838475813e-02, 7.16696483080115e-03, 0.0, 0.0, 0.0, 0.0],
        [3.88218088704253e-02, -5.44329744454984e-01, 0.00000000000000e+00, 3.89516189693211e-01, 1.36433486528546e-01, 1.03290582800845e-01, -1.79720579323281e-01, 5.59882558852296e-02, 0.0, 0.0, 0.0, 0.0],
        [1.28088632226564e-01, -4.19172130036008e-01, -5.57707021445779e-02, 0.00000000000000e+00, 1.24714160903055e-01, 2.81285212519100e-01, -3.94470423942641e-02, -1.96981310738430e-02, 0.0, 0.0, 0.0, 0.0],
        [-1.82119472519009e-02, 9.09986646154550e-02, -8.51090570277506e-02, -5.43362886365301e-01, 0.00000000000000e+00, 6.37392455438558e-01, -1.02950081118829e-01, 2.98964956216039e-02, -8.65364391190110e-03, 0.0, 0.0, 0.0],
        [-7.93147196245203e-02, 2.22875323171502e-01, -2.07999824391436e-02, -3.95611167748401e-01, -2.05756876210586e-01, 0.00000000000000e+00, 5.45876519966127e-01, -9.42727926638298e-02, 2.97971812952850e-02, -2.79348574643297e-03, 0.0, 0.0],
        [2.75587615266177e-02, -8.71295642560637e-02, 5.01135077563584e-02, 7.68229253600969e-02, 4.60181213406519e-02, -7.55873581663580e-01, 0.00000000000000e+00, 8.21713248844682e-01, -2.16615355227872e-01, 4.12600676624518e-02, -3.86813134335486e-03, 0.0],
        [5.84767272160451e-03, -1.08336661209337e-02, -1.42810403117803e-02, 3.50919361287023e-02, -1.22244235731112e-02, 1.19411743193552e-01, -7.51668243727123e-01, 0.00000000000000e+00, 7.92601963555477e-01, -1.98150490888869e-01, 3.77429506454989e-02, -3.53840162301552e-03],
    ]);
}

impl SbpOperator1d for SBP8 {
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
}

fn diff_op_row_local(prev: ndarray::ArrayView2<Float>, mut fut: ndarray::ArrayViewMut2<Float>) {
    let mut flipmatrix = SBP8::BLOCK_MATRIX;
    flipmatrix *= &-1.0;

    for (p, mut f) in prev
        .axis_iter(ndarray::Axis(0))
        .zip(fut.axis_iter_mut(ndarray::Axis(0)))
    {
        super::diff_op_1d_slice_matrix(
            &SBP8::BLOCK_MATRIX,
            &flipmatrix,
            &SBP8::DIAG_MATRIX,
            super::OperatorType::Normal,
            p.as_slice().unwrap(),
            f.as_slice_mut().unwrap(),
        )
    }
}

impl SbpOperator2d for SBP8 {
    fn diffxi(&self, prev: ArrayView2<Float>, mut fut: ArrayViewMut2<Float>) {
        assert_eq!(prev.shape(), fut.shape());
        assert!(prev.shape()[1] >= 2 * SBP8::BLOCK.len());

        let symmetry = super::Symmetry::AntiSymmetric;
        let optype = super::OperatorType::Normal;

        match (prev.strides(), fut.strides()) {
            ([_, 1], [_, 1]) => {
                //diff_op_row(SBP8::BLOCK, SBP8::DIAG, symmetry, optype)(prev, fut);
                diff_op_row_local(prev, fut);
            }
            ([1, _], [1, _]) => {
                diff_op_col(SBP8::BLOCK, SBP8::DIAG, symmetry, optype)(prev, fut);
            }
            ([_, _], [_, _]) => {
                // Fallback, work row by row
                for (r0, r1) in prev.outer_iter().zip(fut.outer_iter_mut()) {
                    SBP8.diff(r0, r1);
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

#[test]
fn test_trad8() {
    use super::testing::*;
    let nx = 32;
    let ny = 16;

    // Order one polynomial
    check_operator_on(
        SBP8,
        (ny, nx),
        |x, y| x + 2.0 * y,
        |_x, _y| 1.0,
        |_x, _y| 2.0,
        1e-4,
    );

    // Order two polynomial
    check_operator_on(
        SBP8,
        (ny, nx),
        |x, y| x * x + 0.5 * y * y,
        |x, _y| 2.0 * x,
        |_x, y| y,
        1e-4,
    );
    check_operator_on(SBP8, (ny, nx), |x, y| x * y, |_x, y| y, |x, _y| x, 1e-4);

    // Order three polynomials
    check_operator_on(
        SBP8,
        (ny, nx),
        |x, y| x * x * x + y * y * y / 6.0,
        |x, _y| 3.0 * x * x,
        |_x, y| y * y / 2.0,
        1e-4,
    );
    check_operator_on(
        SBP8,
        (ny, nx),
        |x, y| x * x * y + x * y * y / 2.0,
        |x, y| 2.0 * x * y + y * y / 2.0,
        |x, y| x * x + x * y,
        1e-4,
    );
    check_operator_on(
        SBP8,
        (ny, nx),
        |x, y| x.powi(3) + 2.0 * x.powi(2) * y + 3.0 * x * y.powi(2) + 4.0 * y.powi(3),
        |x, y| 3.0 * x.powi(2) + 4.0 * x * y + 3.0 * y.powi(2),
        |x, y| 2.0 * x.powi(2) + 6.0 * x * y + 12.0 * y.powi(2),
        1e-4,
    );

    // Order four polynomials
    check_operator_on(
        SBP8,
        (ny, nx),
        |x, y| x.powi(4) + x.powi(3) * y + x.powi(2) * y.powi(2) + x * y.powi(3) + y.powi(4),
        |x, y| 4.0 * x.powi(3) + 3.0 * x.powi(2) * y + 2.0 * x * y.powi(2) + y.powi(3),
        |x, y| x.powi(3) + 2.0 * x.powi(2) * y + 3.0 * x * y.powi(2) + 4.0 * y.powi(3),
        1e-4,
    );
}
