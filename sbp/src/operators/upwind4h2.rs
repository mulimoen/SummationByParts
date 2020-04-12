use super::{SbpOperator, UpwindOperator};
use crate::Float;
use ndarray::{ArrayView1, ArrayViewMut1};

#[derive(Debug)]
pub struct Upwind4h2 {}

impl Upwind4h2 {
    #[rustfmt::skip]
    const HBLOCK: &'static [Float] = &[
        0.91e2/0.720e3, 0.325e3/0.384e3, 0.595e3/0.576e3, 0.1909e4/0.1920e4,
    ];
    #[rustfmt::skip]
    const DIAG: &'static [Float] = &[
        -1.43229166666667e-02, 1.40625000000000e-01, -7.38281250000000e-01, 0.00000000000000e+00, 7.38281250000000e-01, -1.40625000000000e-01, 1.43229166666667e-02
    ];
    #[rustfmt::skip]
    const BLOCK: &'static [[Float; 7]] = &[
        [-3.95604395604396e+00, 5.41758241758242e+00, -1.94505494505495e+00, 4.83516483516484e-01, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],
        [-8.09025641025641e-01, 0.00000000000000e+00, 1.03948717948718e+00, -2.47384615384615e-01, 1.69230769230769e-02, 0.00000000000000e+00, 0.00000000000000e+00],
        [2.37983193277311e-01, -8.51680672268908e-01, 0.00000000000000e+00, 7.35966386554622e-01, -1.36134453781513e-01, 1.38655462184874e-02, 0.00000000000000e+00],
        [-6.14632442814737e-02, 2.10581456259822e-01, -7.64623712240265e-01, 0.00000000000000e+00, 7.42535358826611e-01, -1.41435306443164e-01, 1.44054478784704e-02],
    ];

    #[rustfmt::skip]
    const DISS_BLOCK: &'static [[Float; 7]; 4] = &[
        [-2.76989342315976e-01, 5.19355016842454e-01, -3.46236677894969e-01, 1.03871003368491e-01, 0.00000000000000e+00, 0.00000000000000e+00, 0.00000000000000e+00],
        [7.75570158484731e-02, -1.62342481638964e-01, 1.47715500579822e-01, -7.98531117124082e-02, 1.69230769230769e-02, 0.00000000000000e+00, 0.00000000000000e+00],
        [-4.23630758836198e-02, 1.21027405937249e-01, -1.91609307039399e-01, 1.82272708078206e-01, -8.31932773109244e-02, 1.38655462184874e-02, 0.00000000000000e+00],
        [1.32037874021759e-02, -6.79734450144910e-02, 1.89370108794365e-01, -2.78654929966754e-01, 2.16081718177056e-01, -8.64326872708224e-02, 1.44054478784704e-02],
    ];

    #[rustfmt::skip]
    const DISS_DIAG: &'static [Float; 7] = &[
        1.43229166666667e-02, -8.59375000000000e-02, 2.14843750000000e-01, -2.86458333333333e-01, 2.14843750000000e-01, -8.59375000000000e-02, 1.43229166666667e-02,
    ];
}

impl SbpOperator for Upwind4h2 {
    fn diff1d(prev: ArrayView1<Float>, fut: ArrayViewMut1<Float>) {
        super::diff_op_1d(
            ndarray::arr2(Self::BLOCK).view(),
            ndarray::arr1(Self::DIAG).view(),
            false,
            true,
            prev,
            fut,
        )
    }

    fn h() -> &'static [Float] {
        Self::HBLOCK
    }
    fn is_h2() -> bool {
        true
    }
}

#[test]
fn upwind4h2_test() {
    let nx = 20;

    let x = crate::utils::h2linspace(0.0, 1.0, nx);

    let mut res = ndarray::Array1::zeros(nx);

    Upwind4h2::diff1d(x.view(), res.view_mut());
    let ans = &x * 0.0 + 1.0;
    approx::assert_abs_diff_eq!(&res, &ans, epsilon = 1e-4);

    res.fill(0.0);
    let y = &x * &x / 2.0;
    Upwind4h2::diff1d(y.view(), res.view_mut());
    let ans = &x;
    approx::assert_abs_diff_eq!(&res, &ans, epsilon = 1e-4);

    res.fill(0.0);
    let y = &x * &x * &x / 3.0;
    Upwind4h2::diff1d(y.view(), res.view_mut());
    let ans = &x * &x;
    approx::assert_abs_diff_eq!(&res, &ans, epsilon = 1e-2);
}

impl UpwindOperator for Upwind4h2 {
    fn diss1d(prev: ArrayView1<Float>, fut: ArrayViewMut1<Float>) {
        super::diff_op_1d(
            ndarray::arr2(Self::DISS_BLOCK).view(),
            ndarray::arr1(Self::DISS_DIAG).view(),
            true,
            true,
            prev,
            fut,
        )
    }
}
