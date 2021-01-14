use super::{
    diff_op_col, diff_op_row, SbpOperator1d, SbpOperator2d, UpwindOperator1d, UpwindOperator2d,
};
use crate::Float;
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis};

#[derive(Debug, Copy, Clone)]
pub struct Upwind4;

/// Simdtype used in diff_simd_col
#[cfg(feature = "f32")]
type SimdT = packed_simd::f32x8;
#[cfg(not(feature = "f32"))]
type SimdT = packed_simd::f64x8;

macro_rules! diff_simd_col_7_47 {
    ($name: ident, $BLOCK: expr, $DIAG: expr, $symmetric: expr) => {
        #[inline(never)]
        fn $name(prev: ArrayView2<Float>, mut fut: ArrayViewMut2<Float>) {
            use std::slice;
            assert_eq!(prev.shape(), fut.shape());
            assert_eq!(prev.stride_of(Axis(0)), 1);
            assert_eq!(fut.stride_of(Axis(0)), 1);
            let ny = prev.len_of(Axis(0));
            let nx = prev.len_of(Axis(1));
            assert!(nx >= 2 * $BLOCK.len());
            assert!(ny >= SimdT::lanes());
            assert!(ny % SimdT::lanes() == 0);

            let dx = 1.0 / (nx - 1) as Float;
            let idx = 1.0 / dx;

            for j in (0..ny).step_by(SimdT::lanes()) {
                let a = unsafe {
                    [
                        SimdT::from_slice_unaligned(slice::from_raw_parts(
                            prev.uget((j, 0)) as *const Float,
                            SimdT::lanes(),
                        )),
                        SimdT::from_slice_unaligned(slice::from_raw_parts(
                            prev.uget((j, 1)) as *const Float,
                            SimdT::lanes(),
                        )),
                        SimdT::from_slice_unaligned(slice::from_raw_parts(
                            prev.uget((j, 2)) as *const Float,
                            SimdT::lanes(),
                        )),
                        SimdT::from_slice_unaligned(slice::from_raw_parts(
                            prev.uget((j, 3)) as *const Float,
                            SimdT::lanes(),
                        )),
                        SimdT::from_slice_unaligned(slice::from_raw_parts(
                            prev.uget((j, 4)) as *const Float,
                            SimdT::lanes(),
                        )),
                        SimdT::from_slice_unaligned(slice::from_raw_parts(
                            prev.uget((j, 5)) as *const Float,
                            SimdT::lanes(),
                        )),
                        SimdT::from_slice_unaligned(slice::from_raw_parts(
                            prev.uget((j, 6)) as *const Float,
                            SimdT::lanes(),
                        )),
                    ]
                };

                for (i, bl) in $BLOCK.iter().enumerate() {
                    let b = idx
                        * (a[0] * bl[0]
                            + a[1] * bl[1]
                            + a[2] * bl[2]
                            + a[3] * bl[3]
                            + a[4] * bl[4]
                            + a[5] * bl[5]
                            + a[6] * bl[6]);
                    unsafe {
                        b.write_to_slice_unaligned(slice::from_raw_parts_mut(
                            fut.uget_mut((j, i)) as *mut Float,
                            SimdT::lanes(),
                        ));
                    }
                }

                let mut a = a;
                for i in $BLOCK.len()..nx - $BLOCK.len() {
                    // Push a onto circular buffer
                    a = [a[1], a[2], a[3], a[4], a[5], a[6], unsafe {
                        SimdT::from_slice_unaligned(slice::from_raw_parts(
                            prev.uget((j, i + 3)) as *const Float,
                            SimdT::lanes(),
                        ))
                    }];
                    let b = idx
                        * (a[0] * $DIAG[0]
                            + a[1] * $DIAG[1]
                            + a[2] * $DIAG[2]
                            + a[3] * $DIAG[3]
                            + a[4] * $DIAG[4]
                            + a[5] * $DIAG[5]
                            + a[6] * $DIAG[6]);
                    unsafe {
                        b.write_to_slice_unaligned(slice::from_raw_parts_mut(
                            fut.uget_mut((j, i)) as *mut Float,
                            SimdT::lanes(),
                        ));
                    }
                }

                let a = unsafe {
                    [
                        SimdT::from_slice_unaligned(slice::from_raw_parts(
                            prev.uget((j, nx - 1)) as *const Float,
                            SimdT::lanes(),
                        )),
                        SimdT::from_slice_unaligned(slice::from_raw_parts(
                            prev.uget((j, nx - 2)) as *const Float,
                            SimdT::lanes(),
                        )),
                        SimdT::from_slice_unaligned(slice::from_raw_parts(
                            prev.uget((j, nx - 3)) as *const Float,
                            SimdT::lanes(),
                        )),
                        SimdT::from_slice_unaligned(slice::from_raw_parts(
                            prev.uget((j, nx - 4)) as *const Float,
                            SimdT::lanes(),
                        )),
                        SimdT::from_slice_unaligned(slice::from_raw_parts(
                            prev.uget((j, nx - 5)) as *const Float,
                            SimdT::lanes(),
                        )),
                        SimdT::from_slice_unaligned(slice::from_raw_parts(
                            prev.uget((j, nx - 6)) as *const Float,
                            SimdT::lanes(),
                        )),
                        SimdT::from_slice_unaligned(slice::from_raw_parts(
                            prev.uget((j, nx - 7)) as *const Float,
                            SimdT::lanes(),
                        )),
                    ]
                };

                for (i, bl) in $BLOCK.iter().enumerate() {
                    let idx = if $symmetric { idx } else { -idx };
                    let b = idx
                        * (a[0] * bl[0]
                            + a[1] * bl[1]
                            + a[2] * bl[2]
                            + a[3] * bl[3]
                            + a[4] * bl[4]
                            + a[5] * bl[5]
                            + a[6] * bl[6]);
                    unsafe {
                        b.write_to_slice_unaligned(slice::from_raw_parts_mut(
                            fut.uget_mut((j, nx - 1 - i)) as *mut Float,
                            SimdT::lanes(),
                        ));
                    }
                }
            }
        }
    };
}

diff_simd_col_7_47!(diff_simd_col, Upwind4::BLOCK, Upwind4::DIAG, false);
diff_simd_col_7_47!(diss_simd_col, Upwind4::DISS_BLOCK, Upwind4::DISS_DIAG, true);

impl Upwind4 {
    #[rustfmt::skip]
    pub const HBLOCK: &'static [Float] = &[
        49.0 / 144.0, 61.0 / 48.0, 41.0 / 48.0, 149.0 / 144.0
    ];
    #[rustfmt::skip]
    const DIAG: &'static [Float] = &[
        -1.0 / 24.0, 1.0 / 4.0, -7.0 / 8.0, 0.0, 7.0 / 8.0, -1.0 / 4.0, 1.0 / 24.0
    ];
    #[rustfmt::skip]
    const BLOCK: &'static [&'static [Float]] = &[
        &[  -72.0 / 49.0, 187.0 / 98.0,   -20.0 / 49.0,   -3.0 / 98.0,           0.0,           0.0,         0.0],
        &[-187.0 / 366.0,          0.0,   69.0 / 122.0, -16.0 / 183.0,    2.0 / 61.0,           0.0,         0.0],
        &[  20.0 / 123.0, -69.0 / 82.0,            0.0, 227.0 / 246.0,  -12.0 / 41.0,    2.0 / 41.0,         0.0],
        &[   3.0 / 298.0, 16.0 / 149.0, -227.0 / 298.0,           0.0, 126.0 / 149.0, -36.0 / 149.0, 6.0 / 149.0],
    ];

    #[rustfmt::skip]
    const DISS_BLOCK: &'static [&'static [Float]] = &[
        &[-3.0 / 49.0,    9.0 / 49.0,  -9.0 / 49.0,     3.0 / 49.0,          0.0,           0.0,         0.0],
        &[ 3.0 / 61.0,  -11.0 / 61.0,  15.0 / 61.0,    -9.0 / 61.0,   2.0 / 61.0,           0.0,         0.0],
        &[-3.0 / 41.0,   15.0 / 41.0, -29.0 / 41.0,    27.0 / 41.0, -12.0 / 41.0,    2.0 / 41.0,         0.0],
        &[3.0 / 149.0, -27.0 / 149.0, 81.0 / 149.0, -117.0 / 149.0, 90.0 / 149.0, -36.0 / 149.0, 6.0 / 149.0],
    ];

    #[rustfmt::skip]
    const DISS_DIAG: &'static [Float; 7] = &[
        1.0 / 24.0, -1.0 / 4.0, 5.0 / 8.0, -5.0 / 6.0, 5.0 / 8.0, -1.0 / 4.0, 1.0 / 24.0
    ];
}

impl SbpOperator1d for Upwind4 {
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

    fn upwind(&self) -> Option<&dyn UpwindOperator1d> {
        Some(&Self)
    }
}

impl<SBP: SbpOperator1d> SbpOperator2d for (&SBP, &Upwind4) {
    fn diffxi(&self, prev: ArrayView2<Float>, mut fut: ArrayViewMut2<Float>) {
        assert_eq!(prev.shape(), fut.shape());
        assert!(prev.shape()[1] >= 2 * Upwind4::BLOCK.len());

        match (prev.strides(), fut.strides()) {
            ([_, 1], [_, 1]) => diff_op_row(
                Upwind4::BLOCK,
                Upwind4::DIAG,
                super::Symmetry::AntiSymmetric,
                super::OperatorType::Normal,
            )(prev, fut),
            ([1, _], [1, _]) if prev.len_of(Axis(0)) % SimdT::lanes() == 0 => {
                diff_simd_col(prev, fut)
            }
            ([1, _], [1, _]) => diff_op_col(
                Upwind4::BLOCK,
                Upwind4::DIAG,
                super::Symmetry::AntiSymmetric,
                super::OperatorType::Normal,
            )(prev, fut),
            ([_, _], [_, _]) => {
                // Fallback, work row by row
                for (r0, r1) in prev.outer_iter().zip(fut.outer_iter_mut()) {
                    Upwind4.diff(r0, r1);
                }
            }
            _ => unreachable!("Should only be two elements in the strides vectors"),
        }
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
        super::diff_op_1d(
            Self::DISS_BLOCK,
            Self::DISS_DIAG,
            super::Symmetry::Symmetric,
            super::OperatorType::Normal,
            prev,
            fut,
        )
    }

    fn as_sbp(&self) -> &dyn SbpOperator1d {
        self
    }

    #[cfg(feature = "sparse")]
    fn diss_matrix(&self, n: usize) -> sprs::CsMat<Float> {
        super::sparse_from_block(
            Self::DISS_BLOCK,
            Self::DISS_DIAG,
            super::Symmetry::Symmetric,
            super::OperatorType::Normal,
            n,
        )
    }
}

impl<UO: UpwindOperator1d> UpwindOperator2d for (&UO, &Upwind4) {
    fn dissxi(&self, prev: ArrayView2<Float>, mut fut: ArrayViewMut2<Float>) {
        assert_eq!(prev.shape(), fut.shape());
        assert!(prev.shape()[1] >= 2 * Upwind4::BLOCK.len());

        match (prev.strides(), fut.strides()) {
            ([_, 1], [_, 1]) => diff_op_row(
                Upwind4::DISS_BLOCK,
                Upwind4::DISS_DIAG,
                super::Symmetry::Symmetric,
                super::OperatorType::Normal,
            )(prev, fut),
            ([1, _], [1, _]) if prev.len_of(Axis(0)) % SimdT::lanes() == 0 => {
                diss_simd_col(prev, fut);
            }
            ([1, _], [1, _]) => diff_op_col(
                Upwind4::DISS_BLOCK,
                Upwind4::DISS_DIAG,
                super::Symmetry::Symmetric,
                super::OperatorType::Normal,
            )(prev, fut),
            ([_, _], [_, _]) => {
                // Fallback, work row by row
                for (r0, r1) in prev.outer_iter().zip(fut.outer_iter_mut()) {
                    Upwind4.diss(r0, r1);
                }
            }
            _ => unreachable!("Should only be two elements in the strides vectors"),
        }
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
