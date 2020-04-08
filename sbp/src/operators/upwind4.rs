use super::{SbpOperator, UpwindOperator};
use crate::diff_op_1d;
use crate::Float;
use ndarray::{s, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis};

#[derive(Debug)]
pub struct Upwind4 {}

/// Simdtype used in diff_simd_col and diff_simd_row
#[cfg(feature = "f32")]
type SimdT = packed_simd::f32x8;
#[cfg(not(feature = "f32"))]
type SimdT = packed_simd::f64x8;

diff_op_1d!(diff_1d, Upwind4::BLOCK, Upwind4::DIAG, false);
diff_op_1d!(diss_1d, Upwind4::DISS_BLOCK, Upwind4::DISS_DIAG, true);

macro_rules! diff_simd_row_7_47 {
    ($name: ident, $BLOCK: expr, $DIAG: expr, $symmetric: expr) => {
        #[inline(never)]
        fn $name(prev: ArrayView2<Float>, mut fut: ArrayViewMut2<Float>) {
            assert_eq!(prev.shape(), fut.shape());
            assert!(prev.len_of(Axis(1)) >= 2 * $BLOCK.len());
            assert!(prev.len() >= SimdT::lanes());
            // The prev and fut array must have contiguous last dimension
            assert_eq!(prev.strides()[1], 1);
            assert_eq!(fut.strides()[1], 1);

            let nx = prev.len_of(Axis(1));
            let dx = 1.0 / (nx - 1) as Float;
            let idx = 1.0 / dx;

            for j in 0..prev.len_of(Axis(0)) {
                use std::slice;
                let prev = unsafe { slice::from_raw_parts(prev.uget((j, 0)) as *const Float, nx) };
                let fut =
                    unsafe { slice::from_raw_parts_mut(fut.uget_mut((j, 0)) as *mut Float, nx) };

                let first_elems = unsafe { SimdT::from_slice_unaligned_unchecked(prev) };
                let block = {
                    let bl = $BLOCK;
                    [
                        SimdT::new(
                            bl[0][0], bl[0][1], bl[0][2], bl[0][3], bl[0][4], bl[0][5], bl[0][6],
                            0.0,
                        ),
                        SimdT::new(
                            bl[1][0], bl[1][1], bl[1][2], bl[1][3], bl[1][4], bl[1][5], bl[1][6],
                            0.0,
                        ),
                        SimdT::new(
                            bl[2][0], bl[2][1], bl[2][2], bl[2][3], bl[2][4], bl[2][5], bl[2][6],
                            0.0,
                        ),
                        SimdT::new(
                            bl[3][0], bl[3][1], bl[3][2], bl[3][3], bl[3][4], bl[3][5], bl[3][6],
                            0.0,
                        ),
                    ]
                };
                fut[0] = idx * (block[0] * first_elems).sum();
                fut[1] = idx * (block[1] * first_elems).sum();
                fut[2] = idx * (block[2] * first_elems).sum();
                fut[3] = idx * (block[3] * first_elems).sum();

                let diag = {
                    let diag = $DIAG;
                    SimdT::new(
                        diag[0], diag[1], diag[2], diag[3], diag[4], diag[5], diag[6], 0.0,
                    )
                };
                for i in 4..nx - 4 {
                    unsafe {
                        let prev = SimdT::from_slice_unaligned_unchecked(&prev[i - 3..]);
                        *fut.get_unchecked_mut(i) = idx * (prev * diag).sum();
                    }
                }

                let last_elems = unsafe { SimdT::from_slice_unaligned_unchecked(&prev[nx - 8..]) }
                    .shuffle1_dyn([7, 6, 5, 4, 3, 2, 1, 0].into());
                if $symmetric {
                    fut[nx - 4] = idx * (block[3] * last_elems).sum();
                    fut[nx - 3] = idx * (block[2] * last_elems).sum();
                    fut[nx - 2] = idx * (block[1] * last_elems).sum();
                    fut[nx - 1] = idx * (block[0] * last_elems).sum();
                } else {
                    fut[nx - 4] = -idx * (block[3] * last_elems).sum();
                    fut[nx - 3] = -idx * (block[2] * last_elems).sum();
                    fut[nx - 2] = -idx * (block[1] * last_elems).sum();
                    fut[nx - 1] = -idx * (block[0] * last_elems).sum();
                }
            }
        }
    };
}

diff_simd_row_7_47!(diff_simd_row, Upwind4::BLOCK, Upwind4::DIAG, false);
diff_simd_row_7_47!(diss_simd_row, Upwind4::DISS_BLOCK, Upwind4::DISS_DIAG, true);

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
    const HBLOCK: &'static [Float] = &[
        49.0 / 144.0, 61.0 / 48.0, 41.0 / 48.0, 149.0 / 144.0
    ];
    #[rustfmt::skip]
    const DIAG: &'static [Float] = &[
        -1.0 / 24.0, 1.0 / 4.0, -7.0 / 8.0, 0.0, 7.0 / 8.0, -1.0 / 4.0, 1.0 / 24.0
    ];
    #[rustfmt::skip]
    const BLOCK: &'static [[Float; 7]] = &[
        [  -72.0 / 49.0, 187.0 / 98.0,   -20.0 / 49.0,   -3.0 / 98.0,           0.0,           0.0,         0.0],
        [-187.0 / 366.0,          0.0,   69.0 / 122.0, -16.0 / 183.0,    2.0 / 61.0,           0.0,         0.0],
        [  20.0 / 123.0, -69.0 / 82.0,            0.0, 227.0 / 246.0,  -12.0 / 41.0,    2.0 / 41.0,         0.0],
        [   3.0 / 298.0, 16.0 / 149.0, -227.0 / 298.0,           0.0, 126.0 / 149.0, -36.0 / 149.0, 6.0 / 149.0],
    ];

    #[rustfmt::skip]
    const DISS_BLOCK: &'static [[Float; 7]; 4] = &[
        [-3.0 / 49.0,    9.0 / 49.0,  -9.0 / 49.0,     3.0 / 49.0,          0.0,           0.0,         0.0],
        [ 3.0 / 61.0,  -11.0 / 61.0,  15.0 / 61.0,    -9.0 / 61.0,   2.0 / 61.0,           0.0,         0.0],
        [-3.0 / 41.0,   15.0 / 41.0, -29.0 / 41.0,    27.0 / 41.0, -12.0 / 41.0,    2.0 / 41.0,         0.0],
        [3.0 / 149.0, -27.0 / 149.0, 81.0 / 149.0, -117.0 / 149.0, 90.0 / 149.0, -36.0 / 149.0, 6.0 / 149.0],
    ];

    #[rustfmt::skip]
    const DISS_DIAG: &'static [Float; 7] = &[
        1.0 / 24.0, -1.0 / 4.0, 5.0 / 8.0, -5.0 / 6.0, 5.0 / 8.0, -1.0 / 4.0, 1.0 / 24.0
    ];
}

impl SbpOperator for Upwind4 {
    fn diff1d(prev: ArrayView1<Float>, fut: ArrayViewMut1<Float>) {
        diff_1d(prev, fut)
    }
    fn diffxi(prev: ArrayView2<Float>, mut fut: ArrayViewMut2<Float>) {
        assert_eq!(prev.shape(), fut.shape());
        assert!(prev.shape()[1] >= 2 * Self::BLOCK.len());

        match (prev.strides(), fut.strides()) {
            ([_, 1], [_, 1]) => {
                diff_simd_row(prev, fut);
            }
            ([1, _], [1, _]) if prev.len_of(Axis(0)) % SimdT::lanes() == 0 => {
                diff_simd_col(prev, fut);
            }
            ([_, _], [_, _]) => {
                // Fallback, work row by row
                for (r0, r1) in prev.outer_iter().zip(fut.outer_iter_mut()) {
                    Self::diff1d(r0, r1);
                }
            }
            _ => unreachable!("Should only be two elements in the strides vectors"),
        }
    }

    fn h() -> &'static [Float] {
        Self::HBLOCK
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
    Upwind4::diff1d(source.view(), res.view_mut());
    approx::assert_abs_diff_eq!(&res, &target, epsilon = 1e-4);
    {
        let source = source.to_owned().insert_axis(ndarray::Axis(0));
        let mut res = res.to_owned().insert_axis(ndarray::Axis(0));
        let target = target.to_owned().insert_axis(ndarray::Axis(0));
        res.fill(0.0);
        Upwind4::diffxi(source.view(), res.view_mut());
        approx::assert_abs_diff_eq!(&res, &target, epsilon = 1e-2);
    }

    {
        let source = Array2::from_shape_fn((nx, 8), |(i, _)| source[i]);
        let target = Array2::from_shape_fn((nx, 8), |(i, _)| target[i]);
        let mut res = Array2::zeros((nx, 8));
        res.fill(0.0);
        Upwind4::diffeta(source.view(), res.view_mut());
        approx::assert_abs_diff_eq!(&res.to_owned(), &target.to_owned(), epsilon = 1e-2);
    }

    for i in 0..nx {
        let x = i as Float * dx;
        source[i] = x * x;
        target[i] = 2.0 * x;
    }
    res.fill(0.0);
    Upwind4::diff1d(source.view(), res.view_mut());
    approx::assert_abs_diff_eq!(&res, &target, epsilon = 1e-4);
    {
        let source = source.to_owned().insert_axis(ndarray::Axis(0));
        let mut res = res.to_owned().insert_axis(ndarray::Axis(0));
        let target = target.to_owned().insert_axis(ndarray::Axis(0));
        res.fill(0.0);
        Upwind4::diffxi(source.view(), res.view_mut());
        approx::assert_abs_diff_eq!(&res, &target, epsilon = 1e-2);
    }

    {
        let source = Array2::from_shape_fn((nx, 8), |(i, _)| source[i]);
        let target = Array2::from_shape_fn((nx, 8), |(i, _)| target[i]);
        let mut res = Array2::zeros((nx, 8));
        res.fill(0.0);
        Upwind4::diffeta(source.view(), res.view_mut());
        approx::assert_abs_diff_eq!(&res.to_owned(), &target.to_owned(), epsilon = 1e-2);
    }

    for i in 0..nx {
        let x = i as Float * dx;
        source[i] = x * x * x;
        target[i] = 3.0 * x * x;
    }
    res.fill(0.0);
    Upwind4::diff1d(source.view(), res.view_mut());
    approx::assert_abs_diff_eq!(&res, &target, epsilon = 1e-2);

    {
        let source = source.to_owned().insert_axis(ndarray::Axis(0));
        let mut res = res.to_owned().insert_axis(ndarray::Axis(0));
        let target = target.to_owned().insert_axis(ndarray::Axis(0));
        res.fill(0.0);
        Upwind4::diffxi(source.view(), res.view_mut());
        approx::assert_abs_diff_eq!(&res, &target, epsilon = 1e-2);
    }

    {
        let source = Array2::from_shape_fn((nx, 8), |(i, _)| source[i]);
        let target = Array2::from_shape_fn((nx, 8), |(i, _)| target[i]);
        let mut res = Array2::zeros((nx, 8));
        res.fill(0.0);
        Upwind4::diffeta(source.view(), res.view_mut());
        approx::assert_abs_diff_eq!(&res.to_owned(), &target.to_owned(), epsilon = 1e-2);
    }
}

impl UpwindOperator for Upwind4 {
    fn diss1d(prev: ArrayView1<Float>, fut: ArrayViewMut1<Float>) {
        diss_1d(prev, fut)
    }
    fn dissxi(prev: ArrayView2<Float>, mut fut: ArrayViewMut2<Float>) {
        assert_eq!(prev.shape(), fut.shape());
        assert!(prev.shape()[1] >= 2 * Self::BLOCK.len());

        match (prev.strides(), fut.strides()) {
            ([_, 1], [_, 1]) => {
                diss_simd_row(prev, fut);
            }
            ([1, _], [1, _]) if prev.len_of(Axis(0)) % SimdT::lanes() == 0 => {
                diss_simd_col(prev, fut);
            }
            ([_, _], [_, _]) => {
                // Fallback, work row by row
                for (r0, r1) in prev.outer_iter().zip(fut.outer_iter_mut()) {
                    Self::diss1d(r0, r1);
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

    check_operator_on::<Upwind4, _, _, _>(
        (ny, nx),
        |x, y| x + 2.0 * y,
        |_, _| 1.0,
        |_, _| 2.0,
        1e-4,
    );
    check_operator_on::<Upwind4, _, _, _>(
        (ny, nx),
        |x, y| x * x + 2.0 * x * y + 3.0 * y * y,
        |x, y| 2.0 * x + 2.0 * y,
        |x, y| 2.0 * x + 6.0 * y,
        1e-3,
    );
    check_operator_on::<Upwind4, _, _, _>(
        (ny, nx),
        |x, y| x.powi(3) + 2.0 * x.powi(2) * y + 3.0 * x * y.powi(2) + 4.0 * y.powi(3),
        |x, y| 3.0 * x.powi(2) + 4.0 * x * y + 3.0 * y.powi(2),
        |x, y| 2.0 * x.powi(2) + 6.0 * x * y + 12.0 * y.powi(2),
        1e-1,
    );
    check_operator_on::<Upwind4, _, _, _>(
        (32, 32),
        |x, y| x.powi(3) + 2.0 * x.powi(2) * y + 3.0 * x * y.powi(2) + 4.0 * y.powi(3),
        |x, y| 3.0 * x.powi(2) + 4.0 * x * y + 3.0 * y.powi(2),
        |x, y| 2.0 * x.powi(2) + 6.0 * x * y + 12.0 * y.powi(2),
        1e-1,
    );
}
