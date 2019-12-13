use super::{SbpOperator, UpwindOperator};
use crate::diff_op_1d;
use ndarray::{s, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Axis};

pub struct Upwind4 {}

/// Simdtype used in diff_simd_col
type SimdT = packed_simd::f32x8;

diff_op_1d!(Upwind4, diff_1d, Upwind4::BLOCK, Upwind4::DIAG, false);
diff_op_1d!(
    Upwind4,
    diss_1d,
    Upwind4::DISS_BLOCK,
    Upwind4::DISS_DIAG,
    true
);

impl Upwind4 {
    #[rustfmt::skip]
    const HBLOCK: &'static [f32] = &[
        49.0 / 144.0, 61.0 / 48.0, 41.0 / 48.0, 149.0 / 144.0
    ];
    #[rustfmt::skip]
    const DIAG: &'static [f32] = &[
        -1.0 / 24.0, 1.0 / 4.0, -7.0 / 8.0, 0.0, 7.0 / 8.0, -1.0 / 4.0, 1.0 / 24.0
    ];
    #[rustfmt::skip]
    const BLOCK: &'static [[f32; 7]] = &[
        [  -72.0 / 49.0, 187.0 / 98.0,   -20.0 / 49.0,   -3.0 / 98.0,           0.0,           0.0,         0.0],
        [-187.0 / 366.0,          0.0,   69.0 / 122.0, -16.0 / 183.0,    2.0 / 61.0,           0.0,         0.0],
        [  20.0 / 123.0, -69.0 / 82.0,            0.0, 227.0 / 246.0,  -12.0 / 41.0,    2.0 / 41.0,         0.0],
        [   3.0 / 298.0, 16.0 / 149.0, -227.0 / 298.0,           0.0, 126.0 / 149.0, -36.0 / 149.0, 6.0 / 149.0],
    ];

    #[rustfmt::skip]
    const DISS_BLOCK: &'static [[f32; 7]; 4] = &[
        [-3.0 / 49.0,    9.0 / 49.0,  -9.0 / 49.0,     3.0 / 49.0,          0.0,           0.0,         0.0],
        [ 3.0 / 61.0,  -11.0 / 61.0,  15.0 / 61.0,    -9.0 / 61.0,   2.0 / 61.0,           0.0,         0.0],
        [-3.0 / 41.0,   15.0 / 41.0, -29.0 / 41.0,    27.0 / 41.0, -12.0 / 41.0,    2.0 / 41.0,         0.0],
        [3.0 / 149.0, -27.0 / 149.0, 81.0 / 149.0, -117.0 / 149.0, 90.0 / 149.0, -36.0 / 149.0, 6.0 / 149.0],
    ];

    #[rustfmt::skip]
    const DISS_DIAG: &'static [f32; 7] = &[
        1.0 / 24.0, -1.0 / 4.0, 5.0 / 8.0, -5.0 / 6.0, 5.0 / 8.0, -1.0 / 4.0, 1.0 / 24.0
    ];

    #[inline(never)]
    fn diff_simd_row(prev: ArrayView2<f32>, mut fut: ArrayViewMut2<f32>) {
        use packed_simd::{f32x8, u32x8};
        assert_eq!(prev.shape(), fut.shape());
        assert!(prev.len_of(Axis(1)) >= 2 * Self::BLOCK.len());
        assert!(prev.len() >= f32x8::lanes());
        // The prev array must have contigous last dimension
        assert_eq!(prev.strides()[1], 1);

        let nx = prev.len_of(Axis(1));
        let dx = 1.0 / (nx - 1) as f32;
        let idx = 1.0 / dx;

        for j in 0..prev.len_of(Axis(0)) {
            //use std::slice;
            //let prev = unsafe { slice::from_raw_parts(prev.slice(s![j, ..]).as_ptr(), nx) };
            let prev = prev.slice(s![j, ..]);
            let prev = prev.as_slice_memory_order().unwrap();

            let first_elems = unsafe { f32x8::from_slice_unaligned_unchecked(prev) };
            let block = {
                let bl = Self::BLOCK;
                [
                    f32x8::new(
                        bl[0][0], bl[0][1], bl[0][2], bl[0][3], bl[0][4], bl[0][5], bl[0][6], 0.0,
                    ),
                    f32x8::new(
                        bl[1][0], bl[1][1], bl[1][2], bl[1][3], bl[1][4], bl[1][5], bl[1][6], 0.0,
                    ),
                    f32x8::new(
                        bl[2][0], bl[2][1], bl[2][2], bl[2][3], bl[2][4], bl[2][5], bl[2][6], 0.0,
                    ),
                    f32x8::new(
                        bl[3][0], bl[3][1], bl[3][2], bl[3][3], bl[3][4], bl[3][5], bl[3][6], 0.0,
                    ),
                ]
            };
            fut[(j, 0)] = idx * (block[0] * first_elems).sum();
            fut[(j, 1)] = idx * (block[1] * first_elems).sum();
            fut[(j, 2)] = idx * (block[2] * first_elems).sum();
            fut[(j, 3)] = idx * (block[3] * first_elems).sum();

            let diag = {
                let diag = Self::DIAG;
                f32x8::new(
                    diag[0], diag[1], diag[2], diag[3], diag[4], diag[5], diag[6], 0.0,
                )
            };
            for (f, p) in fut
                .slice_mut(s![j, ..])
                .iter_mut()
                .skip(block.len())
                .zip(
                    prev.windows(f32x8::lanes())
                        .map(f32x8::from_slice_unaligned)
                        .skip(1),
                )
                .take(nx - 2 * block.len())
            {
                *f = idx * (p * diag).sum();
            }

            let last_elems = unsafe { f32x8::from_slice_unaligned_unchecked(&prev[nx - 8..]) }
                .shuffle1_dyn(u32x8::new(7, 6, 5, 4, 3, 2, 1, 0));
            fut[(j, nx - 4)] = -idx * (block[3] * last_elems).sum();
            fut[(j, nx - 3)] = -idx * (block[2] * last_elems).sum();
            fut[(j, nx - 2)] = -idx * (block[1] * last_elems).sum();
            fut[(j, nx - 1)] = -idx * (block[0] * last_elems).sum();
        }
    }

    #[inline(never)]
    fn diff_simd_col(prev: ArrayView2<f32>, mut fut: ArrayViewMut2<f32>) {
        use std::slice;
        assert_eq!(prev.shape(), fut.shape());
        assert_eq!(prev.stride_of(Axis(0)), 1);
        assert_eq!(prev.stride_of(Axis(0)), 1);
        let ny = prev.len_of(Axis(0));
        let nx = prev.len_of(Axis(1));
        assert!(nx >= 2 * Self::BLOCK.len());
        assert!(ny >= SimdT::lanes());
        assert!(ny % SimdT::lanes() == 0);

        let dx = 1.0 / (nx - 1) as f32;
        let idx = 1.0 / dx;

        for j in (0..ny).step_by(SimdT::lanes()) {
            let a = unsafe {
                [
                    SimdT::from_slice_unaligned(slice::from_raw_parts(
                        prev.slice(s![j.., 0]).as_ptr(),
                        SimdT::lanes(),
                    )),
                    SimdT::from_slice_unaligned(slice::from_raw_parts(
                        prev.slice(s![j.., 1]).as_ptr(),
                        SimdT::lanes(),
                    )),
                    SimdT::from_slice_unaligned(slice::from_raw_parts(
                        prev.slice(s![j.., 2]).as_ptr(),
                        SimdT::lanes(),
                    )),
                    SimdT::from_slice_unaligned(slice::from_raw_parts(
                        prev.slice(s![j.., 3]).as_ptr(),
                        SimdT::lanes(),
                    )),
                    SimdT::from_slice_unaligned(slice::from_raw_parts(
                        prev.slice(s![j.., 4]).as_ptr(),
                        SimdT::lanes(),
                    )),
                    SimdT::from_slice_unaligned(slice::from_raw_parts(
                        prev.slice(s![j.., 5]).as_ptr(),
                        SimdT::lanes(),
                    )),
                    SimdT::from_slice_unaligned(slice::from_raw_parts(
                        prev.slice(s![j.., 6]).as_ptr(),
                        SimdT::lanes(),
                    )),
                ]
            };

            for (i, bl) in Self::BLOCK.iter().enumerate() {
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
                        fut.slice_mut(s![j.., i]).as_mut_ptr(),
                        SimdT::lanes(),
                    ));
                }
            }

            let mut a = a;
            for i in Self::BLOCK.len()..nx - Self::BLOCK.len() {
                // Push a onto circular buffer
                a = [a[1], a[2], a[3], a[4], a[5], a[6], unsafe {
                    SimdT::from_slice_unaligned(slice::from_raw_parts(
                        prev.slice(s![j.., i + 3]).as_ptr(),
                        SimdT::lanes(),
                    ))
                }];
                let b = idx
                    * (a[0] * Self::DIAG[0]
                        + a[1] * Self::DIAG[1]
                        + a[2] * Self::DIAG[2]
                        + a[3] * Self::DIAG[3]
                        + a[4] * Self::DIAG[4]
                        + a[5] * Self::DIAG[5]
                        + a[6] * Self::DIAG[6]);
                unsafe {
                    b.write_to_slice_unaligned(slice::from_raw_parts_mut(
                        fut.slice_mut(s![j.., i]).as_mut_ptr(),
                        SimdT::lanes(),
                    ));
                }
            }

            let a = unsafe {
                [
                    SimdT::from_slice_unaligned(slice::from_raw_parts(
                        prev.slice(s![j.., nx - 1]).as_ptr(),
                        SimdT::lanes(),
                    )),
                    SimdT::from_slice_unaligned(slice::from_raw_parts(
                        prev.slice(s![j.., nx - 2]).as_ptr(),
                        SimdT::lanes(),
                    )),
                    SimdT::from_slice_unaligned(slice::from_raw_parts(
                        prev.slice(s![j.., nx - 3]).as_ptr(),
                        SimdT::lanes(),
                    )),
                    SimdT::from_slice_unaligned(slice::from_raw_parts(
                        prev.slice(s![j.., nx - 4]).as_ptr(),
                        SimdT::lanes(),
                    )),
                    SimdT::from_slice_unaligned(slice::from_raw_parts(
                        prev.slice(s![j.., nx - 5]).as_ptr(),
                        SimdT::lanes(),
                    )),
                    SimdT::from_slice_unaligned(slice::from_raw_parts(
                        prev.slice(s![j.., nx - 6]).as_ptr(),
                        SimdT::lanes(),
                    )),
                    SimdT::from_slice_unaligned(slice::from_raw_parts(
                        prev.slice(s![j.., nx - 7]).as_ptr(),
                        SimdT::lanes(),
                    )),
                ]
            };

            for (i, bl) in Self::BLOCK.iter().enumerate() {
                let b = -idx
                    * (a[0] * bl[0]
                        + a[1] * bl[1]
                        + a[2] * bl[2]
                        + a[3] * bl[3]
                        + a[4] * bl[4]
                        + a[5] * bl[5]
                        + a[6] * bl[6]);
                unsafe {
                    b.write_to_slice_unaligned(slice::from_raw_parts_mut(
                        fut.slice_mut(s![j.., nx - 1 - i]).as_mut_ptr(),
                        SimdT::lanes(),
                    ));
                }
            }
        }
    }

    #[inline(never)]
    fn diss_simd(prev: &[f32], fut: &mut [f32]) {
        use packed_simd::{f32x8, u32x8};
        assert_eq!(prev.len(), fut.len());
        assert!(prev.len() >= 2 * Self::DISS_BLOCK.len());
        let nx = prev.len();
        let dx = 1.0 / (nx - 1) as f32;
        let idx = 1.0 / dx;

        let first_elems = unsafe { f32x8::from_slice_unaligned_unchecked(prev) };
        let block = [
            f32x8::new(
                Self::DISS_BLOCK[0][0],
                Self::DISS_BLOCK[0][1],
                Self::DISS_BLOCK[0][2],
                Self::DISS_BLOCK[0][3],
                Self::DISS_BLOCK[0][4],
                Self::DISS_BLOCK[0][5],
                Self::DISS_BLOCK[0][6],
                0.0,
            ),
            f32x8::new(
                Self::DISS_BLOCK[1][0],
                Self::DISS_BLOCK[1][1],
                Self::DISS_BLOCK[1][2],
                Self::DISS_BLOCK[1][3],
                Self::DISS_BLOCK[1][4],
                Self::DISS_BLOCK[1][5],
                Self::DISS_BLOCK[1][6],
                0.0,
            ),
            f32x8::new(
                Self::DISS_BLOCK[2][0],
                Self::DISS_BLOCK[2][1],
                Self::DISS_BLOCK[2][2],
                Self::DISS_BLOCK[2][3],
                Self::DISS_BLOCK[2][4],
                Self::DISS_BLOCK[2][5],
                Self::DISS_BLOCK[2][6],
                0.0,
            ),
            f32x8::new(
                Self::DISS_BLOCK[3][0],
                Self::DISS_BLOCK[3][1],
                Self::DISS_BLOCK[3][2],
                Self::DISS_BLOCK[3][3],
                Self::DISS_BLOCK[3][4],
                Self::DISS_BLOCK[3][5],
                Self::DISS_BLOCK[3][6],
                0.0,
            ),
        ];
        unsafe {
            *fut.get_unchecked_mut(0) = idx * (block[0] * first_elems).sum();
            *fut.get_unchecked_mut(1) = idx * (block[1] * first_elems).sum();
            *fut.get_unchecked_mut(2) = idx * (block[2] * first_elems).sum();
            *fut.get_unchecked_mut(3) = idx * (block[3] * first_elems).sum()
        };

        let diag = f32x8::new(
            Self::DISS_DIAG[0],
            Self::DISS_DIAG[1],
            Self::DISS_DIAG[2],
            Self::DISS_DIAG[3],
            Self::DISS_DIAG[4],
            Self::DISS_DIAG[5],
            Self::DISS_DIAG[6],
            0.0,
        );
        for (f, p) in fut
            .iter_mut()
            .skip(block.len())
            .zip(
                prev.windows(f32x8::lanes())
                    .map(f32x8::from_slice_unaligned)
                    .skip(1),
            )
            .take(nx - 2 * block.len())
        {
            *f = idx * (p * diag).sum();
        }

        let last_elems = unsafe { f32x8::from_slice_unaligned_unchecked(&prev[nx - 8..]) }
            .shuffle1_dyn(u32x8::new(7, 6, 5, 4, 3, 2, 1, 0));
        unsafe {
            *fut.get_unchecked_mut(nx - 4) = idx * (block[3] * last_elems).sum();
            *fut.get_unchecked_mut(nx - 3) = idx * (block[2] * last_elems).sum();
            *fut.get_unchecked_mut(nx - 2) = idx * (block[1] * last_elems).sum();
            *fut.get_unchecked_mut(nx - 1) = idx * (block[0] * last_elems).sum();
        }
    }

    #[inline(never)]
    fn disseta_simd(prev: &[f32], fut: &mut [f32], nx: usize, ny: usize) {
        assert!(ny >= 2 * Self::DISS_BLOCK.len());
        assert!(nx >= SimdT::lanes());
        assert!(nx % SimdT::lanes() == 0);
        assert_eq!(prev.len(), fut.len());
        assert_eq!(prev.len(), nx * ny);

        let dy = 1.0 / (ny - 1) as f32;
        let idy = 1.0 / dy;

        for j in (0..nx).step_by(SimdT::lanes()) {
            let a = [
                SimdT::from_slice_unaligned(&prev[0 * nx + j..]),
                SimdT::from_slice_unaligned(&prev[1 * nx + j..]),
                SimdT::from_slice_unaligned(&prev[2 * nx + j..]),
                SimdT::from_slice_unaligned(&prev[3 * nx + j..]),
                SimdT::from_slice_unaligned(&prev[4 * nx + j..]),
                SimdT::from_slice_unaligned(&prev[5 * nx + j..]),
                SimdT::from_slice_unaligned(&prev[6 * nx + j..]),
            ];

            for (i, bl) in Self::DISS_BLOCK.iter().enumerate() {
                let b = idy
                    * (a[0] * bl[0]
                        + a[1] * bl[1]
                        + a[2] * bl[2]
                        + a[3] * bl[3]
                        + a[4] * bl[4]
                        + a[5] * bl[5]
                        + a[6] * bl[6]);
                b.write_to_slice_unaligned(&mut fut[i * nx + j..]);
            }

            let mut a = a;
            for i in Self::DISS_BLOCK.len()..ny - Self::DISS_BLOCK.len() {
                // Push a onto circular buffer
                a = [
                    a[1],
                    a[2],
                    a[3],
                    a[4],
                    a[5],
                    a[6],
                    SimdT::from_slice_unaligned(&prev[nx * (i + 3) + j..]),
                ];
                let b = idy
                    * (a[0] * Self::DISS_DIAG[0]
                        + a[1] * Self::DISS_DIAG[1]
                        + a[2] * Self::DISS_DIAG[2]
                        + a[3] * Self::DISS_DIAG[3]
                        + a[4] * Self::DISS_DIAG[4]
                        + a[5] * Self::DISS_DIAG[5]
                        + a[6] * Self::DISS_DIAG[6]);
                b.write_to_slice_unaligned(&mut fut[nx * i + j..]);
            }

            let a = [
                SimdT::from_slice_unaligned(&prev[(ny - 1) * nx + j..]),
                SimdT::from_slice_unaligned(&prev[(ny - 2) * nx + j..]),
                SimdT::from_slice_unaligned(&prev[(ny - 3) * nx + j..]),
                SimdT::from_slice_unaligned(&prev[(ny - 4) * nx + j..]),
                SimdT::from_slice_unaligned(&prev[(ny - 5) * nx + j..]),
                SimdT::from_slice_unaligned(&prev[(ny - 6) * nx + j..]),
                SimdT::from_slice_unaligned(&prev[(ny - 7) * nx + j..]),
            ];

            for (i, bl) in Self::DISS_BLOCK.iter().enumerate() {
                let b = idy
                    * (a[0] * bl[0]
                        + a[1] * bl[1]
                        + a[2] * bl[2]
                        + a[3] * bl[3]
                        + a[4] * bl[4]
                        + a[5] * bl[5]
                        + a[6] * bl[6]);
                b.write_to_slice_unaligned(&mut fut[(ny - 1 - i) * nx + j..]);
            }
        }
    }
}

impl SbpOperator for Upwind4 {
    fn diffxi(prev: ArrayView2<f32>, mut fut: ArrayViewMut2<f32>) {
        assert_eq!(prev.shape(), fut.shape());
        assert!(prev.shape()[1] >= 2 * Self::BLOCK.len());

        match (prev.strides(), fut.strides()) {
            ([_, 1], [_, _]) => {
                Self::diff_simd_row(prev, fut);
            }
            ([1, _], [1, _]) if prev.len_of(Axis(0)) % SimdT::lanes() == 0 => {
                Self::diff_simd_col(prev, fut);
            }
            ([_, _], [_, _]) => {
                // Fallback, work row by row
                for (r0, r1) in prev.outer_iter().zip(fut.outer_iter_mut()) {
                    Self::diff_1d(r0, r1);
                }
            }
            _ => unreachable!("Should only be two elements in the strides vectors"),
        }
    }

    fn diffeta(prev: ArrayView2<f32>, fut: ArrayViewMut2<f32>) {
        // transpose then use diffxi
        Self::diffxi(prev.reversed_axes(), fut.reversed_axes());
    }

    fn h() -> &'static [f32] {
        Self::HBLOCK
    }
}

#[test]
fn upwind4_test() {
    use ndarray::prelude::*;
    let nx = 20;
    let dx = 1.0 / (nx - 1) as f32;
    let mut source: ndarray::Array1<f32> = ndarray::Array1::zeros(nx);
    let mut res = ndarray::Array1::zeros(nx);
    let mut target = ndarray::Array1::zeros(nx);

    for i in 0..nx {
        source[i] = i as f32 * dx;
        target[i] = 1.0;
    }
    res.fill(0.0);
    Upwind4::diff_1d(source.view(), res.view_mut());
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
        let x = i as f32 * dx;
        source[i] = x * x;
        target[i] = 2.0 * x;
    }
    res.fill(0.0);
    Upwind4::diff_1d(source.view(), res.view_mut());
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
        let x = i as f32 * dx;
        source[i] = x * x * x;
        target[i] = 3.0 * x * x;
    }
    res.fill(0.0);
    Upwind4::diff_1d(source.view(), res.view_mut());
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
    fn dissxi(prev: ArrayView2<f32>, mut fut: ArrayViewMut2<f32>) {
        assert_eq!(prev.shape(), fut.shape());
        assert!(prev.shape()[1] >= 2 * Self::DISS_BLOCK.len());
        for (r0, r1) in prev.outer_iter().zip(fut.outer_iter_mut()) {
            Self::diss_1d(r0, r1)
        }
    }

    fn disseta(prev: ArrayView2<f32>, mut fut: ArrayViewMut2<f32>) {
        assert_eq!(prev.shape(), fut.shape());
        assert!(prev.shape()[0] >= 2 * Self::DISS_BLOCK.len());
        let nx = prev.shape()[1];
        let ny = prev.shape()[0];
        if nx >= SimdT::lanes() && nx % SimdT::lanes() == 0 {
            if let (Some(p), Some(f)) = (prev.as_slice(), fut.as_slice_mut()) {
                Self::disseta_simd(p, f, nx, ny);
                return;
            }
        }
        // diffeta = transpose then use diffxi
        Self::dissxi(prev.reversed_axes(), fut.reversed_axes());
    }
}
