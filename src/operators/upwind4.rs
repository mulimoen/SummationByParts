use super::SbpOperator;
use ndarray::{arr1, arr2, s, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};

pub struct Upwind4 {}

impl Upwind4 {
    const HBLOCK: &'static [f32] = &[49.0 / 144.0, 61.0 / 48.0, 41.0 / 48.0, 149.0 / 144.0];
    const DIAG: &'static [f32] = &[
        -1.0 / 24.0,
        1.0 / 4.0,
        -7.0 / 8.0,
        0.0,
        7.0 / 8.0,
        -1.0 / 4.0,
        1.0 / 24.0,
    ];
    const BLOCK: &'static [[f32; 7]] = &[
        [
            -72.0 / 49.0_f32,
            187.0 / 98.0,
            -20.0 / 49.0,
            -3.0 / 98.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            -187.0 / 366.0,
            0.0,
            69.0 / 122.0,
            -16.0 / 183.0,
            2.0 / 61.0,
            0.0,
            0.0,
        ],
        [
            20.0 / 123.0,
            -69.0 / 82.0,
            0.0,
            227.0 / 246.0,
            -12.0 / 41.0,
            2.0 / 41.0,
            0.0,
        ],
        [
            3.0 / 298.0,
            16.0 / 149.0,
            -227.0 / 298.0,
            0.0,
            126.0 / 149.0,
            -36.0 / 149.0,
            6.0 / 149.0,
        ],
    ];

    fn diff_simd(prev: &[f32], fut: &mut [f32]) {
        use packed_simd::{f32x8, u32x8};
        assert_eq!(prev.len(), fut.len());
        assert!(prev.len() > 8);
        let nx = prev.len();
        let dx = 1.0 / (nx - 1) as f32;
        let idx = 1.0 / dx;

        let first_elems = unsafe { f32x8::from_slice_unaligned_unchecked(prev) };
        let block = [
            f32x8::new(
                Self::BLOCK[0][0],
                Self::BLOCK[0][1],
                Self::BLOCK[0][2],
                Self::BLOCK[0][3],
                Self::BLOCK[0][4],
                Self::BLOCK[0][5],
                Self::BLOCK[0][6],
                0.0,
            ),
            f32x8::new(
                Self::BLOCK[1][0],
                Self::BLOCK[1][1],
                Self::BLOCK[1][2],
                Self::BLOCK[1][3],
                Self::BLOCK[1][4],
                Self::BLOCK[1][5],
                Self::BLOCK[1][6],
                0.0,
            ),
            f32x8::new(
                Self::BLOCK[2][0],
                Self::BLOCK[2][1],
                Self::BLOCK[2][2],
                Self::BLOCK[2][3],
                Self::BLOCK[2][4],
                Self::BLOCK[2][5],
                Self::BLOCK[2][6],
                0.0,
            ),
            f32x8::new(
                Self::BLOCK[3][0],
                Self::BLOCK[3][1],
                Self::BLOCK[3][2],
                Self::BLOCK[3][3],
                Self::BLOCK[3][4],
                Self::BLOCK[3][5],
                Self::BLOCK[3][6],
                0.0,
            ),
        ];
        unsafe {
            *fut.get_unchecked_mut(0) += idx * (block[0] * first_elems).sum();
            *fut.get_unchecked_mut(1) += idx * (block[1] * first_elems).sum();
            *fut.get_unchecked_mut(2) += idx * (block[2] * first_elems).sum();
            *fut.get_unchecked_mut(3) += idx * (block[3] * first_elems).sum()
        };

        let diag = f32x8::new(
            Self::DIAG[0],
            Self::DIAG[1],
            Self::DIAG[2],
            Self::DIAG[3],
            Self::DIAG[4],
            Self::DIAG[5],
            Self::DIAG[6],
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
            *f += idx * (p * diag).sum();
        }

        let last_elems = unsafe { f32x8::from_slice_unaligned_unchecked(&prev[nx - 8..]) }
            .shuffle1_dyn(u32x8::new(7, 6, 5, 4, 3, 2, 1, 0));
        unsafe {
            *fut.get_unchecked_mut(nx - 4) += -idx * (block[3] * last_elems).sum();
            *fut.get_unchecked_mut(nx - 3) += -idx * (block[2] * last_elems).sum();
            *fut.get_unchecked_mut(nx - 2) += -idx * (block[1] * last_elems).sum();
            *fut.get_unchecked_mut(nx - 1) += -idx * (block[0] * last_elems).sum();
        }
    }

    fn diffy_simd(prev: &[f32], fut: &mut [f32], nx: usize, ny: usize) {
        use packed_simd::f32x4;
        assert!(ny > 8);
        assert!(nx > 4);
        assert!(nx % 4 == 0);
        assert_eq!(prev.len(), fut.len());
        assert_eq!(prev.len(), nx * ny);

        let dy = 1.0 / (ny - 1) as f32;
        let idy = 1.0 / dy;

        for j in (0..nx).step_by(4) {
            let a = [
                f32x4::from_slice_unaligned(&prev[0 * nx + j..]),
                f32x4::from_slice_unaligned(&prev[1 * nx + j..]),
                f32x4::from_slice_unaligned(&prev[2 * nx + j..]),
                f32x4::from_slice_unaligned(&prev[3 * nx + j..]),
                f32x4::from_slice_unaligned(&prev[4 * nx + j..]),
                f32x4::from_slice_unaligned(&prev[5 * nx + j..]),
                f32x4::from_slice_unaligned(&prev[6 * nx + j..]),
            ];

            for (i, bl) in Self::BLOCK.iter().enumerate() {
                let mut b = f32x4::from_slice_unaligned(&fut[i * nx + j..]);
                b += idy
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
            for i in Self::BLOCK.len()..ny - Self::BLOCK.len() {
                // Push a onto circular buffer
                a = [
                    a[1],
                    a[2],
                    a[3],
                    a[4],
                    a[5],
                    a[6],
                    f32x4::from_slice_unaligned(&prev[nx * (i + 3) + j..]),
                ];
                let mut b = f32x4::from_slice_unaligned(&fut[nx * i + j..]);
                b += idy
                    * (a[0] * Self::DIAG[0]
                        + a[1] * Self::DIAG[1]
                        + a[2] * Self::DIAG[2]
                        + a[3] * Self::DIAG[3]
                        + a[4] * Self::DIAG[4]
                        + a[5] * Self::DIAG[5]
                        + a[6] * Self::DIAG[6]);
                b.write_to_slice_unaligned(&mut fut[nx * i + j..]);
            }

            let a = [
                f32x4::from_slice_unaligned(&prev[(ny - 1) * nx + j..]),
                f32x4::from_slice_unaligned(&prev[(ny - 2) * nx + j..]),
                f32x4::from_slice_unaligned(&prev[(ny - 3) * nx + j..]),
                f32x4::from_slice_unaligned(&prev[(ny - 4) * nx + j..]),
                f32x4::from_slice_unaligned(&prev[(ny - 5) * nx + j..]),
                f32x4::from_slice_unaligned(&prev[(ny - 6) * nx + j..]),
                f32x4::from_slice_unaligned(&prev[(ny - 7) * nx + j..]),
            ];

            for (i, bl) in Self::BLOCK.iter().enumerate() {
                let mut b = f32x4::from_slice_unaligned(&fut[(ny - 1 - i) * nx + j..]);
                b += -idy
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

    fn diff(prev: ArrayView1<f32>, mut fut: ArrayViewMut1<f32>) {
        assert_eq!(prev.shape(), fut.shape());
        let nx = prev.shape()[0];

        if let (Some(p), Some(f)) = (prev.as_slice(), fut.as_slice_mut()) {
            Self::diff_simd(p, f);
            return;
        }

        let dx = 1.0 / (nx - 1) as f32;
        let idx = 1.0 / dx;

        let diag = arr1(Self::DIAG);
        let block = arr2(Self::BLOCK);

        let first_elems = prev.slice(s!(..7));
        for (bl, f) in block.outer_iter().zip(&mut fut) {
            let diff = first_elems.dot(&bl);
            *f += diff * idx;
        }

        for (window, f) in prev
            .windows(diag.len())
            .into_iter()
            .skip(1)
            .zip(fut.iter_mut().skip(4))
            .take(nx - 8)
        {
            let diff = diag.dot(&window);
            *f += diff * idx;
        }

        let last_elems = prev.slice(s!(nx - 7..;-1));
        for (bl, f) in block.outer_iter().zip(&mut fut.slice_mut(s![nx - 4..;-1])) {
            let diff = -bl.dot(&last_elems);
            *f += diff * idx;
        }
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
    Upwind4::diff(source.view(), res.view_mut());
    approx::assert_abs_diff_eq!(&res, &target, epsilon = 1e-4);
    {
        let source = source.to_owned().insert_axis(ndarray::Axis(0));
        let mut res = res.to_owned().insert_axis(ndarray::Axis(0));
        let target = target.to_owned().insert_axis(ndarray::Axis(0));
        res.fill(0.0);
        Upwind4::diffx(source.view(), res.view_mut());
        approx::assert_abs_diff_eq!(&res, &target, epsilon = 1e-2);
    }

    {
        let source = Array2::from_shape_fn((nx, 8), |(i, _)| source[i]);
        let target = Array2::from_shape_fn((nx, 8), |(i, _)| target[i]);
        let mut res = Array2::zeros((nx, 8));
        res.fill(0.0);
        Upwind4::diffy(source.view(), res.view_mut());
        approx::assert_abs_diff_eq!(&res.to_owned(), &target.to_owned(), epsilon = 1e-2);
    }

    for i in 0..nx {
        let x = i as f32 * dx;
        source[i] = x * x;
        target[i] = 2.0 * x;
    }
    res.fill(0.0);
    Upwind4::diff(source.view(), res.view_mut());
    approx::assert_abs_diff_eq!(&res, &target, epsilon = 1e-4);
    {
        let source = source.to_owned().insert_axis(ndarray::Axis(0));
        let mut res = res.to_owned().insert_axis(ndarray::Axis(0));
        let target = target.to_owned().insert_axis(ndarray::Axis(0));
        res.fill(0.0);
        Upwind4::diffx(source.view(), res.view_mut());
        approx::assert_abs_diff_eq!(&res, &target, epsilon = 1e-2);
    }

    {
        let source = Array2::from_shape_fn((nx, 8), |(i, _)| source[i]);
        let target = Array2::from_shape_fn((nx, 8), |(i, _)| target[i]);
        let mut res = Array2::zeros((nx, 8));
        res.fill(0.0);
        Upwind4::diffy(source.view(), res.view_mut());
        approx::assert_abs_diff_eq!(&res.to_owned(), &target.to_owned(), epsilon = 1e-2);
    }

    for i in 0..nx {
        let x = i as f32 * dx;
        source[i] = x * x * x;
        target[i] = 3.0 * x * x;
    }
    res.fill(0.0);
    Upwind4::diff(source.view(), res.view_mut());
    approx::assert_abs_diff_eq!(&res, &target, epsilon = 1e-2);

    {
        let source = source.to_owned().insert_axis(ndarray::Axis(0));
        let mut res = res.to_owned().insert_axis(ndarray::Axis(0));
        let target = target.to_owned().insert_axis(ndarray::Axis(0));
        res.fill(0.0);
        Upwind4::diffx(source.view(), res.view_mut());
        approx::assert_abs_diff_eq!(&res, &target, epsilon = 1e-2);
    }

    {
        let source = Array2::from_shape_fn((nx, 8), |(i, _)| source[i]);
        let target = Array2::from_shape_fn((nx, 8), |(i, _)| target[i]);
        let mut res = Array2::zeros((nx, 8));
        res.fill(0.0);
        Upwind4::diffy(source.view(), res.view_mut());
        approx::assert_abs_diff_eq!(&res.to_owned(), &target.to_owned(), epsilon = 1e-2);
    }
}

impl SbpOperator for Upwind4 {
    fn diffx(prev: ArrayView2<f32>, mut fut: ArrayViewMut2<f32>) {
        for (r0, r1) in prev.outer_iter().zip(fut.outer_iter_mut()) {
            Self::diff(r0, r1)
        }
    }

    fn diffy(prev: ArrayView2<f32>, mut fut: ArrayViewMut2<f32>) {
        let nx = prev.shape()[1];
        let ny = prev.shape()[0];
        if nx >= 4 && nx % 4 == 0 {
            if let (Some(p), Some(f)) = (prev.as_slice(), fut.as_slice_mut()) {
                Self::diffy_simd(p, f, nx, ny);
                return;
            }
        }
        // diffy = transpose then use diffx
        Self::diffx(prev.reversed_axes(), fut.reversed_axes());
    }

    fn h() -> &'static [f32] {
        Self::HBLOCK
    }
}
