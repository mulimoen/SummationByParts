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

    fn diff(prev: ArrayView1<f32>, mut fut: ArrayViewMut1<f32>) {
        assert_eq!(prev.shape(), fut.shape());
        let nx = prev.shape()[0];

        let dx = 1.0 / (nx - 1) as f32;

        let diag = arr1(Self::DIAG);
        let block = arr2(Self::BLOCK);

        let first_elems = prev.slice(s!(..7));
        for i in 0..4 {
            let diff = first_elems.dot(&block.slice(s!(i, ..)));
            fut[i] += diff / dx;
        }

        for i in 4..nx - 4 {
            let diff = diag.dot(&prev.slice(s!(i - 3..=i + 3)));
            fut[(i)] += diff / dx;
        }
        let last_elems = prev.slice(s!(nx - 7..));
        for i in 0..4 {
            let ii = nx - 4 + i;
            let block = block.slice(s!(3 - i, ..;-1));
            let diff = last_elems.dot(&block);
            fut[ii] += -diff / dx;
        }
    }
}

#[test]
fn upwind4_test() {
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

    for i in 0..nx {
        let x = i as f32 * dx;
        source[i] = x * x;
        target[i] = 2.0 * x;
    }
    res.fill(0.0);
    Upwind4::diff(source.view(), res.view_mut());
    approx::assert_abs_diff_eq!(&res, &target, epsilon = 1e-4);

    for i in 0..nx {
        let x = i as f32 * dx;
        source[i] = x * x * x;
        target[i] = 3.0 * x * x;
    }
    res.fill(0.0);
    Upwind4::diff(source.view(), res.view_mut());
    approx::assert_abs_diff_eq!(&res, &target, epsilon = 1e-2);
}

impl SbpOperator for Upwind4 {
    fn diffx(prev: ArrayView2<f32>, mut fut: ArrayViewMut2<f32>) {
        for j in 0..prev.shape()[0] {
            Self::diff(prev.slice(s!(j, ..)), fut.slice_mut(s!(j, ..)));
        }
    }

    fn diffy(prev: ArrayView2<f32>, mut fut: ArrayViewMut2<f32>) {
        for i in 0..prev.shape()[1] {
            Self::diff(prev.slice(s!(.., i)), fut.slice_mut(s!(.., i)));
        }
    }

    fn h() -> &'static [f32] {
        Self::HBLOCK
    }
}
