use ndarray::{arr1, arr2, s, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};

pub(crate) fn diffx(prev: ArrayView2<f32>, mut fut: ArrayViewMut2<f32>) {
    for j in 0..prev.shape()[0] {
        upwind4(prev.slice(s!(j, ..)), fut.slice_mut(s!(j, ..)));
    }
}

pub(crate) fn diffx_periodic(prev: ArrayView2<f32>, mut fut: ArrayViewMut2<f32>) {
    for j in 0..prev.shape()[0] {
        upwind4_periodic(prev.slice(s!(j, ..)), fut.slice_mut(s!(j, ..)));
    }
}

pub(crate) fn diffy(prev: ArrayView2<f32>, mut fut: ArrayViewMut2<f32>) {
    for i in 0..prev.shape()[1] {
        upwind4(prev.slice(s!(.., i)), fut.slice_mut(s!(.., i)));
    }
}

pub(crate) fn diffy_periodic(prev: ArrayView2<f32>, mut fut: ArrayViewMut2<f32>) {
    for i in 0..prev.shape()[1] {
        upwind4_periodic(prev.slice(s!(.., i)), fut.slice_mut(s!(.., i)));
    }
}

pub(crate) fn dissx(prev: ArrayView2<f32>, mut fut: ArrayViewMut2<f32>) {
    for j in 0..prev.shape()[0] {
        upwind4_diss(prev.slice(s!(j, ..)), fut.slice_mut(s!(j, ..)));
    }
}
pub(crate) fn dissy(prev: ArrayView2<f32>, mut fut: ArrayViewMut2<f32>) {
    for i in 0..prev.shape()[1] {
        upwind4_diss(prev.slice(s!(.., i)), fut.slice_mut(s!(.., i)));
    }
}

fn trad4_periodic(prev: ArrayView1<f32>, mut fut: ArrayViewMut1<f32>) {
    assert_eq!(prev.shape(), fut.shape());
    let nx = prev.shape()[0];

    let dx = 1.0 / (nx - 1) as f32;

    let diag = [1.0 / 12.0, -2.0 / 3.0, 0.0, 2.0 / 3.0, -1.0 / 12.0];

    let diff = diag[0] * prev[(nx - 2)]
        + diag[1] * prev[(nx - 1)]
        + diag[2] * prev[(0)]
        + diag[3] * prev[(1)]
        + diag[4] * prev[(2)];
    fut[(0)] += diff / dx;
    let diff = diag[0] * prev[(nx - 1)]
        + diag[1] * prev[(0)]
        + diag[2] * prev[(1)]
        + diag[3] * prev[(2)]
        + diag[4] * prev[(3)];
    fut[(1)] += diff / dx;
    for i in 2..nx - 2 {
        let diff = diag[0] * prev[(i - 2)]
            + diag[1] * prev[(i - 1)]
            + diag[2] * prev[(i)]
            + diag[3] * prev[(i + 1)]
            + diag[4] * prev[(i + 2)];
        fut[(i)] += diff / dx;
    }
    let diff = diag[0] * prev[(nx - 4)]
        + diag[1] * prev[(nx - 3)]
        + diag[2] * prev[(nx - 2)]
        + diag[3] * prev[(nx - 1)]
        + diag[4] * prev[(0)];
    fut[(nx - 2)] += diff / dx;
    let diff = diag[0] * prev[(nx - 3)]
        + diag[1] * prev[(nx - 2)]
        + diag[2] * prev[(nx - 1)]
        + diag[3] * prev[(0)]
        + diag[4] * prev[(1)];
    fut[(nx - 1)] += diff / dx;
}

fn upwind4(prev: ArrayView1<f32>, mut fut: ArrayViewMut1<f32>) {
    assert_eq!(prev.shape(), fut.shape());
    let nx = prev.shape()[0];

    let dx = 1.0 / (nx - 1) as f32;

    let diag = arr1(&[
        -1.0 / 24.0,
        1.0 / 4.0,
        -7.0 / 8.0,
        0.0,
        7.0 / 8.0,
        -1.0 / 4.0,
        1.0 / 24.0,
    ]);

    let block = arr2(&[
        [
            -72.0 / 49.0f32,
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
    ]);

    let h_block = [49.0 / 144.0, 61.0 / 48.0, 41.0 / 48.0, 149.0 / 144.0];

    let first_elems = prev.slice(s!(..7));
    for i in 0..4 {
        let diff = first_elems.dot(&block.slice(s!(i, ..)));
        fut[i] += diff / dx * h_block[i];
    }

    for i in 4..nx - 4 {
        let diff = diag.dot(&prev.slice(s!(i - 3..i + 3 + 1)));
        fut[(i)] += diff / dx;
    }

    let last_elems = prev.slice(s!(nx - 7..));
    for i in 0..4 {
        let ii = nx - 4 + i;
        let block = block.slice(s!(3 - i, ..;-1));
        let diff = last_elems.dot(&block);
        fut[ii] += diff / dx * h_block[3 - i];
    }
}

fn upwind4_periodic(prev: ArrayView1<f32>, mut fut: ArrayViewMut1<f32>) {
    assert_eq!(prev.shape(), fut.shape());
    let nx = prev.shape()[0];

    let dx = 1.0 / (nx - 1) as f32;

    let diag = [
        -1.0 / 24.0,
        1.0 / 4.0,
        -7.0 / 8.0,
        0.0,
        7.0 / 8.0,
        -1.0 / 4.0,
        1.0 / 24.0,
    ];

    let diff = diag[0] * prev[(nx - 3)]
        + diag[1] * prev[(nx - 2)]
        + diag[2] * prev[(nx - 1)]
        + diag[3] * prev[(0)]
        + diag[4] * prev[(1)]
        + diag[5] * prev[(2)]
        + diag[6] * prev[(3)];
    fut[0] += diff / dx;
    let diff = diag[0] * prev[(nx - 2)]
        + diag[1] * prev[(nx - 1)]
        + diag[2] * prev[(0)]
        + diag[3] * prev[(1)]
        + diag[4] * prev[(2)]
        + diag[5] * prev[(3)]
        + diag[6] * prev[(4)];
    fut[1] += diff / dx;
    let diff = diag[0] * prev[(nx - 1)]
        + diag[1] * prev[(0)]
        + diag[2] * prev[(1)]
        + diag[3] * prev[(2)]
        + diag[4] * prev[(3)]
        + diag[5] * prev[(4)]
        + diag[6] * prev[(5)];
    fut[2] += diff / dx;

    for i in 3..nx - 3 {
        let diff = diag[0] * prev[(i - 3)]
            + diag[1] * prev[(i - 2)]
            + diag[2] * prev[(i - 1)]
            + diag[3] * prev[(i)]
            + diag[4] * prev[(i + 1)]
            + diag[5] * prev[(i + 2)]
            + diag[6] * prev[(i + 3)];
        fut[(i)] += diff / dx;
    }
    let diff = diag[0] * prev[(nx - 6)]
        + diag[1] * prev[(nx - 5)]
        + diag[2] * prev[(nx - 4)]
        + diag[3] * prev[(nx - 3)]
        + diag[4] * prev[(nx - 2)]
        + diag[5] * prev[(nx - 1)]
        + diag[6] * prev[(0)];
    fut[(nx - 3)] += diff / dx;
    let diff = diag[0] * prev[(nx - 5)]
        + diag[1] * prev[(nx - 4)]
        + diag[2] * prev[(nx - 3)]
        + diag[3] * prev[(nx - 2)]
        + diag[4] * prev[(nx - 1)]
        + diag[5] * prev[(0)]
        + diag[6] * prev[(1)];
    fut[(nx - 2)] += diff / dx;
    let diff = diag[0] * prev[(nx - 4)]
        + diag[1] * prev[(nx - 3)]
        + diag[2] * prev[(nx - 2)]
        + diag[3] * prev[(nx - 1)]
        + diag[4] * prev[(0)]
        + diag[5] * prev[(1)]
        + diag[6] * prev[(2)];
    fut[(nx - 1)] += diff / dx;
}

fn upwind4_diss(prev: ArrayView1<f32>, mut fut: ArrayViewMut1<f32>) {
    assert_eq!(prev.shape(), fut.shape());
    let nx = prev.shape()[0];

    let dx = 1.0 / (nx - 1) as f32;

    let diag = [
        1.0 / 24.0,
        -1.0 / 4.0,
        5.0 / 8.0,
        -5.0 / 6.0,
        5.0 / 8.0,
        -1.0 / 4.0,
        1.0 / 24.0,
    ];

    let diff = diag[0] * prev[(nx - 3)]
        + diag[1] * prev[(nx - 2)]
        + diag[2] * prev[(nx - 1)]
        + diag[3] * prev[(0)]
        + diag[4] * prev[(1)]
        + diag[5] * prev[(2)]
        + diag[6] * prev[(3)];
    fut[0] += diff / dx;
    let diff = diag[0] * prev[(nx - 2)]
        + diag[1] * prev[(nx - 1)]
        + diag[2] * prev[(0)]
        + diag[3] * prev[(1)]
        + diag[4] * prev[(2)]
        + diag[5] * prev[(3)]
        + diag[6] * prev[(4)];
    fut[1] += diff / dx;
    let diff = diag[0] * prev[(nx - 1)]
        + diag[1] * prev[(0)]
        + diag[2] * prev[(1)]
        + diag[3] * prev[(2)]
        + diag[4] * prev[(3)]
        + diag[5] * prev[(4)]
        + diag[6] * prev[(5)];
    fut[2] += diff / dx;

    for i in 3..nx - 3 {
        let diff = diag[0] * prev[(i - 3)]
            + diag[1] * prev[(i - 2)]
            + diag[2] * prev[(i - 1)]
            + diag[3] * prev[(i)]
            + diag[4] * prev[(i + 1)]
            + diag[5] * prev[(i + 2)]
            + diag[6] * prev[(i + 3)];
        fut[(i)] += diff / dx;
    }
    let diff = diag[0] * prev[(nx - 6)]
        + diag[1] * prev[(nx - 5)]
        + diag[2] * prev[(nx - 4)]
        + diag[3] * prev[(nx - 3)]
        + diag[4] * prev[(nx - 2)]
        + diag[5] * prev[(nx - 1)]
        + diag[6] * prev[(0)];
    fut[(nx - 3)] += diff / dx;
    let diff = diag[0] * prev[(nx - 5)]
        + diag[1] * prev[(nx - 4)]
        + diag[2] * prev[(nx - 3)]
        + diag[3] * prev[(nx - 2)]
        + diag[4] * prev[(nx - 1)]
        + diag[5] * prev[(0)]
        + diag[6] * prev[(1)];
    fut[(nx - 2)] += diff / dx;
    let diff = diag[0] * prev[(nx - 4)]
        + diag[1] * prev[(nx - 3)]
        + diag[2] * prev[(nx - 2)]
        + diag[3] * prev[(nx - 1)]
        + diag[4] * prev[(0)]
        + diag[5] * prev[(1)]
        + diag[6] * prev[(2)];
    fut[(nx - 1)] += diff / dx;
}
