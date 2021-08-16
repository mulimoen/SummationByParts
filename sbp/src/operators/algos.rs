use super::*;
use ndarray::s;
use num_traits::Zero;
use std::convert::TryInto;

pub(crate) use constmatrix::{ColVector, Matrix, RowVector};

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DiagonalMatrix<const B: usize> {
    pub start: [Float; B],
    pub diag: Float,
    pub end: [Float; B],
}

impl<const B: usize> DiagonalMatrix<B> {
    pub const fn new(block: [Float; B]) -> Self {
        let start = block;
        let diag = 1.0;
        let mut end = block;
        let mut i = 0;
        while i < B {
            end[i] = block[B - 1 - i];
            i += 1;
        }
        Self { start, diag, end }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BlockMatrix<T, const M: usize, const N: usize, const D: usize> {
    pub start: Matrix<T, M, N>,
    pub diag: RowVector<T, D>,
    pub end: Matrix<T, M, N>,
}

impl<T, const M: usize, const N: usize, const D: usize> BlockMatrix<T, M, N, D> {
    pub const fn new(start: Matrix<T, M, N>, diag: RowVector<T, D>, end: Matrix<T, M, N>) -> Self {
        Self { start, diag, end }
    }
}

#[derive(PartialEq, Copy, Clone)]
pub(crate) enum OperatorType {
    Normal,
    H2,
    // TODO: D2
}

#[inline(always)]
/// Works on all 1d vectors
pub(crate) fn diff_op_1d_fallback<const M: usize, const N: usize, const D: usize>(
    matrix: &BlockMatrix<Float, M, N, D>,
    optype: OperatorType,
    prev: ArrayView1<Float>,
    mut fut: ArrayViewMut1<Float>,
) {
    assert_eq!(prev.shape(), fut.shape());
    let nx = prev.shape()[0];
    assert!(nx >= 2 * M);
    assert!(nx >= N);

    let dx = if optype == OperatorType::H2 {
        1.0 / (nx - 2) as Float
    } else {
        1.0 / (nx - 1) as Float
    };
    let idx = 1.0 / dx;

    let (futstart, futmid, futend) =
        fut.multi_slice_mut((s![..M], s![M..nx - 2 * M], s![nx - 2 * M..]));

    for (bl, f) in matrix.start.iter_rows().zip(futstart) {
        let diff = dotproduct(bl.iter(), prev.iter());
        *f = diff * idx;
    }

    // The window needs to be aligned to the diagonal elements,
    // based on the block size
    let window_elems_to_skip = M - ((D - 1) / 2);

    for (window, f) in prev
        .windows(D)
        .into_iter()
        .skip(window_elems_to_skip)
        .zip(futmid)
    {
        let diff = dotproduct(matrix.diag.row(0), window);
        *f = diff * idx;
    }

    let prev = prev.slice(ndarray::s![nx - N..]);
    for (bl, f) in matrix.end.iter_rows().zip(futend) {
        let diff = dotproduct(bl, prev);
        *f = diff * idx;
    }
}

#[inline(always)]
/// diff op in 1d for slices
pub(crate) fn diff_op_1d_slice<const M: usize, const N: usize, const D: usize>(
    matrix: &BlockMatrix<Float, M, N, D>,
    optype: OperatorType,
    prev: &[Float],
    fut: &mut [Float],
) {
    #[inline(never)]
    /// This prevents code bloat, both start and end block gives
    /// a matrix multiplication with the same matrix sizes
    fn dedup_matmul<const M: usize, const N: usize>(
        c: &mut ColVector<Float, M>,
        a: &Matrix<Float, M, N>,
        b: &ColVector<Float, N>,
    ) {
        c.matmul_float_into(a, b)
    }

    assert_eq!(prev.len(), fut.len());
    let nx = prev.len();
    assert!(nx >= 2 * M);
    assert!(nx >= N);
    let prev = &prev[..nx];
    let fut = &mut fut[..nx];

    let dx = if optype == OperatorType::H2 {
        1.0 / (nx - 2) as Float
    } else {
        1.0 / (nx - 1) as Float
    };
    let idx = 1.0 / dx;

    // Help aliasing analysis
    let (futb1, fut) = fut.split_at_mut(M);
    let (fut, futb2) = fut.split_at_mut(nx - 2 * M);

    {
        let prev = ColVector::<_, N>::map_to_col(prev.array_windows::<N>().next().unwrap());
        let fut = ColVector::<_, M>::map_to_col_mut(futb1.try_into().unwrap());

        dedup_matmul(fut, &matrix.start, prev);
        *fut *= idx;
    }

    {
        // The window needs to be aligned to the diagonal elements,
        // based on the block size
        let window_elems_to_skip = M - ((D - 1) / 2);

        // The compiler is pretty clever right here. It is able to
        // inline the entire computation into a very tight loop.
        // It seems the "optimal" way is to broadcast each element of diag
        // into separate SIMD vectors.
        // Then an unroll of SIMD::lanes items from fut and prev:
        // f[0] = diag[0] * in[0] + diag[1] * in[1] + diag[2] * in[3] + ...
        // f[1] = diag[0] * in[1] + diag[1] * in[2] + diag[2] * in[4] + ...
        //                   \-- fma's along the vertical axis, then horizontal
        //
        // The resulting inner loop performs:
        // one mul, D-1 fma,
        // one mul (by idx),
        // one store
        // two integer adds (input and output offsets)
        // one test + jump
        //
        // The compiler is clever enough to combine two such unrollings and merge
        // these computations to prevent stalling
        for (window, f) in prev[window_elems_to_skip..]
            .array_windows::<D>()
            .zip(fut.array_chunks_mut::<1>())
        {
            let fut = ColVector::<_, 1>::map_to_col_mut(f);
            let prev = ColVector::<_, D>::map_to_col(window);

            fut.matmul_float_into(&matrix.diag, prev);
            *fut *= idx;
        }
    }

    {
        let prev = prev.array_windows::<N>().next_back().unwrap();
        let prev = ColVector::<_, N>::map_to_col(prev);
        let fut = ColVector::<_, M>::map_to_col_mut(futb2.try_into().unwrap());

        dedup_matmul(fut, &matrix.end, prev);
        *fut *= idx;
    }
}

#[inline(always)]
/// Will always work on 1d, delegated based on slicedness
pub(crate) fn diff_op_1d<const M: usize, const N: usize, const D: usize>(
    matrix: &BlockMatrix<Float, M, N, D>,
    optype: OperatorType,
    prev: ArrayView1<Float>,
    mut fut: ArrayViewMut1<Float>,
) {
    assert_eq!(prev.shape(), fut.shape());
    let nx = prev.shape()[0];
    assert!(nx >= 2 * M);

    if let Some((prev, fut)) = prev.as_slice().zip(fut.as_slice_mut()) {
        diff_op_1d_slice(matrix, optype, prev, fut)
    } else {
        diff_op_1d_fallback(matrix, optype, prev, fut)
    }
}

#[inline(always)]
/// 2D diff fallback for when matrices are not slicable
pub(crate) fn diff_op_2d_fallback<const M: usize, const N: usize, const D: usize>(
    matrix: &BlockMatrix<Float, M, N, D>,
    optype: OperatorType,
    prev: ArrayView2<Float>,
    mut fut: ArrayViewMut2<Float>,
) {
    assert_eq!(prev.shape(), fut.shape());
    let nx = prev.shape()[1];
    let ny = prev.shape()[0];
    assert!(nx >= 2 * M);

    let dx = if optype == OperatorType::H2 {
        1.0 / (nx - 2) as Float
    } else {
        1.0 / (nx - 1) as Float
    };
    let idx = 1.0 / dx;

    fut.fill(0.0);
    let (mut fut0, mut futmid, mut futn) = fut.multi_slice_mut((
        ndarray::s![.., ..M],
        ndarray::s![.., M..nx - M],
        ndarray::s![.., nx - M..],
    ));

    // First block
    for (bl, mut fut) in matrix
        .start
        .iter_rows()
        .zip(fut0.axis_iter_mut(ndarray::Axis(1)))
    {
        debug_assert_eq!(fut.len(), prev.shape()[0]);
        for (&bl, prev) in bl.iter().zip(prev.axis_iter(ndarray::Axis(1))) {
            if bl.is_zero() {
                continue;
            }
            debug_assert_eq!(prev.len(), fut.len());
            fut.scaled_add(idx * bl, &prev);
        }
    }

    let window_elems_to_skip = M - ((D - 1) / 2);

    // Diagonal entries
    for (mut fut, id) in futmid
        .axis_iter_mut(ndarray::Axis(1))
        .zip(prev.windows((ny, D)).into_iter().skip(window_elems_to_skip))
    {
        for (&d, id) in matrix.diag.iter().zip(id.axis_iter(ndarray::Axis(1))) {
            if d.is_zero() {
                continue;
            }
            fut.scaled_add(idx * d, &id)
        }
    }

    // End block
    let prev = prev.slice(ndarray::s!(.., nx - N..));
    for (bl, mut fut) in matrix
        .end
        .iter_rows()
        .zip(futn.axis_iter_mut(ndarray::Axis(1)))
    {
        for (&bl, prev) in bl.iter().zip(prev.axis_iter(ndarray::Axis(1))) {
            if bl.is_zero() {
                continue;
            }
            fut.scaled_add(idx * bl, &prev);
        }
    }
}

#[inline(always)]
/// 2D diff when first axis is contiguous
#[allow(unused)]
#[allow(clippy::assign_op_pattern)]
pub(crate) fn diff_op_2d_sliceable_y<const M: usize, const N: usize, const D: usize>(
    matrix: &BlockMatrix<Float, M, N, D>,
    optype: OperatorType,
    prev: ArrayView2<Float>,
    mut fut: ArrayViewMut2<Float>,
) {
    assert_eq!(prev.shape(), fut.shape());
    let nx = prev.shape()[1];
    let ny = prev.shape()[0];
    assert!(nx >= 2 * M);
    assert_eq!(prev.strides()[0], 1);
    assert_eq!(fut.strides()[0], 1);

    let dx = if optype == OperatorType::H2 {
        1.0 / (nx - 2) as Float
    } else {
        1.0 / (nx - 1) as Float
    };
    let idx = 1.0 / dx;

    fut.fill(0.0);
    let (mut fut0, mut futmid, mut futn) = fut.multi_slice_mut((
        ndarray::s![.., ..M],
        ndarray::s![.., M..nx - M],
        ndarray::s![.., nx - M..],
    ));

    // First block
    for (bl, mut fut) in matrix
        .start
        .iter_rows()
        .zip(fut0.axis_iter_mut(ndarray::Axis(1)))
    {
        let fut = &mut fut.as_slice_mut().unwrap()[..ny];
        for (&bl, prev) in bl.iter().zip(prev.axis_iter(ndarray::Axis(1))) {
            if bl.is_zero() {
                continue;
            }
            let prev = &prev.as_slice().unwrap()[..ny];
            for (fut, prev) in fut.iter_mut().zip(prev) {
                *fut = *fut + idx * prev * bl;
            }
        }
    }

    let window_elems_to_skip = M - ((D - 1) / 2);

    // Diagonal entries
    for (mut fut, id) in futmid
        .axis_iter_mut(ndarray::Axis(1))
        .zip(prev.windows((ny, D)).into_iter().skip(window_elems_to_skip))
    {
        let fut = &mut fut.as_slice_mut().unwrap()[..ny];
        for (&d, id) in matrix.diag.iter().zip(id.axis_iter(ndarray::Axis(1))) {
            if d.is_zero() {
                continue;
            }
            let id = id.as_slice().unwrap();
            for (fut, id) in fut.iter_mut().zip(id) {
                *fut = *fut + idx * id * d;
            }
        }
    }

    // End block
    let prev = prev.slice(ndarray::s!(.., nx - N..));
    for (bl, mut fut) in matrix
        .end
        .iter_rows()
        .zip(futn.axis_iter_mut(ndarray::Axis(1)))
    {
        let fut = &mut fut.as_slice_mut().unwrap()[..ny];
        for (&bl, prev) in bl.iter().zip(prev.axis_iter(ndarray::Axis(1))) {
            if bl.is_zero() {
                continue;
            }
            let prev = &prev.as_slice().unwrap()[..ny];
            for (fut, prev) in fut.iter_mut().zip(prev) {
                *fut = *fut + idx * prev * bl;
            }
        }
    }
}

#[inline(always)]
pub(crate) fn diff_op_2d_sliceable_y_simd<const M: usize, const N: usize, const D: usize>(
    matrix: &BlockMatrix<Float, M, N, D>,
    optype: OperatorType,
    prev: ArrayView2<Float>,
    mut fut: ArrayViewMut2<Float>,
) {
    assert_eq!(prev.shape(), fut.shape());
    let nx = prev.shape()[1];
    let ny = prev.shape()[0];
    assert!(nx >= 2 * M);

    assert_eq!(prev.strides(), fut.strides());
    assert_eq!(prev.strides(), &[1, ny as isize]);

    let prev = prev.as_slice_memory_order().unwrap();
    let fut = fut.as_slice_memory_order_mut().unwrap();
    let prev = &prev[..nx * ny];
    let fut = &mut fut[..nx * ny];

    let dx = if optype == OperatorType::H2 {
        1.0 / (nx - 2) as Float
    } else {
        1.0 / (nx - 1) as Float
    };
    let idx = 1.0 / dx;

    use core_simd::Vector;
    #[cfg(not(feature = "f32"))]
    type SimdT = core_simd::f64x8;
    #[cfg(feature = "f32")]
    type SimdT = core_simd::f32x16;

    // How many elements that can be simdified
    let simdified = SimdT::LANES * (ny / SimdT::LANES);

    let (fut0, futmid) = fut.split_at_mut(M * ny);
    let (futmid, futn) = futmid.split_at_mut((nx - 2 * M) * ny);

    #[inline(always)]
    fn block_multiply<const M: usize, const N: usize>(
        matrix: &Matrix<Float, M, N>,
        idx: Float,
        ny: usize,
        simdified: usize,
        prev: &[Float],
        fut: &mut [Float],
    ) {
        assert_eq!(prev.len(), N * ny);
        assert_eq!(fut.len(), M * ny);
        let prev = &prev[..N * ny];
        let fut = &mut fut[..M * ny];

        /*
        let prevcol = {
            let prev_ptr = prev.as_ptr();
            move |i: usize| -> &[Float] {
                unsafe { std::slice::from_raw_parts(prev_ptr.add(i * ny), ny) }
            }
        };
        */
        let prevcol = |i: usize| -> &[Float] { &prev[i * ny..(i + 1) * ny] };

        for (&bl, fut) in matrix.iter_rows().zip(fut.chunks_exact_mut(ny)) {
            let mut fut = fut.array_chunks_mut::<{ SimdT::LANES }>();
            for (j, fut) in fut.by_ref().enumerate() {
                let index_to_simd = |i| {
                    SimdT::from_array(
                        (&prevcol(i)[SimdT::LANES * j..SimdT::LANES * (j + 1)])
                            .try_into()
                            .unwrap(),
                    )
                };
                let mut f = SimdT::splat(0.0);
                for (iprev, &bl) in bl.iter().enumerate() {
                    f = index_to_simd(iprev).mul_add(SimdT::splat(bl), f);
                }
                f *= idx;
                fut.clone_from_slice(f.as_array());
            }
            for (j, fut) in (simdified..ny).zip(fut.into_remainder()) {
                let mut f = 0.0;
                for (iprev, bl) in bl.iter().enumerate() {
                    f += bl * prevcol(iprev)[j];
                }
                *fut = f * idx;
            }
        }
    }

    // First block
    {
        let prev = &prev[..N * ny];
        block_multiply(&matrix.start, idx, ny, simdified, prev, fut0);
    }

    // Diagonal elements
    {
        let half_diag_width = (D - 1) / 2;
        assert!(half_diag_width <= M);

        let prevcol = {
            let prev_ptr = prev.as_ptr();
            move |i: usize| -> &[Float] {
                unsafe { std::slice::from_raw_parts(prev_ptr.add(i * ny), ny) }
            }
        };
        //let prevcol = |i: usize| -> &[Float] { &prev[i * ny..(i + 1) * ny] };

        for (fut, ifut) in futmid.chunks_exact_mut(ny).zip(M..nx - M) {
            let mut fut = fut.array_chunks_mut::<{ SimdT::LANES }>();
            for (j, fut) in fut.by_ref().enumerate() {
                //let index_to_simd =
                //    |i| SimdT::from_slice_unaligned(&prevcol(i)[SimdT::lanes() * j..]);
                let index_to_simd = |i: usize| unsafe {
                    let prev = std::slice::from_raw_parts(
                        prev.as_ptr().add(i * ny + SimdT::LANES * j),
                        SimdT::LANES,
                    )
                    .try_into()
                    .unwrap();
                    SimdT::from_array(prev)
                };
                let mut f = SimdT::splat(0.0);
                // direct iter does not optimize well here
                for (id, &d) in matrix.diag.row(0).iter().enumerate() {
                    let offset = ifut - half_diag_width + id;
                    f = index_to_simd(offset).mul_add(SimdT::splat(d), f);
                }
                f *= idx;
                fut.clone_from_slice(f.as_array());
            }
            for (j, fut) in (simdified..ny).zip(fut.into_remainder()) {
                let mut f = 0.0;
                for (id, &d) in matrix.diag.iter().enumerate() {
                    let offset = ifut - half_diag_width + id;
                    {
                        f += d * prevcol(offset)[j];
                    }
                }
                *fut = idx * f;
            }
        }
    }

    // End block
    {
        let prev = &prev[((nx - N) * ny)..];
        block_multiply(&matrix.end, idx, ny, simdified, prev, futn);
    }
}

#[inline(always)]
pub(crate) fn diff_op_2d_sliceable<const M: usize, const N: usize, const D: usize>(
    matrix: &BlockMatrix<Float, M, N, D>,
    optype: OperatorType,
    prev: ArrayView2<Float>,
    mut fut: ArrayViewMut2<Float>,
) {
    assert_eq!(prev.shape(), fut.shape());
    let nx = prev.shape()[1];
    for (prev, mut fut) in prev.outer_iter().zip(fut.outer_iter_mut()) {
        let prev = &prev.as_slice().unwrap()[..nx];
        let fut = &mut fut.as_slice_mut().unwrap()[..nx];
        diff_op_1d_slice(matrix, optype, prev, fut)
    }
}

#[inline(always)]
/// Dispatch based on strides
pub(crate) fn diff_op_2d<const M: usize, const N: usize, const D: usize>(
    matrix: &BlockMatrix<Float, M, N, D>,
    optype: OperatorType,
    prev: ArrayView2<Float>,
    fut: ArrayViewMut2<Float>,
) {
    assert_eq!(prev.shape(), fut.shape());
    match (prev.strides(), fut.strides()) {
        ([_, 1], [_, 1]) => diff_op_2d_sliceable(matrix, optype, prev, fut),
        ([1, _], [1, _])
            if prev.as_slice_memory_order().is_some() && fut.as_slice_memory_order().is_some() =>
        {
            diff_op_2d_sliceable_y_simd(matrix, optype, prev, fut)
        }
        _ => diff_op_2d_fallback(matrix, optype, prev, fut),
    }
}

/*
#[inline(always)]
/// Way to too much overhead with SIMD:
/// output SIMD oriented:
/// |S     | =   |P0 P1|     |P0 P1|
/// |S     | = a1|P0 P1| + b1|P0 P1|
/// |S     | =   |P0 P1|     |P0 P1|
///
/// | S    | =   |P0 P1|     |P0 P1|
/// | S    | = a2|P0 P1| + b1|P0 P1|
/// | S    | =   |P0 P1|     |P0 P1|
pub(crate) fn diff_op_col_matrix<const M: usize, const N: usize, const D: usize>(
    matrix: &BlockMatrix<M, N, D>,
    optype: OperatorType,
    prev: ArrayView2<Float>,
    fut: ArrayViewMut2<Float>,
) {
    assert_eq!(prev.shape(), fut.shape());
    let nx = prev.shape()[1];
    assert!(nx >= 2 * M);

    assert_eq!(prev.strides()[0], 1);
    assert_eq!(fut.strides()[0], 1);

    let dx = if optype == OperatorType::H2 {
        1.0 / (nx - 2) as Float
    } else {
        1.0 / (nx - 1) as Float
    };
    let idx = 1.0 / dx;

    #[cfg(not(feature = "f32"))]
    type SimdT = packed_simd::f64x8;
    #[cfg(feature = "f32")]
    type SimdT = packed_simd::f32x16;

    let ny = prev.shape()[0];
    // How many elements that can be simdified
    let simdified = SimdT::lanes() * (ny / SimdT::lanes());

    let half_diag_width = (D - 1) / 2;
    assert!(half_diag_width <= M);

    let fut_stride = fut.strides()[1];

    let prev_base_ptr = prev.as_ptr();
    let prev_stride = prev.strides()[1];
    let prev_ptr = |j, i| {
        debug_assert!(j < ny && i < nx);
        unsafe { prev_base_ptr.offset(prev_stride * i as isize + j as isize) }
    };

    // Not algo necessary, but gives performance increase
    assert_eq!(fut_stride, prev_stride);

    use ndarray::Axis;
    let (mut fut1, fut) = fut.split_at(Axis(1), M);
    let (mut fut, mut fut2) = fut.split_at(Axis(1), nx - 2 * M);

    // First block
    {
        let prev = prev.slice(ndarray::s![.., ..N]);
        let (prevb, prevl) = prev.split_at(Axis(0), simdified);
        for (mut fut, &bl) in fut1.axis_iter_mut(Axis(1)).zip(matrix.start.iter_rows()) {
            let fut = fut.as_slice_mut().unwrap();
            let fut = &mut fut[..ny];

            let mut fut = fut.chunks_exact_mut(SimdT::lanes());
            let mut prev = prevb.axis_chunks_iter(Axis(0), SimdT::lanes());

            for (fut, prev) in fut.by_ref().zip(prev.by_ref()) {
                let mut f = SimdT::splat(0.0);
                for (&bl, prev) in bl.iter().zip(prev.axis_iter(Axis(1))) {
                    let prev = prev.to_slice().unwrap();
                    let prev = SimdT::from_slice_unaligned(prev);
                    f = prev.mul_adde(SimdT::splat(bl), f);
                }
                f *= idx;
                f.write_to_slice_unaligned(fut);
            }
            for (fut, prev) in fut
                .into_remainder()
                .iter_mut()
                .zip(prevl.axis_iter(Axis(0)))
            {
                let mut f = 0.0;
                for (bl, prev) in bl.iter().zip(prev.iter()) {
                    f += bl * prev;
                }
                *fut = f * idx;
            }
        }
    }

    // Diagonal elements
    {
        let window_elems_to_skip = M - ((D - 1) / 2);
        let prev = prev.slice(ndarray::s![.., window_elems_to_skip..]);
        let prev = prev.windows((ny, D));
        for (mut fut, prev) in fut.axis_iter_mut(Axis(1)).zip(prev) {
            let fut = fut.as_slice_mut().unwrap();
            let fut = &mut fut[..ny];

            let mut fut = fut.chunks_exact_mut(SimdT::lanes());

            let (prevb, prevl) = prev.split_at(Axis(0), simdified);
            let prev = prevb.axis_chunks_iter(Axis(0), SimdT::lanes());

            for (fut, prev) in fut.by_ref().zip(prev) {
                let mut f = SimdT::splat(0.0);
                for (&d, prev) in matrix.diag.iter().zip(prev.axis_iter(Axis(1))) {
                    let prev = prev.to_slice().unwrap();
                    let prev = SimdT::from_slice_unaligned(prev);
                    f = prev.mul_adde(SimdT::splat(d), f);
                }
                f *= idx;
                f.write_to_slice_unaligned(fut);
            }

            for (fut, prev) in fut
                .into_remainder()
                .into_iter()
                .zip(prevl.axis_iter(Axis(0)))
            {
                let mut f = 0.0;
                for (&d, prev) in matrix.diag.iter().zip(prev) {
                    f += d * prev;
                }
                *fut = idx * f;
            }
        }
    }

    // End block
    {
        for (mut fut, &bl) in fut2.axis_iter_mut(Axis(1)).zip(matrix.end.iter_rows()) {
            let fut = fut.as_slice_mut().unwrap();
            let fut = &mut fut[..ny];
            let mut fut = fut.chunks_exact_mut(SimdT::lanes());

            for (fut, j) in fut.by_ref().zip((0..simdified).step_by(SimdT::lanes())) {
                let index_to_simd = |i| unsafe {
                    // j never moves past end of slice due to step_by and
                    // rounding down
                    SimdT::from_slice_unaligned(std::slice::from_raw_parts(
                        prev_ptr(j, i),
                        SimdT::lanes(),
                    ))
                };
                let mut f = SimdT::splat(0.0);
                for (iprev, &bl) in (nx - N..nx).zip(bl.iter()) {
                    f = index_to_simd(iprev).mul_adde(SimdT::splat(bl), f);
                }
                f *= idx;
                f.write_to_slice_unaligned(fut);
            }
            for (fut, j) in fut.into_remainder().into_iter().zip(simdified..ny) {
                unsafe {
                    let mut f = 0.0;
                    for (iprev, bl) in (nx - N..nx).zip(bl.iter()) {
                        f += bl * *prev_ptr(j, iprev);
                    }
                    *fut = f * idx;
                }
            }
        }
    }
}
*/

#[inline(always)]
fn dotproduct<'a>(
    u: impl IntoIterator<Item = &'a Float>,
    v: impl IntoIterator<Item = &'a Float>,
) -> Float {
    u.into_iter()
        .zip(v.into_iter())
        .fold(0.0, |acc, (&u, &v)| Float::mul_add(u, v, acc))
}

#[cfg(feature = "sparse")]
pub(crate) fn sparse_from_block<const M: usize, const N: usize, const D: usize>(
    matrix: &BlockMatrix<Float, M, N, D>,
    optype: OperatorType,
    n: usize,
) -> sprs::CsMat<Float> {
    assert!(n >= 2 * M);

    let nnz = {
        let blockstart_elems = matrix
            .start
            .iter()
            .fold(0, |acc, &x| if x != 0.0 { acc + 1 } else { acc });

        let diag_elems = matrix
            .diag
            .iter()
            .fold(0, |acc, &x| if x != 0.0 { acc + 1 } else { acc });

        let blockend_elems = matrix
            .end
            .iter()
            .fold(0, |acc, &x| if x != 0.0 { acc + 1 } else { acc });

        blockstart_elems + (n - 2 * M) * diag_elems + blockend_elems
    };

    let mut mat = sprs::TriMat::with_capacity((n, n), nnz);

    let dx = if optype == OperatorType::H2 {
        1.0 / (n - 2) as Float
    } else {
        1.0 / (n - 1) as Float
    };
    let idx = 1.0 / dx;

    for (j, bl) in matrix.start.iter_rows().enumerate() {
        for (i, &b) in bl.iter().enumerate() {
            if b == 0.0 {
                continue;
            }
            mat.add_triplet(j, i, b * idx);
        }
    }

    for j in M..n - M {
        let half_diag_len = D / 2;
        for (&d, i) in matrix.diag.iter().zip(j - half_diag_len..) {
            if d == 0.0 {
                continue;
            }
            mat.add_triplet(j, i, d * idx);
        }
    }

    for (bl, j) in matrix.end.iter_rows().zip(n - M..) {
        for (&b, i) in bl.iter().zip(n - N..) {
            if b == 0.0 {
                continue;
            }
            mat.add_triplet(j, i, b * idx);
        }
    }

    mat.to_csr()
}

#[cfg(feature = "sparse")]
pub(crate) fn h_matrix<const D: usize>(
    hmatrix: &DiagonalMatrix<D>,
    n: usize,
    is_h2: bool,
) -> sprs::CsMat<Float> {
    let h = if is_h2 {
        1.0 / (n - 2) as Float
    } else {
        1.0 / (n - 1) as Float
    };
    let nmiddle = n - 2 * D;
    let iter = hmatrix
        .start
        .iter()
        .chain(std::iter::repeat(&hmatrix.diag).take(nmiddle))
        .chain(hmatrix.end.iter())
        .map(|&x| h * x);

    let mut mat = sprs::TriMat::with_capacity((n, n), n);
    for (i, d) in iter.enumerate() {
        mat.add_triplet(i, i, d);
    }
    mat.to_csr()
}
