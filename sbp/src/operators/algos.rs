use super::*;

pub(crate) mod constmatrix;
pub(crate) use constmatrix::{flip_lr, flip_sign, flip_ud, ColVector, Matrix, RowVector};

#[cfg(feature = "fast-float")]
mod fastfloat;
#[cfg(feature = "fast-float")]
use fastfloat::FastFloat;

#[inline(always)]
pub(crate) fn diff_op_1d_matrix<const M: usize, const N: usize, const D: usize>(
    block: &Matrix<Float, M, N>,
    blockend: &Matrix<Float, M, N>,
    diag: &RowVector<Float, D>,
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

    for (bl, f) in block.iter_rows().zip(&mut fut) {
        let diff = bl
            .iter()
            .zip(prev.iter())
            .map(|(x, y)| x * y)
            .sum::<Float>();
        *f = diff * idx;
    }

    // The window needs to be aligned to the diagonal elements,
    // based on the block size
    let window_elems_to_skip = M - ((D - 1) / 2);

    for (window, f) in prev
        .windows(D)
        .into_iter()
        .skip(window_elems_to_skip)
        .zip(fut.iter_mut().skip(M))
        .take(nx - 2 * M)
    {
        let diff = diag.iter().zip(&window).map(|(x, y)| x * y).sum::<Float>();
        *f = diff * idx;
    }

    let prev = prev.slice(ndarray::s![nx - N..]);
    for (bl, f) in blockend.iter_rows().zip(fut.iter_mut().rev().take(M).rev()) {
        let diff = bl
            .iter()
            .zip(prev.iter())
            .map(|(x, y)| x * y)
            .sum::<Float>();

        *f = diff * idx;
    }
}

#[inline(always)]
pub(crate) fn diff_op_1d_slice_matrix<const M: usize, const N: usize, const D: usize>(
    block: &Matrix<Float, M, N>,
    endblock: &Matrix<Float, M, N>,
    diag: &RowVector<Float, D>,
    optype: OperatorType,
    prev: &[Float],
    fut: &mut [Float],
) {
    #[cfg(feature = "fast-float")]
    let (block, endblock, diag, prev, fut) = {
        use std::mem::transmute;
        unsafe {
            (
                transmute::<_, &Matrix<FastFloat, M, N>>(block),
                transmute::<_, &Matrix<FastFloat, M, N>>(endblock),
                transmute::<_, &RowVector<FastFloat, D>>(diag),
                transmute::<_, &[FastFloat]>(prev),
                transmute::<_, &mut [FastFloat]>(fut),
            )
        }
    };

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
    #[cfg(feature = "fast-float")]
    let idx = FastFloat::from(idx);

    // Help aliasing analysis
    let (futb1, fut) = fut.split_at_mut(M);
    let (fut, futb2) = fut.split_at_mut(nx - 2 * M);

    use std::convert::TryInto;
    {
        let prev = ColVector::<_, N>::map_to_col(prev.array_windows::<N>().next().unwrap());
        let fut = ColVector::<_, M>::map_to_col_mut(futb1.try_into().unwrap());

        fut.matmul_into(block, prev);
        *fut *= idx;
    }

    // The window needs to be aligned to the diagonal elements,
    // based on the block size
    let window_elems_to_skip = M - ((D - 1) / 2);

    for (window, f) in prev
        .array_windows::<D>()
        .skip(window_elems_to_skip)
        .zip(fut.array_chunks_mut::<1>())
    {
        let fut = ColVector::<_, 1>::map_to_col_mut(f);
        let prev = ColVector::<_, D>::map_to_col(window);

        fut.matmul_into(diag, prev);
        *fut *= idx;
    }

    {
        let prev = prev.array_windows::<N>().next_back().unwrap();
        let prev = ColVector::<_, N>::map_to_col(prev);
        let fut = ColVector::<_, M>::map_to_col_mut(futb2.try_into().unwrap());

        fut.matmul_into(endblock, prev);
        *fut *= idx;
    }
}

#[inline(always)]
pub(crate) fn diff_op_1d(
    block: &[&[Float]],
    diag: &[Float],
    symmetry: Symmetry,
    optype: OperatorType,
    prev: ArrayView1<Float>,
    mut fut: ArrayViewMut1<Float>,
) {
    assert_eq!(prev.shape(), fut.shape());
    let nx = prev.shape()[0];
    assert!(nx >= 2 * block.len());

    let dx = if optype == OperatorType::H2 {
        1.0 / (nx - 2) as Float
    } else {
        1.0 / (nx - 1) as Float
    };
    let idx = 1.0 / dx;

    for (bl, f) in block.iter().zip(&mut fut) {
        let diff = bl
            .iter()
            .zip(prev.iter())
            .map(|(x, y)| x * y)
            .sum::<Float>();
        *f = diff * idx;
    }

    // The window needs to be aligned to the diagonal elements,
    // based on the block size
    let window_elems_to_skip = block.len() - ((diag.len() - 1) / 2);

    for (window, f) in prev
        .windows(diag.len())
        .into_iter()
        .skip(window_elems_to_skip)
        .zip(fut.iter_mut().skip(block.len()))
        .take(nx - 2 * block.len())
    {
        let diff = diag.iter().zip(&window).map(|(x, y)| x * y).sum::<Float>();
        *f = diff * idx;
    }

    for (bl, f) in block.iter().zip(fut.iter_mut().rev()) {
        let diff = bl
            .iter()
            .zip(prev.iter().rev())
            .map(|(x, y)| x * y)
            .sum::<Float>();

        *f = idx
            * if symmetry == Symmetry::Symmetric {
                diff
            } else {
                -diff
            };
    }
}

#[derive(PartialEq, Copy, Clone)]
pub(crate) enum Symmetry {
    Symmetric,
    AntiSymmetric,
}

#[derive(PartialEq, Copy, Clone)]
pub(crate) enum OperatorType {
    Normal,
    H2,
}

#[inline(always)]
#[allow(unused)]
pub(crate) fn diff_op_col_naive(
    block: &'static [&'static [Float]],
    diag: &'static [Float],
    symmetry: Symmetry,
    optype: OperatorType,
) -> impl Fn(ArrayView2<Float>, ArrayViewMut2<Float>) {
    #[inline(always)]
    move |prev: ArrayView2<Float>, mut fut: ArrayViewMut2<Float>| {
        assert_eq!(prev.shape(), fut.shape());
        let nx = prev.shape()[1];
        assert!(nx >= 2 * block.len());

        assert_eq!(prev.strides()[0], 1);
        assert_eq!(fut.strides()[0], 1);

        let dx = if optype == OperatorType::H2 {
            1.0 / (nx - 2) as Float
        } else {
            1.0 / (nx - 1) as Float
        };
        let idx = 1.0 / dx;

        fut.fill(0.0);

        // First block
        for (bl, mut fut) in block.iter().zip(fut.axis_iter_mut(ndarray::Axis(1))) {
            debug_assert_eq!(fut.len(), prev.shape()[0]);
            for (&bl, prev) in bl.iter().zip(prev.axis_iter(ndarray::Axis(1))) {
                debug_assert_eq!(prev.len(), fut.len());
                fut.scaled_add(idx * bl, &prev);
            }
        }

        let half_diag_width = (diag.len() - 1) / 2;
        assert!(half_diag_width <= block.len());

        // Diagonal entries
        for (ifut, mut fut) in fut
            .axis_iter_mut(ndarray::Axis(1))
            .enumerate()
            .skip(block.len())
            .take(nx - 2 * block.len())
        {
            for (id, &d) in diag.iter().enumerate() {
                let offset = ifut - half_diag_width + id;
                fut.scaled_add(idx * d, &prev.slice(ndarray::s![.., offset]))
            }
        }

        // End block
        for (bl, mut fut) in block.iter().zip(fut.axis_iter_mut(ndarray::Axis(1)).rev()) {
            fut.fill(0.0);
            for (&bl, prev) in bl.iter().zip(prev.axis_iter(ndarray::Axis(1)).rev()) {
                if symmetry == Symmetry::Symmetric {
                    fut.scaled_add(idx * bl, &prev);
                } else {
                    fut.scaled_add(-idx * bl, &prev);
                }
            }
        }
    }
}

#[inline(always)]
#[allow(unused)]
pub(crate) fn diff_op_col_naive_matrix<const M: usize, const N: usize, const D: usize>(
    block: &Matrix<Float, M, N>,
    blockend: &Matrix<Float, M, N>,
    diag: &RowVector<Float, D>,
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
    for (bl, mut fut) in block.iter_rows().zip(fut0.axis_iter_mut(ndarray::Axis(1))) {
        debug_assert_eq!(fut.len(), prev.shape()[0]);
        for (&bl, prev) in bl.iter().zip(prev.axis_iter(ndarray::Axis(1))) {
            if bl == 0.0 {
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
        for (&d, id) in diag.iter().zip(id.axis_iter(ndarray::Axis(1))) {
            if d == 0.0 {
                continue;
            }
            fut.scaled_add(idx * d, &id)
        }
    }

    // End block
    let prev = prev.slice(ndarray::s!(.., nx - N..));
    for (bl, mut fut) in blockend
        .iter_rows()
        .zip(futn.axis_iter_mut(ndarray::Axis(1)))
    {
        fut.fill(0.0);
        for (&bl, prev) in bl.iter().zip(prev.axis_iter(ndarray::Axis(1))) {
            if bl == 0.0 {
                continue;
            }
            fut.scaled_add(idx * bl, &prev);
        }
    }
}

#[inline(always)]
pub(crate) fn diff_op_col(
    block: &'static [&'static [Float]],
    diag: &'static [Float],
    symmetry: Symmetry,
    optype: OperatorType,
) -> impl Fn(ArrayView2<Float>, ArrayViewMut2<Float>) {
    diff_op_col_simd(block, diag, symmetry, optype)
}

#[inline(always)]
pub(crate) fn diff_op_col_simd(
    block: &'static [&'static [Float]],
    diag: &'static [Float],
    symmetry: Symmetry,
    optype: OperatorType,
) -> impl Fn(ArrayView2<Float>, ArrayViewMut2<Float>) {
    #[inline(always)]
    move |prev: ArrayView2<Float>, mut fut: ArrayViewMut2<Float>| {
        assert_eq!(prev.shape(), fut.shape());
        let nx = prev.shape()[1];
        assert!(nx >= 2 * block.len());

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

        let half_diag_width = (diag.len() - 1) / 2;
        assert!(half_diag_width <= block.len());

        let fut_base_ptr = fut.as_mut_ptr();
        let fut_stride = fut.strides()[1];
        let fut_ptr = |j, i| {
            debug_assert!(j < ny && i < nx);
            unsafe { fut_base_ptr.offset(fut_stride * i as isize + j as isize) }
        };

        let prev_base_ptr = prev.as_ptr();
        let prev_stride = prev.strides()[1];
        let prev_ptr = |j, i| {
            debug_assert!(j < ny && i < nx);
            unsafe { prev_base_ptr.offset(prev_stride * i as isize + j as isize) }
        };

        // Not algo necessary, but gives performance increase
        assert_eq!(fut_stride, prev_stride);

        // First block
        {
            for (ifut, &bl) in block.iter().enumerate() {
                for j in (0..simdified).step_by(SimdT::lanes()) {
                    let index_to_simd = |i| unsafe {
                        // j never moves past end of slice due to step_by and
                        // rounding down
                        SimdT::from_slice_unaligned(std::slice::from_raw_parts(
                            prev_ptr(j, i),
                            SimdT::lanes(),
                        ))
                    };
                    let mut f = SimdT::splat(0.0);
                    for (iprev, &bl) in bl.iter().enumerate() {
                        f = index_to_simd(iprev).mul_adde(SimdT::splat(bl), f);
                    }
                    f *= idx;

                    unsafe {
                        f.write_to_slice_unaligned(std::slice::from_raw_parts_mut(
                            fut_ptr(j, ifut),
                            SimdT::lanes(),
                        ));
                    }
                }
                for j in simdified..ny {
                    unsafe {
                        let mut f = 0.0;
                        for (iprev, bl) in bl.iter().enumerate() {
                            f += bl * *prev_ptr(j, iprev);
                        }
                        *fut_ptr(j, ifut) = f * idx;
                    }
                }
            }
        }

        // Diagonal elements
        {
            for ifut in block.len()..nx - block.len() {
                for j in (0..simdified).step_by(SimdT::lanes()) {
                    let index_to_simd = |i| unsafe {
                        // j never moves past end of slice due to step_by and
                        // rounding down
                        SimdT::from_slice_unaligned(std::slice::from_raw_parts(
                            prev_ptr(j, i),
                            SimdT::lanes(),
                        ))
                    };
                    let mut f = SimdT::splat(0.0);
                    for (id, &d) in diag.iter().enumerate() {
                        let offset = ifut - half_diag_width + id;
                        f = index_to_simd(offset).mul_adde(SimdT::splat(d), f);
                    }
                    f *= idx;
                    unsafe {
                        // puts simd along stride 1, j never goes past end of slice
                        f.write_to_slice_unaligned(std::slice::from_raw_parts_mut(
                            fut_ptr(j, ifut),
                            SimdT::lanes(),
                        ));
                    }
                }
                for j in simdified..ny {
                    let mut f = 0.0;
                    for (id, &d) in diag.iter().enumerate() {
                        let offset = ifut - half_diag_width + id;
                        unsafe {
                            f += d * *prev_ptr(j, offset);
                        }
                    }
                    unsafe {
                        *fut_ptr(j, ifut) = idx * f;
                    }
                }
            }
        }

        // End block
        {
            // Get blocks and corresponding offsets
            // (rev to iterate in ifut increasing order)
            for (bl, ifut) in block.iter().zip((0..nx).rev()) {
                for j in (0..simdified).step_by(SimdT::lanes()) {
                    let index_to_simd = |i| unsafe {
                        // j never moves past end of slice due to step_by and
                        // rounding down
                        SimdT::from_slice_unaligned(std::slice::from_raw_parts(
                            prev_ptr(j, i),
                            SimdT::lanes(),
                        ))
                    };
                    let mut f = SimdT::splat(0.0);
                    for (&bl, iprev) in bl.iter().zip((0..nx).rev()) {
                        f = index_to_simd(iprev).mul_adde(SimdT::splat(bl), f);
                    }
                    f = if symmetry == Symmetry::Symmetric {
                        f * idx
                    } else {
                        -f * idx
                    };
                    unsafe {
                        f.write_to_slice_unaligned(std::slice::from_raw_parts_mut(
                            fut_ptr(j, ifut),
                            SimdT::lanes(),
                        ));
                    }
                }

                for j in simdified..ny {
                    unsafe {
                        let mut f = 0.0;
                        for (&bl, iprev) in bl.iter().zip((0..nx).rev()).rev() {
                            f += bl * *prev_ptr(j, iprev);
                        }
                        *fut_ptr(j, ifut) = if symmetry == Symmetry::Symmetric {
                            f * idx
                        } else {
                            -f * idx
                        };
                    }
                }
            }
        }
    }
}

#[inline(always)]
fn dotproduct<'a>(u: impl Iterator<Item = &'a Float>, v: impl Iterator<Item = &'a Float>) -> Float {
    u.zip(v).fold(0.0, |acc, (&u, &v)| {
        #[cfg(feature = "fast-float")]
        {
            // We do not care about the order of multiplication nor addition
            (FastFloat::from(acc) + FastFloat::from(u) * FastFloat::from(v)).into()
        }
        #[cfg(not(feature = "fast-float"))]
        {
            acc + u * v
        }
    })
}

#[inline(always)]
pub(crate) fn diff_op_col_matrix<const M: usize, const N: usize, const D: usize>(
    block: &Matrix<Float, M, N>,
    block2: &Matrix<Float, M, N>,
    diag: &RowVector<Float, D>,
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
        for (mut fut, &bl) in fut1.axis_iter_mut(Axis(1)).zip(block.iter_rows()) {
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
                for (&d, prev) in diag.iter().zip(prev.axis_iter(Axis(1))) {
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
                for (&d, prev) in diag.iter().zip(prev) {
                    f += d * prev;
                }
                *fut = idx * f;
            }
        }
    }

    // End block
    {
        for (mut fut, &bl) in fut2.axis_iter_mut(Axis(1)).zip(block2.iter_rows()) {
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

#[inline(always)]
pub(crate) fn diff_op_row(
    block: &'static [&'static [Float]],
    diag: &'static [Float],
    symmetry: Symmetry,
    optype: OperatorType,
) -> impl Fn(ArrayView2<Float>, ArrayViewMut2<Float>) {
    #[inline(always)]
    move |prev: ArrayView2<Float>, mut fut: ArrayViewMut2<Float>| {
        assert_eq!(prev.shape(), fut.shape());
        let nx = prev.shape()[1];
        assert!(nx >= 2 * block.len());

        assert_eq!(prev.strides()[1], 1);
        assert_eq!(fut.strides()[1], 1);

        let dx = if optype == OperatorType::H2 {
            1.0 / (nx - 2) as Float
        } else {
            1.0 / (nx - 1) as Float
        };
        let idx = 1.0 / dx;

        for (prev, mut fut) in prev
            .axis_iter(ndarray::Axis(0))
            .zip(fut.axis_iter_mut(ndarray::Axis(0)))
        {
            let prev = prev.as_slice().unwrap();
            let fut = fut.as_slice_mut().unwrap();
            assert_eq!(prev.len(), fut.len());
            assert!(prev.len() >= 2 * block.len());

            for (bl, f) in block.iter().zip(fut.iter_mut()) {
                let diff = dotproduct(bl.iter(), prev[..bl.len()].iter());
                *f = diff * idx;
            }

            // The window needs to be aligned to the diagonal elements,
            // based on the block size
            let window_elems_to_skip = block.len() - ((diag.len() - 1) / 2);

            for (window, f) in prev
                .windows(diag.len())
                .skip(window_elems_to_skip)
                .zip(fut.iter_mut().skip(block.len()))
                .take(nx - 2 * block.len())
            {
                let diff = dotproduct(diag.iter(), window.iter());
                *f = diff * idx;
            }

            for (bl, f) in block.iter().zip(fut.iter_mut().rev()) {
                let diff = dotproduct(bl.iter(), prev.iter().rev());

                *f = idx
                    * if symmetry == Symmetry::Symmetric {
                        diff
                    } else {
                        -diff
                    };
            }
        }
    }
}

#[cfg(feature = "sparse")]
pub(crate) fn sparse_from_block(
    block: &[&[Float]],
    diag: &[Float],
    symmetry: Symmetry,
    optype: OperatorType,
    n: usize,
) -> sprs::CsMat<Float> {
    assert!(n >= 2 * block.len());

    let nnz = {
        let block_elems = block.iter().fold(0, |acc, x| {
            acc + x
                .iter()
                .fold(0, |acc, &x| if x != 0.0 { acc + 1 } else { acc })
        });

        let diag_elems = diag
            .iter()
            .fold(0, |acc, &x| if x != 0.0 { acc + 1 } else { acc });

        2 * block_elems + (n - 2 * block.len()) * diag_elems
    };

    let mut mat = sprs::TriMat::with_capacity((n, n), nnz);

    let dx = if optype == OperatorType::H2 {
        1.0 / (n - 2) as Float
    } else {
        1.0 / (n - 1) as Float
    };
    let idx = 1.0 / dx;

    for (j, bl) in block.iter().enumerate() {
        for (i, &b) in bl.iter().enumerate() {
            if b == 0.0 {
                continue;
            }
            mat.add_triplet(j, i, b * idx);
        }
    }

    for j in block.len()..n - block.len() {
        let half_diag_len = diag.len() / 2;
        for (&d, i) in diag.iter().zip(j - half_diag_len..) {
            if d == 0.0 {
                continue;
            }
            mat.add_triplet(j, i, d * idx);
        }
    }

    for (bl, j) in block.iter().zip((0..n).rev()).rev() {
        for (&b, i) in bl.iter().zip((0..n).rev()).rev() {
            if b == 0.0 {
                continue;
            }
            if symmetry == Symmetry::AntiSymmetric {
                mat.add_triplet(j, i, -b * idx);
            } else {
                mat.add_triplet(j, i, b * idx);
            }
        }
    }

    mat.to_csr()
}

#[cfg(feature = "sparse")]
pub(crate) fn h_matrix(diag: &[Float], n: usize, is_h2: bool) -> sprs::CsMat<Float> {
    let h = if is_h2 {
        1.0 / (n - 2) as Float
    } else {
        1.0 / (n - 1) as Float
    };
    let nmiddle = n - 2 * diag.len();
    let iter = diag
        .iter()
        .chain(std::iter::repeat(&1.0).take(nmiddle))
        .chain(diag.iter().rev())
        .map(|&x| h * x);

    let mut mat = sprs::TriMat::with_capacity((n, n), n);
    for (i, d) in iter.enumerate() {
        mat.add_triplet(i, i, d);
    }
    mat.to_csr()
}
