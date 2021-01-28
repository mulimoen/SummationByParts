use super::*;

pub(crate) mod constmatrix {
    /// A row-major matrix
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct Matrix<T, const M: usize, const N: usize> {
        data: [[T; N]; M],
    }

    impl<T: Default, const M: usize, const N: usize> Default for Matrix<T, M, N> {
        fn default() -> Self {
            use std::mem::MaybeUninit;
            let mut d: [[MaybeUninit<T>; N]; M] = unsafe { MaybeUninit::uninit().assume_init() };

            for row in d.iter_mut() {
                for item in row.iter_mut() {
                    *item = MaybeUninit::new(T::default());
                }
            }

            let data = unsafe { std::mem::transmute_copy::<_, [[T; N]; M]>(&d) };
            Self { data }
        }
    }

    impl<T, const M: usize, const N: usize> core::ops::Index<(usize, usize)> for Matrix<T, M, N> {
        type Output = T;
        #[inline(always)]
        fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
            &self.data[i][j]
        }
    }

    impl<T, const M: usize, const N: usize> core::ops::IndexMut<(usize, usize)> for Matrix<T, M, N> {
        #[inline(always)]
        fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
            &mut self.data[i][j]
        }
    }

    impl<T, const M: usize, const N: usize> core::ops::Index<usize> for Matrix<T, M, N> {
        type Output = [T; N];
        #[inline(always)]
        fn index(&self, i: usize) -> &Self::Output {
            &self.data[i]
        }
    }

    impl<T, const M: usize, const N: usize> core::ops::IndexMut<usize> for Matrix<T, M, N> {
        #[inline(always)]
        fn index_mut(&mut self, i: usize) -> &mut Self::Output {
            &mut self.data[i]
        }
    }

    impl<T, const M: usize, const N: usize> Matrix<T, M, N> {
        pub const fn new(data: [[T; N]; M]) -> Self {
            Self { data }
        }
        pub const fn nrows(&self) -> usize {
            M
        }
        pub const fn ncols(&self) -> usize {
            N
        }
        pub fn matmul<const P: usize>(&self, other: &Matrix<T, N, P>) -> Matrix<T, M, P>
        where
            T: Default + core::ops::AddAssign<T>,
            for<'f> &'f T: std::ops::Mul<Output = T>,
        {
            let mut out = Matrix::default();
            self.matmul_into(other, &mut out);
            out
        }
        pub fn matmul_into<const P: usize>(
            &self,
            other: &Matrix<T, N, P>,
            out: &mut Matrix<T, M, P>,
        ) where
            T: Default + core::ops::AddAssign<T>,
            for<'f> &'f T: std::ops::Mul<Output = T>,
        {
            *out = Default::default();
            for i in 0..M {
                for j in 0..P {
                    for k in 0..N {
                        out[(i, j)] += &self[(i, k)] * &other[(k, j)];
                    }
                }
            }
        }
        pub fn iter_rows(
            &self,
        ) -> impl ExactSizeIterator<Item = &[T; N]> + DoubleEndedIterator<Item = &[T; N]> {
            (0..M).map(move |i| &self[i])
        }
    }

    #[cfg(test)]
    mod tests {
        use super::{super::*, *};
        #[test]
        fn construct_copy_type() {
            let _m0 = Matrix::<i32, 4, 3>::default();
            let _m1: Matrix<u8, 8, 8> = Matrix::default();

            let _m2 = Matrix::new([[1, 2], [3, 4]]);
        }
        #[test]
        fn construct_non_copy() {
            let _m = Matrix::<String, 2, 1>::default();
        }

        #[test]
        fn matmul() {
            let m1 = Matrix::new([[1_u8, 2, 3], [4, 5, 6]]);
            let m2 = Matrix::new([[7_u8, 8, 9, 10], [11, 12, 13, 14], [15, 16, 17, 18]]);

            let m3 = m1.matmul(&m2);
            assert_eq!(m3, Matrix::new([[74, 80, 86, 92], [173, 188, 203, 218]]));
        }
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
fn product_fast<'a>(
    u: impl Iterator<Item = &'a Float>,
    v: impl Iterator<Item = &'a Float>,
) -> Float {
    use std::intrinsics::{fadd_fast, fmul_fast};
    u.zip(v).fold(0.0, |acc, (&u, &v)| unsafe {
        // We do not care about the order of multiplication nor addition
        fadd_fast(acc, fmul_fast(u, v))
    })
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
                let diff = product_fast(bl.iter(), prev[..bl.len()].iter());
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
                let diff = product_fast(diag.iter(), window.iter());
                *f = diff * idx;
            }

            for (bl, f) in block.iter().zip(fut.iter_mut().rev()) {
                let diff = product_fast(bl.iter(), prev.iter().rev());

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
