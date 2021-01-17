#![allow(clippy::excessive_precision)]
#![allow(clippy::unreadable_literal)]

use crate::Float;
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};

/// One-dimensional Summation By Parts operator
pub trait SbpOperator1d: Send + Sync {
    /// Differentiate on unit grid
    fn diff(&self, prev: ArrayView1<Float>, fut: ArrayViewMut1<Float>);

    /// block component of `H`, with diagonal component 1
    fn h(&self) -> &'static [Float];
    /// Whether the operator acts on a `h2` spaced computational grid
    fn is_h2(&self) -> bool {
        false
    }
    #[cfg(feature = "sparse")]
    fn diff_matrix(&self, n: usize) -> sprs::CsMat<Float>;
    #[cfg(feature = "sparse")]
    fn h_matrix(&self, n: usize) -> sprs::CsMat<Float>;

    fn upwind(&self) -> Option<&dyn UpwindOperator1d> {
        None
    }
    fn d2(&self) -> Option<&dyn SbpOperator1d2> {
        None
    }
}

pub trait SbpOperator1d2: SbpOperator1d {
    /// Double differentiation on unit grid
    fn diff2(&self, prev: ArrayView1<Float>, fut: ArrayViewMut1<Float>);
    /// Result of H^-1 * d1, without the 1/h scaling
    fn d1(&self) -> &[Float];

    #[cfg(feature = "sparse")]
    fn diff2_matrix(&self, n: usize) -> sprs::CsMat<Float>;
    #[cfg(feature = "sparse")]
    fn d1_vec(&self, n: usize, front: bool) -> sprs::CsMat<Float>;
}

pub trait UpwindOperator1d: SbpOperator1d + Send + Sync {
    /// Dissipation operator
    fn diss(&self, prev: ArrayView1<Float>, fut: ArrayViewMut1<Float>);
    fn as_sbp(&self) -> &dyn SbpOperator1d;

    #[cfg(feature = "sparse")]
    fn diss_matrix(&self, n: usize) -> sprs::CsMat<Float>;
}

pub trait SbpOperator2d: Send + Sync {
    fn diffxi(&self, prev: ArrayView2<Float>, mut fut: ArrayViewMut2<Float>) {
        assert_eq!(prev.shape(), fut.shape());
        for (p, f) in prev.outer_iter().zip(fut.outer_iter_mut()) {
            self.op_xi().diff(p, f)
        }
    }
    fn diffeta(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>) {
        self.diffxi(prev.reversed_axes(), fut.reversed_axes())
    }

    fn hxi(&self) -> &'static [Float] {
        self.op_xi().h()
    }
    fn heta(&self) -> &'static [Float] {
        self.op_eta().h()
    }

    fn is_h2xi(&self) -> bool {
        self.op_xi().is_h2()
    }
    fn is_h2eta(&self) -> bool {
        self.op_eta().is_h2()
    }

    fn op_xi(&self) -> &dyn SbpOperator1d;
    fn op_eta(&self) -> &dyn SbpOperator1d;

    fn upwind(&self) -> Option<Box<dyn UpwindOperator2d>> {
        None
    }
}

pub trait UpwindOperator2d: Send + Sync {
    fn dissxi(&self, prev: ArrayView2<Float>, mut fut: ArrayViewMut2<Float>) {
        assert_eq!(prev.shape(), fut.shape());
        for (p, f) in prev.outer_iter().zip(fut.outer_iter_mut()) {
            UpwindOperator2d::op_xi(self).diss(p, f)
        }
    }
    // Assuming operator is symmetrical x/y
    fn disseta(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>) {
        self.dissxi(prev.reversed_axes(), fut.reversed_axes())
    }

    fn op_xi(&self) -> &dyn UpwindOperator1d;
    fn op_eta(&self) -> &dyn UpwindOperator1d;
}

pub trait InterpolationOperator: Send + Sync {
    /// Interpolation from a grid with twice resolution
    fn fine2coarse(&self, fine: ArrayView1<Float>, coarse: ArrayViewMut1<Float>);
    /// Interpolation from a grid with half resolution
    fn coarse2fine(&self, coarse: ArrayView1<Float>, fine: ArrayViewMut1<Float>);
}

impl SbpOperator2d for (Box<dyn SbpOperator2d>, Box<dyn SbpOperator2d>) {
    fn diffxi(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>) {
        self.1.diffxi(prev, fut)
    }
    fn diffeta(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>) {
        self.0.diffeta(prev, fut)
    }

    fn op_xi(&self) -> &dyn SbpOperator1d {
        self.1.op_xi()
    }
    fn op_eta(&self) -> &dyn SbpOperator1d {
        self.0.op_eta()
    }
    fn upwind(&self) -> Option<Box<dyn UpwindOperator2d>> {
        match (self.0.upwind(), self.1.upwind()) {
            (Some(u), Some(v)) => Some(Box::new((u, v))),
            _ => None,
        }
    }
}

impl UpwindOperator2d for (Box<dyn UpwindOperator2d>, Box<dyn UpwindOperator2d>) {
    fn dissxi(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>) {
        self.1.dissxi(prev, fut)
    }
    fn disseta(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>) {
        self.0.disseta(prev, fut)
    }
    fn op_xi(&self) -> &dyn UpwindOperator1d {
        self.1.op_xi()
    }
    fn op_eta(&self) -> &dyn UpwindOperator1d {
        self.0.op_eta()
    }
}

#[inline(always)]
fn diff_op_1d(
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
enum Symmetry {
    Symmetric,
    AntiSymmetric,
}

#[derive(PartialEq, Copy, Clone)]
enum OperatorType {
    Normal,
    H2,
}

#[inline(always)]
#[allow(unused)]
fn diff_op_col_naive(
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
fn diff_op_col(
    block: &'static [&'static [Float]],
    diag: &'static [Float],
    symmetry: Symmetry,
    optype: OperatorType,
) -> impl Fn(ArrayView2<Float>, ArrayViewMut2<Float>) {
    diff_op_col_simd(block, diag, symmetry, optype)
}

#[inline(always)]
fn diff_op_col_simd(
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
fn diff_op_row(
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
fn sparse_from_block(
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
fn h_matrix(diag: &[Float], n: usize, is_h2: bool) -> sprs::CsMat<Float> {
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

mod upwind4;
pub use upwind4::Upwind4;
mod upwind9;
pub use upwind9::Upwind9;

mod upwind4h2;
pub use upwind4h2::Upwind4h2;
mod upwind9h2;
pub use upwind9h2::Upwind9h2;

mod traditional4;
pub use traditional4::SBP4;
mod traditional8;
pub use traditional8::SBP8;

mod interpolation;
pub use interpolation::{Interpolation4, Interpolation8, Interpolation9, Interpolation9h2};

#[cfg(test)]
pub(crate) mod testing {
    use super::*;
    use ndarray::prelude::*;
    pub(crate) fn grid_eval<F: Fn(Float, Float) -> Float>(
        n: (usize, usize),
        f: F,
    ) -> Array2<Float> {
        let nx = n.1;
        let dx = 1.0 / (nx - 1) as Float;
        let ny = n.0;
        let dy = 1.0 / (ny - 1) as Float;
        Array2::from_shape_fn(n, |(j, i)| {
            let x = dx * i as Float;
            let y = dy * j as Float;
            f(x, y)
        })
    }

    pub(crate) fn check_operator_on<SBP, F, FX, FY>(
        op: SBP,
        n: (usize, usize),
        f: F,
        dfdx: FX,
        dfdy: FY,
        eps: Float,
    ) where
        SBP: SbpOperator2d,
        F: Fn(Float, Float) -> Float,
        FX: Fn(Float, Float) -> Float,
        FY: Fn(Float, Float) -> Float,
    {
        let mut y = Array2::zeros(n);
        let x = grid_eval(n, f);

        y.fill(0.0);
        op.diffxi(x.view(), y.view_mut());
        approx::assert_abs_diff_eq!(&y, &grid_eval(n, dfdx), epsilon = eps);

        y.fill(0.0);
        op.diffeta(x.view(), y.view_mut());
        approx::assert_abs_diff_eq!(&y, &grid_eval(n, dfdy), epsilon = eps);
    }
}
