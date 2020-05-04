#![allow(clippy::excessive_precision)]
#![allow(clippy::unreadable_literal)]

use crate::Float;
use ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2};

pub trait SbpOperator1d: Send + Sync {
    fn diff(&self, prev: ArrayView1<Float>, fut: ArrayViewMut1<Float>);

    fn h(&self) -> &'static [Float];
    fn is_h2(&self) -> bool {
        false
    }
}

pub trait SbpOperator2d: Send + Sync {
    fn diffxi(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>);
    fn diffeta(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>);

    fn hxi(&self) -> &'static [Float];
    fn heta(&self) -> &'static [Float];

    fn is_h2xi(&self) -> bool;
    fn is_h2eta(&self) -> bool;
}

impl<SBPeta: SbpOperator1d, SBPxi: SbpOperator1d> SbpOperator2d for (&SBPeta, &SBPxi) {
    default fn diffxi(&self, prev: ArrayView2<Float>, mut fut: ArrayViewMut2<Float>) {
        assert_eq!(prev.shape(), fut.shape());
        for (r0, r1) in prev.outer_iter().zip(fut.outer_iter_mut()) {
            self.1.diff(r0, r1)
        }
    }
    fn diffeta(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>) {
        let ba = (self.1, self.0);
        ba.diffxi(prev.reversed_axes(), fut.reversed_axes())
    }
    fn hxi(&self) -> &'static [Float] {
        self.1.h()
    }
    fn heta(&self) -> &'static [Float] {
        self.0.h()
    }
    fn is_h2xi(&self) -> bool {
        self.1.is_h2()
    }
    fn is_h2eta(&self) -> bool {
        self.0.is_h2()
    }
}

impl<SBP: SbpOperator1d + Copy> SbpOperator2d for SBP {
    fn diffxi(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>) {
        <(&SBP, &SBP) as SbpOperator2d>::diffxi(&(self, self), prev, fut)
    }
    fn diffeta(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>) {
        <(&SBP, &SBP) as SbpOperator2d>::diffeta(&(self, self), prev, fut)
    }
    fn hxi(&self) -> &'static [Float] {
        <(&SBP, &SBP) as SbpOperator2d>::hxi(&(self, self))
    }
    fn heta(&self) -> &'static [Float] {
        <(&SBP, &SBP) as SbpOperator2d>::heta(&(self, self))
    }
    fn is_h2xi(&self) -> bool {
        <(&SBP, &SBP) as SbpOperator2d>::is_h2xi(&(self, self))
    }
    fn is_h2eta(&self) -> bool {
        <(&SBP, &SBP) as SbpOperator2d>::is_h2eta(&(self, self))
    }
}

pub trait UpwindOperator1d: SbpOperator1d + Send + Sync {
    fn diss(&self, prev: ArrayView1<Float>, fut: ArrayViewMut1<Float>);
    fn as_sbp(&self) -> &dyn SbpOperator1d;
}

pub trait UpwindOperator2d: SbpOperator2d + Send + Sync {
    fn dissxi(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>);
    fn disseta(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>);
    fn as_sbp(&self) -> &dyn SbpOperator2d;
}

impl<UOeta: UpwindOperator1d, UOxi: UpwindOperator1d> UpwindOperator2d for (&UOeta, &UOxi) {
    default fn dissxi(&self, prev: ArrayView2<Float>, mut fut: ArrayViewMut2<Float>) {
        assert_eq!(prev.shape(), fut.shape());
        for (r0, r1) in prev.outer_iter().zip(fut.outer_iter_mut()) {
            self.1.diss(r0, r1);
        }
    }
    fn disseta(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>) {
        let ba = (self.1, self.0);
        ba.dissxi(prev.reversed_axes(), fut.reversed_axes())
    }
    fn as_sbp(&self) -> &dyn SbpOperator2d {
        self
    }
}

impl<UO: UpwindOperator1d + Copy> UpwindOperator2d for UO {
    fn dissxi(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>) {
        <(&UO, &UO) as UpwindOperator2d>::dissxi(&(self, self), prev, fut)
    }
    fn disseta(&self, prev: ArrayView2<Float>, fut: ArrayViewMut2<Float>) {
        <(&UO, &UO) as UpwindOperator2d>::disseta(&(self, self), prev, fut)
    }
    fn as_sbp(&self) -> &dyn SbpOperator2d {
        self
    }
}

pub trait InterpolationOperator: Send + Sync {
    fn fine2coarse(&self, fine: ArrayView1<Float>, coarse: ArrayViewMut1<Float>);
    fn coarse2fine(&self, coarse: ArrayView1<Float>, fine: ArrayViewMut1<Float>);
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
                    f = f * idx;

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
                    f = f * idx;
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
/// Manual unrolling for some common sizes for vectorisation
///
/// best used when the lengths of u and v are known at
/// compile time
fn product(u: &[Float], v: &[Float]) -> Float {
    use packed_simd::*;
    #[cfg(feature = "f32")]
    type SimdT = f32x2;
    #[cfg(not(feature = "f32"))]
    type SimdT = f64x2;

    assert_eq!(u.len(), v.len());
    match u.len() {
        0 => 0.0,
        1 => u[0] * v[0],
        2 => (SimdT::from_slice_unaligned(&u[0..]) * SimdT::from_slice_unaligned(&v[0..])).sum(),
        3 => {
            (SimdT::from_slice_unaligned(&u[0..]) * SimdT::from_slice_unaligned(&v[0..])).sum()
                + u[2] * v[2]
        }
        4 => {
            let u0 = SimdT::from_slice_unaligned(&u[0..]);
            let v0 = SimdT::from_slice_unaligned(&v[0..]);

            let u1 = SimdT::from_slice_unaligned(&u[2..]);
            let v1 = SimdT::from_slice_unaligned(&v[2..]);

            let p = u0 * v0;
            let p = u1.mul_adde(v1, p);
            p.sum()
        }
        5 => {
            let u0 = SimdT::from_slice_unaligned(&u[0..]);
            let v0 = SimdT::from_slice_unaligned(&v[0..]);

            let u1 = SimdT::from_slice_unaligned(&u[2..]);
            let v1 = SimdT::from_slice_unaligned(&v[2..]);

            let p = u0 * v0;
            let p = u1.mul_adde(v1, p);
            p.sum() + u[4] * v[4]
        }
        6 => {
            let u0 = SimdT::from_slice_unaligned(&u[0..]);
            let v0 = SimdT::from_slice_unaligned(&v[0..]);

            let u1 = SimdT::from_slice_unaligned(&u[2..]);
            let v1 = SimdT::from_slice_unaligned(&v[2..]);

            let u2 = SimdT::from_slice_unaligned(&u[4..]);
            let v2 = SimdT::from_slice_unaligned(&v[4..]);

            let p = u0 * v0;
            let p = u1.mul_adde(v1, p);
            let p = u2.mul_adde(v2, p);
            p.sum()
        }
        7 => {
            let u0 = SimdT::from_slice_unaligned(&u[0..]);
            let v0 = SimdT::from_slice_unaligned(&v[0..]);

            let u1 = SimdT::from_slice_unaligned(&u[2..]);
            let v1 = SimdT::from_slice_unaligned(&v[2..]);

            let u2 = SimdT::from_slice_unaligned(&u[4..]);
            let v2 = SimdT::from_slice_unaligned(&v[4..]);

            let p = u0 * v0;
            let p = u1.mul_adde(v1, p);
            let p = u2.mul_adde(v2, p);
            p.sum() + u[6] * v[6]
        }
        8 => {
            let u0 = SimdT::from_slice_unaligned(&u[0..]);
            let v0 = SimdT::from_slice_unaligned(&v[0..]);

            let u1 = SimdT::from_slice_unaligned(&u[2..]);
            let v1 = SimdT::from_slice_unaligned(&v[2..]);

            let u2 = SimdT::from_slice_unaligned(&u[4..]);
            let v2 = SimdT::from_slice_unaligned(&v[4..]);

            let u3 = SimdT::from_slice_unaligned(&u[6..]);
            let v3 = SimdT::from_slice_unaligned(&v[6..]);

            let p = u0 * v0;
            let p = u1.mul_adde(v1, p);
            let p = u2.mul_adde(v2, p);
            let p = u3.mul_adde(v3, p);
            p.sum()
        }
        9 => {
            let u0 = SimdT::from_slice_unaligned(&u[0..]);
            let v0 = SimdT::from_slice_unaligned(&v[0..]);

            let u1 = SimdT::from_slice_unaligned(&u[2..]);
            let v1 = SimdT::from_slice_unaligned(&v[2..]);

            let u2 = SimdT::from_slice_unaligned(&u[4..]);
            let v2 = SimdT::from_slice_unaligned(&v[4..]);

            let u3 = SimdT::from_slice_unaligned(&u[6..]);
            let v3 = SimdT::from_slice_unaligned(&v[6..]);

            let p = u0 * v0;
            let p = u1.mul_adde(v1, p);
            let p = u2.mul_adde(v2, p);
            let p = u3.mul_adde(v3, p);
            p.sum() + u[8] * v[8]
        }
        10 => {
            let u0 = SimdT::from_slice_unaligned(&u[0..]);
            let v0 = SimdT::from_slice_unaligned(&v[0..]);

            let u1 = SimdT::from_slice_unaligned(&u[2..]);
            let v1 = SimdT::from_slice_unaligned(&v[2..]);

            let u2 = SimdT::from_slice_unaligned(&u[4..]);
            let v2 = SimdT::from_slice_unaligned(&v[4..]);

            let u3 = SimdT::from_slice_unaligned(&u[6..]);
            let v3 = SimdT::from_slice_unaligned(&v[6..]);

            let u4 = SimdT::from_slice_unaligned(&u[8..]);
            let v4 = SimdT::from_slice_unaligned(&v[8..]);

            let p = u0 * v0;
            let p = u1.mul_adde(v1, p);
            let p = u2.mul_adde(v2, p);
            let p = u3.mul_adde(v3, p);
            let p = u4.mul_adde(v4, p);
            p.sum()
        }
        11 => {
            let u0 = SimdT::from_slice_unaligned(&u[0..]);
            let v0 = SimdT::from_slice_unaligned(&v[0..]);

            let u1 = SimdT::from_slice_unaligned(&u[2..]);
            let v1 = SimdT::from_slice_unaligned(&v[2..]);

            let u2 = SimdT::from_slice_unaligned(&u[4..]);
            let v2 = SimdT::from_slice_unaligned(&v[4..]);

            let u3 = SimdT::from_slice_unaligned(&u[6..]);
            let v3 = SimdT::from_slice_unaligned(&v[6..]);

            let u4 = SimdT::from_slice_unaligned(&u[8..]);
            let v4 = SimdT::from_slice_unaligned(&v[8..]);

            let p = u0 * v0;
            let p = u1.mul_adde(v1, p);
            let p = u2.mul_adde(v2, p);
            let p = u3.mul_adde(v3, p);
            let p = u4.mul_adde(v4, p);
            p.sum() + u[10] * v[10]
        }
        12 => {
            let u0 = SimdT::from_slice_unaligned(&u[0..]);
            let v0 = SimdT::from_slice_unaligned(&v[0..]);

            let u1 = SimdT::from_slice_unaligned(&u[2..]);
            let v1 = SimdT::from_slice_unaligned(&v[2..]);

            let u2 = SimdT::from_slice_unaligned(&u[4..]);
            let v2 = SimdT::from_slice_unaligned(&v[4..]);

            let u3 = SimdT::from_slice_unaligned(&u[6..]);
            let v3 = SimdT::from_slice_unaligned(&v[6..]);

            let u4 = SimdT::from_slice_unaligned(&u[8..]);
            let v4 = SimdT::from_slice_unaligned(&v[8..]);

            let u5 = SimdT::from_slice_unaligned(&u[10..]);
            let v5 = SimdT::from_slice_unaligned(&v[10..]);

            let p = u0 * v0;
            let p = u1.mul_adde(v1, p);
            let p = u2.mul_adde(v2, p);
            let p = u3.mul_adde(v3, p);
            let p = u4.mul_adde(v4, p);
            let p = u5.mul_adde(v5, p);
            p.sum()
        }
        13 => {
            let u0 = SimdT::from_slice_unaligned(&u[0..]);
            let v0 = SimdT::from_slice_unaligned(&v[0..]);

            let u1 = SimdT::from_slice_unaligned(&u[2..]);
            let v1 = SimdT::from_slice_unaligned(&v[2..]);

            let u2 = SimdT::from_slice_unaligned(&u[4..]);
            let v2 = SimdT::from_slice_unaligned(&v[4..]);

            let u3 = SimdT::from_slice_unaligned(&u[6..]);
            let v3 = SimdT::from_slice_unaligned(&v[6..]);

            let u4 = SimdT::from_slice_unaligned(&u[8..]);
            let v4 = SimdT::from_slice_unaligned(&v[8..]);

            let u5 = SimdT::from_slice_unaligned(&u[10..]);
            let v5 = SimdT::from_slice_unaligned(&v[10..]);

            let p = u0 * v0;
            let p = u1.mul_adde(v1, p);
            let p = u2.mul_adde(v2, p);
            let p = u3.mul_adde(v3, p);
            let p = u4.mul_adde(v4, p);
            let p = u5.mul_adde(v5, p);
            p.sum() + u[12] * v[12]
        }
        _ => u.iter().zip(v.iter()).map(|(u, v)| u * v).sum::<Float>(),
    }
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
                let diff = product(bl, &prev[..bl.len()]);
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
                let diff = product(diag, window);
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
    }
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
