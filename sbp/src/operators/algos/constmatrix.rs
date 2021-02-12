#![allow(unused)]

use num_traits::identities::Zero;

/// A row-major matrix
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Matrix<T, const M: usize, const N: usize> {
    pub data: [[T; N]; M],
}
pub type RowVector<T, const N: usize> = Matrix<T, 1, N>;
pub type ColVector<T, const N: usize> = Matrix<T, N, 1>;

impl<T: Copy + Default, const M: usize, const N: usize> Default for Matrix<T, M, N> {
    fn default() -> Self {
        Self {
            data: [[T::default(); N]; M],
        }
    }
}

impl<T: Copy + Zero + PartialEq, const M: usize, const N: usize> Zero for Matrix<T, M, N> {
    fn zero() -> Self {
        Self {
            data: [[T::zero(); N]; M],
        }
    }
    fn is_zero(&self) -> bool {
        self == &Self::zero()
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
    #[inline(always)]
    pub const fn nrows(&self) -> usize {
        M
    }
    #[inline(always)]
    pub const fn ncols(&self) -> usize {
        N
    }
    #[inline(always)]
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.data.iter().flatten()
    }
    #[inline(always)]
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.data.iter_mut().flatten()
    }
    #[inline(always)]
    pub fn iter_rows(
        &self,
    ) -> impl ExactSizeIterator<Item = &[T; N]> + DoubleEndedIterator<Item = &[T; N]> {
        self.data.iter()
    }
    #[inline(always)]
    pub const fn row(&self, idx: usize) -> &[T; N] {
        &self.data[idx]
    }
}

impl<T, const N: usize> ColVector<T, N> {
    #[inline(always)]
    pub fn map_to_col(slice: &[T; N]) -> &ColVector<T, N> {
        unsafe { &*(slice.as_ptr().cast()) }
    }
    #[inline(always)]
    pub fn map_to_col_mut(slice: &mut [T; N]) -> &mut ColVector<T, N> {
        unsafe { &mut *(slice.as_mut_ptr().cast()) }
    }
}

impl<T, const N: usize> RowVector<T, N> {
    pub fn map_to_row(slice: &[T; N]) -> &Self {
        unsafe { &*(slice.as_ptr().cast()) }
    }
    pub fn map_to_row_mut(slice: &mut [T; N]) -> &mut Self {
        unsafe { &mut *(slice.as_mut_ptr().cast()) }
    }
}

impl<T, const M: usize, const P: usize> Matrix<T, M, P> {
    #[inline(always)]
    pub fn matmul_into<const N: usize>(&mut self, lhs: &Matrix<T, M, N>, rhs: &Matrix<T, N, P>)
    where
        T: Default + Copy + core::ops::Mul<Output = T> + core::ops::Add<Output = T>,
    {
        for i in 0..M {
            for j in 0..P {
                let mut t = T::default();
                for k in 0..N {
                    t = t + lhs[(i, k)] * rhs[(k, j)];
                }
                self[(i, j)] = t;
            }
        }
    }
}

macro_rules! impl_op_mul_mul {
    ($lhs:ty, $rhs:ty) => {
        impl<T, const N: usize, const M: usize, const P: usize> core::ops::Mul<$rhs> for $lhs
        where
            T: Copy + Default + core::ops::Add<Output = T> + core::ops::Mul<Output = T>,
        {
            type Output = Matrix<T, M, P>;
            fn mul(self, lhs: $rhs) -> Self::Output {
                let mut out = Matrix::default();
                out.matmul_into(&self, &lhs);
                out
            }
        }
    };
}

impl_op_mul_mul! {  Matrix<T, M, N>,  Matrix<T, N, P> }
impl_op_mul_mul! { &Matrix<T, M, N>,  Matrix<T, N, P> }
impl_op_mul_mul! {  Matrix<T, M, N>, &Matrix<T, N, P> }
impl_op_mul_mul! { &Matrix<T, M, N>, &Matrix<T, N, P> }

impl<T, const M: usize, const N: usize> core::ops::MulAssign<T> for Matrix<T, M, N>
where
    T: Copy + core::ops::MulAssign<T>,
{
    #[inline(always)]
    fn mul_assign(&mut self, other: T) {
        self.iter_mut().for_each(|x| *x *= other)
    }
}

impl<T, const N: usize, const M: usize> core::ops::Add<Matrix<T, M, N>> for Matrix<T, M, N>
where
    T: Copy + Zero + core::ops::Add<Output = T> + PartialEq,
{
    type Output = Self;
    fn add(self, lhs: Self) -> Self::Output {
        let mut out = Matrix::zero();
        for i in 0..M {
            for j in 0..N {
                out[(i, j)] = self[(i, j)] + lhs[(i, j)];
            }
        }
        out
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
    fn matmul() {
        let m1 = Matrix::new([[1_u8, 2, 3], [4, 5, 6]]);
        let m2 = Matrix::new([[7_u8, 8, 9, 10], [11, 12, 13, 14], [15, 16, 17, 18]]);

        let m3 = m1 * m2;
        assert_eq!(m3, Matrix::new([[74, 80, 86, 92], [173, 188, 203, 218]]));
    }
    #[test]
    fn iter() {
        let m = Matrix::new([[1_u8, 2, 3], [4, 5, 6]]);
        let mut iter = m.iter();
        assert_eq!(iter.next(), Some(&1));
        assert_eq!(iter.next(), Some(&2));
        assert_eq!(iter.next(), Some(&3));
        assert_eq!(iter.next(), Some(&4));
        assert_eq!(iter.next(), Some(&5));
        assert_eq!(iter.next(), Some(&6));
        assert_eq!(iter.next(), None);
    }
}

mod approx {
    use super::Matrix;
    use ::approx::{AbsDiffEq, RelativeEq, UlpsEq};

    impl<T, const M: usize, const N: usize> AbsDiffEq for Matrix<T, M, N>
    where
        T: AbsDiffEq,
    {
        type Epsilon = T::Epsilon;
        fn default_epsilon() -> Self::Epsilon {
            T::default_epsilon()
        }
        fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
            self.iter()
                .zip(other.iter())
                .all(|(r, l)| r.abs_diff_eq(l, T::default_epsilon()))
        }
    }
    impl<T, const M: usize, const N: usize> RelativeEq for Matrix<T, M, N>
    where
        T: RelativeEq,
        Self::Epsilon: Copy,
    {
        fn default_max_relative() -> Self::Epsilon {
            T::default_max_relative()
        }
        fn relative_eq(
            &self,
            other: &Self,
            epsilon: Self::Epsilon,
            max_relative: Self::Epsilon,
        ) -> bool {
            self.iter()
                .zip(other.iter())
                .all(|(r, l)| r.relative_eq(l, epsilon, max_relative))
        }
    }
    impl<T, const M: usize, const N: usize> UlpsEq for Matrix<T, M, N>
    where
        T: UlpsEq,
        Self::Epsilon: Copy,
    {
        fn default_max_ulps() -> u32 {
            T::default_max_ulps()
        }
        fn ulps_eq(&self, other: &Self, epsilon: Self::Epsilon, max_ulps: u32) -> bool {
            self.iter()
                .zip(other.iter())
                .all(|(r, l)| r.ulps_eq(l, epsilon, max_ulps))
        }
    }
}

pub(crate) const fn flip_ud<const M: usize, const N: usize>(
    mut m: Matrix<super::Float, M, N>,
) -> Matrix<super::Float, M, N> {
    let mut i = 0;
    while i < M / 2 {
        let tmp = m.data[i];
        m.data[i] = m.data[M - 1 - i];
        m.data[M - 1 - i] = tmp;
        i += 1;
    }
    m
}

pub(crate) const fn flip_lr<const M: usize, const N: usize>(
    mut m: Matrix<super::Float, M, N>,
) -> Matrix<super::Float, M, N> {
    let mut i = 0;
    while i < M {
        let mut j = 0;
        while j < N / 2 {
            let tmp = m.data[i][j];
            m.data[i][j] = m.data[i][N - 1 - j];
            m.data[i][N - 1 - j] = tmp;
            j += 1;
        }
        i += 1;
    }
    m
}

/// Flip all sign bits
pub(crate) const fn flip_sign<const M: usize, const N: usize>(
    mut m: Matrix<super::Float, M, N>,
) -> Matrix<super::Float, M, N> {
    let mut i = 0;
    while i < M {
        let mut j = 0;
        while j < N {
            m.data[i][j] = -m.data[i][j];
            j += 1;
        }
        i += 1;
    }
    m
}
mod flipping {
    use super::*;

    #[test]
    fn flip_lr_test() {
        let m = Matrix::new([[1.0, 2.0, 3.0, 4.0]]);
        let flipped = flip_lr(m);
        assert_eq!(flipped, Matrix::new([[4.0, 3.0, 2.0, 1.0]]));
        let m = Matrix::new([[1.0, 2.0, 3.0, 4.0, 5.0]]);
        let flipped = flip_lr(m);
        assert_eq!(flipped, Matrix::new([[5.0, 4.0, 3.0, 2.0, 1.0]]));
    }
    #[test]
    fn flip_ud_test() {
        let m = Matrix::new([[1.0], [2.0], [3.0], [4.0]]);
        let flipped = flip_ud(m);
        assert_eq!(flipped, Matrix::new([[4.0], [3.0], [2.0], [1.0]]));
        let m = Matrix::new([[1.0], [2.0], [3.0], [4.0], [5.0]]);
        let flipped = flip_ud(m);
        assert_eq!(flipped, Matrix::new([[5.0], [4.0], [3.0], [2.0], [1.0]]));
    }
}
