use super::*;

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub(crate) struct FastFloat(Float);

use core::intrinsics::{fadd_fast, fmul_fast};

impl core::ops::Mul for FastFloat {
    type Output = Self;
    #[inline(always)]
    fn mul(self, o: Self) -> Self::Output {
        unsafe { Self(fmul_fast(self.0, o.0)) }
    }
}

impl core::ops::Add for FastFloat {
    type Output = Self;
    #[inline(always)]
    fn add(self, o: FastFloat) -> Self::Output {
        unsafe { Self(fadd_fast(self.0, o.0)) }
    }
}

impl core::ops::MulAssign<FastFloat> for FastFloat {
    #[inline(always)]
    fn mul_assign(&mut self, o: FastFloat) {
        unsafe {
            self.0 = fmul_fast(self.0, o.0);
        }
    }
}

impl core::ops::Mul for &FastFloat {
    type Output = FastFloat;
    #[inline(always)]
    fn mul(self, o: Self) -> Self::Output {
        unsafe { FastFloat(fmul_fast(self.0, o.0)) }
    }
}

impl From<Float> for FastFloat {
    #[inline(always)]
    fn from(f: Float) -> Self {
        Self(f)
    }
}
impl From<FastFloat> for Float {
    #[inline(always)]
    fn from(f: FastFloat) -> Self {
        f.0
    }
}
