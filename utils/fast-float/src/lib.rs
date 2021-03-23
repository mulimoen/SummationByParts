#![no_std]
#![feature(core_intrinsics)]

use float::Float;

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, Default)]
pub struct FastFloat(Float);

use core::intrinsics::{fadd_fast, fdiv_fast, fmul_fast, fsub_fast};

impl core::ops::Mul for FastFloat {
    type Output = Self;
    #[inline(always)]
    fn mul(self, o: Self) -> Self::Output {
        unsafe { Self(fmul_fast(self.0, o.0)) }
    }
}

impl core::ops::Mul<Float> for FastFloat {
    type Output = FastFloat;
    #[inline(always)]
    fn mul(self, o: Float) -> Self::Output {
        unsafe { Self(fmul_fast(self.0, o)) }
    }
}

impl core::ops::Mul<FastFloat> for Float {
    type Output = FastFloat;
    #[inline(always)]
    fn mul(self, o: FastFloat) -> Self::Output {
        unsafe { FastFloat(fmul_fast(self, o.0)) }
    }
}

impl core::ops::Add<FastFloat> for FastFloat {
    type Output = FastFloat;
    #[inline(always)]
    fn add(self, o: FastFloat) -> Self::Output {
        unsafe { Self(fadd_fast(self.0, o.0)) }
    }
}

impl core::ops::Add<Float> for FastFloat {
    type Output = FastFloat;
    #[inline(always)]
    fn add(self, o: Float) -> Self::Output {
        unsafe { Self(fadd_fast(self.0, o)) }
    }
}

impl core::ops::Add<FastFloat> for Float {
    type Output = FastFloat;
    #[inline(always)]
    fn add(self, o: FastFloat) -> Self::Output {
        unsafe { FastFloat(fadd_fast(self, o.0)) }
    }
}

impl core::ops::Sub for FastFloat {
    type Output = Self;
    #[inline(always)]
    fn sub(self, o: FastFloat) -> Self::Output {
        unsafe { Self(fsub_fast(self.0, o.0)) }
    }
}

impl core::ops::Div for FastFloat {
    type Output = Self;
    #[inline(always)]
    fn div(self, o: FastFloat) -> Self::Output {
        unsafe { Self(fdiv_fast(self.0, o.0)) }
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

mod numt {
    use super::{FastFloat, Float};
    use num_traits::identities::{One, Zero};

    impl One for FastFloat {
        fn one() -> Self {
            Self(Float::one())
        }
    }
    impl Zero for FastFloat {
        fn zero() -> Self {
            Self(Float::zero())
        }
        fn is_zero(&self) -> bool {
            self.0.is_zero()
        }
    }
}
