use rand::Rng;
use wide::{f64x2, f64x4};

use super::{FloatLike, Real};
use crate::domains::rational::Rational;

macro_rules! simd_impl {
    ($t:ty, $p:ident) => {
        impl FloatLike for $t {
            #[inline(always)]
            fn set_from(&mut self, other: &Self) {
                *self = *other;
            }

            #[inline(always)]
            fn mul_add(&self, a: &Self, b: &Self) -> Self {
                *self * *a + b
            }

            #[inline(always)]
            fn neg(&self) -> Self {
                -*self
            }

            #[inline(always)]
            fn zero(&self) -> Self {
                Self::ZERO
            }

            #[inline(always)]
            fn new_zero() -> Self {
                Self::ZERO
            }

            #[inline(always)]
            fn one(&self) -> Self {
                Self::ONE
            }

            #[inline]
            fn pow(&self, e: u64) -> Self {
                // FIXME: use binary exponentiation
                debug_assert!(e <= i32::MAX as u64);
                (*self).powf(e as f64)
            }

            #[inline(always)]
            fn inv(&self) -> Self {
                Self::ONE / *self
            }

            #[inline(always)]
            fn from_usize(&self, a: usize) -> Self {
                Self::from(a as f64)
            }

            #[inline(always)]
            fn from_i64(&self, a: i64) -> Self {
                Self::from(a as f64)
            }

            #[inline(always)]
            fn get_precision(&self) -> u32 {
                53
            }

            #[inline(always)]
            fn get_epsilon(&self) -> f64 {
                f64::EPSILON / 2.
            }

            #[inline(always)]
            fn fixed_precision(&self) -> bool {
                true
            }

            fn sample_unit<R: Rng + ?Sized>(&self, rng: &mut R) -> Self {
                Self::from(rng.random::<f64>())
            }

            fn is_fully_zero(&self) -> bool {
                (*self).eq(&Self::ZERO)
            }
        }

        impl Real for $t {
            #[inline(always)]
            fn pi(&self) -> Self {
                std::f64::consts::PI.into()
            }

            #[inline(always)]
            fn e(&self) -> Self {
                std::f64::consts::E.into()
            }

            #[inline(always)]
            fn euler(&self) -> Self {
                0.577_215_664_901_532_9.into()
            }

            #[inline(always)]
            fn phi(&self) -> Self {
                1.618_033_988_749_895.into()
            }

            #[inline(always)]
            fn i(&self) -> Option<Self> {
                None
            }

            #[inline(always)]
            fn conj(&self) -> Self {
                (*self)
            }

            #[inline(always)]
            fn norm(&self) -> Self {
                (*self).abs()
            }

            #[inline(always)]
            fn sqrt(&self) -> Self {
                (*self).sqrt()
            }

            #[inline(always)]
            fn log(&self) -> Self {
                (*self).ln()
            }

            #[inline(always)]
            fn exp(&self) -> Self {
                (*self).exp()
            }

            #[inline(always)]
            fn sin(&self) -> Self {
                (*self).sin()
            }

            #[inline(always)]
            fn cos(&self) -> Self {
                (*self).cos()
            }

            #[inline(always)]
            fn tan(&self) -> Self {
                (*self).tan()
            }

            #[inline(always)]
            fn asin(&self) -> Self {
                (*self).asin()
            }

            #[inline(always)]
            fn acos(&self) -> Self {
                (*self).acos()
            }

            #[inline(always)]
            fn atan2(&self, x: &Self) -> Self {
                (*self).atan2(*x)
            }

            #[inline(always)]
            fn sinh(&self) -> Self {
                unimplemented!("Hyperbolic geometric functions are not supported with SIMD");
            }

            #[inline(always)]
            fn cosh(&self) -> Self {
                unimplemented!("Hyperbolic geometric functions are not supported with SIMD");
            }

            #[inline(always)]
            fn tanh(&self) -> Self {
                unimplemented!("Hyperbolic geometric functions are not supported with SIMD");
            }

            #[inline(always)]
            fn asinh(&self) -> Self {
                unimplemented!("Hyperbolic geometric functions are not supported with SIMD");
            }

            #[inline(always)]
            fn acosh(&self) -> Self {
                unimplemented!("Hyperbolic geometric functions are not supported with SIMD");
            }

            #[inline(always)]
            fn atanh(&self) -> Self {
                unimplemented!("Hyperbolic geometric functions are not supported with SIMD");
            }

            #[inline(always)]
            fn powf(&self, e: &Self) -> Self {
                (*self).$p(*e)
            }
        }

        impl From<&Rational> for $t {
            fn from(value: &Rational) -> Self {
                value.to_f64().into()
            }
        }
    };
}

simd_impl!(f64x2, pow_f64x2);
simd_impl!(f64x4, pow_f64x4);
