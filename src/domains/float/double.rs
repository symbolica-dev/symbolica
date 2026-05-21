use std::{
    fmt::{self, Debug, Display, Formatter, LowerExp},
    hash::Hash,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use num_traits::FromPrimitive;
use rand::Rng;
use rug::Float as MultiPrecisionFloat;
use simba::scalar::{ComplexField, RealField};
use xprec::{CompensatedArithmetic, Df64};

use super::{Constructible, FixedPrecision, FloatLike, Real, RealLike, SingleFloat};
use crate::domains::{InternalOrdering, integer::Integer, rational::Rational};

/// A 106-bit precision floating point number represented by the compensated sum of two `f64` values.
///
/// This float has much faster arithmetic operations than `f128` (>3x) and a 106-bit precision `Float`.
/// Make sure to compile with AVX2 on X64 architectures to make use of
/// faster fused multiply-addition.
#[repr(transparent)]
#[derive(Debug, Copy, Clone)]
pub struct DoubleFloat(pub(crate) Df64);

impl Default for DoubleFloat {
    fn default() -> Self {
        Self(Df64::new(0.))
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for DoubleFloat {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        [self.0.hi(), self.0.lo()].serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for DoubleFloat {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let [hi, lo] = <[f64; 2]>::deserialize(deserializer)?;
        if hi + lo == hi || !hi.is_finite() {
            Ok(DoubleFloat(unsafe { Df64::new_full(hi, lo) }))
        } else {
            Err(serde::de::Error::custom("invalid Df64 hi/lo pair"))
        }
    }
}

#[cfg(feature = "bincode")]
impl bincode::Encode for DoubleFloat {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        [self.0.hi(), self.0.lo()].encode(encoder)
    }
}

#[cfg(feature = "bincode")]
bincode::impl_borrow_decode!(DoubleFloat);
#[cfg(feature = "bincode")]
impl<Context> bincode::Decode<Context> for DoubleFloat {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let [hi, lo] = <[f64; 2]>::decode(decoder)?;

        if hi + lo == hi || !hi.is_finite() {
            return Ok(DoubleFloat(unsafe { Df64::new_full(hi, lo) }));
        }

        Err(bincode::error::DecodeError::Other(
            "Failed to decode DoubleFloat",
        ))
    }
}

impl DoubleFloat {
    pub fn into_inner(self) -> Df64 {
        self.0
    }

    /// Returns the sum of `a` and `b`, with compensation for rounding errors.
    pub fn from_compensated_sum(a: f64, b: f64) -> Self {
        Df64::compensated_sum(a, b).into()
    }

    #[inline(always)]
    fn is_nan(&self) -> bool {
        self.0.hi().is_nan() || self.0.lo().is_nan()
    }

    #[inline]
    fn binary_exp(mut base: Df64, mut exp: u64) -> Df64 {
        let mut result = Df64::ONE;

        while exp != 0 {
            if exp & 1 == 1 {
                result *= base;
            }
            exp >>= 1;
            if exp != 0 {
                base *= base;
            }
        }

        result
    }

    #[inline]
    fn powi(base: Df64, exp: i64) -> Df64 {
        if exp >= 0 {
            Self::binary_exp(base, exp as u64)
        } else {
            Self::binary_exp(base, exp.unsigned_abs()).recip()
        }
    }

    #[inline]
    fn get_integer_exponent(exp: Df64) -> Option<i64> {
        if !exp.hi().is_finite() {
            return None;
        }

        let truncated = exp.trunc();
        if exp != truncated {
            return None;
        }

        let hi = truncated.hi();
        if !(i64::MIN as f64..=i64::MAX as f64).contains(&hi) {
            return None;
        }

        Some(hi as i64)
    }
}

impl FloatLike for DoubleFloat {
    #[inline(always)]
    fn set_from(&mut self, other: &Self) {
        *self = *other;
    }

    #[inline(always)]
    fn mul_add(&self, a: &Self, b: &Self) -> Self {
        (self.0 * a.0 + b.0).into()
    }

    #[inline(always)]
    fn neg(&self) -> Self {
        (-self.0).into()
    }

    #[inline(always)]
    fn zero(&self) -> Self {
        0f64.into()
    }

    #[inline(always)]
    fn new_zero() -> Self {
        0f64.into()
    }

    #[inline(always)]
    fn one(&self) -> Self {
        1f64.into()
    }

    #[inline]
    fn pow(&self, e: u64) -> Self {
        debug_assert!(e <= i32::MAX as u64);

        // `Df64::powi` is implemented as `exp(e * log(self))` and does not handle base <= 0
        if e == 0 {
            return 1f64.into();
        }

        if !self.0.hi().is_finite() {
            return self.0.hi().powi(e as i32).into();
        }

        Self::binary_exp(self.0, e).into()
    }

    #[inline(always)]
    fn inv(&self) -> Self {
        self.0.recip().into()
    }

    #[inline(always)]
    fn from_usize(&self, a: usize) -> Self {
        Df64::from_usize(a).unwrap().into()
    }

    #[inline(always)]
    fn from_i64(&self, a: i64) -> Self {
        Df64::from_i64(a).unwrap().into()
    }

    #[inline(always)]
    fn get_precision(&self) -> u32 {
        106
    }

    #[inline(always)]
    fn get_epsilon(&self) -> f64 {
        f64::EPSILON * f64::EPSILON / 2.0
    }

    #[inline(always)]
    fn fixed_precision(&self) -> bool {
        true
    }

    #[inline(always)]
    fn sample_unit<R: Rng + ?Sized>(&self, rng: &mut R) -> Self {
        rng.random::<f64>().into()
    }

    #[inline(always)]
    fn is_fully_zero(&self) -> bool {
        self.0.hi() == 0. && self.0.lo() == 0.
    }
}

impl Neg for DoubleFloat {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        (-self.0).into()
    }
}

impl Add<&DoubleFloat> for DoubleFloat {
    type Output = Self;

    #[inline]
    fn add(self, rhs: &Self) -> Self::Output {
        (self.0 + rhs.0).into()
    }
}

impl Add<DoubleFloat> for DoubleFloat {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        (self.0 + rhs.0).into()
    }
}

impl Sub<&DoubleFloat> for DoubleFloat {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: &Self) -> Self::Output {
        (self.0 - rhs.0).into()
    }
}

impl Sub<DoubleFloat> for DoubleFloat {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        (self.0 - rhs.0).into()
    }
}

impl Mul<&DoubleFloat> for DoubleFloat {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: &Self) -> Self::Output {
        (self.0 * rhs.0).into()
    }
}

impl Mul<DoubleFloat> for DoubleFloat {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        (self.0 * rhs.0).into()
    }
}

impl Div<&DoubleFloat> for DoubleFloat {
    type Output = Self;

    #[inline]
    fn div(self, rhs: &Self) -> Self::Output {
        (self.0 / rhs.0).into()
    }
}

impl Div<DoubleFloat> for DoubleFloat {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        (self.0 / rhs.0).into()
    }
}

impl AddAssign<&DoubleFloat> for DoubleFloat {
    #[inline]
    fn add_assign(&mut self, rhs: &DoubleFloat) {
        self.0 += rhs.0;
    }
}

impl AddAssign<DoubleFloat> for DoubleFloat {
    #[inline]
    fn add_assign(&mut self, rhs: DoubleFloat) {
        self.0 += rhs.0;
    }
}

impl SubAssign<&DoubleFloat> for DoubleFloat {
    #[inline]
    fn sub_assign(&mut self, rhs: &DoubleFloat) {
        self.0 -= rhs.0;
    }
}

impl SubAssign<DoubleFloat> for DoubleFloat {
    #[inline]
    fn sub_assign(&mut self, rhs: DoubleFloat) {
        self.0 -= rhs.0;
    }
}

impl MulAssign<&DoubleFloat> for DoubleFloat {
    #[inline]
    fn mul_assign(&mut self, rhs: &DoubleFloat) {
        self.0 *= rhs.0;
    }
}

impl MulAssign<DoubleFloat> for DoubleFloat {
    #[inline]
    fn mul_assign(&mut self, rhs: DoubleFloat) {
        self.0 *= rhs.0;
    }
}

impl DivAssign<&DoubleFloat> for DoubleFloat {
    #[inline]
    fn div_assign(&mut self, rhs: &DoubleFloat) {
        self.0 /= rhs.0;
    }
}

impl DivAssign<DoubleFloat> for DoubleFloat {
    #[inline]
    fn div_assign(&mut self, rhs: DoubleFloat) {
        self.0 /= rhs.0;
    }
}

impl SingleFloat for DoubleFloat {
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0 == Df64::ZERO
    }

    #[inline(always)]
    fn is_one(&self) -> bool {
        self.0 == Df64::ONE
    }

    #[inline(always)]
    fn is_finite(&self) -> bool {
        self.0.is_finite()
    }

    #[inline(always)]
    fn from_rational(&self, rat: &Rational) -> Self {
        rat.into()
    }
}

impl RealLike for DoubleFloat {
    fn to_usize_clamped(&self) -> usize {
        if !self.0.is_finite() {
            return if self.0.hi().is_sign_negative() {
                0
            } else {
                usize::MAX
            };
        }

        if self.0.hi().is_sign_negative() {
            0
        } else {
            let truncated = self.0.trunc().hi();
            if truncated >= usize::MAX as f64 {
                usize::MAX
            } else {
                truncated as usize
            }
        }
    }

    fn to_f64(&self) -> f64 {
        self.0.hi() + self.0.lo()
    }

    #[inline(always)]
    fn round_to_nearest_integer(&self) -> Integer {
        // TODO: change API to Result<Integer, _>
        if !self.0.is_finite() {
            return if self.0.hi().is_sign_negative() {
                i64::MIN.into()
            } else {
                i64::MAX.into()
            };
        }

        Integer::from_f64((self.0.round()).hi())
    }
}

impl Constructible for DoubleFloat {
    #[inline(always)]
    fn new_one() -> Self {
        1f64.into()
    }

    #[inline(always)]
    fn new_from_usize(a: usize) -> Self {
        Df64::from_usize(a).unwrap().into()
    }

    #[inline(always)]
    fn new_from_i64(a: i64) -> Self {
        Df64::from_i64(a).unwrap().into()
    }

    #[inline(always)]
    fn new_sample_unit<R: Rng + ?Sized>(rng: &mut R) -> Self {
        rng.random::<f64>().into()
    }
}

impl Real for DoubleFloat {
    #[inline(always)]
    fn pi(&self) -> Self {
        Df64::pi().into()
    }

    #[inline(always)]
    fn e(&self) -> Self {
        Df64::e().into()
    }

    #[inline(always)]
    fn euler(&self) -> Self {
        Df64::compensated_sum(0.577_215_664_901_532_9, -4.942_915_152_430_647e-18).into()
    }

    #[inline(always)]
    fn phi(&self) -> Self {
        Df64::compensated_sum(1.618_033_988_749_895, -5.432_115_203_682_505_5e-17).into()
    }

    #[inline(always)]
    fn i(&self) -> Option<Self> {
        None
    }

    #[inline(always)]
    fn conj(&self) -> Self {
        *self
    }

    #[inline(always)]
    fn norm(&self) -> Self {
        self.0.abs().into()
    }

    #[inline(always)]
    fn sqrt(&self) -> Self {
        let hi = self.0.hi();
        if hi == 0. {
            // avoid relying on subnormals inside the compensated sqrt path,
            // as DAZ (Denormals Are Zero) may be enabled
            return hi.into();
        }
        if hi < 0.0 || !hi.is_finite() {
            return hi.sqrt().into();
        }

        self.0.sqrt().into()
    }

    #[inline(always)]
    fn log(&self) -> Self {
        if !self.0.hi().is_finite() {
            return self.0.hi().ln().into();
        }

        self.0.ln().into()
    }

    #[inline(always)]
    fn exp(&self) -> Self {
        self.0.exp().into()
    }

    #[inline(always)]
    fn sin(&self) -> Self {
        if !self.0.hi().is_finite() {
            return self.0.hi().sin().into();
        }

        self.0.sin().into()
    }

    #[inline(always)]
    fn cos(&self) -> Self {
        if !self.0.hi().is_finite() {
            return self.0.hi().cos().into();
        }

        self.0.cos().into()
    }

    #[inline(always)]
    fn tan(&self) -> Self {
        if !self.0.hi().is_finite() {
            return self.0.hi().tan().into();
        }

        self.0.tan().into()
    }

    #[inline(always)]
    fn asin(&self) -> Self {
        self.0.asin().into()
    }

    #[inline(always)]
    fn acos(&self) -> Self {
        self.0.acos().into()
    }

    #[inline(always)]
    fn atan2(&self, x: &Self) -> Self {
        self.0.atan2(x.0).into()
    }

    #[inline(always)]
    fn sinh(&self) -> Self {
        self.0.sinh().into()
    }

    #[inline(always)]
    fn cosh(&self) -> Self {
        self.0.cosh().into()
    }

    #[inline(always)]
    fn tanh(&self) -> Self {
        self.0.tanh().into()
    }

    #[inline(always)]
    fn asinh(&self) -> Self {
        self.0.asinh().into()
    }

    #[inline(always)]
    fn acosh(&self) -> Self {
        self.0.acosh().into()
    }

    #[inline(always)]
    fn atanh(&self) -> Self {
        self.0.atanh().into()
    }

    #[inline(always)]
    fn powf(&self, e: &Self) -> Self {
        if e.0 == Df64::ZERO || self.0 == Df64::ONE {
            return 1f64.into();
        }

        if self.0 == Df64::from(-1.0) && e.0.hi().is_infinite() {
            return 1f64.into();
        }

        if self.is_nan() || e.is_nan() {
            return Df64::NAN.into();
        }

        if self.0.hi() == 0.0 {
            return self.0.hi().powf(e.0.hi()).into();
        }

        if let Some(integer_exponent) = Self::get_integer_exponent(e.0) {
            return Self::powi(self.0, integer_exponent).into();
        }

        if self.0.hi().is_sign_negative() || !self.0.hi().is_finite() || !e.0.hi().is_finite() {
            return self.0.hi().powf(e.0.hi()).into();
        }

        self.0.powf(e.0).into()
    }
}

impl FixedPrecision for DoubleFloat {
    const BINARY_PRECISION: usize = 106;
}

impl From<f64> for DoubleFloat {
    #[inline(always)]
    fn from(value: f64) -> Self {
        DoubleFloat(value.into())
    }
}

impl From<Df64> for DoubleFloat {
    #[inline(always)]
    fn from(value: Df64) -> Self {
        DoubleFloat(value)
    }
}

impl From<DoubleFloat> for Df64 {
    #[inline(always)]
    fn from(value: DoubleFloat) -> Self {
        value.0
    }
}

impl From<&Rational> for DoubleFloat {
    fn from(value: &Rational) -> Self {
        value.to_multi_prec_float(106).to_double_float()
    }
}

impl From<Rational> for DoubleFloat {
    fn from(value: Rational) -> Self {
        value.to_multi_prec_float(106).to_double_float()
    }
}

impl PartialEq for DoubleFloat {
    fn eq(&self, other: &Self) -> bool {
        if self.is_nan() && other.is_nan() {
            true
        } else if self.is_nan() || other.is_nan() {
            false
        } else {
            self.0.partial_cmp(&other.0) == Some(std::cmp::Ordering::Equal)
        }
    }
}

impl PartialOrd for DoubleFloat {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        if self.is_nan() || other.is_nan() {
            None
        } else {
            self.0.partial_cmp(&other.0)
        }
    }
}

impl InternalOrdering for DoubleFloat {
    fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl Display for DoubleFloat {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let value = MultiPrecisionFloat::with_val(106, self.0.hi()) + self.0.lo();
        Display::fmt(&value, f)
    }
}

impl LowerExp for DoubleFloat {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let value = MultiPrecisionFloat::with_val(106, self.0.hi()) + self.0.lo();
        LowerExp::fmt(&value, f)
    }
}

impl Eq for DoubleFloat {}

impl Hash for DoubleFloat {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        if self.is_nan() {
            state.write_u64(0x7ff8000000000000);
            return;
        }

        if !self.0.is_finite() {
            state.write_u64(0x7ff0000000000000);
            return;
        }

        if self.0.hi() == 0. {
            state.write_u64(0);
            return;
        }

        state.write_u64(self.0.hi().to_bits());
        state.write_u64(self.0.lo().to_bits());
    }
}
