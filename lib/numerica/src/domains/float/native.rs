use std::{
    fmt::{self, Debug, Display, Formatter, LowerExp},
    hash::Hash,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use rand::Rng;

use super::{Constructible, FixedPrecision, FloatLike, Real, RealLike, SingleFloat};
use crate::domains::{InternalOrdering, integer::Integer, rational::Rational};

impl FloatLike for f64 {
    #[inline(always)]
    fn set_from(&mut self, other: &Self) {
        *self = *other;
    }

    #[inline(always)]
    fn mul_add(&self, a: &Self, b: &Self) -> Self {
        f64::mul_add(*self, *a, *b)
    }

    #[inline(always)]
    fn neg(&self) -> Self {
        -self
    }

    #[inline(always)]
    fn zero(&self) -> Self {
        0.
    }

    #[inline(always)]
    fn new_zero() -> Self {
        0.
    }

    #[inline(always)]
    fn one(&self) -> Self {
        1.
    }

    #[inline]
    fn pow(&self, e: u64) -> Self {
        // FIXME: use binary exponentiation
        debug_assert!(e <= i32::MAX as u64);
        self.powi(e as i32)
    }

    #[inline(always)]
    fn inv(&self) -> Self {
        1. / self
    }

    #[inline(always)]
    fn from_usize(&self, a: usize) -> Self {
        a as f64
    }

    #[inline(always)]
    fn from_i64(&self, a: i64) -> Self {
        a as f64
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
        rng.random()
    }

    #[inline(always)]
    fn is_fully_zero(&self) -> bool {
        *self == 0.
    }
}

impl SingleFloat for f64 {
    #[inline(always)]
    fn is_zero(&self) -> bool {
        *self == 0.
    }

    #[inline(always)]
    fn is_one(&self) -> bool {
        *self == 1.
    }

    #[inline(always)]
    fn is_finite(&self) -> bool {
        (*self).is_finite()
    }

    #[inline(always)]
    fn from_rational(&self, rat: &Rational) -> Self {
        rat.to_f64()
    }
}

impl RealLike for f64 {
    fn to_usize_clamped(&self) -> usize {
        *self as usize
    }

    fn to_f64(&self) -> f64 {
        *self
    }

    #[inline(always)]
    fn round_to_nearest_integer(&self) -> Integer {
        if *self < 0. {
            Integer::from_f64(*self - 0.5)
        } else {
            Integer::from_f64(*self + 0.5)
        }
    }
}

impl Constructible for f64 {
    #[inline(always)]
    fn new_one() -> Self {
        1.
    }

    #[inline(always)]
    fn new_from_usize(a: usize) -> Self {
        a as f64
    }

    #[inline(always)]
    fn new_from_i64(a: i64) -> Self {
        a as f64
    }

    #[inline(always)]
    fn new_sample_unit<R: Rng + ?Sized>(rng: &mut R) -> Self {
        rng.random()
    }
}

impl Real for f64 {
    #[inline(always)]
    fn pi(&self) -> Self {
        std::f64::consts::PI
    }

    #[inline(always)]
    fn e(&self) -> Self {
        std::f64::consts::E
    }

    #[inline(always)]
    fn euler(&self) -> Self {
        0.577_215_664_901_532_9
    }

    #[inline(always)]
    fn phi(&self) -> Self {
        1.618_033_988_749_895
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
        f64::abs(*self)
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
        (*self).sinh()
    }

    #[inline(always)]
    fn cosh(&self) -> Self {
        (*self).cosh()
    }

    #[inline(always)]
    fn tanh(&self) -> Self {
        (*self).tanh()
    }

    #[inline(always)]
    fn asinh(&self) -> Self {
        (*self).asinh()
    }

    #[inline(always)]
    fn acosh(&self) -> Self {
        (*self).acosh()
    }

    #[inline(always)]
    fn atanh(&self) -> Self {
        (*self).atanh()
    }

    #[inline]
    fn powf(&self, e: &f64) -> Self {
        (*self).powf(*e)
    }
}

impl FixedPrecision for f64 {
    const BINARY_PRECISION: usize = 53;
}

impl From<&Rational> for f64 {
    fn from(value: &Rational) -> Self {
        value.to_f64()
    }
}

impl From<Rational> for f64 {
    fn from(value: Rational) -> Self {
        value.to_f64()
    }
}

/// A wrapper around `f64` that implements `Eq`, `Ord`, and `Hash`.
/// All `NaN` values are considered equal, and `-0` is considered equal to `0`.
#[repr(transparent)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[derive(Debug, Copy, Clone)]
pub struct F64(pub f64);

impl F64 {
    pub fn into_inner(self) -> f64 {
        self.0
    }
}

impl FloatLike for F64 {
    #[inline(always)]
    fn set_from(&mut self, other: &Self) {
        *self = *other;
    }

    #[inline(always)]
    fn mul_add(&self, a: &Self, b: &Self) -> Self {
        self.0.mul_add(a.0, b.0).into()
    }

    #[inline(always)]
    fn neg(&self) -> Self {
        (-self.0).into()
    }

    #[inline(always)]
    fn zero(&self) -> Self {
        (0.).into()
    }

    #[inline(always)]
    fn new_zero() -> Self {
        (0.).into()
    }

    #[inline(always)]
    fn one(&self) -> Self {
        (1.).into()
    }

    #[inline(always)]
    fn pow(&self, e: u64) -> Self {
        FloatLike::pow(&self.0, e).into()
    }

    #[inline(always)]
    fn inv(&self) -> Self {
        self.0.inv().into()
    }

    #[inline(always)]
    fn from_usize(&self, a: usize) -> Self {
        self.0.from_usize(a).into()
    }

    #[inline(always)]
    fn from_i64(&self, a: i64) -> Self {
        self.0.from_i64(a).into()
    }

    #[inline(always)]
    fn get_precision(&self) -> u32 {
        self.0.get_precision()
    }

    #[inline(always)]
    fn get_epsilon(&self) -> f64 {
        self.0.get_epsilon()
    }

    #[inline(always)]
    fn fixed_precision(&self) -> bool {
        self.0.fixed_precision()
    }

    #[inline(always)]
    fn sample_unit<R: Rng + ?Sized>(&self, rng: &mut R) -> Self {
        self.0.sample_unit(rng).into()
    }

    #[inline(always)]
    fn is_fully_zero(&self) -> bool {
        self.0 == 0.
    }
}

impl Neg for F64 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        self.0.neg().into()
    }
}

impl Add<&F64> for F64 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: &Self) -> Self::Output {
        (self.0 + rhs.0).into()
    }
}

impl Add<F64> for F64 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        (self.0 + rhs.0).into()
    }
}

impl Sub<&F64> for F64 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: &Self) -> Self::Output {
        (self.0 - rhs.0).into()
    }
}

impl Sub<F64> for F64 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        (self.0 - rhs.0).into()
    }
}

impl Mul<&F64> for F64 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: &Self) -> Self::Output {
        (self.0 * rhs.0).into()
    }
}

impl Mul<F64> for F64 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        (self.0 * rhs.0).into()
    }
}

impl Div<&F64> for F64 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: &Self) -> Self::Output {
        (self.0 / rhs.0).into()
    }
}

impl Div<F64> for F64 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        (self.0 / rhs.0).into()
    }
}

impl AddAssign<&F64> for F64 {
    #[inline]
    fn add_assign(&mut self, rhs: &F64) {
        self.0 += rhs.0;
    }
}

impl AddAssign<F64> for F64 {
    #[inline]
    fn add_assign(&mut self, rhs: F64) {
        self.0 += rhs.0;
    }
}

impl SubAssign<&F64> for F64 {
    #[inline]
    fn sub_assign(&mut self, rhs: &F64) {
        self.0 -= rhs.0;
    }
}

impl SubAssign<F64> for F64 {
    #[inline]
    fn sub_assign(&mut self, rhs: F64) {
        self.0 -= rhs.0;
    }
}

impl MulAssign<&F64> for F64 {
    #[inline]
    fn mul_assign(&mut self, rhs: &F64) {
        self.0 *= rhs.0;
    }
}

impl MulAssign<F64> for F64 {
    #[inline]
    fn mul_assign(&mut self, rhs: F64) {
        self.0 *= rhs.0;
    }
}

impl DivAssign<&F64> for F64 {
    #[inline]
    fn div_assign(&mut self, rhs: &F64) {
        self.0 /= rhs.0
    }
}

impl DivAssign<F64> for F64 {
    #[inline]
    fn div_assign(&mut self, rhs: F64) {
        self.0 /= rhs.0
    }
}

impl SingleFloat for F64 {
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    #[inline(always)]
    fn is_one(&self) -> bool {
        self.0.is_one()
    }

    #[inline(always)]
    fn is_finite(&self) -> bool {
        self.0.is_finite()
    }

    #[inline(always)]
    fn from_rational(&self, rat: &Rational) -> Self {
        rat.to_f64().into()
    }
}

impl RealLike for F64 {
    fn to_usize_clamped(&self) -> usize {
        self.0.to_usize_clamped()
    }

    fn to_f64(&self) -> f64 {
        self.0.to_f64()
    }

    #[inline(always)]
    fn round_to_nearest_integer(&self) -> Integer {
        self.0.round_to_nearest_integer()
    }
}

impl Constructible for F64 {
    #[inline(always)]
    fn new_one() -> Self {
        f64::new_one().into()
    }

    #[inline(always)]
    fn new_from_usize(a: usize) -> Self {
        f64::new_from_usize(a).into()
    }

    #[inline(always)]
    fn new_from_i64(a: i64) -> Self {
        f64::new_from_i64(a).into()
    }

    #[inline(always)]
    fn new_sample_unit<R: Rng + ?Sized>(rng: &mut R) -> Self {
        f64::new_sample_unit(rng).into()
    }
}

impl Real for F64 {
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
        *self
    }

    #[inline(always)]
    fn norm(&self) -> Self {
        self.0.norm().into()
    }

    #[inline(always)]
    fn sqrt(&self) -> Self {
        self.0.sqrt().into()
    }

    #[inline(always)]
    fn log(&self) -> Self {
        self.0.ln().into()
    }

    #[inline(always)]
    fn exp(&self) -> Self {
        self.0.exp().into()
    }

    #[inline(always)]
    fn sin(&self) -> Self {
        self.0.sin().into()
    }

    #[inline(always)]
    fn cos(&self) -> Self {
        self.0.cos().into()
    }

    #[inline(always)]
    fn tan(&self) -> Self {
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
        self.0.powf(e.0).into()
    }
}

impl FixedPrecision for F64 {
    const BINARY_PRECISION: usize = 53;
}

impl From<f64> for F64 {
    #[inline(always)]
    fn from(value: f64) -> Self {
        F64(value)
    }
}

impl PartialEq for F64 {
    fn eq(&self, other: &Self) -> bool {
        if self.0.is_nan() && other.0.is_nan() {
            true
        } else {
            self.0 == other.0
        }
    }
}

impl PartialOrd for F64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl InternalOrdering for F64 {
    fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl Display for F64 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0, f)
    }
}

impl LowerExp for F64 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        LowerExp::fmt(&self.0, f)
    }
}

impl Eq for F64 {}

impl Hash for F64 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        if self.0.is_nan() {
            state.write_u64(0x7ff8000000000000);
        } else if self.0 == 0. {
            state.write_u64(0);
        } else {
            state.write_u64(self.0.to_bits());
        }
    }
}
