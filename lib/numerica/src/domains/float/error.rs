use std::{
    fmt::{self, Debug, Display, Formatter, LowerExp, Write},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use rand::Rng;

use super::{FixedPrecision, FloatLike, Magnitude, Real, RealLike, SingleFloat};
use crate::domains::{integer::Integer, rational::Rational};

/// A float that does linear error propagation.
///
/// Absolute errors use an eight-byte, upward-rounded [`Magnitude`] with 32
/// significant bits and a wide binary exponent. The central value keeps its
/// original arithmetic and precision semantics. Linear propagation is an
/// estimate, not a certified enclosure, and does not retain input correlations.
/// Use [`Self::get_error_magnitude`] to inspect the full error range; the
/// compatibility [`Self::get_absolute_error`] getter projects upward to `f64`.
#[derive(Copy, Clone)]
pub struct ErrorPropagatingFloat<T: FloatLike> {
    value: T,
    abs_err: Magnitude,
}

impl<T: FloatLike + From<f64>> From<f64> for ErrorPropagatingFloat<T> {
    fn from(value: f64) -> Self {
        if value == 0. {
            ErrorPropagatingFloat {
                value: value.into(),
                abs_err: Magnitude::ONE.scale_pow2(-52),
            }
        } else {
            ErrorPropagatingFloat {
                value: value.into(),
                abs_err: Magnitude::from_f64(value).scale_pow2(-52),
            }
        }
    }
}

impl<T: FloatLike> Neg for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        ErrorPropagatingFloat {
            value: -self.value,
            abs_err: self.abs_err,
        }
    }
}

impl<T: FloatLike> Add<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: &Self) -> Self::Output {
        ErrorPropagatingFloat {
            abs_err: self.abs_err + rhs.abs_err,
            value: self.value + &rhs.value,
        }
    }
}

impl<T: FloatLike> Add<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl<T: FloatLike> Sub<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: &Self) -> Self::Output {
        self - rhs.clone()
    }
}

impl<T: FloatLike> Sub<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl<T: RealLike> Mul<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: &Self) -> Self::Output {
        let value = self.value.clone() * &rhs.value;
        let r = rhs.value.absolute_magnitude();
        let s = self.value.absolute_magnitude();

        if s.is_zero() && r.is_zero() {
            ErrorPropagatingFloat {
                value,
                abs_err: self.abs_err * rhs.abs_err,
            }
        } else {
            ErrorPropagatingFloat {
                value,
                abs_err: self.abs_err * r + rhs.abs_err * s,
            }
        }
    }
}

impl<T: RealLike + Add<Rational, Output = T>, R: Into<Rational>> Add<R>
    for ErrorPropagatingFloat<T>
{
    type Output = Self;

    #[inline]
    fn add(self, rhs: R) -> Self::Output {
        ErrorPropagatingFloat {
            abs_err: self.abs_err,
            value: self.value + rhs.into(),
        }
    }
}

impl<T: RealLike + Add<Rational, Output = T>, R: Into<Rational>> Sub<R>
    for ErrorPropagatingFloat<T>
{
    type Output = Self;

    #[inline]
    fn sub(self, rhs: R) -> Self::Output {
        self + -rhs.into()
    }
}

impl<T: RealLike + Mul<Rational, Output = T>, R: Into<Rational>> Mul<R>
    for ErrorPropagatingFloat<T>
{
    type Output = Self;

    #[inline]
    fn mul(self, rhs: R) -> Self::Output {
        let rhs = rhs.into();
        ErrorPropagatingFloat {
            abs_err: self.abs_err * rhs.absolute_magnitude(),
            value: self.value * rhs,
        }
        .truncate()
    }
}

impl<T: RealLike + Div<Rational, Output = T>, R: Into<Rational>> Div<R>
    for ErrorPropagatingFloat<T>
{
    type Output = Self;

    #[inline]
    fn div(self, rhs: R) -> Self::Output {
        let rhs = rhs.into();
        ErrorPropagatingFloat {
            abs_err: self.abs_err / rhs.absolute_magnitude_lower(),
            value: self.value.clone() / rhs,
        }
        .truncate()
    }
}

impl<T: FloatLike + From<f64>> Add<f64> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: f64) -> Self::Output {
        self + Self::from(rhs)
    }
}

impl<T: FloatLike + From<f64>> Sub<f64> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: f64) -> Self::Output {
        self - Self::from(rhs)
    }
}

impl<T: RealLike + From<f64>> Mul<f64> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        self * Self::from(rhs)
    }
}

impl<T: RealLike + From<f64>> Div<f64> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        self / Self::from(rhs)
    }
}

impl<T: RealLike> Mul<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self * &rhs
    }
}

impl<T: RealLike> Div<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: &Self) -> Self::Output {
        // Evaluate the linear weights without a narrowed or rounded reciprocal.
        // Divide centers directly so a finite quotient does not overflow its
        // otherwise unnecessary reciprocal intermediate.
        let denominator = rhs.value.absolute_magnitude_lower();
        let quotient = self.value.absolute_magnitude() / denominator;
        Self {
            abs_err: self.abs_err / denominator + (rhs.abs_err / denominator) * quotient,
            value: self.value / &rhs.value,
        }
    }
}

impl<T: RealLike> Div<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self / &rhs
    }
}

impl<T: RealLike> AddAssign<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn add_assign(&mut self, rhs: &ErrorPropagatingFloat<T>) {
        // TODO: optimize
        *self = self.clone() + rhs;
    }
}

impl<T: RealLike> AddAssign<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn add_assign(&mut self, rhs: ErrorPropagatingFloat<T>) {
        self.add_assign(&rhs)
    }
}

impl<T: RealLike> SubAssign<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: &ErrorPropagatingFloat<T>) {
        // TODO: optimize
        *self = self.clone() - rhs;
    }
}

impl<T: RealLike> SubAssign<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: ErrorPropagatingFloat<T>) {
        self.sub_assign(&rhs)
    }
}

impl<T: RealLike> MulAssign<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &ErrorPropagatingFloat<T>) {
        // TODO: optimize
        *self = self.clone() * rhs;
    }
}

impl<T: RealLike> MulAssign<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: ErrorPropagatingFloat<T>) {
        self.mul_assign(&rhs)
    }
}

impl<T: RealLike> DivAssign<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn div_assign(&mut self, rhs: &ErrorPropagatingFloat<T>) {
        // TODO: optimize
        *self = self.clone() / rhs;
    }
}

impl<T: RealLike> DivAssign<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn div_assign(&mut self, rhs: ErrorPropagatingFloat<T>) {
        self.div_assign(&rhs)
    }
}

impl<T: RealLike> ErrorPropagatingFloat<T> {
    /// Create a new precision tracking float with a number of precise decimal digits `prec`.
    /// The `prec` must be smaller than the precision of the underlying float.
    ///
    /// If the value provided is 0, the precision argument is interpreted as an accuracy (
    /// the number of digits of the absolute error).
    pub fn new(value: T, prec: f64) -> Self {
        let r = value.absolute_magnitude();

        if r.is_zero() {
            ErrorPropagatingFloat {
                abs_err: Magnitude::from_decimal_accuracy(prec),
                value,
            }
        } else {
            ErrorPropagatingFloat {
                abs_err: Magnitude::from_decimal_accuracy(prec) * r,
                value,
            }
        }
    }

    /// Construct with the underlying domain's native precision, without
    /// converting its epsilon through the binary64 exponent range.
    /// A zero center keeps the same absolute epsilon as [`Self::new`].
    /// Fixed-precision domains supply their native epsilon; variable-precision
    /// domains use `2^-bits`, where `bits` is the stored binary precision.
    pub fn new_with_native_precision(value: T) -> Self {
        let epsilon = if value.fixed_precision() {
            Magnitude::from_f64(value.get_epsilon())
        } else {
            Magnitude::ONE.scale_pow2(-i64::from(value.get_precision()))
        };
        let magnitude = value.absolute_magnitude();
        Self {
            abs_err: epsilon
                * if magnitude.is_zero() {
                    Magnitude::ONE
                } else {
                    magnitude
                },
            value,
        }
    }

    /// Return the error rounded upward to `f64`. Positive underflow saturates
    /// to the smallest subnormal and overflow to infinity. For checked conversion,
    /// use `get_error_magnitude().to_f64()`, which returns `None` outside that range.
    pub fn get_absolute_error(&self) -> f64 {
        self.abs_err.to_f64_saturating()
    }

    /// Return the stored error without narrowing its exponent range.
    pub fn get_error_magnitude(&self) -> Magnitude {
        self.abs_err
    }

    /// Return the relative error estimate, rounded upward to `f64` with the
    /// same saturation behavior as [`Self::get_absolute_error`].
    pub fn get_relative_error(&self) -> f64 {
        (self.abs_err / self.value.absolute_magnitude_lower()).to_f64_saturating()
    }

    /// Get the precision in number of decimal digits, or `None` for a zero center.
    /// The logarithmic ratio uses the stored wide exponents directly, so its
    /// computation does not narrow the error to `f64` first.
    #[inline(always)]
    pub fn get_precision(&self) -> Option<f64> {
        let r = self.value.absolute_magnitude();
        if r.is_zero() {
            None
        } else {
            Some(r.log2_ratio(self.abs_err) * std::f64::consts::LOG10_2)
        }
    }

    /// Get the absolute accuracy in decimal digits as `-log10(error)`.
    /// This uses the stored wide magnitude directly; zero error gives infinity.
    #[inline(always)]
    pub fn get_accuracy(&self) -> f64 {
        -self.abs_err.log10()
    }

    /// Truncate the precision to the maximal number of stable decimal digits
    /// of the underlying float.
    #[inline(always)]
    pub fn truncate(mut self) -> Self {
        if self.value.fixed_precision() {
            self.abs_err = self.abs_err.max(
                Magnitude::from_f64(self.value.get_epsilon()) * self.value.absolute_magnitude(),
            );
        }
        self
    }
}

impl<T: FloatLike> ErrorPropagatingFloat<T> {
    /// Construct with an explicit, nonnegative error magnitude.
    /// Stores the supplied value and error without adding roundoff or changing
    /// the value's precision. Infinity denotes an unbounded estimate; NaN denotes
    /// an unavailable estimate.
    pub fn new_with_error_magnitude(value: T, abs_err: Magnitude) -> Self {
        Self { value, abs_err }
    }

    /// Construct with absolute error `10^-acc`, rounded upward in the wide
    /// magnitude representation. Positive infinite accuracy gives zero error;
    /// negative infinite accuracy gives infinite error, and NaN remains NaN.
    pub fn new_with_accuracy(value: T, acc: f64) -> Self {
        ErrorPropagatingFloat {
            value,
            abs_err: Magnitude::from_decimal_accuracy(acc),
        }
    }

    /// Get the number.
    #[inline(always)]
    pub fn get_num(&self) -> &T {
        &self.value
    }
}

impl<T: RealLike> fmt::Display for ErrorPropagatingFloat<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if let Some(p) = self.get_precision() {
            if p < 0. {
                f.write_char('0')
            } else {
                f.write_fmt(format_args!("{0:.1$}", self.value, p as usize))
            }
        } else {
            f.write_char('0')
        }
    }
}

impl<T: RealLike> Debug for ErrorPropagatingFloat<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.value, f)?;

        if let Some(p) = self.get_precision() {
            f.write_fmt(format_args!("`{p:.2}"))
        } else {
            f.write_fmt(format_args!("``{:.2}", -self.abs_err.log10()))
        }
    }
}

impl<T: RealLike> LowerExp for ErrorPropagatingFloat<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl<T: FloatLike> PartialEq for ErrorPropagatingFloat<T> {
    fn eq(&self, other: &Self) -> bool {
        // TODO: ignore precision for partial equality?
        self.value == other.value
    }
}

impl<T: FloatLike + PartialOrd> PartialOrd for ErrorPropagatingFloat<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<T: RealLike> FloatLike for ErrorPropagatingFloat<T> {
    fn nan(&self) -> Option<Self> {
        Some(Self {
            value: self.value.nan()?,
            abs_err: Magnitude::NAN,
        })
    }

    fn set_from(&mut self, other: &Self) {
        self.value.set_from(&other.value);
        self.abs_err = other.abs_err;
    }

    fn mul_add(&self, a: &Self, b: &Self) -> Self {
        self.clone() * a + b
    }

    fn neg(&self) -> Self {
        -self.clone()
    }

    fn zero(&self) -> Self {
        ErrorPropagatingFloat {
            value: self.value.zero(),
            abs_err: Magnitude::ONE.scale_pow2(-i64::from(self.value.get_precision())),
        }
    }

    fn new_zero() -> Self {
        ErrorPropagatingFloat {
            value: T::new_zero(),
            abs_err: Magnitude::ONE.scale_pow2(-53),
        }
    }

    fn one(&self) -> Self {
        ErrorPropagatingFloat {
            value: self.value.one(),
            abs_err: Magnitude::ONE.scale_pow2(-i64::from(self.value.get_precision())),
        }
    }

    fn pow(&self, e: u64) -> Self {
        let i = self.value.absolute_magnitude();

        if i.is_zero() {
            return ErrorPropagatingFloat {
                value: self.value.pow(e),
                abs_err: self.abs_err.powf(e as f64),
            };
        }

        let r = self.value.pow(e);
        ErrorPropagatingFloat {
            abs_err: (self.abs_err / self.value.absolute_magnitude_lower())
                * Magnitude::from_f64(e as f64)
                * r.absolute_magnitude(),
            value: r,
        }
    }

    fn inv(&self) -> Self {
        let r = self.value.inv();
        let reciprocal = Magnitude::ONE / self.value.absolute_magnitude_lower();
        ErrorPropagatingFloat {
            abs_err: (self.abs_err * reciprocal) * reciprocal,
            value: r,
        }
    }

    /// Convert from a `usize`.
    fn from_usize(&self, a: usize) -> Self {
        let v = self.value.from_usize(a);
        let r = v.absolute_magnitude();
        if r.is_zero() {
            ErrorPropagatingFloat {
                value: v,
                abs_err: Magnitude::ONE.scale_pow2(-i64::from(self.value.get_precision())),
            }
        } else {
            ErrorPropagatingFloat {
                value: v,
                abs_err: Magnitude::ONE.scale_pow2(-i64::from(self.value.get_precision())) * r,
            }
        }
    }

    /// Convert from a `i64`.
    fn from_i64(&self, a: i64) -> Self {
        let v = self.value.from_i64(a);
        let r = v.absolute_magnitude();
        if r.is_zero() {
            ErrorPropagatingFloat {
                value: v,
                abs_err: Magnitude::ONE.scale_pow2(-i64::from(self.value.get_precision())),
            }
        } else {
            ErrorPropagatingFloat {
                value: v,
                abs_err: Magnitude::ONE.scale_pow2(-i64::from(self.value.get_precision())) * r,
            }
        }
    }

    fn get_precision(&self) -> u32 {
        // return the precision of the underlying float instead
        // of the current tracked precision
        self.value.get_precision()
    }

    fn set_precision(&mut self, precision: u32) {
        self.value.set_precision(precision);
        // reset the tracked error to the roundoff of the value
        let scale = self.value.absolute_magnitude();
        self.abs_err = if scale.is_zero() {
            Magnitude::ONE
        } else {
            scale
        } * Magnitude::ONE.scale_pow2(-i64::from(self.value.get_precision()));
    }

    fn get_epsilon(&self) -> f64 {
        Magnitude::ONE
            .scale_pow2(-i64::from(self.value.get_precision()))
            .to_f64_saturating()
    }

    #[inline(always)]
    fn fixed_precision(&self) -> bool {
        self.value.fixed_precision()
    }

    fn sample_unit<R: Rng + ?Sized>(&self, rng: &mut R) -> Self {
        let v = self.value.sample_unit(rng);
        ErrorPropagatingFloat {
            abs_err: self.abs_err * v.absolute_magnitude(),
            value: v,
        }
    }

    fn is_fully_zero(&self) -> bool {
        self.value.is_fully_zero()
    }
}

impl<T: RealLike> SingleFloat for ErrorPropagatingFloat<T> {
    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    fn is_one(&self) -> bool {
        self.value.is_one()
    }

    fn is_finite(&self) -> bool {
        self.value.is_finite()
    }

    fn from_rational(&self, rat: &Rational) -> Self {
        if rat.is_zero() {
            ErrorPropagatingFloat {
                value: self.value.from_rational(rat),
                abs_err: self.abs_err,
            }
        } else {
            ErrorPropagatingFloat {
                value: self.value.from_rational(rat),
                abs_err: self.abs_err * rat.absolute_magnitude(),
            }
        }
    }
}

impl<T: RealLike> RealLike for ErrorPropagatingFloat<T> {
    fn absolute_magnitude(&self) -> Magnitude {
        self.value.absolute_magnitude()
    }

    fn absolute_magnitude_lower(&self) -> Magnitude {
        self.value.absolute_magnitude_lower()
    }

    fn to_usize_clamped(&self) -> usize {
        self.value.to_usize_clamped()
    }

    fn to_f64(&self) -> f64 {
        self.value.to_f64()
    }

    fn round_to_nearest_integer(&self) -> Integer {
        // TODO: what does this do with the error?
        self.value.round_to_nearest_integer()
    }
}

impl<T: Real + RealLike> Real for ErrorPropagatingFloat<T> {
    fn pi(&self) -> Self {
        let v = self.value.pi();
        ErrorPropagatingFloat {
            abs_err: Magnitude::ONE.scale_pow2(-i64::from(self.value.get_precision()))
                * v.absolute_magnitude(),
            value: v,
        }
    }

    fn e(&self) -> Self {
        let v = self.value.e();
        ErrorPropagatingFloat {
            abs_err: Magnitude::ONE.scale_pow2(-i64::from(self.value.get_precision()))
                * v.absolute_magnitude(),
            value: v,
        }
    }

    fn euler(&self) -> Self {
        let v = self.value.euler();
        ErrorPropagatingFloat {
            abs_err: Magnitude::ONE.scale_pow2(-i64::from(self.value.get_precision()))
                * v.absolute_magnitude(),
            value: v,
        }
    }

    fn phi(&self) -> Self {
        let v = self.value.phi();
        ErrorPropagatingFloat {
            abs_err: Magnitude::ONE.scale_pow2(-i64::from(self.value.get_precision()))
                * v.absolute_magnitude(),
            value: v,
        }
    }

    #[inline(always)]
    fn i(&self) -> Option<Self> {
        Some(ErrorPropagatingFloat {
            value: self.value.i()?,
            abs_err: Magnitude::ONE.scale_pow2(-i64::from(self.value.get_precision())),
        })
    }

    fn conj(&self) -> Self {
        ErrorPropagatingFloat {
            abs_err: self.abs_err,
            value: self.value.conj(),
        }
    }

    fn norm(&self) -> Self {
        ErrorPropagatingFloat {
            abs_err: self.abs_err,
            value: self.value.norm(),
        }
    }

    fn hypot(&self, other: &Self) -> Self {
        let value = self.value.hypot(&other.value);
        if value.is_zero() || !value.is_finite() {
            // Keep the existing arithmetic at the nondifferentiable origin and
            // for exceptional inputs, without inventing an uncertainty bound.
            return (self.clone() * self + other.clone() * other).sqrt();
        }
        // Form bounded derivative weights in the original numeric domain.
        // Squaring either input first can overflow or underflow unnecessarily.
        let denominator = value.absolute_magnitude_lower();
        let x_weight = self.value.absolute_magnitude() / denominator;
        let y_weight = other.value.absolute_magnitude() / denominator;
        ErrorPropagatingFloat {
            abs_err: self.abs_err * x_weight + other.abs_err * y_weight,
            value,
        }
        .truncate()
    }

    fn sqrt(&self) -> Self {
        let v = self.value.sqrt();
        ErrorPropagatingFloat {
            abs_err: self.abs_err / v.absolute_magnitude_lower().scale_pow2(1),
            value: v,
        }
        .truncate()
    }

    fn log(&self) -> Self {
        let r = self.value.log();
        ErrorPropagatingFloat {
            abs_err: self.abs_err / self.value.absolute_magnitude_lower(),
            value: r,
        }
        .truncate()
    }

    fn exp(&self) -> Self {
        let v = self.value.exp();
        ErrorPropagatingFloat {
            abs_err: v.absolute_magnitude() * self.abs_err,
            value: v,
        }
        .truncate()
    }

    fn sin(&self) -> Self {
        ErrorPropagatingFloat {
            abs_err: self.abs_err * self.value.cos().absolute_magnitude(),
            value: self.value.sin(),
        }
        .truncate()
    }

    fn cos(&self) -> Self {
        ErrorPropagatingFloat {
            abs_err: self.abs_err * self.value.sin().absolute_magnitude(),
            value: self.value.cos(),
        }
        .truncate()
    }

    fn tan(&self) -> Self {
        let t = self.value.tan();
        let tt = t.absolute_magnitude();

        ErrorPropagatingFloat {
            abs_err: self.abs_err * (Magnitude::ONE + tt * tt),
            value: t,
        }
        .truncate()
    }

    fn asin(&self) -> Self {
        let t = self.value.asin();
        let denominator = (self.value.one() - self.value.clone() * &self.value).sqrt();
        ErrorPropagatingFloat {
            abs_err: self.abs_err / denominator.absolute_magnitude_lower(),
            value: t,
        }
        .truncate()
    }

    fn acos(&self) -> Self {
        let t = self.value.acos();
        let denominator = (self.value.one() - self.value.clone() * &self.value).sqrt();
        ErrorPropagatingFloat {
            abs_err: self.abs_err / denominator.absolute_magnitude_lower(),
            value: t,
        }
        .truncate()
    }

    fn atan2(&self, x: &Self) -> Self {
        let t = self.value.atan2(&x.value);
        // Algebraically the same first-order weights, without forming y/x or
        // squaring a large radius before canceling it in the final quotient.
        let radius = self.value.hypot(&x.value).absolute_magnitude_lower();
        ErrorPropagatingFloat {
            abs_err: (self.abs_err / radius) * (x.value.absolute_magnitude() / radius)
                + (x.abs_err / radius) * (self.value.absolute_magnitude() / radius),
            value: t,
        }
        .truncate()
    }

    fn sinh(&self) -> Self {
        ErrorPropagatingFloat {
            abs_err: self.abs_err * self.value.cosh().absolute_magnitude(),
            value: self.value.sinh(),
        }
        .truncate()
    }

    fn cosh(&self) -> Self {
        ErrorPropagatingFloat {
            abs_err: self.abs_err * self.value.sinh().absolute_magnitude(),
            value: self.value.cosh(),
        }
        .truncate()
    }

    fn tanh(&self) -> Self {
        let t = self.value.tanh();
        let derivative = t.one() - t.clone() * &t;
        ErrorPropagatingFloat {
            abs_err: self.abs_err * derivative.absolute_magnitude(),
            value: t,
        }
        .truncate()
    }

    fn asinh(&self) -> Self {
        let t = self.value.asinh();
        let denominator = self.value.hypot(&self.value.one());
        ErrorPropagatingFloat {
            abs_err: self.abs_err / denominator.absolute_magnitude_lower(),
            value: t,
        }
        .truncate()
    }

    fn acosh(&self) -> Self {
        let t = self.value.acosh();
        let denominator = (self.value.clone() - self.value.one()).sqrt()
            * (self.value.clone() + self.value.one()).sqrt();
        ErrorPropagatingFloat {
            abs_err: self.abs_err / denominator.absolute_magnitude_lower(),
            value: t,
        }
        .truncate()
    }

    fn atanh(&self) -> Self {
        let t = self.value.atanh();
        let denominator = self.value.one() - self.value.clone() * &self.value;
        ErrorPropagatingFloat {
            abs_err: self.abs_err / denominator.absolute_magnitude_lower(),
            value: t,
        }
        .truncate()
    }

    fn powf(&self, e: &Self) -> Self {
        if self.value.is_zero() {
            return ErrorPropagatingFloat {
                value: self.value.powf(&e.value),
                abs_err: Magnitude::ZERO,
            };
        }

        let r = self.value.powf(&e.value);
        let base_weight = self.abs_err / self.value.absolute_magnitude_lower();
        let exponent_weight = e.abs_err * self.value.norm().log().absolute_magnitude();
        ErrorPropagatingFloat {
            abs_err: (base_weight * e.value.absolute_magnitude() + exponent_weight)
                * r.absolute_magnitude(),
            value: r,
        }
        .truncate()
    }
}

impl<T: FloatLike + FixedPrecision> FixedPrecision for ErrorPropagatingFloat<T> {
    const BINARY_PRECISION: usize = T::BINARY_PRECISION;
    const DECIMAL_PRECISION: usize = T::DECIMAL_PRECISION;
}

#[cfg(test)]
mod hypot_tests {
    use crate::domains::float::{
        Complex, DoubleFloat, ErrorPropagatingFloat, Float, FloatLike, Real, RealLike, SingleFloat,
    };

    fn tracked<T: FloatLike>(value: T, accuracy: f64) -> ErrorPropagatingFloat<T> {
        ErrorPropagatingFloat::new_with_accuracy(value, accuracy)
    }

    fn scaled_hypot<T: Real + RealLike>(one: T) {
        let big = one.from_usize(10).pow(200);
        let tiny = big.inv();
        for scale in [big, tiny] {
            for sign in [1, -1] {
                let x = scale.clone() * one.from_i64(sign);
                let y = scale.clone() * one.from_usize(2);
                let expected = x.hypot(&y);
                let a = tracked(x, 220.);
                let b = tracked(y, 220.);
                let value = a.hypot(&b);
                assert!(value.is_finite());
                assert!(((value.get_num().clone() / &expected).to_f64() - 1.).abs() < 1e-14);
                assert!(value.get_absolute_error() > 0.);
                let reversed = b.hypot(&a);
                assert!(((reversed.get_num().clone() / &expected).to_f64() - 1.).abs() < 1e-14);
                let norm = Complex::new(a, b).norm();
                assert!(norm.re.is_finite());
                assert!(((norm.re.get_num().clone() / &expected).to_f64() - 1.).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn tracked_hypot_uses_existing_scaled_native_arithmetic() {
        scaled_hypot(1_f64);
        scaled_hypot(DoubleFloat::from(1.));
        scaled_hypot(Float::with_val(192, 1));
    }

    #[test]
    fn tracked_hypot_retains_local_uncertainty() {
        let value = tracked(3_f64, 6.).hypot(&tracked(4., 6.));
        assert_eq!(*value.get_num(), 5.);
        // Existing linear propagation can overestimate shared intermediates, but
        // must retain both first derivatives, 3/5 and 4/5.
        assert!(value.get_absolute_error() >= 1.4e-6 * (1. - 1e-14));
        let value = tracked(3_f64, 6.).hypot(&tracked(0., 6.));
        assert_eq!(*value.get_num(), 3.);
        assert!(value.get_absolute_error() >= 1e-6);
    }
}
