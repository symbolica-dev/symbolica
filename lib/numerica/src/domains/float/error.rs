use std::{
    f64::consts::LOG10_2,
    fmt::{self, Debug, Display, Formatter, LowerExp},
    hash::Hash,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use rand::Rng;

use super::{Complex, FixedPrecision, Float, FloatLike, Magnitude, Real, RealLike, SingleFloat};
use crate::domains::{InternalOrdering, integer::Integer, rational::Rational};

/// A float that estimates propagated numerical error.
///
/// Equality, hashing and ordering compare the value, ignoring its error.
/// Absolute errors use a compact binary magnitude with an extended exponent
/// range. Arithmetic and real elementary functions use finite-perturbation
/// estimates, with explicit handling of stationary points, poles and principal
/// argument cuts. These estimates are not certified enclosures or a check that
/// every perturbed input is in the domain; input correlations are not retained.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[derive(Copy, Clone)]
pub struct ErrorPropagatingFloat<T: FloatLike> {
    pub value: T,
    abs_err: Magnitude,
}

impl<T: FloatLike + InternalOrdering> InternalOrdering for ErrorPropagatingFloat<T> {
    fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.value.internal_cmp(&other.value)
    }
}

impl<T: FloatLike + Hash> Hash for ErrorPropagatingFloat<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl<T: FloatLike + Eq> Eq for ErrorPropagatingFloat<T> {}

impl ErrorPropagatingFloat<Float> {
    /// Write the precision, error, and canonical binary center without decimal rounding.
    pub(crate) fn write_lossless_component<W: fmt::Write>(&self, out: &mut W) -> fmt::Result {
        write!(out, "{}:", self.value.prec())?;
        for byte in self.abs_err.to_le_bytes() {
            write!(out, "{byte:02x}")?;
        }
        out.write_char(':')?;
        for byte in self.value.serialize() {
            write!(out, "{byte:02x}")?;
        }
        Ok(())
    }

    fn parse_lossless_component(s: &str) -> Result<Self, String> {
        fn decode_hex(s: &str) -> Result<Vec<u8>, String> {
            if s.is_empty() || !s.len().is_multiple_of(2) {
                return Err("Expected a nonempty sequence of hexadecimal byte pairs".into());
            }
            s.as_bytes()
                .chunks_exact(2)
                .map(|pair| {
                    let digit = |c: u8| match c {
                        b'0'..=b'9' => Ok(c - b'0'),
                        b'a'..=b'f' => Ok(c - b'a' + 10),
                        _ => Err("Expected lowercase hexadecimal bytes".to_owned()),
                    };
                    Ok(digit(pair[0])? * 16 + digit(pair[1])?)
                })
                .collect()
        }
        let mut fields = s.split(':');
        let precision = fields
            .next()
            .unwrap_or("")
            .parse::<u32>()
            .map_err(|_| "Invalid lossless float precision".to_owned())?;
        let error = decode_hex(fields.next().unwrap_or(""))?;
        let value = decode_hex(fields.next().unwrap_or(""))?;
        if fields.next().is_some() {
            return Err("Too many fields in lossless float".into());
        }
        let error = Magnitude::from_le_bytes(
            error
                .try_into()
                .map_err(|_| "A lossless float error must contain eight bytes".to_owned())?,
        )
        .map_err(str::to_owned)?;
        Ok(Self::new_with_error_magnitude(
            Float::try_deserialize(&value, precision)?,
            error,
        ))
    }

    /// Serialize a scalar as `0f[bits:error_hex:center_hex]` for exact text round trips.
    /// Centers use the same canonical zero and NaN as binary serialization.
    pub fn to_lossless_string(&self) -> String {
        let mut out = String::from("0f[");
        self.write_lossless_component(&mut out).unwrap();
        out.push(']');
        out
    }

    pub fn is_negative(&self) -> bool {
        self.value.is_negative()
    }

    /// Parse a decimal float, optionally followed by a backtick and decimal
    /// precision. A zero with an empty backtick suffix denotes an exact zero.
    /// Two backticks introduce an absolute decimal accuracy.
    pub fn parse(s: &str, prec: Option<u32>) -> Result<Self, String> {
        let s = s.trim();
        if s.starts_with("0f[") || s.starts_with("-0f[") {
            let negative = s.starts_with('-');
            let literal = s.strip_prefix('-').unwrap_or(s);
            let payload = literal
                .strip_prefix("0f[")
                .and_then(|s| s.strip_suffix(']'))
                .ok_or_else(|| "Unterminated lossless float literal".to_owned())?;
            let value = Self::parse_lossless_component(payload)?;
            if prec.is_some_and(|prec| prec != value.value.prec()) {
                return Err("A lossless float literal already specifies its precision".into());
            }
            return Ok(if negative { -value } else { value });
        }
        if let Some((value, accuracy)) = s.split_once("``") {
            let accuracy = accuracy
                .parse::<f64>()
                .map_err(|e| format!("Invalid accuracy: {e}"))?;
            if !accuracy.is_finite() {
                return Err("Accuracy must be finite".into());
            }
            let value = Float::parse(value, prec)?;
            return Ok(Self::new_with_accuracy(value, accuracy));
        }
        let value = if let Some((literal, suffix)) = s.split_once('`') {
            if !suffix.is_empty() {
                let digits = suffix
                    .parse::<f64>()
                    .map_err(|e| format!("Invalid precision: {e}"))?;
                if !digits.is_finite() {
                    return Err("Precision must be finite".into());
                }
                // Nonpositive tracked precision describes an uncertainty at
                // least as large as the central value; storage still needs bits.
                let bits = match prec {
                    Some(bits) => bits,
                    None if digits > 0. => Float::decimal_digits_to_bits(digits)?,
                    None => 53,
                };
                let value = Float::parse(literal, Some(bits))?;
                if prec.is_none() {
                    return Ok(Self::new(value, digits));
                }
                value
            } else {
                let value = Float::parse(literal, prec)?;
                if prec.is_none() && value.is_fully_zero() {
                    return Ok(Self::new_with_absolute_error(value, 0.));
                }
                value
            }
        } else {
            Float::parse(s, prec)?
        };
        let digits = LOG10_2 * value.prec() as f64;
        Ok(Self::new(value, digits))
    }
}

impl Complex<ErrorPropagatingFloat<Float>> {
    /// Parse a lossless complex coefficient. Both component precisions and errors
    /// are restored directly, without arithmetic on its real and imaginary parts.
    pub fn parse_lossless(s: &str) -> Result<Self, String> {
        let s = s.trim();
        let negative = s.starts_with('-');
        let literal = s.strip_prefix('-').unwrap_or(s);
        let payload = literal
            .strip_prefix("0f[")
            .and_then(|s| s.strip_suffix(']'))
            .ok_or_else(|| "Invalid lossless complex float literal".to_owned())?;
        let mut components = payload.split(';');
        let real =
            ErrorPropagatingFloat::parse_lossless_component(components.next().unwrap_or(""))?;
        let imaginary = if let Some(component) = components.next() {
            ErrorPropagatingFloat::parse_lossless_component(component)?
        } else {
            ErrorPropagatingFloat::exact_zero(real.value.prec())
        };
        if components.next().is_some() {
            return Err("A lossless complex float has at most two components".into());
        }
        let value = Self::new(real, imaginary);
        Ok(if negative { -value } else { value })
    }
}

impl From<f64> for ErrorPropagatingFloat<Float> {
    fn from(value: f64) -> Self {
        let scale = if value == 0. {
            Magnitude::from_f64(1.)
        } else {
            Magnitude::from_f64(value)
        };
        ErrorPropagatingFloat {
            abs_err: scale.scale_pow2(-52),
            value: value.into(),
        }
    }
}

impl<T: RealLike> From<T> for ErrorPropagatingFloat<T> {
    fn from(value: T) -> Self {
        let scale = if value.is_fully_zero() {
            Magnitude::from_f64(1.)
        } else {
            value.absolute_magnitude()
        };
        ErrorPropagatingFloat {
            abs_err: scale * Self::epsilon_magnitude(&value),
            value,
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

impl<T: RealLike> Add<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: &Self) -> Self::Output {
        ErrorPropagatingFloat {
            abs_err: self.abs_err + rhs.abs_err,
            value: self.value + &rhs.value,
        }
        .with_roundoff()
    }
}

impl<T: RealLike> Add<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl<T: RealLike> Sub<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: &Self) -> Self::Output {
        ErrorPropagatingFloat {
            abs_err: self.abs_err + rhs.abs_err,
            value: self.value - &rhs.value,
        }
        .with_roundoff()
    }
}

impl<T: RealLike> Sub<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
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
        if self.is_exact_zero() && rhs.value.is_finite()
            || rhs.is_exact_zero() && self.value.is_finite()
        {
            return Self {
                value: self.value * &rhs.value,
                abs_err: Magnitude::ZERO,
            };
        }
        if let Some(negative) = rhs.exact_unit_sign() {
            return if negative { -self } else { self };
        }
        if let Some(negative) = self.exact_unit_sign() {
            return if negative { -rhs.clone() } else { rhs.clone() };
        }
        let r = rhs.value.absolute_magnitude();
        let s = self.value.absolute_magnitude();
        let value = self.value * &rhs.value;
        // Include the product of the two errors, including when only one
        // central value is zero. Factoring saves one magnitude multiply.
        let mut abs_err = self.abs_err * (r + rhs.abs_err) + rhs.abs_err * s;
        if value.is_fully_zero() && !r.is_zero() && !s.is_zero() {
            // Nonzero central inputs may underflow before a result-relative
            // roundoff estimate can see them.
            abs_err += r * s;
        }

        ErrorPropagatingFloat { value, abs_err }.with_roundoff()
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
        .truncate()
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
            value: self.value / rhs,
        }
        .truncate()
    }
}

impl<T: RealLike + From<f64>> Add<f64> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: f64) -> Self::Output {
        self + &T::from(rhs).into()
    }
}

impl<T: RealLike + From<f64>> Sub<f64> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: f64) -> Self::Output {
        self - &T::from(rhs).into()
    }
}

impl<T: RealLike + From<f64>> Mul<f64> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: f64) -> Self::Output {
        self * &T::from(rhs).into()
    }
}

impl<T: RealLike + From<f64>> Div<f64> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: f64) -> Self::Output {
        self / &T::from(rhs).into()
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
        if let Some(negative) = rhs.exact_unit_sign() {
            return if negative { -self } else { self };
        }
        let (mut abs_err, quotient) = self.division_error(rhs);
        let finite_denominator = rhs.value.is_finite();
        let value = self.value / &rhs.value;
        if value.is_fully_zero() && finite_denominator {
            abs_err += quotient;
        }
        Self { value, abs_err }.with_roundoff()
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
        self.value += &rhs.value;
        self.abs_err = self.abs_err + rhs.abs_err;
        self.add_roundoff();
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
        self.value -= &rhs.value;
        self.abs_err = self.abs_err + rhs.abs_err;
        self.add_roundoff();
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
        if self.is_exact_zero() && rhs.value.is_finite()
            || rhs.is_exact_zero() && self.value.is_finite()
        {
            self.value *= &rhs.value;
            self.abs_err = Magnitude::ZERO;
            return;
        }
        if let Some(negative) = rhs.exact_unit_sign() {
            if negative {
                self.value = -self.value.clone();
            }
            return;
        }
        if let Some(negative) = self.exact_unit_sign() {
            *self = if negative { -rhs.clone() } else { rhs.clone() };
            return;
        }
        let r = rhs.value.absolute_magnitude();
        let s = self.value.absolute_magnitude();
        self.abs_err = self.abs_err * (r + rhs.abs_err) + rhs.abs_err * s;
        self.value *= &rhs.value;
        if self.value.is_fully_zero() && !r.is_zero() && !s.is_zero() {
            self.abs_err += r * s;
        }
        self.add_roundoff();
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
        if let Some(negative) = rhs.exact_unit_sign() {
            if negative {
                self.value = -self.value.clone();
            }
            return;
        }
        let (abs_err, quotient) = self.division_error(rhs);
        self.abs_err = abs_err;
        self.value /= &rhs.value;
        if self.value.is_fully_zero() && rhs.value.is_finite() {
            self.abs_err += quotient;
        }
        self.add_roundoff();
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
        let accuracy = Magnitude::from_decimal_accuracy(prec);
        ErrorPropagatingFloat {
            abs_err: if r.is_zero() { accuracy } else { accuracy * r },
            value,
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

    #[inline]
    fn epsilon_magnitude(value: &T) -> Magnitude {
        if value.fixed_precision() {
            // Preserve each fixed domain's epsilon, including exact rationals.
            Magnitude::from_f64(value.get_epsilon())
        } else {
            Magnitude::from_f64(1.).scale_pow2(-(value.get_precision() as i64))
        }
    }

    #[inline]
    fn roundoff(value: &T) -> Magnitude {
        (value.absolute_magnitude() * Self::epsilon_magnitude(value)).max(value.minimum_roundoff())
    }

    /// Include the rounding of a computed derivative or function value before
    /// using it as an upper magnitude in a finite-perturbation estimate.
    #[inline]
    fn rounded_magnitude(value: &T) -> Magnitude {
        value.absolute_magnitude() + Self::roundoff(value)
    }

    #[inline]
    fn bounded_error(estimate: Magnitude, limit: Magnitude) -> Magnitude {
        // Both arguments are independently valid bounds. A range overflow in
        // one formula (for example 0 * infinity) may leave the other usable.
        if limit.is_nan() {
            limit
        } else if !estimate.is_nan() && estimate < limit {
            estimate
        } else {
            limit
        }
    }

    #[inline]
    fn scaled_error(error: Magnitude, factor: Magnitude) -> Magnitude {
        if error.is_zero() {
            Magnitude::ZERO
        } else {
            error * factor
        }
    }

    /// Bound exp(delta)-1, or its remainder after subtracting delta. For small
    /// delta the quadratic bound avoids both cancellation and a logarithm.
    fn exponential_change(delta: Magnitude, remainder: bool) -> Magnitude {
        if delta <= Magnitude::ONE {
            let quadratic = delta * delta;
            if remainder {
                quadratic
            } else {
                delta + quadratic
            }
        } else {
            let logarithm = (delta.to_f64_saturating() * std::f64::consts::LOG2_E)
                .next_up()
                .next_up();
            Magnitude::from_log2(logarithm)
        }
    }

    fn circular_inverse_error(&self) -> Magnitude {
        let center = self.value.absolute_magnitude();
        if center > Magnitude::ONE || center.is_nan() || self.abs_err.is_nan() {
            return Magnitude::NAN;
        }
        if self.abs_err.is_zero() {
            return Magnitude::ZERO;
        }
        let pi = Magnitude::from_f64(std::f64::consts::PI.next_up());
        // The inverse circular functions are globally 1/2-Hölder on [-1,1].
        // This covers their finite endpoint changes when a derivative diverges;
        // it does not claim that perturbations outside the domain are defined.
        let limit = Self::bounded_error(pi * self.abs_err.sqrt(), pi);
        let outer = center + self.abs_err;
        let clearance = Magnitude::ONE.sub_lower(outer * outer);
        if clearance.is_zero() {
            limit
        } else {
            Self::bounded_error(self.abs_err / clearance.sqrt().next_down(), limit)
        }
    }

    /// Bound both |x|^n and the finite change caused by |h| <= error. Propagating
    /// the difference alongside the power avoids subtracting nearly equal
    /// magnitudes, and retains the full u64 exponent instead of converting it.
    fn power_magnitudes(
        mut value: Magnitude,
        mut error: Magnitude,
        mut n: u64,
    ) -> (Magnitude, Magnitude) {
        let mut power = Magnitude::ONE;
        let mut change = Magnitude::ZERO;
        while n != 0 {
            if n & 1 != 0 {
                change =
                    Self::scaled_error(change, value + error) + Self::scaled_error(error, power);
                power *= value;
            }
            n >>= 1;
            if n != 0 {
                error = Self::scaled_error(error, value.scale_pow2(1) + error);
                value *= value;
            }
        }
        (power, change)
    }

    #[inline]
    fn exact_unit_sign(&self) -> Option<bool> {
        if !self.abs_err.is_zero() || self.value.has_error() {
            None
        } else if self.value.is_one() {
            Some(false)
        } else if !self.value.is_fully_zero() && (-self.value.clone()).is_one() {
            Some(true)
        } else {
            None
        }
    }

    /// Propagate a quotient using the distance from the denominator's
    /// uncertainty to zero. The quotient magnitude uses input bounds rather
    /// than the rounded central quotient, whose rounding could understate it.
    #[inline]
    fn division_error(&self, rhs: &Self) -> (Magnitude, Magnitude) {
        if rhs.abs_err.is_zero()
            && rhs.value.real_classify() == Some(std::num::FpCategory::Infinite)
            && self.value.is_finite()
        {
            return (Magnitude::ZERO, Magnitude::ZERO);
        }
        let denominator = rhs.value.absolute_magnitude_lower();
        let quotient = self.value.absolute_magnitude() / denominator;
        if rhs.abs_err.is_zero() {
            return (self.abs_err / denominator, quotient);
        }
        let clearance = denominator.sub_lower(rhs.abs_err);
        let error = if clearance.is_zero() {
            Magnitude::INFINITY
        } else {
            (self.abs_err + quotient * rhs.abs_err) / clearance
        };
        (error, quotient)
    }

    #[inline]
    fn add_roundoff(&mut self) {
        if !self.value.is_fully_zero() {
            self.abs_err += Self::roundoff(&self.value);
        }
    }

    #[inline]
    fn with_roundoff(mut self) -> Self {
        self.add_roundoff();
        self
    }

    /// Construct zero with no error and a binary precision hint.
    pub fn exact_zero(prec: u32) -> Self {
        Self {
            value: T::zero_with_prec_hint(prec),
            abs_err: Magnitude::ZERO,
        }
    }

    pub fn is_exact(&self) -> bool {
        self.abs_err.is_zero()
    }

    pub fn is_exact_zero(&self) -> bool {
        self.is_exact() && self.value.is_fully_zero()
    }

    /// Whether zero lies within the tracked absolute error of the central value.
    /// This is a numerical test, not a proof of exact equality.
    pub fn is_zero_within_error(&self) -> bool {
        self.value.absolute_magnitude() <= self.abs_err
    }

    /// Conservatively detect an uncertain input whose neighborhood can contain
    /// zero, for functions with a pole there. Exact zero retains the underlying
    /// domain's undefined-value behavior instead.
    #[inline]
    fn uncertainty_reaches_zero(&self) -> bool {
        !self.abs_err.is_zero() && self.abs_err >= self.value.absolute_magnitude_lower()
    }

    /// Construct a value with an absolute error magnitude.
    pub fn new_with_absolute_error(value: T, abs_err: f64) -> Self {
        Self {
            value,
            abs_err: Magnitude::from_f64(abs_err),
        }
    }

    /// Return the error rounded upward to `f64`. Positive underflow saturates
    /// to the smallest subnormal and overflow to infinity. For checked conversion,
    /// use `get_error_magnitude().to_f64()`, which returns `None` outside that range.
    pub fn get_absolute_error(&self) -> f64 {
        self.abs_err.to_f64_saturating()
    }

    /// Return the lossless stored error magnitude.
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
            Some(r.log2_ratio(self.abs_err) * LOG10_2)
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
        self.truncate_error();
        self
    }

    #[inline]
    fn truncate_error(&mut self) {
        // Every finite-precision result has a roundoff floor, including
        // negative results and results computed with variable precision.
        if !self.value.is_fully_zero() {
            self.abs_err = self.abs_err.max(Self::roundoff(&self.value));
        }
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

    pub fn into_value(self) -> T {
        self.value
    }
}

impl<T: RealLike> fmt::Display for ErrorPropagatingFloat<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if let Some(p) = self.get_precision() {
            if p < 0. {
                f.write_str("0.")
            } else if p.is_finite() {
                f.write_fmt(format_args!("{0:.1$}", self.value, p as usize))
            } else {
                Display::fmt(&self.value, f)
            }
        } else {
            f.write_str("0.")
        }
    }
}

impl<T: RealLike> Debug for ErrorPropagatingFloat<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.value, f)?;

        if let Some(p) = self.get_precision() {
            f.write_fmt(format_args!("`{p:.2}"))
        } else if self.is_exact_zero() {
            f.write_str("`")
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
    fn has_error(&self) -> bool {
        !self.abs_err.is_zero()
    }

    fn real_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.real_cmp(&other.value)
    }

    fn real_classify(&self) -> Option<std::num::FpCategory> {
        self.value.real_classify()
    }

    fn needs_rescaling(&self) -> bool {
        self.value.needs_rescaling()
    }

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
            abs_err: Magnitude::ZERO,
        }
    }

    fn new_zero() -> Self {
        ErrorPropagatingFloat {
            value: T::new_zero(),
            abs_err: Magnitude::ZERO,
        }
    }

    fn one(&self) -> Self {
        ErrorPropagatingFloat {
            value: self.value.one(),
            // This constructs the exactly representable algebraic unit. It
            // does not remove uncertainty from a user-provided floating one.
            abs_err: Magnitude::ZERO,
        }
    }

    fn pow(&self, e: u64) -> Self {
        if e == 0 {
            return self.one();
        }
        if e == 1 {
            return self.clone();
        }
        if self.is_exact_zero() {
            return if e & 1 != 0 {
                self.clone()
            } else {
                Self {
                    value: self.value.clone() * &self.value,
                    abs_err: Magnitude::ZERO,
                }
            };
        }
        if let Some(negative) = self.exact_unit_sign() {
            return if negative && e & 1 != 0 {
                self.clone()
            } else {
                self.one()
            };
        }
        let value = self.value.pow(e);
        let (power, mut abs_err) =
            Self::power_magnitudes(self.value.absolute_magnitude(), self.abs_err, e);
        if value.is_fully_zero() && !self.value.is_fully_zero() {
            abs_err += power;
        }
        Self { value, abs_err }.with_roundoff()
    }

    fn inv(&self) -> Self {
        if self.exact_unit_sign().is_some() {
            return self.clone();
        }
        let r = self.value.inv();
        let input = self.value.absolute_magnitude_lower();
        let clearance = input.sub_lower(self.abs_err);
        let mut abs_err = if !self.abs_err.is_zero() && clearance.is_zero() {
            Magnitude::INFINITY
        } else {
            (self.abs_err / input) / clearance
        };
        if r.is_fully_zero() && self.value.is_finite() {
            abs_err += Magnitude::ONE / input;
        }
        ErrorPropagatingFloat { abs_err, value: r }.with_roundoff()
    }

    /// Convert from a `usize`.
    fn from_usize(&self, a: usize) -> Self {
        let v = self.value.from_usize(a);
        let r = v.absolute_magnitude();
        if r.is_zero() {
            ErrorPropagatingFloat {
                value: v,
                abs_err: Magnitude::from_f64(1.).scale_pow2(-(self.value.get_precision() as i64)),
            }
        } else {
            ErrorPropagatingFloat {
                value: v,
                abs_err: r.scale_pow2(-(self.value.get_precision() as i64)),
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
                abs_err: Magnitude::from_f64(1.).scale_pow2(-(self.value.get_precision() as i64)),
            }
        } else {
            ErrorPropagatingFloat {
                value: v,
                abs_err: r.scale_pow2(-(self.value.get_precision() as i64)),
            }
        }
    }

    fn get_precision(&self) -> u32 {
        // return the precision of the underlying float instead
        // of the current tracked precision
        self.value.get_precision()
    }

    fn set_precision(&mut self, precision: u32) {
        let previous = self.value.get_precision();
        self.value.set_precision(precision);
        if self.value.get_precision() < previous && !self.value.is_fully_zero() {
            // Reducing storage precision introduces fresh roundoff. Increasing it
            // cannot recover digits already lost to the tracked uncertainty.
            self.abs_err = self.abs_err + Self::roundoff(&self.value);
        }
    }

    fn get_epsilon(&self) -> f64 {
        Self::epsilon_magnitude(&self.value).to_f64_saturating()
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
        let value = self.value.from_rational(rat);
        let abs_err = if rat.is_zero() {
            Magnitude::ZERO
        } else {
            Self::roundoff(&value)
        };
        Self { value, abs_err }
    }
}

impl<T: RealLike> RealLike for ErrorPropagatingFloat<T> {
    fn absolute_magnitude(&self) -> Magnitude {
        self.value.absolute_magnitude()
    }

    fn absolute_magnitude_lower(&self) -> Magnitude {
        self.value.absolute_magnitude_lower()
    }

    fn minimum_roundoff(&self) -> Magnitude {
        self.value.minimum_roundoff()
    }

    fn zero_with_prec_hint(prec: u32) -> Self {
        Self::exact_zero(prec)
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
            abs_err: v
                .absolute_magnitude()
                .scale_pow2(-(self.value.get_precision() as i64)),
            value: v,
        }
    }

    fn e(&self) -> Self {
        let v = self.value.e();
        ErrorPropagatingFloat {
            abs_err: v
                .absolute_magnitude()
                .scale_pow2(-(self.value.get_precision() as i64)),
            value: v,
        }
    }

    fn euler(&self) -> Self {
        let v = self.value.euler();
        ErrorPropagatingFloat {
            abs_err: v
                .absolute_magnitude()
                .scale_pow2(-(self.value.get_precision() as i64)),
            value: v,
        }
    }

    fn phi(&self) -> Self {
        let v = self.value.phi();
        ErrorPropagatingFloat {
            abs_err: v
                .absolute_magnitude()
                .scale_pow2(-(self.value.get_precision() as i64)),
            value: v,
        }
    }

    #[inline(always)]
    fn i(&self) -> Option<Self> {
        Some(ErrorPropagatingFloat {
            value: self.value.i()?,
            abs_err: Magnitude::from_f64(1.).scale_pow2(-(self.value.get_precision() as i64)),
        })
    }

    fn conj(&self) -> Self {
        ErrorPropagatingFloat {
            abs_err: self.abs_err,
            value: self.value.conj(),
        }
    }

    fn copy_sign(&self, sign: &Self) -> Self {
        Self {
            value: self.value.copy_sign(&sign.value),
            abs_err: if sign.abs_err.is_nan() {
                Magnitude::NAN
            } else if sign.uncertainty_reaches_zero() {
                self.abs_err + self.value.absolute_magnitude().scale_pow2(1)
            } else {
                self.abs_err
            },
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
        // Use the finite change in the squared radius, divided by the sum of
        // radii. The displacement also bounds the result globally, including
        // transverse perturbations at a stationary first derivative.
        let radius = value
            .absolute_magnitude_lower()
            .sub_lower(Self::roundoff(&value));
        let displacement = self.abs_err + other.abs_err;
        let denominator = radius + radius.sub_lower(displacement);
        let denominator = denominator.next_down();
        let estimate = Self::scaled_error(
            self.abs_err / denominator,
            self.value.absolute_magnitude().scale_pow2(1) + self.abs_err,
        ) + Self::scaled_error(
            other.abs_err / denominator,
            other.value.absolute_magnitude().scale_pow2(1) + other.abs_err,
        );
        ErrorPropagatingFloat {
            abs_err: Self::bounded_error(estimate, displacement),
            value,
        }
        .with_roundoff()
    }

    fn sqrt_complex_branch_cut(&self, imaginary: &Self) -> Option<(Self, Self)> {
        if !self.value.is_finite()
            || !imaginary.value.is_finite()
            || self.abs_err.is_nan()
            || imaginary.abs_err.is_nan()
        {
            return None;
        }
        let real_reaches_zero =
            !self.abs_err.is_zero() && self.abs_err >= self.value.absolute_magnitude_lower();
        let imaginary_reaches_zero = !imaginary.abs_err.is_zero()
            && imaginary.abs_err >= imaginary.value.absolute_magnitude_lower();
        let reaches_negative_axis = self.value.real_cmp(&self.value.zero())
            != Some(std::cmp::Ordering::Greater)
            || real_reaches_zero;
        if !(imaginary.is_exact_zero() && real_reaches_zero
            || imaginary_reaches_zero && reaches_negative_axis)
        {
            return None;
        }

        // Principal roots lie in Re >= 0 and have modulus <= sqrt(|Re z| + |Im z|).
        // This bound covers both branch lips, including a jump across the negative
        // real axis; a derivative evaluated at the center cannot cover that jump.
        // It encloses the supplied component-error rectangle, but does not
        // certify the error estimates produced by the preceding computation.
        let re_limit = self.value.absolute_magnitude() + self.abs_err;
        let im_limit = imaginary.value.absolute_magnitude() + imaginary.abs_err;
        let limit = (re_limit + im_limit).sqrt();
        let center = super::Complex::new(self.value.clone(), imaginary.value.clone()).sqrt();
        let re_error = limit.max(center.re.absolute_magnitude());
        let im_error = limit + center.im.absolute_magnitude();
        Some((
            Self {
                value: center.re,
                abs_err: re_error,
            }
            .truncate(),
            Self {
                value: center.im,
                abs_err: im_error,
            }
            .truncate(),
        ))
    }

    fn sqrt(&self) -> Self {
        if self.value.is_fully_zero() {
            return Self {
                value: self.value.sqrt(),
                abs_err: self.abs_err.sqrt(),
            };
        }
        let v = self.value.sqrt();
        let r = v.absolute_magnitude_lower().sub_lower(Self::roundoff(&v));
        let input = self.value.absolute_magnitude_lower();
        // For defined real perturbations, |sqrt(x+h)-sqrt(x)| <= sqrt(|h|).
        // This remains finite at zero, but does not certify that x+h >= 0.
        let limit = self.abs_err.sqrt();
        let abs_err = if self.abs_err >= input {
            limit
        } else {
            // For delta < x, the largest change is toward zero:
            // delta / (sqrt(x) + sqrt(x-delta)). Only the bounded correction
            // uses f64; neither the value nor its error loses exponent range.
            let ratio = (self.abs_err / input).to_f64_saturating();
            let correction =
                Magnitude::from_f64((1. / (1. + (1. - ratio).sqrt())).next_up().next_up());
            let estimate = (self.abs_err / r) * correction;
            if estimate > limit { limit } else { estimate }
        };

        ErrorPropagatingFloat { abs_err, value: v }.with_roundoff()
    }

    fn log(&self) -> Self {
        let r = self.value.log();
        let clearance = self
            .value
            .absolute_magnitude_lower()
            .sub_lower(self.abs_err);
        ErrorPropagatingFloat {
            abs_err: if !self.abs_err.is_zero() && clearance.is_zero() {
                Magnitude::INFINITY
            } else {
                // The real logarithm's derivative is largest at the endpoint
                // closest to zero. This finite-error estimate avoids log calls.
                self.abs_err / clearance
            },
            value: r,
        }
        .with_roundoff()
    }

    fn log1p(&self) -> Self {
        let value = self.value.log1p();
        let denominator = self.value.one() + &self.value;
        let clearance = denominator
            .absolute_magnitude_lower()
            .sub_lower(Self::roundoff(&denominator))
            .sub_lower(self.abs_err);
        let abs_err = if value.absolute_magnitude().is_nan() {
            Magnitude::NAN
        } else if self.abs_err.is_zero() {
            Magnitude::ZERO
        } else if clearance.is_zero() {
            Magnitude::INFINITY
        } else {
            self.abs_err / clearance
        };
        Self { value, abs_err }.with_roundoff()
    }

    fn exp(&self) -> Self {
        let v = self.value.exp();
        let mut abs_err = Self::scaled_error(
            Self::exponential_change(self.abs_err, false),
            Self::rounded_magnitude(&v),
        );
        if v.is_fully_zero() && self.value.is_finite() {
            // exp of a finite real value is positive even if the backend
            // underflows. Its rounding floor also bounds this lost value.
            abs_err += v.minimum_roundoff().max(Magnitude::MIN_POSITIVE);
        }
        ErrorPropagatingFloat { abs_err, value: v }.with_roundoff()
    }

    fn sin(&self) -> Self {
        let value = self.value.sin();
        let derivative = Self::rounded_magnitude(&self.value.cos());
        let estimate = Self::scaled_error(self.abs_err, derivative)
            + (self.abs_err * self.abs_err).scale_pow2(-1);
        let limit = if self.abs_err.is_nan() {
            Magnitude::NAN
        } else {
            Magnitude::ONE + value.absolute_magnitude()
        };
        ErrorPropagatingFloat {
            abs_err: Self::bounded_error(estimate, limit),
            value,
        }
        .with_roundoff()
    }

    fn cos(&self) -> Self {
        let value = self.value.cos();
        let derivative = Self::rounded_magnitude(&self.value.sin());
        let estimate = Self::scaled_error(self.abs_err, derivative)
            + (self.abs_err * self.abs_err).scale_pow2(-1);
        let limit = if self.abs_err.is_nan() {
            Magnitude::NAN
        } else {
            Magnitude::ONE + value.absolute_magnitude()
        };
        ErrorPropagatingFloat {
            abs_err: Self::bounded_error(estimate, limit),
            value,
        }
        .with_roundoff()
    }

    fn tan(&self) -> Self {
        let t = self.value.tan();
        // cos is 1-Lipschitz. A positive clearance proves that no pole is
        // crossed, and bounds the largest secant-squared derivative.
        let cosine = self.value.cos();
        let clearance = cosine
            .absolute_magnitude_lower()
            .sub_lower(Self::roundoff(&cosine))
            .sub_lower(self.abs_err);
        ErrorPropagatingFloat {
            abs_err: if self.abs_err.is_zero() {
                Magnitude::ZERO
            } else if clearance.is_zero() {
                Magnitude::INFINITY
            } else {
                (self.abs_err / clearance) / clearance
            },
            value: t,
        }
        .with_roundoff()
    }

    fn asin(&self) -> Self {
        let t = self.value.asin();
        ErrorPropagatingFloat {
            abs_err: self.circular_inverse_error(),
            value: t,
        }
        .with_roundoff()
    }

    fn acos(&self) -> Self {
        let t = self.value.acos();
        ErrorPropagatingFloat {
            abs_err: self.circular_inverse_error(),
            value: t,
        }
        .with_roundoff()
    }

    fn atan2(&self, x: &Self) -> Self {
        let value = self.value.atan2(&x.value);
        let displacement = self.abs_err + x.abs_err;
        if displacement.is_zero() {
            return Self {
                value,
                abs_err: Magnitude::ZERO,
            }
            .with_roundoff();
        }
        let limit = if displacement.is_nan() {
            Magnitude::NAN
        } else {
            Magnitude::from_f64(std::f64::consts::PI.next_up()) + value.absolute_magnitude()
        };
        let norm = self.value.hypot(&x.value);
        let radius = norm
            .absolute_magnitude_lower()
            .sub_lower(Self::roundoff(&norm));
        let clearance = radius.sub_lower(displacement);
        let crosses_cut = self.uncertainty_reaches_zero()
            && (x.value.real_cmp(&x.value.zero()) != Some(std::cmp::Ordering::Greater)
                || x.uncertainty_reaches_zero());
        let abs_err = if crosses_cut || clearance.is_zero() {
            // A principal argument can jump from +pi to -pi across this cut.
            // The finite range bound also covers neighborhoods of the origin.
            limit
        } else {
            let estimate = Self::scaled_error(
                self.abs_err / clearance,
                (x.value.absolute_magnitude() + x.abs_err) / clearance,
            ) + Self::scaled_error(
                x.abs_err / clearance,
                (self.value.absolute_magnitude() + self.abs_err) / clearance,
            );
            Self::bounded_error(estimate, limit)
        };
        Self { value, abs_err }.with_roundoff()
    }

    fn sinh(&self) -> Self {
        let value = self.value.sinh();
        let derivative = Self::rounded_magnitude(&self.value.cosh());
        let remainder_scale = Self::rounded_magnitude(&value) + derivative;
        ErrorPropagatingFloat {
            abs_err: Self::scaled_error(self.abs_err, derivative)
                + Self::scaled_error(
                    Self::exponential_change(self.abs_err, true),
                    remainder_scale,
                ),
            value,
        }
        .with_roundoff()
    }

    fn cosh(&self) -> Self {
        let value = self.value.cosh();
        let derivative = Self::rounded_magnitude(&self.value.sinh());
        let remainder_scale = Self::rounded_magnitude(&value) + derivative;
        ErrorPropagatingFloat {
            abs_err: Self::scaled_error(self.abs_err, derivative)
                + Self::scaled_error(
                    Self::exponential_change(self.abs_err, true),
                    remainder_scale,
                ),
            value,
        }
        .with_roundoff()
    }

    fn tanh(&self) -> Self {
        let t = self.value.tanh();
        let sech = Self::rounded_magnitude(&self.value.sech());
        // |(log sech^2)'| <= 2, while tanh is globally 1-Lipschitz.
        let derivative_growth =
            Magnitude::ONE + Self::exponential_change(self.abs_err.scale_pow2(1), false);
        let estimate = Self::scaled_error(self.abs_err, sech * sech * derivative_growth);
        let limit = Self::bounded_error(self.abs_err, Magnitude::ONE + t.absolute_magnitude());
        ErrorPropagatingFloat {
            abs_err: if self.abs_err.is_nan() {
                Magnitude::NAN
            } else {
                Self::bounded_error(estimate, limit)
            },
            value: t,
        }
        .with_roundoff()
    }

    fn asinh(&self) -> Self {
        let t = self.value.asinh();
        let closest = self
            .value
            .absolute_magnitude_lower()
            .sub_lower(self.abs_err);
        // sqrt(1+x^2) >= max(1,|x|), without a square that can overflow.
        let denominator = closest.max(Magnitude::ONE);
        ErrorPropagatingFloat {
            abs_err: self.abs_err / denominator,
            value: t,
        }
        .with_roundoff()
    }

    fn acosh(&self) -> Self {
        let t = self.value.acosh();
        let closest = self
            .value
            .absolute_magnitude_lower()
            .sub_lower(self.abs_err);
        let lower = closest.sub_lower(Magnitude::ONE).sqrt().next_down();
        let upper = (closest + Magnitude::ONE).next_down().sqrt().next_down();
        // For defined real perturbations, the largest change at the domain
        // boundary is acosh(1+delta) <= sqrt(2*delta).
        let limit = self.abs_err.scale_pow2(1).sqrt();
        let estimate = if self.abs_err.is_zero() {
            Magnitude::ZERO
        } else {
            self.abs_err / lower / upper
        };
        ErrorPropagatingFloat {
            abs_err: if t.absolute_magnitude().is_nan() {
                Magnitude::NAN
            } else {
                Self::bounded_error(estimate, limit)
            },
            value: t,
        }
        .with_roundoff()
    }

    fn atanh(&self) -> Self {
        let t = self.value.atanh();
        let outer = self.value.absolute_magnitude() + self.abs_err;
        let clearance = Magnitude::ONE.sub_lower(outer * outer);
        ErrorPropagatingFloat {
            abs_err: if t.absolute_magnitude().is_nan() {
                Magnitude::NAN
            } else if self.abs_err.is_zero() {
                Magnitude::ZERO
            } else if clearance.is_zero() {
                Magnitude::INFINITY
            } else {
                self.abs_err / clearance
            },
            value: t,
        }
        .with_roundoff()
    }

    fn powf(&self, e: &Self) -> Self {
        if e.is_exact() {
            if e.value.is_fully_zero() {
                return self.one();
            }
            if e.value.is_one() {
                return self.clone();
            }
            if e.value == e.value.one() / e.value.from_usize(2) {
                return self.sqrt();
            }
            // Keep common integer powers on the faster finite-change kernel.
            // Verify in the original domain before trusting the f64 conversion.
            let exponent = e.value.to_f64();
            if exponent.abs() <= (1u64 << 53) as f64 && exponent.fract() == 0. {
                let integer = exponent as i64;
                if e.value == e.value.from_i64(integer) {
                    return if integer < 0 {
                        self.inv().pow(integer.unsigned_abs())
                    } else {
                        self.pow(integer as u64)
                    };
                }
            }
        }
        let value = self.value.powf(&e.value);
        if self.value.real_cmp(&self.value.zero()) != Some(std::cmp::Ordering::Greater) {
            let positive_exponents = e.value.real_cmp(&e.value.zero())
                == Some(std::cmp::Ordering::Greater)
                && e.value.absolute_magnitude_lower().sub_lower(e.abs_err) > Magnitude::ZERO;
            let abs_err = if value.absolute_magnitude().is_nan()
                || self.abs_err.is_nan()
                || e.abs_err.is_nan()
            {
                Magnitude::NAN
            } else if self.is_exact_zero() && positive_exponents || self.is_exact() && e.is_exact()
            {
                Magnitude::ZERO
            } else {
                // Noninteger powers through zero or a negative base can cross
                // a real-domain boundary. Keep the direct central value but do
                // not present a first derivative as a finite uncertainty bound.
                Magnitude::INFINITY
            };
            return Self { value, abs_err }.with_roundoff();
        }
        // exp(e*log(x)) supplies a finite-perturbation estimate, including the
        // mixed base/exponent term. Retain the backend's direct powf central
        // value, accounting for its difference from the composed evaluation.
        let estimate = (self.log() * e).exp();
        let correction = estimate.value - &value;
        let abs_err = estimate.abs_err + Self::rounded_magnitude(&correction);
        Self { value, abs_err }.with_roundoff()
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

#[cfg(test)]
mod coefficient_error_tests {
    use super::*;
    use crate::{
        domains::{SelfRing, float::Complex},
        printer::{PrintOptions, PrintState},
    };

    #[test]
    fn parsing_preserves_accuracy_and_rejects_invalid_precision() {
        let value = ErrorPropagatingFloat::parse("-0.12345678901234567890123456789", None).unwrap();
        assert!(value.get_num().prec() > 90);
        assert!(value.get_absolute_error() > 0.);
        for text in ["0`", "0.000`", "-0e20`"] {
            assert!(
                ErrorPropagatingFloat::parse(text, None)
                    .unwrap()
                    .is_exact_zero()
            );
        }
        let zero = ErrorPropagatingFloat::parse("0.``6", None).unwrap();
        assert!((zero.get_absolute_error() / 1e-6 - 1.).abs() < 1e-8);
        assert!(!zero.is_exact_zero());
        for text in ["1`NaN", "1`inf", "1e"] {
            assert!(ErrorPropagatingFloat::parse(text, None).is_err(), "{text}");
        }
        assert!(ErrorPropagatingFloat::parse("1", Some(0)).is_err());
        for (text, error) in [("1`0", 1.), ("1`-2", 100.), ("0.``-4", 1e4)] {
            assert!(
                (ErrorPropagatingFloat::parse(text, None)
                    .unwrap()
                    .get_absolute_error()
                    / error
                    - 1.)
                    .abs()
                    < 1e-8
            );
        }
    }

    #[test]
    fn precision_changes_keep_uncertainty() {
        let mut value =
            ErrorPropagatingFloat::new_with_absolute_error(Float::with_val(128, 1), 1e-10);
        let initial_error = value.get_error_magnitude();
        value.set_precision(256);
        assert_eq!(value.get_error_magnitude(), initial_error);
        value.set_precision(10);
        assert!(value.get_absolute_error() >= 1e-10 + 2f64.powi(-10));
        let mut zero = ErrorPropagatingFloat::<Float>::exact_zero(128);
        zero.set_precision(10);
        assert!(zero.is_exact_zero());
        assert_eq!(zero.get_num().prec(), 10);
    }

    #[test]
    fn exact_zero_and_singular_derivatives_keep_meaningful_errors() {
        let zero = ErrorPropagatingFloat::<Float>::exact_zero(128);
        assert!(zero.sqrt().is_exact_zero());
        assert!(zero.sin().is_exact_zero());
        assert!(zero.cos().get_absolute_error() > 0.);
        let uncertain_zero = ErrorPropagatingFloat::new_with_absolute_error(0., 1e-6);
        assert!((uncertain_zero.sqrt().get_absolute_error() / 1e-3 - 1.).abs() < 1e-8);
        assert!(
            uncertain_zero
                .powf(&ErrorPropagatingFloat::new(1., 12.))
                .get_absolute_error()
                > 0.
        );
        let x = ErrorPropagatingFloat::new_with_absolute_error(0., 1e-6);
        let y = ErrorPropagatingFloat::new_with_absolute_error(1., 1e-6);
        for angle in [y.atan2(&x), x.atan2(&y)] {
            // The finite-neighborhood derivative bound includes a small
            // nonlinear correction instead of retaining just the center slope.
            let ratio = angle.get_absolute_error() / 1e-6;
            assert!((1. ..1.00001).contains(&ratio));
        }
    }

    #[test]
    fn unit_factory_is_exact_but_floating_one_keeps_its_uncertainty() {
        let floating_one = ErrorPropagatingFloat::<f64>::from(1.);
        assert!(!floating_one.is_exact());
        let unit = floating_one.one();
        assert_eq!(*unit.get_num(), 1.);
        assert!(unit.is_exact());
        assert!(floating_one.pow(0).is_exact());
        assert!(floating_one.powf(&floating_one.zero()).is_exact());
        assert!(!floating_one.powf(&floating_one).is_exact());
        assert!(!floating_one.is_exact());

        let floating_one = ErrorPropagatingFloat::<Float>::from(1.);
        assert!(!floating_one.is_exact());
        assert!(floating_one.one().is_exact());
        assert!(floating_one.pow(0).is_exact());
    }

    #[test]
    fn sqrt_uses_nonlinear_error_near_zero() {
        let exact_half = ErrorPropagatingFloat::new_with_absolute_error(0.5, 0.);
        for (center, error, minimum, maximum) in [
            (4., 3., 1., 1.00000001),
            (4., 4., 2., 2.00000001),
            (1e-12, 1e-6, 1e-3, 1.00000001e-3),
        ] {
            let value = ErrorPropagatingFloat::new_with_absolute_error(center, error);
            let root = value.sqrt();
            assert_eq!(*root.get_num(), center.sqrt());
            assert!(root.get_absolute_error() >= minimum);
            assert!(root.get_absolute_error() <= maximum);
            assert_eq!(value.powf(&exact_half).get_num(), root.get_num());
            assert_eq!(
                value.powf(&exact_half).get_error_magnitude(),
                root.get_error_magnitude()
            );
        }

        // A finite estimate at the real-domain boundary does not claim that
        // negative perturbations have a real square root.
        let invalid = ErrorPropagatingFloat::new_with_absolute_error(-1., 2.).sqrt();
        assert!(invalid.value.is_nan());
        assert!(invalid.get_error_magnitude().is_nan());
        let uncertain_half = ErrorPropagatingFloat::new_with_absolute_error(0.5, 1e-3);
        let base = ErrorPropagatingFloat::new_with_absolute_error(4., 0.);
        assert!(base.powf(&uncertain_half).get_absolute_error() > 1e-3);
    }

    #[test]
    fn nonlinear_sqrt_keeps_wide_exponent_errors() {
        let large = Float::with_val(192, 2).pow(4000);
        for scale in [large.clone(), large.inv()] {
            let center = scale.clone() * Rational::from(4);
            let error = (scale.clone() * Rational::from(3)).absolute_magnitude();
            let value = ErrorPropagatingFloat::new_with_error_magnitude(center, error);
            let result = value.sqrt();
            let expected_error = scale.sqrt().absolute_magnitude();
            assert!(result.get_error_magnitude().is_finite());
            assert!(!result.get_error_magnitude().is_zero());
            assert!(result.get_error_magnitude() >= expected_error);
            let ratio = (result.get_error_magnitude() / expected_error)
                .to_f64()
                .unwrap();
            assert!(ratio <= 1.00000001);
            assert!(result.get_error_magnitude().to_f64().is_none());
        }
    }

    #[test]
    fn uncertain_inputs_reaching_poles_have_unbounded_error() {
        let numerator = ErrorPropagatingFloat::new_with_absolute_error(2., 0.);
        let negative_power = ErrorPropagatingFloat::new_with_absolute_error(-2., 0.);
        for error in [1., 2.] {
            let input = ErrorPropagatingFloat::new_with_absolute_error(1., error);
            assert!(input.inv().get_error_magnitude().is_infinite());
            assert!(input.log().get_error_magnitude().is_infinite());
            assert!((numerator / input).get_error_magnitude().is_infinite());
            let mut quotient = numerator;
            quotient /= input;
            assert!(quotient.get_error_magnitude().is_infinite());
            assert!(
                input
                    .powf(&negative_power)
                    .get_error_magnitude()
                    .is_infinite()
            );
            assert!((-input).inv().get_error_magnitude().is_infinite());
        }
        let separated = ErrorPropagatingFloat::new_with_absolute_error(1., 1e-3);
        assert!(separated.inv().get_error_magnitude().is_finite());
        assert!(separated.log().get_error_magnitude().is_finite());
        assert!(
            separated
                .powf(&negative_power)
                .get_error_magnitude()
                .is_finite()
        );
    }

    #[test]
    fn finite_product_errors_cover_endpoint_perturbations() {
        for (a, da, b, db) in [
            (2., 0.5, 3., 0.25),
            (-2., 0.5, 3., 0.25),
            (0., 2., 3., 4.),
            (3., 4., 0., 2.),
            (0., 2., 0., 4.),
        ] {
            let lhs = ErrorPropagatingFloat::new_with_absolute_error(a, da);
            let rhs = ErrorPropagatingFloat::new_with_absolute_error(b, db);
            let product = lhs * rhs;
            let mut assigned = lhs;
            assigned *= rhs;
            assert_eq!(product.get_num(), assigned.get_num());
            assert_eq!(
                product.get_error_magnitude(),
                assigned.get_error_magnitude()
            );
            for sa in [-1., 1.] {
                for sb in [-1., 1.] {
                    let perturbed = (a + sa * da) * (b + sb * db);
                    assert!((perturbed - product.value).abs() <= product.get_absolute_error());
                }
            }
        }
    }

    #[test]
    fn addition_keeps_input_uncertainty_and_new_roundoff() {
        let a = ErrorPropagatingFloat::new_with_absolute_error(Float::with_val(4, 1), 0.125);
        let b = ErrorPropagatingFloat::new_with_absolute_error(Float::with_val(4, 0.0625), 0.);
        let endpoint = Float::with_val(256, 1.1875);
        let sum = a.clone() + &b;
        let difference = (endpoint - Float::with_val(256, sum.value.to_f64())).absolute_magnitude();
        assert!(sum.get_error_magnitude() >= difference);
        let mut assigned = a.clone();
        assigned += &b;
        assert_eq!(assigned.get_error_magnitude(), sum.get_error_magnitude());
        assert_eq!(assigned.get_num(), sum.get_num());
        let subtracted = a.clone() - &(-b.clone());
        assert_eq!(subtracted.get_error_magnitude(), sum.get_error_magnitude());
        assigned = a.clone();
        assigned -= &(-b);
        assert_eq!(assigned.get_error_magnitude(), sum.get_error_magnitude());

        let cancellation = a.clone() - &a;
        assert!(cancellation.value.is_fully_zero());
        assert_eq!(
            cancellation.get_error_magnitude(),
            Magnitude::from_f64(0.25)
        );
        assert!(!cancellation.is_exact_zero());
    }

    #[test]
    fn finite_quotient_errors_cover_near_pole_perturbations() {
        for (a, da, b, db) in [
            (2., 0.5, 1., 0.5),
            (-2., 0.5, 1., 0.5),
            (2., 0.5, -1., 0.5),
            (0., 0.5, 1., 0.5),
            (2., 0., 1., 0.999),
        ] {
            let lhs = ErrorPropagatingFloat::new_with_absolute_error(a, da);
            let rhs = ErrorPropagatingFloat::new_with_absolute_error(b, db);
            let quotient = lhs / rhs;
            let mut assigned = lhs;
            assigned /= rhs;
            assert_eq!(quotient.get_num(), assigned.get_num());
            assert_eq!(
                quotient.get_error_magnitude(),
                assigned.get_error_magnitude()
            );
            assert!(quotient.get_error_magnitude().is_finite());
            for sa in [-1., 1.] {
                for sb in [-1., 1.] {
                    let perturbed = (a + sa * da) / (b + sb * db);
                    assert!((perturbed - quotient.value).abs() <= quotient.get_absolute_error());
                }
            }
            let inverse = rhs.inv();
            assert!(inverse.get_error_magnitude().is_finite());
            for sign in [-1., 1.] {
                let perturbed = 1. / (b + sign * db);
                assert!((perturbed - inverse.value).abs() <= inverse.get_absolute_error());
            }
        }
        let logarithm = ErrorPropagatingFloat::new_with_absolute_error(1., 0.5).log();
        assert!(logarithm.get_absolute_error() >= 2f64.ln());
        assert!(logarithm.get_absolute_error() < 1.00000001);
    }

    #[test]
    fn quotient_and_inverse_errors_keep_wide_exponents() {
        let large = Float::with_val(192, 2).pow(4000);
        for scale in [large.clone(), large.inv()] {
            let denominator = ErrorPropagatingFloat::new_with_error_magnitude(
                scale.clone(),
                scale.absolute_magnitude().scale_pow2(-1),
            );
            let numerator = ErrorPropagatingFloat::new_with_error_magnitude(
                scale.clone(),
                scale.absolute_magnitude().scale_pow2(-2),
            );
            let quotient = numerator / &denominator;
            assert!((quotient.value.to_f64() - 1.).abs() < 1e-14);
            assert!(quotient.get_absolute_error() >= 1.5);
            assert!(quotient.get_absolute_error() < 1.50000001);
            let inverse = denominator.inv();
            assert!(inverse.get_error_magnitude().is_finite());
            assert!(inverse.get_error_magnitude() >= scale.inv().absolute_magnitude());
            assert!(inverse.get_error_magnitude().to_f64().is_none());
        }
    }

    #[test]
    fn arithmetic_preserves_exact_units_and_accounts_for_roundoff() {
        let uncertain = ErrorPropagatingFloat::new_with_absolute_error(3., 0.125);
        for unit in [uncertain.one(), -uncertain.one()] {
            assert!(unit.inv().is_exact());
            assert!((unit * unit).is_exact());
            assert!((unit / unit).is_exact());
            for result in [uncertain * unit, unit * uncertain, uncertain / unit] {
                assert_eq!(
                    result.get_error_magnitude(),
                    uncertain.get_error_magnitude()
                );
            }
        }
        let three = ErrorPropagatingFloat::new_with_absolute_error(3., 0.);
        assert!(!three.inv().is_exact());
        assert!(!(three.one() / three).is_exact());
        let zero = uncertain.zero();
        assert!((zero * uncertain).is_exact_zero());
        assert!((zero / uncertain).is_exact_zero());

        // The reciprocal must account for storage rounding at low precision,
        // even when the input itself is exact.
        let coarse = ErrorPropagatingFloat::new_with_absolute_error(Float::with_val(4, 3), 0.);
        let inverse = coarse.inv();
        assert!(inverse.get_absolute_error() >= (inverse.value.to_f64() - 1. / 3.).abs());
    }

    #[test]
    fn finite_underflow_does_not_become_exact_zero() {
        let a = ErrorPropagatingFloat::new_with_absolute_error(1e-200, 0.);
        let product = a * a;
        let mut assigned_product = a;
        assigned_product *= a;
        assert_eq!(product.value, 0.);
        assert_eq!(
            product.get_error_magnitude(),
            assigned_product.get_error_magnitude()
        );
        assert!(!product.is_exact_zero());
        let expected =
            (Float::with_val(192, a.value) * Float::with_val(192, a.value)).absolute_magnitude();
        assert!(product.get_error_magnitude() >= expected);
        assert!(product.get_error_magnitude().to_f64().is_none());

        let numerator = ErrorPropagatingFloat::new_with_absolute_error(1e-300, 0.);
        let denominator = ErrorPropagatingFloat::new_with_absolute_error(1e100, 0.);
        let quotient = numerator / denominator;
        let mut assigned_quotient = numerator;
        assigned_quotient /= denominator;
        assert_eq!(quotient.value, 0.);
        assert_eq!(
            quotient.get_error_magnitude(),
            assigned_quotient.get_error_magnitude()
        );
        assert!(!quotient.is_exact_zero());
        let expected = (Float::with_val(192, numerator.value)
            / Float::with_val(192, denominator.value))
        .absolute_magnitude();
        assert!(quotient.get_error_magnitude() >= expected);
        assert!(quotient.get_error_magnitude().to_f64().is_none());

        let zero = numerator.zero();
        assert!((zero * denominator).is_exact_zero());
        assert!((zero / denominator).is_exact_zero());
    }

    #[test]
    fn subnormal_quotient_error_includes_absolute_roundoff_floor() {
        let numerator = ErrorPropagatingFloat::new_with_absolute_error(1e-300, 0.);
        let denominator = ErrorPropagatingFloat::new_with_absolute_error(1e20, 0.);
        let quotient = numerator / denominator;
        assert_eq!(quotient.value.classify(), std::num::FpCategory::Subnormal);
        let precise =
            Float::with_val(192, numerator.value) / Float::with_val(192, denominator.value);
        let discrepancy = (precise - Float::with_val(192, quotient.value)).absolute_magnitude();
        assert!(!discrepancy.is_zero());
        assert!(quotient.get_error_magnitude() >= discrepancy);
        assert!(quotient.get_error_magnitude() >= Magnitude::ONE.scale_pow2(-1075));
    }

    #[test]
    fn copying_signed_zero_preserves_sign_and_uncertainty() {
        let sign = ErrorPropagatingFloat::new_with_absolute_error(-0., 0.);
        for center in [0., 2.] {
            let input = ErrorPropagatingFloat::new_with_absolute_error(center, 0.125);
            let copied = input.copy_sign(&sign);
            assert!(copied.value.is_sign_negative());
            assert_eq!(copied.value.abs(), center);
            assert_eq!(copied.get_error_magnitude(), input.get_error_magnitude());
        }
    }

    #[test]
    fn exact_zero_annihilates_unbounded_error_at_finite_centers() {
        let zero = ErrorPropagatingFloat::new_with_absolute_error(-0., 0.);
        for error in [Magnitude::INFINITY, Magnitude::NAN] {
            let finite = ErrorPropagatingFloat::new_with_error_magnitude(2., error);
            for product in [zero * finite, finite * zero] {
                assert!(product.is_exact_zero());
                assert!(product.value.is_sign_negative());
            }
            let mut assigned = zero;
            assigned *= finite;
            assert!(assigned.is_exact_zero());
            assert!(assigned.value.is_sign_negative());
            assigned = finite;
            assigned *= zero;
            assert!(assigned.is_exact_zero());

            let real = Complex::new(finite, zero);
            assert!((real * real).im.is_exact_zero());
            assert!(real.sin().im.is_exact_zero());
        }
        let infinity = ErrorPropagatingFloat::new_with_absolute_error(f64::INFINITY, 0.);
        assert!((zero * infinity).value.is_nan());
        let uncertain = ErrorPropagatingFloat::new_with_absolute_error(2., 0.125);
        assert!((uncertain / infinity).is_exact_zero());
    }

    #[test]
    fn negative_values_and_powers_have_nonnegative_roundoff() {
        let value =
            ErrorPropagatingFloat::new_with_absolute_error(Float::with_val(64, -1), 0.).truncate();
        assert!(value.get_absolute_error() > 0.);
        let negative = value.from_rational(&Rational::from(-3));
        assert!(negative.get_absolute_error() > 0.);
        let power = ErrorPropagatingFloat::new(2., 12.).powf(&ErrorPropagatingFloat::new(-2., 12.));
        assert!(power.get_absolute_error() > 0.);
        assert!(power.get_precision().unwrap().is_finite());
    }

    #[test]
    fn in_place_arithmetic_preserves_values_and_errors() {
        let a = ErrorPropagatingFloat::new(Float::with_val(128, -3), 25.);
        let b = ErrorPropagatingFloat::new(Float::with_val(96, 7), 20.);
        let mut actual = a.clone();
        actual += &b;
        let expected = a.clone() + &b;
        assert_eq!(actual.get_num(), expected.get_num());
        assert_eq!(actual.get_absolute_error(), expected.get_absolute_error());
        actual = a.clone();
        actual -= &b;
        let expected = a.clone() - &b;
        assert_eq!(actual.get_num(), expected.get_num());
        assert_eq!(actual.get_absolute_error(), expected.get_absolute_error());
        actual = a.clone();
        actual *= &b;
        let expected = a.clone() * &b;
        assert_eq!(actual.get_num(), expected.get_num());
        assert_eq!(actual.get_absolute_error(), expected.get_absolute_error());
        actual = a.clone();
        actual /= &b;
        let expected = a / &b;
        assert_eq!(actual.get_num(), expected.get_num());
        assert_eq!(actual.get_absolute_error(), expected.get_absolute_error());
    }

    #[test]
    fn complex_shortcuts_preserve_component_and_exponent_uncertainty() {
        let z = Complex::new(
            ErrorPropagatingFloat::new_with_absolute_error(4., 0.),
            ErrorPropagatingFloat::new_with_absolute_error(0., 1e-6),
        );
        assert!(!z.is_real());
        assert!(z.to_real().is_none());
        assert!(z.sin().im.get_absolute_error() > 0.);
        let real_base = Complex::new(z.re, ErrorPropagatingFloat::<f64>::exact_zero(53));
        let exponent = Complex::new(
            ErrorPropagatingFloat::new_with_absolute_error(0.5, 1e-3),
            ErrorPropagatingFloat::<f64>::exact_zero(53),
        );
        assert!(real_base.powf(&exponent).re.get_absolute_error() > 1e-3);
    }

    #[test]
    fn real_complex_conversion_preserves_exact_imaginary_zero() {
        let base: Complex<ErrorPropagatingFloat<Float>> =
            Complex::new(Float::with_val(128, 4), Float::new(128)).into();
        let exponent: Complex<ErrorPropagatingFloat<Float>> =
            Complex::new(Float::with_val(128, 0.5), Float::new(128)).into();
        assert!(base.im.is_exact_zero());
        assert!(exponent.im.is_exact_zero());
        let power = base.powf(&exponent);
        assert!(power.im.is_exact_zero());
        assert!((power.re.to_f64() - 2.).abs() < 1e-12);
    }

    #[test]
    fn large_trigonometric_arguments_keep_derivative_precision() {
        // The unit increment affects argument reduction, but is lost in f64.
        let central = Float::with_val(1100, 2).pow(1000) + Rational::from(1);
        let value = ErrorPropagatingFloat::new_with_absolute_error(central.clone(), 1e-3);
        let expected_sin_error = central.cos().to_f64().abs() * 1e-3;
        let expected_cos_error = central.sin().to_f64().abs() * 1e-3;
        assert!(expected_sin_error > 0.);
        assert!(expected_cos_error > 0.);
        for (result, derivative_term) in [
            (value.sin(), expected_sin_error),
            (value.cos(), expected_cos_error),
        ] {
            assert!(result.get_absolute_error() >= derivative_term);
            assert!(result.get_absolute_error() <= derivative_term + 5.0001e-7);
        }
        assert!((central.to_f64().cos().abs() * 1e-3 - expected_sin_error).abs() > 1e-5);
    }

    #[test]
    fn magnitudes_preserve_errors_outside_the_native_exponent_range() {
        let large = ErrorPropagatingFloat::parse("1e10000`40", None).unwrap();
        let small = ErrorPropagatingFloat::parse("1e-10000`40", None).unwrap();
        for value in [&large, &small] {
            assert!(value.get_error_magnitude().is_finite());
            assert!(!value.get_error_magnitude().is_zero());
            assert!((value.get_precision().unwrap() - 40.).abs() < 1e-6);
            assert!(!value.is_zero_within_error());
            let logged = value.log();
            assert!(logged.get_error_magnitude().is_finite());
            assert!(!logged.get_error_magnitude().is_zero());
        }
        let product = large.clone() * &small;
        assert!((product.to_f64() - 1.).abs() < 1e-14);
        assert!(product.get_precision().unwrap() > 39.);
        let cancellation = large.clone() - &large;
        assert!(cancellation.is_zero_within_error());
        assert!(!cancellation.is_exact_zero());
        assert!(cancellation.get_error_magnitude().is_finite());
        let angle = large.atan2(&large);
        assert!(angle.get_error_magnitude().is_finite());
        assert!(angle.get_precision().unwrap() > 39.);
        let uncertain = ErrorPropagatingFloat::new_with_accuracy(large.into_value(), -10001.);
        assert!(uncertain.is_zero_within_error());
    }

    #[test]
    fn high_precision_roundoff_and_zero_accuracy_do_not_underflow() {
        let zero = ErrorPropagatingFloat::new_with_accuracy(Float::new(4000), 1000.);
        assert!(!zero.is_exact_zero());
        assert!(!zero.get_error_magnitude().is_zero());
        assert!((zero.get_accuracy() - 1000.).abs() < 1e-6);
        let one = ErrorPropagatingFloat::<Float>::exact_zero(4000).cos();
        assert!(!one.get_error_magnitude().is_zero());
        assert!((one.get_precision().unwrap() - 4000. * LOG10_2).abs() < 1e-6);
    }

    #[test]
    fn imaginary_zero_prints_its_own_accuracy() {
        let z = Complex::new(
            ErrorPropagatingFloat::<Float>::exact_zero(53),
            ErrorPropagatingFloat::new_with_absolute_error(Float::new(53), 1e-6),
        );
        let mut opts = PrintOptions::new();
        opts.print_precision = true;
        let mut output = String::new();
        z.format(&opts, PrintState::new(), &mut output).unwrap();
        assert!(output.contains("0.``6.00"), "{output}");
    }
}
