use std::{
    f64::consts::{LOG2_10, LOG10_2},
    fmt::{self, Debug, Display, Formatter, LowerExp},
    hash::Hash,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use rand::Rng;
use xprec::{CompensatedArithmetic, Df64};

use super::{DoubleFloat, FloatLike, Real, RealLike, SingleFloat};
use crate::domains::{
    InternalOrdering,
    backend::float::{
        Assign, CompleteRound, Constant, MultiPrecisionFloat, MultiPrecisionFloatInteger,
        MultiPrecisionFloatRational, MultiPrecisionFloatRounding, Pow, RoundingDirection,
    },
    integer::Integer,
    rational::Rational,
};

/// A multi-precision floating point type. Operations on this type
/// loosely track the precision of the result, but always overestimate.
/// Some operations may improve precision, such as `sqrt` or adding an
/// infinite-precision integer.
///
/// Floating point output with less than five significant binary digits
/// should be considered unreliable.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone)]
pub struct Float(MultiPrecisionFloat);

#[cfg(feature = "bincode")]
impl bincode::Encode for Float {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        self.0.prec().encode(encoder)?;
        self.0.to_string_radix(16, None).encode(encoder)
    }
}

#[cfg(feature = "bincode")]
bincode::impl_borrow_decode!(Float);
#[cfg(feature = "bincode")]
impl<Context> bincode::Decode<Context> for Float {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let prec = u32::decode(decoder)?;
        let r = String::decode(decoder)?;
        let val = MultiPrecisionFloat::parse_radix(&r, 16)
            .map_err(|_| bincode::error::DecodeError::Other("Failed to parse float from string"))?
            .complete(prec);
        Ok(Float(val))
    }
}

impl Debug for Float {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl PartialEq for Float {
    fn eq(&self, other: &Self) -> bool {
        if self.0.is_nan() && other.0.is_nan() {
            true
        } else {
            self.0 == other.0
        }
    }
}

impl Eq for Float {}

impl Hash for Float {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        if self.0.is_nan() {
            state.write_u64(0x7ff8000000000000);
            return;
        }

        if self.0.is_zero() {
            state.write_u64(0);
            return;
        }

        // Backend hashes omit precision-dependent zero padding in the significand.
        #[cfg(feature = "float-mpfr")]
        self.0.as_ord().hash(state);
        #[cfg(feature = "float-astro")]
        self.0.hash(state);
    }
}

impl InternalOrdering for Float {
    fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other)
            .unwrap_or_else(|| self.0.is_nan().cmp(&other.0.is_nan()))
    }
}

impl Display for Float {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        // print only the significant digits
        // the original float value may not be reconstructible
        // from this output
        if f.precision().is_none() {
            if f.sign_plus() {
                f.write_fmt(format_args!(
                    "{0:+.1$}",
                    self.0,
                    (self.0.prec() as f64 * LOG10_2).floor() as usize
                ))
            } else {
                f.write_fmt(format_args!(
                    "{0:.1$}",
                    self.0,
                    (self.0.prec() as f64 * LOG10_2).floor() as usize
                ))
            }
        } else {
            Display::fmt(&self.0, f)
        }
    }
}

impl LowerExp for Float {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if f.precision().is_none() {
            f.write_fmt(format_args!(
                "{0:.1$e}",
                self.0,
                (self.0.prec() as f64 * LOG10_2).floor() as usize
            ))
        } else {
            LowerExp::fmt(&self.0, f)
        }
    }
}

impl PartialOrd for Float {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl Neg for Float {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        self.0.neg().into()
    }
}

impl Add<&Float> for Float {
    type Output = Self;

    /// Add two floats, while keeping loose track of the precision.
    /// The precision of the output will be at most 2 binary digits too high.
    #[inline]
    fn add(mut self, rhs: &Self) -> Self::Output {
        if rhs.is_zero() && !self.is_zero() {
            return self;
        }

        if self.is_zero() && !rhs.is_zero() {
            return rhs.clone();
        }

        let sp = self.prec();
        if self.prec() < rhs.prec() {
            self.set_prec(rhs.prec());
        }

        let e1 = self.0.get_exp();

        let mut r = self.0 + &rhs.0;

        if let Some(e) = r.get_exp()
            && let Some(e1) = e1
            && let Some(e2) = rhs.0.get_exp()
        {
            // the max is at most 2 binary digits off
            let max_prec = e + 1 - (e1 - sp as i32).max(e2 - rhs.prec() as i32);

            // set the min precision to 1, from this point on the result is unreliable
            r.set_prec(1.max(max_prec.min(r.prec() as i32)) as u32);
        }

        r.into()
    }
}

impl Add<Float> for Float {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        if rhs.prec() > self.prec() {
            rhs + &self
        } else {
            self + &rhs
        }
    }
}

impl Sub<&Float> for Float {
    type Output = Self;

    #[inline]
    fn sub(mut self, rhs: &Self) -> Self::Output {
        if rhs.is_zero() && !self.is_zero() {
            return self;
        }

        if self.is_zero() && !rhs.is_zero() {
            return -rhs.clone();
        }

        let sp = self.prec();
        if self.prec() < rhs.prec() {
            self.set_prec(rhs.prec());
        }

        let e1 = self.0.get_exp();

        let mut r = self.0 - &rhs.0;

        if let Some(e) = r.get_exp()
            && let Some(e1) = e1
            && let Some(e2) = rhs.0.get_exp()
        {
            let max_prec = e + 1 - (e1 - sp as i32).max(e2 - rhs.prec() as i32);
            r.set_prec(1.max(max_prec.min(r.prec() as i32)) as u32);
        }

        r.into()
    }
}

impl Sub<Float> for Float {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        if rhs.prec() > self.prec() {
            -rhs + &self
        } else {
            self - &rhs
        }
    }
}

impl Mul<&Float> for Float {
    type Output = Self;

    #[inline]
    fn mul(mut self, rhs: &Self) -> Self::Output {
        if self.prec() > rhs.prec() {
            self.set_prec(rhs.prec());
        }

        (self.0 * &rhs.0).into()
    }
}

impl Mul<Float> for Float {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        if rhs.prec() < self.prec() {
            (rhs.0 * self.0).into()
        } else {
            (self.0 * rhs.0).into()
        }
    }
}

impl Div<&Float> for Float {
    type Output = Self;

    #[inline]
    fn div(mut self, rhs: &Self) -> Self::Output {
        if self.prec() > rhs.prec() {
            self.set_prec(rhs.prec());
        }

        (self.0 / &rhs.0).into()
    }
}

impl Div<Float> for Float {
    type Output = Self;

    #[inline]
    fn div(mut self, rhs: Self) -> Self::Output {
        if self.prec() > rhs.prec() {
            self.set_prec(rhs.prec());
        }

        (self.0 / rhs.0).into()
    }
}

impl AddAssign<&Float> for Float {
    #[inline]
    fn add_assign(&mut self, rhs: &Float) {
        if rhs.is_zero() && !self.is_zero() {
            return;
        }
        if self.is_zero() && !rhs.is_zero() {
            *self = rhs.clone();
            return;
        }
        let sp = self.prec();
        if self.prec() < rhs.prec() {
            self.set_prec(rhs.prec());
        }

        let e1 = self.0.get_exp();

        self.0.add_assign(&rhs.0);

        if let Some(e) = self.0.get_exp()
            && let Some(e1) = e1
            && let Some(e2) = rhs.0.get_exp()
        {
            let max_prec = e + 1 - (e1 - sp as i32).max(e2 - rhs.prec() as i32);
            self.set_prec(1.max(max_prec.min(self.prec() as i32)) as u32);
        }
    }
}

impl AddAssign<Float> for Float {
    #[inline]
    fn add_assign(&mut self, rhs: Float) {
        self.add_assign(&rhs)
    }
}

impl SubAssign<&Float> for Float {
    #[inline]
    fn sub_assign(&mut self, rhs: &Float) {
        if rhs.is_zero() && !self.is_zero() {
            return;
        }
        if self.is_zero() && !rhs.is_zero() {
            *self = -rhs.clone();
            return;
        }
        let sp = self.prec();
        if self.prec() < rhs.prec() {
            self.set_prec(rhs.prec());
        }

        let e1 = self.0.get_exp();

        self.0.sub_assign(&rhs.0);

        if let Some(e) = self.0.get_exp()
            && let Some(e1) = e1
            && let Some(e2) = rhs.0.get_exp()
        {
            let max_prec = e + 1 - (e1 - sp as i32).max(e2 - rhs.prec() as i32);
            self.set_prec(1.max(max_prec.min(self.prec() as i32)) as u32);
        }
    }
}

impl SubAssign<Float> for Float {
    #[inline]
    fn sub_assign(&mut self, rhs: Float) {
        self.sub_assign(&rhs)
    }
}

impl MulAssign<&Float> for Float {
    #[inline]
    fn mul_assign(&mut self, rhs: &Float) {
        if self.prec() > rhs.prec() {
            self.set_prec(rhs.prec());
        }

        self.0.mul_assign(&rhs.0);
    }
}

impl MulAssign<Float> for Float {
    #[inline]
    fn mul_assign(&mut self, rhs: Float) {
        if self.prec() > rhs.prec() {
            self.set_prec(rhs.prec());
        }

        self.0.mul_assign(rhs.0);
    }
}

impl DivAssign<&Float> for Float {
    #[inline]
    fn div_assign(&mut self, rhs: &Float) {
        if self.prec() > rhs.prec() {
            self.set_prec(rhs.prec());
        }

        self.0.div_assign(&rhs.0);
    }
}

impl DivAssign<Float> for Float {
    #[inline]
    fn div_assign(&mut self, rhs: Float) {
        if self.prec() > rhs.prec() {
            self.set_prec(rhs.prec());
        }

        self.0.div_assign(rhs.0);
    }
}

impl Add<Float> for i64 {
    type Output = Float;

    /// Add a float to an infinite-precision `i64`.
    #[inline]
    fn add(self, rhs: Float) -> Self::Output {
        rhs + self
    }
}

impl Sub<Float> for i64 {
    type Output = Float;

    /// Subtract a float from an infinite-precision `i64`.
    #[inline]
    fn sub(self, rhs: Float) -> Self::Output {
        -rhs + self
    }
}

impl Mul<Float> for i64 {
    type Output = Float;

    /// Multiply a float to an infinite-precision `i64`.
    #[inline]
    fn mul(self, rhs: Float) -> Self::Output {
        (self * rhs.0).into()
    }
}

impl Div<Float> for i64 {
    type Output = Float;

    /// Divide a float from an infinite-precision `i64`.
    #[inline]
    fn div(self, rhs: Float) -> Self::Output {
        (self / rhs.0).into()
    }
}

impl<R: Into<Rational>> Add<R> for Float {
    type Output = Self;

    /// Add an infinite-precision rational to the float.
    #[inline]
    fn add(mut self, rhs: R) -> Self::Output {
        fn get_bits(i: &Integer) -> i32 {
            match i {
                Integer::Single(n) => n.unsigned_abs().ilog2() as i32 + 1,
                Integer::Double(n) => n.get().unsigned_abs().ilog2() as i32 + 1,
                Integer::Large(r) => r.significant_bits() as i32,
            }
        }

        let rhs = rhs.into();
        if rhs.is_zero() {
            return self;
        }

        let Some(e1) = self.0.get_exp() else {
            let np = self.prec();
            return (self.0 + rhs.to_multi_prec_float(np).0).into();
        };

        if rhs.denominator_ref().is_one() {
            let e2 = get_bits(&rhs.numerator_ref());
            let old_prec = self.prec();

            if e1 <= e2 {
                self.set_prec(old_prec + (e2 as i32 - e1) as u32 + 1);
            }

            let mut r = match rhs.numerator() {
                Integer::Single(n) => self.0 + n,
                Integer::Double(n) => self.0 + n.get(),
                Integer::Large(n) => self.0.add_integer(n),
            };

            if let Some(e) = r.get_exp() {
                r.set_prec((1.max(old_prec as i32 + 1 - (e1 - e))) as u32);
            }

            return r.into();
        }

        // TODO: check off-by-one errors
        let e2 = get_bits(rhs.numerator_ref()) - get_bits(rhs.denominator_ref());

        let old_prec = self.prec();

        if e1 <= e2 {
            self.set_prec(old_prec + (e2 - e1) as u32 + 1);
        }

        let np = self.prec();
        let mut r = self.0 + rhs.to_multi_prec_float(np).0;

        if let Some(e) = r.get_exp() {
            r.set_prec((1.max(old_prec as i32 + 1 - (e1 - e))) as u32);
        }

        r.into()
    }
}

impl<R: Into<Rational>> Sub<R> for Float {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: R) -> Self::Output {
        self + -rhs.into()
    }
}

impl<R: Into<Rational>> Mul<R> for Float {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: R) -> Self::Output {
        let r = rhs.into();
        if r.is_integer() {
            match r.numerator() {
                Integer::Single(n) => self.0 * n,
                Integer::Double(n) => self.0 * n.get(),
                Integer::Large(n) => self.0.mul_integer(n),
            }
            .into()
        } else {
            let num = r.numerator().to_multi_prec();
            let den = r.denominator().to_multi_prec();
            self.0.mul_integer_ratio(num, den).into()
        }
    }
}

impl<R: Into<Rational>> Div<R> for Float {
    type Output = Self;

    #[inline]
    fn div(self, rhs: R) -> Self::Output {
        let r = rhs.into();
        if r.is_integer() {
            match r.numerator() {
                Integer::Single(n) => self.0 / n,
                Integer::Double(n) => self.0 / n.get(),
                Integer::Large(n) => self.0.div_integer(n),
            }
            .into()
        } else {
            let num = r.numerator().to_multi_prec();
            let den = r.denominator().to_multi_prec();
            self.0.div_integer_ratio(num, den).into()
        }
    }
}

impl From<f64> for Float {
    fn from(value: f64) -> Self {
        Float::with_val(53, value)
    }
}

impl From<DoubleFloat> for Float {
    fn from(value: DoubleFloat) -> Self {
        Float(MultiPrecisionFloat::with_val(106, value.0.hi()) + value.0.lo())
    }
}

impl From<&DoubleFloat> for Float {
    fn from(value: &DoubleFloat) -> Self {
        Float(MultiPrecisionFloat::with_val(106, value.0.hi()) + value.0.lo())
    }
}

impl Float {
    /// Wrap a value from the selected arbitrary-precision float backend.
    #[inline]
    pub fn from_raw(value: MultiPrecisionFloat) -> Self {
        Self(value)
    }

    /// Borrow the value from the selected arbitrary-precision float backend.
    #[inline]
    pub fn as_raw(&self) -> &MultiPrecisionFloat {
        &self.0
    }

    /// Clone the value from the selected arbitrary-precision float backend.
    #[inline]
    pub fn to_raw(&self) -> MultiPrecisionFloat {
        self.0.clone()
    }

    pub fn new(prec: u32) -> Self {
        Float(MultiPrecisionFloat::new(prec))
    }

    pub fn with_val<T>(prec: u32, val: T) -> Self
    where
        MultiPrecisionFloat: Assign<T>,
    {
        Float(MultiPrecisionFloat::with_val(prec, val))
    }

    /// Construct a multi-precision float from a backend-independent integer.
    pub fn with_integer(prec: u32, value: crate::domains::integer::MultiPrecisionInteger) -> Self {
        Float(MultiPrecisionFloat::from_integer(prec, value))
    }

    pub fn prec(&self) -> u32 {
        self.0.prec()
    }

    pub fn set_prec(&mut self, prec: u32) {
        self.0.set_prec(prec);
    }

    /// Adds `rhs` and rounds the result to `prec` binary digits in `direction`.
    ///
    /// In particular, [`RoundingDirection::Down`] and
    /// [`RoundingDirection::Up`] give lower and upper bounds, respectively,
    /// for the exact sum of the represented values.
    pub fn add_round(&self, rhs: &Self, prec: u32, direction: RoundingDirection) -> Self {
        self.0.add_round(&rhs.0, prec, direction).into()
    }

    /// Subtracts `rhs` and rounds the result to `prec` binary digits in
    /// `direction`.
    ///
    /// In particular, [`RoundingDirection::Down`] and
    /// [`RoundingDirection::Up`] give lower and upper bounds, respectively,
    /// for the exact difference of the represented values.
    pub fn sub_round(&self, rhs: &Self, prec: u32, direction: RoundingDirection) -> Self {
        self.0.sub_round(&rhs.0, prec, direction).into()
    }

    /// Multiplies by `rhs` and rounds the result to `prec` binary digits in
    /// `direction`.
    ///
    /// In particular, [`RoundingDirection::Down`] and
    /// [`RoundingDirection::Up`] give lower and upper bounds, respectively,
    /// for the exact product of the represented values.
    pub fn mul_round(&self, rhs: &Self, prec: u32, direction: RoundingDirection) -> Self {
        self.0.mul_round(&rhs.0, prec, direction).into()
    }

    /// Divides by `rhs` and rounds the result to `prec` binary digits in
    /// `direction`.
    ///
    /// For nonzero `rhs`, [`RoundingDirection::Down`] and
    /// [`RoundingDirection::Up`] give lower and upper bounds, respectively,
    /// for the exact quotient of the represented values.
    pub fn div_round(&self, rhs: &Self, prec: u32, direction: RoundingDirection) -> Self {
        self.0.div_round(&rhs.0, prec, direction).into()
    }

    /// Converts the exact rational `value` to `prec` binary digits, rounding
    /// in `direction`.
    ///
    /// [`RoundingDirection::Down`] and [`RoundingDirection::Up`] give a lower
    /// and upper bound, respectively, for `value`.
    pub fn from_rational_round(value: &Rational, prec: u32, direction: RoundingDirection) -> Self {
        MultiPrecisionFloat::from_integer_ratio_round(
            value.numerator().to_multi_prec(),
            value.denominator().to_multi_prec(),
            prec,
            direction,
        )
        .into()
    }

    pub fn is_finite(&self) -> bool {
        self.0.is_finite()
    }

    /// Return whether the value is strictly less than zero.
    /// Both signed zeros and NaN return false; negative infinity returns true.
    pub fn is_negative(&self) -> bool {
        self.0.is_sign_negative() && !self.0.is_zero() && !self.0.is_nan()
    }

    /// Return whether the sign bit is negative, including for negative zero.
    /// Use this for sign copying and signed-zero branch-cut conventions.
    /// For NaN, the result depends on the backend's representation.
    pub fn is_sign_negative(&self) -> bool {
        self.0.is_sign_negative()
    }

    /// Converts this float to a `DoubleFloat`.
    pub fn to_double_float(&self) -> DoubleFloat {
        let hi = self.0.to_f64();

        if !hi.is_finite() {
            return DoubleFloat(Df64::new(hi));
        }

        let mut residual = MultiPrecisionFloat::with_val(self.prec().max(106) + 8, &self.0);
        residual -= hi;

        DoubleFloat(Df64::compensated_sum(hi, residual.to_f64()))
    }

    /// Parse decimal notation, optionally with an `e`/`E` exponent, or NaN/infinity.
    /// An explicit `prec` is in bits and overrides a backtick suffix.
    /// Precision can be specified by a trailing backtick followed by the precision.
    /// For example: ```1.234`20``` for a precision of 20 decimal digits.
    /// The precision is allowed to be a floating point number.
    ///  If `prec` is `None` and no precision is specified (either no backtick
    /// or a backtick without a number following), the precision is derived from the string, with
    /// a minimum of 53 bits (`f64` precision).
    pub fn parse(s: &str, prec: Option<u32>) -> Result<Self, String> {
        let (value, suffix) = s
            .trim()
            .split_once('`')
            .map_or((s.trim(), None), |(v, p)| (v, Some(p)));
        let suffix_precision = suffix
            .filter(|p| !p.is_empty())
            .map(|p| {
                Self::decimal_digits_to_bits(
                    p.parse::<f64>()
                        .map_err(|e| format!("Invalid precision: {e}"))?,
                )
            })
            .transpose()?;
        let precision = if let Some(prec) = prec {
            Self::check_precision(prec)?;
            prec
        } else if let Some(prec) = suffix_precision {
            prec
        } else {
            // Count significant decimal digits in the significand, excluding the
            // sign, decimal point, leading zeroes, and scientific exponent.
            let digits = value
                .split(['e', 'E'])
                .next()
                .unwrap_or(value)
                .chars()
                .filter(char::is_ascii_digit)
                .skip_while(|c| *c == '0')
                .count();
            Self::decimal_digits_to_bits(digits.max(1) as f64)?.max(53)
        };
        // Handle special values consistently across backends. Astro's parser
        // uses NaN to report invalid input, so it cannot distinguish a NaN literal.
        let special = match value.to_ascii_lowercase().as_str() {
            "nan" | "+nan" | "-nan" => Some(f64::NAN),
            "inf" | "+inf" | "infinity" | "+infinity" => Some(f64::INFINITY),
            "-inf" | "-infinity" => Some(f64::NEG_INFINITY),
            _ => None,
        };
        if let Some(value) = special {
            return Ok(Float::with_val(precision, value));
        }

        // Validate the whole decimal literal: some backends accept a valid
        // prefix (for example, `1e`) instead of reporting malformed input.
        let unsigned = value.strip_prefix(['+', '-']).unwrap_or(value);
        let (mantissa, exponent) = unsigned
            .split_once(['e', 'E'])
            .map_or((unsigned, None), |(m, e)| (m, Some(e)));
        let valid_exponent = exponent.is_none_or(|e| {
            let e = e.strip_prefix(['+', '-']).unwrap_or(e);
            !e.is_empty() && e.bytes().all(|c| c.is_ascii_digit())
        });
        if !valid_exponent
            || !mantissa.bytes().any(|c| c.is_ascii_digit())
            || mantissa.bytes().any(|c| !c.is_ascii_digit() && c != b'.')
            || mantissa.bytes().filter(|&c| c == b'.').count() > 1
        {
            return Err(format!("Invalid decimal float: {value}"));
        }

        #[cfg(feature = "float-astro")]
        return MultiPrecisionFloat::parse_at_prec(value, precision).map(Float);

        #[cfg(feature = "float-mpfr")]
        Ok(Float(
            MultiPrecisionFloat::parse(value)
                .map_err(|e| e.to_string())?
                .complete(precision),
        ))
    }

    /// Convert a positive, finite decimal precision to a supported binary precision.
    pub fn decimal_digits_to_bits(digits: f64) -> Result<u32, String> {
        let bits = (digits * LOG2_10).ceil();
        if !digits.is_finite() || digits <= 0. || bits > u32::MAX as f64 {
            return Err(format!(
                "Invalid decimal precision {digits}: expected a positive finite precision fitting in a u32 binary precision"
            ));
        }
        let bits = bits as u32;
        Self::check_precision(bits)?;
        Ok(bits)
    }

    pub(crate) fn check_precision(prec: u32) -> Result<(), String> {
        #[cfg(feature = "float-mpfr")]
        let valid = (rug::float::prec_min()..=rug::float::prec_max()).contains(&prec);
        #[cfg(not(feature = "float-mpfr"))]
        let valid = prec > 0;
        if valid {
            Ok(())
        } else {
            Err(format!("Invalid binary precision {prec}"))
        }
    }

    pub fn serialize(&self) -> Vec<u8> {
        if self.0 == 0 {
            // serialize 0 and -0 as '0'
            vec![48]
        } else {
            self.0.to_string_radix(16, None).into_bytes()
        }
    }

    pub fn deserialize(d: &[u8], prec: u32) -> Float {
        MultiPrecisionFloat::parse_radix(d, 16)
            .unwrap()
            .complete(prec)
            .into()
    }

    pub fn to_rational(&self) -> Rational {
        let (num, den) = self.0.to_integer_ratio().unwrap();
        Rational::from_int_unchecked(num, den)
    }

    pub fn try_to_rational(&self) -> Option<Rational> {
        self.0
            .to_integer_ratio()
            .map(|(num, den)| Rational::from_int_unchecked(num, den))
    }

    /// Consume this wrapper and return the selected backend's value.
    #[inline]
    pub fn into_raw(self) -> MultiPrecisionFloat {
        self.0
    }
}

impl From<MultiPrecisionFloat> for Float {
    fn from(value: MultiPrecisionFloat) -> Self {
        Self::from_raw(value)
    }
}

impl FloatLike for Float {
    fn nan(&self) -> Option<Self> {
        Some(Float::with_val(self.prec(), f64::NAN))
    }

    #[inline(always)]
    fn real_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.partial_cmp(other)
    }

    #[inline]
    fn real_classify(&self) -> Option<std::num::FpCategory> {
        use std::num::FpCategory;
        Some(if self.0.is_nan() {
            FpCategory::Nan
        } else if !self.is_finite() {
            FpCategory::Infinite
        } else if self.0.is_zero() {
            FpCategory::Zero
        } else {
            FpCategory::Normal
        })
    }

    #[inline(always)]
    fn needs_rescaling(&self) -> bool {
        !self.is_finite() || self.is_zero()
    }

    #[inline(always)]
    fn set_from(&mut self, other: &Self) {
        self.0.clone_from(&other.0);
    }

    #[inline(always)]
    fn mul_add(&self, a: &Self, b: &Self) -> Self {
        self.clone() * a + b
    }

    #[inline(always)]
    fn neg(&self) -> Self {
        (-self.0.clone()).into()
    }

    #[inline(always)]
    fn zero(&self) -> Self {
        Float::new(self.prec())
    }

    #[inline(always)]
    fn new_zero() -> Self {
        Float::new(1)
    }

    #[inline(always)]
    fn one(&self) -> Self {
        Float::with_val(self.prec(), 1.)
    }

    #[inline]
    fn pow(&self, e: u64) -> Self {
        MultiPrecisionFloat::with_val(self.prec(), Pow::pow(&self.0, e)).into()
    }

    #[inline(always)]
    fn inv(&self) -> Self {
        self.0.clone().recip().into()
    }

    /// Convert from a `usize`. This may involve a loss of precision.
    #[inline(always)]
    fn from_usize(&self, a: usize) -> Self {
        Float::with_val(self.prec(), a)
    }

    /// Convert from a `i64`. This may involve a loss of precision.
    #[inline(always)]
    fn from_i64(&self, a: i64) -> Self {
        Float::with_val(self.prec(), a)
    }

    fn get_precision(&self) -> u32 {
        self.prec()
    }

    #[inline(always)]
    fn set_precision(&mut self, precision: u32) {
        self.set_prec(precision);
    }

    #[inline(always)]
    fn get_epsilon(&self) -> f64 {
        2.0f64.powi(-(self.prec() as i32))
    }

    #[inline(always)]
    fn fixed_precision(&self) -> bool {
        false
    }

    fn sample_unit<R: Rng + ?Sized>(&self, rng: &mut R) -> Self {
        let f: f64 = rng.random();
        Float::with_val(self.prec(), f)
    }

    fn is_fully_zero(&self) -> bool {
        self.0.is_zero()
    }
}

impl SingleFloat for Float {
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0 == 0.
    }

    #[inline(always)]
    fn is_one(&self) -> bool {
        self.0 == 1.
    }

    #[inline(always)]
    fn is_finite(&self) -> bool {
        self.0.is_finite()
    }

    #[inline(always)]
    fn from_rational(&self, rat: &Rational) -> Self {
        rat.to_multi_prec_float(self.prec())
    }
}

impl RealLike for Float {
    fn to_usize_clamped(&self) -> usize {
        self.0
            .to_integer()
            .unwrap()
            .to_usize()
            .unwrap_or(usize::MAX)
    }

    fn to_f64(&self) -> f64 {
        self.0.to_f64()
    }

    #[inline(always)]
    fn round_to_nearest_integer(&self) -> Integer {
        self.0.to_integer_exact().unwrap().into()
    }
}

impl Real for Float {
    #[cfg(feature = "float-mpfr")]
    #[inline(always)]
    fn log1p(&self) -> Self {
        self.0.clone().ln_1p().into()
    }

    #[inline(always)]
    fn copy_sign(&self, sign: &Self) -> Self {
        if sign.is_sign_negative() {
            -self.norm()
        } else {
            self.norm()
        }
    }

    #[inline(always)]
    fn pi(&self) -> Self {
        MultiPrecisionFloat::with_val(self.prec(), Constant::Pi).into()
    }

    #[inline(always)]
    fn e(&self) -> Self {
        self.one().exp()
    }

    #[inline(always)]
    fn euler(&self) -> Self {
        MultiPrecisionFloat::with_val(self.prec(), Constant::Euler).into()
    }

    #[inline(always)]
    fn phi(&self) -> Self {
        (self.one() + self.from_i64(5).sqrt()) / 2
    }

    #[inline(always)]
    fn i(&self) -> Option<Self> {
        None
    }

    #[inline(always)]
    fn conj(&self) -> Self {
        self.clone()
    }

    #[inline(always)]
    fn norm(&self) -> Self {
        self.0.clone().abs().into()
    }

    #[inline(always)]
    fn sqrt(&self) -> Self {
        MultiPrecisionFloat::with_val(self.prec() + 1, self.0.sqrt_ref()).into()
    }

    #[inline(always)]
    fn log(&self) -> Self {
        // Log grows in precision if the input is less than 1/e and more than e
        if let Some(e) = self.0.get_exp()
            && !(0..2).contains(&e)
        {
            MultiPrecisionFloat::with_val(
                self.0.prec() + e.unsigned_abs().ilog2() + 1,
                self.0.ln_ref(),
            )
            .into()
        } else {
            self.0.clone().ln().into()
        }
    }

    #[inline(always)]
    fn exp(&self) -> Self {
        if let Some(e) = self.0.get_exp() {
            // Exp grows in precision when e < 0
            MultiPrecisionFloat::with_val(
                1.max(self.0.prec() as i32 - e + 1) as u32,
                self.0.exp_ref(),
            )
            .into()
        } else {
            self.0.clone().exp().into()
        }
    }

    #[inline(always)]
    fn sin(&self) -> Self {
        self.0.clone().sin().into()
    }

    #[inline(always)]
    fn cos(&self) -> Self {
        self.0.clone().cos().into()
    }

    #[inline(always)]
    fn tan(&self) -> Self {
        self.0.clone().tan().into()
    }

    #[inline(always)]
    fn asin(&self) -> Self {
        self.0.clone().asin().into()
    }

    #[inline(always)]
    fn acos(&self) -> Self {
        self.0.clone().acos().into()
    }

    #[inline(always)]
    fn atan2(&self, x: &Self) -> Self {
        if self.prec() == x.prec() {
            return self.0.clone().atan2(&x.0).into();
        }
        let precision = self.prec().max(x.prec());
        if !self.is_finite() || !x.is_finite() || self.is_zero() || x.is_zero() {
            // Axis angles are exact zeros or exact multiples of pi. Preserve
            // the backend's signed-zero/nonfinite conventions without making
            // the precision of an exact zero throttle a known angle.
            return MultiPrecisionFloat::with_val(precision, &self.0)
                .atan2(&x.0)
                .into();
        }

        // A tiny component can retain few relative bits after cancellation,
        // yet the phase near +/-pi or +/-pi/2 is accurately known in absolute
        // terms. Compute only the small correction at the ratio's precision;
        // Float subtraction then tracks its absolute uncertainty against a
        // freshly computed exact constant. Never pad a computed small angle.
        let pi = || Float::with_val(precision, Constant::Pi);
        if x.norm() >= self.norm() {
            let correction: Float = (self.clone() / x.norm()).0.atan().into();
            if x.is_negative() {
                let axis = if self.is_negative() { -pi() } else { pi() };
                axis - correction
            } else {
                correction
            }
        } else {
            let correction: Float = (x.clone() / self).0.atan().into();
            let half_pi = pi() / 2;
            let axis = if self.is_negative() {
                -half_pi
            } else {
                half_pi
            };
            axis - correction
        }
    }

    #[inline(always)]
    fn sinh(&self) -> Self {
        self.0.clone().sinh().into()
    }

    #[inline(always)]
    fn cosh(&self) -> Self {
        self.0.clone().cosh().into()
    }

    #[inline(always)]
    fn tanh(&self) -> Self {
        if let Some(e) = self.0.get_exp()
            && e > 0
        {
            return MultiPrecisionFloat::with_val(
                self.0.prec() + 3 * e.unsigned_abs() + 1,
                self.0.tanh_ref(),
            )
            .into();
        }

        self.0.clone().tanh().into()
    }

    #[inline(always)]
    fn asinh(&self) -> Self {
        self.0.clone().asinh().into()
    }

    #[inline(always)]
    fn acosh(&self) -> Self {
        self.0.clone().acosh().into()
    }

    #[inline(always)]
    fn atanh(&self) -> Self {
        self.0.clone().atanh().into()
    }

    #[inline]
    fn powf(&self, e: &Self) -> Self {
        let mut c = self.0.clone();
        if let Some(exp) = e.0.get_exp()
            && let Some(eb) = self.0.get_exp()
        {
            // eb is (over)estimate of ln(self)
            // TODO: prevent taking the wrong branch when self = 1
            if eb == 0 {
                c.set_prec(1.max((self.0.prec() as i32 - exp + 1) as u32));
            } else {
                c.set_prec(
                    1.max(
                        (self.0.prec() as i32)
                            .min((e.0.prec() as i32) + eb.unsigned_abs().ilog2() as i32)
                            - exp
                            + 1,
                    ) as u32,
                );
            }
        }

        c.pow(&e.0).into()
    }
}

impl Rational {
    // Convert the rational number to a multi-precision float with precision `prec`.
    pub fn to_multi_prec_float(&self, prec: u32) -> Float {
        Float::from_rational_round(self, prec, RoundingDirection::Nearest)
    }
}

#[cfg(test)]
mod precision_tests {
    use super::Float;

    #[test]
    fn zero_does_not_restore_lost_precision() {
        let x = Float::with_val(40, 1) - Float::with_val(40, 1.0 - 2.0_f64.powi(-30));
        assert!(x.prec() < 40);
        let zero = Float::with_val(1000, 0);
        for result in [
            x.clone() + &zero,
            zero.clone() + &x,
            x.clone() - &zero,
            zero.clone() - &x,
        ] {
            assert_eq!(result.prec(), x.prec());
        }
        let mut result = x.clone();
        result += &zero;
        assert_eq!(result.prec(), x.prec());
        result -= &zero;
        assert_eq!(result.prec(), x.prec());
        let mut result = zero.clone();
        result += &x;
        assert_eq!(result.prec(), x.prec());
        let mut result = zero;
        result -= &x;
        assert_eq!(result.prec(), x.prec());
    }

    #[test]
    fn invalid_precision_returns_errors() {
        assert!(Float::parse("1", Some(0)).is_err());
        for text in ["1`0", "1`-1", "1`NaN", "1`inf", "1`1e100"] {
            assert!(Float::parse(text, None).is_err(), "{text}");
        }
        for digits in [0., -1., f64::NAN, f64::INFINITY, u32::MAX as f64] {
            assert!(Float::decimal_digits_to_bits(digits).is_err());
        }
    }

    #[test]
    fn valid_precision_preserves_existing_parsing() {
        assert_eq!(Float::decimal_digits_to_bits(40.).unwrap(), 133);
        assert_eq!(Float::parse("1", Some(80)).unwrap().prec(), 80);
        assert_eq!(Float::parse("1`40", None).unwrap().prec(), 133);
        assert_eq!(Float::parse("1`", None).unwrap().prec(), 53);
    }
}
