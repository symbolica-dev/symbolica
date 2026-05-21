use std::{
    f64::consts::{LOG2_10, LOG10_2},
    fmt::{self, Debug, Display, Formatter, LowerExp},
    hash::Hash,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use rand::Rng;
use rug::{
    Assign, Float as MultiPrecisionFloat,
    ops::{CompleteRound, Pow},
};
use xprec::{CompensatedArithmetic, Df64};

use super::{DoubleFloat, FloatLike, Real, RealLike, SingleFloat};
use crate::domains::{InternalOrdering, integer::Integer, rational::Rational};

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

        self.0.get_exp().hash(state);
        if let Some(s) = self.0.get_significand() {
            s.hash(state);
        } else {
            state.write_u64(0x7ff8000000000000)
        }
    }
}

impl InternalOrdering for Float {
    fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
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
                Integer::Double(n) => n.unsigned_abs().ilog2() as i32 + 1,
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
                Integer::Double(n) => self.0 + n,
                Integer::Large(n) => self.0 + n,
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
                Integer::Double(n) => self.0 * n,
                Integer::Large(n) => self.0 * n,
            }
            .into()
        } else {
            (self.0 * r.to_multi_prec()).into()
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
                Integer::Double(n) => self.0 / n,
                Integer::Large(n) => self.0 / n,
            }
            .into()
        } else {
            (self.0 / r.to_multi_prec()).into()
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
    pub fn new(prec: u32) -> Self {
        Float(MultiPrecisionFloat::new(prec))
    }

    pub fn with_val<T>(prec: u32, val: T) -> Self
    where
        MultiPrecisionFloat: Assign<T>,
    {
        Float(MultiPrecisionFloat::with_val(prec, val))
    }

    pub fn prec(&self) -> u32 {
        self.0.prec()
    }

    pub fn set_prec(&mut self, prec: u32) {
        self.0.set_prec(prec);
    }

    pub fn is_finite(&self) -> bool {
        self.0.is_finite()
    }

    pub fn is_negative(&self) -> bool {
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

    /// Parse a float from a string.
    /// Precision can be specified by a trailing backtick followed by the precision.
    /// For example: ```1.234`20``` for a precision of 20 decimal digits.
    /// The precision is allowed to be a floating point number.
    ///  If `prec` is `None` and no precision is specified (either no backtick
    /// or a backtick without a number following), the precision is derived from the string, with
    /// a minimum of 53 bits (`f64` precision).
    pub fn parse(s: &str, prec: Option<u32>) -> Result<Self, String> {
        if let Some(prec) = prec {
            Ok(Float(
                MultiPrecisionFloat::parse(s)
                    .map_err(|e| e.to_string())?
                    .complete(prec),
            ))
        } else if let Some((f, p)) = s.split_once('`') {
            let prec = if p.is_empty() {
                53
            } else {
                (p.parse::<f64>()
                    .map_err(|e| format!("Invalid precision: {e}"))?
                    * LOG2_10)
                    .ceil() as u32
            };

            Ok(Float(
                MultiPrecisionFloat::parse(f)
                    .map_err(|e| e.to_string())?
                    .complete(prec),
            ))
        } else {
            // get the number of accurate digits
            let digits = s
                .chars()
                .skip_while(|x| *x == '.' || *x == '0')
                .take_while(|x| x.is_ascii_digit())
                .count();

            let prec = ((digits as f64 * LOG2_10).ceil() as u32).max(53);
            Ok(Float(
                MultiPrecisionFloat::parse(s)
                    .map_err(|e| e.to_string())?
                    .complete(prec),
            ))
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
        self.0.to_rational().unwrap().into()
    }

    pub fn try_to_rational(&self) -> Option<Rational> {
        self.0.to_rational().map(|x| x.into())
    }

    pub fn into_inner(self) -> MultiPrecisionFloat {
        self.0
    }
}

impl From<MultiPrecisionFloat> for Float {
    fn from(value: MultiPrecisionFloat) -> Self {
        Float(value)
    }
}

impl FloatLike for Float {
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
        MultiPrecisionFloat::with_val(self.prec(), rug::ops::Pow::pow(&self.0, e)).into()
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
        self.0.to_integer().unwrap().into()
    }
}

impl Real for Float {
    #[inline(always)]
    fn pi(&self) -> Self {
        MultiPrecisionFloat::with_val(self.prec(), rug::float::Constant::Pi).into()
    }

    #[inline(always)]
    fn e(&self) -> Self {
        self.one().exp()
    }

    #[inline(always)]
    fn euler(&self) -> Self {
        MultiPrecisionFloat::with_val(self.prec(), rug::float::Constant::Euler).into()
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
        self.0.clone().atan2(&x.0).into()
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
        Float::with_val(
            prec,
            rug::Rational::from((
                self.numerator().to_multi_prec(),
                self.denominator().to_multi_prec(),
            )),
        )
    }
}
