//! Complex numerical values with arbitrary-precision error information.
//!
//! Arithmetic encloses the supplied input balls. `Bounded` describes the error
//! model, not the provenance of external numerical data: validated decimal
//! samples are estimates unless the caller supplies a proved enclosure.

use super::{Complex, ComplexBall, Float, FloatLike, RealBall, RoundingDirection, SingleFloat};
use crate::domains::{integer::Integer, rational::Rational};

/// Whether every uncertainty entering a result has a supported bound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorQuality {
    Bounded,
    Estimated,
}

impl ErrorQuality {
    /// Combine error models; one estimated ingredient makes the result estimated.
    pub fn combine(self, other: Self) -> Self {
        if self == Self::Bounded && other == Self::Bounded {
            Self::Bounded
        } else {
            Self::Estimated
        }
    }
}

/// Accuracy information for one real component of a tracked value.
#[derive(Clone, Debug)]
pub struct TrackedComponentPrecision {
    /// Conservative significant binary digits, capped at the center's precision.
    ///
    /// For `Some(p)`, the error divided by the smallest possible absolute true
    /// value is at most `2^(1-p)`. `None` means either an exact zero (see
    /// `exact_zero`) or that no significant digit is established.
    pub significant_bits: Option<u32>,
    pub absolute_error: Float,
    pub exact_zero: bool,
    pub error_quality: ErrorQuality,
}

impl TrackedComponentPrecision {
    /// Whole significant decimal digits, conservatively rounded down.
    pub fn significant_digits(&self) -> Option<u32> {
        // This 18-digit lower bound to log10(2) avoids rounding up without
        // discarding a decimal digit at the precisions used for large tables.
        self.significant_bits.map(|bits| {
            ((u128::from(bits.saturating_sub(1)) * 301_029_995_663_981_195)
                / 1_000_000_000_000_000_000) as u32
        })
    }

    pub fn significant_bits(&self) -> Option<u32> {
        self.significant_bits
    }

    pub fn absolute_error(&self) -> &Float {
        &self.absolute_error
    }

    pub fn exact_zero(&self) -> bool {
        self.exact_zero
    }

    pub fn quality(&self) -> ErrorQuality {
        self.error_quality
    }
}

/// Independent accuracy information for the real and imaginary components.
#[derive(Clone, Debug)]
pub struct TrackedPrecision {
    pub real: TrackedComponentPrecision,
    pub imaginary: TrackedComponentPrecision,
}

impl TrackedPrecision {
    pub fn real(&self) -> &TrackedComponentPrecision {
        &self.real
    }

    pub fn imaginary(&self) -> &TrackedComponentPrecision {
        &self.imaginary
    }
}

/// A complex value with arbitrary-precision uncertainty and its error quality.
///
/// This type deliberately does not implement the ordinary float traits: an
/// operation without an error model must not silently discard uncertainty.
#[derive(Clone, Debug)]
pub struct TrackedComplex {
    pub ball: ComplexBall,
    pub error_quality: ErrorQuality,
}

impl TrackedComplex {
    pub fn from_ball(ball: ComplexBall, error_quality: ErrorQuality) -> Self {
        Self {
            ball,
            error_quality,
        }
    }

    /// Enclose exact rational input, including conversion rounding.
    pub fn from_exact_rational(value: &Complex<Rational>, working_bits: u32) -> Self {
        Self::from_ball(
            Complex::new(
                RealBall::from_rational_bounds(&value.re, &value.re, working_bits),
                RealBall::from_rational_bounds(&value.im, &value.im, working_bits),
            ),
            ErrorQuality::Bounded,
        )
    }

    /// Enclose a real decimal value with an explicit absolute error bound.
    ///
    /// Both strings are parsed as exact decimals before outward conversion, so
    /// the supplied error is retained even when it is much smaller than the
    /// working precision. Conversion rounding is added to that error. The
    /// imaginary component is exactly zero. The caller selects `error_quality`
    /// according to the provenance of the bound; this constructor does not
    /// authenticate external data or turn an estimated error into a proof.
    ///
    /// The error must be finite and nonnegative. A zero bound is allowed for a
    /// known exact decimal, including a proven zero. Precision suffixes are
    /// rejected because accuracy is supplied explicitly by `absolute_error`.
    pub fn from_decimal_with_error(
        value: &str,
        absolute_error: &str,
        working_bits: u32,
        error_quality: ErrorQuality,
    ) -> Result<Self, String> {
        let center = parse_decimal(value, working_bits)
            .map_err(|error| format!("invalid decimal value: {error}"))?
            .value;
        let radius = parse_decimal(absolute_error, working_bits)
            .map_err(|error| format!("invalid absolute error bound: {error}"))?
            .value;
        if radius.is_negative() {
            return Err("absolute error bound must be nonnegative".into());
        }
        let ball = RealBall::from_rational_ball(&center, &radius, working_bits);
        if !ball.is_finite() {
            return Err("decimal value and error exceed the finite numerical range".into());
        }
        Ok(Self::from_ball(
            Complex::new(ball, RealBall::exact(Float::new(working_bits))),
            error_quality,
        ))
    }

    /// Construct a real estimate from a decimal sample with validated digits.
    ///
    /// The source radius is one unit in the last validated significant decimal
    /// place, plus conversion rounding. It does not decrease when working
    /// precision increases. A decimal zero uses absolute radius `10^-digits`;
    /// use `from_exact_rational` for a proven zero. The imaginary part is exact
    /// zero because this constructor explicitly receives a real input.
    pub fn from_source_decimal(
        value: &str,
        validated_decimal_digits: u32,
        working_bits: u32,
    ) -> Result<Self, String> {
        if validated_decimal_digits == 0 {
            return Err("source decimal precision must be positive".into());
        }
        let parsed = parse_decimal(value, working_bits)?;
        let radius_scale = parsed
            .magnitude
            .unwrap_or(0)
            .checked_sub(i64::from(validated_decimal_digits))
            .ok_or("source decimal uncertainty exponent is out of range")?;
        let radius = decimal_power(radius_scale);
        Ok(Self::from_ball(
            Complex::new(
                RealBall::from_rational_ball(&parsed.value, &radius, working_bits),
                RealBall::exact(Float::new(working_bits)),
            ),
            ErrorQuality::Estimated,
        ))
    }

    /// Estimate uncertainty from each component's existing stored precision.
    ///
    /// A nonzero component receives relative radius `2^(1-stored_bits)`. A zero
    /// component receives that absolute radius, since a Float alone cannot
    /// distinguish a rounded zero from a proven zero. Raising working precision
    /// preserves this source uncertainty. Nonfinite inputs remain nonfinite and
    /// are rejected by checked export.
    pub fn from_float(value: Complex<Float>, working_bits: u32) -> Self {
        fn component(value: Float, working_bits: u32) -> RealBall {
            if !value.is_finite() {
                return RealBall::exact(value);
            }
            let center = value.to_rational();
            let scale = if center.is_zero() {
                Rational::from(1)
            } else {
                center.abs()
            };
            let radius = scale * binary_power(1 - i64::from(value.prec()));
            RealBall::from_rational_ball(&center, &radius, working_bits)
        }
        Self::from_ball(
            Complex::new(
                component(value.re, working_bits),
                component(value.im, working_bits),
            ),
            ErrorQuality::Estimated,
        )
    }

    pub fn working_precision(&self) -> u32 {
        self.ball.get_precision()
    }

    pub fn is_finite(&self) -> bool {
        self.ball.is_finite()
    }

    pub fn add(&self, other: &Self) -> Self {
        Self::from_ball(
            self.ball.clone() + &other.ball,
            self.error_quality.combine(other.error_quality),
        )
    }

    pub fn sub(&self, other: &Self) -> Self {
        Self::from_ball(
            self.ball.clone() - &other.ball,
            self.error_quality.combine(other.error_quality),
        )
    }

    pub fn mul(&self, other: &Self) -> Self {
        Self::from_ball(
            self.ball.clone() * &other.ball,
            self.error_quality.combine(other.error_quality),
        )
    }

    /// Divide when the denominator enclosure excludes complex zero.
    pub fn div(&self, other: &Self) -> Result<Self, String> {
        if !self.is_finite() || !other.is_finite() {
            return Err("tracked division requires finite input enclosures".into());
        }
        if other.ball.contains_zero() {
            return Err("tracked division denominator contains zero".into());
        }
        let result = Self::from_ball(
            self.ball.clone() / &other.ball,
            self.error_quality.combine(other.error_quality),
        );
        if !result.is_finite() {
            return Err("tracked division could not produce a finite enclosure".into());
        }
        Ok(result)
    }

    pub fn neg(&self) -> Self {
        Self::from_ball(-self.ball.clone(), self.error_quality)
    }

    pub fn conj(&self) -> Self {
        Self::from_ball(self.ball.conj(), self.error_quality)
    }

    pub fn pow(&self, mut exponent: u64) -> Self {
        let mut result = self.ball.one();
        let mut base = self.ball.clone();
        while exponent != 0 {
            if exponent & 1 != 0 {
                result *= &base;
            }
            exponent >>= 1;
            if exponent != 0 {
                base = base.clone() * base;
            }
        }
        Self::from_ball(result, self.error_quality)
    }

    pub fn powi(&self, exponent: i64) -> Result<Self, String> {
        if exponent >= 0 {
            return Ok(self.pow(exponent as u64));
        }
        let one = Self::from_exact_rational(
            &Complex::new(Rational::from(1), Rational::from(0)),
            self.working_precision(),
        );
        one.div(&self.pow(exponent.unsigned_abs()))
    }

    pub fn precision(&self) -> TrackedPrecision {
        TrackedPrecision {
            real: component_precision(&self.ball.re, self.error_quality),
            imaginary: component_precision(&self.ball.im, self.error_quality),
        }
    }

    /// Round each center to independently attained precision and discard radii.
    ///
    /// The conversion checks the error including final rounding. It rejects
    /// nonfinite values, uncertain zero, and components without a reliable bit.
    /// Exact zero is preserved. Two significant bits are normally reserved for
    /// final rounding; this is not a second evaluation or a precision retry.
    /// The returned Float carries accuracy in its stored precision, but cannot
    /// carry absolute error or `ErrorQuality`; retain `precision()` when needed.
    pub fn to_float_with_precision(&self) -> Result<Complex<Float>, String> {
        self.to_float_with_max_precision(u32::MAX)
    }

    /// Checked export with a maximum precision for each nonzero component.
    ///
    /// Lowering this cap includes the extra rounding in the acceptance check.
    pub fn to_float_with_max_precision(&self, max_bits: u32) -> Result<Complex<Float>, String> {
        let rounded = self.rounded(Some(max_bits))?;
        Ok(Complex::new(rounded.ball.re.center, rounded.ball.im.center))
    }

    /// Round centers once while retaining source and final-rounding uncertainty.
    ///
    /// Each nonzero center's stored precision is no greater than the achieved
    /// precision reported for the returned enclosure. This keeps the numerical
    /// result and its accuracy metadata together without another evaluation.
    pub fn rounded(&self, max_bits: Option<u32>) -> Result<Self, String> {
        let max_bits = max_bits.unwrap_or(u32::MAX);
        if max_bits == 0 {
            return Err("maximum export precision must be positive".into());
        }
        Ok(Self::from_ball(
            Complex::new(
                rounded_component(&self.ball.re, max_bits)
                    .map_err(|e| format!("real component: {e}"))?,
                rounded_component(&self.ball.im, max_bits)
                    .map_err(|e| format!("imaginary component: {e}"))?,
            ),
            self.error_quality,
        ))
    }
}

struct ParsedDecimal {
    value: Rational,
    /// For nonzero x, the integer d such that 10^(d-1) <= |x| < 10^d.
    magnitude: Option<i64>,
}

fn parse_decimal(value: &str, working_bits: u32) -> Result<ParsedDecimal, String> {
    // Use the common parser for syntax, working precision, and exponent-range
    // validation, but never use its rounded value to construct the rational.
    let parsed = Float::parse(value, Some(working_bits))?;
    if !parsed.is_finite() || value.contains('`') {
        return Err("expected a finite decimal without a precision suffix".into());
    }
    let value = value.trim();
    let (mantissa, exponent) = value
        .split_once(['e', 'E'])
        .map_or((value, "0"), |(m, e)| (m, e));
    let exponent: i64 = exponent
        .parse()
        .map_err(|_| "decimal exponent is out of range")?;
    let fractional_digits = mantissa.split_once('.').map_or(0, |(_, f)| f.len());
    let fractional_digits = i64::try_from(fractional_digits).map_err(|_| "decimal is too long")?;
    let scale = exponent
        .checked_sub(fractional_digits)
        .ok_or("decimal exponent is out of range")?;
    let digits = mantissa.replace('.', "");
    let integer: Integer = digits.parse().map_err(|_| "invalid decimal")?;
    if integer.is_zero() {
        return Ok(ParsedDecimal {
            value: Rational::from(0),
            magnitude: None,
        });
    }
    if parsed.is_zero() {
        return Err("decimal underflows the supported exponent range".into());
    }
    let significant_digits = digits
        .trim_start_matches(['+', '-'])
        .trim_start_matches('0')
        .len();
    let magnitude = scale
        .checked_add(i64::try_from(significant_digits).map_err(|_| "decimal is too long")?)
        .ok_or("decimal magnitude exponent is out of range")?;
    Ok(ParsedDecimal {
        value: Rational::from(integer) * decimal_power(scale),
        magnitude: Some(magnitude),
    })
}

fn binary_power(exponent: i64) -> Rational {
    let power = Integer::from(2).pow(exponent.unsigned_abs());
    if exponent >= 0 {
        Rational::from(power)
    } else {
        Rational::from((Integer::one(), power))
    }
}

fn decimal_power(exponent: i64) -> Rational {
    let power = Integer::from(10).pow(exponent.unsigned_abs());
    if exponent >= 0 {
        Rational::from(power)
    } else {
        Rational::from((Integer::one(), power))
    }
}

/// Floor(log2(value)) for a positive rational, using exact integer comparison.
fn floor_log2(value: &Rational) -> i64 {
    let numerator = value.numerator_ref();
    let denominator = value.denominator_ref();
    let candidate = numerator.significant_bits() as i64 - denominator.significant_bits() as i64;
    if *value < binary_power(candidate) {
        candidate - 1
    } else {
        candidate
    }
}

fn component_precision(value: &RealBall, quality: ErrorQuality) -> TrackedComponentPrecision {
    let exact_zero = value.is_zero();
    let significant_bits = if !value.is_finite() || exact_zero {
        None
    } else if value.radius.is_zero() {
        Some(value.center.prec())
    } else {
        let radius = value.radius.to_rational().abs();
        let min_magnitude = value.center.to_rational().abs() - &radius;
        if min_magnitude <= 0 {
            None
        } else {
            let bits = floor_log2(&(min_magnitude / radius)) + 1;
            if bits < 1 {
                None
            } else {
                Some((bits as u64).min(u64::from(value.center.prec())) as u32)
            }
        }
    };
    TrackedComponentPrecision {
        significant_bits,
        absolute_error: value.radius.clone(),
        exact_zero,
        error_quality: quality,
    }
}

fn export_component(value: &RealBall, max_bits: u32) -> Result<Float, String> {
    if !value.is_finite() {
        return Err("nonfinite tracked value or uncertainty".into());
    }
    if value.is_zero() {
        return Ok(value.center.clone());
    }
    if value.radius.is_zero() {
        let mut center = value.center.clone();
        center.set_prec(center.prec().min(max_bits));
        if !center.is_finite() || center.is_zero() {
            return Err("rounding does not preserve a nonzero finite center".into());
        }
        return Ok(center);
    }
    let available = component_precision(value, ErrorQuality::Estimated)
        .significant_bits
        .ok_or("no reliable significant bit (the enclosure may contain zero)")?;
    let bits = if available == value.center.prec() && max_bits >= available {
        // The center already carries no more precision than its enclosure
        // supports. Re-exporting it must not lose two more bits each time.
        available
    } else {
        available.saturating_sub(2).max(1).min(max_bits)
    };
    let mut center = value.center.clone();
    center.set_prec(bits);
    if !center.is_finite() || center.is_zero() {
        return Err("rounding does not preserve a nonzero finite center".into());
    }
    let old_center = value.center.to_rational();
    let radius = value.radius.to_rational().abs();
    let min_magnitude = old_center.abs() - &radius;
    let rounding_error = (center.to_rational() - old_center).abs();
    if radius + rounding_error > min_magnitude * binary_power(1 - i64::from(bits)) {
        return Err("no reliable significant bit after final rounding".into());
    }
    Ok(center)
}

fn rounded_component(value: &RealBall, max_bits: u32) -> Result<RealBall, String> {
    let center = export_component(value, max_bits)?;
    if value.is_zero() {
        return Ok(RealBall::exact(center));
    }
    let radius = value.radius.to_rational().abs()
        + (center.to_rational() - value.center.to_rational()).abs();
    let radius = Float::from_rational_round(
        &radius,
        value.center.prec().max(value.radius.prec()),
        RoundingDirection::Up,
    );
    let rounded = RealBall::new(center, radius);
    let precision = component_precision(&rounded, ErrorQuality::Estimated);
    if precision
        .significant_bits
        .is_none_or(|bits| bits < rounded.center.prec())
    {
        return Err("final rounding enclosure does not establish the exported precision".into());
    }
    Ok(rounded)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exact(re: i64, im: i64, bits: u32) -> TrackedComplex {
        TrackedComplex::from_exact_rational(&Complex::new(re.into(), im.into()), bits)
    }

    #[test]
    fn exact_arithmetic_and_zero_export() {
        let a = exact(3, 2, 128);
        let b = exact(1, -1, 128);
        let result = a.mul(&b).div(&b).unwrap();
        assert!(result.ball.contains_complex(&Complex::new(
            Float::with_val(128, 3),
            Float::with_val(128, 2)
        )));
        assert_eq!(result.error_quality, ErrorQuality::Bounded);
        let zero = a.sub(&a);
        assert!(zero.precision().real.exact_zero);
        assert!(zero.precision().imaginary.exact_zero);
        assert!(zero.to_float_with_precision().unwrap().is_zero());
        assert!(a.conj().ball.im.center.is_negative());
        assert!(
            a.powi(-1)
                .unwrap()
                .mul(&a)
                .ball
                .contains(&Float::with_val(128, 1))
        );
    }

    #[test]
    fn finite_source_precision_survives_higher_working_precision() {
        let low = TrackedComplex::from_source_decimal("1.234567890123456789", 18, 128).unwrap();
        let high = TrackedComplex::from_source_decimal("1.234567890123456789", 18, 512).unwrap();
        let expected = decimal_power(-17);
        assert!(low.ball.re.radius.to_rational() >= expected);
        assert!(high.ball.re.radius.to_rational() >= expected);
        assert_eq!(
            low.precision().real.significant_bits,
            high.precision().real.significant_bits
        );
        assert_eq!(high.error_quality, ErrorQuality::Estimated);
        assert!(high.precision().imaginary.exact_zero);
    }

    #[test]
    fn explicit_decimal_bound_keeps_twenty_thousand_digit_exponent() {
        let expected = decimal_power(-20_000);
        for bits in [32, 256] {
            let value = TrackedComplex::from_decimal_with_error(
                "0",
                "1e-20000",
                bits,
                ErrorQuality::Bounded,
            )
            .unwrap();
            let radius = value.ball.re.radius.to_rational();
            assert!(radius >= expected);
            assert!(radius < &expected * &Rational::from((1001, 1000)));
            assert!(!value.precision().real.exact_zero);
            assert!(value.precision().imaginary.exact_zero);
            assert_eq!(value.error_quality, ErrorQuality::Bounded);
            assert!(value.to_float_with_precision().is_err());
        }
    }

    #[test]
    fn explicit_decimal_bound_includes_conversion_and_export_rounding() {
        let center_text = "1.23456789012345678901234567890123456789";
        let center = parse_decimal(center_text, 256).unwrap().value;
        let error = decimal_power(-25);
        for bits in [32, 256] {
            let value = TrackedComplex::from_decimal_with_error(
                center_text,
                "1e-25",
                bits,
                ErrorQuality::Estimated,
            )
            .unwrap();
            let rounded = value.rounded(Some(20)).unwrap();
            for enclosure in [&value.ball.re, &rounded.ball.re] {
                let converted_center = enclosure.center.to_rational();
                let converted_radius = enclosure.radius.to_rational();
                assert!(&converted_center - &converted_radius <= &center - &error);
                assert!(converted_center + converted_radius >= &center + &error);
            }
            assert_eq!(rounded.error_quality, ErrorQuality::Estimated);
            assert!(
                rounded.precision().real.significant_bits.unwrap() >= rounded.ball.re.center.prec()
            );
        }
        let exact_zero =
            TrackedComplex::from_decimal_with_error("0.0e-50", "0", 128, ErrorQuality::Bounded)
                .unwrap();
        assert!(exact_zero.precision().real.exact_zero);
        assert!(exact_zero.to_float_with_precision().unwrap().is_zero());
        let exact_point =
            TrackedComplex::from_decimal_with_error("1.25", "0", 128, ErrorQuality::Bounded)
                .unwrap();
        assert!(exact_point.ball.re.radius.is_zero());
    }

    #[test]
    fn explicit_decimal_bounds_reject_invalid_inputs_and_precision_suffixes() {
        for value in ["NaN", "inf", "-inf", "1e", "1/2", "1.25`20000"] {
            assert!(
                TrackedComplex::from_decimal_with_error(value, "0", 128, ErrorQuality::Bounded)
                    .is_err()
            );
        }
        for error in ["-1", "-1e-20000", "NaN", "inf", "1e", "0.01`20000"] {
            assert!(
                TrackedComplex::from_decimal_with_error("1", error, 128, ErrorQuality::Bounded)
                    .is_err()
            );
        }
        assert!(
            TrackedComplex::from_decimal_with_error("1", "0", 0, ErrorQuality::Bounded).is_err()
        );
    }

    #[test]
    fn source_decimal_fallback_preserves_significant_place_conventions() {
        for literal in ["1.23450", "123.450e-2", "0001.2345000E+0", "+.0012345e3"] {
            let value = TrackedComplex::from_source_decimal(literal, 5, 128).unwrap();
            let explicit = TrackedComplex::from_decimal_with_error(
                "1.2345",
                "1e-4",
                128,
                ErrorQuality::Estimated,
            )
            .unwrap();
            assert_eq!(value.ball, explicit.ball);
        }
        let fallback_zero = TrackedComplex::from_source_decimal("0.0e+40", 5, 128).unwrap();
        let explicit_zero =
            TrackedComplex::from_decimal_with_error("0", "1e-5", 128, ErrorQuality::Estimated)
                .unwrap();
        assert_eq!(fallback_zero.ball, explicit_zero.ball);
    }

    #[test]
    fn cancellation_retains_uncertainty_and_loses_digits() {
        let a = TrackedComplex::from_source_decimal("1.000000000001", 15, 256).unwrap();
        let b = TrackedComplex::from_source_decimal("1.000000000000", 15, 256).unwrap();
        let difference = a.sub(&b);
        assert!(difference.precision().real.significant_bits.unwrap() < 7);
        let uncertain_zero = a.sub(&a);
        assert!(!uncertain_zero.precision().real.exact_zero);
        assert!(uncertain_zero.precision().real.significant_bits.is_none());
        assert!(uncertain_zero.to_float_with_precision().is_err());
    }

    #[test]
    fn components_export_independently_and_include_rounding() {
        let re = TrackedComplex::from_source_decimal("1.23456789", 8, 256)
            .unwrap()
            .ball
            .re;
        let im = TrackedComplex::from_source_decimal(
            "0.000000000000000000000000000000000000000000000000000000000000123456789",
            5,
            256,
        )
        .unwrap()
        .ball
        .re;
        let value = TrackedComplex::from_ball(Complex::new(re, im), ErrorQuality::Estimated);
        let exported = value.to_float_with_precision().unwrap();
        assert!(exported.re.prec() > exported.im.prec());
        for (old, new) in [
            (&value.ball.re, &exported.re),
            (&value.ball.im, &exported.im),
        ] {
            let source_radius = old.radius.to_rational();
            let minimum = old.center.to_rational().abs() - &source_radius;
            let total = source_radius + (new.to_rational() - old.center.to_rational()).abs();
            assert!(total <= minimum * binary_power(1 - i64::from(new.prec())));
        }
    }

    #[test]
    fn twenty_thousand_decimal_digits_do_not_underflow() {
        let value = TrackedComplex::from_source_decimal("1.25", 20_000, 66_500).unwrap();
        assert!(!value.ball.re.radius.is_zero());
        assert!(value.precision().real.significant_bits.unwrap() > 66_400);
        let exported = value.to_float_with_precision().unwrap();
        assert!(exported.re.prec() > 66_400);
    }

    #[test]
    fn float_input_never_upgrades_source_precision_or_proves_zero() {
        let value = TrackedComplex::from_float(
            Complex::new(Float::with_val(200, 1), Float::with_val(60, 2)),
            512,
        );
        let precision = value.precision();
        assert!(precision.real.significant_bits.unwrap() < 200);
        assert!(precision.imaginary.significant_bits.unwrap() < 60);
        let zero = TrackedComplex::from_float(Complex::new(Float::new(53), Float::new(53)), 512);
        assert!(!zero.precision().real.exact_zero);
        assert!(zero.to_float_with_precision().is_err());
    }

    #[test]
    fn invalid_inputs_and_zero_containing_divisors_fail() {
        for invalid in ["NaN", "inf", "1e", "1`20"] {
            assert!(TrackedComplex::from_source_decimal(invalid, 10, 128).is_err());
        }
        assert!(TrackedComplex::from_source_decimal("1", 0, 128).is_err());
        let zero = TrackedComplex::from_source_decimal("0", 10, 128).unwrap();
        assert!(exact(1, 0, 128).div(&zero).is_err());
        assert_eq!(
            exact(1, 0, 128).add(&zero).error_quality,
            ErrorQuality::Estimated
        );
    }

    #[test]
    fn final_precision_cap_is_checked_independently() {
        let value = TrackedComplex::from_source_decimal("1.234567890123456789", 18, 256).unwrap();
        let capped = value.to_float_with_max_precision(12).unwrap();
        assert_eq!(capped.re.prec(), 12);
        assert!(capped.im.is_zero());
        assert!(value.to_float_with_max_precision(0).is_err());
        assert!(value.precision().real().significant_digits().unwrap() >= 16);
        assert_eq!(value.precision().real().quality(), ErrorQuality::Estimated);
        assert!(value.precision().imaginary().exact_zero());
    }

    #[test]
    fn powers_include_signed_exponent_extremes() {
        let one = exact(1, 0, 128);
        assert!(one.pow(u64::MAX).ball.re.is_one());
        assert!(one.powi(i64::MIN).unwrap().ball.re.is_one());
        assert!(one.neg().ball.re.center.is_negative());
    }

    #[test]
    fn rounded_metadata_accounts_for_adversarial_binary_boundaries() {
        for text in [
            "0.999999999999999999999999",
            "1.000000000000000000000001",
            "1.499999999999999999999999",
            "1.500000000000000000000001",
            "1.999999999999999999999999",
            "-1.999999999999999999999999",
        ] {
            let input = TrackedComplex::from_source_decimal(text, 24, 256).unwrap();
            for cap in [8, 12, 24, 64, 128] {
                let rounded = input.rounded(Some(cap)).unwrap();
                let precision = rounded.precision();
                assert!(precision.real.significant_bits.unwrap() >= rounded.ball.re.center.prec());
                // Check containment using exact endpoints; the deliberately
                // reduced center precision must not affect this assertion.
                let shift = (rounded.ball.re.center.to_rational()
                    - input.ball.re.center.to_rational())
                .abs();
                assert!(
                    rounded.ball.re.radius.to_rational()
                        >= input.ball.re.radius.to_rational() + shift
                );
                assert!(precision.imaginary.exact_zero);
                let repeated = rounded.rounded(Some(cap)).unwrap();
                assert_eq!(
                    repeated.ball.re.center.prec(),
                    rounded.ball.re.center.prec()
                );
                assert_eq!(repeated.ball.re.radius, rounded.ball.re.radius);
            }
        }
    }

    #[cfg(feature = "float-mpfr")]
    #[test]
    fn exact_center_rounding_overflow_is_rejected_without_panicking() {
        use crate::domains::backend::float::MultiPrecisionFloat;

        let center =
            Float::from_raw(MultiPrecisionFloat::with_val(4, 1.875) << (rug::float::exp_max() - 1));
        let input = TrackedComplex::from_ball(
            Complex::new(RealBall::exact(center), RealBall::exact(Float::new(4))),
            ErrorQuality::Bounded,
        );
        assert!(input.is_finite());
        // The one-bit rounded center overflows before an export radius is built.
        assert!(input.to_float_with_max_precision(1).is_err());
    }
}
