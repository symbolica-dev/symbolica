//! Fixed-precision tracked evaluation of the built-in transcendental functions.
//!
//! MPFR-directed elementary operations and the convergent polylogarithm series
//! and polygamma expansions below produce enclosures. Existing general complex
//! gamma and zeta algorithms do not expose their internal error budgets, so
//! those cases return an explicit unsupported-operation error. A requested
//! working precision is never substituted for an actual error bound.

use crate::{
    atom::{AtomView, Symbol},
    domains::{
        float::{
            Complex, ComplexBall, ErrorQuality, Float, FloatLike, Real, RealBall,
            RoundingDirection, SingleFloat, TrackedComplex,
        },
        integer::Integer,
        rational::Rational,
    },
};

#[derive(Clone, Copy)]
enum Unary {
    Exp,
    Log,
    Sin,
    Cos,
    Gamma,
    Digamma,
    Zeta,
}

fn point(value: i64, bits: u32) -> RealBall {
    let rational = Rational::from(value);
    RealBall::from_rational_bounds(&rational, &rational, bits)
}

fn real(value: RealBall, quality: ErrorQuality) -> TrackedComplex {
    let bits = value.center.prec();
    TrackedComplex {
        ball: Complex::new(value, point(0, bits)),
        error_quality: quality,
    }
}

fn quality(args: &[&TrackedComplex]) -> ErrorQuality {
    if args
        .iter()
        .any(|x| x.error_quality == ErrorQuality::Estimated)
    {
        ErrorQuality::Estimated
    } else {
        ErrorQuality::Bounded
    }
}

fn checked(ball: ComplexBall, error_quality: ErrorQuality) -> Result<TrackedComplex, String> {
    if !ball.re.is_finite() || !ball.im.is_finite() {
        Err("tracked evaluation produced a nonfinite center or uncertainty".into())
    } else {
        Ok(TrackedComplex {
            ball,
            error_quality,
        })
    }
}

fn validate(arg: &TrackedComplex, bits: u32) -> Result<(), String> {
    if bits < 2 {
        return Err("tracked transcendental evaluation requires at least two working bits".into());
    }
    if !arg.ball.re.is_finite() || !arg.ball.im.is_finite() {
        return Err("tracked transcendental argument has a nonfinite center or uncertainty".into());
    }
    if arg.ball.re.radius.is_negative() || arg.ball.im.radius.is_negative() {
        return Err("tracked transcendental argument has negative uncertainty".into());
    }
    Ok(())
}

fn real_zero(x: &RealBall) -> bool {
    x.center.is_fully_zero() && x.radius.is_fully_zero()
}

fn is_real(x: &ComplexBall) -> bool {
    real_zero(&x.im)
}

fn widen(value: RealBall, radius: &Float, bits: u32) -> RealBall {
    RealBall::new(
        value.center,
        value.radius.add_round(radius, bits, RoundingDirection::Up),
    )
}

/// A correctly rounded point enclosure. Resizing the argument before applying
/// the function would discard information, so MPFR's incomplete operation is
/// completed directly at the requested output precision.
fn directed_unary(which: Unary, value: &Float, bits: u32) -> Result<RealBall, String> {
    #[cfg(feature = "float-mpfr")]
    {
        use crate::domains::backend::float::MultiPrecisionFloat;
        use rug::float::Round;
        let x = value.as_raw();
        let evaluate = |round| -> Float {
            match which {
                Unary::Exp => MultiPrecisionFloat::with_val_round(bits, x.exp_ref(), round).0,
                Unary::Log => MultiPrecisionFloat::with_val_round(bits, x.ln_ref(), round).0,
                Unary::Sin => MultiPrecisionFloat::with_val_round(bits, x.sin_ref(), round).0,
                Unary::Cos => MultiPrecisionFloat::with_val_round(bits, x.cos_ref(), round).0,
                Unary::Gamma => MultiPrecisionFloat::with_val_round(bits, x.gamma_ref(), round).0,
                Unary::Digamma => {
                    MultiPrecisionFloat::with_val_round(bits, x.digamma_ref(), round).0
                }
                Unary::Zeta => MultiPrecisionFloat::with_val_round(bits, x.zeta_ref(), round).0,
            }
            .into()
        };
        let lower = evaluate(Round::Down);
        let upper = evaluate(Round::Up);
        if !lower.is_finite() || !upper.is_finite() {
            Err("tracked special function is singular or exceeds the numerical range".into())
        } else {
            Ok(RealBall::from_bounds(lower, upper))
        }
    }
    #[cfg(not(feature = "float-mpfr"))]
    {
        let _ = (which, value, bits);
        Err("tracked transcendental bounds require the float-mpfr backend".into())
    }
}

fn monotone(
    which: Unary,
    value: &RealBall,
    increasing: bool,
    bits: u32,
) -> Result<RealBall, String> {
    let left = directed_unary(which, &value.lower_bound(), bits)?;
    let right = directed_unary(which, &value.upper_bound(), bits)?;
    Ok(if increasing {
        RealBall::from_bounds(left.lower_bound(), right.upper_bound())
    } else {
        RealBall::from_bounds(right.lower_bound(), left.upper_bound())
    })
}

fn trig(which: Unary, value: &RealBall, bits: u32) -> Result<RealBall, String> {
    // Both sine and cosine are globally 1-Lipschitz on the real axis.
    let center = directed_unary(which, &value.center, bits)?;
    let expanded = widen(center, &value.radius, bits);
    let one = Float::with_val(bits, 1);
    let minus_one = -one.clone();
    let lower = expanded.lower_bound();
    let upper = expanded.upper_bound();
    Ok(RealBall::from_bounds(
        if lower < minus_one { minus_one } else { lower },
        if upper > one { one } else { upper },
    ))
}

fn square(value: &RealBall, bits: u32) -> RealBall {
    let lower = value.lower_bound();
    let upper = value.upper_bound();
    let zero = Float::new(bits);
    let (small, large) = if lower >= zero {
        (lower, upper)
    } else if upper <= zero {
        (-upper, -lower)
    } else {
        let large = if -lower.clone() > upper {
            -lower
        } else {
            upper
        };
        (zero, large)
    };
    RealBall::from_bounds(
        small.mul_round(&small, bits, RoundingDirection::Down),
        large.mul_round(&large, bits, RoundingDirection::Up),
    )
}

fn norm(value: &ComplexBall, bits: u32) -> RealBall {
    (square(&value.re, bits) + square(&value.im, bits)).sqrt()
}

fn exp_ball(value: &ComplexBall, bits: u32) -> Result<ComplexBall, String> {
    let amplitude = monotone(Unary::Exp, &value.re, true, bits)?;
    Ok(Complex::new(
        amplitude.clone() * trig(Unary::Cos, &value.im, bits)?,
        amplitude * trig(Unary::Sin, &value.im, bits)?,
    ))
}

fn log_ball(value: &ComplexBall, bits: u32) -> Result<ComplexBall, String> {
    let zero = Float::new(bits);
    if value.contains_zero() {
        return Err("logarithm argument enclosure contains zero".into());
    }
    if is_real(value) {
        if value.re.is_strictly_positive() {
            return Ok(Complex::new(
                monotone(Unary::Log, &value.re, true, bits)?,
                point(0, bits),
            ));
        }
        // The principal logarithm of a negative real number has imaginary part +pi.
        return Ok(Complex::new(
            monotone(Unary::Log, &(-value.re.clone()), true, bits)?,
            constant(Symbol::PI, bits)?.ball.re,
        ));
    }
    if value.re.lower_bound() < zero
        && value.im.lower_bound() <= zero
        && value.im.upper_bound() >= zero
    {
        return Err("logarithm argument enclosure crosses the principal branch cut".into());
    }
    #[cfg(feature = "float-mpfr")]
    {
        use crate::domains::backend::float::MultiPrecisionFloat;
        use rug::float::Round;
        // Away from zero and the cut, angle extrema on a rectangle occur at
        // its corners (including degenerate rectangles).
        let mut angles = Vec::with_capacity(8);
        for y in [value.im.lower_bound(), value.im.upper_bound()] {
            for x in [value.re.lower_bound(), value.re.upper_bound()] {
                for round in [Round::Down, Round::Up] {
                    let angle: Float = MultiPrecisionFloat::with_val_round(
                        bits,
                        y.as_raw().atan2_ref(x.as_raw()),
                        round,
                    )
                    .0
                    .into();
                    angles.push(angle);
                }
            }
        }
        let lower = angles
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            .clone();
        let upper = angles
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap()
            .clone();
        let modulus_squared = square(&value.re, bits) + square(&value.im, bits);
        Ok(Complex::new(
            monotone(Unary::Log, &modulus_squared, true, bits)? / point(2, bits),
            RealBall::from_bounds(lower, upper),
        ))
    }
    #[cfg(not(feature = "float-mpfr"))]
    Err("tracked complex logarithm bounds require the float-mpfr backend".into())
}

fn sin_cos_ball(value: &ComplexBall, cosine: bool, bits: u32) -> Result<ComplexBall, String> {
    let sin = trig(Unary::Sin, &value.re, bits)?;
    let cos = trig(Unary::Cos, &value.re, bits)?;
    let exp = monotone(Unary::Exp, &value.im, true, bits)?;
    let inverse_exp = monotone(Unary::Exp, &(-value.im.clone()), true, bits)?;
    let cosh = (exp.clone() + inverse_exp.clone()) / point(2, bits);
    let sinh = (exp - inverse_exp) / point(2, bits);
    Ok(if cosine {
        Complex::new(cos * cosh, -(sin * sinh))
    } else {
        Complex::new(sin * cosh, cos * sinh)
    })
}

fn constant(symbol: Symbol, bits: u32) -> Result<TrackedComplex, String> {
    if bits < 2 {
        return Err("tracked constants require at least two working bits".into());
    }
    #[cfg(feature = "float-mpfr")]
    {
        use crate::domains::backend::float::{Constant, MultiPrecisionFloat};
        use rug::float::Round;
        let endpoint = |round| -> Float {
            if symbol == Symbol::E {
                MultiPrecisionFloat::with_val_round(
                    bits,
                    MultiPrecisionFloat::with_val(bits, 1).exp_ref(),
                    round,
                )
                .0
            } else {
                let constant = if symbol == Symbol::PI {
                    Constant::Pi
                } else {
                    Constant::Euler
                };
                MultiPrecisionFloat::with_val_round(bits, constant, round).0
            }
            .into()
        };
        Ok(real(
            RealBall::from_bounds(endpoint(Round::Down), endpoint(Round::Up)),
            ErrorQuality::Bounded,
        ))
    }
    #[cfg(not(feature = "float-mpfr"))]
    {
        let _ = symbol;
        Err("tracked constant bounds require the float-mpfr backend".into())
    }
}

pub(crate) fn evaluate_builtin_constant_tracked(
    symbol: Symbol,
    tags: &[AtomView<'_>],
    bits: u32,
) -> Option<Result<TrackedComplex, String>> {
    if symbol != Symbol::PI && symbol != Symbol::E && symbol != super::euler_gamma() {
        return None;
    }
    Some(if tags.is_empty() {
        constant(symbol, bits)
    } else {
        Err("built-in numerical constant does not accept tags".into())
    })
}

fn gamma_positive_real(value: &RealBall, bits: u32) -> Result<RealBall, String> {
    let middle = directed_unary(Unary::Gamma, &value.center, bits)?;
    if value.radius.is_fully_zero() {
        return Ok(middle);
    }
    let lower = value.lower_bound();
    let upper = value.upper_bound();
    // log Gamma is convex, and digamma is increasing on the positive axis.
    // Thus these endpoint maxima bound |Gamma'| throughout the interval.
    let gl = directed_unary(Unary::Gamma, &lower, bits)?.upper_bound();
    let gu = directed_unary(Unary::Gamma, &upper, bits)?.upper_bound();
    let gamma_max = if gl > gu { gl } else { gu };
    let dl = directed_unary(Unary::Digamma, &lower, bits)?
        .norm()
        .upper_bound();
    let du = directed_unary(Unary::Digamma, &upper, bits)?
        .norm()
        .upper_bound();
    let digamma_max = if dl > du { dl } else { du };
    let sensitivity = gamma_max.mul_round(&digamma_max, bits, RoundingDirection::Up);
    Ok(widen(
        middle,
        &sensitivity.mul_round(&value.radius, bits, RoundingDirection::Up),
        bits,
    ))
}

fn gamma_tracked(arg: &TrackedComplex, bits: u32) -> Result<TrackedComplex, String> {
    let mut shifted = arg.ball.clone();
    let mut denominator = Complex::new(point(1, bits), point(0, bits));
    let mut shifts = 0;
    while shifted.re.lower_bound() <= Float::with_val(bits, 1) {
        if shifts == 4096 {
            return Err("tracked gamma recurrence exceeds its 4096-shift budget".into());
        }
        if shifted.contains_zero() {
            return Err("gamma argument enclosure contains a pole".into());
        }
        denominator *= shifted.clone();
        shifted.re += point(1, bits);
        shifts += 1;
    }
    if !is_real(&shifted) {
        return Err("tracked complex gamma is unavailable: its numerical algorithm does not expose an error bound".into());
    }
    let result = Complex::new(gamma_positive_real(&shifted.re, bits)?, point(0, bits));

    if denominator.contains_zero() {
        return Err("gamma recurrence denominator enclosure contains zero".into());
    }
    checked(result / denominator, arg.error_quality)
}

fn polygamma_tracked(
    order: u32,
    arg: &TrackedComplex,
    bits: u32,
) -> Result<TrackedComplex, String> {
    if order > 4096 {
        return Err("tracked polygamma supports orders through 4096".into());
    }
    let cutoff = bits
        .checked_add(order)
        .and_then(|n| n.checked_div(2))
        .and_then(|n| n.checked_add(16))
        .ok_or("tracked polygamma precision exceeds the recurrence budget")?;
    let max_shifts = bits
        .checked_mul(8)
        .and_then(|n| n.checked_add(1024))
        .ok_or("tracked polygamma precision exceeds the recurrence budget")?;
    let mut shifted = arg.ball.clone();
    let mut correction = Complex::new(point(0, bits), point(0, bits));
    let mut shifts = 0;
    while shifted.re.lower_bound() < Float::with_val(bits, cutoff) {
        if shifts == max_shifts || shifted.contains_zero() {
            return Err(
                "polygamma argument enclosure contains a pole or exceeds the recurrence budget"
                    .into(),
            );
        }
        let inverse = Complex::new(point(1, bits), point(0, bits)) / shifted.clone();
        correction += if order == 0 {
            -inverse
        } else {
            inverse.pow(u64::from(order) + 1)
        };
        shifted.re += point(1, bits);
        shifts += 1;
    }
    // DLMF 5.11.2 and 5.15.9, with the same Re(z)>0 remainder
    // majorant as the ordinary evaluator. Ball arithmetic additionally retains
    // every rounding error and the full source interval, including recurrence.
    let inverse = Complex::new(point(1, bits), point(0, bits)) / shifted.clone();
    let inverse_squared = inverse.pow(2);
    let inverse_sigma = point(1, bits) / RealBall::exact(shifted.re.lower_bound());
    let inverse_sigma_squared = inverse_sigma.pow(2);
    let mut power = inverse.pow(u64::from(order) + 2);
    let mut bound_power = inverse_sigma.pow(u64::from(order) + 2);
    let mut sum = if order == 0 {
        log_ball(&shifted, bits)? - inverse.clone() / point(2, bits)
    } else {
        inverse.pow(u64::from(order)) / point(i64::from(order), bits)
            + inverse.pow(u64::from(order) + 1) / point(2, bits)
    };
    let mut rising = Rational::from(order + 1);
    let mut factorial = Rational::from(2);
    let tolerance = Float::with_val(bits, 0.5).pow(u64::from(bits) + 8);
    let max_terms = bits
        .checked_add(32)
        .ok_or("tracked polygamma precision exceeds the term budget")?;
    for k in 1..=max_terms {
        let bernoulli = super::bernoulli_number(2 * k);
        let coefficient = if order == 0 {
            -bernoulli / Rational::from(2 * k)
        } else {
            bernoulli * &rising / &factorial
        };
        let coefficient = RealBall::from_rational_bounds(&coefficient, &coefficient, bits);
        sum += power.clone() * coefficient.clone();
        let remainder = (coefficient.norm() * bound_power.clone()).upper_bound();
        if remainder <= tolerance {
            let mut total = sum + correction;
            total.re = widen(total.re, &remainder, bits);
            if !is_real(&arg.ball) {
                total.im = widen(total.im, &remainder, bits);
            }
            if order > 0 {
                let factor = Rational::from(Integer::factorial(order));
                let factor = if order.is_multiple_of(2) {
                    -factor
                } else {
                    factor
                };
                total *= RealBall::from_rational_bounds(&factor, &factor, bits);
            }
            return checked(total, arg.error_quality);
        }
        power *= inverse_squared.clone();
        bound_power *= inverse_sigma_squared.clone();
        rising *= Rational::from(order + 2 * k) * Rational::from(order + 2 * k + 1);
        factorial *= Rational::from(2 * k + 1) * Rational::from(2 * k + 2);
    }
    Err("tracked polygamma expansion exhausted its fixed-precision remainder budget".into())
}

fn zeta_tracked(arg: &TrackedComplex, bits: u32) -> Result<TrackedComplex, String> {
    if is_real(&arg.ball) {
        if arg.ball.re.radius.is_fully_zero() {
            #[cfg(feature = "float-mpfr")]
            if let Some(order) = arg
                .ball
                .re
                .center
                .try_to_rational()
                .and_then(|r| {
                    if r.is_integer() {
                        r.numerator().to_i64()
                    } else {
                        None
                    }
                })
                .and_then(|n| u32::try_from(n).ok())
            {
                use crate::domains::backend::float::MultiPrecisionFloat;
                use rug::float::Round;
                let lower: Float = MultiPrecisionFloat::with_val_round(
                    bits,
                    MultiPrecisionFloat::zeta_u(order),
                    Round::Down,
                )
                .0
                .into();
                let upper: Float = MultiPrecisionFloat::with_val_round(
                    bits,
                    MultiPrecisionFloat::zeta_u(order),
                    Round::Up,
                )
                .0
                .into();
                if !lower.is_finite() || !upper.is_finite() {
                    return Err("zeta argument is its pole at one".into());
                }
                return Ok(real(RealBall::from_bounds(lower, upper), arg.error_quality));
            }
            return Ok(real(
                directed_unary(Unary::Zeta, &arg.ball.re.center, bits)?,
                arg.error_quality,
            ));
        }
        if arg.ball.re.lower_bound() <= Float::with_val(bits, 1) {
            return Err(
                "tracked zeta with uncertain real input requires the entire interval to exceed one"
                    .into(),
            );
        }
        return Ok(real(
            monotone(Unary::Zeta, &arg.ball.re, false, bits)?,
            arg.error_quality,
        ));
    }
    Err("tracked complex zeta is unavailable: its numerical algorithm does not expose an error bound".into())
}

/// Fixed-point enclosure for an exact half argument. Every retained positive
/// term is rounded down by less than one unit of 2^-bits. There are at most
/// `bits` retained terms, and the remaining tail is below 2^-bits because
/// 1/k^order <= 1. The integer sum and its error budget are both exact.
fn polylog_half(order: u32, bits: u32, error_quality: ErrorQuality) -> TrackedComplex {
    let scale = Integer::one() << bits;
    let mut numerator = scale.clone();
    let mut sum = Integer::zero();
    for k in 1..=bits {
        numerator >>= 1u32;
        let term = &numerator / Integer::from(k).pow(u64::from(order));
        if term.is_zero() {
            // Subsequent scaled terms decrease monotonically and also round
            // to zero. The original `bits`-term error budget still covers them.
            break;
        }
        sum += term;
    }
    let upper = &sum + Integer::from(u64::from(bits) + 1);
    let lower = Rational::from((sum, scale.clone()));
    let upper = Rational::from((upper, scale));
    real(
        RealBall::from_rational_bounds(&lower, &upper, bits),
        error_quality,
    )
}

fn polylog_tracked(order: i64, arg: &TrackedComplex, bits: u32) -> Result<TrackedComplex, String> {
    let one = Complex::new(point(1, bits), point(0, bits));
    if order == 0 {
        let denominator = one - arg.ball.clone();
        if denominator.contains_zero() {
            return Err("polylogarithm Li_0 argument enclosure contains its pole at one".into());
        }
        return checked(arg.ball.clone() / denominator, arg.error_quality);
    }
    if order == 1 {
        return checked(
            -log_ball(&(one - arg.ball.clone()), bits)?,
            arg.error_quality,
        );
    }
    if order < 0 {
        return Err("tracked polylogarithms currently support nonnegative integer orders".into());
    }
    if order > 4096 {
        return Err("tracked polylogarithm order exceeds 4096".into());
    }
    if arg.ball.is_fully_zero() {
        return checked(arg.ball.clone(), arg.error_quality);
    }
    if is_real(&arg.ball)
        && arg.ball.re.radius.is_fully_zero()
        && arg.ball.re.center == Float::with_val(bits, 0.5)
    {
        return Ok(polylog_half(order as u32, bits, arg.error_quality));
    }
    if is_real(&arg.ball) && (arg.ball.re.is_one() || (-arg.ball.re.clone()).is_one()) {
        let value = zeta_tracked(&real(point(order, bits), arg.error_quality), bits)?;
        if arg.ball.re.is_one() {
            return Ok(value);
        }
        let coefficient =
            Rational::from((Integer::one(), Integer::from(2).pow((order - 1) as u64)))
                - Rational::from(1);
        let coefficient = RealBall::from_rational_bounds(&coefficient, &coefficient, bits);
        return checked(value.ball * coefficient, arg.error_quality);
    }
    let radius = norm(&arg.ball, bits).upper_bound();
    if radius >= Float::with_val(bits, 1) {
        return Err(
            "tracked polylogarithm series requires the entire argument enclosure inside |z| < 1"
                .into(),
        );
    }
    // The scalar majorant uses |z| <= r, and for s>=2 its tail after N
    // terms is at most r^(N+1)/((N+1)^s (1-r)). All arithmetic, including
    // the accumulated source uncertainty, remains at the selected precision.
    let r = RealBall::exact(radius);
    let gap = point(1, bits) - r.clone();
    let tolerance = Float::with_val(bits, 0.5).pow(u64::from(bits) + 8);
    let max_terms = bits.saturating_mul(64).max(1024);
    let mut power = arg.ball.clone();
    let mut r_power = r.clone();
    let mut sum = Complex::new(point(0, bits), point(0, bits));
    for n in 1..=max_terms {
        let denominator = Rational::from(Integer::from(n).pow(order as u64));
        let denominator = RealBall::from_rational_bounds(&denominator, &denominator, bits);
        sum += power.clone() / denominator;
        r_power *= r.clone();
        let next_denominator = Rational::from(Integer::from(u64::from(n) + 1).pow(order as u64));
        let next_denominator =
            RealBall::from_rational_bounds(&next_denominator, &next_denominator, bits);
        let tail = (r_power.clone() / (next_denominator * gap.clone())).upper_bound();
        if tail <= tolerance {
            sum.re = widen(sum.re, &tail, bits);
            if !is_real(&arg.ball) {
                sum.im = widen(sum.im, &tail, bits);
            }
            return checked(sum, arg.error_quality);
        }
        power *= arg.ball.clone();
    }
    Err("tracked polylogarithm series exhausted its fixed-precision term budget".into())
}

pub(crate) fn pow_tracked(
    base: &TrackedComplex,
    exponent: &TrackedComplex,
    bits: u32,
) -> Result<TrackedComplex, String> {
    validate(base, bits)?;
    validate(exponent, bits)?;
    if base.ball.is_fully_zero()
        && is_real(&exponent.ball)
        && exponent.ball.re.is_strictly_positive()
    {
        return checked(
            Complex::new(point(0, bits), point(0, bits)),
            quality(&[base, exponent]),
        );
    }
    let logarithm = log_ball(&base.ball, bits)?;
    checked(
        exp_ball(&(exponent.ball.clone() * logarithm), bits)?,
        quality(&[base, exponent]),
    )
}

pub(crate) fn evaluate_builtin_tracked(
    symbol: Symbol,
    tags: &[AtomView<'_>],
    args: &[TrackedComplex],
    bits: u32,
) -> Option<Result<TrackedComplex, String>> {
    let supported = [
        Symbol::EXP,
        Symbol::LOG,
        Symbol::SIN,
        Symbol::COS,
        Symbol::SQRT,
        Symbol::CONJ,
        Symbol::ABS,
    ]
    .contains(&symbol)
        || symbol == super::tan()
        || symbol == super::gamma()
        || symbol == super::polygamma()
        || symbol == super::zeta()
        || symbol == super::polylog();
    if !supported {
        return None;
    }
    Some((|| {
        let [arg] = args else {
            return Err("tracked built-in function requires exactly one numerical argument".into());
        };
        validate(arg, bits)?;
        if symbol == super::polygamma() {
            let [tag] = tags else {
                return Err("polygamma requires one exact nonnegative integer order tag".into());
            };
            let order = u32::try_from(*tag)
                .map_err(|_| "polygamma order must be an exact nonnegative integer")?;
            return polygamma_tracked(order, arg, bits);
        }
        if symbol == super::polylog() {
            let [tag] = tags else {
                return Err("polylog requires one exact integer order tag".into());
            };
            let order = i64::try_from(*tag)
                .map_err(|_| "tracked polylog order must be an exact integer")?;
            return polylog_tracked(order, arg, bits);
        }
        if !tags.is_empty() {
            return Err("this tracked built-in function does not accept tags".into());
        }
        if symbol == super::gamma() {
            return gamma_tracked(arg, bits);
        }
        if symbol == super::zeta() {
            return zeta_tracked(arg, bits);
        }
        let ball = match symbol {
            s if s == Symbol::CONJ => arg.ball.conj(),
            s if s == Symbol::ABS => Complex::new(norm(&arg.ball, bits), point(0, bits)),
            s if s == Symbol::EXP => exp_ball(&arg.ball, bits)?,
            s if s == Symbol::LOG => log_ball(&arg.ball, bits)?,
            s if s == Symbol::SIN => sin_cos_ball(&arg.ball, false, bits)?,
            s if s == Symbol::COS => sin_cos_ball(&arg.ball, true, bits)?,
            s if s == Symbol::SQRT && is_real(&arg.ball) => {
                if arg.ball.re.lower_bound() >= Float::new(bits) {
                    Complex::new(arg.ball.re.sqrt(), point(0, bits))
                } else if arg.ball.re.upper_bound() <= Float::new(bits) {
                    Complex::new(point(0, bits), (-arg.ball.re.clone()).sqrt())
                } else {
                    return Err(
                        "square-root argument interval crosses its branch point at zero".into(),
                    );
                }
            }
            s if s == Symbol::SQRT => {
                exp_ball(&(log_ball(&arg.ball, bits)? / point(2, bits)), bits)?
            }
            _ => {
                let cosine = sin_cos_ball(&arg.ball, true, bits)?;
                if cosine.contains_zero() {
                    return Err("tangent denominator enclosure contains zero".into());
                }
                sin_cos_ball(&arg.ball, false, bits)? / cosine
            }
        };
        checked(ball, arg.error_quality)
    })())
}

#[cfg(test)]
mod tests;
