//! Analytic argument derivatives for local special-function error estimates.
//!
//! These are derivatives of the selected branch at the center, not bounds over
//! an uncertainty neighborhood. The caller retains the pole, cut, and curvature
//! checks and falls back to numerical probes when this helper declines.

use super::{
    Complex, Float, FloatLike, Real, SpecialFunction, complex_float_to_integer, complex_one,
    finite, polygamma_numeric_eval,
};
use crate::domains::backend::float::Constant;

fn real(value: f64, precision: u32) -> Complex<Float> {
    Complex::new(Float::with_val(precision, value), Float::new(precision))
}

fn bessel_at_zero(
    function: SpecialFunction,
    order: &Complex<Float>,
    precision: u32,
) -> Option<(Complex<Float>, Complex<Float>)> {
    let n = complex_float_to_integer(order)?;
    let (first, second) = match function {
        SpecialFunction::BesselJ => (
            match n {
                -1 => -0.5,
                1 => 0.5,
                _ => 0.,
            },
            match n {
                0 => -0.5,
                -2 | 2 => 0.25,
                _ => 0.,
            },
        ),
        SpecialFunction::BesselI => (
            if matches!(n, -1 | 1) { 0.5 } else { 0. },
            match n {
                0 => 0.5,
                -2 | 2 => 0.25,
                _ => 0.,
            },
        ),
        // Y and K are singular at zero; noninteger J/I orders need their
        // branch-dependent limiting behavior rather than integer power series.
        _ => return None,
    };
    Some((real(first, precision), real(second, precision)))
}

/// Return the first and second derivatives with respect to the argument.
/// `origin` is the function value at `argument`, evaluated at `precision`.
pub(super) fn derivatives(
    function: SpecialFunction,
    order: Option<&Complex<Float>>,
    argument: &Complex<Float>,
    origin: &Complex<Float>,
    precision: u32,
) -> Option<(Complex<Float>, Complex<Float>)> {
    if precision == 0 || !finite(argument) || !finite(origin) || order.is_some_and(|n| !finite(n)) {
        return None;
    }
    let one = complex_one(precision);
    let two = Float::with_val(precision, 2);
    let at_zero = argument.is_fully_zero();
    let (first, second) = match function {
        SpecialFunction::Gamma => {
            // DLMF 5.2.2: psi = Gamma'/Gamma. Differentiation gives
            // Gamma'' = Gamma * (psi^2 + psi'). Reuse the central Gamma value.
            if origin.is_fully_zero() {
                return None; // Gamma has no zeros: a zero value has underflowed.
            }
            let psi = polygamma_numeric_eval(0, argument, precision);
            let psi_prime = polygamma_numeric_eval(1, argument, precision);
            if !finite(&psi) || !finite(&psi_prime) {
                return None;
            }
            (
                origin.clone() * &psi,
                origin.clone() * (psi.clone() * psi + psi_prime),
            )
        }
        SpecialFunction::Erf => {
            // DLMF 7.10.1: erf' = 2 exp(-z^2)/sqrt(pi), erf'' = -2z erf'.
            let first = (-(argument.clone() * argument)).exp()
                * (two.clone() / Float::with_val(precision, Constant::Pi).sqrt());
            let second = -(argument.clone() * &first) * &two;
            if at_zero && finite(&first) && !first.is_fully_zero() {
                return Some((first, second));
            }
            (first, second)
        }
        SpecialFunction::Polygamma(n) => {
            // psi^(n+1) and psi^(n+2), by the definition in DLMF 5.15.
            // Evaluate the larger order first so a backend order limit declines
            // before doing the other potentially expensive evaluation.
            let second = polygamma_numeric_eval(n.checked_add(2)?, argument, precision);
            if !finite(&second) {
                return None;
            }
            (
                polygamma_numeric_eval(n.checked_add(1)?, argument, precision),
                second,
            )
        }
        SpecialFunction::Polylog => {
            let order = order?;
            if at_zero {
                // The defining series (DLMF 25.12.10) gives Li_s'(0)=1 and
                // Li_s''(0)=2^(1-s); division by z would lose these limits.
                let second = real(2., precision).powf(&(one.clone() - order));
                return (finite(&second) && !second.is_fully_zero()).then_some((one, second));
            }
            let lower = function.value(Some(&(order.clone() - &one)), argument, precision)?;
            let lower_two =
                function.value(Some(&(order.clone() - &one - &one)), argument, precision)?;
            if !finite(&lower) || !finite(&lower_two) {
                return None;
            }
            // Differentiate Li_s'=Li_(s-1)/z. Divide sequentially, avoiding
            // overflow or underflow of an explicitly formed z^2.
            (
                lower.clone() / argument,
                (lower_two - lower) / argument / argument,
            )
        }
        SpecialFunction::BesselJ
        | SpecialFunction::BesselY
        | SpecialFunction::BesselI
        | SpecialFunction::BesselK => {
            let order = order?;
            if at_zero {
                // DLMF 10.2.2/10.25.2 and integer-order reflection identities.
                return bessel_at_zero(function, order, precision);
            }
            let below = function.value(Some(&(order.clone() - &one)), argument, precision)?;
            let above = function.value(Some(&(order.clone() + &one)), argument, precision)?;
            if !finite(&below) || !finite(&above) {
                return None;
            }
            // DLMF 10.6.1 and 10.29.1: adjacent-order derivative identities.
            let first = match function {
                SpecialFunction::BesselJ | SpecialFunction::BesselY => (below - above) / &two,
                SpecialFunction::BesselI => (below + above) / &two,
                SpecialFunction::BesselK => -(below + above) / &two,
                _ => unreachable!(),
            };
            // Bessel/modified Bessel equations (DLMF 10.2.1/10.25.1):
            // f'' = ((nu/z)^2 -/+ 1) f - f'/z.
            let ratio = order.clone() / argument;
            let squared = ratio.clone() * ratio;
            let factor = match function {
                SpecialFunction::BesselJ | SpecialFunction::BesselY => squared - &one,
                _ => squared + &one,
            };
            let second = factor * origin - first.clone() / argument;
            (first, second)
        }
        // No numerical zeta-derivative backend is available yet.
        SpecialFunction::Zeta => return None,
    };
    // Apart from the explicit power-series limits above, a fully zero result
    // may be underflow or cancellation. Do not interpret it as exact local
    // insensitivity; stationary points can use the existing probe fallback.
    (finite(&first) && finite(&second) && !first.is_fully_zero() && !second.is_fully_zero())
        .then_some((first, second))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domains::float::RealLike;

    #[test]
    fn analytic_bessel_zero_limits_cover_signed_orders() {
        let precision = 160;
        let zero = real(0., precision);
        for n in -3..=3 {
            let order = real(f64::from(n), precision);
            for function in [SpecialFunction::BesselJ, SpecialFunction::BesselI] {
                let origin = function.value(Some(&order), &zero, precision).unwrap();
                let (first, second) =
                    derivatives(function, Some(&order), &zero, &origin, precision).unwrap();
                let is_j = matches!(function, SpecialFunction::BesselJ);
                let expected_first = match n {
                    -1 if is_j => -0.5,
                    -1 | 1 => 0.5,
                    _ => 0.,
                };
                let expected_second = match n {
                    0 if is_j => -0.5,
                    0 => 0.5,
                    -2 | 2 => 0.25,
                    _ => 0.,
                };
                assert_eq!(first, real(expected_first, precision));
                assert_eq!(second, real(expected_second, precision));
            }
        }
        assert!(
            derivatives(
                SpecialFunction::BesselJ,
                Some(&real(0.5, precision)),
                &zero,
                &zero,
                precision
            )
            .is_none()
        );
        assert!(
            derivatives(
                SpecialFunction::BesselY,
                Some(&zero),
                &zero,
                &zero,
                precision
            )
            .is_none()
        );
    }

    #[test]
    fn analytic_polylog_and_erf_zero_limits_are_finite() {
        let precision = 160;
        let zero = real(0., precision);
        for n in [-2, 0, 1, 2] {
            let order = real(f64::from(n), precision);
            let (first, second) = derivatives(
                SpecialFunction::Polylog,
                Some(&order),
                &zero,
                &zero,
                precision,
            )
            .unwrap();
            assert_eq!(first, real(1., precision));
            assert_eq!(second, real(2_f64.powi(1 - n), precision));
        }
        let (first, second) =
            derivatives(SpecialFunction::Erf, None, &zero, &zero, precision).unwrap();
        assert!((first.re.to_f64() - 2. / std::f64::consts::PI.sqrt()).abs() < 1e-15);
        assert!(first.im.is_fully_zero());
        assert!(second.is_fully_zero());
    }

    #[test]
    fn analytic_argument_derivatives_match_independent_symmetric_differences() {
        let precision = 192;
        let order = Complex::new(
            Float::with_val(precision, 0.5),
            Float::with_val(precision, 0.125),
        );
        let argument = Complex::new(
            Float::with_val(precision, 0.375),
            Float::with_val(precision, 0.125),
        );
        let step = Float::with_val(precision, 0.5).pow(36);
        let two = Float::with_val(precision, 2);
        for function in [
            SpecialFunction::Gamma,
            SpecialFunction::Erf,
            SpecialFunction::Polygamma(1),
            SpecialFunction::Polylog,
            SpecialFunction::BesselJ,
            SpecialFunction::BesselY,
            SpecialFunction::BesselI,
            SpecialFunction::BesselK,
        ] {
            let origin = function.value(Some(&order), &argument, precision).unwrap();
            let (first, second) =
                derivatives(function, Some(&order), &argument, &origin, precision).unwrap();
            let below = function
                .value(Some(&order), &(argument.clone() - &step), precision)
                .unwrap();
            let above = function
                .value(Some(&order), &(argument.clone() + &step), precision)
                .unwrap();
            let first_difference = (above.clone() - &below) / (step.clone() * &two);
            let second_difference = (above + below - origin * &two) / (step.clone() * &step);
            for (analytic, difference) in [(first, first_difference), (second, second_difference)] {
                for (analytic, difference) in
                    [(analytic.re, difference.re), (analytic.im, difference.im)]
                {
                    let scale = analytic.to_f64().abs().max(1.);
                    assert!((analytic - difference).to_f64().abs() < 1e-12 * scale);
                }
            }
        }
        assert!(
            derivatives(
                SpecialFunction::Zeta,
                None,
                &argument,
                &real(1., precision),
                precision
            )
            .is_none()
        );
        assert!(
            derivatives(
                SpecialFunction::Polygamma(u32::MAX),
                None,
                &argument,
                &real(1., precision),
                precision
            )
            .is_none()
        );
    }
}
