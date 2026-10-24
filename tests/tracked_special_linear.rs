//! The local error estimate should approach the analytic Jacobian as the input
//! uncertainty shrinks. These tests deliberately keep output roundoff below
//! the propagated error; they do not assert certified interval enclosures.

use std::collections::HashMap;

use symbolica::{
    atom::{Atom, AtomCore},
    domains::float::{Complex, ErrorPropagatingFloat, Float, Magnitude, RealLike, SingleFloat},
    parse,
};

type Tracked = ErrorPropagatingFloat<Float>;

const PRECISION: u32 = 256;

// These independent identities also exercise the signs in the Bessel
// recurrences. The polylog argument stays inside its defining disk.
const DERIVATIVES: &[(&str, &str, f64)] = &[
    ("erf(x)", "2*exp(-x^2)/sqrt(pi)", 1.25),
    ("gamma(x)", "gamma(x)*polygamma(0,x)", 1.25),
    ("polygamma(2,x)", "polygamma(3,x)", 1.25),
    ("polylog(2,x)", "-log(1-x)/x", 0.375),
    ("bessel_j(1,x)", "(bessel_j(0,x)-bessel_j(2,x))/2", 1.25),
    ("bessel_y(1,x)", "(bessel_y(0,x)-bessel_y(2,x))/2", 1.25),
    ("bessel_i(1,x)", "(bessel_i(0,x)+bessel_i(2,x))/2", 1.25),
    ("bessel_k(1,x)", "-(bessel_k(0,x)+bessel_k(2,x))/2", 1.25),
];

fn input(re: f64, im: f64, re_error: Magnitude, im_error: Magnitude) -> Complex<Tracked> {
    Complex::new(
        Tracked::new_with_error_magnitude(Float::with_val(PRECISION, re), re_error),
        Tracked::new_with_error_magnitude(Float::with_val(PRECISION, im), im_error),
    )
}

fn evaluate(expression: &Atom, value: Complex<Tracked>) -> Complex<Tracked> {
    expression
        .evaluate_with_prec(&HashMap::from([(parse!("x"), value)]), PRECISION)
        .unwrap()
}

fn derivative(source: &str, re: f64, im: f64) -> Complex<Float> {
    parse!(source)
        .evaluate_with_prec(
            &HashMap::from([(
                parse!("x"),
                Complex::new(
                    Float::with_val(PRECISION + 64, re),
                    Float::with_val(PRECISION + 64, im),
                ),
            )]),
            PRECISION + 64,
        )
        .unwrap()
}

fn close_error(actual: Magnitude, expected: Magnitude, context: &str) {
    assert!(actual.is_finite(), "{context}: nonfinite error {actual:?}");
    assert!(!actual.is_zero(), "{context}: input error was discarded");
    assert!(expected.is_finite() && !expected.is_zero());
    // Magnitudes have 32 significant bits. Avoid narrowing either operand to
    // f64 before taking the ratio: some cases intentionally lie far below it.
    let ratio = actual.log2_ratio(expected).exp2();
    assert!(
        (ratio - 1.).abs() <= 2e-6,
        "{context}: actual {actual:?}, Jacobian {expected:?}, ratio {ratio}"
    );
}

fn jacobian_errors(
    derivative: &Complex<Float>,
    re_error: Magnitude,
    im_error: Magnitude,
) -> [Magnitude; 2] {
    let real = derivative.re.absolute_magnitude();
    let imaginary = derivative.im.absolute_magnitude();
    [
        real * re_error + imaginary * im_error,
        imaginary * re_error + real * im_error,
    ]
}

#[test]
fn real_axis_errors_scale_with_analytic_derivatives() {
    for &(source, derivative_source, center) in DERIVATIVES {
        let expression = parse!(source);
        let slope = derivative(derivative_source, center, 0.);
        for exponent in [-35, -65] {
            let error = Magnitude::ONE.scale_pow2(exponent);
            let result = evaluate(&expression, input(center, 0., error, Magnitude::ZERO));
            assert!(result.re.get_num().is_finite(), "{source}");
            assert!(result.im.is_exact_zero(), "{source}: lost realness");
            close_error(
                result.re.get_error_magnitude(),
                slope.re.absolute_magnitude() * error,
                &format!("{source}, real axis, delta=2^{exponent}"),
            );
        }
    }
}

#[test]
fn complex_errors_follow_componentwise_cauchy_riemann_weights() {
    let re_error = Magnitude::ONE.scale_pow2(-35);
    let im_error = Magnitude::from_f64(3.).scale_pow2(-40);
    for &(source, derivative_source, center) in DERIVATIVES {
        let slope = derivative(derivative_source, center, 0.25);
        let expected = jacobian_errors(&slope, re_error, im_error);
        let result = evaluate(&parse!(source), input(center, 0.25, re_error, im_error));
        close_error(
            result.re.get_error_magnitude(),
            expected[0],
            &format!("Re {source}, complex argument"),
        );
        close_error(
            result.im.get_error_magnitude(),
            expected[1],
            &format!("Im {source}, complex argument"),
        );
    }
}

#[test]
fn unequal_axis_errors_are_weighted_independently() {
    for &(source, derivative_source, center) in &DERIVATIVES[..4] {
        let slope = derivative(derivative_source, center, 0.25);
        for (re_exponent, im_exponent) in [(-40, -180), (-180, -40)] {
            let re_error = Magnitude::ONE.scale_pow2(re_exponent);
            let im_error = Magnitude::ONE.scale_pow2(im_exponent);
            let expected = jacobian_errors(&slope, re_error, im_error);
            let result = evaluate(&parse!(source), input(center, 0.25, re_error, im_error));
            close_error(
                result.re.get_error_magnitude(),
                expected[0],
                &format!("Re {source}, deltas 2^{re_exponent},2^{im_exponent}"),
            );
            close_error(
                result.im.get_error_magnitude(),
                expected[1],
                &format!("Im {source}, deltas 2^{re_exponent},2^{im_exponent}"),
            );
        }
    }
}

#[test]
fn transverse_real_axis_uncertainty_has_no_spurious_linear_real_term() {
    let re_error = Magnitude::ONE.scale_pow2(-180);
    let im_error = Magnitude::ONE.scale_pow2(-50);
    for &(source, derivative_source, center) in &DERIVATIVES[..4] {
        let slope = derivative(derivative_source, center, 0.).re;
        let result = evaluate(&parse!(source), input(center, 0., re_error, im_error));
        assert!(result.im.get_num().is_zero(), "{source}");
        close_error(
            result.im.get_error_magnitude(),
            slope.absolute_magnitude() * im_error,
            &format!("Im {source}, transverse uncertainty"),
        );
        // At these fixed regular centers, a generous quadratic allowance
        // permits curvature while excluding O(probe_step * delta_im) leakage.
        let quadratic_allowance = (im_error * im_error).scale_pow2(6)
            + (slope.absolute_magnitude() * re_error).scale_pow2(1);
        assert!(
            result.re.get_error_magnitude() <= quadratic_allowance,
            "{source}: real error {:?} exceeds quadratic allowance {quadratic_allowance:?}",
            result.re.get_error_magnitude()
        );
    }
}

#[test]
fn zero_center_erf_preserves_linear_errors_below_binary64_range() {
    let expression = parse!("erf(x)");
    let slope = derivative("2/sqrt(pi)", 0., 0.).re.absolute_magnitude();
    for exponent in [-40, -80, -4000] {
        let error = Magnitude::ONE.scale_pow2(exponent);
        if exponent == -4000 {
            assert!(error.to_f64().is_none());
        }
        for imaginary in [false, true] {
            let errors = if imaginary {
                [Magnitude::ZERO, error]
            } else {
                [error, Magnitude::ZERO]
            };
            let result = evaluate(&expression, input(0., 0., errors[0], errors[1]));
            assert!(result.re.get_num().is_zero() && result.im.get_num().is_zero());
            let (active, inactive) = if imaginary {
                (&result.im, &result.re)
            } else {
                (&result.re, &result.im)
            };
            close_error(
                active.get_error_magnitude(),
                slope * error,
                &format!("erf(0), imaginary={imaginary}, delta=2^{exponent}"),
            );
            assert!(inactive.is_exact_zero(), "erf(0): uncertainty changed axes");
        }
    }
}

#[test]
fn integer_order_bessel_derivatives_at_zero_are_finite() {
    let error = Magnitude::ONE.scale_pow2(-40);
    let expected = error.scale_pow2(-1);
    for name in ["bessel_j", "bessel_i"] {
        for order in [-1, 1] {
            let expression = parse!(format!("{name}({order},x)"));
            for imaginary in [false, true] {
                let errors = if imaginary {
                    [Magnitude::ZERO, error]
                } else {
                    [error, Magnitude::ZERO]
                };
                let result = evaluate(&expression, input(0., 0., errors[0], errors[1]));
                assert!(result.re.get_num().is_zero() && result.im.get_num().is_zero());
                let active = if imaginary { &result.im } else { &result.re };
                close_error(
                    active.get_error_magnitude(),
                    expected,
                    &format!("{name}({order},0), imaginary={imaginary}"),
                );
                if !imaginary {
                    assert!(result.im.is_exact_zero(), "{name}({order},0)");
                }
            }
        }
    }
}

#[test]
fn polylog_derivative_at_zero_uses_its_removable_limit() {
    let error = Magnitude::ONE.scale_pow2(-40);
    for order in [2, 3] {
        let expression = parse!(format!("polylog({order},x)"));
        let result = evaluate(&expression, input(0., 0., error, Magnitude::ZERO));
        assert!(result.re.get_num().is_zero());
        assert!(result.im.is_exact_zero());
        close_error(
            result.re.get_error_magnitude(),
            error,
            &format!("polylog({order},0): derivative is one"),
        );
    }
}

#[test]
fn stationary_integer_bessel_values_keep_finite_nonzero_error() {
    let error = Magnitude::ONE.scale_pow2(-40);
    for name in ["bessel_j", "bessel_i"] {
        for order in [-2, 0, 2] {
            let expression = parse!(format!("{name}({order},x)"));
            let result = evaluate(&expression, input(0., 0., error, Magnitude::ZERO));
            assert!(result.re.get_num().is_finite(), "{name}({order},0)");
            assert!(
                result.re.get_error_magnitude().is_finite() && !result.re.is_exact(),
                "{name}({order},0): stationary uncertainty was lost or unbounded"
            );
            assert!(result.im.is_exact_zero(), "{name}({order},0)");
        }
    }
}
