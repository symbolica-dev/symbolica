use super::*;
#[cfg(feature = "float-mpfr")]
use crate::atom::Atom;

fn exact_real(value: i64, bits: u32) -> TrackedComplex {
    real(point(value, bits), ErrorQuality::Bounded)
}

#[cfg(feature = "float-mpfr")]
fn rational_real(numerator: i64, denominator: i64, bits: u32) -> TrackedComplex {
    let value = Rational::from((numerator, denominator));
    real(
        RealBall::from_rational_bounds(&value, &value, bits),
        ErrorQuality::Bounded,
    )
}

fn call(symbol: Symbol, arg: &TrackedComplex, bits: u32) -> Result<TrackedComplex, String> {
    evaluate_builtin_tracked(symbol, &[], &[arg.clone()], bits).unwrap()
}

#[test]
fn exact_sqrt_and_uncertain_zero_are_distinct() {
    let bits = 128;
    let sqrt = call(Symbol::SQRT, &exact_real(4, bits), bits).unwrap();
    assert!(sqrt.ball.re.contains(&Float::with_val(bits, 2)));
    assert!(real_zero(&sqrt.ball.im));
    let uncertain = real(
        RealBall::new(Float::new(bits), Float::with_val(bits, 1)),
        ErrorQuality::Bounded,
    );
    assert!(call(Symbol::SQRT, &uncertain, bits).is_err());
    let magnitude = call(Symbol::ABS, &uncertain, bits).unwrap();
    assert!(magnitude.ball.re.contains(&Float::new(bits)));
    assert!(magnitude.ball.re.contains(&Float::with_val(bits, 1)));
    assert!(!magnitude.ball.re.radius.is_fully_zero());
}

#[cfg(feature = "float-mpfr")]
#[test]
fn constants_and_real_functions_have_directed_bounds() {
    let bits = 180;
    for symbol in [Symbol::PI, Symbol::E, super::super::euler_gamma()] {
        let low = constant(symbol, bits).unwrap();
        let high = constant(symbol, bits * 2).unwrap();
        assert!(low.ball.contains_ball(&high.ball));
        assert_eq!(low.error_quality, ErrorQuality::Bounded);
        assert!(!low.ball.re.radius.is_fully_zero());
        assert!(real_zero(&low.ball.im));
    }
    for symbol in [
        Symbol::EXP,
        Symbol::LOG,
        Symbol::SIN,
        Symbol::COS,
        super::super::gamma(),
        super::super::zeta(),
    ] {
        let low = call(symbol, &exact_real(3, bits), bits).unwrap();
        let high = call(symbol, &exact_real(3, bits * 2), bits * 2).unwrap();
        assert!(low.ball.contains_ball(&high.ball), "{}", symbol);
        assert_eq!(low.error_quality, ErrorQuality::Bounded);
        assert!(real_zero(&low.ball.im));
    }
}

#[cfg(feature = "float-mpfr")]
#[test]
fn source_uncertainty_survives_higher_working_precision() {
    let evaluate = |bits| {
        let arg = real(
            RealBall::from_rational_ball(&Rational::from(2), &Rational::from((1, 1000)), bits),
            ErrorQuality::Estimated,
        );
        call(Symbol::LOG, &arg, bits).unwrap()
    };
    let low = evaluate(128);
    let high = evaluate(512);
    let floor = Rational::from((1, 3000)).to_multi_prec_float(512);
    assert!(low.ball.re.radius > floor);
    assert!(high.ball.re.radius > floor);
    assert_eq!(high.error_quality, ErrorQuality::Estimated);
    let left = directed_unary(
        Unary::Log,
        &Rational::from((1999, 1000)).to_multi_prec_float(512),
        512,
    )
    .unwrap();
    let right = directed_unary(
        Unary::Log,
        &Rational::from((2001, 1000)).to_multi_prec_float(512),
        512,
    )
    .unwrap();
    assert!(high.ball.re.intersects(&left));
    assert!(high.ball.re.intersects(&right));
}

#[cfg(feature = "float-mpfr")]
#[test]
fn cancellation_preserves_nonzero_error() {
    let bits = 128;
    let a = call(Symbol::EXP, &exact_real(1, bits), bits).unwrap();
    let b = constant(Symbol::E, bits).unwrap();
    let difference = a.ball - b.ball;
    assert!(difference.re.contains(&Float::new(bits)));
    assert!(!difference.re.radius.is_fully_zero());
    assert!(real_zero(&difference.im));
}

#[cfg(feature = "float-mpfr")]
#[test]
fn complex_log_power_and_branch_rejections() {
    let bits = 160;
    let i = TrackedComplex {
        ball: Complex::new(point(0, bits), point(1, bits)),
        error_quality: ErrorQuality::Bounded,
    };
    let log_i = call(Symbol::LOG, &i, bits).unwrap();
    let half_pi = constant(Symbol::PI, bits * 2).unwrap().ball.re / point(2, bits * 2);
    assert!(log_i.ball.im.contains_ball(&half_pi));
    assert!(log_i.ball.re.contains(&Float::new(bits)));
    let root = pow_tracked(&exact_real(-1, bits), &rational_real(1, 2, bits), bits).unwrap();
    assert!(
        root.ball
            .contains_complex(&Complex::new(Float::new(bits), Float::with_val(bits, 1)))
    );
    let crossing = TrackedComplex {
        ball: Complex::new(
            point(-1, bits),
            RealBall::new(Float::new(bits), Float::with_val(bits, 0.01)),
        ),
        error_quality: ErrorQuality::Bounded,
    };
    assert!(
        call(Symbol::LOG, &crossing, bits)
            .unwrap_err()
            .contains("branch cut")
    );
    assert!(pow_tracked(&crossing, &rational_real(1, 2, bits), bits).is_err());
    let nonfinite = real(
        RealBall::exact(Float::with_val(bits, f64::INFINITY)),
        ErrorQuality::Bounded,
    );
    assert!(call(Symbol::EXP, &nonfinite, bits).is_err());
}

#[cfg(feature = "float-mpfr")]
#[test]
fn gamma_and_polygamma_keep_input_sensitivity_and_reject_poles() {
    let bits = 160;
    let x = real(
        RealBall::from_rational_ball(&Rational::from(3), &Rational::from((1, 10000)), bits),
        ErrorQuality::Bounded,
    );
    let gamma = call(super::super::gamma(), &x, bits).unwrap();
    for endpoint in [x.ball.re.lower_bound(), x.ball.re.upper_bound()] {
        let reference = directed_unary(Unary::Gamma, &endpoint, bits * 2).unwrap();
        assert!(gamma.ball.re.contains_ball(&reference));
    }
    let tag = Atom::num(1);
    let psi = evaluate_builtin_tracked(super::super::polygamma(), &[tag.as_view()], &[x], bits)
        .unwrap()
        .unwrap();
    assert_eq!(psi.error_quality, ErrorQuality::Bounded);
    assert!(psi.ball.re.radius > Rational::from((1, 1_000_000)).to_multi_prec_float(bits));
    for pole in [0, -1, -4] {
        assert!(call(super::super::gamma(), &exact_real(pole, bits), bits).is_err());
        assert!(
            evaluate_builtin_tracked(
                super::super::polygamma(),
                &[tag.as_view()],
                &[exact_real(pole, bits)],
                bits
            )
            .unwrap()
            .is_err()
        );
    }
}

#[cfg(feature = "float-mpfr")]
#[test]
fn polylog_half_matches_independent_identity_and_preserves_source_radius() {
    let bits = 200;
    let tag = Atom::num(2);
    let half = rational_real(1, 2, bits);
    let value = evaluate_builtin_tracked(super::super::polylog(), &[tag.as_view()], &[half], bits)
        .unwrap()
        .unwrap();
    let high_bits = bits * 2;
    let pi = constant(Symbol::PI, high_bits).unwrap().ball.re;
    let log_two = directed_unary(Unary::Log, &Float::with_val(high_bits, 2), high_bits).unwrap();
    let identity = pi.pow(2) / point(12, high_bits) - log_two.pow(2) / point(2, high_bits);
    assert!(value.ball.re.contains_ball(&identity));
    assert_eq!(value.error_quality, ErrorQuality::Bounded);
    assert!(real_zero(&value.ball.im));
    let uncertain = real(
        RealBall::from_rational_ball(
            &Rational::from((1, 2)),
            &Rational::from((1, 1_000_000)),
            bits,
        ),
        ErrorQuality::Bounded,
    );
    let value = evaluate_builtin_tracked(
        super::super::polylog(),
        &[tag.as_view()],
        &[uncertain],
        bits,
    )
    .unwrap()
    .unwrap();
    assert!(value.ball.re.radius > Rational::from((1, 1_000_000)).to_multi_prec_float(bits));
    let outside = exact_real(2, bits);
    assert!(
        evaluate_builtin_tracked(super::super::polylog(), &[tag.as_view()], &[outside], bits)
            .unwrap()
            .is_err()
    );
}

#[cfg(feature = "float-mpfr")]
#[test]
fn zeta_integer_and_uncertain_real_arguments() {
    let bits = 180;
    let value = call(super::super::zeta(), &exact_real(3, bits), bits).unwrap();
    let reference = directed_unary(Unary::Zeta, &Float::with_val(bits * 2, 3), bits * 2).unwrap();
    assert!(value.ball.re.contains_ball(&reference));
    assert_eq!(value.error_quality, ErrorQuality::Bounded);
    let uncertain = real(
        RealBall::from_rational_ball(&Rational::from(3), &Rational::from((1, 1000)), bits),
        ErrorQuality::Bounded,
    );
    let value = call(super::super::zeta(), &uncertain, bits).unwrap();
    assert!(value.ball.re.radius > Rational::from((1, 100_000)).to_multi_prec_float(bits));
    assert!(call(super::super::zeta(), &exact_real(1, bits), bits).is_err());
}

#[cfg(feature = "float-mpfr")]
#[test]
fn polygamma_remainder_contains_independent_integer_values() {
    let bits = 160;
    let high = bits * 2;
    for (order, reference) in [
        (
            0,
            -constant(super::super::euler_gamma(), high).unwrap().ball.re,
        ),
        (
            1,
            constant(Symbol::PI, high).unwrap().ball.re.pow(2) / point(6, high),
        ),
        (
            2,
            directed_unary(Unary::Zeta, &Float::with_val(high, 3), high).unwrap() * point(-2, high),
        ),
    ] {
        let tag = Atom::num(order);
        let value = evaluate_builtin_tracked(
            super::super::polygamma(),
            &[tag.as_view()],
            &[exact_real(1, bits)],
            bits,
        )
        .unwrap()
        .unwrap();
        assert!(
            value.ball.re.contains_ball(&reference),
            "polygamma order {order}"
        );
        assert_eq!(value.error_quality, ErrorQuality::Bounded);
        assert!(real_zero(&value.ball.im));
    }
    let negative_half = call(super::super::gamma(), &rational_real(-1, 2, bits), bits).unwrap();
    let reference = constant(Symbol::PI, high).unwrap().ball.re.sqrt() * point(-2, high);
    assert!(negative_half.ball.re.contains_ball(&reference));
}

#[cfg(feature = "float-mpfr")]
#[test]
fn twenty_thousand_digit_constants_keep_arbitrary_precision_radii() {
    let bits = 66_500;
    let tiny = Float::with_val(bits, 0.5).pow(66_000);
    for value in [
        constant(Symbol::PI, bits).unwrap(),
        call(super::super::zeta(), &exact_real(3, bits), bits).unwrap(),
    ] {
        assert!(value.ball.re.radius > Float::new(bits));
        assert!(value.ball.re.radius < tiny);
        assert!(real_zero(&value.ball.im));
        assert_eq!(value.error_quality, ErrorQuality::Bounded);
    }
}

#[cfg(feature = "float-mpfr")]
#[test]
#[ignore = "slow acceptance: 20,000 decimal digit interval polylogarithm"]
fn twenty_thousand_digit_polylog_half_contains_independent_identity() {
    let bits = 66_439 + 128;
    let high_bits = bits + 128;
    let tag = Atom::num(2);
    let value = evaluate_builtin_tracked(
        super::super::polylog(),
        &[tag.as_view()],
        &[rational_real(1, 2, bits)],
        bits,
    )
    .unwrap()
    .unwrap();
    let pi = constant(Symbol::PI, high_bits).unwrap().ball.re;
    let log_two = directed_unary(Unary::Log, &Float::with_val(high_bits, 2), high_bits).unwrap();
    let identity = pi.pow(2) / point(12, high_bits) - log_two.pow(2) / point(2, high_bits);
    assert!(value.ball.re.contains_ball(&identity));
    assert!(real_zero(&value.ball.im));
    assert_eq!(value.error_quality, ErrorQuality::Bounded);
    // At least 20,000 decimal digits survive the accumulated arithmetic.
    let unit = Float::with_val(bits, 0.5).pow(66_439);
    assert!(value.ball.re.radius < unit);
}

#[cfg(feature = "float-mpfr")]
#[test]
fn complex_polygamma_encloses_uncertain_argument_corners() {
    let bits = 160;
    let radius = Rational::from((1, 1_000_000));
    let argument = TrackedComplex {
        ball: Complex::new(
            RealBall::from_rational_ball(&Rational::from((3, 2)), &radius, bits),
            RealBall::from_rational_ball(&Rational::from((1, 4)), &radius, bits),
        ),
        error_quality: ErrorQuality::Bounded,
    };
    let tag = Atom::num(2);
    let value = evaluate_builtin_tracked(
        super::super::polygamma(),
        &[tag.as_view()],
        &[argument.clone()],
        bits,
    )
    .unwrap()
    .unwrap();
    for re in [
        argument.ball.re.lower_bound(),
        argument.ball.re.upper_bound(),
    ] {
        for im in [
            argument.ball.im.lower_bound(),
            argument.ball.im.upper_bound(),
        ] {
            let reference =
                super::super::polygamma_numeric_eval(2, &Complex::new(re.clone(), im), bits * 2);
            assert!(value.ball.contains_complex(&reference));
        }
    }
    assert_eq!(value.error_quality, ErrorQuality::Bounded);
    for function in [super::super::gamma(), super::super::zeta()] {
        assert!(
            call(function, &argument, bits)
                .unwrap_err()
                .contains("does not expose an error bound")
        );
    }
}

#[cfg(feature = "float-mpfr")]
#[test]
fn fixed_point_polylog_half_bounds_cover_multiple_orders() {
    let bits = 160;
    for order in [2, 3, 4, 8] {
        let value = polylog_half(order, bits, ErrorQuality::Bounded);
        let reference = super::super::polylog_numeric_eval(
            &Complex::new(Float::with_val(bits * 2, order), Float::new(bits * 2)),
            &Complex::new(Float::with_val(bits * 2, 0.5), Float::new(bits * 2)),
            bits * 2,
        )
        .unwrap();
        assert!(
            value.ball.contains_complex(&reference),
            "polylog order {order}"
        );
        assert!(real_zero(&value.ball.im));
    }
}

#[cfg(not(feature = "float-mpfr"))]
#[test]
fn unsupported_portable_transcendental_bounds_return_errors() {
    let bits = 128;
    for symbol in [Symbol::PI, Symbol::E, super::super::euler_gamma()] {
        assert!(constant(symbol, bits).unwrap_err().contains("float-mpfr"));
    }
    for symbol in [
        Symbol::LOG,
        Symbol::EXP,
        super::super::gamma(),
        super::super::zeta(),
    ] {
        assert!(
            call(symbol, &exact_real(2, bits), bits)
                .unwrap_err()
                .contains("float-mpfr")
        );
    }
    let sqrt = call(Symbol::SQRT, &exact_real(4, bits), bits).unwrap();
    assert!(sqrt.ball.re.contains(&Float::with_val(bits, 2)));
    assert_eq!(sqrt.error_quality, ErrorQuality::Bounded);
    let half = polylog_half(2, bits, ErrorQuality::Bounded);
    assert!(half.ball.re.is_strictly_positive());
    assert!(real_zero(&half.ball.im));
}
