use std::collections::HashMap;

use crate::{
    atom::{Atom, AtomCore, Symbol},
    domains::float::{Complex, ErrorQuality, Float, RealBall, TrackedComplex},
    function, parse,
};

#[test]
fn mapped_inputs_obey_working_precision_without_erasing_source_error() {
    let variable = parse!("tracked_review_input");
    let source = TrackedComplex::from_source_decimal("1.234567890123456789", 18, 512).unwrap();
    let source_center = source.ball.re.center.to_rational();
    let source_radius = source.ball.re.radius.to_rational();
    let inputs = HashMap::from([(variable.clone(), source)]);

    for bits in [32, 768] {
        let result = variable.evaluate_tracked(&inputs, bits).unwrap();
        assert_eq!(result.working_precision(), bits);
        assert_eq!(result.ball.re.center.prec(), bits);
        assert_eq!(result.error_quality, ErrorQuality::Estimated);
        assert!(result.precision().imaginary.exact_zero);
        let center_shift = (result.ball.re.center.to_rational() - &source_center).abs();
        assert!(result.ball.re.radius.to_rational() >= &source_radius + &center_shift);
        assert!(result.precision().real.significant_bits.unwrap() < 64);

        let squared = variable
            .pow(Atom::num(2))
            .evaluate_tracked(&inputs, bits)
            .unwrap();
        assert!(squared.ball.re.center.prec() <= bits);
        assert_eq!(squared.error_quality, ErrorQuality::Estimated);
    }
}

#[test]
fn conditional_selection_retains_estimated_condition_quality() {
    let condition = parse!("tracked_review_condition");
    let expression = function!(Symbol::IF, condition.clone(), 2, 3);
    let mut inputs = HashMap::from([(
        condition.clone(),
        TrackedComplex::from_source_decimal("1", 20, 128).unwrap(),
    )]);
    let result = expression.evaluate_tracked(&inputs, 128).unwrap();
    assert_eq!(result.ball.re.center, Float::with_val(128, 2));
    assert_eq!(result.error_quality, ErrorQuality::Estimated);

    inputs.insert(
        condition,
        TrackedComplex::from_source_decimal("0", 20, 128).unwrap(),
    );
    assert!(expression.evaluate_tracked(&inputs, 128).is_err());
}

#[test]
fn tracked_input_validation_rejects_invalid_enclosures_and_working_precision() {
    let variable = parse!("tracked_review_invalid");
    for (center, radius) in [
        (f64::NAN, 0.0),
        (f64::INFINITY, 0.0),
        (1.0, f64::NAN),
        (1.0, f64::INFINITY),
        (1.0, -1.0),
    ] {
        let invalid = TrackedComplex::from_ball(
            Complex::new(
                RealBall {
                    center: Float::with_val(128, center),
                    radius: Float::with_val(128, radius),
                },
                RealBall::exact(Float::new(128)),
            ),
            ErrorQuality::Bounded,
        );
        let inputs = HashMap::from([(variable.clone(), invalid)]);
        assert!(variable.evaluate_tracked(&inputs, 64).is_err());
    }

    let inputs = HashMap::<Atom, TrackedComplex>::new();
    for bits in [0, 1] {
        assert!(Atom::num(1).evaluate_tracked(&inputs, bits).is_err());
    }
}

#[cfg(all(feature = "float-mpfr", feature = "integer-gmp"))]
#[test]
fn finite_values_that_overflow_during_reprecision_are_rejected() {
    let center = Float::from_raw(rug::Float::with_val(4, 1.875) << (rug::float::exp_max() - 1));
    let input = TrackedComplex::from_ball(
        Complex::new(RealBall::exact(center), RealBall::exact(Float::new(4))),
        ErrorQuality::Bounded,
    );
    assert!(input.is_finite());
    // Rebuilding a finite input ball can also overflow its midpoint sum.
    let variable = parse!("tracked_review_exponent_limit");
    let inputs = HashMap::from([(variable.clone(), input)]);
    assert!(variable.evaluate_tracked(&inputs, 4).is_err());
}
