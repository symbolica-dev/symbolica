use numerica::domains::{
    float::{Float, Real, RealBall, SingleFloat},
    integer::Integer,
    rational::Rational,
};

fn assert_sqrt_enclosure(ball: &RealBall, lower: &Rational, upper: &Rational) {
    let result = ball.sqrt();
    assert!(result.is_finite(), "{ball} => {result}");
    let lo = result.lower_bound().to_rational();
    let hi = result.upper_bound().to_rational();
    // Exact rational squares test both irrational root bounds without using
    // an approximate higher-precision square root as the reference.
    assert!(lo <= 0 || &lo * &lo <= *lower, "{lo}^2 exceeds {lower}");
    assert!(hi >= 0 && &hi * &hi >= *upper, "{hi}^2 is below {upper}");
    // Also enclose the represented interval, which may be wider than the
    // original rational constructor bounds.
    let represented_lower = ball.lower_bound().to_rational();
    let represented_upper = ball.upper_bound().to_rational();
    assert!(lo <= 0 || &lo * &lo <= represented_lower);
    assert!(hi >= 0 && &hi * &hi >= represented_upper);
}

#[test]
fn exact_irrational_root_has_a_certified_nonzero_radius() {
    for precision in [8, 24, 96, 256] {
        let input = RealBall::exact(Float::with_val(precision, 2));
        assert_sqrt_enclosure(&input, &2.into(), &2.into());
        assert!(!input.sqrt().radius.is_zero());
    }
}

#[test]
fn zero_and_binary_perfect_squares_remain_exact() {
    for (input, expected) in [(0, 0), (1, 1), (4, 2), (9, 3), (16, 4)] {
        let input = RealBall::exact(Float::with_val(64, input));
        let result = input.sqrt();
        assert_eq!(result.center.to_rational(), Rational::from(expected));
        assert!(result.radius.is_zero());
    }
}

#[test]
fn finite_positive_intervals_and_zero_endpoint_are_enclosed() {
    for (lower, upper) in [((0, 1), (7, 3)), ((1, 7), (13, 2)), ((13, 2), (99, 1))] {
        let lower: Rational = lower.into();
        let upper: Rational = upper.into();
        for precision in [8, 96] {
            let ball = RealBall::from_rational_bounds(&lower, &upper, precision);
            // Ball construction may widen [0,b] slightly below zero. Sqrt
            // correctly rejects that represented domain instead of clipping.
            if ball.lower_bound() < Float::with_val(precision, 0) {
                assert!(!ball.sqrt().is_finite());
            } else {
                assert_sqrt_enclosure(&ball, &lower, &upper);
            }
        }
    }
    let ball = RealBall::from_bounds(Float::with_val(64, 0), Float::with_val(64, 4));
    assert_sqrt_enclosure(&ball, &0.into(), &4.into());
}

#[test]
fn tiny_and_huge_exact_inputs_keep_scale_and_enclose_the_root() {
    let huge: Rational = Integer::from(2).pow(2048).into();
    let tiny = Rational::from((Integer::one(), Integer::from(2).pow(2048)));
    for scale in [tiny, huge] {
        let value = &scale * &Rational::from((7, 3));
        let ball = RealBall::from_rational_bounds(&value, &value, 96);
        assert_sqrt_enclosure(&ball, &value, &value);
        let root = ball.sqrt();
        assert!(root.radius < root.center / 100);
    }
}

#[test]
fn exact_float_sweep_encloses_both_rounding_directions() {
    for precision in [8, 32, 96] {
        for numerator in 1..80 {
            let value: Rational = (numerator, 17).into();
            let float = Float::new(precision).from_rational(&value);
            let exact = float.to_rational();
            assert_sqrt_enclosure(&RealBall::exact(float), &exact, &exact);
        }
    }
}

#[test]
fn negative_or_nonfinite_domains_report_invalid_instead_of_fabricating_bounds() {
    for (center, radius) in [
        (-1.0, 0.0),
        (0.0, 1.0),
        (f64::INFINITY, 0.0),
        (f64::NEG_INFINITY, 0.0),
        (f64::NAN, 0.0),
        (1.0, f64::INFINITY),
        (1.0, f64::NAN),
    ] {
        let input = RealBall::new(Float::with_val(96, center), Float::with_val(96, radius));
        assert!(!input.sqrt().is_finite());
    }
}
