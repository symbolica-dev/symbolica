//! Range regressions from PR #128, using the stored magnitude as the authority.
//!
//! Errors outside binary64's range remain finite; only explicit unavailable
//! errors are NaN. The 32-bit error significand deliberately does not promise
//! binary64 relative accuracy. These cases do not exercise backend exponent limits.
use numerica::domains::{
    float::{ErrorPropagatingFloat as E, Float, FloatLike, Magnitude, Real, RealLike, SingleFloat},
    rational::Rational,
};

// Enough center precision to keep storage roundoff out of the tiny-error
// fixtures. These tests do not require arbitrary exact inputs to produce exact
// arithmetic results, or change the existing approximate propagation model.
const BITS: u32 = 2048;

fn number(text: &str) -> Float {
    Float::parse(text, Some(BITS)).unwrap()
}

fn p10(exponent: i32) -> Float {
    number(&format!("1e{exponent}"))
}

fn exact(value: Float) -> E<Float> {
    E::new_with_accuracy(value, f64::INFINITY)
}

fn near(actual: Magnitude, expected: Float) {
    assert!(actual.is_finite() && !actual.is_zero(), "{actual:?}");
    let ratio = (actual / expected.absolute_magnitude())
        .to_f64()
        .expect("comparable nonzero magnitudes");
    assert!(
        (ratio - 1.).abs() < 1e-7,
        "error {actual:?}, expected {expected:e}, ratio {ratio}"
    );
}

fn outside_f64(error: Magnitude) {
    assert!(error.is_finite() && !error.is_zero(), "{error:?}");
    assert_eq!(error.to_f64(), None);
}

#[test]
fn constructors_keep_tiny_and_huge_nonzero_centers_distinct_from_zero() {
    for (exponent, error_exponent) in [(-400, -410), (400, 390)] {
        let value = E::new(p10(exponent), 10.);
        assert!(!value.get_num().is_zero());
        assert!(value.get_num().is_finite());
        near(value.get_error_magnitude(), p10(error_exponent));
        outside_f64(value.get_error_magnitude());
        assert!((value.get_precision().unwrap() - 10.).abs() < 1e-7);
        assert!((value.get_accuracy() + f64::from(error_exponent)).abs() < 1e-7);
        // The legacy getter is deliberately lossy, but never invents exactness.
        assert!(value.get_absolute_error() > 0.);
        assert!(!value.get_absolute_error().is_nan());
    }
}

#[test]
fn floating_weight_preserves_error_below_f64_range_including_assignment() {
    let value = E::new(p10(-200), 10.);
    let weight = exact(p10(-200));
    let product = value.clone() * &weight;
    assert!(!product.get_num().is_zero());
    near(product.get_error_magnitude(), p10(-410));
    outside_f64(product.get_error_magnitude());
    assert_eq!(product.get_absolute_error(), f64::from_bits(1));
    let mut assigned = value;
    assigned *= &weight;
    assert_eq!(assigned.get_num(), product.get_num());
    assert_eq!(
        assigned.get_error_magnitude(),
        product.get_error_magnitude()
    );
}

#[test]
fn rational_weights_keep_tiny_errors_for_multiplication_and_division() {
    let value = E::new(p10(-200), 10.);
    let large = Rational::from(10).pow(200);
    for result in [value.clone() * large.clone().inv(), value / large] {
        assert!(!result.get_num().is_zero());
        near(result.get_error_magnitude(), p10(-410));
        outside_f64(result.get_error_magnitude());
    }
}

#[test]
fn wide_floating_and_rational_weights_preserve_compensated_errors() {
    let value = E::new_with_accuracy(Float::with_val(BITS, 1), 200.);
    let rational = Rational::from(10).pow(400);
    for result in [
        value.clone() * exact(p10(400)),
        value.clone() / exact(p10(-400)),
        value.clone() * rational.clone(),
        value / rational.inv(),
    ] {
        near(result.get_error_magnitude(), p10(200));
        assert!(result.get_absolute_error().is_finite());
    }
}

#[test]
fn reciprocal_and_quotient_keep_small_denominator_contributions() {
    let value = E::new_with_accuracy(p10(200), -100.);
    near(value.inv().get_error_magnitude(), p10(-300));

    let value = E::new(p10(300), 50.);
    let quotient = value.clone() / &value;
    assert_eq!(quotient.get_num().to_f64(), 1.);
    near(quotient.get_error_magnitude(), number("2e-50"));
    let mut assigned = value.clone();
    assigned /= value;
    assert_eq!(
        assigned.get_error_magnitude(),
        quotient.get_error_magnitude()
    );

    let tiny = E::new(1e-310_f64, 10.);
    let quotient = tiny / tiny;
    assert_eq!(*quotient.get_num(), 1.);
    near(quotient.get_error_magnitude(), number("2e-10"));
}

#[test]
fn representable_subnormal_errors_survive_without_decimal_projection() {
    let value = E::new_with_accuracy(Float::with_val(BITS, 1), 300.);
    let product = value * exact(p10(-20));
    near(product.get_error_magnitude(), p10(-320));
    assert!(product.get_error_magnitude().to_f64().unwrap() > 0.);
}

#[test]
fn subnormal_contributions_are_added_before_f64_conversion() {
    let minimum = Magnitude::ONE.scale_pow2(-1074);
    let value = E::new_with_error_magnitude(Float::with_val(BITS, 0.5), minimum);
    let product = value.clone() * value;
    near(
        product.get_error_magnitude(),
        Float::with_val(BITS, f64::from_bits(1)),
    );
    assert_eq!(
        product.get_error_magnitude().to_f64(),
        Some(f64::from_bits(1))
    );
}

#[test]
fn composite_decimal_accuracy_is_computed_without_underflowing_its_scale() {
    for (center, digits, expected) in [(200, 400., -200), (310, 10., 300), (308, 323., -15)] {
        near(
            E::new(p10(center), digits).get_error_magnitude(),
            p10(expected),
        );
    }
}

#[test]
fn finite_accuracy_outside_f64_range_is_not_unavailable_or_exact() {
    let value = E::new_with_accuracy(1_f64, 400.);
    near(value.get_error_magnitude(), p10(-400));
    outside_f64(value.get_error_magnitude());
    // Main's fixed-precision truncation floor is still allowed to dominate.
    let truncated = value.truncate();
    assert!(truncated.get_error_magnitude().is_finite());
    assert_eq!(truncated.get_absolute_error(), 1_f64.get_epsilon());

    let zero = E::new(Float::with_val(192, 0), 400.);
    near(zero.get_error_magnitude(), p10(-400));
    outside_f64(zero.get_error_magnitude());
}

#[test]
fn unavailable_errors_remain_nan_through_scaling_and_truncation() {
    let unknown = E::new_with_error_magnitude(Float::with_val(BITS, 1), Magnitude::NAN);
    for result in [
        unknown.clone() * exact(Float::with_val(BITS, 0)),
        unknown.clone() * Rational::from(0),
        unknown.clone() / Rational::from(2),
        unknown.truncate(),
    ] {
        assert!(result.get_error_magnitude().is_nan());
        assert!(result.get_absolute_error().is_nan());
    }
    let fixed = E::new_with_error_magnitude(1_f64, Magnitude::NAN).truncate();
    assert!(fixed.get_error_magnitude().is_nan());
}

#[test]
fn ordinary_signed_weights_preserve_the_absolute_error_scale() {
    for sign in [-1, 1] {
        let value = E::new_with_accuracy(Float::with_val(BITS, sign * 3), 6.);
        for result in [
            value.clone() * Rational::from(sign * 2),
            value.clone() * exact(Float::with_val(BITS, sign * 2)),
        ] {
            near(result.get_error_magnitude(), number("2e-6"));
        }
        for result in [
            value.clone() / Rational::from(sign * 2),
            value / exact(Float::with_val(BITS, sign * 2)),
        ] {
            near(result.get_error_magnitude(), number("5e-7"));
        }
    }
}

#[test]
fn inexact_f64_weight_projection_does_not_discard_a_tiny_native_error() {
    let minimum = Magnitude::ONE.scale_pow2(-1074);
    let value = E::new_with_error_magnitude(p10(-300), minimum);
    let rational = Rational::from((1, 2)) + Rational::from(2).pow(100).inv();
    let weight = rational.to_multi_prec_float(BITS);
    assert_eq!(weight.to_f64(), 0.5);
    let expected = weight.clone() * Rational::try_from(f64::from_bits(1)).unwrap();
    let product = value * exact(weight);
    near(product.get_error_magnitude(), expected);
    outside_f64(product.get_error_magnitude());
    assert_eq!(product.get_absolute_error(), f64::from_bits(1));
}

#[test]
fn subnormal_native_divisor_keeps_representable_quotient_error() {
    let value = E::new(Float::with_val(BITS, 1), 15.);
    let weight = number("7e-324");
    let expected = p10(-15) / &weight;
    let quotient = value / exact(weight);
    near(quotient.get_error_magnitude(), expected);
    assert!(quotient.get_error_magnitude().to_f64().unwrap().is_finite());
}

#[test]
fn internal_division_ratio_does_not_underflow_before_final_error() {
    let numerator = exact(p10(300));
    let denominator = E::new_with_accuracy(number("1.5e123"), 200.);
    let quotient = numerator / denominator;
    near(
        quotient.get_error_magnitude(),
        number("4.444444444444444444444444444444e-147"),
    );
}

#[test]
fn scalar_conversion_and_zero_center_products_preserve_nonzero_errors() {
    let tiny = E::<f64>::from(f64::from_bits(1));
    assert_eq!(tiny.get_error_magnitude(), Magnitude::ONE.scale_pow2(-1126));
    outside_f64(tiny.get_error_magnitude());
    assert_eq!(E::<f64>::from(0.).get_absolute_error(), f64::EPSILON);

    for accuracy in [2., 200.] {
        let zero = E::new_with_accuracy(Float::with_val(BITS, 0), accuracy);
        let product = zero.clone() * zero;
        assert!(product.get_num().is_zero());
        near(product.get_error_magnitude(), p10((-2. * accuracy) as i32));
    }
    let huge = E::new_with_accuracy(Float::with_val(BITS, 1), -200.) * exact(p10(200));
    near(huge.get_error_magnitude(), p10(400));
    outside_f64(huge.get_error_magnitude());
    assert!(huge.get_absolute_error().is_infinite());
}

#[test]
fn projected_divisor_does_not_hide_error_above_f64_maximum() {
    let value = E::new(Float::with_val(BITS, 0), 15.25);
    let weight = number("2.5e-324");
    let expected = Float::with_val(BITS, 10).powf(&Float::with_val(BITS, -15.25)) / &weight;
    let quotient = value / exact(weight);
    near(quotient.get_error_magnitude(), expected);
    outside_f64(quotient.get_error_magnitude());
    assert!(quotient.get_absolute_error().is_infinite());
}

#[test]
fn projected_subnormal_weight_preserves_a_normal_composite_error() {
    let value = E::new(p10(308), 8.);
    let product = value * exact(number("7e-324"));
    near(product.get_error_magnitude(), number("7e-24"));
}

#[test]
fn rational_constructors_bound_work_without_losing_compensated_errors() {
    for exponent in [1000_i32, -1000] {
        let power = Rational::from(10).pow(exponent.unsigned_abs() as u64);
        let value = if exponent < 0 { power.inv() } else { power };
        let result = E::new(value.clone(), f64::from(exponent));
        assert_eq!(result.get_num(), &value);
        near(result.get_error_magnitude(), Float::with_val(BITS, 1));
    }
    let tiny = Rational::from(10).pow(1000).inv();
    near(E::new(tiny.clone(), -1025.).get_error_magnitude(), p10(25));
    // These requests must saturate the fixed-size magnitude; they must not
    // materialize a rational power with a trillion decimal digits.
    assert!(
        E::new(tiny.clone(), -1e12)
            .get_error_magnitude()
            .is_infinite()
    );
    let very_small = E::new(tiny, 1e12).get_error_magnitude();
    assert!(very_small.is_finite() && !very_small.is_zero());
}

#[test]
fn nested_rational_constructors_do_not_materialize_huge_decimal_powers() {
    let center = E::new_with_accuracy(Rational::from((1, 3)), f64::INFINITY);
    let value = E::new(center.clone(), -1e12);
    assert!(value.get_error_magnitude().is_infinite());
    assert_eq!(value.get_num().get_num(), &Rational::from((1, 3)));
    assert!(value.get_num().get_error_magnitude().is_zero());
    let nested = E::new_with_accuracy(center, f64::INFINITY);
    let value = E::new(nested, -1e12);
    assert!(value.get_error_magnitude().is_infinite());
    assert_eq!(value.get_num().get_num().get_num(), &Rational::from((1, 3)));
}

#[test]
fn normal_projected_centers_do_not_hide_error_above_f64_maximum() {
    let weight = Float::with_val(BITS, f64::MAX) + Float::with_val(BITS, 2).pow(969);
    assert_eq!(weight.to_f64(), f64::MAX);
    let left = E::new_with_error_magnitude(weight.clone(), Magnitude::from_f64(0.75));
    let right_error = 0.25 + 2_f64.powi(-54);
    let right = E::new_with_error_magnitude(weight.clone(), Magnitude::from_f64(right_error));
    let expected = (Float::with_val(BITS, 0.75) + Float::with_val(BITS, right_error)) * &weight;
    let product = left * right;
    near(product.get_error_magnitude(), expected);
    outside_f64(product.get_error_magnitude());
}

#[test]
fn exact_projections_retain_sub_f64_product_and_quotient_errors() {
    let minimum = f64::from_bits(1);
    let left = E::new_with_error_magnitude(
        Float::with_val(BITS, f64::MIN_POSITIVE),
        Magnitude::ONE.scale_pow2(-1074),
    );
    for (center, exponent, divide) in [(0.5, -152, false), (2., -150, true)] {
        let right = E::new_with_error_magnitude(
            Float::with_val(BITS, center),
            Magnitude::ONE.scale_pow2(exponent),
        );
        let error = Float::with_val(BITS, 2).pow(exponent.unsigned_abs()).inv();
        let expected = if divide {
            Float::with_val(BITS, minimum) / right.get_num()
                + error / right.get_num() * (left.get_num().clone() / right.get_num())
        } else {
            Float::with_val(BITS, minimum) * right.get_num() + error * left.get_num()
        };
        let result = if divide {
            left.clone() / right
        } else {
            left.clone() * right
        };
        near(result.get_error_magnitude(), expected);
        outside_f64(result.get_error_magnitude());
        assert_eq!(result.get_absolute_error(), minimum);
    }
}

#[test]
fn explicit_exactness_is_distinct_from_finite_accuracy_requests() {
    let tiny = exact(p10(-200));
    for result in [
        tiny.clone() * exact(p10(400)),
        tiny.clone() / exact(p10(-400)),
        tiny * Rational::from(10).pow(400),
        E::new(p10(-200), 10.) * exact(Float::with_val(BITS, 0)),
    ] {
        assert!(result.get_num().is_finite());
        assert!(result.get_error_magnitude().is_zero());
        assert_eq!(result.get_absolute_error(), 0.);
    }
    let exact_one = exact(Float::with_val(BITS, 1));
    assert!(exact_one.get_error_magnitude().is_zero());
    let exact_zero = exact(Float::with_val(BITS, 0));
    assert!(
        (exact_one.clone() * exact_zero)
            .get_error_magnitude()
            .is_zero()
    );
    assert!(
        (exact_one.clone() / exact_one)
            .get_error_magnitude()
            .is_zero()
    );
    for digits in [400., 1e12] {
        assert!(
            !E::new_with_accuracy(Float::with_val(BITS, 1), digits)
                .get_error_magnitude()
                .is_zero()
        );
    }
}
