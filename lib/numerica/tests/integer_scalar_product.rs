use numerica::domains::{
    float::{F64, Float, RealLike, SingleFloat},
    integer::Integer,
};

fn assert_product_matches(left: &Float, right: &Float, coefficient: &Integer) {
    let expected = left.clone() * right;
    let actual = left.mul_integer_coefficient(right, coefficient);
    assert_eq!(
        actual,
        expected,
        "product differs for {left} ({} bits) * {right} ({} bits), coefficient {coefficient}",
        left.prec(),
        right.prec(),
    );
    assert_eq!(actual.prec(), expected.prec());
    if !actual.as_raw().is_nan() {
        assert_eq!(actual.is_sign_negative(), expected.is_sign_negative());
    }
}

#[test]
fn integer_products_preserve_value_precision_and_sign() {
    for left_precision in [2, 7, 53, 64, 192, 6312] {
        for right_precision in [2, 7, 53, 64, 192, 6312] {
            for coefficient in [
                i64::MIN,
                i64::MIN + 1,
                -(1 << 53) - 1,
                -129,
                -3,
                -1,
                1,
                3,
                129,
                (1 << 53) + 1,
                i64::MAX,
            ] {
                let left = Float::with_val(left_precision, coefficient);
                for right in ["0.1", "-0.1", "1.5", "-1.5", "1e-1000", "-1e1000"] {
                    let right = Float::parse(right, Some(right_precision)).unwrap();
                    assert_product_matches(&left, &right, &coefficient.into());
                }
            }
        }
    }
}

#[test]
fn coefficient_rounding_precedes_multiplication() {
    // 5 first rounds to 4 at two bits. Using the original exact coefficient
    // instead would round 5 * 0.75 to 4, instead of the required result of 3.
    let coefficient = Integer::from(5);
    let left = Float::with_val(192, 5);
    let right = Float::with_val(2, 0.75);
    let result = left.mul_integer_coefficient(&right, &coefficient);
    assert_eq!(result, Float::with_val(2, 3));
    assert_product_matches(&left, &right, &coefficient);

    // Coefficients can also have rounded before the multiplication is called.
    let left = Float::with_val(2, 5);
    let right = Float::with_val(192, 0.75);
    assert_product_matches(&left, &right, &coefficient);
}

#[test]
fn large_integer_coefficients_keep_existing_rounding() {
    for exponent in [80_u32, 200, 700] {
        for offset in [-1_i64, 0, 1] {
            for sign in [-1_i64, 1] {
                let coefficient = ((Integer::one() << exponent) + offset) * sign;
                for left_precision in [7, 64, 256] {
                    let left =
                        Float::new(left_precision).from_rational(&coefficient.clone().into());
                    for right_precision in [3, 53, 192] {
                        let right = Float::parse("-1.3456789", Some(right_precision)).unwrap();
                        assert_product_matches(&left, &right, &coefficient);
                    }
                }
            }
        }
    }
}

#[test]
fn inaccurate_hints_and_special_values_keep_existing_behavior() {
    for left_value in [
        0.0,
        -0.0,
        1.5,
        -1.5,
        3.0,
        -3.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ] {
        for right_value in [
            0.0,
            -0.0,
            1.25,
            -1.25,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ] {
            for hint in [-3, -1, 0, 1, 3, 5] {
                assert_product_matches(
                    &Float::with_val(192, left_value),
                    &Float::with_val(53, right_value),
                    &Integer::from(hint),
                );
            }
        }
    }
}

#[test]
fn fixed_precision_default_preserves_multiplication() {
    for left in [0.0, -0.0, 3.0, -3.0, 1.5, f64::INFINITY] {
        for right in [0.0, -0.0, 0.75, -1.25, f64::INFINITY] {
            let actual = F64(left).mul_integer_coefficient(&F64(right), &Integer::from(3));
            let expected = left * right;
            if expected.is_nan() {
                assert!(actual.0.is_nan());
            } else {
                assert_eq!(actual.0.to_bits(), expected.to_bits());
            }
        }
    }
}
