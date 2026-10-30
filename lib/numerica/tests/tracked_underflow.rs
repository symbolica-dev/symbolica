use numerica::domains::float::{Float, Magnitude, RealLike};

#[cfg(feature = "float-mpfr")]
#[test]
fn mpfr_nonzero_underflow_includes_the_absolute_rounding_floor() {
    use numerica::domains::float::ErrorPropagatingFloat;

    // No exponent-range mutation: construct exact operands around the current
    // minimum. Their product is 9/16 of the minimum positive MPFR value, so
    // round-to-nearest rounds it upward with an error of 7/16 of that value.
    let minimum_exponent = i64::from(rug::float::exp_min()) - 1;
    let exponent_sum = minimum_exponent - 2;
    let left_exponent = exponent_sum.div_euclid(2);
    let right_exponent = exponent_sum - left_exponent;
    let minimum = Magnitude::ONE.scale_pow2(minimum_exponent);
    let expected_floor = Magnitude::ONE.scale_pow2(minimum_exponent - 1);
    let actual_error = Magnitude::from_f64(0.4375).scale_pow2(minimum_exponent);

    for precision in [24, 53, 256] {
        let scaled = |mantissa: f64, exponent: i64| {
            Float::from_raw(rug::Float::with_val(precision, mantissa) << exponent as i32)
        };
        for sign in [-1., 1.] {
            let left = ErrorPropagatingFloat::new_with_error_magnitude(
                scaled(sign * 1.5, left_exponent),
                Magnitude::ZERO,
            );
            let right = ErrorPropagatingFloat::new_with_error_magnitude(
                scaled(1.5, right_exponent),
                Magnitude::ZERO,
            );
            let product = left.clone() * &right;
            assert_eq!(product.get_num().absolute_magnitude(), minimum);
            assert_eq!(product.get_num().minimum_roundoff(), expected_floor);
            assert!(product.get_error_magnitude() >= actual_error);
            assert_eq!(product.get_error_magnitude(), expected_floor);

            let mut assigned = left;
            assigned *= &right;
            assert_eq!(assigned.get_num(), product.get_num());
            assert_eq!(assigned.get_error_magnitude(), expected_floor);

            let exponent_difference = minimum_exponent - 1;
            let numerator_exponent = exponent_difference.div_euclid(2);
            let denominator_exponent = numerator_exponent - exponent_difference;
            let numerator = ErrorPropagatingFloat::new_with_error_magnitude(
                scaled(sign * 1.125, numerator_exponent),
                Magnitude::ZERO,
            );
            let denominator = ErrorPropagatingFloat::new_with_error_magnitude(
                scaled(1., denominator_exponent),
                Magnitude::ZERO,
            );
            let quotient = numerator.clone() / &denominator;
            assert_eq!(quotient.get_num().absolute_magnitude(), minimum);
            assert!(quotient.get_error_magnitude() >= actual_error);
            assert_eq!(quotient.get_error_magnitude(), expected_floor);

            let mut assigned = numerator;
            assigned /= &denominator;
            assert_eq!(assigned.get_num(), quotient.get_num());
            assert_eq!(assigned.get_error_magnitude(), expected_floor);
        }
    }
}

#[cfg(feature = "float-astro")]
#[test]
fn astro_rounding_floor_uses_word_aligned_subnormal_spacing() {
    let word_bits = astro_float::WORD_BIT_SIZE as i64;
    for precision in [1u32, 24, 64, 65, 256] {
        let storage_precision = (i64::from(precision) + word_bits - 1) / word_bits * word_bits;
        let expected =
            Magnitude::ONE.scale_pow2(i64::from(astro_float::EXPONENT_MIN) - storage_precision - 1);
        assert_eq!(Float::with_val(precision, 1).minimum_roundoff(), expected);
        assert!(!expected.is_zero());
    }
}
