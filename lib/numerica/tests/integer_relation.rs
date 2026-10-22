#[cfg(feature = "float-mpfr")]
use numerica::domains::float::{FloatLike, RoundingDirection};
use numerica::domains::{
    float::{F64, Float, Real, RealLike, SingleFloat},
    integer::{Integer, IntegerRelationError},
};

#[test]
fn fixed_precision_relation_requires_multiple_iterations() {
    let mut values = [2.0_f64, 3.0, 5.0, 7.0]
        .map(|value| F64(value.sqrt()))
        .to_vec();
    let coefficients = [7.0, -11.0, 13.0, -17.0];
    let last = values
        .iter()
        .zip(coefficients)
        .map(|(value, coefficient)| value.0 * coefficient)
        .sum::<f64>();
    values.push(F64(-last));

    assert!(matches!(
        Integer::solve_integer_relation(&values, F64(1e-12), 1, None, None),
        Err(IntegerRelationError::IterationLimit(_))
    ));
    let relation =
        Integer::solve_integer_relation(&values, F64(1e-12), 1000, Some(Integer::from(100)), None)
            .unwrap();
    assert!(relation.iter().any(|coefficient| !coefficient.is_zero()));
    assert!(relation.iter().all(|coefficient| coefficient.abs() <= 100));
    let residual = values
        .iter()
        .zip(&relation)
        .map(|(value, coefficient)| value.0 * F64(0.0).from_rational(&coefficient.clone().into()).0)
        .sum::<f64>();
    assert!(residual.abs() < 1e-10, "residual: {residual}");
}

#[cfg(feature = "float-mpfr")]
fn planted_relation_values(precision: u32) -> Vec<Float> {
    const PRIMES: [i64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    const COEFFICIENTS: [i64; 12] = [17, -23, 5, -11, 29, -3, 13, -19, 7, -31, 2, -37];
    let mut values = PRIMES
        .iter()
        .enumerate()
        .map(|(index, prime)| {
            let root = Float::with_val(precision, *prime).sqrt();
            if index % 3 == 0 { -root } else { root }
        })
        .collect::<Vec<_>>();
    let mut last = Float::new(precision);
    for (value, coefficient) in values.iter().zip(COEFFICIENTS) {
        last -= value.clone() * Float::with_val(precision, coefficient);
    }
    values.push(last);
    values
}

#[test]
#[cfg(feature = "float-mpfr")]
fn higher_dimensional_relation_has_independent_high_precision_residual() {
    let values = planted_relation_values(384);
    let tolerance = Float::parse("1e-100", Some(384)).unwrap();
    let relation =
        Integer::solve_integer_relation(&values, tolerance, 5000, Some(Integer::from(100)), None)
            .unwrap();

    assert_eq!(relation.len(), values.len());
    assert!(relation.iter().any(|coefficient| !coefficient.is_zero()));
    assert!(relation.iter().all(|coefficient| coefficient.abs() <= 100));

    // Recompute the inputs at twice the solver's precision so the residual does
    // not merely confirm cancellation in the solver's transformed coordinates.
    let mut residual = Float::new(768);
    for (value, coefficient) in planted_relation_values(768).iter().zip(&relation) {
        residual += value.clone() * value.from_rational(&coefficient.clone().into());
    }
    assert!(
        residual.norm() < Float::parse("1e-200", Some(768)).unwrap(),
        "independently computed residual: {residual}"
    );
}

#[test]
fn coefficient_limit_stops_search_for_a_bounded_relation() {
    let values = [Float::with_val(256, 2).sqrt(), Float::with_val(256, 1)];
    assert_eq!(
        Integer::solve_integer_relation(
            &values,
            Float::parse("1e-60", Some(256)).unwrap(),
            100,
            Some(Integer::from(1)),
            None,
        ),
        Err(IntegerRelationError::CoefficientLimit)
    );
}

#[test]
#[cfg(feature = "float-mpfr")]
fn initial_reduction_distinguishes_quotients_on_both_sides_of_half() {
    let half = Float::with_val(192, 1) / Float::with_val(192, 2);
    // These perturbations disappear in f64. Hermite quotients must retain the
    // input precision when deciding which integer to use for the reduction.
    let epsilon = Float::with_val(192, 2).pow(80).inv();
    for sign in [-1, 1] {
        for (offset, rounded) in [(-epsilon.clone(), 0), (epsilon.clone(), 1)] {
            let value = (half.clone() + offset) * Float::with_val(192, sign);
            let result = Integer::solve_integer_relation(
                &[value, Float::with_val(192, 1)],
                Float::parse("1e-50", Some(192)).unwrap(),
                0,
                None,
                None,
            );
            assert_eq!(
                result,
                Err(IntegerRelationError::IterationLimit(vec![
                    Integer::from(1),
                    Integer::from(-sign * rounded),
                ])),
                "sign: {sign}, rounded magnitude: {rounded}"
            );
        }
    }
}

#[cfg(feature = "float-mpfr")]
fn assert_quotient_rounding_matches_full_precision(numerator: Float, denominator: Float) {
    let expected = (numerator.clone() / &denominator).round_to_nearest_integer();
    assert_eq!(
        numerator.div_round_to_nearest_integer(&denominator),
        expected,
        "quotient rounding differs for {numerator} ({} bits) / {denominator} ({} bits)",
        numerator.prec(),
        denominator.prec(),
    );
}

#[test]
#[cfg(feature = "float-mpfr")]
fn quotient_rounding_matches_full_precision_near_half_integers() {
    for precision in [54, 55, 192] {
        for integer_part in [0_u64, 1, 2, 17, 1 << 20] {
            // The perturbation is one representable step at this precision,
            // and may disappear when a quotient is first rounded to f64.
            let odd = 2 * integer_part + 1;
            let shift = precision - (64 - odd.leading_zeros());
            let center = Integer::from(odd) << shift;
            let divisor = Integer::one() << (shift + 1);
            for offset in [-1, 0, 1] {
                for numerator_sign in [-1, 1] {
                    for denominator_sign in [-1, 1] {
                        let numerator = Float::from_rational_round(
                            &(
                                (&center + Integer::from(offset)) * Integer::from(numerator_sign),
                                divisor.clone(),
                            )
                                .into(),
                            precision,
                            RoundingDirection::Nearest,
                        );
                        for denominator_precision in [54, 55, 192] {
                            let denominator =
                                Float::with_val(denominator_precision, denominator_sign);
                            assert_quotient_rounding_matches_full_precision(
                                numerator.clone(),
                                denominator,
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
#[cfg(feature = "float-mpfr")]
fn quotient_rounding_matches_full_precision_outside_the_fast_range() {
    for precision in [54, 55, 192] {
        for exponent in [52_u32, 53, 130] {
            for offset in [-3, -1, 0, 1, 3] {
                for sign in [-1, 1] {
                    // Integer and half-integer quotients on both sides of the
                    // range where f64 can distinguish adjacent integers.
                    let numerator = Float::from_rational_round(
                        &(
                            ((Integer::one() << (exponent + 1)) + Integer::from(offset))
                                * Integer::from(sign),
                            Integer::from(2),
                        )
                            .into(),
                        precision,
                        RoundingDirection::Nearest,
                    );
                    assert_quotient_rounding_matches_full_precision(
                        numerator,
                        Float::with_val(precision, 1),
                    );
                }
            }
        }
    }

    for (numerator, denominator) in [
        ("1e-1000", "1"),
        ("-1e-1000", "1"),
        ("1", "1e1000"),
        ("1", "-1e1000"),
        ("1e1000", "1"),
        ("-1e1000", "1"),
        ("0", "3"),
    ] {
        assert_quotient_rounding_matches_full_precision(
            Float::parse(numerator, Some(192)).unwrap(),
            Float::parse(denominator, Some(192)).unwrap(),
        );
    }
}

#[test]
#[cfg(feature = "float-mpfr")]
fn quotient_rounding_matches_full_precision_for_mixed_operand_precisions() {
    // Division first lowers the numerator to the smaller operand precision.
    // At two bits, 5/8 first rounds to 1/2 and then to the even integer zero.
    // Rounding the full-precision quotient directly would instead give one.
    assert_quotient_rounding_matches_full_precision(
        Float::parse("0.625", Some(192)).unwrap(),
        Float::with_val(2, 1),
    );

    let mut state = 0x5eed_f00d_1234_5678_u64;
    let mut next = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        state
    };
    for (numerator_precision, denominator_precision) in [
        (1, 1),
        (2, 3),
        (3, 2),
        (24, 53),
        (53, 192),
        (54, 54),
        (55, 192),
        (192, 2),
        (192, 53),
        (192, 55),
        (192, 192),
        (256, 384),
    ] {
        for _ in 0..64 {
            let numerator = (next() % 2_000_000_001) as i64 - 1_000_000_000;
            let denominator = (next() % 100_003 + 1) as i64;
            let denominator = if next() & 1 == 0 {
                denominator
            } else {
                -denominator
            };
            assert_quotient_rounding_matches_full_precision(
                Float::from_rational_round(
                    &(Integer::from(numerator), Integer::from(17)).into(),
                    numerator_precision,
                    RoundingDirection::Nearest,
                ),
                Float::from_rational_round(
                    &(Integer::from(denominator), Integer::from(19)).into(),
                    denominator_precision,
                    RoundingDirection::Nearest,
                ),
            );
        }
    }
}

#[test]
fn fixed_precision_quotient_rounding_matches_division_followed_by_rounding() {
    for numerator in [-7.5, -2.5, -0.5, 0.0, 0.5, 2.5, 7.5, 1e30] {
        for denominator in [-3.0, -1.0, 1.0, 3.0] {
            let numerator = F64(numerator);
            let denominator = F64(denominator);
            assert_eq!(
                numerator.div_round_to_nearest_integer(&denominator),
                (numerator / denominator).round_to_nearest_integer(),
            );
        }
    }
}
