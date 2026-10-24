#![cfg(any(feature = "float-mpfr", feature = "float-astro"))]

use numerica::domains::{
    float::{Float, FloatLike, Real},
    integer::{Integer, IntegerRelationError},
    rational::Rational,
};

const PRIMES: [i64; 13] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41];

fn assert_exact_residual(values: &[Float], relation: &[Integer], tolerance: &Float) {
    assert_eq!(relation.len(), values.len());
    assert!(relation.iter().any(|coefficient| !coefficient.is_zero()));
    let mut residual = Rational::zero();
    let mut squared_norm = Rational::zero();
    for (value, coefficient) in values.iter().zip(relation) {
        let value = value.to_rational();
        residual += &value * &Rational::from(coefficient);
        squared_norm += &value * &value;
    }
    let tolerance = tolerance.to_rational();
    assert!(
        &residual * &residual < &tolerance * &tolerance * squared_norm,
        "the returned relation fails the normalized tolerance on the exact input floats"
    );
}

fn planted_values(precision: u32, large_coefficients: bool) -> Vec<Float> {
    let mut values: Vec<_> = PRIMES[..12]
        .iter()
        .enumerate()
        .map(|(index, prime)| {
            let root = Float::with_val(precision, *prime).sqrt();
            if index % 3 == 0 { -root } else { root }
        })
        .collect();
    let coefficients: [i64; 12] = if large_coefficients {
        [
            1_238_774_393_845_211_063,
            -1_711_352_875_326_852_447,
            1_997_163_091_729_868_103,
            -1_414_379_920_046_395_571,
            1_087_745_297_588_081_179,
            -1_864_210_353_776_555_123,
            1_329_426_389_874_731_069,
            -1_572_923_514_285_092_211,
            1_149_734_688_248_768_337,
            -1_937_551_383_912_027_347,
            1_677_083_329_639_617_133,
            -1_294_597_912_274_743_667,
        ]
    } else {
        [17, -23, 5, -11, 29, -3, 13, -19, 7, -31, 2, -37]
    };
    let mut last = Float::new(precision);
    for (value, coefficient) in values.iter().zip(coefficients) {
        last -= value.clone() * Float::with_val(precision, coefficient);
    }
    if large_coefficients {
        // Keep the inputs similarly scaled while the primitive relation needs
        // 61-bit coefficients and repeated updates of the full-precision state.
        last /= Float::with_val(precision, 2).pow(61);
    }
    values.push(last);
    values
}

#[test]
fn high_precision_relation_has_exact_and_independently_recomputed_residuals() {
    let values = planted_values(1024, false);
    let tolerance = Float::parse("1e-250", Some(1024)).unwrap();
    let relation = Integer::solve_integer_relation(
        &values,
        tolerance.clone(),
        5000,
        Some(Integer::from(100)),
        None,
    )
    .unwrap();
    assert!(relation.iter().all(|coefficient| coefficient.abs() <= 100));
    assert_exact_residual(&values, &relation, &tolerance);
    assert_exact_residual(
        &planted_values(2048, false),
        &relation,
        &Float::parse("1e-550", Some(2048)).unwrap(),
    );
}

#[test]
fn large_coefficients_survive_repeated_precision_refreshes() {
    let values = planted_values(1024, true);
    let tolerance = Float::parse("1e-240", Some(1024)).unwrap();
    let limit = Integer::one() << 63_u32;
    let relation = Integer::solve_integer_relation(
        &values,
        tolerance.clone(),
        20000,
        Some(limit.clone()),
        None,
    )
    .unwrap();
    assert!(
        relation
            .iter()
            .all(|coefficient| coefficient.abs() <= limit)
    );
    assert!(
        relation
            .iter()
            .any(|coefficient| coefficient.significant_bits() >= 60)
    );
    assert_exact_residual(&values, &relation, &tolerance);
    assert_exact_residual(
        &planted_values(2048, true),
        &relation,
        &Float::parse("1e-550", Some(2048)).unwrap(),
    );
}

#[test]
fn values_below_native_float_range_keep_their_relation() {
    let precision = 2048;
    let mut values: Vec<_> = PRIMES[..11]
        .iter()
        .map(|prime| Float::with_val(precision, *prime).sqrt())
        .collect();
    let tiny = Float::with_val(precision, 37).sqrt() / Float::with_val(precision, 2).pow(1200);
    values.push(tiny.clone());
    values.push(-tiny * Float::with_val(precision, 3));
    let tolerance = Float::with_val(precision, 2).pow(1600).inv();
    let relation = Integer::solve_integer_relation(
        &values,
        tolerance.clone(),
        5000,
        Some(Integer::from(10)),
        None,
    )
    .unwrap();
    assert_exact_residual(&values, &relation, &tolerance);
}

#[test]
fn high_precision_search_respects_zero_and_one_iteration_budgets() {
    let values = planted_values(1024, true);
    for max_iter in [0, 1] {
        let error = Integer::solve_integer_relation(
            &values,
            Float::parse("1e-240", Some(1024)).unwrap(),
            max_iter,
            None,
            None,
        )
        .unwrap_err();
        let IntegerRelationError::IterationLimit(estimate) = error else {
            panic!("expected an iteration limit for budget {max_iter}, got {error:?}");
        };
        assert_eq!(estimate.len(), values.len());
        assert!(estimate.iter().any(|coefficient| !coefficient.is_zero()));
    }
}

#[test]
fn initial_relation_is_found_before_the_first_pivot() {
    let precision = 1024;
    let mut values: Vec<_> = PRIMES[..7]
        .iter()
        .map(|prime| Float::with_val(precision, *prime).sqrt())
        .collect();
    let last = values.last().unwrap().clone();
    *values.last_mut().unwrap() *= Float::with_val(precision, 2);
    values.push(last);
    let tolerance = Float::parse("1e-250", Some(precision)).unwrap();
    for max_iter in [0, 1] {
        let relation = Integer::solve_integer_relation(
            &values,
            tolerance.clone(),
            max_iter,
            Some(Integer::from(2)),
            None,
        )
        .unwrap();
        assert_exact_residual(&values, &relation, &tolerance);
        assert!(relation.iter().all(|c| c.abs() <= 2));
        assert!(
            Integer::solve_integer_relation(
                &values,
                tolerance.clone(),
                max_iter,
                Some(Integer::one()),
                None,
            )
            .is_err()
        );
    }
}

#[test]
// Astro's existing integer conversion truncates; even the full-precision
// solver reaches IterationLimit instead of CoefficientLimit for this input.
#[cfg(feature = "float-mpfr")]
fn high_precision_search_respects_coefficient_limit() {
    let values: Vec<_> = PRIMES
        .iter()
        .map(|prime| Float::with_val(1024, *prime).sqrt())
        .collect();
    assert_eq!(
        Integer::solve_integer_relation(
            &values,
            Float::parse("1e-240", Some(1024)).unwrap(),
            5000,
            Some(Integer::one()),
            None,
        ),
        Err(IntegerRelationError::CoefficientLimit)
    );
}

#[test]
fn relation_visible_only_at_low_precision_is_not_accepted() {
    // Leave room above the 512-bit dispatch threshold for the input sums'
    // tracked precision loss, so this case exercises the accelerated path.
    let precision = 640;
    let mut values: Vec<_> = PRIMES[..7]
        .iter()
        .map(|prime| Float::with_val(precision, *prime).sqrt())
        .collect();
    let mut last = Float::new(precision);
    for value in &values {
        last -= value;
    }
    // The apparent relation [1, ..., 1] vanishes at hardware precision, but
    // misses the requested tolerance by roughly two hundred binary digits.
    last += Float::with_val(precision, 2).pow(220).inv();
    values.push(last);
    let tolerance = Float::with_val(precision, 2).pow(420).inv();
    let result = Integer::solve_integer_relation(
        &values,
        tolerance.clone(),
        5000,
        Some(Integer::from(100)),
        None,
    );
    if let Ok(relation) = &result {
        assert_exact_residual(&values, relation, &tolerance);
    }
    assert!(
        result.is_err(),
        "accepted the deliberately perturbed relation"
    );
}
