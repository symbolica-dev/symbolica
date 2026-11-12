//! Regression tests for denominator entry points and reuse of the basis.

use std::cell::RefCell;

use super::*;
use crate::{domains::rational::Q, symbol};

thread_local! {
    static FACTORIZATIONS: RefCell<Vec<Vec<Rational>>> = const { RefCell::new(Vec::new()) };
}

fn recorded_factorization(
    field: &RationalField,
    polynomial: &[Rational],
) -> Vec<(Vec<Rational>, usize)> {
    FACTORIZATIONS.with(|calls| calls.borrow_mut().push(polynomial.to_vec()));
    factorize(field, polynomial)
}

fn coefficients(c: &[i64]) -> Vec<Rational> {
    c.iter().map(|&c| c.into()).collect()
}

fn recording_ring(factors: &[&[i64]]) -> PartialFractionRing<RationalField> {
    FACTORIZATIONS.with(|calls| calls.borrow_mut().clear());
    PartialFractionRing::from_monic_factors(
        Q,
        Arc::new(symbol!("pf_entry_x").into()),
        factors.iter().map(|f| coefficients(f)).collect(),
        Some(recorded_factorization),
    )
    .unwrap()
}

fn factorization_inputs() -> Vec<Vec<Rational>> {
    FACTORIZATIONS.with(|calls| calls.borrow().clone())
}

#[test]
fn recombined_known_factors_do_not_invoke_factorization() {
    let ring = recording_ring(&[&[-1, 1], &[-2, 1]]);
    let first = coefficients(&[-1, 1]);
    let second = coefficients(&[-2, 1]);
    let known = dense::mul(&Q, &dense::pow(&Q, &first, 3), &dense::pow(&Q, &second, 2));
    assert_eq!(ring.denominator_factors(&known), Some(vec![(0, 3), (1, 2)]));
    assert!(factorization_inputs().is_empty());

    let unseen = dense::pow(&Q, &coefficients(&[-3, 1]), 2);
    let denominator = dense::mul(&Q, &known, &unseen);
    assert_eq!(
        ring.denominator_factors(&denominator),
        Some(vec![(0, 3), (1, 2), (2, 2)])
    );
    assert_eq!(factorization_inputs(), vec![unseen]);
    assert_eq!(ring.factor_count(), 3);

    // The whole-denominator cache is still used on repeated imports.
    assert_eq!(
        ring.denominator_factors(&denominator),
        Some(vec![(0, 3), (1, 2), (2, 2)])
    );
    assert_eq!(factorization_inputs().len(), 1);

    let strict = ring.strict();
    let unknown = dense::mul(&Q, &known, &coefficients(&[-4, 1]));
    assert_eq!(strict.denominator_factors(&unknown), None);
    assert_eq!(factorization_inputs().len(), 1);
    assert_eq!(ring.factor_count(), 3);
}

#[test]
fn factorized_import_keeps_factors_and_leading_coefficient_powers() {
    let ring = recording_ring(&[]);
    let first = ring.polynomial(coefficients(&[2, -2])); // -2 (x - 1)
    let second = ring.polynomial(coefficients(&[6, 3])); // 3 (x + 2)
    let numerator = ring.polynomial(coefficients(&[3, 1]));
    let input = FactorizedRationalPolynomial::<RationalField, u16> {
        numerator: numerator.clone().to_multivariate(),
        numer_coeff: 5.into(),
        denom_coeff: 7.into(),
        denominators: vec![
            (first.clone().to_multivariate(), 3),
            (second.clone().to_multivariate(), 2),
        ],
    };
    let actual = ring
        .try_from_factorized_rational_polynomial(&input)
        .unwrap();
    // Each supplied factor is resolved once; its powers are not expanded
    // before factorization, and the two factors are never multiplied together.
    assert_eq!(
        factorization_inputs(),
        vec![coefficients(&[-1, 1]), coefficients(&[2, 1])]
    );
    let denominator = dense::mul(
        &Q,
        &dense::pow(&Q, &first.coefficients, 3),
        &dense::pow(&Q, &second.coefficients, 2),
    );
    let numerator = ring.polynomial(
        numerator
            .coefficients
            .iter()
            .map(|c| Q.mul(c, &Rational::from((5, 7))))
            .collect(),
    );
    let expected = ring
        .try_from_fraction(&numerator, &ring.polynomial(denominator))
        .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(factorization_inputs().len(), 2);
}

#[test]
fn factorized_import_falls_back_for_fixed_basis_cancellation() {
    let seed = recording_ring(&[&[-1, 1]]);
    let ring = seed.strict();
    let known = ring.polynomial(coefficients(&[-2, 2])); // 2 (x - 1)
    let outside = ring.polynomial(coefficients(&[1, 1]));
    let input = FactorizedRationalPolynomial::<RationalField, u16> {
        numerator: outside.clone().to_multivariate(),
        numer_coeff: 4.into(),
        denom_coeff: 1.into(),
        denominators: vec![(known.to_multivariate(), 2), (outside.to_multivariate(), 1)],
    };
    let actual = ring
        .try_from_factorized_rational_polynomial(&input)
        .unwrap();
    assert_eq!(actual, ring.inv_factor_power(0, 2));
    assert!(factorization_inputs().is_empty());
    assert_eq!(ring.factor_count(), 1);
}

#[test]
fn factorized_zero_import_still_validates_denominators() {
    let ring = recording_ring(&[]);
    let zero = ring.polynomial(vec![]).to_multivariate::<u16>();
    let input = FactorizedRationalPolynomial {
        numerator: zero.clone(),
        numer_coeff: 1.into(),
        denom_coeff: 1.into(),
        denominators: vec![(zero, 1)],
    };
    assert_eq!(
        ring.try_from_factorized_rational_polynomial(&input),
        Err(PartialFractionError::ZeroDenominator)
    );
    let mut input = input;
    input.denominators.clear();
    input.denom_coeff = 0.into();
    assert_eq!(
        ring.try_from_factorized_rational_polynomial(&input),
        Err(PartialFractionError::ZeroDenominator)
    );
    assert!(factorization_inputs().is_empty());
}
