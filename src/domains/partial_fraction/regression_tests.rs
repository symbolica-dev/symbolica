//! Regressions for localization semantics and univariate input validation.

use super::*;
use crate::{
    domains::{finite_field::Zp64, integer::Z, rational::Q},
    symbol,
};

fn variable() -> Arc<PolyVariable> {
    Arc::new(symbol!("pf_regression_x").into())
}

fn poly(x: &Arc<PolyVariable>, coefficients: &[i64]) -> UnivariatePolynomial<RationalField> {
    UnivariatePolynomial::from_coefficients(
        &Q,
        coefficients.iter().map(|&c| c.into()).collect(),
        x.clone(),
    )
}

fn composite_ring() -> FixedPartialFractionRing<RationalField> {
    let x = variable();
    PartialFractionRing::with_fixed_basis(Q, x.clone(), &[poly(&x, &[-1, 0, 1])]).unwrap()
}

fn rational(
    n: &UnivariatePolynomial<RationalField>,
    d: &UnivariatePolynomial<RationalField>,
) -> RationalPolynomial<IntegerRing, u16> {
    RationalPolynomial::from_num_den(
        n.clone().to_multivariate(),
        d.clone().to_multivariate(),
        &Z,
        true,
    )
}

#[test]
fn composite_basis_imports_divisors_and_preserves_canonical_equality() {
    let x = variable();
    let ring = composite_ring();
    let a = ring
        .try_from_fraction(&poly(&x, &[1]), &poly(&x, &[-1, 1]))
        .unwrap();
    let b = ring
        .try_from_fraction(&poly(&x, &[1, 1]), &poly(&x, &[-1, 0, 1]))
        .unwrap();
    assert_eq!(a, b);
    assert_eq!(ring.factor_count(), 1);
    assert_eq!(a.denominator_exponents().collect::<Vec<_>>(), vec![(0, 1)]);
    assert!(ring.is_one(&ring.mul(&a, &ring.element_from_polynomial(&poly(&x, &[-1, 1])))));
}

#[test]
fn composite_basis_exports_reduced_rational_functions() {
    let x = variable();
    let ring = composite_ring();
    let one = poly(&x, &[1]);
    let reduced_denominator = poly(&x, &[-1, 1]);
    let a = ring
        .try_from_fraction(&poly(&x, &[1, 1]), &poly(&x, &[-1, 0, 1]))
        .unwrap();
    assert_eq!(
        ring.to_fraction(&a),
        (one.clone(), reduced_denominator.clone())
    );
    let actual: RationalPolynomial<IntegerRing, u16> = ring.to_rational_polynomial(&a, &Z);
    assert_eq!(actual, rational(&one, &reduced_denominator));
    let actual: FactorizedRationalPolynomial<IntegerRing, u16> =
        ring.to_factorized_rational_polynomial(&a, &Z);
    let expected = FactorizedRationalPolynomial::from_num_den(
        one.to_multivariate(),
        vec![(reduced_denominator.to_multivariate(), 1)],
        &Z,
        true,
    );
    assert_eq!(actual, expected);
    assert_eq!(
        ring.try_from_factorized_rational_polynomial(&actual)
            .unwrap(),
        a
    );
}

#[test]
fn composite_basis_evaluation_distinguishes_removable_and_actual_poles() {
    let x = variable();
    let ring = composite_ring();
    let a = ring
        .try_from_fraction(&poly(&x, &[1, 1]), &poly(&x, &[-1, 0, 1]))
        .unwrap();
    assert_eq!(ring.evaluate(&a, &(-1).into()), Some((-1, 2).into()));
    assert_eq!(ring.evaluate(&a, &1.into()), None);
    assert_eq!(ring.evaluate(&a, &0.into()), Some((-1).into()));
}

#[test]
fn repeated_composite_basis_covers_unequal_divisor_multiplicities() {
    let x = variable();
    // f=(x-1)^2(x+1); a denominator (x-1)^3(x+1) requires f^2.
    let factor = poly(&x, &[1, -1, -1, 1]);
    let ring = PartialFractionRing::with_fixed_basis(Q, x.clone(), &[factor]).unwrap();
    let minus = poly(&x, &[-1, 1]);
    let plus = poly(&x, &[1, 1]);
    let denominator = &minus.pow(3) * &plus;
    let a = ring
        .try_from_fraction(&poly(&x, &[1]), &denominator)
        .unwrap();
    assert_eq!(a.denominator_exponents().collect::<Vec<_>>(), vec![(0, 2)]);
    assert_eq!(ring.to_fraction(&a), (poly(&x, &[1]), denominator.clone()));
    let cancelled = ring.try_from_fraction(&plus, &denominator).unwrap();
    assert_eq!(ring.to_fraction(&cancelled), (poly(&x, &[1]), minus.pow(3)));
    assert_eq!(
        ring.evaluate(&cancelled, &(-1).into()),
        Some((-1, 8).into())
    );
    assert_eq!(ring.evaluate(&cancelled, &1.into()), None);
    let expected = ring
        .try_from_fraction(&poly(&x, &[1]), &minus.pow(3))
        .unwrap();
    assert_eq!(cancelled, expected);
}

#[test]
fn fixed_basis_cancellation_zero_and_outside_denominators() {
    let x = variable();
    let ring = composite_ring();
    let zero = poly(&x, &[]);
    let outside = poly(&x, &[-2, 1]);
    assert!(ring.is_zero(&ring.try_from_fraction(&zero, &outside).unwrap()));
    assert!(ring.is_one(&ring.try_from_fraction(&outside, &outside).unwrap()));
    assert_eq!(
        ring.try_from_fraction(&poly(&x, &[1]), &outside),
        Err(PartialFractionError::OutsideBasis)
    );
    assert_eq!(
        ring.try_from_fraction(&zero, &zero),
        Err(PartialFractionError::ZeroDenominator)
    );
    assert_eq!(
        ring.try_from_fraction(&poly(&x, &[1]), &zero),
        Err(PartialFractionError::ZeroDenominator)
    );
    // Cancel an uncovered factor before covering a proper divisor of the basis.
    let denominator = &outside * &poly(&x, &[-1, 1]);
    let a = ring.try_from_fraction(&outside, &denominator).unwrap();
    assert_eq!(ring.to_fraction(&a), (poly(&x, &[1]), poly(&x, &[-1, 1])));
}

#[test]
fn fixed_polynomial_ring_divides_nonunits_when_the_quotient_exists() {
    let x = variable();
    let ring = PartialFractionRing::with_fixed_basis(Q, x.clone(), &[]).unwrap();
    let a = ring.element_from_polynomial(&poly(&x, &[0, 1]));
    assert!(ring.try_inv(&a).is_none());
    assert_eq!(ring.try_div(&a, &a), Some(ring.one()));
    assert_eq!(ring.try_div(&ring.zero(), &a), Some(ring.zero()));
    assert!(ring.try_div(&ring.one(), &a).is_none());
    assert!(ring.try_div(&a, &ring.zero()).is_none());
}

#[test]
fn mapped_irreducible_factor_cancels_after_splitting_modulo_a_prime() {
    let x = variable();
    let ring = PartialFractionRing::new(Q, x.clone(), &[poly(&x, &[1, 0, 1])]);
    let a = ring
        .try_from_fraction(&poly(&x, &[2, 1]), &poly(&x, &[1, 0, 1]))
        .unwrap();
    let field = Zp64::new(5);
    let mapped = ring.to_finite_field(&field).unwrap();
    let image = ring.element_to_finite_field(&mapped, &a).unwrap();
    let fp = |c: &[i64]| {
        UnivariatePolynomial::from_coefficients(
            &field,
            c.iter().map(|&c| field.nth(c.into())).collect(),
            x.clone(),
        )
    };
    assert_eq!(mapped.to_fraction(&image), (fp(&[1]), fp(&[-2, 1])));
    assert_eq!(
        mapped.evaluate(&image, &field.to_element(3)),
        Some(field.one())
    );
    assert_eq!(mapped.evaluate(&image, &field.to_element(2)), None);
    assert_eq!(
        mapped.try_from_fraction(&fp(&[1]), &fp(&[-2, 1])).unwrap(),
        image
    );
    let actual: RationalPolynomial<Zp64, u16> = mapped.to_rational_polynomial(&image, &field);
    let expected = RationalPolynomial::from_num_den(
        fp(&[1]).to_multivariate(),
        fp(&[-2, 1]).to_multivariate(),
        &field,
        true,
    );
    assert_eq!(actual, expected);
}

#[test]
fn fraction_import_checks_nonconstant_input_variables_but_accepts_constants() {
    let x = variable();
    let y = Arc::new(symbol!("pf_regression_y").into());
    let ring = PartialFractionRing::new(Q, x.clone(), &[]);
    assert_eq!(
        ring.try_from_fraction(&poly(&y, &[0, 1]), &poly(&x, &[0, 1])),
        Err(PartialFractionError::OtherVariable)
    );
    assert_eq!(
        ring.try_from_fraction(&poly(&x, &[1]), &poly(&y, &[0, 1])),
        Err(PartialFractionError::OtherVariable)
    );
    assert_eq!(
        ring.try_from_fraction(&poly(&y, &[2]), &poly(&y, &[2])),
        Ok(ring.one())
    );
    assert_eq!(ring.element_from_polynomial(&poly(&y, &[1])), ring.one());
}

#[test]
fn constructors_check_basis_variable_identity() {
    let x = variable();
    let y = Arc::new(symbol!("pf_regression_y").into());
    let wrong = poly(&y, &[0, 1]);
    assert_eq!(
        PartialFractionRing::with_fixed_basis(Q, x.clone(), &[wrong.clone()]).unwrap_err(),
        PartialFractionError::OtherVariable
    );
    assert!(std::panic::catch_unwind(|| PartialFractionRing::new(Q, x.clone(), &[wrong])).is_err());
}

#[test]
fn polynomial_entry_points_reject_nonconstant_wrong_variables() {
    let x = variable();
    let y = Arc::new(symbol!("pf_regression_y").into());
    let ring = composite_ring();
    let wrong = poly(&y, &[0, 1]);
    let one = ring.one();
    let rejects = |f: &dyn Fn()| {
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).is_err());
    };
    rejects(&|| {
        ring.element_from_polynomial(&wrong);
    });
    rejects(&|| {
        ring.mul_polynomial(&one, &wrong);
    });
    rejects(&|| {
        ring.mul_fraction(&one, &wrong, &[]);
    });
    rejects(&|| {
        ring.mul_fraction_cached(&one, &wrong, &[], &mut PowerCache::new());
    });
    // A constant's variable metadata does not change its value.
    assert_eq!(
        ring.mul_polynomial(&one, &poly(&y, &[2])),
        ring.element_from_polynomial(&poly(&x, &[2]))
    );
}
