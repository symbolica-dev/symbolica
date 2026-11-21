//! Regressions for basis extension and univariate input validation.

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

fn composite_ring() -> PartialFractionRing<RationalField> {
    let x = variable();
    PartialFractionRing::with_basis(Q, x.clone(), &[poly(&x, &[-1, 0, 1])]).unwrap()
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
    let ring = PartialFractionRing::with_basis(Q, x.clone(), &[factor]).unwrap();
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
fn seeded_basis_cancellation_zero_and_new_denominators() {
    let x = variable();
    let ring = composite_ring();
    let zero = poly(&x, &[]);
    let outside = poly(&x, &[-2, 1]);
    assert!(ring.is_zero(&ring.try_from_fraction(&zero, &outside).unwrap()));
    assert!(ring.is_one(&ring.try_from_fraction(&outside, &outside).unwrap()));
    assert_eq!(ring.factor_count(), 1);
    let inverse = ring.try_from_fraction(&poly(&x, &[1]), &outside).unwrap();
    assert_eq!(ring.factor_count(), 2);
    assert_eq!(
        ring.to_fraction(&inverse),
        (poly(&x, &[1]), outside.clone())
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
fn empty_seeded_basis_inverts_nonzero_polynomials() {
    let x = variable();
    let ring = PartialFractionRing::with_basis(Q, x.clone(), &[]).unwrap();
    let a = ring.element_from_polynomial(&poly(&x, &[0, 1]));
    let inverse = ring.inv(&a);
    assert_eq!(ring.factor_count(), 1);
    assert_eq!(ring.mul(&a, &inverse), ring.one());
    assert_eq!(ring.try_inv(&a), Some(inverse.clone()));
    assert_eq!(ring.try_div(&a, &a), Some(ring.one()));
    assert_eq!(ring.try_div(&ring.zero(), &a), Some(ring.zero()));
    assert_eq!(ring.try_div(&ring.one(), &a), Some(inverse));
    assert!(ring.try_inv(&ring.zero()).is_none());
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
        PartialFractionRing::with_basis(Q, x.clone(), &[wrong.clone()]).unwrap_err(),
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

#[test]
fn fused_add_sub_products_preserve_cancellation_and_composite_bases() {
    let x = variable();
    let ring =
        PartialFractionRing::with_basis(Q, x.clone(), &[poly(&x, &[-1, 0, 1]), poly(&x, &[2, 1])])
            .unwrap();
    let values = vec![
        ring.zero(),
        ring.one(),
        ring.nth((-1).into()),
        ring.element_from_polynomial(&poly(&x, &[1, 0, 1])),
        ring.inv_factor_power(0, 2),
        ring.try_from_fraction(&poly(&x, &[1, 1]), &poly(&x, &[-1, 0, 1]))
            .unwrap(),
        ring.inv_factor_power(1, 4),
    ];
    for a in &values {
        for b in &values {
            let ra: RationalPolynomial<_, u16> = ring.to_rational_polynomial(a, &Z);
            let rb: RationalPolynomial<_, u16> = ring.to_rational_polynomial(b, &Z);
            let expected = &ra * &rb;
            let mut actual = ring.zero();
            ring.add_mul_assign(&mut actual, a, b);
            let flat: RationalPolynomial<_, u16> = ring.to_rational_polynomial(&actual, &Z);
            assert_eq!(flat, expected);
            ring.sub_mul_assign(&mut actual, a, b);
            assert!(ring.is_zero(&actual));
            ring.sub_mul_assign(&mut actual, a, b);
            let flat: RationalPolynomial<_, u16> = ring.to_rational_polynomial(&actual, &Z);
            assert_eq!(flat, -expected);
            ring.add_mul_assign(&mut actual, a, b);
            assert!(ring.is_zero(&actual));
        }
    }
}

#[test]
fn nonlinear_division_recurrence_handles_high_poles_and_unequal_degrees() {
    fn check<F: Field>(field: F) {
        let x = variable();
        let polynomial = |c: &[i64]| {
            UnivariatePolynomial::from_coefficients(
                &field,
                c.iter().map(|&c| field.nth(c.into())).collect(),
                x.clone(),
            )
        };
        let ring = PartialFractionRing::with_basis(
            field.clone(),
            x.clone(),
            &[
                polynomial(&[-1, 1]),
                polynomial(&[1, 0, 1]),
                polynomial(&[1, 1, 0, 1]),
            ],
        )
        .unwrap();
        for h in 0..3 {
            for g in 0..3 {
                if h == g {
                    continue;
                }
                for exponent in [1, 4, 16] {
                    let coefficients = (0..ring.factor(h).degree() * exponent)
                        .map(|i| field.nth(((i as i64 % 5) - 2).into()))
                        .collect();
                    let a = ring
                        .element_from_coefficients(vec![field.one()], vec![(h, coefficients)])
                        .unwrap();
                    let divided = ring.div_factor_power(&a, g, 3);
                    // Polynomial multiplication is independent of the CRT
                    // division recurrence and must recover every coefficient.
                    assert_eq!(ring.mul_polynomial(&divided, &ring.factor(g).pow(3)), a);
                }
            }
        }
    }
    check(Q);
    check(Zp64::new(101));
}

#[test]
fn composite_basis_extends_for_a_new_factor_without_splitting_existing_blocks() {
    let x = variable();
    let ring = composite_ring();
    let original = ring.factor(0);
    let known_divisor = poly(&x, &[-1, 1]);
    let new_factor = poly(&x, &[-2, 1]);
    let denominator = &known_divisor.pow(3) * &new_factor.pow(2);
    let a = ring
        .try_from_fraction(&poly(&x, &[1]), &denominator)
        .unwrap();
    assert_eq!(ring.factor_count(), 2);
    assert_eq!(ring.factor(0), original);
    assert_eq!(ring.to_fraction(&a), (poly(&x, &[1]), denominator.clone()));
    assert_eq!(ring.mul_polynomial(&a, &denominator), ring.one());
    // The other divisor still uses the original block after basis growth.
    let other = ring
        .try_from_fraction(&poly(&x, &[1]), &poly(&x, &[1, 1]))
        .unwrap();
    assert_eq!(ring.factor_count(), 2);
    assert_eq!(
        other.denominator_exponents().collect::<Vec<_>>(),
        vec![(0, 1)]
    );
}

#[test]
fn nested_rational_coefficients_support_extension_without_outer_factorization() {
    use crate::domains::{
        factorized_rational_polynomial::FactorizedRationalPolynomialField,
        rational_polynomial::RationalPolynomialField,
    };
    type Polynomial = MultivariatePolynomial<IntegerRing, u16>;
    type Coefficient = RationalPolynomial<IntegerRing, u16>;
    type FactoredCoefficient = FactorizedRationalPolynomial<IntegerRing, u16>;

    fn check<F: Field>(field: F, coefficient: F::Element) {
        let polynomial = UnivariatePolynomial::from_coefficients(
            &field,
            vec![coefficient, field.zero(), field.one()],
            variable(),
        );
        let ring = PartialFractionRing::with_basis(field, variable(), &[]).unwrap();
        let a = ring.element_from_polynomial(&polynomial);
        let inverse = ring.inv(&a);
        assert_eq!(ring.factor_count(), 1);
        assert_eq!(ring.factor(0), polynomial);
        assert_eq!(ring.mul(&a, &inverse), ring.one());
        assert_eq!(ring.inv(&inverse), a);
    }

    let y = Arc::new(vec![symbol!("pf_extension_y").into()]);
    let linear_y = |constant: i64| {
        let mut p = Polynomial::new(&Z, None, y.clone());
        p.append_monomial(constant.into(), &[0]);
        p.append_monomial(1.into(), &[1]);
        p
    };
    check(
        RationalPolynomialField::<IntegerRing, u16>::new(Z),
        Coefficient::from_num_den(linear_y(1), linear_y(2), &Z, true),
    );
    check(
        FactorizedRationalPolynomialField::<IntegerRing, u16>::new(Z, y.clone()),
        FactoredCoefficient::from_num_den(linear_y(1), vec![(linear_y(2), 1)], &Z, false),
    );
}

#[test]
fn concurrent_overlapping_denominators_preserve_values_and_coprime_blocks() {
    use std::sync::Barrier;

    for _ in 0..24 {
        let x = variable();
        let ring = PartialFractionRing::with_basis(Q, x.clone(), &[]).unwrap();
        let denominators = [
            poly(&x, &[-1, 0, 1]),
            poly(&x, &[-1, 1]),
            poly(&x, &[1, 1]).pow(2),
        ];
        let start = Barrier::new(denominators.len());
        let answers = std::thread::scope(|scope| {
            let handles: Vec<_> = denominators
                .iter()
                .map(|denominator| {
                    let ring = &ring;
                    let start = &start;
                    let x = &x;
                    scope.spawn(move || {
                        start.wait();
                        ring.try_from_fraction(&poly(x, &[1]), denominator).unwrap()
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        for (answer, denominator) in answers.iter().zip(&denominators) {
            assert_eq!(
                ring.to_fraction(answer),
                (poly(&x, &[1]), denominator.clone())
            );
            assert_eq!(ring.mul_polynomial(answer, denominator), ring.one());
        }
        for i in 0..ring.factor_count() {
            for j in 0..i {
                assert_eq!(
                    dense::gcd(
                        &Q,
                        &ring.factor(i).coefficients,
                        &ring.factor(j).coefficients
                    ),
                    vec![Q.one()]
                );
            }
        }
    }
}
