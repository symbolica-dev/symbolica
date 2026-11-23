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

/// Monic polynomials from which the random denominators are multiplied:
/// linear factors, quadratics that are irreducible or split, cubics and a
/// cyclotomic quartic.
const POOL: &[&[i64]] = &[
    &[-1, 1],
    &[-2, 1],
    &[-3, 1],
    &[-4, 1],
    &[-5, 1],
    &[-6, 1],
    &[-7, 1],
    &[-8, 1],
    &[-9, 1],
    &[-10, 1],
    &[-11, 1],
    &[-12, 1],
    &[1, 0, 1],
    &[2, 1, 1],
    &[-2, 0, 1],
    &[-1, 0, 1],
    &[3, 0, 0, 1],
    &[1, 1, 1, 1, 1],
];

/// A xorshift step: a pseudo-random number below `bound`.
fn next_random(state: &mut u64, bound: usize) -> i64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state % bound as u64) as i64
}

/// A polynomial in [variable] with integer coefficients.
fn polynomial<F: Field>(field: &F, coefficients: &[i64]) -> UnivariatePolynomial<F> {
    let coefficients = coefficients.iter().map(|&c| field.nth(c.into())).collect();
    UnivariatePolynomial::from_coefficients(field, coefficients, variable())
}

/// A random numerator with a nonzero constant term and a random product of up
/// to four powers of [POOL] polynomials.
fn random_fraction<F: Field>(
    field: &F,
    state: &mut u64,
) -> (UnivariatePolynomial<F>, UnivariatePolynomial<F>) {
    let mut d = polynomial(field, &[1]);
    for _ in 0..=next_random(state, 4) {
        let f = polynomial(field, POOL[next_random(state, POOL.len()) as usize]);
        d = &d * &f.pow(1 + next_random(state, 3) as usize);
    }
    let n = polynomial(
        field,
        &[
            1 + next_random(state, 11),
            next_random(state, 11),
            next_random(state, 3),
        ],
    );
    (n, d)
}

/// Check that the factors `before` keep their indices and that all blocks are
/// pairwise coprime.
fn assert_basis_extends<F: Field>(
    ring: &PartialFractionRing<F>,
    before: &[UnivariatePolynomial<F>],
) {
    let field = ring.field();
    for (i, f) in before.iter().enumerate() {
        assert_eq!(&ring.factor(i), f);
    }
    for i in 0..ring.factor_count() {
        for j in 0..i {
            let g = dense::gcd(
                field,
                &ring.factor(i).coefficients,
                &ring.factor(j).coefficients,
            );
            assert_eq!(g, vec![field.one()]);
        }
    }
}

/// Before every import of a random fraction, check that the scan of the known
/// factors agrees with trial division by every factor of the basis; check the
/// import and its inverse, which enters the numerator as a new denominator.
fn check_known_factor_scan<F: Field>(ring: &PartialFractionRing<F>, seed: u64) {
    let field = ring.field();
    let before: Vec<_> = (0..ring.factor_count()).map(|i| ring.factor(i)).collect();
    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    for _ in 0..30 {
        let (n, d) = random_fraction(field, &mut state);
        let mut expected = vec![];
        let mut rest = d.coefficients.clone();
        for i in 0..ring.factor_count() {
            let mut e = 0;
            loop {
                let (q, r) = dense::divrem_monic(field, &rest, ring.basis.monic(i as u32));
                if !r.is_empty() {
                    break;
                }
                rest = q;
                e += 1;
            }
            if e > 0 {
                expected.push((i, e));
            }
        }
        let (mut exponents, cofactor) = ring.divide_known_factors(&d.coefficients);
        exponents.sort_unstable();
        assert_eq!((exponents, cofactor), (expected, rest));

        let a = ring.try_from_fraction(&n, &d).unwrap();
        assert_eq!(
            ring.mul_polynomial(&a, &d),
            ring.element_from_polynomial(&n)
        );
        let inverse = ring.inv(&a);
        assert_eq!(
            ring.mul_polynomial(&inverse, &n),
            ring.element_from_polynomial(&d)
        );
    }
    assert_basis_extends(ring, &before);
}

#[test]
fn known_factor_scan_agrees_with_trial_division_over_every_basis() {
    let zp = Zp64::new(1_000_000_007);
    for seed in 0..2 {
        check_known_factor_scan(&PartialFractionRing::new(Q, variable(), &[]), seed);
        let hints = [
            poly(&variable(), &[-1, 0, 1]),
            poly(&variable(), &[1, 0, 1]),
        ];
        check_known_factor_scan(&PartialFractionRing::new(Q, variable(), &hints), seed);
        let blocks = [
            poly(&variable(), &[6, -5, 1]),
            poly(&variable(), &[1, 0, 1]),
        ];
        let ring = PartialFractionRing::with_basis(Q, variable(), &blocks).unwrap();
        check_known_factor_scan(&ring, seed);
        check_known_factor_scan(&PartialFractionRing::new(zp.clone(), variable(), &[]), seed);
        let hints: Vec<_> = (1..=6).map(|i| polynomial(&zp, &[-i, 1])).collect();
        check_known_factor_scan(
            &PartialFractionRing::new(zp.clone(), variable(), &hints),
            seed,
        );
        let blocks = [polynomial(&zp, &[2, -3, 1]), polynomial(&zp, &[1, 0, 1])];
        let ring = PartialFractionRing::with_basis(zp.clone(), variable(), &blocks).unwrap();
        check_known_factor_scan(&ring, seed);
    }
}

/// Import random fractions and their inverses on four threads at once, and
/// check the values, the products with the denominators, that re-importing
/// gives the same element, that the existing factors keep their indices and
/// that the blocks stay pairwise coprime.
fn check_random_denominators<F: Field + Send + Sync>(ring: &PartialFractionRing<F>, seed: u64)
where
    F::Element: Send + Sync,
{
    use std::sync::Barrier;

    let field = ring.field();
    let before: Vec<_> = (0..ring.factor_count()).map(|i| ring.factor(i)).collect();
    let start = Barrier::new(4);
    let imports: Vec<_> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4u64)
            .map(|t| {
                let start = &start;
                scope.spawn(move || {
                    let mut state = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ (t + 1);
                    start.wait();
                    (0..20)
                        .map(|_| {
                            let (n, d) = random_fraction(field, &mut state);
                            let d = &d * &polynomial(field, &[2 + next_random(&mut state, 7)]);
                            let a = ring.try_from_fraction(&n, &d).unwrap();
                            let inverse = ring.inv(&a);
                            (n, d, a, inverse)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().unwrap())
            .collect()
    });

    assert_basis_extends(ring, &before);
    for (n, d, a, inverse) in &imports {
        assert_eq!(ring.mul_polynomial(a, d), ring.element_from_polynomial(n));
        assert_eq!(
            ring.mul_polynomial(inverse, n),
            ring.element_from_polynomial(d)
        );
        assert_eq!(&ring.try_from_fraction(n, d).unwrap(), a);
        for p in 100..103 {
            let p = field.nth(p.into());
            let value = field.div(
                &dense::evaluate(field, &n.coefficients, &p),
                &dense::evaluate(field, &d.coefficients, &p),
            );
            assert_eq!(ring.evaluate(a, &p), Some(value));
        }
    }
}

#[test]
fn concurrent_random_denominators_decompose_over_every_basis() {
    let x = variable();
    let zp = Zp64::new(1_000_000_007);
    let blocks = [
        poly(&x, &[-1, 0, 1]),
        poly(&x, &[6, -5, 1]),
        poly(&x, &[1, 0, 1]),
    ];
    let hints: Vec<_> = (1..=8).map(|i| poly(&x, &[-i, 1])).collect();
    for seed in 0..4 {
        check_random_denominators(&PartialFractionRing::new(Q, x.clone(), &[]), seed);
        check_random_denominators(&PartialFractionRing::new(Q, x.clone(), &hints), seed);
        let ring = PartialFractionRing::with_basis(Q, x.clone(), &blocks).unwrap();
        check_random_denominators(&ring, seed);
        check_random_denominators(&PartialFractionRing::new(zp.clone(), x.clone(), &[]), seed);
        let ring = PartialFractionRing::with_basis(zp.clone(), x.clone(), &[]).unwrap();
        check_random_denominators(&ring, seed);
    }
}

#[test]
fn denominator_cache_is_cleared_when_full_without_changing_decompositions() {
    let zp = Zp64::new(1_000_000_007);
    let hints: Vec<_> = (1..=17).map(|i| polynomial(&zp, &[-i, 1])).collect();
    let ring = PartialFractionRing::new(zp.clone(), variable(), &hints);
    // The product of the hints selected by the bits of `id`.
    let denominator = |id: usize| {
        let mut d = vec![zp.one()];
        for (i, f) in hints.iter().enumerate() {
            if id >> i & 1 == 1 {
                d = dense::mul(&zp, &d, &f.coefficients);
            }
        }
        d
    };
    let exponents = |id: usize| -> Vec<_> {
        (0..hints.len())
            .filter(|i| id >> i & 1 == 1)
            .map(|i| (i, 1))
            .collect()
    };
    let total = DENOMINATOR_CACHE_CAPACITY + 6000;
    std::thread::scope(|scope| {
        for t in 0..4 {
            let (ring, denominator, exponents) = (&ring, &denominator, &exponents);
            scope.spawn(move || {
                for id in (1 + t..=total).step_by(4) {
                    let e = ring.denominator_factors(&denominator(id));
                    assert_eq!(e, Some(exponents(id)));
                    let cached = ring.basis.denominators.lock().unwrap().len();
                    assert!(cached <= DENOMINATOR_CACHE_CAPACITY);
                }
            });
        }
    });
    assert_eq!(ring.factor_count(), hints.len());
    for id in (1..=total).step_by(97) {
        assert_eq!(
            ring.denominator_factors(&denominator(id)),
            Some(exponents(id))
        );
    }
}

/// A composite basis over Q(y), where the roots of the linear blocks are
/// rational functions of y.
#[test]
fn composite_basis_over_rational_functions_covers_linear_blocks_by_evaluation() {
    use crate::domains::rational_polynomial::RationalPolynomialField;
    type Coefficient = RationalPolynomial<IntegerRing, u16>;

    let field = RationalPolynomialField::<IntegerRing, u16>::new(Z);
    let y = Arc::new(vec![symbol!("pf_regression_y").into()]);
    let poly_y = |c: &[i64]| {
        let mut p = MultivariatePolynomial::<IntegerRing, u16>::new(&Z, None, y.clone());
        for (i, &c) in c.iter().enumerate() {
            p.append_monomial(c.into(), &[i as u16]);
        }
        p
    };
    let r = |n: &[i64], d: &[i64]| Coefficient::from_num_den(poly_y(n), poly_y(d), &Z, true);
    // The polynomial x - root, or x^2 - root.
    let monic = |root: Coefficient, degree: usize| {
        let mut c = vec![field.zero(); degree + 1];
        c[0] = field.neg(&root);
        c[degree] = field.one();
        UnivariatePolynomial::from_coefficients(&field, c, variable())
    };
    let blocks = [
        monic(r(&[0, 1], &[1]), 1),
        monic(r(&[1, 1], &[1]), 1),
        monic(r(&[0, 1], &[1]), 2),
    ];
    let ring = PartialFractionRing::with_basis(field.clone(), variable(), &blocks).unwrap();
    let mut pool = blocks.to_vec();
    pool.extend([
        monic(r(&[0, 2], &[1]), 1),
        monic(r(&[1], &[1, 1]), 1),
        monic(r(&[3], &[1]), 1),
        monic(r(&[0, 0, 1], &[1]), 2),
    ]);
    let mut state = 12345;
    let mut previous = None;
    for _ in 0..40 {
        let mut d = blocks[0].one();
        for _ in 0..=next_random(&mut state, 3) {
            let f = &pool[next_random(&mut state, pool.len()) as usize];
            d = &d * &f.pow(1 + next_random(&mut state, 3) as usize);
        }
        let n = UnivariatePolynomial::from_coefficients(
            &field,
            vec![
                r(&[next_random(&mut state, 5) - 2, 1], &[1]),
                r(&[1], &[1 + next_random(&mut state, 3)]),
            ],
            variable(),
        );
        let a = ring.try_from_fraction(&n, &d).unwrap();
        assert_eq!(
            ring.mul_polynomial(&a, &d),
            ring.element_from_polynomial(&n)
        );
        // Dividing by `a` enters its numerator as a new denominator.
        if let Some(b) = &previous {
            assert_eq!(&ring.mul(&ring.div(b, &a), &a), b);
        }
        previous = Some(a);
    }
    assert_basis_extends(&ring, &blocks);
}
