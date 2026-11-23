use rand::{SeedableRng, rngs::StdRng};

use super::*;
use crate::{
    atom::AtomCore,
    domains::{
        UniformlySampleableRing,
        finite_field::{FiniteFieldSamplingPolicy, Zp64},
        integer::Z,
        rational::Q,
        rational_polynomial::RationalPolynomialField,
    },
    parse, symbol,
};

fn variable() -> Arc<PolyVariable> {
    Arc::new(symbol!("pf_x").into())
}

fn q_poly(
    x: &Arc<PolyVariable>,
    coefficients: &[(i64, i64)],
) -> UnivariatePolynomial<RationalField> {
    UnivariatePolynomial::from_coefficients(
        &Q,
        coefficients.iter().map(|&(n, d)| (n, d).into()).collect(),
        x.clone(),
    )
}

/// Linear factors with small and fractional roots, two coprime quadratics and
/// a cubic.
fn seeded_q_ring() -> PartialFractionRing<RationalField> {
    let x = variable();
    let factors = [
        q_poly(&x, &[(-1, 1), (1, 1)]),                // x - 1
        q_poly(&x, &[(2, 1), (1, 1)]),                 // x + 2
        q_poly(&x, &[(0, 1), (1, 1)]),                 // x
        q_poly(&x, &[(1, 3), (1, 1)]),                 // x + 1/3
        q_poly(&x, &[(1, 1), (0, 1), (1, 1)]),         // x^2 + 1
        q_poly(&x, &[(3, 1), (1, 1), (2, 1)]),         // 2 x^2 + x + 3
        q_poly(&x, &[(5, 1), (0, 1), (1, 1), (1, 1)]), // x^3 + x^2 + 5
    ];
    PartialFractionRing::with_basis(Q, x, &factors).unwrap()
}

fn q_policy() -> PartialFractionSamplingPolicy<RangeInclusive<Integer>> {
    PartialFractionSamplingPolicy {
        degree: 0..=2,
        exponent: 0..=2,
        coefficient: Integer::from(-5)..=Integer::from(5),
    }
}

fn zp_policy() -> PartialFractionSamplingPolicy<FiniteFieldSamplingPolicy> {
    PartialFractionSamplingPolicy {
        degree: 0..=3,
        exponent: 0..=3,
        coefficient: FiniteFieldSamplingPolicy::Full,
    }
}

/// Sample an element with only a few factors, as in typical use.
fn sparse_sample<F: Field + SampleableRing>(
    ring: &PartialFractionRing<F>,
    rng: &mut StdRng,
    policy: &PartialFractionSamplingPolicy<F::SamplingPolicy>,
) -> PartialFraction<F> {
    let mut a = ring.sample(rng, policy);
    // Drop all but two random factors.
    let keep: Vec<usize> = (0..2)
        .map(|_| rng.random_range(0..ring.factor_count()))
        .collect();
    let factors = a
        .denominator_exponents()
        .filter(|(f, _)| keep.contains(f))
        .map(|(f, _)| (f, a.factor_coefficients(f).to_vec()))
        .collect();
    a = ring
        .element_from_coefficients(a.polynomial_part().to_vec(), factors)
        .unwrap();
    a
}

fn to_q_rational(
    ring: &PartialFractionRing<RationalField>,
    a: &PartialFraction<RationalField>,
) -> RationalPolynomial<IntegerRing, u16> {
    ring.to_rational_polynomial(a, &Z)
}

#[test]
fn arithmetic_agrees_with_rational_polynomials_over_q() {
    let mut rng = StdRng::seed_from_u64(1);
    for _ in 0..60 {
        // Keep the sample pool bounded: inversion below may append a factor.
        let ring = seeded_q_ring();
        let a = sparse_sample(&ring, &mut rng, &q_policy());
        let b = sparse_sample(&ring, &mut rng, &q_policy());
        let (ra, rb) = (to_q_rational(&ring, &a), to_q_rational(&ring, &b));
        assert_eq!(ring.try_from_rational_polynomial(&ra).unwrap(), a);
        assert_eq!(to_q_rational(&ring, &ring.add(&a, &b)), &ra + &rb);
        assert_eq!(to_q_rational(&ring, &ring.sub(&a, &b)), &ra - &rb);
        assert_eq!(to_q_rational(&ring, &ring.mul(&a, &b)), &ra * &rb);
        assert_eq!(to_q_rational(&ring, &ring.neg(&a)), -ra.clone());
        if !b.is_zero() {
            let quotient = &ra / &rb;
            let q = ring.div(&a, &b);
            assert_eq!(to_q_rational(&ring, &q), quotient);
        }
    }
}

#[test]
fn sum_agrees_with_repeated_addition() {
    let ring = seeded_q_ring();
    let mut rng = StdRng::seed_from_u64(3);
    for n in [0, 1, 2, 3, 10, 40] {
        let mut terms: Vec<_> = (0..n)
            .map(|_| sparse_sample(&ring, &mut rng, &q_policy()))
            .collect();
        // Zero summands and summands that cancel the highest poles.
        if n >= 3 {
            terms.push(ring.zero());
            terms.push(ring.neg(&terms[0]));
            terms.push(ring.neg(&terms[2]));
        }
        let expected = terms
            .iter()
            .fold(to_q_rational(&ring, &ring.zero()), |s, t| {
                &s + &to_q_rational(&ring, t)
            });
        let sum = ring.sum(terms.clone());
        assert_eq!(to_q_rational(&ring, &sum), expected);
        let mut folded = ring.zero();
        for t in &terms {
            ring.add_assign(&mut folded, t);
        }
        assert_eq!(sum, folded);
    }

    // Complete cancellation gives the canonical zero.
    let a = sparse_sample(&ring, &mut rng, &q_policy());
    assert!(ring.sum([a.clone(), ring.neg(&a), ring.zero()]).is_zero());
}

#[test]
fn arithmetic_agrees_with_rational_polynomials_over_zp() {
    let x = variable();
    let field = Zp64::new(1_000_000_007);
    let e = |v: i64| field.nth(v.into());
    let poly = |c: &[i64]| {
        UnivariatePolynomial::from_coefficients(
            &field,
            c.iter().map(|&v| e(v)).collect(),
            x.clone(),
        )
    };
    let factors = [
        poly(&[-1, 1]),
        poly(&[2, 1]),
        poly(&[0, 1]),
        poly(&[7, 3]),
        poly(&[1, 0, 1]),
        poly(&[3, 1, 2]),
        poly(&[5, 0, 1, 1]),
    ];
    let ring = PartialFractionRing::with_basis(field.clone(), x, &factors).unwrap();
    let mut rng = StdRng::seed_from_u64(2);
    for _ in 0..100 {
        let a = sparse_sample(&ring, &mut rng, &zp_policy());
        let b = sparse_sample(&ring, &mut rng, &zp_policy());
        let to = |v: &PartialFraction<Zp64>| -> RationalPolynomial<Zp64, u16> {
            ring.to_rational_polynomial(v, &field)
        };
        let (ra, rb) = (to(&a), to(&b));
        assert_eq!(ring.try_from_rational_polynomial(&ra).unwrap(), a);
        assert_eq!(to(&ring.add(&a, &b)), &ra + &rb);
        assert_eq!(to(&ring.sub(&a, &b)), &ra - &rb);
        assert_eq!(to(&ring.mul(&a, &b)), &ra * &rb);
        let mut c = a.clone();
        ring.add_mul_assign(&mut c, &a, &b);
        assert_eq!(to(&c), &ra + &(&ra * &rb));
        // Evaluation at a random point.
        let point = field.sample_uniform(&mut rng);
        let expected = ra
            .numerator
            .replace(0, &point)
            .to_univariate_from_univariate(0)
            .get_constant();
        let den_a = ra
            .denominator
            .replace(0, &point)
            .to_univariate_from_univariate(0)
            .get_constant();
        match ring.evaluate(&a, &point) {
            Some(v) => assert_eq!(field.mul(&v, &den_a), expected),
            None => assert!(field.is_zero(&den_a)),
        }
    }
}

#[test]
fn ring_axioms() {
    let ring = seeded_q_ring();
    let mut rng = StdRng::seed_from_u64(3);
    for _ in 0..30 {
        let a = sparse_sample(&ring, &mut rng, &q_policy());
        let b = sparse_sample(&ring, &mut rng, &q_policy());
        let c = sparse_sample(&ring, &mut rng, &q_policy());
        assert_eq!(ring.add(&a, &b), ring.add(&b, &a));
        assert_eq!(ring.mul(&a, &b), ring.mul(&b, &a));
        assert_eq!(
            ring.add(&ring.add(&a, &b), &c),
            ring.add(&a, &ring.add(&b, &c))
        );
        assert_eq!(
            ring.mul(&ring.mul(&a, &b), &c),
            ring.mul(&a, &ring.mul(&b, &c))
        );
        assert_eq!(
            ring.mul(&a, &ring.add(&b, &c)),
            ring.add(&ring.mul(&a, &b), &ring.mul(&a, &c))
        );
        assert_eq!(ring.mul(&a, &ring.one()), a);
        assert_eq!(ring.add(&a, &ring.zero()), a);
        assert!(ring.is_zero(&ring.sub(&a, &a)));
        assert_eq!(ring.pow(&a, 3), ring.mul(&a, &ring.mul(&a, &a)));
        // Owned and borrowed operations agree.
        assert_eq!(ring.add(a.clone(), b.clone()), ring.add(&a, &b));
        assert_eq!(ring.mul(a.clone(), b.clone()), ring.mul(&a, &b));
        let mut d = a.clone();
        ring.sub_mul_assign(&mut d, &b, &c);
        assert_eq!(d, ring.sub(&a, &ring.mul(&b, &c)));
    }
}

#[test]
fn polynomial_products_and_factor_division() {
    let ring = seeded_q_ring();
    let x = ring.variable().clone();
    let mut rng = StdRng::seed_from_u64(4);
    for _ in 0..30 {
        let a = sparse_sample(&ring, &mut rng, &q_policy());
        let p = q_poly(&x, &[(3, 1), (-2, 5), (0, 1), (1, 7)]);
        let expected = ring.mul(&a, &ring.element_from_polynomial(&p));
        assert_eq!(ring.mul_polynomial(&a, &p), expected);
        let mut y = a.clone();
        ring.mul_variable_assign(&mut y);
        assert_eq!(
            y,
            ring.mul(
                &a,
                &ring.element_from_polynomial(&q_poly(&x, &[(0, 1), (1, 1)]))
            )
        );
        for factor in 0..ring.factor_count() {
            let divided = ring.div_factor_power(&a, factor, 2);
            assert_eq!(
                ring.mul_polynomial(&divided, &ring.factor(factor).pow(2)),
                a
            );
            assert_eq!(divided, ring.mul(&a, &ring.inv_factor_power(factor, 2)));
        }
        // Products sharing the powers a x^i agree with the plain ones, also
        // when a x^i loses a pole (the factor x).
        let mut cache = PowerCache::new();
        for q in [
            p.clone(),
            q_poly(&x, &[(1, 1)]),
            q_poly(&x, &[(0, 1), (0, 1), (0, 1), (0, 1), (2, 1), (1, 3)]),
        ] {
            for den in [&[][..], &[(2, 1)][..], &[(1, 2), (4, 1), (1, 1)][..]] {
                assert_eq!(
                    ring.mul_fraction_cached(&a, &q, den, &mut cache),
                    ring.mul_fraction(&a, &q, den)
                );
            }
        }
        let x_pole = ring.inv_factor_power(2, 1);
        cache.clear();
        assert_eq!(
            ring.mul_fraction_cached(&x_pole, &p, &[(0, 1)], &mut cache),
            ring.mul_fraction(&x_pole, &p, &[(0, 1)])
        );
        // Repeated and empty entries in the denominator list.
        assert_eq!(
            ring.mul_fraction(&a, &p, &[(1, 1), (4, 1), (0, 0), (1, 1)]),
            ring.mul_fraction(&a, &p, &[(1, 2), (4, 1)])
        );
        let fraction = ring.mul_fraction(&a, &p, &[(1, 2), (4, 1)]);
        let denominator = ring.mul(&ring.inv_factor_power(1, 2), &ring.inv_factor_power(4, 1));
        assert_eq!(fraction, ring.mul(&expected, &denominator));
    }
}

#[test]
fn conversions() {
    let ring = seeded_q_ring();
    let x = ring.variable().clone();
    let variables = Arc::new(vec![x.as_ref().clone()]);
    let r: RationalPolynomial<IntegerRing, u16> = parse!(
        "(pf_x^5 + 3)/((pf_x-1)^2*(pf_x^2+1)*(3*pf_x+1))"
    )
    .to_rational_polynomial(&Q, &Z, Some(variables.clone()));
    let a = ring.try_from_rational_polynomial(&r).unwrap();
    assert_eq!(ring.to_rational_polynomial::<_, u16>(&a, &Z), r);
    assert_eq!(
        a.denominator_exponents().collect::<Vec<_>>(),
        [(0, 2), (3, 1), (4, 1)]
    );
    assert_eq!(ring.denominator_degree(&a), 5);

    let f: FactorizedRationalPolynomial<IntegerRing, u16> =
        parse!("(pf_x^5 + 3)/((pf_x-1)^2*(pf_x^2+1)*(3*pf_x+1))")
            .to_factorized_rational_polynomial(&Q, &Z, Some(variables.clone()));
    assert_eq!(ring.try_from_factorized_rational_polynomial(&f).unwrap(), a);
    let back: FactorizedRationalPolynomial<IntegerRing, u16> =
        ring.to_factorized_rational_polynomial(&a, &Z);
    assert_eq!(
        ring.try_from_factorized_rational_polynomial(&back).unwrap(),
        a
    );

    // A new denominator extends the seeded basis.
    let outside: RationalPolynomial<IntegerRing, u16> =
        parse!("1/(pf_x-2)").to_rational_polynomial(&Q, &Z, Some(variables.clone()));
    let count = ring.factor_count();
    let imported = ring.try_from_rational_polynomial(&outside).unwrap();
    assert_eq!(ring.factor_count(), count + 1);
    assert_eq!(to_q_rational(&ring, &imported), outside);
    // A cancelling factor outside the basis is also accepted.
    let n = q_poly(&x, &[(-4, 1), (1, 1)]);
    let d = q_poly(&x, &[(4, 1), (-5, 1), (1, 1)]); // (x-1)(x-4)
    assert_eq!(
        ring.try_from_fraction(&n, &d).unwrap(),
        ring.inv_factor_power(0, 1)
    );
    // Another variable.
    let other: RationalPolynomial<IntegerRing, u16> =
        parse!("1/(pf_x-1)+pf_y").to_rational_polynomial(&Q, &Z, None);
    assert_eq!(
        ring.try_from_rational_polynomial(&other),
        Err(PartialFractionError::OtherVariable)
    );

    // Units and inverses.
    let unit = ring.mul(&ring.nth(3.into()), &ring.inv_factor_power(4, 2));
    let inverse = ring.try_inv(&unit).unwrap();
    assert!(ring.is_one(&ring.mul(&unit, &inverse)));
    // Inverting (x + 3)/(x + 2) discovers x + 3 in the seeded basis.
    let unit = ring.add(&ring.one(), &ring.inv_factor_power(1, 1));
    let count = ring.factor_count();
    let inverse = ring.inv(&unit);
    assert_eq!(ring.factor_count(), count + 1);
    assert!(ring.is_one(&ring.mul(&unit, &inverse)));
    let unit = ring.add(&ring.one(), &ring.inv_factor_power(0, 1));
    assert!(ring.is_one(&ring.mul(&unit, &ring.try_inv(&unit).unwrap())));

    // Raw coefficient round trip.
    let parts = a
        .denominator_exponents()
        .map(|(f, _)| (f, a.factor_coefficients(f).to_vec()))
        .collect();
    assert_eq!(
        ring.element_from_coefficients(a.polynomial_part().to_vec(), parts),
        Some(a.clone())
    );

    assert!(!format!("{}", ring.printer(&a)).is_empty());
    assert!(format!("{ring}").contains("1/"));
}

#[test]
fn constructors() {
    let x = variable();
    let variables = Arc::new(vec![x.as_ref().clone()]);
    let p = |s: &str| parse!(s).to_polynomial::<_, u16>(&Q, Some(variables.clone()));
    let ring = PartialFractionRing::from_denominators(
        Q,
        x.clone(),
        &[p("(pf_x-1)^2*(2*pf_x+3)"), p("(pf_x^2+1)*(pf_x-1)"), p("5")],
    )
    .unwrap();
    assert_eq!(ring.factor_count(), 3);

    assert_eq!(
        PartialFractionRing::with_basis(Q, x.clone(), &[q_poly(&x, &[(2, 1)])]),
        Err(PartialFractionError::ConstantFactor(0))
    );
    assert_eq!(
        PartialFractionRing::with_basis(
            Q,
            x.clone(),
            &[
                q_poly(&x, &[(-1, 1), (1, 1)]),
                q_poly(&x, &[(-2, 1), (2, 1)])
            ]
        ),
        Err(PartialFractionError::NotCoprime(0, 1))
    );
    assert_eq!(
        PartialFractionRing::with_basis(
            Q,
            x.clone(),
            &[
                q_poly(&x, &[(-1, 1), (0, 1), (1, 1)]),
                q_poly(&x, &[(1, 1), (1, 1)])
            ]
        ),
        Err(PartialFractionError::NotCoprime(0, 1))
    );
    // Hints are factored: x^2 - 1 and x + 1 give the basis x - 1, x + 1.
    let ring = PartialFractionRing::new(
        Q,
        x.clone(),
        &[
            q_poly(&x, &[(-1, 1), (0, 1), (1, 1)]),
            q_poly(&x, &[(1, 1), (1, 1)]),
        ],
    );
    assert_eq!(ring.factor_count(), 2);
}

#[test]
fn factorizing_basis_discovers_and_reuses_factors() {
    let x = variable();
    let variables = Arc::new(vec![x.as_ref().clone()]);
    let ring = PartialFractionRing::new(Q, x.clone(), &[q_poly(&x, &[(-1, 1), (1, 1)])]);
    let seeded = seeded_q_ring();
    let mut rng = StdRng::seed_from_u64(6);
    for _ in 0..20 {
        // Elements of a seeded basis enter the factorizing ring through
        // their rational form, and arithmetic agrees.
        let a = sparse_sample(&seeded, &mut rng, &q_policy());
        let b = sparse_sample(&seeded, &mut rng, &q_policy());
        let (ra, rb) = (to_q_rational(&seeded, &a), to_q_rational(&seeded, &b));
        let (ea, eb) = (
            ring.try_from_rational_polynomial(&ra).unwrap(),
            ring.try_from_rational_polynomial(&rb).unwrap(),
        );
        assert_eq!(to_q_rational(&ring, &ring.mul(&ea, &eb)), &ra * &rb);
        assert_eq!(to_q_rational(&ring, &ring.add(&ea, &eb)), &ra + &rb);
        if !b.is_zero() {
            assert_eq!(to_q_rational(&ring, &ring.div(&ea, &eb)), &ra / &rb);
            assert!(ring.is_one(&ring.mul(&eb, &ring.inv(&eb))));
        }
    }
    // Factor indices never change: the first factor is still x - 1.
    assert_eq!(ring.factor(0), q_poly(&x, &[(-1, 1), (1, 1)]));
    let count = ring.factor_count();
    assert!(count > 1);
    // A new irreducible denominator extends the learned basis.
    let r: RationalPolynomial<IntegerRing, u16> =
        parse!("1/(pf_x^2-7)").to_rational_polynomial(&Q, &Z, Some(variables));
    let e = ring.try_from_rational_polynomial(&r).unwrap();
    assert_eq!(ring.factor_count(), count + 1);
    assert_eq!(ring.to_rational_polynomial::<_, u16>(&e, &Z), r);
    // Clones share the basis, also across threads.
    let shared = ring.clone();
    let handle = std::thread::spawn(move || {
        let y = shared.variable().clone();
        let f = q_poly(&y, &[(11, 1), (0, 1), (1, 1)]);
        shared
            .try_from_fraction(&q_poly(&y, &[(1, 1)]), &f)
            .unwrap()
    });
    let from_thread = handle.join().unwrap();
    assert_eq!(ring.factor_count(), count + 2);
    assert!(ring.is_one(&ring.mul(
        &from_thread,
        &ring.element_from_polynomial(&q_poly(&x, &[(11, 1), (0, 1), (1, 1)]))
    )));
}

#[test]
fn images_modulo_primes() {
    let ring = seeded_q_ring();
    let mut rng = StdRng::seed_from_u64(5);
    let field = Zp64::new(1_000_000_007);
    let image = ring.to_finite_field(&field).unwrap();
    for _ in 0..30 {
        let a = sparse_sample(&ring, &mut rng, &q_policy());
        let b = sparse_sample(&ring, &mut rng, &q_policy());
        let map =
            |v: &PartialFraction<RationalField>| ring.element_to_finite_field(&image, v).unwrap();
        assert_eq!(map(&ring.mul(&a, &b)), image.mul(&map(&a), &map(&b)));
        assert_eq!(map(&ring.add(&a, &b)), image.add(&map(&a), &map(&b)));
        let rational: RationalPolynomial<IntegerRing, u16> = ring.to_rational_polynomial(&a, &Z);
        assert_eq!(
            image.to_rational_polynomial::<_, u16>(&map(&a), &field),
            rational.to_finite_field(&field)
        );
    }

    // x + 1/3 has no image modulo 3.
    assert_eq!(
        ring.to_finite_field(&Zp64::new(3)),
        Err(PartialFractionError::UnmappableCoefficient)
    );
    // x - 1 and x + 2 coincide modulo 3; x^2 + 1 = (x - 2)(x + 2) modulo 5.
    let x = ring.variable().clone();
    let small = PartialFractionRing::with_basis(
        Q,
        x.clone(),
        &[
            q_poly(&x, &[(-1, 1), (1, 1)]),
            q_poly(&x, &[(2, 1), (1, 1)]),
        ],
    )
    .unwrap();
    assert_eq!(
        small.to_finite_field(&Zp64::new(3)),
        Err(PartialFractionError::NotCoprime(0, 1))
    );
    let quadratic = PartialFractionRing::with_basis(
        Q,
        x.clone(),
        &[
            q_poly(&x, &[(1, 1), (0, 1), (1, 1)]),
            q_poly(&x, &[(-2, 1), (1, 1)]),
        ],
    )
    .unwrap();
    assert_eq!(
        quadratic.to_finite_field(&Zp64::new(5)),
        Err(PartialFractionError::NotCoprime(0, 1))
    );
    assert!(quadratic.to_finite_field(&Zp64::new(7)).is_ok());
    // A coefficient denominator divisible by the prime.
    let (prime, target) = [11, 13, 17, 19, 23]
        .into_iter()
        .find_map(|p| Some((p, ring.to_finite_field(&Zp64::new(p)).ok()?)))
        .unwrap();
    let a = ring.add(
        &ring.inv_factor_power(4, 1),
        &ring.element_from_dense(vec![(1, prime as i64).into()]),
    );
    assert_eq!(
        ring.element_to_finite_field(&target, &a),
        Err(PartialFractionError::UnmappableCoefficient)
    );
}

/// Check products, fused multiply-adds, divisions and their canonical forms
/// against rational polynomials over Zp, for samples of `ring`.
fn check_against_rational_polynomials_over_zp(
    ring: &PartialFractionRing<Zp64>,
    field: &Zp64,
    rng: &mut StdRng,
    policy: &PartialFractionSamplingPolicy<FiniteFieldSamplingPolicy>,
    sparse: bool,
    iterations: usize,
) {
    let to = |v: &PartialFraction<Zp64>| -> RationalPolynomial<Zp64, u16> {
        ring.to_rational_polynomial(v, field)
    };
    // Canonical: equal to the element rebuilt from its rational form.
    let canon = |v: &PartialFraction<Zp64>, r: &RationalPolynomial<Zp64, u16>| {
        assert_eq!(&ring.try_from_rational_polynomial(r).unwrap(), v);
    };
    for it in 0..iterations {
        let s = |rng: &mut StdRng| {
            if sparse {
                sparse_sample(ring, rng, policy)
            } else {
                ring.sample(rng, policy)
            }
        };
        let a = s(rng);
        let b = s(rng);
        let c = s(rng);
        let (ra, rb, rc) = (to(&a), to(&b), to(&c));
        let p = ring.mul(&a, &b);
        let rp = &ra * &rb;
        assert_eq!(to(&p), rp, "mul it {it}");
        canon(&p, &rp);
        // add_mul / sub_mul onto a nonzero accumulator.
        let mut d = c.clone();
        ring.add_mul_assign(&mut d, &a, &b);
        let rd = &rc + &rp;
        assert_eq!(to(&d), rd);
        canon(&d, &rd);
        let mut d = c.clone();
        ring.sub_mul_assign(&mut d, &a, &b);
        let rd = &rc - &rp;
        assert_eq!(to(&d), rd);
        canon(&d, &rd);
        // Total and partial cancellation.
        let mut d = p.clone();
        ring.sub_mul_assign(&mut d, &b, &a);
        assert!(d.is_zero(), "full cancellation");
        assert!(d.coefficients().is_empty());
        let mut d = ring.add(&p, &c);
        ring.sub_mul_assign(&mut d, &a, &b);
        assert_eq!(d, c);
        let mut d = ring.neg(&p);
        ring.add_mul_assign(&mut d, &a, &b);
        assert!(d.is_zero());
        // Zero accumulator, sub path.
        let mut d = ring.zero();
        ring.sub_mul_assign(&mut d, &a, &b);
        assert_eq!(d, ring.neg(&p));
        // Squares, one, zero.
        let sq = ring.mul(&a, &a);
        let rsq = &ra * &ra;
        assert_eq!(to(&sq), rsq);
        canon(&sq, &rsq);
        assert_eq!(ring.mul(&a, &ring.one()), a);
        assert_eq!(ring.mul(&ring.one(), &a), a);
        assert!(ring.mul(&a, &ring.zero()).is_zero());
        let mut z = ring.zero();
        ring.add_mul_assign(&mut z, &a, &ring.zero());
        assert!(z.is_zero());
        // Factor division and powers.
        for f in 0..ring.factor_count() {
            for e in [1, 3] {
                let q = ring.div_factor_power(&a, f, e);
                let rq = &ra * &to(&ring.inv_factor_power(f, e));
                assert_eq!(to(&q), rq);
                canon(&q, &rq);
            }
        }
        // Division by elements whose inverse needs no new basis factor: a
        // division by an arbitrary element would append composite factors
        // to the shared basis, which later samples use.
        for f in 0..ring.factor_count() {
            let d = ring.inv_factor_power(f, 2);
            let q = ring.div(&a, &d);
            let rq = &ra / &to(&d);
            assert_eq!(to(&q), rq, "div it {it}");
            canon(&q, &rq);
        }
        // Three-fold product with mixed association.
        let t = ring.mul(&ring.mul(&a, &b), &c);
        assert_eq!(t, ring.mul(&a, &ring.mul(&b, &c)));
        assert_eq!(to(&t), &rp * &rc);
    }
}

fn zp_ring(prime: u64, factors: &[&[i64]]) -> (PartialFractionRing<Zp64>, Zp64) {
    let x = variable();
    let field = Zp64::new(prime);
    let fs: Vec<_> = factors
        .iter()
        .map(|c| {
            UnivariatePolynomial::from_coefficients(
                &field,
                c.iter().map(|&v| field.nth(v.into())).collect(),
                x.clone(),
            )
        })
        .collect();
    (
        PartialFractionRing::with_basis(field.clone(), x, &fs).unwrap(),
        field,
    )
}

#[test]
fn products_over_zp_with_high_linear_exponents() {
    let (ring, field) = zp_ring(
        1_000_000_007,
        &[&[-1, 1], &[2, 1], &[0, 1], &[7, 3], &[-5, 1]],
    );
    let mut rng = StdRng::seed_from_u64(11);
    let policy = PartialFractionSamplingPolicy {
        degree: 0..=4,
        exponent: 0..=6,
        coefficient: FiniteFieldSamplingPolicy::Full,
    };
    check_against_rational_polynomials_over_zp(&ring, &field, &mut rng, &policy, false, 40);
    check_against_rational_polynomials_over_zp(&ring, &field, &mut rng, &policy, true, 60);
}

#[test]
fn products_over_zp_with_nonlinear_and_composite_factors() {
    // The blocks (x - 1)(x + 3) and (x + 5)(x^2 + 2) are composite, the
    // others are x + 2, x^2 + 1, x, 2 x^2 + x + 3 and x^4 + x^3 + 9.
    let (ring, field) = zp_ring(
        1_000_000_007,
        &[
            &[-3, 2, 1],
            &[2, 1],
            &[1, 0, 1],
            &[10, 2, 5, 1],
            &[0, 1],
            &[3, 1, 2],
            &[9, 0, 0, 1, 1],
        ],
    );
    let mut rng = StdRng::seed_from_u64(12);
    let policy = PartialFractionSamplingPolicy {
        degree: 0..=5,
        exponent: 0..=4,
        coefficient: FiniteFieldSamplingPolicy::Full,
    };
    check_against_rational_polynomials_over_zp(&ring, &field, &mut rng, &policy, true, 60);
    let policy = PartialFractionSamplingPolicy {
        degree: 0..=2,
        exponent: 0..=2,
        coefficient: FiniteFieldSamplingPolicy::Full,
    };
    check_against_rational_polynomials_over_zp(&ring, &field, &mut rng, &policy, false, 20);
    // Small coefficients: many zero blocks and accidental cancellations.
    let small = PartialFractionSamplingPolicy {
        degree: 0..=3,
        exponent: 0..=5,
        coefficient: (0..=2).into(),
    };
    check_against_rational_polynomials_over_zp(&ring, &field, &mut rng, &small, true, 80);
}

#[test]
fn products_over_a_small_field_with_cancellations() {
    // A tiny prime makes zero coefficients and cancellations frequent.
    let (ring, field) = zp_ring(5, &[&[1, 1], &[2, 0, 1], &[1, 1, 0, 1]]);
    let mut rng = StdRng::seed_from_u64(13);
    let policy = PartialFractionSamplingPolicy {
        degree: 0..=3,
        exponent: 0..=4,
        coefficient: FiniteFieldSamplingPolicy::Full,
    };
    check_against_rational_polynomials_over_zp(&ring, &field, &mut rng, &policy, false, 150);
}

#[test]
fn products_over_q_with_dense_high_exponents() {
    let ring = seeded_q_ring();
    let mut rng = StdRng::seed_from_u64(14);
    let policy = PartialFractionSamplingPolicy {
        degree: 0..=3,
        exponent: 0..=4,
        coefficient: Integer::from(-3)..=Integer::from(3),
    };
    for _ in 0..30 {
        let a = sparse_sample(&ring, &mut rng, &policy);
        let b = sparse_sample(&ring, &mut rng, &policy);
        let c = sparse_sample(&ring, &mut rng, &policy);
        let (ra, rb, rc) = (
            to_q_rational(&ring, &a),
            to_q_rational(&ring, &b),
            to_q_rational(&ring, &c),
        );
        let p = ring.mul(&a, &b);
        let rp = &ra * &rb;
        assert_eq!(to_q_rational(&ring, &p), rp);
        assert_eq!(ring.try_from_rational_polynomial(&rp).unwrap(), p);
        let mut d = c.clone();
        ring.sub_mul_assign(&mut d, &a, &b);
        let rd = &rc - &rp;
        assert_eq!(ring.try_from_rational_polynomial(&rd).unwrap(), d);
        let mut d = c.clone();
        ring.add_mul_assign(&mut d, &b, &a);
        let rd = &rc + &rp;
        assert_eq!(ring.try_from_rational_polynomial(&rd).unwrap(), d);
    }
    // Dense samples over all seven factors.
    let policy = PartialFractionSamplingPolicy {
        degree: 0..=2,
        exponent: 0..=2,
        coefficient: Integer::from(-2)..=Integer::from(2),
    };
    for _ in 0..8 {
        let a = ring.sample(&mut rng, &policy);
        let b = ring.sample(&mut rng, &policy);
        let rp = &to_q_rational(&ring, &a) * &to_q_rational(&ring, &b);
        assert_eq!(
            ring.try_from_rational_polynomial(&rp).unwrap(),
            ring.mul(&a, &b)
        );
    }
}

#[test]
fn factorizing_ring_used_from_several_threads() {
    // A factorizing ring that discovers (linear) factors from several threads.
    let x = variable();
    let field = Zp64::new(1_000_000_007);
    let ring = PartialFractionRing::new(field.clone(), x.clone(), &[]);
    let handles: Vec<_> = (0..4)
        .map(|t| {
            let ring = ring.clone();
            let field = field.clone();
            let x = x.clone();
            std::thread::spawn(move || {
                let mut rng = StdRng::seed_from_u64(100 + t);
                let poly = |c: &[u64]| {
                    UnivariatePolynomial::from_coefficients(
                        &field,
                        c.iter().map(|&v| field.nth(v.into())).collect(),
                        x.clone(),
                    )
                };
                for _ in 0..40 {
                    let r1: u64 = rng.random_range(0..12);
                    let r2: u64 = rng.random_range(0..12);
                    let q: u64 = rng.random_range(1..12);
                    // (x+r1)^2 (x+r2) (x^2+q) denominators and numerators.
                    let den = &(&(&poly(&[r1, 1]) * &poly(&[r1, 1])) * &poly(&[r2 + 13, 1]))
                        * &poly(&[q, 0, 1]);
                    let num = poly(&[rng.random_range(0..9), rng.random_range(0..9), 1]);
                    let Ok(a) = ring.try_from_fraction(&num, &den) else {
                        continue;
                    };
                    let b = ring.try_from_fraction(&den, &poly(&[q + 1, 1, 1])).unwrap();
                    let to = |v: &PartialFraction<Zp64>| -> RationalPolynomial<Zp64, u16> {
                        ring.to_rational_polynomial(v, &field)
                    };
                    let (ra, rb) = (to(&a), to(&b));
                    let p = ring.mul(&a, &b);
                    assert_eq!(to(&p), &ra * &rb);
                    if !a.is_zero() {
                        let d = ring.div(&b, &a);
                        assert_eq!(to(&d), &rb / &ra);
                    }
                    let mut s = a.clone();
                    ring.sub_mul_assign(&mut s, &a, &b);
                    assert_eq!(to(&s), &ra - &(&ra * &rb));
                }
            })
        })
        .collect();
    for h in handles {
        h.join().unwrap();
    }
}

/// A numerator and a denominator.
type Fraction<F> = (UnivariatePolynomial<F>, UnivariatePolynomial<F>);

fn fraction_mul<F: Field>(a: &Fraction<F>, b: &Fraction<F>) -> Fraction<F> {
    (&a.0 * &b.0, &a.1 * &b.1)
}

fn fraction_add<F: Field>(a: &Fraction<F>, b: &Fraction<F>, subtract: bool) -> Fraction<F> {
    let (x, y) = (&a.0 * &b.1, &b.0 * &a.1);
    (if subtract { &x - &y } else { &x + &y }, &a.1 * &b.1)
}

/// Check the products, fused multiply-adds, products with fractions and
/// polynomials, factor divisions, multiplication by the variable and sums of
/// `a`, `b` and `c` against the elements imported from the same operations on
/// their fractions, an independent code path. None of them may extend the
/// basis.
fn check_against_fractions<F: Field>(
    ring: &PartialFractionRing<F>,
    a: &PartialFraction<F>,
    b: &PartialFraction<F>,
    c: &PartialFraction<F>,
    polynomials: &[UnivariatePolynomial<F>],
) {
    let import = |(n, d): &Fraction<F>| ring.try_from_fraction(n, d).unwrap();
    let (fa, fb, fc) = (
        ring.to_fraction(a),
        ring.to_fraction(b),
        ring.to_fraction(c),
    );
    let factor_count = ring.factor_count();
    assert_eq!(&import(&fa), a);
    let fab = fraction_mul(&fa, &fb);
    let ab = import(&fab);
    assert_eq!(ring.mul(a, b), ab);
    assert_eq!(ring.mul(b, a), ab);
    assert_eq!(ring.mul(a, a), import(&fraction_mul(&fa, &fa)));
    for subtract in [false, true] {
        let expected = import(&fraction_add(&fc, &fab, subtract));
        for (x, y) in [(a, b), (b, a)] {
            let mut d = c.clone();
            if subtract {
                ring.sub_mul_assign(&mut d, x, y);
            } else {
                ring.add_mul_assign(&mut d, x, y);
            }
            assert_eq!(d, expected);
            // A zero accumulator that keeps the storage of c.
            let mut d = c.clone();
            ring.sub_assign(&mut d, c);
            assert!(d.is_zero() && d.coefficients().is_empty());
            if subtract {
                ring.sub_mul_assign(&mut d, x, y);
                assert_eq!(d, ring.neg(&ab));
            } else {
                ring.add_mul_assign(&mut d, x, y);
                assert_eq!(d, ab);
            }
        }
    }
    let mut d = ring.add(&ab, c);
    ring.sub_mul_assign(&mut d, b, a);
    assert_eq!(&d, c);
    let abc = import(&fraction_mul(&fab, &fc));
    assert_eq!(ring.mul(&ab, c), abc);
    assert_eq!(ring.mul(a, &ring.mul(b, c)), abc);

    let one = ring.factor(0).one();
    let mut cache = PowerCache::new();
    for (i, p) in polynomials.iter().enumerate() {
        let denominator: Vec<_> = (0..factor_count)
            .filter(|f| (i + f) % 3 != 0)
            .map(|f| (f, (i + f) % 4))
            .collect();
        let d = denominator
            .iter()
            .fold(one.clone(), |d, &(f, e)| &d * &ring.factor(f).pow(e));
        let expected = import(&fraction_mul(&fa, &(p.clone(), d)));
        assert_eq!(ring.mul_fraction(a, p, &denominator), expected);
        assert_eq!(
            ring.mul_fraction_cached(a, p, &denominator, &mut cache),
            expected
        );
        let expected = import(&fraction_mul(&fa, &(p.clone(), one.clone())));
        assert_eq!(ring.mul_polynomial(a, p), expected);
    }
    for f in 0..factor_count {
        for e in [1, 2, 5] {
            let expected = import(&(fa.0.clone(), &fa.1 * &ring.factor(f).pow(e)));
            assert_eq!(ring.div_factor_power(a, f, e), expected);
        }
    }
    let mut d = a.clone();
    ring.mul_variable_assign(&mut d);
    let x = one.monomial(ring.field().one(), 1);
    assert_eq!(d, import(&(&fa.0 * &x, fa.1.clone())));
    let sum = ring.sum([a.clone(), b.clone(), c.clone(), ring.neg(a)]);
    assert_eq!(sum, import(&fraction_add(&fb, &fc, false)));
    assert_eq!(ring.factor_count(), factor_count);
}

/// A random element with a polynomial part of up to `poly` coefficients and
/// up to `factors` distinct factors of exponents up to `exponent`, whose
/// coefficients, from `coefficient`, may be zero anywhere.
fn random_element<F: Field>(
    ring: &PartialFractionRing<F>,
    rng: &mut StdRng,
    (poly, factors, exponent): (usize, usize, usize),
    coefficient: &mut impl FnMut(&mut StdRng) -> F::Element,
) -> PartialFraction<F> {
    let polynomial = (0..rng.random_range(0..=poly))
        .map(|_| coefficient(rng))
        .collect();
    let mut blocks: Vec<(usize, Vec<F::Element>)> = vec![];
    for _ in 0..rng.random_range(0..=factors) {
        let f = rng.random_range(0..ring.factor_count());
        if blocks.iter().all(|(g, _)| *g != f) {
            let len = rng.random_range(1..=exponent) * ring.factor(f).degree();
            blocks.push((f, (0..len).map(|_| coefficient(rng)).collect()));
        }
    }
    ring.element_from_coefficients(polynomial, blocks).unwrap()
}

/// Check random elements of `ring`, including constants and polynomials, with
/// [check_against_fractions].
fn check_random_against_fractions<F: Field>(
    ring: &PartialFractionRing<F>,
    rng: &mut StdRng,
    iterations: usize,
    exponent: usize,
    polynomials: &[UnivariatePolynomial<F>],
    mut coefficient: impl FnMut(&mut StdRng) -> F::Element,
) {
    let n = ring.factor_count();
    for i in 0..iterations {
        let shape = [(0, 1), (1, 2), (3, n), (6, n)][i % 4];
        let a = random_element(ring, rng, (shape.0, shape.1, exponent), &mut coefficient);
        let b = random_element(ring, rng, (4, 2, exponent), &mut coefficient);
        let c = random_element(ring, rng, (3, n, exponent), &mut coefficient);
        check_against_fractions(ring, &a, &b, &c, polynomials);
        let constant = random_element(ring, rng, (1, 0, 1), &mut coefficient);
        let polynomial = random_element(ring, rng, (4, 0, 1), &mut coefficient);
        check_against_fractions(ring, &constant, &a, &b, polynomials);
        check_against_fractions(ring, &a, &constant, &polynomial, polynomials);
        check_against_fractions(ring, &polynomial, &b, &c, polynomials);
    }
}

#[test]
fn arithmetic_agrees_with_fractions_over_small_fields() {
    let check =
        |prime: u64, factors: &[&[i64]], seed: u64, iterations: usize| {
            let (ring, field) = zp_ring(prime, factors);
            let mut rng = StdRng::seed_from_u64(seed);
            let polynomials: Vec<_> = (1..=4)
                .map(|len| {
                    let c = (0..len)
                        .map(|_| field.nth(rng.random_range(0..prime).into()))
                        .collect();
                    UnivariatePolynomial::from_coefficients(&field, c, ring.variable().clone())
                })
                .collect();
            // Mostly zero or small coefficients, which cancel often.
            check_random_against_fractions(&ring, &mut rng, iterations, 6, &polynomials, |rng| {
                match rng.random_range(0..10) {
                    0..3 => field.zero(),
                    3..7 => field.nth(rng.random_range(0..3.min(prime)).into()),
                    _ => field.nth(rng.random_range(0..prime).into()),
                }
            });
        };
    check(3, &[&[1, 1], &[1, 0, 1], &[0, 1], &[2, 2, 0, 1]], 4, 100);
    // x^2 + 1, the composite (x + 1)(x + 2), x^3 + x + 1, x + 3 and x^4 + 2.
    check(
        7,
        &[
            &[1, 0, 1],
            &[2, 3, 1],
            &[1, 1, 0, 1],
            &[3, 1],
            &[2, 0, 0, 0, 1],
        ],
        2,
        60,
    );
}

#[test]
fn arithmetic_agrees_with_fractions_over_rational_functions() {
    type Coefficient = RationalPolynomial<IntegerRing, u16>;
    let field = RationalPolynomialField::<IntegerRing, u16>::new(Z);
    let y = Arc::new(vec![symbol!("pf_y").into()]);
    // (a y + b) / (c y + d).
    let element = |[a, b, c, d]: [i64; 4]| -> Coefficient {
        let linear = |slope: i64, constant: i64| {
            let mut p = MultivariatePolynomial::new(&Z, None, y.clone());
            p.append_monomial(constant.into(), &[0]);
            p.append_monomial(slope.into(), &[1]);
            p
        };
        Coefficient::from_num_den(linear(a, b), linear(c, d), &Z, true)
    };
    let x = variable();
    let poly = |c: Vec<Coefficient>| UnivariatePolynomial::from_coefficients(&field, c, x.clone());
    let (zero, one) = (field.zero(), field.one());
    // x - y, x - 2 y - 1, x^2 + y and x.
    let ring = PartialFractionRing::with_basis(
        field.clone(),
        x.clone(),
        &[
            poly(vec![element([-1, 0, 0, 1]), one.clone()]),
            poly(vec![element([-2, -1, 0, 1]), one.clone()]),
            poly(vec![element([1, 0, 0, 1]), zero.clone(), one.clone()]),
            poly(vec![zero, one.clone()]),
        ],
    )
    .unwrap();
    let mut rng = StdRng::seed_from_u64(6);
    let polynomials = [poly(vec![one, element([1, 0, 0, 1])])];
    check_random_against_fractions(&ring, &mut rng, 4, 2, &polynomials, |rng| {
        match rng.random_range(0..4) {
            0 => field.zero(),
            1 => element([0, rng.random_range(-2..=2), 0, 1]),
            _ => element([
                rng.random_range(-2..=2),
                rng.random_range(-2..=2),
                1,
                rng.random_range(1..=3),
            ]),
        }
    });
}
