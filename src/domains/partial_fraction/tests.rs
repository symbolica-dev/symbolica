use rand::{SeedableRng, rngs::StdRng};

use super::*;
use crate::{
    atom::AtomCore,
    domains::{
        UniformlySampleableRing,
        finite_field::{FiniteFieldSamplingPolicy, Zp64},
        integer::Z,
        rational::Q,
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
fn q_ring() -> PartialFractionRing<RationalField> {
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
    PartialFractionRing::with_fixed_basis(Q, x, &factors).unwrap()
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
    let ring = q_ring();
    let mut rng = StdRng::seed_from_u64(1);
    for _ in 0..60 {
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
            match ring.try_div(&a, &b) {
                Some(q) => assert_eq!(to_q_rational(&ring, &q), quotient),
                None => assert!(ring.try_from_rational_polynomial(&quotient).is_err()),
            }
        }
    }
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
    let ring = PartialFractionRing::with_fixed_basis(field.clone(), x, &factors).unwrap();
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
    let ring = q_ring();
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
    let ring = q_ring();
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
    let ring = q_ring();
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

    // A denominator outside the basis.
    let outside: RationalPolynomial<IntegerRing, u16> =
        parse!("1/(pf_x-2)").to_rational_polynomial(&Q, &Z, Some(variables.clone()));
    assert_eq!(
        ring.try_from_rational_polynomial(&outside),
        Err(PartialFractionError::OutsideBasis)
    );
    // A cancelling factor outside the basis is fine.
    let n = q_poly(&x, &[(-2, 1), (1, 1)]);
    let d = q_poly(&x, &[(2, 1), (-3, 1), (1, 1)]); // (x-1)(x-2)
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
    // 1 + 1/(x + 2) = (x + 3)/(x + 2) is not a unit; 1 + 1/(x - 1) = x/(x - 1) is.
    assert!(
        ring.try_inv(&ring.add(&ring.one(), &ring.inv_factor_power(1, 1)))
            .is_none()
    );
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
        PartialFractionRing::with_fixed_basis(Q, x.clone(), &[q_poly(&x, &[(2, 1)])]),
        Err(PartialFractionError::ConstantFactor(0))
    );
    assert_eq!(
        PartialFractionRing::with_fixed_basis(
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
        PartialFractionRing::with_fixed_basis(
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
    assert!(ring.is_self_extending());
    assert!(!ring.strict().is_self_extending());
}

#[test]
fn self_extending_basis() {
    let x = variable();
    let variables = Arc::new(vec![x.as_ref().clone()]);
    let ring = PartialFractionRing::new(Q, x.clone(), &[q_poly(&x, &[(-1, 1), (1, 1)])]);
    let fixed = q_ring();
    let mut rng = StdRng::seed_from_u64(6);
    for _ in 0..20 {
        // Elements of another basis enter the self-extending ring through
        // their rational form, and arithmetic agrees.
        let a = sparse_sample(&fixed, &mut rng, &q_policy());
        let b = sparse_sample(&fixed, &mut rng, &q_policy());
        let (ra, rb) = (to_q_rational(&fixed, &a), to_q_rational(&fixed, &b));
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
    // A strict view refuses new factors; the self-extending ring accepts them.
    let r: RationalPolynomial<IntegerRing, u16> =
        parse!("1/(pf_x^2-7)").to_rational_polynomial(&Q, &Z, Some(variables));
    assert_eq!(
        ring.strict().try_from_rational_polynomial(&r),
        Err(PartialFractionError::OutsideBasis)
    );
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
    let ring = q_ring();
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
    let small = PartialFractionRing::with_fixed_basis(
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
    let quadratic = PartialFractionRing::with_fixed_basis(
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
