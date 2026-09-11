//! Tests for core univariate-polynomial arithmetic.

use crate::{
    atom::AtomCore,
    domains::{SampleableRing, integer::Z, rational::Q},
    parse,
    poly::univariate::{UnivariatePolynomialRing, UnivariatePolynomialSamplingPolicy},
};
use rand::{SeedableRng, rngs::StdRng};

#[test]
fn samples_with_degree_and_coefficient_policies() {
    let template = parse!("x")
        .to_polynomial::<_, u8>(&Z, None)
        .to_univariate_from_univariate(0);
    let ring = UnivariatePolynomialRing::from_polynomial(&template);
    let policy = UnivariatePolynomialSamplingPolicy {
        degree: 3..=3,
        coefficient: 1.into()..=1.into(),
    };
    let mut rng = StdRng::seed_from_u64(1);

    let sample = ring.sample(&mut rng, &policy);

    assert_eq!(sample.degree(), 3);
    assert!(
        sample
            .coefficients()
            .iter()
            .all(|coefficient| coefficient == &1)
    );
}

#[test]
fn derivative_and_integral_are_inverses() {
    let polynomial = parse!("x^2+5x+x^7+3")
        .to_polynomial::<_, u8>(&Q, None)
        .to_univariate_from_univariate(0);

    assert_eq!(polynomial.integrate().derivative(), polynomial);
}

#[test]
fn arithmetic_and_evaluation() {
    let a = parse!("x^2+5x+x^7+3")
        .to_polynomial::<_, u8>(&Z, None)
        .to_univariate_from_univariate(0);
    let b = parse!("x^2 + 6")
        .to_polynomial::<_, u8>(&Z, None)
        .to_univariate_from_univariate(0);

    let expected_sum = parse!("9+5*x+2*x^2+x^7")
        .to_polynomial::<_, u8>(&Z, None)
        .to_univariate_from_univariate(0);
    let expected_product = parse!("18+30*x+9*x^2+5*x^3+x^4+6*x^7+x^9")
        .to_polynomial::<_, u8>(&Z, None)
        .to_univariate_from_univariate(0);
    let expected_quotient = parse!("1+36*x+-6*x^3+x^5")
        .to_polynomial::<_, u8>(&Z, None)
        .to_univariate_from_univariate(0);
    let expected_remainder = parse!("-3+-211*x")
        .to_polynomial::<_, u8>(&Z, None)
        .to_univariate_from_univariate(0);

    assert_eq!(&a + &b, expected_sum);
    assert_eq!(&a * &b, expected_product);
    assert_eq!(a.quot_rem(&b), (expected_quotient, expected_remainder));
    assert_eq!(a.evaluate(&5.into()), 78178);
}

#[test]
fn pseudo_remainder_does_not_divide_coefficients() {
    let factor = parse!("2*x^2+3*x+5").to_polynomial::<_, u8>(&Z, None);
    let variables = factor.variables().clone();
    let cofactor = parse!("7*x^2-x+4")
        .to_polynomial::<_, u8>(&Z, Some(variables.clone()))
        .to_univariate_from_univariate(0);
    let factor = factor.to_univariate_from_univariate(0);
    let product = &factor * &cofactor;
    let non_factor = parse!("2*x+1")
        .to_polynomial::<_, u8>(&Z, Some(variables))
        .to_univariate_from_univariate(0);

    assert!(product.pseudo_remainder(&factor).is_zero());
    assert!(!product.pseudo_remainder(&non_factor).is_zero());
}

#[test]
fn gcd_over_rationals_uses_polynomial_gcd() {
    let a = parse!("(x^12+x+1)*(x^8-3*x^2+2)")
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0);
    let b = parse!("(x^10-x+3)*(x^8-3*x^2+2)")
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0);
    let expected = parse!("x^8-3*x^2+2")
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0);

    assert_eq!(a.gcd(&b), expected);
    assert_eq!(a.gcd_euclidean(&b), expected);
}
