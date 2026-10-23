//! Arithmetic shortcuts must also work for publicly constructible, unreduced
//! rational polynomials and coefficient domains other than the integers.

use super::*;
use crate::{
    atom::{Atom, AtomCore},
    domains::rational::Q,
    parse, symbol,
};

fn fixture<R>(
    ring: &R,
    variables: &Arc<Vec<PolyVariable>>,
    numerator: Atom,
    denominator: Atom,
    reduce: bool,
) -> RationalPolynomial<R, u16>
where
    R: EuclideanDomain + PolynomialGCD<u16>,
    RationalPolynomial<R, u16>: FromNumeratorAndDenominator<R, R, u16>,
{
    let convert = |atom: Atom| {
        atom.to_polynomial::<_, u16>(&Z, Some(variables.clone()))
            .map_coeff(|coefficient| ring.nth(coefficient.clone()), ring.clone())
    };
    RationalPolynomial::from_num_den(convert(numerator), convert(denominator), ring, reduce)
}

fn arithmetic_oracle<R>(
    left: &RationalPolynomial<R, u16>,
    right: &RationalPolynomial<R, u16>,
    subtract: bool,
) -> RationalPolynomial<R, u16>
where
    R: EuclideanDomain + PolynomialGCD<u16>,
    RationalPolynomial<R, u16>: FromNumeratorAndDenominator<R, R, u16>,
{
    let (mut left, mut right) = (left.clone(), right.clone());
    left.unify_variables(&mut right);
    let first = &left.numerator * &right.denominator;
    let second = &right.numerator * &left.denominator;
    let numerator = if subtract {
        first - second
    } else {
        first + second
    };
    RationalPolynomial::from_num_den(
        numerator,
        &left.denominator * &right.denominator,
        left.numerator.ring(),
        true,
    )
}

fn shared_denominator_matrix<R>(ring: R)
where
    R: EuclideanDomain + PolynomialGCD<u16>,
    RationalPolynomial<R, u16>: FromNumeratorAndDenominator<R, R, u16>,
{
    let variables = Arc::new(vec![
        symbol!("rat_shortcut_x").into(),
        symbol!("rat_shortcut_y").into(),
        symbol!("rat_shortcut_unused").into(),
    ]);
    for (first, second, denominator, reduce) in [
        (
            parse!("rat_shortcut_x+rat_shortcut_y"),
            parse!("1-rat_shortcut_x"),
            parse!("rat_shortcut_y+1"),
            true,
        ),
        (
            parse!("rat_shortcut_x+1"),
            parse!("(rat_shortcut_x+1)*rat_shortcut_y"),
            parse!("(rat_shortcut_x+1)*(rat_shortcut_y+1)"),
            false,
        ),
        (
            parse!("3*rat_shortcut_x+6"),
            parse!("-3*rat_shortcut_x-6"),
            parse!("6*(rat_shortcut_y+1)"),
            false,
        ),
        (
            parse!("rat_shortcut_x+1"),
            parse!("rat_shortcut_y-1"),
            parse!("1"),
            true,
        ),
    ] {
        let left = fixture(&ring, &variables, first, denominator.clone(), reduce);
        let right = fixture(&ring, &variables, second, denominator, reduce);
        assert_eq!(left.denominator, right.denominator);
        let inputs = (left.clone(), right.clone());
        assert_eq!(&left + &right, arithmetic_oracle(&left, &right, false));
        assert_eq!(&left - &right, arithmetic_oracle(&left, &right, true));
        assert_eq!(
            left.clone() - right.clone(),
            arithmetic_oracle(&left, &right, true)
        );
        let zero = &left - &left;
        assert!(zero.is_zero());
        assert!(zero.denominator.is_one());
        assert_eq!(zero.numerator.variables(), &variables);
        assert_eq!(zero.denominator.variables(), &variables);
        assert_eq!((left, right), inputs);
    }

    // Equal mathematical denominators reach the shortcut after the existing
    // variable unification, not by comparing diagnostic names or indices.
    let reordered = Arc::new(vec![
        variables[1].clone(),
        variables[0].clone(),
        variables[2].clone(),
    ]);
    let left = fixture(
        &ring,
        &variables,
        parse!("rat_shortcut_x+1"),
        parse!("rat_shortcut_y+2"),
        true,
    );
    let right = fixture(
        &ring,
        &reordered,
        parse!("rat_shortcut_y-1"),
        parse!("rat_shortcut_y+2"),
        true,
    );
    for (actual, expected) in [
        (&left + &right, arithmetic_oracle(&left, &right, false)),
        (&left - &right, arithmetic_oracle(&left, &right, true)),
    ] {
        assert_eq!(actual, expected);
        assert_eq!(actual.numerator.variables(), &variables);
        assert_eq!(actual.denominator.variables(), &variables);
    }

    // Keep the non-equal-denominator path under the same independent oracle.
    let other = fixture(
        &ring,
        &variables,
        parse!("rat_shortcut_y+1"),
        parse!("rat_shortcut_x+2"),
        true,
    );
    assert_eq!(&left + &other, arithmetic_oracle(&left, &other, false));
    assert_eq!(&left - &other, arithmetic_oracle(&left, &other, true));
}

fn parameter_derivative_matrix<R>(ring: R)
where
    R: EuclideanDomain + PolynomialGCD<u16>,
    RationalPolynomial<R, u16>: FromNumeratorAndDenominator<R, R, u16>,
{
    let variables = Arc::new(vec![
        symbol!("rat_shortcut_y").into(),
        symbol!("rat_shortcut_x").into(),
        symbol!("rat_shortcut_unused").into(),
    ]);
    for (numerator, denominator, reduce) in [
        (
            parse!(
                "rat_shortcut_x^3+2*rat_shortcut_y*rat_shortcut_x^2+rat_shortcut_y^2*rat_shortcut_x+1"
            ),
            parse!("(rat_shortcut_y+1)^2"),
            true,
        ),
        (
            parse!("rat_shortcut_x^3+1"),
            parse!("6*(rat_shortcut_y+1)"),
            true,
        ),
        (
            parse!("(rat_shortcut_y+1)*rat_shortcut_x^2"),
            parse!("(rat_shortcut_y+1)^2"),
            false,
        ),
        (
            parse!("rat_shortcut_x^17"),
            parse!("rat_shortcut_y+1"),
            true,
        ),
        (parse!("1"), parse!("rat_shortcut_y+1"), true),
        (
            parse!("rat_shortcut_x^3+rat_shortcut_y"),
            parse!("rat_shortcut_x+rat_shortcut_y+1"),
            true,
        ),
    ] {
        let input = fixture(&ring, &variables, numerator, denominator, reduce);
        let original = input.clone();
        for variable in 0..variables.len() {
            let expected = RationalPolynomial::from_num_den(
                &input.numerator.derivative(variable) * &input.denominator
                    - &input.numerator * &input.denominator.derivative(variable),
                &input.denominator * &input.denominator,
                &ring,
                true,
            );
            let actual = input.derivative(variable);
            assert_eq!(actual, expected);
            assert_eq!(actual.numerator.variables(), &variables);
            assert_eq!(actual.denominator.variables(), &variables);
        }
        assert_eq!(input, original);
    }
}

#[test]
fn integer_shared_denominators_normalize_cancellations_and_unreduced_inputs() {
    shared_denominator_matrix(Z);
}

#[test]
fn rational_shared_denominators_normalize_cancellations_and_unreduced_inputs() {
    // Symbolica represents Q-rational functions over Z; there is no
    // RationalPolynomial<Q> constructor. Exercise its Q -> Z boundary.
    let variables = Arc::new(vec![
        symbol!("rat_shortcut_x").into(),
        symbol!("rat_shortcut_y").into(),
    ]);
    for reduce in [false, true] {
        let make = |numerator: Atom| {
            RationalPolynomial::from_num_den(
                numerator.to_polynomial::<_, u16>(&Q, Some(variables.clone())),
                parse!("(rat_shortcut_y+1)/5").to_polynomial::<_, u16>(&Q, Some(variables.clone())),
                &Z,
                reduce,
            )
        };
        let left = make(parse!("(rat_shortcut_x+rat_shortcut_y)/3"));
        let right = make(parse!("(1-rat_shortcut_x)/3"));
        assert_eq!(left.denominator, right.denominator);
        assert_eq!(&left + &right, arithmetic_oracle(&left, &right, false));
        assert_eq!(&left - &right, arithmetic_oracle(&left, &right, true));
        assert!((&left - &left).is_zero());
    }
}

#[test]
fn finite_field_shared_denominators_normalize_cancellations_and_unreduced_inputs() {
    shared_denominator_matrix(super::super::finite_field::Zp::new(17));
}

#[test]
fn integer_parameter_derivatives_normalize_exposed_content() {
    parameter_derivative_matrix(Z);
}

#[test]
fn rational_parameter_derivatives_normalize_exposed_content() {
    let variables = Arc::new(vec![
        symbol!("rat_shortcut_x").into(),
        symbol!("rat_shortcut_y").into(),
    ]);
    for reduce in [false, true] {
        let input = RationalPolynomial::from_num_den(
            parse!("(rat_shortcut_x^3+1)/7").to_polynomial::<_, u16>(&Q, Some(variables.clone())),
            parse!("6*(rat_shortcut_y+1)/5").to_polynomial::<_, u16>(&Q, Some(variables.clone())),
            &Z,
            reduce,
        );
        let expected: RationalPolynomial<_, u16> = parse!(
            "5*rat_shortcut_x^2/(14*(rat_shortcut_y+1))"
        )
        .to_rational_polynomial(&Q, &Z, Some(variables.clone()));
        assert_eq!(input.derivative(0), expected);
    }
}

#[test]
fn finite_field_parameter_derivatives_keep_characteristic_cancellation() {
    parameter_derivative_matrix(super::super::finite_field::Zp::new(17));
}
