use super::*;
use crate::domains::factorized_rational_polynomial::{
    FactorizedRationalPolynomial, FromNumeratorAndFactorizedDenominator,
};
use crate::domains::rational_polynomial::{FromNumeratorAndDenominator, RationalPolynomial};
use crate::prelude::*;

fn materialize<R>(value: FactorizedRationalPolynomial<R, u16>) -> RationalPolynomial<R, u16>
where
    R: EuclideanDomain + PolynomialGCD<u16>,
    RationalPolynomial<R, u16>: FromNumeratorAndDenominator<R, R, u16>,
{
    let numerator = value.numerator.mul_coeff(value.numer_coeff);
    let ring = numerator.ring().clone();
    let denominator = value.denominators.into_iter().fold(
        numerator.constant(value.denom_coeff),
        |product, (base, power)| product * &base.pow(power),
    );
    RationalPolynomial::from_num_den(numerator, denominator, &ring, true)
}

#[test]
fn integer_powered_cancellation_preserves_every_denominator_occurrence() {
    let x1 = parse!("x+1").to_polynomial::<_, u16>(&Z, None);
    for (numerator_power, denominator_power) in [(1, 2), (2, 3), (3, 2)] {
        let numerator = x1.pow(numerator_power);
        let expected = RationalPolynomial::from_num_den(
            numerator.clone(),
            x1.pow(denominator_power),
            &Z,
            true,
        );
        let actual = FactorizedRationalPolynomial::from_num_den(
            numerator,
            vec![(x1.clone(), denominator_power)],
            &Z,
            true,
        );
        assert_eq!(materialize(actual), expected);
    }

    let x2 = parse!("x+2").to_polynomial::<_, u16>(&Z, None);
    let base = &x1 * &x2;
    let expected = RationalPolynomial::from_num_den(x1.clone(), base.pow(3), &Z, true);
    let actual = FactorizedRationalPolynomial::from_num_den(x1, vec![(base, 3)], &Z, true);
    assert_eq!(materialize(actual), expected);
}

#[test]
fn integer_duplicate_and_overlapping_factors_preserve_multiplicities() {
    let x1 = parse!("x+1").to_polynomial::<_, u16>(&Z, None);
    let x2 = parse!("x+2").to_polynomial::<_, u16>(&Z, None);
    let numerator = x1.pow(2);
    let bases = vec![(x1.clone(), 2), (&x1 * &x2, 3), (x2.clone(), 1)];
    let denominator = bases
        .iter()
        .fold(x1.one(), |acc, (base, p)| acc * &base.pow(*p));
    let expected = RationalPolynomial::from_num_den(numerator.clone(), denominator, &Z, true);
    let actual = FactorizedRationalPolynomial::from_num_den(numerator, bases, &Z, true);
    assert_eq!(actual.denominators.len(), 2);
    assert!(actual.denominators.contains(&(x1, 3)));
    assert!(actual.denominators.contains(&(x2, 4)));
    assert_eq!(materialize(actual), expected);
}

#[test]
fn rational_contents_are_scaled_independently_for_every_power() {
    for numerator in ["1/2", "-2/3", "(x+1)/2", "x/2+y/3", "0"] {
        let numerator = Atom::parse(numerator, "factored_construction", ParseSettings::default())
            .unwrap()
            .to_polynomial::<_, u16>(&Q, None);
        for factors in [
            vec![],
            vec![("(x+1)/2", 2)],
            vec![("-2*(x+1)/3", 3)],
            vec![("(x+1)/2", 2), ("(x+1)/3", 1)],
            vec![("(x+1)/2", 2), ("(y+2)/5", 3)],
            vec![("x/2+y/3", 2), ("-3/5", 3), ("z+1", 0)],
            vec![("0", 0), ("(x+1)/2", 1)],
        ] {
            let bases = factors
                .into_iter()
                .map(|(base, power)| {
                    (
                        Atom::parse(base, "factored_construction", ParseSettings::default())
                            .unwrap()
                            .to_polynomial::<_, u16>(&Q, None),
                        power,
                    )
                })
                .collect::<Vec<_>>();
            let denominator = bases
                .iter()
                .fold(numerator.one(), |acc, (base, p)| acc * &base.pow(*p));
            let expected =
                RationalPolynomial::from_num_den(numerator.clone(), denominator, &Z, true);
            for do_factor in [false, true] {
                let actual = FactorizedRationalPolynomial::from_num_den(
                    numerator.clone(),
                    bases.clone(),
                    &Z,
                    do_factor,
                );
                let mut actual = materialize(actual);
                for variable in numerator.variables().iter().chain(
                    bases
                        .iter()
                        .filter(|(_, power)| *power != 0)
                        .flat_map(|(base, _)| base.variables().iter()),
                ) {
                    assert!(actual.get_variables().contains(variable));
                }
                let mut expected = expected.clone();
                actual.unify_variables(&mut expected);
                assert_eq!(actual, expected);
            }
        }
    }
}

#[test]
#[should_panic(expected = "Zero denominator")]
fn rational_zero_denominator_is_rejected_before_zero_numerator_shortcuts() {
    let zero = parse!("0").to_polynomial::<_, u16>(&Q, None);
    let _ = FactorizedRationalPolynomial::from_num_den(zero.clone(), vec![(zero, 2)], &Z, true);
}

#[test]
fn every_native_constructor_rejects_positive_powers_of_zero() {
    fn verify<R>(zero: MultivariatePolynomial<R, u16>)
    where
        R: EuclideanDomain + PolynomialGCD<u16>,
        FactorizedRationalPolynomial<R, u16>: FromNumeratorAndFactorizedDenominator<R, R, u16>,
    {
        for do_factor in [false, true] {
            for numerator in [zero.clone(), zero.one()] {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    FactorizedRationalPolynomial::from_num_den(
                        numerator.clone(),
                        vec![(zero.clone(), 2)],
                        zero.ring(),
                        do_factor,
                    )
                }));
                assert!(result.is_err());
                let valid = FactorizedRationalPolynomial::from_num_den(
                    numerator.clone(),
                    vec![(zero.clone(), 0)],
                    zero.ring(),
                    do_factor,
                );
                assert_eq!(valid.numerator.is_zero(), numerator.is_zero());
                assert!(!zero.ring().is_zero(&valid.denom_coeff));
            }
        }
    }
    verify(parse!("0").to_polynomial::<_, u16>(&Z, None));
    verify(parse!("0").to_polynomial::<_, u16>(&Zp::new(17), None));
    let algebraic = AlgebraicExtension::new(parse!("a^2-2").to_polynomial::<_, u16>(&Q, None));
    verify(parse!("0").to_polynomial::<_, u16>(&algebraic, None));
}

fn assert_duplicate_arithmetic<R>(
    duplicate: FactorizedRationalPolynomial<R, u16>,
    single: FactorizedRationalPolynomial<R, u16>,
) where
    R: EuclideanDomain + PolynomialGCD<u16>,
    MultivariatePolynomial<R, u16>: Factorize,
    FactorizedRationalPolynomial<R, u16>: FromNumeratorAndFactorizedDenominator<R, R, u16>,
    RationalPolynomial<R, u16>: FromNumeratorAndDenominator<R, R, u16>,
{
    if duplicate.is_zero() {
        assert!(duplicate.denominators.is_empty());
    } else {
        assert_eq!(duplicate.denominators.len(), 1);
    }
    let duplicate_value = materialize(duplicate.clone());
    let single_value = materialize(single.clone());
    assert_eq!(
        materialize(&duplicate + &single),
        &duplicate_value + &single_value
    );
    assert_eq!(
        materialize(&single + &duplicate),
        &single_value + &duplicate_value
    );
    assert_eq!(
        materialize(&duplicate - &single),
        &duplicate_value - &single_value
    );
    assert_eq!(
        materialize(&duplicate * &single),
        &duplicate_value * &single_value
    );
    assert_eq!(
        materialize(&single * &duplicate),
        &single_value * &duplicate_value
    );
    assert_eq!(
        materialize(&duplicate / &single),
        &duplicate_value / &single_value
    );
}

#[test]
fn unfactored_integer_duplicates_and_sign_associates_support_exact_arithmetic() {
    let base = parse!("x+1").to_polynomial::<_, u16>(&Z, None);
    let single =
        FactorizedRationalPolynomial::from_num_den(base.one(), vec![(base.clone(), 1)], &Z, false);
    for numerator in [base.one(), base.clone(), base.zero()] {
        for (first, second) in [(1, 1), (2, 3)] {
            for sign in [1, -1] {
                let duplicate = FactorizedRationalPolynomial::from_num_den(
                    numerator.clone(),
                    vec![
                        (base.clone(), first),
                        (base.clone().mul_coeff(sign.into()), second),
                    ],
                    &Z,
                    false,
                );
                if !duplicate.is_zero() {
                    assert_eq!(duplicate.denominators[0].1, first + second);
                }
                assert_duplicate_arithmetic(duplicate, single.clone());
            }
        }
    }
    let rational_base = parse!("(x+1)/2").to_polynomial::<_, u16>(&Q, None);
    let other = parse!("-(x+1)/3").to_polynomial::<_, u16>(&Q, None);
    let duplicate = FactorizedRationalPolynomial::from_num_den(
        rational_base.one(),
        vec![(rational_base, 2), (other, 3)],
        &Z,
        false,
    );
    assert_duplicate_arithmetic(duplicate, single);
}

#[test]
fn unfactored_field_associates_merge_after_monic_normalization() {
    let field = Zp::new(17);
    let base = parse!("x+1").to_polynomial::<_, u16>(&field, None);
    let single = FactorizedRationalPolynomial::from_num_den(
        base.one(),
        vec![(base.clone(), 1)],
        &field,
        false,
    );
    let duplicate = FactorizedRationalPolynomial::from_num_den(
        base.one(),
        vec![
            (base.clone().mul_coeff(field.nth(3.into())), 2),
            (base.mul_coeff(field.nth(5.into())), 3),
        ],
        &field,
        false,
    );
    assert_eq!(duplicate.denominators[0].1, 5);
    assert_duplicate_arithmetic(duplicate, single);

    let field = AlgebraicExtension::new(parse!("a^2-2").to_polynomial::<_, u16>(&Q, None));
    let base = parse!("x+1").to_polynomial::<_, u16>(&field, None);
    let single = FactorizedRationalPolynomial::from_num_den(
        base.one(),
        vec![(base.clone(), 1)],
        &field,
        false,
    );
    let duplicate = FactorizedRationalPolynomial::from_num_den(
        base.one(),
        vec![
            (base.clone().mul_coeff(field.generator()), 2),
            (base.mul_coeff(field.nth(3.into())), 3),
        ],
        &field,
        false,
    );
    assert_eq!(duplicate.denominators[0].1, 5);
    assert_duplicate_arithmetic(duplicate, single);
}

#[test]
#[should_panic(expected = "denominator factor exponent overflow")]
fn duplicate_exponent_addition_is_checked_without_factorization() {
    let base = parse!("x+1").to_polynomial::<_, u16>(&Z, None);
    let _ = FactorizedRationalPolynomial::from_num_den(
        base.one(),
        vec![(base.clone(), usize::MAX), (base, 1)],
        &Z,
        false,
    );
}

#[test]
fn integer_negative_base_normalization_observes_power_parity() {
    let base = parse!("-2*x-2").to_polynomial::<_, u16>(&Z, None);
    for power in [1, 2, 3, 4] {
        for do_factor in [false, true] {
            let expected = RationalPolynomial::from_num_den(base.one(), base.pow(power), &Z, true);
            let actual = FactorizedRationalPolynomial::from_num_den(
                base.one(),
                vec![(base.clone(), power)],
                &Z,
                do_factor,
            );
            assert_eq!(materialize(actual), expected);
        }
    }
}

#[test]
fn finite_field_repeated_scalar_units_and_cancellations_are_exact() {
    let field = Zp::new(17);
    let base = parse!("3*x+3").to_polynomial::<_, u16>(&field, None);
    for power in [2, 3] {
        for do_factor in [false, true] {
            for numerator in [base.one(), base.clone()] {
                let expected = RationalPolynomial::from_num_den(
                    numerator.clone(),
                    base.pow(power),
                    &field,
                    true,
                );
                let actual = FactorizedRationalPolynomial::from_num_den(
                    numerator,
                    vec![(base.clone(), power)],
                    &field,
                    do_factor,
                );
                assert_eq!(materialize(actual), expected);
            }
        }
    }
}

#[test]
fn algebraic_field_repeated_leading_units_are_exact() {
    let field = AlgebraicExtension::new(parse!("a^2-2").to_polynomial::<_, u16>(&Q, None));
    let base = parse!("x+1")
        .to_polynomial::<_, u16>(&field, None)
        .mul_coeff(field.generator());
    for power in [2, 3] {
        for do_factor in [false, true] {
            let expected =
                RationalPolynomial::from_num_den(base.one(), base.pow(power), &field, true);
            let actual = FactorizedRationalPolynomial::from_num_den(
                base.one(),
                vec![(base.clone(), power)],
                &field,
                do_factor,
            );
            assert_eq!(materialize(actual), expected);
        }
    }
}

#[test]
#[should_panic(expected = "denominator factor exponent overflow")]
fn factored_exponent_multiplication_is_checked() {
    let base = parse!("(x+1)^2").to_polynomial::<_, u16>(&Z, None);
    let _ =
        FactorizedRationalPolynomial::from_num_den(base.one(), vec![(base, usize::MAX)], &Z, true);
}
