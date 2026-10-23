use super::*;
use crate::prelude::*;

fn assert_reconstructs<R: EuclideanDomain + PolynomialGCD<u16>>(
    input: &Factored<R, u16>,
    variable: usize,
    proper_linear: bool,
) where
    Factored<R, u16>: FromNumeratorAndFactorizedDenominator<R, R, u16>,
    RationalPolynomial<R, u16>: FromNumeratorAndDenominator<R, R, u16>,
    Polynomial<R, u16>: Factorize,
{
    let expected = input.to_rational_polynomial();
    let mut reconstructed: RationalPolynomial<R, u16> = input.numerator.zero().into();
    let terms = input.apart_factored_denominators(variable);
    for (numerator, denominator, power) in terms {
        if proper_linear {
            assert_eq!(numerator.numerator.degree(variable), 0);
            assert!(
                numerator
                    .denominators
                    .iter()
                    .all(|(base, _)| base.degree(variable) == 0)
            );
            assert_eq!(denominator.numerator.degree(variable), 1);
            let monic = denominator.to_rational_polynomial();
            assert_eq!(
                monic.numerator.to_univariate(variable).coefficients()[1],
                monic.denominator
            );
        }
        for original in input.get_variables() {
            assert!(numerator.get_variables().contains(original));
            assert!(denominator.get_variables().contains(original));
        }
        reconstructed = &reconstructed
            + &(&numerator.to_rational_polynomial()
                / &denominator.to_rational_polynomial().pow(power as u64));
    }
    let mut expected = expected;
    reconstructed.unify_variables(&mut expected);
    assert_eq!(reconstructed, expected);
}

fn rational_input(numerator: &str, bases: &[(&str, usize)]) -> (Factored<IntegerRing, u16>, usize) {
    let variables = parse!("x+a+b+c+d+y+unused")
        .to_polynomial::<_, u16>(&Q, None)
        .variables()
        .clone();
    let variable = parse!("x").to_polynomial::<_, u16>(&Q, None).variables()[0].clone();
    let index = variables.iter().position(|v| v == &variable).unwrap();
    let polynomial = |text: &str| parse!(text).to_polynomial::<_, u16>(&Q, Some(variables.clone()));
    let numerator = polynomial(numerator);
    let bases = bases
        .iter()
        .map(|(base, power)| (polynomial(base), *power))
        .collect();
    (Factored::from_num_den(numerator, bases, &Z, false), index)
}

#[test]
fn repeated_linear_poles_reconstruct_exactly_with_units_and_overlaps() {
    for (numerator, factors) in [
        ("x+a", vec![("x-a", 3), ("x-b", 2), ("x-c", 1)]),
        ("(x+a)/3", vec![("(2*y+1)*x+a", 3), ("(y+2)*x+b", 2)]),
        ("(x-a)*(x-b)", vec![("x-a", 2), ("-2*x+2*a", 2), ("x-b", 1)]),
        ("x+1", vec![("y*(x-a)", 2), ("(y+1)*(x-b)", 2), ("y-1", 3)]),
        ("1", vec![("x", 3), ("-3*(x+1)/2", 2), ("-5/7", 3)]),
        ("(x-a)^2", vec![("x-a", 4), ("x-a", 1)]),
        ("x+1", vec![("x-a", 3), ("y+1", 0)]),
    ] {
        let (input, variable) = rational_input(numerator, &factors);
        assert!(try_linear(&input, variable).is_some());
        assert_reconstructs(&input, variable, true);
    }
}

#[test]
fn nonlinear_improper_polynomial_and_zero_inputs_keep_exact_fallbacks() {
    for (numerator, factors) in [
        ("x+1", vec![("x^2+a", 2), ("x-b", 1)]),
        ("x^5+1", vec![("x-a", 2), ("x-b", 1)]),
        ("x^2+a", vec![("y+1", 2)]),
        ("0", vec![("x-a", 2)]),
        ("x+1", vec![]),
    ] {
        let (input, variable) = rational_input(numerator, &factors);
        assert_reconstructs(&input, variable, false);
    }
}

#[test]
fn scalar_metadata_is_not_dropped() {
    let (mut input, variable) = rational_input("x+a", &[("x-a", 2), ("x-b", 2)]);
    input.numer_coeff = (-3).into();
    input.denom_coeff = 5.into();
    assert_reconstructs(&input, variable, true);
}

#[test]
fn finite_field_residues_preserve_leading_units() {
    let (template, variable) = rational_input("x+a", &[("3*x+a", 3), ("5*x+b", 2)]);
    let field = Zp::new(17);
    let input = template.to_finite_field(&field);
    assert_reconstructs(&input, variable, true);
}

#[test]
fn algebraic_coefficients_remain_exact() {
    let field = AlgebraicExtension::new(parse!("z^2-2").to_polynomial::<_, u16>(&Q, None));
    let variables = parse!("x+y+unused")
        .to_polynomial::<_, u16>(&Q, None)
        .variables()
        .clone();
    let x = parse!("x").to_polynomial::<_, u16>(&field, Some(variables.clone()));
    let variable = x
        .exponents_iter()
        .next()
        .unwrap()
        .iter()
        .position(|exponent| *exponent == 1)
        .unwrap();
    let first = (&x + &x.one()).mul_coeff(field.generator());
    let second = &x + &x.constant(field.nth(2.into()));
    let input = Factored::from_num_den(&x + &x.one(), vec![(first, 3), (second, 2)], &field, false);
    assert_reconstructs(&input, variable, true);
}

#[test]
fn wide_linear_power_is_never_materialized() {
    let (input, variable) = rational_input("x+1", &[("x+a+b+c+d+y+unused+1", 32)]);
    let terms = input.apart_factored_denominators(variable);
    assert_eq!(terms.len(), 2);
    assert!(terms.iter().all(|(_, denominator, power)| {
        denominator.numerator.nterms() == 8 && *power >= 31 && denominator.denominators.is_empty()
    }));
    for value in [2, 3, 5] {
        let values = vec![Q.nth(value.into()); input.numerator.nvars()];
        let evaluate = |f: &Factored<IntegerRing, u16>| {
            f.evaluate_with_coeff_map(|c| Q.to_element_numerator(c.clone()), &values, &Q)
        };
        let reconstructed = terms
            .iter()
            .fold(Q.zero(), |sum, (numerator, denominator, power)| {
                sum + Q.div(
                    &evaluate(numerator),
                    &Q.pow(&evaluate(denominator), *power as u64),
                )
            });
        assert_eq!(reconstructed, evaluate(&input));
    }
}

#[test]
fn wide_parameter_leading_powers_remain_factored_in_monic_residues() {
    let (input, variable) = rational_input("x+1", &[("(a+b+c+d+y+unused+1)*x+1", 32)]);
    let terms = input.apart_factored_denominators(variable);
    assert_eq!(terms.len(), 2);
    for (numerator, denominator, power) in &terms {
        assert!(*power >= 31);
        assert!(numerator.numerator.nterms() <= 7);
        assert_eq!(numerator.denominators.len(), 1);
        assert_eq!(numerator.denominators[0].0.nterms(), 7);
        assert!([32, 33].contains(&numerator.denominators[0].1));
        assert_eq!(denominator.denominators.len(), 1);
        assert_eq!(denominator.denominators[0].1, 1);
    }
    for value in [2, 3] {
        let values = vec![Q.nth(value.into()); input.numerator.nvars()];
        let evaluate = |f: &Factored<IntegerRing, u16>| {
            f.evaluate_with_coeff_map(|c| Q.to_element_numerator(c.clone()), &values, &Q)
        };
        let reconstructed = terms
            .iter()
            .fold(Q.zero(), |sum, (numerator, denominator, power)| {
                sum + Q.div(
                    &evaluate(numerator),
                    &Q.pow(&evaluate(denominator), *power as u64),
                )
            });
        assert_eq!(reconstructed, evaluate(&input));
    }
}

#[test]
#[should_panic(expected = "partial fraction variable out of range")]
fn invalid_variable_is_checked_even_for_zero() {
    let (input, _) = rational_input("0", &[]);
    input.apart_factored_denominators(input.numerator.nvars());
}
