use super::*;
use crate::{atom::AtomCore, parse};

type NumberFieldPolynomial = MultivariatePolynomial<AlgebraicExtension<RationalField>, u16>;

fn pair(quotient: &str, divisor: &str) -> (NumberFieldPolynomial, NumberFieldPolynomial) {
    let polynomial = |expression| {
        crate::atom::Atom::parse(
            expression,
            "division_kernel",
            crate::parser::ParseSettings::default(),
        )
        .unwrap()
        .to_polynomial::<_, u16>(&Q, None)
    };
    let mut quotient = polynomial(quotient);
    let mut divisor = polynomial(divisor);
    quotient.unify_variables(&mut divisor);
    let field = AlgebraicExtension::new(polynomial("a^3+a/2-2"));
    (
        quotient.to_number_field(&field),
        divisor.to_number_field(&field),
    )
}

#[test]
fn polynomial_division_rejects_different_coefficient_rings() {
    let polynomial = |expression: &str| parse!(expression).to_polynomial::<_, u16>(&Q, None);
    let left = polynomial("a").to_number_field(&AlgebraicExtension::new(polynomial("a^2-2")));
    let right = polynomial("a").to_number_field(&AlgebraicExtension::new(polynomial("a^2-3")));
    for left in [left.clone(), left.zero()] {
        for right in [right.clone(), right.one(), right.zero()] {
            assert!(std::panic::catch_unwind(|| left.try_div(&right)).is_err());
            assert!(std::panic::catch_unwind(|| left.clone().try_div_owned(&right)).is_err());
            assert!(std::panic::catch_unwind(|| left.quot_rem(&right, false)).is_err());
            assert!(
                std::panic::catch_unwind(|| left.clone().quot_rem_owned(&right, false)).is_err()
            );
        }
    }
}

#[test]
fn number_field_checked_division_dispatches_and_rejects_remainders() {
    let (quotient, divisor) = pair("(a/3)*x^3+(a^2+1)*x+2", "(a+1)*x^2+a/2*x+1");
    let dividend = &quotient * &divisor;
    assert_eq!(
        dividend.try_div_kernel(&divisor),
        DivisionAttempt::Quotient(quotient.clone())
    );
    assert_eq!(dividend.try_div(&divisor), Some(quotient.clone()));
    assert_eq!(
        dividend.clone().try_div_owned(&divisor),
        Some(quotient.clone())
    );
    let (reference, remainder) = dividend.quot_rem(&divisor, false);
    assert!(remainder.is_zero());
    assert_eq!(reference, quotient);

    let inexact = &dividend + &dividend.one();
    assert_eq!(
        inexact.try_div_kernel(&divisor),
        DivisionAttempt::NotDivisible
    );
    assert!(inexact.try_div(&divisor).is_none());
    assert!(inexact.try_div_owned(&divisor).is_none());
}

#[test]
fn checked_division_preserves_context_and_handles_zero_and_constants() {
    let (quotient, divisor) = pair("x^2+a", "a*x+1");
    let dividend = &quotient * &divisor;
    let mut variables = divisor.variables().as_ref().clone();
    variables.reverse();
    variables.insert(0, PolyVariable::Temporary(17));
    let reordered = divisor.rearrange_with_growth(&variables).unwrap();
    for result in [
        dividend.try_div(&reordered),
        dividend.clone().try_div_owned(&reordered),
    ] {
        let mut reconstructed = result.unwrap() * &reordered;
        let mut expected = dividend.clone();
        reconstructed.unify_variables(&mut expected);
        assert_eq!(reconstructed, expected);
    }
    let reordered_dividend = dividend.rearrange_with_growth(&variables).unwrap();
    assert!(matches!(
        reordered_dividend.try_div_kernel(&reordered),
        DivisionAttempt::Quotient(_)
    ));

    let zero = dividend.zero();
    assert_eq!(zero.try_div(&divisor), Some(zero.clone()));
    assert_eq!(zero.clone().try_div_owned(&divisor), Some(zero.clone()));
    for value in [&zero, &dividend] {
        assert!(value.try_div(&zero).is_none());
        assert!(value.clone().try_div_owned(&zero).is_none());
    }
    let constant = divisor.constant(divisor.lcoeff());
    let scaled = &quotient * &constant;
    assert_eq!(scaled.try_div(&constant), Some(quotient.clone()));
    assert_eq!(scaled.try_div_owned(&constant), Some(quotient));
    let scalar = MultivariatePolynomial::<_, u16>::new_zero(divisor.ring()).one();
    let two = &scalar + &scalar;
    assert_eq!(two.try_div_kernel(&two), DivisionAttempt::Unsupported);
    assert_eq!(two.try_div(&two), Some(scalar));
}

#[test]
fn checked_division_falls_back_for_multivariate_and_sparse_high_degrees() {
    for (quotient, divisor) in [
        pair("x+y+a", "a*x+y+1"),
        pair("x^1000+a", "a*x+1"),
        pair("x^5000+a", "a*x+1"),
    ] {
        let dividend = &quotient * &divisor;
        assert_eq!(
            dividend.try_div_kernel(&divisor),
            DivisionAttempt::Unsupported
        );
        assert_eq!(dividend.try_div(&divisor), Some(quotient.clone()));
        assert_eq!(dividend.try_div_owned(&divisor), Some(quotient));
    }
}

#[test]
fn checked_division_handles_signed_exponents() {
    let (quotient, divisor) = pair("x^2+a", "a*x+1");
    let signed = |polynomial: &NumberFieldPolynomial| {
        MultivariatePolynomial::from_parts(
            polynomial.coefficients.clone(),
            polynomial
                .exponents
                .iter()
                .map(|&exponent| exponent as i16)
                .collect(),
            polynomial.ring().clone(),
            polynomial.variables().clone(),
        )
    };
    let mut quotient = signed(&quotient);
    let divisor = signed(&divisor);
    let variable = (0..quotient.nvars())
        .find(|&v| quotient.degree(v) != 0)
        .unwrap();
    let mut shift = vec![0; quotient.nvars()];
    shift[variable] = -3;
    quotient = quotient.mul_exp(&shift);
    let dividend = &quotient * &divisor;
    assert_eq!(
        dividend.try_div_kernel(&divisor),
        DivisionAttempt::Unsupported
    );
    assert_eq!(dividend.try_div(&divisor), Some(quotient.clone()));
    assert_eq!(dividend.try_div_owned(&divisor), Some(quotient));
}

#[test]
fn nonunit_leading_coefficient_uses_generic_division() {
    let extension = AlgebraicExtension::new(parse!("a^2-2").to_polynomial(&Z, None));
    let variables = Arc::new(vec![PolyVariable::Temporary(0)]);
    let mut divisor = MultivariatePolynomial::<_, u16>::new(&extension, None, variables);
    divisor.append_monomial(extension.nth(2.into()), &[0]);
    divisor.append_monomial(extension.nth(2.into()), &[1]);
    let quotient = divisor.one().mul_exp(&[1]) + divisor.one();
    let dividend = &quotient * &divisor;
    assert_eq!(
        dividend.try_div_kernel(&divisor),
        DivisionAttempt::Unsupported
    );
    assert_eq!(dividend.try_div(&divisor), Some(quotient.clone()));
    assert_eq!(dividend.try_div_owned(&divisor), Some(quotient));
}

#[test]
fn heap_division_handles_chains_and_growing_quotients() {
    for nvars in [3, 8, 9] {
        let variables = Arc::new((0..nvars).map(PolyVariable::Temporary).collect());
        let zero = MultivariatePolynomial::<_, u16>::new(&Z, None, variables);
        for divisor_terms in [1, 2, 5, 17] {
            for quotient_terms in [1, 2, 4, 5, 6, 16, 17, 18, 40] {
                for sparse in [false, true] {
                    let make = |terms: usize, offset: usize| {
                        let mut p = zero.clone();
                        for i in 0..terms {
                            let mut powers = vec![0; nvars];
                            powers[0] = (i + 1) as u16;
                            powers[1] = if sparse {
                                ((i * 7 + offset) % 19) as u16
                            } else {
                                0
                            };
                            powers[nvars - 1] = if sparse {
                                ((i * i + offset) % 11) as u16
                            } else {
                                0
                            };
                            let coefficient = if i % 3 == 0 { -2 } else { 2 };
                            p.append_monomial(Integer::from(coefficient), &powers);
                        }
                        p
                    };
                    let divisor = make(divisor_terms, 0);
                    let quotient = make(quotient_terms, 1);
                    let product = &quotient * &divisor;
                    let remainder = zero.one();
                    for packed in [None, Some(true), Some(false)] {
                        if packed == Some(true) && nvars > 8 || packed == Some(false) && nvars > 4 {
                            continue;
                        }
                        let divide = |p: MultivariatePolynomial<IntegerRing, u16>, abort, exact| {
                            if let Some(pack_u8) = packed {
                                p.heap_division_packed_exp(&divisor, abort, pack_u8, exact, None)
                            } else {
                                p.heap_division(&divisor, abort, exact, None)
                            }
                        };
                        for exact in [false, true] {
                            assert_eq!(
                                divide(product.clone(), true, exact),
                                (quotient.clone(), zero.clone())
                            );
                        }
                        let inexact = &product + &remainder;
                        assert_eq!(
                            divide(inexact.clone(), false, false),
                            (quotient.clone(), remainder.clone())
                        );
                        assert!(!divide(inexact, true, false).1.is_zero());
                    }
                }
            }
        }
    }
}
