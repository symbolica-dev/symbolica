//! Exact Q-polynomial division through a primitive integer divisor.
//!
//! This plan serves the coefficient-domain `try_div_exact` API. Ring-generic
//! `try_div` remains the unmodified fallback and does not call this dispatch.

use crate::domains::integer::{Integer, IntegerRing, Z};
use crate::domains::rational::{Q, Rational, RationalField};
use crate::domains::{EuclideanDomain, Ring, RingOps};
use crate::poly::PositiveExponent;
use crate::poly::polynomial::MultivariatePolynomial;

// Clearing coefficient denominators has a linear setup cost. Keep short
// speculative divisions on the ordinary heap path, and bound the coefficient
// growth caused by large collections of unrelated rational denominators.
const MIN_DIVIDEND_TERMS: usize = 32;
const MIN_DIVISOR_TERMS: usize = 4;
const MAX_COMMON_DENOMINATOR_BITS: u64 = 4096;

pub(super) fn try_div_exact<E: PositiveExponent>(
    dividend: &MultivariatePolynomial<RationalField, E>,
    divisor: &MultivariatePolynomial<RationalField, E>,
) -> Option<MultivariatePolynomial<RationalField, E>> {
    if dividend.nterms() < MIN_DIVIDEND_TERMS || divisor.nterms() < MIN_DIVISOR_TERMS {
        return dividend.try_div(divisor);
    }
    if dividend.variables() != divisor.variables() {
        let mut dividend = dividend.clone();
        let mut divisor = divisor.clone();
        dividend.unify_variables(&mut divisor);
        return try_div_exact(&dividend, &divisor);
    }
    if (0..dividend.nvars()).any(|variable| dividend.degree(variable) < divisor.degree(variable)) {
        return None;
    }

    let Some(dividend_denominator) = common_denominator(dividend) else {
        return dividend.try_div(divisor);
    };
    let Some(divisor_denominator) = common_denominator(divisor) else {
        return dividend.try_div(divisor);
    };
    let dividend_integer = clear_denominator(dividend, &dividend_denominator);
    let divisor_integer = clear_denominator(divisor, &divisor_denominator);
    let divisor_content = divisor_integer.content();
    let divisor_primitive = if divisor_content.is_one() {
        divisor_integer
    } else {
        divisor_integer.div_coeff(&divisor_content)
    };

    // Gauss's lemma: a primitive Z-polynomial divides an integer polynomial
    // over Q exactly when it divides it over Z. Removing the divisor content
    // is essential: (x+1)/(2*x+2) is divisible over Q, but not over Z.
    let quotient = dividend_integer.try_div(&divisor_primitive)?;
    let scale = Rational::from((divisor_denominator, dividend_denominator * &divisor_content));
    Some(if scale.is_integer() {
        quotient.map_coeff(
            |coefficient| Q.to_element_numerator(coefficient * scale.numerator_ref()),
            Q,
        )
    } else {
        quotient.map_coeff(
            |coefficient| Q.mul(&Q.to_element_numerator(coefficient.clone()), &scale),
            Q,
        )
    })
}

fn common_denominator<E: PositiveExponent>(
    polynomial: &MultivariatePolynomial<RationalField, E>,
) -> Option<Integer> {
    let mut denominator = Integer::one();
    for coefficient in &polynomial.coefficients {
        let next = coefficient.denominator_ref();
        if next.is_one() || next == &denominator {
            continue;
        }
        let gcd = Z.gcd(&denominator, next);
        denominator = Z.exact_div_owned(denominator, &gcd) * next;
        if denominator.significant_bits() > MAX_COMMON_DENOMINATOR_BITS {
            return None;
        }
    }
    Some(denominator)
}

fn clear_denominator<E: PositiveExponent>(
    polynomial: &MultivariatePolynomial<RationalField, E>,
    denominator: &Integer,
) -> MultivariatePolynomial<IntegerRing, E> {
    polynomial.map_coeff(
        |coefficient| {
            if denominator.is_one() {
                coefficient.numerator_ref().clone()
            } else {
                coefficient.numerator_ref()
                    * &Z.exact_div_owned(denominator.clone(), coefficient.denominator_ref())
            }
        },
        Z,
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::atom::Atom;
    use crate::poly::PolyVariable;
    use crate::prelude::{AtomCore, ParseSettings, Symbol};

    fn polynomial(expression: &str) -> MultivariatePolynomial<RationalField, u16> {
        let variables = Arc::new(
            ["x", "y", "z"]
                .map(|name| PolyVariable::Symbol(Symbol::parse(name, "rational_division").unwrap()))
                .to_vec(),
        );
        Atom::parse(expression, "rational_division", ParseSettings::default())
            .unwrap()
            .to_polynomial(&Q, Some(variables))
    }

    fn dense(seed: usize, denominator: usize) -> MultivariatePolynomial<RationalField, u16> {
        let terms = (0..=4)
            .flat_map(|x| {
                (0..=4 - x).flat_map(move |y| {
                    (0..=4 - x - y).map(move |z| {
                        let numerator = (1 + (17 * x + 11 * y + 5 * z + seed) % 29) as i64;
                        let numerator = if (x + y + z + seed) % 3 == 0 {
                            -numerator
                        } else {
                            numerator
                        };
                        // Distinct denominators, rather than merely one
                        // common scalar denominator for the whole input.
                        let denominator = 1 + (x + 3 * y + 7 * z + denominator) % 11;
                        format!("({numerator}/{denominator})*x^{x}*y^{y}*z^{z}")
                    })
                })
            })
            .collect::<Vec<_>>()
            .join("+");
        polynomial(&terms)
    }

    #[test]
    fn mixed_denominators_and_signed_content_reconstruct_exactly() {
        for seed in 0..8 {
            let quotient = dense(seed, 3).mul_coeff(Rational::from((-7, 13)));
            let divisor = dense(seed + 1, 5).mul_coeff(Rational::from((17, 19)));
            let dividend = &quotient * &divisor;
            assert_eq!(try_div_exact(&dividend, &divisor), Some(quotient.clone()));
            assert_eq!(dividend.try_div_exact(&divisor), Some(quotient));
            assert_eq!(
                try_div_exact(&dividend, &divisor),
                dividend.try_div(&divisor)
            );
        }
    }

    #[test]
    fn primitive_divisor_allows_fractional_quotient_content() {
        let divisor = dense(7, 0).mul_coeff(Rational::from(210));
        let expected = dense(2, 6).mul_coeff(Rational::from((-1, 330)));
        let dividend = &divisor * &expected;
        assert_eq!(try_div_exact(&dividend, &divisor), Some(expected));
    }

    #[test]
    fn nondivision_trivial_cases_and_large_denominator_fallback() {
        let divisor = dense(1, 5);
        let dividend = &divisor * &dense(3, 2) + polynomial("1");
        assert_eq!(try_div_exact(&dividend, &divisor), None);
        let zero = divisor.zero();
        assert_eq!(try_div_exact(&divisor, &zero), None);
        assert_eq!(try_div_exact(&zero, &divisor), Some(zero.clone()));
        assert_eq!(try_div_exact(&zero, &zero), None);
        let constant = polynomial("-7/11");
        assert_eq!(
            try_div_exact(&divisor, &constant),
            divisor.try_div(&constant)
        );

        let huge = Integer::from(2).pow(MAX_COMMON_DENOMINATOR_BITS + 1) + Integer::one();
        let divisor = dense(1, 5).mul_coeff(Rational::from((Integer::one(), huge)));
        let dividend = &divisor * &dense(3, 2);
        assert!(common_denominator(&divisor).is_none());
        assert_eq!(
            try_div_exact(&dividend, &divisor),
            dividend.try_div(&divisor)
        );
    }

    #[test]
    fn context_unification_and_sparse_high_degrees_match_generic_division() {
        let divisor = dense(1, 0);
        let quotient = dense(2, 1);
        let dividend = &divisor * &quotient;
        let mut variables = divisor.variables().as_ref().clone();
        variables.reverse();
        variables.push(PolyVariable::Symbol(
            Symbol::parse("unused", "rational_division").unwrap(),
        ));
        let divisor = divisor.rearrange_with_growth(&variables).unwrap();
        assert_eq!(
            try_div_exact(&dividend, &divisor),
            dividend.try_div(&divisor)
        );

        let divisor = polynomial("x^1024 + 2*y^512 + 3*z^256 + 7");
        let dividend = &divisor * &polynomial("x^500 + y^100 + z^50 + 1");
        assert_eq!(
            try_div_exact(&dividend, &divisor),
            dividend.try_div(&divisor)
        );
    }
}
