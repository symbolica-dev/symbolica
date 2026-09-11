//! Exact rational polynomial division by clearing denominators and dividing
//! by the primitive part of the resulting integer divisor.

use crate::domains::integer::{Integer, IntegerRing, Z};
use crate::domains::rational::{Q, Rational, RationalField};
use crate::domains::{EuclideanDomain, Ring, RingOps};
use crate::kernels::DivisionAttempt;
use crate::poly::Exponent;
use crate::poly::polynomial::MultivariatePolynomial;

// Clearing coefficient denominators has a linear setup cost. Keep short
// speculative divisions on the ordinary heap path, and bound the coefficient
// growth caused by large collections of unrelated rational denominators.
pub(super) const MIN_DIVIDEND_TERMS: usize = 32;
pub(super) const MIN_DIVISOR_TERMS: usize = 4;
const MAX_COMMON_DENOMINATOR_BITS: u64 = 4096;

impl<E: Exponent> MultivariatePolynomial<RationalField, E> {
    pub(super) fn try_div_via_integers(&self, divisor: &Self) -> DivisionAttempt<Self> {
        use DivisionAttempt::{NotDivisible, Quotient, Unsupported};
        if divisor.is_zero() {
            return NotDivisible;
        }
        if self.is_zero() {
            return Quotient(self.clone());
        }
        if self.nterms() < MIN_DIVIDEND_TERMS || divisor.nterms() < MIN_DIVISOR_TERMS {
            return Unsupported;
        }
        if !self.is_polynomial() || !divisor.is_polynomial() {
            return Unsupported;
        }
        if self.variables() != divisor.variables() {
            let mut dividend = self.clone();
            let mut divisor = divisor.clone();
            dividend.unify_variables(&mut divisor);
            return dividend.try_div_via_integers(&divisor);
        }
        if (0..self.nvars()).any(|variable| self.degree(variable) < divisor.degree(variable)) {
            return NotDivisible;
        }

        let Some(dividend_denominator) = self.common_denominator(MAX_COMMON_DENOMINATOR_BITS)
        else {
            return Unsupported;
        };
        let Some(divisor_denominator) = divisor.common_denominator(MAX_COMMON_DENOMINATOR_BITS)
        else {
            return Unsupported;
        };
        let dividend_integer = self.clear_denominator(&dividend_denominator);
        let divisor_integer = divisor.clear_denominator(&divisor_denominator);
        let divisor_content = divisor_integer.content();
        let divisor_primitive = if divisor_content.is_one() {
            divisor_integer
        } else {
            divisor_integer.div_coeff(&divisor_content)
        };

        // Gauss's lemma: a primitive Z-polynomial divides an integer polynomial
        // over Q exactly when it divides it over Z. Removing the divisor content
        // is essential: (x+1)/(2*x+2) is divisible over Q, but not over Z.
        let Some(quotient) = dividend_integer.try_div(&divisor_primitive) else {
            return NotDivisible;
        };
        let scale = Rational::from((divisor_denominator, dividend_denominator * &divisor_content));
        Quotient(if scale.is_integer() {
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
}

impl<E: Exponent> MultivariatePolynomial<RationalField, E> {
    /// Return the least common multiple of the coefficient denominators, or
    /// `None` if it exceeds `max_bits` bits. The zero polynomial has denominator one.
    pub(super) fn common_denominator(&self, max_bits: u64) -> Option<Integer> {
        let mut denominator = Integer::one();
        if max_bits == 0 {
            return None;
        }
        for coefficient in &self.coefficients {
            let next = coefficient.denominator_ref();
            if next.is_one() || next == &denominator {
                continue;
            }
            let gcd = Z.gcd(&denominator, next);
            denominator = Z.exact_div_owned(denominator, &gcd) * next;
            if denominator.significant_bits() > max_bits {
                return None;
            }
        }
        Some(denominator)
    }

    /// Multiply all coefficients by `denominator` and return an integer polynomial.
    /// `denominator` must be a positive common multiple of the coefficient denominators.
    pub(super) fn clear_denominator(
        &self,
        denominator: &Integer,
    ) -> MultivariatePolynomial<IntegerRing, E> {
        self.map_coeff(
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
                        // Vary denominators to exercise LCM computation.
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
        }
    }

    #[test]
    fn primitive_divisor_allows_fractional_quotient_content() {
        let divisor = dense(7, 0).mul_coeff(Rational::from(210));
        let expected = dense(2, 6).mul_coeff(Rational::from((-1, 330)));
        let dividend = &divisor * &expected;
        assert_eq!(
            dividend.try_div_kernel(&divisor),
            DivisionAttempt::Quotient(expected.clone())
        );
        assert_eq!(dividend.try_div(&divisor), Some(expected));
    }

    #[test]
    fn nondivision_and_trivial_cases() {
        let divisor = dense(1, 5);
        let dividend = &divisor * &dense(3, 2) + polynomial("1");
        assert_eq!(
            dividend.try_div_kernel(&divisor),
            DivisionAttempt::NotDivisible
        );
        assert_eq!(dividend.try_div(&divisor), None);
        assert_eq!(dividend.try_div_owned(&divisor), None);
        let zero = divisor.zero();
        assert_eq!(divisor.try_div(&zero), None);
        assert_eq!(zero.try_div(&divisor), Some(zero.clone()));
        assert_eq!(zero.try_div(&zero), None);
        let constant = polynomial("-7/11");
        assert_eq!(
            divisor.try_div_kernel(&constant),
            DivisionAttempt::Unsupported
        );
        assert_eq!(
            divisor.try_div(&constant),
            Some(divisor.quot_rem(&constant, false).0)
        );
    }

    #[test]
    fn large_denominators_decline_and_fall_back_without_recursion() {
        let huge = Integer::from(2).pow(MAX_COMMON_DENOMINATOR_BITS + 1) + Integer::one();
        let divisor = dense(1, 5).mul_coeff(Rational::from((Integer::one(), huge)));
        let quotient = dense(3, 2);
        let dividend = &divisor * &quotient;
        assert!(
            divisor
                .common_denominator(MAX_COMMON_DENOMINATOR_BITS)
                .is_none()
        );
        assert_eq!(
            dividend.try_div_kernel(&divisor),
            DivisionAttempt::Unsupported
        );
        assert_eq!(dividend.try_div(&divisor), Some(quotient.clone()));
        assert_eq!(dividend.try_div_owned(&divisor), Some(quotient));
    }

    #[test]
    fn context_unification_and_sparse_high_degree_fallback() {
        let divisor = dense(1, 0);
        let quotient = dense(2, 1);
        let dividend = &divisor * &quotient;
        let mut variables = divisor.variables().as_ref().clone();
        variables.reverse();
        variables.push(PolyVariable::Symbol(
            Symbol::parse("unused", "rational_division").unwrap(),
        ));
        let reordered = divisor.rearrange_with_growth(&variables).unwrap();
        for result in [
            dividend.try_div(&reordered),
            dividend.clone().try_div_owned(&reordered),
        ] {
            let mut actual = result.unwrap();
            let mut expected = quotient.clone();
            actual.unify_variables(&mut expected);
            assert_eq!(actual, expected);
        }

        let divisor = polynomial("x^1024 + 2*y^512 + 3*z^256 + 7");
        let quotient = polynomial("(x+y+z+1)^3");
        let dividend = &divisor * &quotient;
        assert!(dividend.nterms() >= MIN_DIVIDEND_TERMS);
        assert_eq!(
            dividend.try_div_kernel(&divisor),
            DivisionAttempt::Unsupported
        );
        assert_eq!(dividend.try_div(&divisor), Some(quotient));
    }

    #[test]
    fn laurent_division_shifts_before_integer_reduction() {
        let divisor = dense(1, 0).map_exp(|exponent| i16::try_from(*exponent).unwrap());
        let quotient = dense(2, 1)
            .map_exp(|exponent| i16::try_from(*exponent).unwrap())
            .mul_exp(&[-2, 0, 0]);
        let dividend = &divisor * &quotient;
        assert_eq!(
            dividend.try_div_kernel(&divisor),
            DivisionAttempt::Unsupported
        );
        assert_eq!(dividend.try_div(&divisor), Some(quotient.clone()));
        assert_eq!(dividend.try_div_owned(&divisor), Some(quotient));
    }
}
