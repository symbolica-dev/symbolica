//! Partial fractions with factored rational-function Taylor coefficients.

use super::{FactorizedRationalPolynomial, FromNumeratorAndFactorizedDenominator};
use crate::domains::EuclideanDomain;
use crate::domains::rational_polynomial::{FromNumeratorAndDenominator, RationalPolynomial};
use crate::poly::{
    PositiveExponent, factor::Factorize, gcd::PolynomialGCD, polynomial::MultivariatePolynomial,
};

type Factored<R, E> = FactorizedRationalPolynomial<R, E>;
type Polynomial<R, E> = MultivariatePolynomial<R, E>;
type ApartTerm<R, E> = (Factored<R, E>, Factored<R, E>, usize);

mod linear;
use linear::try_linear;

impl<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent> Factored<R, E>
where
    RationalPolynomial<R, E>: FromNumeratorAndDenominator<R, R, E>,
{
    /// Materialize the denominator product and return a canonical rational polynomial.
    pub fn to_rational_polynomial(&self) -> RationalPolynomial<R, E> {
        let numerator = self.numerator.clone().mul_coeff(self.numer_coeff.clone());
        let denominator = self.denominators.iter().fold(
            self.numerator.constant(self.denom_coeff.clone()),
            |product, (base, power)| product * &base.pow(*power),
        );
        RationalPolynomial::from_num_den(numerator, denominator, self.numerator.ring(), true)
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent> Factored<R, E>
where
    Self: FromNumeratorAndFactorizedDenominator<R, R, E>,
    RationalPolynomial<R, E>: FromNumeratorAndDenominator<R, R, E>,
    Polynomial<R, E>: Factorize,
{
    /// Compute partial fractions in `var`, returning terms
    /// `(numerator, denominator, exponent)` for `numerator / denominator^exponent`.
    ///
    /// Proper products of linear denominator blocks use truncated Taylor series
    /// over factored rational functions. Neither the input's powered linear
    /// blocks nor their product is expanded. Coefficient denominators remain
    /// factored as well; call [`Self::to_rational_polynomial`] only when a
    /// canonical expanded coefficient is required.
    ///
    /// For proper all-linear inputs, terms have monic, unpowered rational linear blocks
    /// `var - pole` as denominators; their numerators are independent of `var`.
    /// Supplied duplicate, proportional,
    /// or overlapping blocks need not already be canonical. Nonlinear and
    /// improper inputs retain the general rational-polynomial decomposition.
    pub fn apart_factored_denominators(&self, var: usize) -> Vec<(Self, Self, usize)> {
        assert!(
            var < self.numerator.nvars(),
            "partial fraction variable out of range"
        );
        assert!(
            !self.numerator.ring().is_zero(&self.denom_coeff),
            "zero denominator"
        );
        assert!(
            self.denominators
                .iter()
                .all(|(base, power)| *power == 0 || !base.is_zero()),
            "zero denominator"
        );

        // Public fields and the nonfactoring constructor allow distinct maps.
        // Reconstruct without factorization to unify maps and scalar units;
        // the source variable remains at its original index in the numerator.
        let mut denominators = self.denominators.clone();
        denominators.push((self.numerator.constant(self.denom_coeff.clone()), 1));
        let input = Self::from_num_den(
            self.numerator.clone().mul_coeff(self.numer_coeff.clone()),
            denominators,
            self.numerator.ring(),
            false,
        );
        if let Some(terms) = try_linear(&input, var) {
            return terms;
        }

        input
            .to_rational_polynomial()
            .apart_factored_denominators(var)
            .into_iter()
            .map(|(numerator, denominator, exponent)| {
                let from_rational = |value: RationalPolynomial<R, E>| {
                    Self::from_num_den(
                        value.numerator,
                        vec![(value.denominator, 1)],
                        input.numerator.ring(),
                        true,
                    )
                };
                (
                    from_rational(numerator),
                    from_rational(denominator),
                    exponent,
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod tests;
