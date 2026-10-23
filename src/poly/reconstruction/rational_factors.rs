//! Small-factor hypotheses shared across coefficient-lifting primes.
use super::*;
use crate::domains::{
    finite_field::ToFiniteField,
    integer::{Integer, IntegerRing, Z},
    rational::{Q, Rational},
};

type IntegerPolynomial = MultivariatePolynomial<IntegerRing, u16>;

pub(super) struct Factors(pub [IntegerPolynomial; 2]);

impl Factors {
    pub(super) fn discover(image: &Fraction) -> Option<Self> {
        let total = image.numerator.nterms() + image.denominator.nterms();
        if image.numerator.nvars() < 2 || total < 32 {
            return None;
        }
        let field = image.numerator.ring();
        let modulus = Integer::from(field.get_prime());
        let template = IntegerPolynomial::new(&Z, None, image.numerator.variables().clone());
        let mut factors = [template.one(), template.one()];
        for (side, polynomial) in [&image.numerator, &image.denominator]
            .into_iter()
            .enumerate()
        {
            for variable in 0..polynomial.nvars() {
                let content = polynomial.multivariate_content(variable);
                // Monomial shifts already cost very little during interpolation.
                if content.nterms() < 2 {
                    continue;
                }
                let mut lifted =
                    MultivariatePolynomial::new(&Q, None, polynomial.variables().clone());
                let mut small = true;
                for term in &content {
                    let residue = Integer::from(field.from_element(term.coefficient));
                    let Ok(coefficient) =
                        Rational::maximal_quotient_reconstruction(&residue, &modulus, None)
                    else {
                        small = false;
                        break;
                    };
                    if coefficient.numerator_ref().abs() * coefficient.denominator_ref()
                        >= 1u64 << 32
                    {
                        small = false;
                        break;
                    }
                    lifted.append_monomial(coefficient, term.exponents);
                }
                if small {
                    // Any fixed nonzero scalar multiple defines an equivalent
                    // factor-removal transform; use primitive integer factors.
                    let denominator = lifted.one();
                    let factor =
                        RationalPolynomial::from_num_den(lifted, denominator, &Z, true).numerator;
                    factors[side] = &factors[side] * &factor;
                }
            }
        }
        if factors.iter().all(|factor| factor.is_one()) {
            return None;
        }
        let candidate = Self(factors);
        let reduced = candidate.reduce(image)?;
        // Pay for a changed lifting representation only for substantial support
        // compression. This compares learned images, never source information.
        (4 * (reduced.numerator.nterms() + reduced.denominator.nterms()) <= 3 * total)
            .then_some(candidate)
    }

    pub(super) fn modular(&self, field: &Zp64) -> [Polynomial; 2] {
        self.0
            .each_ref()
            .map(|p| p.map_coeff(|c| c.to_finite_field(field), field.clone()))
    }

    pub(super) fn reduce(&self, image: &Fraction) -> Option<Fraction> {
        let field = image.numerator.ring();
        let [numerator, denominator] = self.modular(field);
        if numerator.is_zero() || denominator.is_zero() {
            return None;
        }
        let numerator = image.numerator.try_div(&numerator)?;
        let denominator = image.denominator.try_div(&denominator)?;
        Some(Fraction::from_num_den(numerator, denominator, field, true))
    }

    pub(super) fn restore(
        &self,
        numerator: &mut MultivariatePolynomial<crate::domains::rational::RationalField, u16>,
        denominator: &mut MultivariatePolynomial<crate::domains::rational::RationalField, u16>,
    ) {
        *numerator = &*numerator * &self.0[0].map_coeff(|c| Rational::from(c.clone()), Q);
        *denominator = &*denominator * &self.0[1].map_coeff(|c| Rational::from(c.clone()), Q);
    }
}
