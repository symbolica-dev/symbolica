//! Checked polynomial division over algebraic extensions.

use super::{AlgebraicExtension, AlgebraicNumber, EuclideanDomain, Ring, RingOps};
use crate::kernels::{
    CheckedPolynomialDivisionKernels, DivisionAttempt, UnivariatePolynomialDivisionRequest,
};

// Synthetic division allocates a coefficient for every degree. Bound both the
// allocation and the amount of empty space scanned for sparse inputs.
const MAX_DIVIDEND_DEGREE: usize = 4096;
const MAX_DEGREE_PER_TERM: usize = 64;

impl<R: EuclideanDomain> AlgebraicExtension<R> {
    /// Invert by extended Euclid when the intermediate leading coefficients
    /// are units in the base ring. This applies to every nonzero coefficient
    /// over a field; other base rings can require a different division algorithm.
    fn try_inv_with_euclid(&self, value: &AlgebraicNumber<R>) -> Option<AlgebraicNumber<R>> {
        let mut previous = self.poly().clone();
        let mut current = value.poly.clone();
        let mut previous_bezout = previous.zero();
        let mut current_bezout = previous.one();

        while !current.is_zero() {
            let inverse = self.poly().ring().try_inv(&current.lcoeff())?;
            current = current.mul_coeff(inverse.clone());
            current_bezout = current_bezout.mul_coeff(inverse);
            let (quotient, remainder) = previous.quot_rem(&current, false);
            let next_bezout = previous_bezout - &quotient * &current_bezout;
            (previous, current) = (current, remainder);
            (previous_bezout, current_bezout) = (current_bezout, next_bezout);
        }

        previous.is_one().then_some(AlgebraicNumber {
            poly: previous_bezout,
        })
    }
}

impl<R: EuclideanDomain> CheckedPolynomialDivisionKernels<AlgebraicNumber<R>>
    for AlgebraicExtension<R>
{
    fn try_univariate_division(
        &self,
        request: UnivariatePolynomialDivisionRequest<'_, AlgebraicNumber<R>>,
    ) -> DivisionAttempt<Vec<(u32, AlgebraicNumber<R>)>> {
        use DivisionAttempt::{NotDivisible, Quotient, Unsupported};

        for (coefficients, exponents) in [
            (request.dividend_coefficients, request.dividend_exponents),
            (request.divisor_coefficients, request.divisor_exponents),
        ] {
            if coefficients.len() != exponents.len()
                || exponents.windows(2).any(|pair| pair[0] >= pair[1])
                || exponents
                    .last()
                    .is_some_and(|&degree| degree > i32::MAX as u32)
                || coefficients
                    .iter()
                    .any(|coefficient| self.is_zero(coefficient))
            {
                return Unsupported;
            }
        }
        let Some(&divisor_degree) = request.divisor_exponents.last() else {
            return NotDivisible;
        };
        let Some(&dividend_degree) = request.dividend_exponents.last() else {
            return Quotient(Vec::new());
        };
        if dividend_degree < divisor_degree {
            return NotDivisible;
        }
        if dividend_degree as usize > MAX_DIVIDEND_DEGREE
            || dividend_degree as usize
                > request
                    .dividend_coefficients
                    .len()
                    .saturating_mul(MAX_DEGREE_PER_TERM)
                    .max(128)
        {
            return Unsupported;
        }

        let leading = request.divisor_coefficients.last().unwrap();
        let inverse = if self.is_one(leading) {
            self.one()
        } else if let Some(inverse) = self.try_inv_with_euclid(leading) {
            inverse
        } else {
            // A nonunit leading coefficient can still divide every quotient
            // coefficient. Generic checked division handles that case.
            return Unsupported;
        };

        // Normalize the divisor once, divide by its monic associate, and rescale
        // the quotient after divisibility has been established.
        let rescale = !self.is_one(&inverse);
        let monic: Vec<_> = request.divisor_coefficients[..request.divisor_coefficients.len() - 1]
            .iter()
            .map(|coefficient| {
                if !rescale {
                    coefficient.clone()
                } else {
                    self.mul(coefficient, &inverse)
                }
            })
            .collect();
        let mut remainder = vec![self.zero(); dividend_degree as usize + 1];
        for (&degree, coefficient) in request
            .dividend_exponents
            .iter()
            .zip(request.dividend_coefficients)
        {
            remainder[degree as usize] = coefficient.clone();
        }
        let mut quotient = Vec::new();
        for degree in (divisor_degree..=dividend_degree).rev() {
            let coefficient = std::mem::replace(&mut remainder[degree as usize], self.zero());
            if self.is_zero(&coefficient) {
                continue;
            }
            let quotient_degree = degree - divisor_degree;
            for (&divisor_degree, divisor_coefficient) in
                request.divisor_exponents.iter().zip(&monic)
            {
                self.sub_mul_assign(
                    &mut remainder[(quotient_degree + divisor_degree) as usize],
                    divisor_coefficient,
                    &coefficient,
                );
            }
            quotient.push((quotient_degree, coefficient));
        }
        if remainder[..divisor_degree as usize]
            .iter()
            .any(|coefficient| !self.is_zero(coefficient))
        {
            return NotDivisible;
        }
        if rescale {
            for (_, coefficient) in &mut quotient {
                *coefficient = self.mul(&*coefficient, &inverse);
            }
        }
        quotient.reverse();
        Quotient(quotient)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{atom::AtomCore, domains::rational::Q, parse};

    #[test]
    fn checked_kernel_distinguishes_invalid_inputs_and_nondivision() {
        let field = AlgebraicExtension::new(parse!("a^2-2").to_polynomial(&Q, None));
        let coefficients = [field.one(), field.one()];
        let divide = |dividend_coefficients: &[_],
                      dividend_exponents: &[_],
                      divisor_coefficients: &[_],
                      divisor_exponents: &[_]| {
            field.try_univariate_division(UnivariatePolynomialDivisionRequest {
                dividend_coefficients,
                dividend_exponents,
                divisor_coefficients,
                divisor_exponents,
            })
        };
        assert_eq!(divide(&[], &[], &[], &[]), DivisionAttempt::NotDivisible);
        assert_eq!(
            divide(&coefficients, &[0, 1], &[], &[]),
            DivisionAttempt::NotDivisible
        );
        assert_eq!(
            divide(&[], &[], &coefficients, &[0, 1]),
            DivisionAttempt::Quotient(vec![])
        );
        assert_eq!(
            divide(&coefficients[..1], &[0], &coefficients, &[0, 1]),
            DivisionAttempt::NotDivisible
        );
        assert_eq!(
            divide(&coefficients, &[0], &coefficients, &[0, 1]),
            DivisionAttempt::Unsupported
        );
        assert_eq!(
            divide(&coefficients, &[1, 1], &coefficients, &[0, 1]),
            DivisionAttempt::Unsupported
        );
    }
}
