use super::super::FromNumeratorAndFactorizedDenominator;
use super::{ApartTerm, Factored, Polynomial};
use crate::domains::EuclideanDomain;
use crate::poly::{PositiveExponent, factor::Factorize, gcd::PolynomialGCD};

struct LinearPole<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent> {
    constant: Polynomial<R, E>,
    leading: Polynomial<R, E>,
    point: Factored<R, E>,
    multiplicity: usize,
}

fn from_polynomial<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent>(
    polynomial: Polynomial<R, E>,
) -> Factored<R, E>
where
    Factored<R, E>: FromNumeratorAndFactorizedDenominator<R, R, E>,
{
    let ring = polynomial.ring().clone();
    Factored::from_num_den(polynomial, vec![], &ring, false)
}

fn multiply_shifted<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent>(
    coefficients: &mut [Factored<R, E>],
    point: &Factored<R, E>,
    degree: usize,
) -> usize
where
    Factored<R, E>: FromNumeratorAndFactorizedDenominator<R, R, E>,
{
    let degree = (degree + 1).min(coefficients.len() - 1);
    for index in (1..=degree).rev() {
        let scaled = &coefficients[index] * point;
        coefficients[index] = &scaled + &coefficients[index - 1];
    }
    coefficients[0] = &coefficients[0] * point;
    degree
}

fn shifted_numerator<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent>(
    coefficients: &[Polynomial<R, E>],
    point: &Factored<R, E>,
    terms: usize,
) -> Vec<Factored<R, E>>
where
    Factored<R, E>: FromNumeratorAndFactorizedDenominator<R, R, E>,
{
    let mut shifted = vec![from_polynomial(point.numerator.zero()); terms];
    let mut degree = 0;
    for (index, coefficient) in coefficients.iter().rev().enumerate() {
        if index != 0 {
            degree = multiply_shifted(&mut shifted, point, degree);
        }
        shifted[0] = &shifted[0] + &from_polynomial(coefficient.clone());
    }
    shifted
}

pub(super) fn try_linear<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent>(
    input: &Factored<R, E>,
    var: usize,
) -> Option<Vec<ApartTerm<R, E>>>
where
    Factored<R, E>: FromNumeratorAndFactorizedDenominator<R, R, E>,
    Polynomial<R, E>: Factorize,
{
    let one = from_polynomial(input.numerator.one());
    if input.is_zero() {
        return Some(vec![(from_polynomial(input.numerator.zero()), one, 1)]);
    }
    let mut total_degree = 0usize;
    for (base, power) in &input.denominators {
        if *power == 0 {
            continue;
        }
        match base.degree(var).to_i32() {
            0 => {}
            1 => {
                total_degree = total_degree
                    .checked_add(*power)
                    .expect("denominator degree overflow")
            }
            _ => return None,
        }
    }
    if total_degree == 0 {
        return Some(vec![(input.clone(), one, 1)]);
    }
    if input.numerator.degree(var).to_i32() as usize >= total_degree {
        return None;
    }

    let ring = input.numerator.ring();
    let mut coefficient_denominators =
        vec![(input.numerator.constant(input.denom_coeff.clone()), 1)];
    let mut poles: Vec<LinearPole<R, E>> = vec![];
    for (base, power) in &input.denominators {
        if *power == 0 {
            continue;
        }
        if base.degree(var) == E::zero() {
            coefficient_denominators.push((base.clone(), *power));
            continue;
        }
        let univariate = base.to_univariate(var);
        let constant = univariate.coefficients()[0].clone();
        let leading = univariate.coefficients()[1].clone();
        coefficient_denominators.push((leading.clone(), *power));

        // Compare roots by cross multiplication, not by a representation's
        // factor ordering or by an expensive canonical rational-function GCD.
        if let Some(pole) = poles
            .iter_mut()
            .find(|pole| &pole.constant * &leading == &constant * &pole.leading)
        {
            pole.multiplicity = pole
                .multiplicity
                .checked_add(*power)
                .expect("pole multiplicity overflow");
        } else {
            let point =
                Factored::from_num_den(-constant.clone(), vec![(leading.clone(), 1)], ring, true);
            poles.push(LinearPole {
                constant,
                leading,
                point,
                multiplicity: *power,
            });
        }
    }
    let inverse_scalar = Factored::from_num_den(
        input.numerator.constant(input.numer_coeff.clone()),
        coefficient_denominators,
        ring,
        true,
    );
    let numerator = input.numerator.to_univariate(var);
    let mut coordinate_exponents = vec![E::zero(); input.numerator.nvars()];
    coordinate_exponents[var] = E::one();
    let coordinate = from_polynomial(input.numerator.monomial(ring.one(), coordinate_exponents));
    let mut output = vec![];
    for (index, pole) in poles.iter().enumerate() {
        let terms = pole.multiplicity;
        let numerator_series = shifted_numerator(numerator.coefficients(), &pole.point, terms);
        let mut cofactor = vec![from_polynomial(input.numerator.zero()); terms];
        cofactor[0] = one.clone();
        let mut degree = 0;
        let mut inverse_constant = one.clone();
        for (other_index, other) in poles.iter().enumerate() {
            if index == other_index {
                continue;
            }
            let difference = &pole.point - &other.point;
            if difference.is_zero() {
                return None;
            }
            let inverse_difference = difference.clone().inv();
            for _ in 0..other.multiplicity {
                degree = multiply_shifted(&mut cofactor, &difference, degree);
                inverse_constant = &inverse_constant * &inverse_difference;
            }
        }

        let mut quotient: Vec<Factored<R, E>> = Vec::with_capacity(terms);
        for exponent in 0..terms {
            let mut coefficient = numerator_series[exponent].clone();
            for cofactor_exponent in 1..=exponent {
                if cofactor[cofactor_exponent].is_zero()
                    || quotient[exponent - cofactor_exponent].is_zero()
                {
                    continue;
                }
                coefficient = &coefficient
                    - &(&cofactor[cofactor_exponent] * &quotient[exponent - cofactor_exponent]);
            }
            quotient.push(&coefficient * &inverse_constant);
        }

        // A monic rational linear block avoids constructing or subsequently
        // undoing powers of a parameter-dependent leading coefficient.
        // Reuse the already-factored pole instead of factoring its leading
        // polynomial again for every output coefficient.
        let denominator = &coordinate - &pole.point;
        for (exponent, coefficient) in quotient.into_iter().enumerate() {
            if coefficient.is_zero() {
                continue;
            }
            let power = terms - exponent;
            let coefficient = &coefficient * &inverse_scalar;
            output.push((coefficient, denominator.clone(), power));
        }
    }
    Some(output)
}
