//! Direct numerical interpretation retaining source uncertainty and exact inputs.

use std::{collections::HashMap, hash::BuildHasher};

use crate::{
    atom::{AtomCore, AtomView, KeyLookup, Symbol},
    coefficient::CoefficientView,
    domains::{
        float::{Complex, Float, FloatLike, RealBall, SingleFloat, TrackedComplex},
        rational::Rational,
    },
    evaluate::EvaluationError,
    transcendental::tracked::{
        evaluate_builtin_constant_tracked, evaluate_builtin_tracked, pow_tracked,
    },
};

/// Evaluate an expression once at a fixed working precision, preserving absolute errors.
///
/// Unlike the optimized rational-coefficient evaluator, this interpreter retains
/// floating literal precision and distinguishes those literals from exact rationals.
/// Function callbacks must explicitly support tracked evaluation. A missing error
/// model is an error, never a zero-radius approximation.
pub fn evaluate_tracked<A: AtomCore + KeyLookup>(
    expression: AtomView<'_>,
    inputs: &HashMap<A, TrackedComplex, impl BuildHasher>,
    working_bits: u32,
) -> Result<TrackedComplex, EvaluationError> {
    if working_bits < 2 {
        return Err(failure(
            expression,
            "tracked evaluation requires at least two working bits",
        ));
    }
    evaluate(expression, inputs, &mut HashMap::new(), working_bits, 0)
}

fn failure(expression: AtomView<'_>, reason: impl Into<String>) -> EvaluationError {
    EvaluationError::EvaluationFailed {
        expression: expression.to_owned(),
        reason: reason.into(),
    }
}

fn checked(
    expression: AtomView<'_>,
    value: Result<TrackedComplex, String>,
) -> Result<TrackedComplex, EvaluationError> {
    let value = value.map_err(|reason| failure(expression, reason))?;
    if !value.is_finite() {
        return Err(failure(
            expression,
            "tracked numerical result is not finite",
        ));
    }
    if value.ball.re.radius.is_negative() || value.ball.im.radius.is_negative() {
        return Err(failure(
            expression,
            "tracked numerical uncertainty must be nonnegative",
        ));
    }
    Ok(value)
}

fn evaluate<'a, A: AtomCore + KeyLookup>(
    expression: AtomView<'a>,
    inputs: &HashMap<A, TrackedComplex, impl BuildHasher>,
    cache: &mut HashMap<AtomView<'a>, TrackedComplex>,
    bits: u32,
    depth: usize,
) -> Result<TrackedComplex, EvaluationError> {
    if depth > 1024 {
        return Err(failure(
            expression,
            "tracked expression recursion limit exceeded",
        ));
    }
    if let Some(value) = inputs.get::<[u8]>(expression.get_data()) {
        let mut value = checked(expression, Ok(value.clone()))?;
        // Arithmetic precision is fixed by this evaluation, independently of
        // the accuracy of a supplied source. Reprecision retains the enclosure.
        value.ball.set_precision(bits);
        return checked(expression, Ok(value));
    }
    if let Some(value) = cache.get(&expression) {
        return Ok(value.clone());
    }
    let value = match expression {
        AtomView::Num(number) => match number.get_coeff_view() {
            CoefficientView::Float(real, imaginary) => {
                let real = real.to_float();
                let imaginary = imaginary.to_float();
                let real_zero = real.is_zero();
                let imaginary_zero = imaginary.is_zero();
                let mut value = TrackedComplex::from_float(Complex::new(real, imaginary), bits);
                // Zero components in an Atom's numerical representation are
                // structural zeros. Uncertain zeros belong in tracked inputs;
                // checked export refuses to erase their uncertainty into Atoms.
                if real_zero {
                    value.ball.re = RealBall::exact(Float::new(bits));
                }
                if imaginary_zero {
                    value.ball.im = RealBall::exact(Float::new(bits));
                }
                value
            }
            _ => {
                let rational = Complex::<Rational>::try_from(expression).map_err(|error| {
                    EvaluationError::UnsupportedCoefficient {
                        coefficient: error.to_string(),
                    }
                })?;
                TrackedComplex::from_exact_rational(&rational, bits)
            }
        },
        AtomView::Var(variable) => {
            let symbol = variable.get_symbol();
            if let Some(info) = symbol.get_evaluation_info()
                && info.has_tracked_evaluator()
            {
                checked(expression, info.evaluate_tracked(&[], &[], bits))?
            } else if let Some(value) = evaluate_builtin_constant_tracked(symbol, &[], bits) {
                checked(expression, value)?
            } else {
                return Err(EvaluationError::MissingEvaluator {
                    expression: expression.to_owned(),
                    eval_type: "TrackedComplex (a tracked constant callback is required)".into(),
                });
            }
        }
        AtomView::Add(sum) => {
            let mut value = TrackedComplex::from_exact_rational(
                &Complex::new(Rational::from(0), Rational::from(0)),
                bits,
            );
            for term in sum.iter() {
                value = value.add(&evaluate(term, inputs, cache, bits, depth + 1)?);
            }
            value
        }
        AtomView::Mul(product) => {
            let mut value = TrackedComplex::from_exact_rational(
                &Complex::new(Rational::from(1), Rational::from(0)),
                bits,
            );
            for factor in product.iter() {
                value = value.mul(&evaluate(factor, inputs, cache, bits, depth + 1)?);
            }
            value
        }
        AtomView::Pow(power) => {
            let base = evaluate(power.get_base(), inputs, cache, bits, depth + 1)?;
            if let Ok(exponent) = i64::try_from(power.get_exp()) {
                checked(expression, base.powi(exponent))?
            } else {
                let exponent = evaluate(power.get_exp(), inputs, cache, bits, depth + 1)?;
                checked(expression, pow_tracked(&base, &exponent, bits))?
            }
        }
        AtomView::Fun(function) => {
            let symbol = function.get_symbol();
            if symbol == Symbol::IF {
                if function.get_nargs() != 3 {
                    return Err(EvaluationError::WrongNumberOfArguments {
                        function: symbol,
                        expected: 3,
                        actual: function.get_nargs(),
                    });
                }
                let mut arguments = function.iter();
                let condition =
                    evaluate(arguments.next().unwrap(), inputs, cache, bits, depth + 1)?;
                let yes = arguments.next().unwrap();
                let no = arguments.next().unwrap();
                let precision = condition.precision();
                let branch = if precision.real.exact_zero && precision.imaginary.exact_zero {
                    no
                } else if condition.ball.contains_zero() {
                    return Err(failure(expression, "conditional test is uncertain at zero"));
                } else {
                    yes
                };
                let mut value = evaluate(branch, inputs, cache, bits, depth + 1)?;
                value.error_quality = value.error_quality.combine(condition.error_quality);
                value
            } else {
                let info = symbol.get_evaluation_info();
                let tag_count = info.map_or(0, |info| info.get_tag_count());
                if function.get_nargs() < tag_count {
                    return Err(EvaluationError::WrongNumberOfArguments {
                        function: symbol,
                        expected: tag_count,
                        actual: function.get_nargs(),
                    });
                }
                let mut arguments = function.iter();
                let tags = arguments.by_ref().take(tag_count).collect::<Vec<_>>();
                let values = arguments
                    .map(|argument| evaluate(argument, inputs, cache, bits, depth + 1))
                    .collect::<Result<Vec<_>, _>>()?;
                if let Some(info) = info
                    && info.has_tracked_evaluator()
                {
                    checked(expression, info.evaluate_tracked(&tags, &values, bits))?
                } else if values.is_empty()
                    && let Some(value) = evaluate_builtin_constant_tracked(symbol, &tags, bits)
                {
                    checked(expression, value)?
                } else if let Some(value) = evaluate_builtin_tracked(symbol, &tags, &values, bits) {
                    checked(expression, value)?
                } else {
                    return Err(EvaluationError::MissingEvaluator {
                        expression: expression.to_owned(),
                        eval_type: "TrackedComplex (a tracked function callback is required)"
                            .into(),
                    });
                }
            }
        }
    };
    let value = checked(expression, Ok(value))?;
    cache.insert(expression, value.clone());
    Ok(value)
}

#[cfg(test)]
mod review_tests;
#[cfg(test)]
mod tests;
