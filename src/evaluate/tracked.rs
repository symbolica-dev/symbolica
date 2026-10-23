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
    evaluate(expression, inputs, &mut HashMap::new(), working_bits)
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
    let value = match value {
        Ok(value) => value,
        Err(reason) => {
            crate::utils::resources::checkpoint().map_err(EvaluationError::ResourceLimit)?;
            return Err(failure(expression, reason));
        }
    };
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

enum Task<'a> {
    Evaluate(AtomView<'a>),
    Combine(AtomView<'a>),
    Conditional(AtomView<'a>, AtomView<'a>, AtomView<'a>, AtomView<'a>),
    Selected(
        AtomView<'a>,
        AtomView<'a>,
        crate::domains::float::ErrorQuality,
    ),
}

fn cached<'a, 'b>(
    cache: &'b HashMap<AtomView<'a>, TrackedComplex>,
    expression: AtomView<'a>,
) -> Result<&'b TrackedComplex, EvaluationError> {
    cache
        .get(&expression)
        .ok_or_else(|| failure(expression, "tracked interpreter dependency is absent"))
}

fn evaluate<'a, A: AtomCore + KeyLookup>(
    expression: AtomView<'a>,
    inputs: &HashMap<A, TrackedComplex, impl BuildHasher>,
    cache: &mut HashMap<AtomView<'a>, TrackedComplex>,
    bits: u32,
) -> Result<TrackedComplex, EvaluationError> {
    let mut pending = vec![Task::Evaluate(expression)];
    while let Some(task) = pending.pop() {
        crate::utils::resources::checkpoint().map_err(EvaluationError::ResourceLimit)?;
        match task {
            Task::Evaluate(current) => {
                if cache.contains_key(&current) {
                    continue;
                }
                if let Some(input) = inputs.get::<[u8]>(current.get_data()) {
                    let mut input = checked(current, Ok(input.clone()))?;
                    input.ball.set_precision(bits);
                    cache.insert(current, checked(current, Ok(input))?);
                    continue;
                }
                if let AtomView::Fun(function) = current
                    && function.get_symbol() == Symbol::IF
                {
                    if function.get_nargs() != 3 {
                        return Err(EvaluationError::WrongNumberOfArguments {
                            function: Symbol::IF,
                            expected: 3,
                            actual: function.get_nargs(),
                        });
                    }
                    let mut arguments = function.iter();
                    let condition = arguments.next().unwrap();
                    pending.push(Task::Conditional(
                        current,
                        condition,
                        arguments.next().unwrap(),
                        arguments.next().unwrap(),
                    ));
                    pending.push(Task::Evaluate(condition));
                    continue;
                }
                let children = match current {
                    AtomView::Add(sum) => sum.iter().collect::<Vec<_>>(),
                    AtomView::Mul(product) => product.iter().collect(),
                    AtomView::Pow(power) => {
                        let mut children = vec![power.get_base()];
                        if i64::try_from(power.get_exp()).is_err() {
                            children.push(power.get_exp());
                        }
                        children
                    }
                    AtomView::Fun(function) => {
                        let symbol = function.get_symbol();
                        let tags = symbol
                            .get_evaluation_info()
                            .map_or(0, |info| info.get_tag_count());
                        if function.get_nargs() < tags {
                            return Err(EvaluationError::WrongNumberOfArguments {
                                function: symbol,
                                expected: tags,
                                actual: function.get_nargs(),
                            });
                        }
                        function.iter().skip(tags).collect()
                    }
                    AtomView::Num(_) | AtomView::Var(_) => Vec::new(),
                };
                pending.push(Task::Combine(current));
                pending.extend(children.into_iter().rev().map(Task::Evaluate));
            }
            Task::Combine(current) => {
                let result = evaluate_ready(current, cache, bits)?;
                cache.insert(current, result);
            }
            Task::Conditional(current, condition, yes, no) => {
                let condition = cached(cache, condition)?;
                let precision = condition.precision();
                let branch = if precision.real.exact_zero && precision.imaginary.exact_zero {
                    no
                } else if condition.ball.contains_zero() {
                    return Err(failure(current, "conditional test is uncertain at zero"));
                } else {
                    yes
                };
                pending.push(Task::Selected(current, branch, condition.error_quality));
                pending.push(Task::Evaluate(branch));
            }
            Task::Selected(current, branch, quality) => {
                let mut result = cached(cache, branch)?.clone();
                result.error_quality = result.error_quality.combine(quality);
                cache.insert(current, checked(current, Ok(result))?);
            }
        }
    }
    cached(cache, expression).cloned()
}

fn evaluate_ready<'a>(
    expression: AtomView<'a>,
    cache: &HashMap<AtomView<'a>, TrackedComplex>,
    bits: u32,
) -> Result<TrackedComplex, EvaluationError> {
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
                value = value.add(cached(cache, term)?);
            }
            value
        }
        AtomView::Mul(product) => {
            let mut value = TrackedComplex::from_exact_rational(
                &Complex::new(Rational::from(1), Rational::from(0)),
                bits,
            );
            for factor in product.iter() {
                value = value.mul(cached(cache, factor)?);
            }
            value
        }
        AtomView::Pow(power) => {
            let base = cached(cache, power.get_base())?.clone();
            if let Ok(exponent) = i64::try_from(power.get_exp()) {
                checked(expression, base.powi(exponent))?
            } else {
                let exponent = cached(cache, power.get_exp())?.clone();
                checked(expression, pow_tracked(&base, &exponent, bits))?
            }
        }
        AtomView::Fun(function) => {
            let symbol = function.get_symbol();
            if symbol == Symbol::IF {
                return Err(failure(
                    expression,
                    "tracked conditional continuation is absent",
                ));
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
                    .map(|argument| cached(cache, argument).cloned())
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
    checked(expression, Ok(value))
}

#[cfg(test)]
mod review_tests;
#[cfg(test)]
mod tests;
