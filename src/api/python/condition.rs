//! Shared comparison, Condition, and numeric hashing helpers.
use super::*;

pub(super) fn relation(lhs: Pattern, rhs: Pattern, op: CompareOp) -> PythonCondition {
    let relation = match op {
        CompareOp::Eq => Relation::Eq(lhs, rhs),
        CompareOp::Ne => Relation::Ne(lhs, rhs),
        CompareOp::Lt => Relation::Lt(lhs, rhs),
        CompareOp::Le => Relation::Le(lhs, rhs),
        CompareOp::Gt => Relation::Gt(lhs, rhs),
        CompareOp::Ge => Relation::Ge(lhs, rhs),
    };
    PythonCondition {
        condition: relation.into(),
    }
}

/// Matching supplies wildcard bindings, whereas a Transformer supplies an input.
/// Validate the supported matcher operands before making an infallible predicate.
pub(super) fn matching_relation(value: Relation) -> Result<PatternRestriction, &'static str> {
    let (lhs, rhs) = match &value {
        Relation::Eq(a, b)
        | Relation::Ne(a, b)
        | Relation::Lt(a, b)
        | Relation::Le(a, b)
        | Relation::Gt(a, b)
        | Relation::Ge(a, b) => (a, b),
        _ => unreachable!(),
    };
    let mut wildcards = Vec::new();
    let mut valid = true;
    for operand in [lhs, rhs] {
        operand.visitor(&mut |p| match p {
            Pattern::Wildcard(name, _) => wildcards.push(*name),
            Pattern::Transformer(_) | Pattern::Alternative(_) => valid = false,
            Pattern::Fn(name, _) if name.get_wildcard_level() > 0 => valid = false,
            _ => {}
        });
    }
    if !valid {
        return Err(
            "Matching comparisons require expression operands with fixed function names. Transformers can be used in Transformer.if_then conditions.",
        );
    }
    Ok(PatternRestriction::MatchStack(Box::new(
        move |stack: &MatchStack| {
            if wildcards.iter().any(|name| stack.get(*name).is_none()) {
                return ConditionResult::Inconclusive;
            }
            let (a, b) = match &value {
                Relation::Eq(a, b)
                | Relation::Ne(a, b)
                | Relation::Lt(a, b)
                | Relation::Le(a, b)
                | Relation::Gt(a, b)
                | Relation::Ge(a, b) => (a, b),
                _ => unreachable!(),
            };
            // Validation above excludes the fallible transformer and function-name
            // substitution paths. All referenced wildcard values are available.
            let a = a.replace_wildcards_with_matches(stack);
            let b = b.replace_wildcards_with_matches(stack);
            match value {
                Relation::Eq(..) => crate::id::equal_atoms(a.as_view(), b.as_view()).into(),
                Relation::Ne(..) => (!crate::id::equal_atoms(a.as_view(), b.as_view())).into(),
                _ => {
                    let Some(order) = crate::id::compare_real_atoms(a.as_view(), b.as_view())
                    else {
                        return ConditionResult::Inconclusive;
                    };
                    match value {
                        Relation::Lt(..) => order.is_lt(),
                        Relation::Le(..) => order.is_le(),
                        Relation::Gt(..) => order.is_gt(),
                        Relation::Ge(..) => order.is_ge(),
                        _ => unreachable!(),
                    }
                    .into()
                }
            }
        },
    )))
}

fn rational_to_python<'py>(value: Rational, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    if value.is_integer() {
        return value.numerator().clone().into_bound_py_any(py);
    }
    py.import("fractions")?
        .getattr("Fraction")?
        .call1((value.numerator().clone(), value.denominator().clone()))
}

/// Exact numeric values are used for Python equality and hashing. In particular,
/// never round an arbitrary-precision number through f64 or its display string.
pub(super) fn numeric_parts<'py>(
    atom: AtomView<'_>,
    py: Python<'py>,
) -> PyResult<Option<(Bound<'py, PyAny>, Bound<'py, PyAny>)>> {
    if let Some(value) = crate::id::exact_numeric_value(atom) {
        return Ok(Some((
            rational_to_python(value.re, py)?,
            rational_to_python(value.im, py)?,
        )));
    }
    if let AtomView::Num(n) = atom
        && let Coefficient::Infinity(Some(phase)) = n.get_coeff_view().to_owned()
        && phase.im.is_zero()
    {
        return Ok(Some((
            if phase.re.is_negative() {
                f64::NEG_INFINITY
            } else {
                f64::INFINITY
            }
            .into_bound_py_any(py)?,
            0.into_bound_py_any(py)?,
        )));
    }
    Ok(None)
}

pub(super) fn expression_eq(
    lhs: &PythonExpression,
    rhs: &Bound<'_, PyAny>,
) -> PyResult<Option<bool>> {
    let py = rhs.py();
    let left = numeric_parts(lhs.expr.as_view(), py)?;
    if let Ok(other) = rhs.extract::<PythonExpression>() {
        if let (Some((lr, li)), Some((rr, ri))) = (left, numeric_parts(other.expr.as_view(), py)?) {
            return Ok(Some(lr.eq(rr)? && li.eq(ri)?));
        }
        return Ok(Some(lhs.expr == other.expr));
    }
    if !rhs.is_instance(&py.import("numbers")?.getattr("Number")?)?
        && !rhs.is_instance(&py.import("decimal")?.getattr("Decimal")?)?
    {
        return Ok(None);
    }
    let Some((lr, li)) = left else {
        return Ok(Some(false));
    };
    Ok(Some(
        lr.eq(rhs.getattr("real")?)? && li.eq(rhs.getattr("imag")?)?,
    ))
}

pub(super) fn expression_compare(
    lhs: &PythonExpression,
    rhs: &Bound<'_, PyAny>,
    op: CompareOp,
) -> PyResult<Py<PyAny>> {
    let py = rhs.py();
    if matches!(op, CompareOp::Eq | CompareOp::Ne) {
        return match expression_eq(lhs, rhs)? {
            Some(equal) => (if matches!(op, CompareOp::Eq) {
                equal
            } else {
                !equal
            })
            .into_py_any(py),
            None => Ok(py.NotImplemented()),
        };
    }
    let Ok(other) = rhs.extract::<ConvertibleToOpenPattern>() else {
        return Ok(py.NotImplemented());
    };
    relation(lhs.expr.to_pattern(), other.to_pattern()?.expr, op).into_py_any(py)
}

pub(super) fn expression_hash(atom: AtomView<'_>, py: Python<'_>) -> PyResult<isize> {
    if let Some((re, im)) = numeric_parts(atom, py)? {
        let multiplier: isize = py
            .import("sys")?
            .getattr("hash_info")?
            .getattr("imag")?
            .extract()?;
        let hash = re.hash()?.wrapping_add(multiplier.wrapping_mul(im.hash()?));
        return Ok(if hash == -1 { -2 } else { hash });
    }
    let mut hasher = ahash::AHasher::default();
    atom.hash(&mut hasher);
    let hash = hasher.finish() as isize;
    Ok(if hash == -1 { -2 } else { hash })
}
