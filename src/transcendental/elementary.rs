use std::cmp::Ordering;

use crate::{
    atom::{Atom, AtomView},
    coefficient::CoefficientView,
    domains::float::{Complex, ErrorPropagatingFloat, FloatLike, Magnitude, Real, RealLike},
    utils::Settable,
};

#[derive(Clone, Copy)]
pub(super) enum Operator {
    Tan,
    Cot,
    Sec,
    Csc,
    Asin,
    Acos,
    Atan,
    Acot,
    Asec,
    Acsc,
    Sinh,
    Cosh,
    Tanh,
    Coth,
    Sech,
    Csch,
    Asinh,
    Acosh,
    Atanh,
    Acoth,
    Asech,
    Acsch,
}

impl Operator {
    fn evaluate<T: Real>(self, x: &T) -> T {
        match self {
            Self::Tan => x.tan(),
            Self::Cot => x.tan().inv(),
            Self::Sec => x.cos().inv(),
            Self::Csc => x.sin().inv(),
            Self::Asin => x.asin(),
            Self::Acos => x.acos(),
            Self::Atan => x.atan2(&x.one()),
            Self::Acot => x.inv().atan2(&x.one()),
            Self::Asec => x.inv().acos(),
            Self::Acsc => x.inv().asin(),
            Self::Sinh => x.sinh(),
            Self::Cosh => x.cosh(),
            Self::Tanh => x.tanh(),
            Self::Coth => x.tanh().inv(),
            Self::Sech => x.sech(),
            Self::Csch => x.csch(),
            Self::Asinh => x.asinh(),
            Self::Acosh => x.acosh(),
            Self::Atanh => x.atanh(),
            Self::Acoth => x.inv().atanh(),
            Self::Asech => x.inv().acosh(),
            Self::Acsch => x.inv().asinh(),
        }
    }

    fn is_real<T: Real + RealLike>(self, x: &ErrorPropagatingFloat<T>) -> bool {
        if !x.value.is_finite() {
            return false;
        }
        let error = x.get_error_magnitude();
        if error.is_zero() {
            let comparison = x.value.norm().real_cmp(&x.value.one());
            return match self {
                Self::Asin | Self::Acos | Self::Atanh => {
                    matches!(comparison, Some(Ordering::Less | Ordering::Equal))
                }
                Self::Asec | Self::Acsc | Self::Acoth => {
                    matches!(comparison, Some(Ordering::Greater | Ordering::Equal))
                }
                Self::Acosh => matches!(
                    x.value.real_cmp(&x.value.one()),
                    Some(Ordering::Greater | Ordering::Equal)
                ),
                Self::Asech => {
                    x.value.real_cmp(&x.value.zero()) == Some(Ordering::Greater)
                        && matches!(comparison, Some(Ordering::Less | Ordering::Equal))
                }
                _ => true,
            };
        }
        let one = Magnitude::ONE;
        match self {
            Self::Asin | Self::Acos | Self::Atanh => x.value.absolute_magnitude() + error <= one,
            Self::Asec | Self::Acsc | Self::Acoth => {
                x.value.absolute_magnitude_lower().sub_lower(error) >= one
            }
            Self::Acosh => {
                x.value.real_cmp(&x.value.zero()) == Some(Ordering::Greater)
                    && x.value.absolute_magnitude_lower().sub_lower(error) >= one
            }
            Self::Asech => {
                x.value.real_cmp(&x.value.zero()) == Some(Ordering::Greater)
                    && x.value.absolute_magnitude_lower() > error
                    && x.value.absolute_magnitude() + error <= one
            }
            _ => true,
        }
    }
}

pub(super) fn evaluate<T: Real + RealLike>(
    args: &[Complex<ErrorPropagatingFloat<T>>],
    operator: Operator,
) -> Complex<ErrorPropagatingFloat<T>> {
    if let (Operator::Atan, [x, y]) = (operator, args) {
        // The two-argument real form uses atan2(y, x), including its quadrant.
        // Complex arguments retain the existing atan(y / x) convention.
        return if x.im.is_exact_zero() && y.im.is_exact_zero() {
            Complex::new(y.re.atan2(&x.re), y.im.zero())
        } else {
            y.atan2(x)
        };
    }
    let [arg] = args else {
        panic!(
            "Wrong number of arguments for transcendental function: {}",
            args.len()
        );
    };
    // Evaluating directly in the real domain preserves exact imaginary zeros
    // and avoids artificial uncertainty from intermediate complex formulas.
    // Uncertainty crossing a real-domain boundary or nonzero imaginary
    // uncertainty must use the complex path, including at a zero center.
    if arg.im.is_exact_zero() && operator.is_real(&arg.re) {
        Complex::new(operator.evaluate(&arg.re), arg.im.zero())
    } else {
        operator.evaluate(arg)
    }
}

pub(super) fn normalize(args: &[AtomView], out: &mut Settable<Atom>, operator: Operator) -> bool {
    let Some(coefficients) = args
        .iter()
        .map(|arg| match arg {
            AtomView::Num(number) => Some(number.get_coeff_view()),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()
    else {
        return false;
    };
    // Exact arguments remain symbolic. With a floating-point argument, convert
    // rational companions at its precision and retain both component errors.
    let Some(precision) = coefficients
        .iter()
        .filter_map(|coefficient| match coefficient {
            CoefficientView::Float(re, im) => {
                Some(re.to_float().value.prec().max(im.to_float().value.prec()))
            }
            _ => None,
        })
        .max()
    else {
        return false;
    };
    let Ok(values) = coefficients
        .iter()
        .map(|coefficient| coefficient.to_tracked_float(precision))
        .collect::<Result<Vec<_>, _>>()
    else {
        return false;
    };
    out.to_num(evaluate(&values, operator));
    true
}
