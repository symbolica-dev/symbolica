//! Connect special-function error estimates to expression normalization and evaluators.

use super::{
    Atom, AtomView, Coefficient, CoefficientView, Complex, ErrorPropagatingFloat, EvaluationInfo,
    Float, RealLike, Settable, atom_float_precision, nonnegative_integer_tag,
    tracked::{self, SpecialFunction},
};
use crate::domains::float::Magnitude;

type Tracked = ErrorPropagatingFloat<Float>;

fn invalid(prec: u32) -> Complex<Tracked> {
    let nan =
        Tracked::new_with_error_magnitude(Float::with_val(prec, f64::NAN), Magnitude::INFINITY);
    Complex::new(nan.clone(), nan)
}

fn numeric_tag(tag: AtomView, prec: u32) -> Option<Complex<Tracked>> {
    let AtomView::Num(number) = tag else {
        return None;
    };
    let coefficient = number.get_coeff_view();
    let value = coefficient.to_tracked_float(prec).ok()?;
    // Exactly representable symbolic orders remain fixed parameters. In
    // particular, artificial uncertainty would spoil integer-order Bessel
    // limits. Other rationals retain their conversion error.
    if matches!(
        coefficient,
        CoefficientView::Natural(..) | CoefficientView::Large(..)
    ) {
        let Coefficient::Complex(exact) = coefficient.to_owned() else {
            unreachable!()
        };
        let re_error = if value.re.value.try_to_rational().as_ref() == Some(&exact.re) {
            Magnitude::ZERO
        } else {
            value.re.get_error_magnitude()
        };
        let im_error = if value.im.value.try_to_rational().as_ref() == Some(&exact.im) {
            Magnitude::ZERO
        } else {
            value.im.get_error_magnitude()
        };
        Some(Complex::new(
            Tracked::new_with_error_magnitude(value.re.into_value(), re_error),
            Tracked::new_with_error_magnitude(value.im.into_value(), im_error),
        ))
    } else {
        Some(value)
    }
}

fn evaluate(
    mut function: SpecialFunction,
    tags: &[Atom],
    args: &[Complex<Tracked>],
) -> Complex<Tracked> {
    let prec = args
        .first()
        .map_or(53, |z| z.re.value.prec().max(z.im.value.prec()));
    let [arg] = args else {
        return invalid(prec);
    };
    let order = if matches!(function, SpecialFunction::Polygamma(_)) {
        let views: Vec<_> = tags.iter().map(Atom::as_view).collect();
        let Ok(n) = nonnegative_integer_tag("polygamma", &views) else {
            return invalid(prec);
        };
        function = SpecialFunction::Polygamma(n);
        None
    } else if let [tag] = tags {
        let Some(order) = numeric_tag(tag.as_view(), prec) else {
            return invalid(prec);
        };
        Some(order)
    } else if tags.is_empty() {
        None
    } else {
        return invalid(prec);
    };
    tracked::evaluate(function, order.as_ref(), arg).unwrap_or_else(|| invalid(prec))
}

fn real_result(value: Complex<Tracked>) -> Tracked {
    if value.im.is_exact_zero() {
        value.re
    } else {
        invalid(value.re.value.prec()).re
    }
}

fn from_f64(value: &ErrorPropagatingFloat<f64>) -> Tracked {
    Tracked::new_with_error_magnitude(
        Float::with_val(53, value.value),
        value.get_error_magnitude(),
    )
}

fn to_f64(value: Tracked) -> ErrorPropagatingFloat<f64> {
    let center = value.value.to_f64();
    // Include conversion loss, including underflow below the binary64 range.
    let conversion = if center.is_finite() && value.value.is_finite() {
        (value.value.clone() - Float::with_val(value.value.prec().max(53), center))
            .absolute_magnitude()
    } else {
        Magnitude::INFINITY
    };
    ErrorPropagatingFloat::new_with_error_magnitude(
        center,
        value.get_error_magnitude() + conversion,
    )
}

pub(super) fn register(info: EvaluationInfo, function: SpecialFunction) -> EvaluationInfo {
    info.register_tagged(move |tags| {
        let tags = tags.iter().map(|x| x.to_owned()).collect::<Vec<_>>();
        Box::new(move |args: &[Complex<Tracked>]| evaluate(function, &tags, args))
    })
    .register_tagged(move |tags| {
        let tags = tags.iter().map(|x| x.to_owned()).collect::<Vec<_>>();
        Box::new(move |args: &[Tracked]| {
            let args: Vec<_> = args
                .iter()
                .map(|x| Complex::new(x.clone(), Tracked::exact_zero(x.value.prec())))
                .collect();
            real_result(evaluate(function, &tags, &args))
        })
    })
    .register_tagged(move |tags| {
        let tags = tags.iter().map(|x| x.to_owned()).collect::<Vec<_>>();
        Box::new(move |args: &[Complex<ErrorPropagatingFloat<f64>>]| {
            let args: Vec<_> = args
                .iter()
                .map(|x| Complex::new(from_f64(&x.re), from_f64(&x.im)))
                .collect();
            let result = evaluate(function, &tags, &args);
            Complex::new(to_f64(result.re), to_f64(result.im))
        })
    })
    .register_tagged(move |tags| {
        let tags = tags.iter().map(|x| x.to_owned()).collect::<Vec<_>>();
        Box::new(move |args: &[ErrorPropagatingFloat<f64>]| {
            let args: Vec<_> = args
                .iter()
                .map(|x| Complex::new(from_f64(x), Tracked::exact_zero(53)))
                .collect();
            to_f64(real_result(evaluate(function, &tags, &args)))
        })
    })
}

pub(super) fn normalize(
    function: SpecialFunction,
    order: Option<AtomView>,
    arg: AtomView,
    out: &mut Settable<Atom>,
) -> bool {
    let Some(prec) = atom_float_precision(arg).or_else(|| order.and_then(atom_float_precision))
    else {
        return false;
    };
    let AtomView::Num(number) = arg else {
        return false;
    };
    let Ok(arg) = number.get_coeff_view().to_tracked_float(prec) else {
        return false;
    };
    let order = match order {
        Some(tag) => {
            let Some(order) = numeric_tag(tag, prec) else {
                return false;
            };
            Some(order)
        }
        None => None,
    };
    let Some(value) = tracked::evaluate(function, order.as_ref(), &arg) else {
        return false;
    };
    out.to_num(value);
    true
}
