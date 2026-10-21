//! Evaluatable real and imaginary projections that retain component precision.

use crate::{
    atom::{EvaluationInfo, Symbol},
    domains::float::{Complex, Float, RealBall, TrackedComplex},
    get_symbol,
    state::StateInitializer,
    symbol,
};

use super::SymbolCache;

static REAL_PART: SymbolCache<Symbol> =
    SymbolCache::new(|| get_symbol!("real_part").expect("real-part projection was initialized"));
static IMAGINARY_PART: SymbolCache<Symbol> = SymbolCache::new(|| {
    get_symbol!("imaginary_part").expect("imaginary-part projection was initialized")
});

#[cfg(not(doctest))]
crate::_inventory::submit! {
    StateInitializer::new("symbolica::complex_projections", || {
        let _ = symbol!("real_part"; Real; eval = evaluation_info(false));
        let _ = symbol!("imaginary_part"; Real; eval = evaluation_info(true));
    }, &["symbolica"])
}

/// The real-valued numerical function `real_part(z)`.
///
/// Evaluation selects the real component without reducing it to the precision
/// of the imaginary component. Tracked evaluation retains its absolute error
/// and error quality. Symbolic arguments remain unevaluated.
pub fn real_part() -> Symbol {
    *REAL_PART
}

/// The real-valued numerical function `imaginary_part(z)`.
///
/// The result is the coefficient of `i`, represented as a real value with a
/// structural zero imaginary component. Tracked evaluation retains the selected
/// component's absolute error and error quality.
pub fn imaginary_part() -> Symbol {
    *IMAGINARY_PART
}

fn evaluation_info(imaginary: bool) -> EvaluationInfo {
    EvaluationInfo::new()
        .register(move |args: &[Complex<Float>]| {
            let [value] = args else {
                let bits = args.first().map_or(53, |value| {
                    if imaginary {
                        value.im.prec()
                    } else {
                        value.re.prec()
                    }
                });
                return Complex::new(Float::with_val(bits, f64::NAN), Float::new(bits));
            };
            let selected = if imaginary { &value.im } else { &value.re };
            Complex::new(selected.clone(), Float::new(selected.prec()))
        })
        .register(move |args: &[Float]| {
            let [value] = args else {
                return Float::with_val(args.first().map_or(53, Float::prec), f64::NAN);
            };
            if imaginary {
                Float::new(value.prec())
            } else {
                value.clone()
            }
        })
        .register(move |args: &[Complex<f64>]| {
            let [value] = args else {
                return Complex::new(f64::NAN, 0.0);
            };
            Complex::new(if imaginary { value.im } else { value.re }, 0.0)
        })
        .register(move |args: &[f64]| {
            let [value] = args else {
                return f64::NAN;
            };
            if imaginary { 0.0 } else { *value }
        })
        .register_tracked(move |tags, args, _bits| {
            if !tags.is_empty() {
                return Err("real and imaginary projections do not accept symbolic tags".into());
            }
            let [value] = args else {
                return Err(
                    "real and imaginary projections require exactly one numerical argument".into(),
                );
            };
            if !value.is_finite()
                || value.ball.re.radius.is_negative()
                || value.ball.im.radius.is_negative()
            {
                return Err("projection argument has invalid or nonfinite uncertainty".into());
            }
            let selected = if imaginary {
                &value.ball.im
            } else {
                &value.ball.re
            };
            Ok(TrackedComplex::from_ball(
                Complex::new(
                    selected.clone(),
                    RealBall::exact(Float::new(selected.center.prec())),
                ),
                value.error_quality,
            ))
        })
}

#[cfg(test)]
mod tests;
