use super::*;

/// A numeric domain supported by the expression evaluator.
///
/// It finds the best implementation to numericaly evaluate a function.
pub trait EvaluationDomain: Sized + 'static {
    /// The single supported binary precision for fixed-precision domains.
    ///
    /// Use `None` for variable-precision domains such as `Float` and `Complex<Float>`.
    const FIXED_PRECISION: Option<u32>;

    /// Try to convert a `Complex<Float>` to the type.
    fn try_from_complex_float(_: Complex<Float>) -> Result<Self, String> {
        Err(format!(
            "Cannot convert from Complex<Float> to {}",
            std::any::type_name::<Self>()
        ))
    }

    /// Resolve a function implementation for the given tags, if available for the type.
    /// If no implementation exists for the type, a conversion from a 'larger' type may
    /// be used instead.
    fn resolve_function(
        tags: &[AtomView],
        info: &EvaluationInfo,
    ) -> Option<Box<dyn ExternalFunction<Self>>> {
        info.get_evaluator(tags)
    }
}

impl EvaluationDomain for Complex<Rational> {
    const FIXED_PRECISION: Option<u32> = None;

    fn resolve_function(
        _: &[AtomView],
        _: &EvaluationInfo,
    ) -> Option<Box<dyn ExternalFunction<Self>>> {
        None
    }
}

/// Allow exact-coefficient evaluators to be mapped to real balls.
///
/// Map rational coefficients with `RealBall::from_rational_ball` and the
/// requested precision. The default conversion from an already rounded
/// `Complex<Float>` deliberately remains unsupported: it cannot enclose the
/// unknown rounding error of a mathematical constant. Only explicitly
/// registered ball callbacks are resolved, without a floating-point fallback.
/// Evaluator support does not make inherited transcendental ball operations
/// certifying; restrict certified computations to the documented ball algebra.
impl EvaluationDomain for crate::domains::float::RealBall {
    const FIXED_PRECISION: Option<u32> = None;
}

/// Complex counterpart of the real-ball domain, with the same restrictions on
/// coefficient conversion, callbacks and certified arithmetic.
impl EvaluationDomain for crate::domains::float::ComplexBall {
    const FIXED_PRECISION: Option<u32> = None;
}

impl EvaluationDomain for f64 {
    const FIXED_PRECISION: Option<u32> = Some(53);

    fn try_from_complex_float(f: Complex<Float>) -> Result<Self, String> {
        if f.is_real() {
            Ok(f.re.to_f64())
        } else {
            Err(format!(
                "Cannot convert from Complex<Float> to f64 because the result {f} is not real"
            ))
        }
    }
}

impl EvaluationDomain for F64 {
    const FIXED_PRECISION: Option<u32> = Some(53);

    fn try_from_complex_float(f: Complex<Float>) -> Result<Self, String> {
        if f.is_real() {
            Ok(f.re.to_f64().into())
        } else {
            Err(format!(
                "Cannot convert from Complex<Float> to f64 because the result {f} is not real"
            ))
        }
    }

    fn resolve_function(
        tags: &[AtomView],
        info: &EvaluationInfo,
    ) -> Option<Box<dyn ExternalFunction<Self>>> {
        if let Some(f) = info.get_evaluator::<F64>(tags) {
            return Some(f);
        }

        if let Some(f) = f64::resolve_function(tags, info) {
            Some(Box::new(move |args| {
                let args: &[f64] = unsafe { std::mem::transmute(args) };
                f(args).into()
            }))
        } else {
            None
        }
    }
}

impl EvaluationDomain for Complex<f64> {
    const FIXED_PRECISION: Option<u32> = Some(53);

    fn try_from_complex_float(f: Complex<Float>) -> Result<Self, String> {
        Ok(Complex::new(f.re.to_f64(), f.im.to_f64()))
    }
}

impl EvaluationDomain for wide::f64x4 {
    const FIXED_PRECISION: Option<u32> = Some(53);

    fn try_from_complex_float(f: Complex<Float>) -> Result<Self, String> {
        if f.is_real() {
            let r = f.re.to_f64();
            Ok(wide::f64x4::new([r; 4]))
        } else {
            Err(format!(
                "Cannot convert from Complex<Float> to f64 because the result {f} is not real"
            ))
        }
    }

    fn resolve_function(
        tags: &[AtomView],
        info: &EvaluationInfo,
    ) -> Option<Box<dyn ExternalFunction<Self>>> {
        if let Some(f) = info.get_evaluator::<wide::f64x4>(tags) {
            return Some(f);
        }

        // create a vectorized version of the scalar function if it exists
        if let Some(f) = f64::resolve_function(tags, info) {
            Some(Box::new(move |args| {
                let mut buffer =
                    smallvec::SmallVec::<[f64; 4]>::from_iter(std::iter::repeat_n(0., args.len()));
                let mut res = [0.; 4];

                for i in 0..4 {
                    for (b, v) in buffer.iter_mut().zip(args) {
                        *b = v.as_array()[i];
                    }

                    res[i] = f(&buffer);
                }

                res.into()
            }))
        } else {
            None
        }
    }
}

impl EvaluationDomain for wide::f64x8 {
    const FIXED_PRECISION: Option<u32> = Some(53);

    fn try_from_complex_float(f: Complex<Float>) -> Result<Self, String> {
        if f.is_real() {
            let r = f.re.to_f64();
            Ok(wide::f64x8::new([r; 8]))
        } else {
            Err(format!(
                "Cannot convert from Complex<Float> to f64 because the result {f} is not real"
            ))
        }
    }

    fn resolve_function(
        tags: &[AtomView],
        info: &EvaluationInfo,
    ) -> Option<Box<dyn ExternalFunction<Self>>> {
        if let Some(f) = info.get_evaluator::<wide::f64x8>(tags) {
            return Some(f);
        }

        // create a vectorized version of the scalar function if it exists
        if let Some(f) = f64::resolve_function(tags, info) {
            Some(Box::new(move |args| {
                let mut buffer =
                    smallvec::SmallVec::<[f64; 4]>::from_iter(std::iter::repeat_n(0., args.len()));
                let mut res = [0.; 8];

                for i in 0..8 {
                    for (b, v) in buffer.iter_mut().zip(args) {
                        *b = v.as_array()[i];
                    }

                    res[i] = f(&buffer);
                }

                res.into()
            }))
        } else {
            None
        }
    }
}

impl EvaluationDomain for DoubleFloat {
    const FIXED_PRECISION: Option<u32> = Some(106);

    fn try_from_complex_float(f: Complex<Float>) -> Result<Self, String> {
        if f.is_real() {
            Ok(f.re.to_double_float())
        } else {
            Err(format!(
                "Cannot convert from Complex<Float> to DoubleFloat because the result {f} is not real"
            ))
        }
    }

    fn resolve_function(
        tags: &[AtomView],
        info: &EvaluationInfo,
    ) -> Option<Box<dyn ExternalFunction<Self>>> {
        if let Some(f) = info.get_evaluator::<DoubleFloat>(tags) {
            return Some(f);
        }

        // use arbitrary precision evaluation if double-float implementation is missing
        if let Some(f) = Float::resolve_function(tags, info) {
            Some(Box::new(move |args| {
                let args_float = args
                    .iter()
                    .map(|x| x.into())
                    .collect::<smallvec::SmallVec<[Float; 4]>>();
                f(&args_float).to_double_float()
            }))
        } else {
            None
        }
    }
}

impl EvaluationDomain for Complex<DoubleFloat> {
    const FIXED_PRECISION: Option<u32> = Some(106);

    fn try_from_complex_float(f: Complex<Float>) -> Result<Self, String> {
        Ok(f.to_double_float())
    }

    fn resolve_function(
        tags: &[AtomView],
        info: &EvaluationInfo,
    ) -> Option<Box<dyn ExternalFunction<Self>>> {
        if let Some(f) = info.get_evaluator::<Complex<DoubleFloat>>(tags) {
            return Some(f);
        }

        // use arbitrary precision evaluation if double-float implementation is missing
        if let Some(f) = Complex::<Float>::resolve_function(tags, info) {
            Some(Box::new(move |args| {
                let args_float = args
                    .iter()
                    .map(|x| Complex::new(x.re.into(), x.im.into()))
                    .collect::<smallvec::SmallVec<[Complex<Float>; 4]>>();
                let r = f(&args_float);
                Complex::new(r.re.to_double_float(), r.im.to_double_float())
            }))
        } else {
            None
        }
    }
}

impl EvaluationDomain for Float {
    const FIXED_PRECISION: Option<u32> = None;

    fn try_from_complex_float(f: Complex<Float>) -> Result<Self, String> {
        if f.is_real() {
            Ok(f.re)
        } else {
            Err(format!(
                "Cannot convert from Complex<Float> to Float because the result {f} is not real"
            ))
        }
    }
}
impl EvaluationDomain for Complex<Float> {
    const FIXED_PRECISION: Option<u32> = None;

    fn try_from_complex_float(f: Complex<Float>) -> Result<Self, String> {
        Ok(f)
    }
}

impl EvaluationDomain for ErrorPropagatingFloat<f64> {
    const FIXED_PRECISION: Option<u32> = f64::FIXED_PRECISION;

    fn try_from_complex_float(f: Complex<Float>) -> Result<Self, String> {
        if f.is_real() {
            Ok(ErrorPropagatingFloat::new(
                f.re.to_f64(),
                15.954589770191003,
            ))
        } else {
            Err(format!(
                "Cannot convert from Complex<Float> to Float because the result {f} is not real"
            ))
        }
    }
}

impl EvaluationDomain for ErrorPropagatingFloat<Float> {
    const FIXED_PRECISION: Option<u32> = None;

    fn try_from_complex_float(f: Complex<Float>) -> Result<Self, String> {
        let dr = -f.re.get_epsilon().log10();
        if f.is_real() {
            Ok(ErrorPropagatingFloat::new(f.re, dr))
        } else {
            Err(format!(
                "Cannot convert from Complex<Float> to Float because the result {f} is not real"
            ))
        }
    }
}

impl EvaluationDomain for Complex<ErrorPropagatingFloat<f64>> {
    const FIXED_PRECISION: Option<u32> = Some(53);

    fn try_from_complex_float(f: Complex<Float>) -> Result<Self, String> {
        Ok(Complex::new(
            ErrorPropagatingFloat::new(f.re.to_f64(), 15.954589770191003),
            ErrorPropagatingFloat::new(f.im.to_f64(), 15.954589770191003),
        ))
    }
}

impl EvaluationDomain for Complex<ErrorPropagatingFloat<Float>> {
    const FIXED_PRECISION: Option<u32> = Some(53);

    fn try_from_complex_float(f: Complex<Float>) -> Result<Self, String> {
        let dr = -f.re.get_epsilon().log10();
        let di = -f.im.get_epsilon().log10();
        Ok(Complex::new(
            ErrorPropagatingFloat::new(f.re, dr),
            ErrorPropagatingFloat::new(f.im, di),
        ))
    }
}

#[cfg(test)]
mod ball_tests {
    use crate::{
        atom::{AtomCore, EvaluationInfo},
        domains::{
            float::{Complex, ComplexBall, Float, RealBall},
            rational::Rational,
        },
        evaluate::EvaluationDomain,
        parse,
    };

    #[test]
    fn polynomial_evaluators_enclose_exact_rational_values() {
        let expression = parse!("x^3/3+2*x/7+5/11");
        let exact = expression.evaluator(&[parse!("x")]).build().unwrap();
        let rational = Rational::from((8, 3)) + Rational::from((4, 7)) + Rational::from((5, 11));
        let reference = RealBall::from_rational_ball(&rational, &Rational::from(0), 160);
        let mut real = exact.clone().map_coeff_with_prec(
            &|c| RealBall::from_rational_ball(&c.re, &Rational::from(0), 80),
            80,
        );
        let value = real.evaluate_single(&[RealBall::exact(Float::with_val(80, 2))]);
        assert!(value.contains_ball(&reference));

        let mut complex = exact.map_coeff_with_prec(
            &|c| ComplexBall::from_rational_ball(c, &Rational::from(0), 80),
            80,
        );
        let value = complex.evaluate_single(&[ComplexBall::new(
            RealBall::exact(Float::with_val(80, 2)),
            RealBall::exact(Float::with_val(80, 0)),
        )]);
        assert!(value.re.contains_ball(&reference));
        assert!(value.im.contains(&Float::with_val(80, 0)));
    }

    #[test]
    fn rounded_constants_and_float_callback_fallbacks_are_not_admitted() {
        let rounded = Complex::new(Float::with_val(80, 1), Float::with_val(80, 0));
        assert!(RealBall::try_from_complex_float(rounded.clone()).is_err());
        assert!(ComplexBall::try_from_complex_float(rounded).is_err());
        let info = EvaluationInfo::new().register(|x: &[Complex<Float>]| x[0].clone());
        assert!(RealBall::resolve_function(&[], &info).is_none());
        assert!(ComplexBall::resolve_function(&[], &info).is_none());
        assert!(RealBall::FIXED_PRECISION.is_none());
        assert!(ComplexBall::FIXED_PRECISION.is_none());
    }
}
