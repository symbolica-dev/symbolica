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
                let mut buffer = smallvec::SmallVec::<[f64; 4]>::from([0.; 4]);
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
