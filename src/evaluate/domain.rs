use super::*;

/// A numeric domain supported by the expression evaluator.
///
/// It finds the best implementation to numericaly evaluate a function.
pub trait EvaluationDomain: Sized + 'static {
    /// The single supported binary precision for fixed-precision domains.
    ///
    /// Use `None` for variable-precision domains such as `Float` and `Complex<Float>`.
    const FIXED_PRECISION: Option<u32>;

    /// Whether evaluation must preserve input uncertainty. Such domains require
    /// a matching tracked callback for constants with uncertain symbolic tags.
    /// Plain numeric domains evaluate the central value instead.
    const TRACKS_UNCERTAINTY: bool = false;

    /// Try to convert a `Complex<Float>` to the type.
    fn try_from_complex_float(_: Complex<Float>) -> Result<Self, String> {
        Err(format!(
            "Cannot convert from Complex<Float> to {}",
            std::any::type_name::<Self>()
        ))
    }

    /// Convert a coefficient while preserving uncertainty when supported by the domain.
    /// Ordinary floating-point domains receive the central value; error-tracking
    /// domains override this method to retain the coefficient's absolute error.
    fn try_from_complex_tracked_float(
        f: Complex<ErrorPropagatingFloat<Float>>,
    ) -> Result<Self, String> {
        Self::try_from_complex_float(Complex::new(f.re.into_value(), f.im.into_value()))
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
    const TRACKS_UNCERTAINTY: bool = true;

    const FIXED_PRECISION: Option<u32> = f64::FIXED_PRECISION;

    fn try_from_complex_float(f: Complex<Float>) -> Result<Self, String> {
        Self::try_from_complex_tracked_float(f.into())
    }

    fn try_from_complex_tracked_float(
        f: Complex<ErrorPropagatingFloat<Float>>,
    ) -> Result<Self, String> {
        let f = Complex::<ErrorPropagatingFloat<f64>>::try_from_complex_tracked_float(f)?;
        if f.im.is_exact_zero() {
            Ok(f.re)
        } else {
            Err("Cannot convert an uncertain complex coefficient to a real domain".into())
        }
    }
}

impl EvaluationDomain for ErrorPropagatingFloat<Float> {
    const TRACKS_UNCERTAINTY: bool = true;

    const FIXED_PRECISION: Option<u32> = None;

    fn try_from_complex_float(f: Complex<Float>) -> Result<Self, String> {
        Self::try_from_complex_tracked_float(f.into())
    }

    fn try_from_complex_tracked_float(
        f: Complex<ErrorPropagatingFloat<Float>>,
    ) -> Result<Self, String> {
        if f.im.is_exact_zero() {
            Ok(f.re)
        } else {
            Err("Cannot convert an uncertain complex coefficient to a real domain".into())
        }
    }
}

impl EvaluationDomain for Complex<ErrorPropagatingFloat<f64>> {
    const TRACKS_UNCERTAINTY: bool = true;

    const FIXED_PRECISION: Option<u32> = Some(53);

    fn try_from_complex_float(f: Complex<Float>) -> Result<Self, String> {
        Self::try_from_complex_tracked_float(f.into())
    }

    fn try_from_complex_tracked_float(
        f: Complex<ErrorPropagatingFloat<Float>>,
    ) -> Result<Self, String> {
        let convert = |f: ErrorPropagatingFloat<Float>| {
            let value = f.get_num().to_f64();
            let rounding = if value.is_finite() && f.get_num().is_finite() {
                // Compare centers directly: relative epsilon understates the
                // rounding of nonzero subnormals, and misses total underflow.
                (f.get_num().clone() - Float::with_val(f.get_num().prec().max(53), value))
                    .absolute_magnitude()
            } else {
                crate::domains::float::Magnitude::INFINITY
            };
            ErrorPropagatingFloat::new_with_error_magnitude(
                value,
                f.get_error_magnitude() + rounding,
            )
        };
        Ok(Complex::new(convert(f.re), convert(f.im)))
    }
}

impl EvaluationDomain for Complex<ErrorPropagatingFloat<Float>> {
    const TRACKS_UNCERTAINTY: bool = true;

    const FIXED_PRECISION: Option<u32> = None;

    fn try_from_complex_float(f: Complex<Float>) -> Result<Self, String> {
        // Plain complex zeros identify exact axes. Preserve them when attaching
        // native-precision errors to the nonzero components.
        Ok(f.into())
    }

    fn try_from_complex_tracked_float(
        f: Complex<ErrorPropagatingFloat<Float>>,
    ) -> Result<Self, String> {
        Ok(f)
    }
}

#[cfg(test)]
mod tracked_precision_tests {
    use crate::{
        atom::{Atom, AtomCore, Symbol},
        domains::{
            float::{Complex, ErrorPropagatingFloat, Float, FloatLike, Magnitude, RealLike},
            rational::Rational,
        },
        evaluate::EvaluationDomain,
    };

    type Tracked = ErrorPropagatingFloat<Float>;

    #[test]
    fn tracked_real_conversion_preserves_errors_beyond_f64_range() {
        for value in [0, 1, -2] {
            let converted = Tracked::try_from_complex_float(Complex::new(
                Float::with_val(2048, value),
                Float::with_val(2048, 0),
            ))
            .unwrap();
            assert_eq!(converted.get_num(), &Float::with_val(2048, value));
            assert_eq!(converted.get_num().prec(), 2048);
            if value == 0 {
                assert!(converted.is_exact_zero());
                continue;
            }
            assert_eq!(
                converted.get_error_magnitude(),
                Magnitude::from_f64((value as f64).abs()).scale_pow2(-2048)
            );
            assert!(converted.get_error_magnitude().is_finite());
            assert!(!converted.get_error_magnitude().is_zero());
            assert_eq!(converted.get_error_magnitude().to_f64(), None);
            assert!(converted.get_absolute_error() > 0.);
        }
    }

    #[test]
    fn tracked_complex_conversion_uses_each_components_precision() {
        let converted = Complex::<Tracked>::try_from_complex_float(Complex::new(
            Float::with_val(2048, -2),
            Float::with_val(4096, 0),
        ))
        .unwrap();
        assert_eq!(converted.re.get_num().prec(), 2048);
        assert_eq!(converted.im.get_num().prec(), 4096);
        assert_eq!(
            converted.re.get_error_magnitude(),
            Magnitude::from_f64(2.).scale_pow2(-2048)
        );
        // Plain zero components are exact in the tracking coefficient domain.
        assert!(converted.im.is_exact_zero());
        assert_eq!(converted.re.get_error_magnitude().to_f64(), None);
        assert_eq!(converted.im.get_error_magnitude().to_f64(), Some(0.));
    }

    #[test]
    fn tracked_constants_use_the_requested_arbitrary_precision() {
        assert_eq!(Tracked::FIXED_PRECISION, None);
        assert_eq!(Complex::<Tracked>::FIXED_PRECISION, None);
        let pi = Atom::var(Symbol::PI);
        let direct = pi
            .evaluate_with_prec(
                &std::collections::HashMap::<Atom, Complex<Tracked>>::new(),
                2048,
            )
            .unwrap();
        let mut evaluator = pi
            .evaluator::<Atom>(&[])
            .build()
            .unwrap()
            .map_coeff_with_prec(
                &|c: &Complex<Rational>| {
                    Complex::new(
                        Tracked::new_with_native_precision(c.re.to_multi_prec_float(2048)),
                        Tracked::new_with_native_precision(c.im.to_multi_prec_float(2048)),
                    )
                },
                2048,
            );
        let mapped = evaluator.evaluate_single(&[]);
        for (direct, mapped) in [(&direct.re, &mapped.re), (&direct.im, &mapped.im)] {
            assert_eq!(mapped.get_num().prec(), 2048);
            assert_eq!(mapped.get_num(), direct.get_num());
            assert_eq!(mapped.get_error_magnitude(), direct.get_error_magnitude());
            assert!(mapped.get_error_magnitude().is_finite());
            if mapped.is_exact_zero() {
                assert_eq!(mapped.get_error_magnitude().to_f64(), Some(0.));
            } else {
                assert!(!mapped.get_error_magnitude().is_zero());
                assert_eq!(mapped.get_error_magnitude().to_f64(), None);
            }
        }
        assert!(direct.im.is_exact_zero());
        assert!(mapped.im.is_exact_zero());
    }

    #[test]
    fn tracked_binary64_conversion_preserves_exact_axes_and_underflow_error() {
        type Tracked64 = ErrorPropagatingFloat<f64>;
        let real = Complex::new(Float::with_val(24, 1), Float::new(4096));
        let converted = Complex::<Tracked64>::try_from_complex_float(real).unwrap();
        assert!(converted.im.is_exact_zero());
        assert!(converted.re.get_error_magnitude() >= Magnitude::ONE.scale_pow2(-24));

        let tiny = Float::with_val(128, 0.5).pow(4000);
        let source = Complex::new(Float::with_val(2048, 1), tiny.clone());
        let wide = Complex::<Tracked>::try_from_complex_float(source.clone()).unwrap();
        assert_eq!(wide.im.get_num(), &tiny);
        assert_eq!(wide.im.get_num().prec(), 128);
        assert!(!wide.im.get_error_magnitude().is_zero());
        assert_eq!(wide.im.get_error_magnitude().to_f64(), None);

        let converted = Complex::<Tracked64>::try_from_complex_float(source.clone()).unwrap();
        assert_eq!(*converted.im.get_num(), 0.);
        assert!(!converted.im.is_exact_zero());
        assert!(converted.im.get_error_magnitude() >= tiny.absolute_magnitude());
        assert_eq!(converted.im.get_error_magnitude().to_f64(), None);
        assert!(Tracked64::try_from_complex_float(source).is_err());

        // 3 * 2^-1076 rounds upward to 2^-1074. Its conversion error is
        // 2^-1076, much larger than a relative epsilon at this subnormal.
        let quarter_subnormal = Float::with_val(256, 0.5).pow(1076);
        let source = Complex::new(
            Float::with_val(256, 1),
            quarter_subnormal.clone() * Float::with_val(256, 3),
        );
        let converted = Complex::<Tracked64>::try_from_complex_float(source).unwrap();
        assert_eq!(*converted.im.get_num(), f64::from_bits(1));
        assert!(converted.im.get_error_magnitude() >= quarter_subnormal.absolute_magnitude());
    }

    #[test]
    fn tracked_binary64_real_constants_stay_real_in_both_evaluators() {
        type Tracked64 = ErrorPropagatingFloat<f64>;
        let expression = Atom::var(Symbol::PI) + Atom::var(Symbol::E);
        let direct = expression
            .evaluate(&std::collections::HashMap::<Atom, Complex<Tracked64>>::new())
            .unwrap();
        let mut evaluator = expression
            .evaluator::<Atom>(&[])
            .build()
            .unwrap()
            .map_coeff(&|c: &Complex<Rational>| {
                Complex::<Tracked64>::try_from_complex_float(Complex::new(
                    c.re.to_multi_prec_float(53),
                    c.im.to_multi_prec_float(53),
                ))
                .unwrap()
            });
        let optimized = evaluator.evaluate_single(&[]);
        for value in [direct, optimized] {
            assert!(value.im.is_exact_zero());
            assert!(!value.re.is_exact());
            assert!(value.re.get_error_magnitude().is_finite());
            assert!((value.re.value - std::f64::consts::PI - std::f64::consts::E).abs() < 1e-14);
        }
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
