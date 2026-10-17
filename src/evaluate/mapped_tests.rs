use super::*;
use crate::parse;

#[test]
fn mapped_eager_validates_before_setters_and_reuses_native_values() {
    let mut program = parse!("x^2+1/x")
        .evaluator(&[parse!("x")])
        .build()
        .unwrap()
        .map_coeff(&|value| value.re.to_f64());
    let mut output = [(99.,)];
    assert!(
        program
            .try_evaluate_mapped::<u8, _>(
                &[],
                &mut output,
                |_, _| panic!("invalid input reached mapper"),
                |_, _| panic!("invalid input reached output mapper")
            )
            .is_err()
    );
    assert_eq!(output[0].0, 99.);
    assert!(
        program
            .try_evaluate_mapped(
                &[(2.,)],
                &mut [] as &mut [u8],
                |_, _| panic!("invalid output reached mapper"),
                |_, _| panic!("invalid output reached output mapper")
            )
            .is_err()
    );
    for (input, expected) in [(2., 4.5), (-2., 3.5), (0., f64::INFINITY), (4., 16.25)] {
        program
            .try_evaluate_mapped(
                &[(input,)],
                &mut output,
                |destination, source| *destination = source.0,
                |destination, source| destination.0 = *source,
            )
            .unwrap();
        assert_eq!(output[0].0, expected);
    }
}

#[test]
fn mapped_rows_keep_native_reduction_duals_and_lazy_poles() {
    let x = parse!("mapped_rows::x");
    let c = parse!("mapped_rows::c");
    let gate = parse!("mapped_rows::gate");
    let mut program = parse!("if(mapped_rows::gate, mapped_rows::c/mapped_rows::x, 0)")
        .evaluator(&[x, c.clone(), gate])
        .build_parameter_rows(&[c], &[vec![1.into()], vec![2.into()]])
        .unwrap()
        .vectorize(&Dualizer::new(
            crate::domains::dual::HyperDual::<Complex<Rational>>::new(vec![
                vec![0],
                vec![1],
                vec![2],
            ]),
            vec![(1, 1), (1, 2)],
        ))
        .unwrap()
        .map_coeff(&|value| value.re.to_f64());
    let mut output = [(99.,); 3];
    assert!(
        program
            .evaluate_mapped::<(f64,), (f64,)>(
                &[],
                &mut output,
                Rational::to_f64,
                |_, _| panic!("invalid input reached mapper"),
                |_, _| panic!("invalid input reached output mapper")
            )
            .is_err()
    );
    for (x, gate, expected) in [
        (2., 1., [1.5, -0.75, 0.375]),
        (0., 0., [0., 0., 0.]),
        (-2., 1., [-1.5, -0.75, -0.375]),
        (0., 0., [0., 0., 0.]),
    ] {
        let input = [x, 1., 0., gate, 0., 0.].map(|value| (value,));
        program
            .evaluate_mapped(
                &input,
                &mut output,
                Rational::to_f64,
                |destination, source| *destination = source.0,
                |destination, source| destination.0 = *source,
            )
            .unwrap();
        assert_eq!(output.map(|value| value.0), expected);
    }
}

#[test]
fn mapped_eager_keeps_arbitrary_precision() {
    let rational = Rational::from((1, 7));
    let prototype = Float::with_val(192, 0);
    let mut program = parse!("x/3")
        .evaluator(&[parse!("x")])
        .build()
        .unwrap()
        .map_coeff_with_prec(&|value| prototype.from_rational(&value.re), 192);
    let input = [prototype.from_rational(&rational)];
    let mut output = [Float::with_val(192, 0)];
    program
        .try_evaluate_mapped(&input, &mut output, Float::set_from, Float::set_from)
        .unwrap();
    assert!(output[0].get_precision() > 180);
    let reference = prototype.from_rational(&Rational::from((1, 21)));
    assert!((output[0].clone() - reference).norm() < Float::with_val(192, 1e-55));
}

#[test]
fn mapped_rows_refresh_precision_in_reused_workspace() {
    let x = parse!("mapped_precision::x");
    let c = parse!("mapped_precision::c");
    let mut program = (&x * &c)
        .evaluator(&[x.clone(), c.clone()])
        .build_parameter_rows(
            &[c],
            &[vec![Rational::from((1, 3))], vec![Rational::from((2, 3))]],
        )
        .unwrap()
        .map_coeff_with_prec(
            &|value| Float::with_val(1024, 0).from_rational(&value.re),
            1024,
        );
    let mut output = [(Float::with_val(32, 0),)];
    for precision in [80, 256, 128] {
        let prototype = Float::with_val(precision, 0);
        let reference = prototype.from_rational(&Rational::from((1, 7)));
        program
            .evaluate_mapped(
                &[(reference.clone(),)],
                &mut output,
                |value| prototype.from_rational(value),
                |destination, source| destination.set_from(&source.0),
                |destination, source| destination.0.set_from(source),
            )
            .unwrap();
        assert!(output[0].0.get_precision() >= precision - 8);
        let tolerance = Float::with_val(precision, 2)
            .pow(u64::from(precision - 8))
            .inv();
        assert!((output[0].0.clone() - reference).norm() < tolerance);
    }
}

#[derive(Clone)]
struct ScalarWrapper(Float);

impl From<ScalarWrapper> for Float {
    fn from(value: ScalarWrapper) -> Self {
        value.0
    }
}

impl EvaluationDomain for ScalarWrapper {
    const FIXED_PRECISION: Option<u32> = None;
    fn try_from_complex_float(value: Complex<Float>) -> Result<Self, String> {
        if value.im.is_fully_zero() {
            Ok(Self(value.re))
        } else {
            Err("not real".into())
        }
    }
}

impl ComplexEvaluationDomain for ScalarWrapper {}

#[test]
fn complex_domain_opt_in_preserves_component_precision_and_callback_priority() {
    type C = Complex<ScalarWrapper>;
    let value = Complex::new(Float::with_val(192, 2), Float::with_val(320, 3));
    let converted = C::try_from_complex_float(value).unwrap();
    assert_eq!(converted.re.0.get_precision(), 192);
    assert_eq!(converted.im.0.get_precision(), 320);
    let fallback = EvaluationInfo::new().register(|args: &[Complex<Float>]| args[0].clone());
    let result = C::resolve_function(&[], &fallback).unwrap()(std::slice::from_ref(&converted));
    assert_eq!(result.re.0, converted.re.0);
    assert_eq!(result.im.0.get_precision(), 320);
    let both = fallback.register(|args: &[C]| Complex::new(args[0].im.clone(), args[0].re.clone()));
    let result = C::resolve_function(&[], &both).unwrap()(&[converted]);
    assert_eq!(result.re.0, Float::with_val(320, 3));
    assert_eq!(result.im.0.get_precision(), 192);
    assert!(Complex::<Rational>::resolve_function(&[], &both).is_none());
}

#[test]
fn mapped_row_backend_error_preserves_output_and_allows_reuse() {
    let x = parse!("mapped_error::x");
    let c = parse!("mapped_error::c");
    let program = (&x * &c)
        .evaluator(&[x.clone(), c.clone()])
        .build_parameter_rows(&[c], &[vec![1.into()], vec![2.into()]])
        .unwrap()
        .map_coeff(&|value| value.re.to_f64());
    let (plan, mut prelude, mut body) = program.into_parts();
    let mut workspace = ParameterRowWorkspace::default();
    let mut output = [(77.,)];
    let failed = plan.evaluate_mapped_with(
        &[(2.,)],
        &mut output,
        &mut workspace,
        Rational::to_f64,
        |destination, source| *destination = source.0,
        |destination, source| destination.0 = *source,
        |input, output| prelude.try_evaluate(input, output),
        |_, _| {
            Err(EvaluationError::InvalidOutputCount {
                expected: 1,
                actual: 0,
            })
        },
    );
    assert!(failed.is_err());
    assert_eq!(output, [(77.,)]);
    plan.evaluate_mapped_with(
        &[(3.,)],
        &mut output,
        &mut workspace,
        Rational::to_f64,
        |destination, source| *destination = source.0,
        |destination, source| destination.0 = *source,
        |input, output| prelude.try_evaluate(input, output),
        |input, output| body.try_evaluate(input, output),
    )
    .unwrap();
    assert_eq!(output, [(9.,)]);
}
