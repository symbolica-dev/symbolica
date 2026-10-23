use symbolica::{
    atom::{Atom, AtomCore, EvaluationInfo},
    domains::rational::Rational,
    domains::{
        dual::HyperDual,
        float::{Complex, ErrorPropagatingFloat, F64, Float, RealLike, SingleFloat},
    },
    evaluate::{
        Dualizer, EvaluatorComposer, ExpressionEvaluator, FunctionMap, FunctionRegistrationOptions,
        InliningPolicy, OptimizationSettings,
    },
    parse, symbol,
};

type Tracked = ErrorPropagatingFloat<Float>;
type ExactEvaluator = ExpressionEvaluator<Complex<Rational>>;

fn tracked(center: f64, error: f64) -> Tracked {
    Tracked::new_with_absolute_error(Float::with_val(128, center), error)
}

fn map_real(evaluator: ExactEvaluator) -> ExpressionEvaluator<Tracked> {
    evaluator
        .try_map_coeff_with_prec(&|value| tracked(0., 0.).from_rational(&value.re), 128)
        .unwrap()
}

fn build(expression: &Atom, direct: bool) -> ExactEvaluator {
    expression
        .evaluator(&[parse!("x")])
        .direct_translation(direct)
        .horner_iterations(3)
        .cores(1)
        .build()
        .unwrap()
}

#[test]
fn both_builders_keep_uncertain_zero_and_unit_through_optimization() {
    for direct in [false, true] {
        for (center, error) in [(0., 0.125), (1., 0.25)] {
            let expression = Atom::num(tracked(center, error)) * parse!("x") + 1;
            let exact = build(&expression, direct);
            let exported = exact.export_instructions();
            assert_eq!(exported.floating_coefficients.len(), 1);
            assert!(exported.constant_functions.is_empty());
            let original = exported.floating_coefficients[0]
                .coefficient
                .to_tracked_float(None)
                .unwrap();
            assert_eq!(original.re.get_absolute_error(), error);
            let result = map_real(exact.clone()).evaluate_single(&[tracked(2., 0.)]);
            assert_eq!(result.get_num().to_f64(), center * 2. + 1.);
            assert!(result.get_absolute_error() >= 2. * error);
            let mut plain = exact.map_coeff(&|value| value.re.to_f64());
            assert_eq!(plain.evaluate_single(&[2.]), center * 2. + 1.);
        }
    }
}

#[test]
fn tree_mapping_preserves_original_coefficient_error() {
    let expression = Atom::num(tracked(0., 0.125)) * parse!("x") + 1;
    let tree = expression
        .as_view()
        .to_evaluation_tree(&FunctionMap::new(), &[parse!("x")])
        .unwrap();
    let mut tree = tree.map_coeff(&|value| tracked(0., 0.).from_rational(&value.re));
    let mut out = [tracked(0., 0.)];
    tree.evaluate(&[tracked(2., 0.)], &mut out);
    assert_eq!(out[0].get_num().to_f64(), 1.);
    assert!(out[0].get_absolute_error() >= 0.25);
}

#[test]
fn distinct_errors_survive_merge_and_composition() {
    let first = Atom::num(tracked(1., 0.125)) * parse!("x");
    let second = Atom::num(tracked(1., 0.5)) * parse!("x");
    for direct in [false, true] {
        let first = build(&first, direct);
        let second = build(&second, direct);
        let mut merged = first.clone();
        merged.merge(second.clone(), Some(2)).unwrap();
        assert_eq!(merged.export_instructions().floating_coefficients.len(), 2);
        let mut mapped = map_real(merged);
        let mut out = [tracked(0., 0.), tracked(0., 0.)];
        mapped.evaluate(&[tracked(2., 0.)], &mut out);
        assert!(out[0].get_absolute_error() >= 0.25);
        assert!(out[1].get_absolute_error() >= 1.);
        assert!(out[0].get_absolute_error() < out[1].get_absolute_error());

        let (mut composer, inputs) = EvaluatorComposer::new_with_inputs(1);
        let mut outputs = composer.add(&first, &inputs).unwrap();
        outputs.extend(composer.add(&second, &inputs).unwrap());
        let composed = composer
            .finish(&outputs, OptimizationSettings::new())
            .unwrap();
        assert_eq!(
            composed.export_instructions().floating_coefficients.len(),
            2
        );
        let mut mapped = map_real(composed);
        mapped.evaluate(&[tracked(2., 0.)], &mut out);
        assert!(out[0].get_absolute_error() >= 0.25);
        assert!(out[1].get_absolute_error() >= 1.);
        assert!(out[0].get_absolute_error() < out[1].get_absolute_error());
    }
}

#[test]
fn mapped_evaluator_merge_keeps_resolved_floating_constants() {
    let first = Atom::num(tracked(1., 0.125)) * parse!("x");
    let second = Atom::num(tracked(2., 0.5)) * parse!("x");
    let mut merged = build(&first, true).map_coeff(&|c| F64::from(c.re.to_f64()));
    merged
        .merge(
            build(&second, true).map_coeff(&|c| F64::from(c.re.to_f64())),
            Some(2),
        )
        .unwrap();
    let mut plain = [F64::from(0.), F64::from(0.)];
    merged.evaluate(&[F64::from(2.)], &mut plain);
    assert_eq!(plain[0].to_f64(), 2.);
    assert_eq!(plain[1].to_f64(), 4.);
    let mut merged = merged
        .try_map_coeff_with_prec(&|c| tracked(c.to_f64(), 0.), 128)
        .unwrap();
    let mut out = [tracked(0., 0.), tracked(0., 0.)];
    merged.evaluate(&[tracked(2., 0.)], &mut out);
    assert_eq!(out[0].get_num().to_f64(), 2.);
    assert_eq!(out[1].get_num().to_f64(), 4.);
    assert!(out[1].get_absolute_error() >= 1.);
}

#[test]
fn non_inlined_function_coefficients_keep_their_errors() {
    let name = symbol!("evaluator_uncertainty::function");
    let body = Atom::num(tracked(1., 0.125)) * parse!("x");
    let expression = name.call(parse!("x"));
    let exact = expression
        .evaluator(&[parse!("x")])
        .add_function_with_options(
            name,
            vec![symbol!("x")],
            body,
            FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
        )
        .unwrap()
        .build()
        .unwrap();
    let value = map_real(exact).evaluate_single(&[tracked(2., 0.)]);
    assert_eq!(value.get_num().to_f64(), 2.);
    assert!(value.get_absolute_error() >= 0.25);
}

#[test]
fn uncertain_imaginary_zero_requires_a_complex_target() {
    let expression = Atom::num(Complex::new(tracked(1., 0.), tracked(0., 0.125)));
    for direct in [false, true] {
        let exact = build(&expression, direct);
        assert!(exact.is_real());
        let plain = exact
            .clone()
            .map_coeff(&|c| Complex::new(c.re.to_f64(), c.im.to_f64()));
        assert!(plain.is_real());
        assert!(
            exact
                .clone()
                .try_map_coeff_with_prec(&|c| tracked(0., 0.).from_rational(&c.re), 128)
                .is_err()
        );
        let mut evaluator = exact
            .try_map_coeff_with_prec(
                &|c| {
                    Complex::new(
                        tracked(0., 0.).from_rational(&c.re),
                        tracked(0., 0.).from_rational(&c.im),
                    )
                },
                128,
            )
            .unwrap();
        assert!(!evaluator.is_real());
        let result = evaluator.evaluate_single(&[Complex::new(tracked(2., 0.), tracked(0., 0.))]);
        assert!(result.im.get_num().is_zero());
        assert_eq!(result.im.get_absolute_error(), 0.125);
    }
}

#[cfg(feature = "bincode")]
#[test]
fn evaluator_binary_roundtrip_keeps_original_metadata() {
    let expression = Atom::num(tracked(0., 0.125)) * parse!("x");
    let original = build(&expression, true);
    let config = bincode::config::standard();
    let bytes = bincode::encode_to_vec(&original, config).unwrap();
    let (decoded, consumed): (ExactEvaluator, _) =
        bincode::decode_from_slice(&bytes, config).unwrap();
    assert_eq!(consumed, bytes.len());
    let result = map_real(decoded).evaluate_single(&[tracked(2., 0.)]);
    assert!(result.get_absolute_error() >= 0.25);
    assert!(bincode::decode_from_slice::<ExactEvaluator, _>(&[0], config).is_err());
}

#[cfg(feature = "serde")]
#[test]
fn evaluator_serde_roundtrip_keeps_original_metadata() {
    let expression = Atom::num(tracked(0., 0.125)) * parse!("x");
    let original = build(&expression, true);
    let json = serde_json::to_string(&original).unwrap();
    let decoded: ExactEvaluator = serde_json::from_str(&json).unwrap();
    let result = map_real(decoded).evaluate_single(&[tracked(2., 0.)]);
    assert!(result.get_absolute_error() >= 0.25);
}

#[test]
fn vectorized_derivatives_retain_coefficient_uncertainty() {
    let expression = Atom::num(tracked(1., 0.125)) * parse!("x");
    for direct in [false, true] {
        let evaluator = build(&expression, direct)
            .vectorize(&Dualizer::new(
                HyperDual::<Complex<Rational>>::new(vec![vec![0], vec![1]]),
                vec![],
            ))
            .unwrap();
        let mut evaluator = map_real(evaluator);
        let mut output = [tracked(0., 0.), tracked(0., 0.)];
        evaluator.evaluate(&[tracked(2., 0.), tracked(1., 0.)], &mut output);
        assert_eq!(output[0].get_num().to_f64(), 2.);
        assert_eq!(output[1].get_num().to_f64(), 1.);
        assert!(output[0].get_absolute_error() >= 0.25);
        assert!(output[1].get_absolute_error() >= 0.125);
    }
}

#[test]
fn constant_callbacks_cannot_drop_uncertainty_in_tags() {
    let name = symbol!(
        "evaluator_uncertainty::tagged_constant",
        eval = EvaluationInfo::constant(|tags, precision| {
            tags[0]
                .evaluate_with_prec(
                    &std::collections::HashMap::<Atom, Complex<Float>>::new(),
                    precision,
                )
                .map_err(|error| error.to_string())
        })
        .with_tags(1)
    );
    let expression = name.call(Atom::num(tracked(1., 0.125)));
    for direct in [false, true] {
        let evaluator = build(&expression, direct);
        let mut plain = evaluator.clone().map_coeff(&|c| c.re.to_f64());
        assert_eq!(plain.evaluate_single(&[2.]), 1.);
        assert!(
            evaluator
                .try_map_coeff_with_prec(&|c| tracked(0., 0.).from_rational(&c.re), 128)
                .is_err()
        );
    }
}

#[test]
fn merge_after_composition_keeps_duplicate_constant_slots_distinct() {
    let expression = Atom::num(tracked(1., 0.125)) * parse!("x");
    let evaluator = build(&expression, true);
    let (mut composer, inputs) = EvaluatorComposer::new_with_inputs(1);
    let mut outputs = composer.add(&evaluator, &inputs).unwrap();
    outputs.extend(composer.add(&evaluator, &inputs).unwrap());
    let mut composed = composer
        .finish(&outputs, OptimizationSettings::new())
        .unwrap();
    let second = Atom::num(tracked(3., 0.5)) * parse!("x");
    composed.merge(build(&second, true), Some(2)).unwrap();
    let mut mapped = map_real(composed);
    let mut output = [tracked(0., 0.), tracked(0., 0.), tracked(0., 0.)];
    mapped.evaluate(&[tracked(2., 0.)], &mut output);
    assert_eq!(output[0].get_num().to_f64(), 2.);
    assert_eq!(output[1].get_num().to_f64(), 2.);
    assert_eq!(output[2].get_num().to_f64(), 6.);
    assert!(output[2].get_absolute_error() >= 1.);
}

#[test]
fn mapping_preserves_precision_and_out_of_range_uncertainty() {
    let center = Float::parse("1e-1000", Some(192)).unwrap();
    let original = ErrorPropagatingFloat::from(center);
    let original_error = original.get_error_magnitude();
    let original_scale = original.get_num().absolute_magnitude();
    for direct in [false, true] {
        let evaluator = build(&Atom::num(original.clone()), direct);
        let mut precise = evaluator
            .clone()
            .try_map_coeff_with_prec(&|c| tracked(0., 0.).from_rational(&c.re), 256)
            .unwrap();
        let value = precise.evaluate_single(&[tracked(2., 0.)]);
        assert_eq!(value.get_num().prec(), 192);
        assert_eq!(value.get_error_magnitude(), original_error);
        let mut fixed = evaluator
            .map_coeff(&|c| ErrorPropagatingFloat::new_with_absolute_error(c.re.to_f64(), 0.));
        let value =
            fixed.evaluate_single(&[ErrorPropagatingFloat::new_with_absolute_error(2., 0.)]);
        assert_eq!(*value.get_num(), 0.);
        assert!(value.get_error_magnitude() >= original_scale);
    }
}

#[cfg(feature = "native_code_generation")]
#[test]
fn source_exports_resolve_floating_centers_before_codegen() {
    use symbolica::evaluate::{ExportSettings, InlineASM, NumberClass};
    let floating = build(&Atom::num(tracked(1.25, 0.125)), true);
    let rational = build(&Atom::num(Rational::from((5, 4))), true);
    let settings = ExportSettings::default().inline_asm(InlineASM::None);
    assert_eq!(
        floating
            .export_cpp_str::<f64>("coefficient", settings.clone())
            .unwrap(),
        rational
            .export_cpp_str::<f64>("coefficient", settings.clone())
            .unwrap(),
    );
    assert_eq!(
        floating.export_simd_str("coefficient", settings.clone(), false, InlineASM::None),
        rational.export_simd_str("coefficient", settings.clone(), false, InlineASM::None),
    );
    assert_eq!(
        floating.export_cuda_str("coefficient", settings.clone(), NumberClass::RealF64),
        rational.export_cuda_str("coefficient", settings.clone(), NumberClass::RealF64),
    );
    let complex = build(
        &Atom::num(Complex::new(tracked(1.25, 0.125), tracked(0.5, 0.125))),
        true,
    );
    assert!(
        complex
            .export_cpp_str::<f64>("coefficient", settings.clone())
            .is_err()
    );
    let rational = build(
        &Atom::num(Complex::new(Rational::from((5, 4)), Rational::from((1, 2)))),
        true,
    );
    assert_eq!(
        complex
            .export_cpp_str::<Complex<f64>>("coefficient", settings.clone())
            .unwrap(),
        rational
            .export_cpp_str::<Complex<f64>>("coefficient", settings)
            .unwrap(),
    );
}
