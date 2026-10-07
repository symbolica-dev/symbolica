use symbolica::{
    evaluate::{EvaluatorComposer, FunctionRegistrationOptions, InliningPolicy, Slot},
    prelude::*,
};
#[test]
fn composition_preserves_native_control_flow_callbacks_and_admission() {
    let settings = OptimizationSettings::new()
        .horner_iterations(0)
        .cores(1)
        .cpe_iterations(Some(1000));
    let p = Atom::evaluator_multiple(
        &[
            parse!("x+y+x*y"),
            parse!("if(x,if(y,x+y,x-7),y+11)"),
            parse!("gamma(x+1)+gamma(1/2)"),
        ],
        &[parse!("x"), parse!("y")],
    )
    .optimization_settings(settings.clone())
    .build()
    .unwrap();
    let dependent = parse!("a+b*c")
        .evaluator(&[parse!("a"), parse!("b"), parse!("c")])
        .optimization_settings(settings.clone())
        .build()
        .unwrap();
    let constant = Atom::evaluator_multiple(
        &[
            parse!("-123456789012345678901234567890123456789/7"),
            parse!("gamma(1/2)"),
        ],
        &[] as &[Atom],
    )
    .optimization_settings(settings.clone())
    .build()
    .unwrap();
    let body = parse!("f(x)")
        .evaluator(&[parse!("x")])
        .add_function_with_options(
            symbol!("f"),
            vec![symbol!("x")],
            parse!("x^2+1"),
            FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
        )
        .unwrap()
        .build()
        .unwrap();
    let mut c = EvaluatorComposer::new(2);
    let a = c.append(&p, &[Slot::Param(0), Slot::Param(1)]).unwrap();
    // Rejected work after valid work must preserve its labels, constants and outputs.
    assert!(c.append(&p, &[]).is_err());
    for bad in [
        Slot::Out(0),
        Slot::Temp(usize::MAX),
        Slot::Const(usize::MAX),
        Slot::Param(2),
    ] {
        assert!(c.append(&p, &[bad, Slot::Param(0)]).is_err());
    }
    assert!(c.append(&body, &[Slot::Param(0)]).is_err());
    let b = c.append(&p, &[Slot::Param(1), Slot::Param(0)]).unwrap();
    let k = c.append(&constant, &[]).unwrap();
    let d = c.append(&dependent, &[a[1], b[1], a[2]]).unwrap();
    let selected = vec![
        d[0],
        a[0],
        b[0],
        a[1],
        b[1],
        a[2],
        b[2],
        k[0],
        k[1],
        Slot::Param(1),
        d[0],
    ];
    let exact = c.finish(&selected, settings.clone()).unwrap();
    let mut e = exact.clone().map_coeff(&|q| q.re.to_f64());
    let mut first = p.clone().map_coeff(&|q| q.re.to_f64());
    let mut second = p.clone().map_coeff(&|q| q.re.to_f64());
    let mut fixed = constant.map_coeff(&|q| q.re.to_f64());
    let mut fixed_out = [0.; 2];
    fixed.evaluate(&[], &mut fixed_out);
    #[cfg(feature = "native_code_generation")]
    let mut jit = exact
        .jit_compile::<f64>(
            JITCompilationSettings::default()
                .optimization_level(2)
                .with_option("use_threads", "false"),
        )
        .unwrap();
    for (x, y) in [
        (0., 0.),
        (2., 0.),
        (0., 3.),
        (2., 3.),
        (0., 0.),
        (2., 0.),
        (0., 3.),
    ] {
        let mut av = [0.; 3];
        let mut bv = [0.; 3];
        first.evaluate(&[x, y], &mut av);
        second.evaluate(&[y, x], &mut bv);
        let expected = [
            av[1] + bv[1] * av[2],
            av[0],
            bv[0],
            av[1],
            bv[1],
            av[2],
            bv[2],
            fixed_out[0],
            fixed_out[1],
            y,
            av[1] + bv[1] * av[2],
        ];
        let mut actual = [0.; 11];
        let mut j = [0.; 11];
        e.evaluate(&[x, y], &mut actual);
        #[cfg(feature = "native_code_generation")]
        jit.evaluate(&[x, y], &mut j);
        #[cfg(not(feature = "native_code_generation"))]
        j.copy_from_slice(&actual);
        for (i, ((a, b), v)) in actual.into_iter().zip(j).zip(expected).enumerate() {
            assert!(
                (a - v).abs() <= 1e-12 * v.abs().max(1.),
                "eager {i}: {a} {v}"
            );
            assert!((b - v).abs() <= 1e-12 * v.abs().max(1.), "jit {i}: {b} {v}");
        }
    }
    assert!(
        EvaluatorComposer::<Complex<Rational>>::new(0)
            .finish(&[Slot::Param(0)], settings.clone())
            .is_err()
    );
    let mut empty = EvaluatorComposer::<Complex<Rational>>::new(0)
        .finish(&[], settings)
        .unwrap()
        .map_coeff(&|q| q.re.to_f64());
    empty.evaluate(&[], &mut []);
    println!(
        "PASS: appended nested branch programs repeatedly toggle both conditions; reordered/fanout/repeated outputs; dynamic gamma and precision-aware gamma constants; huge negative exact rational; zero-input constants; failed append atomicity after valid work (arity, all slot kinds, non-inlined function); empty output and invalid finish; eager and SymJIT O2 agree with independent native program calls."
    );
}

#[test]
fn selected_outputs_prune_dead_jets_and_callbacks_before_optimization() {
    use symbolica::{domains::dual::HyperDual, evaluate::Dualizer};
    let settings = OptimizationSettings::new()
        .horner_iterations(0)
        .cores(1)
        .cpe_iterations(Some(0));
    let original = Atom::evaluator_multiple(
        &[parse!("x+x^2+x^3+x^4"), parse!("gamma(x)+gamma(1/2)")],
        &[parse!("x")],
    )
    .optimization_settings(settings.clone())
    .build()
    .unwrap();
    let shape = (0..5).map(|i| vec![i]).collect::<Vec<_>>();
    let lowered = original
        .vectorize(&Dualizer::new(
            HyperDual::<Complex<Rational>>::new(shape),
            vec![],
        ))
        .unwrap();
    let original_instructions = lowered.export_instructions().instructions.len();
    let seed = Atom::evaluator_multiple(
        &[
            parse!("x"),
            Atom::num(1),
            Atom::num(0),
            Atom::num(0),
            Atom::num(0),
        ],
        &[parse!("x")],
    )
    .optimization_settings(settings.clone())
    .build()
    .unwrap();
    let mut composed = EvaluatorComposer::new(1);
    let input = composed.append(&seed, &[Slot::Param(0)]).unwrap();
    let output = composed.append(&lowered, &input).unwrap();
    let exact = composed
        .finish(&[output[0], output[1]], settings.clone())
        .unwrap();
    let exported = exact.export_instructions();
    assert!(exported.instructions.len() < original_instructions);
    assert!(exported.constant_functions.is_empty());
    let mut evaluator = exact.map_coeff(&|c| c.re.to_f64());
    for x in [0., 0.25, 1., 2.] {
        let mut values = [0.; 2];
        evaluator.evaluate(&[x], &mut values);
        assert_eq!(
            values,
            [
                x + x * x + x * x * x + x * x * x * x,
                1. + 2. * x + 3. * x * x + 4. * x * x * x
            ]
        );
    }

    // A dead callback-only owner contributes neither a constant nor a callback.
    let callback = parse!("gamma(1/2)")
        .evaluator(&[] as &[Atom])
        .optimization_settings(settings.clone())
        .build()
        .unwrap();
    let mut unused = EvaluatorComposer::new(1);
    unused.append(&callback, &[]).unwrap();
    let passthrough = unused.finish(&[Slot::Param(0)], settings).unwrap();
    let exported = passthrough.export_instructions();
    assert!(exported.instructions.iter().all(|instruction| matches!(
        instruction,
        symbolica::evaluate::Instruction::Assign(Slot::Out(0), Slot::Param(0))
    )));
    assert!(exported.constants.is_empty());
    assert!(exported.constant_functions.is_empty());
}

#[test]
fn repeated_constants_and_instructions_share_native_storage() {
    let settings = OptimizationSettings::new()
        .horner_iterations(0)
        .cores(1)
        .cpe_iterations(Some(0));
    let original = Atom::evaluator_multiple(
        &[
            parse!("(x+2)^3+gamma(1/2)"),
            Atom::num(0),
            parse!("gamma(1/3)"),
        ],
        &[parse!("x")],
    )
    .optimization_settings(settings.clone())
    .build()
    .unwrap();
    let mut c = EvaluatorComposer::new(1);
    let first = c.append(&original, &[Slot::Param(0)]).unwrap();
    let second = c.append(&original, &[Slot::Param(0)]).unwrap();
    let exact = c
        .finish(&[first[0], second[0], first[1], second[2]], settings)
        .unwrap();
    let exported = exact.export_instructions();
    assert_eq!(exported.constant_functions.len(), 2);
    assert_eq!(exported.constants.len(), original.get_constants().len());
    assert_eq!(exact.count_operations(), original.count_operations());
    let mut expected = original.map_coeff(&|c| c.re.to_f64());
    let mut actual = exact.map_coeff(&|c| c.re.to_f64());
    for x in [-2., 0., 2.] {
        let mut e = [0.; 3];
        let mut a = [0.; 4];
        expected.evaluate(&[x], &mut e);
        actual.evaluate(&[x], &mut a);
        assert_eq!(a, [e[0], e[0], 0., e[2]]);
    }
}
