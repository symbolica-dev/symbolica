use super::super::*;
use crate::{
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
    let (mut c, inputs) = EvaluatorComposer::new_with_inputs(2);
    let a = c.add(&p, &inputs).unwrap();
    // Invalid work after valid work must preserve labels, constants and outputs.
    assert_eq!(
        c.add(&p, &[]),
        Err(ComposeError::ArityMismatch {
            expected: 2,
            actual: 0
        })
    );
    let mut foreign = EvaluatorComposer::<Complex<Rational>>::new();
    let other = foreign.add_input();
    assert_eq!(
        c.add(&p, &[other, inputs[0]]),
        Err(ComposeError::ForeignSlot)
    );
    // Registered function bodies are now accepted and remain shared.
    c.add(&body, &[inputs[0]]).unwrap();
    let b = c.add(&p, &[inputs[1], inputs[0]]).unwrap();
    let k = c.add(&constant, &[]).unwrap();
    let d = c.add(&dependent, &[a[1], b[1], a[2]]).unwrap();
    let selected = vec![
        d[0], a[0], b[0], a[1], b[1], a[2], b[2], k[0], k[1], inputs[1], d[0],
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
    assert_eq!(
        EvaluatorComposer::<Complex<Rational>>::new()
            .finish(&[other], settings.clone())
            .err(),
        Some(ComposeError::ForeignSlot)
    );
    let mut empty = EvaluatorComposer::<Complex<Rational>>::new()
        .finish(&[], settings)
        .unwrap()
        .map_coeff(&|q| q.re.to_f64());
    empty.evaluate(&[], &mut []);
}

#[test]
fn selected_outputs_prune_dead_jets_and_callbacks_before_optimization() {
    use crate::{domains::dual::HyperDual, evaluate::Dualizer};
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
    let (mut composed, inputs) = EvaluatorComposer::new_with_inputs(1);
    let input = composed.add(&seed, &inputs).unwrap();
    let output = composed.add(&lowered, &input).unwrap();
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
    let (mut unused, inputs) = EvaluatorComposer::new_with_inputs(1);
    unused.add(&callback, &[]).unwrap();
    let passthrough = unused.finish(&inputs, settings).unwrap();
    let exported = passthrough.export_instructions();
    assert!(exported.instructions.iter().all(|instruction| matches!(
        instruction,
        crate::evaluate::Instruction::Assign(Slot::Out(0), Slot::Param(0))
    )));
    assert!(exported.constants.is_empty());
    assert!(exported.constant_functions.is_empty());
}

fn settings() -> OptimizationSettings {
    OptimizationSettings::new().horner_iterations(0).cores(1)
}

#[test]
fn dynamic_inputs_float_constants_and_stable_handles() {
    let sum = parse!("x+y")
        .evaluator(&[parse!("x"), parse!("y")])
        .build()
        .unwrap()
        .map_coeff(&|q| F64(q.re.to_f64()));
    let mut c = EvaluatorComposer::default();
    let x = c.add_input();
    let two = c.constant(F64(2.));
    let first = c.add(&sum, &[x, two]).unwrap();
    let y = c.add_input(); // Inputs may be added after constants and applications.
    let _unused = c.add_input();
    let second = c.add(&sum, &[first[0], y]).unwrap();
    let negative_zero = c.constant(F64(-0.0));
    let nan = c.constant(F64(f64::NAN));
    let mut e = c
        .finish(
            &[second[0], x, two, negative_zero, nan, second[0]],
            settings(),
        )
        .unwrap()
        .map_coeff(&|q| q.0);
    assert_eq!(e.get_input_len(), 3);
    let mut out = [0.; 6];
    e.evaluate(&[3., 4., 99.], &mut out);
    assert_eq!(&out[..3], &[9., 3., 2.]);
    assert!(out[3].is_sign_negative());
    assert!(out[4].is_nan());
    assert_eq!(out[5], 9.);
    assert_eq!(EvaluatorComposer::<F64>::new_with_inputs(0).1.len(), 0);

    let mut c = EvaluatorComposer::<Complex<Rational>>::new();
    let one = c.constant(Complex::new(Rational::from(1), Rational::from(0)));
    let another_one = c.constant(Complex::new(Rational::from(1), Rational::from(0)));
    let input = c.add_input();
    let mut e = c
        .finish(&[another_one, input, one], settings())
        .unwrap()
        .map_coeff(&|q| q.re.to_f64());
    assert_eq!(e.get_constants().len(), 1);
    let mut out = [0.; 3];
    e.evaluate(&[7.], &mut out);
    assert_eq!(out, [1., 7., 1.]);
}

#[test]
fn foreign_results_and_constants_cannot_alias_local_values() {
    let sum = parse!("x+y")
        .evaluator(&[parse!("x"), parse!("y")])
        .build()
        .unwrap();
    let (mut c, local) = EvaluatorComposer::new_with_inputs(2);
    let (mut foreign, other) = EvaluatorComposer::new_with_inputs(2);
    let local_result = c.add(&sum, &local).unwrap();
    let foreign_result = foreign.add(&sum, &other).unwrap();
    let foreign_constant = foreign.constant(Complex::<Rational>::default());
    for value in [other[0], foreign_result[0], foreign_constant] {
        assert_eq!(
            c.add(&sum, &[local[0], value]),
            Err(ComposeError::ForeignSlot)
        );
    }
    let mut e = c
        .finish(&local_result, settings())
        .unwrap()
        .map_coeff(&|q| q.re.to_f64());
    assert_eq!(e.evaluate_single(&[2., 3.]), 5.);
    assert_eq!(
        foreign.finish(&local_result, settings()).err(),
        Some(ComposeError::ForeignSlot)
    );
}

#[test]
fn pruning_removes_dead_branches_beside_live_nested_branches() {
    let program = Atom::evaluator_multiple(
        &[
            parse!("if(x,if(y,x+y,1/y),if(y,1/x,7))"),
            parse!("if(x,gamma(y)+gamma(1/2),if(y,gamma(x),gamma(1/3)))"),
            parse!("x+y"),
        ],
        &[parse!("x"), parse!("y")],
    )
    .optimization_settings(settings())
    .build()
    .unwrap();
    let (mut c, inputs) = EvaluatorComposer::new_with_inputs(2);
    let out = c.add(&program, &inputs).unwrap();
    c.add(&program, &[inputs[1], inputs[0]]).unwrap();
    let e = c.finish(&[out[0]], settings()).unwrap();
    assert!(e.external_fns.is_empty());
    assert!(e.export_instructions().constant_functions.is_empty());
    assert_eq!(
        e.instructions
            .iter()
            .filter(|(i, _)| matches!(i, super::Instr::IfElse(..)))
            .count(),
        3
    );
    let mut e = e.map_coeff(&|q| q.re.to_f64());
    #[cfg(feature = "native_code_generation")]
    let mut jit = e
        .clone()
        .jit_compile(
            JITCompilationSettings::default()
                .optimization_level(2)
                .with_option("use_threads", "false"),
        )
        .unwrap();
    for (x, y, expected) in [
        (0., 0., 7.),
        (2., 3., 5.),
        (0., 4., f64::INFINITY),
        (2., 0., f64::INFINITY),
        (0., 0., 7.),
    ] {
        assert_eq!(e.evaluate_single(&[x, y]), expected);
        #[cfg(feature = "native_code_generation")]
        {
            let mut actual = [0.];
            jit.evaluate(&[x, y], &mut actual);
            assert_eq!(actual[0], expected);
        }
    }
    // Selecting a branch-independent result removes every branch, including its
    // callbacks and precision-dependent constants from the same application.
    let (mut c, inputs) = EvaluatorComposer::new_with_inputs(2);
    let out = c.add(&program, &inputs).unwrap();
    let e = c.finish(&[out[2]], settings()).unwrap();
    assert!(e.external_fns.is_empty());
    assert!(
        e.instructions
            .iter()
            .all(|(i, _)| matches!(i, super::Instr::Add(..)))
    );
    let mut e = e.map_coeff(&|q| q.re.to_f64());
    assert_eq!(e.evaluate_single(&[2., 3.]), 5.);
}

#[test]
fn composition_preserves_vectorized_nested_branches() {
    use crate::{domains::dual::HyperDual, evaluate::Dualizer};

    let dualizer = Dualizer::new(
        HyperDual::<Complex<Rational>>::new(vec![vec![0], vec![1]]),
        vec![],
    );
    // Repeated z^2 exercises CSE after the inner branch's multiple joins;
    // the common y+z pair also exercises CPE across branch boundaries.
    let unconditional = parse!("z^2+y+z+1")
        .evaluator(&[parse!("y"), parse!("z")])
        .optimization_settings(settings())
        .build()
        .unwrap()
        .vectorize(&dualizer)
        .unwrap();

    for nested_on_true in [true, false] {
        let expression = if nested_on_true {
            parse!("if(x,if(y,x+y,x-y)+z^2+y+z,7)")
        } else {
            parse!("if(x,7,if(y,x+y,x-y)+z^2+y+z)")
        };
        let original = expression
            .evaluator(&[parse!("x"), parse!("y"), parse!("z")])
            .optimization_settings(settings())
            .build()
            .unwrap()
            .vectorize(&dualizer)
            .unwrap();
        for rounds in [Some(0), None] {
            let mut all_joins = 0;
            for selected in [vec![0, 1, 2, 3], vec![1, 2]] {
                let (mut c, inputs) = EvaluatorComposer::new_with_inputs(3);
                let zero = c.constant(Complex::<Rational>::default());
                let one = c.constant(Complex::new(Rational::from(1), Rational::from(0)));
                let mut out = c
                    .add(
                        &original,
                        &[inputs[0], zero, inputs[1], zero, inputs[2], one],
                    )
                    .unwrap();
                out.extend(
                    c.add(&unconditional, &[inputs[1], zero, inputs[2], one])
                        .unwrap(),
                );
                let selected_outputs: Vec<_> = selected.iter().map(|i| out[*i]).collect();
                let exact = c
                    .finish(&selected_outputs, settings().cpe_iterations(rounds))
                    .unwrap();
                let joins = exact
                    .instructions
                    .iter()
                    .filter(|(i, _)| matches!(i, Instr::Join(..)))
                    .count();
                if selected.len() == 4 {
                    all_joins = joins;
                } else {
                    assert!(joins < all_joins, "unused vector components must be pruned");
                }
                #[cfg(feature = "native_code_generation")]
                let mut jit = exact
                    .jit_compile::<f64>(
                        JITCompilationSettings::default()
                            .optimization_level(2)
                            .with_option("use_threads", "false"),
                    )
                    .unwrap();
                let mut e = exact.map_coeff(&|q| q.re.to_f64());
                #[cfg(feature = "native_code_generation")]
                let (source, library, mut cpp) = {
                    let base = std::env::temp_dir().join(format!(
                        "symbolica_composed_branches_{}_{nested_on_true}_{rounds:?}_{}",
                        std::process::id(),
                        selected.len(),
                    ));
                    let source = base.with_extension("cpp");
                    let library = base.with_extension("so");
                    let cpp = e
                        .export_cpp::<f64>(
                            &source,
                            "composed_branches",
                            ExportSettings::new().inline_asm(InlineASM::None),
                        )
                        .unwrap()
                        .compile(
                            &library,
                            CompileOptions::default()
                                .compiler(
                                    std::env::var("SYMBOLICA_TEST_CXX")
                                        .unwrap_or_else(|_| "g++".into()),
                                )
                                .fast_math(false)
                                .unsafe_math(false)
                                .arg("-std=c++17"),
                        )
                        .unwrap()
                        .load()
                        .unwrap();
                    (source, library, cpp)
                };
                for (x, y, z) in [(0., 0., 3.), (2., 1., 4.), (2., 0., 5.), (0., 1., 6.)] {
                    let nested = (x != 0.) == nested_on_true;
                    let value = if nested {
                        (if y == 0. { x - y } else { x + y }) + z * z + y + z
                    } else {
                        7.
                    };
                    let expected = [
                        value,
                        if nested { 2. * z + 1. } else { 0. },
                        z * z + y + z + 1.,
                        2. * z + 1.,
                    ];
                    let mut actual = vec![0.; selected.len()];
                    e.evaluate(&[x, y, z], &mut actual);
                    for (actual, selected) in actual.iter().zip(&selected) {
                        assert_eq!(*actual, expected[*selected], "x={x}, y={y}, z={z}");
                    }
                    #[cfg(feature = "native_code_generation")]
                    {
                        let mut compiled = vec![0.; selected.len()];
                        jit.evaluate(&[x, y, z], &mut compiled);
                        assert_eq!(compiled, actual, "JIT: x={x}, y={y}, z={z}");
                        cpp.evaluate(&[x, y, z], &mut compiled);
                        assert_eq!(compiled, actual, "C++: x={x}, y={y}, z={z}");
                    }
                }
                #[cfg(feature = "native_code_generation")]
                {
                    drop(cpp);
                    std::fs::remove_file(source).unwrap();
                    std::fs::remove_file(library).unwrap();
                }
            }
        }
    }
}

#[test]
fn shared_functions_remap_transitive_calls_and_reject_conflicts_atomically() {
    let make = |body| {
        parse!("outer(x)+outer(y)")
            .evaluator(&[parse!("x"), parse!("y")])
            .add_function_with_options(
                symbol!("inner"),
                vec![symbol!("t")],
                body,
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap()
            .add_function_with_options(
                symbol!("outer"),
                vec![symbol!("t")],
                parse!("inner(t)+gamma(t+1)+gamma(1/2)"),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap()
            .optimization_settings(settings())
            .build()
            .unwrap()
    };
    let original = make(parse!("t^2+1"));
    let conflicting = make(parse!("t^3+1"));
    let dead = parse!("dead(x)")
        .evaluator(&[parse!("x")])
        .add_function_with_options(
            symbol!("dead"),
            vec![symbol!("t")],
            parse!("t+777"),
            FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
        )
        .unwrap()
        .build()
        .unwrap();
    let (mut c, inputs) = EvaluatorComposer::new_with_inputs(2);
    c.add(&dead, &[inputs[0]]).unwrap(); // Force live function-table indices to move.
    let a = c.add(&original, &inputs).unwrap();
    assert!(matches!(
        c.add(&conflicting, &inputs),
        Err(ComposeError::ConflictingFunction { .. })
    ));
    let b = c.add(&original, &[inputs[1], inputs[0]]).unwrap();
    let root_instructions = original.instructions.clone();
    let original_bodies: Vec<_> = original
        .external_fns
        .iter()
        .map(|f| f.body.clone())
        .collect();
    let exact = c.finish(&[a[0], b[0]], settings()).unwrap();
    assert_eq!(
        exact
            .external_fns
            .iter()
            .filter(|f| f.body.is_some())
            .count(),
        2
    );
    assert!(
        !exact
            .external_fns
            .iter()
            .any(|f| f.symbol == symbol!("dead"))
    );
    assert_eq!(original.instructions, root_instructions);
    assert_eq!(
        original
            .external_fns
            .iter()
            .map(|f| f.body.clone())
            .collect::<Vec<_>>(),
        original_bodies
    );
    let mut independent = original.map_coeff(&|q| q.re.to_f64());
    let mut e = exact.clone().map_coeff(&|q| q.re.to_f64());
    #[cfg(feature = "native_code_generation")]
    let mut jit = exact
        .jit_compile::<f64>(
            JITCompilationSettings::default()
                .optimization_level(2)
                .with_option("use_threads", "false"),
        )
        .unwrap();
    for inputs in [[0., 2.], [1., 3.], [2., 0.], [0., 2.]] {
        let expected = independent.evaluate_single(&inputs);
        let mut out = [0.; 2];
        e.evaluate(&inputs, &mut out);
        assert_eq!(out, [expected, expected]);
        #[cfg(feature = "native_code_generation")]
        {
            let mut out = [0.; 2];
            jit.evaluate(&inputs, &mut out);
            assert!((out[0] - expected).abs() < 1e-12 * expected.abs().max(1.));
            assert!((out[1] - expected).abs() < 1e-12 * expected.abs().max(1.));
        }
    }
}

#[test]
fn finish_eliminates_common_instructions_and_pairs_across_applications() {
    let params = [parse!("x"), parse!("y"), parse!("z"), parse!("w")];
    let first = parse!("x+y+z")
        .evaluator(&params)
        .optimization_settings(settings())
        .build()
        .unwrap();
    let second = parse!("x+z+w")
        .evaluator(&params)
        .optimization_settings(settings())
        .build()
        .unwrap();
    let compose = |rounds| {
        let (mut c, inputs) = EvaluatorComposer::new_with_inputs(4);
        let a = c.add(&first, &inputs).unwrap();
        let duplicate = c.add(&first, &inputs).unwrap();
        let b = c.add(&second, &inputs).unwrap();
        c.finish(
            &[a[0], duplicate[0], b[0]],
            settings().cpe_iterations(rounds),
        )
        .unwrap()
    };
    let no_cpe = compose(Some(0));
    // CSE merges identical applications even when CPE is disabled.
    assert_eq!(no_cpe.count_operations().additions, 4);
    let optimized = compose(None);
    // CPE can now see the shared x+z pair in the restored n-ary instructions.
    assert_eq!(optimized.count_operations().additions, 3);
    let mut e = optimized.map_coeff(&|q| q.re.to_f64());
    let mut out = [0.; 3];
    e.evaluate(&[2., 3., 5., 7.], &mut out);
    assert_eq!(out, [10., 10., 14.]);
}

#[test]
fn pruning_preserves_lazy_callbacks_in_live_branches() {
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    let calls = Arc::new(AtomicUsize::new(0));
    let mut original = parse!("if(x,gamma(y),7)")
        .evaluator(&[parse!("x"), parse!("y")])
        .build()
        .unwrap()
        .map_coeff(&|q| F64(q.re.to_f64()));
    let observed = calls.clone();
    original
        .external_fns
        .iter_mut()
        .find(|f| f.constant_index.is_none())
        .unwrap()
        .imp = Some(Box::new(move |args: &[F64]| {
        observed.fetch_add(1, Ordering::Relaxed);
        assert!(args[0].0 > 0., "inactive branch callback was evaluated");
        F64(2. * args[0].0)
    }));
    let (mut c, inputs) = EvaluatorComposer::new_with_inputs(2);
    let out = c.add(&original, &inputs).unwrap();
    let mut e = c.finish(&out, settings()).unwrap();
    assert_eq!(e.evaluate_single(&[F64(0.), F64(-100.)]).0, 7.);
    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(e.evaluate_single(&[F64(1.), F64(3.)]).0, 6.);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
    assert_eq!(e.evaluate_single(&[F64(0.), F64(-100.)]).0, 7.);
    assert_eq!(calls.load(Ordering::Relaxed), 1);
}

#[test]
fn finish_automatically_interns_constants_for_cross_evaluator_cse() {
    let original = Atom::evaluator_multiple(&[parse!("x+2"), parse!("gamma(1/2)")], &[parse!("x")])
        .optimization_settings(settings())
        .build()
        .unwrap();
    let exported = original.export_instructions();
    let callback = &exported.constant_functions[0];
    let placeholder = exported.constants[callback.index].clone();
    let (mut c, inputs) = EvaluatorComposer::new_with_inputs(1);
    let a = c.add(&original, &inputs).unwrap();
    let b = c.add(&original, &inputs).unwrap();
    // A literal matching a callback's placeholder must stay distinct from it.
    let literal = c.constant(placeholder.clone());
    let two = c.constant(Complex::new(Rational::from(2), Rational::from(0)));
    let exact = c
        .finish(
            &[b[1], literal, a[0], b[0], two, a[1]],
            settings().cpe_iterations(Some(0)),
        )
        .unwrap();
    // Equal constants from separate applications enable CSE without any opt-in.
    assert_eq!(exact.count_operations().additions, 1);
    assert_eq!(exact.get_constants().len(), 3);
    assert_eq!(exact.export_instructions().constant_functions.len(), 1);
    let mut expected = original.map_coeff(&|q| q.re.to_f64());
    let mut e = exact.map_coeff(&|q| q.re.to_f64());
    for x in [0., 3., -2.] {
        let mut reference = [0.; 2];
        expected.evaluate(&[x], &mut reference);
        let mut out = [0.; 6];
        e.evaluate(&[x], &mut out);
        assert_eq!(
            out,
            [
                reference[1],
                placeholder.re.to_f64(),
                reference[0],
                reference[0],
                2.,
                reference[1]
            ]
        );
    }
}
