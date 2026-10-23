use symbolica::evaluate::{
    FunctionRegistrationOptions, InliningPolicy, Instruction, OperationCount, OptimizationSettings,
};
use symbolica::prelude::*;

fn configurations() -> impl Iterator<Item = (bool, bool, usize)> {
    [false, true].into_iter().flat_map(|direct| {
        [false, true].into_iter().flat_map(move |use_cpe| {
            [0, 1, 10].map(move |iterations| (direct, use_cpe, iterations))
        })
    })
}

fn settings(direct: bool, use_cpe: bool, iterations: usize) -> OptimizationSettings {
    OptimizationSettings::new()
        .direct_translation(direct)
        .horner_use_cpe(use_cpe)
        .horner_iterations(iterations)
        .cores(1)
}

#[test]
fn single_function_inputs_survive_factoring_and_coefficient_cleanup() {
    let parameters = [
        parse!("horner_inputs::f(x+x*y)"),
        parse!("horner_inputs::h(2*x+2*y)"),
        parse!("x"),
        parse!("y"),
    ];
    // Each function occurs only once in its output. Its argument must remain
    // the input's exact identity even when the surrounding algebra is factored.
    let outputs = [
        parse!("(x+x*y)*horner_inputs::f(x+x*y)"),
        parse!("(6*x+10*y)*horner_inputs::h(2*x+2*y)"),
    ];
    for (direct, use_cpe, iterations) in configurations() {
        let mut evaluator = Atom::evaluator_multiple(&outputs, &parameters)
            .optimization_settings(settings(direct, use_cpe, iterations))
            .build()
            .unwrap()
            .map_coeff(&|c| c.re.to_f64());
        assert_eq!(evaluator.count_operations().function_calls, 0);
        for [f, h, x, y] in [[2., 7., 3., 4.], [-3., 0.5, -2., 0.25], [0., 2., 7., 0.]] {
            let mut actual = [0.; 2];
            evaluator.evaluate(&[f, h, x, y], &mut actual);
            assert_eq!(actual, [f * (x + x * y), h * (6. * x + 10. * y)]);
        }
    }
}

#[test]
fn nested_function_inputs_remain_independent_in_either_parameter_order() {
    let f = parse!("horner_nested_inputs::f(x+x*y)");
    let g = parse!("horner_nested_inputs::g(horner_nested_inputs::f(x+x*y))");
    let source = parse!(
        "(x+x*y)*horner_nested_inputs::f(x+x*y)\
         +(x+x*y)*horner_nested_inputs::g(horner_nested_inputs::f(x+x*y))"
    );
    for reverse in [false, true] {
        let parameters = if reverse {
            [g.clone(), f.clone(), parse!("x"), parse!("y")]
        } else {
            [f.clone(), g.clone(), parse!("x"), parse!("y")]
        };
        for (direct, use_cpe, iterations) in configurations() {
            let mut evaluator = source
                .evaluator(&parameters)
                .optimization_settings(settings(direct, use_cpe, iterations))
                .build()
                .unwrap()
                .map_coeff(&|c| c.re.to_f64());
            assert_eq!(evaluator.count_operations().function_calls, 0);
            for [f, g, x, y] in [
                [2., 7., 3., 4.],
                [-3., 0.5, -2., 0.25],
                [0., 2., 0.25, -0.5],
            ] {
                let point = if reverse { [g, f, x, y] } else { [f, g, x, y] };
                assert_eq!(evaluator.evaluate_single(&point), (f + g) * (x + x * y));
            }
        }
    }
}

#[test]
fn normalization_hooks_cannot_observe_temporary_parameter_symbols() {
    let guard = symbol!(
        "horner_guard_inputs::guard",
        norm = |view, out| {
            if let AtomView::Fun(function) = view
                && matches!(function.iter().next(), Some(AtomView::Var(_)))
            {
                out.to_num(0);
            }
        }
    );
    assert!(
        guard
            .call((parse!("horner_guard_inputs::probe"),))
            .is_zero()
    );
    let parameter = parse!("horner_guard_inputs::p(x+x*y)");
    let guarded = guard.call((&parameter,));
    let nested_guarded = guard.call((&guarded,));
    let factor = parse!("x+x*x");
    let outputs = [&guarded * &factor, &nested_guarded * &factor];
    let parameters = [parameter, parse!("x")];
    for (direct, use_cpe, iterations) in configurations() {
        let mut evaluator = Atom::evaluator_multiple(&outputs, &parameters)
            .add_function(
                guard,
                vec![symbol!("horner_guard_inputs::z")],
                parse!("horner_guard_inputs::z+1"),
            )
            .unwrap()
            .optimization_settings(settings(direct, use_cpe, iterations))
            .build()
            .unwrap()
            .map_coeff(&|c| c.re.to_f64());
        // Both enclosing custom-normalized calls must keep their original
        // function arguments until lowering; a placeholder Var would erase them.
        for ([p, x], expected) in [
            ([3., 2.], [24., 30.]),
            ([-2., 0.5], [-0.75, 0.]),
            ([7., -2.], [16., 18.]),
        ] {
            let mut actual = [0.; 2];
            evaluator.evaluate(&[p, x], &mut actual);
            assert_eq!(actual, expected);
        }
    }
}

#[test]
fn compound_local_arguments_and_aliases_keep_their_binding_scopes() {
    let parameter = parse!("horner_local_inputs::h(2*x+2*y)");
    let parameters = [parse!("u"), parse!("x"), parse!("y"), parameter.clone()];
    let declared: Vec<Indeterminate> = [parameter, parse!("x"), parse!("y")]
        .into_iter()
        .map(|p| p.try_into().unwrap())
        .collect();
    let alias = parse!("horner_local_inputs::value");
    let source = parse!("horner_local_inputs::outer(u,x+1,y+2)+horner_local_inputs::value");
    for policy in [InliningPolicy::Always, InliningPolicy::Never] {
        for (direct, use_cpe, iterations) in configurations() {
            let mut evaluator = source
                .evaluator(&parameters)
                .add_aliases([(
                    alias.clone(),
                    parse!("(x+x*y)*horner_local_inputs::h(2*x+2*y)"),
                )])
                .unwrap()
                .add_function_with_options(
                    symbol!("horner_local_inputs::inner"),
                    Vec::<Symbol>::new(),
                    alias.clone(),
                    FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
                )
                .unwrap()
                .add_function_with_options(
                    symbol!("horner_local_inputs::outer"),
                    declared.clone(),
                    parse!("horner_local_inputs::value+horner_local_inputs::inner()"),
                    FunctionRegistrationOptions::new().inlining(policy),
                )
                .unwrap()
                .optimization_settings(settings(direct, use_cpe, iterations))
                .build()
                .unwrap()
                .map_coeff(&|c| c.re.to_f64());
            // The alias in outer inherits its arguments; the alias in inner
            // and at the root sees the global inputs instead.
            for [u, x, y, h] in [[2., 3., 4., 11.], [-3., 0.5, -0.25, 7.]] {
                let expected = u * ((x + 1.) + (x + 1.) * (y + 2.)) + 2. * h * (x + x * y);
                assert_eq!(evaluator.evaluate_single(&[u, x, y, h]), expected);
            }
        }
    }
}

#[test]
fn temporary_parameter_names_do_not_capture_existing_symbols() {
    let source = parse!(
        "(x+x*y)*horner_collision::f(x+x*y)\
         +symbolica::horner_parameter_0*horner_collision::f(x+x*y)\
         +symbolica::horner_parameter_0"
    );
    let parameters = [
        parse!("horner_collision::f(x+x*y)"),
        parse!("x"),
        parse!("y"),
        parse!("symbolica::horner_parameter_0"),
    ];
    for (direct, use_cpe, iterations) in configurations() {
        let mut evaluator = source
            .evaluator(&parameters)
            .optimization_settings(settings(direct, use_cpe, iterations))
            .build()
            .unwrap()
            .map_coeff(&|c| c.re.to_f64());
        for [f, x, y, existing] in [[2., 3., 4., 11.], [-3., 0.5, -0.25, 7.]] {
            assert_eq!(
                evaluator.evaluate_single(&[f, x, y, existing]),
                f * (x + x * y + existing) + existing
            );
        }
    }
}

#[test]
fn ordinary_symbol_inputs_retain_exported_arithmetic_quality() {
    let source = parse!("a*x+b*x+c");
    let parameters = [parse!("a"), parse!("b"), parse!("c"), parse!("x")];
    for (direct, use_cpe, iterations) in configurations() {
        let mut evaluator = source
            .evaluator(&parameters)
            .optimization_settings(settings(direct, use_cpe, iterations))
            .build()
            .unwrap()
            .map_coeff(&|c| c.re.to_f64());
        let expected = OperationCount::new(2, if iterations == 0 { 2 } else { 1 }, 0, 0);
        assert_eq!(evaluator.count_operations(), expected);
        let exported = evaluator.export_instructions();
        assert_eq!((exported.input_count, exported.output_count), (4, 1));
        assert!(exported.sub_evaluators.is_empty());
        assert!(exported.constant_functions.is_empty());
        let mut exported_cost = OperationCount::default();
        for instruction in &exported.instructions {
            match instruction {
                Instruction::Add(_, inputs, _) => exported_cost.additions += inputs.len() - 1,
                Instruction::Mul(_, inputs, _) => exported_cost.multiplications += inputs.len() - 1,
                Instruction::Assign(..) => {}
                other => panic!("unexpected instruction in polynomial evaluator: {other:?}"),
            }
        }
        assert_eq!(exported_cost, expected);
        for [a, b, c, x] in [[2., 7., 3., 4.], [-3., 0.5, -2., 0.25]] {
            assert_eq!(evaluator.evaluate_single(&[a, b, c, x]), (a + b) * x + c);
        }
    }
}
