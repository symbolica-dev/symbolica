use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

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

fn close(actual: f64, expected: f64) {
    assert!(
        (actual - expected).abs() <= 1e-12 * (1. + expected.abs()),
        "{actual} != {expected}"
    );
}

#[test]
fn repeated_and_nested_function_arguments_are_factored_and_shared() {
    let parameters = [parse!("a"), parse!("b"), parse!("x")];
    let outputs = [
        parse!("sin(a*x+b*x)"),
        parse!("cos(a*x+b*x)"),
        parse!("sin(sin(a*x+b*x)+cos(a*x+b*x))"),
    ];
    for (direct, use_cpe, iterations) in configurations() {
        let mut evaluator = Atom::evaluator_multiple(&outputs, &parameters)
            .optimization_settings(settings(direct, use_cpe, iterations))
            .build()
            .unwrap()
            .map_coeff(&|c| c.re.to_f64());
        // The common argument must use one multiplication after Horner and
        // appear only once even though it is used at different nesting depths.
        if direct {
            assert_eq!(
                evaluator.count_operations(),
                OperationCount::new(2, if iterations == 0 { 2 } else { 1 }, 0, 3),
                "direct={direct}, use_cpe={use_cpe}, iterations={iterations}"
            );
        }
        for [a, b, x] in [[2., 3., 0.25], [-3., 0.5, -2.], [7., -7., 3.]] {
            let argument: f64 = (a + b) * x;
            let expected = [
                argument.sin(),
                argument.cos(),
                (argument.sin() + argument.cos()).sin(),
            ];
            let mut actual = [0.; 3];
            evaluator.evaluate(&[a, b, x], &mut actual);
            for (actual, expected) in actual.into_iter().zip(expected) {
                close(actual, expected);
            }
        }
    }
}

#[test]
fn tagged_calls_keep_symbolic_tags_and_optimize_only_runtime_arguments() {
    let tag = parse!("horner_tag::a+horner_tag::a*horner_tag::b");
    let function = symbol!("horner_tag::evaluate");
    let source = function.call((&tag, parse!("a*x+b*x")));
    let parameters = [parse!("a"), parse!("b"), parse!("x")];
    let mut functions = FunctionMap::new();
    functions
        .add_tagged_function_with_options(
            function,
            vec![tag.clone()],
            vec![symbol!("horner_tag::z")],
            parse!("horner_tag::z^2+1"),
            FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
        )
        .unwrap();
    for (direct, use_cpe, iterations) in configurations() {
        let mut evaluator = source
            .evaluator(&parameters)
            .function_map(functions.clone())
            .optimization_settings(settings(direct, use_cpe, iterations))
            .build()
            .unwrap()
            .map_coeff(&|c| c.re.to_f64());
        // The tag's symbols are intentionally absent from the input list:
        // they identify the definition and must never become runtime inputs.
        assert_eq!(
            evaluator.count_operations(),
            OperationCount::new(1, if iterations == 0 { 2 } else { 1 }, 0, 1),
            "direct={direct}, use_cpe={use_cpe}, iterations={iterations}"
        );
        let exported = evaluator.export_instructions();
        let body = exported
            .sub_evaluators
            .iter()
            .find(|body| body.symbol == function)
            .unwrap();
        assert_eq!(body.tags, [tag.to_canonical_string()]);
        for [a, b, x] in [[2., 3., 0.25], [-3., 0.5, -2.], [7., -7., 3.]] {
            let argument = (a + b) * x;
            assert_eq!(
                evaluator.evaluate_single(&[a, b, x]),
                argument * argument + 1.
            );
        }
    }
}

#[test]
fn power_children_are_optimized_without_losing_integer_power_factoring() {
    let parameters = [
        parse!("a"),
        parse!("b"),
        parse!("x"),
        parse!("c"),
        parse!("d"),
        parse!("y"),
    ];
    for (direct, use_cpe, iterations) in configurations() {
        let mut evaluator = parse!("(a*x+b*x)^(c*y+d*y)")
            .evaluator(&parameters)
            .optimization_settings(settings(direct, use_cpe, iterations))
            .build()
            .unwrap()
            .map_coeff(&|c| c.re.to_f64());
        if direct {
            assert_eq!(
                evaluator.count_operations(),
                OperationCount::new(2, if iterations == 0 { 4 } else { 2 }, 0, 1),
                "direct={direct}, use_cpe={use_cpe}, iterations={iterations}"
            );
        }
        for [a, b, x, c, d, y] in [[2., 3., 0.5, 1., 2., 0.25], [1., 0.5, 2., -3., 1., 0.5]] {
            let base: f64 = (a + b) * x;
            let exponent = (c + d) * y;
            close(
                evaluator.evaluate_single(&[a, b, x, c, d, y]),
                base.powf(exponent),
            );
        }

        let mut squared = parse!("(a*x+b*x)^2")
            .evaluator(&parameters[..3])
            .optimization_settings(settings(direct, use_cpe, iterations))
            .build()
            .unwrap()
            .map_coeff(&|c| c.re.to_f64());
        if direct {
            // Atom normalization distributes an integer power over the
            // factored product: (x*(a+b))^2 becomes x^2*(a+b)^2.
            assert_eq!(
                squared.count_operations(),
                OperationCount::new(1, 3, 0, 0),
                "direct={direct}, use_cpe={use_cpe}, iterations={iterations}"
            );
        }
        assert_eq!(squared.evaluate_single(&[2., 3., 0.5]), 6.25);

        let mut affine_squared = parse!("(a*x+b*x+c)^2")
            .evaluator(&parameters[..4])
            .optimization_settings(settings(direct, use_cpe, iterations))
            .build()
            .unwrap()
            .map_coeff(&|c| c.re.to_f64());
        if direct {
            assert_eq!(
                affine_squared.count_operations(),
                OperationCount::new(2, if iterations == 0 { 3 } else { 2 }, 0, 0),
                "direct={direct}, use_cpe={use_cpe}, iterations={iterations}"
            );
        }
        assert_eq!(affine_squared.evaluate_single(&[2., 3., 0.5, 1.]), 12.25);

        let mut polynomial = parse!("x^2+x")
            .evaluator(&[parse!("x")])
            .optimization_settings(settings(direct, use_cpe, iterations))
            .build()
            .unwrap()
            .map_coeff(&|c| c.re.to_f64());
        assert_eq!(
            polynomial.count_operations(),
            OperationCount::new(1, 1, 0, 0),
            "direct={direct}, use_cpe={use_cpe}, iterations={iterations}"
        );
        if direct && iterations > 0 {
            let exported = polynomial.export_instructions();
            assert!(
                exported.instructions.iter().any(|instruction| {
                    matches!(instruction, Instruction::Mul(_, inputs, _)
                        if inputs.len() == 2 && inputs[0] != inputs[1])
                }),
                "x^2+x should become x*(x+1): {:?}",
                exported.instructions
            );
        }
        for x in [-2., 0., 0.5, 3.] {
            assert_eq!(polynomial.evaluate_single(&[x]), x * (x + 1.));
        }
    }
}

#[test]
fn rewritten_function_calls_can_match_a_supplied_input() {
    let source = parse!("sin(x+x*y)");
    let parameters = [parse!("sin(x*(1+y))"), parse!("x"), parse!("y")];
    for use_cpe in [false, true] {
        for iterations in [0, 1, 10] {
            let mut evaluator = source
                .evaluator(&parameters)
                .optimization_settings(settings(true, use_cpe, iterations))
                .build()
                .unwrap()
                .map_coeff(&|c| c.re.to_f64());
            let actual = evaluator.evaluate_single(&[99., 1., 1.]);
            if iterations == 0 {
                close(actual, 2_f64.sin());
            } else {
                // Rebuilding sin after optimizing its argument makes it
                // exactly equal to the supplied function-shaped input.
                assert_eq!(actual, 99., "use_cpe={use_cpe}, iterations={iterations}");
            }
        }
    }
}

#[test]
fn nested_search_cancellation_is_latched_across_later_regions() {
    let source = parse!("sin(x*y+x*z+y*z+x*y*z)+cos(x*y+x*z+y*z+2*x*y*z)");
    let parameters = [parse!("x"), parse!("y"), parse!("z")];
    for use_cpe in [false, true] {
        let polls = Arc::new(AtomicUsize::new(0));
        let callback_polls = polls.clone();
        let settings = settings(true, use_cpe, 10).abort_check(Some(Box::new(move || {
            callback_polls.fetch_add(1, Ordering::Relaxed) == 0
        })));
        let mut evaluator = source
            .evaluator(&parameters)
            .optimization_settings(settings)
            .build()
            .unwrap()
            .map_coeff(&|c| c.re.to_f64());
        assert_eq!(polls.load(Ordering::Relaxed), 1, "use_cpe={use_cpe}");
        for [x, y, z] in [[2., 3., 4.], [-2., 0.5, 0.25]] {
            let common: f64 = x * y + x * z + y * z;
            let product = x * y * z;
            close(
                evaluator.evaluate_single(&[x, y, z]),
                (common + product).sin() + (common + 2. * product).cos(),
            );
        }
    }
}

#[test]
fn shared_alias_arguments_keep_root_and_local_bindings_distinct() {
    let parameters = [parse!("a"), parse!("b"), parse!("x"), parse!("u")];
    let alias = parse!("horner_region_scope::value");
    let source = parse!("horner_region_scope::value+horner_region_scope::outer(u)");
    for policy in [InliningPolicy::Always, InliningPolicy::Never] {
        for (direct, use_cpe, iterations) in configurations() {
            let mut evaluator = source
                .evaluator(&parameters)
                .add_aliases([(alias.clone(), parse!("sin(a*x+b*x)"))])
                .unwrap()
                .add_function_with_options(
                    symbol!("horner_region_scope::outer"),
                    vec![symbol!("x")],
                    alias.clone(),
                    FunctionRegistrationOptions::new().inlining(policy),
                )
                .unwrap()
                .optimization_settings(settings(direct, use_cpe, iterations))
                .build()
                .unwrap()
                .map_coeff(&|c| c.re.to_f64());
            for [a, b, x, u] in [[2., 3., 0.25, 0.5], [-3., 0.5, -2., 3.]] {
                let root: f64 = (a + b) * x;
                let local: f64 = (a + b) * u;
                close(
                    evaluator.evaluate_single(&[a, b, x, u]),
                    root.sin() + local.sin(),
                );
            }
        }
    }
}
