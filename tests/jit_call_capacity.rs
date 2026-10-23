#![cfg(feature = "native_code_generation")]

use symbolica::{
    evaluate::{FunctionMap, FunctionRegistrationOptions, InliningPolicy},
    prelude::*,
};

fn retained_call(arity: usize) -> ExpressionEvaluator<Complex<Rational>> {
    let parameters = (0..arity)
        .map(|index| parse!(&format!("formal_{index}")))
        .collect::<Vec<_>>();
    let body = parameters
        .iter()
        .fold(Atom::num(3), |sum, value| sum + value);
    let formals = parameters
        .into_iter()
        .map(|parameter| Indeterminate::try_from(parameter).unwrap())
        .collect::<Vec<_>>();
    let mut functions = FunctionMap::new();
    functions
        .add_function_with_options(
            symbol!("callee"),
            formals,
            body,
            FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
        )
        .unwrap();
    functions
        .add_function_with_options(
            symbol!("outer"),
            vec![symbol!("z")],
            symbol!("callee").call(&vec![parse!("z"); arity]),
            FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
        )
        .unwrap();
    parse!("outer(x)")
        .evaluator(&[parse!("x")])
        .function_map(functions)
        .build()
        .unwrap()
}

#[test]
fn jit_reserves_nested_call_arguments_including_constants() {
    for arity in [63, 64, 65, 2047] {
        let evaluator = retained_call(arity);
        for direct in [false, true] {
            let settings = JITCompilationSettings::new()
                .direct_translation(direct)
                .optimization_level(2)
                .with_option("num_args", "64");
            let mut real = evaluator.jit_compile::<f64>(settings.clone()).unwrap();
            let mut output = [0.];
            real.evaluate(&[2.], &mut output);
            assert_eq!(output, [2. * arity as f64 + 3.]);

            let mapped = evaluator
                .clone()
                .map_coeff(&|value| Complex::new(value.re.to_f64(), value.im.to_f64()));
            let mut complex = mapped.jit_compile(settings).unwrap();
            let mut output = [Complex::new(0., 0.)];
            complex.evaluate(&[Complex::new(2., 1.)], &mut output);
            assert_eq!(output, [Complex::new(2. * arity as f64 + 3., arity as f64)]);
        }
    }
}

#[test]
fn jit_without_calls_needs_no_argument_reservation() {
    let evaluator = parse!("x^2 + 1").evaluator(&[parse!("x")]).build().unwrap();
    for direct in [false, true] {
        let mut compiled = evaluator
            .jit_compile::<f64>(JITCompilationSettings::new().direct_translation(direct))
            .unwrap();
        let mut output = [0.];
        compiled.evaluate(&[3.], &mut output);
        assert_eq!(output, [10.]);
    }
}

#[test]
fn jit_rejects_calls_exceeding_backend_capacity() {
    // The constant in the retained body is also passed at the call site.
    let evaluator = retained_call(2048);
    for direct in [false, true] {
        let error = evaluator
            .jit_compile::<f64>(JITCompilationSettings::new().direct_translation(direct))
            .err()
            .expect("unsupported calls must fail before producing executable code");
        assert!(error.contains("call arguments"), "{error}");
    }
}
