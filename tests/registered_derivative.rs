use symbolica::{
    atom::{Atom, AtomCore, Symbol},
    domains::{float::Complex, rational::Rational},
    evaluate::{ExpressionEvaluator, FunctionMap, FunctionRegistrationOptions, InliningPolicy},
    symbol,
};

fn build(policy: InliningPolicy) -> ExpressionEvaluator<Complex<Rational>> {
    let (x, y, f) = symbol!(
        "registered_derivative::x",
        "registered_derivative::y",
        "registered_derivative::f"
    );
    let inputs = [Atom::var(x), Atom::var(y)];
    let body = inputs[0].pow(2) * &inputs[1] + Atom::num(3) * &inputs[0] * inputs[1].pow(2);
    let call = f.call_args(inputs.clone());
    let mut functions = FunctionMap::new();
    let options = FunctionRegistrationOptions::new().inlining(policy);
    functions
        .add_function_with_options(f, vec![x, y], body.clone(), options.clone())
        .unwrap();
    for orders in [[1u32, 0u32], [1, 1], [0, 2], [1, 2]] {
        let mut derivative = body.clone();
        for (variable, order) in [x, y].into_iter().zip(orders) {
            for _ in 0..order {
                derivative = derivative.derivative(variable);
            }
        }
        functions
            .add_tagged_function_with_options(
                Symbol::DERIVATIVE,
                vec![Atom::num(orders[0]), Atom::num(orders[1]), Atom::var(f)],
                vec![x, y],
                derivative,
                options.clone(),
            )
            .unwrap();
    }
    Atom::evaluator_multiple(
        &[
            call.clone(),
            call.derivative(x),
            call.derivative(x).derivative(y),
            call.derivative(y).derivative(y),
            call.derivative(y).derivative(y).derivative(x),
        ],
        &inputs,
    )
    .function_map(functions)
    .horner_iterations(0)
    .build()
    .unwrap()
}

fn points() -> [[Complex<f64>; 2]; 3] {
    [
        [Complex::new(0., 0.), Complex::new(0., 0.)],
        [Complex::new(2., 1.), Complex::new(3., -2.)],
        [Complex::new(-0.5, 0.25), Complex::new(0.125, 0.5)],
    ]
}

fn expected([x, y]: [Complex<f64>; 2]) -> [Complex<f64>; 5] {
    let three = Complex::new(3., 0.);
    let two = Complex::new(2., 0.);
    let six = Complex::new(6., 0.);
    [
        x * x * y + three * x * y * y,
        two * x * y + three * y * y,
        two * x + six * y,
        six * x,
        six,
    ]
}

#[test]
fn registered_derivatives_respect_inlining_policy() {
    for policy in [InliningPolicy::Always, InliningPolicy::Never] {
        let exact = build(policy);
        let exported = exact.export_instructions();
        assert_eq!(exported.input_count, 2);
        assert_eq!(
            exported.sub_evaluators.is_empty(),
            policy == InliningPolicy::Always
        );
        let mut eager =
            exact.map_coeff(&|value| Complex::new(value.re.to_f64(), value.im.to_f64()));
        for point in points() {
            let mut actual = [Complex::new(0., 0.); 5];
            eager.evaluate(&point, &mut actual);
            assert_eq!(actual, expected(point));
        }
    }
}

#[cfg(all(feature = "bincode", feature = "native_code_generation"))]
#[test]
fn registered_derivatives_restore_owned_eager_and_jit_programs() {
    use symbolica::evaluate::{JITCompilationSettings, JITCompiledEvaluator};
    for policy in [InliningPolicy::Always, InliningPolicy::Never] {
        let exact = build(policy);
        let bytes = bincode::encode_to_vec(&exact, bincode::config::standard()).unwrap();
        drop(exact);
        let (restored, used): (ExpressionEvaluator<Complex<Rational>>, _) =
            bincode::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        assert_eq!(used, bytes.len());
        let jit = restored
            .jit_compile::<Complex<f64>>(
                JITCompilationSettings::default()
                    .optimization_level(2)
                    .with_option("use_threads", "false"),
            )
            .unwrap();
        let jit_bytes = bincode::encode_to_vec(&jit, bincode::config::standard()).unwrap();
        let mut eager =
            restored.map_coeff(&|value| Complex::new(value.re.to_f64(), value.im.to_f64()));
        drop(jit);
        let (mut jit, used): (JITCompiledEvaluator<Complex<f64>>, _) =
            bincode::decode_from_slice(&jit_bytes, bincode::config::standard()).unwrap();
        assert_eq!(used, jit_bytes.len());
        for point in points() {
            let mut eager_out = [Complex::new(0., 0.); 5];
            let mut jit_out = eager_out;
            eager.evaluate(&point, &mut eager_out);
            jit.evaluate(&point, &mut jit_out);
            assert_eq!(eager_out, expected(point));
            assert_eq!(jit_out, eager_out);
        }
    }
}

#[test]
fn ordinary_builtin_precedence_and_arity_are_unchanged() {
    let x = symbol!("registered_derivative_builtin::x");
    let input = Atom::var(x);
    for policy in [InliningPolicy::Always, InliningPolicy::Never] {
        let mut functions = FunctionMap::new();
        functions
            .add_function_with_options(
                Symbol::LOG,
                vec![x],
                Atom::num(999),
                FunctionRegistrationOptions::new().inlining(policy),
            )
            .unwrap();
        let log = Symbol::LOG.call_args([input.clone()]);
        let mut evaluator = log
            .evaluator(std::slice::from_ref(&input))
            .function_map(functions.clone())
            .build()
            .unwrap()
            .map_coeff(&|value| value.re.to_f64());
        assert_eq!(evaluator.evaluate_single(&[2.]), 2f64.ln());
        assert!(
            Symbol::LOG
                .call_args([input.clone(), input.clone()])
                .evaluator(std::slice::from_ref(&input))
                .function_map(functions)
                .build()
                .is_err()
        );
    }
    let f = symbol!("registered_derivative_builtin::unregistered");
    assert!(
        f.call_args([input.clone(), input.clone()])
            .derivative(x)
            .evaluator(&[input])
            .build()
            .is_err()
    );
}
