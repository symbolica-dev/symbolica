use super::*;
use crate::{parse, symbol};

/// Exact powers of two keep the nonzero answers in the subnormal range while
/// the corresponding positive power overflows. These references use only the
/// elementary powers of i and 1+i, not the evaluator's inverse or power code.
fn range_cases(power: i64) -> Vec<(Complex<f64>, Complex<f64>)> {
    let exponent = match power {
        2 => 520,
        3 => 350,
        4 => 260,
        _ => unreachable!(),
    };
    let a = 2.0_f64.powi(exponent);
    let r = f64::from_bits(1_u64 << (1074 - exponent as i64 * power));
    let imaginary = match power {
        2 => Complex::new(-r, 0.),
        3 => Complex::new(0., r),
        4 => Complex::new(r, 0.),
        _ => unreachable!(),
    };
    let diagonal = match power {
        2 => Complex::new(0., -r / 2.),
        3 => Complex::new(-r / 4., -r / 4.),
        4 => Complex::new(-r / 4., 0.),
        _ => unreachable!(),
    };
    let parity = if power % 2 == 0 { 1. } else { -1. };
    vec![
        (Complex::new(a, 0.), Complex::new(r, 0.)),
        (Complex::new(-a, 0.), Complex::new(parity * r, 0.)),
        (Complex::new(0., a), imaginary),
        (
            Complex::new(0., -a),
            Complex::new(parity * imaginary.re, parity * imaginary.im),
        ),
        (Complex::new(a, a), diagonal),
        (Complex::new(a, -a), Complex::new(diagonal.re, -diagonal.im)),
        (
            Complex::new(-a, -a),
            Complex::new(parity * diagonal.re, parity * diagonal.im),
        ),
    ]
}

fn assert_range_value(actual: &Complex<f64>, expected: &Complex<f64>) {
    for (actual, expected) in [(actual.re, expected.re), (actual.im, expected.im)] {
        assert!(
            actual.is_finite() && (actual - expected).abs() <= 4. * f64::from_bits(1),
            "{actual:e} != {expected:e}"
        );
        if expected != 0. {
            assert_ne!(actual, 0., "a representable component was lost");
        }
    }
}

fn check_program(
    source: &ExpressionEvaluator<Complex<Rational>>,
    inputs: &[Vec<Complex<f64>>],
    expected: &[Complex<f64>],
) {
    let mut eager = source
        .clone()
        .map_coeff(&|c| Complex::new(c.re.to_f64(), c.im.to_f64()));
    for (input, expected) in inputs.iter().zip(expected) {
        assert_range_value(&eager.evaluate_single(input), expected);
    }
    #[cfg(feature = "native_code_generation")]
    for direct in [false, true] {
        let mut compiled = eager
            .jit_compile(
                JITCompilationSettings::new()
                    .optimization_level(2)
                    .direct_translation(direct),
            )
            .unwrap();
        for (input, expected) in inputs.iter().zip(expected) {
            let mut output = [Complex::new(0., 0.)];
            compiled.evaluate(input, &mut output);
            assert_range_value(&output[0], expected);
        }
    }
}

#[test]
fn negative_integer_powers_preserve_subnormals_in_both_lowerings() {
    let z = parse!("negative_power::z");
    for power in 2..=4 {
        let expression = z.pow(-power);
        let (inputs, expected): (Vec<_>, Vec<_>) = range_cases(power)
            .into_iter()
            .map(|(input, expected)| (vec![input], expected))
            .unzip();
        for direct in [false, true] {
            let source = expression
                .evaluator(std::slice::from_ref(&z))
                .direct_translation(direct)
                .build()
                .unwrap();
            check_program(&source, &inputs, &expected);
        }
    }
}

#[test]
fn negative_integer_powers_keep_retained_calls_and_inactive_poles_lazy() {
    let z = parse!("negative_power_guard::z");
    let q = parse!("negative_power_guard::q");
    let argument = symbol!("negative_power_guard::argument");
    let function = symbol!("negative_power_guard::function");
    let expression =
        Symbol::IF.call_args([q.clone(), function.call_args([z.clone()]), Atom::num(7)]);
    for power in 2..=4 {
        let (mut inputs, mut expected): (Vec<_>, Vec<_>) = range_cases(power)
            .into_iter()
            .map(|(input, expected)| (vec![input, Complex::new(1., 0.)], expected))
            .unzip();
        inputs.push(vec![Complex::new(0., 0.); 2]);
        expected.push(Complex::new(7., 0.));
        for policy in [InliningPolicy::Always, InliningPolicy::Never] {
            for direct in [false, true] {
                let source = expression
                    .evaluator(&[z.clone(), q.clone()])
                    .add_function_with_options(
                        function,
                        vec![argument],
                        Atom::var(argument).pow(-power),
                        FunctionRegistrationOptions::new().inlining(policy),
                    )
                    .unwrap()
                    .direct_translation(direct)
                    .build()
                    .unwrap();
                assert_eq!(
                    source
                        .external_fns
                        .iter()
                        .filter(|f| f.body.is_some())
                        .count(),
                    usize::from(policy == InliningPolicy::Never),
                );
                check_program(&source, &inputs, &expected);
            }
        }
    }
}

#[test]
fn direct_atom_tree_and_exact_ring_agree_on_negative_powers() {
    let z = parse!("negative_power_numeric::z");
    let ring = crate::domains::float::FloatField::from_rep(Complex::<Rational>::default());
    for power in 2..=4 {
        let expression = z.pow(-power);
        let tree = expression
            .as_view()
            .to_evaluation_tree(&FunctionMap::new(), std::slice::from_ref(&z))
            .unwrap();
        let mut numeric_tree = tree.map_coeff(&|c| Complex::new(c.re.to_f64(), c.im.to_f64()));
        for (input, expected) in range_cases(power) {
            let bindings = std::collections::HashMap::from([(z.clone(), input)]);
            assert_range_value(&expression.evaluate(&bindings).unwrap(), &expected);
            let mut output = [Complex::new(0., 0.)];
            numeric_tree.evaluate(&[input], &mut output);
            assert_range_value(&output[0], &expected);
        }
        let input = Complex::new(Rational::from(2), Rational::zero());
        let expected = Complex::new(Rational::from((1, 1_i64 << power)), Rational::zero());
        let bindings = std::collections::HashMap::from([(z.clone(), input.clone())]);
        assert_eq!(
            expression.evaluate_in_ring(&bindings, &ring).unwrap(),
            expected
        );
        for direct in [false, true] {
            let mut evaluator = expression
                .evaluator(std::slice::from_ref(&z))
                .direct_translation(direct)
                .build()
                .unwrap();
            let mut output = [Complex::default()];
            evaluator.evaluate_in_ring(std::slice::from_ref(&input), &mut output, &ring);
            assert_eq!(output[0], expected);
        }
    }
}

#[test]
fn native_negative_power_instructions_preserve_the_same_range() {
    // Imported evaluators can contain an integer Pow even when current source
    // lowering normally decomposes it. Exercise the native interpreter too.
    for power in 2..=4 {
        let source = ExpressionEvaluator {
            stack: vec![Complex::<Rational>::default(); 2],
            param_count: 1,
            reserved_indices: 1,
            instructions: vec![(Instr::Pow(1, 0, -power), ComplexPhase::Any)],
            result_indices: vec![1],
            external_fns: Vec::new(),
            settings: OptimizationSettings::default(),
        };
        let (inputs, expected): (Vec<_>, Vec<_>) = range_cases(power)
            .into_iter()
            .map(|(input, expected)| (vec![input], expected))
            .unzip();
        check_program(&source, &inputs, &expected);
    }
}

#[test]
#[cfg(feature = "native_code_generation")]
fn raw_negative_powers_keep_their_range_in_cpp_and_assembly() {
    let directory =
        std::env::temp_dir().join(format!("symbolica-negative-powers-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    for power in 2..=4 {
        let source = ExpressionEvaluator {
            stack: vec![Complex::<Rational>::default(); 2],
            param_count: 1,
            reserved_indices: 1,
            instructions: vec![(Instr::Pow(1, 0, -power), ComplexPhase::Any)],
            result_indices: vec![1],
            external_fns: Vec::new(),
            settings: OptimizationSettings::default(),
        };
        for (mode, assembly) in [InlineASM::None, InlineASM::default()]
            .into_iter()
            .enumerate()
        {
            let options = CompileOptions::default()
                .compiler(std::env::var("SYMBOLICA_TEST_CXX").unwrap_or_else(|_| "g++".into()))
                .fast_math(false)
                .unsafe_math(false)
                .arg("-std=c++17");
            let name = format!("real_power_{power}_{mode}");
            let path = directory.join(&name);
            let mut real = source
                .export_cpp::<f64>(
                    directory.join(format!("{name}.cpp")),
                    &name,
                    ExportSettings::new().inline_asm(assembly),
                )
                .unwrap()
                .compile(&path, options.clone())
                .unwrap()
                .load()
                .unwrap();
            for (input, expected) in range_cases(power).into_iter().take(2) {
                let mut output = [0.];
                real.evaluate(&[input.re], &mut output);
                assert_range_value(&Complex::new(output[0], 0.), &expected);
            }

            let name = format!("complex_power_{power}_{mode}");
            let path = directory.join(&name);
            let mut complex = source
                .export_cpp::<Complex<f64>>(
                    directory.join(format!("{name}.cpp")),
                    &name,
                    ExportSettings::new().inline_asm(assembly),
                )
                .unwrap()
                .compile(&path, options)
                .unwrap()
                .load()
                .unwrap();
            for (input, expected) in range_cases(power) {
                let mut output = [Complex::new(0., 0.)];
                complex.evaluate(&[input], &mut output);
                assert_range_value(&output[0], &expected);
            }
        }
    }
    std::fs::remove_dir_all(directory).unwrap();
}
