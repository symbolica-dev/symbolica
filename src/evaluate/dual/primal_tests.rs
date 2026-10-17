use super::*;
use crate::{domains::dual::HyperDual, parse, symbol};

type Evaluate = Box<dyn FnMut(&[Complex<f64>], &mut [Complex<f64>])>;

fn backend(mut evaluator: ExpressionEvaluator<Complex<f64>>, mode: usize) -> Evaluate {
    #[cfg(feature = "native_code_generation")]
    if mode != 0 {
        let mut evaluator = evaluator
            .jit_compile(
                JITCompilationSettings::new()
                    .optimization_level(2)
                    .direct_translation(mode == 2),
            )
            .unwrap();
        return Box::new(move |input, output| evaluator.evaluate(input, output));
    }
    assert_eq!(mode, 0);
    Box::new(move |input, output| evaluator.evaluate(input, output))
}

fn same_value(actual: Complex<f64>, expected: Complex<f64>) {
    for (actual, expected) in [(actual.re, expected.re), (actual.im, expected.im)] {
        assert!(
            actual == expected || (actual.is_nan() && expected.is_nan()),
            "{actual:?} != {expected:?}"
        );
    }
}

#[test]
fn row_vectorization_preserves_native_primal_products_and_jet_domains() {
    let x = parse!("primal_rows::x");
    let c = parse!("primal_rows::c");
    let source = (Symbol::IF.call_args([c.clone(), Atom::one(), Atom::Zero]) / &x)
        .evaluator(&[x, c])
        .build()
        .unwrap();
    let rows = vec![vec![Rational::zero()], vec![Rational::one()]];
    let coefficient =
        |value: &Complex<Rational>| Complex::new(value.re.to_f64(), value.im.to_f64());
    for shape in [
        vec![vec![0]],
        vec![vec![0], vec![1]],
        vec![vec![0], vec![1], vec![2]],
        vec![vec![0], vec![1], vec![2], vec![3]],
        vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![1, 1]],
    ] {
        let width = shape.len();
        let baseline = source
            .clone()
            .vectorize(&Dualizer::new(
                HyperDual::<Complex<Rational>>::new(shape.clone()),
                (1..width).map(|component| (1, component)).collect(),
            ))
            .unwrap()
            .map_coeff(&coefficient);
        let (plan, prelude, body) =
            ParameterRowEvaluator::new(source.clone(), vec![1], rows.clone())
                .unwrap()
                .vectorize(&Dualizer::new(
                    HyperDual::<Complex<Rational>>::new(shape.clone()),
                    vec![],
                ))
                .unwrap()
                .map_coeff(&coefficient)
                .into_parts();
        for mode in 0..if cfg!(feature = "native_code_generation") {
            3
        } else {
            1
        } {
            let mut original = backend(source.clone().map_coeff(&coefficient), mode);
            let mut baseline = backend(baseline.clone(), mode);
            let mut prelude = backend(prelude.clone(), mode);
            let mut body = backend(body.clone(), mode);
            let mut workspace = ParameterRowWorkspace::default();
            for point in [0., 0.5, -0.5, 0., 0.5] {
                let mut input = vec![Complex::new(0., 0.); 2 * width];
                input[0].re = point;
                if width > 1 {
                    input[1].re = 1.;
                }
                let mut expected = vec![Complex::new(0., 0.); width];
                let mut primal = Complex::new(0., 0.);
                for coefficient in [0., 1.] {
                    input[width].re = coefficient;
                    let mut output = vec![Complex::new(0., 0.); width];
                    baseline(&input, &mut output);
                    for (sum, value) in expected.iter_mut().zip(output) {
                        *sum += value;
                    }
                    let mut output = [Complex::new(0., 0.)];
                    original(&[input[0], input[width]], &mut output);
                    primal += output[0];
                }
                let mut output = vec![Complex::new(0., 0.); width];
                plan.evaluate_with(
                    &input[..width],
                    &mut output,
                    &mut workspace,
                    |value| Complex::new(value.to_f64(), 0.),
                    |input, output| {
                        prelude(input, output);
                        Ok(())
                    },
                    |input, output| {
                        body(input, output);
                        Ok(())
                    },
                )
                .unwrap();
                same_value(output[0], primal);
                for (actual, expected) in output.iter().zip(expected) {
                    same_value(*actual, expected);
                }
                if point != 0. {
                    for (component, actual) in shape.iter().zip(output) {
                        let expected = if component.iter().skip(1).any(|power| *power != 0) {
                            0.
                        } else {
                            (-1.0f64).powi(component[0] as i32)
                                / point.powi(component[0] as i32 + 1)
                        };
                        same_value(actual, Complex::new(expected, 0.));
                    }
                }
            }
        }
    }
}

#[test]
fn native_and_retained_powf_preserve_the_scalar_primal() {
    let x = parse!("primal_powf::x");
    let p = parse!("primal_powf::p");
    let function = symbol!("primal_powf::function");
    let expression = x.pow(p.clone());
    let coefficient =
        |value: &Complex<Rational>| Complex::new(value.re.to_f64(), value.im.to_f64());
    for retained in [false, true] {
        let source = if retained {
            function
                .call_args([x.clone(), p.clone()])
                .evaluator(&[x.clone(), p.clone()])
                .add_function_with_options(
                    function,
                    vec![symbol!("primal_powf::x"), symbol!("primal_powf::p")],
                    expression.clone(),
                    FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
                )
                .unwrap()
                .build()
                .unwrap()
        } else {
            expression
                .evaluator(&[x.clone(), p.clone()])
                .build()
                .unwrap()
        };
        let dual = source
            .clone()
            .vectorize(&Dualizer::new(
                HyperDual::<Complex<Rational>>::new(vec![
                    vec![0, 0],
                    vec![1, 0],
                    vec![0, 1],
                    vec![1, 1],
                ]),
                vec![],
            ))
            .unwrap()
            .map_coeff(&coefficient);
        for mode in 0..if cfg!(feature = "native_code_generation") {
            3
        } else {
            1
        } {
            let mut scalar = backend(source.clone().map_coeff(&coefficient), mode);
            let mut dual = backend(dual.clone(), mode);
            for (x, p) in [(0., 0.), (2.0f64, 3.0f64), (4., 0.5)] {
                let mut primal = [Complex::new(0., 0.)];
                scalar(&[Complex::new(x, 0.), Complex::new(p, 0.)], &mut primal);
                let mut output = [Complex::new(0., 0.); 4];
                dual(
                    &[x, 1., 0., 0., p, 0., 1., 0.].map(|value| Complex::new(value, 0.)),
                    &mut output,
                );
                same_value(output[0], primal[0]);
                if x == 0. {
                    assert_eq!(output[0], Complex::new(1., 0.));
                } else {
                    let expected = [
                        x.powf(p),
                        p * x.powf(p - 1.),
                        x.powf(p) * x.ln(),
                        x.powf(p - 1.) * (1. + p * x.ln()),
                    ];
                    for (actual, expected) in output.iter().zip(expected) {
                        assert!(
                            (actual.re - expected).abs()
                                <= 128. * f64::EPSILON * expected.abs().max(1.)
                        );
                        assert_eq!(actual.im, 0.);
                    }
                }
            }
        }
    }
}
