use super::*;
use crate::{domains::dual::HyperDual, parse, symbol};

type Evaluate = Box<dyn FnMut(&[Complex<f64>], &mut [Complex<f64>])>;

fn backend(mut evaluator: ExpressionEvaluator<Complex<f64>>, mode: usize) -> Evaluate {
    #[cfg(feature = "native_code_generation")]
    if mode != 0 {
        let mut evaluator = evaluator
            .jit_compile(
                crate::evaluate::JITCompilationSettings::new()
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

#[test]
fn zero_component_assumptions_preserve_native_primal_products() {
    let source = parse!("native_primal::x*native_primal::y")
        .evaluator(&[parse!("native_primal::x"), parse!("native_primal::y")])
        .build()
        .unwrap();
    for shape in [vec![vec![0]], vec![vec![0], vec![1], vec![2]]] {
        let width = shape.len();
        let dual = source
            .clone()
            .vectorize(&Dualizer::new(
                HyperDual::<Complex<Rational>>::new(shape),
                (0..width).map(|component| (1, component)).collect(),
            ))
            .unwrap();
        let coefficient = |c: &Complex<Rational>| Complex::new(c.re.to_f64(), c.im.to_f64());
        for mode in 0..if cfg!(feature = "native_code_generation") {
            3
        } else {
            1
        } {
            let mut scalar = backend(source.clone().map_coeff(&coefficient), mode);
            let mut dual = backend(dual.clone().map_coeff(&coefficient), mode);
            for point in [3., f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
                let mut input = vec![Complex::new(0., 0.); 2 * width];
                input[0].re = point;
                let mut expected = [Complex::new(0., 0.)];
                scalar(&[input[0], input[width]], &mut expected);
                let mut output = vec![Complex::new(0., 0.); width];
                dual(&input, &mut output);
                same_value(output[0], expected[0]);
                if point.is_finite() {
                    assert_eq!(output[0], Complex::new(0., 0.));
                } else {
                    assert!(output[0].re.is_nan());
                }
            }
        }
    }
}
