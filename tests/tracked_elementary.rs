use std::collections::HashMap;

use symbolica::{
    atom::{Atom, AtomCore, AtomView},
    coefficient::Coefficient,
    domains::float::{Complex, ErrorPropagatingFloat, Float, RealLike},
    parse,
};

type Tracked = ErrorPropagatingFloat<f64>;
type TrackedFloat = ErrorPropagatingFloat<Float>;

const FUNCTIONS: [&str; 22] = [
    "tan", "cot", "sec", "csc", "asin", "acos", "atan", "acot", "asec", "acsc", "sinh", "cosh",
    "tanh", "coth", "sech", "csch", "asinh", "acosh", "atanh", "acoth", "asech", "acsch",
];

fn tracked(re: f64, im: f64, re_error: f64, im_error: f64) -> Complex<Tracked> {
    Complex::new(
        Tracked::new_with_absolute_error(re, re_error),
        Tracked::new_with_absolute_error(im, im_error),
    )
}

fn multiprecision(value: Complex<Tracked>) -> Complex<TrackedFloat> {
    Complex::new(
        TrackedFloat::new_with_error_magnitude(
            Float::with_val(128, value.re.value),
            value.re.get_error_magnitude(),
        ),
        TrackedFloat::new_with_error_magnitude(
            Float::with_val(128, value.im.value),
            value.im.get_error_magnitude(),
        ),
    )
}

fn normalized_value(expression: &Atom) -> Complex<TrackedFloat> {
    let result = expression.to_float(35);
    let AtomView::Num(number) = result.as_view() else {
        panic!("did not evaluate {expression}: {result}");
    };
    let Coefficient::Float(value) = number.get_coeff_view().to_owned() else {
        panic!("expected tracked float for {expression}");
    };
    value
}

#[test]
fn complex_elementary_evaluators_propagate_both_component_errors() {
    let x = parse!("x");
    for name in FUNCTIONS {
        let expression = parse!(&format!("{name}(x)"));
        let input = tracked(0.4, 0.2, 1e-6, 2e-6);
        let value = expression
            .evaluate(&HashMap::from([(x.clone(), input)]))
            .unwrap();
        let wide = expression
            .evaluate_with_prec(&HashMap::from([(x.clone(), multiprecision(input))]), 128)
            .unwrap();
        let normalized = normalized_value(
            &expression
                .replace(x.clone())
                .with(Atom::num(multiprecision(input))),
        );
        for (label, re, im, re_error, im_error) in [
            (
                "f64",
                value.re.value,
                value.im.value,
                value.re.get_absolute_error(),
                value.im.get_absolute_error(),
            ),
            (
                "Float",
                wide.re.value.to_f64(),
                wide.im.value.to_f64(),
                wide.re.get_absolute_error(),
                wide.im.get_absolute_error(),
            ),
            (
                "to_float",
                normalized.re.value.to_f64(),
                normalized.im.value.to_f64(),
                normalized.re.get_absolute_error(),
                normalized.im.get_absolute_error(),
            ),
        ] {
            assert!(re.is_finite() && im.is_finite(), "{name} {label}");
            assert!(
                re_error.is_finite() && re_error > 0.,
                "{name} {label}: {re_error}"
            );
            assert!(
                im_error.is_finite() && im_error > 0.,
                "{name} {label}: {im_error}"
            );
            for dr in [-1e-6, 0., 1e-6] {
                for di in [-2e-6, 0., 2e-6] {
                    let perturbed = expression
                        .evaluate(&HashMap::from([(
                            x.clone(),
                            Complex::new(0.4 + dr, 0.2 + di),
                        )]))
                        .unwrap();
                    assert!(
                        (re - perturbed.re).abs() <= re_error + 1e-13,
                        "{name} {label} real"
                    );
                    assert!(
                        (im - perturbed.im).abs() <= im_error + 1e-13,
                        "{name} {label} imaginary"
                    );
                }
            }
        }
    }
}

#[test]
fn real_inputs_keep_exact_imaginary_zeros_without_erasing_uncertain_zeros() {
    let x = parse!("x");
    for name in FUNCTIONS {
        let expression = parse!(&format!("{name}(x)"));
        let center = match name {
            "asec" | "acsc" | "acoth" => -2.,
            "acosh" => 2.,
            "asech" => 0.4,
            _ => -0.4,
        };
        let expected = expression
            .evaluate(&HashMap::from([(x.clone(), center)]))
            .unwrap();
        for im_error in [0., 1e-6] {
            let value = expression
                .evaluate(&HashMap::from([(
                    x.clone(),
                    tracked(center, 0., 1e-6, im_error),
                )]))
                .unwrap();
            assert!((value.re.value - expected).abs() < 1e-13, "{name}");
            if im_error == 0. {
                assert!(value.im.is_exact_zero(), "{name}");
            } else {
                assert!(value.im.get_absolute_error() > 0., "{name}");
            }
        }
    }
}

#[test]
fn inverse_functions_continue_to_the_complex_domain() {
    let x = parse!("x");
    for (name, center) in [
        ("asin", 2.),
        ("acos", -2.),
        ("asec", 0.4),
        ("acsc", -0.4),
        ("acosh", -2.),
        ("atanh", 2.),
        ("acoth", 0.4),
        ("asech", -0.4),
    ] {
        let expression = parse!(&format!("{name}(x)"));
        let value = expression
            .evaluate(&HashMap::from([(x.clone(), tracked(center, 0., 1e-6, 0.))]))
            .unwrap();
        let expected = expression
            .evaluate(&HashMap::from([(x.clone(), Complex::new(center, 0.))]))
            .unwrap();
        assert!((value.re.value - expected.re).abs() < 1e-13, "{name} real");
        assert!(
            (value.im.value - expected.im).abs() < 1e-13,
            "{name} imaginary"
        );
        assert_ne!(value.im.value, 0., "{name}");
    }
}

#[test]
fn inverse_domain_boundary_uncertainty_is_complex() {
    let x = parse!("x");
    for name in ["asin", "acos", "asec", "acsc", "acosh", "asech"] {
        let expression = parse!(&format!("{name}(x)"));
        let input = tracked(1., 0., 0.01, 0.);
        let value = expression
            .evaluate(&HashMap::from([(x.clone(), input)]))
            .unwrap();
        assert!(value.im.get_absolute_error() > 0., "{name}");
        let wide = expression
            .evaluate_with_prec(&HashMap::from([(x.clone(), multiprecision(input))]), 128)
            .unwrap();
        assert!(wide.im.get_absolute_error() > 0., "{name} Float");
    }
}

#[test]
fn binary_atan_preserves_real_quadrants_and_complex_errors() {
    let x = parse!("x");
    let y = parse!("y");
    let expression = parse!("atan(x,y)");
    for re_x in [-1., 0., 1.] {
        for re_y in [-1., 1.] {
            let input = HashMap::from([
                (x.clone(), tracked(re_x, 0., 1e-6, 0.)),
                (y.clone(), tracked(re_y, 0., 2e-6, 0.)),
            ]);
            let value = expression.evaluate(&input).unwrap();
            assert!((value.re.value - re_y.atan2(re_x)).abs() < 1e-13);
            assert!(value.im.is_exact_zero());
            assert!(value.re.get_absolute_error() > 0.);
            let wide_input = input
                .into_iter()
                .map(|(key, value)| (key, multiprecision(value)))
                .collect::<HashMap<_, _>>();
            let wide = expression.evaluate_with_prec(&wide_input, 128).unwrap();
            assert!((wide.re.value.to_f64() - re_y.atan2(re_x)).abs() < 1e-13);
            assert!(wide.im.is_exact_zero());
            assert!(wide.re.get_absolute_error() > 0.);
        }
    }
    let input = HashMap::from([
        (x.clone(), tracked(-0.8, 0.2, 1e-6, 2e-6)),
        (y.clone(), tracked(0.4, -0.3, 2e-6, 1e-6)),
    ]);
    let value = expression.evaluate(&input).unwrap();
    let expected = expression
        .evaluate(&HashMap::from([
            (x, Complex::new(-0.8, 0.2)),
            (y, Complex::new(0.4, -0.3)),
        ]))
        .unwrap();
    assert!((value.re.value - expected.re).abs() < 1e-13);
    assert!((value.im.value - expected.im).abs() < 1e-13);
    assert!(value.re.get_absolute_error() > 0.);
    assert!(value.im.get_absolute_error() > 0.);

    for expression in [parse!("atan(-1.0,1)"), parse!("atan(-1,1.0)")] {
        assert!(matches!(expression.as_view(), AtomView::Num(_)));
        let value = normalized_value(&expression);
        assert!((value.re.value.to_f64() - 1_f64.atan2(-1.)).abs() < 1e-13);
        assert!(value.im.is_exact_zero());
    }
}
