use std::collections::HashMap;

use symbolica::{
    atom::{Atom, AtomCore},
    domains::float::{Complex, ErrorPropagatingFloat, Float, Magnitude, RealLike, SingleFloat},
    parse,
};

type Tracked = ErrorPropagatingFloat<Float>;
type Tracked64 = ErrorPropagatingFloat<f64>;

const FUNCTIONS: &[(&str, f64)] = &[
    ("gamma(x)", 1.25),
    ("erf(x)", 1.25),
    ("zeta(x)", 1.25),
    ("polygamma(0,x)", 1.25),
    ("polygamma(2,x)", 1.25),
    ("polylog(2,x)", 0.375),
    ("bessel_j(0,x)", 1.25),
    ("bessel_y(0,x)", 1.25),
    ("bessel_i(0,x)", 1.25),
    ("bessel_k(0,x)", 1.25),
];

fn tracked(precision: u32, center: f64, error: f64) -> Tracked {
    Tracked::new_with_absolute_error(Float::with_val(precision, center), error)
}

fn reference(expression: &Atom, re: f64, im: f64) -> Complex<Float> {
    expression
        .evaluate_with_prec(
            &HashMap::from([(
                parse!("x"),
                Complex::new(Float::with_val(256, re), Float::with_val(256, im)),
            )]),
            256,
        )
        .unwrap()
}

fn covers(result: &Tracked, reference: &Float, context: &str) {
    assert!(reference.is_finite(), "invalid reference: {context}");
    let difference = (reference.clone() - result.get_num()).absolute_magnitude();
    assert!(
        difference <= result.get_error_magnitude(),
        "{context}: difference {difference:?} exceeds error {:?}, center {}",
        result.get_error_magnitude(),
        result.get_num()
    );
}

fn covers64(result: &Tracked64, reference: &Float, context: &str) {
    let result = Tracked::new_with_error_magnitude(
        Float::with_val(256, *result.get_num()),
        result.get_error_magnitude(),
    );
    covers(&result, reference, context);
}

#[test]
fn real_special_functions_propagate_uncertainty_in_both_float_domains() {
    let error = 0.001;
    for &(source, center) in FUNCTIONS {
        let expression = parse!(source);
        let result64 = expression
            .evaluate(&HashMap::from([(
                parse!("x"),
                Tracked64::new_with_absolute_error(center, error),
            )]))
            .unwrap();
        assert!(result64.get_error_magnitude().is_finite(), "{source}");
        assert!(!result64.is_exact(), "{source}");
        for precision in [80, 128] {
            let result = expression
                .evaluate_with_prec(
                    &HashMap::from([(parse!("x"), tracked(precision, center, error))]),
                    precision,
                )
                .unwrap();
            assert!(result.get_error_magnitude().is_finite(), "{source}");
            assert!(!result.is_exact(), "{source}");
            for step in [-1., -0.5, 0., 0.5, 1.] {
                let expected = reference(&expression, center + step * error, 0.);
                let context = format!("{source}, precision {precision}, step {step}");
                covers(&result, &expected.re, &context);
                covers64(&result64, &expected.re, &context);
            }
        }
    }
}

#[test]
fn complex_special_functions_propagate_errors_in_both_components() {
    let precision = 96;
    let re_error = 0.001;
    let im_error = 0.002;
    for &(source, center) in FUNCTIONS {
        let expression = parse!(source);
        let input = Complex::new(
            tracked(precision, center, re_error),
            tracked(precision, 0.25, im_error),
        );
        let result = expression
            .evaluate_with_prec(&HashMap::from([(parse!("x"), input)]), precision)
            .unwrap();
        let result64 = expression
            .evaluate(&HashMap::from([(
                parse!("x"),
                Complex::new(
                    Tracked64::new_with_absolute_error(center, re_error),
                    Tracked64::new_with_absolute_error(0.25, im_error),
                ),
            )]))
            .unwrap();
        for component in [&result.re, &result.im] {
            assert!(component.get_error_magnitude().is_finite(), "{source}");
            assert!(!component.is_exact(), "{source}");
        }
        for (re_step, im_step) in [(-1., -1.), (-1., 1.), (0., 0.), (1., -1.), (1., 1.)] {
            let expected = reference(
                &expression,
                center + re_step * re_error,
                0.25 + im_step * im_error,
            );
            covers(&result.re, &expected.re, source);
            covers(&result.im, &expected.im, source);
            covers64(&result64.re, &expected.re, source);
            covers64(&result64.im, &expected.im, source);
        }
    }
}

#[test]
fn optimized_evaluators_resolve_tracked_special_functions() {
    for direct in [false, true] {
        for &(source, center) in FUNCTIONS {
            let expression = parse!(source);
            let evaluator = expression
                .evaluator(&[parse!("x")])
                .direct_translation(direct)
                .horner_iterations(0)
                .cores(1)
                .build()
                .unwrap();
            let result = evaluator
                .clone()
                .try_map_coeff_with_prec(&|c| tracked(96, 0., 0.).from_rational(&c.re), 96)
                .unwrap()
                .evaluate_single(&[tracked(96, center, 0.001)]);
            let result64 = evaluator
                .map_coeff(&|c| Tracked64::new_with_absolute_error(c.re.to_f64(), 0.))
                .evaluate_single(&[Tracked64::new_with_absolute_error(center, 0.001)]);
            let expected = reference(&expression, center + 0.001, 0.);
            covers(&result, &expected.re, source);
            covers64(&result64, &expected.re, source);
        }
    }
}

#[test]
fn normalization_and_to_float_keep_special_function_uncertainty() {
    for &(source, center) in FUNCTIONS {
        let expression = parse!(source);
        let normalized = expression
            .replace(parse!("x"))
            .with(Atom::num(tracked(128, center, 0.001)));
        for result in [normalized.clone(), normalized.to_float(20)] {
            let number = result
                .as_num_view()
                .unwrap_or_else(|| panic!("{source} did not normalize: {result}"));
            let value = number.get_coeff_view().to_tracked_float(128).unwrap();
            assert!(value.im.is_exact_zero(), "{source}: {value}");
            assert!(value.re.get_error_magnitude().is_finite(), "{source}");
            assert!(!value.re.is_exact(), "{source}");
            covers(
                &value.re,
                &reference(&expression, center + 0.001, 0.).re,
                source,
            );
        }
        let exact_expression = expression.replace(parse!("x")).with(if center < 1. {
            parse!("3/8")
        } else {
            parse!("5/4")
        });
        let evaluated = exact_expression.to_float(24);
        assert!(evaluated.as_num_view().is_some(), "{source}: {evaluated}");
    }
}

#[test]
fn stationary_points_still_have_finite_nonzero_uncertainty() {
    for (source, center) in [
        ("gamma(x)", 1.4616321449683623),
        ("bessel_j(0,x)", 0.),
        ("bessel_i(0,x)", 0.),
    ] {
        let expression = parse!(source);
        let result = expression
            .evaluate_with_prec(
                &HashMap::from([(parse!("x"), tracked(128, center, 0.125))]),
                128,
            )
            .unwrap();
        assert!(result.get_error_magnitude().is_finite(), "{source}");
        assert!(!result.is_exact(), "{source}");
        for step in [-1., -0.5, 0.5, 1.] {
            covers(
                &result,
                &reference(&expression, center + step * 0.125, 0.).re,
                source,
            );
        }
    }
}

#[test]
fn neighborhoods_containing_poles_have_unbounded_error() {
    for (source, center) in [
        ("gamma(x)", 0.0625),
        ("polygamma(0,x)", 0.0625),
        ("polygamma(2,x)", -0.9375),
        ("zeta(x)", 1.0625),
        ("bessel_y(0,x)", 0.0625),
        ("bessel_k(0,x)", 0.0625),
    ] {
        let result = parse!(source)
            .evaluate_with_prec(
                &HashMap::from([(parse!("x"), tracked(96, center, 0.125))]),
                96,
            )
            .unwrap();
        assert!(
            result.get_error_magnitude().is_infinite(),
            "{source}: {result:?}"
        );
    }
}

#[test]
fn uncertain_imaginary_zero_retains_the_complex_branch_uncertainty() {
    for (source, center) in [
        ("polylog(2,x)", 1.5),
        ("bessel_y(0,x)", -1.5),
        ("bessel_k(0,x)", -1.5),
    ] {
        let expression = parse!(source);
        let result = expression
            .evaluate_with_prec(
                &HashMap::from([(
                    parse!("x"),
                    Complex::new(tracked(96, center, 0.), tracked(96, 0., 0.001)),
                )]),
                96,
            )
            .unwrap();
        assert!(!result.im.is_exact_zero(), "{source}");
        for side in [-0.001, 0.001] {
            let expected = reference(&expression, center, side);
            covers(&result.re, &expected.re, source);
            covers(&result.im, &expected.im, source);
        }
    }
}

#[test]
fn uncertain_numeric_order_tags_contribute_to_the_result_error() {
    for name in ["polylog", "bessel_j", "bessel_y", "bessel_i", "bessel_k"] {
        let template = parse!(format!("{name}(order,x)"));
        let expression = template
            .replace(parse!("order"))
            .with(Atom::num(tracked(96, 1.25, 0.001)));
        let result = expression
            .evaluate_with_prec(&HashMap::from([(parse!("x"), tracked(96, 0.375, 0.))]), 96)
            .unwrap();
        assert!(result.get_error_magnitude().is_finite(), "{name}");
        assert!(!result.is_exact(), "{name}");
        for order in [1.249, 1.251] {
            let reference_expression = template
                .replace(parse!("order"))
                .with(Atom::num(tracked(256, order, 0.)));
            let expected = reference(&reference_expression, 0.375, 0.);
            covers(&result, &expected.re, name);
        }
    }
}

#[test]
fn errors_beyond_the_binary64_range_are_not_discarded() {
    let error = Magnitude::ONE.scale_pow2(-4000);
    assert!(error.to_f64().is_none());
    for source in ["gamma(x)", "erf(x)"] {
        let expression = parse!(source);
        let input = Tracked::new_with_error_magnitude(Float::with_val(256, 1.25), error);
        let result = expression
            .evaluate_with_prec(&HashMap::from([(parse!("x"), input)]), 256)
            .unwrap();
        assert!(!result.get_error_magnitude().is_zero(), "{source}");
        assert!(result.get_error_magnitude().is_finite(), "{source}");

        let input = Complex::new(
            tracked(256, 1.25, 0.),
            Tracked::new_with_error_magnitude(Float::new(256), error),
        );
        let result = expression
            .evaluate_with_prec(&HashMap::from([(parse!("x"), input)]), 256)
            .unwrap();
        assert_eq!(result.im.get_num(), &Float::new(256), "{source}");
        assert!(!result.im.get_error_magnitude().is_zero(), "{source}");
        assert!(result.im.get_error_magnitude().is_finite(), "{source}");
    }
}

#[test]
fn invalid_orders_and_nonreal_values_do_not_panic_in_real_domains() {
    for source in ["polygamma(-1,x)", "polygamma(1/2,x)", "polylog(2,x)"] {
        let expression = parse!(source);
        let result = expression
            .evaluate_with_prec(&HashMap::from([(parse!("x"), tracked(96, 1.5, 0.001))]), 96)
            .unwrap();
        assert!(result.get_num().to_f64().is_nan(), "{source}");
        let result = expression
            .evaluate(&HashMap::from([(
                parse!("x"),
                Tracked64::new_with_absolute_error(1.5, 0.001),
            )]))
            .unwrap();
        assert!(result.get_num().is_nan(), "{source}");
    }
}

#[test]
fn broad_oscillatory_bessel_neighborhoods_do_not_alias_the_error_samples() {
    let center = 99.7468198586806;
    let expression = parse!("bessel_j(0,x)");
    // The center and endpoints are close to roots, while the interior contains
    // an extremum. Local samples cannot resolve this broad neighborhood.
    let result = expression
        .evaluate_with_prec(
            &HashMap::from([(parse!("x"), tracked(96, center, std::f64::consts::PI))]),
            96,
        )
        .unwrap();
    assert!(result.get_error_magnitude().is_infinite());
    covers(
        &result,
        &reference(&expression, center + std::f64::consts::FRAC_PI_2, 0.).re,
        "oscillatory bessel_j",
    );
}

#[test]
fn oscillatory_complex_erf_neighborhoods_have_unbounded_error() {
    let expression = parse!("erf(x)");
    let error = std::f64::consts::PI / 10.;
    // Along this horizontal perturbation the phase of exp(-x^2) nearly makes
    // a full turn. Endpoint samples alone can miss the interior variation.
    let result = expression
        .evaluate_with_prec(
            &HashMap::from([(
                parse!("x"),
                Complex::new(tracked(96, 0., error), tracked(96, 10., 0.)),
            )]),
            96,
        )
        .unwrap();
    assert!(result.re.get_error_magnitude().is_infinite());
    assert!(result.im.get_error_magnitude().is_infinite());
    let expected = reference(&expression, error / 2., 10.);
    covers(&result.re, &expected.re, "oscillatory erf real component");
    covers(
        &result.im,
        &expected.im,
        "oscillatory erf imaginary component",
    );
}

#[test]
fn oscillatory_complex_gamma_neighborhoods_have_unbounded_error() {
    let expression = parse!("gamma(x)");
    let error = 1.365861791023665;
    // At this large real argument, imaginary perturbations rotate gamma's
    // phase. The half-step changes sign while the endpoints nearly agree.
    let result = expression
        .evaluate_with_prec(
            &HashMap::from([(
                parse!("x"),
                Complex::new(tracked(96, 100., 0.), tracked(96, 0., error)),
            )]),
            96,
        )
        .unwrap();
    assert!(result.re.get_error_magnitude().is_infinite());
    assert!(result.im.get_error_magnitude().is_infinite());
    let expected = reference(&expression, 100., error / 2.);
    covers(&result.re, &expected.re, "oscillatory gamma real component");
    covers(
        &result.im,
        &expected.im,
        "oscillatory gamma imaginary component",
    );
}

#[test]
fn polylog_at_one_detects_a_pole_in_the_uncertain_order_tag() {
    // Keep the argument symbolic until evaluation, so this exercises the
    // polylog callback rather than normalization to zeta(order).
    let expression = parse!("polylog(order,x)")
        .replace(parse!("order"))
        .with(Atom::num(tracked(96, 1.01, 0.02)));
    let result = expression
        .evaluate_with_prec(&HashMap::from([(parse!("x"), tracked(96, 1., 0.))]), 96)
        .unwrap();
    assert!(result.get_error_magnitude().is_infinite());
}

#[test]
fn integer_order_bessel_functions_preserve_real_parity() {
    for name in ["bessel_j", "bessel_i"] {
        for order in [1, 2] {
            let expression = parse!(format!("{name}({order},x)"));
            let result = expression
                .evaluate_with_prec(
                    &HashMap::from([(parse!("x"), tracked(96, -1.25, 0.001))]),
                    96,
                )
                .unwrap();
            let result64 = expression
                .evaluate(&HashMap::from([(
                    parse!("x"),
                    Tracked64::new_with_absolute_error(-1.25, 0.001),
                )]))
                .unwrap();
            assert!(result.get_num().is_finite(), "{name}, order {order}");
            assert!(result.get_error_magnitude().is_finite());
            for positive in [1.249, 1.251] {
                let positive_reference = reference(&expression, positive, 0.).re;
                let expected = if order % 2 == 0 {
                    positive_reference
                } else {
                    -positive_reference
                };
                covers(&result, &expected, name);
                covers64(&result64, &expected, name);
            }
            let normalized = expression
                .replace(parse!("x"))
                .with(Atom::num(tracked(96, -1.25, 0.001)));
            let value = normalized
                .as_num_view()
                .unwrap()
                .get_coeff_view()
                .to_tracked_float(96)
                .unwrap();
            assert!(value.im.is_exact_zero(), "{name}, order {order}");
        }
    }
}
