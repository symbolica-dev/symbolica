use numerica::domains::float::{DoubleFloat, ErrorPropagatingFloat, Float, Real, RealLike};
use symbolica::solve::{
    BracketedRootConvergence, BracketedRootError, BracketedRootOptions, nsolve_bracketed,
};

fn options<N: RealLike>(one: &N, tolerance: N) -> BracketedRootOptions<N> {
    BracketedRootOptions {
        absolute_tolerance: one.zero(),
        relative_tolerance: tolerance,
        max_iterations: 256,
        initial_guess: None,
        convergence: BracketedRootConvergence::Bracket,
    }
}

fn quadratic<N: RealLike + Real + PartialOrd>(one: N, tolerance: N) {
    let two = one.from_usize(2);
    let settings = options(&one, tolerance.clone());
    let result = nsolve_bracketed(one.zero(), two.clone(), &settings, |x| {
        (x.clone() * x - &two, two.clone() * x)
    })
    .unwrap();
    assert!((result.root.clone() * &result.root - &two).norm() < tolerance * one.from_usize(8));
    assert!(result.lower <= result.root && result.root <= result.upper);
    assert!(result.lower_value <= one.zero() && result.upper_value >= one.zero());
    assert!(result.iterations < 64);
    assert!(result.evaluations >= result.iterations);
}

#[test]
fn native_scalar_domains_keep_their_own_precision() {
    quadratic(1f64, 1e-14);
    quadratic(DoubleFloat::from(1.), DoubleFloat::from(1e-29));
    quadratic(Float::with_val(192, 1), Float::with_val(192, 1e-50));
}

#[test]
fn prepared_polynomial_callback_and_equation_scaling() {
    let settings = options(&1f64, 1e-14);
    let mut baseline: Option<f64> = None;
    for scale in [1e-200, 1., 1e200, -1e-200, -1., -1e200] {
        let mut calls = 0;
        let root = nsolve_bracketed(0., 1., &settings, |x| {
            calls += 1;
            let x2 = x * x;
            (
                scale * (x2 * (1. + x2 * (2. + 3. * x2)) - 1.),
                scale * (2. * x * (1. + x2 * (4. + 9. * x2))),
            )
        })
        .unwrap();
        assert_eq!(root.evaluations, calls);
        let x2 = root.root * root.root;
        assert!((x2 * (1. + x2 * (2. + 3. * x2)) - 1.).abs() < 1e-13);
        if let Some(previous) = baseline {
            assert!((root.root - previous).abs() < 1e-14);
        }
        baseline = Some(root.root);
    }
}

#[test]
fn relative_tolerance_handles_small_roots_and_endpoints() {
    for root in [0., 1e-120, 1e-20, 0.5, 1. - f64::EPSILON, 1.] {
        let mut settings = options(&1f64, 1e-14);
        settings.initial_guess = Some(root);
        let scale = root.max(1e-120);
        let result =
            nsolve_bracketed(0., 1., &settings, |x| ((x - root) / scale, 1. / scale)).unwrap();
        assert_eq!(result.root, root);
        assert_eq!(result.residual, 0.);
    }
    let settings = options(&1f64, 1e-14);
    let root = nsolve_bracketed(0., 1., &settings, |x| (x - 1e-20, 1.)).unwrap();
    assert!((root.root / 1e-20 - 1.).abs() < 1e-14);
}

#[test]
fn zero_derivative_away_from_the_root_falls_back_safely() {
    let mut settings = options(&1f64, 1e-13);
    settings.initial_guess = Some(0.);
    let result = nsolve_bracketed(0., 1., &settings, |x| (x * x * x - 0.001, 3. * x * x)).unwrap();
    assert!((result.root - 0.1).abs() < 1e-13);
}

#[test]
fn center_zero_and_endpoint_roots_retain_coefficient_uncertainty() {
    tracked_control(1f64);
    tracked_control(DoubleFloat::from(1.));
    tracked_control(Float::with_val(192, 1));
}

fn tracked_control<N: RealLike + Real + PartialOrd>(unit: N) {
    let exact = |x| ErrorPropagatingFloat::new_with_accuracy(x, f64::INFINITY);
    for coefficient in [1, 2, 4] {
        let coefficient = ErrorPropagatingFloat::new(unit.from_usize(coefficient), 6.);
        let one = exact(unit.clone());
        let two = exact(unit.from_usize(2));
        let settings = options(
            &one,
            exact(unit.clone() / unit.from_i64(100_000_000_000_000)),
        );
        let result = nsolve_bracketed(exact(unit.zero()), one.clone(), &settings, |x| {
            (
                coefficient.clone() * x * x - &one,
                two.clone() * &coefficient * x,
            )
        })
        .unwrap();
        let expected = unit.clone() / coefficient.get_num().sqrt();
        assert!((result.root.get_num().clone() - &expected).norm().to_f64() < 1e-13);
        // Local dr/da = -r/(2a): uncertain coefficients must not produce
        // an allegedly exact endpoint or exact initial-guess root.
        let local_error = expected.to_f64() * coefficient.get_relative_error() / 2.;
        assert!(result.root.get_absolute_error() >= 0.99 * local_error);
        assert!(result.root.get_absolute_error().is_finite());
    }
}

#[test]
fn invalid_ranges_nonfinite_callbacks_and_limits_fail_explicitly() {
    let mut settings = options(&1f64, 1e-14);
    let square = |x: &f64| (x * x - 2., 2. * x);
    assert_eq!(
        nsolve_bracketed(2., 3., &settings, square).unwrap_err(),
        BracketedRootError::NotBracketed
    );
    assert_eq!(
        nsolve_bracketed(0., 1., &settings, |_| (f64::NAN, 1.)).unwrap_err(),
        BracketedRootError::NonFinite
    );
    assert_eq!(
        nsolve_bracketed(0., 1., &settings, |_| (1., f64::INFINITY)).unwrap_err(),
        BracketedRootError::NonFinite
    );
    assert_eq!(
        nsolve_bracketed(0., 1., &settings, |x| (x * x * x, 3. * x * x)).unwrap_err(),
        BracketedRootError::SingularRootDerivative
    );
    settings.max_iterations = 1;
    assert_eq!(
        nsolve_bracketed(0., 2., &settings, square).unwrap_err(),
        BracketedRootError::IterationLimit
    );
    settings.max_iterations = 0;
    assert!(matches!(
        nsolve_bracketed(0., 2., &settings, square),
        Err(BracketedRootError::InvalidInput(_))
    ));
    settings.max_iterations = 256;
    settings.relative_tolerance = 0.;
    assert!(matches!(
        nsolve_bracketed(0., 2., &settings, square),
        Err(BracketedRootError::InvalidInput(_))
    ));
    settings.relative_tolerance = 1e-14;
    settings.initial_guess = Some(3.);
    assert!(matches!(
        nsolve_bracketed(0., 2., &settings, square),
        Err(BracketedRootError::InvalidInput(_))
    ));
}

#[test]
fn unrepresentable_root_tolerance_reports_stagnation() {
    let settings = BracketedRootOptions {
        absolute_tolerance: 1e-40,
        relative_tolerance: 0.,
        max_iterations: 32,
        initial_guess: None,
        convergence: BracketedRootConvergence::Bracket,
    };
    assert_eq!(
        nsolve_bracketed(1., 1. + f64::EPSILON, &settings, |x| (
            (x - 1.) - f64::EPSILON / 2.,
            1.
        ))
        .unwrap_err(),
        BracketedRootError::Stagnation
    );
}

#[test]
fn invalid_endpoints_and_options_do_not_execute_the_callback() {
    let valid = options(&1f64, 1e-14);
    let unexpected = |_: &f64| -> (f64, f64) { panic!("invalid input executed callback") };
    for (lower, upper) in [(1., 0.), (1., 1.), (f64::NAN, 1.), (0., f64::INFINITY)] {
        assert!(matches!(
            nsolve_bracketed(lower, upper, &valid, unexpected),
            Err(BracketedRootError::InvalidInput(_))
        ));
    }
    for tolerance in [-1., f64::NAN, f64::INFINITY] {
        let mut settings = valid.clone();
        settings.absolute_tolerance = tolerance;
        assert!(matches!(
            nsolve_bracketed(0., 1., &settings, unexpected),
            Err(BracketedRootError::InvalidInput(_))
        ));
        settings = valid.clone();
        settings.relative_tolerance = tolerance;
        assert!(matches!(
            nsolve_bracketed(0., 1., &settings, unexpected),
            Err(BracketedRootError::InvalidInput(_))
        ));
    }
    for guess in [-1., 2., f64::NAN, f64::INFINITY] {
        let mut settings = valid.clone();
        settings.initial_guess = Some(guess);
        assert!(matches!(
            nsolve_bracketed(0., 1., &settings, unexpected),
            Err(BracketedRootError::InvalidInput(_))
        ));
    }
}

#[test]
fn nonfinite_values_are_rejected_during_refinement_and_final_evaluation() {
    let settings = options(&1f64, 1e-14);
    for fail_at in [3, 4] {
        let mut calls = 0;
        let result = nsolve_bracketed(0., 1., &settings, |x| {
            calls += 1;
            if calls == fail_at {
                (f64::NAN, 1.)
            } else {
                (x - 0.5, 1.)
            }
        });
        assert_eq!(result.unwrap_err(), BracketedRootError::NonFinite);
        assert_eq!(calls, fail_at);
    }
}

#[test]
fn endpoint_roots_report_actual_evaluation_counts() {
    for endpoint in [0., 1.] {
        let mut calls = 0;
        let result = nsolve_bracketed(0., 1., &options(&1f64, 1e-14), |x| {
            calls += 1;
            (x - endpoint, 1.)
        })
        .unwrap();
        assert_eq!(result.root, endpoint);
        assert_eq!(result.iterations, 1);
        assert_eq!(result.evaluations, 3);
        assert_eq!(result.evaluations, calls);
        assert_eq!(
            result.termination,
            symbolica::solve::BracketedRootTermination::NumericalZero
        );
    }
}

#[test]
fn bracket_policy_avoids_a_small_step_false_convergence() {
    let mut settings = options(&1f64, 1e-8);
    settings.initial_guess = Some(0.75);
    let sharp = |x: &f64| {
        let argument = 1e20 * (x - 0.75);
        (
            x - 0.5 + 1e-10 * argument.atan(),
            1. + 1e10 / (1. + argument * argument),
        )
    };
    let strict = nsolve_bracketed(0., 1., &settings, sharp).unwrap();
    assert!((strict.root - 0.5).abs() < 1e-8);
    assert!(strict.residual.abs() < 1e-8);
    settings.convergence = BracketedRootConvergence::NewtonOrBracket;
    let heuristic = nsolve_bracketed(0., 1., &settings, sharp).unwrap();
    assert_eq!(
        heuristic.termination,
        symbolica::solve::BracketedRootTermination::NewtonCorrection
    );
    assert!(heuristic.residual.abs() > 0.2);
    // The returned bracket remains available; its width does not imply that
    // the heuristic candidate has the requested global root-distance accuracy.
    assert!(heuristic.upper - heuristic.lower > 0.5);
}

#[test]
fn subnormal_brackets_do_not_lose_width_when_endpoints_are_halved() {
    let unit = f64::from_bits(1);
    let settings = BracketedRootOptions {
        absolute_tolerance: unit,
        relative_tolerance: 0.,
        max_iterations: 32,
        initial_guess: Some(3. * unit),
        convergence: BracketedRootConvergence::Bracket,
    };
    let result = nsolve_bracketed(3. * unit, 5. * unit, &settings, |x| {
        let z = x / unit;
        (
            2f64.powi(-1000) * (z - 4.5) * (z - 2.875),
            2f64.powi(74) * (2. * z - 7.375),
        )
    })
    .unwrap();
    assert_eq!(
        result.termination,
        symbolica::solve::BracketedRootTermination::BracketWidth
    );
    assert!(result.upper - result.lower <= unit);
    assert!((result.root / unit - 4.5).abs() <= 1.);
    assert!(result.lower <= result.root && result.root <= result.upper);
    assert!(result.lower_value <= 0. && result.upper_value >= 0.);
}

#[test]
fn a_small_bracket_without_a_usable_correction_reports_stagnation() {
    let epsilon = f64::EPSILON;
    let settings = BracketedRootOptions {
        absolute_tolerance: epsilon,
        relative_tolerance: 0.,
        max_iterations: 32,
        initial_guess: None,
        convergence: BracketedRootConvergence::Bracket,
    };
    // The simple root is halfway between adjacent representable endpoints.
    // Both endpoint derivatives vanish: returning an exact endpoint would lose
    // the promised Newton propagation of coefficient uncertainty.
    let error = nsolve_bracketed(1., 1. + epsilon, &settings, |x| {
        let t = (x - 1.) / epsilon;
        (
            3. * t * t - 2. * t * t * t - 0.5,
            (6. * t - 6. * t * t) / epsilon,
        )
    })
    .unwrap_err();
    assert_eq!(error, BracketedRootError::Stagnation);
}

#[test]
fn a_singular_final_candidate_is_not_returned_as_a_simple_root() {
    let settings = BracketedRootOptions {
        absolute_tolerance: 3.,
        relative_tolerance: 0.,
        max_iterations: 32,
        initial_guess: Some(1.5),
        convergence: BracketedRootConvergence::Bracket,
    };
    // The bracket contains a simple root at 2, but Newton's correction from
    // 1.5 lands exactly on the double root at 1. Reject it consistently with
    // a singular initial or endpoint root.
    let error = nsolve_bracketed(0., 3., &settings, |x| {
        ((x - 1.) * (x - 1.) * (x - 2.), 3. * x * x - 8. * x + 5.)
    })
    .unwrap_err();
    assert_eq!(error, BracketedRootError::SingularRootDerivative);
}
