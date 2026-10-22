//! Resource limits must refuse before index overflow or dense-grid allocation.

use std::sync::Arc;

use symbolica::{
    atom::{Atom, AtomCore},
    domains::{
        atom::AtomField,
        rational::{Q, Rational},
    },
    parse,
    poly::series::{Series, SeriesDepth, SeriesError, SeriesSettings},
    symbol,
};

#[test]
fn absolute_depth_increment_and_generic_taylor_count_are_checked() {
    let x = symbol!("series_size_boundary::x");
    for order in [i64::MAX, i64::MAX - 1] {
        assert!(matches!(
            Atom::var(x).series(x, 0, SeriesDepth::absolute(order)),
            Err(SeriesError::IndexOverflow { .. })
        ));
    }
    let function = parse!("series_size_boundary::f(series_size_boundary::x)");
    assert!(matches!(
        function.series(x, 0, SeriesDepth::absolute(1_i64 << 32)),
        Err(SeriesError::IndexOverflow { .. })
    ));
    // Each factor fits u32, but the native rational-grid derivative estimate
    // must not wrap its product into a smaller, apparently successful count.
    assert!(matches!(
        function.series(x, 0, SeriesDepth::absolute((65_539, 65_537))),
        Err(SeriesError::IndexOverflow { .. })
    ));
}

#[test]
fn joint_ramification_is_checked_before_dense_allocation() {
    let x = symbol!("series_joint_grid::x");
    let first = Series::new(
        &Q,
        None,
        Arc::new(x.into()),
        Rational::from(0),
        Rational::from((1, 3_037_000_499_i64)),
    )
    .constant(Rational::from(1));
    let second = Series::new(
        &Q,
        None,
        Arc::new(x.into()),
        Rational::from(0),
        Rational::from((1, 3_037_000_501_i64)),
    )
    .constant(Rational::from(1));
    assert!(matches!(
        first.try_add(&second, &SeriesSettings::default()),
        Err(SeriesError::IndexOverflow { .. })
    ));
    assert!(matches!(
        first.try_mul(&second, &SeriesSettings::default()),
        Err(SeriesError::IndexOverflow { .. })
    ));
}

#[test]
fn explicit_grid_and_slot_limits_refuse_instead_of_returning_partial_series() {
    let x = symbol!("series_container_limits::x");
    let input = parse!("series_container_limits::x^(1/3037000499)");
    let limited_grid = SeriesSettings::default().max_ramification(64);
    assert!(matches!(
        input.series_with_settings(x, 0, 1, &limited_grid),
        Err(SeriesError::LimitExceeded { .. })
    ));
    let rational = parse!("(1 + series_container_limits::x)^-1");
    let limited_slots = SeriesSettings::default().max_coefficient_slots(2);
    assert!(matches!(
        rational.series_with_settings(x, 0, 6, &limited_slots),
        Err(SeriesError::LimitExceeded { .. })
    ));
    let enough = SeriesSettings::default()
        .max_ramification(64)
        .max_coefficient_slots(64);
    assert_eq!(
        rational
            .series_with_settings(x, 0, 6, &enough)
            .unwrap()
            .to_atom(),
        rational.series(x, 0, 6).unwrap().to_atom()
    );
}

#[test]
fn oversized_combined_shifts_return_a_structured_error() {
    let x = symbol!("series_shift_limits::x");
    let settings = SeriesSettings::default();
    let prototype = Series::try_new(
        &Q,
        None,
        Arc::new(x.into()),
        Rational::from(0),
        Rational::from(3),
        &settings,
    )
    .unwrap();
    let shifted = prototype
        .try_monomial(Rational::from(1), Rational::from(i64::MAX / 2), &settings)
        .unwrap();
    assert!(matches!(
        shifted.try_mul(&shifted, &settings),
        Err(SeriesError::IndexOverflow { .. })
    ));
    let prototype = Series::try_new(
        &AtomField::default(),
        None,
        Arc::new(x.into()),
        Atom::num(0),
        Rational::from(3),
        &settings,
    )
    .unwrap();
    let most_negative = prototype
        .try_monomial(Atom::num(1), Rational::from(i64::MIN), &settings)
        .unwrap();
    assert!(matches!(
        most_negative.rpow_with_settings(Rational::from((1, 2)), &settings),
        Err(SeriesError::IndexOverflow { .. })
    ));
}

#[test]
fn cancellation_retries_share_one_local_limit() {
    let x = symbol!("series_retry_limits::x");
    let input = parse!("(sin(series_retry_limits::x) - series_retry_limits::x)^-1");
    let exhausted = SeriesSettings::default().max_precision_retries(0);
    assert!(matches!(
        input.series_with_settings(x, 0, 0, &exhausted),
        Err(SeriesError::LimitExceeded { .. })
    ));
    let enough = SeriesSettings::default().max_precision_retries(32);
    let first = input
        .series_with_settings(x, 0, 0, &enough)
        .unwrap()
        .to_atom();
    let second = input.series(x, 0, 0).unwrap().to_atom();
    assert_eq!(first, second);
    assert!(matches!(
        input.series_with_settings(x, 0, 0, &exhausted),
        Err(SeriesError::LimitExceeded { .. })
    ));
}

#[test]
fn checked_settings_preserve_native_laurent_and_puiseux_results() {
    let x = symbol!("series_checked_parity::x");
    let settings = SeriesSettings::default()
        .max_ramification(64)
        .max_coefficient_slots(4096)
        .max_precision_retries(32);
    for input in [
        "1/(series_checked_parity::x^2*(1+series_checked_parity::x))",
        "series_checked_parity::x^(1/2) + series_checked_parity::x^(2/3)",
        "exp(series_checked_parity::x) + sin(series_checked_parity::x) + cos(series_checked_parity::x)",
        "log(1+series_checked_parity::x)",
    ] {
        let input = parse!(input);
        assert_eq!(
            input
                .series_with_settings(x, 0, 4, &settings)
                .unwrap()
                .to_atom(),
            input.series(x, 0, 4).unwrap().to_atom()
        );
    }
    let laurent = parse!("1/(series_checked_parity::x^10 + series_checked_parity::x^20)");
    assert_eq!(
        laurent
            .series_with_settings(x, 0, SeriesDepth::absolute(-1), &settings)
            .unwrap()
            .to_atom(),
        parse!("series_checked_parity::x^-10")
    );
    assert!(matches!(
        laurent.series_with_settings(x, 0, SeriesDepth::relative(-1), &settings),
        Err(SeriesError::NonPositiveRelativeDepth { .. })
    ));
}

#[test]
fn custom_and_unknown_function_expansions_keep_native_settings() {
    let x = symbol!("series_callback_limits::x");
    let _ = symbol!(
        "series_callback_limits::f",
        series = |args| {
            let [arg] = args else { unreachable!() };
            let point = arg.coefficient(0.into())?;
            let delta = arg.to_atom() - &point;
            Some((Atom::num(1), Atom::num(1) / (Atom::num(1) - &delta)))
        }
    );
    let custom = parse!("series_callback_limits::f(series_callback_limits::x)");
    let enough = SeriesSettings::default().max_coefficient_slots(32);
    assert_eq!(
        custom
            .series_with_settings(x, 0, 3, &enough)
            .unwrap()
            .to_atom(),
        parse!(
            "1 + series_callback_limits::x + series_callback_limits::x^2 + series_callback_limits::x^3"
        )
    );
    assert!(matches!(
        custom.series_with_settings(x, 0, 3, &SeriesSettings::default().max_coefficient_slots(2)),
        Err(SeriesError::LimitExceeded { .. })
    ));
    let unknown = parse!("series_callback_limits::g(series_callback_limits::x)");
    assert_eq!(
        unknown
            .series_with_settings(x, 0, 3, &enough)
            .unwrap()
            .to_atom(),
        unknown.series(x, 0, 3).unwrap().to_atom()
    );
}

#[test]
fn concurrent_expansions_have_independent_precision_retry_contexts() {
    let x = symbol!("series_concurrent_limits::x");
    let input = parse!("(sin(series_concurrent_limits::x) - series_concurrent_limits::x)^-1");
    let exhausted = SeriesSettings::default().max_precision_retries(0);
    let enough = SeriesSettings::default().max_precision_retries(32);
    let (limited, complete) = std::thread::scope(|scope| {
        let limited = scope.spawn(|| input.series_with_settings(x, 0, 0, &exhausted));
        let complete = scope.spawn(|| input.series_with_settings(x, 0, 0, &enough));
        (limited.join().unwrap(), complete.join().unwrap())
    });
    assert!(matches!(limited, Err(SeriesError::LimitExceeded { .. })));
    assert_eq!(
        complete.unwrap().to_atom(),
        input.series(x, 0, 0).unwrap().to_atom()
    );
}

#[test]
fn retained_inverse_heuristic_does_not_establish_a_zero_divisor() {
    let x = symbol!("series_unresolved_inverse::x");
    let input = parse!("1/sin(series_unresolved_inverse::x^2048)");
    let result = input.series_with_settings(
        x,
        0,
        SeriesDepth::absolute(0),
        &SeriesSettings::default()
            .max_coefficient_slots(4096)
            .max_precision_retries(32),
    );
    let error = result.unwrap_err();
    assert!(matches!(
        error,
        SeriesError::UnresolvedInverseLeadingCoefficient {
            heuristic_limit: 1000,
            ..
        }
    ));
    assert!(error.to_string().contains("does not prove"));
}

#[test]
fn empty_zero_remainders_have_conservative_function_images() {
    let x = symbol!("series_zero_remainder::x");
    let settings = SeriesSettings::default().max_coefficient_slots(64);
    let zero = Series::try_new(
        &AtomField::default(),
        None,
        Arc::new(x.into()),
        Atom::num(0),
        Rational::from((3, 2)),
        &settings,
    )
    .unwrap()
    .try_mul_exp_units(-1, &settings)
    .unwrap();
    // Relative precision alone is insufficient: this negative shift still has
    // a positive established absolute remainder precision of one.
    assert_eq!(zero.absolute_order(), Rational::from(1));
    let sine = zero.sin_with_settings(&settings).unwrap();
    assert!(sine.to_atom().is_zero());
    assert_eq!(sine.absolute_order(), Rational::from(1));
    assert_eq!(sine.relative_order(), Rational::from(1));
    let exponential = zero.exp_with_settings(&settings).unwrap();
    assert_eq!(exponential.to_atom(), Atom::num(1));
    assert_eq!(exponential.absolute_order(), Rational::from(1));
    let cosine = zero.cos_with_settings(&settings).unwrap();
    assert_eq!(cosine.to_atom(), Atom::num(1));
    assert_eq!(cosine.absolute_order(), Rational::from(2));
    let fractional = Series::try_new(
        &AtomField::default(),
        None,
        Arc::new(x.into()),
        Atom::num(0),
        Rational::from((3, 2)),
        &settings,
    )
    .unwrap();
    assert_eq!(
        fractional
            .cos_with_settings(&settings)
            .unwrap()
            .absolute_order(),
        Rational::from(3)
    );
    let shifted = Series::try_new(
        &AtomField::default(),
        None,
        Arc::new(x.into()),
        Atom::num(0),
        Rational::from(0),
        &settings,
    )
    .unwrap()
    .try_mul_exp_units(3, &settings)
    .unwrap();
    assert_eq!(
        shifted
            .cos_with_settings(&settings)
            .unwrap()
            .absolute_order(),
        Rational::from(6)
    );
    assert!(matches!(
        zero.log_with_settings(&settings),
        Err(SeriesError::MissingLogCoefficient)
    ));
    let unknown = zero.try_mul_exp_units(-2, &settings).unwrap();
    for result in [
        unknown.exp_with_settings(&settings),
        unknown.sin_with_settings(&settings),
        unknown.cos_with_settings(&settings),
    ] {
        assert!(matches!(
            result,
            Err(SeriesError::MissingFunctionCoefficient { .. })
        ));
    }
    let _ = symbol!(
        "series_zero_remainder::g",
        series = |_args| Some((Atom::num(1), Atom::num(0)))
    );
    let input = parse!("sin(series_zero_remainder::g(series_zero_remainder::x))");
    assert!(
        input
            .series_with_settings(x, 0, 3, &settings)
            .unwrap()
            .to_atom()
            .is_zero()
    );
}

#[test]
fn sibling_inverse_refinements_use_one_cumulative_retry_budget() {
    let x = symbol!("series_sibling_retries::x");
    let first = parse!("(sin(series_sibling_retries::x)-series_sibling_retries::x)^-1");
    let second = parse!("(cos(series_sibling_retries::x)-1)^-1");
    let source = &first + &second;
    let individual_limit = (0..32)
        .find(|&limit| {
            let settings = SeriesSettings::default().max_precision_retries(limit);
            first.series_with_settings(x, 0, 0, &settings).is_ok()
                && second.series_with_settings(x, 0, 0, &settings).is_ok()
        })
        .unwrap();
    assert!(individual_limit > 0);
    assert!(matches!(
        source.series_with_settings(
            x,
            0,
            0,
            &SeriesSettings::default().max_precision_retries(individual_limit)
        ),
        Err(SeriesError::LimitExceeded {
            resource: "precision retries",
            ..
        })
    ));
    assert_eq!(
        source
            .series_with_settings(
                x,
                0,
                0,
                &SeriesSettings::default().max_precision_retries(32)
            )
            .unwrap()
            .to_atom(),
        parse!(
            "-6*series_sibling_retries::x^-3-2*series_sibling_retries::x^-2-3/10*series_sibling_retries::x^-1-1/6"
        )
    );
}
