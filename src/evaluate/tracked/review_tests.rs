use std::collections::HashMap;

use crate::{
    atom::{Atom, AtomCore, Symbol},
    domains::float::{Complex, ErrorQuality, Float, RealBall, SingleFloat, TrackedComplex},
    function, parse,
};

#[test]
fn constant_budget_failure_does_not_poison_concurrent_success_cache() {
    use crate::{
        atom::EvaluationInfo,
        utils::resources::{self, ResourceBudget, ResourceLimits},
    };
    use std::sync::{
        Arc, Barrier,
        atomic::{AtomicUsize, Ordering},
    };

    let calls = Arc::new(AtomicUsize::new(0));
    let callback_calls = calls.clone();
    let info = Arc::new(EvaluationInfo::constant(move |tags, bits| {
        assert!(tags.is_empty());
        assert_eq!(bits, 53);
        callback_calls.fetch_add(1, Ordering::SeqCst);
        resources::checkpoint().map_err(|error| error.to_string())?;
        Ok(Complex::new(Float::with_val(bits, 2.5), Float::new(bits)))
    }));
    let observations = Arc::new(AtomicUsize::new(0));
    let budget = ResourceBudget::with_memory_monitor(
        ResourceLimits {
            max_process_rss_bytes: Some(1),
            ..Default::default()
        },
        move || {
            Ok(if observations.fetch_add(1, Ordering::SeqCst) == 0 {
                0
            } else {
                2
            })
        },
    );
    assert!(budget.run(|| info.evaluate_constant(&[], 53)).is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 1);

    let ready = Arc::new(Barrier::new(9));
    let workers = (0..8)
        .map(|_| {
            let info = info.clone();
            let ready = ready.clone();
            std::thread::spawn(move || {
                ready.wait();
                info.evaluate_constant(&[], 53).unwrap()
            })
        })
        .collect::<Vec<_>>();
    ready.wait();
    for worker in workers {
        assert_eq!(worker.join().unwrap().re, Float::with_val(53, 2.5));
    }
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    let exhausted = ResourceBudget::new(ResourceLimits {
        max_elapsed: Some(std::time::Duration::ZERO),
        ..Default::default()
    });
    assert!(exhausted.run(|| info.evaluate_constant(&[], 53)).is_err());
    assert_eq!(
        info.evaluate_constant(&[], 53).unwrap().re,
        Float::with_val(53, 2.5)
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn deep_tracked_functions_use_an_explicit_stack_and_keep_callbacks_lazy() {
    use crate::atom::EvaluationInfo;
    use crate::utils::resources::{ResourceBudget, ResourceLimits};

    let identity = crate::symbol!(
        "tracked_review_deep_identity",
        eval = EvaluationInfo::new().register_tracked(|tags, arguments, _bits| {
            assert!(tags.is_empty());
            let [value] = arguments else {
                return Err("identity expects one argument".into());
            };
            Ok(value.clone())
        })
    );
    let mut expression = Atom::num(1);
    for _ in 0..1536 {
        // A unary function without attributes is canonical when its argument
        // is canonical. Construct it directly to isolate interpreter traversal
        // from normalization's independent stack requirements.
        let mut outer = Atom::new();
        let function = outer.to_fun(identity);
        function.add_arg(expression.as_view());
        function.set_normalized(true);
        expression = outer;
    }
    let inputs = HashMap::<Atom, TrackedComplex>::new();
    let result = expression.evaluate_tracked(&inputs, 80).unwrap();
    assert_eq!(result.ball.re.center, Float::with_val(80, 1));
    assert!(result.ball.re.radius.is_zero());
    assert!(result.precision().imaginary.exact_zero);

    let budget = ResourceBudget::new(ResourceLimits {
        max_elapsed: Some(std::time::Duration::ZERO),
        ..Default::default()
    });
    let error = budget
        .run(|| expression.evaluate_tracked(&inputs, 80))
        .unwrap_err();
    assert!(error.to_string().contains("elapsed time"));
    assert!(expression.evaluate_tracked(&inputs, 80).is_ok());

    let unavailable = crate::symbol!("tracked_review_unavailable_branch").call(1);
    let conditional = function!(Symbol::IF, 1, 2, unavailable);
    assert_eq!(
        conditional
            .evaluate_tracked(&inputs, 80)
            .unwrap()
            .ball
            .re
            .center,
        Float::with_val(80, 2)
    );
}

#[test]
fn mapped_inputs_obey_working_precision_without_erasing_source_error() {
    let variable = parse!("tracked_review_input");
    let source = TrackedComplex::from_source_decimal("1.234567890123456789", 18, 512).unwrap();
    let source_center = source.ball.re.center.to_rational();
    let source_radius = source.ball.re.radius.to_rational();
    let inputs = HashMap::from([(variable.clone(), source)]);

    for bits in [32, 768] {
        let result = variable.evaluate_tracked(&inputs, bits).unwrap();
        assert_eq!(result.working_precision(), bits);
        assert_eq!(result.ball.re.center.prec(), bits);
        assert_eq!(result.error_quality, ErrorQuality::Estimated);
        assert!(result.precision().imaginary.exact_zero);
        let center_shift = (result.ball.re.center.to_rational() - &source_center).abs();
        assert!(result.ball.re.radius.to_rational() >= &source_radius + &center_shift);
        assert!(result.precision().real.significant_bits.unwrap() < 64);

        let squared = variable
            .pow(Atom::num(2))
            .evaluate_tracked(&inputs, bits)
            .unwrap();
        assert!(squared.ball.re.center.prec() <= bits);
        assert_eq!(squared.error_quality, ErrorQuality::Estimated);
    }
}

#[test]
fn conditional_selection_retains_estimated_condition_quality() {
    let condition = parse!("tracked_review_condition");
    let expression = function!(Symbol::IF, condition.clone(), 2, 3);
    let mut inputs = HashMap::from([(
        condition.clone(),
        TrackedComplex::from_source_decimal("1", 20, 128).unwrap(),
    )]);
    let result = expression.evaluate_tracked(&inputs, 128).unwrap();
    assert_eq!(result.ball.re.center, Float::with_val(128, 2));
    assert_eq!(result.error_quality, ErrorQuality::Estimated);

    inputs.insert(
        condition,
        TrackedComplex::from_source_decimal("0", 20, 128).unwrap(),
    );
    assert!(expression.evaluate_tracked(&inputs, 128).is_err());
}

#[test]
fn tracked_input_validation_rejects_invalid_enclosures_and_working_precision() {
    let variable = parse!("tracked_review_invalid");
    for (center, radius) in [
        (f64::NAN, 0.0),
        (f64::INFINITY, 0.0),
        (1.0, f64::NAN),
        (1.0, f64::INFINITY),
        (1.0, -1.0),
    ] {
        let invalid = TrackedComplex::from_ball(
            Complex::new(
                RealBall {
                    center: Float::with_val(128, center),
                    radius: Float::with_val(128, radius),
                },
                RealBall::exact(Float::new(128)),
            ),
            ErrorQuality::Bounded,
        );
        let inputs = HashMap::from([(variable.clone(), invalid)]);
        assert!(variable.evaluate_tracked(&inputs, 64).is_err());
    }

    let inputs = HashMap::<Atom, TrackedComplex>::new();
    for bits in [0, 1] {
        assert!(Atom::num(1).evaluate_tracked(&inputs, bits).is_err());
    }
}

#[cfg(all(feature = "float-mpfr", feature = "integer-gmp"))]
#[test]
fn finite_values_that_overflow_during_reprecision_are_rejected() {
    let center = Float::from_raw(rug::Float::with_val(4, 1.875) << (rug::float::exp_max() - 1));
    let input = TrackedComplex::from_ball(
        Complex::new(RealBall::exact(center), RealBall::exact(Float::new(4))),
        ErrorQuality::Bounded,
    );
    assert!(input.is_finite());
    // Rebuilding a finite input ball can also overflow its midpoint sum.
    let variable = parse!("tracked_review_exponent_limit");
    let inputs = HashMap::from([(variable.clone(), input)]);
    assert!(variable.evaluate_tracked(&inputs, 4).is_err());
}
