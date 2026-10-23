use std::sync::{
    Arc,
    atomic::{AtomicU64, AtomicUsize, Ordering},
};
use symbolica::{
    domains::finite_field::{FiniteFieldCore, Zp64},
    poly::{
        PolyVariable,
        polynomial::MultivariatePolynomial,
        reconstruction::{
            ReconstructionError, ReconstructionMethod::BalancedZippel, ReconstructionOptions,
            reconstruct_rational_function, reconstruct_rational_function_fallible,
            reconstruct_rational_function_over_q_fallible,
        },
    },
    prelude::*,
    utils::resources::{ResourceBudget, ResourceLimitError, ResourceLimits},
};

#[test]
fn ordinary_search_widens_initial_degree_and_does_not_enforce_diagnostic_counts() {
    let field = Zp64::new(1_000_003);
    let variables: Arc<Vec<PolyVariable>> = Arc::new(vec![symbol!("adaptive_x").into()]);
    let numerator: MultivariatePolynomial<_, u16> =
        parse!("adaptive_x^7+3*adaptive_x+1").to_polynomial(&field, variables.clone());
    let denominator: MultivariatePolynomial<_, u16> =
        parse!("adaptive_x+7").to_polynomial(&field, variables.clone());
    let options = ReconstructionOptions {
        max_degree: 1,
        max_probes: 0,
        max_attempts: 0,
        ..Default::default()
    };
    let oracle = |f: &Zp64, point: &[_]| {
        let d = denominator.replace_all(point);
        (!f.is_zero(&d)).then(|| f.div(&numerator.replace_all(point), &d))
    };
    let (result, stats) = reconstruct_rational_function(
        field.clone(),
        variables.clone(),
        oracle,
        BalancedZippel,
        &options,
    )
    .unwrap();
    assert_eq!(
        &result.numerator * &denominator,
        &result.denominator * &numerator
    );
    assert!(stats.attempts > 1);
    let inert_counters = ReconstructionOptions {
        max_probes: usize::MAX,
        max_attempts: usize::MAX,
        ..options.clone()
    };
    let (same_result, same_work) = reconstruct_rational_function(
        field.clone(),
        variables.clone(),
        oracle,
        BalancedZippel,
        &inert_counters,
    )
    .unwrap();
    assert_eq!(result, same_result);
    assert_eq!(stats.attempts, same_work.attempts);
    assert_eq!(stats.probes, same_work.probes);
    let diagnostic = ReconstructionOptions {
        max_probes: 10_000,
        max_attempts: 1,
        ..options
    }
    .with_diagnostics();
    assert_eq!(
        reconstruct_rational_function(field, variables, oracle, BalancedZippel, &diagnostic)
            .unwrap_err(),
        ReconstructionError::AttemptsExhausted
    );
}

#[test]
fn oracle_errors_are_not_unavailable_points_or_retried_anchors() {
    let field = Zp64::new(1_000_003);
    let variables = Arc::new(vec![symbol!("oracle_x").into()]);
    let mut calls = 0;
    let error = reconstruct_rational_function_fallible(
        field,
        variables.clone(),
        |_, _| {
            calls += 1;
            Err(ReconstructionError::OracleFailure(
                "invalid routing witness".into(),
            ))
        },
        BalancedZippel,
        &ReconstructionOptions::default(),
    )
    .unwrap_err();
    assert_eq!(
        error,
        ReconstructionError::OracleFailure("invalid routing witness".into())
    );
    assert_eq!(calls, 1);
    calls = 0;
    let error = reconstruct_rational_function_over_q_fallible(
        variables,
        |_, _| {
            calls += 1;
            Err(ReconstructionError::OracleFailure(
                "invalid exact transport".into(),
            ))
        },
        BalancedZippel,
        &ReconstructionOptions::default(),
        0,
    )
    .unwrap_err();
    assert_eq!(
        error,
        ReconstructionError::OracleFailure("invalid exact transport".into())
    );
    assert_eq!(calls, 1);
}

#[test]
fn typed_resource_failure_survives_the_oracle_boundary() {
    let field = Zp64::new(1_000_003);
    let variables = Arc::new(vec![symbol!("resource_x").into()]);
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = calls.clone();
    let budget = ResourceBudget::with_memory_monitor(
        ResourceLimits {
            max_process_rss_bytes: Some(10),
            ..Default::default()
        },
        move || {
            Ok(if observed.load(Ordering::SeqCst) == 0 {
                0
            } else {
                11
            })
        },
    );
    let result = budget.run(|| {
        reconstruct_rational_function_fallible(
            field,
            variables,
            |f, _| {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(Some(f.one()))
            },
            BalancedZippel,
            &ReconstructionOptions::default(),
        )
    });
    assert!(matches!(
        result,
        Err(ReconstructionError::ResourceLimit(
            ResourceLimitError::Memory {
                limit_bytes: 10,
                ..
            }
        ))
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn one_ambient_budget_covers_all_prime_images_and_final_verification() {
    let variables = Arc::new(vec![symbol!("prime_scope_x").into()]);
    let primes = Arc::new(AtomicUsize::new(0));
    let last_prime = AtomicU64::new(0);
    let observed = primes.clone();
    let budget = ResourceBudget::with_memory_monitor(
        ResourceLimits {
            max_process_rss_bytes: Some(10),
            ..Default::default()
        },
        move || {
            Ok(if observed.load(Ordering::SeqCst) < 2 {
                0
            } else {
                11
            })
        },
    );
    let result = budget.run(|| {
        reconstruct_rational_function_over_q_fallible(
            variables,
            |f, _| {
                if last_prime.swap(f.get_prime(), Ordering::SeqCst) != f.get_prime() {
                    primes.fetch_add(1, Ordering::SeqCst);
                }
                Ok(Some(f.to_element(3)))
            },
            BalancedZippel,
            &ReconstructionOptions::default(),
            0,
        )
    });
    assert!(matches!(
        result,
        Err(ReconstructionError::ResourceLimit(
            ResourceLimitError::Memory {
                limit_bytes: 10,
                ..
            }
        ))
    ));
    assert_eq!(primes.load(Ordering::SeqCst), 2);
}
