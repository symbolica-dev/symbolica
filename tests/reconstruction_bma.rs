use std::{cell::Cell, sync::Arc};
use symbolica::{
    domains::finite_field::{SMOOTH_PRIMES, ToFiniteField},
    poly::reconstruction::{
        ReconstructionError, ReconstructionMethod::*, ReconstructionOptions,
        reconstruct_rational_function, reconstruct_rational_function_over_q,
    },
    prelude::*,
};

fn field() -> Zp64 {
    Zp64::new(
        SMOOTH_PRIMES
            .iter()
            .find(|(p, _, _)| *p > 1 << 61)
            .unwrap()
            .0,
    )
}

#[test]
fn polynomial_bma_samples_depend_on_terms_not_exponent_size() {
    let f = field();
    let vars = Arc::new(vec![
        symbol!("x").into(),
        symbol!("y").into(),
        symbol!("z").into(),
    ]);
    for seed in [3, 19] {
        let mut counts = Vec::new();
        for degree in [5, 50, 500, 2000] {
            let n: MultivariatePolynomial<_, u16> = parse!(format!(
                "3*x^{degree}*y^2+5*y^{degree}*z^3+7*z^{degree}*x+11"
            ))
            .to_polynomial(&f, vars.clone());
            let calls = Cell::new(0);
            let (result, stats) = reconstruct_rational_function(
                f.clone(),
                vars.clone(),
                |_, point| {
                    calls.set(calls.get() + 1);
                    Some(n.replace_all(point))
                },
                PolynomialBma,
                &ReconstructionOptions {
                    max_degree: 2000,
                    seed,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(result.numerator, n);
            assert!(result.denominator.is_one());
            assert_eq!(calls.get(), stats.probes);
            assert!(stats.bma_sequences > 0);
            counts.push(stats.probes);
        }
        assert!(counts.iter().all(|&count| count == counts[0]), "{counts:?}");
        assert!(counts[0] <= 14, "{counts:?}");
    }
}

#[test]
fn rational_bma_normalizes_mixed_denominators_and_polynomial_slices() {
    let f = field();
    let vars = Arc::new(vec![
        symbol!("x").into(),
        symbol!("y").into(),
        symbol!("z").into(),
    ]);
    for (num, den) in [
        ("3*x^51+5*y^43+7*z^2+11", "z^2+3*z+7"),
        ("x^7*y^3+z^3+2*x+5", "x^4*z^2+y^5*z+3*x*y+7"),
        ("(x+y+z+1)^3+7", "(x+2*y+3*z+5)^2"),
        ("0", "1"),
        ("7", "3"),
    ] {
        let n: MultivariatePolynomial<_, u16> = parse!(num).to_polynomial(&f, vars.clone());
        let d: MultivariatePolynomial<_, u16> = parse!(den).to_polynomial(&f, vars.clone());
        for seed in [3, 19] {
            let calls = Cell::new(0);
            let (result, stats) = reconstruct_rational_function(
                f.clone(),
                vars.clone(),
                |f, point| {
                    calls.set(calls.get() + 1);
                    let denominator = d.replace_all(point);
                    (!f.is_zero(&denominator)).then(|| f.div(&n.replace_all(point), &denominator))
                },
                HuMonagan,
                &ReconstructionOptions {
                    max_degree: 64,
                    max_probes: 10000,
                    seed,
                    ..Default::default()
                },
            )
            .unwrap_or_else(|e| panic!("{num}/{den}: {e}"));
            assert_eq!(&result.numerator * &d, &result.denominator * &n);
            assert_eq!(stats.probes, calls.get());
            assert!(stats.bma_sequences > 0);
        }
    }
}

#[test]
fn bma_retries_missing_geometric_samples_and_counts_the_budget() {
    let f = field();
    let vars = Arc::new(vec![symbol!("x").into()]);
    let mut calls = 0;
    let (r, stats) = reconstruct_rational_function(
        f.clone(),
        vars.clone(),
        |f, p| {
            calls += 1;
            (calls != 2).then(|| f.add(&f.pow(&p[0], 100), &f.one()))
        },
        PolynomialBma,
        &ReconstructionOptions::default(),
    )
    .unwrap();
    assert_eq!(r.numerator.degree(0), 100);
    assert_eq!(stats.attempts, 2);
    assert_eq!(stats.poles, 1);
    assert_eq!(stats.probes, calls);
    for method in [PolynomialBma, HuMonagan] {
        let mut calls = 0;
        let result = reconstruct_rational_function(
            f.clone(),
            vars.clone(),
            |f, p| {
                calls += 1;
                Some(f.add(&f.pow(&p[0], 100), &f.one()))
            },
            method,
            &ReconstructionOptions {
                max_probes: 5,
                ..Default::default()
            },
        );
        assert_eq!(result.unwrap_err(), ReconstructionError::ProbeLimit);
        assert_eq!(calls, 5);
    }
}

#[test]
fn bma_rejects_unsupported_fields_and_noninjective_boxes_before_probing() {
    let vars = Arc::new(vec![symbol!("x").into(), symbol!("y").into()]);
    for (field, options) in [
        (Zp64::new(1_000_003), ReconstructionOptions::default()),
        (
            Zp64::new(65537),
            ReconstructionOptions {
                max_degree: 300,
                ..Default::default()
            },
        ),
        (
            field(),
            ReconstructionOptions {
                bma_degree_bounds: Some(vec![1]),
                ..Default::default()
            },
        ),
    ] {
        assert!(
            reconstruct_rational_function(
                field,
                vars.clone(),
                |_, _| panic!("unexpected oracle call"),
                PolynomialBma,
                &options
            )
            .is_err()
        );
    }
    // Individual bounds can make a small-field encoding injective even when
    // the uniform max-degree box would wrap around the multiplicative group.
    let f = Zp64::new(65537);
    let n: MultivariatePolynomial<_, u16> =
        parse!("x^3*y^2+7*x+11").to_polynomial(&f, vars.clone());
    let (result, _) = reconstruct_rational_function(
        f,
        vars,
        |_, point| Some(n.replace_all(point)),
        PolynomialBma,
        &ReconstructionOptions {
            max_degree: 300,
            bma_degree_bounds: Some(vec![3, 2]),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(result.numerator, n);
}

#[test]
fn bma_checks_a_misleading_zero_sequence_at_fresh_points() {
    use symbolica::rand::{Rng, SeedableRng, rngs::StdRng};
    let f = field();
    let vars = Arc::new(vec![symbol!("x").into()]);
    let options = ReconstructionOptions {
        seed: 17,
        ..Default::default()
    };
    let mut rng = StdRng::seed_from_u64(options.seed);
    let mut root = f.to_element(rng.random_range(1..f.get_prime()));
    let alpha = f.to_element(u64::from(
        SMOOTH_PRIMES
            .iter()
            .find(|(p, _, _)| *p == f.get_prime())
            .unwrap()
            .1,
    ));
    let x: MultivariatePolynomial<_, u16> = parse!("x").to_polynomial(&f, vars.clone());
    let mut n = x.one();
    for _ in 0..4 {
        n = &n * &(&x - &x.constant(root));
        f.mul_assign(&mut root, &alpha);
    }
    let (r, stats) = reconstruct_rational_function(
        f,
        vars,
        |_, point| Some(n.replace_all(point)),
        PolynomialBma,
        &options,
    )
    .unwrap();
    assert_eq!(r.numerator, n);
    assert_eq!(stats.attempts, 2);
}

#[test]
fn bma_lifts_polynomial_and_rational_coefficients_over_q() {
    let vars = Arc::new(vec![symbol!("x").into(), symbol!("y").into()]);
    for (method, denominator) in [(PolynomialBma, "1"), (HuMonagan, "y^2+3*y+7")] {
        let n: MultivariatePolynomial<_, u16> =
            parse!("(10^70+13)*x^55+17*x^3*y^2+19*y^7+23").to_polynomial(&Z, vars.clone());
        let d: MultivariatePolynomial<_, u16> = parse!(denominator).to_polynomial(&Z, vars.clone());
        let mut calls = 0;
        let (r, stats) = reconstruct_rational_function_over_q(
            vars.clone(),
            |f, p| {
                calls += 1;
                let d = d
                    .map_coeff(|c| c.to_finite_field(f), f.clone())
                    .replace_all(p);
                (!f.is_zero(&d)).then(|| {
                    f.div(
                        &n.map_coeff(|c| c.to_finite_field(f), f.clone())
                            .replace_all(p),
                        &d,
                    )
                })
            },
            method,
            &ReconstructionOptions {
                max_degree: 64,
                ..Default::default()
            },
            20,
        )
        .unwrap();
        assert_eq!(&r.numerator * &d, &r.denominator * &n);
        assert_eq!(stats.probes, calls);
        assert!(stats.successful_images > 2);
    }
}

#[test]
fn rational_bma_retries_a_zero_normalization_slice() {
    use symbolica::rand::{Rng, SeedableRng, rngs::StdRng};
    let f = field();
    let vars = Arc::new(vec![symbol!("x").into(), symbol!("y").into()]);
    let options = ReconstructionOptions {
        seed: 23,
        max_probes: 1000,
        ..Default::default()
    };
    let mut rng = StdRng::seed_from_u64(options.seed);
    let _ = rng.random_range(1..f.get_prime());
    let anchor = rng.random_range(1..f.get_prime());
    let n: MultivariatePolynomial<_, u16> =
        parse!(format!("(y-{anchor})*(x+1)")).to_polynomial(&f, vars.clone());
    let d: MultivariatePolynomial<_, u16> = parse!("x+y+3").to_polynomial(&f, vars.clone());
    let (r, stats) = reconstruct_rational_function(
        f,
        vars,
        |f, point| {
            let d = d.replace_all(point);
            (!f.is_zero(&d)).then(|| f.div(&n.replace_all(point), &d))
        },
        HuMonagan,
        &options,
    )
    .unwrap();
    assert_eq!(&r.numerator * &d, &r.denominator * &n);
    assert_eq!(stats.attempts, 2);
}
