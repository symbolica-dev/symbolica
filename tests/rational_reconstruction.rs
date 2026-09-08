use std::{cell::Cell, sync::Arc};
use symbolica::domains::finite_field::Zp64;
use symbolica::{
    poly::reconstruction::{
        ReconstructionError, ReconstructionMethod::*, ReconstructionOptions,
        reconstruct_rational_function,
    },
    prelude::*,
};

fn check(num: &str, den: &str, names: &[&str], seed: u64) {
    let field = Zp64::new(2_305_843_009_213_693_951);
    let vars: Arc<Vec<symbolica::poly::PolyVariable>> =
        Arc::new(names.iter().map(|s| symbol!(s).into()).collect());
    let n: MultivariatePolynomial<_, u16> = parse!(num).to_polynomial(&field, vars.clone());
    let d: MultivariatePolynomial<_, u16> = parse!(den).to_polynomial(&field, vars.clone());
    for method in [
        CuytLee,
        CuytLeePruned,
        BalancedZippel,
        BalancedZippelSeparated,
    ] {
        let calls = Cell::new(0);
        let (r, stats) = reconstruct_rational_function(
            field.clone(),
            vars.clone(),
            |f, x| {
                calls.set(calls.get() + 1);
                let dv = d.replace_all(x);
                (!f.is_zero(&dv)).then(|| f.div(&n.replace_all(x), &dv))
            },
            method,
            &ReconstructionOptions {
                seed,
                max_degree: 40,
                degree_race: true,
                ..Default::default()
            },
        )
        .unwrap_or_else(|e| panic!("{method:?}: {num}/{den}: {e}"));
        assert_eq!(
            &r.numerator * &d,
            &r.denominator * &n,
            "{method:?}: {num}/{den}"
        );
        assert_eq!(stats.probes, calls.get());
    }
}

#[test]
fn paper_examples_and_normalization() {
    check("x*y+2", "x*y-2*x+4", &["x", "y"], 1);
    check("3*x+7*y", "x+y+4*x*y", &["x", "y"], 2);
    check("x^3*y+2*z^2+7", "x*y^2+z+3", &["x", "y", "z"], 3);
    check("x^4+y^4+z^4", "x*y+y*z+x*z", &["x", "y", "z"], 4);
    check("x^3*y^2+z^3", "x^2*y^3+z^2", &["x", "y", "z"], 5);
}

#[test]
fn balanced_meets_paper_probe_budget() {
    let field = Zp64::new(2_305_843_009_213_693_951);
    let vars = Arc::new(vec![symbol!("y").into(), symbol!("d").into()]);
    let n: MultivariatePolynomial<_, u16> =
        parse!("(d+13)^30*(y^2+9)^7+1").to_polynomial(&field, vars.clone());
    let d: MultivariatePolynomial<_, u16> =
        parse!("(d-4)^29*(y^2-1)^5").to_polynomial(&field, vars.clone());
    for (method, degree_race, budget) in [
        (BalancedZippel, false, 451),
        (BalancedZippel, true, 448),
        (BalancedZippelSeparated, false, 306),
        (BalancedZippelSeparated, true, 303),
    ] {
        for seed in 0..10 {
            let mut calls = 0;
            let (r, stats) = reconstruct_rational_function(
                field.clone(),
                vars.clone(),
                |f, x| {
                    calls += 1;
                    let dv = d.replace_all(x);
                    (!f.is_zero(&dv)).then(|| f.div(&n.replace_all(x), &dv))
                },
                method,
                &ReconstructionOptions {
                    seed,
                    max_degree: 64,
                    degree_race,
                    max_probes: budget,
                    ..Default::default()
                },
            )
            .unwrap_or_else(|e| panic!("seed {seed}: {e}"));
            assert_eq!(&r.numerator * &d, &r.denominator * &n);
            assert_eq!(calls, stats.probes);
        }
    }
}

#[test]
fn completed_homogeneous_components_reduce_probes() {
    let field = Zp64::new(2_305_843_009_213_693_951);
    let vars = Arc::new(vec![
        symbol!("x").into(),
        symbol!("y").into(),
        symbol!("z").into(),
    ]);
    for (ns, ds) in [
        ("(1+x+y+z)^6", "1"),
        ("(1+x)^8+y^8+z^8", "1"),
        ("(1+x+y+z)^6-1", "y-z+(x*y*z)^3"),
    ] {
        let n: MultivariatePolynomial<_, u16> = parse!(ns).to_polynomial(&field, vars.clone());
        let d: MultivariatePolynomial<_, u16> = parse!(ds).to_polynomial(&field, vars.clone());
        for seed in 0..5 {
            let mut counts = Vec::new();
            for method in [CuytLee, CuytLeePruned] {
                let (r, s) = reconstruct_rational_function(
                    field.clone(),
                    vars.clone(),
                    |f, p| {
                        let dv = d.replace_all(p);
                        (!f.is_zero(&dv)).then(|| f.div(&n.replace_all(p), &dv))
                    },
                    method,
                    &ReconstructionOptions {
                        seed,
                        max_degree: 32,
                        ..Default::default()
                    },
                )
                .unwrap();
                assert_eq!(&r.numerator * &d, &r.denominator * &n);
                assert_eq!(s.attempts, 1);
                counts.push(s.probes);
            }
            assert!(counts[1] < counts[0], "{ns}/{ds}: {counts:?}");
        }
    }
}

#[test]
fn balanced_removes_learned_monomial_factors() {
    let field = Zp64::new(2_305_843_009_213_693_951);
    let vars = Arc::new(vec![symbol!("x").into(), symbol!("y").into()]);
    let n: MultivariatePolynomial<_, u16> =
        parse!("y^20*(x^4+x^3*y+x^2*y^2+x*y^3+y^4)").to_polynomial(&field, vars.clone());
    let d: MultivariatePolynomial<_, u16> =
        parse!("x^3+x^2*y+x*y^2+y^3+1").to_polynomial(&field, vars.clone());
    for seed in 0..10 {
        for inverse in [false, true] {
            let (n, d) = if inverse { (&d, &n) } else { (&n, &d) };
            let (r, stats) = reconstruct_rational_function(
                field.clone(),
                vars.clone(),
                |f, x| {
                    let dv = d.replace_all(x);
                    (!f.is_zero(&dv)).then(|| f.div(&n.replace_all(x), &dv))
                },
                BalancedZippel,
                &ReconstructionOptions {
                    seed,
                    max_degree: 40,
                    max_probes: 100,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(&r.numerator * d, &r.denominator * n);
            assert_eq!(stats.attempts, 1);
        }
    }
}

#[test]
fn degree_race_covers_medium_degree_denominators() {
    let field = Zp64::new(2_305_843_009_213_693_951);
    let vars = Arc::new(vec![symbol!("x").into()]);
    let n: MultivariatePolynomial<_, u16> = parse!("x^50+5").to_polynomial(&field, vars.clone());
    let d: MultivariatePolynomial<_, u16> = parse!("x^20+3").to_polynomial(&field, vars.clone());
    for seed in 0..3 {
        for (n, d) in [(&n, &d), (&d, &n)] {
            let (r, _) = reconstruct_rational_function(
                field.clone(),
                vars.clone(),
                |f, p| {
                    let dv = d.replace_all(p);
                    (!f.is_zero(&dv)).then(|| f.div(&n.replace_all(p), &dv))
                },
                BalancedZippel,
                &ReconstructionOptions {
                    seed,
                    degree_race: true,
                    max_degree: 64,
                    max_probes: 80,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(&r.numerator * d, &r.denominator * n);
        }
    }
}

#[test]
fn reciprocal_degree_race_handles_an_initial_zero() {
    let field = Zp64::new(2_305_843_009_213_693_951);
    let vars = Arc::new(vec![symbol!("x").into()]);
    let x: MultivariatePolynomial<_, u16> = parse!("x").to_polynomial(&field, vars.clone());
    let d: MultivariatePolynomial<_, u16> = parse!("x^30+5").to_polynomial(&field, vars.clone());
    for seed in 0..8 {
        // Choose the numerator root at the first query, then keep this same
        // rational function for every subsequent query and the exact checker.
        let root = Cell::new(None);
        let (r, _) = reconstruct_rational_function(
            field.clone(),
            vars.clone(),
            |f, p| {
                let root_value = root.get().unwrap_or_else(|| {
                    root.set(Some(p[0]));
                    p[0]
                });
                let dv = d.replace_all(p);
                (!f.is_zero(&dv)).then(|| f.div(&f.sub(&p[0], &root_value), &dv))
            },
            BalancedZippel,
            &ReconstructionOptions {
                seed,
                degree_race: true,
                max_degree: 40,
                max_probes: 45,
                ..Default::default()
            },
        )
        .unwrap();
        let n = &x - &x.constant(root.get().unwrap());
        assert_eq!(&r.numerator * &d, &r.denominator * &n);
    }
}

#[test]
fn unbalanced_degree_race_and_numerator_completion() {
    let field = Zp64::new(2_305_843_009_213_693_951);
    let vars = Arc::new(vec![symbol!("x").into()]);
    let n: MultivariatePolynomial<_, u16> = parse!("x^30+3").to_polynomial(&field, vars.clone());
    let d: MultivariatePolynomial<_, u16> = parse!("x^3+5").to_polynomial(&field, vars.clone());
    for seed in 0..8 {
        for (n, d) in [(&n, &d), (&d, &n)] {
            let (r, _) = reconstruct_rational_function(
                field.clone(),
                vars.clone(),
                |f, x| {
                    let dv = d.replace_all(x);
                    (!f.is_zero(&dv)).then(|| f.div(&n.replace_all(x), &dv))
                },
                BalancedZippel,
                &ReconstructionOptions {
                    seed,
                    max_degree: 40,
                    degree_race: true,
                    max_probes: 40,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(&r.numerator * d, &r.denominator * n);
        }
        check(
            "(x+1)*(y^5+3)",
            "x^4*(y^4+2)+x^2*(y+5)+3",
            &["x", "y"],
            seed,
        );
        check(
            "x^4*(y^4+2)+x^2*(y+5)+3",
            "(x+1)*(y^5+3)",
            &["x", "y"],
            seed,
        );
    }
}

#[test]
fn zero_constants_polynomials_and_cancellation() {
    for (n, d) in [
        ("0", "1"),
        ("7", "3"),
        ("x^7+3*x+1", "1"),
        ("1", "x^7+3*x+1"),
        ("x^2-1", "x-1"),
        ("x", "x"),
        ("1", "x^2"),
    ] {
        check(n, d, &["x"], 6);
    }
    check("0", "x+y", &["x", "y"], 7);
    check("7", "3", &["x", "y", "z"], 8);
    check("x^2-1", "x*y-y", &["x", "y"], 9);
    check("y^4+2*z+1", "y*z^2+5", &["x", "y", "z"], 10);
    check("(x+1)*(y+2)*(z+3)", "(x+1)*(x*y+z+1)", &["x", "y", "z"], 11);
}

#[test]
fn rejected_denominator_separation_falls_back_within_one_attempt() {
    let field = Zp64::new(2_305_843_009_213_693_951);
    let vars = Arc::new(vec![symbol!("x").into(), symbol!("y").into()]);
    let n: MultivariatePolynomial<_, u16> = parse!("x*y+3*x+5").to_polynomial(&field, vars.clone());
    let d: MultivariatePolynomial<_, u16> = parse!("x+y+7").to_polynomial(&field, vars.clone());
    let (r, stats) = reconstruct_rational_function(
        field,
        vars,
        |f, x| {
            let dv = d.replace_all(x);
            (!f.is_zero(&dv)).then(|| f.div(&n.replace_all(x), &dv))
        },
        BalancedZippelSeparated,
        &ReconstructionOptions {
            seed: 1,
            max_attempts: 1,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(&r.numerator * &d, &r.denominator * &n);
    assert_eq!(stats.attempts, 1);
    assert_eq!(stats.separation_fallbacks, 1);
}

#[test]
fn many_seeds_and_variable_orders() {
    for seed in 0..8 {
        check(
            "x^3*y+2*y*z+3*w+5",
            "y^2*w+z^2+x+1",
            &["x", "y", "z", "w"],
            seed,
        );
        check(
            "x^3*y+2*y*z+3*w+5",
            "y^2*w+z^2+x+1",
            &["w", "z", "y", "x"],
            seed,
        );
    }
}

#[test]
fn bounded_failure_and_poles() {
    let field = Zp64::new(1_000_003);
    let vars: Arc<Vec<symbolica::poly::PolyVariable>> = Arc::new(vec![symbol!("x").into()]);
    for method in [
        CuytLee,
        CuytLeePruned,
        BalancedZippel,
        BalancedZippelSeparated,
    ] {
        let mut calls = 0;
        let r = reconstruct_rational_function(
            field.clone(),
            vars.clone(),
            |_, _| {
                calls += 1;
                None
            },
            method,
            &ReconstructionOptions {
                max_probes: 5,
                ..Default::default()
            },
        );
        assert_eq!(r.unwrap_err(), ReconstructionError::ProbeLimit);
        assert_eq!(calls, 5);
        let r = reconstruct_rational_function(
            field.clone(),
            vars.clone(),
            |f, x| Some(f.pow(&x[0], 5)),
            method,
            &ReconstructionOptions {
                max_degree: 2,
                ..Default::default()
            },
        );
        assert_eq!(r.unwrap_err(), ReconstructionError::AttemptsExhausted);
        let r = reconstruct_rational_function(
            field.clone(),
            vars.clone(),
            |_, _| unreachable!(),
            method,
            &ReconstructionOptions {
                verification_points: 0,
                ..Default::default()
            },
        );
        assert_eq!(r.unwrap_err(), ReconstructionError::InvalidOptions);
        // A deterministic oracle with many unavailable points must still reconstruct.
        let (r, stats) = reconstruct_rational_function(
            field.clone(),
            vars.clone(),
            |f, x| {
                (!f.is_zero(&x[0]) && f.from_element(&x[0]) % 3 == 0)
                    .then(|| f.div(&f.add(&x[0], &f.one()), &x[0]))
            },
            method,
            &ReconstructionOptions::default(),
        )
        .unwrap();
        assert!(stats.poles > 0);
        let x: MultivariatePolynomial<_, u16> = parse!("x").to_polynomial(&field, vars.clone());
        assert_eq!(&r.numerator * &x, &r.denominator * &(x.clone() + x.one()));
    }
}

#[test]
fn lift_large_rational_coefficients() {
    use symbolica::domains::finite_field::ToFiniteField;
    use symbolica::poly::reconstruction::reconstruct_rational_function_over_q;
    let vars: Arc<Vec<symbolica::poly::PolyVariable>> =
        Arc::new(vec![symbol!("x").into(), symbol!("y").into()]);
    let n: MultivariatePolynomial<_, u16> =
        parse!("1234567890123456789012345678901234567/37*x^2+7/19*y+11")
            .to_polynomial(&Q, vars.clone());
    let d: MultivariatePolynomial<_, u16> =
        parse!("13*x*y+17/23*y^2+29").to_polynomial(&Q, vars.clone());
    for method in [
        CuytLee,
        CuytLeePruned,
        BalancedZippel,
        BalancedZippelSeparated,
    ] {
        let mut calls = 0;
        let (r, stats) = reconstruct_rational_function_over_q(
            vars.clone(),
            |f, p| {
                calls += 1;
                let nv = n
                    .map_coeff(|c| c.to_finite_field(f), f.clone())
                    .replace_all(p);
                let dv = d
                    .map_coeff(|c| c.to_finite_field(f), f.clone())
                    .replace_all(p);
                (!f.is_zero(&dv)).then(|| f.div(&nv, &dv))
            },
            method,
            &ReconstructionOptions::default(),
            12,
        )
        .unwrap();
        assert!(stats.successful_images >= 2);
        assert_eq!(stats.probes, calls);
        let rn = r.numerator.map_coeff(|c| Rational::from(c.clone()), Q);
        let rd = r.denominator.map_coeff(|c| Rational::from(c.clone()), Q);
        assert_eq!(&rn * &d, &rd * &n);
    }
}

#[test]
fn unlucky_anchor_is_retried() {
    use rand::{Rng, SeedableRng, rngs::StdRng};
    let field = Zp64::new(1_000_003);
    let options = ReconstructionOptions {
        max_degree: 8,
        seed: 92,
        ..Default::default()
    };
    let vars: Arc<Vec<symbolica::poly::PolyVariable>> =
        Arc::new(vec![symbol!("x").into(), symbol!("y").into()]);
    let mut rng = StdRng::seed_from_u64(options.seed);
    let _ = rng.random_range(1..1_000_003u64);
    let unlucky_y = field.to_element(rng.random_range(1..1_000_003u64));
    let (r, stats) = reconstruct_rational_function(
        field.clone(),
        vars.clone(),
        |f, p| Some(f.add(&f.mul(&p[0], &f.sub(&p[1], &unlucky_y)), &f.one())),
        BalancedZippel,
        &options,
    )
    .unwrap();
    assert!(stats.attempts > 1);
    let mut expected = MultivariatePolynomial::<_, u16>::new(&field, None, vars);
    expected.append_monomial(field.one(), &[1, 1]);
    expected.append_monomial(field.neg(&unlucky_y), &[1, 0]);
    expected.append_monomial(field.one(), &[0, 0]);
    assert_eq!(r.numerator, expected);
    assert!(r.denominator.is_one());
}

#[test]
fn generated_sparse_functions_over_a_smaller_prime() {
    use rand::{Rng, SeedableRng, rngs::StdRng};
    let field = Zp64::new(1_000_003);
    let mut rng = StdRng::seed_from_u64(1984);
    for case in 0..24 {
        let nvars = 1 + case % 4;
        let vars: Arc<Vec<symbolica::poly::PolyVariable>> = Arc::new(
            ["x", "y", "z", "w"][..nvars]
                .iter()
                .map(|s| symbol!(*s).into())
                .collect(),
        );
        let mut n = MultivariatePolynomial::<_, u16>::new(&field, None, vars.clone());
        let mut d = n.zero();
        for p in [&mut n, &mut d] {
            for _ in 0..rng.random_range(1..9) {
                let ex: Vec<_> = (0..nvars).map(|_| rng.random_range(0..4u16)).collect();
                p.append_monomial(field.to_element(rng.random_range(1..1_000_003)), &ex);
            }
        }
        assert!(!d.is_zero());
        for method in [
            CuytLee,
            CuytLeePruned,
            BalancedZippel,
            BalancedZippelSeparated,
        ] {
            let (r, _) = reconstruct_rational_function(
                field.clone(),
                vars.clone(),
                |f, p| {
                    let dv = d.replace_all(p);
                    (!f.is_zero(&dv)).then(|| f.div(&n.replace_all(p), &dv))
                },
                method,
                &ReconstructionOptions {
                    max_degree: 16,
                    degree_race: true,
                    seed: case as u64,
                    ..Default::default()
                },
            )
            .unwrap();
            assert_eq!(
                &r.numerator * &d,
                &r.denominator * &n,
                "case {case}, {method:?}"
            );
        }
    }
}

#[test]
fn lifting_recovers_after_an_unlucky_prime() {
    use symbolica::domains::finite_field::{PrimeIteratorU64, ToFiniteField};
    use symbolica::poly::reconstruction::reconstruct_rational_function_over_q;
    let first_prime = PrimeIteratorU64::new(1 << 61).next().unwrap();
    let vars: Arc<Vec<symbolica::poly::PolyVariable>> = Arc::new(vec![symbol!("x").into()]);
    let n: MultivariatePolynomial<_, u16> = parse!("x^2").to_polynomial(&Z, vars.clone());
    let n = n.clone().mul_coeff(Integer::from(first_prime)) + n.one();
    let d: MultivariatePolynomial<_, u16> = parse!("x+3").to_polynomial(&Z, vars.clone());
    let (r, stats) = reconstruct_rational_function_over_q(
        vars,
        |f, p| {
            let nv = n
                .map_coeff(|c| c.to_finite_field(f), f.clone())
                .replace_all(p);
            let dv = d
                .map_coeff(|c| c.to_finite_field(f), f.clone())
                .replace_all(p);
            (!f.is_zero(&dv)).then(|| f.div(&nv, &dv))
        },
        BalancedZippel,
        &ReconstructionOptions::default(),
        12,
    )
    .unwrap();
    assert!(stats.support_resets > 0);
    assert_eq!(&r.numerator * &d, &r.denominator * &n);
}
