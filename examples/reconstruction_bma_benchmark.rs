//! Same-prime sample-count comparisons for the GCD-based BMA reconstruction route.
//! Usage: reconstruction_bma_benchmark [case substring] [repeats, default 3]
//! Each success is checked by exact polynomial cross multiplication.
use std::{sync::Arc, time::Instant};
use symbolica::{
    domains::finite_field::SMOOTH_PRIMES,
    poly::{
        PolyVariable,
        reconstruction::{
            ReconstructionMethod::*, ReconstructionOptions, reconstruct_rational_function,
        },
    },
    prelude::*,
};

struct Case {
    name: String,
    expression: String,
    variables: Vec<String>,
    degree: u16,
    polynomial: bool,
}

#[derive(Debug)]
struct TimeLimit;

fn main() {
    let old_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if !info.payload().is::<TimeLimit>() {
            old_hook(info);
        }
    }));
    let timeout = std::env::var("BENCH_TIMEOUT")
        .map(|s| s.parse::<f64>().unwrap())
        .unwrap_or(60.);
    let args: Vec<_> = std::env::args().collect();
    let filter = args.get(1).map(String::as_str).unwrap_or("");
    let repeats = args.get(2).map(|s| s.parse::<u64>().unwrap()).unwrap_or(3);
    let prime = SMOOTH_PRIMES
        .iter()
        .find(|(p, _, _)| *p > 1 << 61)
        .unwrap()
        .0;
    let field = Zp64::new(prime);
    let mut cases = Vec::new();
    for degree in [5, 50, 500, 2000] {
        cases.push(Case {
            name: format!("polynomial_sparse3_{degree}"),
            expression: format!("3*x^{degree}*y^2+5*y^{degree}*z^3+7*z^{degree}*x+11"),
            variables: ["x", "y", "z"].map(String::from).to_vec(),
            degree: 2000,
            polynomial: true,
        });
    }
    for degree in [5, 50, 500] {
        cases.push(Case {
            name: format!("rational_sparse3_{degree}"),
            expression: format!("(3*x^{degree}+5*y^{degree}+7*z^2+11)/(z^2+3*z+7)"),
            variables: ["x", "y", "z"].map(String::from).to_vec(),
            degree: 512,
            polynomial: false,
        });
    }
    for (name, expression, variables, polynomial) in [
        ("polynomial_dense2", "(x+y+1)^10", vec!["x", "y"], true),
        (
            "polynomial_dense3",
            "(x+y+z+1)^6",
            vec!["x", "y", "z"],
            true,
        ),
        (
            "rational_mixed3",
            "(x^7*y^3+z^3+2*x+5)/(x^4*z^2+y^5*z+3*x*y+7)",
            vec!["x", "y", "z"],
            false,
        ),
        (
            "rational_dense3",
            "((x+y+z+1)^4+7)/(x+2*y+3*z+5)^3",
            vec!["x", "y", "z"],
            false,
        ),
        (
            "paper_eq28",
            "((d+13)^30*(y^2+9)^7+1)/((d-4)^29*(y^2-1)^5)",
            vec!["y", "d"],
            false,
        ),
    ] {
        cases.push(Case {
            name: name.into(),
            expression: expression.into(),
            variables: variables.into_iter().map(String::from).collect(),
            degree: 128,
            polynomial,
        });
    }
    // Existing independently generated Kira/Ratracer fixtures. Missing files
    // are reported, so a synthetic-only checkout is not mistaken for IBP coverage.
    for family in ["box2l", "diamond3l", "xbox2l2m", "tth2l_b16"] {
        let name = format!("ibp_{family}_rank0001");
        let folder = format!("target/reconstruction-external/ibp-inputs/{family}");
        let path = std::path::Path::new(&folder).join(&name);
        if !path.exists() {
            eprintln!("IBP fixture unavailable: {}", path.display());
            continue;
        }
        cases.push(Case {
            name: name.clone(),
            expression: std::fs::read_to_string(&path).unwrap(),
            variables: std::fs::read_to_string(path.with_file_name(format!("{name}.variables")))
                .unwrap()
                .split_whitespace()
                .map(String::from)
                .collect(),
            degree: 128,
            polynomial: false,
        });
    }
    let _ = symbol!("bma_benchmark_init");
    println!(
        "case,method,seed,prime,status,probes,elapsed_us,num_terms,den_terms,bma_sequences,oracle"
    );
    for case in cases.into_iter().filter(|c| c.name.contains(filter)) {
        let vars: Arc<Vec<PolyVariable>> = Arc::new(
            case.variables
                .iter()
                .map(|v| symbol!(v.as_str()).into())
                .collect(),
        );
        let original: RationalPolynomial<_, u16> = parse!(
            case.expression.trim().trim_end_matches(';')
        )
        .to_rational_polynomial(&field, &field, Some(vars.clone()));
        let cached = original.numerator.nterms() + original.denominator.nterms() > 64;
        let mut powers: Vec<Vec<_>> = (0..case.variables.len())
            .map(|i| {
                vec![
                    field.one();
                    1 + original
                        .numerator
                        .degree(i)
                        .max(original.denominator.degree(i)) as usize
                ]
            })
            .collect();
        let mut methods = vec![Automatic, BalancedZippel, HuMonagan];
        if case.polynomial {
            methods.push(PolynomialBma);
        }
        for seed in 1..=repeats {
            let rotate = seed as usize % methods.len();
            methods.rotate_left(rotate);
            for &method in &methods {
                let options = ReconstructionOptions {
                    max_degree: case.degree,
                    max_probes: std::env::var("MAX_PROBES")
                        .map(|s| s.parse().unwrap())
                        .unwrap_or(100_000),
                    max_attempts: 2,
                    seed,
                    ..Default::default()
                };
                let mut probes = 0;
                let start = Instant::now();
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    reconstruct_rational_function(
                        field.clone(),
                        vars.clone(),
                        |f, point| {
                            if start.elapsed().as_secs_f64() > timeout {
                                std::panic::panic_any(TimeLimit);
                            }
                            probes += 1;
                            if cached {
                                for (powers, x) in powers.iter_mut().zip(point) {
                                    for i in 1..powers.len() {
                                        powers[i] = f.mul(&powers[i - 1], x);
                                    }
                                }
                                let evaluate = |poly: &MultivariatePolynomial<_, u16>| {
                                    poly.into_iter().fold(f.zero(), |sum, term| {
                                        let mut value = *term.coefficient;
                                        for (powers, &e) in powers.iter().zip(term.exponents) {
                                            if e != 0 {
                                                f.mul_assign(&mut value, &powers[e as usize]);
                                            }
                                        }
                                        f.add(&sum, &value)
                                    })
                                };
                                let d = evaluate(&original.denominator);
                                return (!f.is_zero(&d))
                                    .then(|| f.div(&evaluate(&original.numerator), &d));
                            }
                            let d = original.denominator.replace_all(point);
                            (!f.is_zero(&d))
                                .then(|| f.div(&original.numerator.replace_all(point), &d))
                        },
                        method,
                        &options,
                    )
                }));
                let elapsed = start.elapsed().as_secs_f64() * 1e6;
                let (status, sequences) = match result {
                    Ok(Ok((r, stats))) => {
                        assert_eq!(stats.probes, probes);
                        assert_eq!(
                            &r.numerator * &original.denominator,
                            &r.denominator * &original.numerator
                        );
                        ("ok".into(), stats.bma_sequences)
                    }
                    Ok(Err(error)) => (format!("{error:?}"), 0),
                    Err(error) if error.is::<TimeLimit>() => ("time_limit".into(), 0),
                    Err(error) => std::panic::resume_unwind(error),
                };
                println!(
                    "{},{method:?},{seed},{prime},{status},{probes},{elapsed:.3},{},{},{sequences},{}",
                    case.name,
                    original.numerator.nterms(),
                    original.denominator.nterms(),
                    if cached {
                        "cached_powers"
                    } else {
                        "direct_evaluation"
                    },
                );
            }
        }
    }
}
