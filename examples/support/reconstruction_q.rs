use super::*;
#[cfg(feature = "native_code_generation")]
#[path = "trace_oracle.rs"]
mod trace_oracle;
use std::collections::BTreeMap;
use symbolica::{
    domains::finite_field::ToFiniteField,
    poly::reconstruction::reconstruct_rational_function_over_q,
};

#[derive(Debug)]
struct ProbeLimit;

pub(super) fn run(
    case: &str,
    source: &str,
    names: &[String],
    vars: Arc<Vec<PolyVariable>>,
    method: ReconstructionMethod,
    seed: u64,
) {
    let setup = Instant::now();
    let original: RationalPolynomial<_, u16> = parse!(source.trim().trim_end_matches(';'))
        .to_rational_polynomial(&Q, &Z, Some(vars.clone()));
    if let Ok(path) = std::env::var("EXPORT_ORACLE") {
        let mut out = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        writeln!(
            out,
            "{} 0 {} {}",
            names.len(),
            original.numerator.nterms(),
            original.denominator.nterms()
        )
        .unwrap();
        writeln!(out, "{}", names.join(" ")).unwrap();
        for p in [&original.numerator, &original.denominator] {
            for term in p {
                write!(out, "{}", term.coefficient).unwrap();
                for e in term.exponents {
                    write!(out, " {e}").unwrap();
                }
                writeln!(out).unwrap();
            }
        }
        return;
    }
    #[cfg(feature = "native_code_generation")]
    let mut trace = trace_oracle::TraceOracle::from_env(names);
    #[cfg(not(feature = "native_code_generation"))]
    assert!(
        std::env::var_os("TRACE_ORACLE_PATH").is_none(),
        "trace oracle requires native_code_generation"
    );
    let setup_ms = setup.elapsed().as_secs_f64() * 1000.;
    let timeout = std::env::var("BENCH_TIMEOUT")
        .map(|s| s.parse::<f64>().unwrap())
        .unwrap_or(180.);
    let cap = std::env::var("MAX_TOTAL_PROBES")
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(2_000_000);
    let max_primes = std::env::var("MAX_PRIMES")
        .map(|s| s.parse().unwrap())
        .unwrap_or(32);
    let options = ReconstructionOptions {
        seed,
        max_degree: 512,
        max_attempts: 2,
        max_probes: std::env::var("MAX_PROBES")
            .map(|s| s.parse().unwrap())
            .unwrap_or(200_000),
        degree_race: std::env::var_os("RECONSTRUCTION_DEGREE_RACE").is_some(),
        reuse_coefficients: std::env::var_os("RECONSTRUCTION_NO_REUSE").is_none(),
        reuse_rational_factors: std::env::var_os("RECONSTRUCTION_NO_FACTOR_REUSE").is_none(),
        reuse_row_support: std::env::var_os("RECONSTRUCTION_DENSE_ROWS").is_none(),
        ..Default::default()
    };
    let mut image: Option<(u64, RationalPolynomial<Zp64, u16>, CachedOracle)> = None;
    let mut probes_by_prime = BTreeMap::<u64, usize>::new();
    let mut calls = 0;
    let old_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if !info.payload().is::<TimeLimit>() && !info.payload().is::<ProbeLimit>() {
            old_hook(info);
        }
    }));
    let start = Instant::now();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        reconstruct_rational_function_over_q(
            vars,
            |f, point| {
                if start.elapsed().as_secs_f64() > timeout {
                    std::panic::panic_any(TimeLimit);
                }
                if calls >= cap {
                    std::panic::panic_any(ProbeLimit);
                }
                calls += 1;
                *probes_by_prime.entry(f.get_prime()).or_default() += 1;
                #[cfg(feature = "native_code_generation")]
                if let Some(trace) = &mut trace {
                    return trace.evaluate(f, point);
                }
                if image
                    .as_ref()
                    .is_none_or(|(prime, _, _)| *prime != f.get_prime())
                {
                    let modular = RationalPolynomial {
                        numerator: original
                            .numerator
                            .map_coeff(|c| c.to_finite_field(f), f.clone()),
                        denominator: original
                            .denominator
                            .map_coeff(|c| c.to_finite_field(f), f.clone()),
                    };
                    let cached = CachedOracle::new(&modular);
                    image = Some((f.get_prime(), modular, cached));
                }
                let (_, modular, cached) = image.as_mut().unwrap();
                cached.evaluate(modular, point)
            },
            method,
            &options,
            max_primes,
        )
    }));
    let elapsed_us = start.elapsed().as_secs_f64() * 1e6;
    #[cfg(feature = "native_code_generation")]
    if let Some(trace) = &trace {
        assert_eq!(trace.calls(), calls);
    }
    let mut selected = String::new();
    let mut factor_reductions = String::new();
    let (status, primes, images, reuses, fallbacks) = match result {
        Ok(Ok((r, stats))) => {
            factor_reductions = stats.factor_reductions.to_string();
            selected = stats
                .selected_methods
                .iter()
                .map(|m| format!("{m:?}"))
                .collect::<Vec<_>>()
                .join(";");
            assert_eq!(stats.probes, calls);
            assert_eq!(
                &r.numerator * &original.denominator,
                &r.denominator * &original.numerator
            );
            (
                "ok".to_string(),
                stats.primes,
                stats.successful_images.to_string(),
                stats.support_reuses.to_string(),
                stats.support_fallbacks.to_string(),
            )
        }
        Ok(Err(e)) => (
            format!("{e:?}"),
            probes_by_prime.len(),
            String::new(),
            String::new(),
            String::new(),
        ),
        Err(e) if e.is::<TimeLimit>() => (
            "time_limit".into(),
            probes_by_prime.len(),
            String::new(),
            String::new(),
            String::new(),
        ),
        Err(e) if e.is::<ProbeLimit>() => (
            "probe_limit".into(),
            probes_by_prime.len(),
            String::new(),
            String::new(),
            String::new(),
        ),
        Err(e) => std::panic::resume_unwind(e),
    };
    let distribution = probes_by_prime
        .iter()
        .map(|(p, n)| format!("{p}:{n}"))
        .collect::<Vec<_>>()
        .join(";");
    println!(
        "case,method,seed,status,elapsed_us,probes,primes,images,support_reuses,support_fallbacks,probes_by_prime,setup_ms,num_terms,den_terms,selected_methods,factor_reductions"
    );
    println!(
        "{case},{method:?},{seed},{status},{elapsed_us:.3},{calls},{primes},{images},{reuses},{fallbacks},{distribution},{setup_ms:.3},{},{},{selected},{factor_reductions}",
        original.numerator.nterms(),
        original.denominator.nterms()
    );
}
