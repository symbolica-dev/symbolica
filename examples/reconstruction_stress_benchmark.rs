//! One bounded reconstruction of a public or generated stress-test function.
//! Usage: reconstruction_stress_benchmark CASE METHOD SEED [variables in reverse order: reverse]
use std::{cell::Cell, sync::Arc, time::Instant};
use symbolica::{
    domains::finite_field::Zp64,
    poly::{
        PolyVariable,
        reconstruction::{
            ReconstructionMethod, ReconstructionOptions, reconstruct_rational_function,
        },
    },
    prelude::*,
};

#[derive(Debug)]
struct TimeLimit;

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let case = args.get(1).expect("case");
    let method = match args.get(2).map(String::as_str) {
        Some("CuytLee") => ReconstructionMethod::CuytLee,
        Some("BalancedZippel") => ReconstructionMethod::BalancedZippel,
        Some("BalancedZippelSeparated") => ReconstructionMethod::BalancedZippelSeparated,
        _ => panic!("method must be CuytLee, BalancedZippel, or BalancedZippelSeparated"),
    };
    let seed = args.get(3).expect("seed").parse::<u64>().unwrap();
    let (source, mut names): (String, Vec<String>) = match case.as_str() {
        "firefly_f1" => (std::fs::read_to_string("target/reconstruction-external/firefly/benchmarks/f1.m").unwrap(), (1..=20).map(|i| format!("x{i}")).collect()),
        "firefly_f2" | "firefly_f3" | "firefly_f4" => (std::fs::read_to_string(format!("target/reconstruction-external/firefly/benchmarks/{}.m", &case[8..])).unwrap(), (1..=5).map(|i| format!("x{i}")).collect()),
        "aajamp" | "aajamp_mod" => (std::fs::read_to_string(format!("target/reconstruction-external/scaling-rec/data/{case}")).unwrap(), ["x23","x34","x45","x51"].map(String::from).to_vec()),
        "coeff_prop_4l" | "coeff_prop_4l_mod" => (std::fs::read_to_string(format!("target/reconstruction-external/scaling-rec/data/{case}")).unwrap(), ["z","d"].map(String::from).to_vec()),
        "mixed_sparse5" => ("(x^37*y^3+7*y^29*z^2+11*z^23*w+13*w^19*v^2+17*v^17*x+19*x*y*z*w*v+23)/(x^11*z^7+3*y^13*w^2+5*z^9*v^3+7*w^7*x^2+11*v^5*y+13)".into(), ["x","y","z","w","v"].map(String::from).to_vec()),
        "separated_dense4" => ("((x+2)^5*(y+3)^4*(z+5)^3*(w+7)^8+x*y*z+1)/((x+y+3)^3*(z+5)^2*(w-11)^7)".into(), ["x","y","z","w"].map(String::from).to_vec()),
        _ => panic!("unknown case"),
    };
    let reverse = args.get(4).is_some_and(|x| x == "reverse");
    if reverse {
        names.reverse();
    }
    let field = Zp64::new(2_305_843_009_213_693_951);
    let vars: Arc<Vec<PolyVariable>> =
        Arc::new(names.iter().map(|s| symbol!(s.as_str()).into()).collect());
    let setup = Instant::now();
    let original: RationalPolynomial<_, u16> = parse!(source.trim().trim_end_matches(';'))
        .to_rational_polynomial(&field, &field, Some(vars.clone()));
    let setup_ms = setup.elapsed().as_secs_f64() * 1000.;
    let timeout = std::env::var("BENCH_TIMEOUT")
        .map(|s| s.parse::<f64>().unwrap())
        .unwrap_or(120.);
    let options = ReconstructionOptions {
        seed,
        max_degree: 512,
        degree_race: std::env::var_os("RECONSTRUCTION_DEGREE_RACE").is_some(),
        max_probes: std::env::var("MAX_PROBES")
            .map(|s| s.parse().unwrap())
            .unwrap_or(200_000),
        max_attempts: 2,
        ..Default::default()
    };
    let calls = Cell::new(0usize);
    let start = Instant::now();
    let old_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if !info.payload().is::<TimeLimit>() {
            old_hook(info);
        }
    }));
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        reconstruct_rational_function(
            field,
            vars,
            |f, p| {
                if start.elapsed().as_secs_f64() > timeout {
                    std::panic::panic_any(TimeLimit);
                }
                calls.set(calls.get() + 1);
                let d = original.denominator.replace_all(p);
                (!f.is_zero(&d)).then(|| f.div(&original.numerator.replace_all(p), &d))
            },
            method,
            &options,
        )
    }));
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
    let (status, attempts) = match result {
        Ok(Ok((r, s))) => {
            assert_eq!(s.probes, calls.get());
            assert_eq!(
                &r.numerator * &original.denominator,
                &r.denominator * &original.numerator
            );
            ("ok".to_string(), s.attempts)
        }
        Ok(Err(e)) => (format!("{e:?}"), 0),
        Err(e) if e.is::<TimeLimit>() => ("time_limit".into(), 0),
        Err(e) => std::panic::resume_unwind(e),
    };
    println!(
        "case,method,seed,order,status,setup_ms,elapsed_us,probes,attempts,num_terms,den_terms"
    );
    println!(
        "{case},{method:?},{seed},{},{status},{setup_ms:.3},{:.3},{},{},{},{}",
        if reverse { "reverse" } else { "original" },
        elapsed_ms * 1000.,
        calls.get(),
        attempts,
        original.numerator.nterms(),
        original.denominator.nterms()
    );
}
