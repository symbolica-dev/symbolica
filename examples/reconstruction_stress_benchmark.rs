//! One bounded reconstruction of a public or generated stress-test function.
//! Usage: reconstruction_stress_benchmark CASE METHOD SEED [variables in reverse order: reverse]
//! Custom cases read BENCH_INPUT_DIR/CASE and its whitespace-separated CASE.variables sidecar.
use std::{cell::Cell, io::Write, sync::Arc, time::Instant};
use symbolica::{
    domains::finite_field::{FiniteFieldCore, FiniteFieldElement, Zp64},
    poly::{
        PolyVariable,
        reconstruction::{
            ReconstructionMethod, ReconstructionOptions, reconstruct_rational_function,
        },
    },
    prelude::*,
};

#[path = "support/reconstruction_q.rs"]
mod q_benchmark;

#[derive(Debug)]
struct TimeLimit;

// Benchmark-only oracle: identical power-table/term-loop algorithm to the
// external FireFly adapter. Source supports are never given to reconstruction.
struct CachedOracle {
    powers: Vec<Vec<FiniteFieldElement<u64>>>,
}

impl CachedOracle {
    fn new(r: &RationalPolynomial<Zp64, u16>) -> Self {
        Self {
            powers: (0..r.numerator.nvars())
                .map(|i| {
                    vec![
                        r.numerator.ring().one();
                        1 + r.numerator.degree(i).max(r.denominator.degree(i)) as usize
                    ]
                })
                .collect(),
        }
    }

    fn evaluate(
        &mut self,
        r: &RationalPolynomial<Zp64, u16>,
        point: &[FiniteFieldElement<u64>],
    ) -> Option<FiniteFieldElement<u64>> {
        let f = r.numerator.ring();
        for (powers, x) in self.powers.iter_mut().zip(point) {
            for i in 1..powers.len() {
                powers[i] = f.mul(&powers[i - 1], x);
            }
        }
        let eval = |p: &MultivariatePolynomial<Zp64, u16>| {
            let mut result = f.zero();
            for term in p {
                let mut c = *term.coefficient;
                for (powers, &e) in self.powers.iter().zip(term.exponents) {
                    if e != 0 {
                        f.mul_assign(&mut c, &powers[e as usize]);
                    }
                }
                f.add_assign(&mut result, &c);
            }
            result
        };
        let d = eval(&r.denominator);
        (!f.is_zero(&d)).then(|| f.div(&eval(&r.numerator), &d))
    }
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let case = args.get(1).expect("case");
    let method = match args.get(2).map(String::as_str) {
        Some("Automatic") => ReconstructionMethod::Automatic,
        Some("CuytLee") => ReconstructionMethod::CuytLee,
        Some("CuytLeePruned") => ReconstructionMethod::CuytLeePruned,
        Some("BalancedZippel") => ReconstructionMethod::BalancedZippel,
        Some("BalancedZippelSeparated") => ReconstructionMethod::BalancedZippelSeparated,
        _ => panic!(
            "method must be Automatic, CuytLee, CuytLeePruned, BalancedZippel, or BalancedZippelSeparated"
        ),
    };
    let seed = args.get(3).expect("seed").parse::<u64>().unwrap();
    let (source, mut names): (String, Vec<String>) = match case.as_str() {
        "firefly_f1" => (std::fs::read_to_string("target/reconstruction-external/firefly/benchmarks/f1.m").unwrap(), (1..=20).map(|i| format!("x{i}")).collect()),
        "firefly_f2" | "firefly_f3" | "firefly_f4" => (std::fs::read_to_string(format!("target/reconstruction-external/firefly/benchmarks/{}.m", &case[8..])).unwrap(), (1..=5).map(|i| format!("x{i}")).collect()),
        "aajamp" | "aajamp_mod" => (std::fs::read_to_string(format!("target/reconstruction-external/scaling-rec/data/{case}")).unwrap(), ["x23","x34","x45","x51"].map(String::from).to_vec()),
        "coeff_prop_4l" | "coeff_prop_4l_mod" => (std::fs::read_to_string(format!("target/reconstruction-external/scaling-rec/data/{case}")).unwrap(), ["z","d"].map(String::from).to_vec()),
        "fire7_nb0_largest" => (std::fs::read_to_string("target/reconstruction-external/fire-table-inputs/fire7_nb0_largest").unwrap(), ["u","v","w","d"].map(String::from).to_vec()),
        "mixed_sparse5" => ("(x^37*y^3+7*y^29*z^2+11*z^23*w+13*w^19*v^2+17*v^17*x+19*x*y*z*w*v+23)/(x^11*z^7+3*y^13*w^2+5*z^9*v^3+7*w^7*x^2+11*v^5*y+13)".into(), ["x","y","z","w","v"].map(String::from).to_vec()),
        "separated_dense4" => ("((x+2)^5*(y+3)^4*(z+5)^3*(w+7)^8+x*y*z+1)/((x+y+3)^3*(z+5)^2*(w-11)^7)".into(), ["x","y","z","w"].map(String::from).to_vec()),
        _ => {
            let directory = std::path::PathBuf::from(
                std::env::var("BENCH_INPUT_DIR").expect("unknown case; set BENCH_INPUT_DIR for custom inputs"),
            );
            let source = std::fs::read_to_string(directory.join(case)).expect("custom expression file");
            let names = std::fs::read_to_string(directory.join(format!("{case}.variables")))
                .expect("custom variable-order file")
                .split_whitespace().map(String::from).collect();
            (source, names)
        },
    };
    let reverse = args.get(4).is_some_and(|x| x == "reverse");
    if reverse {
        names.reverse();
    }
    let field = Zp64::new(
        std::env::var("BENCH_PRIME")
            .map(|s| s.parse().unwrap())
            .unwrap_or(2_305_843_009_213_693_951),
    );
    let vars: Arc<Vec<PolyVariable>> =
        Arc::new(names.iter().map(|s| symbol!(s.as_str()).into()).collect());
    if std::env::var_os("RECONSTRUCTION_OVER_Q").is_some() {
        q_benchmark::run(case, &source, &names, vars, method, seed);
        return;
    }
    let setup = Instant::now();
    let original: RationalPolynomial<_, u16> = parse!(source.trim().trim_end_matches(';'))
        .to_rational_polynomial(&field, &field, Some(vars.clone()));
    if let Ok(path) = std::env::var("EXPORT_ORACLE") {
        let mut out = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
        writeln!(
            out,
            "{} {} {} {}",
            names.len(),
            field.get_prime(),
            original.numerator.nterms(),
            original.denominator.nterms()
        )
        .unwrap();
        writeln!(out, "{}", names.join(" ")).unwrap();
        for p in [&original.numerator, &original.denominator] {
            for term in p {
                write!(out, "{}", field.from_element(term.coefficient)).unwrap();
                for e in term.exponents {
                    write!(out, " {e}").unwrap();
                }
                writeln!(out).unwrap();
            }
        }
        return;
    }
    let mut cached = std::env::var_os("CACHED_ORACLE").map(|_| CachedOracle::new(&original));
    // Compare the optimized oracle to the original before entering the timer.
    if let Some(cached) = &mut cached {
        for seed in 1..=8 {
            let point: Vec<_> = (0..names.len())
                .map(|i| field.nth((1009 + seed * 31 + i * 17).into()))
                .collect();
            let d = original.denominator.replace_all(&point);
            let expected = (!field.is_zero(&d))
                .then(|| field.div(&original.numerator.replace_all(&point), &d));
            assert_eq!(cached.evaluate(&original, &point), expected);
        }
    }
    let setup_ms = setup.elapsed().as_secs_f64() * 1000.;
    let timeout = std::env::var("BENCH_TIMEOUT")
        .map(|s| s.parse::<f64>().unwrap())
        .unwrap_or(120.);
    let options = ReconstructionOptions {
        seed,
        max_degree: 512,
        degree_race: std::env::var_os("RECONSTRUCTION_DEGREE_RACE").is_some(),
        reuse_row_support: std::env::var_os("RECONSTRUCTION_DENSE_ROWS").is_none(),
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
                if let Some(cached) = &mut cached {
                    return cached.evaluate(&original, p);
                }
                let d = original.denominator.replace_all(p);
                (!f.is_zero(&d)).then(|| f.div(&original.numerator.replace_all(p), &d))
            },
            method,
            &options,
        )
    }));
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.;
    let mut sparse_rows = String::new();
    let mut sparse_fallbacks = String::new();
    let (status, attempts, selected, selection_probes) = match result {
        Ok(Ok((r, s))) => {
            sparse_rows = s.sparse_rows.to_string();
            sparse_fallbacks = s.sparse_row_fallbacks.to_string();
            assert_eq!(s.probes, calls.get());
            assert_eq!(
                &r.numerator * &original.denominator,
                &r.denominator * &original.numerator
            );
            (
                "ok".to_string(),
                s.attempts,
                format!("{:?}", s.selected_method.unwrap()),
                s.selection_probes.to_string(),
            )
        }
        Ok(Err(e)) => (format!("{e:?}"), 0, String::new(), String::new()),
        Err(e) if e.is::<TimeLimit>() => ("time_limit".into(), 0, String::new(), String::new()),
        Err(e) => std::panic::resume_unwind(e),
    };
    println!(
        "case,method,seed,order,status,setup_ms,elapsed_us,probes,attempts,num_terms,den_terms,selected_methods,selection_probes,sparse_rows,sparse_row_fallbacks"
    );
    println!(
        "{case},{method:?},{seed},{},{status},{setup_ms:.3},{:.3},{},{},{},{},{selected},{selection_probes},{sparse_rows},{sparse_fallbacks}",
        if reverse { "reverse" } else { "original" },
        elapsed_ms * 1000.,
        calls.get(),
        attempts,
        original.numerator.nterms(),
        original.denominator.nterms()
    );
}
