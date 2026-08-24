use std::{hint::black_box, sync::atomic::Ordering, time::Instant};

use symbolica::GLOBAL_SETTINGS;
use symbolica::prelude::*;

#[derive(Clone, Copy)]
struct Timings {
    construction: f64,
    multiplication: f64,
    gcd: f64,
    validation: f64,
    total: f64,
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn main() {
    let benchmark_case = std::env::var("GCD_BENCH_CASE").unwrap_or_else(|_| "dense".to_owned());
    assert!(
        matches!(
            benchmark_case.as_str(),
            "dense" | "sparse" | "high-gap" | "high-height"
        ),
        "GCD_BENCH_CASE must be dense, sparse, high-gap, or high-height"
    );
    let gap = std::env::var("GCD_BENCH_GAP")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(10);
    assert!(gap > 0, "GCD_BENCH_GAP must be positive");
    let backend = std::env::var("GCD_BENCH_BACKEND").unwrap_or_else(|_| "auto".to_owned());
    match backend.as_str() {
        "auto" => {
            GLOBAL_SETTINGS
                .use_hu_monagan_poly_gcd
                .store(true, Ordering::Relaxed);
            GLOBAL_SETTINGS
                .force_hu_monagan_poly_gcd
                .store(false, Ordering::Relaxed);
        }
        "hu" => {
            GLOBAL_SETTINGS
                .use_hu_monagan_poly_gcd
                .store(true, Ordering::Relaxed);
            GLOBAL_SETTINGS
                .force_hu_monagan_poly_gcd
                .store(true, Ordering::Relaxed);
        }
        "zippel" => {
            GLOBAL_SETTINGS
                .use_hu_monagan_poly_gcd
                .store(false, Ordering::Relaxed);
            GLOBAL_SETTINGS
                .force_hu_monagan_poly_gcd
                .store(false, Ordering::Relaxed);
        }
        _ => panic!("GCD_BENCH_BACKEND must be auto, hu, or zippel"),
    }

    let samples = std::env::var("GCD_BENCH_SAMPLES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);
    assert!(samples > 0, "GCD_BENCH_SAMPLES must be positive");

    println!("Symbolica {}", env!("CARGO_PKG_VERSION"));
    println!("case {benchmark_case}");
    if benchmark_case == "high-gap" {
        println!("gap {gap}");
    }
    println!("backend {backend}");
    if backend == "hu" && matches!(benchmark_case.as_str(), "dense" | "high-height") {
        println!("effective_backend zippel (dense Hu handoff)");
    }
    println!("samples {samples}");

    let mut timings = Vec::with_capacity(samples);
    let mut term_counts = None;
    for _ in 0..samples {
        let total_start = Instant::now();

        let phase_start = Instant::now();
        let mut polys = if benchmark_case == "dense" {
            [
                parse!("(1+3*x1+5*x2+7*x3+9*x4+11*x5+13*x6+15*x7)^7-1")
                    .to_polynomial::<_, u16>(&Z, None),
                parse!("(1-3*x1-5*x2-7*x3+9*x4-11*x5-13*x6+15*x7)^7+1")
                    .to_polynomial::<_, u16>(&Z, None),
                parse!("(1+3*x1+5*x2+7*x3+9*x4+11*x5+13*x6-15*x7)^7+3")
                    .to_polynomial::<_, u16>(&Z, None),
            ]
        } else if benchmark_case == "high-height" {
            [
                parse!("(1+1000000007*x1+1000000009*x2+1000000033*x3+1000000087*x4+1000000093*x5+1000000097*x6+1000000103*x7)^7-1")
                    .to_polynomial::<_, u16>(&Z, None),
                parse!("(1-1000000007*x1-1000000009*x2-1000000033*x3+1000000087*x4-1000000093*x5-1000000097*x6+1000000103*x7)^7+1")
                    .to_polynomial::<_, u16>(&Z, None),
                parse!("(1+1000000007*x1+1000000009*x2+1000000033*x3+1000000087*x4+1000000093*x5+1000000097*x6-1000000103*x7)^7+3")
                    .to_polynomial::<_, u16>(&Z, None),
            ]
        } else if benchmark_case == "sparse" {
            [
                parse!("(1+3*x1+5*x2+7*x3+9*x4+11*x5+13*x6+15*x7)^7-1")
                    .to_polynomial::<_, u16>(&Z, None),
                parse!("(1-3*x1-5*x2-7*x3+9*x4-11*x5-13*x6+15*x7)^7+1")
                    .to_polynomial::<_, u16>(&Z, None),
                parse!("1+x1^7+2*x2^7+3*x3^7+5*x4^7+7*x5^7+11*x6^7+13*x7^7")
                    .to_polynomial::<_, u16>(&Z, None),
            ]
        } else {
            let g_expression = format!(
                "1+x1^{gap}+2*x2^{gap}+3*x3^{gap}+5*x4^{gap}+7*x5^{gap}+11*x6^{gap}+13*x7^{gap}"
            );
            [
                parse!("(1+3*x1+5*x2+7*x3+9*x4+11*x5+13*x6+15*x7)^7-1")
                    .to_polynomial::<_, u16>(&Z, None),
                parse!("(1-3*x1-5*x2-7*x3+9*x4-11*x5-13*x6+15*x7)^7+1")
                    .to_polynomial::<_, u16>(&Z, None),
                Atom::parse(&g_expression, "gcd_benchmark", ParseSettings::default())
                    .unwrap()
                    .to_polynomial::<_, u16>(&Z, None),
            ]
        };
        MultivariatePolynomial::unify_variables_list(&mut polys);
        let construction = phase_start.elapsed().as_secs_f64();
        let [a, b, g] = polys;

        let phase_start = Instant::now();
        let ag = black_box(&a) * black_box(&g);
        let bg = black_box(&b) * black_box(&g);
        let multiplication = phase_start.elapsed().as_secs_f64();

        let phase_start = Instant::now();
        let computed_gcd = black_box(&ag).gcd(black_box(&bg));
        let gcd = phase_start.elapsed().as_secs_f64();

        let phase_start = Instant::now();
        let difference = &computed_gcd - &g;
        assert!(difference.is_zero(), "computed GCD differs from g");
        black_box(difference);
        let validation = phase_start.elapsed().as_secs_f64();

        let total = total_start.elapsed().as_secs_f64();
        term_counts = Some((
            a.nterms(),
            b.nterms(),
            g.nterms(),
            ag.nterms(),
            bg.nterms(),
            computed_gcd.nterms(),
        ));
        timings.push(Timings {
            construction,
            multiplication,
            gcd,
            validation,
            total,
        });
    }

    let summarize =
        |field: fn(Timings) -> f64| median(timings.iter().copied().map(field).collect());
    println!(
        "terms a/b/g/ag/bg/gcd {}/{}/{}/{}/{}/{}",
        term_counts.unwrap().0,
        term_counts.unwrap().1,
        term_counts.unwrap().2,
        term_counts.unwrap().3,
        term_counts.unwrap().4,
        term_counts.unwrap().5,
    );
    println!(
        "construction   {:10.3} ms",
        summarize(|t| t.construction) * 1_000.0
    );
    println!(
        "multiplication {:10.3} ms",
        summarize(|t| t.multiplication) * 1_000.0
    );
    println!("gcd            {:10.3} ms", summarize(|t| t.gcd) * 1_000.0);
    println!(
        "validation     {:10.3} ms",
        summarize(|t| t.validation) * 1_000.0
    );
    println!(
        "total          {:10.3} ms",
        summarize(|t| t.total) * 1_000.0
    );
}
