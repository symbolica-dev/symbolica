//! Shared, alias-independent inputs and timing for paired expansion benchmarks.
//!
//! Copy this file unchanged with expansion_performance.rs to the main checkout.
//! The alias executable changes only how the designated polynomial bodies are
//! stored; it uses the same expression construction and numerical reference.

use std::{
    collections::HashMap,
    hint::black_box,
    time::{Duration, Instant},
};

use symbolica::{
    atom::{Atom, AtomCore},
    parse,
};

struct Benchmark {
    samples: usize,
    target: Duration,
    filter: String,
}

impl Benchmark {
    fn from_args() -> Self {
        let mut samples = 7usize;
        let mut target_ms = 100u64;
        let mut filter = String::new();
        let mut arguments = std::env::args().skip(1);
        while let Some(argument) = arguments.next() {
            match argument.as_str() {
                "--samples" => {
                    samples = arguments
                        .next()
                        .expect("missing sample count")
                        .parse()
                        .unwrap()
                }
                "--target-ms" => {
                    target_ms = arguments.next().expect("missing duration").parse().unwrap()
                }
                "--filter" => filter = arguments.next().expect("missing workload filter"),
                "--help" | "-h" => {
                    eprintln!("[--samples 7] [--target-ms 100] [--filter substring]");
                    std::process::exit(0);
                }
                _ => panic!("unknown argument: {argument}"),
            }
        }
        assert!(samples > 0 && target_ms > 0);
        Self {
            samples,
            target: Duration::from_millis(target_ms),
            filter,
        }
    }

    fn measure(&self, name: &str, input: &Atom, result_bytes: usize) {
        let mut operation = || black_box(input).expand();
        let mut iterations = 1usize;
        let calibration = loop {
            let duration = elapsed(iterations, &mut operation);
            if duration >= Duration::from_millis(10) {
                break duration;
            }
            iterations = iterations.checked_mul(2).unwrap();
        };
        let iterations = ((iterations as f64 * self.target.as_secs_f64()
            / calibration.as_secs_f64())
        .round() as usize)
            .max(1);
        for sample in 0..self.samples {
            let duration = elapsed(iterations, &mut operation);
            println!(
                "{name},{sample},{iterations},{},{:.3},{result_bytes}",
                duration.as_nanos(),
                duration.as_nanos() as f64 / iterations as f64
            );
        }
    }
}

fn elapsed(iterations: usize, operation: &mut impl FnMut() -> Atom) -> Duration {
    let start = Instant::now();
    for _ in 0..iterations {
        // Destruction of the newly allocated output is part of every operation.
        let _ = black_box(operation());
    }
    start.elapsed()
}

struct Workload {
    name: &'static str,
    plain: Atom,
    input: Atom,
    numerical_reference: fn([f64; 4]) -> f64,
}

fn large_polynomial_value([x, y, _, _]: [f64; 4]) -> f64 {
    (0..12)
        .map(|i| (i + 1) as f64 * x.powi(i % 4) * y.powi(i / 4))
        .sum()
}

fn workloads(mut represent: impl FnMut(&Atom) -> Atom) -> Vec<Workload> {
    let small = parse!("1+paired_exp_x+paired_exp_y+paired_exp_z+paired_exp_w");
    let a = parse!("1+paired_exp_x+paired_exp_y");
    let b = parse!("2+paired_exp_x+paired_exp_z");
    let c = parse!("3+paired_exp_y+paired_exp_w");
    let large = parse!(
        &(0..12)
            .map(|i| format!("{}*paired_exp_x^{}*paired_exp_y^{}", i + 1, i % 4, i / 4,))
            .collect::<Vec<_>>()
            .join("+")
    );
    let z = parse!("paired_exp_z");
    let w = parse!("paired_exp_w");
    let small_input = represent(&small);
    let a_input = represent(&a);
    let b_input = represent(&b);
    let c_input = represent(&c);
    let large_input = represent(&large);
    let product = |a: &Atom, b: &Atom, c: &Atom| a.pow(3) * b.pow(3) * c.pow(2);
    // The same large body appears in two different sum factors. Normalization
    // cannot merge those two references into a single power before expansion.
    let repeated = |body: &Atom| (body.pow(2) + &z) * (body.pow(2) + &w);
    vec![
        Workload {
            name: "expand_power_small",
            plain: small.pow(7),
            input: small_input.pow(7),
            numerical_reference: |[x, y, z, w]| (1. + x + y + z + w).powi(7),
        },
        Workload {
            name: "expand_product_small",
            plain: product(&a, &b, &c),
            input: product(&a_input, &b_input, &c_input),
            numerical_reference: |[x, y, z, w]| {
                (1. + x + y).powi(3) * (2. + x + z).powi(3) * (3. + y + w).powi(2)
            },
        },
        Workload {
            name: "expand_power_large",
            plain: large.pow(4),
            input: large_input.pow(4),
            numerical_reference: |point| large_polynomial_value(point).powi(4),
        },
        Workload {
            name: "expand_product_large_repeated",
            plain: repeated(&large),
            input: repeated(&large_input),
            numerical_reference: |point| {
                let square = large_polynomial_value(point).powi(2);
                (square + point[2]) * (square + point[3])
            },
        },
    ]
}

fn validate_numerically(workload: &Workload, result: &Atom) {
    let variables = [
        parse!("paired_exp_x"),
        parse!("paired_exp_y"),
        parse!("paired_exp_z"),
        parse!("paired_exp_w"),
    ];
    for point in [
        [0.25, -0.5, 0.75, -0.125],
        [-0.375, 0.125, -0.25, 0.5],
        [0.5, 0.25, 0.125, -0.25],
    ] {
        let values: HashMap<Atom, f64> = variables.iter().cloned().zip(point).collect();
        let expected = (workload.numerical_reference)(point);
        for expression in [&workload.plain, &workload.input, result] {
            let actual = expression.evaluate(&values).unwrap();
            assert!(
                actual.is_finite() && (actual - expected).abs() <= 1e-10 * expected.abs().max(1.),
                "{}: numerical mismatch at {point:?}: {actual} != {expected}",
                workload.name
            );
        }
    }
}

/// Representation setup and validation callbacks are never called in the timed loop.
pub(crate) fn run(
    represent: impl FnMut(&Atom) -> Atom,
    fully_inline: impl Fn(&Atom) -> Atom,
    check_input: impl Fn(&Atom, &Atom),
) {
    let benchmark = Benchmark::from_args();
    println!("workload,sample,iterations,total_ns,ns_per_iteration,result_bytes");
    for workload in workloads(represent) {
        if !workload.name.contains(&benchmark.filter) {
            continue;
        }
        check_input(&workload.input, &workload.plain);
        assert_eq!(
            workload.input, workload.plain,
            "{}: input semantics",
            workload.name
        );
        let inlined_input = fully_inline(&workload.input);
        assert_eq!(
            inlined_input.as_view().get_data(),
            workload.plain.as_view().get_data(),
            "{}: input encoding after inlining",
            workload.name
        );
        let expected = workload.plain.expand();
        let result = workload.input.expand();
        assert_eq!(
            result, expected,
            "{}: expanded result semantics",
            workload.name
        );
        let inlined_result = fully_inline(&result);
        // A no-op returning an aliased power still differs from this distributed
        // polynomial after alias inlining; it must not produce a false speedup.
        assert_eq!(
            inlined_result.as_view().get_data(),
            expected.as_view().get_data(),
            "{}: fully expanded result bytes",
            workload.name
        );
        validate_numerically(&workload, &result);
        let result_bytes = result.as_view().get_byte_size();
        eprintln!(
            "{}: input_bytes={} fully_inlined_input_bytes={} result_bytes={} fully_inlined_result_bytes={}",
            workload.name,
            workload.input.as_view().get_byte_size(),
            inlined_input.as_view().get_byte_size(),
            result_bytes,
            inlined_result.as_view().get_byte_size()
        );
        drop(result);
        drop(inlined_result);
        drop(inlined_input);
        drop(expected);
        benchmark.measure(workload.name, &workload.input, result_bytes);
    }
}
