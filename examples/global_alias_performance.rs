//! Regression benchmarks with live global aliases.
//!
//! Build with `cargo build --release --example global_alias_performance`, then run
//! the binary directly and sequentially for each checkout:
//!
//! ```text
//! SYMBOLICA_HIDE_BANNER=1 target/release/examples/global_alias_performance \
//!     --samples 7 --target-ms 100 > global-alias-timings.csv
//! ```
//!
//! The CSV format matches `alias_performance`. Setup, semantic checks using
//! `inline_aliases`, and calibration are outside the measured samples. The source
//! uses the global-alias API present at commit 3dab8484.

use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use symbolica::{
    atom::{Atom, AtomCore},
    parse, symbol,
};

fn elapsed<F: FnMut() -> Atom>(iterations: usize, operation: &mut F) -> Duration {
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = black_box(operation());
    }
    start.elapsed()
}

fn measure<F: FnMut() -> Atom>(
    name: &str,
    expected: &Atom,
    samples: usize,
    target: Duration,
    filter: &str,
    mut operation: F,
) {
    if !name.contains(filter) {
        return;
    }

    let result = black_box(operation());
    assert_eq!(result.inline_aliases(true), *expected, "{name}");
    let result_bytes = result.as_view().get_byte_size();
    drop(result);

    let mut iterations = 1usize;
    let calibration = loop {
        let duration = elapsed(iterations, &mut operation);
        if duration >= Duration::from_millis(10) {
            break duration;
        }
        iterations = iterations.checked_mul(2).unwrap();
    };
    let iterations =
        ((iterations as f64 * target.as_secs_f64() / calibration.as_secs_f64()).round() as usize)
            .max(1);

    for sample in 0..samples {
        let duration = elapsed(iterations, &mut operation);
        println!(
            "{name},{sample},{iterations},{},{:.3},{result_bytes}",
            duration.as_nanos(),
            duration.as_nanos() as f64 / iterations as f64,
        );
    }
}

fn main() {
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
                    .unwrap();
            }
            "--target-ms" => {
                target_ms = arguments
                    .next()
                    .expect("missing sample duration")
                    .parse()
                    .unwrap();
            }
            "--filter" => filter = arguments.next().expect("missing workload filter"),
            "--help" | "-h" => {
                eprintln!(
                    "global_alias_performance [--samples 7] [--target-ms 100] [--filter substring]"
                );
                return;
            }
            _ => panic!("unknown argument: {argument}"),
        }
    }
    assert!(samples > 0 && target_ms > 0);
    let target = Duration::from_millis(target_ms);

    let (x, y, z) = symbol!("x", "y", "z");
    let (x, y, z) = (Atom::var(x), Atom::var(y), Atom::var(z));
    let construct_expected = parse!("(x+y+1)^2*(z+2)+x*y");
    let unrelated = parse!("unrelated(u+v+w)").alias(false);
    assert!(!unrelated.get_aliases().is_empty());

    println!("workload,sample,iterations,total_ns,ns_per_iteration,result_bytes");

    // Run before registering any related aliases. This is the same expression as
    // the `construct` workload in alias_performance, with an unrelated alias live.
    measure(
        "construct_with_live_alias",
        &construct_expected,
        samples,
        target,
        &filter,
        || {
            (black_box(&x) + black_box(&y) + 1).pow(2) * (black_box(&z) + 2)
                + black_box(&x) * black_box(&y)
        },
    );
    black_box(&unrelated);

    let base = parse!("f(x+y+z)");
    let base_alias = base.alias(false);
    let aliased_power = base_alias.pow(7);
    let plain_power = base.pow(7);
    assert!(!aliased_power.get_aliases().is_empty());
    measure(
        "copy_alias_power",
        &plain_power,
        samples,
        target,
        &filter,
        || black_box(aliased_power.as_view()).to_owned(),
    );

    let plain_term = parse!("x*y*f(x+y)^3");
    let aliased_term = plain_term.alias(false);
    let sum_expected = &plain_term * 2;
    measure(
        "add_alias_plain",
        &sum_expected,
        samples,
        target,
        &filter,
        || black_box(&aliased_term) + black_box(&plain_term),
    );

    let left_power = base_alias.pow(3);
    let right_power = base_alias.pow(4);
    assert!(!left_power.get_aliases().is_empty() && !right_power.get_aliases().is_empty());
    measure(
        "multiply_alias_powers",
        &plain_power,
        samples,
        target,
        &filter,
        || black_box(&left_power) * black_box(&right_power),
    );
}
