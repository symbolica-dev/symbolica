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
    atom::{Atom, AtomCore, AtomView, FunctionBuilder},
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

    // A compressed power should not repeatedly traverse or hash its large body.
    let large_body = parse!(&format!(
        "large_alias_body({})",
        (0..512)
            .map(|i| format!("large_x{i}*large_y{i}"))
            .collect::<Vec<_>>()
            .join("+")
    ));
    let large_alias = large_body.alias(false);
    let large_power = large_alias.pow(7);
    let large_expected = large_body.pow(7);
    measure(
        "copy_large_alias_power",
        &large_expected,
        samples,
        target,
        &filter,
        || black_box(large_power.as_view()).to_owned(),
    );

    let large_left = large_alias.pow(3);
    let large_right = large_alias.pow(4);
    measure(
        "multiply_large_alias_powers",
        &large_expected,
        samples,
        target,
        &filter,
        || black_box(&large_left) * black_box(&large_right),
    );

    // Repeated aliases exercise deduplication; a child view carries the parent's
    // whole ownership set, so copying it must select only the handles it uses.
    let child_name = symbol!("alias_copy_child");
    let parent_name = symbol!("alias_copy_parent");
    let children: Vec<_> = (0..64)
        .map(|i| {
            let alias = parse!(&format!("alias_copy_body(x+{i})")).alias(i % 2 == 0);
            FunctionBuilder::new(child_name)
                .add_args(std::iter::repeat_n(&alias, 8))
                .finish()
        })
        .collect();
    let multi_owner = FunctionBuilder::new(parent_name)
        .add_args(&children)
        .finish();
    assert_eq!(multi_owner.get_aliases().len(), 64);
    let multi_owner_expected = multi_owner.inline_aliases(true);
    measure(
        "copy_repeated_multi_aliases",
        &multi_owner_expected,
        samples,
        target,
        &filter,
        || black_box(multi_owner.as_view()).to_owned(),
    );

    let AtomView::Fun(parent) = multi_owner.as_view() else {
        unreachable!()
    };
    let child = parent.iter().next().unwrap();
    let child_expected = children[0].inline_aliases(true);
    measure(
        "copy_alias_child_from_multi_owner",
        &child_expected,
        samples,
        target,
        &filter,
        || black_box(child).to_owned(),
    );
}
