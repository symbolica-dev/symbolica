//! Measure packed floating coefficients and their public access paths.
//!
//! Run with `cargo run --release --example real_float_coefficients -- 10000 current`.
//! On Linux, pin comparative runs to the same CPU with `taskset -c 2`.
//! Timings are medians of three runs using the shared benchmark clock. Packed
//! bytes exclude allocator capacity, Atom metadata, and separately stored symbols.
//! Construction/repacking include cloning the supplied owned coefficient; setup
//! of the multiprecision inputs is outside the measured loops.
//! A separate retention measurement is available as
//! `real_float_coefficients memory 500000 current real_same_precision 53`.
//! RSS includes allocator size classes and caching; it is not an exact heap census.

use std::hint::black_box;

use symbolica::{
    atom::{Atom, AtomView, representation::Num},
    coefficient::{Coefficient, CoefficientView},
    domains::float::{Complex, ErrorPropagatingFloat, Float, Magnitude},
};

#[path = "../lib/numerica/examples/support/clock.rs"]
mod clock;
use clock::Clock;

type Tracked = ErrorPropagatingFloat<Float>;

fn median_ns(clock: Clock, iterations: usize, mut operation: impl FnMut()) -> f64 {
    for _ in 0..16 {
        operation();
    }
    let mut timings = [0.; 3];
    for elapsed in &mut timings {
        let start = clock.start();
        for _ in 0..iterations {
            operation();
        }
        *elapsed = start.elapsed().as_nanos() as f64 / iterations as f64;
    }
    timings.sort_by(f64::total_cmp);
    timings[1]
}

fn coefficient_view(atom: &Atom) -> CoefficientView<'_> {
    let AtomView::Num(number) = atom.as_view() else {
        panic!("expected a floating coefficient");
    };
    number.get_coeff_view()
}

fn coefficient(precision: u32, case: &str) -> (Coefficient, u32) {
    // Nonterminating binary expansions exercise the requested precision, rather
    // than merely increasing the storage precision of a short f64 mantissa.
    let re = Tracked::from(Float::with_val(precision, 1) / Float::with_val(precision, 3));
    let imaginary_precision = if case == "real_different_precision" {
        precision + 37
    } else {
        precision
    };
    let im = match case {
        "real_same_precision" | "real_different_precision" => {
            Tracked::exact_zero(imaginary_precision)
        }
        "uncertain_imaginary_zero" => Tracked::new_with_error_magnitude(
            Float::new(imaginary_precision),
            Magnitude::ONE.scale_pow2(-(imaginary_precision as i64)),
        ),
        "complex" => Tracked::from(
            Float::with_val(imaginary_precision, 2) / Float::with_val(imaginary_precision, 7),
        ),
        _ => unreachable!(),
    };
    (Coefficient::from(Complex::new(re, im)), imaginary_precision)
}

fn validate(atom: &Atom, source: &Coefficient, case: &str) {
    let view = coefficient_view(atom);
    assert_eq!(view.is_real(), case.starts_with("real_"));
    let Coefficient::Float(decoded) = view.to_owned() else {
        panic!("floating coefficient changed type");
    };
    let Coefficient::Float(source) = source else {
        unreachable!();
    };
    for (original, copied) in [(&source.re, &decoded.re), (&source.im, &decoded.im)] {
        assert_eq!(original.get_num(), copied.get_num());
        assert_eq!(original.get_num().prec(), copied.get_num().prec());
        assert_eq!(original.get_error_magnitude(), copied.get_error_magnitude());
    }
}

fn resident_bytes() -> Option<usize> {
    // Linux exposes RSS in KiB here, avoiding assumptions about the page size.
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        line.strip_prefix("VmRSS:")?
            .split_whitespace()
            .next()?
            .parse::<usize>()
            .ok()?
            .checked_mul(1024)
    })
}

fn memory(mut arguments: impl Iterator<Item = String>) {
    let count = arguments
        .next()
        .map(|value| value.parse::<usize>().expect("invalid retained atom count"))
        .unwrap_or(500_000);
    assert!(count > 0, "retained atom count must be positive");
    let revision = arguments.next().unwrap_or_else(|| "current".into());
    let case = arguments
        .next()
        .unwrap_or_else(|| "real_same_precision".into());
    let precision = arguments
        .next()
        .map(|value| value.parse::<u32>().expect("invalid binary precision"))
        .unwrap_or(53);
    assert!(precision > 0, "binary precision must be positive");
    let (source, imaginary_precision) = coefficient(precision, &case);
    let sample = Atom::num(source.clone());
    validate(&sample, &source, &case);
    for _ in 0..1024 {
        drop(black_box(Atom::num(black_box(&source).clone())));
    }
    let before = resident_bytes();
    let mut atoms = Vec::with_capacity(count);
    for _ in 0..count {
        atoms.push(Atom::num(black_box(&source).clone()));
    }
    let packed_bytes: usize = atoms
        .iter()
        .map(|atom| atom.as_view().get_byte_size())
        .sum();
    black_box(&atoms);
    let after = resident_bytes();
    let delta = match (before, after) {
        (Some(before), Some(after)) => (after as i128 - before as i128).to_string(),
        _ => "unavailable".into(),
    };
    let display = |bytes: Option<usize>| {
        bytes.map_or_else(|| "unavailable".into(), |bytes| bytes.to_string())
    };
    let atom_metadata_bytes = std::mem::size_of::<Atom>() * count;
    println!("# Linux RSS delta after warmup, including the retained Vec and Atom allocations");
    println!(
        "# RSS has page granularity and includes allocator caching; packed bytes exclude capacity"
    );
    println!("# Precision columns are requested bits; backend storage precision may be rounded up");
    println!(
        "revision,precision,imaginary_precision,case,count,packed_bytes,atom_metadata_bytes,rss_before_bytes,rss_after_bytes,rss_delta_bytes"
    );
    println!(
        "{revision},{precision},{imaginary_precision},{case},{count},{packed_bytes},{atom_metadata_bytes},{},{},{delta}",
        display(before),
        display(after),
    );
    drop(black_box(atoms));
}

fn main() {
    let mut arguments = std::env::args().skip(1);
    let first = arguments.next();
    if first.as_deref() == Some("memory") {
        memory(arguments);
        return;
    }
    let iterations = first
        .map(|value| value.parse::<usize>().expect("invalid iteration count"))
        .unwrap_or(10_000);
    assert!(iterations > 0, "iteration count must be positive");
    let revision = arguments.next().unwrap_or_else(|| "current".into());
    let clock = Clock::from_env();
    println!("# Timing clock: {}", clock.label());
    println!("# CPU frequency and cache contention can still affect timings");
    println!("# Packed bytes exclude allocation capacity and Atom metadata; timings include drops");
    println!("# Construction and repacking include cloning the input coefficient");
    println!("# Precision columns are requested bits; backend storage precision may be rounded up");
    println!(
        "revision,precision,imaginary_precision,case,packed_bytes,operation,iterations,median_ns"
    );

    for precision in [53, 256, 4096] {
        for case in [
            "real_same_precision",
            "real_different_precision",
            "uncertain_imaginary_zero",
            "complex",
        ] {
            let (source, imaginary_precision) = coefficient(precision, case);
            let atom = Atom::num(source.clone());
            validate(&atom, &source, case);
            let packed_bytes = atom.as_view().get_byte_size();
            let report = |operation: &str, ns: f64| {
                println!(
                    "{revision},{precision},{imaginary_precision},{case},{packed_bytes},{operation},{iterations},{ns:.2}"
                );
            };

            report(
                "construct_atom_clone",
                median_ns(clock, iterations, || {
                    drop(black_box(Atom::num(black_box(&source).clone())));
                }),
            );
            let mut reusable = Num::new(source.clone());
            report(
                "repack_reused_num_clone",
                median_ns(clock, iterations, || {
                    reusable.set_from_coeff(black_box(&source).clone());
                    black_box(&reusable);
                }),
            );
            report(
                "borrow_coefficient",
                median_ns(clock, iterations, || {
                    black_box(coefficient_view(black_box(&atom)));
                }),
            );
            report(
                "borrowed_predicates",
                median_ns(clock, iterations, || {
                    let view = coefficient_view(black_box(&atom));
                    black_box((view.is_real(), view.is_zero(), view.is_one()));
                }),
            );
            report(
                "decode_imaginary",
                median_ns(clock, iterations, || {
                    let CoefficientView::Float(_, im) = coefficient_view(black_box(&atom)) else {
                        unreachable!();
                    };
                    black_box(im.to_float());
                }),
            );
            report(
                "decode_coefficient",
                median_ns(clock, iterations, || {
                    black_box(coefficient_view(black_box(&atom)).to_owned());
                }),
            );
            report(
                "copy_packed_atom",
                median_ns(clock, iterations, || {
                    drop(black_box(black_box(&atom).as_view().to_owned()));
                }),
            );
            report(
                "add_atoms",
                median_ns(clock, iterations, || {
                    drop(black_box(black_box(&atom) + black_box(&atom)));
                }),
            );
            report(
                "multiply_atoms",
                median_ns(clock, iterations, || {
                    drop(black_box(black_box(&atom) * black_box(&atom)));
                }),
            );
        }
    }
}
