//! Compare borrowed binary ordering with decoding both operands first.
//! Run with `cargo run -p numerica --release --example binary_float_comparison -- 20000`.

use std::hint::black_box;

#[path = "support/clock.rs"]
mod clock;
use clock::Clock;

use numerica::domains::float::{Float, FloatLike};

fn measure(
    clock: Clock,
    iterations: usize,
    pairs: &[(u32, Vec<u8>, Vec<u8>)],
    decode: bool,
) -> f64 {
    let mut samples = [0.; 3];
    for sample in &mut samples {
        let start = clock.start();
        for _ in 0..iterations {
            for (precision, left, right) in pairs {
                let order = if decode {
                    Float::deserialize(black_box(left), *precision)
                        .partial_cmp(&Float::deserialize(black_box(right), *precision))
                } else {
                    Float::serialized_partial_cmp(black_box(left), black_box(right))
                };
                black_box(order);
            }
        }
        *sample = start.elapsed().as_secs_f64() * 1e9 / (iterations * pairs.len()) as f64;
    }
    samples.sort_by(f64::total_cmp);
    samples[1]
}

fn main() {
    let iterations = std::env::args()
        .nth(1)
        .map(|value| {
            value
                .parse::<usize>()
                .expect("iterations must be an integer")
        })
        .unwrap_or(20_000);
    assert!(iterations > 0);
    let clock = Clock::from_env();
    println!("# Clock: {}", clock.label());
    println!(
        "median of 3 samples; {iterations} iterations of 8 comparisons; nanoseconds/comparison"
    );
    for precision in [53, 256, 4096] {
        let third = Float::with_val(precision, 1) / Float::with_val(precision, 3);
        let seventh = Float::with_val(precision, 1) / Float::with_val(precision, 7);
        let delta = Float::with_val(precision, 2)
            .pow(u64::from(precision - 4))
            .inv();
        let adjacent = third.clone() + delta;
        let raw_pairs = [
            (third.clone(), seventh.clone()),
            (third.clone(), adjacent),
            (third.clone(), third.clone()),
            (third.clone(), -third.clone()),
            (-third.clone(), -seventh),
            (Float::with_val(precision, 0), third.clone()),
            (Float::with_val(precision, f64::INFINITY), third.clone()),
            (Float::with_val(precision, f64::NAN), third),
        ];
        let pairs: Vec<_> = raw_pairs
            .iter()
            .map(|(left, right)| (precision, left.serialize(), right.serialize()))
            .collect();
        for ((_, left, right), (left_value, right_value)) in pairs.iter().zip(&raw_pairs) {
            assert_eq!(
                Float::serialized_partial_cmp(left, right),
                left_value.partial_cmp(right_value)
            );
        }
        let borrowed = measure(clock, iterations, &pairs, false);
        let decoded = measure(clock, iterations, &pairs, true);
        println!(
            "{precision:4} bits: borrowed={borrowed:9.2} decode+compare={decoded:9.2} speedup={:.2}x",
            decoded / borrowed
        );
    }
}
