//! Compare scalar magnitude arithmetic with native binary64 arithmetic.
//!
//! Run with `cargo run -p numerica --release --example magnitude_arithmetic`.
//! Inputs stay near one so that both types operate without overflow/underflow.
//! Magnitude includes conservative rounding and a wider exponent range; these
//! timings measure the cost of those features, not equivalent binary64 semantics.

use std::hint::black_box;

#[path = "support/clock.rs"]
mod clock;
use clock::Clock;

use numerica::domains::float::Magnitude;

fn median_ns<T: Copy>(clock: Clock, values: &[T; 256], operation: impl Fn(T, T) -> T) -> f64 {
    const ITERATIONS: usize = 2_000_000;
    let mut timings = [0.; 5];
    for time in &mut timings {
        let values = black_box(values);
        let start = clock.start();
        for index in 0..ITERATIONS {
            let left = black_box(values[index & 255]);
            let right = black_box(values[(index + 73) & 255]);
            black_box(operation(left, right));
        }
        *time = start.elapsed().as_nanos() as f64 / ITERATIONS as f64;
    }
    timings.sort_by(f64::total_cmp);
    timings[2]
}

fn main() {
    let clock = Clock::from_env();
    println!("Timing clock: {}", clock.label());
    println!("CPU frequency and cache contention can still affect timings");
    let native = std::array::from_fn(|index| 1. + index as f64 / 256.);
    let magnitudes = native.map(Magnitude::from_f64);
    let results = [
        (
            "add",
            median_ns(clock, &native, |a, b| a + b),
            median_ns(clock, &magnitudes, |a, b| a + b),
        ),
        (
            "multiply",
            median_ns(clock, &native, |a, b| a * b),
            median_ns(clock, &magnitudes, |a, b| a * b),
        ),
        (
            "divide",
            median_ns(clock, &native, |a, b| a / b),
            median_ns(clock, &magnitudes, |a, b| a / b),
        ),
    ];
    println!("Scalar arithmetic; median of five runs, 2,000,000 operations each");
    println!("operation, f64 ns/op, magnitude ns/op, ratio");
    for (operation, native, magnitude) in results {
        println!(
            "{operation}, {native:.2}, {magnitude:.2}, {:.2}x",
            magnitude / native
        );
    }
}
