//! Compare assignment with the previous clone-and-replace arithmetic pattern.
//! Run with `cargo run -p numerica --release --example error_tracking_arithmetic`.
use std::hint::black_box;

#[path = "support/clock.rs"]
mod clock;
use clock::Clock;

use numerica::domains::float::{Complex, ErrorPropagatingFloat, Float};

type Tracked = ErrorPropagatingFloat<Float>;

fn measure(
    clock: Clock,
    name: &str,
    iterations: usize,
    mut operation: impl FnMut(&mut Tracked, &Tracked),
) {
    let mut value = ErrorPropagatingFloat::new(Float::with_val(256, 1), 65.);
    let rhs = ErrorPropagatingFloat::new(Float::with_val(256, 1.000_001), 65.);
    let start = clock.start();
    for _ in 0..iterations {
        operation(black_box(&mut value), black_box(&rhs));
    }
    let elapsed = start.elapsed();
    black_box(value);
    println!(
        "{name}: {:.1} ns/op",
        elapsed.as_nanos() as f64 / iterations as f64
    );
}

fn measure_complex(clock: Clock, name: &str, iterations: usize, imaginary: f64, divide: bool) {
    let scalar = |x| {
        if x == 0. {
            ErrorPropagatingFloat::<Float>::exact_zero(256)
        } else {
            ErrorPropagatingFloat::new(Float::with_val(256, x), 65.)
        }
    };
    let mut value = Complex::new(scalar(1.), scalar(imaginary));
    let rhs = Complex::new(scalar(1.000_001), scalar(imaginary));
    let start = clock.start();
    for _ in 0..iterations {
        if divide {
            *black_box(&mut value) /= black_box(&rhs);
        } else {
            *black_box(&mut value) *= black_box(&rhs);
        }
    }
    black_box(value);
    println!(
        "{name}: {:.1} ns/op",
        start.elapsed().as_nanos() as f64 / iterations as f64
    );
}

fn measure_plain(clock: Clock, name: &str, iterations: usize, divide: bool) {
    let mut value = Float::with_val(256, 1.);
    let rhs = Float::with_val(256, 1.000_001);
    let start = clock.start();
    for _ in 0..iterations {
        if divide {
            *black_box(&mut value) /= black_box(&rhs);
        } else {
            *black_box(&mut value) *= black_box(&rhs);
        }
    }
    black_box(value);
    println!(
        "{name}: {:.1} ns/op",
        start.elapsed().as_nanos() as f64 / iterations as f64
    );
}

fn main() {
    let clock = Clock::from_env();
    println!("Timing clock: {}", clock.label());
    println!("CPU frequency and cache contention can still affect timings");
    let iterations = std::env::args()
        .nth(1)
        .map(|n| n.parse().unwrap())
        .unwrap_or(1_000_000);
    for _ in 0..3 {
        measure_plain(clock, "plain mul", iterations, false);
        measure_plain(clock, "plain div", iterations, true);
        measure_complex(clock, "real complex mul", iterations, 0., false);
        measure_complex(clock, "real complex div", iterations, 0., true);
        measure_complex(clock, "complex mul", iterations, 0.000_001, false);
        measure_complex(clock, "complex div", iterations, 0.000_001, true);
        measure(clock, "clone/add", iterations, |a, b| *a = a.clone() + b);
        measure(clock, "in-place add", iterations, |a, b| *a += b);
        measure(clock, "clone/sub", iterations, |a, b| *a = a.clone() - b);
        measure(clock, "in-place sub", iterations, |a, b| *a -= b);
        measure(clock, "clone/mul", iterations, |a, b| *a = a.clone() * b);
        measure(clock, "in-place mul", iterations, |a, b| *a *= b);
        measure(clock, "clone/div", iterations, |a, b| *a = a.clone() / b);
        measure(clock, "in-place div", iterations, |a, b| *a /= b);
    }
}
