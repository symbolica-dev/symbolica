//! Compare production Float binary I/O across revisions with the same harness.
//!
//! Copy this file unchanged into lib/numerica/examples/float_binary_io.rs in each
//! checkout; it uses only the public Float API and the existing support clock.
//! Build either with default MPFR features or with
//! --no-default-features --features integer-malachite,float-astro.
//! Run: NUMERICA_BENCH_CLOCK=thread taskset -c 2 <saved-binary> 100000 <revision>.
//!
//! Each record is the median of three warmed samples. The allocated writer and
//! decoder include drops; the reused writer retains its buffer. Values and
//! validation are prepared outside timing. Payload sizes exclude precision,
//! error magnitudes and enclosing coefficient metadata. Precision columns report
//! both requested and actual precision, which can differ across backends.

use numerica::domains::float::{Float, FloatLike};
use std::hint::black_box;

#[path = "support/clock.rs"]
mod clock;
use clock::Clock;

fn backend() -> &'static str {
    #[cfg(feature = "float-mpfr")]
    {
        "mpfr"
    }
    #[cfg(all(not(feature = "float-mpfr"), feature = "float-astro"))]
    {
        "astro"
    }
}

fn median_ns(clock: Clock, iterations: usize, mut operation: impl FnMut()) -> f64 {
    for _ in 0..32 {
        operation();
    }
    let mut samples = [0.; 3];
    for sample in &mut samples {
        let timer = clock.start();
        for _ in 0..iterations {
            operation();
        }
        *sample = timer.elapsed().as_nanos() as f64 / iterations as f64;
    }
    samples.sort_by(f64::total_cmp);
    samples[1]
}

fn validate(value: &Float) -> Vec<u8> {
    let bytes = value.serialize();
    assert_eq!(bytes.len(), value.serialized_size());
    let decoded = Float::try_deserialize(&bytes, value.prec()).expect("binary roundtrip failed");
    assert_eq!(decoded, *value);
    assert_eq!(decoded.prec(), value.prec());
    assert_eq!(decoded.serialize(), bytes);

    // serialize_into appends to an existing buffer, including a nonempty one.
    let mut appended = vec![0x5a, 0xc3];
    value.serialize_into(&mut appended);
    assert_eq!(&appended[..2], &[0x5a, 0xc3]);
    assert_eq!(&appended[2..], bytes.as_slice());
    bytes
}

fn main() {
    let mut arguments = std::env::args().skip(1);
    let iterations = arguments
        .next()
        .map(|value| value.parse::<usize>().expect("invalid iteration count"))
        .unwrap_or(100_000);
    assert!(iterations > 0, "iteration count must be positive");
    let revision = arguments.next().unwrap_or_else(|| "current".into());
    assert!(
        !revision.chars().any(|c| matches!(c, ',' | '\n' | '\r')),
        "revision must be a single CSV field"
    );
    assert!(arguments.next().is_none(), "unexpected extra argument");
    let clock = Clock::from_env();
    println!("# Timing clock: {}", clock.label());
    println!("# Three warmed samples per record; process repetitions are external");
    println!("# CPU frequency/cache variation remains; payload excludes precision/error metadata");
    println!("# dense_third = 1/3; power_of_two = 2^4000; setup and validation are untimed");
    println!(
        "backend,revision,requested_precision,actual_precision,case,payload_bytes,operation,iterations,median_ns"
    );

    for requested_precision in [53, 256, 4096] {
        for case in ["dense_third", "power_of_two"] {
            let value = match case {
                "dense_third" => {
                    Float::with_val(requested_precision, 1)
                        / Float::with_val(requested_precision, 3)
                }
                "power_of_two" => Float::with_val(requested_precision, 2).pow(4000),
                _ => unreachable!(),
            };
            let bytes = validate(&value);
            let actual_precision = value.prec();
            let report = |operation: &str, ns: f64| {
                println!(
                    "{},{revision},{requested_precision},{actual_precision},{case},{},{operation},{iterations},{ns:.2}",
                    backend(),
                    bytes.len(),
                );
            };
            report(
                "encode_allocated",
                median_ns(clock, iterations, || {
                    drop(black_box(black_box(&value).serialize()));
                }),
            );
            let mut reusable = Vec::with_capacity(value.serialized_size());
            report(
                "encode_reused",
                median_ns(clock, iterations, || {
                    reusable.clear();
                    black_box(&value).serialize_into(&mut reusable);
                    black_box(&reusable);
                }),
            );
            report(
                "decode",
                median_ns(clock, iterations, || {
                    drop(black_box(
                        Float::try_deserialize(black_box(&bytes), actual_precision)
                            .expect("validated payload should decode"),
                    ));
                }),
            );
        }
    }
}
