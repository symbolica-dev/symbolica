//! Production binary float payload versus the former hexadecimal payload.
//!
//! Run with `cargo run --release -p numerica --example binary_float_serialization -- 100000`.
//! Precision, tracked error, and enclosing payload length remain separate
//! metadata in both cases. The binary path uses the production checked parser
//! and allocation-free writer; the hexadecimal baseline reproduces the old
//! encoding. Full coefficient benchmarks live in examples/real_float_coefficients.rs.

#[cfg(all(feature = "float-mpfr", feature = "integer-gmp"))]
#[path = "support/clock.rs"]
mod clock;

#[cfg(all(feature = "float-mpfr", feature = "integer-gmp"))]
mod benchmark {
    use super::clock::Clock;
    use numerica::domains::float::Float;
    use std::hint::black_box;

    fn encode(value: &Float, out: &mut Vec<u8>) {
        value.serialize_into(out);
    }

    fn decode(data: &[u8], precision: u32) -> Float {
        Float::try_deserialize(data, precision).unwrap()
    }

    fn hex_encode(value: &Float) -> Vec<u8> {
        if value.as_raw().is_zero() || value.as_raw().is_nan() {
            vec![b'0']
        } else {
            value.as_raw().to_string_radix(16, None).into_bytes()
        }
    }

    fn hex_decode(data: &[u8], precision: u32) -> Float {
        use rug::ops::CompleteRound;
        Float::from_raw(
            rug::Float::parse_radix(data, 16)
                .unwrap()
                .complete(precision),
        )
    }

    fn validate() {
        for precision in [2, 7, 8, 9, 53, 63, 64, 65, 256, 4096] {
            let mut values = [
                0.,
                -0.,
                1.,
                -1.,
                1.25,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NAN,
            ]
            .map(|value| Float::with_val(precision, value))
            .to_vec();
            values.push(Float::from_raw(-rug::Float::with_val(
                precision,
                rug::float::Special::Nan,
            )));
            for shift in [-4000, -53, 0, 53, 4000] {
                let mut value = rug::Float::with_val(precision, 1);
                value /= 3;
                value <<= shift;
                values.push(Float::from_raw(value.clone()));
                values.push(Float::from_raw(-value));
            }
            for value in values {
                let mut bytes = Vec::new();
                encode(&value, &mut bytes);
                let decoded = decode(&bytes, precision);
                assert_eq!(value, decoded);
                assert_eq!(decoded.prec(), precision);
                if value.as_raw().is_zero() || value.as_raw().is_nan() {
                    assert!(!decoded.as_raw().is_sign_negative());
                } else {
                    assert_eq!(
                        value.as_raw().is_sign_negative(),
                        decoded.as_raw().is_sign_negative()
                    );
                }
                let mut canonical = Vec::new();
                encode(&decoded, &mut canonical);
                assert_eq!(bytes, canonical);
            }
        }
    }

    fn median(clock: Clock, iterations: usize, mut operation: impl FnMut()) -> f64 {
        for _ in 0..32 {
            operation();
        }
        let mut samples = [0.; 3];
        for ns in &mut samples {
            let timer = clock.start();
            for _ in 0..iterations {
                operation();
            }
            *ns = timer.elapsed().as_nanos() as f64 / iterations as f64;
        }
        samples.sort_by(f64::total_cmp);
        samples[1]
    }

    pub fn run() {
        validate();
        let iterations = std::env::args()
            .nth(1)
            .map_or(100_000, |s| s.parse().unwrap());
        assert!(iterations > 0);
        let clock = Clock::from_env();
        println!("# Clock: {}; three samples per operation", clock.label());
        println!("# Central-value payload only; validation passed for 190 special/boundary cases");
        println!("precision,hex_bytes,binary_bytes,operation,iterations,median_ns");
        for precision in [53, 256, 4096] {
            let value = Float::with_val(precision, 1) / Float::with_val(precision, 3);
            let text = hex_encode(&value);
            let mut binary = Vec::new();
            encode(&value, &mut binary);
            let report = |operation, ns| {
                println!(
                    "{precision},{},{},{operation},{iterations},{ns:.2}",
                    text.len(),
                    binary.len()
                )
            };
            report(
                "hex_encode_allocated",
                median(clock, iterations, || {
                    black_box(hex_encode(black_box(&value)));
                }),
            );
            report(
                "binary_encode_allocated",
                median(clock, iterations, || {
                    let mut buffer = Vec::new();
                    encode(black_box(&value), &mut buffer);
                    black_box(buffer);
                }),
            );
            let mut buffer = Vec::new();
            report(
                "binary_encode_reused",
                median(clock, iterations, || {
                    buffer.clear();
                    encode(black_box(&value), &mut buffer);
                    black_box(&buffer);
                }),
            );
            report(
                "hex_decode",
                median(clock, iterations, || {
                    black_box(hex_decode(black_box(&text), precision));
                }),
            );
            report(
                "binary_decode",
                median(clock, iterations, || {
                    black_box(decode(black_box(&binary), precision));
                }),
            );
        }
    }
}

#[cfg(all(feature = "float-mpfr", feature = "integer-gmp"))]
fn main() {
    benchmark::run();
}

#[cfg(not(all(feature = "float-mpfr", feature = "integer-gmp")))]
fn main() {
    eprintln!("This benchmark requires float-mpfr and integer-gmp.");
}
