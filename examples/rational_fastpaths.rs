//! Reproducible scalar and coefficient-convolution benchmarks for fraction arithmetic.
//!
//! Run the same source with baseline and candidate Numerica revisions. Timings
//! exclude deterministic input construction and verify round-trip cancellation.
use std::{hint::black_box, time::Instant};

use numerica::domains::{
    RingOps,
    integer::Integer,
    rational::{Q, Rational},
};

fn coefficients(bits: u32, rational: bool) -> Vec<Rational> {
    (0..48)
        .map(|index| {
            let small = Integer::from((index * 17 % 113) as i64 - 56);
            let numerator = if bits <= 12 {
                small
            } else {
                (small << (bits - 7)) + Integer::from((index * 23 % 127) as i64)
            };
            let denominator = if rational { 3 + 2 * (index % 7) } else { 1 };
            Q.to_element(numerator, Integer::from(denominator), true)
        })
        .collect()
}

fn main() {
    let rounds: usize = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "2000".into())
        .parse()
        .unwrap();
    println!("bits,rational,operation,rounds,elapsed_ns");
    for bits in [12, 63, 127, 512] {
        for rational in [false, true] {
            let left = coefficients(bits, rational);
            let mut right = left.clone();
            right.reverse();
            for operation in ["add", "multiply", "fused_round_trip", "convolution"] {
                let operation_rounds = if operation == "convolution" {
                    (rounds / 48).max(1)
                } else {
                    rounds
                };
                let start = Instant::now();
                for _ in 0..operation_rounds {
                    if operation == "convolution" {
                        let mut output = vec![Rational::zero(); 95];
                        for (i, a) in left.iter().enumerate() {
                            for (j, b) in right.iter().enumerate() {
                                Q.add_mul_assign(&mut output[i + j], a, b);
                            }
                        }
                        black_box(output);
                    } else {
                        for (a, b) in left.iter().zip(&right) {
                            match operation {
                                "add" => {
                                    black_box(Q.add(black_box(a), black_box(b)));
                                }
                                "multiply" => {
                                    black_box(Q.mul(black_box(a), black_box(b)));
                                }
                                "fused_round_trip" => {
                                    let mut result = a.clone();
                                    Q.add_mul_assign(&mut result, black_box(a), black_box(b));
                                    Q.sub_mul_assign(&mut result, black_box(a), black_box(b));
                                    assert_eq!(black_box(result), *a);
                                }
                                _ => unreachable!(),
                            }
                        }
                    }
                }
                println!(
                    "{bits},{rational},{operation},{operation_rounds},{}",
                    start.elapsed().as_nanos()
                );
            }
        }
    }
}
