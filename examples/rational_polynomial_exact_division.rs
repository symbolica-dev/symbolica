//! Compare checked rational polynomial division with quotient/remainder division.
//!
//! Run with `cargo run --release --example rational_polynomial_exact_division`.
//! Input construction and equality verification are excluded from timings.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;
use symbolica::domains::rational::RationalField;
use symbolica::prelude::*;

type Polynomial = MultivariatePolynomial<RationalField, u16>;

fn coefficient(index: usize, bits: u64, fractions: bool) -> Rational {
    let magnitude =
        Integer::from(2).pow(bits - 1) + Integer::from(((index * 17 + 5) % 1024) as i64);
    let primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31];
    let denominator = if fractions {
        primes[index % primes.len()]
    } else {
        1
    };
    Rational::from((magnitude, Integer::from(denominator)))
}

fn variables(count: usize) -> Arc<Vec<PolyVariable>> {
    Arc::new(
        (0..count)
            .map(|i| {
                PolyVariable::Symbol(
                    Symbol::parse(&format!("v{i}"), "exact_division_probe").unwrap(),
                )
            })
            .collect(),
    )
}

fn dense(degree: u16, seed: usize, bits: u64, fractions: bool) -> Polynomial {
    let mut polynomial = Polynomial::new(&Q, None, variables(3));
    let mut index = seed;
    for x in 0..=degree {
        for y in 0..=degree - x {
            for z in 0..=degree - x - y {
                polynomial.append_monomial(coefficient(index, bits, fractions), &[x, y, z]);
                index += 1;
            }
        }
    }
    polynomial
}

fn sparse(degree: u16, seed: usize, bits: u64, fractions: bool) -> Polynomial {
    let mut polynomial = Polynomial::new(&Q, None, variables(8));
    for i in 0..8 {
        let mut exponents = vec![0; 8];
        exponents[i] = degree;
        polynomial.append_monomial(coefficient(seed + i, bits, fractions), &exponents);
    }
    polynomial
}

fn quot_rem_checked(dividend: &Polynomial, divisor: &Polynomial) -> Option<Polynomial> {
    let (quotient, remainder) = dividend.quot_rem(divisor, false);
    remainder.is_zero().then_some(quotient)
}

fn main() {
    println!(
        "shape,bits,fractions,dividend_terms,divisor_terms,quot_rem_ns,try_div_ns,try_div_over_quot_rem"
    );
    for bits in [12, 63, 127] {
        for fractions in [false, true] {
            for (label, divisor, quotient) in [
                (
                    "tiny",
                    dense(1, 1, bits, fractions),
                    dense(1, 9, bits, fractions),
                ),
                (
                    "sparse8_d128",
                    sparse(128, 1, bits, fractions),
                    sparse(127, 9, bits, fractions),
                ),
                (
                    "dense3_d4",
                    dense(4, 1, bits, fractions),
                    dense(3, 9, bits, fractions),
                ),
                (
                    "nondividing_dense3_d4",
                    dense(4, 1, bits, fractions),
                    dense(3, 9, bits, fractions),
                ),
                (
                    "dense3_d10",
                    dense(10, 1, bits, fractions),
                    dense(9, 9, bits, fractions),
                ),
            ] {
                let mut dividend = &divisor * &quotient;
                let expected = if label.starts_with("nondividing") {
                    dividend = &dividend + &dividend.one();
                    None
                } else {
                    Some(quotient)
                };
                assert_eq!(quot_rem_checked(&dividend, &divisor), expected);
                assert_eq!(dividend.try_div(&divisor), expected);
                let loops = if label == "tiny" { 1024 } else { 1 };
                let mut quot_rem = vec![];
                let mut checked = vec![];
                for pair in 0..5 {
                    for checked_first in [pair % 2 == 0, pair % 2 != 0] {
                        let start = Instant::now();
                        for _ in 0..loops {
                            black_box(if checked_first {
                                black_box(&dividend).try_div(black_box(&divisor))
                            } else {
                                quot_rem_checked(black_box(&dividend), black_box(&divisor))
                            });
                        }
                        let elapsed = start.elapsed().as_nanos() / loops;
                        if checked_first {
                            checked.push(elapsed);
                        } else {
                            quot_rem.push(elapsed);
                        }
                    }
                }
                quot_rem.sort_unstable();
                checked.sort_unstable();
                let g = quot_rem[2];
                let c = checked[2];
                println!(
                    "{label},{bits},{fractions},{},{},{g},{c},{}",
                    dividend.nterms(),
                    divisor.nterms(),
                    c as f64 / g as f64
                );
            }
        }
    }
}
