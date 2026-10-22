//! Deterministic bulk-fraction-kernel benchmark with generic arithmetic as an exact oracle.
//! Includes dense, collision-free sparse, tiny, and denominator-variety fallback controls.
use numerica::{
    domains::{
        Ring, RingOps,
        integer::Integer,
        rational::{Q, Rational},
    },
    kernels::DensePolynomialMulRequest,
};
use std::{hint::black_box, time::Instant};

fn naive(
    left: &[Rational],
    li: &[u32],
    right: &[Rational],
    ri: &[u32],
    length: usize,
) -> Vec<(u32, Rational)> {
    let mut output = vec![Rational::zero(); length];
    for (a, &i) in left.iter().zip(li) {
        for (b, &j) in right.iter().zip(ri) {
            Q.add_mul_assign(&mut output[(i + j) as usize], a, b);
        }
    }
    output
        .into_iter()
        .enumerate()
        .filter(|(_, c)| !c.is_zero())
        .map(|(i, c)| (i as u32, c))
        .collect()
}

fn main() {
    let rounds: usize = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "5".into())
        .parse()
        .unwrap();
    println!("shape,bits,denominators,method,rounds,elapsed_ns");
    for shape in ["dense3_d12", "sparse_unique", "tiny"] {
        let (li, ri, length): (Vec<u32>, Vec<u32>, usize) = if shape == "dense3_d12" {
            let indices = |degree| {
                let mut v = vec![];
                for x in 0..=degree {
                    for y in 0..=degree - x {
                        for z in 0..=degree - x - y {
                            v.push(x * 24 * 24 + y * 24 + z);
                        }
                    }
                }
                v
            };
            (indices(12), indices(11), 24 * 24 * 24)
        } else if shape == "sparse_unique" {
            ((0..32).map(|i| i * 32).collect(), (0..32).collect(), 1024)
        } else {
            (vec![0, 1, 2], vec![0, 1], 4)
        };
        for bits in [12u32, 63, 127] {
            for denominators in ["one", "few", "many"] {
                let coefficients = |len: usize| {
                    (0..len)
                        .map(|i| {
                            let value = (Integer::from((i % 19 + 1) as i64) << bits - 5)
                                + Integer::from((i % 127 + 1) as i64);
                            let denominator = match denominators {
                                "one" => 1,
                                "few" => [1, 3, 5, 7, 11][i % 5],
                                _ => 101 + 2 * i,
                            };
                            Q.to_element(value, Integer::from(denominator), true)
                        })
                        .collect::<Vec<_>>()
                };
                let left = coefficients(li.len());
                let right = coefficients(ri.len());
                let expected = naive(&left, &li, &right, &ri, length);
                let call = || {
                    Q.kernels().polynomial().and_then(|kernel| {
                        kernel.try_dense_mul(DensePolynomialMulRequest {
                            output_len: length,
                            left_coefficients: &left,
                            left_indices: &li,
                            right_coefficients: &right,
                            right_indices: &ri,
                        })
                    })
                };
                if let Some(output) = call() {
                    assert_eq!(output, expected);
                }
                for method in ["generic", "bulk"] {
                    let start = Instant::now();
                    for _ in 0..rounds {
                        black_box(if method == "generic" {
                            naive(&left, &li, &right, &ri, length)
                        } else {
                            call().unwrap_or_else(|| naive(&left, &li, &right, &ri, length))
                        });
                    }
                    println!(
                        "{shape},{bits},{denominators},{method},{rounds},{}",
                        start.elapsed().as_nanos()
                    );
                }
            }
        }
    }
}
