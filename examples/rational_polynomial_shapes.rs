//! Deterministic rational/integer polynomial multiplication and exact-division benchmark.
//! The paired `flint_comparison` Rust benchmark validates and measures these same inputs.
use std::{hint::black_box, sync::Arc, time::Instant};
use symbolica::prelude::*;

type P = MultivariatePolynomial<IntegerRing, u16>;
#[path = "../benches/support/rational_shapes.rs"]
mod rational_shapes;
fn main() {
    let rounds: usize = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "8".into())
        .parse()
        .unwrap();
    let directory = std::env::args().nth(2).unwrap();
    std::fs::create_dir_all(&directory).unwrap();
    let variables: Vec<PolyVariable> = [
        symbol!("x"),
        symbol!("y"),
        symbol!("z"),
        symbol!("a"),
        symbol!("b"),
        symbol!("c"),
        symbol!("d"),
        symbol!("e"),
    ]
    .into_iter()
    .map(Into::into)
    .collect();
    println!(
        "shape,bits,rational,operation,domain,nvars,left_terms,right_terms,output_terms,rounds,elapsed_ns"
    );
    for shape in rational_shapes::SHAPES {
        if std::env::args()
            .nth(3)
            .is_some_and(|filter| filter != shape)
        {
            continue;
        }
        for bits in [12, 63, 127] {
            if std::env::args()
                .nth(4)
                .is_some_and(|filter| filter != bits.to_string())
            {
                continue;
            }
            for rational in [false, true] {
                let nvars = if shape.starts_with("sparse") { 8 } else { 3 };
                let vars = Arc::new(variables[..nvars].to_vec());
                let q = rational_shapes::inputs(vars.clone(), shape, bits, rational);
                let scale = Integer::from(if rational { 1155 } else { 1 });
                let z: Vec<P> = q
                    .iter()
                    .map(|p| p.map_coeff(|c| c.numerator_ref() * (&scale / c.denominator_ref()), Z))
                    .collect();
                let zp = &z[0] * &z[1];
                let qp = &q[0] * &q[1];
                assert_eq!(
                    qp,
                    zp.map_coeff(|c| Q.to_element(c.clone(), &scale * &scale, true), Q)
                );
                assert_eq!(qp.try_div(&q[0]).unwrap(), q[1]);
                assert_eq!(zp.try_div(&z[0]).unwrap(), z[1]);
                let name = format!("{shape}-b{bits}-q{rational}");
                for (suffix, value) in [
                    ("left", q[0].to_string()),
                    ("right", q[1].to_string()),
                    ("product", qp.to_string()),
                ] {
                    std::fs::write(format!("{directory}/{name}.{suffix}"), value).unwrap();
                }
                for operation in ["multiply", "exact_division"] {
                    for domain in ["Q", "Z"] {
                        if std::env::args()
                            .nth(5)
                            .is_some_and(|filter| filter != domain)
                        {
                            continue;
                        }
                        let start = Instant::now();
                        for _ in 0..rounds {
                            if domain == "Q" {
                                black_box(if operation == "multiply" {
                                    black_box(&q[0]) * black_box(&q[1])
                                } else {
                                    black_box(&qp).try_div(black_box(&q[0])).unwrap()
                                });
                            } else {
                                black_box(if operation == "multiply" {
                                    black_box(&z[0]) * black_box(&z[1])
                                } else {
                                    black_box(&zp).try_div(black_box(&z[0])).unwrap()
                                });
                            }
                        }
                        println!(
                            "{shape},{bits},{rational},{operation},{domain},{nvars},{},{},{},{rounds},{}",
                            q[0].nterms(),
                            q[1].nterms(),
                            qp.nterms(),
                            start.elapsed().as_nanos()
                        );
                    }
                }
            }
        }
    }
}
