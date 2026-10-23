//! Compare packed floating coefficients without decoding them.
//! The decoded baseline reproduces the former float/float comparison path.

use std::{cmp::Ordering, hint::black_box};
use symbolica::{
    atom::{Atom, AtomView},
    coefficient::CoefficientView,
    domains::float::{Complex, Float, FloatLike},
};

#[path = "../lib/numerica/examples/support/clock.rs"]
mod clock;
use clock::Clock;

fn view(atom: &Atom) -> CoefficientView<'_> {
    let AtomView::Num(number) = atom.as_view() else {
        unreachable!();
    };
    number.get_coeff_view()
}

fn decoded_comparison(left: CoefficientView<'_>, right: CoefficientView<'_>) -> Ordering {
    let (CoefficientView::Float(lr, li), CoefficientView::Float(rr, ri)) = (left, right) else {
        panic!("benchmark requires floating coefficients");
    };
    lr.to_float()
        .partial_cmp(&rr.to_float())
        .unwrap_or(Ordering::Equal)
        .then_with(|| {
            li.to_float()
                .partial_cmp(&ri.to_float())
                .unwrap_or(Ordering::Equal)
        })
}

fn measure(clock: Clock, iterations: usize, pairs: &[(Atom, Atom)], decode: bool) -> f64 {
    let operation = || {
        for (left, right) in pairs {
            let (left, right) = (view(black_box(left)), view(black_box(right)));
            black_box(if decode {
                decoded_comparison(left, right)
            } else {
                left.cmp(&right)
            });
        }
    };
    for _ in 0..16 {
        operation();
    }
    let mut samples = [0.; 3];
    for ns in &mut samples {
        let timer = clock.start();
        for _ in 0..iterations {
            operation();
        }
        *ns = timer.elapsed().as_nanos() as f64 / (iterations * pairs.len()) as f64;
    }
    samples.sort_by(f64::total_cmp);
    samples[1]
}

fn main() {
    let iterations: usize = std::env::args()
        .nth(1)
        .map_or(50_000, |n| n.parse().unwrap());
    assert!(iterations > 0);
    let clock = Clock::from_env();
    println!(
        "# Clock: {}; three samples per operation; eight pairs per iteration",
        clock.label()
    );
    println!("precision,case,iterations,borrowed_ns,decoded_ns,speedup");
    for precision in [53, 256, 4096] {
        let third = Float::with_val(precision, 1) / Float::with_val(precision, 3);
        let seventh = Float::with_val(precision, 1) / Float::with_val(precision, 7);
        let delta = Float::with_val(precision, 2)
            .pow(u64::from(precision - 4))
            .inv();
        let adjacent = third.clone() + delta;
        for complex in [false, true] {
            let real_pairs = [
                (third.clone(), seventh.clone()),
                (third.clone(), adjacent.clone()),
                (third.clone(), third.clone()),
                (third.clone(), -third.clone()),
                (-third.clone(), -seventh.clone()),
                (
                    Float::with_val(precision, 1.25),
                    Float::with_val(precision + 37, 1.25),
                ),
                (Float::with_val(precision, f64::INFINITY), third.clone()),
                (Float::with_val(precision, f64::NAN), third.clone()),
            ];
            let pairs: Vec<_> = real_pairs
                .into_iter()
                .enumerate()
                .map(|(index, (left, right))| {
                    let li = if complex {
                        seventh.clone()
                    } else {
                        Float::new(precision)
                    };
                    let ri = if complex && index % 2 == 0 {
                        third.clone()
                    } else {
                        li.clone()
                    };
                    (
                        Atom::num(Complex::new(left, li)),
                        Atom::num(Complex::new(right, ri)),
                    )
                })
                .collect();
            for (left, right) in &pairs {
                assert_eq!(
                    view(left).cmp(&view(right)),
                    decoded_comparison(view(left), view(right))
                );
            }
            let borrowed = measure(clock, iterations, &pairs, false);
            let decoded = measure(clock, iterations, &pairs, true);
            println!(
                "{precision},{},{iterations},{borrowed:.2},{decoded:.2},{:.3}",
                if complex { "complex" } else { "real" },
                decoded / borrowed
            );
        }
    }
}
