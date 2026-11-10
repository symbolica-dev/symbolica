//! Manual diagnostic benchmark against the previous rational-index algorithm.
//! Run with `cargo test --release --lib benchmark_series_convolution -- --ignored --nocapture`.
use super::{Series, multiplication_tests::multiply_reference};
use crate::{
    atom::Atom,
    domains::{Ring, atom::AtomField, rational::Q},
    function, symbol,
};
use std::{hint::black_box, sync::Arc, time::Instant};

fn time_case<F: Ring>(label: &str, left: &Series<F>, right: &Series<F>, iterations: usize) {
    let expected = multiply_reference(left, right);
    let actual = left * right;
    assert_eq!(actual.coefficients, expected.coefficients);
    assert_eq!(actual.shift, expected.shift);
    assert_eq!(actual.order, expected.order);
    assert_eq!(actual.ramification, expected.ramification);
    for _ in 0..3 {
        black_box(multiply_reference(left, right));
        black_box(left * right);
    }
    // Alternating order exposes drift without running two native workloads at
    // once. Expression creation and coefficient comparisons are outside timers.
    for repeat in 0..6 {
        let modes = if repeat % 2 == 0 {
            ["rational_indices", "integer_indices"]
        } else {
            ["integer_indices", "rational_indices"]
        };
        for mode in modes {
            let start = Instant::now();
            for _ in 0..iterations {
                let output = if mode == "rational_indices" {
                    multiply_reference(black_box(left), black_box(right))
                } else {
                    black_box(left) * black_box(right)
                };
                black_box(output);
            }
            println!(
                "{{\"case\":\"{label}\",\"mode\":\"{mode}\",\"repeat\":{repeat},\"iterations\":{iterations},\"elapsed_ns\":{}}}",
                start.elapsed().as_nanos()
            );
        }
    }
}

#[test]
#[ignore = "manual release timing; keep other native workloads stopped"]
fn benchmark_series_convolution() {
    let variable = Arc::new(super::PolyVariable::from(symbol!(
        "series_mul_benchmark::epsilon"
    )));
    let left = Series {
        coefficients: (0..64).map(|i| (i % 11 - 4).into()).collect(),
        field: Q,
        variable: variable.clone(),
        expansion_point: 0.into(),
        shift: -3,
        order: 64,
        ramification: 1,
    };
    let mut right = left.clone();
    right.coefficients.reverse();
    time_case("rational_depth64", &left, &right, 1_000);

    let a = symbol!("series_mul_benchmark::a");
    let b = symbol!("series_mul_benchmark::b");
    let field = AtomField {
        statistical_zero_test: false,
        cancel_check_on_division: false,
        custom_normalization: None,
    };
    for (depth, terms, iterations) in [(16, 4, 400), (32, 16, 100)] {
        let left = Series {
            coefficients: (0..depth)
                .map(|i| Atom::add_many((0..terms).map(|j| function!(a, i, j))))
                .collect(),
            field: field.clone(),
            variable: variable.clone(),
            expansion_point: Atom::Zero,
            shift: -3,
            order: depth,
            ramification: 1,
        };
        let mut right = left.clone();
        right.coefficients = (0..depth)
            .map(|i| Atom::add_many((0..terms).map(|j| function!(b, i, j))))
            .collect();
        time_case(
            &format!("symbolic_depth{depth}_terms{terms}"),
            &left,
            &right,
            iterations,
        );
        right.ramification = 3;
        time_case(
            &format!("symbolic_mixed_grid_depth{depth}_terms{terms}"),
            &left,
            &right,
            iterations,
        );
    }
}
