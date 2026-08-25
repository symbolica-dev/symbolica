use std::{fmt::Write, hint::black_box, sync::atomic::Ordering, time::Instant};

use symbolica::GLOBAL_SETTINGS;
use symbolica::prelude::*;

#[derive(Clone, Copy)]
struct Timings {
    construction: f64,
    multiplication: f64,
    gcd: f64,
    validation: f64,
    total: f64,
}

fn median(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn linear_expression(weights: &[String], signs: &[i8]) -> String {
    let mut expression = "1".to_owned();
    for (index, (weight, &sign)) in weights.iter().zip(signs).enumerate() {
        write!(
            expression,
            "{}{weight}*x{}",
            if sign < 0 { '-' } else { '+' },
            index + 1
        )
        .unwrap();
    }
    expression
}

fn power_of_two_plus_offset(bits: u32, offset: u32) -> String {
    let mut decimal_digits = vec![1u8];
    for _ in 1..bits {
        let mut carry = 0;
        for digit in &mut decimal_digits {
            let doubled = *digit as u16 * 2 + carry;
            *digit = (doubled % 10) as u8;
            carry = doubled / 10;
        }
        if carry > 0 {
            decimal_digits.push(carry as u8);
        }
    }

    let mut carry = offset;
    let mut index = 0;
    while carry > 0 {
        if index == decimal_digits.len() {
            decimal_digits.push(0);
        }
        let sum = decimal_digits[index] as u32 + carry;
        decimal_digits[index] = (sum % 10) as u8;
        carry = sum / 10;
        index += 1;
    }

    decimal_digits
        .iter()
        .rev()
        .map(|digit| char::from(b'0' + *digit))
        .collect()
}

fn coefficient_weights(benchmark_case: &str, variable_count: usize, bits: u32) -> Vec<String> {
    const SMALL_WEIGHTS: [u64; 8] = [3, 5, 7, 9, 11, 13, 15, 17];
    const THIRTY_BIT_WEIGHTS: [u64; 8] = [
        1_000_000_007,
        1_000_000_009,
        1_000_000_033,
        1_000_000_087,
        1_000_000_093,
        1_000_000_097,
        1_000_000_103,
        1_000_000_123,
    ];
    const OFFSETS: [u32; 8] = [7, 9, 33, 87, 93, 97, 103, 123];

    let weights = if benchmark_case != "high-height" {
        SMALL_WEIGHTS[..variable_count]
            .iter()
            .map(ToString::to_string)
            .collect()
    } else if bits == 30 {
        THIRTY_BIT_WEIGHTS[..variable_count]
            .iter()
            .map(ToString::to_string)
            .collect()
    } else {
        OFFSETS[..variable_count]
            .iter()
            .map(|&offset| power_of_two_plus_offset(bits, offset))
            .collect()
    };
    weights
}

fn powered_expression(linear: &str, degree: u32, constant: i32) -> String {
    format!(
        "({linear})^{degree}{}{constant}",
        if constant < 0 { "" } else { "+" }
    )
}

fn sparse_expression(variable_count: usize, degree: u32) -> String {
    const COEFFICIENTS: [u64; 8] = [1, 2, 3, 5, 7, 11, 13, 17];
    let mut expression = "1".to_owned();
    for (index, coefficient) in COEFFICIENTS[..variable_count].iter().enumerate() {
        if *coefficient == 1 {
            write!(expression, "+x{}^{degree}", index + 1).unwrap();
        } else {
            write!(expression, "+{coefficient}*x{}^{degree}", index + 1).unwrap();
        }
    }
    expression
}

fn parse_integer_polynomial(expression: &str) -> MultivariatePolynomial<IntegerRing, u16> {
    Atom::parse(expression, "gcd_benchmark", ParseSettings::default())
        .unwrap()
        .to_polynomial(&Z, None)
}

fn construct_case(
    benchmark_case: &str,
    variable_count: usize,
    degree: u32,
    gap: u32,
    coefficient_bits: u32,
) -> [MultivariatePolynomial<IntegerRing, u16>; 3] {
    const B_SIGNS: [i8; 8] = [-1, -1, -1, 1, -1, -1, 1, -1];

    let weights = coefficient_weights(benchmark_case, variable_count, coefficient_bits);
    let positive_signs = vec![1; variable_count];
    let mut gcd_signs = positive_signs.clone();
    gcd_signs[variable_count - 1] = -1;
    let a_linear = linear_expression(&weights, &positive_signs);
    let b_linear = linear_expression(&weights, &B_SIGNS[..variable_count]);
    let gcd_linear = linear_expression(&weights, &gcd_signs);

    let gcd_expression = match benchmark_case {
        "dense" | "high-height" => powered_expression(&gcd_linear, degree, 3),
        "sparse" => sparse_expression(variable_count, degree),
        "high-gap" => sparse_expression(variable_count, gap),
        _ => unreachable!(),
    };

    [
        parse_integer_polynomial(&powered_expression(&a_linear, degree, -1)),
        parse_integer_polynomial(&powered_expression(&b_linear, degree, 1)),
        parse_integer_polynomial(&gcd_expression),
    ]
}

fn should_use_hu(
    a: &MultivariatePolynomial<IntegerRing, u16>,
    b: &MultivariatePolynomial<IntegerRing, u16>,
    gcd: &MultivariatePolynomial<IntegerRing, u16>,
) -> bool {
    let vars = (0..gcd.nvars())
        .filter(|&variable| gcd.degree(variable) > 0)
        .collect::<Vec<_>>();
    if vars.len() < 3 || vars.first() != Some(&0) {
        return false;
    }

    let mut box_size = 1u128;
    let mut cofactor_box_size = 1u128;
    let mut kronecker_range = 1u128;
    for &variable in vars.iter().skip(1) {
        let bound = gcd.degree(variable) as u128;
        box_size = box_size.saturating_mul(bound + 1);
        kronecker_range = kronecker_range.saturating_mul(
            a.degree(variable)
                .max(b.degree(variable))
                .max(gcd.degree(variable)) as u128
                + 1,
        );
        let smaller_degree = if a.nterms() < b.nterms() {
            a.degree(variable)
        } else {
            b.degree(variable)
        } as u128;
        cofactor_box_size =
            cofactor_box_size.saturating_mul(smaller_degree.saturating_sub(bound) + 1);
    }

    kronecker_range <= u32::MAX as u128
        && (cofactor_box_size.saturating_mul(8) < box_size
            || (a.nterms() + b.nterms()) as u128 * 8 < box_size)
}

fn main() {
    let benchmark_case = std::env::var("GCD_BENCH_CASE").unwrap_or_else(|_| "dense".to_owned());
    assert!(
        matches!(
            benchmark_case.as_str(),
            "dense" | "sparse" | "high-gap" | "high-height"
        ),
        "GCD_BENCH_CASE must be dense, sparse, high-gap, or high-height"
    );
    let gap = std::env::var("GCD_BENCH_GAP")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(10);
    assert!(gap > 0, "GCD_BENCH_GAP must be positive");
    let variable_count = std::env::var("GCD_BENCH_NVARS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(7);
    assert!(
        (2..=8).contains(&variable_count),
        "GCD_BENCH_NVARS must be between 2 and 8"
    );
    let degree = std::env::var("GCD_BENCH_DEGREE")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(7);
    assert!(degree > 0, "GCD_BENCH_DEGREE must be positive");
    assert!(degree <= u16::MAX as u32, "GCD_BENCH_DEGREE is too large");
    let coefficient_bits = std::env::var("GCD_BENCH_COEFFICIENT_BITS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(30);
    assert!(
        (8..=1024).contains(&coefficient_bits),
        "GCD_BENCH_COEFFICIENT_BITS must be between 8 and 1024"
    );
    let backend = std::env::var("GCD_BENCH_BACKEND").unwrap_or_else(|_| "auto".to_owned());
    match backend.as_str() {
        "auto" => {
            GLOBAL_SETTINGS
                .use_hu_monagan_poly_gcd
                .store(true, Ordering::Relaxed);
            GLOBAL_SETTINGS
                .force_hu_monagan_poly_gcd
                .store(false, Ordering::Relaxed);
        }
        "hu" => {
            GLOBAL_SETTINGS
                .use_hu_monagan_poly_gcd
                .store(true, Ordering::Relaxed);
            GLOBAL_SETTINGS
                .force_hu_monagan_poly_gcd
                .store(true, Ordering::Relaxed);
        }
        "zippel" | "product" => {
            GLOBAL_SETTINGS
                .use_hu_monagan_poly_gcd
                .store(false, Ordering::Relaxed);
            GLOBAL_SETTINGS
                .force_hu_monagan_poly_gcd
                .store(false, Ordering::Relaxed);
        }
        _ => panic!("GCD_BENCH_BACKEND must be auto, hu, zippel, or product"),
    }

    let samples = std::env::var("GCD_BENCH_SAMPLES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(1);
    assert!(samples > 0, "GCD_BENCH_SAMPLES must be positive");

    println!("Symbolica {}", env!("CARGO_PKG_VERSION"));
    println!("case {benchmark_case}");
    println!("variables {variable_count}");
    println!("degree {degree}");
    if benchmark_case == "high-height" {
        println!("coefficient_bits {coefficient_bits}");
    }
    if benchmark_case == "high-gap" {
        println!("gap {gap}");
    }
    println!("backend {backend}");
    println!("samples {samples}");

    let mut timings = Vec::with_capacity(samples);
    let mut term_counts = None;
    for _ in 0..samples {
        let total_start = Instant::now();

        let phase_start = Instant::now();
        let mut polys = construct_case(
            &benchmark_case,
            variable_count,
            degree,
            gap,
            coefficient_bits,
        );
        MultivariatePolynomial::unify_variables_list(&mut polys);
        let construction = phase_start.elapsed().as_secs_f64();
        let [a, b, g] = polys;

        let phase_start = Instant::now();
        let ag = black_box(&a) * black_box(&g);
        let bg = black_box(&b) * black_box(&g);
        let multiplication = phase_start.elapsed().as_secs_f64();

        if term_counts.is_none() && matches!(backend.as_str(), "auto" | "hu") {
            println!(
                "effective_backend {}",
                if should_use_hu(&ag, &bg, &g) {
                    "hu"
                } else {
                    "zippel (adaptive Hu handoff)"
                }
            );
        }

        let (computed_gcd, gcd) = if backend == "product" {
            (g.clone(), 0.0)
        } else {
            let phase_start = Instant::now();
            let computed_gcd = black_box(&ag).gcd(black_box(&bg));
            (computed_gcd, phase_start.elapsed().as_secs_f64())
        };

        let phase_start = Instant::now();
        let difference = &computed_gcd - &g;
        assert!(difference.is_zero(), "computed GCD differs from g");
        black_box(difference);
        let validation = phase_start.elapsed().as_secs_f64();

        let total = total_start.elapsed().as_secs_f64();
        term_counts = Some((
            a.nterms(),
            b.nterms(),
            g.nterms(),
            ag.nterms(),
            bg.nterms(),
            computed_gcd.nterms(),
        ));
        timings.push(Timings {
            construction,
            multiplication,
            gcd,
            validation,
            total,
        });
    }

    let summarize =
        |field: fn(Timings) -> f64| median(timings.iter().copied().map(field).collect());
    println!(
        "terms a/b/g/ag/bg/gcd {}/{}/{}/{}/{}/{}",
        term_counts.unwrap().0,
        term_counts.unwrap().1,
        term_counts.unwrap().2,
        term_counts.unwrap().3,
        term_counts.unwrap().4,
        term_counts.unwrap().5,
    );
    println!(
        "construction   {:10.3} ms",
        summarize(|t| t.construction) * 1_000.0
    );
    println!(
        "multiplication {:10.3} ms",
        summarize(|t| t.multiplication) * 1_000.0
    );
    println!("gcd            {:10.3} ms", summarize(|t| t.gcd) * 1_000.0);
    println!(
        "validation     {:10.3} ms",
        summarize(|t| t.validation) * 1_000.0
    );
    println!(
        "total          {:10.3} ms",
        summarize(|t| t.total) * 1_000.0
    );
}
