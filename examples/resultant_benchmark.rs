use std::{hint::black_box, time::Instant};

use symbolica::coefficient::ConvertToRing;
use symbolica::prelude::*;

fn median_seconds(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn measure<T>(iterations: usize, mut operation: impl FnMut() -> T) -> f64 {
    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        black_box(operation());
        samples.push(start.elapsed().as_secs_f64());
    }

    median_seconds(samples)
}

fn measure_batched<T>(iterations: usize, mut operation: impl FnMut() -> T) -> f64 {
    let start = Instant::now();
    black_box(operation());
    let calibration = start.elapsed().as_secs_f64();
    let batch_size = ((0.020 / calibration.max(1e-9)) as usize).clamp(1, 256);

    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let start = Instant::now();
        for _ in 0..batch_size {
            black_box(operation());
        }
        samples.push(start.elapsed().as_secs_f64() / batch_size as f64);
    }

    median_seconds(samples)
}

fn benchmark_selected(name: &str) -> bool {
    std::env::var("BENCHMARK_FILTER")
        .map(|filter| name.contains(&filter))
        .unwrap_or(true)
}

fn compare(name: &str, a: &str, b: &str, iterations: usize) {
    if !benchmark_selected(name) {
        return;
    }

    let iterations = std::env::var("RESULTANT_BENCH_SAMPLES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(iterations);
    let mut polys = [
        parse!(a).to_polynomial::<_, u16>(&Z, None),
        parse!(b).to_polynomial::<_, u16>(&Z, None),
    ];
    MultivariatePolynomial::unify_variables_list(&mut polys);

    let variable = polys[0]
        .variables()
        .iter()
        .position(|v| v == &PolyVariable::Symbol(symbol!("x")))
        .expect("benchmark input should contain x");
    let a = polys[0].to_univariate(variable);
    let b = polys[1].to_univariate(variable);

    let algorithm =
        std::env::var("RESULTANT_BENCH_ALGORITHM").unwrap_or_else(|_| "ducos".to_string());
    let expected = a.resultant_ducos(&b);
    if std::env::var_os("RESULTANT_BENCH_DUCOS_ONLY").is_some() || algorithm != "ducos" {
        let elapsed = match algorithm.as_str() {
            "ducos" => measure(iterations, || a.resultant_ducos(&b)),
            "prs" => measure(iterations, || {
                let result = a.resultant_prs(&b);
                assert_eq!(result, expected);
                result
            }),
            "primitive" => measure(iterations, || {
                let result = a.resultant_primitive(&b);
                assert_eq!(result, expected);
                result
            }),
            "crt" => measure(iterations, || {
                let result = a.resultant_ducos_crt(&b);
                assert_eq!(result, expected);
                result
            }),
            _ => panic!("unknown resultant benchmark algorithm: {algorithm}"),
        };
        println!(
            "{name:32} {algorithm:9} {:9.3} ms  terms {}",
            elapsed * 1_000.0,
            expected.nterms(),
        );
        return;
    }

    assert_eq!(a.resultant_ducos_crt(&b), expected);

    let ducos = measure(iterations, || a.resultant_ducos(&b));
    let crt = measure(iterations, || a.resultant_ducos_crt(&b));
    println!(
        "{name:32} Ducos {:9.3} ms  CRT {:9.3} ms  CRT/Ducos {:6.2}x  terms {}",
        ducos * 1_000.0,
        crt * 1_000.0,
        ducos / crt,
        expected.nterms(),
    );
}

fn compare_multiplication(
    name: &str,
    a: &str,
    a_power: usize,
    b: &str,
    b_power: usize,
    iterations: usize,
) {
    compare_multiplication_with_options(name, a, a_power, false, b, b_power, false, iterations);
}

fn compare_exact_division(
    name: &str,
    quotient_base: &str,
    quotient_power: usize,
    divisor_base: &str,
    divisor_power: usize,
    iterations: usize,
) {
    if !benchmark_selected(name) {
        return;
    }

    let iterations = std::env::var("DIVISION_BENCH_SAMPLES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(iterations);
    let quotient = parse!(quotient_base)
        .to_polynomial::<_, u16>(&Z, None)
        .pow(quotient_power);
    let divisor = parse!(divisor_base)
        .to_polynomial::<_, u16>(&Z, None)
        .pow(divisor_power);
    let mut polys = [quotient, divisor];
    MultivariatePolynomial::unify_variables_list(&mut polys);
    let dividend = &polys[0] * &polys[1];
    assert_eq!(dividend.clone().try_div_owned(&polys[1]).unwrap(), polys[0]);

    let mut samples = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        // `try_div_owned` consumes its dividend. Keep the unavoidable benchmark clone outside
        // the timed region so this matches FLINT's non-consuming `fmpz_mpoly_divides` call.
        let owned_dividend = dividend.clone();
        let start = Instant::now();
        let quotient = owned_dividend.try_div_owned(&polys[1]).unwrap();
        black_box(quotient);
        samples.push(start.elapsed().as_secs_f64());
    }
    let division = median_seconds(samples);
    println!(
        "{name:32} DIV   {:9.3} ms  dividend/divisor/quotient terms {}/{}/{}",
        division * 1_000.0,
        dividend.nterms(),
        polys[1].nterms(),
        polys[0].nterms(),
    );
}

#[allow(clippy::too_many_arguments)]
fn compare_multiplication_with_options(
    name: &str,
    a: &str,
    a_power: usize,
    subtract_one_from_a: bool,
    b: &str,
    b_power: usize,
    subtract_one_from_b: bool,
    iterations: usize,
) {
    if !benchmark_selected(name) {
        return;
    }

    let iterations = std::env::var("MULTIPLICATION_BENCH_SAMPLES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(iterations);
    let mut a = parse!(a).to_polynomial::<_, u16>(&Z, None).pow(a_power);
    let mut b = parse!(b).to_polynomial::<_, u16>(&Z, None).pow(b_power);
    if subtract_one_from_a {
        a = a.add_constant((-1).into());
    }
    if subtract_one_from_b {
        b = b.add_constant((-1).into());
    }
    let mut polys = [a, b];
    MultivariatePolynomial::unify_variables_list(&mut polys);

    let product = &polys[0] * &polys[1];
    let multiplication = measure(iterations, || &polys[0] * &polys[1]);
    println!(
        "{name:32} MUL   {:9.3} ms  lhs/rhs/product terms {}/{}/{}",
        multiplication * 1_000.0,
        polys[0].nterms(),
        polys[1].nterms(),
        product.nterms(),
    );
}

#[allow(clippy::too_many_arguments)]
fn compare_finite_field_multiplication<F: EuclideanDomain + ConvertToRing>(
    name: &str,
    ring: &F,
    a: &str,
    a_power: usize,
    subtract_one_from_a: bool,
    b: &str,
    b_power: usize,
    subtract_one_from_b: bool,
    iterations: usize,
) {
    if !benchmark_selected(name) {
        return;
    }

    let iterations = std::env::var("MULTIPLICATION_BENCH_SAMPLES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(iterations);
    let mut a = parse!(a).to_polynomial::<_, u16>(ring, None).pow(a_power);
    let mut b = parse!(b).to_polynomial::<_, u16>(ring, None).pow(b_power);
    if subtract_one_from_a {
        a = a.add_constant(ring.neg(&ring.one()));
    }
    if subtract_one_from_b {
        b = b.add_constant(ring.neg(&ring.one()));
    }
    let mut polys = [a, b];
    MultivariatePolynomial::unify_variables_list(&mut polys);

    let product = &polys[0] * &polys[1];
    let multiplication = measure_batched(iterations, || &polys[0] * &polys[1]);
    println!(
        "{name:48} MUL   {:9.3} ms  lhs/rhs/product terms {}/{}/{}",
        multiplication * 1_000.0,
        polys[0].nterms(),
        polys[1].nterms(),
        product.nterms(),
    );
}

fn compare_finite_field_dense_univariate<F: EuclideanDomain + ConvertToRing>(
    name: &str,
    ring: &F,
    left_degree: usize,
    right_degree: usize,
    iterations: usize,
) {
    if !benchmark_selected(name) {
        return;
    }

    let iterations = std::env::var("MULTIPLICATION_BENCH_SAMPLES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(iterations);
    let template = parse!("x").to_polynomial::<_, u16>(ring, None);
    let mut left =
        MultivariatePolynomial::new(ring, Some(left_degree + 1), template.variables().clone());
    let mut right =
        MultivariatePolynomial::new(ring, Some(right_degree + 1), template.variables().clone());
    for exponent in 0..=left_degree {
        left.append_monomial_back(
            ring.nth(((exponent % 16) as u64 + 1).into()),
            &[exponent as u16],
        );
    }
    for exponent in 0..=right_degree {
        right.append_monomial_back(
            ring.nth((((7 * exponent) % 16) as u64 + 1).into()),
            &[exponent as u16],
        );
    }

    let product = &left * &right;
    let multiplication = measure_batched(iterations, || &left * &right);
    println!(
        "{name:48} MUL   {:9.3} ms  lhs/rhs/product terms {}/{}/{}",
        multiplication * 1_000.0,
        left.nterms(),
        right.nterms(),
        product.nterms(),
    );
}

fn benchmark_finite_field_suite<F: EuclideanDomain + ConvertToRing>(label: &str, ring: &F) {
    compare_finite_field_dense_univariate(
        &format!("{label} dense univariate degree-4912 multiplication"),
        ring,
        4912,
        4911,
        3,
    );
    compare_finite_field_multiplication(
        &format!("{label} dense large multiplication"),
        ring,
        "1+x+y+z",
        24,
        false,
        "1+2*x-y+3*z",
        23,
        false,
        5,
    );
    compare_finite_field_multiplication(
        &format!("{label} dense very large multiplication"),
        ring,
        "1+x+y+z",
        40,
        false,
        "1+2*x-y+3*z",
        39,
        false,
        3,
    );
    compare_finite_field_multiplication(
        &format!("{label} five-variable total-degree multiplication"),
        ring,
        "1+x1+2*x2+3*x3+4*x4+5*x5",
        13,
        true,
        "1+2*x1-3*x2+5*x3-7*x4+11*x5",
        12,
        true,
        3,
    );
    compare_finite_field_multiplication(
        &format!("{label} sparse large multiplication"),
        ring,
        "1+2*x^37*y^11+3*x^5*y^43*z^7+5*x^61*z^29+7*x^17*y^73*z^31+11*x^89*y^19*z^47+13*x^23*y^97*z^59+17*x^107*y^53*z^83",
        7,
        false,
        "1+2*x^37*y^11+3*x^5*y^43*z^7+5*x^61*z^29+7*x^17*y^73*z^31+11*x^89*y^19*z^47+13*x^23*y^97*z^59+17*x^107*y^53*z^83",
        7,
        false,
        1,
    );
    compare_finite_field_multiplication(
        &format!("{label} seven-variable power-minus-one multiplication"),
        ring,
        "1+3*x1+5*x2+7*x3+9*x4+11*x5+13*x6+15*x7",
        7,
        true,
        "1+3*x1+5*x2+7*x3+9*x4+11*x5+13*x6+15*x7",
        7,
        true,
        1,
    );
}

fn main() {
    println!("Symbolica {}", env!("CARGO_PKG_VERSION"));
    compare_multiplication(
        "dense small multiplication",
        "1+x+y+z",
        12,
        "1+2*x-y+3*z",
        11,
        25,
    );
    compare_multiplication(
        "dense high multiplication",
        "1000000000039+x+y+z",
        12,
        "1000000000187+2*x-y+3*z",
        11,
        10,
    );
    compare_multiplication(
        "dense large multiplication",
        "1+x+y+z",
        24,
        "1+2*x-y+3*z",
        23,
        7,
    );
    compare_multiplication(
        "dense very large multiplication",
        "1+x+y+z",
        40,
        "1+2*x-y+3*z",
        39,
        3,
    );
    compare_multiplication(
        "dense high large multiplication",
        "1000000000039+x+y+z",
        20,
        "1000000000187+2*x-y+3*z",
        19,
        3,
    );
    compare_exact_division("dense exact division", "1+x+y+z", 12, "1+2*x-y+3*z", 7, 7);
    compare_exact_division(
        "dense large exact division",
        "1+x+y+z",
        20,
        "1+2*x-y+3*z",
        12,
        5,
    );
    compare_exact_division(
        "high-height exact division",
        "1000000000039+x+y+z",
        12,
        "1000000000187+2*x-y+3*z",
        10,
        5,
    );
    compare_multiplication(
        "sparse separated multiplication",
        "1+2*x^37*y^11+3*x^5*y^43*z^7+5*x^61*z^29+7*x^17*y^73*z^31+11*x^89*y^19*z^47",
        7,
        "1+2*x^37*y^11+3*x^5*y^43*z^7+5*x^61*z^29+7*x^17*y^73*z^31+11*x^89*y^19*z^47",
        7,
        3,
    );
    compare_multiplication(
        "sparse large multiplication",
        "1+2*x^37*y^11+3*x^5*y^43*z^7+5*x^61*z^29+7*x^17*y^73*z^31+11*x^89*y^19*z^47+13*x^23*y^97*z^59+17*x^107*y^53*z^83",
        7,
        "1+2*x^37*y^11+3*x^5*y^43*z^7+5*x^61*z^29+7*x^17*y^73*z^31+11*x^89*y^19*z^47+13*x^23*y^97*z^59+17*x^107*y^53*z^83",
        7,
        1,
    );
    compare_multiplication_with_options(
        "seven-variable power-minus-one multiplication",
        "1+3*x1+5*x2+7*x3+9*x4+11*x5+13*x6+15*x7",
        7,
        true,
        "1+3*x1+5*x2+7*x3+9*x4+11*x5+13*x6+15*x7",
        7,
        true,
        1,
    );
    benchmark_finite_field_suite("GF(17)", &Zp::new(17));
    benchmark_finite_field_suite(
        "GF(18446744073709551557)",
        &Zp64::new(18_446_744_073_709_551_557),
    );
    compare(
        "dense outer degrees 7/6",
        "1+(2+y^2+z^3)*x+(3+y^3+z^2)*x^2+(4+y+z)*x^3+(5+y^2+z^3)*x^4+(6+y^3+z^2)*x^5+(7+y+z)*x^6+(8+y^2+z^3)*x^7",
        "1+(3+y^3-z^2)*x+(5+y^2-z^3)*x^2+(7+y-z)*x^3+(9+y^3-z^2)*x^4+(11+y^2-z^3)*x^5+(13+y-z)*x^6",
        7,
    );
    compare(
        "lacunary outer degrees 18/11",
        "(y+1)*x^18+(z+2)*x^13+(y*z+3)*x^7+(y^2-z)*x^2+1",
        "(z+1)*x^11+(y-2)*x^8+(y+z)*x^3+2",
        7,
    );
    compare(
        "nonunit leading degrees 9/7",
        "(y+1)*x^9+(z^2+2)*x^8+(y*z+1)*x^5+(y^2+z)*x^2+3",
        "(z+1)*x^7+(y^2-1)*x^6+(y+z+1)*x^3+z*x+2",
        7,
    );
    compare(
        "large high-height degrees 14/10",
        "(1000000000039+y^3+z^2)*x^14+(1000000000061+y*z^2-z^3)*x^10+(1000000000091+y*z+y)*x^6+(1000000000163+y^2*z^2+z)*x^2+1000000000169+y+z",
        "(1000000000187+z^2+y)*x^10+(1000000000193+y^3-z^2)*x^7+(1000000000223+y^2*z-z^3)*x^4+(1000000000241+y^2+z^2)*x+1000000000271-z",
        3,
    );
}
