use std::{hint::black_box, time::Instant};

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

fn compare(name: &str, a: &str, b: &str, iterations: usize) {
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

    let expected = a.resultant_ducos(&b);
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
    let iterations = std::env::var("MULTIPLICATION_BENCH_SAMPLES")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(iterations);
    let mut polys = [
        parse!(a).to_polynomial::<_, u16>(&Z, None).pow(a_power),
        parse!(b).to_polynomial::<_, u16>(&Z, None).pow(b_power),
    ];
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
