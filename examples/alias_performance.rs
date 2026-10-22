//! Small, dependency-free regression benchmark for the alias-free arithmetic path.
//!
//! This example deliberately uses APIs shared with the commit before global aliases,
//! so the same source can be copied into both checkouts. Build each checkout with
//! `cargo build --release --example alias_performance`, then run the binaries
//! sequentially, outside Cargo:
//!
//! ```text
//! SYMBOLICA_HIDE_BANNER=1 target/release/examples/alias_performance \
//!     --samples 7 --target-ms 100 > timings.csv
//! ```
//!
//! `--filter expand` selects workload names containing `expand`. Setup, license
//! initialization, calibration, and output-size checks are outside the samples.
//! Each sample reports a fixed number of complete operations, including dropping
//! their results. Compare median `ns_per_iteration` for each workload. This does
//! not install any aliases; an unrelated alias must not affect these expressions.

use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use symbolica::{
    atom::{Atom, AtomCore, FunctionBuilder},
    parse, symbol,
};

fn elapsed<F: FnMut() -> Atom>(iterations: usize, operation: &mut F) -> Duration {
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = black_box(operation());
    }
    start.elapsed()
}

fn measure<F: FnMut() -> Atom>(
    name: &str,
    samples: usize,
    target: Duration,
    filter: &str,
    mut operation: F,
) {
    if !name.contains(filter) {
        return;
    }

    // Exercise initialization and verify a materialized result before calibration.
    let result_bytes = black_box(operation()).as_view().get_byte_size();
    let mut iterations = 1usize;
    let calibration = loop {
        let duration = elapsed(iterations, &mut operation);
        if duration >= Duration::from_millis(10) {
            break duration;
        }
        iterations = iterations.checked_mul(2).unwrap();
    };
    let iterations =
        ((iterations as f64 * target.as_secs_f64() / calibration.as_secs_f64()).round() as usize)
            .max(1);

    for sample in 0..samples {
        let duration = elapsed(iterations, &mut operation);
        println!(
            "{name},{sample},{iterations},{},{:.3},{result_bytes}",
            duration.as_nanos(),
            duration.as_nanos() as f64 / iterations as f64,
        );
    }
}

fn main() {
    let mut samples = 7usize;
    let mut target_ms = 100u64;
    let mut filter = String::new();
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--samples" => {
                samples = arguments
                    .next()
                    .expect("missing sample count")
                    .parse()
                    .unwrap();
            }
            "--target-ms" => {
                target_ms = arguments
                    .next()
                    .expect("missing sample duration")
                    .parse()
                    .unwrap();
            }
            "--filter" => filter = arguments.next().expect("missing workload filter"),
            "--help" | "-h" => {
                eprintln!("alias_performance [--samples 7] [--target-ms 100] [--filter substring]");
                return;
            }
            _ => panic!("unknown argument: {argument}"),
        }
    }
    assert!(samples > 0 && target_ms > 0);

    let target = Duration::from_millis(target_ms);
    let (x_symbol, y_symbol, z_symbol, f) = symbol!("x", "y", "z", "f");
    let (x, y, z) = (
        Atom::var(x_symbol),
        Atom::var(y_symbol),
        Atom::var(z_symbol),
    );
    let terms: Vec<_> = (0..32)
        .map(|i| (x.pow(i % 4 + 1) * y.pow(i % 5 + 1)) * (i + 1))
        .collect();
    let power = parse!("(1+x+y+z+w)^7");
    let product = parse!("(1+x+y)^3*(2+x+z)^3*(3+y+w)^2");
    let factors = [
        parse!("3*x^2*y^3*f(x+y)"),
        parse!("2/7*x^3*y^2*f(x+y)^2"),
        parse!("7/3*x^-2*y^-1*f(x+y)^-1"),
    ];
    let derivative_input = parse!("(x+y+z)^5*sin(x*y)+exp(x+z)*cos(x^2)+log(1+x^2)/(1+x*y)");
    let replacement_input = Atom::add_many((0..24).map(|i| {
        FunctionBuilder::new(f)
            .add_arg(&x + i)
            .add_arg(&y + i)
            .finish()
            * (i + 1)
    }));
    let replacement_pattern = parse!("f(a_,b_)").to_pattern();
    let replacement_rhs = parse!("a_^2+b_^2").to_pattern();

    println!("workload,sample,iterations,total_ns,ns_per_iteration,result_bytes");

    measure("construct", samples, target, &filter, || {
        (black_box(&x) + black_box(&y) + 1).pow(2) * (black_box(&z) + 2)
            + black_box(&x) * black_box(&y)
    });
    measure("add_pair", samples, target, &filter, || {
        black_box(&terms[0]) + black_box(&terms[1])
    });
    measure("add_chain", samples, target, &filter, || {
        let mut result = Atom::new();
        for term in black_box(&terms) {
            result += term;
        }
        result
    });
    measure("add_many", samples, target, &filter, || {
        Atom::add_many(black_box(&terms).iter())
    });
    measure("normalize_product", samples, target, &filter, || {
        Atom::mul_many(black_box(&factors).iter())
    });
    measure("expand_power", samples, target, &filter, || {
        black_box(&power).expand()
    });
    measure("expand_product", samples, target, &filter, || {
        black_box(&product).expand()
    });
    measure("differentiate", samples, target, &filter, || {
        black_box(&derivative_input).derivative(black_box(x_symbol))
    });
    measure("replace", samples, target, &filter, || {
        black_box(&replacement_input)
            .replace(black_box(&replacement_pattern))
            .with(black_box(&replacement_rhs))
    });
}
