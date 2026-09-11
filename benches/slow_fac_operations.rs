//! Replay captured integer operands without timing parsing, validation or result destruction.
//! See slow_fac_operations.md for extraction, provenance and baseline instructions.

#[cfg(feature = "flint_benchmarks")]
#[allow(dead_code)]
#[path = "support/flint.rs"]
mod flint;
#[cfg(feature = "flint_benchmarks")]
#[allow(dead_code)]
#[path = "support/paired.rs"]
mod paired;

use smartstring::{LazyCompact, SmartString};
use std::fmt::Write as _;
use std::time::Instant;
use std::{env, fs, hint::black_box, io::Read, path::Path, sync::Arc};
use symbolica::prelude::*;

type Poly = MultivariatePolynomial<IntegerRing, u16>;

fn read_operand(path: &Path) -> String {
    let mut text = String::new();
    brotli::Decompressor::new(fs::File::open(path).unwrap(), 65536)
        .read_to_string(&mut text)
        .unwrap();
    text
}

fn parse(
    text: &str,
    names: &[SmartString<LazyCompact>],
    variables: &Arc<Vec<PolyVariable>>,
) -> Poly {
    let (rest, poly) = Token::parse_polynomial(text.trim().as_bytes(), variables, names, &Z);
    assert!(
        rest.is_empty(),
        "unparsed polynomial suffix: {:?}",
        &rest[..rest.len().min(50)]
    );
    poly
}

fn operation(kind: &str, a: &Poly, b: &Poly) -> Option<Poly> {
    match kind {
        "mul" => Some(a * b),
        "try_div" => a.try_div(b),
        _ => panic!("unsupported captured operation {kind}"),
    }
}

fn kinematic_degree(poly: &Poly) -> u16 {
    assert_eq!(poly.nvars(), 5);
    let degree = poly.exponents(0)[1..].iter().sum();
    assert!(
        poly.into_iter()
            .all(|t| t.exponents[1..].iter().sum::<u16>() == degree)
    );
    degree
}

fn expanded(poly: &Poly) -> String {
    let mut text = String::new();
    for (i, term) in poly.into_iter().enumerate() {
        if i != 0 && !term.coefficient.is_negative() {
            text.push('+');
        }
        write!(text, "{}", term.coefficient).unwrap();
        for (v, e) in term.exponents.iter().enumerate() {
            if *e != 0 {
                write!(text, "*x{v}^{e}").unwrap();
            }
        }
    }
    text
}

fn restore_m(poly: &Poly, degree: u16) -> Poly {
    let mut coefficients = Vec::with_capacity(poly.nterms());
    let mut exponents = Vec::with_capacity(5 * poly.nterms());
    for term in poly {
        let mut e = term.exponents.to_vec();
        assert_eq!(e[1], 0);
        e[1] = degree.checked_sub(e[2..].iter().sum()).unwrap();
        coefficients.push(term.coefficient.clone());
        exponents.extend(e);
    }
    Poly::from_coefficient_list(coefficients, exponents, poly.get_vars(), &Z)
}

fn measure_symbolica(name: &str, kind: &str, a: &Poly, b: &Poly, samples: usize) {
    drop(black_box(operation(kind, a, b)));
    let mut timings = Vec::with_capacity(samples);
    for _ in 0..samples {
        let start = Instant::now();
        let result = operation(kind, black_box(a), black_box(b));
        timings.push(start.elapsed().as_secs_f64() * 1000.0);
        drop(black_box(result));
    }
    timings.sort_by(f64::total_cmp);
    let median = (timings[(samples - 1) / 2] + timings[samples / 2]) / 2.0;
    println!("{name},{samples},{:.6},{median:.6}", timings[0]);
}

#[cfg(feature = "flint_benchmarks")]
fn parse_flint(
    poly: &flint::FmpzMPoly<'_>,
    names: &[SmartString<LazyCompact>],
    variables: &Arc<Vec<PolyVariable>>,
) -> Poly {
    // FLINT prints descending lex order. Reverse integer terms before using the
    // ascending-order fast parser, avoiding quadratic insertion of huge inputs.
    let text = poly.to_pretty_string().unwrap();
    let starts: Vec<_> = std::iter::once(0)
        .chain(
            text.char_indices()
                .filter_map(|(i, c)| (i > 0 && (c == '+' || c == '-')).then_some(i)),
        )
        .chain(std::iter::once(text.len()))
        .collect();
    let mut ascending = String::with_capacity(text.len() + 1);
    for bounds in starts.windows(2).rev() {
        let term = &text[bounds[0]..bounds[1]];
        if !ascending.is_empty() && !term.starts_with(['+', '-']) {
            ascending.push('+');
        }
        ascending.push_str(term);
    }
    parse(&ascending, names, variables)
}

fn main() {
    // 2.2's restricted-mode initialization installs its own one-thread global
    // pool. Initialize the library first and accept that existing pool.
    let _ = symbol!("slow_fac_benchmark_init");
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global();
    assert_eq!(rayon::current_num_threads(), 1);
    #[cfg(feature = "flint_benchmarks")]
    flint::initialize_single_thread();
    eprintln!(
        "Symbolica {} {}; one thread; parsing and validation excluded",
        env!("CARGO_PKG_VERSION"),
        env!("SYMBOLICA_VERSION")
    );
    #[cfg(feature = "flint_benchmarks")]
    eprintln!("FLINT {}", flint::version());

    let manifest = env::var("SYMBOLICA_SLOW_FAC_MANIFEST").unwrap_or_else(|_| {
        format!(
            "{}/benches/fixtures/slow_fac/cases.tsv",
            env!("CARGO_MANIFEST_DIR")
        )
    });
    let manifest = Path::new(&manifest);
    let directory = manifest.parent().unwrap();
    let filter = env::var("SYMBOLICA_FLINT_BENCH_FILTER").unwrap_or_default();
    let dehomogenize = env::var_os("SYMBOLICA_SLOW_FAC_DEHOMOGENIZE").is_some();
    let samples: usize = env::var("SYMBOLICA_FLINT_BENCH_SAMPLES")
        .unwrap_or_else(|_| "5".into())
        .parse()
        .unwrap();
    assert!(samples > 0);
    #[cfg(feature = "flint_benchmarks")]
    let config = paired::PairedConfig::from_env(samples);
    #[cfg(feature = "flint_benchmarks")]
    let flint_johnson = match env::var("SYMBOLICA_SLOW_FAC_FLINT_MUL")
        .as_deref()
        .unwrap_or("auto")
    {
        "auto" => false,
        "johnson" => true,
        _ => panic!("SYMBOLICA_SLOW_FAC_FLINT_MUL must be auto or johnson"),
    };
    #[cfg(feature = "flint_benchmarks")]
    eprintln!(
        "FLINT multiplication: {}",
        if flint_johnson {
            "Johnson heap"
        } else {
            "auto"
        }
    );
    let symbolica_only = !cfg!(feature = "flint_benchmarks")
        || env::var_os("SYMBOLICA_SLOW_FAC_SYMBOLICA_ONLY").is_some();
    if symbolica_only && env::var_os("SYMBOLICA_SLOW_FAC_DESCRIBE").is_none() {
        println!("case,samples,symbolica_min_ms,symbolica_median_ms");
    }

    let mut count = 0;
    for row in fs::read_to_string(manifest).unwrap().lines() {
        if row.is_empty() || row.starts_with('#') {
            continue;
        }
        let fields: Vec<_> = row.split('\t').collect();
        assert_eq!(
            fields.len(),
            8,
            "expected name, operation, nvars, left, right, left_terms, right_terms, result_terms (or none)"
        );
        let [
            name,
            kind,
            nvars,
            left,
            right,
            left_terms,
            right_terms,
            result_terms,
        ] = fields[..]
        else {
            unreachable!()
        };
        if !name.contains(&filter) {
            continue;
        }
        if dehomogenize && kind != "mul" {
            continue;
        }
        count += 1;
        let nvars: usize = nvars.parse().unwrap();
        let names: Vec<SmartString<LazyCompact>> =
            (0..nvars).map(|i| format!("x{i}").into()).collect();
        let variables = Arc::new(
            names
                .iter()
                .map(|name| symbol!(name.as_str()).into())
                .collect(),
        );
        let mut left_text = read_operand(&directory.join(left));
        let mut right_text = read_operand(&directory.join(right));
        let mut a = parse(&left_text, &names, &variables);
        let mut b = parse(&right_text, &names, &variables);
        assert_eq!(a.nterms(), left_terms.parse::<usize>().unwrap());
        assert_eq!(b.nterms(), right_terms.parse::<usize>().unwrap());
        let mut result = operation(kind, &a, &b);
        if env::var_os("SYMBOLICA_SLOW_FAC_DESCRIBE").is_some() {
            let terms = result
                .as_ref()
                .map_or_else(|| "none".to_string(), |p| p.nterms().to_string());
            println!(
                "{name}\t{kind}\t{nvars}\t{left}\t{right}\t{left_terms}\t{right_terms}\t{terms}"
            );
            continue;
        }
        let expected_terms = if result_terms == "none" {
            None
        } else {
            Some(result_terms.parse::<usize>().unwrap())
        };
        assert_eq!(result.as_ref().map(Poly::nterms), expected_terms, "{name}");
        if kind == "try_div"
            && let Some(q) = &result
        {
            assert_eq!(q * &b, a);
        }
        let name = if dehomogenize {
            let degree = kinematic_degree(&a) + kinematic_degree(&b);
            a = a.replace(1, &Z.one());
            b = b.replace(1, &Z.one());
            let product = &a * &b;
            assert_eq!(restore_m(&product, degree), *result.as_ref().unwrap());
            result = Some(product);
            left_text = expanded(&a);
            right_text = expanded(&b);
            format!("{name}_m1")
        } else {
            name.to_owned()
        };
        let name = name.as_str();
        let degrees = |p: &Poly| (0..nvars).map(|i| p.degree(i)).collect::<Vec<_>>();
        eprintln!(
            "{name}: {} x {} terms; degrees {:?} / {:?}; result {:?}",
            a.nterms(),
            b.nterms(),
            degrees(&a),
            degrees(&b),
            expected_terms
        );

        if symbolica_only {
            drop(result);
            measure_symbolica(name, kind, &a, &b, samples);
            continue;
        }

        #[cfg(feature = "flint_benchmarks")]
        {
            let context = flint::FmpzMPolyContext::new(&names).unwrap();
            let fa = context.parse(left_text.trim()).unwrap();
            let fb = context.parse(right_text.trim()).unwrap();
            assert_eq!(fa.len(), a.nterms());
            assert_eq!(fb.len(), b.nterms());
            // Verify exact input equality as well as the operation result, outside timing.
            assert_eq!(parse_flint(&fa, &names, &variables), a);
            assert_eq!(parse_flint(&fb, &names, &variables), b);
            let flint_op = || match kind {
                "mul" if flint_johnson => Some(fa.mul_johnson(&fb)),
                "mul" => Some(fa.mul(&fb)),
                "try_div" => fa.exact_div(&fb).ok(),
                _ => unreachable!(),
            };
            let fresult = flint_op();
            assert_eq!(
                fresult.as_ref().map(|p| parse_flint(p, &names, &variables)),
                result,
                "{name}: FLINT disagrees"
            );
            drop(fresult);
            drop(result);
            paired::run_paired(
                &config,
                name,
                || operation(kind, black_box(&a), black_box(&b)),
                flint_op,
            );
        }
    }
    assert!(count > 0, "no matching benchmark cases");
}
