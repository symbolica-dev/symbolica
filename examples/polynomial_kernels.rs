//! Replay integer term-list operands from the polynomial benchmark fixtures.
//! Usage: polynomial_kernels INPUT factor|gcd|resultant integer|rational [REPS] [DUMP]
//! Resultant mode measures the native univariate integer resultant.
//! Use resultant-brown, resultant-ducos, or resultant-primitive for explicit algorithms.
//! Parsing and output validation are excluded from the timing.
//! For whole-process profiling, set SYMBOLICA_PROFILE_SKIP_VALIDATION=1.
//! SYMBOLICA_PROFILE_SPANS=1 prints durations of tracing spans above 1 ms.
//! Debug/trace spans require a build without the tracing_max_level_info feature.
//! SYMBOLICA_PROFILE_HU=off|force overrides the experimental GCD dispatch.
use std::{fs, io::Write, str::FromStr, sync::Arc, time::Instant};
use symbolica::{poly::factor::Factorize, prelude::*};

struct ProfileSpans;

impl<S> tracing_subscriber::Layer<S> for ProfileSpans
where
    S: tracing::Subscriber + for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    fn on_new_span(
        &self,
        _: &tracing::span::Attributes<'_>,
        id: &tracing::Id,
        ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        ctx.span(id)
            .unwrap()
            .extensions_mut()
            .insert(Instant::now());
    }

    fn on_close(&self, id: tracing::Id, ctx: tracing_subscriber::layer::Context<'_, S>) {
        let span = ctx.span(&id).unwrap();
        let elapsed = span.extensions().get::<Instant>().unwrap().elapsed();
        if elapsed.as_millis() > 0 {
            eprintln!(
                "SPAN {} {:.3} ms",
                span.name(),
                elapsed.as_secs_f64() * 1000.0
            );
        }
    }
}

fn main() {
    if let Ok(mode) = std::env::var("SYMBOLICA_PROFILE_HU") {
        use std::sync::atomic::Ordering::Relaxed;
        match mode.as_str() {
            "off" => symbolica::GLOBAL_SETTINGS
                .use_hu_monagan_poly_gcd
                .store(false, Relaxed),
            "force" => symbolica::GLOBAL_SETTINGS
                .force_hu_monagan_poly_gcd
                .store(true, Relaxed),
            _ => panic!("SYMBOLICA_PROFILE_HU must be off or force"),
        }
    }
    if std::env::var_os("SYMBOLICA_PROFILE_SPANS").is_some() {
        use tracing_subscriber::prelude::*;
        tracing_subscriber::registry().with(ProfileSpans).init();
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .unwrap();
    let args: Vec<_> = std::env::args().collect();
    let input = fs::read_to_string(&args[1]).unwrap();
    let mut lines = input.lines();
    assert_eq!(lines.next().unwrap(), "polynomial-bench-op 1");
    lines.next().unwrap();
    let vars: Arc<Vec<PolyVariable>> = Arc::new(
        lines
            .next()
            .unwrap()
            .strip_prefix("variables ")
            .unwrap()
            .split_whitespace()
            .map(|n| PolyVariable::Symbol(symbol!(n)))
            .collect(),
    );
    let variable: i32 = lines
        .next()
        .unwrap()
        .strip_prefix("variable ")
        .unwrap()
        .parse()
        .unwrap();
    let count: usize = lines
        .next()
        .unwrap()
        .strip_prefix("polynomials ")
        .unwrap()
        .parse()
        .unwrap();
    let mut polys = Vec::new();
    for _ in 0..count {
        let n: usize = lines
            .next()
            .unwrap()
            .strip_prefix("terms ")
            .unwrap()
            .parse()
            .unwrap();
        let mut terms = Vec::new();
        for _ in 0..n {
            let mut fields = lines.next().unwrap().split_whitespace();
            let c = Integer::from_str(fields.next().unwrap()).unwrap();
            let e: Vec<u32> = fields.map(|s| s.parse().unwrap()).collect();
            terms.push((e, c));
        }
        terms.sort_by(|a, b| a.0.cmp(&b.0));
        let mut p = MultivariatePolynomial::new(&Z, None, vars.clone());
        for (e, c) in terms {
            p.append_monomial(c, &e);
        }
        polys.push(p);
    }
    if args[2] == "gcd" && polys.len() == 1 {
        let v = if variable >= 0 {
            variable as usize
        } else {
            (0..polys[0].nvars())
                .find(|&v| polys[0].degree(v) > 0)
                .unwrap()
        };
        polys.push(polys[0].derivative(v));
    }
    let rational: Vec<_> = polys
        .iter()
        .map(|p| p.map_coeff(|c| Rational::from(c), Q))
        .collect();
    let univariate = args[2].starts_with("resultant").then(|| {
        assert_eq!(args[3], "integer");
        assert_eq!(polys.len(), 2);
        assert!(variable >= 0);
        polys
            .iter()
            .map(|p| p.to_univariate_from_univariate(variable as usize))
            .collect::<Vec<_>>()
    });
    let reps: usize = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(3);
    let mut times = Vec::new();
    let mut result = Vec::new();
    for _ in 0..reps {
        let start = Instant::now();
        if let Some(univariate) = &univariate {
            let value = match args[2].as_str() {
                "resultant-brown" => univariate[0].resultant_brown(&univariate[1]),
                "resultant-ducos" => univariate[0].resultant_ducos(&univariate[1]),
                "resultant-primitive" => univariate[0].resultant_primitive(&univariate[1]),
                _ => univariate[0].resultant(&univariate[1]),
            };
            times.push(start.elapsed().as_secs_f64());
            result = vec![(rational[0].constant(Rational::from(value)), 1)];
        } else if args[3] == "integer" {
            let r = if args[2] == "factor" {
                polys[0].factor()
            } else {
                vec![(polys[0].gcd(&polys[1]), 1)]
            };
            times.push(start.elapsed().as_secs_f64());
            result = r
                .into_iter()
                .map(|(p, m)| (p.map_coeff(|c| Rational::from(c), Q), m))
                .collect();
        } else {
            result = if args[2] == "factor" {
                rational[0].factor()
            } else {
                vec![(rational[0].gcd(&rational[1]), 1)]
            };
            times.push(start.elapsed().as_secs_f64());
        }
        std::hint::black_box(&result);
    }
    if std::env::var_os("SYMBOLICA_PROFILE_SKIP_VALIDATION").is_none() {
        if let Some(univariate) = &univariate {
            let mut reversed = univariate[1].resultant(&univariate[0]);
            if univariate[0].degree() % 2 == 1 && univariate[1].degree() % 2 == 1 {
                reversed = -reversed;
            }
            assert_eq!(result[0].0.get_constant(), Rational::from(reversed));
        } else if args[2] == "factor" {
            let product = result
                .iter()
                .fold(rational[0].one(), |a, (p, m)| a * &p.pow(*m));
            assert_eq!(product, rational[0]);
        } else {
            assert!(rational.iter().all(|p| p.try_div(&result[0].0).is_some()));
        }
    }
    if let Some(path) = args.get(5) {
        let mut out = fs::File::create(path).unwrap();
        for (p, m) in &result {
            writeln!(out, "poly {m}").unwrap();
            for (c, e) in p.coefficients.iter().zip(p.exponents_iter()) {
                write!(out, "{c}").unwrap();
                for x in e {
                    write!(out, " {x}").unwrap();
                }
                writeln!(out).unwrap();
            }
        }
    }
    times.sort_by(f64::total_cmp);
    println!(
        "{} {} {} min={:.9} median={:.9} factors={}",
        args[1],
        args[2],
        args[3],
        times[0],
        times[times.len() / 2],
        result.len()
    );
}
