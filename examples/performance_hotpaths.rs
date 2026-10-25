//! Targeted normalization, expansion, and replacement benchmarks.
//! Run optimized binaries from both revisions on the same idle CPU.
//! HOTPATH_CASE selects one case; HOTPATH_MS and HOTPATH_SAMPLES control sampling.
//! HOTPATH_ITERATIONS instead executes fixed work for hardware counters.
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use symbolica::prelude::*;

fn selected(name: &str) -> bool {
    std::env::var("HOTPATH_CASE").is_ok_and(|case| case == name)
        || std::env::var_os("HOTPATH_CASE").is_none()
}

fn measure(name: &str, bytes: usize, mut operation: impl FnMut() -> usize) {
    let fixed = std::env::var("HOTPATH_ITERATIONS")
        .ok()
        .map(|s| s.parse::<u64>().unwrap());
    let duration = Duration::from_millis(
        std::env::var("HOTPATH_MS")
            .map(|s| s.parse().unwrap())
            .unwrap_or(100),
    );
    let count = std::env::var("HOTPATH_SAMPLES")
        .map(|s| s.parse().unwrap())
        .unwrap_or(7);
    assert!(count > 0 && fixed != Some(0));
    black_box(operation());
    let mut samples = Vec::new();
    for _ in 0..count {
        let start = Instant::now();
        let mut iterations = 0;
        loop {
            black_box(operation());
            iterations += 1;
            if fixed.map_or_else(|| start.elapsed() >= duration, |n| iterations >= n) {
                break;
            }
        }
        samples.push(start.elapsed().as_secs_f64() * 1e9 / iterations as f64);
    }
    samples.sort_by(f64::total_cmp);
    println!(
        "{name},{bytes},{:.3},{:.3},{:.3}",
        samples[count / 2],
        samples[0],
        samples[count - 1]
    );
}

fn main() {
    println!("case,bytes,median_ns,min_ns,max_ns");
    let sym = symbol!("perf_hot_sym"; Symmetric);
    let anti = symbol!("perf_hot_anti"; Antisymmetric);
    let ordinary = symbol!("perf_hot_plain");
    for (label, head, counts) in [
        ("symmetric", sym, &[8, 32, 128][..]),
        ("antisymmetric", anti, &[8, 64, 256, 1024][..]),
        ("plain", ordinary, &[128][..]),
    ] {
        for &n in counts {
            let name = format!("normalize/{label}-{n}");
            if !selected(&name) {
                continue;
            }
            let args: Vec<_> = (0..n)
                .map(|i| {
                    if head == sym {
                        parse!(&format!("g({i},(x+y+z)^3,h(x,y,z,w),x*y*z*w)"))
                    } else {
                        Atom::num(i)
                    }
                })
                .collect();
            // Odd multiplier is a permutation for these power-of-two sizes.
            let builder = FunctionBuilder::new(head)
                .add_args((0..n).map(|i| &args[((i * 73 + 19) % n) as usize]));
            let expected = builder.clone().finish();
            // Cloning a prepared builder isolates normalization from argument construction.
            measure(&name, expected.as_view().get_byte_size(), || {
                black_box(&builder)
                    .clone()
                    .finish()
                    .as_view()
                    .get_byte_size()
            });
            assert_eq!(builder.finish(), expected);
            assert!(!expected.as_view().needs_normalization());
        }
    }

    for (name, input, partial) in [
        ("polynomial-7", "(1+x+y+z)^7", false),
        ("polynomial-12", "(1+x+y+z)^12", false),
        ("coefficients", "(1+2*x+3*y+4*z)^10", false),
        ("functions", "(1+f(x)+g(y)+h(z))^5", false),
        ("product", "(x+1)*(y+2)*(z+3)*(w+4)", false),
        ("wide-power", "(x+y)^32", false),
        ("partial", "(x+y+z)^12", true),
    ] {
        let name = format!("expand/{name}");
        if !selected(&name) {
            continue;
        }
        let input = parse!(input);
        let x = parse!("x");
        let expand = || {
            if partial {
                input.expand_in(&x)
            } else {
                input.expand()
            }
        };
        let expected = expand();
        assert_eq!(
            expected.expand_via_poly::<u16, Atom>(None),
            input.expand_via_poly::<u16, Atom>(None)
        );
        measure(&name, expected.as_view().get_byte_size(), || {
            black_box(expand()).as_view().get_byte_size()
        });
        assert_eq!(expand(), expected);
    }

    let root = FunctionBuilder::new(symbol!("perf_hot_root"))
        .add_args((0..256).map(|i| parse!(&format!("perf_hot_f({i},{},{})", i + 1, i + 2))))
        .finish();
    for (name, lhs, rhs) in [
        (
            "single-wildcards",
            "perf_hot_f(x_,y_,z_)",
            "perf_hot_g(x_,y_,z_)",
        ),
        (
            "sequence-wildcards",
            "perf_hot_f(x__,y__)",
            "perf_hot_g(x__,y__)",
        ),
        (
            "no-match",
            "perf_hot_absent(x_,y_,z_)",
            "perf_hot_g(x_,y_,z_)",
        ),
    ] {
        if !selected(&format!("replace/{name}")) {
            continue;
        }
        let pattern = parse!(lhs).to_pattern();
        let replacement = parse!(rhs).to_pattern();
        let expected = root.replace(&pattern).with(&replacement);
        if name == "no-match" {
            assert_eq!(expected, root);
        } else {
            let check = FunctionBuilder::new(symbol!("perf_hot_root"))
                .add_args((0..256).map(|i| parse!(&format!("perf_hot_g({i},{},{})", i + 1, i + 2))))
                .finish();
            assert_eq!(expected, check);
        }
        measure(
            &format!("replace/{name}"),
            expected.as_view().get_byte_size(),
            || {
                black_box(&root)
                    .replace(black_box(&pattern))
                    .with(black_box(&replacement))
                    .as_view()
                    .get_byte_size()
            },
        );
    }
}
