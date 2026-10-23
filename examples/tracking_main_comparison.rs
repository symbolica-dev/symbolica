//! Identical public-API workload for main, binary-storage main and error tracking.
//! Run each executable on the same pinned CPU; timings use thread CPU time.
use std::hint::black_box;
use symbolica::{
    atom::{Atom, AtomView},
    coefficient::{Coefficient, CoefficientView},
    domains::float::{Complex, Float},
};

#[path = "../lib/numerica/examples/support/clock.rs"]
mod clock;

fn view(atom: &Atom) -> CoefficientView<'_> {
    let AtomView::Num(n) = atom.as_view() else {
        unreachable!()
    };
    n.get_coeff_view()
}

fn source(bits: u32, complex: bool, numerator: u32) -> Coefficient {
    Coefficient::from(Complex::new(
        Float::with_val(bits, numerator) / Float::with_val(bits, 7),
        if complex {
            Float::with_val(bits, numerator + 1) / Float::with_val(bits, 11)
        } else {
            Float::new(bits)
        },
    ))
}

fn timing(iterations: usize, mut f: impl FnMut()) -> f64 {
    for _ in 0..100 {
        f();
    }
    let clock = clock::Clock::from_env();
    let mut ns = [0.; 7];
    for sample in &mut ns {
        let start = clock.start();
        for _ in 0..iterations {
            f();
        }
        *sample = start.elapsed().as_nanos() as f64 / iterations as f64;
    }
    ns.sort_by(f64::total_cmp);
    ns[3]
}

fn rss() -> usize {
    std::fs::read_to_string("/proc/self/status")
        .unwrap()
        .lines()
        .find_map(|line| {
            line.strip_prefix("VmRSS:").map(|v| {
                v.split_whitespace()
                    .next()
                    .unwrap()
                    .parse::<usize>()
                    .unwrap()
                    * 1024
            })
        })
        .unwrap()
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    let label = args.get(1).map(String::as_str).unwrap_or("current");
    if args.get(2).is_some_and(|s| s == "memory") {
        let bits: u32 = args[3].parse().unwrap();
        let complex = args[4] == "complex";
        let count: usize = args.get(5).map(|s| s.parse().unwrap()).unwrap_or(200_000);
        let c = source(bits, complex, 1);
        let sample = Atom::num(c.clone());
        let packed = sample.clone().into_raw().len();
        let mut retained = Vec::with_capacity(count);
        let before = rss();
        for _ in 0..count {
            retained.push(Atom::num(c.clone()));
        }
        let bytes = rss().saturating_sub(before);
        println!(
            "{label},{bits},{},{count},{packed},{bytes}",
            if complex { "complex" } else { "real" }
        );
        black_box(retained);
        return;
    }
    let iterations = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(20_000);
    eprintln!(
        "clock: {}; median of seven samples",
        clock::Clock::from_env().label()
    );
    println!("revision,bits,kind,operation,median_ns,packed_bytes");
    for bits in [53, 256, 4096] {
        for complex in [false, true] {
            let kind = if complex { "complex" } else { "real" };
            let left = source(bits, complex, 1);
            let right = source(bits, complex, 3);
            let a = Atom::num(left.clone());
            let b = Atom::num(right.clone());
            let packed = a.clone().into_raw().len();
            let bench = |operation, f: &mut dyn FnMut()| {
                let ns = timing(iterations, f);
                println!("{label},{bits},{kind},{operation},{ns:.3},{packed}");
            };
            bench("construct", &mut || {
                black_box(Atom::num(black_box(left.clone())));
            });
            bench("decode", &mut || {
                black_box(view(black_box(&a)).to_owned());
            });
            bench("coefficient_add", &mut || {
                black_box(black_box(left.clone()) + black_box(right.clone()));
            });
            bench("coefficient_mul", &mut || {
                black_box(black_box(left.clone()) * black_box(right.clone()));
            });
            bench("atom_div", &mut || {
                black_box(black_box(&a) / black_box(&b));
            });
            bench("atom_add", &mut || {
                black_box(black_box(&a) + black_box(&b));
            });
            bench("atom_mul", &mut || {
                black_box(black_box(&a) * black_box(&b));
            });
            bench("atom_cancel", &mut || {
                black_box(black_box(&a) - black_box(&a));
            });
        }
    }
}
