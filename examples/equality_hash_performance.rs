//! Equality and hashing benchmarks that use only the ordinary Atom API.
//!
//! Build the exact same source on main and the global-alias branch:
//! `cargo build --release --example equality_hash_performance`.
//! Run binaries sequentially, preferably pinned to one CPU:
//! `SYMBOLICA_HIDE_BANNER=1 target/release/examples/equality_hash_performance
//! --samples 7 --target-ms 100`.
//!
//! Setup, assertions, and calibration are outside the measured samples. Hashing
//! uses fixed AHash seeds and includes construction of the per-operation hasher.
//! Equal expressions own separate byte buffers. The final CSV column is the
//! scalar result size (bool or u64), matching the existing timing CSV schema.

use std::{
    fmt::Debug,
    hash::{BuildHasher, Hash, Hasher},
    hint::black_box,
    time::{Duration, Instant},
};

use ahash::RandomState;
use symbolica::parse;

pub(crate) struct Benchmark {
    samples: usize,
    target: Duration,
    filter: String,
}

impl Benchmark {
    pub(crate) fn from_args() -> Self {
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
                        .unwrap()
                }
                "--target-ms" => {
                    target_ms = arguments.next().expect("missing duration").parse().unwrap()
                }
                "--filter" => filter = arguments.next().expect("missing workload filter"),
                "--help" | "-h" => {
                    eprintln!("[--samples 7] [--target-ms 100] [--filter substring]");
                    std::process::exit(0);
                }
                _ => panic!("unknown argument: {argument}"),
            }
        }
        assert!(samples > 0 && target_ms > 0);
        println!("workload,sample,iterations,total_ns,ns_per_iteration,result_bytes");
        Self {
            samples,
            target: Duration::from_millis(target_ms),
            filter,
        }
    }

    pub(crate) fn run<T: Copy + Eq + Debug>(
        &self,
        name: &str,
        expected: T,
        mut operation: impl FnMut() -> T,
    ) {
        if !name.contains(&self.filter) {
            return;
        }
        assert_eq!(black_box(operation()), expected, "{name}");
        let mut iterations = 1usize;
        let calibration = loop {
            let duration = elapsed(iterations, &mut operation);
            if duration >= Duration::from_millis(10) {
                break duration;
            }
            iterations = iterations.checked_mul(2).unwrap();
        };
        let iterations = ((iterations as f64 * self.target.as_secs_f64()
            / calibration.as_secs_f64())
        .round() as usize)
            .max(1);
        for sample in 0..self.samples {
            let duration = elapsed(iterations, &mut operation);
            println!(
                "{name},{sample},{iterations},{},{:.3},{}",
                duration.as_nanos(),
                duration.as_nanos() as f64 / iterations as f64,
                std::mem::size_of::<T>(),
            );
        }
    }
}

fn elapsed<T>(iterations: usize, operation: &mut impl FnMut() -> T) -> Duration {
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = black_box(operation());
    }
    start.elapsed()
}

pub(crate) fn hash_value<T: Hash + ?Sized>(value: &T, state: &RandomState) -> u64 {
    let mut hasher = state.build_hasher();
    value.hash(&mut hasher);
    hasher.finish()
}

pub(crate) fn large_arguments() -> String {
    (0..512)
        .map(|i| format!("hash_eq_term(hash_eq_x{i},hash_eq_y{i})^2"))
        .collect::<Vec<_>>()
        .join(",")
}

fn main() {
    let benchmark = Benchmark::from_args();
    let state = RandomState::with_seeds(1, 2, 3, 4);
    let small = parse!("hash_eq_f(x+y)^3*(z+2)+x*y");
    let small_equal = small.clone();
    let arguments = large_arguments();
    let large = parse!(&format!("hash_eq_large({arguments})"));
    let large_equal = large.clone();
    let unequal_early = parse!(&format!("hash_eq_other({arguments})"));
    let unequal_late = parse!(&format!(
        "hash_eq_large({})",
        arguments.replace(
            "hash_eq_term(hash_eq_x511,hash_eq_y511)",
            "hash_eq_term(hash_eq_y511,hash_eq_x511)"
        )
    ));
    for (left, right) in [(&small, &small_equal), (&large, &large_equal)] {
        assert_eq!(left, right);
        assert_ne!(
            left.as_view().get_data().as_ptr(),
            right.as_view().get_data().as_ptr()
        );
    }
    assert_ne!(large, unequal_early);
    assert_ne!(large, unequal_late);
    let prefix = large
        .as_view()
        .get_data()
        .iter()
        .zip(unequal_late.as_view().get_data())
        .take_while(|(left, right)| left == right)
        .count();
    assert_eq!(
        large.as_view().get_data().len(),
        unequal_late.as_view().get_data().len()
    );
    assert!(prefix > large.as_view().get_data().len() / 2);

    benchmark.run("hash_small", hash_value(&small, &state), || {
        hash_value(black_box(&small), black_box(&state))
    });
    benchmark.run("hash_large", hash_value(&large, &state), || {
        hash_value(black_box(&large), black_box(&state))
    });
    benchmark.run("hash_view_small", hash_value(&small, &state), || {
        hash_value(&black_box(small.as_view()), black_box(&state))
    });
    benchmark.run("hash_view_large", hash_value(&large, &state), || {
        hash_value(&black_box(large.as_view()), black_box(&state))
    });
    benchmark.run("eq_small_equal", true, || {
        black_box(&small) == black_box(&small_equal)
    });
    benchmark.run("eq_large_equal", true, || {
        black_box(&large) == black_box(&large_equal)
    });
    benchmark.run("eq_large_unequal_early", false, || {
        black_box(&large) == black_box(&unequal_early)
    });
    benchmark.run("eq_large_unequal_late", false, || {
        black_box(&large) == black_box(&unequal_late)
    });
    benchmark.run("eq_view_small_equal", true, || {
        black_box(small.as_view()) == black_box(small_equal.as_view())
    });
    benchmark.run("eq_view_large_equal", true, || {
        black_box(large.as_view()) == black_box(large_equal.as_view())
    });
    benchmark.run("ord_owned_small_equal", std::cmp::Ordering::Equal, || {
        Ord::cmp(black_box(&small), black_box(&small_equal))
    });
    benchmark.run("ord_owned_large_equal", std::cmp::Ordering::Equal, || {
        Ord::cmp(black_box(&large), black_box(&large_equal))
    });
    let late_order = Ord::cmp(&large, &unequal_late);
    assert_ne!(late_order, std::cmp::Ordering::Equal);
    benchmark.run("ord_owned_large_unequal_late", late_order, || {
        Ord::cmp(black_box(&large), black_box(&unequal_late))
    });
}
