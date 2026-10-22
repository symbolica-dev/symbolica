//! Diagnostics for the transparent-alias equality/hash contract.
//!
//! `cargo build --release --example alias_equality_hash_performance`
//! `SYMBOLICA_HIDE_BANNER=1 target/release/examples/alias_equality_hash_performance
//! --samples 7 --target-ms 100`
//!
//! This example intentionally asserts the new equality/hash contract before any
//! timing. It cannot validate older branches with byte-only Atom equality/hash.
//! Plain-body cases provide operation-cost comparisons within the same binary.

#[allow(dead_code)]
#[path = "equality_hash_performance.rs"]
mod support;

use std::hint::black_box;

use ahash::RandomState;
use symbolica::{atom::AtomCore, function, parse, symbol};

use support::{Benchmark, hash_value, large_arguments};

fn main() {
    let benchmark = Benchmark::from_args();
    let state = RandomState::with_seeds(1, 2, 3, 4);
    let plain = parse!(&format!("hash_eq_large({})", large_arguments()));
    let plain_equal = plain.clone();
    let aliased = plain.alias(false);
    let nested_name = symbol!("hash_eq_nested");
    let nested_plain = function!(nested_name, plain.pow(3), plain.pow(4));
    let nested_equal = nested_plain.clone();
    let nested_alias = function!(nested_name, aliased.pow(3), aliased.pow(4));
    assert!(!aliased.get_aliases().is_empty());
    assert!(!nested_alias.get_aliases().is_empty());
    assert_ne!(
        plain.as_view().get_data().as_ptr(),
        plain_equal.as_view().get_data().as_ptr()
    );
    assert_ne!(
        nested_plain.as_view().get_data().as_ptr(),
        nested_equal.as_view().get_data().as_ptr()
    );
    assert_eq!(aliased, plain);
    assert_eq!(nested_alias, nested_plain);
    let plain_hash = hash_value(&plain, &state);
    let nested_hash = hash_value(&nested_plain, &state);
    assert_eq!(hash_value(&aliased, &state), plain_hash);
    assert_eq!(hash_value(&nested_alias, &state), nested_hash);

    benchmark.run("hash_plain_body", plain_hash, || {
        hash_value(black_box(&plain), black_box(&state))
    });
    benchmark.run("hash_transparent_alias", plain_hash, || {
        hash_value(black_box(&aliased), black_box(&state))
    });
    benchmark.run("hash_nested_plain", nested_hash, || {
        hash_value(black_box(&nested_plain), black_box(&state))
    });
    benchmark.run("hash_nested_alias", nested_hash, || {
        hash_value(black_box(&nested_alias), black_box(&state))
    });
    benchmark.run("eq_plain_body", true, || {
        black_box(&plain) == black_box(&plain_equal)
    });
    benchmark.run("eq_transparent_alias_plain", true, || {
        black_box(&aliased) == black_box(&plain_equal)
    });
    benchmark.run("eq_nested_plain", true, || {
        black_box(&nested_plain) == black_box(&nested_equal)
    });
    benchmark.run("eq_nested_alias_plain", true, || {
        black_box(&nested_alias) == black_box(&nested_equal)
    });
}
