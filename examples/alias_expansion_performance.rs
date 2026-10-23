//! Compressed inputs for the same workloads as expansion_performance.
//!
//! `cargo build --release --example alias_expansion_performance`
//! `SYMBOLICA_HIDE_BANNER=1 target/release/examples/alias_expansion_performance
//! --samples 7 --target-ms 100`
//!
//! Transparent aliases are registered during setup and remain compressed in the
//! timed input. Validation compares semantics, fully inlined expanded output
//! bytes, and independent numerical formulas at three points. Compression sizes
//! are written to stderr. Only input.expand(), including output destruction, is timed.

#[path = "support/expansion_workloads.rs"]
mod support;

use symbolica::atom::AtomCore;

fn main() {
    support::run(
        |body| body.alias(false),
        |expression| expression.inline_aliases(true),
        |input, plain| {
            assert!(!input.get_aliases().is_empty(), "input lost its aliases");
            assert!(
                input.as_view().get_byte_size() < plain.as_view().get_byte_size(),
                "input should retain useful compression"
            );
        },
    );
}
