//! Ordinary inputs for paired expansion measurements on main and global aliases.
//!
//! Copy this file and support/expansion_workloads.rs unchanged to main. Build with
//! `cargo build --release --example expansion_performance`, then execute directly:
//! `SYMBOLICA_HIDE_BANNER=1 target/release/examples/expansion_performance
//! --samples 7 --target-ms 100`.
//!
//! Pair the same workload names with alias_expansion_performance on the alias
//! branch. Inputs, expected outputs, and numerical checks are shared. Setup and
//! validation are untimed; output allocation and destruction are timed.

#[path = "support/expansion_workloads.rs"]
mod support;

fn main() {
    support::run(
        |body| body.clone(),
        |expression| expression.clone(),
        |_, _| {},
    );
}
