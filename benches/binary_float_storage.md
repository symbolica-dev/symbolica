# Canonical binary float storage

MPFR and Astro share a portable central-value payload. Zero and NaN each have
one encoding; positive and negative infinity each have their own encoding.
Finite values have a minimal signed exponent and normalized significand without
zero-byte padding. Precision is separate metadata. The tracking branch also
stores canonical error magnitudes, which are not part of this main proposal.
Import rejects redundant encodings rather than retaining alternate atom bytes.
Main atom/state versions are 5/10; tracked versions are 6/11.

## Additional optimizations measured on 2026-10-23

These measurements compare the initial binary codec (`9d1854f3`) with the main
extraction (`aca302cf`) and its short-significand Astro followup. They measure
central-value serialization or comparison, not full application speed. The
tracking branch separately records full coefficient-operation and memory tests
in `benches/error_tracking_magnitude_design.md`.

Runs used a Ryzen 9 5900X, Rust 1.99.0, release LTO, CPU 2, thread CPU time,
100,000 iterations and three samples per process. Each IO comparison used three
sequential before/after process pairs; each comparison benchmark used three
processes. MPFR uses Numerica default features, Astro uses
`--no-default-features --features integer-malachite,float-astro`.
CSVs retain binary hashes and min/median/max process medians.

- Small MPFR imports use `mpfr_set_ui_2exp`, avoiding a temporary GMP integer.
  Dense 53-bit decode fell from 42.37 to 33.14 ns (1.28 times faster).
- Astro copies whole words for long byte-aligned significands. Final dense
  4096-bit decode fell from 519.44 to 54.49 ns (9.53 times faster).
- Allocating serialization computes the value parts once. A 4096-bit-precision
  power of two encoded in 38.92 ns versus 62.95 ns with MPFR (1.62 times faster).
- Borrowed value comparison avoids decoding and allocation. Across a fixed
  eight-pair mix, speedups over decoding both values were 8.56/14.43/14.10 times
  for MPFR and 11.48/7.24/1.86 times for Astro at 53/256/4096 bits.

The first Astro whole-word implementation regressed sparse decoding by 8–19%.
The followup keeps the bytewise path for significands shorter than one word:
sparse decode is now within 2% of the initial binary implementation at all three
precisions. The earlier results remain in the CSV rather than being replaced.
MPFR 256-bit dense decode regressed by 7% in the measured medians; substantial
run-to-run variation (including roughly doubled process times) limits precision
of that estimate. Reused-buffer encoding and large MPFR decoding were mostly
unchanged. No general application speedup is inferred from these microbenchmarks.

Data:
- [Initial optimization results](binary_float_io_optimizations.csv)
- [Astro short-significand followup](binary_float_io_astro_final.csv)
- [Borrowed central-value comparisons](binary_float_comparison.csv)

Reproduce with `lib/numerica/examples/float_binary_io.rs` and
`lib/numerica/examples/binary_float_comparison.rs`, setting
`NUMERICA_BENCH_CLOCK=thread` and running under `taskset -c 2`. Both accept an
iteration count; the IO harness also accepts a revision label. The root example
`examples/binary_coefficient_comparison.rs` measures full packed-coefficient
comparison against the previous comparator, including its imaginary-component
short circuit.
