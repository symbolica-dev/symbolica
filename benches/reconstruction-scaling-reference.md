# Scaling reference and faster Thiele arithmetic

This checkpoint adds the actual `rare` scaling implementation to the shared
full-Q benchmark and speeds up Symbolica's Thiele reciprocal differences.
The arithmetic change preserves probe counts; the preceding
[two-variable selection change](reconstruction-bivariate-selection.md) remains
the latest probe-count reduction.

## Independent scaling reference

The standalone benchmark crate pins
[`rare` 0.9.2](https://github.com/a-maier/rare/tree/ed358c18d1129482afa51b9debb66ceef169f642)
at `ed358c18d1129482afa51b9debb66ceef169f642`. Its driver follows
[`scaling-rec`](https://github.com/a-maier/scaling-rec/tree/e79e8886f9a50a577029f7fe98dc538ab5b6e0f3)
at `e79e8886f9a50a577029f7fe98dc538ab5b6e0f3`, using one extra confirmation
point, the native descending 60-bit primes, and a seeded Xoshiro generator.
It supports one to four variables and up to sixteen native primes. No upstream
algorithm code is patched. This GPL-3.0-or-later dependency is confined to the
external benchmark crate.

All implementations receive only values from the same exported exact
coefficient oracle. Each adapter caches powers and reduces integer coefficients
for each prime. Source support and degree bounds stay private to the oracle.
Every call, including a pole, contributes to the probe count. A separate FLINT
process verifies the reconstructed rational function by exact cross multiplication
before a `rare` run can report success. Serialization and this check are outside
the reconstruction timer.

The native prime and stopping policies remain different: Symbolica uses its
usual ascending primes and three confirmation points, while `rare` uses the
authors' setting above. Full-Q totals include all fields, coefficient lifting,
and fresh-field checks. These are end-to-end implementation comparisons.

| Exact full-Q input | Symbolica probes | rare probes | Symbolica seconds | rare seconds |
|---|---:|---:|---:|---:|
| Four-loop propagator | 71,019 | 94,176 | 11.36 | 20.68 |
| Modified four-loop propagator | 71,019 | 94,176 | 10.30 | 18.62 |
| Diphoton-plus-jet amplitude | 43,957 | 169,134 | 15.92 | 298.08 |
| Modified amplitude | 43,955 | 169,134 | 16.10 | 297.49 |
| Largest nb0 coefficient string | 1,731 | 35,856 | 0.038 | 10.27 |

The initial amplitude runs hit the 180-second callback limit after 161,649 and
159,948 probes. Both failures remain archived. Separate runs with a 420-second
callback limit completed with exact results. Process limits were 300 and 600
seconds respectively. Times are single runs on a shared AMD EPYC 9754 host;
they establish observed cost, not controlled speedup estimates.

The first-field counts, 13,594 for the four-loop input and 169,132 for the
amplitude, agree with the published scaling counts. The latter gains two probes
in a fresh field for the full-Q result. The division-free recurrence used below
is equations (6)–(7) of
[Maier's paper](https://arxiv.org/html/2409.08757v2).

On the fixed 32-case `graph5` sample with seeds 1–3, all 192 reconstructions
succeed. Symbolica uses fewer probes in 81 runs and `rare` in 15, with totals
3,096 and 4,977. Counts are stable across the three seeds. The five coefficients
favoring `rare` are ranks 635, 697, 821, 883, and 945; these small cases remain
in the comparison. This sample does not establish dominance on every input.

The fixed 32-case four-variable `nb0` sample also completes exactly with `rare`
at seed 1. Symbolica uses fewer probes on 31 coefficients and ties on one:
23,084 versus 429,093 probes in total. The same source-length sampling rule
used in the earlier FireFly/FIRE7 comparison selects both samples; no cases are
removed based on performance. This run uses a 90-second callback limit and a
150-second process limit for each coefficient.

## Division-free reciprocal differences

The old inner loop divided at each preceding Thiele node. It now carries a
numerator and denominator and divides once after the loop. Intermediate zero
denominators cause the same rejection as before. Convergent construction,
sampling, method selection, and verification settings are unchanged.

The saved baseline is `9e47f9d9`. An interleaved benchmark compares both release
binaries on five deterministic dense univariate rational functions over the
same 63-bit prime, with equal numerator and denominator degrees and seeds 1–9.
Every reconstruction passes exact identity checking and every paired probe
count agrees. Input and binary hashes are recorded in the manifest.

| Degree on each side | Baseline median µs | Current median µs | Median paired speedup |
|---|---:|---:|---:|
| 32 | 512 | 223 | 2.46× |
| 64 | 2,071 | 720 | 2.89× |
| 128 | 21,098 | 16,003 | 1.32× |
| 256 | 40,536 | 20,638 | 1.92× |
| 512 | 123,938 | 45,609 | 2.76× |

These are reconstruction timings with cached-power oracle calls included and
setup excluded. Alternating execution order and CPU affinity reduce drift, but
shared-host scheduling still affects timings. The speedups are specific to
these arithmetic-heavy controls; they do not imply the same gain for every IBP
coefficient.

## Validation

All 38 reconstruction tests pass. A final rerun of all 3,319 distinct `nb0`
coefficients and all 945 `graph5` coefficients passes exact checks and preserves
every probe count, per-prime distribution, and selected method from `9e47f9d9`.
The table totals remain 505,874 and 20,311 probes. Five large full-Q controls
and modular `f1`–`f4` also preserve their counts and methods. Matching oracle
exports between the reference and final runs are byte-identical.

The adapter controls cover 21 exact reconstructions over seven functions and
three seeds, including zero, constants, large coefficients, an origin pole,
and one through four variables in a declared nonlexical order. Five expected
failure controls check time, prime and probe limits and a missing checker.
The FLINT checker separately rejects an incorrect answer and a zero denominator.
Ten forced process timeouts verify that the interleaved runner retains incomplete
outcomes with unknown counts. The five-method smoke test also passes.

Core Clippy completes with the existing `-A clippy::never_loop` allowance and
249 warnings. The standalone adapter passes Clippy with warnings denied. Both
changed Rust sources pass formatting checks. Workspace-wide formatting still
reports an unrelated existing difference in `src/transcendental.rs`; that output
is retained without modifying the file. Release builds, Python compilation,
shell syntax and diff-whitespace checks pass.

## Reproduction and evidence

Build the Symbolica release example using the existing flags and retain a
baseline executable before applying the change. Then run:

```sh
bash benches/external/build_rare.sh
python3 benches/external/check_rare_adapter.py
BENCH_CPU=24 RECONSTRUCTION_REPEATS=9 \
  python3 benches/external/compare_thiele.py \
  target/reconstruction-external/division-free-thiele.csv
BENCH_CPU=25 BENCH_METHODS=Automatic,Rare_scaling \
  BENCH_TIMEOUT=420 PROCESS_TIMEOUT=600 \
  python3 benches/external/run_q_stress.py \
  target/reconstruction-external/scaling-q.csv \
  coeff_prop_4l coeff_prop_4l_mod aajamp aajamp_mod fire7_nb0_largest
```

Use the loader environment from [external setup](external/README.md) if needed.
The pinned GMP 6.3.0 configure checks require the explicit C17 flag in the build
script with GCC 15. The standalone lockfile retains the reference's dependency
versions, including `rug` 1.25.0 and `gmp-mpfr-sys` 1.6.4 (MPFR 4.2.1).

The [result archive](results/reconstruction/scaling-reference/) retains raw
CSVs, per-prime costs, both amplitude timeouts, the longer successful runs,
adapter controls, manifests, exact-check outputs and compressed reconstructed
expressions. The existing FireFly and FIRE7 comparisons remain in the preceding
reports, with their original configurations and results.
