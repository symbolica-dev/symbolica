# Common factors and reconstruction of the remaining variables

The [whole-table factor-lifting checkpoint](reconstruction-factor-lift.md) left
22 coefficients above the scanned FireFly probe count. The larger gaps included
partially factorized numerators and separated functions whose remaining
kinematic dependence was expensive for balanced interpolation.

## Changes

When two generic final-variable slices have the same monic denominator,
Automatic now uses Symbolica's polynomial GCD to find a common numerator factor.
If the whole rational factor does not separate but this GCD has positive degree,
balanced reconstruction temporarily divides oracle values by it. The already
reconstructed pilot row is divided exactly by the same polynomial. A univariate
Horner evaluation computes the divisor at each point; zeros are skipped.

The probe cache retains original oracle values. The transformation is cleared
on both success and failure, and the factor is multiplied back into the result
before full-dimensional verification. A factor common to two slices is only a
hypothesis about the complete function. An incorrect hypothesis may lead to
rejection and ordinary balanced fallback using the same cache and remaining
budget. No source factorization or support is passed to reconstruction.

Balanced reconstruction can now combine two kinds of completed information:
the fully reconstructed numerator and the learned separated denominator shape.
Once both are available, the preceding slice determines the denominator scale
for each geometric row. Those rows require no new oracle calls. This also
improves explicitly selected `BalancedZippelSeparated`; it is not limited to
Automatic or to functions with a common numerator factor.

For a fully separated factor, Automatic can use pruned Cuyt–Lee for the remaining
variables. With at least three remaining variables, a constant first-slice
numerator and a dense denominator of degree at least three, it surveys their
remaining degrees. A denominator span of at least three in the final remaining
variable selects the homogeneous method. Lower span keeps balanced reconstruction.
The survey rows are seeded from the initial slice and reused as geometric row
one when balanced reconstruction is selected, so that choice does not discard
the sampling work. Their oracle calls also contribute to `selection_probes`.

During homogeneous reconstruction the last coordinate is held at its anchor and
has support bound zero. The coordinate restriction is cleared before assembling
the factor and checking the full function. This is a method-selection heuristic,
not proof that every remaining numerator is constant or every such input is dense.

Finally, identical pilot fractions can predict dependence only on the last
variable, extending the preceding constant-candidate shortcut. The candidate
goes through the same fresh full-dimensional checks and fallback as before.

## Complete-table results

All 3,319 distinct coefficients pass exact Q verification. Relative to
`0361fad0`, total probes fall from **571,020 to 509,173 (10.8%)**: 2,195
coefficients improve, 1,123 are unchanged, and one regresses. The comparison
retains every coefficient, including constants and small functions.

| Reference | Symbolica uses fewer | Equal | Symbolica uses more | Reference total probes |
|---|---:|---:|---:|---:|
| Previous Symbolica | 2,195 | 1,123 | 1 | 571,020 |
| FireFly default | 2,977 | 25 | 317 | 2,253,840 |
| FireFly scans | 3,314 | 2 | 3 | 1,211,689 |
| Better FireFly setting per coefficient | 2,977 | 25 | 317 | 1,181,647 |
| FIRE7 balanced Zippel, learned batches | 3,319 | 0 | 0 | 1,927,034 |

The first 2,000 coefficients in the existing source-length ordering all use
fewer probes than either tested FireFly setting. This ordering predates the
optimization and is not a ranking by reconstruction difficulty. Across the
complete table, Symbolica uses 56.9% fewer probes than the sum of the better
FireFly counts. That per-coefficient minimum is an optimistic reference choice,
not a single executable configuration.

Examples, with ranks defined by the existing export manifest:

| Rank | Numerator/denominator terms | Previous | Current | FireFly default | FireFly scans |
|---|---:|---:|---:|---:|---:|
| 1 | 858 / 414 | 1,963 | **1,732** | 18,753 | 8,180 |
| 529 | 26 / 150 | 383 | **261** | 859 | 340 |
| 815 | 20 / 100 | 327 | **251** | 600 | 301 |
| 897 | 6 / 100 | 185 | **135** | 498 | 163 |
| 961 | 7 / 90 | 172 | **135** | 469 | 162 |
| 1110 | 6 / 72 | 168 | **131** | 383 | 154 |
| 2457 | 4 / 14 | **69** | 72 | 73 | 91 |
| 2294 | 20 / 3 | 92 | 92 | **87** | 131 |
| 3308 | 1 / 1 | 25 | 25 | **6** | 22 |
| 3318 | 1 / 1 | 14 | 14 | **3** | 14 |

The residual method heuristic costs three extra probes on rank 2457. Its
remaining denominator has degrees `(3, 1, 3)`, so the homogeneous choice is
slightly more expensive than the preceding balanced route. This is a measured
tradeoff, not an excluded failure. The 317 remaining gaps against the better
FireFly setting involve at most 23 numerator-plus-denominator terms. Ranks
3308–3310 also retain a three-probe gap against scan-enabled FireFly. In
particular, default FireFly handles constants in three probes, while Automatic
uses fourteen. Verification settings have not been weakened to close these gaps.

The earlier large controls still pass with unchanged probe counts: 71,019 for
`coeff_prop_4l`, 43,960 for `aajamp`, 43,958 for `aajamp_mod` over Q, and
1,946 / 1,479 / 26,706 / 53,535 for modular FireFly `f1`–`f4`. These preserve
coverage of the much larger expanded inputs outside the `nb0` table.

The sum of measured reconstruction time on the complete table is 6.97 seconds
for this version and 8.29 seconds for a fresh run of the saved baseline, with
median per-coefficient times of 647 and 722 microseconds. The earlier archived
baseline run took only 4.72 seconds in total, demonstrating substantial host
and run variability. These single sweeps do not establish a reliable latency
speedup. The exact probe reduction is the stronger result; for expensive IBP
oracles it can matter independently of interpolation overhead. All timings
exclude parsing/setup and the final exact identity check as specified by the
existing benchmark driver.

## Validation

All 36 reconstruction tests pass. New tests verify probe savings on common
factors, completed numerator/denominator rows, and a reciprocal polynomial
remaining after factor separation. Other tests deliberately arrange a consistent
rational function to mimic a common factor, a univariate function, or a dense
reciprocal polynomial on the initial samples. They require exact recovery and
correct oracle accounting after rejection, including restoration of unrestricted
coordinates and original values within the same attempt.

The full-factor savings test now compares against ordinary balanced Zippel,
because this checkpoint also improves explicit separated Zippel. Its exact
identity, accounting and probe-saving requirements remain in place. Existing
probe-budget, sparse-row, lifting, multiple-seed and variable-order tests run
alongside the new controls. A low-degree remaining-variable control requires
the survey to be reused without exceeding the preceding probe budget.

The final archive contains **14,275 successful benchmark reconstructions**:
three complete-table sweeps (current, rerun baseline, default FireFly), 4,296
seed checks, 15 additional regression checks, and seven larger controls.
All per-prime probe distributions sum to their reported totals. Exact oracle
exports match across the current implementation, rerun baseline, default
FireFly, seed checks, and preceding complete-table export. Every source
expression hash and the pinned table hash are verified.

An additional 4,296 exact Q checks cover 358 coefficients with three seeds and
four methods: Automatic, explicit separated balanced Zippel, default FireFly,
and scan-enabled FireFly. The cases are the preceding 53-case control set,
all 317 default-FireFly gaps from the initial screen, and three coefficients
that exposed an overly broad residual-method heuristic during development
(duplicates removed). The final regression at rank 2457 is checked separately
with the same four methods and three seeds, plus three runs of the previous
Automatic implementation. All four methods have stable probe counts across
the three seeds, including the 69-to-72 regression.

Clippy passes with the existing `-A clippy::never_loop` allowance and 249
warnings. Formatting checks pass for all changed Rust files. Repository-wide
`cargo fmt --all -- --check` reports an existing unrelated formatting difference
in `src/transcendental.rs`; its diagnostic is retained.

Artifacts are in [`results/reconstruction/common-factors`](results/reconstruction/common-factors/):
[all coefficients](results/reconstruction/common-factors/all.csv),
[rerun baseline](results/reconstruction/common-factors/baseline.csv),
[default FireFly](results/reconstruction/common-factors/firefly-default.csv),
[joined comparison](results/reconstruction/common-factors/comparison.csv),
[every remaining gap](results/reconstruction/common-factors/remaining-gaps.csv),
[regression](results/reconstruction/common-factors/regressions.csv),
[seed checks](results/reconstruction/common-factors/seeds.csv), and
[summary](results/reconstruction/common-factors/summary.json).
The pinned scan/FIRE7 reference data remain in the preceding
[`references.csv`](results/reconstruction/factor-lift/references.csv).
The archive also retains the complete input manifest, exact oracle hashes,
larger controls, validation logs, and compressed per-process outputs.

## Reproduction

Use the pinned [external setup](external/README.md) and the complete-table export:

```sh
python3 benches/external/extract_fire_tables.py --all
export BENCH_INPUT_DIR="$PWD/target/reconstruction-external/fire-table-inputs/all"
mapfile -t all_cases < "$BENCH_INPUT_DIR/suite-cases.txt"
cargo test --locked --test rational_reconstruction
cargo build --locked --release --config profile.release.lto=false \
  --config profile.release.codegen-units=16 --example reconstruction_stress_benchmark
BENCH_CPU=25 BENCH_TIMEOUT=120 PROCESS_TIMEOUT=180 BENCH_METHODS=Automatic \
  python3 benches/external/run_q_stress.py \
  target/reconstruction-external/common-factor-final.csv "${all_cases[@]}"
```

The baseline is `0361fad0`; its complete-table CSV and the pinned FireFly/FIRE7
reference runs are retained in the preceding report. The unchanged baseline
executable is also copied to `target/reconstruction-external/common-factor-baseline`.
It is rerun on the entire table in this checkpoint, with identical probe counts.
Every case remains an independent scalar Q reconstruction using each library's
native prime sequence, actual oracle-call accounting and an exact cross-product
check outside timing. Tests and benchmarks use Rust 1.92 on the shared AMD EPYC
9754 host with uncontrolled load. Probe counts are the primary comparison.

This checkpoint additionally runs **default FireFly on every coefficient**, not
only the scan-enabled configuration. The reference comparison uses the lower
measured count of those two settings, while retaining each raw result. The
configurations have different sampling and verification rules; counts are not
a claim of matched failure probabilities between the libraries.

Run the additional reference and seed comparisons using the external loader
settings from the pinned setup when required by the host:

```sh
BENCH_CPU=24 BENCH_TIMEOUT=120 PROCESS_TIMEOUT=180 BENCH_METHODS=FireFly_default \
  python3 benches/external/run_q_stress.py \
  target/reconstruction-external/common-factor-firefly-default.csv "${all_cases[@]}"
mapfile -t seed_cases < benches/results/reconstruction/common-factors/seed-cases.txt
BENCH_CPU=26 BENCH_TIMEOUT=120 PROCESS_TIMEOUT=180 RECONSTRUCTION_REPEATS=3 \
  BENCH_METHODS=Automatic,BalancedZippelSeparated,FireFly_default,FireFly_scan \
  python3 benches/external/run_q_stress.py \
  target/reconstruction-external/common-factor-seeds-reproduced.csv "${seed_cases[@]}"
```

The archived seed case list includes rank 2457, so this reproduction combines
the original 358-case screen and its separate one-case regression check.
