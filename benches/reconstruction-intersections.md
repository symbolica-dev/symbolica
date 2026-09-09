# Reuse intersecting slices and expand the table comparison

## Change

Automatic now reuses a reconstructed slice's value at its intersection with
later slices. For separated balanced reconstruction, the first-variable row
uses the pilot's anchor and a value computed from that pilot. When common
numerator factors have been removed, the pilot and the oracle use the same
transformation. For ordinary method selection, the final-variable pilot seeds
the other degree-profile rows at their common anchor.

This uses the existing seeded Thiele implementation. It does not reduce the
number of row checks, final full-dimensional checks, or fresh-prime checks over
Q. A pole at the predicted intersection leaves the row unseeded. Rejected
hypotheses still use the existing fallback and shared probe budget.

## Measured savings

Across the complete `nb0` table, total probes fall from **509,173 to 505,874**:
3,299 coefficients each save one probe, 20 are unchanged, and none regress
relative to `2b9e16c6`. The largest source coefficient falls from 1,732 to 1,731.
Against the better of the previously measured default/scanned FireFly settings,
Symbolica now uses fewer probes on 3,002 cases, ties on 16, and uses more on
301. The remaining gaps are retained in the complete comparison.

| Larger control | Baseline probes | Current probes |
|---|---:|---:|
| Four-loop propagator, Q | 71,019 | 71,019 |
| `aajamp`, Q | 43,960 | 43,957 |
| `aajamp_mod`, Q | 43,958 | 43,955 |
| FireFly `f1`, one prime | 1,946 | 1,945 |
| FireFly `f2`, one prime | 1,479 | 1,478 |
| FireFly `f3`, one prime | 26,706 | 26,702 |
| FireFly `f4`, one prime | 53,535 | 53,531 |

This is a small general probe saving. It does not eliminate the small-function
reference gaps or establish that Symbolica is best on every IBP workload.

## Additional public table

The exporter now supports `--table graph5`. It pins the same FIRE revision
`d132e5365dd2a13db9cd9dbaf5c200b53d489cfd` and verifies the new table's SHA-256,
`3f514ae8dd8ceb8477eb605172f85f580dddf3c1b303a0d6f8dbd8bc559df2f3`.
The input is `FIRE7/examples/graph5.tables`, with variables ordered `y, d`.
It contains 1,013 entries and 945 distinct coefficient strings. Every distinct
string is retained, in the same deterministic source-length ordering used for
`nb0`, and exported into a separate directory.

[Upstream](https://gitlab.srcc.msu.ru/feynmanintegrals/fire/-/blob/d132e5365dd2a13db9cd9dbaf5c200b53d489cfd/FIRE7/examples/README.md)
describes this as an analytic table used for synthetic reconstruction
tests: its numerical IBP tables are produced by substituting values into the
analytic coefficients. Our comparison likewise reconstructs independent scalar
coefficients. It does not time a live IBP equation solver. These coefficients
broaden the variable and degree structures tested; they are smaller than the
existing four-loop and amplitude stress inputs. The largest source expression
is 435 bytes, so this is not evidence of a new maximum problem size.

The baseline `2b9e16c6` completes all 4,725 runs: five configurations on each of
945 coefficients. Baseline Automatic uses 21,157 probes in total; explicitly
separated balanced Zippel uses 20,311, default FireFly 20,754, scanned FireFly
49,711, and learned-batch FIRE7 45,519. Automatic uses fewer probes than FIRE7
on every case, but default FireFly uses fewer on 576 cases (25 ties).

Always using the separated method for two variables would be a poor general
default: on `coeff_prop_4l`, it uses 77,910 probes, versus Automatic's 71,019.
That exact Q control is retained. The two-variable selection rule is unchanged.
The final executable repeats the complete 945-case Automatic comparison with
identical counts and exact results.

## Validation and artifacts

All 36 reconstruction tests pass, including the existing adversarial common-factor,
univariate-candidate, restricted-coordinate and fallback controls. Clippy passes
with the existing `-A clippy::never_loop` allowance and 249 warnings. Changed
Rust files pass their formatting checks.

All **10,074 benchmark reconstructions** pass exact checks: the 4,725 baseline
`graph5` runs, one additional four-loop strategy control, 3,319 final `nb0`
runs, 945 final `graph5` runs, 1,077 three-seed controls and seven larger controls.
The 359 seed cases retain identical counts across seeds 1–3. Every full-Q
per-prime distribution sums to the reported total, and all corresponding exact
oracle exports agree with their baseline bytes.

Artifacts in [`results/reconstruction/intersections`](results/reconstruction/intersections/)
include the [summary](results/reconstruction/intersections/summary.json),
[complete nb0 comparison](results/reconstruction/intersections/nb0-comparison.csv),
[all graph5 reference runs](results/reconstruction/intersections/graph5-screen.csv),
[graph5 comparison](results/reconstruction/intersections/graph5-comparison.csv),
[final graph5 runs](results/reconstruction/intersections/graph5.csv),
[seed checks](results/reconstruction/intersections/seeds.csv),
[graph5 input manifest](results/reconstruction/intersections/graph5-manifest.json),
and [validation record](results/reconstruction/intersections/validation.txt).
The archive retains all per-process outputs, oracle hashes and validation logs.
The preceding [nb0 baseline and reference data](reconstruction-common-factors.md)
remain available without duplication.

## Reproduction

Use the pinned setup in [external/README.md](external/README.md), including its
external loader settings where needed. The baseline executable is saved as
`target/reconstruction-external/graph5-baseline` before source changes.

```sh
python3 benches/external/extract_fire_tables.py --table graph5 --all
export BENCH_INPUT_DIR="$PWD/target/reconstruction-external/fire-table-inputs/graph5/all"
mapfile -t graph5_cases < "$BENCH_INPUT_DIR/suite-cases.txt"
BENCH_CPU=25 SYMBOLICA_STRESS_BINARY=target/reconstruction-external/graph5-baseline \
  BENCH_METHODS=Automatic,BalancedZippelSeparated,FireFly_default,FireFly_scan,FIRE7_Q_learned \
  python3 benches/external/run_q_stress.py \
  target/reconstruction-external/graph5-screen.csv "${graph5_cases[@]}"
cargo test --locked --test rational_reconstruction
cargo build --locked --release --config profile.release.lto=false \
  --config profile.release.codegen-units=16 --example reconstruction_stress_benchmark
BENCH_CPU=25 BENCH_METHODS=Automatic python3 benches/external/run_q_stress.py \
  target/reconstruction-external/intersection-graph5.csv "${graph5_cases[@]}"
export BENCH_INPUT_DIR="$PWD/target/reconstruction-external/fire-table-inputs/all"
mapfile -t nb0_cases < "$BENCH_INPUT_DIR/suite-cases.txt"
BENCH_CPU=25 BENCH_METHODS=Automatic python3 benches/external/run_q_stress.py \
  target/reconstruction-external/intersection-nb0.csv "${nb0_cases[@]}"
mapfile -t seed_cases < benches/results/reconstruction/intersections/seed-cases.txt
BENCH_CPU=26 BENCH_METHODS=Automatic RECONSTRUCTION_REPEATS=3 \
  python3 benches/external/run_q_stress.py \
  target/reconstruction-external/intersection-seeds.csv "${seed_cases[@]}"
```

Regenerating `nb0` with the extended exporter gives exactly the existing
manifest and every existing coefficient hash. Input support and coefficients
are never passed to reconstruction. Oracle counts include sampling and
verification; exact result checks remain outside timing. Host load is not
controlled, so elapsed times should not be treated as reliable latency ratios.
