# Select two-variable separation from the first slice

The [graph5 comparison](reconstruction-intersections.md) showed that explicitly
separated balanced reconstruction saves probes on 369 of its 945 coefficients,
but costs 6,891 extra probes on the much larger four-loop propagator control.
Automatic previously used ordinary balanced Zippel for every two-variable input.

## Selection rule and reuse

Automatic now constructs the first-variable slice that balanced reconstruction
would otherwise compute immediately. If every coefficient in its monic
denominator is an integer between -16 and 16 in the prime field, it predicts
that the denominator shape is independent of the remaining variable and tries
separated balanced reconstruction. Otherwise it selects ordinary balanced
Zippel. This includes constant and monomial denominators.

The prediction is deliberately conservative: a mixed-variable coefficient will
usually become a generic field element at the randomly chosen specialization,
while factors such as `x`, `x^2-1` and `(x^2-1)^2` keep small coefficients. This
is a heuristic, not a proof of independence or a coefficient-height bound.
Larger integer coefficients and small rational coefficients can go unrecognized.
The threshold does not depend on an input name, source support, or benchmark
metadata. Final full-dimensional checks and ordinary fallback remain mandatory.

Both choices consume the saved first slice and its anchor, preserving the same
probe count and random stream as the explicitly selected method. No second
discovery slice is required. These first-row probes now contribute to
`selection_probes` for Automatic even though they are reused as interpolation
work, so that statistic must not be interpreted as purely additional overhead.

The saved slice is consumed once. If the separation prediction is rejected,
ordinary fallback chooses a fresh anchor while retaining the original probe
cache and remaining budget. This is necessary because a misleading slice may
hide entire powers that ordinary sparse interpolation would otherwise omit too.

## Results

Automatic selects separated balanced reconstruction on all 945 `graph5`
coefficients and matches the explicit method's counts, with no selection
overhead. Total probes fall from **21,157 to 20,311 (4.0%)**: 369 coefficients
improve, 576 are unchanged, and none regress. The largest source coefficient
falls from 57 to 51 probes; the next from 62 to 55. All counts are stable over
seeds 1–3.

| Reference, complete graph5 table | Current uses fewer | Equal | Current uses more | Reference probes |
|---|---:|---:|---:|---:|
| Previous Automatic | 369 | 576 | 0 | 21,157 |
| Explicit separated balanced | 0 | 945 | 0 | 20,311 |
| Default FireFly | 380 | 47 | 518 | 20,754 |
| Scanned FireFly | 928 | 0 | 17 | 49,711 |
| FIRE7 with learned batches | 945 | 0 | 0 | 45,519 |

The remaining advantage for the better FireFly setting shrinks from 576 to
518 cases. Those gaps remain in the comparison; lower aggregate probes do not
mean that Automatic wins on every coefficient.

The reversed-order 32-case sample totals 1,431 probes, versus 1,433 for the
baseline, 1,471 for default FireFly and 3,368 for scanned FireFly. One case
improves and 31 are unchanged from the baseline. Relative to default FireFly,
16 cases use fewer probes, four tie and 12 use more. The limited improvement
in this order illustrates the conservative nature of the small-integer rule.
All measured configurations retain stable counts across seeds 1–3.

The complete `nb0` table is unchanged at 505,874 probes. All seven larger
controls are unchanged as well: the four-loop Q coefficient remains at 71,019,
the two amplitude variants at 43,957 and 43,955, and modular `f1`–`f4` at
1,945 / 1,478 / 26,702 / 53,531. The four-loop case still selects ordinary
balanced Zippel, avoiding the expensive unconditional-separation strategy.

## Validation

All 38 reconstruction tests pass. New tests check selection for separated,
mixed-variable and monomial denominators across three seeds. They require exact
identity, correct oracle accounting, the intended selected method, and exactly
the same probe count as the corresponding explicit method.

A separate control constructs one consistent rational function whose mixed
denominator reduces to `x^2-1` on the first slice. It requires rejection followed
by exact ordinary recovery within one attempt and a shared 150-probe cap. The
existing higher-dimensional, common-factor, hidden-support and lifting controls
also run. Clippy passes with the existing `-A clippy::never_loop` allowance and
249 warnings.

All **6,545 benchmark reconstructions** pass exact checks: 2,835 complete-table
`graph5` runs, 3,319 `nb0` controls, 288 reversed-order baseline/reference runs,
96 reversed-order final runs, and seven larger controls. Their per-prime probe
distributions sum to the reported totals. Corresponding exact oracle exports
match their baseline bytes, including those of the larger Q inputs.

Artifacts are in [`results/reconstruction/bivariate-selection`](results/reconstruction/bivariate-selection/):
[summary](results/reconstruction/bivariate-selection/summary.json),
[complete graph5 comparison](results/reconstruction/bivariate-selection/graph5-comparison.csv),
[all graph5 seeds](results/reconstruction/bivariate-selection/graph5.csv),
[reversed-order baseline/reference runs](results/reconstruction/bivariate-selection/reverse-baseline.csv),
[reversed-order final runs](results/reconstruction/bivariate-selection/reverse.csv),
[opposite-order manifest](results/reconstruction/bivariate-selection/reverse-manifest.json),
and [validation record](results/reconstruction/bivariate-selection/validation.txt).
The archive also retains all per-process outputs, oracle hashes, the full `nb0`
rerun, larger controls and validation logs. Original-order graph5 reference
runs remain in the [preceding archive](results/reconstruction/intersections/graph5-screen.csv).

## Benchmark scope

The baseline is `d47ee1f2`, saved before source changes as
`target/reconstruction-external/bivariate-baseline`. The pinned inputs, exact
oracle format, FireFly configurations, FIRE7 adapter and native prime policies
are those of the preceding report. No verification settings change.

In addition to the complete `graph5` table, the fixed 32-case suite is checked
with its variable order reversed from `y,d` to `d,y`. It uses the existing rule
of the first 16 source-length ranks plus 16 evenly spaced remaining ranks;
the expressions are byte-identical, only the variable sidecars change. Baseline
Automatic, default FireFly and scanned FireFly each run with seeds 1–3 in this
opposite order. These are separate configurations, not pooled as interchangeable
results. Timings on the shared host remain noisy; probe counts are primary.

## Reproduction

Use the pinned [external setup](external/README.md). Set its external loader
environment variables if the host needs them. The exporter accepts `--reverse`
and stores those cases separately, with an explicit `_reverse` suffix and a
manifest recording the changed order. It preserves the existing case-selection
rule and expression bytes.

```sh
python3 benches/external/extract_fire_tables.py --table graph5 --all
export BENCH_INPUT_DIR="$PWD/target/reconstruction-external/fire-table-inputs/graph5/all"
mapfile -t cases < "$BENCH_INPUT_DIR/suite-cases.txt"
BENCH_CPU=25 BENCH_METHODS=Automatic RECONSTRUCTION_REPEATS=3 \
  python3 benches/external/run_q_stress.py \
  target/reconstruction-external/bivariate-graph5.csv "${cases[@]}"

python3 benches/external/extract_fire_tables.py --table graph5 --suite --reverse
export BENCH_INPUT_DIR="$PWD/target/reconstruction-external/fire-table-inputs/graph5/reverse"
mapfile -t cases < "$BENCH_INPUT_DIR/suite-cases.txt"
BENCH_CPU=25 BENCH_METHODS=Automatic,FireFly_default,FireFly_scan \
  RECONSTRUCTION_REPEATS=3 \
  SYMBOLICA_STRESS_BINARY=target/reconstruction-external/bivariate-baseline \
  python3 benches/external/run_q_stress.py \
  target/reconstruction-external/bivariate-reverse-baseline.csv "${cases[@]}"
BENCH_CPU=26 BENCH_METHODS=Automatic RECONSTRUCTION_REPEATS=3 \
  python3 benches/external/run_q_stress.py \
  target/reconstruction-external/bivariate-reverse.csv "${cases[@]}"

python3 benches/external/extract_fire_tables.py --all
export BENCH_INPUT_DIR="$PWD/target/reconstruction-external/fire-table-inputs/all"
mapfile -t cases < "$BENCH_INPUT_DIR/suite-cases.txt"
BENCH_CPU=26 BENCH_METHODS=Automatic python3 benches/external/run_q_stress.py \
  target/reconstruction-external/bivariate-nb0.csv "${cases[@]}"
```

Build using Rust 1.92 with the existing release flags:

```sh
cargo test --locked --test rational_reconstruction
cargo build --locked --release --config profile.release.lto=false \
  --config profile.release.codegen-units=16 --example reconstruction_stress_benchmark
```
