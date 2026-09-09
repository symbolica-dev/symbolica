# Reconstructing separated rational factors

The [32-coefficient IBP suite](reconstruction-nb0-suite.md) exposed a remaining
probe gap on `nb0` rank 897: Automatic used 261 probes and scanned FireFly 163.
Its numerator depends only on `d`, and its denominator separates `d` from the
kinematic variables. Reconstructing each final-variable row independently
repeated work already visible in the selector's two pilot slices.

## Factor lifting

When the two pilot denominators agree and their nonzero numerators are
proportional, Automatic now hypothesizes that the whole last-variable rational
factor separates. It compares numerator polynomials after cross-multiplying
their leading coefficients, so different slice normalizations are allowed.
The selector receives only oracle samples; no factorization of the source
expression or coefficient metadata is supplied to it.

Suppose `f(x,t) = g(x) h(t)` and the preceding balanced stages reconstructed
`r(x) = f(x,a)`. A pilot row `s(t) = f(b,t)` then restores the full function as
`r(x) s(t) / s(a)`. Symbolica assembles that polynomial product directly instead
of sampling geometric rows for the last variable. A zero or undefined `s(a)`
rejects the hypothesis. When only the denominator appears to separate, the
previous pilot-row reuse and ordinary separated reconstruction remain in use.

Agreement of two constant slices also permits a constant candidate to proceed
directly to verification. This includes zero. Both optimizations retain the
fresh full-dimensional checks, counted probe budget and ordinary balanced
fallback. Neither agreement test is treated as proof of a global factorization.
Pilot state is consumed by the attempt and does not survive into the fallback.
Full-Q coefficient lifting and unused-prime verification are unchanged.

The new tests cover a nonconstant separated rational factor over three seeds,
zero and nonzero constants, and two deliberately misleading pilot patterns.
The latter use one consistent polynomial whose extra term vanishes on both
sampled slices. Fresh verification rejects the false whole-factor or constant
candidate, and ordinary balanced reconstruction recovers the exact function
within the same attempt. All 29 reconstruction tests pass.

## Complete public table coverage

`extract_fire_tables.py --all` exports all 3,319 distinct source coefficient
strings from the pinned `nb0` table, including the previous 32-case suite.
Duplicates retain their first table occurrence; decreasing string length and
table-order tie breaking assign the same ranks as before. The `all` subdirectory
has its own manifest, case list, expressions and variable-order sidecars, so
the original 32-case export is preserved.

All cases retain `(u,v,w,d)`, including those independent of some variables.
Repeated export is byte-for-byte deterministic and every expression hash is
checked. The source is FIRE7 revision
`d132e5365dd2a13db9cd9dbaf5c200b53d489cfd`,
`FIRE7/examples/nb0/intsde-nb0.tables`, SHA-256
`cebddb9e7b6695a3dccb0e91ebe6e821f5093b060107a957a70a60cb6090ea42`.
This covers every distinct source string in one table. It is not a claim about
independent physical processes, shared vector probes or a complete IBP solver.

## Complete-table results

All 13,276 whole-table benchmark runs pass exact identities: 3,319 baseline
Automatic runs, 3,319 final Automatic runs, and 3,319 runs of each reference.
The screen uses seed 1. A further three-seed comparison covers the original
32-case selection plus all remaining FireFly gaps, 53 cases in total. All 318
of those runs pass, with probe counts matching the screen for both Automatic
and FireFly scans across all seeds.

| Configuration | Total probes over 3,319 independent coefficients |
|---|---:|
| Automatic before factor lifting | 574,222 |
| Automatic with factor lifting | **571,020** |
| FireFly shift and factor scans | 1,211,689 |
| FIRE7 learned batches | 1,927,034 |

Symbolica reduces probes on 512 coefficients, leaves 2,807 unchanged and
increases none. The aggregate reduction is 0.56%. It uses fewer probes than
FIRE7 on all 3,319 coefficients. Against FireFly, it is lower on 3,293, equal
on four and higher on 22. These totals sum scalar runs; they do not represent
shared oracle evaluations for reconstructing the complete table as a vector.

The original rank-897 target improves from 261 to **185** probes, a 29.1%
reduction, but remains above FireFly's 163. Constant one improves from 24 to
**14**, matching FireFly. The larger original benchmark inputs retain their
previous counts: modular `f1` 1,946, `f2` 1,479, `f3` 26,706 and `f4` 53,535;
full-Q four-loop 71,019, amplitude 43,960 and modified amplitude 43,958. All
seven regression runs pass exact identities.

The number of coefficients with a FireFly probe advantage drops from 33 to 22.
The largest remaining gap is now rank 529 (26 numerator / 150 denominator
terms): Automatic uses 383 probes versus FireFly's 340. Rank 815 uses 327
versus 301. Several fully separated coefficients still spend more probes
reconstructing the remaining kinematic variables; tiny linear functions account
for additional gaps of three to five probes. These failures to match the
reference are retained in the comparison. Whole-factor lifting is useful, but
it does not yet make Automatic the lowest-probe choice on every input.

## Reproduction

```sh
python3 benches/external/extract_fire_tables.py --all
export BENCH_INPUT_DIR="$PWD/target/reconstruction-external/fire-table-inputs/all"
mapfile -t all_cases < "$BENCH_INPUT_DIR/suite-cases.txt"
cargo build --locked --release --config profile.release.lto=false \
  --config profile.release.codegen-units=16 --example reconstruction_stress_benchmark
BENCH_CPU=25 BENCH_TIMEOUT=120 PROCESS_TIMEOUT=180 BENCH_METHODS=Automatic \
  python3 benches/external/run_q_stress.py \
  target/reconstruction-external/factor-lift-all.csv "${all_cases[@]}"
BENCH_CPU=24 BENCH_TIMEOUT=120 PROCESS_TIMEOUT=180 \
  BENCH_METHODS=FireFly_scan,FIRE7_Q_learned \
  python3 benches/external/run_q_stress.py \
  target/reconstruction-external/factor-lift-all-references.csv "${all_cases[@]}"
```

The baseline uses the unchanged `ef17f7ee` release executable copied to
`target/reconstruction-external/factor-lift-baseline`, selected through
`SYMBOLICA_STRESS_BINARY`. See the [external setup](external/README.md) for pinned
reference sources and optional Nix loader settings. All Q comparisons retain
native prime sequences and exact cross-product checks outside timing. Every
actual oracle callback is counted, including selection and rejected candidates.
Runs use a shared AMD EPYC 9754 host with uncontrolled load and compilation
during part of the screen. Probe counts are the primary comparison; exact
runtime ratios should not be inferred from overlapping benchmark sessions.

The [baseline](results/reconstruction/factor-lift/baseline.csv),
[final screen](results/reconstruction/factor-lift/all.csv),
[references](results/reconstruction/factor-lift/references.csv),
[three-seed checks](results/reconstruction/factor-lift/seeds.csv), and
[remaining gaps](results/reconstruction/factor-lift/remaining-gaps.csv) are
retained as CSV. The [source manifest](results/reconstruction/factor-lift/input-manifest.json)
and [oracle hashes](results/reconstruction/factor-lift/oracles.sha256) cover all
3,319 cases; exact exported oracle bytes agree between the baseline, final and
reference screens. [Process logs](results/reconstruction/factor-lift/process-logs.tar.gz)
retain all 13,601 individual benchmark outputs, including seed checks and
regressions. [Tests](results/reconstruction/factor-lift/tests.log), build and
Clippy logs, and the [validation record](results/reconstruction/factor-lift/validation.txt)
are archived alongside the results. Clippy completes with the existing
`never_loop` allowance and 249 library warnings.
