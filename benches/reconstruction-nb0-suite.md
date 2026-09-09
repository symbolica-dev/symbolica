# Broader IBP sampling and reuse of automatic-selection rows

The [multivariate FIRE7 comparison](reconstruction-fire7-multivariate.md) covered
one coefficient from the public `nb0` table. This checkpoint expands that sample
and reuses work already done by Symbolica's automatic method selector.

## Input selection

`extract_fire_tables.py --suite` deduplicates the table's exact coefficient
strings, keeping the first occurrence of each. It orders the 3,319 distinct
strings by decreasing length, with table order breaking ties. The suite contains
the first 16 ranks and 16 evenly spaced ranks from the remaining range. Selection
is made before running any reconstruction method. The original source contains
3,540 entries and retains its pinned FIRE7 revision and SHA-256.

The first 16 expressions have 431–877 canonical numerator terms and 196–414
denominator terms. String length is a reproducible selection rule, not an
assertion about reconstruction difficulty. The other 16 cases broaden coverage
down to the constant one. They should not all be described as complicated inputs.
All cases retain the table's declared variable order `(u,v,w,d)`, including
coefficients that do not depend on every variable. These are coefficients from
one reduction table, not 32 independent physical processes.

Each exported expression has a `.variables` sidecar. The stress example accepts
custom case names from `BENCH_INPUT_DIR`, so other input collections can use the
same runners without adding hard-coded Rust cases. The suite manifest records
source revision, source hash, selection rule, rank, integral identifiers,
expression length and each exported expression's hash. Repeated extraction is
byte-for-byte deterministic; rank 1 equals the previously exported largest case.

## Reusing the pilot row

Automatic selection compares two final-variable slices to choose the separated
balanced method. Previously it discarded both reconstructed rows and started
balanced sampling with fresh geometric bases. It now retains the first row
and its fixed coordinates. At the final variable stage, those coordinates
become the geometric bases. Geometric power one then reproduces exactly the
pilot's fixed coordinates, so the already reconstructed row supplies the first
row without new probes.

The ordinary balancing equation still determines the row's scale. Learned
degrees and sparse powers are initialized from that row through the existing
path. All pilot probes remain charged to selection and the shared budget.
The optimization adds no coefficient hypothesis and changes no verification
count: the pilot already passed the ordinary row checks, and the resulting
multivariate function still passes fresh full-dimensional checks. A failed
separated attempt discards the pilot before the existing fallback; a new attempt
performs selection again. Explicit methods and the other automatic branches
retain their original paths.

The integration test reconstructs a separated rational function over three
seeds, checks exact identities and oracle accounting, and requires a measurable
repayment of the pilot probe cost relative to explicit separated reconstruction.
The full 26-test reconstruction suite passes, including existing probe limits,
retry, sparse-support and rational-lifting checks.

## Results and remaining gaps

All 128 baseline and 384 final suite runs pass exact checks over Q. The baseline
uses seed 1; the final comparison rotates four configurations over seeds 1, 2
and 3. Probe counts are identical across the three final seeds for every case
and configuration. Every Automatic case improves; the explicit separated method
and both references retain their baseline seed-one counts and prime distributions.

The following totals sum independent scalar reconstructions once per coefficient:

| Configuration | Largest 16 | All 32 |
|---|---:|---:|
| Automatic before pilot reuse | 23,598 | 27,021 |
| Automatic with pilot reuse | **23,262** | **26,514** |
| Explicit separated Zippel | 22,894 | 25,943 |
| FireFly shift and factor scans | 74,953 | 82,403 |
| FIRE7 learned batches | 84,031 | 95,142 |

Pilot reuse reduces Automatic's total by 1.4% on the largest 16 and 1.9% on the
whole suite. The previously measured largest coefficient improves from 1,985
to 1,963 probes. Automatic still pays for selection, so its counts remain above
an explicit choice of the appropriate separated method. It uses fewer probes
than both compared references on each of the 16 largest coefficients.

The expanded sample also identifies two remaining FireFly probe advantages:

| Case | Numerator / denominator terms | Automatic before | Automatic now | FireFly scans |
|---|---:|---:|---:|---:|
| Rank 897 | 6 / 100 | 273 | 261 | **163** |
| Rank 3319, constant one | 1 / 1 | 27 | 24 | **14** |

These cases are retained. Rank 897's numerator depends only on `d`; exact
coefficient checks show that its denominator separates `d` from the kinematic
variables as well. Reusing that complete factorization during reconstruction
is a further optimization opportunity. Simply choosing another existing method
does not close the gap: separated or ordinary balanced Zippel uses 247 probes,
pruned Cuyt–Lee 550, and unpruned Cuyt–Lee 1,522. This checkpoint therefore does
not claim that Automatic beats FireFly on every coefficient in the table.

Seven larger-input regressions pass exact identities and preserve their previous
counts: modular `f1` 1,946, `f2` 1,479, `f3` 26,706 and `f4` 53,535; full-Q
four-loop 71,019, amplitude 43,960 and modified amplitude 43,958. These automatic
branches do not reuse the separated pilot. Their runtime variation is not a
performance result for this change.

## Reproduction

```sh
python3 benches/external/extract_fire_tables.py --suite
export BENCH_INPUT_DIR="$PWD/target/reconstruction-external/fire-table-inputs"
mapfile -t suite_cases < "$BENCH_INPUT_DIR/suite-cases.txt"
cargo build --locked --release --config profile.release.lto=false \
  --config profile.release.codegen-units=16 --example reconstruction_stress_benchmark
BENCH_CPU=25 BENCH_TIMEOUT=120 PROCESS_TIMEOUT=180 RECONSTRUCTION_REPEATS=3 \
  BENCH_METHODS=Automatic,BalancedZippelSeparated,FireFly_scan,FIRE7_Q_learned \
  python3 benches/external/run_q_stress.py \
  target/reconstruction-external/nb0-suite.csv "${suite_cases[@]}"
```

Use the pinned [external build and loader setup](external/README.md). For the
before/after comparison, the initial build includes the custom-input reader
but retains the reconstruction algorithms at `e4f006a`; it is copied to
`target/reconstruction-external/nb0-suite-baseline`. The baseline runner selects
it with `SYMBOLICA_STRESS_BINARY`. Q measurements retain each implementation's
native primes, count every actual oracle invocation, and require an exact
polynomial cross-product identity outside timing. Each run reconstructs one
scalar coefficient; sums of these counts do not measure shared vector probes.

Builds use Rust 1.92, release optimization with LTO disabled and 16 codegen units.
The shared AMD EPYC 9754 host has uncontrolled load, including compilation
during part of the baseline screen. Probe counts are the primary comparison;
small timing differences between builds should not be attributed to this change.

Raw [baseline](results/reconstruction/nb0-suite/baseline.csv),
[final suite](results/reconstruction/nb0-suite/suite.csv),
[remaining-gap method screen](results/reconstruction/nb0-suite/gap-methods.csv),
[modular regressions](results/reconstruction/nb0-suite/modular-regression.csv), and
[Q regressions](results/reconstruction/nb0-suite/q-regression.csv) are retained.
The [source manifest](results/reconstruction/nb0-suite/input-manifest.json),
[oracle hashes](results/reconstruction/nb0-suite/oracles.sha256),
[26-test log](results/reconstruction/nb0-suite/tests.log), and
[validation record](results/reconstruction/nb0-suite/validation.txt) document
selection and checks. The [compressed process logs](results/reconstruction/nb0-suite/process-logs.tar.gz)
contain all 515 individual suite and method-screen outputs. Large regression
logs are stored alongside their CSV files.
