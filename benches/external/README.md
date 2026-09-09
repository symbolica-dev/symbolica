# FireFly, Smirnov–Zeng and scaling implementation comparison

The [factor-lifting update](../reconstruction-factor-lifting.md) retains small
factors across prime images and aligns reused interpolation-line samples
between outputs. The Q runner accepts `AutomaticNoFactorReuse` as an explicit
control alongside `Automatic`; `factor_reductions` records confirmed transforms.
It snapshots the Symbolica executable for each run. The joint runner snapshots
both executables and its interpreter, preventing rebuilds from changing an
ongoing comparison. `SYMBOLICA_JOINT_BINARY` and `FIREFLY_JOINT_BINARY` select
alternate binaries; `probes_by_prime` records actual joint evaluations by field.

The public eight-propagator target also has a complete 147-output preparation:

```sh
python3 benches/external/prepare_ibp_benchmark.py tth2l_b25 \
  'basis[2,1,1,1,1,1,1,1,0,0,0]' --count 147 --label tth2l_b25_all
python3 benches/external/validate_ibp_benchmark.py tth2l_b25_all
python3 benches/external/check_joint_trace.py tth2l_b25_all
MAX_PRIMES=114 JOINT_METHODS=Symbolica_joint_cache,FireFly_joint_scan \
  python3 benches/external/run_joint_stress.py tth2l_b25_all \
  target/reconstruction-external/joint-b25-all.csv
```

Joint-output measurements use the same trace interpreter for Symbolica and
FireFly. Symbolica currently runs its scalar reconstructor for each output in
trace order, caching the complete vector at every distinct prime/point pair.
FireFly uses its native joint reconstructor. Reported probes count interpreter
evaluations returning the entire vector; they are not sums of scalar counts.
The drivers validate input and output order and check every result by exact
polynomial identity outside the reconstruction timer.

```sh
bash benches/external/build_trace_oracle.sh
bash benches/external/build_joint_stress.sh
cargo build --release --example reconstruction_joint_benchmark
python3 benches/external/check_joint_driver.py
python3 benches/external/check_joint_trace.py tth2l_b16
RECONSTRUCTION_REPEATS=3 python3 benches/external/run_joint_stress.py \
  tth2l_b16 target/reconstruction-external/joint-b16.csv
```

`JOINT_METHODS` selects `Symbolica_joint_cache`, `FireFly_joint_scan`, and/or
`FireFly_joint_default`. The runner retains named output order, binary and
trace hashes, commands, failures and incremental Symbolica probe counts.
`MAX_PRIMES`, `MAX_TOTAL_PROBES`, `BENCH_TIMEOUT`, `PROCESS_TIMEOUT` and
the external loader/CPU settings apply as in the scalar runner. The cache is
benchmark infrastructure, not yet a public multi-output reconstruction API.

The published `tth2l_b25` setup has two symbolic variables and very large
rational coefficients. Its first two selected coefficients exceed the default
32-prime budget. Use 114 primes for a common sufficiently large cap across
Symbolica, FireFly, FIRE7 and rare; FIRE7 supports at most 127 and rare 114.
The rare adapter now exposes all 114 entries of its unchanged native table,
while keeping sixteen as its default budget.

```sh
python3 benches/external/prepare_ibp_benchmark.py tth2l_b25 \
  'basis[2,1,1,1,1,1,1,1,0,0,0]'
python3 benches/external/validate_ibp_benchmark.py tth2l_b25
MAX_PRIMES=114 python3 benches/external/run_joint_stress.py \
  tth2l_b25 target/reconstruction-external/joint-b25.csv
```

`--label` preserves a separate preparation when changing the coefficient
selection. For example, the pentabubble target has 32 outputs; selecting all
32 measures every coefficient of this integral together. This is still one
integral, not the complete reduction table.

```sh
python3 benches/external/prepare_ibp_benchmark.py tth2l_b16 \
  'basis[2,1,1,1,1,1,0,0,0,0,0]' --count 32 --label tth2l_b16_all
python3 benches/external/validate_ibp_benchmark.py tth2l_b16_all
python3 benches/external/run_joint_stress.py tth2l_b16_all \
  target/reconstruction-external/joint-b16-all.csv
```

The [variable-order and trace-oracle update](../reconstruction-ordering-traces.md)
adds a seven-variable pentabubble and a three-loop diamond family. Prepare
them with the existing Kira/Ratracer build:

```sh
python3 benches/external/prepare_ibp_benchmark.py diamond3l \
  'basis[1,1,1,1,1,1,1,1,-2]'
python3 benches/external/prepare_ibp_benchmark.py tth2l_b16 \
  'basis[2,1,1,1,1,1,0,0,0,0,0]'
python3 benches/external/validate_ibp_benchmark.py diamond3l
python3 benches/external/validate_ibp_benchmark.py tth2l_b16
bash benches/external/build_rare.sh
python3 benches/external/check_rare_adapter.py
```

The rare adapter now supports one to eight variables with the same pinned
upstream implementation and prime policy. Larger inputs receive an explicit
`unsupported_variable_count` status in the shared runner.

To evaluate the original arithmetic trace during reconstruction, build the
shared interpreter and the updated FireFly driver, then validate the ABI:

```sh
bash benches/external/build_trace_oracle.sh
bash benches/external/build_stress.sh
python3 benches/external/check_trace_oracle.py tth2l_b16
python3 benches/external/check_trace_oracle.py xbox2l2m
python3 benches/external/check_trace_driver.py
export BENCH_INPUT_DIR="$PWD/target/reconstruction-external/ibp-inputs/tth2l_b16"
export TRACE_ORACLE_DIR="$BENCH_INPUT_DIR"
export TRACE_ORACLE_LIBRARY="$PWD/target/reconstruction-external/libtrace-oracle.so"
mapfile -t cases < "$BENCH_INPUT_DIR/suite-cases.txt"
BENCH_METHODS=Automatic,FireFly_scan,FireFly_default RECONSTRUCTION_REPEATS=3 \
  python3 benches/external/run_q_stress.py target/reconstruction-external/trace-b16.csv "${cases[@]}"
```

The Symbolica example needs its default `native_code_generation` feature for
the dynamic library binding. Preparation creates a `CASE.trace` link to each
optimized single-output trace. Re-run preparation if an older input directory
does not contain those links. The checker compares the interpreter with the
expanded input over four moduli, tests intermediate poles, verifies variable
order and counts, and rejects unsupported 64-bit moduli. The runner applies
the external loader to the Rust process too when needed; the Python ABI checker
reexecutes itself under that loader when configured.

`TRACE_ORACLE_PATH` selects a single trace for direct driver invocations.
Unset both trace-selection variables to restore the expanded-polynomial oracle.
The trace backend currently supports Symbolica and FireFly. FIRE7's native
64-bit primes exceed Ratracer's range, and the rare adapter has no trace binding;
the runner records `unsupported_trace_backend` for those combinations. Their
ordinary expanded-oracle comparisons remain available. Trace loading and exact
result checking are outside the timed reconstruction. FireFly reports
`trace_pole` or `trace_error` with attempted counts if its callback cannot return
a value, rather than substituting a fabricated value.

The [independent IBP family comparison](../reconstruction-independent-ibp.md)
generates coefficients from the public Ratracer benchmark configurations using
Kira equation export and Ratracer elimination. Build the pinned tools with
`bash benches/external/build_ibp_tools.sh`. This requires C++17, OpenMP, Meson,
Ninja, CMake, pkg-config, GiNaC, CLN, yaml-cpp, FLINT, GMP and zlib. Set
`FERMATPATH` to a Fermat executable for Kira's startup check; equation export
does not invoke it. The build uses a separate, pinned Ratracer-compatible
FireFly fork for input preparation, preserving the ordinary FireFly reference.

```sh
python3 benches/external/prepare_ibp_benchmark.py box2l \
  'basis[1,1,1,1,1,1,1,-1,-1]'
python3 benches/external/prepare_ibp_benchmark.py xbox2l2m \
  'basis[1,1,1,1,1,1,1,-2,0]'
for family in box2l xbox2l2m; do
  python3 benches/external/validate_ibp_benchmark.py "$family"
  export BENCH_INPUT_DIR="$PWD/target/reconstruction-external/ibp-inputs/$family"
  mapfile -t cases < "$BENCH_INPUT_DIR/suite-cases.txt"
  BENCH_METHODS=Automatic,FireFly_default,FireFly_scan,FIRE7_Q_learned,Rare_scaling \
    python3 benches/external/run_q_stress.py \
      "target/reconstruction-external/ibp-$family.csv" "${cases[@]}"
done
```

Preparation retains source revisions, equation hashes, selected integral,
coefficient ranking, expressions, trace hashes, commands and logs. Four
coefficients are selected by optimized single-output trace instruction size
before measuring reconstruction. The validator compares every expression to
its trace at eight exact rational points. These are probabilistic provenance
checks; benchmark drivers additionally verify reconstructed expressions by
exact polynomial identities. Benchmark oracles evaluate the shared expanded
expressions, so their timings are not end-to-end IBP reduction timings.

`IBP_THREADS` (default 4), `BENCH_CPU_SET` and `IBP_PREPARE_TIMEOUT` (default
600 seconds per native process) control preparation. `EXTERNAL_LOADER` and
`EXTERNAL_LIBRARY_PATH` apply to both preparation and validation where needed.
Preparations for different families may run concurrently; do not prepare the
same family concurrently in one checkout.

The [ordinary survey reuse update](../reconstruction-survey-reuse.md) adds
`python3 benches/external/shear_ibp_inputs.py`. It exports the four largest
pinned `nb0` coefficient strings with the invertible substitution `d -> d+u`
into `target/reconstruction-external/fire-table-inputs/sheared`. Set
`BENCH_INPUT_DIR` to that directory and pass its `suite-cases.txt` to the shared
Q runner. The manifest records the source hashes, substitution and fixed
selection rule. These controls deliberately remove convenient variable
separation; they remain derived from the same public table.

The [scaling reference update](../reconstruction-scaling-reference.md) adds the
actual `rare` 0.9.2 implementation as a standalone benchmark dependency. Build
it and the independent FLINT result checker with
`bash benches/external/build_rare.sh`, then run
`python3 benches/external/check_rare_adapter.py`. The shared `run_q_stress.py`
runner accepts `Rare_scaling` in `BENCH_METHODS`, including alongside
`Automatic,FireFly_default,FireFly_scan,FIRE7_Q_learned`.

`rare` uses its native 60-bit primes, one extra confirmation point, and a seeded
version of the authors' scaling driver. The adapter supports one to eight
variables and at most 114 primes (sixteen by default); `MAX_PRIMES`, `MAX_TOTAL_PROBES` and
`BENCH_TIMEOUT` set its limits. `PROCESS_TIMEOUT` limits the entire process.
Only an independently verified exact Q identity reports `ok`; failed and
incomplete runs retain their status and measured probe counts. Reconstructed
expressions and checker output remain beside the CSV in its logs directory.
The existing external-loader variables also apply to the result checker.

`compare_thiele.py OUTPUT.csv` runs an interleaved arithmetic comparison against
the executable selected by `SYMBOLICA_BASELINE` (default
`target/reconstruction-external/division-free-baseline`). Save that executable
from commit `9e47f9d9` before rebuilding. The script generates deterministic
dense univariate inputs, stores their hashes and both binary hashes, and retains
all results. `SYMBOLICA_STRESS_BINARY`, `BENCH_CPU`, `RECONSTRUCTION_REPEATS`
and `PROCESS_TIMEOUT` control the current binary and execution.

The tables here describe the performance checkpoint `306c375`. See the
[probe-count update](../reconstruction-probes.md) for the subsequent Symbolica
improvements; external implementations and their recorded runs are unchanged.

These benchmarks call actual external implementation code. They complement the
original expanded-polynomial-oracle comparisons; their timings should not be
mixed with those older measurements.

## Sources and build

- [FireFly](https://github.com/jklappert/FireFly), version 2.0.3, commit
  `4ce258e5ace6361513c4bdaac93a247cc0e3fdbb`, GPL-3.0-or-later.
- [FIRE7](https://gitlab.srcc.msu.ru/feynmanintegrals/fire), commit
  `d132e5365dd2a13db9cd9dbaf5c200b53d489cfd`, with FUEL submodule
  `8627acefec5be237080e94c1ea903a52c680aa32`. The public FIRE7 release is described
  in the [FIRE7 paper](https://arxiv.org/abs/2510.07150) and distributed through
  its [software archive](https://data.mendeley.com/datasets/cy6k69pb3y/3).
  Its `FIRE7/sources/tools/reconstruction.cpp` contains `thiele` and
  `balanced_zippel`, including the `balancedZippelNewton` mode from
  [Smirnov–Zeng](https://arxiv.org/abs/2409.19099). The archive identifies GPLv2.

External sources and binaries stay under ignored `target/reconstruction-external`.
They are not incorporated into the Symbolica library. The standalone adapters
are benchmark tooling. `build.sh` clones pinned sources and builds against system
FLINT, GMP, MPFR and zlib. It initializes only the required FUEL submodule.

Two small FireFly patches are recorded in `firefly-portability-seed.patch`: replace
an absolute `/usr/include/flint` include and allow `FIREFLY_BENCH_SEED` to override
wall-clock seeding **before** constructor-generated anchors. The algorithm is
unchanged. FUEL needs an explicit `-include flint/fmpq.h` with FLINT 3; FIRE's
reconstruction algorithms are unmodified. `MPRIME=1` satisfies an unused CLI
configuration branch. Linker section garbage collection removes unused CLI code.

Measured compiler/dependencies: GCC 15.2.0, C++17, `-O3`, FLINT 3.6.0, GMP 6.3.0,
MPFR 4.2.2; Rust 1.92.0, optimization level 3, no LTO, 16 code-generation units.
Runs were pinned to CPU 24 on an AMD EPYC 9754. Each method uses one interpolation
thread; FireFly's coordinator and single worker share the pinned CPU.

```sh
bash benches/external/build.sh
cargo build --locked --release --config profile.release.lto=false \
  --config profile.release.codegen-units=16 --example reconstruction_external_benchmark
BENCH_CPU=24 RECONSTRUCTION_REPEATS=9 python3 benches/external/run.py
```

Choose a permitted CPU or omit `BENCH_CPU`. On this Nix machine, the external
FLINT package requires a newer libc than the compiler's default loader. The run
used `EXTERNAL_LOADER` pointing to glibc 2.42's `ld-linux-x86-64.so.2`, and
`EXTERNAL_LIBRARY_PATH` containing that libc and FLINT's library directories.
These optional runner settings are unnecessary on a consistent system install.

## What is timed and checked

Both paper Eq. (3) and Eq. (28), ordered `(y,d)`, are supplied as cheap, factored
black boxes using the respective library's native finite-field arithmetic. No
expanded source polynomial, support, degree, or coefficient is given to an
interpolator. Exact source polynomials are used only for validation outside the
timer. Every successful measurement passes a polynomial cross-product identity,
over the measured field or over Q as appropriate. Failure stops the runner.

- **Symbolica:** the public APIs, three Thiele checks and three final checks,
  default retries, degree limit 64, cache enabled. Includes setup, probes,
  interpolation, result normalization, and, for Q, CRT and fresh-prime validation.
- **FireFly default:** `Reconstructor(2,1,1,bb,SILENT)`, default interpolation
  settings; `reconstruct(1)` for a modular result and `reconstruct()` for Q.
  Constructor, thread setup and result retrieval are timed. Actual black-box
  invocations are counted, including any oversampling by its scheduler.
- **FireFly scan:** both `enable_factor_scan()` and `enable_shift_scan()` enabled.
  Only Q results are benchmarked in this mode: FireFly's modular getter does not
  restore the scanned factors. Q validation uses its complete result including
  factors and any variable reordering.
- **FIRE7 adapter:** a complete two-variable modular reconstruction using the
  authors' Thiele and balanced-Zippel/Newton functions. It first learns a
  univariate skeleton, learns the number of geometric rows from that skeleton's
  support, and reconstructs each row with the authors' Thiele routine. Values are
  cached and supplied in doubling batches starting at eight. Learned row degrees
  determine Newton's limit. FUEL initialization, all probes, string parsing,
  interpolation and three fresh point checks are timed. Exact checking follows.

The FIRE7 adapter supplies the sampling driver normally provided by FIRE's table
workflow. Its probe count includes unused values at the end of a batch; repeated
Thiele attempts reuse samples. It does not use known supports or factored
denominator separation. It is neither a timing of the entire FIRE CLI nor a claim
of optimal FIRE sampling. It has bounded batches and stops on unlucky samples;
it is benchmark code, without the production Symbolica API's retry handling.
FIRE's internal early-termination rules differ from Symbolica's three-check rule.
Only Symbolica and FireFly have full Q timings here.

All modular comparisons use FireFly's first prime, `9223372036854775783`.
Q uses each implementation's native prime sequence: FireFly starts just below
2^63; Symbolica starts just above 2^61. Seed numbers do not denote identical
sample locations across implementations. Nine measured seeds follow an excluded
seed-zero warm-up. Each C++ run is a fresh process to reset static state, with
case/method order rotated by seed; process startup is outside the timer. Rust's
methods alternate in one process. These startup and scheduling differences are
particularly significant for Eq. (3). No IBP solver or artificial probe delay is
used in these external comparisons.

## Results

Nine-seed medians; times include the black box, in milliseconds:

| Case | Scope | Implementation | Probes | ms |
|---|---|---|---:|---:|
| Eq. (3) | one prime | Symbolica Cuyt–Lee | 28 | 0.034 |
| Eq. (3) | one prime | Symbolica balanced | 21 | 0.023 |
| Eq. (3) | one prime | FireFly default | 14 | 1.014 |
| Eq. (3) | one prime | FIRE7 balanced adapter | 27 | 0.546 |
| Eq. (28) | one prime | Symbolica Cuyt–Lee | 1365 | 19.020 |
| Eq. (28) | one prime | Symbolica balanced | 538 | 3.560 |
| Eq. (28) | one prime | FireFly default | 905 | 76.306 |
| Eq. (28) | one prime | FIRE7 balanced adapter | 547 | 15.256 |
| Eq. (3) | Q | Symbolica Cuyt–Lee | 31 | 0.087 |
| Eq. (3) | Q | Symbolica balanced | 24 | 0.067 |
| Eq. (3) | Q | FireFly default | 14 | 0.920 |
| Eq. (3) | Q | FireFly scans | 41 | 1.517 |
| Eq. (28) | Q | Symbolica Cuyt–Lee | 4098 | 57.218 |
| Eq. (28) | Q | Symbolica balanced | 1617 | 11.524 |
| Eq. (28) | Q | FireFly default | 1922 | 123.405 |
| Eq. (28) | Q | FireFly scans | 1694 | 108.303 |

On Eq. (28), the Symbolica balanced implementation is about 4.3× faster than this
FIRE7 adapter and uses a similar number of probes. It also outperforms the tested
FireFly configurations, including Q reconstruction with scans. Those ratios
include different arithmetic libraries, sampling drivers and scheduling costs.
They should not be extrapolated to large IBP reductions, vector outputs, batched
oracles, multiple threads, or other variable orders. FireFly's lower probe count
than our Cuyt–Lee implementation shows further rational-line pruning is worthwhile.

Raw runs: [`external-cpp.csv`](../results/reconstruction/improved/external-cpp.csv),
[`external-symbolica-ff.csv`](../results/reconstruction/improved/external-symbolica-ff.csv),
[`external-symbolica-q.csv`](../results/reconstruction/improved/external-symbolica-q.csv).
Per-process diagnostic logs are retained locally under
`target/reconstruction-external/runs/`.

## Large coefficients over Q

`build_stress.sh` also builds `firefly-q-stress`. The corresponding
`run_q_stress.py` driver exports exact integer coefficients from Symbolica and
records total probes, probes by prime, failures and each library's native prime
policy. `FireFly_scan` enables shift and factor scans; its complete result,
including restored factors and ordering, is checked exactly over Q with FLINT.
Concurrent runs use separate exported oracle files.

See the [large-Q comparison and coefficient-reuse report](../reconstruction-q-support.md)
for commands, scope, validation and results. These runs complement the modular
FIRE7 adapter.

`fire7-q-stress` now adds a full-Q multivariate comparison through the authors'
unchanged Thiele, balanced-Zippel and coefficient-lifting routines. Select
`FIRE7_Q` or `FIRE7_Q_learned` in `run_q_stress.py`; the latter retains learned
batch sizes across rows and primes. This is an adapter comparison, not a timing
of the entire FIRE table workflow. `check_q_adapters.py` checks both C++ adapters
on small exact inputs and verifies resource-limit reporting.
The adapter reconstructs one additional variable at each stage and accepts up
to 16 variables, matching the upstream coefficient-lifting buffers. See the
[multivariate FIRE7 comparison](../reconstruction-fire7-multivariate.md) for
four-variable controls and comparisons on the amplitude and `nb0` coefficients.

`extract_fire_tables.py` deterministically extracts the largest coefficient string
from the pinned `nb0` IBP table, with source hashes and integral identifiers.
It enables the `fire7_nb0_largest` case. See the
[FIRE7 Q and additional IBP report](../reconstruction-fire7-q.md) for scope,
commands and results.

Both stress runners accept `BENCH_METHODS=Automatic`, using Symbolica's generic
slice-based method selector. The modular CSV records the chosen method and
selection probes; Q records methods used by successful ordinary prime images.
See the [automatic-selection report](../reconstruction-automatic.md).

Sparse balanced-row interpolation is enabled by default. Select
`AutomaticDenseRows` or `BalancedZippelDenseRows` in either stress runner to
disable it for a comparison using the same executable. The modular CSV includes
accepted sparse rows and fallback counts. See the
[sparse-row comparison](../reconstruction-sparse-rows.md) for results and checks.

Use `extract_fire_tables.py --suite` to export a deterministic 32-coefficient
sample from the pinned `nb0` table. Set `BENCH_INPUT_DIR` to the export directory
and pass case names from its `suite-cases.txt` to either stress runner. Custom
inputs consist of an expression file named after the case and a whitespace-separated
variable-order sidecar named `CASE.variables`. See the
[broader IBP suite and pilot-row reuse report](../reconstruction-nb0-suite.md).

`extract_fire_tables.py --all` additionally exports every distinct coefficient
string into `fire-table-inputs/all`, with a separate manifest and case list.
The [whole-table factor-lifting comparison](../reconstruction-factor-lift.md)
records the complete screen, improvements and remaining reference gaps.

The [common-factor update](../reconstruction-common-factors.md) extends this
screen with default FireFly for all 3,319 coefficients, compares against the
better of the two FireFly settings, and tests common numerator factor removal
and selection of the method for the remaining variables. It retains the small
reference gaps and the one probe-count regression.

`extract_fire_tables.py --table graph5 --all` exports the second pinned public
table into `fire-table-inputs/graph5/all`. See the
[intersection reuse and graph5 comparison](../reconstruction-intersections.md)
for all 945 distinct coefficients, the unchanged native-prime comparisons, and
the separate four-loop control that limits an unconditional separation strategy.

The [two-variable selector update](../reconstruction-bivariate-selection.md)
reuses the first balanced slice to predict a simple separated denominator without
extra probes. `extract_fire_tables.py --table graph5 --suite --reverse` reproduces
the opposite-order controls in a separate directory with distinct case names.
