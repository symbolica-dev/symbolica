# FireFly and Smirnov–Zeng implementation comparison

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
