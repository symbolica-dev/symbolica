# Matched public stress benchmarks against FireFly and FIRE7

This extends the [previous stress tests](reconstruction-stress.md) with actual
external implementations on the public coefficient inputs. These results do
**not** establish that Symbolica is best for complicated IBP workloads. They
measure scalar modular reconstruction of individual coefficients, and expose
probe-count gaps that matter when evaluations are expensive.

## Matching the oracle

Both adapters evaluate the same expanded polynomials with a table of powers
for each variable and a sparse term loop, using their native field arithmetic.
Tables are reused between calls. This replaces repeated exponentiation in the
earlier stress oracle, but is a separate benchmark mode: the previous results
remain intact. Symbolica checks its cached oracle against `replace_all` at eight
points before timing. The source supports belong exclusively to the oracle and
checker, never to the reconstruction algorithm.

Symbolica exports canonical numerator/denominator terms at FireFly's first prime,
9223372036854775783. Both external adapters read those exact terms. Exported files
remain under ignored `target/reconstruction-external`. Every successful run is
checked by an exact cross-product identity outside the timer, using Symbolica
for its results and FLINT for the external results. Setup and oracle export are
outside timing; reconstruction, actual probes and result retrieval are inside.
Different field implementations and allocation patterns still affect timing.

FireFly uses one worker with `Reconstructor(nvars,1,1,bb,SILENT)` and
`reconstruct(1)`, default settings and no factor scans. The modular getter does
not restore scanned factors, so factor-scan results cannot be compared this way.
The [existing external setup](external/README.md) documents the pinned source
revisions, compiler options and portability/seed patches; algorithms are unchanged.

The FIRE7 adapter currently covers two variables, so it runs the four-loop
coefficient. It invokes the authors' Thiele and balanced-Zippel/Newton code with
no supplied degrees or supports. Generic field anchors replace the small integer
anchors in the original toy adapter, which encountered a pole in this input.
Two sampling configurations are recorded separately:

- `FIRE7_balanced_adapter`: doubling batches, starting at eight.
- `FIRE7_learned_batch`: after the first successful row, use its observed Thiele
  termination length for the next row; double again if needed. This avoids
  charging the reference for large amounts of unused batch data.

The driver caches samples and counts all actual oracle calls, including unused
tail samples. Three fresh point checks precede the exact check. FIRE7's internal
early-termination rules remain its own, and this is an adapter to its scalar
reconstruction routines, not a benchmark of the full FIRE table workflow.

## Reproduce

```sh
bash benches/external/fetch_stress.sh
bash benches/external/build_stress.sh
cargo build --locked --release --config profile.release.lto=false \
  --config profile.release.codegen-units=16 --example reconstruction_stress_benchmark
BENCH_CPU=24 BENCH_TIMEOUT=180 PROCESS_TIMEOUT=300 \
  python3 benches/external/run_stress.py \
  target/reconstruction-external/matched.csv
BENCH_CPU=24 BENCH_TIMEOUT=120 PROCESS_TIMEOUT=180 RECONSTRUCTION_REPEATS=3 \
  BENCH_METHODS=BalancedZippel,FireFly_default,FIRE7_learned_batch \
  python3 benches/external/run_stress.py \
  target/reconstruction-external/four-loop.csv coeff_prop_4l
```

The build script reuses the pinned external build or calls `build.sh` to create
it. Choose an allowed CPU or omit affinity. Nix hosts with mismatched libc
versions can use the existing `EXTERNAL_LOADER`/`EXTERNAL_LIBRARY_PATH` settings.
The runner always selects the matched prime, original variable order and cached
oracle, clearing inherited degree-race settings. Use method `BalancedZippelRace`
to enable that option explicitly. `SYMBOLICA_STRESS_BINARY` selects a saved
baseline executable; `MAX_PROBES` defaults to 200,000.

The initial screen used 60-second callback limits and 120-second process limits.
Selected external runs were extended to 180/300 seconds. Callback limits are
checked at the next invocation, so arithmetic between calls can exceed the
nominal limit. A timeout's probe count is only an incomplete prefix. All failures
and timeouts remain in the CSV, with per-process logs retained beside it.

## Initial evidence and remaining gaps

Baseline: `229a9d2` reconstruction code with the new cached oracle, release
optimization without LTO and 16 codegen units. All rows below are completed
seed-1 runs; long external runs replace timeout prefixes for comparison.

| Input | Implementation | Probes | Seconds |
|---|---|---:|---:|
| Four-loop coefficient | Symbolica balanced | 12,899 | 0.986 |
| Four-loop coefficient | FireFly default | 16,375 | 16.868 |
| Four-loop coefficient | FIRE7 doubling driver | 21,251 | 3.082 |
| Four-loop coefficient | FIRE7 learned batches | 13,718 | 2.066 |
| Diphoton-plus-jet coefficient | Symbolica balanced | 76,076 | 44.538 |
| Diphoton-plus-jet coefficient | FireFly default | 58,203 | 94.812 |
| FireFly dense `f3` | Symbolica balanced | 124,524 | 51.699 |
| FireFly dense `f3` | FireFly default | 26,816 | 61.931 |
| FireFly 20-variable `f1` | Symbolica balanced | 8,464 | 0.066 |
| FireFly 20-variable `f1` | FireFly default | 3,191 | 0.148 |
| FireFly high-degree `f2` | Symbolica balanced | 2,048 | 0.072 |
| FireFly high-degree `f2` | FireFly default | 2,457 | 1.118 |

The learned-batch FIRE7 driver uses 13,718 probes in each of three seeds, with a
2.051-second median. This is a stronger comparison than the doubling driver.
Symbolica is faster in these runs, but FireFly uses significantly fewer probes
on the amplitude and dense inputs. In particular, the dense `f3` gap is 4.6×.
Neither implementation completed `f4` in the initial 60-second screen.

A CPU profile of the baseline `aajamp` run attributes 74.72% of samples to the
cached oracle and 6.08% to the shifted transposed Vandermonde solve. The whole
process is profiled, including parsing and exact checking; its timed
reconstruction was 40.198 seconds on CPU 25. This supports prioritizing fewer
oracle calls over a small interpolation-only speedup. The scalar component
reconstruction currently keeps asking for full rows after some coefficient
components could have terminated; pruning those components is a remaining
algorithmic target. Variable ordering, factor scans, Q lifting, batched/vector
outputs, more public IBP inputs and full IBP solver oracles also remain necessary
before claiming the thread's broader performance objective is achieved.

Concrete follow-up candidates from the current code and measurements:

1. In Cuyt–Lee, stop including already reconstructed homogeneous coefficients
   as unknowns in `line_solve`; the current cache stores complete lines even
   when only a few coefficients remain unknown. Race component termination
   so easy coefficients can finish before expensive ones.
2. Avoid shifting every coordinate when a smaller shift already removes an
   origin pole. For dense FireFly examples the denominator has only three
   original terms, and unnecessary shifts destroy that sparsity.
3. In balanced interpolation, exploit support bounds for individual coefficient
   polynomials, rather than using the largest previous support for every degree.
   Reuse fixed Vandermonde data across coefficient solves after probe pruning.

These are proposed implementation directions, not measured improvements. Future
changes must be checked against the completed external counts above, including
the stronger FIRE7 sampling driver, with source information confined to oracles.

## Further monomial probe reduction

The first generic row now supplies both minimum and maximum exponents of its
numerator and denominator. Later rows interpolate after removing those learned
monomial factors, then restore them. Thus, known zero low coefficients no longer
consume probes. This also works after numerator/denominator completion, including
reciprocal rows. It requires no extra discovery probes and preserves final
validation and retries. Tests cover a high monomial factor in either side with
a 100-probe cap across ten seeds, plus the existing reference budgets and exact
identity tests.

Completed seed-1 counts after this change:

| Input | Before | After | Reduction |
|---|---:|---:|---:|
| Four-loop coefficient | 12,899 | 12,822 | 0.6% |
| Diphoton-plus-jet coefficient | 76,076 | 73,156 | 3.8% |
| FireFly `f2` | 2,048 | 2,044 | 0.2% |
| FireFly `f1` | 8,464 | 8,464 | 0.0% |
| FireFly `f3` | 124,524 | 124,524 | 0.0% |

Every result passed exact checking. The unchanged `f3` count confirms that
monomial removal does not address the dense coefficient-component gap.
The amplitude's 31.007-second updated run is faster than the initial baseline,
but runs occurred under different host load; the 3.8% probe reduction is the
reproducible algorithmic saving, not a claim that it explains the entire time
difference.

The [three-seed updated runs](results/reconstruction/matched-stress/after-seeds.csv)
all use exactly 12,822 probes for the four-loop coefficient and 73,156 for the
amplitude, with median times of 1.101 and 30.413 seconds on CPU 24. The matching
three-seed four-loop reference medians are 16,374 probes / 17.408 seconds for
FireFly and 13,718 probes / 2.071 seconds for FIRE7 with learned batches. The
pre-change Symbolica median is 12,899 probes / 1.083 seconds: the small probe
improvement does not yield a measured speedup on that cheap oracle.

All 22 targeted reconstruction and rational-polynomial tests pass. Formatting
of changed Rust files, Python syntax checks and shell syntax checks pass.
Clippy passes with the pre-existing `clippy::never_loop` error allowed; it emits
248 existing library warnings and one new argument-count warning for the
internal row interpolation routine's added minimum-degree bounds.

Raw baseline data: [60-second screen](results/reconstruction/matched-stress/before.csv),
[extended external runs](results/reconstruction/matched-stress/external-long.csv),
[three-seed four-loop comparison](results/reconstruction/matched-stress/four-loop-before.csv),
[updated screen](results/reconstruction/matched-stress/after.csv),
[profile](results/reconstruction/matched-stress/profile-before.txt), and
[exported-oracle checksums](results/reconstruction/matched-stress/oracles.sha256).
Measurements used GCC 15.2, Rust 1.92 and the shared AMD EPYC 9754 host. The initial
screen ran on CPU 24; extended external runs and the profile ran on CPU 25.
Host load was not controlled, so single-seed timings are not statistical claims.
