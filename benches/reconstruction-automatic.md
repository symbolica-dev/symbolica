# Automatic reconstruction selection

The [FIRE7 Q and additional IBP comparison](reconstruction-fire7-q.md) showed
that one manually selected method did not work best across the measured
coefficients. The amplitude favored pruned Cuyt–Lee, the four-loop propagator
favored balanced Zippel, and the `nb0` table coefficient favored separated
Zippel. `ReconstructionMethod::Automatic` now chooses a method using only
reconstructed generic slices and a cost heuristic.

## Selection policy

For one or two variables, the selector uses balanced Zippel without any pilot
probes. This preserves the low-dimensional default's cost, including its narrow
full-Q probe advantage over scanned FireFly on the four-loop coefficient.
It does not claim balanced Zippel is optimal for every bivariate function.

For more variables, it reconstructs two generic slices in the last variable.
If their monic denominators agree, it tries separated Zippel. This is a
probabilistic separability hypothesis; the existing fresh full-dimensional
verification and ordinary-balanced fallback still apply.

Otherwise, it surveys individual-variable degrees, minimum powers and numbers
of nonzero terms. A slice with widely separated powers on both sides selects
balanced Zippel early. Sparsity on only one side does not trigger this shortcut:
a sparse denominator can be inexpensive to remove beneath a large dense
numerator, and the same consideration applies after taking the reciprocal.

For the remaining inputs, an affine-line reconstruction estimates total degrees
after removing learned monomial factors. The generic shift prevents a common
power of the line parameter at the origin from hiding those degrees. This shift
is used only for the forecast. If homogeneous reconstruction is selected, it
uses its existing sparse-shift procedure and reuses the individual-degree
survey rather than repeating it. The selector clears the monomial transform
after the pilot even when it fails.

The implementation first checks whether the entire individual-degree box is
already cheaper in the cost model than balanced reconstruction. If so, and the
sum of original individual degrees fits the homogeneous degree cap, no
affine-line forecast is needed: imposing a total-degree bound could only reduce
the homogeneous estimate. This avoids extra probes without changing the
cost-model decision.

The cost model compares:

- the sum of numerator and denominator monomial counts allowed by their
  individual and total degree bounds, estimated by the existing homogeneous
  support-count routine;
- balanced-row costs estimated from individual degree spans and the product of
  observed univariate support sizes in each prefix of the variable order.

These are work estimates, not supplied or proven multivariate supports. Term-count
and cost arithmetic saturates to avoid integer overflow. The smaller
estimate selects pruned Cuyt–Lee or balanced Zippel. Failed degree forecasts,
including a total degree above the homogeneous method's cap, select balanced
Zippel, which can still succeed within its individual-degree cap. The total
degree check includes the removed monomial factors.

Every pilot call counts toward the same `max_probes` limit as interpolation,
retries and verification. `ReconstructionStats` reports the selected method
and `selection_probes`. The latter includes survey probes that are reused by
the selected algorithm, so it is not purely overhead. The Q stats report methods
used by successful ordinary prime images; images handled through support reuse
do not select a method and are omitted, as are failed images. Total Q probe
counts continue to include all actual calls.

Explicit methods remain available. The heuristic cannot guarantee the cheapest
method on an unknown sparse function, and it is not a parallel race between
complete reconstructions. Q coefficient reuse and fresh-prime validation are
unchanged.

## Full-Q IBP results

All five inputs use the same `Automatic` setting. Each was run with three seeds;
all 15 runs passed exact Q identity checks and had identical per-case probe
counts and method choices. Times below are three-seed medians in seconds:

| Input | Selected method | Automatic probes | Seconds | Scanned FireFly probes |
|---|---|---:|---:|---:|
| Four-loop propagator coefficient | Balanced Zippel | 71,019 | 11.304 | 71,199 |
| Author-modified four-loop coefficient | Balanced Zippel | 71,019 | 10.421 | 71,199 |
| Diphoton-plus-jet amplitude coefficient | Pruned Cuyt–Lee | 43,960 | 15.951 | 48,755 |
| Author-modified amplitude coefficient | Pruned Cuyt–Lee | 43,958 | 15.623 | 48,627 |
| Public `nb0` IBP table coefficient | Separated Zippel | 1,985 | 0.037 | 8,180 |

The reference counts are from the preceding
[large-Q](reconstruction-q-support.md) and [additional-IBP](reconstruction-fire7-q.md)
reports. Automatic also remains below the measured full-Q FIRE7 learned-batch
adapter's 108,445 probes on the four-loop input. The Q comparisons retain each
library's native prime sequence; only the modular screen below matches primes.
Reference timings were recorded in preceding runs rather than interleaved with
this screen. The three-seed reference evidence is available for the original
four-loop and `nb0` cases; the amplitude and modified-input FireFly counts come
from their recorded seed-1 runs.

This removes the need to choose a different method manually for these IBP
inputs. It does not reduce the probe count below every explicit Symbolica
configuration: selection adds 35 probes to the amplitude cases and 46 to `nb0`,
while leaving the four-loop count unchanged. All costs include selection,
interpolation, unsuccessful coefficient hypotheses and validation.

The [final Q CSV](results/reconstruction/automatic/q.csv) and per-process logs
record costs by prime and selected methods. The exact exported Q inputs match
both the initial selector screen and the preceding reference exports; hashes
are in [oracles.sha256](results/reconstruction/automatic/oracles.sha256).

## Modular stress results

Three seeds use the same `Automatic` configuration on all four public FireFly
stress inputs, at the matched prime `9223372036854775783`. All results pass
exact polynomial cross-product checks, and each case has identical counts and
method choices across seeds. Times are three-seed medians in seconds:

| Input | Selected method | Automatic probes | Seconds | FireFly probes |
|---|---|---:|---:|---:|
| Sparse 20-variable `f1` | Balanced Zippel | 8,596 | 0.048 | 3,191 |
| Sparse high-degree `f2` | Balanced Zippel | 2,271 | 0.068 | 2,457 |
| Dense `f3` | Pruned Cuyt–Lee | 26,706 | 10.605 | 26,816 |
| Dense `f4` | Pruned Cuyt–Lee | 53,535 | 43.470 | 53,609 |

FireFly reference counts come from the preceding pinned
[matched stress](reconstruction-matched-stress.md) and
[component-pruning](reconstruction-components.md) comparisons. The final
[modular CSV](results/reconstruction/automatic/modular.csv) includes pilot
costs and the selected methods. Pilot calls are 132, 227, 222 and 258 respectively;
the homogeneous cases reuse their degree surveys, so the actual increase over
explicit pruned Cuyt–Lee is just 37 probes for `f3` and 43 for `f4`.

The degree-box shortcut saves 104 probes on both dense inputs relative to the
first selector. The final automatic configuration therefore preserves the
probe advantage on `f3` and restores it on `f4`. It still loses the `f1` probe
comparison. That case's later balanced rows are sampled according to degree
spans even though earlier rows expose few nonzero powers; exploiting those
learned powers is a remaining implementation opportunity, not a measured
improvement in this checkpoint.

## Reproduction and checks

Both stress runners accept `BENCH_METHODS=Automatic` and record selected methods.
For example:

```sh
cargo build --locked --release --config profile.release.lto=false \
  --config profile.release.codegen-units=16 --example reconstruction_stress_benchmark
BENCH_CPU=24 BENCH_TIMEOUT=240 PROCESS_TIMEOUT=480 BENCH_METHODS=Automatic \
  python3 benches/external/run_q_stress.py target/reconstruction-external/auto-q.csv \
  coeff_prop_4l coeff_prop_4l_mod aajamp aajamp_mod fire7_nb0_largest
BENCH_CPU=25 BENCH_TIMEOUT=180 PROCESS_TIMEOUT=300 BENCH_METHODS=Automatic \
  python3 benches/external/run_stress.py target/reconstruction-external/auto-modular.csv \
  firefly_f1 firefly_f2 firefly_f3 firefly_f4
```

Inputs and external reference configurations are pinned as documented in the
[external setup](external/README.md). Exported supports remain confined to the
oracle and the exact identity checker. There are no case names, input-file
properties or source coefficients in the selector.

The reconstruction tests now exercise Automatic on the existing small examples
and large-Q coefficient-lifting case. A dedicated test checks separability,
dense numerator with sparse denominator and its reciprocal, widely separated
powers, affine-degree forecast failure and a large removed monomial factor.
Every result is checked exactly; a five-probe budget must stop after exactly
five oracle calls. The preceding explicit-method probe regressions remain in
the suite.

All 23 tests pass; the [test log](results/reconstruction/automatic/tests.log),
[release build log](results/reconstruction/automatic/build.log), and
[validation summary](results/reconstruction/automatic/validation.txt) are retained.
Clippy completed with the existing `never_loop` allowance and 249 library warnings.
Final measurements used Rust 1.92, LTO disabled, 16 codegen units, and single
processes pinned to CPUs 24 (Q) and 25 (modular) on the shared AMD EPYC 9754.
Host load was uncontrolled. Probe comparisons are more stable than precise
runtime ratios on this host.

The initial selector always performed the affine forecast. Its measured `f4`
count was 53,639, above FireFly's 53,609, despite choosing the intended method.
The [initial modular screen](results/reconstruction/automatic/full-forecast-modular.csv)
and [initial Q screen](results/reconstruction/automatic/full-forecast-q.csv)
are retained, with a zero-context `full-forecast.patch` that restores that policy
on this checkpoint (`git apply --unidiff-zero` in an isolated worktree).
The [explicit separated-method screen](results/reconstruction/automatic/explicit-separated.csv)
was measured with the preceding `1a20455` implementation: selecting separation
unconditionally uses 77,910 four-loop probes and 130,929 amplitude probes.

The broader goal remains open. The sparse `f1` probe gap is explicit, the
heuristic has only been measured on this finite corpus and variable order still
matters. These scalar-oracle benchmarks do not measure shared probes for vector
outputs, batched solver calls or complete IBP reductions.
