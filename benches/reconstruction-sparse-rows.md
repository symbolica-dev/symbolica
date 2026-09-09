# Reusing sparse powers in balanced reconstruction

The [automatic-selection checkpoint](reconstruction-automatic.md) still used
8,596 probes on FireFly's sparse 20-variable `f1`, versus FireFly's 3,191.
Balanced reconstruction knew the degrees of later rational rows, but sampled
them across the full degree spans. The first generic row already exposed the
much smaller set of nonzero powers.

The new path records those powers and uses them as a hypothesis for later rows
at the same variable stage. With an unknown denominator, its highest learned
power fixes the scale: that coefficient is set to one. A small linear system
then solves only for the remaining numerator and denominator coefficients.
When usable, the balancing point from the preceding reconstructed slice supplies
the first equation without another oracle call. Direct finite-field exponentiation avoids
materializing all intervening powers.

If a denominator is already known, only the numerator coefficients are unknown.
The same path supports reciprocal rows when the numerator finishes first;
zero oracle values are skipped when taking the reciprocal. These use the
existing completed-component and separated-denominator machinery.

Each sparse candidate must pass three fresh row checks by default. Singular
systems and rejected candidates fall back to the existing degree-bounded
solver, using the same Context and remaining probe budget. A probe-limit error
propagates immediately. The outer full-dimensional verification, retry logic
and full-Q fresh-prime checks remain in place. No source support or coefficients
are supplied to reconstruction.

To avoid replacing a cheap dense solve with a large matrix for little benefit,
the path is used only when twice the estimated sparse sample count, including
validation, is below the dense sample count. Sparse-row hypotheses are therefore
selective; ordinary dense rows retain their existing algorithm.

`ReconstructionOptions::reuse_row_support` defaults to true. Set it to false to
use the preceding row algorithm. `ReconstructionStats` reports `sparse_rows` and
`sparse_row_fallbacks`; counts include work in attempts that later fail. The
modular benchmark CSV records these counters. The example accepts
`RECONSTRUCTION_DENSE_ROWS=1`, and both stress runners expose `AutomaticDenseRows`
and `BalancedZippelDenseRows` as opt-out configuration labels.

## Sparse benchmark results

Three seeds compare the two row algorithms in the same final executable, with
fresh FireFly runs interleaved by seed. All 18 results pass exact identity checks.
The table shows median milliseconds and probe counts (FireFly `f1` varies):

| Input | Configuration | Probes | Median milliseconds |
|---|---|---:|---:|
| `f1`, 20 variables | Automatic, sparse rows | 1,946 | 40.300 |
| `f1`, 20 variables | Automatic, dense rows | 8,596 | 77.080 |
| `f1`, 20 variables | FireFly default | 3,352–3,356 | 105.800 |
| `f2`, high degree | Automatic, sparse rows | 1,479 | 60.950 |
| `f2`, high degree | Automatic, dense rows | 2,271 | 68.425 |
| `f2`, high degree | FireFly default | 2,457 | 788.890 |

Symbolica's counts are identical over seeds 1, 2 and 3. `f1` accepts 190 sparse
rows and `f2` accepts three; neither has a sparse-row fallback in these runs.
The dense opt-out reproduces the previous Automatic counts exactly. This cuts
`f1`'s total probes by 77.4% and `f2`'s by 34.9%, including row validation and
Automatic's method-selection probes.

The fresh FireFly `f1` counts differ from the earlier recorded 3,191. Both
measurements remain visible: Symbolica's 1,946 is also below that older, lower
reference. No change to FireFly's interpolation code or configuration is made
in this checkpoint. The [paired sparse results](results/reconstruction/sparse-rows/sparse.csv)
and per-process logs retain the variation rather than treating the external
probe count as invariant.

## Larger-input regression checks

All six dense modular runs and all 15 full-Q IBP runs pass exact checks. Their
probe counts and selected methods match the preceding Automatic checkpoint
for each of the three seeds:

| Input | Scope | Probes | Median seconds |
|---|---|---:|---:|
| Dense `f3` | One prime | 26,706 | 10.599 |
| Dense `f4` | One prime | 53,535 | 46.473 |
| Four-loop coefficient | Q | 71,019 | 10.750 |
| Modified four-loop coefficient | Q | 71,019 | 11.179 |
| Amplitude coefficient | Q | 43,960 | 16.142 |
| Modified amplitude coefficient | Q | 43,958 | 16.109 |
| `nb0` table coefficient | Q | 1,985 | 0.037 |

The dense modular cases do not enter the sparse-row path. Across this nine-input
large-stress corpus, Automatic is now below the compared FireFly probe references
at each benchmark's stated scope, including the older, lower `f1` reference.
The four-loop Q results also remain below the measured FIRE7 learned-batch Q
adapter. This is evidence about this scalar corpus, not complete IBP reductions,
shared probes for multiple outputs or arbitrary unseen inputs.

Raw [dense modular results](results/reconstruction/sparse-rows/dense.csv),
[Q results](results/reconstruction/sparse-rows/q.csv), and logs are retained.
All nine [exact oracle hashes](results/reconstruction/sparse-rows/oracles.sha256)
match the preceding checkpoint, and all eight pinned source input hashes pass.

## Validation and reproduction

The two new integration tests check exact identities and probe reductions on
sparse functions with unknown, fixed and reciprocal denominators. A deliberately
exceptional first row hides powers present in later rows. Its oracle is still
one consistent rational function: the added term vanishes on every query before
its fixed parameters are chosen. The test requires a rejected support hypothesis,
recovery, a correct final identity, counted oracle calls and the shared budget.
The preceding regression suite also runs with sparse-row reuse enabled.

All 25 tests pass. The 24 small-Q benchmark runs also preserve their previous
probe counts. Logs for [tests](results/reconstruction/sparse-rows/tests.log),
the [release build](results/reconstruction/sparse-rows/build.log),
[small controls](results/reconstruction/sparse-rows/small-q.csv), and the
[validation summary](results/reconstruction/sparse-rows/validation.txt) are retained.
Clippy completes with the existing `never_loop` allowance and 249 library warnings.
Runs use Rust 1.92, release optimization with LTO disabled and 16 codegen units,
on the shared AMD EPYC 9754 host. Q runs use CPU 24; the modular comparisons and
small controls use CPU 25. Host load is uncontrolled, so cross-checkpoint runtime
differences should not be attributed solely to this change.

```sh
cargo test --locked --test rational_reconstruction
cargo build --locked --release --config profile.release.lto=false \
  --config profile.release.codegen-units=16 --example reconstruction_stress_benchmark
BENCH_CPU=25 RECONSTRUCTION_REPEATS=3 BENCH_METHODS=Automatic,AutomaticDenseRows \
  python3 benches/external/run_stress.py target/reconstruction-external/sparse-rows.csv \
  firefly_f1 firefly_f2
BENCH_CPU=24 RECONSTRUCTION_REPEATS=3 BENCH_METHODS=Automatic \
  python3 benches/external/run_q_stress.py target/reconstruction-external/sparse-rows-q.csv \
  coeff_prop_4l coeff_prop_4l_mod aajamp aajamp_mod fire7_nb0_largest
```

The [external setup](external/README.md) pins the public inputs and reference
implementations. Modular comparisons use the common prime
`9223372036854775783`. Full-Q comparisons retain each library's native prime
sequence. Oracle setup and exact cross-product checks stay outside timing;
interpolation and all oracle calls, including failed hypotheses and validation,
are timed and counted. Shared-host runtime ratios remain less stable than probe
counts.
