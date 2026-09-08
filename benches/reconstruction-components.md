# Pruning homogeneous components

This adds `ReconstructionMethod::CuytLeePruned` alongside the existing Cuyt–Lee
and balanced methods. It addresses the [matched benchmark gap](reconstruction-matched-stress.md)
on dense functions by removing completed homogeneous components from later
reconstruction problems. It does not replace balanced Zippel: method choice
still matters, particularly for very high individual degrees and sparse inputs.

## Implementation

The method learns individual degrees through ordinary black-box slices, then
homogenizes with a shift and a line parameter. It first tries an unshifted origin,
then individual coordinate shifts, then cumulative shifts until it finds a usable
normalization point. This avoids shifting every variable when a smaller shift
already removes the origin pole.

Each numerator/denominator coefficient of the line parameter is a homogeneous
polynomial in the direction variables. Completed components are evaluated on
new directions and removed from the unknowns of the rational linear solve:

```text
sum(unknown_n[i] * t^i) - f(t) * sum(unknown_d[j] * t^j)
    = f(t) * known_denominator(t) - known_numerator(t).
```

The denominator constant fixes normalization and is included in the known part.
Previously every new direction solved for every homogeneous coefficient, even
when earlier stages had already reconstructed some of them. Full line results
remain cached, so later components reuse all information from those solves.

Scheduling uses only learned individual degree bounds. A bounded-composition
count estimates how many monomials each homogeneous component could contain;
saturating arithmetic affects scheduling only. If the entire denominator's
support bound is smaller than the largest numerator component's bound, finish
the denominator first. Otherwise interleave both sides by estimated support size.
This retains the benefit of a cheap denominator without forcing an expensive
denominator to precede easy numerator components.

In this mode, polynomial Zippel also terminates a coefficient after the configured
number of consecutive zero Newton divided differences, or at its proven degree
bound, whichever comes first. Completed coefficient polynomials are subtracted
from subsequent rows. Early termination requires three successful predictions by default; reaching
the degree bound also finishes a coefficient. Final full-dimensional validation
still uses three fresh probes.
All counts include discovery, poles, unsuccessful attempts and final validation.
Source polynomials remain confined to the benchmark oracle and exact checker.

The new mode reconstructs **shifted** components and translates the final result
back. It uses Symbolica's existing `shift_var_cached` routine, also now used by
the original Cuyt–Lee path, instead of expanding polynomial replacements term by
term. Characteristic exceeds the shift degrees under the reconstruction API's
field/degree checks. Shifting can still increase support; avoiding that increase
on the amplitude coefficient is a remaining target.

The optional `degree_race` now searches both the function and its reciprocal,
with the smaller degree bounded by 32. Zeros of the original oracle are omitted
only from the reciprocal search; no extra black-box calls are made. This extends
the previous numerator-heavy search to denominator-heavy lines. The option
remains off by default because it adds arithmetic when it cannot save probes.

## Reproduction

Use the pinned inputs, external builds and cached-power oracle described in the
[matched stress report](reconstruction-matched-stress.md). The comparison field
is FireFly's first prime, 9223372036854775783, with original variable order.

```sh
bash benches/external/fetch_stress.sh
bash benches/external/build_stress.sh
cargo build --locked --release --config profile.release.lto=false \
  --config profile.release.codegen-units=16 --example reconstruction_stress_benchmark
BENCH_CPU=24 BENCH_TIMEOUT=120 PROCESS_TIMEOUT=240 \
  BENCH_METHODS=CuytLeePruned,CuytLeePrunedRace \
  python3 benches/external/run_stress.py \
  target/reconstruction-external/components.csv
```

The runner's `CuytLeePrunedRace` label selects `CuytLeePruned` with
`degree_race=true`; it is not a separate public enum variant. Use
`RECONSTRUCTION_REPEATS` for multiple seeds. The three small benchmark examples
also accept `RECONSTRUCTION_PRUNED=1` to include the new method. Their CSVs retain
method labels, and the matched stress runner records the race flag explicitly.

All successful stress results pass exact polynomial cross-product checking
outside timing. Parsing and canonicalization are also outside timing; oracle
evaluation, reconstruction, normalization and API validation are timed. Callback
limits apply at the next oracle call, and process limits additionally cover long
arithmetic stretches. Incomplete probe prefixes are not compared as completed
reconstruction counts. These are scalar coefficient reconstructions in one
prime, not complete IBP reductions.

## Measured results

The dense tests have 26,333 (`f3`) and 53,129 (`f4`) numerator terms, each with
three denominator terms. The public diphoton-plus-jet coefficient (`aajamp`) has
23,260 numerator and 7,231 denominator terms; the four-loop propagator coefficient
has 6,671 and 6,051. Input origins and pinned downloads are documented in the
[stress report](reconstruction-stress.md) and [external setup](external/README.md).

| Input | Implementation | Probes | Seconds | Seeds |
|---|---|---:|---:|---:|
| Dense `f3` | Symbolica pruned | 26,669 | 15.495 | 3 |
| Dense `f3` | Symbolica pruned + degree race | 26,606 | 16.450 | 3 |
| Dense `f3` | FireFly default | 26,816 | 84.048 | 3 |
| Dense `f4` | Symbolica pruned | 53,492 | 62.550 | 1 |
| Dense `f4` | Symbolica pruned + degree race | 53,417 | 65.220 | 1 |
| Dense `f4` | FireFly default | 53,609 | 225.998 | 1 |
| Diphoton-plus-jet | Symbolica pruned | 60,972 | 27.702 | 1 |
| Diphoton-plus-jet | Symbolica pruned + degree race | 60,966 | 32.612 | 1 |
| Diphoton-plus-jet | Symbolica balanced, preceding checkpoint | 73,156 | 30.413 | 3 |
| Diphoton-plus-jet | FireFly default, preceding measurement | 58,203 | 94.812 | 1 |
| Four-loop propagator | Symbolica pruned | 13,400 | 2.945 | 1 |
| Four-loop propagator | Symbolica balanced, preceding checkpoint | 12,822 | 1.101 | 3 |
| Four-loop propagator | FireFly default, preceding measurement | 16,374 | 17.408 | 3 |
| Four-loop propagator | FIRE7 learned batches, preceding measurement | 13,718 | 2.071 | 3 |

Three-seed entries report medians; both Symbolica variants and FireFly use the
same probe count in every `f3` seed. The preceding balanced `f3` count was
124,524: the new default mode reduces this by 78.6% and beats FireFly by 147
probes. The race saves another 63 probes but increases measured arithmetic time.
On `f4`, the default saves 117 probes against the completed FireFly run; the
race saves 192. These small reference margins include the respective libraries'
discovery and termination overheads, not a claim of uniformly better asymptotic
complexity. The larger contribution is closing the previous dense-case gap.

The amplitude's 16.7% reduction from 73,156 to 60,972 still leaves a 4.8% gap
above FireFly. Balanced Zippel remains better on the four-loop coefficient,
including against the authors' Smirnov–Zeng implementation through the existing
FIRE7 adapter. That adapter currently handles two variables; no FIRE7 result is
claimed for the higher-dimensional dense or amplitude inputs.

There are also unfavorable completed results. On 20-variable `f1`, pruned
reconstruction takes 3,741 probes against FireFly's 3,191 (preceding balanced:
8,464). On sparse high-degree `f2`, it takes 182,095 probes, or 181,223 with the
race, against 2,044 for preceding balanced Zippel and 2,457 for FireFly. Do not
select homogeneous pruning indiscriminately for sparse, high-degree functions.
The [small modular controls](results/reconstruction/components/local.md),
[rational-lifting controls](results/reconstruction/components/rational.md) and
[matched-prime controls](results/reconstruction/components/matched-ff.md) likewise
retain cases where balanced Zippel or verified separation uses fewer probes.
These controls each use three seeds, with the race disabled.

Measurements used Rust 1.92, GCC 15.2 and the shared AMD EPYC 9754 host, with
release LTO disabled and 16 codegen units. The six-case Symbolica screen ran on
CPU 24 with 120-second callback / 240-second process limits. The three-seed `f3`
runs and small controls ran on CPU 25; FireFly `f3` used 180/300-second limits
and `f4` used 240/360 seconds on CPU 25. All external runs use one worker and
unchanged pinned FireFly 2.0.3 algorithms. Host load was not controlled; timings
are observations, and probe reductions do not explain every timing difference.

Raw data and process logs:

- [Final six-case screen](results/reconstruction/components/screen.csv) and
  [process logs](results/reconstruction/components/screen-logs).
- [Three-seed Symbolica f3](results/reconstruction/components/f3-seeds.csv) and
  [FireFly f3](results/reconstruction/components/firefly-f3.csv).
- [Completed FireFly f4](results/reconstruction/components/firefly-f4.csv).
- [Original Cuyt–Lee timeout screen](results/reconstruction/components/baseline-screen.csv),
  using the saved `b093cd8` executable and a 30-second callback limit. These
  incomplete counts are not used as completed baselines.
- [Preceding balanced and external measurements](reconstruction-matched-stress.md#further-monomial-probe-reduction).
- [Release build](results/reconstruction/components/build.log),
  [reconstruction tests](results/reconstruction/components/reconstruction-tests.log),
  [serial rational-polynomial tests](results/reconstruction/components/rational-serial-tests.log),
  and [parallel variable-order failure](results/reconstruction/components/parallel-order-failure.log).

## Validation and scope

All 15 reconstruction tests pass. The new method is included in generated sparse
tests over a smaller prime, variable-order and normalization tests, rational
lifting, pole handling and bounded failure. A probe-reduction regression compares
it against the original Cuyt–Lee method on dense, shifted and sparse-component
examples. Degree-race regressions cover both orientations of degrees 50/20 under
an 80-probe cap and an initial zero that is a pole of the reciprocal.

The separate rational-polynomial suite passes all ten tests with one test thread.
A parallel run hit an assertion expecting `exp(x), exp(y)` variable order but
received the reverse; its failure log is retained. That test does not call
rational reconstruction. Clippy passes with the existing `clippy::never_loop`
error allowed and the same 249 library warnings as the preceding checkpoint.
Formatting of changed Rust files and benchmark-runner syntax are checked.

The broader performance goal remains open. The benchmark suite now has completed
FireFly results for both dense `f3` and `f4`, but scalar modular wins cannot be
extrapolated to vector/batched IBP oracles, Q lifting of these large coefficients,
factor scans or all input structures. The amplitude probe count and very sparse
high-degree inputs still need work; raw results retain unfavorable cases.
