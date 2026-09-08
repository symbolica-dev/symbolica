# Removing monomial factors before homogenization

This follows the [homogeneous-component checkpoint](reconstruction-components.md)
(`6256ef2`). The pruned Cuyt–Lee method now removes monomial factors learned from
its existing degree-profile slices before homogenizing the function. This targets
the remaining probe gap on the public diphoton-plus-jet coefficient.

## Change

For each variable, a generic univariate slice supplies minimum as well as maximum
exponents in its reduced numerator and denominator. The minima predict monomials
`M_N` and `M_D` dividing the full numerator and denominator. Interpolation uses

```text
g(x) = f(x) * M_D(x) / M_N(x).
```

The numerator and denominator of the reconstructed `g` are multiplied by `M_N`
and `M_D` before validation. Individual degree bounds are reduced correspondingly,
which also improves component scheduling. No extra degree-profile probes or
factor scans are required. Source supports are confined to the benchmark oracle
and checker; the reconstruction API still receives evaluations only.

The amplitude's learned denominator factor is `(x23*x34*x45*x51)^2`. Carrying it
through a shift had increased the number of homogeneous coefficients that the
linear solves treated as unknowns. Removing it reduces this work even though
the raw oracle's poles still require shifts.

The cache stores original oracle values. The temporary transformation is applied
after cache lookup and cleared after both successful and failed interpolation
attempts. Final validation explicitly uses the original oracle. Points requiring
a removable-zero or removable-pole limit are skipped; a black-box value is not
treated as such a limit. Reported probe and pole statistics refer to actual oracle
calls. The configured total-degree limit includes the removed factors.

The method remains probabilistic: an exceptional slice can underestimate a
degree or suggest a spurious factor. The existing independent validation and
retries remain in place. This change applies to `CuytLeePruned`; balanced Zippel
already removes learned monomial factors within its interpolation rows.

## Measurements

The production binary confirms the amplitude saving across all three seeds.
Each method used the same probe count in every seed; times below are medians.

| Amplitude reconstruction | Probes | Seconds | Seeds |
|---|---:|---:|---:|
| Symbolica pruned | 43,922 | 17.028 | 3 |
| Symbolica pruned + degree race | 43,908 | 16.694 | 3 |
| FireFly default | 58,203 | 116.254 | 3 |

The default count falls **28.0%** from the preceding pruned result of 60,972,
and is **24.5% below FireFly**. The small timing difference between default and
raced runs is not evidence that racing is generally faster. Both production
variants pass exact cross-product checks against the original 23,260-term
numerator and 7,231-term denominator.

The remaining screen used one seed with the race disabled:

| Input | Pruned before | Pruned after | After seconds | FireFly reference probes |
|---|---:|---:|---:|---:|
| Dense `f3` | 26,669 | 26,669 | 11.813 | 26,816 |
| Dense `f4` | 53,492 | 53,492 | 45.584 | 53,609 |
| Four-loop coefficient | 13,400 | 13,089 | 2.295 | 16,374 |
| 20-variable `f1` | 3,741 | 3,741 | 0.060 | 3,191 |
| Sparse high-degree `f2` | 182,095 | 112,567 | 15.447 | 2,457 |

All five results pass exact checks. Dense counts remain unchanged and below
FireFly's. The four-loop count improves by 2.3%, but the previous balanced-Zippel
count of 12,822 remains better. That count also beats the authors' Smirnov–Zeng
implementation through the FIRE7 learned-batch adapter (13,718). On `f2`, the
pruned result remains much worse than balanced Zippel's 2,044 despite its 38.2%
reduction. The method must still be selected for the input structure.

The before and non-amplitude FireFly counts are the completed measurements in
[the preceding report](reconstruction-components.md) and
[matched comparisons](reconstruction-matched-stress.md#further-monomial-probe-reduction).
FireFly `f3` uses three seeds; `f4`, `f1` and `f2` use one; the four-loop value is
a three-seed median. The amplitude is the newly repeated external comparison.

The amplitude runs used CPU 24 for Symbolica and CPU 25 for FireFly. The
five-case screen and small controls used CPU 25. Production Symbolica limits were
120-second callback / 240-second process limits; FireFly used 180/300 seconds.
Every one of the 14 final stress runs completed and passed exact checking.

Raw artifacts:

- [Amplitude seeds](results/reconstruction/monomials/amplitude-seeds.csv) and
  [FireFly amplitude seeds](results/reconstruction/monomials/firefly-amplitude.csv).
- [Production screen](results/reconstruction/monomials/screen.csv) and
  [per-process logs](results/reconstruction/monomials/screen-logs).
- Three-seed [small modular controls](results/reconstruction/monomials/local.md),
  [rational-lifting controls](results/reconstruction/monomials/rational.md) and
  [matched-prime controls](results/reconstruction/monomials/matched-ff.md), with
  their CSVs in the same directory.
- [Release build](results/reconstruction/monomials/build.log) and
  [18 reconstruction tests](results/reconstruction/monomials/tests.log).

## Reproduction

Use the pinned input and external-library setup from the
[matched stress report](reconstruction-matched-stress.md). All comparisons use
prime 9223372036854775783, original variable order, cached-power oracles and exact
cross-product checking outside the timer. Parsing and canonicalization are also
outside timing. Reconstruction, oracle calls, normalization and API validation
are timed. All probe counts include discovery, poles, retries and final checks.

```sh
cargo build --locked --release --config profile.release.lto=false \
  --config profile.release.codegen-units=16 --example reconstruction_stress_benchmark
BENCH_CPU=24 BENCH_TIMEOUT=120 PROCESS_TIMEOUT=240 \
  BENCH_METHODS=CuytLeePruned,CuytLeePrunedRace RECONSTRUCTION_REPEATS=3 \
  python3 benches/external/run_stress.py \
  target/reconstruction-external/monomial-seeds.csv aajamp
BENCH_CPU=25 BENCH_TIMEOUT=180 PROCESS_TIMEOUT=300 \
  BENCH_METHODS=FireFly_default RECONSTRUCTION_REPEATS=3 \
  python3 benches/external/run_stress.py \
  target/reconstruction-external/firefly-amplitude.csv aajamp
```

Choose an allowed CPU or omit affinity. The existing external loader settings
apply on Nix hosts. The race is optional and remains disabled by default.
Timings come from a shared AMD EPYC 9754 host using Rust 1.92 and GCC 15.2.
Host load is uncontrolled; measured time differences are not all attributable to
the change in probe count.

## Rejected descending schedule

A separate experiment combined monomial removal with strictly descending
homogeneous reconstruction, subtracting known shift contributions to recover
original sparse components. The next side was chosen by the support bound of
its highest remaining degree. It passed exact checks on the amplitude but needed
102,603 probes, versus 43,922 for monomial removal alone. Four-loop counts were
19,336 versus 13,089. On dense `f3` it exhausted 200,000 probes without completing;
that count is an incomplete prefix. It improved the poor pruned result on sparse
`f2` to 72,449, which is still far above balanced Zippel's 2,044.

This schedule is not included in the API. The recorded experiment used a
standalone copy of the reconstruction module linked against Symbolica, with the
same Vandermonde implementation copied to work around its crate-private
visibility. Its probe counts inform the scheduling decision; its timings are
not used as production-library comparisons. An archived patch against `6256ef2`
reproduces the experimental algorithm inside the library; apply the
[patch](results/reconstruction/monomials/descending-experiment.patch) using
`git apply --unidiff-zero` in an isolated checkout of that commit. Enable
`RECONSTRUCTION_DESCENDING=1` after applying it; optional
`RECONSTRUCTION_TRACE=1` prints component sizes and incremental probe counts.
The [experimental results](results/reconstruction/monomials/descending-experiment.csv)
and [monomial-only prototype results](results/reconstruction/monomials/monomial-prototype.csv)
retain unsuccessful outcomes. Exploratory runs used CPU 26 for descending and
CPU 24 for monomial-only, with 120-second callback / 180-second process limits.

## Validation and remaining scope

All 18 reconstruction tests pass. New regressions cover factors in the numerator,
denominator and both, a 500-probe budget across five seeds, an unlucky profile
slice that requires clearing the transformation before retrying, and a degree
limit that would be incorrectly bypassed if removed factors were not counted.
Existing tests also cover rational lifting, smaller-prime generated inputs,
variable orders, normalization and bounded failures.
Clippy passes with the existing `clippy::never_loop` error allowed and the same
249 library warnings as the preceding checkpoint. Changed-file formatting and
diff whitespace checks pass. The refreshed small controls retain their previous
probe counts; timing differences on these unchanged cases illustrate host-load
variability.

These remain scalar modular reconstructions of individual coefficients. They do
not establish superiority for complete IBP reductions, vector or batched oracles,
large-coefficient rational lifting, factor-scan configurations or every input
structure. Balanced Zippel remains the stronger choice on the four-loop and sparse
high-degree examples; the 20-variable `f1` probe gap also remains open.
