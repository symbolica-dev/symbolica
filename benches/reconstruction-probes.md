# Matching the reference probe counts

The performance checkpoint is commit `306c375`. The subsequent changes target
Table 2 of [Smirnov–Zeng](https://arxiv.org/pdf/2409.19099), using the same Eq. (28)
and variable order `(y,d)`. No degrees, support or denominator factors are supplied
to the reconstruction API.

| Method | Paper reference | Previous implementation | Updated implementation |
|---|---:|---:|---:|
| Homogeneous / Cuyt–Lee | 1424 | 1365 | 1365 |
| Balanced Zippel | 509 | 538 | 509 |
| Balanced Zippel with separation | 306 | unavailable | 306 |

These are actual black-box calls per prime, including three final fresh checks.
They do not exclude setup probes, poles, failed attempts, or verification calls.
The paper uses different sampling/termination conventions, so agreement on the
total does not imply an identical sequence of computations.

## Changes

The first slice and first row still use Thiele with three successful checks.
Each subsequent row uses the numerator and denominator degree bounds learned
from that first row. Polynomial interpolation followed by a partial extended
Euclidean algorithm recovers a degree-bounded rational row in quadratic field
operations, using Symbolica's existing dense polynomial arithmetic. All sampled
values are checked against that interpolant without additional oracle calls.

The previous reconstructed slice supplies the value at each row intersection.
This derived value is used directly and is never counted as an oracle probe or
inserted into the oracle cache. For Eq. (28), ordinary balancing uses 31 probes
for the initial slice, 62 for the first row, 59 for each of seven remaining rows,
and three final checks: **31 + 62 + 7×59 + 3 = 509**.

The new `ReconstructionMethod::BalancedZippelSeparated` speculates that the
denominator factors between the final variable and all preceding variables.
The first row supplies that variable's denominator factor, and the previous slice
supplies its scale. Multiplying each new function value by this predicted
denominator leaves polynomial interpolation for the numerator. Eq. (28) then
needs 30 probes for each remaining row: **31 + 62 + 7×30 + 3 = 306**.

Separation is opt-in. The result must pass the same fresh full-dimensional
verification as every other method. If it fails, ordinary balanced Zippel runs
within the same attempt, sharing the oracle cache and total probe budget. There
is no assumption supplied by the caller and no unchecked separation result is
returned. The check is probabilistic, as with the other methods. A nonseparable
function can cost more because its rejected candidate is included in the count.

`verification_points` is unchanged: it controls every Thiele termination and final
validation. Degree-bounded rows use algebraic constraints instead of repeatedly
performing degree discovery; their acceptance ultimately depends on the final
fresh checks. `degree_interpolations` and `separation_fallbacks` expose this work
in the statistics and finite-field benchmark CSV.

## Reproduce and validate

```sh
cargo test --locked --test rational_reconstruction --test rational_polynomial
cargo build --locked --release --config profile.release.lto=false \
  --config profile.release.codegen-units=16 --example reconstruction_benchmark \
  --example reconstruction_rational_benchmark --example reconstruction_external_benchmark
mkdir -p benches/results/reconstruction/probes
RECONSTRUCTION_SEPARATED=1 RECONSTRUCTION_REPEATS=9 taskset -c 24 \
  target/release/examples/reconstruction_benchmark > benches/results/reconstruction/probes/local.csv
RECONSTRUCTION_SEPARATED=1 RECONSTRUCTION_REPEATS=9 taskset -c 24 \
  target/release/examples/reconstruction_rational_benchmark > benches/results/reconstruction/probes/rational.csv
RECONSTRUCTION_SEPARATED=1 RECONSTRUCTION_REPEATS=9 taskset -c 24 \
  target/release/examples/reconstruction_external_benchmark ff > benches/results/reconstruction/probes/matched-ff.csv
RECONSTRUCTION_SEPARATED=1 RECONSTRUCTION_REPEATS=9 taskset -c 24 \
  target/release/examples/reconstruction_external_benchmark q > benches/results/reconstruction/probes/matched-q.csv
python3 benches/summarize_probe_reconstruction.py benches/results/reconstruction/probes/local.csv
```

Set `PROBE_WORK=1000` on the first benchmark for the previous synthetic expensive
oracle experiment. Choose an allowed CPU or omit `taskset`. Setting
`RECONSTRUCTION_SEPARATED` adds the separation method and rotates the three methods;
without it, the original two-method benchmark remains available. Each case has an
excluded warm-up and nine recorded seeds, with exact cross-product verification
outside the timer. The matched benchmark uses factored native-field oracles and
FireFly's first prime; the original benchmark uses expanded-polynomial oracles
and 2^61−1. Their times should not be mixed.

The regression suite enforces 509 and 306 as hard oracle budgets for ten seeds,
while independently checking the returned polynomial identity and oracle call
accounting. The common generated/special-case tests now cover all three methods.
A separate nonseparable-denominator test verifies that rejection falls back even
with `max_attempts=1`. No verification threshold was lowered to meet the budgets.

## Measured results

Release measurements use the same machine, CPU affinity, compiler and optimization
settings as the preceding report. All recorded cases completed within one outer
attempt; nonseparable cases exercised the explicit fallback inside that attempt.

| Case / scope | Ordinary balanced probes | Separated probes | Ordinary ms | Separated ms |
|---|---:|---:|---:|---:|
| Eq. (28), `(y,d)`, expanded oracle, one prime | 509 | 306 | 10.939 | 6.448 |
| Eq. (28), `(d,y)`, expanded oracle, one prime | 816 | 516 | 16.566 | 10.409 |
| Eq. (28), factored oracle, one prime | 509 | 306 | 2.545 | 1.323 |
| Eq. (28), factored oracle, Q | 1530 | 921 | 9.665 | 5.915 |
| Dense box, three variables, one prime | 118 | 88 | 0.580 | 0.388 |
| Dense total degree, three variables, one prime | 160 | 276 | 0.569 | 0.924 |

Q reconstruction still uses three successful CRT images and a fourth prime for
validation, with no support resets. The 1530 and 921 totals include three fresh
verification calls in that fourth prime. Exact integer-polynomial identities
passed on every Q run.

A saved executable from the performance checkpoint was rerun for comparison.
On the factored Eq. (28) oracle, its ordinary balanced method used 538 probes and
a median 4.447 ms, versus 509 / 2.545 ms with learned-degree interpolation and
306 / 1.323 ms with separation. On the expanded oracle, its median was 14.481 ms
versus 10.939 ms and 6.448 ms. Timings vary on this shared host: the unchanged
Cuyt–Lee code also differed by about 8–13% between these measurement groups.
The probe reductions are independent of that timing variation.

Ordinary balancing reduced probes across all nine original cases. Separation
adds work when rejected: for example, the nonseparable dense-total-degree case
uses 276 probes including fallback, compared with 160 for ordinary balancing.
Variable order still matters substantially, as the reversed Eq. (28) row shows.

Raw CSVs and generated per-case summaries are in
[`results/reconstruction/probes/`](results/reconstruction/probes/), including
the rerun baseline, both oracle representations, full Q reconstruction and the
synthetic expensive-oracle variant. This update adds **648 measured runs** plus
warm-ups; all pass exact identity checks. The baseline rerun adds another 198
checked measurements. The 20 targeted tests, formatting and Clippy status are
recorded in [`validation.md`](results/reconstruction/probes/validation.md).
