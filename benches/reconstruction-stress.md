# Further probe reductions and public stress tests

This update follows `0fadfac`. It retains the three-check Thiele termination and
fresh full-dimensional validation settings.

## Algorithm changes

Balanced Zippel now lifts each side as soon as it has enough rows for that side's
support. A completed denominator turns subsequent rows into polynomial numerator
interpolation. A completed numerator does the same for the reciprocal, skipping
zeros where that reciprocal is undefined. Previously every row used rational
interpolation until the larger support was complete. This optimization applies
without a separable-denominator assumption.

With `ReconstructionOptions::degree_race = true`, Thiele also races against
unbalanced rational approximants with denominator degree at most 16. The option
defaults to false because the extra arithmetic can outweigh savings on cheap
oracles. Newton interpolation and partial Euclidean steps construct these
candidates from the **same samples**, without additional black-box calls. A
candidate needs three subsequent successful checks and must satisfy all stored
samples. The bounded race particularly helps high numerator degree with low
denominator degree; it adds arithmetic overhead where it does not save probes.
It does not attempt full sparse univariate interpolation or reciprocal degree
racing.

Regression tests enforce 451 probes for ordinary balanced Eq. (28), down from 509,
or 448 with the degree race enabled. Separation uses 306 by default or 303 with
the race, down from 306. The raced ordinary count is
28 + 62 + 5×59 + 2×30 + 3 = 448; denominator completion saves 58 probes and the
degree race saves three. Separation already exploited the denominator, so it
benefits only from those three degree-discovery probes. Both totals include final
validation. Tests cover numerator completion, unequal degrees, zero functions,
fallback, generated sparse functions, and exact identities over both finite
fields and Q.

## Results

The following are **ordinary Balanced Zippel** probe counts, including validation.
The baseline is `0fadfac`; the optional-race column includes both new optimizations.
Small Eq. (28) results are medians over nine seeds; stress results use seed 1.

| Input | Previous | New default | With degree race | Best reduction |
|---|---:|---:|---:|---:|
| Paper Eq. (28), order `(y,d)` | 509 | 451 | 448 | 12.0% |
| FireFly `f1`, 20 variables | 8,464 | 8,464 | 8,464 | 0.0% |
| FireFly `f2`, powers through 300 | 2,060 | 2,048 | 1,463 | 29.0% |
| Four-loop propagator coefficient | 13,211 | 12,899 | 12,899 | 2.4% |
| Generated mixed sparse, 5 variables | 738 | 733 | 658 | 10.8% |
| Generated dense, 4 variables | 2,017 | 1,341 | 1,340 | 33.6% |

The separated method's Eq. (28) count is 306 by default and 303 with the race.
Over Q, ordinary balancing uses 1,356 probes by default or 1,347 with the race;
separated balancing uses 921 or 912. These totals include three reconstructed
prime images and validation in a fresh prime for this input.

The much larger diphoton-plus-jet coefficient `aajamp` completed with ordinary
balancing and the degree race in **76,071 probes and 177.609 seconds**, plus
5.059 seconds of parsing/canonicalization. Its exact cross-product identity
passed. The baseline reached the 180-second reconstruction cap without finishing.
Its 73,069 probes are only an incomplete prefix; there is no completed baseline
probe count or justified probe-reduction percentage for this case.
See [baseline](results/reconstruction/stress/amplitude-before.csv) and
[new run](results/reconstruction/stress/amplitude-race.csv).

The new default reconstructed the four-loop coefficient in 9.187 seconds and
the generated dense case in 75.162 ms. The optional race's timings were 10.914
seconds and 67.199 ms, respectively. Timings are sensitive to shared host load.
For cheap, factored Eq. (28) evaluations, nine-run median times were 1.743 ms
default versus 4.395 ms with the race (ordinary balancing), and 1.051 versus
3.537 ms (separated). This overhead motivates keeping the race opt-in despite
its lower probe count. It is especially relevant when black-box evaluations
cost substantially more than the benchmark's polynomial evaluations.

Both baseline and raced 30-second screens completed 14 of 24 case/method pairs.
FireFly `f3` and `f4` exceeded that cap with all three methods, as did `aajamp`
before extending its budget. Cuyt–Lee also exceeded the cap on the four-loop
coefficient. The dense FireFly numerators have 26,333 and 53,129 terms; these
remain useful unresolved stress cases. Separation can increase probes when its
hypothesis fails, and the results retain those costs. We benchmarked public
FireFly **inputs** here; the previously recorded actual FireFly/FIRE7 executable
comparisons were not rerun on these larger inputs.

Raw data and comparisons:

- [Baseline screening CSV](results/reconstruction/stress/screening-before.csv),
  [new default CSV](results/reconstruction/stress/screening-default.csv),
  [raced CSV](results/reconstruction/stress/screening-race.csv),
  and comparisons for [default](results/reconstruction/stress/screening-default.md)
  and [race](results/reconstruction/stress/screening-race.md).
- Nine-seed summaries for [expanded finite-field evaluation](results/reconstruction/stress/local.md),
  [Q reconstruction](results/reconstruction/stress/rational.md), and
  [factored finite-field evaluation](results/reconstruction/stress/matched-ff.md).
  Corresponding `*-race.md` summaries and all source CSVs sit beside them.
- [Validation log](results/reconstruction/stress/tests.log): all 21 targeted
  reconstruction and rational-polynomial tests pass, including ten seeds for each
  Eq. (28) method/option budget. Release builds and formatting of changed files
  pass; repository-wide formatting reports an existing difference in
  `src/transcendental.rs`. Clippy passes
  with the existing `clippy::never_loop` error allowed, emitting 248 existing
  library warnings and no diagnostics in the changed reconstruction code.

Builds used Rust 1.92, release mode without LTO and 16 codegen units, on an
AMD EPYC 9754 host. Measurements were pinned to CPU 24; other host workloads
were not controlled. Every successful stress result passes exact polynomial
verification, independently of the reconstructor's probabilistic checks.

## Public inputs

The harness reads the full expressions, without truncation or parameter
specialization. External input data remain under ignored
`target/reconstruction-external`; `fetch_stress.sh` pins commits and verifies
SHA-256 checksums. The public repositories identify GPL licensing; their data
are fetched separately and not copied into the Symbolica source tree.

- FireFly `f1`–`f4`, from its [published benchmark suite](https://github.com/jklappert/FireFly/tree/4ce258e5ace6361513c4bdaac93a247cc0e3fdbb/benchmarks)
  accompanying [arXiv:2004.01463](https://arxiv.org/abs/2004.01463).
  `f1` has 20 variables and degree 20. `f2` has five variables and numerator
  powers 100, 200 and 300. `f3` and `f4` have five variables and dense numerators
  of total degree 17 and 20, respectively.
- `coeff_prop_4l`: a coefficient from a differential equation for a four-loop
  propagator, in two variables, with 6,671 numerator and 6,051 denominator terms
  after canonicalization in the benchmark field.
- `aajamp`: a four-variable coefficient drawn from the two-loop diphoton-plus-jet
  results of [arXiv:2105.04585](https://arxiv.org/abs/2105.04585), with 23,260
  numerator and 7,231 denominator terms. Both physics examples come from
  Andreas Maier's [scaling-rec dataset](https://github.com/a-maier/scaling-rec/blob/e79e8886f9a50a577029f7fe98dc538ab5b6e0f3/data/README.md),
  which documents their precise provenance and supplies FireFly/FiniteFlow drivers.
- Two additional generated cases exercise mixed sparse powers in five variables
  and a dense four-variable function with denominator separation. Their exact
  expressions are in the harness.

These are reconstructions of individual coefficients, not timings of an entire
IBP reduction or amplitude calculation. The source polynomials belong only to the
oracle and exact checker; their degrees, support and factors are not passed to
the reconstructor.

## Reproduction and measurement rules

```sh
bash benches/external/fetch_stress.sh
cargo test --locked --test rational_reconstruction --test rational_polynomial
cargo build --locked --release --config profile.release.lto=false \
  --config profile.release.codegen-units=16 --example reconstruction_stress_benchmark
BENCH_CPU=24 BENCH_TIMEOUT=30 PROCESS_TIMEOUT=120 \
  python3 benches/run_reconstruction_stress.py \
  target/release/examples/reconstruction_stress_benchmark \
  benches/results/reconstruction/stress/screening.csv
```

Select another permitted CPU or omit `BENCH_CPU`. The default runner covers eight
cases and three methods. Positional arguments after the output filename restrict
cases. `BENCH_METHODS` is a comma-separated method list, `RECONSTRUCTION_REPEATS`
sets seeds 1 through N, and `BENCH_ORDER=reverse` reverses variable order.
Set `RECONSTRUCTION_DEGREE_RACE=1` to enable the optional race. The runner records
that setting in its CSV, independently of the choice of balancing method.

Every process reconstructs one function in the field 2^61−1, with degree limit
512, default probe cap 200,000 (`MAX_PROBES` overrides it), and two outer attempts.
The oracle uses Symbolica's expanded-polynomial evaluation for numerator and
denominator. Parsing/canonicalization is outside the reconstruction timer and
reported as `setup_ms`. Timings include actual oracle evaluation, reconstruction
and the API's final checks. An independent exact polynomial cross-product check
follows outside the timer. A normal result is recorded as `ok` only after that
check passes. These runs cover one prime; the original Eq. (28) Q benchmark is
rerun separately.

`BENCH_TIMEOUT` is checked at oracle entry. A private unwind exits that benchmark
when its time budget is exceeded; unexpected panics remain failures. An outer
`PROCESS_TIMEOUT` also bounds setup, long arithmetic stretches without callbacks,
and exact checking. The CSV preserves time limits, errors and process failures.
**Probe counts for timed-out runs are incomplete prefixes, not reconstruction
costs**, and must not be interpreted as improvements. The screening uses one seed
per case/method; it is not a statistical timing study.

The baseline is the same harness linked to the already-built `0fadfac` library
before changing reconstruction code, saved as `target/reconstruction-external/stress-before`.
To reproduce that baseline from source, copy the current harness into a checkout
of `0fadfac`, remove the then-unavailable `degree_race` option from the harness,
and build it with the same Cargo options. Original small-case baseline
executables were saved separately. Both versions use identical source inputs,
variable orders, oracle code and caps. Host load can change timing and timeout
prefixes, so comparisons emphasize completed probe counts.
