# Symbolica–FLINT benchmarks

The benchmark targets share one polynomial case catalog. Both configure Rayon's global pool to use
one thread, and `flint_comparison` additionally configures FLINT to use one thread.

`symbolica_polynomial` and `flint_comparison` construct the Symbolica inputs through the same helper
functions. Building without a FLINT feature therefore removes only the FLINT rows; it does not
change the Symbolica polynomials, exponent type, variable order, or timed operation.

The benchmark profile enables full LTO with one code-generation unit. Default crate features remain
enabled, including `faster_alloc`, GMP, and native code generation.
Each runner labels results with the Git revision embedded when Symbolica was compiled and the
current workspace dirty state, so a copied benchmark binary retains its build provenance instead
of adopting the repository's current revision at run time.

The resultant group measures Symbolica's Brown PRS, Lazard-Ducos, and modular CRT backends against
FLINT's multivariate resultant on the same inputs.

The `generated_gcd_regimes` group is a fixed synthetic suite covering dense low-dimensional cases
in one, two, and three variables, dense and sparse supports in five and eight variables, a 64/256
exponent-gap pair, and a 128/256/512/1024-bit coefficient-height ladder. It also contains an
eight-variable asymmetric ladder where an 8-term cofactor and a 45-term cofactor share a dense
165-, 495-, or 1287-term common factor, plus a 256-bit-coefficient variant. These cases isolate the
regime where the common factor dominates both inputs. The bivariate degree-5 case exercises the
point at which the modular GCD path replaces the small-input heuristic. The group reports the two
input-construction products separately from the automatic GCD, making it possible to distinguish
arithmetic throughput from the GCD algorithm itself. The older `polynomial_gcd` group remains
configurable through the `GCD_BENCH_*` environment variables.

The `generated_factor_products` and `generated_factorization` groups use reducible dense inputs in
one, two, and three variables. Product construction is timed separately from automatic
factorization, and both Symbolica and FLINT expand the returned factorization outside the timed
region to verify it reproduces the input.

## Symbolica-only benchmarks

The `symbolica_polynomial` target covers integer and finite-field multiplication, exact division,
resultants, and configurable polynomial GCDs without building or linking FLINT:

```sh
SYMBOLICA_LICENSE=... cargo bench \
  --features symbolica_benchmarks \
  --bench symbolica_polynomial
```

Divan filters and measurement controls can select an operation or individual case:

```sh
cargo bench --features symbolica_benchmarks --bench symbolica_polynomial -- \
  resultants --sample-count 3 --sample-size 1
```

## polybench regression cases

The `polybench_*` groups contain exact 5- and 8-variable inputs regenerated from
[polybench 0.4.3](https://github.com/tueda/polybench/tree/f3a25498883a80462c6278a87c9dfc93630d8a06)
with seed 42. The fixture set includes uniform and sharp exponent distributions, trivial and
nontrivial GCDs, published Symbolica 2.2 outliers, and inputs from the 8-variable factorization
failure sequence. Both benchmark targets use `MultivariatePolynomial<Z, u16>`, the explicit
`x1,...,xn` variable order, and lexicographic monomial order used by the upstream adapters.

The representative additions are nearest-rank p10, p50, and p90 anchors from selected published
200-case suites. The rows are ranked by their historical Symbolica 2.2 / FLINT 3.5 runtime ratio;
the selected problem numbers are therefore reproducible distribution anchors rather than claims
about current workspace performance. They cover 8-variable uniform and sharp nontrivial GCDs,
5-variable uniform nontrivial factorization, and 8-variable sharp nontrivial factorization.

For the GCD and factorization groups, compact generated factors are parsed and expanded before
timing, so only the requested operation is measured. The product groups keep the compact factors as
operands and time the reconstruction multiplications themselves. Factor and expanded-input term
counts are checked at runtime; the stored hashes record the independent fixture regeneration check.
Run all Symbolica-only regression cases with:

```sh
SYMBOLICA_LICENSE=... cargo bench \
  --features symbolica_benchmarks \
  --bench symbolica_polynomial -- polybench
```

For a same-process comparison with FLINT, use paired mode and select one fixture or operation:

```sh
SYMBOLICA_LICENSE=... \
SYMBOLICA_FLINT_BENCH_PAIRED=1 \
SYMBOLICA_FLINT_BENCH_FILTER='polybench GCD: polybench 8v uniform nontrivial GCD #55' \
SYMBOLICA_FLINT_BENCH_SAMPLES=8 \
cargo bench --features flint_benchmarks --bench flint_comparison
```

The upstream suite reports the mean of 200 distinct measured inputs after 10 distinct warm-ups.
These local regression groups repeatedly measure selected exact inputs, so they are suitable for
tracking individual slow paths but do not replace an end-to-end run of the upstream distribution.
In particular, the Symbolica 2.2 factorization crash depends on randomized search history; exact
protocol reproduction requires replaying the full ordered sequence, not only its eventual failing
input.
The fixtures retain polybench's MIT attribution and source commit in
`support/polybench_cases.rs`.

## FLINT comparisons

The reproducible configuration builds the FLINT source bundled with `flint3-sys`:

```sh
SYMBOLICA_LICENSE=... cargo bench \
  --features flint_benchmarks \
  --bench flint_comparison
```

Building bundled FLINT requires a C toolchain, Autoconf, Automake, and Libtool. To use a FLINT 3.6
installation discoverable through `pkg-config` instead, run:

```sh
SYMBOLICA_LICENSE=... cargo bench \
  --features flint_system_benchmarks \
  --bench flint_comparison
```

When FLINT is installed in a nonstandard prefix, set `PKG_CONFIG_PATH` while building and add its
library directory to the platform's dynamic-library search path while running.

Divan accepts a benchmark-name filter and measurement controls after `--`:

```sh
cargo bench --features flint_benchmarks --bench flint_comparison -- \
  integer_multiplication --sample-count 5 --sample-size 1
```

Use `--test` to construct every selected case, validate its result, and invoke each operation once:

```sh
cargo bench --features flint_benchmarks --bench flint_comparison -- \
  integer_multiplication --test
```

On Linux, pin the process to an otherwise idle core for stable wall-clock measurements:

```sh
taskset -c 2 cargo bench --features flint_benchmarks --bench flint_comparison -- \
  integer_multiplication
```

## Paired comparisons

Paired mode warms both implementations and alternates which implementation runs first for every
sample. It prints the Symbolica/FLINT median ratio directly:

```sh
SYMBOLICA_FLINT_BENCH_PAIRED=1 \
SYMBOLICA_FLINT_BENCH_FILTER='dense small multiplication' \
SYMBOLICA_FLINT_BENCH_SAMPLES=8 \
cargo bench --features flint_benchmarks --bench flint_comparison
```

Use an even sample count so each implementation runs first equally often. Set
`SYMBOLICA_FLINT_BENCH_CSV=1` for CSV output.

## Configurable GCD case

The GCD benchmarks accept the existing case controls:

- `GCD_BENCH_CASE`: `dense`, `sparse`, `high-gap`, or `high-height`
- `GCD_BENCH_NVARS`: 1 through 8
- `GCD_BENCH_DEGREE`: positive and at most 65535
- `GCD_BENCH_GAP`: positive
- `GCD_BENCH_COEFFICIENT_BITS`: 8 through 1024

For example:

```sh
GCD_BENCH_CASE=high-height \
GCD_BENCH_NVARS=8 \
GCD_BENCH_DEGREE=7 \
GCD_BENCH_COEFFICIENT_BITS=256 \
cargo bench --features flint_benchmarks --bench flint_comparison -- polynomial_gcd
```

Input construction and correctness checks run outside timed regions. Each measured call returns a
fresh result polynomial in both implementations. Exact-division cloning is also performed outside
the timed region because FLINT's exact-division operation does not consume its dividend.
