# Reconstruction with the Hu–Monagan GCD primitives

Two opt-in methods now use the sparse interpolation machinery already used by
Symbolica's GCD routines. `PolynomialBma` reconstructs a polynomial directly;
`HuMonagan` reconstructs a rational function through BMA coefficient rows and
recursive normalization slices. Both work over a supported finite field and Q.
`Automatic` is unchanged: the new methods save many samples on sparse,
high-degree inputs, but lose on the dense and IBP inputs measured here.

## Algorithm and API

The polynomial route uses the Kronecker/Ben-Or–Tiwari approach in
[Hu and Monagan, *A fast parallel sparse polynomial GCD algorithm*](https://doi.org/10.1016/j.jsc.2020.06.001).
It reuses the existing `HuMonaganKroneckerMap`, BMA recurrence finder,
finite-field root finder, Pohlig–Hellman discrete logarithms and shifted
transposed Vandermonde solver. The GCD changes only expose the encoding helper
within the crate; they do not change GCD behavior.

A mixed-radix encoding maps monomials to distinct powers of a primitive root.
Samples at consecutive geometric points form an exponential sum. BMA recovers
its recurrence, roots identify monomials through discrete logarithms and radix
decoding, and the Vandermonde solver recovers coefficients. Independent
multiplicative shifts diversify retries. For T terms the sequence needs about
2T samples plus stability checks; the default four-term example uses 11
sequence samples and three fresh checks. Recovery is attempted at every sample
from 4 through 32, then at powers of two to bound repeated batch-BMA work.
Thus larger supports can use appreciably more than 2T samples.

The rational extension is an adaptation using these primitives, not a claim
that the GCD paper presents a general rational reconstruction algorithm.
It recursively reconstructs a slice at an anchor of the last active variable,
uses that slice's denominator to put rational row images on a common scale,
and applies BMA to their numerator and denominator coefficient sequences.
Univariate rational slices still use Thiele and learned-degree interpolation.
Consequently their degrees, variable order, and normalization cost still
matter. A bounded polynomial pilot handles polynomial slices cheaply.

Select either variant in the existing `reconstruct_rational_function` or
`reconstruct_rational_function_over_q` call:

```rust
let method = ReconstructionMethod::PolynomialBma; // Or HuMonagan for rational oracles.
let options = ReconstructionOptions {
    max_degree: 2000,
    bma_degree_bounds: Some(vec![2000, 2000, 2000]),
    bma_polynomial_probe_limit: 32,
    ..Default::default()
};
```

`bma_degree_bounds` is optional; otherwise `max_degree` bounds every variable.
Supply one upper bound per variable, each at most `max_degree`. This supplies
an exponent box, not the source's exact support. The injective encoding must
fit within the field's multiplicative group and the u64 representation.
`PolynomialBma` encodes all variables; the rational coefficient route encodes
all but the last, and skips an optional pilot if its larger box cannot fit.
The field must occur in `SMOOTH_PRIMES`; unsupported fields or boxes return
`UnsupportedBmaGeometry` before oracle calls. For example,
`Zp64::new(2401514164751985937)` supports the examples below.

Set `bma_polynomial_probe_limit` to zero to disable rational polynomial pilots;
it does not limit `PolynomialBma`. `bma_sequences` counts decoded polynomial
sequences, including pilots. Over finite fields, `PolynomialBma` returns
denominator one. Over Q, the usual integer numerator/denominator representation
may retain a constant denominator for rational coefficients.

Missing geometric samples cause retries instead of gaps in the recurrence.
Fresh probes guard misleading zero sequences, lost support and normalization
failures, including a zero anchor slice. Results remain probabilistic under
the existing verification policy; exact identities are an additional benchmark
check. All callback probes, including failed attempts and verification, count
toward the budget. Tight but incorrect user degree bounds are not certified.

Over Q, the new methods choose listed smooth primes in descending order below
2^63 that fit the exponent box. Other methods retain consecutive primes above
2^61. CRT, support reuse and unused-prime verification remain in place. The
eligible smooth-prime list is finite, so an unusually large box or coefficient
can exhaust it and return `PrimeLimit`.

## Matched-prime sample comparison

These are scalar black-box callback counts at the same prime
**2401514164751985937**, with seeds 1, 2 and 3, default three fresh checks,
20,000-probe cap and a 120-second callback-checked soft timeout. Method order
rotates between seeds. Every listed count was identical across the three
seeds. All 159 completed reconstructions passed exact cross multiplication;
three nonplanar HuMonagan runs reached the probe cap and are retained as failures.
The callback receives only field and point, with no support supplied to the
algorithm. PolynomialBma is tested only on known-polynomial cases.

| Input | Automatic | BalancedZippel | HuMonagan | PolynomialBma |
|---|---:|---:|---:|---:|
| Sparse polynomial, D=5 | 78 | 65 | 17 | 14 |
| Sparse polynomial, D=50 | 438 | 335 | 17 | 14 |
| Sparse polynomial, D=500 | 4,038 | 3,035 | 17 | 14 |
| Sparse polynomial, D=2000 | 16,038 | 12,035 | 17 | 14 |
| Sparse rational, D=5 | 52 | 44 | 88 | — |
| Sparse rational, D=50 | 231 | 223 | 88 | — |
| Sparse rational, D=500 | 2,031 | 2,023 | 88 | — |
| Dense polynomial, two variables, 66 terms | 148 | 148 | 327 | 259 |
| Dense polynomial, three variables, 84 terms | 259 | 244 | 593 | 259 |
| Mixed rational, 4/4 terms | 101 | 69 | 229 | — |
| Dense rational, 35/20 terms | 103 | 142 | 659 | — |
| Smirnov–Zeng Eq. 28, 248/180 terms | 306 | 451 | 1,224 | — |
| box2l rank 1, 38/8 terms | 80 | 79 | 282 | — |
| diamond3l rank 1, 60/45 terms | 119 | 139 | 501 | — |
| xbox2l2m rank 1, 7019/4586 terms | 8,300 | 12,387 | incomplete at 20,000 | — |
| tth2l_b16 rank 1, 1102/35 terms | 1,133 | 4,565 | 10,594 | — |

The sparse polynomial is `3*x^D*y^2+5*y^D*z^3+7*z^D*x+11`, with the same
degree bound 2000 throughout. The sparse rational is
`(3*x^D+5*y^D+7*z^2+11)/(z^2+3*z+7)`, with bound 512 throughout and variable
order x,y,z. This deliberately keeps the last-variable rational degree small
while increasing degrees handled by BMA. The other cases use bound 128;
their exact definitions are in the benchmark and archived IBP fixtures.

The degree-2000 polynomial uses about **1,146 times fewer samples** than
Automatic; the degree-500 rational uses about **23 times fewer**. These are
favorable sparse examples, not universal reductions. All four tested IBP
coefficients use more probes with HuMonagan; the nonplanar example does not
finish within the common budget. This comparison uses Symbolica's existing
Automatic and Smirnov–Zeng/BalancedZippel implementations. External FireFly,
FIRE and Kira solver timings were not rerun for this feature.

## Rational coefficients and validation

The Q fixtures use numerator
`(10^70+13)*x^55+17*x^3*y^2+19*y^7+23` and denominator one or `y^2+3*y+7`.
With degree bound 64 and maximum 20 primes, all 18 method/seed combinations
passed exact identity checks:

| Q input | Automatic | BalancedZippel | New method |
|---|---:|---:|---:|
| Polynomial | 165 | 165 | PolynomialBma: 33 |
| Rational | 165 | 165 | HuMonagan: 138 |

These counts include all primes and verification. Prime sequences differ by
method as described above; consult `probes_by_prime` in the CSVs. Use the
finite-field table for a comparison that holds the prime fixed.

All **57 reconstruction tests pass** (50 existing plus seven BMA tests).
Coverage includes degree-independent sparse sample counts, exact mixed and
dense rational reconstruction, missing samples, budget accounting, unsupported
geometry, individual bounds in a small field, misleading zero sequences,
vanishing normalization slices, and multi-prime lifting of 71-digit coefficients.
`cargo check --tests --examples` passes. Unfiltered Clippy hits the existing
`clippy::never_loop` error in `src/poly/factor.rs:11834`; targeted Clippy passes
with that lint allowed, retaining existing repository warnings. Both logs are
archived. Runs are serial under the available license.

The measurements use the **development profile with debug information disabled**
to compare sample counts. CSV timings are diagnostic, not release-speed claims.
Cases with more than 64 source terms use a cached-power evaluation oracle;
smaller cases use direct polynomial evaluation, identically for all methods
within each case. Exact verification is outside the timed reconstruction.

## Reproduction and archive

[The result archive](results/reconstruction/bma/) contains all 180 measurements,
including failures, summary ranges, input expressions and provenance, Q logs,
source and executable hashes, Cargo lockfile, and validation logs. Synthetic
definitions are in `examples/reconstruction_bma_benchmark.rs`. From the repository
root, restore fixtures and run:

```sh
mkdir -p target/reconstruction-external
tar -xzf benches/results/reconstruction/bma/inputs.tar.gz -C target/reconstruction-external
CARGO_PROFILE_DEV_DEBUG=0 cargo test --test reconstruction_bma --test rational_reconstruction -- --test-threads=1
CARGO_PROFILE_DEV_DEBUG=0 cargo build --example reconstruction_bma_benchmark --example reconstruction_stress_benchmark
MAX_PROBES=20000 BENCH_TIMEOUT=120 target/debug/examples/reconstruction_bma_benchmark '' 3

export SYMBOLICA_STRESS_BINARY=target/debug/examples/reconstruction_stress_benchmark
export BENCH_INPUT_DIR=$PWD/target/reconstruction-external/bma-inputs
export MAX_DEGREE=64 MAX_PRIMES=20 RECONSTRUCTION_REPEATS=3
BENCH_METHODS=Automatic,BalancedZippel,PolynomialBma python3 benches/external/run_q_stress.py target/reconstruction-external/bma-q-polynomial.csv bma_q_polynomial
BENCH_METHODS=Automatic,BalancedZippel,HuMonagan python3 benches/external/run_q_stress.py target/reconstruction-external/bma-q-rational.csv bma_q_rational
```

`archive_bma.py` checks full run counts and statuses before copying results.
It expects the finite-field output at `target/reconstruction-external/bma-comparison.raw.csv`,
its stderr at `bma-comparison.log`, and the validation logs at
`/tmp/reconstruction-bma-validated-{tests,build,check,clippy}.log`.
