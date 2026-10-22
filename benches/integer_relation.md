# PSLQ: issue #104

Measured on 2026-10-22 using the unchanged caller and both inputs from
[the issue's reproducer, revision 1e2c24e](https://gist.github.com/SecretGmG/46d0720ddbced1e6c909067c4aecfc43/1e2c24e606cdf95c934e51d6e8168dd842a15390).
The benchmark driver is [integer_relation.py](integer_relation.py); the twelve
measurements are in [integer_relation_results.csv](integer_relation_results.csv).

## Results

Whole-process wall times in seconds, including parsing, solver execution,
output verification and, for Mathematica, kernel startup:

| Input label | Previous optimization, one run | Two-level PSLQ, median [min, max] | Mathematica 13.2, median [min, max] |
| --- | ---: | ---: | ---: |
| 1500 | 84.13 | 6.223 [6.175, 6.324] | 9.732 [9.730, 9.780] |
| 1484 | 79.93 | 6.274 [6.225, 6.324] | 9.778 [9.680, 9.834] |

The final implementation and Mathematica were each measured three times per
input, serially on the same pinned logical CPU, alternating engine order. The
previous implementation's measurements were single observations on the same
machine and CPU, so the roughly 13-fold improvement is less controlled than the
final repeated engine comparison. An unmodified local baseline took 197.44 s
for input 1500 in one earlier run.

Two-level PSLQ takes about 64% of Mathematica's whole-process time on these
inputs. The locally installed engine is **13.2.0 for Linux x86 (64-bit), December
12, 2022**; the issue reports Mathematica 15 on different hardware. These results
do not measure Mathematica 15 or establish a general algorithm performance ratio.

All twelve outputs equal the published primitive coefficient vector, with
positive target coefficient, coefficient gcd one and maximum component at most
10^12. The driver independently checks the absolute and target-relative
residual bounds using exact Python `Fraction` arithmetic on the original closed
decimal inputs. The reference vector is used only after each solver exits.

### Matrix refactor check

Replacing the private native matrix with `Matrix<FloatField<F64>>` and moving
the helpers into `Pslq` and `Block` was checked against the saved binary from the
engine comparison above. The two binaries ran serially on CPU 2, alternating
their order, three times per input. Whole-process wall times in seconds:

| Input | Before refactor, median [min, max] | Matrix refactor, median [min, max] |
| --- | ---: | ---: |
| 1500 | 6.273 [6.273, 6.473] | 6.374 [6.273, 6.524] |
| 1484 | 6.323 [6.223, 6.323] | 6.324 [6.273, 6.674] |

The medians differ by 1.6% and less than 0.1%, with overlapping observed ranges.
All twelve outputs passed the same exact checks.
The refactored binary's SHA-256 is
`d658d05852556725d6e581568e0f3fd58814b1b0c465f9035834223e8dc544f7`.

### Final PR branch

The PR branch was based on `a2051681`, preserving the upstream fix that accepts
relations found by initial Hermite reduction. Repeating the same two inputs
three times on CPU 2 gave the following whole-process times:

| Input | Median [min, max], seconds |
| --- | ---: |
| 1500 | 6.324 [6.272, 6.324] |
| 1484 | 6.273 [6.223, 6.275] |

All six results matched the published vector and passed the exact residual
checks. The binary SHA-256 was
`11f06b85e5f7b2a2c7c018661394a69f30991132e61c1f72a7664b2f758171bd`.

## Workload and environment

- 61 real inputs: one target and 60 basis values. Labels 1500 and 1484 mean 1495
  and 1479 actual target digits; all basis strings have 1800 digits.
- Native caller: 6312-bit MPFR inputs, 250000 iterations, coefficient cap 10^12,
  gamma 7/5, and the unchanged request tolerance.
- Mathematica: the original `FindIntegerNullVector` call, requested working
  precision 1900 decimal digits, original input precision and zero test, and
  the reproducer's one-thread launcher. Its public options and precision model
  differ from Numerica's; neither caller was modified to align algorithms.
- AMD Ryzen 9 5900X, Linux x86_64, logical CPU 2, internal thread counts one.
  Frequency boost was enabled; background load and SMT neighbors were not
  controlled. Builds and other tests were outside the timed comparison.
- Rust 1.99.0, Numerica 3.0.1, GMP/MPFR backend (`rug` 1.30.0,
  `gmp-mpfr-sys` 1.7.1). Standalone release build: opt-level 3, codegen-units 1,
  fat LTO, debug assertions disabled. The baseline and both optimizations use
  the same caller and build profile.
- Binary SHA-256 for the Mathematica comparison:
  `7e73d327d85caf816c4ef236f5c980720c93cda83f77f56d86926aa0436b9fea`.

The byte-identical request hashes, enforced by the driver, are:

| Request | SHA-256 |
| --- | --- |
| request-1500.json | `58bc5f61d0e24c54cad864d8fe269a6d1019ad103e205330d021ae3ee6db2dd7` |
| request-1484.json | `0a1ea3e15a5ff6816ead654cd82b62f842fed9c45110aed29da5804ef7aefcdb` |

## Implementation

For scalar inputs with at least eight entries and at least 512 bits of
precision, the solver uses a native-double matrix to choose reductions. Exact
integer block transforms periodically update the high-precision matrix, while
the relation coefficients remain exact and the transformed input vector stays
at high precision. This avoids most 6312-bit matrix arithmetic. The approach
follows the two-level construction described in sections 3–4 of
[Bailey, *The two-level multipair PSLQ algorithm* (2025)](https://www.davidhbailey.com/dhbpapers/pslqm2-alg.pdf),
using the existing single-pivot iteration.

The solver verifies every accelerated candidate against the original inputs at
additional precision. Unsafe native exponents, quotients or transform growth
resume the full-precision solver with the remaining iteration budget; successful
fallback candidates receive the same independent verification. Small problems,
fixed-precision floats and interval types retain the full-precision path.

The full-precision path also uses cached pivot scores, contiguous exact
coefficient columns, short quotient rounding with an ambiguity fallback, and
exact small-integer MPFR multiplication. Regression coverage includes rounding
boundaries, mixed operand precision, coefficient and iteration limits, repeated
block refreshes, extreme exponents and misleading low-precision relations.

Validation passed with the following commands:

```sh
cargo test -p numerica --lib --tests
cargo test -p numerica --no-default-features --features integer-malachite,float-mpfr \
  --test integer_relation --test integer_relation_multilevel --test integer_scalar_product
cargo test -p numerica --no-default-features --features integer-malachite,float-astro \
  --test integer_relation --test integer_relation_multilevel --test integer_scalar_product
```

On the clean PR branch based on `a2051681`, the default suite contains 237 tests;
the targeted alternate-backend runs contain 20 and 14 tests. The PR preserves
the upstream check for convergence after initial Hermite reduction and tests
that behavior in both solver paths with zero- and one-pivot budgets.
One multilevel coefficient-limit test requires MPFR: Astro's
existing truncating integer conversion reaches the iteration limit instead on
that input, including with acceleration disabled. Its six other multilevel
tests pass, including independent exact residual checks and repeated refreshes.

## Reproduce

Download every file from the pinned gist revision into a directory outside the
workspace. In its `Cargo.toml`, replace only the `numerica` dependency source
with the absolute path to this checkout's `lib/numerica`, preserving
`default-features = false`, `integer-gmp`, `float-mpfr` and all profile settings.
Build before timing:

```sh
cargo build --release --manifest-path /path/to/reproducer/Cargo.toml
python3 benches/integer_relation.py \
  --reproducer /path/to/reproducer \
  --native /path/to/reproducer/target/release/d6-pslq-reproducer \
  --wolfram /path/to/WolframKernel \
  --cpu 2 --repeats 3 --output /path/to/new-results
```

Choose a permitted CPU and a fresh output directory. Omit `--wolfram` to measure
only Numerica. The driver saves each result and log, raw wall/user/system times,
input and binary hashes, and median/minimum/maximum wall times. It requires
Python 3 and Linux `taskset`; Mathematica requires a licensed installation.
