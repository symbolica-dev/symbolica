# Floating coefficients with error estimates

Floating coefficients carry a separate absolute error estimate for each complex
component. The center retains its arbitrary binary precision. Each error uses an
eight-byte `Magnitude` with 32 significant bits and an `i32` binary exponent.
The error estimate's precision does not limit the center's precision.

These are numerical error estimates, not certified interval enclosures. They do
not record correlations between repeated inputs. In particular, subtracting an
uncertain coefficient from itself retains uncertainty. Exact rational arithmetic
continues to use exact zeros and units; floating zeros and units need not be
algebraic identities. Raising working precision does not restore lost accuracy.

## Python

The existing `evaluate` and `evaluate_with_prec` interfaces return ordinary
numerical centers. Use `evaluate_with_error` to retain and inspect uncertainty:

```python
from symbolica import E, S, ComplexFloat, Magnitude, TrackedComplexFloat

x = S("x")
a = TrackedComplexFloat(ComplexFloat("1", precision=160),
                        Magnitude(0.1), Magnitude())
expression = E("1.0``1") * x

result = expression.evaluate_with_error({x: a}, 60)
assert result.real_error.to_float() >= 0.21
assert result.imaginary_error.is_zero()
center = result.value                 # An ordinary ComplexFloat
stored = result.to_expression()      # Preserves centers, precisions and errors
assert stored.get_error() == (result.real_error, result.imaginary_error)

evaluator = expression.evaluator([x])
optimized = evaluator.evaluate_with_error([a], 60)[0]
```

Both methods accept numerical expressions and ordinary numeric scalars as well
as `TrackedComplexFloat` inputs. Ordinary floats acquire a rounding error estimate;
tracked inputs preserve their existing errors. Tracked evaluation uses an
interpreter even when ordinary evaluation uses JIT compilation. The optimized
Python evaluator caches the converted tracked evaluator for repeated calls at the
same precision. Unsupported uncertainty-aware callbacks raise an error rather
than silently using a callback that discards uncertainty.

`TrackedComplexFloat.precision` is the pair of working precisions in **bits**.
`decimal_precision` estimates relative decimal digits for each component and is
`None` for a zero center. `accuracy` estimates absolute decimal digits. These
quantities can be negative or nonfinite. `is_real()` requires both an exactly zero
imaginary center and zero imaginary error.

`Magnitude.to_float()` rounds upward and returns `None` for a finite nonzero
magnitude outside the Python float range. Explicit stored infinity and NaN return
their corresponding float values. `float(magnitude)` raises `OverflowError` for
an out-of-range finite value. The `significand` and `exponent` properties preserve
the full representation:

```python
tiny = Magnitude(1, -50000)            # 2**-50000 without underflow
assert tiny.to_float() is None
assert Magnitude.from_parts(tiny.significand, tiny.exponent) == tiny
```

## Rust evaluators

Evaluator construction retains floating coefficients in distinct constant slots
with their original error and precision metadata. Optimization, composition,
merging, vectorization and serialization preserve those slots. Exact rational
constants remain rational. Select a tracked evaluation domain and use
`try_map_coeff_with_prec` to resolve the floating constants at a requested working
precision; plain domains explicitly receive their centers.

The mapping closure converts the exact constants. Floating coefficients resolve
through `EvaluationDomain::try_from_complex_tracked_float`, independently of that
closure. Do not use a conversion closure to apply an algebraic transformation to
all coefficients; transform the source expression instead. Convert to the final
tracked domain after performing optimizations that require a default value.
Instruction exports list original floats in `floating_coefficients`; their
placeholder entries in an exact evaluator's constant array are not exact zeros.

## Text, pickle and binary persistence

Ordinary formatting remains readable. File formatting, Python `repr`, and
`format_plain()` use a lossless numeric literal:

```text
0f[real_bits:real_error_hex:real_center_hex;imag_bits:imag_error_hex:imag_center_hex]
```

The center bytes use the canonical binary float codec, and the error bytes use
the canonical eight-byte `Magnitude` codec. The literal preserves unequal
component precisions, uncertain imaginary zeros, exact floating units, extreme
exponents, NaNs and infinities. One complex literal avoids introducing arithmetic
rounding while reconstructing a number. The readable decimal backtick suffixes
remain accepted for input and optional human formatting; they do not encode all
stored metadata losslessly.

Expression pickles use a `SYMPKL\x01` envelope containing a checked atom export and
the required symbol state. They can restore expressions in a fresh process with a
different symbol registration order. Legacy raw-expression pickles are rejected;
recreate them using an environment that can read their original representation.
Saved evaluators also have a new explicit format version and must be rebuilt from
their expressions when upgrading from an older format. Checked atom/state exports
use versions 7/12 and reject all earlier versions before reading their payloads.
Atom/state versions 6/11 were used by both main's untracked compact layout and an
older tracked layout, so these identifiers cannot distinguish the incompatible
representations. To migrate an older export, read it with its original version
and export lossless expression text for the current version.

## Numerical propagation

Integer powers include finite perturbation cross terms and result rounding.
Elementary transcendental functions account for finite perturbations, including
stationary points such as `cos(0)` and `cosh(0)`. Tangent and inverse hyperbolic
tangent detect uncertain neighborhoods reaching their poles. `atan2`, and hence
complex `log`, cover principal-argument jumps across the negative real axis.
Square root retains its nonlinear handling at zero and across complex branch
cuts. Bounds can deliberately become large or unbounded near singularities.

For small argument errors at fixed order, gamma, erf, polygamma, polylog and
Bessel J/Y/I/K use analytic first derivatives. A derivative `u + i*v` maps
component errors `dx, dy` to `|u|*dx + |v|*dy` and `|v|*dx + |u|*dy`, without
narrowing them to binary64. A local second-derivative allowance retains curvature
in stationary output directions. Integer-order Bessel and polylog limits at zero
are evaluated explicitly; real inputs stay real, and purely imaginary erf inputs
stay purely imaginary.

Larger or poorly conditioned neighborhoods, uncertain order parameters, zeta,
and unavailable analytic derivatives retain the extra-precision numerical-probe
fallback. Pole, branch-cut and broad-oscillation checks remain in place. The
curvature allowance and the probe fallback are numerical estimates, not certified
interval bounds.

The implementation still relies on backend arithmetic and precision-based
roundoff estimates. It does not certify every intermediate rounding operation or
the domain of every perturbed input, and it does not track correlations.
