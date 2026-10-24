# Magnitudes for tracked coefficient errors

`ErrorPropagatingFloat` stores its absolute error in `float::Magnitude`, an
eight-byte binary magnitude with a 32-bit significand and a signed 32-bit
exponent. The significand precision is sufficient for an approximate error
estimate; the extended exponent range matches the needs of arbitrary-precision
coefficients without making arithmetic repeatedly convert to logarithms.

## Representation and arithmetic

A positive finite magnitude represents
`significand * 2^(exponent - 31)`, with the top significand bit set. Zero,
infinity and unknown/NaN have explicit canonical encodings. Exact zero is the
all-zero eight-byte encoding. Positive underflow saturates to the smallest
positive magnitude, and overflow becomes infinity; underflow never invents an
exact zero.

Addition aligns significands and preserves discarded bits through outward
rounding. Multiplication uses a 64-bit product, and division uses a shifted
integer quotient and remainder. Normalization rounds upward. Powers of two
adjust the exponent, so arbitrary-precision roundoff floors need no logarithms
or exponentiation. Square roots and general powers work on the normalized
representation; constructing decimal accuracy and querying decimal
precision/accuracy may use logarithms.

`log2_ratio` subtracts exponents before converting them to floating point, so
precision queries avoid cancellation between enormous absolute logarithms.
One upward normalization changes the magnitude by at most approximately
`2^-31` relatively. Long operation chains can accumulate this inflation; the
32-bit significand deliberately trades error-estimate precision for exponent
range while retaining the previous error payload size.

## Central-value extraction and propagation

`RealLike::absolute_magnitude` has a native fallback and a full-range `Float`
override. MPFR exposes a bounded significand and separate exponent; Astro reads
its raw mantissa words, exponent and sticky remainder. Neither `Float` path
formats decimal strings or converts the full central value to `f64`.
Rationals use directed conversion, double floats retain the signed low limb, and
balls include their radius. Denominator estimates use a separate lower-magnitude
accessor; balls crossing zero have lower magnitude zero.

Error propagation uses magnitudes for addition, multiplication, division,
roundoff, cancellation, precision changes and mathematical derivatives.
Sine/cosine derivatives evaluate in the central-value domain before extracting
a magnitude, preserving argument reduction for values such as `2^1000 + 1`.
Other derivative formulas similarly retain central-domain precision or operate
directly on magnitudes. Rational scaling converts the rational into the central
value's domain before extracting its magnitude; this preserves the range of
`Float` at the cost of a temporary conversion.

## APIs and serialization

`get_error_magnitude` and `new_with_error_magnitude` expose the stored error
without narrowing its exponent. `get_absolute_error() -> f64` remains as a
compatibility accessor with outward rounding and explicit saturation.
`Magnitude::to_f64() -> Option<f64>` reports `None` when a finite nonzero
magnitude is outside the finite nonzero `f64` range; callers that intentionally
accept saturation can use `to_f64_saturating()`. Internal
exact-zero tests, printing and serialization use the magnitude directly.
`is_zero_within_error` compares the central magnitude with the tracked error;
it is distinct from exact equality.

Each owned complex coefficient tracks real and imaginary uncertainty separately.
Its two magnitude fields occupy sixteen bytes in total, the same payload size as
two native `f64` errors; an untracked coefficient has neither field.

Packed expressions omit the imaginary value and error when both are exactly
zero. The compact float's imaginary payload length is zero when the component
precisions match, or four when only a distinct imaginary precision must be
stored. This saves thirteen or nine bytes respectively. The borrowed imaginary
view contains just the four-byte precision (borrowed from the real component
when shared); decoding reconstructs exact zero at that precision. Neither the
borrowed view nor the owned complex arithmetic type grows. An imaginary zero
with any nonzero error, including NaN, infinity, or errors outside the binary64
range, still uses the full payload. Signed zero is canonicalized as before.

Each stored magnitude occupies eight bytes. Tracked atom format version 6 and state
format version 11 reject previous external formats, including the hexadecimal
float payloads in versions 3 and 8, the initial binary formats 4 and 9, and
the untracked main proposal formats 5 and 10. Older readers cannot decode the binary
central values or shortened imaginary payloads. Rejection also
prevents mixing alternate encodings of equal atoms, whose equality and hashing
use packed bytes. Bincode uses the same versioned atom format when enabled.

## Special cases and identity tests

An exact zero stays exact through square root. For uncertain real inputs, square
root uses the nonlinear change toward zero,
`delta / (sqrt(x) + sqrt(x - delta))`, when `delta < x`, and is bounded by
`sqrt(delta)` at the origin. These real estimates concern perturbations for which
the real square root is defined; they do not certify domain membership. A complex
input centered at zero with nonzero uncertainty retains finite uncertainty in
both output components, including the possible imaginary part from negative real
inputs. The same protection now applies when a nonzero real center's uncertainty
crosses zero, or imaginary uncertainty spans both lips of the negative real axis.
The fallback bounds the supplied component-error rectangle by
`sqrt(|re| + delta_re + |im| + delta_im)` and retains the ordinary central root.
This bound is deliberately loose near the branch cut. Inverse, logarithm and negative powers report unbounded error when the
input uncertainty reaches their pole at zero. Away from zero, reciprocal and
division use the remaining denominator clearance, computed by downward-rounded
`Magnitude::sub_lower`, instead of relying only on a derivative at the center.
Multiplication includes the cross term `delta_a * delta_b`.

All geometric functions, including two-argument `atan`, now register real and
complex tracked evaluators. Their normalization uses the same tracked operations;
inverse functions leave the real fast path when the input uncertainty crosses
the real domain boundary.

Gamma, erf, zeta, polygamma, polylog and the four Bessel functions also support
tracked normalization and evaluation in binary64 and arbitrary precision.
Special-function estimates use symmetric probes with extra working precision,
including simultaneous argument and order perturbations. Responses are scaled
with `Magnitude`, so errors outside the binary64 exponent range survive. Exact
representable order tags remain exact, while uncertain tags and rational
conversion errors contribute to the result. Known pole and branch-cut
neighborhoods produce unbounded uncertainty. Conservative locality checks use
the argument size, order and distance to singularities to reject neighborhoods
where probes could miss oscillations. Complex erf additionally checks the
phase variation of its exponential derivative. These numerical estimates are not
certified enclosures; probes cannot bound every interior extremum or certify the
underlying special-function approximation.

Arithmetic adds rounding uncertainty to propagated input uncertainty. Underflow
to zero retains a bound from the input magnitudes, and native scalar subnormal
results have an absolute rounding floor. MPFR's abrupt underflow and Astro's
subnormal spacing also supply backend-specific floors. Exact zero still annihilates a finite
central value even when that value's uncertainty is unbounded.

Ordinary real computations retain an exactly zero imaginary component, including
evaluation of real constants and conversion to tracked binary64. A nonzero
imaginary component that underflows during conversion retains its uncertainty.
Mixed-precision plain Float multiplication and division round once; the plain
value's unit fast paths preserve the other operand's precision. An uncertain
floating unit still propagates its error through the tracking wrapper.

The `one()` factory constructs an exact unit, while conversion of a user-provided
floating one preserves its uncertainty. Symbolic normalization and algebraic
float-field zero/unit tests must not erase that uncertainty. The lower-level
`SingleFloat::is_zero` and `is_one` still inspect the central value: use
`is_exact_zero` for exact zero and `is_zero_within_error` for consistency with zero.
`Atom::check_zero` returns `Inconclusive` for an unresolved uncertain zero;
numerical samples cannot prove an identity. Treating stored floating centers as
exact requires the explicit `treat_floats_as_exact(true)` setting.

## Performance and limitations

### Production binary float payloads (2026-10-23)

Both MPFR and Astro now use the same canonical binary central-value format.
Tracked atom/state formats are 6/11; the independent main proposal uses 5/10. Packed coefficients write directly into their final
buffer; calculating their size does not allocate. Borrowed zero and finite
checks read the class byte without constructing a multiprecision value.
`Float::serialize()` returns binary bytes, `serialize_into()` appends to a
reusable vector, and `serialized_size()` returns the payload length.
`Float::try_deserialize()` and `SerializedFloat::try_to_float()` return errors
for malformed or unsupported input; their existing infallible counterparts
are for trusted payloads. Bincode and serde encode a Float as precision plus
binary bytes, and no longer accept the previous hexadecimal representation.

Atom import validates container boundaries, float metadata and binary payloads
before constructing borrowed views. The explicit stack avoids recursive
validation for deeply nested expressions. This checks float encodings and
structural boundaries and rejects reserved numeric-node flags; it is not a
comprehensive validator of every other coefficient domain. Old atom/state versions are rejected before their payloads
are read.

The standalone
[`binary_float_serialization` example](../lib/numerica/examples/binary_float_serialization.rs)
now compares the production implementation with an explicit hexadecimal
baseline. The prototype measurements below predate production integration;
full coefficient measurements are recorded separately.

The format stores a class/sign byte, a zigzag LEB128 signed 64-bit normalized
exponent E, and little-endian unsigned significand bytes M with all low zero
bytes removed. Its value is `sign * M * 2^(E - 8 * byte_count)`. The highest bit
of the last byte is set. Precision, error magnitude and payload length stay in
the enclosing coefficient metadata. Both zero signs encode as positive zero,
and every NaN encodes with the same unsigned tag. Positive and negative infinity
are distinct numbers, each with one encoding. Precision and error bounds remain
part of coefficient identity: different accuracy metadata intentionally produces
different coefficients, even when their central values agree.

On MPFR, Rug's `get_significand()` borrows the limbs without an intermediate
integer allocation. Writing each limb with `to_le_bytes()` avoids a
general-purpose digit-export loop and native-endian memory dumps. Decode uses
`mpfr_set_ui_2exp` for significands fitting a native unsigned long, or a single
`mpfr_set_z_2exp` otherwise, checking that the result is exact. Significands up
to four native limbs use a read-only GMP integer backed by aligned stack storage;
larger values use an owned temporary integer. [MPFR documents this integer/exponent assignment](https://mpfr.org/mpfr-current/mpfr.html#Assignment-Functions).
Small and medium significands avoid the temporary integer allocation. Reusable writer
buffers avoid the output allocation as a separate optimization.

Three sequential runs on CPU 2 of the same Ryzen 9 5900X used release LTO,
Rust 1.99.0, Numerica's default MPFR/GMP features and the system allocator.
Each run measured three 100,000-iteration samples with thread CPU time. Inputs
were dense multiprecision `1/3` values. The table reports the range of per-run
speedup ratios; both columns compare allocating APIs. Absolute timings varied
substantially between runs, but every run favored the binary encoding.

| Precision | Hex / binary payload | Encode speedup | Decode speedup |
| --- | ---: | ---: | ---: |
| 53 bits | 18 / 9 bytes | 4.0–4.4× | 2.7–3.2× |
| 256 bits | 69 / 34 bytes | 2.1–2.5× | 3.8–4.3× |
| 4096 bits | 1029 / 514 bytes | 6.3–7.2× | 4.4–5.9× |

The [timing CSV](binary_float_serialization.csv) includes median/minimum/maximum
process medians and the separately labeled reused-buffer writer. These are
central-value payload benchmarks, excluding precision/error metadata and full
atom operations; they are not an application-wide speedup estimate. Validation
checks 190 finite/special-value roundtrips and canonical re-encoding cases,
including precision boundaries, negative NaNs and exponents ±4000.

Astro borrows its raw limbs when encoding and constructs raw words when
decoding, without hexadecimal conversion. It can retain more significant bits
than its requested precision: for example, division at a requested 53 bits can
retain a full 64-bit word. Both those bits and the requested precision survive
a roundtrip. MPFR rejects that 64-bit value at precision 53 instead of rounding
it silently. Backend exponent limits also differ; exact values beyond the
destination backend's range are rejected. Astro supports exact subnormals,
including values requiring non-byte-aligned normalization. NaN signs are
discarded on export for both backends, and imports reject the former signed
NaN tag. This preserves a single representative across backends.

Class tags are `0` (canonical positive zero), `1` (positive finite nonzero),
`2` (negative finite nonzero), `3` (+infinity), `4` (-infinity), and `5` (NaN).
Tag `6` and all other unknown tags are rejected. Specials are exactly one byte. Finite exponent varints must be
minimal and fit a signed 64-bit value; the significand must be nonempty, have
no low zero bytes and have its highest bit set. Native word padding and
endianness do not appear in the format. Precision remains a separate `u32`.
Imports also reject nonminimal coefficient-length headers and redundant full
encodings of exact imaginary zero. An uncertain imaginary zero retains its
value and error payload, so canonicalization cannot erase uncertainty.
Magnitude metadata likewise accepts only normalized finite values and its one
assigned encoding for each of zero, infinity and NaN. Serde, bincode and atom
imports all use checked decoders. Trusted borrowed-view APIs still require
valid bytes supplied by the caller.

Tests cover exact wire fixtures, mixed-backend precision behavior, signed
zeros, infinities, NaNs, precision/word boundaries, exponent endpoints,
Astro subnormals, truncated and noncanonical payloads, format rejection,
tracked errors and omitted imaginary zeros. Existing arithmetic tests cover
square roots near zero, exactness checks and real-only computations.

```sh
CARGO_INCREMENTAL=0 cargo build --release -p numerica --example binary_float_serialization
NUMERICA_BENCH_CLOCK=thread taskset -c 2 target/release/examples/binary_float_serialization 100000
```

### Further optimizations after rebasing on the main proposal (2026-10-23)

The representation and codec changes are independently reviewable in
[PR 126](https://github.com/symbolica-dev/symbolica/pull/126). The tracking branch
builds on that proposal, preserving its additional per-component error metadata.
The [historical codec results](https://github.com/symbolica-dev/symbolica/blob/1e51900739f5e0f3271acfc9cba794f1833d8a18/benches/binary_float_storage.md) describe direct packed comparison,
small MPFR imports, whole-word Astro imports and the short-significand fallback.

The intermediate coefficient benchmark binaries were built from tracking snapshot
`bbc010eabd72a6d998b0d6448cab418faf20a0f7`, based on main proposal commit
`64a4870ee87ff5425767959418f9ff3925e8c665`. The next changes reject reserved
numeric-node flags on import and fix an order-dependent parser test; neither
changes the measured paths. The hexadecimal prototype example also received a
NaN baseline correction, which does not affect either harness. The later
allocation follow-up below changes coefficient paths and has separate results;
the comparison table here describes the intermediate snapshot.

The full comparison harness measures packed coefficient views against the former
comparator, including its shortcut that skips the imaginary component when real
parts differ. Each process uses an eight-pair real/complex mix, 50,000 iterations
and three internal samples; the table reports three process medians on CPU 2
using thread CPU time. These figures include view access, unlike the earlier
Float-payload-only comparison.

| Backend | Component precision | Real comparison speedup | Complex comparison speedup |
| --- | ---: | ---: | ---: |
| MPFR | 53 | 3.22x | 3.23x |
| MPFR | 256 | 5.96x | 5.69x |
| MPFR | 4096 | 10.93x | 8.62x |
| Astro | 53 | 3.39x | 3.73x |
| Astro | 256 | 4.09x | 3.25x |
| Astro | 4096 | 2.11x | 1.30x |

The [full comparison CSV](binary_coefficient_comparison.csv) records binary
hashes and process ranges. Reproduce with the
[historical comparison harness](https://github.com/symbolica-dev/symbolica/blob/1e51900739f5e0f3271acfc9cba794f1833d8a18/examples/binary_coefficient_comparison.rs)
and `NUMERICA_BENCH_CLOCK=thread`.

The separate [full operation CSV](binary_float_coefficients_final.csv) compares
`9d1854f3` with measured snapshot `bbc010ea`. This includes the upstream-main
rebase as well as the codec optimizations, so it does not isolate one change.
The unchanged `examples/real_float_coefficients.rs` harness ran three alternating
before/intermediate process pairs per backend, with 50,000 iterations, three internal
samples, `NUMERICA_BENCH_CLOCK=thread` and `taskset -c 2`. The table reports the
percentage change in median time; negative values mean faster. Real cases have
equal component precisions and an exact imaginary zero.

| Backend | Precision | Case | Construct | Decode | Add | Multiply |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| MPFR | 53 | Real | +3.2% | -18.3% | -5.7% | -12.5% |
| MPFR | 256 | Real | +4.4% | +30.0% | +11.9% | +20.1% |
| MPFR | 4096 | Real | -6.6% | +4.5% | +6.3% | +6.8% |
| MPFR | 53 | Complex | -1.0% | -35.2% | -7.0% | -7.5% |
| MPFR | 256 | Complex | +1.2% | -0.5% | +4.1% | +1.4% |
| MPFR | 4096 | Complex | -4.8% | +0.4% | -1.3% | +0.2% |
| Astro | 53 | Real | +6.1% | -14.8% | -5.3% | -7.9% |
| Astro | 256 | Real | +6.0% | -8.6% | -4.5% | -5.1% |
| Astro | 4096 | Real | -5.1% | -78.8% | -4.7% | -21.0% |
| Astro | 53 | Complex | +8.5% | -14.4% | -4.9% | -4.9% |
| Astro | 256 | Complex | +4.5% | -45.2% | -9.6% | -17.7% |
| Astro | 4096 | Complex | -3.1% | -88.1% | -7.2% | -13.1% |

Astro's 4096-bit real decode improves from 558.60 to 118.44 ns (4.72x), and
complex decode from 1053.31 to 125.11 ns (8.42x). MPFR's 256-bit real decode
regresses from 97.65 ns (process range 97.26–101.94) to 126.99 ns
(124.96–130.82); its addition also has nonoverlapping process ranges,
595.39–627.94 versus 678.92–697.80 ns. These regressions remain in the results.

Reconstructing an omitted imaginary zero is slower: MPFR at 256/4096 bits takes
about 43–45 ns versus 24 ns (+77–90%), and Astro at those precisions about
43–47 ns versus 30–34 ns (+35–58%). Astro's 53-bit real case with a distinct
imaginary precision likewise regresses in full decode, 80.38 to 103.50 ns
(+28.8%), and multiply (+7.5%). MPFR's 4096-bit complex packed copy rises from
31.64 to 37.87 ns (+19.7%) despite unchanged storage. The CSV retains all nine
operations, four cases, binary hashes and process ranges, including these
controls and any outliers.

All 216 before/intermediate comparisons have identical packed byte counts; this stage
does not change retained storage and RSS was not measured again. The machine's
one-minute load average ranged from 2.1 to 2.8. Thread CPU time
excludes scheduler preemption but not frequency or cache variation. Ranges are
the minimum and maximum of three process medians, not confidence intervals;
small changes should be interpreted accordingly. The backend build features and
allocator choices match the initial-stage setup below.

### Allocation follow-up (2026-10-23)

The [allocation follow-up CSV](binary_float_coefficients_allocation_followup.csv)
compares the protected `9d1854f3` binaries with tracking snapshot
`e9736eceddb63b077086db69af137ddb74fc4ef0`, based on main proposal commit
`eee2128090319be34661ec24937b168d8336a120`. This snapshot constructs exact zero
directly at its final precision and imports MPFR significands of at most four
limbs through a read-only stack integer. It includes the earlier upstream rebase;
the measurements do not isolate either allocation change. The later rebase onto
upstream `2667847c`, producing tracking `7ed95a55`, adds power-view inlining and
factor-buffer copying changes. Those normalization changes are not measured
here, although the float codec is unchanged.

The unchanged operation harness again ran three alternating process pairs per
backend, 50,000 iterations and three internal samples, pinned to CPU 2 with
thread CPU time. No comparison or RSS benchmark was repeated. The table shows
median time changes against `9d1854f3`; negative values mean faster. Real cases
have matching component precisions.

| Backend | Precision | Case | Construct | Decode | Add | Multiply |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| MPFR | 53 | Real | -2.0% | -35.9% | +3.9% | -9.4% |
| MPFR | 256 | Real | -5.2% | -48.2% | +0.3% | -14.8% |
| MPFR | 4096 | Real | +1.9% | -2.6% | +0.9% | +5.8% |
| MPFR | 53 | Complex | -11.2% | -30.9% | -2.9% | -4.8% |
| MPFR | 256 | Complex | -7.7% | -44.1% | -6.5% | -7.7% |
| MPFR | 4096 | Complex | -0.3% | +0.2% | +2.7% | +4.0% |
| Astro | 53 | Real | +2.7% | -25.4% | -5.7% | -4.3% |
| Astro | 256 | Real | +16.8% | -44.0% | -6.2% | -14.2% |
| Astro | 4096 | Real | +2.7% | -83.4% | -3.0% | -21.6% |
| Astro | 53 | Complex | +6.0% | -6.8% | +0.4% | -4.8% |
| Astro | 256 | Complex | +6.5% | -39.1% | -4.8% | -12.5% |
| Astro | 4096 | Complex | +4.2% | -87.6% | -6.8% | -13.4% |

The targeted regressions are addressed. Across both imaginary-precision cases,
exact-zero decoding takes about 12–13 ns on MPFR and 13–15 ns on Astro,
compared with 23–34 ns in the initial binary version and 43–47 ns in the
affected intermediate cases. MPFR 256-bit real decode is now 50.88 ns
(50.75–50.93), versus 98.22 ns (97.76–98.73): 1.93x faster. Its addition is
effectively flat, with overlapping ranges, and multiplication improves 14.8%.
Astro's 53-bit distinct-imaginary-precision decode also improves, 77.66 to
57.64 ns; its earlier regression is gone. Astro 4096-bit decode improves 6.02x
for real coefficients and 8.03x for complex coefficients.

Remaining regressions are retained. MPFR 4096-bit real multiplication rises
from 2871.55 ns (2836.75–2906.01) to 3038.16 ns (2938.01–3178.68), or 5.8%.
With distinct imaginary precision it rises 7.4%; that case also has a 14.9%
decode increase with a broad final range of 553.09–728.49 ns, and a 10.8%
addition increase. Astro construction is slower at 256 bits: 16.8% for matching
precisions, 19.0% for distinct precisions, 9.9% for uncertain imaginary zero,
and 6.5% for complex values. Its 53-bit construction increases by 6–10% in the
distinct-precision, uncertain-zero and complex cases. Smaller changes in packed
views and predicates remain in the CSV as well; these microbenchmarks do not
attribute every change to the float codec.

All twelve processes passed their embedded validation, with unchanged packed
sizes in all 216 comparisons. The one-minute load average ranged from 2.34 to
2.61. Binary hashes and all process ranges are recorded; outliers and the earlier
stage's results are preserved. Thread CPU time still includes frequency and
cache effects, and the ranges are not confidence intervals.

### Full-coefficient binary encoding: initial stage (2026-10-23)

These measurements isolate `f7fccdb6` → `9d1854f3`, the first production binary
encoding, using the identical `examples/real_float_coefficients.rs` harness.
They exclude subsequent canonicalization changes, serialization fast paths and
rebasing. The [complete timing CSV](binary_float_coefficients_timing.csv) records
all 216 comparisons, including real components with distinct imaginary precision,
uncertain imaginary zeros, borrowed views, predicates, repacking and packed copies.

Both checkouts used the same dependency lockfile, release LTO, one codegen unit,
native CPU flags and disabled incremental compilation. MPFR used default
GMP/MPFR/mimalloc features; Astro used `--no-default-features --features
integer-malachite,float-astro`, hence the system allocator. Compare each backend
with its own baseline. Three before/after process pairs ran sequentially on
CPU 2 with Linux thread CPU time. Each record is the median of three samples of
50,000 operations; the table uses the median of the three process records.
Ranges in the CSV are process minima/maxima, not confidence intervals. Load
averages over all runs were 1.37–2.63; thread CPU timing excludes descheduling,
but not frequency or cache variation.

The real input is a multiprecision `1/3`. Real cases have an explicit exact
imaginary zero at the same precision; complex cases use `2/7` as the imaginary
component. Construction includes cloning the owned input and dropping the
result. Decoding constructs the entire owned coefficient, while addition and
multiplication operate on fixed input atoms. Setup is outside the measured loops.
Precisions below are requested bits; validation compares actual backend precision,
central values and both error magnitudes. A speedup is before time / after time.

| Backend | Bits | Case | Packed bytes, before → after | Construct | Decode | Add | Multiply |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| MPFR | 53 | real | 35 → 26 | 2.14× | 2.04× | 1.57× | 1.74× |
| MPFR | 53 | complex | 65 → 47 | 2.73× | 2.59× | 1.84× | 1.71× |
| MPFR | 256 | real | 86 → 51 | 2.38× | 3.51× | 1.95× | 2.35× |
| MPFR | 256 | complex | 167 → 97 | 3.07× | 3.95× | 2.22× | 2.20× |
| MPFR | 4096 | real | 1047 → 532 | 3.34× | 5.55× | 3.51× | 3.20× |
| MPFR | 4096 | complex | 2089 → 1059 | 3.61× | 5.74× | 4.10× | 2.49× |
| Astro | 53 | real | 39 → 27 | 3.38× | 2.09× | 1.21× | 1.74× |
| Astro | 53 | complex | 73 → 49 | 4.36× | 2.97× | 1.28× | 1.59× |
| Astro | 256 | real | 87 → 51 | 3.79× | 2.93× | 1.29× | 2.14× |
| Astro | 256 | complex | 169 → 97 | 5.61× | 3.44× | 1.47× | 2.03× |
| Astro | 4096 | real | 1048 → 532 | 8.86× | 5.00× | 1.26× | 2.56× |
| Astro | 4096 | complex | 2091 → 1059 | 10.03× | 5.17× | 1.42× | 1.93× |

For scale, 4096-bit real decoding measured 3142.42 [3091.46–3194.19] →
566.34 [552.98–616.30] ns on MPFR and 2767.38 [2720.76–2846.70] →
553.76 [552.37–559.30] ns on Astro. The only median regressions above 5% were
53-bit omitted-imaginary decoding: MPFR same precision 21.44 → 23.43 ns,
MPFR different precision 21.72 → 23.39 ns, and Astro same precision
18.89 [18.87–19.20] → 28.88 [28.34–29.76] ns. These small absolute regressions
are retained in the results; complete coefficient decoding still improves.

Separate processes retained 500,000 freshly constructed atoms after warmup,
again in three before/after pairs. RSS increases include the containing vector,
16,000,000 bytes of Atom metadata, spare capacity and allocator overhead/caching.
Packed bytes exclude those costs. The [memory CSV](binary_float_coefficients_memory.csv)
records exact byte counts and ranges; observed RSS range widths were at most
65,536 bytes. MB below means 1,000,000 bytes; MiB means 1,048,576 bytes.

| Backend | Case | Bits | Packed payload, MB before → after | RSS increase, MiB before → after | RSS reduction |
| --- | --- | ---: | ---: | ---: | ---: |
| MPFR | real | 53 | 17.5 → 13 | 38.00 → 32.00 | 15.8% |
| MPFR | real | 256 | 43 → 25.5 | 62.00 → 46.00 | 25.8% |
| MPFR | complex | 53 | 32.5 → 23.5 | 54.00 → 38.00 | 29.6% |
| MPFR | complex | 256 | 83.5 → 48.5 | 108.01 → 70.07 | 35.1% |
| Astro | real | 53 | 19.5 → 13.5 | 38.21 → 38.15 | 0.2% |
| Astro | real | 256 | 43.5 → 25.5 | 61.04 → 45.78 | 25.0% |
| Astro | complex | 53 | 36.5 → 24.5 | 61.04 → 45.78 | 25.0% |
| Astro | complex | 256 | 84.5 → 48.5 | 106.88 → 68.67 | 35.8% |

Astro's 53-bit real RSS difference is within the observed page/cache variation;
its packed saving does not establish a resident-memory reduction. Other measured
collections save approximately 16–36% RSS. Allocation size classes make these
workload-specific results; packed byte savings are deterministic, RSS savings
are not. No 4096-bit retention collection was measured.

To reproduce, build in separate checkouts at the two revisions, preserving the
same lockfile. Save each executable before switching backend or checkout:

```sh
# Default MPFR/GMP/mimalloc build; save as mpfr-before or mpfr-after.
CARGO_INCREMENTAL=0 CARGO_PROFILE_RELEASE_DEBUG=0 cargo build --locked --release --example real_float_coefficients
# Astro/system-allocator build; save as astro-before or astro-after.
CARGO_INCREMENTAL=0 CARGO_PROFILE_RELEASE_DEBUG=0 cargo build --locked --release --no-default-features --features integer-malachite,float-astro --example real_float_coefficients
# For each backend, repeat these before/after invocations three times sequentially.
NUMERICA_BENCH_CLOCK=thread taskset -c 2 ./mpfr-before 50000 before_f7fccdb6
NUMERICA_BENCH_CLOCK=thread taskset -c 2 ./mpfr-after 50000 after_9d1854f3
# Repeat memory pairs for real_same_precision and complex, each at 53 and 256 bits.
NUMERICA_BENCH_CLOCK=thread taskset -c 2 ./mpfr-before memory 500000 before_f7fccdb6 real_same_precision 53
NUMERICA_BENCH_CLOCK=thread taskset -c 2 ./mpfr-after memory 500000 after_9d1854f3 real_same_precision 53
```

Use the saved `astro-before`/`astro-after` executables for the corresponding Astro
runs. The measured executables used
`CARGO_TARGET_DIR=/home/ben/.cache/symbolica-error-tracking-target`; using a shared
target directory requires copying each binary out before the next build.

### Omitting exact imaginary zeros (2026-10-23)

Compared `4a5fa06d` with `ed16694f`, using the identical
`examples/real_float_coefficients.rs` harness copied into both checkouts. Both
were built with Rust 1.99.0, the default MPFR/GMP/mimalloc features, release LTO,
one codegen unit, and incremental compilation disabled. Runs were pinned to
CPU 2 on an AMD Ryzen 9 5900X, using `NUMERICA_BENCH_CLOCK=thread`. Three
before/after process pairs ran sequentially; each process reported the median
of three 50,000-iteration samples. The figures below are medians of those three
process medians. The [complete timing table](real_float_coefficients_timing.csv)
also records their minimum and maximum, including outliers.

The real central value is a multiprecision `1/3`; the imaginary value is an
explicit exact zero. Packed atom sizes include the atom tag but exclude spare
buffer capacity and the owned Atom's metadata:

| Binary precision | Before | Same imaginary precision | Different imaginary precision |
| --- | ---: | ---: | ---: |
| 53 | 48 bytes | 35 bytes (−27.1%) | 39 bytes (−18.8%) |
| 256 | 99 bytes | 86 bytes (−13.1%) | 90 bytes (−9.1%) |
| 4096 | 1060 bytes | 1047 bytes (−1.2%) | 1051 bytes (−0.8%) |

The savings are thirteen or nine bytes per coefficient regardless of the real
mantissa's length. The uncertain-imaginary-zero and nonzero-complex controls
retain exactly their previous packed sizes. The owned arithmetic type remains
unchanged; these savings concern its packed representation inside expressions.

Separate Linux processes retained 500,000 freshly constructed atoms after
allocator warmup. The following RSS increases include 16,000,000 bytes of Atom
metadata, the containing vector, spare buffer capacity, allocator overhead and
caching. Three alternating before/after pairs produced identical deltas within
each cell; RSS is measured with page granularity. See the
[complete memory table](real_float_coefficients_memory.csv) for exact byte counts.

| Retained coefficients | Packed bytes, before → after | RSS increase, before → after |
| --- | ---: | ---: |
| Real, same precision, 53 bits | 24.0 → 17.5 MB | 38.00 → 38.00 MiB |
| Real, different precision, 53 bits | 24.0 → 19.5 MB | 38.00 → 38.00 MiB |
| Real, same precision, 256 bits | 49.5 → 43.0 MB | 70.01 → 62.00 MiB (−11.4%) |
| Complex, 53 bits | 32.5 → 32.5 MB | 54.00 → 54.00 MiB |

Here MB means 1,000,000 bytes and MiB means 1,048,576 bytes. The 53-bit packed
reduction does not reduce this collection's resident footprint, consistent with
allocator size classes absorbing the difference. The 256-bit collection saves
8,392,704 resident bytes (about 8 MiB). Serialized byte savings are deterministic;
resident savings depend on allocation layout and are not a universal percentage.

Selected same-precision real-coefficient timings:

| Operation | 53 bits, before → after | 256 bits, before → after | 4096 bits, before → after |
| --- | ---: | ---: | ---: |
| Construct atom, including coefficient clone | 199.9 → 191.2 ns | 239.7 → 233.3 ns | 898.8 → 910.9 ns |
| Decode entire coefficient | 197.5 → 158.9 ns | 354.9 → 360.5 ns | 3159.2 → 3130.6 ns |
| Add two atoms | 899.3 → 874.9 ns | 1244.9 → 1189.5 ns | 4815.7 → 4802.8 ns |
| Multiply two atoms | 951.5 → 871.7 ns | 1366.6 → 1426.0 ns | 8855.1 → 9005.8 ns |
| Copy packed atom | 17.9 → 19.4 ns | 18.2 → 21.0 ns | 30.5 → 31.6 ns |

Avoiding imaginary-zero text parsing reduces 53-bit full decoding by 19.5% and
multiplication by 8.4% in this benchmark. This is not a uniform arithmetic
speedup: 256-bit multiplication is 4.3% slower, and 4096-bit operations change
little. Borrowing a coefficient changes by less than half a nanosecond for these
real cases. Packed copies are about 1–3 ns slower, including unchanged complex
controls; no copy implementation was modified. Complex construction and
arithmetic medians mostly change by a few percent. One 256-bit complex control
had a large transient slowdown in the first after-run, retained in the ranges.
Thread CPU time excludes descheduling but cannot remove frequency, cache, or
allocator variation; small differences should not be treated as universal.

The harness can be rerun with:

```sh
CARGO_INCREMENTAL=0 CARGO_PROFILE_RELEASE_DEBUG=0 cargo build --release --example real_float_coefficients
NUMERICA_BENCH_CLOCK=thread taskset -c 2 target/release/examples/real_float_coefficients 50000 current
taskset -c 2 target/release/examples/real_float_coefficients memory 500000 current real_same_precision 53
```

### Follow-up against `107e2268` (2026-10-23)

The follow-up merges main through `81f08a88` and strengthens arithmetic and
branch-cut handling. These are paired release microbenchmarks against the
previous tracking commit `107e2268`, with identical harnesses, dependency lock,
`codegen-units = 1`, LTO, and native CPU flags. Both executables were pinned to
CPU 2. Timings use Linux thread CPU time to exclude scheduler interruptions on
a busy machine; CPU frequency and cache contention can still affect them.

`error_tracking_arithmetic` ran three alternating before/after invocations,
with three samples of 200,000 operations each (nine samples per entry). Values
are 256-bit MPFR floats with 65 decimal digits of input precision. The real
complex cases use explicitly exact imaginary zeros. Medians:

| In-place operation | Before, ns/op | After, ns/op |
| --- | ---: | ---: |
| Tracked add | 57.3 | 57.6 |
| Tracked subtract | 57.9 | 62.5 |
| Tracked multiply | 94.2 | 102.8 |
| Tracked divide | 200.0 | 137.7 |
| Real complex multiply | 358.3 | 194.6 |
| Real complex divide | 928.3 | 235.0 |
| General complex multiply | 612.2 | 649.6 |
| General complex divide | 1319.1 | 1280.7 |

Direct division avoids a reciprocal temporary and its separate central multiply.
Real complex assignment reuses the real component and avoids the squared
complex denominator. These changes improve the measured real complex operations
by 1.84–3.95× and tracked division by 1.45×. Stronger propagation also costs work:
tracked multiplication is 9% slower, subtraction 8%, and general complex
multiplication 6%. Plain Float multiplication/division changed only about 1–2%,
so this measurement does not establish a material plain-Float speedup.

The same CPU clock measured magnitude multiplication at 3.86 → 2.78 ns and
division at 4.10 → 2.59 ns (median of five runs of two million operations).
Addition measured 4.93 → 5.29 ns. Magnitude arithmetic remains more expensive
than native `f64`; these changes retain its eight-byte representation and wide
exponent range.

`error_tracking_allocations` installs forwarding GMP callbacks in its own
process and counts only warmed arithmetic loops. Across 200,000 operations:

| In-place operation | GMP allocation requests/op, before → after | Requested bytes/op, before → after |
| --- | ---: | ---: |
| Tracked multiply | 0 → 0 | 0 → 0 |
| Tracked divide | 1 → 0 | 40 → 0 |
| Real complex multiply | 6 → 2 | 240 → 80 |
| Real complex divide | 10 → 2 | 400 → 80 |

No reallocations occurred in these loops. Requested bytes are cumulative
allocation traffic, not peak/live memory, and exclude Rust allocations. These
benchmarks measure isolated operations, not whole-expression performance.
Set `NUMERICA_BENCH_CLOCK=wall` to use the original wall-clock measurement.

### Initial magnitude implementation

The measurements below were recorded for the initial magnitude implementation
at commit `8622c2a5`, before the subsequent special-case and identity fixes.

Release microbenchmarks on the local machine measured the following scalar
costs. `magnitude_arithmetic` reports the median of five runs of two million
operations, with native input values in `[1, 2)`. The ranges below cover repeated
benchmark invocations.

| Operation | Native `f64`, ns/op | `Magnitude`, ns/op |
| --- | ---: | ---: |
| Add | 0.91–0.93 | 4.71–4.84 |
| Multiply | 0.91–1.18 | 3.67–3.71 |
| Divide | 1.01 | 3.85–3.88 |

`error_tracking_arithmetic` measured tracked 256-bit MPFR values, using 100,000
operations per run and the median of three runs:

| Operation | Magnitude, clone/replace, ns/op | Magnitude, in-place, ns/op | Earlier `f64` error prototype, in-place, ns/op |
| --- | ---: | ---: | ---: |
| Add | 84.5 | 57.5 | 47.6 |
| Subtract | 80.6 | 60.6 | 47.2 |
| Multiply | 106.1 | 86.9 | 60.0 |
| Divide | 200.7 | 183.2 | 150.9 |

The clone/replace and in-place columns use the same new magnitude kernels. They
show a 1.10–1.47× improvement from reusing the central value's allocation. The
earlier prototype used an `f64` error field. Final in-place runtimes are
21–45% higher than that prototype. The wider error range therefore has a measurable
arithmetic cost, while retaining the eight-byte payload. These comparisons do not measure
the historical branch or application-wide performance.

The ignored `packed_float_benchmark` compares fixed and compact length fields
using the same magnitude payload. Two optimized runs of 200,000 writes measured:

| Coefficient | Fixed / compact bytes | Fixed ns/write | Compact ns/write |
| --- | ---: | ---: | ---: |
| Zero imaginary center with error, 53 bits | 60 / 47 | 299–302 | 223–224 |
| Complex, 53 bits | 83 / 70 | 388–390 | 406–418 |
| Zero imaginary center with error, 256 bits | 111 / 98 | 309–312 | 258–259 |
| Complex, 4096 bits | 2104 / 2093 | 1987–1989 | 1982–1992 |

Compact length encoding saves 13 bytes for the small coefficients and improves
zero-imaginary-center coefficient write time by 17–26%, with a 5–7% write-time cost for the small complex
case. Checking serialized exact zero directly took about 1.1 ns, compared with
59–60 ns when decoding a multiprecision value first. These are isolated operation
measurements, not end-to-end expression benchmarks.
These older fixtures carry imaginary uncertainty, so they do not exercise
omission of an exact imaginary zero.

Astro avoids its decimal-formatting `to_f64` conversion during error propagation,
but these timings cover MPFR and native scalar arithmetic. Re-run both examples
for the target backend and workload. Owned multiplication and rational division
also avoid unnecessary clones; the rational magnitude conversion described
above still creates a temporary central-domain value.

The magnitude exponent remains bounded by `i32`. Its endpoint saturation is
conservative for magnitude arithmetic, but the complete tracking model is not
a certified interval enclosure. It does not retain input correlations, and
several transcendental functions still use first-order propagation. Their
estimates can miss nonlinear effects when errors are large; near singularities,
a first derivative may be undefined or insufficient. Derivatives are evaluated using the central domain's precision and
range, and custom `RealLike` types using the native fallback retain its range
and precision limitations until they supply their own magnitude extraction. Rigorous
certification continues to require a separate interval or ball model.
