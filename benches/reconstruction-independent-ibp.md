# Sampled univariate factors and independent IBP families

Two independent IBP families now supplement the public `nb0` and `graph5`
tables and their coordinate controls. They exposed a probe-count regression
against scanned FireFly on the larger coefficients. Automatic reconstruction
now removes sampled univariate factors and reuses the transformed survey in
balanced Zippel reconstruction.

## Algorithm

The existing last-variable survey takes two independent slices. When their
denominators differ, their numerator and denominator GCDs can still expose
common univariate factors. A nonmonomial common factor triggers a second
survey of the other variables when the selector would otherwise choose pruned
Cuyt–Lee. The second survey uses the first survey's degree bounds and shares
one new intersection, avoiding repeated Thiele degree discovery. Monomial
factors retain their existing treatment.

The oracle divides out the sampled numerator factors and multiplies by the
sampled denominator factors. Corresponding transformations of the saved
univariate rows allow balanced reconstruction to reuse its initial slice and
first geometric rows. Symbolica performs the GCDs, exact polynomial divisions,
interpolation and final polynomial multiplication. Reconstruction receives no
source supports, factors, degrees or IBP metadata.

The factorization is a hypothesis. Factors are restored before the existing
full-dimensional checks and fresh-prime validation. Raw cached evaluations
remain unchanged, transformations are cleared on failures, and retries choose
fresh anchors. Roots that make division undefined are unusable probes. The
restored total degree remains bounded. Explicit reconstruction methods retain
their existing behavior.

## Independent inputs

The pinned [Ratracer benchmark generator](https://github.com/magv/ibp-benchmark/tree/f518de1f4f89cc716a31d5d9a9a9ba0a3b72f458)
provides the public solver configurations associated with the
[Ratracer paper](https://arxiv.org/abs/2211.03572). Kira 3.1 exports the equations
and Ratracer solves them into arithmetic traces. The selected integrals are:

| Family | Integral | Variable order |
|---|---|---|
| Massive planar two-loop box, `box2l` | `basis[1,1,1,1,1,1,1,-1,-1]` | `s23,s12,d` |
| Nonplanar two-loop box with two masses, `xbox2l2m` | `basis[1,1,1,1,1,1,1,-2,0]` | `t,s,mb2,d` |

The four coefficients with the largest optimized single-output trace
instruction sizes are selected before measuring reconstruction. This ranks
calculation size, which need not match the final expression size. The exporter
unfinalizes each split trace before optimization so that unused instructions
are actually removed. Manifests retain every coefficient's rank and original
master-integral label. The initial nonplanar target ending in `-1,-1` was
already a preferred master and was rejected as a trivial benchmark before
timing; the preparation script now checks this explicitly.

The authors' compatible FireFly fork reconstructs four reference expressions
jointly. Its vector probe count is input-preparation work and is not compared
with separate scalar reconstructions. Each reference expression is checked
against its trace at eight exact rational points with independently sampled
coordinates. These are probabilistic provenance checks, not symbolic IBP
identity certificates. Every benchmark result is additionally checked by an
exact polynomial identity against the fixed reference expression.

## Full rational reconstruction probe counts

Baseline: `d85dbb2d`. Counts include discovery, interpolation, lifting, failures
and verification. Each row reconstructs one coefficient independently.

| Family / rank | Numerator + denominator terms | Baseline | Current | FireFly default | FireFly scan | FIRE7 learned | rare |
|---|---:|---:|---:|---:|---:|---:|---:|
| Planar 1 | 38 + 8 | 115 | 83 | 98 | 147 | 244 | 130 |
| Planar 2 | 257 + 111 | 617 | 472 | 835 | 732 | 1,445 | 913 |
| Planar 3 | 153 + 60 | 367 | 323 | 449 | 501 | 1,037 | 554 |
| Planar 4 | 165 + 60 | 407 | 282 | 502 | 455 | 933 | 585 |
| Nonplanar 1 | 7,019 + 4,586 | 15,913 | 8,303 | 24,375 | 11,353 | 35,303 | 25,200 |
| Nonplanar 2 | 7,877 + 5,810 | 21,448 | 11,597 | 28,925 | 18,908 | 38,366 | 31,615 |
| Nonplanar 3 | 14,045 + 9,070 | 31,651 | 18,029 | 75,118 | 23,581 | 64,982 | 46,943 |
| Nonplanar 4 | 1,631 + 1,059 | 1,862 | 1,862 | 6,984 | 2,319 | 10,096 | 6,315 |

The nonplanar total decreases from 70,874 to 39,791 probes, a 43.9% reduction;
it is 29.2% below scanned FireFly. Symbolica uses fewer probes than all four
reference configurations on each of these eight coefficients. This is a
comparison of the selected cases, not a claim of universal superiority across
IBP reductions. Smaller cases favoring FireFly in the existing complete tables
remain part of the regression suite.

Each version's Symbolica counts are stable across seeds 1–3. Scanned FireFly
is also repeated with all three seeds; both FireFly settings are repeated for
the planar family. The other reference configurations use seed 1.

The adapters use the same expanded-polynomial cached-power oracle. Their
timings therefore measure reconstruction with this oracle, not an end-to-end
Kira/Ratracer reduction. Libraries retain their native prime and coefficient
confirmation policies, documented in the [scaling comparison](reconstruction-scaling-reference.md).
All references are actual implementations: standard FireFly, FIRE7's learned
Smirnov–Zeng path and `rare` 0.9.2's scaling implementation.

## Reproduction and validation

See the [build and preparation commands](external/README.md). The
[archive](results/reconstruction/independent-ibp/) retains pinned source
revisions, selected expressions, equation/trace hashes, exact trace checks,
native preparation commands, reference CSVs and reconstruction logs. The
Ratracer-compatible FireFly fork is built separately from the standard FireFly
reference. Upstream source checkouts are unmodified; the FLINT C++ header
overlay follows the authors' Makefile. The copied Kira export job omits only
the disabled `run_firefly: false` entry, which a build without FireFly rejects.

The new synthetic factor control falls from 449 to 140 full-Q probes. Its
modular regression test requires exact recovery within 160 probes, including
actual oracle-call accounting. Additional tests reject a false factor hidden
on both initial slices and verify that restored factors survive multiple-prime
coefficient lifting and support reuse.

All 44 reconstruction tests pass. All 3,319 `nb0` and 945 `graph5` coefficients
pass exact reconstruction and retain their probe counts, per-prime counts and
selected methods; totals remain 505,874 and 20,311. Clippy completes with the
existing `clippy::never_loop` allowance and the same 249 warnings. Changed Rust
files pass formatting checks. The pinned external build script and both exact
trace validators have also been executed successfully.

All 4,330 final benchmark runs pass their identity checks. The larger modular
FireFly controls, full-Q four-loop and amplitude inputs, and all four sheared
`nb0` controls retain their earlier probe counts and per-prime distributions.
The archive separates exploratory debug measurements and input-setup failures
from the final release results.
