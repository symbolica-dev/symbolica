# Polynomial GCD and factorization performance handoff

This is an operational handoff for the single-core Symbolica/FLINT performance work. It records
the state on 2026-08-27, including code already on `dev`, isolated candidates, measurements,
unsuccessful experiments, and the next experiments to run.

The current priority order is:

1. dense univariate integer GCD at degree 64;
2. the one-variable integer polynomial product used by the factor benchmark;
3. one-variable integer factorization;
4. the remaining multivariate GCD and factorization losses.

Do not introduce runtime parallelism while pursuing these items. The comparison target is
single-core FLINT performance.

## Repository state at handoff

The main worktree is `/home/codexB/symbolica` on `dev`. It was clean when this handoff was
finalized. Use `git status --short --branch` and `git log -3 --oneline` rather than relying on a
hard-coded head or ahead count, since the commit containing this document necessarily changes
both. The latest integrated polynomial-performance commit is `a3f721c` (`Specialize heuristic GCD
for univariate integers`); later inequality-solver and handoff commits are unrelated to the
performance implementation.

Use author and committer `Ben Ruijl <ben@ruijl.ch>` for commits made for this work.

The important isolated worktrees are:

| Worktree | Branch/commit | State | Purpose |
|---|---|---|---|
| `/tmp/symbolica-univariate-modular` | `codex/univariate-modular-gcd`, `810875a` | clean | First useful modular univariate integer GCD candidate |
| `/tmp/symbolica-univariate-dense-div` | `codex/univariate-dense-zp64-gcd`, `ea09e01` | clean | Direct dense `Zp64` Euclidean image GCD on top of `810875a` |
| `/tmp/symbolica-univariate-dense-modular` | `codex/univariate-dense-modular`, `5a975f1` | clean | Larger dense image plus dense integer certificate experiment; probably reject |
| `/tmp/symbolica-univariate-borrowed-cert` | `codex/univariate-borrowed-divisor-cert`, based on `810875a` | uncommitted, unformatted, unbuilt | In-progress reconstruction of the fast checked dense exact-division certificate |
| `/tmp/symbolica-zippel-dense-image` | `codex/zippel-dense-image`, `c4748b6` | clean | Dense single-scale Zippel reconstruction experiment; reject |

All univariate candidates are based on `a3f721c`, so they need to be rebased or cherry-picked
onto the current `dev` before integration. The commits already have the required author metadata.

## Benchmark infrastructure and protocol

Shared inputs are in `benches/support/cases.rs`. Symbolica-only rows are in
`benches/symbolica_polynomial.rs`; paired Symbolica/FLINT rows are in
`benches/flint_comparison.rs`. The paired harness:

- configures Rayon and FLINT for one thread;
- validates inputs and outputs outside the timed region;
- warms both implementations;
- alternates execution order;
- reports the median Symbolica/FLINT ratio directly;
- is built with default features, including `faster_alloc`, GMP, native code generation, full LTO,
  and one code-generation unit.

The 1-, 2-, and 3-variable GCD and factorization entries are actual timed benchmark rows. Unit tests
around their constructors only validate fixtures.

Build against the installed FLINT 3.6.0 with:

```sh
export PATH=/nix/store/9pb4ikjdw4gp766ayxl6gg3b7hqm6ds4-rust-stable-with-components-2026-07-09/bin:/nix/store/vdaz6sk455r8sbpi7wzaf9vlz5i9yyvx-gcc-wrapper-15.2.0/bin:$PATH
export PKG_CONFIG_PATH=/nix/store/k1hb5mgjbaiq1sx65zpbd9a1sfy0jl6n-flint-3.6.0/lib/pkgconfig:/nix/store/2vh1aj9bwhab9blr6wfm8v4mxd7nw15w-gmp-with-cxx-6.3.0-dev/lib/pkgconfig:/nix/store/yzf04s5m4s9w70wfqij2gba3421gnll4-mpfr-4.2.2-dev/lib/pkgconfig
export LD_LIBRARY_PATH=/nix/store/k1hb5mgjbaiq1sx65zpbd9a1sfy0jl6n-flint-3.6.0/lib:/nix/store/sy8ark85hgjhhcr1ycgarf5w7ajw8jcc-gmp-with-cxx-6.3.0/lib:/nix/store/bdvnvbidl34l6pvng6587m6axbzwz9hi-mpfr-4.2.2/lib
cargo build --release --features flint_system_benchmarks --bench flint_comparison
```

Set `SYMBOLICA_LICENSE` from the task/user environment. Do not copy its value into source,
benchmarks, logs intended for commit, or this document.

The configurable dense degree-64 GCD comparison is:

```sh
taskset -c 8 env \
  SYMBOLICA_FLINT_BENCH_PAIRED=1 \
  SYMBOLICA_FLINT_BENCH_FILTER='GCD auto: dense 1 variables degree 64' \
  SYMBOLICA_FLINT_BENCH_SAMPLES=201 \
  SYMBOLICA_FLINT_BENCH_CSV=1 \
  GCD_BENCH_CASE=dense \
  GCD_BENCH_NVARS=1 \
  GCD_BENCH_DEGREE=64 \
  ./target/release/deps/flint_comparison-<hash>
```

Use sequential processes on an otherwise idle pinned core. Do not run multiple performance jobs at
once, even on different cores. For a decision-quality result, use at least five fresh processes per
binary and compare the median normalized ratio. A first full LTO link can take about eleven minutes.

## Retained optimizations on `dev`

### Integer and polynomial multiplication

`MultiplicationOptimizations.md` contains the detailed multiplication design. The important
retained changes are:

- `9f1bb90`: bounded thread-local reuse of large GMP integer allocations and better storage reuse
  for owned operations;
- `2cd9103`: fixed-width, Kronecker, and direct GMP dense integer multiplication paths;
- `c1df9cc`, `6c587a1`, `2095b10`, and `492d3bf`: specialized polynomial multiplication kernels,
  operation-context organization, large-input kernels, sparse output decoding, and dense-simplex
  multiplication;
- `0e42296`: checked dense exact-division path;
- `1efa82d`: specialized finite-field multiplication kernels.

The integer GMP accumulator kernels use fused multiply-add/subtract operations rather than
materializing a temporary product. The large-integer cache helps allocation-heavy paths, but it did
not by itself close the gap to FLINT; algorithm and data-layout differences remain more important.

### Resultants

The default field/ring entry point is `resultant`, which implements Lazard-Ducos. Brown remains
available explicitly as `resultant_brown`. Integer and rational modular reconstruction remains
available explicitly as `resultant_crt`; it is not the automatic default because the current CRT
path is not generally competitive enough to replace direct Ducos.

Retained shortcuts include zero/constant/linear/small-degree handling, the adjacent-degree Ducos
recurrence, checked dense divisions, coefficient-domain multiplication kernels, and verified CRT
termination. The zero-polynomial behavior now agrees with FLINT: a resultant is zero when either
input is zero, including the zero-versus-constant case that previously reached `0^0`.

### Multivariate GCD

The retained GCD work on `dev` includes:

- cheaper sparse sampling and backend benchmark controls (`10bce75`, `b8bea0a`);
- faster Zippel/Hu evaluation and interpolation (`d8efc31`, `9025108`, `5b0fa6a`, `ef1be23`);
- field-width and prime selection from coefficient and interpolation bounds (`1b8014e`, `0635d18`);
- faster modular reconstruction and no arbitrary initial prime skipping (`52f88ad`, `0bb8e63`);
- early reconstruction and early base-degree exit (`0e316f1`, `fb0634a`);
- reused input/sample metadata and evaluation buffers (`32d097b`, `4e582fa`, `96abdb4`,
  `63d99ab`, `d0e4769`);
- single-pass terminal evaluation and cached dense degrees (`d6d71b0`, `8d494d2`);
- bounded dense workspaces and sparse-image guards (`1728c1d`, `5155af4`);
- variable-order correctness checks and Horner replacement evaluation (`a65c55f`, `844a81b`);
- dense bivariate and univariate heuristic specialization (`72de929`, `a3f721c`).

These changes made most generated 3- to 8-variable cases faster than FLINT. They did not solve the
high-degree univariate case because that case eventually spends most of its time in modular image
GCDs and exact integer verification.

### Factorization

Retained factorization work includes algebraic factor reconstruction (`829d5c9`), the reorganized
integer factorization paths and benchmark infrastructure (`ef3b809`), full-domain sampling fixes,
and evaluated multivariate Hensel lifting (`e4712cf`). The active work should not touch the Hensel
backend until the requested one-variable product/factorization investigation is complete.

## Current measured performance

Ratios below are median `Symbolica / FLINT`; lower is better and values below one mean Symbolica is
faster. Small changes under about 3% should be treated as noise until reproduced across processes.

### GCD overview

| Case | Ratio | Interpretation |
|---|---:|---|
| dense 1 variable, degree 32 | `0.99` | parity on the current scalar heuristic path |
| dense 2 variables, degree 5 | `1.03` | small remaining loss |
| dense 3 variables, degree 7 | `0.20` | Symbolica about 5 times faster |
| generated 3- to 8-variable cases | generally `< 1` | Symbolica generally faster |
| generated high-height cases | about `0.5` | Symbolica about 2 times faster |
| PolyBench 5-variable uniform case #11 | `1.05`-`1.08` | small remaining Zippel loss |
| PolyBench case #140 | about `1.13` | code-layout-sensitive remaining loss |
| dense 1 variable, degree 64 on current `dev` | `2.225` | current primary loss |

### Low-dimensional factorization overview

The product is timed separately from factorization:

| Generated case | Product ratio | Factorization ratio |
|---|---:|---:|
| 1 variable, degrees 32/31 | `3.06` | `7.26` median; even minima are about `4.85` |
| 2 variables, degrees 10/9 | `0.91` | `0.64` |
| 3 variables, degrees 6/5 | `0.96` | `1.97` |

The one-variable factor input is the product of `(1+3*x1)^32-1` and
`(1-5*x1)^31+1`. Its multiplication loss is real and should be profiled independently before
attributing the entire factorization loss to the factor algorithm.

## Dense univariate degree-64 investigation

### Baseline and useful candidates

The clean baseline binary is `/tmp/univariate-scalar-screen-a3f721c`, SHA-256
`c955b0c618d96471f954f6fdc9179ce83f35a6b3688ae6a10284fe3720a9b834`.

Commit `810875a` adds `UnivariateModularGcdContext`, which:

- separates primitive contents and restores their GCD;
- computes normalized 63-bit modular images;
- rejects degree-dropping/unlucky primes and resets when a smaller image degree is found;
- incrementally reconstructs coefficients with integer CRT;
- certifies a candidate using exact division of both inputs;
- backs off geometrically after failed reconstruction probes;
- selects modular reconstruction for the high-work/high-degree univariate regimes while leaving the
  degree-32 scalar case unchanged.

Its source-matched binary is `/tmp/univariate-modular-810875a-source-matched-screen`, SHA-256
`a994881cc210b355318567a509c107c171f8197127b13e8343bbff23fc892354`.

Five-process measurements made before this handoff were:

| Degree | Current scalar ratio | `810875a` ratio | Relative Symbolica gain |
|---:|---:|---:|---:|
| 32 | `0.982` | `0.985` | unchanged/noise |
| 48 | `1.705` | `1.598` | about 6% |
| 64 | `2.232` | `1.824` | about 18% |
| 80 | `2.294` | `1.933` | about 16% |

A fresh 201-sample degree-64 run on 2026-08-27 gave `2.225` for the scalar path and `1.816` for
`810875a`, consistent with the earlier screen.

Commit `ea09e01` adds a bounded dense `Zp64` Euclidean image workspace. It converts the two integer
polynomials directly into dense Montgomery coefficient arrays, performs one leading-coefficient
inverse per Euclidean division instead of making every remainder monic, and normalizes only the
final image. Unsupported sparse or degree-dropped inputs use the unchanged fallback.

Its binary is `/tmp/univariate-field-ea09e014-screen`, SHA-256
`883aae3613ac511c76ffca3cc6c37467f05b59ee31559a06088317b816e53334`. A fresh degree-64 run gave
`1.716`, so this is a repeatable improvement over `810875a`, but it is still substantially slower
than FLINT.

### Profile of `810875a`

The original degree-64 LBR profile is `/tmp/univariate-modular-d64-screen.perf.data`. Approximate
phase accounting was:

| Phase | Symbolica share/time | FLINT comparison | Approximate contribution to total gap |
|---|---:|---:|---:|
| two exact-division certificate calls | 49%, about 0.324 ms | `_fmpz_poly_divides`, about 0.166 ms | about 53% |
| modular field GCD | 31%, about 0.201 ms | `_nmod_poly_gcd_euclidean`, about 0.130 ms | about 24% |
| integer-to-field conversion | about 5% | smaller in FLINT | secondary |
| prime generation | about 5% | smaller/amortized in FLINT | secondary |
| CRT merge | about 2% | not dominant | minor |

In the generic field GCD, repeated monic normalization and `FiniteField::inv` were visible. In the
integer certificate, generic quotient/remainder conversion created `MultiPrecisionInteger` values
from `i128`, used GMP subtraction/multiplication paths, and paid drop/allocation overhead. These two
areas, not CRT itself, are the primary opportunities.

The later dense-candidate profile is preserved as
`/tmp/univariate-dense-modular-d64-5a975f17.perf.data`. Re-open profiles with a compatible `perf`,
for example `nix shell nixpkgs#linuxPackages.perf -c perf report -i <file> --stdio`.

### Decisive `ea09e01` versus `1.325` profile

A final source-matched comparison used 20,001 benchmark samples, `cycles:u` at 999 Hz, LBR call
graphs, and sequential processes pinned to core 8. The measured rows were:

| Binary | Symbolica | FLINT | Ratio |
|---|---:|---:|---:|
| `ea09e01` | `0.617177 ms` | `0.360278 ms` | `1.713058` |
| `nonalias-608f7b7` | `0.479768 ms` | `0.360221 ms` | `1.331871` |

The entire `0.137409 ms` Symbolica improvement is accounted for by exact certification:

| Phase | `ea09e01` | best binary | Change |
|---|---:|---:|---:|
| two exact-division certificates | `0.3147 ms` | `0.1767 ms` | `-0.1380 ms` |
| dense field Euclid | `0.1699 ms` | `0.1734 ms` | unchanged/noise |
| integer-to-field conversion | `0.0396 ms` | `0.0396 ms` | unchanged |
| prime generation | `0.0280 ms` | `0.0291 ms` | unchanged/noise |
| CRT merge | `0.0118 ms` | `0.0102 ms` | secondary |

This confirms that the binary name is misleading: finite-field non-aliasing did not produce the
gain. Generic-certificate multi-precision drop glue and `MultiPrecisionInteger::from<i128>` cost
about `0.099 ms`; the dense context reduces analogous conversion/drop work to roughly `0.014 ms`
by retaining multi-precision storage and using fused GMP loops.

FLINT spends about `0.164 ms` in `_fmpz_poly_divides`, so the recovered dense certificate is close
to FLINT on that phase. After it is integrated, the approximate residual gap is field Euclid
`+0.040 ms`, modular conversion `+0.029 ms`, prime generation `+0.029 ms`, and certificate
`+0.012 ms`, partly offset by Symbolica's roughly `0.009 ms` faster CRT merge. FLINT's
`n_nextprime` is effectively free in this profile.

Primary artifacts are:

- `/tmp/profile-d64-ea09e014-callgraph-lbr.perf.data` and
  `/tmp/profile-d64-nonalias-608f7b7-callgraph-lbr.perf.data`;
- corresponding `*.stdout`, `*-callgraph.report.txt`, `*-callgraph-flat.report.txt`, and
  `*-run.annotate.txt` files under `/tmp`.

## Experiments that did not justify integration

### Dense images plus dense integer certificate (`5a975f1`)

This adds about 494 lines on top of `810875a`, including dense image buffers, a fixed-length dense
CRT merge, and a private dense integer quotient/remainder certificate. It passed the focused and
full GCD test groups, but its extra gain over `810875a` was too small:

| Degree | Current scalar | `810875a` | `5a975f1` |
|---:|---:|---:|---:|
| 48 | `1.710` | `1.598` | `1.546` |
| 64 | `2.228` | `1.824` | `1.807`-`1.783` |
| 80 | `2.302` | `1.933` | `1.962` |

It regressed degree 80 relative to the simpler modular candidate. The local helper also consumes a
large numerator while cloning/converting the divisor to a fresh multi-precision value for each
quotient. Do not merge this patch as-is. If dense certificate work is revisited, implement it as a
private operation context and use borrowed-divisor/reused GMP storage rather than adding methods to
`Ring`.

Binary: `/tmp/univariate-dense-modular-screen-5a975f17`, SHA-256
`d2d12bcd8e801d25df1ca8c172dc2f026cce40614b885cad04dc06cd021bad3f`.

### Borrowed versus by-value finite-field calls

A 2026-08-27 reconstruction changed four calls in the private dense field kernel from borrowed
`RingOps<&FiniteFieldElement<u64>>` overloads to the by-value overloads. Five fresh degree-64
processes produced ratios from `1.708` to `1.713`, which is indistinguishable from `ea09e01`.

Therefore the preserved `1.325` experimental binary described below was not accelerated merely by
better reference alias analysis at these call sites. The source change was reverted and the
`ea09e01` worktree is clean.

Rejected binary: `/tmp/univariate-field-byvalue-screen`, SHA-256
`b59481951bbd38351eaeb2de97c097f7d983ad4aa245f353f1a5e5b850cd3c42`.

### Other transient univariate experiments

Several transient builds were preserved as binaries, but their source edits were not committed.
Treat them as evidence and reverse-engineering aids, not merge candidates:

| Binary | Degree-64 ratio | Result |
|---|---:|---|
| `/tmp/univariate-field-q1-primes-69d6f235-screen` | `2.091` | rejected; worse |
| `/tmp/univariate-field-primes-3e8ebf6-screen` | `1.626` | promising prime-generation/kernel combination |
| `/tmp/univariate-divconquer-eb3f3cf-screen` | `2.736` | rejected; divide-and-conquer attempt much worse |
| `/tmp/univariate-field-nonalias-608f7b7-screen` | `1.325` | best experimental result, but source not preserved |

The best binary has SHA-256
`e4bca4030cebec6377e70e21009d80867d13847a64bcde0d252fc0776250adf5`. Machine-code comparison
shows that its dense Montgomery Euclidean inner loop is structurally similar to `ea09e01`; the
four by-value call experiment did not reproduce the gain. The decisive profile above shows that
the private checked exact-division context, not the field image loop or prime selection, accounts
for the gain. Do not infer experiment contents from transient binary filenames.

The `field-primes` binary has SHA-256
`b83b169d66f517456cdd6f6f6c80fdc92418ca6db7fc9a8b59fdb43ceb85241d`. Prime generation was only
about 5% of the `810875a` profile, so its approximately 5% improvement over `ea09e01` is plausible;
it cannot explain the full `1.325` result.

### Reverse-engineered certificate from the `1.325` binary

Static symbol and machine-code comparison after the initial handoff draft found a much stronger
lead than the binary's `nonalias` filename suggests. The best binary contains a private
`DenseUnivariateIntegerDivisionContext`; the `field-primes` binary does not. Its checked
`try_div` path uses dense `Vec<MultiPrecisionInteger>` storage and a sparse
`Vec<(usize, MultiPrecisionInteger)>` quotient, with direct calls to GMP `__gmpz_tdiv_qr` and
`__gmpz_submul`. Its generated function is also appreciably smaller than the tagged-`Integer`
certificate in `5a975f1`.

The phase profile confirms that this pure-multiprecision checked exact-division certificate
explains essentially all of the improvement from `ea09e01` at about `1.713` to the best binary at
about `1.332`. Do not repeat the already-disproved four-call finite-field alias experiment.

A likely source reconstruction is a GCD-private operation context that:

1. converts the dividend and divisor coefficients to dense `MultiPrecisionInteger` arrays once;
2. walks leading degrees downward;
3. computes quotient and remainder with an owned numerator and borrowed divisor;
4. rejects immediately when the coefficient remainder is nonzero;
5. applies `MultiPrecisionInteger::sub_mul_assign` to the lower coefficients;
6. records only nonzero `(degree, coefficient)` quotient terms;
7. verifies the remaining low coefficients are zero before converting the quotient back to
   `Integer`.

`MultiPrecisionInteger` is re-exported as
`crate::domains::integer::MultiPrecisionInteger` and already supplies `div_rem_euc`, `div_rem_ref`,
`div_exact_owned`, and `sub_mul_assign`. If its public operations cannot reuse the numerator while
borrowing the divisor, keep a low-level GMP quotient/remainder helper private to the context. For
this certificate truncating division is sufficient because only a zero remainder is accepted.

The existing `DenseIntegerExactDivision` in
`lib/numerica/src/domains/integer/polynomial_kernels.rs` already uses pure multiprecision buffers,
but only for `assume_exact = true`. Reconstruction calls `MultivariatePolynomial::try_div`, which
uses `assume_exact = false`, so that kernel is not selected. Do not reuse the unchecked kernel as a
certificate without adding the remainder checks.

An in-progress source reconstruction is in
`/tmp/symbolica-univariate-borrowed-cert` on branch
`codex/univariate-borrowed-divisor-cert`, based on `810875a`. It is deliberately left uncommitted,
unformatted, unbuilt, and unbenchmarked. The current edits are:

- `lib/numerica/src/domains/backend/integer.rs`: adds
  `MultiPrecisionInteger::div_rem_owned_ref_assign`, which consumes the numerator as quotient
  storage, borrows the divisor, writes into a reusable remainder, calls GMP `mpz_tdiv_qr` with the
  quotient aliasing the numerator, and has a non-GMP fallback plus a backend test;
- `src/poly/gcd.rs`: adds `DenseUnivariateIntegerDivisionContext`, converts each divisor once,
  reuses one remainder scratch across both certificates, performs fused multiprecision
  subtraction/multiplication, retains only nonzero quotient terms, and falls back to generic
  `try_div` for unsupported layouts.

Before committing that worktree, run `cargo fmt`, compile both the normal GMP configuration and
`--no-default-features --features no_gmp`, and add direct tests for exact and inexact division,
inactive variables, output term order, and sparse/high-gap fallback. Inspect the non-GMP move and
assignment branch closely. The GMP numerator/quotient alias is intentional and matches Rug's own
internal `tdiv_qr` use.

The `field-primes` binary also contains a small `ModularGcdPrimeIterator` that yields known 64-bit
primes before falling back to `PrimeIteratorU64`. Binary data identifies the first two candidates
as `18_346_744_073_709_552_031` and `18_346_744_073_709_552_043`. Verify both with the repository's
prime code before adding them. The best `nonalias` binary appears to use the original prime
iterator, so the dense checked certificate and the known-prime improvement may be independently
composable.

### Dense single-scale Zippel reconstruction (`c4748b6`)

This experiment kept single-scale Zippel images dense and added roughly 700 lines. Focused tests
and all GCD module tests passed. On PolyBench 5-variable uniform GCD #11, five processes with 501
samples changed the median ratio from `1.0782` to `1.0639`, only a 1.32% normalized gain. This is
below the 3% threshold and does not justify the complexity. Do not integrate it.

Preserved branch: `codex/zippel-dense-image`. Frozen binary:
`/tmp/flint-comparison-zippel-dense-image-c4748b6-screen`, SHA-256
`89b6514a7d236e8d1673aa601b4a4c04f66a1397fbc6c6fc58de738c0286dca7`.

### Resultant alternatives

CRT-on-Ducos and Brown PRS remain useful explicit alternatives and correctness cross-checks, but
benchmarking did not support making either the general automatic entry point. Keep `resultant` as
the Ducos entry point unless new regime-wide measurements show otherwise. The sparse multivariate
resultant idea attempted earlier was not competitive because interpolation/reconstruction bounds
and coefficient work overwhelmed the sparsity benefit on the tested cases.

## Recommended next steps

### 1. Finish and validate the degree-64 dense certificate

The profile has confirmed the source-level cause. Continue from
`/tmp/symbolica-univariate-borrowed-cert`, review the uncommitted implementation described above,
and complete its focused correctness checks. It currently sits on `810875a`; combine it with
`ea09e01` only after it works independently so the gains remain attributable. Avoid expanding
`Ring` or adding another general ring trait.

Build one source-matched release benchmark, freeze and checksum it, then compare it against
`810875a`, `ea09e01`, and the frozen `1.325` binary. If it does not reproduce the certificate phase,
use the preserved LBR artifacts to compare direct `tdiv_qr`, `submul`, construction, and drop
nodes. The already-disproved finite-field alias hypothesis should not be repeated.

After the certificate is reproduced and combined with `ea09e01`, address the measured residual
items in order: raw finite-field Euclid, prevalidated/cheap prime selection, and direct GMP-limb
modular conversion. Keep raw dense Montgomery coefficients inside a private GCD image context,
load modulus/Montgomery constants once, and compare the emitted loop with FLINT's
`_nmod_poly_gcd_euclidean`. Do not introduce a global polynomial helper.

### 2. Guard the chosen GCD change

Before integration, run at least degrees 32, 48, 64, and 80 with five processes per binary. Degree
32 must remain on the fast scalar path. Also run:

- focused univariate modular GCD tests;
- the full `poly::gcd::tests` group;
- zero, constant, coprime, content, unlucky-prime, inactive-variable, and sparse/high-gap cases;
- the generated GCD matrix and selected PolyBench rows #11 and #140.

Keep the simple `810875a` commit if the more ambitious kernel cannot beat it by enough to justify
its code. It already gives a repeatable 16-18% gain at degrees 64-80 without changing degree 32.

### 3. Profile the one-variable product

Use the `generated factor product: dense 1-variable degrees 32/31` paired filter. Confirm the
current approximately `3.06` ratio across processes, then determine which multiplication kernel is
selected. Record:

- input and output term counts;
- coefficient bit-height distribution;
- dense-box size and collision count;
- whether fixed-width, direct GMP array, Kronecker, generic dense, or heap multiplication runs;
- time in coefficient multiplication versus result construction.

The static dispatch trace is already known for this exact fixture:

- both inputs have 32 dense terms; the result has 63 nonzero terms over degrees 1 through 63;
- the left coefficients are all tagged `Integer::Single`, while the upper half of the right input
  and most result coefficients require `Integer::Double`;
- fixed-width `i64`, mixed `i64`/`i128`, and bounded `i128` strategies reject the coefficient
  shape or conservative bound;
- Kronecker substitution is suitable but its current density gate rejects the 32-by-32 case
  because `product_count = 1024` while the packed-output threshold is `64 * 128 = 8192`;
- the large-array route rejects because the inputs contain no `Integer::Large` values;
- Symbolica consequently executes 1,024 generic coefficient multiply-adds, with roughly 244
  promoted GMP operations, whereas FLINT recognizes one variable and selects
  `_fmpz_poly_mul_KS`, performing one packed GMP multiplication.

The smallest candidate fix is in the existing `DenseIntegerMul` context: allow Kronecker
substitution for sufficiently large contiguous supports while preserving the stricter gate for
sparse supports. Benchmark nearby degrees, coefficient heights, and sparse/high-gap inputs to
guard the selector. This needs neither a global helper nor a new `Ring` method.

### 4. Profile one-variable factorization

After the product is improved, rerun `generated factorization: dense 1-variable degrees 32/31`.
Profile the remaining factor-only time. The old median was host-bimodal, so compare process medians
and minima and inspect algorithm selection. Do not count input construction, product formation, or
factor expansion in the timed region.

Potential areas to distinguish with evidence are square-free decomposition, modular factorization,
Hensel/reconstruction, repeated exact division, and coefficient conversion. Do not assume the GCD
fix alone explains the factor gap.

### 5. Resume the broader matrix

Once the primary one-variable losses are addressed, rerun the complete generated and PolyBench GCD
matrix, then the factor matrix. The next known GCD candidates are PolyBench #11 and #140; the next
known factor loss is the 3-variable generated case at about `1.97` times FLINT.

## Code-structure preferences to preserve

- Keep runtime algorithms single-threaded for this work.
- Prefer short-lived operation contexts that own reusable buffers and precomputed metadata.
- Do not grow `Ring` for one coefficient-domain special case.
- Keep evaluation-only operations outside the multiplication/division kernel interface.
- Comments should explain what a function computes and where it is used, not defend the chosen
  module structure.
- Preserve generic fallbacks when a specialized layout, degree, coefficient, or size bound is not
  satisfied.
- Benchmark input setup and correctness validation outside timed regions.
- Do not accept large specialized patches for gains below roughly 3% without stronger evidence.
