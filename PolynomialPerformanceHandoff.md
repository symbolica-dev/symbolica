# Polynomial performance continuation handoff

This is the live continuation record for the single-core Symbolica/FLINT polynomial-performance
work. It was refreshed on 2026-08-27 after integrating exact-subproblem local Hensel bounds through
`1a8b4d7`.
Keep this file current whenever an experiment is accepted, rejected, or left partly complete. The
purpose is that another agent can resume without reconstructing decisions from chat history or
transient binary names.

## Working contract

- The target is single-core performance comparable to FLINT. Do not add runtime parallelism.
- The Rust comparison harness uses `flint3-sys` and FLINT 3.6.0. Prefer it to subprocess or Python
  comparisons because both implementations receive the same generated inputs in one process.
- Release/default-feature builds include `faster_alloc`; the default feature set in `Cargo.toml`
  contains `faster_alloc`, GMP, and native code generation. Do not accidentally benchmark a
  `--no-default-features` binary as the main result.
- Run paired timing processes sequentially on an otherwise idle pinned core. Do not run two
  performance measurements concurrently.
- Preserve generic fallbacks. Specializations must reject unsupported layouts, coefficient
  ranges, or size regimes without changing semantics.
- Use author and committer `Ben Ruijl <ben@ruijl.ch>` for this work.
- The Symbolica license is supplied by the user through `SYMBOLICA_LICENSE`. Never put its value in
  this file, source, committed logs, or benchmark commands saved in the repository.
- Small changes below roughly 3% need particularly strong, repeated evidence before their
  complexity is accepted.

## Resume here

The accepted product and factor candidates are integrated on `dev` through `1a8b4d7`. Do not
cherry-pick their old worktree hashes again. The integrated chain is:

| Integrated commit | Change |
|---|---|
| `418d7b0` | Select Kronecker multiplication for dense contiguous univariate integer products |
| `9c2d7b9` | Direct fixed-width Kronecker packing and native signed unpacking |
| `0710ce3` | Fixed-width coefficient statistics with unchanged GMP fallback |
| `598a628` | Extend the high-pressure factor-prime path through degree 64 |
| `9609704` | Use a bounded roughly 26-bit prime so modular factorization stays in dense `u64` |
| `f360be0` | Add total-degree 33, 64, and 65 factorization boundary benchmarks |
| `4a2b9c7` | Use guarded quadratic Hensel lifting with integer modular operation contexts |
| `4f3b591` | Bound losing DDF images and defer EDF until after modular-prime selection |
| `303381c` | Bound finite-field accumulators per output coefficient |
| `31d50a9` | Maintain the scaled linear-Hensel residual incrementally |
| `f367a8a` | Reconstruct exact Hensel subproblems at child-local coefficient bounds |
| `817226b` | Cover non-monic exact children and base-prime local precision |
| `ef0d716` | Preserve the root linear/quadratic strategy throughout exact subtrees |
| `1a8b4d7` | Cover local LLL recombination with more than ten modular leaves |

The latest measured full-LTO candidate binary is
`/tmp/flint-comparison-factor-local-subtree-residual-v2-screen`; its runtime source is `ef0d716`
and the later `1a8b4d7` changes tests only. Its source-matched control is
`/tmp/flint-comparison-factor-residual-v2-screen`, built with the same ignored `Cargo.lock`.
Twelve alternating 100-sample processes reduce the degree-64 factor median from `11.475622 ms` to
`11.111040 ms`, or 3.18%, and the paired ratio from `4.481832` to `4.339923`. The target product is
unchanged within noise at `0.005929 ms` versus `0.005934 ms` for the control.

The local-bound profile reduces Hensel cycles another 9.9% while modular screening remains within
0.3%. The next small, contained experiment is a fused dense residual update, but its realistic
ceiling is only about 2-5%. The next architectural target is coherent simultaneous lifting over a
product tree; FLINT's degree-greedy tree has about 27.7% less degree-weighted internal work and
updates precision in a small number of tree walks instead of independent near-global binary lifts.

Residual and local-subtree validation completed before integration:

- root factor module under default features `48/48`;
- root factor module under `no_gmp,native_code_generation` `48/48`;
- base-prime precision, binary-prime multi-round lifting, and non-monic nontrivial-`gamma` cases;
- the exact scaled-residual invariant after every debug linear-lift round;
- `cargo check --all-targets`, `cargo fmt --check`, and `git diff --check`.

The combined local-subtree source passed `50/50` factor tests under both GMP and `no_gmp` before
the final LLL test was added. That LLL test then passed separately under both backends; it is the
only change after the release build.

The complete default `cargo test --workspace` gate was last run on the preceding integrated chain;
the residual edit is confined to linear Hensel lifting and received the two complete factor-module
passes above.

The exact worktree state is:

| Worktree | Branch/head | State | Purpose |
|---|---|---|---|
| `/home/codexB/symbolica` | `dev`, `1a8b4d7` | only handoff Markdown modified | Integrated product/factor winner and live lab notebook |
| `/tmp/symbolica-factor-residual-update` | `codex/factor-incremental-residual`, `f4f47be` | clean; accepted source commit | Isolated incremental-residual reference |
| `/tmp/symbolica-factor-residual-control` | `codex/factor-residual-control`, `3b387a4` | clean except ignored lock | Source-matched full-LTO control |
| `/tmp/symbolica-factor-local-subtree` | `codex/factor-local-subtree`, `3037a2a` | clean historical source | Pre-rebase local-bound experiment |
| `/tmp/symbolica-factor-local-subtree-residual` | `codex/factor-local-subtree-residual`, `1a8b4d7` | clean; accepted source | Integrated residual plus local-bound reference |
| `/tmp/symbolica-factor-screening` | `codex/factor-prime-screening`, `966deda` | clean; accepted source commit | Isolated bounded-DDF/deferred-EDF reference |
| `/tmp/symbolica-quadratic-hensel` | `codex/quadratic-hensel`, `ed39caf` | clean; accepted source commit | Isolated quadratic Hensel reference |
| `/tmp/symbolica-univariate-product` | `codex/univariate-product-kronecker`, `ddab46e` | clean; release build and profiling complete | Accepted product conversion plus retained fixed-width statistics follow-up |
| `/tmp/symbolica-factor-large-prime` | `codex/factor-large-prime-degree64`, `dca8bd1` | clean; release build complete | Bounded dense-u64 factor-prime candidate based on `85be422` |
| `/tmp/symbolica-zp64-r2` | `codex/zp64-hybrid-inverse`, `374b8e2` | clean | Source branch for the already integrated inverse; reference only |

Before acting, confirm rather than assume:

```sh
git -C /home/codexB/symbolica status --short --branch
git -C /tmp/symbolica-univariate-product status --short --branch
git -C /tmp/symbolica-factor-large-prime status --short --branch
ps -eo user,pid,pcpu,stat,args | \
  rg 'rustc|cargo (build|test|bench)|rustred-.*test-threads' | \
  rg -v 'rg rustc|symbolica-univariate'
```

## Integrated state on `dev`

### Dense univariate integer GCD

The following source chain was accepted and is now represented by commits `38dc332` through
`55a758b` on `dev`:

| Integrated commit | Change |
|---|---|
| `38dc332` | Modular univariate integer GCD with CRT reconstruction and exact certification |
| `009bf85` | Direct dense `Zp64` Euclidean modular images |
| `05e550d` | Reused pure-multiprecision dense exact-division certificate |
| `02d6fca` | Precomputed `R^2` Montgomery conversion instead of division-based conversion |
| `c4c45f6` | Cached verified full-word primes for this univariate path |
| `55a758b` | Hybrid Euclidean inverse with quotient-1/2/3 subtraction steps |

The source commits on the old candidate chain had different hashes
(`810875a`, `ea09e01`, `f46fcff`, `7fdc099`, `b4db253`, `374b8e2`). They are historical references,
not commits to reapply. The timing binary for the inverse was frozen from production-identical
source at old commit `b7727d9`; the final source commit `374b8e2` only strengthened its test before
the code was cherry-picked as `55a758b`.

Validation completed on integrated `dev`:

- all default `poly::gcd::tests`: `28/28`;
- `poly::gcd::tests` with `--no-default-features --features no_gmp,native_code_generation`:
  `28/28`;
- Numerica GMP integer tests: `21/21`, finite-field tests: `12/12`;
- Numerica `no_gmp` integer tests: `16/16`, finite-field tests: `12/12`;
- `cargo fmt --check` and `git diff --check`.

The bare root `no_gmp` configuration without `native_code_generation` currently trips an unrelated
evaluator test configuration. The supported root check above is the relevant one for this chain.

### Other retained polynomial work

The broader retained work is already on `dev`:

- bounded thread-local reuse of large GMP integer storage and owned-operation storage reuse;
- fixed-width, Kronecker, direct-GMP, sparse-output, dense-simplex, integer, and finite-field
  multiplication kernels;
- operation-context organization in `lib/numerica/src/domains/integer/polynomial_kernels.rs` and
  `src/poly/kernels.rs`, without adding coefficient-specialized methods to `Ring`;
- checked dense exact division;
- Zippel/Hu evaluation, interpolation, prime selection, reconstruction, metadata reuse, early exit,
  and sparse/dense-image selection improvements;
- 1-, 2-, 3-, 5-, and 8-variable generated benchmarks, coefficient-height sweeps, PolyBench cases,
  and paired GCD/factor/product rows;
- the quadratic-variable factorization shortcut in `ef3b809`: for a square-free primitive factor
  quadratic in a variable, form `b^2 - 4ac`, recover an exact polynomial square root through
  square-free decomposition, and reconstruct the two factors. A `gcd(p,p')` alone is not an exact
  square-root algorithm because it does not recover all coefficient and multiplicity information;
- evaluated multivariate Hensel lifting in `e4712cf`;
- `resultant` as the main Lazard-Ducos entry point, `resultant_brown` as the explicit Brown PRS
  alternative, and `resultant_crt` as the explicit modular alternative. The zero-polynomial edge is
  fixed: if either polynomial is zero the resultant is zero, including zero versus constant.

`MultiplicationOptimizations.md` contains the detailed multiplication architecture and older
integer microbenchmark history. This file records the current continuation decisions.

## Current performance versus FLINT

Ratios are `Symbolica / FLINT`; lower is better. Values below one mean Symbolica is faster. These
are single-core paired measurements with default release features, including `faster_alloc`.

| Case | Current or best accepted ratio | Status |
|---|---:|---|
| dense integer GCD, 1 variable, degree 32 | about `0.99` | parity; remains on scalar path |
| dense integer GCD, 1 variable, degree 48 | `1.086` | small residual loss |
| dense integer GCD, 1 variable, degree 64 | `1.193` | primary GCD loss mostly closed |
| dense integer GCD, 1 variable, degree 80 | `1.249` | residual high-degree loss |
| dense integer GCD, 2 variables, degree 5 | `1.001` | parity |
| dense integer GCD, 3 variables, degree 7 | `0.203` | Symbolica about five times faster |
| generated 3- to 8-variable GCD | generally below `1` | Symbolica generally faster |
| generated high-height GCD | about `0.5` | Symbolica about twice as fast |
| PolyBench 5-variable uniform #11 | `1.040` | small remaining Zippel loss |
| PolyBench 8-variable sharp #140 | `1.211` | residual Hu/Zippel loss |
| factor fixture product, 1 variable, degrees 33/31 | `1.089` | unchanged by local Hensel bounds |
| factorization, 1 variable, degrees 32/31 | `4.224` | local-bound guard is about 1% slower than its control |
| factorization, high-height 1 variable, total degree 33 | `4.547` | local-bound guard improves about 0.9% |
| factorization, 1 variable, total degree 64 | `4.340` | local bounds save another 3.18% after incremental residuals |
| factorization, 1 variable, total degree 65 | `5.927` | local-bound change is neutral |
| factorization, 2 variables, degrees 10/9 | `0.616` | faster than FLINT |
| factorization, 3 variables, degrees 6/5 | `1.968` | later modular/multivariate target |
| PolyBench 8-variable uniform factor #105 | `1.105` | local-bound change is neutral |
| PolyBench 8-variable sharp factor #178 | `2.982` | local-bound change is neutral |

The GCD rows use their retained source-matched measurements. The latest product and factor rows use
the `ef0d716` runtime source: twelve alternating 100-sample processes for degree 64 and six
alternating processes for the product and other guards. The degree-63 value is the six-process
local-bound guard; the stronger residual-only measurement immediately before it was `4.254`.
Earlier tables below remain as attribution evidence.

## Accepted incremental linear-Hensel residual

Commit `31d50a9` maintains the exact scaled residual

```text
E = (a - u*w) / m
```

through each base-prime correction. For `u' = u + m*tau` and `w' = w + m*r`, it computes

```text
E' = (E - tau*w - r*u') / p
```

using the old `w` and updated `u'`. This replaces 260 growing large-by-large residual products and
10,913 divisions by the growing modulus on the degree-64 fixture with small-correction products
and exact division by the fixed base prime. A debug assertion reconstructs `a-u*w` after every
round.

The source-matched degree-64 result is:

| Version | Symbolica minimum | Symbolica median | FLINT median | S/F |
|---|---:|---:|---:|---:|
| `3b387a4` control | `8.607947 ms` | `13.926576 ms` | `2.566790 ms` | `5.419751` |
| incremental residual | `6.822662 ms` | `11.504680 ms` | `2.556910 ms` | `4.499028` |

This is a 20.74% minimum-time improvement, 17.39% median improvement, and 16.99% ratio
improvement. Raw rows are
`/tmp/factor-residual-v2-matched-d64-{candidate,control}-block-{1..12}.csv`.

The product regression guard is neutral:

| Version | Symbolica median | FLINT median | S/F |
|---|---:|---:|---:|
| `3b387a4` control | `0.005960 ms` | `0.005408 ms` | `1.100941` |
| incremental residual | `0.005942 ms` | `0.005416 ms` | `1.095146` |

Raw rows are
`/tmp/factor-residual-v2-matched-product64-{candidate,control}-block-{1..12}.csv`.

A 12-process rerun of the degree-63 boundary gave `12.902416 ms` versus `13.212540 ms`, or a
2.35% improvement, with ratios `4.253628` versus `4.365554`. Degree 65 improves from about
`22.93 ms` to `17.84 ms`, or 22%. The high-height quadratic path, generated two- and
three-variable rows, and PolyBench #105/#178 are neutral within process noise. Raw guard rows use
the prefix `/tmp/factor-residual-v2-matched-*`.

The LBR profile contains 500 calls per implementation. Normalized Symbolica factor cycles fall
from `51.919 M` to `44.548 M` per call; Hensel cycles fall from `36.838 M` to `27.459 M`. The old
integer residual group falls 32.2%. Integer polynomial subtraction and growing-modulus division
disappear; the main remaining Hensel branches are the two correction convolutions, finite-field
correction work, and intermediate add/negate/materialization passes.

Artifact provenance:

| Artifact | SHA-256 |
|---|---|
| `/tmp/flint-comparison-factor-residual-v2-screen` | `7f28a11b15fe3190c66fcda43770f7d0ad28ce533f6cce8e032f0750308526af` |
| `/tmp/flint-comparison-factor-residual-control-v2-screen` | `f7a7d54d23cb9770efeefb02ee693197b07cc012f17351cbb1debe7d4a79cbad` |
| `/tmp/profile-factor-residual-v2-d64-candidate-lbr.perf.data` | `a0ba24ef5ea866e94d98cd88a746570baaa21158f96746ecb7ceb829f2899fca` |

The candidate binary predates only test/comment/`debug_assert` additions; its release runtime code
is identical to `31d50a9`. Its ignored lock has SHA-256
`af8148d739e4e55630658a3f8a35d9676484ce198c535e5f503dbbde47db5511`.

## Accepted exact-subproblem local Hensel bounds

Commits `f367a8a` through `1a8b4d7` preserve the `Ok`/`Err` result at every binary Hensel split.
An `Ok` result certifies literal integer equality, so each exact child is recursively factored at a
modulus computed from that child's coefficient bound. An `Err` result remains a congruence at its
parent modulus: its modular leaves are lifted at that same modulus and recombined against the exact
parent before any result escapes. Every recombination candidate is verified by exact division.

The first full-LTO candidate recomputed `quadratic_lift_allowed` at each child. That unintentionally
activated the known-slow quadratic composite-modulus path inside the four-factor right subtree and
regressed the degree-64 row to `12.75 ms`, about 11% slower than the residual-only control. Commit
`ef0d716` restores the original policy: the root's linear/quadratic decision is propagated through
the whole factor tree. Do not repeat the per-child policy change.

With the strategy confound removed, twelve alternating 100-sample processes give:

| Version | Symbolica minimum | Symbolica median | FLINT median | S/F |
|---|---:|---:|---:|---:|
| incremental-residual control | `6.826477 ms` | `11.475622 ms` | `2.560343 ms` | `4.481832` |
| local exact-subproblem bounds | `6.017853 ms` | `11.111040 ms` | `2.560053 ms` | `4.339923` |

This is an 11.85% minimum-time improvement, 3.18% median improvement, and 3.17% ratio improvement.
Every one of the twelve paired processes favored the candidate. Raw rows are
`/tmp/factor-local-subtree-residual-v2-d64-{candidate,control}-block-{1..12}.csv`.

Six-process guards, shown as candidate versus residual-only control, are:

| Case | Candidate | Control | Change |
|---|---:|---:|---:|
| high-height degree 33 | `6.318319 ms` | `6.375701 ms` | `-0.90%` |
| degree 63 | `12.702184 ms` | `12.524596 ms` | `+1.42%` |
| degree 65 | `18.028802 ms` | `17.970795 ms` | `+0.32%` |
| generated factor, 2 variables | `6.020666 ms` | `6.111690 ms` | `-1.49%` |
| generated factor, 3 variables | `8.160884 ms` | `8.325584 ms` | `-1.98%` |
| PolyBench #105 | `32.005746 ms` | `31.948833 ms` | `+0.18%` |
| PolyBench #178 | `41.705666 ms` | `41.703882 ms` | `+0.00%` |
| degree-64 input product | `0.005929 ms` | `0.005934 ms` | `-0.09%` |

The LBR candidate profile contains 500 calls per implementation. Total sampled cycles fall from
`29.343 B` to `27.701 B`. Normalized Symbolica factor cycles are about `40.74 M` per call versus
`44.55 M`, an 8.6% reduction; Hensel cycles are about `24.74 M` versus `27.46 M`, a 9.9%
reduction. Modular screening is unchanged at about `13.82 M` versus `13.86 M`. The wall-time gain
is smaller than the sampled-cycle change but is repeatable across all twelve primary processes.
Inside Hensel, integer polynomial multiplication falls from about `9.44 M` to `8.06 M` cycles per
call and explains roughly half of the Hensel saving; finite-field multiplication falls from about
`4.06 M` to `3.64 M`.

Validation includes exact/inexact child recombination, child-local moduli, non-monic exact
children whose local modulus equals the base prime, and local LLL recombination with twelve modular
leaves. The combined factor suite passed `50/50` under both GMP and `no_gmp`; the later LLL test
passed separately under both backends.

Artifact provenance:

| Artifact | SHA-256 |
|---|---|
| `/tmp/flint-comparison-factor-local-subtree-residual-v1-screen` | `f9888c5f1322d4c9012ccc35f0840ccdef251ed95a313b313bcb2cf18b8a93d2` |
| `/tmp/flint-comparison-factor-local-subtree-residual-v2-screen` | `af6ca9ab6d788afce0640cd218f08da45376f77f8b09b75eab5328f71adcfdc0` |
| `/tmp/flint-comparison-factor-local-subtree-residual-v2-build.jsonl` | `ef15011b97816b718070465577e9a8d99194b1bd796729cbb3aaa1d0bd4682ff` |
| `/tmp/profile-factor-local-subtree-residual-v2-d64-candidate-lbr.perf.data` | `95ffaa2ac625bd48002a01ff53bedd60e4ac342e95f9a7cfeed29d1bca345ab3` |
| `/tmp/profile-factor-local-subtree-residual-v2-d64-candidate-lbr.children-symbols.txt` | `3d798ef951b6020d364057a586f2b3d9585c9ef889503b4854d2ed00c3cf372c` |
| `/tmp/profile-factor-local-subtree-residual-v2-d64-candidate.csv` | `f05211165b12eb6af850e0399cbb24a9832bc91b8c0dbc767d043cceb5b86003` |

The v2 release binary is runtime-identical to integrated `1a8b4d7`; the later source changes add
only the LLL test. It uses the same ignored lock SHA-256 as the residual-only control.

## Final integrated `f360be0` measurement

The full-LTO build took 9m12s. Its provenance is:

| Artifact | SHA-256 |
|---|---|
| `/tmp/flint-comparison-dev-f360be0-screen` | `e908c74b3768482a907e09d284cb840d721213abb5f0fea9bb8ddbd9a517f85d` |
| `/tmp/flint-comparison-dev-f360be0-screen.d` | `2d0e4607381d47350254a8bb19bb1ed9778d91d07b495214e3449a0eb07037b0` |
| `/tmp/flint-comparison-dev-f360be0-build.jsonl` | `d8dc4996c62edcd67094c8c827e78c3b1b273492f0b1a62478e204560e43edca` |

The printed version is stale (`ddab46e+dirty`) because Cargo reused build-script output. The Git
source, JSON artifact path, dependency sidecar, and checksums above are authoritative.

Five interleaved product processes against the frozen pre-integration `55a758b` binary gave:

```text
f360be0: 1.067077, 1.062026, 1.068749, 1.073482, 1.061922; median 1.067077
55a758b: 3.087455, 3.059827, 3.187895, 3.143390, 3.115863; median 3.115863
```

Median Symbolica time fell from `16.539 us` to `5.612 us`, a 66.1% reduction. Five factor
processes gave:

```text
f360be0: 5.293791, 5.362441, 5.304613, 5.095155, 5.246005; median 5.293791
55a758b: 6.975339, 7.132943, 7.057195, 7.021013, 7.140162; median 7.057195
```

Median Symbolica factor time fell from `21.251629 ms` to `15.896048 ms`, a 25.2% reduction. Raw
files are `/tmp/integrated-f360be0-{product,factor}-run-0{1..5}.csv` and
`/tmp/integrated-control-55a758b-{product,factor}-run-0{1..5}.csv`.

The final product guard matrix is:

| Case | Symbolica median | FLINT median | S/F |
|---|---:|---:|---:|
| dense high large | `104.099 ms` | `108.728 ms` | `0.957` |
| dense very large | `128.104 ms` | `282.618 ms` | `0.450` |
| sparse separated | `12.954 ms` | `18.613 ms` | `0.693` |
| seven-variable power-minus-one | `139.881 ms` | `102.234 ms` | `1.367` |
| GF(17), seven variables | `39.065 ms` | `98.931 ms` | `0.395` |
| 64-bit field, seven variables | `45.461 ms` | `98.755 ms` | `0.460` |

These source-matched rows resolved the transient regressions seen between the isolated LTO
binaries: the unchanged finite-field rows recovered and the high-large case remained at parity.
Raw files are `/tmp/integrated-f360be0-guard-{high_large,very_large,sparse,v7}-run-0{1..5}.csv`.

The final dense univariate GCD sweep is:

| Degree | Symbolica median | FLINT median | S/F |
|---:|---:|---:|---:|
| 32 | `0.082593 ms` | `0.083226 ms` | `0.993399` |
| 48 | `0.257568 ms` | `0.237171 ms` | `1.085959` |
| 64 | `0.422516 ms` | `0.354129 ms` | `1.193018` |
| 80 | `0.736127 ms` | `0.589607 ms` | `1.248505` |

Raw files are `/tmp/integrated-f360be0-gcd-d{32,48,64,80}-run-0{1..5}.csv`.

The final generated low-dimensional and PolyBench anchors are:

| Case | Symbolica median | FLINT median | S/F |
|---|---:|---:|---:|
| generated GCD, dense 2 variables degree 5 | `0.061601 ms` | `0.061549 ms` | `1.000845` |
| generated GCD, dense 3 variables degree 7 | `0.755848 ms` | `3.718048 ms` | `0.203292` |
| generated factor, 2 variables degrees 10/9 | `6.370784 ms` | `10.116565 ms` | `0.629738` |
| generated factor, 3 variables degrees 6/5 | `8.716825 ms` | `4.306574 ms` | `2.033647` |
| PolyBench #11, 5-variable uniform GCD | `10.229013 ms` | `9.849162 ms` | `1.039703` |
| PolyBench #140, 8-variable sharp GCD | `5.768123 ms` | `4.758808 ms` | `1.211093` |
| PolyBench #105 factorization | `36.555738 ms` | `29.023706 ms` | `1.259605` |
| PolyBench #178 factorization | `41.874724 ms` | `14.058810 ms` | `2.977637` |

Raw files are `/tmp/integrated-f360be0-{gcd2,gcd3,factor2,factor3}-run-0{1..5}.csv` and
`/tmp/integrated-f360be0-pb{11,105,140,178}-run-0{1..5}.csv`.

## Completed dense degree-64 GCD decision

The accepted inverse candidate produced five process ratios:

```text
1.148490, 1.146328, 1.148613, 1.145996, 1.147013
median 1.147013
```

Three interleaved cached-prime controls produced `1.189134`, `1.189990`, and `1.189119`; an earlier
five-control median was `1.191412`. Candidate absolute process-median median was approximately
`0.415537 ms` versus approximately `0.429187 ms` for the interleaved control, a 3.2% absolute gain
and roughly 3.5% normalized-ratio gain.

Neighboring screens were:

| Degree | Candidate median | Control median | Interpretation |
|---:|---:|---:|---|
| 32 | about `1.000` | about `0.988` | small code-layout movement; scalar selector unchanged |
| 48 | `1.042712` | `1.088555` | about 4.2% normalized improvement |
| 64 | `1.147013` | about `1.189` | accepted |
| 80 | about `1.224714` | about `1.246` | about 1.7% improvement |

The profile and hardware counters explain the gain. The candidate cut
`divider_active` from `2,635,765,564` to `1,453,906,797` (about 45%), while combined cycles fell
from `69,792,599,989` to `68,639,827,005`. The Euclidean invariant proves the small-quotient
subtractions cannot overflow: in the relevant state `u1*v3 + v1*u3 = p`, so `q*v1 <= p`; the two
Montgomery de-scalings are intentional and independently tested.

Useful artifacts:

| Artifact | SHA-256 |
|---|---|
| `/tmp/univariate-zp64-inverse-b7727d9-screen` | `83852a6af0bd1c2ce90373e03cfbc966c451ca051e5e29b28884e647d92976a5` |
| `/tmp/univariate-prime-r2-b4db253-screen` | `c94b9c4d6a083d438cee123e1dc60254782307434c59a5ef0cd0079915b9a7e5` |
| `/tmp/flint-comparison-dev-55a758b-screen` | `9d02ea2dd470bcaa75ecc0e3bbdc488de3e992a63f87240e114c3eca901a9ef3` |
| `/tmp/profile-d64-inverse-b7727d9-lbr.perf.data` | `4f6368db3a03df3f9a67398ae065337d9f20aa9cc3ac4da6b5979f7da867bf82` |

Raw timing files are `/tmp/d64-inverse-run-0{2..5}.csv`,
`/tmp/d64-control-run-0{1..3}.csv`, `/tmp/d64-integrated-55a758b-run-01.csv`, and the corresponding
degree-32/48/80 files under `/tmp`.

## Integrated one-variable product history

### Fixture and baseline

The exact paired row is:

```text
generated factor product: dense 1-variable degrees 32/31
```

It multiplies `(1+3*x1)^32-1` by `(1-5*x1)^31+1`. Both inputs contain 32 contiguous nonzero terms;
the output contains 63 nonzero terms. The left coefficients are tagged `Integer::Single`; the
upper half of the right side and most output coefficients require `Integer::Double`.

Five fresh 5,000-sample `55a758b` controls gave ratios:

```text
3.036960, 3.040487, 3.080261, 3.042722, 3.025231
median about 3.0405
```

Symbolica took about `16.0 us`; FLINT took about `5.25 us`.

### Selector candidate `85be422` (integrated as `418d7b0`)

Commit `85be422` (`Use Kronecker multiplication for dense univariate products`) relaxes
`DenseIntegerMul::try_kronecker` for sufficiently large contiguous supports, while preserving the
old high-collision gate for sparse or shifted supports. It adds exact-fixture and rejection tests.
Its five 5,000-sample ratios were:

```text
2.547483, 2.508818, 2.555409, 2.543075, 2.549193
median about 2.5475
```

Symbolica fell to about `13.37 us`, a 16-17% gain. The commit is a useful improvement but should be
integrated together with the pending pack/unpack optimization if that patch validates.

Frozen binary `/tmp/flint-comparison-product-85be422-screen` has SHA-256
`fdb7c4726934adeed3ed3ac686d876503f1bf6f4af59f6f635107b569ada5269`.
Raw files are `/tmp/product-85be422-run-0{1..5}.csv` and
`/tmp/product-control-55a758b-run-0{1..5}.csv`.

### Why the remaining product gap exists

The product profiles are:

- control: `/tmp/profile-product-control-55a758b.perf.data`, SHA-256
  `dfd81e1559cc6918662a4499309ee9f1bfd1e244d81ae8355c8e9926e5c4ae5b`;
- selector candidate: `/tmp/profile-product-85be422.perf.data`, SHA-256
  `e61b6c4be0bd59d501c6bcdb37a32b7ebd85c08a09e5f4aa2295c16d189eab4d`.

The Kronecker candidate costs about `61.65k` Symbolica cycles/product versus `27.15k` for FLINT.
`DenseIntegerMul::run` is about 95% of Symbolica time. The GMP packed multiplication itself is
already slightly cheaper than FLINT's; it is not the missing factor of 2.5. The excess is conversion:

- absolute coefficient statistics: about `5.9k` cycles;
- packing: about `11.2k` cycles;
- GMP multiply: about `8.7k` cycles;
- unpack/carry/demotion and failed fixed-width probes: about `32.8k` cycles.

In the candidate profile, unpacking is about 44.85% of Symbolica time, packing 18.17%, and failed
fixed-width probes 6.50%. GMP allocation/free and multiprecision drop/cache overhead account for
about 22.23% of Symbolica time inside the unpack region. Generic polynomial result construction is
under 4% and is not worth restructuring yet. Every target output coefficient fits
`Integer::Double` (at most 126 bits), but the 143-bit packed-radix decoder currently creates a GMP
integer for each slot, adjusts its sign in GMP, and immediately demotes it.

This is what the earlier phrase “dense multiplication still performs generic pairwise coefficient
multiplication” meant for the pre-`85be422` path: the support was dense, but the selector rejected
Kronecker and executed 1,024 generic coefficient multiply-adds, including roughly 244 promoted GMP
operations. It did not mean the current Kronecker candidate still uses pairwise multiplication.

### Direct pack/unpack candidate `846620d`

Commit `846620d` (`Speed up Kronecker coefficient conversion`) changes only:

```text
lib/numerica/src/domains/integer.rs
lib/numerica/src/domains/integer/polynomial_kernels.rs
```

The commit is `+350/-78`, including 103 lines of boundary tests. `cargo fmt --all --check` and
`git diff --check` pass.

The patch:

- proves fixed-width `i64`, mixed `i64/i128`, and `i128` representation/bounds before allocating
  their coefficient vectors;
- packs tagged `Integer::Single` and `Integer::Double` values directly into reusable digit limbs,
  including `i64::MIN`, `i128::MIN`, sign, borrow, and two's-complement cases;
- retains the existing GMP packing semantics for `Integer::Large`;
- decodes signed packed digits directly to `i128` for magnitudes of at most four limbs, including
  carry and `i128::MIN`;
- creates the GMP radix lazily and uses the unchanged GMP fallback for wider or out-of-range
  digits;
- adds `primitive_kronecker_packing_matches_direct_convolution_at_boundaries`.

Static audit points already checked: minimum signed magnitudes avoid signed overflow; reused limb
buffers are completely zeroed; carry is computed before applying product sign; `raw + carry ==
radix` maps to zero with carry; a failed native decode does not mutate the fallback input; existing
width, bound, and `MAX_PACKED_BITS` guards remain unchanged.

Direct decode intentionally stops at four limbs; wide output still allocates GMP temporaries and
`absolute_statistics` is not optimized.

Validation completed from the product worktree:

- five focused Kronecker fixture, primitive-boundary, radix-boundary, large-GMP fallback, and
  selector-rejection tests;
- all GMP integer-domain tests: `25/25`;
- all `no_gmp` integer-domain tests: `16/16`;
- signed-radix differential cases at 63/64/65 and 127/128/129 bits, an exact `i128::MIN` output,
  primitive values whose outputs require GMP fallback, and existing 180-bit `Large` inputs.

The specialized code is `#[cfg(feature = "gmp")]`, so the `no_gmp` run primarily guards fallback
compilation and semantics.

The full-LTO build completed in 9m33s. Its artifacts are:

| Artifact | SHA-256 |
|---|---|
| `/tmp/flint-comparison-product-846620d-screen` | `6be471bbbffe4f04e94b0e2e294176e95efa06c0d905ffcb38171a2a62bb2e26` |
| `/tmp/flint-comparison-product-846620d-screen.d` | `1069f36fc472359d97db137631cd8885072070fb6c0cda8b9614950154dfcdca` |

Five interleaved 5,000-sample processes produced:

```text
846620d: 1.276777, 1.273312, 1.272624, 1.267600, 1.270153; median 1.272624
85be422: 2.551859, 2.500459, 2.513216, 2.563446, 2.556476; median 2.551859
```

The median of Symbolica process medians fell from `13.655 us` to `6.732 us`, a 50.7% gain. Versus
the approximately `16.0 us` `dev` baseline, the combined selector and conversion changes save
about 58%. Raw files are `/tmp/product-846620d-run-0{1..5}.csv` and
`/tmp/product-846620d-control-85be422-run-0{1..5}.csv`.

Profile `/tmp/profile-product-846620d-lbr.perf.data` has SHA-256
`b4095f27500709203001f9bbd3062825548b734f31a7c77e87640dbd324a5169`. Its one-million-sample
timing was `6.708 us` versus FLINT `5.227 us`, ratio `1.283337`. Inclusive combined-cycle shares
are about 46.04% for `DenseIntegerMul::run`, 42.70% for FLINT's `fmpz_mpoly_mul`, 9.31% for
Symbolica `absolute_statistics`, and 7.58% for Symbolica packing. The shared GMP basecase multiply
is 21.54% of combined cycles. The old per-digit GMP unpack/demotion bottleneck is gone. A small-
coefficient `u128` statistics prepass could plausibly close much of the remaining 27%, but benchmark
it as a separate change. First guard nearby product degrees, coefficient heights, sparse/shifted
inputs, and output wider than `i128`.

Commit `ddab46e` is the separate fixed-width-statistics experiment. It accumulates coefficient
magnitudes and maxima in `u128`, converts only the final sum/max pair for each operand to GMP, and
reruns the unchanged multiprecision scan if it sees `Integer::Large` or a sum overflow. The
existing primitive-boundary test exercises sum overflow and the existing 180-bit test supplies
`Large` values. Validation completed before committing:

- all seven Kronecker tests;
- all GMP integer-domain tests, `25/25`;
- all `no_gmp` integer-domain tests, `16/16`;
- `cargo fmt --check` and `git diff --check`.

Its full-LTO build completed in 9m14s. The artifacts are:

| Artifact | SHA-256 |
|---|---|
| `/tmp/flint-comparison-product-ddab46e-screen` | `bdf52a34ad29cecd7620d947d96c12077da245faa344f149439f30d0b4928543` |
| `/tmp/flint-comparison-product-ddab46e-screen.d` | `a363fb551b7971103e2bbf0aa86686b6aff740d4546909763d77918f9edf211d` |
| `/tmp/flint-comparison-product-ddab46e-build.jsonl` | `e81bbb5c192af95cceba67da145ec996794a1ca0dcb743d7e48d77a44cd7e5af` |

Five 5,000-sample processes against the frozen `846620d` binary produced:

```text
ddab46e: 1.064781, 1.077274, 1.079784, 1.069842, 1.081128; median 1.077274
846620d: 1.281904, 1.277403, 1.287012, 1.288996, 1.283804; median 1.283804
```

Median Symbolica process time fell from `6.677 us` to `5.637 us`, a further 15.6%. Raw files are
`/tmp/product-ddab46e-run-0{1..5}.csv` and
`/tmp/product-ddab-control-846620d-run-0{1..5}.csv`. This is a material isolated gain, so retain
`ddab46e`; the accepted pack/unpack result remains independently represented by `846620d`.

The one-million-sample profile measured `5.526 us` versus FLINT `5.157 us`, ratio `1.071553`.
`absolute_statistics` fell from 9.31% to 1.15% of combined cycles. `DenseIntegerMul::run` is now
42.73% inclusive, FLINT `fmpz_mpoly_mul` 46.01%, Symbolica packing 6.71%, and shared GMP basecase
multiplication 25.17%. Profile artifacts are:

| Artifact | SHA-256 |
|---|---|
| `/tmp/profile-product-ddab46e-lbr.perf.data` | `55c67064dadf6a1362dfe12bfa29ca2bd0f0c4fc8434ba8c6551babe211bafa9` |
| `/tmp/profile-product-ddab46e-lbr.symbols.txt` | `269395f30d6e7147f81909e8982c1690367ad340bd65f7300f5dba7e9353f689` |
| `/tmp/profile-product-ddab46e-lbr.children-symbols.txt` | `a32a2d1a378c375ae95d85fa77297f29cb5c60762768fab79bf9d3877b137ad4` |

Additional guards versus `846620d` gave:

| Case | `ddab46e` median Symbolica | `846620d` median Symbolica | Interpretation |
|---|---:|---:|---|
| dense very large, fixed-width input/wide output | `127.124 ms` | `127.875 ms` | 0.6% faster |
| dense high | `4.927 ms` | `4.907 ms` | 0.4% slower |
| dense high large | `105.191 ms` | `104.249 ms` | 0.9% slower |
| sparse separated | `13.269 ms` | `13.310 ms` | unchanged within noise |

The high-height operands begin with `Integer::Large`, so the fixed-width scan exits after one
coefficient before running the unchanged GMP scan; the sub-1% shift is not accumulated prepass
work. The seven-variable product showed a larger binary-to-binary shift: `153.621 ms` versus
`143.351 ms` in five short processes, and `149.060 ms` versus `141.539 ms` in paired 20-sample
profiles. However, those same profiles showed the `ddab46e` binary 4.4% slower on the unchanged
GF(17) row and 10.7% slower on the unchanged 64-bit-field row, while FLINT stayed stable. This is
strong evidence of whole-program-LTO text layout/alignment sensitivity, not changed integer
arithmetic bounds or strategy selection. Preserve the raw files under
`/tmp/product-ddab46e-{high,high-large-balanced,very-large,sparse,v7}-*`; rerun these rows from the
single final integrated build before making a regime decision based on sub-percent effects or the
seven-variable shift.

## Integrated one-variable factorization history

The exact paired factor row is:

```text
generated factorization: dense 1-variable degrees 32/31
```

Five 20-sample `55a758b` processes gave ratios:

```text
6.965303, 7.118203, 6.967145, 6.074375, 7.072056
median 6.967145
```

The timing is bimodal: Symbolica process medians are around `21.06 ms`, but minima are around
`13.9 ms`; FLINT is around `3.02 ms`. Product formation is outside the timed factor row, so the
16-17% product improvement in `85be422` did not materially change factorization time.

Profile `/tmp/profile-factor-dev-55a758b-lbr.perf.data` has SHA-256
`25d905c1f56daaca4f99a551197d1a865d1b41e60fb4355a87c678b0d5f372da`. A 1,000-sample profile
measured ratio `7.044056`. Symbolica factorization accounts for about 75% of combined cycles;
`factor_reconstruct` dominates recursively, and the Hensel subtree accounts for roughly 56-65% of
combined cycles. Square-free decomposition and its GCD are only about 0.5% of Symbolica factor
time (roughly `0.1 ms`). Do not tune the univariate-GCD cutoff for this fixture.

The primitive degree-63 input has 63 terms and true factor degrees
`1,1,2,4,8,16,1,30`. Maximum input height is about 125 bits and factor height about 75 bits. The
current selector chooses `p=11`, sees 11 modular factors, needs about 90 base-11 lifting digits,
and has estimated linear lift work about 900. The existing 31-bit-prime path also sees 11 factors
but needs only about 11 digits/work 110; it was skipped solely because the old
`high_linear_lift_pressure` degree guard was `d <= 32`.

Commit `d018cd4` changes that guard to `d <= 64`. It is intentionally only a one-line selector
experiment. Focused `factor_univariate_` tests (`4/4`), formatting, and diff checks passed. Its
full-LTO release build completed in 9m56s from `/tmp/symbolica-factor-large-prime`; the build log is
`/tmp/flint-comparison-factor-d018cd4-build.jsonl`. The frozen executable and sidecar are:

| Artifact | SHA-256 |
|---|---|
| `/tmp/flint-comparison-factor-d018cd4-screen` | `ae7cb74f1fde32da5ae3d4d665ef89f16e01cb0948e27dd50bb81d3ef5ac4a8b` |
| `/tmp/flint-comparison-factor-d018cd4-screen.d` | `45a1b1cd8b33c38d07537012033f1a99673c86b51c079b500d19c08ca5352179` |

Five interleaved 50-sample factor processes produced:

```text
d018cd4: 5.617377, 5.652122, 5.434618, 5.542794, 5.449818; median 5.542794
85be422: 7.012594, 7.079679, 6.970776, 6.986772, 6.896782; median 6.986772
```

The median of Symbolica process medians fell from `21.694129 ms` to `17.051243 ms`, about 21.4%.
Three 5,000-sample product checks stayed near `13.50 us`, confirming the factor-only selector did
not improve or materially regress the separate product.

The candidate profile is `/tmp/profile-factor-d018cd4-lbr.perf.data`, SHA-256
`306908c0cf6dbc4829b615d9bae14cfbe3ea16072c10001e7d76ce577f04095e`; its 1,000-sample timing was
`16.960193 ms` versus FLINT `3.071320 ms`, ratio `5.522119`. The old small-prime profile was
GMP/integer-Hensel heavy. The 31-bit candidate instead spends 11.82% of combined cycles in
`DenseZpMul::multiply_direct_u128`, 11.80% in `quot_rem_univariate_monic`, 8.10% in finite-field
polynomial multiplication glue, and 6.29% in `__umodti3`. This is evidence for a medium-prime
experiment, not for immediately optimizing square-free GCD or generic integer multiplication.

Commit `dca8bd1` replaces the 31-bit candidate with a bounded dense-u64 prime search starting at
`65_000_000`; the first prime is `65_000_011`. The search stops at
`u32::MAX / (d + 1)`. The bound `p * (d + 1) <= u32::MAX` implies both
`(p-1)^2 * (d+1)^2 < 2^64` and `(p-1)^2 * (d+1) < p * 2^32`, so the degree-bounded dense products
use the existing `u64` accumulator and direct Montgomery reduction. For this fixture the prime is
26 bits, produces the same minimum 11 modular factors, and needs 12 lifting digits/work 120 rather
than 90/work 900 at `p=11`; the 31-bit prime needs 11/work 110. Thus it gives up only one lift digit
to avoid `__umodti3`. The four focused `factor_univariate_` tests, formatting, and diff checks pass.
Its full-LTO build completed in 9m24s. The frozen build artifacts are:

| Artifact | SHA-256 |
|---|---|
| `/tmp/flint-comparison-factor-dca8bd1-screen` | `dea7f4e7321d07d8dde87aadaa048b626ab7d3bd38c280f7d84ff0e6dfaf6fbc` |
| `/tmp/flint-comparison-factor-dca8bd1-screen.d` | `96cb4dc488eb8954f39f515b02fe133d536d8134617afe136334de79957869d1` |
| `/tmp/flint-comparison-factor-dca8bd1-build.jsonl` | `24014a5a77771152f65a560b8c6e1cc5ccd1a533c6b7a3bd9030e0682023e55a` |

Timing was deliberately deferred until an unrelated multi-core Rust test job finished. Five
interleaved 50-sample processes against the frozen `d018cd4` binary then produced:

```text
dca8bd1: 5.103775, 5.029697, 5.081360, 5.034260, 4.997398; median 5.034260
d018cd4: 5.690329, 5.505906, 5.518398, 5.649726, 5.542353; median 5.542353
```

The median of Symbolica process medians fell from `17.030339 ms` to `15.283495 ms`, another
10.3%. Relative to the original small-prime median of `21.694129 ms`, the two selector changes
together save 29.5%. Raw files are `/tmp/factor-dca8bd1-run-0{1..5}.csv` and
`/tmp/factor-dca-control-d018cd4-run-0{1..5}.csv`.

The 1,000-sample profile measured `15.558508 ms` versus FLINT `3.077244 ms`, ratio `5.055988`.
The files and checksums are:

| Artifact | SHA-256 |
|---|---|
| `/tmp/profile-factor-dca8bd1-lbr.perf.data` | `0cc5b66ecbf74d37fcf1a4f66786b5bf5499dd99f39b88411fe8b4bab707c88d` |
| `/tmp/profile-factor-dca8bd1-lbr.symbols.txt` | `c88d5546a4abda2e0d8c916c9ee7f8acfb2aac0490585fa029f5bd18fbbeae5c` |
| `/tmp/profile-factor-dca8bd1-lbr.children-symbols.txt` | `8b8936e09b180d491645e6d7ed67a2dd16842cec58ad3691d09ba63d302f2fb1` |

The expected transition occurred: `multiply_direct_u128` and `__umodti3` disappeared from the
leading samples. `multiply_direct_u64` is now 12.00% flat; monic finite-field division is the
largest flat Symbolica symbol at 15.91%. Modular factorization remains 58.48% inclusive and
Hensel lifting 12.62% inclusive. `dca8bd1` was integrated as `9609704`; its degree and cheap-regime
guards are reported below. The selector must not send cheap cases through a more expensive modular
setup.

The existing low-degree generated factor rows provide an initial cheap-regime guard. Five
20-sample processes showed no regression against `d018cd4`:

| Case | `dca8bd1` median Symbolica | `d018cd4` median Symbolica | Median paired ratio, candidate/control |
|---|---:|---:|---:|
| dense 2-variable degrees 10/9 | `6.339744 ms` | `6.375589 ms` | `0.626315 / 0.628227` |
| dense 3-variable degrees 6/5 | `8.463273 ms` | `8.589960 ms` | `1.981203 / 2.012980` |

These rows are below the high-pressure selector threshold and therefore primarily check that the
new branch does not perturb ordinary recursive factorization. Raw files are
`/tmp/factor-dca8bd1-v{2,3}-run-0{1..5}.csv` and the corresponding
`/tmp/factor-dca-control-d018cd4-v{2,3}-run-0{1..5}.csv` files. The winning univariate fixture has
product degree 63, so that boundary is measured directly; degrees 33, 64, and the excluded degree
65 are measured in the next section.

### Integrated degree and coefficient-height boundaries

The three explicit one-variable factor boundaries were run in five 20-sample processes:

| Case | Symbolica median | FLINT median | S/F |
|---|---:|---:|---:|
| high height, degrees 17/16, total 33 | `36.051678 ms` | `1.385226 ms` | `26.114583` |
| degrees 33/31, total 64 | `21.868983 ms` | `2.560637 ms` | `8.566840` |
| degrees 33/32, total 65 | `23.145640 ms` | `3.047739 ms` | `7.577071` |

Degree 64 is 5.5% faster in absolute Symbolica time than the excluded degree-65 case, so the
selector boundary itself is behaving as intended. Coefficient height is the sharper unresolved
axis: the total-degree-33 row is much slower despite its lower degree. Raw files are
`/tmp/integrated-f360be0-boundary-d{33,64,65}-run-0{1..5}.csv`.

A 500-sample LBR profile of the high-height row measured `36.865833 ms` versus `1.394914 ms`, ratio
`26.428750`. Within Symbolica's factor call, about 84% lies under `hensel_lift`; modular
factorization is only about 15%. The dominant flat work is GMP multiplication/division driven by
recomputing the full integer product and residual at every base-prime digit. FLINT's same profile
shows `fmpz_poly_hensel_lift_tree`, `fmpz_poly_hensel_lift_once`, and separate inverse/no-inverse
lift stages. This is direct evidence for a quadratic/staged Hensel experiment, not another prime
selector or allocation-only change.

| Profile artifact | SHA-256 |
|---|---|
| `/tmp/profile-factor-f360be0-d33-high-v2-lbr.perf.data` | `405886d45b332e53a96d3d6212df82881a4962af9fb6d8ab0f11bbea8734a5ec` |
| `/tmp/profile-factor-f360be0-d33-high-v2-lbr.symbols.txt` | `95fd5114142cc3aee9fd88b47061ba84922e0e4c7fc300a1b7a76cb5dc876db4` |
| `/tmp/profile-factor-f360be0-d33-high-v2-lbr.children-symbols.txt` | `ef0af92422b26ec90b4a675227df2f06c42640ec09600d4f0f286c7c35b913bc` |
| `/tmp/profile-factor-f360be0-d33-high-v2.csv` | `86b76f2868d5f6b1d8b642cda6bc3ba56c15706196a08bb3fe364b13636b810d` |

### Accepted quadratic Hensel lift

Commit `4a2b9c7` adds a guarded quadratic two-factor lift. Each round lifts both the factors and
their Bezout cofactors from modulus `m` to `m*q`, with `q=m` for a full precision-doubling round
and `q=max_p/m` for the final partial round. It preserves the target's exact non-monic leading
coefficients. The original linear path remains active for `p=2`, fewer than 64 base-prime digits,
and every multi-factor tree whose root has more than four modular factors. The root permission is
propagated unchanged so a large tree cannot reactivate quadratic lifting at a smaller child.

The private `IntegerModularUnivariateContext` performs correction arithmetic modulo the current
prime power. It sends polynomial products through the optimized integer multiplication dispatcher,
then symmetrically reduces coefficients once. Its dense long division computes one inverse of the
divisor's unit leading coefficient and uses fused integer subtraction/multiplication. This replaced
generic pairwise `FiniteField<Integer>` polynomial arithmetic in the quadratic rounds.

The stable candidate matrix used six sequential, balanced-order, 20-sample processes per binary;
degree 64 used twelve. `v2` is the guarded quadratic lift before the integer modular context, and
`control` is `f360be0` before quadratic lifting:

| Case | Accepted Symbolica | FLINT | S/F | versus v2 | versus control |
|---|---:|---:|---:|---:|---:|
| high-height degree 33 | `10.105479 ms` | `1.409566 ms` | `7.068516` | `-29.53%` | `-73.04%` |
| degree 63 | `15.736829 ms` | `3.096926 ms` | `5.091038` | `-3.86%` | `-5.53%` |
| degree 64 | `21.730159 ms` | `2.572684 ms` | `8.432354` | `-1.23%` | `-2.93%` |
| degree 65 | `22.700640 ms` | `3.049224 ms` | `7.445458` | `-1.70%` | `-2.02%` |
| generated 2-variable factor | `6.139238 ms` | `9.833037 ms` | `0.624806` | `-0.69%` | `-0.57%` |
| generated 3-variable factor | `8.263755 ms` | `4.167803 ms` | `1.967807` | `+0.17%` | `-1.79%` |
| PolyBench #105 | `33.819908 ms` | `29.204045 ms` | `1.155242` | `-0.65%` | `-7.73%` |
| PolyBench #178 | `42.827747 ms` | `14.389792 ms` | `2.971608` | `-0.09%` | `-0.12%` |

Only the degree-33 change is attributed to the modular operation context; quadratic lifting is
disabled by the root factor-count guard for degrees 63, 64, and 65. Their small movements are LTO
layout/noise and are retained only as regression evidence. The first source-matched integrated run
was performed while another compiler saturated the host and FLINT medians shifted by 40-100%; do
not use its absolute times. Its high-height ratios were `6.55` integrated, `6.74` isolated, and
`24.41` control, which independently confirms the structural win. Raw contaminated files are
`/tmp/quadratic-hensel-integrated-*-block-*.csv`; rerun them on an idle host before replacing the
stable table.

A 500-sample v3 LBR profile measured `10.293209 ms` versus `1.506839 ms`, ratio `6.830995`.
Hensel lifting fell from about 74% of Symbolica's factor call in v2 to about 56% in v3. In absolute
terms its estimated contribution fell from roughly `10.4 ms` to `5.7 ms`. Generic
`FiniteField<Integer>` polynomial multiplication disappeared. The remaining quadratic-lift cost is
principally `IntegerModularUnivariateContext::quot_rem` and coefficientwise `symmetric_mod`;
modular factorization is now about 44% of the Symbolica call.

| Artifact | SHA-256 |
|---|---|
| `/tmp/flint-comparison-dev-4a2b9c7-screen` | `a4f671b28e2c97f0411f8d02f8e35c033d6c4a1eb58ae2a4deb99a74c8ebe2cf` |
| `/tmp/flint-comparison-dev-4a2b9c7-screen.d` | `6d3a63a3bd02be4f8b9ab157f5d57856926ba26d2bdc1080c73fb0efcdc11f87` |
| `/tmp/flint-comparison-dev-4a2b9c7-build.jsonl` | `2059b48064ea6de91defb79a61270682f393f3cdf2bf7ae132a75d5374111339` |
| `/tmp/flint-comparison-quadratic-hensel-v3-screen` | `8944f90e972b149bf6fce303608ef210eafe826786346ad1669ff42dd9b637e8` |
| `/tmp/flint-comparison-quadratic-hensel-v3-build.jsonl` | `fcc941b63a7adf8b05ee5a23ed30a5737256fcd444b9a80693958567c04c7145` |
| `/tmp/profile-factor-quadratic-hensel-v3-d33-lbr.perf.data` | `e3194a871f98a0a993bb4f1b038886f29169eadf9c960e22b9d4c4d606518f95` |
| `/tmp/profile-factor-quadratic-hensel-v3-d33-lbr.symbols.txt` | `448e79302c68a928d2281e6bd7c81f604e541166623f69a5c0949592087c9624` |
| `/tmp/profile-factor-quadratic-hensel-v3-d33-lbr.children-symbols.txt` | `01503aeefe8ba241bd3ed0cc58be7c7a9de32bff6e048c11124866b6f50aade5` |
| `/tmp/profile-factor-quadratic-hensel-v3-d33.csv` | `abd1d0f742e9d92bb80419211d92d9ed947c6527e936205532180bd0b73319d2` |

Stable raw timing files are `/tmp/quadratic-hensel-v3-d33-block-*.csv` and
`/tmp/quadratic-hensel-v3-matrix-*-block-*.csv`.

### Modular factorization attribution and revised degree-64 target

The degree-63 profile's 15.91% flat `quot_rem_univariate_monic` cost is attributed rather than
guessed. For its selected prime `65_000_011`, distinct-degree factorization calls
`exp_mod_univariate` for degrees 1 through 15. The 26-bit prime has popcount 17, so each modular
exponentiation makes 43 monic remainder calls. This gives 645 DDF exponentiation calls, four DDF
factor-extraction divisions, and six EDF exact divisions. About 98.5% of that image's
modular-factorization monic calls therefore come from DDF exponentiation.

The caller report agrees: 14.01 percentage points of the 15.91% flat cost lie below DDF, 13.59
points specifically pass through `quot_rem_univariate` from exponentiation, 0.42 points are DDF
factor extraction, and 0.87 points are Hensel lifting. The concise report is
`/tmp/profile-factor-dca8bd1-lbr.monic-callers-concise.txt`, SHA-256
`650a0140fbf11c992d70fb5eb23e4db500f747ba65913d0457dad601a5217530`.

Before `4f3b591`, both degree 63 and degree 64 performed about 945 DDF remainder calls across four
prime trials because every trial completed DDF and EDF before the selector compared factor counts.
Degree 64 tries primes `7`, `13`, `17`, and `65_000_011`, which yield `9`, `8`, `7`, and `20`
factors; it retains `p=17`. For the final large-prime trial, the degree-one and degree-two DDF
blocks already imply `4+15=19` factors and one nonconstant residual, so the exact lower bound is
20. It can be rejected after about 86 remainder calls instead of completing 645 calls and then
EDF-splitting 20 discarded factors.

### Accepted bounded DDF screening and deferred EDF

Commit `4f3b591` stores distinct-degree blocks with the exact count
`sum(block_degree/distinct_degree)`. A bounded DDF returns as soon as the completed count plus one
for a nonconstant residual exceeds an inclusive selector limit. Unsuitable primes remain distinct
from suitable images that exceed the limit, preserving suitable-prime accounting and the original
first-suitable direct-prime rule. Only the selected image undergoes equal-degree factorization.

The degree-64 tests prove the exact counts `9/8/7/20`, reject `65_000_011` at DDF degree two with
lower bound 20, select `p=17`, execute one EDF completion, and reconstruct six square-free integer
factors of degrees `1/1/2/10/20/30`. The generic bounded-DDF test also covers mixed degree-one and
degree-two blocks, exact-count admission, strict rejection, and the internal monic-one empty-block
representation while preserving the legacy public constant result.

The full-LTO candidate used balanced sequential process order and 20 samples per backend. Degree
64 used twelve processes per binary; the other rows used six. The last column compares paired
ratios, which is more reliable than absolute time for the thermally variable degree-33 processes:

| Case | Accepted Symbolica | FLINT | S/F | Control S/F | ratio change |
|---|---:|---:|---:|---:|---:|
| high-height degree 33 | `7.054343 ms` | `1.547683 ms` | `4.393851` | `7.130152` | `-38.37%` |
| degree 63 | `14.936981 ms` | `3.197383 ms` | `4.563498` | `5.470708` | `-16.58%` |
| degree 64 | `14.528937 ms` | `2.568576 ms` | `5.660441` | `8.357020` | `-32.27%` |
| degree 65 | `23.648133 ms` | `3.169588 ms` | `7.500939` | `7.342534` | `+2.16%` |
| generated 2-variable factor | `6.295513 ms` | `10.116293 ms` | `0.622243` | `0.623644` | `-0.22%` |
| generated 3-variable factor | `8.475188 ms` | `4.318369 ms` | `1.963720` | `1.961584` | `+0.11%` |
| PolyBench #105 | `31.716480 ms` | `29.069148 ms` | `1.091498` | `1.142249` | `-4.44%` |
| PolyBench #178 | `41.720974 ms` | `14.077220 ms` | `2.969817` | `2.961093` | `+0.29%` |

For degree 64, where the FLINT medians match closely between binaries, absolute Symbolica time is
`21.586502 -> 14.528937 ms`, a 32.69% reduction. The inferred DDF call reduction was about
`945 -> 386`; the measured whole-row result confirms the original 20-35% opportunity estimate.
The high-height absolute medians experienced frequency drift, but its paired ratio improvement is
large and consistent with eliminating discarded modular factorizations.

A 500-sample LBR profile measured `14.744165 ms` versus `2.601388 ms`, ratio `5.667807`. Of the
paired cycles, 76.53% are in Symbolica factorization. Relative to that Symbolica subtree, Hensel
lifting is about 62.17% (`~9.17 ms`), modular screening is 28.42% (`~4.19 ms`), DDF itself is
27.49% (`~4.05 ms`), and retained EDF is 8.09% (`~1.19 ms`). Hensel lifting is therefore the next
primary target; the dense DDF context no longer has the largest ceiling.

| Artifact | SHA-256 |
|---|---|
| `/tmp/flint-comparison-factor-screening-v1-screen` | `7e96241378b4e8e93cdc19e2534c5e75694796eae31cd9bebdc3868baf11c6f5` |
| `/tmp/flint-comparison-factor-screening-v1-screen.d` | `c8c0444a41613cf70d21e8f72745eb50f2810d7ab3dd63bbf69de10f75429a84` |
| `/tmp/flint-comparison-factor-screening-v1-build.jsonl` | `c94dc3dbad0f4b43f7f35c74b7894887599e19d1767f82c8b6a97cd482528495` |
| `/tmp/profile-factor-screening-v1-d64-lbr.perf.data` | `f5046c2999babcb550ea0497c01baac01b3a52c47326186cfc5a5c6e5c144ff9` |
| `/tmp/profile-factor-screening-v1-d64-lbr.symbols.txt` | `551c50b63716ce41a5f57224132def4388473ad353ba230471951c6cf71a7886` |
| `/tmp/profile-factor-screening-v1-d64-lbr.children-symbols.txt` | `574a34aa3913c2c0d93886312b3ec0f5dcad8e477d877947aa24ce0d25c1757c` |
| `/tmp/profile-factor-screening-v1-d64.csv` | `3e447b96fb6f028e450fe92b7a278bdfa9778923b2bdef25ffff67082734cdc7` |

The release build took 9m27s. Its pre-commit source diff SHA-256 is
`3ee2b7d5f052e9976e521d9a5dd0f84a81141bb553e6315ad15d7ce4bf8c4124`; that content is commit
`966deda`, cherry-picked unchanged as `4f3b591`. Raw timing files are
`/tmp/factor-screening-v1-{d33h,d63,d64,d65,factor2,factor3,pb105,pb178}-*-block-*.csv`.

### Rejected 500M modular-prime experiment

Prime `500_000_003` has the same seven modular factors as `p=17`, but the primitive degree-64
fixture needs only 11 linear-lift digits instead of 77. The experiment admitted `u64` accumulation
using the exact per-output collision bound `min(left_len, right_len)`, tested both dense and
touched-slot buffers on a 129-by-65 maximum-residue product, and selected the wider prime only for
degree 64. The intended Hensel saving was real, but high-bit modular factorization cost more than
it saved.

Twelve alternating 50-sample process pairs measured:

| Path | Symbolica | FLINT | S/F |
|---|---:|---:|---:|
| accepted bounded-DDF control | `14.497333 ms` | `2.572823 ms` | `5.621938` |
| 500M prime | `19.630564 ms` | `2.575012 ms` | `7.638689` |

Every process pair regressed. Symbolica was 35.41% slower and the paired ratio was 35.87% worse,
while the FLINT median changed by only 0.09%. A separate 500-sample LBR profile measured
`19.680943 ms` versus the control profile's `14.744165 ms`. Normalizing immediate children below
the measured `factor_reconstruct` subtree gives the following approximate stage costs:

| Stage | accepted control | 500M prime | change |
|---|---:|---:|---:|
| Hensel lifting | `9.17 ms` | `1.26 ms` | `-7.91 ms` |
| modular-prime screening | `4.19 ms` | `10.00 ms` | `+5.81 ms` |
| retained EDF | `1.19 ms` | `8.27 ms` | `+7.08 ms` |

DDF is nested inside prime screening and must not be added to it. Recursive global LBR rows also
double-count repeated frames; use the outermost immediate-child shares for comparisons. The flat
`DenseZpMul::multiply_direct_u64` estimate rose from about `1.54 ms` to `6.74 ms`, and monic
finite-field remainder work rose from about `3.15 ms` to `6.21 ms`. Thus the 500M policy moved the
bottleneck out of Hensel lifting and into DDF/EDF powering and division. Do not retry a wider
factorization prime without an independently faster high-bit DDF/EDF implementation or a selector
that prices modular work rather than only factor count and lifting digits.

| Artifact | SHA-256 |
|---|---|
| `/tmp/flint-comparison-factor-prime500-v1-screen` | `2211c2c541b9bb1540aa45043668a1016fc919c75f3c69663caa3e92a9070690` |
| `/tmp/flint-comparison-factor-prime500-v1-screen.d` | `8896a6598fc26f38403156d627be5145a8bc5a59d8a176fed12f62a002db2613` |
| `/tmp/flint-comparison-factor-prime500-v1-build.jsonl` | `6cceb0f3a3821c2151c47c261b4321bd2a9cb34c945fe3944bdaee0a7cbd8ea5` |
| `/tmp/profile-factor-prime500-v1-d64-lbr.perf.data` | `f0c7d283f3788288aad4a668d313602a0fb298213bdf132dd7d6bc9848d13969` |
| `/tmp/profile-factor-prime500-v1-d64-lbr.symbols.txt` | `b4efb9cdf5cb08a0ecc4ed40f203d3cf1e1e319eb54dfb6cd8a9fa63707e686e` |
| `/tmp/profile-factor-prime500-v1-d64-lbr.children-symbols.txt` | `9bd8e693ceb1d791bf1e7ee52c31e1ffb0fa45d9f7bfadfbeb6d3006add7dc70` |
| `/tmp/profile-factor-prime500-v1-d64.csv` | `ef27c6b7d0387365451f270a238ead35893a33937424c3b0a95f97102785b135` |

### Accepted per-output finite-field accumulator bound

Commit `303381c` replaces `DenseZpMul`'s total-pair overflow estimate with the exact maximum number
of products contributing to one output coefficient. Strictly increasing input indices imply at
most `min(left_len, right_len)` collisions. A named strategy mode distinguishes direct Montgomery
reduction, `u64` plus a native remainder, and the existing `u128` fallback. Differential tests use
129-by-65 maximum raw residues with dense and touched-slot layouts at both `p=65_000_011` and
`p=500_000_003`; the small-prime direct route and near-`u32::MAX` fallback are also asserted.

The paired benchmark infrastructure now contains a dedicated dense-univariate degree-128 by
degree-64 case for those two fields. Twelve alternating 1,000-sample process pairs measured:

| Field and route | old Symbolica | new Symbolica | FLINT with new binary | old S/F | new S/F |
|---|---:|---:|---:|---:|---:|
| `GF(65000011)`, direct Montgomery | `12.254 us` | `6.930 us` | `7.238 us` | `1.539` | `0.960` |
| `GF(500000003)`, native remainder | `12.264 us` | `8.137 us` | `7.959 us` | `1.395` | `1.023` |

The absolute Symbolica gains are 43.4% and 33.6%; the paired-ratio gains are 37.6% and 26.7%.
FLINT itself varied between the separately linked binaries, so retain both absolute and paired
comparisons. Degree-64 factorization was neutral (`14.503 ms` versus `14.487 ms`) because its 65M
image's 4,225 total products already fit the old bound. The degree-33/63/65, generated 2/3-variable,
PolyBench #105/#178, GF(17) direct/KS, and 64-bit-field guards were all neutral; the largest
observed Symbolica guard movement was +2.1% on the generated three-variable factor row.

| Artifact | SHA-256 |
|---|---|
| `/tmp/flint-comparison-zp-collision-bench-v1-screen` | `73d74bfedbd9ffb1531e58260368c47aac5a41ec91df82e111596013ac6459d7` |
| `/tmp/flint-comparison-zp-collision-bench-v1-screen.d` | `8896a6598fc26f38403156d627be5145a8bc5a59d8a176fed12f62a002db2613` |
| `/tmp/flint-comparison-zp-collision-bench-v1-build.jsonl` | `d32859f538ff62c95289ad0b2d1985320e6b74a8db20e5e2a3cee05956a1f951` |
| `/tmp/flint-comparison-zp-collision-old-bound-v1-screen` | `572c8fadc5dbfa8c0d8a103abf8b1482437fe3fca5978f72371aa1f1f51fca0b` |
| `/tmp/flint-comparison-zp-collision-old-bound-v1-build.jsonl` | `ebf195a1e61ffc73247ee4c51906e17e08afb8a7fbab5c7461eaebc3f45940cd` |

Raw files are `/tmp/zp-collision-bench-v1-p{65000011,500000003}-*-block-*.csv` and
`/tmp/zp-collision-bench-v1-{d33h,d63,d65,factor2,factor3,pb105,pb178,gf17_*,gf64_*}-*-block-*.csv`;
degree 64 is `/tmp/zp-collision-v1-d64-*-block-*.csv`.

### Rejected root-only quadratic Hensel experiment

The experiment retained recursive quadratic lifting for at most four modular factors, permitted
one quadratic lift at the root for five through eight factors, and forced all descendants back to
linear lifting. Routing and reconstruction tests passed under the default and `no_gmp` feature
sets, including exact call-count checks at the four-, five-, eight-, and nine-factor boundaries.
It nevertheless made the seven-factor degree-64 fixture consistently slower.

Twelve alternating 50-sample process pairs measured:

| Path | Symbolica | FLINT | S/F |
|---|---:|---:|---:|
| accepted control | `13.852066 ms` | `2.568130 ms` | `5.401450` |
| root-only quadratic lift | `14.657123 ms` | `2.566990 ms` | `5.707710` |

Symbolica regressed by `0.805057 ms`, or 5.81%, while FLINT was stable. The candidate process
medians occupied the narrow range `14.596027..14.785413 ms`. The median of each process's minimum
also regressed by 9.65%, so the result is not explained by the fixture's occasional fast mode.

A separate 500-sample LBR profile measured `14.741254 ms` versus `13.887101 ms`. Immediate-child
accounting attributes essentially the full loss to the direct lift at the root:

| Stage | accepted control | root-only candidate | change |
|---|---:|---:|---:|
| top multi-factor Hensel subtree | `8.9106 ms` | `9.7900 ms` | `+0.8793 ms` |
| recursive multi-factor child | `7.4456 ms` | `7.4737 ms` | `+0.0281 ms` |
| direct root Hensel lift | `1.4632 ms` | `2.3125 ms` | `+0.8493 ms` |
| modular-prime screening | `3.6135 ms` | `3.5622 ms` | `-0.0512 ms` |
| retained EDF | `1.1371 ms` | `1.2020 ms` | `+0.0650 ms` |

The candidate direct root lift spends about `1.6428 ms` in
`IntegerModularUnivariateContext::quot_rem`, including about `1.1353 ms` in `symmetric_mod` and
`0.8739 ms` in `Integer::rem`; these rows are nested and must not be added. The corresponding
finite-field quotient in the control costs about `0.2332 ms`. Candidate multiplication and
subtraction became cheaper, but not enough to offset arbitrary-modulus quotient/reduction. Reject
this policy: a quadratic root lift is only useful after its integer modular quotient path is made
substantially cheaper.

| Artifact | SHA-256 |
|---|---|
| `/tmp/flint-comparison-factor-root-only-v1-screen` | `d4f12840b868dfb50b6a73151370dbc93369d6bb2bfd03c492e8567d6e485863` |
| `/tmp/flint-comparison-factor-root-only-v1-screen.d` | `6c6a050ca594f945542bafd61069ff932c58d01a285ed15edd735ec909ed623b` |
| `/tmp/flint-comparison-factor-root-only-v1-build.jsonl` | `728a0d6cbdfc7cd0db97669a4026420b87db97f4b189fd9c09189a1889f47477` |
| `/tmp/profile-factor-root-only-v1-d64-candidate-lbr.perf.data` | `f1e496da7aabfb933af640252886ed10f098b1478102b44d1f561598d9f921dc` |
| `/tmp/profile-factor-root-only-v1-d64-candidate-lbr.symbols.txt` | `3f6790d7f913af0353b07885660a037a53f43140b2e2d18d3e211323e9e47af0` |
| `/tmp/profile-factor-root-only-v1-d64-candidate-lbr.children-symbols.txt` | `e4eeb6270859727222437fe34623eb256bc70a048a64abd4267e251279bfe603` |
| `/tmp/profile-factor-root-only-v1-d64-candidate.csv` | `130c04c7428e1d7839549ff53869094692edd3c04bd7eeca85f92dfb69c428ee` |
| `/tmp/profile-factor-root-only-v1-d64-control-lbr.perf.data` | `345dc15bb4a6e15892f6171080e2708fdc4969a05e0e33b0d2386483044046a9` |
| `/tmp/profile-factor-root-only-v1-d64-control-lbr.symbols.txt` | `74d0487bc6059aec0f0c91140bb15e4d89c8499fe9939cdb11d273eae940b31c` |
| `/tmp/profile-factor-root-only-v1-d64-control-lbr.children-symbols.txt` | `12cc8cea50313560a0465c239c81133b909a2824e3646e656dc8c5e29688557d` |
| `/tmp/profile-factor-root-only-v1-d64-control.csv` | `e92636bcba89d46e6b91a071b635effadaf779c1d3cc9f5836dcfa4449fe3f71` |

The sorted SHA-256 manifests for the twelve candidate blocks, twelve control blocks, and all 24
blocks are respectively `a912da263bbea52ca65533e9fb778133734cde6ae6a48680fff1460b778e0796`,
`3f77e40d08d85e2e4548889583f5e31da0f9829be660eba455e1eef6911933f4`, and
`ad5d372058a85d2e25c88f6dfaccee8ea2a04fe45f8e3fd9b9cf8eddecb7efeb`. Raw files are
`/tmp/factor-root-only-v1-d64-{candidate,control}-block-*.csv`. The rejected source remains only in
`/tmp/symbolica-factor-root-only`; it was not integrated.

### Exact-subproblem modulus audit

`hensel_lift_with_strategy` returns `Ok` only after the integer residual is literally zero, so an
`Ok` split of an already exact target certifies exact integer child products. Those children may
use independently computed coefficient bounds. A child-specific modulus cannot, however, be
introduced by merely changing the recursive `max_p`: the existing routine returns monic modular
leaves that top-level recombination interprets modulo the global modulus. A leaf lifted only modulo
the smaller child modulus is not generally a valid representative modulo the global modulus, and
mixed exact nonmonic children violate the same recombination assumptions.

The safe implementation is an exact-subproblem solver. It recursively treats `Ok` children as
exact targets with local bounds; when a node returns `Err`, it completes that target's descendant
lifts at one local modulus and recombines them locally against the exact target before returning
exact integer factors. The existing LLL/subset recombination tail should first be extracted into a
helper that explicitly consumes one exact target and one consistently lifted modular-factor set.
Calling `factor_reconstruct` again on every exact child is correct but needlessly repeats prime
selection, DDF, and EDF.

For the degree-64 fixture at `p=17`, the current Gelfond bounds are:

| Exact target | bound bits | base-17 digits |
|---|---:|---:|
| full degree 64 | `312` | `77` |
| root children, degrees 4 and 60 | `13`, `300` | `4`, `74` |
| lucky right children, degrees 20 and 40 | `91`, `215` | `23`, `53` |

The modular leaves have degrees `[1,1,2 | 10,10,10,30]`, so the root `4|60` split is always exact.
Only one of the three equal-degree-factor orderings pairs the two degree-10 leaves belonging to the
integer degree-20 factor. Successful exact splits already stop as soon as the residual is zero, so
their lower ceiling alone saves no lift rounds. In the ordinary unlucky ordering, the failed right
split only falls from 77 to 74 digits. Consequently the safe local-modulus experiment has modest
median-performance confidence; its largest saving occurs in the already lucky ordering, where a
failed descendant inside the exact degree-20 target can fall from 77 to 23 digits.

The larger strategic target is coherent simultaneous linear Hensel lifting over a product tree. It
would update all leaves and product nodes for each p-adic digit rather than attempting fake binary
integer factorizations at unlucky intermediate partitions. The observed fast/slow spread is about
5.3 ms, so this has a materially larger ceiling than local bounds, at the cost of a substantially
larger implementation. Tree sorting alone is unjustified because the three degree-10 modular
factors are indistinguishable over `GF(17)`. A stabilization/exact-division shortcut also needs
rational reconstruction or leading-coefficient allocation: nonmonic lifting scales both children
to the full target leading coefficient, so the small primitive factor does not simply stabilize.

A private dense univariate remainder context remains a later option. It would cache one monic
modulus, retain dense coefficient/index buffers through binary exponentiation, use the existing
coefficient-domain multiplication kernel, and materialize a polynomial only for each GCD. Its
current DDF ceiling is about 27.5% of Symbolica time, so reassess it after the prime experiment.
The square-free GCD remains Amdahl-limited and is not a useful factorization target.

## Benchmark infrastructure

Inputs are in `benches/support/cases.rs` and `benches/support/polybench_cases.rs`. Symbolica-only
rows are in `benches/symbolica_polynomial.rs`; paired Symbolica/FLINT rows are in
`benches/flint_comparison.rs`. Support code under `benches/support/{symbolica,flint,paired}.rs`
constructs the same cases for both libraries and validates outputs outside the timed region. The
harness fixes Rayon and FLINT to one thread, warms both sides, alternates execution order, and
reports median `Symbolica / FLINT` directly.

The installed build environment used for the measurements is:

```sh
export PATH=/nix/store/9pb4ikjdw4gp766ayxl6gg3b7hqm6ds4-rust-stable-with-components-2026-07-09/bin:/nix/store/vdaz6sk455r8sbpi7wzaf9vlz5i9yyvx-gcc-wrapper-15.2.0/bin:/nix/store/yqmfmarywhqadkkvd5w9zbz8lw9pzkyj-pkg-config-0.29.2/bin:$PATH
export PKG_CONFIG_PATH=/nix/store/k1hb5mgjbaiq1sx65zpbd9a1sfy0jl6n-flint-3.6.0/lib/pkgconfig:/nix/store/2vh1aj9bwhab9blr6wfm8v4mxd7nw15w-gmp-with-cxx-6.3.0-dev/lib/pkgconfig:/nix/store/yzf04s5m4s9w70wfqij2gba3421gnll4-mpfr-4.2.2-dev/lib/pkgconfig
export LD_LIBRARY_PATH=/nix/store/k1hb5mgjbaiq1sx65zpbd9a1sfy0jl6n-flint-3.6.0/lib:/nix/store/sy8ark85hgjhhcr1ycgarf5w7ajw8jcc-gmp-with-cxx-6.3.0/lib:/nix/store/bdvnvbidl34l6pvng6587m6axbzwz9hi-mpfr-4.2.2/lib
```

Build with full release settings and save the JSON artifact path:

```sh
set -o pipefail
CARGO_TARGET_DIR=/tmp/symbolica-univariate-dense-div/target \
  CARGO_BUILD_JOBS=1 \
  cargo build --release --features flint_system_benchmarks \
  --bench flint_comparison --message-format=json-render-diagnostics \
  | tee /tmp/flint-comparison-build.jsonl
jq -r \
  'select(.reason == "compiler-artifact" and .target.name == "flint_comparison" and .executable != null) | .executable' \
  /tmp/flint-comparison-build.jsonl | tail -1
```

The shared target is `/tmp/symbolica-univariate-dense-div/target`; its commonly selected hashed
binary is `release/deps/flint_comparison-0df1dfb2f578e077`. A later build overwrites it. Always copy
the executable and matching `.d` sidecar to a source-specific `/tmp` name immediately. Full LTO
usually takes 9-11 minutes. The printed Symbolica version can be stale because build-script output
is reused; Git state, build JSON, dependency sidecar, and SHA-256 are authoritative provenance.

One product process is:

```sh
taskset -c 8 env \
  -u GCD_BENCH_CASE -u GCD_BENCH_NVARS -u GCD_BENCH_DEGREE \
  -u GCD_BENCH_GAP -u GCD_BENCH_COEFFICIENT_BITS \
  SYMBOLICA_FLINT_BENCH_PAIRED=1 \
  SYMBOLICA_FLINT_BENCH_FILTER='generated factor product: dense 1-variable degrees 32/31' \
  SYMBOLICA_FLINT_BENCH_SAMPLES=5000 \
  SYMBOLICA_FLINT_BENCH_CSV=1 \
  /tmp/flint-comparison-candidate > /tmp/product-candidate-run-01.csv
```

One factor process is the same except for filter and sample count:

```sh
taskset -c 8 env \
  -u GCD_BENCH_CASE -u GCD_BENCH_NVARS -u GCD_BENCH_DEGREE \
  -u GCD_BENCH_GAP -u GCD_BENCH_COEFFICIENT_BITS \
  SYMBOLICA_FLINT_BENCH_PAIRED=1 \
  SYMBOLICA_FLINT_BENCH_FILTER='generated factorization: dense 1-variable degrees 32/31' \
  SYMBOLICA_FLINT_BENCH_SAMPLES=20 \
  SYMBOLICA_FLINT_BENCH_CSV=1 \
  /tmp/flint-comparison-candidate > /tmp/factor-candidate-run-01.csv
```

One configurable degree-64 GCD process is:

```sh
taskset -c 8 env \
  SYMBOLICA_FLINT_BENCH_PAIRED=1 \
  SYMBOLICA_FLINT_BENCH_FILTER='GCD auto: dense 1 variables degree 64' \
  SYMBOLICA_FLINT_BENCH_SAMPLES=200 \
  SYMBOLICA_FLINT_BENCH_CSV=1 \
  GCD_BENCH_CASE=dense GCD_BENCH_NVARS=1 GCD_BENCH_DEGREE=64 \
  GCD_BENCH_GAP=10 GCD_BENCH_COEFFICIENT_BITS=30 \
  /tmp/flint-comparison-candidate > /tmp/d64-candidate-run-01.csv
```

The presence of `SYMBOLICA_FLINT_BENCH_PAIRED`, even with value `0`, enables paired mode. Use even
sample counts to balance alternating order. Clear stale `GCD_BENCH_*` variables before product or
factor rows because paired setup constructs the configurable GCD case before filtering.

Perf is available at:

```text
/nix/store/lc3a14lcfrbii8f008kjgl45xibjm1w8-perf-linux-7.1.3/bin/perf
```

Use `cycles:u`, 999 Hz, and `--call-graph lbr` for profiles. Save stdout and `sha256sum` for both
the perf data and generated reports.

## Rejected or low-value experiments

Do not repeat these without a genuinely new mechanism:

| Experiment | Evidence | Decision |
|---|---|---|
| Direct 1-4-limb `Integer` to `Zp64` conversion, `8440390` | degree-64 median `1.179371` versus `1.191412`; only about 1% | correct but Amdahl-limited; leave off `dev` |
| Fused adjacent-degree/Q1 modular remainder, `0f7fa97` | ratio regressed to about `1.484` | reject; three-limb reduction cost exceeds two Montgomery reductions |
| Dense single-scale Zippel reconstruction, `c4748b6` | PolyBench #11 improved only about 1.3% | too much code for gain |
| Broad dense modular GCD experiment, `5a975f1` | little gain at degree 64, degree-80 regression | superseded by smaller dense-image/certificate contexts |
| Borrowed versus by-value finite-field calls | no measurable difference | not the source of the old unexplained `1.325` result |
| Divide-and-conquer univariate attempt | degree-64 ratio about `2.736` | decisive regression |
| Direct-limb conversion as the whole product answer | current product profile shows packing/unpacking, not GMP multiply, dominates | target conversion as an operation context; do not redesign `Ring` |
| Large-integer recycled storage alone | reduces allocation-heavy paths but did not close FLINT gaps | retain it, but algorithm/data layout matter more |
| Product selector `85be422` as a factorization fix | product improved 16-17%; factor row did not materially move | factor input product is outside timed region |
| Square-free/GCD tuning for the 1-variable factor fixture | about 0.5% of Symbolica factor time | not a plausible explanation for a 7x gap |
| Automatic CRT resultant | not generally competitive with direct Ducos | keep explicit `resultant_crt`, not default |
| Brown PRS as general resultant default | not competitive regime-wide | keep explicit `resultant_brown` |
| Sparse multivariate resultant interpolation prototype | reconstruction/bound work overwhelmed sparsity | not used as the general practical method |
| Unguarded quadratic Hensel lifting | high-height degree 33 improved to about 14 ms, but degree 63 regressed from about 16 ms to 34-37 ms and degree 64 to about 35 ms | superseded by the 64-digit and root-wide four-factor guard in `4a2b9c7` |

Historical frozen binaries and perf data remain under `/tmp`; it is ephemeral. The most useful
older checksums and ratios are retained in Git history of this document at commits `386174d` and
`3f2d7eb` if deeper attribution is needed.

## Code map and design preferences

- Integer multiplication dispatch and Kronecker conversion:
  `lib/numerica/src/domains/integer/polynomial_kernels.rs`, especially `DenseIntegerMul` and
  `try_kronecker`.
- Integer multiplication differential tests:
  `lib/numerica/src/domains/integer.rs`.
- Dense univariate modular GCD and exact certificate contexts:
  `src/poly/gcd.rs`, especially `DenseZp64UnivariateGcdImage`,
  `DenseUnivariateIntegerDivisionContext`, and `UnivariateModularGcdContext`.
- Integer factor selection/reconstruction/Hensel lifting: `src/poly/factor.rs`; the active pressure
  selector is around `high_linear_lift_pressure`.
- Shared polynomial operation kernels: `src/poly/kernels.rs`.
- Generated and PolyBench fixtures: `benches/support/cases.rs` and
  `benches/support/polybench_cases.rs`.

Prefer short-lived operation-context structs that own reusable buffers and precomputed metadata.
Keep multiplication kernels attached to the integer domain where they need tagged integer/GMP
knowledge; call them from polynomial dispatch after the ring type is known. Do not add narrowly
specialized methods to `Ring`. `exact_div_owned`-style ownership-aware operations are useful when
their semantics are general; coefficient-domain-only fast paths belong in an operation context or
a narrow trait, not the base ring interface.

Comments should say what a function computes, its input invariants, and where it is used. Avoid
comments that justify file organization by contrasting it with designs not present in the code.

## Ordered next actions

1. Implement coherent simultaneous linear Hensel lifting over a degree-greedy product tree. Reuse
   products and residual data across leaves at each p-adic precision so an unlucky intermediate
   partition does not trigger a full fake binary factorization. Start with monic, pairwise-coprime
   univariate factors and preserve the existing binary fallback. For fixture degrees
   `[1,1,2,10,10,10,30]`, assert the degree-greedy internal degrees `[2,4,14,20,34,64]` and sum
   `138`; the current count-split tree's sum is `191`.
2. Use synchronized precision stages equivalent to `1 -> 2 -> 3 -> 5 -> 10 -> 20 -> 39 -> 77`
   base-prime digits on the degree-64 fixture. Walk top-down so every child target is at the newly
   reached precision. Use the two-monic-remainder factor and Bezout updates; a prototype using the
   current generic composite-modulus quotient/remainder is a correctness step, not a
   performance-ready endpoint.
3. A smaller independent experiment is the fused dense update
   `residual - tau*w - r*u'`. Put it behind `PolynomialKernels` and implement the integer operation
   through a short-lived context that admits the whole request before consuming coefficients. Its
   realistic profile ceiling is only 2-5%, so reject it if repeated whole-factor timings do not
   move.
4. For the high-height degree-33 path, the smallest measured follow-up is to defer coefficient
   reduction inside `IntegerModularUnivariateContext::quot_rem`: leave non-pivot remainder cells
   unreduced during fused subtractions and symmetrically reduce only pivots and the final
   remainder. In the v3 profile quotient/remainder is about 52% of Hensel, and about 70% of that
   subtree is `symmetric_mod`. Differential-test this at small and large composite prime powers
   before benchmarking; do not infer the full profile ceiling as an expected gain.
5. Keep the private dense DDF remainder context as a later option. Bounded prime screening reduced
   DDF's current degree-64 ceiling to about 27.5%, and both the 500M prime and root-only quadratic
   experiments show that moving work between modular factorization and Hensel lifting is not itself
   a win.
6. Keep the one-variable product path as a regression guard: its target factor-fixture product is
   about 9.5% slower than FLINT in the latest source-matched build. Return to dense univariate GCD
   only after the degree-64 factor
   prime/Hensel decision; current degree-64 GCD is about 19% slower than FLINT.
7. Freeze and hash every accepted full-LTO binary and profile, integrate only measured winners with
   Ben Ruijl's identity, and keep this file current after every accepted or rejected experiment.
