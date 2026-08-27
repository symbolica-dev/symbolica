# Polynomial GCD and factorization performance handoff

This is an operational handoff for the single-core Symbolica/FLINT performance work. It records
the state on 2026-08-27, including code already on `dev`, isolated candidates, measurements,
unsuccessful experiments, exact benchmark recipes, and the next experiments to run. It is intended
to let a new agent continue without needing the preceding conversation.

The current priority order is:

1. dense univariate integer GCD at degree 64;
2. the one-variable integer polynomial product used by the factor benchmark;
3. one-variable integer factorization;
4. the remaining multivariate GCD and factorization losses.

Do not introduce runtime parallelism while pursuing these items. The comparison target is
single-core FLINT performance.

## Immediate resume point

Do not restart the degree-64 investigation from the scalar baseline, and do not resume the direct
small-limb conversion experiment. That experiment is complete and rejected for only about 1%
end-to-end improvement.

Resume in `/tmp/symbolica-zp64-r2`, despite the old worktree name. It is clean, is now on branch
`codex/zp64-hybrid-inverse`, and has head `b7727d9` (`Speed up dense Zp64 GCD inverses`). The
candidate replaces the generic dense-image leading-coefficient inverse with a private Euclidean
inverse that handles quotients 1, 2, and 3 by subtraction before using hardware division. Its first
200-sample degree-64 screen measured ratio `1.148490`, versus a reproducible `b4db253` control
median near `1.1914`. This is promising but not yet a decision-quality result: run four more
candidate processes, interleave fresh control processes, record the raw outputs, then capture an
LBR profile. The literal commands and acceptance checks are in
[Recommended next steps](#recommended-next-steps).

After the inverse verdict, integrate the accepted base chain through `b4db253` onto `dev` whether
or not the inverse passes. If the inverse gain reproduces, first finish its remaining `no_gmp` and
neighboring-regime validation, then append `b7727d9`:

```text
810875a -> ea09e01 -> f46fcff -> 7fdc099 -> b4db253 [-> b7727d9 if accepted]
```

Do not integrate `8440390` (direct limb conversion), `0f7fa97` (fused Q1), `bfed508` in addition to
`f46fcff`, or `f3b13b4` in addition to `b4db253`.

## Repository state at handoff

The main worktree is `/home/codexB/symbolica` on `dev`. Before this document update it was clean at
`386174d` and 54 commits ahead of `origin/dev`. Use `git status --short --branch` and
`git log -3 --oneline` rather than relying on those values after this document is committed. The
latest integrated polynomial-performance commit is `a3f721c` (`Specialize heuristic GCD for
univariate integers`); later inequality-solver and handoff commits are unrelated to the performance
implementation.

Use author and committer `Ben Ruijl <ben@ruijl.ch>` for commits made for this work.

The useful degree-64 candidate chain is not on `dev` yet:

```text
a3f721c  current integrated univariate heuristic
  |
810875a  modular univariate integer GCD
  |
ea09e01  direct dense Zp64 image GCD
  |
f46fcff  checked dense multiprecision certificate
  |
7fdc099  precomputed R^2 Montgomery conversion
  |
b4db253  cached verified univariate primes
  |
  +-- b7727d9  private dense-image inverse (active candidate)
  |
  +-- 8440390  direct 1-4-limb Integer -> Zp64 conversion (rejected)
```

`bfed508` is the certificate commit by itself on top of `810875a`; `f46fcff` is the same change
cherry-picked on top of `ea09e01`. Do not apply both. `f3b13b4` is the cached-prime change on top of
`f46fcff`; `b4db253` is the same change after `7fdc099`. Again, apply only the commit from the
chosen chain.

The important isolated worktrees are:

| Worktree | Branch/head | State | Purpose |
|---|---|---|---|
| `/tmp/symbolica-univariate-modular` | `codex/univariate-modular-gcd`, `810875a` | clean | Modular univariate integer GCD |
| `/tmp/symbolica-univariate-dense-div` | `codex/univariate-dense-zp64-gcd`, `ea09e01` | clean | Direct dense `Zp64` image GCD; its `target/` is also the shared release target |
| `/tmp/symbolica-univariate-borrowed-cert` | `codex/univariate-borrowed-divisor-cert`, `bfed508` | clean | Certificate-only candidate based on `810875a` |
| `/tmp/symbolica-univariate-combined` | `codex/univariate-combined`, `f46fcff` | clean | Dense field image plus certificate |
| `/tmp/symbolica-zp64-r2` | `codex/zp64-hybrid-inverse`, `b7727d9` | clean | Active private dense-image inverse candidate on the complete accepted base chain |
| `/tmp/symbolica-univariate-primes` | `codex/univariate-prime-table`, `f3b13b4` | clean | Cached-prime change without R^2; useful only for attribution |
| `/tmp/symbolica-zp64-limb-conversion` | `codex/zp64-limb-conversion`, `8440390` | clean | Complete direct 1-4-limb integer-to-`Zp64` experiment; measured gain too small, reject |
| `/tmp/symbolica-univariate-fused-q1` | `codex/univariate-fused-q1`, `0f7fa97` | clean | Correct but decisively slower fused adjacent-degree remainder; reject |
| `/tmp/symbolica-univariate-dense-modular` | `codex/univariate-dense-modular`, `5a975f1` | clean | Superseded large combined experiment; reject |
| `/tmp/symbolica-zippel-dense-image` | `codex/zippel-dense-image`, `c4748b6` | clean | Dense single-scale Zippel reconstruction; reject |

All candidate commits descend from `a3f721c`. A read-only sequential merge audit found no textual
conflicts when applying `810875a`, `ea09e01`, `f46fcff`, `7fdc099`, and `b4db253` to `dev` at
`386174d`; `b7727d9` is a direct child of that chain. Recheck after any new `dev` changes and run
the complete validation rather than treating a clean cherry-pick as proof of correctness. Every
listed committed candidate has author and committer `Ben Ruijl <ben@ruijl.ch>`.

Branch `codex/zp64-r2-conversion` still points cleanly at `b4db253`, but it is no longer checked
out: `/tmp/symbolica-zp64-r2` now belongs to `codex/zp64-hybrid-inverse`. Use the branch name or the
frozen control binary if the `b4db253` control must be rebuilt.

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

Build against the installed FLINT 3.6.0 with the shared release target. `CARGO_BUILD_JOBS=1`
limits this build to one compiler job, reducing peak memory and disk pressure:

```sh
set -o pipefail
export PATH=/nix/store/9pb4ikjdw4gp766ayxl6gg3b7hqm6ds4-rust-stable-with-components-2026-07-09/bin:/nix/store/vdaz6sk455r8sbpi7wzaf9vlz5i9yyvx-gcc-wrapper-15.2.0/bin:/nix/store/yqmfmarywhqadkkvd5w9zbz8lw9pzkyj-pkg-config-0.29.2/bin:$PATH
export PKG_CONFIG_PATH=/nix/store/k1hb5mgjbaiq1sx65zpbd9a1sfy0jl6n-flint-3.6.0/lib/pkgconfig:/nix/store/2vh1aj9bwhab9blr6wfm8v4mxd7nw15w-gmp-with-cxx-6.3.0-dev/lib/pkgconfig:/nix/store/yzf04s5m4s9w70wfqij2gba3421gnll4-mpfr-4.2.2-dev/lib/pkgconfig
export LD_LIBRARY_PATH=/nix/store/k1hb5mgjbaiq1sx65zpbd9a1sfy0jl6n-flint-3.6.0/lib:/nix/store/sy8ark85hgjhhcr1ycgarf5w7ajw8jcc-gmp-with-cxx-6.3.0/lib:/nix/store/bdvnvbidl34l6pvng6587m6axbzwz9hi-mpfr-4.2.2/lib
CARGO_TARGET_DIR=/tmp/symbolica-univariate-dense-div/target \
  CARGO_BUILD_JOBS=1 \
  cargo build --release --features flint_system_benchmarks --bench flint_comparison \
  --message-format=json-render-diagnostics | tee /tmp/flint-comparison-build.jsonl
jq -r \
  'select(.reason == "compiler-artifact" and .target.name == "flint_comparison" and .executable != null) | .executable' \
  /tmp/flint-comparison-build.jsonl | tail -1
```

Set `SYMBOLICA_LICENSE` from the task/user environment. Do not copy its value into source,
benchmarks, logs intended for commit, or this document.

The configurable dense degree-64 GCD comparison is:

```sh
taskset -c 8 env \
  SYMBOLICA_FLINT_BENCH_PAIRED=1 \
  SYMBOLICA_FLINT_BENCH_FILTER='GCD auto: dense 1 variables degree 64' \
  SYMBOLICA_FLINT_BENCH_SAMPLES=200 \
  SYMBOLICA_FLINT_BENCH_CSV=1 \
  GCD_BENCH_CASE=dense \
  GCD_BENCH_NVARS=1 \
  GCD_BENCH_DEGREE=64 \
  GCD_BENCH_GAP=10 \
  GCD_BENCH_COEFFICIENT_BITS=30 \
  /tmp/univariate-prime-r2-b4db253-screen
```

The command above runs the frozen `b4db253` control. Replace only the final path to run a candidate.
The paired filter is a case-sensitive substring. The presence of
`SYMBOLICA_FLINT_BENCH_PAIRED`, even with value `0`, selects paired mode. Use an even sample count so
the alternating Symbolica-first/FLINT-first order is balanced. Historical tables in this document
used 201 samples; new decision runs should use 200 or 202.

The shared target currently contains more than one hashed executable. The known
`flint_comparison-0df1dfb2f578e077` entry currently matches the frozen `b7727d9` hybrid-inverse
binary byte for byte and embeds `/tmp/symbolica-zp64-r2` plus version
`symbolica-v2.2.0-131-gb7727d9`. A later build may overwrite it. Ordinary `cargo build` does not
print an unambiguous executable path. For a new build, add
`--message-format=json-render-diagnostics`, save the JSON stream, and extract the non-null
`executable` from the `compiler-artifact` whose target name is `flint_comparison`. Immediately copy
that executable and its matching `.d` sidecar to uniquely named `/tmp` artifacts.

For every frozen binary record:

- `git rev-parse HEAD`, branch name, and whether the source was dirty;
- a saved patch plus its SHA-256 if the source was dirty;
- the executable SHA-256 and matching `.d` sidecar;
- raw per-process CSV/stdout, host state, CPU pin, and sample count.

Do not use the executable's printed `Symbolica workspace 2.2.0` label as commit provenance. Build
script output was reused in the shared target, and the dirty-state part consults the embedded source
worktree path at runtime. Filenames, checksums, recorded Git state, and the dependency sidecar are
the authoritative provenance.

If a non-paired Divan run is needed, its filters are different from the paired row strings. The
exact paths are:

- `flint_comparison::polynomial_gcd::{symbolica_auto,flint_auto}`;
- `flint_comparison::generated_factor_products::{symbolica,flint}::dense 1-variable degrees 32/31`;
- `flint_comparison::generated_factorization::{symbolica,flint}::dense 1-variable degrees 32/31`.

Paired mode bypasses Divan CLI filtering, so do not mix the two filter forms.

Before timing, make sure another Rust job is not loading the host:

```sh
ps -eo user,pid,pcpu,stat,args | \
  rg 'rustc|cargo (build|test|bench)|rustred-.*test-threads' | \
  rg -v 'rg rustc|symbolica-univariate'
```

Use sequential processes on an otherwise idle pinned core. Do not run multiple performance jobs at
once, even on different cores. For a decision-quality result, use at least five fresh processes per
binary and compare the median normalized ratio. A first full LTO link can take about eleven minutes.
The preliminary `b4db253` and fused-Q1 screens below were pinned to core 8, but an unrelated Rust
job was active; repeat the best `b4db253` comparison on an idle host before integration.

The shared debug target for validation is `/tmp/symbolica-univariate-test-target`. Disk space was
exhausted once by duplicate Cargo targets. If cleanup is needed, remove only generated `target/`
directories after resolving their explicit paths; do not remove worktree source directories or the
frozen benchmark binaries.

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
| dense 1 variable, degree 64 on current `dev` | `2.225` | current primary loss; accepted-base chain is `1.191`, active inverse candidate first screened at `1.148` |

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

### Result summary and attribution

Historical rows below use five independent 201-sample paired processes pinned to core 8. The later
idle `b4db253` and `8440390` decision rows use five 200-sample processes, while `b7727d9` currently
has only the one explicitly shown 200-sample process. The ratio is `Symbolica / FLINT`; lower is
better. Each binary was frozen immediately after its source-matched release build.

| Source | Degree-64 ratios | Median | Main change from previous row |
|---|---|---:|---|
| current scalar `a3f721c` | representative current result | `2.225` | integrated starting point |
| modular `810875a` | representative current result | `1.816` | modular reconstruction replaces the high-degree scalar heuristic |
| certificate only `bfed508` | `1.441165`, `1.425816`, `1.427506`, `1.427256`, `1.430441` | `1.428` | reusable pure-multiprecision exact certificate |
| field plus certificate `f46fcff` | `1.330403`, `1.324185`, `1.321667`, `1.326932`, `1.333425` | `1.327` | direct dense `Zp64` image GCD composes with the certificate |
| plus R^2 conversion `7fdc099` | `1.281967`, `1.288584`, `1.282771`, `1.283670`, `1.274269` | `1.283` | removes division-based conversion to Montgomery form |
| plus cached primes `b4db253`, earlier run | `1.189030`, `1.191018`, `1.189864`, `1.191163`, `1.188682` | `1.190` | avoids repeated dynamic next-prime search |
| cached-prime control, later idle run | `1.193639`, `1.190578`, `1.195843`, `1.188940`, `1.191412` | `1.191412` | confirms the earlier result |
| direct small-limb conversion `8440390` | `1.178260`, `1.179542`, `1.179371`, `1.194325`, `1.178938` | `1.179371` | only about 1% normalized gain; reject under the 3% threshold |
| private hybrid inverse `b7727d9` | first process `1.148490` | **provisional** | promising; four more candidate processes plus fresh controls required |
| fused adjacent-degree remainder `0f7fa97` | `1.487387`, `1.488532`, `1.471457`, `1.481061`, `1.483546` | `1.484` | decisive regression; reject |

For scale, the `b4db253` screen put Symbolica near `0.428 ms`; the fused experiment was
near `0.535 ms`. The latter is too large a regression to be explained by host noise. The first
hybrid-inverse process measured Symbolica minimum/median `0.399472/0.409370 ms` and FLINT
minimum/median `0.345555/0.356442 ms`, producing ratio `1.148490`.
The largest isolated gain was the checked dense certificate. The first modular switch was next;
the direct dense field image, R^2 conversion, and cached-prime changes then compounded to approach
FLINT. Do not attribute this progress to allocation caching alone.

The current frozen binaries are:

| Source | Frozen binary | SHA-256 |
|---|---|---|
| `bfed508` | `/tmp/univariate-borrowed-cert-bfed508-screen` | `c6567397403bba49ca9e357f8822fd98a3c12fe3441472c95f931f21f72c1af9` |
| `f46fcff` | `/tmp/univariate-combined-f46fcff-screen` | `76fc25fae540aac71516d5b5957cd54d5dcc9a039194c74a5c7cd43cca4f29a2` |
| `7fdc099` | `/tmp/univariate-zp64-r2-7fdc099-screen` | `9af5a41ddb8288323fcc1585619ff2d3ed8ccd705630ec1c625b09c1a847429f` |
| `b4db253` | `/tmp/univariate-prime-r2-b4db253-screen` | `c94b9c4d6a083d438cee123e1dc60254782307434c59a5ef0cd0079915b9a7e5` |
| rejected `8440390` | `/tmp/univariate-zp64-limb-8440390-screen` | `59f5b16f6ec2362b85c0594c42d246fbf9e68ecb19c2418f27128ba67dbf79dd` |
| active `b7727d9` | `/tmp/univariate-zp64-inverse-b7727d9-screen` | `83852a6af0bd1c2ce90373e03cfbc966c451ca051e5e29b28884e647d92976a5` |
| rejected `0f7fa97` | `/tmp/univariate-fused-q1-0f7fa97-screen` | `db3f21e6f09dfbadb6e14a65232e292aef7ce50f64847d3bb1e993795f5ce2c2` |

The matching dependency-sidecar SHA-256 values are
`b4f62b3124c3379eab1c74862ab7a61abdc15eaba79513a93baf0ffe8cce04d2` for `8440390` and
`c967ebd25431b91eee8f4f5cb8bd55abeff0c0a955d3415f8806bd5145a3793e` for `b7727d9`.

The earlier scalar, modular, and field-image checksums remain documented below.

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
`1.716`. The later `f46fcff` combined result demonstrates that this field-image change composes
with the much larger certificate improvement.

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

These profiles were captured on an Intel Xeon W-2135, Linux `6.18.37`, with perf `7.1.2`,
`cycles:u` at 999 Hz, LBR call graphs, and core 8. `/tmp` is ephemeral and the perf files do not
have a committed checksum manifest. No raw CSV/stdout survives for the later five-process
`bfed508`, `f46fcff`, `7fdc099`, `b4db253`, or fused-Q1 screens, nor for the quoted one-variable
product/factor ratios. Their frozen binaries and recorded process ratios remain, but all future
measurements should preserve the raw output and provenance described above.

## Historical experiments that did not justify integration

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

## Implemented and active degree-64 candidates

### Checked dense certificate (`bfed508` and `f46fcff`)

The reverse-engineered certificate has now been implemented, tested, committed, and benchmarked.
`bfed508` adds:

- `MultiPrecisionInteger::div_rem_owned_ref_assign`, which consumes the numerator as quotient
  storage, borrows the divisor, writes into reusable remainder storage, calls GMP `mpz_tdiv_qr`
  with the quotient aliasing the numerator, and retains a `no_gmp` fallback;
- a private `DenseUnivariateIntegerDivisionContext` in `src/poly/gcd.rs` that converts the divisor
  once, performs checked dense long division in pure multiprecision storage, reuses the remainder,
  uses fused subtraction/multiplication, and falls back to generic `try_div` for unsupported
  layouts;
- direct exact/inexact, sparse, inactive-variable, large-coefficient, and backend differential
  tests.

`bfed508` is based on `810875a`; its five-process median is `1.428`. `f46fcff` is the same patch on
top of `ea09e01`; the combined median is `1.327`. This reproduces the old unexplained `1.325`
binary and establishes that the gain came from certification, not from field-call aliasing.

The certificate is already fairly close to FLINT: both checks take about `177 us` versus roughly
`164 us` in FLINT. Creating its dense `Vec<MultiPrecisionInteger>` buffers costs only about
`0.17 us`; reusing vector capacities alone would save less than `0.5 us`. Retaining GMP limb
capacity in every workspace slot may save `3-5 us`, but the larger remaining cost is the 4,160
scalar GMP multiply-subtract updates, roughly `129 us`. A thresholded divide-and-conquer exact
polynomial division could have more headroom, but it is lower priority than finishing the inverse
decision and the one-variable product/factorization work.

### R^2 Montgomery conversion (`7fdc099`)

Both `Zp64` constructors now precompute `one = R mod p` and `r2 = one^2 mod p`, where
`R = 2^64`. Converting an ordinary residue uses `REDC(a * r2)` instead of a division-based
conversion. Focused conversion tests and the finite-field module tests pass with both GMP and
`no_gmp`. The five-process degree-64 median improved from `1.327` to `1.283`, about a 3.3%
normalized gain.

### Cached univariate GCD primes (`b4db253`)

The univariate modular path now tries 32 consecutive verified full-word primes, starting at
`18_346_744_073_709_552_031` and ending at `18_346_744_073_709_553_459`, before continuing with
the original dynamic iterator. This is deliberately a narrow
`univariate_modular_gcd_prime_iterator`; the shared Hu/Zippel iterator is unchanged. A test checks
all 32 entries and the first fallback against `PrimeIteratorU64`.

The original five-process degree-64 median was `1.190`. A later five-process idle control measured
`1.193639`, `1.190578`, `1.195843`, `1.188940`, and `1.191412`, with median `1.191412`, confirming
the result.

### Rejected direct-limb conversion (`8440390`)

The clean branch `codex/zp64-limb-conversion` in `/tmp/symbolica-zp64-limb-conversion` implements a
private direct conversion of small GMP-backed integers to full-word `Zp64`. The specialization:

- is enabled only for GMP on 64-bit non-Windows targets with 64-bit GMP limbs and no nails;
- handles positive and negative `Integer::Single`, `Integer::Double`, and borrowed
  `Integer::Large` magnitudes of at most four limbs;
- handles only full-word `Zp64` moduli above `i64::MAX`, leaving every other case on the old path;
- folds limbs from high to low while maintaining the Montgomery image and negates once at the end;
- has differential tests for signs, boundary values, `p-1`, `p`, `p+1`, forced one- through
  four-limb `Large` values, a five-limb fallback, and modulus boundaries.

The complete GMP integer and finite-field test groups passed (`22/22` and `12/12`), as did the
corresponding `no_gmp` groups (`16/16` and `12/12`). Formatting and diff checks passed. All 225
stored coefficients in the degree-64 fixture take the specialized conversion: 22 `Single`, 57
`Double`, one two-limb, 42 three-limb, and 103 four-limb values.

Five paired processes measured `1.178260`, `1.179542`, `1.179371`, `1.194325`, and `1.178938`,
median `1.179371`, versus the control median `1.191412`. The median of the per-process Symbolica
medians changed from `0.428919 ms` to `0.424228 ms`, only `4.691 us` or 1.09%. The median of the
per-process minima changed from about `0.4139 ms` to `0.4132 ms`. The normalized ratio improvement
is likewise only about 1%.
Long LBR profiles show the conversion share falling from about 2.48% to 1.28%, roughly halving the
phase, but that phase is too small for a material end-to-end win. Profile artifacts are:

- `/tmp/profile-d64-b4db253-lbr.perf.data`, SHA-256
  `675588de6a63d59ffc84538f5404209bd37e23dee4da6b6cd0880eacd884df35`, and its flat report,
  SHA-256 `4f5f36fc97a2c211d1c21ef15aa9e5469808c733b5ea4f3f93bf4641b2b539b`;
- `/tmp/profile-d64-limb-8440390-lbr.perf.data`, SHA-256
  `4c958e3cfb433d4622bd662e3d6883faff1a5a950e922e2614822056045dc2ae`, and its flat report,
  SHA-256 `b249fb410198218fa2b87626797d51d9117c74fd395c6cb1c35256d3c44e57a9`.

Reject this commit under the 3% acceptance threshold and leave it off `dev`. It works as intended;
the low result is an Amdahl-limit issue, not a missed selector or inlining failure. A more ambitious
reciprocal/modulo context could have headroom, but is not the next experiment.

### Active private dense inverse (`b7727d9`)

The clean branch `codex/zp64-hybrid-inverse` in `/tmp/symbolica-zp64-r2` is based on `b4db253`.
It adds `DenseZp64UnivariateGcdImage::inverse_leading`, used only by the private dense univariate
image context. It preserves the existing Montgomery de-scaling, starts from the Euclidean state
after the known initial zero-quotient step, handles quotients 1, 2, and 3 with subtraction chains,
and uses hardware division for larger quotients. It makes no `Ring` or `Field` API change.

The differential inverse test covers all 32 cached primes, boundary values, and 512 deterministic
nonzero values per prime. That focused test and all `poly::gcd::tests` (`28/28`) pass with the
license configured; formatting and diff checks pass. The default full GCD group was tested, but
the relevant `no_gmp` validation has not yet been run.

The source-matched release build took about 9 minutes with full LTO. Its first 200-sample paired
degree-64 process measured Symbolica minimum/median `0.399472/0.409370 ms`, FLINT
`0.345555/0.356442 ms`, and ratio `1.148490`. Relative to the control median ratio `1.191412`, this
is about a 3.6% normalized improvement and therefore clears the acceptance threshold provisionally.
One process is not sufficient for integration. Run four more candidate processes, preferably
alternating with fresh control processes, and preserve every raw output. No raw CSV/stdout, build
JSON stream, or hybrid perf artifact was saved for the first process; only the frozen executable,
sidecar, recorded timing, and source provenance survive.

The profile that motivated this change had the following flat shares:

| Phase | `b4db253` | `8440390` | FLINT comparison |
|---|---:|---:|---:|
| Symbolica dense-image `run` self | 11.16% | 11.26% | FLINT Q1 kernel 7.46%/7.61% |
| Symbolica `FiniteField<u64>::inv` | 10.65% | 11.01% | FLINT `n_gcdinv` 9.20%/8.98% |

Disassembly showed Symbolica's generic inverse always reaching hardware division, while FLINT's
inverse handles common small quotients with subtraction chains. Profile `b7727d9` for 20,000 or
more samples with LBR call graphs and verify that inverse-exclusive cycles and divider samples
fall while the dense remainder loop remains unchanged. If the timing reproduces, run degree guards
32, 48, 64, and 80, then the full relevant GMP/`no_gmp` tests before integration.

## Rejected follow-up experiment

### Fused adjacent-degree remainder (`e0430d5`/`0f7fa97`)

FLINT's profile exposed `_nmod_poly_divrem_q1_preinv1_fullword`, so a private Symbolica path fused
two adjacent-degree quotient steps into one divisor traversal and preserved the 129th product-sum
bit with an explicit three-limb reduction. Focused differential tests and the full GCD tests pass,
but the degree-64 ratio regressed from about `1.190` on `b4db253` to `1.484` on `0f7fa97`.

Reject this implementation. Its Rust three-limb reduction/code generation is apparently more
expensive than the two existing Montgomery reductions. Preserve the branch as evidence; do not
integrate or retry the same implementation without a new lower-level code-generation argument.
The frozen binary and checksum exist, but no separate CSV/stdout or test log was preserved; the
exact ratios above come from the recorded session measurements.

## Validation ledger for the candidate chain

These checks have already been completed; rerun them after rebasing/integration, not merely to
rediscover whether the isolated branches were sound:

| Candidate | Checks already passed |
|---|---|
| backend owned-numerator division helper | focused GMP test and focused `no_gmp` test |
| `bfed508` certificate | direct certificate differential test; all `poly::gcd::tests` (`24/24`) |
| `f46fcff` combined | all GCD tests (`26/26`) |
| `7fdc099` R^2 | focused conversion in GMP and `no_gmp`; finite-field module tests (`12/12`) in both configurations |
| `b4db253` cached primes | static/dynamic iterator comparison; all GCD tests (`27/27`) |
| fused-Q1 branch | focused tests and all GCD tests (`29/29`, or `30/30` with cached primes); performance rejected it |
| `8440390` direct limb conversion | GMP integer (`22/22`) and finite-field (`12/12`) groups; `no_gmp` integer (`16/16`) and finite-field (`12/12`) groups; performance rejected it |
| `b7727d9` private inverse | focused 32-prime differential test; all default GCD tests (`28/28`); `no_gmp` and neighboring-degree guards still required |

## Other rejected algorithms

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

### 1. Finish the private-inverse decision

Work in `/tmp/symbolica-zp64-r2` at clean commit `b7727d9`; do not recreate the patch. Confirm that
no unrelated Rust build is running, then run four more candidate processes and at least two fresh
controls, sequentially, pinned to core 8. Alternate candidate and control to reduce drift. For each
process use the exact dense degree-64 environment below and change only the executable and output
filename:

```sh
taskset -c 8 env \
  SYMBOLICA_FLINT_BENCH_PAIRED=1 \
  SYMBOLICA_FLINT_BENCH_FILTER='GCD auto: dense 1 variables degree 64' \
  SYMBOLICA_FLINT_BENCH_SAMPLES=200 \
  SYMBOLICA_FLINT_BENCH_CSV=1 \
  GCD_BENCH_CASE=dense \
  GCD_BENCH_NVARS=1 \
  GCD_BENCH_DEGREE=64 \
  GCD_BENCH_GAP=10 \
  GCD_BENCH_COEFFICIENT_BITS=30 \
  /tmp/univariate-zp64-inverse-b7727d9-screen \
  > /tmp/d64-inverse-run-02.csv
```

The control executable is `/tmp/univariate-prime-r2-b4db253-screen`. Set `SYMBOLICA_LICENSE` in
the calling environment; do not add it to the command saved in the repository. Compute the median
of the five candidate process ratios and compare it with the control-process median. Accept only a
repeatable improvement of at least 3%, without a worse absolute Symbolica median hidden by FLINT
noise.

If the five-process candidate median clears the threshold, capture a long profile:

```sh
/nix/store/lc3a14lcfrbii8f008kjgl45xibjm1w8-perf-linux-7.1.3/bin/perf record \
  -e cycles:u -F 999 --call-graph lbr \
  -o /tmp/profile-d64-inverse-b7727d9-lbr.perf.data -- \
  taskset -c 8 env \
    SYMBOLICA_FLINT_BENCH_PAIRED=1 \
    SYMBOLICA_FLINT_BENCH_FILTER='GCD auto: dense 1 variables degree 64' \
    SYMBOLICA_FLINT_BENCH_SAMPLES=20000 \
    SYMBOLICA_FLINT_BENCH_CSV=1 \
    GCD_BENCH_CASE=dense GCD_BENCH_NVARS=1 GCD_BENCH_DEGREE=64 \
    GCD_BENCH_GAP=10 GCD_BENCH_COEFFICIENT_BITS=30 \
    /tmp/univariate-zp64-inverse-b7727d9-screen \
    > /tmp/profile-d64-inverse-b7727d9-lbr.stdout
```

Compare it with `/tmp/profile-d64-b4db253-lbr.perf.data`. The expected discriminator is lower
inverse-exclusive time and fewer sampled divider instructions; the dense `run` remainder loop
should not otherwise change. If the candidate does not reproduce, preserve branch and artifacts
but leave it off `dev`; move to the product rather than reviving either rejected Q1 or limb patch.

### 2. Validate and integrate the accepted degree-64 chain

Always integrate the accepted base chain through `b4db253` after the inverse verdict; it remains
isolated even if `b7727d9` fails. If `b7727d9` passes the timing decision, first run the full GCD
tests and relevant `no_gmp` build/tests from its worktree, plus `cargo fmt --check` and
`git diff --check`, and append it to the sequence:

```sh
CARGO_TARGET_DIR=/tmp/symbolica-univariate-test-target \
  cargo test poly::gcd::tests -- --test-threads=1
CARGO_TARGET_DIR=/tmp/symbolica-univariate-test-target \
  cargo test --no-default-features --features no_gmp poly::gcd::tests -- --test-threads=1
cargo fmt --check
git diff --check
```

The tests require `SYMBOLICA_LICENSE` in the calling environment. Do not put its value in the
repository or committed logs. The integration sequence is:

```text
810875a -> ea09e01 -> f46fcff -> 7fdc099 -> b4db253 [-> b7727d9 if accepted]
```

The first five commits were already checked for textual conflicts against `dev` at `386174d`.
Resolve any later changes carefully, rerun validation from `dev`, build a source-matched release
binary, and preserve it before using the shared target again. Do not cherry-pick `8440390`,
`0f7fa97`, `bfed508` in addition to `f46fcff`, or `f3b13b4` in addition to `b4db253`.

### 3. Guard the chosen GCD chain

Run degrees 32, 48, 64, and 80 with five processes per binary. Degree 32 must remain on the fast
scalar path. Use the same dense-case environment and vary only `GCD_BENCH_DEGREE`. Also run:

- the full `poly::gcd::tests` group in the default and relevant `no_gmp` configurations;
- zero, constant, coprime, content, unlucky-prime, inactive-variable, sparse/high-gap, and large
  coefficient cases;
- the generated GCD matrix;
- PolyBench rows #11, #105, #140, and #178, with special attention to the previously investigated
  #105 and #140 losses.

Record process-level ratios, not just a single Divan summary. If the complete chain regresses an
important neighboring regime, narrow its selector rather than making it the unconditional path.

### 4. Improve the one-variable product

Use the `generated factor product: dense 1-variable degrees 32/31` paired filter. Confirm the
current approximately `3.06` ratio across processes, then determine which multiplication kernel is
selected. No frozen source-matched current-`dev` product/factor baseline has been identified, so
build and preserve that control before changing the selector. Record:

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

The smallest candidate fix is in
`lib/numerica/src/domains/integer/polynomial_kernels.rs`, in
`DenseIntegerMul::try_kronecker`. The current rejection is the
`product_count < 64 || packed_output_len.saturating_mul(128) >= product_count` gate. Allow
Kronecker substitution for sufficiently large contiguous supports while preserving the stricter
gate for sparse supports. Use the existing `DenseIntegerMul::try_kronecker_for_test` hook and
Kronecker tests in `lib/numerica/src/domains/integer.rs`; add a targeted 32-by-32 contiguous-support
test matching the factor fixture before relaxing the selector. The fixture table is
`benches/support/cases.rs::GENERATED_FACTOR_CASES`, and paired construction is in
`benches/flint_comparison.rs::paired_generated_factorization`.

Benchmark nearby degrees, coefficient heights, and sparse/high-gap inputs to guard the selector.
This needs neither a global helper nor a new `Ring` method.

After copying the source-matched build to `/tmp/flint-comparison-candidate`, one complete paired
process is:

```sh
taskset -c 8 env \
  -u GCD_BENCH_CASE \
  -u GCD_BENCH_NVARS \
  -u GCD_BENCH_DEGREE \
  -u GCD_BENCH_GAP \
  -u GCD_BENCH_COEFFICIENT_BITS \
  SYMBOLICA_FLINT_BENCH_PAIRED=1 \
  SYMBOLICA_FLINT_BENCH_FILTER='generated factor product: dense 1-variable degrees 32/31' \
  SYMBOLICA_FLINT_BENCH_SAMPLES=200 \
  SYMBOLICA_FLINT_BENCH_CSV=1 \
  /tmp/flint-comparison-candidate > /tmp/factor-product-run-01.csv
```

Repeat with unique output files. Clearing all `GCD_BENCH_*` variables matters because paired setup
constructs the configurable GCD case before reaching the factor rows; a stale invalid value can
abort an otherwise factor-only run.

This multiplication fix is the next primary task after the degree-64 GCD chain, even if a smaller
residual GCD gap remains.

### 5. Profile one-variable factorization

After the product is improved, rerun `generated factorization: dense 1-variable degrees 32/31`.
Profile the remaining factor-only time. The old median was host-bimodal, so compare process medians
and minima and inspect algorithm selection. Do not count input construction, product formation, or
factor expansion in the timed region.

Before changing multiplication, preserve one source-matched integrated-control binary and use it
for both the product and factor rows. The Symbolica factor timing entry is
`benches/support/symbolica.rs::benchmark_factorization`, which calls `input.factor()`. Integer
polynomial factor dispatch begins in `src/poly/factor.rs` at the `Factorize` implementation for
`MultivariatePolynomial<IntegerRing>`.

Potential areas to distinguish with evidence are square-free decomposition, modular factorization,
Hensel/reconstruction, repeated exact division, and coefficient conversion. Do not assume the GCD
fix alone explains the factor gap.

One complete paired factor process is:

```sh
taskset -c 8 env \
  -u GCD_BENCH_CASE \
  -u GCD_BENCH_NVARS \
  -u GCD_BENCH_DEGREE \
  -u GCD_BENCH_GAP \
  -u GCD_BENCH_COEFFICIENT_BITS \
  SYMBOLICA_FLINT_BENCH_PAIRED=1 \
  SYMBOLICA_FLINT_BENCH_FILTER='generated factorization: dense 1-variable degrees 32/31' \
  SYMBOLICA_FLINT_BENCH_SAMPLES=20 \
  SYMBOLICA_FLINT_BENCH_CSV=1 \
  /tmp/flint-comparison-candidate > /tmp/factorization-run-01.csv
```

### 6. Resume the broader matrix

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
