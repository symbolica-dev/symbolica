# Polynomial performance continuation handoff

This is the live continuation record for the single-core Symbolica/FLINT polynomial-performance
work. It was refreshed on 2026-08-31 after the `v26` bounded automatic bivariate Hu-Monagan GCD
round. The
current 125-row fixed-fixture non-resultant inventory has 124 Symbolica wins and one loss with a
`0.636796` median. Its worst row is generated dense bivariate GCD at `1.022443`. Dense
degree-48/64/80 GCD measure
`0.904854/0.711750/0.993433`. Dense one-variable degree-63/64/65 factorization measures
`0.910657/0.700836/0.945536`, and the stable matching products measure
`0.814256/0.815238/0.841130`. The mixed current 3,200-problem PolyBench distribution has a
`0.699823` paired median with 2,441 Symbolica wins. Rows are measured across accepted source
snapshots, with each replacement and its source-matched evidence documented below.
Keep this file current whenever an experiment is accepted, rejected, or left partly complete. The
purpose is that another agent can resume without reconstructing decisions from chat history or
transient binary names.

The complete current Symbolica/FLINT scoreboard is in
[CURRENT_STATUS.md](CURRENT_STATUS.md), with the latest immutable snapshot in
[CURRENT_STATUS_v26.md](CURRENT_STATUS_v26.md). Its primary statistics contain 125 non-resultant
comparisons: 124 favor Symbolica and one favors FLINT. The six Ducos, six Brown, and six CRT
measurements are retained only in a compact appendix note and do not affect primary ranks or
summary statistics. After every accepted optimization, update the live file and create the next
numbered snapshot with an opening paragraph that describes the changes from its predecessor.
Historical alternative-backend validation labels below are normalized to the current
`integer-malachite,float-astro` feature names; their test counts remain the original checkpoint
results unless a post-rebase rerun is stated explicitly.

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

## Accepted dense-state quadratic Hensel lifting

The quadratic univariate Hensel path formerly converted the lifted factors and Bezout cofactors
between sparse multivariate polynomials and dense coefficient vectors every round. It also formed
the exact residual through generic polynomial multiplication and addition even though the active
polynomials are univariate in one known coordinate.

`DenseIntegerModularUnivariateContext` now keeps the target, both factors, both Bezout cofactors,
and the exact product residual as `Vec<Integer>` values for the complete quadratic lift. Its exact
residual helper computes `target-left*right` directly; its scaled-update helper uses
`Z.add_mul_assign` for `old + scale*delta`. The two prepared modular divisors share one leading
coefficient inverse only when their canonical leading residues are exactly equal. The lifted
factors are converted back to multivariate form once, at the exit. Linear Hensel lifting, partial
final precision, nonunit retries, the `p=2` path, and exact/inexact reconstruction exits retain
their previous algorithms.

There is no new dispatch heuristic. The existing quadratic-lift decision still uses the prime,
p-adic precision schedule, and existing permission checks. The change specializes the internal
representation only after that algebraic path has already been selected.

Six alternating 500-sample processes against the exact preceding source give:

| Source | Symbolica median | FLINT median | Median process S/F |
|---|---:|---:|---:|
| matched control | `1.561047 ms` | `1.379257 ms` | `1.130004` |
| dense-state lift | `1.369634 ms` | `1.380469 ms` | `0.993222` |

Symbolica time falls `12.26%`. A final six-process, 500-sample refresh from the complete v17
artifact gives `1.360815 ms` versus `1.376199 ms`, or `0.987690` S/F. The corresponding product
row remains a Symbolica win at `0.573810`, so both construction and complete factorization are now
faster than FLINT.

The first accepted binary is `/tmp/flint-comparison-quadratic-dense-state-v1`, SHA-256
`0887ae8ddbb7407c77df1761477e3d0ae7aea0584be15d2487318e457793dbf6`; its dependency file SHA is
`bca17cd465a45cdadad6fdfc93944f842eda47f993116ef25df6a8f1cf805867`. Its build JSON is
`/tmp/flint-comparison-quadratic-dense-state-v1-build.jsonl`, SHA-256
`2ba4ab7d750b264f23921b3dac9c872d516e98f49826bc16277bf98aee413ce1`. Raw A/B files are
`/tmp/qhensel-dense-state-d33-{candidate,control}-0{1..6}.csv`; final refresh files are
`/tmp/final-factor-d33-0{1..6}.csv`.

Focused exact-residual, signed-correction, binary-prime, nontrivial-gamma, partial-final-precision,
and high-height route tests pass. The complete factor module passes 96 deterministic tests; its
pre-existing randomized `galois_upgrade` test failed once and passed immediately in isolation.

## Accepted coefficient-content operation context

`factor_separable` used to call `PolynomialGCD::gcd_multiple` over every coefficient layer for
every candidate variable. On PolyBench #105 this content preprocessing accounted for about
`7--9 ms`: the exact-source profile placed `factor_separable` at 21.45% of paired cycles,
`gcd_multiple` at 20.95%, and the nested GCD at 19.32%, while the later factor reconstruction was
not the bottleneck.

`SeparableCoefficientContentContext` now records whether the current polynomial has a trivial
global common monomial: its minimum exponent is zero in every coordinate. Its
`nonconstant_content` operation applies these exact certificates and reductions:

- Any constant coefficient layer proves that the complete coefficient content is constant.
- If the global common monomial is trivial, any one-term layer proves constant content. Every
  divisor of a monomial is a monomial, and a nonconstant common monomial would contradict one of
  the zero coordinatewise minima.
- With more than two layers, the two sparsest layers are probed only when the second support is at
  most one eighth of the support accumulated by the unchanged first `gcd_multiple` stage. The
  common first-layer support cancels from both product proxies, so the comparison neither
  overflows nor depends on a fixture scale.
- A constant exact pair GCD returns immediately. A monomial exact pair GCD uses the same global
  monomial proof. Otherwise the exact pair GCD replaces its two inputs only when its support is no
  larger than the supports it summarizes, and the original `gcd_multiple` remains the fallback.

The one-eighth admission is a decisive work margin for amortizing an extra exact GCD. Selection
uses only exponent minima and support-work bounds; it contains no benchmark identifier, exact
variable count, degree fingerprint, or coefficient values. All speculative results are exact,
and every inconclusive case retains the previous algorithm.

Six alternating 100-sample processes on PolyBench #105 give:

| Source | Symbolica median | FLINT median | Median process S/F |
|---|---:|---:|---:|
| dense-Hensel control | `31.383993 ms` | `29.225538 ms` | `1.073561` |
| content context | `16.604782 ms` | `28.816486 ms` | `0.576212` |

Symbolica time falls `47.09%` against the exact preceding source and is now 1.74x faster than
FLINT. Raw files are `/tmp/separable-monomial-pb105-{candidate,control}-0{1..6}.csv`.

The complete 11-case PolyBench factor screen validates every result. Its final operation ratios
are `0.986146` (#176), `0.652104` (#92), `0.586252` (5v #159), `0.576212` (#105), `0.421076`
(#84), `0.256988` (#131), `0.224514` (#178), `0.164792` (#32), `0.161334` (#163), `0.094995`
(8v #159), and `0.063263` (#44). Generated one-variable cases remain within 0.8% of their exact
source control; generated two- and three-variable factorization refresh to `0.401383` and
`0.554623` S/F. Product-only guards do not call the new context and remain within the noise floor.

The content-context binary is `/tmp/flint-comparison-separable-monomial-v1`, SHA-256
`a4524b87bbbda791cd6c471da225c45de6b80d37d678a29e2978db995914ee7b`; its dependency file SHA is
`bca17cd465a45cdadad6fdfc93944f842eda47f993116ef25df6a8f1cf805867`, and its build JSON SHA is
`5f28c3470463c788f3b62aa7cc171fa34a93b58eb59d7e3f0faf47ade9ad6a16`. The exact-source diff SHA
at build time is `d9888409d0d03a4d42cdfe1c1f370dde88744ccd4f122d8c486d3901b047451d`.

Focused tests cover the one-term certificate, the exact one-eighth boundary and rejection below
it, a genuine global common monomial that must use the complete fallback, and an exact pair-GCD
replacement that preserves nonconstant content. The full library suite passes 531 of 532 tests;
the sole failure is the known unrelated root-isolation test accepting a different valid dyadic
endpoint. Integer, rational, finite-field, and algebraic factor tests all pass.

## Added asymmetric dominant-GCD benchmarks

`GcdCaseConfig` now records the two cofactor degrees and common-factor degree independently.
Existing balanced generated cases retain their expressions and display names. Four new
eight-variable cases use an 8-term degree-one cofactor, a 45-term degree-two cofactor, and dense
common factors of 165, 495, or 1287 terms; the fourth repeats the 165-term geometry with 256-bit
linear weights. Each paired row validates Symbolica's GCD against the known factor and verifies
FLINT returns the same factor before timing. Product construction is reported separately.

Six sequential 100-sample processes pinned to core 8 give:

| Regime | Product S/F | GCD S/F |
|---|---:|---:|
| dense, common degree 3 / 165 terms | `1.327831` | `0.445428` |
| dense, common degree 4 / 495 terms | `1.295552` | `0.460421` |
| dense, common degree 5 / 1287 terms | `0.746378` | `0.467658` |
| 256-bit weights, common degree 3 / 165 terms | `1.174073` | `1.631601` |

The small-height cases show that Symbolica's automatic GCD route handles a large common factor
and very small cofactor well. The coefficient-height change reverses that result and creates the
new overall worst row, so it is the next worst-first profiling target rather than a reason for a
fixture-specific selector.

The exact final binary is `/tmp/flint-comparison-separable-monomial-asymmetric-gcd-v2`, SHA-256
`c836647c2ed4d6e47c4d598962610d7bcf9fef07e7dd38c08397723184ff74ab`; its dependency file SHA is
`bca17cd465a45cdadad6fdfc93944f842eda47f993116ef25df6a8f1cf805867`. Its build JSON is
`/tmp/flint-comparison-separable-monomial-asymmetric-gcd-v2-build.jsonl`, SHA-256
`4b6c8146f9bfe216895b3733dc896c43585cb19454c5d4e3922d6cf472c7e3be`. The full workspace diff
at build time has SHA-256 `36de494a7fee88934c19d4ccd0701883d38a521f01f14caa0f8124b7600a4ddd`.
Raw files are `/tmp/asymmetric-large-gcd-0{1..6}.csv`; their aggregate summary is
`/tmp/asymmetric-large-gcd-summary.csv`, SHA-256
`9f82b16e213a060fb22c6e477e0c604f1ae55afefe46ad7658d3ab8cbecb717d`.

## Accepted coefficient-height-aware Hu prime sizing

The high-height asymmetric GCD required many more CRT images than its interpolation geometry
alone predicted. For coefficient bounds of at least 32 significant bits, Hu's initial prime size
now also targets completion in at most eight images: the height target is
`ceil(significant_bits / 8).min(63)`, and the selected bit count is the maximum of that value and
the existing interpolation lower bound. Smaller coefficients retain the preceding selection.
This is a height/work bound rather than a fixture property; the exact modular algorithm,
reconstruction certificate, and fallback remain unchanged.

Six alternating 100-sample processes on the 256-bit asymmetric eight-variable GCD give:

| Source | Symbolica median | FLINT median | Median process S/F |
|---|---:|---:|---:|
| matched control | `15.4896580 ms` | `9.4926955 ms` | `1.6317470` |
| height-aware prime | `8.9752725 ms` | `9.5195510 ms` | `0.9428320` |

Symbolica time falls 42.06%, and all four asymmetric GCD operation rows now favor Symbolica. Broad
GCD and product guards stayed within 2%; the only larger relative movements were sub-0.1-ms rows
dominated by timer noise. The accepted binary is
`/tmp/flint-comparison-hu-prime-height-v1`, SHA-256
`45ec7f480762b81d3b8870bc7a3637143289821b8433d7c4a27b28fd102ea12f`. The operation-guard summary
is `/tmp/hu-prime-height-operation-guard-summary.txt`, SHA-256
`459c434bb4db69ae168f2b25270ad9773cda2e783012c62e0e56b9fe56a40547`; the product-guard summary
is `/tmp/hu-prime-height-product-guard-summary.txt`, SHA-256
`f742635d293bd76e036fde1049e5b455c6d377032ccdc150d3bcb3f3029798bf`.

## PolyBench 0.4.3 full distribution and bounded retry repair

The complete upstream `0001`--`0008` matrix was run for five and eight variables with PolyBench
0.4.3 (`f3a25498883a80462c6278a87c9dfc93630d8a06`). Every setup used seed 42, 10 warmups, 200
measured problems, 37--50 requested terms, coefficients in `-16384..=16384`, and a 21,600-second
per-solver timeout. Uniform coordinate degrees were 22--30 and sharp coordinate degrees were
0--30. Runs were sequential, pinned to core 8, and used `RAYON_NUM_THREADS=1`. The Symbolica
adapter used a plain release/default-feature build, including `faster_alloc`; FLINT was the exact
upstream vcpkg-built 3.5.0 adapter. The temporary adapter manifest contained only local workspace
path patches and its lockfile was regenerated for the current dependency graph.

The initial sweep found two deterministic Symbolica stack overflows: five-variable `0004`
problem 11 and five-variable `0008` problem 25. Four failed bivariate reconstruction exits could
tail-recurse into the same route after advancing the deterministic sample bound, and the analogous
univariate retry was recursive as well. Both paths now retry iteratively. When automatic selection
initially chooses the bivariate route, it permits three completed reconstruction failures, then
restores the saved route-local univariate variable order and original coefficient cursor before a
one-way univariate fallback. Explicit bivariate and univariate requests retain their existing
terminal behavior. The bivariate failure edges covered are leading-coefficient precomputation,
modular Hensel lifting, sparse coefficient lifting, and final exact-product mismatch.

The repaired problems complete in 93.845 ms and 79.881 ms respectively, instead of aborting after
roughly four and a half minutes. The stack-overflow profile is
`/tmp/profile-polybench-05-0004-problem11-stack-overflow.perf.data`. The deeper likely issue is
repeated post-modular reconstruction failure: sparse coefficient lifting returns no reconstruction
or the final exact product certificate fails. A future repair should type those failure reasons,
then choose between a general Diophantine correction and continued p-adic precision from exact
evidence rather than retrying a fixed number of samples.

The final 16 setups and all 3,200 measured problems completed without wrong or inconsistent
answers. The median paired S/F is `0.962614`, the geometric mean is `0.906571`, the summed-time
ratio is `0.517296`, and Symbolica wins 1,832 problems. The five-variable paired median is
`1.047351`; the eight-variable paired median is `0.722134`. Eight of the 16 setup medians favor
Symbolica. Curated CSVs, run logs, checksums, summary images, and every
`FLINT_vs_Symbolica.png` are in
[`benchmark-results/polybench-0.4.3-current`](benchmark-results/polybench-0.4.3-current/README.md).
The final Symbolica adapter has SHA-256
`ba97e01eee517f2ec43b73f0fbca116959dcd1fc0f3bdf3f16ad95fb2e52834e`.

All final problem streams match the initial seed-42 streams byte for byte. The two failure-stream
SHA-256 values are `a3da7fb4f0e89f0a879f8072714d541b642a879c689ccccfb4b1972ff9d11ede`
for five-variable `0004` and `8fe91053cc23ff27762313843a2f5034637d84ec3ef9e0c76393a0717f0c23ee`
for five-variable `0008`. Focused retry-state and order-restoration tests pass; the final factor
module run passes all 98 tests. Its randomized `galois_upgrade` test had failed once in an earlier
run and passed immediately in isolation as well as in the final module run.

## Accepted four-way direct `DenseZp64` accumulation

The near-`2^64` dense-large fixture was already using `DenseZp64Mul::multiply_direct`: each output
coefficient is an exact little-endian three-limb accumulator, and Montgomery reduction happens
only after the convolution. Its hot row loop nevertheless completed one 64-by-64-bit product and
its dependent add/carry chain before issuing the next multiplication. This serialized independent
work even though neighboring products normally update different output cells.

`add_u64_product_row_unrolled4` now reads four right coefficients and output offsets, issues their
four `u128` products, and then applies four independent three-limb add/carry chains. A scalar tail
handles the final zero to three terms. The surrounding kernel still proves that every exact output
coefficient fits its three limbs, so the unroll changes instruction scheduling but not arithmetic,
layout, or reduction frequency.

There is no new dispatch heuristic. The helper is called unconditionally only inside the blocked
direct branch that was already selected when `product_count >= 1_000_000`. Kronecker substitution
is tried before direct multiplication and returns before this loop when applicable. Requests below
the existing million-product threshold retain the scalar direct loop; sparse multiplication,
other dense dispatch, and every other coefficient domain are unchanged. The boundary is therefore
an existing measured-work class rather than a benchmark name, modulus fingerprint, or
variable-count special case.

Six alternating 500-sample full-LTO processes, pinned sequentially to core 8, give:

| Source | Symbolica median | FLINT median | Median process S/F |
|---|---:|---:|---:|
| matched control | `10.8102935 ms` | `9.3227025 ms` | `1.1601025` |
| four-way candidate | `9.2343605 ms` | `9.308868 ms` | `0.992282` |

Symbolica time falls `14.578078%`. The scoreboard replaces its preceding robust row
`1.137177 -> 0.992282`, making Symbolica slightly faster than FLINT in this regime. The primary
inventory therefore moves from 98/18 to 99/17 Symbolica wins/losses; its median remains
`0.638180`.

The final LBR profile measures `9.276851 ms` versus `9.385683 ms`, or `0.988404` S/F. Symbolica's
`DenseZp64Mul::multiply_direct` accounts for 48.02% of paired cycles; FLINT's
`_nmod_mpoly_addmul_array1_ulong3` accounts for 47.39%, and FLINT's array append for 2.16%.
Annotated assembly shows four `mulq` instructions issued before four separate `add/adc/adc`
chains, confirming that the compiler preserved the intended independent work rather than folding
the source back into a scalar dependency chain.

A 14-case finite-field multiplication screen changes only the intended near-`2^64` dense-large
row materially. The candidate and control CSVs are
`/tmp/zp64-unroll-finite-all-candidate-20.csv`, SHA-256
`1511024ddf67f9190bae56265498bae2a28d72c4b144fd4981ecf253f8243cc3`, and
`/tmp/zp64-unroll-finite-all-control-20.csv`, SHA-256
`a0461ed279a115f1d515601ccf9cdc0cffd470c206ca8852cc1bac7abbc735d0`. The guards cover small and
large primes, dense univariate, dense multivariate, sparse, total-degree, very-large, and
accumulator-bound cases; all remain stable.

The accepted binary is `/tmp/flint-comparison-zp64-unroll4-v1`, SHA-256
`d5d759c6422255d67626809305b8466a388148812bad362ede1ce697d09475b1`; its dependency file is
`/tmp/flint-comparison-zp64-unroll4-v1.d`, SHA-256
`bca17cd465a45cdadad6fdfc93944f842eda47f993116ef25df6a8f1cf805867`. The build JSON is
`/tmp/flint-comparison-zp64-unroll4-build.jsonl`, SHA-256
`9ee3bc1e05d9e209d42ead61907f4009e1e9cde2cc4430dc430a9a7aae1f1bc3`. Raw alternating files are
`/tmp/near64-unroll-{candidate,control}-0{1..6}.csv`.

The profile is `/tmp/profile-near64-unroll4-v1-lbr.perf.data`, SHA-256
`e41373a84604ce1ca205f3da4407f23e2125abad637d2281ea69f2ba20b6d5b5`. Its timing CSV SHA-256 is
`bb25ffe27a2f78f4099ab9633e101750c1f30b35e03ef4b854745c46bf4f0011`, its report SHA-256 is
`779a491ade6fc2d72ff4feaba705589ec552a7c9faeedaf5d36da7b491cc2360`, and its annotated-assembly
SHA-256 is `acf3f0a252169e78489cd99d212831848d7114bf2064b2939bf5580303cc2307`.

Validation passes all 125 Numerica library tests, all 22 Numerica doctests, and all 40 focused root
polynomial tests. The focused Malachite-backend finite-field multiplication test also passes.

## Accepted batched shifted-Vandermonde inversion

`solve_shifted_transposed_vandermonde` reconstructs coefficients from samples
`rhs[k] = sum_i c_i x_i^(k+1)`. Its denominator for coefficient `i` is

`d_i = x_i * product_{j != i}(x_i - x_j)`.

The previous implementation divided by the ordinary Vandermonde norm and by `x_i` separately for
every coefficient. For a finite field, the new implementation forms all `d_i`, builds prefix
products, inverts their total product once, and walks backward with suffix products to recover
every reciprocal. This replaces `2n` field inversions with one inversion and linear-many
multiplications. Infinite rings retain one exact combined division per coefficient so the change
does not introduce coefficient-growth work there. Duplicate points still fail explicitly, and a
zero point remains singular through the same failed inversion semantics.

The selector is the algebraic property `ring.size().is_some()`: bounded finite rings benefit from
trading inversions for multiplication, while infinite domains retain direct division. It does not
inspect polynomial shape, a benchmark label, a modulus fingerprint, or a fixed variable count.
The focused test reconstructs empty systems and lengths 1 through 12, including zero target
coefficients. An independent algebra audit covered finite extensions, unit denominators in
composite modular rings, duplicate points, and zero points.

Six alternating 500-sample full-LTO processes for PolyBench five-variable uniform GCD #11 give:

| Source | Symbolica median | Median process S/F |
|---|---:|---:|
| pre-batch control | `10.708840 ms` | `1.086416` |
| batched inversion | `10.285017 ms` | `1.043121` |

Symbolica time falls `3.957%`. The accepted batch binary is
`/tmp/flint-comparison-vandermonde-batch-v1`, SHA-256
`f3cd42f1a8ca22d02e66a41161d4831cbccea8633a34c646de8bd5202c59fa07`; its dependency file
SHA-256 is `bca17cd465a45cdadad6fdfc93944f842eda47f993116ef25df6a8f1cf805867`. The build JSON is
`/tmp/flint-comparison-vandermonde-batch-build.jsonl`, SHA-256
`426da0ecd05d292a2762f7a3022041b7fc2be0d234910d8abc8e88ce24fe4e35`. Raw files are
`/tmp/pb11-vbatch-{candidate,control}-0{1..6}.csv`.

## Accepted reusable last-variable evaluation context

Recursive Zippel interpolation repeatedly substitutes new values into the same last active
variable of both inputs. The former `replace_last` path rediscovered every lexicographic
coefficient row, allocated fresh output vectors, and recomputed identical powers independently for
the two inputs on every image.

`LastVariableEvaluationContext` now records lexicographic row ranges and maximum degree once,
retains its output allocation, and evaluates each row into that buffer. The pair-level
`RepeatedLastVariableEvaluationContext` shares a generation-stamped power table across both inputs
for each sample value. A zero power is cached correctly because validity is represented by the
generation rather than by the coefficient value. The table is bounded at 100,000 entries; larger
sparse exponents use direct exponentiation, preserving bounded memory. The leading-coefficient
polynomial `gamma`, which is univariate in the sampled variable, is evaluated with Horner's method,
and the accepted sample value is reused for normalization.

Reuse is selected only when the interpolation degree bound plus the leading-coefficient degree
predicts more than one image. One-image calls retain direct `replace_last`; if two failed attempts
relax the degree bound, the same work rule can enable reuse for the subsequent images. These are
predicted repeated work, support order, and exponent bounds. There is no benchmark identifier,
exact variable-count branch, coefficient fingerprint, or high-height exception.

Six alternating 500-sample processes against the batch-only binary reduce #11 by a further
`5.98%`; the first context binary gives a median `0.981326` S/F. The exact final source was then
built again and compared in six final processes:

| Final row | Symbolica median | FLINT median | Median process S/F |
|---|---:|---:|---:|
| PolyBench 5v uniform nontrivial GCD #11 | `9.641149 ms` | `9.861558 ms` | `0.978321` |

The exact final binary is `/tmp/flint-comparison-lastvar-context-final`, SHA-256
`6d351bcb00c16d6e893e3c53a0047446db8e86e98f8d1bfee18390b5d8c4486b`; its dependency file
SHA-256 is `bca17cd465a45cdadad6fdfc93944f842eda47f993116ef25df6a8f1cf805867`. The build JSON is
`/tmp/flint-comparison-lastvar-context-final-build.jsonl`, SHA-256
`cab13020ec52ba47496510822ea136b41990772b9e30df4bd7de29d5ed52463e`. Its `.text` section is
byte-identical to the profiled first context binary, SHA-256
`728f3223ee04dba45b8c8003451de0f0aeca9cd5c1cac8ada96e30200fec374b`. Final raw files are
`/tmp/pb11-lastvar-context-final-{candidate,control}-0{1..6}.csv`.

The exact-source 20-sample PolyBench screen is
`/tmp/lastvar-final-polybench-gcd-20.csv`: all 12 cases validate and #11 measures `0.981196` in
that short screen. The corresponding 14-regime generated screen is
`/tmp/lastvar-final-generated-gcd-10.csv`; it spans one to eight variables, dense, sparse,
high-gap, and 128--1024-bit high-height inputs. Focused validation passes the repeated-evaluation
test, all 36 GCD tests, zero and repeated substitution values, cancellation, a degree above the
power-cache cap, and a substituted variable with a physically trailing absent variable.

The one-image high-height eight-variable guard does not activate cached evaluation. Six final
50-sample processes measure `39.090454 ms` and `0.450165` S/F, versus `38.177243 ms` and
`0.439926` for the batch-only binary. Profiles attribute only 0.15% of the candidate to context
evaluation versus 0.18% to the old `replace_last`. The hot first `u64`
`construct_new_image_single_scale` copy remains exactly `0x48b9` bytes, but its start moves from
address modulo 64 `0x20` to `0x30`; its paired share moves from 11.47% to 12.26%. This is a
whole-program-LTO placement effect rather than extra selected work, so a fixture-specific
high-height exclusion was rejected. Profiles are
`/tmp/profile-highheight8-lastvar-{candidate,control}.perf.data`; final guard files are
`/tmp/highheight8-lastvar-final-{candidate,control}-0{1..6}.csv`.

Final validation passes all 125 Numerica library tests, all 22 Numerica doctests, all 36 focused
GCD tests, and the repeated last-variable evaluation test. The full root library suite passes
527 of 528 tests. The unrelated `poly::univariate::roots::tests::isolate` test expects the exact
dyadic interval `[15/64, 9/32]` for one simple root, while this checkout deterministically returns
the also-valid isolating interval `[3/16, 9/32]`; none of this round's changed files touch root
isolation.

## Accepted pre-content Hu-Monagan main-variable planning

The generic GCD path formerly selected its initial main coordinate and immediately removed
univariate content in that coordinate. Hu-Monagan then inherited this choice even when the
anchored input had a much smaller coefficient row in another active variable. On anisotropic
supports this can multiply the geometric sampling schedule: #140's original main coordinate has
a maximum row of 1,441 terms, while the selected coordinate has a maximum row of 119 terms.

The accepted implementation gives `PolynomialGCD` a narrow `gcd_with_precontent_plan` hook. The
generic caller supplies the active variables, degree bounds, and per-input degrees already found
by its metadata scan. Only the integer-domain implementation overrides the hook. A successful
hook must return the complete GCD in the original coordinate order, including polynomial content;
the generic caller then restores only the shared monomial shift and common exponent scale. A
declined or unsuccessful plan returns `None` and resumes the unchanged content-removal and GCD
path. This keeps the capability on the GCD interface and adds no method to the base `Ring` trait.

`HuMonaganPlanningContext` fixes the smaller input as the interpolation anchor for the entire
operation, using the left input on a term-count tie. Eligibility and execution use that same
anchor. Planning runs only when the current order already satisfies Hu's sparsity test and has a
nonconstant main variable plus at least two nonconstant interpolation variables. It then accepts
an alternative main variable only when all of these input-derived conditions hold:

- Its largest anchored-input coefficient row is at most one quarter of the current row. The 4x
  threshold removes at least two complete levels from Hu's doubling schedule.
- Its degree-weighted image estimate,
  `(anchored_degree + other_degree + 2) * maximum_row_support`, does not increase.
- Removing the candidate coordinate from the exponent encoding gives a strictly smaller
  mixed-radix Kronecker range, and twice that range fits below the largest available smooth-prime
  modulus.
- The candidate has a nonzero GCD degree bound, remains fully Hu-applicable in the permuted
  coordinates, and retains the fourfold row advantage after content extraction.

Before building a row histogram, the planner uses the exact pigeonhole lower bound
`ceil(term_count / (degree + 1))` to reject a candidate that cannot meet the support threshold.
The bounded counter stops at the first oversized row. It stores low exponent indices densely and
migrates exactly to a hash map when a gap would make the dense span exceed the polynomial's term
count, so its storage remains O(term count). Hu's box and cofactor sparsity arithmetic uses exact
checked `u128` operations in the common case and falls back to `Integer` only on overflow.

For an accepted coordinate, `PreparedHuMonaganGcd` computes both univariate contents in that
coordinate, takes their GCD, makes both inputs primitive, permutes the variables and bounds, and
runs Hu with the preapproved fixed anchor. It inverse-permutes the result, multiplies the content
back, and normalizes it. Every preparation or reconstruction failure falls through to the legacy
route. Selection depends on degrees, supports, interpolation work, exponent range, and modulus
feasibility; it contains no benchmark identifier, exact variable-count branch, or fixture
coefficient fingerprint.

A deterministic 600-support sweep covers three-, five-, and eight-variable uniform, clustered,
and anisotropic families. Of 597 inputs eligible for the current Hu path, the selector changes 198
plans, improves all 198 predicted schedules, leaves 399 unchanged, and regresses none. Total
predicted samples fall from 7,968 to 2,672 (`-66.5%`). The selected three-, five-, and
eight-variable subsets change `2336 -> 712` over 64 cases, `2464 -> 936` over 67 cases, and
`3168 -> 1024` over 67 cases. No uniform or clustered input switches, and all 198 selected plans
also have a strictly smaller Kronecker range.

Six alternating 100-sample full-LTO processes, pinned sequentially to core 8, give the following
PolyBench GCD operation results. `Change` compares Symbolica time in the final candidate with the
frozen `v12` binary; negative values are improvements.

| PolyBench GCD row | Candidate S/F | Frozen S/F | Change |
|---|---:|---:|---:|
| 5v uniform #11 | `1.121593` | `1.075969` | `+4.151%` |
| 5v sharp #11 | `0.525423` | `0.518498` | `+1.368%` |
| 8v uniform #11 | `0.679912` | `0.673357` | `+0.970%` |
| 8v uniform #55 | `0.839509` | `0.832288` | `+0.841%` |
| 8v uniform #56 | `0.451333` | `0.448195` | `+0.753%` |
| 8v uniform #168 | `0.546446` | `0.542210` | `+0.819%` |
| 8v uniform #188 | `0.462498` | `0.460475` | `+0.493%` |
| 8v sharp #11 | `0.409963` | `0.747412` | `-45.145%` |
| 8v sharp #53 | `0.395224` | `0.392195` | `+0.818%` |
| 8v sharp #35 | `0.647601` | `0.643469` | `+0.627%` |
| 8v sharp #140 | `0.479027` | `1.263872` | `-62.563%` |
| 8v uniform trivial #11 | `0.957213` | `0.957296` | `-0.059%` |

#140 changes its observed Hu schedule from 80 images to 16; sharp #11 has the same 80-to-16
change. The scoreboard replaces the older robust rows `1.268240 -> 0.479027` and
`0.748107 -> 0.409963`. It also records the final frozen-binary five-variable uniform result
`1.097676 -> 1.121593` rather than hiding the cross-binary regression.

Three alternating 20-sample process pairs give these generated GCD guards:

| Generated GCD row | Candidate S/F | Frozen S/F | Change |
|---|---:|---:|---:|
| dense 1v degree 32 | `0.997344` | `0.988546` | `+1.096%` |
| dense 2v degree 5 | `1.063973` | `1.054802` | `+1.041%` |
| dense 3v degree 7 | `0.197415` | `0.197698` | `+0.442%` |
| dense 5v degree 7 | `0.820838` | `0.770883` | `+6.325%` |
| dense 8v degree 5 | `0.688010` | `0.604876` | `+13.517%` |
| sparse 5v degree 7 | `0.637127` | `0.629221` | `+1.219%` |
| sparse 8v degree 5 | `0.361034` | `0.351518` | `+2.693%` |
| high-gap 5v degree 5, gap 64 | `0.074374` | `0.072728` | `+2.364%` |
| high-gap 8v degree 4, gap 256 | `0.029005` | `0.028213` | `+2.719%` |
| high-height 5v degree 4, 128 bits | `0.428465` | `0.434061` | `-1.292%` |
| high-height 5v degree 4, 256 bits | `0.433884` | `0.439503` | `-1.372%` |
| high-height 5v degree 4, 512 bits | `0.437428` | `0.442556` | `-1.244%` |
| high-height 5v degree 4, 1024 bits | `0.471647` | `0.475337` | `-0.570%` |
| high-height 8v degree 3, 256 bits | `0.465732` | `0.475451` | `-2.286%` |

The scoreboard conservatively replaces dense 5v `0.750982 -> 0.820838` and dense 8v
`0.591952 -> 0.688010`; guard shifts below 3% do not replace stronger existing rows. These two
larger regressions, and the five-variable uniform result above, are not planner-allocation cost.
Runtime plan-on/plan-off measurements in the same trace-enabled binary change Symbolica by only
`+0.20%`, `+1.15%`, and `+0.15%`, respectively. Across separate full-LTO binaries, an LBR profile
instead moves existing Zippel `construct_new_image_single_scale` work from 14.20% to 16.02% of
combined samples, with no planning or allocation hotspot large enough to explain the difference.
The current evidence points to whole-program LTO code placement/code generation in the existing
Zippel image constructor. The frozen-binary regressions remain reported because they are real,
even though the accepted selector does not execute on those dense/uniform guards.

The final binary is `/tmp/flint-comparison-hu-precontent-final-v15`, SHA-256
`9150daeb152ece2480994383bf0aaf60f9e2f4cb34aeff4fb7a8ab321e21426c`; its build JSON is
`/tmp/flint-comparison-hu-precontent-final-v15-build.jsonl`, SHA-256
`ea44e9a7e96b01bde6d0a6af855a5f9d2aa51daba1f26bc9c159c51ba5495711`. Raw PolyBench files are
`/tmp/hu-strict-v15-polybench-01..06.csv` with controls
`/tmp/hu-strict-base-polybench-01..06.csv`; their summary is
`/tmp/hu-strict-v15-polybench-summary.csv`. Generated files use
`/tmp/hu-strict-v15-generated-01..03.csv` and `/tmp/hu-strict-base-generated-01..03.csv`, summarized
in `/tmp/hu-strict-v15-generated-summary.csv`.

The sweep output is `/tmp/hu_fixed_anchor_guarded_sweep_raw.txt`, SHA-256
`6704dfe0ddec66e19d5b7d7e16393a7a081f2048ec36eb70bcd01528107d8cbd`; its generator is
`/tmp/hu_fixed_anchor_sweep.rs`, SHA-256
`f4c1b5ff1c6b4bd68529c6200fb527c0f05a0d63009e267f2fae415c479f6ff4`. The trace binary is
`/tmp/flint-comparison-hu-precontent-trace-v3`, SHA-256
`f9e4fe69fa89d9661709cbb81c98704ef7516a00bd4923fe4f0474032793c0b0`. The candidate and control
five-variable uniform profiles are `/tmp/profile-hu-prefilter-5v-uniform.perf.data`, SHA-256
`e65a32ba92b9a4cf958b9749665e27c6b473d8cdf77c32c4868cb31e34e11500`, and
`/tmp/profile-hu-v12-5v-uniform.perf.data`, SHA-256
`cc9510ba16ea46f63cb5aeca99aebbbec12d097238016c8db4570f2335c749ff`.

`cargo check --lib` is warning-free. Focused validation passes all three `hu_planning` tests, the
pre-content minimum-geometry test, the adaptive counter migration test, and all four
`hu_monagan` tests. Coverage includes the exact 4x boundary and its rejection below the boundary,
image-work and Kronecker-range adversaries, left/right/tied anchors, content restoration, complete
Hu applicability, and end-to-end public `gcd` calls in both input orders with monomial shifts and
common exponent compression. A default root run excluding the `isolate` name filter passes 520
tests; rerunning the pre-existing univariate-root isolation failure alone reproduces its unrelated
interval mismatch (`[3/16, 9/32]` versus `[15/64, 9/32]`). Numerica passes 124/124 tests. With
`integer-malachite,float-astro,native_code_generation`, all 35 polynomial-GCD tests pass.

## Accepted balanced two-leaf Hensel reconstruction

The synchronized univariate Hensel product tree normally lifts every modular leaf to the global
coefficient bound before recombination. A product of a small subset of leaves can sometimes be
certified as an exact integer factor much earlier, and that factor and its complement can have
substantially smaller local coefficient bounds. The accepted path tests one deterministic
two-leaf partition near the end of an otherwise unchanged lift and shortens only the terminal
precision schedule when the certificate and work bound both succeed.

`UnivariateHenselProductTreeTopology::most_balanced_leaf_pair` selects the leaf pair whose degree
sum is closest to half the total degree; original leaf indices resolve ties. The route is considered
only when the product-tree root has two internal children and this pair gives a strictly smaller
degree imbalance than that existing root split. It never loops over every degree-two combination
looking for one that happens to reconstruct. These topology rules are independent of the chosen
prime, polynomial degree, coefficient values, and benchmark identity.

At the penultimate global-precision stage, after the existing whole-root early reconstruction has
failed, the implementation multiplies the selected lifted leaves using the current dense modular
context. It centers and primitive-normalizes that product, then divides the original integer
polynomial exactly by it. Zero content, a constant or full-degree candidate, a degree mismatch, or
a nonzero remainder rejects the partition and preserves the original schedule. Exact division is
the correctness certificate; a modular coincidence cannot change the returned factorization.

For a certified pair, `UnivariateHenselExactPartition` records the exact factor and quotient, the
modular leaves belonging to each side, and each side's own coefficient bound. The required
prime-power exponent is the maximum of those two local reconstruction requirements. The terminal
schedule is shortened only when

`2 * (required_exponent - current_exponent) <= global_exponent - current_exponent`.

Thus the two local continuations consume no more than half the remaining global precision work.
The implementation retains the already lifted leaf and Bezout buffers, truncates the existing
schedule, and appends only the locally required terminal exponent. At that exponent it recombines
the two certified sides independently using their local bounds. No earlier Hensel stage, product-
tree construction, prime selector, or general recombination path is replaced. If the root shape,
balance test, exact certificate, local target, or work guard fails, lifting continues to the
original global target.

For the degree-65 fixture, prime 13 gives modular leaf degrees `[1, 1, 10, 10, 10, 16, 16]`.
The two degree-16 leaves form the unique closest-to-half pair. Their product divides exactly at
`13^43`; the certified local bounds require `13^51`, while the old complete tree continued to
`13^86`. This is an observed instance of the generic degree-balance, local-bound, and exact-
division rules, not a special case in the selector.

The motivating pre-change profile is `/tmp/profile-d65-audit-current-lbr.perf.data`, SHA-256
`9ffe6abbe0f391f01be87e5836d718638dbe3d2e49bb928227c093427bcf99cc`. Synchronized Hensel
lifting accounted for 34.70% of paired cycles versus 24.61% for FLINT's factor tree, so shortening
certified lift precision directly targets the measured gap.

Six alternating 500-sample processes, pinned sequentially to core 8, give:

| Source | Symbolica median | FLINT median | Median process S/F |
|---|---:|---:|---:|
| frozen pre-change control | `3.531218 ms` | `3.025285 ms` | `1.167612` |
| balanced-pair candidate | `3.343246 ms` | `3.020529 ms` | `1.106499` |

Symbolica time falls 5.32% in the source-matched comparison. The scoreboard replaces its stronger
preceding degree-65 row `1.171321 -> 1.106499`; the case remains slower than FLINT, so the primary
win/loss count and `0.638180` median do not change.

Three alternating 100-sample generated-factor guard processes show neighboring Symbolica shifts
of `+1.10%` at degree 63, `+0.29%` for dense 2v, `-0.54%` for dense 3v, `+0.26%` for high-height
degree 33, and `+1.75%` at degree 64. The same short guard measures the intended degree-65 row at
`-4.52%`; the six-process result above is authoritative. In the broad 20-sample PolyBench sweep,
the ten rows other than five-variable #159 remain within about 2%. The initially noisy #159 guard
was repeated in two 100-sample processes: candidate and control Symbolica means are
`65.624250 ms` and `65.022330 ms`, respectively, a `+0.93%` shift. Factor-input product timings
remain neutral: this reconstruction runs after the input product, and their absolute differences
are at the sub-microsecond scale.

The final candidate is `/tmp/flint-comparison-d65-balanced-v1`, SHA-256
`cb90434fa2132af0d655391ab8bd6a9c2a59c73acb851d3d85b2847b1e3e9d1a`; its dependency file is
`/tmp/flint-comparison-d65-balanced-v1.d`, SHA-256
`bca17cd465a45cdadad6fdfc93944f842eda47f993116ef25df6a8f1cf805867`, and its build JSON is
`/tmp/flint-comparison-d65-balanced-v1-build.jsonl`, SHA-256
`4b4bd863ab2acd170f1aa224c51ae97961aeef20f992b797a217b6aac6443dcb`. Raw source-matched files
are `/tmp/d65-balanced-v1-{candidate,control}-01..06.csv`. Generated, PolyBench, long #159, and
product guards are `/tmp/d65-balanced-v1-generated-factor-{candidate,control}-01..03.csv`,
`/tmp/d65-balanced-v1-polybench-factor-{candidate,control}-01..03.csv`,
`/tmp/d65-balanced-v1-pb159-{candidate,control}-01..02.csv`, and
`/tmp/d65-balanced-v1-factor-product-{candidate,control}-01..03.csv`.

All 93 default-feature `poly::factor::test` tests pass single-threaded. Focused coverage proves
deterministic closest-to-half selection including ties and an unrelated leaf-degree family, strict
root-balance improvement, the exact work-guard boundary, and fallback after a spurious modular
split fails exact division. All 93 factor tests also pass with
`--no-default-features --features integer-malachite,float-astro,native_code_generation`.

## Accepted constant-coordinate modular GCD reconstruction

Commit `e58e876` lets `UnivariateModularGcdContext` normalize its modular images by the gcd of
the primitive inputs' constant coefficients when that coordinate predicts strictly fewer 64-bit
images than the leading coordinate. Exact division into both inputs remains the correctness
certificate. The selector retains leading normalization when either constant coefficient gcd is
zero, when the estimates tie, or when leading normalization already predicts one image.

For a true primitive gcd `G`, the two projective representatives are
`H_d = (gamma_d / lc(G)) G` and `H_0 = (gamma_0 / G(0)) G`, where `gamma_d` and `gamma_0` are the
gcds of the corresponding primitive input coefficients. The configured degree-64 fixture has a
roughly 225-bit `H_d` and a roughly 126-bit `H_0 = 2G`; the selected reconstruction therefore
finishes after two modular images instead of four. Constant normalization computes the ordinary
monic modular GCD image, inverts its nonzero constant coefficient, and scales that image to
`gamma_0` before the existing CRT merge.

The coordinate-gcd estimate is not a coefficient bound because the unknown `lc(G)` and `G(0)`
can reverse the apparent ordering. The implementation therefore keeps the old leading threshold
on an independent geometric schedule. When that schedule is due, it derives a temporary leading
representative from the current constant-normalized CRT polynomial `R_0` using
`gamma_d / lc(R_0) mod M`, then applies the same exact-division certificate. This needs neither a
second modular GCD nor a second CRT accumulator. Both schedules reset after an unlucky lower-degree
image. The adversarial test uses `G = 2^384*x + 1`, cofactors with constant `2^192`, and proves that
the selected constant representative can still be incomplete when the derived leading
representative is exact.

Six alternating source-matched 1,000-sample processes, pinned to one core, give:

| Degree | Frozen-control Symbolica | Candidate Symbolica | Candidate FLINT | Control S/F | Candidate S/F | Symbolica change |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | `0.082731 ms` | `0.082140 ms` | `0.083212 ms` | `0.999046` | `0.986571` | `-0.71%` |
| 48 | `0.252478 ms` | `0.210098 ms` | `0.235995 ms` | `1.067784` | `0.889777` | `-16.79%` |
| 64 | `0.409735 ms` | `0.300099 ms` | `0.351792 ms` | `1.161142` | `0.853224` | `-26.76%` |
| 80 | `0.726356 ms` | `0.632327 ms` | `0.587998 ms` | `1.231297` | `1.075107` | `-12.95%` |

The scoreboard compares the final ratios to its preceding `v9` rows: degree 48 changes
`1.072266 -> 0.889777`, degree 64 `1.181340 -> 0.853224`, and degree 80
`1.234835 -> 1.075107`. Degree 32 was a guard rather than an existing configurable scoreboard
row. Symbolica is now faster than FLINT at degrees 48 and 64; degree 80 remains 7.5% slower.

The final binary is `/tmp/flint-comparison-d64-constant-pivot-final`, SHA-256
`3ae577b8aa9480eb7c82009a70e01bd1e0cfe063cefecda77d55057ba3569b6f`. Its build JSON is
`/tmp/flint-comparison-d64-constant-pivot-final-build.jsonl`, SHA-256
`42518a445c61bbb4ab43b0493145e2e6669ca537178c1835e0166f05359fc924`. The frozen control is
`/tmp/flint-comparison-pb131-integer-montgomery-final`, SHA-256
`11c0e4d3d7866e11c7c76a56e6050b097a16ad84f9257f3e25329a05e21b4428`. Raw GCD runs are
`/tmp/dense-gcd-constant-pivot-final-d{32,48,64,80}-{candidate,control}-{01..06}.csv`.

The four 500-sample factorization guards are materially neutral: degree 33 is `1.112958`, degree
63 `1.030347`, degree 64 `1.128065`, and degree 65 `1.161896` S/F. Their raw files are
`/tmp/constant-pivot-final-factor-{d33hh,d63,d64,d65}-{candidate,control}-{01..06}.csv`. The
10,000-sample degree-64 product guard is `1.099408` versus the frozen control's `1.088238`; the
1.54% absolute Symbolica shift is noise for code that does not enter multiplication, so the
stronger existing `1.094628` inventory row remains current. Product raw files are
`/tmp/constant-pivot-final-product-d64-{candidate,control}-{01..06}.csv`.

The pre-change LBR profile is `/tmp/profile-d64-current-gcd-lbr.perf.data`, SHA-256
`a0db8cda609ae6ace27920a2caa2f3faedecd3a966d60609528c5f6114a57ebb`. It attributed 53.43% of
combined samples to Symbolica's GCD and 46.18% to FLINT's; Symbolica's dense `Zp64` modular-GCD
work accounted for 22.33% versus 17.16% in FLINT's `nmod` Euclidean work, while exact division
was nearly tied at 21.61% versus 20.90%. Reducing the modular-image count attacks the measured
dominant difference rather than the already comparable certificate.

An earlier signed-centered fusion of adjacent quotient updates was rejected. Six alternating
degree-64 processes changed Symbolica from `0.409653 ms` to `0.494380 ms`, or 20.68% slower,
despite retiring 1.5% fewer instructions. Across 10,000 paired iterations, branch misses rose from
`286,667,275` to `416,241,973` (+45.2%) and cycles rose 10.4%. The unpredictable signed product
comparison outweighed the saved Montgomery work. Raw runs are
`/tmp/dense-gcd-signed-pair-v1-d64-{candidate,control}-{01..06}.csv`; counters are in
`/tmp/d64-signed-pair-{candidate,control}.perfstat`. No part of that experiment remains in source.

All 30 `poly::gcd::tests` and all 89 `poly::factor::test` unit tests pass single-threaded. Focused
coverage includes constant selection, zero-constant leading selection, unlucky first-degree reset,
inactive-variable content restoration, and the adversarial lazy leading fallback.

## Accepted bounded dense Montgomery multiplication for #131

Commit `f14d182` gives `FiniteField<Integer>` a bounded dense-index polynomial multiplication
kernel. `DenseIntegerMontgomeryMul` accumulates Montgomery-form coefficient products exactly in
`Integer` output cells with fused multiply-adds, then performs one Montgomery reduction per
nonzero output coefficient. If `C * (p - 1) <= R - 1`, where `C` bounds collisions in one output
cell, then `C * (p - 1)^2 < pR` and the exact accumulator is a valid direct Montgomery-reduction
input. Otherwise the kernel first takes one remainder modulo `p`. A request with an output span
above 4,096 cells and above four times the coefficient-pair count declines the kernel and retains
the existing polynomial-dispatch fallback.

Six alternating source-matched 300-sample processes give:

| Source | Symbolica median | FLINT median | Median process S/F |
|---|---:|---:|---:|
| frozen `0632274` control | `16.032689 ms` | `46.365158 ms` | `0.345860` |
| dense-Montgomery candidate | `12.842260 ms` | `46.353227 ms` | `0.276987` |

Symbolica time falls 19.90% in the alternating experiment. Six independent final 300-sample
processes measure `12.827604 ms` versus `46.280535 ms`, with median process ratio `0.277201`;
against the frozen control's Symbolica median this is a 19.99% reduction. Symbolica is now 3.61x
faster than FLINT on #131. Relative to the original `1.745716` result, S/F has fallen 84.12%.

The performance-equivalent final binary is
`/tmp/flint-comparison-pb131-integer-montgomery-final`, SHA-256
`11c0e4d3d7866e11c7c76a56e6050b097a16ad84f9257f3e25329a05e21b4428`; its build JSON SHA-256 is
`4d182a8e5d5606463a38042b6db853a8affde662702effbfd76af22159561616`. The only production-code
difference between this binary and committed source is checked request validation; the
multiplication loop is identical, and the remaining changes are tests and documentation. Raw
final runs are `/tmp/pb131-integer-montgomery-final-{1..6}.csv`; alternating runs are
`/tmp/pb131-integer-montgomery-{1..6}-{control,candidate}.csv`.

The final LBR profile is `/tmp/profile-pb131-integer-montgomery-final-lbr.perf.data`, SHA-256
`81301f21149540652917c60180791719bdab996d55c22c6b6fd846ddc8a843c1`; its 300-sample paired
factorization timing is `14.655552 ms` versus `53.288973 ms`, or `0.275020` S/F. The capture also
contains the much smaller #131 product row at `0.052940 ms` versus `0.085145 ms`. Relative to the
preceding profile, the Bernardin subtree falls from 11.30% to 6.15% of combined cycles, roughly
`7.1 -> 4.2 ms`, and the large-modulus polynomial-multiplication subtree falls from 7.08% to the
new kernel's 2.06%, roughly `4.5 -> 1.4 ms`. The profile therefore localizes the gain to the
intended bivariate modular-product work rather than a factorization-route change.

Tests cover direct reduction with modulus `2^64 + 13`, one-remainder reduction with modulus
`2^127 - 1`, modular cancellation to zero in both modes, and rejection of a 10,001-cell sparse
span. All 122 default-feature Numerica library tests, all 89 default-feature factor tests, the
focused no-GMP Numerica test, and all 89 no-GMP factor tests pass. Formatting and diff checks pass.

Three sequential guard sweeps are in
`/tmp/pb131-integer-montgomery-polybench-factor-{1..3}.csv` and
`/tmp/pb131-integer-montgomery-generated-factor-{1..3}.csv`. Their median process ratios are:

| Guard family | Current S/F values |
|---|---|
| PolyBench 5v | #32 `0.180831`; #159 `0.600275`; #163 `0.216464`; #131 `0.276987` |
| PolyBench 8v | #178 `0.221585`; #92 `0.833277`; #84 `0.685930`; #176 `1.002582`; #44 `0.096734`; #105 `1.085467`; #159 `0.118053` |
| Generated | 1v d63 `1.026392`; 2v `0.400541`; 3v `0.549458`; high-height d33 `1.085990`; d64 `1.126503`; d65 `1.163216` |

No guard shows a material regression. Several bivariate rows improve, but these three-process
50/100-sample sweeps remain regression evidence rather than replacing stronger dedicated
scoreboard measurements in the focused `v9` snapshot.

## Accepted dense degree-64 root certification

Commits `afcab02` and `e9db27b` admit the exact 31-by-34 Kronecker product at the root of the
dense degree-64 Hensel tree, then certify the root factor exactly at exponent 39 and avoid lifting
it to exponent 77. Twelve 500-sample processes measure `2.876536 ms` for Symbolica and
`2.552049 ms` for FLINT, or `1.126451` S/F. The fresh predecessor was `1.319908`, so Symbolica
time falls 14.88%. Neighboring final ratios are degree 63 `1.034949`, degree 65 `1.159761`, and
high-height degree 33 `1.108418`. The final binary is `/tmp/flint-comparison-early-root`, SHA-256
`faf9c69f6106c4638ae26953cd21d526cd44f4bb39247e6c64c624b4bd5f03a6`; its LBR profile is
`/tmp/profile-early-root-d64-factor-lbr.perf.data`, SHA-256
`e8a6b6a4ef4cd75a01afe3253e060d50902dff85417c2f7ecc10d469edb81cb0`.

## Superseded #131 bivariate reconstruction and Wang lifting baseline (`0.347488` S/F)

Commit `0632274` remains an active prerequisite of the current #131 route. This section records
the immediately preceding checkpoint; its `0.347488` measurement is superseded by the bounded
dense-Montgomery result above and must not be used as the current scoreboard row.

Commit `0632274` focuses on PolyBench five-variable uniform factorization #131. The old
`1.745716` S/F result had already become `0.478628` through one-image Wang leading-coefficient
reconstruction. A fresh source-matched control at `ec6a131` measured `22.084579 ms` for Symbolica
and `46.647846 ms` for FLINT, or `0.473178` S/F. Six final 500-sample processes measure
`16.088614 ms` and `46.299843 ms`, with median process ratio `0.347488`. Symbolica time falls
27.15% in this pass and 80.31% from the old `81.725573 ms` state; Symbolica is now 2.88x faster
than FLINT on #131.

The accepted changes are deliberately narrow:

- Wang sampling rejects a scalar-content image before derivative and polynomial-GCD screens. The
  full univariate-content test remains at the original successful-screen point, so non-Wang paths
  keep their old cost ordering.
- An image already certified square-free, primitive, and free of retained-variable monomial
  content enters `bivariate_factor_reconstruct` directly. Its square-free univariate sample calls
  `factor_reconstruct` after primitive/sign normalization instead of repeating the generic
  square-free decomposition.
- `sparse_coefficient_hensel_lift_mod_prime` imposes the true leading coefficients and checks the
  exact product before allocating modular factors, complementary products, or a coefficient bound.
  The bound is supplied lazily only if a p-adic correction is actually necessary.
- A Wang leftover unit `-1` is absorbed into one image factor and one true leading coefficient;
  the 1,665-term target is no longer cloned and negated.
- A two-factor bivariate image of main degree at most 64 and factor-bound height at least 256 bits
  tries eight primes above 65,000,000. The proof `p * (degree + 1) <= u32::MAX` retains direct
  `u64` Montgomery accumulation. Rejected wide candidates fall back to the complete prime search
  above 101. The coefficient modulus `p^k` is constructed as a composite modular ring.
- Wang-certified multivariate Hensel stages use the existing retry context. Sparse stages defer
  intermediate product checks, and the complete unshifted factorization is certified once at the
  end. A sparse failure or failed certificate takes the existing bounded bivariate retry.

The measured checkpoints are:

| Checkpoint | Symbolica median | FLINT median | S/F |
|---|---:|---:|---:|
| fresh `ec6a131` control, 6x500 | `22.084579 ms` | `46.647846 ms` | `0.473178` |
| sampling/direct-dispatch/lazy-setup fast paths, 6x500 | `20.265904 ms` | `46.469836 ms` | `0.436629` |
| bounded 26-bit bivariate prime, 6x500 | `18.369984 ms` | `46.487771 ms` | `0.395320` |
| deferred Wang product verification, final 6x500 | `16.088614 ms` | `46.299843 ms` | `0.347488` |

The checkpoint binary is `/tmp/flint-comparison-pb131-deferred-verify`, SHA-256
`85c2463a10612a717bbaddfdb5f65e24a0004e31d890f7c13b12147cda7432d0`; its build JSON SHA-256 is
`0f22c3b9d7fadab57e2a0efc47947d6a2703066e54b6e5e901a8d61815291e71`. Raw final runs are
`/tmp/pb131-deferred-final-{01..06}.csv`. The frozen fresh control is
`/tmp/flint-comparison-chunked-dense5-final`, SHA-256
`a4e7e2df9a33f1a91cf4234ba6b32bb2a12ea13994d15c2599a4fb7603b0bac4`; its raw runs are
`/tmp/pb131-ec6a131-baseline-{01..06}.csv`.

The checkpoint 300-sample LBR profile is `/tmp/profile-pb131-deferred-final-lbr.perf.data`, SHA-256
`4f5e45f6e1047d80e6e13f63e9c5594b0d223a3870aab7b9c618e9ba94213f2e`; its paired timing is
`16.180866 ms` versus `46.739298 ms`. Relative to the fresh profile, the linear univariate
Diophantine lift falls from 4.17% to 1.14% of combined cycles. Relative to the pre-defer 26-bit
profile, multivariate Hensel falls from 11.28% to 8.18%, and the 3.61% per-stage product subtree
disappears. Bivariate Bernardin lifting remained the largest Symbolica subtree at 11.30% of
combined cycles. The dense arbitrary-modulus kernel documented above implements that next step.

Guard measurements are source-matched, sequential, and single-threaded. #32 changes
`0.272966 -> 0.222788`; #159 changes `0.982902 -> 0.899827`; generated dense three-variable
factorization changes `0.680827 -> 0.623487`; #163 is unchanged within noise at `0.217709`.
The corresponding raw files are `/tmp/pb131-guard-*` and `/tmp/pb32-deferred-ab-*`.

The exact high-height reconstruction tests prove both wide-prime selection and fallback after all
eight wide candidates divide the sampled leading coefficient. All targeted Wang, signed-unit,
bivariate, and evaluated-Hensel tests pass. The 88 other factor tests pass; the unrelated
`galois_upgrade` test is nondeterministic in the aggregate run and passes immediately in isolation.
The full library run excluding that test passes 508 of 509 tests; the only failure is the already
documented root-isolation fixture accepting a different valid isolating interval.

An unbounded prime start near `2^31` was not retained. Three 200-sample processes measured median
`0.395389` S/F, statistically the same as the bounded 26-bit policy, but large finite-field
products leave direct `u64` Montgomery accumulation and the search could exhaust the upper half
of the `u32` prime range. Its binary is `/tmp/flint-comparison-pb131-wide-prime`, SHA-256
`08d4ef5c91c8a20f89858622a0378c43a36e33b12d5a863aec6398620f66d0e7`.

## Accepted chunked mixed-radix integer multiplication

Commit `ec6a131` replaces the full mixed-radix accumulator only for the former worst generated
dense five-variable degree-7 GCD products. Each operand has 791 or 792 total-degree terms, each
product performs about 627,000 coefficient pairs, and the output contains 6,228--6,967 terms. The
old path allocated a flat `15^5 = 759,375`-entry `i128` array, about 11.59 MiB, for every product.

`ChunkedDenseIntegerMul` pulls out the most-significant active lexicographic variable. It convolves
the eight input rows into fifteen output rows and reuses one `15^4 = 50,625`-entry `i128`
accumulator, about 0.77 MiB, for their inner variables. The mixed-radix inner indices remain
additive, so the hot update is still one direct coefficient product into
`accumulator[left_inner + right_inner]`. The implementation processes 128-by-128 term blocks and
scans and clears only the active prefix after each output row.

The request validates strictly increasing input indices, a positive inner length dividing the
output box, at most 256 outer rows, and
`max(left_inner) + max(right_inner) < inner_len`; the last condition proves that inner addition
cannot carry into the outer row. It accepts only one-word input coefficients and uses the existing
conservative absolute coefficient bound to prove `i128` accumulation safe. Bounds that fit `i64`
decline this route so the faster flat `i64` kernel remains first. Every rejection uses the previous
dense or generic fallback.

Polynomial dispatch selects the route only with at least five variables, a mixed box of at least
`2^18` cells that is larger than the coefficient-pair count, an inner chunk of at most `2^16`
cells, at least eight outer rows, and a mixed-box/simplex ratio of at least 64. The selector derives
the chunk from the most-significant active variable, not blindly from the first declared variable.
No measured finite-field, high-height, sparse, or lower-variable case enters this integer kernel.

Six alternating source-matched processes with 500 samples per backend give these process-median
results:

| Source | Symbolica median | FLINT median | S/F |
|---|---:|---:|---:|
| frozen pre-change control | `4.363303 ms` | `2.599683 ms` | `1.679650` |
| chunked candidate | `1.907774 ms` | `2.551780 ms` | `0.748808` |

Symbolica time falls 56.27%, the ratio falls 55.42%, and Symbolica is now about 1.34x faster than
FLINT. The exact final binary independently validates at `1.915284 ms` versus `2.549281 ms`, or
`0.751304` S/F, over 500 samples. Its path is
`/tmp/flint-comparison-chunked-dense5-final`, SHA-256
`a4e7e2df9a33f1a91cf4234ba6b32bb2a12ea13994d15c2599a4fb7603b0bac4`; the build JSON SHA-256 is
`e452febeb49f8f2b42f30a3424c01880870391ce2ee92b4cacfe4a64c87189e0`. The six paired raw files
are `/tmp/dense5-chunk-ab-{01..06}-{control,candidate}.csv`; the exact final validation is
`/tmp/dense5-chunk-final.csv`.

The first compact-simplex attempt was rejected. A fixed-width `i64` by `i64` to `i128`
`TotalDegreeIntegerMul` reduced the workspace to `C(19,5) = 11,628` entries, but its candidate was
still `4.392562 ms` versus FLINT's `2.647936 ms`, or `1.658863` S/F. The old flat profile spent
46.07% of cycles in `DenseIntegerMul::run`, including 14.52% in `memset` while clearing the flat
array. The compact profile spent 45.91% in `TotalDegreeIntegerMul::run`: rank-table loads and
validity branches replaced the saved zeroing cost. The rejected binary is
`/tmp/flint-comparison-total-degree-i128-v1`, SHA-256
`0957fef6b4eb89c96b444b6b5ecccc771e00edf8519f9f0d9d98299d76796940`; profiles are
`/tmp/profile-dense5-{v6-control,total-degree-v1}.perf.data`. That production prototype was removed.

FLINT's corresponding algorithm is the chunked LEX array multiplication selected in its
`fmpz_mpoly/mul.c`, implemented in `fmpz_mpoly/mul_array.c`, with its main-variable split in
`mpoly/main_variable_split.c`. FLINT additionally chooses one- or two-word accumulation per output
row from coefficient bounds; the uniform `i128` Symbolica route is already faster on this target,
so that extra complexity was not copied.

All 121 numerica unit tests and 22 numerica doctests pass. The Symbolica library suite passes 507
of 508 tests; the only failure is the previously documented root-isolation fixture accepting a
different valid interval than its hard-coded expectation. New tests cover cancellation and sorted
output, invalid/carrying/unsorted layouts, excessive rows, conservative overflow fallback, selector
boundaries, and the exact 6,967-term product against the independent total-degree implementation.

## Deferred global checkpoint: dense eight-variable GCD products

After the accepted dense-five replacement, the ordered worst primary row is generated dense
eight-variable degree-5 GCD products at `1.610087` S/F. It has not been investigated in this pass.
The next pass should profile its exact current multiplication route before broadening any selector;
the new chunked path intentionally does not apply to its geometry. Dense seven-variable degree-7
products at `1.584134` and high-height five-variable 128-bit products at `1.576901` are the next
guards.

## Accepted one-image Wang leading-coefficient reconstruction

Commit `54e1fcb` removes repeated leading-coefficient image searches for dense automatic
bivariate-start factorization when the main-variable leading coefficient is one nonconstant
monomial. The old #131 profile spent 49.3% in `find_sample`, 39.0% in
`lcoeff_precomputation`, and 83.6% in the repeated bivariate reconstruction subtree. The selected
order was `[x2,x4,x5,x1,x3]`; the old path factored roughly six bivariate images before lifting.

The new path evaluates sampled coordinates at distinct small primes, factors one admissible
primitive bivariate image, and assigns the irreducible factors of the monomial leading coefficient
from their pairwise-coprime evaluations. It certifies that the reconstructed leading coefficients
multiply to the target up to the integer unit `+1` or `-1`, carries that unit into the existing
Hensel scaling identity, and otherwise returns to the old random three-image path with its bounds
and retry counters reset. Downstream lift or product-verification failures recurse with an explicit
factor bound, which disables the shortcut.

The selector requires Auto mode, no existing factor bound, at most five active variables, a
bivariate-box score strictly above `4.0`, and at most 96 total degree across sampled coordinates.
The density guard keeps the two measured wins (#131 at `4.625`, #32 at `4.812`) and excludes #159
at `3.434`. The degree guard prevents fixed-prime evaluation from creating very large images on
high-gap inputs. Debug assertions verify that aligned sampled and reconstructed factor leading
coefficients are equal and all rescaling corrections are units.

The final 500-sample #131 measurement is `22.252255 ms` for Symbolica versus `46.491802 ms` for
FLINT, or `0.478628` S/F. The v5 Symbolica median was `81.725573 ms`, so time fell 72.8% and the
former 1.75x loss became a 2.09x win. The 200-sample #32 row is `24.829520 ms` versus
`90.962032 ms`, or `0.272966` S/F; its frozen Symbolica control was `103.603116 ms` and its v5
ratio was `1.124230`.

A broader <=5-variable candidate was rejected because #159 changed from a source-matched
`107.901595 ms` control to `109.223013 ms`, a repeatable 1.2% regression. The density guard restores
the final row to `106.208527 ms` versus FLINT's `108.056071 ms`, or `0.982902` S/F. Other final
guards are #163 at `0.216817`, generated dense three-variable factorization at `0.680827`, and
eight-variable #84 at `0.691758`.

The exact final binary is `/tmp/flint-comparison-wang-final`, SHA-256
`b8d2ef9762f44d283087db05be833a5acf0809d1f42d2a986927028df623b137`; its build JSON has SHA-256
`ae4c6847df6ac814b19d4b75294feea89331c79d64e431d0dabc68331620c670`. The rejected broad binary
is `/tmp/flint-comparison-wang-lcoeff-v2`, SHA-256
`981237843c1877b9bc048160ad4c5ca8bdf442850fb12f7d7daf7a3a4ae24140`. The first release candidate
silently fell back because it compared a reconstructed positive leading coefficient against the
negative square-free core without allowing the unit `-1`; its binary is
`/tmp/flint-comparison-wang-lcoeff-candidate`, SHA-256
`9fa9b81240eca6fb19c3e30da3b48a5bdb5bf259a1b2dd663786992d9af3ec3e`.

All 87 sequential factor-module tests pass. They cover a first-image certificate, cyclic prime
rotation, signed lifting, nonunit-content fallback, bounded and forced-bivariate exclusions, the
strict density boundary, and the exact PolyBench density geometries. The old profile is
`/tmp/profile-pb131-wide-u16-lbr.perf.data`; the rejected sign candidate is
`/tmp/profile-pb131-wang-candidate-lbr.perf.data`.

## Accepted wide packed high-gap multiplication

Commit `843ce58` represents five-to-eight-variable polynomial monomials with a `u128` key made
from two independently added packed-`u16` words. The bounded row heap therefore avoids allocating
and hashing a full exponent vector for every coefficient pair when output degrees exceed 255.

The former worst canonical high-gap eight-variable degree-4 row changes
`2.019693 -> 0.524530` S/F. The final 5,000-sample measurement is `0.200774 ms` for Symbolica
versus `0.382769 ms` for FLINT. The frozen source-matched control measured `0.645276 ms` for
Symbolica and `1.654253` S/F, so Symbolica time fell 68.9%. A degree-5 stress variant changes
`1.561459 -> 0.475039` S/F, with the Symbolica median falling `1.854737 -> 0.503020 ms`.

The selector requires five through eight variables, nonnegative polynomial exponents, every output
degree representable in both 16 bits and `E`, and the existing few-row work bound. High and low
`u64` halves are added independently, preventing a carry between variables 4 and 5. Tests cover
the exact target in both operand orders, partial and full low words, sums at 65,535 and rejection
at 65,536, `u8` and `i16` representability rejection, collisions, cancellation, and a finite-field
image. All 37 polynomial tests pass.

The exact final binary is `/tmp/flint-comparison-wide-u16-final`, SHA-256
`0642e60e02f9cd86f4e64bd9135b2e02f9919209b7bd85840d65c3930b36a54c`. The frozen control is
`/tmp/flint-comparison-few-row-cold-candidate`, SHA-256
`39edd985345a93b1c4d80909b6551a8fa6f7fae02759be2060867f3a867865b8`.

Two older packed paths have separate pre-existing signed-width correctness gaps: an `i8` output
sum from 128 through 254 and a four-or-fewer-variable `i16` sum from 32,768 through 65,534 can be
packed and then unpacked through a narrowing cast. The new wide selector explicitly rejects these
cases; fixing the older routes is outside this performance change.

## Accepted few-row packed multiplication

Commit `b8b169f` retains the original 4,096 coefficient-pair limit for general packed row merging,
but allows up to 16,384 pairs when the smaller operand contributes at most 16 rows. The hot
original selector remains inline; the extended predicate is cold and out of line, avoiding the
code-layout movement seen in the first candidate.

In the consistent generated sweep, sparse eight-variable degree-5 construction changes
`1.905386 -> 0.780017` S/F (`0.463216 ms` versus FLINT's `0.593854 ms`), and sparse five-variable
degree-7 changes `1.901314 -> 0.791465`. A 5,000-sample isolated source-matched comparison reduced
the eight-variable Symbolica median from about `1.135 ms` to `0.466791 ms` and changed
`1.340679 -> 0.552281` S/F. This is about a 59% reduction in Symbolica time.

The exact candidate binary is `/tmp/flint-comparison-few-row-cold-candidate`, SHA-256
`39edd985345a93b1c4d80909b6551a8fa6f7fae02759be2060867f3a867865b8`. Its frozen control is
`/tmp/flint-comparison-f39c09b-system-lto`, SHA-256
`a15be1b3118223e9f1a4e180f8b707f40c5c0e79e211cc2943ef03336b438ce2`. All 35 polynomial-module
tests pass. All 23 refreshed PolyBench product rows remain faster than FLINT; their candidate
medians are `0.613690` for GCD products and `0.617710` for factor products.

## Accepted initial primitive bivariate sample

The dense three-variable degree-6/5 factorization previously spent about 87.5% of its Symbolica
cycles in `find_sample`: the sampler factored three complete bivariate images before beginning
reconstruction. The three-image rule was introduced to lower the chance of an expensive lift from
an over-split specialization, not to protect correctness. Current reconstruction verifies the
exact final product and retries failures with an explicit factor-count bound.

Commit `f39c09b` accepts the first admissible image only on the initial unbounded attempt when the
main-variable leading coefficient is constant and the bivariate content is a unit. Any downstream
failure already recurses with `Some(factor_count)`, which disables the shortcut and retains three
admissible images. A one-factor admissible image is always accepted because the preserved main
degree and square-free screens prevent genuine factors from specializing to units. Route tests
prove one factorization for the initial primitive path, three for a bounded multi-factor retry,
and one for a bounded irreducible image. All 83 factor-module tests pass.

The exact committed full-LTO binary is `/tmp/flint-comparison-f39c09b-system-lto`, SHA-256
`a15be1b3118223e9f1a4e180f8b707f40c5c0e79e211cc2943ef03336b438ce2`. Its 1,000-sample final
measurement is `2.797780 ms` for Symbolica versus `4.160090 ms` for FLINT, or `0.672529` S/F. The
source-matched control measured `7.240756 ms`; Symbolica's median therefore fell 61.4%, and the
case moved from the canonical worst to about 1.49x faster than FLINT. The 12 other potentially
affected multivariate factor rows were rerun at the final source and show no structural regression.

The control profile and reports are `/tmp/dense3-factor-current-lbr.perf.data`,
`/tmp/dense3-factor-current-{self,children}.txt`, and `/tmp/dense3-factor-current.csv`. The frozen
exploratory candidate is `/tmp/flint-comparison-first-admissible-candidate`, SHA-256
`f06d8bb62bd485a1e949af708d7d2698fba7f07d568423ddbf43b6fc861ced95`.

The final factor-module run passes all 83 tests. The complete library run passes 499 of 500 tests;
the only failure is the unrelated root-isolation fixture, which repeatedly returns a different
valid-looking rational interval for the same simple root than the test's hard-coded interval. No
integer, multiplication, GCD, resultant, or factorization test fails.

## Rejected exact convolution windows and Newton remainder integration

An exact dense integer convolution-window prototype computed only coefficients in a requested
half-open interval, with i64, i128, and fused GMP accumulation. Exhaustive small-window tests,
tag-boundary tests, invalid intervals, and no-GMP tests passed. On the Hensel fallback shapes it
was generally 2.0--3.7x faster than Symbolica's generic full-product path, although the 34x34
78-bit shape regressed by about 55%. Raw diagnostic output is
`/tmp/convolution-window-vs-full-final.csv`.

The production integration cached reversed-divisor reciprocals and used true low windows for
Newton quotient and remainder recovery. It was correct at divisor lengths 30--34, quotient-degree
boundaries 10/11, and machine through multiprecision moduli, but degree-64 factorization regressed:
the confirmed control was `3.300609 ms`, `1.290968` S/F, while the candidate was `3.360552 ms`,
`1.313523` S/F. The candidate binary is `/tmp/flint-comparison-newton-window-candidate`, SHA-256
`23069ff02e8ea95227017a8751ec206b8d950f1afc372a7824756eedc3f4406c`; its frozen control is
`/tmp/flint-comparison-convolution-window-full-control`, SHA-256
`a2e811e690d3d352cbde1a361b8dcbddfd7d32da685c1127ef2f219bf784331b`.

The integration was reverted. Because the window method then had no production caller, its public
`PolynomialKernels` surface, integer implementation, FLINT wrapper, and 48 diagnostic rows were
also removed instead of committing nearly 1,000 unused lines. Reintroduce it only with a measured
production consumer and keep diagnostic microbenchmarks outside the canonical inventory.

## Previous continuation checkpoint: degree-64 Hensel remainder arithmetic

The accepted source chain now ends with:

| Commit | Change |
|---|---|
| `24f57c6` | Balance three-factor Hensel roots and advance packed-row heap heads in place |
| `25569ba` | Reuse child-factor images at the correction modulus across each Hensel stage |

### Current degree-64 timing and profile

The true degree-64 fixture is `generated factorization: dense 1-variable degrees 33/31 total 64`.
After extracting its exact factor `x`, the degree-63 cofactor selects `p=17` and the modular leaves
`[1,2,10,10,10,30]`. The current accepted implementation measures `1.296474` S/F, approximately
`3.314190 ms` for Symbolica versus `2.556886 ms` for FLINT. Its construction product is
`1.094628` S/F, approximately `5.958 us` versus `5.436 us`.

The most recent pinned-core LBR profile predates the small `25569ba` saving but profiles the same
algorithm and fixture. Its 1,000 calls per backend measure `3.373043/2.580603 ms = 1.307076` and
give the following inclusive per-call attribution:

| Phase | Symbolica cycles | FLINT cycles | Difference |
|---|---:|---:|---:|
| complete factorization | `15.002 M` | `11.450 M` | `+3.552 M` |
| Hensel reconstruction/tree | `9.363 M` | `5.903 M` | `+3.460 M` |
| distinct-degree screening | `4.462 M` | `1.738 M` | `+2.724 M` |
| equal-degree factorization | `0.780 M` | `2.926 M` | `-2.146 M` |

Hensel therefore accounts for about 97% of the net cycle gap. Within Symbolica's Hensel subtree,
`DenseIntegerModularUnivariateContext::remainder_monic` is `5.015 M` inclusive cycles per call and
`multiply_raw` is `2.424 M`. Equal-degree factorization is already about `0.267x` FLINT's cost and
is no longer the target. The five-node, seven-stage product tree makes about 260 monic remainders
and 225 dense products for this fixture.

Profile artifacts are:

- binary `/tmp/flint-comparison-hensel-peek-final-system-lto`, SHA-256
  `d1a02003b7bb6f12af759bfcce6fd85642b76c7fb4602c0a5416677ce6780b00`;
- `/tmp/profile-d64-current-24f57c6-lbr.perf.data`, SHA-256
  `ac4c3487262b88e1f5d642f81894d576407416c850a6b0b5dbc40c0b64358cba`;
- `/tmp/profile-d64-current-24f57c6.csv`, SHA-256
  `51142a9010733cb98d382dfa250ec4af7c9edb975c0e6f927f66e18f29fa2931`;
- `/tmp/profile-d64-current-24f57c6-{children,flat}.txt`, with respective SHA-256 values
  `8ef0ae1ba68ce865615e534d65b2f00eb81239beede5e9bbf2f6a1436e47a7a6` and
  `49ab6b2fab5983bb52e733d15298b87d255ced94b1322ec38ed68410a99a2c55`.

### Accepted correction-modulus image reuse

For an old precision `m=p^old` and correction precision `c=p^(new-old)`, the ceiling-halved
schedule guarantees `c` divides `m`. A lifted child is `U=u+m*du`, hence `U` and `u` have exactly
the same canonical image modulo `c`; the same holds for `W=w+m*dw`. The factor-correction half of
the stage already computes those images. Commit `25569ba` retains them for the subsequent Bezout
correction instead of reducing both lifted children again. The correction degrees are below their
monic divisors, so the leading ones are unchanged.

This removes 60 dense-vector reductions, approximately 852 coefficient `%` operations, from the
degree-64 tree. Although the gain is below the usual 3% threshold, it is retained because the
change only removes provably redundant work, its invariant has a direct small- and GMP-modulus
test, and all three adjacent dense degrees improve in repeated source-matched runs:

| Fixture | Candidate Symbolica | Control Symbolica | Candidate S/F | Control S/F | Absolute change |
|---|---:|---:|---:|---:|---:|
| degree 63 | `3.078308 ms` | `3.107922 ms` | `1.029727` | `1.039401` | `-0.95%` |
| degree 64, 12 processes | `3.314190 ms` | `3.336628 ms` | `1.296474` | `1.306468` | `-0.67%` |
| degree 65 | `3.521314 ms` | `3.550522 ms` | `1.164159` | `1.171766` | `-0.82%` |
| high-height degree 33 | `1.531238 ms` | `1.533379 ms` | `1.112701` | `1.112776` | `-0.14%` |

Every process uses 200 paired samples, pinned-core sequential execution, default release features
including `faster_alloc`, and system FLINT 3.6.0. Raw files are
`/tmp/{d63,d65,highheight}-reusemod-{candidate,control}-{1..6}.csv` and
`/tmp/d64-reusemod-{candidate,control}-{1..12}.csv`. The candidate binary is
`/tmp/flint-comparison-d64-reusemod-system-lto`, SHA-256
`9674b4640edcb5d304a1bc22b18e3b18feb94bdc41496028d38102a6fe343bb5`; its build log SHA-256 is
`6e569da59e9f7c6ea51e319735fefe0162972532bfb15ea397e31ca166c349ad`.
All 81 factor-module tests and `cargo fmt --check` pass.

### Rejected batched classical remainder

A separate candidate reordered monic long division into one owned integer dot product for each
quotient and remainder coefficient. It was correct, kept the GMP accumulator large across the
whole dot product, reduced once per coefficient, and avoided cloning the discarded quotient.
However, it still performed `quotient_length * divisor_degree` coefficient products and could no
longer skip an entire update row when a quotient pivot was zero.

Six alternating 200-sample processes per case gave:

| Fixture | Candidate Symbolica | Control Symbolica | Candidate S/F | Control S/F | Absolute change |
|---|---:|---:|---:|---:|---:|
| degree 63 | `3.109843 ms` | `3.105289 ms` | `1.039564` | `1.037742` | `+0.15%` |
| degree 64 | `3.303800 ms` | `3.345429 ms` | `1.281515` | `1.305739` | `-1.24%` |
| degree 65 | `3.472808 ms` | `3.556066 ms` | `1.147178` | `1.174235` | `-2.34%` |
| high-height degree 33 | `1.522095 ms` | `1.529497 ms` | `1.103676` | `1.110492` | `-0.48%` |

The gains are too small and inconsistent for the added recurrence and sparse-pivot risk, so the
candidate was reverted without a commit. Raw files are
`/tmp/{d63,d65,highheight}-dotrem-{candidate,control}-{1..6}.csv` and
`/tmp/d64-dotrem-{candidate,control}-{1..6}.csv`. The rejected binary is
`/tmp/flint-comparison-d64-dotrem-system-lto`, SHA-256
`a0904b66e79efeffb5b635cb2d14e87dead626af16586e404dff1c4f3320acd2`; its build log SHA-256 is
`891a4e51712c816f1dede8a8bd40351bd40ef70f872796691bbec1501d70def9`.

### Next degree-64 implementation target

FLINT uses coefficient-at-a-time division only when the divisor length is at most 30 or the degree
gap is at most 10. Larger Hensel remainders use Newton division with genuine low and middle
convolution windows. Symbolica's rejected reciprocal prototype `d675fd8` regressed 29.17% because
it formed complete products and truncated them afterward; do not restore it unchanged.

The next implementation should add one exact contiguous `[output_start, output_end)` convolution
capability to `PolynomialKernels`, not another method to `Ring`. A private integer operation
context should initially enumerate only pairs contributing to the requested coefficients and keep
one fused fixed-width or GMP accumulator per output cell. The same capability expresses both
`mullow` and `mulmid`. Benchmark it independently at lengths 8, 16, 24, 30, 31, 32, 48, and 64,
degree gaps around 10/11, and coefficient heights from 16 through 1024 bits. Only after the exact
degree-64 windows beat full-product-and-truncate should it be used for cached reciprocal extension,
quotient recovery, and the low remainder product behind FLINT's `lenB > 30 && gap > 10` guard.

## Previous continuation checkpoint: Hensel balance and packed-row heap advancement

The accepted source chain is:

| Commit | Change |
|---|---|
| `d50fe00` | Use a retained dense finite-field context for equal-degree factorization |
| `d675fd8` | Prototype reciprocal/Newton Hensel remainders; rejected |
| `e68a1d0` | Revert the slower reciprocal/Newton Hensel prototype |
| `5be1daa` | Merge bounded packed sparse multiplication rows |
| `41302eb` | Select guarded sparse-univariate factor orders and preserve bivariate fallback order |
| `24f57c6` | Balance three-factor Hensel roots and advance packed-row heap heads in place |

All commits use author and committer `Ben Ruijl <ben@ruijl.ch>`. The Newton experiment and its
revert are intentionally retained in history so the negative result is discoverable.

### Balanced three-factor Hensel root

The high-height univariate degree-33 factorization is

```text
((1+65537*x)^17-1) * ((1-65539*x)^16+1).
```

After the exact linear factor is removed, screening at the selected prime `p=5` produces modular
leaf degrees `[8,8,16]`. The former factor order split the recursive Hensel root as `8|24`. That
root split is not exact, so it lifts the remaining `8|16` child all the way to the roughly
1,060-bit global bound. FLINT combines the two degree-eight leaves first, making the root
`(8+8)|16`; this split is exact and lets the remaining child use its lower local coefficient
bound.

When high linear-Hensel pressure has selected exactly three modular factors, Symbolica now moves
the factor closest to half of the total degree to the first position. This minimizes the root
degree imbalance without changing the general reconstruction algorithm. A regression test checks
the `[8,8,16] -> [16,8,8]` order and the full high-height test verifies reconstruction, the selected
prime, an exact subtree split, local recombination, a lower child modulus, and use of quadratic
Hensel lifting.

Six alternating 200-sample processes with the current system-FLINT full-LTO binary give a median
S/F of `1.112455`, down from the source-matched control's `1.997930`. Median Symbolica time is about
`1.53 ms`, versus about `1.38 ms` for FLINT; Symbolica absolute time fell by roughly `40%`. The
constituent product is already faster than FLINT at `0.573810` S/F, about `8.3 us` versus
`14.3 us`, so multiplication is not the remaining gap in this case.

Raw final timings are `/tmp/final-factor-highheight-{1..6}.csv` and
`/tmp/final-product-highheight-{1..6}.csv`. The measured binary is
`/tmp/flint-comparison-hensel-peek-final-system-lto`, SHA-256
`d1a02003b7bb6f12af759bfcce6fd85642b76c7fb4602c0a5416677ce6780b00`.

### Bounded in-place packed-row heap advancement

The packed-`u8` row merge previously removed every nonterminal heap head and then inserted the
next entry in that row. For sparse output geometries, replacing the mutable heap root with the next
larger row key performs one heap repair instead of a separate removal and insertion. Equal keys
are still accumulated before an output term is emitted, and an exhausted row still removes its
root.

Every product that reaches this bounded packed-row path uses the in-place update. Compact,
collision-heavy boxes have already been accepted by the earlier mixed-radix dense kernel; a
separate pop-and-push branch here would therefore be unreachable through public multiplication.
The row kernel is isolated in a non-inlined bounded helper so generic dense multiplication call
sites do not absorb its code. Differential tests cover replacement, cancellation, terminal
removal, asymmetric supports, the packed-byte boundary, the large-product fallback, irregular
`u16` storage, and a collision-heavy row merge.

On the generated high-gap five-variable degree-5/gap-64 product, six alternating 10,000-sample
processes give `0.799572` S/F, down from the source-matched old path's `1.040981`; the ratio improves
by `23.2%`. All 23 current PolyBench product rows now beat FLINT. Their current S/F
best/median/worst are `0.585393/0.604569/0.730825` for the 12 GCD products and
`0.566138/0.601621/0.710007` for the 11 factor products.

Lowering the integer Kronecker threshold from 32 to 16 terms is rejected. The 17-by-17
high-height product regressed from about `0.00820 ms` on the existing array path to `0.01262 ms`.
Reusing the destination allocation while importing GMP limbs improved that Kronecker candidate to
`0.01180 ms`, but it remained roughly `44%` slower than the array path and made the complete
factorization roughly `14%` slower. The current array kernel already avoids a temporary GMP
product for all 233 of the 289 coefficient pairs involving a large integer, so allocation reuse is
secondary for this shape.

Raw current product timings are `/tmp/final-product-highgap5-{1..6}.csv` and
`/tmp/final-pb-{gcd,factor}-products-{1..6}.csv`.

Final-source validation passes all `81/81` factor-module tests and all `6/6` packed-row tests with
default features in the test profile. The full-LTO paired harness also validates every measured
Symbolica product and factorization against FLINT before timing. `cargo fmt --check` and
`git diff --check` pass.

### Dense equal-degree factorization

The benchmark named `dense 1-variable degrees 32/31` has total degree 63. The representative true
degree-64 fixture is `dense 1-variable degrees 33/31 total 64`. In particular,
`/tmp/d64-factor-current-*` and `/tmp/profile-d64-current-3eb4eb2-*` are mislabeled degree-63
artifacts and must not be attributed to degree 64.

After extracting the exact linear factor, the true degree-64 fixture screens modulo 17 and retains
six modular leaves with degrees `[1,2,10,10,10,30]`. Its distinct-degree blocks are `(1,1)`,
`(2,2)`, `(10,30)`, and `(30,30)`. The retained equal-degree step must split the degree-30,
distinct-degree-10 block into three degree-10 factors. The old implementation performed its
Cantor-Zassenhaus powering through generic sparse polynomial multiplication and
`quot_rem_univariate_fast`, even though the block is a small dense univariate polynomial.

`DenseZpEqualDegreeContext` now reuses the dense reciprocal/convolution machinery from distinct
degree factorization for modular powering. It keeps the modulus, reciprocal, random residue,
powering buffers, and product workspaces dense, and materializes a polynomial only at a GCD or
returned split boundary. Unsupported or sparse shapes retain the generic equal-degree path. The
factor/cofactor stack order deliberately matches the old traversal because the later synchronized
Hensel tree uses deterministic degree ties.

Six alternating 100-sample full-LTO processes against source control `3eb4eb2` give:

| Version | Symbolica median | FLINT median | S/F |
|---|---:|---:|---:|
| dense-EDF candidate | `3.438806 ms` | about `2.43 ms` | `1.417999` |
| source control | `4.2553655 ms` | about `2.43 ms` | `1.748240` |

Absolute Symbolica time falls by `19.19%`. Independent guards give `43.29%` at degree 63,
`27.01%` at degree 65, and `5.04%` for the high-height degree-33 factorization; their current S/F
ratios are respectively `1.175019`, `1.312650`, and `2.0059855`. A separate 500-sample profile
measures `3.451276 ms` versus `2.437016 ms`, ratio `1.416189`.

The LBR profile must be normalized by the concurrently measured FLINT cycles because the machine
frequency differed between the control and candidate runs. After that calibration, total
Symbolica work falls by `20.6%`, and equal-degree factorization falls by `83.9%`. The generic EDF
`quot_rem_univariate_fast` subtree, formerly about `3.446 M` cycles per factorization, disappears.
EDF is now only `4.9%` of Symbolica time and about `0.305x` FLINT's EDF work. The remaining
Symbolica profile is Hensel/tree `61.5%` and dense DDF/screening `29.4%`; within Hensel,
`remainder_monic` is about `4.74 M` cycles and `multiply_raw` about `2.62 M`.

The next Hensel experiment cached a reciprocal of every sufficiently large reversed monic divisor
and recovered quotients with Newton doubling and two full products. It is rejected. Six
interleaved 100-sample processes measured `7.963034 ms` for that candidate versus `6.164611 ms`
for dense EDF alone under the same shifted machine load, a `29.17%` regression. At the degree-30
divisors in this workload, reciprocal preparation and full-product work cost more than the
coefficient-at-a-time remainder loop. Do not retry this formulation without truncated
multiplication that is independently faster at these exact lengths.

### Bounded packed sparse product rows

All 35 constituent multiplications behind the 23 PolyBench construction rows have 35--50 terms per
operand and at most 2,500 coefficient pairs. Their products are almost collision-free: the
coefficient-pair/output-term ratio is between `1.000` and about `1.039`. The old packed path still
maintained both a `BTreeMap<u64, Vec<(i,j)>>` and a heap, allocating map/vector state for nearly
every output monomial.

The new path is confined to packed-byte products for which the smaller operand has at most 64 terms
and the checked pair count is at most 4,096. It treats every term of the smaller operand as a sorted
row over the larger operand, retains one `(packed exponent,row,column)` cursor per row, merges equal
exponents at the heap front, initializes each output coefficient from its first pair with
`Ring::mul`, and calls `add_mul_assign` only for collisions. The existing total-degree dense path
is tried first. Packed
products outside the bounds, products that require `u16` coordinate packing, and all other
multiplication paths are unchanged. This is generic over `Ring`; it adds no integer specialization
or new `Ring` method.

Across six alternating 100-sample processes, all 23 construction rows beat the source-matched old
path. Taking each case's median paired per-block S/F reduction and then the family median gives
`52.20%` for the 12 GCD construction rows and `54.02%` for the 11 factor construction rows. The
timed transient source `268a353` is performance-equivalent to accepted commit `5be1daa` for these
rows; the later change only generalizes test coverage. A twelve-process, 1,000-sample rerun of the
initially noisy #84 product gives `0.073047 ms` versus
`0.107743 ms`, a `32.20%` reduction. Direct current S/F medians are:

| Family | n | Best S/F | Median S/F | Worst S/F |
|---|---:|---:|---:|---:|
| PolyBench GCD products | 12 | `0.9018375` | `0.94212425` | `1.100743` |
| PolyBench factor products | 11 | `0.9006325` | `0.942414` | `1.2681525` |
| all PolyBench products | 23 | `0.9006325` | `0.942414` | `1.2681525` |

Fourteen of the 23 rows now beat FLINT. These absolute ratios use system-linked FLINT 3.6.0,
whereas the older inventory used bundled FLINT and a different whole-program LTO layout. The
source-matched candidate/control speedups are the stronger attribution measurement.

Release differential tests cover collision cancellation, operand orientation, the exact packed
byte boundary, the 65-by-65 fallback, and irregular eight-variable polynomials stored with `u16`
exponents.

### Guarded sparse-univariate order for PolyBench factor #84

The 1,878-term #84 input has degrees `[8,10,8,32,24,5,3,3]` and leading-layer sizes
`[3,2,2,1,1,8,2,35]`. The old order `[3,4,1,0,2,5,6,7]` starts with degrees 32 and 24. Its box
density is `1878/(33*25) = 2.276`, so Auto selects bivariate start and repeatedly factors large
bivariate images. Moving original variable 6 (the degree-three `x7`) first gives order
`[6,3,4,1,0,2,5,7]` and density `1878/(4*33) = 14.227`, selecting sparse-univariate lifting.
FLINT's compressed sparse order also starts with this original variable.

The production reorder is Auto-only and requires every one of these guards:

- at least 256 terms and at least two active variables;
- no degree-two variable, preserving the certified quadratic route;
- the current first pair selects bivariate start at density at most 5;
- a candidate leading layer no larger than twice the smallest layer;
- at least a fourfold main-degree reduction;
- candidate/main box density at least 10, leaving margin above Auto's boundary.

The lowest-degree eligible candidate is rotated to the front without changing the relative order
of the other variables. An exhaustive check of the 11 PolyBench factor fixtures shows that only
#84 changes route. Forced univariate, forced bivariate, disabled factorization, quadratic inputs,
and fast uniform #159 keep their former orders. The original order is retained through recursive
Auto retries and restored before any terminal bivariate fallback, so a failed speculative sparse
lift does not spoil the previous bivariate choice.

Six alternating 20-sample full-LTO processes from the final implementation source give:

| Version | Symbolica median | FLINT median | S/F |
|---|---:|---:|---:|
| guarded sparse-univariate order | `14.461677 ms` | `19.0584955 ms` | `0.759183` |
| immediately preceding control | `166.0410395 ms` | `19.1663955 ms` | `8.690880` |

Symbolica improves by `91.29%`, or `11.48x`, and now beats FLINT on #84. The reordered route
certifies the factorization without falling back. The complete 11-case sweep reconstructs every
input. Its only non-target exploratory movement above 3% was fast uniform #159; a dedicated
six-process, 20-sample final-source A/B gives `149.2142965 ms` versus `147.576704 ms`. Its median
S/F moves from
`0.2074465` to `0.2093005`, a `0.89%` ratio change below the acceptance threshold. No other route
changes logically.

With the new PolyBench product rows, #84, and current degree-63, degree-64, degree-65, and
high-height degree-33 rows substituted into the 122-row inventory, best/median/worst become
`0.028678`, `0.92545225`, and `2.0059855`. The best remains the generated high-gap eight-variable
GCD; the new worst is the generated high-height univariate degree-33 factorization. PolyBench
operations alone have best/median/worst `0.100709`, `0.748107`,
and `1.7405235`; #84 is no longer an outlier.

Primary artifacts are:

| Artifact | SHA-256 |
|---|---|
| `/tmp/flint-comparison-dense-edf-d50fe00-system-lto` | `d75e23aa61b880585430e8ab77af72195af8eca96b3409e968f987832dd9d8de` |
| `/tmp/profile-d64-dense-edf-d50fe00-system-lbr.perf.data` | `d783c1693670cc099fba8d0555d203a84d740843be6f6a9a7d045924bb0300ee` |
| `/tmp/flint-comparison-dense-edf-hensel-d675fd8-system-lto` | `01d9185e1651aa95b5abec53ba47c2140f46c6ff51883e962b7aafa29ebe0d85` |
| `/tmp/flint-comparison-packed-row-268a353-system-lto` | `e4d4bc36c4c3ae2078eb23dc57ba0da9fe1cafe47ba2d392d3beb15ef8f0bbcd` |
| `/tmp/flint-comparison-pb84-order-a0d0ed8-system-lto` | `c12aa071b927816aa7750d72da925450e6f7afc306d53feafee4847849e8fe57` |
| `/tmp/flint-comparison-final-7387d1a-system-lto` | `2d67dd2c01f4700732d2f35c7f9f6096b1cc6ff0e349bb978b1df7bdb3180849` |
| `/tmp/flint-comparison-control-3eb4eb2-system-lto` | `0f2af494a059e0fbcdc510e7ffc16af68d5ee136dd8bd6859e3f6f539d9e473a` |

Raw timing families are `/tmp/d64-dense-edf-ab-*`, `/tmp/{d63,d65,highheight33}-dense-edf-ab-*`,
`/tmp/d64-hensel-ab-*`, `/tmp/pb-{gcd,factor}-products-row-ab-*`,
`/tmp/pb84-product-row-ab-s1000-*`, `/tmp/pb84-order-ab-s20-*`, and
`/tmp/pb159-order-ab-s20-*`. The exact final-source heuristic guards are
`/tmp/pb{84,159}-final-source-{candidate,control}-*`.

The final heuristic binary was built from implementation commit `7387d1a`; the subsequent amend to
`41302eb` changes only comments and formatting. Its stderr logs show stale Cargo-cached workspace
metadata `g3eb4eb2+dirty`, even though the measured #84 route and frozen binary SHA identify the
new implementation. Use the build-time source state and SHA above for provenance, not that embedded
version string.

The final source passes all `80/80` release factor-module tests with the default GMP/faster-alloc
features and all `80/80` with `integer-malachite,float-astro,native_code_generation`. All `5/5` packed-row differential
tests pass in release mode. Run `cargo fmt --check` and `git diff --check` again after editing this
handoff and before the documentation commit.

## Previous continuation checkpoint: quadratic PolyBench cases #176 and #178

PolyBench factor #176 was not slow because of coefficient height, GMP allocation, Hensel lifting,
or an exponent-lattice dimension missed by Symbolica. The expanded eight-variable input has 2,092
terms and a global `x5` factor. After removing that monomial, its degree vector is
`[33, 5, 17, 4, 19, 2, 11, 3]`. Treating `x6` as the quadratic variable gives coefficient layers
with `21`, `422`, and `1,649` terms. The discriminant has 64,367 terms but is the square of a
422-term polynomial.

FLINT compresses/shears the exponent representation and then uses Hart's heap-based sparse square
root. The compression does not reduce this input's affine rank or variable count, and Symbolica
already sees the degree-two variable. The decisive difference was that Symbolica proved the
discriminant square by recursively running general multivariate square-free factorization and a
large polynomial GCD on all 64,367 terms. A frozen pre-change profile measured 193.629 ms for
Symbolica and 20.313 ms for FLINT, ratio `9.532x`. In that Symbolica profile, the quadratic subtree
was 40.63% of Symbolica cycles, square-free factorization of the discriminant 26.37%, the separate
outer square-free pass 29.94%, all heap division 22.52%, and heap multiplication 13.03%.

The accepted implementation is private to integer polynomial factorization and consists of four
parts:

1. `SparsePolynomialSquareRootContext` reconstructs an exact root in descending lexicographic
   order. It merges input terms with products of already recovered nonleading root terms; every
   residual must divide exactly by twice the leading root coefficient. It supports nonnegative
   exponents in at most eight packed bytes, checks global degree parity and half-degree bounds,
   uses a one-shot context, and has separate cumulative-pair and live-map fallbacks. Unsupported or
   bounded-out inputs retain the square-free-decomposition method.
2. Integer square-free factorization extracts the coordinatewise common monomial before calling
   `factor_separable`, emitting each variable with its exact multiplicity. This transformation is
   algebraically useful but almost performance-neutral for #176 because the subsequent
   eight-variable separability scan still runs.
3. Large inputs with more than two active variables receive a split-only quadratic prepass. It
   removes integer and monomial content, tries all degree-two variables in estimated cost order,
   uses only the bounded direct square-root path, and accepts only two factors whose exact product
   reconstructs the normalized core. A nonsquare, zero, unsupported, parity failure, or failed
   division is inconclusive and falls back to the old square-free path. A linear child with unit
   content in the selected variable is irreducible by Gauss's lemma; other children are factored
   recursively. The packed prepass requires discriminant degrees at most 255 and falls back when
   they do not fit. The later post-square-free quadratic route checks the actual exponent type; if
   a discriminant product would overflow it, the quadratic calculation widens to `u32` and maps
   the certified result back to the input exponent representation.
4. `PackedSparsePolynomialSquareContext` forms the `b^2` part of a quadratic discriminant by
   visiting each unordered term pair once. Diagonal and off-diagonal coefficient sums are kept
   separately, so every coefficient product is computed once and the off-diagonal sum is doubled
   afterward without creating a temporary GMP product for every pair. Dense mixed-radix and
   total-degree layouts, exponents above 127, fewer than 64 terms, and pair counts above `2^20`
   retain the general multiplication dispatch.

The isolated progression on pinned core 2, default release features, one thread per backend, and
FLINT 3.6.0 is:

| Version | Symbolica median | FLINT median | S/F |
|---|---:|---:|---:|
| frozen pre-change profile | `193.628990 ms` | `20.312697 ms` | `9.532412` |
| direct sparse discriminant root | `30.934655 ms` | `20.059765 ms` | `1.543190` |
| monomial extraction plus root micro-optimizations, exploratory | `30.699388 ms` | `19.976017 ms` | `1.536812` |
| certified pre-square-free split, exploratory | `24.315857 ms` | `19.970389 ms` | `1.217596` |
| certified split plus packed triangular square | `19.891008 ms` | `19.933955 ms` | `0.998269` |
| exact final source, including narrow-exponent widening | `20.395108 ms` | `20.151804 ms` | `1.007296` |

The packed-kernel row is the median of six independent 50-sample processes. Per-process ratios
range from `0.993533` to `1.002528`; the ratio of the process medians is `0.997846`. The exact final
source was rebuilt after adding narrow-exponent safety and measured in another six 50-sample
processes. Its per-process ratios range from `1.001435` to `1.029072`, with median `1.0072955`; the
ratio of its listed process medians is `1.012074`. Both sequences are within the standing 3%
parity threshold. Final-source Symbolica time falls by `89.47%` from the frozen profile and is
`9.49x` faster than its old path. A separate 300-sample perf run on the packed-kernel source gives
`20.334568 ms` versus `20.396429 ms`, ratio `0.996967`.

The normalized LBR profile attributes 95.07% of Symbolica cycles to the certified split. Within
Symbolica, the direct root reconstruction takes 41.30%, the packed square 30.68%, generic heap
multiplication used for the remaining products 14.79%, and split reconstruction 4.30%; neither
the old outer square-free pass nor discriminant square-free factorization appears in the profile.
The packed and generic multiplication work totals about `9.25 ms`, essentially FLINT's
`9.29 ms`. Symbolica's direct root is still about `3.3 ms` slower than FLINT's heap square root,
but Symbolica reaches the quadratic directly and avoids FLINT's content, square-free, and exponent
compression preprocessing. Wall-clock timings, rather than the LBR sampled-cycle ratio, are the
acceptance measurement.

Validation at this checkpoint passes `74/74` factor-module tests with default GMP features and
`74/74` with `integer-malachite,float-astro,native_code_generation`; `cargo check --lib --tests`, `cargo fmt --check`, and
`git diff --check` also pass. Coverage includes exact/nonsquare/canceling sparse roots, packed
high-height squaring, signed content and common-monomial multiplicities, certified early splits,
inconclusive fallbacks, exponent-capacity guards, and both irreducible and split quadratic
discriminants that must widen from `u8` to `u32`.

The same certified route improves PolyBench factor #178 from the preceding robust `3.1210205x` to
an exploratory six-process median `0.2389645x`. The full 11-case factor sweep reconstructs every
input. The apparent #84 regression in three-sample sweeps was checked with six interleaved
10-sample processes against the frozen control: `166.094 ms` candidate versus `164.676 ms`
control, only `0.86%`, below the 3% acceptance threshold. #84 remains the worst factor case and is
the next independent algorithmic target at this historical checkpoint. The current checkpoint
above records the accepted fix.

The factor-operation ratios at this historical checkpoint were:

| Case | Variables/regime | S/F |
|---|---|---:|
| #44 | 8v uniform | `0.100709` |
| #159 | 8v uniform | `0.171694` |
| #163 | 5v uniform | `0.224814` |
| #178 | 8v sharp | `0.2389645` |
| #92 | 8v sharp | `0.839857` |
| #159 | 5v uniform | `0.996512` |
| #176 | 8v sharp | `1.0072955` |
| #105 | 8v uniform | `1.0926815` |
| #32 | 5v uniform | `1.1340355` |
| #131 | 5v uniform | `1.7405235` |
| #84 | 8v sharp | `7.8842045` |

Important rejected explanations and follow-ups:

- FLINT's exponent compression is useful generally but does not reduce #176's dimension and is not
  the missing shortcut here.
- Coefficients in the discriminant are at most about 58 bits, so the old gap was not caused by the
  large-`Integer` recycle cache or GMP allocation.
- Common-monomial extraction and hash-entry/bound micro-optimizations moved #176 by only about 1%;
  they did not replace the separability scan.
- The remaining direct-root context eagerly stores nonleading pair products. A lazy FLINT-style
  row heap would reduce worst-case memory from quadratic to linear in the recovered root length,
  but it is no longer required for #176 performance.
- A fused stream for `b^2-4ac` could avoid materializing the discriminant, but the measured case is
  already at FLINT parity. Do not add that complexity without a new benchmark that needs it.

### Historical next target: PolyBench factor #84

#84 was the worst measured row at `7.8842045x`: six 10-sample process medians give about
`166.094 ms` for Symbolica and `21.05 ms` for FLINT. At that point there was no dedicated profile,
so the following was a static, testable hypothesis. It is retained because the current checkpoint
above confirms it and records the production guard.

The 1,878-term product has degrees `[8, 10, 8, 32, 24, 5, 3, 3]` and no quadratic variable.
Symbolica minimizes leading-layer term count and breaks ties toward higher degree, selecting the
degree-32 `x4` followed by degree-24 `x5`. Its density heuristic then chooses bivariate start, and
`find_sample` can factor roughly three degree-32-by-24 bivariate images before multivariate
lifting. FLINT's exponent compression keeps eight variables but orders a degree-three variable
first; its very-low-density route tries sparse Zippel lifting.

The first A/B experiment should add a test-only order override and force zero-based order
`[6, 3, 4, 1, 0, 2, 5, 7]`. With `x7` first, Symbolica's current density score switches to
univariate start. Compare that order with the current order and separately force univariate start
under the current order, while counting and timing outer square-free/separability work,
`find_sample` attempts and nested bivariate factorizations, leading-coefficient reconstruction,
sparse lifting, and final reconstruction. A successful hypothesis should eliminate the bivariate
image factorizations, reduce variance, and substantially lower wall time. If it does not, profile
sparse lifting and the outer coefficient-content GCDs before changing the production heuristic.

Primary artifacts are:

| Artifact | SHA-256 |
|---|---|
| `/tmp/flint-comparison-pb176-final-source-lto` | `42db9b79fac57488b686472e8b24f3d2e0db0c6bc80f12244ea4d46a0c1b82ac` |
| `/tmp/flint-comparison-pb176-early-split-packed-square-lto` | `5280833a5d80d601ec9dcb9a345066b6f099368ff737b74545305f2351f41a37` |
| `/tmp/profile-pb176-early-split-packed-square-lbr.perf.data` | `d040ea3c2390e58331adc9cde013a5fec31343ff46f9840a110a0e4411087761` |
| `/tmp/profile-pb176-early-split-packed-square-normalized-summary.txt` | `e6d758c44bbf19f115ad5f1035608a56d04bb6ea0e6a8029f39677896268adb5` |
| `/tmp/profile-pb176-d73f12d-lbr.perf.data` | `1309cd1f65b99dff92d7c08d514e31feea68864401eee44fc21d8c5221729114` |

## Previous continuation checkpoint: delayed quadratic-Hensel division

Commit `11c878b` keeps `IntegerModularUnivariateContext` private and changes its modular
quotient/remainder schedule to match FLINT's classical basecase invariant. It reduces a cell when
the cell becomes the current pivot, computes the quotient coefficient once, accumulates exact
fused subtractions into lower cells, discards the eliminated leading cell, and symmetrically
reduces only the final remainder. A cached symmetric interval returns already canonical values
without a modular division. The pivot multiplication owns its left operand, and lower updates use
`Z.sub_mul_assign`, so the path does not create borrowed-by-borrowed GMP product temporaries.

For a dense degree-16 divisor and 32 quotient positions, the former loop performed about `576`
modular reductions per call. The delayed schedule needs `79`: one quotient reduction per pivot,
one reduction when each accumulated cell becomes the next pivot, and one for every surviving
remainder coefficient. FLINT 3.6 uses the same schedule in
`fmpz_mod_poly/divrem_basecase.c`; this fixture cannot reach FLINT's Newton division because its
balanced divisor lengths are at most 18.

The factor module passes `67/67` tests with default GMP features and `67/67` with
`integer-malachite,float-astro,native_code_generation`. New differential coverage includes `5^8` and `5^65`, odd/even
symmetric-representative boundaries, a dense degree-16 divisor with a degree-31 quotient and
interior zero coefficients, constant divisors, and degree-short dividends. The high-height route
still proves that quadratic Hensel lifting executes and reconstructs the original polynomial.

Twelve alternating full-LTO processes with 20 paired samples give:

| Version | Symbolica median | FLINT median | S/F |
|---|---:|---:|---:|
| delayed-reduction candidate | `3.574496 ms` | `1.399899 ms` | `2.550696` |
| source-matched control | `5.001481 ms` | `1.4000325 ms` | `3.5696655` |

All twelve blocks favor the candidate. Absolute Symbolica time falls by `28.44%`; the median of
the per-block S/F changes is `-28.32%`. A separate 500-sample cycle profile gives `3.556181 ms`
versus `1.401596 ms`, ratio `2.537237`.

The high-height construction product is independently neutral candidate/control
(`0.999881x` Symbolica time) and now measures `0.578774x` FLINT over twelve 10,000-sample blocks.
The factor speedup therefore comes from Hensel lifting, not construction. Degree-63 and degree-65
factor guards moved by `2.60%` and `1.94%` in S/F, but test-only routing counters prove that neither
fixture executes quadratic Hensel lifting; these are full-LTO layout shifts. Degree-48/64/80 GCD
candidate/control ratios are `0.995105`, `0.993553`, and `0.994587`. The complete widened
PolyBench factor sweep reconstructs all 11 inputs; at that checkpoint its best, median, and worst
S/F were `0.100709`, `0.996512`, and `7.8842045`.

In the post-change paired cycle profile, Symbolica factorization occupies `68.60%` of total cycles
and FLINT `27.54%`. Symbolica's Hensel subtree is still `59.14%` of the paired total, but the direct
quadratic `quot_rem` subtree has fallen to `9.54%`. Dense integer multiplication under the direct
quadratic lift is `9.49%`, modular input reduction is `7.10%`, and the recursively continued lift
still contributes another `20.76%`. The next isolated experiment should replace the quotient
identities with FLINT-like dual-remainder corrections already used by Symbolica's dense product
tree. That can remove quotient reconstruction and the `q*u` multiply/add without changing the
general polynomial or `Ring` interfaces. Only after measuring that change should a private cached
large-modulus reciprocal be considered.

## Previous continuation checkpoint: cached reciprocal DDF

The accepted implementation and follow-up chain is:

| Commit | Change |
|---|---|
| `9e70064` | Reduce dense DDF products with a cached reversed-modulus reciprocal and retained buffers |
| `e58fc62` | Test reciprocal reduction when the recovered quotient begins with zero coefficients |
| `3f24ebd` | Widen PolyBench benchmark exponents from `u8` to `u16` |
| `9f9e690` | Retry a nonunit quadratic Hensel lift with the linear strategy |
| `b2e5d28` | Resolve resultant elimination variables in the benchmark polynomial namespace |
| `d548756` | Document and checksum the accepted cached-reciprocal benchmark inventory |
| `79bcd16` | Keep DDF termination bounds in `usize` at exponent-type boundaries |
| `687f46a` | Bound heap multiplication result preallocation by coefficient-pair storage |

All commits have author and committer `Ben Ruijl <ben@ruijl.ch>`. The production DDF change is
private to `DenseZpDistinctDegreeContext`; it does not add another method to `Ring`. The context
caches the inverse of the reversed monic modulus, retains `u64`/`u128` convolution accumulators and
quotient/low-product buffers, uses truncated dense products to recover the quotient, and refreshes
the reciprocal after a GCD shrinks the modulus. It selects direct Montgomery, native-remainder, or
wide-remainder accumulation from an exact coefficient bound. The classical monic remainder remains
available for unsupported cases and for reductions after a modulus change.

The reciprocal path and degree-bound correction pass all `66/66` factor-module tests under the
default GMP feature set and all `66/66` under `integer-malachite,float-astro,native_code_generation`; `cargo fmt --check`
is clean. Tests cover all
accumulation modes, products and squares, modulus shrinkage, and a quotient such as `q=x`, whose
constant coefficient must remain represented while performing reversed division.

Widening the PolyBench exponent representation exposed three previously hidden benchmark failures:
factor cases #131, #84, and #159 could reach a final quadratic Hensel correction with a leading
coefficient divisible by the base prime. Before entering that branch, the factorizer now verifies
that both modular leading coefficients are units and restarts the same lift with the linear
strategy otherwise. The complete widened PolyBench factor set now finishes and reconstructs all
`11/11` inputs. This is a containment fallback; a future cleanup could instead bound and reduce the
quadratic corrections before their degree can grow.

### Accepted degree-64 result and profile

Six alternating full-LTO processes, with 100 paired samples per backend and process, give:

| Version | Symbolica median | FLINT median | S/F |
|---|---:|---:|---:|
| cached reciprocal candidate | `4.0885525 ms` | `2.554542 ms` | `1.601166` |
| preceding monomial/DDF control | `5.065363 ms` | about `2.565 ms` | `1.9746855` |

The candidate wins all six paired blocks. The median paired improvement is `19.0815%`. A separate
500-sample profile run gives `4.114348 ms` versus `2.580807 ms`, ratio `1.594210`.

The hardware-cycle profiles normalize to the following approximate cycles per factorization:

| Stage | Previous Symbolica | Current Symbolica | Current FLINT | Current S/F |
|---|---:|---:|---:|---:|
| complete factorization | `22.10 M` | `18.22 M` | `11.35 M` | `1.61` |
| dense DDF | `8.08 M` | `4.05 M` | `1.81 M` | `2.24` |
| selected equal-degree factorization | `4.47 M` | `4.43 M` | `2.95 M` | `1.50` |
| integer Hensel reconstruction/tree | `9.28 M` | `9.24 M` | `5.76 M` | `1.60` |
| recombination | `0.231 M` | `0.236 M` | `0.194 M` | `1.22` |

The cached reciprocal cuts DDF by `49.9%`; after excluding its unchanged GCD calls, DDF modular
powering and reduction fall by `58.7%`. Overall sampled Symbolica work falls by `17.5%`, while the
FLINT control remains within `0.9%`. The profile ratio improves from `1.97x` to `1.61x`, agreeing
with the independent `1.601166x` wall-time result.

Of the remaining `6.88 M` cycle gap, Hensel contributes about `3.48 M` (`51%`), DDF `2.24 M`
(`33%`), and equal-degree factorization `1.48 M` (`22%`). These exceed 100% because miscellaneous
work is slightly faster in Symbolica. Current DDF contains about `1.85 M` cycles in
`multiply_low_into` and `1.20 M` in the polynomial GCD boundaries; reciprocal refresh costs only
`0.15 M`. Hensel is dominated by dense integer modular `remainder_monic` (`4.78 M`) and
`multiply_raw` (`2.60 M`). Equal-degree factorization spends about `3.20 M`, or `72%`, in generic
`quot_rem_univariate_fast`. FLINT's DDF profile includes Brent-Kung modular composition; Symbolica
still advances largely by classical Frobenius steps.

### Historical benchmark inventory at `41302eb` (superseded)

The tables in this subsection preserve the checkpoint before the balanced-Hensel and in-place
heap-advance changes. Do not use them as the current scoreboard. The complete refreshed inventory,
including every replacement and a mandatory update protocol, is
[CURRENT_STATUS.md](CURRENT_STATUS.md).

Ratios below are Symbolica median divided by FLINT median, so lower is better. The retained base
rows use the full-LTO/default-feature binary at `b2e5d28`, with `faster_alloc`, GMP, FLINT 3.6.0,
and one thread per implementation. The 23 current PolyBench construction rows come from the six
100-sample system-linked `268a353` runs, with #84's product replaced by its twelve-process,
1,000-sample rerun. The generated degree-63, degree-64, degree-65, and high-height degree-33 factor
rows come from `d50fe00`; PolyBench #84's current operation comes from `41302eb`. High-height
product and other retained delayed-reduction rows use `3436c42`. This is therefore a mixed
system-linked and bundled-FLINT current inventory; source-matched A/B runs provide the attribution
for individual changes. The `*-s3.csv` family scans use three paired samples after warmup; robust
repeated rows
replace their exploratory counterparts. Each configured operation or independently requested
product is counted once. For resultants, the aggregate uses the main Ducos entry point; Brown and
CRT are alternative runs reported separately.

| Family | n | Best S/F and case | Median S/F | Worst S/F and case |
|---|---:|---|---:|---|
| all strict current rows | 122 | `0.028678`, high-gap 8v GCD d4/gap256 | `0.92545225` | `2.0059855`, high-height factor d33 |
| non-PolyBench rows | 76 | `0.028678`, high-gap 8v GCD | `0.892691` | `2.0059855`, high-height univariate factor d33 |
| PolyBench products plus operations | 46 | `0.100709`, factor #44 | `0.93441625` | `1.7405235`, factor #131 |
| PolyBench operations only | 23 | `0.100709`, factor #44 | `0.748107` | `1.7405235`, factor #131 |
| PolyBench construction products | 23 | `0.9006325`, factor #163 product | `0.942414` | `1.2681525`, factor #84 product |
| PolyBench GCD operations | 12 | `0.387980`, #53 | `0.6583255` | `1.268240`, #140 |
| PolyBench GCD products | 12 | `0.9018375`, uniform #55 | `0.94212425` | `1.100743`, sharp #53 |
| PolyBench factor operations | 11 | `0.100709`, #44 | `0.839857` | `1.7405235`, #131 |
| PolyBench factor products | 11 | `0.9006325`, #163 | `0.942414` | `1.2681525`, #84 |
| integer multiplication | 8 | `0.449493`, dense very-large | `0.9351165` | `1.348957`, 7v power-minus-one |
| finite-field multiplication | 14 | `0.252283`, near-2^64 dense univariate d4912 | `0.5908955` | `1.137177`, near-2^64 dense-large |
| all multiplication | 22 | `0.252283`, near-2^64 univariate | `0.7449595` | `1.348957`, 7v power-minus-one |
| exact integer polynomial division | 3 | `0.441603`, dense | `0.442621` | `0.761777`, high-height |
| main Ducos resultant | 6 | `0.519856`, dense outer d7/6 | `0.7161895` | `1.027247`, outer-sparse d12/9 |
| Brown resultant alternative | 6 | `1.124006`, nonunit-leading d9/7 | `1.650471` | `2.581685`, high-height d14/10 |
| CRT resultant alternative | 6 | `0.432838`, outer-sparse d12/9 | `1.189242` | `2.287301`, lacunary d18/11 |
| generated GCD operations | 14 | `0.028678`, high-gap 8v | `0.4536895` | `1.060731`, dense 2v |
| generated GCD products | 14 | `0.291403`, dense 2v | `1.356885` | `1.917285`, high-gap 5v |
| all GCD operations | 30 | `0.028678`, high-gap 8v | `0.613736` | `1.268240`, PolyBench #140 |
| generated factor operations | 6 | `0.575899`, dense 2v | `1.3653245` | `2.0059855`, high-height d33 |
| generated factor products | 6 | `0.578774`, high-height d33 | `0.9919445` | `1.184942`, univariate d65 |
| all factor operations | 17 | `0.100709`, PolyBench #44 | `1.0072955` | `2.0059855`, high-height d33 |
| all construction products | 44 | `0.291403`, generated dense 2v GCD | `1.02256525` | `1.917285`, generated high-gap 5v GCD |

The overall best is generated high-gap GCD with 8 variables, degree 4, and gap 256 (`0.028678x`).
The overall worst is generated high-height univariate degree-33 factorization (`2.0059855x`). The
heterogeneous overall median is descriptive rather than workload weighted. The important
PolyBench split is that both the median core GCD/factor operation and the median construction
product now beat FLINT; their combined median is `0.93441625x`.

More detailed family medians are:

- integer multiplication `0.9351165`; finite-field multiplication `0.5908955`;
- generated GCD operation `0.4536895`, versus its product construction `1.356885`;
- generated factorization `1.3653245` after refreshing degrees 63--65 and high-height degree 33,
  versus its product construction `0.9919445`;
- PolyBench GCD operation `0.6583255`, versus its product construction `0.94212425`;
- PolyBench factorization operation `0.839857`, versus its product construction `0.942414`.

The fresh configured dense univariate GCD sweep uses five 200-sample processes per degree:

| Degree | Symbolica median | FLINT median | Median S/F field |
|---:|---:|---:|---:|
| 48 | `0.254048 ms` | `0.237013 ms` | `1.072266` |
| 64 | `0.418056 ms` | `0.354467 ms` | `1.181340` |
| 80 | `0.728479 ms` | `0.590276 ms` | `1.234835` |

Coefficient-height coverage includes 5-variable GCD products and operations at 128, 256, 512,
and 1024 bits plus an 8-variable 256-bit case. Product ratios range from `1.012638` to `1.499961`
with median `1.056738`; GCD ratios range from `0.429881` to `0.472101` with median `0.437142`.
The generated high-height factor product is `0.578774`, while factorization is `2.0059855`;
high-height exact division is `0.761777`, and the main high-height Ducos resultant is `0.861806`.

### Artifacts and next experiment

The accepted delayed-reduction artifacts are:

| Artifact | SHA-256 |
|---|---|
| `/tmp/flint-comparison-hensel-quotrem-3436c42-lto` | `4dc16b559a537f7b9640c668162a7b5dfb325814674c638627a4521b70560069` |
| `/tmp/flint-comparison-hensel-quotrem-3436c42-lto.d` | `812b0a38e559b0198733ed19ef968135ef60f0a99567991e7a99b6a8000f561a` |
| `/tmp/flint-comparison-hensel-quotrem-3436c42-build.jsonl` | `b786951333ea1af2c8b354520cc2b61656bf9059b126937a8dfa5b518db94fe6` |
| `/tmp/hensel-quotrem-3436c42-artifact-manifest-sha256.txt` | `942cf9db97c873e6883943ddf883a54e1e13f2d1df8bb699f336133374913918` |
| `/tmp/profile-hensel-quotrem-3436c42-highheight-lbr.perf.data` | `b2da99d9a05d2581d30afa8f7c982f22320c8cce366d54f0a30ee7d8029ff7c8` |
| `/tmp/profile-hensel-quotrem-3436c42-highheight.csv` | `e9d212b0532f653f6698f4965a4d4762bd1add1e0b924f1fa6ebc9a9f70fe7c6` |
| `/tmp/profile-hensel-quotrem-3436c42-highheight-children-symbols.txt` | `f04c0f795c1597056db80101c7f09423c4cd69ef9b343658906c3f0498970de7` |
| `/tmp/profile-hensel-quotrem-3436c42-highheight-flat-symbols.txt` | `ff3e26cdd696e84a8b513b35a67f2dbf6d034f69fef65ce0e3b3ca5853a1a27b` |

The build took 9m26s. The manifest covers 179 raw timing and summary artifacts. The frozen binary
identifies source commit `3436c42`; the same content is integrated on `dev` as `11c878b`.

The preceding DDF release binary is `/tmp/flint-comparison-ddf-preinverse-final-lto`, SHA-256
`ef3cb4d8177817f1db2b2d236e701c3339704c2001a95e2945b65d6e0dc1c654`. Its build log is
`/tmp/flint-comparison-ddf-preinverse-final-lto-build.jsonl`, SHA-256
`5fcb0d4a87353f248622f6d6b23004f020cba9214fee1d4aa620e86d9824db90`. The complete 42-file
artifact manifest is `/tmp/ddf-preinverse-final-artifacts-sha256.txt`, whose SHA-256 is
`bdceacc3efbd3806ff4fa951e927285e514ce335463f8a015d6ea052ebc0de5e`.

The accepted profile artifacts are:

| Artifact | SHA-256 |
|---|---|
| `/tmp/profile-factor-ddf-preinverse-final-d64-lbr.perf.data` | `4a7e68a4d161aabc20578c36340a4e9bf0f59fdad7f5becb313856489acc90f5` |
| `/tmp/profile-factor-ddf-preinverse-final-d64.children-symbols.txt` | `50ae71d62eed02aa889baab4c614506d0b5714bfb8aa7e200edd9d2f0423fc27` |
| `/tmp/profile-factor-ddf-preinverse-final-d64.flat-symbols.txt` | `d01763ddb411d425ff349bac56b090ee406f3afcc33eeedbf0731364ace73945` |

### Rejected dense baby-step/giant-step DDF experiment

The first guarded Kaltofen-Shoup-style DDF implementation is preserved in
`/tmp/symbolica-ddf-bsgs`, branch `codex/ddf-bsgs`, through `0acfcf9`; it is deliberately not on
`dev`. Its five commits are `96439eb`, `1aa43aa`, `ae07efb`, `c868471`, and `0acfcf9`, all authored
and committed by `Ben Ruijl <ben@ruijl.ch>`.

The private dense context keeps a degree-one/two classical prefix, constructs baby and giant
Frobenius steps, takes one coarse GCD per interval, refines nonempty intervals, and shrinks the
modulus after extraction. A cost guard requires a residual degree of at least 48 and rejects
unprofitable field/layout regimes. Dedicated tests compare it with classical DDF across the direct,
native, and wide accumulator modes, partial final blocks, and degree boundaries 48, 49, 63, 64,
65, 254, and 255. All factor-module tests pass in the candidate worktree.

Six alternating processes with 100 samples per backend and degree give:

| Degree | Candidate Symbolica | Candidate FLINT | Candidate S/F | Control S/F | Paired candidate change |
|---:|---:|---:|---:|---:|---:|
| 63 | `5.0264295 ms` | `2.9972965 ms` | `1.680326` | `1.797218` | `-7.211982%` |
| 64 | `3.9769745 ms` | `2.559878 ms` | `1.554367` | `1.600751` | `-2.641298%` |
| 65 | `4.895736 ms` | `3.026234 ms` | `1.616546` | `1.631055` | `-1.021248%` |

The representative degree-64 gain is below the standing 3% acceptance threshold. The PolyBench
guard processes also show that merely adding the approximately 800-line implementation changes
full-LTO layout enough to regress unaffected cases:

| Guard | Maximum product degree | Candidate S/F | Control S/F | Paired candidate change |
|---|---:|---:|---:|---:|
| factor #105 | 19 | `1.122494` | `1.1374615` | `-1.286135%` |
| factor #178 | 29 | `3.1852305` | `3.088689` | `+3.239243%` |
| factor #176 | 33 | `9.8575035` | `9.6955485` | `+1.717813%` |

None of those guards reaches the degree-48 threshold, so their movement is not time spent in the
BSGS algorithm. This combination of a sub-threshold representative gain and guard regressions is
why the candidate was rejected.

The profile explains the limited ceiling. DDF sampled work drops from about `2.025 B` to `1.972 B`
cycles, only `2.6%`. Its GCD share falls from `3.95%` to about `2.36%`, but the flat share of
`multiply_low_into` rises from `6.10%` to `7.75%`; batching replaces GCD boundaries with modular
composition and interval products that use the same schoolbook dense kernel. The classical path
also terminates near distinct degree 15 on this fixture rather than running to degree 32, shrinks
its modulus as factors are extracted, and benefits from cheaper squarings. A simple abstract
product/GCD count therefore overestimated the saving.

The final candidate already fixes two important structural issues discovered during review: it
uses O(1) leading-degree checks and shrinks the BSGS modulus after fine refinement. A future retry
should first provide vector baby-step modular composition or a faster dense finite-field `mulmod`;
threshold tuning alone is unlikely to make this implementation worthwhile. The independent O(1),
overflow-safe degree-bound part was retained on `dev` as `79bcd16`.

The final candidate binary is `/tmp/flint-comparison-ddf-bsgs-v3-lto`, SHA-256
`2618e69d5086e90f9f4acb3c1d05a2fb85b81c28129e104fc2b09d1fed9ede92`. Its build log SHA-256 is
`2006fffdea8584dc7bd057cdb8f84575cec5a3f2bb6c9571372a6061fbe93fc5`. The 79-file candidate,
control, guard, and profile manifest is `/tmp/ddf-bsgs-v3-artifacts-sha256.txt`, SHA-256
`99456b1512f161e776677173e8b5a24f47b67c483d0e3c83d59cfb2580639ef1`.

### Rejected cache-aware Kronecker allocation experiment

The first follow-up product experiment is preserved in `/tmp/symbolica-onevar-product`, branch
`codex/onevar-product`, at `73e846e` (`Reuse cached integers in Kronecker conversion`); it is
deliberately not on `dev`. The commit obtains packed and decoded GMP integers through the bounded
`MultiPrecisionInteger` cache, assigns little-endian limbs into the reused allocation, and computes
the packed product through the existing borrowed multiplication implementation. This covers three
direct raw-GMP constructions that bypassed the cache. All seven focused Kronecker tests pass.

Six alternating full-LTO processes used 10,000 paired samples per backend on the degree-64 product
fixture. Process medians are:

| Version | Symbolica median | FLINT median | Median process S/F | Paired Symbolica change |
|---|---:|---:|---:|---:|
| cache-aware candidate | `0.005853 ms` | `0.0053875 ms` | `1.085005` | `-0.97495%` |
| accepted control | `0.005874 ms` | `0.005396 ms` | `1.0946575` | — |

The candidate is faster in four of six paired blocks, but its approximately `1.0%` median gain is
below the standing `3%` threshold and too small to distinguish confidently from whole-program-LTO
layout and sub-microsecond timing effects. The extra conversion API and call-site changes are
therefore rejected. An independent six-process control run gave `1.092838x`, consistent with the
alternating control and replacing the exploratory three-sample `1.138626x` degree-64 product row as
the robust current estimate.

The frozen candidate binary is `/tmp/flint-comparison-product-cache-73e846e-lto`, SHA-256
`2b2b118d48f13a6d3fee961258625832f5d3c14395c81da061eadbf7dfcd07b1`. The 15-file binary,
build-log, and timing manifest is `/tmp/product-cache-73e846e-artifacts-sha256.txt`, SHA-256
`08ca3c71ddbee25f56bfb276054c7c4dd8f8b3dfdf1c11cf01ebc7b995c9fa78`.

### Rejected single-variable dense-layout experiment

The polynomial-layer follow-up is preserved in `/tmp/symbolica-onevar-layout`, branch
`codex/onevar-layout`, at `cf19544` (`Streamline single-variable dense multiplication`); it is
deliberately not on `dev`. For a variable map of length one, it uses the last Lex-ordered exponent
as the degree, copies the stored exponents directly into dense-kernel indices, and reconstructs
output exponents directly from the kernel positions. It retains the existing polynomial-validity
gate and all general multivariate behavior. Its focused regression test covers offset support,
coefficient cancellation, zero operands, and `u8` outputs through degree 254.

Six alternating full-LTO processes used 10,000 paired samples per backend on the degree-64 product:

| Version | Symbolica median | FLINT median | Median process S/F | Paired Symbolica change |
|---|---:|---:|---:|---:|
| single-variable candidate | `0.0058415 ms` | `0.0053875 ms` | `1.083738` | `-0.834978%` |
| accepted control | `0.0058675 ms` | `0.005403 ms` | `1.0878695` | — |

The candidate is faster in four of six paired blocks, but the gain is again below `1%` and the
`3%` acceptance threshold. The profile explains the low ceiling: one-variable
`advance_uni_var` normally takes its no-carry branch and executes no division, while index-vector
allocation and the integer kernel remain unchanged. The candidate is therefore rejected.

The frozen candidate binary is `/tmp/flint-comparison-onevar-layout-cf19544-lto`, SHA-256
`f11bca7dc4578f756d53b57597498f297f389b504a1e87b164f820ff5c723077`. The 15-file manifest is
`/tmp/onevar-layout-cf19544-artifacts-sha256.txt`, SHA-256
`84332b4f0de6d04ccafb7b8a7662aad560af79438324b7a53be6c0e8406c522a`.

### PolyBench multiplication route audit

All 35 constituent PolyBench products reject both dense layouts and use packed-`u8` heap
multiplication. Their mixed-radix boxes contain `224,532` to `6,815,313,600` slots, with box to
coefficient-pair ratios from `134` to about `3.23 million`; every ratio exceeds the current limit
of `64`. Total-degree simplex sizes range from `7,624,512` to `7,392,009,768`, all above the
`1,048,576` cap. Pair-product to output-term ratios are only `1.000` to `1.058`, so these products
are genuinely sparse and relaxing the dense guard is not a broad solution. Raising its relative
limit from `64` to `256` would change only the two products in 5-variable sharp GCD #11.

### Accepted bounded heap-result preallocation

Commit `687f46a` (`Bound heap multiplication result preallocation`) is integrated on `dev`. Its
worktree source commit is `06a51ea`. Both packed and generic heap multiplication now use the number
of coefficient pairs as an upper bound for result capacity when the coefficient and exponent
buffers together require at most one mebibyte. Larger products retain the previous input-sized
reservation. The PolyBench inputs previously reserved only `35` to `50` result terms and returned
about `1,287` to `2,450`; their pair-product upper bounds are close because output collision factors
are only `1.000` to `1.058`.

Six alternating full-LTO processes used 100 samples per backend for every reported PolyBench
construction row:

| Comparison | Preallocation S/F | Input-sized control S/F | Median paired gain | Rows won |
|---|---:|---:|---:|---:|
| production candidate versus frozen control | `1.795966` | `1.8365995` | `2.151677%` | `23/23` |
| same binary, benchmark switch enabled versus disabled | `1.817915` | `1.849358` | `1.868432%` | `23/23` |

The production candidate's best ratio is `1.4934895x` on factor product #178 and its worst is
`1.8567915x` on factor product #32. In the same-binary causal comparison, GCD products improve by a
median `2.034138%` and factor products by `1.762653%`; the best row improves by `3.480891%` (#53)
and the smallest gain is still `1.243282%` (#32). This uniform same-executable result is why the
small 34-line change is accepted despite its median being below the usual `3%` threshold.

A separate three-block scan of all 22 dedicated multiplication rows moved unchanged paths between
about `-12%` and `+10%` across the two independently linked binaries while FLINT stayed stable.
Those shifts are whole-program-LTO layout effects, not capacity-policy work. The experiment-only
commit `3e88257` therefore added a `OnceLock` benchmark switch and was used to obtain the
same-binary comparison above; that switch is not integrated.

The default-feature polynomial module passes `28/28` tests and the focused
`integer-malachite,float-astro,native_code_generation` capacity test passes. The production binary is
`/tmp/flint-comparison-polybench-prealloc-06a51ea-lto`, SHA-256
`0b53b850d29246c0966ae848433eb0e5f396556465af0665f18e88537fdf9d24`. The same-binary control is
`/tmp/flint-comparison-polybench-prealloc-toggle-3e88257-lto`, SHA-256
`e139ebf3cb6de0ad50e36a29e202024b8eb839632107110038793d79db9e9e1e`. The complete 63-file
manifest is `/tmp/polybench-prealloc-final-artifacts-sha256.txt`, SHA-256
`5e8d266bee4ddf8f24d474a4d7b744161c0334f116647173c455366e108506d1`.

### Rejected one-scan polynomial multiplication context

The completed experiment is preserved in `/tmp/symbolica-mul-context`, branch
`codex/mul-context`, at `e7d4894` (`Reuse multiplication support metadata`); it is deliberately not
on `dev`. The private `PolynomialMulContext` holds both operands and scans each support once. It
retains per-variable minima and maxima inline through eight variables, polynomial/Laurent status,
and maximum total degrees. The context then coordinates mixed-radix dense, total-degree dense,
packed-heap, and generic-heap dispatch without adding a `Ring` or kernel method.

The extrema also make exponent overflow handling route-independent. Both the lowest and highest
possible coordinate sums are checked before multiplication, catching unsigned overflow and signed
Laurent underflow. This exposed an existing correctness issue: the generic heap uses unchecked
exponent addition, so release builds can wrap, and a packed route can likewise select storage wider
than the polynomial exponent type. Because the context was rejected for performance, that
correctness fix is not on `dev`; a future standalone fix should preserve the established
`overflow in adding exponents` panic without adding per-product checks to the hot heap loop.

The default polynomial module passes `32/32` tests. New independent-reference tests cover Laurent
products, `u8` and signed `i8` overflow, packed `u8`/`u16` boundaries, generic fallback, asymmetric
heap operand ordering, and public total-degree-simplex dispatch. Default and explicit `integer-malachite`
library checks are clean. The no-default-feature test target itself assumes native-code-generation
APIs in an unrelated evaluation test module, so the no-GMP guard used `cargo check --lib`.

Six alternating full-LTO processes used 100 paired samples for all 23 PolyBench construction rows:

| Slice | Candidate S/F | Control S/F | Paired candidate change | Rows won |
|---|---:|---:|---:|---:|
| all PolyBench products | `1.8036085` | `1.8026630` | `+0.238853%` | `4/23` |
| PolyBench GCD products | `1.78939225` | `1.78079275` | `+0.264074%` | — |
| PolyBench factor products | `1.8036085` | `1.8066765` | `+0.201811%` | — |

The best paired movement is only `-0.242748%` on factor product #44. The worst is `+3.141571%`
on factor product #178. A separate six-process, 10,000-sample degree-64 univariate product guard
regresses by `2.267363%`: candidate Symbolica/FLINT is `1.124837`, versus `1.0939175` for the
accepted control. A three-process scan of all 22 dedicated multiplication rows has a median paired
regression of `0.959433%`, with the same large independently linked LTO-layout shifts already seen
in the preallocation experiment. There is no systematic gain to justify the larger dispatch
refactor or the extra eager total-degree/extrema work, so the candidate is rejected.

The frozen candidate binary is `/tmp/flint-comparison-mul-context-e7d4894-lto`, SHA-256
`c75ae629c02572a0b8ea3642a8846b682b64896f6cc42ae34d1144be7b93f1e5`. Its dependency sidecar
SHA-256 is `06db120092629360016cc5203099ff2fbe3c2635824dc5953e2cf2d3e5150d33`, and its build-log
SHA-256 is `01be9d0586e90c2f682e8af47caa1ebe93133d98d5f1aed40b2c122b9db665df`. The 49-file manifest is
`/tmp/mul-context-e7d4894-artifacts-sha256.txt`, SHA-256
`92199c00901a85077437086b259b042d03674dcff9844acf645311c4ecb65fad`.

The high-height Hensel factorization remains the next factor target. Its strongest known structural
gap is `IntegerModularUnivariateContext::quot_rem`: Symbolica reduces every inner subtraction and
multiplication, while FLINT's corresponding dense modular division delays more reductions around
the pivot and final remainder. Continue using the single-core paired-process methodology and the
`3%` complexity threshold, with same-binary controls when expected gains are small.

## Previous checkpoint: dense degree-64 modular factorization

This is the immediately preceding checkpoint, retained for provenance. Its measured chain is
integrated on `dev` through documentation commit `2fb7fac`:

| Commit | Change | Source-matched degree-64 result |
|---|---|---:|
| `abd0add` | Call the direct univariate GCD path during modular screening and DDF | `-10.064%` |
| `052665d` | Keep bounded classical DDF multiplication and remainder state in dense `Zp` storage | `-9.696%` |
| `7b849e3` | Extract exact univariate monomial factors before screening and reject zero-constant images | `-2.435850%` |

All three source commits have author and committer `Ben Ruijl <ben@ruijl.ch>`. The measured source
head is `7b849e3`; `2fb7fac` adds this checkpoint without changing the executable. The combined
default-feature factor suite passes `63/63`; the no-GMP suite also passes `63/63` on its complete
rerun. One earlier no-GMP
`galois_upgrade` execution failed during randomized finite-field splitting, but its immediate
isolated rerun and the subsequent complete suite both passed.

Six alternating 100-sample full-LTO processes compare the combined binary with the dense-DDF-only
binary. The combined median is `5.0231025 ms` versus `5.1552440 ms`; all six blocks win and the
median paired change is `-2.435850%`. FLINT's median is `2.562000 ms`, so the current ratio is
`1.9578535`. The original source-matched degree-64 baseline was `11.336475 ms`, and the accepted
dense Hensel tree was `6.511109 ms`; the current chain has therefore removed about `55.69%` of the
original time and another `22.85%` after the dense tree.

The exact monomial extraction changes the production fixture from a degree-64 modular target with
a modular `x` leaf to a degree-63 cofactor. Suitable-prime factor counts are now `8`, `7`, and `6`
for `p=7,13,17`; the bounded large-prime count is `19`. Prime 11 is rejected before mapping because
the exact cofactor's constant is zero modulo 11. The selected `p=17` modular leaf degrees are
`[1,2,10,10,10,30]`; the exact `x` is returned separately.

Six-process guard measurements against the accepted dense-tree binary are:

| Case | Current | Previous | Paired change | Current S/F |
|---|---:|---:|---:|---:|
| degree 63 | `7.070403 ms` | `12.091458 ms` | `-41.345%` | `2.335` |
| degree 64 | `5.080938 ms` | `6.401443 ms` | `-20.786%` | `1.960` |
| degree 65 | `5.879263 ms` | `18.092294 ms` | `-67.534%` | `1.925` |
| high-height degree 33 | `5.044321 ms` | `5.685261 ms` | `-11.399%` | `3.590` |
| generated factor, 2 variables | `5.756274 ms` | `6.094812 ms` | `-5.272%` | `0.584` |
| generated factor, 3 variables | `7.527340 ms` | `8.240099 ms` | `-8.997%` | `1.786` |
| PolyBench #105 | `31.412176 ms` | `31.839948 ms` | `-1.237%` | `1.082` |
| PolyBench #178 | `41.619279 ms` | `41.809230 ms` | `-0.537%` | `2.966` |
| degree-64 input product | `0.0059515 ms` | `0.0060130 ms` | `-0.860%` | `1.086` |
| degree-64 GCD, 12 blocks | `0.4172350 ms` | `0.4130085 ms` | `+1.032775%` | `1.183` |

The standalone GCD slowdown is reproducible but the factor-only changes do not enter that execution
path; retain it as a whole-program-LTO layout guard and revisit it during the one-variable GCD
phase rather than attributing it to DDF.

A fresh 500-call LBR profile of the combined binary measures `5.025657 ms` versus FLINT
`2.586184 ms`. Normalized sampled cycles are approximately:

| Stage | Symbolica | FLINT |
|---|---:|---:|
| complete factorization | `22.10 M` | `11.25 M` |
| Hensel product tree | `9.28 M` | `5.86 M` |
| modular prime screening | `8.29 M` | included below |
| dense classical DDF | `8.08 M` | `1.77 M` |
| DDF classical monic remainder | `4.03 M` | — |
| DDF dense convolution | `2.56 M` | — |
| DDF univariate GCD boundaries | `1.20 M` | — |
| selected equal-degree factorization | `4.47 M` | about `2.72 M` |

This profile motivated the private cached reverse-modulus inverse now documented in the current
checkpoint above. Its acceptance confirms that Kaltofen-Shoup/BSGS DDF is the next useful
algorithmic experiment: FLINT's iteration count remains much lower.

Current artifact provenance:

| Artifact | SHA-256 |
|---|---|
| `/tmp/flint-comparison-dense-ddf-monomial-lto` | `0d000c3fcfebd00c089e7670ce33b0609a59027f43b25827f7d44c42b3d60e2b` |
| `/tmp/flint-comparison-dense-ddf-monomial-lto.d` | `110205916d29fe6bbf11e8c89daedadc1288c9c27e9dce256739ee4c3b62a9ba` |
| `/tmp/flint-comparison-dense-ddf-monomial-lto-build.jsonl` | `b70c90903f005bfb13245682f5ebf92777dfdcba53662ed5c5690cea92e9741c` |
| `/tmp/monomial-d64-sha256.txt` | `5c6898e41a596ba3009aa66428a618424ef3e9b894de368b7d501e58e70733e9` |
| `/tmp/monomial-generated-guards-sha256.txt` | `dcf57aeef087649a072e60be94da5bbc7f5f5886a9de38751a0a8a4ff857cc7f` |
| `/tmp/monomial-polybench-sha256.txt` | `05faf0c6720b35415c39a114b96d8bbb19bd02a80edabe7bd7c615b1d3ce52eb` |
| `/tmp/monomial-product-gcd-sha256.txt` | `90e17519c0117beac57c77812136930b1a4f8e4ec2fa7849f11bd41d048c9838` |
| `/tmp/monomial-gcd64-sha256.txt` | `83ec184b510692a69b39fea2af1910fdbd4968bbd6e7260f40b5e9d335ecd61d` |
| `/tmp/profile-factor-dense-ddf-monomial-d64-lbr.perf.data` | `cc7bc63a4f632fb24b44b984267bd615acc38189c80b7373bec971a91a94ba7e` |
| `/tmp/profile-factor-dense-ddf-monomial-d64.children-symbols.txt` | `b6945903a2a739f479da95beeed1763c280ca1cbebd7050ac86c52f721201954` |

## Historical checkpoint: synchronized Hensel product tree

The accepted product and factor candidates are integrated on `dev` through merge `3184631`. Do not
cherry-pick their old worktree hashes again. The pre-tree integrated chain is:

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

Merge `3184631` retains the tested product-tree source commits `b081091` through `6388bd3`; the
complete list and measurements are below. Documentation commit `718a4b7` replaced the provisional
tree status after integration.

The accepted full-LTO tree binary is
`/tmp/flint-comparison-factor-product-tree-dense-final-lto`, frozen from `6388bd3`. Twelve
100-sample processes give `6.511109 ms` versus FLINT `2.620505 ms`, ratio `2.485136`. The historical
product-tree v1 rerun is `7.949879 ms`, ratio `3.030744`; the final candidate wins all twelve blocks
with a median paired improvement of `18.229%`. The source-matched local-bound baseline rerun is
`11.336475 ms`, ratio `4.325662`, so the integrated tree removes `42.665%` of that time.

Residual and local-subtree validation completed before integration:

- root factor module under default features `48/48`;
- root factor module under `integer-malachite,float-astro,native_code_generation` `48/48`;
- base-prime precision, binary-prime multi-round lifting, and non-monic nontrivial-`gamma` cases;
- the exact scaled-residual invariant after every debug linear-lift round;
- `cargo check --all-targets`, `cargo fmt --check`, and `git diff --check`.

The combined local-subtree source passed `50/50` factor tests under both GMP and `integer-malachite` before
the final LLL test was added. That LLL test then passed separately under both backends; it is the
only change after the release build.

### Accepted synchronized Hensel product tree

The complete tested source chain is retained on `dev` by merge `3184631` and remains available at
`codex/product-tree-dense-combined`, commit `6388bd3`:

| Commit | Change |
|---|---|
| `b081091` | Build a degree-greedy Hensel product-tree topology |
| `2e21a3c` | Add modular remainder by a monic polynomial |
| `3c50b09` | Lift all factors through synchronized Hensel precision stages |
| `af06d09` | Keep product-tree arithmetic in a dense integer context |
| `44c968b` | Recheck Hensel pressure after final prime selection and EDF |
| `e288f35` | Reuse owned GMP storage for large-integer remainders |
| `d4c6742` | Use `exact_div_owned` for exact dense Hensel residual quotients |
| `738967d` | Retain nonnegative residues and reuse dead lift buffers |
| `6388bd3` | Hide test-only Hensel topology helpers |

The tree combines the two lowest-degree nodes at each step. After the exact linear factor has been
removed from the degree-64 fixture, its modular leaf degrees `[1,2,10,10,10,30]` produce
degree-weighted internal work `132`, versus `191` for the earlier count-midpoint tree. A precision
schedule such as `1,2,3,5,10,20,39,77` walks every
internal node top-down at each stage, updating factors and Bezout cofactors together; the final
unused inverse update is skipped. Global recombination still uses the existing exact reconstruction
tail, and unsupported or lower-pressure cases retain the binary/local-bound fallback.

Tree dispatch is now decided only after the final modular prime and complete equal-degree
factorization are known. The shared pressure predicate requires degree at most 64, a coefficient
bound of at least 256 bits, at least three factors, at least 64 lifting digits, and estimated work at
least 256. The degree-63 fixture ultimately selects `p=65_000_011` and needs only 12 digits, so it
rejects the tree. The degree-64 fixture selects `p=17` and needs 77 digits, so it uses the tree. This
fixes the v1 mistake of dispatching from the initial small-prime estimate. Tree dispatch also
requires more than four leaves, preserving the existing quadratic/binary path for smaller sets.

`DenseIntegerModularUnivariateContext` stores leaves, internal products, and Bezout cofactors as
trimmed `Vec<Integer>` values. It caches dense indices, calls the integer dense multiplication
kernel when admitted, retains a generic quadratic fallback, implements owned monic remainder
directly, and converts to sparse polynomials only at the boundary. Guaranteed exact product and
Bezout residuals use `exact_div_owned` rather than generic division.

All intermediate residues now use canonical representatives in `[0,m)`. When advancing from scale
`s` to modulus `M`, `old` is in `[0,s)` and the correction `delta` is in `[0,M/s)`, so
`old + s*delta` is already in `[0,M)` and needs no modular reduction. Only the final lifted leaves
are symmetrized. Owned child, cofactor, remainder, and product buffers are reused after their old
values become dead, while the normalized root target is borrowed across stages.

The GMP-specific owned-remainder change avoids cloning both large operands when evaluating an owned
large numerator modulo a borrowed large divisor. In 12 alternating one-million-operation release
microbench processes, median time fell by `32.529%` at 156 bits and `27.705%` at 312 bits. Raw data
is `/tmp/owned-remainder-v1-alternating.csv`, SHA-256
`5d2a3d3926c6bfbcbb5cb4608a2dac242ebb759f4420a76a8bd4be95246e9a01`; the no-GMP backend retains
its safe generic fallback.

The complete factor-module suite passes `59/59` under default GMP features and `59/59` under
`integer-malachite,float-astro,native_code_generation`. Coverage includes small primes, composite prime powers, uneven
degrees, nonmonic targets, interior zeros, GMP-sized coefficients, product congruence, Bezout
identities, exact residuals, and agreement with the binary lift.

FLINT 3.6.0 source was inspected at `/tmp/flint-3.6.0-source`, commit `8d5454b96`.
`hensel_build_tree.c` builds a degree-greedy tree; `hensel_lift_without_inverse.c` and
`hensel_lift_only_inverse.c` keep dense `fmpz_poly` arrays, use low-level modular multiplication and
remainder routines, and skip the final inverse lift. The integrated implementation now matches
those high-level topology and storage choices.

The historical product-tree v1 full-LTO binary is
`/tmp/flint-comparison-factor-product-tree-v1-screen`, SHA-256
`84eedbc43629bbd9896340a832e70937e6dd7a0c9708a08e8adcf94d158aea10`. Twelve alternating
100-sample processes reduced the degree-64 median from `11.083816 ms` to `7.805277 ms`, a paired
improvement of `29.606%`; its Symbolica/FLINT ratio was `3.032826`. Its broad selector regressed
degree 63 by `11.994%`, which is why it was not integrated unchanged. A 500-call LBR profile of v1
attributed `32.08%` of combined cycles to tree lifting, including `15.39%` in generic monic
remainder, `5.67%` in generic reduction, and `6.83%` in integer polynomial multiplication. Those
measurements motivated the dense context above.

The definitive full-LTO degree-64 matrix is:

| Version | Symbolica median | FLINT median | S/F | Change from final |
|---|---:|---:|---:|---:|
| accepted local-bound baseline rerun | `11.336475 ms` | `2.620589 ms` | `4.325662` | `+74.11%` |
| product-tree v1 rerun | `7.949879 ms` | `2.624387 ms` | `3.030744` | `+22.10%` |
| dense product-tree final | `6.511109 ms` | `2.620505 ms` | `2.485136` | baseline |

All twelve final/v1 blocks favor the final implementation; their median paired change is
`-18.229%`. The final/local-bound comparison also favors the final implementation in all twelve
source-matched blocks and changes by `-42.665%`. The earlier no-LTO `d4c6742` screen at
`7.622205 ms` was directionally correct but is not used as accepted evidence.

The degree-63 selector guard uses twelve alternating 100-sample processes. It reports final
`12.290517 ms`, FLINT `3.070392 ms`, ratio `4.001479`, versus baseline `12.405151 ms`, FLINT
`3.065324 ms`, ratio `4.051184`. The median paired change is `-1.092%`, with the final source
winning 9/12 blocks. Thus the final-prime selector restores the legacy path and removes v1's
systematic `11.994%` regression.

Six-process guards are final versus the local-bound baseline:

| Case | Final | Baseline | Paired change | Final S/F |
|---|---:|---:|---:|---:|
| high-height degree 33 | `5.782138 ms` | `6.400096 ms` | `-9.830%` | `4.055466` |
| degree 65 | `18.094752 ms` | `18.181150 ms` | `+0.299%` | `5.858365` |
| generated factor, 2 variables | `6.194100 ms` | `6.130743 ms` | `+1.205%` | `0.618901` |
| generated factor, 3 variables | `8.337006 ms` | `8.297922 ms` | `+0.849%` | `1.949331` |
| PolyBench #105 | `31.955848 ms` | `32.155360 ms` | `-0.671%` | `1.098488` |
| PolyBench #178 | `41.965888 ms` | `41.925534 ms` | `+0.074%` | `2.976358` |
| degree-64 input product | `0.006070 ms` | `0.005929 ms` | `+2.173%` | `1.103536` |
| degree-64 GCD | `0.413845 ms` | `0.408158 ms` | `+1.491%` | `1.171077` |

The product and GCD regressions were small but repeatable in this historical build. They did not use
the Hensel tree and were retained as whole-program code-generation guards. At this checkpoint the
broad PolyBench 8-variable sweep still failed when factorization intermediates exceeded the `u8`
exponent representation. Commit `3f24ebd` later widened those benchmarks to `u16`; the current
full sweep is valid and supersedes the exact-only #105/#178 limitation.

The final 500-call LBR profile records about `28.143 M` Symbolica factor cycles and `11.670 M`
FLINT cycles per call. Product-tree lifting falls from v1's `15.204 M` to `9.282 M` cycles
(`-38.95%`); dense monic remainder falls from `7.294 M` to `4.842 M` (`-33.62%`); reduction falls
from `2.687 M` to `0.608 M` (`-77.36%`). Dense tree multiplication accounts for about `2.297 M`
cycles.

This checkpoint identified modular prime screening as the next bottleneck: `13.701 M` cycles per
call, with bounded DDF at `13.147 M`. The dense DDF context and cached reciprocal work documented
above were the resulting architectural changes; use the current profile rather than these older
ceilings when selecting further work.

Artifact provenance:

| Artifact | SHA-256 |
|---|---|
| `/tmp/flint-comparison-factor-product-tree-dense-final-lto` | `a16283782f86769b48f5d85660c6d76875a613b9785d92bdc503fb2ab9a2ec46` |
| `/tmp/flint-comparison-factor-product-tree-dense-final-lto.d` | `c334141f0f56894fd32a0a9d157f2bb1131ea4107939a9beded35c8f7d1ffa35` |
| `/tmp/flint-comparison-product-tree-dense-final-lto-build.jsonl` | `c3feda1780ca79344df1ef4e916448c1d52dda1514cb36b2194a6acdffa74670` |
| `/tmp/factor-dense-final-d64-sha256.txt` | `0eb0ff8fd461f4679fc158be44712533f7ceb2c5434e7859ad01dd9b786391c9` |
| `/tmp/factor-dense-final-d63-sha256.txt` | `cb8d57f9c2ed0cb65faa9439ee8ea847fe88e53afa6ab906421a60501ad1fa94` |
| `/tmp/factor-dense-final-generated-guards-sha256.txt` | `da81d6762bddd7e2099ffffd3fd8201c596f355967d92d62c6504b18d5ca69c6` |
| `/tmp/factor-dense-final-pb105-sha256.txt` | `a94011a2a73560aef6562618c226f7b9a8b26840b427f00e4915fb10b4a41188` |
| `/tmp/factor-dense-final-pb178-sha256.txt` | `c3d24ac1c005cf0556a4d87e3bae6c853455c0b1a0af7954a7e939a2489161a7` |
| `/tmp/factor-dense-final-product64-sha256.txt` | `869d0a7eede47a7f28d6f84f05a76ceefeb5fc3e1a27a125126cfd12749e0669` |
| `/tmp/factor-dense-final-gcd64-sha256.txt` | `746c8c33db7a860d0e4d67d7d5b3aa986c752ee34adb449ef1b2575b9bfc2281` |
| `/tmp/profile-factor-product-tree-dense-final-d64-lbr.perf.data` | `9a9f219b942efae72166fb682b66756b6bf561cd4815d4d9a27838bef93e2a91` |
| `/tmp/profile-factor-product-tree-dense-final-d64-lbr.children-symbols.txt` | `f0a91729f6adfa3c822be0d269c8d04b4a44448f262becc597ccebb056774375` |

After merge, the factor module again passes `59/59` with default GMP features and `59/59` with
`--no-default-features --features integer-malachite,float-astro,native_code_generation`, both single-threaded and using
the main worktree's pre-existing ignored lockfile. The complete default `cargo test --workspace`
gate was last run on the preceding integrated chain.

The worktree state at that historical checkpoint was the following. It is retained only for source
provenance; the current candidate and integration state are documented at the top of this file.

| Worktree | Branch/head | State | Purpose |
|---|---|---|---|
| `/home/codexB/symbolica` | `dev`, merge `3184631` plus handoff updates | active handoff refresh | Integrated dense synchronized Hensel winner |
| `/tmp/symbolica-product-tree-dense-combined` | `codex/product-tree-dense-combined`, `6388bd3` | clean; tested source | Frozen source of the integrated dense tree binary |
| `/tmp/symbolica-product-tree-topology` | `codex/product-tree-topology`, `3c50b09` | clean; historical v1 source | First synchronized-tree implementation |
| `/tmp/symbolica-product-tree-dense-v2` | `codex/product-tree-dense-v2`, `af06d09` | clean; historical intermediate | Initial dense tree context |
| `/tmp/symbolica-integer-owned-remainder` | `codex/integer-owned-remainder`, `3895df9` | clean; source reference | Original owned GMP remainder optimization |
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
git -C /tmp/symbolica-product-tree-dense-combined status --short --branch
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
- `poly::gcd::tests` with `--no-default-features --features integer-malachite,float-astro,native_code_generation`:
  `28/28`;
- Numerica GMP integer tests: `21/21`, finite-field tests: `12/12`;
- Numerica `integer-malachite` integer tests: `16/16`, finite-field tests: `12/12`;
- `cargo fmt --check` and `git diff --check`.

The former bare alternative-backend test configuration at that checkpoint tripped an unrelated
evaluator test configuration. The current post-rebase build checks are recorded below.

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

## Historical performance versus FLINT before cached reciprocal DDF

Ratios are `Symbolica / FLINT`; lower is better. Values below one mean Symbolica is faster. These
are single-core paired measurements with default release features, including `faster_alloc`.
The current-source inventory near the top of this file supersedes this table.

| Case | Current or best accepted ratio | Status |
|---|---:|---|
| dense integer GCD, 1 variable, degree 32 | about `0.99` | parity; remains on scalar path |
| dense integer GCD, 1 variable, degree 48 | `1.086` | small residual loss |
| dense integer GCD, 1 variable, degree 64 | `1.171` | primary GCD loss mostly closed; final-tree build guard |
| dense integer GCD, 1 variable, degree 80 | `1.249` | residual high-degree loss |
| dense integer GCD, 2 variables, degree 5 | `1.001` | parity |
| dense integer GCD, 3 variables, degree 7 | `0.203` | Symbolica about five times faster |
| generated 3- to 8-variable GCD | generally below `1` | Symbolica generally faster |
| generated high-height GCD | about `0.5` | Symbolica about twice as fast |
| PolyBench 5-variable uniform #11 | `1.040` | small remaining Zippel loss |
| PolyBench 8-variable sharp #140 | `1.211` | residual Hu/Zippel loss |
| factor fixture product, 1 variable, degrees 33/31 | `1.104` | 2.17% slower than the local-bound source guard |
| factorization, 1 variable, degrees 32/31 | `4.001` | final-prime selector retains the legacy path |
| factorization, high-height 1 variable, total degree 33 | `4.055` | owned integer changes improve the guard 9.83% |
| factorization, 1 variable, total degree 64 | `2.485` | dense synchronized tree; 42.67% faster than local-bound baseline |
| factorization, 1 variable, total degree 65 | `5.858` | tree is excluded; source change is neutral |
| factorization, 2 variables, degrees 10/9 | `0.619` | faster than FLINT |
| factorization, 3 variables, degrees 6/5 | `1.949` | later modular/multivariate target |
| PolyBench 8-variable uniform factor #105 | `1.098` | neutral/slightly faster guard |
| PolyBench 8-variable sharp factor #178 | `2.976` | neutral guard |

The latest degree-64 and degree-63 factor rows use twelve 100-sample full-LTO processes. Product,
GCD, neighboring generated factors, and PolyBench rows use six-process guards from the same final
binary. Older tables below remain attribution evidence rather than current status.

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
leaves. The combined factor suite passed `50/50` under both GMP and `integer-malachite`; the later LLL test
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
- all `integer-malachite` integer-domain tests: `16/16`;
- signed-radix differential cases at 63/64/65 and 127/128/129 bits, an exact `i128::MIN` output,
  primitive values whose outputs require GMP fallback, and existing 180-bit `Large` inputs.

The specialized code is `#[cfg(feature = "integer-gmp")]`, so the `integer-malachite` run primarily guards fallback
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
- all `integer-malachite` integer-domain tests, `16/16`;
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
linear lifting. Routing and reconstruction tests passed under the default and `integer-malachite` feature
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

The accepted exact-subproblem solver recursively treats `Ok` children as exact targets with local
bounds. When a node returns `Err`, it completes that target's descendant lifts at one local modulus
and recombines them locally against the exact target before returning exact integer factors. Its
extracted LLL/subset recombination helper consumes one exact target and one consistently lifted
modular-factor set, avoiding repeated prime selection, DDF, and EDF.

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

This audit motivated the synchronized product-tree candidate documented in the historical product-
tree checkpoint above. It
updates all leaves and product nodes at shared p-adic stages instead of attempting fake binary
integer factorizations at unlucky intermediate partitions. The observed 5.3 ms fast/slow spread
established a larger ceiling than local bounds. Degree sorting alone could not distinguish the
three degree-10 factors over `GF(17)`; synchronized lifting removes the need to guess that integer
partition. A separate stabilization/exact-division shortcut would still need rational
reconstruction or leading-coefficient allocation because nonmonic lifting scales both children to
the full target leading coefficient.

The private dense DDF context described here was subsequently implemented and then extended with
cached reciprocal reduction; see the current checkpoint. The square-free GCD remained
Amdahl-limited and was not a useful target for this factorization fixture.

## Accepted contiguous Kronecker packing and geometric prime sampling

### Dense one-variable integer products

The current degree-64 product profile measured `0.005913 ms` for Symbolica and `0.005377 ms` for
FLINT. Symbolica's `DenseIntegerMul::run` occupied 42.75% of paired cycles: GMP multiplication was
17.06%, coefficient packing 8.23%, and the remaining self time included decoding and selector
work. FLINT's corresponding KS path spent 17.52% in multiplication, 17.86% unpacking, and 5.41%
packing. GMP multiplication was therefore not the gap; Symbolica's generic packer and wrapper were
the bounded target.

The total-degree-64 fixture has 33 left terms at indices `1..=33`, 32 right terms at indices
`0..=31`, a 145-bit signed radix digit, and 64 nonzero output coefficients. Sixty-two outputs fit
in signed `i128`; only two require `Integer::Large`. The accepted kernel:

- removes the starting-index offsets when both supports are consecutive and restores their sum
  while decoding;
- packs one-, two-, and three-limb digits from `Integer::Single` and `Integer::Double` through a
  fixed stack array;
- retains the general reordered/GMP-backed packer for sparse support, wider digits, and
  `Integer::Large` inputs.

Six alternating pinned processes with 10,000 samples per backend measured the source-matched
kernel change:

| Product | Control Symbolica | Candidate Symbolica | Candidate S/F | Symbolica change |
|---|---:|---:|---:|---:|
| degrees 32/31, total 63 | `5.627 us` | `5.140 us` | `0.992008` | `-8.66%` |
| degrees 33/31, total 64 | `5.927 us` | `5.475 us` | `1.019703` | `-7.62%` |
| degrees 33/32, total 65 | `6.319 us` | `5.846 us` | `1.087568` | `-7.49%` |

The final linked binary refreshes these inventory ratios to `0.973974`, `1.001645`, and
`1.070910`. The degree-64 product is now effectively tied with FLINT. Degree-64 GCD and the
degree-33/63/64/65 factor guards were neutral when only this kernel changed.

### Dense degree-64 factor-prime selection

Before changing the selector, a 1,000-sample LBR profile measured `2.903360 ms` for Symbolica and
`2.576247 ms` for FLINT, or `1.126973` S/F. Of all paired cycles, synchronized product-tree Hensel
lifting occupied 30.22%, modular-prime screening 17.84%, and retained EDF 2.91%. Relative to the
Symbolica factor subtree, these are approximately 58%, 34%, and 6%, respectively.

The proposed `17^10` subset reconstruction was tested before changing prime selection. At 41 bits
it reconstructed only the exact degree-2 factor and left five modular leaves. The same happened at
82 bits. At `17^39` (160 bits), degree-1, degree-2, degree-10, and degree-30 candidates reconstructed
but two degree-10 leaves remained; the existing root certificate still completed the factorization
at that stage. The experiment did not shorten lifting and its extra exact divisions were removed.

For the retained target, the first suitable image is `p=7` with eight modular factors. Sequential
selection then factors discarded images at `p=13` with seven factors and `p=17` with six factors,
retaining `p=17`. The accepted selector probes a prime at least one bit wider immediately. It ends
the small-prime search only when the wider image improves estimated lift work without increasing
factor count. Otherwise it backfills the skipped prime range. This extra condition matters for the
total-degree-65 guard: its `p=7`, `p=13`, and `p=17` images have 9, 7, and 12 factors. A broad
geometric rule incorrectly selected `p=17` and regressed that row to about `2.17` S/F; the final
rule backfills and retains the seven-factor image.

Six alternating pinned processes with 500 samples per backend measured the final rule against the
product-only control:

| Factorization | Control Symbolica | Candidate Symbolica | Candidate S/F | Symbolica change |
|---|---:|---:|---:|---:|
| high-height total degree 33 | `1.535 ms` | `1.531 ms` | `1.111896` | neutral |
| total degree 63 | `3.083 ms` | `3.125 ms` | `1.042068` | `+1.35%` guard |
| total degree 64 | `2.881 ms` | `2.628 ms` | `1.028278` | `-8.78%` |
| total degree 65 | `3.516 ms` | `3.541 ms` | `1.171321` | `+0.73%` guard |

The final 1,000-sample profile measured `2.640321 ms` against FLINT's `2.577169 ms`, or
`1.024504` S/F. Prime screening fell from 17.84% to 13.97% of paired cycles. Product-tree Hensel
lifting is now the largest remaining Symbolica component, but the whole operation is within about
3% of FLINT and is closed for the current single-core pass.

| Artifact | SHA-256 |
|---|---|
| `/tmp/flint-comparison-d64-constant-pivot-final` | `3ae577b8aa9480eb7c82009a70e01bd1e0cfe063cefecda77d55057ba3569b6f` |
| `/tmp/flint-comparison-d64-product-contiguous-pack` | `8fd9d0ab345e610fef7e19be1909fad17210902807ea753094898f12139e8deb` |
| `/tmp/flint-comparison-d64-product-contiguous-pack-build.jsonl` | `24f78e018dd497467d9c8091d48948396aa81cacb5aa30d33298742f42c19378` |
| `/tmp/profile-d64-product-current-lbr.perf.data` | `f4c6a3327de573550392bd5461de2677fadd5d5f4eb5dc7f6de0acaf5af0c87d` |
| `/tmp/profile-d64-product-current-lbr.flat.txt` | `46f17eaf2a16f4e069eb073961c89239ebce368079854bca66cc6008cd5b101e` |
| `/tmp/profile-d64-product-current-lbr.children.txt` | `a05bc9b9ada9eba61d16998e9aa6d90f14b0b44b13cb7bfb8fef407cbebe999b` |
| `/tmp/flint-comparison-d64-geometric-nonincreasing` | `453b6817a255a9815c3672cef22cd0160a15a021d1bc3d6c58a2add83b8303cc` |
| `/tmp/flint-comparison-d64-geometric-nonincreasing-build.jsonl` | `532e2411d9e49735bfea13c831ffaf2d6cb21289981a2a493649313a633b7d37` |
| `/tmp/profile-d64-factor-geometric-nonincreasing-lbr.perf.data` | `dd64b73906a49c677707af4120f2641db8093b11810dd77fa33fd3425f963c93` |
| `/tmp/profile-d64-factor-geometric-nonincreasing-lbr.flat-symbols.txt` | `f42a5f57e0da33177e9f02479b8babc8be64321e8e44af2a85382ef51ab66430` |
| `/tmp/profile-d64-factor-geometric-nonincreasing-lbr.children-symbols.txt` | `c880497cf7124a8ea5bd9ebdc0b1ad72ca5ef8774e7e2188e884b011ca29fa7a` |
| `/tmp/profile-d64-factor-geometric-nonincreasing.csv` | `c7e04022053fbb832d7761d72bb4614a41dac31471aa1a89f79d5b59714e89b0` |

Raw timing manifests are `465e20ab0927b23c83c5a1d782d585187b37eaeb27159370cbe251066ed21186`
for the product-kernel A/B files, `e4fc6627e5b980b2ec77b2bce2c25b1672f1f57cdc18ebd938a457c89066810d`
for final factor guards, `887ee47b40ff365a214ee56666bb139733877f98e19ccccbd3b650bda9ecf0f9`
for final products, and `f6fc31d23019f853af2a2a8c0ef471210c430127f0e2b216f94924b3a6086031`
for the final degree-64 GCD guard. The corresponding files are under `/tmp` with prefixes
`d64-product-contiguous-pack`, `nonincreasing`, `final-product`, and `final-gcd-d64`.

Validation passes all eight Kronecker-focused Numerica tests and the degree-63, degree-64, and
degree-65 univariate factorization tests, all single-threaded. The complete 89-test factor module
run had only the previously observed order-dependent `galois_upgrade` failure; its immediate exact
single-test rerun passed. The degree-65 regression test exercises the sequential backfill and
checks exact reconstruction with factor degrees 1, 2, 10, 20, and 32. A Symbolica check with
`--no-default-features --features integer-malachite,float-astro` also passes.

### Final post-rebase publication guard

The unpublished performance series was rebased onto `origin/dev` at `f9f7562`, which introduced
the split `integer-gmp`/`integer-malachite` and `float-mpfr`/`float-astro` backend features. All
locally added integer and finite-field conditionals use the new integer feature names. Default
GMP/MPFR and no-default Malachite/Astro checks both pass. Post-rebase focused tests pass all three
dense degree-63/64/65 factorization cases, all eight Kronecker tests, and all 47 solve-filtered
tests that cover the rebase conflict resolution.

A fresh full-LTO binary from the rebased source used default features, including `faster_alloc`,
and FLINT 3.6.0. Six sequential pinned processes give:

| Current-source guard | Samples per process | Pre-rebase row | Post-rebase S/F | Shift |
|---|---:|---:|---:|---:|
| PolyBench 5v uniform factorization #131 | 300 | `0.277201` | `0.279220` | `+0.73%` |
| dense total-degree-64 factorization | 500 | `1.028278` | `1.031329` | `+0.30%` |
| dense total-degree-64 product | 10,000 | `1.001645` | `1.017409` | `+1.57%` |

These are guard-sized shifts rather than attributed regressions. The `v11` primary rows use the
post-rebase values so the live scoreboard describes the exact published source. #131 remains
3.58x faster than FLINT; degree-64 factorization and product remain within 3.2% and 1.7% of FLINT.

The frozen binary is `/tmp/flint-comparison-final-rebased`, SHA-256
`637386f8d4d58072ad23f49b6a1ff89254ec94ab6ab8b3914f3cff09a3ceadf2`. Its build JSON is
`/tmp/flint-comparison-final-rebased-build.jsonl`, SHA-256
`e51f60e214d92e3e371e7fc04ecba73a6f90e15969f29294f843675630fd0f53`. Raw files use the
`/tmp/final-rebased-{pb131,factor-d64,product-d64}-{01..06}.csv` prefixes; their checksum manifest
is `/tmp/final-rebased-guards-sha256.txt`, SHA-256
`8193936446a2842f92e582404ae875c1a80cb47c13fe67c8c8c5c1e3a993fc96`.

## Benchmark infrastructure

Inputs are in `benches/support/cases.rs` and `benches/support/polybench_cases.rs`. Symbolica-only
rows are in `benches/symbolica_polynomial.rs`; paired Symbolica/FLINT rows are in
`benches/flint_comparison.rs`. Support code under `benches/support/{symbolica,flint,paired}.rs`
constructs the same cases for both libraries and validates outputs outside the timed region. The
harness fixes Rayon and FLINT to one thread, warms both sides, alternates execution order, and
reports median `Symbolica / FLINT` directly. PolyBench polynomials use `u16` exponents because
factorization intermediates can reach at least 256. Resultant elimination variables must be parsed
in the same namespace as their input polynomials.

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

The shared target is `/tmp/symbolica-univariate-dense-div/target`; a later build can overwrite its
hashed executable. Always copy the executable and matching `.d` sidecar to a source-specific
`/tmp` name immediately. The current frozen binary is listed in the checkpoint above. Full LTO
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

## Accepted compact-simplex integer multiplication

The dense eight-variable degree-5 factors contain 1,286--1,287 terms each, so each product forms
about 1.65 million coefficient pairs but only 43,758 possible total-degree output cells. Every
input coefficient is `Integer::Single`, and a conservative collision bound fits the accumulated
coefficients in fixed native storage. The old total-degree kernel nevertheless flattened the
inputs to GMP limbs and called low-level GMP multiplication for every pair. A pre-change LBR
profile attributed 31.67% of the combined Symbolica/FLINT samples directly to
`TotalDegreeIntegerMul`, with GMP multiply/add/subtract routines dominating that subtree. FLINT's
corresponding one-word Johnson path uses a small signed accumulator instead of a GMP object per
pair.

`TotalDegreeIntegerMul` now tries a machine-integer strategy before its retained GMP-limb
strategy. It scans both coefficient arrays once, requiring every value to be `Integer::Single`,
and proves

`max_abs(left) * max_abs(right) * min(left_terms, right_terms) <= i128::MAX`.

Distinct input monomials imply that at most `min(left_terms, right_terms)` pairs can contribute to
one output monomial, so this also bounds every partial sum. The kernel selects `i64` when possible
and otherwise `i128`; unsupported coefficients or bounds continue through the existing GMP path.
The rank-table layout is validated before unsafe inner loops, output ranks use checked arithmetic,
and the compact workspace is capped at `2^20` cells. The rank scan emits only nonzero
`(rank, coefficient)` pairs in increasing order. Polynomial reconstruction consumes those pairs
directly, removing the former second dense `Vec<Integer>` and scan.

High-height inputs need a different improvement because their coefficients are already GMP-backed.
`RingKernels` now lets a coefficient domain advertise the minimum compact output density at which
its total-degree multiplication kernel should precede mixed-radix dense multiplication. GMP
integers select density 8. The polynomial dispatcher takes this early route only when the ordinary
density-32 total-degree selector has not already applied and the mixed-radix workspace is at least
eight times larger than the compact simplex. These are support geometry and domain-capability
tests; they contain no benchmark names, fixed variable choice, or coefficient-value fingerprint.
Malachite integers and finite fields do not advertise the preference and retain their old route.

The first broad placement put the preference inside generic `mul_dense`. It produced the direct
product wins but regressed dense seven-variable end-to-end GCD from about `1.61 s` to `1.83 s`.
Profiles localized the loss to finite-field `construct_new_image_single_scale`, not to the native
integer kernel. That version was rejected. The final placement is an integer-domain opt-in before
`mul_dense`; the generic dense implementation is source-identical to its control. Raw rejected
A/B runs are `/tmp/dense7d7-gcd-ab-0{1..6}-{control,candidate}.csv`. The rejected candidate profile
is `/tmp/profile-dense7d7-gcd-total-degree-v3.perf.data`, SHA-256
`86ea30186730b2499e620921fdd92b3b654e52e96cd0a1dcaf35e87816a19fbc`; its control is
`/tmp/profile-dense7d7-gcd-control.perf.data`, SHA-256
`b7c46ced85faa9106ab14772c7471c0322dc60340c116f6a5369f418a1fef3f3`.

An initially defensive `2^20` pair-product floor on native accumulation was also removed after a
source-matched route experiment. On compact-simplex inputs that genuinely enter this kernel, six
alternating 100-sample processes give:

| Shape | Limb-control Symbolica | Native Symbolica | Reduction |
|---|---:|---:|---:|
| dense 5 variables degree 4 | `0.653795 ms` | `0.166174 ms` | `74.58%` |
| dense 8 variables degree 3 | `1.245218 ms` | `0.372313 ms` | `70.10%` |

Raw files are `/tmp/native-small-simplex-{5v4,8v3}-{candidate,control}-{01..06}.csv`. Earlier
three- and four-variable crossover probes were invalid as kernel comparisons because those shapes
continued through the preferred mixed-radix dense implementation; their small timing differences
were route-external noise and must not be used to restore the pair-count floor.

Six final repeated processes, pinned sequentially to core 8, give:

| Primary product row | Symbolica | FLINT | S/F | Previous S/F |
|---|---:|---:|---:|---:|
| dense 8 variables degree 5 | `10.117325 ms` | `30.944390 ms` | `0.326626` | `1.610087` |
| dense 7 variables degree 7 | `86.610154 ms` | `204.985350 ms` | `0.423081` | `1.584134` |
| seven-variable power-minus-one | `45.841458 ms` | `102.265095 ms` | `0.448043` | `1.348957` |
| high-height 5v d4, 128 bits | `1.857376 ms` | `2.171416 ms` | `0.857372` | `1.576901` |
| high-height 5v d4, 256 bits | `4.505168 ms` | `4.793389 ms` | `0.940781` | `1.246310` |
| high-height 5v d4, 512 bits | `14.849971 ms` | `14.082343 ms` | `1.054182` | `1.059355` |
| high-height 5v d4, 1024 bits | `41.013601 ms` | `40.324222 ms` | `1.017696` | `1.019191` |
| high-height 8v d3, 256 bits | `5.484608 ms` | `6.004982 ms` | `0.913425` | `1.054112` |

Raw product files use `/tmp/final-v5-<case>-<run>.csv`, with cases `dense8-product`,
`dense7-product`, `power7`, `high5-product`, and `high8-product`, and runs `01..06`.
End-to-end guards are `0.608434` for dense 8v, `0.927916` for
dense 7v, `0.473474` for high-height 8v, and `0.432587, 0.436824, 0.440813, 0.475476` for
high-height 5v at 128--1024 bits. All shifts are below 3%; the scoreboard retains the preceding
robust operation rows. Guard files use the same pattern with cases `dense8-gcd`, `dense7-gcd`,
`high5-gcd`, and `high8-gcd`.

The exact final full-LTO benchmark binary is `/tmp/flint-comparison-total-degree-fixed-v5`,
SHA-256 `1c99192014cb39bd618bff98e11fcd6ff1c611062fb128d6aa97cd19aa38ea04`; its build JSON is
`/tmp/flint-comparison-total-degree-fixed-v5-build.jsonl`, SHA-256
`1048895be29b47a9fb73d42377e565e962682c255366fa9bb7058b864b36e63e`. The pre-change dense-8
profile is `/tmp/profile-dense8d5-current-lbr.perf.data`, SHA-256
`8ba16391e9968f58d2dd22c0271edd54d9c847423d00caab73fbf16faab537e5`.

Focused default-GMP tests pass 4/4, including native `i64`, native `i128`, GMP-limb, route, and
chunked-route preservation cases. The corresponding Malachite test passes 1/1 with
`integer-malachite,float-astro,native_code_generation`. The current immutable scoreboard is
`CURRENT_STATUS_v12.md`: 97 of 116 primary rows favor Symbolica, 19 favor FLINT, and the median is
`0.640412` S/F.

## Historical dense-univariate checkpoint (`v19`)

This round started from exact source `0ea1d9ba69619e8ea0605d722708ee10e8ad83ec` and the frozen
full-LTO binary `/tmp/flint-comparison-head-0ea1d9b`, SHA-256
`195ee49aacca925a8e976d7b4feee504a32fd5782f0e3f9b3510e13f413d2e8c`. All timings below are
sequential paired measurements pinned to logical CPU 8, with Rayon and FLINT restricted to one
thread and default features including `faster_alloc`.

### Dense degree-64 GCD audit

A 2,000-sample LBR profile places 43.48% of all paired cycles in
`UnivariateModularGcdContext::run`. Within the paired profile, certified reconstruction occupies
27.55% and `DenseUnivariateIntegerDivisionContext::try_div` 25.77%. GMP add/subtract-multiply work
inside the checked exact-division certificate accounts for about 20.16 percentage points, while
`__gmpz_tdiv_qr` is only 0.75 points and divisor conversion about 0.52 points. FLINT certifies with
recursive `__fmpz_poly_divrem_divconquer`; Symbolica's checked certificate is basecase.

The existing `DenseIntegerExactDivision` kernel cannot replace this certificate: its contract
assumes divisibility and calls exact coefficient division, whereas an intermediate CRT candidate
may be wrong and must produce a checked remainder. An endpoint-adaptive experiment that divided
from the low endpoint when its coefficient looked cheaper was also rejected. It moved Symbolica
time by `+9.00%`, `+10.75%`, and `+8.79%` at degrees 48, 64, and 80. Low-to-high division replaced
cheap quotient/remainder operations by products involving the large leading endpoint. Raw files
are `/tmp/endpoint-div-d{48,64,80}-{baseline,candidate}-{1..6}.csv`; the rejected binary is
`/tmp/flint-comparison-endpoint-div`, SHA-256
`67cff74d7f04f17caec55ff58846c0ed657ddc9701ec8801347df3fe6e072411`.

No GCD source change was retained. The final combined binary gives `0.893922`, `0.853611`, and
`1.075259` S/F at degrees 48, 64, and 80. A meaningful next step would be a private *checked*
divide-and-conquer certificate, not reuse of the exact-only domain kernel. The profile is
`/tmp/profile-head-0ea1d9b-gcd-d64-lbr.perf.data`, SHA-256
`32fcaebdec588c8cb97add1727af194a523b006adf3e974d4c89987ede964ed5`.

### Deferred product-tree Bezout updates

Each Hensel stage previously updated every internal node's Bezout cofactors before trying root or
balanced exact reconstruction. Those cofactors are needed only for a subsequent lifting stage.
The retained implementation first lifts all factors and product-tree values, tries every exact
certificate, and performs the unchanged cofactor corrections only if every certificate fails and
another stage remains. The correction modulus divides the old modulus, so lifted children have the
same images needed by the saved old cofactors.

On the degree-64 fixture exact reconstruction succeeds at exponent 39; the final cofactor update is
now exponent 20. A forced irreducible probe at `17^40` verifies the failure path updates cofactors
and continues to `17^80`. Product-tree/binary differential tests cover odd and binary precision.
Six alternating 500-sample processes against exact current HEAD give:

| Factorization | Control Symbolica | Deferred Symbolica | Deferred S/F | Paired candidate/control |
|---|---:|---:|---:|---:|
| total degree 63 | `3.137201 ms` | `3.132702 ms` | `1.046569` | `1.000769` |
| total degree 64 | `2.697291 ms` | `2.188260 ms` | `0.857851` | `0.811924` |
| total degree 65 | `3.366768 ms` | `3.308651 ms` | `1.093720` | `0.982174` |
| high-height total degree 33 | `1.372767 ms` | `1.354936 ms` | `0.982928` | `0.987763` |

The degree-64 gain is 18.8% and changes it from a small loss to a 1.17x Symbolica win. Raw files
are `/tmp/deferred-bezout-{d63,d64,d65,d33h}-{baseline,candidate}-{1..6}.csv`. The factor-only
binary is `/tmp/flint-comparison-deferred-bezout`, SHA-256
`bf7a25b296200d6e6cf9a3a518519928783dca2f8c7464149952fa1301e22715`; its build JSON has SHA-256
`24a5276f59bf52437ae43093a577af3c233a6cb903de777320ce7cccb51c1aa2`.

### Allocation-free Kronecker bound sizing

For fixed-size input coefficients, the old Kronecker selector converted L1 sums and maxima into
GMP integers and allocated products although it consumed only their significant-bit counts. The
new helper computes the exact bit length of a `u128` by `u128` product from four 64-bit partial
products. The selector uses
`min(bits(||a||_1 max(b)), bits(||b||_1 max(a)))`. Its former collision-count bound is redundant:
the L1 norm of the shorter input is at most its term count times its maximum. Inputs containing a
large integer retain the exact GMP fallback.

Six alternating 10,000-sample processes isolate this change against the factor-only binary:

| Product | Control Symbolica | Final Symbolica | Final S/F | Symbolica change |
|---|---:|---:|---:|---:|
| total degree 63 | `5.3735 us` | `5.1370 us` | `0.995256` | `-4.40%` |
| total degree 64 | `5.6600 us` | `5.5075 us` | `1.027605` | `-2.69%` |
| total degree 65 | `6.0685 us` | `5.9220 us` | `1.102061` | `-2.41%` |
| high-height total degree 33 | `8.3330 us` | `8.2925 us` | `0.580402` | `-0.49%` guard |

Raw product files are `/tmp/native-bound-{d63,d64,d65,d33h}-{baseline,candidate}-{1..6}.csv`.
Factor guards use `/tmp/native-bound-factor-*`; GCD guards use `/tmp/native-bound-gcd-*`. In the
combined binary factorization remains neutral relative to the factor-only candidate and measures
`1.047138`, `0.853381`, `1.093223`, and `0.986543` S/F for degree 63, 64, 65, and high-height 33.

The final binary is `/tmp/flint-comparison-deferred-bezout-native-bound`, SHA-256
`edc543e82e9bef78f06d30dd516e5f4a9e93eeea0d25898c8f2022191f4fe467`. Its build JSON is
`/tmp/flint-comparison-deferred-bezout-native-bound-build.jsonl`, SHA-256
`bfdd4f88136954f7b174b9a80268b6e63694558fe6fa421086e756bb79a28579`. Validation passes all 126
Numerica unit tests and 22 doctests, all 98 factor tests on the clean rerun, all 36 GCD tests, and
`cargo fmt --check`. A no-default-features check with
`integer-malachite,float-astro,native_code_generation` also passes. The first factor-suite run had
the unrelated order-sensitive
`galois_upgrade` failure; its exact rerun and the complete clean rerun both passed.

### Why the headline worst rose after `v14`

The v14 and v18 primary-table intersection contains exactly 116 shared rows. Version v17 added
four asymmetric eight-variable configurations with separate product and GCD measurements, hence
eight new rows and no removals. The `v19` `1.327830` worst is the newly added multiplication of
8/45-term cofactors by a 165-term common factor. Its six process ratios are `1.3223..1.3354`, so it
is stable, but it has no v14 counterpart. On v14's fixed population, the v19 worst is `1.102061`,
versus `1.137177` in v14; the comparable envelope improved. The new product progression is
`1.327830` at 165 common-factor terms, `1.295551` at 495, and `0.746378` at 1,287, indicating a
small/medium compact total-degree crossover that still needs its own CPU profile.

## Current worst-first checkpoint (`v20`)

### Lower-band compact total-degree routing

The integer ring advertises a preferred total-degree density of five. Below the former density-eight
boundary, fixed inputs require a proven `i128` accumulator and a mixed-radix/simplex workspace
ratio of at least 16; GMP-backed inputs require a bounded limb accumulator and a ratio of at least
eight. A total-degree box upper bound rejects balanced low-dimensional requests before a second
full exponent scan. Every declined request retains mixed-radix or heap multiplication.

Six alternating source-matched processes measured:

| Product | Control S/F | Candidate S/F |
|---|---:|---:|
| asymmetric 8v, 165-term common factor | `1.327072` | `0.934524` |
| asymmetric 8v, 495-term common factor | `1.291910` | `0.811239` |
| asymmetric 8v, 1,287-term common factor | `0.732704` | `0.724628` |
| asymmetric 8v, 256-bit coefficients | `1.169166` | `1.034837` |

Raw files are `/tmp/v20-lowband-focused_dense-{baseline,candidate}-0{1..6}.csv` and
`/tmp/v20-lowband-focused_high-{baseline,candidate}-0{1..6}.csv`. Whole-program layout changes in
later accepted code move the final combined rows to `0.973857`, `0.871657`, `0.780811`, and
`1.051418`; the source-matched attribution above is the reason the selector was retained.

### Bounded native Hensel multiply/remainder

`DenseIntegerModularUnivariateContext` now computes
`((value rem divisor) * multiplier) rem divisor` in `i128` buffers when the modulus and every
coefficient are canonical fixed-width values, the divisor is monic, and a checked bound covers
both long divisions and the convolution. Factor and Bezout corrections use the fused operation;
unsupported widths follow the unchanged generic integer path.

On dense degree 65, six alternating processes change S/F `1.113511 -> 1.014777` and Symbolica
`3.373044 -> 3.073904 ms`, an 8.87% time reduction. Raw files are
`/tmp/v20-i128-d65-{baseline,candidate}-0{1..6}.csv`. The final degree-63/64/65 factor rows are
`1.048729/0.778657/1.008840`; the high-height degree-33 guard is `0.958149`.

### Native-limb contiguous Kronecker pipeline

The retained pipeline removes transient GMP work around the multiplication core:

- GMP-backed signed coefficients are encoded directly into fixed native digits.
- Packed magnitudes remain limb vectors and one checked `mpn_mul` produces the packed product.
- A streaming cursor decodes one-to-eight-limb digits without per-output index multiplication.
- Large decoded digits are written into recycled GMP storage through the writable-limb interface.
- Bounded native L1 and maximum statistics select a tight radix without temporary GMP sums or
  products; wider values use a collision bound only when its slack is at most `1/32`.

Dense degree-63/64/65 factor-product A/B ratios change
`1.029978/1.020687/1.061134 -> 0.896347/0.901481/0.939417`. The exact final-source rows are
`0.890401/0.884083/0.918407`. On the newly added dense degree-80 product, the final bound/output
step changes `1.049127 -> 1.006626`; the frozen combined row is `1.006322`.

The useful checkpoints are:

| Checkpoint | Degree-80 Symbolica median | S/F |
|---|---:|---:|
| native large-coefficient packing | about `0.0950 ms` | `1.2532` |
| direct packed-limb `mpn_mul` | about `0.0938 ms` | `1.2321` |
| bounded native collision/L1 selection | about `0.0790 ms` | `1.0495` |
| exact fixed-limb statistics and writable output | about `0.0759 ms` | `1.0066` |

The paired profile `/tmp/profile-v20-bounded-collision-d80-product-callgraph-lbr.perf.data` puts
about 34.4% of total cycles in `mpn_mul` for both Symbolica and FLINT. The former gap was bound
scanning, packing, and decoding, not the multiplication core. The pre-final binary
`/tmp/flint-comparison-v20-native-l1-output` has SHA-256
`7fcebf270fa98b20fc32a76d9a4aba5c498db0ac719849ca3d7dc0bf0c237947`.

### Small fixed inputs and the wide total-degree cliff

Short `Integer::Single` inputs now use stack-backed converted slices in the dense `i64` kernel.
The isolated dedicated dense-small A/B improves Symbolica by about 5.1%, while the dense
three-variable factor-input product improves only about 1.3%; its final `1.109145` S/F therefore
remains the current worst and needs dispatch/setup profiling rather than more allocator tuning.

The compact total-degree limb kernel and its workspace predicate now accept inputs through 128
limbs instead of 32. Nominal 512-bit degree-four inputs contain 2,049-bit coefficients and missed
the old boundary by one limb; nominal 1,024-bit inputs contain 4,097-bit coefficients. The old
fallback used a 59,049-cell mixed-radix workspace instead of the 1,287-cell simplex. The larger
cap uses a 2 KiB product scratch buffer and retains the checked `2^26`-limb output ceiling.

Six alternating 100-sample processes give:

| Input height | 32-limb control S/F | 128-limb candidate S/F |
|---:|---:|---:|
| 128 bits | `0.860782` | `0.867228` |
| 256 bits | `0.943686` | `0.949953` |
| 512 bits | `1.052777` | `0.970210` |
| 1,024 bits | `1.017507` | `0.991526` |

The 512-bit Symbolica median falls from about `14.84` to `13.64 ms`; 128/256-bit guards move less
than 1%. Raw files are
`/tmp/v20-wide-total-degree-highheight5-{baseline,candidate}-0{1..6}.csv`. The final binary is
`/tmp/flint-comparison-v20-wide-total-degree`, SHA-256
`f4d51132440463a421dc390c20cd83eaf1f1ebaa764109b5379574c9a0834bc6`.

### Final inventory and next decisive mechanisms

The frozen primary inventory has 125 rows, 116 Symbolica wins, nine losses, median `0.636796`,
worst `1.109145`, and best `0.028678`. The next changes should follow the ordered list below:

1. Profile the 11 us dense three-variable product's polynomial dispatch, index construction, and
   result reconstruction. Stack-backed inputs help another dense-small row by 5.1% but only reduce
   this row about 1.3%, so allocations are not the decisive explanation.
2. Before dense degree-80 GCD long division, apply exact constant-term and `x=1` divisibility
   filters. FLINT uses both before allocating its remainder. Then test a private one-level checked
   divide-and-conquer certificate: for dividend/divisor lengths 161/81 it cuts classical
   sub-multiply updates from 6,480 to about 3,200 and moves cross-work into two existing contiguous
   integer convolutions. No `Ring` method is needed.
3. Let dense `Zp` use the already-validated compact three-variable Kronecker map under a
   work/span selector. The GF(17) very-large case should reduce its packed span from 512,000 to
   256,040 positions.
4. For the remaining high-height asymmetric 8v product, the 8-by-165 half has density 2.67 and
   still uses a row heap. Test bounded GMP-only compact admission below density five together with
   a two-entry rank-table cache; the 45-by-165 half already uses the compact kernel.

## Current worst-first checkpoint (`v21`)

### Dense univariate diagnosis and rejected preliminaries

The initial degree-64 profile measured factorization at `0.782204` S/F, its input product at
`0.882885`, and GCD at `0.853816`. In the GCD profile,
`DenseUnivariateIntegerDivisionContext::try_div` occupied 25.36% of paired cycles while FLINT's
corresponding `_fmpz_poly_divides` subtree occupied 22.48% and used recursive divide-and-conquer
division. In the factor profile, dense distinct-degree factorization and Hensel reconstruction
were the substantial Symbolica subtrees; square-free/GCD work was not the missing mechanism.

The diagnostic timing files are `/tmp/v21-baseline-{factor-d64,product-d64,gcd-d64}-0{1..6}.csv`.
Profiles are `/tmp/profile-v21-d64-{factor,gcd}-lbr.perf.data`, with reports under
`/tmp/codex-readonly-v21-{factor,gcd}-{flat,children}.txt`.

Three preliminary changes were not retained as the final mechanism:

- A dense-DDF change moved degree-64 factorization only `0.779187 -> 0.774422` S/F, below the
  evidence threshold. Its binary is `/tmp/flint-comparison-v21-dense-ddf`, SHA-256
  `bd5e5fc33e556b61eca277f87c348043e39a78bf7e3fbd1357d08eb03ffd6d67`; its build JSON
  `/tmp/flint-comparison-v21-dense-ddf-build.jsonl` has SHA-256
  `d2d3b8aeefdbe87af1dc00c6d0577d473faf24d8678cf8072248c3a7557c64b5`. Raw files are
  `/tmp/v21-dense-ddf-factor-d64-{baseline,candidate}-0{1..6}.csv`.
- A private one-level checked division improved degree 64 `0.853801 -> 0.836601` and degree 80
  `1.073632 -> 1.033246`, but was superseded by the stronger two-ended certificate below. Its
  binary is `/tmp/flint-comparison-v21-divconquer-product-probe`, SHA-256
  `77d4a3768426a82777fcc2145fa0f0f63b38bdb0d1c1c1445125c60854cf0189`; build JSON
  `/tmp/flint-comparison-v21-divconquer-product-probe-build.jsonl` has SHA-256
  `ac98deb22d65bbac69122bf81d9d68c53355165ade1593a48a26b0cf76018a77`. Raw files use
  `/tmp/v21-divconquer-gcd-d{64,80}-{baseline,candidate}-0{1..6}.csv`.
- A genuinely recursive checked division regressed degree 80 `1.073506 -> 1.158099`. Its algebra
  survived independent exact and perturbed model checks, but sub-31 coefficient products missed
  Kronecker substitution and paid generic GMP addmul, allocation, and pair-reconstruction costs.
  The binary is `/tmp/flint-comparison-v21-recursive-divconquer`, SHA-256
  `a43907009395faa412a49119a1d3e265b013206f90f2ef8edadf395803ff39e4`; build JSON
  `/tmp/flint-comparison-v21-recursive-divconquer-build.jsonl` has SHA-256
  `60c31c01852a9bab8f1089b55f95426c569cbe5a21475149834562b26956d02d`. Raw files are
  `/tmp/v21-recursive-gcd-d80-{baseline,candidate}-0{1..6}.csv`.

### Accepted balanced two-ended checked division

`DenseUnivariateIntegerDivisionContext` now recognizes a dense balanced division with divisor
length `n` and dividend length `2n-1`. It solves the low `floor(n/2)` quotient coefficients from
the same number of low product equations using the nonzero constant pivot. It independently solves
the remaining high coefficients from the leading equations. Cross terms cannot reach either
retained endpoint interval, and one complete dense integer product then checks every coefficient,
including the omitted middle equations.

The route requires the GMP integer backend, divisor length at least 64, a fully dense divisor, the
balanced `2n-1` shape, and at least 75% stored-term density in the dividend. A failed pivot or
product comparison proves inexactness; an unsupported shape uses the existing classical checked
division. The specialization is private to the dense univariate operation context and adds no
method to `Ring`.

Six alternating source-matched processes give:

| GCD | Control S/F | Two-ended S/F |
|---|---:|---:|
| dense degree 64 | `0.853675` | `0.799476` |
| dense degree 80 | `1.073332` | `0.990435` |

Degree 48 does not meet the gate and stayed a clear win; dense degree-63/64/65 and high-height
degree-33 factor guards moved only within small whole-program layout shifts. The final combined
GCD rows are `0.894660/0.792087/0.986902` at degrees 48/64/80.

The accepted binary is `/tmp/flint-comparison-v21-two-ended`, SHA-256
`ae4bcbcd9e0dbb544cec254197cc62c523eaa514c651e1203f508c9d683bcce3`; build JSON
`/tmp/flint-comparison-v21-two-ended-build.jsonl` has SHA-256
`31d9744f05b47e8fc111d634ce8f0f05d605e2641517952d1e2222d8847f9acc`. Raw GCD files are
`/tmp/v21-two-ended-gcd-d{48,64,80}-{baseline,candidate}-0{1..6}.csv`; factor guards use
`/tmp/v21-two-ended-factor-{d63,d64,d65,d33h}-{baseline,candidate}-0{1..6}.csv`.
Focused tests cover odd and even quotient lengths, a zero constant quotient, perturbations at the
constant, middle, and leading coefficients, below-threshold and sparse fallbacks, and a 200-bit
GMP divisor.

### Accepted cache-sized chunked integer multiplication

The old worst product multiplies 83 and 56 terms in three active variables. Its mixed-radix output
box has `12^3 = 1,728` cells. FLINT splits one lexicographic variable into 12 outer rows and reuses
a 144-cell inner accumulator, scanning only the active prefix of each output row. The pre-change
paired profile attributed 50.79% of children to Symbolica multiplication and 45.58% to FLINT;
Symbolica's dense kernel had 16.04% self time, while FLINT's chunked array arithmetic and append
path dominated its side.

The integer dense kernel now admits a cache-sized chunk route at any carry-free mixed-radix
boundary. Selection requires at least three active variables, at least 1,024 output cells, an inner
slice of at most 256 cells, 8--256 outer rows, at least two coefficient products per output cell,
and triangular outer-probe work no larger than half the output box. The kernel then verifies the
exact active-prefix scan bound. The largest qualifying inner prefix is used, minimizing the outer
grid. Existing large sparse chunk selection remains a separate fallback.

Prepared coefficients and row ranges are constructed monotonically without per-term division or
remainder. Caller-owned stable slices avoid an inline/heap branch in every multiply-add. A proven
coefficient bound chooses `i64` or `i128`; inner slices of at most 256 cells use a fixed stack
accumulator, rows shorter than 128 terms use a direct loop, and only larger rows enter the blocked
loop. Active-prefix emission preserves sorted output and avoids clearing unused cells.

The first generic prototype regressed the target `1.091145 -> 1.126096`: generic `SmallVec`
accumulator code copied more than 10 KiB of inline state and retained the inline/heap branch in the
hot loop. The final package changes a clean alternating A/B `1.091013 -> 0.967059`; Symbolica's
median falls from `0.010837` to `0.009574 ms`, or about 11.65%. The exact rejection of preferred
total-degree routing below four declared variables also avoids a futile scan: a three-variable
box/simplex ratio is strictly below six and cannot meet the integer route's required ratio eight.

The accepted pre-namespace binary is
`/tmp/flint-comparison-v21-two-ended-chunked-i64-fixed`, SHA-256
`6c92fe08d83b190c35938c73ed54f4b33846bac54f4adc5db0ad8e8f6b76a3f4`; build JSON
`/tmp/flint-comparison-v21-two-ended-chunked-i64-fixed-build.jsonl` has SHA-256
`1cb8c409bcf11ce0dad7719191d99ce74c5e843a80129ec1dc65c50d687c90ee`. Raw clean A/B files are
`/tmp/v21-chunked-i64-fixed-factor-product-3v-{baseline,candidate}-0{1..6}.csv`. The rejected first
prototype and its profile use `/tmp/flint-comparison-v21-two-ended-chunked-i64`,
`/tmp/flint-comparison-v21-two-ended-chunked-i64-build.jsonl`,
`/tmp/v21-chunked-i64-factor-product-3v-{baseline,candidate}-0{1..6}.csv`, and
`/tmp/profile-v21-chunked-i64-3v-lbr.perf.data`. Their binary, build JSON, and profile SHA-256
values are `4f757a82585e22c41528ce4ab462805eebb134ded5fefb733dd6f73d8baef20c`,
`072201d34279c2d2f0e97bb267e8347f3c5bfe44ad2736a27aeebdfc7f4190a3`, and
`1c0a3d40e9a4c91dd6ccfdee5a43e7989a518d3fe2f03d9553a2cf3b4de043b2`.

Tests compare both `i64` and `i128` chunk accumulation with a heap product, exercise invalid
layouts and active-scan rejection, and check the exact 83/56/360-term target. The generalized
selector also covers an eight-variable radix-boundary example rather than recognizing a fixture
identity.

### Generated-factor benchmark namespace correction

`factorization_factors` parsed each base in the `polynomial_benchmark` namespace but previously
constructed its explicit variable map from unqualified symbols. The parsed symbols were therefore
appended after unused global symbols: nominal one-, two-, and three-variable cases stored two,
four, and six coordinates. FLINT always received only the declared variables. The map now uses
the same qualified symbols as the parser and asserts that both factors have exactly the declared
variable count.

This is a benchmark correctness repair, not a Symbolica library optimization. It requires all six
generated factor products and all six factor operations to replace their old canonical rows. The
final products are `0.821422/0.796646/0.809770/0.566259/0.812227/0.839533` for degree 63,
bivariate, three-variable, high-height degree 33, degree 64, and degree 65. The corresponding final
operations after the Montgomery change below are
`1.011864/0.395402/0.526129/0.941478/0.762465/0.989525`.

The clean chunk-kernel A/B remains valid library attribution because both compared binaries used
the same old six-coordinate/three-active-coordinate representation. The canonical 3v product
`0.809770` must not be compared directly with v20's `1.109145` as a library-only speedup. Its
dedicated six-process, 20,000-sample files are
`/tmp/v21-final-factor-product-3v-0{1..6}.csv`; the other corrected products are in
`/tmp/v21-final-generated-factor-products-0{1..6}.csv`.

### Accepted one-step bounded Montgomery reduction

The post-chunk dense degree-63 profile measured `1.038451` S/F. Symbolica spent 24.64% of paired
cycles in modular-prime screening, including 23.63% in `DenseZpDistinctDegreeContext::factor`;
the Hensel subtree accounted for 16.71%. The earlier direct DDF edit had saved only about 0.6%, so
the next target was the arithmetic repeated by all dense DDF products rather than another local
DDF rearrangement. The profile is `/tmp/profile-v21-final-factor-d63-lbr.perf.data`, SHA-256
`c8dd69f3dc37dda95899b6265db0cad8f3071746fc061397b97905742b89057a`; its callgraph report is
`/tmp/profile-v21-final-factor-d63-callgraph.txt`, SHA-256
`5deb579932b50c976f0a9cf0c1b1e1e3180cfad55641374abab6ad92f8fe200c`.

`DenseZpDistinctDegreeContext` already proves for its direct accumulation mode that the exact sum
of raw Montgomery products satisfies `t < p*2^32`. It previously split `t` into high and low words,
performed a field multiplication on the low word, and added the high word. `Zp` now exposes
`reduce_montgomery_product_sum(t)`, which applies the inherent 32-bit Montgomery reduction once to
the proven-bounded accumulator. Other bounds still use native `% p` or the wide fallback. This is
a domain operation with an explicit representation invariant, so it adds no generic `Ring` API.

Six clean alternating processes give:

| Source | Symbolica median | Median S/F |
|---|---:|---:|
| matched baseline | `2.964018 ms` | `1.044007` |
| one-step reduction | `2.879359 ms` | `1.013325` |

Symbolica time falls 2.856%. Final generated-factor guards are `0.395402` for bivariate,
`0.526129` for three-variable, `0.941478` for high-height degree 33, `0.762465` for degree 64, and
`0.989525` for degree 65. The accepted binary is `/tmp/flint-comparison-v21-montred`, SHA-256
`70f3c477effca92937c0ba042ddff88f21685992448eac3de1e56a175dd811d7`; build JSON
`/tmp/flint-comparison-v21-montred-build.jsonl` has SHA-256
`74e26f588ed380389472f476e7d109415e16dc059f1103168476eb479031cb23`. Raw A/B files are
`/tmp/v21-montred-factor-d63-{baseline,candidate}-0{1..6}.csv`; guard files are
`/tmp/v21-montred-generated-factorizations-0{1..6}.csv`.

The deterministic screening trace is `p=7` with 15 factors, `p=11` with 10 factors, `p=29` with
13 factors and therefore discarded, then `p=65000011` with 10 factors and retained. Probing the
existing direct-width prime before the geometric/backfill sequence could avoid discarded small
images when predicted lift pressure is already high, but that reordering was not implemented or
benchmarked in this round. It remains a future principled experiment, not an accepted result.

### Final v21 inventory and artifacts

The primary table now has 125 unique non-resultant rows, 121 Symbolica wins, four losses, median
`0.636796`, worst `1.025828`, and best `0.028678`. The worst operation is generated dense
bivariate GCD at `1.022443`. On the exact 116-row v14-compatible population it is also the worst,
below v14's `1.137177`. Resultants remain an appendix-only historical comparison.

The final pre-Montgomery combined binary is `/tmp/flint-comparison-v21-final`, SHA-256
`d4406d09928e17e12bf0e9b70cab233a5fec7abc21648329437e5113a3c1f3ce`; its build JSON
`/tmp/flint-comparison-v21-final-build.jsonl` has SHA-256
`b2601a05c2facb4acb99ea7e54cacd607f48cf903dcdc03dd76416947fb6ec49`. Final refresh files use
`/tmp/v21-final-*`; they cover all generated factor products and operations, the high-height
asymmetric 8v product, GF(17) dense-very-large multiplication, dense GCD degrees 48/64/80, the
dense degree-80 product, dense bivariate GCD, and dense-large integer multiplication. The accepted
Montgomery artifact and its guards above provide the final factor-operation rows.

The exact final source was rebuilt after documentation-visibility and test-only review changes.
Its binary is `/tmp/flint-comparison-v21-commit`, SHA-256
`83fa559d384b791324de92d40cc49e590a6e26c174834947839d48d7f3ca06f3`; build JSON
`/tmp/flint-comparison-v21-commit-build.jsonl` has SHA-256
`22fca6077dfa38bcc18cd8ae3ff26f7c51b9df8fac87b638428e294abf5c47c6`. Its six-process,
1,000-sample degree-63 refresh is `1.011864` S/F with a `2.879888 ms` Symbolica median; raw files
are `/tmp/v21-commit-factor-d63-0{1..6}.csv`.

The immutable scoreboard for this checkpoint is
[CURRENT_STATUS_v21.md](CURRENT_STATUS_v21.md).

## v22 compact output, bounded prime selection, and post-separable factor routing

### Compact total-degree output and dense-integer scan fusion

The v21 worst primary row was the high-height asymmetric eight-variable product with 8/45
cofactor terms, 165 common-factor terms, degrees 1/2/3, and 256-bit coefficients at `1.025828`
S/F. Its compact-simplex arithmetic was already competitive, but the output path independently
unranked every nonzero coefficient. An LBR profile attributed about 5.5% of timed Symbolica work to
that repeated combinatorial decoding.

`TotalDegreeExponentCursor` now owns the current weak composition and previous compact rank. The
first rank and every rank after a gap still use the exact binomial table. An adjacent rank advances
the existing composition directly, which covers the dense output pattern without changing layout
or selector rules. `TotalDegreeIntegerMul` also converts a negative two's-complement coefficient
slice in place and calls `MultiPrecisionInteger::try_from_lsf_limbs`; it no longer copies every
large result through a temporary `SmallVec` and Rug integer import.

Six alternating 1,000-sample processes on the pre-cursor and cursor binaries changed the former
worst row from `1.038237` to `0.954632`, an 8.05% ratio reduction. The exact final-source refresh is
`0.950184`, with process-median Symbolica/FLINT times `0.701282/0.738084 ms`. Generated product
guards also remained wins; the highest of three candidate-only processes was `0.981627` for the
5-variable 1024-bit product.

The dense Kronecker admission scan separately revisited every coefficient to discover signs before
computing bounded native-limb L1 and maximum statistics. `FixedLimbAbsoluteBitStatistics` now
returns `has_negative`. The preliminary scan stops at the first GMP coefficient; successful
bounded statistics supply the complete sign flag, and a failed bounded pass resumes after the
first large coefficient instead of rescanning the prefix. A regression test covers a negative
coefficient after both a supported and an over-wide GMP value. The clean dense degree-80 product
comparison changed `1.000782 -> 0.995977`; the exact final-source refresh is `0.998814`, with
process-median times `0.075208/0.075263 ms`.

### Bounded competitive direct-prime selection

Dense degree-63 factorization previously completed a third small modular image even after the
existing direct-width prime had become a competitive Hensel base. The final selector screens the
direct-width image and computes the largest small-image factor count that could still improve the
predicted linear lift work. The next small-prime DDF is bounded by that count. It still completes
an irreducibility certificate and any unusually favorable small image, but stops once the proven
factor-count lower bound cannot win.

This is a work comparison, not a degree-63 or benchmark-identity gate. Forced factorization modes
and the existing backfill rule are unchanged. Six alternating 1,000-sample processes changed the
source-matched degree-63 row from `1.014866` to `0.974875`. The exact final-source refresh is
`0.962829`, with process-median Symbolica/FLINT times `2.882831/2.992872 ms`.

### Post-separable univariate scout

The old full PolyBench distributions exposed one common factorization pathology: the automatic
box-density score increasingly preferred a bivariate start as degrees grew, even when a cheap
univariate image could decide the component. The retained factor control flow already supplies the
algebraic condition needed for a safe shortcut. After `factor_separable`, every nonconstant true
factor contains every active variable.

For an active variable of degree one, that condition directly certifies irreducibility. For minimum
active degree 2 through 64, the new scout evaluates every other variable at one. If the degree in
the selected variable is preserved, additivity forces every hypothetical factor to preserve its
positive selected-variable degree. An irreducible primitive univariate image therefore certifies
the multivariate component. Before using the square-free-only univariate reconstruction, the scout
checks `gcd(image, image')`; a repeated image is classified as `ReducibleImage`, not passed through
the square-free routine.

`ReducibleImage` deliberately does not imply that the target is reducible: specialization can make
an irreducible target image reducible. It is only a measured route signal. If Auto would otherwise
choose the bivariate start, a reducible image prefers the univariate start for the bounded retry
window; the existing bivariate fallback remains available. `Inconclusive` preserves the previous
route. Forced Univariate, Bivariate, and Disabled settings never call the scout and override the
preference exactly as before.

The scout factors the minimum-degree variable while the subsequent univariate route uses its
existing independently selected `order[0]`. The pilot work is therefore not reused and may inspect
a different coordinate. This is accepted because the image result is used as route evidence, not
as a reconstruction seed; unconditional reordering would bypass the existing sparse-univariate
guards. Counter-based tests prove that a reducible scout redirects a geometry whose production
density rule selects bivariate, and that forced modes execute zero scouts.

Fresh plain-release replays of all eight factor distributions against the retained exact FLINT
3.5.0 timings give:

| Distribution | Old paired median | v22 paired median | Old total S/F | v22 total S/F | Old wins | v22 wins |
|---|---:|---:|---:|---:|---:|---:|
| 05/0003 uniform trivial factor | 1.425747 | 0.447450 | 1.467039 | 0.474846 | 26 | 191 |
| 05/0004 uniform nontrivial factor | 0.266315 | 0.150398 | 0.436369 | 0.141560 | 173 | 199 |
| 05/0007 sharp trivial factor | 1.595357 | 0.449566 | 2.215536 | 0.766674 | 56 | 173 |
| 05/0008 sharp nontrivial factor | 1.364144 | 0.681384 | 1.940402 | 0.576983 | 89 | 135 |
| 08/0003 uniform trivial factor | 0.515150 | 0.479156 | 0.570522 | 0.505254 | 183 | 187 |
| 08/0004 uniform nontrivial factor | 0.472376 | 0.239445 | 0.193007 | 0.142893 | 186 | 197 |
| 08/0007 sharp trivial factor | 1.001295 | 0.501118 | 1.322808 | 0.809394 | 100 | 167 |
| 08/0008 sharp nontrivial factor | 1.234940 | 1.065408 | 1.530509 | 1.148495 | 84 | 93 |

The route-control adapter established why the reducible-image outcome must guide rather than merely
observe the route. Forced univariate changed 05/0004 to `0.175403` paired median and `0.163558`
total S/F, and 05/0008 to `0.721317/0.639252`. It was worse than the certificate-guided Auto path
on the trivial setups: 05/0003 was `0.653008/0.665963` and 05/0007 was
`1.049850/1.403100`. A global univariate preference is therefore rejected; the exact image outcome
is the principled discriminator.

An earlier gated scout without the reducible-image route improved 05/0003 and 05/0007 but left a
2--3% regression on 05/0004. The combined pilot removes that regression and delivers the large
nontrivial-factor gains above. The scout remains bounded to degree 64 and is invoked only on Auto
paths that are either about to construct a late quadratic discriminant or are planned to start
bivariate.

### Historical v22 full-distribution interpretation and then-next queue

This subsection records the queue as it stood after v22. The v26 bivariate Hu round resolves its
`05/0002` target and supersedes the current-action wording below; retain the figures only as the
before-state for that work.

Combining the eight fresh factor rows with the eight unchanged GCD rows gives 2,277 wins out of
3,200 problems, paired median `0.726631`, and summed-time ratio `0.305399`. Twelve of sixteen setup
medians favored Symbolica at that point. This was a mixed v22 checkpoint, not a claim that the eight GCD
rows were rerun by the v22 adapter.

Do not use one scalar ordering as the work queue. Keep three views:

- paired p50 for broad-path efficiency;
- total S/F together with absolute aggregate excess for throughput;
- p90/p99 and concentration of excess for failure-path tails.

The next broad target is 05/0002 uniform nontrivial GCD: p50 `1.323497`, p90 `1.628273`, total
`1.324827`, and `+442.954 ms` aggregate excess. The five largest cases explain only about 6.2% of
the net excess, so this is a systemic Zippel/kernel deficit rather than one heuristic tail. Use
retained problems 74 (median path), 116 (worst ratio, `2.212x`), and 176 (largest Symbolica time,
`13.148 ms`) from
`/tmp/symbolica-polybench-0.4.3-9QLH94/final-post-route/05/0002.problems.log`.

The second GCD target is 05/0006: p50 `1.213492`, total `1.140903`, and `+111.352 ms`. Profile
problem 203 for the median and 174 for the largest absolute excess; problems 100 and 13 are repeat
tail guards. The median geometry has degree vector `[9,34,30,30,4]`, while the tail is
`[11,36,11,28,4]`, making projected support and variable order plausible input-derived selectors.

The remaining factor deficit is a separate quadratic tail lane. 08/0008 improves broadly but has
p90 `7.942770`; problems 51 and 127 have minimum degree two, and 05/0007 problem 153 has degrees
`[29,31,11,25,2]`. They enter `factor_quadratic_before_square_free` before the post-separable scout.
A bounded modular discriminant-residuosity prefilter is the next principled experiment: evaluate
`b^2-4ac` at a deterministic point modulo a small prime without constructing the polynomial. A
nonsquare value proves that the exact polynomial discriminant is not a square and safely skips the
speculative product. Square or zero is only inconclusive and retains the exact path.

Sharp trivial GCD 05/0005 and 08/0005 have total ratios `2.103359/2.096462` but medians
`0.978281/0.950340`; their aggregate excess is only `62.282/108.117 ms`. Profile problem 130 and
problem 50 with prime/image rejection and backend counters after the broad 05/0002 work. Uniform
trivial GCD 05/0001 is last: its `1.018250` median corresponds to only `1.199 ms` excess across all
200 cases.

### v22 artifacts and verification

The factor-distribution adapter is `/tmp/polybench-symbolica-v22-pilot-route`, SHA-256
`839fae218300e99e40c196fb2a812a934d6b619b8163efd655e7910334b48fa9`. Raw 210-problem output
files are `/tmp/polybench-v22-pilot-results/{05,08}-000{3,4,7,8}.output.csv`; their SHA-256 values
are, in that order by variable count and setup:

```text
05/0003 26f4b397ef47b6388384438e1c0870e3fcd8235f7ad47ef0be019022d4801dc2
05/0004 624261da0c7d3c7fb9f8e09e7fa753ad960c296f22aa2466f5101e531c05dadb
05/0007 1614711f809a4faba1ac4b155cdc10145d952c950f4ebc7e946d496f2a369327
05/0008 299a43ee93b405caa84fba798264fe89872b2f5ed1a60b1305b32b35a93918ff
08/0003 d4c9ce9589dcb443828f5d4bef84c2feedc43cd74b276e906c63b8298d8443b2
08/0004 80a1b211bebb7022e6d95b106cec0aa27b6bc6b09355ab79e853b83f55c3940f
08/0007 9258d2a6eb53c131ea511b4b16f8c5d54fc991104c801c7c2eab6748b836543c
08/0008 d687d89ba333a1f1d96d09268efaf98202fdb7f446bd721e0d77f641728bb854
```

The exact final-source full-LTO binary is `/tmp/flint-comparison-v22-commit`, SHA-256
`38fb06ec5e8e38d2d3543b0245426859ef9a554abee776f2d3cf5ca7430455d6`; its dependency sidecar
SHA-256 is `aa0d39d3519e71525b34ab1cec7cfd8677775b2d720c364d127bc287552d82df`. Build JSON
`/tmp/flint-comparison-v22-commit-build.jsonl` has SHA-256
`7c3cfe8ec4d164955724746eb6210ee4175a06e6222bc7e33f828046c3a11541`. Six-process raw files are
`/tmp/v22-commit-factor-d63-0{1..6}.csv`, `/tmp/v22-commit-product-d80-0{1..6}.csv`, and
`/tmp/v22-commit-asym-highheight-product-0{1..6}.csv`; checksum manifest
`/tmp/v22-commit-key-rows-sha256.txt` has SHA-256
`834c3cf89dd9b2fde6c857ec49e0ab7db5ba72fe1e6029ea30b12d120825c22e`.

Verification passed:

- all 104 `poly::factor::test` tests;
- all 139 Numerica library tests;
- the exhaustive adjacent/gapped compact-rank cursor comparison for 1--8 variables;
- default GMP factor/scout focused tests including the repeated-image regression and global-setting
  race guards;
- `cargo check --no-default-features --features integer-malachite,float-astro`;
- `cargo fmt` and `git diff --check`.

The v22 primary inventory has 125 rows, 124 Symbolica wins, one loss, median `0.636796`, worst
`1.022443`, and best `0.028678`. The one loss is the generated dense bivariate GCD. Its immutable
checkpoint is [CURRENT_STATUS_v22.md](CURRENT_STATUS_v22.md); the live scoreboard has since
advanced to v23.

## v23 near-balanced dense GCD certificate

The accepted two-ended certificate previously required a divisor of `n` coefficients and a
dividend of `2n-1` coefficients, hence an `n`-coefficient quotient. The degree-64 fixture has a
second exact division whose quotient has `n-1` coefficients after its monomial shift. That input
still used classical checked long division.

`DenseUnivariateIntegerDivisionContext::two_ended_quotient_len` now admits quotient length `n` or
`n-1`. It splits a quotient of length `q` into `floor(q/2)` low coefficients and the remaining high
coefficients. The high solve uses the last `high` divisor coefficients rather than assuming its
start is the low split. One complete dense integer product still compares every dividend
coefficient, so endpoint solves cannot accept an incorrect quotient. The existing GMP, minimum
length 64, fully dense divisor, and 75%-dense dividend gates remain. Quotients shorter than `n-1`
fall through to classical checked division.

Six alternating full-LTO processes use 5,000 paired samples per GCD backend. The exact final-source
and v22 results are:

| GCD | v23 Symbolica | v23 FLINT | v23 S/F | v22 Symbolica | v22 S/F | Symbolica change |
|---|---:|---:|---:|---:|---:|---:|
| dense degree 48 | `0.189653 ms` | `0.209607 ms` | `0.904854` | `0.188515 ms` | `0.896055` | `+0.60%` guard |
| dense degree 64 | `0.278830 ms` | `0.391166 ms` | `0.711750` | `0.312692 ms` | `0.792789` | `-10.83%` |
| dense degree 80 | `0.585045 ms` | `0.589469 ms` | `0.993433` | `0.583851 ms` | `0.988098` | `+0.20%` guard |

The one-variable product and factorization triangle was refreshed from the exact same binary:

| Total degree | Product S/F | Factor S/F | Factor Symbolica/FLINT |
|---:|---:|---:|---:|
| 63 | `0.814256` | `0.985446` | `2.949644/2.991130 ms` |
| 64 | `0.815238` | `0.765733` | `1.954782/2.551953 ms` |
| 65 | `0.841130` | `0.993885` | `3.006729/3.024720 ms` |

Degree-64 factorization is neutral relative to v22 (`1.953616 -> 1.954782 ms`). Degree 65 moves
`3.001008 -> 3.006729 ms`. Degree 63 moves `2.885640 -> 2.949644 ms`, a 2.22% guard movement below
the standing 3% threshold. A direct final-versus-near-balanced replay reproduces
`2.950847/2.890511 ms`, while the new checked division is absent from the degree-63 factor profile.
The hot `DenseZpDistinctDegreeContext<u16>::multiply_low_into` address changes from an aligned
`0x...600` in v22 and `0x...560` in the intermediate binary to `0x...530` in the final binary.
This attributes the guard movement to deterministic whole-program LTO placement rather than a new
factorization route. All three operations remain faster than FLINT.

A larger dense-DDF experiment retained the modular residual in dense storage, reused dense GCD and
division buffers, and materialized only discovered blocks. Six alternating 1,000-sample degree-64
factor processes changed Symbolica only `1.950995 -> 1.948138 ms` (`-0.15%`) and S/F
`0.765473 -> 0.763286`; the apparent ratio movement was dominated by the independently measured
FLINT columns. The factor change and its tests were removed.

The final binary is `/tmp/flint-comparison-v23-final`, SHA-256
`7273a731b53e6a9804e2afb3f80b122176ff07be80abf6b79569b5416b40dce1`. Build JSON
`/tmp/flint-comparison-v23-final-build.jsonl` has SHA-256
`d4ae3aecbbf3d0845a7690ed0ded126684baf9f5ff159941941702d915b455e0`; the pre-documentation GCD
source diff has SHA-256 `e222bdd9696669003c20e0ce40627069f88e007e8d6f006c6c89dad71679f3a6`.
Raw timings use `/tmp/v23-final-{gcd-d48,gcd-d64,gcd-d80,product-d63,product-d64,product-d65,
factor-d63,factor-d64,factor-d65}-{final,v22}-*.csv`. The rejected DDF binary is
`/tmp/flint-comparison-v23-dense-ddf`, SHA-256
`9430cb394ab0e67390a716c0ea52ee54167c3b0707122532fe4f7fad4c7e8398`.

Verification passes all 37 `poly::gcd::tests`, the three degree-63/64/65 factor-route tests, the
`integer-malachite,float-astro` check, `cargo fmt`, and `git diff --check`. The immutable scoreboard
is [CURRENT_STATUS_v23.md](CURRENT_STATUS_v23.md), mirrored by
[CURRENT_STATUS.md](CURRENT_STATUS.md).

## v24 bounded post-small-prime search

The degree-65 screen trace exposed two complete modular images that could not become the retained
factorization. After extracting the exact factor `x`, the degree-64 cofactor takes this route:

| Prime | Bound | DDF outcome | Selection |
|---:|---:|---|---|
| 3, 5 | none | rejected before DDF | unsuitable |
| 7 | none | 9 factors | initial best |
| 17 | 9 | lower bound 10 at distinct degree 4 | geometric image cannot improve factor count |
| 11 | 10 | rejected before DDF | unsuitable |
| 13 | 11 | 7 factors | retained |
| 65,000,011 | skipped | formerly lower bound 18 at distinct degree 2 | fourth image is outside the low-factor search budget |

Before v24, prime 17 was completed with 12 factors and discarded, then the dense-u64 image was
screened with limit 8 and discarded. A compile-time experiment that omitted only the final wide
screen changed the six-process degree-65 Symbolica median `3.052915 -> 2.775970 ms` (`-9.07%`)
and paired S/F `0.996431 -> 0.956879`. This proved that selector work, not another allocation
rewrite, was the first target. The experimental binary is `/tmp/flint-comparison-v24-no-wide`,
SHA-256 `92d5df8fbd0ce9428c9a1240d2f594dad80a2ea673dfe320e28e45a9ea0175d7`; its build JSON is
`/tmp/v24-no-wide-build.jsonl`, SHA-256
`c3fd60636cf984879f971129c14ccb015960afd4f7347292f87840e84f997349`.

The retained rule uses the existing high-factor regime rather than a fixture signature. After
three suitable small-prime images, a dense-u64 screen is attempted only if at least ten modular
factors remain. Degree 63 retains ten factors and therefore preserves its profitable wide image.
Degree 64, degree 65, and high-height degree 33 retain six, seven, and three factors and stop after
the small-prime search. During the geometric second image, DDF now stops once its factor-count
lower bound exceeds the current best because the subsequent selection rule already refuses an
increased count.

Six final full-LTO processes per row give:

| Factorization | v24 Symbolica | v24 FLINT | v24 S/F | v23/control Symbolica | Symbolica change |
|---|---:|---:|---:|---:|---:|
| dense degree 63 | `3.025931 ms` | `2.818974 ms` | `1.076190` | `2.955895 ms` | `+2.37%` guard |
| dense degree 64 | `1.715859 ms` | `2.448305 ms` | `0.700836` | `1.964610 ms` | `-12.66%` |
| dense degree 65 | `2.739569 ms` | `2.898279 ms` | `0.945536` | `3.052915 ms` | `-10.26%` |
| high-height degree 33 | `1.267397 ms` | `1.340424 ms` | `0.945894` | `1.348807 ms` | `-6.04%` |

The degree-63 algorithmic route is unchanged and its Symbolica movement remains inside the 3%
whole-program-LTO band. In the strictly alternating A/B run, the FLINT median changes
`2.993279 -> 2.818974 ms` (`-5.82%`) between linked binaries. The exact v24 ratio is retained in
the scoreboard and is the next standalone factorization target; it must not be described as a
wide-prime selector regression.

The unchanged degree-63/64/65 products have exact v24 S/F ratios
`0.970816/0.965768/0.977305`. Their Symbolica medians move only
`0.004240/0.004333/0.004448 -> 0.004312/0.004405/0.004534 ms`, all below 2%; the larger paired
ratio movement comes from FLINT placement. The final-binary dense degree-64 GCD is
`0.266446/0.349917 ms`, or `0.761340` S/F, versus the v23 control
`0.281320/0.396383 ms`. Its source is unchanged and the primary table retains the more stable
v23 GCD and product rows.

The accepted binary is `/tmp/flint-comparison-v24-prime-budget`, SHA-256
`7e1c3daaed25c54fa25f52700c6080baf75a504375339a0ead19b1c0114bea4e`; its build JSON is
`/tmp/v24-prime-budget-build.jsonl`, SHA-256
`67911d68772186d70035d360c48e6e7f9222f420693f60a726a4aa24ccc78ef9`. The 62-file timing
manifest is `/tmp/v24-prime-budget-SHA256SUMS`, SHA-256
`6b76ba8e43c8b25930e4e714b930bc2e0614868d3b00fa40e9d3f6e284cdf61b`. Raw files use
`/tmp/v24-prime-budget-{factor-d63-ab,factor-d64,factor-d65,factor-high33,products,gcd-d64}*`.

All 104 default-GMP factor tests pass. The degree-65 route test also passes with
`integer-malachite,float-astro,native_code_generation`. Route assertions prove three dense DDF
screens at degree 65, one bounded geometric rejection at distinct degree 4, retained prime 13,
and unchanged reconstruction. Formatting and diff checks pass.

The current d65 profile also produced a viable but deferred arithmetic design. A cached Frobenius
operation must store the complete linear map with columns `x^(ip) mod f`; caching only
`x^p mod f` is insufficient because applying Frobenius is modular composition, not multiplication.
For small `p < degree`, columns can be built by monomial shifts followed by the existing reciprocal
reducer. A two-dimensional raw-product/reduction estimate gives break-even reuse counts four at
degree 62 and three at degree 32 for the retained `p=13` image. The map must be invalidated when
the DDF modulus shrinks. This is a genuine future mechanism, but it was not implemented after the
selector change already delivered a double-digit gain. Do not retry the rejected dense-buffer or
Kaltofen-Shoup rearrangements under the name of this map.

## v25 paid-for dense Frobenius map

The degree-63 profile localized the remaining factorization loss to repeated characteristic
powers in the wide-prime DDF screen. At `p=65,000,011`, each classical update computes a `p`th
power with 41 dense modular products. `DenseZpFrobeniusContext` instead stores column `i` as
`x^(i*p) mod f` and applies the resulting linear map to the dense coefficient vector. Over
`GF(p)`, the linear combination is exactly `a(x)^p mod f`.

Map construction uses the existing dense multiplication/remainder workspace. Application sums raw
Montgomery products with the same proved accumulator bounds as dense multiplication: the direct
Montgomery and native-remainder cases use `u64`, while the wide-remainder case uses `u128`.
Columns stay attached to the monic construction modulus. If DDF later removes a block, the current
residual modulus divides that construction modulus, so applying the old map and then reducing by
the residual computes the same residue as rebuilding the map. Tests cover that modulus shrink and
all three accumulator modes.

The selector deliberately does not estimate how many DDF steps remain. It keeps the first four
steps classical and constructs a map only when

```text
(current residual degree - 2) construction products + 1 application
    <= products in the very next classical pth power.
```

Thus the first use already repays construction in the same dense-product cost model. On the
degree-63 fixture, the degree-eight block shrinks the residual degree from 46 to 30. One map is
then built and serves DDF steps 9 through 15. The same gate rejects a degree-64 residual over
`GF(5)`: its 62 construction products plus an application cannot replace a three-product
characteristic power.

Two more aggressive policies were measured and rejected during review. An optimistic future-tail
bound reached about `0.840` S/F but could build immediately before DDF terminated. A policy that
treated completed classical work as rent reached `0.892129`, but the `GF(5)` counterexample showed
that past work did not imply future payback. The accepted one-step rule is weaker on the measured
tail but has a local input-dependent break-even guarantee.

Six independent 1,000-sample processes on the final full-LTO binary give:

| Process | Symbolica | FLINT | S/F |
|---:|---:|---:|---:|
| 1 | `2.714916 ms` | `2.988737 ms` | `0.908382` |
| 2 | `2.732489 ms` | `2.993067 ms` | `0.912939` |
| 3 | `2.717666 ms` | `2.994666 ms` | `0.907502` |
| 4 | `2.734103 ms` | `2.994862 ms` | `0.912931` |
| 5 | `2.739730 ms` | `3.018154 ms` | `0.907750` |
| 6 | `2.778094 ms` | `3.041444 ms` | `0.913413` |
| median | `2.733296 ms` | `2.994764 ms` | `0.910657` |

The clean frozen-source control has a `2.978711 ms` Symbolica median, so source-matched Symbolica
time falls 8.24%, beyond the 3% linked-layout band. The generated bivariate, three-variable,
high-height, degree-64, and degree-65 factor guards show no repeatable regression; their stable
primary measurements remain in the scoreboard.

The final weighted perf attribution is `12.234` million Symbolica cycles per call versus `13.314`
million for FLINT, or `0.918906`. The v24 profile had measured `16.333` million Symbolica cycles
versus `14.896` million for FLINT. Map application itself was below 0.31% in the more aggressive
diagnostic profile, so flattening the column storage does not have a useful present ceiling.

The accepted binary is `/tmp/flint-comparison-v25-frobenius-break-even`, SHA-256
`b0e7a7404a8c1fb3a3648dde9cb39a64249941693a821849828256a97512bad4`. Build JSON
`/tmp/v25-frobenius-break-even-build.jsonl` has SHA-256
`4326c459f8a813eff231199b63260ba9b832f0c47d346e6c0363df181597b9bb`. Raw timing files are
`/tmp/v25-frobenius-break-even-d63-00{1..6}.csv`. The final profile is
`/tmp/profile-v25-frobenius-break-even-d63.perf.data`, SHA-256
`1b56d6bed76812b4a1a1d70f0a325f9147648dfd6cee3e2913a9ceb4de43c545`; its flat and children
reports have SHA-256 `d497b186c8ad7fcb8cea27542d9542ffd4480040bb3afa536e80aacbe2e2e7fe` and
`f6f70e9563f5f17c3129c2f273b587287e5da0a3556918c92335f4c11d2d3aa6`. The artifact manifest is
`/tmp/v25-frobenius-break-even-SHA256SUMS`, SHA-256
`ed1a36ccc715da8c5ebf41bc570812fc08587c125c1706fa4c32fd387767d894`.

All 107 default-GMP factor tests pass. The focused degree-63 route test also passes with
`integer-malachite,float-astro,native_code_generation`, including the assertion that exactly one
Frobenius map is constructed. Formatting and diff checks pass. The immutable scoreboard is
[CURRENT_STATUS_v25.md](CURRENT_STATUS_v25.md).

## v26 bounded automatic bivariate Hu-Monagan GCD

The accepted route keeps two main variables in every modular image and evaluates all remaining
coordinates on one geometric sequence. Each image computes a bivariate finite-field GCD. Sparse
Berlekamp-Massey/root recovery and transposed Vandermonde interpolation reconstruct either the GCD
multiple or the smaller input's cofactor, CRT combines integer images until their coefficients
stabilize, and exact division certifies the result. A failed bounded automatic attempt returns to
the existing one-main-variable Hu/Zippel path; the explicit public bivariate method remains
uncapped.

`HuMonaganBivariatePlanningContext` scores candidate pairs by maximum input degree plus one unit
per 256 terms in the smaller leading row, with coordinate order as the deterministic tie-breaker.
An automatic plan is admitted only when all of the following input-derived conditions hold:

- at least three variables are active and both retained variables have positive GCD-degree bounds;
- the two full input boxes are sparse under the existing factor-eight density margin;
- four initial images satisfy `4 * pair_area <= left_terms + right_terms`;
- operand support is balanced: `max(left_terms, right_terms) <= 2 * min(left_terms, right_terms)`;
- twice the Kronecker exponent range fits the smooth-prime table;
- the projected GCD/cofactor support is at most one quarter of that Kronecker range;
- the retained image is amortized by projection work: `2 * projected_target >= pair_area`;
- `sample_limit = max(4, 2 * (projected_target + 1))` fits within combined input support.

Prime attempts are bounded by eight target CRT images, one stabilization image, and at most
`max(1, combined_terms / (4 * pair_area))` failed-image allowance. The first accepted prime also
caps later primes at its actual sample count. A stable reconstruction that fails certification
causes immediate fallback instead of consuming the remaining smooth-prime table.

The bivariate image helper receives known characteristic-zero degree bounds, strips each image's
content in the first retained variable, computes and restores their content GCD in the second
variable, and calls the known-shape modular routine without repeating public variable scans and
planning. Independent degree checks for both retained variables reject a prime or evaluation that
drops either leading face. Smooth primes fitting in `u32` now use `Zp`; larger primes use `Zp64`,
with one generic bivariate workspace. `ZpDiscreteLogContext` and `Zp64DiscreteLogContext` reuse the
Pohlig-Hellman digit tables and CRT idempotents across every reconstructed root at one prime.

The final image-amortization gate came from an explicit harmful-plan audit. Before that gate, 132
of 148 eligible `05/0006` plans were admitted. Problems 11 and 25 had bivariate-candidate/control
ratios `1.729x` and `2.381x`; their retained pair areas dominated their complete projected
interpolation targets. Requiring `2 * projected_target >= pair_area` rejects exactly those two,
leaving 130 admitted plans, and rejects no uniform `05/0002` plan. Earlier support/balance/sample
gates reject the measured 8v candidates, so their established one-main-variable routes remain
unchanged. This is a cost-geometry rule, not a setup or problem identifier.

Final measurements are:

| Workload | v25/reference S/F | v26 S/F | v26 total S/F | v26 wins |
|---|---:|---:|---:|---:|
| PolyBench 5v `0002`, uniform nontrivial GCD | 1.323497 | 0.749068 | 0.783395 | 185/200 |
| PolyBench 5v `0006`, sharp nontrivial GCD | 1.213492 | 0.983951 | 1.043248 | 103/200 |
| same-process 5v uniform nontrivial GCD #11 | 0.978321 | 0.765008 | n/a | 6/6 processes |
| same-process 5v sharp nontrivial GCD #11 guard | 0.514253 | 0.518311 | n/a | 6/6 processes |

The direct known-bound image helper was separately measured before automatic dispatch: problems
74 and 116 reduced direct bivariate Hu time by 6.4% and 17.8%, while problem 176 was neutral. The
final 12-case same-process guard has a `0.530297` median S/F; its slowest case is the 8v uniform
trivial GCD at `0.956630`. Across eight final GCD distributions plus eight retained v22 factor
distributions, the 3,200-problem median is `0.699823`, summed-time S/F is `0.288438`, Symbolica
wins 2,441 problems, and 14 of 16 setup medians favor Symbolica. The previous worst `05/0002`
distribution is now a broad win; `08/0008` factorization is the new worst setup at `1.065408`.

The final same-process binary is `/tmp/flint-comparison-v26-bivariate-hu-final`, SHA-256
`3d32f46f6d077e48c772da80c87c2ff1116d1b54abde6c081148ab9651b2bff5`; its six CSVs are
`/tmp/v26-bivariate-hu-final-polybench-gcd-00{1..6}.csv`. The final full-distribution adapter is
`/tmp/polybench-symbolica-v26-bivariate-hu-final`, SHA-256
`80eb2155cee27952bd794af5cf0bf6a323eada6871511e7884e529958454f3f0`, with outputs under
`/tmp/polybench-v26-final/{05,08}`. The build record is
`/tmp/v26-bivariate-hu-final-build.jsonl`, SHA-256
`5f1f52b1e02d64dacdabb83e28c391d41a49331f4fedf0536ba097ffbce4e744`. The complete manifest is
`/tmp/v26-bivariate-hu-final-SHA256SUMS`, SHA-256
`9c8d7e329e30445b2cde5193276cbc23fdd5d488aa6c492beb19194cded83dda`. Polynomial result payloads
match the preceding complete replay for all 1,600 GCD problems.

The reusable discrete-log test, all 47 GCD-module tests, the `binary_size` library check, formatting,
and diff checks pass. The complete 559-test library run has 558 passes and one unrelated existing
exact-interval expectation failure in `poly::univariate::roots::tests::isolate`; an isolated rerun
reproduces the same mismatch, as does the untouched v25 commit `d7d8da9`. The immutable scoreboard for this round is
[CURRENT_STATUS_v26.md](CURRENT_STATUS_v26.md).

## Rejected or low-value experiments

Do not repeat these without a genuinely new mechanism:

| Experiment | Evidence | Decision |
|---|---|---|
| Broad automatic bivariate Hu without support, balance, and sample budgets | helped the targeted 5v projected-sparse family but admitted costly 8v images | reject the broad selector; retain the bounded input-geometry plan |
| Bivariate admission before the retained-image amortization test | `05/0006` problems 11 and 25 measured `1.729x/2.381x` candidate/control because `pair_area` dominated projected work | superseded by `2 * projected_target >= pair_area` |
| Eager predicted-tail Frobenius-map gate | dense degree-63 factorization reached about `0.840` S/F | reject; a predicted tail can end before construction is repaid |
| Accumulated-work Frobenius-map gate | dense degree-63 factorization reached `0.892129` S/F, but degree 64 over `GF(5)` could build a 62-product map before a 3-product next step | superseded by the accepted one-step break-even rule |
| Reusable dense DDF GCD/division buffers in the v23 degree-64 factor pass | Symbolica `1.950995 -> 1.948138 ms`, only `-0.15%` across six 1,000-sample processes | reject and remove; materialization and temporary GCD/division buffers are not the remaining DDF bottleneck |
| Dense DDF rearrangement in the v21 degree-64 factor pass | `0.779187 -> 0.774422` S/F, about 0.6% | reject; too small for the code and the later one-step Montgomery reduction attacks arithmetic shared by the whole DDF loop |
| One-level checked dense integer division | degree 64 `0.853801 -> 0.836601`, degree 80 `1.073632 -> 1.033246` | superseded by the simpler and faster two-ended certificate |
| Recursive checked dense integer division | degree 80 `1.073506 -> 1.158099` | reject; small recursive products miss Kronecker and add GMP update, allocation, and reconstruction costs |
| First generic cache-sized chunk accumulator | dense 3v product `1.091145 -> 1.126096` | reject; copied more than 10 KiB of inline state and branched on inline/heap storage inside the multiply-add loop; the stable-slice implementation is retained |
| Enlarged byte-bounded large-integer cache | dense degree-80 product changed `0.094992 -> 0.097946 ms` (`+3.11%`) | reject; retaining substantially more limbs worsens locality |
| Owned packed values with delayed cache clearing | dense degree-80 product changed `0.094919 -> 0.094877 ms`, about `0.04%` | reject; below the evidence threshold |
| Packed exact-division certificate | degree-80 GCD changed `1.076663 -> 1.958108` S/F and Symbolica `0.632408 -> 1.149124 ms` | reject; packing and full-product comparison add more work than classical checked division |
| Direct-limb packed equality certificate | degree-80 GCD changed Symbolica `0.631447 -> 0.667688 ms` | reject; removing GMP imports does not make full-product certification competitive |
| High-half/triangular checked certificate | degree-80 GCD changed Symbolica about `0.633 -> 0.731 ms` | reject; decoded high-product work does not replace enough basecase updates |
| Collision-only Kronecker radix bound | selected a 322-bit digit where exact L1 statistics need 318 bits | superseded by fixed-limb exact L1 statistics; use the collision bound for wider inputs only when its slack is at most `1/32` |
| Endpoint-adaptive checked dense integer division | degree-48/64/80 Symbolica times rose 9.00%, 10.75%, and 8.79% | reject; a small low endpoint does not help when low-to-high division multiplies by the large leading divisor coefficient |
| Signed-centered adjacent quotient fusion | degree-64 GCD rose from `0.409653 ms` to `0.494380 ms`; branch misses increased 45.2% | reject; signed product branching costs more than the saved Montgomery reductions |
| Direct 1-4-limb `Integer` to `Zp64` conversion, `8440390` | degree-64 median `1.179371` versus `1.191412`; only about 1% | correct but Amdahl-limited; leave off `dev` |
| Fused adjacent-degree/Q1 modular remainder, `e0430d5` | ratio regressed to about `1.484` | reject; three-limb reduction cost exceeds two Montgomery reductions |
| Dense single-scale Zippel reconstruction, `c4748b6` | PolyBench #11 improved only about 1.3% | too much code for gain |
| Four-way `u32` geometric-image unroll | #11 measured `10.968324 ms`, `1.110136` S/F versus `10.727240 ms`, `1.087192` control; generated code added scalar/SSE shuffle work | reject and revert; independent products in this loop did not map to the profitable carry-chain schedule of the `u64` accumulator kernel |
| Owned/borrowed evaluated-pair enum | six #11 processes measured median `9.840455 ms`, `0.998494` S/F versus `9.643931 ms`, `0.978386` for the closure-form control | reject and revert; outlining a generic pair-return method changed hot layout and gave back 2.04% without changing arithmetic or selection |
| High-height-specific last-variable reuse exclusion | the one-image 8v guard does not enable reuse; context evaluation is 0.15% of the profile and the unchanged `0x48b9` image constructor shifted alignment | reject a fixture-specific gate; the 2.39% wall movement is whole-program-LTO placement and the final row remains `0.450165` S/F |
| Late, post-content Hu main-variable planning | #140 Symbolica time changed about `6.1273 -> 5.485 ms`, but the old main-coordinate `univariate_content` still consumed 26.50% of combined profile samples | superseded by the accepted pre-content hook, which avoids doing discarded-coordinate content work |
| Full Hu coefficient-row histograms, planner v4 | every candidate row was counted even when its degree made the 4x threshold impossible | reject the unconditional scan; use the exact pigeonhole lower bound and stop a bounded count at the first oversized row |
| Hu prefilter/dense/adaptive/reordered planner layouts, v5/v7/v9/v10 | the intermediate implementations selected the intended anisotropic schedules but did not give stable dense/uniform guard improvements across full-LTO binaries | reject those layouts; retain only the final degree prefilter, O(term-count) adaptive counter, fixed anchor, and pre-content placement |
| Cold generic pre-content wrapper, v13 | did not recover the dense/uniform guard shifts; binary SHA-256 `27fa54fa1bf985293016cf3d16f592bfe824c61480876e05dd8dd3c37320f933`, build JSON `44c27427e3e0eada66dee195be9bedcdf7f891f55505b97d631b3360218480e6` | reject; source was reverted |
| Cold integer pre-content entry, v14 | likewise did not recover the guards; binary SHA-256 `4b277840c17beab15b18abc7aaec9aa90f4a6923fbfc7c4ff25b453628e2b5d3`, build JSON `c60872c53a92b06b2ddd448e031a8001b1e4ba078aa85ff3ae1ccd351ab9846e` | reject; source was reverted and the final result reports the cross-binary LTO sensitivity |
| Broad dense modular GCD experiment, `5a975f1` | little gain at degree 64, degree-80 regression | superseded by smaller dense-image/certificate contexts |
| Borrowed versus by-value finite-field calls | no measurable difference | not the source of the old unexplained `1.325` result |
| Divide-and-conquer univariate attempt | degree-64 ratio about `2.736` | decisive regression |
| Cache-sized chunks inside direct `DenseZp64` multiplication | candidate measured `11.261875 ms` versus FLINT `9.306603 ms`, or `1.210095` S/F; 53.24% of paired profile cycles remained in `ChunkedDenseZp64Mul::run` because per-product chunk addressing replaced the intended cache benefit | reject; source was removed. Binary SHA-256 `5e7dc9c2a33b23feda0288e4c4eabf75594919c645bea971e4a112961b046049`, build JSON `48b2ba4d27d084456a4662eac55c9d23314dca7841b625935f41fd3b2ad8a55c`, profile `f856e521cafe9867124343163c59f08c7c8a1df77af816f2ecb8757ca304be95` |
| Direct-limb conversion as the whole product answer | current product profile shows packing/unpacking, not GMP multiply, dominates | target conversion as an operation context; do not redesign `Ring` |
| Large-integer recycled storage alone | reduces allocation-heavy paths but did not close FLINT gaps | retain it, but algorithm/data layout matter more |
| Product selector `85be422` as a factorization fix | product improved 16-17%; factor row did not materially move | factor input product is outside timed region |
| Square-free/GCD tuning for the 1-variable factor fixture | about 0.5% of Symbolica factor time | not a plausible explanation for a 7x gap |
| Automatic CRT resultant | not generally competitive with direct Ducos | keep explicit `resultant_crt`, not default |
| Brown PRS as general resultant default | not competitive regime-wide | keep explicit `resultant_brown` |
| Sparse multivariate resultant interpolation prototype | reconstruction/bound work overwhelmed sparsity | not used as the general practical method |
| Unguarded quadratic Hensel lifting | high-height degree 33 improved to about 14 ms, but degree 63 regressed from about 16 ms to 34-37 ms and degree 64 to about 35 ms | superseded by the 64-digit and root-wide four-factor guard in `4a2b9c7` |
| Exact integer convolution windows for Newton Hensel remainders | window microkernels often beat Symbolica's full fallback, but degree-64 factorization regressed from `1.290968` to `1.313523` S/F | reject the integration; the unused public window API and 48 diagnostic rows were removed |
| Product-tree subset reconstruction at `17^10` | only the degree-2 factor reconstructed at 41 and 82 bits; the complete certificate remained at `17^39` | reject; failed exact divisions add work without shortening the lift |
| Broad geometric factor-prime sampling | degree 64 improved to about `1.03` S/F, but degree 65 regressed to about `2.17` because a 12-factor `p=17` image displaced the 7-factor `p=13` image | superseded by the non-increasing-factor-count rule with sequential backfill |
| Three bivariate images on every initial sample | dense three-variable factorization spent 87.5% of Symbolica cycles selecting samples; one certified initial image reduced its median 61.4% | superseded by the guarded initial shortcut in `f39c09b`; retain three images on bounded retries |
| Compact fixed-width total-degree multiplication for dense-five degree 7 | `1.658863` S/F; rank-table work replaced the saved flat-array clearing cost | reject for this regime; use the accepted carry-free mixed-radix chunks in `ec6a131` |

Historical frozen binaries and perf data remain under `/tmp`; it is ephemeral. The most useful
older checksums and ratios are retained in Git history of this document at commits `386174d` and
`3f2d7eb` if deeper attribution is needed.

## Code map and design preferences

- Integer multiplication dispatch, chunked mixed-radix accumulation, and Kronecker conversion:
  `lib/numerica/src/domains/integer/polynomial_kernels.rs`, especially `DenseIntegerMul`,
  `ChunkedDenseIntegerMul`, `TotalDegreeIntegerMul`, `try_kronecker`,
  `try_fixed_limb_absolute_bit_statistics`, and `try_decode_fixed_kronecker_digits`.
- Cache-sized chunk selection and carry-free radix-boundary planning:
  `src/poly/polynomial.rs`, especially `cache_sized_chunked_dense_mul_is_preferred` and
  `MultivariatePolynomial::mul_dense`. Compact-simplex output rank decoding is
  `TotalDegreeExponentCursor`.
- Cached GMP integer construction from decoded native magnitudes:
  `lib/numerica/src/domains/backend/integer.rs::MultiPrecisionInteger::try_from_lsf_limbs`.
- Integer multiplication differential tests:
  `lib/numerica/src/domains/integer.rs`.
- Bounded packed-row sparse multiplication and its selector:
  `src/poly/polynomial.rs::{packed_row_merge_is_bounded,try_packed_u8_row_merge_mul}`.
- Dense univariate modular GCD and exact certificate contexts:
  `src/poly/gcd.rs`, especially `DenseZp64UnivariateGcdImage`,
  `DenseUnivariateIntegerDivisionContext::{low_triangular_quotient,triangular_quotient,
  try_div_balanced_two_ended}`, and `UnivariateModularGcdContext`.
- Repeated Zippel image evaluation: `src/poly/polynomial.rs`, especially
  `LastVariablePowerWorkspace` and `LastVariableEvaluationContext`; pair-level selection and
  shifted-Vandermonde batch inversion are in `src/poly/gcd.rs`, especially
  `RepeatedLastVariableEvaluationContext` and `solve_shifted_transposed_vandermonde`.
- Pre-content Hu-Monagan main-variable planning: `src/poly/gcd.rs`, especially
  `HuMonaganAnchor`, `HuMonaganPlanningContext`, `CoefficientRowCounter`, and
  `PreparedHuMonaganGcd`. The same file contains the generic pre-content call site and the narrow
  `PolynomialGCD::gcd_with_precontent_plan` hook; only `IntegerRing` overrides it.
- Automatic two-main-variable Hu-Monagan planning and execution: `src/poly/gcd.rs`, especially
  `HuMonaganBivariatePlanningContext`, `PreparedHuMonaganBivariateGcd`,
  `gcd_hu_monagan_bivariate`, `hu_monagan_bivariate_prime_image`,
  `hu_monagan_bivariate_image_gcd`, and `hu_monagan_sparse_interpolate_bivariate`. The bounded
  automatic route shares the public implementation but supplies sample and prime-attempt budgets.
- Reusable fixed-word Pohlig-Hellman contexts: `lib/numerica/src/domains/finite_field.rs`, especially
  `FiniteFieldDiscreteLogContext`, `ZpDiscreteLogContext`, and `Zp64DiscreteLogContext`. The
  bivariate Hu workspace and `DenseRootPrimeField` bridge both word widths in `src/poly/gcd.rs`.
- Integer factor selection/reconstruction/Hensel lifting: `src/poly/factor.rs`; bivariate image
  selection is `find_sample`, the active pressure selector is around `high_linear_lift_pressure`,
  and the guarded #84 order selection is
  `reorder_integer_factor_variables_for_sparse_univariate`. Product-tree lifting defers its
  Bezout-cofactor updates until exact reconstruction probes have failed. Native bounded factor
  and Bezout corrections are implemented by
  `DenseIntegerModularUnivariateContext::try_i128_multiply_remainder_monic`. The v22 post-separable
  certificate and route signal are in `univariate_specialization_factorization`; direct-prime
  competition is controlled by `competitive_small_prime_factor_limit` and
  `preferred_dense_u64_factorization_work`.
- Balanced two-leaf Hensel reconstruction: `src/poly/factor.rs`, especially
  `UnivariateHenselProductTreeTopology::{most_balanced_leaf_pair,balanced_leaf_pair_improving_root}`,
  `UnivariateHenselExactPartition`, `univariate_hensel_shortened_target`,
  `try_reconstruct_balanced_leaf_pair`, and `recombine_exact_product_tree_partition`.
- Shared polynomial operation kernels: `src/poly/kernels.rs`.
- Coefficient-domain polynomial kernel capabilities: `lib/numerica/src/kernels.rs`; the integer
  implementation is `lib/numerica/src/domains/integer/polynomial_kernels.rs`.
- Large-modulus finite-field and modular-ring multiplication:
  `lib/numerica/src/domains/finite_field/polynomial_kernels.rs`. The 64-bit-prime direct path is
  `DenseZp64Mul::multiply_direct` with `add_u64_product_row_unrolled4`; arbitrary-size moduli use
  `DenseIntegerMontgomeryMul`. Both are exposed through the existing `PolynomialKernels`
  capability.
- Bounded one-step 32-bit Montgomery reduction:
  `lib/numerica/src/domains/finite_field.rs::Zp::reduce_montgomery_product_sum`; its dense DDF
  consumer is `src/poly/factor.rs::DenseZpDistinctDegreeContext::reduce_accumulator`.
- Dense high-characteristic DDF powering: `src/poly/factor.rs`, especially
  `DenseZpFrobeniusContext`, `DenseZpDistinctDegreeContext::characteristic_power_product_count`,
  and `should_cache_frobenius`. The cache stores the Frobenius linear map for one construction
  modulus and remains correct after reduction to a factor of that modulus.
- Generated and PolyBench fixtures: `benches/support/cases.rs` and
  `benches/support/polybench_cases.rs`; generated-factor variable-map construction and its
  namespace assertion are in `benches/support/symbolica.rs::factorization_factors`.

Prefer short-lived operation-context structs that own reusable buffers and precomputed metadata.
Keep multiplication kernels attached to the integer domain where they need tagged integer/GMP
knowledge; call them from polynomial dispatch after the ring type is known. Do not add narrowly
specialized methods to `Ring`. `exact_div_owned`-style ownership-aware operations are useful when
their semantics are general; coefficient-domain-only fast paths belong in an operation context or
a narrow trait, not the base ring interface.

Comments should say what a function computes, its input invariants, and where it is used. Avoid
comments that justify file organization by contrasting it with designs not present in the code.

## Current ordered next actions

1. Attack `08/0008`, the current worst setup at `1.065408` median, `7.942770` p90, and
   `+185.726 ms`. Prototype a one-sided modular nonsquare certificate before
   `factor_quadratic_before_square_free` constructs `b^2-4ac`: a nonsquare value at any
   deterministic prime/point proves the exact discriminant is not a square, while square or zero
   remains inconclusive. Measure problems 51/127 and reducible quadratic guards.
2. Treat `05/0001` (`1.027028`, only `+2.514 ms`) and the generated dense bivariate GCD
   (`1.022443`) as fixed-overhead work. For the dense primary row, first test FLINT's deterministic
   evaluation-divisibility prefilter and the near-balanced two-ended certificate generalization;
   do not introduce a new broad backend for a 2.2% gap.
3. Profile sharp trivial GCD tails `05/0005` problem 130 and `08/0005` problem 50 with backend,
   prime/image, collision, rejection, and exact-certificate counters. Their medians already favor
   Symbolica, but total S/F `2.181255/2.106076` identifies concentrated expensive failures.
4. Audit the remaining positive-delta `05/0006` cases after the accepted selector. Its median is
   now `0.983951`, but total S/F is `1.043248`. Any widening must use projected support and image
   cost, retain the amortization gate, and preserve both rejected problem geometries without
   referring to their IDs.
5. `05/0002` is complete for this round at `0.749068` median and `0.783395` total S/F with 185/200
   wins. If this family is revisited, remove the sparse-polynomial round trip inside known-shape
   bivariate images with a reusable dense image context; do not loosen the proven automatic gates.
6. Keep degree-63/64/65 products as guards at their stable primary values
   `0.814256/0.815238/0.841130`, and dense degree-63/64/65 factorization at
   `0.910657/0.700836/0.945536`. Do not add another product abstraction without a mechanism
   different from the rejected broad multiplication contexts.
7. The dense degree-80 product is `0.998814`; defer its exact top-bit correction until the larger
   factor and GCD tails are resolved.
8. Keep resultants in the historical appendix, freeze and hash each accepted full-LTO binary, and
   create the next immutable `CURRENT_STATUS_v<i>.md` snapshot after every accepted round.

## Historical ordered next actions

The following list drove the preceding Hensel, dense-EDF, and multiplication experiments. Items
1--3 were implemented, item 5 produced the accepted dense EDF and rejected reciprocal/Newton
variant, and item 6 was superseded by the accepted bounded packed-row merge. It is retained to make
the experiment sequence reconstructible.

1. In a separate candidate, replace the quadratic-Hensel quotient identities with direct
   dual-remainder corrections. Compute corrections such as `((E rem w) * s) rem w` and the
   corresponding correction modulo `u`, matching the identities already used by the dense product
   tree. This should remove quotient reconstruction and the `q*u` multiply/add. Preserve nonunit
   unit-leading divisors through the delayed general remainder path; do not normalize factors in a
   way that changes the lifted leading-coefficient allocation.
2. Differential-test the new correction formulas against the retained quotient formulation at
   small and large composite prime powers, partial final precision, unsuccessful congruences, and
   nonunit modular leading coefficients. Keep the implementation private to the Hensel operation
   context and add no `Ring` methods.
3. Measure high-height degree 33 against the accepted delayed-division binary, then recheck its
   product, degree-63/64/65 factor boundaries, 2- and 3-variable factors, the widened PolyBench
   factor sweep, and degree-48/64/80 GCD. Profile only a full-LTO winner.
4. If direct remainders leave large-modulus division dominant, test a private cached reciprocal for
   the correction modulus. FLINT caches a reciprocal for very large `fmpz_mod_ctx` moduli; Newton
   polynomial division is not relevant to the degree-33 divisor sizes.
5. For dense degree 64, the next larger factor candidates are a guarded reciprocal/Newton remainder
   at FLINT's divisor/quotient-size threshold and a private dense finite-field EDF context. Keep
   these separate from the high-height quotient change.
6. Return to the integer input-product path only with a mechanism different from the rejected
   one-scan `PolynomialMulContext`; that candidate regressed the PolyBench construction median by
   `0.24%`, the dedicated multiplication median by `0.96%`, and the degree-64 product by `2.27%`.
7. Freeze and hash every accepted full-LTO binary and profile, integrate only measured winners with
   Ben Ruijl's identity, and keep this file current after every accepted or rejected experiment.
