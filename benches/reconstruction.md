# Rational function reconstruction experiment

Latest update: [homogeneous component pruning](reconstruction-components.md)
adds a selectable method that closes the dense `f3`/`f4` probe gap against
FireFly and reduces the amplitude coefficient count.
The [matched public stress benchmarks](reconstruction-matched-stress.md) compare
actual FireFly and FIRE7 on public coefficients and remove learned monomial
factors from later interpolation rows.
The [preceding update](reconstruction-stress.md) adds early numerator/denominator
completion and an optional unbalanced-degree search.
The earlier [reference comparison](reconstruction-probes.md) records the preceding version.

Worktree: `/common/dev/symbolica-reconstruction`, branch `codex/rational-reconstruction`,
based on `d78823e`. Implementations are in `src/poly/reconstruction.rs` and its
`rational` submodule. The original worktree's uncommitted changes are not included.

## Algorithms and scope

Kira 2.0 delegates rational reconstruction to FireFly
([Kira paper](https://arxiv.org/abs/2008.06494)). This experiment implements
Cuyt–Lee as described in [FireFly, section 2.2](https://arxiv.org/abs/1904.00009):
homogenization, Thiele degree discovery, normalized linear solves, and sparse
polynomial Zippel interpolation. Random shifts handle a pole at the origin;
subtracting their contributions from higher homogeneous components preserves
sparsity while reconstructing lower components.

The alternative is balanced Zippel from
[Smirnov–Zeng, section 2.5](https://arxiv.org/abs/2409.19099). Each stage reconstructs
the next variable on geometric sampling rows and balances their normalization
against the preceding slice. A transposed Vandermonde solve lifts each coefficient
of that variable; this is algebraically equivalent to applying those linear solves
to values and then Newton-interpolating.

The initial measurements below time new in-tree implementations. The later
[performance update](#performance-update) also compares actual FireFly and FIRE7 code.
They reuse Symbolica's polynomial representation, Newton interpolation, shifted
transposed Vandermonde solver (including batched inversions), finite-field matrix
solver, polynomial GCD, CRT and maximal-quotient rational reconstruction.

Both methods accept a callback returning a finite-field value or `None` for an
unusable evaluation. Only the variable map is supplied: no hidden support or degree
information comes from the benchmark's source polynomials. Probes are cached.
Thiele and final results require three successful independent checks by default.
Failed reconstruction attempts restart with new anchors. Resource limits and the
seed are configurable. Inputs must use an odd prime modulus supported by `Zp64`.

`reconstruct_rational_function_over_q` lifts normalized coefficients across primes
above 2^61 and validates the candidate in a fresh prime before returning Symbolica's
integer numerator/denominator representation. This initial implementation relearns
support in each prime; changing support resets the CRT accumulation. Validation is
probabilistic, not a deterministic coefficient-height certificate.

## Reproduce

```sh
cp benches/results/reconstruction/Cargo.lock.snapshot Cargo.lock
cargo test --locked --test rational_reconstruction
cargo build --locked --release --config profile.release.lto=false \
  --config profile.release.codegen-units=16 --example reconstruction_benchmark \
  --example reconstruction_rational_benchmark
RECONSTRUCTION_REPEATS=9 taskset -c 24 target/release/examples/reconstruction_benchmark > benches/results/reconstruction/local.csv
RECONSTRUCTION_REPEATS=9 PROBE_WORK=1000 taskset -c 24 target/release/examples/reconstruction_benchmark > benches/results/reconstruction/expensive.csv
taskset -c 24 target/release/examples/reconstruction_rational_benchmark > benches/results/reconstruction/rational.csv
python3 benches/summarize_reconstruction.py benches/results/reconstruction/local.csv
python3 benches/summarize_reconstruction.py benches/results/reconstruction/rational.csv
```

Choose an available CPU instead of 24 on other machines, or omit `taskset`.
The benchmark defaults to five measured seeds plus an unrecorded warm-up per case
and method. Set `RECONSTRUCTION_REPEATS` to change this; an optional positional
argument filters case names. Method order alternates. The field is 2^61 - 1.
Parsing and source-polynomial construction are outside the timer. The timer includes
all probes, interpolation, random verification, and final GCD normalization.
Every run must also pass an **exact polynomial cross-product identity check**
outside the timed region. Failed runs stop the benchmark rather than being omitted.

`PROBE_WORK` adds dependent modular multiplications to every actual oracle call.
It is a controlled cost model, **not a measured IBP workload**. CSV records that
setting, total elapsed time, probe count, poles, retries, Thiele interpolations,
linear solves, and output support sizes. Timing runs should be made on the same
machine with the same build and options; probes are independent of machine speed.
The measured build uses optimization level 3, 16 code-generation units and no LTO
for both methods, avoiding the long full-LTO development build. Results are not
claimed to be default-profile LTO timings.

The paper's Eq. (28) is included exactly, in orders `(y,d)` and `(d,y)`. Its
published Table 2 lists 509 probes for balanced Zippel and 1424 for homogeneous
scaling. Our stopping checks, sampling reuse and optimizations differ, so these
numbers are a reference, not an assertion about expected identical counts.

## Known limitations and next steps

- The first version used total-degree bounds for Newton interpolation. The
  performance update adds individual-degree and homogeneous-support pruning,
  including removal of completed terms from polynomial Zippel systems. It still
  lacks removal of solved homogeneous coefficients from rational line systems,
  Ben-Or/Tiwari racing, and FireFly 2's hybrid strategy.
- Balanced Zippel now reuses row intersections and learned degrees; optional
  verified denominator separation is available. Variable order can greatly affect
  work. Neither automatic ordering nor selection of the separation mode is implemented.
- There is no asynchronous/batched oracle, vector-valued reconstruction, parallel
  scheduling, automatic variable reordering, or support reuse between primes.
- The initial API fixes 64-bit prime fields and `u16` exponents. The degree limit
  bounds total degree for Cuyt–Lee and individual degrees for balanced Zippel.
- An oracle must return consistent values, and report poles as `None`. Random
  specialization and validation can fail with small probability. Field elements
  returned by the callback must belong to the requested field.

## First-version results (commit `4d2752b`)

On an AMD EPYC 9754, pinned to CPU 24, using Rust 1.92.0 and the build flags above,
the nine-seed per-prime measurements show:

| Case | Cuyt–Lee probes | Balanced probes | Cuyt–Lee ms | Balanced ms | C/B time |
|---|---:|---:|---:|---:|---:|
| Paper Eq. (28), `(y,d)` | 3791 | 538 | 122.09 | 12.68 | 9.63× |
| Paper Eq. (28), `(d,y)` | 3791 | 1027 | 115.39 | 20.31 | 5.68× |
| Dense total degree, three variables | 207 | 234 | 0.548 | 0.804 | 0.68× |
| Dense box, three variables | 649 | 192 | 2.209 | 0.684 | 3.23× |
| Sparse, three variables | 147 | 121 | 0.399 | 0.445 | 0.90× |
| Sparse, four variables | 169 | 102 | 0.345 | 0.253 | 1.36× |

Balanced Zippel is faster on the paper's benchmark and the dense-box case, while
Cuyt–Lee wins for the dense total-degree case. Fewer probes alone do not ensure
lower runtime: sparse3 favors Cuyt–Lee with cheap probes but favors balanced Zippel
by 1.09× after adding 1000 modular multiplications per probe. The reversed paper
variable order nearly doubles balanced Zippel's probe count. These findings favor
keeping both strategies and investigating automatic selection and variable order.

**The 9.63× figure compares the implementations in this worktree.** The baseline's
missing FireFly pruning is significant: its 3791 probes exceed the paper's 1424
homogeneous-scaling reference. These results do not establish a 9.63× advantage
over production FireFly or Kira. All finite-field runs completed on their first
attempt, with exact cross-product checks passing.

Full per-case summaries, including the simpler examples, are in
[`results/reconstruction/local.md`](results/reconstruction/local.md) and
[`results/reconstruction/expensive.md`](results/reconstruction/expensive.md).
Their CSV files preserve every individual run. The dependency lock snapshot,
compiler/CPU details and test output accompany them.

The separate end-to-end Q benchmark includes conversion of the oracle's integer
polynomials to each requested finite field, modular reconstruction, coefficient
lifting, fresh-prime verification and final normalization. Five measured seeds
give:

| Case | Cuyt–Lee probes | Balanced probes | Cuyt–Lee ms | Balanced ms | C/B time | Primes, both |
|---|---:|---:|---:|---:|---:|---:|
| Paper Eq. (3) | 23 | 24 | 0.084 | 0.078 | 1.09× | 2 |
| Paper Eq. (28), `(y,d)` | 11376 | 1617 | 353.16 | 40.04 | 8.82× | 4 |

For Eq. (28), both methods used three CRT images and a fourth, unused prime for
validation; there were no support resets. Every output passed an exact
**integer-polynomial** cross-product identity check. See
[`results/reconstruction/rational.md`](results/reconstruction/rational.md) and
[`results/reconstruction/rational.csv`](results/reconstruction/rational.csv).
This confirms the balanced method's advantage on this example survives coefficient
lifting; it remains a comparison with the current prototype baseline.

## Validation

`cargo test --test rational_reconstruction` passes eight tests covering exact
cross-product identities for the papers' examples, zero/constants, polynomials,
univariate degree imbalance, cancellation, unused variables, variable order,
multiple seeds, generated sparse functions over a smaller prime, poles, bounded
failure, deliberately unlucky anchors, large rational coefficients and recovery
from an unlucky prime during CRT lifting. In the first version, the Q example reconstructs Eq. (3) in
24 probes across two primes.
The ten existing tests in `tests/rational_polynomial.rs` also pass. Each benchmark
checks the reconstructed result exactly outside the timer, including all 324
measured finite-field runs and 20 measured Q runs (plus warm-ups).

An ordinary Clippy run is blocked by the pre-existing `clippy::never_loop` error
in `src/poly/factor.rs:11833`. With that lint allowed on the command line, Clippy
finishes (248 existing library warnings) and reports no diagnostics in the new
reconstruction sources or examples. The existing code was not modified to silence
these diagnostics.

## Performance update

After first-version commit `4d2752b`, Thiele convergents use Symbolica's existing
dense univariate polynomials and Horner evaluation. Cuyt–Lee learns individual
variable degrees from black-box slices. Homogeneity then supplies lower and upper
bounds on each coefficient's next-variable power. Factoring out the lower power
reduces Newton work; completed terms are subtracted from subsequent polynomial
Zippel systems. Univariate inputs go straight to Thiele.

A baseline executable saved before these changes was rerun on the same CPU and
build settings. Nine-seed medians with the original expanded-polynomial oracle:

| Case / method | Before ms | After ms | Before probes | After probes | Speedup |
|---|---:|---:|---:|---:|---:|
| Eq. (28), `(y,d)`, Cuyt–Lee | 106.462 | 37.141 | 3791 | 1365 | 2.87× |
| Eq. (28), `(y,d)`, balanced | 11.753 | 10.720 | 538 | 538 | 1.10× |
| Dense box, Cuyt–Lee | 2.989 | 0.840 | 649 | 228 | 3.56× |
| Sparse3, balanced | 0.361 | 0.270 | 121 | 121 | 1.34× |
| Sparse3, Cuyt–Lee | 0.317 | 0.347 | 147 | 175 | 0.91× |
| Eq. (3), Cuyt–Lee | 0.030 | 0.033 | 20 | 28 | 0.91× |

Degree discovery costs extra probes: it regresses some small or already sparse
Cuyt–Lee cases. Adaptive selection of whether to perform these slices is a useful
next optimization. The baseline profile attributed about 48% of sampled CPU time
to polynomial evaluation and 39% to linear solves on Eq. (28), motivating both
the reduced line count and a separate factored-oracle comparison.

Raw measurements and summaries are under
[`results/reconstruction/improved/`](results/reconstruction/improved/).
The improved expanded-oracle Q benchmark uses 4098 Cuyt–Lee probes versus 11376
in the first version; balanced remains at 1617. Both still use three CRT images
and a fresh verification prime. All 18 targeted tests pass, as do every benchmark's
exact identity checks. A repeated release build now finishes in 0.12 seconds:
`build.rs` resolves Git's actual metadata paths, avoiding the nonexistent
`.git/HEAD` watch that caused complete rebuilds in linked worktrees.

Actual external implementations were also built and benchmarked. With matched
factored black boxes, Eq. (28) takes 3.56 ms for Symbolica balanced Zippel, 15.26 ms
for the FIRE7 balanced-Zippel adapter, and 76.31 ms for default FireFly, per prime.
Over Q, Symbolica balanced takes 11.52 ms and FireFly with factor/shift scans takes
108.30 ms. These measure the stated configurations and synthetic function only;
they do not establish an IBP-workload speedup. See the
[external benchmark report](external/README.md) for pinned sources, complete
timing tables, sampling differences, limitations and reproduction commands.
