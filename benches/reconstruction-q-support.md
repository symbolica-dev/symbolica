# Reusing coefficients during rational lifting

This extends the [modular monomial-removal comparison](reconstruction-monomials.md)
to complete reconstruction over Q. The benchmark now compares the same exact
integer-coefficient inputs against FireFly both with and without factor scans,
including coefficients from a four-loop propagator and a two-loop
diphoton-plus-jet amplitude. The authors' variants with monomial factors removed
are also included; they are modified versions of those coefficients, not new
physical reductions.

## Why change the lifting stage

At `856aec5`, Symbolica reconstructed every prime image independently. The
four-loop coefficient needed seven reconstructed images: balanced Zippel used
12,822 probes in each, plus three probes in a fresh verification prime, for
89,757 total. The expanded integer coefficients reach 343 bits in the numerator
and 332 bits in the denominator. This exposes work that the earlier one-prime
comparison could not measure.

The new lifting stage records rational reconstruction candidates for each
coefficient after CRT. A candidate `a/b` can be reused when it agrees with the
previous reconstruction or satisfies `abs(a)*b < floor(M/2^32)`, where `M` is the
accumulated CRT modulus and `b` is positive. The size margin avoids waiting for
another full image before trying small coefficients. It is a scheduling
heuristic, not an asserted error probability or a bound on the true coefficient.
These values are probabilistic hypotheses, and every reused image receives fresh oracle
checks. Before expensive remaining interpolation, the first line is also checked
at fresh parameters; this rejects inconsistent known components early. Rejected
hypotheses immediately fall back to the selected ordinary method within
the remaining per-prime probe budget. A support change resets the CRT and the
coefficient hypotheses.

Reusing isolated coefficients alone does not fix the scale of a homogenized
line. The new path waits until at least one entire nonzero homogeneous component
is known. It then evaluates the known numerator and denominator contributions,
solves only for unknown components, and interpolates the remaining known supports
using Symbolica's shifted transposed Vandermonde solver. Components with fewer
unknown terms finish first and are removed from later line solves. No shift is
needed because a known component fixes the normalization; an origin pole is
allowed.

On an ordinary successful attempt, the number of interpolation probes equals
the number of unknown coefficients. First-line validation adds three successful
probes when there is enough remaining interpolation to justify that early check;
final image validation adds three by default. Poles and failed hypotheses count too.
`RationalReconstructionStats` reports `support_reuses` and `support_fallbacks`.
The single-prime reconstruction algorithms are unchanged.

`ReconstructionOptions::reuse_coefficients` defaults to true and applies only to
Q lifting. Set it to false to reconstruct each prime independently, which can
be faster for cheap oracles. The benchmark runner exposes this as
`BalancedZippelNoReuse` or `CuytLeePrunedNoReuse`; these are configuration labels,
not extra reconstruction methods. The Q example also accepts
`RECONSTRUCTION_NO_REUSE=1` directly.

## Benchmark contract

`RECONSTRUCTION_OVER_Q=1` enables the Q mode of
`reconstruction_stress_benchmark`. Its export contains exact integer coefficients,
marked by a zero in the modulus field. Both adapters reduce those coefficients
when the prime changes, then use the same power-table and sparse-term-loop oracle
with their native field arithmetic. Parsing and initial canonicalization are
outside timing; per-prime coefficient conversion, reconstruction, lifting,
normalization and result retrieval are timed. Exact polynomial cross-product
checking is outside timing, using Symbolica or FLINT respectively.

FireFly runs one worker. Its scan configuration enables both shift and factor
scans. Serialization restores factors and internal variable ordering before
the exact Q check, so this comparison includes the full reconstructed function.
The adapter was also checked on a small input with a coefficient larger than
64 bits and a negative denominator coefficient, in both configurations.

The comparison uses **each library's default prime sequence**. Symbolica starts
just above 2^61; FireFly starts just below 2^63 and proceeds downward. The CSV
records the prime policy and the number of probes at every prime. These are
default-configuration end-to-end costs, not a matched-prime claim. Prime counts
also reflect the libraries' different rational-reconstruction and termination
rules. Every successful result must pass the same kind of exact Q identity
check, regardless of its prime sequence or stopping rule.

All inputs come from the pinned `scaling-rec` revision already recorded in the
[external setup](external/README.md); their hashes are in
[the input manifest](external/stress-inputs.sha256). Oracle supports and exact
coefficients remain confined to the benchmark oracle and checker. Reused support
and coefficients inside reconstruction come only from earlier reconstructed
prime images.

## Measured results

The final balanced-Zippel configuration reconstructs the four-loop coefficient in
**71,019 probes**, down **20.88%** from 89,757 and 180 probes below factor-scanned
FireFly. All three seeds have the same counts. These are complete Q results,
including failed candidates and verification primes, with exact identities
checked after reconstruction.

| Four-loop coefficient, three seeds | Probes | Median seconds | Primes |
|---|---:|---:|---:|
| Symbolica balanced Zippel, reuse enabled | 71,019 | 10.289 | 8 |
| Symbolica balanced Zippel, reuse disabled | 89,757 | 7.172 | 8 |
| FireFly, shift and factor scans | 71,199 | 30.073 | 11 |

Reuse enabled and disabled were measured with the same final binary. Reuse saves
probes but increases arithmetic time on this cheap polynomial oracle. The
opt-out is therefore useful. In a simple model that adds a constant extra cost
to every oracle call, the measured tradeoff crosses over at about 166 microseconds
per call. This is a cost model, not a measured IBP-solver result.

The four-input screen below uses seed 1. The method is selected separately for
each case from the two tested methods; there is no automatic selector. The
amplitude needs only one reconstructed image, so its probe count does not change
with this lifting improvement.

| Input | Symbolica method | Symbolica probes | Seconds | FireFly scan probes | Seconds |
|---|---|---:|---:|---:|---:|
| Four-loop propagator coefficient | Balanced Zippel | 71,019 | 10.308 | 71,199 | 30.098 |
| Author-modified four-loop coefficient | Balanced Zippel | 71,019 | 10.278 | 71,199 | 31.045 |
| Diphoton-plus-jet amplitude coefficient | Pruned Cuyt–Lee | 43,925 | 15.961 | 48,755 | 59.865 |
| Author-modified amplitude coefficient | Pruned Cuyt–Lee | 43,923 | 15.586 | 48,627 | 61.068 |

The expanded supports contain 6,671 numerator and 6,051 denominator terms for
the four-loop input, and 23,260 numerator and 7,231 denominator terms for the
amplitude. The unselected methods remain in the raw results: pruned Cuyt–Lee
uses 71,286 four-loop probes, above FireFly's scanned count; balanced Zippel uses
73,159 amplitude probes, also above FireFly. FireFly without scans uses 122,116
four-loop probes and 58,203 amplitude probes. Thus these results establish a
per-case advantage among the measured configurations, not universal dominance.

The small Q controls also improve: Eq. (28) uses 1,119 balanced-Zippel probes
and 829 separated-Zippel probes, versus the previous 1,356 and 921. Eq. (3)
is unchanged. An intermediate policy that waited for repeated coefficient
agreement used 79,683 four-loop probes. A subsequent size-margin policy reached
71,001 but regressed Eq. (28) to 2,464 balanced-Zippel probes. Tiny rational
residues can be incorrect hypotheses for large powers of two at the native
prime sequence. Early line checking and immediate fallback resolve that
regression; their 18 extra four-loop checks are included in the final 71,019.

Runs used a shared AMD EPYC 9754 host with individual processes pinned to CPUs
24 or 25, Rust 1.92 and GCC 15.2. Host load was uncontrolled. Release builds
disabled LTO and used 16 codegen units. FireFly is pinned to
`4ce258e5ace6361513c4bdaac93a247cc0e3fdbb`, with the portability and seed patches
documented in the external setup.

Raw data and logs are in [q-support](results/reconstruction/q-support/):
[final screen](results/reconstruction/q-support/final.csv),
[three-seed comparison](results/reconstruction/q-support/final-seeds.csv),
[FireFly screen](results/reconstruction/q-support/firefly.csv),
[FireFly seeds](results/reconstruction/q-support/firefly-seeds.csv),
[baseline](results/reconstruction/q-support/baseline.csv), and
[small controls](results/reconstruction/q-support/rational.md).
The directory also retains the superseded policies and their small-control
regressions. `baseline-adapter.patch` adds the Q benchmark adapter to `856aec5`;
`stable-only.patch` and `margin-no-line.patch` restore their respective older
policies on this checkpoint. Apply these zero-context patches in isolated
worktrees with `git apply --unidiff-zero`. The current policy's regression tests
are not expected to pass on all superseded policies.
The exported exact inputs matched across runs; their hashes are recorded in
[oracles.sha256](results/reconstruction/q-support/oracles.sha256).

## Reproduction

```sh
bash benches/external/fetch_stress.sh
bash benches/external/build_stress.sh
cargo build --locked --release --config profile.release.lto=false \
  --config profile.release.codegen-units=16 --example reconstruction_stress_benchmark
BENCH_CPU=24 BENCH_TIMEOUT=240 PROCESS_TIMEOUT=480 \
  BENCH_METHODS=BalancedZippel,CuytLeePruned \
  python3 benches/external/run_q_stress.py target/reconstruction-external/q-symbolica.csv \
  coeff_prop_4l aajamp coeff_prop_4l_mod aajamp_mod
BENCH_CPU=25 BENCH_TIMEOUT=300 PROCESS_TIMEOUT=600 \
  BENCH_METHODS=FireFly_default,FireFly_scan \
  python3 benches/external/run_q_stress.py target/reconstruction-external/q-firefly.csv \
  coeff_prop_4l aajamp
```

Use `RECONSTRUCTION_REPEATS=3` for three seeds. Choose an allowed CPU or omit
affinity. Nix hosts may require the previously documented external loader
settings. The default limits are 32 primes, 200,000 probes per reconstructed
Symbolica image and 2,000,000 total oracle calls. The runner stores independent
oracle exports per invocation so concurrent comparisons cannot overwrite them.
Timeouts and probe-limit failures are retained as incomplete outcomes.
For the opt-out comparison, set
`BENCH_METHODS=BalancedZippel,BalancedZippelNoReuse` and use `coeff_prop_4l`.

## Validation and scope

All 22 reconstruction tests pass; the [test log](results/reconstruction/q-support/tests.log)
and [release build log](results/reconstruction/q-support/build.log) are retained.
Clippy completed with the existing `clippy::never_loop` allowance and 249 library
warnings. Changed-file formatting, Python syntax, shell syntax and benchmark
CSV consistency checks also pass. The large-rational-coefficient test now requires
successful support reuse for every reconstruction method. A new test makes a
term disappear in the first two reconstructed primes and reappear later; it
requires a fallback, a support reset, eventual reuse, an exact final identity,
and preservation of the per-prime probe budget. Another three-variable test with
an origin pole requires later primes to use just one interpolation probe and
three independent validation probes.
A separate test deliberately presents a small but incorrect coefficient guess
without changing support: the first residue is 7 while the true coefficient is
one whole prime larger. It requires the hypothesis to fail, ordinary
reconstruction to recover, and the exact result and per-prime budget to hold.
A Q regression enforces the previous Eq. (28) total probe budgets of 1,356 for
balanced Zippel and 921 for separation, including the cost of a rejected early
coefficient hypothesis. Disabling reuse must reproduce those counts exactly.

The results concern scalar coefficient reconstruction. They do not measure full
IBP reductions or vector/batched oracles, and the modified inputs do not replace
the need for additional independent IBP families. Shared-host timing and the
different default prime sequences limit broader conclusions.
The existing FIRE7/Smirnov–Zeng adapter covers modular bivariate reconstruction;
it is not presented here as a full-Q reference for these runs.
