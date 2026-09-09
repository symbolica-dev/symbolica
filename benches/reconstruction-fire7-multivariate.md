# Multivariate FIRE7 reconstruction over Q

The reference adapter now reconstructs the four-variable public amplitude and
`nb0` coefficients through the authors' unchanged FIRE7 routines. This closes
the two-variable limitation of the [previous adapter](reconstruction-fire7-q.md).
Symbolica's implementation remains at the [sparse-row checkpoint](reconstruction-sparse-rows.md).

## Adapter and validation

The sampling driver starts with a univariate skeleton and adds variables one
at a time. At each stage, the preceding reconstructed numerator and denominator
determine the required geometric rows. All preceding variables are evaluated
at powers of independently sampled bases; future variables remain at their
anchors. The driver supplies the corresponding name-to-base map to the authors'
`balanced_zippel` routine in Newton mode. `thiele` reconstructs each row, and
`rational_reconstruct_multiple` lifts the complete modular images over Q.

The pinned FIRE7 interpolation and prime-table sources are unchanged. Default
batches start at eight and double; learned batches retain observed termination
counts by variable across rows and primes. The adapter learns supports and
degrees only from oracle samples. Original expressions are used for evaluation
and independent exact checking, and are not passed to interpolation.

Each modular image passes three fresh full-dimensional checks. A lifted candidate
passes another three checks at an unused prime, followed by an exact FLINT
polynomial cross-product check outside timing. All actual oracle calls count,
including unused batch samples and verification. The adapter uses FIRE7's
native primes just below 2^64; Symbolica and FireFly retain their different native
prime sequences. These results measure the adapter, not an entire FIRE reduction
or an optimal scheduling of its table workflow.

Controls cover one, two, three and four variables, large integer coefficients,
a pole at the origin, and declared variable names `z,a,m,b` whose order differs
from the lexical map order used inside FIRE7. All 28 exact controls pass.
All 16 pre-existing control probe counts and per-prime distributions are
preserved, including the bivariate sampling order. Both adapters report the
expected probe and prime budget exhaustion in four additional controls.
FIRE7 rejects a 17-variable input before reaching upstream fixed 16-entry
exponent buffers. The adapter remains benchmark tooling: unlucky poles or
inconsistent supports fail instead of invoking Symbolica's production retries.

## Four-variable IBP table

The `nb0` input is the same deterministically selected longest coefficient
string from the pinned FIRE7 table, in order `(u,v,w,d)`. Its canonical numerator
has 858 terms and denominator 414. Twelve interleaved runs on CPU 24 cover
seeds 1, 2 and 3. Every run passes exact checking, and each configuration has
the same probe count for all seeds:

| Configuration | Probes | Median milliseconds |
|---|---:|---:|
| Symbolica Automatic | 1,985 | 37.143 |
| FIRE7 learned batches | 6,721 | 208.085 |
| FIRE7 default batches | 10,697 | 321.712 |
| FireFly shift and factor scans | 8,180 | 653.837 |

Symbolica uses 70.5% fewer probes than the learned FIRE7 adapter and 75.7% fewer
than scanned FireFly, including Automatic's selection work. It reconstructs
one image and checks an unused prime. FIRE7 reconstructs two images and checks
a third prime. The learned FIRE7 first and second images use 3,377 and 3,341
probes, respectively, followed by three validation probes.

## Larger amplitude coefficient

The amplitude input has four variables `(x23,x34,x45,x51)`, with 23,260 numerator
and 7,231 denominator terms. Its original and author-modified versions are the
same pinned inputs used in the preceding checkpoint. The table contains fresh
seed-one runs: FIRE7 on CPU 25 and Symbolica/FireFly on CPU 24 during the same
benchmark window. These are individual times, not multi-seed medians.

| Input | Configuration | Probes | Seconds |
|---|---|---:|---:|
| Original | Symbolica Automatic | 43,960 | 15.873 |
| Original | FireFly scans | 48,755 | 56.192 |
| Original | FIRE7 learned batches | 230,621 | 107.944 |
| Original | FIRE7 default batches | 432,201 | 190.129 |
| Modified | Symbolica Automatic | 43,958 | 17.662 |
| Modified | FireFly scans | 48,627 | 58.668 |
| Modified | FIRE7 learned batches | 230,621 | 98.695 |
| Modified | FIRE7 default batches | 432,201 | 192.663 |

Every run completes and passes exact checking. Symbolica's Automatic selector
chooses pruned Cuyt–Lee, and its probe counts match the preceding three-seed
results. The FireFly scan counts also match their prior references. Symbolica
uses about 81% fewer probes than the learned FIRE7 adapter and about 10% fewer
than scanned FireFly on these inputs. FIRE7's learned-batch counts are identical
between original and modified inputs, with two reconstructed images and a
third verification prime.

The result supports Symbolica's advantage on these scalar benchmark inputs.
The oracle evaluates a cached-power expanded rational function; this is not a
measurement of an IBP solver callback, a vector-valued reconstruction sharing
probes between coefficients, or a complete reduction. The modified amplitude
is a related input, not an independent physics problem. The differing native
primes, stopping rules and variable strategies remain part of the comparison.

## Large bivariate regression

A further seed-one run of the original four-loop coefficient reproduces both
previous FIRE7 probe counts exactly: 108,445 with learned batches and 170,011
with default batches. The eight reconstructed images, verification prime and
complete per-prime probe distributions are unchanged. The new vector-valued
cache key and multivariate driver therefore preserve the recorded two-variable
sampling behavior on this large input as well as the small controls.

## Reproduction

Build the pinned external sources using the [external setup](external/README.md).
This extension changes only the adapter and its controls; the Symbolica binary
is the release build at `2f61c8e`. External builds use GCC 15.2, C++17, `-O3`,
FLINT 3.6.0 and the previously pinned FUEL library. The host is a shared AMD
EPYC 9754 with uncontrolled load. CPU affinity and a single interpolation
thread are retained; runtime ratios are less stable than probe counts.

```sh
bash benches/external/build_stress.sh
python3 benches/external/check_q_adapters.py
BENCH_CPU=25 BENCH_TIMEOUT=240 PROCESS_TIMEOUT=480 \
  BENCH_METHODS=FIRE7_Q,FIRE7_Q_learned \
  python3 benches/external/run_q_stress.py \
  target/reconstruction-external/fire7-multivariate.csv \
  fire7_nb0_largest aajamp aajamp_mod
BENCH_CPU=24 RECONSTRUCTION_REPEATS=3 BENCH_TIMEOUT=240 PROCESS_TIMEOUT=480 \
  BENCH_METHODS=Automatic,FIRE7_Q,FIRE7_Q_learned,FireFly_scan \
  python3 benches/external/run_q_stress.py \
  target/reconstruction-external/fire7-multivariate-nb0.csv fire7_nb0_largest
BENCH_CPU=24 BENCH_TIMEOUT=240 PROCESS_TIMEOUT=480 \
  BENCH_METHODS=FIRE7_Q,FIRE7_Q_learned \
  python3 benches/external/run_q_stress.py \
  target/reconstruction-external/fire7-multivariate-fourloop.csv coeff_prop_4l
```

Optional `EXTERNAL_LOADER` and `EXTERNAL_LIBRARY_PATH` retain the documented
Nix loader setup. The callback time limit is 240 seconds and the process limit
480 seconds; the latter also bounds interpolation work between oracle calls.
The default two-million-probe and 32-prime limits remain in place. Failures and
timeouts are retained by the runner rather than omitted from its CSV.

Raw [reference screen](results/reconstruction/fire7-multivariate/screen.csv),
[three-seed nb0 comparison](results/reconstruction/fire7-multivariate/nb0.csv),
[fresh amplitude comparisons](results/reconstruction/fire7-multivariate/amplitude-peers.csv),
[bivariate regression](results/reconstruction/fire7-multivariate/fourloop.csv),
[controls](results/reconstruction/fire7-multivariate/controls/results.csv), and
per-process logs are retained. All 52 successful reconstruction runs pass exact
identity checks; five further controls check resource and dimension limits.
The four [oracle hashes](results/reconstruction/fire7-multivariate/oracles.sha256)
match the preceding checkpoint byte for byte, and all eight pinned source
input hashes pass. The [validation record](results/reconstruction/fire7-multivariate/validation.txt)
records the build, checks and source revisions. Symbolica core code is unchanged;
its preceding 25-test result is linked from the sparse-row report rather than
claimed as a new run here.
