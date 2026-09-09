# Full-Q FIRE7 comparison and an additional IBP table

The subsequent [multivariate extension](reconstruction-fire7-multivariate.md)
adds FIRE7 comparisons for the amplitude and `nb0` coefficients. This report
retains the two-variable adapter contract and results at its original checkpoint.

This extends the [coefficient-reuse checkpoint](reconstruction-q-support.md)
with a full-Q bivariate FIRE7 adapter and an independent four-variable input
from the public FIRE7 `nb0` IBP table. The Symbolica reconstruction algorithms
are unchanged in this checkpoint.

## FIRE7 adapter contract

`fire7-q-stress` calls the pinned authors' unmodified `thiele`,
`balanced_zippel` (Newton mode), and `rational_reconstruct_multiple` routines.
The adapter provides the sampling driver normally supplied by FIRE's table
workflow. At each prime it learns a univariate skeleton, learns the required
geometric rows, reconstructs those rows and checks the resulting image at three
fresh points. The coefficient-lifting routine combines the reconstructed images
and requires repeated coefficient agreement. A resulting Q candidate must pass
three more points at an unused prime. Finally, an exact FLINT polynomial
cross-product check gates every successful benchmark result outside timing.

`FIRE7_Q` supplies doubling batches starting at eight probes for each row.
`FIRE7_Q_learned` uses the preceding observed Thiele termination count to size
the next batch, retaining these hints across primes. Neither configuration gets
support, degrees or coefficients from the input. Hints affect only batch sizes;
there is no reuse of reconstructed coefficient values across modular images.
All actual oracle calls, unused batch samples and validation calls count.

The input parser, per-prime coefficient reduction, cached-power oracle and exact
Q checker are now shared between the FireFly and FIRE7 C++ adapters. Initial
input parsing and final exact checking are outside timing. FUEL initialization,
modular interpolation, candidate lifting, string conversion and fresh-prime
checks are inside timing. The adapter retains modular images and calls the
authors' lifting routine again after each additional image; this includes the
cost of those unsuccessful intermediate lifting attempts.

FIRE7 uses its native prime table starting at index 1, just below 2^64.
Symbolica starts just above 2^61 and FireFly just below 2^63. FIRE7's lifting
routine normalizes coefficients to the leading numerator coefficient, while
the libraries also have different termination rules. Prime counts therefore
cannot be interpreted as a comparison of CRT implementations alone. These
are adapter and default-prime measurements, not matched-prime results or timings
of a complete FIRE reduction. No claim is made that the sampling driver is the
optimal way to schedule FIRE table reconstruction.

The adapter currently accepts two variables, so the four-variable amplitude and
`nb0` inputs do not receive a FIRE7 comparison. Poles in random sampling and
inconsistent modular support stop this reference adapter; it does not implement
Symbolica's production retry behavior. Probe, elapsed-time and prime limits
produce explicit incomplete outcomes, and the Python runner retains process
failures and timeouts.

## Four-loop Q results

All 12 FIRE7 runs completed and passed exact identity checks. Each configuration
has identical probe counts over seeds 1, 2 and 3:

| Input | Configuration | Probes | Median seconds | Reconstructed images + verification prime |
|---|---|---:|---:|---:|
| Four-loop coefficient | FIRE7 doubling batches | 170,011 | 27.472 | 8 + 1 |
| Four-loop coefficient | FIRE7 learned batches | 108,445 | 18.198 | 8 + 1 |
| Author-modified four-loop coefficient | FIRE7 doubling batches | 170,011 | 27.571 | 8 + 1 |
| Author-modified four-loop coefficient | FIRE7 learned batches | 108,445 | 18.089 | 8 + 1 |

For comparison, the preceding three-seed results on the original coefficient
were 71,019 probes / 10.289 seconds for Symbolica balanced Zippel with coefficient
reuse, 89,757 / 7.172 without reuse, and 71,199 / 30.073 for scanned FireFly.
Those measurements use the same host and compiler settings, but are not
interleaved with the new FIRE7 runs. Timing varied: the first learned FIRE7 run
took 28.153 seconds and the following two took 18.048 and 18.198 seconds.
The probe advantage is stable across the tested seeds; precise runtime ratios
should not be generalized from this shared host.

The learned FIRE7 adapter uses 13,718 probes in the first image and 13,532 in each
of the next seven, plus three fresh-prime checks. In contrast, Symbolica's
coefficient reuse progressively reduces its later-image work, reaching 2,485
probes in the last reconstructed image. The result establishes an advantage
over these measured full-Q FIRE7 adapter configurations on this coefficient.
It does not establish the fastest complete IBP reduction workflow.

Raw [FIRE7 runs](results/reconstruction/fire7-q/fire7.csv), per-process logs,
the [16 control results](results/reconstruction/fire7-q/controls/results.csv),
and the [control check log](results/reconstruction/fire7-q/controls.log) are retained.

## Additional public IBP coefficient

`extract_fire_tables.py` reads `FIRE7/examples/nb0/intsde-nb0.tables` at the same
pinned FIRE7 revision, checks its SHA-256 and selects the longest coefficient
string before any timing. There are 3,540 coefficient entries. The selected
31,460-byte expression is from row `113002000000000000000000000000`, with master
identifier `2127127128128128127127127127`. The variable order `(u,v,w,d)` follows
the accompanying `nb0-de.config`.

This is an additional public IBP table, rather than another transformation of
the earlier four-loop or amplitude input. Selection by string length is a
reproducible size heuristic and does not assert that this is the table's hardest
coefficient. The manifest records the revision, input and output hashes,
selection rule, row and master identifiers. The extracted expression stays
under ignored `target/reconstruction-external`; reconstruction receives only a
finite-field oracle and the variable names.

After canonicalization, the shared exact input contains 858 numerator and 414
denominator terms. All five configurations passed exact checks in the initial
screen and in three further seeds. Counts were identical across those seeds;
the table gives three-seed medians, with all five configurations rotating order
on CPU 25:

| Configuration | Probes | Median milliseconds | Primes |
|---|---:|---:|---:|
| Symbolica balanced Zippel | 2,299 | 43.335 | 2 |
| Symbolica separated Zippel | 1,939 | 36.075 | 2 |
| Symbolica pruned Cuyt–Lee | 12,726 | 275.765 | 2 |
| FireFly default | 18,753 | 2,640.085 | 3 |
| FireFly shift and factor scans | 8,180 | 616.789 | 3 |

The best measured Symbolica configuration uses 76.3% fewer probes than scanned
FireFly on this input. Symbolica needs one reconstructed image plus a fresh
verification prime, so this case does not exercise cross-prime coefficient
reuse. Pruned Cuyt–Lee uses more probes than scanned FireFly, despite taking
less time on this oracle. The preferred method differs from the earlier
amplitude coefficient, where pruned Cuyt–Lee was the better choice. These are
explicit per-case configuration choices; there is no automatic method selector.

The [input manifest](results/reconstruction/fire7-q/nb0-manifest.json),
[initial screen](results/reconstruction/fire7-q/nb0-screen.csv),
[three-seed runs](results/reconstruction/fire7-q/nb0-seeds.csv) and process logs
are retained. [Oracle hashes](results/reconstruction/fire7-q/oracles.sha256)
confirm identical exported inputs across runs, and the two four-loop exports
also match the previous checkpoint's exact inputs.

## Reproduction and validation

```sh
bash benches/external/build_stress.sh
python3 benches/external/check_q_adapters.py
python3 benches/external/extract_fire_tables.py
cargo build --locked --release --config profile.release.lto=false \
  --config profile.release.codegen-units=16 --example reconstruction_stress_benchmark
BENCH_CPU=24 BENCH_TIMEOUT=240 PROCESS_TIMEOUT=480 \
  BENCH_METHODS=FIRE7_Q,FIRE7_Q_learned RECONSTRUCTION_REPEATS=3 \
  python3 benches/external/run_q_stress.py target/reconstruction-external/q-fire7.csv \
  coeff_prop_4l coeff_prop_4l_mod
BENCH_CPU=25 BENCH_TIMEOUT=240 PROCESS_TIMEOUT=480 \
  RECONSTRUCTION_REPEATS=3 \
  BENCH_METHODS=BalancedZippel,BalancedZippelSeparated,CuytLeePruned,FireFly_default,FireFly_scan \
  python3 benches/external/run_q_stress.py target/reconstruction-external/q-nb0.csv \
  fire7_nb0_largest
```

Use the [documented Nix loader settings](external/README.md) if needed, and choose
an allowed CPU or omit affinity. The same pinned FIRE7, FUEL and FireFly revisions
and compiler flags as the previous report apply. Runs use one interpolation
thread on the shared AMD EPYC 9754 host; host load is uncontrolled.

The adapter control script checks all four C++ configurations on Eq. (3), a
coefficient above 64 bits, an origin pole and Eq. (28): 16 exact reconstruction
checks. It also verifies that both adapters report an exhausted probe budget
and an exhausted prime budget, for four additional resource-limit checks.
The extracted coefficient and all large successful results receive exact Q
identity checks. No Symbolica core code changes are included here; the preceding
22-test reconstruction suite remains the core validation checkpoint.

The broader goal remains open: these are scalar coefficients with manually
selected methods. They do not measure complete IBP reductions, shared probes
for vector outputs, batched solver callbacks or a multivariate FIRE7 driver.
Automatic method selection also needs evidence across families because the
existing methods have substantially different costs on the tested inputs.
