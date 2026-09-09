# Sharing probes during the initial image

Different outputs of an IBP reduction have different rational degrees. Previously,
each row consumed samples from the same random stream as subsequent geometric
anchors. Reconstructing a longer row moved every later anchor, preventing reuse
of vector evaluations even when the outputs used the same variable order.

Thiele interpolation now consumes one seed from the enclosing stream and draws
its samples from a separate stream. Interpolation at learned degrees derives its
stream from the reconstruction seed, variable index and geometric row point.
Different row lengths and different numbers of preceding rows therefore do not
move subsequent geometric anchors. Sparse-row verification retains its local
freshness checks, and full-dimensional verification uses the enclosing stream
and rejects previously cached points. The enclosing stream is restored on errors
as well as success.

The change does not reduce a scalar reconstruction's required interpolation
equations. It makes prefixes of those equations reusable between outputs through
the existing vector-value cache. Both successful and pole evaluations still count
as actual interpreter calls.

All 50 reconstruction tests pass. The new test checks exact reconstruction and
sharing between different three-variable output degrees, in both output orders
and at two seeds. Existing pole, unlucky-specialization and budget tests remain
in the same passing suite. All 4,325 scalar and modular regression runs succeed
exactly and retain their previous probe counts. Formatting checks pass; Clippy
completes with 250 library warnings, including the existing 249 warnings and an
argument-count warning on the row wrapper's matching internal signature.

## Selected-output results

Both four-output controls pass exact identity checks at seed 1:

| Target | Previous Symbolica | Current Symbolica | FireFly scan reference |
|---|---:|---:|---:|
| `xbox2l2m` | 38,794 | 19,718 | 26,176 |
| `tth2l_b25` | 14,694 | 14,248 | 16,107 |

The nonplanar reduction is 49.2% relative to the preceding Symbolica version
and 24.7% relative to scanned FireFly. Symbolica serves 20,073 of 39,791 scalar
requests from the vector cache. Its first prime costs 19,715 evaluations and
the verification prime costs three.

The selected `b25` first image falls from 1,024 to 578 evaluations. Later
images retain the preceding factor-lifting implementation and its 13,670
joint evaluations. The first observed reconstruction times are 113.266 seconds
for `xbox2l2m` and 97.008 seconds for `b25`; these selected-output follow-ups
are individual observations, not repeated timing medians.

The full 32-output seven-variable `tth2l_b16` control falls from 4,222 to 1,580
joint evaluations, stable over seeds 1–3. The preceding scanned FireFly
comparison takes 4,963 evaluations.

## Expanded benchmark

The nonplanar `xbox2l2m` target now includes all 71 outputs of
`basis[1,1,1,1,1,1,1,-2,0]`, in the public variable order `t s mb2 d`.
The preceding comparison selected four expensive outputs. The new selection
includes every output before either reconstruction starts.

Preparation uses the same pinned Kira/Ratracer tools as the
[preceding joint benchmark](reconstruction-joint-ibp.md). All outputs pass eight
exact rational input checks. The joint trace also agrees with the separately
optimized scalar traces at 97 finite vectors; 35 other attempts encounter
intermediate poles, for 132 counted interpreter calls over four prime fields.
Its SHA-256 is
`ea869bdca82c40af52703b29c3bd6b18b6c52d18385ec52559fc2676672b3686`.

The full target still concerns one integral, not an entire reduction table.
Reconstruction drivers independently check all resulting rational functions by
exact identity. Timings exclude preparation, trace loading and that final exact
comparison, and run on a shared host with different native prime policies.

Both methods reconstruct every output exactly at seeds 1–3. Median probe
counts for the full-output comparisons are:

| Target | Outputs | Previous Symbolica | Current Symbolica | FireFly scan |
|---|---:|---:|---:|---:|
| `xbox2l2m` | 71 | Not previously measured | 20,264 | 26,503 |
| `tth2l_b25` | 147 | 18,491 | 14,383 | 16,107 |

Symbolica uses 23.5% fewer evaluations than scanned FireFly on the nonplanar
target and 10.7% fewer on the high-coefficient-height target. The full `b25`
first image now takes 578 evaluations, down from 4,686. That is exactly the
same first-image cost as its four selected outputs: all additional outputs
reuse those evaluations. Later images cost 13,805, unchanged from the preceding
factor-lifting version.

The nonplanar vector cache serves 76,559 of 96,823 scalar requests, and retains
all 71 exact outputs. Its total is only 546 evaluations above the selected
four-output run. The full target's first reconstruction image costs 20,261
evaluations, followed by three at its verification prime.

Across nonplanar seeds 1–3, Symbolica takes 20,264, 20,264 and 20,262 probes;
FireFly takes 26,503 each time. Median reconstruction times are 137.630 and
189.584 seconds respectively, a 27.4% reduction for Symbolica on this shared
host. The cache and per-prime detail above refer to seed 1.

All three `b25` seeds take 14,383 probes for Symbolica and 16,107 for FireFly.
Their median reconstruction times are 146.598 and
195.458 seconds respectively.

These results close the two joint probe-count gaps identified by the preceding
checkpoint. They establish comparisons on these measured targets, rather than
a claim that one algorithm wins for every rational function. The authors'
Smirnov–Zeng implementation is covered by the scalar FIRE7 adapter comparison
below; the joint benchmark uses the actual FireFly vector reconstructor.

## Current comparison with FIRE7

A fresh seed-1 scalar follow-up runs all three actual implementations on the
same four selected `b25` rational functions, using expanded polynomial oracles
and a common 114-prime cap. Every output passes an exact identity check:

| Rank | Symbolica probes | FireFly scan probes | FIRE7 learned probes |
|---|---:|---:|---:|
| 1 | 14,242 | 16,107 | 33,177 |
| 2 | 6,027 | 7,476 | 14,886 |
| 3 | 1,727 | 2,033 | 3,604 |
| 4 | 1,778 | 2,038 | 3,897 |

For rank 1, the observed reconstruction times are 2.502, 9.961 and 17.687
seconds respectively. These scalar timings use cheaper expanded oracles and
must not be compared directly with the joint trace timings above. They are
single observations; the repeated scalar probe counts from the preceding
checkpoint agree with this follow-up. The FIRE7 adapter invokes the authors'
Smirnov–Zeng routines with learned batch sizes; it is not a full FIRE reduction.

Kira supplies the IBP systems and traces used here, while the measured
reconstruction reference is its FireFly backend. Complete Kira and FIRE
reduction runtimes, including equation generation and solving, remain
unmeasured by this comparison.

The shared vector-value cache is currently in the benchmark driver, which
invokes Symbolica's scalar reconstruction API for each output. The row-stream
changes are in the library, but a public joint reconstruction API and integration
into a complete IBP solver remain future work.

## Reproduction

After setting up the pinned external tools and their loader environment:

```sh
python3 benches/external/prepare_ibp_benchmark.py xbox2l2m \
  'basis[1,1,1,1,1,1,1,-2,0]' --count 71 --label xbox2l2m_all
python3 benches/external/validate_ibp_benchmark.py xbox2l2m_all
python3 benches/external/check_joint_trace.py xbox2l2m_all
MAX_PRIMES=114 RECONSTRUCTION_REPEATS=3 \
  JOINT_METHODS=Symbolica_joint_cache,FireFly_joint_scan \
  python3 benches/external/run_joint_stress.py xbox2l2m_all \
  target/reconstruction-external/row-stream-xbox-all.csv
```

Use family `tth2l_b25_all` for the previously prepared 147-output control.
`run_row_stream_regressions.py` replays the preceding 4,325 scalar and modular
runs and records exact outcomes and every probe-count change. Its inputs and
baseline CSV files are the preceding factor-lifting checkpoint's prepared data.

The [archived results](results/reconstruction/row-streams/) retain all 4,354
completed benchmark rows, raw logs, the new full-target inputs, and hashes of
the source and measured executables.
