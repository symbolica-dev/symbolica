# Automatic variable ordering and original IBP trace oracles

The seven-variable pentabubble benchmark exposed a substantial ordering cost:
the previous automatic method used 24,738 probes on its largest selected
coefficient, while scanned FireFly used 6,032. Symbolica now finds a suitable
separated denominator variable from oracle slices and moves it to the last
internal interpolation position. That case takes 1,136 probes in the original
caller-supplied variable order.

## Reconstruction change

For more than four variables, a dense generic denominator slice can trigger
a search for a nonconstant separated denominator in another variable. The
search is skipped when the last variable already supplies such a separation
or when its denominator support is sparse. These are cost heuristics, not
correctness assumptions. Pure monomial denominators are not candidates: the
existing interpolation already removes them cheaply.

Two independent slices must agree on the candidate's monic denominator. The
search uses the intersections already obtained from the initial survey. It
does not inspect variable names, source expressions, degrees supplied by the
caller, or IBP metadata. After one internal swap, the ordinary selector and
reconstruction methods run again. Existing validation and fallback behavior
remain responsible for exceptional slices.

Raw cache keys and black-box arguments always use caller coordinates. Fresh
verification and sparse-row checks consult that same cache through the
coordinate mapping. The returned polynomial exponents are restored before
normalization and rational coefficient lifting. A new attempt resets the
internal ordering while retaining the original-coordinate cache.

## Additional public families

Both families use the pinned Kira/Ratracer preparation described in the
[preceding report](reconstruction-independent-ibp.md), including its fixed
selection by optimized trace instruction size and exact rational trace checks.
The source is the [public Ratracer benchmark generator](https://github.com/magv/ibp-benchmark/tree/f518de1f4f89cc716a31d5d9a9a9ba0a3b72f458).

| Family | Selected integral | Variable order |
|---|---|---|
| Three-loop massive diamond, `diamond3l` | `basis[1,1,1,1,1,1,1,1,-2]` | `mb2,ma2,d` |
| Pentabubble, `tth2l_b16` | `basis[2,1,1,1,1,1,0,0,0,0,0]` | `x35,mh2,x41,d,x23,x12,x54` |

The diamond coefficients simplify considerably and serve as additional
controls. The pentabubble coefficients have seven variables and exposed the
larger reconstruction cost. Four coefficients per family were selected before
timing. Every reference expression passes eight exact rational checks against
its calculation trace; these provenance checks are probabilistic rather than
symbolic IBP identity certificates.

## Complete Q probe counts

Baseline: `e9e6fe92`. All discovery, interpolation, coefficient lifting and
confirmation calls are included. The primary comparison retains the original
variable order for every implementation.

| Pentabubble rank | Baseline Symbolica | Current Symbolica | FireFly default | FireFly scan | FIRE7 learned | rare |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 24,738 | 1,136 | 10,695 | 6,032 | 21,575 | 76,627 |
| 2 | 1,064 | 212 | 1,429 | 476 | 2,402 | 3,582 |
| 3 | 1,684 | 378 | 2,019 | 1,206 | 4,280 | 8,178 |
| 4 | 1,402 | 380 | 1,765 | 1,243 | 3,441 | 6,569 |

The total falls from 28,888 to 2,106 probes, a 92.7% reduction, and is 76.5%
below scanned FireFly. Counts are stable across seeds 1–3 for Symbolica and
both FireFly settings. The other references use seed 1.

As an ordering diagnostic, swapping `d` to the last position before calling
the previous Symbolica binary gives 1,067/162/323/327 probes. The automatic
search pays the additional discovery cost itself. FireFly scan with that same
external swap uses 6,702/476/1,298/1,387 probes. FIRE7 improves to
6,539/956/1,616/1,533 with the favorable order, and remains above current
Symbolica in the original order.

| Diamond rank | Symbolica | FireFly default | FireFly scan | FIRE7 learned | rare |
|---|---:|---:|---:|---:|---:|
| 1 | 122 | 211 | 251 | 427 | 237 |
| 2 | 129 | 212 | 258 | 439 | 237 |
| 3 | 127 | 214 | 258 | 427 | 237 |
| 4 | 187 | 330 | 355 | 671 | 410 |

The rare adapter now instantiates the unchanged upstream algorithm for one
through eight variables. No dependency versions or prime policies change.
Its previous four-variable limit was an adapter restriction, not evidence
that the scaling method could not handle the new family.

## Measuring with the original calculation trace

The new shared native oracle executes the pinned Ratracer interpreter over
the optimized single-output trace. Symbolica and FireFly dynamically load the
same library and call the same evaluator. The shared runner labels these rows
`ratracer_trace`; expanded-polynomial rows retain `cached_powers_Q`.

The ABI validates the input variable order, supplies zero-padded instruction
memory as required by Ratracer, and counts interpreter calls independently.
Both drivers check their reported probe totals against this counter. Field
conversion preserves canonical integer representatives. Trace loading, Kira
equation generation, elimination, and exact result checking are outside the
timed reconstruction. Each benchmark result is checked by an exact polynomial
identity against the independently validated reference expression.

The interpreter is tested over four moduli, including a small field, at random
points and coordinate zeros. Successful evaluations agree with the expanded
input; failed inverses are reported explicitly. Some intermediate poles are
removable in the final expression. Symbolica treats them as unavailable probes;
FireFly records `trace_pole` if its callback encounters one. Driver fault tests
also verify `trace_error` and preservation of attempted probe counts. Unsupported
64-bit moduli are rejected before entering the interpreter.

For the pentabubble, median seconds over seeds 1–3 with the shared trace oracle:

| Rank | Symbolica | FireFly scan | FireFly default |
|---|---:|---:|---:|
| 1 | 0.201 | 1.325 | 3.383 |
| 2 | 0.018 | 0.059 | 0.180 |
| 3 | 0.028 | 0.130 | 0.235 |
| 4 | 0.025 | 0.122 | 0.191 |

The preceding nonplanar two-mass family also runs against its original trace.
Median seconds over seeds 1–3:

| Nonplanar rank | Symbolica | FireFly scan |
|---|---:|---:|
| 1 | 25.150 | 31.043 |
| 2 | 22.630 | 37.765 |
| 3 | 40.995 | 48.920 |
| 4 | 2.042 | 2.603 |

Both libraries retain their expanded-oracle probe counts on every tested
trace run. Symbolica has the lower observed median on all eight traced cases.

This is a single-threaded reconstruction comparison after loading the trace.
The shared host is not isolated, and native prime policies still differ.
The runtime results therefore do not establish universal speed ratios.
The scalar, single-output comparison also does not measure sharing probes
across multiple coefficients in a complete IBP reduction.

The trace backend currently binds Symbolica and FireFly. FIRE7's native
64-bit primes exceed Ratracer's supported range, and rare has no trace binding
in this adapter. Those combinations receive explicit unsupported statuses;
their expanded-oracle comparisons above still execute the actual libraries.

## Validation and reproduction

Use the [external setup and trace commands](external/README.md). The
[archive](results/reconstruction/ordering-traces/) retains source revisions,
selected coefficient labels, trace and expression hashes, preparation logs,
oracle validation, native-driver controls and raw benchmark results.

All 46 reconstruction tests pass. New tests move the separated variable
through every position in a five-variable function, check exact recovery in
caller coordinates, reject duplicate raw calls, and verify multi-prime lifting
and support reuse after reordering. The extended rare adapter passes 33 exact
reconstruction controls, five failure controls and two checker rejection tests.
Clippy completes with the existing `never_loop` allowance and 249 library
warnings. Changed Rust files pass formatting checks.

The complete `nb0` and `graph5` tables retain all 3,319 and 945 exact successes
and their 505,874 and 20,311 probe totals. The larger modular controls, four-loop
and amplitude inputs, sheared inputs, and preceding independent families retain
their probe counts and available per-prime and selection counters.

All 4,441 final benchmark runs pass exact identity checks. Deliberate budget
and driver-failure controls, plus the superseded pre-extension rare restriction,
are labeled separately in the archive.
