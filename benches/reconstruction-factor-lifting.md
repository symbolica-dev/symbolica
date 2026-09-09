# Retaining factors during rational coefficient lifting

The [following checkpoint](reconstruction-row-streams.md) addresses initial-image
sharing and expands the nonplanar comparison to all 71 outputs of its target.

The preceding `tth2l_b25` benchmark exposed a large repeated lifting cost.
Symbolica reconstructed the expanded numerator and denominator coefficients
at every prime, while scanned FireFly could retain factors. This update
uses Symbolica's polynomial content and exact division operations to retain
small univariate factors across prime fields.

## Algorithm

When coefficient reconstruction is still unresolved after three successful
images, the reconstructor examines univariate contents of that learned image.
It considers functions with at least two variables and 32 terms, ignores pure
monomial contents, and proposes rational factor coefficients whose numerator
times denominator magnitude is below `2^32`. The proposed integer factors
must reduce the learned support by at least 25%. These are cost heuristics,
not source information or correctness certificates.

The same factors must divide another prime image checked with fresh oracle values.
An unlucky factor that appears only at the discovery prime is rejected.
Upon confirmation, the previous and current quotient images seed a new CRT;
later oracle values are multiplied by the denominator factor and divided by
the numerator factor. The factors are restored before validating a rational
candidate against the original black box at an unused prime.

`ReconstructionOptions::reuse_rational_factors` controls this behavior and
requires coefficient reuse. The statistics distinguish proposed hypotheses
from confirmed reductions. The Q runner's `AutomaticNoFactorReuse` method
isolates this optimization while retaining ordinary coefficient reuse.

Support interpolation also gives each directional line its own sample stream,
derived from the reconstruction seed and line index. A shorter preceding line
therefore does not move the sample sequence on later lines. Different outputs
can reuse prefixes of vector-oracle evaluations even when their coefficient
counts differ. Fresh line and whole-image checks use the independent enclosing
random stream and still reject points already present in the local cache.

## Scalar comparison

All four selected `tth2l_b25` coefficients pass exact identity checks across
seeds 1–3, with stable counts. All drivers receive `MAX_PRIMES=114`, retaining
their native prime policies and accounting. Reference counts come
from the [joint IBP checkpoint](reconstruction-joint-ibp.md).

| Rank | Factor reuse disabled | Current Symbolica | FireFly scan |
|---|---:|---:|---:|
| 1 | 22,771 | 14,242 | 16,107 |
| 2 | 9,960 | 6,027 | 7,476 |
| 3 | 2,368 | 1,727 | 2,033 |
| 4 | 2,621 | 1,778 | 2,038 |

The paired `AutomaticNoFactorReuse` runs reproduce the previous Symbolica
counts. The four-case total falls from 37,720 to 23,774 probes, a 37.0%
reduction, and is 14.0% below scanned FireFly.

For rank 1, ordinary reused images initially cost 567 probes. After confirming
the factors, the corresponding quotient images cost 270. All discovery and
confirmation work is included in these totals.

## Joint reconstruction

All four selected outputs together now take 14,694 actual trace evaluations,
down from 31,503. FireFly scan takes 16,107. Counts are stable over seeds 1–3,
and every result passes an exact identity check. Symbolica serves 9,080 of its
23,774 scalar requests from the shared cache. Median seconds are 109.202 for
Symbolica and 139.861 for FireFly scan on the shared host.

The separately prepared full target contains 147 outputs. All pass eight
exact rational provenance checks, and both drivers reconstruct all 147 exactly.
The first full-target seed gives:

| Method | Vector trace evaluations | Reconstruction seconds |
|---|---:|---:|
| Symbolica | 18,491 | 203.033 |
| FireFly scan | 16,107 | 259.318 |

Symbolica is faster in this observation but still uses 14.8% more probes.
This is one seed, not a repeated median. Timings exclude preparation, loading
and exact result checking. Native prime policies differ; the host is not
isolated. Counts refer to actual joint evaluations, not sums of scalar runs.

The runners now retain private executable snapshots; the joint runner also
snapshots its interpreter library. New joint driver diagnostics count probes
by prime and check their sum against the independent interpreter counter.

## Validation

All 49 reconstruction tests pass. New tests cover exact restoration after
factored lifting of a 221-digit coefficient, rejecting an unlucky-prime factor,
and sharing samples between outputs with different polynomial degrees.
Clippy completes with the existing allowance and 249 library warnings.

All 4,325 preceding regression runs retain exact success and their probe,
prime, image and available selection counters. This includes the complete
3,319-entry `nb0` and 945-entry `graph5` tables, the large modular and Q inputs,
sheared inputs and four preceding independent families. The paired new scalar
comparison adds 24 exact successes. Joint driver controls cover output order,
cache accounting, budget limits, poles and interpreter errors.

## Remaining comparison scope

The first nonplanar `xbox2l2m` joint seed uses 38,794 probes for the previous
Symbolica driver and 26,176 for FireFly scan. These coefficients need only one
reconstruction image, so the new lifting changes do not address that gap.
Sharing evaluations during the initial interpolation remains outstanding.

Per-prime diagnostics confirm where additional sharing is needed:

| Symbolica output set | First prime | All later primes |
|---|---:|---:|
| `tth2l_b25`, four selected outputs | 1,024 | 13,670 |
| `tth2l_b25`, all 147 outputs | 4,686 | 13,805 |
| `xbox2l2m`, four selected outputs | 38,791 | 3 |

Of the 3,797 additional evaluations needed for the full `tth2l_b25` target,
3,662 occur in the initial image. Sharing its initial interpolation is the
next target, along with the nonplanar first-image gap. The full 32-output
pentabubble control still takes 4,222 evaluations.

These full output sets belong to individual integrals, rather than an entire
IBP reduction table. The goal of outperforming the references across these
complicated IBP benchmarks is not yet established.

See the [reproduction commands](external/README.md) and
[archived results](results/reconstruction/rational-factors/).
