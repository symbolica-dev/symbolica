# Joint IBP probes and large rational coefficients

The benchmark now evaluates all selected outputs of an original Ratracer trace
at each counted prime/point pair. Symbolica's first joint driver caches that
vector and runs the existing scalar reconstructor for each coefficient in
trace order. FireFly uses its native multi-output reconstructor. This measures
actual shared evaluations rather than adding independent scalar probe counts.
No source supports, degrees or factors are passed to either reconstructor.

## Pentabubble: every coefficient of one integral

The pinned public `tth2l_b16` setup and target
`basis[2,1,1,1,1,1,0,0,0,0,0]` produce 32 coefficient outputs. The new separately
labeled preparation selects all 32, preserving the earlier four-output input.
Every reference expression passes eight exact rational trace checks, and every
timed reconstruction passes an exact polynomial identity check for every output.

| Outputs | Symbolica vector probes | FireFly scan vector probes | FireFly default vector probes |
|---|---:|---:|---:|
| Earlier four selected coefficients | 2,005 | 6,032 | 10,695 |
| All 32 coefficients of the target | 4,222 | 4,963 | 10,698 |

These probe counts are stable over seeds 1–3. For all 32 outputs, Symbolica
requests 5,484 scalar values, obtains 1,262 from the shared cache, and performs
4,222 actual vector trace evaluations. That is 23.0% fewer evaluations than
its scalar requests and 14.9% fewer than scanned FireFly's joint reconstruction.

Median reconstruction seconds over the three seeds:

| Outputs | Symbolica | FireFly scan | FireFly default |
|---|---:|---:|---:|
| Four | 0.387 | 1.570 | 3.988 |
| All 32 | 0.872 | 1.433 | 3.978 |

FireFly scan improves when additional outputs are included. Its scan and
reconstruction choices depend on the entire output set; a scalar maximum is
not a substitute for an actual joint run. Timings exclude loading, preparation,
serialization and independent exact result checks. Native prime policies
differ and the shared host is not isolated.

The vector cache is currently benchmark infrastructure. It stores all values
at a requested point, including unavailable trace evaluations, until the run
ends. It does not yet provide a public joint reconstruction API or coordinate
the interpolation schedules across coefficients.

## Harder coefficient lifting: the public `tth2l_b25` setup

The eight-propagator target is `basis[2,1,1,1,1,1,1,1,0,0,0]`. The source setup
specializes kinematics to two variables, `lam,d`; the coefficients can contain
hundreds of digits. This stresses rational coefficient lifting despite having
fewer variables. Four outputs are again chosen by optimized single-output
instruction size before measuring reconstruction. All pass eight exact
rational checks against their original traces.

The first two cases exhaust Symbolica and FireFly's default 32-prime budgets.
The final scalar comparison therefore supplies the same 114-prime cap to all
five methods. Every successful result passes an exact identity check.

| Rank | Symbolica | FireFly default | FireFly scan | FIRE7 learned | rare |
|---|---:|---:|---:|---:|---:|
| 1 | 22,771 | 23,564 | 16,107 | 33,177 | Incomplete: 69,609 |
| 2 | 9,960 | 10,112 | 7,476 | 14,886 | 32,901 |
| 3 | 2,368 | 2,414 | 2,033 | 3,604 | 6,799 |
| 4 | 2,621 | 2,683 | 2,038 | 3,897 | 8,165 |

Scanned FireFly uses fewer probes on every case. Symbolica is therefore not
yet best across these complicated IBP benchmarks. On rank 1, Symbolica uses
52 reconstruction images plus a verification prime, with 51 support reuses.
Many early reused images still cost 567 probes. Carrying small polynomial
factors through coefficient lifting, rather than repeatedly lifting expanded
coefficients, is a plausible next optimization; it is not implemented here.

The rare adapter now dispatches over all 114 primes already present in the
pinned upstream table; its algorithm, prime order and default sixteen-prime
budget are unchanged. A 401-digit coefficient control verifies successful
reconstruction beyond sixteen primes. Rank 1 nevertheless exhausts the full
table and is reported as incomplete, not as a successful probe count.

The archive also retains initial default-budget failures and a diagnostic
128-prime run. FIRE7 rejects 128 because its supported maximum is 127; those
configuration errors are superseded by the common 114-prime comparison.

## Validation and remaining work

The shared ABI now accepts named multi-output traces through additional entry
points while preserving its single-output interface. Across four moduli,
joint values agree with separately optimized scalar traces. The checks cover
the four-output and all-output pentabubble sets, `tth2l_b25`, and the preceding
nonplanar two-mass family. Invalid dimensions and unsupported moduli leave the
interpreter counter unchanged.

Driver controls check output-order rejection, exact recovery of asymmetric
constant outputs, actual cache reuse, and one-probe budget/pole accounting.
FireFly reports callback failures from its worker before exiting, preserving
the attempted count. The scalar ABI and driver checks still pass. The rare
adapter passes 34 exact reconstructions, six failure controls and two checker
rejection controls. The Rust example builds and passes clippy with the existing
library allowance and warnings; changed Rust files pass formatting checks.

Repeated joint trace benchmarks for `tth2l_b25` and `xbox2l2m` are running
separately from the completed tables above. Their completion and comparison
are outstanding. The first complete `tth2l_b25` seed uses 31,503 vector probes
and 202.270 seconds for Symbolica, versus 16,107 probes and 130.630 seconds for
FireFly scan. Symbolica saves 6,217 of its 37,720 scalar requests through the
cache, but this does not close the joint reconstruction gap. This is a single
seed observation, not a repeated median.

None of these measurements establishes performance for an
entire IBP reduction table: the largest complete output set here belongs to
one integral.

See the [commands](external/README.md) and
[archived evidence](results/reconstruction/joint-ibp/) for reproduction.
