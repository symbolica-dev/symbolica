# Current Symbolica/FLINT polynomial performance

Version `v17` adds two generic factorization optimizations relative to `v16`. Quadratic
univariate Hensel lifting now keeps the target, factors, Bezout cofactors, and exact residuals in
dense integer storage across all p-adic rounds; the high-height degree-33 factorization changes
from `1.111896 -> 0.987690` S/F. Multivariate `factor_separable` now uses an
operation-scoped coefficient-content context: a monomial coefficient divisor certifies constant
content only when the full polynomial has zero coordinatewise exponent minima, and a speculative
two-layer exact GCD is admitted only when its support-product work proxy is at most one eighth of
the accumulated first stage. PolyBench #105 changes from `1.106606 -> 0.576212` S/F, with broad
gains across the other representative factorization inputs.

The selectors use exact divisibility, exponent minima, support sizes, and a fixed decisive work
margin. They contain no benchmark identity, exact fixture, coefficient fingerprint, or
variable-count exception. Every inconclusive pair probe preserves the existing
`gcd_multiple` fallback.

Version `v17` also adds four generated eight-variable cases with asymmetric 8/45-term cofactors
and dominant 165-, 495-, or 1287-term common factors, including a 256-bit coefficient-height
variant. The three small-height GCD rows are Symbolica wins at `0.445428`, `0.460421`, and
`0.467658` S/F. The new high-height GCD row is the overall worst at `1.631601` S/F and is the
next profiling target. Product construction ranges from `1.327831` to `0.746378` S/F.

Ducos, Brown, and CRT resultant measurements remain a historical appendix note and never
participate in the primary ranks or summary.

The live latest view is [CURRENT_STATUS.md](CURRENT_STATUS.md).

`S/F` is Symbolica's median time divided by FLINT's median time. Lower is better: a value below
`1` means Symbolica is faster. Repeated-process rows use the median of the process-level paired
ratios. Benchmarks are sequential and single-threaded, use release/default features (including
`faster_alloc`), and compare against system FLINT 3.6.0 in the same process. Rows accumulated
over several accepted source snapshots, so comparisons between different rows are descriptive;
use source-matched alternating A/B runs to attribute an optimization.

Detailed optimization decisions, profiles, and rejected experiments are in
[PolynomialPerformanceHandoff.md](PolynomialPerformanceHandoff.md).

## Summary

| Measure | Current result |
|---|---:|
| Primary comparisons | 124 |
| Symbolica faster than FLINT | 107 |
| Symbolica slower than FLINT | 17 |
| Worst primary S/F | **1.631601** — asymmetric high-height eight-variable GCD with an 8-term cofactor and 165-term common factor |
| Worst pre-existing S/F | **1.098172** — generated dense degree-65 factorization |
| Median primary S/F | **0.636158** |
| Best primary S/F | **0.028678** — generated high-gap eight-variable GCD |
| Focused #131 S/F | **0.256988** — Symbolica is 3.89x faster than FLINT |
| PolyBench GCD product median (12 cases) | **0.613690** |
| PolyBench factor product median (11 cases) | **0.617710** |
| All PolyBench product median (23 cases) | **0.617324** |

## Primary comparisons, worst to best

Resultant backends are excluded from this table and its statistics.

| Rank | S/F | Category | Benchmark |
|---:|---:|---|---|
| 1 | 1.631601 | `generated_gcd_operation` | generated GCD auto: high-height 8 variables asymmetric cofactor terms 8/45 common terms 165 degrees 1/2/3 coefficient bits 256 |
| 2 | 1.327830 | `generated_gcd_product` | generated GCD products: dense 8 variables asymmetric cofactor terms 8/45 common terms 165 degrees 1/2/3 |
| 3 | 1.295551 | `generated_gcd_product` | generated GCD products: dense 8 variables asymmetric cofactor terms 8/45 common terms 495 degrees 1/2/4 |
| 4 | 1.174073 | `generated_gcd_product` | generated GCD products: high-height 8 variables asymmetric cofactor terms 8/45 common terms 165 degrees 1/2/3 coefficient bits 256 |
| 5 | 1.098172 | `generated_factor_operation` | generated factorization: dense 1-variable degrees 33/32 total 65 |
| 6 | 1.075107 | `dense_gcd_operation` | configured dense univariate GCD degree 80 |
| 7 | 1.070910 | `generated_factor_product` | generated factor product: dense 1-variable degrees 33/32 total 65 |
| 8 | 1.060731 | `generated_gcd_operation` | generated GCD auto: dense 2 variables degree 5 |
| 9 | 1.056183 | `dedicated_multiplication` | dense large multiplication |
| 10 | 1.055155 | `dedicated_multiplication` | GF(17) dense very large multiplication |
| 11 | 1.054182 | `generated_gcd_product` | generated GCD products: high-height 5 variables degree 4 coefficient bits 512 |
| 12 | 1.049838 | `generated_factor_operation` | generated factorization: dense 1-variable degrees 32/31 |
| 13 | 1.041276 | `generated_factor_operation` | generated factorization: dense 1-variable degrees 33/31 total 64 |
| 14 | 1.029726 | `generated_gcd_product` | generated GCD products: dense 1 variables degree 32 |
| 15 | 1.017696 | `generated_gcd_product` | generated GCD products: high-height 5 variables degree 4 coefficient bits 1024 |
| 16 | 1.017409 | `generated_factor_product` | generated factor product: dense 1-variable degrees 33/31 total 64 |
| 17 | 1.009108 | `generated_factor_product` | generated factor product: dense 3-variable degrees 6/5 |
| 18 | 0.994841 | `generated_gcd_operation` | generated GCD auto: dense 1 variables degree 32 |
| 19 | 0.992282 | `dedicated_multiplication` | GF(18446744073709551557) dense large multiplication |
| 20 | 0.987689 | `generated_factor_operation` | generated factorization: dense high-height 1-variable degrees 17/16 total 33 |
| 21 | 0.986146 | `polybench_factor_operation` | polybench factorization: polybench 8v sharp nontrivial factor #176 |
| 22 | 0.978321 | `polybench_gcd_operation` | polybench GCD: polybench 5v uniform nontrivial GCD #11 |
| 23 | 0.973974 | `generated_factor_product` | generated factor product: dense 1-variable degrees 32/31 |
| 24 | 0.968978 | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform trivial GCD #11 |
| 25 | 0.966693 | `dedicated_multiplication` | dense high multiplication |
| 26 | 0.956438 | `dedicated_multiplication` | dense high large multiplication |
| 27 | 0.940972 | `dedicated_multiplication` | GF(500000003) dense univariate degrees 128/64 accumulator-bound multiplication |
| 28 | 0.940781 | `generated_gcd_product` | generated GCD products: high-height 5 variables degree 4 coefficient bits 256 |
| 29 | 0.913795 | `dedicated_multiplication` | dense small multiplication |
| 30 | 0.913425 | `generated_gcd_product` | generated GCD products: high-height 8 variables degree 3 coefficient bits 256 |
| 31 | 0.902871 | `default_gcd_operation` | GCD auto: dense 7 variables degree 7 |
| 32 | 0.889777 | `dense_gcd_operation` | configured dense univariate GCD degree 48 |
| 33 | 0.878519 | `generated_gcd_product` | generated GCD products: dense 3 variables degree 7 |
| 34 | 0.870516 | `dedicated_multiplication` | GF(17) dense univariate degree-4912 multiplication |
| 35 | 0.858115 | `generated_factor_product` | generated factor product: dense 2-variable degrees 10/9 |
| 36 | 0.857372 | `generated_gcd_product` | generated GCD products: high-height 5 variables degree 4 coefficient bits 128 |
| 37 | 0.852433 | `dense_gcd_operation` | configured dense univariate GCD degree 64 |
| 38 | 0.828262 | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform nontrivial GCD #55 |
| 39 | 0.822914 | `generated_gcd_product` | generated GCD products: high-gap 5 variables degree 5 gap 64 |
| 40 | 0.821527 | `dedicated_multiplication` | GF(17) dense large multiplication |
| 41 | 0.820838 | `generated_gcd_operation` | generated GCD auto: dense 5 variables degree 7 |
| 42 | 0.798104 | `dedicated_multiplication` | GF(65000011) dense univariate degrees 128/64 accumulator-bound multiplication |
| 43 | 0.791465 | `generated_gcd_product` | generated GCD products: sparse 5 variables degree 7 |
| 44 | 0.780017 | `generated_gcd_product` | generated GCD products: sparse 8 variables degree 5 |
| 45 | 0.761777 | `exact_integer_division` | high-height exact division |
| 46 | 0.750365 | `polybench_gcd_product` | polybench GCD products: polybench 8v sharp nontrivial GCD #53 |
| 47 | 0.748808 | `generated_gcd_product` | generated GCD products: dense 5 variables degree 7 |
| 48 | 0.746378 | `generated_gcd_product` | generated GCD products: dense 8 variables asymmetric cofactor terms 8/45 common terms 1287 degrees 1/2/5 |
| 49 | 0.723532 | `polybench_factor_product` | polybench factor product: polybench 8v sharp nontrivial factor #176 |
| 50 | 0.711976 | `polybench_factor_product` | polybench factor product: polybench 8v sharp nontrivial factor #84 |
| 51 | 0.704684 | `polybench_gcd_product` | polybench GCD products: polybench 8v sharp nontrivial GCD #11 |
| 52 | 0.691815 | `dedicated_multiplication` | sparse separated multiplication |
| 53 | 0.688808 | `polybench_gcd_product` | polybench GCD products: polybench 8v sharp nontrivial GCD #35 |
| 54 | 0.688010 | `generated_gcd_operation` | generated GCD auto: dense 8 variables degree 5 |
| 55 | 0.675222 | `polybench_factor_product` | polybench factor product: polybench 8v sharp nontrivial factor #92 |
| 56 | 0.674700 | `polybench_gcd_product` | polybench GCD products: polybench 5v sharp nontrivial GCD #11 |
| 57 | 0.674333 | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform nontrivial GCD #11 |
| 58 | 0.652103 | `polybench_factor_operation` | polybench factorization: polybench 8v sharp nontrivial factor #92 |
| 59 | 0.642318 | `polybench_gcd_operation` | polybench GCD: polybench 8v sharp nontrivial GCD #35 |
| 60 | 0.638505 | `polybench_factor_product` | polybench factor product: polybench 8v uniform nontrivial factor #105 |
| 61 | 0.637854 | `polybench_factor_product` | polybench factor product: polybench 8v sharp nontrivial factor #178 |
| 62 | 0.636796 | `polybench_gcd_product` | polybench GCD products: polybench 8v sharp nontrivial GCD #140 |
| 63 | 0.635520 | `generated_gcd_operation` | generated GCD auto: sparse 5 variables degree 7 |
| 64 | 0.617710 | `polybench_factor_product` | polybench factor product: polybench 8v uniform nontrivial factor #44 |
| 65 | 0.617324 | `polybench_gcd_product` | polybench GCD products: polybench 8v uniform nontrivial GCD #56 |
| 66 | 0.615198 | `polybench_factor_product` | polybench factor product: polybench 5v uniform nontrivial factor #32 |
| 67 | 0.613128 | `polybench_factor_product` | polybench factor product: polybench 5v uniform nontrivial factor #131 |
| 68 | 0.610055 | `polybench_gcd_product` | polybench GCD products: polybench 8v uniform nontrivial GCD #188 |
| 69 | 0.607964 | `polybench_gcd_product` | polybench GCD products: polybench 5v uniform nontrivial GCD #11 |
| 70 | 0.607539 | `dedicated_multiplication` | GF(17) sparse large multiplication |
| 71 | 0.603600 | `polybench_gcd_product` | polybench GCD products: polybench 8v uniform nontrivial GCD #55 |
| 72 | 0.602930 | `polybench_gcd_product` | polybench GCD products: polybench 8v uniform trivial GCD #11 |
| 73 | 0.601913 | `polybench_factor_product` | polybench factor product: polybench 8v uniform nontrivial factor #159 |
| 74 | 0.600192 | `polybench_gcd_product` | polybench GCD products: polybench 8v uniform nontrivial GCD #168 |
| 75 | 0.599923 | `polybench_factor_product` | polybench factor product: polybench 5v uniform nontrivial factor #159 |
| 76 | 0.599179 | `polybench_gcd_product` | polybench GCD products: polybench 8v uniform nontrivial GCD #11 |
| 77 | 0.586252 | `polybench_factor_operation` | polybench factorization: polybench 5v uniform nontrivial factor #159 |
| 78 | 0.586162 | `dedicated_multiplication` | sparse large multiplication |
| 79 | 0.586125 | `polybench_factor_product` | polybench factor product: polybench 5v uniform nontrivial factor #163 |
| 80 | 0.576211 | `polybench_factor_operation` | polybench factorization: polybench 8v uniform nontrivial factor #105 |
| 81 | 0.574252 | `dedicated_multiplication` | GF(18446744073709551557) sparse large multiplication |
| 82 | 0.573810 | `generated_factor_product` | generated factor product: dense high-height 1-variable degrees 17/16 total 33 |
| 83 | 0.554623 | `generated_factor_operation` | generated factorization: dense 3-variable degrees 6/5 |
| 84 | 0.537410 | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform nontrivial GCD #168 |
| 85 | 0.524530 | `generated_gcd_product` | generated GCD products: high-gap 8 variables degree 4 gap 256 |
| 86 | 0.514253 | `polybench_gcd_operation` | polybench GCD: polybench 5v sharp nontrivial GCD #11 |
| 87 | 0.513866 | `dedicated_multiplication` | GF(18446744073709551557) dense very large multiplication |
| 88 | 0.479027 | `polybench_gcd_operation` | polybench GCD: polybench 8v sharp nontrivial GCD #140 |
| 89 | 0.470237 | `generated_gcd_operation` | generated GCD auto: high-height 5 variables degree 4 coefficient bits 1024 |
| 90 | 0.467658 | `generated_gcd_operation` | generated GCD auto: dense 8 variables asymmetric cofactor terms 8/45 common terms 1287 degrees 1/2/5 |
| 91 | 0.460421 | `generated_gcd_operation` | generated GCD auto: dense 8 variables asymmetric cofactor terms 8/45 common terms 495 degrees 1/2/4 |
| 92 | 0.458120 | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform nontrivial GCD #188 |
| 93 | 0.450165 | `generated_gcd_operation` | generated GCD auto: high-height 8 variables degree 3 coefficient bits 256 |
| 94 | 0.449493 | `dedicated_multiplication` | dense very large multiplication |
| 95 | 0.448043 | `dedicated_multiplication` | seven-variable power-minus-one multiplication |
| 96 | 0.447907 | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform nontrivial GCD #56 |
| 97 | 0.445428 | `generated_gcd_operation` | generated GCD auto: dense 8 variables asymmetric cofactor terms 8/45 common terms 165 degrees 1/2/3 |
| 98 | 0.442621 | `exact_integer_division` | dense large exact division |
| 99 | 0.441603 | `exact_integer_division` | dense exact division |
| 100 | 0.437142 | `generated_gcd_operation` | generated GCD auto: high-height 5 variables degree 4 coefficient bits 512 |
| 101 | 0.436875 | `dedicated_multiplication` | GF(18446744073709551557) seven-variable power-minus-one multiplication |
| 102 | 0.434133 | `generated_gcd_operation` | generated GCD auto: high-height 5 variables degree 4 coefficient bits 256 |
| 103 | 0.429881 | `generated_gcd_operation` | generated GCD auto: high-height 5 variables degree 4 coefficient bits 128 |
| 104 | 0.423081 | `default_gcd_product` | GCD products: dense 7 variables degree 7 |
| 105 | 0.421076 | `polybench_factor_operation` | polybench factorization: polybench 8v sharp nontrivial factor #84 |
| 106 | 0.413575 | `dedicated_multiplication` | GF(18446744073709551557) five-variable total-degree multiplication |
| 107 | 0.409963 | `polybench_gcd_operation` | polybench GCD: polybench 8v sharp nontrivial GCD #11 |
| 108 | 0.401382 | `generated_factor_operation` | generated factorization: dense 2-variable degrees 10/9 |
| 109 | 0.390026 | `dedicated_multiplication` | GF(17) seven-variable power-minus-one multiplication |
| 110 | 0.387980 | `polybench_gcd_operation` | polybench GCD: polybench 8v sharp nontrivial GCD #53 |
| 111 | 0.378042 | `dedicated_multiplication` | GF(17) five-variable total-degree multiplication |
| 112 | 0.356182 | `generated_gcd_operation` | generated GCD auto: sparse 8 variables degree 5 |
| 113 | 0.326626 | `generated_gcd_product` | generated GCD products: dense 8 variables degree 5 |
| 114 | 0.289808 | `generated_gcd_product` | generated GCD products: dense 2 variables degree 5 |
| 115 | 0.256987 | `polybench_factor_operation` | polybench factorization: polybench 5v uniform nontrivial factor #131 |
| 116 | 0.252283 | `dedicated_multiplication` | GF(18446744073709551557) dense univariate degree-4912 multiplication |
| 117 | 0.224514 | `polybench_factor_operation` | polybench factorization: polybench 8v sharp nontrivial factor #178 |
| 118 | 0.201668 | `generated_gcd_operation` | generated GCD auto: dense 3 variables degree 7 |
| 119 | 0.164792 | `polybench_factor_operation` | polybench factorization: polybench 5v uniform nontrivial factor #32 |
| 120 | 0.161334 | `polybench_factor_operation` | polybench factorization: polybench 5v uniform nontrivial factor #163 |
| 121 | 0.094995 | `polybench_factor_operation` | polybench factorization: polybench 8v uniform nontrivial factor #159 |
| 122 | 0.073806 | `generated_gcd_operation` | generated GCD auto: high-gap 5 variables degree 5 gap 64 |
| 123 | 0.063262 | `polybench_factor_operation` | polybench factorization: polybench 8v uniform nontrivial factor #44 |
| 124 | 0.028678 | `generated_gcd_operation` | generated GCD auto: high-gap 8 variables degree 4 gap 256 |

## Current high-interest replacements

| Optimization | Earlier/reference S/F | Current S/F | Result |
|---|---:|---:|---|
| Exact monomial coefficient-content certificate, PolyBench 8v #105 | 1.106606 | 0.576212 | Symbolica time falls 47.1% against the exact preceding source and is now 1.74x faster than FLINT |
| Asymmetric 8v GCD benchmark family | new | 0.445428--1.631601 | Small-height dominant common factors favor Symbolica; the 256-bit variant exposes the next worst regime |
| Batched Vandermonde inversion and reusable last-variable images, PolyBench 5v GCD #11 | 1.121593 | 0.978321 | One inversion replaces linear-many finite-field inversions, and repeated images reuse row/power/output storage; Symbolica is now 1.02x faster than FLINT |
| Four-way direct `DenseZp64` accumulation, near-`2^64` dense-large multiplication | 1.137177 | 0.992282 | Source-matched Symbolica median falls 14.58%; the kernel now overlaps four independent carry chains |
| Balanced two-leaf Hensel exact reconstruction, dense degree 65 | 1.171321 | 1.098172 | Symbolica median falls 6.25%; the exact pair leaves `p^51` rather than `p^86` root precision |
| Pre-content Hu main-variable planning, PolyBench 8v sharp GCD #140 / #11 | 1.268240 / 0.748107 | 0.479027 / 0.409963 | Both schedules use 16 rather than 80 images; Symbolica is now 2.09x and 2.44x faster than FLINT |
| Fixed-width compact-simplex integer accumulation, dense 8v / dense 7v | 1.610087 / 1.584134 | 0.326626 / 0.423081 | Symbolica medians are now 3.06x and 2.36x faster than FLINT |
| Chunked mixed-radix dense-five degree-7 multiplication | 1.679650 | 0.748808 | Symbolica median reduced 56.27%; now 1.34x faster than FLINT |
| Cumulative bivariate reconstruction, Wang lifting, dense Montgomery products, and coefficient-content certificates, PolyBench 5v #131 | 1.745716 | 0.256988 | Symbolica median reduced 85.3%; now 3.89x faster than FLINT |
| Dense large-modulus Montgomery products, PolyBench 5v #131 | 0.345860 | 0.276987 | Source-matched Symbolica median reduced 19.90% in commit `f14d182` |
| Bivariate reconstruction, Wang lift fast paths, and coefficient-content certificate, PolyBench 5v #32 | 1.124230 | 0.164792 | Symbolica median reduced 85.3%; now 6.07x faster than FLINT |
| Primitive bivariate dispatch, lazy exact-product exit, and coefficient-content certificates, PolyBench 5v #159 | 0.991235 | 0.586252 | Symbolica is now 1.71x faster than FLINT |
| Wide packed `u16` row merge, high-gap eight-variable degree 4 | 1.654253 | 0.524530 | Symbolica median reduced 68.9%; now 1.91x faster than FLINT |
| Few-row packed merge, sparse eight-variable degree 5 | 1.340679 | 0.552281 | Symbolica median reduced about 58.9%; the consistent full-sweep inventory is 0.780017 S/F |
| Initial primitive bivariate sample, direct reconstruction, and coefficient-content certificate, dense three-variable degrees 6/5 | 1.747643 | 0.554623 | Symbolica median is 68.3% below the original ratio |
| Dense-state quadratic Hensel lifting, high-height degree 33 | 1.997930 | 0.987690 | Symbolica is now slightly faster than FLINT; the exact residual and corrections remain dense across all p-adic rounds |
| Fixed-width contiguous Kronecker packing, dense degree-64 product | 1.094628 | 1.017409 | Source-matched Symbolica median reduced 7.62%; final post-rebase guard is within 1.7% of FLINT |
| Sparse high-gap five-variable product heap advance | 1.040981 | 0.799572 | S/F reduced about 23.2% |
| PolyBench factorization #84 variable order and coefficient-content certificate | 8.690880 | 0.421076 | Current S/F is 20.64x below the original ratio |
| Guarded geometric prime sampling, dense degree-64 factorization | 1.748240 | 1.041276 | Current final-artifact guard is within 4.2% of FLINT |
| Constant-coordinate reconstruction, dense univariate GCD degree 64 | 1.161142 | 0.852433 | Source-matched Symbolica median reduced 26.76%; now 1.17x faster than FLINT |

The high-height degree-33 product itself is `0.573810` S/F and final factorization is `0.987690`,
so both construction and the complete operation are now faster than FLINT.

## Appendix note: historical resultant backend measurements

No resultant measurement participates in the primary scoreboard. For historical algorithm
reference only, these 18 frozen `v5` measurements repeat the same six workloads for three
Symbolica backends.

| Workload | Ducos S/F | Brown S/F | CRT S/F |
|---|---:|---:|---:|
| dense outer degrees 7/6 | 0.519856 | 1.333630 | 2.059289 |
| nonunit leading degrees 9/7 | 0.554473 | 1.124006 | 1.129343 |
| dense outer degrees 10/8 CRT crossover | 0.932951 | 2.464208 | 0.607374 |
| outer-sparse degrees 12/9 CRT crossover | 1.027247 | 1.862295 | 0.432838 |
| large high-height degrees 14/10 | 0.861806 | 2.581685 | 1.249141 |
| lacunary outer degrees 18/11 | 0.570573 | 1.438647 | 2.287301 |

In this frozen `v5` sample, Ducos is faster than FLINT on five of six inputs, with median
`0.716190` S/F. Brown is an explicit reference alternative and is slower on all six; CRT wins two
of six. FLINT's resultant and Symbolica's default both use a Ducos-family recurrence, so the Brown
rows are not default-path performance gaps.

## Versioning and update protocol

`CURRENT_STATUS.md` is the live latest view. Every accepted refresh must also create the next
immutable `CURRENT_STATUS_v<i>.md` snapshot. Never edit an older numbered snapshot. The first
paragraph of each new snapshot must identify the preceding version and summarize every affected
case or family with its old and new S/F values.

After every accepted performance change:

1. Rerun every affected case sequentially with the current source-matched Symbolica/FLINT binary.
2. For repeated processes, replace the row with the median process-level S/F. Do not retain an old
   candidate or control as another current row.
3. Merge the 124 primary non-resultant rows, sort them numerically by S/F in descending order, and
   renumber them. Keep the 18 resultant-backend references in the appendix only.
4. Recompute the summary and verify 124 unique primary workload/category pairs, 18 appendix
   measurements arranged as six workloads by three algorithms, and monotonic descending primary
   ratios.
5. Create the next numbered snapshot with its opening delta paragraph, then make
   `CURRENT_STATUS.md` mirror that latest snapshot.
6. Record benchmark binary hashes, raw timing artifacts, profiles, accepted changes, and rejected
   experiments in [PolynomialPerformanceHandoff.md](PolynomialPerformanceHandoff.md).

The `v17` inventory was programmatically checked for 124 unique primary rows, 107 rows faster than
FLINT, 17 rows slower than FLINT, median `0.636158`, and monotonically descending S/F values. The
18 resultant measurements remain six appendix workloads by three explicit algorithms.
