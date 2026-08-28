# Current Symbolica/FLINT polynomial performance

Version `v5` records commit `843ce58` relative to `v4`. Five-to-eight-variable bounded row
multiplication can now represent each exponent with 16 bits across a two-word `u128` key. This
removes per-pair exponent-vector allocation and hashing when a small row heap has output degrees
above 255. The former worst canonical row, generated high-gap eight-variable degree-4
multiplication, changes `2.019693 -> 0.524530` S/F: the exact final 5,000-sample median is
`0.200774 ms` versus the FLINT median of `0.382769 ms`. Against the frozen source-matched control,
Symbolica changes `0.645276 -> 0.200774 ms`, a 68.9% reduction, while S/F changes
`1.654253 -> 0.524530`. A noncanonical degree-5 stress variant independently changes
`1.561459 -> 0.475039` S/F (`1.854737 -> 0.503020 ms` for Symbolica).

The selector is restricted to polynomial exponents representable both in 16 bits and in the
selected exponent type, five through eight variables, and the existing bounded few-row work
budget. Tests cover the exact canonical fixture, both operand orders, five- and eight-variable
layouts, 65,535 and 65,536 boundaries, all four low-word lanes at their maximum sum,
`u8`/`i16` rejection, collisions, cancellation, and a finite-field image. The next canonical
worst, PolyBench five-variable uniform factorization `#131`, was refreshed on the exact final
binary from `1.756528` to `1.745716` S/F (`81.725573 ms` versus `46.814936 ms`); this timing
movement is not attributed to the multiplication change and it is the mandatory next target.

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
| All current comparisons | 134 |
| Canonical comparisons | 122 |
| Duplicate Brown/CRT comparisons | 12 |
| Symbolica faster than FLINT, all rows | 92 |
| Symbolica slower than FLINT, all rows | 42 |
| Worst S/F across all rows | **2.581685** — Brown high-height resultant |
| Best S/F across all rows | **0.028678** — high-gap eight-variable GCD |
| Worst canonical S/F | **1.745716** — PolyBench 5v uniform factorization #131 |
| Median canonical S/F | **0.688690** |
| Best canonical S/F | **0.028678** — high-gap eight-variable GCD |
| Median across all 134 rows | **0.717754** |
| PolyBench GCD product median (12 cases) | **0.613690** |
| PolyBench factor product median (11 cases) | **0.617710** |
| All PolyBench product median (23 cases) | **0.617324** |

The canonical inventory contains one current implementation per workload. For resultants that is
the main Ducos entry point. Brown and CRT measurements duplicate those six inputs and are marked
`alternative`; they appear in the globally sorted table but do not alter the 122-row canonical
summary.

## All current comparisons, worst to best

| Rank | S/F | Inventory | Category | Benchmark |
|---:|---:|---|---|---|
| 1 | 2.581685 | `alternative` | `brown_resultant_alternative` | resultant Brown: large high-height degrees 14/10 |
| 2 | 2.464208 | `alternative` | `brown_resultant_alternative` | resultant Brown: dense outer degrees 10/8 CRT crossover |
| 3 | 2.287301 | `alternative` | `crt_resultant_alternative` | resultant CRT: lacunary outer degrees 18/11 |
| 4 | 2.059289 | `alternative` | `crt_resultant_alternative` | resultant CRT: dense outer degrees 7/6 |
| 5 | 1.862295 | `alternative` | `brown_resultant_alternative` | resultant Brown: outer-sparse degrees 12/9 CRT crossover |
| 6 | 1.745716 | `canonical` | `polybench_factor_operation` | polybench factorization: polybench 5v uniform nontrivial factor #131 |
| 7 | 1.706529 | `canonical` | `generated_gcd_product` | generated GCD products: dense 5 variables degree 7 |
| 8 | 1.610087 | `canonical` | `generated_gcd_product` | generated GCD products: dense 8 variables degree 5 |
| 9 | 1.584134 | `canonical` | `default_gcd_product` | GCD products: dense 7 variables degree 7 |
| 10 | 1.576901 | `canonical` | `generated_gcd_product` | generated GCD products: high-height 5 variables degree 4 coefficient bits 128 |
| 11 | 1.438647 | `alternative` | `brown_resultant_alternative` | resultant Brown: lacunary outer degrees 18/11 |
| 12 | 1.348957 | `canonical` | `dedicated_multiplication` | seven-variable power-minus-one multiplication |
| 13 | 1.333630 | `alternative` | `brown_resultant_alternative` | resultant Brown: dense outer degrees 7/6 |
| 14 | 1.296474 | `canonical` | `generated_factor_operation` | generated factorization: dense 1-variable degrees 33/31 total 64 |
| 15 | 1.268240 | `canonical` | `polybench_gcd_operation` | polybench GCD: polybench 8v sharp nontrivial GCD #140 |
| 16 | 1.249141 | `alternative` | `crt_resultant_alternative` | resultant CRT: large high-height degrees 14/10 |
| 17 | 1.246310 | `canonical` | `generated_gcd_product` | generated GCD products: high-height 5 variables degree 4 coefficient bits 256 |
| 18 | 1.234835 | `canonical` | `dense_gcd_operation` | configured dense univariate GCD degree 80 |
| 19 | 1.184942 | `canonical` | `generated_factor_product` | generated factor product: dense 1-variable degrees 33/32 total 65 |
| 20 | 1.181340 | `canonical` | `dense_gcd_operation` | configured dense univariate GCD degree 64 |
| 21 | 1.164159 | `canonical` | `generated_factor_operation` | generated factorization: dense 1-variable degrees 33/32 total 65 |
| 22 | 1.137177 | `canonical` | `dedicated_multiplication` | GF(18446744073709551557) dense large multiplication |
| 23 | 1.129343 | `alternative` | `crt_resultant_alternative` | resultant CRT: nonunit leading degrees 9/7 |
| 24 | 1.124230 | `canonical` | `polybench_factor_operation` | polybench factorization: polybench 5v uniform nontrivial factor #32 |
| 25 | 1.124006 | `alternative` | `brown_resultant_alternative` | resultant Brown: nonunit leading degrees 9/7 |
| 26 | 1.112701 | `canonical` | `generated_factor_operation` | generated factorization: dense high-height 1-variable degrees 17/16 total 33 |
| 27 | 1.106606 | `canonical` | `polybench_factor_operation` | polybench factorization: polybench 8v uniform nontrivial factor #105 |
| 28 | 1.097676 | `canonical` | `polybench_gcd_operation` | polybench GCD: polybench 5v uniform nontrivial GCD #11 |
| 29 | 1.094628 | `canonical` | `generated_factor_product` | generated factor product: dense 1-variable degrees 33/31 total 64 |
| 30 | 1.072266 | `canonical` | `dense_gcd_operation` | configured dense univariate GCD degree 48 |
| 31 | 1.060731 | `canonical` | `generated_gcd_operation` | generated GCD auto: dense 2 variables degree 5 |
| 32 | 1.059355 | `canonical` | `generated_gcd_product` | generated GCD products: high-height 5 variables degree 4 coefficient bits 512 |
| 33 | 1.056183 | `canonical` | `dedicated_multiplication` | dense large multiplication |
| 34 | 1.055155 | `canonical` | `dedicated_multiplication` | GF(17) dense very large multiplication |
| 35 | 1.054112 | `canonical` | `generated_gcd_product` | generated GCD products: high-height 8 variables degree 3 coefficient bits 256 |
| 36 | 1.034406 | `canonical` | `generated_factor_product` | generated factor product: dense 1-variable degrees 32/31 |
| 37 | 1.029727 | `canonical` | `generated_factor_operation` | generated factorization: dense 1-variable degrees 32/31 |
| 38 | 1.029726 | `canonical` | `generated_gcd_product` | generated GCD products: dense 1 variables degree 32 |
| 39 | 1.027247 | `canonical` | `main_ducos_resultant` | resultant Ducos: outer-sparse degrees 12/9 CRT crossover |
| 40 | 1.019191 | `canonical` | `generated_gcd_product` | generated GCD products: high-height 5 variables degree 4 coefficient bits 1024 |
| 41 | 1.009108 | `canonical` | `generated_factor_product` | generated factor product: dense 3-variable degrees 6/5 |
| 42 | 1.005858 | `canonical` | `polybench_factor_operation` | polybench factorization: polybench 8v sharp nontrivial factor #176 |
| 43 | 0.994841 | `canonical` | `generated_gcd_operation` | generated GCD auto: dense 1 variables degree 32 |
| 44 | 0.982263 | `canonical` | `polybench_factor_operation` | polybench factorization: polybench 5v uniform nontrivial factor #159 |
| 45 | 0.968978 | `canonical` | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform trivial GCD #11 |
| 46 | 0.966693 | `canonical` | `dedicated_multiplication` | dense high multiplication |
| 47 | 0.956438 | `canonical` | `dedicated_multiplication` | dense high large multiplication |
| 48 | 0.940972 | `canonical` | `dedicated_multiplication` | GF(500000003) dense univariate degrees 128/64 accumulator-bound multiplication |
| 49 | 0.932951 | `canonical` | `main_ducos_resultant` | resultant Ducos: dense outer degrees 10/8 CRT crossover |
| 50 | 0.913795 | `canonical` | `dedicated_multiplication` | dense small multiplication |
| 51 | 0.902871 | `canonical` | `default_gcd_operation` | GCD auto: dense 7 variables degree 7 |
| 52 | 0.878519 | `canonical` | `generated_gcd_product` | generated GCD products: dense 3 variables degree 7 |
| 53 | 0.870516 | `canonical` | `dedicated_multiplication` | GF(17) dense univariate degree-4912 multiplication |
| 54 | 0.861806 | `canonical` | `main_ducos_resultant` | resultant Ducos: large high-height degrees 14/10 |
| 55 | 0.858115 | `canonical` | `generated_factor_product` | generated factor product: dense 2-variable degrees 10/9 |
| 56 | 0.836402 | `canonical` | `polybench_factor_operation` | polybench factorization: polybench 8v sharp nontrivial factor #92 |
| 57 | 0.828262 | `canonical` | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform nontrivial GCD #55 |
| 58 | 0.822914 | `canonical` | `generated_gcd_product` | generated GCD products: high-gap 5 variables degree 5 gap 64 |
| 59 | 0.821527 | `canonical` | `dedicated_multiplication` | GF(17) dense large multiplication |
| 60 | 0.798104 | `canonical` | `dedicated_multiplication` | GF(65000011) dense univariate degrees 128/64 accumulator-bound multiplication |
| 61 | 0.791465 | `canonical` | `generated_gcd_product` | generated GCD products: sparse 5 variables degree 7 |
| 62 | 0.780017 | `canonical` | `generated_gcd_product` | generated GCD products: sparse 8 variables degree 5 |
| 63 | 0.761777 | `canonical` | `exact_integer_division` | high-height exact division |
| 64 | 0.750982 | `canonical` | `generated_gcd_operation` | generated GCD auto: dense 5 variables degree 7 |
| 65 | 0.750365 | `canonical` | `polybench_gcd_product` | polybench GCD products: polybench 8v sharp nontrivial GCD #53 |
| 66 | 0.748107 | `canonical` | `polybench_gcd_operation` | polybench GCD: polybench 8v sharp nontrivial GCD #11 |
| 67 | 0.723532 | `canonical` | `polybench_factor_product` | polybench factor product: polybench 8v sharp nontrivial factor #176 |
| 68 | 0.711976 | `canonical` | `polybench_factor_product` | polybench factor product: polybench 8v sharp nontrivial factor #84 |
| 69 | 0.704684 | `canonical` | `polybench_gcd_product` | polybench GCD products: polybench 8v sharp nontrivial GCD #11 |
| 70 | 0.691815 | `canonical` | `dedicated_multiplication` | sparse separated multiplication |
| 71 | 0.688808 | `canonical` | `polybench_gcd_product` | polybench GCD products: polybench 8v sharp nontrivial GCD #35 |
| 72 | 0.688572 | `canonical` | `polybench_factor_operation` | polybench factorization: polybench 8v sharp nontrivial factor #84 |
| 73 | 0.675222 | `canonical` | `polybench_factor_product` | polybench factor product: polybench 8v sharp nontrivial factor #92 |
| 74 | 0.674700 | `canonical` | `polybench_gcd_product` | polybench GCD products: polybench 5v sharp nontrivial GCD #11 |
| 75 | 0.674333 | `canonical` | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform nontrivial GCD #11 |
| 76 | 0.672529 | `canonical` | `generated_factor_operation` | generated factorization: dense 3-variable degrees 6/5 |
| 77 | 0.642318 | `canonical` | `polybench_gcd_operation` | polybench GCD: polybench 8v sharp nontrivial GCD #35 |
| 78 | 0.638505 | `canonical` | `polybench_factor_product` | polybench factor product: polybench 8v uniform nontrivial factor #105 |
| 79 | 0.637854 | `canonical` | `polybench_factor_product` | polybench factor product: polybench 8v sharp nontrivial factor #178 |
| 80 | 0.636796 | `canonical` | `polybench_gcd_product` | polybench GCD products: polybench 8v sharp nontrivial GCD #140 |
| 81 | 0.635520 | `canonical` | `generated_gcd_operation` | generated GCD auto: sparse 5 variables degree 7 |
| 82 | 0.617710 | `canonical` | `polybench_factor_product` | polybench factor product: polybench 8v uniform nontrivial factor #44 |
| 83 | 0.617324 | `canonical` | `polybench_gcd_product` | polybench GCD products: polybench 8v uniform nontrivial GCD #56 |
| 84 | 0.615198 | `canonical` | `polybench_factor_product` | polybench factor product: polybench 5v uniform nontrivial factor #32 |
| 85 | 0.613128 | `canonical` | `polybench_factor_product` | polybench factor product: polybench 5v uniform nontrivial factor #131 |
| 86 | 0.610055 | `canonical` | `polybench_gcd_product` | polybench GCD products: polybench 8v uniform nontrivial GCD #188 |
| 87 | 0.607964 | `canonical` | `polybench_gcd_product` | polybench GCD products: polybench 5v uniform nontrivial GCD #11 |
| 88 | 0.607539 | `canonical` | `dedicated_multiplication` | GF(17) sparse large multiplication |
| 89 | 0.607374 | `alternative` | `crt_resultant_alternative` | resultant CRT: dense outer degrees 10/8 CRT crossover |
| 90 | 0.603600 | `canonical` | `polybench_gcd_product` | polybench GCD products: polybench 8v uniform nontrivial GCD #55 |
| 91 | 0.602930 | `canonical` | `polybench_gcd_product` | polybench GCD products: polybench 8v uniform trivial GCD #11 |
| 92 | 0.601913 | `canonical` | `polybench_factor_product` | polybench factor product: polybench 8v uniform nontrivial factor #159 |
| 93 | 0.600192 | `canonical` | `polybench_gcd_product` | polybench GCD products: polybench 8v uniform nontrivial GCD #168 |
| 94 | 0.599923 | `canonical` | `polybench_factor_product` | polybench factor product: polybench 5v uniform nontrivial factor #159 |
| 95 | 0.599179 | `canonical` | `polybench_gcd_product` | polybench GCD products: polybench 8v uniform nontrivial GCD #11 |
| 96 | 0.591952 | `canonical` | `generated_gcd_operation` | generated GCD auto: dense 8 variables degree 5 |
| 97 | 0.586162 | `canonical` | `dedicated_multiplication` | sparse large multiplication |
| 98 | 0.586125 | `canonical` | `polybench_factor_product` | polybench factor product: polybench 5v uniform nontrivial factor #163 |
| 99 | 0.574252 | `canonical` | `dedicated_multiplication` | GF(18446744073709551557) sparse large multiplication |
| 100 | 0.573810 | `canonical` | `generated_factor_product` | generated factor product: dense high-height 1-variable degrees 17/16 total 33 |
| 101 | 0.570573 | `canonical` | `main_ducos_resultant` | resultant Ducos: lacunary outer degrees 18/11 |
| 102 | 0.565520 | `canonical` | `generated_factor_operation` | generated factorization: dense 2-variable degrees 10/9 |
| 103 | 0.554473 | `canonical` | `main_ducos_resultant` | resultant Ducos: nonunit leading degrees 9/7 |
| 104 | 0.537410 | `canonical` | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform nontrivial GCD #168 |
| 105 | 0.524530 | `canonical` | `generated_gcd_product` | generated GCD products: high-gap 8 variables degree 4 gap 256 |
| 106 | 0.519856 | `canonical` | `main_ducos_resultant` | resultant Ducos: dense outer degrees 7/6 |
| 107 | 0.514253 | `canonical` | `polybench_gcd_operation` | polybench GCD: polybench 5v sharp nontrivial GCD #11 |
| 108 | 0.513866 | `canonical` | `dedicated_multiplication` | GF(18446744073709551557) dense very large multiplication |
| 109 | 0.472101 | `canonical` | `generated_gcd_operation` | generated GCD auto: high-height 8 variables degree 3 coefficient bits 256 |
| 110 | 0.470237 | `canonical` | `generated_gcd_operation` | generated GCD auto: high-height 5 variables degree 4 coefficient bits 1024 |
| 111 | 0.458120 | `canonical` | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform nontrivial GCD #188 |
| 112 | 0.449493 | `canonical` | `dedicated_multiplication` | dense very large multiplication |
| 113 | 0.447907 | `canonical` | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform nontrivial GCD #56 |
| 114 | 0.442621 | `canonical` | `exact_integer_division` | dense large exact division |
| 115 | 0.441603 | `canonical` | `exact_integer_division` | dense exact division |
| 116 | 0.437142 | `canonical` | `generated_gcd_operation` | generated GCD auto: high-height 5 variables degree 4 coefficient bits 512 |
| 117 | 0.436875 | `canonical` | `dedicated_multiplication` | GF(18446744073709551557) seven-variable power-minus-one multiplication |
| 118 | 0.434133 | `canonical` | `generated_gcd_operation` | generated GCD auto: high-height 5 variables degree 4 coefficient bits 256 |
| 119 | 0.432838 | `alternative` | `crt_resultant_alternative` | resultant CRT: outer-sparse degrees 12/9 CRT crossover |
| 120 | 0.429881 | `canonical` | `generated_gcd_operation` | generated GCD auto: high-height 5 variables degree 4 coefficient bits 128 |
| 121 | 0.413575 | `canonical` | `dedicated_multiplication` | GF(18446744073709551557) five-variable total-degree multiplication |
| 122 | 0.390026 | `canonical` | `dedicated_multiplication` | GF(17) seven-variable power-minus-one multiplication |
| 123 | 0.387980 | `canonical` | `polybench_gcd_operation` | polybench GCD: polybench 8v sharp nontrivial GCD #53 |
| 124 | 0.378042 | `canonical` | `dedicated_multiplication` | GF(17) five-variable total-degree multiplication |
| 125 | 0.356182 | `canonical` | `generated_gcd_operation` | generated GCD auto: sparse 8 variables degree 5 |
| 126 | 0.289808 | `canonical` | `generated_gcd_product` | generated GCD products: dense 2 variables degree 5 |
| 127 | 0.252283 | `canonical` | `dedicated_multiplication` | GF(18446744073709551557) dense univariate degree-4912 multiplication |
| 128 | 0.239560 | `canonical` | `polybench_factor_operation` | polybench factorization: polybench 8v sharp nontrivial factor #178 |
| 129 | 0.221031 | `canonical` | `polybench_factor_operation` | polybench factorization: polybench 5v uniform nontrivial factor #163 |
| 130 | 0.201668 | `canonical` | `generated_gcd_operation` | generated GCD auto: dense 3 variables degree 7 |
| 131 | 0.170000 | `canonical` | `polybench_factor_operation` | polybench factorization: polybench 8v uniform nontrivial factor #159 |
| 132 | 0.097756 | `canonical` | `polybench_factor_operation` | polybench factorization: polybench 8v uniform nontrivial factor #44 |
| 133 | 0.073806 | `canonical` | `generated_gcd_operation` | generated GCD auto: high-gap 5 variables degree 5 gap 64 |
| 134 | 0.028678 | `canonical` | `generated_gcd_operation` | generated GCD auto: high-gap 8 variables degree 4 gap 256 |

## Current high-interest replacements

| Optimization | Previous source-matched S/F | Current S/F | Result |
|---|---:|---:|---|
| Wide packed `u16` row merge, high-gap eight-variable degree 4 | 1.654253 | 0.524530 | Symbolica median reduced 68.9%; now 1.91x faster than FLINT |
| Few-row packed merge, sparse eight-variable degree 5 | 1.340679 | 0.552281 | Symbolica median reduced about 58.9%; the consistent full-sweep inventory is 0.780017 S/F |
| Initial primitive bivariate sample, dense three-variable degrees 6/5 | 1.747643 | 0.672529 | Symbolica median reduced 61.4%; now 1.49x faster than FLINT |
| Three-factor Hensel root, high-height degree 33 | 1.997930 | 1.112701 | Symbolica absolute time reduced about 40% |
| Sparse high-gap five-variable product heap advance | 1.040981 | 0.799572 | S/F reduced about 23.2% |
| PolyBench factorization #84 variable order | 8.690880 | 0.688572 | Current S/F is 12.62x below the pre-selection ratio |
| Dense degree-64 factorization | 1.748240 | 1.296474 | Dense EDF reduced absolute time 19.2%; Hensel image reuse adds 0.67% |

The high-height degree-33 product itself is `0.573810` S/F, so construction is faster than FLINT;
the remaining factorization gap is in the factorization pipeline, not that multiplication.

## Versioning and update protocol

`CURRENT_STATUS.md` is the live latest view. Every accepted refresh must also create the next
immutable `CURRENT_STATUS_v<i>.md` snapshot. Never edit an older numbered snapshot. The first
paragraph of each new snapshot must identify the preceding version and summarize every affected
case or family with its old and new S/F values.

After every accepted performance change:

1. Rerun every affected case sequentially with the current source-matched Symbolica/FLINT binary.
2. For repeated processes, replace the row with the median process-level S/F. Do not retain an old
   candidate or control as another current row.
3. Merge the 122 canonical rows and 12 alternative Brown/CRT rows, sort all 134 numerically by S/F
   in descending order, and renumber them.
4. Recompute the summary and verify 122 unique canonical workload/category pairs, 12 alternatives,
   and monotonic descending ratios.
5. Create the next numbered snapshot with its opening delta paragraph, then make
   `CURRENT_STATUS.md` mirror that latest snapshot.
6. Record benchmark binary hashes, raw timing artifacts, profiles, accepted changes, and rejected
   experiments in [PolynomialPerformanceHandoff.md](PolynomialPerformanceHandoff.md).

The `v5` inventory was programmatically checked for 122 unique canonical rows, 12 alternatives,
and monotonically descending S/F values after the high-gap replacement and next-worst refresh.
