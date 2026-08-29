# Current Symbolica/FLINT polynomial performance

Version `v10` records the constant-coordinate projective reconstruction for dense univariate
integer GCDs in commit `e58e876`, relative to `v9`. Six alternating source-matched processes
change the configured GCD rows at degree 48 from `1.072266 -> 0.889777` S/F, degree 64 from
`1.181340 -> 0.853224`, and degree 80 from `1.234835 -> 1.075107`. The same current-binary
refresh changes generated factorization at degree 33 from `1.108418 -> 1.112958`, degree 63 from
`1.034949 -> 1.030347`, degree 64 from `1.126451 -> 1.128065`, and degree 65 from
`1.159761 -> 1.161896`; these sub-percent factor shifts are guard noise rather than attributed
gains. The degree-64 factor-product guard is likewise neutral and does not replace its stronger
existing inventory row.

For the degree-64 GCD fixture, leading-coordinate normalization reconstructs a roughly 225-bit
representative and takes four modular images. The selected constant-coordinate representative is
`2G`, roughly 126 bits, and certifies after two images. A lazy leading-coordinate reconstruction
uses the same CRT state on an independent geometric schedule, so an underestimated constant
representative cannot postpone the older leading-coordinate exit indefinitely. The final
source-matched Symbolica median falls from `0.409735 ms` to `0.300099 ms`, or 26.76%, while
FLINT remains near `0.352 ms`.

The preceding `v9` PolyBench five-variable uniform factorization #131 result remains
`0.277201` S/F: Symbolica is 3.61x faster than FLINT. Its bounded dense
`FiniteField<Integer>` multiplication context accumulates exact coefficient products with fused
integer multiply-adds and performs one Montgomery reduction per output coefficient. Ducos, Brown,
and CRT resultant measurements remain a historical appendix note and never participate in the
primary ranks or summary.

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
| Primary comparisons | 116 |
| Symbolica faster than FLINT | 90 |
| Symbolica slower than FLINT | 26 |
| Worst primary S/F | **1.610087** — generated dense eight-variable degree-5 GCD products |
| Median primary S/F | **0.674961** |
| Best primary S/F | **0.028678** — generated high-gap eight-variable GCD |
| Focused #131 S/F | **0.277201** — Symbolica is 3.61x faster than FLINT |
| PolyBench GCD product median (12 cases) | **0.613690** |
| PolyBench factor product median (11 cases) | **0.617710** |
| All PolyBench product median (23 cases) | **0.617324** |

## Primary comparisons, worst to best

Resultant backends are excluded from this table and its statistics.

| Rank | S/F | Category | Benchmark |
|---:|---:|---|---|
| 1 | 1.610087 | `generated_gcd_product` | generated GCD products: dense 8 variables degree 5 |
| 2 | 1.584134 | `default_gcd_product` | GCD products: dense 7 variables degree 7 |
| 3 | 1.576901 | `generated_gcd_product` | generated GCD products: high-height 5 variables degree 4 coefficient bits 128 |
| 4 | 1.348957 | `dedicated_multiplication` | seven-variable power-minus-one multiplication |
| 5 | 1.268240 | `polybench_gcd_operation` | polybench GCD: polybench 8v sharp nontrivial GCD #140 |
| 6 | 1.246310 | `generated_gcd_product` | generated GCD products: high-height 5 variables degree 4 coefficient bits 256 |
| 7 | 1.184942 | `generated_factor_product` | generated factor product: dense 1-variable degrees 33/32 total 65 |
| 8 | 1.161896 | `generated_factor_operation` | generated factorization: dense 1-variable degrees 33/32 total 65 |
| 9 | 1.137177 | `dedicated_multiplication` | GF(18446744073709551557) dense large multiplication |
| 10 | 1.128065 | `generated_factor_operation` | generated factorization: dense 1-variable degrees 33/31 total 64 |
| 11 | 1.112958 | `generated_factor_operation` | generated factorization: dense high-height 1-variable degrees 17/16 total 33 |
| 12 | 1.106606 | `polybench_factor_operation` | polybench factorization: polybench 8v uniform nontrivial factor #105 |
| 13 | 1.097676 | `polybench_gcd_operation` | polybench GCD: polybench 5v uniform nontrivial GCD #11 |
| 14 | 1.094628 | `generated_factor_product` | generated factor product: dense 1-variable degrees 33/31 total 64 |
| 15 | 1.075107 | `dense_gcd_operation` | configured dense univariate GCD degree 80 |
| 16 | 1.060731 | `generated_gcd_operation` | generated GCD auto: dense 2 variables degree 5 |
| 17 | 1.059355 | `generated_gcd_product` | generated GCD products: high-height 5 variables degree 4 coefficient bits 512 |
| 18 | 1.056183 | `dedicated_multiplication` | dense large multiplication |
| 19 | 1.055155 | `dedicated_multiplication` | GF(17) dense very large multiplication |
| 20 | 1.054112 | `generated_gcd_product` | generated GCD products: high-height 8 variables degree 3 coefficient bits 256 |
| 21 | 1.034406 | `generated_factor_product` | generated factor product: dense 1-variable degrees 32/31 |
| 22 | 1.030347 | `generated_factor_operation` | generated factorization: dense 1-variable degrees 32/31 |
| 23 | 1.029726 | `generated_gcd_product` | generated GCD products: dense 1 variables degree 32 |
| 24 | 1.019191 | `generated_gcd_product` | generated GCD products: high-height 5 variables degree 4 coefficient bits 1024 |
| 25 | 1.009108 | `generated_factor_product` | generated factor product: dense 3-variable degrees 6/5 |
| 26 | 1.005858 | `polybench_factor_operation` | polybench factorization: polybench 8v sharp nontrivial factor #176 |
| 27 | 0.994841 | `generated_gcd_operation` | generated GCD auto: dense 1 variables degree 32 |
| 28 | 0.968978 | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform trivial GCD #11 |
| 29 | 0.966693 | `dedicated_multiplication` | dense high multiplication |
| 30 | 0.956438 | `dedicated_multiplication` | dense high large multiplication |
| 31 | 0.940972 | `dedicated_multiplication` | GF(500000003) dense univariate degrees 128/64 accumulator-bound multiplication |
| 32 | 0.913795 | `dedicated_multiplication` | dense small multiplication |
| 33 | 0.902871 | `default_gcd_operation` | GCD auto: dense 7 variables degree 7 |
| 34 | 0.899827 | `polybench_factor_operation` | polybench factorization: polybench 5v uniform nontrivial factor #159 |
| 35 | 0.889777 | `dense_gcd_operation` | configured dense univariate GCD degree 48 |
| 36 | 0.878519 | `generated_gcd_product` | generated GCD products: dense 3 variables degree 7 |
| 37 | 0.870516 | `dedicated_multiplication` | GF(17) dense univariate degree-4912 multiplication |
| 38 | 0.858115 | `generated_factor_product` | generated factor product: dense 2-variable degrees 10/9 |
| 39 | 0.853224 | `dense_gcd_operation` | configured dense univariate GCD degree 64 |
| 40 | 0.836402 | `polybench_factor_operation` | polybench factorization: polybench 8v sharp nontrivial factor #92 |
| 41 | 0.828262 | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform nontrivial GCD #55 |
| 42 | 0.822914 | `generated_gcd_product` | generated GCD products: high-gap 5 variables degree 5 gap 64 |
| 43 | 0.821527 | `dedicated_multiplication` | GF(17) dense large multiplication |
| 44 | 0.798104 | `dedicated_multiplication` | GF(65000011) dense univariate degrees 128/64 accumulator-bound multiplication |
| 45 | 0.791465 | `generated_gcd_product` | generated GCD products: sparse 5 variables degree 7 |
| 46 | 0.780017 | `generated_gcd_product` | generated GCD products: sparse 8 variables degree 5 |
| 47 | 0.761777 | `exact_integer_division` | high-height exact division |
| 48 | 0.750982 | `generated_gcd_operation` | generated GCD auto: dense 5 variables degree 7 |
| 49 | 0.750365 | `polybench_gcd_product` | polybench GCD products: polybench 8v sharp nontrivial GCD #53 |
| 50 | 0.748808 | `generated_gcd_product` | generated GCD products: dense 5 variables degree 7 |
| 51 | 0.748107 | `polybench_gcd_operation` | polybench GCD: polybench 8v sharp nontrivial GCD #11 |
| 52 | 0.723532 | `polybench_factor_product` | polybench factor product: polybench 8v sharp nontrivial factor #176 |
| 53 | 0.711976 | `polybench_factor_product` | polybench factor product: polybench 8v sharp nontrivial factor #84 |
| 54 | 0.704684 | `polybench_gcd_product` | polybench GCD products: polybench 8v sharp nontrivial GCD #11 |
| 55 | 0.691815 | `dedicated_multiplication` | sparse separated multiplication |
| 56 | 0.691758 | `polybench_factor_operation` | polybench factorization: polybench 8v sharp nontrivial factor #84 |
| 57 | 0.688808 | `polybench_gcd_product` | polybench GCD products: polybench 8v sharp nontrivial GCD #35 |
| 58 | 0.675222 | `polybench_factor_product` | polybench factor product: polybench 8v sharp nontrivial factor #92 |
| 59 | 0.674700 | `polybench_gcd_product` | polybench GCD products: polybench 5v sharp nontrivial GCD #11 |
| 60 | 0.674333 | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform nontrivial GCD #11 |
| 61 | 0.642318 | `polybench_gcd_operation` | polybench GCD: polybench 8v sharp nontrivial GCD #35 |
| 62 | 0.638505 | `polybench_factor_product` | polybench factor product: polybench 8v uniform nontrivial factor #105 |
| 63 | 0.637854 | `polybench_factor_product` | polybench factor product: polybench 8v sharp nontrivial factor #178 |
| 64 | 0.636796 | `polybench_gcd_product` | polybench GCD products: polybench 8v sharp nontrivial GCD #140 |
| 65 | 0.635520 | `generated_gcd_operation` | generated GCD auto: sparse 5 variables degree 7 |
| 66 | 0.623487 | `generated_factor_operation` | generated factorization: dense 3-variable degrees 6/5 |
| 67 | 0.617710 | `polybench_factor_product` | polybench factor product: polybench 8v uniform nontrivial factor #44 |
| 68 | 0.617324 | `polybench_gcd_product` | polybench GCD products: polybench 8v uniform nontrivial GCD #56 |
| 69 | 0.615198 | `polybench_factor_product` | polybench factor product: polybench 5v uniform nontrivial factor #32 |
| 70 | 0.613128 | `polybench_factor_product` | polybench factor product: polybench 5v uniform nontrivial factor #131 |
| 71 | 0.610055 | `polybench_gcd_product` | polybench GCD products: polybench 8v uniform nontrivial GCD #188 |
| 72 | 0.607964 | `polybench_gcd_product` | polybench GCD products: polybench 5v uniform nontrivial GCD #11 |
| 73 | 0.607539 | `dedicated_multiplication` | GF(17) sparse large multiplication |
| 74 | 0.603600 | `polybench_gcd_product` | polybench GCD products: polybench 8v uniform nontrivial GCD #55 |
| 75 | 0.602930 | `polybench_gcd_product` | polybench GCD products: polybench 8v uniform trivial GCD #11 |
| 76 | 0.601913 | `polybench_factor_product` | polybench factor product: polybench 8v uniform nontrivial factor #159 |
| 77 | 0.600192 | `polybench_gcd_product` | polybench GCD products: polybench 8v uniform nontrivial GCD #168 |
| 78 | 0.599923 | `polybench_factor_product` | polybench factor product: polybench 5v uniform nontrivial factor #159 |
| 79 | 0.599179 | `polybench_gcd_product` | polybench GCD products: polybench 8v uniform nontrivial GCD #11 |
| 80 | 0.591952 | `generated_gcd_operation` | generated GCD auto: dense 8 variables degree 5 |
| 81 | 0.586162 | `dedicated_multiplication` | sparse large multiplication |
| 82 | 0.586125 | `polybench_factor_product` | polybench factor product: polybench 5v uniform nontrivial factor #163 |
| 83 | 0.574252 | `dedicated_multiplication` | GF(18446744073709551557) sparse large multiplication |
| 84 | 0.573810 | `generated_factor_product` | generated factor product: dense high-height 1-variable degrees 17/16 total 33 |
| 85 | 0.565520 | `generated_factor_operation` | generated factorization: dense 2-variable degrees 10/9 |
| 86 | 0.537410 | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform nontrivial GCD #168 |
| 87 | 0.524530 | `generated_gcd_product` | generated GCD products: high-gap 8 variables degree 4 gap 256 |
| 88 | 0.514253 | `polybench_gcd_operation` | polybench GCD: polybench 5v sharp nontrivial GCD #11 |
| 89 | 0.513866 | `dedicated_multiplication` | GF(18446744073709551557) dense very large multiplication |
| 90 | 0.472101 | `generated_gcd_operation` | generated GCD auto: high-height 8 variables degree 3 coefficient bits 256 |
| 91 | 0.470237 | `generated_gcd_operation` | generated GCD auto: high-height 5 variables degree 4 coefficient bits 1024 |
| 92 | 0.458120 | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform nontrivial GCD #188 |
| 93 | 0.449493 | `dedicated_multiplication` | dense very large multiplication |
| 94 | 0.447907 | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform nontrivial GCD #56 |
| 95 | 0.442621 | `exact_integer_division` | dense large exact division |
| 96 | 0.441603 | `exact_integer_division` | dense exact division |
| 97 | 0.437142 | `generated_gcd_operation` | generated GCD auto: high-height 5 variables degree 4 coefficient bits 512 |
| 98 | 0.436875 | `dedicated_multiplication` | GF(18446744073709551557) seven-variable power-minus-one multiplication |
| 99 | 0.434133 | `generated_gcd_operation` | generated GCD auto: high-height 5 variables degree 4 coefficient bits 256 |
| 100 | 0.429881 | `generated_gcd_operation` | generated GCD auto: high-height 5 variables degree 4 coefficient bits 128 |
| 101 | 0.413575 | `dedicated_multiplication` | GF(18446744073709551557) five-variable total-degree multiplication |
| 102 | 0.390026 | `dedicated_multiplication` | GF(17) seven-variable power-minus-one multiplication |
| 103 | 0.387980 | `polybench_gcd_operation` | polybench GCD: polybench 8v sharp nontrivial GCD #53 |
| 104 | 0.378042 | `dedicated_multiplication` | GF(17) five-variable total-degree multiplication |
| 105 | 0.356182 | `generated_gcd_operation` | generated GCD auto: sparse 8 variables degree 5 |
| 106 | 0.289808 | `generated_gcd_product` | generated GCD products: dense 2 variables degree 5 |
| 107 | 0.277201 | `polybench_factor_operation` | polybench factorization: polybench 5v uniform nontrivial factor #131 |
| 108 | 0.252283 | `dedicated_multiplication` | GF(18446744073709551557) dense univariate degree-4912 multiplication |
| 109 | 0.239560 | `polybench_factor_operation` | polybench factorization: polybench 8v sharp nontrivial factor #178 |
| 110 | 0.222788 | `polybench_factor_operation` | polybench factorization: polybench 5v uniform nontrivial factor #32 |
| 111 | 0.217709 | `polybench_factor_operation` | polybench factorization: polybench 5v uniform nontrivial factor #163 |
| 112 | 0.201668 | `generated_gcd_operation` | generated GCD auto: dense 3 variables degree 7 |
| 113 | 0.170000 | `polybench_factor_operation` | polybench factorization: polybench 8v uniform nontrivial factor #159 |
| 114 | 0.097756 | `polybench_factor_operation` | polybench factorization: polybench 8v uniform nontrivial factor #44 |
| 115 | 0.073806 | `generated_gcd_operation` | generated GCD auto: high-gap 5 variables degree 5 gap 64 |
| 116 | 0.028678 | `generated_gcd_operation` | generated GCD auto: high-gap 8 variables degree 4 gap 256 |

## Current high-interest replacements

| Optimization | Earlier/reference S/F | Current S/F | Result |
|---|---:|---:|---|
| Chunked mixed-radix dense-five degree-7 multiplication | 1.679650 | 0.748808 | Symbolica median reduced 56.27%; now 1.34x faster than FLINT |
| Cumulative bivariate reconstruction, Wang lifting, and dense Montgomery products, PolyBench 5v #131 | 1.745716 | 0.277201 | Symbolica median reduced 84.1%; now 3.61x faster than FLINT |
| Dense large-modulus Montgomery products, PolyBench 5v #131 | 0.345860 | 0.276987 | Source-matched Symbolica median reduced 19.90% in commit `f14d182` |
| Bivariate reconstruction and Wang lift fast paths, PolyBench 5v #32 | 1.124230 | 0.222788 | Symbolica median reduced about 80.5%; now 4.49x faster than FLINT |
| Primitive bivariate dispatch and lazy exact-product exit, PolyBench 5v #159 | 0.991235 | 0.899827 | Source-matched Symbolica median reduced 9.95% |
| Wide packed `u16` row merge, high-gap eight-variable degree 4 | 1.654253 | 0.524530 | Symbolica median reduced 68.9%; now 1.91x faster than FLINT |
| Few-row packed merge, sparse eight-variable degree 5 | 1.340679 | 0.552281 | Symbolica median reduced about 58.9%; the consistent full-sweep inventory is 0.780017 S/F |
| Initial primitive bivariate sample and direct reconstruction, dense three-variable degrees 6/5 | 1.747643 | 0.623487 | Symbolica median is about 64% below the original source-matched control |
| Three-factor Hensel root, high-height degree 33 | 1.997930 | 1.112958 | Symbolica absolute time remains about 44.3% below the original measurement |
| Sparse high-gap five-variable product heap advance | 1.040981 | 0.799572 | S/F reduced about 23.2% |
| PolyBench factorization #84 variable order | 8.690880 | 0.691758 | Current S/F is 12.56x below the pre-selection ratio |
| Dense degree-64 factorization | 1.748240 | 1.128065 | Symbolica absolute time remains about 35.5% below the original measurement |
| Constant-coordinate reconstruction, dense univariate GCD degree 64 | 1.161142 | 0.853224 | Source-matched Symbolica median reduced 26.76%; now 1.17x faster than FLINT |

The high-height degree-33 product itself is `0.573810` S/F, so construction is faster than FLINT;
the remaining factorization gap is in the factorization pipeline, not that multiplication.

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
3. Merge the 116 primary non-resultant rows, sort them numerically by S/F in descending order, and
   renumber them. Keep the 18 resultant-backend references in the appendix only.
4. Recompute the summary and verify 116 unique primary workload/category pairs, 18 appendix
   measurements arranged as six workloads by three algorithms, and monotonic descending primary
   ratios.
5. Create the next numbered snapshot with its opening delta paragraph, then make
   `CURRENT_STATUS.md` mirror that latest snapshot.
6. Record benchmark binary hashes, raw timing artifacts, profiles, accepted changes, and rejected
   experiments in [PolynomialPerformanceHandoff.md](PolynomialPerformanceHandoff.md).

The `v10` inventory was programmatically checked for 116 unique primary rows, 90 rows faster than
FLINT, 26 rows slower than FLINT, median `0.674961`, and monotonically descending S/F values.
