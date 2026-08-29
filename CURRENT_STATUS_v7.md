# Current Symbolica/FLINT polynomial performance

Version `v7` records commit `ec6a131`, the chunked mixed-radix integer multiplication change,
relative to `v6`.
For generated dense five-variable degree-7 GCD products, the six-process source-matched median
changes `1.679650 -> 0.748808` S/F (`4.363303 -> 1.907774 ms` for Symbolica), a 56.27%
reduction in Symbolica time; the accumulated `v6` inventory value was `1.706529`. Symbolica is
now 1.34x faster than FLINT on this construction. The primary scoreboard continues to exclude all
algorithm-specific resultant measurements: Ducos, Brown, and CRT appear only in the appendix and
do not affect ranks or statistics.

The accepted integer kernel splits the most-significant mixed-radix variable into outer rows and
reuses one 50,625-entry `i128` inner accumulator. Its carry-free additive inner indices avoid the
rank-table work of compact simplex multiplication, while the 0.77 MiB workspace fits the per-core
L2 cache. A narrow selector requires at least five variables, a large mixed box, a decisive
box-to-simplex ratio, at least eight outer rows, and a bounded inner chunk; unsupported coefficient
or layout regimes retain the previous dense fallback.

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
| Symbolica faster than FLINT | 88 |
| Symbolica slower than FLINT | 28 |
| Worst primary S/F | **1.610087** — generated dense eight-variable degree-5 GCD products |
| Median primary S/F | **0.678025** |
| Best primary S/F | **0.028678** — generated high-gap eight-variable GCD |
| Former worst #131 S/F | **0.478628** — Symbolica is 2.09x faster than FLINT |
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
| 5 | 1.296474 | `generated_factor_operation` | generated factorization: dense 1-variable degrees 33/31 total 64 |
| 6 | 1.268240 | `polybench_gcd_operation` | polybench GCD: polybench 8v sharp nontrivial GCD #140 |
| 7 | 1.246310 | `generated_gcd_product` | generated GCD products: high-height 5 variables degree 4 coefficient bits 256 |
| 8 | 1.234835 | `dense_gcd_operation` | configured dense univariate GCD degree 80 |
| 9 | 1.184942 | `generated_factor_product` | generated factor product: dense 1-variable degrees 33/32 total 65 |
| 10 | 1.181340 | `dense_gcd_operation` | configured dense univariate GCD degree 64 |
| 11 | 1.164159 | `generated_factor_operation` | generated factorization: dense 1-variable degrees 33/32 total 65 |
| 12 | 1.137177 | `dedicated_multiplication` | GF(18446744073709551557) dense large multiplication |
| 13 | 1.112701 | `generated_factor_operation` | generated factorization: dense high-height 1-variable degrees 17/16 total 33 |
| 14 | 1.106606 | `polybench_factor_operation` | polybench factorization: polybench 8v uniform nontrivial factor #105 |
| 15 | 1.097676 | `polybench_gcd_operation` | polybench GCD: polybench 5v uniform nontrivial GCD #11 |
| 16 | 1.094628 | `generated_factor_product` | generated factor product: dense 1-variable degrees 33/31 total 64 |
| 17 | 1.072266 | `dense_gcd_operation` | configured dense univariate GCD degree 48 |
| 18 | 1.060731 | `generated_gcd_operation` | generated GCD auto: dense 2 variables degree 5 |
| 19 | 1.059355 | `generated_gcd_product` | generated GCD products: high-height 5 variables degree 4 coefficient bits 512 |
| 20 | 1.056183 | `dedicated_multiplication` | dense large multiplication |
| 21 | 1.055155 | `dedicated_multiplication` | GF(17) dense very large multiplication |
| 22 | 1.054112 | `generated_gcd_product` | generated GCD products: high-height 8 variables degree 3 coefficient bits 256 |
| 23 | 1.034406 | `generated_factor_product` | generated factor product: dense 1-variable degrees 32/31 |
| 24 | 1.029727 | `generated_factor_operation` | generated factorization: dense 1-variable degrees 32/31 |
| 25 | 1.029726 | `generated_gcd_product` | generated GCD products: dense 1 variables degree 32 |
| 26 | 1.019191 | `generated_gcd_product` | generated GCD products: high-height 5 variables degree 4 coefficient bits 1024 |
| 27 | 1.009108 | `generated_factor_product` | generated factor product: dense 3-variable degrees 6/5 |
| 28 | 1.005858 | `polybench_factor_operation` | polybench factorization: polybench 8v sharp nontrivial factor #176 |
| 29 | 0.994841 | `generated_gcd_operation` | generated GCD auto: dense 1 variables degree 32 |
| 30 | 0.982902 | `polybench_factor_operation` | polybench factorization: polybench 5v uniform nontrivial factor #159 |
| 31 | 0.968978 | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform trivial GCD #11 |
| 32 | 0.966693 | `dedicated_multiplication` | dense high multiplication |
| 33 | 0.956438 | `dedicated_multiplication` | dense high large multiplication |
| 34 | 0.940972 | `dedicated_multiplication` | GF(500000003) dense univariate degrees 128/64 accumulator-bound multiplication |
| 35 | 0.913795 | `dedicated_multiplication` | dense small multiplication |
| 36 | 0.902871 | `default_gcd_operation` | GCD auto: dense 7 variables degree 7 |
| 37 | 0.878519 | `generated_gcd_product` | generated GCD products: dense 3 variables degree 7 |
| 38 | 0.870516 | `dedicated_multiplication` | GF(17) dense univariate degree-4912 multiplication |
| 39 | 0.858115 | `generated_factor_product` | generated factor product: dense 2-variable degrees 10/9 |
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
| 58 | 0.680827 | `generated_factor_operation` | generated factorization: dense 3-variable degrees 6/5 |
| 59 | 0.675222 | `polybench_factor_product` | polybench factor product: polybench 8v sharp nontrivial factor #92 |
| 60 | 0.674700 | `polybench_gcd_product` | polybench GCD products: polybench 5v sharp nontrivial GCD #11 |
| 61 | 0.674333 | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform nontrivial GCD #11 |
| 62 | 0.642318 | `polybench_gcd_operation` | polybench GCD: polybench 8v sharp nontrivial GCD #35 |
| 63 | 0.638505 | `polybench_factor_product` | polybench factor product: polybench 8v uniform nontrivial factor #105 |
| 64 | 0.637854 | `polybench_factor_product` | polybench factor product: polybench 8v sharp nontrivial factor #178 |
| 65 | 0.636796 | `polybench_gcd_product` | polybench GCD products: polybench 8v sharp nontrivial GCD #140 |
| 66 | 0.635520 | `generated_gcd_operation` | generated GCD auto: sparse 5 variables degree 7 |
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
| 90 | 0.478628 | `polybench_factor_operation` | polybench factorization: polybench 5v uniform nontrivial factor #131 |
| 91 | 0.472101 | `generated_gcd_operation` | generated GCD auto: high-height 8 variables degree 3 coefficient bits 256 |
| 92 | 0.470237 | `generated_gcd_operation` | generated GCD auto: high-height 5 variables degree 4 coefficient bits 1024 |
| 93 | 0.458120 | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform nontrivial GCD #188 |
| 94 | 0.449493 | `dedicated_multiplication` | dense very large multiplication |
| 95 | 0.447907 | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform nontrivial GCD #56 |
| 96 | 0.442621 | `exact_integer_division` | dense large exact division |
| 97 | 0.441603 | `exact_integer_division` | dense exact division |
| 98 | 0.437142 | `generated_gcd_operation` | generated GCD auto: high-height 5 variables degree 4 coefficient bits 512 |
| 99 | 0.436875 | `dedicated_multiplication` | GF(18446744073709551557) seven-variable power-minus-one multiplication |
| 100 | 0.434133 | `generated_gcd_operation` | generated GCD auto: high-height 5 variables degree 4 coefficient bits 256 |
| 101 | 0.429881 | `generated_gcd_operation` | generated GCD auto: high-height 5 variables degree 4 coefficient bits 128 |
| 102 | 0.413575 | `dedicated_multiplication` | GF(18446744073709551557) five-variable total-degree multiplication |
| 103 | 0.390026 | `dedicated_multiplication` | GF(17) seven-variable power-minus-one multiplication |
| 104 | 0.387980 | `polybench_gcd_operation` | polybench GCD: polybench 8v sharp nontrivial GCD #53 |
| 105 | 0.378042 | `dedicated_multiplication` | GF(17) five-variable total-degree multiplication |
| 106 | 0.356182 | `generated_gcd_operation` | generated GCD auto: sparse 8 variables degree 5 |
| 107 | 0.289808 | `generated_gcd_product` | generated GCD products: dense 2 variables degree 5 |
| 108 | 0.272966 | `polybench_factor_operation` | polybench factorization: polybench 5v uniform nontrivial factor #32 |
| 109 | 0.252283 | `dedicated_multiplication` | GF(18446744073709551557) dense univariate degree-4912 multiplication |
| 110 | 0.239560 | `polybench_factor_operation` | polybench factorization: polybench 8v sharp nontrivial factor #178 |
| 111 | 0.216817 | `polybench_factor_operation` | polybench factorization: polybench 5v uniform nontrivial factor #163 |
| 112 | 0.201668 | `generated_gcd_operation` | generated GCD auto: dense 3 variables degree 7 |
| 113 | 0.170000 | `polybench_factor_operation` | polybench factorization: polybench 8v uniform nontrivial factor #159 |
| 114 | 0.097756 | `polybench_factor_operation` | polybench factorization: polybench 8v uniform nontrivial factor #44 |
| 115 | 0.073806 | `generated_gcd_operation` | generated GCD auto: high-gap 5 variables degree 5 gap 64 |
| 116 | 0.028678 | `generated_gcd_operation` | generated GCD auto: high-gap 8 variables degree 4 gap 256 |

## Current high-interest replacements

| Optimization | Previous source-matched S/F | Current S/F | Result |
|---|---:|---:|---|
| Chunked mixed-radix dense-five degree-7 multiplication | 1.679650 | 0.748808 | Symbolica median reduced 56.27%; now 1.34x faster than FLINT |
| One-image Wang leading-coefficient reconstruction, PolyBench 5v #131 | 1.745716 | 0.478628 | Symbolica median reduced 72.8%; now 2.09x faster than FLINT |
| One-image Wang leading-coefficient reconstruction, PolyBench 5v #32 | 1.124230 | 0.272966 | Symbolica median reduced about 76.0%; now 3.66x faster than FLINT |
| Wide packed `u16` row merge, high-gap eight-variable degree 4 | 1.654253 | 0.524530 | Symbolica median reduced 68.9%; now 1.91x faster than FLINT |
| Few-row packed merge, sparse eight-variable degree 5 | 1.340679 | 0.552281 | Symbolica median reduced about 58.9%; the consistent full-sweep inventory is 0.780017 S/F |
| Initial primitive bivariate sample, dense three-variable degrees 6/5 | 1.747643 | 0.680827 | Symbolica median remains about 61% below the source-matched pre-optimization control |
| Three-factor Hensel root, high-height degree 33 | 1.997930 | 1.112701 | Symbolica absolute time reduced about 40% |
| Sparse high-gap five-variable product heap advance | 1.040981 | 0.799572 | S/F reduced about 23.2% |
| PolyBench factorization #84 variable order | 8.690880 | 0.691758 | Current S/F is 12.56x below the pre-selection ratio |
| Dense degree-64 factorization | 1.748240 | 1.296474 | Dense EDF reduced absolute time 19.2%; Hensel image reuse adds 0.67% |

The high-height degree-33 product itself is `0.573810` S/F, so construction is faster than FLINT;
the remaining factorization gap is in the factorization pipeline, not that multiplication.

## Appendix: resultant backend reference

These 18 frozen `v5` measurements repeat the same six workloads for three Symbolica algorithms.
They are retained for algorithm study but are excluded from every primary count, rank, and summary
statistic.

| Workload | Ducos S/F | Brown S/F | CRT S/F |
|---|---:|---:|---:|
| dense outer degrees 7/6 | 0.519856 | 1.333630 | 2.059289 |
| nonunit leading degrees 9/7 | 0.554473 | 1.124006 | 1.129343 |
| dense outer degrees 10/8 CRT crossover | 0.932951 | 2.464208 | 0.607374 |
| outer-sparse degrees 12/9 CRT crossover | 1.027247 | 1.862295 | 0.432838 |
| large high-height degrees 14/10 | 0.861806 | 2.581685 | 1.249141 |
| lacunary outer degrees 18/11 | 0.570573 | 1.438647 | 2.287301 |

Ducos is faster than FLINT on five of six inputs, with median `0.716190` S/F. Brown is an explicit
reference alternative and is slower on all six; CRT wins two of six. FLINT's resultant and
Symbolica's default both use a Ducos-family recurrence, so the Brown rows are not default-path
performance gaps.

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

The `v7` inventory was programmatically checked for 116 unique primary rows, 88 rows faster than
FLINT, 28 rows slower than FLINT, and monotonically descending S/F values after the chunked dense
multiplication refresh.
