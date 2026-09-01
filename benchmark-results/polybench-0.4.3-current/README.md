# FLINT versus Symbolica on PolyBench 0.4.3

This directory contains a complete rerun of the 5- and 8-variable PolyBench
setups `0001` through `0008`. Its layout matches the published
[PolyBench results](https://github.com/tueda/polybench-result/tree/main/0.4.3):
each setup has the measured CSV, normal run log, summary plot, and
`FLINT_vs_Symbolica.png`. The large generated problem logs are intentionally
omitted.

## Result

Across all 3,200 paired measured problems, the median per-problem
`Symbolica / FLINT` ratio is **0.567327**. Symbolica wins 2,704 of 3,200
problems, the geometric-mean ratio is **0.514302**, and the ratio of summed
measured times is **0.227502**. The median paired ratio is **0.686613** for the
5-variable problems and **0.492275** for the 8-variable problems.

Each Symbolica CSV value is the per-problem median of three complete final-v28
streams. The FLINT column is retained from the original same-seed PolyBench
3.5.0 run. The table is sorted from worst to best for Symbolica. `Paired S/F`
is the median of the 200 ratios `Symbolica_i / FLINT_i`; `total S/F` divides
the sums of the 200 measured times. The independent solver medians can
correspond to different problems and therefore do not, in general, have a
ratio equal to `Paired S/F`.

| Setup | Workload | FLINT median (ms) | Symbolica median (ms) | Paired S/F | Total S/F | S wins | Artifacts |
|---|---|---:|---:|---:|---:|---:|---|
| 05/0001 | uniform trivial GCD | 0.345627 | 0.359000 | 1.037358 | 1.038460 | 12/200 | [plot](05/0001.figures/FLINT_vs_Symbolica.png), [CSV](05/0001.csv), [log](05/0001.log) |
| 05/0005 | sharp trivial GCD | 0.284435 | 0.280000 | 0.985439 | 0.994697 | 139/200 | [plot](05/0005.figures/FLINT_vs_Symbolica.png), [CSV](05/0005.csv), [log](05/0005.log) |
| 08/0001 | uniform trivial GCD | 0.522521 | 0.505000 | 0.971488 | 0.947113 | 192/200 | [plot](08/0001.figures/FLINT_vs_Symbolica.png), [CSV](08/0001.csv), [log](08/0001.log) |
| 08/0005 | sharp trivial GCD | 0.493900 | 0.461000 | 0.930193 | 0.951852 | 192/200 | [plot](08/0005.figures/FLINT_vs_Symbolica.png), [CSV](08/0005.csv), [log](08/0005.log) |
| 08/0008 | sharp nontrivial factorization | 2.741800 | 2.007500 | 0.906664 | 0.692100 | 112/200 | [plot](08/0008.figures/FLINT_vs_Symbolica.png), [CSV](08/0008.csv), [log](08/0008.log) |
| 05/0006 | sharp nontrivial GCD | 3.812655 | 3.371000 | 0.882073 | 0.899982 | 135/200 | [plot](05/0006.figures/FLINT_vs_Symbolica.png), [CSV](05/0006.csv), [log](05/0006.log) |
| 05/0002 | uniform nontrivial GCD | 6.720045 | 4.781000 | 0.713426 | 0.743830 | 194/200 | [plot](05/0002.figures/FLINT_vs_Symbolica.png), [CSV](05/0002.csv), [log](05/0002.log) |
| 05/0008 | sharp nontrivial factorization | 6.883760 | 3.580500 | 0.670631 | 0.500058 | 150/200 | [plot](05/0008.figures/FLINT_vs_Symbolica.png), [CSV](05/0008.csv), [log](05/0008.log) |
| 08/0003 | uniform trivial factorization | 4.277095 | 1.950500 | 0.465421 | 0.453153 | 193/200 | [plot](08/0003.figures/FLINT_vs_Symbolica.png), [CSV](08/0003.csv), [log](08/0003.log) |
| 05/0003 | uniform trivial factorization | 2.664025 | 1.200000 | 0.446185 | 0.461814 | 193/200 | [plot](05/0003.figures/FLINT_vs_Symbolica.png), [CSV](05/0003.csv), [log](05/0003.log) |
| 08/0006 | sharp nontrivial GCD | 5.341565 | 2.248000 | 0.410405 | 0.433058 | 198/200 | [plot](08/0006.figures/FLINT_vs_Symbolica.png), [CSV](08/0006.csv), [log](08/0006.log) |
| 08/0002 | uniform nontrivial GCD | 6.869210 | 2.698500 | 0.401268 | 0.409579 | 200/200 | [plot](08/0002.figures/FLINT_vs_Symbolica.png), [CSV](08/0002.csv), [log](08/0002.log) |
| 05/0007 | sharp trivial factorization | 2.427320 | 0.689500 | 0.288299 | 0.227549 | 199/200 | [plot](05/0007.figures/FLINT_vs_Symbolica.png), [CSV](05/0007.csv), [log](05/0007.log) |
| 08/0004 | uniform nontrivial factorization | 58.536850 | 13.336500 | 0.230087 | 0.134639 | 197/200 | [plot](08/0004.figures/FLINT_vs_Symbolica.png), [CSV](08/0004.csv), [log](08/0004.log) |
| 08/0007 | sharp trivial factorization | 4.756305 | 1.028000 | 0.225626 | 0.199631 | 198/200 | [plot](08/0007.figures/FLINT_vs_Symbolica.png), [CSV](08/0007.csv), [log](08/0007.log) |
| 05/0004 | uniform nontrivial factorization | 56.179300 | 8.201000 | 0.143235 | 0.131367 | 200/200 | [plot](05/0004.figures/FLINT_vs_Symbolica.png), [CSV](05/0004.csv), [log](05/0004.log) |

## Setup map

| ID | Exponent distribution | Operation | Construction |
|---|---|---|---|
| 0001 | uniform | GCD | trivial, `gcd(a*b,c*d)` |
| 0002 | uniform | GCD | nontrivial, `gcd(a*g,b*g)` |
| 0003 | uniform | factorization | usually irreducible, `factor(a*b+c)` |
| 0004 | uniform | factorization | nontrivial, `factor(a*b)` |
| 0005 | sharp | GCD | trivial, `gcd(a*b,c*d)` |
| 0006 | sharp | GCD | nontrivial, `gcd(a*g,b*g)` |
| 0007 | sharp | factorization | usually irreducible, `factor(a*b+c)` |
| 0008 | sharp | factorization | nontrivial, `factor(a*b)` |

## Protocol and provenance

- PolyBench tag `0.4.3`, commit
  `f3a25498883a80462c6278a87c9dfc93630d8a06`, using the published
  [benchmark script](https://github.com/tueda/polybench/blob/0.4.3/scripts/benchmark/0.3.0.sh)
  and [plot implementation](https://github.com/tueda/polybench/blob/0.4.3/polybench/plot.py).
- Seed 42; three complete streams of 10 warmups and 200 measured problems per setup; requested basic
  polynomial sizes 37--50 terms; degrees 22--30 for uniform inputs and 0--30
  for sharp inputs; coefficients `-16384..=16384`; timeout 21,600 seconds per
  solver. Parsing is outside the timed region.
- Sequential, single-core execution pinned to logical CPU 8 with
  `RAYON_NUM_THREADS=1` on an Intel Xeon W-2135, Linux 6.18.37, Python 3.12.13.
- FLINT 3.5.0 built through PolyBench's vcpkg recipe with GCC 15.2.0.
- Symbolica 2.2.0 from the final v28 performance source. Its benchmark-relevant
  paths were finalized in commit `0352f1b`. The upstream adapter was built with
  plain `cargo build --release`, default features including `faster_alloc`, and
  no added LTO. The
  adapter used local path patches for Symbolica, Graphica, and Numerica; its
  lockfile was regenerated because the upstream lock pins dependencies that
  are incompatible with the current workspace.
- Final Symbolica adapter binary SHA-256:
  `181ecbef526d29cdd349084cc9b035d67908b53b203135738e99f3f76bf51e17`.

Every CSV has the exact header `problem_number,FLINT,Symbolica`, 200 finite
positive timing rows numbered 11 through 210, and two corresponding verified
640x480 PNGs. All three normalized Symbolica result streams agree with each
other and the prior verified v28 streams at all 3,360 positions, including
warmups; observed textual differences are factor ordering only. The retained
`.log` files document the original same-seed run that supplied the FLINT
columns and setup configuration; their Symbolica timing summaries predate v28
and are not the plotted Symbolica column. Those logs contain no warning, error,
failed, wrong-answer, or inconsistent-answer record.

See [SHA256SUMS](SHA256SUMS) for hashes of the curated artifacts.
