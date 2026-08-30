# FLINT versus Symbolica on PolyBench 0.4.3

This directory contains a complete rerun of the 5- and 8-variable PolyBench
setups `0001` through `0008`. Its layout matches the published
[PolyBench results](https://github.com/tueda/polybench-result/tree/main/0.4.3):
each setup has the measured CSV, normal run log, summary plot, and
`FLINT_vs_Symbolica.png`. The large generated problem logs are intentionally
omitted.

## Result

Across all 3,200 paired measured problems, the median per-problem
`Symbolica / FLINT` ratio is **0.962614**. Symbolica wins 1,832 of 3,200
problems, the geometric-mean ratio is **0.906571**, and the ratio of summed
measured times is **0.517296**. The median paired ratio is **1.047351** for the
5-variable problems and **0.722134** for the 8-variable problems.

The table is sorted from worst to best for Symbolica. `Paired S/F` is the
median of the 200 ratios `Symbolica_i / FLINT_i`; `total S/F` divides the sums
of the 200 measured times. The independent solver medians can correspond to
different problems and therefore do not, in general, have a ratio equal to
`Paired S/F`.

| Setup | Workload | FLINT median (ms) | Symbolica median (ms) | Paired S/F | Total S/F | S wins | Artifacts |
|---|---|---:|---:|---:|---:|---:|---|
| 05/0007 | sharp trivial factorization | 2.427320 | 4.367500 | 1.595357 | 2.215536 | 56/200 | [plot](05/0007.figures/FLINT_vs_Symbolica.png), [CSV](05/0007.csv), [log](05/0007.log) |
| 05/0003 | uniform trivial factorization | 2.664025 | 3.812000 | 1.425747 | 1.467039 | 26/200 | [plot](05/0003.figures/FLINT_vs_Symbolica.png), [CSV](05/0003.csv), [log](05/0003.log) |
| 05/0008 | sharp nontrivial factorization | 6.883760 | 4.276500 | 1.364144 | 1.940402 | 89/200 | [plot](05/0008.figures/FLINT_vs_Symbolica.png), [CSV](05/0008.csv), [log](05/0008.log) |
| 05/0002 | uniform nontrivial GCD | 6.720045 | 9.067500 | 1.323497 | 1.324827 | 11/200 | [plot](05/0002.figures/FLINT_vs_Symbolica.png), [CSV](05/0002.csv), [log](05/0002.log) |
| 08/0008 | sharp nontrivial factorization | 2.741800 | 3.634000 | 1.234940 | 1.530509 | 84/200 | [plot](08/0008.figures/FLINT_vs_Symbolica.png), [CSV](08/0008.csv), [log](08/0008.log) |
| 05/0006 | sharp nontrivial GCD | 3.812655 | 4.582000 | 1.213492 | 1.140903 | 84/200 | [plot](05/0006.figures/FLINT_vs_Symbolica.png), [CSV](05/0006.csv), [log](05/0006.log) |
| 05/0001 | uniform trivial GCD | 0.345627 | 0.350000 | 1.018250 | 1.017314 | 31/200 | [plot](05/0001.figures/FLINT_vs_Symbolica.png), [CSV](05/0001.csv), [log](05/0001.log) |
| 08/0007 | sharp trivial factorization | 4.756305 | 4.271500 | 1.001295 | 1.322808 | 100/200 | [plot](08/0007.figures/FLINT_vs_Symbolica.png), [CSV](08/0007.csv), [log](08/0007.log) |
| 05/0005 | sharp trivial GCD | 0.284435 | 0.285000 | 0.978281 | 2.103359 | 131/200 | [plot](05/0005.figures/FLINT_vs_Symbolica.png), [CSV](05/0005.csv), [log](05/0005.log) |
| 08/0001 | uniform trivial GCD | 0.522521 | 0.505500 | 0.969194 | 0.956597 | 174/200 | [plot](08/0001.figures/FLINT_vs_Symbolica.png), [CSV](08/0001.csv), [log](08/0001.log) |
| 08/0005 | sharp trivial GCD | 0.493900 | 0.494000 | 0.950340 | 2.096462 | 132/200 | [plot](08/0005.figures/FLINT_vs_Symbolica.png), [CSV](08/0005.csv), [log](08/0005.log) |
| 08/0006 | sharp nontrivial GCD | 5.341565 | 2.857000 | 0.523724 | 0.565561 | 188/200 | [plot](08/0006.figures/FLINT_vs_Symbolica.png), [CSV](08/0006.csv), [log](08/0006.log) |
| 08/0003 | uniform trivial factorization | 4.277095 | 2.126000 | 0.515150 | 0.570522 | 183/200 | [plot](08/0003.figures/FLINT_vs_Symbolica.png), [CSV](08/0003.csv), [log](08/0003.log) |
| 08/0002 | uniform nontrivial GCD | 6.869210 | 3.281000 | 0.484165 | 0.572330 | 184/200 | [plot](08/0002.figures/FLINT_vs_Symbolica.png), [CSV](08/0002.csv), [log](08/0002.log) |
| 08/0004 | uniform nontrivial factorization | 58.536850 | 14.035500 | 0.472376 | 0.193007 | 186/200 | [plot](08/0004.figures/FLINT_vs_Symbolica.png), [CSV](08/0004.csv), [log](08/0004.log) |
| 05/0004 | uniform nontrivial factorization | 56.179300 | 15.056000 | 0.266315 | 0.436369 | 173/200 | [plot](05/0004.figures/FLINT_vs_Symbolica.png), [CSV](05/0004.csv), [log](05/0004.log) |

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
- Seed 42; 10 warmups and 200 measured problems per setup; requested basic
  polynomial sizes 37--50 terms; degrees 22--30 for uniform inputs and 0--30
  for sharp inputs; coefficients `-16384..=16384`; timeout 21,600 seconds per
  solver. Parsing is outside the timed region.
- Sequential, single-core execution pinned to logical CPU 8 with
  `RAYON_NUM_THREADS=1` on an Intel Xeon W-2135, Linux 6.18.37, Python 3.12.13.
- FLINT 3.5.0 built through PolyBench's vcpkg recipe with GCC 15.2.0.
- Symbolica 2.2.0 from the repository revision containing this directory,
  built by the upstream adapter with plain `cargo build --release`, Rust
  1.97.0, default features including `faster_alloc`, and no added LTO. The
  adapter used local path patches for Symbolica, Graphica, and Numerica; its
  lockfile was regenerated because the upstream lock pins dependencies that
  are incompatible with the current workspace.
- Final Symbolica adapter binary SHA-256:
  `ba97e01eee517f2ec43b73f0fbca116959dcd1fc0f3bdf3f16ad95fb2e52834e`.

Every CSV has the exact header `problem_number,FLINT,Symbolica`, 200 finite
positive timing rows numbered 11 through 210, and a corresponding verified
640x480 PNG. The run logs contain no warning, error, failed, wrong-answer, or
inconsistent-answer record. All sixteen final generated problem streams were
also byte-identical to the initial seed-42 streams.

See [SHA256SUMS](SHA256SUMS) for hashes of the curated artifacts.
