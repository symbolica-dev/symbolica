# Polynomial benchmark fixtures

These 80 cases extend the standard Symbolica/FLINT comparison suite with 36
factorizations, 18 derived GCDs, 25 multivariate resultants and discriminants,
and one large segmentation-encoded resultant. They are included in both the
`flint_comparison` and `symbolica_polynomial` benchmark runners.

Inputs use Brotli-compressed integer term lists with the format header
`polynomial-bench-op 1`. Variable order, unused coordinates, coefficients, and
`u32` exponents are preserved. `manifest.json` records the pinned source commit,
source operand paths, hashes and sizes, derived-operand reconstruction data,
unavailable source operands, and hashes and sizes for the stored fixtures. Only
the format header is normalized during import; polynomial data is unchanged.

GCD cases construct `(P, derivative(P, v))`, using the derivative variable
recorded in the manifest. Symbolica computes factorizations and monic GCDs over
Q, while FLINT runs over Z. Factor timings measure native complete
factorization, including multiplicities.

The resultant `g33S2-res-02-41f64e7684-segenc` is reconstructed from its compact
parent during setup, using 652-bit digits and weights `[0, 217, 0, 0, 1]`,
eliminating coordinate 2. The importer verifies the original 9,930,036-byte
operand against its source SHA-256 before storing the parent. Symbolica's
default integer resultant is compared with FLINT's
`fmpz_poly_resultant_euclidean`.

The other elimination cases run parsed integer operands through the native
multivariate operations. Symbolica uses its default resultant and computes
discriminants through the resultant with the derivative and signed
leading-coefficient division. Setup checks identical inputs, monic GCD equality,
factor reconstruction, irreducible factors and multiplicities up to rational
units, and exact resultant/discriminant outputs against FLINT. Decompression,
parsing, GCD derivative construction, and correctness checks are outside timed
closures.

Run the fixture groups with:

```sh
cargo bench --features flint_benchmarks --bench flint_comparison -- fixture_
cargo bench --features symbolica_benchmarks --bench symbolica_polynomial -- polynomial_fixtures
```

For alternating paired measurements and CSV output:

```sh
SYMBOLICA_FLINT_BENCH_PAIRED=1 \
SYMBOLICA_FLINT_BENCH_FILTER=polynomial/ \
SYMBOLICA_FLINT_BENCH_SAMPLES=3 \
SYMBOLICA_FLINT_BENCH_CSV=1 \
cargo bench --features flint_benchmarks --bench flint_comparison
```

Use `polynomial/gcd/`, `polynomial/factor/`, `polynomial/resultant/`,
`polynomial/discriminant/`, or an operand ID to narrow a paired run. The large
resultant takes tens of seconds per sample. With a system FLINT installation,
use `flint_system_benchmarks` in place of `flint_benchmarks`.

The paired suite also has 12 generated rational GCD controls covering exact
divisibility, proportional inputs, coprime inputs, and shared factors at three
sizes. Set `SYMBOLICA_FLINT_BENCH_FILTER='rational GCD control:'` and use at least
501 samples to check whether shortcuts for failed division slow down successful
division. Known GCDs and reconstruction are checked before timing.

The default benchmark profile disables LTO and uses 16 codegen units for shorter
compilation. To match the library's release settings, add
`CARGO_PROFILE_BENCH_LTO=fat`, `CARGO_PROFILE_BENCH_CODEGEN_UNITS=1`, and
`CARGO_PROFILE_BENCH_INCREMENTAL=false`. Use the same compiler settings when
comparing source changes.

To regenerate fixtures from a checkout containing the pinned source commit:

```sh
python3 benches/import_polynomial_fixtures.py /path/to/source
```

Importing requires git and the `brotli` command. Running the suite requires no
network access.
