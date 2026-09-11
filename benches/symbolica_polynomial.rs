//! Single-core Symbolica polynomial benchmarks.

mod support;

use symbolica::prelude::{Zp, Zp64};

use support::cases::{
    EXACT_DIVISION_CASES, ExactDivisionCase, FINITE_FIELD_MULTIPLICATION_CASES, FINITE_FIELDS,
    FiniteFieldMultiplicationCase, GENERATED_GCD_CASES, GcdCaseConfig,
    INTEGER_MULTIPLICATION_CASES, IntegerMultiplicationCase, RESULTANT_CASES, ResultantCase,
};
use support::polybench_cases::{
    POLYBENCH_FACTOR_CASES, POLYBENCH_GCD_CASES, POLYBENCH_SEED, POLYBENCH_SOURCE_COMMIT,
    PolybenchFactorCase, PolybenchGcdCase,
};
use support::symbolica::{self as symbolica_bench, GcdAlgorithm};

fn main() {
    symbolica_bench::initialize_single_thread();
    eprintln!(
        "Symbolica {}; one Rayon thread",
        symbolica_bench::workspace_version_label()
    );
    eprintln!("polybench fixtures {POLYBENCH_SOURCE_COMMIT}; seed {POLYBENCH_SEED}");
    divan::main();
}

#[divan::bench_group(sample_count = 5, sample_size = 1, skip_ext_time)]
mod integer_multiplication {
    use super::*;

    #[divan::bench(args = INTEGER_MULTIPLICATION_CASES)]
    fn symbolica(bencher: divan::Bencher, case: IntegerMultiplicationCase) {
        symbolica_bench::benchmark_integer_multiplication(bencher, case);
    }
}

#[divan::bench_group(sample_count = 5, sample_size = 1, skip_ext_time)]
mod exact_division {
    use super::*;

    #[divan::bench(args = EXACT_DIVISION_CASES)]
    fn symbolica(bencher: divan::Bencher, case: ExactDivisionCase) {
        symbolica_bench::benchmark_exact_division(bencher, case);
    }
}

#[divan::bench_group(sample_count = 3, sample_size = 1, skip_ext_time)]
mod finite_field_17_multiplication {
    use super::*;

    #[divan::bench(args = FINITE_FIELD_MULTIPLICATION_CASES)]
    fn symbolica(bencher: divan::Bencher, case: FiniteFieldMultiplicationCase) {
        symbolica_bench::benchmark_finite_multiplication(
            bencher,
            &Zp::new(FINITE_FIELDS[0].modulus as u32),
            case,
        );
    }
}

#[divan::bench_group(sample_count = 3, sample_size = 1, skip_ext_time)]
mod finite_field_64_multiplication {
    use super::*;

    #[divan::bench(args = FINITE_FIELD_MULTIPLICATION_CASES)]
    fn symbolica(bencher: divan::Bencher, case: FiniteFieldMultiplicationCase) {
        symbolica_bench::benchmark_finite_multiplication(
            bencher,
            &Zp64::new(FINITE_FIELDS[1].modulus),
            case,
        );
    }
}

#[divan::bench_group(sample_count = 3, sample_size = 1, skip_ext_time)]
mod resultants {
    use super::*;

    #[divan::bench(args = RESULTANT_CASES)]
    fn symbolica_default(bencher: divan::Bencher, case: ResultantCase) {
        symbolica_bench::benchmark_resultant_default(bencher, case);
    }

    #[divan::bench(args = RESULTANT_CASES)]
    fn symbolica_brown(bencher: divan::Bencher, case: ResultantCase) {
        symbolica_bench::benchmark_resultant_brown(bencher, case);
    }

    #[divan::bench(args = RESULTANT_CASES)]
    fn symbolica_crt(bencher: divan::Bencher, case: ResultantCase) {
        symbolica_bench::benchmark_resultant_crt(bencher, case);
    }
}

#[divan::bench_group(sample_count = 1, sample_size = 1, skip_ext_time)]
mod polynomial_gcd {
    use super::*;

    #[divan::bench]
    fn symbolica_products(bencher: divan::Bencher) {
        symbolica_bench::benchmark_gcd_products(bencher);
    }

    #[divan::bench]
    fn symbolica_auto(bencher: divan::Bencher) {
        symbolica_bench::benchmark_gcd(bencher, GcdAlgorithm::Auto);
    }

    #[divan::bench]
    fn symbolica_hu(bencher: divan::Bencher) {
        symbolica_bench::benchmark_gcd(bencher, GcdAlgorithm::Hu);
    }

    #[divan::bench]
    fn symbolica_zippel(bencher: divan::Bencher) {
        symbolica_bench::benchmark_gcd(bencher, GcdAlgorithm::Zippel);
    }
}

#[divan::bench_group(sample_count = 1, sample_size = 1, skip_ext_time)]
mod generated_gcd_regimes {
    use super::*;

    #[divan::bench(args = GENERATED_GCD_CASES)]
    fn products(bencher: divan::Bencher, case: GcdCaseConfig) {
        symbolica_bench::benchmark_gcd_products_for(bencher, case);
    }

    #[divan::bench(args = GENERATED_GCD_CASES)]
    fn gcd(bencher: divan::Bencher, case: GcdCaseConfig) {
        symbolica_bench::benchmark_gcd_for(bencher, GcdAlgorithm::Auto, case);
    }
}

#[divan::bench_group(sample_count = 5, sample_size = 1, skip_ext_time)]
mod polybench_gcd_products {
    use super::*;

    #[divan::bench(args = POLYBENCH_GCD_CASES)]
    fn symbolica(bencher: divan::Bencher, case: PolybenchGcdCase) {
        symbolica_bench::benchmark_polybench_gcd_products(bencher, case);
    }
}

#[divan::bench_group(sample_count = 5, sample_size = 1, skip_ext_time)]
mod polybench_factor_products {
    use super::*;

    #[divan::bench(args = POLYBENCH_FACTOR_CASES)]
    fn symbolica(bencher: divan::Bencher, case: PolybenchFactorCase) {
        symbolica_bench::benchmark_polybench_factor_product(bencher, case);
    }
}

#[divan::bench_group(sample_count = 3, sample_size = 1, skip_ext_time)]
mod polybench_gcd {
    use super::*;

    #[divan::bench(args = POLYBENCH_GCD_CASES)]
    fn symbolica(bencher: divan::Bencher, case: PolybenchGcdCase) {
        symbolica_bench::benchmark_polybench_gcd(bencher, case);
    }
}

#[divan::bench_group(sample_count = 3, sample_size = 1, skip_ext_time)]
mod polybench_factorization {
    use super::*;

    #[divan::bench(args = POLYBENCH_FACTOR_CASES)]
    fn symbolica(bencher: divan::Bencher, case: PolybenchFactorCase) {
        symbolica_bench::benchmark_polybench_factorization(bencher, case);
    }
}
