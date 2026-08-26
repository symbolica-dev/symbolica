//! Single-core polynomial benchmarks comparing Symbolica with FLINT.

mod support;

use std::env;

use symbolica::coefficient::ConvertToRing;
use symbolica::prelude::*;

use support::cases::{
    EXACT_DIVISION_CASES, ExactDivisionCase, FINITE_FIELD_MULTIPLICATION_CASES, FINITE_FIELDS,
    FactorizationCase, FiniteFieldCase, FiniteFieldMultiplicationCase,
    FiniteFieldMultiplicationInput, GENERATED_FACTOR_CASES, GENERATED_GCD_CASES, GcdCaseConfig,
    INTEGER_MULTIPLICATION_CASES, IntegerMultiplicationCase, PoweredPolynomial, RESULTANT_CASES,
    ResultantCase,
};
use support::flint::{
    FmpzMPoly, FmpzMPolyContext, GcdAlgorithm as FlintGcdAlgorithm, NmodMPoly, NmodMPolyContext,
};
use support::paired::{PairedConfig, run_paired};
use support::polybench_cases::{
    POLYBENCH_FACTOR_CASES, POLYBENCH_GCD_CASES, POLYBENCH_SEED, POLYBENCH_SOURCE_COMMIT,
    PolybenchFactorCase, PolybenchGcdCase, PolybenchGcdConstruction,
};
use support::symbolica::{
    BENCHMARK_NAMESPACE, GcdAlgorithm as SymbolicaGcdAlgorithm, IntegerPolynomial,
    PolybenchIntegerPolynomial, configure_factorization_auto,
    configure_gcd as configure_symbolica_gcd, factorization_input as symbolica_factorization_input,
    finite_pair as symbolica_finite_pair, gcd_case_config,
    gcd_factors_for as symbolica_gcd_factors_for, parse_integer_polynomial,
    parse_polybench_integer_polynomial, polybench_factor_input as symbolica_polybench_factor_input,
    polybench_gcd_inputs as symbolica_polybench_gcd_inputs, powered_pair as symbolica_powered_pair,
    resultant_inputs as symbolica_resultant_inputs,
    validate_factorization as validate_symbolica_factorization, validate_polybench_factorization,
    validate_polybench_gcd,
};

fn main() {
    support::flint::initialize_single_thread();
    support::symbolica::initialize_single_thread();

    eprintln!(
        "Symbolica {}; FLINT {}; one thread per implementation",
        support::symbolica::workspace_version_label(),
        support::flint::version()
    );
    eprintln!("polybench fixtures {POLYBENCH_SOURCE_COMMIT}; seed {POLYBENCH_SEED}");

    if env::var_os("SYMBOLICA_FLINT_BENCH_PAIRED").is_some() {
        paired_benchmarks();
    } else {
        divan::main();
    }
}

fn flint_powered_integer<'context>(
    context: &'context FmpzMPolyContext,
    polynomial: PoweredPolynomial,
) -> FmpzMPoly<'context> {
    let result = context
        .parse(polynomial.base)
        .unwrap()
        .pow(polynomial.power as u64)
        .unwrap();
    if polynomial.constant == 0 {
        result
    } else {
        result.add_si(polynomial.constant as flint3_sys::slong)
    }
}

fn flint_powered_modular<'context>(
    context: &'context NmodMPolyContext,
    polynomial: PoweredPolynomial,
) -> NmodMPoly<'context> {
    let result = context
        .parse(polynomial.base)
        .unwrap()
        .pow(polynomial.power as u64)
        .unwrap();
    match polynomial.constant.cmp(&0) {
        std::cmp::Ordering::Less => result
            .sub_ui(polynomial.constant.unsigned_abs() as u64)
            .unwrap(),
        std::cmp::Ordering::Equal => result,
        std::cmp::Ordering::Greater => result.add_ui(polynomial.constant as u64).unwrap(),
    }
}

fn assert_integer_results_equal(flint: &FmpzMPoly<'_>, symbolica: &IntegerPolynomial) {
    let mut polynomials = [
        parse_integer_polynomial(&flint.to_pretty_string().unwrap()),
        symbolica.clone(),
    ];
    MultivariatePolynomial::unify_variables_list(&mut polynomials);
    assert!((&polynomials[0] - &polynomials[1]).is_zero());
}

fn assert_polybench_integer_results_equal(
    flint: &FmpzMPoly<'_>,
    symbolica: &PolybenchIntegerPolynomial,
    variables: &[&str],
) {
    let parsed = parse_polybench_integer_polynomial(&flint.to_pretty_string().unwrap(), variables);
    assert_eq!(parsed, *symbolica);
}

fn assert_polybench_integer_gcds_associate(
    flint: &FmpzMPoly<'_>,
    symbolica: &PolybenchIntegerPolynomial,
    variables: &[&str],
) {
    let mut polynomials = [
        parse_polybench_integer_polynomial(&flint.to_pretty_string().unwrap(), variables),
        symbolica.clone(),
    ];
    MultivariatePolynomial::unify_variables_list(&mut polynomials);
    assert!(
        polynomials[0] == polynomials[1] || polynomials[0] == -polynomials[1].clone(),
        "FLINT and Symbolica returned nonassociate integer GCDs:\nFLINT: {:?}\nSymbolica: {:?}",
        polynomials[0],
        polynomials[1],
    );
}

fn flint_polybench_gcd_inputs<'context>(
    context: &'context FmpzMPolyContext,
    case: PolybenchGcdCase,
) -> (FmpzMPoly<'context>, FmpzMPoly<'context>) {
    match case.construction {
        PolybenchGcdConstruction::Nontrivial {
            left_cofactor,
            right_cofactor,
            common_factor,
            ..
        } => {
            let common = context.parse(common_factor).unwrap();
            (
                context.parse(left_cofactor).unwrap().mul(&common),
                context.parse(right_cofactor).unwrap().mul(&common),
            )
        }
        PolybenchGcdConstruction::Trivial {
            left_factors,
            right_factors,
            ..
        } => (
            context
                .parse(left_factors[0])
                .unwrap()
                .mul(&context.parse(left_factors[1]).unwrap()),
            context
                .parse(right_factors[0])
                .unwrap()
                .mul(&context.parse(right_factors[1]).unwrap()),
        ),
    }
}

fn flint_polybench_factor_input<'context>(
    context: &'context FmpzMPolyContext,
    case: PolybenchFactorCase,
) -> FmpzMPoly<'context> {
    context
        .parse(case.factors[0])
        .unwrap()
        .mul(&context.parse(case.factors[1]).unwrap())
}

fn flint_factorization_input<'context>(
    context: &'context FmpzMPolyContext,
    case: FactorizationCase,
) -> FmpzMPoly<'context> {
    flint_powered_integer(context, case.left).mul(&flint_powered_integer(context, case.right))
}

fn assert_modular_results_equal<R>(
    flint: &NmodMPoly<'_>,
    symbolica: &MultivariatePolynomial<R, u16>,
    ring: &R,
    description: &str,
) where
    R: EuclideanDomain + ConvertToRing,
{
    let mut polynomials = [
        Atom::parse(
            &flint.to_pretty_string().unwrap(),
            BENCHMARK_NAMESPACE,
            ParseSettings::default(),
        )
        .unwrap()
        .to_polynomial(ring, None),
        symbolica.clone(),
    ];
    MultivariatePolynomial::unify_variables_list(&mut polynomials);
    let difference = &polynomials[0] - &polynomials[1];
    assert!(
        difference.is_zero(),
        "FLINT and Symbolica {description} differ (FLINT terms: {}, Symbolica terms: {}, difference terms: {})",
        polynomials[0].nterms(),
        polynomials[1].nterms(),
        difference.nterms(),
    );
}

fn flint_finite_pair<'context>(
    context: &'context NmodMPolyContext,
    case: FiniteFieldMultiplicationCase,
) -> [NmodMPoly<'context>; 2] {
    match case.input {
        FiniteFieldMultiplicationInput::DenseUnivariate { .. } => {
            let (left, right) = case.input.dense_univariate_coefficients().unwrap();
            [
                context.dense_univariate(&left).unwrap(),
                context.dense_univariate(&right).unwrap(),
            ]
        }
        FiniteFieldMultiplicationInput::Powered { left, right } => [
            flint_powered_modular(context, left),
            flint_powered_modular(context, right),
        ],
    }
}

#[divan::bench_group(sample_count = 5, sample_size = 1, skip_ext_time)]
mod integer_multiplication {
    use super::*;

    #[divan::bench(args = INTEGER_MULTIPLICATION_CASES)]
    fn symbolica(bencher: divan::Bencher, case: IntegerMultiplicationCase) {
        support::symbolica::benchmark_integer_multiplication(bencher, case);
    }

    #[divan::bench(args = INTEGER_MULTIPLICATION_CASES)]
    fn flint(bencher: divan::Bencher, case: IntegerMultiplicationCase) {
        let context = FmpzMPolyContext::new(case.variables).unwrap();
        let left = flint_powered_integer(&context, case.left);
        let right = flint_powered_integer(&context, case.right);
        let [symbolica_left, symbolica_right] = symbolica_powered_pair(&Z, case.left, case.right);
        let expected = &symbolica_left * &symbolica_right;
        assert_integer_results_equal(&left.mul(&right), &expected);
        bencher.bench_local(|| left.mul(&right));
    }
}

#[divan::bench_group(sample_count = 5, sample_size = 1, skip_ext_time)]
mod exact_division {
    use super::*;

    #[divan::bench(args = EXACT_DIVISION_CASES)]
    fn symbolica(bencher: divan::Bencher, case: ExactDivisionCase) {
        support::symbolica::benchmark_exact_division(bencher, case);
    }

    #[divan::bench(args = EXACT_DIVISION_CASES)]
    fn flint(bencher: divan::Bencher, case: ExactDivisionCase) {
        let context = FmpzMPolyContext::new(case.variables).unwrap();
        let quotient = flint_powered_integer(&context, case.quotient);
        let divisor = flint_powered_integer(&context, case.divisor);
        let dividend = quotient.mul(&divisor);
        assert!(dividend.exact_div(&divisor).unwrap().equals(&quotient));
        bencher.bench_local(|| dividend.exact_div(&divisor).unwrap());
    }
}

fn benchmark_flint_finite(
    bencher: divan::Bencher<'_, '_>,
    modulus: u64,
    case: FiniteFieldMultiplicationCase,
) {
    let context = NmodMPolyContext::new(case.variables, modulus).unwrap();
    let [left, right] = flint_finite_pair(&context, case);
    let actual = left.mul(&right);
    if modulus <= u32::MAX as u64 {
        let ring = Zp::new(modulus as u32);
        let [symbolica_left, symbolica_right] = symbolica_finite_pair(&ring, case);
        assert_modular_results_equal(&left, &symbolica_left, &ring, "left operands");
        assert_modular_results_equal(&right, &symbolica_right, &ring, "right operands");
        assert_modular_results_equal(
            &actual,
            &(&symbolica_left * &symbolica_right),
            &ring,
            "products",
        );
    } else {
        let ring = Zp64::new(modulus);
        let [symbolica_left, symbolica_right] = symbolica_finite_pair(&ring, case);
        assert_modular_results_equal(&left, &symbolica_left, &ring, "left operands");
        assert_modular_results_equal(&right, &symbolica_right, &ring, "right operands");
        assert_modular_results_equal(
            &actual,
            &(&symbolica_left * &symbolica_right),
            &ring,
            "products",
        );
    }
    bencher.bench_local(|| left.mul(&right));
}

#[divan::bench_group(sample_count = 3, sample_size = 1, skip_ext_time)]
mod finite_field_17_multiplication {
    use super::*;

    #[divan::bench(args = FINITE_FIELD_MULTIPLICATION_CASES)]
    fn symbolica(bencher: divan::Bencher, case: FiniteFieldMultiplicationCase) {
        support::symbolica::benchmark_finite_multiplication(
            bencher,
            &Zp::new(FINITE_FIELDS[0].modulus as u32),
            case,
        );
    }

    #[divan::bench(args = FINITE_FIELD_MULTIPLICATION_CASES)]
    fn flint(bencher: divan::Bencher, case: FiniteFieldMultiplicationCase) {
        benchmark_flint_finite(bencher, FINITE_FIELDS[0].modulus, case);
    }
}

#[divan::bench_group(sample_count = 3, sample_size = 1, skip_ext_time)]
mod finite_field_64_multiplication {
    use super::*;

    #[divan::bench(args = FINITE_FIELD_MULTIPLICATION_CASES)]
    fn symbolica(bencher: divan::Bencher, case: FiniteFieldMultiplicationCase) {
        support::symbolica::benchmark_finite_multiplication(
            bencher,
            &Zp64::new(FINITE_FIELDS[1].modulus),
            case,
        );
    }

    #[divan::bench(args = FINITE_FIELD_MULTIPLICATION_CASES)]
    fn flint(bencher: divan::Bencher, case: FiniteFieldMultiplicationCase) {
        benchmark_flint_finite(bencher, FINITE_FIELDS[1].modulus, case);
    }
}

#[divan::bench_group(sample_count = 3, sample_size = 1, skip_ext_time)]
mod resultants {
    use super::*;

    #[divan::bench(args = RESULTANT_CASES)]
    fn symbolica_brown(bencher: divan::Bencher, case: ResultantCase) {
        support::symbolica::benchmark_resultant_brown(bencher, case);
    }

    #[divan::bench(args = RESULTANT_CASES)]
    fn symbolica_ducos(bencher: divan::Bencher, case: ResultantCase) {
        support::symbolica::benchmark_resultant_default(bencher, case);
    }

    #[divan::bench(args = RESULTANT_CASES)]
    fn symbolica_crt(bencher: divan::Bencher, case: ResultantCase) {
        support::symbolica::benchmark_resultant_crt(bencher, case);
    }

    #[divan::bench(args = RESULTANT_CASES)]
    fn flint(bencher: divan::Bencher, case: ResultantCase) {
        let context = FmpzMPolyContext::new(case.variables).unwrap();
        let left = context.parse(case.left).unwrap();
        let right = context.parse(case.right).unwrap();
        let (symbolica_left, symbolica_right) = symbolica_resultant_inputs(case);
        let expected = symbolica_left.resultant(&symbolica_right);
        let actual = left.resultant(&right, case.elimination_variable).unwrap();
        assert_integer_results_equal(&actual, &expected);
        bencher.bench_local(|| left.resultant(&right, case.elimination_variable).unwrap());
    }
}

fn benchmark_flint_gcd(bencher: divan::Bencher<'_, '_>, algorithm: FlintGcdAlgorithm) {
    benchmark_flint_gcd_for(bencher, algorithm, gcd_case_config());
}

fn benchmark_flint_gcd_for(
    bencher: divan::Bencher<'_, '_>,
    algorithm: FlintGcdAlgorithm,
    config: GcdCaseConfig,
) {
    let generated = config.generate().unwrap();
    let context = FmpzMPolyContext::new(generated.variables()).unwrap();
    let left_cofactor = context.parse(&generated.left_cofactor).unwrap();
    let right_cofactor = context.parse(&generated.right_cofactor).unwrap();
    let expected = context.parse(&generated.gcd).unwrap();
    let left = left_cofactor.mul(&expected);
    let right = right_cofactor.mul(&expected);
    assert!(left.gcd(&right, algorithm).unwrap().equals(&expected));
    bencher.bench_local(|| left.gcd(&right, algorithm).unwrap());
}

#[divan::bench_group(sample_count = 1, sample_size = 1, skip_ext_time)]
mod polynomial_gcd {
    use super::*;

    #[divan::bench]
    fn symbolica_products(bencher: divan::Bencher) {
        support::symbolica::benchmark_gcd_products(bencher);
    }

    #[divan::bench]
    fn flint_products(bencher: divan::Bencher) {
        let generated = gcd_case_config().generate().unwrap();
        let context = FmpzMPolyContext::new(generated.variables()).unwrap();
        let left = context.parse(&generated.left_cofactor).unwrap();
        let right = context.parse(&generated.right_cofactor).unwrap();
        let gcd = context.parse(&generated.gcd).unwrap();
        bencher.bench_local(|| (left.mul(&gcd), right.mul(&gcd)));
    }

    #[divan::bench]
    fn symbolica_auto(bencher: divan::Bencher) {
        support::symbolica::benchmark_gcd(bencher, SymbolicaGcdAlgorithm::Auto);
    }

    #[divan::bench]
    fn symbolica_hu(bencher: divan::Bencher) {
        support::symbolica::benchmark_gcd(bencher, SymbolicaGcdAlgorithm::Hu);
    }

    #[divan::bench]
    fn symbolica_zippel(bencher: divan::Bencher) {
        support::symbolica::benchmark_gcd(bencher, SymbolicaGcdAlgorithm::Zippel);
    }

    #[divan::bench]
    fn flint_auto(bencher: divan::Bencher) {
        benchmark_flint_gcd(bencher, FlintGcdAlgorithm::Auto);
    }

    #[divan::bench]
    fn flint_hensel(bencher: divan::Bencher) {
        benchmark_flint_gcd(bencher, FlintGcdAlgorithm::Hensel);
    }

    #[divan::bench]
    fn flint_zippel(bencher: divan::Bencher) {
        benchmark_flint_gcd(bencher, FlintGcdAlgorithm::Zippel);
    }

    #[divan::bench]
    fn flint_zippel2(bencher: divan::Bencher) {
        benchmark_flint_gcd(bencher, FlintGcdAlgorithm::Zippel2);
    }
}

#[divan::bench_group(sample_count = 1, sample_size = 1, skip_ext_time)]
mod generated_gcd_regimes {
    use super::*;

    #[divan::bench(args = GENERATED_GCD_CASES)]
    fn symbolica_products(bencher: divan::Bencher, case: GcdCaseConfig) {
        support::symbolica::benchmark_gcd_products_for(bencher, case);
    }

    #[divan::bench(args = GENERATED_GCD_CASES)]
    fn flint_products(bencher: divan::Bencher, case: GcdCaseConfig) {
        let generated = case.generate().unwrap();
        let context = FmpzMPolyContext::new(generated.variables()).unwrap();
        let left = context.parse(&generated.left_cofactor).unwrap();
        let right = context.parse(&generated.right_cofactor).unwrap();
        let gcd = context.parse(&generated.gcd).unwrap();
        bencher.bench_local(|| (left.mul(&gcd), right.mul(&gcd)));
    }

    #[divan::bench(args = GENERATED_GCD_CASES)]
    fn symbolica_gcd(bencher: divan::Bencher, case: GcdCaseConfig) {
        support::symbolica::benchmark_gcd_for(bencher, SymbolicaGcdAlgorithm::Auto, case);
    }

    #[divan::bench(args = GENERATED_GCD_CASES)]
    fn flint_gcd(bencher: divan::Bencher, case: GcdCaseConfig) {
        benchmark_flint_gcd_for(bencher, FlintGcdAlgorithm::Auto, case);
    }
}

#[divan::bench_group(sample_count = 3, sample_size = 1, skip_ext_time)]
mod generated_factor_products {
    use super::*;

    #[divan::bench(args = GENERATED_FACTOR_CASES)]
    fn symbolica(bencher: divan::Bencher, case: FactorizationCase) {
        support::symbolica::benchmark_factor_product(bencher, case);
    }

    #[divan::bench(args = GENERATED_FACTOR_CASES)]
    fn flint(bencher: divan::Bencher, case: FactorizationCase) {
        let context = FmpzMPolyContext::new(case.variables).unwrap();
        let left = flint_powered_integer(&context, case.left);
        let right = flint_powered_integer(&context, case.right);
        let expected = symbolica_factorization_input(case);
        assert_integer_results_equal(&left.mul(&right), &expected);
        drop(expected);
        bencher.bench_local(|| left.mul(&right));
    }
}

#[divan::bench_group(sample_count = 3, sample_size = 1, skip_ext_time)]
mod generated_factorization {
    use super::*;

    #[divan::bench(args = GENERATED_FACTOR_CASES)]
    fn symbolica(bencher: divan::Bencher, case: FactorizationCase) {
        support::symbolica::benchmark_factorization(bencher, case);
    }

    #[divan::bench(args = GENERATED_FACTOR_CASES)]
    fn flint(bencher: divan::Bencher, case: FactorizationCase) {
        let context = FmpzMPolyContext::new(case.variables).unwrap();
        let input = flint_factorization_input(&context, case);
        let expected = symbolica_factorization_input(case);
        assert_integer_results_equal(&input, &expected);
        let factors = input.factor().unwrap();
        assert!(
            factors.len() >= 2,
            "the known reducible generated input was not split"
        );
        assert!(factors.expand().unwrap().equals(&input));
        drop((expected, factors));
        bencher.bench_local(|| input.factor().unwrap());
    }
}

#[divan::bench_group(sample_count = 5, sample_size = 1, skip_ext_time)]
mod polybench_gcd_products {
    use super::*;

    #[divan::bench(args = POLYBENCH_GCD_CASES)]
    fn symbolica(bencher: divan::Bencher, case: PolybenchGcdCase) {
        support::symbolica::benchmark_polybench_gcd_products(bencher, case);
    }

    #[divan::bench(args = POLYBENCH_GCD_CASES)]
    fn flint(bencher: divan::Bencher, case: PolybenchGcdCase) {
        let context = FmpzMPolyContext::new(case.variables()).unwrap();
        let (expected_left, expected_right, _) = symbolica_polybench_gcd_inputs(case);
        match case.construction {
            PolybenchGcdConstruction::Nontrivial {
                left_cofactor,
                right_cofactor,
                common_factor,
                ..
            } => {
                let left = context.parse(left_cofactor).unwrap();
                let right = context.parse(right_cofactor).unwrap();
                let common = context.parse(common_factor).unwrap();
                assert_polybench_integer_results_equal(
                    &left.mul(&common),
                    &expected_left,
                    case.variables(),
                );
                assert_polybench_integer_results_equal(
                    &right.mul(&common),
                    &expected_right,
                    case.variables(),
                );
                drop((expected_left, expected_right));
                bencher.bench_local(|| (left.mul(&common), right.mul(&common)));
            }
            PolybenchGcdConstruction::Trivial {
                left_factors,
                right_factors,
                ..
            } => {
                let left_a = context.parse(left_factors[0]).unwrap();
                let left_b = context.parse(left_factors[1]).unwrap();
                let right_a = context.parse(right_factors[0]).unwrap();
                let right_b = context.parse(right_factors[1]).unwrap();
                assert_polybench_integer_results_equal(
                    &left_a.mul(&left_b),
                    &expected_left,
                    case.variables(),
                );
                assert_polybench_integer_results_equal(
                    &right_a.mul(&right_b),
                    &expected_right,
                    case.variables(),
                );
                drop((expected_left, expected_right));
                bencher.bench_local(|| (left_a.mul(&left_b), right_a.mul(&right_b)));
            }
        }
    }
}

#[divan::bench_group(sample_count = 5, sample_size = 1, skip_ext_time)]
mod polybench_factor_products {
    use super::*;

    #[divan::bench(args = POLYBENCH_FACTOR_CASES)]
    fn symbolica(bencher: divan::Bencher, case: PolybenchFactorCase) {
        support::symbolica::benchmark_polybench_factor_product(bencher, case);
    }

    #[divan::bench(args = POLYBENCH_FACTOR_CASES)]
    fn flint(bencher: divan::Bencher, case: PolybenchFactorCase) {
        let context = FmpzMPolyContext::new(case.variables()).unwrap();
        let left = context.parse(case.factors[0]).unwrap();
        let right = context.parse(case.factors[1]).unwrap();
        let expected = symbolica_polybench_factor_input(case);
        assert_polybench_integer_results_equal(&left.mul(&right), &expected, case.variables());
        drop(expected);
        bencher.bench_local(|| left.mul(&right));
    }
}

#[divan::bench_group(sample_count = 3, sample_size = 1, skip_ext_time)]
mod polybench_gcd {
    use super::*;

    #[divan::bench(args = POLYBENCH_GCD_CASES)]
    fn symbolica(bencher: divan::Bencher, case: PolybenchGcdCase) {
        support::symbolica::benchmark_polybench_gcd(bencher, case);
    }

    #[divan::bench(args = POLYBENCH_GCD_CASES)]
    fn flint(bencher: divan::Bencher, case: PolybenchGcdCase) {
        let context = FmpzMPolyContext::new(case.variables()).unwrap();
        let (left, right) = flint_polybench_gcd_inputs(&context, case);
        let (expected_left, expected_right, known_divisor) = symbolica_polybench_gcd_inputs(case);
        assert_polybench_integer_results_equal(&left, &expected_left, case.variables());
        assert_polybench_integer_results_equal(&right, &expected_right, case.variables());
        let reference = expected_left.gcd(&expected_right);
        validate_polybench_gcd(&expected_left, &expected_right, &known_divisor, &reference);
        let actual = left.gcd(&right, FlintGcdAlgorithm::Auto).unwrap();
        assert_polybench_integer_gcds_associate(&actual, &reference, case.variables());
        drop((
            expected_left,
            expected_right,
            known_divisor,
            reference,
            actual,
        ));
        bencher.bench_local(|| left.gcd(&right, FlintGcdAlgorithm::Auto).unwrap());
    }
}

#[divan::bench_group(sample_count = 3, sample_size = 1, skip_ext_time)]
mod polybench_factorization {
    use super::*;

    #[divan::bench(args = POLYBENCH_FACTOR_CASES)]
    fn symbolica(bencher: divan::Bencher, case: PolybenchFactorCase) {
        support::symbolica::benchmark_polybench_factorization(bencher, case);
    }

    #[divan::bench(args = POLYBENCH_FACTOR_CASES)]
    fn flint(bencher: divan::Bencher, case: PolybenchFactorCase) {
        let context = FmpzMPolyContext::new(case.variables()).unwrap();
        let input = flint_polybench_factor_input(&context, case);
        let expected = symbolica_polybench_factor_input(case);
        assert_polybench_integer_results_equal(&input, &expected, case.variables());
        assert_eq!(input.len(), case.expected_input_terms);
        let factors = input.factor().unwrap();
        assert!(
            factors.len() >= 2,
            "the known reducible input was not split"
        );
        assert!(factors.expand().unwrap().equals(&input));
        drop((expected, factors));
        bencher.bench_local(|| input.factor().unwrap());
    }
}

fn paired_benchmarks() {
    paired_integer_multiplication();
    paired_exact_division();
    paired_finite_field(&Zp::new(FINITE_FIELDS[0].modulus as u32), FINITE_FIELDS[0]);
    paired_finite_field(&Zp64::new(FINITE_FIELDS[1].modulus), FINITE_FIELDS[1]);
    paired_resultants();
    paired_gcd();
    paired_generated_gcd_regimes();
    paired_generated_factorization();
    paired_polybench();
}

fn paired_integer_multiplication() {
    for case in INTEGER_MULTIPLICATION_CASES {
        let config = PairedConfig::from_env(case.default_samples);
        if !config.matches(case.name) {
            continue;
        }
        let [symbolica_left, symbolica_right] = symbolica_powered_pair(&Z, case.left, case.right);
        let context = FmpzMPolyContext::new(case.variables).unwrap();
        let flint_left = flint_powered_integer(&context, case.left);
        let flint_right = flint_powered_integer(&context, case.right);
        let symbolica_expected = &symbolica_left * &symbolica_right;
        assert_integer_results_equal(&flint_left.mul(&flint_right), &symbolica_expected);
        run_paired(
            &config,
            case.name,
            || &symbolica_left * &symbolica_right,
            || flint_left.mul(&flint_right),
        );
    }
}

fn paired_exact_division() {
    for case in EXACT_DIVISION_CASES {
        let config = PairedConfig::from_env(case.default_samples);
        if !config.matches(case.name) {
            continue;
        }
        let [symbolica_quotient, symbolica_divisor] =
            symbolica_powered_pair(&Z, case.quotient, case.divisor);
        let symbolica_dividend = &symbolica_quotient * &symbolica_divisor;
        let mut owned_dividends = (0..=config.samples())
            .map(|_| symbolica_dividend.clone())
            .collect::<Vec<_>>();

        let context = FmpzMPolyContext::new(case.variables).unwrap();
        let flint_quotient = flint_powered_integer(&context, case.quotient);
        let flint_divisor = flint_powered_integer(&context, case.divisor);
        let flint_dividend = flint_quotient.mul(&flint_divisor);
        assert!(
            flint_dividend
                .exact_div(&flint_divisor)
                .unwrap()
                .equals(&flint_quotient)
        );
        run_paired(
            &config,
            case.name,
            || {
                owned_dividends
                    .pop()
                    .unwrap()
                    .try_div_owned(&symbolica_divisor)
                    .unwrap()
            },
            || flint_dividend.exact_div(&flint_divisor).unwrap(),
        );
    }
}

fn paired_finite_field<R>(ring: &R, field: FiniteFieldCase)
where
    R: EuclideanDomain + ConvertToRing,
{
    for case in FINITE_FIELD_MULTIPLICATION_CASES {
        let name = case.display_name(field);
        let config = PairedConfig::from_env(case.default_samples);
        if !config.matches(&name) {
            continue;
        }
        let [symbolica_left, symbolica_right] = symbolica_finite_pair(ring, case);
        let context = NmodMPolyContext::new(case.variables, field.modulus).unwrap();
        let [flint_left, flint_right] = flint_finite_pair(&context, case);
        let symbolica_expected = &symbolica_left * &symbolica_right;
        assert_modular_results_equal(&flint_left, &symbolica_left, ring, "left operands");
        assert_modular_results_equal(&flint_right, &symbolica_right, ring, "right operands");
        assert_modular_results_equal(
            &flint_left.mul(&flint_right),
            &symbolica_expected,
            ring,
            "products",
        );
        run_paired(
            &config,
            &name,
            || &symbolica_left * &symbolica_right,
            || flint_left.mul(&flint_right),
        );
    }
}

fn paired_resultants() {
    for case in RESULTANT_CASES {
        let config = PairedConfig::from_env(case.default_samples);
        let brown_name = format!("resultant Brown: {}", case.name);
        let ducos_name = format!("resultant Ducos: {}", case.name);
        let crt_name = format!("resultant CRT: {}", case.name);
        let measure_brown = config.matches(&brown_name);
        let measure_ducos = config.matches(&ducos_name);
        let measure_crt = config.matches(&crt_name);
        if !measure_brown && !measure_ducos && !measure_crt {
            continue;
        }
        let (symbolica_left, symbolica_right) = symbolica_resultant_inputs(case);
        let symbolica_expected = symbolica_left.resultant(&symbolica_right);
        if measure_brown {
            assert_eq!(
                symbolica_left.resultant_brown(&symbolica_right),
                symbolica_expected
            );
        }
        if measure_crt {
            assert_eq!(
                symbolica_left.resultant_crt(&symbolica_right),
                symbolica_expected
            );
        }
        let context = FmpzMPolyContext::new(case.variables).unwrap();
        let flint_left = context.parse(case.left).unwrap();
        let flint_right = context.parse(case.right).unwrap();
        let flint_expected = flint_left
            .resultant(&flint_right, case.elimination_variable)
            .unwrap();
        assert_integer_results_equal(&flint_expected, &symbolica_expected);
        if measure_brown {
            run_paired(
                &config,
                &brown_name,
                || symbolica_left.resultant_brown(&symbolica_right),
                || {
                    flint_left
                        .resultant(&flint_right, case.elimination_variable)
                        .unwrap()
                },
            );
        }
        if measure_ducos {
            run_paired(
                &config,
                &ducos_name,
                || symbolica_left.resultant(&symbolica_right),
                || {
                    flint_left
                        .resultant(&flint_right, case.elimination_variable)
                        .unwrap()
                },
            );
        }
        if measure_crt {
            run_paired(
                &config,
                &crt_name,
                || symbolica_left.resultant_crt(&symbolica_right),
                || {
                    flint_left
                        .resultant(&flint_right, case.elimination_variable)
                        .unwrap()
                },
            );
        }
    }
}

fn paired_gcd() {
    paired_gcd_case(gcd_case_config(), "GCD");
}

fn paired_generated_gcd_regimes() {
    for case in GENERATED_GCD_CASES {
        paired_gcd_case(case, "generated GCD");
    }
}

fn paired_generated_factorization() {
    configure_factorization_auto();
    for case in GENERATED_FACTOR_CASES {
        let config = PairedConfig::from_env(case.default_samples);
        let product_name = format!("generated factor product: {case}");
        let factor_name = format!("generated factorization: {case}");
        if !config.matches(&product_name) && !config.matches(&factor_name) {
            continue;
        }

        let [symbolica_left, symbolica_right] = symbolica_powered_pair(&Z, case.left, case.right);
        let symbolica_input = &symbolica_left * &symbolica_right;
        let context = FmpzMPolyContext::new(case.variables).unwrap();
        let flint_left = flint_powered_integer(&context, case.left);
        let flint_right = flint_powered_integer(&context, case.right);
        let flint_input = flint_left.mul(&flint_right);
        assert_integer_results_equal(&flint_input, &symbolica_input);

        if config.matches(&product_name) {
            run_paired(
                &config,
                &product_name,
                || &symbolica_left * &symbolica_right,
                || flint_left.mul(&flint_right),
            );
        }

        if config.matches(&factor_name) {
            let symbolica_factors = symbolica_input.factor();
            validate_symbolica_factorization(&symbolica_input, &symbolica_factors);
            let flint_factors = flint_input.factor().unwrap();
            assert!(
                flint_factors.len() >= 2,
                "the known reducible generated input was not split"
            );
            assert!(flint_factors.expand().unwrap().equals(&flint_input));
            drop((symbolica_factors, flint_factors));
            run_paired(
                &config,
                &factor_name,
                || symbolica_input.factor(),
                || flint_input.factor().unwrap(),
            );
        }
    }
}

fn paired_gcd_case(case: GcdCaseConfig, label: &str) {
    let generated = case.generate().unwrap();
    let config = PairedConfig::from_env(1);
    let product_name = format!("{label} products: {}", generated.display_name());
    let gcd_name = format!("{label} auto: {}", generated.display_name());
    if !config.matches(&product_name) && !config.matches(&gcd_name) {
        return;
    }

    configure_symbolica_gcd(SymbolicaGcdAlgorithm::Auto);
    let [
        symbolica_left_cofactor,
        symbolica_right_cofactor,
        symbolica_expected,
    ] = symbolica_gcd_factors_for(case);
    let context = FmpzMPolyContext::new(generated.variables()).unwrap();
    let flint_gcd = context.parse(&generated.gcd).unwrap();
    let flint_left_cofactor = context.parse(&generated.left_cofactor).unwrap();
    let flint_right_cofactor = context.parse(&generated.right_cofactor).unwrap();
    run_paired(
        &config,
        &product_name,
        || {
            (
                &symbolica_left_cofactor * &symbolica_expected,
                &symbolica_right_cofactor * &symbolica_expected,
            )
        },
        || {
            (
                flint_left_cofactor.mul(&flint_gcd),
                flint_right_cofactor.mul(&flint_gcd),
            )
        },
    );

    if !config.matches(&gcd_name) {
        return;
    }

    let symbolica_left = &symbolica_left_cofactor * &symbolica_expected;
    let symbolica_right = &symbolica_right_cofactor * &symbolica_expected;
    let flint_left = flint_left_cofactor.mul(&flint_gcd);
    let flint_right = flint_right_cofactor.mul(&flint_gcd);
    let symbolica_actual = symbolica_left.gcd(&symbolica_right);
    assert!((&symbolica_actual - &symbolica_expected).is_zero());
    assert!(
        flint_left
            .gcd(&flint_right, FlintGcdAlgorithm::Auto)
            .unwrap()
            .equals(&flint_gcd)
    );
    run_paired(
        &config,
        &gcd_name,
        || symbolica_left.gcd(&symbolica_right),
        || {
            flint_left
                .gcd(&flint_right, FlintGcdAlgorithm::Auto)
                .unwrap()
        },
    );
}

fn paired_polybench() {
    paired_polybench_gcd();
    paired_polybench_factorization();
}

fn paired_polybench_gcd() {
    for case in POLYBENCH_GCD_CASES {
        let config = PairedConfig::from_env(3);
        let product_name = format!("polybench GCD products: {case}");
        let gcd_name = format!("polybench GCD: {case}");
        if !config.matches(&product_name) && !config.matches(&gcd_name) {
            continue;
        }

        if config.matches(&product_name) {
            let context = FmpzMPolyContext::new(case.variables()).unwrap();
            match case.construction {
                PolybenchGcdConstruction::Nontrivial {
                    left_cofactor,
                    right_cofactor,
                    common_factor,
                    ..
                } => {
                    let symbolica_left =
                        parse_polybench_integer_polynomial(left_cofactor, case.variables());
                    let symbolica_right =
                        parse_polybench_integer_polynomial(right_cofactor, case.variables());
                    let symbolica_common =
                        parse_polybench_integer_polynomial(common_factor, case.variables());
                    let flint_left = context.parse(left_cofactor).unwrap();
                    let flint_right = context.parse(right_cofactor).unwrap();
                    let flint_common = context.parse(common_factor).unwrap();
                    let expected_left = &symbolica_left * &symbolica_common;
                    let expected_right = &symbolica_right * &symbolica_common;
                    assert_polybench_integer_results_equal(
                        &flint_left.mul(&flint_common),
                        &expected_left,
                        case.variables(),
                    );
                    assert_polybench_integer_results_equal(
                        &flint_right.mul(&flint_common),
                        &expected_right,
                        case.variables(),
                    );
                    drop((expected_left, expected_right));
                    run_paired(
                        &config,
                        &product_name,
                        || {
                            (
                                &symbolica_left * &symbolica_common,
                                &symbolica_right * &symbolica_common,
                            )
                        },
                        || {
                            (
                                flint_left.mul(&flint_common),
                                flint_right.mul(&flint_common),
                            )
                        },
                    );
                }
                PolybenchGcdConstruction::Trivial {
                    left_factors,
                    right_factors,
                    ..
                } => {
                    let [symbolica_left_a, symbolica_left_b] = left_factors
                        .map(|factor| parse_polybench_integer_polynomial(factor, case.variables()));
                    let [symbolica_right_a, symbolica_right_b] = right_factors
                        .map(|factor| parse_polybench_integer_polynomial(factor, case.variables()));
                    let flint_left_a = context.parse(left_factors[0]).unwrap();
                    let flint_left_b = context.parse(left_factors[1]).unwrap();
                    let flint_right_a = context.parse(right_factors[0]).unwrap();
                    let flint_right_b = context.parse(right_factors[1]).unwrap();
                    let expected_left = &symbolica_left_a * &symbolica_left_b;
                    let expected_right = &symbolica_right_a * &symbolica_right_b;
                    assert_polybench_integer_results_equal(
                        &flint_left_a.mul(&flint_left_b),
                        &expected_left,
                        case.variables(),
                    );
                    assert_polybench_integer_results_equal(
                        &flint_right_a.mul(&flint_right_b),
                        &expected_right,
                        case.variables(),
                    );
                    drop((expected_left, expected_right));
                    run_paired(
                        &config,
                        &product_name,
                        || {
                            (
                                &symbolica_left_a * &symbolica_left_b,
                                &symbolica_right_a * &symbolica_right_b,
                            )
                        },
                        || {
                            (
                                flint_left_a.mul(&flint_left_b),
                                flint_right_a.mul(&flint_right_b),
                            )
                        },
                    );
                }
            }
        }

        if config.matches(&gcd_name) {
            configure_symbolica_gcd(SymbolicaGcdAlgorithm::Auto);
            let (symbolica_left, symbolica_right, known_divisor) =
                symbolica_polybench_gcd_inputs(case);
            let context = FmpzMPolyContext::new(case.variables()).unwrap();
            let (flint_left, flint_right) = flint_polybench_gcd_inputs(&context, case);
            assert_polybench_integer_results_equal(&flint_left, &symbolica_left, case.variables());
            assert_polybench_integer_results_equal(
                &flint_right,
                &symbolica_right,
                case.variables(),
            );
            let symbolica_actual = symbolica_left.gcd(&symbolica_right);
            validate_polybench_gcd(
                &symbolica_left,
                &symbolica_right,
                &known_divisor,
                &symbolica_actual,
            );
            let flint_actual = flint_left
                .gcd(&flint_right, FlintGcdAlgorithm::Auto)
                .unwrap();
            assert_polybench_integer_gcds_associate(
                &flint_actual,
                &symbolica_actual,
                case.variables(),
            );
            drop((known_divisor, symbolica_actual, flint_actual));
            run_paired(
                &config,
                &gcd_name,
                || symbolica_left.gcd(&symbolica_right),
                || {
                    flint_left
                        .gcd(&flint_right, FlintGcdAlgorithm::Auto)
                        .unwrap()
                },
            );
        }
    }
}

fn paired_polybench_factorization() {
    configure_factorization_auto();
    for case in POLYBENCH_FACTOR_CASES {
        let config = PairedConfig::from_env(2);
        let product_name = format!("polybench factor product: {case}");
        let factor_name = format!("polybench factorization: {case}");
        if !config.matches(&product_name) && !config.matches(&factor_name) {
            continue;
        }

        if config.matches(&product_name) {
            let [symbolica_left, symbolica_right] = case
                .factors
                .map(|factor| parse_polybench_integer_polynomial(factor, case.variables()));
            let context = FmpzMPolyContext::new(case.variables()).unwrap();
            let flint_left = context.parse(case.factors[0]).unwrap();
            let flint_right = context.parse(case.factors[1]).unwrap();
            let expected = &symbolica_left * &symbolica_right;
            assert_polybench_integer_results_equal(
                &flint_left.mul(&flint_right),
                &expected,
                case.variables(),
            );
            drop(expected);
            run_paired(
                &config,
                &product_name,
                || &symbolica_left * &symbolica_right,
                || flint_left.mul(&flint_right),
            );
        }

        if config.matches(&factor_name) {
            let symbolica_input = symbolica_polybench_factor_input(case);
            assert_eq!(symbolica_input.nterms(), case.expected_input_terms);
            let symbolica_factors = symbolica_input.factor();
            validate_polybench_factorization(&symbolica_input, &symbolica_factors);
            drop(symbolica_factors);
            let context = FmpzMPolyContext::new(case.variables()).unwrap();
            let flint_input = flint_polybench_factor_input(&context, case);
            assert_polybench_integer_results_equal(
                &flint_input,
                &symbolica_input,
                case.variables(),
            );
            assert_eq!(flint_input.len(), case.expected_input_terms);
            let flint_factors = flint_input.factor().unwrap();
            assert!(
                flint_factors.len() >= 2,
                "the known reducible input was not split"
            );
            assert!(flint_factors.expand().unwrap().equals(&flint_input));
            drop(flint_factors);
            run_paired(
                &config,
                &factor_name,
                || symbolica_input.factor(),
                || flint_input.factor().unwrap(),
            );
        }
    }
}
