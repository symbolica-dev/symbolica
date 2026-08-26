//! Symbolica input construction and timed operations shared by benchmark targets.

use std::{
    env,
    process::Command,
    str::FromStr,
    sync::{Arc, OnceLock, atomic::Ordering},
};

use symbolica::GLOBAL_SETTINGS;
use symbolica::coefficient::ConvertToRing;
use symbolica::prelude::*;

use super::cases::{
    ExactDivisionCase, FactorizationCase, FiniteFieldMultiplicationCase,
    FiniteFieldMultiplicationInput, GcdCaseConfig, GcdCaseKind, IntegerMultiplicationCase,
    PoweredPolynomial, ResultantCase,
};
use super::polybench_cases::{PolybenchFactorCase, PolybenchGcdCase, PolybenchGcdConstruction};

pub type IntegerPolynomial = MultivariatePolynomial<IntegerRing, u16>;
pub type IntegerUnivariatePolynomial = UnivariatePolynomial<PolynomialRing<IntegerRing, u16>>;
pub type PolybenchIntegerPolynomial = MultivariatePolynomial<IntegerRing, u8>;

/// Namespace used to give benchmark variables stable symbol identities.
pub const BENCHMARK_NAMESPACE: &str = "polynomial_benchmark";

/// Returns a benchmark label containing the package version, Git revision, and
/// whether the source workspace has uncommitted changes.
pub fn workspace_version_label() -> String {
    let repository = env!("CARGO_MANIFEST_DIR");
    let dirty = Command::new("git")
        .current_dir(repository)
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .is_some_and(|output| !output.stdout.is_empty());
    format!(
        "workspace {} ({}{dirty_suffix})",
        env!("CARGO_PKG_VERSION"),
        env!("SYMBOLICA_VERSION"),
        dirty_suffix = if dirty { "+dirty" } else { "" }
    )
}

/// Configures Rayon's global worker pool before any benchmark input is constructed.
pub fn initialize_single_thread() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build_global()
        .expect("the benchmark must initialize Rayon's global pool first");
}

pub fn parse_integer_polynomial(expression: &str) -> IntegerPolynomial {
    Atom::parse(expression, BENCHMARK_NAMESPACE, ParseSettings::default())
        .unwrap()
        .to_polynomial(&Z, None)
}

/// Parses a polybench fixture with the exact variable order and `u8` exponent
/// representation used by the upstream Symbolica adapter.
pub fn parse_polybench_integer_polynomial(
    expression: &str,
    variable_names: &[&str],
) -> PolybenchIntegerPolynomial {
    let variables = Arc::new(
        variable_names
            .iter()
            .map(|name| PolyVariable::Symbol(symbol!(name)))
            .collect(),
    );
    let polynomial = parse!(expression).to_polynomial(&Z, Some(variables));
    assert_eq!(polynomial.nvars(), variable_names.len());
    polynomial
}

/// Verifies that a returned factor list, including multiplicities, expands to
/// the original polybench input.
pub fn validate_polybench_factorization(
    template: &PolybenchIntegerPolynomial,
    factors: &[(PolybenchIntegerPolynomial, usize)],
) {
    assert!(
        factors
            .iter()
            .filter(|(factor, _)| !factor.is_constant())
            .count()
            >= 2,
        "the known reducible polybench input was not split"
    );
    let expanded = factors
        .iter()
        .fold(template.one(), |product, (factor, power)| {
            product * &factor.pow(*power)
        });
    assert_eq!(expanded, *template);
}

/// Builds the two expanded GCD inputs and a known common divisor from the
/// compact factors stored in a polybench fixture.
pub fn polybench_gcd_inputs(
    case: PolybenchGcdCase,
) -> (
    PolybenchIntegerPolynomial,
    PolybenchIntegerPolynomial,
    PolybenchIntegerPolynomial,
) {
    match case.construction {
        PolybenchGcdConstruction::Nontrivial {
            left_cofactor,
            right_cofactor,
            common_factor,
            expected_factor_terms,
        } => {
            let left_cofactor = parse_polybench_integer_polynomial(left_cofactor, case.variables());
            let right_cofactor =
                parse_polybench_integer_polynomial(right_cofactor, case.variables());
            let common_factor = parse_polybench_integer_polynomial(common_factor, case.variables());
            assert_eq!(
                [
                    left_cofactor.nterms(),
                    right_cofactor.nterms(),
                    common_factor.nterms(),
                ],
                expected_factor_terms
            );
            let known_divisor = if common_factor.lcoeff().is_negative() {
                -common_factor.clone()
            } else {
                common_factor.clone()
            };
            (
                &left_cofactor * &common_factor,
                &right_cofactor * &common_factor,
                known_divisor,
            )
        }
        PolybenchGcdConstruction::Trivial {
            left_factors,
            right_factors,
            expected_factor_terms,
        } => {
            let [left_a, left_b] = left_factors
                .map(|factor| parse_polybench_integer_polynomial(factor, case.variables()));
            let [right_a, right_b] = right_factors
                .map(|factor| parse_polybench_integer_polynomial(factor, case.variables()));
            assert_eq!(
                [
                    left_a.nterms(),
                    left_b.nterms(),
                    right_a.nterms(),
                    right_b.nterms(),
                ],
                expected_factor_terms
            );
            let left = &left_a * &left_b;
            let right = &right_a * &right_b;
            let known_divisor = left.one();
            (left, right, known_divisor)
        }
    }
}

/// Checks that a computed GCD divides both inputs and contains the common
/// divisor used to construct the benchmark problem.
pub fn validate_polybench_gcd(
    left: &PolybenchIntegerPolynomial,
    right: &PolybenchIntegerPolynomial,
    known_divisor: &PolybenchIntegerPolynomial,
    actual: &PolybenchIntegerPolynomial,
) {
    assert!(left.try_div(actual).is_some());
    assert!(right.try_div(actual).is_some());
    assert!(actual.try_div(known_divisor).is_some());
}

/// Builds the expanded factorization input from the two compact factors in a
/// polybench fixture.
pub fn polybench_factor_input(case: PolybenchFactorCase) -> PolybenchIntegerPolynomial {
    let [left, right] = case
        .factors
        .map(|factor| parse_polybench_integer_polynomial(factor, case.variables()));
    assert_eq!([left.nterms(), right.nterms()], case.expected_factor_terms);
    &left * &right
}

/// Measures only the multiplications used to construct a polybench GCD pair.
pub fn benchmark_polybench_gcd_products(bencher: divan::Bencher<'_, '_>, case: PolybenchGcdCase) {
    match case.construction {
        PolybenchGcdConstruction::Nontrivial {
            left_cofactor,
            right_cofactor,
            common_factor,
            expected_factor_terms,
        } => {
            let left = parse_polybench_integer_polynomial(left_cofactor, case.variables());
            let right = parse_polybench_integer_polynomial(right_cofactor, case.variables());
            let common = parse_polybench_integer_polynomial(common_factor, case.variables());
            assert_eq!(
                [left.nterms(), right.nterms(), common.nterms()],
                expected_factor_terms
            );
            let expected_left = &left * &common;
            let expected_right = &right * &common;
            assert_eq!(
                [expected_left.nterms(), expected_right.nterms()],
                case.expected_input_terms
            );
            drop((expected_left, expected_right));
            bencher.bench_local(|| (&left * &common, &right * &common));
        }
        PolybenchGcdConstruction::Trivial {
            left_factors,
            right_factors,
            expected_factor_terms,
        } => {
            let [left_a, left_b] = left_factors
                .map(|factor| parse_polybench_integer_polynomial(factor, case.variables()));
            let [right_a, right_b] = right_factors
                .map(|factor| parse_polybench_integer_polynomial(factor, case.variables()));
            assert_eq!(
                [
                    left_a.nterms(),
                    left_b.nterms(),
                    right_a.nterms(),
                    right_b.nterms(),
                ],
                expected_factor_terms
            );
            let expected_left = &left_a * &left_b;
            let expected_right = &right_a * &right_b;
            assert_eq!(
                [expected_left.nterms(), expected_right.nterms()],
                case.expected_input_terms
            );
            drop((expected_left, expected_right));
            bencher.bench_local(|| (&left_a * &left_b, &right_a * &right_b));
        }
    }
}

/// Measures the multiplication that constructs a polybench factorization
/// input from its two generated factors.
pub fn benchmark_polybench_factor_product(
    bencher: divan::Bencher<'_, '_>,
    case: PolybenchFactorCase,
) {
    let [left, right] = case
        .factors
        .map(|factor| parse_polybench_integer_polynomial(factor, case.variables()));
    assert_eq!([left.nterms(), right.nterms()], case.expected_factor_terms);
    let expected = &left * &right;
    assert_eq!(expected.nterms(), case.expected_input_terms);
    drop(expected);
    bencher.bench_local(|| &left * &right);
}

/// Measures Symbolica's automatic integer-polynomial GCD on an exact
/// polybench fixture. Input expansion and result validation are untimed.
pub fn benchmark_polybench_gcd(bencher: divan::Bencher<'_, '_>, case: PolybenchGcdCase) {
    configure_gcd(GcdAlgorithm::Auto);
    let (left, right, known_divisor) = polybench_gcd_inputs(case);
    assert_eq!(left.nterms(), case.expected_input_terms[0]);
    assert_eq!(right.nterms(), case.expected_input_terms[1]);
    let actual = left.gcd(&right);
    validate_polybench_gcd(&left, &right, &known_divisor, &actual);
    drop((known_divisor, actual));
    bencher.bench_local(|| left.gcd(&right));
}

/// Selects automatic integer factorization, which may choose either the
/// univariate-start or bivariate-start lifting algorithm for each input.
pub fn configure_factorization_auto() {
    GLOBAL_SETTINGS
        .use_univariate_factorization
        .store(true, Ordering::Relaxed);
    GLOBAL_SETTINGS
        .use_bivariate_factorization
        .store(true, Ordering::Relaxed);
}

/// Measures Symbolica's automatic integer factorization on an exact polybench
/// fixture. Input expansion and result validation are untimed.
pub fn benchmark_polybench_factorization(
    bencher: divan::Bencher<'_, '_>,
    case: PolybenchFactorCase,
) {
    configure_factorization_auto();
    let input = polybench_factor_input(case);
    assert_eq!(input.nterms(), case.expected_input_terms);
    let factors = input.factor();
    validate_polybench_factorization(&input, &factors);
    drop(factors);
    bencher.bench_local(|| input.factor());
}

/// Constructs the reducible integer polynomial for a generated factorization case.
pub fn factorization_input(case: FactorizationCase) -> IntegerPolynomial {
    let [left, right] = powered_pair(&Z, case.left, case.right);
    &left * &right
}

/// Verifies that a generated factorization expands to its original polynomial.
pub fn validate_factorization(input: &IntegerPolynomial, factors: &[(IntegerPolynomial, usize)]) {
    let nonconstant_multiplicity = factors
        .iter()
        .filter(|(factor, _)| !factor.is_constant())
        .map(|(_, power)| *power)
        .sum::<usize>();
    assert!(
        nonconstant_multiplicity >= 2,
        "the known reducible generated input was not split"
    );
    let expanded = factors
        .iter()
        .fold(input.one(), |product, (factor, power)| {
            product * &factor.pow(*power)
        });
    assert_eq!(expanded, *input);
}

/// Measures the multiplication used to construct a generated factorization input.
pub fn benchmark_factor_product(bencher: divan::Bencher<'_, '_>, case: FactorizationCase) {
    let [left, right] = powered_pair(&Z, case.left, case.right);
    bencher.bench_local(|| &left * &right);
}

/// Measures automatic integer factorization of a generated low-dimensional input.
pub fn benchmark_factorization(bencher: divan::Bencher<'_, '_>, case: FactorizationCase) {
    configure_factorization_auto();
    let input = factorization_input(case);
    let factors = input.factor();
    validate_factorization(&input, &factors);
    drop(factors);
    bencher.bench_local(|| input.factor());
}

pub fn powered_polynomial<R>(
    ring: &R,
    polynomial: PoweredPolynomial,
) -> MultivariatePolynomial<R, u16>
where
    R: EuclideanDomain + ConvertToRing,
{
    let mut result = Atom::parse(
        polynomial.base,
        BENCHMARK_NAMESPACE,
        ParseSettings::default(),
    )
    .unwrap()
    .to_polynomial(ring, None)
    .pow(polynomial.power as usize);
    if polynomial.constant != 0 {
        result = result.add_constant(ring.nth(Integer::from(polynomial.constant)));
    }
    result
}

pub fn powered_pair<R>(
    ring: &R,
    left: PoweredPolynomial,
    right: PoweredPolynomial,
) -> [MultivariatePolynomial<R, u16>; 2]
where
    R: EuclideanDomain + ConvertToRing,
{
    let mut polynomials = [
        powered_polynomial(ring, left),
        powered_polynomial(ring, right),
    ];
    MultivariatePolynomial::unify_variables_list(&mut polynomials);
    polynomials
}

fn dense_univariate<R>(ring: &R, coefficients: &[u64]) -> MultivariatePolynomial<R, u16>
where
    R: EuclideanDomain + ConvertToRing,
{
    let template = Atom::parse("x", BENCHMARK_NAMESPACE, ParseSettings::default())
        .unwrap()
        .to_polynomial::<_, u16>(ring, None);
    let mut polynomial =
        MultivariatePolynomial::new(ring, Some(coefficients.len()), template.variables().clone());
    for (exponent, &coefficient) in coefficients.iter().enumerate() {
        let coefficient = ring.nth(Integer::from(coefficient));
        if !ring.is_zero(&coefficient) {
            polynomial.append_monomial_back(coefficient, &[exponent as u16]);
        }
    }
    polynomial
}

pub fn finite_pair<R>(
    ring: &R,
    case: FiniteFieldMultiplicationCase,
) -> [MultivariatePolynomial<R, u16>; 2]
where
    R: EuclideanDomain + ConvertToRing,
{
    match case.input {
        FiniteFieldMultiplicationInput::DenseUnivariate { .. } => {
            let (left, right) = case.input.dense_univariate_coefficients().unwrap();
            [
                dense_univariate(ring, &left),
                dense_univariate(ring, &right),
            ]
        }
        FiniteFieldMultiplicationInput::Powered { left, right } => powered_pair(ring, left, right),
    }
}

pub fn resultant_inputs(
    case: ResultantCase,
) -> (IntegerUnivariatePolynomial, IntegerUnivariatePolynomial) {
    let mut polynomials = [
        parse_integer_polynomial(case.left),
        parse_integer_polynomial(case.right),
    ];
    MultivariatePolynomial::unify_variables_list(&mut polynomials);
    let variable = polynomials[0]
        .variables()
        .iter()
        .position(|variable| variable == &PolyVariable::Symbol(symbol!("x")))
        .expect("resultant cases must contain x");
    (
        polynomials[0].to_univariate(variable),
        polynomials[1].to_univariate(variable),
    )
}

pub fn benchmark_integer_multiplication(
    bencher: divan::Bencher<'_, '_>,
    case: IntegerMultiplicationCase,
) {
    let [left, right] = powered_pair(&Z, case.left, case.right);
    bencher.bench_local(|| &left * &right);
}

pub fn benchmark_exact_division(bencher: divan::Bencher<'_, '_>, case: ExactDivisionCase) {
    let [quotient, divisor] = powered_pair(&Z, case.quotient, case.divisor);
    let dividend = &quotient * &divisor;
    assert_eq!(dividend.clone().try_div_owned(&divisor).unwrap(), quotient);
    drop(quotient);
    bencher
        .with_inputs(|| dividend.clone())
        .bench_local_values(|owned_dividend| owned_dividend.try_div_owned(&divisor).unwrap());
}

pub fn benchmark_finite_multiplication<R>(
    bencher: divan::Bencher<'_, '_>,
    ring: &R,
    case: FiniteFieldMultiplicationCase,
) where
    R: EuclideanDomain + ConvertToRing,
{
    let [left, right] = finite_pair(ring, case);
    bencher.bench_local(|| &left * &right);
}

pub fn benchmark_resultant_default(bencher: divan::Bencher<'_, '_>, case: ResultantCase) {
    let (left, right) = resultant_inputs(case);
    bencher.bench_local(|| left.resultant(&right));
}

pub fn benchmark_resultant_brown(bencher: divan::Bencher<'_, '_>, case: ResultantCase) {
    let (left, right) = resultant_inputs(case);
    let expected = left.resultant(&right);
    assert_eq!(left.resultant_brown(&right), expected);
    drop(expected);
    bencher.bench_local(|| left.resultant_brown(&right));
}

pub fn benchmark_resultant_crt(bencher: divan::Bencher<'_, '_>, case: ResultantCase) {
    let (left, right) = resultant_inputs(case);
    let expected = left.resultant(&right);
    assert_eq!(left.resultant_crt(&right), expected);
    drop(expected);
    bencher.bench_local(|| left.resultant_crt(&right));
}

#[derive(Clone, Copy)]
pub enum GcdAlgorithm {
    Auto,
    Hu,
    Zippel,
}

pub fn configure_gcd(algorithm: GcdAlgorithm) {
    let (use_hu, force_hu) = match algorithm {
        GcdAlgorithm::Auto => (true, false),
        GcdAlgorithm::Hu => (true, true),
        GcdAlgorithm::Zippel => (false, false),
    };
    GLOBAL_SETTINGS
        .use_hu_monagan_poly_gcd
        .store(use_hu, Ordering::Relaxed);
    GLOBAL_SETTINGS
        .force_hu_monagan_poly_gcd
        .store(force_hu, Ordering::Relaxed);
}

fn parse_env<T>(name: &str, default: T) -> T
where
    T: FromStr,
    T::Err: std::fmt::Display,
{
    env::var(name)
        .map(|value| {
            value
                .parse()
                .unwrap_or_else(|error| panic!("invalid {name}: {error}"))
        })
        .unwrap_or(default)
}

pub fn gcd_case_config() -> GcdCaseConfig {
    static CONFIG: OnceLock<GcdCaseConfig> = OnceLock::new();
    *CONFIG.get_or_init(|| {
        let config = GcdCaseConfig {
            kind: parse_env("GCD_BENCH_CASE", GcdCaseKind::Dense),
            variable_count: parse_env("GCD_BENCH_NVARS", 7),
            degree: parse_env("GCD_BENCH_DEGREE", 7),
            gap: parse_env("GCD_BENCH_GAP", 10),
            coefficient_bits: parse_env("GCD_BENCH_COEFFICIENT_BITS", 30),
        };
        config.validate().unwrap();
        eprintln!("GCD case: {config}");
        config
    })
}

pub fn gcd_factors_for(config: GcdCaseConfig) -> [IntegerPolynomial; 3] {
    let generated = config.generate().unwrap();
    let mut polynomials = [
        parse_integer_polynomial(&generated.left_cofactor),
        parse_integer_polynomial(&generated.right_cofactor),
        parse_integer_polynomial(&generated.gcd),
    ];
    MultivariatePolynomial::unify_variables_list(&mut polynomials);
    polynomials
}

pub fn gcd_inputs_for(
    config: GcdCaseConfig,
) -> (IntegerPolynomial, IntegerPolynomial, IntegerPolynomial) {
    let [left_cofactor, right_cofactor, gcd] = gcd_factors_for(config);
    let left = &left_cofactor * &gcd;
    let right = &right_cofactor * &gcd;
    (left, right, gcd)
}

pub fn benchmark_gcd_products_for(bencher: divan::Bencher<'_, '_>, config: GcdCaseConfig) {
    let [left, right, gcd] = gcd_factors_for(config);
    bencher.bench_local(|| (&left * &gcd, &right * &gcd));
}

pub fn benchmark_gcd_products(bencher: divan::Bencher<'_, '_>) {
    benchmark_gcd_products_for(bencher, gcd_case_config());
}

pub fn benchmark_gcd_for(
    bencher: divan::Bencher<'_, '_>,
    algorithm: GcdAlgorithm,
    config: GcdCaseConfig,
) {
    configure_gcd(algorithm);
    let (left, right, expected) = gcd_inputs_for(config);
    let actual = left.gcd(&right);
    assert!((&actual - &expected).is_zero());
    drop((actual, expected));
    bencher.bench_local(|| left.gcd(&right));
}

pub fn benchmark_gcd(bencher: divan::Bencher<'_, '_>, algorithm: GcdAlgorithm) {
    benchmark_gcd_for(bencher, algorithm, gcd_case_config());
}
