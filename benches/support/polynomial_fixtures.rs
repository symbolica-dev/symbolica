//! Polynomial fixtures preserving variable maps and u32 exponents.
use std::{fmt, fs::File, io::Read, path::PathBuf, str::FromStr, sync::Arc};
use symbolica::domains::rational::RationalField;
use symbolica::prelude::*;

pub type IntegerPoly = MultivariatePolynomial<IntegerRing, u32>;
pub type RationalPoly = MultivariatePolynomial<RationalField, u32>;

#[derive(Clone, Copy)]
pub struct EliminationCase {
    pub id: &'static str,
    pub discriminant: bool,
}

impl fmt::Display for EliminationCase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id)
    }
}

pub struct EliminationInput {
    pub polynomials: Vec<IntegerPoly>,
    pub names: Vec<String>,
    pub variable: usize,
}

struct FixtureInput {
    polynomials: Vec<IntegerPoly>,
    names: Vec<String>,
    variable: Option<usize>,
}

fn load_fixture(id: &str, operation: &str) -> FixtureInput {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("benches/fixtures/polynomial")
        .join(format!("{id}.op.br"));
    let mut text = String::new();
    brotli::Decompressor::new(File::open(path).unwrap(), 65536)
        .read_to_string(&mut text)
        .unwrap();
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some("polynomial-bench-op 1"));
    assert_eq!(
        lines.next().unwrap().strip_prefix("operation "),
        Some(operation)
    );
    let names: Vec<String> = lines
        .next()
        .unwrap()
        .strip_prefix("variables ")
        .unwrap()
        .split_whitespace()
        .map(str::to_owned)
        .collect();
    let variable = match lines.next().unwrap().strip_prefix("variable ").unwrap() {
        "-1" => None,
        index => Some(index.parse::<usize>().unwrap()),
    };
    assert_eq!(variable.is_none(), operation == "factor");
    assert!(variable.is_none_or(|v| v < names.len()));
    let count: usize = lines
        .next()
        .unwrap()
        .strip_prefix("polynomials ")
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(count, if operation == "resultant" { 2 } else { 1 });
    let variables: Arc<Vec<PolyVariable>> = Arc::new(
        names
            .iter()
            .map(|n| PolyVariable::Symbol(symbol!(n.as_str())))
            .collect(),
    );
    let polynomials = (0..count)
        .map(|_| {
            let n = lines
                .next()
                .unwrap()
                .strip_prefix("terms ")
                .unwrap()
                .parse()
                .unwrap();
            let mut p = IntegerPoly::new(&Z, Some(n), variables.clone());
            let mut terms = Vec::with_capacity(n);
            for _ in 0..n {
                let mut fields = lines.next().unwrap().split_whitespace();
                let coefficient = Integer::from_str(fields.next().unwrap()).unwrap();
                let exponents: Vec<u32> = fields.map(|e| e.parse().unwrap()).collect();
                assert_eq!(exponents.len(), names.len());
                terms.push((exponents, coefficient));
            }
            terms.sort_by(|a, b| a.0.cmp(&b.0));
            for (e, c) in terms {
                p.append_monomial(c, &e);
            }
            assert_eq!(p.nterms(), n);
            p
        })
        .collect();
    assert!(lines.next().is_none());
    FixtureInput {
        polynomials,
        names,
        variable,
    }
}

pub fn load_elimination(case: EliminationCase) -> EliminationInput {
    let input = load_fixture(
        case.id,
        if case.discriminant {
            "discriminant"
        } else {
            "resultant"
        },
    );
    EliminationInput {
        polynomials: input.polynomials,
        names: input.names,
        variable: input.variable.unwrap(),
    }
}

impl EliminationInput {
    /// Includes conversion and derivative construction, as for the multivariate
    /// FLINT API. Both sides start with already parsed integer polynomials.
    pub fn resultant_or_discriminant(&self) -> IntegerPoly {
        let left = self.polynomials[0].to_univariate(self.variable);
        if self.polynomials.len() == 2 {
            left.resultant(&self.polynomials[1].to_univariate(self.variable))
        } else {
            let mut result = left.resultant(&left.derivative()) / &left.lcoeff();
            if left.degree() % 4 >= 2 {
                result = -result;
            }
            result
        }
    }
}

#[derive(Clone, Copy)]
pub struct SegmentedCase {
    pub id: &'static str,
    pub parent: &'static str,
    pub weights: &'static [usize],
    pub digit_bits: usize,
}

impl fmt::Display for SegmentedCase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id)
    }
}

/// Reconstruct the published univariate operand before timing. The importer
/// verifies its exact serialization against the upstream SHA-256.
pub fn load_resultant(
    case: SegmentedCase,
) -> [symbolica::poly::univariate::UnivariatePolynomial<IntegerRing>; 2] {
    use symbolica::poly::univariate::UnivariatePolynomial;
    let input = load_elimination(EliminationCase {
        id: case.parent,
        discriminant: false,
    });
    assert_eq!(input.names.len(), case.weights.len());
    std::array::from_fn(|i| {
        let mut coefficients = Vec::new();
        for term in &input.polynomials[i] {
            let shift = term
                .exponents
                .iter()
                .zip(case.weights)
                .map(|(e, w)| *e as usize * w)
                .sum::<usize>()
                * case.digit_bits;
            let degree = term.exponents[input.variable] as usize;
            if coefficients.len() <= degree {
                coefficients.resize(degree + 1, Integer::zero());
            }
            // Shift at arbitrary precision, including for small input coefficients.
            coefficients[degree] +=
                Integer::from(term.coefficient.clone().to_multi_prec() << shift);
        }
        UnivariatePolynomial::from_coefficients(
            &Z,
            coefficients,
            Arc::new(PolyVariable::Symbol(symbol!("X"))),
        )
    })
}

#[derive(Clone, Copy)]
pub struct Case {
    pub id: &'static str,
    pub terms: usize,
    pub gcd_variable: Option<usize>,
}

impl fmt::Display for Case {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.id)
    }
}

include!("../fixtures/polynomial/cases.rs");

pub struct Input {
    pub names: Vec<String>,
    pub integer: IntegerPoly,
    pub rational: RationalPoly,
}

pub fn load(case: Case) -> Input {
    let mut input = load_fixture(case.id, "factor");
    let integer = input.polynomials.pop().unwrap();
    assert_eq!(integer.nterms(), case.terms);
    let rational = integer.map_coeff(|c| Rational::from(c), Q);
    Input {
        names: input.names,
        integer,
        rational,
    }
}

/// FLINT construction and formatting are outside all timed closures.
#[cfg(feature = "flint_benchmarks")]
pub fn expression(poly: &IntegerPoly, names: &[String]) -> String {
    use fmt::Write;
    let mut text = String::new();
    for (i, t) in poly.into_iter().enumerate() {
        if i > 0 {
            text.push('+');
        }
        write!(&mut text, "({})", t.coefficient).unwrap();
        for (name, e) in names.iter().zip(t.exponents) {
            if *e != 0 {
                write!(&mut text, "*{name}^{e}").unwrap();
            }
        }
    }
    if text.is_empty() {
        text.push('0');
    }
    text
}

#[cfg(feature = "flint_benchmarks")]
pub fn parse_flint(poly: &super::flint::FmpzMPoly<'_>, input: &Input) -> RationalPoly {
    let names: Vec<smartstring::alias::String> =
        input.names.iter().map(|s| s.as_str().into()).collect();
    let text = poly.to_pretty_string().unwrap();
    let (rest, p) = Token::parse_polynomial(text.as_bytes(), input.integer.variables(), &names, &Q);
    assert!(rest.is_empty());
    p
}

pub fn validate_factorization(input: &RationalPoly, factors: &[(RationalPoly, usize)]) {
    let reconstructed = factors.iter().fold(input.one(), |p, (f, m)| p * &f.pow(*m));
    assert_eq!(&reconstructed, input);
}

#[cfg(feature = "flint_benchmarks")]
pub fn validate_flint_factors(
    input: &Input,
    factors: &[(RationalPoly, usize)],
    flint: &mut super::flint::FmpzMPolyFactor<'_>,
) {
    validate_factorization(&input.rational, factors);
    assert_eq!(parse_flint(&flint.expand().unwrap(), input), input.rational);
    let mut expected: Vec<_> = factors
        .iter()
        .filter(|(f, _)| !f.is_constant())
        .map(|(f, m)| (f.clone().make_monic(), *m))
        .collect();
    assert_eq!(expected.len(), flint.len());
    for (f, m) in flint.factors() {
        let f = parse_flint(&f, input).make_monic();
        let i = expected
            .iter()
            .position(|(g, n)| *n == m && *g == f)
            .expect("factorizations differ");
        expected.swap_remove(i);
    }
    assert!(expected.is_empty());
}
