//! Inputs shared by the Symbolica and FLINT polynomial benchmarks.

#![allow(dead_code)]

use std::{fmt, fmt::Write};

pub const X_VARIABLES: [&str; 1] = ["x"];
pub const X1_VARIABLES: [&str; 1] = ["x1"];
pub const X1_TO_X2_VARIABLES: [&str; 2] = ["x1", "x2"];
pub const X1_TO_X3_VARIABLES: [&str; 3] = ["x1", "x2", "x3"];
pub const XYZ_VARIABLES: [&str; 3] = ["x", "y", "z"];
pub const XY1Y2_VARIABLES: [&str; 3] = ["x", "y1", "y2"];
pub const X1_TO_X5_VARIABLES: [&str; 5] = ["x1", "x2", "x3", "x4", "x5"];
pub const X1_TO_X7_VARIABLES: [&str; 7] = ["x1", "x2", "x3", "x4", "x5", "x6", "x7"];
pub const X1_TO_X8_VARIABLES: [&str; 8] = ["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8"];

/// Constructs a polynomial by raising `base` to `power` and then adding `constant`.
#[derive(Clone, Copy, Debug)]
pub struct PoweredPolynomial {
    pub base: &'static str,
    pub power: u32,
    pub constant: i32,
}

impl PoweredPolynomial {
    pub const fn new(base: &'static str, power: u32) -> Self {
        Self {
            base,
            power,
            constant: 0,
        }
    }

    pub const fn with_constant(base: &'static str, power: u32, constant: i32) -> Self {
        Self {
            base,
            power,
            constant,
        }
    }
}

/// An integer-polynomial multiplication input and its normal sample count.
#[derive(Clone, Copy, Debug)]
pub struct IntegerMultiplicationCase {
    pub name: &'static str,
    pub variables: &'static [&'static str],
    pub left: PoweredPolynomial,
    pub right: PoweredPolynomial,
    pub default_samples: usize,
}

impl fmt::Display for IntegerMultiplicationCase {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name)
    }
}

pub const INTEGER_MULTIPLICATION_CASES: [IntegerMultiplicationCase; 8] = [
    IntegerMultiplicationCase {
        name: "dense small multiplication",
        variables: &XYZ_VARIABLES,
        left: PoweredPolynomial::new("1+x+y+z", 12),
        right: PoweredPolynomial::new("1+2*x-y+3*z", 11),
        default_samples: 25,
    },
    IntegerMultiplicationCase {
        name: "dense high multiplication",
        variables: &XYZ_VARIABLES,
        left: PoweredPolynomial::new("1000000000039+x+y+z", 12),
        right: PoweredPolynomial::new("1000000000187+2*x-y+3*z", 11),
        default_samples: 10,
    },
    IntegerMultiplicationCase {
        name: "dense large multiplication",
        variables: &XYZ_VARIABLES,
        left: PoweredPolynomial::new("1+x+y+z", 24),
        right: PoweredPolynomial::new("1+2*x-y+3*z", 23),
        default_samples: 7,
    },
    IntegerMultiplicationCase {
        name: "dense very large multiplication",
        variables: &XYZ_VARIABLES,
        left: PoweredPolynomial::new("1+x+y+z", 40),
        right: PoweredPolynomial::new("1+2*x-y+3*z", 39),
        default_samples: 3,
    },
    IntegerMultiplicationCase {
        name: "dense high large multiplication",
        variables: &XYZ_VARIABLES,
        left: PoweredPolynomial::new("1000000000039+x+y+z", 20),
        right: PoweredPolynomial::new("1000000000187+2*x-y+3*z", 19),
        default_samples: 3,
    },
    IntegerMultiplicationCase {
        name: "sparse separated multiplication",
        variables: &XYZ_VARIABLES,
        left: PoweredPolynomial::new(
            "1+2*x^37*y^11+3*x^5*y^43*z^7+5*x^61*z^29+7*x^17*y^73*z^31+11*x^89*y^19*z^47",
            7,
        ),
        right: PoweredPolynomial::new(
            "1+2*x^37*y^11+3*x^5*y^43*z^7+5*x^61*z^29+7*x^17*y^73*z^31+11*x^89*y^19*z^47",
            7,
        ),
        default_samples: 3,
    },
    IntegerMultiplicationCase {
        name: "sparse large multiplication",
        variables: &XYZ_VARIABLES,
        left: PoweredPolynomial::new(
            "1+2*x^37*y^11+3*x^5*y^43*z^7+5*x^61*z^29+7*x^17*y^73*z^31+11*x^89*y^19*z^47+13*x^23*y^97*z^59+17*x^107*y^53*z^83",
            7,
        ),
        right: PoweredPolynomial::new(
            "1+2*x^37*y^11+3*x^5*y^43*z^7+5*x^61*z^29+7*x^17*y^73*z^31+11*x^89*y^19*z^47+13*x^23*y^97*z^59+17*x^107*y^53*z^83",
            7,
        ),
        default_samples: 1,
    },
    IntegerMultiplicationCase {
        name: "seven-variable power-minus-one multiplication",
        variables: &X1_TO_X7_VARIABLES,
        left: PoweredPolynomial::with_constant("1+3*x1+5*x2+7*x3+9*x4+11*x5+13*x6+15*x7", 7, -1),
        right: PoweredPolynomial::with_constant("1+3*x1+5*x2+7*x3+9*x4+11*x5+13*x6+15*x7", 7, -1),
        default_samples: 1,
    },
];

/// An exact-division input, where `dividend = quotient * divisor`.
#[derive(Clone, Copy, Debug)]
pub struct ExactDivisionCase {
    pub name: &'static str,
    pub variables: &'static [&'static str],
    pub quotient: PoweredPolynomial,
    pub divisor: PoweredPolynomial,
    pub default_samples: usize,
}

impl fmt::Display for ExactDivisionCase {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name)
    }
}

pub const EXACT_DIVISION_CASES: [ExactDivisionCase; 3] = [
    ExactDivisionCase {
        name: "dense exact division",
        variables: &XYZ_VARIABLES,
        quotient: PoweredPolynomial::new("1+x+y+z", 12),
        divisor: PoweredPolynomial::new("1+2*x-y+3*z", 7),
        default_samples: 7,
    },
    ExactDivisionCase {
        name: "dense large exact division",
        variables: &XYZ_VARIABLES,
        quotient: PoweredPolynomial::new("1+x+y+z", 20),
        divisor: PoweredPolynomial::new("1+2*x-y+3*z", 12),
        default_samples: 5,
    },
    ExactDivisionCase {
        name: "high-height exact division",
        variables: &XYZ_VARIABLES,
        quotient: PoweredPolynomial::new("1000000000039+x+y+z", 12),
        divisor: PoweredPolynomial::new("1000000000187+2*x-y+3*z", 10),
        default_samples: 5,
    },
];

/// A reducible integer polynomial constructed as the product of two powered polynomials.
#[derive(Clone, Copy, Debug)]
pub struct FactorizationCase {
    pub name: &'static str,
    pub variables: &'static [&'static str],
    pub left: PoweredPolynomial,
    pub right: PoweredPolynomial,
    pub default_samples: usize,
}

impl fmt::Display for FactorizationCase {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name)
    }
}

/// Low-dimensional factorization cases with comparable dense powered inputs.
pub const GENERATED_FACTOR_CASES: [FactorizationCase; 6] = [
    FactorizationCase {
        name: "dense 1-variable degrees 32/31",
        variables: &X1_VARIABLES,
        left: PoweredPolynomial::with_constant("1+3*x1", 32, -1),
        right: PoweredPolynomial::with_constant("1-5*x1", 31, 1),
        default_samples: 4,
    },
    FactorizationCase {
        name: "dense 2-variable degrees 10/9",
        variables: &X1_TO_X2_VARIABLES,
        left: PoweredPolynomial::with_constant("1+3*x1+5*x2", 10, -1),
        right: PoweredPolynomial::with_constant("1-3*x1+5*x2", 9, 1),
        default_samples: 4,
    },
    FactorizationCase {
        name: "dense 3-variable degrees 6/5",
        variables: &X1_TO_X3_VARIABLES,
        left: PoweredPolynomial::with_constant("1+3*x1+5*x2+7*x3", 6, -1),
        right: PoweredPolynomial::with_constant("1-3*x1+5*x2-7*x3", 5, 1),
        default_samples: 4,
    },
    FactorizationCase {
        name: "dense high-height 1-variable degrees 17/16 total 33",
        variables: &X1_VARIABLES,
        left: PoweredPolynomial::with_constant("1+65537*x1", 17, -1),
        right: PoweredPolynomial::with_constant("1-65539*x1", 16, 1),
        default_samples: 2,
    },
    FactorizationCase {
        name: "dense 1-variable degrees 33/31 total 64",
        variables: &X1_VARIABLES,
        left: PoweredPolynomial::with_constant("1+3*x1", 33, -1),
        right: PoweredPolynomial::with_constant("1-5*x1", 31, 1),
        default_samples: 2,
    },
    FactorizationCase {
        name: "dense 1-variable degrees 33/32 total 65",
        variables: &X1_VARIABLES,
        left: PoweredPolynomial::with_constant("1+3*x1", 33, -1),
        right: PoweredPolynomial::with_constant("1-5*x1", 32, 1),
        default_samples: 2,
    },
];

/// A prime field used by every finite-field multiplication input.
#[derive(Clone, Copy, Debug)]
pub struct FiniteFieldCase {
    pub name: &'static str,
    pub modulus: u64,
}

impl fmt::Display for FiniteFieldCase {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name)
    }
}

pub const FINITE_FIELDS: [FiniteFieldCase; 2] = [
    FiniteFieldCase {
        name: "GF(17)",
        modulus: 17,
    },
    FiniteFieldCase {
        name: "GF(18446744073709551557)",
        modulus: 18_446_744_073_709_551_557,
    },
];

/// Prime fields that exercise the 32-bit finite-field accumulator boundaries.
pub const U32_ACCUMULATION_FIELDS: [FiniteFieldCase; 2] = [
    FiniteFieldCase {
        name: "GF(65000011)",
        modulus: 65_000_011,
    },
    FiniteFieldCase {
        name: "GF(500000003)",
        modulus: 500_000_003,
    },
];

/// The operands used by a finite-field multiplication case.
#[derive(Clone, Copy, Debug)]
pub enum FiniteFieldMultiplicationInput {
    /// Generates coefficient `(stride * exponent) % period + 1` at every exponent.
    DenseUnivariate {
        left_degree: u32,
        right_degree: u32,
        left_stride: u64,
        right_stride: u64,
        coefficient_period: u64,
    },
    Powered {
        left: PoweredPolynomial,
        right: PoweredPolynomial,
    },
}

impl FiniteFieldMultiplicationInput {
    /// Returns the dense coefficient vectors for a generated univariate input.
    pub fn dense_univariate_coefficients(self) -> Option<(Vec<u64>, Vec<u64>)> {
        let Self::DenseUnivariate {
            left_degree,
            right_degree,
            left_stride,
            right_stride,
            coefficient_period,
        } = self
        else {
            return None;
        };

        assert!(
            coefficient_period > 0,
            "coefficient period must be positive"
        );
        let coefficients = |degree: u32, stride: u64| {
            (0..=degree)
                .map(|exponent| {
                    (u128::from(stride) * u128::from(exponent) % u128::from(coefficient_period))
                        as u64
                        + 1
                })
                .collect()
        };
        Some((
            coefficients(left_degree, left_stride),
            coefficients(right_degree, right_stride),
        ))
    }
}

/// A finite-field multiplication input applied to every field in [`FINITE_FIELDS`].
#[derive(Clone, Copy, Debug)]
pub struct FiniteFieldMultiplicationCase {
    pub name: &'static str,
    pub variables: &'static [&'static str],
    pub input: FiniteFieldMultiplicationInput,
    pub default_samples: usize,
}

impl FiniteFieldMultiplicationCase {
    pub fn display_name(self, field: FiniteFieldCase) -> String {
        format!("{} {}", field.name, self.name)
    }
}

impl fmt::Display for FiniteFieldMultiplicationCase {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name)
    }
}

pub const FINITE_FIELD_MULTIPLICATION_CASES: [FiniteFieldMultiplicationCase; 6] = [
    FiniteFieldMultiplicationCase {
        name: "dense univariate degree-4912 multiplication",
        variables: &X_VARIABLES,
        input: FiniteFieldMultiplicationInput::DenseUnivariate {
            left_degree: 4912,
            right_degree: 4911,
            left_stride: 1,
            right_stride: 7,
            coefficient_period: 16,
        },
        default_samples: 3,
    },
    FiniteFieldMultiplicationCase {
        name: "dense large multiplication",
        variables: &XYZ_VARIABLES,
        input: FiniteFieldMultiplicationInput::Powered {
            left: PoweredPolynomial::new("1+x+y+z", 24),
            right: PoweredPolynomial::new("1+2*x-y+3*z", 23),
        },
        default_samples: 5,
    },
    FiniteFieldMultiplicationCase {
        name: "dense very large multiplication",
        variables: &XYZ_VARIABLES,
        input: FiniteFieldMultiplicationInput::Powered {
            left: PoweredPolynomial::new("1+x+y+z", 40),
            right: PoweredPolynomial::new("1+2*x-y+3*z", 39),
        },
        default_samples: 3,
    },
    FiniteFieldMultiplicationCase {
        name: "five-variable total-degree multiplication",
        variables: &X1_TO_X5_VARIABLES,
        input: FiniteFieldMultiplicationInput::Powered {
            left: PoweredPolynomial::with_constant("1+x1+2*x2+3*x3+4*x4+5*x5", 13, -1),
            right: PoweredPolynomial::with_constant("1+2*x1-3*x2+5*x3-7*x4+11*x5", 12, -1),
        },
        default_samples: 3,
    },
    FiniteFieldMultiplicationCase {
        name: "sparse large multiplication",
        variables: &XYZ_VARIABLES,
        input: FiniteFieldMultiplicationInput::Powered {
            left: PoweredPolynomial::new(
                "1+2*x^37*y^11+3*x^5*y^43*z^7+5*x^61*z^29+7*x^17*y^73*z^31+11*x^89*y^19*z^47+13*x^23*y^97*z^59+17*x^107*y^53*z^83",
                7,
            ),
            right: PoweredPolynomial::new(
                "1+2*x^37*y^11+3*x^5*y^43*z^7+5*x^61*z^29+7*x^17*y^73*z^31+11*x^89*y^19*z^47+13*x^23*y^97*z^59+17*x^107*y^53*z^83",
                7,
            ),
        },
        default_samples: 1,
    },
    FiniteFieldMultiplicationCase {
        name: "seven-variable power-minus-one multiplication",
        variables: &X1_TO_X7_VARIABLES,
        input: FiniteFieldMultiplicationInput::Powered {
            left: PoweredPolynomial::with_constant(
                "1+3*x1+5*x2+7*x3+9*x4+11*x5+13*x6+15*x7",
                7,
                -1,
            ),
            right: PoweredPolynomial::with_constant(
                "1+3*x1+5*x2+7*x3+9*x4+11*x5+13*x6+15*x7",
                7,
                -1,
            ),
        },
        default_samples: 1,
    },
];

/// An unbalanced convolution whose worst-case sum over all product pairs exceeds `u64` while the
/// worst-case sum contributing to any one output coefficient still fits.
pub const U32_ACCUMULATION_MULTIPLICATION_CASE: FiniteFieldMultiplicationCase =
    FiniteFieldMultiplicationCase {
        name: "dense univariate degrees 128/64 accumulator-bound multiplication",
        variables: &X_VARIABLES,
        input: FiniteFieldMultiplicationInput::DenseUnivariate {
            left_degree: 128,
            right_degree: 64,
            left_stride: 499_999_937,
            right_stride: 271_828_183,
            coefficient_period: 500_000_002,
        },
        default_samples: 200,
    };

/// Two integer polynomials and the variable eliminated by the resultant.
#[derive(Clone, Copy, Debug)]
pub struct ResultantCase {
    pub name: &'static str,
    pub variables: &'static [&'static str],
    pub left: &'static str,
    pub right: &'static str,
    pub elimination_variable: usize,
    pub default_samples: usize,
}

impl fmt::Display for ResultantCase {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name)
    }
}

pub const RESULTANT_CASES: [ResultantCase; 6] = [
    ResultantCase {
        name: "dense outer degrees 7/6",
        variables: &XYZ_VARIABLES,
        left: "1+(2+y^2+z^3)*x+(3+y^3+z^2)*x^2+(4+y+z)*x^3+(5+y^2+z^3)*x^4+(6+y^3+z^2)*x^5+(7+y+z)*x^6+(8+y^2+z^3)*x^7",
        right: "1+(3+y^3-z^2)*x+(5+y^2-z^3)*x^2+(7+y-z)*x^3+(9+y^3-z^2)*x^4+(11+y^2-z^3)*x^5+(13+y-z)*x^6",
        elimination_variable: 0,
        default_samples: 7,
    },
    ResultantCase {
        name: "lacunary outer degrees 18/11",
        variables: &XYZ_VARIABLES,
        left: "(y+1)*x^18+(z+2)*x^13+(y*z+3)*x^7+(y^2-z)*x^2+1",
        right: "(z+1)*x^11+(y-2)*x^8+(y+z)*x^3+2",
        elimination_variable: 0,
        default_samples: 7,
    },
    ResultantCase {
        name: "nonunit leading degrees 9/7",
        variables: &XYZ_VARIABLES,
        left: "(y+1)*x^9+(z^2+2)*x^8+(y*z+1)*x^5+(y^2+z)*x^2+3",
        right: "(z+1)*x^7+(y^2-1)*x^6+(y+z+1)*x^3+z*x+2",
        elimination_variable: 0,
        default_samples: 7,
    },
    ResultantCase {
        name: "large high-height degrees 14/10",
        variables: &XYZ_VARIABLES,
        left: "(1000000000039+y^3+z^2)*x^14+(1000000000061+y*z^2-z^3)*x^10+(1000000000091+y*z+y)*x^6+(1000000000163+y^2*z^2+z)*x^2+1000000000169+y+z",
        right: "(1000000000187+z^2+y)*x^10+(1000000000193+y^3-z^2)*x^7+(1000000000223+y^2*z-z^3)*x^4+(1000000000241+y^2+z^2)*x+1000000000271-z",
        elimination_variable: 0,
        default_samples: 3,
    },
    ResultantCase {
        name: "dense outer degrees 10/8 CRT crossover",
        variables: &XY1Y2_VARIABLES,
        left: "(129+9*y1^2+16*y2+2*y1*y2)+(135+14*y1-4*y2^2+3*y1*y2)*x+(134-2*y1^2+9*y2+4*y1*y2)*x^2+(133+7*y1+14*y2^2+5*y1*y2)*x^3+(132+12*y1^2-2*y2+6*y1*y2)*x^4+(131-17*y1+7*y2^2+7*y1*y2)*x^5+(130+5*y1^2+12*y2+8*y1*y2)*x^6+(129+10*y1-17*y2^2+2*y1*y2)*x^7+(135-15*y1^2+5*y2+3*y1*y2)*x^8+(134+3*y1+10*y2^2+4*y1*y2)*x^9+(133+8*y1^2-15*y2+5*y1*y2)*x^10",
        right: "(132+12*y1-2*y2^2+3*y1*y2)+(131-17*y1^2+7*y2+4*y1*y2)*x+(130+5*y1+12*y2^2+5*y1*y2)*x^2+(129+10*y1^2-17*y2+6*y1*y2)*x^3+(135-15*y1+5*y2^2+7*y1*y2)*x^4+(134+3*y1^2+10*y2+8*y1*y2)*x^5+(133+8*y1-15*y2^2+2*y1*y2)*x^6+(132-13*y1^2+3*y2+3*y1*y2)*x^7+(131+18*y1+8*y2^2+4*y1*y2)*x^8",
        elimination_variable: 0,
        default_samples: 3,
    },
    ResultantCase {
        name: "outer-sparse degrees 12/9 CRT crossover",
        variables: &XY1Y2_VARIABLES,
        left: "(129+9*y1^2+16*y2+2*y1*y2)+(135+14*y1-4*y2^2+3*y1*y2)*x+(130+5*y1^2+12*y2+8*y1*y2)*x^6+(131+18*y1^2+8*y2+7*y1*y2)*x^12",
        right: "(132+12*y1-2*y2^2+3*y1*y2)+(131-17*y1^2+7*y2+4*y1*y2)*x+(135-15*y1+5*y2^2+7*y1*y2)*x^4+(130+6*y1^2-13*y2+5*y1*y2)*x^9",
        elimination_variable: 0,
        default_samples: 5,
    },
];

/// The support and coefficient pattern used to construct a multivariate GCD input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GcdCaseKind {
    Dense,
    Sparse,
    HighGap,
    HighHeight,
}

impl GcdCaseKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Dense => "dense",
            Self::Sparse => "sparse",
            Self::HighGap => "high-gap",
            Self::HighHeight => "high-height",
        }
    }
}

impl fmt::Display for GcdCaseKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl std::str::FromStr for GcdCaseKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "dense" => Ok(Self::Dense),
            "sparse" => Ok(Self::Sparse),
            "high-gap" => Ok(Self::HighGap),
            "high-height" => Ok(Self::HighHeight),
            _ => Err(format!(
                "unknown GCD case {value:?}; expected dense, sparse, high-gap, or high-height"
            )),
        }
    }
}

/// Parameters for constructing cofactors `a`, `b`, and their expected common factor `g`.
#[derive(Clone, Copy, Debug)]
pub struct GcdCaseConfig {
    pub kind: GcdCaseKind,
    pub variable_count: usize,
    pub degree: u32,
    pub gap: u32,
    pub coefficient_bits: u32,
}

impl Default for GcdCaseConfig {
    fn default() -> Self {
        Self {
            kind: GcdCaseKind::Dense,
            variable_count: 7,
            degree: 7,
            gap: 10,
            coefficient_bits: 30,
        }
    }
}

/// Fixed generated cases that exercise support shape, dimension, exponent span,
/// and coefficient height independently of the imported polybench fixtures.
pub const GENERATED_GCD_CASES: [GcdCaseConfig; 14] = [
    GcdCaseConfig {
        kind: GcdCaseKind::Dense,
        variable_count: 1,
        degree: 32,
        gap: 10,
        coefficient_bits: 30,
    },
    GcdCaseConfig {
        kind: GcdCaseKind::Dense,
        variable_count: 2,
        degree: 5,
        gap: 10,
        coefficient_bits: 30,
    },
    GcdCaseConfig {
        kind: GcdCaseKind::Dense,
        variable_count: 3,
        degree: 7,
        gap: 10,
        coefficient_bits: 30,
    },
    GcdCaseConfig {
        kind: GcdCaseKind::Dense,
        variable_count: 5,
        degree: 7,
        gap: 10,
        coefficient_bits: 30,
    },
    GcdCaseConfig {
        kind: GcdCaseKind::Dense,
        variable_count: 8,
        degree: 5,
        gap: 10,
        coefficient_bits: 30,
    },
    GcdCaseConfig {
        kind: GcdCaseKind::Sparse,
        variable_count: 5,
        degree: 7,
        gap: 10,
        coefficient_bits: 30,
    },
    GcdCaseConfig {
        kind: GcdCaseKind::Sparse,
        variable_count: 8,
        degree: 5,
        gap: 10,
        coefficient_bits: 30,
    },
    GcdCaseConfig {
        kind: GcdCaseKind::HighGap,
        variable_count: 5,
        degree: 5,
        gap: 64,
        coefficient_bits: 30,
    },
    GcdCaseConfig {
        kind: GcdCaseKind::HighGap,
        variable_count: 8,
        degree: 4,
        gap: 256,
        coefficient_bits: 30,
    },
    GcdCaseConfig {
        kind: GcdCaseKind::HighHeight,
        variable_count: 5,
        degree: 4,
        gap: 10,
        coefficient_bits: 128,
    },
    GcdCaseConfig {
        kind: GcdCaseKind::HighHeight,
        variable_count: 5,
        degree: 4,
        gap: 10,
        coefficient_bits: 256,
    },
    GcdCaseConfig {
        kind: GcdCaseKind::HighHeight,
        variable_count: 5,
        degree: 4,
        gap: 10,
        coefficient_bits: 512,
    },
    GcdCaseConfig {
        kind: GcdCaseKind::HighHeight,
        variable_count: 5,
        degree: 4,
        gap: 10,
        coefficient_bits: 1024,
    },
    GcdCaseConfig {
        kind: GcdCaseKind::HighHeight,
        variable_count: 8,
        degree: 3,
        gap: 10,
        coefficient_bits: 256,
    },
];

impl GcdCaseConfig {
    pub fn validate(self) -> Result<(), String> {
        if !(1..=8).contains(&self.variable_count) {
            return Err("GCD variable count must be between 1 and 8".to_owned());
        }
        if self.degree == 0 || self.degree > u16::MAX as u32 {
            return Err(format!("GCD degree must be between 1 and {}", u16::MAX));
        }
        if self.gap == 0 {
            return Err("GCD exponent gap must be positive".to_owned());
        }
        if !(8..=1024).contains(&self.coefficient_bits) {
            return Err("GCD coefficient size must be between 8 and 1024 bits".to_owned());
        }
        Ok(())
    }

    /// Generates the two cofactors and their expected common factor.
    pub fn generate(self) -> Result<GeneratedGcdCase, String> {
        self.validate()?;

        const B_SIGNS: [i8; 8] = [-1, -1, -1, 1, -1, -1, 1, -1];

        let weights = coefficient_weights(self);
        let positive_signs = vec![1; self.variable_count];
        let mut gcd_signs = positive_signs.clone();
        gcd_signs[self.variable_count - 1] = -1;
        let left_linear = linear_expression(&weights, &positive_signs);
        let right_linear = linear_expression(&weights, &B_SIGNS[..self.variable_count]);
        let gcd_linear = linear_expression(&weights, &gcd_signs);

        let gcd = match self.kind {
            GcdCaseKind::Dense | GcdCaseKind::HighHeight => {
                powered_expression(&gcd_linear, self.degree, 3)
            }
            GcdCaseKind::Sparse => sparse_expression(self.variable_count, self.degree),
            GcdCaseKind::HighGap => sparse_expression(self.variable_count, self.gap),
        };

        Ok(GeneratedGcdCase {
            config: self,
            left_cofactor: powered_expression(&left_linear, self.degree, -1),
            right_cofactor: powered_expression(&right_linear, self.degree, 1),
            gcd,
        })
    }

    pub fn display_name(self) -> String {
        match self.kind {
            GcdCaseKind::Dense | GcdCaseKind::Sparse => format!(
                "{} {} variables degree {}",
                self.kind, self.variable_count, self.degree
            ),
            GcdCaseKind::HighGap => format!(
                "{} {} variables degree {} gap {}",
                self.kind, self.variable_count, self.degree, self.gap
            ),
            GcdCaseKind::HighHeight => format!(
                "{} {} variables degree {} coefficient bits {}",
                self.kind, self.variable_count, self.degree, self.coefficient_bits
            ),
        }
    }
}

impl fmt::Display for GcdCaseConfig {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.display_name())
    }
}

/// Generated expressions used to form `a*g` and `b*g` for a GCD benchmark.
#[derive(Clone, Debug)]
pub struct GeneratedGcdCase {
    pub config: GcdCaseConfig,
    pub left_cofactor: String,
    pub right_cofactor: String,
    pub gcd: String,
}

impl GeneratedGcdCase {
    pub fn variables(&self) -> &'static [&'static str] {
        &X1_TO_X8_VARIABLES[..self.config.variable_count]
    }

    pub fn display_name(&self) -> String {
        self.config.display_name()
    }
}

fn linear_expression(weights: &[String], signs: &[i8]) -> String {
    let mut expression = "1".to_owned();
    for (index, (weight, &sign)) in weights.iter().zip(signs).enumerate() {
        write!(
            expression,
            "{}{weight}*x{}",
            if sign < 0 { '-' } else { '+' },
            index + 1
        )
        .unwrap();
    }
    expression
}

fn coefficient_weights(config: GcdCaseConfig) -> Vec<String> {
    const SMALL_WEIGHTS: [u64; 8] = [3, 5, 7, 9, 11, 13, 15, 17];
    const THIRTY_BIT_WEIGHTS: [u64; 8] = [
        1_000_000_007,
        1_000_000_009,
        1_000_000_033,
        1_000_000_087,
        1_000_000_093,
        1_000_000_097,
        1_000_000_103,
        1_000_000_123,
    ];
    const OFFSETS: [u32; 8] = [7, 9, 33, 87, 93, 97, 103, 123];

    if config.kind != GcdCaseKind::HighHeight {
        SMALL_WEIGHTS[..config.variable_count]
            .iter()
            .map(ToString::to_string)
            .collect()
    } else if config.coefficient_bits == 30 {
        THIRTY_BIT_WEIGHTS[..config.variable_count]
            .iter()
            .map(ToString::to_string)
            .collect()
    } else {
        OFFSETS[..config.variable_count]
            .iter()
            .map(|&offset| power_of_two_plus_offset(config.coefficient_bits, offset))
            .collect()
    }
}

fn powered_expression(linear: &str, degree: u32, constant: i32) -> String {
    format!(
        "({linear})^{degree}{}{constant}",
        if constant < 0 { "" } else { "+" }
    )
}

fn sparse_expression(variable_count: usize, degree: u32) -> String {
    const COEFFICIENTS: [u64; 8] = [1, 2, 3, 5, 7, 11, 13, 17];

    let mut expression = "1".to_owned();
    for (index, coefficient) in COEFFICIENTS[..variable_count].iter().enumerate() {
        if *coefficient == 1 {
            write!(expression, "+x{}^{degree}", index + 1).unwrap();
        } else {
            write!(expression, "+{coefficient}*x{}^{degree}", index + 1).unwrap();
        }
    }
    expression
}

fn power_of_two_plus_offset(bits: u32, offset: u32) -> String {
    let mut decimal_digits = vec![1u8];
    for _ in 1..bits {
        let mut carry = 0;
        for digit in &mut decimal_digits {
            let doubled = *digit as u16 * 2 + carry;
            *digit = (doubled % 10) as u8;
            carry = doubled / 10;
        }
        if carry > 0 {
            decimal_digits.push(carry as u8);
        }
    }

    let mut carry = offset;
    let mut index = 0;
    while carry > 0 {
        if index == decimal_digits.len() {
            decimal_digits.push(0);
        }
        let sum = decimal_digits[index] as u32 + carry;
        decimal_digits[index] = (sum % 10) as u8;
        carry = sum / 10;
        index += 1;
    }

    decimal_digits
        .iter()
        .rev()
        .map(|digit| char::from(b'0' + *digit))
        .collect()
}

#[cfg(test)]
#[allow(unused_imports)]
mod tests {
    use super::*;

    #[test]
    fn generated_gcd_cases_match_the_original_expressions() {
        let dense = GcdCaseConfig {
            kind: GcdCaseKind::Dense,
            variable_count: 2,
            degree: 2,
            gap: 10,
            coefficient_bits: 30,
        }
        .generate()
        .unwrap();
        assert_eq!(dense.left_cofactor, "(1+3*x1+5*x2)^2-1");
        assert_eq!(dense.right_cofactor, "(1-3*x1-5*x2)^2+1");
        assert_eq!(dense.gcd, "(1+3*x1-5*x2)^2+3");

        let sparse = GcdCaseConfig {
            kind: GcdCaseKind::Sparse,
            ..dense.config
        }
        .generate()
        .unwrap();
        assert_eq!(sparse.gcd, "1+x1^2+2*x2^2");

        let high_gap = GcdCaseConfig {
            kind: GcdCaseKind::HighGap,
            ..dense.config
        }
        .generate()
        .unwrap();
        assert_eq!(high_gap.gcd, "1+x1^10+2*x2^10");
    }

    #[test]
    fn generated_high_height_coefficients_have_the_requested_size() {
        let generated = GcdCaseConfig {
            kind: GcdCaseKind::HighHeight,
            variable_count: 2,
            degree: 1,
            gap: 10,
            coefficient_bits: 8,
        }
        .generate()
        .unwrap();
        assert_eq!(generated.left_cofactor, "(1+135*x1+137*x2)^1-1");
    }

    #[test]
    fn dense_univariate_coefficients_match_the_existing_case() {
        let (left, right) = FINITE_FIELD_MULTIPLICATION_CASES[0]
            .input
            .dense_univariate_coefficients()
            .unwrap();
        assert_eq!(
            &left[..18],
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 1, 2]
        );
        assert_eq!(&right[..5], &[1, 8, 15, 6, 13]);
        assert_eq!(left.len(), 4913);
        assert_eq!(right.len(), 4912);
    }
}
