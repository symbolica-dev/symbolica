//! Methods related to series expansion.
//!
//! # Examples
//! ```
//! use symbolica::prelude::*;
//!
//! let x = symbol!("x");
//! let a = parse!("(1-cos(x))/sin(x)");
//!
//! let out = a.series(x, 0, 4).unwrap();
//! println!("{}", out);
//! assert_eq!(out.to_atom(), parse!("1/2*x+1/24*x^3"));
//! ```
use std::{
    cmp::Ordering,
    ops::{Add, Div, Mul, Neg, Sub},
    sync::Arc,
};

use crate::{
    atom::{Atom, AtomCore, AtomView, FunctionBuilder, Symbol},
    coefficient::CoefficientView,
    domains::{
        EuclideanDomain, InternalOrdering, Ring, SelfRing,
        atom::AtomField,
        integer::Integer,
        rational::{Q, Rational},
    },
    printer::{AtomPrinter, PrintOptions, PrintState},
};

use super::PolyVariable;

/// Errors that can occur during series expansion.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SeriesError {
    /// Operands do not share a field, indeterminate, and expansion point.
    IncompatibleSeries,
    /// A finite series index, exponent grid, or loop count cannot be represented.
    IndexOverflow { operation: &'static str },
    /// An explicit caller limit would be exceeded before an allocation or retry.
    LimitExceeded {
        resource: &'static str,
        limit: usize,
        requested: usize,
    },
    /// Reserving a checked coefficient vector failed.
    AllocationFailed { requested: usize },
    /// A prototype cannot have a negative relative precision.
    NegativeOrder { order: Rational },
    /// The existing inverse refinement heuristic ended without establishing a
    /// nonzero leading coefficient. This does not establish a zero divisor.
    UnresolvedInverseLeadingCoefficient {
        expression: Atom,
        relative_depth: Rational,
        heuristic_limit: usize,
    },
    /// The requested relative truncation depth is not positive.
    NonPositiveRelativeDepth {
        /// The rejected truncation depth.
        depth: SeriesDepth,
    },
    /// An `if` expression has a condition that cannot be resolved for expansion.
    NonConstantIf {
        /// The conditional expression.
        expression: Atom,
    },
    /// A function argument has a pole without a suitable series expansion rule.
    FunctionArgumentPole {
        /// The function call requiring an expansion rule.
        expression: Atom,
    },
    /// A power has a non-real exponent that this expansion does not support.
    UnsupportedComplexExponent {
        /// The power expression.
        expression: Atom,
    },
    /// An exponent exceeds the supported size for series expansion.
    UnsupportedLargeExponent {
        /// The power expression.
        expression: Atom,
    },
    /// A power of the expansion variable does not have a rational exponent.
    NonRationalPower {
        /// The expression containing the unsupported power.
        expression: Atom,
    },
    /// Simplifying exponentials and logarithms produced an unsupported form.
    UnexpectedExpLogSimplification {
        /// The expression that could not be converted back to a series.
        expression: Atom,
    },
    /// Applying the function to a pole would produce an essential singularity.
    EssentialSingularity {
        /// The function being expanded.
        function: Symbol,
    },
    /// The logarithm requires a known nonzero leading coefficient.
    MissingLogCoefficient,
    /// A zero remainder does not establish the positive absolute precision
    /// needed to expand a regular function around its zero argument.
    MissingFunctionCoefficient { function: Symbol },
    /// A coefficient treated as constant still depends on the expansion variable.
    ConstantTermDependsOnExpansionVariable {
        /// The function being expanded.
        function: Symbol,
        /// The supposedly constant coefficient.
        constant: Atom,
        /// The expansion variable that still occurs in the coefficient.
        variable: PolyVariable,
    },
    /// The requested expansion variable is not an indeterminate.
    NonIndeterminateSeriesVariable {
        /// The rejected expansion variable.
        variable: PolyVariable,
    },
    /// Raising the series to zero would require representing one at infinite precision.
    ZeroPowerRequiresInfinitePrecision,
    /// A series inversion or division has a zero divisor.
    DivisionByZero,
}

impl std::error::Error for SeriesError {}

impl std::fmt::Display for SeriesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::IncompatibleSeries => {
                f.write_str("series operands have different fields, variables, or expansion points")
            }
            Self::IndexOverflow { operation } => {
                write!(f, "series index overflow during {operation}")
            }
            Self::LimitExceeded {
                resource,
                limit,
                requested,
            } => write!(
                f,
                "series {resource} limit {limit} exceeded by request {requested}"
            ),
            Self::AllocationFailed { requested } => {
                write!(f, "cannot allocate {requested} series coefficient slots")
            }
            Self::NegativeOrder { order } => {
                write!(f, "series prototype order must be nonnegative, got {order}")
            }
            Self::UnresolvedInverseLeadingCoefficient {
                expression,
                relative_depth,
                heuristic_limit,
            } => write!(
                f,
                "could not establish a nonzero leading coefficient of inverse base {expression} by relative depth {relative_depth} under the existing native precision heuristic (limit {heuristic_limit}); this does not prove that the base is zero"
            ),
            SeriesError::NonPositiveRelativeDepth { depth } => {
                write!(f, "relative series depth must be positive, got {depth:?}")
            }
            SeriesError::NonConstantIf { expression } => {
                write!(
                    f,
                    "cannot series expand non-constant if function {expression}"
                )
            }
            SeriesError::FunctionArgumentPole { expression } => write!(
                f,
                "cannot series expand {expression} since an argument has poles; define a 'series' attribute on the symbol that extracts the principal part"
            ),
            SeriesError::UnsupportedComplexExponent { expression } => {
                write!(
                    f,
                    "cannot series expand {expression} with a complex exponent"
                )
            }
            SeriesError::UnsupportedLargeExponent { expression } => {
                write!(
                    f,
                    "cannot series expand {expression} with a large exponent yet"
                )
            }
            SeriesError::NonRationalPower { expression } => {
                write!(f, "power of variable in {expression} must be rational")
            }
            SeriesError::UnexpectedExpLogSimplification { expression } => write!(
                f,
                "unexpected term {expression} in exp-log series simplification"
            ),
            SeriesError::EssentialSingularity { function } => write!(
                f,
                "cannot series expand {function} of a series with poles; this would create an essential singularity"
            ),
            SeriesError::MissingLogCoefficient => {
                f.write_str("log argument needs to have a coefficient")
            }
            Self::MissingFunctionCoefficient { function } => write!(
                f,
                "cannot establish a constant coefficient for {function} from a zero remainder without positive absolute precision"
            ),
            SeriesError::ConstantTermDependsOnExpansionVariable {
                function,
                constant,
                variable,
            } => write!(
                f,
                "cannot compute {function} of a series whose constant term {constant} depends on {variable}"
            ),
            SeriesError::NonIndeterminateSeriesVariable { variable } => {
                write!(f, "series variable {variable} is not an indeterminate")
            }
            SeriesError::ZeroPowerRequiresInfinitePrecision => f.write_str(
                "cannot raise series to the power of zero, as this generates infinite precision 1",
            ),
            SeriesError::DivisionByZero => {
                f.write_str("cannot invert series with a zero constant term")
            }
        }
    }
}

/// Optional limits on native series containers and precision retries.
///
/// The default imposes no optional resource cap. Machine-size arithmetic is
/// checked independently. Slots bound the requested length of each individual
/// coefficient vector, not total live slots or allocator capacity. Coefficient
/// algebra and registered callback internals, including library-provided
/// callbacks, are outside these controls and are not subject to a process
/// memory/time limit.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SeriesSettings {
    max_coefficient_slots: Option<usize>,
    max_ramification: Option<usize>,
    max_precision_retries: Option<usize>,
}

impl SeriesSettings {
    /// Maximum slots requested by one native coefficient container, if capped.
    pub fn coefficient_slots_limit(&self) -> Option<usize> {
        self.max_coefficient_slots
    }

    /// Maximum intermediate exponent-grid denominator, if capped.
    pub fn ramification_limit(&self) -> Option<usize> {
        self.max_ramification
    }

    /// Maximum precision retries shared by one expansion, if capped.
    pub fn precision_retries_limit(&self) -> Option<usize> {
        self.max_precision_retries
    }

    pub fn max_coefficient_slots(mut self, limit: impl Into<Option<usize>>) -> Self {
        self.max_coefficient_slots = limit.into();
        self
    }

    pub fn max_ramification(mut self, limit: impl Into<Option<usize>>) -> Self {
        self.max_ramification = limit.into();
        self
    }

    pub fn max_precision_retries(mut self, limit: impl Into<Option<usize>>) -> Self {
        self.max_precision_retries = limit.into();
        self
    }

    fn check(
        &self,
        resource: &'static str,
        requested: usize,
        limit: Option<usize>,
    ) -> Result<(), SeriesError> {
        if let Some(limit) = limit
            && requested > limit
        {
            return Err(SeriesError::LimitExceeded {
                resource,
                limit,
                requested,
            });
        }
        Ok(())
    }

    fn slots(&self, requested: usize) -> Result<(), SeriesError> {
        self.check("coefficient slots", requested, self.max_coefficient_slots)
    }

    fn ramification(&self, requested: usize) -> Result<(), SeriesError> {
        self.check("ramification", requested, self.max_ramification)
    }
}

/// One shared retry counter for the complete native expansion call.
pub(crate) struct SeriesContext<'a> {
    settings: &'a SeriesSettings,
    retries: usize,
}

impl<'a> SeriesContext<'a> {
    pub(crate) fn new(settings: &'a SeriesSettings) -> Self {
        Self {
            settings,
            retries: 0,
        }
    }

    pub(crate) fn settings(&self) -> &'a SeriesSettings {
        self.settings
    }

    pub(crate) fn retry(&mut self) -> Result<(), SeriesError> {
        let requested = self
            .retries
            .checked_add(1)
            .ok_or(SeriesError::IndexOverflow {
                operation: "precision retries",
            })?;
        self.settings.check(
            "precision retries",
            requested,
            self.settings.max_precision_retries,
        )?;
        self.retries = requested;
        Ok(())
    }
}

/// The requested truncation depth of a series expansion.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SeriesDepth {
    /// Truncate at an absolute power of the expansion variable.
    Absolute(Rational),
    /// Truncate relative to the lowest order that appears in the expansion.
    Relative(Rational),
}

impl SeriesDepth {
    /// Truncate at an absolute power of the expansion variable.
    pub fn absolute<T: Into<Rational>>(depth: T) -> Self {
        Self::Absolute(depth.into())
    }

    /// Truncate relative to the lowest order that appears in the expansion.
    pub fn relative<T: Into<Rational>>(depth: T) -> Self {
        Self::Relative(depth.into())
    }

    pub(crate) fn order(&self) -> &Rational {
        match self {
            SeriesDepth::Absolute(depth) | SeriesDepth::Relative(depth) => depth,
        }
    }

    pub(crate) fn is_absolute(&self) -> bool {
        matches!(self, SeriesDepth::Absolute(_))
    }
}

impl<T: Into<Rational>> From<T> for SeriesDepth {
    fn from(depth: T) -> Self {
        Self::absolute(depth.into())
    }
}

/// A Puiseux series. The truncation order is
/// relative to the lowest degree: i.e., a series in `x` with depth `d` is viewed as
/// `a*x^n*(1+bx+.. + O(x^(d+1)))`.
///
/// # Examples
///
/// A series can be constructed from an [Atom]:
/// ```
/// use symbolica::prelude::*;
///
/// let x = symbol!("x");
/// let a = parse!("(1-cos(x))/sin(x)");
///
/// let out = a.series(x, 0, 4).unwrap();
/// println!("{}", out);
/// assert_eq!(out.to_atom(), parse!("1/2*x+1/24*x^3"));
/// ```
#[derive(Clone)]
pub struct Series<F: Ring> {
    coefficients: Vec<F::Element>,
    variable: Arc<PolyVariable>,
    expansion_point: F::Element,
    field: F,
    shift: isize, // a shift in units of the ramification that indicates the starting power of the series
    order: usize, // the order of truncation in units of the ramification
    ramification: usize, // the factor that makes the exponents integer
}

impl<F: Ring + std::fmt::Debug> std::fmt::Debug for Series<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.order == 0 {
            return write!(f, "[]");
        }
        let mut first = true;
        write!(f, "[ ")?;
        for c in self.coefficients.iter() {
            if first {
                first = false;
            } else {
                write!(f, ", ")?;
            }
            write!(f, "{{ {c:?} }}")?;
        }
        write!(f, " ]")
    }
}

impl<F: Ring + std::fmt::Display> std::fmt::Display for Series<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.format(&PrintOptions::from_fmt(f), PrintState::from_fmt(f), f)
            .map(|_| ())
    }
}

impl<F: Ring> Series<F> {
    /// Constructs a zero series with fixed order. Instead of using this constructor,
    /// prefer to create new series from existing ones, so that the
    /// variable map and field are inherited.
    #[inline]
    pub fn new(
        field: &F,
        cap: Option<usize>,
        variable: Arc<PolyVariable>,
        expansion_point: F::Element,
        order: Rational,
    ) -> Self {
        Self::try_new(
            field,
            cap,
            variable,
            expansion_point,
            order,
            &SeriesSettings::default(),
        )
        .expect("invalid series prototype")
    }

    /// Fallible prototype construction with checked finite native geometry.
    pub fn try_new(
        field: &F,
        cap: Option<usize>,
        variable: Arc<PolyVariable>,
        expansion_point: F::Element,
        order: Rational,
        settings: &SeriesSettings,
    ) -> Result<Self, SeriesError> {
        if order.is_negative() {
            return Err(SeriesError::NegativeOrder { order });
        }
        let order_units = Self::unsigned(order.numerator_ref(), "prototype order")?;
        if order_units == i64::MAX as usize {
            return Err(SeriesError::IndexOverflow {
                operation: "finite prototype order",
            });
        }
        let ramification = Self::unsigned(order.denominator_ref(), "prototype ramification")?;
        let mut result = Self {
            coefficients: Vec::new(),
            expansion_point,
            field: field.clone(),
            variable,
            shift: 0,
            order: order_units,
            ramification,
        };
        result.check_geometry(settings)?;
        Self::reserve(&mut result.coefficients, cap.unwrap_or(0), settings)?;
        Ok(result)
    }

    fn unsigned(value: &Integer, operation: &'static str) -> Result<usize, SeriesError> {
        value
            .to_i64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or(SeriesError::IndexOverflow { operation })
    }

    fn signed(value: &Integer, operation: &'static str) -> Result<isize, SeriesError> {
        value
            .to_i64()
            .and_then(|value| isize::try_from(value).ok())
            .ok_or(SeriesError::IndexOverflow { operation })
    }

    fn reserve(
        values: &mut Vec<F::Element>,
        length: usize,
        settings: &SeriesSettings,
    ) -> Result<(), SeriesError> {
        settings.slots(length)?;
        if length > isize::MAX as usize / std::mem::size_of::<F::Element>().max(1) {
            return Err(SeriesError::IndexOverflow {
                operation: "coefficient allocation",
            });
        }
        if length > values.len() {
            values
                .try_reserve_exact(length - values.len())
                .map_err(|_| SeriesError::AllocationFailed { requested: length })?;
        }
        Ok(())
    }

    fn resize(
        &self,
        values: &mut Vec<F::Element>,
        length: usize,
        settings: &SeriesSettings,
    ) -> Result<(), SeriesError> {
        Self::reserve(values, length, settings)?;
        values.resize(length, self.field.zero());
        Ok(())
    }

    fn check_geometry(&self, settings: &SeriesSettings) -> Result<(), SeriesError> {
        if self.ramification == 0
            || self.ramification > i64::MAX as usize
            || self.order > i64::MAX as usize
        {
            return Err(SeriesError::IndexOverflow {
                operation: "series geometry",
            });
        }
        settings.ramification(self.ramification)?;
        settings.slots(self.coefficients.len())?;
        // i64::MAX is the existing internal infinite-precision sentinel. Its
        // order is combined with a finite operand before publication/access.
        if self.order != i64::MAX as usize {
            let order = isize::try_from(self.order).map_err(|_| SeriesError::IndexOverflow {
                operation: "precision units",
            })?;
            self.shift
                .checked_add(order)
                .ok_or(SeriesError::IndexOverflow {
                    operation: "absolute precision",
                })?;
        }
        Ok(())
    }

    /// Clone the coefficient container through its checked allocation boundary.
    pub fn try_clone(&self, settings: &SeriesSettings) -> Result<Self, SeriesError> {
        self.check_geometry(settings)?;
        let mut coefficients = Vec::new();
        Self::reserve(&mut coefficients, self.coefficients.len(), settings)?;
        coefficients.extend(self.coefficients.iter().cloned());
        Ok(Self {
            coefficients,
            variable: self.variable.clone(),
            expansion_point: self.expansion_point.clone(),
            field: self.field.clone(),
            shift: self.shift,
            order: self.order,
            ramification: self.ramification,
        })
    }

    fn rescaled_order(order: usize, ratio: usize) -> Result<usize, SeriesError> {
        if order == i64::MAX as usize {
            return Ok(order);
        }
        order
            .checked_mul(ratio)
            .filter(|&value| value < i64::MAX as usize)
            .ok_or(SeriesError::IndexOverflow {
                operation: "rescaled precision",
            })
    }

    /// Constructs a finite-precision zero series, inheriting the field and variable from `self`.
    #[inline]
    pub fn zero(&self) -> Self {
        Self {
            coefficients: vec![],
            field: self.field.clone(),
            variable: self.variable.clone(),
            expansion_point: self.expansion_point.clone(),
            shift: 0,
            order: self.order,
            ramification: self.ramification,
        }
    }

    /// Constructs a finite-precision constant series,
    /// inheriting the field and variable from `self`.
    #[inline]
    pub fn constant(&self, coeff: F::Element) -> Self {
        self.try_constant(coeff, &SeriesSettings::default())
            .expect("invalid constant series")
    }

    pub fn try_constant(
        &self,
        coeff: F::Element,
        settings: &SeriesSettings,
    ) -> Result<Self, SeriesError> {
        self.check_geometry(settings)?;
        if self.field.is_zero(&coeff) {
            return Ok(self.zero());
        }
        let mut coefficients = Vec::new();
        Self::reserve(&mut coefficients, 1, settings)?;
        coefficients.push(coeff);
        Ok(Self {
            coefficients,
            field: self.field.clone(),
            variable: self.variable.clone(),
            expansion_point: self.expansion_point.clone(),
            shift: 0,
            order: self.order,
            ramification: self.ramification,
        })
    }

    /// Constructs a finite-precision series that is one, inheriting the field and variable from `self`.
    #[inline]
    pub fn one(&self) -> Self {
        self.constant(self.field.one())
    }

    /// Constructs a series that is one with infinite digits.
    /// Must be added or multiplied with another finite series.
    #[inline]
    fn try_one_inf_prec(&self, settings: &SeriesSettings) -> Result<Self, SeriesError> {
        let mut result = self.try_constant(self.field.one(), settings)?;
        result.order = i64::MAX as usize;
        result.ramification = 1;
        Ok(result)
    }

    /// Constructs a series with a single term.
    #[inline]
    pub fn monomial(&self, coeff: F::Element, exponent: Rational) -> Self {
        self.try_monomial(coeff, exponent, &SeriesSettings::default())
            .expect("invalid monomial series")
    }

    pub fn try_monomial(
        &self,
        coeff: F::Element,
        exponent: Rational,
        settings: &SeriesSettings,
    ) -> Result<Self, SeriesError> {
        self.check_geometry(settings)?;
        if self.field.is_zero(&coeff) {
            return Ok(self.zero());
        }
        let ram = Self::unsigned(
            &Integer::from(self.ramification).lcm(exponent.denominator_ref()),
            "monomial ramification",
        )?;
        settings.ramification(ram)?;
        let shift = Self::signed(
            &(exponent.numerator_ref() * (Integer::from(ram) / exponent.denominator_ref())),
            "monomial shift",
        )?;
        let order = Self::rescaled_order(self.order, ram / self.ramification)?;
        let mut result = self.try_constant(coeff, settings)?;
        result.shift = shift;
        result.order = order;
        result.ramification = ram;
        result.check_geometry(settings)?;
        Ok(result)
    }

    /// Constructs a series that is `x+a`, where `a` is a constant.
    #[inline]
    pub fn shifted_variable(&self, coeff: F::Element) -> Self {
        self.try_shifted_variable(coeff, &SeriesSettings::default())
            .expect("invalid shifted series variable")
    }

    pub fn try_shifted_variable(
        &self,
        coeff: F::Element,
        settings: &SeriesSettings,
    ) -> Result<Self, SeriesError> {
        if self.field.is_zero(&coeff) {
            return self.try_monomial(self.field.one(), (1, 1).into(), settings);
        }
        self.try_constant(coeff, settings)?.try_add(
            &self.try_monomial(self.field.one(), (1, 1).into(), settings)?,
            settings,
        )
    }

    /// Get the field of the series coefficients.
    #[inline]
    pub fn get_field(&self) -> &F {
        &self.field
    }

    // Map an index in the coefficient array to its power.
    #[inline]
    fn get_exponent(&self, index: usize) -> Rational {
        Self::get_exponent_from_shift_ram(index, self.shift, self.ramification)
    }

    /// Map an index in the coefficient array to its power, given a shift and ramification.
    fn get_exponent_from_shift_ram(index: usize, shift: isize, ram: usize) -> Rational {
        (Rational::from(index as i64) + Rational::from(shift as i64)) / Rational::from(ram as i64)
    }

    fn try_index(&self, p: Rational) -> Result<usize, SeriesError> {
        let r = p * &Rational::from(self.ramification as i64);
        if !r.is_integer() {
            return Err(SeriesError::IndexOverflow {
                operation: "coefficient grid alignment",
            });
        }
        Self::unsigned(
            &(r.numerator_ref() - Integer::from(self.shift)),
            "coefficient index",
        )
    }

    /// Get the relative order of the series expansion.
    #[inline]
    pub fn relative_order(&self) -> Rational {
        assert!(self.order < i64::MAX as usize);
        (self.order as i64, self.ramification as i64).into()
    }

    /// Get the absolute order of the series expansion.
    #[inline]
    pub fn absolute_order(&self) -> Rational {
        assert!(self.order < i64::MAX as usize);
        (
            self.order as i64 + self.shift as i64,
            self.ramification as i64,
        )
            .into()
    }

    /// Get the expansion point.
    #[inline]
    pub fn get_expansion_point(&self) -> F::Element {
        self.expansion_point.clone()
    }

    fn try_change_ramification(
        &mut self,
        ram: usize,
        settings: &SeriesSettings,
    ) -> Result<(), SeriesError> {
        self.check_geometry(settings)?;
        let ram = Self::unsigned(
            &Integer::from(self.ramification).lcm(&Integer::from(ram)),
            "combined ramification",
        )?;
        settings.ramification(ram)?;

        if ram == self.ramification {
            return Ok(());
        }
        let ratio = ram / self.ramification;
        let shift = self
            .shift
            .checked_mul(
                isize::try_from(ratio).map_err(|_| SeriesError::IndexOverflow {
                    operation: "ramification ratio",
                })?,
            )
            .ok_or(SeriesError::IndexOverflow {
                operation: "rescaled shift",
            })?;
        let order = Self::rescaled_order(self.order, ratio)?;
        let length =
            self.coefficients
                .len()
                .checked_mul(ratio)
                .ok_or(SeriesError::IndexOverflow {
                    operation: "rescaled coefficient length",
                })?;
        let mut s = Self {
            coefficients: Vec::new(),
            field: self.field.clone(),
            variable: self.variable.clone(),
            expansion_point: self.expansion_point.clone(),
            shift,
            order,
            ramification: ram,
        };
        s.check_geometry(settings)?;
        self.resize(&mut s.coefficients, length, settings)?;

        for (i, c) in std::mem::take(&mut self.coefficients)
            .into_iter()
            .enumerate()
        {
            let index = i.checked_mul(ratio).ok_or(SeriesError::IndexOverflow {
                operation: "rescaled coefficient index",
            })?;
            s.coefficients[index] = c;
        }

        *self = s;
        Ok(())
    }

    /// Truncate the series to the desired relative order.
    /// If the new order is larger, nothing happens.
    #[inline]
    pub fn truncate_relative_order(&mut self, order: Rational) {
        self.try_truncate_relative_order(order, &SeriesSettings::default())
            .expect("invalid relative series truncation")
    }

    pub fn try_truncate_relative_order(
        &mut self,
        order: Rational,
        settings: &SeriesSettings,
    ) -> Result<(), SeriesError> {
        self.check_geometry(settings)?;
        if self.relative_order() < order {
            return Ok(());
        }

        if order.is_negative() {
            return Err(SeriesError::NegativeOrder { order });
        }

        let ram = Self::unsigned(order.denominator_ref(), "truncation ramification")?;
        self.try_change_ramification(ram, settings)?;
        let new_order = Self::unsigned(
            &(order.numerator_ref() * Integer::from(self.ramification / ram)),
            "relative truncation precision",
        )?;
        if new_order >= i64::MAX as usize {
            return Err(SeriesError::IndexOverflow {
                operation: "relative truncation precision",
            });
        }

        self.coefficients.truncate(new_order);
        self.order = new_order;
        self.truncate();
        self.check_geometry(settings)
    }

    /// Truncate the series to the desired absolute order.
    /// If the new order is larger, nothing happens.
    /// An empty series retains its known order since no terms need to be removed.
    #[inline]
    pub fn truncate_absolute_order(&mut self, order: Rational) {
        self.try_truncate_absolute_order(order, &SeriesSettings::default())
            .expect("invalid absolute series truncation")
    }

    pub fn try_truncate_absolute_order(
        &mut self,
        order: Rational,
        settings: &SeriesSettings,
    ) -> Result<(), SeriesError> {
        self.check_geometry(settings)?;
        if self.coefficients.is_empty() || self.absolute_order() < order {
            return Ok(());
        }

        self.try_change_ramification(
            Self::unsigned(order.denominator_ref(), "truncation ramification")?,
            settings,
        )?;

        for i in (0..self.coefficients.len()).rev() {
            if self.get_exponent(i) >= order {
                self.coefficients.pop();
            } else {
                break;
            }
        }

        if self.coefficients.is_empty() {
            self.order = 0;
        } else {
            self.order = self.try_index(order)?;
        }
        self.check_geometry(settings)
    }

    #[inline]
    fn try_joint_ramification(
        &self,
        other: &Self,
        settings: &SeriesSettings,
    ) -> Result<usize, SeriesError> {
        let ram = Self::unsigned(
            &Integer::from(self.ramification).lcm(&Integer::from(other.ramification)),
            "joint ramification",
        )?;
        settings.ramification(ram)?;
        Ok(ram)
    }

    #[inline]
    /// Return whether no nonzero terms are stored. This tests the known terms,
    /// not whether the unknown remainder is identically zero.
    pub fn is_zero(&self) -> bool {
        self.coefficients.is_empty()
    }

    #[inline]
    /// Return whether the stored terms consist only of the constant one.
    /// The unknown remainder is not tested.
    pub fn is_one(&self) -> bool {
        self.coefficients.len() == 1 && self.field.is_one(&self.coefficients[0]) && self.shift == 0
    }

    /// Returns true if the series is constant.
    #[inline]
    pub fn is_constant(&self) -> bool {
        self.coefficients.len() <= 1 && self.shift == 0
    }

    #[inline]
    /// Return the coefficient of the lowest stored power, or zero when no terms
    /// are stored.
    pub fn get_trailing_coefficient(&self) -> F::Element {
        if self.coefficients.is_empty() {
            self.field.zero()
        } else {
            self.coefficients[0].clone()
        }
    }

    #[inline]
    /// Return the lowest exponent represented by the series storage, including
    /// the shift and ramification.
    pub fn get_trailing_exponent(&self) -> Rational {
        self.get_exponent(0)
    }

    #[inline]
    /// Return the common denominator used to represent fractional exponents.
    /// Adjacent coefficient slots differ in exponent by its reciprocal.
    pub fn get_ramification(&self) -> usize {
        self.ramification
    }

    /// Get a copy of the variable/
    pub fn get_variable(&self) -> Arc<PolyVariable> {
        self.variable.clone()
    }

    /// Get a reference to the variables
    pub fn get_vars_ref(&self) -> &PolyVariable {
        self.variable.as_ref()
    }

    /// Get the leading coefficient.
    pub fn lcoeff(&self) -> F::Element {
        self.coefficients
            .last()
            .unwrap_or(&self.field.zero())
            .clone()
    }

    /// Return the highest stored exponent, or zero when the series has no
    /// known coefficients. This differs from the truncation order.
    pub fn degree(&self) -> Rational {
        if self.order == 0 {
            return 0.into();
        }

        self.get_exponent(self.coefficients.len() - 1)
    }

    /// Compute `self^pow`. `pow` must be a positive integer.
    pub fn npow(&self, pow: usize) -> Self {
        self.try_npow(pow, &SeriesSettings::default())
            .expect("invalid series power")
    }

    pub fn try_npow(&self, mut pow: usize, settings: &SeriesSettings) -> Result<Self, SeriesError> {
        if pow == 0 {
            return Err(SeriesError::ZeroPowerRequiresInfinitePrecision);
        }
        let mut x = self.try_clone(settings)?;
        let mut y = self.try_one_inf_prec(settings)?;
        while pow != 1 {
            if pow % 2 == 1 {
                y = y.try_mul(&x, settings)?;
                pow -= 1;
            }

            x = x.try_mul(&x, settings)?;
            pow /= 2;
        }

        x.try_mul(&y, settings)
    }

    /// Multiply the exponents `exp` units of the ramification.
    pub fn mul_exp_units(self, exp: isize) -> Self {
        self.try_mul_exp_units(exp, &SeriesSettings::default())
            .expect("invalid series shift")
    }

    pub fn try_mul_exp_units(
        mut self,
        exp: isize,
        settings: &SeriesSettings,
    ) -> Result<Self, SeriesError> {
        self.shift = self
            .shift
            .checked_add(exp)
            .ok_or(SeriesError::IndexOverflow {
                operation: "exponent shift",
            })?;
        self.check_geometry(settings)?;
        Ok(self)
    }

    /// Multiply every stored coefficient by `coeff`, preserving the truncation
    /// information.
    pub fn mul_coeff(mut self, coeff: &F::Element) -> Self {
        for c in &mut self.coefficients {
            if !self.field.is_zero(c) {
                self.field.mul_assign(c, coeff);
            }
        }

        self.truncate(); // zeros may have occurred

        self
    }

    /// Truncate the leading and trailing non-zeroes.
    fn truncate(&mut self) {
        if self.coefficients.is_empty() {
            return;
        }

        let d = self
            .coefficients
            .iter_mut()
            .rev()
            .position(|c| !self.field.is_zero(c))
            .unwrap_or(self.coefficients.len());

        self.coefficients.truncate(self.coefficients.len() - d);

        if self.coefficients.is_empty() {
            self.shift += self.order as isize;
            self.order = 0;
            return;
        }

        let d = self
            .coefficients
            .iter_mut()
            .position(|c| !self.field.is_zero(c))
            .unwrap_or(self.coefficients.len());

        self.shift += d as isize;
        self.order -= d;
        self.coefficients.drain(0..d);
    }

    /// Remove the constant term, if it is first and it exists.
    fn remove_constant(mut self) -> Self {
        if self.order > 0 && self.shift == 0 {
            self.coefficients[0] = self.field.zero();
            self.truncate();
        }

        self
    }

    /// Get the coefficient of `(variable - expansion_point)^exponent`.
    ///
    /// Returns `None` at or above [`Self::absolute_order`], where the remainder
    /// is unknown. Below that order, absent terms (including exponents outside
    /// the ramification grid) have coefficient `Some(0)`.
    pub fn coefficient(&self, exponent: Rational) -> Option<F::Element> {
        if exponent >= self.absolute_order() {
            return None;
        }
        let r = exponent * &Rational::from(self.ramification as i64);
        if !r.is_integer() {
            return Some(self.field.zero());
        }

        let i = r - Rational::from(self.shift as i64);
        if i >= 0 && i < Rational::from(self.coefficients.len() as i64) {
            Some(self.coefficients[i.numerator().to_i64().unwrap() as usize].clone())
        } else {
            Some(self.field.zero())
        }
    }

    /// Iterate over the terms of the series, yielding the exponent and its coefficient.
    pub fn terms(&self) -> impl Iterator<Item = (Rational, &F::Element)> {
        self.coefficients
            .iter()
            .enumerate()
            .map(|(i, c)| (self.get_exponent(i), c))
    }

    /// Map the coefficients of the series using a function `f`.
    pub fn map_coeff<M: Fn(&F::Element) -> F::Element>(&self, f: M) -> Series<F> {
        let mut s = Series {
            coefficients: self.coefficients.iter().map(f).collect(),
            variable: self.variable.clone(),
            expansion_point: self.expansion_point.clone(),
            shift: self.shift,
            ramification: self.ramification,
            field: self.field.clone(),
            order: self.order,
        };

        s.truncate();
        s
    }
}

impl<'a, R: Ring> IntoIterator for Series<R> {
    type Item = (Rational, R::Element);
    type IntoIter = std::vec::IntoIter<(Rational, R::Element)>;

    /// Create an iterator over the terms of the series, yielding the exponent and its coefficient.
    fn into_iter(self) -> Self::IntoIter {
        let shift = self.shift;
        let ram = self.ramification;
        let v: Vec<_> = self
            .coefficients
            .into_iter()
            .enumerate()
            .map(|(i, c)| (Self::get_exponent_from_shift_ram(i, shift, ram), c))
            .collect();

        v.into_iter()
    }
}

impl<'a, R: Ring> IntoIterator for &'a Series<R> {
    type Item = (Rational, &'a R::Element);
    type IntoIter = std::vec::IntoIter<(Rational, &'a R::Element)>;

    /// Create an iterator over the terms of the series, yielding the exponent and its coefficient.
    fn into_iter(self) -> Self::IntoIter {
        let v: Vec<_> = self
            .coefficients
            .iter()
            .enumerate()
            .map(|(i, c)| (self.get_exponent(i), c))
            .collect();

        v.into_iter()
    }
}

impl<F: Ring> SelfRing for Series<F> {
    fn is_zero(&self) -> bool {
        self.is_zero()
    }

    fn is_one(&self) -> bool {
        self.is_one()
    }

    fn format<W: std::fmt::Write>(
        &self,
        opts: &PrintOptions,
        mut state: PrintState,
        f: &mut W,
    ) -> Result<bool, std::fmt::Error> {
        fn format_exponent<W: std::fmt::Write>(
            exponent: &Rational,
            opts: &PrintOptions,
            state: PrintState,
            f: &mut W,
        ) -> std::fmt::Result {
            let superscript =
                opts.mode.is_symbolica() && opts.num_exp_as_superscript && exponent.is_integer();
            if !superscript {
                f.write_char('^')?;
            }
            if opts.mode.is_latex() {
                f.write_char('{')?;
            }

            if superscript {
                if exponent.is_negative() {
                    f.write_char('⁻')?;
                }

                let mut exponent_state = state.step(false, false, true, false);
                exponent_state.superscript = true;
                AtomPrinter::format_digits(
                    exponent.numerator_ref().abs().to_string(),
                    opts,
                    &exponent_state,
                    f,
                )?;
            } else {
                Q.format(exponent, opts, state.step(false, false, true, false), f)?;
            }

            if opts.mode.is_latex() {
                f.write_char('}')?;
            }
            Ok(())
        }

        let v = if self.field.is_zero(&self.expansion_point) {
            self.variable.format_string(
                opts,
                PrintState {
                    in_exp_base: true,
                    ..state
                },
            )
        } else {
            let v = self.variable.format_string(
                opts,
                PrintState {
                    in_sum: true,
                    ..state
                },
            );
            format!("({}-{})", v, self.field.printer(&self.expansion_point))
        };

        if self.coefficients.is_empty() {
            let o = self.absolute_order();
            if opts.mode.is_latex() {
                write!(f, "\\mathcal{{O}}\\left({v}^{{{o}}}\\right)")?;
            } else {
                write!(f, "𝒪({v}")?;
                format_exponent(&o, opts, state, f)?;
                f.write_char(')')?;
            }
            return Ok(false);
        }

        let add_paren = state.in_product || state.in_exp || state.in_exp_base;
        if add_paren {
            if state.in_sum {
                f.write_str("+")?;
                state.in_sum = false;
            }

            state.in_product = false;
            state.in_exp = false;
            state.in_exp_base = false;
            f.write_str("(")?;
        }
        let in_product = state.in_product;

        for (e, c) in self.coefficients.iter().enumerate() {
            if self.field.is_zero(c) {
                continue;
            }

            let e = self.get_exponent(e);

            state.in_product = in_product || !e.is_zero();
            state.suppress_one = !e.is_zero();
            let suppressed_one = self.field.format(
                c,
                opts,
                state.step(state.in_sum, state.in_product, false, false),
                f,
            )?;

            if !suppressed_one && !e.is_zero() {
                f.write_char(opts.multiplication_operator)?;
            }

            if e.is_one() {
                write!(f, "{v}")?;
            } else if !e.is_zero() {
                write!(f, "{v}")?;
                state.suppress_one = false;
                format_exponent(&e, opts, state, f)?;
            }

            state.in_sum = true;
        }

        let o = self.absolute_order();

        if opts.mode.is_latex() {
            write!(f, "+\\mathcal{{O}}\\left({v}^{{{o}}}\\right)")?;
        } else {
            write!(f, "+𝒪({v}")?;
            format_exponent(&o, opts, state, f)?;
            f.write_char(')')?;
        }

        if add_paren {
            f.write_str(")")?;
        }

        Ok(false)
    }
}

impl<F: Ring> InternalOrdering for Series<F> {
    fn internal_cmp(&self, other: &Self) -> Ordering {
        if self.variable != other.variable {
            return self.variable.cmp(&other.variable);
        }

        if self.shift != other.shift {
            return self.shift.cmp(&other.shift);
        }

        if self.ramification != other.ramification {
            return self.ramification.cmp(&other.ramification);
        }

        if self.order != other.order {
            return self.order.cmp(&other.order);
        }

        self.coefficients.internal_cmp(&other.coefficients)
    }
}

impl<F: Ring> PartialEq for Series<F> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if self.variable != other.variable {
            if self.is_constant() != other.is_constant() {
                return false;
            }

            if self.order != other.order {
                // TODO: compare the common orders?
                return false;
            }

            if self.order == 0 {
                return true;
            }

            if self.is_constant() {
                return self.coefficients[0] == other.coefficients[0];
            }

            // TODO: what is expected here?
            unimplemented!("Cannot compare non-constant series with different variable maps yet");
        }

        if self.degree() != other.degree() {
            return false;
        }
        self.coefficients.eq(&other.coefficients)
    }
}

impl<F: Ring> std::hash::Hash for Series<F> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.coefficients.hash(state);
        self.variable.hash(state);
    }
}

impl<F: Ring> Eq for Series<F> {}

impl<R: Ring> PartialOrd for Series<R> {
    /// An ordering of series that has no intuitive meaning.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.coefficients.internal_cmp(&other.coefficients))
    }
}

impl<F: Ring> Series<F> {
    fn add_same_grid(
        mut self,
        other: &Self,
        settings: &SeriesSettings,
    ) -> Result<Self, SeriesError> {
        self.order = self.order.min(other.order);
        self.coefficients.truncate(self.order);
        for (position, coefficient) in other.coefficients.iter().enumerate() {
            if position < self.coefficients.len() {
                self.field
                    .add_assign(&mut self.coefficients[position], coefficient);
            }
        }
        self.truncate();
        self.check_geometry(settings)?;
        Ok(self)
    }

    fn try_add_owned(
        mut self,
        mut other: Self,
        settings: &SeriesSettings,
    ) -> Result<Self, SeriesError> {
        self.check_geometry(settings)?;
        other.check_geometry(settings)?;
        if self.field != other.field
            || self.variable != other.variable
            || self.expansion_point != other.expansion_point
        {
            return Err(SeriesError::IncompatibleSeries);
        }
        if self.shift == other.shift && self.ramification == other.ramification {
            if self.coefficients.len() < other.coefficients.len() {
                std::mem::swap(&mut self, &mut other);
            }
            return self.add_same_grid(&other, settings);
        }
        self.try_add(&other, settings)
    }

    /// Add using the native series kernel with checked geometry and allocation.
    pub fn try_add(&self, other: &Self, settings: &SeriesSettings) -> Result<Self, SeriesError> {
        self.check_geometry(settings)?;
        other.check_geometry(settings)?;
        if self.field != other.field
            || self.variable != other.variable
            || self.expansion_point != other.expansion_point
        {
            return Err(SeriesError::IncompatibleSeries);
        }
        if self.shift == other.shift && self.ramification == other.ramification {
            let (long, short) = if self.coefficients.len() >= other.coefficients.len() {
                (self, other)
            } else {
                (other, self)
            };
            return long.try_clone(settings)?.add_same_grid(short, settings);
        }
        let ramification = self.try_joint_ramification(other, settings)?;
        let left_ratio = ramification / self.ramification;
        let right_ratio = ramification / other.ramification;
        let left_shift = Self::signed(
            &(Integer::from(self.shift) * Integer::from(left_ratio)),
            "addition left shift",
        )?;
        let right_shift = Self::signed(
            &(Integer::from(other.shift) * Integer::from(right_ratio)),
            "addition right shift",
        )?;
        let shift = left_shift.min(right_shift);
        let endpoint =
            |source: &Self, ratio, scaled_shift| -> Result<Option<Integer>, SeriesError> {
                if source.order == i64::MAX as usize {
                    Ok(None)
                } else {
                    Ok(Some(
                        Integer::from(Self::rescaled_order(source.order, ratio)?)
                            + Integer::from(scaled_shift),
                    ))
                }
            };
        let end = match (
            endpoint(self, left_ratio, left_shift)?,
            endpoint(other, right_ratio, right_shift)?,
        ) {
            (Some(left), Some(right)) => Some(left.min(right)),
            (Some(end), None) | (None, Some(end)) => Some(end),
            (None, None) => None,
        };
        let order = match end {
            Some(end) => {
                let order = Self::unsigned(&(end - Integer::from(shift)), "addition precision")?;
                if order == i64::MAX as usize {
                    return Err(SeriesError::IndexOverflow {
                        operation: "finite addition precision",
                    });
                }
                order
            }
            None => i64::MAX as usize,
        };
        let mut result = Self {
            coefficients: Vec::new(),
            field: self.field.clone(),
            variable: self.variable.clone(),
            expansion_point: self.expansion_point.clone(),
            shift,
            order,
            ramification,
        };
        result.check_geometry(settings)?;
        self.resize(
            &mut result.coefficients,
            self.coefficients.len().max(other.coefficients.len()),
            settings,
        )?;
        for source in [self, other] {
            for (position, coefficient) in source.coefficients.iter().enumerate() {
                let index = result.try_index(source.get_exponent(position))?;
                if index < result.order {
                    if index >= result.coefficients.len() {
                        let length = index
                            .checked_mul(2)
                            .and_then(|length| {
                                index.checked_add(1).map(|minimum| length.max(minimum))
                            })
                            .ok_or(SeriesError::IndexOverflow {
                                operation: "addition coefficient growth",
                            })?;
                        self.resize(&mut result.coefficients, length, settings)?;
                    }
                    self.field
                        .add_assign(&mut result.coefficients[index], coefficient);
                }
            }
        }
        result.truncate();
        result.check_geometry(settings)?;
        Ok(result)
    }

    /// Multiply using the existing convolution with checked native indices.
    pub fn try_mul(&self, rhs: &Self, settings: &SeriesSettings) -> Result<Self, SeriesError> {
        self.check_geometry(settings)?;
        rhs.check_geometry(settings)?;
        if self.field != rhs.field
            || self.variable != rhs.variable
            || self.expansion_point != rhs.expansion_point
        {
            return Err(SeriesError::IncompatibleSeries);
        }
        let ramification = self.try_joint_ramification(rhs, settings)?;
        let left_ratio = ramification / self.ramification;
        let right_ratio = ramification / rhs.ramification;
        let shift = Self::signed(
            &(Integer::from(self.shift) * Integer::from(left_ratio)
                + Integer::from(rhs.shift) * Integer::from(right_ratio)),
            "multiplication shift",
        )?;
        let order = Self::rescaled_order(self.order, left_ratio)?
            .min(Self::rescaled_order(rhs.order, right_ratio)?);
        let mut result = Self {
            coefficients: Vec::new(),
            field: self.field.clone(),
            variable: self.variable.clone(),
            expansion_point: self.expansion_point.clone(),
            shift,
            order,
            ramification,
        };
        result.check_geometry(settings)?;
        self.resize(
            &mut result.coefficients,
            self.coefficients.len().max(rhs.coefficients.len()),
            settings,
        )?;
        for (left_position, left) in self.coefficients.iter().enumerate() {
            if self.field.is_zero(left) {
                continue;
            }
            let left_exponent = self.get_exponent(left_position);
            for (right_position, right) in rhs.coefficients.iter().enumerate() {
                if self.field.is_zero(right) {
                    continue;
                }
                let index = result.try_index(&left_exponent + &rhs.get_exponent(right_position))?;
                if index < result.order {
                    if index >= result.coefficients.len() {
                        let length = index
                            .checked_mul(2)
                            .and_then(|length| {
                                index.checked_add(1).map(|minimum| length.max(minimum))
                            })
                            .ok_or(SeriesError::IndexOverflow {
                                operation: "multiplication coefficient growth",
                            })?;
                        self.resize(&mut result.coefficients, length, settings)?;
                    }
                    self.field
                        .add_mul_assign(&mut result.coefficients[index], left, right);
                }
            }
        }
        result.truncate();
        result.check_geometry(settings)?;
        Ok(result)
    }
}

impl<F: Ring> Add for Series<F> {
    type Output = Self;
    fn add(self, other: Self) -> Self::Output {
        self.try_add_owned(other, &SeriesSettings::default())
            .expect("invalid series addition")
    }
}

impl<'a, F: Ring> Add<&'a Series<F>> for &Series<F> {
    type Output = Series<F>;

    fn add(self, other: &'a Series<F>) -> Self::Output {
        (self.clone()).add(other.clone())
    }
}

impl<F: Ring> Sub for Series<F> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self.add(other.neg())
    }
}

impl<'a, F: Ring> Sub<&'a Series<F>> for &Series<F> {
    type Output = Series<F>;

    fn sub(self, other: &'a Series<F>) -> Self::Output {
        (self.clone()).add(other.clone().neg())
    }
}

impl<F: Ring> Neg for Series<F> {
    type Output = Self;
    fn neg(mut self) -> Self::Output {
        // Negate coefficients of all terms.
        for c in &mut self.coefficients {
            *c = self.field.neg(&*c);
        }
        self
    }
}

impl<'a, F: Ring> Mul<&'a Series<F>> for &Series<F> {
    type Output = Series<F>;
    fn mul(self, rhs: &'a Series<F>) -> Self::Output {
        self.try_mul(rhs, &SeriesSettings::default())
            .expect("invalid series multiplication")
    }
}

impl<'a, F: Ring> Mul<&'a Series<F>> for Series<F> {
    type Output = Series<F>;

    #[inline]
    fn mul(self, rhs: &'a Series<F>) -> Self::Output {
        (&self) * rhs
    }
}

impl<'a> Div<&'a Series<AtomField>> for &Series<AtomField> {
    type Output = Series<AtomField>;

    fn div(self, other: &'a Series<AtomField>) -> Self::Output {
        other.rpow((-1, 1).into()).unwrap() * self
    }
}

impl<'a> Div<&'a Series<AtomField>> for Series<AtomField> {
    type Output = Series<AtomField>;

    fn div(self, other: &'a Series<AtomField>) -> Self::Output {
        (&self).div(other)
    }
}

impl<F: EuclideanDomain> Series<F> {
    /// Get the content from the coefficients.
    pub fn content(&self) -> F::Element {
        if self.coefficients.is_empty() {
            return self.field.zero();
        }
        let mut c = self.coefficients.first().unwrap().clone();
        for cc in self.coefficients.iter().skip(1) {
            // early return if possible (not possible for rationals)
            if F::one_is_gcd_unit() && self.field.is_one(&c) {
                break;
            }

            c = self.field.gcd(&c, cc);
        }
        c
    }

    /// Divide every coefficient with `other`.
    pub fn div_coeff(mut self, other: &F::Element) -> Self {
        for c in &mut self.coefficients {
            let (quot, rem) = self.field.quot_rem(c, other);
            debug_assert!(self.field.is_zero(&rem));
            *c = quot;
        }
        self
    }

    /// Make the series primitive by removing the content.
    pub fn make_primitive(self) -> Self {
        let c = self.content();
        self.div_coeff(&c)
    }
}

impl Series<AtomField> {
    fn regular_zero_remainder(
        &self,
        function: Symbol,
        settings: &SeriesSettings,
    ) -> Result<Option<Self>, SeriesError> {
        if !self.is_zero() {
            return Ok(None);
        }
        if function == Symbol::LOG {
            return Err(SeriesError::MissingLogCoefficient);
        }
        if self.order == i64::MAX as usize {
            return Err(SeriesError::IndexOverflow {
                operation: "finite zero-remainder precision",
            });
        }
        let precision = self.absolute_order();
        if precision <= 0 {
            return Err(SeriesError::MissingFunctionCoefficient { function });
        }
        if function == Symbol::SIN {
            return Self::try_new(
                &self.field,
                None,
                self.variable.clone(),
                self.expansion_point.clone(),
                precision,
                settings,
            )
            .map(Some);
        }
        let precision = if function == Symbol::COS {
            &precision * &Rational::from(2)
        } else {
            precision
        };
        Self::try_new(
            &self.field,
            Some(1),
            self.variable.clone(),
            self.expansion_point.clone(),
            precision,
            settings,
        )?
        .try_constant(self.field.one(), settings)
        .map(Some)
    }

    /// Remove coefficients at the lowest powers that the zero test proves zero.
    pub(crate) fn normalize_trailing_coefficient(&mut self) {
        let zeros = self
            .coefficients
            .iter()
            .take_while(|c| c.check_zero().is_true())
            .count();
        if zeros == 0 {
            return;
        }

        if zeros == self.coefficients.len() {
            self.coefficients.clear();
            self.shift += self.order as isize;
            self.order = 0;
        } else {
            self.coefficients.drain(..zeros);
            self.shift += zeros as isize;
            self.order -= zeros;
        }
    }

    /// Extract powers of `x` from an expression that comes from simplifying an exponential with logs
    /// i.e.: `exp(c + 3 log(x^5)) = exp(c)*x^15`.
    fn extract_exp_log(
        &self,
        e: AtomView,
        s: AtomView,
        settings: &SeriesSettings,
    ) -> Result<Self, SeriesError> {
        if !e.contains(s) {
            return self.try_constant(e.to_owned(), settings);
        }
        match e {
            AtomView::Pow(power) => {
                let (base, exponent) = power.get_base_exp();
                if base != s {
                    return Err(SeriesError::UnexpectedExpLogSimplification {
                        expression: e.to_owned(),
                    });
                }
                if let AtomView::Num(number) = exponent {
                    if let CoefficientView::Natural(n, d, imaginary, _) = number.get_coeff_view() {
                        if imaginary == 0 {
                            self.try_monomial(self.field.one(), (n, d).into(), settings)
                        } else {
                            Err(SeriesError::UnsupportedComplexExponent {
                                expression: e.to_owned(),
                            })
                        }
                    } else {
                        Err(SeriesError::UnsupportedLargeExponent {
                            expression: e.to_owned(),
                        })
                    }
                } else {
                    Err(SeriesError::NonRationalPower {
                        expression: e.to_owned(),
                    })
                }
            }
            AtomView::Var(_) => self.try_monomial(self.field.one(), 1.into(), settings),
            AtomView::Mul(product) => {
                let mut result = self.try_one_inf_prec(settings)?;
                for factor in product {
                    result =
                        result.try_mul(&self.extract_exp_log(factor, s, settings)?, settings)?;
                }
                Ok(result)
            }
            _ => Err(SeriesError::UnexpectedExpLogSimplification {
                expression: e.to_owned(),
            }),
        }
    }

    fn factorial_order(order: usize) -> Result<u32, SeriesError> {
        u32::try_from(order).map_err(|_| SeriesError::IndexOverflow {
            operation: "factorial Taylor order",
        })
    }

    /// Expand the exponential to available precision using checked native indices.
    pub fn exp(&self) -> Result<Self, SeriesError> {
        self.exp_with_settings(&SeriesSettings::default())
    }

    pub fn exp_with_settings(&self, settings: &SeriesSettings) -> Result<Self, SeriesError> {
        self.check_geometry(settings)?;
        if let Some(result) = self.regular_zero_remainder(Symbol::EXP, settings)? {
            return Ok(result);
        }
        if self.shift < 0 {
            return Err(SeriesError::EssentialSingularity {
                function: Symbol::EXP,
            });
        }
        if self.order == 0 {
            return self.try_one_inf_prec(settings)?.try_add(
                &Self::try_new(
                    &self.field,
                    Some(1),
                    self.variable.clone(),
                    self.expansion_point.clone(),
                    (self.shift as i64, 1).into(),
                    settings,
                )?,
                settings,
            );
        }
        let c = if self.shift == 0 {
            self.coefficients[0].clone()
        } else {
            Atom::new()
        };
        let e = FunctionBuilder::new(Symbol::EXP).add_arg(&c).finish();
        let variable = self.variable.to_atom() - &self.expansion_point;
        let shift_series = self.extract_exp_log(e.as_view(), variable.as_view(), settings)?;
        let p = self.try_clone(settings)?.remove_constant();
        let mut result = self.try_one_inf_prec(settings)?;
        let mut power = p.try_clone(settings)?;
        for i in 1..=Self::factorial_order(self.order)? {
            let term = power
                .try_clone(settings)?
                .div_coeff(&Atom::num(Integer::factorial(i)));
            power = power.try_mul(&p, settings)?;
            result = result.try_add(&term, settings)?;
        }
        result.try_mul(&shift_series, settings)
    }

    /// Expand the logarithm through the available native precision.
    pub fn log(&self) -> Result<Self, SeriesError> {
        self.log_with_settings(&SeriesSettings::default())
    }

    pub fn log_with_settings(&self, settings: &SeriesSettings) -> Result<Self, SeriesError> {
        self.check_geometry(settings)?;
        if let Some(result) = self.regular_zero_remainder(Symbol::LOG, settings)? {
            return Ok(result);
        }
        if self.order == 0 {
            return Err(SeriesError::MissingLogCoefficient);
        }
        let c = (self.variable.to_atom() - &self.expansion_point).pow(self.get_exponent(0))
            * &self.coefficients[0];
        let opposite = self.shift.checked_neg().ok_or(SeriesError::IndexOverflow {
            operation: "logarithm shift",
        })?;
        let normalized = self
            .try_clone(settings)?
            .div_coeff(&self.coefficients[0])
            .try_mul_exp_units(opposite, settings)?;
        let one = self.try_constant(self.field.one(), settings)?;
        let p = normalized.try_add(&(-one), settings)?;
        let mut result = self.try_constant(
            FunctionBuilder::new(Symbol::LOG).add_arg(&c).finish(),
            settings,
        )?;
        let mut power = p.try_clone(settings)?;
        for i in 1..=self.order {
            let mut term = power.try_clone(settings)?.div_coeff(&Atom::num(i as i64));
            power = power.try_mul(&p, settings)?;
            if i % 2 == 0 {
                term = -term;
            }
            result = result.try_add(&term, settings)?;
        }
        Ok(result)
    }

    /// Expand the sine using checked native coefficient containers.
    pub fn sin(&self) -> Result<Self, SeriesError> {
        self.sin_with_settings(&SeriesSettings::default())
    }

    pub fn sin_with_settings(&self, settings: &SeriesSettings) -> Result<Self, SeriesError> {
        self.check_geometry(settings)?;
        if let Some(result) = self.regular_zero_remainder(Symbol::SIN, settings)? {
            return Ok(result);
        }
        if self.shift < 0 {
            return Err(SeriesError::EssentialSingularity {
                function: Symbol::SIN,
            });
        }
        if self.order == 0 {
            return Self::try_new(
                &self.field,
                Some(1),
                self.variable.clone(),
                self.expansion_point.clone(),
                (self.shift as i64, 1).into(),
                settings,
            );
        }
        let c = if self.shift == 0 {
            self.coefficients[0].clone()
        } else {
            Atom::new()
        };
        if c.contains(self.variable.to_atom()) {
            return Err(SeriesError::ConstantTermDependsOnExpansionVariable {
                function: Symbol::SIN,
                constant: c,
                variable: self.variable.as_ref().clone(),
            });
        }
        let p = self.try_clone(settings)?.remove_constant();
        let mut result = self.try_constant(
            FunctionBuilder::new(Symbol::SIN).add_arg(&c).finish(),
            settings,
        )?;
        let mut power = p.try_clone(settings)?;
        for i in 1..=Self::factorial_order(self.order)? {
            let mut coefficient =
                FunctionBuilder::new(if i % 2 == 1 { Symbol::COS } else { Symbol::SIN })
                    .add_arg(&c)
                    .finish();
            if i % 4 >= 2 {
                coefficient = coefficient.neg();
            }
            let term = power
                .try_clone(settings)?
                .mul_coeff(&coefficient)
                .div_coeff(&Atom::num(Integer::factorial(i)));
            power = power.try_mul(&p, settings)?;
            result = result.try_add(&term, settings)?;
        }
        Ok(result)
    }

    /// Expand the cosine using checked native coefficient containers.
    pub fn cos(&self) -> Result<Self, SeriesError> {
        self.cos_with_settings(&SeriesSettings::default())
    }

    pub fn cos_with_settings(&self, settings: &SeriesSettings) -> Result<Self, SeriesError> {
        self.check_geometry(settings)?;
        if let Some(result) = self.regular_zero_remainder(Symbol::COS, settings)? {
            return Ok(result);
        }
        if self.shift < 0 {
            return Err(SeriesError::EssentialSingularity {
                function: Symbol::COS,
            });
        }
        if self.order == 0 {
            let precision = Integer::from(self.shift) * Integer::from(2);
            return self.try_one_inf_prec(settings)?.try_add(
                &Self::try_new(
                    &self.field,
                    Some(1),
                    self.variable.clone(),
                    self.expansion_point.clone(),
                    precision.into(),
                    settings,
                )?,
                settings,
            );
        }
        let c = if self.shift == 0 {
            self.coefficients[0].clone()
        } else {
            Atom::new()
        };
        if c.contains(self.variable.to_atom()) {
            return Err(SeriesError::ConstantTermDependsOnExpansionVariable {
                function: Symbol::COS,
                constant: c,
                variable: self.variable.as_ref().clone(),
            });
        }
        let p = self.try_clone(settings)?.remove_constant();
        let mut result = self.try_constant(
            FunctionBuilder::new(Symbol::COS).add_arg(&c).finish(),
            settings,
        )?;
        let mut power = p.try_clone(settings)?;
        for i in 1..=Self::factorial_order(self.order)? {
            let mut coefficient =
                FunctionBuilder::new(if i % 2 == 1 { Symbol::SIN } else { Symbol::COS })
                    .add_arg(&c)
                    .finish();
            if i % 2 == 0 {
                coefficient = -coefficient;
            }
            if i % 4 < 2 {
                coefficient = coefficient.neg();
            }
            let term = power
                .try_clone(settings)?
                .mul_coeff(&coefficient)
                .div_coeff(&Atom::num(Integer::factorial(i)));
            power = power.try_mul(&p, settings)?;
            result = result.try_add(&term, settings)?;
        }
        Ok(result)
    }

    /// Raise to a series power using the native logarithm/exponential route.
    pub fn pow(&self, power: &Self) -> Result<Self, SeriesError> {
        self.pow_with_settings(power, &SeriesSettings::default())
    }

    pub fn pow_with_settings(
        &self,
        power: &Self,
        settings: &SeriesSettings,
    ) -> Result<Self, SeriesError> {
        self.log_with_settings(settings)?
            .try_mul(power, settings)?
            .exp_with_settings(settings)
    }

    /// Divide with checked native reciprocal and convolution operations.
    pub fn try_div(&self, rhs: &Self, settings: &SeriesSettings) -> Result<Self, SeriesError> {
        self.try_mul(&rhs.rpow_with_settings((-1, 1).into(), settings)?, settings)
    }

    /// Raise to an exact rational power using checked native indices.
    pub fn rpow(&self, power: Rational) -> Result<Self, SeriesError> {
        self.rpow_with_settings(power, &SeriesSettings::default())
    }

    pub fn rpow_with_settings(
        &self,
        power: Rational,
        settings: &SeriesSettings,
    ) -> Result<Self, SeriesError> {
        self.check_geometry(settings)?;
        if power.is_zero() {
            return Err(SeriesError::ZeroPowerRequiresInfinitePrecision);
        }
        if power.is_integer() && !power.is_negative() {
            return self.try_npow(
                Self::unsigned(power.numerator_ref(), "integer series power")?,
                settings,
            );
        }
        let mut base = self.try_clone(settings)?;
        base.normalize_trailing_coefficient();
        if base.is_zero() {
            if power.is_negative() {
                return Err(SeriesError::DivisionByZero);
            }
            base.shift = Self::signed(
                &(Integer::from(base.shift) * power.numerator_ref()),
                "zero power shift",
            )?;
            base.order = Self::unsigned(
                &(Integer::from(base.order) * power.numerator_ref()),
                "zero power precision",
            )?;
            base.ramification = Self::unsigned(
                &(Integer::from(base.ramification) * power.denominator_ref()),
                "zero power ramification",
            )?;
            if base.order == i64::MAX as usize {
                return Err(SeriesError::IndexOverflow {
                    operation: "zero power precision",
                });
            }
            base.check_geometry(settings)?;
            return Ok(base);
        }
        let c = base.coefficients[0].clone();
        let order = base.order;
        let shift = base.shift;
        let leading =
            Rational::from((Integer::from(shift), Integer::from(base.ramification))) * &power;
        let target_ramification = Self::unsigned(
            &Integer::from(base.ramification).lcm(leading.denominator_ref()),
            "rational power ramification",
        )?;
        settings.ramification(target_ramification)?;
        let target_shift = Self::signed(
            &(leading.numerator_ref()
                * (Integer::from(target_ramification) / leading.denominator_ref())),
            "rational power shift",
        )?;
        let mut result = base.try_constant(self.field.one(), settings)?;
        let mut term_power = result.try_clone(settings)?;
        let opposite = shift.checked_neg().ok_or(SeriesError::IndexOverflow {
            operation: "rational power normalization shift",
        })?;
        let normalized = base.div_coeff(&c).try_mul_exp_units(opposite, settings)?;
        let delta = normalized.try_add(
            &(-result.try_constant(self.field.one(), settings)?),
            settings,
        )?;
        let mut numerator = Rational::one();
        for i in 1..=Self::factorial_order(order)? {
            numerator = &numerator * &(&power - &Rational::from(i - 1));
            term_power = term_power.try_mul(&delta, settings)?;
            let term = term_power
                .try_clone(settings)?
                .mul_coeff(&Atom::num(numerator.clone()))
                .div_coeff(&Atom::num(Integer::factorial(i)));
            result = result.try_add(&term, settings)?;
        }
        result.try_change_ramification(target_ramification, settings)?;
        result
            .mul_coeff(&c.pow(power))
            .try_mul_exp_units(target_shift, settings)
    }

    /// Return the stored terms as an expression, discarding the unknown remainder.
    pub fn to_atom(&self) -> Atom {
        let mut a = Atom::new();
        self.to_atom_into(&mut a);
        a
    }

    /// Write the stored terms as an expression into `out`, replacing its previous
    /// contents and discarding the unknown remainder.
    pub fn to_atom_into(&self, out: &mut Atom) {
        out.to_num(0);

        if self.order == 0 {
            return;
        }

        let v = self.variable.to_atom() - &self.expansion_point;
        for (e, c) in self.coefficients.iter().enumerate() {
            let p = self.get_exponent(e);
            if !c.is_zero() {
                *out = &*out + &(v.pow(p) * c);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        atom::{Atom, AtomCore},
        domains::rational::{Q, Rational},
        parse,
        poly::series::{Series, SeriesDepth},
        symbol,
    };
    use std::sync::Arc;

    #[test]
    fn absolute_truncation_preserves_finite_zero_precision() {
        let x = symbol!("absolute_zero_precision_x");
        for known_order in [Rational::from(6), (3, 2).into()] {
            let prototype =
                Series::new(&Q, None, Arc::new(x.into()), 0.into(), known_order.clone());
            for requested in [
                known_order.clone(),
                (1, 3).into(),
                0.into(),
                (-2).into(),
                7.into(),
            ] {
                let mut zero = prototype.zero();
                zero.truncate_absolute_order(requested.clone());
                assert_eq!(zero.absolute_order(), known_order);
                assert_eq!(
                    zero.coefficient(known_order.clone() - Rational::from(1)),
                    Some(0.into())
                );
                assert_eq!(zero.coefficient(known_order.clone()), None);
                zero.truncate_absolute_order(requested);
                assert_eq!(zero.absolute_order(), known_order);
            }
        }
    }

    #[test]
    fn absolute_truncation_keeps_leading_zero_information() {
        let x = symbol!("absolute_last_term_x");
        let prototype = Series::new(&Q, None, Arc::new(x.into()), 0.into(), 6.into());
        for (exponent, order) in [
            (Rational::from(4), Rational::from(2)),
            ((-1).into(), (-2).into()),
            ((1, 2).into(), (1, 3).into()),
        ] {
            let mut series = prototype.monomial(1.into(), exponent.clone());
            series.truncate_absolute_order(order);
            assert!(series.is_zero());
            assert_eq!(series.absolute_order(), exponent);
            assert_eq!(
                series.coefficient(exponent.clone() - Rational::from(1)),
                Some(0.into())
            );
            assert_eq!(series.coefficient(exponent), None);
        }
    }

    #[test]
    fn absolute_truncation_after_cancellation_preserves_known_zeros() {
        let x = symbol!("absolute_cancelled_x");
        let series = parse!("absolute_cancelled_x^2").series(x, 0, 6).unwrap();
        let mut zero = &series - &series;
        let known_order = zero.absolute_order();
        zero.truncate_absolute_order(3.into());
        assert_eq!(zero.absolute_order(), known_order);
        assert_eq!(zero.coefficient(6.into()), Some(Atom::Zero));
        assert_eq!(zero.coefficient(known_order), None);

        let mut nonzero = parse!("1+absolute_cancelled_x^4").series(x, 0, 6).unwrap();
        nonzero.truncate_absolute_order(3.into());
        assert_eq!(nonzero.coefficient(0.into()), Some(Atom::num(1)));
        assert_eq!(nonzero.coefficient(2.into()), Some(Atom::Zero));
        assert_eq!(nonzero.coefficient(3.into()), None);
    }

    #[test]
    fn coefficient_distinguishes_unknown_from_zero() {
        let x = symbol!("coefficient_range_x");
        let s = parse!("exp(coefficient_range_x)").series(x, 0, 2).unwrap();
        assert_eq!(s.coefficient(2.into()), Some(Atom::num((1, 2))));
        assert_eq!(s.coefficient(3.into()), None);
        assert_eq!(s.coefficient((7, 2).into()), None);
        assert_eq!(s.coefficient((-1).into()), Some(Atom::Zero));
        assert_eq!(s.coefficient((1, 2).into()), Some(Atom::Zero));
        let huge: crate::domains::rational::Rational = parse!("1000000000000000000000000000000000")
            .try_into()
            .unwrap();
        assert_eq!(s.coefficient(huge.clone()), None);
        assert_eq!(s.coefficient(-huge), Some(Atom::Zero));

        let sparse = parse!("coefficient_range_x^2").series(x, 0, 4).unwrap();
        assert_eq!(sparse.coefficient(0.into()), Some(Atom::Zero));
        assert_eq!(sparse.coefficient(4.into()), Some(Atom::Zero));
        assert_eq!(sparse.coefficient(5.into()), None);
        let zero = &sparse - &sparse;
        assert_eq!(zero.coefficient(4.into()), Some(Atom::Zero));
        assert_eq!(zero.coefficient(5.into()), None);
    }

    #[test]
    fn coefficient_handles_laurent_and_puiseux_remainders() {
        let x = symbol!("coefficient_fraction_x");
        let s = parse!("1/coefficient_fraction_x+sqrt(coefficient_fraction_x)")
            .series(x, 0, SeriesDepth::absolute((3, 2)))
            .unwrap();
        assert_eq!(s.coefficient((-1).into()), Some(Atom::num(1)));
        assert_eq!(s.coefficient((1, 2).into()), Some(Atom::num(1)));
        assert_eq!(s.coefficient((1, 3).into()), Some(Atom::Zero));
        assert_eq!(s.coefficient((3, 2).into()), Some(Atom::Zero));
        assert_eq!(s.coefficient(s.absolute_order()), None);
        let pole = parse!("1/coefficient_fraction_x^3")
            .series(x, 0, -2)
            .unwrap();
        assert_eq!(pole.coefficient((-3).into()), Some(Atom::num(1)));
        assert_eq!(pole.coefficient((-2).into()), Some(Atom::Zero));
        assert_eq!(pole.coefficient(0.into()), None);
    }

    #[test]
    fn map_coeff() {
        let a = parse!("((v1+1)^2 - (v1^2 + 2v1 + 1))/v2")
            .series(symbol!("v2"), Atom::Zero, SeriesDepth::absolute(0))
            .unwrap();
        assert_eq!(a.get_trailing_exponent(), -1);

        let b = a.map_coeff(|x| x.expand());
        assert_eq!(b.get_trailing_exponent(), 1);
    }

    #[test]
    fn reciprocal_retries_after_exact_coefficient_cancellation() {
        let x = parse!("series_cancellation_x");
        for text in [
            "(a+1)^2-a^2-2*a-1",
            "1/a-1/(a+1)-1/(a*(a+1))",
            "(a+1i*b)^2-a^2-2i*a*b+b^2",
            "sin(a)^2+cos(a)^2-1",
            "sqrt(2)*sqrt(3)-sqrt(6)",
        ] {
            let c = parse!(text);
            assert!(!c.is_zero(), "{text}");
            assert!(c.check_zero().is_true(), "{text}");
            for power in [1, 2, 4, 10] {
                // Start with the original 1/(c+x) regression, then cancel two
                // coefficients and force deeper expansion to find a nonzero one.
                let zeros = if power == 1 { c.clone() } else { &c * (&x + 1) };
                let input = (zeros + x.pow(power)).pow(-1);
                let result = input
                    .series(
                        symbol!("series_cancellation_x"),
                        0,
                        SeriesDepth::absolute(1),
                    )
                    .unwrap();
                assert_eq!(result.to_atom(), x.pow(-power), "{input}");
                assert_eq!(result.get_trailing_exponent(), -power, "{input}");
            }
        }
    }

    #[test]
    fn rational_powers_remove_exact_zero_leading_coefficients() {
        let x = symbol!("series_rpow_cancellation_x");
        let c = parse!("1/a-1/(a+1)-1/(a*(a+1))");
        let vanishing = &c * parse!("1+sqrt(series_rpow_cancellation_x)");
        for text in [
            "series_rpow_cancellation_x^2*(1+series_rpow_cancellation_x)",
            "series_rpow_cancellation_x^(3/2)*(1+sqrt(series_rpow_cancellation_x))",
        ] {
            let clean = parse!(text).series(x, 0, SeriesDepth::absolute(4)).unwrap();
            let base = (parse!(text) + &vanishing)
                .series(x, 0, SeriesDepth::absolute(4))
                .unwrap();
            for power in [(-1, 1), (1, 2), (-1, 2)] {
                let actual = base.rpow(power.into()).unwrap();
                let expected = clean.rpow(power.into()).unwrap();
                assert_eq!(actual.to_atom(), expected.to_atom(), "{text}, {power:?}");
                assert_eq!(actual.absolute_order(), expected.absolute_order());
            }
        }

        let zero = vanishing.series(x, 0, SeriesDepth::absolute(4)).unwrap();
        assert!(matches!(
            zero.rpow((-1, 1).into()),
            Err(super::SeriesError::DivisionByZero)
        ));
        let root = zero.rpow((1, 2).into()).unwrap();
        assert!(root.is_zero());
        assert_eq!(
            root.absolute_order(),
            zero.absolute_order() / super::Rational::from(2)
        );
    }

    #[test]
    fn coefficient_cleanup_preserves_nonzero_and_unresolved_terms() {
        for text in ["(a+1)^2-a^2-2*a-1+1/10^100", "log(2)+log(3)-log(6)"] {
            let coefficient = parse!(text);
            assert!(!coefficient.together().is_zero());
            let mut series = (coefficient.clone() + parse!("series_cleanup_x"))
                .series(symbol!("series_cleanup_x"), 0, SeriesDepth::absolute(2))
                .unwrap();
            let original = series.to_atom();
            let order = series.absolute_order();
            series.normalize_trailing_coefficient();
            assert_eq!(series.get_trailing_coefficient(), coefficient);
            assert_eq!(series.to_atom(), original);
            assert_eq!(series.absolute_order(), order);
        }
    }
}

#[cfg(test)]
mod puiseux_tests {
    use crate::{
        atom::{Atom, AtomCore},
        parse,
        poly::series::SeriesDepth,
        symbol,
    };

    #[test]
    fn fractional_product_and_laurent_series_have_correct_coefficients() {
        let x = symbol!("puiseux_precision::x");
        let s = parse!("puiseux_precision::x^(1/3)*(1+puiseux_precision::x)^(1/2)")
            .series(x, 0, SeriesDepth::absolute((7, 3)))
            .unwrap();
        for (exponent, coefficient) in [((1, 3), (1, 1)), ((4, 3), (1, 2)), ((7, 3), (-1, 8))] {
            assert_eq!(s.coefficient(exponent.into()), Some(Atom::num(coefficient)));
        }
        let square = (&s * &s).map_coeff(|a| a.expand());
        assert_eq!(square.coefficient((2, 3).into()), Some(Atom::num(1)));
        assert_eq!(square.coefficient((5, 3).into()), Some(Atom::num(1)));
        assert_eq!(square.coefficient((8, 3).into()), Some(Atom::Zero));
        let pole = parse!("puiseux_precision::x^(-1/3)*(1+puiseux_precision::x)")
            .series(x, 0, SeriesDepth::absolute((2, 3)))
            .unwrap();
        assert_eq!(pole.coefficient((-1, 3).into()), Some(Atom::num(1)));
        assert_eq!(pole.coefficient((2, 3).into()), Some(Atom::num(1)));
    }

    #[test]
    fn fractional_depth_at_nonzero_centre_and_integer_control() {
        let x = symbol!("puiseux_precision::x");
        for depth in [SeriesDepth::absolute((7, 3)), SeriesDepth::absolute(3)] {
            let s = parse!("puiseux_precision::x^2")
                .series(x, 2, depth)
                .unwrap();
            assert_eq!(s.coefficient(0.into()), Some(Atom::num(4)));
            assert_eq!(s.coefficient(1.into()), Some(Atom::num(4)));
            assert_eq!(s.coefficient(2.into()), Some(Atom::num(1)));
            assert_eq!(s.coefficient((1, 3).into()), Some(Atom::Zero));
        }
    }
}
