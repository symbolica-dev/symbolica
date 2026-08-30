//! Compute the greatest common divisor (GCD) of multivariate polynomials with coefficients that implement [PolynomialGCD].

use ahash::{HashMap, HashSet, HashSetExt};
use rand;
use smallvec::{SmallVec, smallvec};
use std::borrow::Cow;
use std::cmp::{Ordering, max, min};
use std::mem;
use std::ops::{Add, RangeInclusive};
use tracing::{debug, instrument};

use crate::domains::algebraic::{AlgebraicExtension, GaloisField};
use crate::domains::finite_field::{
    FiniteField, FiniteFieldCore, FiniteFieldElement, FiniteFieldWorkspace, PrimeIteratorU64,
    SMOOTH_PRIME_BASE, SMOOTH_PRIMES, ToFiniteField, Zp, Zp64, Zp64DiscreteLogContext,
};
use crate::domains::float::{FloatField, SingleFloat};
use crate::domains::integer::{
    FromFiniteField, Integer, IntegerRing, MultiPrecisionInteger, SMALL_PRIMES, Z,
};
use crate::domains::rational::{Q, Rational, RationalField};
use crate::domains::{
    EuclideanDomain, Field, InternalOrdering, Ring, RingOps, SampleableRing, Set,
};
use crate::kernels::GeometricSequenceStepRequest;
use crate::poly::INLINED_EXPONENTS;
use crate::tensors::matrix::{Matrix, MatrixError};
use crate::{GLOBAL_SETTINGS, warn};

use super::PositiveExponent;
use super::polynomial::{
    IntegerPolynomialCrtContext, LastVariableEvaluationContext, LastVariablePowerWorkspace,
    MultivariatePolynomial, WordCrt,
};
use super::univariate::DenseFiniteFieldRootContext;

#[cfg(feature = "binary_size")]
type ModularGcdFieldWorkspace = u64;
#[cfg(not(feature = "binary_size"))]
type ModularGcdFieldWorkspace = u32;
type ModularGcdField = FiniteField<ModularGcdFieldWorkspace>;

/// The maximum power of a variable that is cached
pub(crate) const POW_CACHE_SIZE: usize = 1000;
pub(crate) const INITIAL_POW_MAP_SIZE: usize = 1000;

/// Largest univariate image degree for which all GCD variable bounds are sampled together.
/// Larger images use the sparse per-variable sampler to avoid allocating dense coefficient rows.
const FUSED_GCD_BOUND_MAX_DEGREE: usize = 9999;

/// Largest dense coefficient buffer used for a sampled univariate GCD.
const DENSE_UNIVARIATE_GCD_MAX_COEFFICIENTS: usize = 4096;

/// Maximum coefficient-buffer length relative to the number of stored input terms.
const DENSE_UNIVARIATE_GCD_MAX_SPARSITY_RATIO: usize = 8;

/// Largest direct degree-to-shape table used during Zippel interpolation.
const ZIPPEL_SHAPE_INDEX_MAX_DEGREE_SPAN: usize = 4096;

/// Maximum direct-table length relative to the number of GCD shape coefficients.
const ZIPPEL_SHAPE_INDEX_MAX_SPARSITY_RATIO: usize = 32;

/// Maximum estimated coefficient size produced during recursive integer-heuristic substitution.
const HEURISTIC_GCD_MAX_EVALUATED_COEFFICIENT_BITS: u64 = 32 * 1024;

/// Estimated scalar images above this size use word-prime univariate GCDs. The estimate combines
/// coefficient height and degree, which are the two main costs of the Horner/GMP path.
const UNIVARIATE_MODULAR_GCD_MIN_EVALUATION_BITS: u64 = 8 * 1024;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum UnivariateIntegerGcdAlgorithm {
    Scalar,
    Modular,
}

/// Chooses between one large scalar image and a sequence of word-prime images.
fn select_univariate_integer_gcd(
    scalar_heuristic_allowed: bool,
    estimated_evaluation_bits: u64,
) -> UnivariateIntegerGcdAlgorithm {
    if !scalar_heuristic_allowed
        || estimated_evaluation_bits >= UNIVARIATE_MODULAR_GCD_MIN_EVALUATION_BITS
    {
        UnivariateIntegerGcdAlgorithm::Modular
    } else {
        UnivariateIntegerGcdAlgorithm::Scalar
    }
}

/// The upper bound of the range to be sampled during the computation of multiple gcds
pub(crate) const MAX_RNG_PREFACTOR: u32 = 50000;

/// Samples a nonzero field element while allowing every extension-basis
/// coefficient to range over the full prime field.
fn sample_nonzero_field_element<F>(ring: &F, rng: &mut impl rand::RngCore) -> F::Element
where
    F: Field + SampleableRing<SamplingPolicy = RangeInclusive<i64>>,
{
    let upper = match ring.characteristic().to_i64() {
        Some(characteristic) if characteristic > 0 => characteristic - 1,
        _ => MAX_RNG_PREFACTOR as i64 - 1,
    };
    let policy = 0..=upper;
    loop {
        let value = ring.sample(rng, &policy);
        if !ring.is_zero(&value) {
            return value;
        }
    }
}

/// Reuses evaluation points, power tables, and dense univariate images while sampling every
/// active variable bound of a polynomial pair.
struct GcdBoundSamplingContext<F: Field> {
    ring: F,
    sampled_variables: SmallVec<[usize; INLINED_EXPONENTS]>,
    retained_variables: SmallVec<[usize; INLINED_EXPONENTS]>,
    points: Vec<F::Element>,
    inverse_points: Vec<F::Element>,
    powers: Vec<Vec<F::Element>>,
    inverse_powers: Vec<Vec<F::Element>>,
    left_images: Vec<Vec<F::Element>>,
    right_images: Vec<Vec<F::Element>>,
}

impl<F: Field> GcdBoundSamplingContext<F> {
    /// Creates dense images for variables that occur in both input polynomials.
    fn new<E: PositiveExponent>(
        left: &MultivariatePolynomial<F, E>,
        right: &MultivariatePolynomial<F, E>,
        variables: &[usize],
    ) -> Option<Self> {
        let retained_variables = variables
            .iter()
            .copied()
            .filter(|variable| {
                left.degree(*variable) > E::zero() && right.degree(*variable) > E::zero()
            })
            .collect::<SmallVec<[_; INLINED_EXPONENTS]>>();

        if retained_variables.len() < 3 {
            return None;
        }

        let mut maximum_degrees = vec![0usize; left.nvars()];
        for variable in variables {
            let maximum_degree =
                left.degree(*variable).max(right.degree(*variable)).to_u32() as usize;
            if maximum_degree > FUSED_GCD_BOUND_MAX_DEGREE {
                return None;
            }
            maximum_degrees[*variable] = maximum_degree;
        }

        let ring = left.ring().clone();
        let points = vec![ring.one(); left.nvars()];
        let inverse_points = points.clone();
        let mut powers = (0..left.nvars()).map(|_| Vec::new()).collect::<Vec<_>>();
        let mut inverse_powers = powers.clone();
        for variable in variables {
            let cache_length = (maximum_degrees[*variable] + 1).min(POW_CACHE_SIZE);
            powers[*variable] = vec![ring.one(); cache_length];
            inverse_powers[*variable] = vec![ring.one(); cache_length];
        }

        let left_images = retained_variables
            .iter()
            .map(|variable| vec![ring.zero(); left.degree(*variable).to_u32() as usize + 1])
            .collect();
        let right_images = retained_variables
            .iter()
            .map(|variable| vec![ring.zero(); right.degree(*variable).to_u32() as usize + 1])
            .collect();

        Some(Self {
            ring,
            sampled_variables: variables.iter().copied().collect(),
            retained_variables,
            points,
            inverse_points,
            powers,
            inverse_powers,
            left_images,
            right_images,
        })
    }

    /// Sets one nonzero evaluation point and fills its direct and inverse power tables.
    fn set_point(&mut self, variable: usize, point: F::Element) {
        debug_assert!(!self.ring.is_zero(&point));
        let inverse_point = self.ring.inv(&point);
        self.points[variable] = point.clone();
        self.inverse_points[variable] = inverse_point.clone();

        let mut power = self.ring.one();
        for cached_power in &mut self.powers[variable] {
            *cached_power = power.clone();
            self.ring.mul_assign(&mut power, &point);
        }

        let mut inverse_power = self.ring.one();
        for cached_power in &mut self.inverse_powers[variable] {
            *cached_power = inverse_power.clone();
            self.ring.mul_assign(&mut inverse_power, &inverse_point);
        }
    }

    /// Samples one nonzero point per variable and prepares its power tables.
    fn sample_points(&mut self, rng: &mut impl rand::RngCore)
    where
        F: SampleableRing<SamplingPolicy = RangeInclusive<i64>>,
    {
        for index in 0..self.sampled_variables.len() {
            let variable = self.sampled_variables[index];
            let point = sample_nonzero_field_element(&self.ring, rng);
            self.set_point(variable, point);
        }
    }

    /// Builds every univariate image in one pass over each input polynomial.
    fn fill_images<E: PositiveExponent>(
        &mut self,
        polynomial: &MultivariatePolynomial<F, E>,
        left: bool,
    ) {
        let images = if left {
            &mut self.left_images
        } else {
            &mut self.right_images
        };
        for image in &mut *images {
            image.fill(self.ring.zero());
        }

        for term in polynomial {
            let mut full_evaluation = term.coefficient.clone();
            for variable in &self.sampled_variables {
                let exponent = term.exponents[*variable].to_u32() as usize;
                if exponent == 0 {
                    continue;
                }
                if let Some(power) = self.powers[*variable].get(exponent) {
                    self.ring.mul_assign(&mut full_evaluation, power);
                } else {
                    self.ring.mul_assign(
                        &mut full_evaluation,
                        &self.ring.pow(&self.points[*variable], exponent as u64),
                    );
                }
            }

            for (image, variable) in images.iter_mut().zip(&self.retained_variables) {
                let exponent = term.exponents[*variable].to_u32() as usize;
                let mut coefficient = full_evaluation.clone();
                if exponent > 0 {
                    if let Some(inverse_power) = self.inverse_powers[*variable].get(exponent) {
                        self.ring.mul_assign(&mut coefficient, inverse_power);
                    } else {
                        self.ring.mul_assign(
                            &mut coefficient,
                            &self
                                .ring
                                .pow(&self.inverse_points[*variable], exponent as u64),
                        );
                    }
                }
                self.ring.add_assign(&mut image[exponent], &coefficient);
            }
        }
    }

    /// Returns whether every sampled image retains the input degree of its free variable.
    fn degrees_are_preserved(&self) -> bool {
        self.left_images
            .iter()
            .chain(&self.right_images)
            .all(|image| {
                image
                    .last()
                    .is_some_and(|coefficient| !self.ring.is_zero(coefficient))
            })
    }

    /// Converts an ascending dense coefficient row into a univariate multivariate polynomial.
    fn image_polynomial<E: PositiveExponent>(
        ring: &F,
        template: &MultivariatePolynomial<F, E>,
        variable: usize,
        coefficients: Vec<F::Element>,
    ) -> MultivariatePolynomial<F, E> {
        let mut image = template.zero_with_capacity(coefficients.len());
        let mut exponents = vec![E::zero(); template.nvars()];
        for (degree, coefficient) in coefficients.into_iter().enumerate() {
            if !ring.is_zero(&coefficient) {
                exponents[variable] = E::from_u32(degree as u32);
                image.append_monomial_back(coefficient, &exponents);
            }
        }
        image
    }

    /// Computes the GCD degree of each pair of currently filled univariate images.
    fn bounds_from_images<E: PositiveExponent>(
        self,
        left: &MultivariatePolynomial<F, E>,
        right: &MultivariatePolynomial<F, E>,
    ) -> SmallVec<[E; INLINED_EXPONENTS]> {
        let mut bounds = (0..left.nvars())
            .map(|_| E::zero())
            .collect::<SmallVec<[_; INLINED_EXPONENTS]>>();
        for ((variable, left_coefficients), right_coefficients) in self
            .retained_variables
            .into_iter()
            .zip(self.left_images)
            .zip(self.right_images)
        {
            let left_image = Self::image_polynomial(&self.ring, left, variable, left_coefficients);
            let right_image =
                Self::image_polynomial(&self.ring, right, variable, right_coefficients);
            bounds[variable] = left_image.univariate_gcd(&right_image).ldegree_max();
        }
        bounds
    }

    /// Computes all sampled GCD degree bounds, or returns `None` when a shared good sample point
    /// cannot be found quickly enough for the coefficient field.
    fn sample_bounds<E: PositiveExponent>(
        mut self,
        left: &MultivariatePolynomial<F, E>,
        right: &MultivariatePolynomial<F, E>,
    ) -> Option<SmallVec<[E; INLINED_EXPONENTS]>>
    where
        F: SampleableRing<SamplingPolicy = RangeInclusive<i64>>,
    {
        let mut rng = rand::rng();
        let mut fail_count = 0;
        loop {
            self.sample_points(&mut rng);
            self.fill_images(left, true);
            self.fill_images(right, false);
            if self.degrees_are_preserved() {
                return Some(self.bounds_from_images(left, right));
            }

            if let Some(size) = self.ring.size()
                && fail_count * 2 > size
            {
                return None;
            }
            fail_count += 1;
        }
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
enum GCDError {
    BadOriginalImage,
    BadCurrentImage,
}

/// Per-variable exponent bounds for one input to a multivariate GCD operation.
///
/// The bounds identify the monomial shift of the input, its degree after that
/// shift is removed, and which variables remain in the shifted polynomial.
#[derive(Debug, Clone, PartialEq, Eq)]
struct GcdInputMetadata<E: PositiveExponent> {
    variables: SmallVec<[GcdVariableMetadata<E>; INLINED_EXPONENTS]>,
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
struct GcdVariableMetadata<E: PositiveExponent> {
    min_degree: E,
    max_degree: E,
}

impl<E: PositiveExponent> GcdInputMetadata<E> {
    fn scan<R: Ring>(polynomial: &MultivariatePolynomial<R, E>) -> Self {
        debug_assert!(!polynomial.is_zero());

        let mut variables: SmallVec<[GcdVariableMetadata<E>; INLINED_EXPONENTS]> = polynomial
            .exponents(0)
            .iter()
            .map(|exponent| GcdVariableMetadata {
                min_degree: *exponent,
                max_degree: *exponent,
            })
            .collect();
        for exponents in polynomial.exponents_iter().skip(1) {
            for (metadata, exponent) in variables.iter_mut().zip(exponents) {
                metadata.min_degree = metadata.min_degree.min(*exponent);
                metadata.max_degree = metadata.max_degree.max(*exponent);
            }
        }

        Self { variables }
    }

    #[inline]
    fn shifted_degree(&self, variable: usize) -> E {
        self.variables[variable].max_degree - self.variables[variable].min_degree
    }

    #[inline]
    fn occurs_after_shift(&self, variable: usize) -> bool {
        self.variables[variable].min_degree != self.variables[variable].max_degree
    }

    /// Removes the input's monomial factor from all of its terms.
    fn remove_monomial_shift<R: Ring>(
        &self,
        polynomial: &mut Cow<'_, MultivariatePolynomial<R, E>>,
    ) {
        if self
            .variables
            .iter()
            .all(|metadata| metadata.min_degree == E::zero())
        {
            return;
        }

        for exponents in polynomial.to_mut().exponents_iter_mut() {
            for (exponent, metadata) in exponents.iter_mut().zip(&self.variables) {
                *exponent = *exponent - metadata.min_degree;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum HuMonaganAnchor {
    Left,
    Right,
}

impl HuMonaganAnchor {
    /// Selects the smaller input whose cofactor is interpolated alongside the GCD.
    fn from_inputs<E: PositiveExponent>(
        left: &MultivariatePolynomial<IntegerRing, E>,
        right: &MultivariatePolynomial<IntegerRing, E>,
    ) -> Self {
        if left.nterms() <= right.nterms() {
            Self::Left
        } else {
            Self::Right
        }
    }

    /// Returns the anchored input first and the other GCD input second.
    fn order_inputs<'a, E: PositiveExponent>(
        self,
        left: &'a MultivariatePolynomial<IntegerRing, E>,
        right: &'a MultivariatePolynomial<IntegerRing, E>,
    ) -> (
        &'a MultivariatePolynomial<IntegerRing, E>,
        &'a MultivariatePolynomial<IntegerRing, E>,
    ) {
        match self {
            Self::Left => (left, right),
            Self::Right => (right, left),
        }
    }
}

/// Checks that Hu-Monagan has a main variable and at least two interpolation variables.
fn hu_monagan_has_minimum_geometry<E: PositiveExponent>(vars: &[usize], bounds: &[E]) -> bool {
    vars.len() >= 3
        && vars
            .first()
            .and_then(|variable| bounds.get(*variable))
            .is_some_and(|bound| *bound > E::zero())
        && bounds.get(vars[1]).is_some_and(|bound| *bound > E::zero())
        && bounds.get(vars[2]).is_some_and(|bound| *bound > E::zero())
}

/// Tests whether a variable order has sufficiently sparse interpolation geometry for Hu-Monagan.
fn hu_monagan_plan_is_applicable<E: PositiveExponent>(
    a: &MultivariatePolynomial<IntegerRing, E>,
    b: &MultivariatePolynomial<IntegerRing, E>,
    vars: &[usize],
    bounds: &[E],
    anchor: HuMonaganAnchor,
) -> bool {
    let (anchored_input, _) = anchor.order_inputs(a, b);
    hu_monagan_plan_is_applicable_with_degree(a, b, vars, bounds, |variable| {
        anchored_input.degree(variable)
    })
}

/// Tests Hu-Monagan interpolation geometry using degrees already known by GCD planning.
fn hu_monagan_plan_is_applicable_with_degrees<E: PositiveExponent>(
    a: &MultivariatePolynomial<IntegerRing, E>,
    b: &MultivariatePolynomial<IntegerRing, E>,
    vars: &[usize],
    bounds: &[E],
    anchored_degrees: &[E],
) -> bool {
    debug_assert_eq!(anchored_degrees.len(), a.nvars());
    debug_assert_eq!(a.nvars(), b.nvars());
    hu_monagan_plan_is_applicable_with_degree(a, b, vars, bounds, |variable| {
        anchored_degrees[variable]
    })
}

/// Tests Hu-Monagan interpolation geometry with a supplied anchored-input degree lookup.
fn hu_monagan_plan_is_applicable_with_degree<E, F>(
    a: &MultivariatePolynomial<IntegerRing, E>,
    b: &MultivariatePolynomial<IntegerRing, E>,
    vars: &[usize],
    bounds: &[E],
    mut anchored_degree: F,
) -> bool
where
    E: PositiveExponent,
    F: FnMut(usize) -> E,
{
    if !hu_monagan_has_minimum_geometry(vars, bounds) {
        return false;
    }

    let nterms = a.nterms() + b.nterms();
    const SPARSITY_MARGIN: u32 = 8;

    let mut box_size_u128 = Some(1u128);
    let mut cofactor_box_size_u128 = Some(1u128);
    for variable in vars.iter().copied().skip(1) {
        let bound = bounds[variable].to_u32();
        box_size_u128 = box_size_u128.and_then(|size| size.checked_mul(u128::from(bound) + 1));
        let cofactor_degree = anchored_degree(variable).to_u32().checked_sub(bound);
        cofactor_box_size_u128 = cofactor_box_size_u128
            .zip(cofactor_degree)
            .and_then(|(size, degree)| size.checked_mul(u128::from(degree) + 1));
    }
    if let (Some(box_size), Some(cofactor_box_size)) = (box_size_u128, cofactor_box_size_u128) {
        let largest_sparse_size = (box_size - 1) / u128::from(SPARSITY_MARGIN);
        return cofactor_box_size <= largest_sparse_size || nterms as u128 <= largest_sparse_size;
    }

    let mut box_size = Integer::from(1);
    let mut cofactor_box_size = Integer::from(1);
    for v in vars.iter().skip(1) {
        let bound = bounds[*v].to_u32();
        box_size *= bound + 1;
        cofactor_box_size *= Integer::from(anchored_degree(*v).to_u32()) - bound + 1;
    }

    cofactor_box_size * SPARSITY_MARGIN < box_size
        || Integer::from(nterms) * SPARSITY_MARGIN < box_size
}

fn should_use_hu_monagan_with_anchor<E: PositiveExponent>(
    a: &MultivariatePolynomial<IntegerRing, E>,
    b: &MultivariatePolynomial<IntegerRing, E>,
    vars: &[usize],
    bounds: &[E],
    anchor: HuMonaganAnchor,
) -> bool {
    vars.first() == Some(&0) && hu_monagan_plan_is_applicable(a, b, vars, bounds, anchor)
}

fn should_use_hu_monagan<E: PositiveExponent>(
    a: &MultivariatePolynomial<IntegerRing, E>,
    b: &MultivariatePolynomial<IntegerRing, E>,
    vars: &[usize],
    bounds: &[E],
) -> bool {
    should_use_hu_monagan_with_anchor(a, b, vars, bounds, HuMonaganAnchor::from_inputs(a, b))
}

/// Minimum row reduction that removes two levels from Hu's geometric sample schedule.
const HU_MONAGAN_MAIN_VARIABLE_ROW_REDUCTION: usize = 4;

/// Plans the main-variable change for one Hu-Monagan GCD operation.
struct HuMonaganPlanningContext<'a, E: PositiveExponent> {
    left: &'a MultivariatePolynomial<IntegerRing, E>,
    right: &'a MultivariatePolynomial<IntegerRing, E>,
    variables: &'a [usize],
    bounds: &'a [E],
    anchor: HuMonaganAnchor,
    anchored_degrees: SmallVec<[E; INLINED_EXPONENTS]>,
    other_degrees: SmallVec<[E; INLINED_EXPONENTS]>,
    maximum_row_supports: SmallVec<[usize; INLINED_EXPONENTS]>,
}

impl<'a, E: PositiveExponent> HuMonaganPlanningContext<'a, E> {
    /// Computes the input degrees and coefficient-row supports used by the plan.
    #[cfg(test)]
    fn new(
        left: &'a MultivariatePolynomial<IntegerRing, E>,
        right: &'a MultivariatePolynomial<IntegerRing, E>,
        variables: &'a [usize],
        bounds: &'a [E],
        anchor: HuMonaganAnchor,
    ) -> Self {
        let left_degrees = polynomial_degrees(left);
        let right_degrees = polynomial_degrees(right);
        Self::new_with_degrees(
            left,
            right,
            variables,
            bounds,
            anchor,
            &left_degrees,
            &right_degrees,
        )
    }

    /// Computes coefficient-row supports using degrees from the generic GCD input scan.
    fn new_with_degrees(
        left: &'a MultivariatePolynomial<IntegerRing, E>,
        right: &'a MultivariatePolynomial<IntegerRing, E>,
        variables: &'a [usize],
        bounds: &'a [E],
        anchor: HuMonaganAnchor,
        left_degrees: &[E],
        right_degrees: &[E],
    ) -> Self {
        debug_assert_eq!(left_degrees.len(), left.nvars());
        debug_assert_eq!(right_degrees.len(), right.nvars());
        let (anchored_input, _) = anchor.order_inputs(left, right);
        let (anchored_degrees, other_degrees) = match anchor {
            HuMonaganAnchor::Left => (left_degrees, right_degrees),
            HuMonaganAnchor::Right => (right_degrees, left_degrees),
        };
        let anchored_degrees: SmallVec<[E; INLINED_EXPONENTS]> =
            anchored_degrees.iter().copied().collect();
        let other_degrees: SmallVec<[E; INLINED_EXPONENTS]> =
            other_degrees.iter().copied().collect();

        // `usize::MAX` marks variables whose row support is already proven too large.
        let mut maximum_row_supports: SmallVec<[usize; INLINED_EXPONENTS]> =
            smallvec![usize::MAX; anchored_input.nvars()];
        if let Some(current_variable) = variables.first().copied() {
            let current_support = maximum_coefficient_row_support_bounded(
                anchored_input,
                current_variable,
                usize::MAX,
            )
            .unwrap();
            maximum_row_supports[current_variable] = current_support;
            let maximum_candidate_support =
                current_support / HU_MONAGAN_MAIN_VARIABLE_ROW_REDUCTION;

            if maximum_candidate_support > 0 {
                let current_image_work = hu_monagan_main_image_work(
                    &anchored_degrees,
                    &other_degrees,
                    current_variable,
                    current_support,
                );
                let current_range = hu_monagan_kronecker_range(
                    variables,
                    bounds,
                    &anchored_degrees,
                    &other_degrees,
                    current_variable,
                );
                let maximum_modulus = SMOOTH_PRIMES.last().map(|prime| prime.0);

                for variable in variables.iter().copied().skip(1) {
                    if bounds[variable] == E::zero() {
                        continue;
                    }

                    // With degree d, N terms occupy at most d + 1 exponent rows, so the
                    // pigeonhole principle gives a lower bound on the largest row.
                    let row_count = u128::from(anchored_degrees[variable].to_u32()) + 1;
                    let minimum_support = (anchored_input.nterms() as u128).div_ceil(row_count);
                    if minimum_support > maximum_candidate_support as u128
                        || hu_monagan_main_image_work(
                            &anchored_degrees,
                            &other_degrees,
                            variable,
                            minimum_support as usize,
                        ) > current_image_work
                    {
                        continue;
                    }

                    let range_is_feasible = hu_monagan_kronecker_range(
                        variables,
                        bounds,
                        &anchored_degrees,
                        &other_degrees,
                        variable,
                    )
                    .is_some_and(|candidate_range| {
                        current_range.is_none_or(|range| candidate_range <= range)
                            && candidate_range.checked_mul(2).is_some_and(|bound| {
                                maximum_modulus.is_some_and(|modulus| bound <= modulus)
                            })
                    });
                    if !range_is_feasible {
                        continue;
                    }

                    if let Some(support) = maximum_coefficient_row_support_bounded(
                        anchored_input,
                        variable,
                        maximum_candidate_support,
                    ) {
                        maximum_row_supports[variable] = support;
                    }
                }
            }
        }

        Self {
            left,
            right,
            variables,
            bounds,
            anchor,
            anchored_degrees,
            other_degrees,
            maximum_row_supports,
        }
    }

    /// Orders the active variables with `main_variable` first and the interpolation variables by
    /// descending GCD degree bound.
    fn variable_order(&self, main_variable: usize) -> SmallVec<[usize; INLINED_EXPONENTS]> {
        let mut order: SmallVec<[_; INLINED_EXPONENTS]> = self.variables.iter().copied().collect();
        let main_index = order
            .iter()
            .position(|variable| *variable == main_variable)
            .expect("Hu main variable is active");
        order.swap(0, main_index);
        order[1..].sort_by(|left, right| self.bounds[*right].cmp(&self.bounds[*left]));
        order
    }

    /// Estimates the univariate image work for one main variable.
    fn main_image_work(&self, variable: usize) -> u128 {
        hu_monagan_main_image_work(
            &self.anchored_degrees,
            &self.other_degrees,
            variable,
            self.maximum_row_supports[variable],
        )
    }

    /// Returns the mixed-radix range used to encode all variables except the main variable.
    fn kronecker_range(&self, main_variable: usize) -> Option<u64> {
        hu_monagan_kronecker_range(
            self.variables,
            self.bounds,
            &self.anchored_degrees,
            &self.other_degrees,
            main_variable,
        )
    }

    /// Selects a main variable with a substantially smaller row and a strictly smaller
    /// mixed-radix interpolation range.
    fn alternative_main_variable(&self) -> Option<usize> {
        let current_variable = *self.variables.first()?;
        let current_support = self.maximum_row_supports[current_variable];
        let maximum_candidate_support = current_support / HU_MONAGAN_MAIN_VARIABLE_ROW_REDUCTION;
        let current_image_work = self.main_image_work(current_variable);
        let current_range = self.kronecker_range(current_variable);
        let maximum_modulus = SMOOTH_PRIMES.last()?.0;

        self.variables
            .iter()
            .copied()
            .filter(|variable| {
                *variable != current_variable
                    && self.bounds[*variable] > E::zero()
                    && self.maximum_row_supports[*variable] <= maximum_candidate_support
                    && self.main_image_work(*variable) <= current_image_work
            })
            .filter(|variable| {
                self.kronecker_range(*variable)
                    .is_some_and(|candidate_range| {
                        current_range.is_none_or(|range| candidate_range < range)
                            && candidate_range
                                .checked_mul(2)
                                .is_some_and(|bound| bound <= maximum_modulus)
                    })
            })
            .min_by_key(|variable| self.maximum_row_supports[*variable])
    }

    /// Extracts content in the selected main variable and prepares the permuted Hu inputs.
    fn prepare(&self, main_variable: usize) -> Option<PreparedHuMonaganGcd<E>> {
        let left_content = self.left.univariate_content(main_variable);
        let right_content = self.right.univariate_content(main_variable);
        let content = left_content.gcd(&right_content);

        let left_primitive = if left_content.is_one() {
            Cow::Borrowed(self.left)
        } else {
            Cow::Owned(self.left / &left_content)
        };
        let right_primitive = if right_content.is_one() {
            Cow::Borrowed(self.right)
        } else {
            Cow::Owned(self.right / &right_content)
        };

        let order = self.variable_order(main_variable);
        let left = left_primitive.rearrange_impl(&order, false, false);
        let right = right_primitive.rearrange_impl(&order, false, false);
        let mut bounds: SmallVec<[_; INLINED_EXPONENTS]> = smallvec![E::zero(); self.bounds.len()];
        for (new_variable, old_variable) in order.iter().copied().enumerate() {
            bounds[new_variable] = self.bounds[old_variable];
        }
        let variables = bounds
            .iter()
            .enumerate()
            .filter_map(|(variable, bound)| (*bound > E::zero()).then_some(variable))
            .collect::<SmallVec<[_; INLINED_EXPONENTS]>>();
        if variables.len() < 3 || variables.first() != Some(&0) {
            return None;
        }

        let (anchored_input, _) = self.anchor.order_inputs(&left, &right);
        let selected_support = maximum_coefficient_row_support(anchored_input, 0);
        // Candidate content can change its row geometry. Retain the plan only if the selected
        // primitive input still has the required advantage over the original current row.
        if selected_support
            > self.maximum_row_supports[self.variables[0]] / HU_MONAGAN_MAIN_VARIABLE_ROW_REDUCTION
        {
            return None;
        }

        Some(PreparedHuMonaganGcd {
            left,
            right,
            bounds,
            order,
            content,
            anchor: self.anchor,
        })
    }
}

/// Estimates the cost of the univariate images for one main variable.
fn hu_monagan_main_image_work<E: PositiveExponent>(
    anchored_degrees: &[E],
    other_degrees: &[E],
    variable: usize,
    row_support: usize,
) -> u128 {
    let degree_span = u128::from(anchored_degrees[variable].to_u32())
        + u128::from(other_degrees[variable].to_u32())
        + 2;
    degree_span * row_support as u128
}

/// Returns the mixed-radix range for every active variable except `main_variable`.
fn hu_monagan_kronecker_range<E: PositiveExponent>(
    variables: &[usize],
    bounds: &[E],
    anchored_degrees: &[E],
    other_degrees: &[E],
    main_variable: usize,
) -> Option<u64> {
    let mut range = 1u64;
    for variable in variables.iter().copied() {
        if variable == main_variable {
            continue;
        }

        let radix = anchored_degrees[variable]
            .max(other_degrees[variable])
            .max(bounds[variable])
            .to_u32()
            .checked_add(1)?;
        range = range.checked_mul(u64::from(radix))?;
    }
    Some(range)
}

/// Returns the largest number of terms that share one exponent in `variable`.
fn maximum_coefficient_row_support<E: PositiveExponent>(
    polynomial: &MultivariatePolynomial<IntegerRing, E>,
    variable: usize,
) -> usize {
    maximum_coefficient_row_support_bounded(polynomial, variable, usize::MAX).unwrap()
}

/// Returns the largest row, or `None` as soon as one exponent row exceeds `maximum` terms.
fn maximum_coefficient_row_support_bounded<E: PositiveExponent>(
    polynomial: &MultivariatePolynomial<IntegerRing, E>,
    variable: usize,
    maximum: usize,
) -> Option<usize> {
    let mut rows = CoefficientRowCounter::default();
    for exponents in polynomial.exponents_iter() {
        if rows.increment(exponents[variable].to_u32(), polynomial.nterms()) > maximum {
            return None;
        }
    }
    Some(rows.largest)
}

/// Counts low exponent indices densely and switches to a map when the dense span would exceed the
/// polynomial's term count.
#[derive(Default)]
struct CoefficientRowCounter {
    dense: Vec<usize>,
    sparse: Option<HashMap<u32, usize>>,
    largest: usize,
}

impl CoefficientRowCounter {
    /// Increments one exponent row while keeping the number of stored counters O(term count).
    fn increment(&mut self, exponent: u32, term_count: usize) -> usize {
        let count = if let Some(sparse) = &mut self.sparse {
            sparse.entry(exponent).or_insert(0)
        } else if let Ok(index) = usize::try_from(exponent)
            && index < term_count
        {
            if self.dense.len() <= index {
                self.dense.resize(index + 1, 0);
            }
            &mut self.dense[index]
        } else {
            let mut sparse = HashMap::<u32, usize>::default();
            for (index, count) in std::mem::take(&mut self.dense).into_iter().enumerate() {
                if count != 0 {
                    sparse.insert(index as u32, count);
                }
            }
            self.sparse = Some(sparse);
            self.sparse.as_mut().unwrap().entry(exponent).or_insert(0)
        };
        *count += 1;
        self.largest = self.largest.max(*count);
        *count
    }
}

/// Computes every variable degree in one traversal of the exponent matrix.
#[cfg(test)]
fn polynomial_degrees<E: PositiveExponent>(
    polynomial: &MultivariatePolynomial<IntegerRing, E>,
) -> SmallVec<[E; INLINED_EXPONENTS]> {
    let mut degrees: SmallVec<[E; INLINED_EXPONENTS]> = smallvec![E::zero(); polynomial.nvars()];
    for exponents in polynomial.exponents_iter() {
        for (degree, exponent) in degrees.iter_mut().zip(exponents) {
            *degree = (*degree).max(*exponent);
        }
    }
    degrees
}

/// Primitive, permuted inputs for one Hu-Monagan main-variable plan.
struct PreparedHuMonaganGcd<E: PositiveExponent> {
    left: MultivariatePolynomial<IntegerRing, E>,
    right: MultivariatePolynomial<IntegerRing, E>,
    bounds: SmallVec<[E; INLINED_EXPONENTS]>,
    order: SmallVec<[usize; INLINED_EXPONENTS]>,
    content: MultivariatePolynomial<IntegerRing, E>,
    anchor: HuMonaganAnchor,
}

impl<E: PositiveExponent> PreparedHuMonaganGcd<E> {
    /// Runs Hu in the selected coordinates, restores the original variables, and restores content.
    fn run(self) -> Option<MultivariatePolynomial<IntegerRing, E>> {
        let mut gcd = self.left.gcd_hu_monagan_with_preapproved_plan(
            &self.right,
            &self.bounds,
            self.anchor,
        )?;
        gcd = gcd.rearrange_impl(&self.order, true, false);
        if !self.content.is_one() {
            gcd = gcd * &self.content;
        }
        Some(<IntegerRing as PolynomialGCD<E>>::normalize(gcd))
    }
}

/// Returns the minimum modulus for a Hu-Monagan interpolation image.
///
/// The Kronecker range bound keeps the encoded exponents distinct in the
/// multiplicative group. When twice the largest input coefficient fits below
/// `2^32`, the same modulus also preserves its symmetric integer
/// representative. Larger coefficients are reconstructed by CRT; the
/// coefficient-height bound chooses word primes large enough to target at
/// most eight images without exceeding the usable `u64` field range.
fn hu_monagan_prime_lower_bound(
    kronecker_range: u64,
    delta: u32,
    twice_largest_coefficient: &Integer,
) -> u64 {
    let interpolation_bound = kronecker_range.saturating_mul(2u64.saturating_pow(delta));
    if twice_largest_coefficient < &(1i64 << 32) {
        interpolation_bound.max(twice_largest_coefficient.to_u64().unwrap())
    } else {
        const TARGET_CRT_IMAGES: u64 = 8;
        const MAX_TARGET_PRIME_BITS: u64 = 63;
        let target_prime_bits = twice_largest_coefficient
            .significant_bits()
            .div_ceil(TARGET_CRT_IMAGES)
            .min(MAX_TARGET_PRIME_BITS);
        let coefficient_height_bound = 1u64 << target_prime_bits;
        interpolation_bound.max(coefficient_height_bound)
    }
}

/// Mixed-radix Kronecker map used to encode the variables sampled by Hu-Monagan interpolation.
///
/// If the active radices are `r_s, ..., r_n`, the evaluation powers are
/// `r_s, r_s r_(s+1), ...`, and an exponent vector is encoded as
/// `e_s + r_s e_(s+1) + ...`. Interpolation recovers that encoded exponent as a `u64`; `decode`
/// expands it directly into the polynomial's exponent type.
#[derive(Debug, Clone, PartialEq, Eq)]
struct HuMonaganKroneckerMap {
    start_index: usize,
    radices: Vec<u32>,
    powers: Vec<u64>,
    range: u64,
}

impl HuMonaganKroneckerMap {
    fn new(radices: &[u32], start_index: usize) -> Option<Self> {
        let radices = radices.get(start_index..)?;
        let mut product = 1u64;
        let mut powers = Vec::with_capacity(radices.len());
        for radix in radices {
            if *radix == 0 {
                return None;
            }
            product = product.checked_mul(*radix as u64)?;
            powers.push(product);
        }

        Some(Self {
            start_index,
            radices: radices.to_vec(),
            powers,
            range: product,
        })
    }

    fn powers(&self) -> &[u64] {
        &self.powers
    }

    fn range(&self) -> u64 {
        self.range
    }

    fn decode<E: PositiveExponent>(&self, mut encoded: u64, exponents: &mut [E]) -> Option<()> {
        if encoded >= self.range {
            return None;
        }

        let decoded = exponents.get_mut(self.start_index..)?;
        if decoded.len() != self.radices.len() {
            return None;
        }

        for (exponent, radix) in decoded.iter_mut().zip(&self.radices) {
            *exponent = E::from_u32((encoded % *radix as u64) as u32);
            encoded /= *radix as u64;
        }

        (encoded == 0).then_some(())
    }
}

trait ModularGcdWorkspace: FiniteFieldWorkspace + WordCrt {
    fn first_prime() -> u64;
}

/// Use 64-bit modular images once the reconstruction scale or input coefficient height is large
/// enough that halving the number of CRT images outweighs the slower arithmetic in each image.
#[cfg(not(feature = "binary_size"))]
const U64_ZIPPEL_HEIGHT_BITS: u64 = 512;

/// Returns whether any coefficient reaches the given significant-bit threshold.
#[cfg(not(feature = "binary_size"))]
#[inline]
fn has_coefficient_with_bits<E: PositiveExponent>(
    polynomial: &MultivariatePolynomial<IntegerRing, E>,
    bits: u64,
) -> bool {
    polynomial
        .coefficients
        .iter()
        .any(|coefficient| coefficient.significant_bits() >= bits)
}

/// Selects 64-bit modular images when the GCD scale or both input coefficient heights predict a
/// long CRT reconstruction.
#[cfg(not(feature = "binary_size"))]
#[inline]
fn should_use_u64_zippel<E: PositiveExponent>(
    a: &MultivariatePolynomial<IntegerRing, E>,
    b: &MultivariatePolynomial<IntegerRing, E>,
    gamma: &Integer,
) -> bool {
    gamma.significant_bits() >= U64_ZIPPEL_HEIGHT_BITS
        || (has_coefficient_with_bits(a, U64_ZIPPEL_HEIGHT_BITS)
            && has_coefficient_with_bits(b, U64_ZIPPEL_HEIGHT_BITS))
}

impl ModularGcdWorkspace for u32 {
    fn first_prime() -> u64 {
        u32::get_large_prime() as u64
    }
}

/// Consecutive 64-bit primes used for univariate modular reconstruction before searching for
/// further primes.
const UNIVARIATE_U64_MODULAR_GCD_PRIMES: &[u64] = &[
    18_346_744_073_709_552_031,
    18_346_744_073_709_552_043,
    18_346_744_073_709_552_047,
    18_346_744_073_709_552_049,
    18_346_744_073_709_552_353,
    18_346_744_073_709_552_491,
    18_346_744_073_709_552_521,
    18_346_744_073_709_552_601,
    18_346_744_073_709_552_673,
    18_346_744_073_709_552_691,
    18_346_744_073_709_552_701,
    18_346_744_073_709_552_811,
    18_346_744_073_709_552_829,
    18_346_744_073_709_552_841,
    18_346_744_073_709_552_857,
    18_346_744_073_709_552_863,
    18_346_744_073_709_552_923,
    18_346_744_073_709_552_929,
    18_346_744_073_709_552_973,
    18_346_744_073_709_552_989,
    18_346_744_073_709_553_009,
    18_346_744_073_709_553_133,
    18_346_744_073_709_553_169,
    18_346_744_073_709_553_171,
    18_346_744_073_709_553_199,
    18_346_744_073_709_553_253,
    18_346_744_073_709_553_309,
    18_346_744_073_709_553_321,
    18_346_744_073_709_553_331,
    18_346_744_073_709_553_417,
    18_346_744_073_709_553_451,
    18_346_744_073_709_553_459,
];

impl ModularGcdWorkspace for u64 {
    fn first_prime() -> u64 {
        UNIVARIATE_U64_MODULAR_GCD_PRIMES[0]
    }
}

/// Yields prevalidated primes for univariate modular reconstruction, discovering further primes
/// after the fixed sequence is exhausted.
fn univariate_modular_gcd_prime_iterator() -> impl Iterator<Item = u64> {
    let last = *UNIVARIATE_U64_MODULAR_GCD_PRIMES
        .last()
        .expect("univariate modular GCD prime table must not be empty");
    UNIVARIATE_U64_MODULAR_GCD_PRIMES
        .iter()
        .copied()
        .chain(PrimeIteratorU64::new(last))
}

/// Yields a known modular GCD prime first, followed by its consecutive successors.
struct ModularGcdPrimeIterator {
    first: Option<u64>,
    successors: PrimeIteratorU64,
}

impl ModularGcdPrimeIterator {
    fn for_workspace<UField: ModularGcdWorkspace>() -> Self {
        let first = UField::first_prime();
        Self {
            first: Some(first),
            successors: PrimeIteratorU64::new(first),
        }
    }
}

impl Iterator for ModularGcdPrimeIterator {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        self.first.take().or_else(|| self.successors.next())
    }
}

fn modular_gcd_prime_iterator() -> ModularGcdPrimeIterator {
    ModularGcdPrimeIterator::for_workspace::<ModularGcdFieldWorkspace>()
}

fn next_modular_gcd_prime(
    primes: &mut ModularGcdPrimeIterator,
    context: &str,
) -> ModularGcdFieldWorkspace {
    let Some(p) = primes.next().and_then(|p| {
        <ModularGcdFieldWorkspace as FiniteFieldWorkspace>::try_from_integer(p.into())
    }) else {
        panic!("Ran out of primes for {context}");
    };
    p
}

impl<R: Ring, E: PositiveExponent> MultivariatePolynomial<R, E> {
    /// Evaluation of the exponents by filling in the variables
    #[inline(always)]
    pub(crate) fn evaluate_exponents(
        &self,
        r: &[(usize, R::Element)],
        cache: &mut [Vec<R::Element>],
    ) -> Vec<R::Element> {
        let mut eval = vec![self.ring().one(); self.nterms()];
        for (c, t) in eval.iter_mut().zip(self) {
            // evaluate each exponent
            for (n, v) in r {
                let exp = t.exponents[*n].to_u32() as usize;
                if exp > 0 {
                    if exp < cache[*n].len() {
                        if self.ring().is_zero(&cache[*n][exp]) {
                            cache[*n][exp] = self.ring().pow(v, exp as u64);
                        }

                        self.ring().mul_assign(c, &cache[*n][exp]);
                    } else {
                        self.ring().mul_assign(c, &self.ring().pow(v, exp as u64));
                    }
                }
            }
        }
        eval
    }

    /// Build a univariate polynomial from coefficient-weighted monomial evaluations and advance
    /// every geometric sequence by one step.
    ///
    /// `current_evals[i]` already includes the coefficient of term `i`, while
    /// `term_evals[i]` is its geometric ratio. This lets the bulk kernel update each term with one
    /// multiplication per sample instead of multiplying by both the ratio and the coefficient.
    #[inline(always)]
    pub(crate) fn evaluate_and_advance_weighted_terms(
        &self,
        current_evals: &mut [R::Element],
        term_evals: &[R::Element],
        main_var: usize,
        rows: &[(E, usize, usize)],
        out: &mut MultivariatePolynomial<R, E>,
    ) {
        out.clear();
        let mut new_exp = vec![E::zero(); self.nvars()];
        let geometric_sequence_kernels = self.ring().kernels().geometric_sequences();
        for (exponent, start, end) in rows {
            let current_row = &mut current_evals[*start..*end];
            let term_row = &term_evals[*start..*end];
            let coefficient = geometric_sequence_kernels
                .and_then(|kernels| {
                    kernels.try_sum_and_advance_geometric_sequences(GeometricSequenceStepRequest {
                        current: &mut *current_row,
                        ratios: term_row,
                    })
                })
                .unwrap_or_else(|| {
                    let mut coefficient = self.ring().zero();
                    for (current, term_eval) in current_row.iter_mut().zip(term_row) {
                        self.ring().add_assign(&mut coefficient, &*current);
                        self.ring().mul_assign(current, term_eval);
                    }
                    coefficient
                });

            if !self.ring().is_zero(&coefficient) {
                new_exp[main_var] = *exponent;
                out.coefficients.push(coefficient);
                out.exponents.extend_from_slice(&new_exp);
            }
        }
    }

    /// Find the contiguous ranges belonging to each exponent of `main_var`.
    fn univariate_row_ranges(&self, main_var: usize) -> Vec<(E, usize, usize)> {
        let mut rows = Vec::new();
        let mut exponents = self.exponents.chunks(self.nvars());
        let Some(first) = exponents.next() else {
            return rows;
        };

        let mut row_exponent = first[main_var];
        let mut row_start = 0;
        for (index, exponents) in exponents.enumerate() {
            let term_index = index + 1;
            if exponents[main_var] != row_exponent {
                rows.push((row_exponent, row_start, term_index));
                row_exponent = exponents[main_var];
                row_start = term_index;
            }
        }
        rows.push((row_exponent, row_start, self.nterms()));
        rows
    }
}

/// Dense coefficient storage for the Euclidean GCDs used by modular images.
///
/// Each input is converted once. Remainders reuse the two coefficient buffers, and only the final
/// monic GCD is converted back to a multivariate polynomial.
struct DenseUnivariateGcdContext<'a, F: Field, E: PositiveExponent> {
    prototype: &'a MultivariatePolynomial<F, E>,
    variable: Option<usize>,
}

/// Supplies repeated evaluations of a polynomial pair in their last active variable.
///
/// Calls that require several images cache row boundaries, powers, and output buffers. A single
/// image is evaluated directly because it cannot amortize the metadata scan.
struct RepeatedLastVariableEvaluationContext<'a, F: Field, E: PositiveExponent> {
    left: &'a MultivariatePolynomial<F, E>,
    right: &'a MultivariatePolynomial<F, E>,
    variable: usize,
    cached: Option<(
        LastVariableEvaluationContext<'a, F, E>,
        LastVariableEvaluationContext<'a, F, E>,
        LastVariablePowerWorkspace<F>,
    )>,
}

impl<'a, F: Field, E: PositiveExponent> RepeatedLastVariableEvaluationContext<'a, F, E> {
    fn new(
        left: &'a MultivariatePolynomial<F, E>,
        right: &'a MultivariatePolynomial<F, E>,
        variable: usize,
        expected_evaluations: usize,
    ) -> Self {
        let mut context = Self {
            left,
            right,
            variable,
            cached: None,
        };
        if expected_evaluations > 1 {
            context.enable_reuse();
        }
        context
    }

    /// Prepare cached row metadata and buffers before an interpolation needs multiple images.
    fn enable_reuse(&mut self) {
        if self.cached.is_some() {
            return;
        }

        let left_context = LastVariableEvaluationContext::new(self.left, self.variable);
        let right_context = LastVariableEvaluationContext::new(self.right, self.variable);
        let maximum_degree = left_context
            .maximum_degree()
            .max(right_context.maximum_degree());
        let powers = LastVariablePowerWorkspace::new(self.left.ring(), maximum_degree);
        self.cached = Some((left_context, right_context, powers));
    }

    /// Evaluate both inputs at `value` and pass the results to `operation`.
    fn with_evaluations<T>(
        &mut self,
        value: &F::Element,
        operation: impl FnOnce(&MultivariatePolynomial<F, E>, &MultivariatePolynomial<F, E>) -> T,
    ) -> T {
        if let Some((left, right, powers)) = &mut self.cached {
            powers.start_value(value);
            let left = left.evaluate(powers);
            let right = right.evaluate(powers);
            operation(left, right)
        } else {
            let left = self.left.replace_last(self.variable, value);
            let right = self.right.replace_last(self.variable, value);
            operation(&left, &right)
        }
    }
}

/// Maps a sampled univariate degree to its coefficient in the known GCD shape.
enum ZippelShapeIndex<E: PositiveExponent> {
    Dense {
        minimum_degree: usize,
        indices: Vec<Option<usize>>,
    },
    Sparse(Vec<E>),
}

impl<E: PositiveExponent> ZippelShapeIndex<E> {
    fn new(degrees: impl IntoIterator<Item = E>) -> Self {
        let degrees = degrees.into_iter().collect::<Vec<_>>();
        let minimum_degree = degrees
            .iter()
            .map(|degree| degree.to_u32() as usize)
            .min()
            .expect("a GCD image must have at least one term");
        let maximum_degree = degrees
            .iter()
            .map(|degree| degree.to_u32() as usize)
            .max()
            .unwrap();
        let degree_span = maximum_degree - minimum_degree + 1;
        if degree_span > ZIPPEL_SHAPE_INDEX_MAX_DEGREE_SPAN
            || degree_span
                > degrees
                    .len()
                    .saturating_mul(ZIPPEL_SHAPE_INDEX_MAX_SPARSITY_RATIO)
        {
            return Self::Sparse(degrees);
        }

        let mut indices = vec![None; degree_span];
        for (index, degree) in degrees.into_iter().enumerate() {
            let slot = &mut indices[degree.to_u32() as usize - minimum_degree];
            debug_assert!(slot.is_none());
            *slot = Some(index);
        }
        Self::Dense {
            minimum_degree,
            indices,
        }
    }

    fn get(&self, degree: E) -> Option<usize> {
        match self {
            Self::Dense {
                minimum_degree,
                indices,
            } => (degree.to_u32() as usize)
                .checked_sub(*minimum_degree)
                .and_then(|degree| indices.get(degree))
                .and_then(|index| *index),
            Self::Sparse(degrees) => degrees
                .iter()
                .position(|shape_degree| *shape_degree == degree),
        }
    }
}

impl<'a, F: Field, E: PositiveExponent> DenseUnivariateGcdContext<'a, F, E> {
    fn new(left: &'a MultivariatePolynomial<F, E>, right: &MultivariatePolynomial<F, E>) -> Self {
        let variable = left
            .last_exponents()
            .iter()
            .position(|exponent| !exponent.is_zero())
            .or_else(|| {
                right
                    .last_exponents()
                    .iter()
                    .position(|exponent| !exponent.is_zero())
            });

        debug_assert!(
            left.exponents_iter()
                .chain(right.exponents_iter())
                .all(
                    |exponents| exponents.iter().enumerate().all(|(index, exponent)| {
                        exponent.is_zero() || variable.is_some_and(|variable| index == variable)
                    })
                )
        );

        Self {
            prototype: left,
            variable,
        }
    }

    /// Return whether dense storage remains bounded for this polynomial pair.
    fn storage_is_bounded(
        &self,
        left: &MultivariatePolynomial<F, E>,
        right: &MultivariatePolynomial<F, E>,
    ) -> bool {
        let Some(variable) = self.variable else {
            return true;
        };
        let coefficient_count = left.last_exponents()[variable]
            .max(right.last_exponents()[variable])
            .to_u32() as usize
            + 1;
        coefficient_count <= DENSE_UNIVARIATE_GCD_MAX_COEFFICIENTS
            && coefficient_count
                <= (left.nterms() + right.nterms())
                    .saturating_mul(DENSE_UNIVARIATE_GCD_MAX_SPARSITY_RATIO)
    }

    /// Compute the monic GCD when at least one input consists of one monomial.
    fn gcd_with_monomial(
        &self,
        left: &MultivariatePolynomial<F, E>,
        right: &MultivariatePolynomial<F, E>,
    ) -> MultivariatePolynomial<F, E> {
        let Some(variable) = self.variable else {
            return self.prototype.one();
        };
        let degree = left.exponents(0)[variable].min(right.exponents(0)[variable]);
        if degree.is_zero() {
            return self.prototype.one();
        }
        let mut exponents = vec![E::zero(); self.prototype.nvars()];
        exponents[variable] = degree;
        self.prototype
            .monomial(self.prototype.ring().one(), exponents)
    }

    /// Copy a sparse univariate polynomial into a degree-indexed coefficient buffer.
    fn coefficients(&self, polynomial: &MultivariatePolynomial<F, E>) -> Vec<F::Element> {
        let Some(variable) = self.variable else {
            return vec![polynomial.coefficients[0].clone()];
        };
        let degree = polynomial.last_exponents()[variable].to_u32() as usize;
        let mut coefficients = vec![polynomial.ring().zero(); degree + 1];
        for term in polynomial {
            coefficients[term.exponents[variable].to_u32() as usize] = term.coefficient.clone();
        }
        coefficients
    }

    /// Scale a nonzero dense polynomial so its leading coefficient is one.
    fn make_monic(&self, polynomial: &mut [F::Element]) {
        let Some(leading_coefficient) = polynomial.last() else {
            return;
        };
        if self.prototype.ring().is_one(leading_coefficient) {
            return;
        }
        let inverse = self.prototype.ring().inv(leading_coefficient);
        for coefficient in polynomial {
            self.prototype.ring().mul_assign(coefficient, &inverse);
        }
    }

    /// Replace `dividend` by its remainder modulo the monic `divisor`.
    fn rem_monic(&self, dividend: &mut Vec<F::Element>, divisor: &[F::Element]) {
        debug_assert!(!divisor.is_empty());
        debug_assert!(self.prototype.ring().is_one(divisor.last().unwrap()));
        if dividend.len() < divisor.len() {
            return;
        }
        if divisor.len() == 1 {
            dividend.clear();
            return;
        }

        let divisor_degree = divisor.len() - 1;
        for degree in (divisor_degree..dividend.len()).rev() {
            let leading_coefficient =
                std::mem::replace(&mut dividend[degree], self.prototype.ring().zero());
            if self.prototype.ring().is_zero(&leading_coefficient) {
                continue;
            }

            let shift = degree - divisor_degree;
            for (coefficient, divisor_coefficient) in dividend[shift..degree]
                .iter_mut()
                .zip(&divisor[..divisor_degree])
            {
                self.prototype.ring().sub_mul_assign(
                    coefficient,
                    divisor_coefficient,
                    &leading_coefficient,
                );
            }
        }

        dividend.truncate(divisor_degree);
        while dividend
            .last()
            .is_some_and(|coefficient| self.prototype.ring().is_zero(coefficient))
        {
            dividend.pop();
        }
    }

    /// Convert a dense coefficient buffer back to the original polynomial representation.
    fn polynomial(&self, coefficients: Vec<F::Element>) -> MultivariatePolynomial<F, E> {
        let mut result = self.prototype.zero_with_capacity(coefficients.len());
        let Some(variable) = self.variable else {
            return result.add_constant(coefficients.into_iter().next().unwrap());
        };
        let mut exponents = vec![E::zero(); self.prototype.nvars()];
        for (degree, coefficient) in coefficients.into_iter().enumerate() {
            if !self.prototype.ring().is_zero(&coefficient) {
                exponents[variable] = E::from_u32(degree as u32);
                result.append_monomial_back(coefficient, &exponents);
            }
        }
        result
    }

    /// Compute a monic GCD while retaining all intermediate remainders in dense storage.
    fn gcd(
        &self,
        left: &MultivariatePolynomial<F, E>,
        right: &MultivariatePolynomial<F, E>,
    ) -> MultivariatePolynomial<F, E> {
        let mut left = self.coefficients(left);
        let mut right = self.coefficients(right);
        if left.len() < right.len() {
            mem::swap(&mut left, &mut right);
        }
        self.make_monic(&mut right);

        while !right.is_empty() {
            if right.len() == 1 {
                return self.prototype.one();
            }
            self.rem_monic(&mut left, &right);
            mem::swap(&mut left, &mut right);
            self.make_monic(&mut right);
        }

        self.polynomial(left)
    }
}

impl<F: Field, E: PositiveExponent> MultivariatePolynomial<F, E> {
    /// Compute the univariate GCD using Euclid's algorithm. The result is normalized to 1.
    pub fn univariate_gcd(&self, b: &Self) -> Self {
        if self.is_zero() {
            return b.clone();
        }
        if b.is_zero() {
            return self.clone();
        }

        let dense = DenseUnivariateGcdContext::new(self, b);
        if self.nterms() == 1 || b.nterms() == 1 {
            return dense.gcd_with_monomial(self, b);
        }
        if dense.storage_is_bounded(self, b) {
            return dense.gcd(self, b);
        }

        // Use the existing polynomial-buffer division path when exponent gaps would make the
        // dedicated dense workspace disproportionately large.
        let mut left = self.clone();
        let mut right = b.clone();
        if self.ldegree_max() < b.ldegree_max() {
            mem::swap(&mut left, &mut right);
        }
        let mut remainder = left.quot_rem_univariate(&mut right).1;
        while !remainder.is_zero() {
            left = right;
            right = remainder;
            remainder = left.quot_rem_univariate(&mut right).1;
        }

        if let Some(leading_coefficient) = right.coefficients.last()
            && !right.ring().is_one(leading_coefficient)
        {
            let inverse = right.ring().inv(leading_coefficient);
            let (ring, coefficients) = right.ring_and_coefficients_mut();
            for coefficient in coefficients {
                ring.mul_assign(coefficient, &inverse);
            }
        }
        right
    }

    /// Replace all variables except `v` in the polynomial by elements from
    /// a finite field of size `p`.
    pub fn sample_polynomial(
        &self,
        v: usize,
        r: &[(usize, F::Element)],
        cache: &mut [Vec<F::Element>],
        tm: &mut HashMap<E, F::Element>,
    ) -> Self {
        for mv in self.into_iter() {
            let mut c = mv.coefficient.clone();
            for (n, vv) in r {
                let exp = mv.exponents[*n].to_u32() as usize;
                if exp > 0 {
                    if exp < cache[*n].len() {
                        if self.ring().is_zero(&cache[*n][exp]) {
                            cache[*n][exp] = self.ring().pow(vv, exp as u64);
                        }

                        self.ring().mul_assign(&mut c, &cache[*n][exp]);
                    } else {
                        self.ring()
                            .mul_assign(&mut c, &self.ring().pow(vv, exp as u64));
                    }
                }
            }

            tm.entry(mv.exponents[v])
                .and_modify(|e| self.ring().add_assign(e, &c))
                .or_insert(c);
        }

        let mut res = self.zero();
        let mut e = vec![E::zero(); self.nvars()];
        for (k, c) in tm.drain() {
            if !self.ring().is_zero(&c) {
                e[v] = k;
                res.append_monomial(c, &e);
                e[v] = E::zero();
            }
        }

        res
    }

    /// Find the upper bound of a variable `var` in the gcd.
    /// This is done by computing the univariate gcd by
    /// substituting all variables except `var`. This
    /// upper bound could be too tight due to an unfortunate
    /// sample point, but this is rare.
    fn get_gcd_var_bound(ap: &Self, bp: &Self, vars: &[usize], var: usize) -> E
    where
        F: SampleableRing<SamplingPolicy = RangeInclusive<i64>>,
    {
        let mut rng = rand::rng();

        // store a table for variables raised to a certain power
        let mut cache = (0..ap.nvars())
            .map(|i| {
                vec![
                    ap.ring().zero();
                    min(
                        max(ap.degree(i), bp.degree(i)).to_u32() as usize + 1,
                        POW_CACHE_SIZE
                    )
                ]
            })
            .collect::<Vec<_>>();

        // store a power map for the univariate polynomials that will be sampled
        // the sampling_polynomial routine will set the power to 0 after use
        let mut tm = HashMap::with_capacity_and_hasher(INITIAL_POW_MAP_SIZE, Default::default());

        // generate random numbers for all non-leading variables
        // TODO: apply a Horner scheme to speed up the substitution?

        let mut fail_count = 0;
        let (_, a1, b1) = loop {
            for v in &mut cache {
                for vi in v {
                    *vi = ap.ring().zero();
                }
            }

            let r: Vec<_> = vars
                .iter()
                .map(|i| (*i, sample_nonzero_field_element(ap.ring(), &mut rng)))
                .collect();

            let a1 = ap.sample_polynomial(var, &r, &mut cache, &mut tm);
            let b1 = bp.sample_polynomial(var, &r, &mut cache, &mut tm);

            if a1.ldegree(var) == ap.degree(var) && b1.ldegree(var) == bp.degree(var) {
                break (r, a1, b1);
            }

            if let Some(size) = ap.ring().size()
                && fail_count * 2 > size
            {
                debug!("Field is too small to find a good sample point");
                // TODO: upgrade to larger field?
                return ap.degree(var).min(bp.degree(var));
            }

            debug!(
                "Degree error during sampling: trying again: a={}, a1={}, bp={}, b1={}",
                ap, a1, bp, b1
            );
            fail_count += 1;
        };

        let g1 = a1.univariate_gcd(&b1);
        g1.ldegree_max()
    }

    /// Samples each variable bound independently. This is used for coefficient fields or degree
    /// ranges for which the fused dense sampler is not suitable.
    fn get_gcd_var_bounds_separately(
        ap: &Self,
        bp: &Self,
        vars: &[usize],
    ) -> SmallVec<[E; INLINED_EXPONENTS]>
    where
        F: SampleableRing<SamplingPolicy = RangeInclusive<i64>>,
    {
        let mut bounds = (0..ap.nvars())
            .map(|_| E::zero())
            .collect::<SmallVec<[_; INLINED_EXPONENTS]>>();
        for var in vars {
            if ap.degree(*var) == E::zero() || bp.degree(*var) == E::zero() {
                continue;
            }

            let sampled_variables = vars
                .iter()
                .filter(|variable| *variable != var)
                .copied()
                .collect::<SmallVec<[usize; INLINED_EXPONENTS]>>();
            bounds[*var] = Self::get_gcd_var_bound(ap, bp, &sampled_variables, *var);
        }
        bounds
    }

    fn solve_vandermonde(
        &self,
        main_var: usize,
        shape: &[(MultivariatePolynomial<F, E>, E)],
        row_sample_values: Vec<Vec<F::Element>>,
        samples: Vec<Vec<F::Element>>,
    ) -> MultivariatePolynomial<F, E> {
        let mut gp = self.zero();

        for (((shape_part, ex), sample_powers), rhs) in
            shape.iter().zip(&row_sample_values).zip(&samples)
        {
            let coeffs = self.solve_shifted_transposed_vandermonde(sample_powers, rhs);

            for (coeff, term) in coeffs.into_iter().zip(shape_part) {
                let mut ee: SmallVec<[E; INLINED_EXPONENTS]> = term.exponents.into();
                ee[main_var] = *ex;
                gp.append_monomial(coeff, &ee);
            }
        }

        gp
    }

    /// Solve `rhs[k] = sum_i c_i * x[i]^(k+1)`.
    pub(crate) fn solve_shifted_transposed_vandermonde(
        &self,
        x: &[F::Element],
        rhs: &[F::Element],
    ) -> Vec<F::Element> {
        debug_assert_eq!(x.len(), rhs.len());

        match x.len() {
            0 => vec![],
            1 => vec![self.ring().div(&rhs[0], &x[0])],
            len => {
                let mut master = vec![self.ring().zero(); len + 1];
                master[0] = self.ring().one();

                for (i, x) in x.iter().enumerate() {
                    let first = &mut master[0];
                    let mut old_last = first.clone();
                    self.ring().mul_assign(first, &self.ring().neg(x));
                    for m in &mut master[1..=i] {
                        let ov = m.clone();
                        self.ring().mul_assign(m, &self.ring().neg(x));
                        self.ring().add_assign(m, &old_last);
                        old_last = ov;
                    }
                    master[i + 1] = self.ring().one();
                }

                let mut sol = Vec::with_capacity(len);
                let mut denominators = Vec::with_capacity(len);
                for (i, s) in x.iter().enumerate() {
                    // sample master/(1-s_i) by using the factorized form
                    let mut norm = self.ring().one();
                    for (j, l) in x.iter().enumerate() {
                        if j != i {
                            let diff = self.ring().sub(s, l);
                            if self.ring().is_zero(&diff) {
                                panic!("Vandermonde matrix has duplicate entries");
                            }
                            self.ring().mul_assign(&mut norm, &diff);
                        }
                    }

                    // divide out 1-s_i
                    let mut coeff = self.ring().zero();
                    let mut last_q = self.ring().zero();
                    for (m, rhs) in master.iter().skip(1).zip(rhs).rev() {
                        last_q = self.ring().add(m, &self.ring().mul(s, &last_q));
                        self.ring().add_mul_assign(&mut coeff, &last_q, rhs);
                    }

                    // Multiplying by x[i] converts the ordinary transposed Vandermonde
                    // denominator for powers k into the shifted denominator for powers k + 1.
                    self.ring().mul_assign(&mut norm, &x[i]);
                    denominators.push(norm);
                    sol.push(coeff);
                }

                if self.ring().size().is_some() {
                    // In a finite field, recover every reciprocal from one inversion and a pair
                    // of prefix/suffix multiplication passes.
                    let mut prefixes = Vec::with_capacity(len);
                    prefixes.push(self.ring().one());
                    let mut product = denominators[0].clone();
                    for denominator in &denominators[1..] {
                        prefixes.push(product.clone());
                        self.ring().mul_assign(&mut product, denominator);
                    }

                    let mut inverse_suffix = self.ring().inv(&product);
                    for index in (1..len).rev() {
                        let inverse_denominator =
                            self.ring().mul(&prefixes[index], &inverse_suffix);
                        self.ring()
                            .mul_assign(&mut sol[index], &inverse_denominator);
                        self.ring()
                            .mul_assign(&mut inverse_suffix, &denominators[index]);
                    }
                    self.ring().mul_assign(&mut sol[0], &inverse_suffix);
                } else {
                    for (coefficient, denominator) in sol.iter_mut().zip(&denominators) {
                        self.ring().div_assign(coefficient, denominator);
                    }
                }

                sol
            }
        }
    }

    /// Perform Newton interpolation in the variable `x`, by providing
    /// a list of sample points `a` and their evaluations `u`.
    pub fn newton_interpolation(
        a: &[F::Element],
        u: &[MultivariatePolynomial<F, E>],
        x: usize, // the variable index to extend the polynomial by
    ) -> MultivariatePolynomial<F, E> {
        let field = &u[0].ring();

        // compute inverses
        let mut gammas = Vec::with_capacity(a.len());
        for k in 1..a.len() {
            let mut pr = field.sub(&a[k], &a[0]);
            for i in 1..k {
                u[0].ring().mul_assign(&mut pr, &field.sub(&a[k], &a[i]));
            }
            gammas.push(u[0].ring().inv(&pr));
        }

        // compute Newton coefficients
        let mut v = vec![u[0].clone()];
        for k in 1..a.len() {
            let mut tmp = v[k - 1].clone();
            for j in (0..k - 1).rev() {
                tmp = tmp.mul_coeff(field.sub(&a[k], &a[j])).add(v[j].clone());
            }

            let mut r = u[k].clone() - tmp;
            r = r.mul_coeff(gammas[k - 1].clone());
            v.push(r);
        }

        // convert to standard form
        let mut e = vec![E::zero(); u[0].nvars()];
        e[x] = E::one();
        let xp = u[0].monomial(field.one(), e);
        let mut u = v[v.len() - 1].clone();
        for k in (0..v.len() - 1).rev() {
            // TODO: prevent cloning
            u = u * &(xp.clone() - v[0].constant(a[k].clone())) + v[k].clone();
        }
        u
    }

    #[instrument(level = "trace", fields(%a, %b))]
    fn construct_new_image_single_scale(
        a: &MultivariatePolynomial<F, E>,
        b: &MultivariatePolynomial<F, E>,
        a_ldegree: E,
        b_ldegree: E,
        bounds: &mut [E],
        single_scale: usize,
        vars: &[usize],
        main_var: usize,
        shape: &[(MultivariatePolynomial<F, E>, E)],
    ) -> Result<MultivariatePolynomial<F, E>, GCDError>
    where
        F: SampleableRing<SamplingPolicy = RangeInclusive<i64>>,
    {
        if vars.is_empty() {
            // return gcd divided by the single scale factor
            let g = a.univariate_gcd(b);

            if g.ldegree(main_var) < bounds[main_var] {
                // original image and form and degree bounds are unlucky
                // change the bound and try a new prime
                debug!("Unlucky degree bound: {} vs {}", g, bounds[main_var]);
                bounds[main_var] = g.ldegree(main_var);
                return Err(GCDError::BadOriginalImage);
            }

            if g.ldegree(main_var) > bounds[main_var] {
                return Err(GCDError::BadCurrentImage);
            }

            // check if all the monomials of the image appear in the shape
            // if not, the original shape is bad
            for m in g.into_iter() {
                if shape.iter().all(|(_, pow)| *pow != m.exponents[main_var]) {
                    debug!("Bad shape: terms missing");
                    return Err(GCDError::BadOriginalImage);
                }
            }

            // construct the scaling coefficient
            let (_, d) = &shape[single_scale];
            for t in &g {
                if t.exponents[main_var] == *d {
                    let scale_factor = a.ring().neg(&a.ring().inv(t.coefficient)); // TODO: why -1?
                    return Ok(g.mul_coeff(scale_factor));
                }
            }

            // the scaling term is missing, so the assumed form is wrong
            debug!("Bad original image");
            return Err(GCDError::BadOriginalImage);
        }

        let mut rng = rand::rng();

        let mut failure_count = 0;

        let shape_index_by_degree =
            ZippelShapeIndex::new(shape.iter().map(|(_, exponent)| *exponent));

        // Store powers only for the variables evaluated at this recursion level. Scan both
        // exponent arrays once to find their required cache sizes.
        let mut evaluated_degrees = vec![E::zero(); a.nvars()];
        for polynomial in [a, b] {
            for exponents in polynomial.exponents_iter() {
                for &variable in vars {
                    evaluated_degrees[variable] =
                        evaluated_degrees[variable].max(exponents[variable]);
                }
            }
        }
        let mut cache = evaluated_degrees
            .into_iter()
            .map(|degree| {
                if degree == E::zero() {
                    Vec::new()
                } else {
                    vec![a.ring().zero(); (degree.to_u32() as usize + 1).min(POW_CACHE_SIZE)]
                }
            })
            .collect::<Vec<_>>();

        // find a set of sample points that yield unique coefficients for every coefficient of a term in the shape
        let (row_sample_values, samples) = 'find_root_sample: loop {
            for v in &mut cache {
                for vi in v {
                    *vi = a.ring().zero();
                }
            }

            let r_orig: SmallVec<[_; INLINED_EXPONENTS]> = vars
                .iter()
                .map(|i| (*i, sample_nonzero_field_element(a.ring(), &mut rng)))
                .collect();

            let mut row_sample_values = Vec::with_capacity(shape.len()); // coefficients for the linear system
            let mut samples_needed = 0;
            for (c, _) in shape.iter() {
                samples_needed = samples_needed.max(c.nterms());
                let mut row = Vec::with_capacity(c.nterms());
                let mut seen = HashSet::new();

                for t in c {
                    // evaluate each exponent
                    let mut c = a.ring().one();
                    for (n, v) in &r_orig {
                        let exp = t.exponents[*n].to_u32() as usize;
                        if exp > 0 {
                            if exp < cache[*n].len() {
                                if a.ring().is_zero(&cache[*n][exp]) {
                                    cache[*n][exp] = a.ring().pow(v, exp as u64);
                                }

                                a.ring().mul_assign(&mut c, &cache[*n][exp]);
                            } else {
                                a.ring().mul_assign(&mut c, &a.ring().pow(v, exp as u64));
                            }
                        }
                    }
                    row.push(c.clone());

                    // check if each element is unique
                    if !seen.insert(c.clone()) {
                        debug!("Duplicate element: restarting");
                        continue 'find_root_sample;
                    }
                }

                row_sample_values.push(row);
            }

            debug_assert_eq!(row_sample_values[single_scale].len(), 1);
            let scale_ratio = row_sample_values[single_scale][0].clone();
            let mut scale_value = scale_ratio.clone();

            let mut samples = vec![Vec::with_capacity(samples_needed); shape.len()];
            let mut r = r_orig.clone();

            let a_eval = a.evaluate_exponents(&r_orig, &mut cache);
            let b_eval = b.evaluate_exponents(&r_orig, &mut cache);

            let mut a_current = a
                .coefficients
                .iter()
                .zip(&a_eval)
                .map(|(coefficient, eval)| a.ring().mul(coefficient, eval))
                .collect::<Vec<_>>();
            let mut b_current = b
                .coefficients
                .iter()
                .zip(&b_eval)
                .map(|(coefficient, eval)| b.ring().mul(coefficient, eval))
                .collect::<Vec<_>>();
            let a_rows = a.univariate_row_ranges(main_var);
            let b_rows = b.univariate_row_ranges(main_var);

            let mut a_poly = a.zero_with_capacity(a_ldegree.to_u32() as usize + 1);
            let mut b_poly = b.zero_with_capacity(b_ldegree.to_u32() as usize + 1);
            let mut sampled_term_by_shape = vec![None; shape.len()];

            for sample_index in 0..samples_needed {
                // sample at r^i
                if sample_index > 0 {
                    for (c, rr) in r.iter_mut().zip(&r_orig) {
                        *c = (c.0, a.ring().mul(&c.1, &rr.1));
                    }
                }

                // now construct the univariate polynomials from the current evaluated monomials
                a.evaluate_and_advance_weighted_terms(
                    &mut a_current,
                    &a_eval,
                    main_var,
                    &a_rows,
                    &mut a_poly,
                );
                b.evaluate_and_advance_weighted_terms(
                    &mut b_current,
                    &b_eval,
                    main_var,
                    &b_rows,
                    &mut b_poly,
                );

                if a_poly.ldegree(main_var) != a_ldegree || b_poly.ldegree(main_var) != b_ldegree {
                    continue 'find_root_sample;
                }

                let g = a_poly.univariate_gcd(&b_poly);
                debug!(
                    "GCD of sample at point {:?} in main var {}: {}",
                    r, main_var, g
                );

                if g.ldegree(main_var) < bounds[main_var] {
                    // original image and form and degree bounds are unlucky
                    // change the bound and try a new prime

                    debug!("Unlucky degree bound: {} vs {}", g, bounds[main_var]);
                    bounds[main_var] = g.ldegree(main_var);
                    return Err(GCDError::BadOriginalImage);
                }

                if g.ldegree(main_var) > bounds[main_var] {
                    failure_count += 1;
                    if failure_count > 2 {
                        // p is likely unlucky
                        debug!(
                            "Bad current image: gcd({},{}) mod {} under {:?} = {}",
                            a,
                            b,
                            a.ring(),
                            r,
                            g
                        );
                        return Err(GCDError::BadCurrentImage);
                    }
                    debug!("Degree too high");
                    continue 'find_root_sample;
                }

                sampled_term_by_shape.fill(None);
                for (term_index, term) in (&g).into_iter().enumerate() {
                    let Some(shape_index) = shape_index_by_degree.get(term.exponents[main_var])
                    else {
                        debug!("Bad shape: terms missing");
                        return Err(GCDError::BadOriginalImage);
                    };
                    sampled_term_by_shape[shape_index] = Some(term_index);
                }

                // Normalize the sampled image by the monomial coefficient chosen in the first
                // image. Its value follows the same geometric sequence as the sample points.
                let Some(scale_term) = sampled_term_by_shape[single_scale] else {
                    debug!("Bad original image");
                    return Err(GCDError::BadOriginalImage);
                };
                let coefficient = scale_value.clone();
                a.ring().mul_assign(&mut scale_value, &scale_ratio);
                let scale_factor = g.ring().div(&coefficient, &g.coefficients[scale_term]);

                // construct the right-hand side
                for (i, (rhs, (shape_part, _))) in samples.iter_mut().zip(shape).enumerate() {
                    // we may not need all terms
                    if rhs.len() == shape_part.nterms() {
                        continue;
                    }

                    if let Some(term_index) = sampled_term_by_shape[i] {
                        rhs.push(
                            a.ring()
                                .neg(&a.ring().mul(&g.coefficients[term_index], &scale_factor)),
                        );
                    } else {
                        rhs.push(a.ring().zero());
                    }
                }
            }

            break (row_sample_values, samples);
        };

        Ok(a.solve_vandermonde(main_var, shape, row_sample_values, samples))
    }

    /// Construct an image in the case where no monomial in the main variable is a single term.
    /// Using Javadi's method to solve the normalization problem, we first determine the coefficients of a single monomial using
    /// Gaussian elimination. Then, we are back in the single term case and we use a Vandermonde
    /// matrix to solve for every coefficient.
    #[instrument(level = "trace", fields(%a, %b))]
    fn construct_new_image_multiple_scales(
        a: &MultivariatePolynomial<F, E>,
        b: &MultivariatePolynomial<F, E>,
        a_ldegree: E,
        b_ldegree: E,
        bounds: &mut [E],
        vars: &[usize],
        main_var: usize,
        shape: &[(MultivariatePolynomial<F, E>, E)],
    ) -> Result<MultivariatePolynomial<F, E>, GCDError>
    where
        F: SampleableRing<SamplingPolicy = RangeInclusive<i64>>,
    {
        let mut rng = rand::rng();

        let mut failure_count = 0;

        // store a table for variables raised to a certain power
        let mut cache = (0..a.nvars())
            .map(|i| {
                vec![
                    a.ring().zero();
                    min(
                        max(a.degree(i), b.degree(i)).to_u32() as usize + 1,
                        POW_CACHE_SIZE
                    )
                ]
            })
            .collect::<Vec<_>>();

        // sort the shape based on the number of terms in the coefficient
        let mut shape_map: Vec<_> = (0..shape.len()).collect();
        shape_map.sort_unstable_by_key(|i| shape[*i].0.nterms());

        let mut scaling_var_relations: Vec<Vec<F::Element>> = vec![];

        let max_terms = shape[*shape_map.last().unwrap()].0.nterms();

        // find a set of sample points that yield unique coefficients for every coefficient of a term in the shape
        let (row_sample_values, samples) = 'find_root_sample: loop {
            for v in &mut cache {
                for vi in v {
                    *vi = a.ring().zero();
                }
            }

            let r_orig: SmallVec<[_; INLINED_EXPONENTS]> = vars
                .iter()
                .map(|i| (*i, sample_nonzero_field_element(a.ring(), &mut rng)))
                .collect();

            let mut row_sample_values = Vec::with_capacity(shape.len()); // coefficients for the linear system

            let max_samples_needed = 2 * max_terms - 1;
            for (c, _) in shape.iter() {
                let mut row = Vec::with_capacity(c.nterms());
                let mut seen = HashSet::new();

                for t in c {
                    // evaluate each exponent
                    let mut c = a.ring().one();
                    for (n, v) in &r_orig {
                        let exp = t.exponents[*n].to_u32() as usize;
                        if exp > 0 {
                            if exp < cache[*n].len() {
                                if a.ring().is_zero(&cache[*n][exp]) {
                                    cache[*n][exp] = a.ring().pow(v, exp as u64);
                                }

                                a.ring().mul_assign(&mut c, &cache[*n][exp]);
                            } else {
                                a.ring().mul_assign(&mut c, &a.ring().pow(v, exp as u64));
                            }
                        }
                    }
                    row.push(c.clone());

                    // check if each element is unique
                    if !seen.insert(c) {
                        debug!("Duplicate element: restarting");
                        continue 'find_root_sample;
                    }
                }

                row_sample_values.push(row);
            }

            let mut samples = vec![Vec::with_capacity(max_samples_needed); shape.len()];
            let mut r = r_orig.clone();

            let a_eval = a.evaluate_exponents(&r_orig, &mut cache);
            let b_eval = b.evaluate_exponents(&r_orig, &mut cache);

            let mut a_current = a
                .coefficients
                .iter()
                .zip(&a_eval)
                .map(|(coefficient, eval)| a.ring().mul(coefficient, eval))
                .collect::<Vec<_>>();
            let mut b_current = b
                .coefficients
                .iter()
                .zip(&b_eval)
                .map(|(coefficient, eval)| b.ring().mul(coefficient, eval))
                .collect::<Vec<_>>();
            let a_rows = a.univariate_row_ranges(main_var);
            let b_rows = b.univariate_row_ranges(main_var);

            let mut a_poly = a.zero_with_capacity(a.degree(main_var).to_u32() as usize + 1);
            let mut b_poly = b.zero_with_capacity(b.degree(main_var).to_u32() as usize + 1);

            let mut second_index = 1;
            let mut solved_coeff = None;
            for sample_index in 0..max_samples_needed {
                if solved_coeff.is_some() && sample_index >= max_terms {
                    // we have enough samples
                    break;
                }

                // sample at r^i
                if sample_index > 0 {
                    for (c, rr) in r.iter_mut().zip(&r_orig) {
                        *c = (c.0, a.ring().mul(&c.1, &rr.1));
                    }
                }

                // now construct the univariate polynomials from the current evaluated monomials
                a.evaluate_and_advance_weighted_terms(
                    &mut a_current,
                    &a_eval,
                    main_var,
                    &a_rows,
                    &mut a_poly,
                );
                b.evaluate_and_advance_weighted_terms(
                    &mut b_current,
                    &b_eval,
                    main_var,
                    &b_rows,
                    &mut b_poly,
                );

                if a_poly.ldegree(main_var) != a_ldegree || b_poly.ldegree(main_var) != b_ldegree {
                    continue 'find_root_sample;
                }

                let mut g = a_poly.univariate_gcd(&b_poly);
                debug!(
                    "GCD of sample at point {:?} in main var {}: {}",
                    r, main_var, g
                );

                if g.ldegree(main_var) < bounds[main_var] {
                    // original image and form and degree bounds are unlucky
                    // change the bound and try a new prime

                    debug!("Unlucky degree bound: {} vs {}", g, bounds[main_var]);
                    bounds[main_var] = g.ldegree(main_var);
                    return Err(GCDError::BadOriginalImage);
                }

                if g.ldegree(main_var) > bounds[main_var] {
                    failure_count += 1;
                    if failure_count > 2 {
                        // p is likely unlucky
                        debug!(
                            "Bad current image: gcd({},{}) mod {} under {:?} = {}",
                            a,
                            b,
                            a.ring(),
                            r,
                            g
                        );
                        return Err(GCDError::BadCurrentImage);
                    }
                    debug!("Degree too high");
                    continue 'find_root_sample;
                }

                // check if all the monomials of the image appear in the shape
                // if not, the original shape is bad
                for m in g.into_iter() {
                    if shape.iter().all(|(_, pow)| *pow != m.exponents[main_var]) {
                        debug!("Bad shape: terms missing");
                        return Err(GCDError::BadOriginalImage);
                    }
                }

                // set the coefficient of the scaling term in the gcd to 1
                let (_, d) = &shape[shape_map[0]];
                let mut found = false;
                for t in &g {
                    if t.exponents[main_var] == *d {
                        let scale_factor = g.ring().inv(t.coefficient);
                        g = g.mul_coeff(scale_factor);
                        found = true;
                        break;
                    }
                }

                if !found {
                    // the scaling term is missing, so the sample point is bad
                    debug!("Bad sample point: scaling term missing");
                    // TODO: check if this happen a number of times in a row
                    // as the prime may be too small to generate n samples that
                    // all contain the scaling term
                    continue 'find_root_sample;
                }

                // construct the right-hand side
                'rhs: for (i, (rhs, (shape_part, exp))) in samples.iter_mut().zip(shape).enumerate()
                {
                    // we may not need all terms
                    if solved_coeff.is_some() && rhs.len() == shape_part.nterms() {
                        continue;
                    }

                    // find the associated term in the sample, trying the usual place first
                    if i < g.nterms() && g.exponents(i)[main_var] == *exp {
                        rhs.push(g.coefficients[i].clone());
                    } else {
                        // find the matching term if it exists
                        for m in g.into_iter() {
                            if m.exponents[main_var] == *exp {
                                rhs.push(m.coefficient.clone());
                                continue 'rhs;
                            }
                        }

                        rhs.push(a.ring().zero());
                    }
                }

                // see if we have collected enough samples to solve for the scaling factor
                while solved_coeff.is_none() {
                    // try to solve the system!
                    let vars_scale = shape[shape_map[0]].0.nterms() - 1;
                    let vars_second = shape[shape_map[second_index]].0.nterms();
                    let samples_needed = vars_scale + vars_second;
                    let rows = samples_needed + scaling_var_relations.len();

                    if sample_index + 1 < samples_needed {
                        break; // obtain more samples
                    }

                    let mut gfm = Vec::with_capacity(rows * samples_needed);
                    let mut new_rhs = Vec::with_capacity(rows);

                    for sample_index in 0..samples_needed {
                        let rhs_sec = &samples[shape_map[second_index]][sample_index];
                        let row_eval_sec = &row_sample_values[shape_map[second_index]];
                        let row_eval_first = &row_sample_values[shape_map[0]];

                        // assume first constant is 1, which will form the rhs of our equation
                        let actual_rhs = a.ring().mul(
                            rhs_sec,
                            &a.ring().pow(&row_eval_first[0], sample_index as u64 + 1),
                        );

                        for aa in row_eval_sec {
                            gfm.push(a.ring().pow(aa, sample_index as u64 + 1));
                        }

                        // place the scaling term variables at the end
                        for aa in &row_eval_first[1..] {
                            gfm.push(
                                a.ring().neg(
                                    &a.ring()
                                        .mul(rhs_sec, &a.ring().pow(aa, sample_index as u64 + 1)),
                                ),
                            );
                        }

                        new_rhs.push(actual_rhs);
                    }

                    // add extra relations between the scaling term variables coming from previous tries
                    // that yielded underdetermined systems
                    for extra_relations in &scaling_var_relations {
                        for _ in 0..vars_second {
                            gfm.push(a.ring().zero());
                        }

                        for v in &extra_relations[..vars_scale] {
                            gfm.push(v.clone());
                        }
                        new_rhs.push(extra_relations.last().unwrap().clone());
                    }

                    let m = Matrix::from_linear(
                        gfm,
                        rows as u32,
                        samples_needed as u32,
                        a.ring().clone(),
                    )
                    .unwrap();
                    let rhs = Matrix::new_vec(new_rhs, a.ring().clone());

                    match m.solve(&rhs) {
                        Ok(r) => {
                            debug!("Solved {}x{} system", rows, samples_needed);
                            debug!(
                                "Solved with {} and {} term",
                                shape[shape_map[0]].0, shape[shape_map[second_index]].0
                            );

                            let mut r = r.into_vec();
                            r.drain(0..vars_second);
                            solved_coeff = Some(r);
                        }
                        Err(MatrixError::Underdetermined {
                            row_reduced_augmented_matrix,
                            ..
                        }) => {
                            // extract relations between the variables in the scaling term from the row reduced augmented matrix

                            debug!(
                                "Underdetermined system {} and {} term; row reduction={}, rhs={}",
                                shape[shape_map[0]].0,
                                shape[shape_map[second_index]].0,
                                row_reduced_augmented_matrix,
                                rhs
                            );

                            for x in row_reduced_augmented_matrix.row_iter() {
                                if x[..vars_second].iter().all(|x| a.ring().is_zero(x))
                                    && x.iter().any(|y| !a.ring().is_zero(y))
                                {
                                    scaling_var_relations.push(x[vars_second..].to_vec());
                                }
                            }

                            second_index += 1;
                            if second_index == shape.len() {
                                // the system remains underdetermined, that means the shape is bad
                                debug!(
                                    "Could not determine monomial scaling due to a bad shape\na={}\nb={}\na_ldegree={}, b_ldegree={}\nbounds={:?}, vars={:?}, main_var={},\nmat={}\nrhs={},\nshape=",
                                    a,
                                    b,
                                    a_ldegree,
                                    b_ldegree,
                                    bounds,
                                    vars,
                                    main_var,
                                    row_reduced_augmented_matrix,
                                    rhs
                                );
                                for s in shape {
                                    debug!("\t({}, {})", s.0, s.1);
                                }

                                return Err(GCDError::BadOriginalImage);
                            }
                        }
                        Err(MatrixError::Inconsistent) => {
                            debug!("Inconsistent system: bad shape");
                            return Err(GCDError::BadOriginalImage);
                        }
                        Err(
                            MatrixError::NotSquare
                            | MatrixError::ShapeMismatch
                            | MatrixError::RightHandSideIsNotVector
                            | MatrixError::Singular
                            | MatrixError::ResultNotInDomain,
                        ) => {
                            unreachable!()
                        }
                    }
                }
            }

            if let Some(r) = solved_coeff {
                // evaluate the scaling term for every sample
                let mut lcoeff_cache = Vec::with_capacity(max_terms);
                for sample_index in 0..max_terms {
                    let row_eval_first = &row_sample_values[shape_map[0]];
                    let mut scaling_factor =
                        a.ring().pow(&row_eval_first[0], sample_index as u64 + 1); // coeff eval is 1
                    for (exp_eval, coeff_eval) in
                        row_sample_values[shape_map[0]][1..].iter().zip(&r)
                    {
                        a.ring().add_mul_assign(
                            &mut scaling_factor,
                            coeff_eval,
                            &a.ring().pow(exp_eval, sample_index as u64 + 1),
                        );
                    }

                    debug!(
                        "Scaling fac {}: {}",
                        sample_index,
                        a.ring().printer(&scaling_factor)
                    );
                    lcoeff_cache.push(scaling_factor);
                }

                for ((c, _), rhs) in shape.iter().zip(&mut samples) {
                    rhs.truncate(c.nterms()); // drop unneeded samples
                    for (r, scale) in rhs.iter_mut().zip(&lcoeff_cache) {
                        a.ring().mul_assign(r, scale);
                    }
                }
            } else {
                debug!(
                    "Could not solve the system with just 2 terms: a={}, b={}",
                    a, b
                );
            }

            break (row_sample_values, samples);
        };

        debug!("VDM with {} samples", samples.len());
        Ok(a.solve_vandermonde(main_var, shape, row_sample_values, samples))
    }
}

impl<
    F: Field + PolynomialGCD<E> + SampleableRing<SamplingPolicy = RangeInclusive<i64>>,
    E: PositiveExponent,
> MultivariatePolynomial<F, E>
{
    /// Compute the gcd shape of two polynomials in a finite field by filling in random
    /// numbers.
    #[instrument(level = "debug", skip_all)]
    fn gcd_shape_modular(
        a: &Self,
        b: &Self,
        vars: &[usize],         // variables
        bounds: &mut [E],       // degree bounds
        tight_bounds: &mut [E], // tighter degree bounds
    ) -> Option<Self> {
        let lastvar = *vars.last().unwrap();
        debug!("GCD shape modular: vars={vars:?} bounds={bounds:?}");
        debug_assert!(
            (lastvar + 1..a.nvars())
                .all(|variable| a.degree(variable).is_zero() && b.degree(variable).is_zero())
        );

        // if we are in the univariate case, return the univariate gcd
        // TODO: this is a modification of the algorithm!
        if vars.len() == 1 {
            let gg = a.univariate_gcd(b);
            if gg.degree(vars[0]) > bounds[vars[0]] {
                debug!(
                    "Unexpectedly high GCD bound: {} vs {}",
                    gg.degree(vars[0]),
                    bounds[vars[0]]
                );
                return None;
            }
            bounds[vars[0]] = gg.degree(vars[0]); // update degree bound
            return Some(gg);
        }

        // the gcd of the content in the last variable should be 1
        let c = a.multivariate_content_gcd(b, lastvar);
        if !c.is_one() {
            debug!("Content in last variable is not 1, but {}", c);
            // TODO: we assume that a content of -1 is also allowed
            // like in the special case gcd_(-x0*x1,-x0-x0*x1)
            if c.nterms() != 1 || c.coefficients[0] != a.ring().neg(&a.ring().one()) {
                return None;
            }
        }

        let gamma = a
            .lcoeff_last_varorder(vars)
            .univariate_gcd(&b.lcoeff_last_varorder(vars));
        let expected_evaluations = (tight_bounds[lastvar].to_u32() as usize)
            .saturating_add(gamma.ldegree_max().to_u32() as usize)
            .saturating_add(1);
        let mut evaluation_context =
            RepeatedLastVariableEvaluationContext::new(a, b, lastvar, expected_evaluations);

        let mut rng = rand::rng();

        let mut failure_count = 0;

        'newfirstnum: loop {
            // if we had two failures, it may be that the tight degree bound
            // was too tight due to an unfortunate prime/evaluation, so we relax it
            if failure_count == 2 {
                debug!(
                    "Changing tight bound for x{} from {} to {}",
                    lastvar, tight_bounds[lastvar], bounds[lastvar]
                );
                tight_bounds[lastvar] = bounds[lastvar];
                let relaxed_evaluations = (tight_bounds[lastvar].to_u32() as usize)
                    .saturating_add(gamma.ldegree_max().to_u32() as usize)
                    .saturating_add(1);
                if relaxed_evaluations > 1 {
                    evaluation_context.enable_reuse();
                }
            }
            failure_count += 1;

            if let Some(size) = a.ring().size()
                && failure_count * 2 > size
            {
                debug!("Cannot find unique sampling points: prime field is likely too small");
                return None;
            }

            let mut sample_fail_count = 0i64;
            let (v, gamma_value) = loop {
                let r = sample_nonzero_field_element(a.ring(), &mut rng);
                let gamma_value = gamma.evaluate_univariate_horner(lastvar, &r);
                if !gamma.ring().is_zero(&gamma_value) {
                    break (r, gamma_value);
                }

                sample_fail_count += 1;
                if let Some(size) = a.ring().size()
                    && sample_fail_count * 2 > size
                {
                    debug!("Cannot find unique sampling points: prime field is likely too small");
                    continue 'newfirstnum;
                }
            };

            debug!("Chosen variable: {}", a.ring().printer(&v));

            // performance dense reconstruction
            let mut gv = evaluation_context.with_evaluations(&v, |av, bv| {
                if vars.len() > 2 {
                    MultivariatePolynomial::gcd_shape_modular(
                        av,
                        bv,
                        &vars[..vars.len() - 1],
                        bounds,
                        tight_bounds,
                    )
                } else {
                    let gg = av.univariate_gcd(bv);
                    if gg.degree(vars[0]) > bounds[vars[0]] {
                        debug!(
                            "Unexpectedly high GCD bound: {} vs {}",
                            gg.degree(vars[0]),
                            bounds[vars[0]]
                        );
                        return None;
                    }
                    bounds[vars[0]] = gg.degree(vars[0]); // update degree bound
                    Some(gg)
                }
            })?;

            debug!(
                "GCD shape suggestion for sample point {} and gamma {}: {}",
                a.ring().printer(&v),
                gamma,
                gv
            );

            // construct a new assumed form
            let gfu = gv.to_univariate_polynomial_list(vars[0]);

            // find a coefficient of x1 in gg that is a monomial (single scaling)
            let mut single_scale = None;
            let mut nx = 0; // count the minimal number of samples needed
            for (i, (c, _e)) in gfu.iter().enumerate() {
                if c.nterms() > nx {
                    nx = c.nterms();
                }
                if c.nterms() == 1 {
                    single_scale = Some(i);
                }
            }

            // In the case of multiple scaling, each sample adds an
            // additional unknown, except for the first
            if single_scale.is_none() {
                let mut nx1 = (gv.nterms() - 1) / (gfu.len() - 1);
                if (gv.nterms() - 1) % (gfu.len() - 1) != 0 {
                    nx1 += 1;
                }
                if nx < nx1 {
                    nx = nx1;
                }
                debug!("Multiple scaling case: sample {} times", nx);
            }

            let mut lc = gv.lcoeff_varorder(vars);

            let mut gseq = vec![gv.clone().mul_coeff(gamma.ring().div(&gamma_value, &lc))];
            let mut vseq = vec![v];

            // sparse reconstruction
            debug!(
                "Sparse reconstruction to bound {} + {}",
                tight_bounds[lastvar],
                gamma.ldegree_max()
            );
            'newnum: loop {
                if gseq.len()
                    == (tight_bounds[lastvar].to_u32() + gamma.ldegree_max().to_u32() + 1) as usize
                {
                    break;
                }

                let (v, gamma_value) = loop {
                    let v = sample_nonzero_field_element(a.ring(), &mut rng);
                    let gamma_value = gamma.evaluate_univariate_horner(lastvar, &v);
                    if !gamma.ring().is_zero(&gamma_value) {
                        // we need unique sampling points
                        if !vseq.contains(&v) {
                            break (v, gamma_value);
                        }
                    }

                    sample_fail_count += 1;
                    if let Some(size) = a.ring().size()
                        && sample_fail_count * 2 > size
                    {
                        debug!(
                            "Cannot find unique sampling points: prime field is likely too small"
                        );
                        continue 'newfirstnum;
                    }
                };

                let rec = evaluation_context.with_evaluations(&v, |av, bv| {
                    if let Some(single_scale) = single_scale {
                        Self::construct_new_image_single_scale(
                            av,
                            bv,
                            av.degree(vars[0]),
                            bv.degree(vars[0]),
                            bounds,
                            single_scale,
                            &vars[1..vars.len() - 1],
                            vars[0],
                            &gfu,
                        )
                    } else {
                        Self::construct_new_image_multiple_scales(
                            av,
                            bv,
                            // NOTE: different from paper where they use a.degree(..)
                            // it could be that the degree in av is lower than that of a
                            // which means the sampling will never terminate
                            av.degree(vars[0]),
                            bv.degree(vars[0]),
                            bounds,
                            &vars[1..vars.len() - 1],
                            vars[0],
                            &gfu,
                        )
                    }
                });

                match rec {
                    Ok(r) => {
                        gv = r;
                    }
                    Err(GCDError::BadOriginalImage) => {
                        debug!("Bad original image");
                        continue 'newfirstnum;
                    }
                    Err(GCDError::BadCurrentImage) => {
                        debug!("Bad current image");
                        sample_fail_count += 1;

                        if let Some(size) = a.ring().size()
                            && sample_fail_count * 2 > size
                        {
                            debug!("Too many bad current images: prime field is likely too small");
                            continue 'newfirstnum;
                        }

                        continue 'newnum;
                    }
                }

                lc = gv.lcoeff_varorder(vars);

                gseq.push(gv.clone().mul_coeff(gamma.ring().div(&gamma_value, &lc)));
                vseq.push(v);
            }

            // use interpolation to construct x_n dependence
            let mut gc = Self::newton_interpolation(&vseq, &gseq, lastvar);
            // remove content in x_n (wrt all other variables)
            let cont = gc.multivariate_content(lastvar);
            if !cont.is_one() {
                debug!("Removing content in x{}: {}", lastvar, cont);
                gc = gc.try_div(&cont).unwrap();
            }

            // do a probabilistic division test
            let (g1, a1, b1) = loop {
                // store a table for variables raised to a certain power
                let mut cache = (0..a.nvars())
                    .map(|i| {
                        vec![
                            a.ring().zero();
                            min(
                                max(a.degree(i), b.degree(i)).to_u32() as usize + 1,
                                POW_CACHE_SIZE
                            )
                        ]
                    })
                    .collect::<Vec<_>>();

                let r: Vec<_> = vars
                    .iter()
                    .skip(1)
                    .map(|i| (*i, sample_nonzero_field_element(a.ring(), &mut rng)))
                    .collect();

                let g1 = gc.replace_except(vars[0], &r, &mut cache);

                if g1.ldegree(vars[0]) == gc.degree(vars[0]) {
                    let a1 = a.replace_except(vars[0], &r, &mut cache);
                    let b1 = b.replace_except(vars[0], &r, &mut cache);
                    break (g1, a1, b1);
                }
            };

            if g1.is_one() || (a1.try_div(&g1).is_some() && b1.try_div(&g1).is_some()) {
                return Some(gc);
            }

            // if the gcd is bad, we had a bad number
            debug!(
                "Division test failed: gcd may be bad or probabilistic division test is unlucky: a1 {} b1 {} g1 {}",
                a1, b1, g1
            );
        }
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent> MultivariatePolynomial<R, E> {
    /// Get the content of a multivariate polynomial viewed as a
    /// univariate polynomial in `x`.
    pub fn univariate_content(&self, x: usize) -> MultivariatePolynomial<R, E> {
        let a = self.to_univariate_polynomial_list(x);

        let mut f = Vec::with_capacity(a.len());
        for (c, _) in a {
            f.push(c);
        }

        PolynomialGCD::gcd_multiple(f)
    }

    /// Get the content of a multivariate polynomial viewed as a
    /// univariate polynomial in `x` and `y`.
    pub fn bivariate_content(&self, x: usize, y: usize) -> MultivariatePolynomial<R, E> {
        let af = self.to_multivariate_polynomial_list(&[x, y], true);
        PolynomialGCD::gcd_multiple(af.into_values().collect())
    }

    /// Get the content of a multivariate polynomial viewed as a
    /// multivariate polynomial in all variables except `x`.
    pub fn multivariate_content(&self, x: usize) -> MultivariatePolynomial<R, E> {
        let af = self.to_multivariate_polynomial_list(&[x], false);
        PolynomialGCD::gcd_multiple(af.into_values().collect())
    }

    /// Compute the gcd of the univariate content in `x`.
    pub fn univariate_content_gcd(
        &self,
        b: &MultivariatePolynomial<R, E>,
        x: usize,
    ) -> MultivariatePolynomial<R, E> {
        let af = self.to_univariate_polynomial_list(x);
        let bf = b.to_univariate_polynomial_list(x);

        let mut f = Vec::with_capacity(af.len() + bf.len());
        for (c, _) in af.into_iter().chain(bf.into_iter()) {
            f.push(c);
        }

        PolynomialGCD::gcd_multiple(f)
    }

    /// Get the GCD of the contents of a polynomial and another one,
    /// viewed as a multivariate polynomial in all variables except `x`.
    pub fn multivariate_content_gcd(
        &self,
        b: &MultivariatePolynomial<R, E>,
        x: usize,
    ) -> MultivariatePolynomial<R, E> {
        let af = self.to_multivariate_polynomial_list(&[x], false);
        let bf = b.to_multivariate_polynomial_list(&[x], false);

        let f = af.into_values().chain(bf.into_values()).collect();

        PolynomialGCD::gcd_multiple(f)
    }

    /// Apply a GCD repeatedly to a list of polynomials.
    #[inline(always)]
    pub fn repeated_gcd(mut f: Vec<MultivariatePolynomial<R, E>>) -> MultivariatePolynomial<R, E> {
        if f.len() == 1 {
            return f.swap_remove(0);
        }

        if f.len() == 2 {
            return f[0].gcd(&f[1]);
        }

        f.sort_unstable_by_key(|p| p.nterms());

        let mut gcd = f.pop().unwrap();
        for p in f {
            if R::one_is_gcd_unit() && gcd.is_one() {
                return gcd;
            }

            gcd = gcd.gcd(&p);
        }
        gcd
    }

    /// Compute a standard GCD-free basis. The input should not
    /// contain 0 or units.
    pub fn gcd_free_basis(mut polys: Vec<Self>) -> Vec<Self> {
        let mut i = 0;
        while i + 1 < polys.len() {
            if polys[i].is_one() {
                i += 1;
                continue;
            }

            let mut j = i + 1;
            while j < polys.len() {
                if polys[j].is_one() {
                    j += 1;
                    continue;
                }

                let g = polys[i].gcd(&polys[j]);
                if !g.is_one() {
                    polys[i] = &polys[i] / &g;
                    polys[j] = &polys[j] / &g;
                    polys.push(g);
                }

                j += 1;
            }

            i += 1;
        }

        polys.retain(|p| !p.is_one());
        polys
    }

    /// Compute the GCD for simple cases.
    #[inline(always)]
    fn simple_gcd(&self, b: &MultivariatePolynomial<R, E>) -> Option<MultivariatePolynomial<R, E>> {
        if self == b {
            return Some(self.clone());
        }

        if self.is_zero() {
            return Some(b.clone());
        }
        if b.is_zero() {
            return Some(self.clone());
        }

        if self.is_one() {
            return Some(self.clone());
        }

        if b.is_one() {
            return Some(b.clone());
        }

        if self.is_constant() {
            let mut gcd = self.coefficients[0].clone();
            for c in &b.coefficients {
                gcd = self.ring().gcd(&gcd, c);
                if R::one_is_gcd_unit() && self.ring().is_one(&gcd) {
                    break;
                }
            }
            return Some(self.constant(gcd));
        }

        if b.is_constant() {
            let mut gcd = b.coefficients[0].clone();
            for c in &self.coefficients {
                gcd = self.ring().gcd(&gcd, c);
                if R::one_is_gcd_unit() && self.ring().is_one(&gcd) {
                    break;
                }
            }
            return Some(self.constant(gcd));
        }

        None
    }

    /// Compute the gcd of two multivariate polynomials.
    #[instrument(skip_all)]
    pub fn gcd(&self, b: &MultivariatePolynomial<R, E>) -> MultivariatePolynomial<R, E> {
        debug!("gcd of {} and {}", self, b);

        if let Some(g) = self.simple_gcd(b) {
            debug!("Simple {} ", g);
            return PolynomialGCD::normalize(g);
        }

        // a and b are only copied when needed
        let mut a = Cow::Borrowed(self);
        let mut b = Cow::Borrowed(b);

        if self.variables() != b.variables() {
            a.to_mut().unify_variables(b.to_mut());
        }

        let a_metadata = GcdInputMetadata::scan(&a);
        let b_metadata = GcdInputMetadata::scan(&b);

        // Retain the common part of the two monomial factors for the result,
        // then remove each input's complete monomial factor before computing
        // the polynomial part of the GCD.
        let shared_degree: SmallVec<[E; INLINED_EXPONENTS]> = a_metadata
            .variables
            .iter()
            .zip(&b_metadata.variables)
            .map(|(left, right)| left.min_degree.min(right.min_degree))
            .collect();
        a_metadata.remove_monomial_shift(&mut a);
        b_metadata.remove_monomial_shift(&mut b);

        let mut base_degree: SmallVec<[Option<E>; INLINED_EXPONENTS]> = smallvec![None; a.nvars()];

        if let Some(g) = MultivariatePolynomial::simple_gcd(&a, &b) {
            return rescale_gcd(g, &shared_degree, &base_degree, &a.constant(a.ring().one()));
        }

        // check if the polynomials are functions of x^n, n > 1
        let mut unresolved_base_degrees = a.nvars();
        'base_degrees: for p in [&a, &b] {
            for t in p.into_iter() {
                for (md, v) in base_degree.iter_mut().zip(t.exponents) {
                    if !v.is_zero() {
                        if let Some(mm) = md.as_mut() {
                            if *mm != E::one() {
                                *mm = mm.gcd(v);
                                if *mm == E::one() {
                                    unresolved_base_degrees -= 1;
                                }
                            }
                        } else {
                            *md = Some(*v);
                            if *v == E::one() {
                                unresolved_base_degrees -= 1;
                            }
                        }
                    }
                }

                if unresolved_base_degrees == 0 {
                    break 'base_degrees;
                }
            }
        }

        // rename x^base_deg to x
        if base_degree
            .iter()
            .any(|d| d.is_some() && d.unwrap() > E::one())
        {
            let aa = a.to_mut();
            for e in aa.exponents_iter_mut() {
                for (v, d) in e.iter_mut().zip(&base_degree) {
                    if let Some(d) = d {
                        *v = *v / *d;
                    }
                }
            }

            let bb = b.to_mut();
            for e in bb.exponents_iter_mut() {
                for (v, d) in e.iter_mut().zip(&base_degree) {
                    if let Some(d) = d {
                        *v = *v / *d;
                    }
                }
            }
        }

        /// Undo simplifications made to the input polynomials and normalize the gcd.
        #[inline(always)]
        fn rescale_gcd<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent>(
            mut g: MultivariatePolynomial<R, E>,
            shared_degree: &[E],
            base_degree: &[Option<E>],
            content: &MultivariatePolynomial<R, E>,
        ) -> MultivariatePolynomial<R, E> {
            if !content.is_one() {
                g = g * content;
            }

            if shared_degree.iter().any(|d| *d > E::from_u32(0))
                || base_degree
                    .iter()
                    .any(|d| d.map(|bd| bd > E::one()).unwrap_or(false))
            {
                for e in g.exponents_iter_mut() {
                    for ((v, d), s) in e.iter_mut().zip(base_degree).zip(shared_degree) {
                        if let Some(d) = d {
                            *v = *v * *d;
                        }

                        *v += *s;
                    }
                }
            }

            PolynomialGCD::normalize(g)
        }

        if let Some(gcd) = PolynomialGCD::heuristic_gcd(&a, &b) {
            debug!("Heuristic gcd succeeded: {}", gcd.0);
            return rescale_gcd(
                gcd.0,
                &shared_degree,
                &base_degree,
                &a.constant(a.ring().one()),
            );
        }

        // store which variables appear in which expression
        let scratch: SmallVec<[i32; INLINED_EXPONENTS]> = (0..a.nvars())
            .map(|variable| {
                i32::from(a_metadata.occurs_after_shift(variable))
                    | i32::from(b_metadata.occurs_after_shift(variable)) << 1
            })
            .collect();

        if a == b {
            debug!("Equal {} ", a);
            return rescale_gcd(a.into_owned(), &shared_degree, &base_degree, &b.one());
        }

        // compute the gcd efficiently if some variables do not occur in both
        // polynomials
        if scratch.iter().any(|x| *x > 0 && *x < 3) {
            let inca: SmallVec<[_; INLINED_EXPONENTS]> = scratch
                .iter()
                .enumerate()
                .filter_map(|(i, v)| if *v == 1 || *v == 3 { Some(i) } else { None })
                .collect();

            let incb: SmallVec<[_; INLINED_EXPONENTS]> = scratch
                .iter()
                .enumerate()
                .filter_map(|(i, v)| if *v == 2 || *v == 3 { Some(i) } else { None })
                .collect();

            // extract the variables of b in the coefficient of a and vice versa
            let a1 = a.to_multivariate_polynomial_list(&incb, false);
            let b1 = b.to_multivariate_polynomial_list(&inca, false);

            let f = a1.into_values().chain(b1.into_values()).collect();

            return rescale_gcd(
                PolynomialGCD::gcd_multiple(f),
                &shared_degree,
                &base_degree,
                &a.one(),
            );
        }

        // try if b divides a or vice versa, doing a heuristical length check first
        if a.nterms() >= b.nterms() && a.try_div(&b).is_some() {
            return rescale_gcd(b.into_owned(), &shared_degree, &base_degree, &a.one());
        }
        if a.nterms() <= b.nterms() && b.try_div(&a).is_some() {
            return rescale_gcd(a.into_owned(), &shared_degree, &base_degree, &b.one());
        }

        // check if the polynomial is linear in a variable and compute the gcd using the univariate content
        for (p1, p2, metadata) in [(&a, &b, &a_metadata), (&b, &a, &b_metadata)] {
            if let Some(var) = (0..p1.nvars()).find(|v| {
                let degree = metadata.shifted_degree(*v);
                base_degree[*v].is_some_and(|base| degree / base == E::one())
            }) {
                let mut cont = p1.univariate_content(var);

                let p1_prim = p1.as_ref() / &cont;

                if !cont.is_one() || !R::one_is_gcd_unit() {
                    let cont_p2 = p2.univariate_content(var);
                    cont = cont.gcd(&cont_p2);
                }

                if p2.try_div(&p1_prim).is_some() {
                    return rescale_gcd(p1_prim, &shared_degree, &base_degree, &cont);
                } else {
                    return rescale_gcd(
                        cont,
                        &shared_degree,
                        &base_degree,
                        &p1.constant(p1.ring().one()),
                    );
                }
            }
        }

        let mut vars: SmallVec<[_; INLINED_EXPONENTS]> = scratch
            .iter()
            .enumerate()
            .filter_map(|(i, v)| if *v == 3 { Some(i) } else { None })
            .collect();

        // find upper bounds for all variables
        let mut bounds = R::get_gcd_var_bounds(&a, &b, &vars);

        // if all bounds are 0, the gcd is a constant
        if bounds.iter().all(|x| x.is_zero()) {
            return rescale_gcd(
                a.constant(a.ring().gcd(&a.content(), &b.content())),
                &shared_degree,
                &base_degree,
                &a.one(),
            );
        }

        // if some variables do not appear in the gcd, split the polynomials in these variables
        if bounds.iter().any(|x| x.is_zero()) {
            let zero_bound: SmallVec<[_; INLINED_EXPONENTS]> = bounds
                .iter()
                .enumerate()
                .filter_map(|(i, v)| {
                    if *v == E::zero() && a_metadata.occurs_after_shift(i) {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect();

            if !zero_bound.is_empty() {
                let a1 = a.to_multivariate_polynomial_list(&zero_bound, true);
                let b1 = b.to_multivariate_polynomial_list(&zero_bound, true);

                let f = a1.into_values().chain(b1.into_values()).collect();

                return rescale_gcd(
                    PolynomialGCD::gcd_multiple(f),
                    &shared_degree,
                    &base_degree,
                    &a.one(),
                );
            }
        }

        // Determine a good variable ordering
        let first_variable_index = (0..vars.len())
            .min_by_key(|i| {
                let var = vars[*i];
                let max_terms = a
                    .terms_with_max_degree(var)
                    .min(b.terms_with_max_degree(var));
                debug!("{var}: bounds: {}, max terms {}", bounds[var], max_terms);
                bounds[var].to_u32() as usize + max_terms
            })
            .unwrap();
        vars.swap(0, first_variable_index);
        vars[1..].sort_by(|&i, &j| bounds[j].cmp(&bounds[i])); // sort descending
        debug!("Order: {:?}", vars);

        let normalized_degrees = |metadata: &GcdInputMetadata<E>| {
            metadata
                .variables
                .iter()
                .zip(&base_degree)
                .map(|(variable, base)| {
                    base.map_or(variable.max_degree - variable.min_degree, |base| {
                        (variable.max_degree - variable.min_degree) / base
                    })
                })
                .collect::<SmallVec<[E; INLINED_EXPONENTS]>>()
        };
        let a_degrees = normalized_degrees(&a_metadata);
        let b_degrees = normalized_degrees(&b_metadata);
        if let Some(g) = PolynomialGCD::gcd_with_precontent_plan(
            a.as_ref(),
            b.as_ref(),
            &vars,
            &bounds,
            &a_degrees,
            &b_degrees,
        ) {
            return rescale_gcd(g, &shared_degree, &base_degree, &a.one());
        }

        // strip the gcd of the univariate contents wrt the new first variable
        let content = if vars.len() > 1 {
            let c_a = a.univariate_content(vars[0]);
            let c_b = b.univariate_content(vars[0]);
            let c_g = c_a.gcd(&c_b);

            debug!("GCD of content: {}", c_g);

            if !c_a.is_one() {
                a = Cow::Owned(a.as_ref() / &c_a);
            }

            if !c_b.is_one() {
                b = Cow::Owned(b.as_ref() / &c_b);
            }

            // TODO: lower bounds?
            // for (bound, content_degree) in bounds.iter_mut().zip(0..content.nvars()) {
            //     let content_degree = content.degree(content_degree);
            //     *bound = if *bound > content_degree {
            //         *bound - content_degree
            //     } else {
            //         E::zero()
            //     };
            // }

            // even if variables got removed, benchmarks show that it is not
            // worth it do restart the gcd computation
            c_g
        } else {
            // get the integer content for univariate polynomials
            let uca = a.content();
            let ucb = b.content();
            let content = a.ring().gcd(&a.content(), &b.content());
            let p = a.zero_with_capacity(1);

            if !a.ring().is_one(&uca) {
                a = Cow::Owned(a.into_owned().div_coeff(&uca));
            }
            if !a.ring().is_one(&ucb) {
                b = Cow::Owned(b.into_owned().div_coeff(&ucb));
            }

            p.add_constant(content)
        };

        let rearrange = vars.len() > 1 && vars.windows(2).any(|s| s[0] > s[1]);
        if rearrange {
            debug!("Rearranging variables with map: {:?}", vars);
            a = Cow::Owned(a.rearrange_impl(&vars, false, false));
            b = Cow::Owned(b.rearrange_impl(&vars, false, false));

            let mut newbounds: SmallVec<[_; INLINED_EXPONENTS]> =
                smallvec![E::zero(); bounds.len()];
            for x in 0..vars.len() {
                newbounds[x] = bounds[vars[x]];
            }
            bounds = newbounds;
        }

        let mut g = PolynomialGCD::gcd(
            &a,
            &b,
            &if rearrange {
                Cow::Owned((0..vars.len()).collect::<SmallVec<[usize; INLINED_EXPONENTS]>>())
            } else {
                Cow::Borrowed(&vars)
            },
            &mut bounds,
        );

        if rearrange {
            g = g.rearrange_impl(&vars, true, false);
        }

        rescale_gcd(g, &shared_degree, &base_degree, &content)
    }
}

/// An error that can occur during the heuristic GCD algorithm.
#[derive(Debug)]
pub enum HeuristicGCDError {
    MaxSizeExceeded,
    BadReconstruction,
}

#[inline]
fn ceil_log2_usize(value: usize) -> u64 {
    if value <= 1 {
        0
    } else {
        (usize::BITS - (value - 1).leading_zeros()) as u64
    }
}

/// Bounds the largest coefficient produced by recursively substituting every shared variable.
fn estimated_heuristic_gcd_evaluation_bits<E: PositiveExponent>(
    a: &MultivariatePolynomial<IntegerRing, E>,
    b: &MultivariatePolynomial<IntegerRing, E>,
) -> u64 {
    let statistics = |polynomial: &MultivariatePolynomial<IntegerRing, E>| {
        let mut coefficient_bits = 0u64;
        let mut degrees = vec![0u64; polynomial.nvars()];
        for term in polynomial {
            coefficient_bits = coefficient_bits.max(term.coefficient.significant_bits());
            for (degree, exponent) in degrees.iter_mut().zip(term.exponents) {
                *degree = (*degree).max(exponent.to_u32() as u64);
            }
        }
        (
            coefficient_bits,
            degrees,
            ceil_log2_usize(polynomial.nterms()),
        )
    };
    let (mut a_bits, a_degrees, a_sum_bits) = statistics(a);
    let (mut b_bits, b_degrees, b_sum_bits) = statistics(b);

    for (a_degree, b_degree) in a_degrees.into_iter().zip(b_degrees) {
        if a_degree == 0 || b_degree == 0 {
            continue;
        }

        // `xi = 2*min(max_coeff(a), max_coeff(b)) + 29`; two extra bits bound the
        // addition, including the small-coefficient case.
        let xi_bits = a_bits.min(b_bits).saturating_add(2).max(6);
        a_bits = a_bits
            .saturating_add(xi_bits.saturating_mul(a_degree))
            .saturating_add(a_sum_bits);
        b_bits = b_bits
            .saturating_add(xi_bits.saturating_mul(b_degree))
            .saturating_add(b_sum_bits);
    }

    a_bits.max(b_bits)
}

/// Computes a normalized univariate GCD image in a dense `Zp64` coefficient workspace.
struct DenseZp64UnivariateGcdImage<'a, E: PositiveExponent> {
    field: &'a Zp64,
    prototype: &'a MultivariatePolynomial<IntegerRing, E>,
    variable: usize,
    left: Vec<FiniteFieldElement<u64>>,
    right: Vec<FiniteFieldElement<u64>>,
}

impl<'a, E: PositiveExponent> DenseZp64UnivariateGcdImage<'a, E> {
    /// Converts both integer polynomials directly to bounded degree-indexed field coefficients.
    fn new(
        left: &'a MultivariatePolynomial<IntegerRing, E>,
        right: &MultivariatePolynomial<IntegerRing, E>,
        variable: usize,
        field: &'a Zp64,
    ) -> Option<Self> {
        if left.is_zero()
            || right.is_zero()
            || left.nvars() != right.nvars()
            || variable >= left.nvars()
            || left
                .exponents_iter()
                .chain(right.exponents_iter())
                .any(|exponents| {
                    exponents
                        .iter()
                        .enumerate()
                        .any(|(index, exponent)| index != variable && !exponent.is_zero())
                })
        {
            return None;
        }

        let coefficient_count =
            left.degree(variable).max(right.degree(variable)).to_u32() as usize + 1;
        if coefficient_count > DENSE_UNIVARIATE_GCD_MAX_COEFFICIENTS
            || coefficient_count
                > left
                    .nterms()
                    .saturating_add(right.nterms())
                    .saturating_mul(DENSE_UNIVARIATE_GCD_MAX_SPARSITY_RATIO)
        {
            return None;
        }

        let coefficients = |polynomial: &MultivariatePolynomial<IntegerRing, E>| {
            let degree = polynomial.degree(variable).to_u32() as usize;
            let mut coefficients = vec![field.zero(); degree + 1];
            for term in polynomial {
                coefficients[term.exponents[variable].to_u32() as usize] =
                    term.coefficient.to_finite_field(field);
            }
            coefficients
        };
        let left_coefficients = coefficients(left);
        let right_coefficients = coefficients(right);
        if left_coefficients
            .last()
            .is_none_or(|coefficient| field.is_zero(coefficient))
            || right_coefficients
                .last()
                .is_none_or(|coefficient| field.is_zero(coefficient))
        {
            return None;
        }

        Some(Self {
            field,
            prototype: left,
            variable,
            left: left_coefficients,
            right: right_coefficients,
        })
    }

    /// Computes the Montgomery inverse used for the leading coefficient in dense division.
    #[inline]
    fn inverse_leading(
        field: &Zp64,
        coefficient: &FiniteFieldElement<u64>,
    ) -> FiniteFieldElement<u64> {
        debug_assert!(!field.is_zero(coefficient));

        let raw_one = FiniteFieldElement::from_inner(1);
        let residue = *field
            .mul(&field.mul(coefficient, &raw_one), &raw_one)
            .inner();
        let modulus = field.get_prime();

        // These are the Euclidean state after the known initial zero-quotient step for
        // `residue < modulus`. The coefficient magnitudes alternate signs at every step.
        let mut u1 = 0u64;
        let mut v1 = 1u64;
        let mut u3 = modulus;
        let mut v3 = residue;
        let mut u1_is_positive = false;

        while v3 != 0 {
            debug_assert!(u3 > v3);
            let first_remainder = u3 - v3;
            let (next_coefficient, remainder) = if first_remainder < v3 {
                (u1 + v1, first_remainder)
            } else {
                let second_remainder = first_remainder - v3;
                if second_remainder < v3 {
                    (u1 + 2 * v1, second_remainder)
                } else {
                    let third_remainder = second_remainder - v3;
                    if third_remainder < v3 {
                        (u1 + 3 * v1, third_remainder)
                    } else {
                        let quotient = u3 / v3;
                        (u1 + quotient * v1, u3 - quotient * v3)
                    }
                }
            };

            u1 = v1;
            v1 = next_coefficient;
            u3 = v3;
            v3 = remainder;
            u1_is_positive = !u1_is_positive;
        }

        debug_assert_eq!(u3, 1);
        FiniteFieldElement::from_inner(if u1_is_positive { u1 } else { modulus - u1 })
    }

    /// Replaces `dividend` by its remainder and returns the inverse of the divisor leading term.
    fn remainder(
        field: &Zp64,
        dividend: &mut Vec<FiniteFieldElement<u64>>,
        divisor: &[FiniteFieldElement<u64>],
    ) -> FiniteFieldElement<u64> {
        debug_assert!(!divisor.is_empty());
        debug_assert!(dividend.len() >= divisor.len());
        let divisor_degree = divisor.len() - 1;
        let inverse_leading = Self::inverse_leading(field, divisor.last().unwrap());

        for degree in (divisor_degree..dividend.len()).rev() {
            let leading = std::mem::replace(&mut dividend[degree], field.zero());
            if field.is_zero(&leading) {
                continue;
            }

            let quotient = field.mul(&leading, &inverse_leading);
            let shift = degree - divisor_degree;
            for (coefficient, divisor_coefficient) in dividend[shift..degree]
                .iter_mut()
                .zip(&divisor[..divisor_degree])
            {
                field.sub_mul_assign(coefficient, divisor_coefficient, &quotient);
            }
        }

        dividend.truncate(divisor_degree);
        while dividend
            .last()
            .is_some_and(|coefficient| field.is_zero(coefficient))
        {
            dividend.pop();
        }
        inverse_leading
    }

    /// Encodes dense coefficients as a polynomial in the retained input variable.
    fn polynomial(
        &self,
        coefficients: Vec<FiniteFieldElement<u64>>,
    ) -> MultivariatePolynomial<Zp64, E> {
        let capacity = coefficients
            .iter()
            .filter(|coefficient| !self.field.is_zero(coefficient))
            .count();
        let mut result = MultivariatePolynomial::new(
            self.field,
            Some(capacity),
            self.prototype.variables().clone(),
        );
        let mut exponents = vec![E::zero(); self.prototype.nvars()];
        for (degree, coefficient) in coefficients.into_iter().enumerate() {
            if !self.field.is_zero(&coefficient) {
                exponents[self.variable] = E::from_u32(degree as u32);
                result.append_monomial_back(coefficient, &exponents);
            }
        }
        result
    }

    /// Runs Euclid's algorithm and gives the result the requested leading coefficient.
    fn run(
        mut self,
        leading_coefficient: FiniteFieldElement<u64>,
    ) -> MultivariatePolynomial<Zp64, E> {
        debug_assert!(!self.field.is_zero(&leading_coefficient));
        if self.left.len() < self.right.len() {
            mem::swap(&mut self.left, &mut self.right);
        }

        loop {
            if self.right.len() == 1 {
                return self.polynomial(vec![leading_coefficient]);
            }

            let inverse_leading = Self::remainder(self.field, &mut self.left, &self.right);
            if self.left.is_empty() {
                let scale = self.field.mul(&leading_coefficient, &inverse_leading);
                for coefficient in &mut self.right {
                    self.field.mul_assign(coefficient, &scale);
                }
                let coefficients = mem::take(&mut self.right);
                return self.polynomial(coefficients);
            }
            mem::swap(&mut self.left, &mut self.right);
        }
    }
}

/// Dense integer long division used to certify a reconstructed univariate GCD.
struct DenseUnivariateIntegerDivisionContext {
    variable: usize,
    coefficients: Vec<MultiPrecisionInteger>,
    degrees: Vec<usize>,
    degree: usize,
    division_remainder: MultiPrecisionInteger,
}

impl DenseUnivariateIntegerDivisionContext {
    /// Construct a dense division workspace for a divisor and the two inputs it must divide.
    fn new<E: PositiveExponent>(
        divisor: &MultivariatePolynomial<IntegerRing, E>,
        left: &MultivariatePolynomial<IntegerRing, E>,
        right: &MultivariatePolynomial<IntegerRing, E>,
        variable: usize,
    ) -> Option<Self> {
        if variable >= divisor.nvars()
            || divisor.variables() != left.variables()
            || divisor.variables() != right.variables()
        {
            return None;
        }

        fn dense_coefficient_count<E: PositiveExponent>(
            polynomial: &MultivariatePolynomial<IntegerRing, E>,
            variable: usize,
        ) -> Option<usize> {
            if polynomial.is_zero()
                || polynomial.exponents_iter().any(|exponents| {
                    exponents
                        .iter()
                        .enumerate()
                        .any(|(index, exponent)| index != variable && !exponent.is_zero())
                })
            {
                return None;
            }

            let coefficient_count = polynomial.degree(variable).to_u32() as usize + 1;
            if coefficient_count > DENSE_UNIVARIATE_GCD_MAX_COEFFICIENTS
                || coefficient_count
                    > polynomial
                        .nterms()
                        .saturating_mul(DENSE_UNIVARIATE_GCD_MAX_SPARSITY_RATIO)
            {
                return None;
            }
            Some(coefficient_count)
        }

        dense_coefficient_count(divisor, variable)?;
        dense_coefficient_count(left, variable)?;
        dense_coefficient_count(right, variable)?;

        let degrees = divisor
            .exponents_iter()
            .map(|exponents| exponents[variable].to_u32() as usize)
            .collect::<Vec<_>>();
        let degree = *degrees.last()?;
        let coefficients = divisor
            .coefficients
            .iter()
            .cloned()
            .map(Integer::to_multi_prec)
            .collect();
        Some(Self {
            variable,
            coefficients,
            degrees,
            degree,
            division_remainder: MultiPrecisionInteger::default(),
        })
    }

    /// Divide one input and return its quotient only when every coefficient division is exact.
    fn try_div<E: PositiveExponent>(
        &mut self,
        dividend: &MultivariatePolynomial<IntegerRing, E>,
    ) -> Option<MultivariatePolynomial<IntegerRing, E>> {
        let dividend_degree = dividend.degree(self.variable).to_u32() as usize;
        let divisor_degree = self.degree;
        if dividend_degree < divisor_degree {
            return None;
        }

        let mut remainder = (0..=dividend_degree)
            .map(|_| MultiPrecisionInteger::default())
            .collect::<Vec<_>>();
        for (coefficient, exponents) in dividend.coefficients.iter().zip(dividend.exponents_iter())
        {
            remainder[exponents[self.variable].to_u32() as usize] =
                coefficient.clone().to_multi_prec();
        }

        let leading_divisor = self.coefficients.last().unwrap();
        let mut quotient = Vec::with_capacity(dividend_degree - divisor_degree + 1);
        for degree in (divisor_degree..=dividend_degree).rev() {
            let leading_remainder = mem::take(&mut remainder[degree]);
            if leading_remainder.is_zero() {
                continue;
            }

            let coefficient = leading_remainder
                .div_rem_owned_ref_assign(leading_divisor, &mut self.division_remainder);
            if !self.division_remainder.is_zero() {
                return None;
            }

            let shift = degree - divisor_degree;
            for (&divisor_degree, divisor_coefficient) in self.degrees[..self.degrees.len() - 1]
                .iter()
                .zip(&self.coefficients[..self.coefficients.len() - 1])
            {
                let target = shift + divisor_degree;
                debug_assert!(target < degree);
                remainder[target].sub_mul_assign(&coefficient, divisor_coefficient);
            }
            quotient.push((shift, coefficient));
        }
        if remainder[..divisor_degree]
            .iter()
            .any(|coefficient| !coefficient.is_zero())
        {
            return None;
        }

        let mut result = dividend.zero_with_capacity(quotient.len());
        let mut exponents = vec![E::zero(); dividend.nvars()];
        for (degree, coefficient) in quotient.into_iter().rev() {
            exponents[self.variable] = E::from_u32(degree as u32);
            result.append_monomial_back(Integer::from(coefficient), &exponents);
        }
        Some(result)
    }
}

/// Selects an integer coefficient that fixes the scale of every modular GCD image.
enum UnivariateGcdProjectiveNormalization {
    Leading(Integer),
    Constant { constant: Integer, leading: Integer },
}

impl UnivariateGcdProjectiveNormalization {
    /// Return the integer value assigned to the selected coefficient in every modular image.
    fn coefficient(&self) -> &Integer {
        match self {
            Self::Leading(coefficient) => coefficient,
            Self::Constant { constant, .. } => constant,
        }
    }

    /// Return the leading projective-coordinate value used by the fallback reconstruction.
    fn leading_coefficient(&self) -> &Integer {
        match self {
            Self::Leading(leading) | Self::Constant { leading, .. } => leading,
        }
    }

    /// Return whether modular images are normalized by their constant coefficient.
    fn uses_constant(&self) -> bool {
        matches!(self, Self::Constant { .. })
    }
}

/// Reconstructs a primitive univariate integer GCD from normalized 64-bit modular images.
struct UnivariateModularGcdContext<'a, E: PositiveExponent> {
    left: &'a MultivariatePolynomial<IntegerRing, E>,
    right: &'a MultivariatePolynomial<IntegerRing, E>,
    primitive_left: Cow<'a, MultivariatePolynomial<IntegerRing, E>>,
    primitive_right: Cow<'a, MultivariatePolynomial<IntegerRing, E>>,
    variable: usize,
    content_gcd: Integer,
    normalization: UnivariateGcdProjectiveNormalization,
    reconstruction_start_bits: u64,
}

impl<'a, E: PositiveExponent> UnivariateModularGcdContext<'a, E> {
    /// Removes the input contents and determines when coefficient reconstruction should start.
    fn new(
        left: &'a MultivariatePolynomial<IntegerRing, E>,
        right: &'a MultivariatePolynomial<IntegerRing, E>,
        variable: usize,
    ) -> Self {
        let left_content = left.content();
        let right_content = right.content();
        let content_gcd = Z.gcd(&left_content, &right_content);
        let primitive_left = if Z.is_one(&left_content) {
            Cow::Borrowed(left)
        } else {
            Cow::Owned(left.clone().div_coeff(&left_content))
        };
        let primitive_right = if Z.is_one(&right_content) {
            Cow::Borrowed(right)
        } else {
            Cow::Owned(right.clone().div_coeff(&right_content))
        };

        let leading_gcd = Z.gcd(&primitive_left.lcoeff(), &primitive_right.lcoeff());
        let probe_images = |coefficient: &Integer| {
            coefficient
                .significant_bits()
                .saturating_add(2)
                .saturating_add(u64::BITS as u64 - 1)
                / u64::BITS as u64
        };
        let leading_probe_images = probe_images(&leading_gcd);
        let constant_gcd = if leading_probe_images > 1 {
            Z.gcd(
                &primitive_left.get_constant(),
                &primitive_right.get_constant(),
            )
        } else {
            Integer::zero()
        };
        let normalization =
            if !constant_gcd.is_zero() && probe_images(&constant_gcd) < leading_probe_images {
                UnivariateGcdProjectiveNormalization::Constant {
                    constant: constant_gcd,
                    leading: leading_gcd,
                }
            } else {
                UnivariateGcdProjectiveNormalization::Leading(leading_gcd)
            };
        let reconstruction_start_bits = normalization
            .coefficient()
            .significant_bits()
            .saturating_add(2);

        Self {
            left,
            right,
            primitive_left,
            primitive_right,
            variable,
            content_gcd,
            normalization,
            reconstruction_start_bits,
        }
    }

    /// Removes the projective integer scale and restores the common input content.
    fn reconstructed_candidate(
        &self,
        reconstruction: &MultivariatePolynomial<IntegerRing, E>,
        degree: E,
    ) -> Option<MultivariatePolynomial<IntegerRing, E>> {
        if reconstruction.is_zero() || reconstruction.degree(self.variable) != degree {
            return None;
        }

        let content = reconstruction.content();
        if content.is_zero() {
            return None;
        }

        let mut candidate = reconstruction.clone().div_coeff(&content);
        if candidate.lcoeff().is_negative() {
            candidate = -candidate;
        }
        Some(candidate.mul_coeff(self.content_gcd.clone()))
    }

    /// Certify a reconstructed GCD by exact division into both original inputs.
    fn certified_reconstruction(
        &self,
        reconstruction: &MultivariatePolynomial<IntegerRing, E>,
        degree: E,
    ) -> Option<(
        MultivariatePolynomial<IntegerRing, E>,
        MultivariatePolynomial<IntegerRing, E>,
        MultivariatePolynomial<IntegerRing, E>,
    )> {
        let candidate = self.reconstructed_candidate(reconstruction, degree)?;
        let exact_cofactors = match DenseUnivariateIntegerDivisionContext::new(
            &candidate,
            self.left,
            self.right,
            self.variable,
        ) {
            Some(mut division) => match division.try_div(self.left) {
                Some(left_cofactor) => division
                    .try_div(self.right)
                    .map(|right_cofactor| (left_cofactor, right_cofactor)),
                None => None,
            },
            None => self.left.try_div(&candidate).and_then(|left_cofactor| {
                self.right
                    .try_div(&candidate)
                    .map(|right_cofactor| (left_cofactor, right_cofactor))
            }),
        }?;
        Some((candidate, exact_cofactors.0, exact_cofactors.1))
    }

    /// Rescale a constant-normalized CRT polynomial to the leading-coordinate representative.
    fn leading_reconstruction(
        &self,
        reconstruction: &MultivariatePolynomial<IntegerRing, E>,
        modulus: &Integer,
    ) -> MultivariatePolynomial<IntegerRing, E> {
        let leading = reconstruction.lcoeff();
        debug_assert!(Z.is_one(&Z.gcd(&leading, modulus)));
        let scale = (self
            .normalization
            .leading_coefficient()
            .clone()
            .symmetric_mod(modulus)
            * leading.mod_inverse(modulus))
        .symmetric_mod(modulus);
        reconstruction.map_coeff(
            |coefficient| (coefficient * &scale).symmetric_mod(modulus),
            Z,
        )
    }

    /// Merges modular GCD images until the reconstructed polynomial divides both inputs.
    fn run(
        self,
    ) -> Option<(
        MultivariatePolynomial<IntegerRing, E>,
        MultivariatePolynomial<IntegerRing, E>,
        MultivariatePolynomial<IntegerRing, E>,
    )> {
        let mut primes = univariate_modular_gcd_prime_iterator();
        let mut gcd_degree = None;
        let mut reconstruction = self.left.zero();
        let mut modulus = Integer::one();
        let mut next_reconstruction_bits = self.reconstruction_start_bits;
        let mut failed_probe_image_gap = 1u64;
        let leading_reconstruction_start_bits = self
            .normalization
            .leading_coefficient()
            .significant_bits()
            .saturating_add(2);
        let mut next_leading_reconstruction_bits = self
            .normalization
            .uses_constant()
            .then_some(leading_reconstruction_start_bits);
        let mut failed_leading_probe_image_gap = 1u64;

        loop {
            let prime = primes.next()?;
            let field = Zp64::new(prime);
            let normalization_image = self.normalization.coefficient().to_finite_field(&field);
            if field.is_zero(&normalization_image) {
                continue;
            }

            let left_leading_image = self.primitive_left.lcoeff().to_finite_field(&field);
            let right_leading_image = self.primitive_right.lcoeff().to_finite_field(&field);
            if field.is_zero(&left_leading_image) || field.is_zero(&right_leading_image) {
                continue;
            }

            let normalize_constant = self.normalization.uses_constant();
            let requested_leading = if normalize_constant {
                field.one()
            } else {
                normalization_image
            };
            let mut image = if let Some(dense_image) = DenseZp64UnivariateGcdImage::new(
                self.primitive_left.as_ref(),
                self.primitive_right.as_ref(),
                self.variable,
                &field,
            ) {
                dense_image.run(requested_leading)
            } else {
                let left_image = self.primitive_left.map_coeff(
                    |coefficient| coefficient.to_finite_field(&field),
                    field.clone(),
                );
                let right_image = self.primitive_right.map_coeff(
                    |coefficient| coefficient.to_finite_field(&field),
                    field.clone(),
                );
                debug_assert_eq!(
                    left_image.degree(self.variable),
                    self.primitive_left.degree(self.variable)
                );
                debug_assert_eq!(
                    right_image.degree(self.variable),
                    self.primitive_right.degree(self.variable)
                );
                left_image
                    .univariate_gcd(&right_image)
                    .mul_coeff(requested_leading)
            };
            if normalize_constant {
                let constant_image = image.get_constant();
                if field.is_zero(&constant_image) {
                    continue;
                }
                let constant_inverse =
                    DenseZp64UnivariateGcdImage::<E>::inverse_leading(&field, &constant_image);
                image = image.mul_coeff(field.mul(&normalization_image, &constant_inverse));
            }
            let image_degree = image.degree(self.variable);
            if image_degree.is_zero() {
                let candidate = self.left.constant(self.content_gcd.clone());
                let left_cofactor = self.left.clone().div_coeff(&self.content_gcd);
                let right_cofactor = self.right.clone().div_coeff(&self.content_gcd);
                return Some((candidate, left_cofactor, right_cofactor));
            }

            match gcd_degree {
                Some(degree) if image_degree > degree => continue,
                Some(degree) if image_degree == degree => {
                    IntegerPolynomialCrtContext::new(&modulus, &field)
                        .expect("univariate modular GCD prime repeated during CRT")
                        .merge_assign(&mut reconstruction, &image);
                    modulus *= prime;
                }
                _ => {
                    debug!(
                        "Starting univariate modular GCD reconstruction at degree {} modulo {}",
                        image_degree, prime
                    );
                    gcd_degree = Some(image_degree);
                    reconstruction =
                        image.map_coeff(|coefficient| field.to_symmetric_integer(coefficient), Z);
                    modulus = Integer::from(prime);
                    next_reconstruction_bits = self.reconstruction_start_bits;
                    failed_probe_image_gap = 1;
                    next_leading_reconstruction_bits = self
                        .normalization
                        .uses_constant()
                        .then_some(leading_reconstruction_start_bits);
                    failed_leading_probe_image_gap = 1;
                }
            }

            let modulus_bits = modulus.significant_bits();
            let selected_reconstruction_due = modulus_bits >= next_reconstruction_bits;
            let leading_reconstruction_due =
                next_leading_reconstruction_bits.is_some_and(|next_bits| modulus_bits >= next_bits);
            if !selected_reconstruction_due && !leading_reconstruction_due {
                continue;
            }

            let degree = gcd_degree.unwrap();
            let image_bits = Integer::from(prime).significant_bits();
            if selected_reconstruction_due {
                if let Some(result) = self.certified_reconstruction(&reconstruction, degree) {
                    // Nonzero input leading terms preserve the characteristic-zero GCD degree.
                    // Exact divisibility at that degree certifies the complete GCD.
                    return Some(result);
                }

                // The selected projective coordinate provides a cheap first probe point, not a
                // coefficient bound. Geometric backoff limits exact divisions of incomplete CRT
                // reconstructions.
                next_reconstruction_bits =
                    modulus_bits.saturating_add(image_bits.saturating_mul(failed_probe_image_gap));
                failed_probe_image_gap = failed_probe_image_gap.saturating_mul(2);
            }

            if leading_reconstruction_due {
                let leading_reconstruction = self.leading_reconstruction(&reconstruction, &modulus);
                if (!selected_reconstruction_due || leading_reconstruction != reconstruction)
                    && let Some(result) =
                        self.certified_reconstruction(&leading_reconstruction, degree)
                {
                    return Some(result);
                }

                next_leading_reconstruction_bits = Some(
                    modulus_bits
                        .saturating_add(image_bits.saturating_mul(failed_leading_probe_image_gap)),
                );
                failed_leading_probe_image_gap = failed_leading_probe_image_gap.saturating_mul(2);
            }
        }
    }
}

impl<E: PositiveExponent> MultivariatePolynomial<IntegerRing, E> {
    /// Reconstruct an integer polynomial from the symmetric digits of `value` in base `xi`.
    fn interpolate_univariate_integer(
        &self,
        mut value: Integer,
        variable: usize,
        xi: &Integer,
    ) -> Self {
        let xi_half = xi / &Integer::Single(2);
        let mut result = self.zero();
        let mut exponents = vec![E::zero(); self.nvars()];
        let mut exponent = 0u32;

        while !value.is_zero() {
            let (mut quotient, mut digit) = value.quot_rem(xi);
            if digit > xi_half {
                digit -= xi;
                quotient += 1i64;
            }

            if !digit.is_zero() {
                exponents[variable] = E::from_u32(exponent);
                result.append_monomial_back(digit, &exponents);
            }

            value = quotient;
            exponent += 1;
        }

        result
    }

    /// Run the integer heuristic directly when both inputs depend on one variable.
    fn heuristic_gcd_univariate(
        &self,
        b: &Self,
        variable: usize,
    ) -> Result<(Self, Self, Self), HeuristicGCDError> {
        let content_gcd = self.ring().gcd(&self.content(), &b.content());
        let mut a = Cow::Borrowed(self);
        let mut b = Cow::Borrowed(b);
        if !a.ring().is_one(&content_gcd) {
            a = Cow::Owned(a.into_owned().div_coeff(&content_gcd));
            b = Cow::Owned(b.into_owned().div_coeff(&content_gcd));
        }

        let max_a = a
            .coefficients
            .iter()
            .max_by(|left, right| left.abs_cmp(right))
            .unwrap_or(&Integer::Single(0));
        let max_b = b
            .coefficients
            .iter()
            .max_by(|left, right| left.abs_cmp(right))
            .unwrap_or(&Integer::Single(0));
        let minimum_maximum = if max_a.abs_cmp(max_b) == Ordering::Greater {
            max_b.abs()
        } else {
            max_a.abs()
        };
        let mut xi = &(&minimum_maximum * &Integer::Single(2)) + &Integer::Single(29);

        for retry in 0..6 {
            debug!("univariate round {}, xi={}", retry, xi);
            let evaluation_bits = |polynomial: &Self, maximum_coefficient: &Integer| {
                maximum_coefficient
                    .significant_bits()
                    .saturating_add(
                        xi.significant_bits()
                            .saturating_mul(polynomial.degree(variable).to_u32() as u64),
                    )
                    .saturating_add(ceil_log2_usize(polynomial.nterms()))
            };
            let estimated_bits = evaluation_bits(&a, max_a).max(evaluation_bits(&b, max_b));
            if estimated_bits > HEURISTIC_GCD_MAX_EVALUATED_COEFFICIENT_BITS {
                return Err(HeuristicGCDError::MaxSizeExceeded);
            }

            let evaluated_a = a.evaluate_univariate_horner(variable, &xi);
            let evaluated_b = b.evaluate_univariate_horner(variable, &xi);
            let evaluated_gcd = Z.gcd(&evaluated_a, &evaluated_b);

            let candidate = a.interpolate_univariate_integer(evaluated_gcd, variable, &xi);
            let candidate_content = candidate.content();
            let primitive_candidate = candidate.div_coeff(&candidate_content);
            if let Some(a_cofactor) = a.try_div(&primitive_candidate)
                && let Some(b_cofactor) = b.try_div(&primitive_candidate)
            {
                return Ok((
                    primitive_candidate.mul_coeff(content_gcd),
                    a_cofactor,
                    b_cofactor,
                ));
            }

            let evaluated_gcd = Z.gcd(&evaluated_a, &evaluated_b);
            let evaluated_a_cofactor = Z.exact_div_owned(evaluated_a, &evaluated_gcd);
            let a_cofactor = a.interpolate_univariate_integer(evaluated_a_cofactor, variable, &xi);
            if let Some(candidate) = a.try_div(&a_cofactor)
                && let Some(b_cofactor) = b.try_div(&candidate)
            {
                return Ok((candidate.mul_coeff(content_gcd), a_cofactor, b_cofactor));
            }

            let evaluated_b_cofactor = Z.exact_div_owned(evaluated_b, &evaluated_gcd);
            let b_cofactor = b.interpolate_univariate_integer(evaluated_b_cofactor, variable, &xi);
            if let Some(candidate) = b.try_div(&b_cofactor)
                && let Some(a_cofactor) = a.try_div(&candidate)
            {
                return Ok((candidate.mul_coeff(content_gcd), a_cofactor, b_cofactor));
            }

            xi = Z
                .quot_rem(&(&xi * &Integer::Single(73794)), &Integer::Single(27011))
                .0;
        }

        Err(HeuristicGCDError::BadReconstruction)
    }

    /// Perform a heuristic GCD algorithm.
    #[instrument(level = "debug", skip_all)]
    pub fn heuristic_gcd(&self, b: &Self) -> Result<(Self, Self, Self), HeuristicGCDError> {
        fn interpolate<E: PositiveExponent>(
            gamma: MultivariatePolynomial<IntegerRing, E>,
            var: usize,
            xi: &Integer,
        ) -> MultivariatePolynomial<IntegerRing, E> {
            let xi_half = xi / &Integer::Single(2);
            let mut coefficients = Vec::with_capacity(gamma.nterms());
            let mut exponents = Vec::with_capacity(gamma.exponents.len());

            // Each coefficient is an independent symmetric xi-adic integer. Decode it directly
            // instead of constructing and subtracting a polynomial for every radix digit.
            for term in &gamma {
                debug_assert!(term.exponents[var].is_zero());
                let mut coefficient = term.coefficient.clone();
                let mut exponent = 0u32;
                while !coefficient.is_zero() {
                    let (mut quotient, mut digit) = coefficient.quot_rem(xi);
                    if digit > xi_half {
                        digit -= xi;
                        quotient += 1i64;
                    }

                    if !digit.is_zero() {
                        coefficients.push(digit);
                        exponents.extend_from_slice(term.exponents);
                        let exponent_index = exponents.len() - gamma.nvars() + var;
                        exponents[exponent_index] = E::from_u32(exponent);
                    }

                    coefficient = quotient;
                    exponent += 1;
                }
            }

            if coefficients.is_empty() {
                return gamma.zero();
            }

            MultivariatePolynomial::from_coefficient_list(
                coefficients,
                exponents,
                gamma.variables().clone(),
                gamma.ring(),
            )
        }

        debug!("a={}; b={}", self, b);

        // do integer GCD
        let content_gcd = self.ring().gcd(&self.content(), &b.content());

        debug!("content={}", content_gcd);

        let mut a = Cow::Borrowed(self);
        let mut b = Cow::Borrowed(b);

        if !a.ring().is_one(&content_gcd) {
            a = Cow::Owned(a.into_owned().div_coeff(&content_gcd));
            b = Cow::Owned(b.into_owned().div_coeff(&content_gcd));
        }

        debug!("a_red={}; b_red={}", a, b);

        let estimated_bits = estimated_heuristic_gcd_evaluation_bits(&a, &b);
        if estimated_bits > HEURISTIC_GCD_MAX_EVALUATED_COEFFICIENT_BITS {
            debug!(
                "Estimated recursive heuristic evaluation coefficient is {} bits",
                estimated_bits
            );
            return Err(HeuristicGCDError::MaxSizeExceeded);
        }

        if let Some(var) =
            (0..a.nvars()).find(|x| a.degree(*x) > E::zero() && b.degree(*x) > E::zero())
        {
            let max_a = a
                .coefficients
                .iter()
                .max_by(|x1, x2| x1.abs_cmp(x2))
                .unwrap_or(&Integer::Single(0));

            let max_b = b
                .coefficients
                .iter()
                .max_by(|x1, x2| x1.abs_cmp(x2))
                .unwrap_or(&Integer::Single(0));

            let min = if max_a.abs_cmp(max_b) == Ordering::Greater {
                max_b.abs()
            } else {
                max_a.abs()
            };

            let mut xi = &(&min * &Integer::Single(2)) + &Integer::Single(29);

            for retry in 0..6 {
                debug!("round {}, xi={}", retry, xi);
                let evaluation_bits = |polynomial: &Self, max_coefficient: &Integer| {
                    max_coefficient
                        .significant_bits()
                        .saturating_add(
                            xi.significant_bits()
                                .saturating_mul(polynomial.degree(var).to_u32() as u64),
                        )
                        .saturating_add(ceil_log2_usize(polynomial.nterms()))
                };
                let estimated_bits = evaluation_bits(&a, max_a).max(evaluation_bits(&b, max_b));
                if estimated_bits > HEURISTIC_GCD_MAX_EVALUATED_COEFFICIENT_BITS {
                    debug!(
                        "Estimated heuristic evaluation coefficient is {} bits",
                        estimated_bits
                    );
                    return Err(HeuristicGCDError::MaxSizeExceeded);
                }

                let aa = a.replace(var, &xi);
                let bb = b.replace(var, &xi);

                let (gamma, co_fac_p, co_fac_q) = match aa.heuristic_gcd(&bb) {
                    Ok(x) => x,
                    Err(HeuristicGCDError::MaxSizeExceeded) => {
                        return Err(HeuristicGCDError::MaxSizeExceeded);
                    }
                    Err(HeuristicGCDError::BadReconstruction) => {
                        xi = Z
                            .quot_rem(&(&xi * &Integer::Single(73794)), &Integer::Single(27011))
                            .0;
                        continue;
                    }
                };

                debug!("gamma={}", gamma);

                let g = interpolate(gamma, var, &xi);
                let g_cont = g.content();

                let gc = g.div_coeff(&g_cont);

                if let Some(q) = a.try_div(&gc)
                    && let Some(q1) = b.try_div(&gc)
                {
                    debug!("match {} {}", q, q1);
                    return Ok((gc.mul_coeff(content_gcd), q, q1));
                }

                debug!("co_fac_p {}", co_fac_p);

                if !co_fac_p.is_zero() {
                    let a_co_fac = interpolate(co_fac_p, var, &xi);

                    if let Some(q) = a.try_div(&a_co_fac)
                        && let Some(q1) = b.try_div(&q)
                    {
                        return Ok((q.mul_coeff(content_gcd), a_co_fac, q1));
                    }
                }

                if !co_fac_q.is_zero() {
                    let b_co_fac = interpolate(co_fac_q, var, &xi);
                    debug!("cofac b {}", b_co_fac);

                    if let Some(q) = b.try_div(&b_co_fac)
                        && let Some(q1) = a.try_div(&q)
                    {
                        return Ok((q.mul_coeff(content_gcd), q1, b_co_fac));
                    }
                }

                xi = Z
                    .quot_rem(&(&xi * &Integer::Single(73794)), &Integer::Single(27011))
                    .0;
            }

            Err(HeuristicGCDError::BadReconstruction)
        } else {
            Ok((self.constant(content_gcd), a.into_owned(), b.into_owned()))
        }
    }

    /// Compute the gcd of multiple polynomials efficiently.
    /// `gcd(f0,f1,f2,...)=gcd(f0,f1+k2*f(2)+k3*f(3))`
    /// with high likelihood.
    pub fn gcd_multiple(
        mut f: Vec<MultivariatePolynomial<IntegerRing, E>>,
    ) -> MultivariatePolynomial<IntegerRing, E> {
        assert!(!f.is_empty());

        let mut prime_index = 1; // skip prime 2
        let mut loop_counter = 0;
        loop {
            if f.len() == 1 {
                return f.swap_remove(0);
            }

            if f.len() == 2 {
                return f[0].gcd(&f[1]);
            }

            // check if any entry is a number, as the gcd is then the gcd of the contents
            if let Some(n) = f.iter().find(|x| x.is_constant()) {
                let mut gcd = n.content();
                for x in f.iter() {
                    if x.ring().is_one(&gcd) {
                        break;
                    }

                    gcd = x.ring().gcd(&gcd, &x.content());
                }
                return n.constant(gcd);
            }

            f.sort_unstable_by(|a, b| b.nterms().cmp(&a.nterms())); // sort in decreasing order

            let a = f.pop().unwrap();

            // add all other polynomials
            let term_bound = f.iter().map(|x| x.nterms()).sum();
            let mut b = a.zero_with_capacity(term_bound);

            // prevent sampling f[i] and f[i+prime_len] with the same
            // prefactor every iteration
            let num_primes = if f.len().is_multiple_of(SMALL_PRIMES.len()) {
                SMALL_PRIMES.len() - 1
            } else {
                SMALL_PRIMES.len()
            };

            // try the 20 smallest chunks
            for p in f.iter().rev().take(20) {
                let k = Integer::Single(SMALL_PRIMES[prime_index % num_primes]);
                prime_index += 1;
                b = b + p.clone().mul_coeff(k);
            }

            let mut gcd = a.gcd(&b);
            if gcd.is_one() {
                return gcd;
            }

            // remove the content from the gcd before the division test as the odds
            // of an unlucky content are high
            let content = gcd.content();
            gcd = gcd.div_coeff(&content);
            let mut content_gcd = content;

            let old_length = f.len();

            f.retain(|x| {
                if x.try_div(&gcd).is_some() {
                    content_gcd = gcd.ring().gcd(&content_gcd, &x.content());
                    false
                } else {
                    true
                }
            });

            gcd = gcd.mul_coeff(content_gcd);

            if f.is_empty() {
                return gcd;
            }

            debug!(
                "Multiply GCD not found in one try, current estimate: {}",
                gcd
            );

            f.push(gcd);

            if f.len() == old_length + 1 && loop_counter > 5 {
                debug!("Multiple GCD failed");
                return MultivariatePolynomial::repeated_gcd(f);
            }

            loop_counter += 1;
        }
    }

    /// Lift projectively normalized CRT coefficients with a reconstructed common denominator.
    ///
    /// If the modular coefficients represent `g_i / g_pivot`, multiplying them by the common
    /// denominator makes every coefficient integral. A symmetric lift followed by removal of the
    /// integer content then recovers the primitive GCD candidate. The caller verifies the result
    /// by exact division because one reconstructed coefficient can expose only a proper divisor of
    /// the full common denominator.
    fn lift_projective_gcd(
        modular_gcd: &Self,
        modulus: &Integer,
        denominator: &Integer,
    ) -> Option<Self> {
        let mut candidate = modular_gcd.clone();
        for coefficient in &mut candidate.coefficients {
            let lifted = (coefficient.clone() * denominator).symmetric_mod(modulus);
            if lifted.is_zero() {
                return None;
            }
            *coefficient = lifted;
        }

        let content = candidate.content();
        if content.is_zero() {
            return None;
        }
        candidate = candidate.div_coeff(&content);
        if candidate.lcoeff().is_negative() {
            candidate = candidate.mul_coeff(Integer::from(-1));
        }
        Some(candidate)
    }

    /// Compute the gcd of two multivariate polynomials using Zippel's algorithm with a selected
    /// modular word size.
    fn gcd_zippel_auto(
        &self,
        b: &Self,
        vars: &[usize],
        bounds: &mut [E],
        tight_bounds: &mut [E],
    ) -> Self {
        let gamma = self
            .ring()
            .gcd(&self.lcoeff_varorder(vars), &b.lcoeff_varorder(vars));
        debug!(
            "gamma {} ({} significant bits)",
            gamma,
            gamma.significant_bits()
        );

        #[cfg(feature = "binary_size")]
        {
            Self::gcd_zippel::<u64>(self, b, vars, bounds, tight_bounds, &gamma)
        }

        #[cfg(not(feature = "binary_size"))]
        {
            if should_use_u64_zippel(self, b, &gamma) {
                debug!("Using 64-bit modular images for Zippel GCD");
                Self::gcd_zippel::<u64>(self, b, vars, bounds, tight_bounds, &gamma)
            } else {
                debug!("Using 32-bit modular images for Zippel GCD");
                Self::gcd_zippel::<u32>(self, b, vars, bounds, tight_bounds, &gamma)
            }
        }
    }

    /// Compute the gcd using Zippel's algorithm and the supplied modular workspace.
    /// TODO: provide a parallel implementation?
    #[instrument(level = "debug", skip_all)]
    fn gcd_zippel<UField: ModularGcdWorkspace>(
        &self,
        b: &Self,
        vars: &[usize], // variables
        bounds: &mut [E],
        tight_bounds: &mut [E],
        gamma: &Integer,
    ) -> Self
    where
        FiniteField<UField>: FiniteFieldCore<UField> + Set<Element = FiniteFieldElement<UField>>,
        <FiniteField<UField> as Set>::Element: Copy,
        Integer: ToFiniteField<UField> + FromFiniteField<UField>,
    {
        debug!("Zippel gcd of {} and {}", self, b);
        #[cfg(debug_assertions)]
        {
            self.check_consistency();
            b.check_consistency();
        }

        let mut primes = ModularGcdPrimeIterator::for_workspace::<UField>();

        'newfirstprime: loop {
            let Some(p) = primes.next() else {
                panic!("Ran out of primes for gcd reconstruction.\ngcd({self},{b})");
            };
            let Some(p) = UField::try_from_integer(p.into()) else {
                panic!("Ran out of primes for gcd reconstruction.\ngcd({self},{b})");
            };

            let mut finite_field = FiniteField::<UField>::new(p.clone());
            let mut gammap = gamma.to_finite_field(&finite_field);

            if finite_field.is_zero(&gammap) {
                continue 'newfirstprime;
            }

            let ap = self.map_coeff(|c| c.to_finite_field(&finite_field), finite_field.clone());
            let bp = b.map_coeff(|c| c.to_finite_field(&finite_field), finite_field.clone());

            debug!("New first image: gcd({},{}) mod {}", ap, bp, p);

            // calculate modular gcd image
            let mut gp = match MultivariatePolynomial::gcd_shape_modular(
                &ap,
                &bp,
                vars,
                bounds,
                tight_bounds,
            ) {
                Some(x) => x,
                None => {
                    debug!("Modular GCD failed: getting new prime");
                    continue 'newfirstprime;
                }
            };

            debug!("GCD suggestion: {}", gp);

            bounds[vars[0]] = gp.degree(vars[0]);

            // construct a new assumed form
            // we have to find the proper normalization
            let gfu = gp.to_univariate_polynomial_list(vars[0]);

            // find a coefficient of x1 in gf that is a monomial (single scaling)
            let mut single_scale = None;
            let mut nx = 0; // count the minimal number of samples needed
            for (i, (c, _e)) in gfu.iter().enumerate() {
                if c.nterms() > nx {
                    nx = c.nterms();
                }
                if c.nterms() == 1 {
                    single_scale = Some(i);
                }
            }

            // In the case of multiple scaling, each sample adds an
            // additional unknown, except for the first
            if single_scale.is_none() {
                let mut nx1 = (gp.nterms() - 1) / (gfu.len() - 1);
                if (gp.nterms() - 1) % (gfu.len() - 1) != 0 {
                    nx1 += 1;
                }
                if nx < nx1 {
                    nx = nx1;
                }
                debug!("Multiple scaling case: sample {} times", nx);
            }

            // Reconstruct the gcd projectively. Scaling every modular image by a coefficient of
            // the gcd itself avoids the potentially much larger `gamma / lc(g)` multiplier used
            // by classical integer CRT. Prefer the constant term: it is often small, and then
            // rational reconstruction needs roughly the height of the gcd instead of the height
            // of the input leading coefficients.
            let pivot_index = gp
                .exponents_iter()
                .position(|exponents| exponents.iter().all(|exponent| *exponent == E::zero()))
                .unwrap_or_else(|| {
                    gp.exponents_iter()
                        .enumerate()
                        .min_by_key(|(_, exponents)| {
                            exponents
                                .iter()
                                .map(|exponent| exponent.to_u32() as u64)
                                .sum::<u64>()
                        })
                        .map(|(index, _)| index)
                        .unwrap_or(0)
                });
            let pivot_exponents = gp.exponents(pivot_index).to_vec();
            let pivot_inverse = gp.ring().inv(&gp.coefficients[pivot_index]);
            gp = gp.mul_coeff(pivot_inverse);

            // construct the gcd suggestion in Z
            let mut gm = self.zero_with_capacity(gp.nterms());
            gm.exponents.clone_from(&gp.exponents);
            gm.coefficients = gp
                .coefficients
                .iter()
                .map(|coefficient| gp.ring().to_symmetric_integer(coefficient))
                .collect();

            let mut m = Integer::from_prime(&finite_field); // size of finite field

            debug!("Projective GCD suggestion: {} mod {} ", gm, p);
            let mut reconstruction_probe = (pivot_index != 0)
                .then_some(0)
                .unwrap_or_else(|| if gp.nterms() > 1 { 1 } else { 0 });
            let mut accepted_images = 1usize;
            let mut next_reconstruction_image = 1usize;
            let mut consecutive_probe_failures = 0usize;
            let mut failed_full_reconstruction_probe = None;

            // add new primes until we can reconstruct the full gcd
            'newprime: loop {
                // A prime dividing `gamma` was rejected above, so reduction preserves the true
                // GCD's leading multidegree. The lifted candidate retains every nonzero term of
                // the modular GCD. If it divides both integer inputs, its multidegree is therefore
                // both at least and at most that of the true GCD, which certifies the full GCD.
                // Probe one coefficient first so that insufficient moduli do not trigger a full
                // reconstruction attempt.
                'reconstruction_attempt: {
                    if accepted_images < next_reconstruction_image {
                        break 'reconstruction_attempt;
                    }

                    let reconstructed_probe = Rational::maximal_quotient_reconstruction(
                        &gm.coefficients[reconstruction_probe],
                        &m,
                        None,
                    );

                    let reconstructed_probe = match reconstructed_probe {
                        Ok(coefficient) if !Q.is_zero(&coefficient) => coefficient,
                        _ => {
                            consecutive_probe_failures += 1;
                            failed_full_reconstruction_probe = None;
                            next_reconstruction_image = accepted_images
                                + if consecutive_probe_failures >= 2 {
                                    2
                                } else {
                                    1
                                };
                            break 'reconstruction_attempt;
                        }
                    };

                    consecutive_probe_failures = 0;

                    // The reduced projective coefficient denominators all divide the primitive
                    // pivot coefficient. When the probe exposes that common denominator, it can
                    // lift the entire candidate without reconstructing every coefficient.
                    if let Some(gc) =
                        Self::lift_projective_gcd(&gm, &m, reconstructed_probe.denominator_ref())
                    {
                        debug!("Common-denominator GCD suggestion: {}", gc);
                        if gc.is_one() || (self.try_div(&gc).is_some() && b.try_div(&gc).is_some())
                        {
                            return gc;
                        }
                    }

                    let stable_after_failed_full_reconstruction = failed_full_reconstruction_probe
                        .as_ref()
                        .is_none_or(|previous| previous == &reconstructed_probe);
                    if !stable_after_failed_full_reconstruction {
                        failed_full_reconstruction_probe = Some(reconstructed_probe);
                        next_reconstruction_image = accepted_images + 1;
                    } else {
                        failed_full_reconstruction_probe = None;
                        let mut rational_coefficients = Vec::with_capacity(gm.nterms());
                        let mut failed_coefficient = None;

                        for (coefficient_index, coefficient) in gm.coefficients.iter().enumerate() {
                            let reconstructed = if coefficient_index == reconstruction_probe {
                                Ok(reconstructed_probe.clone())
                            } else {
                                Rational::maximal_quotient_reconstruction(coefficient, &m, None)
                            };
                            let Ok(coefficient) = reconstructed else {
                                failed_coefficient = Some(coefficient_index);
                                break;
                            };
                            if Q.is_zero(&coefficient) {
                                failed_coefficient = Some(coefficient_index);
                                break;
                            }
                            rational_coefficients.push(coefficient);
                        }

                        if let Some(coefficient_index) = failed_coefficient {
                            // Probe the first coefficient that blocked the full sweep after the
                            // next modular image has been accumulated.
                            reconstruction_probe = coefficient_index;
                            next_reconstruction_image = accepted_images + 1;
                        } else {
                            let hardest_coefficient = rational_coefficients
                                .iter()
                                .enumerate()
                                .max_by_key(|(_, coefficient)| {
                                    coefficient.numerator_ref().significant_bits()
                                        + coefficient.denominator_ref().significant_bits()
                                })
                                .map(|(index, _)| index)
                                .unwrap_or(reconstruction_probe);

                            let rational_gcd = MultivariatePolynomial::from_parts(
                                rational_coefficients,
                                gm.exponents.clone(),
                                Q,
                                gm.variables().clone(),
                            );
                            let content = rational_gcd.content();
                            let mut gc = rational_gcd.map_coeff(
                                |coefficient| Q.div(coefficient, &content).numerator(),
                                Z,
                            );
                            if gc.lcoeff().is_negative() {
                                gc = gc.mul_coeff(Integer::from(-1));
                            }

                            debug!("Final projective GCD suggestion: {}", gc);
                            if gc.is_one()
                                || (self.try_div(&gc).is_some() && b.try_div(&gc).is_some())
                            {
                                return gc;
                            }

                            reconstruction_probe = hardest_coefficient;
                            failed_full_reconstruction_probe =
                                Some(rational_gcd.coefficients[hardest_coefficient].clone());
                            next_reconstruction_image = accepted_images + 1;
                            debug!("Projective reconstruction does not divide: more primes needed");
                        }
                    }
                }

                loop {
                    let Some(p) = primes.next() else {
                        panic!(
                            "Ran out of primes for gcd images.\ngcd({self},{b})\nAttempt: {gm}\n vars: {vars:?}, bounds: {bounds:?}; {tight_bounds:?}"
                        );
                    };
                    let Some(p) = UField::try_from_integer(p.into()) else {
                        panic!(
                            "Ran out of primes for gcd images.\ngcd({self},{b})\nAttempt: {gm}\n vars: {vars:?}, bounds: {bounds:?}; {tight_bounds:?}"
                        );
                    };

                    finite_field = FiniteField::<UField>::new(p.clone());

                    gammap = gamma.to_finite_field(&finite_field);

                    if !finite_field.is_zero(&gammap) {
                        break;
                    }
                }

                let ap = self.map_coeff(|c| c.to_finite_field(&finite_field), finite_field.clone());
                let bp = b.map_coeff(|c| c.to_finite_field(&finite_field), finite_field.clone());
                debug!("New image: gcd({},{})", ap, bp);

                // for the univariate case, we don't need to construct an image
                if vars.len() == 1 {
                    gp = ap.univariate_gcd(&bp);
                    if gp.degree(vars[0]) < bounds[vars[0]] {
                        // original image and variable bound unlucky: restart
                        debug!("Unlucky original image: restart");
                        continue 'newfirstprime;
                    }

                    if gp.degree(vars[0]) > bounds[vars[0]] {
                        // prime is probably unlucky
                        debug!("Unlucky current image: try new one");
                        continue 'newprime;
                    }

                    for m in gp.into_iter() {
                        if gfu.iter().all(|(_, pow)| *pow != m.exponents[vars[0]]) {
                            debug!("Bad shape: terms missing");
                            continue 'newfirstprime;
                        }
                    }
                } else {
                    let rec = if let Some(single_scale) = single_scale {
                        MultivariatePolynomial::construct_new_image_single_scale(
                            &ap,
                            &bp,
                            ap.degree(vars[0]),
                            bp.degree(vars[0]),
                            bounds,
                            single_scale,
                            &vars[1..],
                            vars[0],
                            &gfu,
                        )
                    } else {
                        MultivariatePolynomial::construct_new_image_multiple_scales(
                            &ap,
                            &bp,
                            // NOTE: different from paper where they use a.degree(..)
                            // it could be that the degree in ap is lower than that of a
                            // which means the sampling will never terminate
                            ap.degree(vars[0]),
                            bp.degree(vars[0]),
                            bounds,
                            &vars[1..],
                            vars[0],
                            &gfu,
                        )
                    };

                    match rec {
                        Ok(r) => {
                            gp = r;
                        }
                        Err(GCDError::BadOriginalImage) => continue 'newfirstprime,
                        Err(GCDError::BadCurrentImage) => continue 'newprime,
                    }
                }

                // Use the same projective normalization as the first image. A missing pivot means
                // this prime is unlucky for the assumed support.
                let Some(pivot_coefficient) = gp
                    .into_iter()
                    .find(|term| term.exponents == pivot_exponents)
                    .map(|term| *term.coefficient)
                else {
                    continue 'newprime;
                };
                gp = gp.mul_coeff(ap.ring().inv(&pivot_coefficient));
                debug!("gp: {} mod {}", gp, gp.ring().get_prime());

                let crt = IntegerPolynomialCrtContext::new(&m, gp.ring())
                    .expect("modular GCD prime repeated during CRT reconstruction");
                crt.merge_assign(&mut gm, &gp);

                self.ring()
                    .mul_assign(&mut m, &Integer::from_prime(&gp.ring()));
                accepted_images += 1;

                debug!("gm: {} from ring {}", gm, m);
            }
        }
    }

    /// Prepare the coefficient-free ratios and coefficient-weighted starting values for geometric
    /// evaluation of `poly`.
    fn evaluate_terms<PE: PositiveExponent>(
        p: &Zp64,
        poly: &MultivariatePolynomial<Zp64, PE>,
        term_bases: &[FiniteFieldElement<u64>],
        shifted_bases: &[FiniteFieldElement<u64>],
        term_powers: &[Vec<FiniteFieldElement<u64>>],
        shifted_powers: &[Vec<FiniteFieldElement<u64>>],
    ) -> (
        Vec<(PE, usize, usize)>,
        Vec<FiniteFieldElement<u64>>,
        Vec<FiniteFieldElement<u64>>,
    ) {
        debug_assert_eq!(term_bases.len(), poly.nvars() - 1);
        debug_assert_eq!(shifted_bases.len(), poly.nvars() - 1);
        debug_assert_eq!(term_powers.len(), term_bases.len());
        debug_assert_eq!(shifted_powers.len(), shifted_bases.len());
        let rows = poly.univariate_row_ranges(0);
        let mut term_evals = Vec::with_capacity(poly.nterms());
        let mut current_evals = Vec::with_capacity(poly.nterms());
        for (coefficient, exponents) in poly
            .coefficients
            .iter()
            .zip(poly.exponents.chunks(poly.nvars()))
        {
            let mut term_eval = p.one();
            let mut current_eval = *coefficient;
            for (variable, exponent) in exponents.iter().skip(1).enumerate() {
                let exponent = exponent.to_u32() as usize;
                if exponent > 0 {
                    let term_power = term_powers[variable]
                        .get(exponent)
                        .copied()
                        .unwrap_or_else(|| p.pow(&term_bases[variable], exponent as u64));
                    let shifted_power = shifted_powers[variable]
                        .get(exponent)
                        .copied()
                        .unwrap_or_else(|| p.pow(&shifted_bases[variable], exponent as u64));
                    p.mul_assign(&mut term_eval, &term_power);
                    p.mul_assign(&mut current_eval, &shifted_power);
                }
            }
            term_evals.push(term_eval);
            current_evals.push(current_eval);
        }
        (rows, term_evals, current_evals)
    }

    /// Evaluate the geometric image of the polynomial at the given beta points and return the result as a polynomial.
    fn eval_geometric_image<PE: PositiveExponent>(
        poly: &MultivariatePolynomial<Zp64, PE>,
        rows: &[(PE, usize, usize)],
        term_evals: &[FiniteFieldElement<u64>],
        current_evals: &mut [FiniteFieldElement<u64>],
    ) -> MultivariatePolynomial<Zp64, PE> {
        let mut image = poly.zero_with_capacity(rows.len());
        poly.evaluate_and_advance_weighted_terms(current_evals, term_evals, 0, rows, &mut image);
        image
    }

    fn evaluate_terms_bivariate<PE: PositiveExponent>(
        p: &Zp64,
        poly: &MultivariatePolynomial<Zp64, PE>,
        betas: &[FiniteFieldElement<u64>],
    ) -> (Vec<(PE, PE)>, Vec<(usize, FiniteFieldElement<u64>)>) {
        let mut unique_indices = vec![];
        let mut index_map = HashMap::default();
        for p in poly.exponents_iter() {
            let row = (p[0], p[1]);
            index_map.entry(row).or_insert_with(|| {
                unique_indices.push(row);
                0
            });
        }
        unique_indices.sort();
        for (index, e) in unique_indices.iter().enumerate() {
            index_map.insert(*e, index);
        }

        (
            unique_indices,
            poly.exponents
                .chunks(poly.nvars())
                .map(|ee| {
                    let mut eval = p.one();
                    for (e, beta) in ee.iter().skip(2).zip(betas) {
                        if *e > PE::zero() {
                            p.mul_assign(&mut eval, &p.pow(beta, e.to_u32() as u64));
                        }
                    }
                    (index_map[&(ee[0], ee[1])], eval)
                })
                .collect(),
        )
    }

    fn evaluate_geometric_image_bivariate<PE: PositiveExponent>(
        p: &Zp64,
        poly: &MultivariatePolynomial<Zp64, PE>,
        row_exponents: &[(PE, PE)],
        term_evals: &[(usize, FiniteFieldElement<u64>)],
        current_evals: &mut [FiniteFieldElement<u64>],
    ) -> MultivariatePolynomial<Zp64, u32> {
        let mut coefficients = vec![p.zero(); row_exponents.len()];
        let mut exp = vec![0; poly.nvars() * row_exponents.len()];
        for (index, (exponent0, exponent1)) in row_exponents.iter().enumerate() {
            let exp_offset = index * poly.nvars();
            exp[exp_offset] = exponent0.to_u32();
            exp[exp_offset + 1] = exponent1.to_u32();
        }

        for ((index, term_eval), current_eval) in term_evals.iter().zip(current_evals) {
            p.add_assign(&mut coefficients[*index], &*current_eval);
            p.mul_assign(current_eval, term_eval);
        }

        // remove zeros
        let mut current_index = 0;
        for i in 0..coefficients.len() {
            if current_index != i {
                coefficients[current_index] = coefficients[i];
                exp.copy_within(
                    (i * poly.nvars())..(i + 1) * poly.nvars(),
                    current_index * poly.nvars(),
                );
            }

            if !p.is_zero(&coefficients[i]) {
                current_index += 1;
            }
        }
        coefficients.truncate(current_index);
        exp.truncate(current_index * poly.nvars());

        MultivariatePolynomial::from_parts(coefficients, exp, p.clone(), poly.variables().clone())
    }

    /// Recover recurrence roots with 32-bit arithmetic whenever the Hu interpolation prime fits
    /// in a `u32`, then convert the roots back to the surrounding 64-bit field representation.
    fn hu_monagan_recurrence_roots(
        p: &Zp64,
        coefficients: &[FiniteFieldElement<u64>],
    ) -> Option<Vec<FiniteFieldElement<u64>>> {
        if let Ok(prime) = u32::try_from(p.get_prime()) {
            let small_field = Zp::new(prime);
            let small_coefficients = coefficients
                .iter()
                .map(|coefficient| small_field.to_element(p.from_element(coefficient) as u32))
                .collect::<Vec<_>>();
            let mut root_context = DenseFiniteFieldRootContext::new(&small_field);
            return root_context
                .find_distinct_nonzero_roots(&small_coefficients)
                .map(|roots| {
                    roots
                        .iter()
                        .map(|root| p.to_element(small_field.from_element(root) as u64))
                        .collect()
                });
        }

        DenseFiniteFieldRootContext::new(p).find_distinct_nonzero_roots(coefficients)
    }

    fn hu_monagan_sparse_interpolate<PE: PositiveExponent>(
        p: &Zp64,
        images: &[MultivariatePolynomial<Zp64, PE>],
        sample_points: &[FiniteFieldElement<u64>],
        alpha: &FiniteFieldElement<u64>,
        discrete_log_context: &Zp64DiscreteLogContext<'_>,
        kronecker: &HuMonaganKroneckerMap,
        d_0: PE,
    ) -> Option<MultivariatePolynomial<Zp64, PE>> {
        if images.len() < 4 || !images.len().is_multiple_of(2) {
            return None;
        }

        let l = images.len() / 2;
        let mut res = images[0].zero();
        let mut image_exp = vec![PE::zero(); images[0].nvars()];
        let mut result_exp = vec![PE::zero(); images[0].nvars()];
        for i in 0..=d_0.to_u32() {
            image_exp[0] = PE::from_u32(i);
            let row = images
                .iter()
                .map(|x| x.coefficient(&image_exp).unwrap_or(p.zero()))
                .collect::<Vec<_>>();

            if row.iter().all(|x| p.is_zero(x)) {
                continue;
            }

            let (recurrence, stable_count) = p.find_linear_recurrence_relation(&row);
            let t = recurrence.len();
            if t == 0 || t >= l || stable_count < 2 || p.is_zero(&recurrence[0]) {
                debug!(
                    "Failed to find recurrence relation for row at x^{}: stable={}",
                    i, stable_count
                );
                return None;
            }

            // The recurrence roots are the distinct monomial evaluations alpha^e. The prime
            // and Kronecker range checks above guarantee that these are distinct, so the
            // polynomial is already square-free and completely split into linear factors.
            let mut bma_coefficients = recurrence
                .iter()
                .rev()
                .map(|coefficient| p.neg(coefficient))
                .collect::<Vec<_>>();
            bma_coefficients.push(p.one());
            let Some(roots) = Self::hu_monagan_recurrence_roots(p, &bma_coefficients) else {
                debug!("Failed to recover BMA roots at x^{}", i);
                return None;
            };

            let mut monomials = Vec::with_capacity(t);
            for m in roots {
                let ee = discrete_log_context.discrete_log(&m);
                if ee >= kronecker.range() {
                    debug!("Factor too large: {}", ee);
                    return None;
                }

                monomials.push(ee);
            }
            monomials.sort_unstable();

            let sample_generators = monomials
                .iter()
                .map(|e| p.pow(alpha, *e))
                .collect::<Vec<_>>();
            let mut sol =
                images[0].solve_shifted_transposed_vandermonde(&sample_generators, &row[..t]);
            for ((coeff, sample_generator), e) in
                sol.iter_mut().zip(&sample_generators).zip(&monomials)
            {
                p.mul_assign(coeff, sample_generator);
                let initial_power = p.pow(&sample_points[0], *e);
                p.div_assign(coeff, &initial_power);
            }

            for (sample_point, expected) in sample_points.iter().zip(&row).skip(t) {
                // Evaluate the encoded exponent directly so values above the signed exponent
                // range are not truncated by a polynomial substitution.
                let mut evaluated = p.zero();
                for (coefficient, exponent) in sol.iter().zip(&monomials) {
                    p.add_assign(
                        &mut evaluated,
                        &p.mul(coefficient, &p.pow(sample_point, *exponent)),
                    );
                }
                if evaluated != *expected {
                    debug!("Sparse interpolation row at x^{} failed sample check", i);
                    return None;
                }
            }

            let mut row_poly = res.zero();
            for (coeff, e) in sol.into_iter().zip(&monomials) {
                result_exp[0] = PE::from_u32(i);
                kronecker.decode(*e, &mut result_exp)?;
                row_poly.append_monomial(coeff, &result_exp);
            }

            res = res + row_poly;
        }

        Some(res)
    }

    fn hu_monagan_sparse_interpolate_bivariate<PE: PositiveExponent>(
        p: &Zp64,
        images: &[MultivariatePolynomial<Zp64, u32>],
        sample_points: &[FiniteFieldElement<u64>],
        alpha: &FiniteFieldElement<u64>,
        discrete_log_context: &Zp64DiscreteLogContext<'_>,
        kronecker: &HuMonaganKroneckerMap,
        d_0_1: (u32, u32),
    ) -> Option<MultivariatePolynomial<Zp64, PE>> {
        if images.len() < 4 || !images.len().is_multiple_of(2) {
            return None;
        }

        let l = images.len() / 2;
        let mut rows = HashSet::default();
        for image in images {
            for m in image {
                if m.exponents[0] > d_0_1.0 || m.exponents[1] > d_0_1.1 {
                    return None;
                }

                rows.insert((m.exponents[0], m.exponents[1]));
            }
        }

        let mut rows = rows.into_iter().collect::<Vec<_>>();
        rows.sort_unstable();

        let mut res =
            MultivariatePolynomial::<Zp64, PE>::new(p, None, images[0].variables().clone());
        let mut image_exp = vec![0; images[0].nvars()];
        let mut result_exp = vec![PE::zero(); images[0].nvars()];
        for (e0, e1) in rows {
            image_exp[0] = e0;
            image_exp[1] = e1;
            let row = images
                .iter()
                .map(|x| x.coefficient(&image_exp).unwrap_or(p.zero()))
                .collect::<Vec<_>>();

            let (recurrence, stable_count) = p.find_linear_recurrence_relation(&row);
            let t = recurrence.len();
            if t == 0 || t >= l || stable_count < 2 || p.is_zero(&recurrence[0]) {
                debug!(
                    "Failed to find recurrence relation for bivariate row x^{} y^{}: stable={}",
                    e0, e1, stable_count
                );
                return None;
            }

            let mut bma_coefficients = recurrence
                .iter()
                .rev()
                .map(|coefficient| p.neg(coefficient))
                .collect::<Vec<_>>();
            bma_coefficients.push(p.one());
            let Some(roots) = Self::hu_monagan_recurrence_roots(p, &bma_coefficients) else {
                debug!(
                    "Failed to recover BMA roots at bivariate row x^{} y^{}",
                    e0, e1
                );
                return None;
            };

            let mut monomials = Vec::with_capacity(t);
            for m in roots {
                let ee = discrete_log_context.discrete_log(&m);
                if ee >= kronecker.range() {
                    debug!("Factor too large: {}", ee);
                    return None;
                }

                monomials.push(ee);
            }
            monomials.sort_unstable();

            let sample_generators = monomials
                .iter()
                .map(|e| p.pow(alpha, *e))
                .collect::<Vec<_>>();
            let mut sol =
                images[0].solve_shifted_transposed_vandermonde(&sample_generators, &row[..t]);
            for ((coeff, sample_generator), e) in
                sol.iter_mut().zip(&sample_generators).zip(&monomials)
            {
                p.mul_assign(coeff, sample_generator);
                let initial_power = p.pow(&sample_points[0], *e);
                p.div_assign(coeff, &initial_power);
            }

            for (sample_point, expected) in sample_points.iter().zip(&row).skip(t) {
                // Evaluate the encoded exponent directly so values above the signed exponent
                // range are not truncated by a polynomial substitution.
                let mut evaluated = p.zero();
                for (coefficient, exponent) in sol.iter().zip(&monomials) {
                    p.add_assign(
                        &mut evaluated,
                        &p.mul(coefficient, &p.pow(sample_point, *exponent)),
                    );
                }
                if evaluated != *expected {
                    debug!(
                        "Sparse interpolation bivariate row x^{} y^{} failed sample check",
                        e0, e1
                    );
                    return None;
                }
            }

            let mut row_poly = res.zero();
            for (coeff, e) in sol.into_iter().zip(&monomials) {
                result_exp[0] = PE::from_u32(e0);
                result_exp[1] = PE::from_u32(e1);
                kronecker.decode(*e, &mut result_exp)?;
                row_poly.append_monomial(coeff, &result_exp);
            }

            res = res + row_poly;
        }

        Some(res)
    }

    /// Compute the gcd using the Hu-Monagan algorithm that interpolates the gcd and a cofactor at
    /// the same time. Dense supports are handed to the known-shape Zippel kernel, since recovering
    /// thousands of Hu recurrence roots is substantially more expensive than reusing their shape.
    ///
    /// The polynomials must be primitive in the main variable.
    ///
    /// References:
    /// - "Speeding up polynomial GCD, a crucial operation in Maple" by Michael Monagan
    /// - "A fast parallel sparse polynomial GCD algorithm" by Jiaxiong Hu and Michael Monagan
    #[instrument(level = "debug", skip_all)]
    pub fn gcd_hu_monagan(&self, b: &Self, bounds: &[E]) -> Option<Self> {
        self.gcd_hu_monagan_with_anchor(b, bounds, HuMonaganAnchor::from_inputs(self, b))
    }

    /// Runs Hu-Monagan while interpolating the cofactor of the selected input.
    fn gcd_hu_monagan_with_anchor(
        &self,
        b: &Self,
        bounds: &[E],
        anchor: HuMonaganAnchor,
    ) -> Option<Self> {
        self.gcd_hu_monagan_with_plan(b, bounds, anchor, false)
    }

    /// Runs a main-variable plan whose coefficient-row reduction was checked after content removal.
    fn gcd_hu_monagan_with_preapproved_plan(
        &self,
        b: &Self,
        bounds: &[E],
        anchor: HuMonaganAnchor,
    ) -> Option<Self> {
        self.gcd_hu_monagan_with_plan(b, bounds, anchor, true)
    }

    /// Executes Hu-Monagan with an explicit anchor and an optional preapproved sparse-row plan.
    fn gcd_hu_monagan_with_plan(
        &self,
        b: &Self,
        bounds: &[E],
        anchor: HuMonaganAnchor,
        plan_is_preapproved: bool,
    ) -> Option<Self> {
        debug!(
            "Hu-Monagan gcd of {} and {} with bounds {:?}",
            self, b, bounds
        );
        assert!(bounds[0] > E::zero());
        assert!(self.nvars() > 1);

        let vars = (0..self.nvars())
            .filter(|&variable| bounds[variable] > E::zero())
            .collect::<SmallVec<[_; INLINED_EXPONENTS]>>();
        if !plan_is_preapproved
            && !should_use_hu_monagan_with_anchor(self, b, &vars, bounds, anchor)
        {
            let mut bounds = bounds.to_vec();
            let mut tight_bounds: SmallVec<[E; INLINED_EXPONENTS]> =
                bounds.iter().copied().collect();

            return Some(MultivariatePolynomial::gcd_zippel_auto(
                self,
                b,
                &vars,
                &mut bounds,
                &mut tight_bounds,
            ));
        }

        #[derive(Debug, PartialEq, Eq, Copy, Clone)]
        enum ImageKind {
            GcdMultiple,
            CofactorMultiple,
        }

        let (a, b) = anchor.order_inputs(self, b);
        let h_zero = MultivariatePolynomial::<_, E>::new(&IntegerRing, None, a.variables().clone());

        let largest_coeff = a
            .coefficients
            .iter()
            .chain(&b.coefficients)
            .max_by(|a, b| a.abs_cmp(b))
            .unwrap()
            .abs()
            * 2i64;

        let mut r: Vec<_> = (0..a.nvars())
            .map(|i| a.degree(i).max(b.degree(i)).max(bounds[i]).to_u32())
            .collect();

        let delta = 1u32;
        let mut d_0 = bounds[0];
        let mut smooth_prime_index = 0;
        let mut rng = rand::rng();

        'kronecker_prime: loop {
            for rr in &mut r {
                *rr += 1;
            }

            let Some(kronecker) = HuMonaganKroneckerMap::new(&r, 1) else {
                debug!("Hu-Monagan Kronecker range does not fit in u64; using Zippel");
                return None;
            };

            let mut h = h_zero.clone();
            let mut m = Integer::one();
            let mut image_kind = None;

            'new_image: loop {
                let prime_bound =
                    hu_monagan_prime_lower_bound(kronecker.range(), delta, &largest_coeff);

                let (p, totient_primes, alpha, a_p, b_p) = 'new_prime: loop {
                    let Some((p, alpha, fs)) = SMOOTH_PRIMES.get(smooth_prime_index) else {
                        warn!(
                            "Ran out of smooth primes for Hu-Monagan2 GCD.\ngcd({},{})",
                            self, b
                        );
                        return None;
                    };

                    smooth_prime_index += 1;

                    if *p < prime_bound {
                        continue;
                    }

                    let field = Zp64::new(*p);
                    let a_p = a.map_coeff(|c| c.to_finite_field(&field), field.clone());
                    let b_p = b.map_coeff(|c| c.to_finite_field(&field), field.clone());

                    if a_p.degree(0) < a.degree(0) || b_p.degree(0) < b.degree(0) {
                        debug!("Bad prime {}", p);
                        continue 'new_prime;
                    }

                    let mut totient_primes = vec![];
                    for (f, prime) in fs.iter().zip(&SMOOTH_PRIME_BASE) {
                        if *f > 0 {
                            totient_primes.push((*prime, *f as u32));
                        }
                    }

                    let alpha = field.to_element(*alpha as u64);
                    break (field, totient_primes, alpha, a_p, b_p);
                };
                let discrete_log_context =
                    Zp64DiscreteLogContext::new(&p, &alpha, p.get_prime() - 1, &totient_primes);

                let mut betas = Vec::with_capacity(a.nvars() - 1);
                betas.push(alpha);
                for power in kronecker.powers().iter().take(a.nvars().saturating_sub(2)) {
                    betas.push(p.pow(&alpha, *power));
                }

                let shift = p.from_element(&p.sample_small_integer(&mut rng, 0..=i64::MAX - 1));
                let shifted_betas = betas
                    .iter()
                    .map(|beta| p.pow(beta, shift))
                    .collect::<Vec<_>>();
                let mut term_powers = Vec::with_capacity(betas.len());
                let mut shifted_powers = Vec::with_capacity(betas.len());
                for ((beta, shifted_beta), radix) in betas.iter().zip(&shifted_betas).zip(&r[1..]) {
                    let mut term_power = p.one();
                    let mut shifted_power = p.one();
                    let cache_len = (*radix as usize).min(POW_CACHE_SIZE);
                    let mut variable_term_powers = Vec::with_capacity(cache_len);
                    let mut variable_shifted_powers = Vec::with_capacity(cache_len);
                    for _ in 0..cache_len {
                        variable_term_powers.push(term_power);
                        variable_shifted_powers.push(shifted_power);
                        p.mul_assign(&mut term_power, beta);
                        p.mul_assign(&mut shifted_power, shifted_beta);
                    }
                    term_powers.push(variable_term_powers);
                    shifted_powers.push(variable_shifted_powers);
                }

                let (a_rows, a_term_evals, mut a_current_evals) = Self::evaluate_terms(
                    &p,
                    &a_p,
                    &betas,
                    &shifted_betas,
                    &term_powers,
                    &shifted_powers,
                );
                let (b_rows, b_term_evals, mut b_current_evals) = Self::evaluate_terms(
                    &p,
                    &b_p,
                    &betas,
                    &shifted_betas,
                    &term_powers,
                    &shifted_powers,
                );

                let mut gcd_images = Vec::new();
                let mut cofactor_images = Vec::new();
                let mut sample_points = Vec::new();
                let mut next_num_samples = 4usize;
                let mut sample_point = p.pow(&alpha, shift);

                let selected_image = 'new_sample: loop {
                    for _ in 0..2 {
                        let current_sample_point = sample_point;
                        p.mul_assign(&mut sample_point, &alpha);

                        let a_j = Self::eval_geometric_image(
                            &a_p,
                            &a_rows,
                            &a_term_evals,
                            &mut a_current_evals,
                        );
                        let b_j = Self::eval_geometric_image(
                            &b_p,
                            &b_rows,
                            &b_term_evals,
                            &mut b_current_evals,
                        );

                        if a_j.degree(0) < a_p.degree(0) || b_j.degree(0) < b_p.degree(0) {
                            debug!("Bad Kronecker image, trying new prime");
                            continue 'new_image;
                        }

                        let g_j = a_j.univariate_gcd(&b_j);
                        let g_degree = g_j.degree(0);
                        if g_degree < d_0 {
                            debug!("Unlucky degree bound: {} vs {}", g_degree, d_0);
                            d_0 = g_degree;
                            continue 'kronecker_prime;
                        }
                        if g_degree > d_0 {
                            debug!("Unlucky evaluation point, trying new prime");
                            continue 'new_image;
                        }

                        let lc_a_j = a_j.univariate_lcoeff(0);
                        let Some(a_cofactor_j) = a_j.try_div(&g_j) else {
                            debug!("Univariate image division failed for a, trying new prime");
                            continue 'new_image;
                        };

                        gcd_images.push(g_j * &lc_a_j);
                        cofactor_images.push(a_cofactor_j);

                        sample_points.push(current_sample_point);
                    }

                    if gcd_images.len() < next_num_samples {
                        continue 'new_sample;
                    }

                    // Doubling is useful while the recurrence is small, but it can oversample
                    // substantially after a failed 64-point reconstruction. Continue in
                    // quarter-size increments so moderately sparse images do not pay for the
                    // next full power of two.
                    next_num_samples += if next_num_samples < 64 {
                        next_num_samples
                    } else {
                        next_num_samples / 4
                    };

                    if image_kind.is_none() || image_kind == Some(ImageKind::GcdMultiple) {
                        let gcd_image = Self::hu_monagan_sparse_interpolate(
                            &p,
                            &gcd_images,
                            &sample_points,
                            &alpha,
                            &discrete_log_context,
                            &kronecker,
                            d_0,
                        );

                        if let Some(gcd_image) = gcd_image {
                            image_kind = Some(ImageKind::GcdMultiple);
                            break 'new_sample gcd_image;
                        }
                    }

                    if image_kind.is_none() || image_kind == Some(ImageKind::CofactorMultiple) {
                        let cofactor_image = Self::hu_monagan_sparse_interpolate(
                            &p,
                            &cofactor_images,
                            &sample_points,
                            &alpha,
                            &discrete_log_context,
                            &kronecker,
                            a_p.degree(0) - d_0,
                        );

                        if let Some(cofactor_image) = cofactor_image {
                            image_kind = Some(ImageKind::CofactorMultiple);
                            break 'new_sample cofactor_image;
                        }
                    }
                };

                let old_h = h.clone();

                if m == 1 {
                    h = selected_image.map_coeff(|c| p.to_symmetric_integer(c), Z);
                    m = p.get_prime().into();
                } else {
                    let crt = IntegerPolynomialCrtContext::new(&m, &p)
                        .expect("Hu-Monagan prime repeated during CRT reconstruction");
                    crt.merge_assign(&mut h, &selected_image);
                    m *= p.get_prime();
                }

                if h != old_h && !old_h.is_zero() {
                    continue 'new_image;
                }

                let hm = h.clone();
                let content = hm.univariate_content(0);
                let primitive = hm / &content;

                let gcd_candidate = match image_kind.unwrap() {
                    ImageKind::GcdMultiple => PolynomialGCD::normalize(primitive),
                    ImageKind::CofactorMultiple => {
                        if let Some(q) = a.try_div(&primitive) {
                            PolynomialGCD::normalize(q)
                        } else {
                            debug!("Cofactor image does not divide");

                            if old_h.is_zero() {
                                continue 'new_image;
                            } else {
                                continue 'kronecker_prime;
                            }
                        }
                    }
                };

                if a.try_div(&gcd_candidate).is_some() && b.try_div(&gcd_candidate).is_some() {
                    debug!("Found GCD: {}", gcd_candidate);
                    return Some(PolynomialGCD::normalize(gcd_candidate));
                }

                debug!("Non-division of {}, trying new image", gcd_candidate);

                if !old_h.is_zero() {
                    continue 'kronecker_prime;
                }
            }
        }
    }

    /// Compute the gcd using the Hu-Monagan algorithm that
    /// uses a bivariate image and interpolates the gcd and a cofactor
    /// at the same time.
    #[instrument(level = "debug", skip_all)]
    pub fn gcd_hu_monagan_bivariate(&self, b: &Self, bounds: &[E]) -> Option<Self> {
        debug!(
            "Bivariate Hu-Monagan2 gcd of {} and {} with bounds {:?}",
            self, b, bounds
        );
        assert!(bounds[0] > E::zero());
        assert!(bounds[1] > E::zero());
        assert!(self.nvars() > 2);

        let a_content = self.bivariate_content(0, 1);
        let b_content = b.bivariate_content(0, 1);
        if !a_content.is_one() || !b_content.is_one() {
            if let Some(g) = (self / &a_content).gcd_hu_monagan_bivariate(&(b / &b_content), bounds)
            {
                let content = a_content.gcd(&b_content);
                return Some(content * &g);
            } else {
                return None;
            }
        }

        #[derive(Debug, PartialEq, Eq, Copy, Clone)]
        enum ImageKind {
            GcdMultiple,
            CofactorMultiple,
        }

        let (a, b) = if self.nterms() <= b.nterms() {
            (self, b)
        } else {
            (b, self)
        };
        let h_zero = MultivariatePolynomial::<_, E>::new(&IntegerRing, None, a.variables().clone());

        let largest_coeff = a
            .coefficients
            .iter()
            .chain(&b.coefficients)
            .max_by(|a, b| a.abs_cmp(b))
            .unwrap()
            .abs()
            * 2i64;

        let start_exp = 2;
        let mut r: Vec<_> = (0..a.nvars())
            .map(|i| a.degree(i).max(b.degree(i)).max(bounds[i]).to_u32())
            .collect();

        let delta = 1u32;
        let mut d_0_1 = (bounds[0].to_u32(), bounds[1].to_u32());
        let mut smooth_prime_index = 0;
        let mut rng = rand::rng();

        'kronecker_prime: loop {
            for rr in &mut r {
                *rr += 1;
            }

            let Some(kronecker) = HuMonaganKroneckerMap::new(&r, start_exp) else {
                debug!("Bivariate Hu-Monagan Kronecker range does not fit in u64; using Zippel");
                return None;
            };

            let mut h = h_zero.clone();
            let mut m = Integer::one();
            let mut image_kind = None;

            'new_image: loop {
                let prime_bound =
                    hu_monagan_prime_lower_bound(kronecker.range(), delta, &largest_coeff);

                let (p, totient_primes, alpha, a_p, b_p) = 'new_prime: loop {
                    let Some((p, alpha, fs)) = SMOOTH_PRIMES.get(smooth_prime_index) else {
                        warn!(
                            "Ran out of smooth primes for bivariate Hu-Monagan2 GCD.\ngcd({},{})",
                            self, b
                        );
                        return None;
                    };

                    smooth_prime_index += 1;

                    if *p < prime_bound {
                        continue;
                    }

                    let field = Zp64::new(*p);
                    let a_p = a.map_coeff(|c| c.to_finite_field(&field), field.clone());
                    let b_p = b.map_coeff(|c| c.to_finite_field(&field), field.clone());

                    let a_deg = a.bivariate_deg();
                    let b_deg = b.bivariate_deg();
                    if a_p.bivariate_deg() < a_deg || b_p.bivariate_deg() < b_deg {
                        debug!("Bad prime {}", p);
                        continue 'new_prime;
                    }

                    let mut totient_primes = vec![];
                    for (f, prime) in fs.iter().zip(&SMOOTH_PRIME_BASE) {
                        if *f > 0 {
                            totient_primes.push((*prime, *f as u32));
                        }
                    }

                    let alpha = field.to_element(*alpha as u64);
                    break (field, totient_primes, alpha, a_p, b_p);
                };
                let discrete_log_context =
                    Zp64DiscreteLogContext::new(&p, &alpha, p.get_prime() - 1, &totient_primes);

                let mut betas = Vec::with_capacity(a.nvars() - start_exp);
                betas.push(alpha);
                for power in kronecker
                    .powers()
                    .iter()
                    .take(a.nvars().saturating_sub(start_exp + 1))
                {
                    betas.push(p.pow(&alpha, *power));
                }

                let (a_row_exponents, a_term_evals) =
                    Self::evaluate_terms_bivariate(&p, &a_p, &betas);
                let (b_row_exponents, b_term_evals) =
                    Self::evaluate_terms_bivariate(&p, &b_p, &betas);

                let shift = p.from_element(&p.sample_small_integer(&mut rng, 0..=i64::MAX - 1));
                let mut a_current_evals = a_term_evals
                    .iter()
                    .zip(&a_p.coefficients)
                    .map(|((_, x), coefficient)| p.mul(coefficient, &p.pow(x, shift)))
                    .collect::<Vec<_>>();
                let mut b_current_evals = b_term_evals
                    .iter()
                    .zip(&b_p.coefficients)
                    .map(|((_, x), coefficient)| p.mul(coefficient, &p.pow(x, shift)))
                    .collect::<Vec<_>>();

                let mut gcd_images = Vec::new();
                let mut cofactor_images = Vec::new();
                let mut sample_points = Vec::new();
                let mut next_num_samples = 4usize;

                let selected_image = 'new_sample: loop {
                    for _ in 0..2 {
                        let sample_point = p.pow(&alpha, shift + gcd_images.len() as u64);

                        let a_j = Self::evaluate_geometric_image_bivariate(
                            &p,
                            &a_p,
                            &a_row_exponents,
                            &a_term_evals,
                            &mut a_current_evals,
                        );
                        let b_j = Self::evaluate_geometric_image_bivariate(
                            &p,
                            &b_p,
                            &b_row_exponents,
                            &b_term_evals,
                            &mut b_current_evals,
                        );

                        let a_p_deg = a_p.bivariate_deg();
                        let b_p_deg = b_p.bivariate_deg();
                        if a_j.bivariate_deg() < (a_p_deg.0.to_u32(), a_p_deg.1.to_u32())
                            || b_j.bivariate_deg() < (b_p_deg.0.to_u32(), b_p_deg.1.to_u32())
                        {
                            debug!("Bad bivariate Kronecker image, trying new prime");
                            continue 'new_image;
                        }

                        let g_j = a_j.gcd(&b_j);
                        let g_degree = (g_j.degree(0), g_j.degree(1));
                        if g_degree.0 < d_0_1.0 || g_degree.1 < d_0_1.1 {
                            debug!(
                                "Unlucky bivariate degree bound: {:?} vs {:?}",
                                g_degree, d_0_1
                            );
                            d_0_1 = (g_degree.0.min(d_0_1.0), g_degree.1.min(d_0_1.1));
                            continue 'kronecker_prime;
                        }
                        if g_degree.0 > d_0_1.0 || g_degree.1 > d_0_1.1 {
                            debug!("Unlucky bivariate evaluation point, trying new prime");
                            continue 'new_image;
                        }

                        let lc_a_j = a_j.bivariate_lcoeff();
                        let Some(a_cofactor_j) = a_j.try_div(&g_j) else {
                            debug!("Bivariate image division failed for a, trying new prime");
                            continue 'new_image;
                        };

                        gcd_images.push(g_j * &lc_a_j);
                        cofactor_images.push(a_cofactor_j);
                        sample_points.push(sample_point);
                    }

                    if gcd_images.len() < next_num_samples {
                        continue 'new_sample;
                    }

                    next_num_samples *= 2;

                    if image_kind.is_none() || image_kind == Some(ImageKind::CofactorMultiple) {
                        let a_deg = (a_p.degree(0), a_p.degree(1));
                        let cofactor_image = Self::hu_monagan_sparse_interpolate_bivariate(
                            &p,
                            &cofactor_images,
                            &sample_points,
                            &alpha,
                            &discrete_log_context,
                            &kronecker,
                            (
                                a_deg.0.to_u32().saturating_sub(d_0_1.0),
                                a_deg.1.to_u32().saturating_sub(d_0_1.1),
                            ),
                        );

                        if let Some(cofactor_image) = cofactor_image {
                            image_kind = Some(ImageKind::CofactorMultiple);
                            break 'new_sample cofactor_image;
                        }
                    }

                    if image_kind.is_none() || image_kind == Some(ImageKind::GcdMultiple) {
                        let gcd_image = Self::hu_monagan_sparse_interpolate_bivariate(
                            &p,
                            &gcd_images,
                            &sample_points,
                            &alpha,
                            &discrete_log_context,
                            &kronecker,
                            d_0_1,
                        );

                        if let Some(gcd_image) = gcd_image {
                            image_kind = Some(ImageKind::GcdMultiple);
                            break 'new_sample gcd_image;
                        }
                    }
                };

                let old_h = h.clone();

                if m == 1 {
                    h = selected_image.map_coeff(|c| p.to_symmetric_integer(c), Z);
                    m = p.get_prime().into();
                } else {
                    let crt = IntegerPolynomialCrtContext::new(&m, &p)
                        .expect("Hu-Monagan prime repeated during CRT reconstruction");
                    crt.merge_assign(&mut h, &selected_image);
                    m *= p.get_prime();
                }

                if h != old_h && !old_h.is_zero() {
                    continue 'new_image;
                }

                let image_poly = h.clone();
                let content = image_poly.bivariate_content(0, 1);
                let primitive = image_poly / &content;

                let gcd_candidate = match image_kind.unwrap() {
                    ImageKind::GcdMultiple => PolynomialGCD::normalize(primitive),
                    ImageKind::CofactorMultiple => {
                        let cofactor_candidate = PolynomialGCD::normalize(primitive);
                        if let Some(q) = a.try_div(&cofactor_candidate) {
                            PolynomialGCD::normalize(q)
                        } else if old_h.is_zero() {
                            debug!("Cofactor image does not divide a yet");
                            continue 'new_image;
                        } else {
                            debug!("Stable cofactor image does not divide a");
                            continue 'kronecker_prime;
                        }
                    }
                };

                if a.try_div(&gcd_candidate).is_some() && b.try_div(&gcd_candidate).is_some() {
                    debug!("Found bivariate GCD: {}", gcd_candidate);
                    return Some(PolynomialGCD::normalize(gcd_candidate));
                }

                debug!("Non-division of {}, trying new image", gcd_candidate);

                if !old_h.is_zero() {
                    continue 'kronecker_prime;
                }
            }
        }
    }
}

/// Polynomial GCD functions for a certain coefficient type `Self`.
pub trait PolynomialGCD<E: PositiveExponent>: Ring {
    /// Divide two polynomials exactly, allowing the coefficient domain to
    /// select a more suitable algorithm than generic term division.
    fn try_div_exact(
        dividend: &MultivariatePolynomial<Self, E>,
        divisor: &MultivariatePolynomial<Self, E>,
    ) -> Option<MultivariatePolynomial<Self, E>> {
        dividend.try_div(divisor)
    }

    /// Test exact divisibility. Coefficient domains can override this to avoid
    /// constructing the quotient.
    fn divides_exact(
        dividend: &MultivariatePolynomial<Self, E>,
        divisor: &MultivariatePolynomial<Self, E>,
    ) -> bool {
        Self::try_div_exact(dividend, divisor).is_some()
    }

    /// Tries a coefficient-domain GCD plan before generic univariate content removal.
    ///
    /// `vars` gives the active input coordinates with the generic main variable first, and
    /// `bounds` gives the GCD degree bound in every input coordinate. A returned polynomial must
    /// be the complete GCD of `a` and `b`, including coefficient content, in their coordinate
    /// order. `a_degrees` and `b_degrees` describe the inputs after monomial shifts and common
    /// exponent scales have been removed. Each degree slice has one entry per input coordinate in
    /// the same coordinate order as its polynomial.
    fn gcd_with_precontent_plan(
        _a: &MultivariatePolynomial<Self, E>,
        _b: &MultivariatePolynomial<Self, E>,
        _vars: &[usize],
        _bounds: &[E],
        _a_degrees: &[E],
        _b_degrees: &[E],
    ) -> Option<MultivariatePolynomial<Self, E>> {
        None
    }

    fn heuristic_gcd(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
    ) -> Option<(
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
    )>;
    fn gcd_multiple(f: Vec<MultivariatePolynomial<Self, E>>) -> MultivariatePolynomial<Self, E>;
    fn gcd(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
        vars: &[usize],
        bounds: &mut [E],
    ) -> MultivariatePolynomial<Self, E>;
    fn get_gcd_var_bounds(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
        vars: &[usize],
    ) -> SmallVec<[E; INLINED_EXPONENTS]>;
    fn normalize(a: MultivariatePolynomial<Self, E>) -> MultivariatePolynomial<Self, E>;
}

impl<E: PositiveExponent> PolynomialGCD<E> for IntegerRing {
    #[inline(never)]
    fn gcd_with_precontent_plan(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
        vars: &[usize],
        bounds: &[E],
        a_degrees: &[E],
        b_degrees: &[E],
    ) -> Option<MultivariatePolynomial<Self, E>> {
        let hu_enabled = GLOBAL_SETTINGS
            .force_hu_monagan_poly_gcd
            .load(std::sync::atomic::Ordering::Relaxed)
            || GLOBAL_SETTINGS
                .use_hu_monagan_poly_gcd
                .load(std::sync::atomic::Ordering::Relaxed);
        if !hu_enabled {
            return None;
        }
        if a_degrees.len() != a.nvars() || b_degrees.len() != b.nvars() || a.nvars() != b.nvars() {
            return None;
        }
        if !hu_monagan_has_minimum_geometry(vars, bounds) {
            return None;
        }

        let anchor = HuMonaganAnchor::from_inputs(a, b);
        let (anchored_degrees, other_degrees) = match anchor {
            HuMonaganAnchor::Left => (a_degrees, b_degrees),
            HuMonaganAnchor::Right => (b_degrees, a_degrees),
        };
        if !hu_monagan_plan_is_applicable_with_degrees(a, b, vars, bounds, anchored_degrees) {
            return None;
        }
        let current_variable = *vars.first()?;
        let current_range = hu_monagan_kronecker_range(
            vars,
            bounds,
            anchored_degrees,
            other_degrees,
            current_variable,
        );
        let maximum_modulus = SMOOTH_PRIMES.last()?.0;
        // Making a different variable main must remove a larger radix than it adds. This reduces
        // the interpolation exponent lattice independently of the later coefficient-row test.
        let has_smaller_range = vars.iter().copied().skip(1).any(|variable| {
            bounds[variable] > E::zero()
                && hu_monagan_kronecker_range(
                    vars,
                    bounds,
                    anchored_degrees,
                    other_degrees,
                    variable,
                )
                .is_some_and(|candidate_range| {
                    current_range.is_none_or(|range| candidate_range < range)
                        && candidate_range
                            .checked_mul(2)
                            .is_some_and(|bound| bound <= maximum_modulus)
                })
        });
        if !has_smaller_range {
            return None;
        }
        let planning = HuMonaganPlanningContext::new_with_degrees(
            a, b, vars, bounds, anchor, a_degrees, b_degrees,
        );
        let main_variable = planning.alternative_main_variable()?;
        let prepared = planning.prepare(main_variable)?;
        debug!(
            "Hu main variable {} with maximum coefficient row {} replaces {} with row {}",
            main_variable,
            planning.maximum_row_supports[main_variable],
            vars[0],
            planning.maximum_row_supports[vars[0]],
        );
        prepared.run()
    }

    fn heuristic_gcd(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
    ) -> Option<(
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
    )> {
        // estimate if the heuristic gcd will overflow
        let mut max_deg_a = 0;
        let mut contains_a: SmallVec<[bool; INLINED_EXPONENTS]> = smallvec![false; a.nvars()];
        for t in a {
            let mut deg = 1;
            for (var, e) in t.exponents.iter().enumerate() {
                let v = e.to_u32() as usize;
                if v > 0 {
                    contains_a[var] = true;
                    deg *= v + 1;
                }
            }

            if deg > max_deg_a {
                max_deg_a = deg;
            }
        }

        let mut max_deg_b = 0;
        let mut contains_b: SmallVec<[bool; INLINED_EXPONENTS]> = smallvec![false; b.nvars()];
        for t in b {
            let mut deg = 1;
            for (var, e) in t.exponents.iter().enumerate() {
                let v = e.to_u32() as usize;
                if v > 0 {
                    contains_b[var] = true;
                    deg *= v + 1;
                }
            }

            if deg > max_deg_b {
                max_deg_b = deg;
            }
        }

        let num_shared_vars = contains_a
            .iter()
            .zip(&contains_b)
            .filter(|(a, b)| **a && **b)
            .count();

        let heuristic_allowed = max_deg_a < 20
            || max_deg_b < 20
            || num_shared_vars < 3 && max_deg_a.min(max_deg_b) < 150;

        let mut active_variables = contains_a
            .iter()
            .zip(&contains_b)
            .enumerate()
            .filter_map(|(variable, (in_a, in_b))| (*in_a || *in_b).then_some(variable));
        if let Some(variable) = active_variables.next()
            && active_variables.next().is_none()
            && contains_a[variable]
            && contains_b[variable]
        {
            let evaluation_bits = estimated_heuristic_gcd_evaluation_bits(a, b);
            match select_univariate_integer_gcd(heuristic_allowed, evaluation_bits) {
                UnivariateIntegerGcdAlgorithm::Scalar => {
                    return a.heuristic_gcd_univariate(b, variable).ok();
                }
                UnivariateIntegerGcdAlgorithm::Modular => {
                    debug!(
                        "Using modular univariate integer GCD for an estimated {}-bit scalar image",
                        evaluation_bits
                    );
                    return UnivariateModularGcdContext::new(a, b, variable).run();
                }
            }
        }

        if !heuristic_allowed {
            return None;
        }

        a.heuristic_gcd(b).ok()
    }

    fn gcd_multiple(f: Vec<MultivariatePolynomial<Self, E>>) -> MultivariatePolynomial<Self, E> {
        MultivariatePolynomial::gcd_multiple(f)
    }

    /// Compute the gcd of two multivariate polynomials using a combination of heuristics and Zippel's algorithm.
    /// Assumes a and b are primitive in the main variable.
    fn gcd(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
        vars: &[usize],
        bounds: &mut [E],
    ) -> MultivariatePolynomial<Self, E> {
        let force_hu = GLOBAL_SETTINGS
            .force_hu_monagan_poly_gcd
            .load(std::sync::atomic::Ordering::Relaxed);
        if force_hu
            || (GLOBAL_SETTINGS
                .use_hu_monagan_poly_gcd
                .load(std::sync::atomic::Ordering::Relaxed)
                && should_use_hu_monagan(a, b, vars, bounds))
        {
            // TODO: find out when the bivariate case is faster
            // currently it can be much slower due to the call to bivariate Zippel,
            // that may involve a costly Newton interpolation that has to be called
            // for every sample. Full Zippel would only call it once, since it stores
            // the shape of the polynomial and can reuse it for all samples.
            // if a.nvars() > 3
            //     && vars[1] == 1
            //     && bounds.get(1).is_some_and(|b| *b > E::zero())
            //     && bounds.get(vars[2]).is_some_and(|b| *b > E::zero())
            // {
            //     // if let Some(g) = a.gcd_hu_monagan_bivariate(b, bounds) {
            //     //     return g;
            //     // }
            // } else
            let anchor = HuMonaganAnchor::from_inputs(a, b);
            if let Some(g) = a.gcd_hu_monagan_with_anchor(b, bounds, anchor) {
                return g;
            }
        }

        let mut tight_bounds: SmallVec<[E; INLINED_EXPONENTS]> = bounds.iter().cloned().collect();
        MultivariatePolynomial::gcd_zippel_auto(a, b, vars, bounds, &mut tight_bounds)
    }

    fn get_gcd_var_bounds(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
        vars: &[usize],
    ) -> SmallVec<[E; INLINED_EXPONENTS]> {
        let mut primes = modular_gcd_prime_iterator();

        let mut f = ModularGcdField::new(next_modular_gcd_prime(
            &mut primes,
            "gcd var bound detection",
        ));
        let mut ap = a.map_coeff(|c| c.to_finite_field(&f), f.clone());
        let mut bp = b.map_coeff(|c| c.to_finite_field(&f), f.clone());

        while vars.iter().any(|variable| {
            a.degree(*variable) > E::zero()
                && b.degree(*variable) > E::zero()
                && (ap.degree(*variable) != a.degree(*variable)
                    || bp.degree(*variable) != b.degree(*variable))
        }) {
            debug!("Variable bounds failed due to bad prime");

            let p = next_modular_gcd_prime(&mut primes, "gcd var bound detection");
            f = ModularGcdField::new(p);
            ap = a.map_coeff(|c| c.to_finite_field(&f), f.clone());
            bp = b.map_coeff(|c| c.to_finite_field(&f), f.clone());
        }

        GcdBoundSamplingContext::new(&ap, &bp, vars)
            .and_then(|context| context.sample_bounds(&ap, &bp))
            .unwrap_or_else(|| {
                MultivariatePolynomial::get_gcd_var_bounds_separately(&ap, &bp, vars)
            })
    }

    fn normalize(a: MultivariatePolynomial<Self, E>) -> MultivariatePolynomial<Self, E> {
        if a.lcoeff().is_negative() { -a } else { a }
    }
}

impl<E: PositiveExponent> PolynomialGCD<E> for RationalField {
    fn heuristic_gcd(
        _a: &MultivariatePolynomial<Self, E>,
        _b: &MultivariatePolynomial<Self, E>,
    ) -> Option<(
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
    )> {
        // TODO: restructure
        None
    }

    fn gcd_multiple(f: Vec<MultivariatePolynomial<Self, E>>) -> MultivariatePolynomial<Self, E> {
        MultivariatePolynomial::repeated_gcd(f)
    }

    fn gcd(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
        vars: &[usize],
        bounds: &mut [E],
    ) -> MultivariatePolynomial<Self, E> {
        // remove the content so that the polynomials have integer coefficients
        let a_c = a.content();
        let a_int = a.map_coeff(|c| a.ring().div(c, &a_c).numerator(), Z);
        let b_c = b.content();
        let b_int = b.map_coeff(|c| b.ring().div(c, &b_c).numerator(), Z);

        PolynomialGCD::gcd(&a_int, &b_int, vars, bounds)
            .map_coeff(|c| c.to_rational(), Q)
            .make_monic()
    }

    fn get_gcd_var_bounds(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
        vars: &[usize],
    ) -> SmallVec<[E; INLINED_EXPONENTS]> {
        // remove the content so that the polynomials have integer coefficients
        let a_c = a.content();
        let a_int = a.map_coeff(|c| a.ring().div(c, &a_c).numerator(), Z);
        let b_c = b.content();
        let b_int = b.map_coeff(|c| b.ring().div(c, &b_c).numerator(), Z);

        PolynomialGCD::get_gcd_var_bounds(&a_int, &b_int, vars)
    }

    fn normalize(a: MultivariatePolynomial<Self, E>) -> MultivariatePolynomial<Self, E> {
        a.make_monic()
    }
}

impl<
    UField: FiniteFieldWorkspace,
    F: GaloisField<Base = FiniteField<UField>> + SampleableRing<SamplingPolicy = RangeInclusive<i64>>,
    E: PositiveExponent,
> PolynomialGCD<E> for F
where
    FiniteField<UField>: FiniteFieldCore<UField>,
    <FiniteField<UField> as Set>::Element: Copy,
{
    fn heuristic_gcd(
        _a: &MultivariatePolynomial<Self, E>,
        _b: &MultivariatePolynomial<Self, E>,
    ) -> Option<(
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
    )> {
        None
    }

    fn gcd(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
        vars: &[usize],
        bounds: &mut [E],
    ) -> MultivariatePolynomial<Self, E> {
        assert!(!a.is_zero() || !b.is_zero());
        match MultivariatePolynomial::gcd_shape_modular(
            a,
            b,
            vars,
            bounds,
            &mut bounds
                .iter()
                .cloned()
                .collect::<SmallVec<[E; INLINED_EXPONENTS]>>(),
        ) {
            Some(x) => x,
            None => {
                // upgrade to a Galois field that is large enough
                // TODO: start at a better bound?
                // TODO: run with Zp[var]/m_i instead and use CRT
                let field = a.ring().upgrade(a.ring().extension_degree() as usize + 1);
                let ag = a.map_coeff(|c| a.ring().upgrade_element(c, &field), field.clone());
                let bg = b.map_coeff(|c| a.ring().upgrade_element(c, &field), field.clone());
                let g = PolynomialGCD::gcd(&ag, &bg, vars, bounds);

                // workaround for ICE https://github.com/rust-lang/rust/issues/146965
                // inline the following call: g.map_coeff(|c| a.ring.downgrade_element(c), a.ring.clone())
                let mut coefficients = Vec::with_capacity(g.coefficients.len());
                let mut exponents = Vec::with_capacity(g.exponents.len());

                for m in g.into_iter() {
                    let nc = a.ring().downgrade_element(m.coefficient);
                    if !a.ring().is_zero(&nc) {
                        coefficients.push(nc);
                        exponents.extend(m.exponents);
                    }
                }

                MultivariatePolynomial::from_parts(
                    coefficients,
                    exponents,
                    a.ring().clone(),
                    g.variables().clone(),
                )
            }
        }
    }

    fn get_gcd_var_bounds(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
        vars: &[usize],
    ) -> SmallVec<[E; INLINED_EXPONENTS]> {
        let mut tight_bounds: SmallVec<[_; INLINED_EXPONENTS]> =
            (0..a.nvars()).map(|_| E::zero()).collect();
        for var in vars {
            let vvars: SmallVec<[usize; INLINED_EXPONENTS]> =
                vars.iter().filter(|i| *i != var).cloned().collect();
            tight_bounds[*var] = MultivariatePolynomial::get_gcd_var_bound(a, b, &vvars, *var);
        }
        tight_bounds
    }

    fn gcd_multiple(f: Vec<MultivariatePolynomial<Self, E>>) -> MultivariatePolynomial<Self, E> {
        MultivariatePolynomial::repeated_gcd(f)
    }

    fn normalize(a: MultivariatePolynomial<Self, E>) -> MultivariatePolynomial<Self, E> {
        a.make_monic()
    }
}

impl<E: PositiveExponent> PolynomialGCD<E> for AlgebraicExtension<RationalField> {
    fn try_div_exact(
        dividend: &MultivariatePolynomial<Self, E>,
        divisor: &MultivariatePolynomial<Self, E>,
    ) -> Option<MultivariatePolynomial<Self, E>> {
        if dividend.variables() != divisor.variables() {
            let mut dividend = dividend.clone();
            let mut divisor = divisor.clone();
            dividend.unify_variables(&mut divisor);
            return Self::try_div_exact(&dividend, &divisor);
        }

        let active_variables = (0..dividend.nvars())
            .filter(|&variable| {
                dividend.degree(variable) != E::zero() || divisor.degree(variable) != E::zero()
            })
            .count();
        if active_variables == 1 {
            dividend.try_div_univariate_field(divisor)
        } else {
            dividend.try_div(divisor)
        }
    }

    fn divides_exact(
        dividend: &MultivariatePolynomial<Self, E>,
        divisor: &MultivariatePolynomial<Self, E>,
    ) -> bool {
        assert_eq!(dividend.ring(), divisor.ring());
        assert_eq!(dividend.variables(), divisor.variables());

        if divisor.is_zero() {
            return false;
        }
        if dividend.is_zero() || divisor.is_constant() {
            return true;
        }

        let mut active_variables = (0..dividend.nvars()).filter(|&candidate| {
            dividend.degree(candidate) != E::zero() || divisor.degree(candidate) != E::zero()
        });
        let variable = active_variables
            .next()
            .expect("a nonconstant polynomial must have an active variable");
        if active_variables.next().is_some() {
            return Self::try_div_exact(dividend, divisor).is_some();
        }
        if dividend.degree(variable) < divisor.degree(variable) {
            return false;
        }

        let defining_polynomial = dividend.ring().poly();
        let algebraic_variable = &defining_polynomial.variables()[0];
        assert!(
            dividend
                .variables()
                .iter()
                .position(|candidate| candidate == algebraic_variable)
                .is_none_or(|position| {
                    dividend.degree(position) == E::zero() && divisor.degree(position) == E::zero()
                }),
            "the number-field generator cannot also be an active polynomial variable"
        );

        // Flatten Q(alpha)[x] to Q[alpha, x] and clear all coefficient
        // denominators at once, as in the ordinary rational polynomial GCD.
        let to_integer_associate = |polynomial: &MultivariatePolynomial<RationalField, E>| {
            let content = polynomial.content();
            polynomial.map_coeff(
                |coefficient| polynomial.ring().div(coefficient, &content).numerator(),
                Z,
            )
        };
        let dividend_integer = to_integer_associate(&dividend.from_number_field());
        let divisor_integer = to_integer_associate(&divisor.from_number_field());
        let defining_polynomial_content = defining_polynomial.content();
        let defining_polynomial_integer = defining_polynomial.map_coeff(
            |coefficient| {
                defining_polynomial
                    .ring()
                    .div(coefficient, &defining_polynomial_content)
                    .numerator()
            },
            Z,
        );

        if defining_polynomial_integer
            .ring()
            .is_one(&defining_polynomial_integer.lcoeff())
        {
            // The existing integer algebraic extension reduces every product
            // modulo the defining polynomial and prevents degree growth in
            // alpha during pseudo-division.
            let integer_extension = AlgebraicExtension::new(defining_polynomial_integer);
            let dividend_integer = dividend_integer.to_number_field(&integer_extension);
            let divisor_integer = divisor_integer.to_number_field(&integer_extension);
            return dividend_integer
                .to_univariate_from_univariate(variable)
                .pseudo_remainder(&divisor_integer.to_univariate_from_univariate(variable))
                .is_zero();
        }

        // If denominator clearing makes the defining polynomial non-monic,
        // remain in Z[alpha, x] and reduce the outer pseudo-remainder modulo it.
        let algebraic_variable_position = dividend_integer
            .variables()
            .iter()
            .position(|candidate| candidate == algebraic_variable)
            .expect("flattening must add the number-field generator");
        let remainder = dividend_integer
            .to_univariate(variable)
            .pseudo_remainder(&divisor_integer.to_univariate(variable));
        if remainder.is_zero() {
            return true;
        }

        let defining_polynomial_univariate =
            defining_polynomial_integer.to_univariate_from_univariate(0);
        remainder.coefficients().iter().all(|coefficient| {
            coefficient
                .to_univariate_from_univariate(algebraic_variable_position)
                .pseudo_remainder(&defining_polynomial_univariate)
                .is_zero()
        })
    }

    fn heuristic_gcd(
        _a: &MultivariatePolynomial<Self, E>,
        _b: &MultivariatePolynomial<Self, E>,
    ) -> Option<(
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
    )> {
        None
    }

    fn gcd_multiple(f: Vec<MultivariatePolynomial<Self, E>>) -> MultivariatePolynomial<Self, E> {
        MultivariatePolynomial::repeated_gcd(f)
    }

    fn gcd(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
        vars: &[usize],
        bounds: &mut [E],
    ) -> MultivariatePolynomial<Self, E> {
        let content = a.ring().poly().content().inv();
        let a_integer =
            AlgebraicExtension::new(a.ring().poly().map_coeff(|c| (c * &content).numerator(), Z));
        let a_lcoeff = a_integer.poly().lcoeff();

        debug!("Zippel gcd of {} and {} % {}", a, b, a_integer);
        #[cfg(debug_assertions)]
        {
            a.check_consistency();
            b.check_consistency();
        }

        let mut primes = modular_gcd_prime_iterator();

        let mut tight_bounds: SmallVec<[E; INLINED_EXPONENTS]> = bounds.iter().cloned().collect();

        'newfirstprime: loop {
            let p = next_modular_gcd_prime(&mut primes, "gcd reconstruction");
            let mut finite_field = ModularGcdField::new(p);
            let mut algebraic_field_ff = a.ring().to_finite_field(&finite_field);

            let a_lcoeff_p = a_lcoeff.to_finite_field(&finite_field);

            if finite_field.is_zero(&a_lcoeff_p) {
                continue 'newfirstprime;
            }

            let ap = a.map_coeff(
                |c| c.to_finite_field(&finite_field),
                algebraic_field_ff.clone(),
            );
            let bp = b.map_coeff(
                |c| c.to_finite_field(&finite_field),
                algebraic_field_ff.clone(),
            );

            debug!("New first image: gcd({},{}) mod {}", ap, bp, p);

            // calculate modular gcd image
            let mut gp = match MultivariatePolynomial::gcd_shape_modular(
                &ap,
                &bp,
                vars,
                bounds,
                &mut tight_bounds,
            ) {
                Some(x) => x,
                None => {
                    debug!("Modular GCD failed: getting new prime");
                    continue 'newfirstprime;
                }
            };

            debug!("GCD suggestion: {}", gp);

            bounds[vars[0]] = gp.degree(vars[0]);

            // construct a new assumed form
            // we have to find the proper normalization
            let gfu = gp.to_univariate_polynomial_list(vars[0]);

            // find a coefficient of x1 in gf that is a monomial (single scaling)
            let mut single_scale = None;
            let mut nx = 0; // count the minimal number of samples needed
            for (i, (c, _e)) in gfu.iter().enumerate() {
                if c.nterms() > nx {
                    nx = c.nterms();
                }
                if c.nterms() == 1 {
                    single_scale = Some(i);
                }
            }

            // In the case of multiple scaling, each sample adds an
            // additional unknown, except for the first
            if single_scale.is_none() {
                let mut nx1 = (gp.nterms() - 1) / (gfu.len() - 1);
                if (gp.nterms() - 1) % (gfu.len() - 1) != 0 {
                    nx1 += 1;
                }
                if nx < nx1 {
                    nx = nx1;
                }
                debug!("Multiple scaling case: sample {} times", nx);
            }

            let gpc = gp.lcoeff_varorder(vars);
            let lcoeff_factor = gp.ring().inv(&gpc);

            // construct the gcd suggestion in Z
            // contrary to the integer case, we do not know the leading coefficient in Z
            // as it cannot easily be predicted from the two input polynomials
            // we use rational reconstruction to recover it
            let mut gm: MultivariatePolynomial<AlgebraicExtension<IntegerRing>, E> =
                MultivariatePolynomial::new(&a_integer, gp.nterms().into(), a.variables().clone());
            gm.exponents.clone_from(&gp.exponents);
            gm.coefficients = gp
                .coefficients
                .iter()
                .map(|x| {
                    a_integer.element_from_polynomial(
                        gp.ring()
                            .mul(x, &lcoeff_factor)
                            .poly
                            .map_coeff(|c| finite_field.to_symmetric_integer(c), Z),
                    )
                })
                .collect();

            let mut m = Integer::from_prime(&finite_field); // size of finite field

            debug!("GCD suggestion with gamma: {} mod {} ", gm, p);

            // add new primes until we can reconstruct the full gcd
            'newprime: loop {
                loop {
                    let p = next_modular_gcd_prime(&mut primes, "gcd images");
                    finite_field = ModularGcdField::new(p);
                    algebraic_field_ff = a.ring().to_finite_field(&finite_field);

                    let a_lcoeff_p = a_lcoeff.to_finite_field(&finite_field);

                    if !finite_field.is_zero(&a_lcoeff_p) {
                        break;
                    }
                }

                let ap = a.map_coeff(
                    |c| c.to_finite_field(&finite_field),
                    algebraic_field_ff.clone(),
                );
                let bp = b.map_coeff(
                    |c| c.to_finite_field(&finite_field),
                    algebraic_field_ff.clone(),
                );
                debug!("New image: gcd({},{})", ap, bp);

                // for the univariate case, we don't need to construct an image
                if vars.len() == 1 {
                    gp = ap.univariate_gcd(&bp);
                    if gp.degree(vars[0]) < bounds[vars[0]] {
                        // original image and variable bound unlucky: restart
                        debug!("Unlucky original image: restart");
                        continue 'newfirstprime;
                    }

                    if gp.degree(vars[0]) > bounds[vars[0]] {
                        // prime is probably unlucky
                        debug!("Unlucky current image: try new one");
                        continue 'newprime;
                    }

                    for m in gp.into_iter() {
                        if gfu.iter().all(|(_, pow)| *pow != m.exponents[vars[0]]) {
                            debug!("Bad shape: terms missing");
                            continue 'newfirstprime;
                        }
                    }
                } else {
                    let rec = if let Some(single_scale) = single_scale {
                        MultivariatePolynomial::construct_new_image_single_scale(
                            &ap,
                            &bp,
                            ap.degree(vars[0]),
                            bp.degree(vars[0]),
                            bounds,
                            single_scale,
                            &vars[1..],
                            vars[0],
                            &gfu,
                        )
                    } else {
                        MultivariatePolynomial::construct_new_image_multiple_scales(
                            &ap,
                            &bp,
                            // NOTE: different from paper where they use a.degree(..)
                            // it could be that the degree in ap is lower than that of a
                            // which means the sampling will never terminate
                            ap.degree(vars[0]),
                            bp.degree(vars[0]),
                            bounds,
                            &vars[1..],
                            vars[0],
                            &gfu,
                        )
                    };

                    match rec {
                        Ok(r) => {
                            gp = r;
                        }
                        Err(GCDError::BadOriginalImage) => continue 'newfirstprime,
                        Err(GCDError::BadCurrentImage) => continue 'newprime,
                    }
                }

                // scale the new image
                let gpc = gp.lcoeff_varorder(vars);
                gp = gp.mul_coeff(ap.ring().inv(&gpc));
                debug!("gp: {} mod {}", gp, gp.ring());

                // use chinese remainder theorem to merge coefficients and map back to Z
                // terms could be missing in gp, but not in gm (TODO: check this?)
                let mut gpi = 0;
                for t in 0..gm.nterms() {
                    let gpc = if gm.exponents(t) == gp.exponents(gpi) {
                        gpi += 1;
                        gp.coefficients[gpi - 1].clone()
                    } else {
                        ap.ring().zero()
                    };

                    let gmc_a = &mut gm.coefficients[t];

                    // apply CRT to each integer coefficient in the algebraic number ring
                    let mut gpc_pos = 0;
                    let mut gmc_pos = 0;
                    for i in 0..a.ring().poly().degree(0) {
                        let gpc =
                            if gpc_pos < gpc.poly.nterms() && i == gpc.poly.exponents(gpc_pos)[0] {
                                gpc_pos += 1;
                                Integer::from_finite_field(
                                    &finite_field,
                                    gpc.poly.coefficients[gpc_pos - 1],
                                )
                            } else {
                                Integer::zero()
                            };

                        let gpm = if gmc_pos < gmc_a.poly.nterms()
                            && i == gmc_a.poly.exponents(gmc_pos)[0]
                        {
                            gmc_pos += 1;
                            let r = &gmc_a.poly.coefficients[gmc_pos - 1];
                            if r.is_negative() { r + &m } else { r.clone() }
                        } else {
                            Integer::zero()
                        };

                        let absent = gpm.is_zero();

                        let res = Integer::chinese_remainder(
                            gpm,
                            gpc,
                            m.clone(),
                            Integer::from_prime(&finite_field),
                        );

                        if absent {
                            if !res.is_zero() {
                                gmc_a.poly.append_monomial(res, &[i]);
                                gmc_pos += 1;
                            }
                        } else {
                            assert!(!res.is_zero());
                            gmc_a.poly.coefficients[gmc_pos - 1] = res;
                        }
                    }
                }

                m *= &Integer::from_prime(&finite_field);

                debug!("gm: {} from ring {}", gm, m);

                // do rational reconstruction
                // TODO: don't try every iteration?
                let mut gc = a.zero();

                for c in &gm.coefficients {
                    let mut nc = a.ring().poly().zero();

                    for aa in &c.poly.coefficients {
                        match Rational::maximal_quotient_reconstruction(aa, &m, None) {
                            Ok(x) => nc.coefficients.push(x),
                            Err(e) => {
                                debug!("Bad rational reconstruction: {}", e);
                                // more samples!
                                continue 'newprime;
                            }
                        }
                    }

                    nc.exponents.clone_from(&c.poly.exponents);
                    gc.coefficients.push(a.ring().element_from_polynomial(nc));
                }

                gc.exponents.clone_from(&gm.exponents);

                debug!("Final suggested gcd: {}", gc);
                if gc.is_one() || (Self::divides_exact(a, &gc) && Self::divides_exact(b, &gc)) {
                    return gc;
                }

                // if it does not divide, we need more primes
                debug!("Does not divide: more primes needed");
            }
        }
    }

    fn get_gcd_var_bounds(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
        vars: &[usize],
    ) -> SmallVec<[E; INLINED_EXPONENTS]> {
        let mut bounds: SmallVec<[_; INLINED_EXPONENTS]> =
            (0..a.nvars()).map(|_| E::zero()).collect();
        let mut primes = modular_gcd_prime_iterator();

        let mut f = ModularGcdField::new(next_modular_gcd_prime(
            &mut primes,
            "gcd var bound detection",
        ));
        let mut algebraic_field_ff = a.ring().to_finite_field(&f);
        let mut ap = a.map_coeff(|c| c.to_finite_field(&f), algebraic_field_ff.clone());
        let mut bp = b.map_coeff(|c| c.to_finite_field(&f), algebraic_field_ff.clone());

        for var in vars.iter() {
            if a.degree(*var) == E::zero() || b.degree(*var) == E::zero() {
                continue;
            }

            while ap.degree(*var) != a.degree(*var) || bp.degree(*var) != b.degree(*var) {
                debug!("Variable bounds failed due to bad prime");

                let p = next_modular_gcd_prime(&mut primes, "gcd var bound detection");
                f = ModularGcdField::new(p);
                algebraic_field_ff = a.ring().to_finite_field(&f);
                ap = a.map_coeff(|c| c.to_finite_field(&f), algebraic_field_ff.clone());
                bp = b.map_coeff(|c| c.to_finite_field(&f), algebraic_field_ff.clone());
            }

            let vvars: SmallVec<[usize; INLINED_EXPONENTS]> =
                vars.iter().filter(|i| *i != var).cloned().collect();
            bounds[*var] = MultivariatePolynomial::get_gcd_var_bound(&ap, &bp, &vvars, *var);
        }

        bounds
    }

    fn normalize(a: MultivariatePolynomial<Self, E>) -> MultivariatePolynomial<Self, E> {
        a.make_monic()
    }
}

/// Polynomial GCD functions for floating point coefficient return 1 (for now).
impl<T: SingleFloat + std::hash::Hash + Eq + InternalOrdering, E: PositiveExponent> PolynomialGCD<E>
    for FloatField<T>
{
    fn heuristic_gcd(
        _a: &MultivariatePolynomial<Self, E>,
        _b: &MultivariatePolynomial<Self, E>,
    ) -> Option<(
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
    )> {
        None
    }

    fn gcd_multiple(f: Vec<MultivariatePolynomial<Self, E>>) -> MultivariatePolynomial<Self, E> {
        f[0].one()
    }

    /// Returns 1 (for now).
    fn gcd(
        a: &MultivariatePolynomial<Self, E>,
        _b: &MultivariatePolynomial<Self, E>,
        _vars: &[usize],
        _bounds: &mut [E],
    ) -> MultivariatePolynomial<Self, E> {
        a.one()
    }

    fn get_gcd_var_bounds(
        a: &MultivariatePolynomial<Self, E>,
        _b: &MultivariatePolynomial<Self, E>,
        _vars: &[usize],
    ) -> SmallVec<[E; INLINED_EXPONENTS]> {
        (0..a.nvars()).map(|_| E::zero()).collect()
    }

    fn normalize(a: MultivariatePolynomial<Self, E>) -> MultivariatePolynomial<Self, E> {
        a.one()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atom::AtomCore;
    use crate::domains::finite_field::Z2;
    use crate::parse;
    use crate::poly::PolyVariable;

    fn hu_planning_fixture(expression: &str) -> MultivariatePolynomial<IntegerRing, u8> {
        let variables = ["x", "y", "z"]
            .map(|variable| {
                parse!(variable)
                    .to_polynomial::<IntegerRing, u8>(&Z, None)
                    .variables()[0]
                    .clone()
            })
            .to_vec();
        parse!(expression).to_polynomial::<_, u8>(&Z, Some(std::sync::Arc::new(variables)))
    }

    #[test]
    fn coefficient_row_counter_migrates_without_losing_counts() {
        let mut rows = CoefficientRowCounter::default();
        assert_eq!(rows.increment(0, 4), 1);
        assert_eq!(rows.increment(2, 4), 1);
        assert_eq!(rows.increment(0, 4), 2);
        assert!(rows.sparse.is_none());

        assert_eq!(rows.increment(100, 4), 1);
        assert_eq!(rows.increment(2, 4), 2);
        assert_eq!(rows.increment(100, 4), 2);
        assert_eq!(rows.largest, 2);
        assert!(rows.dense.is_empty());
        assert_eq!(rows.sparse.as_ref().unwrap().len(), 3);
    }

    #[test]
    fn hu_planning_requires_two_sampling_doublings() {
        let left = hu_planning_fixture("1+z+y+y*z^2+y^2*z+y^2*z^3+y^3*z^2+y^3*z^4+x*y^4*z^3");
        let right =
            hu_planning_fixture("1+z+y+y*z^2+y^2*z+y^2*z^3+y^3*z^2+y^3*z^4+x*y^4*z^3+x*z^4");
        let variables = [0, 1, 2];
        let bounds = [1, 4, 4];
        let anchor = HuMonaganAnchor::from_inputs(&left, &right);
        let planning = HuMonaganPlanningContext::new(&left, &right, &variables, &bounds, anchor);

        assert_eq!(anchor, HuMonaganAnchor::Left);
        assert_eq!(planning.maximum_row_supports.as_slice(), [8, 2, 2]);
        assert_eq!(planning.alternative_main_variable(), Some(1));

        let below_threshold = hu_planning_fixture("1+z+y+y*z^2+y^2*z+y^2*z^3+y^3*z^2+x*y^4*z^3");
        let larger = hu_planning_fixture("1+z+y+y*z^2+y^2*z+y^2*z^3+y^3*z^2+x*y^4*z^3+x*z^4");
        let planning = HuMonaganPlanningContext::new(
            &below_threshold,
            &larger,
            &variables,
            &bounds,
            HuMonaganAnchor::Left,
        );
        assert_eq!(planning.maximum_row_supports[0], 7);
        assert_eq!(planning.maximum_row_supports[1], usize::MAX);
        assert_eq!(planning.alternative_main_variable(), None);

        let uniform = hu_planning_fixture("(1+x+x^2)*(1+y+y^2)*(1+z+z^2)");
        let larger = hu_planning_fixture("(1+x+x^2)*(1+y+y^2)*(1+z+z^2)+x^3*y^3*z^3");
        let planning = HuMonaganPlanningContext::new(
            &uniform,
            &larger,
            &variables,
            &bounds,
            HuMonaganAnchor::Left,
        );
        assert_eq!(planning.maximum_row_supports[0], 9);
        assert_eq!(planning.maximum_row_supports[1], usize::MAX);
        assert_eq!(planning.maximum_row_supports[2], usize::MAX);
        assert_eq!(planning.alternative_main_variable(), None);
    }

    #[test]
    fn hu_planning_accounts_for_image_degree_and_kronecker_range() {
        let variables = [0, 1, 2];
        let bounds = [1, 1, 1];

        let high_image_degree = hu_planning_fixture("1+y^30+y^60+y^90+y^120+y^150+y^180+y^210+x*z");
        let larger = hu_planning_fixture("1+y^30+y^60+y^90+y^120+y^150+y^180+y^210+x*z+x*z^2");
        let planning = HuMonaganPlanningContext::new(
            &high_image_degree,
            &larger,
            &variables,
            &bounds,
            HuMonaganAnchor::Left,
        );
        assert_eq!(planning.maximum_row_supports[0], 8);
        assert_eq!(planning.maximum_row_supports[1], usize::MAX);
        assert!(
            hu_monagan_main_image_work(&planning.anchored_degrees, &planning.other_degrees, 1, 2,)
                > planning.main_image_work(0)
        );
        assert!(planning.kronecker_range(1) < planning.kronecker_range(0));
        assert_eq!(planning.alternative_main_variable(), None);

        let larger_kronecker_range = hu_planning_fixture("1+y+y^2+y^3+y^4+y^5+y^6+y^7+x^100*z");
        let larger = hu_planning_fixture("1+y+y^2+y^3+y^4+y^5+y^6+y^7+x^100*z+x^100*z^2");
        let planning = HuMonaganPlanningContext::new(
            &larger_kronecker_range,
            &larger,
            &variables,
            &bounds,
            HuMonaganAnchor::Left,
        );
        assert_eq!(planning.maximum_row_supports[0], 8);
        assert_eq!(planning.maximum_row_supports[1], usize::MAX);
        assert!(
            hu_monagan_main_image_work(&planning.anchored_degrees, &planning.other_degrees, 1, 2,)
                < planning.main_image_work(0)
        );
        assert!(planning.kronecker_range(1) > planning.kronecker_range(0));
        assert_eq!(planning.alternative_main_variable(), None);
    }

    #[test]
    fn hu_planning_preserves_anchor_and_restores_new_main_content() {
        let mut polynomials = [
            hu_planning_fixture("1+z"),
            hu_planning_fixture("1+z+z^2"),
            hu_planning_fixture(
                "(1+z^8)*(1+z+y+y*z^2+y^2*z+y^2*z^3+y^3*z^2+y^3*z^4+x^2*y^4*z^3+x^3*y^4*z^4)",
            ),
        ];
        MultivariatePolynomial::unify_variables_list(&mut polynomials);
        let [left_cofactor, right_cofactor, common_factor] = polynomials;
        let left = &left_cofactor * &common_factor;
        let right = &right_cofactor * &common_factor;
        let variables = [0, 1, 2];
        let bounds = [3, 4, 12];
        let anchor = HuMonaganAnchor::from_inputs(&left, &right);
        assert!(
            variables
                .iter()
                .all(|variable| left.degree(*variable) > 1 && right.degree(*variable) > 1)
        );
        assert!(<IntegerRing as PolynomialGCD<u8>>::heuristic_gcd(&left, &right).is_none());
        assert_eq!(
            HuMonaganAnchor::from_inputs(&right, &left),
            HuMonaganAnchor::Right
        );
        assert_eq!(
            HuMonaganAnchor::from_inputs(&left, &left),
            HuMonaganAnchor::Left
        );
        let planning = HuMonaganPlanningContext::new(&left, &right, &variables, &bounds, anchor);
        let prepared = planning.prepare(1).unwrap();

        assert_eq!(prepared.anchor, anchor);
        assert!(prepared.left.univariate_content(0).is_one());
        assert!(prepared.right.univariate_content(0).is_one());
        assert_eq!(prepared.run(), Some(common_factor.clone()));

        let optimized = <IntegerRing as PolynomialGCD<u8>>::gcd_with_precontent_plan(
            &left,
            &right,
            &variables,
            &bounds,
            &polynomial_degrees(&left),
            &polynomial_degrees(&right),
        );
        assert_eq!(optimized, Some(common_factor.clone()));

        let transform = |mut polynomial: MultivariatePolynomial<IntegerRing, u8>,
                         shift: [u8; 3]| {
            for exponents in polynomial.exponents_iter_mut() {
                for ((exponent, scale), offset) in exponents.iter_mut().zip([2, 3, 2]).zip(shift) {
                    *exponent = *exponent * scale + offset;
                }
            }
            polynomial
        };
        let shifted_left = transform(left, [3, 2, 1]);
        let shifted_right = transform(right, [1, 4, 2]);
        let expected = transform(common_factor, [1, 2, 1]);
        assert_eq!(shifted_left.gcd(&shifted_right), expected);
        assert_eq!(shifted_right.gcd(&shifted_left), expected);
    }

    #[test]
    fn hu_precontent_plan_requires_sparse_interpolation_geometry() {
        let left = hu_planning_fixture("1+z+y+y*z^2+y^2*z+y^2*z^3+y^3*z^2+y^3*z^4+x*y^4*z^3");
        let right =
            hu_planning_fixture("1+z+y+y*z^2+y^2*z+y^2*z^3+y^3*z^2+y^3*z^4+x*y^4*z^3+x*z^4");
        let variables = [0, 1, 2];
        let bounds = [1, 1, 1];
        let anchor = HuMonaganAnchor::from_inputs(&left, &right);
        let left_degrees = polynomial_degrees(&left);
        let right_degrees = polynomial_degrees(&right);
        let planning = HuMonaganPlanningContext::new(&left, &right, &variables, &bounds, anchor);

        assert_eq!(planning.alternative_main_variable(), Some(1));
        assert!(!hu_monagan_plan_is_applicable_with_degrees(
            &left,
            &right,
            &variables,
            &bounds,
            &left_degrees,
        ));
        assert_eq!(
            <IntegerRing as PolynomialGCD<u8>>::gcd_with_precontent_plan(
                &left,
                &right,
                &variables,
                &bounds,
                &left_degrees,
                &right_degrees,
            ),
            None,
        );
    }

    #[test]
    fn dense_univariate_gcd_handles_sparse_images_and_constants() {
        let field = Zp::new(2_147_483_659);
        let factor = parse!("x^7+3*x^4+5*x+2").to_polynomial::<_, u8>(&field, None);
        let left_cofactor =
            parse!("x^5+7*x^2+11").to_polynomial::<_, u8>(&field, factor.variables().clone());
        let right_cofactor =
            parse!("x^4+13*x^3+17").to_polynomial::<_, u8>(&field, factor.variables().clone());
        let left = &factor * &left_cofactor;
        let right = &factor * &right_cofactor;

        assert_eq!(left.univariate_gcd(&right), factor);
        assert_eq!(
            left.univariate_gcd(&left.constant(field.nth(Integer::from(3)))),
            left.one()
        );
        assert_eq!(
            left.constant(field.nth(Integer::from(3)))
                .univariate_gcd(&left.constant(field.nth(Integer::from(5)))),
            left.one()
        );
        assert_eq!(left.zero().univariate_gcd(&left), left);

        let large_gap = parse!("x^100000+x^50000").to_polynomial::<_, u32>(&field, None);
        let monomial =
            parse!("x^75000").to_polynomial::<_, u32>(&field, large_gap.variables().clone());
        let expected =
            parse!("x^50000").to_polynomial::<_, u32>(&field, large_gap.variables().clone());
        assert_eq!(large_gap.univariate_gcd(&monomial), expected);

        let sparse_left =
            parse!("x^100000+1").to_polynomial::<_, u32>(&field, large_gap.variables().clone());
        let sparse_right =
            parse!("x^99999+2").to_polynomial::<_, u32>(&field, large_gap.variables().clone());
        assert!(
            !DenseUnivariateGcdContext::new(&sparse_left, &sparse_right)
                .storage_is_bounded(&sparse_left, &sparse_right)
        );
    }

    #[test]
    fn shifted_transposed_vandermonde_recovers_coefficients() {
        let field = Zp::new(2_147_483_659);
        let polynomial = parse!("x").to_polynomial::<_, u8>(&field, None);
        assert!(
            polynomial
                .solve_shifted_transposed_vandermonde(&[], &[])
                .is_empty()
        );

        for len in 1..=12 {
            let points = (0..len)
                .map(|index| field.to_element(index as u32 + 2))
                .collect::<Vec<_>>();
            let expected = (0..len)
                .map(|index| field.to_element(if index % 4 == 0 { 0 } else { index as u32 + 11 }))
                .collect::<Vec<_>>();
            let rhs = (0..len)
                .map(|power| {
                    points.iter().zip(&expected).fold(
                        field.zero(),
                        |mut value, (point, coefficient)| {
                            field.add_mul_assign(
                                &mut value,
                                coefficient,
                                &field.pow(point, power as u64 + 1),
                            );
                            value
                        },
                    )
                })
                .collect::<Vec<_>>();

            assert_eq!(
                polynomial.solve_shifted_transposed_vandermonde(&points, &rhs),
                expected
            );
        }
    }

    #[test]
    fn zippel_shape_index_keeps_large_degree_gaps_sparse() {
        let sparse = ZippelShapeIndex::new([3u32, 1_000_003]);
        assert!(matches!(&sparse, ZippelShapeIndex::Sparse(_)));
        assert_eq!(sparse.get(3), Some(0));
        assert_eq!(sparse.get(1_000_003), Some(1));
        assert_eq!(sparse.get(4), None);

        let dense = ZippelShapeIndex::new([3u32, 5, 6]);
        assert!(matches!(&dense, ZippelShapeIndex::Dense { .. }));
        assert_eq!(dense.get(5), Some(1));
    }

    #[test]
    fn dense_univariate_gcd_selector_rejects_sparse_gap_images() {
        let field = Zp::new(2_147_483_659);
        let sparse_left = parse!("x^260+x^256+1").to_polynomial::<_, u16>(&field, None);
        let sparse_right = parse!("x^259+2*x^128+3")
            .to_polynomial::<_, u16>(&field, sparse_left.variables().clone());
        assert!(
            !DenseUnivariateGcdContext::new(&sparse_left, &sparse_right)
                .storage_is_bounded(&sparse_left, &sparse_right)
        );

        let dense_left =
            parse!("(1+x)^20").to_polynomial::<_, u16>(&field, sparse_left.variables().clone());
        let dense_right =
            parse!("(1+2*x)^18").to_polynomial::<_, u16>(&field, sparse_left.variables().clone());
        assert!(
            DenseUnivariateGcdContext::new(&dense_left, &dense_right)
                .storage_is_bounded(&dense_left, &dense_right)
        );
    }

    #[test]
    fn modular_gcd_primes_start_with_the_workspace_prime() {
        let known_prime = u32::get_large_prime() as u64;
        assert!(Integer::from(known_prime).is_prime(0));
        let mut primes = ModularGcdPrimeIterator::for_workspace::<u32>();
        assert_eq!(primes.next(), Some(known_prime));

        let successor = primes.next().unwrap();
        assert!(successor > known_prime);
        assert!(Integer::from(successor).is_prime(0));

        let u64_lower_bound = u64::get_large_prime();
        let mut u64_primes = ModularGcdPrimeIterator::for_workspace::<u64>();
        let first_u64_prime = u64_primes.next().unwrap();
        assert!(first_u64_prime > u64_lower_bound);
        assert!(Integer::from(first_u64_prime).is_prime(0));
        let successor = u64_primes.next().unwrap();
        assert!(successor > first_u64_prime);
        assert!(Integer::from(successor).is_prime(0));
    }

    #[test]
    fn univariate_modular_gcd_primes_match_dynamic_iterator() {
        assert_eq!(UNIVARIATE_U64_MODULAR_GCD_PRIMES.len(), 32);

        let mut actual = univariate_modular_gcd_prime_iterator();
        let mut expected = PrimeIteratorU64::new(u64::get_large_prime());

        // Compare every fixed prime and the first dynamically discovered fallback.
        for _ in 0..=UNIVARIATE_U64_MODULAR_GCD_PRIMES.len() {
            assert_eq!(actual.next(), expected.next());
        }
    }

    #[cfg(not(feature = "binary_size"))]
    #[test]
    fn zippel_word_selector_uses_high_gamma() {
        let polynomial = parse!("x+1").to_polynomial::<_, u16>(&Z, None);
        let gamma = Integer::from(1) << (U64_ZIPPEL_HEIGHT_BITS as usize - 1);
        assert!(should_use_u64_zippel(&polynomial, &polynomial, &gamma));
    }

    #[cfg(not(feature = "binary_size"))]
    #[test]
    fn zippel_word_selector_uses_two_high_inputs() {
        let coefficient = Integer::from(1) << (U64_ZIPPEL_HEIGHT_BITS as usize - 1);
        let polynomial = parse!(&format!("{coefficient}*x+1")).to_polynomial::<_, u16>(&Z, None);
        assert!(should_use_u64_zippel(
            &polynomial,
            &polynomial,
            &Integer::from(1)
        ));
    }

    #[cfg(not(feature = "binary_size"))]
    #[test]
    fn zippel_word_selector_keeps_one_high_input_on_u32() {
        let coefficient = Integer::from(1) << (U64_ZIPPEL_HEIGHT_BITS as usize - 1);
        let high = parse!(&format!("{coefficient}*x+1")).to_polynomial::<_, u16>(&Z, None);
        let low = parse!("x+1").to_polynomial::<_, u16>(&Z, None);
        assert!(!should_use_u64_zippel(&high, &low, &Integer::from(1)));
        assert!(!should_use_u64_zippel(&low, &high, &Integer::from(1)));
    }

    #[test]
    fn fused_gcd_bound_images_match_per_variable_sampling() {
        fn check_case(
            left_cofactor: &str,
            right_cofactor: &str,
            common_factor: &str,
            variable_order: &[usize],
        ) {
            let mut polynomials = [
                parse!(left_cofactor).to_polynomial::<_, u8>(&Z, None),
                parse!(right_cofactor).to_polynomial::<_, u8>(&Z, None),
                parse!(common_factor).to_polynomial::<_, u8>(&Z, None),
            ];
            MultivariatePolynomial::unify_variables_list(&mut polynomials);
            let [left_cofactor, right_cofactor, common_factor] = polynomials;
            let left = &left_cofactor * &common_factor;
            let right = &right_cofactor * &common_factor;

            let mut primes = modular_gcd_prime_iterator();
            let field =
                ModularGcdField::new(next_modular_gcd_prime(&mut primes, "fused GCD bound test"));
            let left_mod = left.map_coeff(
                |coefficient| coefficient.to_finite_field(&field),
                field.clone(),
            );
            let right_mod = right.map_coeff(
                |coefficient| coefficient.to_finite_field(&field),
                field.clone(),
            );

            let mut context =
                GcdBoundSamplingContext::new(&left_mod, &right_mod, variable_order).unwrap();
            let mut points = vec![field.one(); left_mod.nvars()];
            for (index, variable) in variable_order.iter().enumerate() {
                let point = field.to_element((index as u64 + 101) as ModularGcdFieldWorkspace);
                points[*variable] = point;
                context.set_point(*variable, point);
            }
            context.fill_images(&left_mod, true);
            context.fill_images(&right_mod, false);
            assert!(context.degrees_are_preserved());

            for (image_index, variable) in context.retained_variables.iter().enumerate() {
                let sampled_variables = variable_order
                    .iter()
                    .filter(|sampled_variable| *sampled_variable != variable)
                    .map(|sampled_variable| (*sampled_variable, points[*sampled_variable]))
                    .collect::<Vec<_>>();
                let mut cache = (0..left_mod.nvars())
                    .map(|sampled_variable| {
                        vec![
                            field.zero();
                            min(
                                max(
                                    left_mod.degree(sampled_variable),
                                    right_mod.degree(sampled_variable)
                                )
                                .to_u32() as usize
                                    + 1,
                                POW_CACHE_SIZE
                            )
                        ]
                    })
                    .collect::<Vec<_>>();
                let mut terms =
                    HashMap::with_capacity_and_hasher(INITIAL_POW_MAP_SIZE, Default::default());
                let reference_left = left_mod.sample_polynomial(
                    *variable,
                    &sampled_variables,
                    &mut cache,
                    &mut terms,
                );
                let reference_right = right_mod.sample_polynomial(
                    *variable,
                    &sampled_variables,
                    &mut cache,
                    &mut terms,
                );
                let fused_left = GcdBoundSamplingContext::image_polynomial(
                    &field,
                    &left_mod,
                    *variable,
                    context.left_images[image_index].clone(),
                );
                let fused_right = GcdBoundSamplingContext::image_polynomial(
                    &field,
                    &right_mod,
                    *variable,
                    context.right_images[image_index].clone(),
                );
                assert_eq!(fused_left, reference_left);
                assert_eq!(fused_right, reference_right);
            }

            let fused = context.bounds_from_images(&left_mod, &right_mod);
            for variable in variable_order {
                assert_eq!(fused[*variable], common_factor.degree(*variable));
            }
        }

        check_case(
            "1+2*x1+3*x2^2+5*x3*x4+7*x5^2",
            "2+3*x1^2+5*x2+7*x3^2+11*x4*x5",
            "3+x1+x2*x3+x4^2+x5^3",
            &[4, 0, 3, 1, 2],
        );
        check_case(
            "1+x1+2*x2^2+3*x3*x4+5*x5*x6+7*x7^2+11*x8",
            "2+3*x1*x2+5*x3^2+7*x4+11*x5*x7+13*x6^2+17*x8^2",
            "5+x1*x8+x2*x7+x3*x6+x4*x5",
            &[7, 2, 5, 0, 6, 1, 4, 3],
        );
    }

    #[test]
    fn gcd_base_degree_scan_handles_mixed_degrees_and_absent_variables() {
        let mut polynomials = [
            parse!("1+x^2*y^3").to_polynomial::<_, u8>(&Z, None),
            parse!("1+x^4+z^5").to_polynomial::<_, u8>(&Z, None),
            parse!("1+y^6+w^7").to_polynomial::<_, u8>(&Z, None),
            parse!("q").to_polynomial::<_, u8>(&Z, None),
        ];
        MultivariatePolynomial::unify_variables_list(&mut polynomials);
        let [common, left_cofactor, right_cofactor, _absent_variable] = polynomials;
        let left = &common * &left_cofactor;
        let right = &common * &right_cofactor;
        assert_eq!(left.gcd(&right), common);

        let mut polynomials = [
            parse!("1+x*y+y*z+z*x").to_polynomial::<_, u8>(&Z, None),
            parse!("1+x^2+y^2+z^2").to_polynomial::<_, u8>(&Z, None),
            parse!("2+x^3+y^3+z^3").to_polynomial::<_, u8>(&Z, None),
        ];
        MultivariatePolynomial::unify_variables_list(&mut polynomials);
        let [common, left_cofactor, right_cofactor] = polynomials;
        let left = &common * &left_cofactor;
        let right = &common * &right_cofactor;
        assert_eq!(left.gcd(&right), common);
    }

    #[test]
    fn gcd_input_metadata_tracks_monomial_shifts() {
        let polynomial = parse!("x^3*y^5*z^2*w^6 + 2*x^7*y^5*w^6 + 3*x^5*y^9*z*w^6")
            .to_polynomial::<_, u16>(&Z, None);
        let metadata = GcdInputMetadata::scan(&polynomial);

        for variable in 0..polynomial.nvars() {
            let (minimum, maximum) = polynomial.degree_bounds(variable);
            assert_eq!(metadata.variables[variable].min_degree, minimum);
            assert_eq!(metadata.variables[variable].max_degree, maximum);
            assert_eq!(metadata.shifted_degree(variable), maximum - minimum);
            assert_eq!(metadata.occurs_after_shift(variable), minimum != maximum);
        }

        let mut shifted = Cow::Owned(polynomial);
        metadata.remove_monomial_shift(&mut shifted);
        for variable in 0..shifted.nvars() {
            assert_eq!(
                shifted.degree_bounds(variable),
                (0, metadata.shifted_degree(variable))
            );
        }
    }

    #[test]
    fn gcd_metadata_preserves_shifts_powers_and_unified_variables() {
        let left = parse!("x^3*y^5*(x^4+y^2+1)*(z+1)").to_polynomial::<_, u16>(&Z, None);
        let right = parse!("x^2*y^7*(x^4+y^2+1)*(w+1)").to_polynomial::<_, u16>(&Z, None);
        let mut expected = parse!("x^2*y^5*(x^4+y^2+1)").to_polynomial::<_, u16>(&Z, None);
        let mut actual = left.gcd(&right);
        actual.unify_variables(&mut expected);

        assert_eq!(actual, expected);
    }

    #[test]
    fn galois_gcd_upgrade_samples_outside_the_prime_subfield() {
        let field = AlgebraicExtension::galois_field(Z2, 2, PolyVariable::Temporary(0));
        let mut factors = [
            parse!("x+y^2+y+1").to_polynomial::<_, u8>(&field, None),
            parse!("x+y+1").to_polynomial::<_, u8>(&field, None),
            parse!("x^2+x+y+1").to_polynomial::<_, u8>(&field, None),
        ];
        MultivariatePolynomial::unify_variables_list(&mut factors);
        let [common, left_cofactor, right_cofactor] = factors;
        let left = &common * &left_cofactor;
        let right = &common * &right_cofactor;

        assert_eq!(left.gcd(&right), common.make_monic());
    }

    #[test]
    fn integer_heuristic_reconstructs_dense_bivariate_gcd() {
        let mut polynomials = [
            parse!("(1+3*x+5*y)^5-1").to_polynomial::<_, u16>(&Z, None),
            parse!("(1-3*x-5*y)^5+1").to_polynomial::<_, u16>(&Z, None),
            parse!("(1+3*x-5*y)^5+3").to_polynomial::<_, u16>(&Z, None),
        ];
        MultivariatePolynomial::unify_variables_list(&mut polynomials);
        let [left_cofactor, right_cofactor, common_factor] = polynomials;
        let left = &left_cofactor * &common_factor;
        let right = &right_cofactor * &common_factor;

        let (actual, actual_left_cofactor, actual_right_cofactor) =
            left.heuristic_gcd(&right).unwrap();
        assert!(actual == common_factor || actual == -common_factor);
        assert_eq!(&actual * &actual_left_cofactor, left);
        assert_eq!(&actual * &actual_right_cofactor, right);
    }

    #[test]
    fn integer_heuristic_uses_univariate_horner_path() {
        let [left_cofactor, right_cofactor, common_factor] = [
            parse!("(1+3*x)^32-1").to_polynomial::<_, u16>(&Z, None),
            parse!("(1-3*x)^32+1").to_polynomial::<_, u16>(&Z, None),
            parse!("(1-3*x)^32+3").to_polynomial::<_, u16>(&Z, None),
        ];
        let left = &left_cofactor * &common_factor;
        let right = &right_cofactor * &common_factor;

        let (actual, actual_left_cofactor, actual_right_cofactor) =
            <IntegerRing as PolynomialGCD<u16>>::heuristic_gcd(&left, &right).unwrap();
        assert!(actual == common_factor || actual == -common_factor.clone());
        assert_eq!(&actual * &actual_left_cofactor, left);
        assert_eq!(&actual * &actual_right_cofactor, right);
        assert_eq!(left.gcd(&right), common_factor);

        let mut inactive_left = left.clone();
        let mut inactive_right = right.clone();
        let mut inactive_expected = common_factor.clone();
        let mut variable_template = parse!("y+1").to_polynomial::<IntegerRing, u16>(&Z, None);
        inactive_left.unify_variables(&mut variable_template);
        inactive_right.unify_variables(&mut variable_template);
        inactive_expected.unify_variables(&mut variable_template);
        assert_eq!(inactive_left.nvars(), 2);
        let (inactive_gcd, inactive_left_cofactor, inactive_right_cofactor) =
            <IntegerRing as PolynomialGCD<u16>>::heuristic_gcd(&inactive_left, &inactive_right)
                .unwrap();
        assert_eq!(inactive_gcd, inactive_expected);
        assert_eq!(&inactive_gcd * &inactive_left_cofactor, inactive_left);
        assert_eq!(&inactive_gcd * &inactive_right_cofactor, inactive_right);

        let variables_before_active = std::sync::Arc::new(vec![
            PolyVariable::Temporary(0),
            common_factor.variables()[0].clone(),
        ]);
        let [
            preceding_left_cofactor,
            preceding_right_cofactor,
            preceding_common_factor,
        ] = [
            parse!("(1+3*x)^32-1")
                .to_polynomial::<_, u16>(&Z, Some(variables_before_active.clone())),
            parse!("(1-3*x)^32+1")
                .to_polynomial::<_, u16>(&Z, Some(variables_before_active.clone())),
            parse!("(1-3*x)^32+3").to_polynomial::<_, u16>(&Z, Some(variables_before_active)),
        ];
        let preceding_left = &preceding_left_cofactor * &preceding_common_factor;
        let preceding_right = &preceding_right_cofactor * &preceding_common_factor;
        assert_eq!(preceding_left.degree(0), 0);
        assert_ne!(preceding_left.degree(1), 0);
        let (preceding_gcd, preceding_left_result, preceding_right_result) =
            <IntegerRing as PolynomialGCD<u16>>::heuristic_gcd(&preceding_left, &preceding_right)
                .unwrap();
        assert!(
            preceding_gcd == preceding_common_factor || preceding_gcd == -preceding_common_factor
        );
        assert_eq!(&preceding_gcd * &preceding_left_result, preceding_left);
        assert_eq!(&preceding_gcd * &preceding_right_result, preceding_right);

        let scaled_left = left.mul_coeff(Integer::from(6));
        let scaled_right = right.mul_coeff(Integer::from(10));
        let (scaled_gcd, scaled_left_cofactor, scaled_right_cofactor) =
            <IntegerRing as PolynomialGCD<u16>>::heuristic_gcd(&scaled_left, &scaled_right)
                .unwrap();
        assert_eq!(scaled_gcd, common_factor.mul_coeff(Integer::from(2)));
        assert_eq!(&scaled_gcd * &scaled_left_cofactor, scaled_left);
        assert_eq!(&scaled_gcd * &scaled_right_cofactor, scaled_right);
    }

    #[test]
    fn univariate_integer_gcd_selector_separates_scalar_and_modular_images() {
        let [left_cofactor_32, right_cofactor_32, common_factor_32] = [
            parse!("(1+3*x)^32-1").to_polynomial::<_, u16>(&Z, None),
            parse!("(1-3*x)^32+1").to_polynomial::<_, u16>(&Z, None),
            parse!("(1-3*x)^32+3").to_polynomial::<_, u16>(&Z, None),
        ];
        let left_32 = &left_cofactor_32 * &common_factor_32;
        let right_32 = &right_cofactor_32 * &common_factor_32;
        assert_eq!(
            select_univariate_integer_gcd(
                true,
                estimated_heuristic_gcd_evaluation_bits(&left_32, &right_32),
            ),
            UnivariateIntegerGcdAlgorithm::Scalar,
        );

        let [left_cofactor_48, right_cofactor_48, common_factor_48] = [
            parse!("(1+3*x)^48-1").to_polynomial::<_, u16>(&Z, None),
            parse!("(1-3*x)^48+1").to_polynomial::<_, u16>(&Z, None),
            parse!("(1-3*x)^48+3").to_polynomial::<_, u16>(&Z, None),
        ];
        let left_48 = &left_cofactor_48 * &common_factor_48;
        let right_48 = &right_cofactor_48 * &common_factor_48;
        assert_eq!(
            select_univariate_integer_gcd(
                true,
                estimated_heuristic_gcd_evaluation_bits(&left_48, &right_48),
            ),
            UnivariateIntegerGcdAlgorithm::Modular,
        );
        let (actual_48, left_result_48, right_result_48) =
            <IntegerRing as PolynomialGCD<u16>>::heuristic_gcd(&left_48, &right_48).unwrap();
        assert_eq!(actual_48, common_factor_48);
        assert_eq!(&actual_48 * &left_result_48, left_48);
        assert_eq!(&actual_48 * &right_result_48, right_48);

        // Generated degree-80 factors produce degree-160 inputs, which the scalar heuristic's
        // existing degree gate rejects before considering the evaluation-size estimate.
        let max_deg_left_80 = 2usize * 80 + 1;
        let max_deg_right_80 = max_deg_left_80;
        let num_shared_variables_80 = 1usize;
        let scalar_heuristic_allowed_80 = max_deg_left_80 < 20
            || max_deg_right_80 < 20
            || num_shared_variables_80 < 3 && max_deg_left_80.min(max_deg_right_80) < 150;
        assert_eq!(
            select_univariate_integer_gcd(scalar_heuristic_allowed_80, 0),
            UnivariateIntegerGcdAlgorithm::Modular,
        );
    }

    #[test]
    fn dense_zp64_leading_inverse_matches_field_inverse() {
        for prime in univariate_modular_gcd_prime_iterator()
            .take(UNIVARIATE_U64_MODULAR_GCD_PRIMES.len() + 4)
        {
            let field = Zp64::new(prime);
            let mut residues = vec![1, 2, prime / 2, prime - 2, prime - 1];
            let mut state = prime ^ 0xd1b5_4a32_d192_ed03;
            for _ in 0..512 {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let residue = if state >= prime { state - prime } else { state };
                residues.push(residue.max(1));
            }

            for residue in residues {
                let coefficient = field.to_element(residue);
                let inverse =
                    DenseZp64UnivariateGcdImage::<u16>::inverse_leading(&field, &coefficient);
                assert_eq!(
                    inverse,
                    field.inv(&coefficient),
                    "inverse of {residue} modulo {prime}",
                );
                assert_eq!(
                    field.mul(&coefficient, &inverse),
                    field.one(),
                    "inverse product of {residue} modulo {prime}",
                );
            }
        }
    }

    #[test]
    fn dense_zp64_univariate_gcd_image_matches_monic_field_gcd() {
        let field = Zp64::new(
            ModularGcdPrimeIterator::for_workspace::<u64>()
                .next()
                .unwrap(),
        );
        let [left_cofactor, right_cofactor, common_factor] = [
            parse!("(1-3*x)^11+5").to_polynomial::<_, u16>(&Z, None),
            parse!("(1+5*x)^10+7").to_polynomial::<_, u16>(&Z, None),
            parse!("(1+2*x)^12+3").to_polynomial::<_, u16>(&Z, None),
        ];
        let left = &left_cofactor * &common_factor;
        let right = &right_cofactor * &common_factor;
        let leading = Integer::from(37).to_finite_field(&field);

        let actual = DenseZp64UnivariateGcdImage::new(&right, &left, 0, &field)
            .unwrap()
            .run(leading);
        let left_image = left.map_coeff(
            |coefficient| coefficient.to_finite_field(&field),
            field.clone(),
        );
        let right_image = right.map_coeff(
            |coefficient| coefficient.to_finite_field(&field),
            field.clone(),
        );
        let expected = left_image.univariate_gcd(&right_image).mul_coeff(leading);
        assert_eq!(actual, expected);
        assert_eq!(actual.lcoeff(), leading);
    }

    #[test]
    fn dense_zp64_univariate_gcd_image_handles_inactive_and_sparse_variables() {
        let field = Zp64::new(
            ModularGcdPrimeIterator::for_workspace::<u64>()
                .next()
                .unwrap(),
        );
        let variable_template = parse!("x").to_polynomial::<IntegerRing, u16>(&Z, None);
        let variables = std::sync::Arc::new(vec![
            PolyVariable::Temporary(0),
            variable_template.variables()[0].clone(),
        ]);
        let left = parse!("(x+1)^15*(x+2)^8").to_polynomial::<_, u16>(&Z, Some(variables.clone()));
        let right = parse!("(x+1)^15*(x+3)^7").to_polynomial::<_, u16>(&Z, Some(variables));
        let leading = Integer::from(11).to_finite_field(&field);
        let actual = DenseZp64UnivariateGcdImage::new(&left, &right, 1, &field)
            .unwrap()
            .run(leading);
        let expected = parse!("(x+1)^15")
            .to_polynomial::<_, u16>(&Z, Some(actual.variables().clone()))
            .map_coeff(
                |coefficient| coefficient.to_finite_field(&field),
                field.clone(),
            )
            .mul_coeff(leading);
        assert_eq!(actual, expected);
        assert_eq!(actual.lcoeff(), leading);

        let dropped_leading = parse!("x")
            .to_polynomial::<IntegerRing, u16>(&Z, Some(actual.variables().clone()))
            .mul_coeff(Integer::from(field.get_prime()))
            .add_constant(Integer::one());
        assert!(DenseZp64UnivariateGcdImage::new(&dropped_leading, &left, 1, &field).is_none());

        let sparse_left = parse!("x^100000+1").to_polynomial::<IntegerRing, u32>(&Z, None);
        let sparse_right = parse!("x^99999+2")
            .to_polynomial::<IntegerRing, u32>(&Z, sparse_left.variables().clone());
        assert!(DenseZp64UnivariateGcdImage::new(&sparse_left, &sparse_right, 0, &field).is_none());
    }

    #[test]
    fn modular_univariate_integer_gcd_resets_an_unlucky_first_degree() {
        let first_prime = ModularGcdPrimeIterator::for_workspace::<u64>()
            .next()
            .unwrap();
        let common_factor = parse!("x")
            .to_polynomial::<_, u16>(&Z, None)
            .mul_coeff(Integer::one() << 80usize)
            .add_constant(Integer::from(-1));
        let left_cofactor =
            parse!("5*x+1").to_polynomial::<_, u16>(&Z, common_factor.variables().clone());
        let right_cofactor = left_cofactor
            .clone()
            .add_constant(Integer::from(first_prime));
        let left = &common_factor * &left_cofactor;
        let right = &common_factor * &right_cofactor;

        let context = UnivariateModularGcdContext::new(&left, &right, 0);
        assert!(matches!(
            &context.normalization,
            UnivariateGcdProjectiveNormalization::Constant { .. }
        ));
        let (actual, left_result, right_result) = context.run().unwrap();
        assert_eq!(actual, common_factor);
        assert_eq!(&actual * &left_result, left);
        assert_eq!(&actual * &right_result, right);

        let coprime_right = left_cofactor.clone().add_constant(Integer::from(1));
        let (unit, unit_left_result, unit_right_result) =
            UnivariateModularGcdContext::new(&left_cofactor, &coprime_right, 0)
                .run()
                .unwrap();
        assert_eq!(unit, left_cofactor.one());
        assert_eq!(&unit * &unit_left_result, left_cofactor);
        assert_eq!(&unit * &unit_right_result, coprime_right);
    }

    #[test]
    fn modular_univariate_integer_gcd_uses_leading_coordinate_for_zero_constants() {
        let common_factor = parse!("x").to_polynomial::<_, u16>(&Z, None);
        let left_cofactor =
            parse!("6*x+1").to_polynomial::<_, u16>(&Z, common_factor.variables().clone());
        let right_cofactor =
            parse!("10*x+1").to_polynomial::<_, u16>(&Z, common_factor.variables().clone());
        let left = &common_factor * &left_cofactor;
        let right = &common_factor * &right_cofactor;

        let context = UnivariateModularGcdContext::new(&left, &right, 0);
        assert!(matches!(
            &context.normalization,
            UnivariateGcdProjectiveNormalization::Leading(_)
        ));
        let (actual, left_result, right_result) = context.run().unwrap();
        assert_eq!(actual, common_factor);
        assert_eq!(&actual * &left_result, left);
        assert_eq!(&actual * &right_result, right);
    }

    #[test]
    fn modular_univariate_integer_gcd_retains_leading_coordinate_fallback() {
        let leading = Integer::one() << 384usize;
        let constant = Integer::one() << 192usize;
        let common_factor = parse!("x")
            .to_polynomial::<_, u16>(&Z, None)
            .mul_coeff(leading)
            .add_constant(Integer::one());
        let left_cofactor = parse!("x^2+x")
            .to_polynomial::<_, u16>(&Z, common_factor.variables().clone())
            .add_constant(constant.clone());
        let right_cofactor = parse!("x^2+2*x")
            .to_polynomial::<_, u16>(&Z, common_factor.variables().clone())
            .add_constant(constant.clone());
        let left = &common_factor * &left_cofactor;
        let right = &common_factor * &right_cofactor;

        let context = UnivariateModularGcdContext::new(&left, &right, 0);
        assert!(matches!(
            &context.normalization,
            UnivariateGcdProjectiveNormalization::Constant { .. }
        ));

        let mut modulus = Integer::one();
        for prime in univariate_modular_gcd_prime_iterator().take(7) {
            modulus *= prime;
        }
        let constant_reconstruction = common_factor
            .clone()
            .mul_coeff(constant)
            .map_coeff(|coefficient| coefficient.clone().symmetric_mod(&modulus), Z);
        assert!(
            context
                .certified_reconstruction(&constant_reconstruction, 1u16)
                .is_none()
        );
        let leading_reconstruction =
            context.leading_reconstruction(&constant_reconstruction, &modulus);
        assert_eq!(leading_reconstruction, common_factor);
        assert_eq!(
            context
                .certified_reconstruction(&leading_reconstruction, 1u16)
                .unwrap()
                .0,
            common_factor
        );

        let (actual, left_result, right_result) =
            UnivariateModularGcdContext::new(&left, &right, 0)
                .run()
                .unwrap();
        assert_eq!(actual, common_factor);
        assert_eq!(&actual * &left_result, left);
        assert_eq!(&actual * &right_result, right);
    }

    #[test]
    fn modular_univariate_integer_gcd_restores_content_with_an_inactive_variable() {
        let variable_template = parse!("x").to_polynomial::<IntegerRing, u16>(&Z, None);
        let variables = std::sync::Arc::new(vec![
            PolyVariable::Temporary(0),
            variable_template.variables()[0].clone(),
        ]);
        let [left_cofactor, right_cofactor, common_factor] = [
            parse!("(1+3*x)^24-1").to_polynomial::<_, u16>(&Z, Some(variables.clone())),
            parse!("(1-3*x)^24+1").to_polynomial::<_, u16>(&Z, Some(variables.clone())),
            parse!("(1-3*x)^24+3").to_polynomial::<_, u16>(&Z, Some(variables)),
        ];
        let left = (&left_cofactor * &common_factor).mul_coeff(Integer::from(6));
        let right = (&right_cofactor * &common_factor).mul_coeff(Integer::from(10));
        assert_eq!(left.degree(0), 0);
        assert_ne!(left.degree(1), 0);

        let context = UnivariateModularGcdContext::new(&left, &right, 1);
        assert!(matches!(
            &context.normalization,
            UnivariateGcdProjectiveNormalization::Constant { .. }
        ));
        let (actual, left_result, right_result) = context.run().unwrap();
        assert_eq!(actual, common_factor.mul_coeff(Integer::from(2)));
        assert_eq!(&actual * &left_result, left);
        assert_eq!(&actual * &right_result, right);
    }

    #[test]
    fn dense_univariate_integer_division_certificate_matches_generic_division() {
        let variable_template = parse!("x").to_polynomial::<IntegerRing, u16>(&Z, None);
        let variables = std::sync::Arc::new(vec![
            PolyVariable::Temporary(0),
            variable_template.variables()[0].clone(),
        ]);
        let scale = Integer::from(1) << 200usize;
        let divisor = parse!("-2*x^5+3*x^2-7")
            .to_polynomial::<_, u16>(&Z, Some(variables.clone()))
            .mul_coeff(scale);
        let quotient =
            parse!("-5*x^7+11*x^3-13").to_polynomial::<_, u16>(&Z, Some(variables.clone()));
        let dividend = &divisor * &quotient;
        let mut division =
            DenseUnivariateIntegerDivisionContext::new(&divisor, &dividend, &dividend, 1).unwrap();
        let actual = division.try_div(&dividend).unwrap();
        actual.check_consistency();
        assert_eq!(actual, quotient);
        assert_eq!(Some(actual), dividend.try_div(&divisor));

        let leading_inexact_divisor =
            parse!("2*x+1").to_polynomial::<_, u16>(&Z, Some(variables.clone()));
        let leading_inexact_dividend =
            parse!("x^2").to_polynomial::<_, u16>(&Z, Some(variables.clone()));
        let mut leading_inexact = DenseUnivariateIntegerDivisionContext::new(
            &leading_inexact_divisor,
            &leading_inexact_dividend,
            &leading_inexact_dividend,
            1,
        )
        .unwrap();
        assert!(leading_inexact.try_div(&leading_inexact_dividend).is_none());

        let final_remainder_divisor =
            parse!("x+1").to_polynomial::<_, u16>(&Z, Some(variables.clone()));
        let final_remainder_dividend =
            parse!("x^2+1").to_polynomial::<_, u16>(&Z, Some(variables.clone()));
        let mut final_remainder = DenseUnivariateIntegerDivisionContext::new(
            &final_remainder_divisor,
            &final_remainder_dividend,
            &final_remainder_dividend,
            1,
        )
        .unwrap();
        assert!(final_remainder.try_div(&final_remainder_dividend).is_none());

        let sparse_divisor =
            parse!("x^1000+1").to_polynomial::<_, u16>(&Z, Some(variables.clone()));
        let sparse_dividend = &sparse_divisor * &quotient;
        assert!(
            DenseUnivariateIntegerDivisionContext::new(
                &sparse_divisor,
                &sparse_dividend,
                &sparse_dividend,
                1,
            )
            .is_none()
        );

        let off_variable = parse!("y+x+1").to_polynomial::<_, u16>(&Z, None);
        assert!(
            DenseUnivariateIntegerDivisionContext::new(
                &off_variable,
                &off_variable,
                &off_variable,
                0,
            )
            .is_none()
        );
    }

    #[test]
    fn hu_monagan_prime_bound_combines_interpolation_and_coefficient_bounds() {
        assert_eq!(
            hu_monagan_prime_lower_bound(100, 1, &Integer::from(500)),
            500
        );
        assert_eq!(
            hu_monagan_prime_lower_bound(1_000, 1, &Integer::from(500)),
            2_000
        );

        // This height still needs no larger prime than the interpolation
        // lattice because eight small images already cover its bits.
        assert_eq!(
            hu_monagan_prime_lower_bound(1_000, 1, &Integer::from(1u64 << 32)),
            2_000
        );
        assert_eq!(
            hu_monagan_prime_lower_bound(1_000, 1, &(Integer::one() << 255usize)),
            1u64 << 32
        );
        assert_eq!(
            hu_monagan_prime_lower_bound(1_000, 1, &(Integer::one() << 1023usize)),
            1u64 << 63
        );
        assert_eq!(
            hu_monagan_prime_lower_bound(u64::MAX, 1, &Integer::from(1)),
            u64::MAX
        );
    }

    #[test]
    fn hu_geometric_term_setup_falls_back_beyond_the_power_cache() {
        let field = Zp64::new(1_088_391_169);
        let integer_polynomial =
            parse!("3*x+5*x*y^1001+7*x^2*y^17").to_polynomial::<_, u16>(&Z, None);
        assert_eq!(integer_polynomial.degree(1), 1001);
        let polynomial = integer_polynomial.map_coeff(
            |coefficient| coefficient.to_finite_field(&field),
            field.clone(),
        );
        let term_base = field.to_element(7);
        let shifted_base = field.pow(&term_base, 19);

        // A one-element table contains only x^0 and forces every positive exponent through the
        // same direct-power fallback used beyond POW_CACHE_SIZE.
        let (_, ratios, current) = MultivariatePolynomial::<IntegerRing, u16>::evaluate_terms(
            &field,
            &polynomial,
            &[term_base],
            &[shifted_base],
            &[vec![field.one()]],
            &[vec![field.one()]],
        );

        for (((exponents, coefficient), ratio), current) in polynomial
            .exponents
            .chunks(polynomial.nvars())
            .zip(&polynomial.coefficients)
            .zip(&ratios)
            .zip(&current)
        {
            let exponent = exponents[1].to_u32() as u64;
            assert_eq!(*ratio, field.pow(&term_base, exponent));
            assert_eq!(
                *current,
                field.mul(coefficient, &field.pow(&shifted_base, exponent))
            );
        }
    }

    #[test]
    fn hu_monagan_kronecker_map_decodes_u64_exponents() {
        let map = HuMonaganKroneckerMap::new(&[0, 24, 24, 24, 24, 24, 24, 24], 1).unwrap();
        assert_eq!(map.powers().last(), Some(&4_586_471_424));
        assert_eq!(map.range(), 4_586_471_424);
        assert!(map.range() > u32::MAX as u64);

        let expected = [0u16, 23, 7, 0, 19, 3, 11, 5];
        let encoded = expected
            .iter()
            .skip(1)
            .zip(std::iter::once(1u64).chain(map.powers().iter().copied()))
            .map(|(exponent, power)| *exponent as u64 * power)
            .sum();
        let mut decoded = [0u16; 8];
        map.decode(encoded, &mut decoded).unwrap();
        assert_eq!(decoded, expected);
        assert!(map.decode(map.range(), &mut decoded).is_none());
    }

    #[test]
    fn hu_monagan_kronecker_map_rejects_invalid_ranges() {
        assert!(HuMonaganKroneckerMap::new(&[0, u32::MAX, u32::MAX, 2], 1).is_none());
        assert!(HuMonaganKroneckerMap::new(&[0, 2, 0, 3], 1).is_none());
        assert!(HuMonaganKroneckerMap::new(&[2, 3], 3).is_none());
    }

    #[test]
    fn hu_monagan_large_kronecker_exponents() {
        let map = HuMonaganKroneckerMap::new(&[0, 261, 261, 261, 261, 261, 261, 261], 1).unwrap();
        assert_eq!(map.range(), 82_505_623_639_781_421);

        let mut polynomials = [
            parse!("1+x1+2*x2+3*x3+4*x4+5*x5+6*x6+7*x7+8*x8").to_polynomial::<_, u16>(&Z, None),
            parse!("1-x1+2*x2-3*x3+4*x4-5*x5+6*x6-7*x7+8*x8").to_polynomial::<_, u16>(&Z, None),
            parse!("1+x1^256+2*x2^256+3*x3^256+5*x4^256+7*x5^256+11*x6^256+13*x7^256+17*x8^256")
                .to_polynomial::<_, u16>(&Z, None),
        ];
        MultivariatePolynomial::unify_variables_list(&mut polynomials);
        let [a, b, gcd] = polynomials;
        let ag = &a * &gcd;
        let bg = &b * &gcd;
        let bounds = (0..gcd.nvars())
            .map(|variable| gcd.degree(variable))
            .collect::<Vec<_>>();

        assert_eq!(ag.gcd_hu_monagan(&bg, &bounds), Some(gcd));
    }

    #[test]
    fn projective_integer_gcd_reconstruction_with_large_coefficients() {
        let cofactors = [
            parse!("(1+x+y+z)^3-1").to_polynomial::<_, u16>(&Z, None),
            parse!("(2+x-y+2*z)^3+1").to_polynomial::<_, u16>(&Z, None),
        ];
        let gcds = [
            // A small constant pivot lets projective reconstruction stop after roughly the
            // coefficient height instead of twice that height.
            parse!(
                "3+100000000000000000000000000000000000000000000000000000000007*x^2\
                 -100000000000000000000000000000000000000000000000000000000009*x*y\
                 +100000000000000000000000000000000000000000000000000000000033*y^2+5*z^2"
            )
            .to_polynomial::<_, u16>(&Z, None),
            // Exercise the minimum-total-degree fallback when no constant term is present.
            parse!(
                "100000000000000000000000000000000000000000000000000000000007*x^2\
                 -100000000000000000000000000000000000000000000000000000000009*x*y\
                 +100000000000000000000000000000000000000000000000000000000033*y^2+5*z^2"
            )
            .to_polynomial::<_, u16>(&Z, None),
            // The selected probe reconstructs 3/6 = 1/2, while another coefficient needs the
            // denominator 3. The common-denominator lift must fail safely before full MQR returns
            // the primitive polynomial.
            parse!("2*x^2+3*y+6").to_polynomial::<_, u16>(&Z, None),
        ];

        for gcd in gcds {
            let mut polynomials = [cofactors[0].clone(), cofactors[1].clone(), gcd];
            MultivariatePolynomial::unify_variables_list(&mut polynomials);
            let [a, b, gcd] = polynomials;
            let ag = &a * &gcd;
            let bg = &b * &gcd;
            let vars = (0..gcd.nvars()).collect::<Vec<_>>();
            let mut bounds = (0..gcd.nvars())
                .map(|variable| gcd.degree(variable))
                .collect::<Vec<_>>();
            let mut tight_bounds = bounds.clone();
            let gamma = Z.gcd(&ag.lcoeff_varorder(&vars), &bg.lcoeff_varorder(&vars));

            let reconstructed = MultivariatePolynomial::gcd_zippel::<u32>(
                &ag,
                &bg,
                &vars,
                &mut bounds,
                &mut tight_bounds,
                &gamma,
            );
            assert_eq!(reconstructed, gcd);
        }
    }
}
