//! Multivariate polynomial structures and methods.

use ahash::{HashMap, HashMapExt};
use std::cell::{Cell, RefCell, UnsafeCell};
use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BinaryHeap};
use std::fmt::Display;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Add, Div, Mul, Neg, RangeInclusive, Sub};
use std::sync::Arc;

use rand::Rng;

use crate::domains::algebraic::AlgebraicExtension;
use crate::domains::finite_field::{
    FiniteField, FiniteFieldCore, FiniteFieldElement, FiniteFieldWorkspace,
};
use crate::domains::float::FloatLike;
use crate::domains::integer::{Integer, IntegerRing, Z};
use crate::domains::rational::{Fraction, FractionField, FractionNormalization, Q, RationalField};
use crate::domains::{
    Derivable, EuclideanDomain, Field, InternalOrdering, RealEmbedding, Ring, RingOps,
    SampleableRing, SelfRing, Set,
};
use crate::kernels::{
    DensePolynomialExactDivisionRequest, DensePolynomialMulRequest, TotalDegreePolynomialMulRequest,
};
use crate::printer::{AtomPrinter, PrintOptions, PrintState};

use super::gcd::PolynomialGCD;
use super::univariate::UnivariatePolynomial;
use super::{Exponent, INLINED_EXPONENTS, LexOrder, MonomialOrder, PolyVariable, PositiveExponent};
use smallvec::{SmallVec, smallvec};

const MAX_DENSE_MUL_BUFFER_SIZE: usize = 1 << 24;
const MAX_DENSE_DIV_BUFFER_SIZE: usize = 1 << 20;
const MAX_MIXED_RADIX_DENSE_TO_PAIR_PRODUCT_RATIO: usize = 64;
const MAX_PACKED_ROW_MERGE_TERMS: usize = 64;
const MAX_PACKED_ROW_MERGE_PAIR_PRODUCTS: usize = 4096;
thread_local! { static DENSE_MUL_BUFFER: Cell<Vec<u32>> = const { Cell::new(Vec::new()) }; }

/// Return whether scanning a full mixed-radix coefficient box is bounded relative to the
/// coefficient products. Multivariate multiplication uses this before allocating its dense
/// workspace so that highly lacunary boxes continue with heap multiplication.
#[inline]
fn mixed_radix_dense_work_is_bounded(
    output_len: usize,
    left_terms: usize,
    right_terms: usize,
) -> bool {
    let pair_products = left_terms.saturating_mul(right_terms);
    output_len <= pair_products.saturating_mul(MAX_MIXED_RADIX_DENSE_TO_PAIR_PRODUCT_RATIO)
}

/// Return whether a multivariate dense multiplication can use a bounded coefficient box.
/// Sparse product checks use the same condition to avoid replacing this path with hashing.
pub(super) fn mixed_radix_dense_mul_is_bounded(
    output_len: usize,
    left_terms: usize,
    right_terms: usize,
) -> bool {
    output_len <= MAX_DENSE_MUL_BUFFER_SIZE
        && mixed_radix_dense_work_is_bounded(output_len, left_terms, right_terms)
}

/// Return whether a packed sparse product is small enough to enumerate with a bounded row heap.
#[inline]
fn packed_row_merge_is_bounded(left_terms: usize, right_terms: usize) -> bool {
    left_terms.min(right_terms) <= MAX_PACKED_ROW_MERGE_TERMS
        && left_terms
            .checked_mul(right_terms)
            .is_some_and(|products| products <= MAX_PACKED_ROW_MERGE_PAIR_PRODUCTS)
}

struct TotalDegreeRankTable {
    variable_count: usize,
    total_degree: usize,
    binomial_size: usize,
    binomial: Vec<usize>,
    prefix_length: usize,
    suffix_length: usize,
    prefix_code_count: usize,
    suffix_code_count: usize,
    prefix_rank: Vec<u32>,
    prefix_remaining: Vec<u8>,
    suffix_rank: Vec<u32>,
}

thread_local! {
    static TOTAL_DEGREE_RANK_TABLE: RefCell<Option<Arc<TotalDegreeRankTable>>> = const {
        RefCell::new(None)
    };
}

impl TotalDegreeRankTable {
    #[inline(always)]
    fn choose(&self, n: usize, k: usize) -> usize {
        self.binomial[n * self.binomial_size + k]
    }

    fn build(variable_count: usize, total_degree: usize) -> Option<Self> {
        #[inline]
        fn decode_code(mut code: usize, length: usize, radix: usize, digits: &mut [usize]) {
            for digit in digits[..length].iter_mut().rev() {
                *digit = code % radix;
                code /= radix;
            }
        }

        let radix = total_degree + 1;
        let prefix_length = variable_count / 2;
        let suffix_length = variable_count - prefix_length;
        let prefix_code_count = radix.checked_pow(prefix_length as u32)?;
        let suffix_code_count = radix.checked_pow(suffix_length as u32)?;
        let suffix_table_size = suffix_code_count.checked_mul(radix)?;
        if suffix_table_size > 1 << 22 {
            return None;
        }

        let binomial_size = total_degree.checked_add(variable_count)?.checked_add(2)?;
        let mut binomial = vec![0usize; binomial_size.checked_mul(binomial_size)?];
        for n in 0..binomial_size {
            binomial[n * binomial_size] = 1;
            binomial[n * binomial_size + n] = 1;
            for k in 1..n {
                binomial[n * binomial_size + k] = binomial[(n - 1) * binomial_size + k - 1]
                    .checked_add(binomial[(n - 1) * binomial_size + k])?;
            }
        }
        let choose = |n: usize, k: usize| binomial[n * binomial_size + k];

        let mut prefix_rank = vec![u32::MAX; prefix_code_count];
        let mut prefix_remaining = vec![u8::MAX; prefix_code_count];
        let mut digits = vec![0usize; variable_count];
        for code in 0..prefix_code_count {
            decode_code(code, prefix_length, radix, &mut digits);
            let degree = digits[..prefix_length].iter().sum::<usize>();
            if degree > total_degree {
                continue;
            }

            let mut rank = 0usize;
            let mut used_degree = 0usize;
            for (index, &exponent) in digits[..prefix_length].iter().enumerate() {
                let remaining_variables = variable_count - index - 1;
                let available_degree = total_degree - used_degree;
                rank += choose(
                    remaining_variables + available_degree + 1,
                    remaining_variables + 1,
                ) - choose(
                    remaining_variables + available_degree - exponent + 1,
                    remaining_variables + 1,
                );
                used_degree += exponent;
            }
            prefix_rank[code] = u32::try_from(rank).ok()?;
            prefix_remaining[code] = u8::try_from(total_degree - degree).ok()?;
        }

        let mut suffix_rank = vec![u32::MAX; suffix_table_size];
        for available_degree in 0..=total_degree {
            for code in 0..suffix_code_count {
                decode_code(code, suffix_length, radix, &mut digits);
                if digits[..suffix_length].iter().sum::<usize>() > available_degree {
                    continue;
                }

                let mut rank = 0usize;
                let mut used_degree = 0usize;
                for (index, &exponent) in digits[..suffix_length].iter().enumerate() {
                    let remaining_variables = suffix_length - index - 1;
                    let remaining_degree = available_degree - used_degree;
                    rank += choose(
                        remaining_variables + remaining_degree + 1,
                        remaining_variables + 1,
                    ) - choose(
                        remaining_variables + remaining_degree - exponent + 1,
                        remaining_variables + 1,
                    );
                    used_degree += exponent;
                }
                suffix_rank[available_degree * suffix_code_count + code] =
                    u32::try_from(rank).ok()?;
            }
        }

        Some(Self {
            variable_count,
            total_degree,
            binomial_size,
            binomial,
            prefix_length,
            suffix_length,
            prefix_code_count,
            suffix_code_count,
            prefix_rank,
            prefix_remaining,
            suffix_rank,
        })
    }
}

fn total_degree_rank_table(
    variable_count: usize,
    total_degree: usize,
) -> Option<Arc<TotalDegreeRankTable>> {
    TOTAL_DEGREE_RANK_TABLE.with(|cache| {
        if let Some(table) = cache.borrow().as_ref()
            && table.variable_count == variable_count
            && table.total_degree == total_degree
        {
            return Some(table.clone());
        }

        let table = Arc::new(TotalDegreeRankTable::build(variable_count, total_degree)?);
        *cache.borrow_mut() = Some(table.clone());
        Some(table)
    })
}

/// A ring for multivariate polynomials.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct PolynomialRing<R: Ring, E: Exponent = u16> {
    pub(crate) ring: R,
    _phantom_exp: PhantomData<E>,
}

/// Sampling policy for a multivariate polynomial ring.
///
/// Each requested term gets an independently sampled coefficient and one
/// exponent per variable. Zero coefficients and duplicate monomials can make
/// the resulting polynomial contain fewer terms than the sampled term count.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PolynomialSamplingPolicy<P> {
    /// Variables of the sampled polynomial, in exponent-vector order.
    pub variables: Arc<Vec<PolyVariable>>,
    /// Inclusive exponent bounds for each variable.
    pub degree_bounds: Vec<RangeInclusive<u32>>,
    /// Inclusive range from which the number of attempted terms is sampled.
    pub term_count: RangeInclusive<usize>,
    /// Policy used to sample every coefficient.
    pub coefficient: P,
}

impl<R: Ring + FractionNormalization, E: Exponent> FractionNormalization for PolynomialRing<R, E> {
    fn get_normalization_factor(&self, a: &Self::Element) -> Self::Element {
        a.constant(a.ring().get_normalization_factor(&a.lcoeff()))
    }
}

impl<R: EuclideanDomain + FractionNormalization, E: Exponent> PolynomialRing<FractionField<R>, E> {
    pub fn to_rational_polynomial(
        &self,
        e: &<Self as Set>::Element,
    ) -> Fraction<PolynomialRing<R, E>> {
        let ring = self.ring.ring();
        let mut lcm = ring.one();
        for x in &e.coefficients {
            let g = ring.gcd(&lcm, x.denominator_ref());
            lcm = ring.mul(&lcm, &ring.quot_rem(x.denominator_ref(), &g).0);
        }

        let e2 = e.map_coeff(
            |c| {
                ring.mul(
                    c.numerator_ref(),
                    &ring.quot_rem(&lcm, c.denominator_ref()).0,
                )
            },
            ring.clone(),
        );

        Fraction::from_unchecked(e2.constant(lcm), e2)
    }
}

impl<R: Ring, E: Exponent> PolynomialRing<R, E> {
    pub fn new(coeff_ring: R) -> PolynomialRing<R, E> {
        PolynomialRing {
            ring: coeff_ring,
            _phantom_exp: PhantomData,
        }
    }

    pub fn from_poly(poly: &MultivariatePolynomial<R, E>) -> PolynomialRing<R, E> {
        PolynomialRing {
            ring: poly.ring().clone(),
            _phantom_exp: PhantomData,
        }
    }

    /// Get the coefficient ring.
    pub fn coefficient_ring(&self) -> &R {
        &self.ring
    }
}

impl<R: Ring, E: Exponent> std::fmt::Display for PolynomialRing<R, E> {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl<R: Ring, E: Exponent> Set for PolynomialRing<R, E> {
    type Element = MultivariatePolynomial<R, E>;

    fn size(&self) -> Option<Integer> {
        None
    }
}

impl<R: Ring, E: Exponent> RingOps<MultivariatePolynomial<R, E>> for PolynomialRing<R, E> {
    #[inline]
    fn add(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        a + b
    }

    #[inline]
    fn sub(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        a - b
    }

    #[inline]
    fn mul(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        a * &b
    }

    #[inline]
    fn add_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a = std::mem::replace(a, b.zero()) + b;
    }

    #[inline]
    fn sub_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a = std::mem::replace(a, b.zero()) - b;
    }

    #[inline]
    fn mul_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a = &*a * &b;
    }

    #[inline]
    fn add_mul_assign(&self, a: &mut Self::Element, b: Self::Element, c: Self::Element) {
        *a = std::mem::replace(a, b.zero()) + b * &c
    }

    #[inline]
    fn sub_mul_assign(&self, a: &mut Self::Element, b: Self::Element, c: Self::Element) {
        *a = std::mem::replace(a, b.zero()) - b * &c
    }

    #[inline]
    fn neg(&self, a: Self::Element) -> Self::Element {
        a.neg()
    }
}

impl<R: Ring, E: Exponent> RingOps<&MultivariatePolynomial<R, E>> for PolynomialRing<R, E> {
    #[inline]
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a + b
    }

    #[inline]
    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a - b
    }

    #[inline]
    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a * b
    }

    #[inline]
    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = &*a + b;
    }

    #[inline]
    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = &*a - b;
    }

    #[inline]
    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = std::mem::replace(a, b.zero()) * b;
    }

    #[inline]
    fn add_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        *a = std::mem::replace(a, b.zero()) + b * c
    }

    #[inline]
    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        *a = std::mem::replace(a, b.zero()) - b * c
    }

    #[inline]
    fn neg(&self, a: &Self::Element) -> Self::Element {
        a.clone().neg()
    }
}

impl<R: Ring, E: Exponent> Ring for PolynomialRing<R, E> {
    #[inline]
    fn zero(&self) -> Self::Element {
        MultivariatePolynomial::new(&self.ring, None, Arc::new(vec![]))
    }

    #[inline]
    fn one(&self) -> Self::Element {
        self.zero().one()
    }

    #[inline]
    fn nth(&self, n: Integer) -> Self::Element {
        self.zero().constant(self.ring.nth(n))
    }

    #[inline]
    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element {
        b.pow(e as usize)
    }

    #[inline]
    fn is_zero(&self, a: &Self::Element) -> bool {
        a.is_zero()
    }

    #[inline]
    fn is_one(&self, a: &Self::Element) -> bool {
        a.is_one()
    }

    fn one_is_gcd_unit() -> bool {
        false
    }

    fn characteristic(&self) -> Integer {
        self.ring.characteristic()
    }

    fn try_inv(&self, a: &Self::Element) -> Option<Self::Element> {
        if a.is_constant() {
            let inv = self.ring.try_inv(&a.get_constant())?;
            Some(a.constant(inv))
        } else {
            None
        }
    }

    fn try_div(&self, a: &Self::Element, b: &Self::Element) -> Option<Self::Element> {
        a.try_div(b)
    }

    #[inline]
    fn try_div_owned(&self, a: Self::Element, b: &Self::Element) -> Option<Self::Element> {
        a.try_div_owned(b)
    }

    #[inline]
    fn exact_div_owned(&self, a: Self::Element, b: &Self::Element) -> Self::Element {
        a.exact_div_owned(b)
    }

    fn format<W: std::fmt::Write>(
        &self,
        element: &Self::Element,
        opts: &PrintOptions,
        state: PrintState,
        f: &mut W,
    ) -> Result<bool, std::fmt::Error> {
        element.format(opts, state, f)
    }

    fn has_independent_elements(&self) -> bool {
        // the coefficient ring is stored in the polynomial
        true
    }
}

impl<R: SampleableRing, E: Exponent> SampleableRing for PolynomialRing<R, E> {
    type SamplingPolicy = PolynomialSamplingPolicy<R::SamplingPolicy>;

    fn sample<G: rand::RngCore + ?Sized>(
        &self,
        rng: &mut G,
        policy: &Self::SamplingPolicy,
    ) -> Self::Element {
        assert_eq!(
            policy.variables.len(),
            policy.degree_bounds.len(),
            "a degree bound is required for every polynomial variable"
        );

        let term_count = rng.random_range(policy.term_count.clone());
        let mut polynomial =
            MultivariatePolynomial::new(&self.ring, Some(term_count), policy.variables.clone());
        for _ in 0..term_count {
            let coefficient = self.ring.sample(rng, &policy.coefficient);
            let exponents = policy
                .degree_bounds
                .iter()
                .map(|range| {
                    E::from_i32(
                        rng.random_range(range.clone())
                            .try_into()
                            .expect("polynomial degree exceeds i32::MAX"),
                    )
                })
                .collect::<Vec<_>>();
            polynomial.append_monomial(coefficient, &exponents);
        }

        polynomial
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent> EuclideanDomain
    for PolynomialRing<R, E>
{
    fn rem(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.rem(b)
    }

    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        a.quot_rem(b, false)
    }

    #[inline]
    fn quot_rem_owned(
        &self,
        a: Self::Element,
        b: &Self::Element,
    ) -> (Self::Element, Self::Element) {
        a.quot_rem_owned(b, false)
    }

    fn gcd(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.gcd(b)
    }
}

/// Multivariate polynomial with a sparse degree and dense variable representation.
/// Negative exponents are supported, if they are allowed by the exponent type.
#[derive(Clone)]
pub struct MultivariatePolynomial<F: Ring, E: Exponent = u16, O: MonomialOrder = LexOrder> {
    // Data format: the i-th monomial is stored as coefficients[i] and
    // exponents[i * nvars .. (i + 1) * nvars]. Terms are always expanded and sorted by the exponents via
    // cmp_exponents().
    pub coefficients: Vec<F::Element>,
    pub exponents: Vec<E>,
    context: Arc<PolynomialContext<F>>,
    pub(crate) _phantom: PhantomData<O>,
}

/// An error encountered while counting the positive real roots of a
/// polynomial.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PositiveRealRootCountError<E> {
    /// The polynomial has more than one variable.
    NotUnivariate { variables: usize },
    /// The zero polynomial has infinitely many roots.
    ZeroPolynomial,
    /// A coefficient could not be compared through its real embedding.
    Comparison(E),
    /// The sign variations obtained from the Sturm sequence were inconsistent.
    InvalidSturmSequence,
}

impl<E: Display> Display for PositiveRealRootCountError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotUnivariate { variables } => {
                write!(
                    f,
                    "expected a univariate polynomial, got {variables} variables"
                )
            }
            Self::ZeroPolynomial => f.write_str("cannot count the roots of the zero polynomial"),
            Self::Comparison(error) => write!(f, "could not determine a coefficient sign: {error}"),
            Self::InvalidSturmSequence => {
                f.write_str("invalid Sturm sequence while counting positive real roots")
            }
        }
    }
}

impl<E: std::fmt::Debug + Display> std::error::Error for PositiveRealRootCountError<E> {}

/// Shared coefficient ring and variable map of a multivariate polynomial.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct PolynomialContext<F: Ring> {
    ring: F,
    variables: Arc<Vec<PolyVariable>>,
}

#[cfg(feature = "bincode")]
impl<F: Ring + bincode::Encode, E: Exponent + bincode::Encode, O: MonomialOrder> bincode::Encode
    for MultivariatePolynomial<F, E, O>
where
    F::Element: bincode::Encode,
{
    fn encode<EN: bincode::enc::Encoder>(
        &self,
        encoder: &mut EN,
    ) -> Result<(), bincode::error::EncodeError> {
        bincode::Encode::encode(self.ring(), encoder)?;
        bincode::Encode::encode(&self.coefficients, encoder)?;
        bincode::Encode::encode(&self.exponents, encoder)?;
        bincode::Encode::encode(self.variables(), encoder)
    }
}

#[cfg(feature = "bincode")]
impl<
    C: crate::state::HasStateMap,
    F: Ring + bincode::Decode<C>,
    E: Exponent + bincode::Decode<C>,
    O: MonomialOrder,
> bincode::Decode<C> for MultivariatePolynomial<F, E, O>
where
    F::Element: for<'a> bincode::Decode<&'a F>,
{
    fn decode<D: bincode::de::Decoder<Context = C>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let ring = F::decode(decoder)?;

        let coefficients = Vec::<F::Element>::decode(&mut decoder.with_context(&ring))?;
        let exponents = Vec::<E>::decode(decoder)?;
        let variables = Arc::<Vec<PolyVariable>>::decode(decoder)?;
        Ok(MultivariatePolynomial {
            coefficients,
            exponents,
            context: Arc::new(PolynomialContext { ring, variables }),
            _phantom: PhantomData,
        })
    }
}

impl<F: Ring, E: Exponent, O: MonomialOrder> MultivariatePolynomial<F, E, O> {
    /// Returns the coefficient ring.
    #[inline]
    pub fn ring(&self) -> &F {
        &self.context.ring
    }

    /// Returns the shared variable map.
    #[inline]
    pub fn variables(&self) -> &Arc<Vec<PolyVariable>> {
        &self.context.variables
    }

    /// Returns the coefficient ring and mutable coefficient buffer as disjoint borrows.
    #[inline]
    pub(crate) fn ring_and_coefficients_mut(&mut self) -> (&F, &mut Vec<F::Element>) {
        (&self.context.ring, &mut self.coefficients)
    }

    /// Replaces the variable map without changing the coefficient ring.
    #[inline]
    pub(crate) fn set_variables(&mut self, variables: Arc<Vec<PolyVariable>>) {
        if Arc::ptr_eq(&self.context.variables, &variables) {
            return;
        }
        Arc::make_mut(&mut self.context).variables = variables;
    }

    /// Constructs a zero polynomial. Instead of using this constructor,
    /// prefer to create new polynomials from existing ones, so that the
    /// variable map and field are inherited.
    #[inline]
    pub fn new(ring: &F, cap: Option<usize>, variables: Arc<Vec<PolyVariable>>) -> Self {
        Self {
            coefficients: Vec::with_capacity(cap.unwrap_or(0)),
            exponents: Vec::with_capacity(cap.unwrap_or(0) * variables.len()),
            context: Arc::new(PolynomialContext {
                ring: ring.clone(),
                variables,
            }),
            _phantom: PhantomData,
        }
    }

    /// Constructs an empty polynomial with capacity, sharing `context`.
    #[inline]
    fn from_context(cap: Option<usize>, context: Arc<PolynomialContext<F>>) -> Self {
        let nvars = context.variables.len();
        Self {
            coefficients: Vec::with_capacity(cap.unwrap_or(0)),
            exponents: Vec::with_capacity(cap.unwrap_or(0) * nvars),
            context,
            _phantom: PhantomData,
        }
    }

    /// Constructs a polynomial from its raw term buffers and metadata.
    #[inline]
    pub(crate) fn from_parts(
        coefficients: Vec<F::Element>,
        exponents: Vec<E>,
        ring: F,
        variables: Arc<Vec<PolyVariable>>,
    ) -> Self {
        debug_assert_eq!(exponents.len(), coefficients.len() * variables.len());
        Self {
            coefficients,
            exponents,
            context: Arc::new(PolynomialContext { ring, variables }),
            _phantom: PhantomData,
        }
    }

    /// Constructs a zero polynomial. Instead of using this constructor,
    /// prefer to create new polynomials from existing ones, so that the
    /// variable map is inherited.
    #[inline]
    pub fn new_zero(ring: &F) -> Self {
        Self {
            coefficients: vec![],
            exponents: vec![],
            context: Arc::new(PolynomialContext {
                ring: ring.clone(),
                variables: Arc::new(vec![]),
            }),
            _phantom: PhantomData,
        }
    }

    /// Constructs a polynomial that is one. Instead of using this constructor,
    /// prefer to create new polynomials from existing ones, so that the
    /// variable map is inherited.
    #[inline]
    pub fn new_one(ring: &F) -> Self {
        Self {
            coefficients: vec![ring.one()],
            exponents: vec![],
            context: Arc::new(PolynomialContext {
                ring: ring.clone(),
                variables: Arc::new(vec![]),
            }),
            _phantom: PhantomData,
        }
    }

    /// Constructs a zero polynomial, inheriting the field and variable map from `self`.
    #[inline]
    pub fn zero(&self) -> Self {
        Self::from_context(None, self.context.clone())
    }

    /// Constructs a zero polynomial with the given number of variables and capacity,
    /// inheriting the field and variable map from `self`.
    #[inline]
    pub fn zero_with_capacity(&self, cap: usize) -> Self {
        Self::from_context(Some(cap), self.context.clone())
    }

    /// Constructs a constant polynomial,
    /// inheriting the field and variable map from `self`.
    #[inline]
    pub fn constant(&self, coeff: F::Element) -> Self {
        if self.ring().is_zero(&coeff) {
            return self.zero();
        }

        Self {
            coefficients: vec![coeff],
            exponents: vec![E::zero(); self.nvars()],
            context: self.context.clone(),
            _phantom: PhantomData,
        }
    }

    /// Constructs a polynomial that is one, inheriting the field and variable map from `self`.
    #[inline]
    pub fn one(&self) -> Self {
        Self {
            coefficients: vec![self.ring().one()],
            exponents: vec![E::zero(); self.nvars()],
            context: self.context.clone(),
            _phantom: PhantomData,
        }
    }

    /// Constructs a polynomial with a single term.
    #[inline]
    pub fn monomial(&self, coeff: F::Element, exponents: Vec<E>) -> Self {
        debug_assert!(self.nvars() == exponents.len());

        if self.ring().is_zero(&coeff) {
            return self.zero();
        }

        Self {
            coefficients: vec![coeff],
            exponents,
            context: self.context.clone(),
            _phantom: PhantomData,
        }
    }

    /// Constructs a polynomial with a single term that is a variable.
    #[inline]
    pub fn variable(&self, var: &PolyVariable) -> Result<Self, String> {
        if let Some(pos) = self.variables().iter().position(|v| v == var) {
            let mut exp = vec![E::zero(); self.nvars()];
            exp[pos] = E::one();
            Ok(self.monomial(self.ring().one(), exp))
        } else {
            Err(format!("Variable {} not found", var))
        }
    }

    /// Get the ith monomial
    pub fn to_monomial_view(&self, i: usize) -> MonomialView<'_, F, E> {
        assert!(i < self.nterms());

        MonomialView {
            coefficient: &self.coefficients[i],
            exponents: self.exponents(i),
        }
    }

    #[inline]
    pub fn reserve(&mut self, cap: usize) -> &mut Self {
        self.coefficients.reserve(cap);
        self.exponents.reserve(cap * self.nvars());
        self
    }

    /// Shrinks the coefficient and exponent buffers to fit their current lengths.
    ///
    /// This is intended for polynomials that will be stored for a while. Calling it
    /// repeatedly on intermediate values can cause avoidable reallocations.
    #[inline]
    pub fn compact(&mut self) -> &mut Self {
        self.coefficients.shrink_to_fit();
        self.exponents.shrink_to_fit();
        self
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.nterms() == 0
    }

    #[inline]
    pub fn is_one(&self) -> bool {
        self.nterms() == 1
            && self.ring().is_one(&self.coefficients[0])
            && self.exponents.iter().all(|x| x.is_zero())
    }

    /// Returns the number of terms in the polynomial.
    #[inline]
    pub fn nterms(&self) -> usize {
        self.coefficients.len()
    }

    /// Returns the number of variables in the polynomial.
    #[inline]
    pub fn nvars(&self) -> usize {
        self.variables().len()
    }

    /// Returns true if the polynomial is constant.
    #[inline]
    pub fn is_constant(&self) -> bool {
        if self.is_zero() {
            return true;
        }
        if self.nterms() >= 2 {
            return false;
        }
        debug_assert!(!self.ring().is_zero(self.coefficients.first().unwrap()));
        self.exponents.iter().all(|e| e.is_zero())
    }

    /// Get the constant term of the polynomial.
    #[inline]
    pub fn get_constant(&self) -> F::Element {
        if self.is_zero() || !self.exponents(0).iter().all(|e| e.is_zero()) {
            return self.ring().zero();
        }

        self.coefficients[0].clone()
    }

    /// Returns the `index`th monomial, starting from the back.
    #[inline]
    pub fn coefficient_back(&self, index: usize) -> &F::Element {
        &self.coefficients[self.nterms() - index - 1]
    }

    /// Returns the slice for the exponents of the specified monomial.
    #[inline]
    pub fn exponents(&self, index: usize) -> &[E] {
        //&self.exponents[index * self.nvars()..(index + 1) * self.nvars()]
        unsafe {
            self.exponents
                .get_unchecked(index * self.nvars()..(index + 1) * self.nvars())
        }
    }

    /// Returns the slice for the exponents of the specified monomial
    /// starting from the back.
    #[inline]
    pub fn exponents_back(&self, index: usize) -> &[E] {
        let index = self.nterms() - index - 1;
        &self.exponents[index * self.nvars()..(index + 1) * self.nvars()]
    }

    #[inline(always)]
    pub fn last_exponents(&self) -> &[E] {
        //assert!(self.nterms() > 0);
        &self.exponents[(self.nterms() - 1) * self.nvars()..self.nterms() * self.nvars()]
    }

    /// Returns the mutable slice for the exponents of the specified monomial.
    #[inline]
    pub fn exponents_mut(&mut self, index: usize) -> &mut [E] {
        let nvars = self.nvars();
        &mut self.exponents[index * nvars..(index + 1) * nvars]
    }

    /// Returns an iterator over the exponents of every monomial.
    #[inline]
    pub fn exponents_iter(&self) -> std::slice::Chunks<'_, E> {
        self.exponents.chunks(self.nvars())
    }

    /// Returns an iterator over the mutable exponents of every monomial.
    #[inline]
    pub fn exponents_iter_mut(&mut self) -> std::slice::ChunksMut<'_, E> {
        let nvars = self.nvars();
        self.exponents.chunks_mut(nvars)
    }

    /// Reset the polynomial to 0.
    #[inline]
    pub fn clear(&mut self) {
        self.coefficients.clear();
        self.exponents.clear();
    }

    /// Get a copy of the variable list.
    pub fn get_vars(&self) -> Arc<Vec<PolyVariable>> {
        self.variables().clone()
    }

    /// Get a reference to the variables list.
    pub fn get_vars_ref(&self) -> &[PolyVariable] {
        self.variables().as_ref()
    }

    /// Rename a variable.
    pub fn rename_variable(&mut self, old: &PolyVariable, new: &PolyVariable) {
        if let Some(pos) = self.variables().iter().position(|v| v == old) {
            let mut new_vars = self.variables().as_ref().clone();
            new_vars[pos] = new.clone();
            self.set_variables(Arc::new(new_vars));
        }
    }

    /// Unify the variable maps of two polynomials, i.e.
    /// rewrite a polynomial in `x` and one in `y` to a
    /// two polynomial in `x` and `y`.
    ///
    /// The variable map will be inherited from
    /// `self` and will be extended by variables occurring
    /// in `other`.
    #[inline(always)]
    pub fn unify_variables(&mut self, other: &mut Self) {
        if self.variables() == other.variables() {
            // Reduce the number of equal context copies as well as variable copies.
            if self.ring() == other.ring() && !Arc::ptr_eq(&self.context, &other.context) {
                if Arc::as_ptr(&self.context) < Arc::as_ptr(&other.context) {
                    other.context = self.context.clone();
                } else {
                    self.context = other.context.clone();
                }
            }
            return;
        }

        self.unify_variables_impl(other)
    }

    fn unify_variables_impl(&mut self, other: &mut Self) {
        let mut new_var_map = self.variables().as_ref().clone();
        let mut new_var_pos_other = vec![0; other.nvars()];
        for (pos, v) in new_var_pos_other.iter_mut().zip(other.variables().as_ref()) {
            if let Some(p) = new_var_map.iter().position(|x| x == v) {
                *pos = p;
            } else {
                *pos = new_var_map.len();
                new_var_map.push(v.clone());
            }
        }

        let mut newexp = vec![E::zero(); new_var_map.len() * self.nterms()];

        for t in 0..self.nterms() {
            newexp[t * new_var_map.len()..t * new_var_map.len() + self.nvars()]
                .copy_from_slice(self.exponents(t));
        }

        self.set_variables(Arc::new(new_var_map));
        self.exponents = newexp;

        // check if term ordering remains unchanged
        if new_var_pos_other.windows(2).all(|w| w[0] <= w[1]) {
            let mut newexp = vec![E::zero(); self.nvars() * other.nterms()];

            if other.nvars() > 0 {
                for (d, t) in newexp
                    .chunks_mut(self.nvars())
                    .zip(other.exponents.chunks(other.nvars()))
                {
                    for (var, e) in t.iter().enumerate() {
                        d[new_var_pos_other[var]] = *e;
                    }
                }
            }

            other.set_variables(self.variables().clone());
            other.exponents = newexp;
            return;
        }

        // reconstruct 'other' with correct monomial ordering
        let mut newother = Self::new(
            other.ring(),
            other.nterms().into(),
            self.variables().clone(),
        );
        let mut newexp = vec![E::zero(); self.nvars()];
        for t in other.into_iter() {
            for c in &mut newexp {
                *c = E::zero();
            }

            for (var, e) in t.exponents.iter().enumerate() {
                newexp[new_var_pos_other[var]] = *e;
            }
            newother.append_monomial(t.coefficient.clone(), &newexp);
        }
        *other = newother;
    }

    /// Unify the variable maps of all polynomials in the slice.
    pub fn unify_variables_list(polys: &mut [Self]) {
        if polys.len() < 2 {
            return;
        }

        let (first, rest) = polys.split_first_mut().unwrap();
        for _ in 0..2 {
            for p in &mut *rest {
                first.unify_variables(p);
            }
        }
    }

    /// Reverse the coefficients: `1+2x+3x^3` becomes `3+2x^2+x^3`.
    pub fn reverse(&mut self) {
        self.coefficients.reverse();
        let nterms = self.nterms();
        let nvars = self.nvars();
        let degs = (0..nvars).map(|i| self.degree(i)).collect::<Vec<_>>();

        for e in self.exponents.chunks_mut(nvars) {
            for (ee, d) in e.iter_mut().zip(&degs) {
                *ee = *d - *ee;
            }
        }

        let midu = if nterms.is_multiple_of(2) {
            self.nvars() * (nterms / 2)
        } else {
            self.nvars() * (nterms / 2 + 1)
        };

        let (l, r) = self.exponents.split_at_mut(midu);

        let rend = r.len();
        for i in 0..nterms / 2 {
            l[i * nvars..(i + 1) * nvars]
                .swap_with_slice(&mut r[rend - (i + 1) * nvars..rend - i * nvars]);
        }
    }

    /// Reverse the monomial ordering in-place.
    fn reverse_monomials(&mut self) {
        let nterms = self.nterms();
        let nvars = self.nvars();
        if nterms < 2 {
            return;
        }

        self.coefficients.reverse();

        let midu = if nterms.is_multiple_of(2) {
            self.nvars() * (nterms / 2)
        } else {
            self.nvars() * (nterms / 2 + 1)
        };

        let (l, r) = self.exponents.split_at_mut(midu);

        let rend = r.len();
        for i in 0..nterms / 2 {
            l[i * nvars..(i + 1) * nvars]
                .swap_with_slice(&mut r[rend - (i + 1) * nvars..rend - i * nvars]);
        }
    }

    /// Add a variable to the polynomial if it is not already present.
    pub fn add_variable(&mut self, var: &PolyVariable) {
        if self.variables().iter().any(|v| v == var) {
            return;
        }

        let l = self.variables().len();

        let mut new_exp = vec![E::zero(); (l + 1) * self.nterms()];

        if l > 0 {
            for (en, e) in new_exp.chunks_mut(l + 1).zip(self.exponents.chunks(l)) {
                en[..l].copy_from_slice(e);
            }
        }

        let mut new_vars = self.variables().as_ref().clone();
        new_vars.push(var.clone());
        self.set_variables(Arc::new(new_vars));
        self.exponents = new_exp;
    }

    /// Add several variables to the polynomial if they are not already present.
    pub fn add_variables(&mut self, vars: &[PolyVariable]) {
        // collect only genuinely new variables
        let new_vars: Vec<_> = vars
            .iter()
            .filter(|var| !self.variables().iter().any(|v| v == *var))
            .cloned()
            .collect();

        if new_vars.is_empty() {
            return;
        }

        let l_old = self.variables().len();
        let n_new = new_vars.len();
        let l_new = l_old + n_new;

        let mut new_exp = vec![E::zero(); l_new * self.nterms()];

        if l_old > 0 {
            for (en, e) in new_exp.chunks_mut(l_new).zip(self.exponents.chunks(l_old)) {
                en[..l_old].copy_from_slice(e);
            }
        }

        let mut variables = self.variables().as_ref().clone();
        variables.extend(new_vars);

        self.set_variables(Arc::new(variables));
        self.exponents = new_exp;
    }

    /// Check if the polynomial is sorted and has only non-zero coefficients
    pub fn check_consistency(&self) {
        assert_eq!(self.coefficients.len(), self.nterms());
        assert_eq!(self.exponents.len(), self.nterms() * self.nvars());

        for c in &self.coefficients {
            if self.ring().is_zero(c) {
                panic!("Inconsistent polynomial (0 coefficient): {self}");
            }
        }

        for t in 1..self.nterms() {
            match O::cmp(self.exponents(t), self.exponents(t - 1)) {
                Ordering::Equal => panic!("Inconsistent polynomial (equal monomials): {self}"),
                Ordering::Less => {
                    panic!("Inconsistent polynomial (wrong monomial ordering): {self}")
                }
                Ordering::Greater => {}
            }
        }
    }

    /// Append a monomial to the back. It merges with the last monomial if the
    /// exponents are equal.
    #[inline]
    pub fn append_monomial_back(&mut self, coefficient: F::Element, exponents: &[E]) {
        if self.ring().is_zero(&coefficient) {
            return;
        }

        let nterms = self.nterms();
        if nterms > 0 && exponents == self.last_exponents() {
            self.context
                .ring
                .add_assign(&mut self.coefficients[nterms - 1], &coefficient);

            if self.context.ring.is_zero(&self.coefficients[nterms - 1]) {
                self.coefficients.pop();
                self.exponents.truncate((nterms - 1) * self.nvars());
            }
        } else {
            self.coefficients.push(coefficient);
            self.exponents.extend_from_slice(exponents);
        }
    }

    /// Appends a monomial to the polynomial.
    pub fn append_monomial(&mut self, coefficient: F::Element, exponents: &[E]) {
        if self.ring().is_zero(&coefficient) {
            return;
        }
        if self.nvars() != exponents.len() {
            panic!(
                "nvars mismatched: got {}, expected {}",
                exponents.len(),
                self.nvars()
            );
        }

        // should we append to the back?
        if self.nterms() == 0 || O::cmp(self.last_exponents(), exponents).is_lt() {
            self.coefficients.push(coefficient);
            self.exponents.extend_from_slice(exponents);
            return;
        }

        if O::cmp(self.exponents(0), exponents).is_gt() {
            self.coefficients.insert(0, coefficient);
            self.exponents.splice(0..0, exponents.iter().cloned());
            return;
        }

        // Binary search to find the insert-point.
        let mut l = 0;
        let mut r = self.nterms();

        while l <= r {
            let m = (l + r) / 2;
            let c = O::cmp(exponents, self.exponents(m)); // note the reversal

            match c {
                Ordering::Equal => {
                    // Add the two coefficients.
                    self.context
                        .ring
                        .add_assign(&mut self.coefficients[m], &coefficient);
                    if self.context.ring.is_zero(&self.coefficients[m]) {
                        // The coefficient becomes zero. Remove this monomial.
                        self.coefficients.remove(m);
                        let i = m * self.nvars();
                        self.exponents.splice(i..i + self.nvars(), Vec::new());
                    }
                    return;
                }
                Ordering::Greater => {
                    l = m + 1;

                    if l == self.nterms() {
                        self.coefficients.push(coefficient);
                        self.exponents.extend_from_slice(exponents);
                        return;
                    }
                }
                Ordering::Less => {
                    if m == 0 {
                        self.coefficients.insert(0, coefficient);
                        self.exponents.splice(0..0, exponents.iter().cloned());
                        return;
                    }

                    r = m - 1;
                }
            }
        }

        self.coefficients.insert(l, coefficient);
        let i = l * self.nvars();
        self.exponents.splice(i..i, exponents.iter().cloned());
    }
}

impl<F: Ring, E: Exponent, O: MonomialOrder> SelfRing for MultivariatePolynomial<F, E, O> {
    #[inline]
    fn is_zero(&self) -> bool {
        self.is_zero()
    }

    #[inline]
    fn is_one(&self) -> bool {
        self.is_one()
    }

    fn format<W: std::fmt::Write>(
        &self,
        opts: &PrintOptions,
        mut state: PrintState,
        f: &mut W,
    ) -> Result<bool, std::fmt::Error> {
        let print_ring = opts.print_ring && !self.ring().has_independent_elements();

        if self.is_constant() {
            if self.is_zero() {
                if state.in_sum {
                    f.write_str("+")?;
                }
                f.write_char('0')?;
                return Ok(false);
            } else if !print_ring || state.level > 0 {
                return self.ring().format(&self.coefficients[0], opts, state, f);
            }
        }

        let add_paren = (self.nterms() > 1 || print_ring) && state.in_product
            || ((state.in_exp || state.in_exp_base)
                && (self.nterms() > 1
                    || print_ring
                    || self.exponents(0).iter().filter(|e| **e > E::zero()).count() > 1
                    || !self.ring().is_one(&self.coefficients[0])));

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

        let var_map: Vec<String> = self
            .variables()
            .as_ref()
            .iter()
            .map(|v| {
                v.format_string(
                    opts,
                    PrintState {
                        in_exp: true,
                        ..state
                    },
                )
            })
            .collect();

        state.level += 1;
        for monomial in self {
            let has_var = monomial.exponents.iter().any(|e| !e.is_zero());
            state.in_product = in_product || has_var;
            state.suppress_one = has_var; // any products before should not be considered
            state.in_exp |= print_ring; // make sure to add parentheses

            let mut suppressed_one = self.ring().format(monomial.coefficient, opts, state, f)?;

            for (var_id, e) in var_map.iter().zip(monomial.exponents) {
                if e.is_zero() {
                    continue;
                }
                if suppressed_one {
                    suppressed_one = false;
                } else if !opts.mode.is_latex() {
                    f.write_char(opts.multiplication_operator)?;
                }

                f.write_str(var_id)?;

                if e.to_i32() != 1 {
                    if opts.mode.is_latex() {
                        write!(f, "^{{{e}}}")?;
                    } else if opts.double_star_for_exponentiation {
                        write!(f, "**{e}")?;
                    } else if opts.mode.is_symbolica() && opts.num_exp_as_superscript {
                        state.superscript = true;
                        AtomPrinter::format_digits(e.to_string(), opts, &state, f)?;
                        state.superscript = false;
                    } else {
                        write!(f, "^{e}")?;
                    }
                }
            }

            state.in_sum = true;
        }
        state.level -= 1;

        if self.is_zero() {
            f.write_char('0')?;
        }

        if print_ring && state.level == 0 {
            self.ring().format_ring(opts, state, f)?;
        }

        if add_paren {
            f.write_str(")")?;
        }

        Ok(false)
    }
}

impl<F: Ring + std::fmt::Debug, E: Exponent + std::fmt::Debug, O: MonomialOrder> std::fmt::Debug
    for MultivariatePolynomial<F, E, O>
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.is_zero() {
            return write!(f, "[]");
        }
        let mut first = true;
        write!(f, "[ ")?;
        for monomial in self {
            if first {
                first = false;
            } else {
                write!(f, ", ")?;
            }
            write!(
                f,
                "{{ {:?}, {:?} }}",
                monomial.coefficient, monomial.exponents
            )?;
        }
        write!(f, " ]")
    }
}

impl<F: Ring + Display, E: Exponent, O: MonomialOrder> Display for MultivariatePolynomial<F, E, O> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.format(&PrintOptions::from_fmt(f), PrintState::from_fmt(f), f)
            .map(|_| ())
    }
}

impl<F: Ring, E: Exponent, O: MonomialOrder> PartialEq for MultivariatePolynomial<F, E, O> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if self.variables() != other.variables() {
            if self.is_constant() != other.is_constant() {
                return false;
            }

            if self.is_zero() != other.is_zero() {
                return false;
            }

            if self.is_zero() {
                return true;
            }

            if self.is_constant() {
                return self.coefficients[0] == other.coefficients[0];
            }

            return false;
        }
        if self.nterms() != other.nterms() {
            return false;
        }
        self.exponents.eq(&other.exponents) && self.coefficients.eq(&other.coefficients)
    }
}

impl<F: Ring, E: Exponent, O: MonomialOrder> std::hash::Hash for MultivariatePolynomial<F, E, O> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.coefficients.hash(state);
        self.exponents.hash(state);

        if !self.is_constant() {
            self.variables().hash(state);
        }
    }
}

impl<F: Ring, E: Exponent, O: MonomialOrder> Eq for MultivariatePolynomial<F, E, O> {}

impl<R: Ring, E: Exponent, O: MonomialOrder> InternalOrdering for MultivariatePolynomial<R, E, O> {
    /// An ordering of polynomials that has no intuitive meaning.
    fn internal_cmp(&self, other: &Self) -> Ordering {
        // TODO: what about different variables?
        Ord::cmp(&self.exponents, &other.exponents)
            .then_with(|| self.coefficients.internal_cmp(&other.coefficients))
    }
}

impl<F: Ring, E: Exponent, O: MonomialOrder> Add for MultivariatePolynomial<F, E, O> {
    type Output = Self;

    fn add(mut self, mut other: Self) -> Self::Output {
        assert_eq!(self.ring(), other.ring());

        self.unify_variables(&mut other);

        if self.is_zero() {
            return other;
        }
        if other.is_zero() {
            return self;
        }

        // Merge the two polynomials, which are assumed to be already sorted.

        let mut new_coefficients = vec![self.ring().zero(); self.nterms() + other.nterms()];
        let mut new_exponents: Vec<E> =
            vec![E::zero(); self.nvars() * (self.nterms() + other.nterms())];
        let mut new_nterms = 0;
        let mut i = 0;
        let mut j = 0;

        macro_rules! insert_monomial {
            ($source:expr, $index:expr) => {
                mem::swap(
                    &mut new_coefficients[new_nterms],
                    &mut $source.coefficients[$index],
                );

                new_exponents[new_nterms * $source.nvars()..(new_nterms + 1) * $source.nvars()]
                    .clone_from_slice($source.exponents($index));
                new_nterms += 1;
            };
        }

        while i < self.nterms() && j < other.nterms() {
            let c = O::cmp(self.exponents(i), other.exponents(j));
            match c {
                Ordering::Less => {
                    insert_monomial!(self, i);
                    i += 1;
                }
                Ordering::Greater => {
                    insert_monomial!(other, j);
                    j += 1;
                }
                Ordering::Equal => {
                    self.context
                        .ring
                        .add_assign(&mut self.coefficients[i], &other.coefficients[j]);
                    if !self.context.ring.is_zero(&self.coefficients[i]) {
                        insert_monomial!(self, i);
                    }
                    i += 1;
                    j += 1;
                }
            }
        }

        while i < self.nterms() {
            insert_monomial!(self, i);
            i += 1;
        }

        while j < other.nterms() {
            insert_monomial!(other, j);
            j += 1;
        }

        new_coefficients.truncate(new_nterms);
        new_exponents.truncate(self.nvars() * new_nterms);

        Self {
            coefficients: new_coefficients,
            exponents: new_exponents,
            context: self.context,
            _phantom: PhantomData,
        }
    }
}

impl<F: Ring, E: Exponent, O: MonomialOrder> Add<&MultivariatePolynomial<F, E, O>>
    for &MultivariatePolynomial<F, E, O>
{
    type Output = MultivariatePolynomial<F, E, O>;

    fn add(self, other: &MultivariatePolynomial<F, E, O>) -> Self::Output {
        assert_eq!(self.ring(), other.ring());

        if self.is_zero() {
            return other.clone();
        }
        if other.is_zero() {
            return self.clone();
        }

        if self.variables() != other.variables() {
            let mut c1 = self.clone();
            let mut c2 = other.clone();
            c1.unify_variables(&mut c2);
            return c1 + c2;
        }

        // Merge the two polynomials, which are assumed to be already sorted.
        let mut new_coefficients = vec![self.ring().zero(); self.nterms() + other.nterms()];
        let mut new_exponents: Vec<E> =
            vec![E::zero(); self.nvars() * (self.nterms() + other.nterms())];
        let mut new_nterms = 0;
        let mut i = 0;
        let mut j = 0;

        macro_rules! insert_monomial {
            ($source:expr, $index:expr) => {
                new_coefficients[new_nterms] = $source.coefficients[$index].clone();
                new_exponents[new_nterms * $source.nvars()..(new_nterms + 1) * $source.nvars()]
                    .clone_from_slice($source.exponents($index));
                new_nterms += 1;
            };
        }

        while i < self.nterms() && j < other.nterms() {
            let c = O::cmp(self.exponents(i), other.exponents(j));
            match c {
                Ordering::Less => {
                    insert_monomial!(self, i);
                    i += 1;
                }
                Ordering::Greater => {
                    insert_monomial!(other, j);
                    j += 1;
                }
                Ordering::Equal => {
                    let coeff = self
                        .ring()
                        .add(&self.coefficients[i], &other.coefficients[j]);
                    if !self.ring().is_zero(&coeff) {
                        new_coefficients[new_nterms] = coeff;
                        new_exponents[new_nterms * self.nvars()..(new_nterms + 1) * self.nvars()]
                            .clone_from_slice(self.exponents(i));
                        new_nterms += 1;
                    }
                    i += 1;
                    j += 1;
                }
            }
        }

        while i < self.nterms() {
            insert_monomial!(self, i);
            i += 1;
        }

        while j < other.nterms() {
            insert_monomial!(other, j);
            j += 1;
        }

        new_coefficients.truncate(new_nterms);
        new_exponents.truncate(self.nvars() * new_nterms);

        MultivariatePolynomial {
            coefficients: new_coefficients,
            exponents: new_exponents,
            context: self.context.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<F: Ring, E: Exponent, O: MonomialOrder> Sub for MultivariatePolynomial<F, E, O> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self.add(other.neg())
    }
}

impl<'a, F: Ring, E: Exponent, O: MonomialOrder> Sub<&'a MultivariatePolynomial<F, E, O>>
    for &MultivariatePolynomial<F, E, O>
{
    type Output = MultivariatePolynomial<F, E, O>;

    fn sub(self, other: &'a MultivariatePolynomial<F, E, O>) -> Self::Output {
        self + &other.clone().neg() // TODO: improve
    }
}

impl<F: Ring, E: Exponent, O: MonomialOrder> Neg for MultivariatePolynomial<F, E, O> {
    type Output = Self;
    fn neg(mut self) -> Self::Output {
        // Negate coefficients of all terms.
        let ring = &self.context.ring;
        for c in &mut self.coefficients {
            *c = ring.neg(&*c);
        }
        self
    }
}

impl<'a, F: Ring, E: Exponent> Mul<&'a MultivariatePolynomial<F, E, LexOrder>>
    for &MultivariatePolynomial<F, E, LexOrder>
{
    type Output = MultivariatePolynomial<F, E, LexOrder>;

    #[inline]
    fn mul(self, rhs: &'a MultivariatePolynomial<F, E, LexOrder>) -> Self::Output {
        assert_eq!(self.ring(), rhs.ring());

        if self.nterms() == 0 || rhs.nterms() == 0 {
            return self.zero();
        }

        if self.is_constant() {
            return rhs.clone().mul_coeff(self.coefficients[0].clone());
        }
        if rhs.is_constant() {
            return self.clone().mul_coeff(rhs.coefficients[0].clone());
        }

        if self.variables() != rhs.variables() {
            let mut c1 = self.clone();
            let mut c2 = rhs.clone();
            c1.unify_variables(&mut c2);
            return c1.mul(&c2);
        }

        if self.nterms() == 1 {
            return rhs
                .clone()
                .mul_monomial(&self.coefficients[0], &self.exponents);
        }

        if rhs.nterms() == 1 {
            return self
                .clone()
                .mul_monomial(&rhs.coefficients[0], &rhs.exponents);
        }

        if let Some(r) = self.mul_dense(rhs) {
            r
        } else {
            self.heap_mul(rhs)
        }
    }
}

impl<'a, F: Ring, E: Exponent> Mul<&'a MultivariatePolynomial<F, E, LexOrder>>
    for MultivariatePolynomial<F, E, LexOrder>
{
    type Output = MultivariatePolynomial<F, E, LexOrder>;

    /// Multiply two polynomials, using either use dense multiplication or heap multiplication.
    #[inline]
    fn mul(self, rhs: &'a MultivariatePolynomial<F, E, LexOrder>) -> Self::Output {
        (&self) * rhs
    }
}

impl<'a, F: EuclideanDomain, E: PositiveExponent> Div<&'a MultivariatePolynomial<F, E, LexOrder>>
    for &MultivariatePolynomial<F, E, LexOrder>
{
    type Output = MultivariatePolynomial<F, E, LexOrder>;

    fn div(self, other: &'a MultivariatePolynomial<F, E, LexOrder>) -> Self::Output {
        self.try_div(other)
            .unwrap_or_else(|| panic!("No clean division of {self} by {other}"))
    }
}

impl<'a, F: EuclideanDomain, E: PositiveExponent> Div<&'a MultivariatePolynomial<F, E, LexOrder>>
    for MultivariatePolynomial<F, E, LexOrder>
{
    type Output = MultivariatePolynomial<F, E, LexOrder>;

    fn div(
        self: MultivariatePolynomial<F, E, LexOrder>,
        other: &'a MultivariatePolynomial<F, E, LexOrder>,
    ) -> Self::Output {
        (&self).div(other)
    }
}

impl<F: Ring, E: Exponent, O: MonomialOrder> MultivariatePolynomial<F, E, O> {
    /// Change the monomial order of the polynomial from `O` to `ON`.
    pub fn reorder<ON: MonomialOrder>(&self) -> MultivariatePolynomial<F, E, ON> {
        let mut sorted_index: Vec<_> = (0..self.nterms()).collect();
        sorted_index.sort_by(|a, b| ON::cmp(self.exponents(*a), self.exponents(*b)));

        let coefficients: Vec<_> = sorted_index
            .iter()
            .map(|i| self.coefficients[*i].clone())
            .collect();
        let exponents: Vec<_> = sorted_index
            .iter()
            .flat_map(|i| self.exponents(*i))
            .cloned()
            .collect();

        MultivariatePolynomial {
            coefficients,
            exponents,
            context: self.context.clone(),
            _phantom: PhantomData,
        }
    }

    /// Multiply every coefficient with `other`.
    pub fn mul_coeff(mut self, other: F::Element) -> Self {
        if self.ring().is_one(&other) {
            return self;
        }

        let ring = &self.context.ring;
        for c in &mut self.coefficients {
            ring.mul_assign(c, &other);
        }

        for i in (0..self.nterms()).rev() {
            if self.ring().is_zero(&self.coefficients[i]) {
                self.coefficients.remove(i);
                self.exponents
                    .drain(i * self.nvars()..(i + 1) * self.nvars());
            }
        }

        self
    }

    /// Map a coefficient using the function `f`.
    pub fn map_coeff<U: Ring, T: Fn(&F::Element) -> U::Element>(
        &self,
        f: T,
        field: U,
    ) -> MultivariatePolynomial<U, E, O> {
        let mut coefficients = Vec::with_capacity(self.coefficients.len());
        let mut exponents = Vec::with_capacity(self.exponents.len());

        for m in self.into_iter() {
            let nc = f(m.coefficient);
            if !field.is_zero(&nc) {
                coefficients.push(nc);
                exponents.extend(m.exponents);
            }
        }

        MultivariatePolynomial {
            coefficients,
            exponents,
            context: Arc::new(PolynomialContext {
                ring: field,
                variables: self.variables().clone(),
            }),
            _phantom: PhantomData,
        }
    }

    /// Add `exponents` to every exponent.
    pub fn mul_exp(mut self, exponents: &[E]) -> Self {
        debug_assert_eq!(self.nvars(), exponents.len());

        if self.nvars() == 0 {
            return self;
        }

        for e in self.exponents_iter_mut() {
            for (e1, e2) in e.iter_mut().zip(exponents) {
                *e1 = e1.checked_add(e2).expect("overflow in adding exponents");
            }
        }

        self
    }

    #[inline]
    pub fn max_coeff(&self) -> &F::Element {
        self.coefficients.last().unwrap()
    }

    #[inline]
    pub fn max_exp(&self) -> &[E] {
        if self.coefficients.is_empty() {
            panic!("Cannot get max exponent of empty polynomial");
        }

        &self.exponents[(self.nterms() - 1) * self.nvars()..self.nterms() * self.nvars()]
    }

    /// Add a new monomial with coefficient `other` and exponent one.
    pub fn add_constant(mut self, other: F::Element) -> Self {
        let nvars = self.nvars();
        self.append_monomial(other, &vec![E::zero(); nvars]);
        self
    }

    #[inline]
    pub fn mul_monomial(self, coefficient: &F::Element, exponents: &[E]) -> Self {
        self.mul_coeff(coefficient.clone()).mul_exp(exponents)
    }

    /// Check if the polynomial contains a variable `x`.
    pub fn contains(&self, x: usize) -> bool {
        if self.nvars() == 0 {
            return false;
        }

        for e in self.exponents.iter().skip(x).step_by(self.nvars()) {
            if *e != E::zero() {
                return true;
            }
        }
        false
    }

    /// Get the degree of the variable `x`.
    /// This operation is O(n).
    pub fn degree(&self, x: usize) -> E {
        if self.nvars() == 0 || self.is_zero() {
            return E::zero();
        }

        let mut max = self.exponents[x];
        for t in 1..self.nterms() {
            let e = self.exponents[x + t * self.nvars()];
            if max < e {
                max = e;
            }
        }
        max
    }

    /// Get the lowest and highest exponent of the variable `x`.
    /// This operation is O(n).
    pub fn degree_bounds(&self, x: usize) -> (E, E) {
        if self.nvars() == 0 || self.is_zero() {
            return (E::zero(), E::zero());
        }

        let mut min = None;
        let mut max = None;
        for e in self.exponents.iter().skip(x).step_by(self.nvars()) {
            if max.map(|max| max < *e).unwrap_or(true) {
                max = Some(*e);
            }
            if min.map(|min| min > *e).unwrap_or(true) {
                min = Some(*e);
            }
        }
        (min.unwrap_or(E::zero()), max.unwrap_or(E::zero()))
    }

    // Get the highest degree of a variable in the leading monomial.
    pub fn ldegree(&self, v: usize) -> E {
        if self.is_zero() {
            return E::zero();
        }
        self.last_exponents()[v]
    }

    /// Get the highest degree of the leading monomial.
    pub fn ldegree_max(&self) -> E {
        if self.is_zero() {
            return E::zero();
        }
        *self.last_exponents().iter().max().unwrap_or(&E::zero())
    }

    /// Get the leading coefficient.
    pub fn lcoeff(&self) -> F::Element {
        if self.is_zero() {
            return self.ring().zero();
        }
        self.coefficients.last().unwrap().clone()
    }

    /// Perform self % var^pow.
    pub fn mod_var(&self, var: usize, pow: E) -> Self {
        let mut m = self.zero();
        for t in self.into_iter() {
            if t.exponents[var] < pow {
                m.append_monomial(t.coefficient.clone(), t.exponents);
            }
        }
        m
    }

    /// Take the derivative of the polynomial w.r.t the variable `var`.
    pub fn derivative(&self, var: usize) -> Self {
        debug_assert!(var < self.nvars());

        let mut res = self.zero_with_capacity(self.nterms());

        let mut exp = vec![E::zero(); self.nvars()];
        for x in self {
            if x.exponents[var] > E::zero() {
                exp.copy_from_slice(x.exponents);
                let pow = exp[var].to_i32() as u64;
                exp[var] = exp[var] - E::one();
                res.append_monomial(
                    self.ring().mul(x.coefficient, &self.ring().nth(pow.into())),
                    &exp,
                );
            }
        }

        res
    }

    /// Get the coefficient of the monomial with the given exponents if it is present.
    pub fn coefficient(&self, exponents: &[E]) -> Option<F::Element> {
        if self.is_zero() {
            if exponents.iter().all(|e| *e == E::zero()) {
                return Some(self.ring().zero());
            }
            return None;
        }

        let mut low = 0;
        let mut high = self.coefficients.len();

        while low < high {
            let mid = low + (high - low) / 2;

            match O::cmp(
                &self.exponents[mid * self.nvars()..(mid + 1) * self.nvars()],
                exponents,
            ) {
                Ordering::Equal => return Some(self.coefficients[mid].clone()),
                Ordering::Less => {
                    low = mid + 1;
                }
                Ordering::Greater => {
                    high = mid;
                }
            }
        }

        None
    }

    pub fn map_exp<E2: Exponent>(&self, f: impl Fn(&E) -> E2) -> MultivariatePolynomial<F, E2, O> {
        MultivariatePolynomial {
            coefficients: self.coefficients.clone(),
            exponents: self.exponents.iter().map(f).collect::<Vec<_>>(),
            context: self.context.clone(),
            _phantom: PhantomData,
        }
    }

    /// Kronecker map all variables starting from `start_index` using the powers given in `powers`:
    /// `x_i -> x_{start_index}^powers[i - start_index]` for `i > start_index`.
    pub fn kronecker_map(&self, powers: &[E], start_index: usize) -> Self {
        let mut res = self.zero_with_capacity(self.nterms());
        let mut new_exponents = vec![E::zero(); self.nvars()];
        for a in self {
            new_exponents[..(start_index + 1)].copy_from_slice(&a.exponents[..(start_index + 1)]);

            for (i, e) in a.exponents.iter().skip(start_index + 1).enumerate() {
                new_exponents[start_index] += powers[i] * *e;
            }
            res.append_monomial(a.coefficient.clone(), &new_exponents);
        }
        res
    }

    /// Invert a Kronecker map.
    pub fn kronecker_inv_map(&self, powers: &[E], start_index: usize) -> Self {
        let mut res = self.zero_with_capacity(self.nterms());
        let mut new_exponents = vec![E::zero(); self.nvars()];
        for a in self {
            new_exponents[..start_index].copy_from_slice(&a.exponents[..start_index]);

            let mut total = a.exponents[start_index];
            new_exponents[start_index] = total % powers[0];
            for i in (start_index + 1)..self.nvars() {
                let previous_power = if i == start_index + 1 {
                    E::one()
                } else {
                    powers[i - start_index - 2]
                };
                total = total - new_exponents[i - 1] * previous_power;
                new_exponents[i] = total % powers[i - start_index];
                new_exponents[i] = new_exponents[i] / powers[i - start_index - 1];
            }
            res.append_monomial(a.coefficient.clone(), &new_exponents);
        }
        res
    }

    /// Create a polynomial from an unordered list of coefficients and flattened exponents.
    pub fn from_coefficient_list(
        mut coefficients: Vec<F::Element>,
        exponents: Vec<E>,
        vars: Arc<Vec<PolyVariable>>,
        ring: &F,
    ) -> Self {
        let nterms = coefficients.len();
        let nvars = exponents.len() / coefficients.len();
        let mut indices = (0..nterms).collect::<Vec<_>>();
        indices.sort_unstable_by(|&i, &j| {
            O::cmp(
                &exponents[i * nvars..(i + 1) * nvars],
                &exponents[j * nvars..(j + 1) * nvars],
            )
        });

        let mut poly = MultivariatePolynomial::new(ring, Some(nterms), vars);

        for i in indices {
            poly.append_monomial_back(
                std::mem::replace(&mut coefficients[i], ring.zero()),
                &exponents[i * nvars..(i + 1) * nvars],
            );
        }

        poly
    }
}

impl<F: Ring, E: PositiveExponent> MultivariatePolynomial<F, E, LexOrder> {
    /// Evaluate a polynomial whose only active variable is `variable` with Horner's method.
    pub(crate) fn evaluate_univariate_horner(
        &self,
        variable: usize,
        value: &F::Element,
    ) -> F::Element {
        debug_assert!(variable < self.nvars());
        debug_assert!(self.exponents_iter().all(|exponents| {
            exponents
                .iter()
                .enumerate()
                .all(|(index, exponent)| index == variable || exponent.is_zero())
        }));

        let Some(last_term) = self.nterms().checked_sub(1) else {
            return self.ring().zero();
        };

        let ring = self.ring();
        let mut result = self.coefficients[last_term].clone();
        let mut previous_exponent = self.exponents(last_term)[variable];
        for term in (0..last_term).rev() {
            let exponent = self.exponents(term)[variable];
            let gap = (previous_exponent - exponent).to_u32() as u64;
            if gap == 1 {
                ring.mul_assign(&mut result, value);
            } else if gap > 1 {
                ring.mul_assign(&mut result, &ring.pow(value, gap));
            }
            ring.add_assign(&mut result, &self.coefficients[term]);
            previous_exponent = exponent;
        }

        let trailing_exponent = previous_exponent.to_u32() as u64;
        if trailing_exponent == 1 {
            ring.mul_assign(&mut result, value);
        } else if trailing_exponent > 1 {
            ring.mul_assign(&mut result, &ring.pow(value, trailing_exponent));
        }

        result
    }

    /// Remove all non-occurring variables from the polynomial.
    pub fn condense(&mut self) {
        if self.nvars() == 0 {
            return;
        }

        let degrees: Vec<_> = (0..self.nvars())
            .filter(|i| self.degree(*i) > E::zero())
            .collect();

        let mut new_exponents = vec![E::zero(); self.nterms() * degrees.len()];

        if degrees.is_empty() {
            self.exponents = new_exponents;
            self.set_variables(Arc::new(vec![]));
            return;
        }

        for (d, e) in new_exponents
            .chunks_mut(degrees.len())
            .zip(self.exponents_iter())
        {
            for (dr, s) in d.iter_mut().zip(&degrees) {
                *dr = e[*s];
            }
        }

        self.exponents = new_exponents;
        self.set_variables(Arc::new(
            degrees
                .into_iter()
                .map(|x| self.variables()[x].clone())
                .collect(),
        ));
    }

    /// Replace a variable `n` in the polynomial by an element from
    /// the ring `v`.
    pub fn replace(&self, n: usize, v: &F::Element) -> MultivariatePolynomial<F, E, LexOrder> {
        if (n + 1..self.nvars()).all(|i| self.degree(i) == E::zero()) {
            return self.replace_last(n, v);
        }

        let mut coefficients = self.coefficients.clone();
        let mut exponents = self.exponents.clone();

        // TODO: cache power taking?
        for (coefficient, exponent) in coefficients
            .iter_mut()
            .zip(exponents.chunks_mut(self.nvars()))
        {
            if exponent[n] == E::zero() {
                continue;
            }

            self.ring().mul_assign(
                coefficient,
                &self.ring().pow(v, exponent[n].to_i32() as u64),
            );
            exponent[n] = E::zero();
        }

        Self::from_coefficient_list(
            coefficients,
            exponents,
            self.variables().clone(),
            &self.ring(),
        )
    }

    /// Replace the last active variable `n` in the polynomial by the ring element `v`.
    /// Variables after `n` must be absent from the polynomial.
    pub fn replace_last(&self, n: usize, v: &F::Element) -> MultivariatePolynomial<F, E, LexOrder> {
        if self.nvars() == 1 {
            debug_assert_eq!(n, 0);
            return self.constant(self.evaluate_univariate_horner(0, v));
        }

        const MAX_EXP_BUF: usize = 100000;
        debug_assert!((n + 1..self.nvars()).all(|variable| self.degree(variable).is_zero()));
        let mut res = self.zero_with_capacity(self.nterms());
        let nvars = self.nvars();
        let cache_size = (self.degree(n).to_u32() as usize + 1).min(MAX_EXP_BUF);
        let mut power_cache = vec![self.ring().zero(); cache_size];

        // Lexicographic order makes terms with the same exponents before `n` contiguous when all
        // later variables are absent. Evaluate each row into one output coefficient.
        let mut row_start = 0;
        while row_start < self.nterms() {
            let row_exponents = self.exponents(row_start);
            let mut row_end = row_start + 1;
            while row_end < self.nterms() && self.exponents(row_end)[..n] == row_exponents[..n] {
                row_end += 1;
            }

            let mut coefficient = self.ring().zero();
            for term_index in row_start..row_end {
                let exponent = self.exponents(term_index)[n].to_u32() as usize;
                if exponent == 0 {
                    self.ring()
                        .add_assign(&mut coefficient, &self.coefficients[term_index]);
                } else if exponent < cache_size {
                    if self.ring().is_zero(&power_cache[exponent]) {
                        power_cache[exponent] = self.ring().pow(v, exponent as u64);
                    }
                    self.ring().add_mul_assign(
                        &mut coefficient,
                        &self.coefficients[term_index],
                        &power_cache[exponent],
                    );
                } else {
                    let power = self.ring().pow(v, exponent as u64);
                    self.ring().add_mul_assign(
                        &mut coefficient,
                        &self.coefficients[term_index],
                        &power,
                    );
                }
            }

            if !self.ring().is_zero(&coefficient) {
                res.coefficients.push(coefficient);
                res.exponents.extend_from_slice(row_exponents);
                let output_exponent = res.exponents.len() - nvars + n;
                res.exponents[output_exponent] = E::zero();
            }

            row_start = row_end;
        }

        res
    }

    pub fn evaluate<T: FloatLike, M: Fn(&F::Element) -> T>(&self, map_coeff: M, point: &[T]) -> T {
        let mut res = map_coeff(&self.ring().zero());

        for t in self {
            let mut c = map_coeff(t.coefficient);

            for (i, v) in point.iter().zip(t.exponents) {
                if v != &E::zero() {
                    c *= i.pow(v.to_u32() as u64);
                }
            }

            res += c;
        }

        res
    }

    /// Evaluate the polynomial at the given point, mapping coefficients to the ring `U`.
    pub fn evaluate_with_coeff_map<U: Ring, T: Fn(&F::Element) -> U::Element>(
        &self,
        map_coeff: T,
        point: &[U::Element],
        ring: &U,
    ) -> U::Element {
        let mut res = map_coeff(&self.ring().zero());
        assert_eq!(point.len(), self.nvars());

        for t in self {
            let mut c = map_coeff(t.coefficient);

            for (i, v) in point.iter().zip(t.exponents) {
                if v != &E::zero() {
                    ring.mul_assign(&mut c, &ring.pow(i, v.to_u32() as u64));
                }
            }

            ring.add_assign(&mut res, &c);
        }

        res
    }

    /// Replace all variables in the polynomial by an element from
    /// the ring `v`.
    pub fn replace_all(&self, r: &[F::Element]) -> F::Element {
        let mut res = self.ring().zero();

        // TODO: cache power taking?
        for t in self {
            let mut c = t.coefficient.clone();

            for (i, v) in r.iter().zip(t.exponents) {
                if v != &E::zero() {
                    self.ring()
                        .mul_assign(&mut c, &self.ring().pow(i, v.to_u32() as u64));
                }
            }

            self.ring().add_assign(&mut res, &c);
        }

        res
    }

    /// Replace a variable `n` in the polynomial by a polynomial `v`.
    pub fn replace_with_poly(&self, n: usize, v: &Self) -> Self {
        assert_eq!(self.variables(), v.variables());

        if v.is_constant() {
            return self.replace(n, &v.lcoeff());
        }

        let mut res = self.zero_with_capacity(self.nterms());
        let mut exp = vec![E::zero(); self.nvars()];
        for t in self {
            if t.exponents[n] == E::zero() {
                res.append_monomial(t.coefficient.clone(), &t.exponents[..self.nvars()]);
                continue;
            }

            exp.copy_from_slice(t.exponents);
            exp[n] = E::zero();

            // TODO: cache v^e
            res = res
                + (&v.pow(t.exponents[n].to_i32() as usize)
                    * &self.monomial(t.coefficient.clone(), exp.clone()));
        }
        res
    }

    /// Replace all variables except `v` in the polynomial by elements from
    /// the ring.
    pub fn replace_except(
        &self,
        v: usize,
        r: &[(usize, F::Element)],
        cache: &mut [Vec<F::Element>],
    ) -> MultivariatePolynomial<F, E, LexOrder> {
        let mut tm: HashMap<E, F::Element> = HashMap::new();

        for t in self {
            let mut c = t.coefficient.clone();
            for (n, vv) in r {
                let p = t.exponents[*n].to_i32() as usize;
                if p > 0 {
                    if p < cache[*n].len() {
                        if self.ring().is_zero(&cache[*n][p]) {
                            cache[*n][p] = self.ring().pow(vv, p as u64);
                        }

                        self.ring().mul_assign(&mut c, &cache[*n][p]);
                    } else {
                        self.ring()
                            .mul_assign(&mut c, &self.ring().pow(vv, p as u64));
                    }
                }
            }

            tm.entry(t.exponents[v])
                .and_modify(|e| self.ring().add_assign(e, &c))
                .or_insert(c);
        }

        let mut res = self.zero();
        let mut e = vec![E::zero(); self.nvars()];
        for (k, c) in tm {
            e[v] = k;
            res.append_monomial(c, &e);
            e[v] = E::zero();
        }

        res
    }

    /// Shift a variable `var` to `var+shift`.
    pub fn shift_var(&self, var: usize, shift: &F::Element) -> Self {
        let d = self.degree(var).to_i32() as usize;

        let y_poly = self.to_univariate_polynomial_list(var);

        let mut v = vec![self.zero(); d + 1];
        for (x_poly, p) in y_poly {
            v[p.to_i32() as usize] = x_poly;
        }

        for k in 0..d {
            for j in (k..d).rev() {
                v[j] = &v[j] + &v[j + 1].clone().mul_coeff(shift.clone());
            }
        }

        let mut shifted_coefficients = Vec::with_capacity(v.len());
        for (i, mut v) in v.into_iter().enumerate() {
            for x in v.exponents.chunks_mut(self.nvars()) {
                x[var] = E::from_i32(i as i32);
            }
            shifted_coefficients.push(v);
        }

        self.merge_shifted_univariate_coefficients(shifted_coefficients)
    }

    /// Merge sorted coefficient polynomials after assigning each one a distinct
    /// exponent in the shifted variable.
    ///
    /// The distinct exponent makes their monomial supports disjoint. A heap tracks
    /// the next monomial of each coefficient polynomial so every term is appended
    /// once without repeatedly scanning all coefficient degrees.
    fn merge_shifted_univariate_coefficients(&self, mut coefficients: Vec<Self>) -> Self {
        let capacity = coefficients.iter().map(|p| p.nterms()).sum();
        let mut poly = self.zero_with_capacity(capacity);
        let mut indices = vec![0usize; coefficients.len()];
        let mut heap: SmallVec<[usize; 32]> = SmallVec::with_capacity(coefficients.len());

        for (coefficient_index, coefficient) in coefficients.iter().enumerate() {
            if coefficient.is_zero() {
                continue;
            }

            heap.push(coefficient_index);
            let mut child = heap.len() - 1;
            while child > 0 {
                let parent = (child - 1) / 2;
                let child_poly = heap[child];
                let parent_poly = heap[parent];
                if !LexOrder::cmp(
                    coefficients[child_poly].exponents(indices[child_poly]),
                    coefficients[parent_poly].exponents(indices[parent_poly]),
                )
                .is_lt()
                {
                    break;
                }
                heap.swap(child, parent);
                child = parent;
            }
        }

        while let Some(&next_poly) = heap.first() {
            let next_term = indices[next_poly];
            let coefficient = std::mem::replace(
                &mut coefficients[next_poly].coefficients[next_term],
                self.ring().zero(),
            );
            let exponents = coefficients[next_poly].exponents(next_term);
            debug_assert!(
                poly.is_zero() || LexOrder::cmp(poly.last_exponents(), exponents).is_lt(),
                "shifted coefficient streams must have disjoint sorted supports"
            );
            poly.coefficients.push(coefficient);
            poly.exponents.extend_from_slice(exponents);
            indices[next_poly] += 1;

            if indices[next_poly] == coefficients[next_poly].nterms() {
                heap.swap_remove(0);
            }

            if heap.is_empty() {
                continue;
            }

            let mut parent = 0;
            loop {
                let left = 2 * parent + 1;
                if left >= heap.len() {
                    break;
                }
                let right = left + 1;
                let mut child = left;
                if right < heap.len() {
                    let left_poly = heap[left];
                    let right_poly = heap[right];
                    if LexOrder::cmp(
                        coefficients[right_poly].exponents(indices[right_poly]),
                        coefficients[left_poly].exponents(indices[left_poly]),
                    )
                    .is_lt()
                    {
                        child = right;
                    }
                }

                let parent_poly = heap[parent];
                let child_poly = heap[child];
                if !LexOrder::cmp(
                    coefficients[child_poly].exponents(indices[child_poly]),
                    coefficients[parent_poly].exponents(indices[parent_poly]),
                )
                .is_lt()
                {
                    break;
                }
                heap.swap(parent, child);
                parent = child;
            }
        }

        poly
    }

    /// Compute the inverse of the univariate polynomial in `var` up until `pow` using Newton's method.
    pub fn inverse_univariate(&self, var: usize, pow: E) -> Self {
        let mut g = self.constant(
            self.ring()
                .try_div(&self.ring().one(), &self.get_constant())
                .unwrap(),
        );
        let mut exp = E::one();
        while exp < pow {
            exp = exp * E::from_u32(2);
            let h = g.clone().mul_coeff(self.ring().nth(2.into()))
                - self * &(&g * &g).mod_var(var, exp);
            g = h.mod_var(var, exp);
        }
        g
    }
}

impl<F: Ring, E: Exponent> MultivariatePolynomial<F, E, LexOrder> {
    /// Check if all exponents are positive.
    pub fn is_polynomial(&self) -> bool {
        self.is_zero() || self.exponents.iter().all(|e| *e >= E::zero())
    }

    /// Get the leading coefficient under a given variable ordering.
    /// This operation is O(n) if the variables are out of order.
    pub fn lcoeff_varorder(&self, vars: &[usize]) -> F::Element {
        if vars.windows(2).all(|s| s[0] < s[1]) {
            return self.lcoeff();
        }

        let mut highest = vec![E::zero(); self.nvars()];
        let mut highestc = &self.ring().zero();

        'nextmon: for m in self.into_iter() {
            let mut more = false;
            for &v in vars {
                if more {
                    highest[v] = m.exponents[v];
                } else {
                    match m.exponents[v].cmp(&highest[v]) {
                        Ordering::Less => {
                            continue 'nextmon;
                        }
                        Ordering::Greater => {
                            highest[v] = m.exponents[v];
                            more = true;
                        }
                        Ordering::Equal => {}
                    }
                }
            }
            highestc = m.coefficient;
        }
        debug_assert!(!self.ring().is_zero(highestc));
        highestc.clone()
    }

    /// Get the leading coefficient of a multivariate polynomial viewed as a
    /// univariate polynomial in `x`.
    pub fn univariate_lcoeff(&self, x: usize) -> MultivariatePolynomial<F, E, LexOrder> {
        let d = self.degree(x);
        let mut lcoeff = self.zero();

        if self.coefficients.is_empty() {
            return lcoeff;
        }

        if d == E::zero() {
            return self.clone();
        }

        let mut e = vec![E::zero(); self.nvars()];
        for t in self {
            if t.exponents[x] == d {
                e.copy_from_slice(t.exponents);
                e[x] = E::zero();
                lcoeff.append_monomial(t.coefficient.clone(), &e);
            }
        }

        lcoeff
    }

    /// Count the number of terms that have the maximum degree in the given variable.
    pub fn terms_with_max_degree(&self, x: usize) -> usize {
        let mut max_degree = self.exponents[x];
        let mut count = 0;
        for t in 1..self.nterms() {
            let e = self.exponents[t * self.nvars() + x];
            if e == max_degree {
                count += 1;
            } else if e > max_degree {
                max_degree = e;
                count = 1;
            }
        }
        count
    }

    /// Get the bivariate degree of a multivariate polynomial viewed as a bivariate polynomial in the first two variables.
    pub fn bivariate_deg(&self) -> (E, E) {
        if self.is_zero() {
            return (E::zero(), E::zero());
        }

        let mut d = self.exponents[0];
        let mut d1 = self.exponents[1];
        let nvars = self.nvars();
        assert!(self.exponents.len() == self.nterms() * nvars);
        for e in 0..self.nterms() {
            if self.exponents[e * nvars] > d {
                d = self.exponents[e * nvars];
                d1 = self.exponents[e * nvars + 1];
            } else if self.exponents[e * nvars] == d && self.exponents[e * nvars + 1] > d1 {
                d1 = self.exponents[e * nvars + 1];
            }
        }
        (d, d1)
    }

    /// Get the leading coefficient of a multivariate polynomial viewed as a bivariate polynomial in the first two variables.
    pub fn bivariate_lcoeff(&self) -> MultivariatePolynomial<F, E, LexOrder> {
        let mut lcoeff = self.zero();

        if self.coefficients.is_empty() {
            return lcoeff;
        }

        let (d, d1) = self.bivariate_deg();

        let mut e = vec![E::zero(); self.nvars()];
        for t in self {
            if t.exponents[0] == d && t.exponents[1] == d1 {
                e.copy_from_slice(t.exponents);
                e[0] = E::zero();
                e[1] = E::zero();
                lcoeff.append_monomial(t.coefficient.clone(), &e);
            }
        }

        lcoeff
    }

    /// Get the leading coefficient viewed as a polynomial
    /// in all variables except the last variable `n`.
    pub fn lcoeff_last(&self, n: usize) -> MultivariatePolynomial<F, E, LexOrder> {
        if self.is_zero() {
            return self.clone();
        }
        // the last variable should have the least sorting priority,
        // so the last term should still be the lcoeff
        let last = self.last_exponents();

        let mut res = self.zero();
        let mut e: SmallVec<[E; INLINED_EXPONENTS]> = smallvec![E::zero(); self.nvars()];

        for t in (0..self.nterms()).rev() {
            if (0..self.nvars() - 1).all(|i| self.exponents(t)[i] == last[i] || i == n) {
                e[n] = self.exponents(t)[n];
                res.append_monomial(self.coefficients[t].clone(), &e);
                e[n] = E::zero();
            } else {
                break;
            }
        }

        res
    }

    /// Get the leading coefficient viewed as a polynomial
    /// in all variables with order as described in `vars` except the last variable in `vars`.
    /// This operation is O(n) if the variables are out of order.
    pub fn lcoeff_last_varorder(&self, vars: &[usize]) -> MultivariatePolynomial<F, E, LexOrder> {
        if self.is_zero() {
            return self.clone();
        }

        if vars.windows(2).all(|s| s[0] < s[1]) {
            return self.lcoeff_last(*vars.last().unwrap());
        }

        let (vars, lastvar) = vars.split_at(vars.len() - 1);

        let mut highest = vec![E::zero(); self.nvars()];
        let mut indices = Vec::with_capacity(10);

        'nextmon: for (i, m) in self.into_iter().enumerate() {
            let mut more = false;
            for &v in vars {
                if more {
                    highest[v] = m.exponents[v];
                } else {
                    match m.exponents[v].cmp(&highest[v]) {
                        Ordering::Less => {
                            continue 'nextmon;
                        }
                        Ordering::Greater => {
                            highest[v] = m.exponents[v];
                            indices.clear();
                            more = true;
                        }
                        Ordering::Equal => {}
                    }
                }
            }
            indices.push(i);
        }

        let mut res = self.zero();
        let mut e = vec![E::zero(); self.nvars()];
        for i in indices {
            e[lastvar[0]] = self.exponents(i)[lastvar[0]];
            res.append_monomial(self.coefficients[i].clone(), &e);
            e[lastvar[0]] = E::zero();
        }
        res
    }

    /// Change the order of the variables in the polynomial, using `order`.
    /// The map can also be reversed, by setting `inverse` to `true`.
    pub(crate) fn rearrange_impl(
        &self,
        order: &[usize],
        inverse: bool,
        update_variables: bool,
    ) -> MultivariatePolynomial<F, E, LexOrder> {
        let mut new_exp = vec![E::zero(); self.nterms() * self.nvars()];
        for (e, er) in new_exp.chunks_mut(self.nvars()).zip(self.exponents_iter()) {
            for x in 0..order.len() {
                if !inverse {
                    e[x] = er[order[x]];
                } else {
                    e[order[x]] = er[x];
                }
            }
        }

        let mut indices: Vec<usize> = (0..self.nterms()).collect();
        indices.sort_unstable_by_key(|&i| &new_exp[i * self.nvars()..(i + 1) * self.nvars()]);

        let mut res = self.zero_with_capacity(self.nterms());

        for i in indices {
            res.append_monomial(
                self.coefficients[i].clone(),
                &new_exp[i * self.nvars()..(i + 1) * self.nvars()],
            );
        }

        if update_variables {
            let mut vm = self.variables().as_ref().clone();
            for x in 0..order.len() {
                if !inverse {
                    vm[x] = self.variables()[order[x]].clone();
                } else {
                    vm[order[x]] = self.variables()[x].clone();
                }
            }

            res.set_variables(Arc::new(vm));
        }

        res
    }

    /// Change the order of the variables in the polynomial, using `order`.
    /// The map can also be reversed, by setting `inverse` to `true`.
    pub fn rearrange(
        &self,
        order: &[usize],
        inverse: bool,
    ) -> MultivariatePolynomial<F, E, LexOrder> {
        self.rearrange_impl(order, inverse, true)
    }

    /// Change the order of the variables in the polynomial, using `order`.
    /// The order may contain `None`, to signal unmapped indices. This operation
    /// allows the polynomial to grow in size.
    pub fn rearrange_with_growth(
        &self,
        order: &[PolyVariable],
    ) -> Result<MultivariatePolynomial<F, E, LexOrder>, String> {
        let new_order: Vec<_> = order
            .iter()
            .map(|x| self.variables().iter().position(|e| e == x))
            .collect();

        for (i, v) in self.variables().iter().enumerate() {
            if !new_order.contains(&Some(i)) && self.contains(i) {
                return Err(format!("Variable {v} is not in the new order"));
            }
        }

        let mut new_exp = vec![E::zero(); self.nterms() * order.len()];
        for (e, er) in new_exp.chunks_mut(order.len()).zip(self.exponents_iter()) {
            for x in 0..new_order.len() {
                if let Some(v) = new_order[x] {
                    e[x] = er[v];
                }
            }
        }

        let mut indices: Vec<usize> = (0..self.nterms()).collect();
        indices.sort_unstable_by_key(|&i| &new_exp[i * order.len()..(i + 1) * order.len()]);

        let mut res = MultivariatePolynomial::new(
            self.ring(),
            self.nterms().into(),
            Arc::new(order.to_vec()),
        );

        for i in indices {
            res.append_monomial(
                self.coefficients[i].clone(),
                &new_exp[i * order.len()..(i + 1) * order.len()],
            );
        }

        Ok(res)
    }

    /// Compute `self^pow`.
    pub fn pow(&self, pow: usize) -> Self {
        if pow == 0 {
            return self.one();
        }
        if pow == 1 {
            return self.clone();
        }

        if self.is_constant() {
            return self.constant(self.ring().pow(&self.lcoeff(), pow as u64));
        }

        if self.coefficients.len() == 1 {
            return self.monomial(
                self.ring().pow(&self.coefficients[0], pow as u64),
                self.exponents
                    .iter()
                    .map(|x| *x * E::from_i32(pow as i32))
                    .collect(),
            );
        }

        // heuristic for when to use heap_pow
        if pow > 10 || (0..self.nvars()).all(|x| self.degree(x) <= E::from_i32(2)) {
            // if the characteristic is non-zero, a division by the exponent in the heap_pow algorithm
            // may cause a division by 0
            if self.ring().characteristic() == 0
                || self.nvars() == 1
                    && self.degree(0).to_i32() as usize + 1 < self.ring().characteristic()
            {
                return self.heap_pow(pow);
            }
        }

        // perform repeated multiplication instead of binary exponentiation, as
        // the latter is often much slower for sparse polynomials
        let mut res = self * self;
        for _ in 2..pow {
            res = &res * self;
        }
        res
    }

    /// View the polynomial as a polynomial in the given variables, with all
    /// remaining variables in the coefficient ring.
    pub fn to_polynomial_in(
        &self,
        vars: &[usize],
    ) -> MultivariatePolynomial<PolynomialRing<F, E>, E, LexOrder> {
        let split = self.to_multivariate_polynomial_list(vars, true);

        let mut exponents = vec![];
        let mut coefficients = vec![];

        for (e, c) in split {
            coefficients.push(c);
            exponents.extend(vars.iter().map(|i| e[*i]));
        }

        let vars = Arc::new(
            vars.iter()
                .map(|i| self.variables()[*i].clone())
                .collect::<Vec<_>>(),
        );

        let ring = PolynomialRing::new(self.ring().clone());

        MultivariatePolynomial::from_coefficient_list(coefficients, exponents, vars, &ring)
    }

    pub fn to_univariate(&self, var: usize) -> UnivariatePolynomial<PolynomialRing<F, E>> {
        let c = self.to_univariate_polynomial_list(var);

        let mut p = UnivariatePolynomial::new(
            &PolynomialRing::from_poly(self),
            None,
            Arc::new(self.variables()[var].clone()),
        );

        if c.is_empty() {
            return p;
        }

        p.coefficients = vec![self.zero(); c.last().unwrap().1.to_i32() as usize + 1];
        for (q, e) in c {
            if e < E::zero() {
                panic!("Negative exponent in univariate conversion");
            }

            p.coefficients[e.to_i32() as usize] = q;
        }

        p
    }

    pub fn to_univariate_from_univariate(&self, var: usize) -> UnivariatePolynomial<F> {
        let mut p =
            UnivariatePolynomial::new(self.ring(), None, Arc::new(self.variables()[var].clone()));

        if self.is_zero() {
            return p;
        }

        p.coefficients = vec![p.ring.zero(); self.degree(var).to_i32() as usize + 1];
        for (q, e) in self.coefficients.iter().zip(self.exponents_iter()) {
            if e[var] < E::zero() {
                panic!("Negative exponent in univariate conversion");
            }

            p.coefficients[e[var].to_i32() as usize] = q.clone();
        }

        p
    }

    /// Create a univariate polynomial coefficient list out of a multivariate polynomial.
    /// The output is sorted in the degree.
    pub fn to_univariate_polynomial_list(
        &self,
        x: usize,
    ) -> Vec<(MultivariatePolynomial<F, E, LexOrder>, E)> {
        if self.coefficients.is_empty() {
            return vec![];
        }

        let first_degree = self.exponents(0)[x].to_i32();
        let (mut min_degree, mut max_degree) = (first_degree, first_degree);
        for exponents in self.exponents_iter().skip(1) {
            let degree = exponents[x].to_i32();
            min_degree = min_degree.min(degree);
            max_degree = max_degree.max(degree);
        }

        let degree_span = (i64::from(max_degree) - i64::from(min_degree) + 1)
            .try_into()
            .expect("polynomial degree range does not fit in memory");
        let mut coefficients: Vec<Option<Self>> =
            std::iter::repeat_with(|| None).take(degree_span).collect();
        let mut coefficient_exponents: SmallVec<[E; INLINED_EXPONENTS]> =
            smallvec![E::zero(); self.nvars()];

        for (coefficient, exponents) in self.coefficients.iter().zip(self.exponents_iter()) {
            let index = usize::try_from(i64::from(exponents[x].to_i32()) - i64::from(min_degree))
                .expect("coefficient degree must be in the polynomial degree range");
            coefficient_exponents.copy_from_slice(exponents);
            coefficient_exponents[x] = E::zero();

            // Terms in one degree bucket form a sorted subsequence of the input.
            // Clearing their equal x coordinate therefore preserves strict Lex order.
            coefficients[index]
                .get_or_insert_with(|| self.zero())
                .append_monomial_back(coefficient.clone(), &coefficient_exponents);
        }

        coefficients
            .into_iter()
            .enumerate()
            .filter_map(|(offset, coefficient)| {
                coefficient.map(|coefficient| {
                    let degree = i64::from(min_degree) + offset as i64;
                    let degree = i32::try_from(degree)
                        .expect("coefficient degree must remain in the exponent range");
                    (coefficient, E::from_i32(degree))
                })
            })
            .collect()
    }

    /// Split the polynomial as a polynomial in `xs` if include is true,
    /// else excluding `xs`.
    pub fn to_multivariate_polynomial_list(
        &self,
        xs: &[usize],
        include: bool,
    ) -> HashMap<SmallVec<[E; INLINED_EXPONENTS]>, MultivariatePolynomial<F, E, LexOrder>> {
        if self.coefficients.is_empty() {
            return HashMap::new();
        }

        let mut tm: HashMap<
            SmallVec<[E; INLINED_EXPONENTS]>,
            MultivariatePolynomial<F, E, LexOrder>,
        > = HashMap::new();
        let mut e_not_in_xs = smallvec![E::zero(); self.nvars()];
        let mut e_in_xs = smallvec![E::zero(); self.nvars()];
        for t in self {
            for (i, ee) in t.exponents.iter().enumerate() {
                e_not_in_xs[i] = *ee;
                e_in_xs[i] = E::zero();
            }

            for x in xs {
                e_in_xs[*x] = e_not_in_xs[*x];
                e_not_in_xs[*x] = E::zero();
            }

            if include {
                tm.entry(e_in_xs.clone())
                    .and_modify(|x| x.append_monomial(t.coefficient.clone(), &e_not_in_xs))
                    .or_insert_with(|| {
                        MultivariatePolynomial::monomial(
                            self,
                            t.coefficient.clone(),
                            e_not_in_xs.to_vec(),
                        )
                    });
            } else {
                tm.entry(e_not_in_xs.clone())
                    .and_modify(|x| x.append_monomial(t.coefficient.clone(), &e_in_xs))
                    .or_insert_with(|| {
                        MultivariatePolynomial::monomial(
                            self,
                            t.coefficient.clone(),
                            e_in_xs.to_vec(),
                        )
                    });
            }
        }

        tm
    }

    pub(crate) fn mul_univariate_dense(&self, rhs: &Self, max_pow: Option<usize>) -> Self {
        if self.is_constant() {
            if let Some(m) = max_pow
                && let Some(var) = rhs.last_exponents().iter().position(|e| *e != E::zero())
                && rhs.degree(var).to_i32() > m as i32
            {
                return rhs
                    .mod_var(var, E::from_i32(m as i32 + 1))
                    .mul_coeff(self.lcoeff());
            }
            return rhs.clone().mul_coeff(self.lcoeff());
        }

        if rhs.is_constant() {
            if let Some(m) = max_pow
                && let Some(var) = self.last_exponents().iter().position(|e| *e != E::zero())
                && self.degree(var).to_i32() > m as i32
            {
                return self
                    .mod_var(var, E::from_i32(m as i32 + 1))
                    .mul_coeff(rhs.lcoeff());
            }
            return self.clone().mul_coeff(rhs.lcoeff());
        }

        let var = self
            .last_exponents()
            .iter()
            .position(|e| *e != E::zero())
            .unwrap();

        let d1 = self.degree(var);
        let d2 = rhs.degree(var);
        let mut max = (d1.to_i32() + d2.to_i32()) as usize;
        if let Some(m) = max_pow {
            max = max.min(m);
        }

        let mut coeffs = vec![self.ring().zero(); max + 1];

        for x in self {
            for y in rhs {
                let pos = x.exponents[var].to_i32() + y.exponents[var].to_i32();
                if pos as usize > max {
                    continue;
                }

                self.ring()
                    .add_mul_assign(&mut coeffs[pos as usize], x.coefficient, y.coefficient);
            }
        }

        let mut exp = vec![E::zero(); self.nvars()];
        let mut res = self.zero_with_capacity(coeffs.len());
        for (p, c) in coeffs.into_iter().enumerate() {
            if !self.ring().is_zero(&c) {
                exp[var] = E::from_i32(p as i32);
                res.append_monomial(c, &exp);
            }
        }
        res
    }

    /// Synthetic division for univariate polynomials, where `div` is monic.
    pub(crate) fn quot_rem_univariate_monic(
        &self,
        div: &MultivariatePolynomial<F, E, LexOrder>,
    ) -> (
        MultivariatePolynomial<F, E, LexOrder>,
        MultivariatePolynomial<F, E, LexOrder>,
    ) {
        debug_assert_eq!(div.lcoeff(), self.ring().one());
        if self.is_zero() {
            return (self.clone(), self.clone());
        }

        let Some(var) = div
            .last_exponents()
            .iter()
            .position(|exponent| !exponent.is_zero())
        else {
            // The only monic constant polynomial is one.
            return (self.clone(), self.zero());
        };

        debug_assert_eq!(self.get_vars_ref(), div.get_vars_ref());
        debug_assert!(self.is_polynomial() && div.is_polynomial());
        debug_assert!(
            self.exponents_iter()
                .chain(div.exponents_iter())
                .all(|exponents| exponents
                    .iter()
                    .enumerate()
                    .all(|(index, exponent)| index == var || exponent.is_zero()))
        );

        let div_degree = div.degree(var).to_i32() as usize;
        let dividend_degree = self.degree(var).to_i32() as usize;
        if dividend_degree < div_degree {
            return (self.zero(), self.clone());
        }

        let mut coefficients = vec![self.ring().zero(); dividend_degree + 1];
        for term in self {
            coefficients[term.exponents[var].to_i32() as usize] = term.coefficient.clone();
        }

        let quotient_degree = dividend_degree - div_degree;
        let mut quotient_coefficients = vec![self.ring().zero(); quotient_degree + 1];
        for degree in (div_degree..=dividend_degree).rev() {
            let coefficient = std::mem::replace(&mut coefficients[degree], self.ring().zero());
            if self.ring().is_zero(&coefficient) {
                continue;
            }

            for term in div {
                let term_degree = term.exponents[var].to_i32() as usize;
                if term_degree == div_degree {
                    continue;
                }

                self.ring().sub_mul_assign(
                    &mut coefficients[degree - div_degree + term_degree],
                    term.coefficient,
                    &coefficient,
                );
            }
            quotient_coefficients[degree - div_degree] = coefficient;
        }

        let mut exponent = vec![E::zero(); self.nvars()];
        let mut q = self.zero_with_capacity(quotient_coefficients.len());
        for (degree, coefficient) in quotient_coefficients.into_iter().enumerate() {
            if !self.ring().is_zero(&coefficient) {
                exponent[var] = E::from_i32(degree as i32);
                q.append_monomial(coefficient, &exponent);
            }
        }

        let mut r = self.zero_with_capacity(div_degree);
        for (degree, coefficient) in coefficients.into_iter().take(div_degree).enumerate() {
            if !self.ring().is_zero(&coefficient) {
                exponent[var] = E::from_i32(degree as i32);
                r.append_monomial(coefficient, &exponent);
            }
        }

        #[cfg(test)]
        {
            if !(&q * div + r.clone() - self.clone()).is_zero() {
                panic!("Division failed: ({self})/({div}): q={q}, r={r}");
            }
        }

        (q, r)
    }

    fn mul_dense(
        &self,
        rhs: &MultivariatePolynomial<F, E, LexOrder>,
    ) -> Option<MultivariatePolynomial<F, E, LexOrder>> {
        if !self.is_polynomial() || !rhs.is_polynomial() {
            return None;
        }

        let max_degs_rev = (0..self.nvars())
            .rev()
            .map(|i| 1 + self.degree(i).to_i32() as usize + rhs.degree(i).to_i32() as usize)
            .collect::<Vec<_>>();

        let univariate = max_degs_rev.iter().filter(|x| **x > 1).count() == 1;
        let use_generic_univariate_dense = max_degs_rev.iter().sum::<usize>() < 10000;

        let mut total: usize = 1;
        for x in &max_degs_rev {
            if *x > MAX_DENSE_MUL_BUFFER_SIZE {
                return None;
            }

            if let Some(r) = total.checked_mul(*x) {
                total = r;
            } else {
                return None;
            }
        }

        if total > MAX_DENSE_MUL_BUFFER_SIZE {
            return None;
        }

        if !univariate && !mixed_radix_dense_work_is_bounded(total, self.nterms(), rhs.nterms()) {
            return None;
        }

        #[inline(always)]
        fn to_uni_var<E: Exponent>(s: &[E], max_degs_rev: &[usize]) -> u32 {
            let mut shift = 1;
            let mut res = s.last().unwrap().to_i32() as u32;
            for (ee, &x) in s.iter().rev().skip(1).zip(max_degs_rev) {
                shift *= x as u32;
                res += ee.to_i32() as u32 * shift;
            }
            res
        }

        #[inline(always)]
        fn from_uni_var<E: Exponent>(mut p: u32, max_degs_rev: &[usize], exp: &mut [E]) {
            for (ee, &x) in exp.iter_mut().rev().zip(max_degs_rev) {
                *ee = E::from_i32((p % x as u32) as i32);
                p /= x as u32;
            }
        }

        #[inline(always)]
        fn advance_uni_var<E: Exponent>(mut delta: u32, max_degs_rev: &[usize], exp: &mut [E]) {
            for (ee, &radix) in exp.iter_mut().rev().zip(max_degs_rev) {
                if delta == 0 {
                    break;
                }

                let value = ee.to_i32() as u32 + delta;
                if value < radix as u32 {
                    *ee = E::from_i32(value as i32);
                    delta = 0;
                    break;
                }

                *ee = E::from_i32((value % radix as u32) as i32);
                delta = value / radix as u32;
            }
            debug_assert_eq!(delta, 0);
        }

        let mut uni_exp_self = vec![0; self.coefficients.len()];
        for (es, s) in &mut uni_exp_self.iter_mut().zip(self.exponents_iter()) {
            *es = to_uni_var(s, &max_degs_rev);
        }

        let mut uni_exp_rhs = vec![0; rhs.coefficients.len()];
        for (es, s) in &mut uni_exp_rhs.iter_mut().zip(rhs.exponents_iter()) {
            *es = to_uni_var(s, &max_degs_rev);
        }

        if let Some(coefficients) = self.ring().kernels().polynomial().and_then(|kernels| {
            kernels.try_dense_mul(DensePolynomialMulRequest {
                output_len: total,
                left_coefficients: &self.coefficients,
                left_indices: &uni_exp_self,
                right_coefficients: &rhs.coefficients,
                right_indices: &uni_exp_rhs,
            })
        }) {
            let mut exp = vec![E::zero(); self.nvars()];
            let mut result = self.zero_with_capacity(coefficients.len());
            let mut previous_position = 0;
            for (position, coefficient) in coefficients {
                debug_assert!(!self.ring().is_zero(&coefficient));
                debug_assert!(result.coefficients.is_empty() || position > previous_position);
                advance_uni_var(position - previous_position, &max_degs_rev, &mut exp);
                previous_position = position;
                result.coefficients.push(coefficient);
                result.exponents.extend_from_slice(&exp);
            }
            return Some(result);
        }

        if univariate {
            return use_generic_univariate_dense.then(|| self.mul_univariate_dense(rhs, None));
        }

        let mut exp = vec![E::zero(); self.nvars()];
        let mut r = self.zero_with_capacity(self.nterms().max(rhs.nterms()));

        // check if we need to use a dense indexing array to save memory
        if total < 1000 {
            let mut coeffs = vec![self.ring().zero(); total];

            for (c1, e1) in self.coefficients.iter().zip(&uni_exp_self) {
                for (c2, e2) in rhs.coefficients.iter().zip(&uni_exp_rhs) {
                    let pos = *e1 as usize + *e2 as usize;
                    self.ring().add_mul_assign(&mut coeffs[pos], c1, c2);
                }
            }

            for (p, c) in coeffs.into_iter().enumerate() {
                if !self.ring().is_zero(&c) {
                    from_uni_var(p as u32, &max_degs_rev, &mut exp);
                    r.append_monomial(c, &exp);
                }
            }

            Some(r)
        } else {
            let mut coeffs = Vec::with_capacity(self.nterms().max(rhs.nterms()));

            let mut coeff_index = DENSE_MUL_BUFFER.take();

            if coeff_index.len() < total {
                coeff_index.resize(total, 0u32);
            }

            for (c1, e1) in self.coefficients.iter().zip(&uni_exp_self) {
                for (c2, e2) in rhs.coefficients.iter().zip(&uni_exp_rhs) {
                    let pos = *e1 as usize + *e2 as usize;
                    if coeff_index[pos] == 0 {
                        coeffs.push(self.ring().mul(c1, c2));
                        coeff_index[pos] = coeffs.len() as u32;
                    } else {
                        self.ring().add_mul_assign(
                            &mut coeffs[coeff_index[pos] as usize - 1],
                            c1,
                            c2,
                        );
                    }
                }
            }

            for (p, c) in coeff_index[..total].iter_mut().enumerate() {
                if *c != 0 {
                    from_uni_var(p as u32, &max_degs_rev, &mut exp);
                    r.append_monomial(
                        std::mem::replace(&mut coeffs[*c as usize - 1], self.ring().zero()),
                        &exp,
                    );
                    *c = 0;
                }
            }

            DENSE_MUL_BUFFER.set(coeff_index);

            Some(r)
        }
    }

    fn heap_mul(
        &self,
        rhs: &MultivariatePolynomial<F, E, LexOrder>,
    ) -> MultivariatePolynomial<F, E, LexOrder> {
        // place the smallest polynomial first, as this is faster
        // in the heap algorithm
        if self.nterms() > rhs.nterms() {
            return rhs.heap_mul(self);
        }

        let degree_sum: Vec<_> = (0..self.nvars())
            .map(|i| self.degree(i).to_i32() as i64 + rhs.degree(i).to_i32() as i64)
            .collect();

        // use a special routine if the exponents can be packed into a u64
        let mut pack_u8 = true;
        if self.nvars() <= 8
            && self.is_polynomial()
            && rhs.is_polynomial()
            && degree_sum.iter().all(|deg| {
                if *deg > 255 {
                    pack_u8 = false;
                }

                *deg <= 255 || self.nvars() <= 4 && *deg <= 65535
            })
        {
            return self.heap_mul_packed_exp(rhs, pack_u8);
        }

        let mut monomials = Vec::with_capacity(self.nterms() * self.nvars());
        monomials.extend(
            self.exponents(0)
                .iter()
                .zip(rhs.exponents(0))
                .map(|(e1, e2)| *e1 + *e2),
        );

        let monomials = UnsafeCell::new((self.nvars(), monomials));

        /// In order to prevent allocations of the exponents, store them in a single
        /// append-only vector and use a key to index into it. For performance,
        /// we use an unsafe cell.
        #[derive(Clone, Copy)]
        struct Key<'a, E: Exponent> {
            index: usize,
            monomials: &'a UnsafeCell<(usize, Vec<E>)>,
        }

        impl<E: Exponent> PartialEq for Key<'_, E> {
            #[inline(always)]
            fn eq(&self, other: &Self) -> bool {
                unsafe {
                    let b1 = &*self.monomials.get();
                    b1.1.get_unchecked(self.index..self.index + b1.0)
                        == b1.1.get_unchecked(other.index..other.index + b1.0)
                }
            }
        }

        impl<E: Exponent> Eq for Key<'_, E> {}

        impl<E: Exponent> PartialOrd for Key<'_, E> {
            #[inline(always)]
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        impl<E: Exponent> Ord for Key<'_, E> {
            #[inline(always)]
            fn cmp(&self, other: &Self) -> Ordering {
                unsafe {
                    let b1 = &*self.monomials.get();
                    b1.1.get_unchecked(self.index..self.index + b1.0)
                        .cmp(b1.1.get_unchecked(other.index..other.index + b1.0))
                }
            }
        }

        impl<E: Exponent> std::hash::Hash for Key<'_, E> {
            #[inline(always)]
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                unsafe {
                    let b = &*self.monomials.get();
                    b.1.get_unchecked(self.index..self.index + b.0).hash(state);
                }
            }
        }

        let mut res = self.zero_with_capacity(self.heap_mul_result_capacity(rhs));

        let mut cache: HashMap<_, Vec<(usize, usize)>> = HashMap::new();
        let mut q_cache: Vec<Vec<(usize, usize)>> = vec![];

        // create a min-heap since our polynomials are sorted smallest to largest
        let mut h: BinaryHeap<Reverse<_>> = BinaryHeap::with_capacity(self.nterms());

        cache.insert(
            Key {
                index: 0,
                monomials: &monomials,
            },
            vec![(0, 0)],
        );
        h.push(Reverse(Key {
            index: 0,
            monomials: &monomials,
        }));

        // i=merged_index[j] signifies that self[i]*other[j] has been merged
        let mut merged_index = vec![0; rhs.nterms()];
        // in_heap[j] signifies that other[j] is in the heap
        let mut in_heap = vec![false; rhs.nterms()];
        in_heap[0] = true;

        while !h.is_empty() {
            let cur_mon = h.pop().unwrap();

            let mut coefficient = self.ring().zero();

            let mut q = cache.remove(&cur_mon.0).unwrap();

            for (i, j) in q.drain(..) {
                self.ring().add_mul_assign(
                    &mut coefficient,
                    &self.coefficients[i],
                    &rhs.coefficients[j],
                );

                merged_index[j] = i + 1;

                if i + 1 < self.nterms() && (j == 0 || merged_index[j - 1] > i + 1) {
                    let m = unsafe {
                        let b = &mut *monomials.get();
                        let index = b.1.len();
                        b.1.extend(
                            self.exponents(i + 1)
                                .iter()
                                .zip(rhs.exponents(j))
                                .map(|(e1, e2)| *e1 + *e2),
                        );

                        Key {
                            index,
                            monomials: &monomials,
                        }
                    };

                    if let Some(e) = cache.get_mut(&m) {
                        e.push((i + 1, j));
                    } else {
                        h.push(Reverse(m)); // only add when new
                        if let Some(mut qq) = q_cache.pop() {
                            qq.push((i + 1, j));
                            cache.insert(m, qq);
                        } else {
                            cache.insert(m, vec![(i + 1, j)]);
                        }
                    }
                } else {
                    in_heap[j] = false;
                }

                if j + 1 < rhs.nterms() && !in_heap[j + 1] {
                    let m = unsafe {
                        let b = &mut *monomials.get();
                        let index = b.1.len();
                        b.1.extend(
                            self.exponents(i)
                                .iter()
                                .zip(rhs.exponents(j + 1))
                                .map(|(e1, e2)| *e1 + *e2),
                        );

                        Key {
                            index,
                            monomials: &monomials,
                        }
                    };

                    if let Some(e) = cache.get_mut(&m) {
                        e.push((i, j + 1));
                    } else {
                        h.push(Reverse(m)); // only add when new

                        if let Some(mut qq) = q_cache.pop() {
                            qq.push((i, j + 1));
                            cache.insert(m, qq);
                        } else {
                            cache.insert(m, vec![(i, j + 1)]);
                        }
                    }

                    in_heap[j + 1] = true;
                }
            }

            q_cache.push(q);

            if !self.ring().is_zero(&coefficient) {
                res.coefficients.push(coefficient);

                unsafe {
                    let b = &*monomials.get();
                    res.exponents
                        .extend_from_slice(&b.1[cur_mon.0.index..cur_mon.0.index + b.0]);
                }
            }
        }

        res
    }

    /// Estimate a bounded result capacity for heap multiplication.
    ///
    /// Every coefficient pair contributes to at most one output term. Reserve that upper bound
    /// when the coefficient and exponent buffers together fit in one mebibyte; larger products
    /// retain the input-sized reservation and grow according to the number of terms encountered.
    fn heap_mul_result_capacity(&self, other: &MultivariatePolynomial<F, E, LexOrder>) -> usize {
        const MAX_PREALLOCATED_BYTES: usize = 1 << 20;

        let input_capacity = self.nterms().max(other.nterms());
        let bytes_per_term = mem::size_of::<F::Element>()
            .saturating_add(self.nvars().saturating_mul(mem::size_of::<E>()))
            .max(1);
        let maximum_terms = MAX_PREALLOCATED_BYTES / bytes_per_term;
        self.nterms()
            .checked_mul(other.nterms())
            .filter(|pair_products| *pair_products <= maximum_terms)
            .unwrap_or(input_capacity)
    }

    /// Heap multiplication, but with the exponents packed into a `u64`.
    /// Each exponent is limited to 65535 if there are four or fewer variables,
    /// or 255 if there are 8 or fewer variables.
    fn heap_mul_packed_exp(
        &self,
        other: &MultivariatePolynomial<F, E, LexOrder>,
        pack_u8: bool,
    ) -> MultivariatePolynomial<F, E, LexOrder> {
        if pack_u8 && let Some(result) = self.try_total_degree_dense_mul(other) {
            return result;
        }

        if pack_u8 && let Some(result) = self.try_packed_u8_row_merge_mul(other) {
            return result;
        }

        let mut res = self.zero_with_capacity(self.heap_mul_result_capacity(other));

        let pack_a: Vec<_> = if pack_u8 {
            self.exponents_iter().map(|c| E::pack(c)).collect()
        } else {
            self.exponents_iter().map(|c| E::pack_u16(c)).collect()
        };
        let pack_b: Vec<_> = if pack_u8 {
            other.exponents_iter().map(|c| E::pack(c)).collect()
        } else {
            other.exponents_iter().map(|c| E::pack_u16(c)).collect()
        };

        let mut cache: BTreeMap<u64, Vec<(usize, usize)>> = BTreeMap::new();
        let mut q_cache: Vec<Vec<(usize, usize)>> = vec![];

        // create a min-heap since our polynomials are sorted smallest to largest
        let mut h: BinaryHeap<Reverse<u64>> = BinaryHeap::with_capacity(self.nterms());

        let monom: u64 = pack_a[0] + pack_b[0];
        cache.insert(monom, vec![(0, 0)]);
        h.push(Reverse(monom));

        // i=merged_index[j] signifies that self[i]*other[j] has been merged
        let mut merged_index = vec![0; other.nterms()];
        // in_heap[j] signifies that other[j] is in the heap
        let mut in_heap = vec![false; other.nterms()];
        in_heap[0] = true;

        while let Some(cur_mon) = h.pop() {
            let mut coefficient = self.ring().zero();

            let mut q = cache.remove(&cur_mon.0).unwrap();

            for (i, j) in q.drain(..) {
                self.ring().add_mul_assign(
                    &mut coefficient,
                    &self.coefficients[i],
                    &other.coefficients[j],
                );

                merged_index[j] = i + 1;

                if i + 1 < self.nterms() && (j == 0 || merged_index[j - 1] > i + 1) {
                    let m = pack_a[i + 1] + pack_b[j];
                    match cache.entry(m) {
                        std::collections::btree_map::Entry::Occupied(mut entry) => {
                            entry.get_mut().push((i + 1, j));
                        }
                        std::collections::btree_map::Entry::Vacant(entry) => {
                            h.push(Reverse(m)); // only add when new
                            if let Some(mut qq) = q_cache.pop() {
                                qq.push((i + 1, j));
                                entry.insert(qq);
                            } else {
                                entry.insert(vec![(i + 1, j)]);
                            }
                        }
                    }
                } else {
                    in_heap[j] = false;
                }

                if j + 1 < other.nterms() && !in_heap[j + 1] {
                    let m = pack_a[i] + pack_b[j + 1];
                    match cache.entry(m) {
                        std::collections::btree_map::Entry::Occupied(mut entry) => {
                            entry.get_mut().push((i, j + 1));
                        }
                        std::collections::btree_map::Entry::Vacant(entry) => {
                            h.push(Reverse(m)); // only add when new
                            if let Some(mut qq) = q_cache.pop() {
                                qq.push((i, j + 1));
                                entry.insert(qq);
                            } else {
                                entry.insert(vec![(i, j + 1)]);
                            }
                        }
                    }

                    in_heap[j + 1] = true;
                }
            }

            q_cache.push(q);

            if !self.ring().is_zero(&coefficient) {
                res.coefficients.push(coefficient);
                let len = res.exponents.len();

                res.exponents.resize(len + self.nvars(), E::zero());

                if pack_u8 {
                    E::unpack(cur_mon.0, &mut res.exponents[len..len + self.nvars()]);
                } else {
                    E::unpack_u16(cur_mon.0, &mut res.exponents[len..len + self.nvars()]);
                }
            }
        }
        res
    }

    /// Multiply a small packed-`u8` sparse product by merging its sorted coefficient-product rows.
    ///
    /// A row fixes one term of the smaller polynomial and advances through the ordered terms of the
    /// larger polynomial. The heap exposes the next monomial from every row, so equal monomials can
    /// be accumulated before the next result term is emitted.
    fn try_packed_u8_row_merge_mul(
        &self,
        other: &MultivariatePolynomial<F, E, LexOrder>,
    ) -> Option<MultivariatePolynomial<F, E, LexOrder>> {
        if !packed_row_merge_is_bounded(self.nterms(), other.nterms()) {
            return None;
        }

        debug_assert_eq!(self.nvars(), other.nvars());
        debug_assert!(self.is_polynomial() && other.is_polynomial());
        debug_assert!((0..self.nvars()).all(|variable| {
            self.degree(variable).to_i32() as i64 + other.degree(variable).to_i32() as i64
                <= u8::MAX as i64
        }));

        let packed_self: Vec<_> = self.exponents_iter().map(E::pack).collect();
        let packed_other: Vec<_> = other.exponents_iter().map(E::pack).collect();
        let (rows, row_monomials, columns, column_monomials) = if self.nterms() <= other.nterms() {
            (self, packed_self, other, packed_other)
        } else {
            (other, packed_other, self, packed_self)
        };

        // Each entry is the next coefficient product in one sorted row.
        let mut heap = BinaryHeap::with_capacity(rows.nterms());
        for row in 0..rows.nterms() {
            heap.push(Reverse((row_monomials[row] + column_monomials[0], row, 0)));
        }

        #[inline(always)]
        fn push_next(
            heap: &mut BinaryHeap<Reverse<(u64, usize, usize)>>,
            row_monomials: &[u64],
            column_monomials: &[u64],
            monomial: u64,
            row: usize,
            column: usize,
        ) {
            if column + 1 < column_monomials.len() {
                let next_column = column + 1;
                let next_monomial = row_monomials[row] + column_monomials[next_column];
                debug_assert!(next_monomial > monomial);
                heap.push(Reverse((next_monomial, row, next_column)));
            }
        }

        let mut result = self.zero_with_capacity(self.heap_mul_result_capacity(other));
        while let Some(Reverse((monomial, row, column))) = heap.pop() {
            let mut coefficient = self
                .ring()
                .mul(&rows.coefficients[row], &columns.coefficients[column]);
            push_next(
                &mut heap,
                &row_monomials,
                &column_monomials,
                monomial,
                row,
                column,
            );

            while heap
                .peek()
                .is_some_and(|Reverse((next_monomial, _, _))| *next_monomial == monomial)
            {
                let Reverse((_, row, column)) = heap.pop().unwrap();
                self.ring().add_mul_assign(
                    &mut coefficient,
                    &rows.coefficients[row],
                    &columns.coefficients[column],
                );
                push_next(
                    &mut heap,
                    &row_monomials,
                    &column_monomials,
                    monomial,
                    row,
                    column,
                );
            }

            if !self.ring().is_zero(&coefficient) {
                result.coefficients.push(coefficient);
                let exponent_start = result.exponents.len();
                result
                    .exponents
                    .resize(exponent_start + self.nvars(), E::zero());
                E::unpack(
                    monomial,
                    &mut result.exponents[exponent_start..exponent_start + self.nvars()],
                );
            }
        }

        Some(result)
    }

    /// Return the total degree and coefficient count for a bounded dense
    /// total-degree multiplication, including the packed-exponent limits.
    fn total_degree_dense_mul_shape(
        &self,
        other: &MultivariatePolynomial<F, E, LexOrder>,
    ) -> Option<(usize, usize)> {
        let variable_count = self.nvars();
        if variable_count != other.nvars()
            || !(2..=8).contains(&variable_count)
            || !self.is_polynomial()
            || !other.is_polynomial()
            || (0..variable_count).any(|variable| {
                self.degree(variable).to_i32() as i64 + other.degree(variable).to_i32() as i64
                    > u8::MAX as i64
            })
        {
            return None;
        }

        let maximum_total_degree = |polynomial: &Self| {
            polynomial
                .exponents_iter()
                .map(|exponents| {
                    exponents
                        .iter()
                        .map(|exponent| exponent.to_i32() as usize)
                        .sum::<usize>()
                })
                .max()
                .unwrap_or(0)
        };
        let left_total_degree = maximum_total_degree(self);
        let right_total_degree = maximum_total_degree(other);
        let total_degree = left_total_degree.checked_add(right_total_degree)?;
        if total_degree >= 255 {
            return None;
        }

        fn checked_binomial(n: usize, k: usize) -> Option<usize> {
            let k = k.min(n - k);
            let mut result = 1usize;
            for index in 0..k {
                result = result.checked_mul(n - index)?.checked_div(index + 1)?;
            }
            Some(result)
        }

        let coefficient_count = checked_binomial(total_degree + variable_count, variable_count)?;
        let product_count = self.nterms().checked_mul(other.nterms())?;
        if coefficient_count > MAX_DENSE_DIV_BUFFER_SIZE
            || product_count < coefficient_count.saturating_mul(32)
        {
            return None;
        }

        Some((total_degree, coefficient_count))
    }

    /// Return whether multiplication can use the bounded total-degree simplex workspace.
    pub(super) fn total_degree_dense_mul_is_bounded(
        &self,
        other: &MultivariatePolynomial<F, E, LexOrder>,
    ) -> bool {
        self.total_degree_dense_mul_shape(other).is_some()
    }

    /// Multiply polynomials that densely occupy a bounded total-degree simplex.
    ///
    /// A full mixed-radix box is prohibitively large for many variables, even when the
    /// number of total-degree monomials is modest. Split the exponent vector and use a
    /// compact perfect-rank table for the simplex instead of maintaining a monomial heap.
    fn try_total_degree_dense_mul(
        &self,
        other: &MultivariatePolynomial<F, E, LexOrder>,
    ) -> Option<MultivariatePolynomial<F, E, LexOrder>> {
        let variable_count = self.nvars();
        let (total_degree, coefficient_count) = self.total_degree_dense_mul_shape(other)?;

        let rank_table = total_degree_rank_table(variable_count, total_degree)?;
        let radix = total_degree + 1;
        let prefix_length = rank_table.prefix_length;
        let suffix_code_count = rank_table.suffix_code_count;
        debug_assert_eq!(rank_table.suffix_length, variable_count - prefix_length);
        debug_assert_eq!(
            rank_table.prefix_code_count,
            radix.pow(prefix_length as u32)
        );

        let encode_terms = |polynomial: &Self| {
            polynomial
                .exponents_iter()
                .map(|exponents| {
                    let mut prefix = 0usize;
                    for exponent in &exponents[..prefix_length] {
                        prefix = prefix * radix + exponent.to_i32() as usize;
                    }
                    let mut suffix = 0usize;
                    for exponent in &exponents[prefix_length..] {
                        suffix = suffix * radix + exponent.to_i32() as usize;
                    }
                    (prefix, suffix)
                })
                .collect::<Vec<_>>()
        };
        let left_codes = encode_terms(self);
        let right_codes = encode_terms(other);

        let mut coefficients = vec![self.ring().zero(); coefficient_count];
        let specialized = self.ring().kernels().polynomial().and_then(|kernels| {
            kernels.try_total_degree_mul(TotalDegreePolynomialMulRequest {
                output_len: coefficient_count,
                left_coefficients: &self.coefficients,
                left_codes: &left_codes,
                right_coefficients: &other.coefficients,
                right_codes: &right_codes,
                prefix_rank: &rank_table.prefix_rank,
                prefix_remaining: &rank_table.prefix_remaining,
                suffix_rank: &rank_table.suffix_rank,
                suffix_code_count,
            })
        });

        if let Some(specialized) = specialized {
            for (rank, coefficient) in specialized {
                *coefficients.get_mut(rank as usize)? = coefficient;
            }
        } else {
            for (left_coefficient, &(left_prefix, left_suffix)) in
                self.coefficients.iter().zip(&left_codes)
            {
                for (right_coefficient, &(right_prefix, right_suffix)) in
                    other.coefficients.iter().zip(&right_codes)
                {
                    let prefix = left_prefix + right_prefix;
                    let suffix = left_suffix + right_suffix;
                    let remaining_degree =
                        unsafe { *rank_table.prefix_remaining.get_unchecked(prefix) } as usize;
                    debug_assert_ne!(remaining_degree, u8::MAX as usize);
                    let rank = unsafe { *rank_table.prefix_rank.get_unchecked(prefix) } as usize
                        + unsafe {
                            *rank_table
                                .suffix_rank
                                .get_unchecked(remaining_degree * suffix_code_count + suffix)
                        } as usize;
                    debug_assert!(rank < coefficient_count);
                    self.ring().add_mul_assign(
                        unsafe { coefficients.get_unchecked_mut(rank) },
                        left_coefficient,
                        right_coefficient,
                    );
                }
            }
        }

        fn unrank(
            mut rank: usize,
            total_degree: usize,
            exponents: &mut [usize],
            choose: impl Fn(usize, usize) -> usize,
        ) {
            let mut available_degree = total_degree;
            let variable_count = exponents.len();
            for (index, exponent) in exponents.iter_mut().enumerate() {
                let remaining_variables = variable_count - index - 1;
                for value in 0..=available_degree {
                    let count = choose(
                        remaining_variables + available_degree - value,
                        remaining_variables,
                    );
                    if rank < count {
                        *exponent = value;
                        available_degree -= value;
                        break;
                    }
                    rank -= count;
                }
            }
            debug_assert_eq!(rank, 0);
        }

        let choose = |n: usize, k: usize| rank_table.choose(n, k);
        let mut digits = vec![0usize; variable_count];
        let mut result = self.zero_with_capacity(coefficient_count);
        for (rank, coefficient) in coefficients.into_iter().enumerate() {
            if self.ring().is_zero(&coefficient) {
                continue;
            }
            unrank(rank, total_degree, &mut digits, choose);
            result.coefficients.push(coefficient);
            result
                .exponents
                .extend(digits.iter().map(|&exponent| E::from_i32(exponent as i32)));
        }
        Some(result)
    }

    /// Compute `self^pow` using a heap-based algorithm of "Sparse Polynomial Powering Using Heaps"
    /// by Michael Monagan and Roman Pearce.
    ///
    /// The caller must assure that the ring's cardinality is large enough to contain the exponents
    /// after Kronecker mapping.
    pub fn heap_pow(&self, pow: usize) -> Self {
        if self.is_constant() {
            return self.constant(self.ring().pow(&self.lcoeff(), pow as u64));
        }

        if self.coefficients.len() == 1 {
            return self.monomial(
                self.ring().pow(&self.coefficients[0], pow as u64),
                self.exponents
                    .iter()
                    .map(|x| *x * E::from_i32(pow as i32))
                    .collect(),
            );
        }

        #[inline(always)]
        fn to_uni_var<E: Exponent>(s: &[E], max_degs_rev: &[usize]) -> Integer {
            let mut shift = 1;
            let mut res = Integer::from(s.last().unwrap().to_i32());
            for (ee, &x) in s.iter().rev().skip(1).zip(max_degs_rev) {
                shift *= x as u32;
                res += ee.to_i32() as u32 * shift;
            }
            res
        }

        #[inline(always)]
        fn from_uni_var<E: Exponent>(mut p: Integer, max_degs_rev: &[usize], exp: &mut [E]) {
            for (ee, &x) in exp.iter_mut().rev().zip(max_degs_rev) {
                *ee = E::from_i32(((&p % x as u64).to_i64().unwrap() as u32) as i32);
                p /= x as u32;
            }
        }

        let degree_bounds = (0..self.nvars())
            .map(|v| self.degree_bounds(v))
            .collect::<Vec<_>>();

        let max_degs_rev = degree_bounds
            .iter()
            .rev()
            .map(|v| (v.1 - v.0).to_i32() as usize * pow + 1)
            .collect::<Vec<_>>();

        let mut exp = vec![E::zero(); self.nvars()];
        let mut f_exp: Vec<_> = self
            .exponents_iter()
            .map(|c| {
                for ((ee, x), d) in exp.iter_mut().zip(c.iter()).zip(&degree_bounds) {
                    *ee = *x - d.0;
                }

                to_uni_var(&exp, &max_degs_rev)
            })
            .collect();
        f_exp.reverse(); // descending order

        let mut g_coeff = vec![
            self.ring()
                .pow(self.coefficients.last().unwrap(), pow as u64),
        ];
        let mut g_exp = vec![f_exp[0].clone() * pow as u64];

        let mut cache: BTreeMap<Integer, Vec<(usize, usize)>> = BTreeMap::new();
        let mut q_cache: Vec<Vec<(usize, usize)>> = vec![];

        // create a min-heap since our polynomials are sorted smallest to largest
        let mut h: BinaryHeap<Integer> = BinaryHeap::with_capacity(self.nterms());

        let monom = f_exp[1].clone() + &g_exp[0];
        cache.insert(monom.clone(), vec![(1, 0)]);
        h.push(monom);

        // i=merged_index[j] signifies that self[i]*g[j] has been merged
        let mut merged_index = vec![0; self.nterms()];
        // in_heap[j] signifies that g[j] is in the heap
        let mut in_heap = vec![false; self.nterms()];
        in_heap[0] = true;

        while let Some(cur_mon) = h.pop() {
            let mut coefficient = self.ring().zero();

            let mut q = cache.remove(&cur_mon).unwrap();

            for (i, j) in q.drain(..) {
                self.ring().add_mul_assign(
                    &mut coefficient,
                    &g_coeff[j],
                    &self.ring().mul(
                        self.coefficient_back(i),
                        &self
                            .ring()
                            .nth(g_exp[j].clone() - f_exp[i].clone() * pow as u64),
                    ),
                );

                if j + 1 >= merged_index.len() {
                    merged_index.resize(j + 2, 0);
                    in_heap.resize(j + 2, false);
                }

                merged_index[j] = i + 1;

                if i + 1 < self.nterms() && (j == 0 || merged_index[j - 1] > i + 1) {
                    let m = f_exp[i + 1].clone() + &g_exp[j];
                    if let Some(e) = cache.get_mut(&m) {
                        e.push((i + 1, j));
                    } else {
                        h.push(m.clone()); // only add when new
                        if let Some(mut qq) = q_cache.pop() {
                            qq.push((i + 1, j));
                            cache.insert(m, qq);
                        } else {
                            cache.insert(m, vec![(i + 1, j)]);
                        }
                    }
                } else {
                    in_heap[j] = false;
                }

                if j + 1 < g_exp.len() && !in_heap[j + 1] {
                    let m = f_exp[i].clone() + &g_exp[j + 1];
                    if let Some(e) = cache.get_mut(&m) {
                        e.push((i, j + 1));
                    } else {
                        h.push(m.clone()); // only add when new

                        if let Some(mut qq) = q_cache.pop() {
                            qq.push((i, j + 1));
                            cache.insert(m, qq);
                        } else {
                            cache.insert(m, vec![(i, j + 1)]);
                        }
                    }

                    in_heap[j + 1] = true;
                }
            }

            q_cache.push(q);

            if !self.ring().is_zero(&coefficient) {
                g_exp.push(&cur_mon - &f_exp[0]);

                let q = self
                    .ring()
                    .try_div(
                        &coefficient,
                        &self.ring().mul(
                            self.coefficient_back(0),
                            &self.ring().nth(g_exp[0].clone() + &f_exp[0] - cur_mon),
                        ),
                    )
                    .unwrap();
                g_coeff.push(q);

                if g_exp.len() >= in_heap.len() {
                    merged_index.resize(g_exp.len(), 0);
                    in_heap.resize(g_exp.len(), false);
                }

                if !in_heap[g_exp.len() - 1] {
                    let m = f_exp[1].clone() + &g_exp[g_exp.len() - 1];
                    if let Some(e) = cache.get_mut(&m) {
                        e.push((1, g_exp.len() - 1));
                    } else {
                        h.push(m.clone());
                        if let Some(mut qq) = q_cache.pop() {
                            qq.push((1, g_exp.len() - 1));
                            cache.insert(m, qq);
                        } else {
                            cache.insert(m, vec![(1, g_exp.len() - 1)]);
                        }
                    }

                    in_heap[g_exp.len() - 1] = true;
                }
            }
        }

        let mut res = self.zero();
        for (c, e) in g_coeff.into_iter().zip(g_exp).rev() {
            from_uni_var(e, &max_degs_rev, &mut exp);

            for (ee, d) in exp.iter_mut().zip(&degree_bounds) {
                *ee += d.0 * E::from_i32(pow as i32);
            }

            res.append_monomial(c, &exp);
        }
        res
    }
}

impl<F: Field, E: PositiveExponent> MultivariatePolynomial<F, E, LexOrder> {
    /// Divide univariate polynomials over a field if the division is exact.
    ///
    /// Unlike [`Self::try_div`], this method uses the field inverse of the leading coefficient
    /// once and then performs synthetic division by a monic polynomial. This is important for
    /// extension fields where a generic exact coefficient-division test may be much more
    /// expensive than inversion.
    pub(crate) fn try_div_univariate_field(&self, div: &Self) -> Option<Self> {
        if div.is_zero() {
            return None;
        }

        if self.variables() != div.variables() {
            let mut dividend = self.clone();
            let mut divisor = div.clone();
            dividend.unify_variables(&mut divisor);
            return dividend.try_div_univariate_field(&divisor);
        }

        if self.is_zero() {
            return Some(self.clone());
        }

        if div.is_constant() {
            return Some(self.clone().mul_coeff(self.ring().inv(&div.get_constant())));
        }

        let active_variables = (0..self.nvars())
            .filter(|&variable| {
                self.degree(variable) != E::zero() || div.degree(variable) != E::zero()
            })
            .count();
        assert_eq!(
            active_variables, 1,
            "try_div_univariate_field requires univariate polynomials"
        );

        if (0..self.nvars()).any(|variable| self.degree(variable) < div.degree(variable)) {
            return None;
        }

        let leading_coefficient = div.lcoeff();
        let (leading_inverse, monic_divisor) = if self.ring().is_one(&leading_coefficient) {
            (self.ring().one(), div.clone())
        } else {
            let inverse = self.ring().inv(&leading_coefficient);
            (inverse.clone(), div.clone().mul_coeff(inverse))
        };
        let (quotient, remainder) = self.quot_rem_univariate_monic(&monic_divisor);
        remainder
            .is_zero()
            .then(|| quotient.mul_coeff(leading_inverse))
    }
}

impl<F: PolynomialGCD<E>, E: PositiveExponent> MultivariatePolynomial<F, E, LexOrder> {
    /// Divide exactly using the algorithm selected by the coefficient domain.
    pub fn try_div_exact(&self, divisor: &Self) -> Option<Self> {
        F::try_div_exact(self, divisor)
    }
}

impl<F: EuclideanDomain, E: PositiveExponent> MultivariatePolynomial<F, E, LexOrder> {
    /// Convert the polynomial to one in a number field, where the variable
    /// of the number field is moved into the coefficient.
    pub fn to_number_field(
        &self,
        field: &AlgebraicExtension<F>,
    ) -> MultivariatePolynomial<AlgebraicExtension<F>, E> {
        let var = &field.poly().get_vars_ref()[0];
        let Some(var_index) = self.get_vars_ref().iter().position(|x| x == var) else {
            return self.map_coeff(
                |c| field.element_from_polynomial(field.poly().constant(c.clone())),
                field.clone(),
            );
        };

        let polys = self.to_multivariate_polynomial_list(&[var_index], false);

        // TODO: remove the variable from the variable map?
        let mut poly =
            MultivariatePolynomial::new(field, self.nterms().into(), self.variables().clone());
        for (e, c) in polys {
            let mut c2 = MultivariatePolynomial::new(
                self.ring(),
                c.nterms().into(),
                Arc::new(vec![self.variables().as_ref()[var_index].clone()]),
            );

            c2.exponents = c
                .exponents_iter()
                .map(|x| x[var_index].to_i32() as u16)
                .collect();
            c2.coefficients = c.coefficients;

            poly.append_monomial(field.element_from_polynomial(c2), &e);
        }
        poly
    }

    /// Get the content from the coefficients.
    pub fn content(&self) -> F::Element {
        if self.coefficients.is_empty() {
            return self.ring().zero();
        }
        let mut c = self.coefficients.first().unwrap().clone();
        for cc in self.coefficients.iter().skip(1) {
            // early return if possible (not possible for rationals)
            if F::one_is_gcd_unit() && self.ring().is_one(&c) {
                break;
            }

            c = self.ring().gcd(&c, cc);
        }
        c
    }

    /// Divide every coefficient with `other`.
    pub fn div_coeff(mut self, other: &F::Element) -> Self {
        let ring = &self.context.ring;
        for c in &mut self.coefficients {
            let (quot, rem) = ring.quot_rem(c, other);
            debug_assert!(ring.is_zero(&rem));
            *c = quot;
        }
        self
    }

    /// Make the polynomial primitive by removing the content.
    pub fn make_primitive(self) -> Self {
        let c = self.content();
        self.div_coeff(&c)
    }
}

impl<F: EuclideanDomain, E: PositiveExponent> MultivariatePolynomial<F, E, LexOrder> {
    /// Compute the remainder `self % div`.
    pub fn rem(&self, div: &MultivariatePolynomial<F, E, LexOrder>) -> Self {
        self.quot_rem(div, false).1
    }

    /// Divide two multivariate polynomials and return the quotient and remainder.
    pub fn quot_rem(
        &self,
        div: &MultivariatePolynomial<F, E, LexOrder>,
        abort_on_remainder: bool,
    ) -> (
        MultivariatePolynomial<F, E, LexOrder>,
        MultivariatePolynomial<F, E, LexOrder>,
    ) {
        self.clone().quot_rem_impl(div, abort_on_remainder, false)
    }

    /// Divide an owned polynomial, reusing its coefficient storage where possible.
    pub fn quot_rem_owned(
        self,
        div: &MultivariatePolynomial<F, E, LexOrder>,
        abort_on_remainder: bool,
    ) -> (
        MultivariatePolynomial<F, E, LexOrder>,
        MultivariatePolynomial<F, E, LexOrder>,
    ) {
        self.quot_rem_impl(div, abort_on_remainder, false)
    }

    /// Compute the p-adic expansion of the polynomial.
    /// It returns `[a0, a1, a2, ...]` such that `a0 + a1 * p^1 + a2 * p^2 + ... = self`.
    pub fn p_adic_expansion(&self, p: &Self) -> Vec<Self> {
        if self.variables() != p.variables() {
            let mut c1 = self.clone();
            let mut c2 = p.clone();
            c1.unify_variables(&mut c2);
            return c1.p_adic_expansion(&c2);
        }

        let mut res = vec![];
        let mut r = self.clone();
        while !r.is_zero() {
            let (q, rem) = r.quot_rem(p, false);
            res.push(rem);
            r = q;
        }
        res
    }
}

impl<F: Ring, E: Exponent> MultivariatePolynomial<F, E, LexOrder> {
    /// Divide `self` by `div` if there is no remainder, else return `None`.
    pub fn try_div(
        &self,
        div: &MultivariatePolynomial<F, E, LexOrder>,
    ) -> Option<MultivariatePolynomial<F, E, LexOrder>> {
        if div.is_zero() {
            return None;
        }

        if self.variables() != div.variables() {
            let mut c1 = self.clone();
            let mut c2 = div.clone();
            c1.unify_variables(&mut c2);
            return c1.try_div(&c2);
        }

        if self.is_zero() {
            return Some(self.clone());
        }

        // check if the leading coefficients divide
        self.ring().try_div(&self.lcoeff(), &div.lcoeff())?;

        if !self.is_polynomial() || !div.is_polynomial() {
            // remove all negative exponents
            let mut c1 = self.clone();
            let mut c2 = div.clone();
            let degrees = (0..self.nvars())
                .map(|v| E::zero() - div.degree_bounds(v).0.min(E::zero()))
                .collect::<Vec<_>>();

            c1 = c1.mul_exp(&degrees);
            c2 = c2.mul_exp(&degrees);

            let mut degrees = (0..self.nvars())
                .map(|v| E::zero() - self.degree_bounds(v).0.min(E::zero()))
                .collect::<Vec<_>>();

            c1 = c1.mul_exp(&degrees);

            let r = c1.try_div(&c2)?;

            for d in &mut degrees {
                *d = E::zero() - *d;
            }

            return Some(r.mul_exp(&degrees));
        }

        if !self.is_polynomial() {
            return None;
        }

        if (0..self.nvars()).any(|v| self.degree(v) < div.degree(v)) {
            return None;
        }

        if self.ring().characteristic().is_zero() {
            // test division of constant term (evaluation at x_i = 0)
            let c = div.get_constant();
            if !self.ring().is_zero(&c)
                && !self.ring().is_one(&c)
                && self.ring().try_div(&self.get_constant(), &c).is_none()
            {
                return None;
            }

            // test division at x_i = 1
            let mut num = self.ring().zero();
            for c in &self.coefficients {
                self.ring().add_assign(&mut num, c);
            }
            let mut den = self.ring().zero();
            for c in &div.coefficients {
                self.ring().add_assign(&mut den, c);
            }

            if !self.ring().is_zero(&den)
                && !self.ring().is_one(&den)
                && self.ring().try_div(&num, &den).is_none()
            {
                return None;
            }
        }

        let (a, b) = self.clone().quot_rem_impl(div, true, false);
        if b.nterms() == 0 { Some(a) } else { None }
    }

    /// Divide an owned polynomial exactly, reusing its coefficient storage where possible.
    pub fn try_div_owned(
        mut self,
        div: &MultivariatePolynomial<F, E, LexOrder>,
    ) -> Option<MultivariatePolynomial<F, E, LexOrder>> {
        if div.is_zero() {
            return None;
        }

        if self.variables() != div.variables() {
            let mut div = div.clone();
            self.unify_variables(&mut div);
            return self.try_div_owned(&div);
        }

        if self.is_zero() {
            return Some(self);
        }

        // Check the leading coefficients before starting polynomial division.
        self.ring().try_div(&self.lcoeff(), &div.lcoeff())?;

        if !self.is_polynomial() || !div.is_polynomial() {
            // Remove all negative exponents.
            let mut divisor = div.clone();
            let divisor_shift = (0..self.nvars())
                .map(|variable| E::zero() - div.degree_bounds(variable).0.min(E::zero()))
                .collect::<Vec<_>>();
            let mut quotient_shift = (0..self.nvars())
                .map(|variable| E::zero() - self.degree_bounds(variable).0.min(E::zero()))
                .collect::<Vec<_>>();

            self = self.mul_exp(&divisor_shift).mul_exp(&quotient_shift);
            divisor = divisor.mul_exp(&divisor_shift);

            let quotient = self.try_div_owned(&divisor)?;
            for degree in &mut quotient_shift {
                *degree = E::zero() - *degree;
            }
            return Some(quotient.mul_exp(&quotient_shift));
        }

        if (0..self.nvars()).any(|variable| self.degree(variable) < div.degree(variable)) {
            return None;
        }

        if self.ring().characteristic().is_zero() {
            // Test division after evaluating every variable at zero.
            let divisor_constant = div.get_constant();
            if !self.ring().is_zero(&divisor_constant)
                && !self.ring().is_one(&divisor_constant)
                && self
                    .ring()
                    .try_div(&self.get_constant(), &divisor_constant)
                    .is_none()
            {
                return None;
            }

            // Test division after evaluating every variable at one.
            let mut numerator_value = self.ring().zero();
            for coefficient in &self.coefficients {
                self.ring().add_assign(&mut numerator_value, coefficient);
            }
            let mut divisor_value = self.ring().zero();
            for coefficient in &div.coefficients {
                self.ring().add_assign(&mut divisor_value, coefficient);
            }
            if !self.ring().is_zero(&divisor_value)
                && !self.ring().is_one(&divisor_value)
                && self
                    .ring()
                    .try_div(&numerator_value, &divisor_value)
                    .is_none()
            {
                return None;
            }
        }

        let (quotient, remainder) = self.quot_rem_impl(div, true, false);
        remainder.is_zero().then_some(quotient)
    }

    /// Divide an owned polynomial under an exact-divisibility invariant.
    fn exact_div_owned(self, div: &MultivariatePolynomial<F, E, LexOrder>) -> Self {
        let (quotient, remainder) = self.quot_rem_impl(div, true, true);
        debug_assert!(remainder.is_zero());
        quotient
    }

    /// Divide two multivariate polynomials and return the quotient and remainder.
    ///
    /// The input must not have negative exponents.
    fn quot_rem_impl(
        mut self,
        div: &MultivariatePolynomial<F, E, LexOrder>,
        abort_on_remainder: bool,
        assume_exact: bool,
    ) -> (
        MultivariatePolynomial<F, E, LexOrder>,
        MultivariatePolynomial<F, E, LexOrder>,
    ) {
        if div.is_zero() {
            panic!("Cannot divide by 0 polynomial");
        }

        if self.is_zero() {
            let remainder = self.zero();
            return (self, remainder);
        }

        if div.is_one() {
            let remainder = self.zero();
            return (self, remainder);
        }

        if self.variables() != div.variables() {
            let mut c2 = div.clone();
            self.unify_variables(&mut c2);
            return self.quot_rem_impl(&c2, abort_on_remainder, assume_exact);
        }

        if self.nterms() == div.nterms() {
            if &self == div {
                return (self.one(), self.zero());
            }

            // check if one is a multiple of the other
            if let Some(q) = self.ring().try_div(&self.lcoeff(), &div.lcoeff())
                && self
                    .into_iter()
                    .zip(div)
                    .all(|(t1, t2)| t1.exponents == t2.exponents)
                && self
                    .into_iter()
                    .zip(div)
                    .all(|(t1, t2)| &self.ring().mul(t2.coefficient, &q) == t1.coefficient)
            {
                return (self.constant(q), self.zero());
            }
        }

        if div.is_constant() {
            let original = (!abort_on_remainder).then(|| self.clone());
            let mut q = self;
            let dive = div.to_monomial_view(0);

            if let Some(i) = div.ring().try_inv(dive.coefficient) {
                let remainder = q.zero();
                return (q.mul_coeff(i), remainder);
            }

            let ring = q.context.ring.clone();
            for c in &mut q.coefficients {
                if assume_exact {
                    let numerator = std::mem::replace(c, ring.zero());
                    *c = ring.exact_div_owned(numerator, dive.coefficient);
                } else if let Some(quotient) = ring.try_div(c, dive.coefficient) {
                    *c = quotient;
                } else if abort_on_remainder {
                    return (q.zero(), q.one());
                } else {
                    return (q.zero(), original.unwrap());
                }
            }

            let remainder = q.zero();
            return (q, remainder);
        }

        // check if the division is univariate with the same variable
        let degree_sum: Vec<_> = (0..self.nvars())
            .map(|i| self.degree(i).to_i32() as usize + div.degree(i).to_i32() as usize)
            .collect();

        if div.ring().is_one(&div.lcoeff()) && degree_sum.iter().filter(|x| **x > 0).count() == 1 {
            return self.quot_rem_univariate_monic(div);
        }

        if (assume_exact || abort_on_remainder)
            && let Some((bases, total)) = self.dense_division_layout()
        {
            return self.dense_division(div, &bases, total, assume_exact);
        }

        let mut pack_u8 = true;
        if self.nvars() <= 8
            && (0..self.nvars()).all(|i| {
                let deg = self.degree(i).to_i32() as u32;
                if deg > 127 {
                    pack_u8 = false;
                }

                deg <= 127 || self.nvars() <= 4 && deg <= 32767
            })
        {
            self.heap_division_packed_exp(div, abort_on_remainder, pack_u8, assume_exact)
        } else {
            self.heap_division(div, abort_on_remainder, assume_exact)
        }
    }

    /// Select a fixed coefficient array when the dividend occupies a reasonably dense
    /// multivariate box. This avoids maintaining a monomial heap and map for every intermediate
    /// product. Callers that do not guarantee exactness abort on the first remainder.
    fn dense_division_layout(&self) -> Option<(Vec<usize>, usize)> {
        let mut bases = Vec::with_capacity(self.nvars());
        let mut total = 1usize;
        for variable in 0..self.nvars() {
            let base = self.degree(variable).to_i32().checked_add(1)? as usize;
            total = total.checked_mul(base)?;
            if total > MAX_DENSE_DIV_BUFFER_SIZE {
                return None;
            }
            bases.push(base);
        }

        // Avoid scanning a very large mostly-empty exponent box. The floor keeps small exact
        // divisions on this path, where setting up the heap costs more than the array scan.
        if total > self.nterms().saturating_mul(64).max(1024) {
            return None;
        }

        Some((bases, total))
    }

    fn dense_division(
        mut self,
        div: &MultivariatePolynomial<F, E, LexOrder>,
        bases: &[usize],
        total: usize,
        assume_exact: bool,
    ) -> (
        MultivariatePolynomial<F, E, LexOrder>,
        MultivariatePolynomial<F, E, LexOrder>,
    ) {
        #[inline(always)]
        fn dense_index<E: Exponent>(exponents: &[E], bases: &[usize]) -> Option<usize> {
            let mut index = 0usize;
            for (exponent, &base) in exponents.iter().zip(bases) {
                let exponent = exponent.to_i32() as usize;
                if exponent >= base {
                    return None;
                }
                index = index.checked_mul(base)?.checked_add(exponent)?;
            }
            Some(index)
        }

        #[inline(always)]
        fn position_is_divisible<E: Exponent>(
            mut position: usize,
            divisor_exponents: &[E],
            bases: &[usize],
        ) -> bool {
            for (divisor_exponent, &base) in divisor_exponents.iter().rev().zip(bases.iter().rev())
            {
                if position % base < divisor_exponent.to_i32() as usize {
                    return false;
                }
                position /= base;
            }
            true
        }

        #[inline(always)]
        fn decode_index<E: Exponent>(mut position: usize, bases: &[usize], exponents: &mut [E]) {
            for (exponent, &base) in exponents.iter_mut().rev().zip(bases.iter().rev()) {
                *exponent = E::from_i32((position % base) as i32);
                position /= base;
            }
            debug_assert_eq!(position, 0);
        }

        #[cfg(test)]
        let verification_input = self.clone();
        let ring = self.context.ring.clone();
        let mut quotient = self.zero_with_capacity(self.nterms());
        let zero_remainder = self.zero();
        let nonzero_remainder = self.one();

        let divisor_indices = div
            .exponents_iter()
            .map(|exponents| dense_index(exponents, bases))
            .collect::<Option<Vec<_>>>();
        let Some(divisor_indices) = divisor_indices else {
            return (quotient, nonzero_remainder);
        };
        let divisor_leading_index = *divisor_indices.last().unwrap();

        let dividend_indices = self
            .exponents_iter()
            .map(|exponents| dense_index(exponents, bases).map(|index| index as u32))
            .collect::<Option<Vec<_>>>()
            .unwrap();
        let divisor_indices_u32 = divisor_indices
            .iter()
            .map(|&index| index as u32)
            .collect::<Vec<_>>();
        if assume_exact
            && let Some(quotient_terms) = ring.kernels().polynomial().and_then(|kernels| {
                kernels.try_dense_exact_division(DensePolynomialExactDivisionRequest {
                    total,
                    dividend_coefficients: &mut self.coefficients,
                    dividend_indices: &dividend_indices,
                    divisor_coefficients: &div.coefficients,
                    divisor_indices: &divisor_indices_u32,
                })
            })
        {
            let mut exponents = vec![E::zero(); self.nvars()];
            for (position, coefficient) in quotient_terms {
                decode_index(position as usize, bases, &mut exponents);
                quotient.coefficients.push(coefficient);
                quotient.exponents.extend_from_slice(&exponents);
            }

            #[cfg(test)]
            {
                if !(&quotient * div - verification_input).is_zero() {
                    panic!("Specialized dense exact division failed");
                }
            }

            return (quotient, zero_remainder);
        }

        let mut coefficients = (0..total).map(|_| ring.zero()).collect::<Vec<_>>();
        for (term, &index) in dividend_indices.iter().enumerate() {
            coefficients[index as usize] =
                std::mem::replace(&mut self.coefficients[term], ring.zero());
        }

        let mut quotient_terms = Vec::with_capacity(self.nterms());
        for position in (0..total).rev() {
            let coefficient = std::mem::replace(&mut coefficients[position], ring.zero());
            if ring.is_zero(&coefficient) {
                continue;
            }

            if !position_is_divisible(position, div.last_exponents(), bases) {
                return (quotient, nonzero_remainder);
            }
            let quotient_coefficient = if assume_exact {
                ring.exact_div_owned(coefficient, div.coefficients.last().unwrap())
            } else {
                let Some(quotient) =
                    ring.try_div_owned(coefficient, div.coefficients.last().unwrap())
                else {
                    return (quotient, nonzero_remainder);
                };
                quotient
            };
            let quotient_position = position - divisor_leading_index;

            for (&divisor_position, divisor_coefficient) in divisor_indices[..div.nterms() - 1]
                .iter()
                .zip(&div.coefficients[..div.nterms() - 1])
            {
                let target = quotient_position + divisor_position;
                if target >= position {
                    return (quotient, nonzero_remainder);
                }
                ring.sub_mul_assign(
                    unsafe { coefficients.get_unchecked_mut(target) },
                    &quotient_coefficient,
                    divisor_coefficient,
                );
            }
            quotient_terms.push((quotient_position, quotient_coefficient));
        }

        quotient_terms.reverse();
        let mut exponents = vec![E::zero(); self.nvars()];
        for (position, coefficient) in quotient_terms {
            decode_index(position, bases, &mut exponents);
            quotient.coefficients.push(coefficient);
            quotient.exponents.extend_from_slice(&exponents);
        }

        #[cfg(test)]
        {
            if !(&quotient * div - verification_input).is_zero() {
                panic!("Dense division failed");
            }
        }

        (quotient, zero_remainder)
    }

    /// Heap division for multivariate polynomials, using a cache so that only unique
    /// monomial exponents appear in the heap.
    /// Reference: "Sparse polynomial division using a heap" by Monagan, Pearce (2011)
    ///
    /// The input must not have negative exponents.
    fn heap_division(
        mut self,
        div: &MultivariatePolynomial<F, E, LexOrder>,
        abort_on_remainder: bool,
        assume_exact: bool,
    ) -> (
        MultivariatePolynomial<F, E, LexOrder>,
        MultivariatePolynomial<F, E, LexOrder>,
    ) {
        let original = (!abort_on_remainder).then(|| self.clone());
        #[cfg(test)]
        let verification_input = self.clone();
        let mut q = self.zero_with_capacity(self.nterms());
        let mut r = self.zero();

        let mut div_monomial_in_heap = vec![false; div.nterms()];
        let mut merged_index_of_div_monomial_in_quotient = vec![0; div.nterms()];

        let mut cache: BTreeMap<Vec<E>, Vec<(usize, usize, bool)>> = BTreeMap::new();

        let mut h: BinaryHeap<Vec<E>> = BinaryHeap::with_capacity(self.nterms());
        let mut q_cache: Vec<Vec<(usize, usize, bool)>> = vec![];

        let mut m = vec![E::zero(); div.nvars()];
        let mut m_cache = vec![E::zero(); div.nvars()];
        let mut c;

        let mut k = 0;
        while !h.is_empty() || k < self.nterms() {
            if k < self.nterms() && (h.is_empty() || self.exponents_back(k) >= h.peek().unwrap()) {
                for (s, e) in m.iter_mut().zip(self.exponents_back(k)) {
                    *s = *e;
                }

                let coefficient_index = self.nterms() - k - 1;
                let zero = self.ring().zero();
                c = std::mem::replace(&mut self.coefficients[coefficient_index], zero);
                k += 1;
            } else {
                for (s, e) in m.iter_mut().zip(h.peek().unwrap().as_slice()) {
                    *s = *e;
                }
                c = self.ring().zero();
            }

            if let Some(monomial) = h.peek()
                && &m == monomial
            {
                h.pop().unwrap();

                let mut qs = cache.remove(&m).unwrap();
                self.ring().sub_mul_assign_many(
                    &mut c,
                    qs.iter()
                        .map(|(i, j, _)| (&q.coefficients[*i], div.coefficient_back(*j))),
                );
                for (i, j, next_in_divisor) in qs.drain(..) {
                    if next_in_divisor && j + 1 < div.nterms() {
                        // quotient heap product
                        for ((m, e1), e2) in m_cache
                            .iter_mut()
                            .zip(q.exponents(i))
                            .zip(div.exponents_back(j + 1))
                        {
                            *m = *e1 + *e2;
                        }

                        // TODO: make macro
                        if let Some(e) = cache.get_mut(&m_cache) {
                            e.push((i, j + 1, true));
                        } else {
                            h.push(m_cache.clone()); // only add when new
                            if let Some(mut qq) = q_cache.pop() {
                                qq.push((i, j + 1, true));
                                cache.insert(m_cache.clone(), qq);
                            } else {
                                cache.insert(m_cache.clone(), vec![(i, j + 1, true)]);
                            }
                        }
                    } else if !next_in_divisor {
                        merged_index_of_div_monomial_in_quotient[j] = i + 1;

                        if i + 1 < q.nterms()
                            && (j == 1 // the divisor starts with the sub-leading term in the heap
                                    || merged_index_of_div_monomial_in_quotient[j - 1] > i + 1)
                        {
                            for ((m, e1), e2) in m_cache
                                .iter_mut()
                                .zip(q.exponents(i + 1))
                                .zip(div.exponents_back(j))
                            {
                                *m = *e1 + *e2;
                            }

                            if let Some(e) = cache.get_mut(&m_cache) {
                                e.push((i + 1, j, false));
                            } else {
                                h.push(m_cache.clone()); // only add when new
                                if let Some(mut qq) = q_cache.pop() {
                                    qq.push((i + 1, j, false));
                                    cache.insert(m_cache.clone(), qq);
                                } else {
                                    cache.insert(m_cache.clone(), vec![(i + 1, j, false)]);
                                }
                            }
                        } else {
                            div_monomial_in_heap[j] = false;
                        }

                        if j + 1 < div.nterms() && !div_monomial_in_heap[j + 1] {
                            for ((m, e1), e2) in m_cache
                                .iter_mut()
                                .zip(q.exponents(i))
                                .zip(div.exponents_back(j + 1))
                            {
                                *m = *e1 + *e2;
                            }

                            if let Some(e) = cache.get_mut(&m_cache) {
                                e.push((i, j + 1, false));
                            } else {
                                h.push(m_cache.clone()); // only add when new

                                if let Some(mut qq) = q_cache.pop() {
                                    qq.push((i, j + 1, false));
                                    cache.insert(m_cache.clone(), qq);
                                } else {
                                    cache.insert(m_cache.clone(), vec![(i, j + 1, false)]);
                                }
                            }

                            div_monomial_in_heap[j + 1] = true;
                        }
                    }
                }

                q_cache.push(qs);
            }

            if self.ring().is_zero(&c) {
                continue;
            }

            if div.last_exponents().iter().zip(&m).all(|(ge, me)| me >= ge) {
                let quotient_coefficient = if assume_exact {
                    self.ring()
                        .exact_div_owned(c, div.coefficients.last().unwrap())
                } else {
                    let Some(quotient) = self.ring().try_div(&c, div.coefficients.last().unwrap())
                    else {
                        if abort_on_remainder {
                            r = self.one();
                            return (q, r);
                        }
                        return (self.zero(), original.unwrap());
                    };
                    quotient
                };
                q.coefficients.push(quotient_coefficient);

                q.exponents.extend(
                    div.last_exponents()
                        .iter()
                        .zip(&m)
                        .map(|(ge, me)| *me - *ge),
                );

                if div.nterms() == 1 {
                    continue;
                }

                for ((m, e1), e2) in m_cache
                    .iter_mut()
                    .zip(q.last_exponents())
                    .zip(div.exponents_back(1))
                {
                    *m = *e1 + *e2;
                }

                if q.nterms() < div.nterms() {
                    // using quotient heap

                    if let Some(e) = cache.get_mut(&m_cache) {
                        e.push((q.nterms() - 1, 1, true));
                    } else {
                        h.push(m_cache.clone()); // only add when new
                        if let Some(mut qq) = q_cache.pop() {
                            qq.push((q.nterms() - 1, 1, true));
                            cache.insert(m_cache.clone(), qq);
                        } else {
                            cache.insert(m_cache.clone(), vec![(q.nterms() - 1, 1, true)]);
                        }
                    }
                } else if q.nterms() >= div.nterms() {
                    // using divisor heap
                    if !div_monomial_in_heap[1] {
                        div_monomial_in_heap[1] = true;

                        if let Some(e) = cache.get_mut(&m_cache) {
                            e.push((q.nterms() - 1, 1, false));
                        } else {
                            h.push(m_cache.clone()); // only add when new
                            if let Some(mut qq) = q_cache.pop() {
                                qq.push((q.nterms() - 1, 1, false));
                                cache.insert(m_cache.clone(), qq);
                            } else {
                                cache.insert(m_cache.clone(), vec![(q.nterms() - 1, 1, false)]);
                            }
                        }
                    }
                } else {
                    // switch to divisor heap
                    for index in &mut merged_index_of_div_monomial_in_quotient {
                        *index = q.nterms() - 1;
                    }
                    debug_assert!(div_monomial_in_heap.iter().any(|c| !c));
                    div_monomial_in_heap[1] = true;

                    if let Some(e) = cache.get_mut(&m_cache) {
                        e.push((q.nterms() - 1, 1, false));
                    } else {
                        h.push(m_cache.clone()); // only add when new
                        if let Some(mut qq) = q_cache.pop() {
                            qq.push((q.nterms() - 1, 1, false));
                            cache.insert(m_cache.clone(), qq);
                        } else {
                            cache.insert(m_cache.clone(), vec![(q.nterms() - 1, 1, false)]);
                        }
                    }
                }
            } else if abort_on_remainder {
                r = self.one();
                return (q, r);
            } else {
                r.coefficients.push(c);
                r.exponents.extend(&m);
            }
        }

        // q and r have the highest monomials first
        q.reverse_monomials();
        r.reverse_monomials();

        #[cfg(test)]
        {
            if !(&q * div + r.clone() - verification_input.clone()).is_zero() {
                panic!("Division failed: ({verification_input})/({div}): q={q}, r={r}");
            }
        }

        (q, r)
    }

    /// Heap division, but with the exponents packed into a `u64`.
    /// Each exponent is limited to 32767 if there are 5 or fewer variables,
    /// or 127 if there are 8 or fewer variables, such that the last bit per byte can
    /// be used to check for subtraction overflow, serving as a division test.
    ///
    /// The input must not have negative exponents.
    fn heap_division_packed_exp(
        mut self,
        div: &MultivariatePolynomial<F, E, LexOrder>,
        abort_on_remainder: bool,
        pack_u8: bool,
        assume_exact: bool,
    ) -> (
        MultivariatePolynomial<F, E, LexOrder>,
        MultivariatePolynomial<F, E, LexOrder>,
    ) {
        let original = (!abort_on_remainder).then(|| self.clone());
        #[cfg(test)]
        let verification_input = self.clone();
        let mut q = self.zero_with_capacity(self.nterms());
        let mut r = self.zero();

        let pack_a: Vec<_> = if pack_u8 {
            self.exponents_iter().map(|c| E::pack(c)).collect()
        } else {
            self.exponents_iter().map(|c| E::pack_u16(c)).collect()
        };
        let pack_div: Vec<_> = if pack_u8 {
            div.exponents_iter().map(|c| E::pack(c)).collect()
        } else {
            div.exponents_iter().map(|c| E::pack_u16(c)).collect()
        };

        let mut div_monomial_in_heap = vec![false; div.nterms()];
        let mut merged_index_of_div_monomial_in_quotient = vec![0; div.nterms()];

        let mut cache: BTreeMap<u64, Vec<(usize, usize, bool)>> = BTreeMap::new();

        #[inline(always)]
        fn divides(a: u64, b: u64, pack_u8: bool) -> Option<u64> {
            let d = a.overflowing_sub(b).0;
            if pack_u8 && (d & 9259542123273814144u64 == 0)
                || !pack_u8 && (d & 9223512776490647552u64 == 0)
            {
                Some(d)
            } else {
                None
            }
        }

        let mut h: BinaryHeap<u64> = BinaryHeap::with_capacity(self.nterms());
        let mut q_cache: Vec<Vec<(usize, usize, bool)>> = Vec::with_capacity(self.nterms());

        let mut m;
        let mut m_cache;
        let mut c;

        let mut q_exp = Vec::with_capacity(self.nterms());

        let mut k = 0;
        while !h.is_empty() || k < self.nterms() {
            if k < self.nterms()
                && (h.is_empty() || pack_a[self.nterms() - k - 1] >= *h.peek().unwrap())
            {
                m = pack_a[self.nterms() - k - 1];

                let coefficient_index = self.nterms() - k - 1;
                let zero = self.ring().zero();
                c = std::mem::replace(&mut self.coefficients[coefficient_index], zero);

                k += 1;
            } else {
                m = *h.peek().unwrap();
                c = self.ring().zero();
            }

            if let Some(monomial) = h.peek()
                && &m == monomial
            {
                h.pop().unwrap();

                let mut qs = cache.remove(&m).unwrap();
                self.ring().sub_mul_assign_many(
                    &mut c,
                    qs.iter()
                        .map(|(i, j, _)| (&q.coefficients[*i], div.coefficient_back(*j))),
                );
                for (i, j, next_in_divisor) in qs.drain(..) {
                    if next_in_divisor && j + 1 < div.nterms() {
                        // quotient heap product
                        m_cache = q_exp[i] + pack_div[div.nterms() - (j + 1) - 1];

                        // TODO: make macro
                        if let Some(e) = cache.get_mut(&m_cache) {
                            e.push((i, j + 1, true));
                        } else {
                            h.push(m_cache); // only add when new
                            if let Some(mut qq) = q_cache.pop() {
                                qq.push((i, j + 1, true));
                                cache.insert(m_cache, qq);
                            } else {
                                cache.insert(m_cache, vec![(i, j + 1, true)]);
                            }
                        }
                    } else if !next_in_divisor {
                        merged_index_of_div_monomial_in_quotient[j] = i + 1;

                        if i + 1 < q.nterms()
                            && (j == 1 // the divisor starts with the sub-leading term in the heap
                                    || merged_index_of_div_monomial_in_quotient[j - 1] > i + 1)
                        {
                            m_cache = q_exp[i + 1] + pack_div[div.nterms() - j - 1];

                            if let Some(e) = cache.get_mut(&m_cache) {
                                e.push((i + 1, j, false));
                            } else {
                                h.push(m_cache); // only add when new
                                if let Some(mut qq) = q_cache.pop() {
                                    qq.push((i + 1, j, false));
                                    cache.insert(m_cache, qq);
                                } else {
                                    cache.insert(m_cache, vec![(i + 1, j, false)]);
                                }
                            }
                        } else {
                            div_monomial_in_heap[j] = false;
                        }

                        if j + 1 < div.nterms() && !div_monomial_in_heap[j + 1] {
                            m_cache = q_exp[i] + pack_div[div.nterms() - (j + 1) - 1];

                            if let Some(e) = cache.get_mut(&m_cache) {
                                e.push((i, j + 1, false));
                            } else {
                                h.push(m_cache); // only add when new

                                if let Some(mut qq) = q_cache.pop() {
                                    qq.push((i, j + 1, false));
                                    cache.insert(m_cache, qq);
                                } else {
                                    cache.insert(m_cache, vec![(i, j + 1, false)]);
                                }
                            }

                            div_monomial_in_heap[j + 1] = true;
                        }
                    }
                }

                q_cache.push(qs);
            }

            if self.ring().is_zero(&c) {
                continue;
            }

            let q_e = divides(m, pack_div[pack_div.len() - 1], pack_u8);
            if let Some(q_e) = q_e {
                let quotient_coefficient = if assume_exact {
                    self.ring()
                        .exact_div_owned(c, div.coefficients.last().unwrap())
                } else {
                    let Some(quotient) = self.ring().try_div(&c, div.coefficients.last().unwrap())
                    else {
                        if abort_on_remainder {
                            r = self.one();
                            return (q, r);
                        }
                        return (self.zero(), original.unwrap());
                    };
                    quotient
                };
                q.coefficients.push(quotient_coefficient);

                let len = q.exponents.len();
                q.exponents.resize(len + self.nvars(), E::zero());

                if pack_u8 {
                    E::unpack(q_e, &mut q.exponents[len..len + self.nvars()]);
                } else {
                    E::unpack_u16(q_e, &mut q.exponents[len..len + self.nvars()]);
                }
                q_exp.push(q_e);

                if div.nterms() == 1 {
                    continue;
                }

                m_cache = q_exp.last().unwrap() + pack_div[pack_div.len() - 2];

                if q.nterms() < div.nterms() {
                    // using quotient heap

                    if let Some(e) = cache.get_mut(&m_cache) {
                        e.push((q.nterms() - 1, 1, true));
                    } else {
                        h.push(m_cache); // only add when new
                        if let Some(mut qq) = q_cache.pop() {
                            qq.push((q.nterms() - 1, 1, true));
                            cache.insert(m_cache, qq);
                        } else {
                            cache.insert(m_cache, vec![(q.nterms() - 1, 1, true)]);
                        }
                    }
                } else if q.nterms() >= div.nterms() {
                    // using divisor heap
                    if !div_monomial_in_heap[1] {
                        div_monomial_in_heap[1] = true;

                        if let Some(e) = cache.get_mut(&m_cache) {
                            e.push((q.nterms() - 1, 1, false));
                        } else {
                            h.push(m_cache); // only add when new
                            if let Some(mut qq) = q_cache.pop() {
                                qq.push((q.nterms() - 1, 1, false));
                                cache.insert(m_cache, qq);
                            } else {
                                cache.insert(m_cache, vec![(q.nterms() - 1, 1, false)]);
                            }
                        }
                    }
                } else {
                    // switch to divisor heap
                    for index in &mut merged_index_of_div_monomial_in_quotient {
                        *index = q.nterms() - 1;
                    }
                    debug_assert!(div_monomial_in_heap.iter().any(|c| !c));
                    div_monomial_in_heap[1] = true;

                    if let Some(e) = cache.get_mut(&m_cache) {
                        e.push((q.nterms() - 1, 1, false));
                    } else {
                        h.push(m_cache); // only add when new
                        if let Some(mut qq) = q_cache.pop() {
                            qq.push((q.nterms() - 1, 1, false));
                            cache.insert(m_cache, qq);
                        } else {
                            cache.insert(m_cache, vec![(q.nterms() - 1, 1, false)]);
                        }
                    }
                }
            } else if abort_on_remainder {
                r = self.one();
                return (q, r);
            } else {
                r.coefficients.push(c);
                let len = r.exponents.len();
                r.exponents.resize(len + self.nvars(), E::zero());

                if pack_u8 {
                    E::unpack(m, &mut r.exponents[len..len + self.nvars()]);
                } else {
                    E::unpack_u16(m, &mut r.exponents[len..len + self.nvars()]);
                }
            }
        }

        // q and r have the highest monomials first
        q.reverse_monomials();
        r.reverse_monomials();

        #[cfg(test)]
        {
            if !(&q * div + r.clone() - verification_input.clone()).is_zero() {
                panic!("Division failed: ({verification_input})/({div}): q={q}, r={r}");
            }
        }

        (q, r)
    }
}

impl<F: Field, E: Exponent, O: MonomialOrder> MultivariatePolynomial<F, E, O> {
    /// Make the polynomial monic, i.e., make the leading coefficient `1` by
    /// multiplying all monomials with `1/lcoeff`.
    pub fn make_monic(self) -> Self {
        if self.lcoeff() != self.ring().one() {
            let ci = self.ring().inv(&self.lcoeff());
            self.mul_coeff(ci)
        } else {
            self
        }
    }
}

impl<R: Field + RealEmbedding, E: PositiveExponent> MultivariatePolynomial<R, E, LexOrder> {
    /// Count the distinct roots of this polynomial in the open interval
    /// `(0, +infinity)` using a Sturm sequence.
    ///
    /// The coefficient ring supplies the real embedding used to determine
    /// coefficient signs. The polynomial must be univariate. A constant
    /// nonzero polynomial has no roots, while the zero polynomial is rejected.
    pub fn count_positive_real_roots(&self) -> Result<usize, PositiveRealRootCountError<R::Error>> {
        if self.is_zero() {
            return Err(PositiveRealRootCountError::ZeroPolynomial);
        }
        if self.nvars() == 0 {
            return Ok(0);
        }
        if self.nvars() != 1 {
            return Err(PositiveRealRootCountError::NotUnivariate {
                variables: self.nvars(),
            });
        }

        let mut previous = self.to_univariate_from_univariate(0);
        let mut current = previous.derivative();
        let mut sturm_sequence = vec![previous.clone()];

        if !current.is_zero() {
            sturm_sequence.push(current.clone());
        }

        while !current.is_zero() {
            let remainder = -previous.rem(&current);
            previous = current;
            current = remainder;
            if !current.is_zero() {
                sturm_sequence.push(current.clone());
            }
        }

        let sign_variations = |at_positive_infinity: bool| {
            let mut previous_sign = Ordering::Equal;
            let mut variations = 0usize;

            for polynomial in &sturm_sequence {
                let value = if at_positive_infinity {
                    polynomial.lcoeff()
                } else {
                    polynomial.get_constant()
                };
                let sign = self
                    .ring()
                    .try_sign(&value)
                    .map_err(PositiveRealRootCountError::Comparison)?;
                if sign == Ordering::Equal {
                    continue;
                }
                if previous_sign != Ordering::Equal && sign != previous_sign {
                    variations += 1;
                }
                previous_sign = sign;
            }

            Ok::<_, PositiveRealRootCountError<R::Error>>(variations)
        };

        let at_zero = sign_variations(false)?;
        let at_positive_infinity = sign_variations(true)?;
        at_zero
            .checked_sub(at_positive_infinity)
            .ok_or(PositiveRealRootCountError::InvalidSturmSequence)
    }
}

impl<F: Field, E: PositiveExponent, O: MonomialOrder> MultivariatePolynomial<F, E, O> {
    /// Integrate the polynomial w.r.t the variable `var`,
    /// producing the antiderivative with zero constant.
    pub fn integrate(&self, var: usize) -> Self {
        debug_assert!(var < self.nvars());
        if self.is_zero() {
            return self.zero();
        }

        let mut res = self.zero_with_capacity(self.nterms());

        let mut exp = vec![E::zero(); self.nvars()];
        for x in self {
            exp.copy_from_slice(x.exponents);
            let pow = exp[var].to_u32() as u64;
            exp[var] += E::one();
            res.append_monomial(
                self.ring()
                    .div(x.coefficient, &self.ring().nth(Integer::from(pow) + 1)),
                &exp,
            );
        }

        res
    }
}

impl<F: Field, E: PositiveExponent> MultivariatePolynomial<F, E, LexOrder> {
    /// Optimized division routine for univariate polynomials over a field, which
    /// makes the divisor monic first.
    pub fn quot_rem_univariate(
        &self,
        div: &mut MultivariatePolynomial<F, E, LexOrder>,
    ) -> (
        MultivariatePolynomial<F, E, LexOrder>,
        MultivariatePolynomial<F, E, LexOrder>,
    ) {
        if self.is_zero() {
            return (self.clone(), self.clone());
        }

        if div.nterms() == 1 {
            // calculate inverse once
            let inv = self.ring().inv(&div.coefficients[0]);

            if div.is_constant() {
                let mut q = self.clone();
                for c in &mut q.coefficients {
                    self.ring().mul_assign(c, &inv);
                }

                return (q, self.zero());
            }

            let mut q = self.zero_with_capacity(self.nterms());
            let mut r = self.zero();
            let dive = div.exponents(0);

            for m in self.into_iter() {
                if m.exponents.iter().zip(dive).all(|(a, b)| a >= b) {
                    q.coefficients.push(self.ring().mul(m.coefficient, &inv));

                    for (ee, ed) in m.exponents.iter().zip(dive) {
                        q.exponents.push(*ee - *ed);
                    }
                } else {
                    r.coefficients.push(m.coefficient.clone());
                    r.exponents.extend(m.exponents);
                }
            }
            return (q, r);
        }

        // normalize the lcoeff to 1 to prevent a costly inversion
        if !self.ring().is_one(&div.lcoeff()) {
            let o = div.lcoeff();
            let inv = self.ring().inv(&div.lcoeff());

            for c in &mut div.coefficients {
                self.ring().mul_assign(c, &inv);
            }

            let mut res = self.quot_rem_univariate_monic(div);

            for c in &mut res.0.coefficients {
                self.ring().mul_assign(c, &inv);
            }

            for c in &mut div.coefficients {
                self.ring().mul_assign(c, &o);
            }

            return res;
        }

        self.quot_rem_univariate_monic(div)
    }

    /// Compute self^n % m where m is a polynomial
    pub fn exp_mod_univariate(&self, mut n: Integer, m: &mut Self) -> Self {
        if n.is_zero() {
            return self.one();
        }

        // use binary exponentiation and mod at every stage
        let mut x = self.quot_rem_univariate(m).1;
        let mut y = self.one();
        while !n.is_one() {
            if (&n % &Integer::Single(2)).is_one() {
                y = (&y * &x).quot_rem_univariate(m).1;
                n -= &Integer::one();
            }

            x = (&x * &x).quot_rem_univariate(m).1;
            n /= 2;
        }

        (x * &y).quot_rem_univariate(m).1
    }

    /// Perform a fast univariate division of `self` by `div`, using a precomputed inverse of `div` reversed.
    pub fn quot_rem_univariate_fast(&self, div: &Self, var: usize, inv_div: &Self) -> (Self, Self) {
        if div.is_zero() {
            panic!("Cannot divide by 0 polynomial");
        }

        if self.is_zero() {
            return (self.clone(), self.clone());
        }

        let deg_a = *self.last_exponents().iter().max().unwrap();
        let deg_b = *div.last_exponents().iter().max().unwrap();
        if deg_a < deg_b {
            return (self.zero(), self.clone());
        }

        let m = deg_a - deg_b;

        let mut self_i = self.clone();
        self_i.reverse();
        let mut q = (&self_i * &inv_div.mod_var(var, m + E::one())).mod_var(var, m + E::one());

        q.reverse();

        if q.degree(var) < m {
            let mut exp = vec![E::zero(); self.nvars()];
            exp[var] = m - q.degree(var);
            q = q.mul_exp(&exp);
        }

        let r = self - &(div * &q);

        (q, r)
    }

    /// Compute `self^n % m` where `m` is a polynomial, using a precomputed inverse of `m` reversed.
    pub fn exp_mod_univariate_fast(
        &mut self,
        var: usize,
        mut n: Integer,
        m: &Self,
        m_inv: &Self,
    ) -> Self {
        if n.is_zero() {
            return self.one();
        }

        // use binary exponentiation and mod at every stage
        let mut x = self.quot_rem_univariate_fast(m, var, m_inv).1;
        let mut y = self.one();
        while !n.is_one() {
            if (&n % &Integer::Single(2)).is_one() {
                y = (&y * &x).quot_rem_univariate_fast(m, var, m_inv).1;
                n -= &Integer::one();
            }

            x = (&x * &x).quot_rem_univariate_fast(m, var, m_inv).1;
            n /= 2;
        }

        (x * &y).quot_rem_univariate_fast(m, var, m_inv).1
    }

    /// Compute `(g, s, t)` where `self * s + other * t = g`
    /// by means of the extended Euclidean algorithm.
    pub fn eea_univariate(&self, other: &Self) -> (Self, Self, Self) {
        let mut r0 = self.clone().make_monic();
        let mut r1 = other.clone().make_monic();
        let mut s0 = self.constant(self.ring().inv(&self.lcoeff()));
        let mut s1 = self.zero();
        let mut t0 = self.zero();
        let mut t1 = self.constant(self.ring().inv(&other.lcoeff()));

        while !r1.is_zero() {
            let (q, r) = r0.quot_rem_univariate(&mut r1);
            if self.ring().is_zero(&r.lcoeff()) {
                return (r1, s1, t1);
            }

            let a = self.ring().inv(&r.lcoeff());
            (r1, r0) = (r.mul_coeff(a.clone()), r1);
            (s1, s0) = ((s0 - &q * &s1).mul_coeff(a.clone()), s1);
            (t1, t0) = ((t0 - q * &t1).mul_coeff(a), t1);
        }

        (r0, s0, t0)
    }

    /// Compute `(s1,...,n2)` where `A0 * s0 + ... + An * sn = g`
    /// where `Ai = prod(polys[j], j != i)`
    /// by means of the extended Euclidean algorithm.
    ///
    /// The `polys` must be pairwise co-prime.
    pub fn diophantine_univariate(polys: &mut [Self], b: &Self) -> Vec<Self> {
        let mut cur = polys.last().unwrap().clone();
        let mut a = vec![cur.clone()];
        for x in polys[1..].iter().rev().skip(1) {
            cur = cur * x;
            a.push(cur.clone());
        }
        a.reverse();

        let mut ss = vec![];
        let mut cur_s = b.clone();
        for (p, aa) in polys.iter_mut().zip(&mut a) {
            let (g, s, t) = p.eea_univariate(aa);
            debug_assert!(g.is_one());
            let new_s = (t * &cur_s).quot_rem_univariate(p).1;
            ss.push(new_s);
            cur_s = (s * &cur_s).quot_rem_univariate(aa).1;
        }

        ss.push(cur_s);
        ss
    }

    /// Find a rational fraction `n(x)/d(x)`, the Pade approximant,
    ///  such that `d(x)*self-n(x)=0 mod x^(deg_n+deg_d+1)` and
    /// `deg(d(x)) <= deg_d` and `deg(n(x) <= deg_n` using the extended Euclidean algorithm.
    pub fn rational_approximant_univariate(&self, deg_n: u32, deg_d: u32) -> Option<(Self, Self)>
    where
        F: PolynomialGCD<E>,
    {
        let Some(var) = self.last_exponents().iter().position(|x| *x > E::zero()) else {
            return Some((self.clone(), self.one()));
        };

        let mut exp = self.last_exponents().to_vec();
        exp[var] = E::from_u32(deg_n) + E::from_u32(deg_d) + E::one();
        let mut v0 = self.monomial(self.ring().one(), exp);
        let mut v1 = self.zero();

        let mut w0 = self.clone();
        let mut w1 = self.one();

        while w0.degree(var).to_u32() > deg_n {
            let (q, r) = v0.quot_rem_univariate(&mut w0);
            (w1, v1) = (v1 - q * &w1, w1);
            (v0, w0) = (w0, r);
        }

        // TODO: normalize denominator?
        let r = w0.gcd(&w1);

        Some((w0 / &r, w1 / &r))
    }

    /// Shift a variable `var` to `var+shift`, using an optimized routine that
    /// uses a power cache. If working in a finite field, the characteristic
    /// should be larger than the degree of the polynomial.
    pub fn shift_var_cached(&self, var: usize, shift: &F::Element) -> Self {
        let d = self.degree(var).to_u32() as usize;

        let y_poly = self.to_univariate_polynomial_list(var);
        let mut sample_powers = Vec::with_capacity(d + 1);
        let mut accum = self.ring().one();

        sample_powers.push(self.ring().one());
        for _ in 0..d {
            self.ring().mul_assign(&mut accum, shift);
            sample_powers.push(accum.clone());
        }
        let mut v = vec![self.zero(); d + 1];
        for (x_poly, p) in y_poly {
            let i = p.to_u32() as usize;
            v[i] = x_poly.mul_coeff(sample_powers[i].clone());
        }

        for k in 0..d {
            for j in (k..d).rev() {
                v[j] = &v[j] + &v[j + 1];
            }
        }

        let mut accum_inv = self.ring().one();
        let sample_point_inv = self.ring().inv(shift);
        let mut shifted_coefficients = Vec::with_capacity(v.len());
        for (i, mut v) in v.into_iter().enumerate() {
            v = v.mul_coeff(accum_inv.clone());

            for x in v.exponents.chunks_mut(self.nvars()) {
                x[var] = E::from_u32(i as u32);
            }

            shifted_coefficients.push(v);

            self.ring().mul_assign(&mut accum_inv, &sample_point_inv);
        }

        self.merge_shifted_univariate_coefficients(shifted_coefficients)
    }
}

impl<R: Ring, E: Exponent> Derivable for PolynomialRing<R, E> {
    type Variable = PolyVariable;

    fn derivative(
        &self,
        p: &MultivariatePolynomial<R, E>,
        x: &PolyVariable,
    ) -> MultivariatePolynomial<R, E> {
        if let Some(pos) = p.get_vars_ref().iter().position(|v| v == x) {
            p.derivative(pos)
        } else {
            self.zero()
        }
    }
}

impl<R: EuclideanDomain, E: Exponent> MultivariatePolynomial<AlgebraicExtension<R>, E> {
    /// Convert the polynomial to a multivariate polynomial that contains the
    /// variable in the number field.
    pub fn from_number_field(&self) -> MultivariatePolynomial<R, E> {
        let var = &self.ring().poly().get_vars_ref()[0];

        let (var_map, var_index) = if let Some(p) =
            self.get_vars_ref().iter().position(|v| v == var)
        {
            if self.degree(p) > E::zero() {
                panic!(
                    "The variable of the minimal polynomial of the coefficient field also appears in the polynomial"
                );
            }
            (self.variables().clone(), p)
        } else {
            let p = self.get_vars_ref().len();
            let mut v = self.get_vars_ref().to_vec();
            v.push(var.clone());
            (Arc::new(v), p)
        };

        let mut poly =
            MultivariatePolynomial::new(self.ring().poly().ring(), self.nterms().into(), var_map);
        let mut exp = vec![E::zero(); poly.nvars()];
        for t in self {
            exp[..self.nvars()].copy_from_slice(t.exponents);
            for t2 in &t.coefficient.poly {
                exp[var_index] = E::from_i32(t2.exponents[0].to_i32());
                poly.append_monomial(t2.coefficient.clone(), &exp);
            }
        }

        poly
    }
}

/// Word operations used by one incremental integer-polynomial CRT merge.
pub(super) trait WordCrt: FiniteFieldWorkspace + Copy {
    /// Reduce an integer to the standard representative in `[0, modulus)`.
    fn reduce_integer(value: &Integer, modulus: Self) -> Self;

    /// Compute the symmetric word `t` satisfying
    /// `t * accumulated_modulus = image - accumulated_coefficient (mod p)`.
    fn correction(
        field: &FiniteField<Self>,
        image: &FiniteFieldElement<Self>,
        accumulated_mod_p: Self,
        modulus_inverse: &FiniteFieldElement<Self>,
    ) -> i64
    where
        FiniteField<Self>: FiniteFieldCore<Self> + Set<Element = FiniteFieldElement<Self>> + Ring;
}

impl WordCrt for u32 {
    #[inline(always)]
    fn reduce_integer(value: &Integer, modulus: Self) -> Self {
        match value {
            Integer::Single(value) => value.rem_euclid(i64::from(modulus)) as u32,
            Integer::Double(value) => value.get().rem_euclid(i128::from(modulus)) as u32,
            Integer::Large(value) => value.mod_u(modulus),
        }
    }

    #[inline(always)]
    fn correction(
        field: &FiniteField<Self>,
        image: &FiniteFieldElement<Self>,
        accumulated_mod_p: Self,
        modulus_inverse: &FiniteFieldElement<Self>,
    ) -> i64 {
        let image = field.from_element(image);
        let delta = if image >= accumulated_mod_p {
            image - accumulated_mod_p
        } else {
            field.get_prime() - (accumulated_mod_p - image)
        };

        // `modulus_inverse` is in Montgomery form. Multiplying it by a standard residue performs
        // exactly one Montgomery reduction and leaves the standard correction in the inner word.
        let correction = *field
            .mul(&FiniteFieldElement::from_inner(delta), modulus_inverse)
            .inner();
        if correction <= field.get_prime() / 2 {
            i64::from(correction)
        } else {
            -i64::from(field.get_prime() - correction)
        }
    }
}

impl WordCrt for u64 {
    #[inline(always)]
    fn reduce_integer(value: &Integer, modulus: Self) -> Self {
        match value {
            &Integer::Single(value) => (i128::from(value)).rem_euclid(i128::from(modulus)) as u64,
            &Integer::Double(value) => value.get().rem_euclid(i128::from(modulus)) as u64,
            Integer::Large(value) => value.mod_u64(modulus),
        }
    }

    #[inline(always)]
    fn correction(
        field: &FiniteField<Self>,
        image: &FiniteFieldElement<Self>,
        accumulated_mod_p: Self,
        modulus_inverse: &FiniteFieldElement<Self>,
    ) -> i64 {
        let image = field.from_element(image);
        let delta = if image >= accumulated_mod_p {
            image - accumulated_mod_p
        } else {
            field.get_prime() - (accumulated_mod_p - image)
        };

        let correction = *field
            .mul(&FiniteFieldElement::from_inner(delta), modulus_inverse)
            .inner();
        if correction <= field.get_prime() / 2 {
            correction as i64
        } else {
            -((field.get_prime() - correction) as i64)
        }
    }
}

/// Precomputed data for merging one word-prime image into an integer polynomial.
///
/// For an accumulated modulus `M` and a new prime `p`, construction computes
/// `(M mod p)^-1` once. Each coefficient then needs one word reduction and one fused update
/// `a += M*t`, where `t` is chosen symmetrically modulo `p`.
pub(super) struct IntegerPolynomialCrtContext<'a, UField>
where
    UField: WordCrt,
    FiniteField<UField>: FiniteFieldCore<UField> + Set<Element = FiniteFieldElement<UField>> + Ring,
{
    modulus: &'a Integer,
    field: &'a FiniteField<UField>,
    modulus_inverse: FiniteFieldElement<UField>,
}

impl<'a, UField> IntegerPolynomialCrtContext<'a, UField>
where
    UField: WordCrt,
    FiniteField<UField>: FiniteFieldCore<UField> + Set<Element = FiniteFieldElement<UField>> + Ring,
{
    /// Precompute the inverse needed by every coefficient in this CRT image.
    pub(super) fn new(modulus: &'a Integer, field: &'a FiniteField<UField>) -> Option<Self> {
        assert!(
            !modulus.is_zero() && !modulus.is_negative(),
            "the accumulated CRT modulus must be positive"
        );
        let reduced_modulus = UField::reduce_integer(modulus, field.get_prime());
        let modulus_inverse = field.try_inv(&field.to_element(reduced_modulus))?;
        Some(Self {
            modulus,
            field,
            modulus_inverse,
        })
    }

    #[inline(always)]
    fn merge_coefficient(&self, accumulated: &mut Integer, image: &FiniteFieldElement<UField>) {
        let accumulated_mod_p = UField::reduce_integer(accumulated, self.field.get_prime());
        let correction =
            UField::correction(self.field, image, accumulated_mod_p, &self.modulus_inverse);
        if correction != 0 {
            Z.add_mul_assign(accumulated, self.modulus, &Integer::Single(correction));
        }
    }

    /// Merge `image` into `accumulator` while keeping every coefficient symmetric modulo `M*p`.
    ///
    /// The coefficients in `accumulator` must initially be symmetric modulo `M`. Equal supports
    /// are updated in place. If a term vanished in one modular image, the sorted fallback merges
    /// both supports and supplies a zero coefficient on the missing side.
    pub(super) fn merge_assign<E: Exponent, O: MonomialOrder>(
        &self,
        accumulator: &mut MultivariatePolynomial<IntegerRing, E, O>,
        image: &MultivariatePolynomial<FiniteField<UField>, E, O>,
    ) {
        assert_eq!(accumulator.variables(), image.variables());
        assert!(
            image.ring().get_prime() == self.field.get_prime(),
            "the modular image must use the CRT context's prime"
        );
        debug_assert!(
            accumulator
                .coefficients
                .iter()
                .all(|coefficient| coefficient.abs() * Integer::from(2) <= *self.modulus)
        );

        if accumulator.exponents == image.exponents {
            for (accumulated, image) in accumulator.coefficients.iter_mut().zip(&image.coefficients)
            {
                self.merge_coefficient(accumulated, image);
                debug_assert!(!accumulated.is_zero());
            }
            return;
        }

        let mut result = accumulator.zero_with_capacity(accumulator.nterms() + image.nterms());
        let zero_image = self.field.zero();
        let mut accumulator_index = 0;
        let mut image_index = 0;

        while accumulator_index < accumulator.nterms() || image_index < image.nterms() {
            let (exponents, mut coefficient, image_coefficient) =
                if accumulator_index < accumulator.nterms() && image_index < image.nterms() {
                    match O::cmp(
                        accumulator.exponents(accumulator_index),
                        image.exponents(image_index),
                    ) {
                        Ordering::Equal => {
                            accumulator_index += 1;
                            image_index += 1;
                            (
                                accumulator.exponents(accumulator_index - 1),
                                accumulator.coefficients[accumulator_index - 1].clone(),
                                &image.coefficients[image_index - 1],
                            )
                        }
                        Ordering::Less => {
                            accumulator_index += 1;
                            (
                                accumulator.exponents(accumulator_index - 1),
                                accumulator.coefficients[accumulator_index - 1].clone(),
                                &zero_image,
                            )
                        }
                        Ordering::Greater => {
                            image_index += 1;
                            (
                                image.exponents(image_index - 1),
                                Integer::zero(),
                                &image.coefficients[image_index - 1],
                            )
                        }
                    }
                } else if accumulator_index < accumulator.nterms() {
                    accumulator_index += 1;
                    (
                        accumulator.exponents(accumulator_index - 1),
                        accumulator.coefficients[accumulator_index - 1].clone(),
                        &zero_image,
                    )
                } else {
                    image_index += 1;
                    (
                        image.exponents(image_index - 1),
                        Integer::zero(),
                        &image.coefficients[image_index - 1],
                    )
                };

            self.merge_coefficient(&mut coefficient, image_coefficient);
            result.append_monomial(coefficient, exponents);
        }

        *accumulator = result;
    }
}

impl<E: Exponent> MultivariatePolynomial<IntegerRing, E> {
    /// Compute the polynomial that is congruent to `self` modulo `m` and `other` modulo `p` using the Chinese Remainder Theorem.
    pub fn chinese_remainder(&self, other: &Self, m: &Integer, p: &Integer) -> Self {
        let mut i = 0;
        let mut j = 0;

        let mut res = self.zero();

        while i < self.nterms() || j < other.nterms() {
            let (exp, mut c1, mut c2) = if i < self.nterms() && j < other.nterms() {
                match self.exponents(i).cmp(other.exponents(j)) {
                    std::cmp::Ordering::Equal => {
                        i += 1;
                        j += 1;
                        (
                            self.exponents(i - 1),
                            self.coefficients[i - 1].clone(),
                            other.coefficients[j - 1].clone(),
                        )
                    }
                    std::cmp::Ordering::Less => {
                        i += 1;
                        (
                            self.exponents(i - 1),
                            self.coefficients[i - 1].clone(),
                            0.into(),
                        )
                    }
                    std::cmp::Ordering::Greater => {
                        j += 1;
                        (
                            other.exponents(j - 1),
                            0.into(),
                            other.coefficients[j - 1].clone(),
                        )
                    }
                }
            } else if i < self.nterms() {
                i += 1;
                (
                    self.exponents(i - 1),
                    self.coefficients[i - 1].clone(),
                    0.into(),
                )
            } else {
                j += 1;
                (
                    other.exponents(j - 1),
                    0.into(),
                    other.coefficients[j - 1].clone(),
                )
            };

            if c1.is_negative() {
                c1 += m;
            }
            if c2.is_negative() {
                c2 += p;
            }

            let coeff = Integer::chinese_remainder(c1, c2, m.clone(), p.clone());
            res.append_monomial(coeff, exp);
        }

        res
    }
}

impl<E: Exponent> From<&MultivariatePolynomial<IntegerRing, E>>
    for MultivariatePolynomial<RationalField, E>
{
    fn from(val: &MultivariatePolynomial<IntegerRing, E>) -> Self {
        MultivariatePolynomial::from_parts(
            val.coefficients.iter().map(|x| x.into()).collect(),
            val.exponents.clone(),
            Q,
            val.variables().clone(),
        )
    }
}

/// View object for a term in a multivariate polynomial.
#[derive(Copy, Clone, Debug)]
pub struct MonomialView<'a, F: 'a + Ring, E: 'a + Exponent> {
    pub coefficient: &'a F::Element,
    pub exponents: &'a [E],
}

/// Iterator over terms in a multivariate polynomial.
pub struct MonomialViewIterator<'a, F: Ring, E: Exponent, O: MonomialOrder> {
    poly: &'a MultivariatePolynomial<F, E, O>,
    index: usize,
}

impl<'a, F: Ring, E: Exponent, O: MonomialOrder> Iterator for MonomialViewIterator<'a, F, E, O> {
    type Item = MonomialView<'a, F, E>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.poly.nterms() {
            None
        } else {
            let view = MonomialView {
                coefficient: &self.poly.coefficients[self.index],
                exponents: self.poly.exponents(self.index),
            };
            self.index += 1;
            Some(view)
        }
    }
}

impl<'a, F: Ring, E: Exponent, O: MonomialOrder> IntoIterator
    for &'a MultivariatePolynomial<F, E, O>
{
    type Item = MonomialView<'a, F, E>;
    type IntoIter = MonomialViewIterator<'a, F, E, O>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            poly: self,
            index: 0,
        }
    }
}

// General implementation for F::Element blocked on https://github.com/rust-lang/rust/issues/20400
impl<E: Exponent, O: MonomialOrder> PartialEq<Integer>
    for MultivariatePolynomial<IntegerRing, E, O>
{
    #[inline]
    fn eq(&self, other: &Integer) -> bool {
        self.is_constant() && self.get_constant() == *other
    }
}

impl<R: Ring + FractionNormalization + EuclideanDomain, E: Exponent, O: MonomialOrder>
    PartialEq<Fraction<R>> for MultivariatePolynomial<FractionField<R>, E, O>
{
    #[inline]
    fn eq(&self, other: &Fraction<R>) -> bool {
        self.is_constant() && self.get_constant() == *other
    }
}

#[cfg(test)]
mod test {
    use std::{collections::BTreeMap, mem::size_of, sync::Arc};

    use rand::{SeedableRng, rngs::StdRng};

    use crate::{
        atom::AtomCore,
        domains::{
            Ring, SampleableRing,
            algebraic::AlgebraicExtension,
            finite_field::{FiniteFieldCore, PrimeIteratorU64, Zp, Zp64},
            integer::{Integer, IntegerRing, Z},
            rational::{Q, RationalField},
        },
        parse,
        poly::Exponent,
        symbol,
    };

    use super::{
        IntegerPolynomialCrtContext, MultivariatePolynomial, PolynomialRing,
        PolynomialSamplingPolicy, mixed_radix_dense_work_is_bounded, packed_row_merge_is_bounded,
    };

    #[test]
    fn replace_univariate_horner_matches_dense_evaluation() {
        let polynomial = parse!("2-3*x+5*x^4-7*x^9").to_polynomial::<_, u16>(&Z, None);
        let univariate = polynomial.to_univariate_from_univariate(0);

        for value in [-3, 0, 1, 5].map(Integer::from) {
            assert_eq!(
                polynomial.replace(0, &value),
                polynomial.constant(univariate.evaluate(&value))
            );
        }

        let cancellation = parse!("x-1").to_polynomial::<_, u16>(&Z, None);
        assert!(cancellation.replace(0, &Integer::from(1)).is_zero());
    }

    #[test]
    fn integer_polynomial_crt_context_matches_generic_crt() {
        let modulus = (Integer::from(1) << 200usize) + 123;
        let half_modulus: Integer = &modulus / 2;

        let mut accumulated = parse!("1+x+y").to_polynomial::<_, u16>(&Z, None);
        accumulated.coefficients = vec![
            half_modulus.clone() - 17,
            -half_modulus.clone() + 29,
            Integer::from(41),
        ];
        let image_integer =
            parse!("5+7*x+11*y").to_polynomial::<_, u16>(&Z, accumulated.variables().clone());

        let field32 = Zp::new(2_147_483_659);
        let image32 = image_integer.map_coeff(|value| field32.nth(value.clone()), field32.clone());
        let image32_integer = image32.map_coeff(|value| field32.to_symmetric_integer(value), Z);
        let expected32 = accumulated.chinese_remainder(
            &image32_integer,
            &modulus,
            &Integer::from(field32.get_prime()),
        );
        let mut actual32 = accumulated.clone();
        IntegerPolynomialCrtContext::new(&modulus, &field32)
            .unwrap()
            .merge_assign(&mut actual32, &image32);
        assert_eq!(actual32, expected32);

        let field64 = Zp64::new(18_346_744_073_709_552_031);
        let image64 = image_integer.map_coeff(|value| field64.nth(value.clone()), field64.clone());
        let image64_integer = image64.map_coeff(|value| field64.to_symmetric_integer(value), Z);
        let expected64 = accumulated.chinese_remainder(
            &image64_integer,
            &modulus,
            &Integer::from(field64.get_prime()),
        );
        let mut actual64 = accumulated;
        IntegerPolynomialCrtContext::new(&modulus, &field64)
            .unwrap()
            .merge_assign(&mut actual64, &image64);
        assert_eq!(actual64, expected64);

        let combined_modulus = &modulus * Integer::from(field64.get_prime());
        assert!(
            actual64
                .coefficients
                .iter()
                .all(|coefficient| { coefficient.abs() * Integer::from(2) <= combined_modulus })
        );
    }

    #[test]
    fn integer_polynomial_crt_context_repeated_merges() {
        let mut target = parse!("1+x+y").to_polynomial::<_, u16>(&Z, None);
        target.coefficients = vec![
            Integer::from(7),
            (Integer::from(1) << 70usize) + 19,
            (Integer::from(1) << 200usize) + 23,
        ];

        let first_prime = 2_147_483_659u32;
        let first_field = Zp::new(first_prime);
        let first_image = target.map_coeff(
            |coefficient| first_field.nth(coefficient.clone()),
            first_field.clone(),
        );
        let mut actual = first_image.map_coeff(
            |coefficient| first_field.to_symmetric_integer(coefficient),
            Z,
        );
        let mut modulus = Integer::from(first_prime);

        for prime in PrimeIteratorU64::new(u64::from(first_prime)).take(8) {
            let prime = u32::try_from(prime).unwrap();
            let field = Zp::new(prime);
            let image =
                target.map_coeff(|coefficient| field.nth(coefficient.clone()), field.clone());
            let image_as_integer =
                image.map_coeff(|coefficient| field.to_symmetric_integer(coefficient), Z);
            let expected =
                actual.chinese_remainder(&image_as_integer, &modulus, &Integer::from(prime));

            IntegerPolynomialCrtContext::new(&modulus, &field)
                .unwrap()
                .merge_assign(&mut actual, &image);
            assert_eq!(actual, expected);

            modulus *= prime;
            assert!(
                actual
                    .coefficients
                    .iter()
                    .all(|coefficient| coefficient.abs() * Integer::from(2) <= modulus)
            );
        }

        assert_eq!(actual, target);
        assert!(
            IntegerPolynomialCrtContext::new(&Integer::from(first_prime), &first_field).is_none()
        );
    }

    #[test]
    fn integer_polynomial_crt_context_merges_different_supports() {
        let modulus = (Integer::from(1) << 130usize) + 51;
        let variables = parse!("1+x+y")
            .to_polynomial::<_, u16>(&Z, None)
            .variables()
            .clone();
        let accumulated = parse!("1+x").to_polynomial::<_, u16>(&Z, variables.clone());
        let image_integer = parse!("2+y").to_polynomial::<_, u16>(&Z, variables);
        let field = Zp::new(2_147_483_659);
        let image = image_integer.map_coeff(|value| field.nth(value.clone()), field.clone());
        let image_as_integer = image.map_coeff(|value| field.to_symmetric_integer(value), Z);
        let expected = accumulated.chinese_remainder(
            &image_as_integer,
            &modulus,
            &Integer::from(field.get_prime()),
        );

        let mut actual = accumulated.clone();
        IntegerPolynomialCrtContext::new(&modulus, &field)
            .unwrap()
            .merge_assign(&mut actual, &image);
        assert_eq!(actual, expected);

        let field64 = Zp64::new(18_346_744_073_709_552_031);
        let image64 = image_integer.map_coeff(|value| field64.nth(value.clone()), field64.clone());
        let image64_as_integer = image64.map_coeff(|value| field64.to_symmetric_integer(value), Z);
        let expected64 = accumulated.chinese_remainder(
            &image64_as_integer,
            &modulus,
            &Integer::from(field64.get_prime()),
        );
        let mut actual64 = accumulated;
        IntegerPolynomialCrtContext::new(&modulus, &field64)
            .unwrap()
            .merge_assign(&mut actual64, &image64);
        assert_eq!(actual64, expected64);
    }

    #[cfg(feature = "bincode")]
    use crate::domains::float::{F64, FloatField};

    fn assert_univariate_list_roundtrip<E: Exponent>(
        polynomial: &MultivariatePolynomial<IntegerRing, E>,
        variable: usize,
    ) -> Vec<E> {
        let coefficients = polynomial.to_univariate_polynomial_list(variable);
        let mut reconstructed = polynomial.zero();
        let mut degrees = Vec::with_capacity(coefficients.len());

        for (coefficient, degree) in coefficients {
            coefficient.check_consistency();
            assert!(
                coefficient
                    .exponents_iter()
                    .all(|exponents| exponents[variable] == E::zero())
            );
            assert!(degrees.last().is_none_or(|previous| previous < &degree));

            for term in &coefficient {
                let mut exponents = term.exponents.to_vec();
                exponents[variable] = degree;
                reconstructed.append_monomial(term.coefficient.clone(), &exponents);
            }
            degrees.push(degree);
        }

        reconstructed.check_consistency();
        assert_eq!(&reconstructed, polynomial);
        degrees
    }

    #[test]
    fn univariate_polynomial_list_roundtrips_each_variable() {
        let variables = Arc::new(vec![
            symbol!("a").into(),
            symbol!("b").into(),
            symbol!("c").into(),
        ]);
        let polynomial = parse!("5+2*a*c^4+3*a*b^2*c+a^3*b^2+7*a^3*b^5*c^2+11*b^5")
            .to_polynomial::<_, u8>(&Z, Some(variables));

        assert_eq!(
            assert_univariate_list_roundtrip(&polynomial, 0),
            vec![0u8, 1, 3]
        );
        assert_eq!(
            assert_univariate_list_roundtrip(&polynomial, 1),
            vec![0u8, 2, 5]
        );
        assert_eq!(
            assert_univariate_list_roundtrip(&polynomial, 2),
            vec![0u8, 1, 2, 4]
        );
    }

    #[test]
    fn univariate_polynomial_list_supports_signed_exponents() {
        let variables = Arc::new(vec![
            symbol!("x").into(),
            symbol!("y").into(),
            symbol!("z").into(),
        ]);
        let mut polynomial = MultivariatePolynomial::<_, i8>::new(&Z, Some(5), variables);
        for (coefficient, exponents) in [
            (2, [0, -2, 2]),
            (3, [1, -2, 0]),
            (5, [2, 0, 1]),
            (7, [0, 3, 4]),
            (11, [2, 3, 0]),
        ] {
            polynomial.append_monomial(coefficient.into(), &exponents);
        }
        polynomial.check_consistency();

        assert_eq!(
            assert_univariate_list_roundtrip(&polynomial, 1),
            vec![-2i8, 0, 3]
        );
    }

    #[test]
    fn samples_with_term_and_degree_policies() {
        let variables = Arc::new(vec![symbol!("x").into(), symbol!("y").into()]);
        let ring = PolynomialRing::<_, u8>::new(Z);
        let policy = PolynomialSamplingPolicy {
            variables: variables.clone(),
            degree_bounds: vec![0..=2, 1..=3],
            term_count: 5..=5,
            coefficient: 1.into()..=1.into(),
        };
        let mut rng = StdRng::seed_from_u64(1);

        let sample = ring.sample(&mut rng, &policy);

        assert_eq!(sample.variables(), &variables);
        assert!(sample.nterms() <= 5);
        assert!(
            sample
                .into_iter()
                .all(|term| { term.exponents[0] <= 2 && (1..=3).contains(&term.exponents[1]) })
        );
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn polynomial_header_layout() {
        assert_eq!(size_of::<MultivariatePolynomial<IntegerRing>>(), 56);
        assert_eq!(size_of::<MultivariatePolynomial<RationalField>>(), 56);
        assert_eq!(size_of::<MultivariatePolynomial<Zp>>(), 56);
        assert_eq!(size_of::<MultivariatePolynomial<Zp64>>(), 56);
        assert_eq!(
            size_of::<MultivariatePolynomial<AlgebraicExtension<RationalField>>>(),
            56
        );
    }

    #[test]
    fn polynomial_context_is_shared_and_copy_on_write() {
        let variables = Arc::new(vec![symbol!("x").into(), symbol!("y").into()]);
        let p = parse!("x+y").to_polynomial::<_, u8>(&Zp64::new(17), variables);
        let mut zero = p.zero();
        assert!(Arc::ptr_eq(&p.context, &zero.context));

        zero.rename_variable(&symbol!("x").into(), &symbol!("z").into());
        assert!(!Arc::ptr_eq(&p.context, &zero.context));
        assert_eq!(p.get_vars_ref(), &[symbol!("x"), symbol!("y")]);
        assert_eq!(zero.get_vars_ref(), &[symbol!("z"), symbol!("y")]);
        assert_eq!(p.ring(), zero.ring());
    }

    #[test]
    fn cached_shift_merges_interleaved_coefficient_streams() {
        let field = Zp::new(2_147_483_659);
        let polynomial = parse!("3+a^7*b^2+2*a^3*c^4-5*a*b^5*c+7*b^6*c^2+11*a^4*b*c^5-13*a^2*c^7")
            .to_polynomial::<_, u8>(&field, None);

        for variable in 0..3 {
            for value in [-1, 1, 7] {
                let shift = field.nth(value.into());
                let direct = polynomial.shift_var(variable, &shift);
                let cached = polynomial.shift_var_cached(variable, &shift);
                direct.check_consistency();
                cached.check_consistency();
                assert_eq!(cached, direct);
            }
        }
    }

    #[test]
    fn compact_preserves_polynomial_and_releases_spare_capacity() {
        let variables = Arc::new(vec![symbol!("x").into()]);
        let mut p = MultivariatePolynomial::<_, u8>::new(&Z, Some(128), variables);
        p.append_monomial(1.into(), &[1]);
        let expected = p.clone();
        let old_coefficient_capacity = p.coefficients.capacity();
        let old_exponent_capacity = p.exponents.capacity();

        p.compact();
        assert_eq!(p, expected);
        assert!(p.coefficients.capacity() <= old_coefficient_capacity);
        assert!(p.exponents.capacity() <= old_exponent_capacity);
    }

    #[test]
    #[cfg(feature = "bincode")]
    fn polynomial_bincode_roundtrip() {
        use bincode::{Decode, de::DecoderImpl, de::read::SliceReader};

        use crate::{poly::PolyVariable, state::StateMap};

        let variables = Arc::new(vec![PolyVariable::Temporary(7)]);
        let mut p =
            MultivariatePolynomial::<_, u8>::new(&FloatField::<F64>::new(), Some(2), variables);
        p.append_monomial(F64(3.), &[0]);
        p.append_monomial(F64(5.), &[2]);

        let encoded = bincode::encode_to_vec(&p, bincode::config::standard()).unwrap();
        let reader = SliceReader::new(&encoded);
        let mut decoder =
            DecoderImpl::new(reader, bincode::config::standard(), StateMap::default());
        let decoded = MultivariatePolynomial::<FloatField<F64>, u8>::decode(&mut decoder).unwrap();

        assert_eq!(decoded, p);
    }

    use super::PositiveRealRootCountError;

    #[test]
    fn count_positive_real_roots() {
        let two_positive = parse!("(x-1)*(x-2)*(x+3)").to_polynomial::<_, u16>(&Q, None);
        assert_eq!(two_positive.count_positive_real_roots(), Ok(2));

        let no_real_roots = parse!("x^2+1").to_polynomial::<_, u16>(&Q, None);
        assert_eq!(no_real_roots.count_positive_real_roots(), Ok(0));

        let zero_is_excluded = parse!("x*(x-1)").to_polynomial::<_, u16>(&Q, None);
        assert_eq!(zero_is_excluded.count_positive_real_roots(), Ok(1));

        let repeated_root = parse!("(x-1)^2*(x+1)").to_polynomial::<_, u16>(&Q, None);
        assert_eq!(repeated_root.count_positive_real_roots(), Ok(1));

        let multivariate = parse!("x+y").to_polynomial::<_, u16>(&Q, None);
        assert_eq!(
            multivariate.count_positive_real_roots(),
            Err(PositiveRealRootCountError::NotUnivariate { variables: 2 })
        );

        let zero = parse!("0").to_polynomial::<_, u16>(&Q, None);
        assert_eq!(
            zero.count_positive_real_roots(),
            Err(PositiveRealRootCountError::ZeroPolynomial)
        );
    }

    #[test]
    fn mul_packed() {
        let p1 = parse!("v1^2+v2^3*v3*+3*v1^4+4*v2*v3").to_polynomial::<_, u8>(&Z, None);
        let b = &p1 * &p1;
        let r = parse!(
            "16*v2^2*v3^2+8*v1^2*v2*v3+v1^4+24*v1^4*v2^4*v3^2+6*v1^6*v2^3*v3+9*v1^8*v2^6*v3^2"
        );
        assert_eq!(b.to_expression(), r)
    }

    fn diagonal_integer_polynomial(
        terms: &[(i64, u8)],
        variables: Arc<Vec<crate::poly::PolyVariable>>,
    ) -> MultivariatePolynomial<IntegerRing, u8> {
        let mut polynomial = MultivariatePolynomial::new(&Z, Some(terms.len()), variables);
        for (coefficient, degree) in terms {
            polynomial.append_monomial((*coefficient).into(), &[*degree; 8]);
        }
        polynomial
    }

    fn reference_integer_product<E: Exponent>(
        left: &MultivariatePolynomial<IntegerRing, E>,
        right: &MultivariatePolynomial<IntegerRing, E>,
    ) -> MultivariatePolynomial<IntegerRing, E> {
        let mut terms = BTreeMap::<Vec<E>, Integer>::new();
        for left_term in left {
            for right_term in right {
                let exponents = left_term
                    .exponents
                    .iter()
                    .zip(right_term.exponents)
                    .map(|(left, right)| *left + *right)
                    .collect::<Vec<_>>();
                let coefficient = left_term.coefficient * right_term.coefficient;
                terms
                    .entry(exponents)
                    .and_modify(|current| *current += &coefficient)
                    .or_insert(coefficient);
            }
        }

        let mut result = left.zero_with_capacity(terms.len());
        for (exponents, coefficient) in terms {
            if !coefficient.is_zero() {
                result.append_monomial(coefficient, &exponents);
            }
        }
        result
    }

    fn packed_row_merge_variables() -> Arc<Vec<crate::poly::PolyVariable>> {
        Arc::new(vec![
            symbol!("x1").into(),
            symbol!("x2").into(),
            symbol!("x3").into(),
            symbol!("x4").into(),
            symbol!("x5").into(),
            symbol!("x6").into(),
            symbol!("x7").into(),
            symbol!("x8").into(),
        ])
    }

    #[test]
    fn packed_row_merge_combines_and_cancels_equal_monomials() {
        let variables = packed_row_merge_variables();
        let left = diagonal_integer_polynomial(&[(1, 0), (1, 100)], variables.clone());
        let right = diagonal_integer_polynomial(&[(1, 0), (-1, 100)], variables);
        let expected = reference_integer_product(&left, &right);

        let direct = left.try_packed_u8_row_merge_mul(&right).unwrap();
        let dispatched = &left * &right;
        direct.check_consistency();
        dispatched.check_consistency();
        assert_eq!(direct, expected);
        assert_eq!(dispatched, expected);
        assert_eq!(direct.nterms(), 2);
    }

    #[test]
    fn packed_row_merge_uses_the_smaller_asymmetric_support_for_rows() {
        let variables = packed_row_merge_variables();
        let left = diagonal_integer_polynomial(
            &[(2, 0), (-3, 7), (5, 29), (11, 61), (-13, 100)],
            variables.clone(),
        );
        let right = diagonal_integer_polynomial(&[(17, 0), (19, 13), (-23, 55)], variables);
        let expected = reference_integer_product(&left, &right);

        assert_eq!(left.try_packed_u8_row_merge_mul(&right).unwrap(), expected);
        assert_eq!(right.try_packed_u8_row_merge_mul(&left).unwrap(), expected);
        assert_eq!(&left * &right, expected);
        assert_eq!(&right * &left, expected);
    }

    #[test]
    fn packed_row_merge_accepts_the_u8_exponent_boundary() {
        let variables = packed_row_merge_variables();
        let left = diagonal_integer_polynomial(&[(2, 0), (3, 200)], variables.clone());
        let right = diagonal_integer_polynomial(&[(5, 0), (7, 55)], variables);
        let expected = reference_integer_product(&left, &right);

        let actual = &left * &right;
        assert_eq!(actual, expected);
        assert_eq!(actual.nterms(), 4);
        assert!(
            actual
                .last_exponents()
                .iter()
                .all(|exponent| *exponent == 255)
        );
    }

    #[test]
    fn packed_row_merge_matches_irregular_u16_polynomials() {
        let variables = packed_row_merge_variables();
        let left = parse!("2*x1^130*x3^7+3*x2^125*x8^4-5*x4^80*x6^55+7*x5^64*x7^70")
            .to_polynomial::<_, u16>(&Z, Some(variables.clone()));
        let right = parse!("11*x1^100*x2^20-13*x3^120*x4^5+17*x5^90*x8^40-19*x6^110*x7^10+23")
            .to_polynomial::<_, u16>(&Z, Some(variables));

        assert!(packed_row_merge_is_bounded(left.nterms(), right.nterms()));
        assert_eq!(&left * &right, reference_integer_product(&left, &right));
    }

    #[test]
    fn packed_row_merge_guard_preserves_the_existing_large_product_path() {
        assert!(packed_row_merge_is_bounded(64, 64));
        assert!(!packed_row_merge_is_bounded(64, 65));

        let variables = packed_row_merge_variables();
        let left_terms = (0..65)
            .map(|degree| (i64::from(degree % 7) + 1, degree as u8))
            .collect::<Vec<_>>();
        let right_terms = (0..65)
            .map(|degree| (-(i64::from(degree % 11) + 1), degree as u8))
            .collect::<Vec<_>>();
        let left = diagonal_integer_polynomial(&left_terms, variables.clone());
        let right = diagonal_integer_polynomial(&right_terms, variables);
        let expected = reference_integer_product(&left, &right);

        assert!(left.try_packed_u8_row_merge_mul(&right).is_none());
        let actual = &left * &right;
        actual.check_consistency();
        assert_eq!(actual, expected);
    }

    #[test]
    fn heap_result_preallocation_is_bounded() {
        let left = parse!("1+x^3").to_polynomial::<_, u16>(&Z, None);
        let right = parse!("1-x^5").to_polynomial::<_, u16>(&Z, left.variables().clone());
        assert_eq!(left.heap_mul_result_capacity(&right), 4);

        let mut large = left.zero_with_capacity(512);
        for degree in 0..512 {
            large.append_monomial(1.into(), &[degree]);
        }
        assert_eq!(large.heap_mul_result_capacity(&large), large.nterms());
    }

    #[test]
    fn mixed_radix_dense_dispatch_rejects_polybench_sparse_boxes() {
        assert!(!mixed_radix_dense_work_is_bounded(5_025_500, 47, 48));
        assert!(!mixed_radix_dense_work_is_bounded(4_809_024, 48, 48));
    }

    #[test]
    fn mixed_radix_dense_dispatch_preserves_dense_products() {
        // The existing three-variable dense-small benchmark has 24 slots per variable and
        // 455-by-364 input terms.
        assert!(mixed_radix_dense_work_is_bounded(24usize.pow(3), 455, 364));
    }

    #[test]
    fn mul_full() {
        let p1 = parse!("v1^2+v2^3*v3*+3*v1^4+4*v2*v3+v4+v5+v6*v1*v2+v7*v5+v8+v9*v8")
            .to_polynomial::<_, u8>(&Z, None);
        let b = &p1 * &p1;

        let r = parse!(
            "16*v2^2*v3^2+8*v1*v2^2*v3*v6+8*v1^2*v2*v3+v1^2*v2^2*v6^2+2*v1^3*v2*v6+v1^4+24*v1^4*v2^4*v3^2+6*v1^5*v2^4*v3*v6+6*v1^6*v2^3*v3+9*v1^8*v2^6*v3^2+8*v8*v2*v3+8*v8*v2*v3*v9+2*v8*v1*v2*v6+2*v8*v1*v2*v9*v6+2*v8*v1^2+2*v8*v1^2*v9+6*v8*v1^4*v2^3*v3+6*v8*v1^4*v2^3*v3*v9+v8^2+2*v8^2*v9+v8^2*v9^2+8*v5*v2*v3+8*v5*v2*v3*v7+2*v5*v1*v2*v6+2*v5*v1*v2*v7*v6+2*v5*v1^2+2*v5*v1^2*v7+6*v5*v1^4*v2^3*v3+6*v5*v1^4*v2^3*v3*v7+2*v5*v8+2*v5*v8*v9+2*v5*v8*v7+2*v5*v8*v7*v9+v5^2+2*v5^2*v7+v5^2*v7^2+8*v4*v2*v3+2*v4*v1*v2*v6+2*v4*v1^2+6*v4*v1^4*v2^3*v3+2*v4*v8+2*v4*v8*v9+2*v4*v5+2*v4*v5*v7+v4^2"
        );
        assert_eq!(b.to_expression(), r)
    }

    #[test]
    fn total_degree_dense_multiplication() {
        fn assert_product(
            left: &MultivariatePolynomial<IntegerRing, u8>,
            right: &MultivariatePolynomial<IntegerRing, u8>,
        ) {
            let actual = left.try_total_degree_dense_mul(right).unwrap();

            let mut expected_terms = BTreeMap::<Vec<u8>, Integer>::new();
            for left_term in left {
                for right_term in right {
                    let exponents = left_term
                        .exponents
                        .iter()
                        .zip(right_term.exponents)
                        .map(|(left, right)| left + right)
                        .collect::<Vec<_>>();
                    let product = left_term.coefficient * right_term.coefficient;
                    expected_terms
                        .entry(exponents)
                        .and_modify(|coefficient| *coefficient += &product)
                        .or_insert(product);
                }
            }
            let mut expected = left.zero_with_capacity(expected_terms.len());
            for (exponents, coefficient) in expected_terms {
                if !coefficient.is_zero() {
                    expected.append_monomial(coefficient, &exponents);
                }
            }

            assert_eq!(actual, expected);
        }

        let left = parse!(
            "(1+123456789012345678901*a-123456789012345678927*b+123456789012345678951*c-123456789012345678977*d)^8-1"
        )
        .to_polynomial::<_, u8>(&Z, None);
        let right = parse!(
            "(1-123456789012345678903*a+123456789012345678929*b+123456789012345678953*c-123456789012345678979*d)^8+1"
        )
        .to_polynomial::<_, u8>(&Z, left.variables().clone());
        assert_product(&left, &right);

        let left = parse!("(1+a-b+c-d+e-f+g-h)^5-1").to_polynomial::<_, u8>(&Z, None);
        let right =
            parse!("(1-a+b+c-d-e+f+g-h)^5+1").to_polynomial::<_, u8>(&Z, left.variables().clone());
        assert_product(&left, &right);
    }

    #[test]
    fn finite_field_dense_large_product_matches_integer_image() {
        let left_integer = parse!("(1+x+y+z)^24").to_polynomial::<_, u16>(&Z, None);
        let right_integer = parse!("(1+2*x-y+3*z)^23")
            .to_polynomial::<_, u16>(&Z, left_integer.variables().clone());
        let field = Zp::new(17);
        let left =
            left_integer.map_coeff(|coefficient| field.nth(coefficient.clone()), field.clone());
        let right =
            right_integer.map_coeff(|coefficient| field.nth(coefficient.clone()), field.clone());
        let left_direct = parse!("1+x+y+z")
            .to_polynomial::<_, u16>(&field, left.variables().clone())
            .pow(24);
        let right_direct = parse!("1+2*x-y+3*z")
            .to_polynomial::<_, u16>(&field, left.variables().clone())
            .pow(23);
        let expected = (&left_integer * &right_integer)
            .map_coeff(|coefficient| field.nth(coefficient.clone()), field.clone());
        let actual = &left * &right;

        assert_eq!(left_direct, left);
        assert_eq!(right_direct, right);
        assert_eq!(
            [left.nterms(), right.nterms(), actual.nterms()],
            [480, 336, 4_752]
        );
        assert_eq!(actual, expected);
    }

    #[test]
    fn div_packed() {
        let p1 = parse!("(v1+v2*5+v3*v2+v1*v2*v3)(v1+v2+v3)").to_polynomial::<_, u8>(&Z, None);

        let p2 = parse!("v1+v2+v3+1").to_polynomial::<_, u8>(&Z, p1.variables().clone());

        let (q, r) = p1.quot_rem(&p2, false);
        assert_eq!(q.to_expression(), parse!("-1+5*v2+v1+v1*v2*v3"));
        assert_eq!(r.to_expression(), parse!("1+v3-4*v2+v2*v3^2+v2^2*v3"));
    }

    #[test]
    fn div_full() {
        let p1 = parse!("(v1+v2*5+v3*v2+v1*v2*v3+v4+v5+v6+v7+v8+v9*v8)(v1+v2+v3)")
            .to_polynomial::<_, u8>(&Z, None);

        let p2 = parse!("v1+v2+v3+1").to_polynomial::<_, u8>(&Z, p1.variables().clone());

        let (q, r) = p1.quot_rem(&p2, false);
        assert_eq!(
            q.to_expression(),
            parse!("-1+v8+v8*v9+v7+v6+v5+v4+5*v2+v1+v1*v2*v3")
        );
        assert_eq!(
            r.to_expression(),
            parse!("1-v8-v8*v9-v7-v6-v5-v4+v3-4*v2+v2*v3^2+v2^2*v3")
        );
    }

    #[test]
    fn replace_last_evaluates_contiguous_rows_and_removes_cancellations() {
        let variables = Arc::new(vec![
            symbol!("x").into(),
            symbol!("y").into(),
            symbol!("z").into(),
            symbol!("w").into(),
        ]);
        let polynomial = parse!("x^2*z^4+3*x^2*z^2-4*x^2+y*z^3-y*z+5+x*z-2*x")
            .to_polynomial::<_, u16>(&Z, variables);

        let at_two = polynomial.replace_last(2, &Integer::from(2));
        let expected_at_two =
            parse!("24*x^2+6*y+5").to_polynomial::<_, u16>(&Z, polynomial.variables().clone());
        assert_eq!(at_two, expected_at_two);

        let at_zero = polynomial.replace_last(2, &Integer::from(0));
        let expected_at_zero =
            parse!("-4*x^2-2*x+5").to_polynomial::<_, u16>(&Z, polynomial.variables().clone());
        assert_eq!(at_zero, expected_at_zero);
    }

    #[test]
    fn replace_last_matches_independent_term_evaluation() {
        let variables = Arc::new(vec![
            symbol!("x").into(),
            symbol!("y").into(),
            symbol!("z").into(),
            symbol!("w").into(),
        ]);
        let mut polynomial = MultivariatePolynomial::<IntegerRing, u16>::new(&Z, None, variables);
        for x in 0..=4u16 {
            for y in 0..=3u16 {
                for z in 0..=6u16 {
                    let coefficient = i64::from((3 * x + 5 * y + 7 * z) % 11) - 5;
                    if coefficient != 0 && (x + 2 * y + 3 * z) % 4 != 0 {
                        polynomial.append_monomial(Integer::from(coefficient), &[x, y, z, 0]);
                    }
                }
            }
        }

        for value in -3..=3 {
            let value = Integer::from(value);
            let mut expected = polynomial.zero_with_capacity(polynomial.nterms());
            for term in &polynomial {
                let mut exponents = term.exponents.to_vec();
                let exponent = exponents[2];
                exponents[2] = 0;
                let power = Z.pow(&value, u64::from(exponent));
                let coefficient = term.coefficient * &power;
                expected.append_monomial(coefficient, &exponents);
            }

            assert_eq!(polynomial.replace_last(2, &value), expected);
        }
    }

    #[test]
    fn dense_checked_division_accepts_exact_and_rejects_inexact_inputs() {
        let quotient = parse!("(1+x+y+z)^8").to_polynomial::<_, u8>(&Z, None);
        let divisor =
            parse!("(1+2*x-y+3*z)^5").to_polynomial::<_, u8>(&Z, quotient.variables().clone());
        let dividend = &quotient * &divisor;
        assert_eq!(
            dividend.clone().try_div_owned(&divisor),
            Some(quotient.clone())
        );

        // This perturbation is zero both at the origin and at (1, 1, 1), so it passes the cheap
        // evaluation filters and exercises the checked dense coefficient loop.
        let perturbation = parse!("x-y").to_polynomial::<_, u8>(&Z, dividend.variables().clone());
        assert!((dividend + perturbation).try_div_owned(&divisor).is_none());
    }

    #[test]
    fn quot_rem_univariate_monic_dense() {
        let variables = Arc::new(vec![symbol!("x").into(), symbol!("y").into()]);
        let dividend = parse!("y^12-3*y^7+2*y^2-5").to_polynomial::<_, u16>(&Q, variables);
        let divisor = parse!("y^4+2*y+1").to_polynomial::<_, u16>(&Q, dividend.variables().clone());
        let (quotient, remainder) = dividend.quot_rem_univariate_monic(&divisor);

        assert_eq!(
            quotient.to_expression(),
            parse!("7+4*y+4*y^2-3*y^3-y^4-2*y^5+y^8")
        );
        assert_eq!(remainder.to_expression(), parse!("-12-18*y-10*y^2-5*y^3"));
    }

    #[test]
    fn quot_rem_univariate_monic_edge_cases() {
        let dividend = parse!("y^2+1").to_polynomial::<_, u16>(&Q, None);
        let larger = parse!("y^3+y+1").to_polynomial::<_, u16>(&Q, dividend.variables().clone());
        let one = parse!("1").to_polynomial::<_, u16>(&Q, dividend.variables().clone());

        assert_eq!(
            dividend.quot_rem_univariate_monic(&larger),
            (dividend.zero(), dividend.clone())
        );
        assert_eq!(
            dividend.quot_rem_univariate_monic(&one),
            (dividend.clone(), dividend.zero())
        );
    }

    #[test]
    fn fuse_variables() {
        let p1 = parse!("v1+v2").to_polynomial::<_, u8>(&Z, None);
        let p2 = parse!("v4").to_polynomial::<_, u8>(&Z, None);

        let p3 = parse!("v3").to_polynomial::<_, u8>(&Z, p1.variables().clone());

        let r = p1 * &p2 + p3;

        assert_eq!(
            r.get_vars_ref(),
            &[symbol!("v1"), symbol!("v2"), symbol!("v4"), symbol!("v3")]
        );
    }

    #[test]
    fn fast_exp_mod() {
        let mut a = parse!("x^3 + 2*x + 3").to_polynomial::<_, u8>(&Q, None);
        let b = parse!("x^2 + x + 2").to_polynomial::<_, u8>(&Q, None);
        let mut b_rev = b.clone();
        b_rev.reverse();
        let c = b_rev.inverse_univariate(0, 2.into());
        let d = a.exp_mod_univariate_fast(0, 4.into(), &b, &c);
        assert_eq!(d.to_expression(), parse!("367+333*x"));
    }
}
