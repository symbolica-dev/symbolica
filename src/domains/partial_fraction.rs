//! Univariate rational functions stored in partial-fraction form over a basis
//! of denominator factors.
//!
//! [PartialFractionRing] is the ring `F[x][1/f_1, ..., 1/f_k]`: the univariate
//! polynomials over a field `F` with the monic, nonconstant, pairwise coprime
//! basis factors `f_1, ..., f_k` inverted. An element is stored as
//!
//! ```text
//! P(x) + sum_i sum_{j=1..e_i} a_{i,j}(x) / f_i(x)^j,   deg a_{i,j} < deg f_i,
//! ```
//!
//! which is unique because the factors are pairwise coprime. Hence
//!
//! - addition is coefficient-wise: there is no common denominator, no
//!   polynomial gcd and no trial division, and a cancelled pole disappears by
//!   itself;
//! - zero tests and equality are exact and structural;
//! - multiplication by a polynomial and division by a basis factor are
//!   linear-time updates of the coefficient vector, using inverses between
//!   the factors that are computed once, when a factor joins the basis (with
//!   closed recurrences when both factors are linear).
//!
//! This makes the type a good fit for long sequences of sums of products
//! whose denominators come from a small set of recurring factors, such as
//! the coefficients that appear in the reduction of integration-by-parts
//! identities, which are rational functions of the space-time dimension with
//! a few recurring (mostly linear) denominator factors.
//!
//! # Self-extending and fixed bases
//!
//! The basis is append-only: a factor's index never changes, so an element
//! remains valid when the basis grows.
//!
//! - [PartialFractionRing::new] creates a *self-extending* ring (for fields
//!   with a factorization, such as `Q` and `Z_p`). Optional hints seed the
//!   basis; a denominator that enters the ring (conversions, inversion,
//!   division) is factored once (cached) and its new irreducible factors are
//!   appended. Sums and products never create new factors, so arithmetic
//!   never factors. The ring is then the field `F(x)`, but inversion and
//!   general division are expensive: they combine their operands into single
//!   fractions and factor the new numerator.
//! - [PartialFractionRing::with_fixed_basis] creates a ring whose basis is
//!   given and never extended (the factors need not be irreducible); an
//!   uncovered denominator is an error and [Ring::try_inv] and [Ring::try_div]
//!   return `None` outside the ring. A fixed basis keeps the same structure in
//!   related rings: [PartialFractionRing::map_field] maps a basis to its image
//!   modulo a prime with a fixed basis, as a factor irreducible over `Q` may
//!   split or collide with another one modulo `p`, so that factoring again
//!   per prime would change the structure. [PartialFractionRing::strict]
//!   gives a non-extending view of a self-extending ring.
//!
//! Clones of a ring share its basis. As in Symbolica's symbol table, the
//! factors are stored in an append-only vector: reading them is lock-free,
//! and appending a factor (rare) takes a lock, so elements created on
//! different threads with clones of one ring stay compatible.
//!
//! # Examples
//!
//! ```
//! use std::sync::Arc;
//! use symbolica::{
//!     atom::AtomCore,
//!     domains::{Field, Ring, RingOps, partial_fraction::PartialFractionRing, rational::Q},
//!     parse,
//!     poly::PolyVariable,
//!     symbol,
//! };
//!
//! let x: Arc<PolyVariable> = Arc::new(symbol!("x").into());
//! let variables = Arc::new(vec![x.as_ref().clone()]);
//! let poly = |s: &str| {
//!     parse!(s)
//!         .to_polynomial::<_, u16>(&Q, Some(variables.clone()))
//!         .to_univariate_from_univariate(0)
//! };
//! // Seed the basis with x^2 - 1 = (x - 1)(x + 1).
//! let ring = PartialFractionRing::new(Q, x, &[poly("x^2-1")]);
//! assert_eq!(ring.factor_count(), 2);
//!
//! // 1/(x - 1) - 1/(x + 1) = 2/(x^2 - 1)
//! let a = ring.try_from_fraction(&poly("1"), &poly("x-1")).unwrap();
//! let b = ring.try_from_fraction(&poly("1"), &poly("x+1")).unwrap();
//! let c = ring.try_from_fraction(&poly("2"), &poly("x^2-1")).unwrap();
//! assert_eq!(ring.sub(&a, &b), c);
//!
//! // Inverting c^-1 + 1 = (x^2 + 1)/2 appends the factor x^2 + 1.
//! let d = ring.inv(&ring.add(&ring.inv(&c), &ring.one()));
//! assert_eq!(ring.factor_count(), 3);
//! assert_eq!(ring.mul(&d, &ring.add(&ring.inv(&c), &ring.one())), ring.one());
//! ```

use std::{
    cmp::Ordering,
    fmt::{Display, Error, Write},
    hash::{Hash, Hasher},
    ops::RangeInclusive,
    sync::{Arc, Mutex},
};

use ahash::HashMap;
use append_only_vec::AppendOnlyVec;
use rand::Rng;

use crate::{
    domains::{
        EuclideanDomain, Field, InternalOrdering, Ring, RingOps, SampleableRing, SelfRing, Set,
        factorized_rational_polynomial::{
            FactorizedRationalPolynomial, FromNumeratorAndFactorizedDenominator,
        },
        finite_field::{FiniteField, FiniteFieldCore, FiniteFieldWorkspace, ToFiniteField},
        integer::{Integer, IntegerRing},
        rational::{Rational, RationalField},
        rational_polynomial::{FromNumeratorAndDenominator, RationalPolynomial},
    },
    poly::{
        LexOrder, PolyVariable, PositiveExponent, factor::Factorize,
        polynomial::MultivariatePolynomial, univariate::UnivariatePolynomial,
    },
    printer::{PrintOptions, PrintState},
};

/// An error in the construction of a [PartialFractionRing] or in a conversion
/// into one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PartialFractionError {
    /// The factor with this index is constant.
    ConstantFactor(usize),
    /// The factors with these indices have a common factor.
    NotCoprime(usize, usize),
    /// A denominator is zero.
    ZeroDenominator,
    /// A denominator has a factor that is not a product of basis factors.
    OutsideBasis,
    /// A polynomial depends on a variable other than the ring's variable.
    OtherVariable,
    /// A coefficient has no image in the target field, for example a rational
    /// number whose denominator is divisible by the prime.
    UnmappableCoefficient,
    /// The basis has too many factors.
    TooManyFactors,
}

impl Display for PartialFractionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConstantFactor(i) => write!(f, "basis factor {i} is constant"),
            Self::NotCoprime(i, j) => write!(f, "basis factors {i} and {j} are not coprime"),
            Self::ZeroDenominator => f.write_str("the denominator is zero"),
            Self::OutsideBasis => {
                f.write_str("the denominator has a factor outside the partial-fraction basis")
            }
            Self::OtherVariable => {
                f.write_str("the polynomial depends on a variable other than the ring's")
            }
            Self::UnmappableCoefficient => {
                f.write_str("a coefficient has no image in the target field")
            }
            Self::TooManyFactors => f.write_str("the basis has too many factors"),
        }
    }
}

impl std::error::Error for PartialFractionError {}

/// Dense polynomial helpers on ascending coefficient slices without trailing
/// zeros. Monic divisors have a last coefficient of one.
mod dense {
    use crate::domains::{Field, Ring};

    pub(super) fn trim<F: Ring>(field: &F, p: &mut Vec<F::Element>) {
        while p.last().is_some_and(|c| field.is_zero(c)) {
            p.pop();
        }
    }

    pub(super) fn add<F: Ring>(field: &F, a: &[F::Element], b: &[F::Element]) -> Vec<F::Element> {
        let (long, short) = if a.len() >= b.len() { (a, b) } else { (b, a) };
        let mut r = long.to_vec();
        for (r, s) in r.iter_mut().zip(short) {
            field.add_assign(r, s);
        }
        trim(field, &mut r);
        r
    }

    pub(super) fn sub<F: Ring>(field: &F, a: &[F::Element], b: &[F::Element]) -> Vec<F::Element> {
        let mut r = a.to_vec();
        if r.len() < b.len() {
            r.resize(b.len(), field.zero());
        }
        for (r, s) in r.iter_mut().zip(b) {
            field.sub_assign(r, s);
        }
        trim(field, &mut r);
        r
    }

    pub(super) fn mul<F: Ring>(field: &F, a: &[F::Element], b: &[F::Element]) -> Vec<F::Element> {
        if a.is_empty() || b.is_empty() {
            return vec![];
        }
        let mut r = vec![field.zero(); a.len() + b.len() - 1];
        for (i, x) in a.iter().enumerate() {
            if field.is_zero(x) {
                continue;
            }
            for (y, r) in b.iter().zip(&mut r[i..]) {
                field.add_mul_assign(r, x, y);
            }
        }
        trim(field, &mut r);
        r
    }

    pub(super) fn pow<F: Ring>(field: &F, a: &[F::Element], e: usize) -> Vec<F::Element> {
        let mut r = vec![field.one()];
        for _ in 0..e {
            r = mul(field, &r, a);
        }
        r
    }

    /// Quotient and remainder of `a` by the monic `m`.
    pub(super) fn divrem_monic<F: Ring>(
        field: &F,
        a: &[F::Element],
        m: &[F::Element],
    ) -> (Vec<F::Element>, Vec<F::Element>) {
        let k = m.len() - 1;
        if a.len() <= k {
            return (vec![], a.to_vec());
        }
        let mut r = a.to_vec();
        let mut q = vec![field.zero(); a.len() - k];
        for i in (0..q.len()).rev() {
            let c = r[i + k].clone();
            if !field.is_zero(&c) {
                for (j, mj) in m[..k].iter().enumerate() {
                    field.sub_mul_assign(&mut r[i + j], &c, mj);
                }
            }
            q[i] = c;
        }
        r.truncate(k);
        trim(field, &mut r);
        trim(field, &mut q);
        (q, r)
    }

    /// Quotient and remainder of `a` by the nonzero `b`.
    pub(super) fn divrem<F: Field>(
        field: &F,
        a: &[F::Element],
        b: &[F::Element],
    ) -> (Vec<F::Element>, Vec<F::Element>) {
        let lead = b.last().expect("nonzero divisor");
        let inverse = field.inv(lead);
        let monic: Vec<_> = b.iter().map(|c| field.mul(c, &inverse)).collect();
        let (mut q, r) = divrem_monic(field, a, &monic);
        for c in &mut q {
            field.mul_assign(c, &inverse);
        }
        (q, r)
    }

    pub(super) fn make_monic<F: Field>(field: &F, p: &mut [F::Element]) {
        if let Some(lead) = p.last() {
            let inverse = field.inv(lead);
            for c in p.iter_mut() {
                field.mul_assign(c, &inverse);
            }
        }
    }

    /// Monic gcd (empty when both are zero).
    pub(super) fn gcd<F: Field>(field: &F, a: &[F::Element], b: &[F::Element]) -> Vec<F::Element> {
        let (mut a, mut b) = (a.to_vec(), b.to_vec());
        while !b.is_empty() {
            let r = divrem(field, &a, &b).1;
            a = std::mem::replace(&mut b, r);
        }
        make_monic(field, &mut a);
        a
    }

    /// The inverse of `a` modulo the monic `m`, if `a` and `m` are coprime.
    pub(super) fn inverse_mod<F: Field>(
        field: &F,
        a: &[F::Element],
        m: &[F::Element],
    ) -> Option<Vec<F::Element>> {
        // Invariant: s_i * a = r_i mod m.
        let (mut r0, mut r1) = (m.to_vec(), divrem_monic(field, a, m).1);
        let (mut s0, mut s1): (Vec<F::Element>, Vec<F::Element>) = (vec![], vec![field.one()]);
        while !r1.is_empty() {
            let (q, r) = divrem(field, &r0, &r1);
            let s = sub(field, &s0, &mul(field, &q, &s1));
            r0 = std::mem::replace(&mut r1, r);
            s0 = std::mem::replace(&mut s1, s);
        }
        if r0.len() != 1 {
            return None;
        }
        let inverse = field.inv(&r0[0]);
        let mut s = divrem_monic(field, &s0, m).1;
        for c in &mut s {
            field.mul_assign(c, &inverse);
        }
        Some(s)
    }

    pub(super) fn evaluate<F: Ring>(field: &F, p: &[F::Element], x: &F::Element) -> F::Element {
        let mut r = field.zero();
        for c in p.iter().rev() {
            field.mul_assign(&mut r, x);
            field.add_assign(&mut r, c);
        }
        r
    }
}

/// The monic irreducible factors of a nonconstant monic polynomial, with
/// their multiplicities.
type Factorizer<F> = fn(&F, &[<F as Set>::Element]) -> Vec<(Vec<<F as Set>::Element>, usize)>;

/// The `(factor, exponent)` lists of monic denominators.
type DenominatorCache<F> = HashMap<Vec<<F as Set>::Element>, Vec<(usize, usize)>>;

/// Data relating a basis factor to an earlier one.
#[derive(Debug)]
enum Pair<E> {
    /// Both factors are linear: `(r_self - r_earlier)^-1`.
    Linear(E),
    /// `f_self^-1 mod f_earlier` and `f_earlier^-1 mod f_self`.
    General(Vec<E>, Vec<E>),
}

/// A basis factor with the data relating it to every earlier factor.
#[derive(Debug)]
struct Factor<F: Field> {
    /// Ascending coefficients, the last one is one.
    monic: Vec<F::Element>,
    /// `r` if the factor is `x - r`.
    root: Option<F::Element>,
    /// One entry per earlier factor.
    pairs: Vec<Pair<F::Element>>,
}

/// The append-only factor basis of a [PartialFractionRing].
///
/// Like Symbolica's symbol table, the factors live in an [AppendOnlyVec]:
/// reading them is lock-free, and appending (rare) is serialized by the
/// `index` lock. A factor's index never changes.
struct FactorBasis<F: Field> {
    factors: AppendOnlyVec<Factor<F>>,
    /// The index of every factor; held while appending.
    index: Mutex<HashMap<Vec<F::Element>, usize>>,
    /// The factor exponents of every monic denominator seen so far.
    denominators: Mutex<DenominatorCache<F>>,
    /// Factorization of new denominators (self-extending bases only).
    factorizer: Option<Factorizer<F>>,
}

impl<F: Field> std::fmt::Debug for FactorBasis<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FactorBasis")
            .field(
                "factors",
                &self.factors.iter().map(|f| &f.monic).collect::<Vec<_>>(),
            )
            .field("self_extending", &self.factorizer.is_some())
            .finish()
    }
}

impl<F: Field> FactorBasis<F> {
    fn new(factorizer: Option<Factorizer<F>>) -> Self {
        Self {
            factors: AppendOnlyVec::new(),
            index: Mutex::new(HashMap::default()),
            denominators: Mutex::new(HashMap::default()),
            factorizer,
        }
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.factors.len()
    }

    #[inline(always)]
    fn monic(&self, factor: u32) -> &[F::Element] {
        &self.factors[factor as usize].monic
    }

    #[inline(always)]
    fn degree(&self, factor: u32) -> usize {
        self.factors[factor as usize].monic.len() - 1
    }

    #[inline(always)]
    fn root(&self, factor: u32) -> Option<&F::Element> {
        self.factors[factor as usize].root.as_ref()
    }

    /// `(r_g - r_h)^-1` for distinct linear factors `g` and `h`.
    #[inline(always)]
    fn inverse_difference(&self, field: &F, g: u32, h: u32) -> F::Element {
        let (high, low) = (g.max(h), g.min(h));
        match &self.factors[high as usize].pairs[low as usize] {
            Pair::Linear(d) if high == g => d.clone(),
            Pair::Linear(d) => field.neg(d),
            Pair::General(..) => unreachable!("linear factors"),
        }
    }

    /// `f_h^-1 mod f_g` for distinct factors `h` and `g`, not both linear.
    fn inverse_mod(&self, h: u32, g: u32) -> &[F::Element] {
        let (high, low) = (g.max(h), g.min(h));
        match &self.factors[high as usize].pairs[low as usize] {
            Pair::General(high_mod_low, _) if high == h => high_mod_low,
            Pair::General(_, low_mod_high) => low_mod_high,
            Pair::Linear(_) => unreachable!("a nonlinear factor"),
        }
    }

    /// The index of the monic `factor`, appended if it is new. Fails if it is
    /// constant or not coprime to a factor of the basis.
    fn push(&self, field: &F, factor: Vec<F::Element>) -> Result<usize, PartialFractionError> {
        let mut index = self.index.lock().unwrap();
        if let Some(&i) = index.get(&factor) {
            return Ok(i);
        }
        let k = self.factors.len();
        if factor.len() < 2 {
            return Err(PartialFractionError::ConstantFactor(k));
        }
        if k >= u32::MAX as usize {
            return Err(PartialFractionError::TooManyFactors);
        }
        let root = (factor.len() == 2).then(|| field.neg(&factor[0]));
        let mut pairs = Vec::with_capacity(k);
        for (j, earlier) in self.factors.iter().enumerate() {
            let pair = match (&root, &earlier.root) {
                (Some(r), Some(s)) => {
                    let difference = field.sub(r, s);
                    if field.is_zero(&difference) {
                        return Err(PartialFractionError::NotCoprime(j, k));
                    }
                    Pair::Linear(field.inv(&difference))
                }
                _ => Pair::General(
                    dense::inverse_mod(field, &factor, &earlier.monic)
                        .ok_or(PartialFractionError::NotCoprime(j, k))?,
                    dense::inverse_mod(field, &earlier.monic, &factor)
                        .ok_or(PartialFractionError::NotCoprime(j, k))?,
                ),
            };
            pairs.push(pair);
        }
        index.insert(factor.clone(), k);
        let pushed = self.factors.push(Factor {
            monic: factor,
            root,
            pairs,
        });
        debug_assert_eq!(pushed, k);
        Ok(k)
    }
}

/// The monic irreducible factors of `p` by Symbolica's factorization.
fn factorize<F: Field>(field: &F, p: &[F::Element]) -> Vec<(Vec<F::Element>, usize)>
where
    MultivariatePolynomial<F, u16, LexOrder>: Factorize,
{
    let polynomial = UnivariatePolynomial::from_coefficients(
        field,
        p.to_vec(),
        Arc::new(PolyVariable::Temporary(0)),
    )
    .to_multivariate::<u16>();
    polynomial
        .factor()
        .into_iter()
        .filter(|(f, _)| !f.is_constant())
        .map(|(f, m)| {
            let mut c = f.to_univariate_from_univariate(0).coefficients;
            dense::make_monic(field, &mut c);
            (c, m)
        })
        .collect()
}

/// The ring `F[x][1/f_1, ..., 1/f_k]` of univariate rational functions over the
/// field `F` whose denominators are products of powers of the monic, pairwise
/// coprime basis factors `f_i`, with elements in partial-fraction form.
///
/// See the [module documentation](self) for the representation, the two
/// kinds of bases (self-extending and fixed) and the sharing model. Clones
/// share the basis and are cheap.
#[derive(Clone, Debug)]
pub struct PartialFractionRing<F: Field> {
    field: F,
    variable: Arc<PolyVariable>,
    basis: Arc<FactorBasis<F>>,
    /// Whether new denominators may extend the basis.
    extend: bool,
}

impl<F: Field> PartialEq for PartialFractionRing<F> {
    fn eq(&self, other: &Self) -> bool {
        self.field == other.field
            && self.variable == other.variable
            && (Arc::ptr_eq(&self.basis, &other.basis)
                || self.basis.factors.iter().map(|f| &f.monic).eq(other
                    .basis
                    .factors
                    .iter()
                    .map(|f| &f.monic)))
    }
}

impl<F: Field> Eq for PartialFractionRing<F> {}

impl<F: Field> Hash for PartialFractionRing<F> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.field.hash(state);
        self.variable.hash(state);
    }
}

/// One basis factor's coefficients `a_1, ..., a_e` in an element, `deg f`
/// ascending coefficients per `a_j`, starting at `offset`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
struct Chain {
    factor: u32,
    exponent: u32,
    offset: u32,
}

/// An element of a [PartialFractionRing].
///
/// The element is stored in canonical form (no trailing zero coefficients),
/// so that structural equality is equality in the ring.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PartialFraction<F: Field> {
    /// Length of the polynomial part, which is stored first.
    poly: usize,
    /// The factors with a nonzero coefficient, sorted by factor.
    chains: Vec<Chain>,
    coefficients: Vec<F::Element>,
}

impl<F: Field> Default for PartialFraction<F> {
    fn default() -> Self {
        Self {
            poly: 0,
            chains: vec![],
            coefficients: vec![],
        }
    }
}

impl<F: Field> InternalOrdering for PartialFraction<F> {
    fn internal_cmp(&self, other: &Self) -> Ordering {
        self.poly
            .cmp(&other.poly)
            .then_with(|| {
                let key = |c: &Chain| (c.factor, c.exponent);
                self.chains
                    .iter()
                    .map(key)
                    .cmp(other.chains.iter().map(key))
            })
            .then_with(|| self.coefficients.internal_cmp(&other.coefficients))
    }
}

impl<F: Field> PartialFraction<F> {
    /// Return `true` iff the element is zero.
    pub fn is_zero(&self) -> bool {
        self.coefficients.is_empty()
    }

    /// The number of stored field coefficients.
    pub fn len(&self) -> usize {
        self.coefficients.len()
    }

    /// Return `true` iff no coefficient is stored (the element is zero).
    pub fn is_empty(&self) -> bool {
        self.coefficients.is_empty()
    }

    /// The coefficients of the polynomial part in ascending degree order.
    pub fn polynomial_part(&self) -> &[F::Element] {
        &self.coefficients[..self.poly]
    }

    /// The highest power of every basis factor in the denominator, as
    /// `(factor index, exponent)`, sorted by factor index.
    pub fn denominator_exponents(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.chains
            .iter()
            .map(|c| (c.factor as usize, c.exponent as usize))
    }

    /// The coefficients `a_1, ..., a_e` of the basis factor `factor`: `e` blocks
    /// of `deg f` ascending coefficients, `a_j` being the numerator of `1/f^j`.
    /// Empty if `factor` does not occur.
    pub fn factor_coefficients(&self, factor: usize) -> &[F::Element] {
        match self
            .chains
            .binary_search_by_key(&(factor as u32), |c| c.factor)
        {
            Ok(i) => {
                let c = self.chains[i];
                let end = self
                    .chains
                    .get(i + 1)
                    .map_or(self.coefficients.len(), |n| n.offset as usize);
                &self.coefficients[c.offset as usize..end]
            }
            Err(_) => &[],
        }
    }

    /// All stored coefficients: the polynomial part, followed by the
    /// [factor coefficients](Self::factor_coefficients) of every factor in
    /// [Self::denominator_exponents] order.
    pub fn coefficients(&self) -> &[F::Element] {
        &self.coefficients
    }
}

impl<F: Field> PartialFractionRing<F>
where
    MultivariatePolynomial<F, u16, LexOrder>: Factorize,
{
    /// Create a self-extending ring over `field` in `variable`. The `hints`
    /// are factored into monic irreducible factors that seed the basis (their
    /// order fixes the first factor indices); every denominator that later
    /// enters the ring (from a conversion, [Ring::try_inv], [Field::inv] or
    /// [Field::div]) is factored once and its new irreducible factors are
    /// appended to the basis. Sums and products never create new factors, so
    /// arithmetic never factors.
    pub fn new(field: F, variable: Arc<PolyVariable>, hints: &[UnivariatePolynomial<F>]) -> Self {
        let ring = Self::from_monic_factors(field, variable, vec![], Some(factorize::<F>))
            .expect("an empty basis");
        for hint in hints {
            let mut c = hint.coefficients.clone();
            dense::trim(&ring.field, &mut c);
            if c.len() > 1 {
                dense::make_monic(&ring.field, &mut c);
                ring.denominator_factors(&c)
                    .expect("irreducible factors extend the basis");
            }
        }
        ring
    }

    /// Create a self-extending ring seeded with the irreducible factors of
    /// the given univariate denominators.
    pub fn from_denominators<E: PositiveExponent>(
        field: F,
        variable: Arc<PolyVariable>,
        denominators: &[MultivariatePolynomial<F, E>],
    ) -> Result<Self, PartialFractionError> {
        let ring = Self::new(field, variable, &[]);
        for denominator in denominators {
            if denominator.is_zero() {
                return Err(PartialFractionError::ZeroDenominator);
            }
            if denominator.is_constant() {
                continue;
            }
            let var = Self::variable_index(&ring.variable, denominator)?;
            let mut d = denominator.to_univariate_from_univariate(var).coefficients;
            dense::trim(&ring.field, &mut d);
            if d.len() > 1 {
                dense::make_monic(&ring.field, &mut d);
                ring.denominator_factors(&d)
                    .expect("irreducible factors extend the basis");
            }
        }
        Ok(ring)
    }
}

/// How a source is scaled when it is added to a destination.
#[derive(Clone, Copy)]
enum Scale<'a, E> {
    One,
    MinusOne,
    By(&'a E),
}

impl<F: Field> PartialFractionRing<F> {
    /// Create a ring with a fixed basis: the given factors, made monic, in
    /// this order. They must be nonconstant and pairwise coprime, but need not
    /// be irreducible. An element whose denominator is not a product of basis
    /// factors cannot enter the ring (an error, or `None` from [Ring::try_inv]
    /// and [Ring::try_div]).
    ///
    /// Use a fixed basis to keep the same structure in several related rings,
    /// for example the images of one basis modulo several primes (see
    /// [Self::map_field]): a factor that is irreducible over `Q` may split
    /// modulo a prime, so factoring again per prime would give a different
    /// basis.
    pub fn with_fixed_basis(
        field: F,
        variable: Arc<PolyVariable>,
        factors: &[UnivariatePolynomial<F>],
    ) -> Result<Self, PartialFractionError> {
        let factors = factors
            .iter()
            .map(|f| {
                let mut c = f.coefficients.clone();
                dense::trim(&field, &mut c);
                dense::make_monic(&field, &mut c);
                c
            })
            .collect();
        Self::from_monic_factors(field, variable, factors, None)
    }

    /// Create the ring from monic factors given as ascending coefficient lists.
    fn from_monic_factors(
        field: F,
        variable: Arc<PolyVariable>,
        factors: Vec<Vec<F::Element>>,
        factorizer: Option<Factorizer<F>>,
    ) -> Result<Self, PartialFractionError> {
        let basis = FactorBasis::new(factorizer);
        for (i, f) in factors.into_iter().enumerate() {
            match basis.push(&field, f) {
                Ok(j) if j == i => {}
                // A repeated factor.
                Ok(j) => return Err(PartialFractionError::NotCoprime(j, i)),
                Err(PartialFractionError::ConstantFactor(_)) => {
                    return Err(PartialFractionError::ConstantFactor(i));
                }
                Err(e) => return Err(e),
            }
        }
        Ok(Self {
            field,
            extend: basis.factorizer.is_some(),
            variable,
            basis: Arc::new(basis),
        })
    }

    /// A ring sharing this ring's basis that never extends it: a denominator
    /// with a factor outside the current basis is refused.
    pub fn strict(&self) -> Self {
        Self {
            extend: false,
            ..self.clone()
        }
    }

    /// Return `true` iff new denominators extend the basis.
    pub fn is_self_extending(&self) -> bool {
        self.extend && self.basis.factorizer.is_some()
    }

    /// The factor exponents of the monic, nonconstant `d` over the basis, or
    /// `None` if `d` has a factor outside the basis and the basis is not
    /// extended. Factorizations are cached.
    fn denominator_factors(&self, d: &[F::Element]) -> Option<Vec<(usize, usize)>> {
        if let Some(e) = self.basis.denominators.lock().unwrap().get(d) {
            return Some(e.clone());
        }
        let field = &self.field;
        let mut exponents: Vec<(usize, usize)> = vec![];
        match self.basis.factorizer.filter(|_| self.extend) {
            Some(factorizer) => {
                for (f, m) in factorizer(field, d) {
                    // Irreducible factors are coprime to every other factor.
                    let i = self.basis.push(field, f).ok()?;
                    match exponents.iter_mut().find(|(j, _)| *j == i) {
                        Some((_, e)) => *e += m,
                        None => exponents.push((i, m)),
                    }
                }
            }
            None => {
                let mut rest = d.to_vec();
                for i in 0..self.basis.len() {
                    let f = self.basis.monic(i as u32);
                    let mut e = 0;
                    while rest.len() >= f.len() {
                        let (q, r) = dense::divrem_monic(field, &rest, f);
                        if !r.is_empty() {
                            break;
                        }
                        rest = q;
                        e += 1;
                    }
                    if e > 0 {
                        exponents.push((i, e));
                    }
                }
                if rest.len() > 1 {
                    return None;
                }
            }
        }
        exponents.sort_unstable();
        self.basis
            .denominators
            .lock()
            .unwrap()
            .insert(d.to_vec(), exponents.clone());
        Some(exponents)
    }

    /// The position of `variable` in the variables of `p`, checking that `p`
    /// depends on no other variable.
    fn variable_index<R: Ring, E: PositiveExponent>(
        variable: &PolyVariable,
        p: &MultivariatePolynomial<R, E>,
    ) -> Result<usize, PartialFractionError> {
        let var = p.variables().iter().position(|v| v == variable);
        for t in p {
            for (i, e) in t.exponents.iter().enumerate() {
                if Some(i) != var && !e.is_zero() {
                    return Err(PartialFractionError::OtherVariable);
                }
            }
        }
        Ok(var.unwrap_or(0))
    }

    /// The coefficient field.
    pub fn field(&self) -> &F {
        &self.field
    }

    /// The variable.
    pub fn variable(&self) -> &Arc<PolyVariable> {
        &self.variable
    }

    /// The number of basis factors.
    ///
    /// A self-extending basis may grow; existing factor indices never change.
    pub fn factor_count(&self) -> usize {
        self.basis.len()
    }

    /// The monic basis factor `index`.
    pub fn factor(&self, index: usize) -> UnivariatePolynomial<F> {
        UnivariatePolynomial::from_coefficients(
            &self.field,
            self.basis.monic(index as u32).to_vec(),
            self.variable.clone(),
        )
    }

    fn polynomial(&self, coefficients: Vec<F::Element>) -> UnivariatePolynomial<F> {
        UnivariatePolynomial::from_coefficients(&self.field, coefficients, self.variable.clone())
    }

    #[inline(always)]
    fn chain_len(&self, c: &Chain) -> usize {
        c.exponent as usize * self.basis.degree(c.factor)
    }

    /// The degree of the denominator of `a`: `sum_i e_i deg f_i`.
    pub fn denominator_degree(&self, a: &PartialFraction<F>) -> usize {
        a.coefficients.len() - a.poly
    }

    /// Construct an element from its partial-fraction coefficients: the
    /// polynomial part and, for basis factors, the blocks `a_1, ..., a_e` as
    /// returned by [PartialFraction::factor_coefficients].
    ///
    /// Returns `None` if a factor index is out of range or repeated, or if a
    /// block length is not a multiple of the factor's degree.
    pub fn element_from_coefficients(
        &self,
        polynomial: Vec<F::Element>,
        factors: Vec<(usize, Vec<F::Element>)>,
    ) -> Option<PartialFraction<F>> {
        let mut factors = factors;
        factors.sort_by_key(|(f, _)| *f);
        let mut a = PartialFraction {
            poly: polynomial.len(),
            chains: Vec::with_capacity(factors.len()),
            coefficients: polynomial,
        };
        for (i, (factor, block)) in factors.into_iter().enumerate() {
            if factor >= self.factor_count() || (i > 0 && a.chains[i - 1].factor == factor as u32) {
                return None;
            }
            let degree = self.basis.degree(factor as u32);
            if block.len() % degree != 0 {
                return None;
            }
            a.chains.push(Chain {
                factor: factor as u32,
                exponent: (block.len() / degree) as u32,
                offset: a.coefficients.len() as u32,
            });
            a.coefficients.extend(block);
        }
        a.chains.retain(|c| c.exponent > 0);
        self.normalize(&mut a);
        Some(a)
    }

    /// The polynomial `p` as an element.
    pub fn element_from_polynomial(&self, p: &UnivariatePolynomial<F>) -> PartialFraction<F> {
        self.element_from_dense(p.coefficients.clone())
    }

    fn element_from_dense(&self, mut p: Vec<F::Element>) -> PartialFraction<F> {
        dense::trim(&self.field, &mut p);
        PartialFraction {
            poly: p.len(),
            chains: vec![],
            coefficients: p,
        }
    }

    /// The element `1 / f^exponent` for the basis factor `factor`.
    pub fn inv_factor_power(&self, factor: usize, exponent: usize) -> PartialFraction<F> {
        let mut a = self.one();
        self.div_factor_power_assign(&mut a, factor, exponent);
        a
    }

    /// Return `true` iff `a` is a constant.
    pub fn is_constant(&self, a: &PartialFraction<F>) -> bool {
        a.chains.is_empty() && a.poly <= 1
    }

    /// Return `true` iff the layout of `a` holds a polynomial part of length
    /// `poly` and every factor power in `chains`.
    fn covers(&self, a: &PartialFraction<F>, poly: usize, chains: &[Chain]) -> bool {
        if poly > a.poly {
            return false;
        }
        let mut targets = a.chains.iter();
        'next: for c in chains {
            for t in targets.by_ref() {
                if t.factor == c.factor {
                    if t.exponent < c.exponent {
                        return false;
                    }
                    continue 'next;
                }
                if t.factor > c.factor {
                    return false;
                }
            }
            return false;
        }
        true
    }

    /// Widen the layout of `a` in place to hold a polynomial part of length
    /// `poly` and every factor power in `chains` (whose offsets are ignored).
    /// New coefficients are zero. Reuses the storage of `a` when its capacity
    /// suffices.
    fn widen(&self, a: &mut PartialFraction<F>, poly: usize, chains: &[Chain]) {
        let new_poly = a.poly.max(poly);
        // The union of the factor lists.
        let (mut i, mut j, mut count, mut total) = (0, 0, 0, new_poly);
        while i < a.chains.len() || j < chains.len() {
            let (factor, exponent) = match (a.chains.get(i), chains.get(j)) {
                (Some(x), Some(y)) if x.factor == y.factor => {
                    i += 1;
                    j += 1;
                    (x.factor, x.exponent.max(y.exponent))
                }
                (Some(x), Some(y)) if x.factor < y.factor => {
                    i += 1;
                    (x.factor, x.exponent)
                }
                (Some(x), None) => {
                    i += 1;
                    (x.factor, x.exponent)
                }
                (_, Some(y)) => {
                    j += 1;
                    (y.factor, y.exponent)
                }
                (None, None) => unreachable!(),
            };
            count += 1;
            total += exponent as usize * self.basis.degree(factor);
        }
        let old_count = a.chains.len();
        a.chains.resize(count, Chain::default());
        a.coefficients.resize(total, self.field.zero());
        // Place the blocks from the back. Every block moves to a position at or
        // after its old one; moved-from positions receive zeros (by induction
        // every destination position holds a zero before the swap).
        let (mut i, mut j, mut k, mut end) = (old_count, chains.len(), count, total);
        while k > 0 {
            let old = (i > 0).then(|| a.chains[i - 1]);
            let new = (j > 0).then(|| chains[j - 1]);
            let (factor, exponent, moved) = match (old, new) {
                (Some(x), Some(y)) if x.factor == y.factor => {
                    i -= 1;
                    j -= 1;
                    (x.factor, x.exponent.max(y.exponent), Some(x))
                }
                (Some(x), Some(y)) if x.factor > y.factor => {
                    i -= 1;
                    (x.factor, x.exponent, Some(x))
                }
                (Some(x), None) => {
                    i -= 1;
                    (x.factor, x.exponent, Some(x))
                }
                (_, Some(y)) => {
                    j -= 1;
                    (y.factor, y.exponent, None)
                }
                (None, None) => unreachable!(),
            };
            let offset = end - exponent as usize * self.basis.degree(factor);
            if let Some(x) = moved {
                let from = x.offset as usize;
                if from != offset {
                    for t in (0..self.chain_len(&x)).rev() {
                        a.coefficients.swap(from + t, offset + t);
                    }
                }
            }
            k -= 1;
            a.chains[k] = Chain {
                factor,
                exponent,
                offset: offset as u32,
            };
            end = offset;
        }
        debug_assert_eq!(end, new_poly);
        a.poly = new_poly;
    }

    /// Restore the canonical form: drop trailing zero coefficients of the
    /// polynomial part and zero top blocks of every factor, in place.
    fn normalize(&self, a: &mut PartialFraction<F>) {
        let field = &self.field;
        let mut poly = a.poly;
        while poly > 0 && field.is_zero(&a.coefficients[poly - 1]) {
            poly -= 1;
        }
        let mut changed = poly != a.poly;
        let mut write = poly;
        let mut kept = 0;
        for index in 0..a.chains.len() {
            let c = a.chains[index];
            let degree = self.basis.degree(c.factor);
            let from = c.offset as usize;
            let mut exponent = c.exponent as usize;
            while exponent > 0
                && a.coefficients[from + (exponent - 1) * degree..from + exponent * degree]
                    .iter()
                    .all(|x| field.is_zero(x))
            {
                exponent -= 1;
            }
            if exponent == 0 {
                changed = true;
                continue;
            }
            let len = exponent * degree;
            if from != write {
                changed = true;
                for t in 0..len {
                    a.coefficients.swap(write + t, from + t);
                }
            }
            changed |= exponent != c.exponent as usize;
            a.chains[kept] = Chain {
                factor: c.factor,
                exponent: exponent as u32,
                offset: write as u32,
            };
            kept += 1;
            write += len;
        }
        if changed || write != a.coefficients.len() {
            a.chains.truncate(kept);
            a.coefficients.truncate(write);
            a.poly = poly;
        }
    }

    /// `a += scale * b`.
    fn add_scaled_assign(
        &self,
        a: &mut PartialFraction<F>,
        b: &PartialFraction<F>,
        scale: Scale<'_, F::Element>,
    ) {
        if b.is_zero() {
            return;
        }
        if let Scale::By(c) = scale
            && self.field.is_zero(c)
        {
            return;
        }
        let field = &self.field;
        if a.is_zero() {
            a.poly = b.poly;
            a.chains.clear();
            a.chains.extend_from_slice(&b.chains);
            a.coefficients.clear();
            match scale {
                Scale::One => a.coefficients.extend_from_slice(&b.coefficients),
                Scale::MinusOne => a
                    .coefficients
                    .extend(b.coefficients.iter().map(|x| field.neg(x))),
                Scale::By(c) => a
                    .coefficients
                    .extend(b.coefficients.iter().map(|x| field.mul(x, c))),
            }
            return;
        }
        if !self.covers(a, b.poly, &b.chains) {
            self.widen(a, b.poly, &b.chains);
        }
        let add = |d: &mut [F::Element], s: &[F::Element]| match scale {
            Scale::One => {
                for (d, s) in d.iter_mut().zip(s) {
                    field.add_assign(d, s);
                }
            }
            Scale::MinusOne => {
                for (d, s) in d.iter_mut().zip(s) {
                    field.sub_assign(d, s);
                }
            }
            Scale::By(c) => {
                for (d, s) in d.iter_mut().zip(s) {
                    field.add_mul_assign(d, s, c);
                }
            }
        };
        add(&mut a.coefficients[..b.poly], &b.coefficients[..b.poly]);
        let mut targets = a.chains.iter();
        for c in &b.chains {
            let t = targets
                .by_ref()
                .find(|t| t.factor == c.factor)
                .expect("the layout covers the source");
            let len = self.chain_len(c);
            let (from, to) = (c.offset as usize, t.offset as usize);
            add(
                &mut a.coefficients[to..to + len],
                &b.coefficients[from..from + len],
            );
        }
        self.normalize(a);
    }

    /// `a *= c` for a field element `c`.
    pub fn mul_scalar_assign(&self, a: &mut PartialFraction<F>, c: &F::Element) {
        if self.field.is_zero(c) {
            *a = PartialFraction::default();
            return;
        }
        for x in &mut a.coefficients {
            self.field.mul_assign(x, c);
        }
    }

    /// `a *= x`, where `x` is the variable of the ring.
    pub fn mul_variable_assign(&self, a: &mut PartialFraction<F>) {
        if a.is_zero() {
            return;
        }
        let poly = a.poly;
        self.widen(a, poly + 1, &[]);
        self.mul_variable_in_layout(a, poly);
        self.normalize(a);
    }

    /// `a *= x` inside the layout of `a`, whose polynomial part has
    /// `logical_poly < a.poly` used coefficients. Zero top blocks of the
    /// factors stay zero.
    fn mul_variable_in_layout(&self, a: &mut PartialFraction<F>, logical_poly: usize) {
        debug_assert!(logical_poly < a.poly);
        let field = &self.field;
        a.coefficients[..logical_poly + 1].rotate_right(1);
        for index in 0..a.chains.len() {
            let c = a.chains[index];
            let o = c.offset as usize;
            let e = c.exponent as usize;
            if let Some(r) = self.basis.root(c.factor) {
                // x a_j / (x - r)^j = a_j / (x - r)^(j-1) + r a_j / (x - r)^j
                let first = a.coefficients[o].clone();
                for j in 0..e {
                    let mut value = field.mul(&a.coefficients[o + j], r);
                    if j + 1 < e {
                        field.add_assign(&mut value, &a.coefficients[o + j + 1]);
                    }
                    a.coefficients[o + j] = value;
                }
                field.add_assign(&mut a.coefficients[0], &first);
            } else {
                // x a_j = top f + rest: rest / f^j + top / f^(j-1).
                let f = self.basis.monic(c.factor);
                let k = f.len() - 1;
                for j in 0..e {
                    let base = o + j * k;
                    let top = a.coefficients[base + k - 1].clone();
                    if field.is_zero(&top) {
                        a.coefficients[base..base + k].rotate_right(1);
                        continue;
                    }
                    for i in (1..k).rev() {
                        let mut value = a.coefficients[base + i - 1].clone();
                        field.sub_mul_assign(&mut value, &top, &f[i]);
                        a.coefficients[base + i] = value;
                    }
                    a.coefficients[base] = field.neg(&field.mul(&top, &f[0]));
                    let target = if j == 0 { 0 } else { base - k };
                    field.add_assign(&mut a.coefficients[target], &top);
                }
            }
        }
    }

    /// `a * p` for a polynomial `p` given by its ascending coefficients.
    fn mul_dense(&self, a: &PartialFraction<F>, p: &[F::Element]) -> PartialFraction<F> {
        self.mul_dense_fraction(a, p, &[])
    }

    /// `a * p / prod_i f_i^e_i`, computed in one final layout: the products
    /// `a x^i` by Horner's rule and then one in-place division per factor
    /// power, without intermediate allocations or normalizations.
    fn mul_dense_fraction(
        &self,
        a: &PartialFraction<F>,
        p: &[F::Element],
        denominator: &[(usize, usize)],
    ) -> PartialFraction<F> {
        let field = &self.field;
        let Some(top) = p.iter().rposition(|c| !field.is_zero(c)) else {
            return PartialFraction::default();
        };
        if a.is_zero() {
            return PartialFraction::default();
        }
        // The final layout: a's factors and the denominator's, with the
        // exponents added, and room for the polynomial part of a p.
        let mut chains = Vec::with_capacity(a.chains.len() + denominator.len());
        let mut sorted: Vec<(u32, u32)> = denominator
            .iter()
            .filter(|(_, e)| *e > 0)
            .map(|&(f, e)| {
                assert!(f < self.factor_count(), "factor index out of range");
                (f as u32, e as u32)
            })
            .collect();
        sorted.sort_unstable();
        // Merge repeated factors.
        sorted.dedup_by(|next, kept| {
            if next.0 == kept.0 {
                kept.1 += next.1;
                true
            } else {
                false
            }
        });
        let (mut i, mut j) = (0, 0);
        let mut offset = a.poly + top;
        while i < a.chains.len() || j < sorted.len() {
            let (factor, exponent) = match (a.chains.get(i), sorted.get(j)) {
                (Some(x), Some(&(g, e))) if x.factor == g => {
                    i += 1;
                    j += 1;
                    (g, x.exponent + e)
                }
                (Some(x), Some(&(g, _))) if x.factor < g => {
                    i += 1;
                    (x.factor, x.exponent)
                }
                (Some(x), None) => {
                    i += 1;
                    (x.factor, x.exponent)
                }
                (_, Some(&(g, e))) => {
                    j += 1;
                    (g, e)
                }
                (None, None) => unreachable!(),
            };
            chains.push(Chain {
                factor,
                exponent,
                offset: offset as u32,
            });
            offset += exponent as usize * self.basis.degree(factor);
        }
        let mut r = PartialFraction {
            poly: a.poly + top,
            chains,
            coefficients: vec![field.zero(); offset],
        };
        // Where every block of a lives in r.
        let mut segments = Vec::with_capacity(a.chains.len() + 1);
        segments.push((0, 0, a.poly));
        let mut targets = r.chains.iter();
        for c in &a.chains {
            let t = targets
                .by_ref()
                .find(|t| t.factor == c.factor)
                .expect("the layout holds a's factors");
            segments.push((c.offset as usize, t.offset as usize, self.chain_len(c)));
        }
        let add = |r: &mut PartialFraction<F>, c: &F::Element| {
            for &(from, to, len) in &segments {
                for (d, s) in r.coefficients[to..to + len]
                    .iter_mut()
                    .zip(&a.coefficients[from..from + len])
                {
                    field.add_mul_assign(d, s, c);
                }
            }
        };
        add(&mut r, &p[top]);
        let mut logical = a.poly;
        for c in p[..top].iter().rev() {
            self.mul_variable_in_layout(&mut r, logical);
            logical += 1;
            if !field.is_zero(c) {
                add(&mut r, c);
            }
        }
        r.poly = logical;
        // One in-place division per factor power.
        let mut scratch = vec![];
        for &(g, e) in &sorted {
            let position = r
                .chains
                .binary_search_by_key(&g, |c| c.factor)
                .expect("the layout holds the denominator");
            let start = a
                .chains
                .binary_search_by_key(&g, |c| c.factor)
                .map_or(0, |i| a.chains[i].exponent as usize);
            let base = start;
            for step in 0..e as usize {
                self.divide_once(&mut r, position, base + step, &mut scratch);
            }
        }
        self.normalize(&mut r);
        r
    }

    /// `a * p` for a polynomial `p`.
    pub fn mul_polynomial(
        &self,
        a: &PartialFraction<F>,
        p: &UnivariatePolynomial<F>,
    ) -> PartialFraction<F> {
        self.mul_dense(a, &p.coefficients)
    }

    /// `a * p / prod_i f_i^e_i` for a polynomial `p` and basis factor powers
    /// `(i, e_i)`: the product with a rational function whose denominator is
    /// given in factored form over the basis. This is faster than a product
    /// with the element `p / prod_i f_i^e_i`.
    pub fn mul_fraction(
        &self,
        a: &PartialFraction<F>,
        p: &UnivariatePolynomial<F>,
        denominator: &[(usize, usize)],
    ) -> PartialFraction<F> {
        self.mul_dense_fraction(a, &p.coefficients, denominator)
    }

    /// `a / f^exponent` for the basis factor `f` with index `factor`.
    pub fn div_factor_power(
        &self,
        a: &PartialFraction<F>,
        factor: usize,
        exponent: usize,
    ) -> PartialFraction<F> {
        let mut r = a.clone();
        self.div_factor_power_assign(&mut r, factor, exponent);
        r
    }

    /// `a /= f^exponent` for the basis factor `f` with index `factor`.
    pub fn div_factor_power_assign(
        &self,
        a: &mut PartialFraction<F>,
        factor: usize,
        exponent: usize,
    ) {
        assert!(factor < self.factor_count(), "factor index out of range");
        if a.is_zero() || exponent == 0 {
            return;
        }
        // Room for every new power at once.
        let current = a
            .chains
            .binary_search_by_key(&(factor as u32), |c| c.factor)
            .map_or(0, |i| a.chains[i].exponent);
        self.widen(
            a,
            0,
            &[Chain {
                factor: factor as u32,
                exponent: current + exponent as u32,
                offset: 0,
            }],
        );
        // The logical exponents of the factors while dividing.
        let position = a
            .chains
            .binary_search_by_key(&(factor as u32), |c| c.factor)
            .expect("the widened layout holds the factor");
        let mut scratch = vec![];
        for step in 0..exponent {
            self.divide_once(a, position, current as usize + step, &mut scratch);
        }
        self.normalize(a);
    }

    /// Divide `a` by the factor `g` of chain `position`, whose block has room
    /// for one more power than its logical exponent `g_exponent`. Every other
    /// chain is used at its full length (zero top blocks are harmless).
    fn divide_once(
        &self,
        a: &mut PartialFraction<F>,
        position: usize,
        g_exponent: usize,
        scratch: &mut Vec<F::Element>,
    ) {
        let field = &self.field;
        let basis = &*self.basis;
        let g_chain = a.chains[position];
        let g = g_chain.factor;
        let k = basis.degree(g);
        let g_offset = g_chain.offset as usize;
        // a_j -> a_(j+1); the new a_1 collects the residues.
        a.coefficients[g_offset..g_offset + (g_exponent + 1) * k].rotate_right(k);
        let g_root = basis.root(g);
        // Polynomial part: P = Q g + R.
        let poly = a.poly;
        if poly > 0 {
            if let Some(s) = g_root {
                let mut carry = a.coefficients[poly - 1].clone();
                for i in (0..poly - 1).rev() {
                    let mut next = a.coefficients[i].clone();
                    field.add_mul_assign(&mut next, &carry, s);
                    a.coefficients[i] = std::mem::replace(&mut carry, next);
                }
                a.coefficients[poly - 1] = field.zero();
                field.add_assign(&mut a.coefficients[g_offset], &carry);
            } else {
                let (q, r) = dense::divrem_monic(field, &a.coefficients[..poly], basis.monic(g));
                for i in 0..poly {
                    a.coefficients[i] = q.get(i).cloned().unwrap_or_else(|| field.zero());
                }
                for (i, r) in r.iter().enumerate() {
                    field.add_assign(&mut a.coefficients[g_offset + i], r);
                }
            }
        }
        for index in 0..a.chains.len() {
            if index == position {
                continue;
            }
            let chain = a.chains[index];
            let offset = chain.offset as usize;
            let e = chain.exponent as usize;
            if g_root.is_some() && basis.root(chain.factor).is_some() {
                // a_i / ((x - r)^i (x - s)) with t = s - r:
                // a'_e = -a_e / t, a'_i = (a'_(i+1) - a_i) / t, residue at s: -a'_1.
                let inverse = &basis.inverse_difference(field, g, chain.factor);
                let mut next = field.zero();
                for i in (0..e).rev() {
                    field.sub_assign(&mut next, &a.coefficients[offset + i]);
                    field.mul_assign(&mut next, inverse);
                    a.coefficients[offset + i] = next.clone();
                }
                field.sub_assign(&mut a.coefficients[g_offset], &next);
                continue;
            }
            let degree = basis.degree(chain.factor);
            scratch.clear();
            scratch.extend_from_slice(&a.coefficients[offset..offset + e * degree]);
            let c = self.divide_chain(chain.factor, e, scratch, g);
            a.coefficients[offset..offset + e * degree].clone_from_slice(scratch);
            for (i, c) in c.iter().enumerate() {
                field.add_assign(&mut a.coefficients[g_offset + i], c);
            }
        }
        a.poly = poly.saturating_sub(k);
    }

    /// Divide the chain `a = (a_1, ..., a_e)` of `h` (that is `A / h^e` with
    /// `A = sum_j a_j h^(e-j)`) by the basis factor `g` coprime to `h`:
    /// `A / (h^e g) = B / h^e + C / g` with `C = A (h^e)^-1 mod g` and
    /// `B = (A - C h^e) / g`. Overwrites `a` with the chain of `B`, returns `C`.
    fn divide_chain(&self, h: u32, e: usize, a: &mut [F::Element], g: u32) -> Vec<F::Element> {
        let field = &self.field;
        let basis = &*self.basis;
        let hm = basis.monic(h);
        let gm = basis.monic(g);
        let degree = hm.len() - 1;
        let mut dense_a = vec![];
        for j in 0..e {
            let mut block = a[j * degree..(j + 1) * degree].to_vec();
            dense::trim(field, &mut block);
            dense_a = dense::add(field, &dense::mul(field, &dense_a, hm), &block);
        }
        let inverse = basis.inverse_mod(h, g);
        let mut inverse_power = vec![field.one()];
        for _ in 0..e {
            inverse_power =
                dense::divrem_monic(field, &dense::mul(field, &inverse_power, inverse), gm).1;
        }
        let residue = dense::divrem_monic(field, &dense_a, gm).1;
        let c = dense::divrem_monic(field, &dense::mul(field, &residue, &inverse_power), gm).1;
        let h_power = dense::pow(field, hm, e);
        let numerator = dense::sub(field, &dense_a, &dense::mul(field, &c, &h_power));
        let (mut b, remainder) = dense::divrem_monic(field, &numerator, gm);
        debug_assert!(
            remainder.is_empty(),
            "partial-fraction division left a remainder"
        );
        for x in a.iter_mut() {
            *x = field.zero();
        }
        for j in (0..e).rev() {
            let (q, r) = dense::divrem_monic(field, &b, hm);
            for (i, r) in r.into_iter().enumerate() {
                a[j * degree + i] = r;
            }
            b = q;
        }
        debug_assert!(b.is_empty());
        c
    }

    /// Reduced numerator and monic denominator of `a`, as ascending
    /// coefficient lists.
    fn to_dense_fraction(&self, a: &PartialFraction<F>) -> (Vec<F::Element>, Vec<F::Element>) {
        let field = &self.field;
        let mut numerator = a.coefficients[..a.poly].to_vec();
        let mut denominator = vec![field.one()];
        for c in &a.chains {
            let h = self.basis.monic(c.factor);
            let degree = h.len() - 1;
            let o = c.offset as usize;
            let mut dense_a = vec![];
            for j in 0..c.exponent as usize {
                let mut block = a.coefficients[o + j * degree..o + (j + 1) * degree].to_vec();
                dense::trim(field, &mut block);
                dense_a = dense::add(field, &dense::mul(field, &dense_a, h), &block);
            }
            let power = dense::pow(field, h, c.exponent as usize);
            numerator = dense::add(
                field,
                &dense::mul(field, &numerator, &power),
                &dense::mul(field, &dense_a, &denominator),
            );
            denominator = dense::mul(field, &denominator, &power);
        }
        // Coprime: the top block of every factor is nonzero, so the numerator
        // does not vanish modulo any factor of the denominator.
        (numerator, denominator)
    }

    /// The reduced numerator and monic denominator of `a`.
    pub fn to_fraction(
        &self,
        a: &PartialFraction<F>,
    ) -> (UnivariatePolynomial<F>, UnivariatePolynomial<F>) {
        let (n, d) = self.to_dense_fraction(a);
        (self.polynomial(n), self.polynomial(d))
    }

    /// The element `numerator / denominator`, or an error if the reduced
    /// denominator has a factor outside the basis.
    pub fn try_from_fraction(
        &self,
        numerator: &UnivariatePolynomial<F>,
        denominator: &UnivariatePolynomial<F>,
    ) -> Result<PartialFraction<F>, PartialFractionError> {
        let mut n = numerator.coefficients.clone();
        let mut d = denominator.coefficients.clone();
        dense::trim(&self.field, &mut n);
        dense::trim(&self.field, &mut d);
        self.element_from_dense_fraction(n, d)
    }

    fn element_from_dense_fraction(
        &self,
        mut n: Vec<F::Element>,
        mut d: Vec<F::Element>,
    ) -> Result<PartialFraction<F>, PartialFractionError> {
        let field = &self.field;
        if d.is_empty() {
            return Err(PartialFractionError::ZeroDenominator);
        }
        if n.is_empty() {
            return Ok(PartialFraction::default());
        }
        // A monic denominator.
        let inverse = field.inv(d.last().unwrap());
        for c in n.iter_mut().chain(d.iter_mut()) {
            field.mul_assign(c, &inverse);
        }
        let exponents = if d.len() == 1 {
            vec![]
        } else if let Some(exponents) = self.denominator_factors(&d) {
            exponents
        } else {
            // A factor outside the basis must cancel against the numerator.
            let g = dense::gcd(field, &n, &d);
            n = dense::divrem_monic(field, &n, &g).0;
            d = dense::divrem_monic(field, &d, &g).0;
            if d.len() == 1 {
                vec![]
            } else {
                self.denominator_factors(&d)
                    .ok_or(PartialFractionError::OutsideBasis)?
            }
        };
        let mut a = self.element_from_dense(n);
        for (i, e) in exponents {
            self.div_factor_power_assign(&mut a, i, e);
        }
        Ok(a)
    }

    /// Evaluate `a` at `x`. Returns `None` if `x` is a pole of `a`.
    pub fn evaluate(&self, a: &PartialFraction<F>, x: &F::Element) -> Option<F::Element> {
        let field = &self.field;
        let mut value = dense::evaluate(field, &a.coefficients[..a.poly], x);
        for c in &a.chains {
            let f = self.basis.monic(c.factor);
            let fx = dense::evaluate(field, f, x);
            if field.is_zero(&fx) {
                return None;
            }
            let inverse = field.inv(&fx);
            let degree = f.len() - 1;
            let o = c.offset as usize;
            let mut chain = field.zero();
            for j in (0..c.exponent as usize).rev() {
                let block = &a.coefficients[o + j * degree..o + (j + 1) * degree];
                field.add_assign(&mut chain, &dense::evaluate(field, block, x));
                field.mul_assign(&mut chain, &inverse);
            }
            field.add_assign(&mut value, &chain);
        }
        Some(value)
    }

    /// The same basis over another field, with every factor coefficient mapped
    /// by `map`, for example the image modulo a prime.
    ///
    /// Fails if a coefficient has no image ([PartialFractionError::UnmappableCoefficient],
    /// for example a denominator divisible by the prime) or if the images are
    /// not pairwise coprime ([PartialFractionError::NotCoprime]). Since the
    /// factors are monic, their degrees are preserved. On success, the
    /// coefficient-wise image of an element ([Self::map_element]) is the image
    /// of the rational function it represents.
    pub fn map_field<G: Field>(
        &self,
        field: G,
        map: impl Fn(&F::Element) -> Option<G::Element>,
    ) -> Result<PartialFractionRing<G>, PartialFractionError> {
        let factors = self
            .basis
            .factors
            .iter()
            .map(|f| f.monic.iter().map(&map).collect::<Option<Vec<_>>>())
            .collect::<Option<Vec<_>>>()
            .ok_or(PartialFractionError::UnmappableCoefficient)?;
        PartialFractionRing::from_monic_factors(field, self.variable.clone(), factors, None)
    }

    /// The coefficient-wise image of `a` in `target`, a ring obtained from
    /// this one with [Self::map_field] and the same `map`.
    pub fn map_element<G: Field>(
        &self,
        target: &PartialFractionRing<G>,
        a: &PartialFraction<F>,
        map: impl Fn(&F::Element) -> Option<G::Element>,
    ) -> Result<PartialFraction<G>, PartialFractionError> {
        debug_assert_eq!(self.factor_count(), target.factor_count());
        let coefficients = a
            .coefficients
            .iter()
            .map(map)
            .collect::<Option<Vec<_>>>()
            .ok_or(PartialFractionError::UnmappableCoefficient)?;
        let mut r = PartialFraction {
            poly: a.poly,
            chains: a.chains.clone(),
            coefficients,
        };
        target.normalize(&mut r);
        Ok(r)
    }

    /// Convert a univariate polynomial with coefficients in `R` to an ascending
    /// coefficient list over `F`.
    fn dense_from_polynomial<R: EmbedInField<F>, E: PositiveExponent>(
        &self,
        p: &MultivariatePolynomial<R, E>,
    ) -> Result<Vec<F::Element>, PartialFractionError> {
        let var = Self::variable_index(&self.variable, p)?;
        let mut r = vec![];
        for t in p {
            let e = if t.exponents.is_empty() {
                0
            } else {
                t.exponents[var].to_i32() as usize
            };
            if r.len() <= e {
                r.resize(e + 1, self.field.zero());
            }
            r[e] = p.ring().embed(&self.field, t.coefficient);
        }
        dense::trim(&self.field, &mut r);
        Ok(r)
    }

    /// The element equal to `r`, or an error if the denominator of `r` has a
    /// factor outside the basis or if `r` depends on another variable.
    pub fn try_from_rational_polynomial<R: EmbedInField<F>, E: PositiveExponent>(
        &self,
        r: &RationalPolynomial<R, E>,
    ) -> Result<PartialFraction<F>, PartialFractionError> {
        let n = self.dense_from_polynomial(&r.numerator)?;
        let d = self.dense_from_polynomial(&r.denominator)?;
        self.element_from_dense_fraction(n, d)
    }

    /// The element equal to `r`, or an error if a denominator factor of `r` is
    /// not a product of basis factors or if `r` depends on another variable.
    pub fn try_from_factorized_rational_polynomial<R: EmbedInField<F>, E: PositiveExponent>(
        &self,
        r: &FactorizedRationalPolynomial<R, E>,
    ) -> Result<PartialFraction<F>, PartialFractionError> {
        let field = &self.field;
        let ring = r.numerator.ring();
        let mut n = self.dense_from_polynomial(&r.numerator)?;
        let scale = field.div(
            &ring.embed(field, &r.numer_coeff),
            &ring.embed(field, &r.denom_coeff),
        );
        for c in &mut n {
            field.mul_assign(c, &scale);
        }
        let mut d = vec![field.one()];
        for (f, e) in &r.denominators {
            d = dense::mul(
                field,
                &d,
                &dense::pow(field, &self.dense_from_polynomial(f)?, *e),
            );
        }
        self.element_from_dense_fraction(n, d)
    }

    /// `a` as a rational polynomial over the ring `ring`, in the ring's variable.
    pub fn to_rational_polynomial<R: Ring, E: PositiveExponent>(
        &self,
        a: &PartialFraction<F>,
        ring: &R,
    ) -> RationalPolynomial<R, E>
    where
        RationalPolynomial<R, E>: FromNumeratorAndDenominator<F, R, E>,
    {
        let (n, d) = self.to_fraction(a);
        RationalPolynomial::from_num_den(n.to_multivariate(), d.to_multivariate(), ring, false)
    }

    /// `a` as a rational polynomial with a factorized denominator over the
    /// ring `ring`, whose denominator factors are basis factors.
    pub fn to_factorized_rational_polynomial<R: Ring, E: PositiveExponent>(
        &self,
        a: &PartialFraction<F>,
        ring: &R,
    ) -> FactorizedRationalPolynomial<R, E>
    where
        FactorizedRationalPolynomial<R, E>: FromNumeratorAndFactorizedDenominator<F, R, E>,
    {
        let (n, _) = self.to_fraction(a);
        let denominators = a
            .chains
            .iter()
            .map(|c| {
                (
                    self.factor(c.factor as usize).to_multivariate(),
                    c.exponent as usize,
                )
            })
            .collect();
        FactorizedRationalPolynomial::from_num_den(n.to_multivariate(), denominators, ring, false)
    }
}

impl PartialFractionRing<RationalField> {
    /// The image of the ring modulo the prime of `field`.
    ///
    /// Fails if the prime divides a denominator of a factor coefficient or if
    /// the images of the factors are not pairwise coprime: the prime is bad
    /// for this basis.
    pub fn to_finite_field<UField: FiniteFieldWorkspace>(
        &self,
        field: &FiniteField<UField>,
    ) -> Result<PartialFractionRing<FiniteField<UField>>, PartialFractionError>
    where
        Integer: ToFiniteField<UField>,
        FiniteField<UField>: FiniteFieldCore<UField>,
    {
        self.map_field(field.clone(), |c| rational_to_finite_field(c, field))
    }

    /// The image of `a` in `target`, the image of this ring modulo a prime
    /// from [Self::to_finite_field]. Fails if the prime divides a coefficient
    /// denominator of `a`.
    pub fn element_to_finite_field<UField: FiniteFieldWorkspace>(
        &self,
        target: &PartialFractionRing<FiniteField<UField>>,
        a: &PartialFraction<RationalField>,
    ) -> Result<PartialFraction<FiniteField<UField>>, PartialFractionError>
    where
        Integer: ToFiniteField<UField>,
        FiniteField<UField>: FiniteFieldCore<UField>,
    {
        let field = target.field();
        self.map_element(target, a, |c| rational_to_finite_field(c, field))
    }
}

fn rational_to_finite_field<UField: FiniteFieldWorkspace>(
    c: &Rational,
    field: &FiniteField<UField>,
) -> Option<<FiniteField<UField> as Set>::Element>
where
    Integer: ToFiniteField<UField>,
    FiniteField<UField>: FiniteFieldCore<UField>,
{
    let d = c.denominator_ref().to_finite_field(field);
    if field.is_zero(&d) {
        return None;
    }
    Some(field.div(&c.numerator_ref().to_finite_field(field), &d))
}

/// A coefficient ring whose elements embed in the field `F`, used to convert
/// rational polynomials into a [PartialFractionRing] over `F`.
pub trait EmbedInField<F: Field>: Ring {
    /// The image of `c` in `field`.
    fn embed(&self, field: &F, c: &Self::Element) -> F::Element;
}

impl EmbedInField<RationalField> for IntegerRing {
    fn embed(&self, _field: &RationalField, c: &Integer) -> Rational {
        Rational::from(c)
    }
}

impl EmbedInField<RationalField> for RationalField {
    fn embed(&self, _field: &RationalField, c: &Rational) -> Rational {
        c.clone()
    }
}

impl<UField: FiniteFieldWorkspace> EmbedInField<FiniteField<UField>> for FiniteField<UField>
where
    FiniteField<UField>: FiniteFieldCore<UField>,
{
    fn embed(
        &self,
        _field: &FiniteField<UField>,
        c: &<FiniteField<UField> as Set>::Element,
    ) -> <FiniteField<UField> as Set>::Element {
        c.clone()
    }
}

impl<F: Field> Display for PartialFractionRing<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}[", self.field)?;
        write!(f, "{}", self.variable)?;
        f.write_char(']')?;
        if self.factor_count() > 0 {
            f.write_str("[")?;
            for i in 0..self.factor_count() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                f.write_str("1/")?;
                self.factor(i).format(
                    &PrintOptions::from_fmt(f),
                    PrintState::new().step(false, false, false, true),
                    f,
                )?;
            }
            f.write_str("]")?;
        }
        Ok(())
    }
}

impl<F: Field> Set for PartialFractionRing<F> {
    type Element = PartialFraction<F>;

    fn size(&self) -> Option<Integer> {
        None
    }
}

impl<F: Field> RingOps<PartialFraction<F>> for PartialFractionRing<F> {
    fn add(&self, mut a: Self::Element, b: Self::Element) -> Self::Element {
        self.add_scaled_assign(&mut a, &b, Scale::One);
        a
    }

    fn sub(&self, mut a: Self::Element, b: Self::Element) -> Self::Element {
        self.add_scaled_assign(&mut a, &b, Scale::MinusOne);
        a
    }

    fn mul(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        self.mul(&a, &b)
    }

    fn neg(&self, mut a: Self::Element) -> Self::Element {
        for x in &mut a.coefficients {
            *x = self.field.neg(&*x);
        }
        a
    }

    fn add_assign(&self, a: &mut Self::Element, b: Self::Element) {
        self.add_scaled_assign(a, &b, Scale::One);
    }

    fn sub_assign(&self, a: &mut Self::Element, b: Self::Element) {
        self.add_scaled_assign(a, &b, Scale::MinusOne);
    }

    fn mul_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a = self.mul(&*a, &b);
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: Self::Element, c: Self::Element) {
        self.add_mul_assign(a, &b, &c);
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: Self::Element, c: Self::Element) {
        self.sub_mul_assign(a, &b, &c);
    }
}

impl<F: Field> RingOps<&PartialFraction<F>> for PartialFractionRing<F> {
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        let mut r = a.clone();
        self.add_scaled_assign(&mut r, b, Scale::One);
        r
    }

    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        let mut r = a.clone();
        self.add_scaled_assign(&mut r, b, Scale::MinusOne);
        r
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        if a.is_zero() || b.is_zero() {
            return PartialFraction::default();
        }
        // Expand the operand with fewer partial fractions.
        let (a, b) = if a.chains.len() > b.chains.len() {
            (b, a)
        } else {
            (a, b)
        };
        if b.chains.is_empty() {
            return self.mul_dense(a, b.polynomial_part());
        }
        let field = &self.field;
        let mut result = self.mul_dense(b, a.polynomial_part());
        for c in &a.chains {
            // b * sum_j a_j / g^j by Horner in 1/g.
            let degree = self.basis.degree(c.factor);
            let o = c.offset as usize;
            let block = |j: usize| &a.coefficients[o + j * degree..o + (j + 1) * degree];
            let mut t = PartialFraction::default();
            for j in (0..c.exponent as usize).rev() {
                let coefficient = block(j);
                if degree == 1 {
                    self.add_scaled_assign(&mut t, b, Scale::By(&coefficient[0]));
                } else if coefficient.iter().any(|x| !field.is_zero(x)) {
                    let product = self.mul_dense(b, coefficient);
                    self.add_scaled_assign(&mut t, &product, Scale::One);
                }
                self.div_factor_power_assign(&mut t, c.factor as usize, 1);
            }
            self.add_scaled_assign(&mut result, &t, Scale::One);
        }
        result
    }

    fn neg(&self, a: &Self::Element) -> Self::Element {
        self.neg(a.clone())
    }

    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        self.add_scaled_assign(a, b, Scale::One);
    }

    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        self.add_scaled_assign(a, b, Scale::MinusOne);
    }

    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(&*a, b);
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        if self.is_constant(c) {
            if let Some(x) = c.coefficients.first() {
                self.add_scaled_assign(a, b, Scale::By(x));
            }
            return;
        }
        if self.is_constant(b) {
            if let Some(x) = b.coefficients.first() {
                self.add_scaled_assign(a, c, Scale::By(x));
            }
            return;
        }
        let product = self.mul(b, c);
        self.add_scaled_assign(a, &product, Scale::One);
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        let product = self.mul(b, c);
        self.add_scaled_assign(a, &product, Scale::MinusOne);
    }
}

impl<F: Field> Ring for PartialFractionRing<F> {
    fn zero(&self) -> Self::Element {
        PartialFraction::default()
    }

    fn one(&self) -> Self::Element {
        self.element_from_dense(vec![self.field.one()])
    }

    fn nth(&self, n: Integer) -> Self::Element {
        self.element_from_dense(vec![self.field.nth(n)])
    }

    fn pow(&self, b: &Self::Element, mut e: u64) -> Self::Element {
        let mut result = self.one();
        let mut base = b.clone();
        while e > 0 {
            if e & 1 == 1 {
                result = self.mul(&result, &base);
            }
            e >>= 1;
            if e > 0 {
                base = self.mul(&base, &base);
            }
        }
        result
    }

    fn is_zero(&self, a: &Self::Element) -> bool {
        a.is_zero()
    }

    fn is_one(&self, a: &Self::Element) -> bool {
        a.chains.is_empty() && a.poly == 1 && self.field.is_one(&a.coefficients[0])
    }

    fn one_is_gcd_unit() -> bool {
        true
    }

    fn characteristic(&self) -> Integer {
        self.field.characteristic()
    }

    /// Invert `a`. With a self-extending basis every nonzero element is
    /// invertible (the numerator of `a` is factored and its new factors are
    /// appended to the basis). With a fixed basis, `a` must be a unit: a
    /// nonzero constant times a product of integer powers of basis factors.
    fn try_inv(&self, a: &Self::Element) -> Option<Self::Element> {
        if a.is_zero() {
            return None;
        }
        let (n, d) = self.to_dense_fraction(a);
        self.element_from_dense_fraction(d, n).ok()
    }

    /// Divide `a` by `b`. With a fixed basis, the quotient must be in the
    /// ring.
    fn try_div(&self, a: &Self::Element, b: &Self::Element) -> Option<Self::Element> {
        if b.is_zero() {
            return None;
        }
        let field = &self.field;
        let (na, da) = self.to_dense_fraction(a);
        let (nb, db) = self.to_dense_fraction(b);
        self.element_from_dense_fraction(dense::mul(field, &na, &db), dense::mul(field, &da, &nb))
            .ok()
    }

    fn format<W: std::fmt::Write>(
        &self,
        element: &Self::Element,
        opts: &PrintOptions,
        mut state: PrintState,
        f: &mut W,
    ) -> Result<bool, Error> {
        if element.is_zero() {
            if state.in_sum {
                f.write_char('+')?;
            }
            f.write_char('0')?;
            return Ok(false);
        }
        let terms = usize::from(element.poly > 0)
            + element
                .chains
                .iter()
                .map(|c| {
                    let degree = self.basis.degree(c.factor);
                    (0..c.exponent as usize)
                        .filter(|j| {
                            element.coefficients[c.offset as usize + j * degree..][..degree]
                                .iter()
                                .any(|x| !self.field.is_zero(x))
                        })
                        .count()
                })
                .sum::<usize>();
        let parenthesize = terms > 1 && (state.in_product || state.in_exp || state.in_exp_base);
        if parenthesize {
            if state.in_sum {
                f.write_char('+')?;
            }
            f.write_char('(')?;
            state = PrintState::new();
        }
        let mut in_sum = state.in_sum;
        if element.poly > 0 {
            self.polynomial(element.polynomial_part().to_vec()).format(
                opts,
                state.step(in_sum, state.in_product, false, false),
                f,
            )?;
            in_sum = true;
        }
        for c in &element.chains {
            let degree = self.basis.degree(c.factor);
            let factor = self.factor(c.factor as usize);
            for j in 0..c.exponent as usize {
                let block = &element.coefficients[c.offset as usize + j * degree..][..degree];
                if block.iter().all(|x| self.field.is_zero(x)) {
                    continue;
                }
                let numerator = self.polynomial(block.to_vec());
                numerator.format(opts, state.step(in_sum, true, false, false), f)?;
                f.write_char('/')?;
                if j == 0 {
                    factor.format(opts, state.step(false, true, false, true), f)?;
                } else {
                    factor.format(opts, state.step(false, false, false, true), f)?;
                    write!(f, "^{}", j + 1)?;
                }
                in_sum = true;
            }
        }
        if parenthesize {
            f.write_char(')')?;
        }
        Ok(false)
    }
}

/// With a self-extending basis the ring is the field `F(x)`. Inversion and
/// division combine their operands into single fractions and factor the new
/// numerator (cached per denominator): this is much more expensive than the
/// other operations. Use [PartialFractionRing::div_factor_power] to divide by
/// basis factors.
impl<F: Field> EuclideanDomain for PartialFractionRing<F> {
    fn rem(&self, _a: &Self::Element, b: &Self::Element) -> Self::Element {
        assert!(!b.is_zero(), "division by zero");
        self.zero()
    }

    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        (self.div(a, b), self.zero())
    }

    fn gcd(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        if a.is_zero() && b.is_zero() {
            self.zero()
        } else {
            self.one()
        }
    }
}

/// See [EuclideanDomain] for the cost of inversion and division.
///
/// # Panics
///
/// [Field::inv] and [Field::div] panic on division by zero, and with a fixed
/// basis if the result is not in the ring (use [Ring::try_inv] and
/// [Ring::try_div] instead).
impl<F: Field> Field for PartialFractionRing<F> {
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        assert!(!b.is_zero(), "division by zero");
        self.try_div(a, b)
            .expect("the quotient has a denominator outside the fixed basis")
    }

    fn div_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.div(a, b);
    }

    fn inv(&self, a: &Self::Element) -> Self::Element {
        assert!(!a.is_zero(), "division by zero");
        self.try_inv(a)
            .expect("the inverse has a denominator outside the fixed basis")
    }
}

/// Sampling policy for a [PartialFractionRing].
///
/// The polynomial part has a degree bound sampled from `degree`, every basis
/// factor occurs with an exponent sampled from `exponent`, and every
/// coefficient is sampled with `coefficient`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PartialFractionSamplingPolicy<P> {
    /// Inclusive range from which the degree bound of the polynomial part is sampled.
    pub degree: RangeInclusive<usize>,
    /// Inclusive range from which the exponent of every basis factor is sampled.
    pub exponent: RangeInclusive<usize>,
    /// Policy used to sample every coefficient.
    pub coefficient: P,
}

impl<F: Field + SampleableRing> SampleableRing for PartialFractionRing<F> {
    type SamplingPolicy = PartialFractionSamplingPolicy<F::SamplingPolicy>;

    fn sample<G: rand::RngCore + ?Sized>(
        &self,
        rng: &mut G,
        policy: &Self::SamplingPolicy,
    ) -> Self::Element {
        let degree = rng.random_range(policy.degree.clone());
        let polynomial = (0..=degree)
            .map(|_| self.field.sample(rng, &policy.coefficient))
            .collect();
        let factors = (0..self.factor_count())
            .map(|i| {
                let e = rng.random_range(policy.exponent.clone());
                let block = (0..e * self.basis.degree(i as u32))
                    .map(|_| self.field.sample(rng, &policy.coefficient))
                    .collect();
                (i, block)
            })
            .collect();
        self.element_from_coefficients(polynomial, factors)
            .expect("valid sampled layout")
    }
}

#[cfg(test)]
mod tests;
