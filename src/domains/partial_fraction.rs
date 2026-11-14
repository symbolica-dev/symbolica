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
//! # Denominator bases
//!
//! Every ring can extend its append-only basis: a factor's index never changes,
//! so existing elements remain valid when new denominators enter.
//!
//! - [PartialFractionRing::new] uses [Factorize] to split new denominators.
//!   Known factors are trial-divided first; only the remaining cofactor is
//!   factored. Denominator decompositions are cached.
//! - [PartialFractionRing::with_basis] preserves supplied pairwise-coprime
//!   blocks, which need not be irreducible. It supports any coefficient field:
//!   uncovered denominators add coprime, possibly composite blocks without
//!   requiring polynomial factorization.
//! - [PartialFractionRing::map_field] preserves the mapped blocks as well,
//!   even when an irreducible rational factor splits modulo a prime.
//!
//! All these constructors produce the field F(x). Sums, products, and division
//! by basis factors do not extend the basis. General inversion and division
//! may do so. [PartialFractionLayout] separately defines fixed coordinates
//! for reconstruction; it does not restrict the arithmetic field.
//!
//! Factorization is supplied internally by an adapter to [Factorize], rather
//! than a bound on all arithmetic: outer polynomials over coefficient fields
//! such as rational-function fields do not necessarily implement that trait.
//!
//! Clones of a ring share its basis. As in Symbolica's symbol table, the
//! factors are stored in an append-only vector: reading them is lock-free,
//! and appending a factor (rare) takes a lock, so elements created on
//! different threads with clones of one ring stay compatible.
//!
//! # Modular reconstruction
//!
//! Before CRT or rational reconstruction, use a common denominator template.
//! [PartialFractionAlignment] converts differently ordered/split bases over
//! one field into the template's fixed blocks; [PartialFractionLayout] exports
//! zero-padded coordinates and maps those same blocks from `Q` to each prime.
//! Local irreducible-factor indices are not identities across primes. If the
//! rational denominator template is unknown, reconstruct it first; these APIs
//! do not infer rational factor grouping from unrelated modular factors.
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
use smallvec::SmallVec;

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

mod alignment;
pub use alignment::{
    PartialFractionAlignment, PartialFractionAlignmentError, PartialFractionLayout,
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
    /// Optional adapter to Factorize, installed by PartialFractionRing::new.
    /// When present, every stored block is irreducible. When absent, preserve
    /// composite blocks and extend by coprime remainders without factoring.
    factorizer: Option<Factorizer<F>>,
}

impl<F: Field> std::fmt::Debug for FactorBasis<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FactorBasis")
            .field(
                "factors",
                &self.factors.iter().map(|f| &f.monic).collect::<Vec<_>>(),
            )
            .field("factors_new_denominators", &self.factorizer.is_some())
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

/// The field of univariate rational functions over F, stored in partial-fraction
/// form over an append-only basis of monic, pairwise-coprime denominator blocks.
///
/// See the [module documentation](self) for the representation, the two
/// basis constructors and the sharing model. Clones share the append-only
/// basis and are cheap. Every ring implements [Field].
#[derive(Clone, Debug)]
pub struct PartialFractionRing<F: Field> {
    field: F,
    variable: Arc<PolyVariable>,
    basis: Arc<FactorBasis<F>>,
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
    /// [Field::div]) is trial-divided by the basis. Only the remaining cofactor
    /// is factored, and its new irreducible factors are appended. Denominator
    /// decompositions are cached. Sums and products never create new factors, so
    /// arithmetic never factors.
    pub fn new(field: F, variable: Arc<PolyVariable>, hints: &[UnivariatePolynomial<F>]) -> Self {
        let ring = Self::from_monic_factors(field, variable, vec![], Some(factorize::<F>))
            .expect("an empty basis");
        for hint in hints {
            ring.check_univariate(hint)
                .expect("hint uses another variable");
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

/// The products `a x, a x^2, ...` of one element `a` with powers of the
/// variable, shared by several [PartialFractionRing::mul_fraction_cached]
/// calls with the same `a`.
#[derive(Clone, Debug)]
pub struct PowerCache<F: Field> {
    powers: Vec<PartialFraction<F>>,
}

impl<F: Field> Default for PowerCache<F> {
    fn default() -> Self {
        Self { powers: vec![] }
    }
}

impl<F: Field> PowerCache<F> {
    /// An empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Forget the cached products, before the cache is used with another
    /// element.
    pub fn clear(&mut self) {
        self.powers.clear();
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
    /// Seed a field with the given blocks, made monic and kept in this order.
    /// They must be nonconstant and pairwise coprime, but need not be irreducible.
    /// Existing blocks are never split or replaced. New denominators are
    /// covered by powers of these blocks and any coprime remainder is appended
    /// as a whole block. No polynomial factorization implementation is needed.
    ///
    /// Use this constructor to preserve a reference denominator decomposition;
    /// use [PartialFractionLayout] when fixed reconstruction coordinates are
    /// required. All nonzero elements remain invertible.
    pub fn with_basis(
        field: F,
        variable: Arc<PolyVariable>,
        factors: &[UnivariatePolynomial<F>],
    ) -> Result<PartialFractionRing<F>, PartialFractionError> {
        for factor in factors {
            if factor.coefficients.len() > 1 && factor.variable != variable {
                return Err(PartialFractionError::OtherVariable);
            }
        }
        let factors = factors
            .iter()
            .map(|f| {
                let mut c = f.coefficients.clone();
                dense::trim(&field, &mut c);
                dense::make_monic(&field, &mut c);
                c
            })
            .collect();
        PartialFractionRing::<F>::from_monic_factors(field, variable, factors, None)
    }
}

impl<F: Field> PartialFractionRing<F> {
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
            variable,
            basis: Arc::new(basis),
        })
    }

    /// Exact factor exponents of a monic denominator. Known blocks are tried
    /// first. With the Factorize adapter, append irreducible remainder factors;
    /// otherwise return None for the general composite-cover path.
    fn denominator_factors(&self, d: &[F::Element]) -> Option<Vec<(usize, usize)>> {
        if let Some(e) = self.basis.denominators.lock().unwrap().get(d) {
            return Some(e.clone());
        }
        let field = &self.field;
        let mut exponents: Vec<(usize, usize)> = vec![];
        let mut rest = d.to_vec();
        // New products and powers of known factors need no factorization.
        // Concurrent insertions after this snapshot are reconciled by `push`.
        for i in 0..self.basis.len() {
            if rest.len() <= 1 {
                break;
            }
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
            let factorizer = self.basis.factorizer?;
            for (f, m) in factorizer(field, &rest) {
                // Irreducible factors are coprime to every other factor.
                let i = self.basis.push(field, f).ok()?;
                match exponents.iter_mut().find(|(j, _)| *j == i) {
                    Some((_, e)) => *e += m,
                    None => exponents.push((i, m)),
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

    /// The degree of the stored basis denominator: `sum_i e_i deg f_i`.
    /// For composite factors this may exceed the reduced denominator degree;
    /// [Self::to_fraction] removes any partial cancellations.
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

    fn check_univariate(&self, p: &UnivariatePolynomial<F>) -> Result<(), PartialFractionError> {
        if p.coefficients.len() > 1 && p.variable != self.variable {
            Err(PartialFractionError::OtherVariable)
        } else {
            Ok(())
        }
    }

    /// The polynomial `p` as an element.
    /// Panics if a nonconstant `p` uses another variable.
    pub fn element_from_polynomial(&self, p: &UnivariatePolynomial<F>) -> PartialFraction<F> {
        self.check_univariate(p)
            .expect("polynomial uses another variable");
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
        let scale = match scale {
            Scale::By(c) if field.is_one(c) => Scale::One,
            scale => scale,
        };
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
        // Only a block that b reaches at the top of a can become a zero top
        // block; otherwise a stays canonical.
        let mut top_touched = b.poly == a.poly;
        let mut targets = a.chains.iter();
        for c in &b.chains {
            let t = targets
                .by_ref()
                .find(|t| t.factor == c.factor)
                .expect("the layout covers the source");
            top_touched |= t.exponent == c.exponent;
            let len = self.chain_len(c);
            let (from, to) = (c.offset as usize, t.offset as usize);
            add(
                &mut a.coefficients[to..to + len],
                &b.coefficients[from..from + len],
            );
        }
        if top_touched {
            self.normalize(a);
        }
    }

    /// Accumulate products directly, without building a complete temporary
    /// product and then scanning it once more for add/sub-multiply.
    fn add_product_assign(
        &self,
        result: &mut PartialFraction<F>,
        a: &PartialFraction<F>,
        b: &PartialFraction<F>,
        subtract: bool,
    ) {
        if a.is_zero() || b.is_zero() {
            return;
        }
        let (a, b) = if a.len() > b.len() { (b, a) } else { (a, b) };
        let field = &self.field;
        let mut accumulate = |product: PartialFraction<F>| {
            if result.is_zero() {
                *result = if subtract { self.neg(product) } else { product };
            } else {
                self.add_scaled_assign(
                    result,
                    &product,
                    if subtract {
                        Scale::MinusOne
                    } else {
                        Scale::One
                    },
                );
            }
        };
        if a.poly > 0 {
            accumulate(self.mul_dense(b, a.polynomial_part()));
        }
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
            accumulate(t);
        }
    }

    /// `a *= c` for a field element `c`.
    pub fn mul_scalar_assign(&self, a: &mut PartialFraction<F>, c: &F::Element) {
        if self.field.is_zero(c) {
            *a = PartialFraction::default();
            return;
        }
        if self.field.is_one(c) {
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
        if p.len() <= 1 {
            if p.is_empty() || self.field.is_zero(&p[0]) {
                return PartialFraction::default();
            }
            let mut result = a.clone();
            self.mul_scalar_assign(&mut result, &p[0]);
            return result;
        }
        self.mul_dense_fraction(a, p, &[], None)
    }

    /// `a * p / prod_i f_i^e_i`, computed in one final layout: the products
    /// `a x^i` (by Horner's rule, or taken from `cache`) and then one in-place
    /// division per factor power, without intermediate normalizations.
    fn mul_dense_fraction(
        &self,
        a: &PartialFraction<F>,
        p: &[F::Element],
        denominator: &[(usize, usize)],
        cache: Option<&mut PowerCache<F>>,
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
        let mut sorted: SmallVec<[(u32, u32); 4]> = denominator
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
        let logical = match cache {
            Some(cache) => {
                // r = sum_i p_i (a x^i) with the cached a x^i.
                while cache.powers.len() < top {
                    let mut next = cache.powers.last().unwrap_or(a).clone();
                    self.mul_variable_assign(&mut next);
                    cache.powers.push(next);
                }
                for (i, c) in p[..=top].iter().enumerate() {
                    if !field.is_zero(c) {
                        let source = if i == 0 { a } else { &cache.powers[i - 1] };
                        self.add_scaled_in_layout(&mut r, source, c);
                    }
                }
                a.poly + top
            }
            None => {
                // Horner: ((p_n a) x + p_(n-1) a) x + ...
                self.add_scaled_in_layout(&mut r, a, &p[top]);
                let mut logical = a.poly;
                for c in p[..top].iter().rev() {
                    self.mul_variable_in_layout(&mut r, logical);
                    logical += 1;
                    if !field.is_zero(c) {
                        self.add_scaled_in_layout(&mut r, a, c);
                    }
                }
                logical
            }
        };
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

    /// `r += c * b` for a source `b` whose layout `r` covers (no widening,
    /// no normalization).
    fn add_scaled_in_layout(
        &self,
        r: &mut PartialFraction<F>,
        b: &PartialFraction<F>,
        c: &F::Element,
    ) {
        let field = &self.field;
        for (d, s) in r.coefficients[..b.poly]
            .iter_mut()
            .zip(&b.coefficients[..b.poly])
        {
            field.add_mul_assign(d, s, c);
        }
        let mut targets = r.chains.iter();
        for chain in &b.chains {
            let t = targets
                .by_ref()
                .find(|t| t.factor == chain.factor)
                .expect("the layout covers the source");
            let len = self.chain_len(chain);
            let (from, to) = (chain.offset as usize, t.offset as usize);
            for (d, s) in r.coefficients[to..to + len]
                .iter_mut()
                .zip(&b.coefficients[from..from + len])
            {
                field.add_mul_assign(d, s, c);
            }
        }
    }

    /// `a * p` for a polynomial `p`.
    pub fn mul_polynomial(
        &self,
        a: &PartialFraction<F>,
        p: &UnivariatePolynomial<F>,
    ) -> PartialFraction<F> {
        self.check_univariate(p)
            .expect("polynomial uses another variable");
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
        self.check_univariate(p)
            .expect("polynomial uses another variable");
        self.mul_dense_fraction(a, &p.coefficients, denominator, None)
    }

    /// [Self::mul_fraction] for many products of the same `a`: the products
    /// `a x^i` are computed once and kept in `cache`. The cache belongs to
    /// `a`: [clear](PowerCache::clear) it before using it with another
    /// element.
    pub fn mul_fraction_cached(
        &self,
        a: &PartialFraction<F>,
        p: &UnivariatePolynomial<F>,
        denominator: &[(usize, usize)],
        cache: &mut PowerCache<F>,
    ) -> PartialFraction<F> {
        self.check_univariate(p)
            .expect("polynomial uses another variable");
        self.mul_dense_fraction(a, &p.coefficients, denominator, Some(cache))
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
            // The final layout reserves poles for later divisions. Do not
            // run field arithmetic over their currently zero top blocks.
            let degree = basis.degree(chain.factor);
            let mut e = chain.exponent as usize;
            while e > 0
                && a.coefficients[offset + (e - 1) * degree..offset + e * degree]
                    .iter()
                    .all(|c| field.is_zero(c))
            {
                e -= 1;
            }
            if e == 0 {
                continue;
            }
            if g_root.is_some() && basis.root(chain.factor).is_some() {
                // a_i / ((x - r)^i (x - s)) with t = s - r:
                // a'_e = -a_e / t, a'_i = (a'_(i+1) - a_i) / t, residue at s: -a'_1.
                let inverse = &basis.inverse_difference(field, g, chain.factor);
                let mut next = field.zero();
                if field.is_one(inverse) {
                    // Avoid normalized multiplication by one in symbolic
                    // coefficient fields; only subtraction is required.
                    for i in (0..e).rev() {
                        field.sub_assign(&mut next, &a.coefficients[offset + i]);
                        a.coefficients[offset + i] = next.clone();
                    }
                } else if field.is_one(&field.neg(inverse)) {
                    // (next - a_i) * (-1) = a_i - next.
                    for i in (0..e).rev() {
                        next = field.sub(&a.coefficients[offset + i], &next);
                        a.coefficients[offset + i] = next.clone();
                    }
                } else {
                    for i in (0..e).rev() {
                        field.sub_assign(&mut next, &a.coefficients[offset + i]);
                        field.mul_assign(&mut next, inverse);
                        a.coefficients[offset + i] = next.clone();
                    }
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

    /// Divide one h-chain by a coprime factor g, without expanding h^e.
    /// Descend from the highest pole using s = g*b + h*carry, where
    /// b = s*(g^-1 mod h) mod h. The carry contributes to the next lower
    /// pole, and the final carry is the numerator over g. Intermediate
    /// degrees depend only on deg(h)+deg(g), not the pole exponent e.
    fn divide_chain(&self, h: u32, e: usize, a: &mut [F::Element], g: u32) -> Vec<F::Element> {
        let field = &self.field;
        let hm = self.basis.monic(h);
        let gm = self.basis.monic(g);
        let degree = hm.len() - 1;
        let inverse = self.basis.inverse_mod(g, h);
        let mut carry = vec![];
        for j in (0..e).rev() {
            let mut block = a[j * degree..(j + 1) * degree].to_vec();
            dense::trim(field, &mut block);
            let s = dense::add(field, &block, &carry);
            let residue = dense::divrem_monic(field, &s, hm).1;
            let b = dense::divrem_monic(field, &dense::mul(field, &residue, inverse), hm).1;
            let numerator = dense::sub(field, &s, &dense::mul(field, gm, &b));
            let (quotient, remainder) = dense::divrem_monic(field, &numerator, hm);
            debug_assert!(
                remainder.is_empty(),
                "partial-fraction recurrence left a remainder"
            );
            carry = quotient;
            for c in &mut a[j * degree..(j + 1) * degree] {
                *c = field.zero();
            }
            a[j * degree..j * degree + b.len()].clone_from_slice(&b);
        }
        debug_assert!(carry.len() < gm.len());
        carry
    }

    /// Numerator and monic basis denominator, before partial cancellation
    /// inside composite basis factors.
    fn to_dense_fraction_unreduced(
        &self,
        a: &PartialFraction<F>,
    ) -> (Vec<F::Element>, Vec<F::Element>) {
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
        // Relative to a composite basis factor, a nonzero top block can
        // still share a proper divisor with that factor.
        (numerator, denominator)
    }

    fn may_have_partial_cancellation(&self, a: &PartialFraction<F>) -> bool {
        self.basis.factorizer.is_none() && a.chains.iter().any(|c| self.basis.degree(c.factor) > 1)
    }

    fn has_partial_cancellation(&self, a: &PartialFraction<F>) -> bool {
        self.may_have_partial_cancellation(a)
            && a.chains.iter().any(|c| {
                let degree = self.basis.degree(c.factor);
                let top = c.offset as usize + (c.exponent as usize - 1) * degree;
                let mut numerator = a.coefficients[top..top + degree].to_vec();
                dense::trim(&self.field, &mut numerator);
                degree > 1
                    && dense::gcd(&self.field, &numerator, self.basis.monic(c.factor)).len() > 1
            })
    }

    fn to_dense_fraction(&self, a: &PartialFraction<F>) -> (Vec<F::Element>, Vec<F::Element>) {
        let (mut n, mut d) = self.to_dense_fraction_unreduced(a);
        if self.has_partial_cancellation(a) {
            let g = dense::gcd(&self.field, &n, &d);
            if g.len() > 1 {
                n = dense::divrem_monic(&self.field, &n, &g).0;
                d = dense::divrem_monic(&self.field, &d, &g).0;
            }
        }
        (n, d)
    }

    /// The reduced numerator and monic denominator of `a`.
    pub fn to_fraction(
        &self,
        a: &PartialFraction<F>,
    ) -> (UnivariatePolynomial<F>, UnivariatePolynomial<F>) {
        let (n, d) = self.to_dense_fraction(a);
        (self.polynomial(n), self.polynomial(d))
    }

    /// Import a fraction, extending the basis as needed.
    /// Fails for a zero denominator or incompatible variable.
    pub fn try_from_fraction(
        &self,
        numerator: &UnivariatePolynomial<F>,
        denominator: &UnivariatePolynomial<F>,
    ) -> Result<PartialFraction<F>, PartialFractionError> {
        self.check_univariate(numerator)?;
        self.check_univariate(denominator)?;
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
            // Reduce before adding new composite blocks or covering proper
            // divisors of existing blocks.
            let g = dense::gcd(field, &n, &d);
            n = dense::divrem_monic(field, &n, &g).0;
            d = dense::divrem_monic(field, &d, &g).0;
            if d.len() == 1 {
                vec![]
            } else {
                if let Some(exponents) = self.denominator_factors(&d) {
                    exponents
                } else {
                    // Localization at a composite factor also inverts each
                    // of its divisors: 1/g = (f/g)/f when g divides f.
                    let (exponents, multiplier) = self.extend_denominator_cover(&d)?;
                    n = dense::mul(field, &n, &multiplier);
                    exponents
                }
            }
        };
        let mut a = self.element_from_dense(n);
        for (i, e) in exponents {
            self.div_factor_power_assign(&mut a, i, e);
        }
        Ok(a)
    }

    /// Read-only coverage, used to validate an explicit reconstruction target.
    fn cover_denominator(
        &self,
        d: &[F::Element],
    ) -> Option<(Vec<(usize, usize)>, Vec<F::Element>)> {
        let (exponents, multiplier, rest) = self.denominator_cover_parts(d);
        (rest.len() <= 1).then_some((exponents, multiplier))
    }

    /// Strip all divisors supported by existing blocks. The returned multiplier
    /// satisfies d * multiplier = rest * product(f_i^e_i).
    fn denominator_cover_parts(
        &self,
        d: &[F::Element],
    ) -> (Vec<(usize, usize)>, Vec<F::Element>, Vec<F::Element>) {
        let field = &self.field;
        let mut rest = d.to_vec();
        let mut multiplier = vec![field.one()];
        let mut exponents = vec![];
        for i in 0..self.basis.len() {
            let f = self.basis.monic(i as u32);
            let mut e = 0;
            while rest.len() > 1 {
                let g = dense::gcd(field, &rest, f);
                if g.len() <= 1 {
                    break;
                }
                rest = dense::divrem_monic(field, &rest, &g).0;
                multiplier = dense::mul(field, &multiplier, &dense::divrem_monic(field, f, &g).0);
                e += 1;
            }
            if e > 0 {
                exponents.push((i, e));
            }
            if rest.len() <= 1 {
                break;
            }
        }
        (exponents, multiplier, rest)
    }

    /// Cover a denominator without refining existing composite blocks.
    fn extend_denominator_cover(
        &self,
        d: &[F::Element],
    ) -> Result<(Vec<(usize, usize)>, Vec<F::Element>), PartialFractionError> {
        loop {
            let (mut exponents, multiplier, rest) = self.denominator_cover_parts(d);
            if rest.len() > 1 {
                match self.basis.push(&self.field, rest) {
                    Ok(i) => exponents.push((i, 1)),
                    // Another thread appended an overlapping block after our
                    // snapshot. Recompute the cover against the enlarged basis.
                    Err(PartialFractionError::NotCoprime(..)) => continue,
                    Err(e) => return Err(e),
                }
            }
            if multiplier.len() == 1 && self.field.is_one(&multiplier[0]) {
                self.basis
                    .denominators
                    .lock()
                    .unwrap()
                    .insert(d.to_vec(), exponents.clone());
            }
            return Ok((exponents, multiplier));
        }
    }

    /// Evaluate `a` at `x`. Returns `None` if `x` is a pole of `a`.
    pub fn evaluate(&self, a: &PartialFraction<F>, x: &F::Element) -> Option<F::Element> {
        let field = &self.field;
        let mut value = dense::evaluate(field, &a.coefficients[..a.poly], x);
        for c in &a.chains {
            let f = self.basis.monic(c.factor);
            let fx = dense::evaluate(field, f, x);
            if field.is_zero(&fx) {
                if !self.may_have_partial_cancellation(a) {
                    return None;
                }
                // A zero of a composite basis factor may be removable.
                let (n, d) = self.to_dense_fraction(a);
                let denominator = dense::evaluate(field, &d, x);
                return (!field.is_zero(&denominator))
                    .then(|| field.div(&dense::evaluate(field, &n, x), &denominator));
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
        PartialFractionRing::<G>::from_monic_factors(field, self.variable.clone(), factors, None)
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

    /// Import a rational polynomial, extending the basis as needed.
    /// Returns an error for an incompatible variable or zero denominator.
    pub fn try_from_rational_polynomial<R: EmbedInField<F>, E: PositiveExponent>(
        &self,
        r: &RationalPolynomial<R, E>,
    ) -> Result<PartialFraction<F>, PartialFractionError> {
        let n = self.dense_from_polynomial(&r.numerator)?;
        let d = self.dense_from_polynomial(&r.denominator)?;
        self.element_from_dense_fraction(n, d)
    }

    /// Import a factorized rational polynomial, extending the basis as needed.
    /// Returns an error for an incompatible variable or zero denominator.
    pub fn try_from_factorized_rational_polynomial<R: EmbedInField<F>, E: PositiveExponent>(
        &self,
        r: &FactorizedRationalPolynomial<R, E>,
    ) -> Result<PartialFraction<F>, PartialFractionError> {
        let field = &self.field;
        let ring = r.numerator.ring();
        let mut n = self.dense_from_polynomial(&r.numerator)?;
        let denominator_coefficient = ring.embed(field, &r.denom_coeff);
        if field.is_zero(&denominator_coefficient) {
            return Err(PartialFractionError::ZeroDenominator);
        }
        let mut scale = field.div(&ring.embed(field, &r.numer_coeff), &denominator_coefficient);
        // Validate every factor before extending the basis, including when
        // the numerator is zero. Preserve the caller's factorization.
        let mut factors = Vec::with_capacity(r.denominators.len());
        for (f, e) in &r.denominators {
            let mut f = self.dense_from_polynomial(f)?;
            if *e == 0 {
                continue;
            }
            let lead = f.last().ok_or(PartialFractionError::ZeroDenominator)?;
            let inverse = field.inv(lead);
            field.mul_assign(&mut scale, &field.pow(&inverse, *e as u64));
            for c in &mut f {
                field.mul_assign(c, &inverse);
            }
            if f.len() > 1 {
                factors.push((f, *e));
            }
        }
        for c in &mut n {
            field.mul_assign(c, &scale);
        }
        dense::trim(field, &mut n);
        if n.is_empty() {
            return Ok(PartialFraction::default());
        }
        let mut exponents: Vec<(usize, usize)> = vec![];
        for (f, e) in &factors {
            let Some(resolved) = self.denominator_factors(f) else {
                // Composite blocks may cover only the reduced denominator, or
                // a multiple of it when a basis factor is reducible. Preserve
                // cancellation across supplied factors in this fallback.
                let d = factors.iter().fold(vec![field.one()], |d, (f, e)| {
                    dense::mul(field, &d, &dense::pow(field, f, *e))
                });
                return self.element_from_dense_fraction(n, d);
            };
            for (i, multiplicity) in resolved {
                let multiplicity = multiplicity * e;
                match exponents.iter_mut().find(|(j, _)| *j == i) {
                    Some((_, exponent)) => *exponent += multiplicity,
                    None => exponents.push((i, multiplicity)),
                }
            }
        }
        exponents.sort_unstable();
        let mut a = self.element_from_dense(n);
        for (i, e) in exponents {
            self.div_factor_power_assign(&mut a, i, e);
        }
        Ok(a)
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
    /// ring `ring`. Irreducible basis factors are reused; composite factors
    /// are split as needed to cancel shared numerator factors.
    pub fn to_factorized_rational_polynomial<R: Ring, E: PositiveExponent>(
        &self,
        a: &PartialFraction<F>,
        ring: &R,
    ) -> FactorizedRationalPolynomial<R, E>
    where
        FactorizedRationalPolynomial<R, E>: FromNumeratorAndFactorizedDenominator<F, R, E>,
    {
        let (n, _) = self.to_dense_fraction_unreduced(a);
        let n = self.polynomial(n);
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
        FactorizedRationalPolynomial::from_num_den(
            n.to_multivariate(),
            denominators,
            ring,
            self.has_partial_cancellation(a),
        )
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
        if a.chains.is_empty() {
            return self.mul_dense(b, a.polynomial_part());
        }
        if b.chains.is_empty() {
            return self.mul_dense(a, b.polynomial_part());
        }
        let mut result = self.zero();
        self.add_product_assign(&mut result, a, b, false);
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
        self.add_product_assign(a, b, c, false);
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.add_product_assign(a, b, c, true);
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

    /// Invert a nonzero element, extending the denominator basis as needed.
    fn try_inv(&self, a: &Self::Element) -> Option<Self::Element> {
        if a.is_zero() {
            return None;
        }
        let (n, d) = self.to_dense_fraction(a);
        self.element_from_dense_fraction(d, n).ok()
    }

    /// Divide by a nonzero element, extending the denominator basis as needed.
    fn try_div(&self, a: &Self::Element, b: &Self::Element) -> Option<Self::Element> {
        if b.is_zero() {
            return None;
        }
        if a.is_zero() {
            return Some(self.zero());
        }
        Some(self.mul(a, &self.try_inv(b)?))
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

/// Inversion combines an operand into a single fraction and extends the basis
/// for its numerator, using Factorize when available or retaining whole blocks.
/// Use [PartialFractionRing::div_factor_power] for cheaper division by known
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
/// [Field::inv] and [Field::div] panic on division by zero. All
/// rings implement [Field] and extend their denominator bases as needed.
impl<F: Field> Field for PartialFractionRing<F> {
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        assert!(!b.is_zero(), "division by zero");
        self.try_div(a, b)
            .expect("self-extending basis accepts every nonzero denominator")
    }

    fn div_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.div(a, b);
    }

    fn inv(&self, a: &Self::Element) -> Self::Element {
        assert!(!a.is_zero(), "division by zero");
        self.try_inv(a)
            .expect("self-extending basis accepts every nonzero denominator")
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

#[cfg(test)]
mod entry_tests;
#[cfg(test)]
mod regression_tests;
