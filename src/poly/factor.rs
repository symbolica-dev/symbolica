//! Factorization methods for multivariate polynomials
//! that implement [Factorize].

use std::{borrow::Cow, cmp::Reverse, ops::RangeInclusive};

use ahash::{HashMap, HashSet, HashSetExt};
use rand::{Rng, SeedableRng, rng, rngs::StdRng};
use tracing::debug;

use crate::{
    GLOBAL_SETTINGS,
    combinatorics::CombinationIterator,
    domains::{
        EuclideanDomain, Field, InternalOrdering, Ring, RingOps, SampleableRing, Set,
        algebraic::{AlgebraicExtension, GaloisField},
        finite_field::{
            FiniteField, FiniteFieldCore, FiniteFieldWorkspace, PrimeIteratorU64, ToFiniteField, Zp,
        },
        integer::{Integer, IntegerRing, Z, gcd_unsigned},
        rational::{Q, RationalField},
    },
    kernels::{DensePolynomialMulRequest, GeometricSequenceStepRequest},
    tensors::matrix::Vector,
};

use super::{
    LexOrder, PositiveExponent,
    gcd::{MAX_RNG_PREFACTOR, POW_CACHE_SIZE, PolynomialGCD},
    polynomial::MultivariatePolynomial,
};

const SPARSE_MDP_SAMPLE_BASE_ATTEMPTS: usize = 1;
// Maximum average number of input terms per bivariate degree-box position for
// selecting the bivariate-start integer factorization algorithm.
const INTEGER_FACTOR_BIVARIATE_SPARSE_BOX_DENSITY_THRESHOLD: f64 = 5.0;
// Number of failed univariate lifting attempts before automatic integer
// factorization retries the polynomial with the bivariate-start algorithm.
const INTEGER_FACTOR_UNIVARIATE_AUTO_RETRIES: usize = 3;
// Number of small deterministic coordinate blocks tried before random sampling.
const WANG_PRIME_SAMPLE_ATTEMPTS: usize = 3;
// Maximum number of retained coefficient cells across the target and factor images
// used by one evaluated Hensel stage.
const MAX_EVALUATED_HENSEL_IMAGE_CELLS: usize = 1 << 22;
// Maximum number of terms grouped into geometric sequences for one polynomial image.
const MAX_EVALUATED_HENSEL_GROUPED_TERMS: usize = 1 << 20;
// Minimum number of base-prime digits for using composite-modulus quadratic Hensel corrections.
const MIN_QUADRATIC_HENSEL_DIGITS: usize = 64;
// Maximum number of term advances across all geometric samples in one image rebuild.
const MAX_EVALUATED_HENSEL_TERM_STEPS: usize = 1 << 24;
// Maximum number of dense coefficient rows retained for either lifted factor.
const MAX_EVALUATED_HENSEL_Y_ROWS: usize = 1 << 16;

/// Distinct-degree blocks together with their exact number of irreducible factors.
struct DistinctDegreeFactorization<P> {
    blocks: Vec<(usize, P)>,
    factor_count: usize,
}

/// A suitable finite-field image whose equal-degree factorization has been deferred.
struct ModularIntegerFactorization<E: PositiveExponent> {
    field: Zp,
    distinct_degree: DistinctDegreeFactorization<MultivariatePolynomial<Zp, E, LexOrder>>,
}

/// Result of screening a degree-preserving, square-free finite-field image.
enum ModularPrimeScreen<E: PositiveExponent> {
    Candidate(ModularIntegerFactorization<E>),
    FactorLimitExceeded { lower_bound: usize },
}

#[cfg(test)]
std::thread_local! {
    pub(crate) static LLL_RECOMBINATION_SUCCESSES: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
    static QUADRATIC_HENSEL_LIFT_CALLS: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
    static MODULAR_INTEGER_EDF_CALLS: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
    static BOUNDED_DDF_REJECTIONS: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
    static LAST_BOUNDED_DDF_REJECTION_DEGREE: std::cell::Cell<usize> = const {
        std::cell::Cell::new(0)
    };
    static LAST_MODULAR_INTEGER_EDF_PRIME: std::cell::Cell<u32> = const {
        std::cell::Cell::new(0)
    };
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum IntegerFactorStart {
    Auto,
    Univariate,
    Bivariate,
    Disabled,
}

enum ExactPolynomialSquareRoot<P> {
    Root(P),
    NotSquare,
}

enum QuadraticFactorization<P> {
    Split([P; 2]),
    Irreducible,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SparseDiophantineFallback {
    Recursive,
    RetrySample,
}

/// Dense coefficient grid for a polynomial in two retained variables.
/// Coefficients are stored by increasing `y` degree, then increasing `x` degree.
#[derive(Clone, Debug, Eq, PartialEq)]
struct DenseBivariateImage<T> {
    x_len: usize,
    y_len: usize,
    coefficients: Vec<T>,
}

/// Nonzero coefficients of a univariate polynomial and their increasing degrees.
struct DenseIndexedUnivariate<T> {
    coefficients: Vec<T>,
    indices: Vec<u32>,
}

/// Coefficients below the omitted leading one of a monic univariate polynomial.
struct DenseMonicModulus<T> {
    lower_coefficients: Vec<T>,
}

/// Data reused to solve the two modular correction equations at one image point.
struct DenseTwoFactorCorrectionContext<T> {
    multipliers: [DenseIndexedUnivariate<T>; 2],
    moduli: [DenseMonicModulus<T>; 2],
}

impl<T> DenseBivariateImage<T> {
    #[inline]
    fn index(&self, x_degree: usize, y_degree: usize) -> usize {
        debug_assert!(x_degree < self.x_len);
        debug_assert!(y_degree < self.y_len);
        y_degree * self.x_len + x_degree
    }

    #[inline]
    fn coefficient(&self, x_degree: usize, y_degree: usize) -> &T {
        &self.coefficients[self.index(x_degree, y_degree)]
    }
}

/// Bézout coefficients for a pair of coprime univariate factor images.
struct TwoFactorImageBezout<P> {
    s: P,
    t: P,
}

/// Stores the geometric interpolation base and univariate Bézout coefficients
/// reused by the sparse Hensel corrections in one lifting stage.
struct SparseDiophantineContext<P, C> {
    two_factor_bezout: HashMap<(P, P), Option<TwoFactorImageBezout<P>>>,
    two_factor_base_points: Option<Vec<(usize, C)>>,
}

impl<P, C> SparseDiophantineContext<P, C> {
    fn new() -> Self {
        Self {
            two_factor_bezout: HashMap::default(),
            two_factor_base_points: None,
        }
    }

    fn clear_two_factor_images(&mut self) {
        self.two_factor_bezout.clear();
        self.two_factor_base_points = None;
    }
}

/// Controls where a multivariate Hensel lift starts and how it handles a
/// sparse correction that cannot be reconstructed exactly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct MultivariateHenselContext {
    start_index: usize,
    sparse_diophantine_fallback: SparseDiophantineFallback,
}

impl MultivariateHenselContext {
    /// Start lifting at `start_index` and use the recursive Taylor-quotient
    /// solver when sparse correction reconstruction fails.
    const fn new(start_index: usize) -> Self {
        Self {
            start_index,
            sparse_diophantine_fallback: SparseDiophantineFallback::Recursive,
        }
    }

    /// Return an error for a failed higher-dimensional sparse correction so
    /// that the caller can retry with a different evaluation sample. Evaluated
    /// stages may then certify the completed unshifted lift once at the end.
    const fn retry_sample_on_sparse_failure(mut self) -> Self {
        self.sparse_diophantine_fallback = SparseDiophantineFallback::RetrySample;
        self
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum MultivariateHenselError {
    SparseDiophantineFailed,
}

fn integer_factor_start_mode() -> IntegerFactorStart {
    match (
        GLOBAL_SETTINGS
            .use_univariate_factorization
            .load(std::sync::atomic::Ordering::Relaxed),
        GLOBAL_SETTINGS
            .use_bivariate_factorization
            .load(std::sync::atomic::Ordering::Relaxed),
    ) {
        (true, true) => IntegerFactorStart::Auto,
        (true, false) => IntegerFactorStart::Univariate,
        (false, true) => IntegerFactorStart::Bivariate,
        (false, false) => IntegerFactorStart::Disabled,
    }
}

/// A polynomial that can be factorized.
pub trait Factorize: Sized {
    /// Perform a square-free factorization.
    /// The output is `a_1^e1*...*a_n^e_n`
    /// where each `a_i` is relative prime.
    fn square_free_factorization(&self) -> Vec<(Self, usize)>;
    /// Factor a polynomial over its coefficient ring.
    fn factor(&self) -> Vec<(Self, usize)>;
    fn is_irreducible(&self) -> bool;
}

impl<F: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent>
    MultivariatePolynomial<F, E, LexOrder>
{
    /// Find factors that do not contain all variables.
    pub fn factor_separable(&self) -> Vec<Self> {
        let mut stripped = self.clone();

        let mut factors = vec![];
        for x in 0..self.nvars() {
            if stripped.degree(x) == E::zero() {
                continue;
            }

            let c = stripped.to_univariate_polynomial_list(x);
            let cs = c.into_iter().map(|x| x.0).collect();

            let gcd = PolynomialGCD::gcd_multiple(cs);

            if !gcd.is_constant() {
                stripped = stripped / &gcd;
                let mut fs = gcd.factor_separable();
                factors.append(&mut fs);
            }
        }

        factors.push(stripped);
        factors
    }

    /// Perform a square free factorization using Yun's algorithm.
    ///
    /// The characteristic of the ring must be 0 and all variables
    /// must occur in every factor.
    fn square_free_factorization_0_char(&self) -> Vec<(Self, usize)> {
        if self.is_constant() {
            if self.is_one() {
                return vec![];
            } else {
                return vec![(self.clone(), 1)];
            }
        }

        // any variable can be selected
        // select the one with the lowest degree
        let lowest_rank_var = (0..self.nvars())
            .filter_map(|x| {
                let d = self.degree(x);
                if d > E::zero() { Some((x, d)) } else { None }
            })
            .min_by_key(|a| a.1)
            .unwrap()
            .0;

        let b = self.derivative(lowest_rank_var);
        let c = self.gcd(&b);

        if c.is_one() {
            return vec![(self.clone(), 1)];
        }

        let mut factors = vec![];

        let mut w = self / &c;
        let mut y = &b / &c;

        let mut i = 1;
        while !w.is_constant() {
            let z = y - w.derivative(lowest_rank_var);
            let g = w.gcd(&z);
            w = w / &g;
            y = z / &g;

            if !g.is_one() {
                factors.push((g, i));
            }
            i += 1
        }

        factors
    }

    /// Use Newton's polygon method to test if a bivariate polynomial is irreducible.
    /// If this method returns `false`, the test is inconclusive.
    ///
    /// The polynomial must have overall factors of single variables removed.
    fn bivariate_irreducibility_test(&self) -> bool {
        /// Compute the convex hull via the Monotone chain algorithm.
        fn convex_hull(mut points: Vec<(isize, isize)>) -> Vec<(isize, isize)> {
            points.sort();
            if points.len() < 2 {
                return points;
            }

            // Cross product of o-a and o-b vectors, positive means ccw turn, negative means cw turn and 0 means collinear.
            fn cross(o: &(isize, isize), a: &(isize, isize), b: &(isize, isize)) -> isize {
                (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
            }

            let mut lower = vec![];
            let mut upper = vec![];

            for (t, rev) in [(&mut lower, false), (&mut upper, true)] {
                for i in 0..points.len() {
                    let p = if rev {
                        points[points.len() - 1 - i]
                    } else {
                        points[i]
                    };
                    while t.len() >= 2 && cross(&t[t.len() - 2], &t[t.len() - 1], &p) <= 0 {
                        t.pop();
                    }
                    t.push(p);
                }
            }

            lower.pop();
            upper.pop();
            lower.extend(upper);
            lower
        }

        let vars: Vec<_> = (0..self.nvars())
            .filter(|v| self.degree(*v) > E::zero())
            .collect();

        if vars.len() != 2 {
            return false;
        }

        let points = self
            .exponents
            .chunks(self.nvars())
            .map(|e| (e[vars[0]].to_u32() as isize, e[vars[1]].to_u32() as isize))
            .collect();

        let hull = convex_hull(points);

        match hull.len() {
            2 => {
                let x_deg = hull[0].0.abs_diff(hull[1].0);
                let y_deg = hull[0].1.abs_diff(hull[1].1);
                gcd_unsigned(x_deg as u64, y_deg as u64) == 1
            }
            3 => {
                // the hull has the form (n, 0), (0, m), (u, v)
                let (mut n, mut m, mut u, mut v) = (-1, -1, -1, -1);
                for (x, y) in hull {
                    if x != 0 && y == 0 {
                        n = x;
                    } else if y != 0 && x == 0 {
                        m = y;
                    } else {
                        u = x;
                        v = y;
                    }
                }

                n != -1
                    && m != -1
                    && u != -1
                    && v != -1
                    && gcd_unsigned(
                        gcd_unsigned(gcd_unsigned(n as u64, m as u64), u as u64),
                        v as u64,
                    ) == 1
            }
            _ => false,
        }
    }
}

impl<R: EuclideanDomain, E: PositiveExponent> MultivariatePolynomial<R, E, LexOrder> {
    /// Check if a parse lift is possible.
    #[allow(dead_code)]
    fn sparse_lift_possible(&self, factors: &[Self], order: &[usize]) -> bool {
        // check if all bivariate monomials occur in the product of factors
        let mut all_monomials = HashSet::with_capacity(self.nterms());
        for e in self.exponents.chunks(self.nvars()) {
            all_monomials.insert((e[order[0]], e[order[1]]));
        }

        let mut total = factors[0].clone();
        for f in &factors[1..] {
            total = &total * f;
        }

        let mut all_monomials_in_factors = HashSet::with_capacity(self.nterms());
        for e in total.exponents.chunks(total.nvars()) {
            all_monomials_in_factors.insert((e[order[0]], e[order[1]]));
        }

        all_monomials == all_monomials_in_factors
    }
}

/// Arithmetic modulo an integer on dense univariate integer polynomials.
///
/// Quadratic Hensel lifting uses this context for factor and Bezout corrections modulo the
/// current prime power. Polynomial products go through the integer multiplication dispatcher;
/// division uses the inverse of the divisor's unit leading coefficient.
struct IntegerModularUnivariateContext<'a, E: PositiveExponent> {
    modulus: &'a Integer,
    variable: usize,
    template: &'a MultivariatePolynomial<IntegerRing, E, LexOrder>,
}

impl<'a, E: PositiveExponent> IntegerModularUnivariateContext<'a, E> {
    fn new(
        modulus: &'a Integer,
        template: &'a MultivariatePolynomial<IntegerRing, E, LexOrder>,
    ) -> Self {
        assert!(!modulus.is_zero() && !modulus.is_negative());
        let variable = template
            .last_exponents()
            .iter()
            .position(|exponent| !exponent.is_zero())
            .expect("a Hensel factor must be nonconstant");
        debug_assert!(template.exponents_iter().all(|exponents| {
            exponents
                .iter()
                .enumerate()
                .all(|(index, exponent)| index == variable || exponent.is_zero())
        }));
        Self {
            modulus,
            variable,
            template,
        }
    }

    fn reduce(
        &self,
        polynomial: &MultivariatePolynomial<IntegerRing, E, LexOrder>,
    ) -> MultivariatePolynomial<IntegerRing, E, LexOrder> {
        polynomial.map_coeff(
            |coefficient| coefficient.clone().symmetric_mod(self.modulus),
            Z,
        )
    }

    fn multiply(
        &self,
        left: &MultivariatePolynomial<IntegerRing, E, LexOrder>,
        right: &MultivariatePolynomial<IntegerRing, E, LexOrder>,
    ) -> MultivariatePolynomial<IntegerRing, E, LexOrder> {
        self.reduce(&(left * right))
    }

    fn add(
        &self,
        left: &MultivariatePolynomial<IntegerRing, E, LexOrder>,
        right: &MultivariatePolynomial<IntegerRing, E, LexOrder>,
    ) -> MultivariatePolynomial<IntegerRing, E, LexOrder> {
        self.reduce(&(left + right))
    }

    fn dense_coefficients(
        &self,
        polynomial: &MultivariatePolynomial<IntegerRing, E, LexOrder>,
    ) -> Vec<Integer> {
        let degree = polynomial.degree(self.variable).to_u32() as usize;
        let mut coefficients = vec![Integer::zero(); degree + 1];
        for term in polynomial {
            debug_assert!(
                term.exponents
                    .iter()
                    .enumerate()
                    .all(|(index, exponent)| index == self.variable || exponent.is_zero())
            );
            coefficients[term.exponents[self.variable].to_u32() as usize] =
                term.coefficient.clone().symmetric_mod(self.modulus);
        }
        coefficients
    }

    fn from_dense_coefficients(
        &self,
        coefficients: Vec<Integer>,
    ) -> MultivariatePolynomial<IntegerRing, E, LexOrder> {
        let capacity = coefficients
            .iter()
            .filter(|coefficient| !coefficient.is_zero())
            .count();
        let mut polynomial = self.template.zero_with_capacity(capacity);
        let mut exponents = vec![E::zero(); self.template.nvars()];
        for (degree, coefficient) in coefficients.into_iter().enumerate() {
            if coefficient.is_zero() {
                continue;
            }
            exponents[self.variable] = E::from_u32(degree as u32);
            polynomial.append_monomial_back(coefficient, &exponents);
        }
        polynomial
    }

    /// Divide modulo the context modulus and return a canonical symmetric quotient and remainder.
    ///
    /// The divisor's leading coefficient must be invertible modulo the context modulus, as it is
    /// for the modular factors used during Hensel lifting.
    fn quot_rem(
        &self,
        dividend: &MultivariatePolynomial<IntegerRing, E, LexOrder>,
        divisor: &MultivariatePolynomial<IntegerRing, E, LexOrder>,
    ) -> (
        MultivariatePolynomial<IntegerRing, E, LexOrder>,
        MultivariatePolynomial<IntegerRing, E, LexOrder>,
    ) {
        assert!(!divisor.is_zero());
        if dividend.is_zero() {
            return (self.template.zero(), self.template.zero());
        }

        let mut remainder = self.dense_coefficients(dividend);
        let divisor = self.dense_coefficients(divisor);
        if remainder.len() < divisor.len() {
            return (
                self.template.zero(),
                self.from_dense_coefficients(remainder),
            );
        }

        let divisor_degree = divisor.len() - 1;
        let leading_inverse = divisor[divisor_degree].mod_inverse(self.modulus);
        let mut quotient = vec![Integer::zero(); remainder.len() - divisor_degree];
        for power in (divisor_degree..remainder.len()).rev() {
            let coefficient = (&remainder[power] * &leading_inverse).symmetric_mod(self.modulus);
            if coefficient.is_zero() {
                continue;
            }
            quotient[power - divisor_degree] = coefficient.clone();
            for (offset, divisor_coefficient) in divisor.iter().enumerate() {
                let index = power - divisor_degree + offset;
                Z.sub_mul_assign(&mut remainder[index], &coefficient, divisor_coefficient);
                let value = std::mem::replace(&mut remainder[index], Integer::zero());
                remainder[index] = value.symmetric_mod(self.modulus);
            }
            debug_assert!(remainder[power].is_zero());
        }

        remainder.truncate(divisor_degree);
        (
            self.from_dense_coefficients(quotient),
            self.from_dense_coefficients(remainder),
        )
    }
}

impl<E: PositiveExponent> MultivariatePolynomial<IntegerRing, E, LexOrder> {
    /// Return the exact square root of an integer polynomial when every
    /// square-free component occurs with even multiplicity.
    ///
    /// This is used for quadratic discriminants. The coefficient content is
    /// handled separately because square-free factorization keeps it as a
    /// constant rather than assigning it a multiplicity.
    fn square_root_from_square_free_decomposition(&self) -> ExactPolynomialSquareRoot<Self> {
        if self.is_zero() {
            return ExactPolynomialSquareRoot::Root(self.zero());
        }

        if self.lcoeff().is_negative() {
            return ExactPolynomialSquareRoot::NotSquare;
        }

        let content = self.content();
        if content.is_negative() {
            return ExactPolynomialSquareRoot::NotSquare;
        }

        let root_content = content.root(2);
        if &root_content * &root_content != content {
            return ExactPolynomialSquareRoot::NotSquare;
        }

        let primitive = self.clone().div_coeff(&content);
        let mut root = self.constant(root_content);
        for (factor, multiplicity) in primitive.square_free_factorization() {
            if factor.is_one() {
                continue;
            }
            if factor.is_constant() || multiplicity % 2 != 0 {
                return ExactPolynomialSquareRoot::NotSquare;
            }
            root = &root * &factor.pow(multiplicity / 2);
        }

        assert!(
            &root * &root == *self,
            "square-free decomposition reconstructed an invalid polynomial square root"
        );
        ExactPolynomialSquareRoot::Root(root)
    }

    /// Split `self = a*x_var^2 + b*x_var + c` into its three coefficient
    /// polynomials. All returned polynomials have degree zero in `var`.
    fn quadratic_coefficients(&self, var: usize) -> Option<(Self, Self, Self)> {
        if self.degree(var) != E::from_u32(2) {
            return None;
        }

        let mut coefficients = [self.zero(), self.zero(), self.zero()];
        for (coefficient, degree) in self.to_univariate_polynomial_list(var) {
            let degree = degree.to_u32() as usize;
            if degree > 2 {
                return None;
            }
            coefficients[degree] = coefficient;
        }

        let [c, b, a] = coefficients;
        if a.is_zero() { None } else { Some((a, b, c)) }
    }

    /// Estimate the multiplication work needed to form the discriminant
    /// `b^2 - 4*a*c` when `var` is used as the quadratic variable.
    fn quadratic_discriminant_cost(&self, var: usize) -> usize {
        let mut terms = [0usize; 3];
        for exponents in self.exponents_iter() {
            let degree = exponents[var].to_u32() as usize;
            if degree <= 2 {
                terms[degree] += 1;
            }
        }

        terms[1]
            .saturating_mul(terms[1])
            .saturating_add(terms[2].saturating_mul(terms[0]))
    }

    /// Factor a primitive square-free polynomial that is quadratic in `var` by
    /// taking the exact square root of its discriminant. Every nonconstant
    /// factor must depend on `var`, as guaranteed after `factor_separable`.
    ///
    /// For `a*x^2 + b*x + c`, let `h = (b + sqrt(b^2-4*a*c))/2` and
    /// `g = gcd(a, h)`. Exact division then reconstructs the two linear factors
    /// `(a/g*x + h/g)` and `(g*x + c/(h/g))`.
    fn factor_quadratic_variable(&self, var: usize) -> QuadraticFactorization<Self> {
        let (a, b, c) = self
            .quadratic_coefficients(var)
            .expect("quadratic factorization requires a degree-two variable");
        let discriminant = &b * &b - (&a * &c).mul_coeff(Integer::from(4));
        let square_root = match discriminant.square_root_from_square_free_decomposition() {
            ExactPolynomialSquareRoot::Root(root) => root,
            ExactPolynomialSquareRoot::NotSquare => {
                return QuadraticFactorization::Irreducible;
            }
        };
        assert!(
            !square_root.is_zero(),
            "a square-free quadratic cannot have zero discriminant"
        );
        let two = self.constant(Integer::from(2));

        for signed_root in [square_root.clone(), -square_root] {
            let h = (b.clone() + signed_root)
                .try_div_owned(&two)
                .expect("a quadratic discriminant root must have the parity of its linear term");

            let g = a.gcd(&h);
            assert!(!g.is_zero(), "the quadratic leading coefficient is nonzero");

            let a_over_g = a
                .clone()
                .try_div_owned(&g)
                .expect("a polynomial gcd must exactly divide its first argument");
            let h_over_g = h
                .try_div_owned(&g)
                .expect("a polynomial gcd must exactly divide its second argument");
            if h_over_g.is_zero() {
                continue;
            }
            let c_over_h = c
                .clone()
                .try_div_owned(&h_over_g)
                .expect("the normalized quadratic root must exactly divide the constant term");

            let mut first = a_over_g;
            for exponents in first.exponents_iter_mut() {
                debug_assert!(exponents[var].is_zero());
                exponents[var] = E::one();
            }
            first = first + h_over_g;

            let mut second = g;
            for exponents in second.exponents_iter_mut() {
                debug_assert!(exponents[var].is_zero());
                exponents[var] = E::one();
            }
            second = second + c_over_h;

            if first.lcoeff().is_negative() {
                first = -first;
                second = -second;
            }

            if &first * &second == *self {
                return QuadraticFactorization::Split([first, second]);
            }
        }

        panic!("quadratic discriminant factors failed exact reconstruction")
    }

    /// Factor using the degree-two variable whose discriminant has the smallest
    /// estimated coefficient-product count. After `factor_separable`, every
    /// irreducible factor has positive degree in this variable. Those degrees
    /// can therefore only partition two as `2` or `1 + 1`, so one quadratic
    /// variable decides whether the polynomial is irreducible or splits.
    fn factor_quadratic(&self, degrees: &[usize]) -> Option<QuadraticFactorization<Self>> {
        let candidate = degrees
            .iter()
            .enumerate()
            .filter_map(|(var, degree)| {
                (*degree == 2).then(|| (self.quadratic_discriminant_cost(var), var))
            })
            .min();

        candidate.map(|(_, var)| self.factor_quadratic_variable(var))
    }
}

impl<E: PositiveExponent> Factorize for MultivariatePolynomial<IntegerRing, E, LexOrder> {
    fn square_free_factorization(&self) -> Vec<(Self, usize)> {
        if self.is_zero() {
            return vec![];
        }

        let mut c = self.content();
        let stripped = self.clone().div_coeff(&c);

        let mut factors = vec![];

        let fs = stripped.factor_separable();

        for mut f in fs {
            // make sure f is primitive
            if f.lcoeff().is_negative() {
                c = -c;
                f = -f;
            }

            let mut nf = f.square_free_factorization_0_char();
            factors.append(&mut nf);
        }

        if !c.is_one() {
            factors.insert(0, (self.constant(c), 1));
        }

        if factors.is_empty() {
            factors.push((self.one(), 1))
        }

        factors
    }

    fn factor(&self) -> Vec<(Self, usize)> {
        let sf = self.square_free_factorization();

        let mut factors = vec![];
        let mut degrees = vec![0; self.nvars()];
        for (f, p) in sf {
            debug!("SFF {} {}", f, p);

            let mut var_count = 0;
            for (v, d) in degrees.iter_mut().enumerate() {
                *d = f.degree(v).to_u32() as usize;
                if *d > 0 {
                    var_count += 1;
                }
            }

            match var_count {
                0 | 1 => factors.extend(f.factor_reconstruct().into_iter().map(|ff| (ff, p))),
                2 => {
                    let mut order: Vec<_> = degrees
                        .iter()
                        .enumerate()
                        .filter(|(_, d)| **d > 0)
                        .collect();
                    order.sort_by_key(|o| Reverse(o.1));
                    let order: Vec<_> = order.into_iter().map(|(v, _)| v).collect();

                    factors.extend(
                        f.bivariate_factor_reconstruct(order[0], order[1])
                            .into_iter()
                            .map(|ff| (ff, p)),
                    )
                }
                _ => {
                    if integer_factor_start_mode() != IntegerFactorStart::Disabled {
                        match f.factor_quadratic(&degrees) {
                            Some(QuadraticFactorization::Split(quadratic_factors)) => {
                                factors.extend(quadratic_factors.into_iter().map(|ff| (ff, p)));
                                continue;
                            }
                            Some(QuadraticFactorization::Irreducible) => {
                                factors.push((f, p));
                                continue;
                            }
                            None => {}
                        }
                    }

                    // select the variable with the smallest leading coefficient and the highest degree to be first
                    let mut lcoeff_length = vec![0; self.nvars()];
                    for x in f.exponents_iter() {
                        for ((lc, e), d) in lcoeff_length.iter_mut().zip(x).zip(&degrees) {
                            if e.to_i32() as usize == *d {
                                *lc += 1;
                            }
                        }
                    }

                    let first = (0..self.nvars())
                        .min_by(|a, b| {
                            lcoeff_length[*a]
                                .cmp(&lcoeff_length[*b])
                                .then_with(|| degrees[*b].cmp(&degrees[*a]))
                        })
                        .unwrap();

                    // TODO: find better order
                    let mut order: Vec<_> = degrees
                        .iter()
                        .enumerate()
                        .filter(|(_, d)| **d > 0)
                        .collect();
                    order.sort_by_key(|o| {
                        if o.0 == first {
                            Reverse(&usize::MAX)
                        } else {
                            Reverse(o.1)
                        }
                    });

                    let mut order: Vec<_> = order.into_iter().map(|(v, _)| v).collect();

                    factors.extend(
                        f.multivariate_factorization(&mut order, 10, None)
                            .into_iter()
                            .map(|ff| (ff, p)),
                    )
                }
            }
        }

        factors
    }

    fn is_irreducible(&self) -> bool {
        let mut sf = self.square_free_factorization();
        if sf.len() > 1 {
            return false;
        }

        let (f, _) = sf.pop().unwrap();

        let mut degrees = vec![0; self.nvars()];
        let mut var_count = 0;
        for (v, d) in degrees.iter_mut().enumerate() {
            *d = f.degree(v).to_u32() as usize;
            if *d > 0 {
                var_count += 1;
            }
        }

        match var_count {
            0 | 1 => f.factor_reconstruct().len() == 1,
            2 => {
                let mut order: Vec<_> = degrees
                    .iter()
                    .enumerate()
                    .filter(|(_, d)| **d > 0)
                    .collect();
                order.sort_by_key(|o| Reverse(o.1));
                let order: Vec<_> = order.into_iter().map(|(v, _)| v).collect();

                f.bivariate_factor_reconstruct(order[0], order[1]).len() == 1
            }
            _ => {
                if integer_factor_start_mode() != IntegerFactorStart::Disabled {
                    match f.factor_quadratic(&degrees) {
                        Some(QuadraticFactorization::Split(_)) => return false,
                        Some(QuadraticFactorization::Irreducible) => return true,
                        None => {}
                    }
                }

                // TODO: find better order
                let mut order: Vec<_> = degrees
                    .iter()
                    .enumerate()
                    .filter(|(_, d)| **d > 0)
                    .collect();
                order.sort_by_key(|o| Reverse(o.1));

                let mut order: Vec<_> = order.into_iter().map(|(v, _)| v).collect();

                f.multivariate_factorization(&mut order, 10, None).len() == 1
            }
        }
    }
}

impl<E: PositiveExponent> Factorize for MultivariatePolynomial<RationalField, E, LexOrder> {
    fn square_free_factorization(&self) -> Vec<(Self, usize)> {
        if self.is_zero() {
            return vec![];
        }

        let c = self.content();

        let stripped = self.map_coeff(
            |coeff| {
                let coeff = self.ring().div(coeff, &c);
                debug_assert!(coeff.is_integer());
                coeff.numerator()
            },
            Z,
        );

        let fs = stripped.square_free_factorization();

        let mut factors: Vec<_> = fs
            .into_iter()
            .map(|(f, e)| (f.map_coeff(|coeff| coeff.into(), Q), e))
            .collect();

        if !c.is_one() {
            factors.push((self.constant(c), 1));
        }

        factors
    }

    fn factor(&self) -> Vec<(Self, usize)> {
        if self.is_zero() {
            return vec![];
        }

        let c = self.content();

        let stripped = self.map_coeff(
            |coeff| {
                let coeff = self.ring().div(coeff, &c);
                debug_assert!(coeff.is_integer());
                coeff.numerator()
            },
            Z,
        );

        let mut factors: Vec<_> = stripped
            .factor()
            .into_iter()
            .map(|(ff, p)| (ff.map_coeff(|coeff| coeff.into(), Q), p))
            .collect();

        if !c.is_one() {
            factors.push((self.constant(c), 1));
        }

        factors
    }

    fn is_irreducible(&self) -> bool {
        let c = self.content();

        let stripped = self.map_coeff(
            |coeff| {
                let coeff = self.ring().div(coeff, &c);
                debug_assert!(coeff.is_integer());
                coeff.numerator()
            },
            Z,
        );

        stripped.is_irreducible()
    }
}

impl<E: PositiveExponent> Factorize
    for MultivariatePolynomial<AlgebraicExtension<RationalField>, E, LexOrder>
{
    fn square_free_factorization(&self) -> Vec<(Self, usize)> {
        if self.is_zero() {
            return vec![];
        }

        let c = self.lcoeff();
        let stripped = self.clone().make_monic();

        let mut factors = vec![];

        let fs = stripped.factor_separable();

        for f in fs {
            let mut nf = f.square_free_factorization_0_char();
            factors.append(&mut nf);
        }

        if factors.is_empty() || !self.ring().is_one(&c) {
            factors.insert(0, (self.constant(c), 1));
        }

        factors
    }

    /// Perform Trager's algorithm for factorization.
    fn factor(&self) -> Vec<(Self, usize)> {
        if self.is_zero() {
            return vec![];
        }

        let sf = self.square_free_factorization();

        let mut constant = self.ring().one();
        let mut full_factors = vec![];
        for (f, p) in &sf {
            if f.is_constant() {
                self.ring()
                    .mul_assign(&mut constant, self.ring().pow(&f.get_constant(), *p as u64));
                continue;
            }

            let (v, s, g, n) = f.norm_with_shift_data();

            let mut factors = n.factor();
            factors.retain(|(f, _)| !f.is_constant());

            if factors.len() == 1 {
                full_factors.push((f.clone(), *p));
                continue;
            }

            let mut g_f = g.to_number_field(&self.ring());
            let alpha_poly = g.variable(&self.get_vars_ref()[v]).unwrap()
                + g.variable(&self.ring().poly().variables()[0]).unwrap()
                    * &g.constant((s as u64).into());
            let last_factor = factors.len() - 1;

            for (factor_index, (f, b)) in factors.into_iter().enumerate() {
                debug!("Rational factor {}", f);
                let gcd = if factor_index == last_factor {
                    // The square-free norm associates every rational factor with a unique factor
                    // of g, so the final unfactored remainder is the final lift.
                    g_f.clone()
                } else {
                    let f = f.to_number_field(&self.ring());
                    let gcd = f.gcd(&g_f);
                    g_f = g_f
                        .try_div_exact(&gcd)
                        .expect("the lifted norm factor must divide the shifted polynomial");
                    gcd
                };

                let g = MultivariatePolynomial::from_number_field(&gcd)
                    .replace_with_poly(v, &alpha_poly)
                    .to_number_field(&self.ring());

                let lc = g.lcoeff();
                self.ring()
                    .mul_assign(&mut constant, &self.ring().pow(&lc, (b * p) as u64));

                full_factors.push((g.mul_coeff(self.ring().inv(&lc)), b * p));
            }
        }

        if !self.ring().is_one(&constant) || full_factors.is_empty() {
            full_factors.push((self.constant(constant), 1));
        }

        full_factors
    }

    fn is_irreducible(&self) -> bool {
        // TODO: improve
        self.factor().len() == 1
    }
}

impl<
    UField: FiniteFieldWorkspace,
    F: GaloisField<Base = FiniteField<UField>>
        + PolynomialGCD<E>
        + SampleableRing<SamplingPolicy = RangeInclusive<i64>>,
    E: PositiveExponent,
> Factorize for MultivariatePolynomial<F, E, LexOrder>
where
    FiniteField<UField>: Field + FiniteFieldCore<UField> + PolynomialGCD<u16>,
    <FiniteField<UField> as Set>::Element: Copy,
    AlgebraicExtension<<F as GaloisField>::Base>: PolynomialGCD<E>,
{
    fn square_free_factorization(&self) -> Vec<(Self, usize)> {
        let c = self.lcoeff();
        let stripped = self.clone().make_monic();

        let mut factors = vec![];
        let fs = stripped.factor_separable();

        for f in fs {
            let mut nf = f.square_free_factorization_bernardin();
            factors.append(&mut nf);
        }

        if factors.is_empty() || !self.ring().is_one(&c) {
            factors.push((self.constant(c), 1))
        }

        factors
    }

    fn factor(&self) -> Vec<(Self, usize)> {
        if self.is_zero() {
            return vec![];
        }

        let sf = self.square_free_factorization();

        let mut factors = vec![];
        let mut degrees = vec![0; self.nvars()];
        for (f, p) in sf {
            debug!("SFF {} {}", f, p);

            let mut var_count = 0;
            for v in 0..self.nvars() {
                degrees[v] = f.degree(v).to_u32() as usize;
                if degrees[v] > 0 {
                    var_count += 1;
                }
            }

            match var_count {
                0 => {
                    factors.push((f, p));
                }
                1 => {
                    for (d2, f2) in f.distinct_degree_factorization() {
                        debug!("DDF {} {}", f2, d2);
                        for f3 in f2.equal_degree_factorization(d2) {
                            debug!("EDF {} {}", f3, p);
                            factors.push((f3, p));
                        }
                    }
                }
                2 => {
                    let mut order: Vec<_> = degrees
                        .iter()
                        .enumerate()
                        .filter(|(_, d)| **d > 0)
                        .collect();
                    order.sort_by_key(|o| Reverse(o.1));
                    let order: Vec<_> = order.into_iter().map(|(v, _)| v).collect();

                    factors.extend(
                        f.bivariate_factorization(order[0], order[1])
                            .into_iter()
                            .map(|ff| (ff, p)),
                    )
                }
                _ => {
                    // TODO: find better order
                    let mut order: Vec<_> = degrees
                        .iter()
                        .enumerate()
                        .filter(|(_, d)| **d > 0)
                        .collect();
                    order.sort_by_key(|o| Reverse(o.1));

                    let mut order: Vec<_> = order.into_iter().map(|(v, _)| v).collect();

                    factors.extend(
                        f.multivariate_factorization(&mut order, 10, None)
                            .into_iter()
                            .map(|ff| (ff, p)),
                    )
                }
            }
        }

        factors
    }

    fn is_irreducible(&self) -> bool {
        let mut sf = self.square_free_factorization();
        if sf.len() > 1 {
            return false;
        }

        let (f, p) = sf.pop().unwrap();

        let mut degrees = vec![0; self.nvars()];
        debug!("SFF {} {}", f, p);

        let mut var_count = 0;
        for v in 0..self.nvars() {
            degrees[v] = f.degree(v).to_u32() as usize;
            if degrees[v] > 0 {
                var_count += 1;
            }
        }

        match var_count {
            0 => true,
            1 => {
                let mut d = f.distinct_degree_factorization();
                if d.len() > 1 {
                    return false;
                }

                let (d2, f2) = d.pop().unwrap();

                f2.equal_degree_factorization(d2).len() == 1
            }
            2 => {
                let mut order: Vec<_> = degrees
                    .iter()
                    .enumerate()
                    .filter(|(_, d)| **d > 0)
                    .collect();
                order.sort_by_key(|o| Reverse(o.1));
                let order: Vec<_> = order.into_iter().map(|(v, _)| v).collect();

                f.bivariate_factorization(order[0], order[1]).len() == 1
            }
            _ => {
                // TODO: find better order
                let mut order: Vec<_> = degrees
                    .iter()
                    .enumerate()
                    .filter(|(_, d)| **d > 0)
                    .collect();
                order.sort_by_key(|o| Reverse(o.1));

                let mut order: Vec<_> = order.into_iter().map(|(v, _)| v).collect();

                f.multivariate_factorization(&mut order, 10, None).len() == 1
            }
        }
    }
}

impl<
    UField: FiniteFieldWorkspace,
    F: GaloisField<Base = FiniteField<UField>>
        + PolynomialGCD<E>
        + SampleableRing<SamplingPolicy = RangeInclusive<i64>>,
    E: PositiveExponent,
> MultivariatePolynomial<F, E, LexOrder>
where
    FiniteField<UField>: Field + FiniteFieldCore<UField> + PolynomialGCD<u16>,
    <FiniteField<UField> as Set>::Element: Copy,
    AlgebraicExtension<<F as GaloisField>::Base>: PolynomialGCD<E>,
{
    /// Bernardin's algorithm for square free factorization.
    fn square_free_factorization_bernardin(&self) -> Vec<(Self, usize)> {
        if self.is_constant() {
            if self.is_one() {
                return vec![];
            } else {
                return vec![(self.clone(), 1)];
            }
        }

        let mut f = self.clone();

        let mut h = HashMap::default();
        let mut hr;
        for var in 0..self.nvars() {
            if f.degree(var) > E::zero() {
                (f, hr) = f.square_free_factorization_ff_yun(var);

                for (part, pow) in hr {
                    h.entry(pow)
                        .and_modify(|f| {
                            *f = &*f * &part;
                        })
                        .or_insert(part);
                }
            }
        }

        // take the pth root
        // the coefficients remain unchanged, since x^1/p = x
        // since the derivative in every var is 0, all powers are divisible by p
        let p = self.ring().characteristic().to_u64().unwrap() as usize;
        let mut b = f.clone();
        for es in b.exponents_iter_mut() {
            for e in es {
                if e.is_zero() {
                    continue;
                }

                if p < u32::MAX as usize {
                    debug_assert_eq!(e.to_u32() as usize % p, 0);
                    *e = *e / E::from_u32(p as u32);
                } else {
                    // at the moment exponents are limited to 32-bits
                    // so only the case where e = 0 is supported
                    assert!(*e == E::zero());
                }
            }
        }

        let mut factors = vec![];
        let sub_factors = b.square_free_factorization_bernardin();

        for (mut k, n) in sub_factors {
            for (powh, hi) in &mut h {
                if *powh < p {
                    let g = k.gcd(hi);
                    if !g.is_constant() {
                        k = k / &g;
                        *hi = &*hi / &g;
                        factors.push((g, n * p + *powh));
                    }
                }
            }

            if !k.is_constant() {
                factors.push((k, n * p));
            }
        }

        for (powh, hi) in h {
            if !hi.is_constant() {
                factors.push((hi, powh));
            }
        }

        factors
    }

    /// A modified version of Yun's square free factorization algorithm.
    fn square_free_factorization_ff_yun(&self, var: usize) -> (Self, Vec<(Self, usize)>) {
        let b = self.derivative(var);
        let mut c = self.gcd(&b);
        let mut w = self / &c;
        let mut v = &b / &c;

        let mut factors = vec![];

        let mut i = 1;
        while !w.is_constant() && i < self.ring().characteristic().to_u64().unwrap() as usize {
            let z = v - w.derivative(var);
            let g = w.gcd(&z);
            w = w / &g;
            v = z / &g;
            c = c / &w;

            if !g.is_one() {
                factors.push((g, i));
            }
            i += 1
        }

        (c, factors)
    }

    /// Perform distinct degree factorization on a monic, univariate and square-free polynomial.
    pub fn distinct_degree_factorization(&self) -> Vec<(usize, Self)> {
        if self.is_constant() {
            return vec![(0, self.clone())];
        }

        self.distinct_degree_factorization_bounded(None)
            .unwrap()
            .blocks
    }

    /// Compute distinct-degree blocks and their exact irreducible-factor count.
    ///
    /// If `max_factor_count` is present, return the first proven lower bound that exceeds this
    /// inclusive limit instead of completing the remaining blocks.
    fn distinct_degree_factorization_bounded(
        &self,
        max_factor_count: Option<usize>,
    ) -> Result<DistinctDegreeFactorization<Self>, usize> {
        let Some(var) = self.last_exponents().iter().position(|x| *x > E::zero()) else {
            debug_assert!(self.is_one(), "bounded DDF requires a monic polynomial");
            return Ok(DistinctDegreeFactorization {
                blocks: vec![],
                factor_count: 0,
            });
        };

        let mut e = self.last_exponents().to_vec();
        e[var] = E::one();
        let x = self.monomial(self.ring().one(), e);

        let mut factors = vec![];
        let mut factor_count = 0usize;
        let mut h = x.clone();
        let mut f = self.clone();
        let mut i: usize = 0;
        if max_factor_count.is_some_and(|limit| limit == 0) {
            #[cfg(test)]
            {
                BOUNDED_DDF_REJECTIONS.with(|rejections| rejections.set(rejections.get() + 1));
                LAST_BOUNDED_DDF_REJECTION_DEGREE.with(|degree| degree.set(0));
            }
            return Err(1);
        }
        while !f.is_one() {
            i += 1;

            h = h.exp_mod_univariate(self.ring().size().unwrap(), &mut f);

            let mut g = f.gcd(&(&h - &x));

            if !g.is_one() {
                f = f.quot_rem_univariate(&mut g).0;
                let block_degree = g.degree(var).to_u32() as usize;
                debug_assert_eq!(block_degree % i, 0);
                factor_count += block_degree / i;
                factors.push((i, g));
            }

            let factor_count_lower_bound = factor_count + usize::from(!f.is_constant());
            if max_factor_count.is_some_and(|limit| factor_count_lower_bound > limit) {
                #[cfg(test)]
                {
                    BOUNDED_DDF_REJECTIONS.with(|rejections| rejections.set(rejections.get() + 1));
                    LAST_BOUNDED_DDF_REJECTION_DEGREE.with(|degree| degree.set(i));
                }
                return Err(factor_count_lower_bound);
            }

            if f.last_exponents()[var] < E::from_u32(2 * (i as u32 + 1)) {
                // f cannot be split more
                if !f.is_constant() {
                    factor_count += 1;
                    factors.push((f.last_exponents()[var].to_u32() as usize, f));
                }
                break;
            }
        }

        Ok(DistinctDegreeFactorization {
            blocks: factors,
            factor_count,
        })
    }

    /// Perform Cantor-Zassenhaus's probabilistic algorithm for
    /// finding irreducible factors of degree `d`.
    pub fn equal_degree_factorization(&self, d: usize) -> Vec<Self> {
        let s = self.clone().make_monic();

        let Some(var) = self.last_exponents().iter().position(|x| *x > E::zero()) else {
            if d == 1 {
                return vec![s];
            } else {
                panic!("Degree mismatch for {self}: {d}");
            }
        };

        let n = self.degree(var).to_u32() as usize;

        if n == d {
            return vec![s];
        }

        let mut rng = rng();
        let mut random_poly = self.zero_with_capacity(d);
        let mut exp = vec![E::zero(); self.nvars()];

        let characteristic = self.ring().characteristic();

        // compute inverse
        let mut s_rev = s.clone();
        s_rev.reverse();
        let s_inv = s_rev.inverse_univariate(var, E::from_u32(n as u32 + 1));

        let factor = loop {
            // generate a random non-constant polynomial
            random_poly.clear();

            for i in 0..n {
                let upper_bound = characteristic.to_i64().unwrap_or(i64::MAX);
                let r = self
                    .ring()
                    .sample(&mut rng, &(0..=upper_bound.saturating_sub(1)));
                if !self.ring().is_zero(&r) {
                    exp[var] = E::from_u32(i as u32);
                    random_poly.append_monomial(r, &exp);
                }
            }

            if random_poly.degree(var) == E::zero() {
                continue;
            }
            *random_poly.coefficients.last_mut().unwrap() = self.ring().one();

            let g = random_poly.gcd(&s);

            if !g.is_one() {
                break g;
            }

            let b = if self.ring().characteristic() == 2 {
                let max = self.ring().extension_degree() as usize * d;

                let mut b = random_poly.clone();
                let mut vcur = b.clone();

                for _ in 1..max {
                    vcur = (&vcur * &vcur).rem(&s);
                    b = b + vcur.clone();
                }

                b
            } else {
                // TODO: use Frobenius map and modular composition to prevent computing large exponent poly^(p^d)
                let p = self.ring().size().unwrap();
                random_poly.exp_mod_univariate_fast(
                    var,
                    &(&p.pow(d as u64) - &1i64.into()) / &2i64.into(),
                    &s,
                    &s_inv,
                ) - self.one()
            };

            if b.is_constant() {
                continue;
            }

            let g = b.gcd(&s);

            if !g.is_one() && g != s {
                break g;
            }
        };

        let mut factors = factor.equal_degree_factorization(d);
        factors.extend((self / &factor).equal_degree_factorization(d));
        factors
    }

    /// Perform distinct and equal degree factorization on a square-free univariate polynomial.
    fn factor_distinct_equal_degree(&self) -> Vec<Self> {
        let mut factors = vec![];
        for (d2, f2) in self.distinct_degree_factorization() {
            debug!("DDF {} {}", f2, d2);
            for f3 in f2.equal_degree_factorization(d2) {
                debug!("EDF {}", f3);
                factors.push(f3);
            }
        }
        factors
    }

    /// Bernardin's algorithm based on
    /// "A new bivariate Hensel lifting algorithm for n factors"
    /// by Garrett Paluck. The formulation of the algorithm in other sources contain serious errors.
    // TODO: merge with an almost similar method for the integer case. A modification that needs
    // to be made here is to make the lcoeff_y=0 monic
    fn bivariate_hensel_lift_bernardin(
        &self,
        interpolation_var: usize,
        lcoeff: &Self,
        univariate_factors: &[Self],
        iterations: usize,
    ) -> Vec<Self> {
        let y_poly = self.to_univariate_polynomial_list(interpolation_var);

        // add the leading coefficient as a first factor
        let mut factors = vec![lcoeff.replace(interpolation_var, &self.ring().zero())];
        factors.extend_from_slice(univariate_factors);

        // extract coefficients in y
        let mut u: Vec<_> = factors
            .iter()
            .map(|f| {
                let mut dense = vec![self.zero(); iterations + 1];
                dense[0] = f.clone();
                dense
            })
            .collect();

        // update the first polynomial as it may contain y, since it's lcoeff
        let y_lcoeff = lcoeff.to_univariate_polynomial_list(interpolation_var);
        for (p, e) in y_lcoeff {
            u[0][e.to_u32() as usize] = p;
        }

        let mut p = u.clone();
        let mut cur_p = p[0][0].clone();
        for x in &mut p.iter_mut().skip(1) {
            cur_p = cur_p * &x[0];
            x[0] = cur_p.clone();
        }

        let delta = Self::diophantine_univariate(&mut factors, &self.one());

        for k in 1..iterations {
            // extract the coefficient required to compute the error in y^k
            // computed using a convolution
            p[0][k] = u[0][k].clone();
            for i in 1..factors.len() {
                for j in 0..k {
                    p[i][k] = &p[i][k] + &(&p[i - 1][k - j] * &u[i][j]);
                }
            }

            // find the kth power of y in f
            // since we compute the error per power of y, we cannot stop on a 0 error
            let e = if let Some((v, _)) = y_poly.iter().find(|e| e.1.to_u32() as usize == k) {
                v - &p.last().unwrap()[k]
            } else {
                -p.last().unwrap()[k].clone()
            };

            if e.is_zero() {
                continue;
            }

            for ((dp, f), d) in u.iter_mut().zip(factors.iter_mut()).zip(&delta) {
                dp[k] = &dp[k] + &(d * &e).quot_rem_univariate(f).1;
            }

            // update the coefficients with the new y^k contributions
            // note that the lcoeff[k] contribution is not new
            let mut t = self.zero();
            for i in 1..factors.len() {
                t = &u[i][0] * &t + &u[i][k] * &p[i - 1][0];
                p[i][k] = &p[i][k] + &t;
            }
        }

        // convert dense polynomials to multivariate polynomials
        u.into_iter()
            .map(|ts| {
                let mut new_poly = self.zero_with_capacity(ts.len());
                for (i, mut f) in ts.into_iter().enumerate() {
                    for x in f.exponents_iter_mut() {
                        x[interpolation_var] = E::from_u32(i as u32);
                    }
                    new_poly = new_poly + f;
                }
                new_poly
            })
            .collect()
    }

    /// Compute the bivariate factorization of a square-free polynomial.
    fn bivariate_factorization(&self, main_var: usize, interpolation_var: usize) -> Vec<Self> {
        assert!(main_var != interpolation_var);

        if self.bivariate_irreducibility_test() {
            return vec![self.clone()];
        }

        // check for problems arising from canceling terms in the derivative
        let der = self.derivative(main_var);
        if der.is_zero() {
            return self.bivariate_factorization(interpolation_var, main_var);
        }

        let g = self.gcd(&der);
        if !g.is_constant() {
            let mut factors = g.bivariate_factorization(main_var, interpolation_var);
            factors.extend((self / &g).bivariate_factorization(main_var, interpolation_var));
            return factors;
        }

        let mut sample_point = self.ring().zero();
        let mut uni_f = self.replace(interpolation_var, &sample_point);

        let mut i = 0;
        let mut rng = rng();
        loop {
            i += 1;
            if self.ring().size() == Some(i.into()) {
                let field = self
                    .ring()
                    .upgrade(self.ring().extension_degree().to_u64().unwrap() as usize + 1);

                debug!(
                    "Upgrading to Galois field with exponent {}",
                    field.extension_degree()
                );

                let s_l = self.map_coeff(|c| self.ring().upgrade_element(c, &field), field.clone());

                let facs = s_l.bivariate_factorization(main_var, interpolation_var);

                return facs
                    .into_iter()
                    .map(|f| f.map_coeff(|c| self.ring().downgrade_element(c), self.ring().clone()))
                    .collect();
            }

            if self.degree(main_var) == uni_f.degree(main_var)
                && uni_f.gcd(&uni_f.derivative(main_var)).is_constant()
            {
                break;
            }

            sample_point = self.ring().sample(&mut rng, &(0..=i));
            uni_f = self.replace(interpolation_var, &sample_point);
        }

        let mut d = self.degree(interpolation_var).to_u32();

        let shifted_poly = if !self.ring().is_zero(&sample_point) {
            self.shift_var_cached(interpolation_var, &sample_point)
        } else {
            self.clone()
        };

        let fs = uni_f.factor_distinct_equal_degree();

        let mut lcoeff = shifted_poly.lcoeff_last_varorder(&[main_var, interpolation_var]);
        let mut lc_d = lcoeff.degree(interpolation_var).to_u32();

        let iter = (d + lc_d + 1) as usize;
        let mut factors =
            shifted_poly.bivariate_hensel_lift_bernardin(interpolation_var, &lcoeff, &fs, iter);

        factors.swap_remove(0); // remove the lcoeff

        let mut rec_factors = vec![];
        // factor recombination
        let mut s = 1;

        let mut rest = shifted_poly;
        'len: while 2 * s <= factors.len() {
            let mut fs = CombinationIterator::new(factors.len(), s);
            while let Some(cs) = fs.next() {
                // TODO: multiply in the leading coefficient here,
                // then we can skip the Pade approximation and reduce the
                // number of iterations in the Hensel lifting to d + 1, like in the integer case?
                let mut g = rest.constant(rest.lcoeff());
                for (i, f) in factors.iter().enumerate() {
                    if cs.contains(&i) {
                        g = (&g * f).mod_var(interpolation_var, E::from_u32(iter as u32 + 1));
                    }
                }

                let y_polys: Vec<_> = g
                    .to_univariate_polynomial_list(main_var)
                    .into_iter()
                    .map(|(x, _)| x)
                    .collect();

                let mut g_lcoeff = Self::lcoeff_reconstruct(&y_polys, d, lc_d);
                g = (&g * &g_lcoeff)
                    .mod_var(interpolation_var, E::from_u32(d + 1))
                    .make_monic();

                let (h, r) = rest.quot_rem(&g, true);

                if r.is_zero() {
                    rec_factors.push(g);

                    for i in cs.iter().rev() {
                        factors.remove(*i);
                    }

                    rest = h;
                    lcoeff = lcoeff.quot_rem_univariate(&mut g_lcoeff).0;
                    lc_d = lcoeff.degree(interpolation_var).to_u32();
                    d = rest.degree(interpolation_var).to_u32();

                    continue 'len;
                }
            }

            s += 1;
        }

        rec_factors.push(rest);

        if !self.ring().is_zero(&sample_point) {
            for x in &mut rec_factors {
                // shift the polynomial to y - sample
                *x = x.shift_var_cached(interpolation_var, &self.ring().neg(&sample_point));
            }
        }

        rec_factors
    }

    /// Reconstruct the leading coefficient using a Pade approximation with numerator degree `deg_n` and
    /// denominator degree `deg_d`. The resulting denominator should be a factor of the leading coefficient.
    fn lcoeff_reconstruct(coeffs: &[Self], deg_n: u32, deg_d: u32) -> Self {
        let mut lcoeff = coeffs[0].constant(coeffs[0].ring().one());
        for x in coeffs {
            let d = x.rational_approximant_univariate(deg_n, deg_d).unwrap().1;
            if !d.is_one() {
                let g = d.gcd(&lcoeff);
                lcoeff = lcoeff * &(d / &g);
            }
        }
        lcoeff
    }

    /// Sort the bivariate factors based on their univariate image so that they are
    /// aligned between the different vars.
    fn canonical_sort(
        biv_polys: &[Self],
        replace_var: usize,
        sample_points: &[(usize, <F as Set>::Element)],
    ) -> Vec<(Self, <F as Set>::Element, Self)> {
        let mut univariate_factors = biv_polys
            .iter()
            .map(|f| {
                let mut u = f.clone();
                for (v, p) in sample_points {
                    if *v == replace_var {
                        u = u.replace(*v, p);
                    }
                }

                (f.clone(), u.lcoeff(), u.make_monic())
            })
            .collect::<Vec<_>>();
        univariate_factors.sort_by(|(_, _, a), (_, _, b)| {
            a.exponents
                .cmp(&b.exponents)
                .then(a.coefficients.internal_cmp(&b.coefficients))
        });

        univariate_factors
    }

    /// Precompute the leading coefficients of the polynomial factors, using an
    /// adapted version of Kaltofen's algorithm that has modifications of Martin Lee and Stanislav Poslavsky.
    fn lcoeff_precomputation(
        &self,
        bivariate_factors: &[Self],
        sample_points: &[(usize, <F as Set>::Element)],
        order: &[usize],
    ) -> Result<(Vec<Self>, Vec<Self>), usize> {
        let lcoeff = self.univariate_lcoeff(order[0]);
        let sqf = lcoeff.square_free_factorization();

        let mut lcoeff_square_free = self.one();
        for (f, _) in &sqf {
            lcoeff_square_free = lcoeff_square_free * f;
        }

        let sorted_main_factors = Self::canonical_sort(bivariate_factors, order[1], sample_points);

        let mut true_lcoeffs: Vec<_> = sorted_main_factors
            .iter()
            .map(|(_, u, _)| self.constant(u.clone()))
            .collect();

        let main_bivariate_factors: Vec<_> =
            sorted_main_factors.into_iter().map(|(f, _, _)| f).collect();

        let mut lcoeff_left = lcoeff.clone();
        for f in &true_lcoeffs {
            lcoeff_left = lcoeff_left / f;
        }

        // TODO: smarter ordering
        for (i, &var) in order[1..].iter().enumerate() {
            if lcoeff_left.is_one() {
                break;
            }

            if lcoeff_left.degree(var).is_zero() {
                continue;
            }

            // only construct factors that depend on var
            let c = lcoeff_square_free.univariate_content(var);
            // make sure that the content removal does not change the unit
            let mut c_eval = c.clone();
            for (v, p) in sample_points {
                c_eval = c_eval.replace(*v, p);
            }

            let lcoeff_square_free_pp = &lcoeff_square_free / &c * &c_eval;
            debug!("Content-free lcsqf {}", lcoeff_square_free_pp);

            // check if the evaluated leading coefficient remains square free
            let mut poly_eval = lcoeff_square_free_pp.clone();
            for (v, p) in sample_points {
                if *v != var {
                    poly_eval = poly_eval.replace(*v, p);
                }
            }
            let sqf = poly_eval.square_free_factorization();
            if sqf.len() != 1 || sqf[0].1 != 1 {
                debug!("Polynomial is not square free: {}", poly_eval);
                return Err(main_bivariate_factors.len());
            }

            let bivariate_factors = if var == order[1] {
                main_bivariate_factors.to_vec()
            } else {
                let mut poly_eval = self.clone();
                for (v, p) in sample_points {
                    if *v != var {
                        poly_eval = poly_eval.replace(*v, p);
                    }
                }

                if poly_eval.degree(order[0]) != self.degree(order[0])
                    || poly_eval.degree(var) != self.degree(var)
                    || poly_eval.univariate_lcoeff(order[0]).degree(var) != lcoeff.degree(var)
                {
                    debug!("Bad sample for reconstructing lcoeff: degrees do not match");
                    return Err(main_bivariate_factors.len());
                }

                let bivariate_factors: Vec<_> =
                    poly_eval.factor().into_iter().map(|(f, _)| f).collect();

                if bivariate_factors.len() != main_bivariate_factors.len() {
                    return Err(bivariate_factors.len().min(main_bivariate_factors.len()));
                }

                Self::canonical_sort(&bivariate_factors, var, sample_points)
                    .into_iter()
                    .map(|(f, _, _)| f)
                    .collect()
            };

            let square_free_lc_biv_factors: Vec<_> = bivariate_factors
                .iter()
                .map(|f| f.univariate_lcoeff(order[0]).square_free_factorization())
                .collect();

            let basis = Self::gcd_free_basis(
                square_free_lc_biv_factors
                    .iter()
                    .flatten()
                    .map(|x| x.0.clone())
                    .filter(|x| !x.is_constant())
                    .collect(),
            );

            if basis.is_empty() {
                continue;
            }

            let lifted = if basis.len() == 1 {
                vec![lcoeff_square_free_pp.clone()]
            } else {
                let mut new_order = order.to_vec();
                new_order.swap(1, i + 1);
                new_order.remove(0);

                lcoeff_square_free_pp.multivariate_hensel_lift_with_auto_lcoeff_fixing(
                    &basis,
                    sample_points,
                    &new_order,
                )
            };

            for (l, fac) in true_lcoeffs.iter_mut().zip(&square_free_lc_biv_factors) {
                let mut contrib = self.one();
                for (full, b) in lifted.iter().zip(&basis) {
                    // check if a GCD-free basis element is a factor of the leading coefficient of this bivariate factor
                    if let Some((_, m)) = fac.iter().find(|(f, _)| f == b || f.try_div(b).is_some())
                    {
                        for _ in 0..*m {
                            contrib = &contrib * full;
                        }
                    }
                }

                let g = contrib.gcd(l);
                let mut new = contrib / &g;

                // make sure the new part keeps the desired image coeff intact
                let mut b_lc_eval = new.clone();
                for (v, p) in sample_points {
                    b_lc_eval = b_lc_eval.replace(*v, p);
                }

                new = new / &b_lc_eval;

                *l = &*l * &new;
                lcoeff_left = &lcoeff_left / &new;
            }
        }

        if !lcoeff_left.is_one() {
            panic!(
                "Could not reconstruct leading coefficient of {self}: order={order:?}, samples={sample_points:?} Rest = {lcoeff_left}"
            );
        }

        Ok((main_bivariate_factors, true_lcoeffs))
    }

    fn multivariate_hensel_lift_with_auto_lcoeff_fixing(
        &self,
        factors: &[Self],
        sample_points: &[(usize, <F as Set>::Element)],
        order: &[usize],
    ) -> Vec<Self> {
        let lcoeff = self.univariate_lcoeff(order[0]);

        if lcoeff.is_constant() {
            // the factors should be properly normalized
            let (mut uni, delta) =
                Self::univariate_diophantine_field(factors, order, sample_points);
            return self
                .multivariate_hensel_lifting(
                    factors,
                    &mut uni,
                    &delta,
                    sample_points,
                    None,
                    order,
                    MultivariateHenselContext::new(1),
                )
                .unwrap();
        }

        // repeat the leading coefficient for every factor so that the leading coefficient is known
        let padded_lcoeffs = vec![lcoeff.clone(); factors.len()];

        let mut self_adjusted = self.clone();
        for _ in 1..factors.len() {
            self_adjusted = &self_adjusted * &lcoeff;
        }

        // set the proper lc
        let mut lc_var_eval = lcoeff.clone();
        for (v, p) in sample_points {
            if *v != order[0] {
                lc_var_eval = lc_var_eval.replace(*v, p);
            }
        }

        let adjusted_factors: Vec<_> = factors
            .iter()
            .map(|f| f.clone().make_monic() * &lc_var_eval)
            .collect();

        let (mut uni, delta) =
            Self::univariate_diophantine_field(&adjusted_factors, order, sample_points);
        self_adjusted
            .multivariate_hensel_lifting(
                &adjusted_factors,
                &mut uni,
                &delta,
                sample_points,
                Some(&padded_lcoeffs),
                order,
                MultivariateHenselContext::new(1),
            )
            .unwrap()
            .into_iter()
            .map(|f| {
                let c = f.univariate_content(order[0]);
                f / &c
            })
            .collect()
    }

    fn univariate_diophantine_field(
        factors: &[Self],
        order: &[usize],
        sample_points: &[(usize, <F as Set>::Element)],
    ) -> (Vec<Self>, Vec<Self>) {
        // produce univariate factors and univariate delta
        let mut univariate_factors = factors.to_vec();
        for f in &mut univariate_factors {
            for (v, s) in sample_points {
                if order[0] != *v {
                    *f = f.replace(*v, s);
                }
            }
        }

        let univariate_deltas = Self::diophantine_univariate(
            &mut univariate_factors,
            &factors[0].constant(factors[0].ring().one()),
        );

        (univariate_factors, univariate_deltas)
    }

    /// Perform multivariate factorization on a square-free polynomial.
    fn multivariate_factorization(
        &self,
        order: &mut [usize],
        mut coefficient_upper_bound: u64,
        max_bivariate_factors: Option<usize>,
    ) -> Vec<Self> {
        if let Some(m) = max_bivariate_factors
            && m == 1
        {
            return vec![self.clone()];
        }

        // check for problems arising from canceling terms in the derivative
        let der = self.derivative(order[0]);
        if der.is_zero() {
            let mut new_order = order.to_vec();
            let v = new_order.remove(0);
            new_order.push(v);
            return self.multivariate_factorization(
                &mut new_order,
                coefficient_upper_bound,
                max_bivariate_factors,
            );
        }

        let g = self.gcd(&der);
        if !g.is_constant() {
            let mut factors =
                g.multivariate_factorization(order, coefficient_upper_bound, max_bivariate_factors);
            factors.extend((self / &g).multivariate_factorization(
                order,
                coefficient_upper_bound,
                max_bivariate_factors,
            ));
            return factors;
        }

        // select a suitable evaluation point
        let mut sample_points: Vec<_> = order[1..]
            .iter()
            .map(|i| (*i, self.ring().zero()))
            .collect();
        let mut uni_f;
        let mut biv_f;
        let mut rng = rng();
        let degree = self.degree(order[0]);

        let uni_lcoeff = self.univariate_lcoeff(order[0]);

        let mut content_fail_count = 0;
        let mut sample_fail = Integer::zero();
        'new_sample: loop {
            sample_fail += &1.into();

            if &sample_fail * &2.into() > self.ring().size().unwrap() {
                // the field is too small, upgrade
                let field = self
                    .ring()
                    .upgrade(self.ring().extension_degree().to_u64().unwrap() as usize + 1);

                debug!(
                    "Upgrading to Galois field with exponent {}",
                    field.extension_degree()
                );

                let s_l = self.map_coeff(|c| self.ring().upgrade_element(c, &field), field.clone());

                let facs = s_l.multivariate_factorization(
                    order,
                    coefficient_upper_bound,
                    max_bivariate_factors,
                );

                return facs
                    .into_iter()
                    .map(|f| f.map_coeff(|c| self.ring().downgrade_element(c), self.ring().clone()))
                    .collect();
            }

            for s in &mut sample_points {
                s.1 = self
                    .ring()
                    .nth(rng.random_range(0..=coefficient_upper_bound).into());
            }

            biv_f = self.clone();
            for ((v, s), rem_var) in sample_points[1..].iter().zip(&order[1..]).rev() {
                biv_f = biv_f.replace(*v, s);
                if biv_f.degree(*rem_var) != self.degree(*rem_var) {
                    coefficient_upper_bound += 10;
                    continue 'new_sample;
                }
            }

            // requirement for leading coefficient precomputation
            if biv_f.univariate_lcoeff(order[0]).degree(order[1]) != uni_lcoeff.degree(order[1]) {
                debug!(
                    "Degree of x{} in leading coefficient of bivariate image is wrong",
                    order[1]
                );
                coefficient_upper_bound += 10;
                continue 'new_sample;
            }

            let biv_df = biv_f.derivative(order[0]);

            uni_f = biv_f.replace(sample_points[0].0, &sample_points[0].1);
            let uni_df = uni_f.derivative(order[0]);

            if degree == biv_f.degree(order[0])
                && degree == uni_f.degree(order[0])
                && biv_f.gcd(&biv_df).is_constant()
                && uni_f.gcd(&uni_df).is_constant()
            {
                if !biv_f.univariate_content(order[0]).is_constant() {
                    content_fail_count += 1;

                    debug!("Univariate content is not constant");
                    if content_fail_count == 4 {
                        // it is likely that we will always find content for this variable ordering, so change the
                        // second variable
                        // TODO: is this guaranteed to work or should we also change the first variable?
                        let sec_var = order[1];
                        order.copy_within(2..order.len(), 1);
                        order[order.len() - 1] = sec_var;

                        for ((vs, _), v) in sample_points.iter_mut().zip(&order[1..]) {
                            *vs = *v;
                        }

                        debug!("Changed the second variable to {}", order[1]);
                        content_fail_count = 0;
                    }
                } else {
                    break;
                }
            }

            coefficient_upper_bound += 10;
        }

        for (v, s) in &sample_points {
            debug!("Sample point {}={}", v, self.ring().printer(s));
        }

        let bivariate_factors = biv_f.bivariate_factorization(order[0], order[1]);

        if bivariate_factors.len() == 1 {
            // the polynomial is irreducible
            return vec![self.clone()];
        }

        if let Some(max) = max_bivariate_factors
            && bivariate_factors.len() > max
        {
            return self.multivariate_factorization(
                order,
                coefficient_upper_bound,
                max_bivariate_factors,
            );
        }

        let (sorted_biv_factors, true_lcoeffs) =
            match self.lcoeff_precomputation(&bivariate_factors, &sample_points, order) {
                Ok((sorted_biv_factors, true_lcoeffs)) => (sorted_biv_factors, true_lcoeffs),
                Err(max_biv) => {
                    // the leading coefficient computation failed because the bivaraite factorization was wrong
                    // try again with other sample points and a better bound
                    return self.multivariate_factorization(
                        order,
                        coefficient_upper_bound + 10,
                        Some(max_biv),
                    );
                }
            };

        for (b, l) in sorted_biv_factors.iter().zip(&true_lcoeffs) {
            debug!("Bivariate factor {} with true lcoeff {}", b, l);
        }

        let sorted_biv_factors = self.impose_true_lcoeffs_on_factors(
            &sorted_biv_factors,
            &true_lcoeffs,
            &sample_points,
            order,
            2,
        );
        let (mut uni, delta) = MultivariatePolynomial::univariate_diophantine_field(
            &sorted_biv_factors,
            order,
            &sample_points,
        );

        let factorization = match self.multivariate_hensel_lifting(
            &sorted_biv_factors,
            &mut uni,
            &delta,
            &sample_points,
            Some(&true_lcoeffs),
            order,
            MultivariateHenselContext::new(2),
        ) {
            Ok(factorization) => factorization,
            Err(_) => {
                return self.multivariate_factorization(
                    order,
                    coefficient_upper_bound + 10,
                    max_bivariate_factors,
                );
            }
        };

        // test the factorization
        let mut test = self.one();
        for f in &factorization {
            debug!("Factor = {}", f);
            test = &test * f;
        }

        if &test == self {
            factorization
        } else {
            debug!(
                "No immediate factorization of {} for sample points {:?}",
                self, sample_points
            );

            // the bivariate factorization has too many factors, try again with other sample points
            self.multivariate_factorization(
                order,
                coefficient_upper_bound + 10,
                Some(max_bivariate_factors.unwrap_or(bivariate_factors.len()) - 1),
            )
        }
    }
}

impl<F: Field + SampleableRing<SamplingPolicy = RangeInclusive<i64>>, E: PositiveExponent>
    MultivariatePolynomial<F, E, LexOrder>
{
    fn multivariate_diophantine(
        univariate_deltas: &[Self],
        univariate_factors: &mut [Self],
        prods: &[Self],
        error: &Self,
        order: &[usize],
        sample_points: &[(usize, F::Element)],
        degrees: &[usize],
        mod_vars: &[MultivariatePolynomial<F, E, LexOrder>],
    ) -> Vec<Self> {
        if order.len() == 1 {
            return univariate_deltas
                .iter()
                .zip(univariate_factors)
                .map(|(d, f)| (d * error).quot_rem_univariate(f).1)
                .collect();
        }

        let last_var = *order.last().unwrap();

        let shift = &sample_points.iter().find(|s| s.0 == last_var).unwrap().1;

        let prods_mod = prods
            .iter()
            .map(|f| f.replace(last_var, shift))
            .collect::<Vec<_>>();
        let error_mod = error.replace(last_var, shift);
        let previous_mod_vars = &mod_vars[..mod_vars.len().saturating_sub(1)];

        debug!("dioph e[x{}^0] = {}", last_var, error_mod);

        let mut deltas = Self::multivariate_diophantine(
            univariate_deltas,
            univariate_factors,
            &prods_mod,
            &error_mod,
            &order[..order.len() - 1],
            sample_points,
            &degrees[..order.len() - 1],
            &mod_vars[..order.len() - 1],
        );
        for d in &mut deltas {
            for m in mod_vars {
                *d = d.quot_rem(m, false).1;
            }
        }

        let mut exp = vec![E::zero(); error.nvars()];
        exp[last_var] = E::one();
        let var_pow = error
            .monomial(error.ring().one(), exp)
            .shift_var(last_var, &error.ring().neg(shift));
        let mut cur_exponent;
        let mut next_exponent = var_pow.clone();

        for j in 1..=*degrees.last().unwrap() {
            cur_exponent = next_exponent.clone();
            next_exponent = &next_exponent * &var_pow;

            let mut e = error.clone();
            for (d, p) in deltas.iter().zip(prods) {
                debug!("delta {} p {}", d, p);
                e = &e - &(d * p);

                for m in previous_mod_vars {
                    e = e.quot_rem(m, false).1;
                }

                // TODO: mod with (x-shift)^(j+1)?
                // then we cannot break on 0 error
            }

            debug!("dioph  e at x{}^{} = {}", last_var, j, e);

            if e.is_zero() {
                break;
            }

            // Take the jth Taylor coefficient in the current variable. Lower
            // powers may survive as representatives in the ambient polynomial
            // ring, but they do not contribute to this coefficient.
            let shifted_e = e.shift_var(last_var, shift);
            let mut e_mod = error.zero();
            for (coefficient, degree) in shifted_e.to_univariate_polynomial_list(last_var) {
                if degree.to_u32() as usize == j {
                    e_mod = e_mod + coefficient;
                }
            }

            debug!("dioph  e[x{}^{}] = {}", last_var, j, e_mod);

            if e_mod.is_zero() {
                continue;
            }

            let mut new_deltas = Self::multivariate_diophantine(
                univariate_deltas,
                univariate_factors,
                &prods_mod,
                &e_mod,
                &order[..order.len() - 1],
                sample_points,
                &degrees[..order.len() - 1],
                &mod_vars[..order.len() - 1],
            );

            for (d, nd) in deltas.iter_mut().zip(&mut new_deltas) {
                debug!("dioph  d[x{}^{}] = {}", last_var, j, nd);

                // multiply (y-s)^j and keep the accumulated correction reduced
                let nd = &*nd * &cur_exponent;
                *d = &*d + &nd;

                for m in mod_vars {
                    *d = d.quot_rem(m, false).1;
                }
            }
        }

        deltas
    }

    fn sparse_multivariate_diophantine_from_skeleton(
        factors: &[Self],
        prods: &[Self],
        error: &Self,
        skeletons: &[Self],
        order: &[usize],
        context: &mut SparseDiophantineContext<Self, F::Element>,
    ) -> Option<Vec<Self>> {
        if prods.len() != skeletons.len() {
            return None;
        }

        if let Some(deltas) = Self::sparse_multivariate_diophantine_two_factor_by_sampling(
            factors, prods, error, skeletons, order, context,
        ) {
            return Some(deltas);
        }

        if factors.len() > 2 {
            if let Some(deltas) = Self::sparse_multivariate_diophantine_by_sampling(
                factors, prods, error, skeletons, order,
            ) {
                return Some(deltas);
            }
        }

        None
    }

    fn evaluate_using_exponents_univariate_grouped(
        &self,
        exp_evals: &[F::Element],
        main_var: usize,
        out: &mut Self,
    ) {
        out.clear();

        let degree = self.degree(main_var).to_u32() as usize;
        let mut coefficients = vec![self.ring().zero(); degree + 1];
        for (term, eval) in self.into_iter().zip(exp_evals) {
            let degree = term.exponents[main_var].to_u32() as usize;
            self.ring()
                .add_mul_assign(&mut coefficients[degree], term.coefficient, eval);
        }

        let mut exponent = vec![E::zero(); self.nvars()];
        for (degree, coefficient) in coefficients.into_iter().enumerate() {
            if !self.ring().is_zero(&coefficient) {
                exponent[main_var] = E::from_u32(degree as u32);
                out.coefficients.push(coefficient);
                out.exponents.extend_from_slice(&exponent);
                exponent[main_var] = E::zero();
            }
        }
    }

    /// Evaluate all sampled variables along a geometric sequence while retaining
    /// `x` and `y` as a dense bivariate coefficient grid. The first image uses
    /// the base points themselves, and each following image raises them to the
    /// next positive integer power.
    fn evaluate_geometric_bivariate_images(
        &self,
        x: usize,
        y: usize,
        base_points: &[(usize, F::Element)],
        sample_count: usize,
        cache: &mut [Vec<F::Element>],
    ) -> Vec<DenseBivariateImage<F::Element>> {
        debug_assert_ne!(x, y);
        debug_assert!(
            base_points
                .iter()
                .all(|(variable, _)| *variable != x && *variable != y)
        );

        if sample_count == 0 {
            return Vec::new();
        }

        let x_len = self.degree(x).to_u32() as usize + 1;
        let y_len = self.degree(y).to_u32() as usize + 1;
        let image_len = x_len
            .checked_mul(y_len)
            .expect("dense bivariate image is too large");

        let mut offsets = vec![0usize; image_len + 1];
        for term in self {
            let x_degree = term.exponents[x].to_u32() as usize;
            let y_degree = term.exponents[y].to_u32() as usize;
            offsets[y_degree * x_len + x_degree + 1] += 1;
        }
        for index in 1..offsets.len() {
            offsets[index] += offsets[index - 1];
        }

        // Group terms by output cell so that each sample sums and advances all
        // geometric sequences in a cell with one coefficient-domain operation.
        let mut positions = offsets[..image_len].to_vec();
        let mut grouped = std::iter::repeat_with(|| None)
            .take(self.nterms())
            .collect::<Vec<Option<(F::Element, F::Element)>>>();
        let ratios = self.evaluate_exponents(base_points, cache);
        for (term, ratio) in self.into_iter().zip(ratios) {
            let x_degree = term.exponents[x].to_u32() as usize;
            let y_degree = term.exponents[y].to_u32() as usize;
            let cell = y_degree * x_len + x_degree;
            let position = positions[cell];
            positions[cell] += 1;
            let current = self.ring().mul(term.coefficient, &ratio);
            grouped[position] = Some((current, ratio));
        }
        let mut current = Vec::with_capacity(grouped.len());
        let mut ratios = Vec::with_capacity(grouped.len());
        for entry in grouped {
            let (value, ratio) = entry.expect("every image term must belong to one dense cell");
            current.push(value);
            ratios.push(ratio);
        }

        let geometric_sequence_kernels = self.ring().kernels().geometric_sequences();
        let mut images = Vec::with_capacity(sample_count);
        for sample_index in 0..sample_count {
            let mut coefficients = vec![self.ring().zero(); image_len];
            for (cell, range) in offsets.windows(2).enumerate() {
                let [start, end] = [range[0], range[1]];
                if start == end {
                    continue;
                }
                let current_cell = &mut current[start..end];
                let ratio_cell = &ratios[start..end];
                coefficients[cell] = (sample_index + 1 < sample_count)
                    .then(|| geometric_sequence_kernels)
                    .flatten()
                    .and_then(|kernels| {
                        kernels.try_sum_and_advance_geometric_sequences(
                            GeometricSequenceStepRequest {
                                current: &mut *current_cell,
                                ratios: ratio_cell,
                            },
                        )
                    })
                    .unwrap_or_else(|| {
                        let mut coefficient = self.ring().zero();
                        for (current, ratio) in current_cell.iter_mut().zip(ratio_cell) {
                            self.ring().add_assign(&mut coefficient, &*current);
                            if sample_index + 1 < sample_count {
                                self.ring().mul_assign(current, ratio);
                            }
                        }
                        coefficient
                    });
            }
            images.push(DenseBivariateImage {
                x_len,
                y_len,
                coefficients,
            });
        }

        images
    }

    /// Convert one dense `x` row of a bivariate image to a polynomial that
    /// contains only the retained variable `x`.
    fn dense_bivariate_x_row_to_univariate(
        &self,
        image: &DenseBivariateImage<F::Element>,
        x: usize,
        y_degree: usize,
    ) -> Self {
        let mut row = self.zero_with_capacity(image.x_len);
        if y_degree >= image.y_len {
            return row;
        }

        let mut exponent = vec![E::zero(); self.nvars()];
        for x_degree in 0..image.x_len {
            let coefficient = image.coefficient(x_degree, y_degree);
            if !self.ring().is_zero(coefficient) {
                exponent[x] = E::from_u32(x_degree as u32);
                row.append_monomial(coefficient.clone(), &exponent);
                exponent[x] = E::zero();
            }
        }
        row
    }

    /// Assemble a polynomial from coefficients in increasing powers of `y`.
    fn from_dense_y_coefficients(&self, coefficients: &[Self], y: usize) -> Self {
        let capacity = coefficients.iter().map(Self::nterms).sum();
        let mut polynomial = self.zero_with_capacity(capacity);
        for (y_degree, coefficient) in coefficients.iter().enumerate() {
            for term in coefficient {
                let mut exponent = term.exponents.to_vec();
                debug_assert_eq!(exponent[y], E::zero());
                exponent[y] = E::from_u32(y_degree as u32);
                polynomial.append_monomial(term.coefficient.clone(), &exponent);
            }
        }
        polynomial
    }

    /// Compute one coefficient row of `target - left * right` in the retained
    /// variable `y`, with the row represented densely in `x`.
    fn dense_bivariate_product_error_row(
        &self,
        target: &DenseBivariateImage<F::Element>,
        left: &DenseBivariateImage<F::Element>,
        right: &DenseBivariateImage<F::Element>,
        y_degree: usize,
    ) -> Vec<F::Element> {
        let product_x_len = left
            .x_len
            .checked_add(right.x_len)
            .and_then(|length| length.checked_sub(1))
            .expect("dense bivariate product image is too large");
        let x_len = target.x_len.max(product_x_len);
        let mut error = vec![self.ring().zero(); x_len];

        if y_degree < target.y_len {
            for (x_degree, coefficient) in error.iter_mut().enumerate().take(target.x_len) {
                *coefficient = target.coefficient(x_degree, y_degree).clone();
            }
        }

        for left_y_degree in 0..=y_degree.min(left.y_len.saturating_sub(1)) {
            let right_y_degree = y_degree - left_y_degree;
            if right_y_degree >= right.y_len {
                continue;
            }

            for left_x_degree in 0..left.x_len {
                let left_coefficient = left.coefficient(left_x_degree, left_y_degree);
                if self.ring().is_zero(left_coefficient) {
                    continue;
                }

                for right_x_degree in 0..right.x_len {
                    let right_coefficient = right.coefficient(right_x_degree, right_y_degree);
                    if !self.ring().is_zero(right_coefficient) {
                        let product = self.ring().mul(left_coefficient, right_coefficient);
                        self.ring()
                            .sub_assign(&mut error[left_x_degree + right_x_degree], product);
                    }
                }
            }
        }

        error
    }

    /// Pack the nonzero coefficients of a dense univariate polynomial for the
    /// coefficient-domain multiplication kernels.
    fn dense_indexed_univariate_from_coefficients(
        ring: &F,
        coefficients: &[F::Element],
    ) -> Option<DenseIndexedUnivariate<F::Element>> {
        let mut packed_coefficients = Vec::new();
        let mut indices = Vec::new();
        for (degree, coefficient) in coefficients.iter().enumerate() {
            if !ring.is_zero(coefficient) {
                packed_coefficients.push(coefficient.clone());
                indices.push(u32::try_from(degree).ok()?);
            }
        }
        Some(DenseIndexedUnivariate {
            coefficients: packed_coefficients,
            indices,
        })
    }

    /// Convert a polynomial containing only `x` to the packed univariate form
    /// used by the dense correction solver.
    fn dense_indexed_univariate(
        &self,
        polynomial: &Self,
        x: usize,
    ) -> Option<DenseIndexedUnivariate<F::Element>> {
        if x >= polynomial.nvars() {
            return None;
        }

        let mut coefficients = vec![self.ring().zero(); polynomial.degree(x).to_u32() as usize + 1];
        for term in polynomial {
            if term
                .exponents
                .iter()
                .enumerate()
                .any(|(variable, exponent)| variable != x && *exponent != E::zero())
            {
                return None;
            }
            let degree = term.exponents[x].to_u32() as usize;
            self.ring()
                .add_assign(&mut coefficients[degree], term.coefficient);
        }
        Self::dense_indexed_univariate_from_coefficients(&self.ring(), &coefficients)
    }

    /// Normalize the constant `y` row of an image to a monic polynomial in
    /// `x`, omitting its leading coefficient because it is one.
    fn dense_monic_modulus(
        &self,
        image: &DenseBivariateImage<F::Element>,
    ) -> Option<DenseMonicModulus<F::Element>> {
        if image.x_len == 0 || image.y_len == 0 {
            return None;
        }
        let leading = image.coefficient(image.x_len - 1, 0);
        if self.ring().is_zero(leading) {
            return None;
        }
        let leading_inverse = self.ring().inv(leading);
        let lower_coefficients = (0..image.x_len - 1)
            .map(|degree| {
                self.ring()
                    .mul(image.coefficient(degree, 0), &leading_inverse)
            })
            .collect();
        Some(DenseMonicModulus { lower_coefficients })
    }

    /// Multiply two packed univariate polynomials and return a dense
    /// coefficient vector. Prime fields use their delayed-reduction kernel.
    fn dense_univariate_mul(
        ring: &F,
        left: &DenseIndexedUnivariate<F::Element>,
        right: &DenseIndexedUnivariate<F::Element>,
    ) -> Option<Vec<F::Element>> {
        let (Some(&left_degree), Some(&right_degree)) = (left.indices.last(), right.indices.last())
        else {
            return Some(Vec::new());
        };
        let output_len = (left_degree as usize)
            .checked_add(right_degree as usize)?
            .checked_add(1)?;

        if let Some(product) = ring.kernels().polynomial().and_then(|kernels| {
            kernels.try_dense_mul(DensePolynomialMulRequest {
                output_len,
                left_coefficients: &left.coefficients,
                left_indices: &left.indices,
                right_coefficients: &right.coefficients,
                right_indices: &right.indices,
            })
        }) {
            let mut coefficients = vec![ring.zero(); output_len];
            for (degree, coefficient) in product {
                let degree = degree as usize;
                if degree >= output_len {
                    return None;
                }
                coefficients[degree] = coefficient;
            }
            while coefficients
                .last()
                .is_some_and(|coefficient| ring.is_zero(coefficient))
            {
                coefficients.pop();
            }
            return Some(coefficients);
        }

        let mut coefficients = vec![ring.zero(); output_len];
        for (left_coefficient, &left_degree) in left.coefficients.iter().zip(&left.indices) {
            for (right_coefficient, &right_degree) in right.coefficients.iter().zip(&right.indices)
            {
                ring.add_mul_assign(
                    &mut coefficients[left_degree as usize + right_degree as usize],
                    left_coefficient,
                    right_coefficient,
                );
            }
        }
        while coefficients
            .last()
            .is_some_and(|coefficient| ring.is_zero(coefficient))
        {
            coefficients.pop();
        }
        Some(coefficients)
    }

    /// Reduce a dense univariate polynomial in place modulo a cached monic
    /// polynomial and return its canonical remainder.
    fn dense_remainder_monic(
        ring: &F,
        mut coefficients: Vec<F::Element>,
        modulus: &DenseMonicModulus<F::Element>,
    ) -> Vec<F::Element> {
        let modulus_degree = modulus.lower_coefficients.len();
        if modulus_degree == 0 {
            return Vec::new();
        }

        while coefficients.len() > modulus_degree {
            let leading = coefficients.pop().unwrap();
            if ring.is_zero(&leading) {
                continue;
            }
            let shift = coefficients.len() - modulus_degree;
            for (coefficient, modulus_coefficient) in coefficients[shift..]
                .iter_mut()
                .zip(&modulus.lower_coefficients)
            {
                ring.sub_mul_assign(coefficient, modulus_coefficient, &leading);
            }
        }
        while coefficients
            .last()
            .is_some_and(|coefficient| ring.is_zero(coefficient))
        {
            coefficients.pop();
        }
        coefficients
    }

    /// Solve both modular correction equations for one sampled error row and
    /// pad the remainders to their factor image widths.
    fn dense_two_factor_corrections(
        &self,
        error: &[F::Element],
        context: &DenseTwoFactorCorrectionContext<F::Element>,
        output_lens: [usize; 2],
    ) -> Option<[Vec<F::Element>; 2]> {
        if error
            .iter()
            .all(|coefficient| self.ring().is_zero(coefficient))
        {
            return Some([
                vec![self.ring().zero(); output_lens[0]],
                vec![self.ring().zero(); output_lens[1]],
            ]);
        }
        let error = Self::dense_indexed_univariate_from_coefficients(&self.ring(), error)?;
        let solve = |factor_index: usize| {
            let product = Self::dense_univariate_mul(
                &self.ring(),
                &context.multipliers[factor_index],
                &error,
            )?;
            let mut correction =
                Self::dense_remainder_monic(&self.ring(), product, &context.moduli[factor_index]);
            if correction.len() > output_lens[factor_index] {
                return None;
            }
            correction.resize(output_lens[factor_index], self.ring().zero());
            Some(correction)
        };
        Some([solve(0)?, solve(1)?])
    }

    /// Compare `left * right` with `self` by accumulating coefficient
    /// differences under exact mixed-radix exponent keys.
    fn product_matches_by_packed_accumulation(&self, left: &Self, right: &Self) -> Option<bool> {
        if self.nvars() != left.nvars()
            || self.nvars() != right.nvars()
            || self.ring() != left.ring()
            || self.ring() != right.ring()
            || self.variables() != left.variables()
            || self.variables() != right.variables()
            || !self.is_polynomial()
            || !left.is_polynomial()
            || !right.is_polynomial()
        {
            return None;
        }

        let mut strides = Vec::with_capacity(self.nvars());
        let mut radices = Vec::with_capacity(self.nvars());
        let mut stride = 1u128;
        for variable in 0..self.nvars() {
            let maximum_degree = left
                .degree(variable)
                .to_u32()
                .checked_add(right.degree(variable).to_u32())?;
            let radix = maximum_degree as u128 + 1;
            strides.push(stride);
            radices.push(radix);
            stride = stride.checked_mul(radix)?;
        }

        let encode = |exponents: &[E]| {
            let mut key = 0u128;
            for ((exponent, &stride), &radix) in exponents.iter().zip(&strides).zip(&radices) {
                let exponent = exponent.to_u32() as u128;
                if exponent >= radix {
                    return None;
                }
                key = key.checked_add(exponent.checked_mul(stride)?)?;
            }
            Some(key)
        };

        let left_keys = left
            .exponents_iter()
            .map(encode)
            .collect::<Option<Vec<_>>>()?;
        let right_keys = right
            .exponents_iter()
            .map(encode)
            .collect::<Option<Vec<_>>>()?;

        let mut differences: HashMap<u128, F::Element> = HashMap::default();
        differences.reserve(self.nterms());
        for term in self {
            let key = encode(term.exponents)?;
            if differences
                .insert(key, self.ring().neg(term.coefficient))
                .is_some()
            {
                return None;
            }
        }

        for (left_coefficient, &left_key) in left.coefficients.iter().zip(&left_keys) {
            for (right_coefficient, &right_key) in right.coefficients.iter().zip(&right_keys) {
                let key = left_key.checked_add(right_key)?;
                let difference = differences.entry(key).or_insert_with(|| self.ring().zero());
                self.ring()
                    .add_mul_assign(difference, left_coefficient, right_coefficient);
            }
        }

        Some(
            differences
                .values()
                .all(|difference| self.ring().is_zero(difference)),
        )
    }

    /// Return whether exact product verification should accumulate sparse coefficient
    /// products instead of constructing the product through the dense multiplier.
    fn packed_product_accumulation_is_preferred(left: &Self, right: &Self) -> bool {
        if left.nvars() != right.nvars()
            || left.ring() != right.ring()
            || left.variables() != right.variables()
            || !left.is_polynomial()
            || !right.is_polynomial()
            || left.nterms() <= 1
            || right.nterms() <= 1
            || left.total_degree_dense_mul_is_bounded(right)
        {
            return false;
        }

        let mut output_len = Some(1usize);
        let mut active_variables = 0;
        for variable in 0..left.nvars() {
            let Some(maximum_degree) = left
                .degree(variable)
                .to_u32()
                .checked_add(right.degree(variable).to_u32())
            else {
                return false;
            };
            let radix = maximum_degree as usize + 1;
            active_variables += usize::from(radix > 1);
            output_len = output_len.and_then(|length| length.checked_mul(radix));
        }

        active_variables > 1
            && output_len.is_none_or(|length| {
                !super::polynomial::mixed_radix_dense_mul_is_bounded(
                    length,
                    left.nterms(),
                    right.nterms(),
                )
            })
    }

    /// Check a sampled two-factor correction identity using dense coefficient
    /// convolution in `x`.
    #[cfg(debug_assertions)]
    fn dense_two_factor_correction_matches(
        &self,
        error: &[F::Element],
        left: &DenseBivariateImage<F::Element>,
        right: &DenseBivariateImage<F::Element>,
        corrections: &[Vec<F::Element>; 2],
    ) -> bool {
        let Some(first_len) = corrections[0]
            .len()
            .checked_add(right.x_len)
            .and_then(|length| length.checked_sub(1))
        else {
            return false;
        };
        let Some(second_len) = corrections[1]
            .len()
            .checked_add(left.x_len)
            .and_then(|length| length.checked_sub(1))
        else {
            return false;
        };
        let result_len = error.len().max(first_len).max(second_len);
        let mut reconstructed = vec![self.ring().zero(); result_len];

        for (correction, factor) in [(&corrections[0], right), (&corrections[1], left)] {
            for (correction_degree, correction_coefficient) in correction.iter().enumerate() {
                if self.ring().is_zero(correction_coefficient) {
                    continue;
                }
                for factor_degree in 0..factor.x_len {
                    let factor_coefficient = factor.coefficient(factor_degree, 0);
                    if !self.ring().is_zero(factor_coefficient) {
                        self.ring().add_mul_assign(
                            &mut reconstructed[correction_degree + factor_degree],
                            correction_coefficient,
                            factor_coefficient,
                        );
                    }
                }
            }
        }

        reconstructed.iter().enumerate().all(|(degree, actual)| {
            let difference = if let Some(expected) = error.get(degree) {
                self.ring().sub(actual, expected)
            } else {
                actual.clone()
            };
            self.ring().is_zero(&difference)
        })
    }

    fn sparse_multivariate_diophantine_two_factor_by_sampling(
        factors: &[Self],
        prods: &[Self],
        error: &Self,
        skeletons: &[Self],
        order: &[usize],
        context: &mut SparseDiophantineContext<Self, F::Element>,
    ) -> Option<Vec<Self>> {
        if factors.len() != 2 || prods.len() != 2 || skeletons.len() != 2 || order.len() < 2 {
            return None;
        }

        let characteristic = error.ring().characteristic();
        if !characteristic.is_zero() && !characteristic.is_prime(0) {
            return None;
        }

        let sparse_factor = skeletons
            .iter()
            .enumerate()
            .filter(|(_, skeleton)| !skeleton.is_zero())
            .min_by_key(|(_, skeleton)| skeleton.nterms())
            .map(|(factor_index, _)| factor_index)?;
        let dense_factor = 1 - sparse_factor;
        let skeleton = &skeletons[sparse_factor];
        let main_var = order[0];
        let sample_vars = &order[1..];

        let mut groups: Vec<(E, Vec<Vec<E>>, Vec<F::Element>)> = Vec::new();
        for exponent in skeleton.exponents.chunks(skeleton.nvars()) {
            let degree = exponent[main_var];
            if let Some((_, exponents, _)) = groups.iter_mut().find(|(d, _, _)| *d == degree) {
                exponents.push(exponent.to_vec());
            } else {
                groups.push((degree, vec![exponent.to_vec()], Vec::new()));
            }
        }

        let samples_needed = groups
            .iter()
            .map(|(_, exponents, _)| exponents.len())
            .max()
            .unwrap_or(0);
        if samples_needed == 0 || skeleton.nterms() > 512 {
            return None;
        }

        let cached_base_points = context.two_factor_base_points.take().filter(|points| {
            points.len() == sample_vars.len()
                && points
                    .iter()
                    .zip(sample_vars)
                    .all(|((point_var, _), sample_var)| point_var == sample_var)
        });
        if cached_base_points.is_none() {
            context.two_factor_bezout.clear();
        }

        let mut rng = rng();
        let sample_base_attempts =
            SPARSE_MDP_SAMPLE_BASE_ATTEMPTS + usize::from(cached_base_points.is_some());
        let mut cached_base_points = cached_base_points;
        'sample_base: for _ in 0..sample_base_attempts {
            let base_points = cached_base_points.take().unwrap_or_else(|| {
                Self::sparse_interpolation_base_points(error, sample_vars, &mut rng)
            });

            for (_, exponents, sample_generators) in &mut groups {
                sample_generators.clear();
                for exponent in exponents {
                    let generator =
                        Self::evaluate_monomial_exponent(&error.ring(), exponent, &base_points);
                    if !Self::sparse_interpolation_generator_is_usable(
                        error,
                        &generator,
                        sample_generators,
                    ) {
                        context.clear_two_factor_images();
                        continue 'sample_base;
                    }
                    sample_generators.push(generator);
                }
            }

            context.two_factor_base_points = Some(base_points.clone());

            let mut rhs: Vec<Vec<F::Element>> = groups
                .iter()
                .map(|_| Vec::with_capacity(samples_needed))
                .collect();

            let mut cache = Self::sample_cache(error, factors, prods);
            let error_base = error.evaluate_exponents(&base_points, &mut cache);
            let factors_base = factors
                .iter()
                .map(|f| f.evaluate_exponents(&base_points, &mut cache))
                .collect::<Vec<_>>();

            let mut error_current = Cow::Borrowed(&error_base);
            let mut factors_current = factors_base.iter().map(Cow::Borrowed).collect::<Vec<_>>();

            let mut error_image =
                error.zero_with_capacity(error.degree(main_var).to_u32() as usize + 1);
            let mut factor_images = factors
                .iter()
                .map(|f| f.zero_with_capacity(f.degree(main_var).to_u32() as usize + 1))
                .collect::<Vec<_>>();

            for sample_index in 0..samples_needed {
                if sample_index > 0 {
                    for (current, base) in error_current.to_mut().iter_mut().zip(&error_base) {
                        error.ring().mul_assign(current, base);
                    }
                    for (current, base) in factors_current.iter_mut().zip(&factors_base) {
                        for (c, b) in current.to_mut().iter_mut().zip(base) {
                            error.ring().mul_assign(c, b);
                        }
                    }
                }

                error.evaluate_using_exponents_univariate_grouped(
                    &error_current,
                    main_var,
                    &mut error_image,
                );
                for ((factor, current), image) in
                    factors.iter().zip(&factors_current).zip(&mut factor_images)
                {
                    factor.evaluate_using_exponents_univariate_grouped(current, main_var, image);
                }

                let Some(delta_image) = Self::try_two_factor_univariate_correction(
                    &mut factor_images,
                    &error_image,
                    sparse_factor,
                    context,
                ) else {
                    context.clear_two_factor_images();
                    continue 'sample_base;
                };

                for (group_index, (degree, _, _)) in groups.iter().enumerate() {
                    rhs[group_index].push(Self::univariate_coefficient(
                        &delta_image,
                        main_var,
                        *degree,
                    ));
                }
            }

            let mut sparse_delta = skeleton.zero();
            for ((_, exponents, sample_generators), rhs) in groups.iter().zip(&rhs) {
                let coefficients = error.solve_shifted_transposed_vandermonde(
                    sample_generators,
                    &rhs[..exponents.len()],
                );
                for (coefficient, exponent) in coefficients.into_iter().zip(exponents) {
                    if !error.ring().is_zero(&coefficient) {
                        sparse_delta.append_monomial(coefficient, exponent);
                    }
                }
            }

            let residual = error - &(&sparse_delta * &prods[sparse_factor]);
            // Exact division reconstructs the other delta and verifies that the
            // interpolated sparse correction satisfies the Diophantine identity.
            let Some(dense_delta) = residual.try_div_owned(&prods[dense_factor]) else {
                context.clear_two_factor_images();
                continue;
            };

            let mut deltas = vec![error.zero(), error.zero()];
            deltas[sparse_factor] = sparse_delta;
            deltas[dense_factor] = dense_delta;
            return Some(deltas);
        }

        None
    }

    fn sparse_multivariate_diophantine_by_sampling(
        factors: &[Self],
        prods: &[Self],
        error: &Self,
        skeletons: &[Self],
        order: &[usize],
    ) -> Option<Vec<Self>> {
        if factors.len() != skeletons.len() || factors.len() != prods.len() || order.len() < 2 {
            return None;
        }

        let main_var = order[0];
        let sample_vars = &order[1..];

        let mut groups = Vec::with_capacity(skeletons.len());
        let mut samples_needed = 0usize;
        let mut total_terms = 0usize;
        for skeleton in skeletons {
            let mut factor_groups: Vec<(E, Vec<Vec<E>>, Vec<F::Element>)> = Vec::new();
            for exponent in skeleton.exponents.chunks(skeleton.nvars()) {
                total_terms += 1;
                let degree = exponent[main_var];
                if let Some((_, exponents, _)) =
                    factor_groups.iter_mut().find(|(d, _, _)| *d == degree)
                {
                    exponents.push(exponent.to_vec());
                } else {
                    factor_groups.push((degree, vec![exponent.to_vec()], Vec::new()));
                }
            }

            for (_, exponents, _) in &factor_groups {
                samples_needed = samples_needed.max(exponents.len());
            }
            groups.push(factor_groups);
        }

        if samples_needed == 0 || total_terms > 512 {
            return None;
        }

        let mut rng = rng();
        'sample_base: for _ in 0..SPARSE_MDP_SAMPLE_BASE_ATTEMPTS {
            let base_points = Self::sparse_interpolation_base_points(error, sample_vars, &mut rng);

            for factor_groups in &mut groups {
                for (_, exponents, sample_generators) in factor_groups {
                    sample_generators.clear();
                    for exponent in exponents {
                        let generator =
                            Self::evaluate_monomial_exponent(&error.ring(), exponent, &base_points);
                        if !Self::sparse_interpolation_generator_is_usable(
                            error,
                            &generator,
                            sample_generators,
                        ) {
                            continue 'sample_base;
                        }
                        sample_generators.push(generator);
                    }
                }
            }

            let mut rhs: Vec<Vec<Vec<F::Element>>> = groups
                .iter()
                .map(|factor_groups| {
                    factor_groups
                        .iter()
                        .map(|_| Vec::with_capacity(samples_needed))
                        .collect()
                })
                .collect();

            let mut cache = Self::sample_cache(error, factors, prods);
            let error_base = error.evaluate_exponents(&base_points, &mut cache);
            let factors_base = factors
                .iter()
                .map(|f| f.evaluate_exponents(&base_points, &mut cache))
                .collect::<Vec<_>>();

            let mut error_current = Cow::Borrowed(&error_base);
            let mut factors_current = factors_base.iter().map(Cow::Borrowed).collect::<Vec<_>>();

            let mut error_image =
                error.zero_with_capacity(error.degree(main_var).to_u32() as usize + 1);
            let mut factor_images = factors
                .iter()
                .map(|f| f.zero_with_capacity(f.degree(main_var).to_u32() as usize + 1))
                .collect::<Vec<_>>();

            for sample_index in 0..samples_needed {
                if sample_index > 0 {
                    for (current, base) in error_current.to_mut().iter_mut().zip(&error_base) {
                        error.ring().mul_assign(current, base);
                    }
                    for (current, base) in factors_current.iter_mut().zip(&factors_base) {
                        for (c, b) in current.to_mut().iter_mut().zip(base) {
                            error.ring().mul_assign(c, b);
                        }
                    }
                }

                error.evaluate_using_exponents_univariate_grouped(
                    &error_current,
                    main_var,
                    &mut error_image,
                );
                for ((factor, current), image) in
                    factors.iter().zip(&factors_current).zip(&mut factor_images)
                {
                    factor.evaluate_using_exponents_univariate_grouped(current, main_var, image);
                }

                let Some(deltas_image) =
                    Self::try_univariate_diophantine(&mut factor_images, &error_image)
                else {
                    continue 'sample_base;
                };

                for (factor_index, factor_groups) in groups.iter().enumerate() {
                    for (group_index, (degree, _, _)) in factor_groups.iter().enumerate() {
                        rhs[factor_index][group_index].push(Self::univariate_coefficient(
                            &deltas_image[factor_index],
                            main_var,
                            *degree,
                        ));
                    }
                }
            }

            let mut deltas: Vec<_> = skeletons.iter().map(|s| s.zero()).collect();
            for (factor_index, factor_groups) in groups.iter().enumerate() {
                for (group_index, (_, exponents, sample_generators)) in
                    factor_groups.iter().enumerate()
                {
                    let coefficients = error.solve_shifted_transposed_vandermonde(
                        sample_generators,
                        &rhs[factor_index][group_index][..exponents.len()],
                    );
                    for (coefficient, exponent) in coefficients.into_iter().zip(exponents) {
                        if !error.ring().is_zero(&coefficient) {
                            deltas[factor_index].append_monomial(coefficient, exponent);
                        }
                    }
                }
            }

            let mut check = error.zero();
            for (delta, prod) in deltas.iter().zip(prods) {
                check = check + delta * prod;
            }

            if &check == error {
                return Some(deltas);
            }
        }

        None
    }

    fn sparse_interpolation_base_points(
        poly: &Self,
        sample_vars: &[usize],
        rng: &mut impl rand::RngCore,
    ) -> Vec<(usize, F::Element)> {
        let upper = match poly.ring().characteristic().to_i64() {
            Some(characteristic) if characteristic > 0 => characteristic - 1,
            _ => MAX_RNG_PREFACTOR as i64 - 1,
        };
        let policy = 0..=upper;
        sample_vars
            .iter()
            .map(|v| {
                let mut value = poly.ring().sample(rng, &policy);
                let mut attempts = 0;
                while poly.ring().is_zero(&value) && attempts < 8 {
                    value = poly.ring().sample(rng, &policy);
                    attempts += 1;
                }
                (*v, value)
            })
            .collect()
    }

    fn sparse_interpolation_generator_is_usable(
        poly: &Self,
        generator: &F::Element,
        previous_generators: &[F::Element],
    ) -> bool {
        if poly.ring().is_zero(generator) {
            return false;
        }

        for prev in previous_generators {
            let diff = poly.ring().sub(generator, prev);
            if poly.ring().is_zero(&diff) {
                return false;
            }
        }

        true
    }

    fn evaluate_monomial_exponent(
        ring: &F,
        exponent: &[E],
        sample_points: &[(usize, F::Element)],
    ) -> F::Element {
        let mut value = ring.one();
        for (var, sample) in sample_points {
            let e = exponent[*var].to_u32();
            if e > 0 {
                ring.mul_assign(&mut value, &ring.pow(sample, e as u64));
            }
        }
        value
    }

    fn sample_cache(f: &Self, factors: &[Self], prods: &[Self]) -> Vec<Vec<F::Element>> {
        let mut degrees = (0..f.nvars())
            .map(|var| f.degree(var).to_u32() as usize)
            .collect::<Vec<_>>();
        for p in factors.iter().chain(prods) {
            for (var, degree) in degrees.iter_mut().enumerate() {
                *degree = (*degree).max(p.degree(var).to_u32() as usize);
            }
        }

        degrees
            .into_iter()
            .map(|degree| vec![f.ring().zero(); (degree + 1).min(POW_CACHE_SIZE)])
            .collect()
    }

    fn univariate_coefficient(poly: &Self, var: usize, degree: E) -> F::Element {
        let mut exponent = vec![E::zero(); poly.nvars()];
        exponent[var] = degree;
        poly.coefficient(&exponent)
            .unwrap_or_else(|| poly.ring().zero())
    }

    /// Solve the two-factor univariate Diophantine equation for one requested
    /// correction. For `f0 * s + f1 * t = 1`, correction zero is
    /// `t * rhs mod f0` and correction one is `s * rhs mod f1`.
    fn try_two_factor_univariate_correction(
        factors: &mut [Self],
        rhs: &Self,
        requested_factor: usize,
        context: &mut SparseDiophantineContext<Self, F::Element>,
    ) -> Option<Self> {
        if factors.len() != 2
            || requested_factor >= 2
            || factors
                .iter()
                .any(|factor| factor.ring().is_zero(&factor.lcoeff()))
        {
            return None;
        }

        let key = (factors[0].clone(), factors[1].clone());
        let bezout = context
            .two_factor_bezout
            .entry(key)
            .or_insert_with(|| {
                let (g, s, t) = factors[0].eea_univariate(&factors[1]);
                g.is_one().then_some(TwoFactorImageBezout { s, t })
            })
            .as_ref()?;

        if requested_factor == 0 {
            Some((&bezout.t * rhs).quot_rem_univariate(&mut factors[0]).1)
        } else {
            Some((&bezout.s * rhs).quot_rem_univariate(&mut factors[1]).1)
        }
    }

    fn try_univariate_diophantine(factors: &mut [Self], rhs: &Self) -> Option<Vec<Self>> {
        if factors
            .iter()
            .any(|factor| factor.ring().is_zero(&factor.lcoeff()))
        {
            return None;
        }

        let mut cur = factors.last()?.clone();
        let mut products = vec![cur.clone()];
        for factor in factors[1..].iter().rev().skip(1) {
            cur = cur * factor;
            products.push(cur.clone());
        }
        products.reverse();

        let mut deltas = Vec::with_capacity(factors.len());
        let mut cur_s = rhs.clone();
        for (factor, product) in factors.iter_mut().zip(&mut products) {
            let (g, s, t) = factor.eea_univariate(product);
            if !g.is_one() {
                return None;
            }

            let new_s = (t * &cur_s).quot_rem_univariate(factor).1;
            deltas.push(new_s);
            cur_s = (s * &cur_s).quot_rem_univariate(product).1;
        }

        deltas.push(cur_s);
        Some(deltas)
    }

    fn impose_true_lcoeffs_on_factors(
        &self,
        factors: &[Self],
        true_lcoeffs: &[Self],
        sample_points: &[(usize, F::Element)],
        order: &[usize],
        current_index: usize,
    ) -> Vec<Self> {
        let mut factors_with_true_lcoeff = Vec::with_capacity(factors.len());

        for (factor, true_lcoeff) in factors.iter().zip(true_lcoeffs) {
            let mut lcoeff = true_lcoeff.clone();
            for &var in &order[current_index + 1..] {
                if let Some((_, sample)) = sample_points.iter().find(|(v, _)| *v == var) {
                    lcoeff = lcoeff.replace(var, sample);
                }
            }

            let mut coefficients = factor.to_univariate_polynomial_list(order[0]);
            coefficients.last_mut().unwrap().0 = lcoeff;

            let mut fixed_factor = self.zero();
            let mut exp = vec![E::zero(); self.nvars()];
            for (coefficient, degree) in coefficients {
                exp[order[0]] = degree;
                fixed_factor = fixed_factor + coefficient.mul_exp(&exp);
            }

            factors_with_true_lcoeff.push(fixed_factor);
        }

        factors_with_true_lcoeff
    }

    /// Lift two factors through the final variable using dense bivariate
    /// geometric images. Sparse interpolation reconstructs both corrections at
    /// each lifted coefficient. When `verify_product` is true, the completed
    /// factors are also certified against the target before they are returned.
    fn try_multivariate_hensel_step_two_factor_evaluated(
        &self,
        factors: &[Self],
        order: &[usize],
        last_degree: usize,
        verify_product: bool,
    ) -> Option<Vec<Self>> {
        if factors.len() != 2 || order.len() <= 2 {
            return None;
        }

        let characteristic = self.ring().characteristic();
        if !characteristic.is_zero() && !characteristic.is_prime(0) {
            return None;
        }

        let x = order[0];
        let y = *order.last()?;
        if x == y {
            return None;
        }
        let y_rows = last_degree.checked_add(1)?;
        if y_rows > MAX_EVALUATED_HENSEL_Y_ROWS {
            return None;
        }
        let sample_vars = &order[1..order.len() - 1];

        let mut u = Vec::with_capacity(2);
        for factor in factors {
            let mut coefficients = vec![self.zero(); y_rows];
            for (coefficient, degree) in factor.to_univariate_polynomial_list(y) {
                let degree = degree.to_u32() as usize;
                if degree > last_degree {
                    return None;
                }
                coefficients[degree] = coefficient;
            }
            u.push(coefficients);
        }

        let expected_factor_x_degrees = [
            u[0][0].degree(x).to_u32() as usize,
            u[1][0].degree(x).to_u32() as usize,
        ];
        if u.iter().any(|factor| factor[0].is_zero()) {
            return None;
        }

        let mut rng = rng();
        let base_points = Self::sparse_interpolation_base_points(self, sample_vars, &mut rng);
        if base_points
            .iter()
            .any(|(_, point)| self.ring().is_zero(point))
        {
            return None;
        }

        let mut image_cache = Self::sample_cache(self, factors, &[]);
        let mut target_images = Vec::new();
        let mut factor_images = vec![Vec::new(), Vec::new()];
        let mut factor_correction_contexts = Vec::new();
        let mut sample_count = 0usize;

        for k in 1..=last_degree {
            let mut sampled_errors = Vec::with_capacity(sample_count);
            if sample_count > 0 {
                let mut sampled_error_is_zero = true;
                for sample_index in 0..sample_count {
                    let error = self.dense_bivariate_product_error_row(
                        &target_images[sample_index],
                        &factor_images[0][sample_index],
                        &factor_images[1][sample_index],
                        k,
                    );
                    let error_is_zero = error
                        .iter()
                        .all(|coefficient| self.ring().is_zero(coefficient));
                    sampled_errors.push(error);
                    if !error_is_zero {
                        sampled_error_is_zero = false;
                        break;
                    }
                }
                if sampled_error_is_zero {
                    continue;
                }
            }

            let skeletons = [&u[0][k - 1], &u[1][k - 1]];
            if skeletons.iter().any(|skeleton| skeleton.nterms() > 512) {
                return None;
            }

            let mut groups: Vec<Vec<(E, Vec<Vec<E>>, Vec<F::Element>)>> = Vec::with_capacity(2);
            let mut samples_needed = 0usize;
            for skeleton in skeletons {
                let mut factor_groups: Vec<(E, Vec<Vec<E>>, Vec<F::Element>)> = Vec::new();
                for exponent in skeleton.exponents.chunks(skeleton.nvars()) {
                    if exponent[y] != E::zero() {
                        return None;
                    }
                    let x_degree = exponent[x];
                    if let Some((_, exponents, _)) = factor_groups
                        .iter_mut()
                        .find(|(degree, _, _)| *degree == x_degree)
                    {
                        exponents.push(exponent.to_vec());
                    } else {
                        factor_groups.push((x_degree, vec![exponent.to_vec()], Vec::new()));
                    }
                }

                for (_, exponents, generators) in &mut factor_groups {
                    samples_needed = samples_needed.max(exponents.len());
                    for exponent in exponents {
                        let generator =
                            Self::evaluate_monomial_exponent(&self.ring(), exponent, &base_points);
                        if !Self::sparse_interpolation_generator_is_usable(
                            self, &generator, generators,
                        ) {
                            return None;
                        }
                        generators.push(generator);
                    }
                }
                groups.push(factor_groups);
            }

            if samples_needed == 0 {
                return None;
            }

            if samples_needed > sample_count {
                let current_factors = u
                    .iter()
                    .map(|coefficients| self.from_dense_y_coefficients(coefficients, y))
                    .collect::<Vec<_>>();
                let mut retained_image_cells = 0usize;
                let mut term_steps = 0usize;
                for polynomial in std::iter::once(self).chain(&current_factors) {
                    if polynomial.nterms() > MAX_EVALUATED_HENSEL_GROUPED_TERMS {
                        return None;
                    }
                    let x_len = (polynomial.degree(x).to_u32() as usize).checked_add(1)?;
                    let image_cells = x_len.checked_mul(y_rows)?;
                    retained_image_cells = retained_image_cells
                        .checked_add(image_cells.checked_mul(samples_needed)?)?;
                    term_steps =
                        term_steps.checked_add(polynomial.nterms().checked_mul(samples_needed)?)?;
                    if retained_image_cells > MAX_EVALUATED_HENSEL_IMAGE_CELLS
                        || term_steps > MAX_EVALUATED_HENSEL_TERM_STEPS
                    {
                        return None;
                    }
                }
                target_images = self.evaluate_geometric_bivariate_images(
                    x,
                    y,
                    &base_points,
                    samples_needed,
                    &mut image_cache,
                );
                factor_images = current_factors
                    .iter()
                    .map(|factor| {
                        factor.evaluate_geometric_bivariate_images(
                            x,
                            y,
                            &base_points,
                            samples_needed,
                            &mut image_cache,
                        )
                    })
                    .collect();

                for image in target_images
                    .iter_mut()
                    .chain(factor_images.iter_mut().flatten())
                {
                    if image.y_len > y_rows {
                        return None;
                    }
                    let image_len = image.x_len.checked_mul(y_rows)?;
                    image.coefficients.resize(image_len, self.ring().zero());
                    image.y_len = y_rows;
                }

                for sample_index in factor_correction_contexts.len()..samples_needed {
                    let images = [
                        self.dense_bivariate_x_row_to_univariate(
                            &factor_images[0][sample_index],
                            x,
                            0,
                        ),
                        self.dense_bivariate_x_row_to_univariate(
                            &factor_images[1][sample_index],
                            x,
                            0,
                        ),
                    ];
                    for (factor_index, image) in images.iter().enumerate() {
                        if image.is_zero()
                            || image.degree(x).to_u32() as usize
                                != expected_factor_x_degrees[factor_index]
                        {
                            return None;
                        }
                    }
                    let (gcd, s, t) = images[0].eea_univariate(&images[1]);
                    if !gcd.is_one() {
                        return None;
                    }
                    factor_correction_contexts.push(DenseTwoFactorCorrectionContext {
                        multipliers: [
                            self.dense_indexed_univariate(&t, x)?,
                            self.dense_indexed_univariate(&s, x)?,
                        ],
                        moduli: [
                            self.dense_monic_modulus(&factor_images[0][sample_index])?,
                            self.dense_monic_modulus(&factor_images[1][sample_index])?,
                        ],
                    });
                }
                sample_count = samples_needed;
            }

            let supported_x_degrees = groups
                .iter()
                .enumerate()
                .map(|(factor_index, factor_groups)| {
                    let x_len = factor_images[factor_index][0].x_len;
                    let mut supported = vec![false; x_len];
                    for (degree, _, _) in factor_groups {
                        let degree = degree.to_u32() as usize;
                        if degree >= x_len {
                            return None;
                        }
                        supported[degree] = true;
                    }
                    Some(supported)
                })
                .collect::<Option<Vec<_>>>()?;

            let mut rhs: Vec<Vec<Vec<F::Element>>> = groups
                .iter()
                .map(|factor_groups| {
                    factor_groups
                        .iter()
                        .map(|_| Vec::with_capacity(samples_needed))
                        .collect()
                })
                .collect();
            sampled_errors.extend((sampled_errors.len()..samples_needed).map(|sample_index| {
                self.dense_bivariate_product_error_row(
                    &target_images[sample_index],
                    &factor_images[0][sample_index],
                    &factor_images[1][sample_index],
                    k,
                )
            }));
            for (sample_index, error_coefficients) in
                sampled_errors.iter().take(samples_needed).enumerate()
            {
                let correction_coefficients = self.dense_two_factor_corrections(
                    error_coefficients,
                    &factor_correction_contexts[sample_index],
                    [
                        factor_images[0][sample_index].x_len,
                        factor_images[1][sample_index].x_len,
                    ],
                );
                let correction_coefficients = correction_coefficients?;
                for factor_index in 0..2 {
                    if correction_coefficients[factor_index]
                        .iter()
                        .enumerate()
                        .any(|(degree, coefficient)| {
                            !self.ring().is_zero(coefficient)
                                && !supported_x_degrees[factor_index]
                                    .get(degree)
                                    .copied()
                                    .unwrap_or(false)
                        })
                    {
                        return None;
                    }
                }
                #[cfg(debug_assertions)]
                {
                    if !self.dense_two_factor_correction_matches(
                        error_coefficients,
                        &factor_images[0][sample_index],
                        &factor_images[1][sample_index],
                        &correction_coefficients,
                    ) {
                        return None;
                    }
                }

                for (factor_index, correction_coefficients) in
                    correction_coefficients.into_iter().enumerate()
                {
                    for (group_index, (degree, _, _)) in groups[factor_index].iter().enumerate() {
                        rhs[factor_index][group_index]
                            .push(correction_coefficients[degree.to_u32() as usize].clone());
                    }
                }
            }

            let mut deltas = [self.zero(), self.zero()];
            let geometric_sequence_kernels = self.ring().kernels().geometric_sequences();
            for factor_index in 0..2 {
                for (group_index, (degree, exponents, generators)) in
                    groups[factor_index].iter().enumerate()
                {
                    let coefficients = self.solve_shifted_transposed_vandermonde(
                        generators,
                        &rhs[factor_index][group_index][..exponents.len()],
                    );
                    let mut current = coefficients
                        .iter()
                        .zip(generators)
                        .map(|(coefficient, generator)| self.ring().mul(coefficient, generator))
                        .collect::<Vec<_>>();
                    let x_degree = degree.to_u32() as usize;
                    for sample_index in 0..sample_count {
                        let reconstructed = (sample_index + 1 < sample_count)
                            .then(|| geometric_sequence_kernels)
                            .flatten()
                            .and_then(|kernels| {
                                kernels.try_sum_and_advance_geometric_sequences(
                                    GeometricSequenceStepRequest {
                                        current: &mut current,
                                        ratios: generators,
                                    },
                                )
                            })
                            .unwrap_or_else(|| {
                                let mut reconstructed = self.ring().zero();
                                for (current, generator) in current.iter_mut().zip(generators) {
                                    self.ring().add_assign(&mut reconstructed, &*current);
                                    if sample_index + 1 < sample_count {
                                        self.ring().mul_assign(current, generator);
                                    }
                                }
                                reconstructed
                            });

                        if sample_index < samples_needed {
                            let expected = &rhs[factor_index][group_index][sample_index];
                            let difference = self.ring().sub(&reconstructed, expected);
                            if !self.ring().is_zero(&difference) {
                                return None;
                            }
                        }

                        if !self.ring().is_zero(&reconstructed) {
                            let image = &mut factor_images[factor_index][sample_index];
                            let index = image.index(x_degree, k);
                            self.ring()
                                .add_assign(&mut image.coefficients[index], reconstructed);
                        }
                    }

                    for (coefficient, exponent) in coefficients.into_iter().zip(exponents) {
                        if !self.ring().is_zero(&coefficient) {
                            deltas[factor_index].append_monomial(coefficient, exponent);
                        }
                    }
                }
            }

            for factor_index in 0..2 {
                u[factor_index][k] = &u[factor_index][k] + &deltas[factor_index];
            }
        }

        let lifted = u
            .iter()
            .map(|coefficients| self.from_dense_y_coefficients(coefficients, y))
            .collect::<Vec<_>>();
        if verify_product {
            let product_matches =
                Self::packed_product_accumulation_is_preferred(&lifted[0], &lifted[1])
                    .then(|| self.product_matches_by_packed_accumulation(&lifted[0], &lifted[1]))
                    .flatten()
                    .unwrap_or_else(|| {
                        let product = &lifted[0] * &lifted[1];
                        &product == self
                    });
            if !product_matches {
                return None;
            }
        }
        Some(lifted)
    }

    fn multivariate_hensel_lifting(
        &self,
        factors: &[Self],
        univariate_factors: &mut [Self],
        univariate_deltas: &[Self],
        sample_points: &[(usize, F::Element)],
        true_lcoeffs: Option<&[Self]>,
        order: &[usize],
        context: MultivariateHenselContext,
    ) -> Result<Vec<Self>, MultivariateHenselError> {
        debug!("Hensel lift {} with order {:?}", self, order);

        let mut degrees: Vec<_> = order
            .iter()
            .map(|v| self.degree(*v).to_u32() as usize)
            .collect();

        // Build each nested specialization once. Stage v keeps order[..=v]
        // symbolic and evaluates every later variable at its sample point.
        let mut specialized_targets: Vec<Option<Self>> =
            std::iter::repeat_with(|| None).take(order.len()).collect();
        let mut specialized_target = self.clone();
        for v in (context.start_index..order.len()).rev() {
            specialized_targets[v] = Some(specialized_target.clone());
            if v > context.start_index {
                let variable = order[v];
                let sample = &sample_points
                    .iter()
                    .find(|(sample_var, _)| *sample_var == variable)
                    .unwrap()
                    .1;
                specialized_target = specialized_target.replace(variable, sample);
            }
        }

        let mut reconstructed_factors = factors.to_vec();
        let mut used_evaluated_lift = false;
        for v in context.start_index..order.len() {
            // Replace the leading coefficient in x0 before this lift step.
            let mut factors_with_true_lcoeff = if let Some(true_lcoeffs) = true_lcoeffs {
                self.impose_true_lcoeffs_on_factors(
                    &reconstructed_factors,
                    true_lcoeffs,
                    sample_points,
                    order,
                    v,
                )
            } else {
                reconstructed_factors
            };

            let mut f = specialized_targets[v]
                .take()
                .expect("every Hensel stage must have a specialized target");

            // shift the polynomial such that the evaluation point is now at 0
            // so that we can use a convolution for fast error computation
            let shift = &sample_points.iter().find(|s| s.0 == order[v]).unwrap().1;
            if !self.ring().is_zero(shift) {
                f = f.shift_var_cached(order[v], shift);

                for f in &mut factors_with_true_lcoeff {
                    *f = f.shift_var_cached(order[v], shift);
                }
            }

            reconstructed_factors = f.multivariate_hensel_step(
                univariate_deltas,
                univariate_factors,
                sample_points,
                &mut factors_with_true_lcoeff,
                &order[..=v],
                &mut degrees[..=v],
                context,
                &mut used_evaluated_lift,
            )?;

            if !self.ring().is_zero(shift) {
                for f in &mut reconstructed_factors {
                    *f = f.shift_var_cached(order[v], &self.ring().neg(shift));
                }
            }

            for f in &reconstructed_factors {
                debug!("Reconstructed factor {}", f);
            }
        }

        // Evaluated stages reconstruct from sampled sparse supports. Certify the
        // completed, unshifted factors once after all nested stages have finished.
        if used_evaluated_lift {
            let product_matches = (reconstructed_factors.len() == 2
                && Self::packed_product_accumulation_is_preferred(
                    &reconstructed_factors[0],
                    &reconstructed_factors[1],
                ))
            .then(|| {
                self.product_matches_by_packed_accumulation(
                    &reconstructed_factors[0],
                    &reconstructed_factors[1],
                )
            })
            .flatten()
            .unwrap_or_else(|| {
                let product = reconstructed_factors
                    .iter()
                    .fold(self.one(), |product, factor| &product * factor);
                &product == self
            });
            if !product_matches {
                return Err(MultivariateHenselError::SparseDiophantineFailed);
            }
        }

        Ok(reconstructed_factors)
    }

    fn multivariate_hensel_step(
        &self,
        univariate_deltas: &[Self],
        univariate_factors: &mut [Self],
        sample_points: &[(usize, F::Element)],
        factors: &mut [Self],
        order: &[usize],
        degrees: &mut [usize],
        context: MultivariateHenselContext,
        used_evaluated_lift: &mut bool,
    ) -> Result<Vec<Self>, MultivariateHenselError> {
        let last_var = *order.last().unwrap();
        let last_degree = *degrees.last().unwrap();

        let defer_product_verification =
            context.sparse_diophantine_fallback == SparseDiophantineFallback::RetrySample;
        if let Some(lifted) = self.try_multivariate_hensel_step_two_factor_evaluated(
            factors,
            order,
            last_degree,
            !defer_product_verification,
        ) {
            *used_evaluated_lift |= defer_product_verification;
            return Ok(lifted);
        }

        // Before a generic stage consumes speculative factors, verify their
        // constant row in the new lifting variable against the target row.
        if *used_evaluated_lift {
            let zero = self.ring().zero();
            let target_at_zero = self.replace(last_var, &zero);
            let product_at_zero = factors.iter().fold(self.one(), |product, factor| {
                &product * &factor.replace(last_var, &zero)
            });
            if product_at_zero != target_at_zero {
                return Err(MultivariateHenselError::SparseDiophantineFailed);
            }
        }

        let y_poly = self.to_univariate_polynomial_list(last_var);

        // extract coefficients in last_var
        let mut u: Vec<_> = factors
            .iter()
            .map(|f| {
                let mut dense = vec![self.zero(); last_degree + 1];

                for (p, e) in f.to_univariate_polynomial_list(last_var) {
                    dense[e.to_u32() as usize] = p;
                }

                dense
            })
            .collect();

        // TODO: do entire initialization here?
        // the loop below cannot be cut short anyway, so it's not extra work to do it all here
        let mut p = u.clone();
        let mut cur_p = p[0][0].clone();
        for x in &mut p.iter_mut().skip(1) {
            for j in &mut *x {
                *j = &*j * &cur_p;
            }

            cur_p = x[0].clone();
        }

        let factors_mod = u
            .iter()
            .map(|factor_coefficients| factor_coefficients[0].clone())
            .collect::<Vec<_>>();
        let prod_mod = if factors_mod.len() == 2 {
            vec![factors_mod[1].clone(), factors_mod[0].clone()]
        } else {
            (0..factors_mod.len())
                .map(|excluded| {
                    factors_mod
                        .iter()
                        .enumerate()
                        .filter(|(index, _)| *index != excluded)
                        .fold(self.one(), |product, (_, factor)| product * factor)
                })
                .collect()
        };

        debug!("in shift {}", self);
        debug!("deg {:?}", degrees);

        let mut sparse_diophantine_context = SparseDiophantineContext::new();

        // create the polynomials (x_i-shift_i)^deg used for modding during Hensel lifting
        let mut mod_vars = Vec::with_capacity(order.len() - 2);
        let mut exp = vec![E::zero(); self.nvars()];
        for r in order[1..order.len() - 1]
            .iter()
            .zip(&degrees[1..order.len() - 1])
        {
            let shift = &sample_points.iter().find(|s| s.0 == *r.0).unwrap().1;
            exp[*r.0] = E::one();
            let var_pow = self
                .monomial(self.ring().one(), exp.clone())
                .shift_var(*r.0, &self.ring().neg(shift))
                .pow(r.1 + 1);
            exp[*r.0] = E::zero();
            mod_vars.push(var_pow);
        }

        for k in 1..=last_degree {
            // extract the coefficient required to compute the error in y^k
            // computed using a convolution
            for i in 1..factors.len() {
                for j in 0..k {
                    if p[i - 1][k - j].is_zero() || u[i][j].is_zero() {
                        continue;
                    }
                    p[i][k] = &p[i][k] + &(&p[i - 1][k - j] * &u[i][j]);
                }
            }

            // find the kth power of y in f
            // since we compute the error per power of y, we cannot stop on a 0 error
            let e = if let Some((v, _)) = y_poly.iter().find(|e| e.1.to_u32() as usize == k) {
                v - &p.last().unwrap()[k]
            } else {
                -p.last().unwrap()[k].clone()
            };

            debug!("hensel e[x{}^{}] = {}", last_var, k, e);

            if e.is_zero() {
                continue;
            }

            let skeletons: Vec<_> = u.iter().map(|ui| ui[k - 1].clone()).collect();
            let sparse_delta = Self::sparse_multivariate_diophantine_from_skeleton(
                &factors_mod,
                &prod_mod,
                &e,
                &skeletons,
                &order[..order.len() - 1],
                &mut sparse_diophantine_context,
            );
            let new_delta = match sparse_delta {
                Some(delta) => delta,
                None if order.len() > 2
                    && context.sparse_diophantine_fallback
                        == SparseDiophantineFallback::RetrySample =>
                {
                    debug!(
                        "Sparse Diophantine correction failed after {} lifted variables; retrying the evaluation sample",
                        order.len() - 1
                    );
                    return Err(MultivariateHenselError::SparseDiophantineFailed);
                }
                None => Self::multivariate_diophantine(
                    univariate_deltas,
                    univariate_factors,
                    &prod_mod,
                    &e,
                    &order[..order.len() - 1],
                    sample_points,
                    &degrees[..order.len() - 1],
                    &mod_vars,
                ),
            };

            // update the coefficients with the new y^k contributions
            let mut t = self.zero();

            for (i, (du, d)) in u.iter_mut().zip(&new_delta).enumerate() {
                debug!("hensel d[x{}^{}] = {}", last_var, k, d);
                du[k] = &du[k] + d;

                if i > 0 {
                    t = &du[0] * &t + d * &p[i - 1][0];
                } else {
                    t = &t + d;
                }

                p[i][k] = &p[i][k] + &t;
            }
        }

        // convert dense polynomials to multivariate polynomials
        let lifted = u
            .into_iter()
            .map(|ts| {
                let mut new_poly = self.zero_with_capacity(ts.len());
                for (i, mut f) in ts.into_iter().enumerate() {
                    for x in f.exponents_iter_mut() {
                        debug_assert_eq!(x[last_var], E::zero());
                        x[last_var] = E::from_u32(i as u32);
                    }
                    new_poly = new_poly + f;
                }
                new_poly
            })
            .collect::<Vec<_>>();

        Ok(lifted)
    }
}

impl<E: PositiveExponent> MultivariatePolynomial<IntegerRing, E, LexOrder> {
    fn integer_factor_start_auto_decision(&self, order: &[usize]) -> (bool, f64) {
        let bivariate_degree_space = order
            .iter()
            .take(2)
            .map(|&var| self.degree(var).to_u32() as f64 + 1.0)
            .product::<f64>();
        let bivariate_box_density = if bivariate_degree_space == 0.0 {
            0.0
        } else {
            self.nterms() as f64 / bivariate_degree_space
        };

        let use_bivariate =
            bivariate_box_density <= INTEGER_FACTOR_BIVARIATE_SPARSE_BOX_DENSITY_THRESHOLD;

        (use_bivariate, bivariate_box_density)
    }

    /// Hensel lift a solution of `self = u * w mod p` to `self = u * w mod max_p`
    /// where `max_p` is a power of `p`.
    ///
    /// If the lifting is successful, i.e. the error is 0 at some stage,
    /// it will return `Ok((u,w))` where `u` and `w` are the true factors over
    /// the integers. If a true factorization is not possible, it returns
    /// `Err((u,w))` where `u` and `w` are monic.
    pub fn hensel_lift<UField: FiniteFieldWorkspace>(
        &self,
        u: MultivariatePolynomial<FiniteField<UField>, E, LexOrder>,
        w: MultivariatePolynomial<FiniteField<UField>, E, LexOrder>,
        gamma: Option<Integer>,
        max_p: &Integer,
    ) -> Result<(Self, Self), (Self, Self)>
    where
        FiniteField<UField>: Field + PolynomialGCD<E> + FiniteFieldCore<UField>,
        Integer: ToFiniteField<UField>,
    {
        self.hensel_lift_with_strategy(u, w, gamma, max_p, true)
    }

    fn hensel_lift_with_strategy<UField: FiniteFieldWorkspace>(
        &self,
        mut u: MultivariatePolynomial<FiniteField<UField>, E, LexOrder>,
        mut w: MultivariatePolynomial<FiniteField<UField>, E, LexOrder>,
        gamma: Option<Integer>,
        max_p: &Integer,
        quadratic_lift_allowed: bool,
    ) -> Result<(Self, Self), (Self, Self)>
    where
        FiniteField<UField>: Field + PolynomialGCD<E> + FiniteFieldCore<UField>,
        Integer: ToFiniteField<UField>,
    {
        let lcoeff = self.lcoeff(); // lcoeff % p != 0
        let mut gamma = gamma.unwrap_or(lcoeff.clone());
        let lcoeff_p = lcoeff.to_finite_field(&u.ring());
        let gamma_p = gamma.to_finite_field(&u.ring());
        let field = u.ring().clone();
        let p = field.get_prime().to_integer();

        let a = self.clone().mul_coeff(gamma.clone());

        u = u.make_monic().mul_coeff(gamma_p.clone());
        w = w.make_monic().mul_coeff(lcoeff_p.clone());

        let (_, s, t) = u.eea_univariate(&w);

        debug_assert!((&s * &u + &t * &w).is_one());

        let mut u_i = u.map_coeff(|c| field.to_symmetric_integer(c), Z);
        let mut w_i = w.map_coeff(|c| field.to_symmetric_integer(c), Z);

        // only replace the leading coefficient
        *u_i.coefficients.last_mut().unwrap() = gamma.clone();
        *w_i.coefficients.last_mut().unwrap() = lcoeff;

        let mut e = &a - &(&u_i * &w_i);

        let mut m = p.clone();

        let use_quadratic_lift = if !quadratic_lift_allowed || p == Integer::Single(2) {
            false
        } else {
            let mut threshold_power = p.clone();
            let mut reaches_threshold = true;
            for _ in 1..MIN_QUADRATIC_HENSEL_DIGITS {
                if &threshold_power >= max_p {
                    reaches_threshold = false;
                    break;
                }
                threshold_power *= &p;
            }
            reaches_threshold
        };

        if use_quadratic_lift {
            #[cfg(test)]
            QUADRATIC_HENSEL_LIFT_CALLS.with(|calls| calls.set(calls.get() + 1));
            let mut s_i = s.map_coeff(|c| field.to_symmetric_integer(c), Z);
            let mut t_i = t.map_coeff(|c| field.to_symmetric_integer(c), Z);

            while !e.is_zero() && &m < max_p {
                // A full round doubles the known p-adic precision. The last round can use the
                // remaining divisor of m so that the resulting modulus is exactly max_p.
                debug_assert!((max_p % &m).is_zero());
                let squared_modulus = &m * &m;
                let step_modulus = if &squared_modulus <= max_p {
                    m.clone()
                } else {
                    max_p / &m
                };
                debug_assert!((&m % &step_modulus).is_zero());
                let next_modulus = &m * &step_modulus;
                let modular_context = IntegerModularUnivariateContext::new(&step_modulus, self);

                let error_quotient = e.map_coeff(
                    |coefficient| {
                        debug_assert!((coefficient % &m).is_zero());
                        coefficient / &m
                    },
                    Z,
                );
                let error_mod = modular_context.reduce(&error_quotient);
                let u_mod = modular_context.reduce(&u_i);
                let w_mod = modular_context.reduce(&w_i);
                let s_mod = modular_context.reduce(&s_i);
                let t_mod = modular_context.reduce(&t_i);

                // If s*u + t*w = 1, division e*s = q*w + r gives
                // e = (e*t + q*u)*w + r*u. These are the two factor corrections.
                let error_times_s = modular_context.multiply(&error_mod, &s_mod);
                let (q, r) = modular_context.quot_rem(&error_times_s, &w_mod);
                let error_times_t = modular_context.multiply(&error_mod, &t_mod);
                let q_times_u = modular_context.multiply(&q, &u_mod);
                let tau = modular_context.add(&error_times_t, &q_times_u);
                u_i = u_i + tau.mul_coeff(m.clone());
                w_i = w_i + r.mul_coeff(m.clone());
                e = &a - &(&u_i * &w_i);

                if e.is_zero() || &next_modulus >= max_p {
                    m = next_modulus;
                    break;
                }

                // Lift the Bezout cofactors to the same doubled modulus. For
                // b=(s*u+t*w-1)/m, solve ds*u+dt*w=-b modulo m and set
                // s'=s+m*ds, t'=t+m*dt.
                let bezout_error = &(&s_i * &u_i) + &(&t_i * &w_i) - u_i.one();
                let negative_bezout_quotient = bezout_error.map_coeff(
                    |coefficient| {
                        debug_assert!((coefficient % &m).is_zero());
                        -(coefficient / &m)
                    },
                    Z,
                );
                let bezout_mod = modular_context.reduce(&negative_bezout_quotient);
                let w_mod = modular_context.reduce(&w_i);
                let bezout_times_s = modular_context.multiply(&bezout_mod, &s_mod);
                let (q, delta_s) = modular_context.quot_rem(&bezout_times_s, &w_mod);
                let bezout_times_t = modular_context.multiply(&bezout_mod, &t_mod);
                let q_times_u = modular_context.multiply(&q, &u_mod);
                let delta_t = modular_context.add(&bezout_times_t, &q_times_u);
                s_i = s_i + delta_s.mul_coeff(m.clone());
                t_i = t_i + delta_t.mul_coeff(m.clone());
                m = next_modulus;
            }
        } else {
            // At precision m, error_quotient is exactly
            // (a - u_i * w_i) / m. Updating it from the p-adic corrections
            // avoids multiplying the two increasingly large lifted factors.
            let divide_by_prime = |mut residual: Self| {
                for coefficient in &mut residual.coefficients {
                    debug_assert!((&*coefficient % &p).is_zero());
                    *coefficient /= &p;
                }
                residual
            };
            let mut error_quotient = divide_by_prime(e);

            while !error_quotient.is_zero() && &m < max_p {
                let e_p = error_quotient.map_coeff(|c| c.to_finite_field(&field), field.clone());
                let (q, r) = (&e_p * &s).quot_rem_univariate(&mut w);
                let tau = &e_p * &t + q * &u;

                let tau_i = tau.map_coeff(|c| field.to_symmetric_integer(c), Z);
                let r_i = r.map_coeff(|c| field.to_symmetric_integer(c), Z);

                // For u' = u + m*tau and w' = w + m*r,
                // (a - u'*w') / m = error_quotient - tau*w - r*u'. Here w is
                // the old w_i and u' is the updated u_i. The right-hand side
                // is divisible coefficient-wise by p.
                let tau_times_w = &tau_i * &w_i;
                u_i = u_i + tau_i.mul_coeff(m.clone());
                let r_times_u = &r_i * &u_i;
                error_quotient = divide_by_prime(error_quotient - tau_times_w - r_times_u);
                w_i = w_i + r_i.mul_coeff(m.clone());

                m = &m * &p;
                debug_assert_eq!(
                    error_quotient.clone().mul_coeff(m.clone()),
                    &a - &(&u_i * &w_i)
                );
            }

            // Only the zero/nonzero state of e is used below. The scaled
            // residual has the same state and avoids rebuilding a - u_i*w_i.
            e = error_quotient;
        }

        if e.is_zero() {
            let content = u_i.content();
            if !content.is_one() {
                u_i = u_i.div_coeff(&content);
                gamma = &gamma / &content;
            }

            if !gamma.is_one() {
                w_i = w_i.div_coeff(&gamma); // true division is possible in this case
            }

            Ok((u_i, w_i))
        } else {
            if !u_i.lcoeff().is_one() {
                let inv = u_i.lcoeff().mod_inverse(&m);
                u_i = u_i.map_coeff(|c| (c * &inv).symmetric_mod(&m), Z);
            }

            if !w_i.lcoeff().is_one() {
                let inv = w_i.lcoeff().mod_inverse(&m);
                w_i = w_i.map_coeff(|c| (c * &inv).symmetric_mod(&m), Z);
            }

            Err((u_i, w_i))
        }
    }

    /// Lift multiple factors by creating a binary tree and lifting each product.
    fn multi_factor_hensel_lift(
        &self,
        hs: &[MultivariatePolynomial<Zp, E, LexOrder>],
        max_p: &Integer,
    ) -> Vec<Self> {
        self.multi_factor_hensel_lift_with_strategy(hs, max_p, hs.len() <= 4)
    }

    fn multi_factor_hensel_lift_with_strategy(
        &self,
        hs: &[MultivariatePolynomial<Zp, E, LexOrder>],
        max_p: &Integer,
        quadratic_lift_allowed: bool,
    ) -> Vec<Self> {
        if hs.len() == 1 {
            if self.lcoeff().is_one() {
                return vec![self.clone()];
            } else {
                let inv = self.lcoeff().mod_inverse(max_p);
                let r = self.map_coeff(|c| (c * &inv).symmetric_mod(max_p), Z);
                return vec![r];
            }
        }

        let (gs, hs) = hs.split_at(hs.len() / 2);

        let mut g = gs[0].one();
        for x in gs {
            g = g * x;
        }

        let mut h = hs[0].one();
        for x in hs {
            h = h * x;
        }

        let (g_i, h_i) = self
            .hensel_lift_with_strategy(g, h, None, max_p, quadratic_lift_allowed)
            .unwrap_or_else(|e| e);

        let mut factors =
            g_i.multi_factor_hensel_lift_with_strategy(gs, max_p, quadratic_lift_allowed);
        factors.extend(h_i.multi_factor_hensel_lift_with_strategy(
            hs,
            max_p,
            quadratic_lift_allowed,
        ));
        factors
    }

    /// Compute distinct-degree data for a suitable univariate image modulo `prime`.
    ///
    /// Equal-degree factorization is deferred until the prime selector retains this candidate.
    /// A candidate whose proven factor-count lower bound exceeds the inclusive limit is reported
    /// separately from a prime that changes the degree or destroys square-freeness.
    fn screen_univariate_mod_prime(
        &self,
        var: usize,
        prime: u32,
        max_factor_count: Option<usize>,
    ) -> Option<ModularPrimeScreen<E>> {
        if (&self.lcoeff() % &Integer::Single(prime as i64)).is_zero() {
            return None;
        }

        let field = Zp::new(prime);
        let image = self.map_coeff(
            |coefficient| coefficient.to_finite_field(&field),
            field.clone(),
        );
        if image.degree(var) != self.degree(var) || !image.gcd(&image.derivative(var)).is_one() {
            return None;
        }

        match image
            .make_monic()
            .distinct_degree_factorization_bounded(max_factor_count)
        {
            Ok(distinct_degree) => {
                debug!(
                    "Prime {prime} yields {} modular factors",
                    distinct_degree.factor_count
                );
                Some(ModularPrimeScreen::Candidate(ModularIntegerFactorization {
                    field,
                    distinct_degree,
                }))
            }
            Err(lower_bound) => {
                debug!(
                    "Prime {prime} exceeds the modular factor limit with lower bound {lower_bound}"
                );
                Some(ModularPrimeScreen::FactorLimitExceeded { lower_bound })
            }
        }
    }

    /// Split the retained distinct-degree blocks into monic irreducible factors.
    fn complete_equal_degree_factorization(
        candidate: ModularIntegerFactorization<E>,
    ) -> (Zp, Vec<MultivariatePolynomial<Zp, E, LexOrder>>) {
        #[cfg(test)]
        {
            MODULAR_INTEGER_EDF_CALLS.with(|calls| calls.set(calls.get() + 1));
            LAST_MODULAR_INTEGER_EDF_PRIME.with(|prime| prime.set(candidate.field.get_prime()));
        }

        let factor_count = candidate.distinct_degree.factor_count;
        let mut factors = Vec::with_capacity(factor_count);
        for (degree, block) in candidate.distinct_degree.blocks {
            debug!("DDF {} {}", block, degree);
            for factor in block.equal_degree_factorization(degree) {
                debug!("EDF {}", factor);
                factors.push(factor);
            }
        }
        debug_assert_eq!(factors.len(), factor_count);
        (candidate.field, factors)
    }

    /// Return the number of base-prime digits and the prime power needed to
    /// exceed an integer factor coefficient bound.
    fn linear_hensel_modulus(bound: &Integer, prime: u32) -> (usize, Integer) {
        let prime = Integer::from(prime);
        let mut modulus = prime.clone();
        let mut digits = 1usize;
        while &modulus < bound {
            modulus *= &prime;
            digits += 1;
        }
        (digits, modulus)
    }

    /// Estimate the number of full correction rounds performed by the binary
    /// tree of linear two-factor Hensel lifts.
    fn linear_hensel_work(factor_count: usize, digits: usize) -> usize {
        factor_count.saturating_sub(1).saturating_mul(digits)
    }

    fn dense_coefficients_mod(&self, var: usize, modulus: &Integer) -> Vec<Integer> {
        let degree = self.degree(var).to_u32() as usize;
        let mut exponents = vec![E::zero(); self.nvars()];
        (0..=degree)
            .map(|power| {
                exponents[var] = E::from_u32(power as u32);
                self.coefficient(&exponents)
                    .unwrap_or_else(Integer::zero)
                    .symmetric_mod(modulus)
            })
            .collect()
    }

    /// Compute the coefficient vector of `self * factor' / factor` modulo `modulus`.
    fn logarithmic_derivative_coefficients(
        &self,
        factor: &Self,
        var: usize,
        modulus: &Integer,
    ) -> Vec<Integer> {
        let dividend = self.dense_coefficients_mod(var, modulus);
        let divisor = factor.dense_coefficients_mod(var, modulus);
        let dividend_degree = dividend.len() - 1;
        let divisor_degree = divisor.len() - 1;
        let leading_inverse = divisor[divisor_degree].mod_inverse(modulus);
        let mut remainder = dividend;
        let mut quotient = vec![Integer::zero(); dividend_degree - divisor_degree + 1];

        for power in (divisor_degree..=dividend_degree).rev() {
            let coefficient = (&remainder[power] * &leading_inverse).symmetric_mod(modulus);
            quotient[power - divisor_degree] = coefficient.clone();
            for (offset, divisor_coefficient) in divisor.iter().enumerate() {
                remainder[power - divisor_degree + offset] = (&remainder
                    [power - divisor_degree + offset]
                    - &coefficient * divisor_coefficient)
                    .symmetric_mod(modulus);
            }
        }

        debug_assert!(
            remainder
                .iter()
                .all(|coefficient| coefficient.clone().symmetric_mod(modulus).is_zero())
        );

        let derivative = divisor
            .iter()
            .enumerate()
            .skip(1)
            .map(|(power, coefficient)| {
                (coefficient * &Integer::from(power)).symmetric_mod(modulus)
            })
            .collect::<Vec<_>>();
        let mut result = vec![Integer::zero(); dividend_degree];
        for (left_power, left) in quotient.iter().enumerate() {
            for (right_power, right) in derivative.iter().enumerate() {
                let index = left_power + right_power;
                result[index] = (&result[index] + left * right).symmetric_mod(modulus);
            }
        }
        result
    }

    /// Use a van-Hoeij-style coefficient-of-logarithmic-derivative lattice to partition lifted
    /// modular factors. Every partition is verified by exact division before it is accepted.
    fn lll_factor_recombination(
        &self,
        factors: &[Self],
        modulus: &Integer,
        var: usize,
    ) -> Option<Vec<Self>> {
        const MAX_DATA_COLUMNS: usize = 8;

        let factor_count = factors.len();
        let degree = self.degree(var).to_u32() as usize;
        if factor_count <= 10 || degree < 2 {
            return None;
        }

        let scale_exponent = (usize::BITS - factor_count.max(20).leading_zeros()) as usize;
        let scale = Integer::Single(2).pow(scale_exponent as u64);
        let retained_bits = factor_count * 3 / 2 + scale_exponent;
        let shift = modulus
            .significant_bits()
            .saturating_sub(retained_bits as u64);
        if shift == 0 {
            return None;
        }
        let divisor = Integer::Single(2).pow(shift);
        let truncated_modulus = modulus / &divisor;

        let logarithmic_derivatives = factors
            .iter()
            .map(|factor| self.logarithmic_derivative_coefficients(factor, var, modulus))
            .collect::<Vec<_>>();
        let data_columns = (0..degree.min(MAX_DATA_COLUMNS))
            .map(|index| {
                if index % 2 == 0 {
                    index / 2
                } else {
                    degree - 1 - index / 2
                }
            })
            .collect::<Vec<_>>();
        let lattice_dimension = factor_count + data_columns.len();
        let mut lattice = Vec::with_capacity(lattice_dimension);

        for factor_index in 0..factor_count {
            let mut row = vec![Integer::zero(); lattice_dimension];
            row[factor_index] = scale.clone();
            for (column, coefficient_index) in data_columns.iter().enumerate() {
                row[factor_count + column] =
                    &logarithmic_derivatives[factor_index][*coefficient_index] / &divisor;
            }
            lattice.push(Vector::new(row, Z));
        }
        for column in 0..data_columns.len() {
            let mut row = vec![Integer::zero(); lattice_dimension];
            row[factor_count + column] = truncated_modulus.clone();
            lattice.push(Vector::new(row, Z));
        }

        let reduced = Vector::basis_reduction_approximate(&lattice, 0.75)?;
        let scale_squared = &scale * &scale;
        let carry_bound = Integer::from(factor_count.div_ceil(2));
        let mut short_bound = &scale_squared * &Integer::from(factor_count + 1);
        short_bound += &Integer::from(data_columns.len()) * &(&carry_bound * &carry_bound);
        short_bound *= 4;

        let mut projected_rows = vec![];
        for row in reduced {
            if row.norm_squared() > short_bound {
                continue;
            }
            let values = row.into_vec();
            if values[..factor_count].iter().all(Integer::is_zero)
                || values[..factor_count]
                    .iter()
                    .any(|value| !(value % &scale).is_zero())
            {
                continue;
            }
            projected_rows.push(
                values[..factor_count]
                    .iter()
                    .map(|value| value / &scale)
                    .collect::<Vec<_>>(),
            );
        }
        if projected_rows.is_empty() {
            return None;
        }

        let mut groups: Vec<(Vec<Integer>, Vec<usize>)> = vec![];
        for factor_index in 0..factor_count {
            let signature = projected_rows
                .iter()
                .map(|row| row[factor_index].clone())
                .collect::<Vec<_>>();
            if let Some((_, indices)) = groups
                .iter_mut()
                .find(|(existing, _)| existing == &signature)
            {
                indices.push(factor_index);
            } else {
                groups.push((signature, vec![factor_index]));
            }
        }
        if groups.len() < 2 {
            return None;
        }
        groups.sort_by_key(|(_, indices)| {
            indices
                .iter()
                .map(|index| factors[*index].degree(var).to_u32())
                .sum::<u32>()
        });

        let mut reconstructed = vec![];
        let mut rest = self.clone();
        for (_, indices) in groups.iter().take(groups.len() - 1) {
            let mut candidate = rest.constant(rest.lcoeff());
            for index in indices {
                candidate = (&candidate * &factors[*index])
                    .map_coeff(|coefficient| coefficient.clone().symmetric_mod(modulus), Z);
            }
            let content = candidate.content();
            candidate = candidate.div_coeff(&content);

            let (quotient, remainder) = rest.quot_rem(&candidate, true);
            if !remainder.is_zero() {
                return None;
            }
            reconstructed.push(candidate);
            let content = quotient.content();
            rest = quotient.div_coeff(&content);
        }
        reconstructed.push(rest);

        debug!(
            "LLL recombination partitioned {} modular factors into {} exact factors",
            factor_count,
            reconstructed.len()
        );
        let mut result = vec![];
        for factor in reconstructed {
            if factor.degree(var) >= self.degree(var) {
                return None;
            }
            result.extend(factor.factor_reconstruct());
        }
        #[cfg(test)]
        LLL_RECOMBINATION_SUCCESSES.with(|successes| successes.set(successes.get() + 1));
        Some(result)
    }

    /// Factor a square-free univariate polynomial over the integers by Hensel lifting factors computed over
    /// a finite field image of the polynomial.
    fn factor_reconstruct(&self) -> Vec<Self> {
        let Some(var) = self.last_exponents().iter().position(|x| *x > E::zero()) else {
            return vec![self.clone()]; // constant polynomial
        };
        let d = self.degree(var).to_u32();

        if d == 1 {
            return vec![self.clone()];
        }

        // Select a suitable prime. The number of modular factors controls the
        // exponential recombination step, so try several small primes and
        // retain the factorization with the fewest factors.
        let prime_trials = if d >= 128 {
            10
        } else if d >= 32 {
            5
        } else {
            1
        };
        let mut best_factorization: Option<ModularIntegerFactorization<E>> = None;
        let mut suitable_primes = 0;
        let mut pi = PrimeIteratorU64::new(2);
        while suitable_primes < prime_trials {
            let p = pi.next().unwrap();
            if p > u32::MAX as u64 {
                panic!("Ran out of primes during factorization of {self}");
            }
            let max_factor_count = best_factorization
                .as_ref()
                .map(|best| best.distinct_degree.factor_count.saturating_sub(1));
            let Some(screen) = self.screen_univariate_mod_prime(var, p as u32, max_factor_count)
            else {
                continue;
            };
            suitable_primes += 1;
            let candidate = match screen {
                ModularPrimeScreen::Candidate(candidate) => candidate,
                ModularPrimeScreen::FactorLimitExceeded { .. } => continue,
            };

            if candidate.distinct_degree.factor_count == 1 {
                // Irreducibility modulo one prime proves irreducibility over Z.
                return vec![self.clone()];
            }

            let replace_best = best_factorization
                .as_ref()
                .map(|best| {
                    candidate.distinct_degree.factor_count < best.distinct_degree.factor_count
                })
                .unwrap_or(true);
            if replace_best {
                best_factorization = Some(candidate);
            }

            if best_factorization
                .as_ref()
                .is_some_and(|best| best.distinct_degree.factor_count <= 12)
            {
                break;
            }
        }

        let bound = self.coefficient_bound();

        // A wider machine prime reduces the number of full correction rounds in
        // the current linear p-adic lift. Restrict it to low-degree, high-height
        // images where that saving clearly outweighs the more expensive finite-
        // field factorization.
        let (initial_factor_count, initial_digits, initial_work) = {
            let candidate = best_factorization.as_ref().unwrap();
            let (digits, _) = Self::linear_hensel_modulus(&bound, candidate.field.get_prime());
            (
                candidate.distinct_degree.factor_count,
                digits,
                Self::linear_hensel_work(candidate.distinct_degree.factor_count, digits),
            )
        };
        let high_linear_lift_pressure = d <= 64
            && bound.significant_bits() >= 256
            && initial_factor_count >= 3
            && initial_digits >= 64
            && initial_work >= 256;

        if high_linear_lift_pressure {
            // Compare three suitable small primes before considering the large
            // candidate, since their modular factor counts can differ sharply.
            while suitable_primes < 3 {
                let p = pi.next().unwrap();
                if p > u32::MAX as u64 {
                    panic!("Ran out of primes during factorization of {self}");
                }
                let candidate_digits = Self::linear_hensel_modulus(&bound, p as u32).0;
                let best = best_factorization.as_ref().unwrap();
                let best_digits = Self::linear_hensel_modulus(&bound, best.field.get_prime()).0;
                let best_work =
                    Self::linear_hensel_work(best.distinct_degree.factor_count, best_digits);
                let max_factor_count =
                    (best_work.saturating_sub(1) / candidate_digits).saturating_add(1);
                let Some(screen) =
                    self.screen_univariate_mod_prime(var, p as u32, Some(max_factor_count))
                else {
                    continue;
                };
                suitable_primes += 1;
                let candidate = match screen {
                    ModularPrimeScreen::Candidate(candidate) => candidate,
                    ModularPrimeScreen::FactorLimitExceeded { .. } => continue,
                };

                if candidate.distinct_degree.factor_count == 1 {
                    return vec![self.clone()];
                }

                let candidate_work = Self::linear_hensel_work(
                    candidate.distinct_degree.factor_count,
                    candidate_digits,
                );
                if candidate_work < best_work {
                    best_factorization = Some(candidate);
                }
            }

            // Keep dense products of two degree-d images in the u64 accumulator kernel. The
            // bound p * (d + 1) <= u32::MAX proves both that all (d + 1)^2 products fit in u64
            // and that one Montgomery reduction is sufficient for each output coefficient.
            let maximum_direct_prime = u64::from(u32::MAX) / (u64::from(d) + 1);
            // This retains 26 bits per lifting digit and leaves room to skip unsuitable primes at
            // degree 64 without crossing the direct-reduction bound.
            let mut direct_primes = PrimeIteratorU64::new(65_000_000);
            let best_factor_count = best_factorization
                .as_ref()
                .unwrap()
                .distinct_degree
                .factor_count;
            let direct_factor_limit = if best_factor_count < 10 {
                best_factor_count + 1
            } else {
                best_factor_count
            };
            let direct_candidate = loop {
                let Some(p) = direct_primes.next() else {
                    break None;
                };
                if p > maximum_direct_prime {
                    break None;
                }
                match self.screen_univariate_mod_prime(var, p as u32, Some(direct_factor_limit)) {
                    None => {}
                    Some(ModularPrimeScreen::Candidate(candidate)) => break Some(candidate),
                    Some(ModularPrimeScreen::FactorLimitExceeded { lower_bound }) => {
                        debug!(
                            "Rejected the first suitable dense-u64 prime at modular factor lower bound {lower_bound}"
                        );
                        break None;
                    }
                }
            };

            if let Some(candidate) = direct_candidate {
                if candidate.distinct_degree.factor_count == 1 {
                    return vec![self.clone()];
                }

                let best = best_factorization.as_ref().unwrap();
                let best_digits = Self::linear_hensel_modulus(&bound, best.field.get_prime()).0;
                let best_work =
                    Self::linear_hensel_work(best.distinct_degree.factor_count, best_digits);
                let candidate_digits =
                    Self::linear_hensel_modulus(&bound, candidate.field.get_prime()).0;
                let candidate_work = Self::linear_hensel_work(
                    candidate.distinct_degree.factor_count,
                    candidate_digits,
                );
                let crosses_recombination_boundary = best.distinct_degree.factor_count <= 10
                    && candidate.distinct_degree.factor_count > 10;
                let same_or_fewer_factors = candidate.distinct_degree.factor_count
                    <= best.distinct_degree.factor_count
                    && candidate_work.saturating_mul(2) <= best_work;
                let one_extra_factor = candidate.distinct_degree.factor_count
                    == best.distinct_degree.factor_count + 1
                    && best.distinct_degree.factor_count < 10
                    && candidate_work.saturating_mul(4) <= best_work;

                if !crosses_recombination_boundary && (same_or_fewer_factors || one_extra_factor) {
                    debug!(
                        "Selected a dense-u64 modular prime: estimated linear Hensel work {best_work} -> {candidate_work}"
                    );
                    best_factorization = Some(candidate);
                }
            }
        }

        let best_factorization = best_factorization.unwrap();
        debug!(
            "Selected modular factorization with {} factors",
            best_factorization.distinct_degree.factor_count
        );
        let (field, hs) = Self::complete_equal_degree_factorization(best_factorization);

        let (_, max_p) = Self::linear_hensel_modulus(&bound, field.get_prime());

        let mut factors = self.multi_factor_hensel_lift(&hs, &max_p);

        #[cfg(debug_assertions)]
        for (h, h_p) in factors.iter().zip(&hs) {
            let hh_p = h
                .map_coeff(|c| c.to_finite_field(&field), field.clone())
                .make_monic();
            if &hh_p != h_p {
                panic!("Mismatch of lifted factor: {hh_p} vs {h_p} in {self}");
            }
        }

        if factors.len() > 10
            && let Some(recombined) = self.lll_factor_recombination(&factors, &max_p, var)
        {
            return recombined;
        }

        let mut rec_factors = vec![];
        // factor recombination
        let mut s = 1;

        let mut rest = self.clone();
        'len: while 2 * s <= factors.len() {
            let mut fs = CombinationIterator::new(factors.len(), s);
            while let Some(cs) = fs.next() {
                // check if the constant term matches
                if rest.exponents[..rest.nvars()]
                    .iter()
                    .all(|e| *e == E::zero())
                {
                    let mut g1 = rest.lcoeff();
                    let mut h1 = rest.lcoeff();
                    for (i, f) in factors.iter().enumerate() {
                        if f.exponents[..rest.nvars()].iter().all(|x| *x == E::zero()) {
                            if cs.contains(&i) {
                                g1 = (&g1 * &f.coefficients[0]).symmetric_mod(&max_p);
                            } else {
                                h1 = (&h1 * &f.coefficients[0]).symmetric_mod(&max_p);
                            }
                        }
                    }

                    // TODO: improve check
                    // for monic factors we can do &g1 * &h1 != &rest.lcoeff() * &rest.coefficients[0]
                    if (&g1 * &h1).abs() > bound {
                        continue;
                    }
                }

                let mut g = rest.constant(rest.lcoeff());
                for (i, f) in factors.iter().enumerate() {
                    if cs.contains(&i) {
                        g = (&g * f).map_coeff(|i| i.clone().symmetric_mod(&max_p), Z);
                    }
                }
                let c = g.content();
                g = g.div_coeff(&c);

                let (h, r) = rest.quot_rem(&g, true);

                if r.is_zero() {
                    // should always happen happen when |g1|_1 * |h1|_1 <= bound
                    rec_factors.push(g);

                    for i in cs.iter().rev() {
                        factors.remove(*i);
                    }

                    let c = h.content();
                    rest = h.div_coeff(&c);

                    continue 'len;
                }
            }

            s += 1;
        }

        rec_factors.push(rest);
        rec_factors
    }

    /// Lift a solution of `poly ≡ lcoeff * univariate_factors mod y mod p^k`
    /// to `mod y^iterations mod p^k`.
    ///
    /// Univariate factors must be monic and `lcoeff_y=0` should be as well.
    fn bivariate_hensel_lift_bernardin(
        poly: &MultivariatePolynomial<FiniteField<Integer>, E, LexOrder>,
        interpolation_var: usize,
        lcoeff: &MultivariatePolynomial<FiniteField<Integer>, E, LexOrder>,
        univariate_factors: &[MultivariatePolynomial<FiniteField<Integer>, E, LexOrder>],
        iterations: usize,
        p: u32,
        k: usize,
    ) -> Vec<MultivariatePolynomial<FiniteField<Integer>, E, LexOrder>> {
        let finite_field = Zp::new(p);

        // add the leading coefficient as a first factor
        let mut factors = vec![lcoeff.replace(interpolation_var, &poly.ring().zero())];

        for f in univariate_factors {
            factors.push(f.clone());
        }

        let delta = Self::lift_diophantine_univariate(
            &mut factors,
            &poly.constant(poly.ring().one()),
            finite_field.get_prime(),
            k,
        );

        let y_poly = poly.to_univariate_polynomial_list(interpolation_var);

        // extract coefficients in y
        let mut u: Vec<_> = factors
            .iter()
            .map(|f| {
                let mut dense = vec![poly.zero(); iterations + 1];
                dense[0] = f.clone();
                dense
            })
            .collect();

        // update the first polynomial as it may contain y, since it's lcoeff
        let y_lcoeff = lcoeff.to_univariate_polynomial_list(interpolation_var);
        for (p, e) in y_lcoeff {
            u[0][e.to_u32() as usize] = p;
        }

        let mut p = u.clone();
        let mut cur_p = p[0][0].clone();
        for x in &mut p.iter_mut().skip(1) {
            cur_p = cur_p * &x[0];
            x[0] = cur_p.clone();
        }

        for k in 1..iterations {
            // extract the coefficient required to compute the error in y^k
            // computed using a convolution
            p[0][k] = u[0][k].clone();
            for i in 1..factors.len() {
                for j in 0..k {
                    p[i][k] = &p[i][k] + &(&p[i - 1][k - j] * &u[i][j]);
                }
            }

            // find the kth power of y in f
            // since we compute the error per power of y, we cannot stop on a 0 error
            let e = if let Some((v, _)) = y_poly.iter().find(|e| e.1.to_u32() as usize == k) {
                v - &p.last().unwrap()[k]
            } else {
                -p.last().unwrap()[k].clone()
            };

            if e.is_zero() {
                continue;
            }

            for ((dp, f), d) in u.iter_mut().zip(factors.iter()).zip(&delta) {
                dp[k] = &dp[k] + &(d * &e).quot_rem_univariate_monic(f).1;
            }

            // update the coefficients with the new y^k contributions
            // note that the lcoeff[k] contribution is not new
            let mut t = poly.zero();
            for i in 1..factors.len() {
                t = &u[i][0] * &t + &u[i][k] * &p[i - 1][0];
                p[i][k] = &p[i][k] + &t;
            }
        }

        // convert dense polynomials to multivariate polynomials
        u.into_iter()
            .map(|ts| {
                let mut new_poly = poly.zero_with_capacity(ts.len());
                for (i, mut f) in ts.into_iter().enumerate() {
                    for x in f.exponents_iter_mut() {
                        x[interpolation_var] = E::from_u32(i as u32);
                    }
                    new_poly = new_poly + f;
                }

                new_poly
            })
            .collect()
    }

    /// Factor a square-free bivariate polynomial over the integers.
    fn bivariate_factor_reconstruct(&self, main_var: usize, interpolation_var: usize) -> Vec<Self> {
        if self.bivariate_irreducibility_test() {
            return vec![self.clone()];
        }

        let d2 = self.degree(interpolation_var).to_u32();

        // select a suitable evaluation point, as small as possible as to not change the coefficient bound
        let mut sample_point;
        let mut uni_f;
        let mut i = 0u64;
        loop {
            sample_point = i.into();
            uni_f = self.replace(interpolation_var, &sample_point);

            if self.degree(main_var) == uni_f.degree(main_var)
                && uni_f.gcd(&uni_f.derivative(main_var)).is_constant()
            {
                break;
            }

            i += 1;
        }

        // factor the univariate polynomial
        let mut uni_fs: Vec<_> = uni_f
            .factor()
            .into_iter()
            .map(|(f, p)| {
                debug_assert_eq!(p, 1);
                f
            })
            .collect();

        // strip potential content
        uni_fs.retain_mut(|f| !f.is_constant());

        // select a suitable prime
        // we try small primes first as the distinct and equal degree algorithms
        // scale as log(p)
        let mut pi = PrimeIteratorU64::new(101);
        let mut field;
        'new_prime: loop {
            let p = pi.next().unwrap();
            if p > u32::MAX as u64 {
                panic!("Ran out of primes during factorization of {self}");
            }
            let p = p as u32;

            if (&uni_f.lcoeff() % &Integer::Single(p as i64)).is_zero() {
                continue;
            }

            field = Zp::new(p);

            // make sure the factors stay coprime
            let fs_p: Vec<_> = uni_fs
                .iter()
                .map(|f| f.map_coeff(|c| c.to_finite_field(&field), field.clone()))
                .collect();

            for (j, f) in fs_p.iter().enumerate() {
                for g in &fs_p[j + 1..] {
                    if !f.gcd(g).is_one() {
                        continue 'new_prime;
                    }
                }
            }

            break;
        }

        let shifted_poly = if !sample_point.is_zero() {
            self.shift_var(interpolation_var, &sample_point)
        } else {
            self.clone()
        };

        // TODO: if bound is less than u64, we may also use Zp64 for the computation
        let bound = shifted_poly.coefficient_bound();

        let p = field.get_prime().to_integer();
        let mut max_p = p.clone();
        let mut k = 1;
        while &max_p * 2 < bound {
            max_p = &max_p * &p;
            k += 1;
        }

        let mod_field = FiniteField::<Integer>::new(max_p.clone());

        // make all factors monic, this is possible since the lcoeff is invertible mod p^k
        let uni_fs_mod: Vec<_> = uni_fs
            .iter()
            .map(|f| {
                let f1 = f.map_coeff(|c| mod_field.to_element(c.clone()), mod_field.clone());
                f1.make_monic()
            })
            .collect();

        let mut f_mod =
            shifted_poly.map_coeff(|c| mod_field.to_element(c.clone()), mod_field.clone());

        // make sure the lcoeff is monic at y=0
        let inv_coeff = mod_field.inv(&mod_field.to_element(uni_f.lcoeff().clone()));
        let f_mod_monic = f_mod.clone().mul_coeff(inv_coeff);
        let lcoeff_monic = f_mod_monic.lcoeff_last_varorder(&[main_var, interpolation_var]);

        let mut factors = Self::bivariate_hensel_lift_bernardin(
            &f_mod_monic,
            interpolation_var,
            &lcoeff_monic,
            &uni_fs_mod,
            (d2 + 1) as usize,
            field.get_prime(),
            k,
        );

        factors.swap_remove(0); // remove the lcoeff

        let mut rec_factors = vec![];
        // factor recombination
        let mut s = 1;

        let mut lcoeff = f_mod.lcoeff_last_varorder(&[main_var, interpolation_var]);
        let mut rest = shifted_poly;
        'len: while 2 * s <= factors.len() {
            let mut fs = CombinationIterator::new(factors.len(), s);
            while let Some(cs) = fs.next() {
                let mut g = lcoeff.clone();
                for (i, f) in factors.iter().enumerate() {
                    if cs.contains(&i) {
                        g = (&g * f).mod_var(interpolation_var, E::from_u32(d2 + 1));
                    }
                }

                // convert to integer
                let mut g_int = g.map_coeff(|c| mod_field.to_symmetric_integer(c), Z);

                let content = g_int.univariate_content(main_var);
                g_int = &g_int / &content;

                let (h, r) = rest.quot_rem(&g_int, true);

                if r.is_zero() {
                    rec_factors.push(g_int);

                    for i in cs.iter().rev() {
                        factors.remove(*i);
                    }

                    rest = h;
                    f_mod = rest.map_coeff(|c| mod_field.to_element(c.clone()), mod_field.clone());
                    lcoeff = f_mod.lcoeff_last_varorder(&[main_var, interpolation_var]);

                    continue 'len;
                }
            }

            s += 1;
        }

        rec_factors.push(rest);

        if !sample_point.is_zero() {
            for x in &mut rec_factors {
                // shift the polynomial to y - sample
                *x = x.shift_var(interpolation_var, &self.ring().neg(&sample_point));
            }
        }

        rec_factors
    }

    /// Solve a Diophantine equation over the ring `Z_p^k` using Newton iteration.
    /// All factors must be monic.
    fn lift_diophantine_univariate(
        factors: &mut [MultivariatePolynomial<FiniteField<Integer>, E, LexOrder>],
        rhs: &MultivariatePolynomial<FiniteField<Integer>, E, LexOrder>,
        p: u32,
        k: usize,
    ) -> Vec<MultivariatePolynomial<FiniteField<Integer>, E, LexOrder>> {
        let field = Zp::new(p);
        let prime: Integer = (p as u64).into();

        let mut f_p: Vec<_> = factors
            .iter()
            .map(|f| {
                f.map_coeff(
                    |c| rhs.ring().to_symmetric_integer(c).to_finite_field(&field),
                    field.clone(),
                )
            })
            .collect();
        let rhs_p = rhs.map_coeff(
            |c| rhs.ring().to_symmetric_integer(c).to_finite_field(&field),
            field.clone(),
        );

        // TODO: recycle from finite field computation that must have happened earlier
        let mut delta =
            MultivariatePolynomial::<Zp, E, LexOrder>::diophantine_univariate(&mut f_p, &rhs_p);

        let mut deltas: Vec<_> = delta
            .iter()
            .map(|s| {
                s.map_coeff(
                    |c| field.to_symmetric_integer(c).to_finite_field(&rhs.ring()),
                    rhs.ring().clone(),
                )
            })
            .collect();

        if k == 1 {
            return deltas;
        }

        let mut tot = rhs.constant(rhs.ring().one());
        for f in factors.iter() {
            tot = &tot * f;
        }

        let pi = factors
            .iter_mut()
            .map(|f| tot.quot_rem_univariate(f).0)
            .collect::<Vec<_>>();

        let mut m = prime.clone();

        for _ in 1..k {
            // TODO: is breaking on e=0 safe?
            let mut e = rhs.clone();
            for (dd, pp) in deltas.iter_mut().zip(&pi) {
                e = &e - &(&*dd * pp);
            }

            let e_m = e.map_coeff(
                |c| (&rhs.ring().to_symmetric_integer(c) / &m).to_finite_field(&field),
                field.clone(),
            );

            for ((p, d_m), d) in f_p.iter_mut().zip(&mut delta).zip(deltas.iter_mut()) {
                let new_delta = (&e_m * &*d_m).quot_rem_univariate(p).1;

                *d = &*d
                    + &new_delta.map_coeff(
                        |c| (&field.to_symmetric_integer(c) * &m).to_finite_field(&rhs.ring()),
                        rhs.ring().clone(),
                    );
            }

            m = &m * &prime;
        }

        deltas
    }

    /// Compute the Gelfond bound for the coefficients magnitude of every factor of this polynomial
    fn coefficient_bound(&self) -> Integer {
        let max_norm = self.coefficients.iter().map(|x| x.abs()).max().unwrap();

        let mut bound = Integer::one();
        let mut total_degree = 0;
        let mut non_zero_vars = 0;
        for v in 0..self.nvars() {
            let d = self.degree(v).to_u32() as u64;
            if d > 0 {
                non_zero_vars += 1;
                total_degree += d;
                bound *= &Integer::from(d + 1);
            }
        }

        // move the 2^n into the sqrt to prevent precision loss when converting the sqrt
        // to an integer
        bound = &bound * &Integer::Single(2).pow((total_degree * 2).saturating_sub(non_zero_vars));

        let root_bound = match &bound {
            Integer::Single(b) => Integer::Single((*b as f64).sqrt() as i64),
            Integer::Double(_) | Integer::Large(_) => bound.root(2),
        };
        bound = &root_bound + &1i64.into();

        &bound * &(&max_norm * &self.lcoeff().abs())
    }

    /// Sort the bivariate factors based on their univariate image so that they are
    /// aligned between the different vars.
    // TODO: merge with the implementation for finite fields as the implementation
    // is almost identical
    fn canonical_sort(
        biv_polys: &[Self],
        replace_var: usize,
        sample_points: &[(usize, Integer)],
    ) -> Vec<(Self, Integer, Self)> {
        let mut univariate_factors = biv_polys
            .iter()
            .map(|f| {
                let mut u = f.clone();
                for (v, p) in sample_points {
                    if *v == replace_var {
                        u = u.replace(*v, p);
                    }
                }

                // make sure the representative is unique
                let mut uni = u.clone().make_primitive();
                if uni.lcoeff().is_negative() {
                    uni = -uni;
                }

                (f.clone(), u.lcoeff(), uni)
            })
            .collect::<Vec<_>>();

        univariate_factors.sort_by(|(_, _, a), (_, _, b)| {
            a.exponents
                .cmp(&b.exponents)
                .then(a.coefficients.partial_cmp(&b.coefficients).unwrap())
        });

        univariate_factors
    }

    /// Return whether `factor` is a unit multiple of one polynomial variable.
    fn is_primitive_variable_factor(factor: &Self) -> bool {
        factor.nterms() == 1
            && factor.lcoeff().abs().is_one()
            && factor
                .exponents
                .iter()
                .filter(|exponent| **exponent != E::zero())
                .count()
                == 1
            && factor
                .exponents
                .iter()
                .any(|exponent| *exponent == E::one())
    }

    fn reconstruct_lcoeffs_from_univariate_sample(
        &self,
        lcoeff: &Self,
        lcoeff_factorization: &[(Self, usize)],
        univariate_factors: &[Self],
        sample_points: &[(usize, Integer)],
        univariate_content: &Integer,
    ) -> Option<Vec<Self>> {
        let lcoeff_content = lcoeff.content().abs();
        let mut lcoeff_factors = Vec::new();
        for (f, pow) in lcoeff_factorization {
            if f.is_constant() {
                continue;
            }

            let mut eval = f.clone();
            for (v, p) in sample_points {
                eval = eval.replace(*v, p);
            }

            if !eval.is_constant() {
                return None;
            }

            let eval = eval.get_constant().abs();
            if eval <= 1 {
                return None;
            }
            if Self::is_primitive_variable_factor(f) && !eval.gcd(&lcoeff_content).is_one() {
                return None;
            }

            lcoeff_factors.push((f, pow, eval));
        }

        for (i, (_, _, a)) in lcoeff_factors.iter().enumerate() {
            for (_, _, b) in &lcoeff_factors[i + 1..] {
                if !a.gcd(b).is_one() {
                    return None;
                }
            }
        }

        let mut true_lcoeffs = vec![self.one(); univariate_factors.len()];
        let mut used = vec![0usize; lcoeff_factors.len()];
        let mut residual_lcs: Vec<_> = univariate_factors.iter().map(|f| f.lcoeff()).collect();

        for (i, residual) in residual_lcs.iter_mut().enumerate() {
            for (j, (factor, multiplicity, eval)) in lcoeff_factors.iter().enumerate() {
                while used[j] < **multiplicity && (&residual.abs() % eval).is_zero() {
                    true_lcoeffs[i] = &true_lcoeffs[i] * factor;
                    *residual = &*residual / eval;
                    used[j] += 1;
                }
            }
        }

        if used
            .iter()
            .zip(&lcoeff_factors)
            .any(|(used, (_, multiplicity, _))| *used != **multiplicity)
        {
            return None;
        }

        for (lcoeff_i, factor) in true_lcoeffs.iter_mut().zip(univariate_factors) {
            let mut eval = lcoeff_i.clone();
            for (v, p) in sample_points {
                eval = eval.replace(*v, p);
            }

            if !eval.is_constant() {
                return None;
            }

            let eval = eval.get_constant();
            if eval.is_zero() {
                return None;
            }

            let univariate_lcoeff = factor.lcoeff();
            if (&univariate_lcoeff % &eval) != 0 {
                return None;
            }

            *lcoeff_i = lcoeff_i.clone().mul_coeff(&univariate_lcoeff / &eval);
        }

        let mut product = self.one();
        for l in &true_lcoeffs {
            product = &product * l;
        }

        if product.clone().mul_coeff(univariate_content.clone()) == *lcoeff {
            Some(true_lcoeffs)
        } else {
            None
        }
    }

    #[allow(dead_code)]
    fn wang_lcoeff_precomputation(
        &self,
        bivariate_factors: &[Self],
        sample_points: &[(usize, Integer)],
        order: &[usize],
    ) -> Result<(Integer, Vec<Self>, Vec<Self>), usize> {
        let lcoeff = self.univariate_lcoeff(order[0]);
        let lcoeff_factorization = lcoeff.factor();
        let sorted_biv_factors = Self::canonical_sort(bivariate_factors, order[1], sample_points)
            .into_iter()
            .map(|(f, _, _)| f)
            .collect::<Vec<_>>();

        let univariate_factor_images = sorted_biv_factors
            .iter()
            .map(|factor| {
                let mut image = factor.clone();
                for (v, p) in sample_points {
                    image = image.replace(*v, p);
                }
                image
            })
            .collect::<Vec<_>>();

        let Some(mut true_lcoeffs) = self.reconstruct_lcoeffs_from_univariate_sample(
            &lcoeff,
            &lcoeff_factorization,
            &univariate_factor_images,
            sample_points,
            &Integer::one(),
        ) else {
            return Err(sorted_biv_factors.len());
        };

        let mut sorted_biv_factors = sorted_biv_factors;
        let mut lcoeff_left = self.one();
        for (f, b) in true_lcoeffs.iter_mut().zip(&mut sorted_biv_factors) {
            let mut b_eval = b.clone();
            for (v, p) in sample_points {
                b_eval = b_eval.replace(*v, p);
            }
            let b_lc = b_eval.lcoeff();

            let mut f_eval = f.clone();
            for (v, p) in sample_points {
                f_eval = f_eval.replace(*v, p);
            }
            let f_lc = f_eval.lcoeff();

            let lcm = b_lc.lcm(&f_lc);
            let b_cor = &lcm / &b_lc;
            let f_cor = lcm / &f_lc;

            *b = b.clone().mul_coeff(b_cor);
            lcoeff_left = lcoeff_left.div_coeff(&f_cor);
            *f = f.clone().mul_coeff(f_cor);
        }

        Ok((lcoeff_left.get_constant(), sorted_biv_factors, true_lcoeffs))
    }

    /// Precompute the leading coefficients of the polynomial factors, using an
    /// adapted version of Kaltofen's algorithm that has modifications of Martin Lee and Stanislav Poslavsky.
    ///
    // TODO: merge with the implementation for finite fields as the implementation
    // is almost identical
    #[allow(dead_code)]
    fn lcoeff_precomputation(
        &self,
        bivariate_factors: &[Self],
        sample_points: &[(usize, Integer)],
        order: &[usize],
        bound: Integer,
        p: u32,
        k: usize,
    ) -> Result<(Integer, Vec<Self>, Vec<Self>), usize> {
        let lcoeff = self.univariate_lcoeff(order[0]);
        let sqf = lcoeff.square_free_factorization();

        let mut lcoeff_square_free = self.one();
        for (f, _) in &sqf {
            lcoeff_square_free = &lcoeff_square_free * f;
        }

        let sorted_main_factors = Self::canonical_sort(bivariate_factors, order[1], sample_points);

        let mut true_lcoeffs: Vec<_> = bivariate_factors.iter().map(|_| self.one()).collect();

        let mut lcoeff_left = lcoeff.clone();

        let mut main_bivariate_factors: Vec<_> =
            sorted_main_factors.into_iter().map(|(f, _, _)| f).collect();

        // TODO: smarter ordering
        for (i, &var) in order[1..].iter().enumerate() {
            if lcoeff_left.is_one() {
                break;
            }

            if lcoeff_left.degree(var).is_zero() {
                continue;
            }

            // only construct factors that depend on var and remove integer content and unit
            let c = lcoeff_square_free.univariate_content(var);
            let mut lcoeff_square_free_pp = &lcoeff_square_free / &c;

            // check if the evaluated leading coefficient remains square free
            let mut poly_eval = lcoeff_square_free_pp.clone();
            for (v, p) in sample_points {
                if *v != var {
                    poly_eval = poly_eval.replace(*v, p);
                }
            }

            if poly_eval.lcoeff().is_negative() {
                lcoeff_square_free_pp = -lcoeff_square_free_pp;
                poly_eval = -poly_eval;
            }
            debug!("Content-free lcsqf {}", lcoeff_square_free_pp);

            let sqf = poly_eval.square_free_factorization();
            if sqf.len() != 1 || sqf[0].1 != 1 {
                debug!("Polynomial is not square free: {}", poly_eval);
                return Err(main_bivariate_factors.len());
            }

            let bivariate_factors = if var == order[1] {
                main_bivariate_factors.to_vec()
            } else {
                let mut poly_eval = self.clone();
                for (v, p) in sample_points {
                    if *v != var {
                        poly_eval = poly_eval.replace(*v, p);
                    }
                }

                if poly_eval.degree(order[0]) != self.degree(order[0])
                    || poly_eval.degree(var) != self.degree(var)
                    || poly_eval.univariate_lcoeff(order[0]).degree(var) != lcoeff.degree(var)
                {
                    debug!("Bad sample for reconstructing lcoeff: degrees do not match");
                    return Err(main_bivariate_factors.len());
                }

                let bivariate_factors: Vec<_> = poly_eval
                    .factor()
                    .into_iter()
                    .map(|(f, _)| f)
                    // remove spurious content caused by particular evaluation point
                    .filter(|f| !f.is_constant())
                    .collect();

                if bivariate_factors.len() != main_bivariate_factors.len() {
                    return Err(bivariate_factors.len().min(main_bivariate_factors.len()));
                }

                Self::canonical_sort(&bivariate_factors, var, sample_points)
                    .into_iter()
                    .map(|(f, _, _)| f)
                    .collect()
            };

            let square_free_lc_biv_factors: Vec<_> = bivariate_factors
                .iter()
                .map(|f| {
                    let mut sff = f.univariate_lcoeff(order[0]).square_free_factorization();
                    // make sure every bivariate factor has positive lcoeff such that the product
                    // of the basis elements equals the evaluated lcoeff_square_free_pp
                    for (b, _) in &mut sff {
                        if b.lcoeff().is_negative() {
                            *b = -b.clone();
                        }
                    }
                    sff
                })
                .collect();

            let basis = Self::gcd_free_basis(
                square_free_lc_biv_factors
                    .iter()
                    .flatten()
                    .map(|x| x.0.clone())
                    .filter(|x| !x.is_constant())
                    .collect(),
            );

            if basis.is_empty() {
                continue;
            }

            let lifted = if basis.len() == 1 {
                vec![lcoeff_square_free_pp.clone()]
            } else {
                let mut new_order = order.to_vec();
                new_order.swap(1, i + 1);
                new_order.remove(0);

                lcoeff_square_free_pp.multivariate_hensel_lift_with_auto_lcoeff_fixing(
                    &basis,
                    sample_points,
                    &new_order,
                    bound.clone(),
                    p,
                    k,
                )
            };

            for (l, fac) in true_lcoeffs.iter_mut().zip(&square_free_lc_biv_factors) {
                let mut contrib = self.one();
                for (full, b) in lifted.iter().zip(&basis) {
                    // check if a GCD-free basis element is a factor of the leading coefficient of this bivariate factor
                    if let Some((_, m)) = fac.iter().find(|(f, _)| f == b || f.try_div(b).is_some())
                    {
                        for _ in 0..*m {
                            contrib = &contrib * full;
                        }
                    }
                }

                let g = contrib.gcd(l);
                let new = (contrib / &g).make_primitive();

                *l = (&*l * &new).make_primitive();

                let (q, r) = lcoeff_left.quot_rem(&new, true);
                if !r.is_zero() {
                    panic!(
                        "Problem with bivariate factor scaling in factorization of {self}: order={order:?}, samples={sample_points:?}"
                    );
                }

                lcoeff_left = q;
            }
        }

        if !lcoeff_left.is_constant() {
            panic!(
                "Could not reconstruct leading coefficient of {self}: order={order:?}, samples={sample_points:?} Rest = {lcoeff_left}"
            );
        }

        // rescale the leading coefficient factors to recover the missing content and sign
        for (f, b) in true_lcoeffs.iter_mut().zip(&mut main_bivariate_factors) {
            let mut b_eval = b.clone();
            for (v, p) in sample_points {
                b_eval = b_eval.replace(*v, p);
            }

            let b_lc = b_eval.lcoeff();

            let mut f_eval = f.clone();
            for (v, p) in sample_points {
                f_eval = f_eval.replace(*v, p);
            }
            let f_lc = f_eval.lcoeff();

            let lcm = b_lc.lcm(&f_lc);

            let b_cor = &lcm / &b_lc;
            let f_cor = lcm / &f_lc;

            *b = b.clone().mul_coeff(b_cor);

            lcoeff_left = lcoeff_left.div_coeff(&f_cor);
            *f = f.clone().mul_coeff(f_cor);
        }

        Ok((
            lcoeff_left.get_constant(),
            main_bivariate_factors,
            true_lcoeffs,
        ))
    }

    #[allow(dead_code)]
    fn multivariate_hensel_lift_with_auto_lcoeff_fixing(
        &self,
        factors: &[Self],
        sample_points: &[(usize, Integer)],
        order: &[usize],
        bound: Integer,
        p: u32,
        k: usize,
    ) -> Vec<Self> {
        let modulus = FiniteField::<Integer>::new(bound);
        let ff = self.map_coeff(|c| modulus.to_element(c.clone()), modulus.clone());
        let factors_ff: Vec<_> = factors
            .iter()
            .map(|f| f.map_coeff(|c| modulus.to_element(c.clone()), modulus.clone()))
            .collect();
        let sample_points_ff: Vec<_> = sample_points
            .iter()
            .map(|(v, p)| (*v, modulus.to_element(p.clone())))
            .collect();
        let lcoeff = ff.univariate_lcoeff(order[0]);

        if lcoeff.is_constant() {
            // the factors should be properly normalized
            let (mut uni, delta) = MultivariatePolynomial::get_univariate_factors_and_deltas(
                &factors_ff,
                order,
                sample_points,
                p,
                k,
            );
            let h = ff
                .multivariate_hensel_lifting(
                    &factors_ff,
                    &mut uni,
                    &delta,
                    &sample_points_ff,
                    None,
                    order,
                    MultivariateHenselContext::new(1),
                )
                .unwrap();

            return h
                .into_iter()
                .map(|f| f.map_coeff(|c| modulus.to_symmetric_integer(c), Z))
                .collect();
        }

        // repeat the leading coefficient for every factor so that the leading coefficient is known
        let padded_lcoeffs = vec![lcoeff.clone(); factors.len()];

        let mut self_adjusted = ff;
        for _ in 1..factors_ff.len() {
            self_adjusted = &self_adjusted * &lcoeff;
        }

        // set the proper lc
        let mut lc_var_eval = lcoeff.clone();
        for (v, p) in sample_points {
            if *v != order[0] {
                lc_var_eval = lc_var_eval.replace(*v, &lc_var_eval.ring().to_element(p.clone()));
            }
        }

        let adjusted_factors: Vec<_> = factors_ff
            .into_iter()
            .map(|f| f.make_monic() * &lc_var_eval)
            .collect();

        let (mut uni, delta) = MultivariatePolynomial::get_univariate_factors_and_deltas(
            &adjusted_factors,
            order,
            sample_points,
            p,
            k,
        );
        let h = self_adjusted
            .multivariate_hensel_lifting(
                &adjusted_factors,
                &mut uni,
                &delta,
                &sample_points_ff,
                Some(&padded_lcoeffs),
                order,
                MultivariateHenselContext::new(1),
            )
            .unwrap();

        h.into_iter()
            .map(|f| {
                let f_i = f.map_coeff(|c| modulus.to_symmetric_integer(c), Z);
                let c = f_i.univariate_content(order[0]);
                f_i / &c
            })
            .collect()
    }

    fn lcoeff_sample_preserves_square_free_images(
        &self,
        order: &[usize],
        lcoeff: &Self,
        lcoeff_square_free: &Self,
        sample_points: &[(usize, Integer)],
    ) -> bool {
        for &var in &order[1..] {
            if lcoeff.degree(var).is_zero() {
                continue;
            }

            let content = lcoeff_square_free.univariate_content(var);
            let lcoeff_square_free_pp = lcoeff_square_free / &content;
            let target_degree = lcoeff_square_free_pp.degree(var);

            let mut poly_eval = lcoeff_square_free_pp;
            for (v, p) in sample_points {
                if *v != var {
                    poly_eval = poly_eval.replace(*v, p);
                }
            }

            if poly_eval.lcoeff().is_negative() {
                poly_eval = -poly_eval;
            }

            if poly_eval.degree(var) != target_degree {
                return false;
            }

            let derivative = poly_eval.derivative(var);
            if !poly_eval.gcd(&derivative).is_constant() {
                return false; // not square-free
            }
        }

        true
    }

    /// Check that evaluated leading-coefficient factors can be assigned uniquely
    /// to the factors of the univariate image.
    fn lcoeff_sample_supports_wang_reconstruction(
        lcoeff_factorization: &[(Self, usize)],
        lcoeff_content: &Integer,
        sample_points: &[(usize, Integer)],
    ) -> bool {
        let mut images = Vec::new();

        for (factor, _) in lcoeff_factorization {
            if factor.is_constant() {
                continue;
            }

            let mut image = factor.clone();
            for (v, p) in sample_points {
                image = image.replace(*v, p);
            }

            if !image.is_constant() {
                return false;
            }

            let image = image.get_constant().abs();
            if image <= 1 {
                return false;
            }
            if Self::is_primitive_variable_factor(factor) && !image.gcd(lcoeff_content).is_one() {
                return false;
            }

            if images
                .iter()
                .any(|prev: &Integer| !prev.gcd(&image).is_one())
            {
                return false;
            }

            images.push(image);
        }

        true
    }

    #[allow(dead_code)]
    fn find_sample(
        &self,
        order: &mut [usize],
        mut coefficient_upper_bound: i64,
        mut max_factors_num: Option<usize>,
    ) -> (Vec<Self>, Vec<(usize, Integer)>, i64, Self) {
        debug!("Find sample for {} with order {:?}", self, order);

        // select a suitable evaluation point, as small as possible as to not change the coefficient bound
        let mut cur_sample_points: Vec<_> =
            order[1..].iter().map(|i| (*i, Integer::zero())).collect();
        let mut cur_uni_f;
        let mut cur_biv_f;
        let mut rng = rng();
        let degree = self.degree(order[0]);
        let mut bivariate_factors: Vec<_>;
        let mut best: Option<(Integer, Vec<Self>, Vec<(usize, Integer)>, i64, Self)> = None;

        let uni_lcoeff = self.univariate_lcoeff(order[0]);
        let mut lcoeff_square_free = self.one();
        for (f, _) in uni_lcoeff.square_free_factorization() {
            lcoeff_square_free = &lcoeff_square_free * &f;
        }

        let mut content_try_count = 0;
        let mut lcoeff_try_count = 0;
        'new_sample: loop {
            for s in &mut cur_sample_points {
                s.1 = Integer::Single(rng.random_range(0..=coefficient_upper_bound));
                debug!("Sample x{} {}", s.0, s.1);
            }

            cur_biv_f = self.clone();
            for ((v, s), rem_var) in cur_sample_points[1..].iter().zip(&order[1..]).rev() {
                cur_biv_f = cur_biv_f.replace(*v, s);
                if cur_biv_f.degree(*rem_var) != self.degree(*rem_var) {
                    coefficient_upper_bound += 10;
                    continue 'new_sample;
                }
            }

            // requirement for leading coefficient precomputation
            if cur_biv_f.univariate_lcoeff(order[0]).degree(order[1]) != uni_lcoeff.degree(order[1])
            {
                debug!(
                    "Degree of x{} in leading coefficient of bivariate image is wrong",
                    order[1]
                );
                coefficient_upper_bound += 10;
                continue 'new_sample;
            }

            if !self.lcoeff_sample_preserves_square_free_images(
                order,
                &uni_lcoeff,
                &lcoeff_square_free,
                &cur_sample_points,
            ) {
                debug!(
                    "Bad sample for reconstructing lcoeff: square-free lcoeff image is not square-free"
                );
                lcoeff_try_count += 1;
                if lcoeff_try_count == 10 {
                    coefficient_upper_bound += 10;
                    lcoeff_try_count = 0;
                }
                continue 'new_sample;
            }
            lcoeff_try_count = 0;

            let biv_df = cur_biv_f.derivative(order[0]);

            cur_uni_f = cur_biv_f.replace(cur_sample_points[0].0, &cur_sample_points[0].1);
            let uni_df = cur_uni_f.derivative(order[0]);

            if degree == cur_biv_f.degree(order[0])
                && degree == cur_uni_f.degree(order[0])
                && cur_biv_f.gcd(&biv_df).is_constant()
                && cur_uni_f.gcd(&uni_df).is_constant()
            {
                let c = cur_biv_f.univariate_content(order[0]);

                if !c.is_constant() {
                    content_try_count += 1;
                    coefficient_upper_bound += 10;

                    debug!("Univariate content is not constant");
                    if content_try_count == 10 {
                        // it is likely that we will always find content for this variable ordering, so change the
                        // second variable
                        // TODO: is this guaranteed to work or should we also change the first variable?
                        let sec_var = order[1];
                        order.copy_within(2..order.len(), 1);
                        order[order.len() - 1] = sec_var;

                        for ((vs, _), v) in cur_sample_points.iter_mut().zip(&order[1..]) {
                            *vs = *v;
                        }

                        debug!("Changed the second variable to {}", order[1]);
                        content_try_count = 0;
                    }

                    continue;
                }

                bivariate_factors = cur_biv_f.factor().into_iter().map(|f| f.0).collect();
                bivariate_factors.retain(|f| !f.is_constant());

                if max_factors_num.is_none() {
                    max_factors_num = Some(bivariate_factors.len());
                }

                // A valid specialization can split true multivariate factors, so the
                // bivariate factor count is an upper bound on the final factor count.
                // Keep the smallest admissible count; a larger count is only useful
                // when the caller supplied a looser bound from an earlier failure.
                if bivariate_factors.len() <= max_factors_num.unwrap() {
                    if bivariate_factors.len() < max_factors_num.unwrap() {
                        max_factors_num = Some(bivariate_factors.len());
                        content_try_count = 0;
                        best = None;
                    }

                    if best.is_none() || c.get_constant().abs() < best.as_ref().unwrap().0 {
                        best = Some((
                            c.get_constant().abs(),
                            bivariate_factors.clone(),
                            cur_sample_points.clone(),
                            coefficient_upper_bound,
                            cur_uni_f.clone(),
                        ));
                    }

                    content_try_count += 1;

                    // try a few times to lower the chance of a costly Hensel lift
                    // with a wrong number of factors
                    if content_try_count > 2 {
                        break;
                    }
                } else {
                    debug!(
                        "Number of factors is too large: {} vs {}",
                        bivariate_factors.len(),
                        max_factors_num.unwrap_or(bivariate_factors.len())
                    );
                }
            }

            coefficient_upper_bound += 10;
            debug!("Growing bound {}", coefficient_upper_bound);
        }

        let (_, bivariate_factors, cur_sample_points, coefficient_upper_bound, cur_uni_f) =
            best.unwrap();

        (
            bivariate_factors,
            cur_sample_points,
            coefficient_upper_bound,
            cur_uni_f,
        )
    }

    fn find_univariate_sample(
        &self,
        order: &mut [usize],
        coefficient_upper_bound: i64,
        max_factors_num: Option<usize>,
    ) -> Option<(
        Vec<Self>,
        Vec<Self>,
        Vec<(usize, Integer)>,
        i64,
        Self,
        Integer,
    )> {
        debug!("Find univariate sample for {} with order {:?}", self, order);
        let mut cur_sample_points: Vec<_> =
            order[1..].iter().map(|i| (*i, Integer::zero())).collect();
        let degree = self.degree(order[0]);
        let mut seed = 0x9e37_79b9_7f4a_7c15u64;
        for (i, &v) in order.iter().enumerate() {
            seed ^= ((v as u64) + 0x517c_c1b7_2722_0a95u64).rotate_left(((i * 13) % 64) as u32);
        }
        seed ^= (self.nterms() as u64).rotate_left(17);
        seed ^= (degree.to_u32() as u64).rotate_left(31);
        seed ^= (coefficient_upper_bound as u64).rotate_left(43);
        let mut rng = StdRng::seed_from_u64(seed);

        let uni_lcoeff = self.univariate_lcoeff(order[0]);
        let lcoeff_content = uni_lcoeff.content().abs();
        let lcoeff_factorization = uni_lcoeff.factor();

        // Store powers for all evaluated variables so that each univariate image can
        // be constructed in one pass over the input terms.
        let mut power_cache = (0..self.nvars())
            .map(|i| {
                vec![Integer::zero(); (self.degree(i).to_u32() as usize + 1).min(POW_CACHE_SIZE)]
            })
            .collect::<Vec<_>>();

        const SAMPLE_ATTEMPTS: usize = 512;
        const ATTEMPTS_PER_BOUND: usize = 32;
        const MAX_SAMPLE_BOUND: i64 = 4096;
        let mut sample_bound = coefficient_upper_bound.clamp(10, MAX_SAMPLE_BOUND);
        let prime_start = if coefficient_upper_bound <= 10 {
            1
        } else {
            coefficient_upper_bound as u64
        };
        let mut primes = PrimeIteratorU64::new(prime_start);

        for attempt in 0..SAMPLE_ATTEMPTS {
            if attempt < WANG_PRIME_SAMPLE_ATTEMPTS {
                // Each block uses fresh prime coordinates. Distinct variable factors
                // then have pairwise-coprime images, while another block can replace
                // an image that acquired content during specialization.
                for sample in &mut cur_sample_points {
                    loop {
                        let prime = Integer::from(primes.next().unwrap());
                        if !(&lcoeff_content % &prime).is_zero() {
                            sample.1 = prime;
                            break;
                        }
                    }
                }
            } else {
                let random_attempt = attempt - WANG_PRIME_SAMPLE_ATTEMPTS;
                if random_attempt > 0 && random_attempt % ATTEMPTS_PER_BOUND == 0 {
                    sample_bound = sample_bound.saturating_mul(2).min(MAX_SAMPLE_BOUND);
                }

                for sample in &mut cur_sample_points {
                    sample.1 = Integer::Single(rng.random_range(2..=sample_bound));
                }
            }

            for sample in &cur_sample_points {
                debug!("Sample x{} {}", sample.0, sample.1);
            }

            if !Self::lcoeff_sample_supports_wang_reconstruction(
                &lcoeff_factorization,
                &lcoeff_content,
                &cur_sample_points,
            ) {
                continue;
            }

            for powers in &mut power_cache {
                powers.fill(Integer::zero());
            }
            let cur_uni_f = self.replace_except(order[0], &cur_sample_points, &mut power_cache);

            if degree != cur_uni_f.degree(order[0]) {
                continue;
            }

            let univariate_content = cur_uni_f.univariate_content(order[0]);
            if !univariate_content.is_constant() {
                continue;
            }

            let univariate_content = univariate_content.get_constant();
            // Primitiveization removes this content from the leading coefficient.
            // If it contains the image of a variable factor, Wang reconstruction
            // cannot assign that factor's full multiplicity to the primitive image.
            if !Self::lcoeff_sample_supports_wang_reconstruction(
                &lcoeff_factorization,
                &univariate_content.abs(),
                &cur_sample_points,
            ) {
                continue;
            }
            let primitive_uni_f = cur_uni_f.clone().div_coeff(&univariate_content);

            let uni_df = primitive_uni_f.derivative(order[0]);
            if !primitive_uni_f.gcd(&uni_df).is_constant() {
                continue;
            }

            // The image is primitive and has just been checked to be square-free,
            // so factor it directly without repeating content and square-free GCDs.
            let factor_target = if primitive_uni_f.lcoeff().is_negative() {
                -primitive_uni_f.clone()
            } else {
                primitive_uni_f.clone()
            };
            let mut univariate_factors = factor_target.factor_reconstruct();
            if univariate_factors.is_empty() {
                continue;
            }

            let mut product = self.one();
            for f in &univariate_factors {
                product = &product * f;
            }

            if product != primitive_uni_f {
                if -product == primitive_uni_f {
                    univariate_factors[0] =
                        univariate_factors[0].clone().mul_coeff(Integer::from(-1));
                } else {
                    continue;
                }
            }

            let Some(true_lcoeffs) = self.reconstruct_lcoeffs_from_univariate_sample(
                &uni_lcoeff,
                &lcoeff_factorization,
                &univariate_factors,
                &cur_sample_points,
                &univariate_content,
            ) else {
                continue;
            };

            if max_factors_num.is_none_or(|max| univariate_factors.len() <= max) {
                return Some((
                    univariate_factors,
                    true_lcoeffs,
                    cur_sample_points,
                    coefficient_upper_bound.max(sample_bound),
                    cur_uni_f,
                    univariate_content,
                ));
            }
        }

        None
    }

    fn impose_true_lcoeffs_on_integer_factors(
        &self,
        factors: &[Self],
        true_lcoeffs: &[Self],
        order: &[usize],
    ) -> Vec<Self> {
        let mut factors_with_true_lcoeff = Vec::with_capacity(factors.len());

        for (factor, true_lcoeff) in factors.iter().zip(true_lcoeffs) {
            let mut coefficients = factor.to_univariate_polynomial_list(order[0]);
            coefficients.last_mut().unwrap().0 = true_lcoeff.clone();

            let mut fixed_factor = self.zero();
            let mut exp = vec![E::zero(); self.nvars()];
            for (coefficient, degree) in coefficients {
                exp[order[0]] = degree;
                fixed_factor = fixed_factor + coefficient.mul_exp(&exp);
            }

            factors_with_true_lcoeff.push(fixed_factor);
        }

        factors_with_true_lcoeff
    }

    fn sparse_coefficient_hensel_lift_mod_prime(
        &self,
        mut factorization: Vec<Self>,
        true_lcoeffs: &[Self],
        p: u32,
        max_p: &Integer,
        order: &[usize],
    ) -> Option<Vec<Self>> {
        let field = Zp::new(p);
        let p_int: Integer = (p as u64).into();
        factorization =
            self.impose_true_lcoeffs_on_integer_factors(&factorization, true_lcoeffs, order);

        let factors_mod_p: Vec<_> = factorization
            .iter()
            .map(|f| f.map_coeff(|c| c.to_finite_field(&field), field.clone()))
            .collect();
        let skeletons: Vec<_> = factors_mod_p
            .iter()
            .map(|f| f.mod_var(order[0], f.degree(order[0])))
            .collect();

        let prods_mod_p: Vec<_> = (0..factors_mod_p.len())
            .map(|i| {
                let mut prod = factors_mod_p[i].one();
                for (j, f) in factors_mod_p.iter().enumerate() {
                    if i != j {
                        prod = prod * f;
                    }
                }
                prod
            })
            .collect();

        let mut product = self.one();
        for f in &factorization {
            product = &product * f;
        }
        let mut error = self - &product;

        let mut m = p_int.clone();
        let mut sparse_diophantine_context = SparseDiophantineContext::new();
        while !error.is_zero() && &m <= max_p {
            let mut error_mod_p = factors_mod_p[0].zero();
            for term in &error {
                if !(term.coefficient % &m).is_zero() {
                    return None;
                }
                let q = term.coefficient / &m;
                error_mod_p.append_monomial(q.to_finite_field(&field), term.exponents);
            }

            let deltas = MultivariatePolynomial::sparse_multivariate_diophantine_from_skeleton(
                &factors_mod_p,
                &prods_mod_p,
                &error_mod_p,
                &skeletons,
                order,
                &mut sparse_diophantine_context,
            )?;

            for (factor, delta) in factorization.iter_mut().zip(deltas) {
                *factor = &*factor
                    + &delta
                        .map_coeff(|c| field.to_symmetric_integer(c), Z)
                        .mul_coeff(m.clone());
            }

            factorization =
                self.impose_true_lcoeffs_on_integer_factors(&factorization, true_lcoeffs, order);

            product = self.one();
            for f in &factorization {
                product = &product * f;
            }
            error = self - &product;
            m = &m * &p_int;
        }

        Some(factorization)
    }

    fn multivariate_factorization_bivariate_start(
        &self,
        order: &mut [usize],
        mut coefficient_upper_bound: i64,
        mut max_bivariate_factors: Option<usize>,
    ) -> Vec<Self> {
        if let Some(m) = max_bivariate_factors
            && m == 1
        {
            return vec![self.clone()];
        }

        let (bivariate_factors, sample_points, uni_f) = loop {
            let (bivariate_factors, sample_points, coeff_b, uni_f) = self.find_sample(
                order,
                coefficient_upper_bound.max(10),
                max_bivariate_factors,
            );

            coefficient_upper_bound = coeff_b;

            if bivariate_factors.len() == 1 {
                return vec![self.clone()];
            }

            if let Some(max) = max_bivariate_factors {
                if bivariate_factors.len() < max {
                    debug!(
                        "Updating bivariate factor bound to {}",
                        bivariate_factors.len()
                    );
                    max_bivariate_factors = Some(bivariate_factors.len());
                }
            } else {
                debug!(
                    "Updating bivariate factor bound to {}",
                    bivariate_factors.len()
                );
                max_bivariate_factors = Some(bivariate_factors.len());
            }

            break (bivariate_factors, sample_points, uni_f);
        };

        let mut prime_iter = PrimeIteratorU64::new(1 << 31);
        let mut field;
        let mut p;
        'new_prime: loop {
            p = prime_iter
                .next()
                .expect("Ran out of primes during factorization");

            if p > u32::MAX as u64 {
                panic!("Ran out of primes during factorization of {self}");
            }

            if (&uni_f.lcoeff() % &p.into()).is_zero() {
                continue;
            }

            field = Zp::new(p as u32);

            let fs_p: Vec<_> = bivariate_factors
                .iter()
                .map(|f| f.map_coeff(|c| c.to_finite_field(&field), field.clone()))
                .collect();

            for (f_p, f_z) in fs_p.iter().zip(&bivariate_factors) {
                if f_p.degree(order[0]) != f_z.degree(order[0])
                    || f_p.degree(order[1]) != f_z.degree(order[1])
                    || f_p.ring().try_inv(&f_p.lcoeff()).is_none()
                {
                    continue 'new_prime;
                }
            }

            for (j, f) in fs_p.iter().enumerate() {
                for g in &fs_p[j + 1..] {
                    if !f.gcd(g).is_one() {
                        continue 'new_prime;
                    }
                }
            }

            break;
        }
        let p32 = p as u32;

        let p_int = field.get_prime().to_integer();
        let mut lcoeff_max_p = p_int.clone();
        let mut k = 1;
        let lcoeff_bound = self.coefficient_bound();
        while &lcoeff_max_p * 2 < lcoeff_bound {
            lcoeff_max_p = &lcoeff_max_p * &p_int;
            k += 1;
        }

        let (leftover_lc, mut sorted_biv_factors, mut true_lcoeffs) = match self
            .lcoeff_precomputation(
                &bivariate_factors,
                &sample_points,
                order,
                lcoeff_max_p,
                p32,
                k,
            ) {
            Ok((leftover_lc, sorted_biv_factors, true_lcoeffs)) => {
                (leftover_lc, sorted_biv_factors, true_lcoeffs)
            }
            Err(max_biv) => {
                return self.multivariate_factorization_bivariate_start(
                    order,
                    coefficient_upper_bound + 10,
                    Some(max_biv),
                );
            }
        };

        let rescaled = if leftover_lc != 1 {
            for (b, l) in sorted_biv_factors.iter_mut().zip(&mut true_lcoeffs) {
                *b = b.clone().mul_coeff(leftover_lc.clone());
                *l = l.clone().mul_coeff(leftover_lc.clone());
            }

            Cow::Owned(
                self.clone()
                    .mul_coeff(leftover_lc.pow(sorted_biv_factors.len() as u64 - 1)),
            )
        } else {
            Cow::Borrowed(self)
        };

        let bound = rescaled.coefficient_bound();
        let mut max_p = p_int.clone();
        while &max_p * 2 < bound {
            max_p = &max_p * &p_int;
        }

        for (b, l) in sorted_biv_factors.iter().zip(&true_lcoeffs) {
            debug!("Bivariate factor {} with true lcoeff {}", b, l);
        }
        let poly_p = rescaled.map_coeff(|c| c.to_finite_field(&field), field.clone());
        let sample_points_p: Vec<_> = sample_points
            .iter()
            .map(|(v, p)| (*v, p.to_finite_field(&field)))
            .collect();
        let true_lcoeffs_p: Vec<_> = true_lcoeffs
            .iter()
            .map(|f| f.map_coeff(|c| c.to_finite_field(&field), field.clone()))
            .collect();
        let sorted_biv_factors_p: Vec<_> = sorted_biv_factors
            .iter()
            .map(|f| f.map_coeff(|c| c.to_finite_field(&field), field.clone()))
            .collect();
        let sorted_biv_factors_p = poly_p.impose_true_lcoeffs_on_factors(
            &sorted_biv_factors_p,
            &true_lcoeffs_p,
            &sample_points_p,
            order,
            2,
        );

        let (mut uni, delta) = MultivariatePolynomial::univariate_diophantine_field(
            &sorted_biv_factors_p,
            order,
            &sample_points_p,
        );

        let Ok(factorization_p) = poly_p.multivariate_hensel_lifting(
            &sorted_biv_factors_p,
            &mut uni,
            &delta,
            &sample_points_p,
            Some(&true_lcoeffs_p),
            order,
            MultivariateHenselContext::new(2),
        ) else {
            return self.multivariate_factorization_bivariate_start(
                order,
                coefficient_upper_bound + 10,
                max_bivariate_factors,
            );
        };

        let factorization_z: Vec<_> = factorization_p
            .into_iter()
            .map(|f| f.map_coeff(|c| field.to_symmetric_integer(c), Z))
            .collect();

        let Some(factorization_z) = rescaled.sparse_coefficient_hensel_lift_mod_prime(
            factorization_z,
            &true_lcoeffs,
            p32,
            &max_p,
            order,
        ) else {
            return self.multivariate_factorization_bivariate_start(
                order,
                coefficient_upper_bound + 10,
                max_bivariate_factors,
            );
        };

        let mut factorization: Vec<MultivariatePolynomial<IntegerRing, E>> = factorization_z
            .into_iter()
            .map(|f| f.make_primitive())
            .collect();

        let mut test = self.one();
        for f in &factorization {
            debug!("Factor = {}", f);
            test = &test * f;
        }

        if self.lcoeff().is_negative() != test.lcoeff().is_negative() {
            test = -test;
            if let Some(neg_coeff) = factorization.iter_mut().find(|f| f.lcoeff().is_negative()) {
                *neg_coeff = -neg_coeff.clone();
            } else {
                factorization[0] = factorization[0].clone().mul_coeff((-1).into());
            }
        }

        if &test == self {
            let mut negated_factors = 0usize;
            for f in &mut factorization {
                if f.lcoeff().is_negative() {
                    *f = -f.clone();
                    negated_factors += 1;
                }
            }
            if negated_factors % 2 == 1 {
                factorization[0] = -factorization[0].clone();
            }
            factorization
        } else {
            self.multivariate_factorization_bivariate_start(
                order,
                coefficient_upper_bound + 10,
                max_bivariate_factors,
            )
        }
    }

    /// Perform multivariate factorization on a square-free polynomial.
    fn multivariate_factorization(
        &self,
        order: &mut [usize],
        coefficient_upper_bound: i64,
        max_univariate_factors: Option<usize>,
    ) -> Vec<Self> {
        self.multivariate_factorization_with_retries(
            order,
            coefficient_upper_bound,
            max_univariate_factors,
            0,
        )
    }

    fn multivariate_factorization_with_retries(
        &self,
        order: &mut [usize],
        mut coefficient_upper_bound: i64,
        mut max_univariate_factors: Option<usize>,
        univariate_retries: usize,
    ) -> Vec<Self> {
        if let Some(m) = max_univariate_factors
            && m == 1
        {
            return vec![self.clone()];
        }

        let integer_start_mode = integer_factor_start_mode();
        let use_bivariate_start = match integer_start_mode {
            IntegerFactorStart::Auto => {
                if univariate_retries >= INTEGER_FACTOR_UNIVARIATE_AUTO_RETRIES {
                    return self.multivariate_factorization_bivariate_start(
                        order,
                        coefficient_upper_bound,
                        max_univariate_factors,
                    );
                }
                let (use_bivariate, _) = self.integer_factor_start_auto_decision(order);
                use_bivariate
            }
            IntegerFactorStart::Univariate => false,
            IntegerFactorStart::Bivariate => true,
            IntegerFactorStart::Disabled => return vec![self.clone()],
        };

        if use_bivariate_start {
            return self.multivariate_factorization_bivariate_start(
                order,
                coefficient_upper_bound,
                max_univariate_factors,
            );
        }

        let (univariate_factors, lc_divs, sample_points, uni_f, univariate_content) = loop {
            let Some((
                univariate_factors,
                lc_divs,
                sample_points,
                coeff_b,
                uni_f,
                univariate_content,
            )) = self.find_univariate_sample(
                order,
                coefficient_upper_bound.max(10),
                max_univariate_factors,
            )
            else {
                if let Some(max) = max_univariate_factors {
                    debug!(
                        "No univariate image with at most {max} factors; relaxing the stale factor bound"
                    );
                    max_univariate_factors = None;
                    coefficient_upper_bound = coefficient_upper_bound.saturating_add(10);
                    continue;
                }

                match integer_start_mode {
                    IntegerFactorStart::Auto | IntegerFactorStart::Bivariate => {
                        return self.multivariate_factorization_bivariate_start(
                            order,
                            coefficient_upper_bound.saturating_add(10),
                            None,
                        );
                    }
                    IntegerFactorStart::Univariate => {
                        // Start a new deterministic sampling block. Including this
                        // monotonically growing cursor in the RNG seed prevents a
                        // failed block from repeating the same candidate images.
                        coefficient_upper_bound = coefficient_upper_bound.saturating_add(10);
                        continue;
                    }
                    IntegerFactorStart::Disabled => return vec![self.clone()],
                }
            };

            coefficient_upper_bound = coeff_b;

            if univariate_factors.len() == 1 {
                // the polynomial is irreducible
                return vec![self.clone()];
            }

            if let Some(max) = max_univariate_factors {
                if univariate_factors.len() < max {
                    debug!(
                        "Updating univariate factor bound to {}",
                        univariate_factors.len()
                    );
                    max_univariate_factors = Some(univariate_factors.len());
                }
            } else {
                debug!(
                    "Updating univariate factor bound to {}",
                    univariate_factors.len()
                );
                max_univariate_factors = Some(univariate_factors.len());
            }

            break (
                univariate_factors,
                lc_divs,
                sample_points,
                uni_f,
                univariate_content,
            );
        };
        let univariate_factor_count = univariate_factors.len();

        // select a suitable prime
        let mut prime_iter = PrimeIteratorU64::new(1 << 31);
        let mut field;
        let mut p;
        'new_prime: loop {
            p = prime_iter
                .next()
                .expect("Ran out of primes during factorization");

            if p > u32::MAX as u64 {
                panic!("Ran out of primes during factorization of {self}");
            }

            if (&uni_f.lcoeff() % &p.into()).is_zero() {
                continue;
            }

            field = Zp::new(p as u32);

            // make sure the univariate factors stay square-free and coprime modulo p
            let fs_p: Vec<_> = univariate_factors
                .iter()
                .map(|f| f.map_coeff(|c| c.to_finite_field(&field), field.clone()))
                .collect();

            for (f_p, f_z) in fs_p.iter().zip(&univariate_factors) {
                if f_p.degree(order[0]) != f_z.degree(order[0])
                    || f_p.ring().try_inv(&f_p.lcoeff()).is_none()
                {
                    continue 'new_prime;
                }
            }

            for (j, f) in fs_p.iter().enumerate() {
                for g in &fs_p[j + 1..] {
                    if !f.gcd(g).is_one() {
                        continue 'new_prime;
                    }
                }
            }

            break;
        }
        let p32 = p as u32;

        let scaled_true_lcoeffs: Vec<_> = lc_divs
            .iter()
            .map(|l| l.clone().mul_coeff(univariate_content.clone()))
            .collect();
        let scaled_univariate_factors: Vec<_> = univariate_factors
            .iter()
            .map(|f| f.clone().mul_coeff(univariate_content.clone()))
            .collect();

        let mut scale_pow = Integer::one();
        for _ in 1..univariate_factor_count {
            scale_pow *= &univariate_content;
        }
        let scaled_self = self.clone().mul_coeff(scale_pow);

        let bound = scaled_self.coefficient_bound();
        let p_int = field.get_prime().to_integer();
        let mut max_p = p_int.clone();
        while &max_p * 2 < bound {
            max_p = &max_p * &p_int;
        }

        for (u, l) in univariate_factors.iter().zip(&lc_divs) {
            debug!("Univariate factor {} with true lcoeff {}", u, l);
        }
        let poly_p = scaled_self.map_coeff(|c| c.to_finite_field(&field), field.clone());
        let sample_points_p: Vec<_> = sample_points
            .iter()
            .map(|(v, p)| (*v, p.to_finite_field(&field)))
            .collect();
        let true_lcoeffs_p: Vec<_> = scaled_true_lcoeffs
            .iter()
            .map(|f| f.map_coeff(|c| c.to_finite_field(&field), field.clone()))
            .collect();
        let univariate_factors_p: Vec<_> = scaled_univariate_factors
            .iter()
            .map(|f| f.map_coeff(|c| c.to_finite_field(&field), field.clone()))
            .collect();
        let mut uni = univariate_factors_p.clone();
        let Some(delta) = MultivariatePolynomial::try_univariate_diophantine(
            &mut uni,
            &univariate_factors_p[0].constant(field.one()),
        ) else {
            return self.multivariate_factorization_with_retries(
                order,
                coefficient_upper_bound + 10,
                max_univariate_factors,
                univariate_retries + 1,
            );
        };

        let hensel_context = if integer_start_mode == IntegerFactorStart::Auto {
            MultivariateHenselContext::new(1).retry_sample_on_sparse_failure()
        } else {
            MultivariateHenselContext::new(1)
        };
        let Ok(factorization_p) = poly_p.multivariate_hensel_lifting(
            &univariate_factors_p,
            &mut uni,
            &delta,
            &sample_points_p,
            Some(&true_lcoeffs_p),
            order,
            hensel_context,
        ) else {
            return self.multivariate_factorization_with_retries(
                order,
                coefficient_upper_bound + 10,
                max_univariate_factors,
                univariate_retries + 1,
            );
        };

        let factorization_z: Vec<_> = factorization_p
            .into_iter()
            .map(|f| f.map_coeff(|c| field.to_symmetric_integer(c), Z))
            .collect();

        let Some(factorization_z) = scaled_self.sparse_coefficient_hensel_lift_mod_prime(
            factorization_z,
            &scaled_true_lcoeffs,
            p32,
            &max_p,
            order,
        ) else {
            return self.multivariate_factorization_with_retries(
                order,
                coefficient_upper_bound + 10,
                max_univariate_factors,
                univariate_retries + 1,
            );
        };

        let mut factorization: Vec<MultivariatePolynomial<IntegerRing, E>> = factorization_z
            .into_iter()
            .map(|f| f.make_primitive())
            .collect();

        // test the factorization
        let mut test = self.one();
        for f in &factorization {
            debug!("Factor = {}", f);
            test = &test * f;
        }

        if self.lcoeff().is_negative() != test.lcoeff().is_negative() {
            test = -test;
            if let Some(neg_coeff) = factorization.iter_mut().find(|f| f.lcoeff().is_negative()) {
                *neg_coeff = -neg_coeff.clone();
            } else {
                factorization[0] = factorization[0].clone().mul_coeff((-1).into());
            }
        }

        if &test == self {
            let mut negated_factors = 0usize;
            for f in &mut factorization {
                if f.lcoeff().is_negative() {
                    *f = -f.clone();
                    negated_factors += 1;
                }
            }
            if negated_factors % 2 == 1 {
                factorization[0] = -factorization[0].clone();
            }
            return factorization;
        } else {
            debug!(
                "No immediate factorization of {} for sample points {:?}, retrying with factor bound {:?}",
                self, sample_points, max_univariate_factors
            );

            return self.multivariate_factorization_with_retries(
                order,
                coefficient_upper_bound + 10,
                max_univariate_factors,
                univariate_retries + 1,
            );
        }
    }
}

impl<E: PositiveExponent> MultivariatePolynomial<FiniteField<Integer>, E, LexOrder> {
    /// Compute a univariate diophantine equation in `Z_p^k` by Newton iteration.
    fn get_univariate_factors_and_deltas(
        factors: &[Self],
        order: &[usize],
        sample_points: &[(usize, Integer)],
        p: u32,
        k: usize,
    ) -> (Vec<Self>, Vec<Self>) {
        // produce univariate factors and univariate delta
        let mut univariate_factors = factors.to_vec();
        for f in &mut univariate_factors {
            for (v, s) in sample_points {
                if order[0] != *v {
                    *f = f.replace(*v, &f.ring().nth(s.clone()));
                }
            }
        }

        let univariate_deltas = MultivariatePolynomial::lift_diophantine_univariate(
            &mut univariate_factors,
            &factors[0].constant(factors[0].ring().one()),
            p,
            k,
        );

        (univariate_factors, univariate_deltas)
    }
}

#[cfg(test)]
mod test {
    use std::sync::{Arc, Mutex, atomic::Ordering};

    use super::{
        BOUNDED_DDF_REJECTIONS, DenseBivariateImage, DenseTwoFactorCorrectionContext,
        ExactPolynomialSquareRoot, IntegerModularUnivariateContext,
        LAST_BOUNDED_DDF_REJECTION_DEGREE, LAST_MODULAR_INTEGER_EDF_PRIME,
        MODULAR_INTEGER_EDF_CALLS, ModularPrimeScreen, QUADRATIC_HENSEL_LIFT_CALLS,
        QuadraticFactorization, SparseDiophantineContext,
    };

    use crate::{
        GLOBAL_SETTINGS,
        atom::AtomCore,
        domains::{
            InternalOrdering, Ring,
            algebraic::AlgebraicExtension,
            finite_field::{FiniteField, FiniteFieldCore, ToFiniteField, Z2, Zp},
            integer::{Integer, IntegerRing, Z},
            rational::Q,
        },
        parse,
        poly::{MultivariatePolynomial, factor::Factorize},
        symbol,
    };

    static GLOBAL_FACTOR_SETTINGS_LOCK: Mutex<()> = Mutex::new(());

    fn multiply_dense_bivariate<R: Ring>(
        ring: &R,
        left: &DenseBivariateImage<R::Element>,
        right: &DenseBivariateImage<R::Element>,
    ) -> DenseBivariateImage<R::Element> {
        let x_len = left.x_len + right.x_len - 1;
        let y_len = left.y_len + right.y_len - 1;
        let mut product = DenseBivariateImage {
            x_len,
            y_len,
            coefficients: vec![ring.zero(); x_len * y_len],
        };

        for left_y in 0..left.y_len {
            for left_x in 0..left.x_len {
                let left_coefficient = left.coefficient(left_x, left_y);
                if ring.is_zero(left_coefficient) {
                    continue;
                }

                for right_y in 0..right.y_len {
                    for right_x in 0..right.x_len {
                        let right_coefficient = right.coefficient(right_x, right_y);
                        if !ring.is_zero(right_coefficient) {
                            let index = product.index(left_x + right_x, left_y + right_y);
                            ring.add_mul_assign(
                                &mut product.coefficients[index],
                                left_coefficient,
                                right_coefficient,
                            );
                        }
                    }
                }
            }
        }

        product
    }

    fn assert_dense_image_matches<R: Ring>(
        expected: &MultivariatePolynomial<R, u8>,
        image: &DenseBivariateImage<R::Element>,
        x: usize,
        y: usize,
    ) {
        let mut exponent = vec![0u8; expected.nvars()];
        for y_degree in 0..image.y_len {
            exponent[y] = y_degree as u8;
            for x_degree in 0..image.x_len {
                exponent[x] = x_degree as u8;
                let coefficient = expected
                    .coefficient(&exponent)
                    .unwrap_or_else(|| expected.ring().zero());
                assert_eq!(image.coefficient(x_degree, y_degree), &coefficient);
            }
            exponent[x] = 0;
        }
    }

    #[test]
    fn geometric_bivariate_images_match_direct_specialization() {
        let field = Zp::new(101);
        // Retained variables have nonzero, nonadjacent indices so the test also
        // checks that image coordinates use variable indices rather than positions.
        let vars = Some(Arc::new(vec![
            symbol!("z").into(),
            symbol!("x").into(),
            symbol!("w").into(),
            symbol!("y").into(),
        ]));
        let poly = parse!("3+5*x+7*y^3+11*x^2*y+3*z*x*y-2*w*x*y+17*z^2*w*x^3+19*z*w^2*x^2*y^2")
            .to_polynomial::<_, u8>(&field, vars);
        let x = 1;
        let y = 3;
        let base_points = vec![(0, field.nth(2.into())), (2, field.nth(3.into()))];
        let mut cache = MultivariatePolynomial::sample_cache(&poly, &[], &[]);
        let images = poly.evaluate_geometric_bivariate_images(x, y, &base_points, 4, &mut cache);

        for (sample_index, image) in images.iter().enumerate() {
            let mut expected = poly.clone();
            for (variable, base) in &base_points {
                let value = field.pow(base, (sample_index + 1) as u64);
                expected = expected.replace(*variable, &value);
            }
            assert_dense_image_matches(&expected, image, x, y);
        }

        // At beta^1, 3*z*x*y - 2*w*x*y = 6*x*y - 6*x*y.
        assert!(field.is_zero(images[0].coefficient(1, 1)));
    }

    #[test]
    fn geometric_bivariate_images_preserve_products() {
        let field = Zp::new(101);
        let vars = Some(Arc::new(vec![
            symbol!("z").into(),
            symbol!("x").into(),
            symbol!("w").into(),
            symbol!("y").into(),
        ]));
        let left = parse!("1+x+z*y+w*x^2*y").to_polynomial::<_, u8>(&field, vars.clone());
        let right = parse!("2+y+w*x+z*x*y^2").to_polynomial::<_, u8>(&field, vars);
        let product = &left * &right;
        let base_points = vec![(0, field.nth(2.into())), (2, field.nth(3.into()))];
        let factors = [left.clone(), right.clone()];
        let mut cache = MultivariatePolynomial::sample_cache(&product, &factors, &[]);
        let left_images =
            left.evaluate_geometric_bivariate_images(1, 3, &base_points, 4, &mut cache);
        let right_images =
            right.evaluate_geometric_bivariate_images(1, 3, &base_points, 4, &mut cache);
        let product_images =
            product.evaluate_geometric_bivariate_images(1, 3, &base_points, 4, &mut cache);

        for ((left_image, right_image), product_image) in
            left_images.iter().zip(&right_images).zip(&product_images)
        {
            assert_eq!(
                multiply_dense_bivariate(&field, left_image, right_image),
                *product_image
            );
        }
    }

    #[test]
    fn evaluated_two_factor_hensel_step_reconstructs_four_variable_factors() {
        let field = Zp::new(1_000_003);
        let vars = Some(Arc::new(vec![
            symbol!("x").into(),
            symbol!("z").into(),
            symbol!("w").into(),
            symbol!("y").into(),
        ]));
        let polynomial = |input| {
            parse!(input)
                .expand()
                .to_polynomial::<_, u8>(&field, vars.clone())
        };
        let expected = vec![
            polynomial("x^2+(z+w)*x+1+((2*z+3*w)*x+4)*y+((5*z+7*w)*x+8)*y^2"),
            polynomial("x^2+(z+w)*x+2+((11*z+13*w)*x+14)*y+((17*z+19*w)*x+20)*y^2"),
        ];
        let target = &expected[0] * &expected[1];
        let initial = expected
            .iter()
            .map(|factor| factor.replace(3, &field.zero()))
            .collect::<Vec<_>>();

        let lifted = (0..8)
            .find_map(|_| {
                target.try_multivariate_hensel_step_two_factor_evaluated(
                    &initial,
                    &[0, 1, 2, 3],
                    target.degree(3) as usize,
                    true,
                )
            })
            .expect("the evaluated two-factor lift should reconstruct both factors");

        assert_eq!(lifted, expected);
        assert_eq!(&lifted[0] * &lifted[1], target);
    }

    #[test]
    fn evaluated_two_factor_hensel_step_verifies_unexamined_product_tail() {
        let field = Zp::new(1_000_003);
        let vars = Some(Arc::new(vec![
            symbol!("x").into(),
            symbol!("z").into(),
            symbol!("y").into(),
        ]));
        let polynomial = |input| {
            parse!(input)
                .expand()
                .to_polynomial::<_, u8>(&field, vars.clone())
        };
        let initial = vec![polynomial("x+z"), polynomial("x+2*z")];
        let target = polynomial("(x+z)*(x+2*z)+(z*(x+2*z)+2*z*(x+z))*y");

        let unchecked = (0..8)
            .find_map(|_| {
                target.try_multivariate_hensel_step_two_factor_evaluated(
                    &initial,
                    &[0, 1, 2],
                    1,
                    false,
                )
            })
            .expect("the sampled corrections should reconstruct");
        assert_ne!(&unchecked[0] * &unchecked[1], target);
        assert!(
            target
                .try_multivariate_hensel_step_two_factor_evaluated(&initial, &[0, 1, 2], 1, true,)
                .is_none()
        );
    }

    #[test]
    fn evaluated_two_factor_hensel_step_rejects_oversized_image_grid() {
        let field = Zp::new(101);
        let vars = Some(Arc::new(vec![
            symbol!("x").into(),
            symbol!("z").into(),
            symbol!("y").into(),
        ]));
        let polynomial = |input| {
            parse!(input)
                .expand()
                .to_polynomial::<_, u32>(&field, vars.clone())
        };
        let expected = vec![polynomial("x^5000000+z+z*y"), polynomial("x+2*z")];
        let target = &expected[0] * &expected[1];
        let initial = expected
            .iter()
            .map(|factor| factor.replace(2, &field.zero()))
            .collect::<Vec<_>>();

        assert!(
            target
                .try_multivariate_hensel_step_two_factor_evaluated(&initial, &[0, 1, 2], 1, true,)
                .is_none()
        );
    }

    #[test]
    fn evaluated_two_factor_hensel_step_rejects_generator_collision() {
        let field = Z2;
        let vars = Some(Arc::new(vec![
            symbol!("x").into(),
            symbol!("z").into(),
            symbol!("w").into(),
            symbol!("y").into(),
        ]));
        let polynomial = |input| {
            parse!(input)
                .expand()
                .to_polynomial::<_, u8>(&field, vars.clone())
        };
        let expected = vec![
            polynomial("x^2+(z+w)*x+1+((z+w)*x+1)*y"),
            polynomial("x^2+(z+w)*x+((z+w)*x+1)*y"),
        ];
        let target = &expected[0] * &expected[1];
        let initial = expected
            .iter()
            .map(|factor| factor.replace(3, &field.zero()))
            .collect::<Vec<_>>();

        assert!(
            target
                .try_multivariate_hensel_step_two_factor_evaluated(
                    &initial,
                    &[0, 1, 2, 3],
                    target.degree(3) as usize,
                    true,
                )
                .is_none()
        );
    }

    #[test]
    fn dense_two_factor_corrections_match_generic_univariate_solver() {
        let field = Zp::new(101);
        let vars = Some(Arc::new(vec![symbol!("x").into(), symbol!("y").into()]));
        let polynomial = |input| {
            parse!(input)
                .expand()
                .to_polynomial::<_, u8>(&field, vars.clone())
        };
        let factors = [polynomial("3*x^3+2*x+1"), polynomial("5*x^2+7*x+4")];
        let mut cache = MultivariatePolynomial::sample_cache(&factors[0], &factors, &[]);
        let factor_images = factors
            .iter()
            .map(|factor| {
                factor
                    .evaluate_geometric_bivariate_images(0, 1, &[], 1, &mut cache)
                    .pop()
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let (gcd, s, t) = factors[0].eea_univariate(&factors[1]);
        assert!(gcd.is_one());
        let context = DenseTwoFactorCorrectionContext {
            multipliers: [
                factors[0].dense_indexed_univariate(&t, 0).unwrap(),
                factors[0].dense_indexed_univariate(&s, 0).unwrap(),
            ],
            moduli: [
                factors[0].dense_monic_modulus(&factor_images[0]).unwrap(),
                factors[0].dense_monic_modulus(&factor_images[1]).unwrap(),
            ],
        };

        for rhs in [
            polynomial("0"),
            polynomial("17"),
            polynomial("11*x^4+13*x^3+17*x+19"),
        ] {
            let rhs_image = rhs
                .evaluate_geometric_bivariate_images(0, 1, &[], 1, &mut cache)
                .pop()
                .unwrap();
            let actual = rhs
                .dense_two_factor_corrections(
                    &rhs_image.coefficients,
                    &context,
                    [factor_images[0].x_len, factor_images[1].x_len],
                )
                .unwrap();
            let expected =
                MultivariatePolynomial::try_univariate_diophantine(&mut factors.to_vec(), &rhs)
                    .unwrap();

            for factor_index in 0..2 {
                let expected_image = expected[factor_index]
                    .evaluate_geometric_bivariate_images(0, 1, &[], 1, &mut cache)
                    .pop()
                    .unwrap();
                let mut expected_coefficients = expected_image.coefficients;
                expected_coefficients.resize(actual[factor_index].len(), field.zero());
                assert_eq!(actual[factor_index], expected_coefficients);
            }
        }
    }

    #[test]
    fn packed_product_accumulation_matches_exact_product() {
        let field = Zp::new(101);
        let vars = Some(Arc::new(vec![
            symbol!("x").into(),
            symbol!("z").into(),
            symbol!("w").into(),
            symbol!("y").into(),
        ]));
        let polynomial = |input| {
            parse!(input)
                .expand()
                .to_polynomial::<_, u8>(&field, vars.clone())
        };
        let left = polynomial("1+x+z*x^2+w*y+x*z*y");
        let right = polynomial("2+w*x+z*y+x^2*y+w*z*x");
        let product = &left * &right;

        assert_eq!(
            product.product_matches_by_packed_accumulation(&left, &right),
            Some(true)
        );
        let wrong_product = &product + &polynomial("1");
        assert_eq!(
            wrong_product.product_matches_by_packed_accumulation(&left, &right),
            Some(false)
        );
    }

    #[test]
    fn packed_product_verification_preserves_total_degree_dense_dispatch() {
        let field = Zp::new(101);
        let vars = Some(Arc::new(vec![
            symbol!("a").into(),
            symbol!("b").into(),
            symbol!("c").into(),
            symbol!("d").into(),
            symbol!("e").into(),
            symbol!("f").into(),
            symbol!("g").into(),
            symbol!("h").into(),
        ]));
        let dense = parse!("(1+a+b+c+d+e+f+g+h)^5")
            .expand()
            .to_polynomial::<_, u8>(&field, vars);

        assert!(dense.total_degree_dense_mul_is_bounded(&dense));
        assert!(!MultivariatePolynomial::packed_product_accumulation_is_preferred(&dense, &dense,));
    }

    #[test]
    fn two_factor_univariate_correction_reuses_bezout_for_both_components() {
        let field = Zp::new(101);
        let vars = Some(Arc::new(vec![symbol!("x").into()]));
        let factors = ["x^2+2", "x+3"]
            .iter()
            .map(|factor| {
                parse!(factor)
                    .expand()
                    .to_polynomial::<_, u8>(&field, vars.clone())
            })
            .collect::<Vec<_>>();
        let rhs = ["7+5*x+11*x^2", "13+17*x+19*x^2"]
            .iter()
            .map(|rhs| {
                parse!(rhs)
                    .expand()
                    .to_polynomial::<_, u8>(&field, vars.clone())
            })
            .collect::<Vec<_>>();
        let mut context = SparseDiophantineContext::new();

        for rhs in &rhs {
            let expected =
                MultivariatePolynomial::try_univariate_diophantine(&mut factors.clone(), rhs)
                    .unwrap();

            for requested_factor in 0..2 {
                let correction = MultivariatePolynomial::try_two_factor_univariate_correction(
                    &mut factors.clone(),
                    rhs,
                    requested_factor,
                    &mut context,
                )
                .unwrap();
                assert_eq!(correction, expected[requested_factor]);

                let residual = rhs - &(&correction * &factors[1 - requested_factor]);
                let remainder = residual
                    .quot_rem_univariate(&mut factors[requested_factor].clone())
                    .1;
                assert!(remainder.is_zero());
                assert_eq!(context.two_factor_bezout.len(), 1);
            }
        }
    }

    #[test]
    fn two_factor_univariate_correction_caches_noncoprime_failure() {
        let field = Zp::new(101);
        let vars = Some(Arc::new(vec![symbol!("x").into()]));
        let factors = ["(x+1)*(x+2)", "(x+1)*(x+3)"]
            .iter()
            .map(|factor| {
                parse!(factor)
                    .expand()
                    .to_polynomial::<_, u8>(&field, vars.clone())
            })
            .collect::<Vec<_>>();
        let rhs = parse!("1+x").to_polynomial::<_, u8>(&field, vars).clone();
        let mut context = SparseDiophantineContext::new();

        for requested_factor in 0..2 {
            assert!(
                MultivariatePolynomial::try_two_factor_univariate_correction(
                    &mut factors.clone(),
                    &rhs,
                    requested_factor,
                    &mut context,
                )
                .is_none()
            );
            assert_eq!(context.two_factor_bezout.len(), 1);
            assert!(context.two_factor_bezout.values().next().unwrap().is_none());
        }
    }

    #[test]
    fn two_factor_sparse_correction_reuses_geometric_base() {
        let field = Zp::new(101);
        let vars = Some(Arc::new(vec![symbol!("x").into(), symbol!("z").into()]));
        let polynomial = |input| {
            parse!(input)
                .expand()
                .to_polynomial::<_, u8>(&field, vars.clone())
        };
        let factors = vec![polynomial("x+z"), polynomial("x+z+1")];
        let prods = vec![factors[1].clone(), factors[0].clone()];
        let skeletons = vec![polynomial("z"), polynomial("z")];
        let corrections = [
            vec![polynomial("2*z"), polynomial("3*z")],
            vec![polynomial("5*z"), polynomial("7*z")],
        ];
        let mut context = SparseDiophantineContext::new();
        let mut first_base = None;

        for expected in corrections {
            let error = &(&expected[0] * &prods[0]) + &(&expected[1] * &prods[1]);
            let actual = MultivariatePolynomial::sparse_multivariate_diophantine_from_skeleton(
                &factors,
                &prods,
                &error,
                &skeletons,
                &[0, 1],
                &mut context,
            )
            .unwrap();

            assert_eq!(actual, expected);
            assert_eq!(&(&actual[0] * &prods[0]) + &(&actual[1] * &prods[1]), error);
            assert_eq!(context.two_factor_bezout.len(), 1);

            if let Some(base) = &first_base {
                assert_eq!(context.two_factor_base_points.as_ref(), Some(base));
            } else {
                first_base = context.two_factor_base_points.clone();
                assert!(first_base.is_some());
            }
        }
    }

    struct FactorSettingsGuard {
        use_univariate: bool,
        use_bivariate: bool,
    }

    impl FactorSettingsGuard {
        fn new() -> Self {
            Self {
                use_univariate: GLOBAL_SETTINGS
                    .use_univariate_factorization
                    .load(Ordering::Relaxed),
                use_bivariate: GLOBAL_SETTINGS
                    .use_bivariate_factorization
                    .load(Ordering::Relaxed),
            }
        }
    }

    impl Drop for FactorSettingsGuard {
        fn drop(&mut self) {
            GLOBAL_SETTINGS
                .use_univariate_factorization
                .store(self.use_univariate, Ordering::Relaxed);
            GLOBAL_SETTINGS
                .use_bivariate_factorization
                .store(self.use_bivariate, Ordering::Relaxed);
        }
    }

    #[test]
    fn integer_factorization_respects_global_start_settings() {
        let _lock = GLOBAL_FACTOR_SETTINGS_LOCK.lock().unwrap();
        let _guard = FactorSettingsGuard::new();
        let input = "(1+v1+v2+v3)*(2+3*v1+5*v2+7*v3)";
        let poly = parse!(input).to_polynomial::<_, u8>(&Z, None);

        GLOBAL_SETTINGS
            .use_univariate_factorization
            .store(true, Ordering::Relaxed);
        GLOBAL_SETTINGS
            .use_bivariate_factorization
            .store(false, Ordering::Relaxed);
        assert_eq!(poly.factor().len(), 2);

        GLOBAL_SETTINGS
            .use_univariate_factorization
            .store(false, Ordering::Relaxed);
        GLOBAL_SETTINGS
            .use_bivariate_factorization
            .store(true, Ordering::Relaxed);
        assert_eq!(poly.factor().len(), 2);

        GLOBAL_SETTINGS
            .use_univariate_factorization
            .store(false, Ordering::Relaxed);
        GLOBAL_SETTINGS
            .use_bivariate_factorization
            .store(false, Ordering::Relaxed);
        assert_eq!(poly.factor(), vec![(poly, 1)]);
    }

    #[test]
    fn factor_ff_square_free() {
        let field = Zp::new(3);
        let poly = parse!("(1+v1)*(1+v1^2)^2*(v1^4+1)^3").to_polynomial::<_, u8>(&field, None);

        let res = [("1+v1^4", 3), ("1+v1^2", 2), ("1+v1", 1)];

        let mut res = res
            .iter()
            .map(|(f, p)| {
                (
                    parse!(f)
                        .expand()
                        .to_polynomial(&field, poly.variables().clone()),
                    *p,
                )
            })
            .collect::<Vec<_>>();
        res.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut r = poly.square_free_factorization();
        r.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));

        assert_eq!(r, res);
    }

    #[test]
    fn factor_ff_bivariate() {
        let field = Zp::new(997);
        let poly = parse!("((v2+1)*v1^2+v1*v2+1)*((v2^2+2)*v1^2+v2+1)")
            .to_polynomial::<_, u8>(&field, None);

        let res = [("1+2*v1^2+v2+v2^2*v1^2", 1), ("1+v1^2+v2*v1+v2*v1^2", 1)];

        let mut res = res
            .iter()
            .map(|(f, p)| {
                (
                    parse!(f)
                        .expand()
                        .to_polynomial(&field, poly.variables().clone()),
                    *p,
                )
            })
            .collect::<Vec<_>>();

        res.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut r = poly.factor();
        r.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        assert_eq!(r, res);
    }

    #[test]
    fn factor_square_free() {
        let poly =
            parse!("3*(2*v1^2+v2)(v1^3+v2)^2(1+4*v2)^2(1+v1)").to_polynomial::<_, u8>(&Z, None);

        let res = [
            ("3", 1),
            ("1+4*v2", 2),
            ("1+v1", 1),
            ("v2+2*v1^2", 1),
            ("v2+v1^3", 2),
        ];

        let mut res = res
            .iter()
            .map(|(f, p)| {
                (
                    parse!(f)
                        .expand()
                        .to_polynomial(&Z, poly.variables().clone()),
                    *p,
                )
            })
            .collect::<Vec<_>>();

        res.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut r = poly.square_free_factorization();
        r.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        assert_eq!(r, res);
    }

    #[test]
    fn factor_univariate_1() {
        let poly = parse!("2*(4 + 3*v1)*(3 + 2*v1 + 3*v1^2)*(3 + 8*v1^2)*(4 + v1 + v1^16)")
            .to_polynomial::<_, u8>(&Z, None);

        let res = [
            ("2", 1),
            ("4+3*v1", 1),
            ("3+2*v1+3*v1^2", 1),
            ("3+8*v1^2", 1),
            ("4+v1+v1^16", 1),
        ];

        let mut res = res
            .iter()
            .map(|(f, p)| {
                (
                    parse!(f)
                        .expand()
                        .to_polynomial(&Z, poly.variables().clone()),
                    *p,
                )
            })
            .collect::<Vec<_>>();

        res.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut r = poly.factor();
        r.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        assert_eq!(r, res);
    }

    #[test]
    fn factor_univariate_2() {
        let poly =
            parse!("(v1+1)(v1+2)(v1+3)^3(v1+4)(v1+5)(v1^2+6)(v1^3+7)(v1+8)^2(v1^4+9)(v1^5+v1+10)")
                .to_polynomial::<_, u8>(&Z, None);

        let res = [
            ("5+v1", 1),
            ("1+v1", 1),
            ("4+v1", 1),
            ("2+v1", 1),
            ("7+v1^3", 1),
            ("10+v1+v1^5", 1),
            ("6+v1^2", 1),
            ("9+v1^4", 1),
            ("8+v1", 2),
            ("3+v1", 3),
        ];

        let mut res = res
            .iter()
            .map(|(f, p)| {
                (
                    parse!(f)
                        .expand()
                        .to_polynomial(&Z, poly.variables().clone()),
                    *p,
                )
            })
            .collect::<Vec<_>>();

        res.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut r = poly.factor();
        r.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        assert_eq!(r, res);
    }

    #[test]
    fn bounded_distinct_degree_factorization_reports_exact_factor_counts() {
        let field = Zp::new(11);
        let one = parse!("1").to_polynomial::<_, u8>(&field, None);
        let bounded_one = one.distinct_degree_factorization_bounded(Some(0)).unwrap();
        assert_eq!(bounded_one.factor_count, 0);
        assert!(bounded_one.blocks.is_empty());
        assert_eq!(one.distinct_degree_factorization(), vec![(0, one.clone())]);

        let polynomial = parse!("x*(x+1)*(x^2+1)")
            .expand()
            .to_polynomial::<_, u8>(&field, None);

        let complete = polynomial
            .distinct_degree_factorization_bounded(None)
            .unwrap();
        assert_eq!(complete.factor_count, 3);
        assert_eq!(complete.blocks.len(), 2);
        let reconstructed = complete
            .blocks
            .iter()
            .fold(polynomial.one(), |product, (_, block)| &product * block);
        assert_eq!(reconstructed, polynomial);

        let limited = polynomial.distinct_degree_factorization_bounded(Some(2));
        let Err(lower_bound) = limited else {
            panic!("a two-factor limit must reject three irreducible factors");
        };
        assert_eq!(lower_bound, 3);

        let admitted = polynomial
            .distinct_degree_factorization_bounded(Some(3))
            .unwrap();
        assert_eq!(admitted.factor_count, complete.factor_count);
        assert_eq!(admitted.blocks, complete.blocks);
    }

    #[test]
    fn degree_64_modular_screening_counts_and_rejects_early() {
        let polynomial = parse!("((1+3*x)^33-1)*((1-5*x)^31+1)")
            .expand()
            .to_polynomial::<_, u8>(&Z, None);

        for (prime, expected_count) in [(7, 9), (13, 8), (17, 7), (65_000_011, 20)] {
            let Some(ModularPrimeScreen::Candidate(candidate)) =
                polynomial.screen_univariate_mod_prime(0, prime, None)
            else {
                panic!("prime {prime} must produce a suitable modular image");
            };
            assert_eq!(candidate.distinct_degree.factor_count, expected_count);
        }

        LAST_BOUNDED_DDF_REJECTION_DEGREE.with(|degree| degree.set(0));
        let Some(ModularPrimeScreen::FactorLimitExceeded { lower_bound }) =
            polynomial.screen_univariate_mod_prime(0, 65_000_011, Some(8))
        else {
            panic!("the large-prime image must exceed the eight-factor limit");
        };
        assert_eq!(lower_bound, 20);
        LAST_BOUNDED_DDF_REJECTION_DEGREE.with(|degree| assert_eq!(degree.get(), 2));
    }

    #[test]
    fn factor_univariate_degree_64_defers_discarded_equal_degree_factorization() {
        MODULAR_INTEGER_EDF_CALLS.with(|calls| calls.set(0));
        BOUNDED_DDF_REJECTIONS.with(|rejections| rejections.set(0));
        LAST_MODULAR_INTEGER_EDF_PRIME.with(|prime| prime.set(0));
        let polynomial = parse!("((1+3*x)^33-1)*((1-5*x)^31+1)")
            .expand()
            .to_polynomial::<_, u8>(&Z, None);

        let factors = polynomial.factor();
        let reconstructed = factors
            .iter()
            .fold(polynomial.one(), |product, (factor, power)| {
                &product * &factor.pow(*power)
            });

        assert_eq!(reconstructed, polynomial);
        let mut degrees = factors
            .iter()
            .filter(|(factor, _)| !factor.is_constant())
            .map(|(factor, power)| {
                assert_eq!(*power, 1);
                factor.degree(0)
            })
            .collect::<Vec<_>>();
        degrees.sort_unstable();
        assert_eq!(degrees, [1u8, 1, 2, 10, 20, 30]);
        MODULAR_INTEGER_EDF_CALLS.with(|calls| assert_eq!(calls.get(), 1));
        BOUNDED_DDF_REJECTIONS.with(|rejections| assert_eq!(rejections.get(), 1));
        LAST_MODULAR_INTEGER_EDF_PRIME.with(|prime| assert_eq!(prime.get(), 17));
    }

    #[test]
    fn quadratic_hensel_lift_handles_partial_final_precision() {
        let variables = Some(Arc::new(vec![symbol!("x").into()]));
        let left = parse!("37+19*x+6*x^2").to_polynomial::<_, u8>(&Z, variables.clone());
        let right = parse!("29-13*x+5*x^2+3*x^3").to_polynomial::<_, u8>(&Z, variables.clone());
        let product = &left * &right;
        let field = Zp::new(11);
        let left_mod = left.map_coeff(
            |coefficient| coefficient.to_finite_field(&field),
            field.clone(),
        );
        let right_mod = right.map_coeff(
            |coefficient| coefficient.to_finite_field(&field),
            field.clone(),
        );
        let max_p = Integer::from(11).pow(65);

        let quadratic = product
            .hensel_lift_with_strategy(left_mod.clone(), right_mod.clone(), None, &max_p, true)
            .unwrap();
        let linear = product
            .hensel_lift_with_strategy(left_mod, right_mod, None, &max_p, false)
            .unwrap();

        assert_eq!(&quadratic.0 * &quadratic.1, product);
        assert_eq!(quadratic, linear);
    }

    #[test]
    fn quadratic_hensel_lift_preserves_unsuccessful_congruence() {
        let variables = Some(Arc::new(vec![symbol!("x").into()]));
        let polynomial = parse!("1+x^2").to_polynomial::<_, u8>(&Z, variables.clone());
        let field = Zp::new(5);
        let left = parse!("x-2").to_polynomial::<_, u8>(&field, variables.clone());
        let right = parse!("x+2").to_polynomial::<_, u8>(&field, variables);
        let max_p = Integer::from(5).pow(65);

        let quadratic = polynomial
            .hensel_lift_with_strategy(left.clone(), right.clone(), None, &max_p, true)
            .unwrap_err();
        let linear = polynomial
            .hensel_lift_with_strategy(left, right, None, &max_p, false)
            .unwrap_err();
        let normalize = |factor: MultivariatePolynomial<IntegerRing, u8>| {
            factor.map_coeff(|coefficient| coefficient.clone().symmetric_mod(&max_p), Z)
        };
        let quadratic = (normalize(quadratic.0), normalize(quadratic.1));
        let linear = (normalize(linear.0), normalize(linear.1));
        let error = &polynomial - &(&quadratic.0 * &quadratic.1);

        assert_eq!(quadratic, linear);
        assert!(
            error
                .coefficients
                .iter()
                .all(|coefficient| (coefficient % &max_p).is_zero())
        );
    }

    #[test]
    fn linear_hensel_lift_handles_base_prime_precision() {
        let variables = Some(Arc::new(vec![symbol!("x").into()]));
        let field = Zp::new(5);
        let left = parse!("x-2").to_polynomial::<_, u8>(&field, variables.clone());
        let right = parse!("x+2").to_polynomial::<_, u8>(&field, variables.clone());
        let max_p = Integer::from(5);

        let exact = parse!("x^2-4").to_polynomial::<_, u8>(&Z, variables.clone());
        let exact_lift = exact
            .hensel_lift_with_strategy(left.clone(), right.clone(), None, &max_p, false)
            .unwrap();
        assert_eq!(&exact_lift.0 * &exact_lift.1, exact);

        let inexact = parse!("1+x^2").to_polynomial::<_, u8>(&Z, variables);
        let inexact_lift = inexact
            .hensel_lift_with_strategy(left, right, None, &max_p, false)
            .unwrap_err();
        let error = &inexact - &(&inexact_lift.0 * &inexact_lift.1);
        assert!(
            error
                .coefficients
                .iter()
                .all(|coefficient| (coefficient % &max_p).is_zero())
        );
    }

    #[test]
    fn linear_hensel_lift_handles_binary_prime_and_nontrivial_gamma() {
        let variables = Some(Arc::new(vec![symbol!("x").into()]));

        let binary_left = parse!("x+5").to_polynomial::<_, u8>(&Z, variables.clone());
        let binary_right = parse!("x^2+3*x+3").to_polynomial::<_, u8>(&Z, variables.clone());
        let binary_product = &binary_left * &binary_right;
        let binary_field = Z2;
        let binary_left_mod = binary_left.map_coeff(
            |coefficient| coefficient.to_finite_field(&binary_field),
            binary_field.clone(),
        );
        let binary_right_mod = binary_right.map_coeff(
            |coefficient| coefficient.to_finite_field(&binary_field),
            binary_field.clone(),
        );
        let binary_lift = binary_product
            .hensel_lift_with_strategy(
                binary_left_mod,
                binary_right_mod,
                None,
                &Integer::from(2).pow(20),
                true,
            )
            .unwrap();
        assert_eq!(&binary_lift.0 * &binary_lift.1, binary_product);

        let nonmonic_left = parse!("2*x+11").to_polynomial::<_, u8>(&Z, variables.clone());
        let nonmonic_right = parse!("3*x+7").to_polynomial::<_, u8>(&Z, variables);
        let nonmonic_product = &nonmonic_left * &nonmonic_right;
        let nonmonic_field = Zp::new(5);
        let nonmonic_left_mod = nonmonic_left.map_coeff(
            |coefficient| coefficient.to_finite_field(&nonmonic_field),
            nonmonic_field.clone(),
        );
        let nonmonic_right_mod = nonmonic_right.map_coeff(
            |coefficient| coefficient.to_finite_field(&nonmonic_field),
            nonmonic_field.clone(),
        );
        let nonmonic_lift = nonmonic_product
            .hensel_lift_with_strategy(
                nonmonic_left_mod,
                nonmonic_right_mod,
                Some(Integer::from(2)),
                &Integer::from(5).pow(10),
                false,
            )
            .unwrap();
        assert_eq!(&nonmonic_lift.0 * &nonmonic_lift.1, nonmonic_product);
    }

    #[test]
    fn integer_modular_univariate_arithmetic_matches_finite_field_reference() {
        let variables = Some(Arc::new(vec![
            symbol!("x").into(),
            symbol!("y").into(),
            symbol!("z").into(),
        ]));
        let dividend = parse!("19-31*y+47*y^2+53*y^3-71*y^5+89*y^8")
            .to_polynomial::<_, u8>(&Z, variables.clone());
        let divisor = parse!("17+23*y-29*y^2-3*y^3").to_polynomial::<_, u8>(&Z, variables);

        for modulus in [Integer::from(5).pow(8), Integer::from(5).pow(65)] {
            let context = IntegerModularUnivariateContext::new(&modulus, &dividend);
            let reduced_dividend = context.reduce(&dividend);
            let reduced_divisor = context.reduce(&divisor);
            let (quotient, remainder) = context.quot_rem(&reduced_dividend, &reduced_divisor);

            let field = FiniteField::<Integer>::new_non_prime(modulus.clone());
            let dividend_mod = dividend.map_coeff(
                |coefficient| coefficient.to_finite_field(&field),
                field.clone(),
            );
            let divisor_mod = divisor.map_coeff(
                |coefficient| coefficient.to_finite_field(&field),
                field.clone(),
            );
            let reference_add = (&dividend_mod + &divisor_mod)
                .map_coeff(|coefficient| field.to_symmetric_integer(coefficient), Z);
            let reference_product = (&dividend_mod * &divisor_mod)
                .map_coeff(|coefficient| field.to_symmetric_integer(coefficient), Z);
            let mut divisor_for_division = divisor_mod.clone();
            let (reference_quotient, reference_remainder) =
                dividend_mod.quot_rem_univariate(&mut divisor_for_division);
            let reference_quotient = reference_quotient
                .map_coeff(|coefficient| field.to_symmetric_integer(coefficient), Z);
            let reference_remainder = reference_remainder
                .map_coeff(|coefficient| field.to_symmetric_integer(coefficient), Z);

            assert_eq!(
                context.add(&reduced_dividend, &reduced_divisor),
                reference_add
            );
            assert_eq!(
                context.multiply(&reduced_dividend, &reduced_divisor),
                reference_product
            );
            assert_eq!(quotient, reference_quotient);
            assert_eq!(remainder, reference_remainder);

            let product = &quotient * &reduced_divisor;
            let error = &(&reduced_dividend - &product) - &remainder;

            assert!(remainder.degree(1) < reduced_divisor.degree(1));
            assert!(
                error
                    .coefficients
                    .iter()
                    .all(|coefficient| (coefficient % &modulus).is_zero())
            );
        }
    }

    #[test]
    fn factor_univariate_high_height_uses_quadratic_hensel_lift() {
        QUADRATIC_HENSEL_LIFT_CALLS.with(|calls| calls.set(0));
        let polynomial = parse!("((1+65537*x)^17-1)*((1-65539*x)^16+1)")
            .expand()
            .to_polynomial::<_, u8>(&Z, None);

        let factors = polynomial.factor();
        let expanded = factors
            .iter()
            .fold(polynomial.one(), |product, (factor, power)| {
                &product * &factor.pow(*power)
            });

        assert!(factors.len() >= 3);
        assert_eq!(expanded, polynomial);
        QUADRATIC_HENSEL_LIFT_CALLS.with(|calls| assert!(calls.get() > 0));
    }

    #[test]
    fn factor_univariate_many_modular_factors_keeps_linear_hensel_lift() {
        QUADRATIC_HENSEL_LIFT_CALLS.with(|calls| calls.set(0));
        let polynomial = parse!("((1+3*x)^32-1)*((1-5*x)^31+1)")
            .expand()
            .to_polynomial::<_, u8>(&Z, None);

        let factors = polynomial.factor();
        let expanded = factors
            .iter()
            .fold(polynomial.one(), |product, (factor, power)| {
                &product * &factor.pow(*power)
            });

        assert_eq!(expanded, polynomial);
        QUADRATIC_HENSEL_LIFT_CALLS.with(|calls| assert_eq!(calls.get(), 0));
    }

    #[test]
    fn factor_bivariate() {
        let input = "(v1^2+v2+v1+1)(3*v1+v2^2+4)*(6*v1*(v2+1)+v2+5)*(7*v1*v2+4)";
        let poly = parse!(input).to_polynomial::<_, u8>(&Z, None);

        let res = [
            ("(1+v2+v1+v1^2)", 1),
            ("(5+v2+6*v1+6*v1*v2)", 1),
            ("(4+v2^2+3*v1)", 1),
            ("(4+7*v1*v2)", 1),
        ];

        let mut res = res
            .iter()
            .map(|(f, p)| {
                (
                    parse!(f)
                        .expand()
                        .to_polynomial(&Z, poly.variables().clone()),
                    *p,
                )
            })
            .collect::<Vec<_>>();

        res.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut r = poly.factor();
        r.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        assert_eq!(r, res);
    }

    #[test]
    fn factor_multivariate() {
        let input = "(v1*(2+2*v2+2*v3)+1)*(v1*(4+v3^2)+v2+3)*(v1*(v4+v4^2+4+v2)+v4+5)";
        let poly = parse!(input).to_polynomial::<_, u8>(&Z, None);

        let res = [
            ("5+v4+4*v1+v1*v4+v1*v4^2+v1*v2", 1),
            ("1+2*v1+2*v1*v3+2*v1*v2 ", 1),
            ("3+v2+4*v1+v1*v3^2", 1),
        ];

        let mut res = res
            .iter()
            .map(|(f, p)| {
                (
                    parse!(f)
                        .expand()
                        .to_polynomial(&Z, poly.variables().clone()),
                    *p,
                )
            })
            .collect::<Vec<_>>();

        res.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut r = poly.factor();
        r.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        assert_eq!(r, res);
    }

    #[test]
    fn quadratic_discriminant_square_root_uses_even_multiplicities() {
        let discriminant = parse!("36*(x+1)^4*(y+2)^6")
            .expand()
            .to_polynomial::<_, u8>(&Z, None);
        let ExactPolynomialSquareRoot::Root(root) =
            discriminant.square_root_from_square_free_decomposition()
        else {
            panic!("the discriminant is an exact square");
        };

        assert_eq!(&root * &root, discriminant);

        let not_a_square = parse!("36*(x+1)^3*(y+2)^6")
            .expand()
            .to_polynomial::<_, u8>(&Z, None);
        assert!(matches!(
            not_a_square.square_root_from_square_free_decomposition(),
            ExactPolynomialSquareRoot::NotSquare
        ));
    }

    #[test]
    fn quadratic_factorization_handles_constant_discriminant() {
        let vars = Some(Arc::new(vec![
            symbol!("x").into(),
            symbol!("y").into(),
            symbol!("z").into(),
        ]));
        let poly = parse!("(x+y+z)*(x+y+z+1)")
            .expand()
            .to_polynomial::<_, u8>(&Z, vars);
        let QuadraticFactorization::Split(factors) = poly.factor_quadratic_variable(0) else {
            panic!("the quadratic must split");
        };

        assert_eq!(&factors[0] * &factors[1], poly);
    }

    #[test]
    fn quadratic_factorization_handles_repeated_discriminant_root_factor() {
        let vars = Some(Arc::new(vec![symbol!("x").into(), symbol!("y").into()]));
        let poly = parse!("(y+x^2)*(y-x^2)")
            .expand()
            .to_polynomial::<_, u8>(&Z, vars);
        let QuadraticFactorization::Split(factors) = poly.factor_quadratic_variable(1) else {
            panic!("the quadratic must split");
        };

        assert_eq!(&factors[0] * &factors[1], poly);
    }

    #[test]
    fn quadratic_factorization_reconstructs_nonunit_leading_coefficients() {
        let vars = Some(Arc::new(vec![
            symbol!("x").into(),
            symbol!("u").into(),
            symbol!("v").into(),
            symbol!("w").into(),
        ]));
        let poly = parse!("((u+1)*x+v+w+1)*((v+2)*x+u+w+3)")
            .expand()
            .to_polynomial::<_, u8>(&Z, vars);
        let QuadraticFactorization::Split(factors) = poly.factor_quadratic_variable(0) else {
            panic!("the quadratic must split");
        };

        assert_eq!(&factors[0] * &factors[1], poly);
        assert_eq!(poly.factor().len(), 2);
    }

    #[test]
    fn quadratic_factorization_rejects_nonsquare_discriminant() {
        let vars = Some(Arc::new(vec![
            symbol!("x").into(),
            symbol!("y").into(),
            symbol!("z").into(),
        ]));
        let poly = parse!("x^2+y^2+z^2+1").to_polynomial::<_, u8>(&Z, vars);

        assert!(matches!(
            poly.factor_quadratic_variable(0),
            QuadraticFactorization::Irreducible
        ));
        assert_eq!(poly.factor(), vec![(poly.clone(), 1)]);
        assert!(poly.is_irreducible());
    }

    #[test]
    fn wang_lcoeff_reconstruction_accepts_unlucky_content_sample() {
        let vars = Some(Arc::new(vec![
            symbol!("m").into(),
            symbol!("u").into(),
            symbol!("v").into(),
        ]));
        let poly = parse!("(3*m-u)*(5*m-v)")
            .expand()
            .to_polynomial::<_, u8>(&Z, vars);
        let sample_points = vec![(1, Integer::from(3)), (2, Integer::from(10))];
        let univariate_factors = ["m-1", "m-2"]
            .iter()
            .map(|f| parse!(f).to_polynomial(&Z, poly.variables().clone()))
            .collect::<Vec<_>>();
        let lcoeff = poly.univariate_lcoeff(0);
        let lcoeff_factorization = lcoeff.factor();

        assert!(
            poly.reconstruct_lcoeffs_from_univariate_sample(
                &lcoeff,
                &lcoeff_factorization,
                &univariate_factors,
                &sample_points,
                &Integer::from(1),
            )
            .is_none()
        );

        let reconstructed = poly
            .reconstruct_lcoeffs_from_univariate_sample(
                &lcoeff,
                &lcoeff_factorization,
                &univariate_factors,
                &sample_points,
                &Integer::from(15),
            )
            .unwrap();
        assert_eq!(reconstructed, vec![poly.one(), poly.one()]);
    }

    #[test]
    fn wang_lcoeff_reconstruction_rejects_scalar_factor_collisions() {
        let vars = Some(Arc::new(vec![symbol!("m").into(), symbol!("x").into()]));
        let poly = parse!("(3*m*x+1)*(5*m*x+1)")
            .expand()
            .to_polynomial::<_, u8>(&Z, vars);
        let lcoeff = poly.univariate_lcoeff(0);
        let lcoeff_factorization = lcoeff.factor();
        let lcoeff_content = lcoeff.content().abs();

        let colliding_sample = vec![(1, Integer::from(3))];
        let colliding_factors = ["9*m+1", "15*m+1"]
            .iter()
            .map(|f| parse!(f).to_polynomial(&Z, poly.variables().clone()))
            .collect::<Vec<_>>();
        assert!(
            !MultivariatePolynomial::lcoeff_sample_supports_wang_reconstruction(
                &lcoeff_factorization,
                &lcoeff_content,
                &colliding_sample,
            )
        );
        assert!(
            poly.reconstruct_lcoeffs_from_univariate_sample(
                &lcoeff,
                &lcoeff_factorization,
                &colliding_factors,
                &colliding_sample,
                &Integer::one(),
            )
            .is_none()
        );

        let coprime_sample = vec![(1, Integer::from(2))];
        let coprime_factors = ["6*m+1", "10*m+1"]
            .iter()
            .map(|f| parse!(f).to_polynomial(&Z, poly.variables().clone()))
            .collect::<Vec<_>>();
        assert!(
            MultivariatePolynomial::lcoeff_sample_supports_wang_reconstruction(
                &lcoeff_factorization,
                &lcoeff_content,
                &coprime_sample,
            )
        );
        let reconstructed = poly
            .reconstruct_lcoeffs_from_univariate_sample(
                &lcoeff,
                &lcoeff_factorization,
                &coprime_factors,
                &coprime_sample,
                &Integer::one(),
            )
            .unwrap();
        let expected = ["3*x", "5*x"]
            .iter()
            .map(|f| parse!(f).to_polynomial(&Z, poly.variables().clone()))
            .collect::<Vec<_>>();
        assert_eq!(reconstructed, expected);
    }

    #[test]
    fn wang_lcoeff_sampling_allows_nonmonomial_fixed_divisors() {
        let vars = Some(Arc::new(vec![symbol!("y").into()]));
        let lcoeff = parse!("2*(y^2-y+2)")
            .expand()
            .to_polynomial::<_, u8>(&Z, vars);
        let lcoeff_factorization = lcoeff.factor();

        assert!(
            MultivariatePolynomial::lcoeff_sample_supports_wang_reconstruction(
                &lcoeff_factorization,
                &lcoeff.content().abs(),
                &[(0, Integer::from(3))],
            )
        );
    }

    #[test]
    fn wang_univariate_sample_starts_with_coprime_prime_coordinates() {
        let vars = Some(Arc::new(vec![
            symbol!("m").into(),
            symbol!("x1").into(),
            symbol!("x2").into(),
            symbol!("x3").into(),
            symbol!("x4").into(),
            symbol!("x5").into(),
            symbol!("x6").into(),
            symbol!("x7").into(),
        ]));
        let poly = parse!("(6*m*x1*x2*x3*x4*x5*x6*x7+1)*(m+1)")
            .expand()
            .to_polynomial::<_, u8>(&Z, vars);
        let mut order = (0..8).collect::<Vec<_>>();

        // The leading-coefficient content is divisible by 2 and 3, so the
        // deterministic coordinates start with the next distinct primes.
        let (factors, true_lcoeffs, sample_points, _, image, content) = poly
            .find_univariate_sample(&mut order, 10, Some(2))
            .unwrap();

        let expected_points = [5i64, 7, 11, 13, 17, 19, 23]
            .into_iter()
            .map(Integer::from)
            .collect::<Vec<_>>();
        assert_eq!(
            sample_points
                .iter()
                .map(|(_, point)| point.clone())
                .collect::<Vec<_>>(),
            expected_points
        );
        assert_eq!(factors.len(), 2);

        let mut repeated_image = poly.clone();
        for (var, point) in sample_points.iter().rev() {
            repeated_image = repeated_image.replace(*var, point);
        }
        assert_eq!(image, repeated_image);

        let reconstructed_lcoeff = true_lcoeffs
            .iter()
            .fold(poly.one(), |product, lcoeff| &product * lcoeff)
            .mul_coeff(content);
        assert_eq!(reconstructed_lcoeff, poly.univariate_lcoeff(order[0]));
    }

    #[test]
    fn wang_univariate_sampling_retries_prime_block_after_specialization_content() {
        let vars = Some(Arc::new(vec![symbol!("m").into(), symbol!("x").into()]));
        let poly = parse!("(x*m+x+2)*(m+1)")
            .expand()
            .to_polynomial::<_, u8>(&Z, vars);
        let mut order = vec![0, 1];

        let (factors, _, sample_points, _, _, content) = poly
            .find_univariate_sample(&mut order, 10, Some(2))
            .unwrap();

        assert_eq!(sample_points, vec![(1, Integer::from(3))]);
        assert!(content.is_one());
        assert_eq!(factors.len(), 2);
    }

    #[test]
    fn wang_univariate_retry_relaxes_stale_factor_bound() {
        let vars = Some(Arc::new(vec![
            symbol!("m").into(),
            symbol!("x1").into(),
            symbol!("x2").into(),
            symbol!("x3").into(),
            symbol!("x4").into(),
            symbol!("x5").into(),
            symbol!("x6").into(),
            symbol!("x7").into(),
        ]));
        let poly =
            parse!("(m*x1*x2*x3*x4*x5*x6*x7+1+x1+x2+x3+x4+x5+x6+x7)*(m+1+x1+x2+x3+x4+x5+x6+x7)")
                .expand()
                .to_polynomial::<_, u8>(&Z, vars);
        let mut order = (0..8).collect::<Vec<_>>();
        assert!(!poly.integer_factor_start_auto_decision(&order).0);

        let factors = poly.multivariate_factorization(&mut order, 10, Some(0));
        let product = factors
            .iter()
            .fold(poly.one(), |product, factor| &product * factor);
        assert_eq!(factors.len(), 2);
        assert_eq!(product, poly);
    }

    #[test]
    fn integer_factor_auto_selects_bivariate_start_below_density_threshold() {
        let vars = Some(Arc::new(vec![
            symbol!("m").into(),
            symbol!("n").into(),
            symbol!("z").into(),
        ]));
        let poly = parse!("(1+m)*(1+n)*(1+z+z^2+z^3)")
            .expand()
            .to_polynomial::<_, u8>(&Z, vars);
        let order = [0, 1, 2];

        let (use_bivariate, density) = poly.integer_factor_start_auto_decision(&order);
        assert_eq!(density, 4.0);
        assert!(use_bivariate);
    }

    const WANG_RESIDUAL_CONTENT_FACTORS: [&str; 4] = [
        "3*m-u-w-v",
        "128*m^6-128*m^5*u-352*m^5*w-112*m^5*v+40*m^4*u^2+232*m^4*u*w+100*m^4*u*v+272*m^4*w^2+432*m^4*w*v+32*m^4*v^2-4*m^3*u^3-30*m^3*u^2*w-22*m^3*u^2*v-134*m^3*u*w^2-276*m^3*u*w*v-25*m^3*u*v^2-80*m^3*w^3-266*m^3*w^2*v-210*m^3*w*v^2-3*m^3*v^3-2*m^2*u^3*w+m^2*u^3*v+12*m^2*u^2*w^2+33*m^2*u^2*w*v+3*m^2*u^2*v^2+22*m^2*u*w^3+126*m^2*u*w^2*v+123*m^2*u*w*v^2+2*m^2*u*v^3+8*m^2*w^4+50*m^2*w^3*v+89*m^2*w^2*v^2+47*m^2*w*v^3+m*u^3*w*v-8*m*u^2*w^2*v-11*m*u^2*w*v^2-13*m*u*w^3*v-38*m*u*w^2*v^2-25*m*u*w*v^3-2*m*w^4*v-8*m*w^3*v^2-10*m*w^2*v^3-4*m*w*v^4+u^2*w^2*v^2+u^2*w*v^3+2*u*w^3*v^2+4*u*w^2*v^3+2*u*w*v^4",
        "128*m^4-80*m^3*u-80*m^3*w-96*m^3*v+12*m^2*u^2+52*m^2*u*v+12*m^2*w^2+52*m^2*w*v+24*m^2*v^2+9*m*u^2*w-10*m*u^2*v+9*m*u*w^2+4*m*u*w*v-12*m*u*v^2-10*m*w^2*v-12*m*w*v^2-2*m*v^3+u^3*v-3*u^2*w*v+2*u^2*v^2-3*u*w^2*v-2*u*w*v^2+u*v^3+w^3*v+2*w^2*v^2+w*v^3",
        "16*m^3-8*m^2*u-8*m^2*w-20*m^2*v+m*u^2+2*m*u*w+6*m*u*v+m*w^2+6*m*w*v+8*m*v^2-u*w*v-u*v^2-w*v^2-v^3",
    ];

    fn wang_residual_content_input() -> String {
        WANG_RESIDUAL_CONTENT_FACTORS
            .iter()
            .map(|f| format!("({f})"))
            .collect::<Vec<_>>()
            .join("*")
    }

    #[test]
    fn wang_univariate_lcoeff_keeps_residual_content_separate() {
        let vars = Some(Arc::new(vec![
            symbol!("m").into(),
            symbol!("u").into(),
            symbol!("w").into(),
            symbol!("v").into(),
        ]));
        let poly = parse!(wang_residual_content_input())
            .expand()
            .to_polynomial::<_, u16>(&Z, vars);
        let mut order = vec![0, 3, 2, 1];
        let (_, lcoeff_divs, _, _, _, univariate_content) =
            poly.find_univariate_sample(&mut order, 10, None).unwrap();
        let lcoeff_div_product = lcoeff_divs
            .iter()
            .fold(poly.one(), |product, lcoeff| &product * lcoeff);
        let actual_lcoeff = poly.univariate_lcoeff(order[0]);

        assert!(!univariate_content.is_one());
        assert_ne!(lcoeff_div_product, actual_lcoeff);
        assert_eq!(
            lcoeff_div_product.mul_coeff(univariate_content),
            actual_lcoeff
        );
    }

    #[test]
    fn factor_multivariate_wang_residual_content() {
        let vars = Some(Arc::new(vec![
            symbol!("m").into(),
            symbol!("u").into(),
            symbol!("w").into(),
            symbol!("v").into(),
        ]));
        let poly = parse!(wang_residual_content_input())
            .expand()
            .to_polynomial::<_, u16>(&Z, vars);

        let mut res = WANG_RESIDUAL_CONTENT_FACTORS
            .iter()
            .map(|f| {
                (
                    parse!(f)
                        .expand()
                        .to_polynomial(&Z, poly.variables().clone()),
                    1,
                )
            })
            .collect::<Vec<_>>();
        res.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));

        let mut r = poly.factor();
        r.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        assert_eq!(r, res);
    }

    #[test]
    fn factor_overall_minus() {
        let poly = parse!("-v1*v3^2-v1*v2*v3^2").to_polynomial::<_, u8>(
            &Z,
            Some(Arc::new(vec![
                symbol!("v1").into(),
                symbol!("v2").into(),
                symbol!("v3").into(),
            ])),
        );

        let res = [("-1", 1), ("v3", 2), ("1+v2", 1), ("v1", 1)];

        let mut res = res
            .iter()
            .map(|(f, p)| {
                (
                    parse!(f)
                        .expand()
                        .to_polynomial(&Z, poly.variables().clone()),
                    *p,
                )
            })
            .collect::<Vec<_>>();

        res.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut r = poly.factor();
        r.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        assert_eq!(r, res);
    }

    #[test]
    fn factor_multivariate_2() {
        let poly = parse!("v2^2*v3-v1*v2*v3+v1*v2*v3^2+v1*v2^2-v1^2*v3^2+v1^2*v2*v3")
            .to_polynomial::<_, u8>(
                &Z,
                Some(Arc::new(vec![
                    symbol!("v1").into(),
                    symbol!("v2").into(),
                    symbol!("v3").into(),
                ])),
            );

        let res = [("v2+v1*v3", 1), ("v2*v3-v1*v3+v1*v2", 1)];

        let mut res = res
            .iter()
            .map(|(f, p)| {
                (
                    parse!(f)
                        .expand()
                        .to_polynomial(&Z, poly.variables().clone()),
                    *p,
                )
            })
            .collect::<Vec<_>>();

        res.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut r = poly.factor();
        r.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        assert_eq!(r, res);
    }

    #[test]
    fn galois_upgrade() {
        let a =
            parse!("x^7(y^5+y^4+y^3+y^2)+x^5(y^3+y)+x^4(y^4+y)+x^3(y^2+y)+x^2y+x*y^2+x*y+x+y+1")
                .to_polynomial::<_, u8>(&Z2, None);

        assert_eq!(a.factor().len(), 2)
    }

    #[test]
    fn algebraic_extension() {
        let a = parse!("z^4+z^3+(2+a-a^2)z^2+(1+a^2-2a^3)z-2").to_polynomial::<_, u8>(&Q, None);
        let f = parse!("a^4-3").to_polynomial::<_, u16>(&Q, None);
        let f = AlgebraicExtension::new(f);

        let mut factors = a.to_number_field(&f).factor();

        let f1 = parse!("(1-a^2)+(1-a)*z+z^2")
            .to_polynomial::<_, u8>(&Q, a.get_vars().clone())
            .to_number_field(&f);
        let f2 = parse!("(1+a^2)+(a)*z+z^2")
            .to_polynomial::<_, u8>(&Q, a.get_vars().clone())
            .to_number_field(&f);

        factors.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));

        assert_eq!(factors, vec![(f1, 1), (f2, 1)])
    }
}
