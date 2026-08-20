//! Certified isolation, refinement, ordering, and caching of exact polynomial roots.
//!
//! The public types in this module are re-exported from [`super`]. Root-cache
//! storage and certification machinery remain private implementation details.

#![warn(missing_docs)]

use super::*;

/// Exact complex-rational coefficient field used by root isolation.
type ExactComplexField = FloatField<Complex<Rational>>;

/// A dense univariate polynomial with exact complex-rational coefficients.
pub type ExactComplexPolynomial = UnivariatePolynomial<FloatField<Complex<Rational>>>;

/// An exact rational disk in the complex plane.
#[derive(Clone, Debug)]
pub struct ComplexDisk {
    center: Complex<Rational>,
    radius: Rational,
}

/// Proven location of an isolated root relative to the coordinate axes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RootLocation {
    /// The root lies on neither coordinate axis.
    Complex,
    /// The root lies on the real axis but is not zero.
    Real,
    /// The root lies on the imaginary axis but is not zero.
    Imaginary,
    /// The root is zero and therefore lies on both coordinate axes.
    Zero,
}

/// An axis of the complex plane.
#[derive(Clone, Copy, Eq, PartialEq)]
enum CoordinateAxis {
    Real,
    Imaginary,
}

/// An isolated root in the complex plane, together with its exact defining
/// polynomial, certified enclosure, and any proven axis location.
#[derive(Clone, Debug)]
pub struct IsolatedRoot {
    poly: Arc<ExactComplexPolynomial>,
    index: usize,
    enclosure: ComplexDisk,
    location: Option<RootLocation>,
}

/// An enclosure and location of a root.
#[derive(Clone, Debug)]
struct CachedRoot {
    enclosure: ComplexDisk,
    location: Option<RootLocation>,
}

/// A root and its multiplicity in a polynomial.
#[derive(Clone)]
struct RootMultisetEntry {
    poly: Arc<ExactComplexPolynomial>,
    index: usize,
    multiplicity: usize,
}

/// A canonically ordered collection of roots, potentially from multiple polynomials, and their multiplicity.
type RootMultiset = Vec<RootMultisetEntry>;

/// Intervals over which a real root is known to exist.
#[derive(Clone)]
struct RealProjection {
    poly: Arc<UnivariatePolynomial<Q>>,
    intervals: Vec<(Rational, Rational)>,
}

/// A projected real root, consisting of a polynomial and an interval.
struct ProjectedRealRoot {
    poly: Arc<UnivariatePolynomial<Q>>,
    interval: (Rational, Rational),
}

/// Process-wide storage for reusable root certificates and root multisets.
struct RootCache {
    rational: PolynomialCache<Rational>,
    complex: PolynomialCache<Complex<Rational>>,
}

/// Cached root data keyed by a polynomial.
struct PolynomialCache<C> {
    roots: RwLock<HashMap<Vec<C>, Arc<RootSetSlot>>>,
    root_multisets: RwLock<HashMap<Vec<C>, Arc<RootMultisetSlot>>>,
}

/// A slot for a root set, used to lazily initialize the root set.
type RootSetSlot = OnceLock<RwLock<Vec<CachedRoot>>>;
/// A slot for a root multiset, used to lazily initialize the root multiset.
type RootMultisetSlot = OnceLock<RootMultiset>;

impl<C: Clone + Eq + std::hash::Hash> PolynomialCache<C> {
    /// Create an empty cache for one coefficient type.
    fn new() -> Self {
        Self {
            roots: RwLock::new(HashMap::new()),
            root_multisets: RwLock::new(HashMap::new()),
        }
    }

    /// Return the lazy root-set slot keyed by `polynomial`'s coefficients.
    fn root_slot<R: Ring<Element = C>>(
        &self,
        polynomial: &UnivariatePolynomial<R>,
    ) -> Arc<RootSetSlot> {
        let coefficients = polynomial.coefficients();
        if let Some(entry) = self.roots.read().unwrap().get(coefficients).cloned() {
            return entry;
        }

        let mut roots = self.roots.write().unwrap();
        if let Some(entry) = roots.get(coefficients).cloned() {
            return entry;
        }

        let entry = Arc::new(RootSetSlot::new());
        roots.insert(coefficients.to_vec(), entry.clone());
        entry
    }

    /// Return the lazy root-multiset slot keyed by `polynomial`'s coefficients.
    fn root_multiset_slot<R: Ring<Element = C>>(
        &self,
        polynomial: &UnivariatePolynomial<R>,
    ) -> Arc<RootMultisetSlot> {
        let coefficients = polynomial.coefficients();
        if let Some(entry) = self
            .root_multisets
            .read()
            .unwrap()
            .get(coefficients)
            .cloned()
        {
            return entry;
        }

        let mut root_multisets = self.root_multisets.write().unwrap();
        if let Some(entry) = root_multisets.get(coefficients).cloned() {
            return entry;
        }

        let entry = Arc::new(RootMultisetSlot::new());
        root_multisets.insert(coefficients.to_vec(), entry.clone());
        entry
    }
}

impl RootCache {
    /// Create empty rational and exact-complex polynomial caches.
    fn new() -> Self {
        Self {
            rational: PolynomialCache::new(),
            complex: PolynomialCache::new(),
        }
    }

    /// Convert roots into canonically ordered cache records.
    fn cache_states(mut roots: Vec<IsolatedRoot>) -> Vec<CachedRoot> {
        UnivariatePolynomial::<Q>::sort_roots_canonically(&mut roots);
        roots
            .into_iter()
            .map(|root| CachedRoot {
                enclosure: root.enclosure,
                location: root.location,
            })
            .collect()
    }

    /// Isolate a square-free defining polynomial to the requested radius.
    fn isolate_roots(
        poly: &ExactComplexPolynomial,
        target_radius: Option<&Rational>,
    ) -> Vec<IsolatedRoot> {
        if let Some(rational) = poly.try_map_to_rational() {
            rational.isolate_square_free_roots(target_radius)
        } else {
            poly.isolate_square_free_roots(target_radius)
        }
    }

    /// Return the initialized root-set slot for a defining polynomial.
    fn root_set(&self, poly: &ExactComplexPolynomial) -> Arc<RootSetSlot> {
        let entry = if let Some(rational) = poly.try_map_to_rational() {
            self.rational.root_slot(&rational)
        } else {
            self.complex.root_slot(poly)
        };
        entry.get_or_init(|| RwLock::new(Self::cache_states(Self::isolate_roots(poly, None))));
        entry
    }

    /// Copy one root certificate from the cache into an isolated-root value.
    fn root_snapshot(&self, poly: Arc<ExactComplexPolynomial>, index: usize) -> IsolatedRoot {
        let entry = self.root_set(&poly);
        let roots = entry.get().unwrap().read().unwrap();
        let state = roots
            .get(index)
            .unwrap_or_else(|| panic!("root index {index} is out of bounds for {poly}"));
        IsolatedRoot {
            poly,
            index,
            enclosure: state.enclosure.clone(),
            location: state.location,
        }
    }

    /// Copy all root certificates for a defining polynomial from the cache.
    fn root_snapshots(&self, poly: Arc<ExactComplexPolynomial>) -> Vec<IsolatedRoot> {
        let entry = self.root_set(&poly);
        let states = entry.get().unwrap().read().unwrap();
        states
            .iter()
            .enumerate()
            .map(|(index, state)| IsolatedRoot {
                poly: poly.clone(),
                index,
                enclosure: state.enclosure.clone(),
                location: state.location,
            })
            .collect()
    }

    /// Combine roots of square-free factors into one canonical multiset.
    fn build_root_multiset(
        &self,
        factors: impl IntoIterator<Item = (Arc<ExactComplexPolynomial>, usize)>,
    ) -> RootMultiset {
        let mut roots = Vec::new();
        let mut multiplicities = HashMap::new();
        for (poly, multiplicity) in factors {
            multiplicities.insert(poly.coefficients.clone(), multiplicity);
            roots.extend(self.root_snapshots(poly));
        }

        UnivariatePolynomial::<Q>::separate_isolated_roots(&mut roots);
        UnivariatePolynomial::<Q>::sort_roots_canonically(&mut roots);
        for root in &roots {
            self.merge_root_certificate(root);
        }

        roots
            .into_iter()
            .map(|root| RootMultisetEntry {
                multiplicity: multiplicities[&root.poly.coefficients],
                poly: root.poly,
                index: root.index,
            })
            .collect()
    }

    /// Refines the root to the given tolerance.
    fn refine_root(&self, root: &mut IsolatedRoot, tolerance: &Rational) {
        let entry = self.root_set(&root.poly);
        let mut states = entry.get().unwrap().write().unwrap();
        let state = states.get(root.index).unwrap_or_else(|| {
            panic!(
                "root index {} is out of bounds for {}",
                root.index, root.poly
            )
        });
        root.enclosure = state.enclosure.clone();
        root.location = state.location;

        if !tolerance.is_zero()
            && root.enclosure.radius > *tolerance
            && !UnivariatePolynomial::<Q>::refine_root_to_tolerance(root, tolerance)
        {
            let replacement = Self::cache_states(Self::isolate_roots(&root.poly, Some(tolerance)));
            *states = replacement;
            let state = states.get(root.index).unwrap_or_else(|| {
                panic!(
                    "root index {} disappeared while refining {}",
                    root.index, root.poly
                )
            });
            root.enclosure = state.enclosure.clone();
            root.location = state.location;
            return;
        }

        states[root.index] = CachedRoot {
            enclosure: root.enclosure.clone(),
            location: root.location,
        };
    }

    /// Merges the root certificate from the given root into the root set if it is more precise than the current state.
    fn merge_root_certificate(&self, root: &IsolatedRoot) {
        let entry = self.root_set(&root.poly);
        let mut states = entry.get().unwrap().write().unwrap();
        let state = states.get_mut(root.index).unwrap_or_else(|| {
            panic!(
                "root index {} is out of bounds for {}",
                root.index, root.poly
            )
        });
        if root.enclosure.radius < state.enclosure.radius {
            state.enclosure = root.enclosure.clone();
        }
        if state.location.is_none() {
            state.location = root.location;
        }
    }

    /// Classifies the root based on the root set state.
    fn classify_root(&self, root: &mut IsolatedRoot) {
        let entry = self.root_set(&root.poly);
        let mut roots = {
            let states = entry.get().unwrap().read().unwrap();
            let state = states.get(root.index).unwrap_or_else(|| {
                panic!(
                    "root index {} is out of bounds for {}",
                    root.index, root.poly
                )
            });
            if state.location.is_some() {
                root.enclosure = state.enclosure.clone();
                root.location = state.location;
                return;
            }

            states
                .iter()
                .enumerate()
                .map(|(index, state)| IsolatedRoot {
                    poly: root.poly.clone(),
                    index,
                    enclosure: state.enclosure.clone(),
                    location: state.location,
                })
                .collect::<Vec<_>>()
        };

        root.poly.classify_root_locations(&mut roots);

        let mut states = entry.get().unwrap().write().unwrap();
        for resolved in roots {
            let state = &mut states[resolved.index];
            if state.location.is_none() {
                state.location = resolved.location;
            }
        }
        let state = &states[root.index];
        root.enclosure = state.enclosure.clone();
        root.location = state.location;
    }

    /// Materialize every distinct root and multiplicity in a cached multiset.
    fn roots_in_multiset(&self, multiset: &RootMultiset) -> Vec<(IsolatedRoot, usize)> {
        multiset
            .iter()
            .map(|entry| {
                let root = self.root_snapshot(entry.poly.clone(), entry.index);
                (root, entry.multiplicity)
            })
            .collect()
    }

    /// Resolve an index that counts multiplicity to its distinct cached root.
    fn root_in_multiset(&self, multiset: &RootMultiset, index: usize) -> Option<IsolatedRoot> {
        let mut seen = 0;
        for entry in multiset {
            if index < seen + entry.multiplicity {
                return Some(self.root_snapshot(entry.poly.clone(), entry.index));
            }
            seen += entry.multiplicity;
        }
        None
    }
}

/// Get the global root cache.
fn root_cache() -> &'static RootCache {
    static CACHE: LazyLock<RootCache> = LazyLock::new(RootCache::new);
    &CACHE
}

impl ComplexDisk {
    /// Return an upper bound for the complex norm used by disk estimates.
    fn norm_upper_bound(z: &Complex<Rational>) -> Rational {
        z.re.abs() + z.im.abs()
    }

    /// Return a lower bound for the complex norm used by disk separation.
    fn norm_lower_bound(z: &Complex<Rational>) -> Rational {
        z.re.abs().max(z.im.abs())
    }

    /// Return whether this disk is certified not to overlap `other`.
    pub fn is_disjoint(&self, other: &Self) -> bool {
        &self.radius + &other.radius < Self::norm_lower_bound(&(&self.center - &other.center))
    }

    /// Return the exact complex-rational center of the disk.
    pub fn center(&self) -> &Complex<Rational> {
        &self.center
    }

    /// Return the exact rational radius of the disk.
    pub fn radius(&self) -> &Rational {
        &self.radius
    }

    /// Convert this exact rational disk to a certified rectangular
    /// floating-point enclosure.
    pub fn to_ball(&self, precision: u32) -> ComplexBall {
        ComplexBall::from_rational_ball(&self.center, &self.radius, precision)
    }
}

impl RootLocation {
    /// Add a proven coordinate-axis membership to an existing location.
    fn with_axis(location: Option<Self>, axis: CoordinateAxis) -> Self {
        match (location, axis) {
            (Some(Self::Imaginary | Self::Zero), CoordinateAxis::Real)
            | (Some(Self::Real | Self::Zero), CoordinateAxis::Imaginary) => Self::Zero,
            (_, CoordinateAxis::Real) => Self::Real,
            (_, CoordinateAxis::Imaginary) => Self::Imaginary,
        }
    }
}

impl CoordinateAxis {
    /// Test whether an axis interval is certified to belong to `root`'s disk.
    fn contains_interval(self, interval: &(Rational, Rational), root: &IsolatedRoot) -> bool {
        let (center, distance_to_axis) = match self {
            Self::Real => (&root.enclosure.center.re, root.enclosure.center.im.abs()),
            Self::Imaginary => (&root.enclosure.center.im, root.enclosure.center.re.abs()),
        };
        let lower_distance = (&interval.0 - center).abs() + &distance_to_axis;
        let upper_distance = (&interval.1 - center).abs() + &distance_to_axis;
        if lower_distance <= root.enclosure.radius && upper_distance <= root.enclosure.radius {
            return true;
        }

        self == Self::Imaginary
            && distance_to_axis <= root.enclosure.radius
            && interval.0 <= center.clone() - &root.enclosure.radius
            && center.clone() + &root.enclosure.radius <= interval.1
    }
}

impl IsolatedRoot {
    /// Construct the absolute tolerance `2^-binary_precision`.
    fn absolute_tolerance(binary_precision: u32) -> Rational {
        Rational::from((
            Integer::one(),
            Integer::from(2).pow(binary_precision as u64),
        ))
    }

    /// Returns a reference to the polynomial defining this root.
    pub fn defining_polynomial(&self) -> &ExactComplexPolynomial {
        &self.poly
    }

    /// Returns the complex disk enclosure of this root.
    pub fn enclosure(&self) -> &ComplexDisk {
        &self.enclosure
    }

    /// Canonical index of this root within its defining polynomial.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Convert this root to the canonical expression-level `root` representation.
    pub fn to_atom(&self) -> Atom {
        let mut polynomial = self.poly.as_ref().clone().to_multivariate::<u16>();
        let variable = polynomial.get_vars_ref()[0].clone();
        let canonical_variable = PolyVariable::Symbol(root_var());
        if variable != canonical_variable {
            polynomial.rename_variable(&variable, &canonical_variable);
        }
        root().call((polynomial.to_expression(), self.index))
    }

    /// Refine this root to an absolute radius, update its cached enclosure,
    /// and return the updated root.
    pub fn refined(mut self, tolerance: &Rational) -> Self {
        root_cache().refine_root(&mut self, tolerance);
        self
    }

    /// Determine whether this root lies on the positive real axis. Complex,
    /// imaginary, and zero roots are not positive.
    pub fn is_positive_real(&mut self) -> bool {
        if self.classify_location() != RootLocation::Real {
            return false;
        }

        let mut binary_precision = 32u32;
        loop {
            if &self.enclosure.center.re - &self.enclosure.radius > Rational::zero() {
                return true;
            }
            if &self.enclosure.center.re + &self.enclosure.radius < Rational::zero() {
                return false;
            }

            let tolerance = Self::absolute_tolerance(binary_precision);
            root_cache().refine_root(self, &tolerance);
            binary_precision = binary_precision.saturating_mul(2);
        }
    }

    /// Resolve this root's exact relationship to the coordinate axes and
    /// update both this root and its defining-root cache entry.
    pub fn classify_location(&mut self) -> RootLocation {
        if self.location.is_none() {
            root_cache().classify_root(self);
        }
        self.location
            .expect("root location resolution must classify every root")
    }

    /// Convert the root's enclosure to a floating-point center.
    pub(crate) fn to_float_center(&self, binary_prec: u32) -> Complex<Float> {
        let mut center = Complex::new(
            self.enclosure.center.re.to_multi_prec_float(binary_prec),
            self.enclosure.center.im.to_multi_prec_float(binary_prec),
        );

        let field = FloatField::from_rep(Complex::new(
            Float::with_val(binary_prec, 1),
            Float::new(binary_prec),
        ));
        let poly = self.poly.map_coeff(
            |c| {
                Complex::new(
                    c.re.to_multi_prec_float(binary_prec),
                    c.im.to_multi_prec_float(binary_prec),
                )
            },
            field,
        );
        let derivative = poly.derivative();
        let tolerance = Rational::from((Integer::one(), Integer::from(2).pow(binary_prec as u64)))
            .to_multi_prec_float(binary_prec);
        let tolerance_squared = tolerance.clone() * tolerance;

        for _ in 0..32 {
            let derivative_at_center = derivative.evaluate(&center);
            if SingleFloat::is_zero(&derivative_at_center) {
                break;
            }

            let correction = poly.evaluate(&center) / derivative_at_center;
            if !correction.is_finite() {
                break;
            }

            center -= correction.clone();
            if correction.norm_squared() < tolerance_squared {
                break;
            }
        }

        center
    }
}

impl IsolatedRoot {
    /// Enclose this root or its image under an optional rational polynomial.
    fn transformed_enclosure(
        &self,
        polynomial: Option<&UnivariatePolynomial<RationalField>>,
        precision: u32,
    ) -> ComplexBall {
        let root_ball = self.enclosure().to_ball(precision);
        match polynomial {
            Some(polynomial) => polynomial.evaluate_complex_ball(&root_ball, precision),
            None => root_ball,
        }
    }
}

/// Failure to prove a relation between isolated roots within the refinement limit.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum RootCertificationError {
    /// Refinement did not isolate the requested number of transformed roots.
    UnexpectedMatchCount {
        /// Number of matches required by the caller.
        expected: usize,
        /// Number of possible matches after the final refinement.
        found: usize,
    },
    /// Refinement could not prove the sign of a real part.
    IndeterminateRealPart,
}

impl std::fmt::Display for RootCertificationError {
    /// Format a concise explanation of the failed certification.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnexpectedMatchCount { expected, found } => write!(
                f,
                "expected {expected} transformed root match(es), but certification left {found} candidate(s)"
            ),
            Self::IndeterminateRealPart => {
                f.write_str("could not certify the sign of the transformed root's real part")
            }
        }
    }
}

impl IsolatedRoot {
    /// Find candidates whose polynomial images may match this root's polynomial image.
    ///
    /// A missing polynomial denotes the identity map.
    pub(crate) fn matching_roots(
        &self,
        polynomial: Option<&UnivariatePolynomial<RationalField>>,
        candidates: &mut [IsolatedRoot],
        candidate_polynomial: Option<&UnivariatePolynomial<RationalField>>,
        expected_count: usize,
    ) -> Result<Vec<usize>, RootCertificationError> {
        let mut target = self.clone();
        let mut binary_precision = 32u32;
        let mut final_match_count = candidates.len();
        for _ in 0..10 {
            let target_ball = target.transformed_enclosure(polynomial, binary_precision);
            let matches = candidates
                .iter()
                .enumerate()
                .filter_map(|(index, candidate)| {
                    let value =
                        candidate.transformed_enclosure(candidate_polynomial, binary_precision);
                    (!value.is_disjoint(&target_ball)).then_some(index)
                })
                .collect::<Vec<_>>();

            final_match_count = matches.len();
            if final_match_count == expected_count {
                return Ok(matches);
            }

            let tolerance = Self::absolute_tolerance(binary_precision);
            target = target.refined(&tolerance);
            let candidates_to_refine: Vec<_> = if matches.is_empty() {
                (0..candidates.len()).collect()
            } else {
                matches
            };
            for index in candidates_to_refine {
                candidates[index] = candidates[index].clone().refined(&tolerance);
            }
            binary_precision = binary_precision.saturating_mul(2);
        }

        Err(RootCertificationError::UnexpectedMatchCount {
            expected: expected_count,
            found: final_match_count,
        })
    }
}

impl UnivariatePolynomial<RationalField> {
    /// Evaluate this rational polynomial over a certified complex enclosure.
    pub(crate) fn evaluate_complex_ball(&self, value: &ComplexBall, precision: u32) -> ComplexBall {
        let zero = RealBall::exact(Float::new(precision));
        let mut result = ComplexBall::new(zero.clone(), zero);
        for coefficient in self.coefficients.iter().rev() {
            let coefficient = RealBall::from_rational_bounds(coefficient, coefficient, precision);
            result = result * value + coefficient;
        }
        result
    }

    /// Determine whether this polynomial evaluated at `root` has a strictly
    /// positive real part.
    pub(crate) fn has_positive_real_part_at(
        &self,
        root: &IsolatedRoot,
    ) -> Result<bool, RootCertificationError> {
        let mut root = root.clone();
        let mut binary_precision = 32u32;
        for _ in 0..10 {
            let root_ball = root.enclosure().to_ball(binary_precision);
            let value = self.evaluate_complex_ball(&root_ball, binary_precision);
            if value.re.is_strictly_positive() {
                return Ok(true);
            }
            if value.re.is_strictly_negative() {
                return Ok(false);
            }

            root = root.refined(&IsolatedRoot::absolute_tolerance(binary_precision));
            binary_precision = binary_precision.saturating_mul(2);
        }

        Err(RootCertificationError::IndeterminateRealPart)
    }

    /// Choose the working tolerance for Aberth iteration at `num_prec` bits.
    fn aberth_tolerance(num_prec: u32) -> Float {
        let bits = num_prec.saturating_sub(8).max(1);
        Rational::from((Integer::one(), Integer::from(2).pow(bits as u64)))
            .to_multi_prec_float(num_prec)
    }

    /// Enclose an exact complex rational in a rectangular complex ball.
    fn exact_complex_to_ball(value: &Complex<Rational>, precision: u32) -> ComplexBall {
        ComplexBall::new(
            RealBall::from_rational_bounds(&value.re, &value.re, precision),
            RealBall::from_rational_bounds(&value.im, &value.im, precision),
        )
    }

    /// Return a certified enclosure of the complex modulus.
    ///
    /// The componentwise maximum is a lower bound for the Euclidean norm and
    /// the componentwise sum is an upper bound. These slightly loose bounds
    /// avoid non-algebraic square roots in the Rouché test.
    fn complex_ball_modulus_bounds(value: &ComplexBall) -> RealBall {
        let re_abs = value.re.norm();
        let im_abs = value.im.norm();
        let re_lower = re_abs.lower_bound();
        let im_lower = im_abs.lower_bound();
        let lower = if re_lower >= im_lower {
            re_lower
        } else {
            im_lower
        };
        let upper = (re_abs + &im_abs).upper_bound();
        RealBall::from_bounds(lower, upper)
    }

    /// Compute certified ball enclosures of the coefficients of
    /// `poly(x + center)` using an in-place quadratic Taylor shift.
    fn shift_var_complex_ball(
        poly: &ExactComplexPolynomial,
        center: &Complex<Rational>,
        precision: u32,
    ) -> Vec<ComplexBall> {
        let center = Self::exact_complex_to_ball(center, precision);
        let mut shifted = poly
            .coefficients
            .iter()
            .map(|coefficient| Self::exact_complex_to_ball(coefficient, precision))
            .collect::<Vec<_>>();

        for i in (0..shifted.len().saturating_sub(1)).rev() {
            for j in i..shifted.len() - 1 {
                shifted[j] = shifted[j].clone() + &shifted[j + 1] * &center;
            }
        }
        shifted
    }

    /// Test Rouché's strict inequality using directed-rounding ball
    /// arithmetic. A false result is inconclusive and may use an exact
    /// rational fallback.
    fn shifted_ball_contains_one_root(
        shifted: &[ComplexBall],
        radius: &Rational,
        precision: u32,
    ) -> bool {
        if radius.is_zero() {
            return false;
        }

        let Some(linear) = shifted.get(1) else {
            return false;
        };

        let radius = RealBall::from_rational_bounds(radius, radius, precision);
        let first_lower_bound = Self::complex_ball_modulus_bounds(linear) * &radius;
        let mut upper_bound = shifted
            .first()
            .map(Self::complex_ball_modulus_bounds)
            .unwrap_or_else(|| RealBall::exact(Float::new(precision)));
        let mut radius_power = radius.clone();
        for coefficient in shifted.iter().skip(2) {
            radius_power *= &radius;
            upper_bound += Self::complex_ball_modulus_bounds(coefficient) * &radius_power;
        }

        first_lower_bound.lower_bound() > upper_bound.upper_bound()
    }

    /// Test Rouché's strict inequality on an already shifted exact
    /// polynomial.
    fn shifted_disk_contains_one_root(shifted: &ExactComplexPolynomial, radius: &Rational) -> bool {
        if radius.is_zero() {
            return false;
        }

        let Some(linear) = shifted.coefficients.get(1) else {
            return false;
        };

        let mut eval_higher_powers = Rational::zero();
        for (pow, c) in shifted.coefficients.iter().enumerate().skip(2) {
            eval_higher_powers += radius.pow(pow as u64) * ComplexDisk::norm_upper_bound(c);
        }

        let first_lower_bound = ComplexDisk::norm_lower_bound(linear) * radius;
        let const_upper = shifted
            .coefficients
            .first()
            .map(ComplexDisk::norm_upper_bound)
            .unwrap_or_else(Rational::zero);

        first_lower_bound > const_upper + eval_higher_powers
    }

    /// Test a disk while reusing Taylor shifts across radius retries.
    fn disk_contains_one_root_with_shift_cache(
        poly: &ExactComplexPolynomial,
        center: &Complex<Rational>,
        radius: &Rational,
        shifted_ball: &[ComplexBall],
        exact_shifted: &mut Option<ExactComplexPolynomial>,
    ) -> bool {
        if Self::shifted_ball_contains_one_root(shifted_ball, radius, 128) {
            return true;
        }

        let shifted = exact_shifted.get_or_insert_with(|| poly.shift_var(center));
        Self::shifted_disk_contains_one_root(shifted, radius)
    }

    /// Test if a disk contains a single root of the polynomial using Rouché's theorem.
    fn disk_contains_one_root(
        poly: &ExactComplexPolynomial,
        center: &Complex<Rational>,
        radius: &Rational,
    ) -> bool {
        let shifted_ball = Self::shift_var_complex_ball(poly, center, 128);
        let mut exact_shifted = None;
        Self::disk_contains_one_root_with_shift_cache(
            poly,
            center,
            radius,
            &shifted_ball,
            &mut exact_shifted,
        )
    }

    /// Choose an initially disjoint candidate radius around one approximate root.
    fn initial_disk_radius(
        centers: &[Complex<Rational>],
        root_index: usize,
        target_radius: Option<&Rational>,
    ) -> Option<Rational> {
        let mut radius = None;

        for (other_index, other_center) in centers.iter().enumerate() {
            if root_index == other_index {
                continue;
            }

            let distance = ComplexDisk::norm_lower_bound(&(&centers[root_index] - other_center));
            radius = Some(match radius {
                Some(r) if r < distance => r,
                _ => distance,
            });
        }

        let mut radius = radius
            .map(|r| r / Rational::from(4))
            .or_else(|| target_radius.cloned())
            .unwrap_or_else(Rational::one);

        if let Some(target_radius) = target_radius {
            if !target_radius.is_zero() && target_radius < &radius {
                radius = target_radius.clone();
            }
        }

        if radius.is_zero() { None } else { Some(radius) }
    }

    /// Gets the `index`-th root of the polynomial. Fails when `index` is out of bounds.
    pub fn root(&self, index: usize) -> Option<IsolatedRoot> {
        if index >= self.degree() {
            return None;
        }

        let cache = root_cache();
        let entry = cache.rational.root_multiset_slot(self);
        let multiset = entry.get_or_init(|| self.build_root_multiset());
        cache.root_in_multiset(multiset, index)
    }

    /// Isolate the distinct complex roots of the polynomial. The result contains
    /// canonically sorted `(root, multiplicity)` pairs. Every root enclosure is a
    /// rational ball containing exactly one root of its defining polynomial.
    pub fn isolate_roots(&self) -> Vec<(IsolatedRoot, usize)> {
        let cache = root_cache();
        let entry = cache.rational.root_multiset_slot(self);
        let multiset = entry.get_or_init(|| self.build_root_multiset());
        cache.roots_in_multiset(multiset)
    }

    /// Isolate the distinct real roots of the polynomial. Resolving whether a
    /// root lies on the real axis may refine its cached enclosure.
    pub fn isolate_real_roots(&self) -> Vec<(IsolatedRoot, usize)> {
        self.isolate_roots()
            .into_iter()
            .filter_map(|(mut root, multiplicity)| {
                let location = root.classify_location();
                matches!(location, RootLocation::Real | RootLocation::Zero)
                    .then_some((root, multiplicity))
            })
            .collect()
    }

    /// Factor this polynomial and build its canonical root multiset.
    fn build_root_multiset(&self) -> RootMultiset {
        let complex_field = FloatField::from_rep(Complex::from(Rational::one()));
        let factors = self
            .clone()
            .to_multivariate::<u16>()
            .factor()
            .into_iter()
            .filter(|(factor, _)| !factor.is_constant())
            .map(|(factor, multiplicity)| {
                let defining_poly = Arc::new(factor.to_univariate_from_univariate(0).map_coeff(
                    |coefficient| Complex::from(coefficient.clone()),
                    complex_field.clone(),
                ));
                (defining_poly, multiplicity)
            });
        root_cache().build_root_multiset(factors)
    }

    /// Sort roots by certified real part and then imaginary part.
    fn sort_roots_canonically(roots: &mut [IsolatedRoot]) {
        // Establishing the order may strengthen the supplied root snapshots.
        // Cache owners must retain those stronger certificates after this call.
        Self::separate_real_projections(roots);
        let known_equal_real_parts = Self::known_equal_real_parts(roots);
        let needs_projection = Self::roots_needing_real_projection(roots, &known_equal_real_parts);

        let mut projection_cache = HashMap::new();
        let projected_roots = roots
            .iter_mut()
            .zip(needs_projection)
            .map(|(root, needs_projection)| {
                if !needs_projection {
                    return None;
                }

                let poly = &root.poly;
                let projection = projection_cache
                    .entry(poly.coefficients.clone())
                    .or_insert_with(|| {
                        Self::real_projection_polynomial(poly).map(|projection| {
                            let intervals = projection
                                .isolate_real_root_intervals()
                                .into_iter()
                                .map(|(lower, upper, _)| (lower, upper))
                                .collect();
                            RealProjection {
                                poly: Arc::new(projection),
                                intervals,
                            }
                        })
                    })
                    .as_ref()?;

                Self::projected_real_root(root, projection)
            })
            .collect::<Vec<_>>();

        let mut order = (0..roots.len()).collect::<Vec<_>>();
        order.sort_by(|&a, &b| {
            Self::cmp_complex_roots_canonical_with_projected(
                &roots[a],
                &roots[b],
                projected_roots[a].as_ref(),
                projected_roots[b].as_ref(),
                known_equal_real_parts[a * roots.len() + b],
            )
        });

        let sorted = order
            .into_iter()
            .map(|index| roots[index].clone())
            .collect::<Vec<_>>();
        roots.clone_from_slice(&sorted);
    }

    /// Refine disks whose real projections overlap before exact comparison.
    fn separate_real_projections(roots: &mut [IsolatedRoot]) {
        // Use bounded-size approximate centers here. Repeated exact rational
        // Newton steps make numerator and denominator sizes grow explosively.
        // Most comparisons resolve at the first target; persistent overlaps
        // are handled exactly by the projection polynomial below.
        for target_radius_bits in [32, 64] {
            let known_equal_real_parts = Self::known_equal_real_parts(roots);
            let overlaps = Self::roots_needing_real_projection(roots, &known_equal_real_parts);
            if !overlaps.iter().any(|overlaps| *overlaps) {
                return;
            }

            let mut refined_any = false;
            for (root, overlaps) in roots.iter_mut().zip(overlaps) {
                if !overlaps {
                    continue;
                }
                let poly = (*root.poly).clone();
                refined_any |= Self::refine_disk_for_ordering(
                    &poly,
                    root,
                    target_radius_bits + 32,
                    target_radius_bits as u64,
                );
            }
            if !refined_any {
                return;
            }
        }
    }

    /// Refine one disk around a bounded-size approximate center for ordering.
    fn refine_disk_for_ordering(
        poly: &ExactComplexPolynomial,
        root: &mut IsolatedRoot,
        binary_precision: u32,
        target_radius_bits: u64,
    ) -> bool {
        let approximate_center = root.to_float_center(binary_precision);
        if !approximate_center.is_finite() {
            return false;
        }
        let center = Complex::new(
            approximate_center.re.to_rational(),
            approximate_center.im.to_rational(),
        );
        let center_distance = ComplexDisk::norm_upper_bound(&(&center - &root.enclosure.center));

        let max_radius = &root.enclosure.radius / &Rational::from(2);
        let mut radius = Rational::from((Integer::one(), Integer::from(2).pow(target_radius_bits)))
            .min(max_radius.clone());
        let shifted_ball = Self::shift_var_complex_ball(poly, &center, 128);
        let mut exact_shifted = None;

        for _ in 0..64 {
            if &center_distance + &radius <= root.enclosure.radius
                && Self::disk_contains_one_root_with_shift_cache(
                    poly,
                    &center,
                    &radius,
                    &shifted_ball,
                    &mut exact_shifted,
                )
            {
                root.enclosure.center = center;
                root.enclosure.radius = radius;
                return true;
            }

            if radius >= max_radius {
                return false;
            }
            radius = (radius * Rational::from(2)).min(max_radius.clone());
        }

        false
    }

    /// Record root pairs whose real parts are known to be equal by conjugation.
    fn known_equal_real_parts(roots: &[IsolatedRoot]) -> Vec<bool> {
        let mut equal = vec![false; roots.len() * roots.len()];
        for i in 0..roots.len() {
            equal[i * roots.len() + i] = true;

            let poly = &roots[i].poly;
            if poly
                .coefficients
                .iter()
                .any(|coefficient| !coefficient.im.is_zero())
            {
                continue;
            }

            // Conjugating a certified root ball of a real polynomial produces
            // another certified one-root ball. It cannot be certified disjoint
            // from the ball containing the conjugate root. Only use the match
            // when it is unique: this proves which isolated root is the
            // conjugate without requiring either independently refined ball to
            // contain the other.
            let mut conjugate = None;
            for (j, candidate) in roots.iter().enumerate() {
                let candidate_poly = &candidate.poly;
                if poly.coefficients != candidate_poly.coefficients {
                    continue;
                }

                let center_distance = ComplexDisk::norm_lower_bound(&Complex::new(
                    &roots[i].enclosure.center.re - &candidate.enclosure.center.re,
                    &roots[i].enclosure.center.im + &candidate.enclosure.center.im,
                ));
                if center_distance <= &roots[i].enclosure.radius + &candidate.enclosure.radius {
                    if conjugate.is_some() {
                        conjugate = None;
                        break;
                    }
                    conjugate = Some(j);
                }
            }

            if let Some(j) = conjugate {
                equal[i * roots.len() + j] = true;
                equal[j * roots.len() + i] = true;
            }
        }
        equal
    }

    /// Mark roots whose real intervals still overlap an unequal real part.
    fn roots_needing_real_projection(
        roots: &[IsolatedRoot],
        known_equal_real_parts: &[bool],
    ) -> Vec<bool> {
        let mut overlaps = vec![false; roots.len()];
        for i in 0..roots.len() {
            let a_lower = &roots[i].enclosure.center.re - &roots[i].enclosure.radius;
            let a_upper = &roots[i].enclosure.center.re + &roots[i].enclosure.radius;

            for j in i + 1..roots.len() {
                if known_equal_real_parts[i * roots.len() + j] {
                    continue;
                }

                let b_lower = &roots[j].enclosure.center.re - &roots[j].enclosure.radius;
                let b_upper = &roots[j].enclosure.center.re + &roots[j].enclosure.radius;
                if a_lower <= b_upper && b_lower <= a_upper {
                    overlaps[i] = true;
                    overlaps[j] = true;
                }
            }
        }
        overlaps
    }

    #[cfg(test)]
    /// Compare two roots using the complete canonical-ordering procedure.
    fn cmp_complex_roots_canonical(a: &IsolatedRoot, b: &IsolatedRoot) -> Ordering {
        let a_projected = Self::compute_projected_real_root(a);
        let b_projected = Self::compute_projected_real_root(b);
        Self::cmp_complex_roots_canonical_with_projected(
            a,
            b,
            a_projected.as_ref(),
            b_projected.as_ref(),
            false,
        )
    }

    /// Compare two roots using their disks and optional exact real projections.
    fn cmp_complex_roots_canonical_with_projected(
        a: &IsolatedRoot,
        b: &IsolatedRoot,
        a_projected: Option<&ProjectedRealRoot>,
        b_projected: Option<&ProjectedRealRoot>,
        known_equal_real_parts: bool,
    ) -> Ordering {
        let a_re_upper = &a.enclosure.center.re + &a.enclosure.radius;
        let b_re_lower = &b.enclosure.center.re - &b.enclosure.radius;
        if a_re_upper < b_re_lower {
            return Ordering::Less;
        }

        let b_re_upper = &b.enclosure.center.re + &b.enclosure.radius;
        let a_re_lower = &a.enclosure.center.re - &a.enclosure.radius;
        if b_re_upper < a_re_lower {
            return Ordering::Greater;
        }

        if known_equal_real_parts {
            return Self::cmp_complex_roots_by_imaginary_part(a, b);
        }

        if let Some(ordering) =
            Self::cmp_complex_roots_by_projected_real_parts(a_projected, b_projected)
        {
            if ordering != Ordering::Equal {
                return ordering;
            }

            return Self::cmp_complex_roots_by_imaginary_part(a, b);
        }

        match a.enclosure.center.re.cmp(&b.enclosure.center.re) {
            Ordering::Equal => {}
            ordering => return ordering,
        }

        Self::cmp_complex_roots_by_imaginary_part(a, b)
    }

    /// Compare roots by certified imaginary part with deterministic fallbacks.
    fn cmp_complex_roots_by_imaginary_part(a: &IsolatedRoot, b: &IsolatedRoot) -> Ordering {
        let a_im_upper = &a.enclosure.center.im + &a.enclosure.radius;
        let b_im_lower = &b.enclosure.center.im - &b.enclosure.radius;
        if a_im_upper < b_im_lower {
            return Ordering::Less;
        }

        let b_im_upper = &b.enclosure.center.im + &b.enclosure.radius;
        let a_im_lower = &a.enclosure.center.im - &a.enclosure.radius;
        if b_im_upper < a_im_lower {
            return Ordering::Greater;
        }

        a.enclosure
            .center
            .im
            .cmp(&b.enclosure.center.im)
            .then_with(|| a.enclosure.radius.cmp(&b.enclosure.radius))
    }

    /// Compare optional exact real projections when both are available.
    fn cmp_complex_roots_by_projected_real_parts(
        a: Option<&ProjectedRealRoot>,
        b: Option<&ProjectedRealRoot>,
    ) -> Option<Ordering> {
        let a = a?;
        let b = b?;

        if Arc::ptr_eq(&a.poly, &b.poly) {
            if a.interval == b.interval {
                return Some(Ordering::Equal);
            }

            if a.interval.1 < b.interval.0 {
                return Some(Ordering::Less);
            }

            if b.interval.1 < a.interval.0 {
                return Some(Ordering::Greater);
            }
        }

        Self::cmp_projected_real_roots(a, b)
    }

    #[cfg(test)]
    /// Compute an exact real projection for one root in tests.
    fn compute_projected_real_root(root: &IsolatedRoot) -> Option<ProjectedRealRoot> {
        let poly = &root.poly;
        let projection = Self::real_projection_polynomial(poly)?;
        let intervals = projection
            .isolate_real_root_intervals()
            .into_iter()
            .map(|(lower, upper, _)| (lower, upper))
            .collect();
        let projection = RealProjection {
            poly: Arc::new(projection),
            intervals,
        };
        let mut root = root.clone();
        Self::projected_real_root(&mut root, &projection)
    }

    /// Match a complex root disk to one interval of a real projection polynomial.
    fn projected_real_root(
        root: &mut IsolatedRoot,
        projection: &RealProjection,
    ) -> Option<ProjectedRealRoot> {
        let mut intervals = projection.intervals.clone();

        for _ in 0..1024 {
            let root_interval = Self::root_real_interval(root);
            let mut candidates = intervals
                .iter()
                .enumerate()
                .filter(|(_, interval)| {
                    Self::rational_intervals_intersect(interval, &root_interval)
                })
                .map(|(i, _)| i)
                .collect::<Vec<_>>();

            if candidates.len() == 1 {
                return Some(ProjectedRealRoot {
                    poly: projection.poly.clone(),
                    interval: intervals.swap_remove(candidates[0]),
                });
            }

            if candidates.is_empty() {
                candidates.extend(0..intervals.len());
            }

            let poly_complex = (*root.poly).clone();
            let derivative = poly_complex.derivative();
            let _ = Self::refine_root_disk_with_newton(&poly_complex, &derivative, root);

            for i in candidates {
                projection
                    .poly
                    .refine_real_root_interval_once(&mut intervals[i]);
            }
        }

        None
    }

    /// Eliminate the imaginary coordinate to obtain a polynomial for real parts.
    fn real_projection_polynomial(
        poly: &ExactComplexPolynomial,
    ) -> Option<UnivariatePolynomial<Q>> {
        let variables = Arc::new(vec![PolyVariable::Temporary(0), PolyVariable::Temporary(1)]);
        let mut real_part = MultivariatePolynomial::<Q, u16>::new(&Q, None, variables.clone());
        let mut imaginary_part = MultivariatePolynomial::<Q, u16>::new(&Q, None, variables);

        for (pow, coeff) in poly.coefficients.iter().enumerate() {
            for y_pow in 0..=pow {
                let x_pow = pow - y_pow;
                let binom = Self::binomial_rational(pow, y_pow);
                let rotated = Self::mul_complex_rational_by_i_power(coeff, y_pow);
                let exponents = [u16::try_from(x_pow).ok()?, u16::try_from(y_pow).ok()?];

                real_part.append_monomial(rotated.re * &binom, &exponents);
                imaginary_part.append_monomial(rotated.im * binom, &exponents);
            }
        }

        if real_part.is_zero() || imaginary_part.is_zero() {
            return None;
        }

        let rational_function_field = RationalPolynomialField::new(Z);
        let real_in_y = real_part.to_univariate(1).map_coeff(
            |c| RationalPolynomial::from_num_den(c.clone(), c.one(), &Z, false),
            rational_function_field.clone(),
        );
        let imaginary_in_y = imaginary_part.to_univariate(1).map_coeff(
            |c| RationalPolynomial::from_num_den(c.clone(), c.one(), &Z, false),
            rational_function_field,
        );

        let resultant = real_in_y.resultant(&imaginary_in_y);
        let mut projection = resultant
            .numerator
            .map_coeff(|c| c.to_rational(), Q)
            .to_univariate_from_univariate(0);
        projection.truncate();

        if projection.is_constant() {
            return None;
        }

        let derivative = projection.derivative();
        if !derivative.is_zero() {
            let repeated = projection.gcd(&derivative);
            if !repeated.is_constant() {
                projection = projection.quot_rem(&repeated).0;
                projection.truncate();
            }
        }

        Some(projection)
    }

    /// Compute a binomial coefficient as an exact rational number.
    fn binomial_rational(n: usize, k: usize) -> Rational {
        let k = k.min(n - k);
        let mut result = Rational::one();

        for i in 0..k {
            result *= Rational::from(n - i);
            result /= Rational::from(i + 1);
        }

        result
    }

    /// Multiply an exact complex rational by `i^pow`.
    fn mul_complex_rational_by_i_power(c: &Complex<Rational>, pow: usize) -> Complex<Rational> {
        match pow % 4 {
            0 => c.clone(),
            1 => Complex::new(-c.im.clone(), c.re.clone()),
            2 => Complex::new(-c.re.clone(), -c.im.clone()),
            _ => Complex::new(c.im.clone(), -c.re.clone()),
        }
    }

    /// Return the real-axis projection of a root's disk.
    fn root_real_interval(root: &IsolatedRoot) -> (Rational, Rational) {
        (
            root.enclosure.center.re.clone() - &root.enclosure.radius,
            root.enclosure.center.re.clone() + &root.enclosure.radius,
        )
    }

    /// Test whether two closed rational intervals intersect.
    fn rational_intervals_intersect(a: &(Rational, Rational), b: &(Rational, Rational)) -> bool {
        a.0 <= b.1 && b.0 <= a.1
    }

    /// Test whether closed rational interval `a` contains `b`.
    fn rational_interval_contains(a: &(Rational, Rational), b: &(Rational, Rational)) -> bool {
        a.0 <= b.0 && b.1 <= a.1
    }

    /// Compare projected real roots by refining their isolating intervals.
    fn cmp_projected_real_roots(a: &ProjectedRealRoot, b: &ProjectedRealRoot) -> Option<Ordering> {
        // TODO: strip GCD first?
        let mut a_interval = a.interval.clone();
        let mut b_interval = b.interval.clone();
        let gcd = a.poly.gcd(&b.poly);
        let mut common_intervals = if gcd.is_constant() {
            vec![]
        } else {
            gcd.isolate_real_root_intervals()
                .into_iter()
                .map(|(lower, upper, _)| (lower, upper))
                .collect::<Vec<_>>()
        };

        for _ in 0..1024 {
            if a_interval.1 < b_interval.0 {
                return Some(Ordering::Less);
            }

            if b_interval.1 < a_interval.0 {
                return Some(Ordering::Greater);
            }

            if common_intervals.iter().any(|interval| {
                Self::rational_interval_contains(&a_interval, interval)
                    && Self::rational_interval_contains(&b_interval, interval)
            }) {
                return Some(Ordering::Equal);
            }

            a.poly.refine_real_root_interval_once(&mut a_interval);
            b.poly.refine_real_root_interval_once(&mut b_interval);
            for interval in &mut common_intervals {
                gcd.refine_real_root_interval_once(interval);
            }
        }

        None
    }

    /// Test whether every pair of root disks is certified disjoint.
    fn root_disks_are_pairwise_disjoint(roots: &[IsolatedRoot]) -> bool {
        for i in 0..roots.len() {
            for j in i + 1..roots.len() {
                if !roots[i].enclosure.is_disjoint(&roots[j].enclosure) {
                    return false;
                }
            }
        }

        true
    }

    /// Attempt one certified Newton refinement of a root disk.
    fn refine_root_disk_with_newton(
        poly: &UnivariatePolynomial<FloatField<Complex<Rational>>>,
        derivative: &UnivariatePolynomial<FloatField<Complex<Rational>>>,
        root: &mut IsolatedRoot,
    ) -> bool {
        let derivative_at_center = derivative.evaluate(&root.enclosure.center);
        if derivative_at_center.is_zero() {
            return false;
        }

        let new_center = &root.enclosure.center
            - &(poly.evaluate(&root.enclosure.center) / derivative_at_center);
        let half_radius = root.enclosure.radius.clone() / Rational::from(2);
        let mut candidate_radii = vec![];

        let quadratic_radius = &root.enclosure.radius * &root.enclosure.radius;
        if !quadratic_radius.is_zero()
            && quadratic_radius < half_radius
            && candidate_radii
                .iter()
                .all(|radius| radius != &quadratic_radius)
        {
            candidate_radii.push(quadratic_radius);
        }
        candidate_radii.push(half_radius);
        let shifted_ball = Self::shift_var_complex_ball(poly, &new_center, 128);
        let mut exact_shifted = None;

        for mut new_radius in candidate_radii {
            for _ in 0..16 {
                if Self::disk_contains_one_root_with_shift_cache(
                    poly,
                    &new_center,
                    &new_radius,
                    &shifted_ball,
                    &mut exact_shifted,
                ) {
                    root.enclosure.center = new_center;
                    root.enclosure.radius = new_radius;
                    return true;
                }

                new_radius *= Rational::from((1, 2));
                if new_radius.is_zero() {
                    break;
                }
            }
        }

        false
    }

    /// Refine roots of one polynomial until their disks are pairwise disjoint.
    fn separate_root_disks(
        poly: &UnivariatePolynomial<FloatField<Complex<Rational>>>,
        derivative: &UnivariatePolynomial<FloatField<Complex<Rational>>>,
        roots: &mut [IsolatedRoot],
    ) -> bool {
        if Self::root_disks_are_pairwise_disjoint(roots) {
            return true;
        }

        for _ in 0..32 {
            for root in roots.iter_mut() {
                if !Self::refine_root_disk_with_newton(poly, derivative, root) {
                    return false;
                }
            }

            if Self::root_disks_are_pairwise_disjoint(roots) {
                return true;
            }
        }

        false
    }

    /// Refine roots from potentially different polynomials until disks separate.
    fn separate_isolated_roots(roots: &mut [IsolatedRoot]) {
        for _ in 0..32 {
            if Self::root_disks_are_pairwise_disjoint(roots) {
                return;
            }

            for root in roots.iter_mut() {
                let poly_complex = (*root.poly).clone();

                let derivative = poly_complex.derivative();

                if !Self::refine_root_disk_with_newton(&poly_complex, &derivative, root) {
                    return;
                }
            }
        }
    }

    /// Refine a root disk to `refine`, returning whether certification succeeded.
    fn refine_root_to_tolerance(root: &mut IsolatedRoot, refine: &Rational) -> bool {
        if refine.is_zero() {
            return true;
        }

        if root.enclosure.radius <= *refine {
            return true;
        }

        // Repeated exact rational Newton steps can grow the center's
        // numerator and denominator exponentially. Obtain a bounded-size
        // numerical center and certify the requested rational disk exactly.
        // The old ball supplies the containment proof that it is the same
        // root.
        let integer_bits = |integer: Integer| match integer {
            Integer::Single(value) => i64::BITS - value.unsigned_abs().leading_zeros(),
            Integer::Double(value) => i128::BITS - value.get().unsigned_abs().leading_zeros(),
            Integer::Large(value) => u32::try_from(value.significant_bits()).unwrap_or(u32::MAX),
        };
        let numerator_bits = integer_bits(refine.numerator());
        let denominator_bits = integer_bits(refine.denominator());
        let mut binary_precision = denominator_bits
            .saturating_sub(numerator_bits)
            .saturating_add(32)
            .max(64);
        let poly = (*root.poly).clone();
        for _ in 0..4 {
            let approximate_center = root.to_float_center(binary_precision);
            if approximate_center.is_finite() {
                let center = Complex::new(
                    approximate_center.re.to_rational(),
                    approximate_center.im.to_rational(),
                );
                let center_distance =
                    ComplexDisk::norm_upper_bound(&(&center - &root.enclosure.center));
                if &center_distance + refine <= root.enclosure.radius
                    && Self::disk_contains_one_root(&poly, &center, refine)
                {
                    root.enclosure.center = center;
                    root.enclosure.radius = refine.clone();
                    return true;
                }
            }
            binary_precision = binary_precision.saturating_mul(2);
        }

        for _ in 0..64 {
            if root.enclosure.radius <= *refine {
                return true;
            }

            let poly_complex = (*root.poly).clone();

            let derivative = poly_complex.derivative();

            if !Self::refine_root_disk_with_newton(&poly_complex, &derivative, root) {
                return false;
            }
        }

        root.enclosure.radius <= *refine
    }

    /// Bisect one real-root interval while retaining the sign change.
    fn refine_real_root_interval_once(&self, interval: &mut (Rational, Rational)) {
        if interval.0 == interval.1 {
            return;
        }

        let left_value = self.evaluate(&interval.0);
        if left_value.is_zero() {
            interval.1 = interval.0.clone();
            return;
        }

        let right_value = self.evaluate(&interval.1);
        if right_value.is_zero() {
            interval.0 = interval.1.clone();
            return;
        }

        let left_is_negative = left_value.is_negative();
        let mid = (&interval.0 + &interval.1) / Rational::from(2);
        let mid_value = self.evaluate(&mid);
        if mid_value.is_zero() {
            interval.0 = mid.clone();
            interval.1 = mid;
        } else if mid_value.is_negative() == left_is_negative {
            interval.0 = mid;
        } else {
            interval.1 = mid;
        }
    }

    /// Split `p(i y)` into its real and imaginary rational polynomials.
    fn imaginary_axis_parts(&self) -> (Self, Self) {
        let mut real = self.zero();
        let mut imaginary = self.zero();
        real.coefficients = vec![self.ring.zero(); self.coefficients.len()];
        imaginary.coefficients = vec![self.ring.zero(); self.coefficients.len()];

        for (pow, coeff) in self.coefficients.iter().enumerate() {
            if self.ring.is_zero(coeff) {
                continue;
            }

            let mut transformed = coeff.clone();
            if (pow / 2) % 2 == 1 {
                transformed = -transformed;
            }

            if pow % 2 == 0 {
                real.coefficients[pow] = transformed;
            } else {
                imaginary.coefficients[pow] = transformed;
            }
        }

        real.truncate();
        imaginary.truncate();
        (real, imaginary)
    }

    /// Isolate all roots directly when every root lies on one of the coordinate
    /// axes
    fn isolate_axis_roots(&self, target_radius: Option<&Rational>) -> Option<Vec<IsolatedRoot>> {
        let real_roots = self.isolate_real_root_intervals();
        let (real_part, imaginary_part) = self.imaginary_axis_parts();
        let imaginary_axis_poly = match (real_part.is_zero(), imaginary_part.is_zero()) {
            (true, true) => return None,
            (true, false) => imaginary_part,
            (false, true) => real_part,
            (false, false) => real_part.gcd(&imaginary_part),
        };
        let imaginary_roots = if imaginary_axis_poly.is_constant() {
            Vec::new()
        } else {
            imaginary_axis_poly.isolate_real_root_intervals()
        };

        let root_count = real_roots
            .iter()
            .chain(&imaginary_roots)
            .map(|(_, _, multiplicity)| *multiplicity)
            .sum::<usize>();
        if root_count != self.degree() {
            return None;
        }

        let complex_field = FloatField::from_rep(Complex::from(Rational::one()));
        let complex_poly = Arc::new(self.map_coeff(|c| Complex::from(c.clone()), complex_field));
        let mut roots = Vec::with_capacity(root_count);

        for (lower, upper, _) in real_roots {
            let center = (&lower + &upper) / Rational::from(2);
            let radius = (&upper - &lower) / Rational::from(2);
            roots.push(IsolatedRoot {
                poly: complex_poly.clone(),
                index: roots.len(),
                enclosure: ComplexDisk {
                    center: Complex::new(center, Rational::zero()),
                    radius,
                },
                location: Some(RootLocation::Real),
            });
        }
        for (lower, upper, _) in imaginary_roots {
            let center = (&lower + &upper) / Rational::from(2);
            let radius = (&upper - &lower) / Rational::from(2);
            roots.push(IsolatedRoot {
                poly: complex_poly.clone(),
                index: roots.len(),
                enclosure: ComplexDisk {
                    center: Complex::new(Rational::zero(), center),
                    radius,
                },
                location: Some(RootLocation::Imaginary),
            });
        }

        let derivative = complex_poly.derivative();
        if !Self::separate_root_disks(&complex_poly, &derivative, &mut roots) {
            return None;
        }

        if let Some(target_radius) = target_radius {
            for root in &mut roots {
                if !Self::refine_root_to_tolerance(root, target_radius) {
                    return None;
                }
            }
        }

        Some(roots)
    }

    /// Isolate every root of this square-free rational polynomial.
    fn isolate_square_free_roots(&self, target_radius: Option<&Rational>) -> Vec<IsolatedRoot> {
        if let Some(roots) = self.isolate_axis_roots(target_radius) {
            return roots;
        }

        let complex_field = FloatField::from_rep(Complex::from(Rational::one()));
        let complex_poly = self.map_coeff(
            |coefficient| Complex::from(coefficient.clone()),
            complex_field,
        );
        complex_poly.isolate_square_free_roots(target_radius)
    }

    /// Isolate the distinct real roots as rational intervals, together with
    /// their multiplicities.
    pub fn isolate_real_root_intervals(&self) -> Vec<(Rational, Rational, usize)> {
        let c = self.content();

        let stripped = self.map_coeff(
            |coeff| {
                let coeff = self.ring.div(coeff, &c);
                debug_assert!(coeff.is_integer());
                coeff.numerator()
            },
            Z,
        );

        stripped.isolate_real_root_intervals()
    }

    /// Approximate the single root of the polynomial in the interval (lower, higher) with a given tolerance
    /// using bisection.
    pub fn refine_root_interval(
        &self,
        mut interval: (Rational, Rational),
        tolerance: &Rational,
    ) -> (Rational, Rational) {
        if interval.0 == interval.1 {
            return interval;
        }

        // make the input square free, so that the derivative is non-zero at the roots
        let mut u = self.one();
        for (f, _pow) in self
            .clone()
            .to_multivariate::<u16>()
            .square_free_factorization()
        {
            if !f.is_constant() {
                u = u * &f.to_univariate_from_univariate(0);
            }
        }

        let left_bound_neg = match u.evaluate(&interval.0).cmp(&(0, 1).into()) {
            Ordering::Less => true,
            Ordering::Greater => false,
            Ordering::Equal => u.derivative().evaluate(&interval.0).is_negative(),
        };
        debug_assert!(u.evaluate(&interval.1).is_negative() != left_bound_neg);

        while (&interval.1 - &interval.0) / (&interval.0 + &interval.1).abs() > *tolerance {
            let mid = (&interval.0 + &interval.1) / &(2, 1).into();
            let mid_val = u.evaluate(&mid);

            if mid_val.is_negative() == left_bound_neg {
                interval.0 = mid;
            } else {
                interval.1 = mid;
            }
        }

        interval
    }

    /// Refine the intervals of two polynomials until they are disjoint.
    /// The polynomials must be square free.
    pub(super) fn refine_root_interval_until_disjoint(
        &self,
        mut interval: (Rational, Rational),
        other: &Self,
        mut other_interval: (Rational, Rational),
    ) -> ((Rational, Rational), (Rational, Rational)) {
        if !(interval.0 >= other_interval.0 && interval.0 < other_interval.1
            || interval.1 > other_interval.0 && interval.1 <= other_interval.1)
        {
            return (interval, other_interval);
        }

        let left_bound_neg = match self.evaluate(&interval.0).cmp(&(0, 1).into()) {
            Ordering::Less => true,
            Ordering::Greater => false,
            Ordering::Equal => self.derivative().evaluate(&interval.0).is_negative(),
        };
        let other_left_bound_neg = match other.evaluate(&other_interval.0).cmp(&(0, 1).into()) {
            Ordering::Less => true,
            Ordering::Greater => false,
            Ordering::Equal => other.derivative().evaluate(&other_interval.0).is_negative(),
        };

        while interval.0 >= other_interval.0 && interval.0 < other_interval.1
            || interval.1 > other_interval.0 && interval.1 <= other_interval.1
        {
            if interval.0 != interval.1 {
                let mid = (&interval.0 + &interval.1) / &(2, 1).into();
                let mid_val = self.evaluate(&mid);

                if mid_val.is_negative() == left_bound_neg {
                    interval.0 = mid;
                } else {
                    interval.1 = mid;
                }
            }

            if other_interval.0 != other_interval.1 {
                let mid = (&other_interval.0 + &other_interval.1) / &(2, 1).into();
                let mid_val = other.evaluate(&mid);

                if mid_val.is_negative() == other_left_bound_neg {
                    other_interval.0 = mid;
                } else {
                    other_interval.1 = mid;
                }
            }
        }

        (interval, other_interval)
    }

    /// Approximate all complex roots of the polynomial.
    /// Returns `Ok(roots)` when all roots were found up to the tolerance, and `Err(roots)` when the number of iterations ran out.
    /// In that case, the current-best estimate for each root is returned.
    pub fn approximate_roots<
        F: Real + SingleFloat + std::hash::Hash + Eq + PartialOrd + InternalOrdering,
    >(
        &self,
        max_iterations: usize,
        tolerance: &F,
    ) -> Result<Vec<(Complex<F>, usize)>, Vec<(Complex<F>, usize)>> {
        let mut roots = vec![];
        let mut iter_bound = false;
        for (f, pow) in self
            .clone()
            .to_multivariate::<u16>()
            .square_free_factorization()
        {
            if f.is_constant() {
                continue;
            }

            // make monic to prevent casting large integers that may overflow the float
            let f = f.to_univariate_from_univariate(0).make_monic();

            match f
                .map_coeff(
                    |c| tolerance.from_rational(c).into(),
                    FloatField::from_rep(tolerance.clone().into()),
                )
                .roots(max_iterations, tolerance)
            {
                Ok(r) => roots.extend(r.into_iter().map(|r| (r, pow))),
                Err(r) => {
                    roots.extend(r.into_iter().map(|r| (r, pow)));
                    iter_bound = true;
                }
            }
        }

        if iter_bound { Err(roots) } else { Ok(roots) }
    }
}

impl UnivariatePolynomial<ExactComplexField> {
    /// Construct the rational polynomial whose roots lie on `axis`.
    fn axis_polynomial(&self, axis: CoordinateAxis) -> Option<UnivariatePolynomial<Q>> {
        let mut real_part = self.map_coeff(|coefficient| coefficient.re.clone(), Q);
        let mut imaginary_part = self.map_coeff(|coefficient| coefficient.im.clone(), Q);
        if axis == CoordinateAxis::Imaginary {
            real_part
                .coefficients
                .resize(self.coefficients.len(), Rational::zero());
            imaginary_part
                .coefficients
                .resize(self.coefficients.len(), Rational::zero());
            for (power, coefficient) in self.coefficients.iter().enumerate() {
                let rotated =
                    UnivariatePolynomial::<Q>::mul_complex_rational_by_i_power(coefficient, power);
                real_part.coefficients[power] = rotated.re;
                imaginary_part.coefficients[power] = rotated.im;
            }
            real_part.truncate();
            imaginary_part.truncate();
        }

        let polynomial = match (real_part.is_zero(), imaginary_part.is_zero()) {
            (true, true) => return None,
            (true, false) => imaginary_part,
            (false, true) => real_part,
            (false, false) => real_part.gcd(&imaginary_part),
        };
        (!polynomial.is_constant()).then_some(polynomial)
    }

    /// Match exact roots of an axis polynomial to complex root disks.
    fn classify_axis_roots(
        axis_polynomial: &UnivariatePolynomial<Q>,
        roots: &mut [IsolatedRoot],
        axis: CoordinateAxis,
    ) {
        for (lower, upper, _) in axis_polynomial.isolate_real_root_intervals() {
            let mut interval = (lower, upper);
            let mut identified = false;
            for _ in 0..4096 {
                for root in roots.iter_mut() {
                    if axis.contains_interval(&interval, root) {
                        root.location = Some(RootLocation::with_axis(root.location, axis));
                        identified = true;
                        break;
                    }
                }

                if identified {
                    break;
                }
                axis_polynomial.refine_real_root_interval_once(&mut interval);
            }
            assert!(
                identified,
                "could not match an exact coordinate-axis root to its complex enclosure"
            );
        }
    }

    /// Classifies the root locations.
    fn classify_root_locations(&self, roots: &mut [IsolatedRoot]) {
        for axis in [CoordinateAxis::Real, CoordinateAxis::Imaginary] {
            if let Some(axis_polynomial) = self.axis_polynomial(axis) {
                Self::classify_axis_roots(&axis_polynomial, roots, axis);
            }
        }
        for root in roots {
            if root.location.is_none() {
                root.location = Some(RootLocation::Complex);
            }
        }
    }

    /// Convert this polynomial when every coefficient has zero imaginary part.
    fn try_map_to_rational(&self) -> Option<UnivariatePolynomial<Q>> {
        if self.coefficients.iter().any(|c| !c.im.is_zero()) {
            return None;
        }

        Some(self.map_coeff(|c| c.re.clone(), Q))
    }

    /// Embed an exact complex rational into the Gaussian algebraic extension.
    fn complex_rational_to_algebraic(
        field: &AlgebraicExtension<Q>,
        c: &Complex<Rational>,
    ) -> AlgebraicNumber<Q> {
        let mut poly = field.poly().constant(c.re.clone());
        if !c.im.is_zero() {
            poly = poly + field.poly().monomial(c.im.clone(), vec![1]);
        }
        field.element_from_polynomial(poly)
    }

    /// Extract an exact complex rational from the Gaussian algebraic extension.
    fn algebraic_to_complex_rational(c: &AlgebraicNumber<Q>) -> Complex<Rational> {
        Complex::new(
            c.poly().coefficient(&[0]).unwrap_or_else(Rational::zero),
            c.poly().coefficient(&[1]).unwrap_or_else(Rational::zero),
        )
    }

    /// Turn approximate roots into certified, pairwise-disjoint rational disks.
    fn certify_approximate_roots(
        &self,
        roots: &[Complex<Float>],
        target_radius: Option<&Rational>,
    ) -> Option<Vec<IsolatedRoot>> {
        let defining_polynomial = Arc::new(self.clone());
        let centers = roots
            .iter()
            .map(|root| Complex::new(root.re.to_rational(), root.im.to_rational()))
            .collect::<Vec<_>>();

        let mut complex_roots = Vec::with_capacity(centers.len());
        for (root_index, center) in centers.iter().enumerate() {
            let mut radius = UnivariatePolynomial::<Q>::initial_disk_radius(
                &centers,
                root_index,
                target_radius,
            )?;
            let shifted_ball = UnivariatePolynomial::<Q>::shift_var_complex_ball(self, center, 128);
            let mut exact_shifted = None;

            let mut certified_radius = None;
            for _ in 0..16 {
                if UnivariatePolynomial::<Q>::disk_contains_one_root_with_shift_cache(
                    self,
                    center,
                    &radius,
                    &shifted_ball,
                    &mut exact_shifted,
                ) {
                    certified_radius = Some(radius);
                    break;
                }

                radius *= Rational::from((1, 2));
                if radius.is_zero() {
                    break;
                }
            }

            complex_roots.push(IsolatedRoot {
                poly: defining_polynomial.clone(),
                index: root_index,
                enclosure: ComplexDisk {
                    center: center.clone(),
                    radius: certified_radius?,
                },
                location: None,
            });
        }

        let derivative = self.derivative();
        if UnivariatePolynomial::<Q>::separate_root_disks(self, &derivative, &mut complex_roots) {
            Some(complex_roots)
        } else {
            None
        }
    }

    /// Gets the `index`-th root of the polynomial. Fails when `index` is out of bounds.
    pub fn root(&self, index: usize) -> Option<IsolatedRoot> {
        if let Some(poly) = self.try_map_to_rational() {
            return poly.root(index);
        }
        if index >= self.degree() {
            return None;
        }

        let cache = root_cache();
        let entry = cache.complex.root_multiset_slot(self);
        let multiset = entry.get_or_init(|| self.build_root_multiset());
        cache.root_in_multiset(multiset, index)
    }

    /// Isolate the distinct complex roots of a polynomial with exact complex
    /// rational coefficients as canonically sorted `(root, multiplicity)` pairs.
    /// If all coefficients are rational, use the rational polynomial path and
    /// its root cache.
    pub fn isolate_roots(&self) -> Vec<(IsolatedRoot, usize)> {
        if let Some(poly) = self.try_map_to_rational() {
            return poly.isolate_roots();
        }

        let cache = root_cache();
        let entry = cache.complex.root_multiset_slot(self);
        let multiset = entry.get_or_init(|| self.build_root_multiset());
        cache.roots_in_multiset(multiset)
    }

    /// Isolate the distinct real roots of the polynomial. Resolving whether a
    /// root lies on the real axis may refine its cached enclosure.
    pub fn isolate_real_roots(&self) -> Vec<(IsolatedRoot, usize)> {
        self.isolate_roots()
            .into_iter()
            .filter_map(|(mut root, multiplicity)| {
                let location = root.classify_location();
                matches!(location, RootLocation::Real | RootLocation::Zero)
                    .then_some((root, multiplicity))
            })
            .collect()
    }

    /// Factor this polynomial and build its canonical root multiset.
    fn build_root_multiset(&self) -> RootMultiset {
        let complex_field = FloatField::from_rep(Complex::from(Rational::one()));
        let algebraic_field = AlgebraicExtension::complex(Q);
        let algebraic_poly = self.map_coeff(
            |c| Self::complex_rational_to_algebraic(&algebraic_field, c),
            algebraic_field.clone(),
        );

        let factors = algebraic_poly
            .to_multivariate::<u16>()
            .square_free_factorization()
            .into_iter()
            .filter(|(factor, _)| !factor.is_constant())
            .map(|(factor, multiplicity)| {
                let defining_poly = Arc::new(
                    factor
                        .to_univariate_from_univariate(0)
                        .map_coeff(Self::algebraic_to_complex_rational, complex_field.clone()),
                );
                (defining_poly, multiplicity)
            });
        root_cache().build_root_multiset(factors)
    }

    /// Isolate every root of this square-free exact-complex polynomial.
    fn isolate_square_free_roots(&self, target_radius: Option<&Rational>) -> Vec<IsolatedRoot> {
        const ABERTH_CERTIFICATION_BATCH: usize = 64;
        const MAX_ABERTH_ITERATIONS_PER_PRECISION: usize = 256;

        let mut num_prec = 128;
        let mut previous_roots: Option<Vec<Complex<Float>>> = None;

        loop {
            let tolerance = UnivariatePolynomial::<Q>::aberth_tolerance(num_prec);
            let field = FloatField::from_rep(Complex::from(tolerance.clone()));
            let c = self.map_coeff(
                |c| {
                    Complex::new(
                        c.re.to_multi_prec_float(num_prec),
                        c.im.to_multi_prec_float(num_prec),
                    )
                },
                field,
            );

            let mut roots_at_precision = previous_roots.take().map(|roots| {
                roots
                    .into_iter()
                    .map(|root: Complex<Float>| {
                        Complex::new(
                            root.re.to_rational().to_multi_prec_float(num_prec),
                            root.im.to_rational().to_multi_prec_float(num_prec),
                        )
                    })
                    .collect::<Vec<_>>()
            });

            let mut iterations = 0;
            while iterations < MAX_ABERTH_ITERATIONS_PER_PRECISION {
                let batch = ABERTH_CERTIFICATION_BATCH
                    .min(MAX_ABERTH_ITERATIONS_PER_PRECISION - iterations);
                let roots = if let Some(initial_guesses) = roots_at_precision.take() {
                    c.roots_hot_start(batch, &tolerance, initial_guesses)
                } else {
                    c.roots(batch, &tolerance)
                };
                iterations += batch;

                let aberth_converged = roots.is_ok();
                let roots = match roots {
                    Ok(roots) => roots,
                    Err(roots) => roots,
                };
                if let Some(complex_roots) = self.certify_approximate_roots(&roots, target_radius) {
                    return complex_roots;
                }

                roots_at_precision = Some(roots);
                if aberth_converged {
                    break;
                }
            }

            previous_roots = roots_at_precision;
            num_prec *= 2;
        }
    }
}

#[cfg(test)]
mod tests;
