//! Algebraic number fields, e.g. fields supporting sqrt(2).

#![warn(missing_docs)]

use std::{
    cmp::Ordering,
    collections::{HashMap, HashSet},
    sync::{Arc, LazyLock, OnceLock, RwLock},
};

use rand::Rng;

use crate::{
    atom::{Atom, AtomCore, AtomView},
    coefficient::ConvertToRing,
    combinatorics::CombinationIterator,
    domains::{
        RingOps, Set,
        atom::AtomField,
        rational::Q,
        rational_polynomial::{
            FromNumeratorAndDenominator, RationalPolynomial, RationalPolynomialField,
        },
    },
    poly::{
        Exponent, IntoVariableMap, PolyVariable, PositiveExponent, factor::Factorize,
        gcd::PolynomialGCD, polynomial::MultivariatePolynomial, univariate::RootLocation,
    },
    symbol,
    tensors::matrix::Matrix,
    transcendental::{TranscendentalFunctions, root, root_var},
};

use super::{
    EuclideanDomain, Field, InternalOrdering, OrderedRing, RealEmbedding, Ring, SelfRing,
    finite_field::{FiniteField, FiniteFieldCore, FiniteFieldWorkspace, ToFiniteField},
    integer::{Integer, IntegerRing, Z},
    rational::Rational,
};

/// A Galois field `GF(p,n)` is a finite field with `p^n` elements.
/// It provides methods to upgrade and downgrade to Galois fields with the
/// same prime but with a different power.
pub trait GaloisField: Field {
    /// Prime finite field over which this extension is defined.
    type Base: Field;

    /// Return the degree over [`Self::Base`].
    fn extension_degree(&self) -> u64;

    /// Upgrade the field to `GF(p,new_pow)`.
    fn upgrade(&self, new_pow: usize) -> AlgebraicExtension<Self::Base>
    where
        Self::Base: PolynomialGCD<u16>,
        <Self::Base as Set>::Element: Copy;

    /// Embed `e` into `larger_field`.
    fn upgrade_element(
        &self,
        e: &Self::Element,
        larger_field: &AlgebraicExtension<Self::Base>,
    ) -> <AlgebraicExtension<Self::Base> as Set>::Element;

    /// Project `e` back into this field.
    ///
    /// The caller must ensure that `e` belongs to the image of this field.
    fn downgrade_element(
        &self,
        e: &<AlgebraicExtension<Self::Base> as Set>::Element,
    ) -> Self::Element;
}

impl<UField: FiniteFieldWorkspace> GaloisField for FiniteField<UField>
where
    FiniteField<UField>: Field + FiniteFieldCore<UField>,
{
    type Base = Self;

    fn extension_degree(&self) -> u64 {
        1
    }

    fn upgrade(&self, new_pow: usize) -> AlgebraicExtension<FiniteField<UField>>
    where
        Self::Base: PolynomialGCD<u16>,
        <Self::Base as Set>::Element: Copy,
    {
        AlgebraicExtension::galois_field(self.clone(), new_pow, PolyVariable::Temporary(0))
    }

    fn upgrade_element(
        &self,
        e: &Self::Element,
        larger_field: &AlgebraicExtension<Self::Base>,
    ) -> <AlgebraicExtension<Self::Base> as Set>::Element {
        larger_field.constant(e.clone())
    }

    fn downgrade_element(
        &self,
        e: &<AlgebraicExtension<Self::Base> as Set>::Element,
    ) -> Self::Element {
        e.poly.get_constant()
    }
}

/// An algebraic number ring, with a monic, irreducible defining polynomial.
///
/// # Examples
///
/// ```
/// use symbolica::prelude::*;
///
/// let extension = AlgebraicExtension::new(parse!("x^2-2").to_polynomial(&Q, None));
/// let sqrt_2 = extension.element_from_polynomial(parse!("x").to_polynomial::<_, u16>(&Q, None));
///
/// let square = extension.mul(&sqrt_2, &sqrt_2);
/// assert_eq!(
///      square,
///      extension.element_from_polynomial(parse!("2").to_polynomial(&Q, None))
/// );
///```
///
/// Galois field:
///
/// ```
/// use symbolica::prelude::*;
///
/// let field = AlgebraicExtension::galois_field(Zp::new(17), 4, symbol!("x0").into());
/// ```
///
// TODO: make special case for degree two and three and hardcode the multiplication table
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct AlgebraicExtension<R: Ring> {
    poly: Arc<MultivariatePolynomial<R, u16>>, // TODO: convert to univariate polynomial
    embedding: usize,                          // root index
}

/// A formal simple algebraic quotient `R[t]/(f)` without a selected analytic
/// embedding.
///
/// This is the appropriate coefficient field for parametric algebraic roots:
/// before parameters are specialized, the conjugates of `f` have no globally
/// stable ordering in the complex plane.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct AlgebraicQuotient<R: Ring> {
    poly: Arc<MultivariatePolynomial<R, u16>>,
}

/// A selected root of a univariate polynomial over `R`.
///
/// For fields with an analytic embedding, such as [`Q`] and
/// [`AlgebraicExtension<Q>`], `index` refers to Symbolica's canonical ordering
/// of complex roots. Over a parametric field such as `Q(a, b, ...)`, it is a
/// formal branch label whose analytic meaning is fixed only after the
/// parameters are specialized.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Root<R: Ring> {
    polynomial: MultivariatePolynomial<R, u16>,
    index: usize,
}

impl TryFrom<AtomView<'_>> for Root<Q> {
    type Error = String;

    /// Try to convert a root function to a [Root].
    fn try_from(value: AtomView<'_>) -> Result<Self, Self::Error> {
        let AtomView::Fun(f) = value else {
            return Err("expected function tag".to_string());
        };
        if f.get_symbol() != root() {
            return Err("expected root function".to_string());
        }

        if f.get_nargs() != 2 && f.get_nargs() != 3 {
            return Err("expected 2 or 3 arguments".to_string());
        }

        let mut i = f.iter();
        let poly = i.next().unwrap();

        let mut var_or_index = i.next().unwrap();
        let mut var = None;

        if f.get_nargs() == 3 {
            if let AtomView::Var(variable) = var_or_index {
                var = Some(variable.get_symbol());
            } else {
                return Err("expected variable tag".to_string());
            };

            var_or_index = i.next().unwrap();
        };

        let Ok(index) = usize::try_from(var_or_index) else {
            return Err("expected index tag".to_string());
        };

        if let Some(variable) = var {
            Self::from_atom_with_variable(poly, variable.into(), index)
        } else {
            Self::from_atom(poly, index)
        }
    }
}

impl<R: Ring> Root<R> {
    /// Construct a root descriptor and discard inactive polynomial variables.
    ///
    /// The active variable is deliberately retained: replacing it with a
    /// fixed temporary variable could collide with the defining variable of a
    /// coefficient-field extension.
    pub fn new(polynomial: MultivariatePolynomial<R, u16>, index: usize) -> Result<Self, String> {
        let active_variables = (0..polynomial.nvars())
            .filter(|&variable| polynomial.degree(variable) > 0)
            .collect::<Vec<_>>();
        if active_variables.len() != 1 {
            return Err(format!(
                "root expects a non-constant univariate polynomial, got {} active variables",
                active_variables.len()
            ));
        }

        let polynomial = polynomial
            .to_univariate_from_univariate(active_variables[0])
            .to_multivariate::<u16>();
        let degree = polynomial.degree(0) as usize;
        if index >= degree {
            return Err(format!(
                "root index {index} is out of bounds for polynomial of degree {degree}"
            ));
        }
        Ok(Self { polynomial, index })
    }

    /// Return the polynomial that defines this root.
    pub fn polynomial(&self) -> &MultivariatePolynomial<R, u16> {
        &self.polynomial
    }

    /// Return the selected root index.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Split this root into its defining polynomial and selected index.
    pub fn into_parts(self) -> (MultivariatePolynomial<R, u16>, usize) {
        (self.polynomial, self.index)
    }
}

/// A number field together with the images of algebraic atoms in that field.
///
/// When the field is enlarged, all stored images are transported to the new
/// primitive-element representation.
#[derive(Clone)]
pub struct AlgebraicContext {
    field: AlgebraicExtension<Q>,
    images: HashMap<Atom, AlgebraicNumber<Q>>,
}

impl<T: FiniteFieldWorkspace> AlgebraicExtension<FiniteField<T>>
where
    FiniteField<T>: FiniteFieldCore<T>,
{
    /// Encode a finite-extension element as an integer in base field order.
    pub fn to_integer(&self, a: &<Self as Set>::Element) -> Integer {
        let mut p = Integer::zero();
        for x in a.poly.into_iter() {
            p += &(self.poly.ring().to_integer(x.coefficient)
                * &self.characteristic().pow(x.exponents[0] as u64));
        }
        p
    }

    /// Encode an element using the symmetric integer representative.
    pub fn to_symmetric_integer(&self, a: &<Self as Set>::Element) -> Integer {
        let r = self.to_integer(a);
        let s = self.size().unwrap();
        if &r * 2 <= s { r } else { &r - &s }
    }
}

impl<T: FiniteFieldWorkspace> GaloisField for AlgebraicExtension<FiniteField<T>>
where
    FiniteField<T>: FiniteFieldCore<T> + PolynomialGCD<u16>,
{
    type Base = FiniteField<T>;

    fn extension_degree(&self) -> u64 {
        self.poly.degree(0) as u64
    }

    fn upgrade(&self, new_pow: usize) -> AlgebraicExtension<Self::Base>
    where
        Self::Base: PolynomialGCD<u16>,
        <Self::Base as Set>::Element: Copy,
    {
        AlgebraicExtension::galois_field(
            self.poly.ring().clone(),
            new_pow,
            self.poly.variables()[0].clone(),
        )
    }

    fn upgrade_element(
        &self,
        e: &Self::Element,
        larger_field: &AlgebraicExtension<Self::Base>,
    ) -> <AlgebraicExtension<Self::Base> as Set>::Element {
        larger_field.element_from_polynomial(e.poly.clone())
    }

    fn downgrade_element(
        &self,
        e: &<AlgebraicExtension<Self::Base> as Set>::Element,
    ) -> Self::Element {
        self.element_from_polynomial(e.poly.clone())
    }
}

impl<UField: FiniteFieldWorkspace> ConvertToRing for AlgebraicExtension<FiniteField<UField>>
where
    FiniteField<UField>: FiniteFieldCore<UField> + PolynomialGCD<u16>,
    Integer: ToFiniteField<UField>,
{
    fn element_from_integer(&self, number: Integer) -> Self::Element {
        let mut q = &number % &self.size().unwrap();
        let mut pow = 0;
        let mut poly = self.poly.zero();
        while !q.is_zero() {
            let (qn, r) = q.quot_rem(&self.poly.ring().size().unwrap());
            poly.append_monomial(r.to_finite_field(&self.poly.ring()), &[pow]);
            pow += 1;
            q = qn;
        }

        AlgebraicNumber { poly }
    }

    fn try_element_from_coefficient(
        &self,
        number: crate::coefficient::Coefficient,
    ) -> Result<Self::Element, String> {
        match number {
            crate::coefficient::Coefficient::Indeterminate => {
                Err("Cannot convert indeterminate to rational".to_string())
            }
            crate::coefficient::Coefficient::Infinity(_) => {
                Err("Cannot convert infinity to rational".to_string())
            }
            crate::coefficient::Coefficient::Complex(r) => {
                if r.is_real() {
                    let n = self.element_from_integer(r.re.numerator());
                    let d = self.element_from_integer(r.re.denominator());
                    Ok(self.div(&n, &d))
                } else if self.poly().exponents == [0, 2]
                    && self.poly().ring().is_one(&self.poly().get_constant())
                {
                    let ring = &self.poly().ring();
                    let re = {
                        let n = ring.element_from_integer(r.re.numerator());
                        let d = ring.element_from_integer(r.re.denominator());
                        ring.div(&n, &d)
                    };

                    let im = {
                        let n = ring.element_from_integer(r.im.numerator());
                        let d = ring.element_from_integer(r.im.denominator());
                        ring.div(&n, &d)
                    };

                    Ok(self.element_from_polynomial(
                        self.poly().monomial(re, vec![1]) + self.poly().constant(im),
                    ))
                } else {
                    Err(
                        "Cannot directly convert complex number to this extension. First create a polynomial with extension x^2+1 and then upgrade.".to_string()
                    )
                }
            }
            crate::coefficient::Coefficient::Float(_) => {
                Err("Cannot convert float coefficient to algebraic number".to_string())
            }
            crate::coefficient::Coefficient::FiniteField(_, _) => {
                // TODO: check if the field is the same? how?
                Err("Cannot convert finite field coefficient to algebraic number".to_string())
            }
            crate::coefficient::Coefficient::RationalPolynomial(_) => Err(
                "Cannot convert rational polynomial coefficient to algebraic number".to_string(),
            ),
        }
    }

    fn try_element_from_coefficient_view(
        &self,
        number: crate::coefficient::CoefficientView<'_>,
    ) -> Result<Self::Element, String> {
        match number {
            crate::coefficient::CoefficientView::Natural(r, d, cr, cd) => {
                if cr == 0 {
                    let n = self.element_from_integer(r.into());
                    let d = self.element_from_integer(d.into());
                    Ok(self.div(&n, &d))
                } else if self.poly().exponents == [0, 2]
                    && self.poly().ring().is_one(&self.poly().get_constant())
                {
                    let ring = &self.poly().ring();
                    let re = {
                        let n = ring.element_from_integer(r.into());
                        let d = ring.element_from_integer(d.into());
                        ring.div(&n, &d)
                    };

                    let im = {
                        let n = ring.element_from_integer(cr.into());
                        let d = ring.element_from_integer(cd.into());
                        ring.div(&n, &d)
                    };

                    Ok(self.element_from_polynomial(
                        self.poly().monomial(re, vec![1]) + self.poly().constant(im),
                    ))
                } else {
                    Err(
                        "Cannot directly convert complex number to this extension. First create a polynomial with extension x^2+1 and then upgrade.".to_string(),
                    )
                }
            }
            crate::coefficient::CoefficientView::Large(r, i) => {
                if i.is_zero() {
                    let r: Rational = r.to_rat();
                    let n = self.element_from_integer(r.numerator());
                    let d = self.element_from_integer(r.denominator());
                    Ok(self.div(&n, &d))
                } else if self.poly().exponents == [0, 2]
                    && self.poly().ring().is_one(&self.poly().get_constant())
                {
                    let ring = &self.poly().ring();
                    let re = {
                        let r = r.to_rat();
                        let n = ring.element_from_integer(r.numerator());
                        let d = ring.element_from_integer(r.denominator());
                        ring.div(&n, &d)
                    };

                    let im = {
                        let cr = i.to_rat();
                        let n = ring.element_from_integer(cr.numerator());
                        let d = ring.element_from_integer(cr.denominator());
                        ring.div(&n, &d)
                    };

                    Ok(self.element_from_polynomial(
                        self.poly().monomial(re, vec![1]) + self.poly().constant(im),
                    ))
                } else {
                    Err(
                        "Cannot directly convert complex number to this extension. First create a polynomial with extension x^2+1 and then upgrade.".to_string(),
                    )
                }
            }
            crate::coefficient::CoefficientView::Float(_, _) => {
                Err("Cannot convert float coefficient to algebraic number".to_string())
            }
            crate::coefficient::CoefficientView::FiniteField(_, _) => {
                Err("Cannot convert finite field coefficient to algebraic number".to_string())
            }
            crate::coefficient::CoefficientView::RationalPolynomial(_) => Err(
                "Cannot convert rational polynomial coefficient to algebraic number".to_string(),
            ),
            crate::coefficient::CoefficientView::Indeterminate => {
                Err("Cannot convert indeterminate to algebraic number".to_string())
            }
            crate::coefficient::CoefficientView::Infinity(_) => {
                Err("Cannot convert infinity to algebraic number".to_string())
            }
        }
    }
}

impl<UField: FiniteFieldWorkspace> AlgebraicExtension<FiniteField<UField>>
where
    FiniteField<UField>: FiniteFieldCore<UField> + PolynomialGCD<u16>,
    <FiniteField<UField> as Set>::Element: Copy,
{
    /// Construct the Galois field GF(prime^exp).
    /// The irreducible polynomial is determined automatically.
    pub fn galois_field(prime: FiniteField<UField>, exp: usize, var: PolyVariable) -> Self {
        assert!(exp > 0);

        if exp == 1 {
            let mut poly = MultivariatePolynomial::new(&prime, None, Arc::new(vec![var]));

            poly.append_monomial(prime.one(), &[1]);
            return AlgebraicExtension::new(poly);
        }

        fn is_irreducible<UField: FiniteFieldWorkspace>(
            coeffs: &[u64],
            poly: &mut MultivariatePolynomial<FiniteField<UField>, u16>,
        ) -> bool
        where
            FiniteField<UField>: FiniteFieldCore<UField> + PolynomialGCD<u16>,
            <FiniteField<UField> as Set>::Element: Copy,
            AlgebraicExtension<FiniteField<UField>>: PolynomialGCD<u16>,
        {
            poly.clear();
            for (i, c) in coeffs.iter().enumerate() {
                poly.append_monomial(poly.ring().nth((*c).into()), &[i as u16]);
            }

            poly.is_irreducible()
        }

        let mut coeffs = vec![0; exp + 1];
        coeffs[exp] = 1;
        let mut poly = MultivariatePolynomial::new(&prime, Some(coeffs.len()), Arc::new(vec![var]));

        // find the minimal polynomial
        let p = prime.get_prime().to_integer();
        if p == 2 {
            coeffs[0] = 1;

            // try all odd number of non-zero coefficients
            for g in 0..exp / 2 {
                let g = 2 * g + 1;

                let mut c = CombinationIterator::new(exp - 1, g);
                while let Some(comb) = c.next() {
                    for i in 0..g {
                        coeffs[comb[i] + 1] = 1;
                    }

                    if is_irreducible(&coeffs, &mut poly) {
                        return AlgebraicExtension::new(poly);
                    }

                    for i in 0..g {
                        coeffs[comb[i] + 1] = 0;
                    }
                }
            }

            unreachable!("No irreducible polynomial found for GF({},{})", prime, exp);
        }

        let sample_max = p.to_i64().unwrap_or(i64::MAX) as u64;
        if exp == 2 {
            for k in 1..sample_max {
                coeffs[0] = k;

                if is_irreducible(&coeffs, &mut poly) {
                    return AlgebraicExtension::new(poly);
                }
            }

            unreachable!("No irreducible polynomial found for GF({},{})", prime, exp);
        }

        // try shape x^n+a*x+b for fast division
        for k in 1..sample_max {
            for k2 in 1..sample_max {
                coeffs[0] = k;
                coeffs[1] = k2;

                if is_irreducible(&coeffs, &mut poly) {
                    return AlgebraicExtension::new(poly);
                }
            }
        }

        // try random polynomials
        let mut r = rand::rng();
        loop {
            for c in coeffs.iter_mut() {
                *c = r.random_range(0..sample_max);
            }
            coeffs[exp] = 1;

            if is_irreducible(&coeffs, &mut poly) {
                return AlgebraicExtension::new(poly);
            }
        }
    }
}

impl AlgebraicContext {
    fn positive_real_root_index(poly: &MultivariatePolynomial<Q, u16>) -> Option<usize> {
        let poly = poly.to_univariate_from_univariate(0);
        let mut index = 0;
        for (mut root, multiplicity) in poly.isolate_roots() {
            if root.is_positive_real() {
                return Some(index);
            }
            index += multiplicity;
        }
        None
    }

    fn coefficient_has_imaginary_part(number: crate::coefficient::CoefficientView<'_>) -> bool {
        match number {
            crate::coefficient::CoefficientView::Natural(_, _, imaginary, _) => imaginary != 0,
            crate::coefficient::CoefficientView::Large(_, imaginary) => !imaginary.is_zero(),
            _ => false,
        }
    }

    fn is_explicit_generator(atom: AtomView<'_>) -> bool {
        match atom {
            AtomView::Fun(function) => function.get_symbol() == root(),
            AtomView::Pow(power) => {
                let (_, exponent) = power.get_base_exp();
                Rational::try_from(exponent)
                    .is_ok_and(|rational| !rational.denominator_ref().is_one())
            }
            AtomView::Num(number) => Self::coefficient_has_imaginary_part(number.get_coeff_view()),
            _ => false,
        }
    }

    /// Create an empty context over `field`.
    pub fn new(field: AlgebraicExtension<Q>) -> Self {
        Self {
            field,
            images: HashMap::new(),
        }
    }

    /// Build a context and convert every algebraic subexpression in `atom`.
    pub fn from_atom(atom: AtomView<'_>) -> Result<Self, String> {
        let mut context = Self::new(AlgebraicExtension::trivial(Q));
        context.extend(atom)?;
        Ok(context)
    }

    /// Build a context from explicit algebraic generators.
    pub fn from_generators(generators: &[Atom]) -> Result<Self, String> {
        if generators.is_empty() {
            return Err("At least one algebraic generator is required".to_string());
        }

        let mut context = Self::new(AlgebraicExtension::trivial(Q));
        context.adjoin_generators(generators)?;
        Ok(context)
    }

    /// Adjoin explicit algebraic generators to this context.
    pub fn adjoin_generators(&mut self, generators: &[Atom]) -> Result<(), String> {
        for generator in generators {
            if !Self::is_explicit_generator(generator.as_view()) {
                return Err(format!(
                    "{} is not an explicit root or a rational power",
                    generator
                ));
            }
            self.convert_atom(generator.as_view())?;
        }
        Ok(())
    }

    /// Return the field containing all recorded images.
    pub fn field(&self) -> &AlgebraicExtension<Q> {
        &self.field
    }

    /// Return whether this context still represents the base field `Q`.
    pub fn is_trivial(&self) -> bool {
        self.field.poly.degree(0) <= 1
    }

    /// Return all known expression-to-field-element mappings.
    pub fn images(&self) -> &HashMap<Atom, AlgebraicNumber<Q>> {
        &self.images
    }

    /// Return the field element corresponding to `atom`, if already known.
    pub fn image(&self, atom: &Atom) -> Option<&AlgebraicNumber<Q>> {
        self.images.get(atom)
    }

    /// Record the image of an atom that is already known to belong to the
    /// current field.
    pub(crate) fn insert_image(&mut self, atom: Atom, image: AlgebraicNumber<Q>) {
        self.images.insert(atom, image);
    }

    /// Convert a field element to an atom and cache that representation.
    pub fn atom_from_element(&mut self, element: AlgebraicNumber<Q>) -> Result<Atom, String> {
        if let Some(atom) = self
            .images
            .iter()
            .filter_map(|(atom, image)| {
                self.field
                    .is_zero(&self.field.sub(image, &element))
                    .then_some(atom)
            })
            .min()
            .cloned()
        {
            return Ok(atom);
        }

        let atom = self.field.element_to_atom(&element);
        self.images.insert(atom.clone(), element);
        Ok(atom)
    }

    /// Adjoin one embedded root of an irreducible polynomial over this field.
    ///
    /// `polynomial` must be univariate, use this context's current field as
    /// its coefficient ring, and be irreducible over that field.
    pub fn adjoin_root(
        &mut self,
        polynomial: MultivariatePolynomial<AlgebraicExtension<Q>, u16>,
        embedding: usize,
    ) -> Result<Atom, String> {
        if polynomial.ring() != &self.field {
            return Err("The root polynomial uses a different coefficient field".to_string());
        }
        if polynomial.nvars() != 1 || polynomial.degree(0) == 0 {
            return Err("The root polynomial must be non-constant and univariate".to_string());
        }
        if embedding >= polynomial.degree(0) as usize {
            return Err(format!(
                "Root index {embedding} is out of bounds for polynomial of degree {}",
                polynomial.degree(0)
            ));
        }

        let extension = AlgebraicExtension::from_polynomial_with_embedding(polynomial, embedding);
        let new_variable = self.field.fresh_variable();
        let (field, old_generator, new_generator) = self
            .field
            .adjoin_with_embedding(&extension, Some(new_variable));

        for image in self.images.values_mut() {
            *image = Self::transport_element(image, &field, &old_generator);
        }
        self.field = field;
        self.atom_from_element(new_generator)
    }

    /// Split this context into its field and expression images.
    pub fn into_parts(self) -> (AlgebraicExtension<Q>, HashMap<Atom, AlgebraicNumber<Q>>) {
        (self.field, self.images)
    }

    fn transport_element(
        element: &AlgebraicNumber<Q>,
        field: &AlgebraicExtension<Q>,
        old_generator: &AlgebraicNumber<Q>,
    ) -> AlgebraicNumber<Q> {
        let mut coefficients = vec![Rational::zero(); element.poly.degree(0) as usize + 1];
        for term in element.poly() {
            coefficients[term.exponents[0] as usize] = term.coefficient.clone();
        }

        let mut result = field.zero();
        for coefficient in coefficients.into_iter().rev() {
            result = field.add(
                &field.mul(&result, old_generator),
                &field.constant(coefficient),
            );
        }
        result
    }

    fn replace_field(&mut self, field: AlgebraicExtension<Q>) -> Result<(), String> {
        if self.field == field {
            return Ok(());
        }

        let old_generator = field.embedded_rational_root(&self.field.poly, self.field.embedding)?;
        for image in self.images.values_mut() {
            *image = Self::transport_element(image, &field, &old_generator);
        }
        self.field = field;
        Ok(())
    }

    fn ensure_field_for(&mut self, atom: AtomView<'_>) -> Result<(), String> {
        if self.is_trivial() {
            if let Some(field) = atom.discover_embedding_field(None) {
                self.replace_field(field)?;
            }
            Ok(())
        } else {
            let field = atom
                .discover_embedding_field(Some(self.field.clone()))
                .ok_or_else(|| format!("Could not construct an embedding field for {atom}"))?;
            self.replace_field(field)
        }
    }

    /// Discover and cache every supported algebraic subexpression in `atom`.
    pub fn extend(&mut self, atom: AtomView<'_>) -> Result<(), String> {
        let key = atom.to_owned();
        if self.images.contains_key(&key) {
            return Ok(());
        }

        match atom {
            AtomView::Num(_) => {
                self.ensure_field_for(atom)?;
                let image = atom.to_algebraic(&self.field)?;
                self.images.insert(key, image);
            }
            AtomView::Var(_) => {}
            AtomView::Fun(function) => {
                if function.get_symbol() == root() {
                    self.ensure_field_for(atom)?;
                    let image = atom.to_algebraic(&self.field)?;
                    self.images.insert(key, image);
                }
            }
            AtomView::Pow(power) => {
                let (base, exponent) = power.get_base_exp();
                self.extend(base)?;
                if !self.images.contains_key(&base.to_owned())
                    || Rational::try_from(exponent).is_err()
                {
                    return Ok(());
                }

                self.ensure_field_for(atom)?;
                let image = atom.to_algebraic(&self.field)?;
                self.images.insert(key, image);
            }
            AtomView::Mul(product) => {
                let factors = product.into_iter().collect::<Vec<_>>();
                for factor in &factors {
                    self.extend(*factor)?;
                }

                let mut result = self.field.one();
                for factor in factors {
                    let Some(image) = self.images.get(&factor.to_owned()) else {
                        return Ok(());
                    };
                    self.field.mul_assign(&mut result, image);
                }
                self.images.insert(key, result);
            }
            AtomView::Add(sum) => {
                let terms = sum.into_iter().collect::<Vec<_>>();
                for term in &terms {
                    self.extend(*term)?;
                }

                let mut result = self.field.zero();
                for term in terms {
                    let Some(image) = self.images.get(&term.to_owned()) else {
                        return Ok(());
                    };
                    self.field.add_assign(&mut result, image);
                }
                self.images.insert(key, result);
            }
        }

        Ok(())
    }

    /// Convert an algebraic atom to its image in the context's current field.
    pub fn convert_atom(&mut self, atom: AtomView<'_>) -> Result<AlgebraicNumber<Q>, String> {
        self.extend(atom)?;
        self.images
            .get(&atom.to_owned())
            .cloned()
            .ok_or_else(|| format!("{atom} is not an algebraic constant"))
    }

    fn polynomial_from_images<E: Exponent>(
        &self,
        atom: AtomView<'_>,
        var_map: &Arc<Vec<PolyVariable>>,
    ) -> Result<MultivariatePolynomial<AlgebraicExtension<Q>, E>, String> {
        if let Some(image) = self.images.get(&atom.to_owned()) {
            return Ok(
                MultivariatePolynomial::new(&self.field, None, var_map.clone())
                    .constant(image.clone()),
            );
        }

        match atom {
            AtomView::Add(sum) => {
                let mut result = MultivariatePolynomial::new(&self.field, None, var_map.clone());
                for term in sum {
                    let mut term = self.polynomial_from_images(term, &result.variables())?;
                    result.unify_variables(&mut term);
                    result = &result + &term;
                }
                Ok(result)
            }
            AtomView::Mul(product) => {
                let mut result =
                    MultivariatePolynomial::new(&self.field, None, var_map.clone()).one();
                for factor in product {
                    let mut factor = self.polynomial_from_images(factor, &result.variables())?;
                    result.unify_variables(&mut factor);
                    result = &result * &factor;
                }
                Ok(result)
            }
            _ => atom
                .try_to_polynomial(&self.field, Some(var_map.clone()))
                .map_err(|error| error.to_string()),
        }
    }

    fn rational_polynomial_from_images<E: PositiveExponent>(
        &self,
        atom: AtomView<'_>,
        var_map: &Arc<Vec<PolyVariable>>,
    ) -> Result<RationalPolynomial<AlgebraicExtension<Q>, E>, String>
    where
        RationalPolynomial<AlgebraicExtension<Q>, E>:
            FromNumeratorAndDenominator<AlgebraicExtension<Q>, AlgebraicExtension<Q>, E>,
    {
        if let Some(image) = self.images.get(&atom.to_owned()) {
            let numerator = MultivariatePolynomial::new(&self.field, None, var_map.clone())
                .constant(image.clone());
            let denominator = numerator.one();
            return Ok(RationalPolynomial::from_num_den(
                numerator,
                denominator,
                &self.field,
                false,
            ));
        }

        match atom {
            AtomView::Add(sum) => {
                let mut result = RationalPolynomial::new(&self.field, var_map.clone());
                for term in sum {
                    let mut term =
                        self.rational_polynomial_from_images(term, &result.numerator.variables())?;
                    result.unify_variables(&mut term);
                    result = &result + &term;
                }
                Ok(result)
            }
            AtomView::Mul(product) => {
                let mut result = RationalPolynomial::new(&self.field, var_map.clone());
                result.numerator = result.numerator.add_constant(self.field.one());
                for factor in product {
                    let mut factor = self
                        .rational_polynomial_from_images(factor, &result.numerator.variables())?;
                    result.unify_variables(&mut factor);
                    result = &result * &factor;
                }
                Ok(result)
            }
            AtomView::Pow(power) => {
                let (base, exponent) = power.get_base_exp();
                if let Ok(exponent) = Rational::try_from(exponent)
                    && exponent.is_integer()
                    && let Some(exponent) = exponent.numerator().to_i64()
                {
                    let base = self.rational_polynomial_from_images(base, var_map)?;
                    if exponent < 0 {
                        Ok(base.inv().pow(exponent.unsigned_abs()))
                    } else {
                        Ok(base.pow(exponent as u64))
                    }
                } else {
                    atom.try_to_rational_polynomial(&self.field, &self.field, Some(var_map.clone()))
                        .map_err(|error| error.to_string())
                }
            }
            _ => atom
                .try_to_rational_polynomial(&self.field, &self.field, Some(var_map.clone()))
                .map_err(|error| error.to_string()),
        }
    }

    fn polynomial_to_atom<E: PositiveExponent>(
        &self,
        polynomial: &MultivariatePolynomial<AlgebraicExtension<Q>, E>,
        generator: &Atom,
        express_in_generator: bool,
        element_atoms: &mut HashMap<AlgebraicNumber<Q>, Atom>,
    ) -> Result<Atom, String> {
        let atom_field = AtomField::new();
        let variables = polynomial
            .variables()
            .iter()
            .map(|variable| {
                if variable == &self.field.poly.get_vars_ref()[0] {
                    PolyVariable::Power(generator.clone())
                } else {
                    variable.clone()
                }
            })
            .collect();
        let mut converted = MultivariatePolynomial::<AtomField, E>::new(
            &atom_field,
            Some(polynomial.nterms()),
            Arc::new(variables),
        );
        for term in polynomial {
            let coefficient = if let Some(coefficient) = element_atoms.get(term.coefficient) {
                coefficient.clone()
            } else {
                let coefficient = if express_in_generator {
                    let mut coefficient = Atom::Zero;
                    for generator_term in term.coefficient.poly() {
                        coefficient += generator.clone().pow(generator_term.exponents[0])
                            * Atom::num(generator_term.coefficient.clone());
                    }
                    coefficient
                } else {
                    self.field.element_to_atom_simplified(term.coefficient)
                };
                element_atoms.insert(term.coefficient.clone(), coefficient.clone());
                coefficient
            };
            converted.append_monomial(coefficient, term.exponents);
        }
        Ok(converted.flatten(false))
    }

    fn factorization_to_atom(
        &self,
        numerator: Vec<(MultivariatePolynomial<AlgebraicExtension<Q>, u16>, usize)>,
        denominator: Vec<(MultivariatePolynomial<AlgebraicExtension<Q>, u16>, usize)>,
    ) -> Result<Atom, String> {
        let mut element_atoms = HashMap::<AlgebraicNumber<Q>, Atom>::new();
        for (atom, element) in &self.images {
            match element_atoms.entry(element.clone()) {
                std::collections::hash_map::Entry::Occupied(mut entry) => {
                    if atom < entry.get() {
                        *entry.get_mut() = atom.clone();
                    }
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(atom.clone());
                }
            }
        }

        let generator_image = self.field.generator();
        let preferred_generator = element_atoms.get(&generator_image).cloned();
        let express_in_generator = preferred_generator.is_some();
        let generator = preferred_generator.unwrap_or_else(|| {
            let generator = self.field.element_to_atom_simplified(&generator_image);
            element_atoms.insert(generator_image, generator.clone());
            generator
        });

        let mut result = Atom::num(1);
        for (factor, exponent) in numerator {
            let factor = self.polynomial_to_atom(
                &factor,
                &generator,
                express_in_generator,
                &mut element_atoms,
            )?;
            result *= factor.pow(Atom::num(exponent));
        }
        for (factor, exponent) in denominator {
            let factor = self.polynomial_to_atom(
                &factor,
                &generator,
                express_in_generator,
                &mut element_atoms,
            )?;
            let exponent = i64::try_from(exponent)
                .map_err(|_| "Factor multiplicity is too large".to_string())?;
            result *= factor.pow(Atom::num(-exponent));
        }
        Ok(result)
    }

    /// Convert an atom to a polynomial whose coefficients are in the context's
    /// algebraic extension. Field construction and image transport happen
    /// before the polynomial is materialized.
    pub fn to_polynomial<E: Exponent>(
        &mut self,
        atom: AtomView<'_>,
        var_map: impl IntoVariableMap,
    ) -> Result<MultivariatePolynomial<AlgebraicExtension<Q>, E>, String> {
        self.extend(atom)?;
        let var_map = var_map
            .into_var_map()?
            .unwrap_or_else(|| Arc::new(Vec::new()));
        self.polynomial_from_images(atom, &var_map)
    }

    /// Convert an atom to a rational polynomial whose coefficients are in the
    /// context's algebraic extension.
    pub fn to_rational_polynomial<E: PositiveExponent>(
        &mut self,
        atom: AtomView<'_>,
        var_map: impl IntoVariableMap,
    ) -> Result<RationalPolynomial<AlgebraicExtension<Q>, E>, String>
    where
        RationalPolynomial<AlgebraicExtension<Q>, E>:
            FromNumeratorAndDenominator<AlgebraicExtension<Q>, AlgebraicExtension<Q>, E>,
    {
        self.extend(atom)?;
        let var_map = var_map
            .into_var_map()?
            .unwrap_or_else(|| Arc::new(Vec::new()));
        self.rational_polynomial_from_images(atom, &var_map)
    }
}

impl AtomView<'_> {
    /// Convert the atom to an algebraic number in the current ring, if possible.
    pub fn to_algebraic(&self, ring: &AlgebraicExtension<Q>) -> Result<AlgebraicNumber<Q>, String> {
        match self {
            AtomView::Num(n) => ring.try_element_from_coefficient_view(n.get_coeff_view()),
            AtomView::Var(_) => {
                Err("variable atoms cannot be converted to algebraic numbers".to_string())
            }
            AtomView::Fun(f) => {
                if f.get_symbol() == root() {
                    let r = Root::<Q>::try_from(*self)?;
                    ring.embedded_rational_root(r.polynomial(), r.index)
                } else {
                    Err("function atoms cannot be converted to algebraic numbers".to_string())
                }
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();
                let base_converted = b.to_algebraic(&ring)?;

                if let Ok(r) = Rational::try_from(e)
                    && let Ok(numerator) = i64::try_from(r.numerator())
                    && let Ok(denominator) = u64::try_from(r.denominator())
                {
                    if r.is_integer() {
                        if r.is_negative() {
                            return Ok(
                                ring.inv(&ring.pow(&base_converted, numerator.unsigned_abs()))
                            );
                        }
                        return Ok(ring.pow(&base_converted, numerator.unsigned_abs()));
                    }

                    let var = ring.fresh_variable();
                    let mut poly = MultivariatePolynomial::<_, u16>::new(
                        ring,
                        None,
                        Arc::new(vec![var.clone()]),
                    );
                    poly = poly.variable(&var).unwrap().pow(denominator as usize)
                        - poly.constant(base_converted);

                    let f = poly.factor();

                    let mut selected_root = None;
                    for (factor, _) in f {
                        if factor.degree(0) != 1 {
                            continue;
                        }

                        let root = ring.neg(&ring.div(&factor.get_constant(), &factor.lcoeff()));
                        if ring.is_positive_real(&root)? {
                            if selected_root.is_some() {
                                return Err(format!(
                                    "More than one positive real value found for {}",
                                    self
                                ));
                            }
                            selected_root = Some(root);
                        }
                    }

                    let Some(mut root) = selected_root else {
                        return Err(format!(
                            "The current extension does not contain the principal root {}",
                            self
                        ));
                    };

                    if numerator < 0 {
                        root = ring.inv(&root);
                    }
                    return Ok(ring.pow(&root, numerator.unsigned_abs()));
                }

                Err("Exponent is not rational".to_string())
            }
            AtomView::Mul(m) => {
                let mut res = ring.one();
                for a in m {
                    ring.mul_assign(&mut res, &a.to_algebraic(&ring)?);
                }
                Ok(res)
            }
            AtomView::Add(a) => {
                let mut res = ring.zero();
                for a in a {
                    ring.add_assign(&mut res, &a.to_algebraic(&ring)?);
                }
                Ok(res)
            }
        }
    }

    /// Return the field that describes all algebraic numbers in the input.
    ///
    /// Use [`algebraic_context`](Self::algebraic_context) when the atom
    /// will subsequently be converted, so that the computed atom images are
    /// retained.
    pub fn embedding_field(&self) -> Option<AlgebraicExtension<Q>> {
        let context = self.algebraic_context().ok()?;
        (!context.is_trivial()).then_some(context.field)
    }

    /// Build a live algebraic context for the atom.
    pub fn algebraic_context(&self) -> Result<AlgebraicContext, String> {
        AlgebraicContext::from_atom(*self)
    }

    /// Factor an expression over an algebraic number field.
    ///
    /// Algebraic numbers already present in the expression define the initial
    /// field. `generators` are then adjoined to that field. Each additional
    /// generator must be an explicit `root(poly, index)`, a rational power
    /// such as `2^(1/2)`, or an exact complex coefficient defining `i`.
    ///
    /// An empty generator list factors over the algebraic extensions already
    /// present in the expression, or over `Q` if there are none.
    ///
    /// # Examples
    ///
    /// ```
    /// use symbolica::prelude::*;
    ///
    /// let factorization = parse!("x^2-2")
    ///     .as_view()
    ///     .factor_in_extension(&[parse!("sqrt(2)")])
    ///     .unwrap();
    /// assert_eq!(factorization, parse!("(x-sqrt(2))*(x+sqrt(2))"));
    /// ```
    pub fn factor_in_extension(&self, generators: &[Atom]) -> Result<Atom, String> {
        let mut context = AlgebraicContext::from_atom(*self)?;
        if context.is_trivial() && generators.is_empty() {
            return Ok(self.factor());
        }
        context.adjoin_generators(generators)?;
        let rational = context.to_rational_polynomial::<u16>(*self, None)?;
        if rational.is_zero() {
            return Ok(Atom::num(0));
        }

        let numerator = rational.numerator.factor();
        let denominator = rational.denominator.factor();
        context.factorization_to_atom(numerator, denominator)
    }

    /// Convert the atom to a polynomial over a generated algebraic extension.
    ///
    /// Returns `Ok(None)` when no algebraic extension is required.
    pub fn to_polynomial_in_algebraic_extension<E: Exponent>(
        &self,
        var_map: impl IntoVariableMap,
    ) -> Result<
        Option<(
            AlgebraicContext,
            MultivariatePolynomial<AlgebraicExtension<Q>, E>,
        )>,
        String,
    > {
        let mut context = self.algebraic_context()?;
        if context.is_trivial() {
            return Ok(None);
        }
        let polynomial = context.to_polynomial(*self, var_map)?;
        Ok(Some((context, polynomial)))
    }

    /// Convert the atom to a rational polynomial over a generated algebraic
    /// extension. Returns `Ok(None)` when no extension is required.
    pub fn to_rational_polynomial_in_algebraic_extension<E: PositiveExponent>(
        &self,
        var_map: impl IntoVariableMap,
    ) -> Result<
        Option<(
            AlgebraicContext,
            RationalPolynomial<AlgebraicExtension<Q>, E>,
        )>,
        String,
    >
    where
        RationalPolynomial<AlgebraicExtension<Q>, E>:
            FromNumeratorAndDenominator<AlgebraicExtension<Q>, AlgebraicExtension<Q>, E>,
    {
        let mut context = self.algebraic_context()?;
        if context.is_trivial() {
            return Ok(None);
        }
        let polynomial = context.to_rational_polynomial(*self, var_map)?;
        Ok(Some((context, polynomial)))
    }

    fn discover_embedding_field(
        &self,
        mut cur: Option<AlgebraicExtension<Q>>,
    ) -> Option<AlgebraicExtension<Q>> {
        match self {
            AtomView::Num(number) => {
                if !AlgebraicContext::coefficient_has_imaginary_part(number.get_coeff_view()) {
                    return cur;
                }

                if let Some(current) = cur {
                    current.with_adjoined_rational_root(
                        &AlgebraicExtension::imaginary_unit_defining_polynomial(),
                        1,
                    )
                } else {
                    Some(AlgebraicExtension::complex(Q))
                }
            }
            AtomView::Var(_) => cur,
            AtomView::Fun(f) => {
                if f.get_symbol() == root() {
                    let (poly, index) = Root::try_from(*self).ok()?.into_parts();
                    if let Some(c) = cur {
                        c.with_adjoined_rational_root(&poly, index)
                    } else {
                        Some(AlgebraicExtension {
                            poly: Arc::new(poly),
                            embedding: index,
                        })
                    }
                } else {
                    cur
                }
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();
                cur = b.discover_embedding_field(cur);
                cur = e.discover_embedding_field(cur);

                if let Ok(r) = Rational::try_from(e)
                    && let Ok(denominator) = u64::try_from(r.denominator())
                    && denominator > 1
                {
                    if let Some(c) = cur {
                        // create min poly and adjoin if needed
                        let base_converted = b.to_algebraic(&c).ok()?;
                        if !c.is_positive_real(&base_converted).ok()? {
                            return None;
                        }

                        let var = c.fresh_variable();
                        let mut poly = MultivariatePolynomial::<_, u16>::new(
                            &c,
                            None,
                            Arc::new(vec![var.clone()]),
                        );
                        poly = poly.variable(&var).unwrap().pow(denominator as usize)
                            - poly.constant(base_converted);

                        let factors = poly.factor();
                        let mut contained_root = false;
                        for (factor, _) in &factors {
                            if factor.degree(0) != 1 {
                                continue;
                            }

                            let candidate = c.neg(&c.div(&factor.get_constant(), &factor.lcoeff()));
                            if c.is_positive_real(&candidate).ok()? {
                                contained_root = true;
                                break;
                            }
                        }

                        if contained_root {
                            cur = Some(c);
                        } else {
                            let mut selected_factor = None;
                            for (factor, _) in factors {
                                let factor = factor.make_monic();
                                let degree = factor.degree(0) as usize;
                                if degree <= 1 {
                                    continue;
                                }

                                if factor.count_positive_real_roots().ok()? > 0 {
                                    if selected_factor.is_some() {
                                        return None;
                                    }
                                    selected_factor = Some((factor, degree));
                                }
                            }

                            let (factor, degree) = selected_factor?;

                            // Among the roots of x^d-base, the positive
                            // root has the greatest real part and is last
                            // in the canonical complex-root ordering.
                            let ext = AlgebraicExtension {
                                poly: Arc::new(factor),
                                embedding: degree - 1,
                            };
                            cur = Some(c.adjoin_with_embedding(&ext, None).0);
                        }
                    } else if let Ok(rat_base) = Rational::try_from(b) {
                        if rat_base.is_negative() || rat_base.is_zero() {
                            return None;
                        }

                        let mut poly = MultivariatePolynomial::<_, u16>::new(
                            &Q,
                            None,
                            Arc::new(vec![PolyVariable::Temporary(0)]),
                        );
                        poly = poly
                            .variable(&PolyVariable::Temporary(0))
                            .unwrap()
                            .pow(denominator as usize)
                            - poly.constant(rat_base);

                        let mut selected = None;
                        for (factor, _) in poly.factor() {
                            let factor = factor.make_monic();
                            let Some(embedding) =
                                AlgebraicContext::positive_real_root_index(&factor)
                            else {
                                continue;
                            };

                            if factor.degree(0) == 1 {
                                continue;
                            }

                            if selected.is_some() {
                                return None;
                            }
                            let ext = AlgebraicExtension {
                                poly: Arc::new(factor),
                                embedding,
                            };
                            selected = Some(ext);
                        }
                        cur = selected;
                    }
                }

                cur
            }
            AtomView::Mul(m) => {
                for a in m {
                    cur = a.discover_embedding_field(cur);
                }
                cur
            }
            AtomView::Add(a) => {
                for a in a {
                    cur = a.discover_embedding_field(cur);
                }
                cur
            }
        }
    }
}

impl<R: EuclideanDomain> AlgebraicExtension<R> {
    /// Construct the degree-one extension `R[t]/(t)`.
    ///
    /// This represents the base field itself while providing the same element
    /// type as a nontrivial algebraic extension.
    pub fn trivial(ring: R) -> AlgebraicExtension<R> {
        let variable = PolyVariable::Temporary(0);
        let prototype =
            MultivariatePolynomial::new(&ring, Some(1), Arc::new(vec![variable.clone()]));
        AlgebraicExtension::new(prototype.variable(&variable).unwrap())
    }

    /// Create a new algebraic extension from a univariate polynomial.
    /// The polynomial should be monic and irreducible.
    ///
    /// The default embedding is root index 0. Use [`from_polynomial_with_embedding`] to specify a different root index.
    pub fn new(poly: MultivariatePolynomial<R, u16>) -> AlgebraicExtension<R> {
        if poly.nvars() == 1 {
            return AlgebraicExtension {
                poly: Arc::new(poly),
                embedding: 0,
            };
        }

        assert_eq!((0..poly.nvars()).filter(|v| poly.degree(*v) > 0).count(), 1);
        let v = (0..poly.nvars()).find(|v| poly.degree(*v) > 0).unwrap();
        let uni = poly.to_univariate_from_univariate(v);

        AlgebraicExtension {
            poly: Arc::new(uni.to_multivariate()),
            embedding: 0,
        }
    }

    /// Create an extension with an explicitly selected root embedding.
    pub fn from_polynomial_with_embedding(
        poly: MultivariatePolynomial<R, u16>,
        root_index: usize,
    ) -> AlgebraicExtension<R> {
        AlgebraicExtension {
            poly: Arc::new(poly),
            embedding: root_index,
        }
    }

    /// Embed a base-ring coefficient as a constant extension element.
    pub fn constant(&self, c: R::Element) -> AlgebraicNumber<R> {
        AlgebraicNumber {
            poly: self.poly.constant(c),
        }
    }

    /// Return the residue class of the defining polynomial's variable.
    pub fn generator(&self) -> AlgebraicNumber<R> {
        self.element_from_polynomial(self.poly.one().mul_exp(&[1]))
    }

    /// Return the polynomial defining this extension.
    pub fn poly(&self) -> &MultivariatePolynomial<R, u16> {
        &self.poly
    }

    /// Return the selected root index of the defining polynomial.
    pub fn embedding(&self) -> usize {
        self.embedding
    }

    /// Map the defining polynomial and its embedding to a finite field.
    pub fn to_finite_field<UField: FiniteFieldWorkspace>(
        &self,
        field: &FiniteField<UField>,
    ) -> AlgebraicExtension<FiniteField<UField>>
    where
        R::Element: ToFiniteField<UField>,
        FiniteField<UField>: FiniteFieldCore<UField>,
    {
        AlgebraicExtension {
            poly: Arc::new(
                self.poly
                    .map_coeff(|c| c.to_finite_field(field), field.clone()),
            ),
            embedding: self.embedding,
        }
    }

    /// Reduce `poly` modulo the defining polynomial.
    ///
    /// Returns an error when `poly` is not univariate in the extension's
    /// generator.
    pub fn try_element_from_polynomial(
        &self,
        poly: MultivariatePolynomial<R, u16>,
    ) -> Result<<Self as Set>::Element, String> {
        if poly.nvars() == 0 {
            let mut new_poly = poly;
            new_poly.set_variables(self.poly.variables().clone());
            new_poly.exponents = vec![0; new_poly.coefficients.len()];

            return Ok(AlgebraicNumber { poly: new_poly });
        }

        if poly.nvars() != 1 {
            return Err(format!(
                "Polynomial has {} variables, expected 1",
                poly.nvars()
            ));
        }

        if poly.get_vars_ref()[0] != self.poly.get_vars_ref()[0] {
            return Err(format!(
                "Polynomial variable {:?} does not match extension variable {:?}",
                poly.get_vars_ref()[0],
                self.poly.get_vars_ref()[0]
            ));
        }

        if poly.degree(0) >= self.poly.degree(0) {
            Ok(AlgebraicNumber {
                poly: poly.quot_rem_univariate_monic(&self.poly).1,
            })
        } else {
            Ok(AlgebraicNumber { poly })
        }
    }

    /// Reduce `poly` modulo the defining polynomial.
    ///
    /// # Panics
    ///
    /// Panics when `poly` is not univariate in the extension's generator.
    pub fn element_from_polynomial(
        &self,
        poly: MultivariatePolynomial<R, u16>,
    ) -> <Self as Set>::Element {
        self.try_element_from_polynomial(poly).unwrap()
    }

    /// Get a variable that is not already used in the polynomial.
    pub(crate) fn fresh_variable(&self) -> PolyVariable {
        match self.poly.get_vars_ref()[0] {
            PolyVariable::Temporary(i) => PolyVariable::Temporary(i + 1),
            _ => PolyVariable::Temporary(0),
        }
    }
}

impl<R: EuclideanDomain> AlgebraicExtension<R> {
    /// Create a new algebraic extension `R(i)`.
    /// This ring can be used to convert expressions with complex coefficients
    /// to polynomials.
    ///
    /// # Examples
    ///
    /// Creating Gaussian rationals:
    /// ```rust
    /// use symbolica::prelude::*;
    /// let Q_i = AlgebraicExtension::complex(Q);
    /// let poly = parse!("(-1+6𝑖)*x+(4+2𝑖)*x^2+3𝑖").to_polynomial::<_, u8>(&Q_i, None);
    /// assert_eq!(poly.factor().len(), 3);
    /// ```
    pub fn complex(ring: R) -> Self {
        let poly = MultivariatePolynomial::new(
            &ring,
            Some(2),
            Arc::new(vec![symbol!(Atom::I_STR).into()]),
        );

        let poly = poly.monomial(ring.one(), vec![2]) + poly.constant(ring.one());

        AlgebraicExtension {
            poly: Arc::new(poly),
            embedding: 1,
        }
    }

    /// Converts an algebraic number to an atom.
    /// For a version that simplifies the representation, see [AlgebraicExtension::element_to_atom_simplified].
    pub fn element_to_atom(&self, element: &<Self as Set>::Element) -> Atom
    where
        R::Element: Into<crate::coefficient::Coefficient>,
    {
        let mut p = self.poly.as_ref().clone();
        let variable = p.get_vars_ref()[0].clone();
        if variable != PolyVariable::Symbol(root_var()) {
            p.rename_variable(&variable, &PolyVariable::Symbol(root_var()));
        }
        let root = p.to_expression().root(self.embedding);

        // TODO: try simplification here

        let mut res = Atom::Zero;
        for t in element.poly() {
            let coefficient = t.coefficient.clone().into();
            if t.exponents[0] == 0 {
                res += coefficient;
            } else {
                res += root.pow(t.exponents[0]) * coefficient;
            }
        }
        return res;
    }
}

impl AlgebraicExtension<Q> {
    fn imaginary_unit_defining_polynomial() -> MultivariatePolynomial<Q, u16> {
        let variable = PolyVariable::Temporary(0);
        let mut polynomial =
            MultivariatePolynomial::new(&Q, Some(2), Arc::new(vec![variable.clone()]));
        polynomial = polynomial.variable(&variable).unwrap().pow(2) + polynomial.one();
        polynomial
    }

    /// Converts an algebraic number to an atom, simplifying the representation if possible.
    /// For a version that does not simplify, see [AlgebraicExtension::element_to_atom].
    pub fn element_to_atom_simplified(&self, element: &AlgebraicNumber<Q>) -> Atom {
        if element.poly.is_constant() {
            return element.poly.get_constant().into();
        }

        let s = self.simplify(element);
        let mut p = s.poly.as_ref().clone();
        let variable = p.get_vars_ref()[0].clone();
        if variable != PolyVariable::Symbol(root_var()) {
            p.rename_variable(&variable, &PolyVariable::Symbol(root_var()));
        }

        p.to_expression().root(s.embedding)
    }
}

impl<R: Ring> std::fmt::Debug for AlgebraicExtension<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, " % {:?}", self.poly)
    }
}

impl<R: Ring> std::fmt::Display for AlgebraicExtension<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, " % {}", self.poly)
    }
}

/// A number in an algebraic number field.
///
/// # Examples
///
/// ```
/// use symbolica::prelude::*;
///
/// let extension = AlgebraicExtension::new(parse!("x^2-2").to_polynomial(&Q, None));
/// let sqrt_2 = extension.element_from_polynomial(parse!("x").to_polynomial::<_, u16>(&Q, None));
///
/// let square = extension.mul(&sqrt_2, &sqrt_2);
/// assert_eq!(
///      square,
///      extension.element_from_polynomial(parse!("2").to_polynomial(&Q, None))
/// );
///```
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct AlgebraicNumber<R: Ring> {
    pub(crate) poly: MultivariatePolynomial<R, u16>,
}

impl<R: Ring> InternalOrdering for AlgebraicNumber<R> {
    fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.poly.internal_cmp(&other.poly)
    }
}

impl<R: Ring> std::fmt::Debug for AlgebraicNumber<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.poly)
    }
}

impl<R: Ring> std::fmt::Display for AlgebraicNumber<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.poly)
    }
}

impl<R: Ring> AlgebraicNumber<R> {
    /// Multiply the polynomial representation by a base-ring coefficient.
    pub fn scale(self, coefficient: R::Element) -> Self {
        AlgebraicNumber {
            poly: self.poly.mul_coeff(coefficient),
        }
    }

    /// Map every coefficient of this number to a finite field.
    pub fn to_finite_field<UField: FiniteFieldWorkspace>(
        &self,
        field: &FiniteField<UField>,
    ) -> AlgebraicNumber<FiniteField<UField>>
    where
        R::Element: ToFiniteField<UField>,
        FiniteField<UField>: FiniteFieldCore<UField>,
    {
        AlgebraicNumber {
            poly: self
                .poly
                .map_coeff(|c| c.to_finite_field(field), field.clone()),
        }
    }

    /// Consume this number and return its reduced polynomial representation.
    pub fn into_polynomial(self) -> MultivariatePolynomial<R, u16> {
        self.poly
    }

    /// Return the reduced polynomial representation of this number.
    pub fn poly(&self) -> &MultivariatePolynomial<R, u16> {
        &self.poly
    }
}

impl<R: EuclideanDomain> Set for AlgebraicExtension<R> {
    type Element = AlgebraicNumber<R>;

    fn size(&self) -> Option<Integer> {
        self.poly
            .ring()
            .size()
            .map(|s| s.pow(self.poly.degree(0) as u64))
    }
}

impl<R: EuclideanDomain> RingOps<AlgebraicNumber<R>> for AlgebraicExtension<R> {
    fn add(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        AlgebraicNumber {
            poly: a.poly + b.poly,
        }
    }

    fn sub(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        AlgebraicNumber {
            poly: a.poly - b.poly,
        }
    }

    fn mul(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        AlgebraicNumber {
            poly: (&a.poly * &b.poly).quot_rem_univariate_monic(&self.poly).1,
        }
    }

    fn add_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a = self.add(&*a, &b);
    }

    fn sub_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a = self.sub(&*a, &b);
    }

    fn mul_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a = self.mul(&*a, &b);
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: Self::Element, c: Self::Element) {
        *a = self.add(&*a, &self.mul(b, c));
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: Self::Element, c: Self::Element) {
        *a = self.sub(&*a, &self.mul(b, c));
    }

    fn neg(&self, a: Self::Element) -> Self::Element {
        AlgebraicNumber { poly: -a.poly }
    }
}

impl<R: EuclideanDomain> RingOps<&AlgebraicNumber<R>> for AlgebraicExtension<R> {
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        AlgebraicNumber {
            poly: &a.poly + &b.poly,
        }
    }

    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        AlgebraicNumber {
            poly: &a.poly - &b.poly,
        }
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        AlgebraicNumber {
            poly: (&a.poly * &b.poly).quot_rem_univariate_monic(&self.poly).1,
        }
    }

    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.add(&*a, b);
    }

    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.sub(&*a, b);
    }

    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(&*a, b);
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        *a = self.add(&*a, &self.mul(b, c));
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        *a = self.sub(&*a, &self.mul(b, c));
    }

    fn neg(&self, a: &Self::Element) -> Self::Element {
        AlgebraicNumber {
            poly: -a.poly.clone(),
        }
    }
}

impl<R: EuclideanDomain> Ring for AlgebraicExtension<R> {
    fn zero(&self) -> Self::Element {
        AlgebraicNumber {
            poly: self.poly.zero(),
        }
    }

    fn one(&self) -> Self::Element {
        AlgebraicNumber {
            poly: self.poly.one(),
        }
    }

    fn nth(&self, n: Integer) -> Self::Element {
        AlgebraicNumber {
            poly: self.poly.constant(self.poly.ring().nth(n)),
        }
    }

    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element {
        let mut result = self.one();
        for _ in 0..e {
            result = self.mul(&result, b);
        }
        result
    }

    fn is_zero(&self, a: &Self::Element) -> bool {
        a.poly.is_zero()
    }

    fn is_one(&self, a: &Self::Element) -> bool {
        a.poly.is_one()
    }

    fn one_is_gcd_unit() -> bool {
        true
    }

    fn characteristic(&self) -> Integer {
        self.poly.ring().characteristic()
    }

    fn try_inv(&self, a: &Self::Element) -> Option<Self::Element> {
        self.try_div(&self.one(), a)
    }

    fn try_div(&self, a: &Self::Element, b: &Self::Element) -> Option<Self::Element> {
        if self.is_zero(b) {
            return None;
        }

        // solve the linear system (c_0 + c_1*x + c_(d-1)*x^(d-1)) * b % self = a
        // TODO: use the inverse if R is a field (requires specialization)
        let d = self.poly.degree(0) as usize;
        let mut m = vec![self.poly.ring().zero(); d * d];

        let mut f = self.one();

        for e in 0..d {
            let c = self.mul(b, &f);
            for monomial in &c.poly {
                m[monomial.exponents[0] as usize * d + e] = monomial.coefficient.clone();
            }
            f.poly.exponents[0] += 1;
        }

        let mut rhs = vec![self.poly.ring().zero(); d];
        for monomial in &a.poly {
            rhs[monomial.exponents[0] as usize] = monomial.coefficient.clone();
        }

        let m = Matrix::from_linear(m, d as u32, d as u32, self.poly.ring().clone()).unwrap();
        let rhs = Matrix::new_vec(rhs, self.poly.ring().clone());

        if let Ok(s) = m.solve_fraction_free(&rhs) {
            let mut new_poly = self.poly.zero();
            for (p, c) in s.into_vec().into_iter().enumerate() {
                new_poly = &new_poly + &new_poly.monomial(c, vec![p as u16]);
            }

            Some(AlgebraicNumber { poly: new_poly })
        } else {
            None
        }
    }

    /// Sample a polynomial.
    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        let coeffs: Vec<_> = (0..self.poly.degree(0))
            .map(|_| self.poly.ring().sample(rng, range))
            .collect();

        let mut poly = self.poly.zero_with_capacity(coeffs.len());
        let mut exp = vec![0];
        for (i, c) in coeffs.into_iter().enumerate() {
            exp[0] = i as u16;
            poly.append_monomial(c, &exp);
        }

        AlgebraicNumber { poly }
    }

    fn format<W: std::fmt::Write>(
        &self,
        element: &Self::Element,
        opts: &crate::printer::PrintOptions,
        state: crate::printer::PrintState,
        f: &mut W,
    ) -> Result<bool, std::fmt::Error> {
        element.poly.format(opts, state, f)
    }
}

impl<R: Field> EuclideanDomain for AlgebraicExtension<R> {
    fn rem(&self, _a: &Self::Element, _b: &Self::Element) -> Self::Element {
        // TODO: due to the remainder requiring an inverse, we need to have R be a field
        // instead of a Euclidean domain. Relax this condition by doing a pseudo-division
        // to get the case where rem = 0 without requiring a field?
        self.zero()
    }

    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        (self.div(a, b), self.zero())
    }

    fn gcd(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        let c1 = a.poly.content();
        let c2 = b.poly.content();
        AlgebraicNumber {
            poly: a.poly.constant(a.poly.ring().gcd(&c1, &c2)),
        }
    }
}

impl<R: Field> Field for AlgebraicExtension<R> {
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.mul(a, &self.inv(b))
    }

    fn div_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.div(a, b);
    }

    fn inv(&self, a: &Self::Element) -> Self::Element {
        if a.poly.is_zero() {
            panic!("Division by zero");
        }

        AlgebraicNumber {
            poly: a.poly.eea_univariate(&self.poly).1,
        }
    }
}

impl<R: Ring> std::fmt::Debug for AlgebraicQuotient<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, " quotient {:?}", self.poly)
    }
}

impl<R: Ring> std::fmt::Display for AlgebraicQuotient<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, " quotient {}", self.poly)
    }
}

impl<R: EuclideanDomain> AlgebraicQuotient<R> {
    /// Construct `R[t]/(f)`.
    ///
    /// The polynomial must be monic and irreducible if the quotient is to be
    /// used as a field. This is not checked.
    pub fn new(poly: MultivariatePolynomial<R, u16>) -> Self {
        let extension = AlgebraicExtension::new(poly);
        Self {
            poly: extension.poly,
        }
    }

    /// Construct the degree-one quotient `R[t]/(t)`.
    pub fn trivial(ring: R) -> Self {
        let extension = AlgebraicExtension::trivial(ring);
        Self {
            poly: extension.poly,
        }
    }

    pub(crate) fn as_extension(&self) -> AlgebraicExtension<R> {
        AlgebraicExtension {
            poly: self.poly.clone(),
            embedding: 0,
        }
    }

    /// Return the polynomial defining this formal quotient.
    pub fn poly(&self) -> &MultivariatePolynomial<R, u16> {
        &self.poly
    }

    /// Embed a base-ring coefficient as a constant quotient element.
    pub fn constant(&self, coefficient: R::Element) -> AlgebraicNumber<R> {
        AlgebraicNumber {
            poly: self.poly.constant(coefficient),
        }
    }

    /// Return the residue class of the defining polynomial's variable.
    pub fn generator(&self) -> AlgebraicNumber<R> {
        self.element_from_polynomial(self.poly.one().mul_exp(&[1]))
    }

    /// Reduce `polynomial` modulo the defining polynomial.
    pub fn try_element_from_polynomial(
        &self,
        polynomial: MultivariatePolynomial<R, u16>,
    ) -> Result<AlgebraicNumber<R>, String> {
        self.as_extension().try_element_from_polynomial(polynomial)
    }

    /// Reduce `polynomial` modulo the defining polynomial.
    ///
    /// # Panics
    ///
    /// Panics when `polynomial` is incompatible with this quotient.
    pub fn element_from_polynomial(
        &self,
        polynomial: MultivariatePolynomial<R, u16>,
    ) -> AlgebraicNumber<R> {
        self.try_element_from_polynomial(polynomial).unwrap()
    }

    pub(crate) fn fresh_variable(&self) -> PolyVariable {
        match self.poly.get_vars_ref()[0] {
            PolyVariable::Temporary(index) => PolyVariable::Temporary(index + 1),
            _ => PolyVariable::Temporary(0),
        }
    }
}

impl<R: EuclideanDomain> Set for AlgebraicQuotient<R> {
    type Element = AlgebraicNumber<R>;

    fn size(&self) -> Option<Integer> {
        self.as_extension().size()
    }
}

impl<R: EuclideanDomain> RingOps<AlgebraicNumber<R>> for AlgebraicQuotient<R> {
    fn add(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        self.as_extension().add(a, b)
    }

    fn sub(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        self.as_extension().sub(a, b)
    }

    fn mul(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        self.as_extension().mul(a, b)
    }

    fn add_assign(&self, a: &mut Self::Element, b: Self::Element) {
        self.as_extension().add_assign(a, b)
    }

    fn sub_assign(&self, a: &mut Self::Element, b: Self::Element) {
        self.as_extension().sub_assign(a, b)
    }

    fn mul_assign(&self, a: &mut Self::Element, b: Self::Element) {
        self.as_extension().mul_assign(a, b)
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: Self::Element, c: Self::Element) {
        self.as_extension().add_mul_assign(a, b, c)
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: Self::Element, c: Self::Element) {
        self.as_extension().sub_mul_assign(a, b, c)
    }

    fn neg(&self, a: Self::Element) -> Self::Element {
        self.as_extension().neg(a)
    }
}

impl<R: EuclideanDomain> RingOps<&AlgebraicNumber<R>> for AlgebraicQuotient<R> {
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.as_extension().add(a, b)
    }

    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.as_extension().sub(a, b)
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.as_extension().mul(a, b)
    }

    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        self.as_extension().add_assign(a, b)
    }

    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        self.as_extension().sub_assign(a, b)
    }

    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        self.as_extension().mul_assign(a, b)
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.as_extension().add_mul_assign(a, b, c)
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.as_extension().sub_mul_assign(a, b, c)
    }

    fn neg(&self, a: &Self::Element) -> Self::Element {
        self.as_extension().neg(a)
    }
}

impl<R: EuclideanDomain> Ring for AlgebraicQuotient<R> {
    fn zero(&self) -> Self::Element {
        self.as_extension().zero()
    }

    fn one(&self) -> Self::Element {
        self.as_extension().one()
    }

    fn nth(&self, n: Integer) -> Self::Element {
        self.as_extension().nth(n)
    }

    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element {
        self.as_extension().pow(b, e)
    }

    fn is_zero(&self, a: &Self::Element) -> bool {
        self.as_extension().is_zero(a)
    }

    fn is_one(&self, a: &Self::Element) -> bool {
        self.as_extension().is_one(a)
    }

    fn one_is_gcd_unit() -> bool {
        true
    }

    fn characteristic(&self) -> Integer {
        self.poly.ring().characteristic()
    }

    fn try_inv(&self, a: &Self::Element) -> Option<Self::Element> {
        self.as_extension().try_inv(a)
    }

    fn try_div(&self, a: &Self::Element, b: &Self::Element) -> Option<Self::Element> {
        self.as_extension().try_div(a, b)
    }

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        self.as_extension().sample(rng, range)
    }

    fn format<W: std::fmt::Write>(
        &self,
        element: &Self::Element,
        opts: &crate::printer::PrintOptions,
        state: crate::printer::PrintState,
        f: &mut W,
    ) -> Result<bool, std::fmt::Error> {
        element.poly.format(opts, state, f)
    }
}

impl<R: Field> EuclideanDomain for AlgebraicQuotient<R> {
    fn rem(&self, _a: &Self::Element, _b: &Self::Element) -> Self::Element {
        self.zero()
    }

    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        (self.div(a, b), self.zero())
    }

    fn gcd(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.as_extension().gcd(a, b)
    }
}

impl<R: Field> Field for AlgebraicQuotient<R> {
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.as_extension().div(a, b)
    }

    fn div_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        self.as_extension().div_assign(a, b)
    }

    fn inv(&self, a: &Self::Element) -> Self::Element {
        self.as_extension().inv(a)
    }
}

impl<R: Field + PolynomialGCD<u16>> AlgebraicQuotient<R> {
    /// Adjoin a formal root and collapse the resulting tower to one primitive
    /// quotient. The polynomial must be monic and irreducible.
    pub fn adjoin_formal(
        &self,
        polynomial: &MultivariatePolynomial<AlgebraicQuotient<R>>,
        new_symbol: Option<PolyVariable>,
    ) -> (AlgebraicQuotient<R>, AlgebraicNumber<R>, AlgebraicNumber<R>) {
        let (field, old_generator, new_generator, _) =
            self.adjoin_formal_with_shift(polynomial, new_symbol);
        (field, old_generator, new_generator)
    }

    pub(crate) fn adjoin_formal_with_shift(
        &self,
        polynomial: &MultivariatePolynomial<AlgebraicQuotient<R>>,
        new_symbol: Option<PolyVariable>,
    ) -> (
        AlgebraicQuotient<R>,
        AlgebraicNumber<R>,
        AlgebraicNumber<R>,
        usize,
    ) {
        assert_eq!(self, polynomial.ring());
        let extension = self.as_extension();
        let polynomial = polynomial.map_coeff(|coefficient| coefficient.clone(), extension.clone());
        let (field, old_generator, new_generator, shift) =
            extension.adjoin_formal_with_shift(&polynomial, new_symbol);
        (
            AlgebraicQuotient { poly: field.poly },
            old_generator,
            new_generator,
            shift,
        )
    }
}

impl<R: Field> AlgebraicExtension<R> {
    /// Compute the monic minimal polynomial of `element` over the base field `R`.
    ///
    /// The first linear dependence between successive powers of `element` in
    /// this finite-dimensional extension determines the polynomial. This
    /// method returns only that polynomial; when `R` has analytic embeddings,
    /// the caller must separately identify which root represents `element`.
    pub fn minimal_polynomial_of_element(
        &self,
        element: &AlgebraicNumber<R>,
    ) -> MultivariatePolynomial<R, u16> {
        let mut polys = vec![];

        let mut x_i = self.one();
        for _ in 0..=self.poly.degree(0) {
            x_i = self.mul(&x_i, element);
            polys.push(x_i.clone());

            // solve system c_0 + c_1 x + c_i x^2 + ... + x^i = 0
            let ncols = self.poly.degree(0).to_u32() as usize;

            let mut m = vec![self.poly.ring().zero(); polys.len() * ncols];
            for (row, p) in m.chunks_mut(ncols).zip(&polys) {
                for monomial in &p.poly {
                    row[monomial.exponents[0].to_u32() as usize] = monomial.coefficient.clone();
                }
            }

            let mut rhs = m.split_off((polys.len() - 1) * ncols);
            for e in &mut rhs {
                *e = self.poly.ring().neg(&*e);
            }

            if polys.len() == 1 {
                continue;
            }

            // TODO: recycle matrix
            let mat = Matrix::from_linear(
                m,
                (polys.len() - 1) as u32,
                ncols as u32,
                self.poly.ring().clone(),
            )
            .unwrap()
            .into_transposed();

            let rhs = Matrix::new_vec(rhs, self.poly.ring().clone());

            if let Ok(s) = mat.solve(&rhs) {
                let mut res = s.into_vec();
                res.push(self.poly.ring().one());
                let mut new_poly = self.poly.zero();
                for (p, c) in res.into_iter().enumerate() {
                    new_poly = &new_poly + &new_poly.monomial(c, vec![p as u16]);
                }

                return new_poly;
            }
        }

        unreachable!("Could not compute the minimal polynomial of an algebraic element");
    }
}

impl AlgebraicExtension<Q> {
    /// Create a new minimal field extension that has the algebraic number `x`
    /// as its generator, preserving the value selected by this field's
    /// embedding.
    pub fn simplify(&self, x: &AlgebraicNumber<Q>) -> AlgebraicExtension<Q> {
        let polynomial = self.minimal_polynomial_of_element(x);
        let embedding = self
            .root_index_of_element(x, &polynomial)
            .unwrap_or_else(|error| {
                panic!(
                    "Could not determine the embedding of a simplified algebraic number: {error}"
                )
            });
        AlgebraicExtension::from_polynomial_with_embedding(polynomial, embedding)
    }
}

impl<R: Field + PolynomialGCD<u16>> AlgebraicExtension<R> {
    /// Formally adjoin a root of a monic irreducible polynomial.
    ///
    /// This performs the primitive-element construction without selecting an
    /// analytic embedding. Irreducibility is required but not checked.
    pub fn adjoin_formal(
        &self,
        b: &MultivariatePolynomial<AlgebraicExtension<R>>,
        new_symbol: Option<PolyVariable>,
    ) -> (
        AlgebraicExtension<R>,
        AlgebraicNumber<R>,
        AlgebraicNumber<R>,
    ) {
        let (field, old_generator, new_generator, _) = self.adjoin_formal_with_shift(b, new_symbol);
        (field, old_generator, new_generator)
    }

    pub(crate) fn adjoin_formal_with_shift(
        &self,
        b: &MultivariatePolynomial<AlgebraicExtension<R>>,
        new_symbol: Option<PolyVariable>,
    ) -> (
        AlgebraicExtension<R>,
        AlgebraicNumber<R>,
        AlgebraicNumber<R>,
        usize,
    ) {
        assert_eq!(self, b.ring());

        let (_, s, g, r) = b.norm_with_shift_data();
        self.adjoin_formal_from_norm(s, g, r, new_symbol)
    }

    fn adjoin_formal_from_norm(
        &self,
        s: usize,
        g: MultivariatePolynomial<R>,
        r: MultivariatePolynomial<R>,
        new_symbol: Option<PolyVariable>,
    ) -> (
        AlgebraicExtension<R>,
        AlgebraicNumber<R>,
        AlgebraicNumber<R>,
        usize,
    ) {
        let mut field = AlgebraicExtension::new(r);
        let mut shifted = g.to_number_field(&field);
        let mut old_minimal_polynomial = self.poly.to_number_field(&field);

        shifted.unify_variables(&mut old_minimal_polynomial);
        let gcd = shifted.univariate_gcd(&old_minimal_polynomial);

        let mut old_generator = field.neg(&field.div(&gcd.get_constant(), &gcd.lcoeff()));
        let primitive_generator = field.generator();
        let mut new_generator = field.sub(
            &primitive_generator,
            &field.mul(&old_generator, &field.nth(s.into())),
        );

        if let Some(variable) = &new_symbol {
            let old_variable = &field.poly.get_vars_ref()[0];
            old_generator.poly.rename_variable(old_variable, variable);
            new_generator.poly.rename_variable(old_variable, variable);

            let mut new_polynomial = field.poly.as_ref().clone();
            new_polynomial.rename_variable(old_variable, variable);
            field = AlgebraicExtension {
                poly: Arc::new(new_polynomial),
                embedding: 0,
            };
        }

        (field, old_generator, new_generator, s)
    }
}

impl<R: Field + PolynomialGCD<u16>> AlgebraicExtension<R> {
    /// Adjoin the current algebraic extension `R[a]` with `b`, whose minimal polynomial
    /// is `R[a][b]` and form `R[b]`. Also return the new representation of `a` and `b`.
    ///
    /// `b` must be irreducible over `R` and `R[a]`; this is not checked.
    ///
    /// If `new_symbol` is provided, the variable of the new extension will be renamed to it.
    /// Otherwise, the variable of the new extension will be the same as that of `b`.
    pub fn adjoin(
        &self,
        b: &MultivariatePolynomial<AlgebraicExtension<R>>,
        new_symbol: Option<PolyVariable>,
    ) -> (
        AlgebraicExtension<R>,
        <AlgebraicExtension<R> as Set>::Element,
        <AlgebraicExtension<R> as Set>::Element,
    )
    where
        AlgebraicExtension<R>: PolynomialGCD<u16> + Ring<Element = AlgebraicNumber<R>>,
        MultivariatePolynomial<R>: Factorize,
        MultivariatePolynomial<AlgebraicExtension<R>>: Factorize,
    {
        assert_eq!(self, b.ring());

        let (_, shift, shifted, norm) = b.norm_with_shift_data();
        debug_assert!(norm.is_irreducible());
        let (field, old_generator, new_generator, _) =
            self.adjoin_formal_from_norm(shift, shifted, norm, new_symbol);
        (field, old_generator, new_generator)
    }
}

impl<R: Field + PolynomialGCD<E>, E: PositiveExponent>
    MultivariatePolynomial<AlgebraicExtension<R>, E>
{
    /// Get the norm of a non-constant square-free polynomial `f` in the algebraic number field.
    pub fn norm(&self) -> MultivariatePolynomial<R, E> {
        self.norm_with_shift_data().3
    }

    /// Get the norm of a non-constant square-free polynomial `f` in the algebraic number field.
    /// Returns `(v, s, g, r)` where `v` is the shifted variable, `s` is the number of steps,
    /// `g` is the shifted polynomial and `r` is the norm.
    pub(crate) fn norm_with_shift_data(
        &self,
    ) -> (
        usize,
        usize,
        MultivariatePolynomial<R, E>,
        MultivariatePolynomial<R, E>,
    ) {
        assert!(!self.is_constant());

        let f = self.from_number_field();

        let alpha = f
            .get_vars()
            .iter()
            .position(|x| x == &self.ring().poly.variables()[0])
            .unwrap();

        let mut poly = f.zero();
        let mut exp = vec![E::zero(); f.nvars()];
        for x in self.ring().poly.into_iter() {
            exp[alpha] = E::from_u32(x.exponents[0] as u32);
            poly.append_monomial(x.coefficient.clone(), &exp);
        }

        let poly_uni = poly.to_univariate(alpha);

        let mut s = 0;
        loop {
            for v in 0..f.nvars() {
                if v == alpha || f.degree(v) == E::zero() {
                    continue;
                }

                // construct f(x-s*a)
                let alpha_poly = f.variable(&self.get_vars_ref()[v]).unwrap()
                    - f.variable(&self.ring().poly.variables()[0]).unwrap()
                        * &f.constant(f.ring().nth(s.into()));
                let g_multi = f.clone().replace_with_poly(v, &alpha_poly);
                let g_uni = g_multi.to_univariate(alpha);

                let r = g_uni.resultant_prs(&poly_uni);

                let d = r.derivative(v);
                if r.univariate_gcd(&d).is_constant() {
                    return (v, s, g_multi, r);
                }
            }

            s += 1;
        }
    }
}

impl AlgebraicExtension<Q> {
    pub(crate) fn is_positive_real(&self, element: &AlgebraicNumber<Q>) -> Result<bool, String> {
        if self.is_zero(element) {
            return Ok(false);
        }
        if element.poly.is_constant() {
            return Ok(!element.poly.get_constant().is_negative());
        }

        let poly = self.poly.to_univariate_from_univariate(0);
        let mut primitive_root = poly.root(self.embedding).unwrap();
        let primitive_location = primitive_root.classify_location();

        if !matches!(primitive_location, RootLocation::Real | RootLocation::Zero) {
            let minimal_field = self.simplify(element);
            let element_embedding = minimal_field.embedding;
            let minimal_polynomial = minimal_field.poly.to_univariate_from_univariate(0);
            let mut root = minimal_polynomial.root(element_embedding).unwrap();
            let location = root.classify_location();
            if !matches!(location, RootLocation::Real | RootLocation::Zero) {
                return Ok(false);
            }
            return Ok(root.is_positive_real());
        }

        element
            .poly
            .to_univariate_from_univariate(0)
            .has_positive_real_part_at(&primitive_root)
            .map_err(|_| format!("Could not determine the sign of {} in {}", element, self))
    }

    /// Determine the sign of the real part at this field's embedding without
    /// first constructing a minimal polynomial for `element`.
    ///
    /// Returns an error if interval refinement cannot certify the sign of the
    /// real part.
    pub(crate) fn has_positive_real_part(
        &self,
        element: &AlgebraicNumber<Q>,
    ) -> Result<bool, String> {
        if element.poly.is_constant() {
            let constant = element.poly.get_constant();
            return Ok(!constant.is_negative() && !constant.is_zero());
        }

        let polynomial = self.poly.to_univariate_from_univariate(0);
        let root = polynomial.root(self.embedding).unwrap();
        element
            .poly
            .to_univariate_from_univariate(0)
            .has_positive_real_part_at(&root)
            .map_err(|_| {
                format!(
                    "Could not determine the sign of the real part of {} in {}",
                    element, self
                )
            })
    }

    fn root_index_of_element(
        &self,
        element: &AlgebraicNumber<Q>,
        polynomial: &MultivariatePolynomial<Q, u16>,
    ) -> Result<usize, String> {
        let extension_polynomial = self.poly.to_univariate_from_univariate(0);
        let polynomial = polynomial.to_univariate_from_univariate(0);
        let extension_root = extension_polynomial.root(self.embedding).unwrap();
        let mut roots = polynomial
            .isolate_roots()
            .into_iter()
            .map(|(root, _)| root)
            .collect::<Vec<_>>();
        let element_polynomial = element.poly.to_univariate_from_univariate(0);
        extension_root
            .matching_roots(Some(&element_polynomial), &mut roots, None, 1)
            .map(|matches| matches[0])
            .map_err(|_| format!("Could not identify {} as a root of {}", element, polynomial))
    }

    fn embedded_rational_root(
        &self,
        polynomial: &MultivariatePolynomial<Q, u16>,
        embedding: usize,
    ) -> Result<AlgebraicNumber<Q>, String> {
        let mut polynomial_over_self = polynomial.clone();
        polynomial_over_self.rename_variable(&polynomial.get_vars_ref()[0], &self.fresh_variable());
        let polynomial_over_self = polynomial_over_self.to_number_field(self);
        let mut selected = None;

        for (factor, _) in polynomial_over_self.factor() {
            if factor.degree(0) != 1 {
                continue;
            }

            let candidate = self.neg(&self.div(&factor.get_constant(), &factor.lcoeff()));
            if self.root_index_of_element(&candidate, polynomial)? == embedding {
                if selected.is_some() {
                    return Err(format!(
                        "More than one field element matches root {} of {}",
                        embedding, polynomial
                    ));
                }
                selected = Some(candidate);
            }
        }

        selected.ok_or_else(|| {
            format!(
                "The current extension does not contain root {} of {}",
                embedding, polynomial
            )
        })
    }

    fn with_adjoined_rational_root(
        &self,
        polynomial: &MultivariatePolynomial<Q, u16>,
        embedding: usize,
    ) -> Option<Self> {
        if self.embedded_rational_root(polynomial, embedding).is_ok() {
            return Some(self.clone());
        }

        let mut polynomial_over_self = polynomial.clone();
        polynomial_over_self.rename_variable(&polynomial.get_vars_ref()[0], &self.fresh_variable());
        let polynomial_over_self = polynomial_over_self.to_number_field(self);

        for (factor, _) in polynomial_over_self.factor() {
            let degree = factor.degree(0) as usize;
            if degree <= 1 {
                continue;
            }

            let (extensions, _, generator) = self.adjoin_with_all_embeddings(&factor, None);
            for extension in extensions {
                if extension
                    .root_index_of_element(&generator, polynomial)
                    .ok()?
                    == embedding
                {
                    return Some(extension);
                }
            }
        }

        None
    }

    pub(crate) fn imaginary_unit(&self) -> Result<AlgebraicNumber<Q>, String> {
        self.embedded_rational_root(&Self::imaginary_unit_defining_polynomial(), 1)
    }

    /// Adjoin the embedded extension `self[b]` and preserve the selected
    /// embeddings of both `self` and `b`.
    ///
    /// The result is represented as a simple extension over `Q`. Its embedding
    /// is the root of the primitive polynomial for which the returned
    /// representations of the old and new generators have the requested
    /// embeddings.
    pub fn adjoin_with_embedding(
        &self,
        b: &AlgebraicExtension<AlgebraicExtension<Q>>,
        new_symbol: Option<PolyVariable>,
    ) -> (
        AlgebraicExtension<Q>,
        <AlgebraicExtension<Q> as Set>::Element,
        <AlgebraicExtension<Q> as Set>::Element,
    ) {
        let (field, old_generator, new_generator, _) =
            self.adjoin_with_embedding_and_generator_field(b, new_symbol);
        (field, old_generator, new_generator)
    }

    /// Adjoin the selected root and also return the minimal embedded field of
    /// the new generator, reusing data computed while ordering the adjunction.
    pub(crate) fn adjoin_with_embedding_and_generator_field(
        &self,
        b: &AlgebraicExtension<AlgebraicExtension<Q>>,
        new_symbol: Option<PolyVariable>,
    ) -> (
        AlgebraicExtension<Q>,
        <AlgebraicExtension<Q> as Set>::Element,
        <AlgebraicExtension<Q> as Set>::Element,
        AlgebraicExtension<Q>,
    ) {
        let embedding = b.embedding;
        let (
            mut extensions,
            old_generator,
            new_generator,
            new_generator_minimal_poly,
            mut new_generator_embeddings,
        ) = self.adjoin_with_all_embeddings_and_generator_data(&b.poly, new_symbol);
        assert!(
            embedding < extensions.len(),
            "Embedding index {} is out of bounds for polynomial of degree {}",
            embedding,
            extensions.len()
        );
        let new_generator_embedding = new_generator_embeddings.swap_remove(embedding);
        (
            extensions.swap_remove(embedding),
            old_generator,
            new_generator,
            AlgebraicExtension::from_polynomial_with_embedding(
                new_generator_minimal_poly,
                new_generator_embedding,
            ),
        )
    }

    /// Adjoin every embedding of an irreducible polynomial over `self`.
    ///
    /// The expensive primitive-element construction is performed once. The
    /// returned fields differ only in their selected embedding and are ordered
    /// by the canonical complex ordering of the new generator over the
    /// selected embedding of `self`.
    pub(crate) fn adjoin_with_all_embeddings(
        &self,
        polynomial: &MultivariatePolynomial<AlgebraicExtension<Q>, u16>,
        new_symbol: Option<PolyVariable>,
    ) -> (
        Vec<AlgebraicExtension<Q>>,
        <AlgebraicExtension<Q> as Set>::Element,
        <AlgebraicExtension<Q> as Set>::Element,
    ) {
        let (extensions, old_generator, new_generator, _, _) =
            self.adjoin_with_all_embeddings_and_generator_data(polynomial, new_symbol);
        (extensions, old_generator, new_generator)
    }

    fn adjoin_with_all_embeddings_and_generator_data(
        &self,
        polynomial: &MultivariatePolynomial<AlgebraicExtension<Q>, u16>,
        new_symbol: Option<PolyVariable>,
    ) -> (
        Vec<AlgebraicExtension<Q>>,
        <AlgebraicExtension<Q> as Set>::Element,
        <AlgebraicExtension<Q> as Set>::Element,
        MultivariatePolynomial<Q, u16>,
        Vec<usize>,
    ) {
        assert_eq!(
            self,
            polynomial.ring(),
            "The base field of the adjoined extension does not match"
        );

        let extension_degree = polynomial.degree(0) as usize;

        if self.poly.degree(0) == 1 {
            let mut rational_polynomial = polynomial.map_coeff(
                |coefficient| {
                    assert!(
                        coefficient.poly.is_constant(),
                        "an element of a degree-one extension must be constant"
                    );
                    coefficient.poly.get_constant()
                },
                Q,
            );
            if let Some(new_symbol) = new_symbol {
                let active_variable = (0..rational_polynomial.nvars())
                    .find(|&variable| rational_polynomial.degree(variable) > 0)
                    .expect("the adjoined polynomial must be non-constant");
                let old_symbol = rational_polynomial.get_vars_ref()[active_variable].clone();
                rational_polynomial.rename_variable(&old_symbol, &new_symbol);
            }

            let extension = AlgebraicExtension::new(rational_polynomial);
            let old_value = Q.neg(&Q.div(&self.poly.get_constant(), &self.poly.lcoeff()));
            let old_generator = extension.constant(old_value);
            let new_generator = extension.generator();
            let new_generator_minimal_poly = extension.poly.as_ref().clone();
            let extensions = (0..extension_degree)
                .map(|embedding| {
                    let mut field = extension.clone();
                    field.embedding = embedding;
                    field
                })
                .collect();

            return (
                extensions,
                old_generator,
                new_generator,
                new_generator_minimal_poly,
                (0..extension_degree).collect(),
            );
        }

        let (extension, old_generator, new_generator) = self.adjoin(polynomial, new_symbol);

        // The minimal polynomial of the image of b lets us reuse the rational
        // complex-root cache to put the roots over the selected embedding of
        // self into the same canonical order as b.
        let new_generator_minimal_poly = extension.minimal_polynomial_of_element(&new_generator);
        let old_poly = self.poly.to_univariate_from_univariate(0);
        let extension_poly = extension.poly.to_univariate_from_univariate(0);
        let new_generator_poly = new_generator_minimal_poly.to_univariate_from_univariate(0);

        let old_root = old_poly.root(self.embedding).unwrap();
        let mut extension_roots = extension_poly
            .isolate_roots()
            .into_iter()
            .map(|(root, _)| root)
            .collect::<Vec<_>>();
        let mut new_generator_roots = new_generator_poly
            .isolate_roots()
            .into_iter()
            .map(|(root, _)| root)
            .collect::<Vec<_>>();
        let old_generator_polynomial = old_generator.poly.to_univariate_from_univariate(0);
        let candidates = old_root
            .matching_roots(
                None,
                &mut extension_roots,
                Some(&old_generator_polynomial),
                extension_degree,
            )
            .unwrap_or_else(|error| {
                panic!(
                    "Could not select embeddings while adjoining roots of {}: {}",
                    polynomial, error
                )
            });

        let new_generator_polynomial = new_generator.poly.to_univariate_from_univariate(0);
        let mut ordered_candidates = candidates
            .into_iter()
            .map(|candidate| {
                let new_generator_embedding = extension_roots[candidate]
                    .matching_roots(
                        Some(&new_generator_polynomial),
                        &mut new_generator_roots,
                        None,
                        1,
                    )
                    .map(|matches| matches[0])
                    .unwrap_or_else(|error| {
                        panic!(
                            "Could not order an embedding while adjoining roots of {}: {}",
                            polynomial, error
                        )
                    });
                (new_generator_embedding, candidate)
            })
            .collect::<Vec<_>>();
        ordered_candidates.sort_unstable();
        assert!(
            !ordered_candidates
                .windows(2)
                .any(|pair| pair[0].0 == pair[1].0),
            "two extension embeddings map to the same new-generator embedding"
        );

        let (new_generator_embeddings, extensions): (Vec<_>, Vec<_>) = ordered_candidates
            .into_iter()
            .map(|(new_generator_embedding, embedding)| {
                let mut field = extension.clone();
                field.embedding = embedding;
                (new_generator_embedding, field)
            })
            .unzip();
        (
            extensions,
            old_generator,
            new_generator,
            new_generator_minimal_poly,
            new_generator_embeddings,
        )
    }

    /// Determine if the algebraic number is negative.
    /// This requires the embedding information to be set.
    pub fn is_negative(&self, element: &AlgebraicNumber<Q>) -> Result<bool, String> {
        self.try_sign(element).map(Ordering::is_lt)
    }

    /// Determine if the algebraic number is positive.
    /// This requires the embedding information to be set.
    pub fn is_positive(&self, element: &AlgebraicNumber<Q>) -> Result<bool, String> {
        self.try_sign(element).map(Ordering::is_gt)
    }
}

impl RealEmbedding for AlgebraicExtension<Q> {
    type Error = String;

    fn try_sign(&self, element: &AlgebraicNumber<Q>) -> Result<Ordering, Self::Error> {
        if self.is_zero(element) {
            return Ok(Ordering::Equal);
        }
        if element.poly.is_constant() {
            return Ok(OrderedRing::cmp(
                &Q,
                &element.poly.get_constant(),
                &Rational::zero(),
            ));
        }

        let polynomial = self.poly.to_univariate_from_univariate(0);
        let mut primitive_root = polynomial.root(self.embedding).unwrap();
        let primitive_location = primitive_root.classify_location();

        if matches!(primitive_location, RootLocation::Real | RootLocation::Zero) {
            return self.is_positive_real(element).map(|positive| {
                if positive {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            });
        }

        let minimal_field = self.simplify(element);
        let minimal_polynomial = minimal_field.poly.to_univariate_from_univariate(0);
        let mut root = minimal_polynomial.root(minimal_field.embedding).unwrap();
        let location = root.classify_location();
        if !matches!(location, RootLocation::Real | RootLocation::Zero) {
            return Err(format!(
                "{} does not have a real image in {}",
                element, self
            ));
        }

        minimal_field
            .is_positive_real(&minimal_field.generator())
            .map(|positive| {
                if positive {
                    Ordering::Greater
                } else {
                    Ordering::Less
                }
            })
    }

    fn try_cmp(
        &self,
        a: &AlgebraicNumber<Q>,
        b: &AlgebraicNumber<Q>,
    ) -> Result<Ordering, Self::Error> {
        // Unlike a genuinely ordered ring, this extension may contain
        // non-real elements. Require both operands, rather than merely their
        // difference, to have real images.
        self.try_sign(a)?;
        self.try_sign(b)?;
        self.try_sign(&self.sub(a, b))
    }
}

impl Root<Q> {
    /// Convert an expression polynomial into a rational root descriptor.
    pub fn from_atom(polynomial: AtomView<'_>, index: usize) -> Result<Self, String> {
        Self::from_atom_with_optional_variable(polynomial, None, index)
    }

    /// Convert an expression polynomial into a rational root descriptor using
    /// `variable` as its polynomial variable.
    pub fn from_atom_with_variable(
        polynomial: AtomView<'_>,
        variable: PolyVariable,
        index: usize,
    ) -> Result<Self, String> {
        Self::from_atom_with_optional_variable(polynomial, Some(variable), index)
    }

    fn from_atom_with_optional_variable(
        polynomial: AtomView<'_>,
        variable: Option<PolyVariable>,
        index: usize,
    ) -> Result<Self, String> {
        let variables = variable.map(|variable| Arc::new(vec![variable]));
        let polynomial = polynomial
            .try_to_polynomial::<_, u16>(&Q, variables)
            .map_err(|error| {
                format!("could not convert root polynomial to a polynomial over Q: {error}")
            })?;
        Self::new(polynomial, index)
    }

    /// Replace the defining polynomial by the irreducible factor containing
    /// the selected root. Root isolation is served by the shared root cache.
    pub fn simplify(&self) -> Result<Self, String> {
        let polynomial = self.polynomial.to_univariate_from_univariate(0);
        let isolated = polynomial.root(self.index).ok_or_else(|| {
            format!(
                "root index {} is out of bounds for polynomial of degree {}",
                self.index,
                polynomial.degree()
            )
        })?;
        let minimal_polynomial = isolated.defining_polynomial();
        if minimal_polynomial.degree() >= polynomial.degree() {
            return Ok(self.clone());
        }
        if minimal_polynomial
            .coefficients
            .iter()
            .any(|coefficient| !coefficient.im.is_zero())
        {
            return Err("a rational polynomial produced a non-rational root factor".to_string());
        }

        let minimal_polynomial =
            minimal_polynomial.map_coeff(|coefficient| coefficient.re.clone(), Q);
        Root::new(
            minimal_polynomial.to_multivariate::<u16>(),
            isolated.index(),
        )
    }

    /// Convert this polynomial-native root to an expression.
    pub fn to_atom(&self) -> Atom {
        let polynomial = self.polynomial.to_univariate_from_univariate(0);
        if polynomial.degree() == 1 {
            return Atom::num(&polynomial.coefficients[0] / &-polynomial.coefficients[1].clone());
        }
        if polynomial.degree() == 2 {
            let [c, b, a] = polynomial.coefficients.as_slice() else {
                unreachable!("a quadratic has three coefficients");
            };
            let discriminant = Atom::num(b * b - a * c * &Rational::from(4));
            return if self.index == 0 {
                ((-b.clone() - discriminant.sqrt()) / (Rational::from(2) * a.clone())).expand()
            } else {
                ((-b.clone() + discriminant.sqrt()) / (Rational::from(2) * a.clone())).expand()
            };
        }

        let mut polynomial = self.polynomial.clone();
        if polynomial.get_vars_ref()[0] != PolyVariable::Symbol(root_var()) {
            polynomial.rename_variable(
                &polynomial.get_vars_ref()[0].clone(),
                &PolyVariable::Symbol(root_var()),
            );
        }
        polynomial.to_expression().root(self.index)
    }
}

impl Root<RationalPolynomialField<IntegerRing, u16>> {
    /// Convert an expression polynomial into a root over `Q(parameters)`.
    ///
    /// The canonical root variable, conventional `z`, or the sole
    /// indeterminate is selected as the polynomial variable. If multiple
    /// other indeterminates occur, the root variable must be supplied
    /// explicitly. Every remaining indeterminate becomes part of the
    /// rational-function coefficient field.
    pub fn from_atom(polynomial: AtomView<'_>, index: usize) -> Result<Self, String> {
        Self::from_atom_with_optional_variable(polynomial, None, index)
    }

    /// Convert an expression polynomial into a root over `Q(parameters)`,
    /// explicitly selecting its polynomial variable.
    pub fn from_atom_with_variable(
        polynomial: AtomView<'_>,
        variable: PolyVariable,
        index: usize,
    ) -> Result<Self, String> {
        Self::from_atom_with_optional_variable(polynomial, Some(variable), index)
    }

    fn from_atom_with_optional_variable(
        polynomial: AtomView<'_>,
        explicit_variable: Option<PolyVariable>,
        index: usize,
    ) -> Result<Self, String> {
        let variables = explicit_variable
            .as_ref()
            .map(|variable| Arc::new(vec![variable.clone()]));
        let rational: RationalPolynomial<IntegerRing, u16> = polynomial
            .try_to_rational_polynomial(&Q, &Z, variables)
            .map_err(|error| format!("could not convert parametric root polynomial: {error}"))?;
        let variables = rational.numerator.variables().as_ref();
        let candidates = variables
            .iter()
            .enumerate()
            .filter(|(position, _)| rational.denominator.degree(*position) == 0)
            .collect::<Vec<_>>();
        let root_variable = if let Some(explicit_variable) = explicit_variable {
            candidates
                .iter()
                .find(|(_, variable)| variable == &&explicit_variable)
                .map(|(_, variable)| (*variable).clone())
                .ok_or_else(|| {
                    format!(
                        "the explicit root variable {explicit_variable} is not a polynomial variable"
                    )
                })?
        } else {
            candidates
                .iter()
                .find(|(_, variable)| variable == &&PolyVariable::from(root_var()))
                .or_else(|| {
                    candidates
                        .iter()
                        .find(|(_, variable)| variable == &&PolyVariable::from(symbol!("z")))
                })
                .map(|(_, variable)| (*variable).clone())
                .or_else(|| (candidates.len() == 1).then(|| candidates[0].1.clone()))
                .ok_or_else(|| {
                    "could not uniquely determine the root variable of the parametric polynomial"
                        .to_string()
                })?
        };
        let polynomial = rational
            .to_polynomial(&[root_variable], false)
            .map_err(str::to_string)?;
        Root::new(polynomial, index)
    }

    /// Normalize roots which have a closed expression over the parametric
    /// field. Branch indices are formal until parameter specialization.
    pub fn simplify(&self) -> Option<Atom> {
        let polynomial = self.polynomial.to_univariate_from_univariate(0);
        let field = &polynomial.ring;

        if polynomial.degree() == 1 {
            let value =
                field.neg(field.div(&polynomial.coefficients[0], &polynomial.coefficients[1]));
            return Some(value.to_expression());
        }

        if polynomial.degree() == 2 {
            let [c, b, a] = polynomial.coefficients.as_slice() else {
                unreachable!("a quadratic has three coefficients");
            };

            // This common form avoids introducing and then simplifying a
            // superfluous factor of four in sqrt(4*a).
            if field.is_zero(b) && field.is_one(a) {
                let square_root = field.neg(c).to_expression().sqrt();
                return Some(if self.index == 0 {
                    -square_root
                } else {
                    square_root
                });
            }

            let four_ac = field.mul(&field.nth(Integer::from(4)), &field.mul(a, c));
            let discriminant = field.sub(&field.mul(b, b), &four_ac).to_expression().sqrt();
            let minus_b = -b.to_expression();
            let numerator = if self.index == 0 {
                minus_b - discriminant
            } else {
                minus_b + discriminant
            };
            return Some((numerator / (Atom::num(2) * a.to_expression())).expand());
        }

        if polynomial.degree() == 3 {
            let [constant, linear, quadratic, leading] = polynomial.coefficients.as_slice() else {
                unreachable!("a cubic has four coefficients");
            };
            if field.is_zero(linear) && field.is_zero(quadratic) {
                let radicand = field.neg(field.div(constant, leading)).to_expression();
                let one_third = Atom::num(Rational::from((1, 3)));
                let principal_root = radicand.pow(one_third.clone());
                let minus_one_root = Atom::num(-1).pow(one_third);
                return Some(match self.index {
                    // For a positive radicand these follow the canonical
                    // complex order: negative-imaginary, positive-imaginary,
                    // and positive-real.
                    0 => -minus_one_root * principal_root,
                    1 => minus_one_root.pow(Atom::num(2)) * principal_root,
                    2 => principal_root,
                    _ => unreachable!("the root index was checked by Root::new"),
                });
            }
        }

        None
    }

    /// Convert this formal polynomial root to an expression-level root.
    pub fn to_atom(&self) -> Atom {
        if let Some(simplified) = self.simplify() {
            return simplified;
        }

        let mut polynomial = self.polynomial.clone();
        if polynomial.get_vars_ref()[0] != PolyVariable::Symbol(root_var()) {
            polynomial.rename_variable(
                &polynomial.get_vars_ref()[0].clone(),
                &PolyVariable::Symbol(root_var()),
            );
        }
        polynomial
            .to_expression_with_coeff_map(|_, coefficient, out| coefficient.to_expression_into(out))
            .root(self.index)
    }
}

#[derive(Clone)]
struct RootCandidate {
    polynomial: MultivariatePolynomial<Q, u16>,
    embedding: usize,
    multiplicity: usize,
    order: Option<usize>,
}

struct RootListSlot {
    roots: OnceLock<Result<Vec<Root<Q>>, String>>,
}

impl RootListSlot {
    fn new() -> Self {
        Self {
            roots: OnceLock::new(),
        }
    }
}

struct RootNormalizationCache {
    entries: RwLock<HashMap<MultivariatePolynomial<AlgebraicExtension<Q>, u16>, Arc<RootListSlot>>>,
}

impl RootNormalizationCache {
    fn global() -> &'static Self {
        static CACHE: LazyLock<RootNormalizationCache> = LazyLock::new(|| RootNormalizationCache {
            entries: RwLock::new(HashMap::new()),
        });
        &CACHE
    }

    fn slot(
        &self,
        polynomial: &MultivariatePolynomial<AlgebraicExtension<Q>, u16>,
    ) -> Arc<RootListSlot> {
        if let Some(entry) = self.entries.read().unwrap().get(polynomial).cloned() {
            return entry;
        }

        self.entries
            .write()
            .unwrap()
            .entry(polynomial.clone())
            .or_insert_with(|| Arc::new(RootListSlot::new()))
            .clone()
    }
}

const MAX_GAUSSIAN_ROOT_NORMALIZATION_DEGREE: usize = 4;

impl RootCandidate {
    fn from_element(
        field: &AlgebraicExtension<Q>,
        value: &AlgebraicNumber<Q>,
        multiplicity: usize,
    ) -> Result<Self, String> {
        let minimal_polynomial = field.minimal_polynomial_of_element(value);
        let embedding = field.root_index_of_element(value, &minimal_polynomial)?;
        Ok(Self::from_minimal_polynomial(
            &minimal_polynomial,
            embedding,
            multiplicity,
        ))
    }

    fn from_minimal_polynomial(
        minimal_polynomial: &MultivariatePolynomial<Q, u16>,
        embedding: usize,
        multiplicity: usize,
    ) -> Self {
        let mut polynomial = minimal_polynomial.clone();
        let variable = polynomial.get_vars_ref()[0].clone();
        polynomial.rename_variable(&variable, &PolyVariable::Temporary(0));
        Self {
            polynomial,
            embedding,
            multiplicity,
            order: None,
        }
    }

    fn as_root(&self) -> Result<Root<Q>, String> {
        Root::new(self.polynomial.clone(), self.embedding)
    }
}

impl Root<AlgebraicExtension<Q>> {
    /// Convert an expression polynomial and all of its algebraic constants to
    /// one embedded simple extension.
    ///
    /// `Ok(None)` means that the expression has no algebraic coefficients.
    pub fn from_atom(polynomial: AtomView<'_>, index: usize) -> Result<Option<Self>, String> {
        Self::from_atom_with_optional_variable(polynomial, None, index)
    }

    /// Convert an expression polynomial and all of its algebraic constants to
    /// one embedded simple extension, explicitly selecting its polynomial
    /// variable.
    pub fn from_atom_with_variable(
        polynomial: AtomView<'_>,
        variable: PolyVariable,
        index: usize,
    ) -> Result<Option<Self>, String> {
        Self::from_atom_with_optional_variable(polynomial, Some(variable), index)
    }

    fn from_atom_with_optional_variable(
        polynomial: AtomView<'_>,
        variable: Option<PolyVariable>,
        index: usize,
    ) -> Result<Option<Self>, String> {
        let mut context = AlgebraicContext::from_atom(polynomial)?;
        if context.is_trivial() {
            return Ok(None);
        }
        let variables = variable.map(|variable| Arc::new(vec![variable]));
        let polynomial = context.to_polynomial::<u16>(polynomial, variables)?;
        Root::new(polynomial, index).map(Some)
    }

    /// Collapse a selected root over an embedded algebraic coefficient field
    /// to its rational minimal polynomial and embedding.
    ///
    /// `Ok(None)` retains the specialized exact-Gaussian isolation path for
    /// high-degree polynomials, where primitive-element collapse is usually
    /// substantially more expensive.
    pub fn simplify(&self) -> Result<Option<Root<Q>>, String> {
        if self.polynomial.ring() == &AlgebraicExtension::complex(Q)
            && self.polynomial.degree(0) as usize > MAX_GAUSSIAN_ROOT_NORMALIZATION_DEGREE
        {
            return Ok(None);
        }

        let entry = RootNormalizationCache::global().slot(&self.polynomial);
        let roots = entry.roots.get_or_init(|| self.simplify_all_roots());
        let roots = roots.as_ref().map_err(Clone::clone)?;
        let result = roots.get(self.index).cloned().ok_or_else(|| {
            format!(
                "root index {} is out of bounds for a polynomial of degree {}",
                self.index,
                self.polynomial.degree(0)
            )
        })?;
        Ok(Some(result))
    }

    fn simplify_all_roots(&self) -> Result<Vec<Root<Q>>, String> {
        let base_field = self.polynomial.ring().clone();
        let mut candidates = Vec::new();
        for (factor, multiplicity) in self.polynomial.factor() {
            if factor.is_constant() {
                continue;
            }

            let degree = factor.degree(0) as usize;
            if degree == 1 {
                let value =
                    base_field.neg(&base_field.div(&factor.get_constant(), &factor.lcoeff()));
                candidates.push(RootCandidate::from_element(
                    &base_field,
                    &value,
                    multiplicity,
                )?);
                continue;
            }

            let variable = base_field.fresh_variable();
            let (_, _, _, minimal_polynomial, embeddings) =
                base_field.adjoin_with_all_embeddings_and_generator_data(&factor, Some(variable));
            for embedding in embeddings {
                candidates.push(RootCandidate::from_minimal_polynomial(
                    &minimal_polynomial,
                    embedding,
                    multiplicity,
                ));
            }
        }

        let counted_degree = candidates
            .iter()
            .map(|candidate| candidate.multiplicity)
            .sum::<usize>();
        if counted_degree != self.polynomial.degree(0) as usize {
            return Err(format!(
                "Factorization produced {counted_degree} roots for a polynomial of degree {}",
                self.polynomial.degree(0)
            ));
        }

        // When all candidates have the same rational minimal polynomial,
        // their embedding indices already provide their canonical global
        // order. In particular, adjoin_with_all_embeddings computed these
        // indices while ordering the primitive extension. Re-isolating the
        // degree-d product below would rediscover exactly the same ordering.
        if candidates
            .windows(2)
            .all(|pair| pair[0].polynomial == pair[1].polynomial)
        {
            candidates.sort_by_key(|candidate| candidate.embedding);
            let mut roots = Vec::with_capacity(self.polynomial.degree(0) as usize);
            for candidate in candidates {
                let root = candidate.as_root()?;
                for _ in 0..candidate.multiplicity {
                    roots.push(root.clone());
                }
            }
            return Ok(roots);
        }

        // Candidate minimal polynomials contain conjugates that need not be
        // roots over the selected embedding of the coefficient field. Their
        // square-free product nevertheless provides one rational, canonically
        // ordered root list into which every selected candidate can be
        // embedded.
        let mut seen_polynomials = HashSet::new();
        let prototype = MultivariatePolynomial::<Q, u16>::new(
            &Q,
            None,
            Arc::new(vec![PolyVariable::Temporary(0)]),
        );
        let mut union = prototype.one();
        for candidate in &candidates {
            let key = candidate
                .polynomial
                .to_univariate_from_univariate(0)
                .coefficients;
            if seen_polynomials.insert(key) {
                union = &union * &candidate.polynomial;
            }
        }
        let union = union.to_univariate_from_univariate(0);

        let mut union_roots = union
            .isolate_roots()
            .into_iter()
            .map(|(root, _)| root)
            .collect::<Vec<_>>();
        for candidate in &mut candidates {
            let candidate_polynomial = candidate.polynomial.to_univariate_from_univariate(0);
            let root = candidate_polynomial
                .root(candidate.embedding)
                .ok_or_else(|| {
                    format!(
                        "Could not isolate root {} of {}",
                        candidate.embedding, candidate.polynomial
                    )
                })?;
            candidate.order = Some(
                root.matching_roots(None, &mut union_roots, None, 1)
                    .map(|matches| matches[0])
                    .map_err(|_| {
                        format!(
                            "Could not canonically place root {} of {}",
                            candidate.embedding, candidate.polynomial
                        )
                    })?,
            );
        }
        candidates.sort_by_key(|candidate| candidate.order.unwrap());

        let mut roots = Vec::with_capacity(self.polynomial.degree(0) as usize);
        for candidate in candidates {
            let root = candidate.as_root()?;
            for _ in 0..candidate.multiplicity {
                roots.push(root.clone());
            }
        }
        Ok(roots)
    }
}

#[cfg(test)]
mod tests;
