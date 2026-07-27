//! Algebraic number fields, e.g. fields supporting sqrt(2).

use std::{collections::HashMap, sync::Arc};

use rand::Rng;

use crate::{
    atom::{Atom, AtomCore, AtomView, representation::FunView},
    coefficient::ConvertToRing,
    combinatorics::CombinationIterator,
    domains::{
        RingOps, Set,
        atom::AtomField,
        rational::Q,
        rational_polynomial::{FromNumeratorAndDenominator, RationalPolynomial},
    },
    poly::{
        Exponent, IntoVariableMap, PolyVariable, PositiveExponent,
        factor::Factorize,
        gcd::PolynomialGCD,
        polynomial::MultivariatePolynomial,
        univariate::{ComplexRootInterval, UnivariatePolynomial},
    },
    symbol,
    tensors::matrix::Matrix,
    transcendental::{TranscendentalFunctions, root},
};

use super::{
    EuclideanDomain, Field, InternalOrdering, Ring, SelfRing,
    finite_field::{FiniteField, FiniteFieldCore, FiniteFieldWorkspace, ToFiniteField},
    float::Complex,
    integer::Integer,
    rational::Rational,
};

#[derive(Clone)]
struct AlgebraicBall {
    center: Complex<Rational>,
    radius: Rational,
}

impl AlgebraicBall {
    fn from_root(root: &ComplexRootInterval) -> Self {
        Self {
            center: root.center().clone(),
            radius: root.radius().clone(),
        }
    }

    fn zero() -> Self {
        Self {
            center: Complex::new(Rational::zero(), Rational::zero()),
            radius: Rational::zero(),
        }
    }

    fn norm_upper_bound(center: &Complex<Rational>) -> Rational {
        center.re.abs() + center.im.abs()
    }

    fn add_rational(mut self, value: Rational) -> Self {
        self.center.re += value;
        self
    }

    fn mul(self, other: &Self) -> Self {
        let center = &self.center * &other.center;
        let self_center_norm = Self::norm_upper_bound(&self.center);
        let other_center_norm = Self::norm_upper_bound(&other.center);
        let radius = &self.radius * &other.radius
            + &self.radius * &other_center_norm
            + &other.radius * &self_center_norm;

        Self { center, radius }
    }

    fn is_disjoint(&self, other: &Self) -> bool {
        let distance = (&self.center.re - &other.center.re)
            .abs()
            .max((&self.center.im - &other.center.im).abs());
        distance > &self.radius + &other.radius
    }
}

/// A Galois field `GF(p,n)` is a finite field with `p^n` elements.
/// It provides methods to upgrade and downgrade to Galois fields with the
/// same prime but with a different power.
pub trait GaloisField: Field {
    type Base: Field;

    fn get_extension_degree(&self) -> u64;

    /// Upgrade the field to `GF(p,new_pow)`.
    fn upgrade(&self, new_pow: usize) -> AlgebraicExtension<Self::Base>
    where
        Self::Base: PolynomialGCD<u16>,
        <Self::Base as Set>::Element: Copy;

    fn upgrade_element(
        &self,
        e: &Self::Element,
        larger_field: &AlgebraicExtension<Self::Base>,
    ) -> <AlgebraicExtension<Self::Base> as Set>::Element;

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

    fn get_extension_degree(&self) -> u64 {
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
/// let sqrt_2 = extension.to_element(parse!("x").to_polynomial::<_, u16>(&Q, None));
///
/// let square = extension.mul(&sqrt_2, &sqrt_2);
/// assert_eq!(
///      square,
///      extension.to_element(parse!("2").to_polynomial(&Q, None))
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
    pub fn to_integer(&self, a: &<Self as Set>::Element) -> Integer {
        let mut p = Integer::zero();
        for x in a.poly.into_iter() {
            p += &(self.poly.ring.to_integer(x.coefficient)
                * &self.characteristic().pow(x.exponents[0] as u64));
        }
        p
    }

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

    fn get_extension_degree(&self) -> u64 {
        self.poly.degree(0) as u64
    }

    fn upgrade(&self, new_pow: usize) -> AlgebraicExtension<Self::Base>
    where
        Self::Base: PolynomialGCD<u16>,
        <Self::Base as Set>::Element: Copy,
    {
        AlgebraicExtension::galois_field(
            self.poly.ring.clone(),
            new_pow,
            self.poly.variables[0].clone(),
        )
    }

    fn upgrade_element(
        &self,
        e: &Self::Element,
        larger_field: &AlgebraicExtension<Self::Base>,
    ) -> <AlgebraicExtension<Self::Base> as Set>::Element {
        larger_field.to_element(e.poly.clone())
    }

    fn downgrade_element(
        &self,
        e: &<AlgebraicExtension<Self::Base> as Set>::Element,
    ) -> Self::Element {
        self.to_element(e.poly.clone())
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
            let (qn, r) = q.quot_rem(&self.poly.ring.size().unwrap());
            poly.append_monomial(r.to_finite_field(&self.poly.ring), &[pow]);
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
                    && self.poly().ring.is_one(&self.poly().get_constant())
                {
                    let ring = &self.poly().ring;
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

                    Ok(self
                        .to_element(self.poly().monomial(re, vec![1]) + self.poly().constant(im)))
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
                    && self.poly().ring.is_one(&self.poly().get_constant())
                {
                    let ring = &self.poly().ring;
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

                    Ok(self
                        .to_element(self.poly().monomial(re, vec![1]) + self.poly().constant(im)))
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
                    && self.poly().ring.is_one(&self.poly().get_constant())
                {
                    let ring = &self.poly().ring;
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

                    Ok(self
                        .to_element(self.poly().monomial(re, vec![1]) + self.poly().constant(im)))
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
                poly.append_monomial(poly.ring.nth((*c).into()), &[i as u16]);
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

fn positive_real_root_index(poly: &MultivariatePolynomial<Q, u16>) -> Option<usize> {
    let poly = poly.to_univariate_from_univariate(0);
    let mut binary_precision = 32u32;

    for _ in 0..10 {
        let tolerance = Rational::from((
            Integer::one(),
            Integer::from(2).pow(binary_precision as u64),
        ));
        let roots = poly.isolate_complex_roots(Some(tolerance));
        let mut unresolved_real_root = false;

        for (index, root) in roots.iter().enumerate() {
            if !root.is_real() {
                continue;
            }

            if &root.center().re - root.radius() > Rational::zero() {
                return Some(index);
            }

            if &root.center().re + root.radius() >= Rational::zero() {
                unresolved_real_root = true;
            }
        }

        if !unresolved_real_root {
            return None;
        }

        binary_precision *= 2;
    }

    None
}

fn rational_exponent_parts(exponent: AtomView<'_>) -> Option<(i64, usize)> {
    let exponent = Rational::try_from(exponent).ok()?;
    let numerator = exponent.numerator().to_i64()?;
    let denominator = usize::try_from(exponent.denominator().to_i64()?).ok()?;
    Some((numerator, denominator))
}

fn imaginary_unit_polynomial() -> MultivariatePolynomial<Q, u16> {
    let variable = PolyVariable::Temporary(0);
    let mut polynomial = MultivariatePolynomial::new(&Q, Some(2), Arc::new(vec![variable.clone()]));
    polynomial = polynomial.variable(&variable).unwrap().pow(2) + polynomial.one();
    polynomial
}

fn has_exact_imaginary_part(number: crate::coefficient::CoefficientView<'_>) -> bool {
    match number {
        crate::coefficient::CoefficientView::Natural(_, _, imaginary, _) => imaginary != 0,
        crate::coefficient::CoefficientView::Large(_, imaginary) => !imaginary.is_zero(),
        _ => false,
    }
}

fn root_descriptor(
    function: &FunView<'_>,
) -> Result<(MultivariatePolynomial<Q, u16>, usize), String> {
    if function.get_nargs() != 2 {
        return Err(format!(
            "root expects 2 arguments, got {}",
            function.get_nargs()
        ));
    }

    let polynomial_atom = function.get(0);
    let requested_index = usize::try_from(function.get(1))
        .map_err(|_| "root index is not a non-negative integer".to_string())?;
    let polynomial = polynomial_atom
        .try_to_polynomial::<_, u16>(&Q, None)
        .map_err(|error| format!("could not convert root polynomial: {error}"))?;

    if polynomial.nvars() != 1 || polynomial.degree(0) == 0 {
        return Err("root expects a non-constant univariate polynomial over Q".to_string());
    }

    let univariate = polynomial.to_univariate_from_univariate(0);
    if requested_index >= univariate.degree() {
        return Err(format!(
            "root index {requested_index} is out of bounds for polynomial of degree {}",
            univariate.degree()
        ));
    }

    let factors = polynomial.factor();
    let mut binary_precision = 32u32;
    for _ in 0..10 {
        let tolerance = Rational::from((
            Integer::one(),
            Integer::from(2).pow(binary_precision as u64),
        ));
        let requested_root = univariate
            .isolate_complex_root(requested_index, binary_precision)
            .ok_or_else(|| "could not isolate requested root".to_string())?;
        let requested_root = AlgebraicBall::from_root(&requested_root);
        let mut matches = Vec::new();

        for (factor, _) in &factors {
            let roots = factor
                .to_univariate_from_univariate(0)
                .isolate_complex_roots(Some(tolerance.clone()));
            for (index, candidate) in roots.iter().enumerate() {
                if !requested_root.is_disjoint(&AlgebraicBall::from_root(candidate)) {
                    matches.push((factor.clone(), index));
                }
            }
        }

        if matches.len() == 1 {
            let (mut factor, embedding) = matches.pop().unwrap();
            factor.variables = Arc::new(vec![PolyVariable::Temporary(0)]);
            return Ok((factor, embedding));
        }

        binary_precision *= 2;
    }

    Err(format!(
        "could not identify root {requested_index} of {polynomial}"
    ))
}

fn initial_embedding_field(atom: AtomView<'_>) -> Option<AlgebraicExtension<Q>> {
    match atom {
        AtomView::Num(number) => has_exact_imaginary_part(number.get_coeff_view())
            .then(|| AlgebraicExtension::new_complex(Q)),
        AtomView::Var(_) => None,
        AtomView::Fun(function) => {
            if function.get_symbol() == root() {
                atom.get_embedding_field_impl(None)
            } else {
                None
            }
        }
        AtomView::Pow(power) => {
            let (base, exponent) = power.get_base_exp();
            initial_embedding_field(base)
                .or_else(|| initial_embedding_field(exponent))
                .or_else(|| atom.get_embedding_field_impl(None))
        }
        AtomView::Mul(product) => product.into_iter().find_map(initial_embedding_field),
        AtomView::Add(sum) => sum.into_iter().find_map(initial_embedding_field),
    }
}

fn is_extension_generator(atom: AtomView<'_>) -> bool {
    match atom {
        AtomView::Fun(function) => function.get_symbol() == root(),
        AtomView::Pow(power) => {
            let (_, exponent) = power.get_base_exp();
            rational_exponent_parts(exponent).is_some_and(|(_, denominator)| denominator > 1)
        }
        AtomView::Num(number) => has_exact_imaginary_part(number.get_coeff_view()),
        _ => false,
    }
}

impl AlgebraicContext {
    pub fn new(field: AlgebraicExtension<Q>) -> Self {
        Self {
            field,
            images: HashMap::new(),
        }
    }

    /// Build a context and convert every algebraic subexpression in `atom`.
    ///
    /// `Ok(None)` means that the atom contains no supported algebraic
    /// extension and can be represented over `Q`.
    pub fn from_atom(atom: AtomView<'_>) -> Result<Option<Self>, String> {
        let Some(field) = initial_embedding_field(atom) else {
            return Ok(None);
        };

        let mut context = Self::new(field);
        context.prepare_atom(atom)?;
        Ok(Some(context))
    }

    /// Build a context from explicit algebraic generators.
    pub fn from_generators(generators: &[Atom]) -> Result<Self, String> {
        if generators.is_empty() {
            return Err("At least one algebraic generator is required".to_string());
        }

        let field = generators
            .iter()
            .find_map(|generator| initial_embedding_field(generator.as_view()))
            .ok_or_else(|| {
                "The supplied generators do not define an algebraic extension".to_string()
            })?;
        let mut context = Self::new(field);
        context.adjoin_generators(generators)?;
        Ok(context)
    }

    /// Adjoin explicit algebraic generators to this context.
    pub fn adjoin_generators(&mut self, generators: &[Atom]) -> Result<(), String> {
        for generator in generators {
            if !is_extension_generator(generator.as_view()) {
                return Err(format!(
                    "{} is not an explicit root or a rational power",
                    generator
                ));
            }
            self.convert_atom(generator.as_view())?;
        }
        Ok(())
    }

    pub fn field(&self) -> &AlgebraicExtension<Q> {
        &self.field
    }

    pub fn images(&self) -> &HashMap<Atom, AlgebraicNumber<Q>> {
        &self.images
    }

    pub fn image(&self, atom: &Atom) -> Option<&AlgebraicNumber<Q>> {
        self.images.get(atom)
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

        let atom = self.field.try_to_atom(&element)?;
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
        if polynomial.ring != self.field {
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

        let extension = AlgebraicExtension::new_with_embedding(polynomial, embedding);
        let new_variable = self.field.get_new_var();
        let (field, old_generator, new_generator) = self
            .field
            .adjoin_with_embedding(&extension, Some(new_variable));

        for image in self.images.values_mut() {
            *image = Self::transport_element(image, &field, &old_generator);
        }
        self.field = field;
        self.atom_from_element(new_generator)
    }

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
        let field = atom
            .get_embedding_field_impl(Some(self.field.clone()))
            .ok_or_else(|| format!("Could not construct an embedding field for {atom}"))?;
        self.replace_field(field)
    }

    /// Discover and cache every supported algebraic subexpression in `atom`.
    pub fn prepare_atom(&mut self, atom: AtomView<'_>) -> Result<(), String> {
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
                self.prepare_atom(base)?;
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
                    self.prepare_atom(*factor)?;
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
                    self.prepare_atom(*term)?;
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
        self.prepare_atom(atom)?;
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
                    let mut term = self.polynomial_from_images(term, &result.variables)?;
                    result.unify_variables(&mut term);
                    result = &result + &term;
                }
                Ok(result)
            }
            AtomView::Mul(product) => {
                let mut result =
                    MultivariatePolynomial::new(&self.field, None, var_map.clone()).one();
                for factor in product {
                    let mut factor = self.polynomial_from_images(factor, &result.variables)?;
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
                        self.rational_polynomial_from_images(term, &result.numerator.variables)?;
                    result.unify_variables(&mut term);
                    result = &result + &term;
                }
                Ok(result)
            }
            AtomView::Mul(product) => {
                let mut result = RationalPolynomial::new(&self.field, var_map.clone());
                result.numerator = result.numerator.add_constant(self.field.one());
                for factor in product {
                    let mut factor =
                        self.rational_polynomial_from_images(factor, &result.numerator.variables)?;
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
    ) -> Result<Atom, String> {
        let atom_field = AtomField::new();
        let generator_image = self.field.to_element(self.field.poly.one().mul_exp(&[1]));
        let preferred_generator = self
            .images
            .iter()
            .filter_map(|(atom, image)| {
                self.field
                    .is_zero(&self.field.sub(image, &generator_image))
                    .then_some(atom)
            })
            .min()
            .cloned();
        let generator = preferred_generator
            .clone()
            .map(Ok)
            .unwrap_or_else(|| self.field.try_to_atom(&generator_image))?;
        let variables = polynomial
            .variables
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
            let coefficient = if let Some(generator) = &preferred_generator {
                let mut coefficient = Atom::Zero;
                for generator_term in term.coefficient.poly() {
                    coefficient += generator.clone().pow(generator_term.exponents[0])
                        * Atom::num(generator_term.coefficient.clone());
                }
                coefficient
            } else {
                self.field.try_to_atom(term.coefficient)?
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
        let mut result = Atom::num(1);
        for (factor, exponent) in numerator {
            let factor = self.polynomial_to_atom(&factor)?;
            result *= factor.pow(Atom::num(exponent));
        }
        for (factor, exponent) in denominator {
            let factor = self.polynomial_to_atom(&factor)?;
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
        self.prepare_atom(atom)?;
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
        self.prepare_atom(atom)?;
        let var_map = var_map
            .into_var_map()?
            .unwrap_or_else(|| Arc::new(Vec::new()));
        self.rational_polynomial_from_images(atom, &var_map)
    }
}

#[test]
fn to_alg() {
    let a = crate::parse!("sqrt(2)+1");
    let ext = a.as_view().get_embedding_field().unwrap();
    let alg = a.as_view().to_algebraic(&ext).unwrap();
    let generator = ext.to_element(ext.poly.one().mul_exp(&[1]));
    assert_eq!(ext.embedding, 1);
    assert_eq!(alg, ext.add(&generator, &ext.one()));

    let b = crate::parse!("sqrt(2)+sqrt(3)");
    let ext = b.as_view().get_embedding_field().unwrap();
    let alg = b.as_view().to_algebraic(&ext).unwrap();
    let generator = ext.to_element(ext.poly.one().mul_exp(&[1]));
    assert_eq!(ext.embedding, 3);
    assert_eq!(alg, generator);

    let b = crate::parse!("sqrt(2)+sqrt(3)+sqrt(6)");
    let ext = b.as_view().get_embedding_field().unwrap();
    let alg = b.as_view().to_algebraic(&ext).unwrap();
    assert_eq!(ext.embedding, 3);
    assert_eq!(ext.is_positive(&alg), Ok(true));

    let b = crate::parse!("sqrt(3+sqrt(2))+1");
    let ext = b.as_view().get_embedding_field().unwrap();
    let alg = b.as_view().to_algebraic(&ext).unwrap();
    let generator = ext.to_element(ext.poly.one().mul_exp(&[1]));
    assert_eq!(ext.embedding, 3);
    assert_eq!(alg, ext.add(&generator, &ext.one()));

    let b = crate::parse!("2^(2/3)");
    let ext = b.as_view().get_embedding_field().unwrap();
    let alg = b.as_view().to_algebraic(&ext).unwrap();
    assert_eq!(ext.embedding, 2);
    assert_eq!(ext.pow(&alg, 3), ext.nth(4.into()));

    let b = crate::parse!("root(1-10*x^2+x^4,3)+1");
    let ext = b.as_view().get_embedding_field().unwrap();
    let alg = b.as_view().to_algebraic(&ext).unwrap();
    let generator = ext.to_element(ext.poly.one().mul_exp(&[1]));
    assert_eq!(ext.embedding, 3);
    assert_eq!(alg, ext.add(&generator, &ext.one()));
    let polynomial = b.as_view().try_to_polynomial::<_, u16>(&ext, None).unwrap();
    assert_eq!(polynomial.nvars(), 0);
    assert_eq!(polynomial.get_constant(), alg);

    let b = crate::parse!("root(x^3-2,0)+sqrt(3)");
    let ext = b.as_view().get_embedding_field().unwrap();
    let complex_cube_root = crate::parse!("root(x^3-2,0)")
        .as_view()
        .to_algebraic(&ext)
        .unwrap();
    let sqrt3 = crate::parse!("sqrt(3)")
        .as_view()
        .to_algebraic(&ext)
        .unwrap();
    let alg = b.as_view().to_algebraic(&ext).unwrap();
    assert_eq!(ext.pow(&complex_cube_root, 3), ext.nth(2.into()));
    assert_eq!(ext.mul(&sqrt3, &sqrt3), ext.nth(3.into()));
    assert_eq!(alg, ext.add(&complex_cube_root, &sqrt3));

    let b = crate::parse!("sqrt(2)+1𝑖");
    let ext = b.as_view().get_embedding_field().unwrap();
    let imaginary_unit = crate::parse!("1𝑖").as_view().to_algebraic(&ext).unwrap();
    let sqrt2 = crate::parse!("sqrt(2)")
        .as_view()
        .to_algebraic(&ext)
        .unwrap();
    let alg = b.as_view().to_algebraic(&ext).unwrap();
    assert_eq!(
        ext.mul(&imaginary_unit, &imaginary_unit),
        ext.neg(&ext.one())
    );
    assert_eq!(ext.mul(&sqrt2, &sqrt2), ext.nth(2.into()));
    assert_eq!(alg, ext.add(&sqrt2, &imaginary_unit));

    let b = crate::parse!("2+3𝑖");
    let ext = b.as_view().get_embedding_field().unwrap();
    assert_eq!(ext.embedding, 1);
    let imaginary_unit = ext.imaginary_unit().unwrap();
    assert_eq!(
        b.as_view().to_algebraic(&ext).unwrap(),
        ext.add(
            &ext.nth(2.into()),
            &ext.mul(&ext.nth(3.into()), &imaginary_unit)
        )
    );
}

#[test]
fn algebraic_context_conversion() {
    let expression = crate::parse!("x+sqrt(2)+sqrt(3)");
    let (context, polynomial) = expression
        .as_view()
        .to_polynomial_in_algebraic_extension::<u16>(symbol!("x"))
        .unwrap()
        .unwrap();
    assert_eq!(context.field().embedding, 3);

    let sqrt2 = context
        .images()
        .get(&crate::parse!("sqrt(2)"))
        .unwrap()
        .clone();
    let sqrt3 = context
        .images()
        .get(&crate::parse!("sqrt(3)"))
        .unwrap()
        .clone();
    assert_eq!(
        context.field().mul(&sqrt2, &sqrt2),
        context.field().nth(2.into())
    );
    assert_eq!(
        context.field().mul(&sqrt3, &sqrt3),
        context.field().nth(3.into())
    );

    assert_eq!(
        polynomial.get_constant(),
        context.field().add(&sqrt2, &sqrt3)
    );
    assert_eq!(polynomial.coefficient(&[1]).unwrap(), context.field().one());

    let expression = crate::parse!("(x+sqrt(2))/(x-sqrt(3))");
    let (mut context, rational) = expression
        .as_view()
        .to_rational_polynomial_in_algebraic_extension::<u16>(symbol!("x"))
        .unwrap()
        .unwrap();
    let sqrt2 = context
        .convert_atom(crate::parse!("sqrt(2)").as_view())
        .unwrap();
    let sqrt3 = context
        .convert_atom(crate::parse!("sqrt(3)").as_view())
        .unwrap();
    assert_eq!(rational.numerator.get_constant(), sqrt2);
    assert_eq!(
        rational.denominator.get_constant(),
        context.field().neg(&sqrt3)
    );
    assert_eq!(
        rational.numerator.coefficient(&[1]).unwrap(),
        context.field().one()
    );
    assert_eq!(
        rational.denominator.coefficient(&[1]).unwrap(),
        context.field().one()
    );
}

#[test]
fn factor_in_extension() {
    let factorization = crate::parse!("x^2+1")
        .factor_in_extension(&[Atom::i()])
        .unwrap();
    assert_eq!(factorization, crate::parse!("(x-1𝑖)*(x+1𝑖)"));

    let factorization = crate::parse!("x^2-2")
        .as_view()
        .factor_in_extension(&[crate::parse!("sqrt(2)")])
        .unwrap();
    assert_eq!(factorization.expand(), crate::parse!("x^2-2"));

    let factorization = crate::parse!("y^3-2")
        .as_view()
        .factor_in_extension(&[crate::parse!("root(x^3-2,2)")])
        .unwrap();
    assert_eq!(
        factorization,
        crate::parse!("(y-root(x^3-2,2))*(y*root(x^3-2,2)+y^2+root(x^3-2,2)^2)")
    );

    let factorization = crate::parse!("x^3-2")
        .as_view()
        .factor_in_extension(&[crate::parse!("root(x^3-2,2)")])
        .unwrap();
    assert_eq!(
        factorization,
        crate::parse!("(x-root(x^3-2,2))*(x*root(x^3-2,2)+x^2+root(x^3-2,2)^2)")
    );

    let factorization = crate::parse!("(x^2-2)/(x^2-3)")
        .as_view()
        .factor_in_extension(&[crate::parse!("sqrt(2)"), crate::parse!("sqrt(3)")])
        .unwrap();
    let mut context =
        AlgebraicContext::from_generators(&[crate::parse!("sqrt(2)"), crate::parse!("sqrt(3)")])
            .unwrap();
    let actual = context
        .to_rational_polynomial::<u16>(factorization.as_view(), None)
        .unwrap();
    let expected = context
        .to_rational_polynomial::<u16>(crate::parse!("(x^2-2)/(x^2-3)").as_view(), None)
        .unwrap();
    assert_eq!(actual, expected);

    let expression = crate::parse!("x^3+sqrt(2)*x^2-3*x-3*sqrt(2)");
    let factorization = expression.as_view().factor_in_extension(&[]).unwrap();
    assert_ne!(factorization, expression);
    assert_eq!(factorization.expand(), expression);

    let factorization = expression
        .as_view()
        .factor_in_extension(&[crate::parse!("sqrt(3)")])
        .unwrap();
    let mut context = AlgebraicContext::from_atom(expression.as_view())
        .unwrap()
        .unwrap();
    context
        .adjoin_generators(&[crate::parse!("sqrt(3)")])
        .unwrap();
    let actual = context
        .to_rational_polynomial::<u16>(factorization.as_view(), None)
        .unwrap();
    let expected = context
        .to_rational_polynomial::<u16>(expression.as_view(), None)
        .unwrap();
    assert_eq!(actual, expected);

    assert_eq!(
        crate::parse!("x^2-1")
            .as_view()
            .factor_in_extension(&[])
            .unwrap(),
        crate::parse!("(x-1)*(x+1)")
    );
    assert!(
        crate::parse!("x^2-2")
            .as_view()
            .factor_in_extension(&[crate::parse!("x")])
            .is_err()
    );
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
                    let (polynomial, embedding) = root_descriptor(f)?;
                    ring.embedded_rational_root(&polynomial, embedding)
                } else {
                    Err("function atoms cannot be converted to algebraic numbers".to_string())
                }
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();
                let base_converted = b.to_algebraic(&ring)?;

                if let Some((numerator, denominator)) = rational_exponent_parts(e) {
                    if denominator == 1 {
                        if numerator < 0 {
                            return Ok(
                                ring.inv(&ring.pow(&base_converted, numerator.unsigned_abs()))
                            );
                        }
                        return Ok(ring.pow(&base_converted, numerator.unsigned_abs()));
                    }

                    let var = ring.get_new_var();
                    let mut poly = MultivariatePolynomial::<_, u16>::new(
                        ring,
                        None,
                        Arc::new(vec![var.clone()]),
                    );
                    poly = poly.variable(&var).unwrap().pow(denominator)
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
    /// Use [`get_algebraic_context`](Self::get_algebraic_context) when the atom
    /// will subsequently be converted, so that the computed atom images are
    /// retained.
    pub fn get_embedding_field(&self) -> Option<AlgebraicExtension<Q>> {
        self.get_algebraic_context()
            .ok()
            .flatten()
            .map(|context| context.field)
    }

    /// Build a live algebraic context for the atom.
    pub fn get_algebraic_context(&self) -> Result<Option<AlgebraicContext>, String> {
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
        let mut context = match AlgebraicContext::from_atom(*self)? {
            Some(mut context) => {
                context.adjoin_generators(generators)?;
                context
            }
            None if generators.is_empty() => return Ok(self.factor()),
            None => AlgebraicContext::from_generators(generators)?,
        };
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
        let Some(mut context) = self.get_algebraic_context()? else {
            return Ok(None);
        };
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
        let Some(mut context) = self.get_algebraic_context()? else {
            return Ok(None);
        };
        let polynomial = context.to_rational_polynomial(*self, var_map)?;
        Ok(Some((context, polynomial)))
    }

    fn get_embedding_field_impl(
        &self,
        mut cur: Option<AlgebraicExtension<Q>>,
    ) -> Option<AlgebraicExtension<Q>> {
        match self {
            AtomView::Num(number) => {
                if !has_exact_imaginary_part(number.get_coeff_view()) {
                    return cur;
                }

                if let Some(current) = cur {
                    current.with_adjoined_rational_root(&imaginary_unit_polynomial(), 1)
                } else {
                    Some(AlgebraicExtension::new_complex(Q))
                }
            }
            AtomView::Var(_) => cur,
            AtomView::Fun(f) => {
                if f.get_symbol() == root() {
                    let (polynomial, embedding) = root_descriptor(f).ok()?;
                    if let Some(c) = cur {
                        c.with_adjoined_rational_root(&polynomial, embedding)
                    } else {
                        Some(AlgebraicExtension {
                            poly: Arc::new(polynomial),
                            embedding,
                        })
                    }
                } else {
                    cur
                }
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();
                cur = b.get_embedding_field_impl(cur);
                cur = e.get_embedding_field_impl(cur);

                if let Some((_, denominator)) = rational_exponent_parts(e)
                    && denominator > 1
                {
                    if let Some(c) = cur {
                        // create min poly and adjoin if needed
                        let base_converted = b.to_algebraic(&c).ok()?;
                        if !c.is_positive_real(&base_converted).ok()? {
                            return None;
                        }

                        let var = c.get_new_var();
                        let mut poly = MultivariatePolynomial::<_, u16>::new(
                            &c,
                            None,
                            Arc::new(vec![var.clone()]),
                        );
                        poly = poly.variable(&var).unwrap().pow(denominator)
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
                                let degree = factor.degree(0) as usize;
                                if degree <= 1 {
                                    continue;
                                }

                                if c.count_positive_real_roots(&factor).ok()? > 0 {
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
                            .pow(denominator)
                            - poly.constant(rat_base);

                        let mut selected = None;
                        for (factor, _) in poly.factor() {
                            let Some(embedding) = positive_real_root_index(&factor) else {
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
                    cur = a.get_embedding_field_impl(cur);
                }
                cur
            }
            AtomView::Add(a) => {
                for a in a {
                    cur = a.get_embedding_field_impl(cur);
                }
                cur
            }
        }
    }
}

impl<R: EuclideanDomain> AlgebraicExtension<R> {
    /// Create a new algebraic extension from a univariate polynomial.
    /// The polynomial should be monic and irreducible.
    ///
    /// The default embedding is root index 0. Use [`new_with_embedding`] to specify a different root index.
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

    pub fn new_with_embedding(
        poly: MultivariatePolynomial<R, u16>,
        root_index: usize,
    ) -> AlgebraicExtension<R> {
        AlgebraicExtension {
            poly: Arc::new(poly),
            embedding: root_index,
        }
    }

    pub fn constant(&self, c: R::Element) -> AlgebraicNumber<R> {
        AlgebraicNumber {
            poly: self.poly.constant(c),
        }
    }

    /// Get the minimal polynomial.
    pub fn poly(&self) -> &MultivariatePolynomial<R, u16> {
        &self.poly
    }

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

    pub fn try_to_element(
        &self,
        poly: MultivariatePolynomial<R, u16>,
    ) -> Result<<Self as Set>::Element, String> {
        if poly.nvars() == 0 {
            let mut new_poly = poly;
            new_poly.variables = self.poly.variables.clone();
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

    pub fn to_element(&self, poly: MultivariatePolynomial<R, u16>) -> <Self as Set>::Element {
        self.try_to_element(poly).unwrap()
    }

    /// Get a variable that is not already used in the polynomial.
    pub(crate) fn get_new_var(&self) -> PolyVariable {
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
    /// let Q_i = AlgebraicExtension::new_complex(Q);
    /// let poly = parse!("(-1+6𝑖)*x+(4+2𝑖)*x^2+3𝑖").to_polynomial::<_, u8>(&Q_i, None);
    /// assert_eq!(poly.factor().len(), 3);
    /// ```
    pub fn new_complex(ring: R) -> Self {
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

    // TODO: no need to try anymore, it always works
    pub fn try_to_atom(&self, element: &<Self as Set>::Element) -> Result<Atom, String>
    where
        R::Element: Into<crate::coefficient::Coefficient>,
    {
        let root = if matches!(self.poly.get_vars_ref()[0], PolyVariable::Temporary(_)) {
            let mut p = self.poly.as_ref().clone();
            p.variables = Arc::new(vec![PolyVariable::Symbol(symbol!("symbolica::root::z"))]);
            p.to_expression().root(self.embedding)
        } else {
            self.poly.to_expression().root(self.embedding)
        };

        // TODO: try simplification here

        let mut res = Atom::Zero;
        for t in element.poly() {
            res += root.pow(t.exponents[0]) * t.coefficient.clone().into();
        }
        return Ok(res);

        // if self.poly.nterms() == 2
        //     && self.poly.degree(0) == 2
        //     && self.poly.get_constant() == self.poly.ring.one()
        //     && self
        //         .poly
        //         .coefficient(&[2])
        //         .is_some_and(|c| c == self.poly.ring.one())
        // {
        //     if element.poly.degree(0) > 1 {
        //         return Err("Polynomial degree is too high".to_string());
        //     }

        //     let re = element
        //         .poly
        //         .coefficient(&[0])
        //         .unwrap_or_else(|| self.poly.ring.zero());
        //     let im = element
        //         .poly
        //         .coefficient(&[1])
        //         .unwrap_or_else(|| self.poly.ring.zero());

        //     Ok(Atom::num(re) + Atom::num(im) * Atom::i())
        // } else if self.poly.nterms() == 2 {
        //     let degree = self.poly.degree(0);
        //     let leading = self
        //         .poly
        //         .coefficient(&[degree])
        //         .unwrap_or_else(|| self.poly.ring.zero());

        //     if degree == 0 || leading != self.poly.ring.one() {
        //         return Err("Algebraic extension is not a binomial root".to_string());
        //     }

        //     let base = self.poly.ring.neg(&self.poly.get_constant());
        //     let root = Atom::num(base).pow(Atom::num((1usize, degree as usize)));
        //     let mut poly = element.poly.clone();
        //     poly.rename_variable(&self.poly.get_vars_ref()[0], &PolyVariable::Power(root));
        //     Ok(poly.to_expression())
        // } else {
        //     Err("Algebraic extension is not complex or a binomial root".to_string())
        // }
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
/// let sqrt_2 = extension.to_element(parse!("x").to_polynomial::<_, u16>(&Q, None));
///
/// let square = extension.mul(&sqrt_2, &sqrt_2);
/// assert_eq!(
///      square,
///      extension.to_element(parse!("2").to_polynomial(&Q, None))
/// );
///```
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct AlgebraicNumber<R: Ring> {
    pub(crate) poly: MultivariatePolynomial<R, u16>,
}

// can we use AlgebraicNumber directly the same as Root?
// index specifies the index of the root of the minimal polynomial

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
    pub fn mul_coeff(self, c: R::Element) -> Self {
        AlgebraicNumber {
            poly: self.poly.mul_coeff(c),
        }
    }

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

    pub fn into_poly(self) -> MultivariatePolynomial<R, u16> {
        self.poly
    }

    pub fn poly(&self) -> &MultivariatePolynomial<R, u16> {
        &self.poly
    }
}

impl<R: EuclideanDomain> Set for AlgebraicExtension<R> {
    type Element = AlgebraicNumber<R>;

    fn size(&self) -> Option<Integer> {
        self.poly
            .ring
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
            poly: self.poly.constant(self.poly.ring.nth(n)),
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
        self.poly.ring.characteristic()
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
        let mut m = vec![self.poly.ring.zero(); d * d];

        let mut f = self.one();

        for e in 0..d {
            let c = self.mul(b, &f);
            for monomial in &c.poly {
                m[monomial.exponents[0] as usize * d + e] = monomial.coefficient.clone();
            }
            f.poly.exponents[0] += 1;
        }

        let mut rhs = vec![self.poly.ring.zero(); d];
        for monomial in &a.poly {
            rhs[monomial.exponents[0] as usize] = monomial.coefficient.clone();
        }

        let m = Matrix::from_linear(m, d as u32, d as u32, self.poly.ring.clone()).unwrap();
        let rhs = Matrix::new_vec(rhs, self.poly.ring.clone());

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
            .map(|_| self.poly.ring.sample(rng, range))
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

impl<R: Field + PolynomialGCD<u16>> EuclideanDomain for AlgebraicExtension<R> {
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
            poly: a.poly.constant(a.poly.ring.gcd(&c1, &c2)),
        }
    }
}

impl<R: Field + PolynomialGCD<u16>> Field for AlgebraicExtension<R> {
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

impl<R: Field> AlgebraicExtension<R> {
    /// Create a new minimal field extension that has the algebraic number `x` as a root.
    pub fn simplify(&self, x: &AlgebraicNumber<R>) -> AlgebraicExtension<R> {
        let mut polys = vec![];

        let mut x_i = self.one();
        for _ in 0..=self.poly.degree(0) {
            x_i = self.mul(&x_i, x);
            polys.push(x_i.clone());

            // solve system c_0 + c_1 x + c_i x^2 + ... + x^i = 0
            let ncols = self.poly.degree(0).to_u32() as usize;

            let mut m = vec![self.poly.ring.zero(); polys.len() * ncols];
            for (row, p) in m.chunks_mut(ncols).zip(&polys) {
                for monomial in &p.poly {
                    row[monomial.exponents[0].to_u32() as usize] = monomial.coefficient.clone();
                }
            }

            let mut rhs = m.split_off((polys.len() - 1) * ncols);
            for e in &mut rhs {
                *e = self.poly.ring.neg(&*e);
            }

            if polys.len() == 1 {
                continue;
            }

            // TODO: recycle matrix
            let mat = Matrix::from_linear(
                m,
                (polys.len() - 1) as u32,
                ncols as u32,
                self.poly.ring.clone(),
            )
            .unwrap()
            .into_transposed();

            let rhs = Matrix::new_vec(rhs, self.poly.ring.clone());

            if let Ok(s) = mat.solve(&rhs) {
                let mut res = s.into_vec();
                res.push(self.poly.ring.one());
                let mut new_poly = self.poly.zero();
                for (p, c) in res.into_iter().enumerate() {
                    new_poly = &new_poly + &new_poly.monomial(c, vec![p as u16]);
                }

                return AlgebraicExtension::new(new_poly);
            }
        }

        unreachable!("Could not simplify algebraic number");
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
        assert_eq!(self, &b.ring);

        let (_, s, g, r) = b.norm_impl();
        debug_assert!(r.is_irreducible());

        let mut f = AlgebraicExtension::new(r);
        let mut g2 = g.to_number_field(&f);
        let mut h = self.poly.to_number_field(&f); // yields constant coeffs

        g2.unify_variables(&mut h);
        let g2 = g2.gcd(&h);

        let mut a = f.neg(&f.div(&g2.get_constant(), &g2.lcoeff()));
        let y = f.to_element(g2.ring.poly.one().mul_exp(&[1]));
        let mut b = f.sub(&y, &f.mul(&a, &f.nth(s.into())));

        if let Some(v) = &new_symbol {
            let old_var = &f.poly.get_vars_ref()[0];
            a.poly.rename_variable(old_var, v);
            b.poly.rename_variable(old_var, v);

            let mut new_poly = f.poly.as_ref().clone();
            new_poly.rename_variable(old_var, v);

            f = AlgebraicExtension {
                poly: Arc::new(new_poly),
                embedding: f.embedding,
            };
        }

        (f, a, b)
    }
}

impl<R: Field + PolynomialGCD<E>, E: PositiveExponent>
    MultivariatePolynomial<AlgebraicExtension<R>, E>
{
    /// Get the norm of a non-constant square-free polynomial `f` in the algebraic number field.
    pub fn norm(&self) -> MultivariatePolynomial<R, E> {
        self.norm_impl().3
    }

    /// Get the norm of a non-constant square-free polynomial `f` in the algebraic number field.
    /// Returns `(v, s, g, r)` where `v` is the shifted variable, `s` is the number of steps,
    /// `g` is the shifted polynomial and `r` is the norm.
    pub(crate) fn norm_impl(
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
            .position(|x| x == &self.ring.poly.variables[0])
            .unwrap();

        let mut poly = f.zero();
        let mut exp = vec![E::zero(); f.nvars()];
        for x in self.ring.poly.into_iter() {
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
                    - f.variable(&self.ring.poly.variables[0]).unwrap()
                        * &f.constant(f.ring.nth(s.into()));
                let g_multi = f.clone().replace_with_poly(v, &alpha_poly);
                let g_uni = g_multi.to_univariate(alpha);

                let r = g_uni.resultant_prs(&poly_uni);

                let d = r.derivative(v);
                if r.gcd(&d).is_constant() {
                    return (v, s, g_multi, r);
                }
            }

            s += 1;
        }
    }
}

impl AlgebraicExtension<Q> {
    fn evaluate_at_root(element: &AlgebraicNumber<Q>, root: &ComplexRootInterval) -> AlgebraicBall {
        let mut coefficients = vec![Rational::zero(); element.poly.degree(0) as usize + 1];
        for term in element.poly() {
            coefficients[term.exponents[0] as usize] = term.coefficient.clone();
        }

        let root = AlgebraicBall::from_root(root);
        let mut value = AlgebraicBall::zero();
        for coefficient in coefficients.into_iter().rev() {
            value = value.mul(&root).add_rational(coefficient);
        }
        value
    }

    fn is_positive_real(&self, element: &AlgebraicNumber<Q>) -> Result<bool, String> {
        if self.is_zero(element) {
            return Ok(false);
        }
        if element.poly.is_constant() {
            return Ok(!element.poly.get_constant().is_negative());
        }

        let poly = self.poly.to_univariate_from_univariate(0);
        let primitive_root = poly
            .isolate_complex_root(self.embedding, 32)
            .ok_or_else(|| {
                format!(
                    "Embedding index {} is out of bounds for polynomial of degree {}",
                    self.embedding,
                    poly.degree()
                )
            })?;

        if !primitive_root.is_real() {
            let minimal_polynomial = self.simplify(element);
            let element_embedding =
                self.root_index_of_element(element, &minimal_polynomial.poly)?;
            let minimal_polynomial = minimal_polynomial.poly.to_univariate_from_univariate(0);
            let mut binary_precision = 32u32;

            for _ in 0..10 {
                let root = minimal_polynomial
                    .isolate_complex_root(element_embedding, binary_precision)
                    .ok_or_else(|| {
                        format!(
                            "Embedding index {} is out of bounds for polynomial of degree {}",
                            element_embedding,
                            minimal_polynomial.degree()
                        )
                    })?;
                if !root.is_real() {
                    return Ok(false);
                }
                if &root.center().re - root.radius() > Rational::zero() {
                    return Ok(true);
                }
                if &root.center().re + root.radius() < Rational::zero() {
                    return Ok(false);
                }

                binary_precision *= 2;
            }

            return Err(format!(
                "Could not determine the sign of {} in {}",
                element, self
            ));
        }

        let mut binary_precision = 32u32;
        for _ in 0..10 {
            let root = poly
                .isolate_complex_root(self.embedding, binary_precision)
                .ok_or_else(|| {
                    format!(
                        "Embedding index {} is out of bounds for polynomial of degree {}",
                        self.embedding,
                        poly.degree()
                    )
                })?;

            let value = Self::evaluate_at_root(element, &root);
            if &value.center.re - &value.radius > Rational::zero() {
                return Ok(true);
            }
            if &value.center.re + &value.radius < Rational::zero() {
                return Ok(false);
            }

            binary_precision *= 2;
        }

        Err(format!(
            "Could not determine the sign of {} in {}",
            element, self
        ))
    }

    fn root_index_of_element(
        &self,
        element: &AlgebraicNumber<Q>,
        polynomial: &MultivariatePolynomial<Q, u16>,
    ) -> Result<usize, String> {
        let extension_polynomial = self.poly.to_univariate_from_univariate(0);
        let polynomial = polynomial.to_univariate_from_univariate(0);
        let mut binary_precision = 32u32;

        for _ in 0..10 {
            let tolerance = Rational::from((
                Integer::one(),
                Integer::from(2).pow(binary_precision as u64),
            ));
            let extension_root = extension_polynomial
                .isolate_complex_root(self.embedding, binary_precision)
                .ok_or_else(|| {
                    format!(
                        "Embedding index {} is out of bounds for polynomial of degree {}",
                        self.embedding,
                        extension_polynomial.degree()
                    )
                })?;
            let value = Self::evaluate_at_root(element, &extension_root);
            let roots = polynomial.isolate_complex_roots(Some(tolerance));
            let matches = roots
                .iter()
                .enumerate()
                .filter_map(|(index, root)| {
                    (!value.is_disjoint(&AlgebraicBall::from_root(root))).then_some(index)
                })
                .collect::<Vec<_>>();

            if matches.len() == 1 {
                return Ok(matches[0]);
            }

            binary_precision *= 2;
        }

        Err(format!(
            "Could not identify {} as a root of {}",
            element, polynomial
        ))
    }

    fn embedded_rational_root(
        &self,
        polynomial: &MultivariatePolynomial<Q, u16>,
        embedding: usize,
    ) -> Result<AlgebraicNumber<Q>, String> {
        let mut polynomial_over_self = polynomial.clone();
        polynomial_over_self.rename_variable(&polynomial.get_vars_ref()[0], &self.get_new_var());
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
        polynomial_over_self.rename_variable(&polynomial.get_vars_ref()[0], &self.get_new_var());
        let polynomial_over_self = polynomial_over_self.to_number_field(self);

        for (factor, _) in polynomial_over_self.factor() {
            let degree = factor.degree(0) as usize;
            if degree <= 1 {
                continue;
            }

            for local_embedding in 0..degree {
                let extension = AlgebraicExtension {
                    poly: Arc::new(factor.clone()),
                    embedding: local_embedding,
                };
                let (extension, _, generator) = self.adjoin_with_embedding(&extension, None);
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
        self.embedded_rational_root(&imaginary_unit_polynomial(), 1)
    }

    fn sign_at_embedding(&self, element: &AlgebraicNumber<Q>) -> Result<i8, String> {
        if self.is_zero(element) {
            Ok(0)
        } else if self.is_positive_real(element)? {
            Ok(1)
        } else {
            Ok(-1)
        }
    }

    /// Count the positive real roots of a square-free polynomial whose
    /// coefficients lie in this embedded real number field.
    fn count_positive_real_roots(
        &self,
        poly: &MultivariatePolynomial<AlgebraicExtension<Q>, u16>,
    ) -> Result<usize, String> {
        let mut previous = poly.to_univariate_from_univariate(0);
        let mut current = previous.derivative();
        let mut sturm_sequence: Vec<UnivariatePolynomial<AlgebraicExtension<Q>>> =
            vec![previous.clone()];

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

        let sign_variations = |at_positive_infinity: bool| -> Result<usize, String> {
            let mut previous_sign = 0;
            let mut variations = 0;

            for polynomial in &sturm_sequence {
                let value = if at_positive_infinity {
                    polynomial.lcoeff()
                } else {
                    polynomial.get_constant()
                };
                let sign = self.sign_at_embedding(&value)?;
                if sign == 0 {
                    continue;
                }
                if previous_sign != 0 && sign != previous_sign {
                    variations += 1;
                }
                previous_sign = sign;
            }

            Ok(variations)
        };

        let at_zero = sign_variations(false)?;
        let at_positive_infinity = sign_variations(true)?;
        at_zero.checked_sub(at_positive_infinity).ok_or_else(|| {
            format!(
                "Invalid Sturm sequence while counting positive roots of {}",
                poly
            )
        })
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
        assert_eq!(
            self, &b.poly.ring,
            "The base field of the adjoined extension does not match"
        );

        let extension_degree = b.poly.degree(0) as usize;
        assert!(
            b.embedding < extension_degree,
            "Embedding index {} is out of bounds for polynomial of degree {}",
            b.embedding,
            extension_degree
        );

        let (mut extension, old_generator, new_generator) = self.adjoin(&b.poly, new_symbol);

        // The minimal polynomial of the image of b lets us reuse the rational
        // complex-root cache to put the roots over the selected embedding of
        // self into the same canonical order as b.
        let new_generator_minimal_poly = extension.simplify(&new_generator);
        let old_poly = self.poly.to_univariate_from_univariate(0);
        let extension_poly = extension.poly.to_univariate_from_univariate(0);
        let new_generator_poly = new_generator_minimal_poly
            .poly
            .to_univariate_from_univariate(0);

        let mut binary_precision = 32u32;
        for _ in 0..10 {
            let tolerance = Rational::from((
                Integer::one(),
                Integer::from(2).pow(binary_precision as u64),
            ));

            // All three calls go through ROOT_CACHE. In particular, increasing
            // the precision refines the cached intervals instead of isolating
            // the same roots from scratch.
            let old_root = old_poly
                .isolate_complex_root(self.embedding, binary_precision)
                .unwrap_or_else(|| {
                    panic!(
                        "Embedding index {} is out of bounds for polynomial of degree {}",
                        self.embedding,
                        old_poly.degree()
                    )
                });
            let old_root = AlgebraicBall::from_root(&old_root);
            let extension_roots = extension_poly.isolate_complex_roots(Some(tolerance.clone()));

            let candidates = extension_roots
                .iter()
                .enumerate()
                .filter_map(|(index, root)| {
                    let image = Self::evaluate_at_root(&old_generator, root);
                    (!image.is_disjoint(&old_root)).then_some(index)
                })
                .collect::<Vec<_>>();

            if candidates.len() != extension_degree {
                binary_precision *= 2;
                continue;
            }

            let new_generator_roots = new_generator_poly.isolate_complex_roots(Some(tolerance));
            let mut ordered_candidates = Vec::with_capacity(candidates.len());
            let mut all_unique = true;

            for candidate in candidates {
                let image = Self::evaluate_at_root(&new_generator, &extension_roots[candidate]);
                let matching_roots = new_generator_roots
                    .iter()
                    .enumerate()
                    .filter_map(|(index, root)| {
                        let root = AlgebraicBall::from_root(root);
                        (!image.is_disjoint(&root)).then_some(index)
                    })
                    .collect::<Vec<_>>();

                if matching_roots.len() != 1 {
                    all_unique = false;
                    break;
                }
                ordered_candidates.push((matching_roots[0], candidate));
            }

            if !all_unique {
                binary_precision *= 2;
                continue;
            }

            ordered_candidates.sort_unstable();
            if ordered_candidates
                .windows(2)
                .any(|pair| pair[0].0 == pair[1].0)
            {
                binary_precision *= 2;
                continue;
            }

            extension.embedding = ordered_candidates[b.embedding].1;
            return (extension, old_generator, new_generator);
        }

        panic!(
            "Could not distinguish the embedding while adjoining root {} of {}",
            b.embedding, b.poly
        );
    }

    /// Determine if the algebraic number is negative.
    /// This requires the embedding information to be set.
    pub fn is_negative(&self, element: &AlgebraicNumber<Q>) -> Result<bool, String> {
        if self.is_zero(element) {
            Ok(false)
        } else {
            self.is_positive(element).map(|b| !b)
        }
    }

    /// Determine if the algebraic number is positive.
    /// This requires the embedding information to be set.
    pub fn is_positive(&self, element: &AlgebraicNumber<Q>) -> Result<bool, String> {
        self.is_positive_real(element)
    }
}

#[cfg(test)]
mod tests {
    use crate::atom::AtomCore;
    use crate::domains::algebraic_number::AlgebraicExtension;
    use crate::domains::finite_field::{PrimeIteratorU64, Z2, Zp};
    use crate::domains::integer::Z;
    use crate::domains::rational::Q;
    use crate::domains::{Ring, RingOps};
    use crate::{parse, symbol};

    // #[test]
    // fn is_algebraic_number_positive() {
    //     let ring = parse!("a^3 + 3a^2 - 46*a + 1").to_polynomial(&Q, None);
    //     let ring = AlgebraicExtension::new_with_embedding(
    //         ring.clone(),
    //         RootInfo::from_index(2, &ring.to_univariate_from_univariate(0)),
    //     );

    //     let a = parse!("1/5a^2-a-1/10").to_polynomial::<_, u16>(&Q, None);
    //     let a = ring.to_element(a);

    //     assert_eq!(ring.is_positive(&a), Ok(true));
    // }
    //
    //

    #[test]
    fn adjoin_and_convert() {
        let sqrt2 =
            AlgebraicExtension::new_with_embedding(parse!("a^2-2").to_polynomial(&Q, None), 1);
        let sqrt3 = AlgebraicExtension::new_with_embedding(
            parse!("b^2-3")
                .to_polynomial(&Q, None)
                .to_number_field(&sqrt2),
            1,
        );

        let (sqrt23, _, _) = sqrt2.adjoin_with_embedding(&sqrt3, Some(symbol!("gamma").into()));

        let poly = parse!("gamma").to_polynomial(&Q, None);
        let var = sqrt23.to_element(poly);

        let var2 = sqrt23.mul(&var, &var);
        let e = sqrt23.try_to_atom(&var2).unwrap();
        println!("{}", e);
        println!(
            "comparison: {}",
            (e - parse!("(sqrt(2) + sqrt(3))^2")).to_float(16)
        );
    }

    #[test]
    fn adjoin_with_embedding() {
        for (sqrt2_embedding, sqrt3_embedding, expected_embedding) in
            [(0, 0, 0), (1, 0, 1), (0, 1, 2), (1, 1, 3)]
        {
            let sqrt2 = AlgebraicExtension::new_with_embedding(
                parse!("a^2-2").to_polynomial(&Q, None),
                sqrt2_embedding,
            );
            let sqrt3 = AlgebraicExtension::new_with_embedding(
                parse!("b^2-3")
                    .to_polynomial(&Q, None)
                    .to_number_field(&sqrt2),
                sqrt3_embedding,
            );

            let (sqrt23, r1, r2) =
                sqrt2.adjoin_with_embedding(&sqrt3, Some(symbol!("gamma").into()));

            assert_eq!(sqrt23.embedding, expected_embedding);
            assert_eq!(sqrt23.mul(&r1, &r1), sqrt23.nth(2.into()));
            assert_eq!(sqrt23.mul(&r2, &r2), sqrt23.nth(3.into()));
        }
    }

    #[test]
    fn adjoin_with_complex_embedding() {
        for (sqrt2_embedding, i_embedding, expected_embedding) in
            [(0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3)]
        {
            let sqrt2 = AlgebraicExtension::new_with_embedding(
                parse!("a^2-2").to_polynomial(&Q, None),
                sqrt2_embedding,
            );
            let i = AlgebraicExtension::new_with_embedding(
                parse!("b^2+1")
                    .to_polynomial(&Q, None)
                    .to_number_field(&sqrt2),
                i_embedding,
            );

            let (extension, r1, r2) =
                sqrt2.adjoin_with_embedding(&i, Some(symbol!("gamma").into()));

            assert_eq!(extension.embedding, expected_embedding);
            assert_eq!(extension.mul(&r1, &r1), extension.nth(2.into()));
            assert_eq!(extension.mul(&r2, &r2), extension.neg(&extension.one()));
        }
    }

    #[test]
    fn algebraic_number_to_atom_complex() {
        let ring = AlgebraicExtension::new_complex(Q);

        let i = ring.to_element(parse!("𝑖").to_polynomial::<_, u16>(&Q, None));
        assert_eq!(ring.try_to_atom(&i).unwrap(), parse!("1𝑖"));

        let one_plus_i = ring.to_element(parse!("1+𝑖").to_polynomial::<_, u16>(&Q, None));
        assert_eq!(ring.try_to_atom(&one_plus_i).unwrap(), parse!("1+1𝑖"));

        let ring =
            AlgebraicExtension::new_with_embedding(parse!("a^2+1").to_polynomial(&Q, None), 1);
        let a = ring.to_element(parse!("a").to_polynomial::<_, u16>(&Q, None));
        assert_eq!(ring.try_to_atom(&a).unwrap(), parse!("1𝑖"));
    }

    #[test]
    fn algebraic_number_to_atom_binomial_root() {
        let ring =
            AlgebraicExtension::new_with_embedding(parse!("a^3-2").to_polynomial(&Q, None), 2);

        let a_squared = ring.to_element(parse!("a^2").to_polynomial::<_, u16>(&Q, None));
        assert_eq!(
            ring.try_to_atom(&a_squared).unwrap(),
            parse!("root(a^3-2,2)^2")
        );
    }

    #[test]
    fn gcd_number_field() {
        let ring = parse!("a^3 + 3a^2 - 46*a + 1").to_polynomial(&Q, None);
        let ring = AlgebraicExtension::new(ring);

        let a = parse!("x^3-2x^2+(-2a^2+8a+2)x-a^2+11a-1")
            .to_polynomial::<_, u16>(&Q, None)
            .to_number_field(&ring);

        let b = parse!("x^3-2x^2-x+1")
            .to_polynomial(&Q, a.variables.clone())
            .to_number_field(&ring);

        let r = a.gcd(&b).from_number_field();

        let expected = parse!("-50/91+x-23/91*a-1/91*a^2").to_polynomial(&Q, a.variables.clone());
        assert_eq!(r, expected);
    }

    #[test]
    fn galois() {
        for j in 1..10 {
            let _ = AlgebraicExtension::galois_field(Z2, j, symbol!("v1").into());
        }

        for i in PrimeIteratorU64::new(2).take(20) {
            for j in 1..10 {
                let _ =
                    AlgebraicExtension::galois_field(Zp::new(i as u32), j, symbol!("v1").into());
            }
        }
    }

    #[test]
    fn norm() {
        let a = parse!("z^4+z^3+(2+a-a^2)z^2+(1+a^2-2a^3)z-2").to_polynomial::<_, u8>(&Q, None);
        let f = parse!("a^4-3").to_polynomial::<_, u16>(&Q, None);
        let f = AlgebraicExtension::new(f);
        let norm = a.to_number_field(&f).norm();

        let res = parse!("16-32*z-64*z^2-64*z^3-52*z^4-40*z^5-132*z^6-24*z^7-50*z^8+120*z^9+66*z^10+92*z^11+47*z^12+32*z^13+14*z^14+4*z^15+z^16")
        .to_polynomial::<_, u8>(&Q, a.variables.clone());

        assert_eq!(norm, res);
    }

    #[test]
    fn extend() {
        let a = parse!("x^2-2").to_polynomial(&Q, None);
        let ae = AlgebraicExtension::new(a);

        let b = parse!("y^2-3").to_polynomial(&Q, None).to_number_field(&ae);

        let (c, rep1, rep2) = ae.adjoin(&b, None);

        let rf = parse!("1-10*y^2+y^4").to_polynomial(&Q, None);

        assert_eq!(c.poly.as_ref(), &rf);

        let r1 = parse!("-9/2y+1/2y^3").to_polynomial::<_, u16>(&Q, None);
        assert_eq!(rep1.poly, r1);

        let r2 = parse!("11/2*y-1/2*y^3").to_polynomial::<_, u16>(&Q, None);
        assert_eq!(rep2.poly, r2);
    }

    #[test]
    fn simplify() {
        let poly = AlgebraicExtension::new(
            parse!("13-16v1+28v1^2+2v1^3+11v1^4+v1^6").to_polynomial(&Q, None),
        );

        let a = poly.to_element(
            parse!("-295/1882 -2693/1882v1 -237/1882v1^2 -385/941v1^3 -9/1882v1^4  -33/941v1^5")
                .to_polynomial::<_, u16>(&Q, None),
        );

        let r = poly.simplify(&a);
        let res = parse!("1+v1+v1^2").to_polynomial(&Q, None);
        assert_eq!(*r.poly, res);
    }

    #[test]
    fn try_div() {
        let extension = AlgebraicExtension::new(parse!("v1^3-2v1+3").to_polynomial(&Z, None));

        let f1 = extension.to_element(parse!("v1^2-2").to_polynomial(&Z, None));
        let f2 = extension.to_element(parse!("v1-5").to_polynomial(&Z, None));
        let prod = extension.mul(&f1, &f2);

        assert_eq!(extension.try_div(&prod, &f2).unwrap(), f1);
        assert_eq!(extension.try_div(&prod, &f1).unwrap(), f2);
        assert!(extension.try_div(&f2, &f1).is_none());
    }
}
