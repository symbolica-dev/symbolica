//! Algebraic number fields, e.g. fields supporting sqrt(2).

use std::{marker::PhantomData, sync::Arc};

use rand::Rng;

use crate::{
    atom::Atom,
    coefficient::ConvertToRing,
    combinatorics::CombinationIterator,
    domains::{RingOps, Set, rational::Q},
    poly::{
        PolyVariable, PositiveExponent, factor::Factorize, gcd::PolynomialGCD,
        polynomial::MultivariatePolynomial, univariate::UnivariatePolynomial,
    },
    symbol,
    tensors::matrix::Matrix,
};

use super::{
    EuclideanDomain, Field, InternalOrdering, Ring, SelfRing,
    finite_field::{FiniteField, FiniteFieldCore, FiniteFieldWorkspace, ToFiniteField},
    integer::Integer,
    rational::Rational,
};

/// Information about a specific root of a polynomial.
/// The roots are sorted in the following way: first
/// all real roots in ascending order, then complex roots
/// sorted by their real part and then by their imaginary part.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RootInfo {
    pub index: u16,
    pub is_real: bool,
    pub isolating_interval: Option<(Rational, Rational)>,
    pub complex_interval: Option<(Rational, Rational)>,
}

impl RootInfo {
    /// Create root information for the `index`-th root of the polynomial `poly`.
    pub fn from_index(index: usize, poly: &UnivariatePolynomial<Q>) -> Self {
        if index > poly.degree() {
            panic!(
                "Index {} is out of bounds for polynomial of degree {}",
                index,
                poly.degree(),
            );
        }

        let roots = poly.isolate_roots(None);

        let mut counter = 0;
        for (a, b, pow) in roots {
            if counter + pow > index {
                return RootInfo {
                    index: index as u16,
                    is_real: true,
                    isolating_interval: Some((a, b)),
                    complex_interval: None,
                };
            } else {
                counter += pow;
            }
        }

        // TODO: complex root isolation?
        RootInfo {
            index: index as u16,
            is_real: false,
            isolating_interval: None,
            complex_interval: None,
        }
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
/// use symbolica::{
///     atom::{Atom, AtomCore},
///     domains::{algebraic_number::AlgebraicExtension, rational::Q, Ring, RingOps},
///     parse,
/// };
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
/// use symbolica::{
///     atom::{Atom, AtomCore, Symbol},
///     domains::{algebraic_number::AlgebraicExtension, finite_field::Zp, rational::Q, Ring},
///     symbol,
/// };
///
/// let field = AlgebraicExtension::galois_field(Zp::new(17), 4, symbol!("x0").into());
/// ```
///
// TODO: make special case for degree two and three and hardcode the multiplication table
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct AlgebraicExtension<R: Ring> {
    poly: Arc<MultivariatePolynomial<R, u16>>, // TODO: convert to univariate polynomial
    embedding: Option<Arc<RootInfo>>,
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

impl<R: EuclideanDomain> AlgebraicExtension<R> {
    /// Create a new algebraic extension from a univariate polynomial.
    /// The polynomial should be monic and irreducible.
    pub fn new(poly: MultivariatePolynomial<R, u16>) -> AlgebraicExtension<R> {
        if poly.nvars() == 1 {
            return AlgebraicExtension {
                poly: Arc::new(poly),
                embedding: None,
            };
        }

        assert_eq!((0..poly.nvars()).filter(|v| poly.degree(*v) > 0).count(), 1);
        let v = (0..poly.nvars()).find(|v| poly.degree(*v) > 0).unwrap();
        let uni = poly.to_univariate_from_univariate(v);

        AlgebraicExtension {
            poly: Arc::new(uni.to_multivariate()),
            embedding: None,
        }
    }

    pub fn new_with_embedding(
        poly: MultivariatePolynomial<R, u16>,
        embedding: RootInfo,
    ) -> AlgebraicExtension<R> {
        AlgebraicExtension {
            poly: Arc::new(poly),
            embedding: Some(Arc::new(embedding)),
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
            embedding: None,
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
}

impl<R: Ring> AlgebraicExtension<R> {
    /// Create a new algebraic extension `R(i)`.
    /// This ring can be used to convert expressions with complex coefficients
    /// to polynomials.
    ///
    /// # Examples
    ///
    /// Creating Gaussian rationals:
    /// ```rust
    /// use symbolica::{parse, atom::AtomCore, domains::{algebraic_number::AlgebraicExtension, rational::Q}, poly::factor::Factorize};
    /// let Q_i = AlgebraicExtension::new_complex(Q);
    /// let poly = parse!("(-1+6𝑖)*x+(4+2𝑖)*x^2+3𝑖").to_polynomial::<_, u8>(&Q_i, None);
    /// assert_eq!(poly.factor().len(), 3);
    /// ```
    pub fn new_complex(ring: R) -> Self {
        let poly = MultivariatePolynomial {
            coefficients: vec![ring.one(), ring.one()],
            exponents: vec![0, 2],
            ring: ring,
            variables: Arc::new(vec![symbol!(Atom::I_STR).into()]),
            _phantom: PhantomData,
        };

        AlgebraicExtension {
            poly: Arc::new(poly),
            embedding: Some(Arc::new(RootInfo {
                index: 1, // positive imaginary part
                is_real: false,
                isolating_interval: Some((0.into(), 0.into())),
                complex_interval: Some((1.into(), 1.into())),
            })),
        }
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
/// use symbolica::{
///     atom::{Atom, AtomCore},
///     domains::{algebraic_number::AlgebraicExtension, rational::Q, Ring, RingOps},
///     parse,
/// };
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
        if element.poly.is_constant() {
            return Ok(!element.poly.get_constant().is_negative());
        }

        if let Some(embedding) = &self.embedding {
            if embedding.is_real {
                // use the isolating interval if available
                if let Some((l, h)) = &embedding.isolating_interval {
                    let mut tolerance = (h - l) / (l + h).abs();
                    let (mut l, mut h) = (l.clone(), h.clone());
                    loop {
                        let eval_lower = element.poly.replace_all(&[l.clone()]);
                        let eval_upper = element.poly.replace_all(&[h.clone()]);

                        if !eval_lower.is_negative() && !eval_upper.is_negative() {
                            return Ok(true);
                        } else if eval_lower.is_negative() && eval_upper.is_negative() {
                            return Ok(false);
                        }

                        // refine the interval
                        let uni = self.poly.to_univariate_from_univariate(0);
                        tolerance *= &(1, 2).into();
                        (l, h) = uni.refine_root_interval((l, h), &tolerance);
                    }
                }

                Err("Isolating interval missing for number field".to_string())
            } else {
                Err(format!(
                    "Cannot determine the sign of a non-real algebraic number {}",
                    self.printer(element)
                ))
            }
        } else {
            Err(format!(
                "Cannot determine the sign of an algebraic number without embedding information {}",
                self.printer(element)
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::atom::AtomCore;
    use crate::domains::algebraic_number::{AlgebraicExtension, RootInfo};
    use crate::domains::finite_field::{PrimeIteratorU64, Z2, Zp};
    use crate::domains::integer::Z;
    use crate::domains::rational::Q;
    use crate::domains::{Ring, RingOps};
    use crate::{parse, symbol};

    #[test]
    fn is_algebraic_number_positive() {
        let ring = parse!("a^3 + 3a^2 - 46*a + 1").to_polynomial(&Q, None);
        let ring = AlgebraicExtension::new_with_embedding(
            ring.clone(),
            RootInfo::from_index(2, &ring.to_univariate_from_univariate(0)),
        );

        let a = parse!("1/5a^2-a-1/10").to_polynomial::<_, u16>(&Q, None);
        let a = ring.to_element(a);

        assert_eq!(ring.is_positive(&a), Ok(true));
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
