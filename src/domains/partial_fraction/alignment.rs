//! Align factor bases before comparing coordinates or reconstructing them.
//!
//! A factor's index or irreducible splitting over one prime has no intrinsic
//! meaning over another prime. Map one reference basis (usually over `Q`) to
//! each prime, retain its factors as whole blocks, then use
//! [`PartialFractionAlignment`] to rebase locally factored answers into those
//! blocks. [`PartialFractionLayout`] supplies zero-padded coordinates with the
//! same meaning at every prime. Neither operation guesses a correspondence
//! between unrelated finite-field factors or reconstructs rational numbers.

use super::*;

/// Failure to align bases or import/export a fixed coordinate layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PartialFractionAlignmentError {
    /// Arithmetic conversion is only defined within one coefficient field.
    DifferentFields,
    /// The rings use different univariate variables.
    DifferentVariables,
    /// The source acquired factors after the alignment plan was created.
    SourceBasisChanged,
    /// The target acquired factors after the plan/layout was created.
    TargetBasisChanged,
    /// The inverse of this source factor needs blocks absent from the target basis.
    UncoveredSourceFactor(usize),
    /// Invalid, repeated, or zero-power factor entry in a coordinate layout.
    InvalidLayout,
    /// A layout's size exceeds the representation's index range.
    LayoutTooLarge,
    /// The element needs a polynomial slot or pole power absent from the layout.
    ElementOutsideLayout,
    /// The coordinate vector does not have the prescribed length.
    CoordinateLength { expected: usize, actual: usize },
    /// A coefficient could not be mapped, or the mapped basis is invalid.
    Conversion(PartialFractionError),
}

impl Display for PartialFractionAlignmentError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DifferentFields => {
                f.write_str("basis alignment requires the same coefficient field")
            }
            Self::DifferentVariables => f.write_str("basis alignment requires the same variable"),
            Self::SourceBasisChanged => {
                f.write_str("the source basis changed; rebuild the alignment plan")
            }
            Self::TargetBasisChanged => {
                f.write_str("the target basis changed; rebuild the alignment plan or layout")
            }
            Self::UncoveredSourceFactor(i) => {
                write!(f, "source factor {i} is not covered by the target basis")
            }
            Self::InvalidLayout => f.write_str("invalid partial-fraction coordinate layout"),
            Self::LayoutTooLarge => {
                f.write_str("partial-fraction coordinate layout exceeds the index range")
            }
            Self::ElementOutsideLayout => {
                f.write_str("the element is not covered by the coordinate layout")
            }
            Self::CoordinateLength { expected, actual } => {
                write!(f, "expected {expected} coordinates, got {actual}")
            }
            Self::Conversion(e) => Display::fmt(e, f),
        }
    }
}

impl std::error::Error for PartialFractionAlignmentError {}

impl From<PartialFractionError> for PartialFractionAlignmentError {
    fn from(value: PartialFractionError) -> Self {
        Self::Conversion(value)
    }
}

#[derive(Clone, Debug)]
enum FactorImage<F: Field> {
    /// The same monic polynomial, possibly at another factor index.
    Exact(usize),
    /// The target representation of 1/f, precomputed once per source factor.
    Inverse(PartialFraction<F>),
}

/// A reusable conversion between two factor bases over the **same** field.
///
/// Exact factor matches use coefficient-block permutation/copying. Regrouping
/// or splitting factors uses precomputed inverses in the target basis and
/// Horner evaluation of each source pole chain; it never builds a common
/// denominator for the entire element.
///
/// Construction requires every source factor's inverse to be representable
/// using the existing target blocks. It never extends the target basis, so
/// alignment cannot silently change a reconstruction template. This is stronger
/// than checking one element: if an uncovered source factor cancels only in a
/// particular value, convert its reduced fraction via
/// [PartialFractionRing::to_fraction] and
/// [PartialFractionRing::try_from_fraction] instead.
///
/// Both arithmetic rings remain extensible. Any later growth of either captured
/// basis invalidates this plan, including growth through another clone. Rebuild
/// the plan explicitly in that case.
/// Elements passed to [`Self::convert`] must belong to the captured source;
/// like other ring operations, elements do not carry an owner tag.
#[derive(Clone, Debug)]
pub struct PartialFractionAlignment<F: Field> {
    source: PartialFractionRing<F>,
    target: PartialFractionRing<F>,
    source_count: usize,
    target_count: usize,
    images: Vec<FactorImage<F>>,
}

impl<F: Field> PartialFractionAlignment<F> {
    /// Prepare a conversion using only the target's current basis.
    /// Neither construction nor conversion appends denominator blocks.
    pub fn new(
        source: &PartialFractionRing<F>,
        target: &PartialFractionRing<F>,
    ) -> Result<Self, PartialFractionAlignmentError> {
        if source.field != target.field {
            return Err(PartialFractionAlignmentError::DifferentFields);
        }
        if source.variable != target.variable {
            return Err(PartialFractionAlignmentError::DifferentVariables);
        }
        let source_count = source.factor_count();
        let target_count = target.factor_count();
        let mut images = Vec::with_capacity(source_count);
        for i in 0..source_count {
            let factor = source.basis.monic(i as u32);
            let exact = (0..target_count).find(|&j| target.basis.monic(j as u32) == factor);
            images.push(match exact {
                Some(j) => FactorImage::Exact(j),
                None => {
                    // Ordinary fraction import may extend a ring. Alignment
                    // instead builds 1/f from a cover by existing blocks only.
                    let (exponents, multiplier) = target
                        .cover_denominator(factor)
                        .ok_or(PartialFractionAlignmentError::UncoveredSourceFactor(i))?;
                    let mut inverse = target.element_from_dense(multiplier);
                    for (factor, exponent) in exponents {
                        target.div_factor_power_assign(&mut inverse, factor, exponent);
                    }
                    FactorImage::Inverse(inverse)
                }
            });
        }
        let plan = Self {
            source: source.clone(),
            target: target.clone(),
            source_count,
            target_count,
            images,
        };
        plan.check_bases()?;
        Ok(plan)
    }

    fn check_bases(&self) -> Result<(), PartialFractionAlignmentError> {
        if self.source.factor_count() != self.source_count {
            return Err(PartialFractionAlignmentError::SourceBasisChanged);
        }
        if self.target.factor_count() != self.target_count {
            return Err(PartialFractionAlignmentError::TargetBasisChanged);
        }
        Ok(())
    }

    /// The target ring whose elements this plan produces.
    pub fn target(&self) -> &PartialFractionRing<F> {
        &self.target
    }

    /// Convert a source element, preserving its rational-function value.
    pub fn convert(
        &self,
        a: &PartialFraction<F>,
    ) -> Result<PartialFraction<F>, PartialFractionAlignmentError> {
        self.check_bases()?;
        let mut exact = Vec::with_capacity(a.chains.len());
        for c in &a.chains {
            let image = self
                .images
                .get(c.factor as usize)
                .ok_or(PartialFractionAlignmentError::SourceBasisChanged)?;
            if let FactorImage::Exact(j) = image {
                exact.push((*j, a.factor_coefficients(c.factor as usize).to_vec()));
            }
        }
        let mut result = self
            .target
            .element_from_coefficients(a.polynomial_part().to_vec(), exact)
            .ok_or(PartialFractionAlignmentError::InvalidLayout)?;
        for c in &a.chains {
            let FactorImage::Inverse(inverse) = &self.images[c.factor as usize] else {
                continue;
            };
            let degree = self.source.basis.degree(c.factor);
            let mut chain = self.target.zero();
            for block in a
                .factor_coefficients(c.factor as usize)
                .chunks_exact(degree)
                .rev()
            {
                let numerator = self.target.element_from_dense(block.to_vec());
                self.target.add_assign(&mut chain, &numerator);
                chain = self.target.mul(&chain, inverse);
            }
            self.target.add_assign(&mut result, &chain);
        }
        // Basis growth during conversion cannot reinterpret old indices, but
        // rejecting it keeps the plan's advertised snapshot contract explicit.
        self.check_bases()?;
        Ok(result)
    }
}

#[derive(Clone, Debug)]
struct CoordinateBlock {
    factor: usize,
    exponent: usize,
    offset: usize,
    len: usize,
}

/// A prescribed, zero-padded coordinate vector over an explicit factor basis.
///
/// Polynomial slots come first, then pole blocks in increasing factor-index
/// order. Each block has `exponent * degree(factor)` entries, with the same
/// order as [`PartialFraction::factor_coefficients`]. Unlike an element's
/// compact storage, missing poles and trailing zeros retain their slots.
///
/// Use [`Self::map_field`] to obtain corresponding layouts over other primes
/// from one reference layout. Align locally split/reordered answers to each
/// mapped [`Self::ring`] before exporting. Then the coordinate at one index
/// has the same meaning for CRT/rational reconstruction at all accepted primes.
/// A new prime requiring larger pole powers or a polynomial part outside the
/// layout is reported, rather than silently shifting/truncating coordinates.
///
/// The coordinate contract is fixed, but the underlying arithmetic ring remains
/// extensible. Growth through any clone invalidates the layout; rebuild it
/// explicitly instead of changing the meanings of existing coordinates.
///
/// This type provides alignment and coordinates, not CRT, rational
/// reconstruction, factor matching across unrelated fields, or bad-prime
/// detection beyond failures of the explicit coefficient/basis map.
#[derive(Clone, Debug)]
pub struct PartialFractionLayout<F: Field> {
    ring: PartialFractionRing<F>,
    factor_count: usize,
    polynomial_slots: usize,
    blocks: Vec<CoordinateBlock>,
    len: usize,
}

impl<F: Field> PartialFractionLayout<F> {
    /// Prescribe polynomial slots and maximum pole powers in the current basis.
    /// Factor entries must have positive exponents, valid indices, and no
    /// duplicates. Input order is irrelevant; coordinates use sorted indices.
    pub fn new(
        ring: &PartialFractionRing<F>,
        polynomial_slots: usize,
        pole_exponents: &[(usize, usize)],
    ) -> Result<Self, PartialFractionAlignmentError> {
        let factor_count = ring.factor_count();
        let mut factors = pole_exponents.to_vec();
        factors.sort_unstable();
        let mut blocks = Vec::with_capacity(factors.len());
        let mut len = polynomial_slots;
        if len > u32::MAX as usize {
            return Err(PartialFractionAlignmentError::LayoutTooLarge);
        }
        for (position, &(factor, exponent)) in factors.iter().enumerate() {
            if factor >= factor_count
                || exponent == 0
                || (position > 0 && factors[position - 1].0 == factor)
            {
                return Err(PartialFractionAlignmentError::InvalidLayout);
            }
            if exponent > u32::MAX as usize {
                return Err(PartialFractionAlignmentError::LayoutTooLarge);
            }
            let block_len = exponent
                .checked_mul(ring.basis.degree(factor as u32))
                .ok_or(PartialFractionAlignmentError::LayoutTooLarge)?;
            let offset = len;
            len = len
                .checked_add(block_len)
                .filter(|&len| len <= u32::MAX as usize)
                .ok_or(PartialFractionAlignmentError::LayoutTooLarge)?;
            blocks.push(CoordinateBlock {
                factor,
                exponent,
                offset,
                len: block_len,
            });
        }
        let layout = Self {
            ring: ring.clone(),
            factor_count,
            polynomial_slots,
            blocks,
            len,
        };
        layout.check_basis()?;
        Ok(layout)
    }

    fn check_basis(&self) -> Result<(), PartialFractionAlignmentError> {
        if self.ring.factor_count() != self.factor_count {
            Err(PartialFractionAlignmentError::TargetBasisChanged)
        } else {
            Ok(())
        }
    }

    /// The basis to which an element must belong before coordinate export.
    pub fn ring(&self) -> &PartialFractionRing<F> {
        &self.ring
    }

    /// Number of polynomial coefficients, including prescribed trailing zeros.
    pub fn polynomial_slots(&self) -> usize {
        self.polynomial_slots
    }

    /// The pole blocks in coordinate order, as `(factor index, maximum power)`.
    pub fn pole_exponents(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.blocks.iter().map(|b| (b.factor, b.exponent))
    }

    /// The exact number of coefficients in the prescribed vector.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the layout has no coefficient slots.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Export an element of this layout's ring, padding missing slots with zero.
    pub fn export(
        &self,
        a: &PartialFraction<F>,
    ) -> Result<Vec<F::Element>, PartialFractionAlignmentError> {
        self.check_basis()?;
        if a.poly > self.polynomial_slots {
            return Err(PartialFractionAlignmentError::ElementOutsideLayout);
        }
        let mut output = vec![self.ring.field.zero(); self.len];
        output[..a.poly].clone_from_slice(a.polynomial_part());
        for c in &a.chains {
            let i = self
                .blocks
                .binary_search_by_key(&(c.factor as usize), |b| b.factor)
                .map_err(|_| PartialFractionAlignmentError::ElementOutsideLayout)?;
            let block = &self.blocks[i];
            if c.exponent as usize > block.exponent {
                return Err(PartialFractionAlignmentError::ElementOutsideLayout);
            }
            let coefficients = a.factor_coefficients(c.factor as usize);
            output[block.offset..block.offset + coefficients.len()].clone_from_slice(coefficients);
        }
        self.check_basis()?;
        Ok(output)
    }

    /// Import a full coordinate vector, restoring the ring's compact canonical form.
    pub fn import(
        &self,
        coefficients: &[F::Element],
    ) -> Result<PartialFraction<F>, PartialFractionAlignmentError> {
        self.check_basis()?;
        if coefficients.len() != self.len {
            return Err(PartialFractionAlignmentError::CoordinateLength {
                expected: self.len,
                actual: coefficients.len(),
            });
        }
        let factors = self
            .blocks
            .iter()
            .map(|b| (b.factor, coefficients[b.offset..b.offset + b.len].to_vec()))
            .collect();
        let result = self
            .ring
            .element_from_coefficients(coefficients[..self.polynomial_slots].to_vec(), factors)
            .ok_or(PartialFractionAlignmentError::InvalidLayout)?;
        self.check_basis()?;
        Ok(result)
    }

    /// Map the reference basis and retain exactly the same coordinate slots.
    ///
    /// No refactorization occurs: a quadratic may split at one prime and stay
    /// irreducible at another, but remains a quadratic coordinate block here.
    /// Coefficient mapping and pairwise-coprimality checks reject invalid images
    /// through [`PartialFractionRing::map_field`]. This does not guarantee that
    /// every numerator coefficient of a later rational answer is mappable.
    pub fn map_field<G: Field>(
        &self,
        field: G,
        map: impl Fn(&F::Element) -> Option<G::Element>,
    ) -> Result<PartialFractionLayout<G>, PartialFractionAlignmentError> {
        self.check_basis()?;
        let target = self.ring.map_field(field, map)?;
        let result = PartialFractionLayout::new(
            &target,
            self.polynomial_slots,
            &self.pole_exponents().collect::<Vec<_>>(),
        )?;
        self.check_basis()?;
        Ok(result)
    }
}

impl PartialFractionLayout<RationalField> {
    /// Map a rational reference layout to a prime, preserving its unsplit
    /// factor blocks and every coordinate slot. Reject coefficient poles and
    /// primes where different reference factors cease to be coprime.
    pub fn to_finite_field<UField: FiniteFieldWorkspace>(
        &self,
        field: &FiniteField<UField>,
    ) -> Result<PartialFractionLayout<FiniteField<UField>>, PartialFractionAlignmentError>
    where
        Integer: ToFiniteField<UField>,
        FiniteField<UField>: FiniteFieldCore<UField>,
    {
        self.map_field(field.clone(), |c| rational_to_finite_field(c, field))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        domains::{finite_field::Zp64, rational::Q},
        symbol,
    };

    fn variable() -> Arc<PolyVariable> {
        Arc::new(symbol!("pf_alignment_x").into())
    }

    fn q_poly(c: &[i64]) -> UnivariatePolynomial<RationalField> {
        UnivariatePolynomial::from_coefficients(
            &Q,
            c.iter().map(|&c| c.into()).collect(),
            variable(),
        )
    }

    fn fp_poly(field: &Zp64, c: &[i64]) -> UnivariatePolynomial<Zp64> {
        UnivariatePolynomial::from_coefficients(
            field,
            c.iter().map(|&c| field.nth(c.into())).collect(),
            variable(),
        )
    }

    #[test]
    fn exact_factor_permutation_preserves_all_powers() {
        let source =
            PartialFractionRing::with_basis(Q, variable(), &[q_poly(&[-1, 1]), q_poly(&[2, 1])])
                .unwrap();
        let target =
            PartialFractionRing::with_basis(Q, variable(), &[q_poly(&[2, 1]), q_poly(&[-1, 1])])
                .unwrap();
        let a = source
            .element_from_coefficients(
                vec![3.into(), 4.into()],
                vec![(0, vec![5.into(), 6.into()]), (1, vec![7.into()])],
            )
            .unwrap();
        let plan = PartialFractionAlignment::new(&source, &target).unwrap();
        assert!(
            plan.images
                .iter()
                .all(|image| matches!(image, FactorImage::Exact(_)))
        );
        let b = plan.convert(&a).unwrap();
        assert_eq!(b.polynomial_part(), a.polynomial_part());
        assert_eq!(b.factor_coefficients(0), a.factor_coefficients(1));
        assert_eq!(b.factor_coefficients(1), a.factor_coefficients(0));
        assert_eq!(source.to_fraction(&a), target.to_fraction(&b));
    }

    #[test]
    fn split_and_grouped_quadratic_bases_round_trip() {
        let field = Zp64::new(5);
        let grouped = PartialFractionRing::with_basis(
            field.clone(),
            variable(),
            &[fp_poly(&field, &[1, 0, 1])],
        )
        .unwrap();
        let split = PartialFractionRing::with_basis(
            field.clone(),
            variable(),
            &[fp_poly(&field, &[-2, 1]), fp_poly(&field, &[2, 1])],
        )
        .unwrap();
        let a = split
            .element_from_coefficients(
                vec![field.nth(3.into())],
                vec![
                    (0, vec![field.nth(2.into()), field.one(), field.one()]),
                    (1, vec![field.one(), field.nth(2.into())]),
                ],
            )
            .unwrap();
        let to_grouped = PartialFractionAlignment::new(&split, &grouped).unwrap();
        let to_split = PartialFractionAlignment::new(&grouped, &split).unwrap();
        let b = to_grouped.convert(&a).unwrap();
        assert_eq!(split.to_fraction(&a), grouped.to_fraction(&b));
        assert_eq!(to_split.convert(&b).unwrap(), a);
        // A value with cancellation inside the source composite factor.
        let cancelled = grouped
            .try_from_fraction(&fp_poly(&field, &[2, 1]), &fp_poly(&field, &[1, 0, 1]))
            .unwrap();
        let actual = to_split.convert(&cancelled).unwrap();
        assert_eq!(actual, split.inv_factor_power(0, 1));
    }

    #[test]
    fn common_template_gives_matching_coordinate_meanings_across_primes() {
        let q_ring = PartialFractionRing::with_basis(Q, variable(), &[q_poly(&[1, 0, 1])]).unwrap();
        let q_layout = PartialFractionLayout::new(&q_ring, 2, &[(0, 2)]).unwrap();
        let a = q_ring
            .element_from_coefficients(
                vec![2.into()],
                vec![(0, vec![3.into(), 1.into(), 4.into(), 2.into()])],
            )
            .unwrap();
        let q_coordinates = q_layout.export(&a).unwrap();
        for prime in [3, 5] {
            let field = Zp64::new(prime);
            let layout = q_layout.to_finite_field(&field).unwrap();
            assert_eq!(layout.len(), 6);
            assert_eq!(layout.ring().factor_count(), 1);
            assert_eq!(layout.ring().factor(0).degree(), 2);
            let local =
                PartialFractionRing::new(field.clone(), variable(), &[fp_poly(&field, &[1, 0, 1])]);
            assert_eq!(local.factor_count(), if prime == 3 { 1 } else { 2 });
            let mapped = q_ring.element_to_finite_field(layout.ring(), &a).unwrap();
            let (n, d) = layout.ring().to_fraction(&mapped);
            let locally_factored = local.try_from_fraction(&n, &d).unwrap();
            let plan = PartialFractionAlignment::new(&local, layout.ring()).unwrap();
            let aligned = plan.convert(&locally_factored).unwrap();
            let expected: Vec<_> = q_coordinates
                .iter()
                .map(|c| rational_to_finite_field(c, &field).unwrap())
                .collect();
            assert_eq!(layout.export(&aligned).unwrap(), expected);
            assert_eq!(layout.import(&expected).unwrap(), mapped);
        }
    }

    #[test]
    fn incompatible_fields_variables_and_uncovered_factors_are_rejected() {
        let f5 = Zp64::new(5);
        let f7 = Zp64::new(7);
        let source =
            PartialFractionRing::with_basis(f5.clone(), variable(), &[fp_poly(&f5, &[-1, 1])])
                .unwrap();
        let other_prime =
            PartialFractionRing::with_basis(f7.clone(), variable(), &[fp_poly(&f7, &[-1, 1])])
                .unwrap();
        assert!(matches!(
            PartialFractionAlignment::new(&source, &other_prime),
            Err(PartialFractionAlignmentError::DifferentFields)
        ));
        let other_var = PartialFractionRing::with_basis(
            f5.clone(),
            Arc::new(symbol!("pf_alignment_y").into()),
            &[],
        )
        .unwrap();
        assert!(matches!(
            PartialFractionAlignment::new(&source, &other_var),
            Err(PartialFractionAlignmentError::DifferentVariables)
        ));
        let missing = PartialFractionRing::with_basis(f5, variable(), &[]).unwrap();
        assert!(matches!(
            PartialFractionAlignment::new(&source, &missing),
            Err(PartialFractionAlignmentError::UncoveredSourceFactor(0))
        ));
        assert_eq!(missing.factor_count(), 0);
    }

    #[test]
    fn alignment_rejects_partial_coverage_without_extending_target() {
        let field = Zp64::new(5);
        let source = PartialFractionRing::with_basis(
            field.clone(),
            variable(),
            &[fp_poly(&field, &[1, 0, 1]), fp_poly(&field, &[-1, 1])],
        )
        .unwrap();
        let target = PartialFractionRing::with_basis(
            field.clone(),
            variable(),
            &[fp_poly(&field, &[-2, 1]), fp_poly(&field, &[2, 1])],
        )
        .unwrap();
        let layout = PartialFractionLayout::new(&target, 0, &[(0, 1), (1, 1)]).unwrap();
        // The first factor can be split over existing blocks. The second
        // would extend the target during an ordinary fraction import.
        assert!(matches!(
            PartialFractionAlignment::new(&source, &target),
            Err(PartialFractionAlignmentError::UncoveredSourceFactor(1))
        ));
        assert_eq!(target.factor_count(), 2);
        assert_eq!(
            layout.export(&target.zero()).unwrap(),
            vec![field.zero(); 2]
        );
    }

    #[test]
    fn layout_preserves_zeros_and_checks_bounds() {
        let ring =
            PartialFractionRing::with_basis(Q, variable(), &[q_poly(&[-1, 1]), q_poly(&[1, 0, 1])])
                .unwrap();
        let layout = PartialFractionLayout::new(&ring, 3, &[(1, 2), (0, 3)]).unwrap();
        let a = ring
            .element_from_coefficients(vec![5.into()], vec![(0, vec![2.into()])])
            .unwrap();
        assert_eq!(layout.len(), 10);
        let coefficients = layout.export(&a).unwrap();
        assert_eq!(
            coefficients,
            vec![
                Rational::from(5),
                0.into(),
                0.into(),
                2.into(),
                0.into(),
                0.into(),
                0.into(),
                0.into(),
                0.into(),
                0.into()
            ]
        );
        assert_eq!(layout.import(&coefficients).unwrap(), a);
        assert_eq!(layout.export(&ring.zero()).unwrap(), vec![Q.zero(); 10]);
        assert!(matches!(
            layout.import(&[]),
            Err(PartialFractionAlignmentError::CoordinateLength {
                expected: 10,
                actual: 0
            })
        ));
        assert_eq!(
            layout.export(&ring.inv_factor_power(0, 4)),
            Err(PartialFractionAlignmentError::ElementOutsideLayout)
        );
        let polynomial = ring.element_from_polynomial(&q_poly(&[0, 0, 0, 1]));
        assert_eq!(
            layout.export(&polynomial),
            Err(PartialFractionAlignmentError::ElementOutsideLayout)
        );
        let no_poles = PartialFractionLayout::new(&ring, 1, &[]).unwrap();
        assert_eq!(
            no_poles.export(&a),
            Err(PartialFractionAlignmentError::ElementOutsideLayout)
        );
        for factors in [&[(0, 0)][..], &[(0, 1), (0, 2)][..], &[(2, 1)][..]] {
            assert!(matches!(
                PartialFractionLayout::new(&ring, 1, factors),
                Err(PartialFractionAlignmentError::InvalidLayout)
            ));
        }
        assert!(matches!(
            PartialFractionLayout::new(&ring, usize::MAX, &[(0, 1)]),
            Err(PartialFractionAlignmentError::LayoutTooLarge)
        ));
    }

    #[test]
    fn captured_source_and_target_growth_is_rejected() {
        let source = PartialFractionRing::new(Q, variable(), &[q_poly(&[-1, 1])]);
        let target = PartialFractionRing::new(Q, variable(), &[q_poly(&[-1, 1]), q_poly(&[-2, 1])]);
        let plan = PartialFractionAlignment::new(&source, &target).unwrap();
        let a = source.inv_factor_power(0, 1);
        let source_alias = source.clone();
        source_alias
            .try_from_fraction(&q_poly(&[1]), &q_poly(&[-2, 1]))
            .unwrap();
        assert_eq!(
            plan.convert(&a),
            Err(PartialFractionAlignmentError::SourceBasisChanged)
        );
        let plan = PartialFractionAlignment::new(&source, &target).unwrap();
        let layout = PartialFractionLayout::new(&target, 1, &[(0, 1), (1, 1)]).unwrap();
        let target_alias = target.clone();
        target_alias
            .try_from_fraction(&q_poly(&[1]), &q_poly(&[-3, 1]))
            .unwrap();
        assert_eq!(
            plan.convert(&a),
            Err(PartialFractionAlignmentError::TargetBasisChanged)
        );
        assert_eq!(
            layout.export(&target.zero()),
            Err(PartialFractionAlignmentError::TargetBasisChanged)
        );
    }

    #[test]
    fn reference_layout_rejects_bad_prime_collisions() {
        let ring =
            PartialFractionRing::with_basis(Q, variable(), &[q_poly(&[-1, 1]), q_poly(&[-6, 1])])
                .unwrap();
        let layout = PartialFractionLayout::new(&ring, 0, &[(0, 1), (1, 1)]).unwrap();
        let field = Zp64::new(5);
        assert!(matches!(
            layout.map_field(field.clone(), |c| rational_to_finite_field(c, &field)),
            Err(PartialFractionAlignmentError::Conversion(
                PartialFractionError::NotCoprime(..)
            ))
        ));
    }
}
