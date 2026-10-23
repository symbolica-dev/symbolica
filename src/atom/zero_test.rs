//! Test whether expressions are identically zero.

use crate::{
    atom::{Atom, AtomCore, AtomOrView, AtomView, FunctionBuilder, Symbol},
    coefficient::{CoefficientView, ConvertToRing},
    domains::{
        Field, Ring, RingOps, UniformlySampleableRing,
        algebraic::{AlgebraicContext, AlgebraicExtension},
        finite_field::{FiniteFieldCore, FiniteFieldElement, Zp64},
        float::{Complex, ErrorPropagatingFloat, Float, FloatLike, Real, SingleFloat},
        integer::Integer,
        rational::{Q, Rational},
    },
    id::{AtomTreeIterator, ConditionResult, MatchSettings},
    numerical_integration::MonteCarloRng,
    poly::PolyVariable,
    transcendental,
};
use ahash::HashMap;
use rand::RngCore;

/// Settings for [`AtomCore::check_zero_with`] and [`AtomCore::check_equal_with`].
/// Defaults match [`AtomCore::check_zero`] and [`AtomCore::check_equal`].
///
/// Control the work and numerical precision used to test zero and equality, and
/// choose how floating coefficients are interpreted. Higher limits may resolve
/// an inconclusive result but do not guarantee an answer. See
/// [`AtomCore::check_zero`] for the result meanings and accuracy limitations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ZeroTestSettings {
    finite_field_samples: usize,
    numerical_samples: usize,
    max_precision: u32,
    exact_fallback: bool,
    treat_floats_as_exact: bool,
}

impl Default for ZeroTestSettings {
    fn default() -> Self {
        Self {
            finite_field_samples: SAMPLE_PRIMES.len(),
            numerical_samples: 5,
            max_precision: 1024,
            exact_fallback: true,
            treat_floats_as_exact: false,
        }
    }
}

impl ZeroTestSettings {
    /// Create the settings used by [`AtomCore::check_zero`] and [`AtomCore::check_equal`].
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the maximum number of finite-field samples (default: five).
    ///
    /// More samples can help detect nonzero expressions, but cannot prove zero.
    /// Values above five have no additional effect. Zero disables these samples.
    pub fn finite_field_samples(mut self, samples: usize) -> Self {
        self.finite_field_samples = samples;
        self
    }

    /// Set the maximum number of numerical sample points (default: five).
    ///
    /// More points can help detect nonzero expressions. Constants need only one
    /// point. Zero disables numerical testing, including for constants, so
    /// numerical heuristics cannot produce a nonzero result.
    pub fn numerical_samples(mut self, samples: usize) -> Self {
        self.numerical_samples = samples;
        self
    }

    /// Set the maximum numerical working precision in **bits** (default: 1024).
    ///
    /// Increase this to help distinguish very small nonzero values from zero,
    /// at the cost of more computation. Values below 128 restrict numerical
    /// testing to machine precision.
    ///
    /// This does not increase the accuracy of existing floating coefficients;
    /// see [`Self::treat_floats_as_exact`]. Unlike this setting,
    /// [`to_float`](AtomCore::to_float) takes its precision in decimal digits.
    pub fn max_precision(mut self, bits: u32) -> Self {
        self.max_precision = bits;
        self
    }

    /// Allow more expensive symbolic checks to resolve an otherwise inconclusive
    /// result (default: true).
    ///
    /// Disabling this can make checks faster, but leaves more zero identities
    /// undecided. The sample limits do not bound the cost of these checks.
    pub fn exact_fallback(mut self, enabled: bool) -> Self {
        self.exact_fallback = enabled;
        self
    }

    /// Treat finite floating coefficients as exact numbers (default: false).
    ///
    /// Enable this when you want to test the exact stored values, rather than
    /// the mathematical values they approximate. Their rounding uncertainty is
    /// then ignored. This applies throughout the expression, including function
    /// arguments and exponents, and does not recover accuracy lost when the
    /// coefficients were created. NaN and infinities are unaffected.
    ///
    /// For example, `pi - pi.to_float(10000)` can be distinguished numerically
    /// with this option and an 80,000-bit working-precision limit. Without this
    /// option, that residual remains inconclusive even at higher working precision.
    /// See [`AtomCore::check_zero_with`] for a complete example.
    pub fn treat_floats_as_exact(mut self, enabled: bool) -> Self {
        self.treat_floats_as_exact = enabled;
        self
    }
}

struct SamplingPrime {
    prime: u64,
    subgroup_order: u64,
    // Ordinary residue, converted to the field's representation on first use.
    imaginary_unit: u64,
}

// Prevalidated p = 4q + 1 with odd prime q and i^2 = -1 mod p. Tests verify
// these constants; probes need neither primality tests nor a search for i.
const SAMPLE_PRIMES: [SamplingPrime; 5] = [
    SamplingPrime {
        prime: 18446744073709548557,
        subgroup_order: 4611686018427387139,
        imaginary_unit: 3770892738701069429,
    },
    SamplingPrime {
        prime: 18446744073709542869,
        subgroup_order: 4611686018427385717,
        imaginary_unit: 6469811227597262597,
    },
    SamplingPrime {
        prime: 18446744073709542629,
        subgroup_order: 4611686018427385657,
        imaginary_unit: 3179777996628670214,
    },
    SamplingPrime {
        prime: 18446744073709533053,
        subgroup_order: 4611686018427383263,
        imaginary_unit: 2660601163918621771,
    },
    SamplingPrime {
        prime: 18446744073709528397,
        subgroup_order: 4611686018427382099,
        imaginary_unit: 7640212023282358280,
    },
];

impl SamplingPrime {
    fn for_prime(prime: u64) -> Option<&'static Self> {
        SAMPLE_PRIMES.iter().find(|sample| sample.prime == prime)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Structural {
    Zero,
    Nonzero,
    // Admissible, but neither zero nor nonzero has been established.
    Defined,
}

impl AtomView<'_> {
    /// Test for structural zero, for example due to monomials with unique formal variables.
    /// This is a cheap, bounded check that does not require sampling or numerical evaluation.
    fn structural_zero(&self, depth: usize, budget_left: &mut usize) -> Option<Structural> {
        if depth >= 64 || *budget_left == 0 {
            return None;
        }
        *budget_left -= 1;
        match *self {
            AtomView::Num(n) => {
                if let Some(zero) = n.small_integer_is_zero() {
                    return Some(if zero {
                        Structural::Zero
                    } else {
                        Structural::Nonzero
                    });
                }
                match n.get_coeff_view() {
                    CoefficientView::Natural(..) | CoefficientView::Large(..) => {}
                    CoefficientView::Float(re, im) => {
                        // The serialized mantissa of a finite float contains no
                        // 'i' or 'n'; avoid decoding an ordinary numeric leaf.
                        for part in [re.0, im.0] {
                            if part[4..]
                                .iter()
                                .any(|b| matches!(b.to_ascii_lowercase(), b'i' | b'n'))
                            {
                                return None;
                            }
                        }
                    }
                    _ => return None,
                }
                Some(if n.is_zero() {
                    Structural::Zero
                } else {
                    Structural::Nonzero
                })
            }
            AtomView::Var(v) => (v.get_symbol().is_formal_sample_variable()
                || [Symbol::E, Symbol::PI].contains(&v.get_symbol()))
            .then_some(Structural::Nonzero),
            AtomView::Pow(p) => {
                let (base, exponent) = p.get_base_exp();
                if matches!(base, AtomView::Var(v) if v.get_symbol() == Symbol::E) {
                    exponent.structural_zero(depth + 1, budget_left)?;
                    return Some(Structural::Nonzero);
                }
                let e = Complex::<Rational>::try_from(exponent).ok()?;
                if !e.is_real() {
                    return None;
                }
                let base = base.structural_zero(depth + 1, budget_left)?;
                if e.re.is_negative() || e.re.is_zero() {
                    (base == Structural::Nonzero).then_some(Structural::Nonzero)
                } else {
                    // Positive rational powers, including sqrt, preserve zeros.
                    Some(base)
                }
            }
            AtomView::Mul(m) => {
                let mut zero = false;
                let mut nonzero = true;
                for arg in m {
                    let value = arg.structural_zero(depth + 1, budget_left)?;
                    zero |= value == Structural::Zero;
                    nonzero &= value == Structural::Nonzero;
                }
                Some(if zero {
                    Structural::Zero
                } else if nonzero {
                    Structural::Nonzero
                } else {
                    Structural::Defined
                })
            }
            AtomView::Add(a) => {
                let mut all_zero = true;
                for arg in a {
                    all_zero &= arg.structural_zero(depth + 1, budget_left)? == Structural::Zero;
                }
                if all_zero {
                    return Some(Structural::Zero);
                }
                for (index, term) in a.iter().enumerate() {
                    let Some(symbol) = term.separating_variable(depth + 1, budget_left) else {
                        continue;
                    };
                    let mut unique = true;
                    for (other, term) in a.iter().enumerate() {
                        if index != other
                            && term.contains_symbol_bounded(symbol, depth + 1, budget_left)?
                        {
                            unique = false;
                            break;
                        }
                    }
                    if unique {
                        return Some(Structural::Nonzero);
                    }
                }
                Some(Structural::Defined)
            }
            AtomView::Fun(f) if f.get_symbol().is_opaque_sample_function() => {
                // Same generic independent-call assumption as the modular sampler.
                Some(Structural::Nonzero)
            }
            AtomView::Fun(f) if f.get_nargs() == 1 => {
                let symbol = f.get_symbol();
                if ![Symbol::ABS, Symbol::CONJ, Symbol::SIN, Symbol::COS].contains(&symbol) {
                    return None;
                }
                let value = f.iter().next()?.structural_zero(depth + 1, budget_left)?;
                Some(if symbol == Symbol::ABS || symbol == Symbol::CONJ {
                    value
                } else {
                    Structural::Defined
                })
            }
            _ => None,
        }
    }

    fn separating_variable(&self, depth: usize, left: &mut usize) -> Option<Symbol> {
        let variable = |a| match a {
            AtomView::Var(v) if v.get_symbol().is_formal_sample_variable() => Some(v.get_symbol()),
            AtomView::Pow(p) => {
                let AtomView::Var(v) = p.get_base() else {
                    return None;
                };
                let n = match p.get_exp() {
                    AtomView::Num(n) if n.small_integer().is_some() => {
                        n.small_integer().unwrap().into()
                    }
                    exponent => Integer::try_from(exponent).ok()?,
                };
                (v.get_symbol().is_formal_sample_variable() && !n.is_zero())
                    .then_some(v.get_symbol())
            }
            _ => None,
        };
        if let Some(s) = variable(*self) {
            return Some(s);
        }
        if let AtomView::Mul(m) = *self {
            for (index, factor) in m.iter().enumerate() {
                let Some(s) = variable(factor) else {
                    continue;
                };
                let mut valid = true;
                for (other, coefficient) in m.iter().enumerate() {
                    if index != other
                        && (coefficient.contains_symbol_bounded(s, depth + 1, left)?
                            || coefficient.structural_zero(depth + 1, left)? != Structural::Nonzero)
                    {
                        valid = false;
                        break;
                    }
                }
                if valid {
                    return Some(s);
                }
            }
        }
        None
    }

    fn contains_symbol_bounded(
        &self,
        symbol: Symbol,
        depth: usize,
        left: &mut usize,
    ) -> Option<bool> {
        if depth >= 64 || *left == 0 {
            return None;
        }
        *left -= 1;
        match *self {
            AtomView::Var(v) => Some(v.get_symbol() == symbol),
            AtomView::Num(n) => match n.get_coeff_view() {
                // Its hidden variables are deliberately not inspected here.
                CoefficientView::RationalPolynomial(_) => None,
                _ => Some(false),
            },
            AtomView::Pow(p) => Some(
                p.get_base()
                    .contains_symbol_bounded(symbol, depth + 1, left)?
                    || p.get_exp()
                        .contains_symbol_bounded(symbol, depth + 1, left)?,
            ),
            _ => {
                let args = match *self {
                    AtomView::Fun(f) => f.to_slice(),
                    AtomView::Add(a) => a.to_slice(),
                    AtomView::Mul(m) => m.to_slice(),
                    _ => unreachable!(),
                };
                for arg in args.iter() {
                    if arg.contains_symbol_bounded(symbol, depth + 1, left)? {
                        return Some(true);
                    }
                }
                Some(false)
            }
        }
    }

    /// Test whether this expression is identically zero using the default settings.
    ///
    /// Returns `True` for a zero identity, `False` for a nonzero result, or
    /// `Inconclusive` if undecided. See [`AtomCore::check_zero`] for examples and
    /// the numerical and user-defined function limitations on nonzero results.
    #[inline]
    pub fn check_zero(&self) -> ConditionResult {
        self.check_zero_with(&ZeroTestSettings::default())
    }

    /// Test for an identically zero expression with configurable sampling and precision.
    /// Defaults match [`check_zero`](Self::check_zero). See [`AtomCore::check_zero_with`]
    /// for examples and the interpretation of floating coefficients.
    pub fn check_zero_with(&self, settings: &ZeroTestSettings) -> ConditionResult {
        if settings.treat_floats_as_exact {
            let exact = self.exact_floating_coefficients();
            exact.as_view().zero_test_impl(settings)
        } else {
            self.zero_test_impl(settings)
        }
    }

    /// Test whether two expressions are mathematically equal using the default settings.
    ///
    /// Returns `True`, `False`, or `Inconclusive`. See [`AtomCore::check_equal`]
    /// for examples and the numerical and user-defined function limitations.
    #[inline]
    pub fn check_equal<'a, T: Into<AtomOrView<'a>>>(&self, other: T) -> ConditionResult {
        self.check_equal_with(other, &ZeroTestSettings::default())
    }

    /// Test mathematical equality with configurable sampling and precision.
    /// See [`AtomCore::check_equal_with`] for examples and the interpretation of
    /// floating coefficients in both expressions.
    pub fn check_equal_with<'a, T: Into<AtomOrView<'a>>>(
        &self,
        other: T,
        settings: &ZeroTestSettings,
    ) -> ConditionResult {
        let other = other.into();
        let difference = if settings.treat_floats_as_exact {
            // Convert before subtraction so rounding cannot erase a difference.
            self.exact_floating_coefficients() - other.as_view().exact_floating_coefficients()
        } else {
            *self - other.as_view()
        };
        let result = difference.as_view().zero_test_impl(settings);
        if result.is_true() && !settings.treat_floats_as_exact && *self != other.as_view() {
            let contains_float = [*self, other.as_view()].into_iter().any(|view| {
                let mut tree = AtomTreeIterator::new(view, MatchSettings::default());
                while let Some(node) = tree.next_into(None) {
                    if matches!(node, AtomView::Num(n) if matches!(n.get_coeff_view(), CoefficientView::Float(..))) {
                        return true;
                    }
                }
                false
            });
            if contains_float {
                // A rounded subtraction can produce zero for distinct inputs.
                // Only retain True if their exact stored values also agree.
                let exact_difference = self.exact_floating_coefficients()
                    - other.as_view().exact_floating_coefficients();
                if !exact_difference
                    .as_view()
                    .zero_test_impl(settings)
                    .is_true()
                {
                    return ConditionResult::Inconclusive;
                }
            }
        }
        result
    }

    fn exact_floating_coefficients(&self) -> Atom {
        self.map_coefficient(|c| {
            if let CoefficientView::Float(re, im) = c {
                let (re, im) = (re.to_float(), im.to_float());
                if re.is_finite() && im.is_finite() {
                    return Complex::new(re.to_rational(), im.to_rational()).into();
                }
            }
            c.to_owned()
        })
    }

    /// Test whether the expression is identically zero with a sample limit.
    ///
    /// Deprecated: use [`check_zero`](Self::check_zero) or
    /// [`check_zero_with`](Self::check_zero_with). See [`AtomCore::zero_test`]
    /// for the meaning of `iterations` and `tolerance`.
    #[deprecated(note = "Use `check_zero()` instead, or `check_zero_with()` for custom settings.")]
    pub fn zero_test(&self, iterations: usize, _tolerance: f64) -> ConditionResult {
        self.check_zero_with(
            &ZeroTestSettings::default()
                .finite_field_samples(iterations)
                .numerical_samples(iterations)
                .exact_fallback(iterations != 0),
        )
    }

    fn zero_test_impl(&self, settings: &ZeroTestSettings) -> ConditionResult {
        match self.structural_zero(0, &mut 512) {
            Some(Structural::Zero) => return ConditionResult::True,
            Some(Structural::Nonzero) => return ConditionResult::False,
            _ => {}
        }
        for sample in SAMPLE_PRIMES.iter().take(settings.finite_field_samples) {
            let field = Zp64::new(sample.prime);
            match self.sample_in_finite_field(&mut HashMap::default(), &field) {
                Ok(value) if !field.is_zero(&value) => return ConditionResult::False,
                Ok(_)
                | Err(
                    PolyError::DivisionByZero | PolyError::NonInvertibleExponentDenominator { .. },
                ) => {}
                // A different sample cannot make an unsupported operation valid.
                Err(_) => break,
            }
        }
        let result = self.numerical_zero_test(settings.numerical_samples, settings.max_precision);
        if result.is_inconclusive() && settings.exact_fallback {
            self.exact_zero_test()
                .unwrap_or(ConditionResult::Inconclusive)
        } else {
            result
        }
    }

    /// The expensive identity proof runs only after all inexpensive witnesses
    /// failed. Formal function atoms can prove cancellation, but their algebraic
    /// independence must never be used as an analytic nonzero certificate here.
    fn exact_zero_test(&self) -> Result<ConditionResult, String> {
        let mut expression = self.together();
        if expression.is_zero() {
            return Ok(ConditionResult::True);
        }
        expression = expression.as_view().zero_test_normalize(None)?;
        let mut context = AlgebraicContext::new(AlgebraicExtension::trivial(Q));
        expression
            .as_view()
            .zero_test_collect_algebraics(&mut context)?;
        // Products and sums reconstructed by normalization can combine exponential
        // arguments. Revisit those arguments before comparing formal generators.
        for _ in 0..4 {
            let next = expression
                .as_view()
                .zero_test_normalize(Some(&mut context))?;
            if next == expression {
                break;
            }
            expression = next;
        }
        let rational = context.to_rational_polynomial::<u32>(expression.as_view(), None)?;
        if rational.numerator.is_zero() {
            return Ok(ConditionResult::True);
        }
        if rational.numerator.get_vars_ref().iter().all(
            |variable| matches!(variable, PolyVariable::Symbol(s) if s.is_formal_sample_variable()),
        ) {
            Ok(ConditionResult::False)
        } else {
            Ok(ConditionResult::Inconclusive)
        }
    }

    /// Collect constants throughout the tree before choosing a common field's
    /// representations, including constants inside otherwise opaque arguments.
    fn zero_test_collect_algebraics(&self, context: &mut AlgebraicContext) -> Result<(), String> {
        match self {
            AtomView::Fun(f) if f.get_symbol() == transcendental::root() => {
                // root's polynomial variable is bound, not a free argument.
            }
            AtomView::Fun(f) => {
                for argument in *f {
                    argument.zero_test_collect_algebraics(context)?;
                }
            }
            AtomView::Pow(p) => {
                let (base, exponent) = p.get_base_exp();
                if let Ok(exponent) = Rational::try_from(exponent)
                    && exponent.denominator() > Integer::from(256)
                {
                    return Err("Algebraic root degree exceeds the zero-test budget".into());
                }
                base.zero_test_collect_algebraics(context)?;
                exponent.zero_test_collect_algebraics(context)?;
            }
            AtomView::Add(a) => {
                for term in *a {
                    term.zero_test_collect_algebraics(context)?;
                }
            }
            AtomView::Mul(m) => {
                for factor in *m {
                    factor.zero_test_collect_algebraics(context)?;
                }
            }
            _ => {}
        }
        context.extend(*self)?;
        Ok(())
    }

    /// Rewrite circular/hyperbolic functions using exponentials and normalize
    /// arguments and exponents recursively. Algebraic coefficients are always
    /// written in the same primitive element, independently of their input form.
    fn zero_test_normalize(
        &self,
        mut context: Option<&mut AlgebraicContext>,
    ) -> Result<Atom, String> {
        let expression = match *self {
            AtomView::Fun(f) if f.get_symbol() == transcendental::root() => self.to_owned(),
            AtomView::Fun(f) => {
                let mut arguments = Vec::with_capacity(f.get_nargs());
                for argument in f {
                    arguments.push(argument.zero_test_normalize(context.as_deref_mut())?);
                }
                if arguments.len() == 1
                    && let Some((function, hyperbolic)) =
                        SampledGeometricFunction::from_symbol(f.get_symbol())
                {
                    let unit = if hyperbolic { Atom::num(1) } else { Atom::i() };
                    let phase = (&unit * &arguments[0]).together();
                    let positive = phase.as_view().zero_test_exponential();
                    let negative = (-phase).as_view().zero_test_exponential();
                    let sine = (&positive - &negative) / (&unit * 2);
                    let cosine = (positive + negative) / 2;
                    match function {
                        SampledGeometricFunction::Sin => sine,
                        SampledGeometricFunction::Cos => cosine,
                        SampledGeometricFunction::Tan => sine / cosine,
                        SampledGeometricFunction::Cot => cosine / sine,
                        SampledGeometricFunction::Sec => cosine.pow(-1),
                        SampledGeometricFunction::Csc => sine.pow(-1),
                    }
                } else {
                    let mut builder = FunctionBuilder::new(f.get_symbol());
                    for argument in arguments {
                        builder = builder.add_arg(argument);
                    }
                    builder.finish()
                }
            }
            AtomView::Pow(p) => {
                let (base, exponent) = p.get_base_exp();
                let exponent = exponent.zero_test_normalize(context.as_deref_mut())?;
                if matches!(base, AtomView::Var(v) if v.get_symbol() == Symbol::E) {
                    exponent.as_view().zero_test_exponential()
                } else {
                    base.zero_test_normalize(context.as_deref_mut())?
                        .pow(exponent)
                }
            }
            AtomView::Add(a) => {
                let mut sum = Atom::num(0);
                for term in a {
                    sum += term.zero_test_normalize(context.as_deref_mut())?;
                }
                sum
            }
            AtomView::Mul(m) => {
                let mut product = Atom::num(1);
                for factor in m {
                    product *= factor.zero_test_normalize(context.as_deref_mut())?;
                }
                product
            }
            _ => self.to_owned(),
        };
        if let Some(context) = context {
            let rational = context.to_rational_polynomial::<u32>(expression.as_view(), None)?;
            Ok(rational.to_expression_with_coeff_map(|field, value, out| {
                *out = field.element_to_atom(value);
            }))
        } else {
            Ok(expression.together())
        }
    }

    /// exp(i*pi*r) = (-1)^r for rational r, with the principal branch of -1.
    /// The algebraic stage then supplies the cyclotomic relations. Other pi
    /// dependence stays analytic; pi is never replaced by a formal scalar here.
    fn zero_test_exponential(&self) -> Atom {
        let expanded = if self.is_expanded(None) {
            self.to_owned()
        } else {
            self.expand()
        };
        if let Ok((regular, period)) = expanded.as_view().split_exponential_period()
            && let Ok(period) = Complex::<Rational>::try_from(period.as_view())
            && period.re.is_zero()
            && !period.im.is_zero()
        {
            return Atom::num(-1).pow(Atom::num(period.im)) * regular.exp();
        }
        self.exp()
    }

    /// Validate a rational exponent without expanding it. The result records
    /// whether pi occurs outside opaque calls and needs exact period extraction.
    fn validate_rational_exponent(&self) -> Result<bool, PolyError> {
        match self {
            AtomView::Num(n) => match n.get_coeff_view() {
                CoefficientView::Natural(..) | CoefficientView::Large(..) => Ok(false),
                _ => Err(PolyError::UnsupportedCoefficient),
            },
            AtomView::Var(v) if v.get_symbol() == Symbol::PI => Ok(true),
            AtomView::Var(v) if v.get_symbol().is_formal_sample_variable() => Ok(false),
            AtomView::Var(v) => Err(PolyError::UnsupportedConstant(v.get_symbol())),
            AtomView::Add(a) => {
                let mut has_pi = false;
                for term in a {
                    has_pi |= term.validate_rational_exponent()?;
                }
                Ok(has_pi)
            }
            AtomView::Mul(m) => {
                let mut has_pi = false;
                for factor in m {
                    has_pi |= factor.validate_rational_exponent()?;
                }
                Ok(has_pi)
            }
            AtomView::Pow(p) => {
                let (base, exponent) = p.get_base_exp();
                if !Complex::<Rational>::try_from(exponent)
                    .is_ok_and(|e| e.is_real() && e.re.is_integer())
                {
                    return Err(PolyError::NonRationalArgument);
                }
                base.validate_rational_exponent()
            }
            // The whole call, including its arguments, is one formal variable.
            AtomView::Fun(f) if f.get_symbol().is_opaque_sample_function() => Ok(false),
            AtomView::Fun(_) => Err(PolyError::NonRationalArgument),
        }
    }

    /// Split A + pi*B without assigning a random value to pi. We only admit
    /// linear pi dependence; the caller additionally requires B to be constant.
    fn split_exponential_period(&self) -> Result<(Atom, Atom), PolyError> {
        match self {
            AtomView::Var(v) if v.get_symbol() == Symbol::PI => Ok((Atom::num(0), Atom::num(1))),
            AtomView::Add(a) => {
                let (mut regular, mut period) = (Atom::num(0), Atom::num(0));
                for term in a {
                    let (r, p) = term.split_exponential_period()?;
                    regular = regular + r;
                    period = period + p;
                }
                Ok((regular, period))
            }
            AtomView::Mul(m) => {
                let (mut regular, mut period) = (Atom::num(1), Atom::num(0));
                for factor in m {
                    let (r, p) = factor.split_exponential_period()?;
                    if !period.is_zero() && !p.is_zero() {
                        return Err(PolyError::UnsupportedPeriod);
                    }
                    period = &period * &r + &regular * p;
                    regular = regular * r;
                }
                Ok((regular, period))
            }
            AtomView::Pow(p) => {
                let (base, exponent) = p.get_base_exp();
                let (regular, period) = base.split_exponential_period()?;
                if period.is_zero() {
                    Ok((regular.pow(exponent), Atom::num(0)))
                } else if exponent.is_one() {
                    Ok((regular, period))
                } else if exponent.is_zero() {
                    Ok((Atom::num(1), Atom::num(0)))
                } else {
                    Err(PolyError::UnsupportedPeriod)
                }
            }
            _ => Ok((self.to_owned(), Atom::num(0))),
        }
    }

    /// Evaluate a rational argument in F_q[i], keeping i formal even when -1
    /// has no square root in F_q. Inverses require a nonzero norm. If F_q[i]
    /// splits, rejecting zero divisors is conservative and still sound.
    fn sample_rational_exponent(
        &self,
        samples: &mut HashMap<AtomOrView<'_>, FiniteFieldElement<u64>>,
        field: &Zp64,
        q: &Zp64,
    ) -> Result<Complex<FiniteFieldElement<u64>>, PolyError> {
        fn mul_exponent_samples(
            q: &Zp64,
            a: Complex<FiniteFieldElement<u64>>,
            b: Complex<FiniteFieldElement<u64>>,
        ) -> Complex<FiniteFieldElement<u64>> {
            Complex::new(
                q.sub(&q.mul(&a.re, &b.re), &q.mul(&a.im, &b.im)),
                q.add(&q.mul(&a.re, &b.im), &q.mul(&a.im, &b.re)),
            )
        }

        match self {
            AtomView::Num(_) => {
                let coefficient = Complex::<Rational>::try_from(*self)
                    .map_err(|_| PolyError::UnsupportedCoefficient)?;
                let convert = |part: &Rational| {
                    let denominator = q.element_from_integer(part.denominator());
                    if q.is_zero(&denominator) {
                        return Err(PolyError::NonInvertibleExponentDenominator {
                            subgroup_order: q.get_prime(),
                        });
                    }
                    Ok(q.div(&q.element_from_integer(part.numerator()), &denominator))
                };
                Ok(Complex::new(
                    convert(&coefficient.re)?,
                    convert(&coefficient.im)?,
                ))
            }
            AtomView::Var(_) | AtomView::Fun(_) => {
                // Reuse the ordinary variable's cached sample to choose a
                // consistent point modulo q, but never mix field encodings.
                let sample = self.sample_in_finite_field_impl(samples, field)?;
                Ok(Complex::new(
                    q.to_element(field.from_element(&sample) % q.get_prime()),
                    q.zero(),
                ))
            }
            AtomView::Add(a) => {
                let mut result = Complex::new(q.zero(), q.zero());
                for term in a {
                    let value = term.sample_rational_exponent(samples, field, q)?;
                    q.add_assign(&mut result.re, &value.re);
                    q.add_assign(&mut result.im, &value.im);
                }
                Ok(result)
            }
            AtomView::Mul(m) => {
                let mut result = Complex::new(q.one(), q.zero());
                for factor in m {
                    let value = factor.sample_rational_exponent(samples, field, q)?;
                    result = mul_exponent_samples(q, result, value);
                }
                Ok(result)
            }
            AtomView::Pow(p) => {
                let (base, exponent) = p.get_base_exp();
                let exponent =
                    Integer::try_from(exponent).map_err(|_| PolyError::NonRationalArgument)?;
                let mut value = base.sample_rational_exponent(samples, field, q)?;
                if exponent.is_negative() {
                    let norm = q.add(&q.mul(&value.re, &value.re), &q.mul(&value.im, &value.im));
                    if q.is_zero(&norm) {
                        return Err(PolyError::NonInvertibleExponentDenominator {
                            subgroup_order: q.get_prime(),
                        });
                    }
                    value = Complex::new(q.div(&value.re, &norm), q.neg(&q.div(&value.im, &norm)));
                }
                let mut result = Complex::new(q.one(), q.zero());
                let mut power = exponent.abs();
                while !power.is_zero() {
                    if &power % 2 == 1 {
                        result = mul_exponent_samples(q, result, value);
                    }
                    power /= 2;
                    if !power.is_zero() {
                        value = mul_exponent_samples(q, value, value);
                    }
                }
                Ok(result)
            }
        }
    }

    /// Sample exp(R), or exp(iR) for circular functions. Rational arguments
    /// share an additive map to the order-q subgroup, so partial fractions and
    /// other rational identities do not need a canonical expanded form.
    fn sample_rational_exponential(
        &self,
        imaginary: bool,
        samples: &mut HashMap<AtomOrView<'_>, FiniteFieldElement<u64>>,
        field: &Zp64,
    ) -> Result<FiniteFieldElement<u64>, PolyError> {
        let has_pi = self.validate_rational_exponent()?;
        let q = exponential_sample_field(field)?;
        let regular;
        let mut phase = field.one();
        let argument = if has_pi {
            let (r, p) = self.split_exponential_period()?;
            let mut coefficient = Complex::<Rational>::try_from(p.as_view())
                .map_err(|_| PolyError::UnsupportedPeriod)?;
            if imaginary {
                coefficient = Complex::new(-coefficient.im, coefficient.re);
            }
            let turns = &coefficient.im * &Rational::from(2);
            if !coefficient.re.is_zero() || !turns.is_integer() {
                return Err(PolyError::UnsupportedPeriod);
            }
            let turns = turns.numerator();
            let power = u64::try_from(turns.abs() % 4).unwrap();
            let i = sample_imaginary_unit(samples, field)?;
            phase = field.pow(&i, power);
            if turns.is_negative() {
                phase = field.inv(&phase);
            }
            regular = r;
            regular.as_view()
        } else {
            *self
        };
        let mut value = argument.sample_rational_exponent(samples, field, &q)?;
        if imaginary {
            value = Complex::new(q.neg(&value.im), value.re);
        }
        let mut result = phase;
        for (part, key) in [
            (value.re, Symbol::E.into()),
            (
                value.im,
                Atom::var(Symbol::E)
                    .pow(Atom::num(Complex::new(Rational::zero(), Rational::one())))
                    .into(),
            ),
        ] {
            if !q.is_zero(&part) {
                let generator = sample_exponential_generator(key, samples, field, &q)?;
                field.mul_assign(&mut result, &field.pow(&generator, q.from_element(&part)));
            }
        }
        Ok(result)
    }

    /// Sample a rational combination of variables, exact Gaussian rationals,
    /// exp(R), and circular/hyperbolic sin, cos, tan, cot, sec, csc of rational R.
    /// R cannot contain built-in functions, functions with evaluators, or
    /// transcendental constants. Constant shifts i*pi*k/2 in exponential arguments
    /// (pi*k/2 for circular functions) are handled exactly, for integer k; other
    /// pi dependence is rejected.
    /// Pi outside these shifts and nested exponentials, including exp(e), are
    /// rejected rather than assuming unproved independence of their values.
    ///
    /// The field must be prime. Complex coefficients require p = 1 mod 4;
    /// exponentials additionally require q = (p - 1)/4 to be an odd prime.
    /// Rational poles modulo p or q return errors. Keep the sample map tied to
    /// one field. It caches variables, opaque calls, i, e, and exp(i).
    ///
    /// A nonzero sample certifies that an accepted expression is not identically
    /// zero, subject to the opaque-call assumption below. A zero sample is
    /// inconclusive; reduction modulo p and q can introduce additional zeros.
    /// This is a statement about identities, not pointwise nonvanishing.
    /// There is no claimed false-zero probability bound for this sampler.
    ///
    /// Certification: after clearing denominators and extracting the exact
    /// phases, an identity is a sum of rational coefficients times exp(R).
    /// At generic rational values of the formal variables, distinct rational
    /// functions R have distinct Gaussian rational values. Lindemann-Weierstrass
    /// then forces every coefficient, grouped by equal R, to vanish. Reduction
    /// of the coefficients modulo p and the additive exponent map modulo q
    /// preserve these cancellations, provided all encountered denominators
    /// are invertible. The two reductions need not use the same sample point.
    ///
    /// OPAQUE-CALL CAVEAT: non-builtin calls without evaluation info are independent
    /// formal variables, keyed by their complete serialized expressions. No
    /// congruence checking is performed on arguments: f((x+1)^2) and f(x^2+2*x+1)
    /// can receive different samples. Nonzero results involving such related calls
    /// or additional user function identities are not certificates for their
    /// analytic interpretation.
    fn sample_in_finite_field(
        &self,
        samples: &mut HashMap<AtomOrView<'_>, FiniteFieldElement<u64>>,
        field: &Zp64,
    ) -> Result<FiniteFieldElement<u64>, PolyError> {
        if SamplingPrime::for_prime(field.get_prime()).is_none() && !field.is_prime_field(0) {
            return Err(PolyError::NonPrimeField {
                modulus: field.get_prime(),
            });
        }
        self.sample_in_finite_field_impl(samples, field)
    }

    fn sample_in_finite_field_impl(
        &self,
        samples: &mut HashMap<AtomOrView<'_>, FiniteFieldElement<u64>>,
        field: &Zp64,
    ) -> Result<FiniteFieldElement<u64>, PolyError> {
        let mut rng = rand::rng();
        match self {
            AtomView::Num(num_view) => match num_view.get_coeff_view() {
                CoefficientView::Natural(n, d, ni, di) => {
                    let num = field.element_from_integer(n.into());
                    let den = field.element_from_integer(d.into());

                    if field.is_zero(&den) {
                        return Err(PolyError::DivisionByZero);
                    }

                    let mut res = field.div(&num, &den);

                    if ni != 0 {
                        let numi = field.element_from_integer(ni.into());
                        let deni = field.element_from_integer(di.into());

                        if field.is_zero(&deni) {
                            return Err(PolyError::DivisionByZero);
                        }

                        let im = field.div(&numi, &deni);

                        let i = sample_imaginary_unit(samples, field)?;
                        res = field.add(&res, &field.mul(&im, &i));
                    }

                    Ok(res)
                }
                CoefficientView::Large(r, i) => {
                    let re = r.to_rat();
                    let im = i.to_rat();

                    let num = field.element_from_integer(re.numerator());
                    let den = field.element_from_integer(re.denominator());

                    if field.is_zero(&den) {
                        return Err(PolyError::DivisionByZero);
                    }

                    let mut res = field.div(&num, &den);

                    if !im.is_zero() {
                        let numi = field.element_from_integer(im.numerator());
                        let deni = field.element_from_integer(im.denominator());

                        if field.is_zero(&deni) {
                            return Err(PolyError::DivisionByZero);
                        }

                        let im = field.div(&numi, &deni);

                        let i = sample_imaginary_unit(samples, field)?;
                        res = field.add(&res, &field.mul(&im, &i));
                    }

                    Ok(res)
                }
                _ => Err(PolyError::UnsupportedCoefficient),
            },
            AtomView::Var(var) => {
                if var.get_symbol() == Symbol::E {
                    let exponent_field = exponential_sample_field(field)?;
                    return sample_exponential_generator(
                        Symbol::E.into(),
                        samples,
                        field,
                        &exponent_field,
                    );
                }
                if !var.get_symbol().is_formal_sample_variable() {
                    return Err(PolyError::UnsupportedConstant(var.get_symbol()));
                }
                if let Some(val) = samples.get::<[u8]>(self.get_data()) {
                    Ok(*val)
                } else {
                    let new_sample = field.sample_uniform_nonzero(&mut rng);
                    samples.insert(var.get_symbol().into(), new_sample);
                    Ok(new_sample)
                }
            }
            AtomView::Pow(p) => {
                let (base, exponent) = p.get_base_exp();
                if let AtomView::Var(v) = base
                    && v.get_symbol() == Symbol::E
                {
                    return exponent.sample_rational_exponential(false, samples, field);
                }

                let AtomView::Num(n) = exponent else {
                    return Err(PolyError::NonIntegerExponent);
                };
                let exponent = match n.get_coeff_view() {
                    CoefficientView::Natural(num, 1, 0, _) => Integer::from(num),
                    CoefficientView::Large(re, im) if im.is_zero() => {
                        let re = re.to_rat();
                        if !re.is_integer() {
                            return Err(PolyError::NonIntegerExponent);
                        }
                        re.numerator()
                    }
                    _ => return Err(PolyError::NonIntegerExponent),
                };

                if exponent.is_zero() {
                    return Ok(field.one());
                }

                let base_val = base.sample_in_finite_field_impl(samples, field)?;
                if field.is_zero(&base_val) {
                    return if exponent.is_negative() {
                        Err(PolyError::DivisionByZero)
                    } else {
                        Ok(field.zero())
                    };
                }

                // Zp64::pow reduces machine-word exponents itself. Avoid
                // constructing the potentially large integer p - 1 for them.
                // Larger exponents reduce in the nonzero group of order p - 1.
                let reduced = if let Some(power) = exponent.to_i64() {
                    power.unsigned_abs()
                } else {
                    u64::try_from(exponent.abs() % Integer::from(field.get_prime() - 1))
                        .expect("reduced exponent fits in u64")
                };
                let result = field.pow(&base_val, reduced);
                if exponent.is_negative() {
                    Ok(field.inv(&result))
                } else {
                    Ok(result)
                }
            }
            AtomView::Fun(fun_view) => {
                let symbol = fun_view.get_symbol();

                if symbol.is_opaque_sample_function() {
                    if let Some(sample) = samples.get::<[u8]>(self.get_data()) {
                        return Ok(*sample);
                    }
                    let sample = field.sample_uniform_nonzero(&mut rng);
                    samples.insert(self.to_owned().into(), sample);
                    return Ok(sample);
                }

                if let Some((function, hyperbolic)) = SampledGeometricFunction::from_symbol(symbol)
                {
                    if fun_view.get_nargs() != 1 {
                        return Err(PolyError::InvalidFunctionArity {
                            function: symbol,
                            expected: 1,
                            actual: fun_view.get_nargs(),
                        });
                    }
                    let argument = fun_view.iter().next().unwrap();
                    let exponential =
                        argument.sample_rational_exponential(!hyperbolic, samples, field)?;
                    let inverse = field.inv(&exponential);
                    let sum = field.add(&exponential, &inverse);
                    let difference = field.sub(&exponential, &inverse);
                    let two = field.add(&field.one(), &field.one());
                    let unit = if hyperbolic
                        || matches!(
                            function,
                            SampledGeometricFunction::Cos | SampledGeometricFunction::Sec
                        ) {
                        field.one()
                    } else {
                        sample_imaginary_unit(samples, field)?
                    };
                    // Use exp(iP) for circular functions and exp(P) for
                    // hyperbolic functions. All six forms share these samples.
                    let (numerator, denominator) = match function {
                        SampledGeometricFunction::Sin => (difference, field.mul(&two, &unit)),
                        SampledGeometricFunction::Cos => (sum, two),
                        SampledGeometricFunction::Tan => (difference, field.mul(&unit, &sum)),
                        SampledGeometricFunction::Cot => (field.mul(&unit, &sum), difference),
                        SampledGeometricFunction::Sec => (two, sum),
                        SampledGeometricFunction::Csc => (field.mul(&two, &unit), difference),
                    };
                    if field.is_zero(&denominator) {
                        return Err(PolyError::DivisionByZero);
                    }
                    return Ok(field.div(&numerator, &denominator));
                }

                Err(PolyError::UnsupportedFunction(symbol))
            }
            AtomView::Mul(mul_view) => {
                let mut res = field.one();
                for m in mul_view {
                    let m_val = m.sample_in_finite_field_impl(samples, field)?;
                    field.mul_assign(&mut res, &m_val);
                }
                Ok(res)
            }
            AtomView::Add(add_view) => {
                let mut res = field.zero();
                for a in add_view {
                    let a_val = a.sample_in_finite_field_impl(samples, field)?;
                    field.add_assign(&mut res, &a_val);
                }
                Ok(res)
            }
        }
    }

    /// The numerical fallback follows the original checker's error-estimate model.
    /// These are heuristic witnesses, not directed-rounding certificates. In
    /// particular, numerical zeros never establish an identity, at any tolerance.
    fn numerical_zero_test(&self, iterations: usize, max_precision: u32) -> ConditionResult {
        if iterations == 0 {
            return ConditionResult::Inconclusive;
        }
        let mut points: HashMap<_, Complex<f64>> = self
            .get_all_indeterminates(true)
            .into_iter()
            .filter(|a| match a {
                AtomView::Var(v) => v.get_symbol().is_formal_sample_variable(),
                AtomView::Fun(f) => f.get_symbol().is_opaque_sample_function(),
                _ => false,
            })
            .map(|a| (a, Complex::new(0., 0.)))
            .collect();
        let mut rng = MonteCarloRng::new(0, 0);
        let attempts = if points.is_empty() { 1 } else { iterations };
        for _ in 0..attempts {
            for (atom, value) in &mut points {
                let s = atom.get_symbol().unwrap();
                // Use full-mantissa samples rather than n/8, which made such
                // expressions as sin(8*pi*x) vanish at every sampled real x.
                let unit = (rng.next_u64() >> 11) as f64 * (1. / (1u64 << 53) as f64);
                value.re = if s.is_integer() {
                    let n = (rng.next_u64() % 63) as i64 - 31;
                    if s.is_positive() {
                        (n.unsigned_abs() + 1) as f64
                    } else {
                        n as f64
                    }
                } else if s.is_positive() {
                    0.125 + 4. * unit
                } else {
                    8. * unit - 4.
                };
                value.im = if s.is_real() || s.is_integer() || s.is_positive() {
                    0.
                } else {
                    (rng.next_u64() >> 11) as f64 * (8. / (1u64 << 53) as f64) - 4.
                };
            }
            if self.numerical_f64_nonzero(&points) {
                return ConditionResult::False;
            }
            if self.numerical_precise_nonzero(&points, max_precision) {
                return ConditionResult::False;
            }
        }
        ConditionResult::Inconclusive
    }

    fn numerical_f64_nonzero(&self, points: &HashMap<AtomView<'_>, Complex<f64>>) -> bool {
        let values: HashMap<_, Complex<ErrorPropagatingFloat<f64>>> = points
            .iter()
            .map(|(a, z)| (*a, Complex::new(z.re.into(), z.im.into())))
            .collect();
        let Ok(value) = self.evaluate(&values, 53) else {
            return false;
        };
        let components = [&value.re, &value.im];
        if !components
            .iter()
            .all(|v| v.get_num().is_finite() && v.get_absolute_error().is_finite())
        {
            return false;
        }
        components
            .iter()
            .any(|v| v.get_num().abs() > 8. * v.get_absolute_error())
    }

    fn numerical_precise_nonzero(
        &self,
        points: &HashMap<AtomView<'_>, Complex<f64>>,
        max_precision: u32,
    ) -> bool {
        let mut previous: Option<Complex<Float>> = None;
        let mut precision = 128;
        while precision <= max_precision {
            let values: HashMap<_, Complex<Float>> = points
                .iter()
                .map(|(a, z)| {
                    (
                        *a,
                        Complex::new(
                            Float::with_val(precision, z.re),
                            Float::with_val(precision, z.im),
                        ),
                    )
                })
                .collect();
            let Ok(value) = self.evaluate(&values, precision) else {
                return false;
            };
            if !value.re.is_finite() || !value.im.is_finite() {
                return false;
            }
            if let Some(old) = previous {
                for (a, b) in [(&old.re, &value.re), (&old.im, &value.im)] {
                    // Float tracks precision loss. Also require agreement across
                    // two working precisions, so a shrinking rounding residual is
                    // not mistaken for a stable nonzero value. Never downcast these
                    // values or errors to f64: this path handles its range failures.
                    if !a.is_zero()
                        && !b.is_zero()
                        && a.get_precision() >= 32
                        && b.get_precision() >= 32
                        && (a.clone() / b - b.one()).norm() < Float::with_val(64, 1e-8)
                    {
                        return true;
                    }
                }
            }
            previous = Some(value);
            if precision == max_precision {
                break;
            }
            precision = precision.saturating_mul(2).min(max_precision);
        }
        false
    }
}

impl Symbol {
    fn is_opaque_sample_function(self) -> bool {
        // A map entry for the whole call would override its evaluator. Even if
        // its numeric domain is unsupported, leave it to evaluation to fail
        // conservatively instead of inventing an independent random value.
        !self.is_builtin() && self.get_evaluation_info().is_none()
    }

    fn is_formal_sample_variable(self) -> bool {
        !self.is_fixed_builtin()
            && self
                .get_evaluation_info()
                .is_none_or(|info| !info.has_constant_evaluator())
    }
}

fn sample_imaginary_unit(
    samples: &mut HashMap<AtomOrView<'_>, FiniteFieldElement<u64>>,
    field: &Zp64,
) -> Result<FiniteFieldElement<u64>, PolyError> {
    if field.get_prime() % 4 != 1 {
        return Err(PolyError::MissingImaginaryUnit {
            prime: field.get_prime(),
        });
    }
    let key = Atom::i();
    let minus_one = field.neg(&field.one());
    if let Some(sample) = samples.get::<[u8]>(key.as_view().get_data()) {
        return if field.mul(sample, sample) == minus_one {
            Ok(*sample)
        } else {
            Err(PolyError::InvalidImaginaryUnit)
        };
    }
    let sample = if let Some(known) = SamplingPrime::for_prime(field.get_prime()) {
        // Register lazily, so real expressions do not allocate an entry for i.
        field.to_element(known.imaginary_unit)
    } else {
        loop {
            let value = field.sample_uniform_nonzero(&mut rand::rng());
            let root = field.pow(&value, (field.get_prime() - 1) / 4);
            if field.mul(&root, &root) == minus_one {
                break root;
            }
        }
    };
    samples.insert(key.into(), sample);
    Ok(sample)
}

#[derive(Clone, Copy)]
enum SampledGeometricFunction {
    Sin,
    Cos,
    Tan,
    Cot,
    Sec,
    Csc,
}

impl SampledGeometricFunction {
    /// Identify the registered symbol, including functions defined outside the
    /// fixed builtin table. The boolean selects the hyperbolic counterpart.
    fn from_symbol(symbol: Symbol) -> Option<(Self, bool)> {
        if symbol == Symbol::SIN {
            return Some((Self::Sin, false));
        }
        if symbol == Symbol::COS {
            return Some((Self::Cos, false));
        }
        [
            (transcendental::tan(), Self::Tan, false),
            (transcendental::cot(), Self::Cot, false),
            (transcendental::sec(), Self::Sec, false),
            (transcendental::csc(), Self::Csc, false),
            (transcendental::sinh(), Self::Sin, true),
            (transcendental::cosh(), Self::Cos, true),
            (transcendental::tanh(), Self::Tan, true),
            (transcendental::coth(), Self::Cot, true),
            (transcendental::sech(), Self::Sec, true),
            (transcendental::csch(), Self::Csc, true),
        ]
        .into_iter()
        .find_map(|(candidate, function, hyperbolic)| {
            (symbol == candidate).then_some((function, hyperbolic))
        })
    }
}

/// The order-q subgroup makes division of rational exponents unambiguous.
fn exponential_sample_field(field: &Zp64) -> Result<Zp64, PolyError> {
    let p = field.get_prime();
    if let Some(known) = SamplingPrime::for_prime(p) {
        return Ok(Zp64::new(known.subgroup_order));
    }
    if p % 4 != 1 {
        return Err(PolyError::MissingImaginaryUnit { prime: p });
    }
    let order = (p - 1) / 4;
    // Reject even or trivial orders before constructing Zp64, which only
    // supports odd moduli. In particular, p = 17 gives the composite order 4.
    if order < 3 || order % 2 == 0 {
        return Err(PolyError::SubgroupNotOddPrime { order });
    }
    let q = Zp64::new(order);
    if !q.is_prime_field(0) {
        return Err(PolyError::SubgroupNotOddPrime { order });
    }
    Ok(q)
}

fn sample_exponential_generator(
    key: AtomOrView<'_>,
    samples: &mut HashMap<AtomOrView<'_>, FiniteFieldElement<u64>>,
    field: &Zp64,
    exponent_field: &Zp64,
) -> Result<FiniteFieldElement<u64>, PolyError> {
    if let Some(sample) = samples.get::<[u8]>(key.as_view().get_data()) {
        if field.pow(sample, exponent_field.get_prime()) != field.one() {
            return Err(PolyError::InvalidExponentialSample {
                subgroup_order: exponent_field.get_prime(),
            });
        }
        return Ok(*sample);
    }
    let sample = field.pow(&field.sample_uniform_nonzero(&mut rand::rng()), 4);
    samples.insert(key.into_owned().into(), sample);
    Ok(sample)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum PolyError {
    /// The supplied modulus is not prime.
    NonPrimeField { modulus: u64 },
    /// This constant cannot be treated as an independent formal variable.
    UnsupportedConstant(Symbol),
    /// Only constant shifts i*pi*k/2 of an exponential argument are supported.
    UnsupportedPeriod,
    /// A cached imaginary-unit sample does not square to -1.
    InvalidImaginaryUnit,
    /// A rational denominator or a base raised to a negative power is zero modulo p.
    DivisionByZero,
    /// Only exact real or complex rational coefficients can be sampled.
    UnsupportedCoefficient,
    /// A power with a base other than e requires an exact integer exponent.
    NonIntegerExponent,
    /// An exponential or geometric argument is not rational over exact complex rationals
    /// in formal variables and opaque calls (for example, it contains nested builtins).
    NonRationalArgument,
    /// This function has no finite-field sampling implementation.
    UnsupportedFunction(Symbol),
    /// A supported function has the wrong number of arguments.
    InvalidFunctionArity {
        function: Symbol,
        expected: usize,
        actual: usize,
    },
    /// Complex coefficients and exponential sampling require p = 1 mod 4.
    MissingImaginaryUnit { prime: u64 },
    /// Exponential sampling requires q = (p - 1)/4 to be an odd prime.
    SubgroupNotOddPrime { order: u64 },
    /// An exponent denominator is not invertible in F_q[i].
    NonInvertibleExponentDenominator { subgroup_order: u64 },
    /// A cached exponential generator is zero or lies outside the order-q subgroup.
    InvalidExponentialSample { subgroup_order: u64 },
}

#[cfg(test)]
mod tests {
    // Keep exercising the compatibility wrapper alongside the replacement API.
    #![allow(deprecated)]

    use super::*;
    use crate::{atom::EvaluationInfo, coefficient::Coefficient, parse};

    #[test]
    fn custom_evaluators_are_not_structural_or_modular_indeterminates() {
        let f = crate::symbol!(
            "zero_check_custom_zero",
            eval = EvaluationInfo::new().register(|args: &[Complex<Float>]| args[0].zero())
        );
        let call = f.call(parse!("x"));
        let field = Zp64::new(SAMPLE_PRIMES[0].prime);
        assert_eq!(
            call.as_view()
                .sample_in_finite_field(&mut HashMap::default(), &field),
            Err(PolyError::UnsupportedFunction(f))
        );
        for atom in [
            call.clone(),
            &call * parse!("y"),
            call.sin(),
            call.exp() - 1,
        ] {
            assert_eq!(atom.check_zero(), ConditionResult::Inconclusive, "{atom}");
            assert_eq!(
                atom.check_zero_with(&ZeroTestSettings::new().numerical_samples(0)),
                ConditionResult::Inconclusive,
                "{atom}"
            );
        }
    }

    #[test]
    fn custom_evaluators_receive_sampled_arguments_including_nested_calls() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static CALLS: AtomicUsize = AtomicUsize::new(0);
        let f = crate::symbol!(
            "zero_check_custom_identity",
            eval = EvaluationInfo::new().register(|args: &[Complex<Float>]| {
                CALLS.fetch_add(1, Ordering::Relaxed);
                args[0].clone()
            })
        );
        let x = parse!("x");
        let identity = f.call(f.call(&x)) - &x;
        assert_eq!(identity.check_zero(), ConditionResult::Inconclusive);
        assert!(CALLS.load(Ordering::Relaxed) > 0);
        CALLS.store(0, Ordering::Relaxed);
        assert_eq!((identity + 1).check_zero(), ConditionResult::False);
        assert!(CALLS.load(Ordering::Relaxed) > 0);
        assert_eq!(f.call(&x).check_equal(&x), ConditionResult::Inconclusive);
    }

    #[test]
    fn custom_evaluators_work_in_the_machine_precision_test() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static CALLS: AtomicUsize = AtomicUsize::new(0);
        let f = crate::symbol!(
            "zero_check_custom_machine_identity",
            eval =
                EvaluationInfo::new().register(|args: &[Complex<ErrorPropagatingFloat<f64>>]| {
                    CALLS.fetch_add(1, Ordering::Relaxed);
                    args[0]
                })
        );
        let x = parse!("x");
        let identity = f.call(&x) - &x;
        let settings = ZeroTestSettings::new().max_precision(53);
        assert_eq!(
            identity.check_zero_with(&settings),
            ConditionResult::Inconclusive
        );
        assert!(CALLS.load(Ordering::Relaxed) > 0);
        CALLS.store(0, Ordering::Relaxed);
        assert_eq!(
            (identity + 1).check_zero_with(&settings),
            ConditionResult::False
        );
        assert!(CALLS.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn custom_evaluators_preserve_symbolic_tags_and_constant_callbacks() {
        let tag = parse!("zero_check_symbolic_tag");
        let expected_tag = tag.clone();
        let f = crate::symbol!(
            "zero_check_custom_tagged_identity",
            eval = EvaluationInfo::new()
                .with_tags(1)
                .register_tagged(move |tags| {
                    assert_eq!(tags, &[expected_tag.as_view()]);
                    Box::new(|args: &[Complex<Float>]| args[0].clone())
                })
        );
        let x = parse!("x");
        let identity = f.call((&tag, &x)) - &x;
        assert_eq!(identity.check_zero(), ConditionResult::Inconclusive);
        assert_eq!((identity + 1).check_zero(), ConditionResult::False);

        let constant = crate::symbol!(
            "zero_check_custom_tagged_constant",
            eval = EvaluationInfo::constant(|tags, precision| {
                assert_eq!(tags, &[parse!("zero_check_symbolic_tag").as_view()]);
                Ok(Complex::new(
                    Float::with_val(precision, 0),
                    Float::with_val(precision, 0),
                ))
            })
            .with_tags(1)
        );
        assert_eq!(
            constant.call(&tag).check_zero(),
            ConditionResult::Inconclusive
        );
        assert_eq!(
            (constant.call(&tag) + 1).check_zero(),
            ConditionResult::False
        );
    }

    #[test]
    fn custom_evaluators_with_missing_domains_or_errors_stay_inconclusive() {
        let real_only = crate::symbol!(
            "zero_check_custom_real_only",
            eval = EvaluationInfo::new().register(|_args: &[f64]| 0.)
        );
        let failed = crate::symbol!(
            "zero_check_custom_failed",
            eval = EvaluationInfo::constant(|_, _| Err("Unavailable".into()))
        );
        let nonfinite = crate::symbol!(
            "zero_check_custom_nonfinite",
            eval = EvaluationInfo::new().register(|_args: &[Complex<Float>]| {
                Complex::new(Float::with_val(128, f64::NAN), Float::with_val(128, 0))
            })
        );
        for atom in [
            real_only.call(parse!("x")),
            FunctionBuilder::new(failed).finish(),
            nonfinite.call(parse!("x")),
        ] {
            assert_eq!(atom.check_zero(), ConditionResult::Inconclusive, "{atom}");
        }
    }

    #[test]
    fn check_equal_distinguishes_identities_nonidentities_and_undecided_cases() {
        fn query<A: AtomCore, B: AtomCore>(lhs: &A, rhs: &B) -> ConditionResult {
            lhs.check_equal(rhs.as_atom_view())
        }

        for (left, right, expected) in [
            ("x", "x", ConditionResult::True),
            ("(x+1)^2", "x^2+2*x+1", ConditionResult::True),
            ("1/x-1/(x+1)", "1/(x*(x+1))", ConditionResult::True),
            ("sqrt(2)*sqrt(3)", "sqrt(6)", ConditionResult::True),
            ("sin(x)^2+cos(x)^2", "1", ConditionResult::True),
            ("x", "y", ConditionResult::False),
            ("(x+1)^2", "x^2+2*x+2", ConditionResult::False),
            ("x+1/10^400", "x", ConditionResult::False),
            ("log(2)+log(3)", "log(6)", ConditionResult::Inconclusive),
        ] {
            let (lhs, rhs) = (parse!(left), parse!(right));
            for result in [
                lhs.check_equal(&rhs),
                rhs.check_equal(&lhs),
                lhs.as_view().check_equal(rhs.as_view()),
                query(&lhs, &rhs),
                query(&AtomOrView::from(lhs.as_view()), &rhs.as_view()),
            ] {
                assert_eq!(result, expected, "{left} == {right}");
            }
        }
        assert_eq!(
            parse!("sin(x)^2+cos(x)^2").check_equal(1),
            ConditionResult::True
        );
    }

    #[test]
    fn check_equal_with_respects_settings_and_exact_float_values() {
        fn query<A: AtomCore>(
            lhs: &A,
            rhs: AtomView<'_>,
            settings: &ZeroTestSettings,
        ) -> ConditionResult {
            lhs.check_equal_with(rhs, settings)
        }

        let (lhs, rhs) = (parse!("(x+1)^2"), parse!("x^2+2*x+1"));
        let quick = ZeroTestSettings::new().exact_fallback(false);
        assert_eq!(
            lhs.check_equal_with(&rhs, &quick),
            ConditionResult::Inconclusive
        );
        assert_eq!(
            query(&lhs.as_view(), rhs.as_view(), &quick),
            ConditionResult::Inconclusive
        );
        assert_eq!(
            lhs.check_equal_with(&rhs, &ZeroTestSettings::default()),
            ConditionResult::True
        );

        // A difference far below the float's input precision must not be
        // accepted as equality, with either interpretation of floats.
        let floating = Atom::num(Float::with_val(53, 1));
        let rational = parse!("1+1/10^100");
        assert!(!floating.check_equal(&rational).is_true());
        assert!(!rational.check_equal(&floating).is_true());
        assert_eq!(floating.check_equal(1), ConditionResult::True);
        let exact = ZeroTestSettings::new().treat_floats_as_exact(true);
        assert_eq!(
            floating.check_equal_with(&rational, &exact),
            ConditionResult::False
        );
        assert_eq!(
            rational.check_equal_with(&floating, &exact),
            ConditionResult::False
        );
        assert_eq!(floating.check_equal_with(1, &exact), ConditionResult::True);

        let f = crate::symbol!("check_equal_exact_float_function");
        let half = Atom::num(Float::with_val(100, 0.5));
        assert_eq!(
            f.call(&half)
                .check_equal_with(f.call(Atom::num((1, 2))), &exact),
            ConditionResult::True
        );
    }

    #[test]
    fn check_equal_does_not_certify_nonfinite_values() {
        for value in [
            Atom::num(Float::with_val(100, f64::NAN)),
            Atom::num(Float::with_val(100, f64::INFINITY)),
            Atom::num(Coefficient::complex_infinity()),
        ] {
            assert_eq!(value.check_equal(&value), ConditionResult::Inconclusive);
            assert_eq!(
                value
                    .check_equal_with(&value, &ZeroTestSettings::new().treat_floats_as_exact(true)),
                ConditionResult::Inconclusive,
            );
        }
    }

    #[test]
    fn legacy_boolean_zero_queries_remain_compatible() {
        // Existing function pointers and Boolean conditions keep their types.
        let query: fn(&Atom) -> bool = Atom::is_zero;
        for (atom, expected) in [
            (parse!("0"), true),
            (parse!("x"), false),
            (parse!("(x+1)^2-x^2-2*x-1"), false),
            (parse!("log(2)+log(3)-log(6)"), false),
        ] {
            assert_eq!(query(&atom), expected);
            assert_eq!(atom.as_view().is_zero(), expected);
            assert_eq!(AtomCore::is_zero(&atom), expected);
            if atom.is_zero() {
                assert!(expected);
            } else {
                assert!(!expected);
            }
        }

        // Generic callers with both traits select the intended zero policy.
        fn generic_queries<A: AtomCore + crate::domains::SelfRing>(atom: &A) -> (bool, bool) {
            (
                AtomCore::is_zero(atom),
                crate::domains::SelfRing::is_zero(atom),
            )
        }
        let identity = parse!("(x+1)^2-x^2-2*x-1");
        assert!(!identity.is_zero());
        assert_eq!(generic_queries(&identity), (false, true));
        assert_eq!(identity.check_zero(), ConditionResult::True);
    }

    #[test]
    fn zero_settings_are_available_on_atom_core() {
        fn query<A: AtomCore>(atom: &A) -> ConditionResult {
            atom.check_zero_with(&ZeroTestSettings::new())
        }
        for text in [
            "0",
            "x",
            "sin(x)^2+cos(x)^2-1",
            "sqrt(2)+sqrt(3)",
            "log(2)+log(3)-log(6)",
        ] {
            let atom = parse!(text);
            let expected = atom.check_zero();
            assert_eq!(atom.zero_test(5, 0.), expected, "{text}");
            assert_eq!(
                atom.check_zero_with(&ZeroTestSettings::default()),
                expected,
                "{text}"
            );
            for result in [
                query(&atom),
                query(&atom.as_view()),
                query(&AtomOrView::from(atom.as_view())),
            ] {
                assert_eq!(result, expected, "{text}");
            }
        }
    }

    #[test]
    fn zero_test_stages_can_be_controlled_independently() {
        let structural = ZeroTestSettings::new()
            .finite_field_samples(0)
            .numerical_samples(0)
            .exact_fallback(false);
        assert_eq!(
            parse!("x").check_zero_with(&structural),
            ConditionResult::False
        );
        let polynomial = parse!("(x+1)^2-x^2-2*x-1");
        assert_eq!(
            polynomial.check_zero_with(&structural),
            ConditionResult::Inconclusive
        );
        assert_eq!(polynomial.zero_test(0, 0.), ConditionResult::Inconclusive);
        assert_eq!(
            polynomial.check_zero_with(&structural.exact_fallback(true)),
            ConditionResult::True
        );

        let constant = parse!("sqrt(2)+sqrt(3)");
        assert_eq!(
            constant.check_zero_with(&structural),
            ConditionResult::Inconclusive
        );
        assert_eq!(
            constant.check_zero_with(&structural.numerical_samples(1)),
            ConditionResult::False
        );
        assert_eq!(
            parse!("cosh(x)^2-sinh(x)^2").check_zero_with(&structural.finite_field_samples(1)),
            ConditionResult::False,
        );
        assert_eq!(
            parse!("sqrt(2)*sqrt(3)-sqrt(6)").check_zero_with(&structural.exact_fallback(true)),
            ConditionResult::True,
        );
    }

    #[test]
    fn numerical_precision_limits_include_the_final_partial_step() {
        let atom = parse!("(sqrt(2)+sqrt(3))/10^400");
        let settings = ZeroTestSettings::new()
            .finite_field_samples(0)
            .numerical_samples(1)
            .exact_fallback(false);
        for limit in [0, 127, 128] {
            assert_eq!(
                atom.check_zero_with(&settings.max_precision(limit)),
                ConditionResult::Inconclusive
            );
        }
        // The 200-bit evaluation confirms the 128-bit result. Rounding the
        // requested limit down to a power of two would miss this witness.
        assert_eq!(
            atom.check_zero_with(&settings.max_precision(200)),
            ConditionResult::False
        );
    }

    #[test]
    fn high_precision_pi_residual_requires_exact_float_semantics() {
        let pi = Atom::var(Symbol::PI);
        let atom = &pi - pi.to_float(10000);
        assert_eq!(atom.check_zero(), ConditionResult::Inconclusive);
        assert_eq!(atom.zero_test(20, 0.), ConditionResult::Inconclusive);

        let high = ZeroTestSettings::new().max_precision(80_000);
        assert_eq!(atom.check_zero_with(&high), ConditionResult::Inconclusive);
        let exact = ZeroTestSettings::new().treat_floats_as_exact(true);
        assert_eq!(atom.check_zero_with(&exact), ConditionResult::Inconclusive);
        assert_eq!(
            atom.check_zero_with(&exact.max_precision(80_000)),
            ConditionResult::False
        );
        // Pi itself is unsupported by the finite-field stage, even after the
        // floating coefficient has been converted to an exact rational.
        assert_eq!(
            atom.check_zero_with(&exact.numerical_samples(0).exact_fallback(false)),
            ConditionResult::Inconclusive,
        );
    }

    #[test]
    fn exact_float_coefficients_participate_in_modular_sampling() {
        let modular = ZeroTestSettings::new()
            .numerical_samples(0)
            .exact_fallback(false);
        for coefficient in [
            Complex::new(Float::with_val(100, 0.5), Float::new(100)),
            Complex::new(Float::with_val(100, 0.5), Float::with_val(100, 0.25)),
        ] {
            let atom = parse!("(x+1)^2-x^2-2*x") + Atom::num(coefficient);
            assert_eq!(
                atom.check_zero_with(&modular),
                ConditionResult::Inconclusive
            );
            assert_eq!(
                atom.check_zero_with(&modular.treat_floats_as_exact(true)),
                ConditionResult::False
            );
        }
        let identity = parse!("(x+1)^2-x^2-2*x") - Atom::num(Float::with_val(100, 1));
        assert_eq!(
            identity.check_zero_with(&modular.treat_floats_as_exact(true)),
            ConditionResult::Inconclusive,
        );
        assert_eq!(
            identity.check_zero_with(&modular.treat_floats_as_exact(true).exact_fallback(true)),
            ConditionResult::True,
        );
    }

    #[test]
    fn exact_float_policy_visits_arguments_and_exponents_and_rejects_nonfinite_values() {
        let settings = ZeroTestSettings::new().treat_floats_as_exact(true);
        let half = Atom::num(Float::with_val(100, 0.5));
        let rational = Atom::num((1, 2));
        let f = crate::symbol!("zero_exact_float_function");
        for atom in [
            f.call(&half) - f.call(&rational),
            parse!("x").pow(&half) - parse!("x").pow(&rational),
        ] {
            assert_eq!(atom.check_zero_with(&settings), ConditionResult::True);
        }
        for atom in [
            Atom::num(Float::with_val(100, f64::NAN)),
            Atom::num(Float::with_val(100, f64::INFINITY)),
            Atom::num(Coefficient::complex_infinity()),
        ] {
            assert_eq!(
                atom.check_zero_with(&settings),
                ConditionResult::Inconclusive
            );
        }
    }

    #[test]
    fn literal_and_identity_zero_queries_have_distinct_contracts() {
        fn query<A: AtomCore>(atom: &A) -> (bool, ConditionResult) {
            (atom.is_zero(), atom.check_zero())
        }

        for (text, literal, expected) in [
            ("0", true, ConditionResult::True),
            ("x", false, ConditionResult::False),
            ("sqrt(2)+x", false, ConditionResult::False),
            ("sqrt(2)+sqrt(3)", false, ConditionResult::False),
            ("(x+1)^2-x^2-2*x", false, ConditionResult::False),
            ("sin(x)^2+cos(x)^2", false, ConditionResult::False),
            ("(x+1)^2-x^2-2*x-1", false, ConditionResult::True),
            ("1/x-1/(x+1)-1/(x*(x+1))", false, ConditionResult::True),
            ("sin(x)^2+cos(x)^2-1", false, ConditionResult::True),
            (
                "x+1/((y+1)^2-y^2-2*y-1)",
                false,
                ConditionResult::Inconclusive,
            ),
        ] {
            let atom = parse!(text);
            assert_eq!(atom.is_zero(), literal, "{text}");
            assert_eq!(atom.as_view().is_zero(), literal, "{text}");
            assert_eq!(atom.check_zero(), expected, "{text}");
            assert_eq!(query(&atom), (literal, expected), "{text}");
            assert_eq!(query(&atom.as_view()), (literal, expected), "{text}");
            assert_eq!(
                query(&AtomOrView::from(atom.as_view())),
                (literal, expected),
                "{text}"
            );
        }
    }

    #[test]
    fn default_zero_query_uses_numerical_fallbacks() {
        for text in [
            "sqrt(2)+sqrt(3)",
            "10^400*(sqrt(2)+sqrt(3))",
            "(sqrt(2)+sqrt(3))/10^400",
            "sqrt(1+1/10^100)-1",
        ] {
            let atom = parse!(text);
            assert_eq!(
                atom.zero_test(0, 0.),
                ConditionResult::Inconclusive,
                "{text}"
            );
            let field = Zp64::new(SAMPLE_PRIMES[0].prime);
            assert!(
                atom.as_view()
                    .sample_in_finite_field(&mut HashMap::default(), &field)
                    .is_err(),
                "{text}"
            );
            assert_eq!(atom.check_zero(), ConditionResult::False, "{text}");
        }
    }

    #[test]
    fn fixed_sampling_fields_support_exponentials() {
        assert_eq!(SAMPLE_PRIMES.len(), 5);
        for (index, sample) in SAMPLE_PRIMES.iter().enumerate() {
            let field = Zp64::new(sample.prime);
            assert!(field.is_prime_field(0));
            assert!(sample.prime > 1 << 63);
            assert!(
                SAMPLE_PRIMES[..index]
                    .iter()
                    .all(|s| s.prime != sample.prime)
            );
            assert_eq!(sample.prime, 4 * sample.subgroup_order + 1);
            assert_eq!(sample.subgroup_order % 2, 1);
            assert!(Zp64::new(sample.subgroup_order).is_prime_field(0));
            assert_eq!(
                exponential_sample_field(&field).unwrap().get_prime(),
                sample.subgroup_order
            );
            assert!(sample.imaginary_unit < sample.prime);
            assert_eq!(
                (sample.imaginary_unit as u128).pow(2) % sample.prime as u128,
                (sample.prime - 1) as u128
            );
            let i = field.to_element(sample.imaginary_unit);
            assert_eq!(field.mul(&i, &i), field.neg(&field.one()));
        }
    }

    #[test]
    fn fixed_imaginary_units_are_cached_and_respect_existing_samples() {
        for sample in &SAMPLE_PRIMES {
            let field = Zp64::new(sample.prime);
            let mut samples = HashMap::default();
            let i = field.to_element(sample.imaginary_unit);
            assert_eq!(sample_imaginary_unit(&mut samples, &field), Ok(i));
            assert_eq!(samples.len(), 1);
            assert_eq!(
                samples.get::<[u8]>(Atom::i().as_view().get_data()),
                Some(&i)
            );
            assert_eq!(sample_imaginary_unit(&mut samples, &field), Ok(i));
            samples.insert(Atom::i().into(), field.neg(&i));
            assert_eq!(
                sample_imaginary_unit(&mut samples, &field),
                Ok(field.neg(&i))
            );
            samples.insert(Atom::i().into(), field.one());
            assert_eq!(
                sample_imaginary_unit(&mut samples, &field),
                Err(PolyError::InvalidImaginaryUnit)
            );
        }
    }

    #[test]
    fn fixed_sampling_fields_preserve_complex_and_transcendental_identities() {
        let cases = [
            "(x+1i)^2-x^2-2i*x+1",
            "sin(x)^2+cos(x)^2-1",
            "cosh(x)^2-sinh(x)^2-1",
            "sin(x+pi/2)-cos(x)",
            "exp(x+1i*pi/2)-1i*exp(x)",
            "exp(x/(x+1))*exp(1/(x+1))-exp(1)",
        ]
        .map(|text| (text, parse!(text)));
        for sample in &SAMPLE_PRIMES {
            let field = Zp64::new(sample.prime);
            let mut samples = HashMap::default();
            samples.insert(parse!("x").into(), field.to_element(sample.prime - 3));
            for (text, atom) in &cases {
                assert_eq!(
                    atom.as_view().sample_in_finite_field(&mut samples, &field),
                    Ok(field.zero()),
                    "{text}, p={}",
                    sample.prime
                );
                assert_eq!(
                    (atom + Atom::one())
                        .as_view()
                        .sample_in_finite_field(&mut samples, &field),
                    Ok(field.one()),
                    "({text})+1, p={}",
                    sample.prime
                );
            }
        }
    }

    #[test]
    fn cheap_structural_proofs_need_no_samples() {
        for text in [
            "1",
            "-2/3",
            "x",
            "3*x^32*y^16*z^8*w^8",
            "x^-2*y^3",
            "sqrt(2)",
            "sqrt(2)+x",
            "sqrt(2)+3*x^7",
            "sqrt(2)+x^-3",
            "(y-1)*x+sqrt(2)",
            "sqrt(2)*x+sin(y)",
            "sqrt(2)*sin(y)+x",
            "(x+1)^1000000",
            "10^1000*(1+x)",
            "exp(sin(x))",
        ] {
            assert_eq!(
                parse!(text).zero_test(0, f64::NAN),
                ConditionResult::False,
                "{text}"
            );
        }
        assert_eq!(parse!("0").zero_test(0, f64::NAN), ConditionResult::True);
    }

    #[test]
    fn rational_polynomial_coefficients_are_structurally_inconclusive() {
        let coefficient = parse!("(x+1)/(x+2)").set_coefficient_ring(crate::symbol!("x"));
        assert!(matches!(
            coefficient.as_view(),
            AtomView::Num(n) if matches!(n.get_coeff_view(), CoefficientView::RationalPolynomial(_))
        ));
        for atom in [
            coefficient.clone(),
            &coefficient + parse!("sqrt(2)"),
            &coefficient * parse!("y"),
            Atom::var(Symbol::E).pow(&coefficient),
        ] {
            assert_eq!(
                atom.zero_test(0, f64::NAN),
                ConditionResult::Inconclusive,
                "{atom}"
            );
        }
    }

    #[test]
    fn uncertain_coefficients_and_shared_variables_do_not_prove_nonzero() {
        for text in [
            "x*((y+1)^2-y^2-2*y-1)",
            "x*(sin(y)^2+cos(y)^2-1)",
            "x*sin(y)+sqrt(2)",
            "sqrt(x)-x",
            "sqrt(2)+sqrt(3)",
            "x+1/((y+1)^2-y^2-2*y-1)",
            "(abs(x)-x)*(abs(x)+x)",
        ] {
            assert_eq!(
                parse!(text).zero_test(0, 0.),
                ConditionResult::Inconclusive,
                "{text}"
            );
        }
    }

    #[test]
    fn exact_cancellation_follows_inconclusive_samples() {
        for text in [
            "(x+1)^2-x^2-2*x-1",
            "1/(x+1)+x/(x+1)-1",
            "1/x-1/(x+1)-1/(x*(x+1))",
            "(x+1i*y)^2-x^2-2i*x*y+y^2",
            "x*((y+1)^2-y^2-2*y-1)",
        ] {
            let atom = parse!(text);
            assert!(!atom.is_zero(), "{text}");
            assert_eq!(
                atom.zero_test(0, 1.),
                ConditionResult::Inconclusive,
                "{text}"
            );
            assert_eq!(
                atom.as_view().numerical_zero_test(3, 1024),
                ConditionResult::Inconclusive,
                "{text}"
            );
            assert_eq!(atom.zero_test(3, 1.), ConditionResult::True, "{text}");
            assert_eq!(atom.check_zero(), ConditionResult::True, "{text}");
            assert_eq!(
                (atom + parse!("1/10^400")).check_zero(),
                ConditionResult::False,
                "{text}"
            );
        }
    }

    #[test]
    fn analytic_identities_cancel_in_the_final_stage() {
        for text in [
            "x*(sin(y)^2+cos(y)^2-1)",
            "sin(x+pi/2)-cos(x)",
            "exp(x/(x+1))*exp(1/(x+1))-exp(1)",
        ] {
            assert!(
                !parse!(text).is_zero(),
                "fixture normalized to zero: {text}"
            );
            assert_eq!(
                parse!(text).zero_test(3, 1.),
                ConditionResult::True,
                "{text}"
            );
            assert_eq!(parse!(text).check_zero(), ConditionResult::True, "{text}");
        }
    }

    #[test]
    fn exact_algebraic_and_analytic_zero_certificates() {
        for text in [
            "sqrt(2)*sqrt(3)-sqrt(6)",
            "sqrt(3+2*sqrt(2))-1-sqrt(2)",
            "root(z^3-z-1,0)^3-root(z^3-z-1,0)-1",
            "sqrt(1i)-(1+1i)/sqrt(2)",
            "sqrt(-1i)-(1-1i)/sqrt(2)",
            "(-1)^(1/3)-(1+1i*sqrt(3))/2",
            "(-1)^(-1/3)-(1-1i*sqrt(3))/2",
            "(-8)^(1/3)-(1+1i*sqrt(3))",
            "(-2+2i)^(1/3)-1-1i",
            "(-2-2i)^(1/3)-1+1i",
            "(-2+2i)^(2/3)-2i",
            "exp(sqrt(3+2*sqrt(2)))-exp(1+sqrt(2))",
            "sin(sqrt(3+2*sqrt(2)))-sin(1+sqrt(2))",
            "sin(x+y)-sin(x)*cos(y)-cos(x)*sin(y)",
            "cosh(x)^2-sinh(x)^2-1",
            "tan(x)-sin(x)/cos(x)",
            "tanh(x)-sinh(x)/cosh(x)",
            "sec(x)*cos(x)-1",
            "csc(x)*sin(x)-1",
            "cot(x)*sin(x)-cos(x)",
            "sech(x)*cosh(x)-1",
            "csch(x)*sinh(x)-1",
            "coth(x)*sinh(x)-cosh(x)",
            "cos(pi/3)-1/2",
            "sin(pi/6)-1/2",
            "exp(x+1i*pi/3)-exp(x)*(1+1i*sqrt(3))/2",
            "1+exp(2i*pi/7)+exp(4i*pi/7)+exp(6i*pi/7)+exp(8i*pi/7)+exp(10i*pi/7)+exp(12i*pi/7)",
        ] {
            let atom = parse!(text);
            assert_eq!(
                atom.as_view().exact_zero_test().unwrap(),
                ConditionResult::True,
                "{text}"
            );
            assert_eq!(atom.check_zero(), ConditionResult::True, "{text}");
        }
    }

    #[test]
    fn algebraic_nonzeros_beyond_the_numerical_budget_are_certified() {
        for text in [
            "sqrt(2)*sqrt(3)-sqrt(6)+1/10^1000",
            "(sqrt(2)*sqrt(3)-sqrt(6)+1/10^1000)*x+y-y",
            "(-8)^(1/3)+2",
            "sqrt(1i)+(1+1i)/sqrt(2)",
        ] {
            let atom = parse!(text);
            assert_eq!(
                atom.as_view().exact_zero_test().unwrap(),
                ConditionResult::False,
                "{text}"
            );
            assert_eq!(atom.check_zero(), ConditionResult::False, "{text}");
        }
    }

    #[test]
    fn unresolved_analytic_relations_do_not_become_formal_nonzero_proofs() {
        for text in [
            "log(2)+log(3)-log(6)",
            "abs(x)^2-x*conj(x)",
            "gamma(x+1)-x*gamma(x)",
            "sin(x)+cos(x)",
            "sqrt(x^2)-x",
            "root(z^3-x,z,0)^3-x",
        ] {
            let result = parse!(text)
                .as_view()
                .exact_zero_test()
                .unwrap_or(ConditionResult::Inconclusive);
            assert_eq!(result, ConditionResult::Inconclusive, "{text}");
        }
    }

    #[test]
    fn algebraic_zero_denominators_remain_inconclusive() {
        for text in [
            "1/(sqrt(2)*sqrt(3)-sqrt(6))",
            "1/((sqrt(2)*sqrt(3)-sqrt(6))*x)",
            "1/sqrt(sqrt(2)*sqrt(3)-sqrt(6))",
            "1/(sin(x)^2+cos(x)^2-1)",
        ] {
            let atom = parse!(text);
            assert_eq!(atom.check_zero(), ConditionResult::Inconclusive, "{text}");
        }
    }

    #[test]
    fn exact_argument_normalization_shares_the_algebraic_field() {
        // The early opaque-call probes still assume independent calls. Exercise
        // the exact stage directly to keep that documented limitation explicit.
        for text in [
            "f((x+1)^2)-f(x^2+2*x+1)",
            "f(sqrt(3+2*sqrt(2)))-f(1+sqrt(2))",
            "f(x*(sqrt(3+2*sqrt(2))-1-sqrt(2)))-f(0)",
        ] {
            assert_eq!(
                parse!(text).as_view().exact_zero_test().unwrap(),
                ConditionResult::True,
                "{text}"
            );
        }
    }

    #[test]
    fn numerical_samples_avoid_fixed_period_grids() {
        let real = crate::symbol!("zero_sample_real"; Real);
        let positive = crate::symbol!("zero_sample_positive"; Positive);
        for atom in [
            (Atom::var(real) * Atom::var(Symbol::PI) * 8).sin(),
            (Atom::var(positive) * Atom::var(Symbol::PI)).sin(),
        ] {
            assert_eq!(
                atom.as_view().numerical_zero_test(5, 1024),
                ConditionResult::False
            );
            assert_eq!(atom.check_zero(), ConditionResult::False);
        }
    }

    #[test]
    fn modular_nonzero_witnesses_precede_numerics() {
        for text in [
            "(x+1)^2-x^2-2*x",
            "sin(x)^2+cos(x)^2",
            "cosh(x)^2-sinh(x)^2",
        ] {
            let atom = parse!(text);
            assert_ne!(
                atom.as_view().structural_zero(0, &mut 512),
                Some(Structural::Nonzero)
            );
            assert_eq!(
                atom.zero_test(1, f64::NAN),
                ConditionResult::False,
                "{text}"
            );
            let field = Zp64::new(SAMPLE_PRIMES[0].prime);
            assert!(
                !field.is_zero(
                    &atom
                        .as_view()
                        .sample_in_finite_field(&mut HashMap::default(), &field)
                        .unwrap()
                )
            );
        }
    }

    #[test]
    fn f64_fallback_handles_unsupported_algebraic_coefficients() {
        let atom = parse!("sqrt(2)+sqrt(3)");
        let points = HashMap::default();
        assert!(atom.as_view().numerical_f64_nonzero(&points));
        assert_eq!(atom.zero_test(1, 1e-10), ConditionResult::False);
    }

    #[test]
    fn arbitrary_precision_recovers_range_and_cancellation_failures() {
        for text in [
            "10^400*(sqrt(2)+sqrt(3))",
            "(sqrt(2)+sqrt(3))/10^400",
            "sqrt(1+1/10^100)-1",
        ] {
            let atom = parse!(text);
            let points = HashMap::default();
            assert!(!atom.as_view().numerical_f64_nonzero(&points), "{text}");
            assert!(
                atom.as_view().numerical_precise_nonzero(&points, 1024),
                "{text}"
            );
            assert_eq!(atom.zero_test(1, 1e-10), ConditionResult::False, "{text}");
        }
    }

    #[test]
    fn numerical_zero_and_unsupported_evaluations_are_inconclusive() {
        for text in [
            "sin(x,y)",
            "x+1/((y+1)^2-y^2-2*y-1)",
            "x+1/((y+1i)^2-y^2-2i*y+1)",
            "1/((x+1)^2-x^2-2*x-1)-1/((x+1)^2-x^2-2*x-1)^2",
        ] {
            assert_eq!(
                parse!(text).zero_test(2, 1.),
                ConditionResult::Inconclusive,
                "{text}"
            );
            assert_eq!(
                parse!(text).check_zero(),
                ConditionResult::Inconclusive,
                "{text}"
            );
        }
        for atom in [
            Atom::num(Coefficient::Indeterminate),
            Atom::num(Coefficient::complex_infinity()),
        ] {
            assert_eq!(atom.zero_test(2, 1.), ConditionResult::Inconclusive);
            assert_eq!(atom.check_zero(), ConditionResult::Inconclusive);
        }
        // A small number is not zero just because it is below the tolerance.
        assert_eq!(
            parse!("(sqrt(2)+sqrt(3))/10^400").zero_test(1, 1.),
            ConditionResult::False
        );
    }

    #[test]
    fn exact_cancellation_rejects_zero_denominators_without_panicking() {
        use crate::{
            domains::{integer::Z, rational::Q},
            poly::PolynomialConversionError,
        };

        for text in ["1/((x+1)^2-x^2-2*x-1)", "1/((x+1)^2-x^2-2*x-1)^2"] {
            let atom = parse!(text);
            assert!(matches!(
                atom.try_to_rational_polynomial::<_, _, u32>(&Q, &Z, None),
                Err(PolynomialConversionError::ZeroDenominator { .. })
            ));
            assert!(matches!(
                atom.try_to_factorized_rational_polynomial::<_, _, u32>(&Q, &Z, None),
                Err(PolynomialConversionError::ZeroDenominator { .. })
            ));
            assert_eq!(atom.together(), atom, "{text}");
            assert_eq!(atom.check_zero(), ConditionResult::Inconclusive, "{text}");
        }
    }

    #[test]
    fn zero_test() {
        let e = parse!(
            "(sin(v1)^2-sin(v1))(sin(v1)^2+sin(v1))^2 - (1/4 sin(2v1)^2-1/2 sin(2v1)cos(v1)-2 cos(v1)^2+1/2 sin(2v1)cos(v1)^3+3 cos(v1)^4-cos(v1)^6)"
        );
        assert_eq!(e.zero_test(10, f64::EPSILON), ConditionResult::True);

        let e = parse!("x + (1+x)^2 + (x+2)*5");
        assert_eq!(e.zero_test(10, f64::EPSILON), ConditionResult::False);
    }

    #[test]
    fn large_coefficients_beyond_f64_range() {
        let a = crate::parse!("10^309+10^309*x");
        let r = a.zero_test(100, 1e-15);
        println!("{:?}", r);
    }

    #[test]
    fn test_big() {
        let field = Zp64::new(13);
        let x = crate::parse!("x");
        let mut samples = HashMap::default();
        samples.insert(x.into(), field.element_from_integer(2.into()));

        for expr in ["x+y", "1/x+y", "sqrt(2)+x", "1/(x+1)+x/(x+1)-1"] {
            let r = crate::parse!(expr)
                .as_view()
                .sample_in_finite_field(&mut samples, &field);

            println!("{expr} = {r:?}");
        }
    }

    #[test]
    fn rational_expressions_and_integer_powers() {
        let field = Zp64::new(13);
        let x = crate::parse!("x");
        let mut samples = HashMap::default();
        samples.insert(x.into(), field.element_from_integer(2.into()));

        for (expr, expected) in [
            ("-2/3", 8),
            ("x^-2", 10),
            ("(x+1)^2-x^2-2*x-1", 0),
            ("1/(x+1)+x/(x+1)-1", 0),
            ("10^100*(1+x)", 9),
            ("x^18446744073709551616", 3),
            ("x^-18446744073709551616", 9),
        ] {
            assert_eq!(
                crate::parse!(expr)
                    .as_view()
                    .sample_in_finite_field(&mut samples, &field),
                Ok(field.element_from_integer(expected.into())),
                "{expr}"
            );
        }
    }

    #[test]
    fn complex_coefficients_share_a_square_root_of_minus_one() {
        let field = Zp64::new(13);
        let mut samples = HashMap::default();
        let i = crate::parse!("1i")
            .as_view()
            .sample_in_finite_field(&mut samples, &field)
            .unwrap();
        assert_eq!(field.mul(&i, &i), field.neg(&field.one()));

        for (expr, re, im) in [
            ("2+3i", 2, 3),
            ("10^100+1i*10^100", 3, 3),
            ("1/10^100+1i/10^100", 9, 9),
            ("10^100", 3, 0),
        ] {
            let expected = field.add(
                &field.element_from_integer(re.into()),
                &field.mul(&field.element_from_integer(im.into()), &i),
            );
            assert_eq!(
                crate::parse!(expr)
                    .as_view()
                    .sample_in_finite_field(&mut samples, &field),
                Ok(expected),
                "{expr}"
            );
        }
        assert_eq!(samples.len(), 1);
    }

    #[test]
    fn variable_samples_survive_the_source_expression() {
        let field = Zp64::new(13);
        let mut samples = HashMap::default();
        let first = crate::parse!("x")
            .as_view()
            .sample_in_finite_field(&mut samples, &field)
            .unwrap();
        assert!(!field.is_zero(&first));
        assert_eq!(
            crate::parse!("x^2")
                .as_view()
                .sample_in_finite_field(&mut samples, &field),
            Ok(field.mul(&first, &first))
        );
        assert_eq!(samples.len(), 1);
    }

    #[test]
    fn singular_samples_return_errors() {
        let field = Zp64::new(13);
        let mut samples = HashMap::default();
        samples.insert(crate::parse!("x").into(), field.one());
        for expr in [
            "1/13",
            "1i/13",
            "1/13^100",
            "1i/13^100",
            "(x-1)^-1",
            "(x-1)^-18446744073709551616",
        ] {
            assert_eq!(
                crate::parse!(expr)
                    .as_view()
                    .sample_in_finite_field(&mut samples, &field),
                Err(PolyError::DivisionByZero),
                "{expr}"
            );
        }
        // A positive multiple of p - 1 is still a positive power of zero.
        assert_eq!(
            crate::parse!("(x-1)^12")
                .as_view()
                .sample_in_finite_field(&mut samples, &field),
            Ok(field.zero())
        );
    }

    #[test]
    fn exponential_samples_preserve_additive_decomposition() {
        let field = Zp64::new(13);
        let mut samples = HashMap::default();
        for (expr, value) in [("x", 2), ("y", 1), ("exp(1)", 9)] {
            samples.insert(
                crate::parse!(expr).into(),
                field.element_from_integer(value.into()),
            );
        }
        for (expr, expected) in [("x", 2), ("exp(x)", 3), ("exp(x+y)", 1)] {
            assert_eq!(
                crate::parse!(expr)
                    .as_view()
                    .sample_in_finite_field(&mut samples, &field),
                Ok(field.element_from_integer(expected.into())),
                "{expr}"
            );
        }

        let first = crate::parse!("exp(z)")
            .as_view()
            .sample_in_finite_field(&mut samples, &field)
            .unwrap();
        assert!(!field.is_zero(&first));
        assert_eq!(
            crate::parse!("exp(z)")
                .as_view()
                .sample_in_finite_field(&mut samples, &field),
            Ok(first)
        );
    }

    #[test]
    fn polynomial_exponentials_preserve_coefficients_and_constants() {
        for prime in [29, 18446744073709548557] {
            let field = Zp64::new(prime);
            let mut samples = HashMap::default();
            let mut sample = |expr: &str| {
                crate::parse!(expr)
                    .as_view()
                    .sample_in_finite_field(&mut samples, &field)
                    .unwrap()
            };
            let x = sample("exp(x)");
            assert_eq!(sample("exp(2*x)"), field.pow(&x, 2));
            assert_eq!(sample("exp(-x)"), field.inv(&x));
            assert_eq!(field.pow(&sample("exp(x/2)"), 2), x);
            assert_eq!(field.pow(&sample("exp(x/3)"), 3), x);
            let large = sample("exp(18446744073709551616*x)");
            assert_eq!(large, field.mul(&field.pow(&x, u64::MAX), &x));

            let e = sample("exp(1)");
            assert_eq!(sample("exp(x+2)"), field.mul(&x, &field.pow(&e, 2)));
            assert_eq!(field.pow(&sample("exp(1/2)"), 2), e);
            assert_eq!(sample("exp(-2)"), field.inv(&field.pow(&e, 2)));
            assert_eq!(
                sample("exp((x+y)^2+1/2)"),
                sample("exp(x^2)*exp(2*x*y)*exp(y^2)*exp(1/2)"),
            );

            let ix = sample("exp(1i*x)");
            assert_eq!(field.pow(&sample("exp(1i*x/2)"), 2), ix);
            assert_eq!(
                sample("exp((2+3i)*x)"),
                field.mul(&field.pow(&x, 2), &field.pow(&ix, 3)),
            );
            assert_eq!(
                sample("exp((1/10^100+1i/10^100)*x)"),
                sample("exp(x/10^100)*exp(1i*x/10^100)"),
            );
        }
    }

    #[test]
    fn sine_polynomials_use_the_same_exponential_generators() {
        let field = Zp64::new(29);
        let mut samples = HashMap::default();
        samples.insert(crate::parse!("x").into(), field.one());
        // A nontrivial, deterministic exp(ix) ensures the sine assertions do
        // not accidentally pass by mapping every exponential to one.
        samples.insert(
            crate::parse!("exp(1i)").into(),
            field.pow(&field.element_from_integer(2.into()), 4),
        );
        let mut sample = |expr: &str| {
            crate::parse!(expr)
                .as_view()
                .sample_in_finite_field(&mut samples, &field)
                .unwrap()
        };
        let sine = sample("sin(x)");
        assert!(!field.is_zero(&sine));
        assert_eq!(sample("sin(-x)"), field.neg(&sine));
        assert_eq!(sample("sin(2*x)"), sample("sin(x)*(exp(1i*x)+exp(-1i*x))"),);
        for (left, right) in [
            ("sin((x+y)^2+1/2)", "sin(x^2+2*x*y+y^2+1/2)"),
            ("sin(x/2)", "(exp(1i*x/2)-exp(-1i*x/2))/(2i)"),
            ("sin(1)", "(exp(1i)-exp(-1i))/(2i)"),
            ("sin(1i*x)", "(exp(-x)-exp(x))/(2i)"),
            (
                "sin((2+3i)*x+1/3)",
                "(exp((-3+2i)*x+1i/3)-exp((3-2i)*x-1i/3))/(2i)",
            ),
        ] {
            assert_eq!(sample(left), sample(right), "{left}");
        }
    }

    #[test]
    fn circular_and_hyperbolic_functions_share_exponential_samples() {
        let field = Zp64::new(29);
        let mut samples = HashMap::default();
        samples.insert(crate::parse!("x").into(), field.one());
        for (expr, base) in [("exp(1i)", 2), ("exp(1)", 3)] {
            samples.insert(
                crate::parse!(expr).into(),
                field.pow(&field.element_from_integer(base.into()), 4),
            );
        }
        let mut sample = |atom: Atom| atom.as_view().sample_in_finite_field(&mut samples, &field);
        let x = crate::parse!("x");
        for (function, formula) in [
            (Symbol::SIN, "(exp(1i*x)-exp(-1i*x))/(2i)"),
            (Symbol::COS, "(exp(1i*x)+exp(-1i*x))/2"),
            (transcendental::tan(), "sin(x)/cos(x)"),
            (transcendental::cot(), "cos(x)/sin(x)"),
            (transcendental::sec(), "1/cos(x)"),
            (transcendental::csc(), "1/sin(x)"),
            (transcendental::sinh(), "(exp(x)-exp(-x))/2"),
            (transcendental::cosh(), "(exp(x)+exp(-x))/2"),
            (transcendental::tanh(), "sinh(x)/cosh(x)"),
            (transcendental::coth(), "cosh(x)/sinh(x)"),
            (transcendental::sech(), "1/cosh(x)"),
            (transcendental::csch(), "1/sinh(x)"),
        ] {
            assert_eq!(
                sample(function.call(&x)).unwrap(),
                sample(crate::parse!(formula)).unwrap(),
                "{function}",
            );
        }
        for identity in [
            "sin(x)^2+cos(x)^2",
            "cosh(x)^2-sinh(x)^2",
            "cos(1i*x)/cosh(x)",
            "sin(1i*x)/(1i*sinh(x))",
            "tan(1i*x)/(1i*tanh(x))",
        ] {
            assert_eq!(
                sample(crate::parse!(identity)),
                Ok(field.one()),
                "{identity}"
            );
        }
        for (left, right) in [
            ("cos((x+y)^2+1/2)", "cos(x^2+2*x*y+y^2+1/2)"),
            ("sin(x+y)", "sin(x)*cos(y)+cos(x)*sin(y)"),
            ("cos(x+y)", "cos(x)*cos(y)-sin(x)*sin(y)"),
            ("sinh(x+y)", "sinh(x)*cosh(y)+cosh(x)*sinh(y)"),
            ("cosh(x+y)", "cosh(x)*cosh(y)+sinh(x)*sinh(y)"),
            (
                "tanh((2+3i)*finite_field_test::f(x)+1/2)",
                "sinh((2+3i)*finite_field_test::f(x)+1/2)/cosh((2+3i)*finite_field_test::f(x)+1/2)",
            ),
        ] {
            assert_eq!(
                sample(crate::parse!(left)),
                sample(crate::parse!(right)),
                "{left}"
            );
        }
    }

    #[test]
    fn geometric_functions_validate_arguments_and_sample_poles() {
        let field = Zp64::new(29);
        let x = crate::parse!("x");
        let nonrational = crate::parse!("x^(1/2)");
        for function in [
            Symbol::SIN,
            Symbol::COS,
            transcendental::tan(),
            transcendental::cot(),
            transcendental::sec(),
            transcendental::csc(),
            transcendental::sinh(),
            transcendental::cosh(),
            transcendental::tanh(),
            transcendental::coth(),
            transcendental::sech(),
            transcendental::csch(),
        ] {
            assert!(function.is_builtin());
            for atom in [
                function.call(&nonrational),
                Atom::var(Symbol::E).pow(function.call(&x)),
                Symbol::SIN.call(function.call(&x)),
            ] {
                assert_eq!(
                    atom.as_view()
                        .sample_in_finite_field(&mut HashMap::default(), &field),
                    Err(PolyError::NonRationalArgument),
                    "{atom}",
                );
            }
            assert_eq!(
                function
                    .call_args([&x, &x])
                    .as_view()
                    .sample_in_finite_field(&mut HashMap::default(), &field),
                Err(PolyError::InvalidFunctionArity {
                    function,
                    expected: 1,
                    actual: 2,
                }),
            );
        }
        let mut samples = HashMap::default();
        samples.insert(crate::parse!("exp(1)").into(), field.one());
        samples.insert(crate::parse!("exp(1i)").into(), field.one());
        for function in [
            transcendental::cot(),
            transcendental::csc(),
            transcendental::coth(),
            transcendental::csch(),
        ] {
            assert_eq!(
                function
                    .call(&x)
                    .as_view()
                    .sample_in_finite_field(&mut samples, &field),
                Err(PolyError::DivisionByZero),
                "{function}",
            );
        }
    }

    #[test]
    fn registered_transcendentals_are_not_opaque() {
        let field = Zp64::new(29);
        let x = crate::parse!("x");
        for function in [
            transcendental::asin(),
            transcendental::atanh(),
            transcendental::gamma(),
        ] {
            assert!(function.is_builtin());
            assert!(!function.is_fixed_builtin());
            let atom = function.call(&x);
            assert_eq!(
                atom.as_view()
                    .sample_in_finite_field(&mut HashMap::default(), &field),
                Err(PolyError::UnsupportedFunction(function)),
            );
            assert_eq!(
                Atom::var(Symbol::E)
                    .pow(atom)
                    .as_view()
                    .sample_in_finite_field(&mut HashMap::default(), &field),
                Err(PolyError::NonRationalArgument),
            );
        }
        // Symbol identity, not the printed function name, determines dispatch.
        let opaque_tan = crate::symbol!("finite_field_test::tan");
        assert!(!opaque_tan.is_builtin());
        assert!(
            opaque_tan
                .call(&x)
                .as_view()
                .sample_in_finite_field(&mut HashMap::default(), &field)
                .is_ok()
        );
    }

    #[test]
    fn opaque_calls_are_cached_as_independent_variables() {
        let field = Zp64::new(29);
        let mut samples = HashMap::default();
        for (expr, value) in [
            ("x", 2),
            ("finite_field_test::f(x)", 3),
            ("finite_field_test::f(y)", 5),
            ("finite_field_test::g(x)", 7),
        ] {
            samples.insert(
                crate::parse!(expr).into(),
                field.element_from_integer(value.into()),
            );
        }
        for (expr, expected) in [
            ("finite_field_test::f(x)", 3),
            ("finite_field_test::f(y)", 5),
            ("finite_field_test::g(x)", 7),
            (
                "finite_field_test::f(x)*finite_field_test::f(y)+finite_field_test::g(x)",
                22,
            ),
            ("finite_field_test::f(x)^-1", 10),
            (
                "(finite_field_test::f(x)+finite_field_test::g(x))^2-finite_field_test::f(x)^2-2*finite_field_test::f(x)*finite_field_test::g(x)-finite_field_test::g(x)^2",
                0,
            ),
        ] {
            assert_eq!(
                crate::parse!(expr)
                    .as_view()
                    .sample_in_finite_field(&mut samples, &field),
                Ok(field.element_from_integer(expected.into())),
                "{expr}",
            );
        }

        // These arguments cannot themselves be sampled, but the outer call is opaque.
        let first = crate::parse!("finite_field_test::f(log(z),1/z)")
            .as_view()
            .sample_in_finite_field(&mut samples, &field)
            .unwrap();
        assert!(!field.is_zero(&first));
        assert_eq!(
            crate::parse!("finite_field_test::f(log(z),1/z)")
                .as_view()
                .sample_in_finite_field(&mut samples, &field),
            Ok(first),
        );
        assert_eq!(samples.len(), 5);
    }

    #[test]
    fn opaque_calls_are_polynomial_variables_in_exp_and_sin() {
        let field = Zp64::new(29);
        let mut samples = HashMap::default();
        samples.insert(
            crate::parse!("finite_field_test::f(x)").into(),
            field.element_from_integer(3.into()),
        );
        let e = field.pow(&field.element_from_integer(2.into()), 4);
        let generator = field.pow(&e, 3);
        samples.insert(Symbol::E.into(), e);
        let mut sample = |expr: &str| {
            crate::parse!(expr)
                .as_view()
                .sample_in_finite_field(&mut samples, &field)
                .unwrap()
        };
        assert_eq!(sample("exp(finite_field_test::f(x))"), generator);
        assert_eq!(
            field.pow(&sample("exp(finite_field_test::f(x)/2)"), 2),
            generator
        );
        assert_eq!(
            sample("finite_field_test::f(x)"),
            field.element_from_integer(3.into())
        );
        for (left, right) in [
            (
                "exp((finite_field_test::f(x)+finite_field_test::g(y))^2)",
                "exp(finite_field_test::f(x)^2)*exp(2*finite_field_test::f(x)*finite_field_test::g(y))*exp(finite_field_test::g(y)^2)",
            ),
            (
                "sin(finite_field_test::f(x)/2+finite_field_test::g(y)^2)",
                "(exp(1i*finite_field_test::f(x)/2+1i*finite_field_test::g(y)^2)-exp(-1i*finite_field_test::f(x)/2-1i*finite_field_test::g(y)^2))/(2i)",
            ),
            (
                "sin(-finite_field_test::f(x))",
                "-sin(finite_field_test::f(x))",
            ),
            (
                "exp(2*finite_field_test::f(log(x)))",
                "exp(finite_field_test::f(log(x)))*exp(finite_field_test::f(log(x)))",
            ),
            (
                "sin(finite_field_test::f(exp(x)))",
                "(exp(1i*finite_field_test::f(exp(x)))-exp(-1i*finite_field_test::f(exp(x))))/(2i)",
            ),
        ] {
            assert_eq!(sample(left), sample(right), "{left}");
        }
    }

    #[test]
    fn half_pi_periods_preserve_exact_identities() {
        let field = Zp64::new(29);
        // Both embeddings of Q(i) must give the same identities.
        for root in [12, 17] {
            let mut samples = HashMap::default();
            for (expr, value) in [("x", 2), ("1i", root), ("exp(1)", 16), ("exp(1i)", 23)] {
                samples.insert(
                    crate::parse!(expr).into(),
                    field.element_from_integer(value.into()),
                );
            }
            for identity in [
                "exp(2i*pi)-1",
                "exp(1i*pi)+1",
                "exp(1i*pi/2)-1i",
                "exp(-1i*pi/2)+1i",
                "exp(100000000000000000000000000000003i*pi)+1",
                "sin(pi)",
                "cos(pi)+1",
                "sin(pi/2)-1",
                "cos(pi/2)",
                "sin(x+2*pi)-sin(x)",
                "sin(x+pi)+sin(x)",
                "sin(x+pi/2)-cos(x)",
                "sin((2*x+pi)/2)-cos(x)",
                "cos(x+pi/2)+sin(x)",
                "tan(x+pi)-tan(x)",
                "cot(x+pi)-cot(x)",
                "sec(x+pi)+sec(x)",
                "csc(x+pi)+csc(x)",
                "exp(x+1i*pi)+exp(x)",
                "sinh(x+1i*pi)+sinh(x)",
                "cosh(x+1i*pi)+cosh(x)",
                "cosh(x+1i*pi/2)-1i*sinh(x)",
                "tanh(x+1i*pi)-tanh(x)",
                "coth(x+1i*pi)-coth(x)",
                "sech(x+1i*pi)+sech(x)",
                "csch(x+1i*pi)+csch(x)",
            ] {
                assert_eq!(
                    crate::parse!(identity)
                        .as_view()
                        .sample_in_finite_field(&mut samples, &field),
                    Ok(field.zero()),
                    "{identity}, i = {root}",
                );
            }
        }
    }

    #[test]
    fn rational_arguments_preserve_identities_without_expansion() {
        for prime in [29, 53] {
            let field = Zp64::new(prime);
            for x in [2, 3] {
                let mut samples = HashMap::default();
                samples.insert(
                    crate::parse!("x").into(),
                    field.element_from_integer(x.into()),
                );
                samples.insert(
                    crate::parse!("finite_field_test::f(x)").into(),
                    field.element_from_integer(x.into()),
                );
                samples.insert(
                    Symbol::E.into(),
                    field.pow(&field.element_from_integer(2.into()), 4),
                );
                samples.insert(
                    crate::parse!("exp(1i)").into(),
                    field.pow(&field.element_from_integer(3.into()), 4),
                );
                for identity in [
                    "exp(x/(x+1))*exp(1/(x+1))-exp(1)",
                    "exp(1/x+1/(x+1))-exp((2*x+1)/(x*(x+1)))",
                    "exp((x^2-1)/(x-1))-exp(x+1)",
                    "exp(1/(x+1i))-exp((x-1i)/(x^2+1))",
                    "exp(1/(x+1i)+1/(x-1i))-exp(2*x/(x^2+1))",
                    "sin(x/(x+1)+1/(x+1))-sin(1)",
                    "cos(1/x)^2+sin(1/x)^2-1",
                    "cosh(1/x)^2-sinh(1/x)^2-1",
                    "sin(1/x+1/(x+1))-sin(1/x)*cos(1/(x+1))-cos(1/x)*sin(1/(x+1))",
                    "exp(1/x+1i*pi)+exp(1/x)",
                    "sin(1/x+pi/2)-cos(1/x)",
                    "exp(1/finite_field_test::f(x))*exp((finite_field_test::f(x)-1)/finite_field_test::f(x))-exp(1)",
                    "exp((x+1)^1000000)*exp(-(x+1)^1000000)-1",
                ] {
                    assert_eq!(
                        crate::parse!(identity)
                            .as_view()
                            .sample_in_finite_field(&mut samples, &field),
                        Ok(field.zero()),
                        "{identity}, p = {prime}, x = {x}",
                    );
                }
                // Exercise a huge unexpanded power even if normalization above
                // cancels the opposite exponents before sampling.
                assert!(
                    crate::parse!("exp((x+1)^1000000)")
                        .as_view()
                        .sample_in_finite_field(&mut samples, &field)
                        .is_ok()
                );
            }
        }
    }

    #[test]
    fn rational_argument_poles_are_checked_in_the_exponent_field() {
        for (prime, x, expr) in [
            (29, 7, "exp(1/x)"),
            (29, 6, "sin(1/(x+1))"),
            // F_13[i] splits: 5+i is a nonzero zero divisor, with norm 26.
            (53, 5, "exp(1/(x+1i))"),
        ] {
            let field = Zp64::new(prime);
            let mut samples = HashMap::default();
            samples.insert(
                crate::parse!("x").into(),
                field.element_from_integer(x.into()),
            );
            assert_eq!(
                crate::parse!(expr)
                    .as_view()
                    .sample_in_finite_field(&mut samples, &field),
                Err(PolyError::NonInvertibleExponentDenominator {
                    subgroup_order: (prime - 1) / 4
                }),
                "{expr}",
            );
        }
    }

    #[test]
    fn unsupported_periods_and_constants_cannot_give_false_nonzeros() {
        let field = Zp64::new(29);
        for expr in [
            "exp(pi)",
            "exp(pi*x)",
            "sin(pi*x)",
            "sin(pi/3)",
            "exp(1i*pi/3)",
            "sin(pi^2)",
            "exp(1/pi)",
            "sin(1/(x+pi))",
        ] {
            assert_eq!(
                crate::parse!(expr)
                    .as_view()
                    .sample_in_finite_field(&mut HashMap::default(), &field),
                Err(PolyError::UnsupportedPeriod),
                "{expr}",
            );
        }
        for (expr, symbol) in [
            ("pi", Symbol::PI),
            ("pi+exp(1)", Symbol::PI),
            ("exp(exp(1))", Symbol::E),
            ("sin(exp(1))", Symbol::E),
        ] {
            assert_eq!(
                crate::parse!(expr)
                    .as_view()
                    .sample_in_finite_field(&mut HashMap::default(), &field),
                Err(PolyError::UnsupportedConstant(symbol)),
                "{expr}",
            );
        }
        let constant = transcendental::euler_gamma();
        assert_eq!(
            Atom::var(constant)
                .as_view()
                .sample_in_finite_field(&mut HashMap::default(), &field),
            Err(PolyError::UnsupportedConstant(constant)),
        );
    }

    #[test]
    fn certification_checks_the_modulus_and_cached_imaginary_unit() {
        assert_eq!(
            crate::parse!("x+1")
                .as_view()
                .sample_in_finite_field(&mut HashMap::default(), &Zp64::new(9)),
            Err(PolyError::NonPrimeField { modulus: 9 }),
        );
        let field = Zp64::new(29);
        let mut samples = HashMap::default();
        samples.insert(crate::parse!("1i").into(), field.one());
        for expr in ["1i", "exp(1i*pi/2)"] {
            assert_eq!(
                crate::parse!(expr)
                    .as_view()
                    .sample_in_finite_field(&mut samples, &field),
                Err(PolyError::InvalidImaginaryUnit),
            );
        }
    }

    #[test]
    fn exponential_and_sine_arguments_must_be_rational() {
        let field = Zp64::new(29);
        for expr in [
            "exp(x^(1/2))",
            "exp(x^y)",
            "exp(exp(x))",
            "exp(sin(x))",
            "sin(x^(1/2))",
            "sin(exp(x))",
            "sin(sin(x))",
            "sin(log(x))",
        ] {
            assert_eq!(
                crate::parse!(expr)
                    .as_view()
                    .sample_in_finite_field(&mut HashMap::default(), &field),
                Err(PolyError::NonRationalArgument),
                "{expr}",
            );
        }
    }

    #[test]
    fn exponential_sampling_checks_subgroup_and_denominators() {
        for (prime, expected) in [
            (5, PolyError::SubgroupNotOddPrime { order: 1 }),
            (7, PolyError::MissingImaginaryUnit { prime: 7 }),
            (17, PolyError::SubgroupNotOddPrime { order: 4 }),
            (37, PolyError::SubgroupNotOddPrime { order: 9 }),
        ] {
            for expr in ["exp(x)", "sin(x)"] {
                assert_eq!(
                    crate::parse!(expr)
                        .as_view()
                        .sample_in_finite_field(&mut HashMap::default(), &Zp64::new(prime)),
                    Err(expected),
                    "{expr} over {prime}",
                );
            }
        }
        let field = Zp64::new(29);
        for expr in ["exp(x/7)", "exp(1i*x/7^100)", "sin(x/7)"] {
            assert_eq!(
                crate::parse!(expr)
                    .as_view()
                    .sample_in_finite_field(&mut HashMap::default(), &field),
                Err(PolyError::NonInvertibleExponentDenominator { subgroup_order: 7 }),
                "{expr}",
            );
        }
        for value in [0, 2] {
            let mut samples = HashMap::default();
            samples.insert(crate::parse!("x").into(), field.one());
            samples.insert(Symbol::E.into(), field.element_from_integer(value.into()));
            assert_eq!(
                crate::parse!("exp(x)")
                    .as_view()
                    .sample_in_finite_field(&mut samples, &field),
                Err(PolyError::InvalidExponentialSample { subgroup_order: 7 }),
            );
        }
    }

    #[test]
    fn unsupported_inputs_return_errors() {
        let field = Zp64::new(13);
        for (expr, expected) in [
            ("x^(1/2)", PolyError::NonIntegerExponent),
            ("x^(18446744073709551616/3)", PolyError::NonIntegerExponent),
            ("x^y", PolyError::NonIntegerExponent),
            ("exp(log(x)*y)", PolyError::NonIntegerExponent),
            ("log(x)", PolyError::UnsupportedFunction(Symbol::LOG)),
            (
                "sin(x,y)",
                PolyError::InvalidFunctionArity {
                    function: Symbol::SIN,
                    expected: 1,
                    actual: 2,
                },
            ),
        ] {
            assert_eq!(
                crate::parse!(expr)
                    .as_view()
                    .sample_in_finite_field(&mut HashMap::default(), &field),
                Err(expected),
                "{expr}"
            );
        }
        for expr in ["1i", "10^100*1i"] {
            assert_eq!(
                crate::parse!(expr)
                    .as_view()
                    .sample_in_finite_field(&mut HashMap::default(), &Zp64::new(7)),
                Err(PolyError::MissingImaginaryUnit { prime: 7 }),
                "{expr}",
            );
        }
        for atom in [
            Atom::num(0.5),
            Atom::num(Coefficient::Indeterminate),
            Atom::num(Coefficient::complex_infinity()),
        ] {
            assert_eq!(
                atom.as_view()
                    .sample_in_finite_field(&mut HashMap::default(), &field),
                Err(PolyError::UnsupportedCoefficient),
            );
        }
    }

    fn high_degree_cases() -> [(&'static str, Atom); 4] {
        // A has degree 32, B degree 31. Keeping the two sides in different
        // factorizations prevents normalization from erasing the identity.
        let a = crate::parse!("(x+2*y-z+3*w)^32/3");
        let b = crate::parse!("(2*x-y+z-w)^31/5");
        let zero = a.pow(2) - b.pow(2) - (&a - &b) * (&a + &b);
        let perturbation = crate::parse!("x^32*y^16*z^8*w^8/7");
        let nonzero = &zero + perturbation;
        let denominator = crate::parse!("(x^2+y^2+z^2+w^2+1)^4");
        let rational_zero = &zero / &denominator;
        let rational_nonzero = &nonzero / denominator;
        assert!(!zero.is_zero());
        assert!(!rational_zero.is_zero());
        [
            ("degree64_polynomial_zero", zero),
            ("degree64_polynomial_nonzero", nonzero),
            ("degree64_rational_zero", rational_zero),
            ("degree64_rational_nonzero", rational_nonzero),
        ]
    }

    #[test]
    fn high_degree_multivariate_rational_zero_and_nonzero() {
        let cases = high_degree_cases();
        for prime in [(1 << 31) - 1, (1 << 61) - 1]
            .into_iter()
            .chain(SAMPLE_PRIMES.iter().map(|sample| sample.prime))
        {
            let field = Zp64::new(prime);
            for point in [[2, 3, 5, 7], [11, 13, 17, 19], [-2, 3, -5, 7]] {
                let mut samples = HashMap::default();
                let values = point.map(|value| field.element_from_integer(value.into()));
                for (name, value) in ["x", "y", "z", "w"].into_iter().zip(values) {
                    samples.insert(crate::parse!(name).into(), value);
                }
                // Compute the expected perturbation directly in the field,
                // independently of the expression sampler and normalizer.
                let mut perturbation = field.inv(&field.element_from_integer(7.into()));
                let mut denominator = field.one();
                for (value, power) in values.iter().zip([32, 16, 8, 8]) {
                    field.mul_assign(&mut perturbation, &field.pow(value, power));
                    field.add_assign(&mut denominator, &field.mul(value, value));
                }
                denominator = field.pow(&denominator, 4);
                assert!(!field.is_zero(&denominator));
                assert!(!field.is_zero(&perturbation));
                let rational_perturbation = field.div(&perturbation, &denominator);
                for ((name, atom), expected) in cases.iter().zip([
                    field.zero(),
                    perturbation,
                    field.zero(),
                    rational_perturbation,
                ]) {
                    assert_eq!(
                        atom.as_view().sample_in_finite_field(&mut samples, &field),
                        Ok(expected),
                        "{name}, p={prime}, point={point:?}",
                    );
                }
            }
        }
    }
}
