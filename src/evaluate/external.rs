use super::*;

/// A registered external function together with its symbolic tags, numerical
/// implementation, and optional shared instruction body. Each evaluator owns one
/// entry per function, including its reusable evaluation stack; cloned evaluators
/// share bodies but have independent stacks.
pub struct ExternalFunctionContainer<T> {
    pub(super) export_name: String,
    pub(super) symbol: Symbol,
    pub(super) tags: Vec<Atom>,
    pub(super) fixed_args: Vec<Complex<Rational>>,
    pub(super) imp: Option<Box<dyn ExternalFunction<T>>>,
    pub(super) constant_index: Option<usize>,
    // Reusable body stack, or argument buffer for callbacks. Allocated on first
    // use and grown for larger callback arities; never shared or serialized.
    pub(super) stack: Vec<T>,
    pub(super) body: Option<Arc<FunctionBody>>,
}

#[cfg(feature = "serde")]
impl<T> serde::Serialize for ExternalFunctionContainer<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (
            &self.export_name,
            &self.symbol,
            self.tags
                .iter()
                .map(|x| x.to_canonical_string())
                .collect::<Vec<_>>(),
            &self.fixed_args,
            &self.constant_index,
            &self.body.as_deref(),
        )
            .serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: EvaluationDomain> serde::Deserialize<'de> for ExternalFunctionContainer<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (export_name, symbol, tags, fixed_args, constant_index, body): (
            String,
            Symbol,
            Vec<String>,
            Vec<Complex<Rational>>,
            Option<usize>,
            Option<Box<FunctionBody>>,
        ) = serde::Deserialize::deserialize(deserializer)?;

        let mut external = Self {
            export_name,
            symbol,
            tags: tags.iter().map(|s| crate::parse!(s)).collect(),
            fixed_args,
            imp: None,
            stack: Vec::new(),
            constant_index,
            body: body.map(Arc::from),
        };
        if external.body.is_none() {
            external.imp = external.fetch_impl_for::<T>();
        }
        Ok(external)
    }
}

#[cfg(feature = "bincode")]
impl<T> bincode::Encode for ExternalFunctionContainer<T> {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> core::result::Result<(), bincode::error::EncodeError> {
        bincode::Encode::encode(&self.export_name, encoder)?;
        bincode::Encode::encode(&self.symbol, encoder)?;
        bincode::Encode::encode(
            &self
                .tags
                .iter()
                .map(|x| x.to_canonical_string())
                .collect::<Vec<_>>(),
            encoder,
        )?;
        bincode::Encode::encode(&self.fixed_args, encoder)?;
        bincode::Encode::encode(&self.constant_index, encoder)?;
        bincode::Encode::encode(&self.body.as_deref(), encoder)
    }
}

#[cfg(feature = "bincode")]
impl<Context, T: EvaluationDomain> bincode::Decode<Context> for ExternalFunctionContainer<T> {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let export_name: String = bincode::Decode::decode(decoder)?;
        let symbol: Symbol = bincode::Decode::decode(decoder)?;
        let tags: Vec<String> = bincode::Decode::decode(decoder)?;
        let fixed_args: Vec<Complex<Rational>> = bincode::Decode::decode(decoder)?;
        let constant_index: Option<usize> = bincode::Decode::decode(decoder)?;
        let body: Option<Box<FunctionBody>> = bincode::Decode::decode(decoder)?;

        let mut external = Self {
            export_name,
            symbol,
            tags: tags.iter().map(|s| crate::parse!(s)).collect(),
            fixed_args,
            imp: None,
            stack: Vec::new(),
            constant_index,
            body: body.map(Arc::from),
        };
        if external.body.is_none() {
            external.imp = external.fetch_impl_for::<T>();
        }

        Ok(external)
    }
}

#[cfg(feature = "bincode")]
impl<'de, Context, T: EvaluationDomain> bincode::BorrowDecode<'de, Context>
    for ExternalFunctionContainer<T>
{
    fn borrow_decode<D: bincode::de::BorrowDecoder<'de, Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let export_name: String = bincode::BorrowDecode::borrow_decode(decoder)?;
        let symbol: Symbol = bincode::BorrowDecode::borrow_decode(decoder)?;
        let tags: Vec<String> = bincode::BorrowDecode::borrow_decode(decoder)?;
        let fixed_args: Vec<Complex<Rational>> = bincode::BorrowDecode::borrow_decode(decoder)?;
        let constant_index: Option<usize> = bincode::BorrowDecode::borrow_decode(decoder)?;
        let body: Option<Box<FunctionBody>> = bincode::BorrowDecode::borrow_decode(decoder)?;

        let mut external = Self {
            export_name,
            symbol,
            tags: tags.iter().map(|s| crate::parse!(s)).collect(),
            fixed_args,
            imp: None,
            stack: Vec::new(),
            constant_index,
            body: body.map(Arc::from),
        };
        if external.body.is_none() {
            external.imp = external.fetch_impl_for::<T>();
        }
        Ok(external)
    }
}

impl<T> Clone for ExternalFunctionContainer<T> {
    fn clone(&self) -> Self {
        Self {
            export_name: self.export_name.clone(),
            symbol: self.symbol,
            tags: self.tags.clone(),
            fixed_args: self.fixed_args.clone(),
            imp: self.imp.clone(),
            stack: Vec::new(),
            constant_index: self.constant_index,
            body: self.body.clone(),
        }
    }
}

impl<T> std::fmt::Debug for ExternalFunctionContainer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternalFunctionContainer")
            .field("export_name", &self.export_name)
            .field("eval_name", &self.symbol)
            .field("tags", &self.tags)
            .field("fixed_args", &self.fixed_args)
            .field("imp", &self.imp.is_some())
            .field("constant_index", &self.constant_index)
            .field("body", &self.body.is_some())
            .finish()
    }
}

impl<T> std::hash::Hash for ExternalFunctionContainer<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.export_name.hash(state);
        self.symbol.hash(state);
        for tag in &self.tags {
            tag.hash(state);
        }
    }
}

impl<T> PartialEq for ExternalFunctionContainer<T> {
    fn eq(&self, other: &Self) -> bool {
        self.export_name == other.export_name
            && self.symbol == other.symbol
            && self.tags == other.tags
            && self.fixed_args == other.fixed_args
    }
}

impl<T> Eq for ExternalFunctionContainer<T> {}

impl<T> ExternalFunctionContainer<T> {
    pub(super) fn new(symbol: Symbol, tags: Vec<Atom>, fixed_args: Vec<Complex<Rational>>) -> Self {
        let mut export_name = symbol
            .get_ascii_name()
            .ok_or_else(|| {
                format!(
                    "No ASCII name for symbol {symbol} available, which is needed for exporting"
                )
            })
            .unwrap();

        // A generator receives the chosen name, so it can use escaped tags.
        // Preserve the historical names for fixed snippets and externally
        // supplied functions, whose definitions may already depend on them.
        let escape_tags = symbol
            .get_evaluation_info()
            .is_some_and(EvaluationInfo::has_cpp_generator);
        for t in &tags {
            export_name += "_";
            let tag = t.to_canonical_string();
            if escape_tags {
                // Escape underscores as well to keep the encoding injective.
                for byte in tag.bytes() {
                    if byte.is_ascii_alphanumeric() {
                        export_name.push(byte as char);
                    } else {
                        export_name += &format!("_{byte:02x}");
                    }
                }
            } else {
                export_name += &tag;
            }
        }

        Self {
            export_name,
            symbol,
            tags,
            fixed_args,
            imp: None,
            stack: Vec::new(),
            constant_index: None,
            body: None,
        }
    }

    pub(super) fn with_body(symbol: Symbol, tags: Vec<Atom>, body: Arc<FunctionBody>) -> Self {
        let mut container = Self::new(symbol, tags, vec![]);
        container.body = Some(body);
        container
    }

    pub(super) fn export_name(&self) -> &str {
        &self.export_name
    }

    pub(super) fn tag_views(&self) -> Vec<AtomView<'_>> {
        self.tags.iter().map(|x| x.as_view()).collect()
    }

    /// Resolve a registered constant or fixed-argument function in the target domain.
    pub(super) fn evaluate_constant<T2: EvaluationDomain>(
        &self,
        binary_prec: u32,
    ) -> Result<T2, String> {
        let info = self
            .symbol
            .get_evaluation_info()
            .ok_or_else(|| format!("External constant '{}' does not have evaluation info", self))?;
        let tags = self.tag_views();
        if self.fixed_args.is_empty() {
            T2::try_from_complex_float(info.evaluate_constant(&tags, binary_prec)?)
        } else {
            info.evaluate_fixed_arguments(&tags, &self.fixed_args, binary_prec)
                .map_err(|cause| format!("External constant '{self}': {cause}"))
        }
    }

    pub(super) fn map<T2: EvaluationDomain>(&self) -> ExternalFunctionContainer<T2> {
        ExternalFunctionContainer {
            export_name: self.export_name.clone(),
            symbol: self.symbol,
            tags: self.tags.clone(),
            fixed_args: self.fixed_args.clone(),
            imp: if self.body.is_none() {
                self.fetch_impl_for::<T2>()
            } else {
                None
            },
            stack: Vec::new(),
            constant_index: self.constant_index,
            body: self.body.clone(),
        }
    }

    #[cfg(feature = "native_code_generation")]
    pub(super) fn callable(&self) -> Option<Box<dyn ExternalFunction<T>>> {
        self.imp.clone()
    }

    pub(super) fn fetch_impl_for<T2: EvaluationDomain>(
        &self,
    ) -> Option<Box<dyn ExternalFunction<T2>>> {
        let info = self.symbol.get_evaluation_info()?;
        let tags = self.tag_views();
        T2::resolve_function(&tags, info)
    }

    #[cfg(feature = "native_code_generation")]
    pub(super) fn cpp(&self) -> Option<std::borrow::Cow<'_, str>> {
        self.symbol
            .get_evaluation_info()?
            .generate_cpp(self.export_name(), &self.tag_views())
    }
}

impl<T> std::fmt::Display for ExternalFunctionContainer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.export_name())
    }
}

/// An external function implementation that can be registered on
/// [`crate::atom::EvaluationInfo`] and called by an evaluator.
pub trait ExternalFunction<T>: Fn(&[T]) -> T + Send + Sync + DynClone + Send + Sync {}
dyn_clone::clone_trait_object!(<T> ExternalFunction<T>);
impl<T, F: Clone + Send + Sync + Fn(&[T]) -> T + Send + Sync> ExternalFunction<T> for F {}

#[cfg(test)]
mod fixed_tracking_tests {
    use crate::{
        atom::{Atom, AtomCore},
        domains::{
            float::{Complex, DoubleFloat, ErrorPropagatingFloat, Float, RealLike},
            rational::Rational,
        },
        evaluate::{EvaluationDomain, ExpressionEvaluator},
    };
    type Tracked = ErrorPropagatingFloat<Float>;

    fn argument(re: Rational) -> Complex<Rational> {
        Complex::new(re, Rational::from(0))
    }
    fn fixed<T: EvaluationDomain>(z: Complex<Rational>, bits: u32) -> Result<T, String> {
        let order = Atom::num(2);
        crate::transcendental::polygamma()
            .get_evaluation_info()
            .unwrap()
            .evaluate_fixed_arguments(&[order.as_view()], &[z], bits)
    }
    fn exact() -> ExpressionEvaluator<Complex<Rational>> {
        crate::transcendental::polygamma()
            .call_args([Atom::num(2), Atom::one()])
            .evaluator(&[] as &[Atom])
            .build()
            .unwrap()
    }
    fn eval(exact: ExpressionEvaluator<Complex<Rational>>, bits: u32) -> Tracked {
        let mut mapped = exact.map_coeff_with_prec(
            &|c| {
                Tracked::try_from_complex_float(Complex::new(
                    c.re.to_multi_prec_float(bits),
                    c.im.to_multi_prec_float(bits),
                ))
                .unwrap()
            },
            bits,
        );
        let mut output = [Tracked::new(Float::new(bits), 30.)];
        mapped.evaluate(&[], &mut output);
        output[0].clone()
    }

    #[test]
    fn exact_fixed_polygamma_retains_native_assumed_precision() {
        assert!(fixed::<Tracked>(argument(Rational::from((1, 3))), 128).is_err());
        let expected = -2.404_113_806_319_188_5;
        let native64: ErrorPropagatingFloat<f64> = fixed(argument(Rational::from(1)), 53).unwrap();
        assert!((native64.get_num() - expected).abs() < 1e-15);
        assert!(native64.get_absolute_error().is_finite() && native64.get_absolute_error() > 0.);
        let complex64: Complex<ErrorPropagatingFloat<f64>> =
            fixed(argument(Rational::from(1)), 53).unwrap();
        assert_eq!(complex64.re.get_num(), native64.get_num());
        assert!(complex64.im.get_absolute_error() > 0.);
        for bits in [128, 256] {
            let value = eval(exact(), bits);
            assert_eq!(value.get_num().prec(), bits);
            assert!((value.get_num().to_f64() - expected).abs() < 1e-15);
            assert!(value.get_absolute_error().is_finite() && value.get_absolute_error() > 0.);
            let complex: Complex<Tracked> = fixed(argument(Rational::from(1)), bits).unwrap();
            assert_eq!(complex.re.get_num(), value.get_num());
            assert_eq!(complex.re.get_num().prec(), bits);
            assert_eq!(complex.im.get_num().prec(), bits);
            assert!(complex.re.get_absolute_error() > 0. && complex.im.get_absolute_error() > 0.);
        }
        // Existing Float and DoubleFloat implementations still take precedence.
        let ordinary: Float = fixed(argument(Rational::from(1)), 128).unwrap();
        let wide: DoubleFloat = fixed(argument(Rational::from(1)), 106).unwrap();
        let wide_complex: Complex<DoubleFloat> = fixed(argument(Rational::from(1)), 106).unwrap();
        assert!((ordinary.to_f64() - expected).abs() < 1e-15);
        assert!((wide.to_f64() - expected).abs() < 1e-15);
        assert_eq!(wide_complex.re, wide);
    }
}
