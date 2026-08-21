use super::*;

pub struct ExternalFunctionContainer<T> {
    pub(super) export_name: String,
    pub(super) symbol: Symbol,
    pub(super) tags: Vec<Atom>,
    pub(super) fixed_args: Vec<Complex<Rational>>,
    pub(super) imp: Option<Box<dyn ExternalFunction<T>>>,
    pub(super) cache: Vec<T>,
    pub(super) constant_index: Option<usize>,
    pub(super) sub_evaluator: Option<Box<ExpressionEvaluator<T>>>,
}

#[cfg(feature = "serde")]
impl<T: serde::Serialize> serde::Serialize for ExternalFunctionContainer<T> {
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
            &self.sub_evaluator,
        )
            .serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: serde::Deserialize<'de> + EvaluationDomain> serde::Deserialize<'de>
    for ExternalFunctionContainer<T>
{
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (export_name, symbol, tags, fixed_args, constant_index, sub_evaluator): (
            String,
            Symbol,
            Vec<String>,
            Vec<Complex<Rational>>,
            Option<usize>,
            Option<Box<ExpressionEvaluator<T>>>,
        ) = serde::Deserialize::deserialize(deserializer)?;

        let mut external = Self {
            export_name,
            symbol,
            tags: tags.iter().map(|s| crate::parse!(s)).collect(),
            fixed_args,
            imp: None,
            cache: vec![],
            constant_index,
            sub_evaluator,
        };
        if external.sub_evaluator.is_none() {
            external.imp = external.fetch_impl_for::<T>();
        }
        Ok(external)
    }
}

#[cfg(feature = "bincode")]
impl<T: bincode::Encode> bincode::Encode for ExternalFunctionContainer<T> {
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
        bincode::Encode::encode(&self.sub_evaluator, encoder)
    }
}

#[cfg(feature = "bincode")]
impl<Context, T: bincode::Decode<Context> + EvaluationDomain> bincode::Decode<Context>
    for ExternalFunctionContainer<T>
{
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let export_name: String = bincode::Decode::decode(decoder)?;
        let symbol: Symbol = bincode::Decode::decode(decoder)?;
        let tags: Vec<String> = bincode::Decode::decode(decoder)?;
        let fixed_args: Vec<Complex<Rational>> = bincode::Decode::decode(decoder)?;
        let constant_index: Option<usize> = bincode::Decode::decode(decoder)?;
        let sub_evaluator: Option<Box<ExpressionEvaluator<T>>> = bincode::Decode::decode(decoder)?;

        let mut external = Self {
            export_name,
            symbol,
            tags: tags.iter().map(|s| crate::parse!(s)).collect(),
            fixed_args,
            imp: None,
            cache: vec![],
            constant_index,
            sub_evaluator,
        };
        if external.sub_evaluator.is_none() {
            external.imp = external.fetch_impl_for::<T>();
        }

        Ok(external)
    }
}

#[cfg(feature = "bincode")]
impl<'de, Context, T: bincode::BorrowDecode<'de, Context> + EvaluationDomain>
    bincode::BorrowDecode<'de, Context> for ExternalFunctionContainer<T>
{
    fn borrow_decode<D: bincode::de::BorrowDecoder<'de, Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let export_name: String = bincode::BorrowDecode::borrow_decode(decoder)?;
        let symbol: Symbol = bincode::BorrowDecode::borrow_decode(decoder)?;
        let tags: Vec<String> = bincode::BorrowDecode::borrow_decode(decoder)?;
        let fixed_args: Vec<Complex<Rational>> = bincode::BorrowDecode::borrow_decode(decoder)?;
        let constant_index: Option<usize> = bincode::BorrowDecode::borrow_decode(decoder)?;
        let sub_evaluator: Option<Box<ExpressionEvaluator<T>>> =
            bincode::BorrowDecode::borrow_decode(decoder)?;

        let mut external = Self {
            export_name,
            symbol,
            tags: tags.iter().map(|s| crate::parse!(s)).collect(),
            fixed_args,
            imp: None,
            cache: vec![],
            constant_index,
            sub_evaluator,
        };
        if external.sub_evaluator.is_none() {
            external.imp = external.fetch_impl_for::<T>();
        }
        Ok(external)
    }
}

impl<T: Clone> Clone for ExternalFunctionContainer<T> {
    fn clone(&self) -> Self {
        Self {
            export_name: self.export_name.clone(),
            symbol: self.symbol,
            tags: self.tags.clone(),
            fixed_args: self.fixed_args.clone(),
            imp: self.imp.clone(),
            cache: vec![],
            constant_index: self.constant_index,
            sub_evaluator: self.sub_evaluator.clone(),
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
            .field("cache_len", &self.cache.len())
            .field("constant_index", &self.constant_index)
            .field("sub_evaluator", &self.sub_evaluator.is_some())
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

        // TODO: escape minus signs, etc
        for t in &tags {
            export_name += "_";
            export_name += &t.to_canonical_string();
        }

        Self {
            export_name,
            symbol,
            tags,
            fixed_args,
            imp: None,
            cache: vec![],
            constant_index: None,
            sub_evaluator: None,
        }
    }

    pub(super) fn new_sub_evaluator(
        symbol: Symbol,
        tags: Vec<Atom>,
        evaluator: ExpressionEvaluator<T>,
    ) -> Self {
        let mut container = Self::new(symbol, tags, vec![]);
        container.sub_evaluator = Some(Box::new(evaluator));
        container
    }

    pub(super) fn export_name(&self) -> &str {
        &self.export_name
    }

    pub(super) fn tag_views(&self) -> Vec<AtomView<'_>> {
        self.tags.iter().map(|x| x.as_view()).collect()
    }

    pub(super) fn map<T2: EvaluationDomain>(&self) -> ExternalFunctionContainer<T2> {
        debug_assert!(self.sub_evaluator.is_none());
        ExternalFunctionContainer {
            export_name: self.export_name.clone(),
            symbol: self.symbol,
            tags: self.tags.clone(),
            fixed_args: self.fixed_args.clone(),
            imp: self.fetch_impl_for::<T2>(),
            cache: vec![],
            constant_index: self.constant_index,
            sub_evaluator: None,
        }
    }

    pub(super) fn map_owned<T2: Default + Clone + EvaluationDomain>(
        self,
    ) -> ExternalFunctionContainer<T2>
    where
        T: Default,
    {
        let imp = if self.sub_evaluator.is_none() {
            self.fetch_impl_for::<T2>()
        } else {
            None
        };
        let sub_evaluator = self
            .sub_evaluator
            .map(|evaluator| Box::new(evaluator.set_coeff(&[])));
        ExternalFunctionContainer {
            export_name: self.export_name,
            symbol: self.symbol,
            tags: self.tags,
            fixed_args: self.fixed_args,
            imp,
            cache: vec![],
            constant_index: self.constant_index,
            sub_evaluator,
        }
    }

    pub(super) fn map_coeff<T2: EvaluationDomain, F: Fn(&T) -> T2>(
        self,
        f: &F,
        binary_prec: u32,
    ) -> ExternalFunctionContainer<T2>
    where
        T: Default,
    {
        let imp = if self.sub_evaluator.is_none() {
            self.fetch_impl_for::<T2>()
        } else {
            None
        };
        let sub_evaluator = self
            .sub_evaluator
            .map(|evaluator| Box::new(evaluator.map_coeff_with_prec(f, binary_prec)));
        ExternalFunctionContainer {
            export_name: self.export_name,
            symbol: self.symbol,
            tags: self.tags,
            fixed_args: self.fixed_args,
            imp,
            cache: vec![],
            constant_index: self.constant_index,
            sub_evaluator,
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
    pub(super) fn cpp(&self) -> Option<&str> {
        self.symbol.get_evaluation_info()?.get_cpp()
    }
}

impl<T> std::fmt::Display for ExternalFunctionContainer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.export_name())
    }
}

#[cfg(feature = "native_code_generation")]
impl ExternalFunctionContainer<Complex<Rational>> {
    pub(super) fn map_rational<T: EvaluationDomain>(
        &self,
        binary_prec: u32,
    ) -> ExternalFunctionContainer<T> {
        let convert = |coefficient: &Complex<Rational>| {
            T::try_from_complex_float(Complex::new(
                coefficient.re.to_multi_prec_float(binary_prec),
                coefficient.im.to_multi_prec_float(binary_prec),
            ))
            .unwrap()
        };

        ExternalFunctionContainer {
            export_name: self.export_name.clone(),
            symbol: self.symbol,
            tags: self.tags.clone(),
            fixed_args: self.fixed_args.clone(),
            imp: if self.sub_evaluator.is_none() {
                self.fetch_impl_for::<T>()
            } else {
                None
            },
            cache: vec![],
            constant_index: self.constant_index,
            sub_evaluator: self.sub_evaluator.as_ref().map(|evaluator| {
                Box::new(
                    (**evaluator)
                        .clone()
                        .map_coeff_with_prec(&convert, binary_prec),
                )
            }),
        }
    }
}

/// An external function implementation that can be registered on
/// [`crate::atom::EvaluationInfo`] and called by an evaluator.
pub trait ExternalFunction<T>: Fn(&[T]) -> T + Send + Sync + DynClone + Send + Sync {}
dyn_clone::clone_trait_object!(<T> ExternalFunction<T>);
impl<T, F: Clone + Send + Sync + Fn(&[T]) -> T + Send + Sync> ExternalFunction<T> for F {}
