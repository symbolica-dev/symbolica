use super::*;

pub struct ExternalFunctionContainer<T> {
    pub(super) export_name: String,
    pub(super) symbol: Symbol,
    pub(super) tags: Vec<Atom>,
    pub(super) imp: Option<Box<dyn ExternalFunction<T>>>,
    pub(super) cache: Vec<T>,
    pub(super) constant_index: Option<usize>,
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
            &self.constant_index,
        )
            .serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: EvaluationDomain> serde::Deserialize<'de> for ExternalFunctionContainer<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (export_name, symbol, tags, constant_index): (
            String,
            Symbol,
            Vec<String>,
            Option<usize>,
        ) = serde::Deserialize::deserialize(deserializer)?;

        let mut external = Self {
            export_name,
            symbol,
            tags: tags.iter().map(|s| crate::parse!(s)).collect(),
            imp: None,
            cache: vec![],
            constant_index,
        };
        external.imp = external.fetch_impl_for::<T>();
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
        bincode::Encode::encode(&self.constant_index, encoder)
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
        let constant_index: Option<usize> = bincode::Decode::decode(decoder)?;

        let mut external = Self {
            export_name,
            symbol,
            tags: tags.iter().map(|s| crate::parse!(s)).collect(),
            imp: None,
            cache: vec![],
            constant_index,
        };
        external.imp = external.fetch_impl_for::<T>();

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
        <Self as bincode::Decode<Context>>::decode(decoder)
    }
}

impl<T> Clone for ExternalFunctionContainer<T> {
    fn clone(&self) -> Self {
        Self {
            export_name: self.export_name.clone(),
            symbol: self.symbol.clone(),
            tags: self.tags.clone(),
            imp: self.imp.clone(),
            cache: vec![],
            constant_index: self.constant_index,
        }
    }
}

impl<T> std::fmt::Debug for ExternalFunctionContainer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternalFunctionContainer")
            .field("export_name", &self.export_name)
            .field("eval_name", &self.symbol)
            .field("tags", &self.tags)
            .field("imp", &self.imp.is_some())
            .field("cache_len", &self.cache.len())
            .field("constant_index", &self.constant_index)
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
    }
}

impl<T> Eq for ExternalFunctionContainer<T> {}

impl<T> ExternalFunctionContainer<T> {
    pub(super) fn new(symbol: Symbol, tags: Vec<Atom>) -> Self {
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
            imp: None,
            cache: vec![],
            constant_index: None,
        }
    }

    pub(super) fn export_name(&self) -> &str {
        &self.export_name
    }

    pub(super) fn tag_views(&self) -> Vec<AtomView<'_>> {
        self.tags.iter().map(|x| x.as_view()).collect()
    }

    pub(super) fn map<T2: EvaluationDomain>(&self) -> ExternalFunctionContainer<T2> {
        ExternalFunctionContainer {
            export_name: self.export_name.clone(),
            symbol: self.symbol,
            tags: self.tags.clone(),
            imp: self.fetch_impl_for::<T2>(),
            cache: vec![],
            constant_index: self.constant_index,
        }
    }

    pub(super) fn fetch_impl_for<T2: EvaluationDomain>(
        &self,
    ) -> Option<Box<dyn ExternalFunction<T2>>> {
        let info = self.symbol.get_evaluation_info()?;
        let tags = self.tag_views();
        T2::resolve_function(&tags, info)
    }

    pub(super) fn cpp(&self) -> Option<&str> {
        self.symbol.get_evaluation_info()?.get_cpp()
    }
}

impl<T> std::fmt::Display for ExternalFunctionContainer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.export_name())
    }
}

/// An optimized evaluator for expressions that can evaluate expressions with parameters.
/// The evaluator can be called directly using [Self::evaluate] or it can be exported
/// to high-performance C++ code using [Self::export_cpp].
///
/// To call the evaluator with external functions, use [Self::with_external_functions] to
/// register implementation for them.
/// An external function that can be called by an evaluator.
pub trait ExternalFunction<T>: Fn(&[T]) -> T + Send + Sync + DynClone + Send + Sync {}
dyn_clone::clone_trait_object!(<T> ExternalFunction<T>);
impl<T, F: Clone + Send + Sync + Fn(&[T]) -> T + Send + Sync> ExternalFunction<T> for F {}
