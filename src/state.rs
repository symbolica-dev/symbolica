//! Manage global state and thread-local workspaces.

use byteorder::{ReadBytesExt, WriteBytesExt};
use std::borrow::Cow;
use std::hash::Hash;
use std::io::{Read, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, RwLock, RwLockWriteGuard};
use std::thread::LocalKey;
use std::{
    cell::RefCell,
    collections::hash_map::Entry,
    ops::{Deref, DerefMut},
};

use ahash::{HashMap, HashMapExt};
use append_only_vec::AppendOnlyVec;
use byteorder::LittleEndian;
use once_cell::sync::Lazy;
use smartstring::alias::String;

use crate::atom::{
    DerivativeFunction, NamespacedSymbol, NormalizationFunction, SymbolAttribute, UserData,
};
use crate::domains::finite_field::Zp64;
use crate::poly::PolyVariable;
use crate::printer::PrintFunction;
use crate::wrap_symbol;
use crate::{
    LicenseManager,
    atom::{Atom, Symbol},
    coefficient::Coefficient,
    domains::finite_field::FiniteFieldCore,
};

pub(crate) const SYMBOLICA_MAGIC: u32 = 0x37871367;
pub(crate) const EXPORT_FORMAT_VERSION: u16 = 2;

/// An id for a given finite field in a registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FiniteFieldIndex(pub(crate) usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct VariableListIndex(pub(crate) usize);

/// A mapping from one state to the other. Used during importing
/// for merging a state on file with the current state.
pub struct StateMap {
    pub(crate) symbols: HashMap<u32, Symbol>,
    pub(crate) finite_fields: HashMap<FiniteFieldIndex, FiniteFieldIndex>,
    pub(crate) variables_lists: HashMap<u64, Arc<Vec<PolyVariable>>>,
}

///Trait for anything that contains a StateMap
pub trait HasStateMap {
    fn get_state_map(&self) -> &StateMap;
}

impl HasStateMap for StateMap {
    fn get_state_map(&self) -> &StateMap {
        self
    }
}

impl StateMap {
    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty() && self.finite_fields.is_empty() && self.variables_lists.is_empty()
    }
}

pub(crate) struct SymbolData {
    pub(crate) name: String,
    pub(crate) namespace: Cow<'static, str>,
    pub(crate) file: Cow<'static, str>,
    pub(crate) line: usize,
    pub(crate) custom_normalization: Option<NormalizationFunction>,
    pub(crate) custom_print: Option<PrintFunction>,
    pub(crate) custom_derivative: Option<DerivativeFunction>,
    pub(crate) tags: Vec<std::string::String>,
    pub(crate) user_data: UserData,
}

static STATE: Lazy<RwLock<State>> = Lazy::new(|| RwLock::new(State::new()));
static ID_TO_STR: AppendOnlyVec<(Symbol, SymbolData)> = AppendOnlyVec::new();
static FINITE_FIELDS: AppendOnlyVec<Zp64> = AppendOnlyVec::new();
static VARIABLE_LISTS: AppendOnlyVec<Arc<Vec<PolyVariable>>> = AppendOnlyVec::new();
static SYMBOL_OFFSET: AtomicUsize = AtomicUsize::new(0);

thread_local!(
    /// A thread-local workspace, that stores recyclable atoms.
    static WORKSPACE: Workspace = const { Workspace::new() }
);

/// A global state, that stores mappings from variable and function names to ids.
pub struct State {
    str_to_id: HashMap<String, Symbol>,
}

impl Default for State {
    fn default() -> Self {
        Self::new()
    }
}

impl State {
    pub(crate) const ARG: Symbol =
        Symbol::raw_fn(0, 0, false, false, false, false, false, false, false, false);
    pub(crate) const COEFF: Symbol =
        Symbol::raw_fn(1, 0, false, false, false, false, true, false, false, false);
    pub(crate) const EXP: Symbol =
        Symbol::raw_fn(2, 0, false, false, false, false, false, false, false, false);
    pub(crate) const LOG: Symbol =
        Symbol::raw_fn(3, 0, false, false, false, false, false, false, false, false);
    pub(crate) const SIN: Symbol =
        Symbol::raw_fn(4, 0, false, false, false, false, false, false, false, false);
    pub(crate) const COS: Symbol =
        Symbol::raw_fn(5, 0, false, false, false, false, false, false, false, false);
    pub(crate) const SQRT: Symbol =
        Symbol::raw_fn(6, 0, false, false, false, false, false, false, false, false);
    pub(crate) const CONJ: Symbol =
        Symbol::raw_fn(7, 0, false, false, false, false, false, false, false, false);
    pub(crate) const DERIVATIVE: Symbol =
        Symbol::raw_fn(8, 0, false, false, false, false, false, false, false, false);
    pub(crate) const E: Symbol =
        Symbol::raw_fn(9, 0, false, false, false, false, true, true, false, true);
    pub(crate) const PI: Symbol =
        Symbol::raw_fn(10, 0, false, false, false, false, true, true, false, true);
    pub(crate) const SEP: Symbol =
        Symbol::raw_fn(11, 0, false, false, false, false, true, true, true, true);

    /// The list of built-in symbols.
    pub const BUILTIN_SYMBOL_NAMES: [&'static str; 12] = [
        "arg",
        "coeff",
        "exp",
        "log",
        "sin",
        "cos",
        "sqrt",
        "conj",
        "der",
        Symbol::E_STR,
        Symbol::PI_STR,
        Symbol::SEP_STR,
    ];

    /// The list of built-in symbols.
    pub const BUILTIN_SYMBOLS: [Symbol; 12] = [
        Self::ARG,
        Self::COEFF,
        Self::EXP,
        Self::LOG,
        Self::SIN,
        Self::COS,
        Self::SQRT,
        Self::CONJ,
        Self::DERIVATIVE,
        Self::E,
        Self::PI,
        Self::SEP,
    ];

    pub fn is_builtin_name<S: AsRef<str>>(str: S) -> bool {
        Self::BUILTIN_SYMBOL_NAMES.contains(&str.as_ref())
    }

    fn new() -> State {
        LicenseManager::check();

        let mut state = State {
            str_to_id: HashMap::new(),
        };

        for (name, symbol) in Self::BUILTIN_SYMBOL_NAMES
            .iter()
            .zip(&Self::BUILTIN_SYMBOLS)
        {
            let r = wrap_symbol!(name);

            let index = ID_TO_STR.push((
                *symbol,
                SymbolData {
                    name: r.symbol.clone().into(),
                    file: r.file,
                    namespace: r.namespace,
                    line: r.line,
                    custom_normalization: None,
                    custom_print: None,
                    custom_derivative: None,
                    tags: vec![],
                    user_data: UserData::None,
                },
            ));
            assert_eq!(symbol.get_id() as usize, index);

            state.str_to_id.insert(r.symbol.into(), *symbol);
        }

        #[cfg(test)]
        {
            state.initialize_test();
        }

        state
    }

    /// Get the global state.
    #[inline]
    pub(crate) fn get_global_state() -> &'static RwLock<State> {
        &STATE
    }

    /// Initialize the global state for testing purposes by allocating
    /// variables and functions with the names v0, ..., v29, f0, ..., f29,
    /// that can be used in concurrently run unit tests without interference.
    #[cfg(test)]
    fn initialize_test(&mut self) {
        use crate::atom::SymbolAttribute;

        for i in 0..30 {
            let _ = self.get_symbol(wrap_symbol!(format!("v{}", i)));
        }
        for i in 0..30 {
            let _ = self.get_symbol(wrap_symbol!(format!("f{}", i)));
        }
        for i in 0..5 {
            let _ = self.get_symbol_with_attributes(
                wrap_symbol!(format!("fs{}", i)),
                &[SymbolAttribute::Symmetric],
                None,
                None,
                None,
                vec![],
                None,
            );
        }
        for i in 0..5 {
            let _ = self.get_symbol_with_attributes(
                wrap_symbol!(format!("fc{}", i)),
                &[SymbolAttribute::Cyclesymmetric],
                None,
                None,
                None,
                vec![],
                None,
            );
        }
        for i in 0..5 {
            let _ = self.get_symbol_with_attributes(
                wrap_symbol!(format!("fa{}", i)),
                &[SymbolAttribute::Antisymmetric],
                None,
                None,
                None,
                vec![],
                None,
            );
        }
        for i in 0..5 {
            let _ = self.get_symbol_with_attributes(
                wrap_symbol!(format!("fl{}", i)),
                &[SymbolAttribute::Linear],
                None,
                None,
                None,
                vec![],
                None,
            );
        }
        for i in 0..5 {
            let _ = self.get_symbol_with_attributes(
                wrap_symbol!(format!("fsl{}", i)),
                &[SymbolAttribute::Symmetric, SymbolAttribute::Linear],
                None,
                None,
                None,
                vec![],
                None,
            );
        }
    }

    /// Remove all user-defined symbols from the state. This will invalidate all
    /// currently existing atoms, and hence this function is unsafe.
    ///
    /// Example:
    /// ```
    /// # use symbolica::symbol;
    /// # use symbolica::state::State;
    /// symbol!("f"; Symmetric);
    /// unsafe { State::reset(); }
    /// symbol!("f"; Antisymmetric);
    /// ```
    pub unsafe fn reset() {
        let mut state = STATE.write().unwrap();

        state.str_to_id.clear();
        SYMBOL_OFFSET.store(ID_TO_STR.len(), Ordering::Relaxed);

        let offset = SYMBOL_OFFSET.load(Ordering::Relaxed);

        for (name, symbol) in Self::BUILTIN_SYMBOL_NAMES
            .iter()
            .zip(&Self::BUILTIN_SYMBOLS)
        {
            let r = wrap_symbol!(name);

            let index = ID_TO_STR.push((
                *symbol,
                SymbolData {
                    name: r.symbol.clone().into(),
                    file: r.file,
                    namespace: r.namespace,
                    line: r.line,
                    custom_normalization: None,
                    custom_print: None,
                    custom_derivative: None,
                    tags: vec![],
                    user_data: UserData::None,
                },
            ));
            assert_eq!(symbol.get_id() as usize, index - offset);

            state.str_to_id.insert(r.symbol.into(), *symbol);
        }

        #[cfg(test)]
        {
            state.initialize_test();
        }
    }

    #[inline(always)]
    #[allow(dead_code)]
    pub(crate) unsafe fn symbol_from_id(id: u32) -> Symbol {
        if ID_TO_STR.len() == 0 {
            let _ = *STATE; // initialize the state
        }

        ID_TO_STR[id as usize].0
    }

    /// Iterate over all defined symbols.
    pub fn symbol_iter() -> impl Iterator<Item = (Symbol, &'static str)> {
        if ID_TO_STR.len() == 0 {
            let _ = *STATE; // initialize the state
        }

        ID_TO_STR
            .iter()
            .skip(SYMBOL_OFFSET.load(Ordering::Relaxed))
            .map(|s| (s.0, s.1.name.as_str()))
    }

    /// Returns `true` iff this identifier is defined by Symbolica.
    pub(crate) fn is_builtin(id: Symbol) -> bool {
        id.get_id() < Self::BUILTIN_SYMBOL_NAMES.len() as u32
    }

    /// Get the symbol for a certain name if the name is already registered,
    pub(crate) fn fetch_symbol(&self, name: &str) -> Option<Symbol> {
        self.str_to_id.get(name).cloned()
    }

    /// Get the next symbol index that will be assigned.
    pub(crate) fn get_next_symbol_index(&self) -> u32 {
        let offset = SYMBOL_OFFSET.load(Ordering::Relaxed);
        (ID_TO_STR.len() - offset) as u32
    }

    /// Get the wildcard level of a symbol name.
    pub(crate) fn get_wildcard_level(str: &str) -> u8 {
        let mut wildcard_level = 0;
        for x in str.chars().rev() {
            if x != '_' {
                break;
            }
            wildcard_level += 1;
        }
        wildcard_level
    }

    /// Get the symbol for a certain name if the name is already registered,
    /// else register it and return a new symbol without attributes.
    pub(crate) fn get_symbol(&mut self, name: NamespacedSymbol) -> Result<Symbol, String> {
        match self.str_to_id.entry(name.symbol.into()) {
            Entry::Occupied(o) => Ok(*o.get()),
            Entry::Vacant(v) => {
                let offset = SYMBOL_OFFSET.load(Ordering::Relaxed);
                if ID_TO_STR.len() - offset == u32::MAX as usize - 1 {
                    panic!("Too many variables defined");
                }

                let wildcard_level = State::get_wildcard_level(v.key());

                // there is no synchronization issue since only one thread can insert at a time
                // as the state itself is behind a mutex
                let id = ID_TO_STR.len() - offset;
                let new_symbol = Symbol::raw_var(id as u32, wildcard_level);
                let id_ret = ID_TO_STR.push((
                    new_symbol,
                    SymbolData {
                        name: v.key().clone(),
                        file: name.file,
                        namespace: name.namespace,
                        line: name.line,
                        custom_normalization: None,
                        custom_print: None,
                        custom_derivative: None,
                        tags: vec![],
                        user_data: UserData::None,
                    },
                )) - offset;
                assert_eq!(id, id_ret);

                v.insert(new_symbol);
                Ok(new_symbol)
            }
        }
    }

    pub(crate) fn get_state_mut() -> RwLockWriteGuard<'static, State> {
        STATE.write().unwrap()
    }

    /// Register a new symbol with the given attributes and a specific function
    /// that is called after normalization of the arguments. This function cannot
    /// be exported, and therefore before importing a state, symbols with special
    /// normalization functions must be registered explicitly.
    ///
    /// If the symbol already exists, an error is returned.
    pub(crate) fn get_symbol_with_attributes(
        &mut self,
        name: NamespacedSymbol,
        attributes: &[SymbolAttribute],
        normalization_function: Option<NormalizationFunction>,
        print_function: Option<PrintFunction>,
        derivative_function: Option<DerivativeFunction>,
        tags: Vec<std::string::String>,
        user_data: Option<UserData>,
    ) -> Result<Symbol, String> {
        match self.str_to_id.entry(name.symbol.into()) {
            Entry::Occupied(o) => {
                let r = *o.get();

                let new_id = Symbol::raw_fn(
                    r.get_id(),
                    r.get_wildcard_level(),
                    attributes.contains(&SymbolAttribute::Symmetric),
                    attributes.contains(&SymbolAttribute::Antisymmetric),
                    attributes.contains(&SymbolAttribute::Cyclesymmetric),
                    attributes.contains(&SymbolAttribute::Linear),
                    attributes.contains(&SymbolAttribute::Scalar),
                    attributes.contains(&SymbolAttribute::Real),
                    attributes.contains(&SymbolAttribute::Integer),
                    attributes.contains(&SymbolAttribute::Positive),
                );

                if r == new_id
                    && normalization_function.is_none()
                    && print_function.is_none()
                    && derivative_function.is_none()
                    && tags == r.get_tags()
                    && user_data.as_ref().unwrap_or(&UserData::None)
                        == &ID_TO_STR[r.get_id() as usize].1.user_data
                {
                    Ok(r)
                } else {
                    let data = &ID_TO_STR[r.get_id() as usize].1;

                    let mut diff_attr = String::new();
                    if r.is_antisymmetric() != new_id.is_antisymmetric() {
                        diff_attr.push_str(&format!(
                            "\tAntisymmetric: {} vs {}\n",
                            r.is_antisymmetric(),
                            new_id.is_antisymmetric()
                        ));
                    }
                    if r.is_symmetric() != new_id.is_symmetric() {
                        diff_attr.push_str(&format!(
                            "\tSymmetric: {} vs {}\n",
                            r.is_symmetric(),
                            new_id.is_symmetric()
                        ));
                    }
                    if r.is_cyclesymmetric() != new_id.is_cyclesymmetric() {
                        diff_attr.push_str(&format!(
                            "\tCyclesymmetric: {} vs {}\n",
                            r.is_cyclesymmetric(),
                            new_id.is_cyclesymmetric()
                        ));
                    }
                    if r.is_linear() != new_id.is_linear() {
                        diff_attr.push_str(&format!(
                            "\tLinear: {} vs {}\n",
                            r.is_linear(),
                            new_id.is_linear()
                        ));
                    }
                    if r.is_scalar() != new_id.is_scalar() {
                        diff_attr.push_str(&format!(
                            "\tScalar: {} vs {}\n",
                            r.is_scalar(),
                            new_id.is_scalar()
                        ));
                    }
                    if r.is_real() != new_id.is_real() {
                        diff_attr.push_str(&format!(
                            "\tReal: {} vs {}\n",
                            r.is_real(),
                            new_id.is_real()
                        ));
                    }
                    if r.is_integer() != new_id.is_integer() {
                        diff_attr.push_str(&format!(
                            "\tInteger: {} vs {}\n",
                            r.is_integer(),
                            new_id.is_integer()
                        ));
                    }
                    if r.is_positive() != new_id.is_positive() {
                        diff_attr.push_str(&format!(
                            "\tPositive: {} vs {}\n",
                            r.is_positive(),
                            new_id.is_positive()
                        ));
                    }

                    if tags != r.get_tags() {
                        diff_attr.push_str(&format!("\tTags: {:?} vs {:?}\n", r.get_tags(), tags));
                    }

                    if normalization_function.is_some() {
                        diff_attr.push_str("\tNew normalization function specified.\n");
                    }
                    if print_function.is_some() {
                        diff_attr.push_str("\tNew print function specified.\n");
                    }
                    if derivative_function.is_some() {
                        diff_attr.push_str("\tNew derivative function specified.\n");
                    }

                    if user_data.as_ref().unwrap_or(&UserData::None) != &data.user_data {
                        diff_attr.push_str(&format!(
                            "\tNew user data specified: {:?} vs {:?}\n",
                            data.user_data, user_data
                        ));
                    }

                    if data.file.is_empty() {
                        Err(format!(
                            "Symbol {} redefined with new attributes:\n{}",
                            data.name, diff_attr
                        )
                        .into())
                    } else {
                        Err(format!("Symbol {} redefined with new attributes: {}The first definition occurred here: {}:{}.", data.name, diff_attr, data.file, data.line).into())
                    }
                }
            }
            Entry::Vacant(v) => {
                let offset = SYMBOL_OFFSET.load(Ordering::Relaxed);
                if ID_TO_STR.len() - offset == u32::MAX as usize - 1 {
                    panic!("Too many variables defined");
                }

                // there is no synchronization issue since only one thread can insert at a time
                // as the state itself is behind a mutex
                let id = ID_TO_STR.len() - offset;

                let wildcard_level = State::get_wildcard_level(v.key());

                let new_symbol = Symbol::raw_fn(
                    id as u32,
                    wildcard_level,
                    attributes.contains(&SymbolAttribute::Symmetric),
                    attributes.contains(&SymbolAttribute::Antisymmetric),
                    attributes.contains(&SymbolAttribute::Cyclesymmetric),
                    attributes.contains(&SymbolAttribute::Linear),
                    attributes.contains(&SymbolAttribute::Scalar),
                    attributes.contains(&SymbolAttribute::Real),
                    attributes.contains(&SymbolAttribute::Integer),
                    attributes.contains(&SymbolAttribute::Positive),
                );

                let id_ret = ID_TO_STR.push((
                    new_symbol,
                    SymbolData {
                        name: v.key().clone(),
                        file: name.file,
                        namespace: name.namespace,
                        line: name.line,
                        custom_normalization: normalization_function,
                        custom_print: print_function,
                        custom_derivative: derivative_function,
                        tags,
                        user_data: user_data.unwrap_or(UserData::None),
                    },
                )) - offset;
                assert_eq!(id, id_ret);

                v.insert(new_symbol);

                Ok(new_symbol)
            }
        }
    }

    /// Get the name for a given symbol.
    #[inline]
    pub(crate) fn get_name(id: Symbol) -> &'static str {
        if ID_TO_STR.len() == 0 {
            let _ = *STATE; // initialize the state
        }

        &ID_TO_STR[id.get_id() as usize + SYMBOL_OFFSET.load(Ordering::Relaxed)]
            .1
            .name
    }

    /// Get the name for a given symbol.
    #[inline]
    pub(crate) fn get_symbol_namespace(id: Symbol) -> &'static str {
        if ID_TO_STR.len() == 0 {
            let _ = *STATE; // initialize the state
        }

        ID_TO_STR[id.get_id() as usize + SYMBOL_OFFSET.load(Ordering::Relaxed)]
            .1
            .namespace
            .as_ref()
    }

    /// Get the name for a given symbol.
    #[inline]
    pub(crate) fn get_symbol_data(id: Symbol) -> &'static SymbolData {
        if ID_TO_STR.len() == 0 {
            let _ = *STATE; // initialize the state
        }

        &ID_TO_STR[id.get_id() as usize + SYMBOL_OFFSET.load(Ordering::Relaxed)].1
    }

    /// Get the user-specified normalization function for the symbol.
    #[inline]
    pub(crate) fn get_normalization_function(id: Symbol) -> Option<&'static NormalizationFunction> {
        if ID_TO_STR.len() == 0 {
            let _ = *STATE; // initialize the state
        }

        ID_TO_STR[id.get_id() as usize + SYMBOL_OFFSET.load(Ordering::Relaxed)]
            .1
            .custom_normalization
            .as_ref()
    }

    pub(crate) fn get_finite_field(fi: FiniteFieldIndex) -> &'static Zp64 {
        &FINITE_FIELDS[fi.0]
    }

    pub(crate) fn get_or_insert_finite_field(f: Zp64) -> FiniteFieldIndex {
        STATE.write().unwrap().get_or_insert_finite_field_impl(f)
    }

    pub(crate) fn get_or_insert_finite_field_impl(&mut self, f: Zp64) -> FiniteFieldIndex {
        for (i, f2) in FINITE_FIELDS.iter().enumerate() {
            if f.get_prime() == f2.get_prime() {
                return FiniteFieldIndex(i);
            }
        }

        let index = FINITE_FIELDS.push(f);
        FiniteFieldIndex(index)
    }

    pub(crate) fn get_variable_list(fi: VariableListIndex) -> Arc<Vec<PolyVariable>> {
        VARIABLE_LISTS[fi.0].clone()
    }

    pub(crate) fn get_or_insert_variable_list(f: Arc<Vec<PolyVariable>>) -> VariableListIndex {
        STATE.write().unwrap().get_or_insert_variable_list_impl(f)
    }

    pub(crate) fn get_or_insert_variable_list_impl(
        &mut self,
        f: Arc<Vec<PolyVariable>>,
    ) -> VariableListIndex {
        for (i, f2) in VARIABLE_LISTS.iter().enumerate() {
            if f2 == &f {
                return VariableListIndex(i);
            }
        }

        let index = VARIABLE_LISTS.push(f);
        VariableListIndex(index)
    }

    /// Write the state to a binary stream.
    #[inline(always)]
    pub fn export<W: Write>(dest: &mut W) -> Result<(), std::io::Error> {
        if ID_TO_STR.len() == 0 {
            let _ = *STATE; // initialize the state
        }

        dest.write_u32::<LittleEndian>(SYMBOLICA_MAGIC)?;
        dest.write_u16::<LittleEndian>(EXPORT_FORMAT_VERSION)?;

        dest.write_u64::<LittleEndian>(
            ID_TO_STR.len() as u64 - SYMBOL_OFFSET.load(Ordering::Relaxed) as u64,
        )?;

        for (s, n) in State::symbol_iter() {
            dest.write_u32::<LittleEndian>(n.len() as u32)?;
            dest.write_all(n.as_bytes())?;

            let namespace = s.get_namespace();
            dest.write_u32::<LittleEndian>(namespace.len() as u32)?;
            dest.write_all(namespace.as_bytes())?;

            let (flags, extra_flags) = s.encode_flags();
            dest.write_u8(flags)?;
            dest.write_u32::<LittleEndian>(extra_flags)?;

            dest.write_u16::<LittleEndian>(s.get_tags().len() as u16)?;
            for t in s.get_tags() {
                dest.write_u32::<LittleEndian>(t.len() as u32)?;
                dest.write_all(t.as_bytes())?;
            }

            s.get_data().write(dest)?;
        }

        dest.write_u64::<LittleEndian>(FINITE_FIELDS.len() as u64)?;
        for x in FINITE_FIELDS.iter() {
            dest.write_u64::<LittleEndian>(x.get_prime())?;
        }

        dest.write_u64::<LittleEndian>(VARIABLE_LISTS.len() as u64)?;
        for x in VARIABLE_LISTS.iter() {
            dest.write_u64::<LittleEndian>(x.len() as u64)?;
            for y in x.iter() {
                match y {
                    PolyVariable::Symbol(s) => {
                        dest.write_u8(0)?;
                        dest.write_u32::<LittleEndian>(s.get_id())?;
                    }
                    PolyVariable::Temporary(u) => {
                        dest.write_u8(1)?;
                        dest.write_u64::<LittleEndian>(*u as u64)?;
                    }
                    PolyVariable::Function(v, t) => {
                        dest.write_u8(2)?;
                        dest.write_u32::<LittleEndian>(v.get_id())?;
                        t.as_view().write(dest.by_ref())?;
                    }
                    PolyVariable::Power(t) => {
                        dest.write_u8(3)?;
                        t.as_view().write(dest.by_ref())?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Import a state, merging it with the current state.
    /// Upon a conflict, i.e. when a symbol with the same name but different attributes is
    /// encountered, `conflict_fn` is called with the conflicting name as argument which
    /// should yield a new name for the symbol.
    #[inline(always)]
    pub fn import<R: Read>(
        source: &mut R,
        conflict_fn: Option<Box<dyn Fn(&str) -> String>>,
    ) -> Result<StateMap, std::io::Error> {
        let magic = source.read_u32::<LittleEndian>()?;

        if magic != SYMBOLICA_MAGIC {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid magic number: the file is not exported from Symbolica",
            ));
        }

        let version = source.read_u16::<LittleEndian>()?;
        if version != EXPORT_FORMAT_VERSION {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Invalid export format version: expected {} but got {}",
                    EXPORT_FORMAT_VERSION, version
                ),
            ));
        }

        let mut state_map = StateMap {
            symbols: HashMap::default(),
            finite_fields: HashMap::default(),
            variables_lists: HashMap::default(),
        };

        let n_symbols = source.read_u64::<LittleEndian>()?;
        let mut attributes = vec![];
        for x in 0..n_symbols {
            let l = source.read_u32::<LittleEndian>()?;
            let mut v = vec![0; l as usize];
            source.read_exact(&mut v)?;

            let mut str: String = std::string::String::from_utf8(v)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
                .into();

            let l = source.read_u32::<LittleEndian>()?;
            let mut v = vec![0; l as usize];
            source.read_exact(&mut v)?;

            let namespace: String = std::string::String::from_utf8(v)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
                .into();

            let flags = source.read_u8()?;
            let extra_flags = source.read_u32::<LittleEndian>()?;

            let s = Symbol::decode_flags(0, flags, extra_flags);

            let mut tags = vec![];
            let num_tags = source.read_u16::<LittleEndian>()?;

            for _ in 0..num_tags {
                let l = source.read_u32::<LittleEndian>()?;
                let mut v = vec![0; l as usize];
                source.read_exact(&mut v)?;

                let tag: String = std::string::String::from_utf8(v)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
                    .into();

                tags.push(tag);
            }

            let extra_data = UserData::read(&mut *source)?;

            attributes.clear();
            if s.is_antisymmetric() {
                attributes.push(SymbolAttribute::Antisymmetric);
            }
            if s.is_symmetric() {
                attributes.push(SymbolAttribute::Symmetric);
            }
            if s.is_cyclesymmetric() {
                attributes.push(SymbolAttribute::Cyclesymmetric);
            }
            if s.is_linear() {
                attributes.push(SymbolAttribute::Linear);
            }
            if s.is_scalar() {
                attributes.push(SymbolAttribute::Scalar);
            }
            if s.is_real() {
                attributes.push(SymbolAttribute::Real);
            }
            if s.is_integer() {
                attributes.push(SymbolAttribute::Integer);
            }
            if s.is_positive() {
                attributes.push(SymbolAttribute::Positive);
            }

            loop {
                match Symbol::new(NamespacedSymbol {
                    symbol: str.to_string().into(),
                    namespace: namespace.to_string().into(),
                    file: "".into(),
                    line: 0,
                })
                .with_attributes(attributes.clone())
                .with_tags(tags.clone())
                .with_user_data(extra_data.clone())
                .build()
                {
                    Ok(id) => {
                        if x as u32 != id.get_id() {
                            state_map.symbols.insert(x as u32, id);
                        }
                        break;
                    }
                    Err(e) => {
                        if let Some(f) = &conflict_fn {
                            let new_name = f(&str);

                            let mut new_wildcard_level = 0;
                            for x in new_name.chars().rev() {
                                if x != '_' {
                                    break;
                                }
                                new_wildcard_level += 1;
                            }

                            if s.get_wildcard_level() == new_wildcard_level {
                                str = new_name;
                            }
                        } else {
                            return Err(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                format!("Symbol conflict: {e}"),
                            ));
                        }
                    }
                }
            }
        }

        let n_finite_fields = source.read_u64::<LittleEndian>()?;
        for x in 0..n_finite_fields {
            let prime = source.read_u64::<LittleEndian>()?;
            let id = State::get_or_insert_finite_field(Zp64::new(prime));
            if x != id.0 as u64 {
                state_map
                    .finite_fields
                    .insert(FiniteFieldIndex(x as usize), id);
            }
        }

        let n_variable_lists = source.read_u64::<LittleEndian>()?;
        for x in 0..n_variable_lists {
            let n_vars = source.read_u64::<LittleEndian>()?;
            let mut variables = vec![];
            for _ in 0..n_vars {
                match source.read_u8()? {
                    0 => {
                        let id = source.read_u32::<LittleEndian>()?;
                        if let Some(new_id) = state_map.symbols.get(&id) {
                            variables.push(PolyVariable::Symbol(*new_id));
                        } else {
                            variables.push(PolyVariable::Symbol(ID_TO_STR[id as usize].0))
                        }
                    }
                    1 => {
                        let u = source.read_u64::<LittleEndian>()?;
                        variables.push(PolyVariable::Temporary(u as usize))
                    }
                    2 => {
                        let id = source.read_u32::<LittleEndian>()?;
                        let symb = if let Some(new_id) = state_map.symbols.get(&id) {
                            *new_id
                        } else {
                            ID_TO_STR[id as usize].0
                        };

                        let mut f = Atom::new();
                        f.read(&mut *source)?;

                        let f_r = f.as_view().rename(&state_map);
                        variables.push(PolyVariable::Function(symb, f_r));
                    }
                    3 => {
                        let mut f = Atom::new();
                        f.read(&mut *source)?;

                        let f_r = f.as_view().rename(&state_map);
                        variables.push(PolyVariable::Power(f_r));
                    }
                    _ => {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "Invalid variable type",
                        ));
                    }
                }
            }

            // see if variables are seen before
            let vars = Arc::new(variables);
            let new_id = State::get_or_insert_variable_list(vars.clone());
            if x != new_id.0 as u64 {
                state_map.variables_lists.insert(x, vars);
            }
        }

        Ok(state_map)
    }
}

/// A workspace that stores recyclable atoms. Upon dropping, the atoms automatically returned to a
/// thread-local workspace (which may be a different one than the one it was created by).
pub struct Workspace {
    atom_buffer: RefCell<Vec<Atom>>,
}

impl Workspace {
    const ATOM_BUFFER_MAX: usize = 30;
    const ATOM_CACHE_SIZE_MAX: usize = 20_000_000;

    /// Create a new workspace.
    const fn new() -> Self {
        Workspace {
            atom_buffer: RefCell::new(Vec::new()),
        }
    }

    /// Get a thread-local workspace.
    #[inline]
    pub fn get_local() -> &'static LocalKey<Workspace> {
        LicenseManager::check();

        &WORKSPACE
    }

    /// Return a recycled atom from this workspace. The atom may have the same value as before.
    #[inline]
    pub fn new_atom(&self) -> RecycledAtom {
        if let Ok(mut a) = self.atom_buffer.try_borrow_mut() {
            if let Some(b) = a.pop() {
                b.into()
            } else {
                Atom::default().into()
            }
        } else {
            Atom::default().into() // very rare
        }
    }

    /// Create a new variable from a recycled atom from this workspace.
    #[inline]
    pub fn new_var(&self, id: Symbol) -> RecycledAtom {
        let mut owned = self.new_atom();
        owned.to_var(id);
        owned
    }

    /// Create a new number from a recycled atom from this workspace.
    #[inline]
    pub fn new_num<T: Into<Coefficient>>(&self, num: T) -> RecycledAtom {
        let mut owned = self.new_atom();
        owned.to_num(num.into());
        owned
    }

    pub fn return_atom(&self, atom: Atom) {
        if let Ok(mut a) = self.atom_buffer.try_borrow_mut() {
            a.push(atom);
        }
    }
}

/// A wrapper around [Atom] that stores the underlying buffer
/// in a thread-local storage cache when dropped.
#[derive(PartialEq, Eq, Debug, Hash, Clone)]
pub struct RecycledAtom(Atom);

impl From<Atom> for RecycledAtom {
    fn from(a: Atom) -> Self {
        RecycledAtom(a)
    }
}

impl std::fmt::Display for RecycledAtom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl Default for RecycledAtom {
    fn default() -> Self {
        Self::new()
    }
}

impl RecycledAtom {
    /// Get a recycled atom from a thread-local workspace.
    #[inline]
    pub fn new() -> RecycledAtom {
        Workspace::get_local().with(|ws| ws.new_atom())
    }

    /// Wrap an atom so that it gets recycled upon dropping.
    pub fn wrap(atom: Atom) -> RecycledAtom {
        RecycledAtom(atom)
    }

    #[inline]
    pub fn new_var(id: Symbol) -> RecycledAtom {
        let mut owned = Self::new();
        owned.to_var(id);
        owned
    }

    /// Create a new number from a recycled atom from this workspace.
    #[inline]
    pub fn new_num<T: Into<Coefficient>>(num: T) -> RecycledAtom {
        let mut owned = Self::new();
        owned.to_num(num.into());
        owned
    }

    /// Yield the atom, which will now no longer be recycled upon dropping.
    pub fn into_inner(mut self) -> Atom {
        std::mem::replace(&mut self.0, Atom::Zero)
    }
}

impl Deref for RecycledAtom {
    type Target = Atom;

    fn deref(&self) -> &Atom {
        &self.0
    }
}

impl DerefMut for RecycledAtom {
    fn deref_mut(&mut self) -> &mut Atom {
        &mut self.0
    }
}

impl AsRef<Atom> for RecycledAtom {
    fn as_ref(&self) -> &Atom {
        self.deref()
    }
}

impl Drop for RecycledAtom {
    #[inline]
    fn drop(&mut self) {
        if let Atom::Zero = self.0 {
            return;
        }

        if self.0.get_capacity() > Workspace::ATOM_CACHE_SIZE_MAX {
            return;
        }

        let _ = WORKSPACE.try_with(
            #[inline(always)]
            |ws| {
                if let Ok(mut a) = ws.atom_buffer.try_borrow_mut()
                    && a.len() < Workspace::ATOM_BUFFER_MAX
                {
                    a.push(std::mem::replace(&mut self.0, Atom::Zero));
                }
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        atom::{Atom, AtomCore, AtomView, Symbol},
        parse, symbol,
    };

    use super::State;

    #[test]
    fn state_export_import() {
        let mut export = vec![];
        State::export(&mut export).unwrap();

        let i = State::import(&mut export.as_slice(), None).unwrap();
        assert!(i.is_empty());
    }

    #[test]
    fn export_symbol_data() {
        let s = symbol!(
            "symbolica::symbol_data::a",
            data = crate::state::UserData::Atom(parse!("z"))
        );

        let s1 = s.to_atom();
        let mut export = vec![];
        s1.export(&mut export).unwrap();

        let a = Atom::import(&mut export.as_slice(), None)
            .unwrap()
            .get_symbol()
            .unwrap();
        assert_eq!(a.get_data(), &crate::state::UserData::Atom(parse!("z")));
    }

    #[test]
    fn custom_normalization() {
        let _real_log = symbol!(
            "custom_normalization_real_log",
            norm = |input, out| {
                if let AtomView::Fun(f) = input {
                    if f.get_nargs() == 1 {
                        let arg = f.iter().next().unwrap();
                        if let AtomView::Fun(f2) = arg {
                            if f2.get_symbol() == Symbol::EXP && f2.get_nargs() == 1 {
                                out.set_from_view(&f2.iter().next().unwrap());
                            }
                        }
                    }
                }
            }
        );

        let e = parse!("custom_normalization_real_log(exp(x))");
        assert_eq!(e, parse!("x"));
    }
}
