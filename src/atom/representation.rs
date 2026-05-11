//! Low-level representation of expressions.

use ahash::HashMap;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use bytes::{Buf, BufMut};
use smartstring::alias::String;
use std::{
    borrow::Borrow,
    cmp::Ordering,
    hash::Hash,
    io::{Read, Write},
    sync::Arc,
};

use crate::{
    alias::{
        AliasHandle, collect_alias_handles_in, collect_alias_handles_with_dependencies,
        get_alias_handle, register_alias_atom,
    },
    atom::{UserData, UserDataKey},
    coefficient::{Coefficient, CoefficientView},
    state::{State, StateMap, Workspace},
};

use super::{
    Atom, AtomView, SliceType, Symbol,
    coefficient::{PackedRationalNumberReader, PackedRationalNumberWriter},
};

const NUM_ID: u8 = 1;
const VAR_ID: u8 = 2;
const FUN_ID: u8 = 3;
const MUL_ID: u8 = 4;
const ADD_ID: u8 = 5;
const POW_ID: u8 = 6;
const ALIAS_ID: u8 = 7;
const TYPE_MASK: u8 = 0b00000_111;
const NOT_NORMALIZED: u8 = 0b10000_000;
const HAS_ALIAS_FLAG: u8 = 0b01000_000;
const SYM_SYMMETRIC_FLAG: u8 = 0b00100_000;
const SYM_ANTISYMMETRIC_FLAG: u8 = 0b00010_000;
/// Coded as symmetric | antisymmetric
const SYM_CYCLESYMMETRIC_FLAG: u8 = 0b00110_000;
const SYM_SCALAR_FLAG: u8 = 0b00001_000;
const SYM_EXTRA_REAL_FLAG: u32 = 0b01;
const SYM_EXTRA_INTEGER_FLAG: u32 = 0b10;
const SYM_EXTRA_POSITIVE_FLAG: u32 = 0b100;
const SYM_EXTRA_LINEAR_FLAG: u32 = 0b100_000;
const SYM_EXTRA_WILDCARD_LEVEL_MASK: u32 = 0b11_000;
const SYM_EXTRA_WILDCARD_LEVEL_1: u32 = 0b01_000;
const SYM_EXTRA_WILDCARD_LEVEL_2: u32 = 0b10_000;
const SYM_EXTRA_WILDCARD_LEVEL_3: u32 = 0b11_000;

const MUL_HAS_COEFF_FLAG: u8 = 0b01000000;
const MUL_HAS_ALIAS_FLAG: u8 = 0b00100000;
const ALIAS_OPAQUE_FLAG: u8 = 0b00001_000;
pub(crate) const ALIAS_EXPORT_SECTION_MAGIC: u64 = 0xA11A_5ECA_1100_0001;

const ZERO_DATA: [u8; 3] = [NUM_ID, 1, 0];
static NO_ALIASES: Vec<Arc<AliasHandle>> = Vec::new();

fn merge_aliases(dst: &mut Vec<Arc<AliasHandle>>, src: &[Arc<AliasHandle>]) {
    if src.is_empty() {
        return;
    }

    dst.extend(src.iter().cloned());
    dst.sort_by_key(|handle| handle.token());
    dst.dedup_by_key(|handle| handle.token());
}

fn aliases_from_view(view: AtomView<'_>) -> Vec<Arc<AliasHandle>> {
    collect_alias_handles_in(view)
}

fn aliases_from_raw_data(data: &[u8]) -> Vec<Arc<AliasHandle>> {
    let mut handles = Vec::new();
    collect_aliases_from_raw(data, &mut handles);
    handles.sort_by_key(|handle| handle.token());
    handles.dedup_by_key(|handle| handle.token());
    handles
}

fn collect_aliases_from_raw(data: &[u8], handles: &mut Vec<Arc<AliasHandle>>) {
    match data[0] & TYPE_MASK {
        NUM_ID | VAR_ID => {}
        ALIAS_ID => {
            let token = data[1..].get_frac_u64().0 as usize;
            if let Some(handle) = get_alias_handle(token) {
                handles.push(handle);
            }
        }
        FUN_ID => {
            if data[0] & HAS_ALIAS_FLAG == 0 {
                return;
            }
            let mut c = &data[1 + 4..];
            let n_args;
            (_, n_args, c) = c.get_frac_u64();
            collect_aliases_from_raw_list(c, n_args as usize, handles);
        }
        POW_ID => collect_aliases_from_raw_list(&data[1..], 2, handles),
        MUL_ID => {
            if data[0] & MUL_HAS_ALIAS_FLAG == 0 {
                return;
            }
            let mut c = &data[1 + 4..];
            let n_args;
            (n_args, _, c) = c.get_frac_u64();
            collect_aliases_from_raw_list(c, n_args as usize, handles);
        }
        ADD_ID => {
            if data[0] & HAS_ALIAS_FLAG == 0 {
                return;
            }
            let mut c = &data[1..];
            let n_args;
            (n_args, _, c) = c.get_frac_u64();
            collect_aliases_from_raw_list(c, n_args as usize, handles);
        }
        x => unreachable!("Bad id {}", x),
    }
}

fn collect_aliases_from_raw_list(
    mut data: &[u8],
    n_args: usize,
    handles: &mut Vec<Arc<AliasHandle>>,
) {
    for _ in 0..n_args {
        collect_aliases_from_raw(data, handles);
        data = skip_raw_atom(data);
    }
}

fn skip_raw_atom(mut data: &[u8]) -> &[u8] {
    let atom_type = data.get_u8() & TYPE_MASK;
    match atom_type {
        NUM_ID | VAR_ID | ALIAS_ID => data.skip_rational(),
        FUN_ID | MUL_ID => {
            let n_size = data.get_u32_le();
            data.advance(n_size as usize);
            data
        }
        ADD_ID => {
            let (_, size, data) = data.get_frac_u64();
            &data[size as usize..]
        }
        POW_ID => skip_raw_atom(skip_raw_atom(data)),
        x => unreachable!("Bad id {}", x),
    }
}

#[inline(always)]
fn alias_flag_for(view: AtomView<'_>) -> u8 {
    if view.has_alias() { HAS_ALIAS_FLAG } else { 0 }
}

#[inline(always)]
fn mul_alias_flag_for(view: AtomView<'_>) -> u8 {
    if view.has_alias() {
        MUL_HAS_ALIAS_FLAG
    } else {
        0
    }
}

pub(crate) fn read_raw_atom<R: Read>(source: &mut R) -> Result<RawAtom, std::io::Error> {
    // should also set whether rat poly coefficient needs to be converted
    let mut flags_buf = [0; 1];
    let mut size_buf = [0; 8];

    source.read_exact(&mut flags_buf)?;
    source.read_exact(&mut size_buf)?;

    let n_size = u64::from_le_bytes(size_buf);
    let mut dest = RawAtom::new();
    dest.resize(n_size as usize, 0);
    source.read_exact(&mut dest)?;
    Ok(dest)
}

pub(crate) fn remap_aliases_in_raw_atom(
    raw: RawAtom,
    aliases: &HashMap<usize, Arc<AliasHandle>>,
) -> Result<RawAtom, std::io::Error> {
    let mut out = RawAtom::new();
    remap_aliases_in_raw_data(&raw, aliases, &mut out)?;
    let handles = aliases_from_raw_data(&out);
    out.sync_aliases_from(&handles);
    Ok(out)
}

fn remap_aliases_in_raw_data(
    data: &[u8],
    aliases: &HashMap<usize, Arc<AliasHandle>>,
    out: &mut RawAtom,
) -> Result<bool, std::io::Error> {
    match data[0] & TYPE_MASK {
        NUM_ID | VAR_ID => {
            let rest = skip_raw_atom(data);
            let len = data.len() - rest.len();
            out.extend_from_slice(&data[..len]);
            Ok(false)
        }
        ALIAS_ID => {
            let token = data[1..].get_frac_u64().0 as usize;
            let Some(handle) = aliases.get(&token) else {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Missing imported alias for token {}", token),
                ));
            };

            out.put_u8(ALIAS_ID | (data[0] & ALIAS_OPAQUE_FLAG));
            (handle.token() as u64, 1).write_packed(out);
            Ok(true)
        }
        FUN_ID => {
            let start = out.len();
            out.put_u8(data[0] & !HAS_ALIAS_FLAG);
            out.put_u32_le(0);

            let args_start = out.len();
            let mut c = &data[1 + 4..];
            let id_and_attrs;
            let n_args;
            (id_and_attrs, n_args, c) = c.get_frac_u64();
            (id_and_attrs, n_args).write_packed(out);

            let mut has_alias = false;
            for _ in 0..n_args {
                has_alias |= remap_aliases_in_raw_data(c, aliases, out)?;
                c = skip_raw_atom(c);
            }

            if has_alias {
                out[start] |= HAS_ALIAS_FLAG;
            }
            let size = out.len() - args_start;
            (&mut out[start + 1..start + 1 + 4]).put_u32_le(size as u32);
            Ok(has_alias)
        }
        POW_ID => {
            out.put_u8(data[0]);
            let mut c = &data[1..];
            let base_has_alias = remap_aliases_in_raw_data(c, aliases, out)?;
            c = skip_raw_atom(c);
            let exp_has_alias = remap_aliases_in_raw_data(c, aliases, out)?;
            Ok(base_has_alias || exp_has_alias)
        }
        MUL_ID => {
            let start = out.len();
            out.put_u8(data[0] & !MUL_HAS_ALIAS_FLAG);
            out.put_u32_le(0);

            let args_start = out.len();
            let mut c = &data[1 + 4..];
            let n_args;
            let one;
            (n_args, one, c) = c.get_frac_u64();
            (n_args, one).write_packed(out);

            let mut has_alias = false;
            for _ in 0..n_args {
                has_alias |= remap_aliases_in_raw_data(c, aliases, out)?;
                c = skip_raw_atom(c);
            }

            if has_alias {
                out[start] |= MUL_HAS_ALIAS_FLAG;
            }
            let size = out.len() - args_start;
            (&mut out[start + 1..start + 1 + 4]).put_u32_le(size as u32);
            Ok(has_alias)
        }
        ADD_ID => {
            let mut c = &data[1..];
            let n_args;
            let _size;
            (n_args, _size, c) = c.get_frac_u64();

            let mut children = RawAtom::new();
            let mut has_alias = false;
            for _ in 0..n_args {
                has_alias |= remap_aliases_in_raw_data(c, aliases, &mut children)?;
                c = skip_raw_atom(c);
            }

            out.put_u8((data[0] & !HAS_ALIAS_FLAG) | if has_alias { HAS_ALIAS_FLAG } else { 0 });
            (n_args, children.len() as u64).write_packed(out);
            out.extend_from_slice(&children);
            Ok(has_alias)
        }
        x => unreachable!("Bad id {}", x),
    }
}

/// The underlying slice of expression data.
pub type BorrowedRawAtom = [u8];
/// A raw atom that does not have explicit variant information.
#[derive(Debug, Clone, Default, Eq)]
pub struct RawAtom {
    data: Vec<u8>,
    aliases: Vec<Arc<AliasHandle>>,
}

impl RawAtom {
    #[inline]
    pub fn new() -> Self {
        Self {
            data: Vec::new(),
            aliases: Vec::new(),
        }
    }

    #[inline]
    pub fn from_data(data: Vec<u8>) -> Self {
        Self {
            data,
            aliases: Vec::new(),
        }
    }

    #[inline]
    pub fn into_data(self) -> Vec<u8> {
        self.data
    }

    #[inline]
    pub fn clear(&mut self) {
        self.data.clear();
        self.aliases.clear();
    }

    #[inline]
    pub(crate) fn sync_aliases_from(&mut self, aliases: &[Arc<AliasHandle>]) {
        self.aliases.clear();
        self.aliases.extend_from_slice(aliases);
    }

    #[inline]
    pub(crate) fn take_aliases_or_collect(&self) -> Vec<Arc<AliasHandle>> {
        if self.aliases.is_empty() {
            aliases_from_raw_data(&self.data)
        } else {
            self.aliases.clone()
        }
    }
}

impl PartialEq for RawAtom {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl Hash for RawAtom {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.hash(state);
    }
}

impl std::ops::Deref for RawAtom {
    type Target = Vec<u8>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl std::ops::DerefMut for RawAtom {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl Borrow<BorrowedRawAtom> for RawAtom {
    #[inline]
    fn borrow(&self) -> &BorrowedRawAtom {
        &self.data
    }
}

impl Borrow<BorrowedRawAtom> for Atom {
    fn borrow(&self) -> &BorrowedRawAtom {
        self.as_view().get_data()
    }
}

impl Borrow<BorrowedRawAtom> for AtomView<'_> {
    fn borrow(&self) -> &BorrowedRawAtom {
        self.get_data()
    }
}

/// Allows the atom to be used as a key and looked up through a mapping to `&[u8]`.
pub trait KeyLookup: Borrow<BorrowedRawAtom> + Eq + Hash {}

impl KeyLookup for Atom {}
impl KeyLookup for AtomView<'_> {}

impl Symbol {
    #[inline]
    pub(crate) fn encode_flags(&self) -> (u8, u32) {
        let mut flags = 0u8;
        if self.is_symmetric {
            flags |= SYM_SYMMETRIC_FLAG;
        }
        if self.is_cyclesymmetric {
            flags |= SYM_CYCLESYMMETRIC_FLAG;
        }
        if self.is_antisymmetric {
            flags |= SYM_ANTISYMMETRIC_FLAG;
        }
        if self.is_scalar {
            flags |= SYM_SCALAR_FLAG;
        }

        let mut extra = 0;

        if self.is_real {
            extra |= SYM_EXTRA_REAL_FLAG;
        }

        if self.is_integer {
            extra |= SYM_EXTRA_INTEGER_FLAG;
        }

        if self.is_positive {
            extra |= SYM_EXTRA_POSITIVE_FLAG;
        }

        if self.is_linear {
            extra |= SYM_EXTRA_LINEAR_FLAG;
        }

        match self.wildcard_level {
            0 => {}
            1 => extra |= SYM_EXTRA_WILDCARD_LEVEL_1,
            2 => extra |= SYM_EXTRA_WILDCARD_LEVEL_2,
            _ => extra |= SYM_EXTRA_WILDCARD_LEVEL_3,
        }

        (flags, extra)
    }

    #[inline]
    pub(crate) fn decode_flags(id: u32, flags: u8, extra: u32) -> Symbol {
        let is_cyclesymmetric = (flags & SYM_CYCLESYMMETRIC_FLAG) == SYM_CYCLESYMMETRIC_FLAG;
        let is_symmetric = !is_cyclesymmetric && (flags & SYM_SYMMETRIC_FLAG) != 0;
        let is_antisymmetric = !is_cyclesymmetric && (flags & SYM_ANTISYMMETRIC_FLAG) != 0;
        let is_linear = (extra & SYM_EXTRA_LINEAR_FLAG) != 0;
        let is_scalar = (flags & SYM_SCALAR_FLAG) != 0;

        let is_real = (extra & SYM_EXTRA_REAL_FLAG) != 0;
        let is_integer = (extra & SYM_EXTRA_INTEGER_FLAG) != 0;
        let is_positive = (extra & SYM_EXTRA_POSITIVE_FLAG) != 0;
        let wildcard_level = match extra & SYM_EXTRA_WILDCARD_LEVEL_MASK {
            SYM_EXTRA_WILDCARD_LEVEL_1 => 1,
            SYM_EXTRA_WILDCARD_LEVEL_2 => 2,
            SYM_EXTRA_WILDCARD_LEVEL_3 => 3,
            _ => 0,
        };

        Symbol {
            id,
            is_symmetric,
            is_linear,
            is_antisymmetric,
            is_cyclesymmetric,
            is_scalar,
            is_real,
            is_integer,
            is_positive,
            wildcard_level,
        }
    }
}

impl UserDataKey {
    pub fn read<R: Read>(source: &mut R) -> Result<UserDataKey, std::io::Error> {
        let tag = source.read_u8()?;
        match tag {
            1 => {
                let value = source.read_i64::<LittleEndian>()?;
                Ok(UserDataKey::Integer(value))
            }
            2 => {
                let len = source.read_u32::<LittleEndian>()? as usize;
                let mut buf = vec![0u8; len];
                source.read_exact(&mut buf)?;
                let s = std::string::String::from_utf8(buf)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                Ok(UserDataKey::String(s))
            }
            3 => {
                let data = Atom::import(source, None)?;
                Ok(UserDataKey::Atom(data))
            }
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid UserDataKey tag",
            )),
        }
    }

    pub fn write<W: std::io::Write>(&self, target: &mut W) -> Result<(), std::io::Error> {
        match self {
            UserDataKey::Integer(value) => {
                target.write_u8(1)?;
                target.write_i64::<LittleEndian>(*value)
            }
            UserDataKey::String(s) => {
                target.write_u8(2)?;
                target.write_u32::<LittleEndian>(s.len() as u32)?;
                target.write_all(s.as_bytes())
            }
            UserDataKey::Atom(a) => {
                target.write_u8(3)?;
                a.as_view().write(target) // export without the state
            }
        }
    }
}

impl UserData {
    pub fn read<R: Read>(source: &mut R) -> Result<UserData, std::io::Error> {
        let tag = source.read_u8()?;
        match tag {
            0 => Ok(UserData::None),
            1 => {
                let value = source.read_i64::<LittleEndian>()?;
                Ok(UserData::Integer(value))
            }
            2 => {
                let len = source.read_u32::<LittleEndian>()? as usize;
                let mut buf = vec![0u8; len];
                source.read_exact(&mut buf)?;
                let s = std::string::String::from_utf8(buf)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                Ok(UserData::String(s))
            }
            3 => {
                let mut a = Atom::Zero;
                a.read(source)?;
                Ok(UserData::Atom(a))
            }
            4 => {
                let len = source.read_u32::<LittleEndian>()? as usize;
                let mut list = Vec::with_capacity(len);
                for _ in 0..len {
                    list.push(UserData::read(source)?);
                }
                Ok(UserData::List(list))
            }
            5 => {
                let len = source.read_u32::<LittleEndian>()? as usize;
                let mut map = HashMap::default();
                for _ in 0..len {
                    let key = UserDataKey::read(source)?;
                    let value = UserData::read(source)?;
                    map.insert(key, value);
                }
                Ok(UserData::Map(map))
            }
            6 => {
                let len = source.read_u32::<LittleEndian>()? as usize;
                let mut buf = vec![0u8; len];
                source.read_exact(&mut buf)?;
                Ok(UserData::Serialized(buf))
            }
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid ExtendedUserData tag",
            )),
        }
    }

    pub fn write<W: std::io::Write>(&self, target: &mut W) -> Result<(), std::io::Error> {
        match self {
            UserData::None => target.write_u8(0),
            UserData::Integer(value) => {
                target.write_u8(1)?;
                target.write_i64::<LittleEndian>(*value)
            }
            UserData::String(s) => {
                target.write_u8(2)?;
                target.write_u32::<LittleEndian>(s.len() as u32)?;
                target.write_all(s.as_bytes())
            }
            UserData::Atom(a) => {
                target.write_u8(3)?;
                a.as_view().write(target) // export without the state
            }
            UserData::List(list) => {
                target.write_u8(4)?;
                target.write_u32::<LittleEndian>(list.len() as u32)?;
                for item in list {
                    item.write(target)?;
                }
                Ok(())
            }
            UserData::Map(map) => {
                target.write_u8(5)?;
                target.write_u32::<LittleEndian>(map.len() as u32)?;
                for (key, value) in map {
                    key.write(target)?;
                    value.write(target)?;
                }
                Ok(())
            }
            UserData::Serialized(buf) => {
                target.write_u8(6)?;
                target.write_u32::<LittleEndian>(buf.len() as u32)?;
                target.write_all(buf)
            }
        }
    }
}

/// An inline variable.
#[derive(Copy, Clone, PartialEq, Eq)]
#[cfg_attr(
    feature = "bincode",
    derive(bincode_trait_derive::Encode),
    derive(bincode_trait_derive::Decode),
    derive(bincode_trait_derive::BorrowDecodeFromDecode),
    trait_decode(trait = crate::state::HasStateMap)
)]
pub struct InlineVar {
    data: [u8; 16],
    size: u8,
}

impl Hash for InlineVar {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_view().hash(state);
    }
}

impl PartialOrd for InlineVar {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.as_view().cmp(&other.as_view()))
    }
}

impl Ord for InlineVar {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_view().cmp(&other.as_view())
    }
}

impl std::fmt::Display for InlineVar {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_view().fmt(fmt)
    }
}

impl std::fmt::Debug for InlineVar {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_view().fmt(fmt)
    }
}

impl InlineVar {
    /// Create a new inline variable.
    pub fn new(symbol: Symbol) -> InlineVar {
        let mut data = [0; 16];
        let (flags, extra) = symbol.encode_flags();
        data[0] = flags | VAR_ID;

        let size = 1 + (symbol.id as u64, (extra * 2) as u64 + 1).get_packed_size() as u8;
        (symbol.id as u64, (extra * 2) as u64 + 1).write_packed_fixed(&mut data[1..]);
        InlineVar { data, size }
    }

    pub fn get_symbol(&self) -> Symbol {
        self.as_var_view().get_symbol()
    }

    pub fn get_data(&self) -> &[u8] {
        &self.data[..self.size as usize]
    }

    pub fn as_var_view(&self) -> VarView<'_> {
        VarView {
            data: &self.data[..self.size as usize],
            aliases: &NO_ALIASES,
        }
    }

    pub fn as_view(&self) -> AtomView<'_> {
        AtomView::Var(VarView {
            data: &self.data[..self.size as usize],
            aliases: &NO_ALIASES,
        })
    }
}

impl From<Symbol> for InlineVar {
    fn from(symbol: Symbol) -> InlineVar {
        InlineVar::new(symbol)
    }
}

/// An inline rational number that has 64-bit components.
#[derive(Copy, Clone, PartialEq, Eq)]
#[cfg_attr(
    feature = "bincode",
    derive(bincode_trait_derive::Encode),
    derive(bincode_trait_derive::Decode),
    derive(bincode_trait_derive::BorrowDecodeFromDecode),
    trait_decode(trait = crate::state::HasStateMap)
)]
pub struct InlineNum {
    data: [u8; 24],
    size: u8,
}

impl Hash for InlineNum {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_view().hash(state);
    }
}

impl PartialOrd for InlineNum {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.as_view().cmp(&other.as_view()))
    }
}

impl Ord for InlineNum {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_view().cmp(&other.as_view())
    }
}

impl std::fmt::Display for InlineNum {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_view().fmt(fmt)
    }
}

impl std::fmt::Debug for InlineNum {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_view().fmt(fmt)
    }
}

impl InlineNum {
    /// Create a new inline number. The gcd of num and den should be 1.
    pub fn new(num: i64, den: u64) -> InlineNum {
        let mut data = [0; 24];
        data[0] = NUM_ID;

        let size = 1 + (num, den).get_packed_size() as u8;
        (num, den).write_packed_fixed(&mut data[1..]);
        InlineNum { data, size }
    }

    pub fn get_data(&self) -> &[u8] {
        &self.data[..self.size as usize]
    }

    pub fn as_num_view(&self) -> NumView<'_> {
        NumView {
            data: &self.data[..self.size as usize],
            aliases: &NO_ALIASES,
        }
    }

    pub fn as_view(&self) -> AtomView<'_> {
        AtomView::Num(NumView {
            data: &self.data[..self.size as usize],
            aliases: &NO_ALIASES,
        })
    }
}

#[cfg(feature = "bincode")]
impl bincode::Encode for Atom {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        use bincode::enc::write::Writer;

        let d = self.as_view().get_data();
        let writer = encoder.writer();
        writer.write(&[0])?;
        writer.write(&d.len().to_le_bytes())?;
        writer.write(d)
    }
}

#[cfg(feature = "bincode")]
impl<C: crate::state::HasStateMap> bincode::Decode<C> for Atom {
    fn decode<D: bincode::de::Decoder<Context = C>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        use bincode::de::read::Reader;
        let atom = {
            // equivalent to Atom::read
            let source = decoder.reader();

            let mut dest = Atom::Zero.into_raw();

            // should also set whether rat poly coefficient needs to be converted
            let mut flags_buf = [0; 1];
            let mut size_buf = [0; 8];

            source.read(&mut flags_buf)?;
            source.read(&mut size_buf)?;

            let n_size = u64::from_le_bytes(size_buf);

            dest.extend(size_buf);
            dest.resize(n_size as usize, 0);
            source.read(&mut dest)?;

            unsafe {
                match dest[0] & TYPE_MASK {
                    NUM_ID => Atom::Num(Num::from_raw(dest)),
                    VAR_ID => Atom::Var(Var::from_raw(dest)),
                    FUN_ID => Atom::Fun(Fun::from_raw(dest)),
                    MUL_ID => Atom::Mul(Mul::from_raw(dest)),
                    ADD_ID => Atom::Add(Add::from_raw(dest)),
                    POW_ID => Atom::Pow(Pow::from_raw(dest)),
                    ALIAS_ID => Atom::Alias(Alias::from_raw(dest)),
                    _ => unreachable!("Unknown type {}", dest[0]),
                }
            }
        };

        let state_map = decoder.context().get_state_map();
        Ok(atom.as_view().rename(state_map))
    }
}

impl Atom {
    /// Read from a binary stream. The format is the byte-length first
    /// followed by the data.
    pub(crate) fn read<R: Read>(&mut self, source: &mut R) -> Result<(), std::io::Error> {
        let dest = read_raw_atom(source)?;
        unsafe {
            match dest[0] & TYPE_MASK {
                NUM_ID => *self = Atom::Num(Num::from_raw(dest)),
                VAR_ID => *self = Atom::Var(Var::from_raw(dest)),
                FUN_ID => *self = Atom::Fun(Fun::from_raw(dest)),
                MUL_ID => *self = Atom::Mul(Mul::from_raw(dest)),
                ADD_ID => *self = Atom::Add(Add::from_raw(dest)),
                POW_ID => *self = Atom::Pow(Pow::from_raw(dest)),
                ALIAS_ID => *self = Atom::Alias(Alias::from_raw(dest)),
                _ => unreachable!("Unknown type {}", dest[0]),
            }
        }

        Ok(())
    }

    /// Import an expression and its state from a binary stream. The state will be merged
    /// with the current one. If a symbol has conflicting attributes, the conflict
    /// can be resolved using the renaming function `conflict_fn`.
    ///
    /// Expressions can be exported using [Atom::export](crate::atom::core::AtomCore::export).
    pub fn import<R: Read>(
        source: &mut R,
        conflict_fn: Option<Box<dyn Fn(&str) -> String>>,
    ) -> Result<Atom, std::io::Error> {
        let state_map = State::import(source, conflict_fn)?;

        let mut marker_or_n_terms_buf = [0; 8];
        source.read_exact(&mut marker_or_n_terms_buf)?;
        let marker_or_n_terms = u64::from_le_bytes(marker_or_n_terms_buf);

        let mut imported_aliases = HashMap::default();
        let has_alias_section = marker_or_n_terms == ALIAS_EXPORT_SECTION_MAGIC;
        let n_terms = if has_alias_section {
            let n_aliases = source.read_u64::<LittleEndian>()?;
            for _ in 0..n_aliases {
                let old_token = source.read_u64::<LittleEndian>()? as usize;
                let raw = read_raw_atom(source)?;
                let raw = remap_aliases_in_raw_atom(raw, &imported_aliases)?;
                let body = unsafe { Atom::from_raw(raw) }.as_view().rename(&state_map);
                let handle = register_alias_atom(body);
                imported_aliases.insert(old_token, handle);
            }

            source.read_u64::<LittleEndian>()?
        } else {
            marker_or_n_terms
        };

        if n_terms == 1 {
            let raw = read_raw_atom(source)?;
            let raw = if has_alias_section {
                remap_aliases_in_raw_atom(raw, &imported_aliases)?
            } else {
                raw
            };
            let a = unsafe { Atom::from_raw(raw) };
            Ok(a.as_view().rename(&state_map))
        } else {
            let mut res = Atom::new();
            let a = res.to_add();

            for _ in 0..n_terms {
                let raw = read_raw_atom(source)?;
                let raw = if has_alias_section {
                    remap_aliases_in_raw_atom(raw, &imported_aliases)?
                } else {
                    raw
                };
                let tmp = unsafe { Atom::from_raw(raw) };
                a.extend(tmp.as_view());
            }

            Ok(res.as_view().rename(&state_map))
        }
    }

    /// Read a stateless expression from a binary stream, renaming the symbols using the provided state map.
    pub fn import_with_map<R: Read>(
        source: &mut R,
        state_map: &StateMap,
    ) -> Result<Atom, std::io::Error> {
        let mut a = Atom::new();
        a.read(source)?;
        Ok(a.as_view().rename(state_map))
    }

    #[allow(dead_code)]
    pub(crate) unsafe fn from_raw(raw: RawAtom) -> Self {
        unsafe {
            match raw[0] & TYPE_MASK {
                NUM_ID => Atom::Num(Num::from_raw(raw)),
                VAR_ID => Atom::Var(Var::from_raw(raw)),
                FUN_ID => Atom::Fun(Fun::from_raw(raw)),
                MUL_ID => Atom::Mul(Mul::from_raw(raw)),
                ADD_ID => Atom::Add(Add::from_raw(raw)),
                POW_ID => Atom::Pow(Pow::from_raw(raw)),
                ALIAS_ID => Atom::Alias(Alias::from_raw(raw)),
                _ => unreachable!("Unknown type {}", raw[0]),
            }
        }
    }

    /// Get the capacity of the underlying buffer.
    pub(crate) fn get_capacity(&self) -> usize {
        match self {
            Atom::Num(n) => n.data.capacity(),
            Atom::Var(v) => v.data.capacity(),
            Atom::Fun(f) => f.data.capacity(),
            Atom::Mul(m) => m.data.capacity(),
            Atom::Add(a) => a.data.capacity(),
            Atom::Pow(p) => p.data.capacity(),
            Atom::Alias(a) => a.data.capacity(),
            Atom::Zero => 0,
        }
    }

    pub(crate) fn clear_alias_handles(&mut self) {
        match self {
            Atom::Num(n) => {
                n.aliases.clear();
                n.data.sync_aliases_from(&n.aliases);
            }
            Atom::Var(v) => {
                v.aliases.clear();
                v.data.sync_aliases_from(&v.aliases);
            }
            Atom::Fun(f) => {
                f.aliases.clear();
                f.data.sync_aliases_from(&f.aliases);
                f.data[0] &= !HAS_ALIAS_FLAG;
            }
            Atom::Mul(m) => {
                m.aliases.clear();
                m.data.sync_aliases_from(&m.aliases);
                m.data[0] &= !MUL_HAS_ALIAS_FLAG;
            }
            Atom::Add(a) => {
                a.aliases.clear();
                a.data.sync_aliases_from(&a.aliases);
                a.data[0] &= !HAS_ALIAS_FLAG;
            }
            Atom::Pow(p) => {
                p.aliases.clear();
                p.data.sync_aliases_from(&p.aliases);
            }
            Atom::Alias(_) => {}
            Atom::Zero => {}
        }
    }

    pub(crate) fn refresh_alias_handles_from_tree(&mut self) {
        let aliases = aliases_from_view(self.as_view());
        match self {
            Atom::Num(n) => {
                n.aliases.clear();
                n.data.sync_aliases_from(&n.aliases);
            }
            Atom::Var(v) => {
                v.aliases.clear();
                v.data.sync_aliases_from(&v.aliases);
            }
            Atom::Fun(f) => {
                f.aliases = aliases;
                f.data.sync_aliases_from(&f.aliases);
                f.refresh_alias_flag_from_tree();
            }
            Atom::Mul(m) => {
                m.aliases = aliases;
                m.data.sync_aliases_from(&m.aliases);
                m.refresh_alias_flag_from_tree();
            }
            Atom::Add(a) => {
                a.aliases = aliases;
                a.data.sync_aliases_from(&a.aliases);
                a.refresh_alias_flag_from_tree();
            }
            Atom::Pow(p) => {
                p.aliases = aliases;
                p.data.sync_aliases_from(&p.aliases);
            }
            Atom::Alias(a) => {
                a.aliases = aliases;
                a.data.sync_aliases_from(&a.aliases);
            }
            Atom::Zero => {}
        }
    }
}

/// A number/coefficient.
#[derive(Debug, Clone)]
pub struct Num {
    data: RawAtom,
    aliases: Vec<Arc<AliasHandle>>,
}

impl PartialEq for Num {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl Eq for Num {}

impl Hash for Num {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.hash(state);
    }
}

impl Num {
    #[inline(always)]
    pub fn zero(mut buffer: RawAtom) -> Num {
        buffer.clear();
        buffer.put_u8(NUM_ID);
        buffer.put_u8(1);
        buffer.put_u8(0);
        Num {
            data: buffer,
            aliases: Vec::new(),
        }
    }

    #[inline]
    pub fn new(num: Coefficient) -> Num {
        let mut buffer = RawAtom::new();
        buffer.put_u8(NUM_ID);
        num.write_packed(&mut buffer);
        Num {
            data: buffer,
            aliases: Vec::new(),
        }
    }

    #[inline(always)]
    pub fn new_into(num: Coefficient, mut buffer: RawAtom) -> Num {
        buffer.clear();
        buffer.put_u8(NUM_ID);
        num.write_packed(&mut buffer);
        Num {
            data: buffer,
            aliases: Vec::new(),
        }
    }

    #[inline]
    pub fn from_view_into(a: &NumView<'_>, mut buffer: RawAtom) -> Num {
        buffer.clear();
        buffer.extend(a.data);
        Num {
            data: buffer,
            aliases: aliases_from_view(AtomView::Num(*a)),
        }
    }

    #[inline]
    pub fn set_from_coeff(&mut self, num: Coefficient) {
        self.data.clear();
        self.aliases.clear();
        self.data.put_u8(NUM_ID);
        num.write_packed(&mut self.data);
    }

    #[inline]
    pub fn set_from_view(&mut self, a: &NumView<'_>) {
        self.data.clear();
        self.data.extend(a.data);
        self.aliases = aliases_from_view(AtomView::Num(*a));
        self.data.sync_aliases_from(&self.aliases);
    }

    pub fn add(&mut self, other: &NumView<'_>) {
        let nv = self.to_num_view();
        let a = nv.get_coeff_view();
        let b = other.get_coeff_view();
        let n = a + b;

        self.data.truncate(1);
        n.write_packed(&mut self.data);
    }

    pub fn mul(&mut self, other: &NumView<'_>) {
        let nv = self.to_num_view();
        let a = nv.get_coeff_view();
        let b = other.get_coeff_view();
        let n = a * b;

        self.data.truncate(1);
        n.write_packed(&mut self.data);
    }

    #[inline]
    pub fn to_num_view(&self) -> NumView<'_> {
        NumView {
            data: &self.data,
            aliases: &self.aliases,
        }
    }

    #[inline(always)]
    pub fn as_view(&self) -> AtomView<'_> {
        AtomView::Num(self.to_num_view())
    }

    #[inline(always)]
    pub fn into_raw(mut self) -> RawAtom {
        self.data.sync_aliases_from(&self.aliases);
        self.data
    }

    #[inline(always)]
    pub(crate) unsafe fn from_raw(raw: RawAtom) -> Num {
        Num {
            data: raw,
            aliases: Vec::new(),
        }
    }
}

/// A variable.
#[derive(Debug, Clone)]
pub struct Var {
    data: RawAtom,
    aliases: Vec<Arc<AliasHandle>>,
}

impl PartialEq for Var {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl Eq for Var {}

impl Hash for Var {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.hash(state);
    }
}

impl Var {
    #[inline]
    pub fn new(symbol: Symbol) -> Var {
        Self::new_into(symbol, RawAtom::new())
    }

    #[inline]
    pub fn new_into(symbol: Symbol, buffer: RawAtom) -> Var {
        let mut f = Var {
            data: buffer,
            aliases: Vec::new(),
        };
        f.set_from_symbol(symbol);
        f
    }

    #[inline]
    pub fn from_view_into(a: &VarView<'_>, mut buffer: RawAtom) -> Var {
        buffer.clear();
        buffer.extend(a.data);
        Var {
            data: buffer,
            aliases: aliases_from_view(AtomView::Var(*a)),
        }
    }

    #[inline]
    pub fn set_from_symbol(&mut self, symbol: Symbol) {
        self.data.clear();
        self.aliases.clear();

        let (flags, extra) = symbol.encode_flags();
        self.data.put_u8(flags | VAR_ID);

        // shift by 1, so that the no-flag case does not take up extra space
        (symbol.id as u64, (extra * 2) as u64 + 1).write_packed(&mut self.data);
    }

    #[inline]
    pub fn to_var_view(&self) -> VarView<'_> {
        VarView {
            data: &self.data,
            aliases: &self.aliases,
        }
    }

    #[inline]
    pub fn set_from_view(&mut self, view: &VarView) {
        self.data.clear();
        self.data.extend(view.data);
        self.aliases = aliases_from_view(AtomView::Var(*view));
        self.data.sync_aliases_from(&self.aliases);
    }

    #[inline(always)]
    pub fn as_view(&self) -> AtomView<'_> {
        AtomView::Var(self.to_var_view())
    }

    #[inline]
    pub fn get_symbol(&self) -> Symbol {
        self.to_var_view().get_symbol()
    }

    #[inline(always)]
    pub fn into_raw(mut self) -> RawAtom {
        self.data.sync_aliases_from(&self.aliases);
        self.data
    }

    #[inline(always)]
    pub(crate) unsafe fn from_raw(raw: RawAtom) -> Var {
        Var {
            data: raw,
            aliases: Vec::new(),
        }
    }
}

/// A reference to an aliased expression.
#[derive(Debug, Clone)]
pub struct Alias {
    data: RawAtom,
    aliases: Vec<Arc<AliasHandle>>,
}

impl PartialEq for Alias {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl Eq for Alias {}

impl Hash for Alias {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.hash(state);
    }
}

impl Alias {
    #[inline]
    pub(crate) fn new(handle: Arc<AliasHandle>) -> Alias {
        Self::new_into(handle, RawAtom::new())
    }

    #[inline]
    pub(crate) fn new_opaque(handle: Arc<AliasHandle>) -> Alias {
        Self::new_into_impl(handle, RawAtom::new(), true)
    }

    #[inline]
    pub(crate) fn new_into(handle: Arc<AliasHandle>, buffer: RawAtom) -> Alias {
        Self::new_into_impl(handle, buffer, false)
    }

    #[inline]
    fn new_into_impl(handle: Arc<AliasHandle>, mut buffer: RawAtom, opaque: bool) -> Alias {
        buffer.clear();
        buffer.put_u8(ALIAS_ID | if opaque { ALIAS_OPAQUE_FLAG } else { 0 });
        (handle.token() as u64, 1).write_packed(&mut buffer);
        buffer.sync_aliases_from(std::slice::from_ref(&handle));
        Alias {
            data: buffer,
            aliases: vec![handle],
        }
    }

    #[inline]
    pub fn from_view_into(a: &AliasView<'_>, mut buffer: RawAtom) -> Alias {
        buffer.clear();
        buffer.extend(a.data);
        let aliases = aliases_from_view(AtomView::Alias(*a));
        buffer.sync_aliases_from(&aliases);
        Alias {
            data: buffer,
            aliases,
        }
    }

    #[inline]
    pub fn to_alias_view(&self) -> AliasView<'_> {
        AliasView {
            data: &self.data,
            aliases: &self.aliases,
        }
    }

    #[inline]
    pub fn set_from_view(&mut self, view: &AliasView) {
        self.data.clear();
        self.data.extend(view.data);
        self.aliases = aliases_from_view(AtomView::Alias(*view));
        self.data.sync_aliases_from(&self.aliases);
    }

    #[inline(always)]
    pub fn as_view(&self) -> AtomView<'_> {
        AtomView::Alias(self.to_alias_view())
    }

    #[inline(always)]
    pub fn into_raw(mut self) -> RawAtom {
        self.data.sync_aliases_from(&self.aliases);
        self.data
    }

    #[inline(always)]
    pub(crate) unsafe fn from_raw(raw: RawAtom) -> Alias {
        let aliases = raw.take_aliases_or_collect();
        Alias { data: raw, aliases }
    }
}

/// A general function.
#[derive(Debug, Clone)]
pub struct Fun {
    data: RawAtom,
    aliases: Vec<Arc<AliasHandle>>,
}

impl PartialEq for Fun {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl Eq for Fun {}

impl Hash for Fun {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.hash(state);
    }
}

impl Fun {
    #[inline]
    pub(crate) fn new_into(id: Symbol, buffer: RawAtom) -> Fun {
        let mut f = Fun {
            data: buffer,
            aliases: Vec::new(),
        };
        f.set_from_symbol(id);
        f
    }

    #[inline]
    pub fn from_view_into(a: &FunView<'_>, mut buffer: RawAtom) -> Fun {
        buffer.clear();
        buffer.extend(a.data);
        let aliases = aliases_from_view(AtomView::Fun(*a));
        buffer.sync_aliases_from(&aliases);
        Fun {
            data: buffer,
            aliases,
        }
    }

    #[inline]
    pub(crate) fn set_from_symbol(&mut self, symbol: Symbol) {
        self.data.clear();
        self.aliases.clear();

        let (flags, extra) = symbol.encode_flags();
        self.data.put_u8(flags | FUN_ID | NOT_NORMALIZED);

        self.data.put_u32_le(0_u32);

        let buf_pos = self.data.len();

        ((extra as u64) << 32 | symbol.id as u64, 0).write_packed(&mut self.data);

        let new_buf_pos = self.data.len();
        let mut cursor = &mut self.data[1..];
        cursor.put_u32_le((new_buf_pos - buf_pos) as u32);
    }

    #[inline]
    pub(crate) fn set_normalized(&mut self, normalized: bool) {
        if !normalized {
            self.data[0] |= NOT_NORMALIZED;
        } else {
            self.data[0] &= !NOT_NORMALIZED;
        }
    }

    pub(crate) fn add_arg(&mut self, other: AtomView) {
        self.data[0] |= NOT_NORMALIZED;
        self.data[0] |= alias_flag_for(other);
        merge_aliases(&mut self.aliases, &aliases_from_view(other));
        self.data.sync_aliases_from(&self.aliases);

        // may increase size of the num of args
        let mut c = &self.data[1 + 4..];

        let buf_pos = 1 + 4;

        let name;
        let mut n_args;
        (name, n_args, c) = c.get_frac_u64();

        let old_size = unsafe { c.as_ptr().offset_from(self.data.as_ptr()) } as usize - 1 - 4;

        n_args += 1;

        let new_size = (name, n_args).get_packed_size() as usize;

        match new_size.cmp(&old_size) {
            Ordering::Equal => {}
            Ordering::Less => {
                self.data.copy_within(1 + 4 + old_size.., 1 + 4 + new_size);
                let new_len = self.data.len() - old_size + new_size;
                self.data.resize(new_len, 0);
            }
            Ordering::Greater => {
                let old_len = self.data.len();
                self.data.resize(old_len + new_size - old_size, 0);
                self.data
                    .copy_within(1 + 4 + old_size..old_len, 1 + 4 + new_size);
            }
        }

        // size should be ok now
        (name, n_args).write_packed_fixed(&mut self.data[1 + 4..1 + 4 + new_size]);

        self.data.extend(other.get_data());

        let new_buf_pos = self.data.len();

        let mut cursor = &mut self.data[1..];
        cursor.put_u32_le((new_buf_pos - buf_pos) as u32);
    }

    pub(crate) fn add_args<'a>(&mut self, other: &[AtomView<'a>]) {
        self.data[0] |= NOT_NORMALIZED;
        if other.iter().any(|item| item.has_alias()) {
            self.data[0] |= HAS_ALIAS_FLAG;
        }

        // may increase size of the num of args
        let mut c = &self.data[1 + 4..];

        let buf_pos = 1 + 4;

        let name;
        let mut n_args;
        (name, n_args, c) = c.get_frac_u64();

        let old_size = unsafe { c.as_ptr().offset_from(self.data.as_ptr()) } as usize - 1 - 4;

        n_args += other.len() as u64;

        let new_size = (name, n_args).get_packed_size() as usize;

        match new_size.cmp(&old_size) {
            Ordering::Equal => {}
            Ordering::Less => {
                self.data.copy_within(1 + 4 + old_size.., 1 + 4 + new_size);
                let new_len = self.data.len() - old_size + new_size;
                self.data.resize(new_len, 0);
            }
            Ordering::Greater => {
                let old_len = self.data.len();
                self.data.resize(old_len + new_size - old_size, 0);
                self.data
                    .copy_within(1 + 4 + old_size..old_len, 1 + 4 + new_size);
            }
        }

        // size should be ok now
        (name, n_args).write_packed_fixed(&mut self.data[1 + 4..1 + 4 + new_size]);

        for item in other {
            merge_aliases(&mut self.aliases, &aliases_from_view(*item));
            self.data.extend(item.get_data());
        }
        self.data.sync_aliases_from(&self.aliases);

        let new_buf_pos = self.data.len();

        let mut cursor = &mut self.data[1..];
        cursor.put_u32_le((new_buf_pos - buf_pos) as u32);
    }

    #[inline(always)]
    pub fn to_fun_view(&self) -> FunView<'_> {
        FunView {
            data: &self.data,
            aliases: &self.aliases,
        }
    }

    pub fn set_from_view(&mut self, view: &FunView) {
        self.data.clear();
        self.data.extend(view.data);
        self.aliases = aliases_from_view(AtomView::Fun(*view));
        self.data.sync_aliases_from(&self.aliases);
    }

    #[inline(always)]
    pub fn as_view(&self) -> AtomView<'_> {
        AtomView::Fun(self.to_fun_view())
    }

    #[inline(always)]
    pub fn get_symbol(&self) -> Symbol {
        self.to_fun_view().get_symbol()
    }

    #[inline(always)]
    pub fn get_nargs(&self) -> usize {
        self.to_fun_view().get_nargs()
    }

    #[inline(always)]
    pub fn into_raw(mut self) -> RawAtom {
        self.data.sync_aliases_from(&self.aliases);
        self.data
    }

    #[inline(always)]
    pub(crate) unsafe fn from_raw(raw: RawAtom) -> Fun {
        let aliases = raw.take_aliases_or_collect();
        Fun { data: raw, aliases }
    }

    #[inline(always)]
    pub(crate) fn refresh_alias_flag_from_tree(&mut self) {
        let has_alias = self.to_fun_view().iter().any(|arg| arg.has_alias());
        if has_alias {
            self.data[0] |= HAS_ALIAS_FLAG;
        } else {
            self.data[0] &= !HAS_ALIAS_FLAG;
        }
    }
}

/// An expression raised to the power of another expression.
#[derive(Debug, Clone)]
pub struct Pow {
    data: RawAtom,
    aliases: Vec<Arc<AliasHandle>>,
}

impl PartialEq for Pow {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl Eq for Pow {}

impl Hash for Pow {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.hash(state);
    }
}

impl Pow {
    #[inline]
    pub(crate) fn new_into(base: AtomView, exp: AtomView, buffer: RawAtom) -> Pow {
        let mut f = Pow {
            data: buffer,
            aliases: Vec::new(),
        };
        f.set_from_base_and_exp(base, exp);
        f
    }

    #[inline]
    pub fn from_view_into(a: &PowView<'_>, mut buffer: RawAtom) -> Pow {
        buffer.clear();
        buffer.extend(a.data);
        let aliases = aliases_from_view(AtomView::Pow(*a));
        buffer.sync_aliases_from(&aliases);
        Pow {
            data: buffer,
            aliases,
        }
    }

    #[inline]
    pub(crate) fn set_from_base_and_exp(&mut self, base: AtomView, exp: AtomView) {
        self.data.clear();
        self.aliases.clear();
        merge_aliases(&mut self.aliases, &aliases_from_view(base));
        merge_aliases(&mut self.aliases, &aliases_from_view(exp));
        self.data.sync_aliases_from(&self.aliases);
        self.data.put_u8(POW_ID | NOT_NORMALIZED);
        self.data.extend(base.get_data());
        self.data.extend(exp.get_data());
    }

    #[inline]
    pub(crate) fn set_normalized(&mut self, normalized: bool) {
        if !normalized {
            self.data[0] |= NOT_NORMALIZED;
        } else {
            self.data[0] &= !NOT_NORMALIZED;
        }
    }

    #[inline(always)]
    pub fn to_pow_view(&self) -> PowView<'_> {
        PowView {
            data: &self.data,
            aliases: &self.aliases,
        }
    }

    #[inline(always)]
    pub fn set_from_view(&mut self, view: &PowView) {
        self.data.clear();
        self.data.extend(view.data);
        self.aliases = aliases_from_view(AtomView::Pow(*view));
        self.data.sync_aliases_from(&self.aliases);
    }

    #[inline(always)]
    pub fn as_view(&self) -> AtomView<'_> {
        AtomView::Pow(self.to_pow_view())
    }

    #[inline(always)]
    pub fn into_raw(mut self) -> RawAtom {
        self.data.sync_aliases_from(&self.aliases);
        self.data
    }

    #[inline(always)]
    pub(crate) unsafe fn from_raw(raw: RawAtom) -> Pow {
        let aliases = raw.take_aliases_or_collect();
        Pow { data: raw, aliases }
    }
}

/// Multiplication of multiple subexpressions.
#[derive(Clone)]
pub struct Mul {
    data: RawAtom,
    aliases: Vec<Arc<AliasHandle>>,
}

impl PartialEq for Mul {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl Eq for Mul {}

impl Hash for Mul {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.hash(state);
    }
}

impl Default for Mul {
    fn default() -> Self {
        Self::new()
    }
}

impl Mul {
    #[inline]
    pub(crate) fn new() -> Mul {
        Self::new_into(RawAtom::new())
    }

    #[inline]
    pub(crate) fn new_into(mut buffer: RawAtom) -> Mul {
        buffer.clear();
        buffer.put_u8(MUL_ID | NOT_NORMALIZED);
        buffer.put_u32_le(0_u32);
        (0u64, 1).write_packed(&mut buffer);
        let len = buffer.len() as u32 - 1 - 4;
        (&mut buffer[1..]).put_u32_le(len);

        Mul {
            data: buffer,
            aliases: Vec::new(),
        }
    }

    #[inline]
    pub fn from_view_into(a: &MulView<'_>, mut buffer: RawAtom) -> Mul {
        buffer.clear();
        buffer.extend(a.data);
        let aliases = aliases_from_view(AtomView::Mul(*a));
        buffer.sync_aliases_from(&aliases);
        Mul {
            data: buffer,
            aliases,
        }
    }

    #[inline]
    pub(crate) fn set_normalized(&mut self, normalized: bool) {
        if !normalized {
            self.data[0] |= NOT_NORMALIZED;
        } else {
            self.data[0] &= !NOT_NORMALIZED;
        }
    }

    #[inline]
    pub fn set_from_view(&mut self, view: &MulView) {
        self.data.clear();
        self.data.extend(view.data);
        self.aliases = aliases_from_view(AtomView::Mul(*view));
        self.data.sync_aliases_from(&self.aliases);
    }

    #[inline]
    pub(crate) fn extend(&mut self, other: AtomView<'_>) {
        self.data[0] |= NOT_NORMALIZED;
        self.data[0] |= mul_alias_flag_for(other);
        merge_aliases(&mut self.aliases, &aliases_from_view(other));
        self.data.sync_aliases_from(&self.aliases);

        // may increase size of the num of args
        let mut c = &self.data[1 + 4..];

        let buf_pos = 1 + 4;

        let mut n_args;
        (n_args, _, c) = c.get_frac_u64(); // TODO: pack size and n_args

        let old_size = unsafe { c.as_ptr().offset_from(self.data.as_ptr()) } as usize - 1 - 4;

        let data_start = match other {
            AtomView::Mul(m) => {
                let mut sd = &m.data[1 + 4..];
                let sub_n_args;
                (sub_n_args, _, sd) = sd.get_frac_u64();

                n_args += sub_n_args;
                sd
            }
            _ => {
                n_args += 1;
                other.get_data()
            }
        };

        let new_size = (n_args, 1).get_packed_size() as usize;

        match new_size.cmp(&old_size) {
            Ordering::Equal => {}
            Ordering::Less => {
                self.data.copy_within(1 + 4 + old_size.., 1 + 4 + new_size);
                let new_len = self.data.len() - old_size + new_size;
                self.data.resize(new_len, 0);
            }
            Ordering::Greater => {
                let old_len = self.data.len();
                self.data.resize(old_len + new_size - old_size, 0);
                self.data
                    .copy_within(1 + 4 + old_size..old_len, 1 + 4 + new_size);
            }
        }

        // size should be ok now
        (n_args, 1).write_packed_fixed(&mut self.data[1 + 4..1 + 4 + new_size]);

        self.data.extend_from_slice(data_start);

        let new_buf_pos = self.data.len();

        let mut cursor = &mut self.data[1..];
        cursor.put_u32_le((new_buf_pos - buf_pos) as u32);
    }

    pub(crate) fn replace_first(&mut self, other: AtomView) {
        let mut c = &self.data[1 + 4..];

        (_, _, c) = c.get_frac_u64(); // TODO: pack size and n_args

        let first_arg_start = unsafe { c.as_ptr().offset_from(self.data.as_ptr()) } as usize;

        // get size of first arg
        let aa = self.to_mul_view().to_slice().get(0);

        let old_first_len = aa.get_data().len();
        let new_first_len = other.get_data().len();

        match new_first_len.cmp(&old_first_len) {
            Ordering::Equal => {}
            Ordering::Less => {
                self.data
                    .copy_within(1 + 4 + old_first_len.., 1 + 4 + new_first_len);
                let new_len = self.data.len() - old_first_len + new_first_len;
                self.data.truncate(new_len);
                (&mut self.data[1..]).put_u32_le((new_len - 1 - 4) as u32);
            }
            Ordering::Greater => {
                let old_len = self.data.len();
                self.data.resize(old_len + new_first_len - old_first_len, 0);
                self.data
                    .copy_within(1 + 4 + old_first_len..old_len, 1 + 4 + new_first_len);
                (&mut self.data[1..])
                    .put_u32_le((old_len - 1 - 4 + new_first_len - old_first_len) as u32);
            }
        }

        self.data[first_arg_start..first_arg_start + new_first_len]
            .copy_from_slice(other.get_data());
        self.aliases = aliases_from_view(self.to_mul_view().as_view());
        self.data.sync_aliases_from(&self.aliases);
        let has_alias = self.to_mul_view().iter().any(|x| x.has_alias());
        if has_alias {
            self.data[0] |= MUL_HAS_ALIAS_FLAG;
        } else {
            self.data[0] &= !MUL_HAS_ALIAS_FLAG;
        }
    }

    #[inline]
    pub fn to_mul_view(&self) -> MulView<'_> {
        MulView {
            data: &self.data,
            aliases: &self.aliases,
        }
    }

    pub(crate) fn set_has_coefficient(&mut self, has_coeff: bool) {
        if has_coeff {
            self.data[0] |= MUL_HAS_COEFF_FLAG;
        } else {
            self.data[0] &= !MUL_HAS_COEFF_FLAG;
        }
    }

    #[inline(always)]
    pub fn as_view(&self) -> AtomView<'_> {
        AtomView::Mul(self.to_mul_view())
    }

    #[inline(always)]
    pub fn get_nargs(&self) -> usize {
        self.to_mul_view().get_nargs()
    }

    #[inline(always)]
    pub fn into_raw(mut self) -> RawAtom {
        self.data.sync_aliases_from(&self.aliases);
        self.data
    }

    #[inline(always)]
    pub(crate) unsafe fn from_raw(raw: RawAtom) -> Mul {
        let aliases = raw.take_aliases_or_collect();
        Mul { data: raw, aliases }
    }

    #[inline(always)]
    pub(crate) fn refresh_alias_flag_from_tree(&mut self) {
        let has_alias = self.to_mul_view().iter().any(|arg| arg.has_alias());
        if has_alias {
            self.data[0] |= MUL_HAS_ALIAS_FLAG;
        } else {
            self.data[0] &= !MUL_HAS_ALIAS_FLAG;
        }
    }
}

/// Addition of multiple subexpressions.
#[derive(Clone)]
pub struct Add {
    data: RawAtom,
    aliases: Vec<Arc<AliasHandle>>,
}

impl PartialEq for Add {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl Eq for Add {}

impl Hash for Add {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.hash(state);
    }
}

impl Default for Add {
    fn default() -> Self {
        Self::new()
    }
}

impl Add {
    #[inline]
    pub(crate) fn new() -> Add {
        Self::new_into(RawAtom::new())
    }

    #[inline]
    pub(crate) fn new_into(mut buffer: RawAtom) -> Add {
        buffer.clear();
        buffer.put_u8(ADD_ID | NOT_NORMALIZED);
        (0u64, 0).write_packed(&mut buffer);
        Add {
            data: buffer,
            aliases: Vec::new(),
        }
    }

    #[inline]
    pub fn from_view_into(a: &AddView<'_>, mut buffer: RawAtom) -> Add {
        buffer.clear();
        buffer.extend(a.data);
        let aliases = aliases_from_view(AtomView::Add(*a));
        buffer.sync_aliases_from(&aliases);
        Add {
            data: buffer,
            aliases,
        }
    }

    #[inline]
    pub(crate) fn set_normalized(&mut self, normalized: bool) {
        if !normalized {
            self.data[0] |= NOT_NORMALIZED;
        } else {
            self.data[0] &= !NOT_NORMALIZED;
        }
    }

    #[inline]
    pub(crate) fn extend(&mut self, other: AtomView<'_>) {
        self.data[0] |= NOT_NORMALIZED;
        self.data[0] |= alias_flag_for(other);
        merge_aliases(&mut self.aliases, &aliases_from_view(other));
        self.data.sync_aliases_from(&self.aliases);

        let mut c = &self.data[1..];

        let mut n_args;
        (n_args, _, c) = c.get_frac_u64();

        let old_header_size = unsafe { c.as_ptr().offset_from(self.data.as_ptr()) } as usize;

        match other {
            AtomView::Add(m) => {
                let mut sd = &m.data[1..];
                let sub_n_args;
                (sub_n_args, _, sd) = sd.get_frac_u64();

                n_args += sub_n_args;
                self.data.extend_from_slice(sd);
            }
            _ => {
                n_args += 1;
                self.data.extend_from_slice(other.get_data());
            }
        };

        let new_len = self.data.len() - old_header_size;
        let new_header_size = (n_args, new_len as u64).get_packed_size() as usize + 1;

        match new_header_size.cmp(&old_header_size) {
            Ordering::Equal => {}
            Ordering::Less => {
                self.data.copy_within(old_header_size.., new_header_size);
                let resized_len = self.data.len() - old_header_size + new_header_size;
                self.data.resize(resized_len, 0);
            }
            Ordering::Greater => {
                let old_len = self.data.len();
                self.data
                    .resize(old_len + new_header_size - old_header_size, 0);
                self.data
                    .copy_within(old_header_size..old_len, new_header_size);
            }
        }

        (n_args, new_len as u64).write_packed_fixed(&mut self.data[1..new_header_size]);
    }

    #[inline(always)]
    pub fn to_add_view(&self) -> AddView<'_> {
        AddView {
            data: &self.data,
            aliases: &self.aliases,
        }
    }

    #[inline(always)]
    pub fn set_from_view(&mut self, view: AddView) {
        self.data.clear();
        self.data.extend(view.data);
        self.aliases = aliases_from_view(AtomView::Add(view));
        self.data.sync_aliases_from(&self.aliases);
    }

    #[inline(always)]
    pub fn as_view(&self) -> AtomView<'_> {
        AtomView::Add(self.to_add_view())
    }

    #[inline(always)]
    pub fn get_nargs(&self) -> usize {
        self.to_add_view().get_nargs()
    }

    #[inline(always)]
    pub fn into_raw(mut self) -> RawAtom {
        self.data.sync_aliases_from(&self.aliases);
        self.data
    }

    #[inline(always)]
    pub(crate) unsafe fn from_raw(raw: RawAtom) -> Add {
        let aliases = raw.take_aliases_or_collect();
        Add { data: raw, aliases }
    }

    #[inline(always)]
    pub(crate) fn refresh_alias_flag_from_tree(&mut self) {
        let has_alias = self.to_add_view().iter().any(|arg| arg.has_alias());
        if has_alias {
            self.data[0] |= HAS_ALIAS_FLAG;
        } else {
            self.data[0] &= !HAS_ALIAS_FLAG;
        }
    }

    pub(crate) fn grow_capacity(&mut self, size: usize) {
        if size > self.data.capacity() {
            let additional = size - self.data.capacity();
            self.data.reserve(additional);
        }
    }
}

impl<'a> VarView<'a> {
    #[inline]
    pub fn to_owned(&self) -> Var {
        Var::from_view_into(self, RawAtom::new())
    }

    #[inline]
    pub fn clone_into(&self, target: &mut Var) {
        target.set_from_view(self);
    }

    #[inline]
    pub fn clone_into_raw(&self, mut buffer: RawAtom) -> Var {
        buffer.clear();
        buffer.extend(self.data);
        let aliases = aliases_from_view(AtomView::Var(*self));
        buffer.sync_aliases_from(&aliases);
        Var {
            data: buffer,
            aliases,
        }
    }

    #[inline(always)]
    pub fn get_symbol(&self) -> Symbol {
        let (id, attrs, _) = self.data[1..].get_frac_u64();

        // attrs are shifted to improve the packing efficiency
        Symbol::decode_flags(id as u32, self.data[0], (attrs >> 1) as u32)
    }

    #[inline(always)]
    pub fn get_symbol_id(&self) -> u32 {
        let (id_and_attrs, _, _) = self.data[1..].get_frac_u64();
        id_and_attrs as u32
    }

    #[inline(always)]
    pub fn get_wildcard_level(&self) -> u8 {
        self.get_symbol().get_wildcard_level()
    }

    #[inline]
    pub fn as_view(&self) -> AtomView<'a> {
        AtomView::Var(*self)
    }

    pub fn get_byte_size(&self) -> usize {
        self.data.len()
    }
}

/// A view of a [Var].
#[derive(Copy, Clone, Eq)]
pub struct VarView<'a> {
    data: &'a [u8],
    aliases: &'a Vec<Arc<AliasHandle>>,
}

impl<'b> PartialEq<VarView<'b>> for VarView<'_> {
    fn eq(&self, other: &VarView<'b>) -> bool {
        self.data == other.data
    }
}

impl Hash for VarView<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.hash(state);
    }
}

/// A view of an [Alias].
#[derive(Copy, Clone, Eq)]
pub struct AliasView<'a> {
    data: &'a [u8],
    aliases: &'a Vec<Arc<AliasHandle>>,
}

impl<'b> PartialEq<AliasView<'b>> for AliasView<'_> {
    fn eq(&self, other: &AliasView<'b>) -> bool {
        self.data == other.data
    }
}

impl Hash for AliasView<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.hash(state);
    }
}

impl<'a> AliasView<'a> {
    #[inline]
    pub fn to_owned(&self) -> Alias {
        Alias::from_view_into(self, RawAtom::new())
    }

    #[inline]
    pub fn clone_into(&self, target: &mut Alias) {
        target.set_from_view(self);
    }

    #[inline]
    pub fn clone_into_raw(&self, mut buffer: RawAtom) -> Alias {
        buffer.clear();
        buffer.extend(self.data);
        let aliases = aliases_from_view(AtomView::Alias(*self));
        buffer.sync_aliases_from(&aliases);
        Alias {
            data: buffer,
            aliases,
        }
    }

    #[inline(always)]
    pub fn get_token(&self) -> usize {
        self.data[1..].get_frac_u64().0 as usize
    }

    #[inline(always)]
    pub fn get_handle(&self) -> Arc<AliasHandle> {
        self.aliases
            .iter()
            .find(|h| h.token() == self.get_token())
            .cloned()
            .or_else(|| get_alias_handle(self.get_token()))
            .expect("Alias handle was released before the alias atom")
    }

    #[inline(always)]
    pub fn get_body(&self) -> AtomView<'a> {
        self.aliases
            .iter()
            .find(|h| h.token() == self.get_token())
            .expect("Alias handle was released before the alias atom")
            .atom()
            .as_view()
    }

    #[inline(always)]
    pub fn is_opaque(&self) -> bool {
        self.data[0] & ALIAS_OPAQUE_FLAG != 0
    }

    #[inline]
    pub fn as_view(&self) -> AtomView<'a> {
        AtomView::Alias(*self)
    }

    pub fn get_byte_size(&self) -> usize {
        self.data.len()
    }
}

/// A view of a [Fun].
#[derive(Copy, Clone, Eq)]
pub struct FunView<'a> {
    data: &'a [u8],
    aliases: &'a Vec<Arc<AliasHandle>>,
}

impl<'b> PartialEq<FunView<'b>> for FunView<'_> {
    fn eq(&self, other: &FunView<'b>) -> bool {
        self.data == other.data
    }
}

impl Hash for FunView<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.hash(state);
    }
}

impl<'a> IntoIterator for FunView<'a> {
    type Item = AtomView<'a>;
    type IntoIter = ListIterator<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoIterator for &FunView<'a> {
    type Item = AtomView<'a>;
    type IntoIter = ListIterator<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> FunView<'a> {
    pub fn to_owned(&self) -> Fun {
        Fun::from_view_into(self, RawAtom::new())
    }

    pub fn clone_into(&self, target: &mut Fun) {
        target.set_from_view(self);
    }

    pub fn clone_into_raw(&self, mut buffer: RawAtom) -> Fun {
        buffer.clear();
        buffer.extend(self.data);
        let aliases = aliases_from_view(AtomView::Fun(*self));
        buffer.sync_aliases_from(&aliases);
        Fun {
            data: buffer,
            aliases,
        }
    }

    #[inline(always)]
    pub fn get_symbol(&self) -> Symbol {
        let (id_and_attrs, _, _) = self.data[1 + 4..].get_frac_u64();
        Symbol::decode_flags(
            id_and_attrs as u32,
            self.data[0],
            (id_and_attrs >> 32) as u32,
        )
    }

    /// Get the symbol ID of the function. Slightly faster than [get_symbol](Self::get_symbol) if only the ID is needed.
    #[inline(always)]
    pub fn get_symbol_id(&self) -> u32 {
        let (id_and_attrs, _, _) = self.data[1 + 4..].get_frac_u64();
        id_and_attrs as u32
    }

    /// Get the argument at the given index.
    pub fn get(&self, index: usize) -> AtomView<'a> {
        if let Some(v) = self.iter().nth(index) {
            v
        } else {
            panic!(
                "Index {} out of bounds for function {}",
                index,
                self.as_view()
            );
        }
    }

    #[inline(always)]
    pub fn is_symmetric(&self) -> bool {
        self.data[0] & SYM_CYCLESYMMETRIC_FLAG == SYM_SYMMETRIC_FLAG
    }

    #[inline(always)]
    pub fn is_antisymmetric(&self) -> bool {
        self.data[0] & SYM_CYCLESYMMETRIC_FLAG == SYM_ANTISYMMETRIC_FLAG
    }

    #[inline(always)]
    pub fn is_cyclesymmetric(&self) -> bool {
        self.data[0] & SYM_CYCLESYMMETRIC_FLAG == SYM_CYCLESYMMETRIC_FLAG
    }

    #[inline(always)]
    pub fn is_linear(&self) -> bool {
        self.get_symbol().is_linear()
    }

    #[inline(always)]
    pub fn get_wildcard_level(&self) -> u8 {
        self.get_symbol().get_wildcard_level()
    }

    #[inline(always)]
    pub fn get_nargs(&self) -> usize {
        self.data[1 + 4..].get_frac_u64().1 as usize
    }

    #[inline(always)]
    pub(crate) fn is_normalized(&self) -> bool {
        (self.data[0] & NOT_NORMALIZED) == 0
    }

    #[inline(always)]
    pub(crate) fn has_alias(&self) -> bool {
        (self.data[0] & HAS_ALIAS_FLAG) != 0
    }

    #[inline]
    pub fn iter(&self) -> ListIterator<'a> {
        let mut c = self.data;
        c.get_u8();
        c.get_u32_le(); // size

        let n_args;
        (_, n_args, c) = c.get_frac_u64(); // name

        ListIterator {
            data: c,
            length: n_args as u32,
            aliases: self.aliases,
        }
    }

    pub fn as_view(&self) -> AtomView<'a> {
        AtomView::Fun(*self)
    }

    pub fn to_slice(&self) -> ListSlice<'a> {
        let mut c = self.data;
        c.get_u8();
        c.get_u32_le(); // size

        let n_args;
        (_, n_args, c) = c.get_frac_u64(); // name

        ListSlice {
            data: c,
            length: n_args as usize,
            slice_type: SliceType::Arg,
            aliases: self.aliases,
        }
    }

    pub fn get_byte_size(&self) -> usize {
        self.data.len()
    }

    pub(crate) fn fast_cmp(&self, other: FunView) -> Ordering {
        self.data.cmp(other.data)
    }
}

/// A view of a [Num].
#[derive(Copy, Clone, Eq)]
pub struct NumView<'a> {
    data: &'a [u8],
    aliases: &'a Vec<Arc<AliasHandle>>,
}

impl<'b> PartialEq<NumView<'b>> for NumView<'_> {
    #[inline]
    fn eq(&self, other: &NumView<'b>) -> bool {
        self.data == other.data
    }
}

impl Hash for NumView<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.hash(state);
    }
}

impl<'a> NumView<'a> {
    #[inline]
    pub fn to_owned(&self) -> Num {
        Num::from_view_into(self, RawAtom::new())
    }

    #[inline]
    pub fn clone_into(&self, target: &mut Num) {
        target.set_from_view(self);
    }

    #[inline]
    pub fn clone_into_raw(&self, mut buffer: RawAtom) -> Num {
        buffer.clear();
        buffer.extend(self.data);
        let aliases = aliases_from_view(AtomView::Num(*self));
        buffer.sync_aliases_from(&aliases);
        Num {
            data: buffer,
            aliases,
        }
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        if self.data.is_small_int() {
            self.data.is_zero_rat()
        } else {
            self.get_coeff_view().is_zero()
        }
    }

    #[inline]
    pub fn is_one(&self) -> bool {
        if self.data.is_small_int() {
            self.data.is_one_rat()
        } else {
            self.get_coeff_view().is_one()
        }
    }

    #[inline]
    pub fn get_coeff_view(&self) -> CoefficientView<'a> {
        self.data[1..].get_coeff_view().0
    }

    pub fn as_view(&self) -> AtomView<'a> {
        AtomView::Num(*self)
    }

    pub fn get_byte_size(&self) -> usize {
        self.data.len()
    }
}

/// A view of a [Pow].
#[derive(Copy, Clone, Eq)]
pub struct PowView<'a> {
    data: &'a [u8],
    aliases: &'a Vec<Arc<AliasHandle>>,
}

impl<'a> IntoIterator for PowView<'a> {
    type Item = AtomView<'a>;
    type IntoIter = ListIterator<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoIterator for &PowView<'a> {
    type Item = AtomView<'a>;
    type IntoIter = ListIterator<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'b> PartialEq<PowView<'b>> for PowView<'_> {
    #[inline]
    fn eq(&self, other: &PowView<'b>) -> bool {
        self.data == other.data
    }
}

impl Hash for PowView<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.hash(state);
    }
}

impl<'a> PowView<'a> {
    #[inline]
    pub fn to_owned(&self) -> Pow {
        Pow::from_view_into(self, RawAtom::new())
    }

    #[inline]
    pub fn clone_into(&self, target: &mut Pow) {
        target.set_from_view(self);
    }

    #[inline]
    pub fn clone_into_raw(&self, mut buffer: RawAtom) -> Pow {
        buffer.clear();
        buffer.extend(self.data);
        let aliases = aliases_from_view(AtomView::Pow(*self));
        buffer.sync_aliases_from(&aliases);
        Pow {
            data: buffer,
            aliases,
        }
    }

    #[inline]
    pub fn get_base(&self) -> AtomView<'a> {
        let (b, _) = self.get_base_exp();
        b
    }

    #[inline]
    pub fn get_exp(&self) -> AtomView<'a> {
        let (_, e) = self.get_base_exp();
        e
    }

    #[inline]
    pub(crate) fn is_normalized(&self) -> bool {
        (self.data[0] & NOT_NORMALIZED) == 0
    }

    #[inline]
    pub(crate) fn has_alias(&self) -> bool {
        let (base, exp) = self.get_base_exp();
        base.has_alias() || exp.has_alias()
    }

    #[inline]
    pub fn get_base_exp(&self) -> (AtomView<'a>, AtomView<'a>) {
        let mut it = self.iter();

        (it.next().unwrap(), it.next().unwrap())
    }

    #[inline]
    pub fn iter(&self) -> ListIterator<'a> {
        ListIterator {
            data: &self.data[1..],
            length: 2,
            aliases: self.aliases,
        }
    }

    #[inline]
    pub fn as_view(&self) -> AtomView<'a> {
        AtomView::Pow(*self)
    }

    #[inline]
    pub fn to_slice(&self) -> ListSlice<'a> {
        ListSlice {
            data: &self.data[1..],
            length: 2,
            slice_type: SliceType::Pow,
            aliases: self.aliases,
        }
    }

    pub fn get_byte_size(&self) -> usize {
        self.data.len()
    }
}

/// A view of a [Mul].
#[derive(Copy, Clone, Eq)]
pub struct MulView<'a> {
    data: &'a [u8],
    aliases: &'a Vec<Arc<AliasHandle>>,
}

impl<'b> PartialEq<MulView<'b>> for MulView<'_> {
    #[inline]
    fn eq(&self, other: &MulView<'b>) -> bool {
        self.data == other.data
    }
}

impl Hash for MulView<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.hash(state);
    }
}

impl<'a> IntoIterator for MulView<'a> {
    type Item = AtomView<'a>;
    type IntoIter = ListIterator<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoIterator for &MulView<'a> {
    type Item = AtomView<'a>;
    type IntoIter = ListIterator<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> MulView<'a> {
    #[inline]
    pub fn to_owned(&self) -> Mul {
        Mul::from_view_into(self, RawAtom::new())
    }

    #[inline]
    pub fn clone_into(&self, target: &mut Mul) {
        target.set_from_view(self);
    }

    #[inline]
    pub fn clone_into_raw(&self, mut buffer: RawAtom) -> Mul {
        buffer.clear();
        buffer.extend(self.data);
        let aliases = aliases_from_view(AtomView::Mul(*self));
        buffer.sync_aliases_from(&aliases);
        Mul {
            data: buffer,
            aliases,
        }
    }

    #[inline]
    pub(crate) fn is_normalized(&self) -> bool {
        (self.data[0] & NOT_NORMALIZED) == 0
    }

    #[inline(always)]
    pub(crate) fn has_alias(&self) -> bool {
        (self.data[0] & MUL_HAS_ALIAS_FLAG) != 0
    }

    pub fn get_nargs(&self) -> usize {
        self.data[1 + 4..].get_frac_u64().0 as usize
    }

    #[inline]
    pub fn iter(&self) -> ListIterator<'a> {
        let mut c = self.data;
        c.get_u8();
        c.get_u32_le(); // size

        let n_args;
        (n_args, _, c) = c.get_frac_u64();

        ListIterator {
            data: c,
            length: n_args as u32,
            aliases: self.aliases,
        }
    }

    #[inline]
    pub fn as_view(&self) -> AtomView<'a> {
        AtomView::Mul(*self)
    }

    pub fn to_slice(&self) -> ListSlice<'a> {
        let mut c = self.data;
        c.get_u8();
        c.get_u32_le(); // size

        let n_args;
        (n_args, _, c) = c.get_frac_u64();

        ListSlice {
            data: c,
            length: n_args as usize,
            slice_type: SliceType::Mul,
            aliases: self.aliases,
        }
    }

    #[inline]
    pub fn has_coefficient(&self) -> bool {
        self.data[0] & MUL_HAS_COEFF_FLAG == MUL_HAS_COEFF_FLAG
    }

    pub fn get_byte_size(&self) -> usize {
        self.data.len()
    }
}

/// A view of a [Add].
#[derive(Copy, Clone, Eq)]
pub struct AddView<'a> {
    data: &'a [u8],
    aliases: &'a Vec<Arc<AliasHandle>>,
}

impl<'b> PartialEq<AddView<'b>> for AddView<'_> {
    #[inline]
    fn eq(&self, other: &AddView<'b>) -> bool {
        self.data == other.data
    }
}

impl Hash for AddView<'_> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.hash(state);
    }
}

macro_rules! impl_view_debug {
    ($view:ident) => {
        impl std::fmt::Debug for $view<'_> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.debug_struct(stringify!($view))
                    .field("data", &self.data)
                    .finish()
            }
        }
    };
}

impl_view_debug!(VarView);
impl_view_debug!(AliasView);
impl_view_debug!(FunView);
impl_view_debug!(NumView);
impl_view_debug!(PowView);
impl_view_debug!(MulView);
impl_view_debug!(AddView);

impl<'a> IntoIterator for AddView<'a> {
    type Item = AtomView<'a>;
    type IntoIter = ListIterator<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoIterator for &AddView<'a> {
    type Item = AtomView<'a>;
    type IntoIter = ListIterator<'a>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> AddView<'a> {
    pub fn to_owned(&self) -> Add {
        Add::from_view_into(self, RawAtom::new())
    }

    pub fn clone_into(&self, target: &mut Add) {
        target.set_from_view(*self);
    }

    pub fn clone_into_raw(&self, mut buffer: RawAtom) -> Add {
        buffer.clear();
        buffer.extend(self.data);
        let aliases = aliases_from_view(AtomView::Add(*self));
        buffer.sync_aliases_from(&aliases);
        Add {
            data: buffer,
            aliases,
        }
    }

    #[inline(always)]
    pub(crate) fn is_normalized(&self) -> bool {
        (self.data[0] & NOT_NORMALIZED) == 0
    }

    #[inline(always)]
    pub(crate) fn has_alias(&self) -> bool {
        (self.data[0] & HAS_ALIAS_FLAG) != 0
    }

    #[inline(always)]
    pub fn get_nargs(&self) -> usize {
        self.data[1..].get_frac_u64().0 as usize
    }

    #[inline]
    pub fn iter(&self) -> ListIterator<'a> {
        let mut c = self.data;
        c.get_u8();

        let n_args;
        (n_args, _, c) = c.get_frac_u64();

        ListIterator {
            data: c,
            length: n_args as u32,
            aliases: self.aliases,
        }
    }

    #[inline]
    pub fn as_view(&self) -> AtomView<'a> {
        AtomView::Add(*self)
    }

    pub fn to_slice(&self) -> ListSlice<'a> {
        let mut c = self.data;
        c.get_u8();

        let n_args;
        (n_args, _, c) = c.get_frac_u64();

        ListSlice {
            data: c,
            length: n_args as usize,
            slice_type: SliceType::Add,
            aliases: self.aliases,
        }
    }

    pub fn get_byte_size(&self) -> usize {
        self.data.len()
    }
}

impl<'a> AtomView<'a> {
    pub const ZERO: Self = Self::Num(NumView {
        data: &ZERO_DATA,
        aliases: &NO_ALIASES,
    });

    pub fn from(source: &'a [u8]) -> AtomView<'a> {
        match source[0] & TYPE_MASK {
            VAR_ID => AtomView::Var(VarView {
                data: source,
                aliases: &NO_ALIASES,
            }),
            FUN_ID => AtomView::Fun(FunView {
                data: source,
                aliases: &NO_ALIASES,
            }),
            NUM_ID => AtomView::Num(NumView {
                data: source,
                aliases: &NO_ALIASES,
            }),
            POW_ID => AtomView::Pow(PowView {
                data: source,
                aliases: &NO_ALIASES,
            }),
            MUL_ID => AtomView::Mul(MulView {
                data: source,
                aliases: &NO_ALIASES,
            }),
            ADD_ID => AtomView::Add(AddView {
                data: source,
                aliases: &NO_ALIASES,
            }),
            ALIAS_ID => AtomView::Alias(AliasView {
                data: source,
                aliases: &NO_ALIASES,
            }),
            x => unreachable!("Bad id: {}", x),
        }
    }

    #[inline(always)]
    pub fn get_data(&self) -> &'a [u8] {
        match self {
            AtomView::Num(n) => n.data,
            AtomView::Var(v) => v.data,
            AtomView::Fun(f) => f.data,
            AtomView::Alias(a) => a.data,
            AtomView::Pow(p) => p.data,
            AtomView::Mul(t) => t.data,
            AtomView::Add(e) => e.data,
        }
    }

    #[inline(always)]
    fn aliases_vec(&self) -> &'a Vec<Arc<AliasHandle>> {
        match self {
            AtomView::Num(n) => n.aliases,
            AtomView::Var(v) => v.aliases,
            AtomView::Fun(f) => f.aliases,
            AtomView::Alias(a) => a.aliases,
            AtomView::Pow(p) => p.aliases,
            AtomView::Mul(t) => t.aliases,
            AtomView::Add(e) => e.aliases,
        }
    }

    #[inline(always)]
    pub(crate) fn has_alias(&self) -> bool {
        match self {
            AtomView::Num(_) | AtomView::Var(_) => false,
            AtomView::Alias(_) => true,
            AtomView::Fun(f) => f.has_alias(),
            AtomView::Pow(p) => p.has_alias(),
            AtomView::Mul(m) => m.has_alias(),
            AtomView::Add(a) => a.has_alias(),
        }
    }

    /// Export the atom and state to a binary stream. It can be loaded
    /// with [Atom::import].
    #[inline(always)]
    pub fn export<W: Write>(&self, dest: &mut W) -> Result<(), std::io::Error> {
        State::export(dest)?;

        dest.write_u64::<LittleEndian>(ALIAS_EXPORT_SECTION_MAGIC)?;

        let aliases = collect_alias_handles_with_dependencies(*self);
        dest.write_u64::<LittleEndian>(aliases.len() as u64)?;
        for alias in aliases {
            dest.write_u64::<LittleEndian>(alias.token() as u64)?;
            alias.atom().as_view().write(dest.by_ref())?;
        }

        dest.write_u64::<LittleEndian>(1)?; // export a single expression

        let d = self.get_data();
        dest.write_u8(0)?;
        dest.write_u64::<LittleEndian>(d.len() as u64)?;
        dest.write_all(d)
    }

    /// Write the expression to a binary stream. The byte-length is written first,
    /// followed by the data. To import the expression in new session, also export the [`State`].
    ///
    /// Most users will want to use [AtomView::export] instead.
    #[inline(always)]
    pub fn write<W: Write>(&self, dest: &mut W) -> Result<(), std::io::Error> {
        let d = self.get_data();
        dest.write_u8(0)?;
        dest.write_u64::<LittleEndian>(d.len() as u64)?;
        dest.write_all(d)
    }

    pub(crate) fn rename(&self, state_map: &StateMap) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut a = ws.new_atom();
            self.rename_no_norm(state_map, ws, &mut a);
            let mut r = Atom::new();
            a.as_view().normalize(ws, &mut r);
            r
        })
    }

    fn rename_no_norm(&self, state_map: &StateMap, ws: &Workspace, out: &mut Atom) {
        match self {
            AtomView::Num(n) => match n.get_coeff_view() {
                CoefficientView::FiniteField(e, i) => {
                    if let Some(s) = state_map.finite_fields.get(&i) {
                        out.to_num(Coefficient::FiniteField(e, *s));
                    } else {
                        out.set_from_view(self);
                    }
                }
                CoefficientView::RationalPolynomial(r) => {
                    let (old_id, _, _) = r.0.get_frac_u64();

                    if let Some(nv) = state_map.variables_lists.get(&old_id) {
                        let mut rr = r.deserialize();
                        rr.numerator.variables = nv.clone();
                        rr.denominator.variables = nv.clone();
                        out.to_num(Coefficient::RationalPolynomial(rr));
                    } else {
                        out.set_from_view(self);
                    }
                }
                _ => out.set_from_view(self),
            },
            AtomView::Var(v) => {
                if let Some(s) = state_map.symbols.get(&v.get_symbol_id()) {
                    out.to_var(*s);
                } else {
                    out.set_from_view(self);
                }
            }
            AtomView::Alias(a) => {
                out.set_from_view(&AtomView::Alias(*a));
            }
            AtomView::Fun(f) => {
                if let Some(s) = state_map.symbols.get(&f.get_symbol_id()) {
                    let nf = out.to_fun(*s);

                    let mut na = ws.new_atom();
                    for a in f {
                        a.rename_no_norm(state_map, ws, &mut na);
                        nf.add_arg(na.as_view());
                    }
                } else {
                    out.set_from_view(self);
                }
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();

                let mut nb = ws.new_atom();
                b.rename_no_norm(state_map, ws, &mut nb);
                let mut ne = ws.new_atom();
                e.rename_no_norm(state_map, ws, &mut ne);

                out.to_pow(nb.as_view(), ne.as_view());
            }
            AtomView::Mul(m) => {
                let nm = out.to_mul();

                let mut na = ws.new_atom();
                for a in m {
                    a.rename_no_norm(state_map, ws, &mut na);
                    nm.extend(na.as_view());
                }
            }
            AtomView::Add(add) => {
                let nm = out.to_add();

                let mut na = ws.new_atom();
                for a in add {
                    a.rename_no_norm(state_map, ws, &mut na);
                    nm.extend(na.as_view());
                }
            }
        }
    }
}

/// An iterator of a list of atoms.
#[derive(Debug, Copy, Clone)]
pub struct ListIterator<'a> {
    data: &'a [u8],
    length: u32,
    aliases: &'a Vec<Arc<AliasHandle>>,
}

impl<'a> Iterator for ListIterator<'a> {
    type Item = AtomView<'a>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.length == 0 {
            return None;
        }

        self.length -= 1;

        let start = self.data;

        let start_id = self.data.get_u8() & TYPE_MASK;
        let mut cur_id = start_id;

        // store how many more atoms to read
        // can be used instead of storing the byte length of an atom
        let mut skip_count = 1;
        loop {
            match cur_id {
                NUM_ID | VAR_ID | ALIAS_ID => {
                    self.data = self.data.skip_rational();
                }
                FUN_ID | MUL_ID => {
                    let n_size = self.data.get_u32_le();
                    self.data.advance(n_size as usize);
                }
                ADD_ID => {
                    let (_, size, np) = self.data.get_frac_u64();
                    self.data = np;
                    self.data.advance(size as usize);
                }
                POW_ID => {
                    skip_count += 2;
                }
                _ => unreachable!("Bad id"),
            }

            skip_count -= 1;

            if skip_count == 0 {
                break;
            }

            cur_id = self.data.get_u8() & TYPE_MASK;
        }

        let len = unsafe { self.data.as_ptr().offset_from(start.as_ptr()) } as usize;

        let data = unsafe { start.get_unchecked(..len) };
        match start_id {
            NUM_ID => Some(AtomView::Num(NumView {
                data,
                aliases: self.aliases,
            })),
            VAR_ID => Some(AtomView::Var(VarView {
                data,
                aliases: self.aliases,
            })),
            FUN_ID => Some(AtomView::Fun(FunView {
                data,
                aliases: self.aliases,
            })),
            MUL_ID => Some(AtomView::Mul(MulView {
                data,
                aliases: self.aliases,
            })),
            ADD_ID => Some(AtomView::Add(AddView {
                data,
                aliases: self.aliases,
            })),
            POW_ID => Some(AtomView::Pow(PowView {
                data,
                aliases: self.aliases,
            })),
            ALIAS_ID => Some(AtomView::Alias(AliasView {
                data,
                aliases: self.aliases,
            })),
            x => unreachable!("Bad id {}", x),
        }
    }
}

impl<'a> ExactSizeIterator for ListIterator<'a> {
    #[inline]
    fn len(&self) -> usize {
        self.length as usize
    }
}

impl<'a, const N: usize> TryInto<[AtomView<'a>; N]> for ListIterator<'a> {
    type Error = &'static str;

    fn try_into(self) -> Result<[AtomView<'a>; N], Self::Error> {
        if self.len() != N {
            return Err("Iterator does not contain the expected number of atoms");
        }

        let mut it = self;
        Ok(std::array::from_fn(|_| {
            it.next()
                .expect("ListIterator length was checked before array conversion")
        }))
    }
}

impl<'a> ListIterator<'a> {
    #[inline]
    pub fn len(&self) -> usize {
        self.length as usize
    }

    #[inline]
    pub fn from_one(atom: AtomView<'a>) -> Self {
        ListIterator {
            data: atom.get_data(),
            length: 1,
            aliases: atom.aliases_vec(),
        }
    }
}

/// A slice of a list of atoms.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct ListSlice<'a> {
    data: &'a [u8],
    length: usize,
    slice_type: SliceType,
    aliases: &'a Vec<Arc<AliasHandle>>,
}

impl<'a> ListSlice<'a> {
    #[inline(always)]
    fn skip(mut pos: &[u8], n: u32) -> &[u8] {
        // store how many more atoms to read
        // can be used instead of storing the byte length of an atom
        let mut skip_count = n;
        while skip_count > 0 {
            skip_count -= 1;

            let atom_type = unsafe { *pos.get_unchecked(0) & TYPE_MASK };
            pos = unsafe { pos.get_unchecked(1..) };
            match atom_type {
                NUM_ID | VAR_ID | ALIAS_ID => {
                    pos = pos.skip_rational();
                }
                FUN_ID | MUL_ID => {
                    let n_size = unsafe {
                        u32::from_le_bytes([
                            *pos.get_unchecked(0),
                            *pos.get_unchecked(1),
                            *pos.get_unchecked(2),
                            *pos.get_unchecked(3),
                        ])
                    };

                    pos = unsafe { pos.get_unchecked(n_size as usize + 4..) };
                }
                ADD_ID => {
                    let (_, size, np) = pos.get_frac_u64();
                    pos = np;

                    pos = unsafe { pos.get_unchecked(size as usize..) };
                }
                POW_ID => {
                    skip_count += 2;
                }
                _ => unreachable!("Bad id"),
            }
        }
        pos
    }

    #[inline]
    pub fn fast_forward(&self, index: usize) -> ListSlice<'a> {
        if index == 0 {
            return *self;
        }

        let mut pos = self.data;

        pos = Self::skip(pos, index as u32);

        ListSlice {
            data: pos,
            length: self.length - index,
            slice_type: self.slice_type,
            aliases: self.aliases,
        }
    }

    fn get_entry<'b>(
        start: &'b [u8],
        aliases: &'b Vec<Arc<AliasHandle>>,
    ) -> (AtomView<'b>, &'b [u8]) {
        let start_id = start[0] & TYPE_MASK;
        let end = Self::skip(start, 1);
        let len = unsafe { end.as_ptr().offset_from(start.as_ptr()) } as usize;

        let data = unsafe { start.get_unchecked(..len) };
        (
            match start_id {
                NUM_ID => AtomView::Num(NumView { data, aliases }),
                VAR_ID => AtomView::Var(VarView { data, aliases }),
                FUN_ID => AtomView::Fun(FunView { data, aliases }),
                MUL_ID => AtomView::Mul(MulView { data, aliases }),
                ADD_ID => AtomView::Add(AddView { data, aliases }),
                POW_ID => AtomView::Pow(PowView { data, aliases }),
                ALIAS_ID => AtomView::Alias(AliasView { data, aliases }),
                x => unreachable!("Bad id {}", x),
            },
            end,
        )
    }

    #[inline]
    pub fn pop_first(&self) -> (AtomView<'a>, ListSlice<'a>) {
        let (res, end) = Self::get_entry(self.data, self.aliases);

        let slice = ListSlice {
            data: end,
            length: self.length - 1,
            slice_type: self.slice_type,
            aliases: self.aliases,
        };

        (res, slice)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.length
    }

    #[inline]
    pub fn get(&self, index: usize) -> AtomView<'a> {
        let start = self.fast_forward(index);
        Self::get_entry(start.data, start.aliases).0
    }

    pub fn get_subslice(&self, range: std::ops::Range<usize>) -> Self {
        let start = self.fast_forward(range.start);

        let mut s = start.data;
        s = Self::skip(s, range.len() as u32);

        let len = unsafe { s.as_ptr().offset_from(start.data.as_ptr()) } as usize;
        ListSlice {
            data: &start.data[..len],
            length: range.len(),
            slice_type: self.slice_type,
            aliases: self.aliases,
        }
    }

    #[inline]
    pub fn get_type(&self) -> SliceType {
        self.slice_type
    }

    #[inline]
    pub fn from_one(view: AtomView<'a>) -> Self {
        ListSlice {
            data: view.get_data(),
            length: 1,
            slice_type: SliceType::One,
            aliases: view.aliases_vec(),
        }
    }

    #[inline]
    pub fn empty() -> Self {
        ListSlice {
            data: &[],
            length: 0,
            slice_type: SliceType::Empty,
            aliases: &NO_ALIASES,
        }
    }

    #[inline]
    pub fn iter(&self) -> ListSliceIterator<'a> {
        ListSliceIterator { data: *self }
    }

    #[inline]
    pub(crate) fn get_data(&self) -> &'a [u8] {
        self.data
    }
}

/// An iterator of a slice of atoms.
pub struct ListSliceIterator<'a> {
    data: ListSlice<'a>,
}

impl<'a> Iterator for ListSliceIterator<'a> {
    type Item = AtomView<'a>;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.data.length > 0 {
            let (res, end) = ListSlice::get_entry(self.data.data, self.data.aliases);
            self.data = ListSlice {
                data: end,
                length: self.data.length - 1,
                slice_type: self.data.slice_type,
                aliases: self.data.aliases,
            };

            Some(res)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{atom::AtomView, parse};

    #[test]
    fn list_iterator_try_into_array() {
        let expr = parse!("f(a,b,c)");
        let AtomView::Fun(f) = expr.as_view() else {
            panic!("expected function");
        };

        let [a, b, c]: [AtomView<'_>; 3] = f.iter().try_into().unwrap();
        assert_eq!(a.to_owned(), parse!("a"));
        assert_eq!(b.to_owned(), parse!("b"));
        assert_eq!(c.to_owned(), parse!("c"));

        let err: Result<[AtomView<'_>; 2], _> = f.iter().try_into();
        assert!(err.is_err());
    }
}
