//! Low-level representation of expressions.

mod compact;
pub use compact::CompactAtomView;

use ahash::HashMap;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use bytes::{Buf, BufMut};
use smartstring::alias::String;
use std::{
    borrow::Borrow,
    cmp::Ordering,
    hash::Hash,
    io::{Read, Write},
};

use crate::{
    atom::{UserData, UserDataKey},
    coefficient::{Coefficient, CoefficientView},
    domains::integer::Integer,
    state::{State, StateMap, Workspace},
    utils::Settable,
};

use super::{
    Atom, AtomOrView, AtomView, SliceType, Symbol,
    coefficient::{
        PackedRationalNumberReader, PackedRationalNumberWriter, read_packed_denominator,
        read_packed_numerator, read_packed_pair, read_packed_pair_checked, skip_packed_function,
        skip_packed_list, skip_packed_variable, validate_packed_coefficient,
    },
};

/// The current export format identifier for atom data.
// Versions 2–4 are reserved for historical error-tracking coefficient layouts.
const ATOM_EXPORT_FORMAT: u8 = 6;

const NUM_ID: u8 = 1;
const VAR_ID: u8 = 2;
const FUN_ID: u8 = 3;
const MUL_ID: u8 = 4;
const ADD_ID: u8 = 5;
const POW_ID: u8 = 6;
const TYPE_MASK: u8 = 0b00000_111;
const NOT_NORMALIZED: u8 = 0b10000_000;
const SYM_LINEAR_FLAG: u8 = 0b01000_000;
const SYM_SYMMETRIC_FLAG: u8 = 0b00100_000;
const SYM_ANTISYMMETRIC_FLAG: u8 = 0b00010_000;
/// Coded as symmetric | antisymmetric
const SYM_CYCLESYMMETRIC_FLAG: u8 = 0b00110_000;
const SYM_SCALAR_FLAG: u8 = 0b00001_000;
const SYM_EXTRA_REAL_FLAG: u32 = 0b01;
const SYM_EXTRA_INTEGER_FLAG: u32 = 0b10;
const SYM_EXTRA_POSITIVE_FLAG: u32 = 0b100;
const SYM_EXTRA_WILDCARD_LEVEL_MASK: u32 = 0b11_000;
const SYM_EXTRA_WILDCARD_LEVEL_1: u32 = 0b01_000;
const SYM_EXTRA_WILDCARD_LEVEL_2: u32 = 0b10_000;
const SYM_EXTRA_WILDCARD_LEVEL_3: u32 = 0b11_000;
const SYM_EXTRA_FLAT_FLAG: u32 = 0b1_00_000;

const MUL_HAS_COEFF_FLAG: u8 = 0b01000000;
// Lets algebra read both children of a compact variable power without scanning
// the variable's packed symbol ID. List traversal still uses ordinary children.
const POW_HAS_SMALL_INT_FLAG: u8 = 0b00001000;

// Upper numeric-header bits encode -15..=15 as codes 1..=31. Code zero
// escapes to the existing coefficient payload, including non-rational domains.
const fn inline_number_header(value: i64) -> u8 {
    (((value + 16) as u8) << 3) | NUM_ID
}

#[inline(always)]
const fn inline_number_value(header: u8) -> i64 {
    (header >> 3) as i64 - 16
}

const ZERO_DATA: [u8; 1] = [inline_number_header(0)];

/// Collapse the ordinary encoding only for exact, real integers in range.
#[inline]
fn canonicalize_number(data: &mut RawAtom) {
    if let [NUM_ID, sign @ (1 | 0x81), magnitude @ 0..=15] = data.as_slice() {
        let value = if *sign == 1 {
            *magnitude as i64
        } else {
            -(*magnitude as i64)
        };
        data[0] = inline_number_header(value);
        data.truncate(1);
    }
}

#[inline]
fn write_number(num: &Coefficient, data: &mut RawAtom) {
    if let Coefficient::Complex(c) = num
        && c.im.is_zero()
        && let (Integer::Single(value), Integer::Single(1)) =
            (c.re.numerator_ref(), c.re.denominator_ref())
        && (-15..=15).contains(value)
    {
        data.push(inline_number_header(*value));
        return;
    }
    data.push(NUM_ID);
    num.write_packed(data);
}

// Fun: packed(byte length, 1), packed(symbol/attributes, argument count), arguments.
// Mul/Add: packed(argument count, byte length), arguments.
// Byte lengths exclude their own header. All lengths use the full u64 range.
#[inline(always)]
fn resize_header(data: &mut Vec<u8>, old_end: usize, new_end: usize) {
    if old_end != new_end {
        resize_header_slow(data, old_end, new_end);
    }
}

#[cold]
fn resize_header_slow(data: &mut Vec<u8>, old_end: usize, new_end: usize) {
    match new_end.cmp(&old_end) {
        Ordering::Equal => {}
        Ordering::Less => {
            data.copy_within(old_end.., new_end);
            data.truncate(data.len() - (old_end - new_end));
        }
        Ordering::Greater => {
            let old_len = data.len();
            data.resize(old_len + (new_end - old_end), 0);
            data.copy_within(old_end..old_len, new_end);
        }
    }
}

#[inline(always)]
fn update_list_header(data: &mut Vec<u8>, old_end: usize, nargs: u64) {
    let size = (data.len() - old_end) as u64;
    if (nargs | size) <= u8::MAX as u64 && size != 1 {
        resize_header(data, old_end, 4);
        data[1..4].copy_from_slice(&[0x11, nargs as u8, size as u8]);
        return;
    }
    update_large_list_header(data, old_end, nargs, size);
}

#[cold]
fn update_large_list_header(data: &mut Vec<u8>, old_end: usize, nargs: u64, size: u64) {
    let header = (nargs, size);
    let new_end = 1 + header.get_packed_size() as usize;
    resize_header(data, old_end, new_end);
    header.write_packed_fixed(&mut data[1..new_end]);
}

#[inline(always)]
fn function_metadata(data: &[u8]) -> &[u8] {
    if data[1] == 1 {
        return &data[3..];
    }
    if data[1] == 2 {
        return &data[4..];
    }
    // The length is an unsigned packed integer with no denominator.
    let size = 1usize << (data[1] - 1);
    &data[2 + size..]
}

#[inline(always)]
fn update_function_size(data: &mut Vec<u8>, old_end: usize) {
    let size = (data.len() - old_end) as u64;
    if size <= u8::MAX as u64 {
        resize_header(data, old_end, 3);
        data[1] = 1;
        data[2] = size as u8;
        return;
    }
    if size <= u16::MAX as u64 {
        resize_header(data, old_end, 4);
        data[1] = 2;
        data[2..4].copy_from_slice(&(size as u16).to_le_bytes());
        return;
    }
    update_large_function_size(data, old_end, size);
}

#[cold]
fn update_large_function_size(data: &mut Vec<u8>, old_end: usize, size: u64) {
    let header = (size, 1);
    let new_end = 1 + header.get_packed_size() as usize;
    resize_header(data, old_end, new_end);
    header.write_packed_fixed(&mut data[1..new_end]);
}

/// Reject unsupported storage layouts before reading or interpreting atom data.
fn check_atom_format(format: u8) -> Result<(), std::io::Error> {
    if !matches!(format, 5 | ATOM_EXPORT_FORMAT) {
        if format == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Cannot load legacy atom format. Please export the expression using strings in the older version.",
            ));
        }

        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("Unsupported atom storage format {format}; expected {ATOM_EXPORT_FORMAT}"),
        ));
    }
    Ok(())
}

/// Check node boundaries and float payloads before constructing trusted views.
/// The explicit stack bounds nesting by input length without recursive calls.
#[cfg(test)]
fn validate_atom_data(data: &[u8]) -> Result<(), std::string::String> {
    validate_atom_data_format(data, ATOM_EXPORT_FORMAT)
}

fn validate_atom_data_format(data: &[u8], format: u8) -> Result<(), std::string::String> {
    fn variable_payload(data: &[u8]) -> Result<&[u8], std::string::String> {
        let (_, _, rest) = read_packed_pair_checked(data)?;
        // The variable scanner indexes its width table without masking a sign.
        if data[0] & 0x80 != 0 {
            return Err("Signed variable metadata is not canonical".to_owned());
        }
        Ok(rest)
    }

    struct Frame {
        end: usize,
        remaining: u64,
        exact_end: bool,
    }
    let mut frames = vec![Frame {
        end: data.len(),
        remaining: 1,
        exact_end: true,
    }];
    let mut position = 0;
    while let Some(frame) = frames.last_mut() {
        if frame.remaining == 0 {
            if frame.exact_end && position != frame.end {
                return Err("Atom payload length does not match its children".to_owned());
            }
            frames.pop();
            continue;
        }
        frame.remaining -= 1;
        let end = frame.end;
        let (&tag, rest) = data
            .get(position..end)
            .and_then(|source| source.split_first())
            .ok_or_else(|| "Truncated atom node".to_owned())?;
        position += 1;
        match tag & TYPE_MASK {
            NUM_ID => {
                if tag != NUM_ID {
                    if format == 5 {
                        return Err("Noncanonical numeric atom flags".to_owned());
                    }
                    // All nonzero upper-five-bit codes are complete integers.
                    continue;
                }
                let remaining = validate_packed_coefficient(rest)?;
                if format == ATOM_EXPORT_FORMAT {
                    let coefficient = &rest[..rest.len() - remaining.len()];
                    if let CoefficientView::Natural(value, 1, 0, 1) = coefficient.get_coeff_view().0
                        && (-15..=15).contains(&value)
                    {
                        return Err("Small integer must use its inline numeric header".to_owned());
                    }
                }
                position = end - remaining.len();
            }
            VAR_ID => {
                position = end - variable_payload(rest)?.len();
            }
            POW_ID => {
                if tag & !(TYPE_MASK | NOT_NORMALIZED | POW_HAS_SMALL_INT_FLAG) != 0 {
                    return Err("Noncanonical power flags".to_owned());
                }
                let exponent = if rest.first().is_some_and(|base| base & TYPE_MASK == VAR_ID) {
                    Some(variable_payload(&rest[1..])?)
                } else {
                    None
                };
                let inline_exponent = exponent.is_some_and(|exp| {
                    exp.first()
                        .is_some_and(|&header| header & TYPE_MASK == NUM_ID && header != NUM_ID)
                });
                if tag & POW_HAS_SMALL_INT_FLAG != 0 {
                    if format == 5 || !inline_exponent {
                        return Err(
                            "Compact power flag requires a variable and inline integer".to_owned()
                        );
                    }
                    // Consume exactly the variable and its one-byte exponent.
                    position = end - exponent.unwrap().len() + 1;
                } else {
                    if format == ATOM_EXPORT_FORMAT && inline_exponent {
                        return Err("Variable power is missing its compact flag".to_owned());
                    }
                    frames.push(Frame {
                        end,
                        remaining: 2,
                        exact_end: false,
                    });
                }
            }
            FUN_ID | MUL_ID | ADD_ID => {
                if tag & TYPE_MASK == FUN_ID && !matches!(rest.first(), Some(1..=4)) {
                    return Err("Invalid function size header".to_owned());
                }
                let (first, second, rest) = read_packed_pair_checked(rest)?;
                position = end - rest.len();
                let (size, mut count) = if tag & TYPE_MASK == FUN_ID {
                    if second != 1 {
                        return Err("Invalid function size header".to_owned());
                    }
                    (first, 0)
                } else {
                    (second, first)
                };
                let size = usize::try_from(size)
                    .map_err(|_| "Atom payload length does not fit in usize".to_owned())?;
                let payload_end = position
                    .checked_add(size)
                    .filter(|&payload_end| payload_end <= end)
                    .ok_or_else(|| "Truncated atom container".to_owned())?;
                if tag & TYPE_MASK == FUN_ID {
                    let (_, arguments, rest) =
                        read_packed_pair_checked(&data[position..payload_end])?;
                    count = arguments;
                    position = payload_end - rest.len();
                }
                frames.push(Frame {
                    end: payload_end,
                    remaining: count,
                    exact_end: true,
                });
            }
            _ => return Err("Unknown atom node type".to_owned()),
        }
    }
    Ok(())
}

/// Convert validated format-5 bytes without consulting symbol state. Every
/// composite is marked dirty for normalization after symbol remapping. Like
/// validation, conversion uses an explicit stack for deeply nested input.
fn convert_legacy_atom(mut data: &[u8]) -> RawAtom {
    struct Frame {
        start: usize,
        header_end: usize,
        count: u64,
        remaining: u64,
    }
    fn finish(data: &mut RawAtom, frame: Frame) {
        let tag = data[frame.start] & TYPE_MASK;
        if tag == POW_ID {
            let children = &data[frame.start + 1..];
            if children[0] & TYPE_MASK == VAR_ID {
                let exponent = unsafe { skip_packed_variable(&children[1..]) };
                if exponent.len() == 1 && exponent[0] & TYPE_MASK == NUM_ID && exponent[0] != NUM_ID
                {
                    data[frame.start] |= POW_HAS_SMALL_INT_FLAG;
                }
            }
        } else {
            let size = (data.len() - frame.header_end) as u64;
            let header = if tag == FUN_ID {
                (size, 1)
            } else {
                (frame.count, size)
            };
            let new_end = frame.start + 1 + header.get_packed_size() as usize;
            resize_header(data, frame.header_end, new_end);
            header.write_packed_fixed(&mut data[frame.start + 1..new_end]);
        }
    }

    let mut result = Vec::with_capacity(data.len());
    let mut frames: Vec<Frame> = Vec::new();
    loop {
        let header = data[0];
        let tag = header & TYPE_MASK;
        match tag {
            NUM_ID | VAR_ID => {
                let rest = data[1..].skip_rational();
                let node = &data[..data.len() - rest.len()];
                if tag == NUM_ID
                    && let CoefficientView::Natural(value, 1, 0, 1) = node[1..].get_coeff_view().0
                    && (-15..=15).contains(&value)
                {
                    result.push(inline_number_header(value));
                } else {
                    result.extend_from_slice(node);
                }
                data = rest;
            }
            POW_ID | FUN_ID | MUL_ID | ADD_ID => {
                let start = result.len();
                result.push(header | NOT_NORMALIZED);
                let count = if tag == POW_ID {
                    data = &data[1..];
                    2
                } else if tag == FUN_ID {
                    let metadata = function_metadata(data);
                    let (_, count, rest) = read_packed_pair(metadata);
                    result.extend([1, 0]);
                    let header_end = result.len();
                    result.extend_from_slice(&metadata[..metadata.len() - rest.len()]);
                    data = rest;
                    frames.push(Frame {
                        start,
                        header_end,
                        count,
                        remaining: count,
                    });
                    if count != 0 {
                        continue;
                    }
                    let frame = frames.pop().unwrap();
                    finish(&mut result, frame);
                    // An empty function is already complete.
                    0
                } else {
                    let (count, _, rest) = read_packed_pair(&data[1..]);
                    data = rest;
                    result.extend([0x11, 0, 0]);
                    count
                };
                if tag != FUN_ID {
                    let frame = Frame {
                        start,
                        header_end: result.len(),
                        count,
                        remaining: count,
                    };
                    if count != 0 {
                        frames.push(frame);
                        continue;
                    }
                    finish(&mut result, frame);
                }
            }
            _ => unreachable!("The format-5 atom was validated"),
        }
        loop {
            let Some(frame) = frames.last_mut() else {
                debug_assert!(data.is_empty());
                return result;
            };
            frame.remaining -= 1;
            if frame.remaining != 0 {
                break;
            }
            let frame = frames.pop().unwrap();
            finish(&mut result, frame);
        }
    }
}

/// The underlying slice of expression data.
pub type BorrowedRawAtom = [u8];
/// A raw atom that does not have explicit variant information.
pub type RawAtom = Vec<u8>;

impl Borrow<BorrowedRawAtom> for &Atom {
    fn borrow(&self) -> &BorrowedRawAtom {
        self.as_view().get_data()
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
        if self.is_linear {
            flags |= SYM_LINEAR_FLAG;
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

        if self.is_flat {
            extra |= SYM_EXTRA_FLAT_FLAG;
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
        let is_linear = (flags & SYM_LINEAR_FLAG) != 0;
        let is_scalar = (flags & SYM_SCALAR_FLAG) != 0;

        let is_real = (extra & SYM_EXTRA_REAL_FLAG) != 0;
        let is_integer = (extra & SYM_EXTRA_INTEGER_FLAG) != 0;
        let is_positive = (extra & SYM_EXTRA_POSITIVE_FLAG) != 0;
        let is_flat = (extra & SYM_EXTRA_FLAT_FLAG) != 0;
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
            is_flat,
            is_scalar,
            is_real,
            is_integer,
            is_positive,
            wildcard_level,
        }
    }
}

impl UserDataKey {
    /// Read a user-data key written by [`Self::write`]. Embedded expressions
    /// refer to the current symbol state; invalid tags, UTF-8, or truncated data
    /// produce an I/O error.
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
                let mut a = Atom::new();
                a.read(source)?;
                Ok(UserDataKey::Atom(a))
            }
            _ => Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Invalid UserDataKey tag",
            )),
        }
    }

    /// Write this key in binary form. Embedded expressions are written without
    /// the symbol state, which must be transferred separately across sessions.
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
    /// Read user data written by [`Self::write`], recursively decoding lists
    /// and maps. Embedded expressions require the corresponding symbol state.
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

    /// Write this value in binary form, recursively encoding lists and maps.
    /// Embedded expressions are written without the symbol state.
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

    /// Return the symbol represented by this variable.
    pub fn get_symbol(&self) -> Symbol {
        self.as_var_view().get_symbol()
    }

    /// Borrow the encoded expression bytes. Symbol metadata is stored separately.
    pub fn get_data(&self) -> &[u8] {
        &self.data[..self.size as usize]
    }

    /// Borrow a typed view without copying the underlying storage.
    pub fn as_var_view(&self) -> VarView<'_> {
        VarView {
            data: &self.data[..self.size as usize],
        }
    }

    /// Borrow this value as an expression view without copying its storage.
    pub fn as_view(&self) -> AtomView<'_> {
        AtomView::Var(VarView {
            data: &self.data[..self.size as usize],
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
    derive(bincode_trait_derive::BorrowDecodeFromDecode),
    trait_decode(trait = crate::state::HasStateMap)
)]
pub struct InlineNum {
    data: [u8; 24],
    size: u8,
}

#[cfg(feature = "bincode")]
impl<C: crate::state::HasStateMap> bincode::Decode<C> for InlineNum {
    fn decode<D: bincode::de::Decoder<Context = C>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let data = <[u8; 24] as bincode::Decode<C>>::decode(decoder)?;
        let size = <u8 as bincode::Decode<C>>::decode(decoder)?;
        if size == 0 || size as usize > data.len() || data[0] & TYPE_MASK != NUM_ID {
            return Err(bincode::error::DecodeError::Other("Invalid inline number"));
        }
        // Older versions derived the same array/length wire format, but kept
        // small integers in three bytes. Rebuild to preserve canonical Eq/Hash.
        if size == 3 {
            if let [NUM_ID, sign @ (1 | 0x81), magnitude @ 0..=15] = &data[..3] {
                let value = if *sign == 1 {
                    *magnitude as i64
                } else {
                    -(*magnitude as i64)
                };
                return Ok(Self::new(value, 1));
            }
        }
        Ok(Self { data, size })
    }
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
        if den == 1 && (-15..=15).contains(&num) {
            data[0] = inline_number_header(num);
            return InlineNum { data, size: 1 };
        }
        data[0] = NUM_ID;

        let size = 1 + (num, den).get_packed_size() as u8;
        (num, den).write_packed_fixed(&mut data[1..]);
        InlineNum { data, size }
    }

    /// Create the exact number zero in inline storage.
    pub const fn zero() -> InlineNum {
        let mut data = [0; 24];
        data[0] = inline_number_header(0);
        InlineNum { data, size: 1 }
    }

    /// Create the exact number one in inline storage.
    pub const fn one() -> InlineNum {
        let mut data = [0; 24];
        data[0] = inline_number_header(1);
        InlineNum { data, size: 1 }
    }

    /// Borrow the encoded expression bytes. Symbol metadata is stored separately.
    pub fn get_data(&self) -> &[u8] {
        &self.data[..self.size as usize]
    }

    /// Borrow a typed view without copying the underlying storage.
    pub fn as_num_view(&self) -> NumView<'_> {
        NumView {
            data: &self.data[..self.size as usize],
        }
    }

    /// Borrow this value as an expression view without copying its storage.
    pub fn as_view(&self) -> AtomView<'_> {
        AtomView::Num(NumView {
            data: &self.data[..self.size as usize],
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
        writer.write(&[ATOM_EXPORT_FORMAT])?;
        writer.write(&(d.len() as u64).to_le_bytes())?;
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
            // Equivalent to Atom::read; remapping follows below.
            let source = decoder.reader();

            let mut dest = Atom::Zero.into_raw();

            let mut flags_buf = [0; 1];
            let mut size_buf = [0; 8];

            source.read(&mut flags_buf)?;
            check_atom_format(flags_buf[0])
                .map_err(|e| bincode::error::DecodeError::OtherString(e.to_string()))?;
            source.read(&mut size_buf)?;

            let n_size = u64::from_le_bytes(size_buf);
            let n_size = usize::try_from(n_size)
                .map_err(|_| bincode::error::DecodeError::OutsideUsizeRange(n_size))?;

            dest.resize(n_size, 0);
            source.read(&mut dest)?;
            validate_atom_data_format(&dest, flags_buf[0])
                .map_err(bincode::error::DecodeError::OtherString)?;
            if flags_buf[0] == 5 {
                dest = convert_legacy_atom(&dest);
            }

            unsafe {
                match dest[0] & TYPE_MASK {
                    NUM_ID => Atom::Num(Num::from_raw(dest)),
                    VAR_ID => Atom::Var(Var::from_raw(dest)),
                    FUN_ID => Atom::Fun(Fun::from_raw(dest)),
                    MUL_ID => Atom::Mul(Mul::from_raw(dest)),
                    ADD_ID => Atom::Add(Add::from_raw(dest)),
                    POW_ID => Atom::Pow(Pow::from_raw(dest)),
                    _ => unreachable!("Unknown type {}", dest[0]),
                }
            }
        };

        let state_map = decoder.context().get_state_map();
        Ok(atom.as_view().rename(state_map))
    }
}

impl Atom {
    /// Read an expression in the current storage format without normalization.
    /// Imported symbols must be remapped before using the expression.
    pub(crate) fn read<R: Read>(&mut self, source: &mut R) -> Result<(), std::io::Error> {
        let format = source.read_u8()?;
        check_atom_format(format)?;

        let mut dest = std::mem::replace(self, Atom::Zero).into_raw();
        let n_size = source.read_u64::<LittleEndian>()?;
        let n_size = usize::try_from(n_size).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "Atom byte length does not fit in usize",
            )
        })?;

        dest.resize(n_size, 0);
        source.read_exact(&mut dest)?;
        validate_atom_data_format(&dest, format)
            .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error))?;
        if format == 5 {
            dest = convert_legacy_atom(&dest);
        }

        unsafe {
            match dest[0] & TYPE_MASK {
                NUM_ID => *self = Atom::Num(Num::from_raw(dest)),
                VAR_ID => *self = Atom::Var(Var::from_raw(dest)),
                FUN_ID => *self = Atom::Fun(Fun::from_raw(dest)),
                MUL_ID => *self = Atom::Mul(Mul::from_raw(dest)),
                ADD_ID => *self = Atom::Add(Add::from_raw(dest)),
                POW_ID => *self = Atom::Pow(Pow::from_raw(dest)),
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

        let n_terms = source.read_u64::<LittleEndian>()?;
        if n_terms == 1 {
            let mut a = Atom::new();
            a.read(source)?;
            Ok(a.as_view().rename(&state_map))
        } else {
            let mut res = Atom::new();
            let a = res.to_add();

            let mut tmp = Atom::new();
            let mut tmp2 = Atom::new();

            Workspace::get_local().with(|ws| {
                for _ in 0..n_terms {
                    tmp.read(&mut *source)?;

                    let mut settable = Settable::from(&mut tmp2);

                    tmp.as_view().rename_no_norm(&state_map, ws, &mut settable);

                    a.extend(settable.as_view_or(tmp.as_view()));
                }

                a.as_view().normalize(ws, &mut tmp);
                Ok(tmp)
            })
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
            Atom::Zero => 0,
        }
    }
}

/// A number/coefficient.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Num {
    data: RawAtom,
}

impl Num {
    #[inline(always)]
    /// Create the number zero, clearing and reusing `buffer`.
    pub fn zero(mut buffer: RawAtom) -> Num {
        buffer.clear();
        buffer.put_u8(inline_number_header(0));
        Num { data: buffer }
    }

    #[inline]
    /// Create an owned number from a coefficient.
    pub fn new(num: Coefficient) -> Num {
        let mut buffer = Vec::new();
        write_number(&num, &mut buffer);
        Num { data: buffer }
    }

    #[inline(always)]
    /// Create a number from a coefficient, clearing and reusing `buffer`.
    pub fn new_into(num: Coefficient, mut buffer: RawAtom) -> Num {
        buffer.clear();
        write_number(&num, &mut buffer);
        Num { data: buffer }
    }

    #[inline]
    /// Copy the view into owned storage, clearing and reusing `buffer`.
    pub fn from_view_into(a: &NumView<'_>, mut buffer: RawAtom) -> Num {
        buffer.clear();
        buffer.extend(a.get_data());
        Num { data: buffer }
    }

    #[inline]
    /// Replace this number with `num`, reusing its allocation.
    pub fn set_from_coeff(&mut self, num: Coefficient) {
        self.data.clear();
        write_number(&num, &mut self.data);
    }

    #[inline]
    /// Replace this value with a copy of the view, reusing its allocation.
    pub fn set_from_view(&mut self, a: &NumView<'_>) {
        self.data.clear();
        self.data.extend(a.get_data());
    }

    /// Add `other` to this number in place. The coefficient domains must support addition.
    pub fn add(&mut self, other: &NumView<'_>) {
        if self.data.len() == 1 && other.data.len() == 1 {
            // Inline numbers are exact real integers in -15..=15. Their sum
            // fits in i64 even when it needs the ordinary number payload.
            let value = inline_number_value(self.data[0]) + inline_number_value(other.data[0]);
            self.set_from_view(&InlineNum::new(value, 1).as_num_view());
            return;
        }
        let nv = self.to_num_view();
        let a = nv.get_coeff_view();
        let b = other.get_coeff_view();
        let n = a + b;

        self.set_from_coeff(n);
    }

    /// Multiply this number by `other` in place. The coefficient domains must support multiplication.
    pub fn mul(&mut self, other: &NumView<'_>) {
        if self.data.len() == 1 && other.data.len() == 1 {
            // The product is bounded by 225; preserve canonical storage when
            // crossing either boundary of the inline range.
            let value = inline_number_value(self.data[0]) * inline_number_value(other.data[0]);
            self.set_from_view(&InlineNum::new(value, 1).as_num_view());
            return;
        }
        let nv = self.to_num_view();
        let a = nv.get_coeff_view();
        let b = other.get_coeff_view();
        let n = a * b;

        self.set_from_coeff(n);
    }

    #[inline]
    /// Borrow a typed view without copying the underlying storage.
    pub fn to_num_view(&self) -> NumView<'_> {
        NumView { data: &self.data }
    }

    #[inline(always)]
    /// Borrow this value as an expression view without copying its storage.
    pub fn as_view(&self) -> AtomView<'_> {
        AtomView::Num(self.to_num_view())
    }

    #[inline(always)]
    /// Consume this value and return its encoded byte buffer.
    pub fn into_raw(self) -> RawAtom {
        self.data
    }

    #[inline(always)]
    pub(crate) unsafe fn from_raw(mut raw: RawAtom) -> Num {
        canonicalize_number(&mut raw);
        Num { data: raw }
    }
}

/// A variable.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Var {
    data: RawAtom,
}

impl Var {
    #[inline]
    /// Create an owned variable with the given symbol.
    pub fn new(symbol: Symbol) -> Var {
        Self::new_into(symbol, RawAtom::new())
    }

    #[inline]
    /// Create a variable with the given symbol, clearing and reusing `buffer`.
    pub fn new_into(symbol: Symbol, buffer: RawAtom) -> Var {
        let mut f = Var { data: buffer };
        f.set_from_symbol(symbol);
        f
    }

    #[inline]
    /// Copy the view into owned storage, clearing and reusing `buffer`.
    pub fn from_view_into(a: &VarView<'_>, mut buffer: RawAtom) -> Var {
        buffer.clear();
        buffer.extend(a.data);
        Var { data: buffer }
    }

    #[inline]
    /// Replace this variable's symbol, reusing its allocation.
    pub fn set_from_symbol(&mut self, symbol: Symbol) {
        self.data.clear();

        let (flags, extra) = symbol.encode_flags();
        self.data.put_u8(flags | VAR_ID);

        // shift by 1, so that the no-flag case does not take up extra space
        (symbol.id as u64, (extra * 2) as u64 + 1).write_packed(&mut self.data);
    }

    #[inline]
    /// Borrow a typed view without copying the underlying storage.
    pub fn to_var_view(&self) -> VarView<'_> {
        VarView { data: &self.data }
    }

    #[inline]
    /// Replace this value with a copy of the view, reusing its allocation.
    pub fn set_from_view(&mut self, view: &VarView) {
        self.data.clear();
        self.data.extend(view.data);
    }

    #[inline(always)]
    /// Borrow this value as an expression view without copying its storage.
    pub fn as_view(&self) -> AtomView<'_> {
        AtomView::Var(self.to_var_view())
    }

    #[inline]
    /// Return the symbol represented by this variable.
    pub fn get_symbol(&self) -> Symbol {
        self.to_var_view().get_symbol()
    }

    #[inline(always)]
    /// Consume this value and return its encoded byte buffer.
    pub fn into_raw(self) -> RawAtom {
        self.data
    }

    #[inline(always)]
    pub(crate) unsafe fn from_raw(raw: RawAtom) -> Var {
        Var { data: raw }
    }
}

/// A general function.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Fun {
    data: RawAtom,
}

impl Fun {
    #[inline]
    pub(crate) fn new_into(id: Symbol, buffer: RawAtom) -> Fun {
        let mut f = Fun { data: buffer };
        f.set_from_symbol(id);
        f
    }

    #[inline]
    /// Copy the view into owned storage, clearing and reusing `buffer`.
    pub fn from_view_into(a: &FunView<'_>, mut buffer: RawAtom) -> Fun {
        buffer.clear();
        buffer.extend(a.data);
        Fun { data: buffer }
    }

    #[inline]
    pub(crate) fn set_from_symbol(&mut self, symbol: Symbol) {
        self.data.clear();

        let (flags, extra) = symbol.encode_flags();
        self.data.put_u8(flags | FUN_ID | NOT_NORMALIZED);

        (0u64, 1).write_packed(&mut self.data);
        let header_end = self.data.len();

        ((extra as u64) << 32 | symbol.id as u64, 0).write_packed(&mut self.data);

        update_function_size(&mut self.data, header_end);
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
        let metadata = function_metadata(&self.data);
        let header_end = self.data.len() - metadata.len();
        let (name, n_args, args) = read_packed_pair(metadata);
        let old_end = self.data.len() - args.len();
        self.data.extend_from_slice(other.get_data());
        self.finish_append(header_end, old_end, name, n_args + 1);
    }

    pub(crate) fn add_args<'a>(&mut self, other: &[AtomView<'a>]) {
        self.add_args_iter(other.iter().copied().map(AtomOrView::from));
    }

    pub(crate) fn add_args_iter<'a>(&mut self, other: impl IntoIterator<Item = AtomOrView<'a>>) {
        self.data[0] |= NOT_NORMALIZED;

        let metadata = function_metadata(&self.data);
        let header_end = self.data.len() - metadata.len();
        let (name, mut n_args, args) = read_packed_pair(metadata);
        let old_end = self.data.len() - args.len();

        // Iterator/conversion code may panic. Until the headers are updated,
        // roll back appended bytes on unwind so the function stays valid.
        struct AppendGuard<'a> {
            data: &'a mut Vec<u8>,
            original_len: usize,
        }
        impl Drop for AppendGuard<'_> {
            fn drop(&mut self) {
                self.data.truncate(self.original_len);
            }
        }
        let original_len = self.data.len();
        let mut guard = AppendGuard {
            data: &mut self.data,
            original_len,
        };
        for item in other {
            guard.data.extend_from_slice(item.as_view().get_data());
            n_args += 1;
        }
        guard.original_len = guard.data.len();
        drop(guard);

        self.finish_append(header_end, old_end, name, n_args);
    }

    #[inline]
    fn finish_append(&mut self, header_end: usize, old_end: usize, name: u64, n_args: u64) {
        let header = (name, n_args);
        let new_end = header_end + header.get_packed_size() as usize;
        resize_header(&mut self.data, old_end, new_end);
        header.write_packed_fixed(&mut self.data[header_end..new_end]);

        update_function_size(&mut self.data, header_end);
    }

    #[inline(always)]
    /// Borrow a typed view without copying the underlying storage.
    pub fn to_fun_view(&self) -> FunView<'_> {
        FunView { data: &self.data }
    }

    /// Replace this value with a copy of the view, reusing its allocation.
    pub fn set_from_view(&mut self, view: &FunView) {
        self.data.clear();
        self.data.extend(view.data);
    }

    #[inline(always)]
    /// Borrow this value as an expression view without copying its storage.
    pub fn as_view(&self) -> AtomView<'_> {
        AtomView::Fun(self.to_fun_view())
    }

    #[inline(always)]
    /// Return the head symbol of this function call.
    pub fn get_symbol(&self) -> Symbol {
        self.to_fun_view().get_symbol()
    }

    #[inline(always)]
    /// Return the number of function arguments.
    pub fn get_nargs(&self) -> usize {
        self.to_fun_view().get_nargs()
    }

    #[inline(always)]
    /// Consume this value and return its encoded byte buffer.
    pub fn into_raw(self) -> RawAtom {
        self.data
    }

    #[inline(always)]
    pub(crate) unsafe fn from_raw(raw: RawAtom) -> Fun {
        Fun { data: raw }
    }
}

/// An expression raised to the power of another expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Pow {
    data: RawAtom,
}

impl Pow {
    #[inline]
    pub(crate) fn new_into(base: AtomView, exp: AtomView, buffer: RawAtom) -> Pow {
        let mut f = Pow { data: buffer };
        f.set_from_base_and_exp(base, exp);
        f
    }

    #[inline]
    /// Copy the view into owned storage, clearing and reusing `buffer`.
    pub fn from_view_into(a: &PowView<'_>, mut buffer: RawAtom) -> Pow {
        buffer.clear();
        buffer.extend(a.data);
        Pow { data: buffer }
    }

    #[inline]
    pub(crate) fn set_from_base_and_exp(&mut self, base: AtomView, exp: AtomView) {
        self.data.clear();
        self.data.put_u8(POW_ID | NOT_NORMALIZED);
        self.data.extend(base.get_data());
        if let (AtomView::Var(_), AtomView::Num(number)) = (base, exp) {
            let data = number.data;
            if data.len() == 1 {
                self.data[0] |= POW_HAS_SMALL_INT_FLAG;
                self.data.push(data[0]);
                return;
            }
        }
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
    /// Borrow a typed view without copying the underlying storage.
    pub fn to_pow_view(&self) -> PowView<'_> {
        PowView { data: &self.data }
    }

    #[inline(always)]
    /// Replace this value with a copy of the view, reusing its allocation.
    pub fn set_from_view(&mut self, view: &PowView) {
        self.data.clear();
        self.data.extend(view.data);
    }

    #[inline(always)]
    /// Borrow this value as an expression view without copying its storage.
    pub fn as_view(&self) -> AtomView<'_> {
        AtomView::Pow(self.to_pow_view())
    }

    #[inline(always)]
    /// Consume this value and return its encoded byte buffer.
    pub fn into_raw(self) -> RawAtom {
        self.data
    }

    #[inline(always)]
    pub(crate) unsafe fn from_raw(raw: RawAtom) -> Pow {
        Pow { data: raw }
    }
}

/// Multiplication of multiple subexpressions.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Mul {
    data: RawAtom,
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
    pub(crate) fn new_into(buffer: RawAtom) -> Mul {
        let mut value = Mul { data: buffer };
        value.clear();
        value
    }

    #[inline]
    pub(crate) fn clear(&mut self) {
        self.data.clear();
        // The packed (argument count, byte size) pair (0, 0) is [0x11, 0, 0].
        self.data
            .extend_from_slice(&[MUL_ID | NOT_NORMALIZED, 0x11, 0, 0]);
    }

    #[inline]
    /// Copy the view into owned storage, clearing and reusing `buffer`.
    pub fn from_view_into(a: &MulView<'_>, mut buffer: RawAtom) -> Mul {
        buffer.clear();
        buffer.extend(a.data);
        Mul { data: buffer }
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
    /// Replace this value with a copy of the view, reusing its allocation.
    pub fn set_from_view(&mut self, view: &MulView) {
        self.data.clear();
        self.data.extend(view.data);
    }

    #[inline]
    pub(crate) fn extend(&mut self, other: AtomView<'_>) {
        self.data[0] |= NOT_NORMALIZED;

        let (mut n_args, _, c) = read_packed_pair(&self.data[1..]);
        let old_end = self.data.len() - c.len();

        let data_start = match other {
            AtomView::Mul(m) => {
                let (sub_n_args, _, sd) = read_packed_pair(&m.data[1..]);

                n_args += sub_n_args;
                sd
            }
            _ => {
                n_args += 1;
                other.get_data()
            }
        };

        self.data.extend_from_slice(data_start);
        update_list_header(&mut self.data, old_end, n_args);
    }

    pub(crate) fn replace_first(&mut self, other: AtomView) {
        let (n_args, _, c) = read_packed_pair(&self.data[1..]);
        let first_arg_start = self.data.len() - c.len();

        // get size of first arg
        let aa = self.to_mul_view().to_slice().get(0);

        let old_first_len = aa.get_data().len();
        let new_first_len = other.get_data().len();

        match new_first_len.cmp(&old_first_len) {
            Ordering::Equal => {}
            Ordering::Less => {
                self.data.copy_within(
                    first_arg_start + old_first_len..,
                    first_arg_start + new_first_len,
                );
                let new_len = self.data.len() - old_first_len + new_first_len;
                self.data.truncate(new_len);
            }
            Ordering::Greater => {
                let old_len = self.data.len();
                self.data.resize(old_len + new_first_len - old_first_len, 0);
                self.data.copy_within(
                    first_arg_start + old_first_len..old_len,
                    first_arg_start + new_first_len,
                );
            }
        }

        self.data[first_arg_start..first_arg_start + new_first_len]
            .copy_from_slice(other.get_data());
        update_list_header(&mut self.data, first_arg_start, n_args);
    }

    #[inline]
    /// Borrow a typed view without copying the underlying storage.
    pub fn to_mul_view(&self) -> MulView<'_> {
        MulView { data: &self.data }
    }

    pub(crate) fn set_has_coefficient(&mut self, has_coeff: bool) {
        if has_coeff {
            self.data[0] |= MUL_HAS_COEFF_FLAG;
        } else {
            self.data[0] &= !MUL_HAS_COEFF_FLAG;
        }
    }

    #[inline(always)]
    /// Borrow this value as an expression view without copying its storage.
    pub fn as_view(&self) -> AtomView<'_> {
        AtomView::Mul(self.to_mul_view())
    }

    #[inline(always)]
    /// Return the number of factors in the product.
    pub fn get_nargs(&self) -> usize {
        self.to_mul_view().get_nargs()
    }

    #[inline(always)]
    /// Consume this value and return its encoded byte buffer.
    pub fn into_raw(self) -> RawAtom {
        self.data
    }

    #[inline(always)]
    pub(crate) unsafe fn from_raw(raw: RawAtom) -> Mul {
        Mul { data: raw }
    }
}

/// Addition of multiple subexpressions.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Add {
    data: RawAtom,
}

impl Default for Add {
    fn default() -> Self {
        Self::new()
    }
}

impl Add {
    /// Store two distinct, sorted, normalized, alias-free terms that are neither
    /// sums nor coefficients. The caller has already checked these invariants.
    #[inline]
    pub(crate) fn set_normalized_pair(&mut self, left: &[u8], right: &[u8]) {
        self.data.clear();
        let data: &mut Vec<u8> = &mut self.data;
        let payload_len = left.len() + right.len();
        if payload_len <= u8::MAX as usize {
            data.reserve(4 + payload_len);
            data.extend_from_slice(&[ADD_ID, 0x11, 2, payload_len as u8]);
        } else {
            let header = (2u64, payload_len as u64);
            data.reserve(1 + header.get_packed_size() as usize + payload_len);
            data.push(ADD_ID);
            header.write_packed(data);
        }
        data.extend_from_slice(left);
        data.extend_from_slice(right);
    }

    /// Create a normalized pair directly in a reusable buffer.
    #[inline]
    pub(crate) fn new_normalized_pair_into(left: &[u8], right: &[u8], buffer: RawAtom) -> Add {
        let mut value = Add { data: buffer };
        value.set_normalized_pair(left, right);
        value
    }

    #[inline]
    pub(crate) fn new() -> Add {
        Self::new_into(RawAtom::new())
    }

    #[inline]
    pub(crate) fn new_into(buffer: RawAtom) -> Add {
        let mut value = Add { data: buffer };
        value.clear();
        value
    }

    #[inline]
    pub(crate) fn clear(&mut self) {
        self.data.clear();
        // The packed (argument count, byte size) pair (0, 0) is [0x11, 0, 0].
        self.data
            .extend_from_slice(&[ADD_ID | NOT_NORMALIZED, 0x11, 0, 0]);
    }

    #[inline]
    /// Copy the view into owned storage, clearing and reusing `buffer`.
    pub fn from_view_into(a: &AddView<'_>, mut buffer: RawAtom) -> Add {
        buffer.clear();
        buffer.extend(a.data);
        Add { data: buffer }
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

        let (mut n_args, _, c) = read_packed_pair(&self.data[1..]);

        let old_header_size = unsafe { c.as_ptr().offset_from(self.data.as_ptr()) } as usize;

        match other {
            AtomView::Add(m) => {
                let (sub_n_args, _, sd) = read_packed_pair(&m.data[1..]);

                n_args += sub_n_args;
                self.data.extend_from_slice(sd);
            }
            _ => {
                n_args += 1;
                self.data.extend_from_slice(other.get_data());
            }
        };

        update_list_header(&mut self.data, old_header_size, n_args);
    }

    #[inline(always)]
    /// Borrow a typed view without copying the underlying storage.
    pub fn to_add_view(&self) -> AddView<'_> {
        AddView { data: &self.data }
    }

    #[inline(always)]
    /// Replace this value with a copy of the view, reusing its allocation.
    pub fn set_from_view(&mut self, view: AddView) {
        self.data.clear();
        self.data.extend(view.data);
    }

    #[inline(always)]
    /// Borrow this value as an expression view without copying its storage.
    pub fn as_view(&self) -> AtomView<'_> {
        AtomView::Add(self.to_add_view())
    }

    #[inline(always)]
    /// Return the number of terms in the sum.
    pub fn get_nargs(&self) -> usize {
        self.to_add_view().get_nargs()
    }

    #[inline(always)]
    /// Consume this value and return its encoded byte buffer.
    pub fn into_raw(self) -> RawAtom {
        self.data
    }

    #[inline(always)]
    pub(crate) unsafe fn from_raw(raw: RawAtom) -> Add {
        Add { data: raw }
    }

    pub(crate) fn grow_capacity(&mut self, size: usize) {
        if size > self.data.capacity() {
            let additional = size - self.data.len();
            self.data.reserve(additional);
        }
    }
}

impl<'a> VarView<'a> {
    #[inline]
    /// Copy this view into a new owned value.
    pub fn to_owned(&self) -> Var {
        Var::from_view_into(self, Vec::new())
    }

    #[inline]
    /// Replace `target` with a copy of this view, reusing its allocation.
    pub fn clone_into(&self, target: &mut Var) {
        target.set_from_view(self);
    }

    #[inline]
    /// Copy this view into owned storage, clearing and reusing `buffer`.
    pub fn clone_into_raw(&self, mut buffer: RawAtom) -> Var {
        buffer.clear();
        buffer.extend(self.data);
        Var { data: buffer }
    }

    #[inline(always)]
    /// Return the symbol represented by this variable.
    pub fn get_symbol(&self) -> Symbol {
        let (id, attrs, _) = self.data[1..].get_frac_u64();

        // attrs are shifted to improve the packing efficiency
        Symbol::decode_flags(id as u32, self.data[0], (attrs >> 1) as u32)
    }

    #[inline(always)]
    /// Return the variable's session-local symbol identifier.
    pub fn get_symbol_id(&self) -> u32 {
        let (id_and_attrs, _, _) = self.data[1..].get_frac_u64();
        id_and_attrs as u32
    }

    #[inline(always)]
    /// Return the symbol's wildcard level: zero for an ordinary symbol,
    /// or one, two, or three for its wildcard suffix.
    pub fn get_wildcard_level(&self) -> u8 {
        self.get_symbol().get_wildcard_level()
    }

    #[inline]
    /// Borrow this value as an expression view without copying its storage.
    pub fn as_view(&self) -> AtomView<'a> {
        AtomView::Var(*self)
    }

    /// Return the size in bytes of the encoded expression, excluding symbol metadata.
    pub fn get_byte_size(&self) -> usize {
        self.data.len()
    }
}

/// A view of a [Var].
#[derive(Debug, Copy, Clone, Eq, Hash)]
pub struct VarView<'a> {
    data: &'a [u8],
}

impl<'b> PartialEq<VarView<'b>> for VarView<'_> {
    fn eq(&self, other: &VarView<'b>) -> bool {
        self.data == other.data
    }
}

/// A view of a [Fun].
#[derive(Debug, Copy, Clone, Eq, Hash)]
pub struct FunView<'a> {
    data: &'a [u8],
}

impl<'b> PartialEq<FunView<'b>> for FunView<'_> {
    fn eq(&self, other: &FunView<'b>) -> bool {
        self.data == other.data
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
    /// Copy this view into a new owned value.
    pub fn to_owned(&self) -> Fun {
        Fun::from_view_into(self, Vec::new())
    }

    /// Replace `target` with a copy of this view, reusing its allocation.
    pub fn clone_into(&self, target: &mut Fun) {
        target.set_from_view(self);
    }

    /// Copy this view into owned storage, clearing and reusing `buffer`.
    pub fn clone_into_raw(&self, mut buffer: RawAtom) -> Fun {
        buffer.clear();
        buffer.extend(self.data);
        Fun { data: buffer }
    }

    #[inline(always)]
    /// Return the head symbol of this function call.
    pub fn get_symbol(&self) -> Symbol {
        let id_and_attrs = read_packed_numerator(function_metadata(self.data));
        Symbol::decode_flags(
            id_and_attrs as u32,
            self.data[0],
            (id_and_attrs >> 32) as u32,
        )
    }

    /// Get the symbol ID of the function. Slightly faster than [get_symbol](Self::get_symbol) if only the ID is needed.
    #[inline(always)]
    pub fn get_symbol_id(&self) -> u32 {
        let id_and_attrs = read_packed_numerator(function_metadata(self.data));
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
    /// Return whether the function has the symmetric attribute.
    pub fn is_symmetric(&self) -> bool {
        self.data[0] & SYM_CYCLESYMMETRIC_FLAG == SYM_SYMMETRIC_FLAG
    }

    #[inline(always)]
    /// Return whether the function has the antisymmetric attribute.
    pub fn is_antisymmetric(&self) -> bool {
        self.data[0] & SYM_CYCLESYMMETRIC_FLAG == SYM_ANTISYMMETRIC_FLAG
    }

    #[inline(always)]
    /// Return whether the function has the cyclic-symmetry attribute.
    pub fn is_cyclesymmetric(&self) -> bool {
        self.data[0] & SYM_CYCLESYMMETRIC_FLAG == SYM_CYCLESYMMETRIC_FLAG
    }

    #[inline(always)]
    /// Return whether the function has the linear attribute.
    pub fn is_linear(&self) -> bool {
        self.data[0] & SYM_LINEAR_FLAG == SYM_LINEAR_FLAG
    }

    #[inline(always)]
    /// Return the function symbol's wildcard level, or zero for an ordinary symbol.
    pub fn get_wildcard_level(&self) -> u8 {
        self.get_symbol().get_wildcard_level()
    }

    #[inline(always)]
    /// Return the number of function arguments.
    pub fn get_nargs(&self) -> usize {
        read_packed_denominator(function_metadata(self.data)).0 as usize
    }

    #[inline(always)]
    pub(crate) fn is_normalized(&self) -> bool {
        (self.data[0] & NOT_NORMALIZED) == 0
    }

    #[inline]
    /// Iterate over the function arguments in their stored order.
    pub fn iter(&self) -> ListIterator<'a> {
        let (n_args, c) = read_packed_denominator(function_metadata(self.data));

        ListIterator {
            data: c,
            length: n_args as usize,
        }
    }

    /// Borrow this value as an expression view without copying its storage.
    pub fn as_view(&self) -> AtomView<'a> {
        AtomView::Fun(*self)
    }

    /// Borrow the function arguments as an argument slice.
    pub fn to_slice(&self) -> ListSlice<'a> {
        let (n_args, c) = read_packed_denominator(function_metadata(self.data));

        ListSlice {
            data: c,
            length: n_args as usize,
            slice_type: SliceType::Arg,
        }
    }

    /// Return the size in bytes of the encoded expression, excluding symbol metadata.
    pub fn get_byte_size(&self) -> usize {
        self.data.len()
    }

    pub(crate) fn fast_cmp(&self, other: FunView) -> Ordering {
        self.data.cmp(other.data)
    }
}

/// A view of a [Num].
#[derive(Debug, Copy, Clone, Eq)]
pub struct NumView<'a> {
    data: &'a [u8],
}

impl<'b> PartialEq<NumView<'b>> for NumView<'_> {
    #[inline]
    fn eq(&self, other: &NumView<'b>) -> bool {
        self.get_data() == other.get_data()
    }
}

impl Hash for NumView<'_> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.get_data().hash(state);
    }
}

impl<'a> NumView<'a> {
    /// Inspect small signed integer exponents without general coefficient decoding.
    #[inline]
    pub(crate) fn small_integer(&self) -> Option<i64> {
        super::coefficient::small_integer_atom(self.data)
    }

    /// The identity check can keep canonical small integers on their existing
    /// header-only path without dispatching the general coefficient encoding.
    #[inline]
    pub(crate) fn small_integer_is_zero(&self) -> Option<bool> {
        if self.data.len() == 1 {
            Some(inline_number_value(self.data[0]) == 0)
        } else {
            self.data.is_small_int().then(|| self.data.is_zero_rat())
        }
    }

    /// Borrow the canonical number bytes at their original location.
    #[inline(always)]
    pub(crate) fn get_data(&self) -> &'a [u8] {
        self.data
    }

    #[inline]
    /// Copy this view into a new owned value.
    pub fn to_owned(&self) -> Num {
        Num::from_view_into(self, Vec::new())
    }

    #[inline]
    /// Replace `target` with a copy of this view, reusing its allocation.
    pub fn clone_into(&self, target: &mut Num) {
        target.set_from_view(self);
    }

    #[inline]
    /// Copy this view into owned storage, clearing and reusing `buffer`.
    pub fn clone_into_raw(&self, mut buffer: RawAtom) -> Num {
        buffer.clear();
        buffer.extend(self.get_data());
        Num { data: buffer }
    }

    #[inline]
    /// Return whether this coefficient is zero.
    pub fn is_zero(&self) -> bool {
        if self.data.len() == 1 {
            return self.data[0] == inline_number_header(0);
        }
        if self.data.is_small_int() {
            self.data.is_zero_rat()
        } else {
            self.get_coeff_view().is_zero()
        }
    }

    #[inline]
    /// Return whether this coefficient is one.
    pub fn is_one(&self) -> bool {
        if self.data.len() == 1 {
            return self.data[0] == inline_number_header(1);
        }
        if self.data.is_small_int() {
            self.data.is_one_rat()
        } else {
            self.get_coeff_view().is_one()
        }
    }

    #[inline]
    pub(crate) fn is_exact_one(&self) -> bool {
        if self.data.len() == 1 {
            return self.data[0] == inline_number_header(1);
        }
        if self.data.is_small_int() {
            self.data.is_one_rat()
        } else {
            self.get_coeff_view().is_exact_one()
        }
    }

    #[inline]
    /// Return whether the coefficient is stored as a rational polynomial.
    pub fn is_rational_polynomial(&self) -> bool {
        self.data.len() != 1 && self.data.is_rational_polynomial()
    }

    #[inline]
    /// Borrow the coefficient representation without allocating an owned coefficient.
    pub fn get_coeff_view(&self) -> CoefficientView<'a> {
        if self.data.len() == 1 {
            CoefficientView::Natural(inline_number_value(self.data[0]), 1, 0, 1)
        } else {
            self.data[1..].get_coeff_view().0
        }
    }

    /// Borrow this value as an expression view without copying its storage.
    pub fn as_view(&self) -> AtomView<'a> {
        AtomView::Num(*self)
    }

    /// Return the size in bytes of the encoded expression, excluding symbol metadata.
    pub fn get_byte_size(&self) -> usize {
        self.data.len()
    }
}

#[cfg(test)]
mod compact_number_tests {
    use super::*;
    use crate::parse;
    use std::hash::{DefaultHasher, Hasher};

    fn hash(value: impl Hash) -> u64 {
        let mut state = DefaultHasher::new();
        value.hash(&mut state);
        state.finish()
    }

    #[test]
    fn inline_numbers_have_canonical_owned_bytes_and_hashes() {
        for value in -15i64..=15 {
            let storage = [inline_number_header(value)];
            let compact = NumView { data: &storage };
            let standalone = InlineNum::new(value as i64, 1);
            let expected = standalone.as_num_view();
            assert_eq!(compact, expected);
            assert_eq!(compact.get_coeff_view(), expected.get_coeff_view());
            assert_eq!(compact.get_data(), standalone.get_data());
            assert_eq!(compact.get_data().as_ptr(), storage.as_ptr());
            assert_eq!(compact.get_byte_size(), 1);
            assert_eq!(compact.is_zero(), value == 0);
            assert_eq!(compact.is_one(), value == 1);
            assert_eq!(compact.is_exact_one(), value == 1);
            assert!(!compact.is_rational_polynomial());
            assert_eq!(hash(compact), hash(expected));
            assert_eq!(hash(compact.as_view()), hash(standalone.get_data()));

            assert_eq!(compact.to_owned().as_view(), expected.as_view());
            let mut copy = Num::zero(Vec::new());
            compact.clone_into(&mut copy);
            assert_eq!(copy.as_view(), expected.as_view());
            assert_eq!(
                compact.clone_into_raw(Vec::new()).into_raw(),
                standalone.get_data()
            );

            let mut lookup = HashMap::default();
            lookup.insert(Atom::Num(compact.to_owned()), value);
            assert_eq!(
                lookup.get::<[u8]>(compact.as_view().get_data()),
                Some(&value)
            );
            assert_eq!(lookup.get::<[u8]>(standalone.get_data()), Some(&value));
        }
    }

    #[test]
    fn inline_numbers_are_canonical_across_writers_and_arithmetic() {
        for value in -17i64..=17 {
            let expected = InlineNum::new(value, 1);
            let parsed = parse!(&value.to_string());
            let atom = Atom::num(value);
            assert_eq!(atom, parsed);
            assert_eq!(atom.as_view(), expected.as_view());
            assert_eq!(hash(atom.as_view()), hash(expected.as_view()));
            assert_eq!(hash(&atom), hash(&parsed));
            assert_eq!(
                atom.as_view().get_byte_size(),
                if (-15..=15).contains(&value) { 1 } else { 3 }
            );

            let mut number = Num::new(value.into());
            assert_eq!(number.as_view(), expected.as_view());
            assert_eq!(
                Num::new_into(value.into(), Vec::new()).as_view(),
                expected.as_view()
            );
            number.set_from_coeff((value + 1).into());
            number.add(&InlineNum::new(-1, 1).as_num_view());
            assert_eq!(number.as_view(), expected.as_view());
            number.set_from_coeff((-value).into());
            number.mul(&InlineNum::new(-1, 1).as_num_view());
            assert_eq!(number.as_view(), expected.as_view());

            let raw = vec![
                NUM_ID,
                if value < 0 { 0x81 } else { 1 },
                value.unsigned_abs() as u8,
            ];
            assert_eq!(unsafe { Num::from_raw(raw) }.as_view(), expected.as_view());
        }
        assert_eq!(InlineNum::zero(), InlineNum::new(0, 1));
        assert_eq!(InlineNum::one(), InlineNum::new(1, 1));
        assert_eq!(AtomView::ZERO, InlineNum::zero().as_view());
    }

    #[test]
    fn inline_numbers_preserve_other_coefficient_domains() {
        use crate::domains::{Ring, finite_field::Zp64};
        let field = Zp64::new(17);
        let one = field.one();
        let finite = Coefficient::from_finite_field(field, one);
        for coeff in [
            Coefficient::from((1, 2)),
            Coefficient::from(2.0),
            finite,
            Coefficient::indeterminate(),
            Coefficient::positive_infinity(),
        ] {
            let number = Num::new(coeff.clone());
            assert_eq!(number.as_view().get_data()[0], NUM_ID);
            assert_eq!(number.to_num_view().get_coeff_view().to_owned(), coeff);
        }
        let complex = parse!("2+3*𝑖");
        assert_eq!(complex.as_view().get_data()[0], NUM_ID);
        let mixed = parse!("inline_num_list(0,1,-1,15,16,-15,-16,1/2,2+3*𝑖,x^2,f(x)^2)");
        let AtomView::Fun(f) = mixed.as_view() else {
            unreachable!()
        };
        let values: Vec<_> = f.iter().collect();
        for (i, value) in values.iter().enumerate() {
            assert_eq!(f.to_slice().get(i), *value);
            assert_eq!(f.to_slice().fast_forward(i).get(0), *value);
            assert_eq!(f.iter().nth(i), Some(*value));
        }
    }

    #[test]
    fn inline_number_arithmetic_matches_coefficient_arithmetic() {
        for left in -15i64..=15 {
            for right in -15i64..=15 {
                let left_coeff = Coefficient::from(left);
                let right_coeff = Coefficient::from(right);
                let right_num = Num::new(right_coeff.clone());
                let mut sum = Num::new(left_coeff.clone());
                let mut product = Num::new(left_coeff.clone());
                sum.add(&right_num.to_num_view());
                product.mul(&right_num.to_num_view());
                for (actual, expected) in [
                    (sum, left_coeff.clone() + right_coeff.clone()),
                    (product, left_coeff * right_coeff),
                ] {
                    let expected_num = Num::new(expected.clone());
                    assert_eq!(actual.to_num_view().get_coeff_view().to_owned(), expected);
                    assert_eq!(actual, expected_num);
                    assert_eq!(hash(&actual), hash(&expected_num));
                    assert_eq!(
                        actual.as_view().get_data(),
                        expected_num.as_view().get_data()
                    );
                    assert_eq!(actual.to_num_view().is_zero(), expected.is_zero());
                    assert_eq!(actual.to_num_view().is_one(), expected.is_one());
                }
            }
        }
    }

    #[test]
    fn inline_number_arithmetic_preserves_mixed_domains() {
        let nonreal = parse!("2+3*𝑖");
        let AtomView::Num(nonreal) = nonreal.as_view() else {
            unreachable!()
        };
        for ordinary in [
            Coefficient::from(-16),
            Coefficient::from(16),
            Coefficient::from((1, 2)),
            Coefficient::from(2.0),
            nonreal.get_coeff_view().to_owned(),
        ] {
            for inline in [-15, -1, 0, 1, 15] {
                let inline = Coefficient::from(inline);
                for (left, right) in [
                    (ordinary.clone(), inline.clone()),
                    (inline.clone(), ordinary.clone()),
                ] {
                    let right_num = Num::new(right.clone());
                    let mut sum = Num::new(left.clone());
                    let mut product = Num::new(left.clone());
                    sum.add(&right_num.to_num_view());
                    product.mul(&right_num.to_num_view());
                    for (actual, expected) in
                        [(sum, left.clone() + right.clone()), (product, left * right)]
                    {
                        let expected = Num::new(expected);
                        assert_eq!(actual, expected);
                        assert_eq!(hash(&actual), hash(&expected));
                    }
                }
            }
        }
    }

    #[test]
    #[cfg(feature = "bincode")]
    fn inline_number_bincode_converts_legacy_storage() {
        for value in -17i64..=17 {
            let expected = InlineNum::new(value, 1);
            let mut legacy = [0; 24];
            legacy[..3].copy_from_slice(&[
                NUM_ID,
                if value < 0 { 0x81 } else { 1 },
                value.unsigned_abs() as u8,
            ]);
            for bytes in [
                bincode::encode_to_vec((legacy, 3u8), bincode::config::standard()).unwrap(),
                bincode::encode_to_vec(expected, bincode::config::standard()).unwrap(),
            ] {
                let (decoded, consumed): (InlineNum, usize) =
                    bincode::decode_from_slice_with_context(
                        &bytes,
                        bincode::config::standard(),
                        StateMap::default(),
                    )
                    .unwrap();
                assert_eq!(decoded, expected);
                assert_eq!(consumed, bytes.len());
                assert_eq!(hash(decoded), hash(expected));
            }
        }
    }
}

/// A view of a [Pow].
#[derive(Debug, Copy, Clone, Eq, Hash)]
pub struct PowView<'a> {
    data: &'a [u8],
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

impl<'a> PowView<'a> {
    #[inline]
    /// Copy this view into a new owned value.
    pub fn to_owned(&self) -> Pow {
        Pow::from_view_into(self, Vec::new())
    }

    #[inline]
    /// Replace `target` with a copy of this view, reusing its allocation.
    pub fn clone_into(&self, target: &mut Pow) {
        target.set_from_view(self);
    }

    #[inline]
    /// Copy this view into owned storage, clearing and reusing `buffer`.
    pub fn clone_into_raw(&self, mut buffer: RawAtom) -> Pow {
        buffer.clear();
        buffer.extend(self.data);
        Pow { data: buffer }
    }

    #[inline(always)]
    /// Borrow the base of this power.
    pub fn get_base(&self) -> AtomView<'a> {
        self.iter().next().unwrap()
    }

    #[inline]
    /// Borrow the exponent of this power.
    pub fn get_exp(&self) -> AtomView<'a> {
        let (_, e) = self.get_base_exp();
        e
    }

    #[inline]
    pub(crate) fn is_normalized(&self) -> bool {
        (self.data[0] & NOT_NORMALIZED) == 0
    }

    /// Decode a compact variable power without constructing a coefficient view.
    #[inline]
    pub(crate) fn get_small_integer_power(&self) -> Option<(VarView<'a>, i64)> {
        if !self.has_small_integer_power() {
            return None;
        }
        let end = self.data.len() - 1;
        Some((
            VarView {
                data: &self.data[1..end],
            },
            inline_number_value(self.data[end]),
        ))
    }

    #[inline]
    pub(crate) fn has_small_integer_power(&self) -> bool {
        self.data[0] & POW_HAS_SMALL_INT_FLAG != 0
    }

    #[inline(always)]
    /// Borrow the base and exponent, in that order.
    pub fn get_base_exp(&self) -> (AtomView<'a>, AtomView<'a>) {
        if self.data[0] & POW_HAS_SMALL_INT_FLAG != 0 {
            let end = self.data.len() - 1;
            return (
                AtomView::Var(VarView {
                    data: &self.data[1..end],
                }),
                AtomView::Num(NumView {
                    data: &self.data[end..],
                }),
            );
        }
        let mut it = self.iter();

        (it.next().unwrap(), it.next().unwrap())
    }

    #[inline]
    /// Iterate over the base followed by the exponent.
    pub fn iter(&self) -> ListIterator<'a> {
        ListIterator {
            data: &self.data[1..],
            length: 2,
        }
    }

    #[inline]
    /// Borrow this value as an expression view without copying its storage.
    pub fn as_view(&self) -> AtomView<'a> {
        AtomView::Pow(*self)
    }

    #[inline]
    /// Borrow a two-element slice containing the base followed by the exponent.
    pub fn to_slice(&self) -> ListSlice<'a> {
        ListSlice {
            data: &self.data[1..],
            length: 2,
            slice_type: SliceType::Pow,
        }
    }

    /// Return the size in bytes of the encoded expression, excluding symbol metadata.
    pub fn get_byte_size(&self) -> usize {
        self.data.len()
    }
}

/// A view of a [Mul].
#[derive(Debug, Copy, Clone, Eq, Hash)]
pub struct MulView<'a> {
    data: &'a [u8],
}

impl<'b> PartialEq<MulView<'b>> for MulView<'_> {
    #[inline]
    fn eq(&self, other: &MulView<'b>) -> bool {
        self.data == other.data
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
    /// Copy this view into a new owned value.
    pub fn to_owned(&self) -> Mul {
        Mul::from_view_into(self, Vec::new())
    }

    #[inline]
    /// Replace `target` with a copy of this view, reusing its allocation.
    pub fn clone_into(&self, target: &mut Mul) {
        target.set_from_view(self);
    }

    #[inline]
    /// Copy this view into owned storage, clearing and reusing `buffer`.
    pub fn clone_into_raw(&self, mut buffer: RawAtom) -> Mul {
        buffer.clear();
        buffer.extend(self.data);
        Mul { data: buffer }
    }

    #[inline]
    pub(crate) fn is_normalized(&self) -> bool {
        (self.data[0] & NOT_NORMALIZED) == 0
    }

    /// Return the number of factors, including an explicit coefficient when present.
    pub fn get_nargs(&self) -> usize {
        read_packed_numerator(&self.data[1..]) as usize
    }

    #[inline]
    /// Iterate over the factors in their stored order.
    pub fn iter(&self) -> ListIterator<'a> {
        let (n_args, _, c) = read_packed_pair(&self.data[1..]);

        ListIterator {
            data: c,
            length: n_args as usize,
        }
    }

    #[inline]
    /// Borrow this value as an expression view without copying its storage.
    pub fn as_view(&self) -> AtomView<'a> {
        AtomView::Mul(*self)
    }

    /// Borrow the factors as a product slice.
    pub fn to_slice(&self) -> ListSlice<'a> {
        let (n_args, _, c) = read_packed_pair(&self.data[1..]);

        ListSlice {
            data: c,
            length: n_args as usize,
            slice_type: SliceType::Mul,
        }
    }

    #[inline]
    /// Return whether this product stores an explicit numerical coefficient.
    pub fn has_coefficient(&self) -> bool {
        self.data[0] & MUL_HAS_COEFF_FLAG == MUL_HAS_COEFF_FLAG
    }

    #[inline]
    /// Borrow the explicit numerical coefficient, or return `None` when it is
    /// absent (the implicit coefficient is one).
    pub fn get_coefficient(&self) -> Option<AtomView<'a>> {
        if self.has_coefficient() {
            self.iter().next()
        } else {
            None
        }
    }

    /// Return the size in bytes of the encoded expression, excluding symbol metadata.
    pub fn get_byte_size(&self) -> usize {
        self.data.len()
    }
}

/// A view of a [Add].
#[derive(Debug, Copy, Clone, Eq, Hash)]
pub struct AddView<'a> {
    data: &'a [u8],
}

impl<'b> PartialEq<AddView<'b>> for AddView<'_> {
    #[inline]
    fn eq(&self, other: &AddView<'b>) -> bool {
        self.data == other.data
    }
}

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
    /// Copy this view into a new owned value.
    pub fn to_owned(&self) -> Add {
        Add::from_view_into(self, Vec::new())
    }

    /// Replace `target` with a copy of this view, reusing its allocation.
    pub fn clone_into(&self, target: &mut Add) {
        target.set_from_view(*self);
    }

    /// Copy this view into owned storage, clearing and reusing `buffer`.
    pub fn clone_into_raw(&self, mut buffer: RawAtom) -> Add {
        buffer.clear();
        buffer.extend(self.data);
        Add { data: buffer }
    }

    #[inline(always)]
    pub(crate) fn is_normalized(&self) -> bool {
        (self.data[0] & NOT_NORMALIZED) == 0
    }

    #[inline(always)]
    /// Return the number of terms in the sum.
    pub fn get_nargs(&self) -> usize {
        read_packed_numerator(&self.data[1..]) as usize
    }

    #[inline]
    /// Iterate over the terms in their stored order.
    pub fn iter(&self) -> ListIterator<'a> {
        let (n_args, _, c) = read_packed_pair(&self.data[1..]);

        ListIterator {
            data: c,
            length: n_args as usize,
        }
    }

    #[inline]
    /// Borrow this value as an expression view without copying its storage.
    pub fn as_view(&self) -> AtomView<'a> {
        AtomView::Add(*self)
    }

    /// Borrow the terms as a sum slice.
    pub fn to_slice(&self) -> ListSlice<'a> {
        let (n_args, _, c) = read_packed_pair(&self.data[1..]);

        ListSlice {
            data: c,
            length: n_args as usize,
            slice_type: SliceType::Add,
        }
    }

    /// Return the size in bytes of the encoded expression, excluding symbol metadata.
    pub fn get_byte_size(&self) -> usize {
        self.data.len()
    }
}

impl<'a> AtomView<'a> {
    /// A view of the exact number zero backed by static storage.
    pub const ZERO: Self = Self::Num(NumView { data: &ZERO_DATA });

    /// Interpret `source` as one complete expression in Symbolica's encoded
    /// representation. The bytes must already be valid, such as those returned
    /// by [`Self::get_data`], and reference the current symbol state.
    /// This does not validate or import serialized input; use [`Atom::import`]
    /// for expressions transferred between sessions.
    #[inline]
    pub fn from(source: &'a [u8]) -> AtomView<'a> {
        let tag = source[0] & TYPE_MASK;
        Self::from_type(source, tag)
    }

    /// Construct a view after traversal has already read its masked type tag.
    #[inline(always)]
    fn from_type(source: &'a [u8], tag: u8) -> AtomView<'a> {
        match tag {
            VAR_ID => AtomView::Var(VarView { data: source }),
            FUN_ID => AtomView::Fun(FunView { data: source }),
            NUM_ID => AtomView::Num(NumView { data: source }),
            POW_ID => AtomView::Pow(PowView { data: source }),
            MUL_ID => AtomView::Mul(MulView { data: source }),
            ADD_ID => AtomView::Add(AddView { data: source }),
            x => unreachable!("Bad id: {}", x),
        }
    }

    #[inline(always)]
    /// Borrow the encoded expression bytes. Symbol metadata is stored separately.
    /// Child views, including inline numbers, borrow their bytes from the parent.
    pub fn get_data(&self) -> &'a [u8] {
        match self {
            AtomView::Num(n) => n.data,
            AtomView::Var(v) => v.data,
            AtomView::Fun(f) => f.data,
            AtomView::Pow(p) => p.data,
            AtomView::Mul(t) => t.data,
            AtomView::Add(e) => e.data,
        }
    }

    /// Export the atom and the required state to a binary stream. It can be loaded
    /// with [Atom::import].
    #[inline(always)]
    pub fn export<W: Write>(&self, dest: &mut W) -> Result<(), std::io::Error> {
        let active_symbols = self.get_all_symbols(true);
        State::export_partial(dest, active_symbols)?;

        dest.write_u64::<LittleEndian>(1)?; // export a single expression

        self.write(dest)
    }

    /// Write the expression to a binary stream. The byte-length is written first,
    /// followed by the data. To import the expression in new session, also export the [`State`].
    ///
    /// Most users will want to use [AtomView::export] instead.
    #[inline(always)]
    pub fn write<W: Write>(&self, dest: &mut W) -> Result<(), std::io::Error> {
        let d = self.get_data();
        dest.write_u8(ATOM_EXPORT_FORMAT)?;
        dest.write_u64::<LittleEndian>(d.len() as u64)?;
        dest.write_all(d)
    }

    /// Rename all symbols in this (imported) atom using the given state map.
    /// Normalization can only take place after all symbols have been renamed.
    pub(crate) fn rename(&self, state_map: &StateMap) -> Atom {
        let mut out = Atom::new();

        Workspace::get_local().with(|ws| {
            let mut set = Settable::from(&mut out);
            self.rename_no_norm(state_map, ws, &mut set);

            if let Some(value) = set.get() {
                let mut a = ws.new_atom();
                value.as_view().normalize(ws, &mut a);
                std::mem::swap(&mut out, &mut a);
            } else {
                self.normalize(ws, &mut out);
            }
        });

        out
    }

    pub(crate) fn rename_no_norm(
        &self,
        state_map: &StateMap,
        ws: &Workspace,
        out: &mut Settable<'_, Atom>,
    ) {
        match self {
            AtomView::Num(n) => match n.get_coeff_view() {
                CoefficientView::FiniteField(e, i) => {
                    if let Some(s) = state_map.finite_fields.get(&i) {
                        out.to_num(Coefficient::FiniteField(e, *s));
                    }
                }
                CoefficientView::RationalPolynomial(r) => {
                    let (old_id, _, _) = r.0.get_frac_u64();

                    if let Some(nv) = state_map.get_variable_list(old_id) {
                        let rr = r.deserialize_with_variables(nv);
                        out.to_num(Coefficient::RationalPolynomial(rr));
                    }
                }
                _ => {}
            },
            AtomView::Var(v) => {
                if let Some(s) = state_map.symbols.get(&v.get_symbol_id()) {
                    out.to_var(*s);
                }
            }
            AtomView::Fun(f) => {
                let mut fun = if let Some(s) = state_map.symbols.get(&f.get_symbol_id()) {
                    Some(out.to_fun(*s))
                } else {
                    None
                };

                let mut arg_h = ws.new_atom();
                for (i, arg) in f.iter().enumerate() {
                    let mut set = Settable::from(&mut *arg_h);
                    arg.rename_no_norm(state_map, ws, &mut set);

                    if fun.is_none() && set.is_set() {
                        let fun_o = out.to_fun(f.get_symbol());

                        for child in f.iter().take(i) {
                            fun_o.add_arg(child);
                        }

                        fun_o.add_arg(set.as_view());
                        fun = Some(fun_o);
                    } else if let Some(fun) = &mut fun {
                        fun.add_arg(set.as_view_or(arg));
                    }
                }
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();

                let mut base_h = ws.new_atom();
                let mut base_set = Settable::from(&mut *base_h);
                base.rename_no_norm(state_map, ws, &mut base_set);

                let mut exp_h = ws.new_atom();
                let mut exp_set = Settable::from(&mut *exp_h);
                exp.rename_no_norm(state_map, ws, &mut exp_set);

                if base_set.is_set() || exp_set.is_set() {
                    out.to_pow(base_set.as_view_or(base), exp_set.as_view_or(exp));
                }
            }
            AtomView::Mul(mm) => {
                let mut mul = None;

                let mut child_h = ws.new_atom();
                for (i, child) in mm.iter().enumerate() {
                    let mut set = Settable::from(&mut *child_h);
                    child.rename_no_norm(state_map, ws, &mut set);

                    if mul.is_none() && set.is_set() {
                        let mul_o = out.to_mul();

                        for child in mm.iter().take(i) {
                            mul_o.extend(child);
                        }
                        mul_o.extend(set.as_view());
                        mul = Some(mul_o);
                    } else if let Some(mul_o) = &mut mul {
                        mul_o.extend(set.as_view_or(child));
                    }
                }
            }
            AtomView::Add(a) => {
                let mut add = None;

                let mut child_h = ws.new_atom();
                for (i, child) in a.iter().enumerate() {
                    let mut set = Settable::from(&mut *child_h);
                    child.rename_no_norm(state_map, ws, &mut set);

                    if add.is_none() && set.is_set() {
                        let add_o = out.to_add();

                        for child in a.iter().take(i) {
                            add_o.extend(child);
                        }
                        add_o.extend(set.as_view());
                        add = Some(add_o);
                    } else if let Some(mul_o) = &mut add {
                        mul_o.extend(set.as_view_or(child));
                    }
                }
            }
        }
    }
}

/// An iterator of a list of atoms.
#[derive(Debug, Copy, Clone)]
pub struct ListIterator<'a> {
    data: &'a [u8],
    length: usize,
}

impl<'a> ListIterator<'a> {
    /// Capture the current boundary for a later `resume_at` on this list.
    pub(crate) fn cursor(&self) -> (&'a [u8], usize) {
        (self.data, self.length)
    }

    /// Resume at a previously visited boundary within this list.
    ///
    /// # Safety
    /// `data` must be a suffix of this iterator's original data starting at an
    /// atom boundary, and `length` must be the exact number of remaining atoms.
    /// The underlying expression must not have changed since recording it.
    pub unsafe fn resume_at(self, data: &'a [u8], length: usize) -> Self {
        assert!(data.as_ptr() >= self.data.as_ptr());
        assert_eq!(data.as_ptr_range().end, self.data.as_ptr_range().end);
        assert!(length <= self.length);
        Self { data, length }
    }
}

impl<'a> Iterator for ListIterator<'a> {
    type Item = AtomView<'a>;

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.length, Some(self.length))
    }

    #[inline(always)]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if n >= self.length {
            self.data = &[];
            self.length = 0;
            return None;
        }

        // Keep constant indices as easy to inline and unroll as repeated next()
        // calls. The upfront length check also handles oversized indices in O(1).
        for _ in 0..n {
            self.next();
        }
        self.next()
    }

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.length == 0 {
            return None;
        }

        self.length -= 1;
        let start = self.data;
        let header = start[0];
        if header & TYPE_MASK == NUM_ID && header != NUM_ID {
            self.data = &start[1..];
            return Some(AtomView::Num(NumView { data: &start[..1] }));
        }
        let start_id = header & TYPE_MASK;
        let data = if self.length == 0 {
            // The remaining slice contains exactly the final atom.
            std::mem::take(&mut self.data)
        } else {
            self.data = &self.data[1..];
            let mut cur_header = start[0];

            // store how many more atoms to read
            // can be used instead of storing the byte length of an atom
            let mut skip_count = 1usize;
            loop {
                match cur_header & TYPE_MASK {
                    NUM_ID => {
                        if cur_header == NUM_ID {
                            self.data = self.data.skip_rational();
                        }
                    }
                    VAR_ID => {
                        self.data = unsafe { skip_packed_variable(self.data) };
                    }
                    FUN_ID => {
                        // Views contain complete, valid atoms, including the payload.
                        self.data = unsafe { skip_packed_function(self.data) };
                    }
                    MUL_ID | ADD_ID => {
                        self.data = unsafe { skip_packed_list(self.data) };
                    }
                    POW_ID => {
                        if cur_header & POW_HAS_SMALL_INT_FLAG != 0 {
                            // The flag guarantees a variable followed by one inline number.
                            let exponent = unsafe { skip_packed_variable(&self.data[1..]) };
                            self.data = &exponent[1..];
                        } else {
                            skip_count += 2;
                        }
                    }
                    _ => unreachable!("Bad id"),
                }

                skip_count -= 1;

                if skip_count == 0 {
                    break;
                }

                cur_header = self.data.get_u8();
            }

            let len = unsafe { self.data.as_ptr().offset_from(start.as_ptr()) } as usize;

            unsafe { start.get_unchecked(..len) }
        };

        Some(AtomView::from_type(data, start_id))
    }
}

impl<'a> ExactSizeIterator for ListIterator<'a> {
    #[inline]
    fn len(&self) -> usize {
        self.length
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
    /// Return the number of expressions still to be yielded.
    pub fn len(&self) -> usize {
        self.length
    }

    #[inline]
    /// Create an iterator that yields `atom` exactly once.
    pub fn from_one(atom: AtomView<'a>) -> Self {
        ListIterator {
            data: atom.get_data(),
            length: 1,
        }
    }
}

/// A slice of a list of atoms.
#[derive(Debug, Copy, Clone, Eq)]
pub struct ListSlice<'a> {
    data: &'a [u8],
    length: usize,
    slice_type: SliceType,
}

impl PartialEq for ListSlice<'_> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.length == other.length
            && self.slice_type == other.slice_type
            && self.data == other.data
    }
}

impl<'a> ListSlice<'a> {
    #[inline(always)]
    fn skip(mut pos: &[u8], n: usize) -> &[u8] {
        // store how many more atoms to read
        // can be used instead of storing the byte length of an atom
        let mut skip_count = n;
        while skip_count > 0 {
            skip_count -= 1;

            let header = unsafe { *pos.get_unchecked(0) };
            pos = unsafe { pos.get_unchecked(1..) };
            match header & TYPE_MASK {
                NUM_ID => {
                    if header == NUM_ID {
                        pos = pos.skip_rational();
                    }
                }
                VAR_ID => {
                    pos = unsafe { skip_packed_variable(pos) };
                }
                FUN_ID => {
                    pos = unsafe { skip_packed_function(pos) };
                }
                MUL_ID | ADD_ID => {
                    pos = unsafe { skip_packed_list(pos) };
                }
                POW_ID => {
                    if header & POW_HAS_SMALL_INT_FLAG != 0 {
                        // Skip the variable header, its packed pair, and the inline exponent.
                        let exponent = unsafe { skip_packed_variable(&pos[1..]) };
                        pos = &exponent[1..];
                    } else {
                        skip_count += 2;
                    }
                }
                _ => unreachable!("Bad id"),
            }
        }
        pos
    }

    #[inline]
    /// Return the suffix after skipping `index` elements, preserving the slice type.
    /// The caller must ensure `index <= self.len()`.
    pub fn fast_forward(&self, index: usize) -> ListSlice<'a> {
        if index == 0 {
            return *self;
        }

        let mut pos = self.data;

        pos = Self::skip(pos, index);

        ListSlice {
            data: pos,
            length: self.length - index,
            slice_type: self.slice_type,
        }
    }

    fn get_entry(start: &[u8]) -> (AtomView<'_>, &[u8]) {
        let start_id = start[0] & TYPE_MASK;
        let end = Self::skip(start, 1);
        let len = unsafe { end.as_ptr().offset_from(start.as_ptr()) } as usize;

        let data = unsafe { start.get_unchecked(..len) };
        (AtomView::from_type(data, start_id), end)
    }

    #[inline]
    /// Return the first element and the remaining slice. The slice must be nonempty.
    pub fn pop_first(&self) -> (AtomView<'a>, ListSlice<'a>) {
        let (res, end) = Self::get_entry(self.data);

        let slice = ListSlice {
            data: end,
            length: self.length - 1,
            slice_type: self.slice_type,
        };

        (res, slice)
    }

    #[inline]
    /// Return the number of expressions in the slice.
    pub fn len(&self) -> usize {
        self.length
    }

    #[inline]
    /// Borrow the element at `index`. The caller must ensure `index < self.len()`.
    pub fn get(&self, index: usize) -> AtomView<'a> {
        let start = self.fast_forward(index);
        Self::get_entry(start.data).0
    }

    /// Borrow the elements in the half-open range, preserving the slice type.
    /// The range must be ordered and contained in `0..self.len()`.
    pub fn get_subslice(&self, range: std::ops::Range<usize>) -> Self {
        let start = self.fast_forward(range.start);

        let mut s = start.data;
        s = Self::skip(s, range.len());

        let len = unsafe { s.as_ptr().offset_from(start.data.as_ptr()) } as usize;
        ListSlice {
            data: &start.data[..len],
            length: range.len(),
            slice_type: self.slice_type,
        }
    }

    #[inline]
    /// Return the operation from which this slice was taken.
    pub fn get_type(&self) -> SliceType {
        self.slice_type
    }

    #[inline]
    /// Create a one-element slice containing `view`.
    pub fn from_one(view: AtomView<'a>) -> Self {
        ListSlice {
            data: view.get_data(),
            length: 1,
            slice_type: SliceType::One,
        }
    }

    #[inline]
    /// Create an empty slice with [`SliceType::Empty`].
    pub fn empty() -> Self {
        ListSlice {
            data: &[],
            length: 0,
            slice_type: SliceType::Empty,
        }
    }

    #[inline]
    /// Iterate over the expressions in this slice.
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
            let (res, end) = ListSlice::get_entry(self.data.data);
            self.data = ListSlice {
                data: end,
                length: self.data.length - 1,
                slice_type: self.data.slice_type,
            };

            Some(res)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse, symbol};

    #[test]
    fn normalized_pair_constructor_reuses_buffer() {
        let left = parse!("pair_constructor_left");
        let right = parse!("pair_constructor_right");
        let mut terms = [left.as_view(), right.as_view()];
        terms.sort_by(|a, b| a.get_term_cmp_slice().cmp(b.get_term_cmp_slice()));
        let mut buffer = parse!("pair_constructor_previous(x)").into_raw();
        buffer.reserve(128);
        let pointer = buffer.as_ptr();
        let capacity = buffer.capacity();

        let sum = Add::new_normalized_pair_into(terms[0].get_data(), terms[1].get_data(), buffer);
        assert_eq!(sum.data.as_ptr(), pointer);
        assert_eq!(sum.data.capacity(), capacity);
        assert_eq!(sum.to_add_view().iter().collect::<Vec<_>>(), terms);
        assert!(!sum.as_view().needs_normalization());
    }

    #[test]
    fn normalized_pair_headers_cover_payload_widths() {
        let right = parse!("pair_header_right");
        let child = Atom::num(16);
        let mut sum = Add::new();
        let mut header_sizes = Vec::new();
        for count in [0, 80, 81, 82, 100, 21_840, 21_841, 21_842, 22_000] {
            let mut left = Atom::new();
            let function = left.to_fun(symbol!("pair_header_left"));
            for _ in 0..count {
                function.add_arg(child.as_view());
            }
            function.set_normalized(true);
            let mut terms = [left.as_view(), right.as_view()];
            terms.sort_by(|a, b| a.get_term_cmp_slice().cmp(b.get_term_cmp_slice()));
            let payload_len = terms
                .iter()
                .map(|term| term.get_data().len())
                .sum::<usize>();

            sum.set_normalized_pair(terms[0].get_data(), terms[1].get_data());
            let (nargs, size, payload) = read_packed_pair(&sum.data[1..]);
            assert_eq!((nargs, size as usize), (2, payload_len));
            assert_eq!(sum.data[0], ADD_ID);
            assert_eq!(sum.to_add_view().iter().collect::<Vec<_>>(), terms);
            header_sizes.push(sum.data.len() - payload.len());

            let mut appended = Add::new();
            appended.extend(terms[0]);
            appended.extend(terms[1]);
            appended.set_normalized(true);
            assert_eq!(sum.data.as_slice(), appended.data.as_slice());
        }
        header_sizes.sort_unstable();
        header_sizes.dedup();
        assert_eq!(header_sizes, vec![4, 5, 7]);
    }

    #[test]
    fn growing_sum_capacity_counts_from_its_length() {
        let mut sum = Add::new();
        sum.data.reserve(64);
        let target = sum.data.capacity() + 1;
        sum.grow_capacity(target);
        assert!(sum.data.capacity() >= target);
        assert_eq!(sum.get_nargs(), 0);
    }

    #[test]
    fn compact_variable_power_boundaries_and_slices() {
        for id in [1, 255, 256, 65535, 65536, u32::MAX] {
            for (flags, extra) in [(0, 0), (SYM_SCALAR_FLAG, SYM_EXTRA_POSITIVE_FLAG)] {
                let base = InlineVar::new(Symbol::decode_flags(id, flags, extra));
                for exp in -17..=16 {
                    let number = InlineNum::new(exp, 1);
                    let power = Pow::new_into(base.as_view(), number.as_view(), Vec::new());
                    let view = power.to_pow_view();
                    let compressed = (-15..=15).contains(&exp);
                    assert_eq!(
                        view.get_byte_size(),
                        1 + base.get_data().len() + if compressed { 1 } else { 3 }
                    );
                    assert_eq!(view.get_base_exp(), (base.as_view(), number.as_view()));
                    assert_eq!(
                        view.iter().collect::<Vec<_>>(),
                        [base.as_view(), number.as_view()]
                    );
                    assert_eq!(view.iter().nth(1), Some(number.as_view()));
                    assert_eq!(view.iter().nth(2), None);
                    let slice = view.to_slice();
                    assert_eq!(slice.get(0), base.as_view());
                    assert_eq!(slice.get(1), number.as_view());
                    assert_eq!(slice.fast_forward(2).len(), 0);
                    assert_eq!(
                        slice.get_subslice(1..2).iter().collect::<Vec<_>>(),
                        [number.as_view()]
                    );
                    assert_eq!(slice.pop_first().1.pop_first().0, number.as_view());
                    assert_eq!(AtomView::from(view.as_view().get_data()), view.as_view());
                }
            }
        }

        let compact = parse!("x^2");
        let regular = parse!("f(x)^2");
        let (AtomView::Pow(compact), AtomView::Pow(regular)) =
            (compact.as_view(), regular.as_view())
        else {
            unreachable!()
        };
        assert_eq!(
            compact.to_slice().get_subslice(1..2),
            regular.to_slice().get_subslice(1..2)
        );
        assert_ne!(compact.to_slice(), regular.to_slice());

        // Rational and complex exponents must not be mistaken for integers.
        for text in [
            "x^(1/2)",
            "x^(-3/2)",
            "x^(2+3*𝑖)",
            "f(x)^2",
            "(x+y)^2",
            "x^y",
        ] {
            let expr = parse!(text);
            let AtomView::Pow(p) = expr.as_view() else {
                panic!("{text}")
            };
            let (b, e) = p.get_base_exp();
            assert_eq!(p.get_byte_size(), 1 + b.get_byte_size() + e.get_byte_size());
        }
    }

    #[test]
    fn compact_power_enclosing_lists_preserve_boundaries_and_borrows() {
        fn check_borrowed_list(atom: AtomView<'_>, expected: &[AtomView<'_>]) {
            let (slice, iter) = match atom {
                AtomView::Fun(f) => (f.to_slice(), f.iter()),
                AtomView::Mul(m) => (m.to_slice(), m.iter()),
                AtomView::Add(a) => (a.to_slice(), a.iter()),
                _ => unreachable!(),
            };
            // Derive boundaries from the original children, independently of
            // either scanner and the enclosing list's decoded byte lengths.
            let mut offsets = vec![0];
            for child in expected {
                offsets.push(offsets.last().unwrap() + child.get_data().len());
            }
            let data = atom.get_data();
            let payload = &data[data.len() - offsets[expected.len()]..];
            assert_eq!(slice.get_data().as_ptr(), payload.as_ptr());
            assert_eq!(slice.len(), expected.len());

            let check_child = |child: AtomView<'_>, index: usize| {
                assert_eq!(child.get_data(), expected[index].get_data());
                assert_eq!(
                    child.get_data().as_ptr(),
                    payload[offsets[index]..].as_ptr()
                );
            };
            let mut remaining = slice;
            let mut sequential = iter;
            for index in 0..expected.len() {
                check_child(sequential.next().unwrap(), index);
                check_child(iter.clone().nth(index).unwrap(), index);
                check_child(slice.get(index), index);
                let suffix = slice.fast_forward(index);
                assert_eq!(
                    suffix.get_data().as_ptr(),
                    payload[offsets[index]..].as_ptr()
                );
                assert_eq!(suffix.len(), expected.len() - index);
                check_child(suffix.get(0), index);
                let (child, rest) = remaining.pop_first();
                check_child(child, index);
                assert_eq!(
                    rest.get_data().as_ptr(),
                    payload[offsets[index + 1]..].as_ptr()
                );
                remaining = rest;
            }
            assert!(sequential.next().is_none());
            assert!(iter.clone().nth(expected.len()).is_none());
            assert_eq!(remaining.len(), 0);
            let end = slice.fast_forward(expected.len());
            assert_eq!(end.len(), 0);
            assert_eq!(end.get_data().as_ptr(), payload[payload.len()..].as_ptr());

            for start in 0..=expected.len() {
                for end in start..=expected.len() {
                    let sub = slice.get_subslice(start..end);
                    assert_eq!(sub.len(), end - start);
                    assert_eq!(sub.get_data(), &payload[offsets[start]..offsets[end]]);
                    assert_eq!(sub.get_data().as_ptr(), payload[offsets[start]..].as_ptr());
                    assert_eq!(sub.iter().count(), end - start);
                    for (index, child) in sub.iter().enumerate() {
                        check_child(child, start + index);
                    }
                }
            }
        }

        let all_extra = SYM_EXTRA_REAL_FLAG
            | SYM_EXTRA_INTEGER_FLAG
            | SYM_EXTRA_POSITIVE_FLAG
            | SYM_EXTRA_WILDCARD_LEVEL_3
            | SYM_EXTRA_FLAT_FLAG;
        let attributes = [
            (0, 0),
            (SYM_SCALAR_FLAG, 0),
            (SYM_LINEAR_FLAG | SYM_SYMMETRIC_FLAG, SYM_EXTRA_REAL_FLAG),
            (
                SYM_ANTISYMMETRIC_FLAG,
                SYM_EXTRA_INTEGER_FLAG | SYM_EXTRA_POSITIVE_FLAG,
            ),
            (SYM_CYCLESYMMETRIC_FLAG | SYM_SCALAR_FLAG, all_extra),
        ];
        let zero = InlineNum::zero();
        let fraction = InlineNum::new(1, 2);
        let large = InlineNum::new(16, 1);
        // These fabricated symbols cover every supported ID width. Raw
        // constructors avoid registering or normalizing these expressions.
        for id in [0, 1, 255, 256, 65535, 65536, u32::MAX] {
            for (flags, extra) in attributes {
                let symbol = Symbol::decode_flags(id, flags, extra);
                let variable = InlineVar::new(symbol);
                for exponent in [-16, -15, -1, 0, 1, 15, 16] {
                    let number = InlineNum::new(exponent, 1);
                    for normalized in [false, true] {
                        let mut power =
                            Pow::new_into(variable.as_view(), number.as_view(), Vec::new());
                        power.set_normalized(normalized);
                        let nested = Pow::new_into(power.as_view(), power.as_view(), Vec::new());
                        let nested =
                            Pow::new_into(nested.as_view(), variable.as_view(), Vec::new());
                        let ordinary =
                            Pow::new_into(variable.as_view(), large.as_view(), Vec::new());
                        let mut inner = Fun::new_into(symbol, Vec::new());
                        inner.add_args(&[power.as_view(), variable.as_view(), zero.as_view()]);
                        let children = [
                            power.as_view(),
                            variable.as_view(),
                            zero.as_view(),
                            nested.as_view(),
                            fraction.as_view(),
                            ordinary.as_view(),
                            inner.as_view(),
                            variable.as_view(),
                            power.as_view(),
                        ];
                        let mut fun = Fun::new_into(symbol, Vec::new());
                        fun.add_args(&children);
                        let mut mul = Mul::new();
                        let mut add = Add::new();
                        for child in children {
                            mul.extend(child);
                            add.extend(child);
                        }
                        for list in [fun.as_view(), mul.as_view(), add.as_view()] {
                            check_borrowed_list(list, &children);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn compact_powers_in_mixed_lists_and_algebra() {
        use crate::atom::AtomCore;
        let expr = parse!("f(x^2,y^-16,z^15,w^16,t^-17,(x+y)^2,g(x^-1),3*x^3,0)");
        let AtomView::Fun(f) = expr.as_view() else {
            unreachable!()
        };
        let expected: Vec<_> = f.iter().collect();
        check_list(expr.as_view(), &expected);
        for (i, child) in expected.iter().enumerate() {
            assert_eq!(f.iter().nth(i), Some(*child));
            assert_eq!(f.to_slice().fast_forward(i).get(0), *child);
        }
        assert_eq!((parse!("x^7") * parse!("x^8")), parse!("x^15"));
        assert_eq!((parse!("x^15") * parse!("x")), parse!("x^16"));
        assert_eq!((parse!("x^-16") / parse!("x")), parse!("x^-17"));
        assert_eq!(
            parse!("(x+y)^3").expand(),
            parse!("x^3+3*x^2*y+3*x*y^2+y^3")
        );
        assert_eq!(
            parse!("f(x^2,y^-16)")
                .replace(parse!("x"))
                .with(parse!("z")),
            parse!("f(z^2,y^-16)")
        );
    }

    // Reconstruct format 5 without using the current number builder.
    fn legacy_power_bytes(atom: AtomView<'_>, format: u8) -> Vec<u8> {
        assert_eq!(format, 5);
        let mut data = vec![atom.get_data()[0]];
        match atom {
            AtomView::Num(n) => {
                data[0] = NUM_ID;
                n.get_coeff_view().to_owned().write_packed(&mut data);
            }
            AtomView::Pow(p) => {
                data[0] &= !POW_HAS_SMALL_INT_FLAG;
                let (base, exp) = p.get_base_exp();
                data.extend(legacy_power_bytes(base, format));
                data.extend(legacy_power_bytes(exp, format));
            }
            AtomView::Fun(f) => {
                data.extend([1, 0]); // provisional packed size
                let metadata = function_metadata(f.data);
                let (_, _, children) = read_packed_pair(metadata);
                data.extend_from_slice(&metadata[..metadata.len() - children.len()]);
                for child in f.iter() {
                    data.extend(legacy_power_bytes(child, format));
                }
                update_function_size(&mut data, 3);
            }
            AtomView::Mul(_) | AtomView::Add(_) => {
                let (count, _, children) = read_packed_pair(&atom.get_data()[1..]);
                data.extend([0x11, 0, 0]);
                for child in (ListIterator {
                    data: children,
                    length: count as usize,
                }) {
                    data.extend(legacy_power_bytes(child, format));
                }
                update_list_header(&mut data, 4, count);
            }
            _ => return atom.get_data().to_vec(),
        }
        data
    }

    #[test]
    fn compact_power_current_and_legacy_serialization() {
        use crate::atom::AtomCore;
        for text in [
            "0",
            "1",
            "-15",
            "-16",
            "15",
            "16",
            "x^2",
            "y^-16",
            "f(0,1,-15,-16,15,16,1/2,2+3*𝑖,f(x)^2)",
            "f(x^2,y^-16,z^16,3*x^15)+x^-2+y^3",
            "x^2+x^16+x^3",
            "1.25",
            "f(0.0,1.25,x^2,2.5+3.75*𝑖)",
        ] {
            let atom = parse!(text);
            for (format, data) in [
                (5, legacy_power_bytes(atom.as_view(), 5)),
                (ATOM_EXPORT_FORMAT, atom.as_view().get_data().to_vec()),
            ] {
                let mut bytes = vec![format];
                bytes.extend_from_slice(&(data.len() as u64).to_le_bytes());
                bytes.extend_from_slice(&data);
                assert_eq!(
                    Atom::import_with_map(&mut &bytes[..], &StateMap::default()).unwrap(),
                    atom
                );
                #[cfg(feature = "bincode")]
                {
                    let (decoded, len): (Atom, usize) = bincode::decode_from_slice_with_context(
                        &bytes,
                        bincode::config::standard(),
                        StateMap::default(),
                    )
                    .unwrap();
                    assert_eq!(len, bytes.len());
                    assert_eq!(decoded, atom);
                }
                let mut full = Vec::new();
                State::export_partial(&mut full, atom.get_all_symbols(true)).unwrap();
                if format < ATOM_EXPORT_FORMAT {
                    full[4..6].copy_from_slice(&(format as u16 + 5).to_le_bytes());
                }
                full.extend_from_slice(&1u64.to_le_bytes());
                full.extend_from_slice(&bytes);
                assert_eq!(Atom::import(&mut &full[..], None).unwrap(), atom);
            }
        }
        for format in [0, 1, 2, 3, 4, 7, 255] {
            assert!(check_atom_format(format).is_err());
        }
    }

    fn check_list(atom: AtomView<'_>, expected: &[AtomView<'_>]) {
        let slice = match atom {
            AtomView::Fun(f) => f.to_slice(),
            AtomView::Mul(m) => m.to_slice(),
            AtomView::Add(a) => a.to_slice(),
            _ => panic!("expected a list"),
        };
        assert_eq!(slice.len(), expected.len());
        assert_eq!(slice.iter().collect::<Vec<_>>(), expected);
        let mut iter = match atom {
            AtomView::Fun(f) => f.iter(),
            AtomView::Mul(m) => m.iter(),
            AtomView::Add(a) => a.iter(),
            _ => unreachable!(),
        };
        assert_eq!(iter.size_hint(), (expected.len(), Some(expected.len())));
        iter.next();
        let remaining = expected.len().saturating_sub(1);
        assert_eq!(iter.size_hint(), (remaining, Some(remaining)));
        assert_eq!(slice.fast_forward(expected.len()).len(), 0);
        assert_eq!(ListIterator::from_one(atom).next().unwrap(), atom);
        if !expected.is_empty() {
            let last = expected.len() - 1;
            assert_eq!(slice.get(last), expected[last]);
            assert_eq!(slice.get_subslice(last..last + 1).get(0), expected[last]);
        }
    }

    #[test]
    fn packed_length_boundaries() {
        use super::super::coefficient::{read_packed_denominator, read_packed_u64};
        let values = [
            0,
            1,
            255,
            256,
            65535,
            65536,
            u32::MAX as u64,
            1u64 << 32,
            u64::MAX,
        ];
        for size in values {
            let mut data = Vec::new();
            (size, 1).write_packed(&mut data);
            data.push(123);
            assert_eq!(read_packed_u64(&data), (size, &[123][..]));
            assert_eq!(read_packed_numerator(&data), size);
            for count in values {
                data.clear();
                (count, size).write_packed(&mut data);
                data.push(123);
                assert_eq!(read_packed_denominator(&data), (size, &[123][..]));
                assert_eq!(read_packed_pair(&data), (count, size, &[123][..]));
                assert_eq!(read_packed_numerator(&data), count);
            }
        }
    }

    #[test]
    fn packed_skip_load_boundaries() {
        // Exercise each width, including non-minimal encodings and a buffer
        // that ends exactly at the payload's last byte.
        for size in [0u64, 1, 2, 3, 7, 8, 9, 255, 256, 65535, 65536] {
            for width_tag in 1..=4u8 {
                let width = 1usize << (width_tag - 1);
                if width < 8 && size >= 1u64 << (8 * width) {
                    continue;
                }
                let mut fun = vec![width_tag];
                fun.extend_from_slice(&size.to_le_bytes()[..width]);
                fun.resize(fun.len() + size as usize, 0xff);
                let fun = fun.into_boxed_slice();
                assert!(unsafe { skip_packed_function(&fun) }.is_empty());
                for count_tag in 1..=4u8 {
                    let count_width = 1usize << (count_tag - 1);
                    let mut list = vec![count_tag | (width_tag << 4)];
                    list.extend_from_slice(&1u64.to_le_bytes()[..count_width]);
                    list.extend_from_slice(&size.to_le_bytes()[..width]);
                    list.resize(list.len() + size as usize, 0xff);
                    let list = list.into_boxed_slice();
                    assert!(unsafe { skip_packed_list(&list) }.is_empty());
                    if size == 1 {
                        let mut implicit = vec![count_tag];
                        implicit.extend_from_slice(&1u64.to_le_bytes()[..count_width]);
                        implicit.push(0xff);
                        assert!(unsafe { skip_packed_list(&implicit) }.is_empty());
                    }
                }
            }
        }
    }

    #[test]
    fn batch_function_append_and_unwind() {
        let x = parse!("x");
        let head = symbol!("batch_function_append");
        for count in [0, 1, 2, 84, 85, 255, 256, 22000, 65536] {
            let mut incremental = Fun::new_into(head, Vec::new());
            for _ in 0..count {
                incremental.add_arg(x.as_view());
            }
            let mut batch = Fun::new_into(head, Vec::new());
            batch.add_args_iter((0..count).map(|_| AtomOrView::from(x.clone())));
            assert_eq!(batch, incremental);

            let before = batch.clone();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                batch.add_args_iter((0..10).map(|i| {
                    assert_ne!(i, 5, "test iterator panic");
                    AtomOrView::from(x.as_view())
                }));
            }));
            assert!(result.is_err());
            assert_eq!(batch, before);
            batch.add_arg(x.as_view());
            incremental.add_arg(x.as_view());
            assert_eq!(batch, incremental);
        }
    }

    #[test]
    fn growing_packed_headers() {
        let x = parse!("x");
        let mut fun = Fun::new_into(symbol!("packed_headers"), Vec::new());
        let mut mul = Mul::new();
        let mut add = Add::new();
        check_list(fun.as_view(), &[]);
        check_list(mul.as_view(), &[]);
        check_list(add.as_view(), &[]);
        for i in 1..=65537 {
            fun.add_arg(x.as_view());
            mul.extend(x.as_view());
            add.extend(x.as_view());
            if [
                1, 2, 83, 84, 85, 86, 255, 256, 257, 21843, 21844, 21845, 21846, 65535, 65536,
                65537,
            ]
            .contains(&i)
            {
                let expected = vec![x.as_view(); i];
                for atom in [fun.as_view(), mul.as_view(), add.as_view()] {
                    check_list(atom, &expected);
                }
                assert_eq!(fun.to_fun_view().iter().count(), i);
                assert_eq!(mul.to_mul_view().iter().count(), i);
                assert_eq!(add.to_add_view().iter().count(), i);
            }
        }
        let mut batched = Fun::new_into(fun.get_symbol(), Vec::new());
        batched.add_args(&vec![x.as_view(); 65537]);
        assert_eq!(batched, fun);
        let mut merged = Mul::new();
        merged.extend(mul.as_view());
        assert_eq!(merged.as_view(), mul.as_view());
    }

    #[test]
    fn replace_first_across_header_widths() {
        let x = parse!("x");
        let small = Atom::num(2);
        let mut large = Fun::new_into(symbol!("large_first"), Vec::new());
        large.add_args(&vec![x.as_view(); 22000]);
        let mut medium = Fun::new_into(large.get_symbol(), Vec::new());
        medium.add_args(&vec![x.as_view(); 100]);
        let mut product = Mul::new();
        product.extend(small.as_view());
        product.extend(x.as_view());
        product.set_has_coefficient(true);
        for first in [
            large.as_view(),
            medium.as_view(),
            small.as_view(),
            medium.as_view(),
            large.as_view(),
            small.as_view(),
        ] {
            product.replace_first(first);
            check_list(product.as_view(), &[first, x.as_view()]);
            assert!(product.to_mul_view().has_coefficient());
            assert_eq!(product.to_mul_view().get_coefficient(), Some(first));
            let mut fresh = Mul::new();
            fresh.extend(first);
            fresh.extend(x.as_view());
            fresh.set_has_coefficient(true);
            assert_eq!(product.as_view(), fresh.as_view());
        }
    }

    #[test]
    fn rejects_previous_atom_formats_before_reading_payload() {
        for format in [0, 1, 2, 3, 4, 7, u8::MAX] {
            let mut atom = Atom::num(42);
            let mut bytes = &[format][..];
            let error = atom.read(&mut bytes).unwrap_err();
            assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
            assert_eq!(atom, Atom::num(42));

            #[cfg(feature = "bincode")]
            assert!(
                bincode::decode_from_slice_with_context::<_, Atom, _>(
                    &[format],
                    bincode::config::standard(),
                    StateMap::default()
                )
                .is_err()
            );
        }
    }

    #[test]
    fn inline_import_validation_rejects_noncanonical_numbers_and_power_flags() {
        let variable = InlineVar::new(Symbol::decode_flags(17, 0, 0));
        let base = variable.get_data();
        let mut invalid = Vec::new();
        for value in -15i64..=15 {
            let mut ordinary = vec![NUM_ID];
            Coefficient::from(value).write_packed(&mut ordinary);
            assert!(validate_atom_data_format(&ordinary, 5).is_ok());
            invalid.push(ordinary);
            let number = InlineNum::new(value, 1);
            assert!(validate_atom_data(number.get_data()).is_ok());
            assert!(validate_atom_data_format(number.get_data(), 5).is_err());
            let mut missing_flag = vec![POW_ID];
            missing_flag.extend_from_slice(base);
            missing_flag.extend_from_slice(number.get_data());
            invalid.push(missing_flag);
            for normalized in [false, true] {
                let mut power = Pow::new_into(variable.as_view(), number.as_view(), Vec::new());
                power.set_normalized(normalized);
                assert!(validate_atom_data(&power.data).is_ok());
                assert!(validate_atom_data_format(&power.data, 5).is_err());
                let mut stream = vec![ATOM_EXPORT_FORMAT];
                stream.extend_from_slice(&(power.data.len() as u64).to_le_bytes());
                stream.extend_from_slice(&power.data);
                let mut restored = Atom::new();
                restored.read(&mut stream.as_slice()).unwrap();
                // Raw reads preserve dirty powers, including temporary exponents 0/1.
                assert_eq!(restored.as_view().get_data(), power.data);
            }
        }
        invalid.push(vec![NUM_ID, 2, 2, 0]); // A wider packed small integer.
        let mut signed_variable = base.to_vec();
        signed_variable[1] |= 0x80;
        assert!(validate_atom_data_format(&signed_variable, 5).is_err());
        invalid.push(signed_variable.clone());
        for children in [
            vec![],
            base.to_vec(),
            [base, base].concat(),
            [base, &[NUM_ID, 1, 16]].concat(),
            [InlineNum::new(2, 1).get_data(), base].concat(),
            [signed_variable.as_slice(), InlineNum::new(2, 1).get_data()].concat(),
        ] {
            let mut raw = vec![POW_ID | POW_HAS_SMALL_INT_FLAG];
            raw.extend(children);
            invalid.push(raw);
        }
        for raw in invalid {
            assert!(validate_atom_data(&raw).is_err(), "accepted {raw:?}");
            let mut stream = vec![ATOM_EXPORT_FORMAT];
            stream.extend_from_slice(&(raw.len() as u64).to_le_bytes());
            stream.extend_from_slice(&raw);
            assert!(Atom::new().read(&mut stream.as_slice()).is_err());
            #[cfg(feature = "bincode")]
            assert!(
                bincode::decode_from_slice_with_context::<_, Atom, _>(
                    &stream,
                    bincode::config::standard(),
                    StateMap::default()
                )
                .is_err()
            );
        }
    }

    #[test]
    fn format_five_conversion_uses_an_explicit_stack() {
        let depth = 10_000;
        let mut data = vec![POW_ID; depth];
        for _ in 0..=depth {
            data.extend([NUM_ID, 1, 0]);
        }
        assert!(validate_atom_data_format(&data, 5).is_ok());
        let converted = convert_legacy_atom(&data);
        assert_eq!(converted.len(), 2 * depth + 1);
        assert!(
            converted[..depth]
                .iter()
                .all(|&tag| tag == POW_ID | NOT_NORMALIZED)
        );
        assert!(validate_atom_data(&converted).is_ok());
    }

    #[test]
    fn imports_reject_malformed_binary_floats_before_traversal() {
        let component = |precision: u32, value: &[u8]| {
            let mut bytes = precision.to_le_bytes().to_vec();
            bytes.extend_from_slice(value);
            bytes
        };
        let coefficient = |real: &[u8], imaginary: &[u8]| {
            let mut bytes = vec![NUM_ID, 11]; // Compact floating coefficient.
            (real.len() as u64, imaginary.len() as u64).write_packed(&mut bytes);
            bytes.extend_from_slice(real);
            bytes.extend_from_slice(imaginary);
            bytes
        };
        let finite = |exponent: i64| {
            let mut bytes = vec![1];
            let mut encoded = ((exponent as u64) << 1) ^ ((exponent >> 63) as u64);
            while encoded >= 128 {
                bytes.push(encoded as u8 | 128);
                encoded >>= 7;
            }
            bytes.push(encoded as u8);
            bytes.push(128);
            bytes
        };
        let valid = component(128, &[0]);
        let mut invalid = vec![
            coefficient(&component(0, &[0]), &[]),
            coefficient(&128u32.to_le_bytes(), &[]), // A real component cannot be implicit.
            coefficient(&valid, &0u32.to_le_bytes()),
            coefficient(&valid, &[0; 3]),
            coefficient(&valid, &[0; 5]),
            coefficient(&valid, &component(128, &[7])),
            // An exact zero either inherits the real precision or stores only
            // its distinct precision; full zero payloads are never canonical.
            coefficient(&valid, &128u32.to_le_bytes()),
            coefficient(&valid, &component(128, &[0])),
            coefficient(&valid, &component(256, &[0])),
        ];
        for flags in 1u8..=31 {
            let mut raw = coefficient(&valid, &[]);
            raw[0] |= flags << 3;
            invalid.push(raw);
        }
        for header in [
            vec![0x91, valid.len() as u8, 0], // Unsigned lengths cannot have a sign bit.
            vec![0x12, valid.len() as u8, 0, 0], // A wider real length is redundant.
            vec![0x21, valid.len() as u8, 0, 0], // A wider imaginary length is redundant.
        ] {
            let mut raw = vec![NUM_ID, 11];
            raw.extend_from_slice(&header);
            raw.extend_from_slice(&valid);
            invalid.push(raw);
        }
        let mut fixed = vec![NUM_ID, 9];
        for part in [&valid, &valid] {
            fixed.extend_from_slice(&(part.len() as u64).to_le_bytes());
            fixed.extend_from_slice(part);
        }
        invalid.push(fixed);
        #[cfg(feature = "float-mpfr")]
        invalid.push(coefficient(
            // MPFR cannot preserve 72 significant bits at precision 64;
            // Astro can preserve a larger actual value and a precision hint.
            &component(64, &[1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 128]),
            &[],
        ));
        #[cfg(feature = "float-mpfr")]
        invalid.push(coefficient(
            // Astro's finite exponent range extends beyond the default MPFR range.
            &component(128, &finite(i64::from(i32::MAX) - 64)),
            &[],
        ));
        for value in [
            vec![],
            vec![6], // NaN has one canonical tag, regardless of its original sign.
            vec![7],
            vec![0, 0],
            vec![3, 0],
            vec![1],
            vec![1, 0],
            vec![1, 0, 127],      // The significand must be normalized.
            vec![1, 0, 0, 128],   // Low zero bytes must be omitted.
            vec![1, 128, 0, 128], // Noncanonical exponent varint.
            finite(i64::MAX),
            finite(i64::MIN),
        ] {
            invalid.push(coefficient(&component(128, &value), &[]));
        }
        let mut truncated = vec![NUM_ID, 11];
        (u64::MAX, 0).write_packed(&mut truncated);
        invalid.push(truncated);

        let mut state = Vec::new();
        State::export_partial(&mut state, Default::default()).unwrap();
        state.write_u64::<LittleEndian>(1).unwrap();
        for raw in invalid {
            let mut sum = vec![ADD_ID];
            (1u64, raw.len() as u64).write_packed(&mut sum);
            sum.extend_from_slice(&raw);
            let mut power = vec![POW_ID];
            power.extend_from_slice(&raw);
            power.extend_from_slice(&ZERO_DATA);
            let mut metadata = Vec::new();
            (0u64, 1).write_packed(&mut metadata);
            let mut function = vec![FUN_ID];
            ((metadata.len() + raw.len()) as u64, 1).write_packed(&mut function);
            function.extend_from_slice(&metadata);
            function.extend_from_slice(&raw);
            for raw in [raw, sum, power, function] {
                let mut stream = vec![ATOM_EXPORT_FORMAT];
                stream.extend_from_slice(&(raw.len() as u64).to_le_bytes());
                stream.extend_from_slice(&raw);
                let mut atom = Atom::new();
                assert_eq!(
                    atom.read(&mut stream.as_slice()).unwrap_err().kind(),
                    std::io::ErrorKind::InvalidData
                );
                assert_eq!(
                    Atom::import_with_map(&mut stream.as_slice(), &StateMap::default())
                        .unwrap_err()
                        .kind(),
                    std::io::ErrorKind::InvalidData
                );
                let mut exported = state.clone();
                exported.extend_from_slice(&stream);
                assert_eq!(
                    Atom::import(&mut exported.as_slice(), None)
                        .unwrap_err()
                        .kind(),
                    std::io::ErrorKind::InvalidData
                );
                #[cfg(feature = "bincode")]
                assert!(
                    bincode::decode_from_slice_with_context::<_, Atom, _>(
                        &stream,
                        bincode::config::standard(),
                        StateMap::default()
                    )
                    .is_err()
                );
            }
        }
    }

    #[test]
    fn import_validation_checks_boundaries_without_recursive_descent() {
        for data in [
            vec![],
            vec![0],
            vec![NUM_ID],
            vec![NUM_ID, 0],
            vec![VAR_ID, 0],
            vec![ADD_ID, 0x11, 0, 1],
            vec![ADD_ID, 0x11, 1, 0],
            vec![FUN_ID, 0x11, 0, 1],
            vec![POW_ID, NUM_ID, 1, 0],
        ] {
            assert!(validate_atom_data(&data).is_err(), "{data:?}");
        }
        let mut nested = vec![POW_ID; 10_000];
        nested.extend_from_slice(&ZERO_DATA);
        for _ in 0..10_000 {
            nested.extend_from_slice(&ZERO_DATA);
        }
        assert!(validate_atom_data(&nested).is_ok());
        nested.pop();
        assert!(validate_atom_data(&nested).is_err());
    }

    #[test]
    fn packed_lengths_roundtrip() {
        for expr in [
            "0",
            "x",
            "f()",
            "f(x)",
            "f(x,y)",
            "3*x*y",
            "x+y",
            "f((x+y)^(2*x),g(x,3*y))+2*x",
        ] {
            let atom = parse!(expr);
            let mut bytes = Vec::new();
            atom.as_view().write(&mut bytes).unwrap();
            assert_eq!(bytes[0], ATOM_EXPORT_FORMAT);
            assert_eq!(
                &bytes[1..9],
                &(atom.as_view().get_data().len() as u64).to_le_bytes()
            );
            assert_eq!(&bytes[9..], atom.as_view().get_data());
            let mut decoded = Atom::new();
            decoded.read(&mut &bytes[..]).unwrap();
            assert_eq!(decoded, atom);

            #[cfg(feature = "bincode")]
            {
                let config = bincode::config::standard();
                let encoded = bincode::encode_to_vec(&atom, config).unwrap();
                assert_eq!(encoded, bytes);
                let (decoded, read): (Atom, usize) =
                    bincode::decode_from_slice_with_context(&encoded, config, StateMap::default())
                        .unwrap();
                assert_eq!(read, encoded.len());
                assert_eq!(decoded, atom);
            }

            let mut exported = Vec::new();
            atom.as_view().export(&mut exported).unwrap();
            assert_eq!(Atom::import(&mut &exported[..], None).unwrap(), atom);
        }
    }

    #[test]
    fn import_normalizes_after_remapping() {
        // This ID is only meaningful in the source state. Normalizing before
        // remapping would try to look up a nonexistent function in this state.
        let foreign_head = Symbol::decode_flags(u32::MAX, 0, 0);
        let head = symbol!("remapped_function");
        let x = parse!("x");
        let mut fun = Fun::new_into(foreign_head, Vec::new());
        fun.add_arg(x.as_view());
        fun.set_normalized(true);
        let mut bytes = Vec::new();
        fun.as_view().write(&mut bytes).unwrap();
        let mut state_map = StateMap::default();
        state_map.symbols.insert(foreign_head.get_id(), head);
        let expected = parse!("remapped_function(x)");
        let imported = Atom::import_with_map(&mut &bytes[..], &state_map).unwrap();
        assert_eq!(imported, expected);
        assert!(!imported.as_view().needs_normalization());

        // User data is read while importing symbol definitions, before the
        // state map is complete. Defer normalization recursively, including keys.
        let mut user_data = vec![4]; // list
        user_data.put_u32_le(1);
        user_data.push(5); // map
        user_data.put_u32_le(1);
        for _ in 0..2 {
            user_data.push(3); // atom key/value
            user_data.extend_from_slice(&bytes);
        }
        let imported = UserData::read(&mut &user_data[..])
            .unwrap()
            .rename_symbols(&state_map);
        assert_eq!(
            imported,
            UserData::List(vec![UserData::Map(HashMap::from_iter([(
                UserDataKey::Atom(expected.clone()),
                UserData::Atom(expected.clone()),
            )]))])
        );

        #[cfg(feature = "bincode")]
        {
            let (imported, read): (Atom, usize) = bincode::decode_from_slice_with_context(
                &bytes,
                bincode::config::standard(),
                state_map,
            )
            .unwrap();
            assert_eq!(read, bytes.len());
            assert_eq!(imported, expected);
        }
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn iterator_count_is_not_truncated() {
        let count = u32::MAX as usize + 1;
        let iterator = ListIterator {
            data: &[],
            length: count,
        };
        assert_eq!(iterator.len(), count);
        assert_eq!(ExactSizeIterator::len(&iterator), count);
        assert_eq!(iterator.size_hint(), (count, Some(count)));
    }

    #[test]
    #[ignore = "allocates about 12 GiB to exercise actual >4 GiB atoms"]
    #[cfg(target_pointer_width = "64")]
    fn products_and_functions_over_four_gib() {
        // Every payload byte is a canonical inline one, so the large payload
        // consists entirely of valid atoms without needing billions of appends.
        let count = u32::MAX as u64 + 1;
        let mut data = vec![ADD_ID | NOT_NORMALIZED];
        (count, count).write_packed(&mut data);
        data.resize(data.len() + count as usize, inline_number_header(1));
        let add = Add { data };
        let x = Atom::num(2);
        let mut product = Mul::new();
        product.extend(add.as_view());
        product.extend(x.as_view());
        check_list(product.as_view(), &[add.as_view(), x.as_view()]);
        let mut fun = Fun::new_into(symbol!("large_packed_function"), Vec::new());
        fun.add_arg(product.as_view());
        fun.add_arg(x.as_view());
        check_list(fun.as_view(), &[product.as_view(), x.as_view()]);
        assert!(fun.to_fun_view().get_byte_size() > u32::MAX as usize);
    }

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
