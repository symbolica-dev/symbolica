//! Canonical binary central values. Precision and error are enclosing metadata.
//!
//! Finite payloads contain a sign/class byte, zigzag LEB128 i64 exponent E,
//! and a little-endian unsigned significand M with its high bit set and all
//! low zero bytes removed. The value is sign*M*2^(E-8*byte_count). This removes
//! native limb padding without depending on native endianness or word size.
//! Zero and NaN each have one encoding, independent of their in-memory sign.

use super::Float;
use std::cmp::Ordering;

#[cfg(feature = "float-mpfr")]
type Word = gmp_mpfr_sys::gmp::limb_t;
#[cfg(feature = "float-astro")]
type Word = astro_float::Word;

const ZERO: u8 = 0;
const POSITIVE: u8 = 1;
const NEGATIVE: u8 = 2;
const POS_INFINITY: u8 = 3;
const NEG_INFINITY: u8 = 4;
const NAN: u8 = 5;

struct Parts<'a> {
    words: &'a [Word],
    exponent: i64,
    skip: usize,
    bytes: usize,
    shift: u32,
}

impl<'a> Parts<'a> {
    fn new(words: &'a [Word], raw_exponent: i64) -> Self {
        let last = words.iter().rposition(|&word| word != 0).unwrap();
        let first = words.iter().position(|&word| word != 0).unwrap();
        let bits = last * Word::BITS as usize + (Word::BITS - words[last].leading_zeros()) as usize;
        let trailing = first * Word::BITS as usize + words[first].trailing_zeros() as usize;
        let shift = ((8 - bits % 8) % 8) as u32;
        let skip = (trailing + shift as usize) / 8;
        Self {
            words,
            exponent: raw_exponent - (words.len() * Word::BITS as usize - bits) as i64,
            skip,
            bytes: bits.div_ceil(8) - skip,
            shift,
        }
    }

    fn zigzag(&self) -> u64 {
        ((self.exponent as u64) << 1) ^ ((self.exponent >> 63) as u64)
    }

    fn size(&self) -> usize {
        1 + (64 - self.zigzag().leading_zeros()).max(1).div_ceil(7) as usize + self.bytes
    }

    fn write(&self, tag: u8, out: &mut Vec<u8>) {
        out.reserve(self.size());
        out.push(tag);
        let mut exponent = self.zigzag();
        loop {
            let byte = (exponent & 0x7f) as u8;
            exponent >>= 7;
            out.push(byte | if exponent == 0 { 0 } else { 0x80 });
            if exponent == 0 {
                break;
            }
        }
        if self.shift == 0 {
            // The common normal-number path copies whole limbs directly in a
            // defined byte order, without a temporary integer or byte vector.
            let limb_bytes = std::mem::size_of::<Word>();
            let first = self.skip / limb_bytes;
            let last = (self.skip + self.bytes - 1) / limb_bytes;
            for index in first..=last {
                let bytes = self.words[index].to_le_bytes();
                let start = self.skip.saturating_sub(index * limb_bytes);
                let end = (self.skip + self.bytes - index * limb_bytes).min(limb_bytes);
                out.extend_from_slice(&bytes[start..end]);
            }
        } else {
            // Astro subnormals need up to seven bits of normalization. The
            // common word-aligned path above does not pay for bytewise shifts.
            let byte = |index: usize| -> u8 {
                let limb_bytes = std::mem::size_of::<Word>();
                self.words
                    .get(index / limb_bytes)
                    .map_or(0, |word| (word >> ((index % limb_bytes) * 8)) as u8)
            };
            for index in self.skip..self.skip + self.bytes {
                let low = byte(index) << self.shift;
                let high = if index == 0 {
                    0
                } else {
                    byte(index - 1) >> (8 - self.shift)
                };
                out.push(low | high);
            }
        }
    }
}

enum Payload<'a> {
    Special(u8),
    Finite {
        negative: bool,
        exponent: i64,
        significand: &'a [u8],
    },
}

fn parse(data: &[u8]) -> Result<Payload<'_>, &'static str> {
    let Some(&tag) = data.first() else {
        return Err("Empty float payload");
    };
    if tag != POSITIVE && tag != NEGATIVE {
        return if matches!(tag, ZERO | POS_INFINITY | NEG_INFINITY | NAN) && data.len() == 1 {
            Ok(Payload::Special(tag))
        } else {
            Err("Invalid float class or special-value payload length")
        };
    }
    let mut zigzag = 0u64;
    let mut end = 1;
    for shift in (0..70).step_by(7) {
        let byte = *data.get(end).ok_or("Truncated float exponent")?;
        end += 1;
        if shift == 63 && byte > 1 {
            return Err("Float exponent exceeds i64 encoding");
        }
        zigzag |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            if shift != 0 && byte == 0 {
                return Err("Noncanonical float exponent");
            }
            break;
        }
    }
    let exponent = ((zigzag >> 1) as i64) ^ -((zigzag & 1) as i64);
    let significand = &data[end..];
    if significand.first().is_none_or(|&byte| byte == 0)
        || significand.last().is_none_or(|byte| byte & 0x80 == 0)
    {
        return Err("Missing or noncanonical float significand");
    }
    Ok(Payload::Finite {
        negative: tag == NEGATIVE,
        exponent,
        significand,
    })
}

impl Float {
    fn serialization_tag(&self) -> u8 {
        let value = self.as_raw();
        if value.is_zero() {
            ZERO
        } else if value.is_nan() {
            NAN
        } else if !value.is_finite() {
            if value.is_sign_negative() {
                NEG_INFINITY
            } else {
                POS_INFINITY
            }
        } else if value.is_sign_negative() {
            NEGATIVE
        } else {
            POSITIVE
        }
    }

    fn with_binary_parts<R>(&self, operation: impl FnOnce(Parts<'_>) -> R) -> R {
        #[cfg(feature = "float-mpfr")]
        {
            let value = self.as_raw();
            let significand = value.get_significand().unwrap();
            // SAFETY: this helper is called only for live, finite nonzero MPFR
            // values. Use exp_t directly to avoid Rug's narrowing to i32.
            let exponent = unsafe { gmp_mpfr_sys::mpfr::get_exp(value.as_raw()) } as i64;
            operation(Parts::new(significand.as_limbs(), exponent))
        }
        #[cfg(feature = "float-astro")]
        {
            let (words, exponent) = self.as_raw().binary_parts().unwrap();
            operation(Parts::new(words, exponent))
        }
    }

    /// Size of the canonical binary value payload, excluding precision and error.
    pub fn serialized_size(&self) -> usize {
        if matches!(self.serialization_tag(), POSITIVE | NEGATIVE) {
            self.with_binary_parts(|parts| parts.size())
        } else {
            1
        }
    }

    /// Append the canonical binary value payload without a temporary allocation.
    /// Precision and tracked error must be stored by the caller. Both signs of
    /// zero and NaN share one encoding, matching expression canonicalization.
    pub fn serialize_into(&self, out: &mut Vec<u8>) {
        let tag = self.serialization_tag();
        if matches!(tag, POSITIVE | NEGATIVE) {
            self.with_binary_parts(|parts| parts.write(tag, out));
        } else {
            out.push(tag);
        }
    }

    /// Serialize a value in canonical binary form, excluding its precision.
    /// This replaces the earlier hexadecimal-text payload.
    pub fn serialize(&self) -> Vec<u8> {
        let tag = self.serialization_tag();
        if matches!(tag, POSITIVE | NEGATIVE) {
            self.with_binary_parts(|parts| {
                let mut out = Vec::with_capacity(parts.size());
                parts.write(tag, &mut out);
                out
            })
        } else {
            vec![tag]
        }
    }

    /// Test a valid serialized payload for canonical zero without decoding it.
    pub fn serialized_is_zero(data: &[u8]) -> bool {
        data == [ZERO]
    }

    /// Test a valid serialized payload for a finite central value.
    pub fn serialized_is_finite(data: &[u8]) -> bool {
        matches!(data.first(), Some(&ZERO | &POSITIVE | &NEGATIVE))
    }

    /// Compare canonical binary values without allocating or decoding a float.
    /// Precision is enclosing metadata and does not affect the comparison.
    /// NaN and malformed payloads return `None`. This validates the wire shape,
    /// but does not check a backend's supported exponent or precision range.
    pub fn serialized_partial_cmp(left: &[u8], right: &[u8]) -> Option<Ordering> {
        let left = parse(left).ok()?;
        let right = parse(right).ok()?;
        let rank = |value: &Payload<'_>| match value {
            Payload::Special(NEG_INFINITY) => Some(0),
            Payload::Finite { negative: true, .. } => Some(1),
            Payload::Special(ZERO) => Some(2),
            Payload::Finite {
                negative: false, ..
            } => Some(3),
            Payload::Special(POS_INFINITY) => Some(4),
            _ => None,
        };
        let order = rank(&left)?.cmp(&rank(&right)?);
        if order != Ordering::Equal {
            return Some(order);
        }
        match (left, right) {
            (
                Payload::Finite {
                    negative,
                    exponent: left_exponent,
                    significand: left_significand,
                },
                Payload::Finite {
                    exponent: right_exponent,
                    significand: right_significand,
                    ..
                },
            ) => {
                let order = left_exponent.cmp(&right_exponent).then_with(|| {
                    // Equal exponents align the high bits. A longer equal
                    // prefix is larger because canonical low bytes are nonzero.
                    left_significand
                        .iter()
                        .rev()
                        .cmp(right_significand.iter().rev())
                });
                Some(if negative { order.reverse() } else { order })
            }
            _ => Some(Ordering::Equal),
        }
    }

    /// Decode a binary value exactly. Reject malformed, noncanonical or
    /// out-of-range payloads, unsupported precision and values whose significant
    /// bits cannot be preserved by the selected backend at the given precision.
    pub fn try_deserialize(data: &[u8], precision: u32) -> Result<Self, String> {
        Self::check_precision(precision)?;
        match parse(data)? {
            Payload::Special(ZERO) => Ok(Self::new(precision)),
            Payload::Special(tag) => Ok(Self::with_val(
                precision,
                match tag {
                    POS_INFINITY => f64::INFINITY,
                    NEG_INFINITY => f64::NEG_INFINITY,
                    NAN => f64::NAN,
                    _ => unreachable!(),
                },
            )),
            Payload::Finite {
                negative,
                exponent,
                significand,
            } => {
                #[cfg(feature = "float-mpfr")]
                {
                    use gmp_mpfr_sys::{gmp, mpfr};
                    let bits = significand
                        .len()
                        .checked_mul(8)
                        .ok_or("Float significand too long")?;
                    if bits - significand[0].trailing_zeros() as usize > precision as usize {
                        return Err("Float significand exceeds MPFR precision".into());
                    }
                    // SAFETY: these MPFR functions read the current thread's
                    // configured exponent bounds and have no pointer arguments.
                    let (min, max) = unsafe { (mpfr::get_emin(), mpfr::get_emax()) };
                    if exponent < min as i64 || exponent > max as i64 {
                        return Err("Float exponent is outside the MPFR range".into());
                    }
                    let scaled = exponent
                        .checked_sub(i64::try_from(bits).map_err(|_| "Float significand too long")?)
                        .ok_or("Float exponent underflow")?;
                    let scaled = scaled
                        .try_into()
                        .map_err(|_| "Float exponent exceeds MPFR exp_t")?;
                    let mut value = rug::Float::new(precision);
                    let rounded = if significand.len()
                        <= std::mem::size_of::<std::os::raw::c_ulong>()
                    {
                        let mut bytes = [0; std::mem::size_of::<std::os::raw::c_ulong>()];
                        bytes[..significand.len()].copy_from_slice(significand);
                        let integer = std::os::raw::c_ulong::from_le_bytes(bytes);
                        // SAFETY: value is initialized and exclusively borrowed.
                        // Import with the final scale to avoid range loss in an
                        // intermediate integer-valued float. No GMP integer is needed.
                        unsafe {
                            mpfr::set_ui_2exp(
                                value.as_raw_mut(),
                                integer,
                                scaled,
                                mpfr::rnd_t::RNDN,
                            )
                        }
                    } else if gmp::NAIL_BITS == 0
                        && significand.len() <= 4 * std::mem::size_of::<gmp::limb_t>()
                    {
                        const LIMB_BYTES: usize = std::mem::size_of::<gmp::limb_t>();
                        let mut limbs = [0; 4];
                        for (limb, chunk) in limbs.iter_mut().zip(significand.chunks(LIMB_BYTES)) {
                            let mut bytes = [0; LIMB_BYTES];
                            bytes[..chunk.len()].copy_from_slice(chunk);
                            *limb = gmp::limb_t::from_le_bytes(bytes);
                        }
                        let count = significand.len().div_ceil(LIMB_BYTES);
                        // SAFETY: the aligned stack array contains `count`
                        // initialized, nail-free limbs in least-significant
                        // order. Canonical payloads have a nonzero top byte,
                        // so the integer is normalized. GMP's read-only
                        // initializer does not own or free the array. MPFR
                        // reads it only during this call and copies the value
                        // into the separately allocated destination.
                        unsafe {
                            let integer =
                                gmp::MPZ_ROINIT_N(limbs.as_mut_ptr(), count as gmp::size_t);
                            mpfr::set_z_2exp(
                                value.as_raw_mut(),
                                &integer,
                                scaled,
                                mpfr::rnd_t::RNDN,
                            )
                        }
                    } else {
                        let integer =
                            rug::Integer::from_digits(significand, rug::integer::Order::Lsf);
                        // SAFETY: pointers reference initialized live GMP/MPFR
                        // objects; the destination is exclusively borrowed.
                        unsafe {
                            mpfr::set_z_2exp(
                                value.as_raw_mut(),
                                integer.as_raw(),
                                scaled,
                                mpfr::rnd_t::RNDN,
                            )
                        }
                    };
                    if rounded != 0 || !value.is_normal() {
                        return Err("Float value is not exactly representable in MPFR".into());
                    }
                    if negative {
                        value = -value;
                    }
                    Ok(Self::from_raw(value))
                }
                #[cfg(feature = "float-astro")]
                {
                    crate::domains::backend::float::MultiPrecisionFloat::from_binary_parts(
                        precision,
                        negative,
                        exponent,
                        significand,
                    )
                    .map(Self::from_raw)
                }
            }
        }
    }

    /// Decode a trusted binary payload. Use [`Self::try_deserialize`] for input
    /// that has not already been validated.
    pub fn deserialize(data: &[u8], precision: u32) -> Self {
        Self::try_deserialize(data, precision).expect("invalid binary float payload")
    }
}
