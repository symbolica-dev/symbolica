use bytes::{Buf, BufMut};

use crate::{
    coefficient::{
        Coefficient, CoefficientView, SerializedFloat, SerializedLargeRational, SerializedRational,
        SerializedRationalPolynomial,
    },
    domains::{
        backend::integer::{from_lsf_bytes, lsf_byte_size, write_lsf_bytes},
        finite_field::FiniteFieldElement,
        integer::{Integer, IntegerRing, Z},
        rational::Fraction,
        rational_polynomial::RationalPolynomial,
    },
    state::{FiniteFieldIndex, State, VariableListIndex},
};

const U8_NUM: u8 = 0b00000001;
const U16_NUM: u8 = 0b00000010;
const U32_NUM: u8 = 0b00000011;
const U64_NUM: u8 = 0b00000100;
const FIN_NUM: u8 = 0b00000101;
const ARB_NUM: u8 = 0b00000111;
const RAT_POLY: u8 = 0b00001000;
const FLOAT: u8 = 0b00001001;
const COMPLEX: u8 = 0b00001010;
const COMPACT_FLOAT: u8 = 0b00001011;
const INDETERMINATE: u8 = 0b00001100;
const COMPLEX_INFINITY: u8 = 0b00001101;
const DIRECTED_INFINITY: u8 = 0b00001111;
const U8_DEN: u8 = 0b00010000;
const U16_DEN: u8 = 0b00100000;
const U32_DEN: u8 = 0b00110000;
const U64_DEN: u8 = 0b01000000;
const ARB_DEN: u8 = 0b01110000;
const NUM_MASK: u8 = 0b00001111;
const DEN_MASK: u8 = 0b01110000;
const SIGN: u8 = 0b10000000;

/// Inspect an inline integer or a one-byte integer payload without coefficient dispatch.
#[inline]
pub(crate) fn small_integer_atom(data: &[u8]) -> Option<i64> {
    if data.len() == 1 {
        return Some(i64::from(data[0] >> 3) - 16);
    }
    if (data[1] | SIGN) != (U8_NUM | SIGN) {
        return None;
    }
    let n = i64::from(data[2]);
    Some(if data[1] & SIGN == 0 { n } else { -n })
}

/// Read only the length when traversing an atom; avoid the full fraction dispatch.
#[inline(always)]
pub(crate) fn read_packed_u64(data: &[u8]) -> (u64, &[u8]) {
    // Keep the common width out of the variable-width dispatch.
    if data[0] == U8_NUM {
        return (data[1] as u64, &data[2..]);
    }
    let mut rest = &data[1..];
    let value = match data[0] {
        U16_NUM => rest.get_u16_le() as u64,
        U32_NUM => rest.get_u32_le() as u64,
        U64_NUM => rest.get_u64_le(),
        _ => unreachable!("Invalid packed length"),
    };
    (value, rest)
}

#[inline(always)]
pub(crate) fn read_packed_denominator(data: &[u8]) -> (u64, &[u8]) {
    let tag = data[0];
    if tag == U8_NUM | U8_DEN {
        return (data[2] as u64, &data[3..]);
    }
    read_packed_denominator_wide(data)
}

// Keep LLVM from merging the common case back into the wide jump table.
#[cold]
#[inline(always)]
fn read_packed_denominator_wide(data: &[u8]) -> (u64, &[u8]) {
    let tag = data[0];
    // The count (or symbol ID) usually fits in one byte, even when the
    // payload length is large. Keep its offset constant in these cases.
    match tag {
        U8_NUM_U16_DEN => {
            return (
                u16::from_le_bytes(data[2..4].try_into().unwrap()) as u64,
                &data[4..],
            );
        }
        U8_NUM_U32_DEN => {
            return (
                u32::from_le_bytes(data[2..6].try_into().unwrap()) as u64,
                &data[6..],
            );
        }
        U8_NUM_U64_DEN => {
            return (
                u64::from_le_bytes(data[2..10].try_into().unwrap()),
                &data[10..],
            );
        }
        U8_NUM => return (1, &data[2..]),
        _ => {}
    }
    // Counts always occupy 1, 2, 4 or 8 bytes. No need to decode their value.
    let count_size = 1usize << ((tag & NUM_MASK) - 1);
    let mut rest = &data[1 + count_size..];
    let value = match tag & DEN_MASK {
        0 => 1,
        U8_DEN => rest.get_u8() as u64,
        U16_DEN => rest.get_u16_le() as u64,
        U32_DEN => rest.get_u32_le() as u64,
        U64_DEN => rest.get_u64_le(),
        _ => unreachable!("Invalid packed length"),
    };
    (value, rest)
}

#[inline(always)]
pub(crate) fn read_packed_numerator(data: &[u8]) -> u64 {
    let tag = data[0] & NUM_MASK;
    if tag == U8_NUM {
        return data[1] as u64;
    }
    let mut rest = &data[1..];
    match tag {
        U16_NUM => rest.get_u16_le() as u64,
        U32_NUM => rest.get_u32_le() as u64,
        U64_NUM => rest.get_u64_le(),
        _ => unreachable!("Invalid packed numerator"),
    }
}

#[inline(always)]
pub(crate) fn read_packed_pair(data: &[u8]) -> (u64, u64, &[u8]) {
    if data[0] == U8_NUM_U8_DEN {
        (data[1] as u64, data[2] as u64, &data[3..])
    } else {
        read_packed_pair_wide(data)
    }
}

/// Read an integer pair at an import boundary, before trusted packed readers run.
pub(crate) fn read_packed_pair_checked(data: &[u8]) -> Result<(u64, u64, &[u8]), String> {
    let (&tag, mut rest) = data
        .split_first()
        .ok_or_else(|| "Missing packed integer header".to_owned())?;
    let numerator_width = tag & NUM_MASK;
    let denominator_width = (tag & DEN_MASK) >> 4;
    if !(1..=4).contains(&numerator_width) || denominator_width > 4 {
        return Err("Invalid packed integer widths".to_owned());
    }
    let mut read = |width: u8| -> Result<u64, String> {
        if width == 0 {
            return Ok(1);
        }
        let size = 1usize << (width - 1);
        let bytes = rest
            .get(..size)
            .ok_or_else(|| "Truncated packed integer".to_owned())?;
        let mut value = [0; 8];
        value[..size].copy_from_slice(bytes);
        rest = &rest[size..];
        Ok(u64::from_le_bytes(value))
    };
    let numerator = read(numerator_width)?;
    let denominator = read(denominator_width)?;
    Ok((numerator, denominator, rest))
}

fn take_checked(data: &[u8], length: u64) -> Result<(&[u8], &[u8]), String> {
    let length = usize::try_from(length)
        .map_err(|_| "Packed payload length does not fit in usize".to_owned())?;
    if length > data.len() {
        return Err("Truncated packed payload".to_owned());
    }
    Ok(data.split_at(length))
}

/// Validate coefficient boundaries and binary floats without creating atom views.
/// Rational-polynomial payloads remain opaque here; their own domain decodes them.
pub(crate) fn validate_packed_coefficient(data: &[u8]) -> Result<&[u8], String> {
    fn rational(data: &[u8], require_large: bool) -> Result<&[u8], String> {
        let (&tag, rest) = data
            .split_first()
            .ok_or_else(|| "Missing rational coefficient".to_owned())?;
        if tag & NUM_MASK == ARB_NUM {
            let (numerator, denominator, rest) = read_packed_pair_checked(rest)?;
            let total = numerator
                .checked_add(denominator)
                .ok_or_else(|| "Rational payload length overflow".to_owned())?;
            Ok(take_checked(rest, total)?.1)
        } else if require_large {
            Err("Invalid infinity direction coefficient".to_owned())
        } else {
            Ok(read_packed_pair_checked(data)?.2)
        }
    }

    fn component(data: &[u8], allow_zero: bool) -> Result<(), String> {
        if data.len() < 13 && !(allow_zero && data.len() == 4) {
            return Err("Invalid floating coefficient component length".to_owned());
        }
        SerializedFloat(data).try_to_float().map(|_| ())
    }

    let (&tag, rest) = data
        .split_first()
        .ok_or_else(|| "Missing coefficient header".to_owned())?;
    match tag {
        COMPACT_FLOAT => {
            let header = rest;
            let (real_length, imaginary_length, rest) = read_packed_pair_checked(header)?;
            if header[0] & SIGN != 0
                || (header.len() - rest.len()) as u64
                    != (real_length, imaginary_length).get_packed_size()
            {
                return Err("Noncanonical floating coefficient length header".to_owned());
            }
            let (real, rest) = take_checked(rest, real_length)?;
            let (imaginary, rest) = take_checked(rest, imaginary_length)?;
            component(real, false)?;
            if !imaginary.is_empty() {
                component(imaginary, true)?;
                if (imaginary.len() == 4 && imaginary == &real[..4])
                    || (imaginary.len() > 4 && SerializedFloat(imaginary).is_exact_zero())
                {
                    return Err("Noncanonical imaginary zero coefficient".to_owned());
                }
            }
            Ok(rest)
        }
        FLOAT => Err("Noncanonical fixed-length floating coefficient".to_owned()),
        COMPLEX | DIRECTED_INFINITY => {
            let rest = rational(rest, tag == DIRECTED_INFINITY)?;
            rational(rest, tag == DIRECTED_INFINITY)
        }
        INDETERMINATE | COMPLEX_INFINITY => Ok(rest),
        RAT_POLY => {
            let (length, rest) = take_checked(rest, 4)?;
            Ok(take_checked(rest, u32::from_le_bytes(length.try_into().unwrap()) as u64)?.1)
        }
        _ if tag & NUM_MASK == FIN_NUM => Ok(read_packed_pair_checked(rest)?.2),
        _ => rational(data, false),
    }
}

#[cold]
#[inline(always)]
fn read_packed_pair_wide(data: &[u8]) -> (u64, u64, &[u8]) {
    data.get_frac_u64()
}

/// Skip a function's length header and payload.
///
/// # Safety
/// `data` must start with a valid packed function length and contain its payload.
#[inline(always)]
pub(crate) unsafe fn skip_packed_function(data: &[u8]) -> &[u8] {
    let (size, rest) = read_packed_u64(data);
    unsafe { rest.get_unchecked(size as usize..) }
}

/// Skip a product/sum's packed count/length header and payload.
///
/// # Safety
/// `data` must contain a valid packed count/length pair and its entire payload.
#[inline(always)]
pub(crate) unsafe fn skip_packed_list(data: &[u8]) -> &[u8] {
    let (size, rest) = read_packed_denominator(data);
    unsafe { rest.get_unchecked(size as usize..) }
}

const U8_NUM_U8_DEN: u8 = U8_NUM | U8_DEN;
const U16_NUM_U8_DEN: u8 = U16_NUM | U8_DEN;
const U32_NUM_U8_DEN: u8 = U32_NUM | U8_DEN;
const U64_NUM_U8_DEN: u8 = U64_NUM | U8_DEN;
const U8_NUM_U16_DEN: u8 = U8_NUM | U16_DEN;
const U16_NUM_U16_DEN: u8 = U16_NUM | U16_DEN;
const U32_NUM_U16_DEN: u8 = U32_NUM | U16_DEN;
const U64_NUM_U16_DEN: u8 = U64_NUM | U16_DEN;
const U8_NUM_U32_DEN: u8 = U8_NUM | U32_DEN;
const U16_NUM_U32_DEN: u8 = U16_NUM | U32_DEN;
const U32_NUM_U32_DEN: u8 = U32_NUM | U32_DEN;
const U64_NUM_U32_DEN: u8 = U64_NUM | U32_DEN;
const U8_NUM_U64_DEN: u8 = U8_NUM | U64_DEN;
const U16_NUM_U64_DEN: u8 = U16_NUM | U64_DEN;
const U32_NUM_U64_DEN: u8 = U32_NUM | U64_DEN;
const U64_NUM_U64_DEN: u8 = U64_NUM | U64_DEN;

const SKIP_TABLE: [u8; (NUM_MASK | DEN_MASK) as usize + 1] = {
    let mut dest = [0u8; (NUM_MASK | DEN_MASK) as usize + 1];
    dest[U8_NUM as usize] = 1;
    dest[U16_NUM as usize] = 2;
    dest[U8_NUM_U8_DEN as usize] = 2;
    dest[U16_NUM_U8_DEN as usize] = 3;
    dest[U8_NUM_U16_DEN as usize] = 3;
    dest[U32_NUM as usize] = 4;
    dest[U16_NUM_U16_DEN as usize] = 4;
    dest[U32_NUM_U8_DEN as usize] = 5;
    dest[U8_NUM_U32_DEN as usize] = 5;
    dest[U32_NUM_U16_DEN as usize] = 6;
    dest[U16_NUM_U32_DEN as usize] = 6;
    dest[U64_NUM as usize] = 8;
    dest[U32_NUM_U32_DEN as usize] = 8;
    dest[U64_NUM_U8_DEN as usize] = 9;
    dest[U8_NUM_U64_DEN as usize] = 9;
    dest[U64_NUM_U16_DEN as usize] = 10;
    dest[U16_NUM_U64_DEN as usize] = 10;
    dest[U64_NUM_U32_DEN as usize] = 12;
    dest[U32_NUM_U64_DEN as usize] = 12;
    dest[U64_NUM_U64_DEN as usize] = 16;
    dest
};

/// Skip a variable's packed symbol ID and extra attributes.
///
/// # Safety
/// `data` must contain the complete unsigned packed pair from a valid variable.
#[inline(always)]
pub(crate) unsafe fn skip_packed_variable(data: &[u8]) -> &[u8] {
    let tag = unsafe { *data.get_unchecked(0) };
    let size = unsafe { *SKIP_TABLE.get_unchecked(tag as usize) } as usize + 1;
    unsafe { data.get_unchecked(size..) }
}

#[inline(always)]
const fn get_size_of_natural(num_type: u8) -> u8 {
    match num_type {
        0 => 0,
        U8_NUM => 1,
        U16_NUM => 2,
        U32_NUM => 4,
        U64_NUM => 8,
        _ => unreachable!(),
    }
}

impl SerializedRationalPolynomial<'_> {
    /// Decode the borrowed coefficient bytes into an owned rational polynomial.
    /// The encoding must be valid and refer to the current symbol state.
    pub fn deserialize(self) -> RationalPolynomial<IntegerRing, u16> {
        let (index, _, _) = self.0.get_frac_u64();
        let vars = State::get_variable_list(VariableListIndex(index as usize));
        self.deserialize_with_variables(vars)
    }

    pub(crate) fn deserialize_with_variables(
        self,
        vars: std::sync::Arc<Vec<crate::poly::PolyVariable>>,
    ) -> RationalPolynomial<IntegerRing, u16> {
        let mut source = self.0;

        let num_nterms;
        let den_nterms;
        (_, num_nterms, source) = source.get_frac_u64();
        (den_nterms, _, source) = source.get_frac_u64();

        let nvars = vars.len();

        let mut poly = RationalPolynomial::new(&Z, vars);

        poly.numerator.exponents = vec![0u16; num_nterms as usize * nvars];
        for i in 0..poly.numerator.exponents.len() {
            poly.numerator.exponents[i] = source.get_u16_le();
        }

        poly.denominator.exponents = vec![0u16; den_nterms as usize * nvars];
        for i in 0..poly.denominator.exponents.len() {
            poly.denominator.exponents[i] = source.get_u16_le();
        }

        fn parse_num(source: &mut &[u8]) -> Integer {
            match source.get_u8() {
                1 => Integer::Single(source.get_i64_le()),
                2 => Integer::Double(source.get_i128_le().into()),
                x @ 4 | x @ 5 => {
                    let (num_digits, _, new_source) = source.get_frac_u64();
                    *source = new_source;
                    let i = from_lsf_bytes(&source[..num_digits as usize]);
                    source.advance(num_digits as usize);
                    if x == 5 {
                        Integer::from(-i)
                    } else {
                        Integer::from(i)
                    }
                }
                _ => unreachable!(),
            }
        }

        for _ in 0..num_nterms {
            poly.numerator.coefficients.push(parse_num(&mut source));
        }

        poly.denominator.coefficients.clear();
        for _ in 0..den_nterms {
            poly.denominator.coefficients.push(parse_num(&mut source));
        }

        poly
    }
}

/// A generalized rational number. The first byte indicates the sign, size and type of the numerator and denominator.
/// The highest four bits give the byte size of the numerator and the lower bits of the denominator.
pub trait PackedRationalNumberWriter {
    /// Write a single number.
    fn write_packed(&self, dest: &mut Vec<u8>);
    /// Write a fraction to a fixed-size buffer.
    fn write_packed_fixed(&self, dest: &mut [u8]);
    /// Get the number of bytes of the packed representation.
    fn get_packed_size(&self) -> u64;
}

impl PackedRationalNumberWriter for Coefficient {
    fn write_packed(&self, dest: &mut Vec<u8>) {
        #[inline(always)]
        fn write_rational(r: &Fraction<IntegerRing>, dest: &mut Vec<u8>) {
            match (r.numerator_ref(), r.denominator_ref()) {
                (Integer::Single(num), Integer::Single(den)) => {
                    (*num, *den as u64).write_packed(dest)
                }
                _ => {
                    write_multi_rational(r, dest);
                }
            }
        }

        #[inline(always)]
        fn write_multi_rational(r: &Fraction<IntegerRing>, dest: &mut Vec<u8>) {
            let num = r.numerator_ref().clone().to_multi_prec();
            let den = r.denominator_ref().clone().to_multi_prec();
            let num_digits = lsf_byte_size(&num);
            let den_digits = lsf_byte_size(&den);
            dest.put_u8(ARB_NUM | ARB_DEN);

            if r.numerator_ref().is_negative() {
                (-(num_digits as i64), den_digits as u64).write_packed(dest);
            } else {
                (num_digits as i64, den_digits as u64).write_packed(dest);
            }

            write_lsf_bytes(&num, dest);
            write_lsf_bytes(&den, dest);
        }

        match self {
            Coefficient::Complex(c) => {
                if c.im.is_zero() {
                    write_rational(&c.re, dest);
                } else {
                    dest.put_u8(COMPLEX);
                    write_rational(&c.re, dest);
                    write_rational(&c.im, dest);
                }
            }
            Coefficient::Indeterminate => {
                dest.put_u8(INDETERMINATE);
            }
            Coefficient::Infinity(None) => {
                dest.put_u8(COMPLEX_INFINITY);
            }
            Coefficient::Infinity(Some(dir)) => {
                dest.put_u8(DIRECTED_INFINITY);
                write_multi_rational(&dir.re, dest);
                write_multi_rational(&dir.im, dest);
            }
            Coefficient::Float(f) => {
                // Zero bytes inherit the real precision; four bytes preserve a
                // distinct imaginary precision. Both mean exactly zero error.
                let im_is_exact_zero = f.im.is_exact_zero();
                let im_len = if im_is_exact_zero {
                    u64::from(f.re.get_num().prec() != f.im.get_num().prec()) * 4
                } else {
                    f.im.get_num().serialized_size() as u64 + 12
                };
                let lengths = (f.re.get_num().serialized_size() as u64 + 12, im_len);
                dest.reserve((1 + lengths.get_packed_size() + lengths.0 + lengths.1) as usize);
                dest.put_u8(COMPACT_FLOAT);
                lengths.write_packed(dest);
                dest.put_u32_le(f.re.get_num().prec());
                dest.extend_from_slice(&f.re.get_error_magnitude().to_le_bytes());
                f.re.get_num().serialize_into(dest);
                if im_len != 0 {
                    dest.put_u32_le(f.im.get_num().prec());
                }
                if !im_is_exact_zero {
                    dest.extend_from_slice(&f.im.get_error_magnitude().to_le_bytes());
                    f.im.get_num().serialize_into(dest);
                }
            }
            Coefficient::FiniteField(num, f) => {
                dest.put_u8(FIN_NUM);
                (*num.inner(), f.0 as u64).write_packed(dest); // this adds an extra tag
            }
            Coefficient::RationalPolynomial(p) => {
                dest.put_u8(RAT_POLY);
                dest.put_u32(0);
                let pos = dest.len();

                let index = State::get_or_insert_variable_list(p.get_variables().clone());

                (index.0 as u64, p.numerator.nterms() as u64).write_packed(dest);
                (p.denominator.nterms() as u64, 1).write_packed(dest);

                for i in p.numerator.exponents.iter().chain(&p.denominator.exponents) {
                    dest.put_u16_le(*i);
                }

                for i in p
                    .numerator
                    .coefficients
                    .iter()
                    .chain(&p.denominator.coefficients)
                {
                    match i {
                        Integer::Single(n) => {
                            dest.put_u8(1);
                            dest.put_i64_le(*n);
                        }
                        Integer::Double(d) => {
                            dest.put_u8(2);
                            dest.put_i128_le(d.get());
                        }
                        Integer::Large(l) => {
                            if l.is_negative() {
                                dest.put_u8(5);
                            } else {
                                dest.put_u8(4);
                            }
                            let num_digits = lsf_byte_size(l);
                            (num_digits as u64, 1).write_packed(dest);
                            write_lsf_bytes(l, dest);
                        }
                    }
                }

                let len = dest.len() - pos;
                if len > u32::MAX as usize {
                    panic!("Rational polynomial too large to serialize");
                }

                dest[pos - 4..pos].copy_from_slice(&(len as u32).to_le_bytes());
            }
        }
    }

    fn write_packed_fixed(&self, mut dest: &mut [u8]) {
        match self {
            Coefficient::Complex(c) => {
                let real = c.im.is_zero();
                if !real {
                    dest.put_u8(COMPLEX);
                }
                match (c.re.numerator_ref(), c.re.denominator_ref()) {
                    (Integer::Single(num), Integer::Single(den)) => {
                        (*num, *den as u64).write_packed_fixed(dest);
                    }
                    _ => todo!("Writing large packed rational not implemented"),
                }

                if !real {
                    match (c.im.numerator_ref(), c.im.denominator_ref()) {
                        (Integer::Single(num), Integer::Single(den)) => {
                            (*num, *den as u64).write_packed_fixed(dest);
                        }
                        _ => todo!("Writing large packed rational not implemented"),
                    }
                }
            }
            Coefficient::Indeterminate => {
                dest.put_u8(INDETERMINATE);
            }
            Coefficient::Infinity(None) => {
                dest.put_u8(COMPLEX_INFINITY);
            }
            Coefficient::Infinity(Some(_)) => {
                dest.put_u8(DIRECTED_INFINITY);
                todo!("Writing large packed rational not implemented");
            }
            Coefficient::Float(_) => todo!("Writing packed float not implemented"),
            Coefficient::RationalPolynomial(_) => {
                todo!("Writing packed rational polynomial not implemented")
            }
            Coefficient::FiniteField(num, f) => {
                dest.put_u8(FIN_NUM);
                (*num.inner(), f.0 as u64).write_packed_fixed(dest);
            }
        }
    }

    fn get_packed_size(&self) -> u64 {
        #[inline(always)]
        fn packed_size_rat(r: &Fraction<IntegerRing>) -> u64 {
            match (r.numerator_ref(), r.denominator_ref()) {
                (Integer::Single(num), Integer::Single(den)) => {
                    (*num, *den as u64).get_packed_size()
                }
                _ => packed_size_multi_rat(r),
            }
        }

        #[inline(always)]
        fn packed_size_multi_rat(r: &Fraction<IntegerRing>) -> u64 {
            let n = lsf_byte_size(&r.numerator_ref().clone().to_multi_prec()) as u64;
            let d = lsf_byte_size(&r.denominator_ref().clone().to_multi_prec()) as u64;
            1 + (n, d).get_packed_size() + n + d
        }

        match self {
            Coefficient::Complex(c) => {
                if c.im.is_zero() {
                    packed_size_rat(&c.re)
                } else {
                    1 + packed_size_rat(&c.re) + packed_size_rat(&c.im)
                }
            }
            Coefficient::Indeterminate => 1,
            Coefficient::Infinity(None) => 1,
            Coefficient::Infinity(Some(c)) => {
                1 + packed_size_multi_rat(&c.re) + packed_size_multi_rat(&c.im)
            }
            Coefficient::Float(f) => {
                let im_len = if f.im.is_exact_zero() {
                    u64::from(f.re.get_num().prec() != f.im.get_num().prec()) * 4
                } else {
                    f.im.get_num().serialized_size() as u64 + 12
                };
                let lengths = (f.re.get_num().serialized_size() as u64 + 12, im_len);
                1 + lengths.get_packed_size() + lengths.0 + lengths.1
            }
            Coefficient::FiniteField(m, i) => 2 + (*m.inner(), i.0 as u64).get_packed_size(),
            Coefficient::RationalPolynomial(_) => {
                unimplemented!("Cannot get the packed size of a rational polynomial")
            }
        }
    }
}

/// A reader for generalized rational numbers. See also [`PackedRationalNumberWriter`].
pub trait PackedRationalNumberReader {
    fn get_coeff_view(&self) -> (CoefficientView<'_>, &[u8]);
    fn get_frac_u64(&self) -> (u64, u64, &[u8]);
    fn get_frac_i64(&self) -> (i64, u64, &[u8]);
    fn skip_rational(&self) -> &[u8];
    fn is_small_int(&self) -> bool;
    fn is_zero_rat(&self) -> bool;
    fn is_one_rat(&self) -> bool;

    fn is_rational_polynomial(&self) -> bool;
}

impl PackedRationalNumberReader for [u8] {
    #[inline(always)]
    fn is_rational_polynomial(&self) -> bool {
        self[1] == RAT_POLY
    }

    #[inline(always)]
    fn get_coeff_view(&self) -> (CoefficientView<'_>, &[u8]) {
        let mut source = self;
        let disc = source.get_u8();
        if disc == RAT_POLY {
            let len = source.get_u32_le() as usize;
            let start = source;
            source.advance(len);
            (
                CoefficientView::RationalPolynomial(SerializedRationalPolynomial(&start[..len])),
                source,
            )
        } else if disc == COMPACT_FLOAT {
            let (len, len_i, start) = read_packed_pair(source);
            let (re, rest) = start.split_at(len as usize);
            let (im, rest) = rest.split_at(len_i as usize);
            // An omitted imaginary zero inherits its precision from the real
            // component. Keep a normal borrowed view without an allocation.
            let im = if im.is_empty() { &re[..4] } else { im };
            (
                CoefficientView::Float(SerializedFloat(re), SerializedFloat(im)),
                rest,
            )
        } else if disc == FLOAT {
            // Alternate fixed-width length encoding with the current magnitude
            // payload. Older exported formats are rejected by their version.
            let len = source.get_u64_le() as usize;
            let start = source;
            source.advance(len);

            let len_i = source.get_u64_le() as usize;
            let start2 = source;
            source.advance(len_i);

            (
                CoefficientView::Float(
                    SerializedFloat(&start[..len]),
                    SerializedFloat(&start2[..len_i]),
                ),
                source,
            )
        } else if disc == COMPLEX {
            let (num, r) = source.get_coeff_view();
            let (den, r) = r.get_coeff_view();

            if let CoefficientView::Natural(n, d, _, _) = num {
                if let CoefficientView::Natural(ni, di, _, _) = den {
                    (CoefficientView::Natural(n, d, ni, di), r)
                } else if let CoefficientView::Large(ri, _) = den {
                    (
                        CoefficientView::Large(SerializedRational::Natural(n, d), ri),
                        r,
                    )
                } else {
                    unreachable!()
                }
            } else if let CoefficientView::Large(rn, _) = num {
                if let CoefficientView::Natural(ni, di, _, _) = den {
                    (
                        CoefficientView::Large(rn, SerializedRational::Natural(ni, di)),
                        r,
                    )
                } else if let CoefficientView::Large(ri, _) = den {
                    (CoefficientView::Large(rn, ri), r)
                } else {
                    unreachable!()
                }
            } else {
                unreachable!()
            }
        } else if disc == INDETERMINATE {
            (CoefficientView::Indeterminate, source)
        } else if disc == COMPLEX_INFINITY {
            (CoefficientView::Infinity(None), source)
        } else if disc == DIRECTED_INFINITY {
            let (num, r) = source.get_coeff_view();
            let (den, r) = r.get_coeff_view();

            if let CoefficientView::Large(rn, _) = num {
                if let CoefficientView::Large(ri, _) = den {
                    (CoefficientView::Infinity(Some((rn, ri))), r)
                } else {
                    unreachable!()
                }
            } else {
                unreachable!()
            }
        } else if (disc & NUM_MASK) == ARB_NUM {
            let is_negative = unsafe { source.get_unchecked(0) & SIGN != 0 };
            let (num, den);
            (num, den, source) = source.get_frac_u64();
            let num_len = num as usize;
            let den_len = den as usize;
            let num_limbs = &source[..num_len];
            let den_limbs = &source[num_len..num_len + den_len];

            (
                CoefficientView::Large(
                    SerializedRational::Large(SerializedLargeRational {
                        is_negative,
                        num_digits: num_limbs,
                        den_digits: den_limbs,
                    }),
                    SerializedRational::Large(SerializedLargeRational {
                        is_negative: false,
                        num_digits: &[],
                        den_digits: &[],
                    }),
                ),
                &source[num_len + den_len..],
            )
        } else if (disc & NUM_MASK) == FIN_NUM {
            let (num, fi);
            (num, fi, source) = source.get_frac_u64();
            (
                CoefficientView::FiniteField(
                    FiniteFieldElement::from_inner(num),
                    FiniteFieldIndex(fi as usize),
                ),
                source,
            )
        } else {
            let (num, den, source) = self.get_frac_i64();
            (CoefficientView::Natural(num, den as i64, 0, 1), source)
        }
    }

    #[inline(always)]
    fn get_frac_u64(&self) -> (u64, u64, &[u8]) {
        let mut source = self;
        let disc = source.get_u8();
        match disc & (NUM_MASK | DEN_MASK) {
            U8_NUM => {
                let n = source.get_u8();
                (n as u64, 1, source)
            }
            U16_NUM => {
                let n = source.get_u16_le();
                (n as u64, 1, source)
            }
            U32_NUM => {
                let n = source.get_u32_le();
                (n as u64, 1, source)
            }
            U64_NUM => {
                let n = source.get_u64_le();
                (n, 1, source)
            }
            U8_NUM_U8_DEN => {
                let n = source.get_u8();
                let d = source.get_u8();
                (n as u64, d as u64, source)
            }
            U16_NUM_U8_DEN => {
                let n = source.get_u16_le();
                let d = source.get_u8();
                (n as u64, d as u64, source)
            }
            U32_NUM_U8_DEN => {
                let n = source.get_u32_le();
                let d = source.get_u8();
                (n as u64, d as u64, source)
            }
            U64_NUM_U8_DEN => {
                let n = source.get_u64_le();
                let d = source.get_u8();
                (n, d as u64, source)
            }
            U8_NUM_U16_DEN => {
                let n = source.get_u8();
                let d = source.get_u16_le();
                (n as u64, d as u64, source)
            }
            U16_NUM_U16_DEN => {
                let n = source.get_u16_le();
                let d = source.get_u16_le();
                (n as u64, d as u64, source)
            }
            U32_NUM_U16_DEN => {
                let n = source.get_u32_le();
                let d = source.get_u16_le();
                (n as u64, d as u64, source)
            }
            U64_NUM_U16_DEN => {
                let n = source.get_u64_le();
                let d = source.get_u16_le();
                (n, d as u64, source)
            }
            U8_NUM_U32_DEN => {
                let n = source.get_u8();
                let d = source.get_u32_le();
                (n as u64, d as u64, source)
            }
            U16_NUM_U32_DEN => {
                let n = source.get_u16_le();
                let d = source.get_u32_le();
                (n as u64, d as u64, source)
            }
            U32_NUM_U32_DEN => {
                let n = source.get_u32_le();
                let d = source.get_u32_le();
                (n as u64, d as u64, source)
            }
            U64_NUM_U32_DEN => {
                let n = source.get_u64_le();
                let d = source.get_u32_le();
                (n, d as u64, source)
            }
            U8_NUM_U64_DEN => {
                let n = source.get_u8();
                let d = source.get_u64_le();
                (n as u64, d, source)
            }
            U16_NUM_U64_DEN => {
                let n = source.get_u16_le();
                let d = source.get_u64_le();
                (n as u64, d, source)
            }
            U32_NUM_U64_DEN => {
                let n = source.get_u32_le();
                let d = source.get_u64_le();
                (n as u64, d, source)
            }
            U64_NUM_U64_DEN => {
                let n = source.get_u64_le();
                let d = source.get_u64_le();
                (n, d, source)
            }
            x => {
                unreachable!("Unsupported numerator/denominator type {}", x)
            }
        }
    }

    #[inline(always)]
    fn get_frac_i64(&self) -> (i64, u64, &[u8]) {
        let mut source = self;
        let disc = source.get_u8();
        let num;
        (num, source) = match disc & NUM_MASK {
            U8_NUM => {
                let v = source.get_u8();
                (v as u64, source)
            }
            U16_NUM => {
                let v = source.get_u16_le();
                (v as u64, source)
            }
            U32_NUM => {
                let v = source.get_u32_le();
                (v as u64, source)
            }
            U64_NUM => {
                let v = source.get_u64_le();
                (v, source)
            }
            x => {
                unreachable!("Unsupported numerator type {}", x)
            }
        };

        let den;
        (den, source) = match disc & DEN_MASK {
            0 => (1u64, source),
            U8_DEN => {
                let v = source.get_u8();
                (v as u64, source)
            }
            U16_DEN => {
                let v = source.get_u16_le();
                (v as u64, source)
            }
            U32_DEN => {
                let v = source.get_u32_le();
                (v as u64, source)
            }
            U64_DEN => {
                let v = source.get_u64_le();
                (v, source)
            }
            x => {
                unreachable!("Unsupported denominator type {}", x)
            }
        };

        let num = if disc & SIGN != 0 {
            if num == (1u64 << 63) {
                i64::MIN
            } else {
                -(num as i64)
            }
        } else {
            num as i64
        };

        (num, den, source)
    }

    #[inline(always)]
    fn skip_rational(&self) -> &[u8] {
        let mut dest = self;
        let disc = dest.get_u8() & (NUM_MASK | DEN_MASK);

        let d = unsafe { *SKIP_TABLE.get_unchecked(disc as usize) };
        if d != 0 {
            return unsafe { dest.get_unchecked(d as usize..) };
        }

        let v_num = disc & NUM_MASK;
        if v_num == COMPLEX {
            dest = dest.skip_rational();
            dest = dest.skip_rational();
        } else if v_num == ARB_NUM {
            let (num_size, den_size);
            (num_size, den_size, dest) = dest.get_frac_u64();
            let num_size = num_size as usize;
            let den_size = den_size as usize;
            dest.advance(num_size + den_size);
        } else if v_num == RAT_POLY {
            let size = dest.get_u32_le() as usize;
            dest.advance(size);
        } else if v_num == FIN_NUM {
            let var_size = dest.get_u8();
            let size = get_size_of_natural(var_size & NUM_MASK)
                + get_size_of_natural((var_size & DEN_MASK) >> 4);
            dest.advance(size as usize);
        } else if v_num == COMPACT_FLOAT {
            let (real_size, imaginary_size, rest) = read_packed_pair(dest);
            dest = &rest[(real_size + imaginary_size) as usize..];
        } else if v_num == FLOAT {
            let size = dest.get_u64_le() as usize;
            dest.advance(size);
            let size = dest.get_u64_le() as usize;
            dest.advance(size);
        } else if v_num == INDETERMINATE || v_num == COMPLEX_INFINITY {
            // nothing to do
        } else if v_num == DIRECTED_INFINITY {
            dest = dest.skip_rational();
            dest = dest.skip_rational();
        } else {
            unreachable!("Unsupported numerator/denominator type {}", disc)
        }

        dest
    }

    #[inline(always)]
    fn is_small_int(&self) -> bool {
        (self[1] | SIGN) == (U8_NUM | SIGN)
    }

    #[inline(always)]
    fn is_zero_rat(&self) -> bool {
        // TODO: make a zero have no number at all (i.e., self[1] = 0)
        self[1] == U8_NUM && self[2] == 0
    }

    #[inline(always)]
    fn is_one_rat(&self) -> bool {
        self[1] == U8_NUM && self[2] == 1
    }
}

impl PackedRationalNumberWriter for (i64, u64) {
    #[inline(always)]
    fn write_packed(&self, dest: &mut Vec<u8>) {
        let p = dest.len();

        (self.0.unsigned_abs(), self.1).write_packed(dest);

        if self.0 < 0 {
            dest[p] |= SIGN;
        }
    }

    #[inline(always)]
    fn write_packed_fixed(&self, dest: &mut [u8]) {
        (self.0.unsigned_abs(), self.1).write_packed_fixed(dest);

        if self.0 < 0 {
            dest[0] |= SIGN;
        }
    }

    fn get_packed_size(&self) -> u64 {
        (self.0.unsigned_abs(), self.1).get_packed_size()
    }
}

impl PackedRationalNumberWriter for (u64, u64) {
    #[inline(always)]
    fn write_packed(&self, dest: &mut Vec<u8>) {
        let p = dest.len();

        if self.0 <= u8::MAX as u64 {
            dest.put_u8(U8_NUM);
            dest.put_u8(self.0 as u8);
        } else if self.0 <= u16::MAX as u64 {
            dest.put_u8(U16_NUM);
            dest.put_u16_le(self.0 as u16);
        } else if self.0 <= u32::MAX as u64 {
            dest.put_u8(U32_NUM);
            dest.put_u32_le(self.0 as u32);
        } else {
            dest.put_u8(U64_NUM);
            dest.put_u64_le(self.0);
        }

        if self.1 == 1 {
        } else if self.1 <= u8::MAX as u64 {
            dest[p] |= U8_DEN;
            dest.put_u8(self.1 as u8);
        } else if self.1 <= u16::MAX as u64 {
            dest[p] |= U16_DEN;
            dest.put_u16_le(self.1 as u16);
        } else if self.1 <= u32::MAX as u64 {
            dest[p] |= U32_DEN;
            dest.put_u32_le(self.1 as u32);
        } else {
            dest[p] |= U64_DEN;
            dest.put_u64_le(self.1);
        }
    }

    #[inline(always)]
    fn write_packed_fixed(&self, dest: &mut [u8]) {
        let (tag, mut dest) = dest.split_first_mut().unwrap();

        if self.0 <= u8::MAX as u64 {
            *tag = U8_NUM;
            dest.put_u8(self.0 as u8);
        } else if self.0 <= u16::MAX as u64 {
            *tag = U16_NUM;
            dest.put_u16_le(self.0 as u16);
        } else if self.0 <= u32::MAX as u64 {
            *tag = U32_NUM;
            dest.put_u32_le(self.0 as u32);
        } else {
            *tag = U64_NUM;
            dest.put_u64_le(self.0);
        }

        if self.1 == 1 {
        } else if self.1 <= u8::MAX as u64 {
            *tag |= U8_DEN;
            dest.put_u8(self.1 as u8);
        } else if self.1 <= u16::MAX as u64 {
            *tag |= U16_DEN;
            dest.put_u16_le(self.1 as u16);
        } else if self.1 <= u32::MAX as u64 {
            *tag |= U32_DEN;
            dest.put_u32_le(self.1 as u32);
        } else {
            *tag |= U64_DEN;
            dest.put_u64_le(self.1);
        }
    }

    #[inline]
    fn get_packed_size(&self) -> u64 {
        let mut size = 1;
        size += if self.0 <= u8::MAX as u64 {
            get_size_of_natural(U8_NUM)
        } else if self.0 <= u16::MAX as u64 {
            get_size_of_natural(U16_NUM)
        } else if self.0 <= u32::MAX as u64 {
            get_size_of_natural(U32_NUM)
        } else {
            get_size_of_natural(U64_NUM)
        };

        size += if self.1 == 1 {
            0
        } else if self.1 <= u8::MAX as u64 {
            get_size_of_natural(U8_NUM)
        } else if self.1 <= u16::MAX as u64 {
            get_size_of_natural(U16_NUM)
        } else if self.1 <= u32::MAX as u64 {
            get_size_of_natural(U32_NUM)
        } else {
            get_size_of_natural(U64_NUM)
        };
        size as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domains::float::{Complex, ErrorPropagatingFloat, Float, Magnitude, SingleFloat};

    fn float_coefficient(precision: u32, complex: bool) -> Coefficient {
        Coefficient::Float(Complex::new(
            ErrorPropagatingFloat::new_with_absolute_error(
                Float::with_val(precision, 1) / Float::with_val(precision, 3),
                1e-12,
            ),
            ErrorPropagatingFloat::new_with_absolute_error(
                if complex {
                    Float::with_val(precision + 17, -2) / Float::with_val(precision + 17, 7)
                } else {
                    Float::with_val(precision + 17, 0)
                },
                // An uncertain zero must retain its error even when its value
                // encoding is shared with exact zero.
                2e-14,
            ),
        ))
    }

    fn write_fixed_length_float(coefficient: &Coefficient, dest: &mut Vec<u8>) {
        let Coefficient::Float(f) = coefficient else {
            unreachable!()
        };
        dest.put_u8(FLOAT);
        for value in [&f.re, &f.im] {
            let data = value.get_num().serialize();
            dest.put_u64_le(data.len() as u64 + 12);
            dest.put_u32_le(value.get_num().prec());
            dest.extend_from_slice(&value.get_error_magnitude().to_le_bytes());
            dest.extend_from_slice(&data);
        }
    }

    fn assert_float_roundtrip(coefficient: &Coefficient, data: &[u8]) {
        if data[0] == COMPACT_FLOAT {
            assert_eq!(
                validate_packed_coefficient(data).unwrap(),
                b"trailing bytes"
            );
        } else {
            // Keep the trusted reader usable, but only compact canonical layouts
            // are accepted at import boundaries under the current format.
            assert!(validate_packed_coefficient(data).is_err());
        }
        let (view, remaining) = data.get_coeff_view();
        assert_eq!(remaining, b"trailing bytes");
        assert_eq!(data.skip_rational(), remaining);
        let Coefficient::Float(expected) = coefficient else {
            unreachable!()
        };
        let Coefficient::Float(actual) = view.to_owned() else {
            panic!("Expected a float coefficient")
        };
        for (expected, actual) in [(&expected.re, &actual.re), (&expected.im, &actual.im)] {
            assert_eq!(expected.get_num(), actual.get_num());
            assert_eq!(expected.get_num().prec(), actual.get_num().prec());
            assert_eq!(
                expected.get_error_magnitude().to_le_bytes(),
                actual.get_error_magnitude().to_le_bytes()
            );
        }
    }

    #[test]
    fn small_integer_atoms_cover_inline_and_escaped_payloads() {
        use crate::atom::{Atom, AtomView};

        for value in -255..=255 {
            let atom = Atom::num(value);
            let AtomView::Num(number) = atom.as_view() else {
                unreachable!()
            };
            assert_eq!(small_integer_atom(atom.as_view().get_data()), Some(value));
            assert_eq!(number.small_integer(), Some(value));
            assert_eq!(number.small_integer_is_zero(), Some(value == 0));
        }
        for source in ["-256", "256", "1/2", "1+2*𝑖", "1.25"] {
            let atom = crate::parse!(source);
            let AtomView::Num(number) = atom.as_view() else {
                unreachable!()
            };
            assert_eq!(small_integer_atom(atom.as_view().get_data()), None);
            assert_eq!(number.small_integer(), None);
            assert_eq!(number.small_integer_is_zero(), None);
        }
    }

    #[test]
    fn checked_packed_coefficients_accept_existing_variants() {
        use crate::domains::{
            finite_field::{FiniteFieldCore, Zp64},
            rational::Rational,
        };

        let field = Zp64::new(17);
        let field_element = field.to_element(5);
        let complex = Complex::new(Rational::from((1, 3)), Rational::from((2, 5)));
        for coefficient in [
            Coefficient::from(0),
            Coefficient::from((-1, 3)),
            Coefficient::from((-i128::MAX, 3)),
            Coefficient::from(complex.clone()),
            Coefficient::from(Complex::new(
                Rational::from(i128::MAX),
                Rational::from((2, 5)),
            )),
            Coefficient::from(Complex::new(
                Rational::from((2, 5)),
                Rational::from(-i128::MAX),
            )),
            Coefficient::Infinity(Some(complex)),
            Coefficient::Infinity(None),
            Coefficient::Indeterminate,
            Coefficient::from_finite_field(field, field_element),
        ] {
            let mut packed = Vec::new();
            coefficient.write_packed(&mut packed);
            packed.extend_from_slice(b"trailing bytes");
            assert_eq!(
                validate_packed_coefficient(&packed).unwrap(),
                b"trailing bytes"
            );
        }
        // Only the outer length is interpreted for rational polynomials.
        let opaque = [RAT_POLY, 3, 0, 0, 0, 255, 0, 128, 42];
        assert_eq!(validate_packed_coefficient(&opaque).unwrap(), &[42]);
    }

    #[test]
    fn packed_float_lengths_and_uncertainty_roundtrip() {
        // Cross the one-byte length limit and include both real and complex
        // values with unequal component precisions.
        for precision in [53, 800, 1024, 4096] {
            for complex in [false, true] {
                let coefficient = float_coefficient(precision, complex);
                let mut packed = Vec::new();
                coefficient.write_packed(&mut packed);
                assert_eq!(packed[0], COMPACT_FLOAT);
                assert_eq!(packed.len() as u64, coefficient.get_packed_size());
                let mut fixed_length = Vec::new();
                write_fixed_length_float(&coefficient, &mut fixed_length);
                if precision <= 800 {
                    assert_eq!(fixed_length.len() - packed.len(), 13);
                } else {
                    assert!(fixed_length.len() > packed.len());
                }
                packed.extend_from_slice(b"trailing bytes");
                fixed_length.extend_from_slice(b"trailing bytes");
                assert_float_roundtrip(&coefficient, &packed);
                assert_float_roundtrip(&coefficient, &fixed_length);
            }
        }
    }

    #[test]
    fn packed_real_floats_omit_exact_imaginary_zeros() {
        for precision in [53, 800, 1024, 4096] {
            for precision_delta in [0, 17] {
                let mut coefficient = float_coefficient(precision, false);
                let Coefficient::Float(f) = &mut coefficient else {
                    unreachable!()
                };
                f.im = ErrorPropagatingFloat::exact_zero(precision + precision_delta);
                let mut packed = Vec::new();
                coefficient.write_packed(&mut packed);
                assert_eq!(packed.len() as u64, coefficient.get_packed_size());
                let (re_len, im_len, _) = read_packed_pair(&packed[1..]);
                assert_eq!(im_len, if precision_delta == 0 { 0 } else { 4 });
                let previous_size = 1 + (re_len, 13).get_packed_size() + re_len + 13;
                assert_eq!(previous_size - packed.len() as u64, 13 - im_len);

                let CoefficientView::Float(re, im) = packed.get_coeff_view().0 else {
                    unreachable!()
                };
                assert_eq!(im.0.len(), 4);
                if precision_delta == 0 {
                    assert_eq!(im.0.as_ptr(), re.0.as_ptr());
                }
                assert!(im.is_zero());
                assert!(im.is_exact_zero());
                assert!(im.is_finite());
                assert!(!im.has_error());
                let mut repacked = Vec::new();
                packed
                    .get_coeff_view()
                    .0
                    .to_owned()
                    .write_packed(&mut repacked);
                assert_eq!(packed, repacked);
                packed.extend_from_slice(b"trailing bytes");
                assert_float_roundtrip(&coefficient, &packed);
            }
        }
    }

    #[test]
    fn packed_imaginary_zero_retains_every_nonzero_error() {
        for error in [
            Magnitude::MIN_POSITIVE,
            Magnitude::ONE.scale_pow2(-4000),
            Magnitude::ONE,
            Magnitude::INFINITY,
            Magnitude::NAN,
        ] {
            let coefficient = Coefficient::Float(Complex::new(
                ErrorPropagatingFloat::exact_zero(53),
                ErrorPropagatingFloat::new_with_error_magnitude(Float::new(91), error),
            ));
            let mut packed = Vec::new();
            coefficient.write_packed(&mut packed);
            assert_eq!(read_packed_pair(&packed[1..]).1, 13);
            assert_eq!(packed.len() as u64, coefficient.get_packed_size());
            let CoefficientView::Float(_, im) = packed.get_coeff_view().0 else {
                unreachable!()
            };
            assert!(im.has_error());
            assert!(im.is_zero());
            assert!(im.is_finite());
            assert!(!im.is_exact_zero());
            packed.extend_from_slice(b"trailing bytes");
            assert_float_roundtrip(&coefficient, &packed);
        }
    }

    #[test]
    fn packed_binary_float_predicates_match_checked_decoding() {
        for precision in [53, 256, 4096] {
            for center in [
                0.,
                -0.,
                1.5,
                -1.5,
                f64::INFINITY,
                f64::NEG_INFINITY,
                f64::NAN,
            ] {
                let coefficient = Coefficient::Float(Complex::new(
                    ErrorPropagatingFloat::new_with_error_magnitude(
                        Float::with_val(precision, center),
                        Magnitude::ONE.scale_pow2(-4000),
                    ),
                    ErrorPropagatingFloat::exact_zero(precision),
                ));
                let mut packed = Vec::new();
                coefficient.write_packed(&mut packed);
                assert_eq!(packed.len() as u64, coefficient.get_packed_size());
                let CoefficientView::Float(real, imaginary) = packed.get_coeff_view().0 else {
                    unreachable!()
                };
                assert_eq!(real.is_zero(), center == 0.);
                assert_eq!(real.is_finite(), center.is_finite());
                assert!(real.has_error());
                assert!(!real.is_exact_zero());
                assert!(imaginary.is_exact_zero() && imaginary.is_finite());
                let decoded = real.try_to_float().unwrap();
                assert_eq!(decoded.get_num().prec(), precision);
                assert_eq!(
                    decoded.get_error_magnitude(),
                    Magnitude::ONE.scale_pow2(-4000)
                );
                let mut repacked = Vec::new();
                packed
                    .get_coeff_view()
                    .0
                    .to_owned()
                    .write_packed(&mut repacked);
                assert_eq!(packed, repacked);
            }
        }
    }

    #[test]
    fn compact_zero_keeps_existing_signed_zero_canonicalization() {
        let mut canonical = Vec::new();
        Coefficient::Float(Complex::new(
            ErrorPropagatingFloat::exact_zero(53),
            ErrorPropagatingFloat::exact_zero(53),
        ))
        .write_packed(&mut canonical);
        for (re, im) in [(0., -0.), (-0., 0.), (-0., -0.)] {
            let coefficient = Coefficient::Float(Complex::new(
                ErrorPropagatingFloat::new_with_error_magnitude(
                    Float::with_val(53, re),
                    Magnitude::ZERO,
                ),
                ErrorPropagatingFloat::new_with_error_magnitude(
                    Float::with_val(53, im),
                    Magnitude::ZERO,
                ),
            ));
            let mut packed = Vec::new();
            coefficient.write_packed(&mut packed);
            assert_eq!(packed, canonical);
            let CoefficientView::Float(re, im) = packed.get_coeff_view().0 else {
                unreachable!()
            };
            assert!(re.is_exact_zero() && im.is_exact_zero());
        }
    }

    /// Run with `cargo test --release -p symbolica packed_float_benchmark -- --ignored --nocapture`.
    #[test]
    #[ignore = "manual packed coefficient memory and throughput benchmark"]
    fn packed_float_benchmark() {
        use std::{hint::black_box, time::Instant};

        let iterations = 200_000;
        for (precision, complex) in [(53, false), (53, true), (256, false), (4096, true)] {
            let coefficient = float_coefficient(precision, complex);
            let mut fixed_length = Vec::new();
            let mut packed = Vec::new();
            write_fixed_length_float(&coefficient, &mut fixed_length);
            coefficient.write_packed(&mut packed);
            let fixed_length_bytes = fixed_length.len();
            let packed_bytes = packed.len();

            let start = Instant::now();
            for _ in 0..iterations {
                fixed_length.clear();
                write_fixed_length_float(black_box(&coefficient), &mut fixed_length);
                black_box(&fixed_length);
            }
            let fixed_length_time = start.elapsed();
            let start = Instant::now();
            for _ in 0..iterations {
                packed.clear();
                black_box(&coefficient).write_packed(&mut packed);
                black_box(&packed);
            }
            let packed_time = start.elapsed();
            eprintln!(
                "precision={precision}, complex={complex}: {fixed_length_bytes} -> {packed_bytes} bytes, fixed_length={:.1} ns/write, compact={:.1} ns/write",
                fixed_length_time.as_nanos() as f64 / iterations as f64,
                packed_time.as_nanos() as f64 / iterations as f64,
            );
        }

        let zero = Coefficient::Float(Complex::new(
            ErrorPropagatingFloat::exact_zero(53),
            ErrorPropagatingFloat::exact_zero(53),
        ));
        let mut data = Vec::new();
        zero.write_packed(&mut data);
        let CoefficientView::Float(real, _) = data.get_coeff_view().0 else {
            unreachable!()
        };
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(black_box(real).to_float().is_exact_zero());
        }
        let old_time = start.elapsed();
        let start = Instant::now();
        for _ in 0..iterations {
            black_box(black_box(real).is_exact_zero());
        }
        eprintln!(
            "exact zero: deserialized={:.1} ns/check, borrowed={:.1} ns/check",
            old_time.as_nanos() as f64 / iterations as f64,
            start.elapsed().as_nanos() as f64 / iterations as f64,
        );
    }
    #[test]
    fn compact_binary_float_roundtrips_and_rejects_fixed_length_imports() {
        for precision in [2, 53, 256, 4096] {
            for imaginary_precision in [precision, precision + 17] {
                for imaginary_value in [0., 0.25] {
                    let value = Complex::new(
                        Float::with_val(precision, 1) / Float::with_val(precision, 3),
                        Float::with_val(imaginary_precision, imaginary_value),
                    );
                    let coefficient = Coefficient::from(value.clone());
                    let mut compact = Vec::new();
                    coefficient.write_packed(&mut compact);
                    assert_eq!(compact.len() as u64, coefficient.get_packed_size());
                    let (_, imaginary_length, _) = read_packed_pair(&compact[1..]);
                    assert_eq!(
                        imaginary_length,
                        if value.im.is_zero() {
                            u64::from(value.re.prec() != value.im.prec()) * 4
                        } else {
                            value.im.serialized_size() as u64 + 12
                        }
                    );
                    let mut fixed = Vec::new();
                    write_fixed_length_float(&coefficient, &mut fixed);
                    assert!(fixed.len() > compact.len());
                    for mut packed in [compact, fixed] {
                        packed.extend_from_slice(b"trailing bytes");
                        if packed[0] == COMPACT_FLOAT {
                            assert_eq!(
                                validate_packed_coefficient(&packed).unwrap(),
                                b"trailing bytes"
                            );
                        } else {
                            assert!(validate_packed_coefficient(&packed).is_err());
                        }
                        assert_eq!(packed.skip_rational(), b"trailing bytes");
                        let (CoefficientView::Float(re, im), rest) = packed.get_coeff_view() else {
                            unreachable!()
                        };
                        assert_eq!(rest, b"trailing bytes");
                        assert_eq!(re.try_to_float().unwrap().get_num(), &value.re);
                        assert_eq!(im.try_to_float().unwrap().get_num(), &value.im);
                        assert_eq!(re.to_float().get_num().prec(), value.re.prec());
                        assert_eq!(im.to_float().get_num().prec(), value.im.prec());
                        assert!(re.is_finite() && im.is_finite());
                        assert_eq!(im.is_zero(), value.im.is_zero());
                    }
                }
            }
        }
    }
}
