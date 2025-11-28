use std::io::Write;

use bytes::{Buf, BufMut};
use rug::{Integer as MultiPrecisionInteger, integer::Order};

use crate::{
    coefficient::{
        Coefficient, CoefficientView, SerializedFloat, SerializedLargeRational, SerializedRational,
        SerializedRationalPolynomial,
    },
    domains::{
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
    pub fn deserialize(self) -> RationalPolynomial<IntegerRing, u16> {
        let mut source = self.0;

        let index;
        let num_nterms;
        let den_nterms;
        (index, num_nterms, source) = source.get_frac_u64();
        (den_nterms, _, source) = source.get_frac_u64();

        let vars = State::get_variable_list(VariableListIndex(index as usize));
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
                2 => Integer::Double(source.get_i128_le()),
                x @ 4 | x @ 5 => {
                    let (num_digits, _, new_source) = source.get_frac_u64();
                    *source = new_source;
                    let i = MultiPrecisionInteger::from_digits(
                        &source[..num_digits as usize],
                        Order::Lsf,
                    );
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
            let r = r.clone().to_multi_prec();
            dest.put_u8(ARB_NUM | ARB_DEN);

            let num_digits = r.numer().significant_digits::<u8>();
            let den_digits = r.denom().significant_digits::<u8>();

            if r.numer() < &0 {
                (-(num_digits as i64), den_digits as u64).write_packed(dest);
            } else {
                (num_digits as i64, den_digits as u64).write_packed(dest);
            }

            let old_len = dest.len();
            dest.resize(old_len + num_digits + den_digits, 0);
            r.numer().write_digits(&mut dest[old_len..], Order::Lsf);
            r.denom()
                .write_digits(&mut dest[old_len + num_digits..], Order::Lsf);
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
                dest.put_u8(FLOAT);

                // TODO: improve serialization
                let s = f.re.get_num().serialize();
                dest.put_u64_le(s.len() as u64 + 12);
                dest.put_u32_le(f.re.get_num().prec());
                dest.put_f64_le(f.re.get_absolute_error());
                dest.write_all(&s).unwrap();

                let s = f.im.get_num().serialize();
                dest.put_u64_le(s.len() as u64 + 12);
                dest.put_u32_le(f.im.get_num().prec());
                dest.put_f64_le(f.im.get_absolute_error());
                dest.write_all(&s).unwrap();
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
                            dest.put_i128_le(*d);
                        }
                        Integer::Large(l) => {
                            if l.is_negative() {
                                dest.put_u8(5);
                            } else {
                                dest.put_u8(4);
                            }
                            let num_digits = l.significant_digits::<u8>();
                            (num_digits as u64, 1).write_packed(dest);
                            let old_len = dest.len();
                            dest.resize(old_len + num_digits, 0);
                            l.write_digits(&mut dest[old_len..], Order::Lsf);
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
            let l = r.clone().to_multi_prec();
            let n = l.numer().significant_digits::<u8>() as u64;
            let d = l.denom().significant_digits::<u8>() as u64;
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
                let s = f.re.get_num().serialize();
                let si = f.im.get_num().serialize();
                1 + 8 + 12 + s.len() as u64 + 8 + 12 + si.len() as u64
            }
            Coefficient::FiniteField(m, i) => 2 + (*m.inner(), i.0 as u64).get_packed_size(),
            Coefficient::RationalPolynomial(_) => {
                unimplemented!("Cannot get the packed size of a rational polynomial")
            }
        }
    }
}

/// A reader for generalized rational numbers. See [`MultiPrecisionRationalNumberWriter`].
pub trait PackedRationalNumberReader {
    fn get_coeff_view(&self) -> (CoefficientView<'_>, &[u8]);
    fn get_frac_u64(&self) -> (u64, u64, &[u8]);
    fn get_frac_i64(&self) -> (i64, i64, &[u8]);
    fn skip_rational(&self) -> &[u8];
    fn is_small_int(&self) -> bool;
    fn is_zero_rat(&self) -> bool;
    fn is_one_rat(&self) -> bool;
}

impl PackedRationalNumberReader for [u8] {
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
        } else if disc == FLOAT {
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
            let (num, den);
            (num, den, source) = source.get_frac_i64();
            let num_len = num.unsigned_abs() as usize;
            let den_len = den.unsigned_abs() as usize;
            let num_limbs = &source[..num_len];
            let den_limbs = &source[num_len..num_len + den_len];

            (
                CoefficientView::Large(
                    SerializedRational::Large(SerializedLargeRational {
                        is_negative: num < 0,
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
            (CoefficientView::Natural(num, den, 0, 1), source)
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
    fn get_frac_i64(&self) -> (i64, i64, &[u8]) {
        let mut source = self;
        let disc = source.get_u8();
        let num;
        (num, source) = match disc & NUM_MASK {
            U8_NUM => {
                let v = source.get_u8();
                (v as i64, source)
            }
            U16_NUM => {
                let v = source.get_u16_le();
                (v as i64, source)
            }
            U32_NUM => {
                let v = source.get_u32_le();
                (v as i64, source)
            }
            U64_NUM => {
                let v = source.get_u64_le();
                (v as i64, source)
            }
            x => {
                unreachable!("Unsupported numerator type {}", x)
            }
        };

        let den;
        (den, source) = match disc & DEN_MASK {
            0 => (1i64, source),
            U8_DEN => {
                let v = source.get_u8();
                (v as i64, source)
            }
            U16_DEN => {
                let v = source.get_u16_le();
                (v as i64, source)
            }
            U32_DEN => {
                let v = source.get_u32_le();
                (v as i64, source)
            }
            U64_DEN => {
                let v = source.get_u64_le();
                (v as i64, source)
            }
            x => {
                unreachable!("Unsupported denominator type {}", x)
            }
        };

        if disc & SIGN != 0 {
            (-num, den, source)
        } else {
            (num, den, source)
        }
    }

    #[inline(always)]
    fn skip_rational(&self) -> &[u8] {
        let mut dest = self;
        let disc = dest.get_u8();

        match disc & (NUM_MASK | DEN_MASK) {
            U8_NUM => {
                dest.advance(1);
            }
            U16_NUM | U8_NUM_U8_DEN => {
                dest.advance(2);
            }
            U16_NUM_U8_DEN | U8_NUM_U16_DEN => {
                dest.advance(3);
            }
            U32_NUM | U16_NUM_U16_DEN => {
                dest.advance(4);
            }
            U32_NUM_U8_DEN | U8_NUM_U32_DEN => {
                dest.advance(5);
            }
            U32_NUM_U16_DEN | U16_NUM_U32_DEN => {
                dest.advance(6);
            }
            U64_NUM | U32_NUM_U32_DEN => {
                dest.advance(8);
            }
            U64_NUM_U8_DEN | U8_NUM_U64_DEN => {
                dest.advance(9);
            }
            U64_NUM_U16_DEN | U16_NUM_U64_DEN => {
                dest.advance(10);
            }
            U64_NUM_U32_DEN | U32_NUM_U64_DEN => {
                dest.advance(12);
            }
            U64_NUM_U64_DEN => {
                dest.advance(16);
            }
            x => {
                let v_num = x & NUM_MASK;
                if v_num == COMPLEX {
                    dest = dest.skip_rational();
                    dest = dest.skip_rational();
                } else if v_num == ARB_NUM {
                    let (num_size, den_size);
                    (num_size, den_size, dest) = dest.get_frac_i64();
                    let num_size = num_size.unsigned_abs() as usize;
                    let den_size = den_size.unsigned_abs() as usize;
                    dest.advance(num_size + den_size);
                } else if v_num == RAT_POLY {
                    let size = dest.get_u32_le() as usize;
                    dest.advance(size);
                } else if v_num == FIN_NUM {
                    let var_size = dest.get_u8();
                    let size = get_size_of_natural(var_size & NUM_MASK)
                        + get_size_of_natural((var_size & DEN_MASK) >> 4);
                    dest.advance(size as usize);
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
            }
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
