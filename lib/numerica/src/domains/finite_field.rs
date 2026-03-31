//! Finite fields and modular rings.

use rand::Rng;
use std::fmt::{Display, Error, Formatter};
use std::hash::Hash;
use std::ops::{Deref, Neg};

use crate::domains::integer::{Integer, gcd_unsigned};
use crate::domains::{RingOps, Set};
use crate::printer::{PrintOptions, PrintState};

use super::integer::Z;
use super::{EuclideanDomain, Field, InternalOrdering, Ring};

const HENSEL_LIFTING_MASK: [u8; 128] = [
    255, 85, 51, 73, 199, 93, 59, 17, 15, 229, 195, 89, 215, 237, 203, 33, 31, 117, 83, 105, 231,
    125, 91, 49, 47, 5, 227, 121, 247, 13, 235, 65, 63, 149, 115, 137, 7, 157, 123, 81, 79, 37, 3,
    153, 23, 45, 11, 97, 95, 181, 147, 169, 39, 189, 155, 113, 111, 69, 35, 185, 55, 77, 43, 129,
    127, 213, 179, 201, 71, 221, 187, 145, 143, 101, 67, 217, 87, 109, 75, 161, 159, 245, 211, 233,
    103, 253, 219, 177, 175, 133, 99, 249, 119, 141, 107, 193, 191, 21, 243, 9, 135, 29, 251, 209,
    207, 165, 131, 25, 151, 173, 139, 225, 223, 53, 19, 41, 167, 61, 27, 241, 239, 197, 163, 57,
    183, 205, 171, 1,
];

/// A 32-bit integer finite field.
pub type Zp = FiniteField<u32>;
/// A 64-bit integer finite field.
pub type Zp64 = FiniteField<u64>;

pub trait ToFiniteField<UField: FiniteFieldWorkspace>
where
    FiniteField<UField>: FiniteFieldCore<UField>,
{
    fn to_finite_field(&self, field: &FiniteField<UField>)
    -> <FiniteField<UField> as Set>::Element;
}

impl ToFiniteField<u32> for u32 {
    fn to_finite_field(&self, field: &FiniteField<u32>) -> <FiniteField<u32> as Set>::Element {
        field.to_element(*self)
    }
}

impl ToFiniteField<u64> for u64 {
    fn to_finite_field(&self, field: &FiniteField<u64>) -> <FiniteField<u64> as Set>::Element {
        field.to_element(*self)
    }
}

impl ToFiniteField<Two> for u32 {
    fn to_finite_field(&self, field: &FiniteField<Two>) -> <FiniteField<Two> as Set>::Element {
        field.to_element(Two((*self % 2) as u8))
    }
}

impl ToFiniteField<Two> for u64 {
    fn to_finite_field(&self, field: &FiniteField<Two>) -> <FiniteField<Two> as Set>::Element {
        field.to_element(Two((*self % 2) as u8))
    }
}

/// A number in a finite field.
#[derive(Debug, Copy, Clone, Hash, PartialEq, PartialOrd, Eq)]
pub struct FiniteFieldElement<UField>(pub(crate) UField);

impl<UField> FiniteFieldElement<UField> {
    /// Get the underlying value, which may be in a special representation.
    pub fn inner(&self) -> &UField {
        &self.0
    }

    /// Create a new finite field element from the inner value.
    /// The value must be in the correct representation for the field.
    pub fn from_inner(value: UField) -> Self {
        FiniteFieldElement(value)
    }
}

impl<UField: PartialOrd> InternalOrdering for FiniteFieldElement<UField> {
    fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

pub trait FiniteFieldWorkspace: Clone + Display + Eq + Hash {
    /// Get a large prime with the guarantee that there are still many primes above
    /// this number in `Self`.
    fn get_large_prime() -> Self;

    fn try_from_integer(n: Integer) -> Option<Self>;

    fn to_integer(&self) -> Integer;

    /// Convert to u64.
    fn to_u64(&self) -> Option<u64> {
        match self.to_integer() {
            Integer::Single(s) => {
                if s >= 0 {
                    Some(s as u64)
                } else {
                    None
                }
            }
            Integer::Double(s) => {
                if s >= 0 && s <= u64::MAX as i128 {
                    Some(s as u64)
                } else {
                    None
                }
            }
            Integer::Large(_) => None,
        }
    }
}

pub trait FiniteFieldCore<UField: FiniteFieldWorkspace>: Field {
    /// Create a new finite field with modulus prime `p`.
    fn new(p: UField) -> Self;
    fn get_prime(&self) -> UField;
    /// Convert a number to a representative in a prime field.
    fn to_element(&self, a: UField) -> Self::Element;
    /// Convert a number from the finite field to standard form `[0,p)`.
    fn from_element(&self, a: &Self::Element) -> UField;
    // Convert a number from the finite field to standard form `[0,p)`.
    fn to_integer(&self, a: &Self::Element) -> Integer;
    /// Convert a number from the finite field to symmetric form `[-p/2,p/2]`.
    fn to_symmetric_integer(&self, a: &Self::Element) -> Integer {
        let i = self.to_integer(a);
        let p = self.get_prime().to_integer();

        if &i * 2 <= p { i } else { i - p }
    }
}

/// The modular ring `Z / mZ`, where `m` can be any odd positive integer. In most cases,
/// `m` will be a prime, and the domain will be a field.
///
/// [Zp] ([`FiniteField<u32>`]) and [Zp64] ([`FiniteField<u64>`]) use Montgomery modular arithmetic
/// to increase the performance of the multiplication operator. For the prime `2`, use [type@Z2] instead.
///
/// For `m` larger than `2^64`, use [`FiniteField<Integer>`].
///
/// The special field [`FiniteField<Mersenne64>`] can be used to have even faster arithmetic
/// for a field with Mersenne prime `2^61-1`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FiniteField<UField> {
    p: UField,
    m: UField,
    one: FiniteFieldElement<UField>,
    is_prime: bool,
}

impl Zp {
    /// Create a new modular ring. `p` must be odd.
    pub fn new_non_prime(p: u32) -> Zp {
        if p.is_multiple_of(2) {
            panic!("Prime 2 is not supported: use Z2 instead.");
        }

        FiniteField {
            p,
            m: Self::inv_2_32(p),
            one: FiniteFieldElement(Self::get_one(p)),
            is_prime: false,
        }
    }

    /// Create a new modular field from an odd prime `p`.
    pub fn new(p: u32) -> Zp {
        if p.is_multiple_of(2) {
            panic!("Prime 2 is not supported: use Z2 instead.");
        }

        FiniteField {
            p,
            m: Self::inv_2_32(p),
            one: FiniteFieldElement(Self::get_one(p)),
            is_prime: true,
        }
    }

    /// Returns the unit element in Montgomory form, ie.e 1 + 2^32 mod a.
    fn get_one(a: u32) -> u32 {
        if a as u64 <= 1u64 << 31 {
            let res = (((1u64 << 31) % a as u64) << 1) as u32;

            if res < a { res } else { res - a }
        } else {
            a.wrapping_neg()
        }
    }

    /// Returns -a^-1 mod 2^32.
    fn inv_2_32(a: u32) -> u32 {
        let mut ret: u32 = HENSEL_LIFTING_MASK[((a >> 1) & 127) as usize] as u32;
        ret = ret.wrapping_mul(a.wrapping_mul(ret).wrapping_add(2));
        ret = ret.wrapping_mul(a.wrapping_mul(ret).wrapping_add(2));
        ret
    }
}

impl FiniteFieldWorkspace for u32 {
    fn get_large_prime() -> u32 {
        2147483659
    }

    fn try_from_integer(n: Integer) -> Option<Self> {
        match n {
            Integer::Single(s) => {
                if s >= 0 && s <= u32::MAX as i64 {
                    Some(s as u32)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn to_integer(&self) -> Integer {
        Integer::Single(*self as i64)
    }
}

impl FiniteFieldCore<u32> for Zp {
    /// Create a new finite field. `n` must be a prime larger than 2.
    fn new(p: u32) -> Zp {
        Self::new(p)
    }

    fn get_prime(&self) -> u32 {
        self.p
    }

    /// Convert a number in a prime field a % n to Montgomory form.
    #[inline(always)]
    fn to_element(&self, a: u32) -> FiniteFieldElement<u32> {
        // TODO: slow, faster alternatives may need assembly
        FiniteFieldElement((((a as u64) << 32) % self.p as u64) as u32)
    }

    /// Convert a number from Montgomory form to standard form.
    #[inline(always)]
    fn from_element(&self, a: &FiniteFieldElement<u32>) -> u32 {
        self.mul(a, &FiniteFieldElement(1)).0
    }

    fn to_integer(&self, a: &Self::Element) -> Integer {
        self.from_element(a).into()
    }
}

impl Set for Zp {
    type Element = FiniteFieldElement<u32>;

    fn size(&self) -> Option<Integer> {
        Some(self.get_prime().into())
    }
}

impl RingOps<FiniteFieldElement<u32>> for Zp {
    /// Add two numbers in Montgomory form.
    #[inline(always)]
    fn add(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        let mut t = a.0 as u64 + b.0 as u64;

        if t >= self.p as u64 {
            t -= self.p as u64;
        }

        FiniteFieldElement(t as u32)
    }

    /// Subtract `b` from `a`, where `a` and `b` are in Montgomory form.
    #[inline(always)]
    fn sub(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        if a.0 >= b.0 {
            FiniteFieldElement(a.0 - b.0)
        } else {
            FiniteFieldElement(a.0 + (self.p - b.0))
        }
    }

    /// Multiply two numbers in Montgomory form.
    #[inline(always)]
    fn mul(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        let t = a.0 as u64 * b.0 as u64;
        let m = (t as u32).wrapping_mul(self.m);
        let (t, overflow) = t.overflowing_add(m as u64 * self.p as u64);
        let u = (t >> 32) as u32;

        if overflow {
            FiniteFieldElement(u.wrapping_sub(self.p))
        } else if u >= self.p {
            FiniteFieldElement(u - self.p)
        } else {
            FiniteFieldElement(u)
        }
    }

    #[inline(always)]
    fn add_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a = self.add(*a, b);
    }

    #[inline(always)]
    fn sub_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a = self.sub(*a, b);
    }

    #[inline(always)]
    fn mul_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a = self.mul(*a, b);
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: Self::Element, c: Self::Element) {
        self.add_assign(a, self.mul(b, c));
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: Self::Element, c: Self::Element) {
        self.sub_assign(a, self.mul(b, c));
    }

    /// Computes -x mod n.
    #[inline]
    fn neg(&self, a: Self::Element) -> Self::Element {
        if a.0 == 0 {
            a
        } else {
            FiniteFieldElement(self.p - a.0)
        }
    }
}

impl RingOps<&FiniteFieldElement<u32>> for Zp {
    /// Add two numbers in Montgomory form.
    #[inline(always)]
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        let mut t = a.0 as u64 + b.0 as u64;

        if t >= self.p as u64 {
            t -= self.p as u64;
        }

        FiniteFieldElement(t as u32)
    }

    /// Subtract `b` from `a`, where `a` and `b` are in Montgomory form.
    #[inline(always)]
    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        if a.0 >= b.0 {
            FiniteFieldElement(a.0 - b.0)
        } else {
            FiniteFieldElement(a.0 + (self.p - b.0))
        }
    }

    /// Multiply two numbers in Montgomory form.
    #[inline(always)]
    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        let t = a.0 as u64 * b.0 as u64;
        let m = (t as u32).wrapping_mul(self.m);
        let (t, overflow) = t.overflowing_add(m as u64 * self.p as u64);
        let u = (t >> 32) as u32;

        if overflow {
            FiniteFieldElement(u.wrapping_sub(self.p))
        } else if u >= self.p {
            FiniteFieldElement(u - self.p)
        } else {
            FiniteFieldElement(u)
        }
    }

    #[inline(always)]
    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.add(&*a, b);
    }

    #[inline(always)]
    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.sub(&*a, b);
    }

    #[inline(always)]
    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(&*a, b);
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.add_assign(a, &self.mul(b, c));
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.sub_assign(a, &self.mul(b, c));
    }

    /// Computes -x mod n.
    #[inline]
    fn neg(&self, a: &Self::Element) -> Self::Element {
        if a.0 == 0 {
            *a
        } else {
            FiniteFieldElement(self.p - a.0)
        }
    }
}

impl Ring for Zp {
    #[inline]
    fn zero(&self) -> Self::Element {
        FiniteFieldElement(0)
    }

    /// Return the unit element in Montgomory form.
    #[inline]
    fn one(&self) -> Self::Element {
        self.one
    }

    #[inline]
    fn nth(&self, n: Integer) -> Self::Element {
        n.to_finite_field(self)
    }

    /// Compute b^e % n.
    #[inline]
    fn pow(&self, b: &Self::Element, mut e: u64) -> Self::Element {
        if self.is_prime && e >= self.get_prime() as u64 - 1 {
            e %= self.get_prime() as u64 - 1;
        }

        if e == 0 {
            return self.one();
        }

        let mut x = *b;
        let mut y = self.one();
        while e != 1 {
            if e % 2 == 1 {
                y = self.mul(&y, &x);
            }

            x = self.mul(&x, &x);
            e /= 2;
        }

        self.mul(&x, &y)
    }

    #[inline]
    fn is_zero(&self, a: &Self::Element) -> bool {
        a.0 == 0
    }

    #[inline]
    fn is_one(&self, a: &Self::Element) -> bool {
        a == &self.one
    }

    fn one_is_gcd_unit() -> bool {
        true
    }

    fn characteristic(&self) -> Integer {
        self.get_prime().into()
    }

    fn try_inv(&self, a: &Self::Element) -> Option<Self::Element> {
        if self.is_zero(a) {
            return None;
        }

        // apply multiplication with 1 twice to get the correct scaling of R=2^32
        // see the paper [Montgomery Arithmetic from a Software Perspective](https://eprint.iacr.org/2017/1057.pdf).
        let x_mont = self
            .mul(&self.mul(a, &FiniteFieldElement(1)), &FiniteFieldElement(1))
            .0;

        // extended Euclidean algorithm: a x + b p = gcd(x, p) = 1 or a x = 1 (mod p)
        let mut u1: u32 = 1;
        let mut u3 = x_mont;
        let mut v1: u32 = 0;
        let mut v3 = self.p;
        let mut even_iter: bool = true;

        while v3 != 0 {
            let q = u3 / v3;
            let t3 = u3 % v3;
            let t1 = u1 + q * v1;
            u1 = v1;
            v1 = t1;
            u3 = v3;
            v3 = t3;
            even_iter = !even_iter;
        }

        if u3 != 1 {
            return None;
        }

        if even_iter {
            Some(FiniteFieldElement(u1))
        } else {
            Some(FiniteFieldElement(self.p - u1))
        }
    }

    fn try_div(&self, a: &Self::Element, b: &Self::Element) -> Option<Self::Element> {
        if self.is_zero(b) {
            None
        } else {
            Some(self.div(a, b))
        }
    }

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        let r = rng.random_range(range.0.max(0)..range.1.min(self.p as i64));
        FiniteFieldElement(r as u32)
    }

    fn format<W: std::fmt::Write>(
        &self,
        element: &Self::Element,
        opts: &PrintOptions,
        state: PrintState,
        f: &mut W,
    ) -> Result<bool, Error> {
        if opts.symmetric_representation_for_finite_field {
            Z.format(&self.to_symmetric_integer(element), opts, state, f)
        } else {
            Z.format(&self.from_element(element).into(), opts, state, f)
        }
    }
}

impl EuclideanDomain for Zp {
    #[inline]
    fn rem(&self, _: &Self::Element, _: &Self::Element) -> Self::Element {
        FiniteFieldElement(0)
    }

    #[inline]
    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        (self.mul(a, &self.inv(b)), FiniteFieldElement(0))
    }

    #[inline]
    fn gcd(&self, _: &Self::Element, _: &Self::Element) -> Self::Element {
        self.one()
    }
}

impl Field for Zp {
    #[inline]
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.mul(a, &self.inv(b))
    }

    #[inline]
    fn div_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(&*a, &self.inv(b));
    }

    /// Computes x^-1 mod n.
    fn inv(&self, a: &Self::Element) -> Self::Element {
        self.try_inv(a)
            .unwrap_or_else(|| panic!("{} is not invertible mod {}", self.printer(a), self.p))
    }
}

impl FiniteFieldWorkspace for u64 {
    fn get_large_prime() -> u64 {
        18346744073709552000
    }

    fn try_from_integer(n: Integer) -> Option<Self> {
        match n {
            Integer::Single(s) => {
                if s >= 0 {
                    Some(s as u64)
                } else {
                    None
                }
            }
            Integer::Double(d) => {
                if d >= 0 && d <= u64::MAX as i128 {
                    Some(d as u64)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    #[inline]
    fn to_integer(&self) -> Integer {
        (*self).into()
    }
}

impl Zp64 {
    /// Create a new modular ring. `n` must be odd.
    pub fn new_non_prime(p: u64) -> Zp64 {
        if p.is_multiple_of(2) {
            panic!("Prime 2 is not supported: use Z2 instead.");
        }

        FiniteField {
            p,
            m: Self::inv_2_64(p),
            one: FiniteFieldElement(Self::get_one(p)),
            is_prime: false,
        }
    }

    // Create a new modular field with odd prime `p`.
    pub fn new(p: u64) -> Zp64 {
        if p.is_multiple_of(2) {
            panic!("Prime 2 is not supported: use Z2 instead.");
        }

        FiniteField {
            p,
            m: Self::inv_2_64(p),
            one: FiniteFieldElement(Self::get_one(p)),
            is_prime: true,
        }
    }

    /// Compute the discrete logarithm
    /// `x = log_base(res) mod p`, where `phi(p) = q_1^e_1 * ... * q_n^e_n`
    /// and `base` is a primitive root of `p`, using the Pohlig–Hellman algorithm.
    pub fn discrete_log(
        &self,
        base: &FiniteFieldElement<u64>,
        res: &FiniteFieldElement<u64>,
        totient: u64,
        totient_primes: &[(u64, u32)],
    ) -> FiniteFieldElement<u64> {
        let mut crt = vec![];
        for (p, e) in totient_primes {
            let p_e = p.to_u64().unwrap().pow(*e);
            let exp = totient.to_u64().unwrap() / p.to_u64().unwrap().pow(*e);

            let g = self.pow(base, exp);
            let g_inv = self.inv(&g);
            let h = self.pow(res, exp);

            let mut x = 0;
            let gamma = self.pow(&g, p.pow(*e - 1));

            'next: for k in 0..*e {
                let hh = self.pow(&self.mul(&self.pow(&g_inv, x), &h), p.pow(*e - 1 - k));

                if self.is_one(&hh) {
                    continue;
                }

                // assume smooth prime with small factors
                // TODO: switch to baby-step giant-step algorithm
                let mut gamma_c = gamma;
                for d in 1..*p {
                    if gamma_c == hh {
                        x += p.pow(k) * d;
                        continue 'next;
                    }

                    self.mul_assign(&mut gamma_c, &gamma);
                }

                panic!(
                    "No discrete logarithm found for base {} and res {} in field with prime {}^{}",
                    base.0, res.0, p, e
                );
            }

            crt.push((x, p_e));
        }

        if crt.len() == 1 {
            return self.to_element(crt[0].0);
        }

        let mut cur = Integer::chinese_remainder(
            crt[0].0.into(),
            crt[1].0.into(),
            crt[0].1.into(),
            crt[1].1.into(),
        );
        let mut prime = Integer::from(crt[0].1) * crt[1].1;
        for x in crt.iter().skip(2) {
            cur = Integer::chinese_remainder(cur, x.0.into(), prime.clone(), x.1.into());
            prime *= x.1;
        }

        if cur < 0 {
            cur += prime;
        }

        let r = self.to_element(cur.to_u64().unwrap());
        debug_assert_eq!(res, &self.pow(base, self.from_element(&r)));
        r
    }

    /// Returns the unit element in Montgomory form, ie.e 1 + 2^64 mod a.
    fn get_one(a: u64) -> u64 {
        if a as u128 <= 1u128 << 63 {
            let res = (((1u128 << 63) % a as u128) << 1) as u64;

            if res < a { res } else { res - a }
        } else {
            a.wrapping_neg()
        }
    }

    /// Returns -a^-1 mod 2^64.
    fn inv_2_64(a: u64) -> u64 {
        let mut ret: u64 = HENSEL_LIFTING_MASK[((a >> 1) & 127) as usize] as u64;
        ret = ret.wrapping_mul(a.wrapping_mul(ret).wrapping_add(2));
        ret = ret.wrapping_mul(a.wrapping_mul(ret).wrapping_add(2));
        ret = ret.wrapping_mul(a.wrapping_mul(ret).wrapping_add(2));
        ret
    }
}

impl FiniteFieldCore<u64> for Zp64 {
    /// Create a new finite field. `n` must be a prime larger than 2.
    fn new(p: u64) -> Zp64 {
        Self::new(p)
    }

    fn get_prime(&self) -> u64 {
        self.p
    }

    /// Convert a number in a prime field a % n to Montgomory form.
    #[inline(always)]
    fn to_element(&self, a: u64) -> FiniteFieldElement<u64> {
        // TODO: slow, faster alternatives may need assembly
        FiniteFieldElement((((a as u128) << 64) % self.p as u128) as u64)
    }

    /// Convert a number from Montgomory form to standard form.
    #[inline(always)]
    fn from_element(&self, a: &FiniteFieldElement<u64>) -> u64 {
        self.mul(a, &FiniteFieldElement(1)).0
    }

    fn to_integer(&self, a: &Self::Element) -> Integer {
        self.from_element(a).into()
    }
}

impl<UField: Display> Display for FiniteField<UField> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, " % {}", self.p)
    }
}

impl Set for Zp64 {
    type Element = FiniteFieldElement<u64>;

    fn size(&self) -> Option<Integer> {
        Some(self.get_prime().into())
    }
}

impl RingOps<FiniteFieldElement<u64>> for Zp64 {
    /// Add two numbers in Montgomory form.
    #[inline(always)]
    fn add(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        // avoid f128 arithmetic
        let (r, overflow) = a.0.overflowing_add(b.0);
        if overflow || r >= self.p {
            FiniteFieldElement(r.wrapping_sub(self.p))
        } else {
            FiniteFieldElement(r)
        }
    }
    /// Subtract `b` from `a`, where `a` and `b` are in Montgomory form.
    #[inline(always)]
    fn sub(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        if a.0 >= b.0 {
            FiniteFieldElement(a.0 - b.0)
        } else {
            FiniteFieldElement(a.0 + (self.p - b.0))
        }
    }

    /// Multiply two numbers in Montgomory form.
    #[inline(always)]
    fn mul(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        let t = a.0 as u128 * b.0 as u128;
        let m = (t as u64).wrapping_mul(self.m);
        let (t, overflow) = t.overflowing_add(m as u128 * self.p as u128);
        let u = (t >> 64) as u64;

        if overflow {
            FiniteFieldElement(u.wrapping_sub(self.p))
        } else if u >= self.p {
            FiniteFieldElement(u - self.p)
        } else {
            FiniteFieldElement(u)
        }
    }

    #[inline]
    fn add_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a = self.add(*a, b);
    }

    #[inline]
    fn sub_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a = self.sub(*a, b);
    }

    #[inline]
    fn mul_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a = self.mul(*a, b);
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: Self::Element, c: Self::Element) {
        self.add_assign(a, &self.mul(b, c));
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: Self::Element, c: Self::Element) {
        self.sub_assign(a, &self.mul(b, c));
    }

    /// Computes -x mod n.
    #[inline]
    fn neg(&self, a: Self::Element) -> Self::Element {
        if a.0 == 0 {
            a
        } else {
            FiniteFieldElement(self.p - a.0)
        }
    }
}

impl RingOps<&FiniteFieldElement<u64>> for Zp64 {
    /// Add two numbers in Montgomory form.
    #[inline(always)]
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        // avoid f128 arithmetic
        let (r, overflow) = a.0.overflowing_add(b.0);
        if overflow || r >= self.p {
            FiniteFieldElement(r.wrapping_sub(self.p))
        } else {
            FiniteFieldElement(r)
        }
    }
    /// Subtract `b` from `a`, where `a` and `b` are in Montgomory form.
    #[inline(always)]
    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        if a.0 >= b.0 {
            FiniteFieldElement(a.0 - b.0)
        } else {
            FiniteFieldElement(a.0 + (self.p - b.0))
        }
    }

    /// Multiply two numbers in Montgomory form.
    #[inline(always)]
    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        let t = a.0 as u128 * b.0 as u128;
        let m = (t as u64).wrapping_mul(self.m);
        let (t, overflow) = t.overflowing_add(m as u128 * self.p as u128);
        let u = (t >> 64) as u64;

        if overflow {
            FiniteFieldElement(u.wrapping_sub(self.p))
        } else if u >= self.p {
            FiniteFieldElement(u - self.p)
        } else {
            FiniteFieldElement(u)
        }
    }

    #[inline]
    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.add(&*a, b);
    }

    #[inline]
    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.sub(&*a, b);
    }

    #[inline]
    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(&*a, b);
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.add_assign(a, &self.mul(b, c));
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.sub_assign(a, &self.mul(b, c));
    }

    /// Computes -x mod n.
    #[inline]
    fn neg(&self, a: &Self::Element) -> Self::Element {
        if a.0 == 0 {
            *a
        } else {
            FiniteFieldElement(self.p - a.0)
        }
    }
}

impl Ring for Zp64 {
    #[inline]
    fn zero(&self) -> Self::Element {
        FiniteFieldElement(0)
    }

    /// Return the unit element in Montgomory form.
    #[inline]
    fn one(&self) -> Self::Element {
        self.one
    }

    #[inline]
    fn nth(&self, n: Integer) -> Self::Element {
        n.to_finite_field(self)
    }

    /// Compute b^e % n.
    #[inline]
    fn pow(&self, b: &Self::Element, mut e: u64) -> Self::Element {
        if self.is_prime && e >= self.get_prime() - 1 {
            e %= self.get_prime() - 1;
        }

        if e == 0 {
            return self.one();
        }

        let mut x = *b;
        let mut y = self.one();
        while e != 1 {
            if e % 2 == 1 {
                y = self.mul(&y, &x);
            }

            x = self.mul(&x, &x);
            e /= 2;
        }

        self.mul(&x, &y)
    }

    #[inline]
    fn is_zero(&self, a: &Self::Element) -> bool {
        a.0 == 0
    }

    #[inline]
    fn is_one(&self, a: &Self::Element) -> bool {
        a == &self.one
    }

    fn one_is_gcd_unit() -> bool {
        true
    }

    fn characteristic(&self) -> Integer {
        self.get_prime().into()
    }

    fn try_inv(&self, a: &Self::Element) -> Option<Self::Element> {
        if self.is_zero(a) {
            return None;
        }

        // apply multiplication with 1 twice to get the correct scaling of R=2^64
        // see the paper [Montgomery Arithmetic from a Software Perspective](https://eprint.iacr.org/2017/1057.pdf).
        let x_mont = self
            .mul(&self.mul(a, &FiniteFieldElement(1)), &FiniteFieldElement(1))
            .0;

        // extended Euclidean algorithm: a x + b p = gcd(x, p) = 1 or a x = 1 (mod p)
        let mut u1: u64 = 1;
        let mut u3 = x_mont;
        let mut v1: u64 = 0;
        let mut v3 = self.p;
        let mut even_iter: bool = true;

        while v3 != 0 {
            let q = u3 / v3;
            let t3 = u3 % v3;
            let t1 = u1 + q * v1;
            u1 = v1;
            v1 = t1;
            u3 = v3;
            v3 = t3;
            even_iter = !even_iter;
        }

        if u3 != 1 {
            return None;
        }

        if even_iter {
            Some(FiniteFieldElement(u1))
        } else {
            Some(FiniteFieldElement(self.p - u1))
        }
    }

    fn try_div(&self, a: &Self::Element, b: &Self::Element) -> Option<Self::Element> {
        if self.is_zero(b) {
            None
        } else {
            Some(self.div(a, b))
        }
    }

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        let r = rng.random_range(range.0.max(0)..range.1.min(self.p.min(i64::MAX as u64) as i64));
        FiniteFieldElement(r as u64)
    }

    fn format<W: std::fmt::Write>(
        &self,
        element: &Self::Element,
        opts: &PrintOptions,
        state: PrintState,
        f: &mut W,
    ) -> Result<bool, Error> {
        if opts.symmetric_representation_for_finite_field {
            Z.format(&self.to_symmetric_integer(element), opts, state, f)
        } else {
            Z.format(&self.from_element(element).into(), opts, state, f)
        }
    }
}

impl EuclideanDomain for Zp64 {
    #[inline]
    fn rem(&self, _: &Self::Element, _: &Self::Element) -> Self::Element {
        FiniteFieldElement(0)
    }

    #[inline]
    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        (self.mul(*a, self.inv(b)), FiniteFieldElement(0))
    }

    #[inline]
    fn gcd(&self, _: &Self::Element, _: &Self::Element) -> Self::Element {
        self.one()
    }
}

impl Field for Zp64 {
    #[inline]
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.mul(*a, self.inv(b))
    }

    #[inline]
    fn div_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(*a, self.inv(b));
    }

    /// Computes x^-1 mod n.
    fn inv(&self, a: &Self::Element) -> Self::Element {
        self.try_inv(a)
            .unwrap_or_else(|| panic!("{} is not invertible mod {}", self.printer(a), self.p))
    }
}

/// The finite field with 0 and 1 as elements.
pub type Z2 = FiniteField<Two>;
/// The finite field with 0 and 1 as elements.
pub const Z2: FiniteField<Two> = Z2::new();

/// A finite field element of the prime 2.
#[derive(Copy, Clone, Hash, Eq, PartialEq)]
pub struct Two(pub(crate) u8);

impl Default for Z2 {
    fn default() -> Self {
        Self::new()
    }
}

impl Z2 {
    /// Create a new finite field with prime 2.
    pub const fn new() -> Z2 {
        FiniteField {
            p: Two(2),
            m: Two(2),
            one: FiniteFieldElement(Two(1)),
            is_prime: true,
        }
    }

    // Get the prime 2.
    pub fn get_prime() -> Two {
        Two(2)
    }
}

impl Two {
    pub const fn new() -> Two {
        Two(2)
    }
}

impl Default for Two {
    fn default() -> Self {
        Two(2)
    }
}

impl Deref for Two {
    type Target = u8;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::fmt::Debug for Two {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.0, f)
    }
}

impl Display for Two {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl FiniteFieldWorkspace for Two {
    fn get_large_prime() -> Two {
        Two(2)
    }

    fn try_from_integer(n: Integer) -> Option<Self> {
        if n == 0 {
            Some(Two(0))
        } else if n == 1 {
            Some(Two(1))
        } else if n == 2 {
            Some(Two(2))
        } else {
            None
        }
    }

    fn to_integer(&self) -> Integer {
        self.0.into()
    }
}

impl FiniteFieldCore<Two> for FiniteField<Two> {
    fn new(p: Two) -> Self {
        FiniteField {
            p,
            m: p,
            one: FiniteFieldElement(Two(1)),
            is_prime: true,
        }
    }

    fn get_prime(&self) -> Two {
        Two(2)
    }

    fn to_element(&self, a: Two) -> Self::Element {
        a.0 % 2
    }

    fn from_element(&self, a: &Self::Element) -> Two {
        Two(*a)
    }

    fn to_integer(&self, a: &Self::Element) -> Integer {
        self.from_element(a).0.into()
    }
}

impl Set for FiniteField<Two> {
    type Element = u8;

    fn size(&self) -> Option<Integer> {
        Some(2.into())
    }
}

impl RingOps<u8> for FiniteField<Two> {
    #[inline(always)]
    fn add(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        a ^ b
    }

    #[inline(always)]
    fn sub(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        a ^ b
    }

    #[inline(always)]
    fn mul(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        a * b
    }

    #[inline]
    fn add_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a = self.add(*a, b);
    }

    #[inline]
    fn sub_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a = self.sub(*a, b);
    }

    #[inline]
    fn mul_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a = self.mul(*a, b);
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: Self::Element, c: Self::Element) {
        self.add_assign(a, &self.mul(b, c));
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: Self::Element, c: Self::Element) {
        self.sub_assign(a, &self.mul(b, c));
    }

    /// Computes -x mod n.
    #[inline]
    fn neg(&self, a: Self::Element) -> Self::Element {
        a
    }
}

impl RingOps<&u8> for FiniteField<Two> {
    #[inline(always)]
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a ^ b
    }

    #[inline(always)]
    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a ^ b
    }

    #[inline(always)]
    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        *a * *b
    }

    #[inline]
    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.add(&*a, b);
    }

    #[inline]
    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.sub(&*a, b);
    }

    #[inline]
    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(&*a, b);
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.add_assign(a, &self.mul(b, c));
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.sub_assign(a, &self.mul(b, c));
    }

    /// Computes -x mod n.
    #[inline]
    fn neg(&self, a: &Self::Element) -> Self::Element {
        *a
    }
}

impl Ring for FiniteField<Two> {
    #[inline]
    fn zero(&self) -> Self::Element {
        0
    }

    /// Return the unit element in Montgomory form.
    #[inline]
    fn one(&self) -> Self::Element {
        1
    }

    #[inline]
    fn nth(&self, n: Integer) -> Self::Element {
        (n % 2i32).to_i64().unwrap() as u8
    }

    /// Compute b^e % n.
    #[inline]
    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element {
        if e == 0 { 1 } else { *b }
    }

    #[inline]
    fn is_zero(&self, a: &Self::Element) -> bool {
        *a == 0
    }

    #[inline]
    fn is_one(&self, a: &Self::Element) -> bool {
        *a == 1
    }

    fn one_is_gcd_unit() -> bool {
        true
    }

    fn characteristic(&self) -> Integer {
        2.into()
    }

    fn try_inv(&self, a: &Self::Element) -> Option<Self::Element> {
        if *a == 0 { None } else { Some(self.inv(a)) }
    }

    fn try_div(&self, a: &Self::Element, b: &Self::Element) -> Option<Self::Element> {
        if *b == 0 { None } else { Some(*a) }
    }

    fn sample(&self, rng: &mut impl rand::RngCore, _range: (i64, i64)) -> Self::Element {
        rng.random_range(0..2)
    }

    fn format<W: std::fmt::Write>(
        &self,
        element: &Self::Element,
        opts: &PrintOptions,
        state: PrintState,
        f: &mut W,
    ) -> Result<bool, Error> {
        if opts.symmetric_representation_for_finite_field {
            Z.format(&self.to_symmetric_integer(element), opts, state, f)
        } else {
            Z.format(&self.from_element(element).0.into(), opts, state, f)
        }
    }
}

impl EuclideanDomain for FiniteField<Two> {
    #[inline]
    fn rem(&self, _: &Self::Element, _: &Self::Element) -> Self::Element {
        0
    }

    #[inline]
    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        (self.mul(*a, self.inv(b)), 0)
    }

    #[inline]
    fn gcd(&self, _: &Self::Element, _: &Self::Element) -> Self::Element {
        1
    }
}

impl Field for FiniteField<Two> {
    #[inline]
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.mul(a, &self.inv(b))
    }

    #[inline]
    fn div_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(*a, self.inv(b));
    }

    /// Computes x^-1 mod n.
    fn inv(&self, a: &Self::Element) -> Self::Element {
        assert!(*a != 0, "0 is not invertible");
        1
    }
}

/// The 64-bit Mersenne prime 2^61 -1.
///
/// Can be used for faster finite field arithmetic
/// w.r.t using Montgomery numbers.
#[derive(Copy, Clone, Hash, Eq, PartialEq)]
pub struct Mersenne64(u64);

impl Default for Mersenne64 {
    fn default() -> Self {
        Self::new()
    }
}

impl Mersenne64 {
    pub fn new() -> Self {
        Mersenne64(Self::PRIME)
    }

    const SHIFT: u8 = 61;
    pub const PRIME: u64 = (1 << Mersenne64::SHIFT) - 1;
}

impl std::fmt::Debug for Mersenne64 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.0, f)
    }
}

impl Display for Mersenne64 {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl FiniteFieldWorkspace for Mersenne64 {
    fn get_large_prime() -> Mersenne64 {
        Mersenne64(Self::PRIME)
    }

    fn try_from_integer(n: Integer) -> Option<Self> {
        if n <= Self::PRIME {
            match n {
                Integer::Single(s) => {
                    if s >= 0 {
                        Some(Mersenne64(s as u64))
                    } else {
                        None
                    }
                }
                Integer::Double(d) => {
                    if d >= 0 && d <= Self::PRIME as i128 {
                        Some(Mersenne64(d as u64))
                    } else {
                        None
                    }
                }
                _ => None,
            }
        } else {
            None
        }
    }

    fn to_integer(&self) -> Integer {
        self.0.into()
    }
}

impl FiniteFieldCore<Mersenne64> for FiniteField<Mersenne64> {
    fn new(p: Mersenne64) -> Self {
        FiniteField {
            p,
            m: p,
            one: FiniteFieldElement(Mersenne64(1)),
            is_prime: true,
        }
    }

    fn get_prime(&self) -> Mersenne64 {
        Mersenne64(Mersenne64::PRIME)
    }

    fn to_element(&self, a: Mersenne64) -> Self::Element {
        if a.0 >= Mersenne64::PRIME {
            a.0 - Mersenne64::PRIME
        } else {
            a.0
        }
    }

    fn from_element(&self, a: &Self::Element) -> Mersenne64 {
        Mersenne64(*a)
    }

    fn to_integer(&self, a: &Self::Element) -> Integer {
        self.from_element(a).0.into()
    }
}

impl Set for FiniteField<Mersenne64> {
    type Element = u64;

    fn size(&self) -> Option<Integer> {
        Some(Mersenne64::PRIME.into())
    }
}

impl RingOps<u64> for FiniteField<Mersenne64> {
    #[inline(always)]
    fn add(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        let mut sum = a + b; // cannot overflow
        if sum >= Mersenne64::PRIME {
            sum -= Mersenne64::PRIME;
        }
        sum
    }

    #[inline(always)]
    fn sub(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        if a >= b {
            a - b
        } else {
            a + (Mersenne64::PRIME - b)
        }
    }

    #[inline(always)]
    fn mul(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        let v = a as u128 * b as u128;
        let q = (v >> Mersenne64::SHIFT) as u64;
        let r = (v as u64 & Mersenne64::PRIME) + (q & Mersenne64::PRIME);

        if r >= Mersenne64::PRIME {
            r - Mersenne64::PRIME
        } else {
            r
        }
    }

    #[inline]
    fn add_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a = self.add(*a, b);
    }

    #[inline]
    fn sub_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a = self.sub(*a, b);
    }

    #[inline]
    fn mul_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a = self.mul(*a, b);
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: Self::Element, c: Self::Element) {
        self.add_assign(a, &self.mul(b, c));
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: Self::Element, c: Self::Element) {
        self.sub_assign(a, &self.mul(b, c));
    }

    /// Computes -x mod n.
    #[inline]
    fn neg(&self, a: Self::Element) -> Self::Element {
        if a == 0 { a } else { Mersenne64::PRIME - a }
    }
}

impl RingOps<&u64> for FiniteField<Mersenne64> {
    #[inline(always)]
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        let mut sum = a + b; // cannot overflow
        if sum >= Mersenne64::PRIME {
            sum -= Mersenne64::PRIME;
        }
        sum
    }

    #[inline(always)]
    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        if *a >= *b {
            *a - *b
        } else {
            *a + (Mersenne64::PRIME - *b)
        }
    }

    #[inline(always)]
    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        let v = *a as u128 * *b as u128;
        let q = (v >> Mersenne64::SHIFT) as u64;
        let r = (v as u64 & Mersenne64::PRIME) + (q & Mersenne64::PRIME);

        if r >= Mersenne64::PRIME {
            r - Mersenne64::PRIME
        } else {
            r
        }
    }

    #[inline]
    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.add(*a, *b);
    }

    #[inline]
    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.sub(*a, *b);
    }

    #[inline]
    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(*a, *b);
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.add_assign(a, self.mul(b, c));
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.sub_assign(a, self.mul(b, c));
    }

    /// Computes -x mod n.
    #[inline]
    fn neg(&self, a: &Self::Element) -> Self::Element {
        if *a == 0 { *a } else { Mersenne64::PRIME - a }
    }
}

impl Ring for FiniteField<Mersenne64> {
    #[inline]
    fn zero(&self) -> Self::Element {
        0
    }

    /// Return the unit element in Montgomory form.
    #[inline]
    fn one(&self) -> Self::Element {
        1
    }

    #[inline]
    fn nth(&self, n: Integer) -> Self::Element {
        self.to_element(Mersenne64(n.to_finite_field(self)))
    }

    /// Compute b^e % n.
    #[inline]
    fn pow(&self, b: &Self::Element, mut e: u64) -> Self::Element {
        if e >= self.get_prime().0 - 1 {
            e %= self.get_prime().0 - 1;
        }

        if e == 0 {
            return self.one();
        }

        let mut x = *b;
        let mut y = self.one();
        while e != 1 {
            if e % 2 == 1 {
                y = self.mul(&y, &x);
            }

            x = self.mul(&x, &x);
            e /= 2;
        }

        self.mul(&x, &y)
    }

    #[inline]
    fn is_zero(&self, a: &Self::Element) -> bool {
        *a == 0
    }

    #[inline]
    fn is_one(&self, a: &Self::Element) -> bool {
        *a == 1
    }

    fn one_is_gcd_unit() -> bool {
        true
    }

    fn characteristic(&self) -> Integer {
        Mersenne64::PRIME.into()
    }

    fn try_inv(&self, a: &Self::Element) -> Option<Self::Element> {
        if *a == 0 { None } else { Some(self.inv(a)) }
    }

    fn try_div(&self, a: &Self::Element, b: &Self::Element) -> Option<Self::Element> {
        if self.is_zero(b) {
            None
        } else {
            Some(self.div(a, b))
        }
    }

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        let r = rng.random_range(
            range.0.max(0)..range.1.min(Mersenne64::PRIME.min(i64::MAX as u64) as i64),
        );
        r as u64
    }

    fn format<W: std::fmt::Write>(
        &self,
        element: &Self::Element,
        opts: &PrintOptions,
        state: PrintState,
        f: &mut W,
    ) -> Result<bool, Error> {
        if opts.symmetric_representation_for_finite_field {
            Z.format(&self.to_symmetric_integer(element), opts, state, f)
        } else {
            Z.format(&self.from_element(element).0.into(), opts, state, f)
        }
    }
}

impl EuclideanDomain for FiniteField<Mersenne64> {
    #[inline]
    fn rem(&self, _: &Self::Element, _: &Self::Element) -> Self::Element {
        0
    }

    #[inline]
    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        (self.mul(a, &self.inv(b)), 0)
    }

    #[inline]
    fn gcd(&self, _: &Self::Element, _: &Self::Element) -> Self::Element {
        1
    }
}

impl Field for FiniteField<Mersenne64> {
    #[inline]
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.mul(a, &self.inv(b))
    }

    #[inline]
    fn div_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(*a, self.inv(b));
    }

    /// Computes x^-1 mod n.
    fn inv(&self, a: &Self::Element) -> Self::Element {
        assert!(*a != 0, "0 is not invertible");

        // extended Euclidean algorithm: a x + b p = gcd(x, p) = 1 or a x = 1 (mod p)
        let mut u1: u64 = 1;
        let mut u3 = *a;
        let mut v1: u64 = 0;
        let mut v3 = Mersenne64::PRIME;
        let mut even_iter: bool = true;

        while v3 != 0 {
            let q = u3 / v3;
            let t3 = u3 % v3;
            let t1 = u1 + q * v1;
            u1 = v1;
            v1 = t1;
            u3 = v3;
            v3 = t3;
            even_iter = !even_iter;
        }

        debug_assert!(u3 == 1);
        if even_iter {
            u1
        } else {
            Mersenne64::PRIME - u1
        }
    }
}

impl FiniteFieldWorkspace for Integer {
    fn get_large_prime() -> Integer {
        Integer::Double(85070591730234615865843651857942052871)
    }

    fn try_from_integer(n: Integer) -> Option<Self> {
        Some(n)
    }

    fn to_integer(&self) -> Integer {
        self.clone()
    }
}

/// A finite field with a large prime modulus.
/// We use the symmetric representation, as this is the most efficient.
impl FiniteFieldCore<Integer> for FiniteField<Integer> {
    fn new(m: Integer) -> FiniteField<Integer> {
        FiniteField {
            p: m.clone(),
            m: Integer::one(),
            one: FiniteFieldElement(Integer::one()),
            is_prime: true,
        }
    }

    #[inline]
    fn get_prime(&self) -> Integer {
        self.p.clone()
    }

    fn to_element(&self, a: Integer) -> Integer {
        a.symmetric_mod(&self.p)
    }

    fn from_element(&self, a: &Integer) -> Integer {
        if a.is_negative() {
            a.clone() + &self.p
        } else {
            a.clone()
        }
    }

    fn to_integer(&self, a: &Self::Element) -> Integer {
        self.from_element(a).into()
    }
}

impl FiniteField<Integer> {
    #[inline(always)]
    fn normalize(&self, mut c: Integer) -> Integer {
        self.normalize_mut(&mut c);
        c
    }

    #[inline(always)]
    fn normalize_mut(&self, c: &mut Integer) {
        let two_c = &*c + &*c;

        if two_c.is_negative() {
            if -two_c >= self.p {
                *c += &self.p;
            }
        } else if two_c >= self.p {
            *c -= &self.p;
        }
    }
}

impl Set for FiniteField<Integer> {
    type Element = Integer;

    fn size(&self) -> Option<Integer> {
        Some(self.get_prime())
    }
}

impl RingOps<Integer> for FiniteField<Integer> {
    fn add(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        self.normalize(a + b)
    }

    fn sub(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        self.normalize(a - b)
    }

    fn mul(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        (a * b).symmetric_mod(&self.p)
    }

    fn add_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a += b;
        self.normalize_mut(a);
    }

    fn sub_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a -= b;
        self.normalize_mut(a);
    }

    fn mul_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a *= b;
        self.normalize_mut(a);
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: Self::Element, c: Self::Element) {
        *a += b * c;
        self.normalize_mut(a);
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: Self::Element, c: Self::Element) {
        *a -= b * c;
        self.normalize_mut(a);
    }

    fn neg(&self, a: Self::Element) -> Self::Element {
        a.neg()
    }
}

impl RingOps<&Integer> for FiniteField<Integer> {
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.normalize(a + b)
    }

    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.normalize(a - b)
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        (a * b).symmetric_mod(&self.p)
    }

    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a += b;
        self.normalize_mut(a);
    }

    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a -= b;
        self.normalize_mut(a);
    }

    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(&*a, b);
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.add_assign(a, &self.mul(b, c));
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.sub_assign(a, &self.mul(b, c));
    }

    fn neg(&self, a: &Self::Element) -> Self::Element {
        a.neg()
    }
}

impl Ring for FiniteField<Integer> {
    fn zero(&self) -> Self::Element {
        Integer::zero()
    }

    fn one(&self) -> Self::Element {
        Integer::one()
    }

    #[inline]
    fn nth(&self, n: Integer) -> Self::Element {
        n.symmetric_mod(&self.p)
    }

    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element {
        // FIXME: intermediate mods
        b.pow(e).symmetric_mod(&self.p)
    }

    fn is_zero(&self, a: &Self::Element) -> bool {
        a.is_zero()
    }

    fn is_one(&self, a: &Self::Element) -> bool {
        a.is_one()
    }

    fn one_is_gcd_unit() -> bool {
        true
    }

    fn characteristic(&self) -> Integer {
        self.get_prime()
    }

    fn try_inv(&self, a: &Self::Element) -> Option<Self::Element> {
        if a.is_zero() {
            return None;
        }

        let mut u1 = Integer::one();
        let mut u3 = a.clone();
        let mut v1 = Integer::zero();
        let mut v3 = self.get_prime();
        let mut even_iter: bool = true;

        while !v3.is_zero() {
            let (q, t3) = Z.quot_rem(&u3, &v3);
            let t1 = &u1 + &(&q * &v1);
            u1 = v1;
            v1 = t1;
            u3 = v3;
            v3 = t3;
            even_iter = !even_iter;
        }

        if !u3.is_one() {
            return None;
        }

        if even_iter {
            Some(u1)
        } else {
            Some(&self.p - &u1)
        }
    }

    fn try_div(&self, a: &Self::Element, b: &Self::Element) -> Option<Self::Element> {
        self.try_inv(b).map(|r| self.mul(a, &r))
    }

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        Z.sample(rng, range).symmetric_mod(&self.p)
    }

    fn format<W: std::fmt::Write>(
        &self,
        element: &Self::Element,
        opts: &PrintOptions,
        state: PrintState,
        f: &mut W,
    ) -> Result<bool, Error> {
        if opts.symmetric_representation_for_finite_field {
            Z.format(&self.to_symmetric_integer(element), opts, state, f)
        } else {
            Z.format(&self.from_element(element), opts, state, f)
        }
    }
}

impl EuclideanDomain for FiniteField<Integer> {
    fn rem(&self, _: &Self::Element, _: &Self::Element) -> Self::Element {
        Integer::zero()
    }

    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        (self.mul(a, &self.inv(b)), Integer::zero())
    }

    fn gcd(&self, _: &Self::Element, _: &Self::Element) -> Self::Element {
        Integer::one()
    }
}

impl Field for FiniteField<Integer> {
    #[inline]
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.mul(a, &self.inv(b))
    }

    #[inline]
    fn div_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(&*a, &self.inv(b));
    }

    /// Compute the inverse when `a` and the modulus are coprime,
    /// otherwise panic.
    fn inv(&self, a: &Self::Element) -> Self::Element {
        if let Some(r) = self.try_inv(a) {
            r
        } else {
            panic!("{} is not invertible mod {}", a, self.p);
        }
    }
}

/// Do a deterministic Miller test to check if `n` is a prime.
/// Since `n` is a `u64`, a basis of only 7 witnesses has to be tested.
///
/// Based on [Wojciech Izykowski's implementation](https://github.com/wizykowski/miller-rabin).
pub fn is_prime_u64(n: u64) -> bool {
    let w = if n < 341531 {
        [9345883071009581737].as_slice()
    } else if n < 1050535501 {
        [336781006125, 9639812373923155].as_slice()
    } else if n < 350269456337 {
        [
            4230279247111683200,
            14694767155120705706,
            16641139526367750375,
        ]
        .as_slice()
    } else {
        // shortest SPRP basis from Jim Sinclair for testing primality of u64
        [2, 325, 9375, 28178, 450775, 9780504, 1795265022].as_slice()
    };

    if n < 2 {
        return false;
    }

    if n.is_multiple_of(2) {
        return n == 2;
    }

    let mut s = 0;
    let mut d = n - 1;
    while d.is_multiple_of(2) {
        d /= 2;
        s += 1;
    }

    let f = Zp64::new(n);
    let neg_one = FiniteFieldElement(n.wrapping_sub(f.one().0));

    'test: for a in w {
        let a = f.to_element(*a);

        if a.0 == 0 {
            continue;
        }

        let mut x = f.pow(&a, d);

        if x == f.one() || x == neg_one {
            continue;
        }

        for _ in 0..s {
            x = f.mul(&x, &x);

            if x == f.one() {
                return false;
            }
            if x == neg_one {
                continue 'test;
            }
        }

        return false;
    }

    true
}

/// An iterator over consecutive 64-bit primes.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq)]
pub struct PrimeIteratorU64 {
    current_number: u64,
}

impl PrimeIteratorU64 {
    /// Create a new prime iterator that is larger than `start`.
    pub fn new(start: u64) -> PrimeIteratorU64 {
        PrimeIteratorU64 {
            current_number: start.max(1),
        }
    }
}

impl Iterator for PrimeIteratorU64 {
    type Item = u64;

    /// Yield the next prime or `None` if `u64::MAX` has been reached.
    fn next(&mut self) -> Option<u64> {
        while self.current_number < u64::MAX {
            self.current_number += 1;

            if is_prime_u64(self.current_number) {
                return Some(self.current_number);
            }
        }

        None
    }
}

/// An iterator that generates smooth primes based on a list of small primes `primes`.
/// A prime `p` is smooth if `p-1`'s prime factors are all in `primes`.
///
/// The output of the iterator is not sequential.
///
/// # Example
/// ```rust
/// # use numerica::domains::finite_field::{SmoothPrimeIterator};
/// let mut iter = SmoothPrimeIterator::new(vec![2, 3]);
///
/// while let Some((p, pows)) = iter.next() {
///     if p > 2 << 60 {
///         assert_eq!(p, 16210220612075905069);
///         assert_eq!(pows, [2, 39]);
///         break;
///     }
/// }
/// ```
pub struct SmoothPrimeIterator {
    primes: Vec<u64>,
    pows: Vec<u64>,
    accum: Vec<u64>,
    first: bool,
    done: bool,
}

impl SmoothPrimeIterator {
    /// Create a new smooth prime iterator with the given small primes.
    pub fn new(primes: Vec<u64>) -> Self {
        let len = primes.len();
        Self {
            primes,
            pows: vec![0; len],
            accum: vec![0; len],
            first: true,
            done: false,
        }
    }

    pub fn get_prime_list(&self) -> &[u64] {
        &self.primes
    }

    /// Returns the next (non-sequential) smooth prime and the powers of the small primes used to generate it.
    pub fn next(&mut self) -> Option<(u64, &[u64])> {
        if self.done {
            return None;
        }

        let mut skip_first = !self.first;
        self.next_impl(0, &mut skip_first, 0)
            .map(|p| (p, self.pows.as_slice()))
    }

    /// Returns the next (non-sequential) smooth prime and the powers of the small primes used to generate it.
    pub fn next_above(&mut self, lower_bound: u64) -> Option<(u64, &[u64])> {
        if self.done {
            return None;
        }

        let mut skip_first = !self.first;
        self.next_impl(0, &mut skip_first, lower_bound)
            .map(|p| (p, self.pows.as_slice()))
    }

    fn next_impl(&mut self, pos: usize, skip_first: &mut bool, lower_bound: u64) -> Option<u64> {
        if pos == self.primes.len() {
            let n = *self.accum.last().unwrap();
            if n >= lower_bound && n < u64::MAX - 1 && is_prime_u64(n + 1) {
                if !*skip_first {
                    self.first = false;
                    return Some(n + 1);
                } else {
                    *skip_first = false;
                    return None;
                }
            }

            return None;
        }

        if self.pows[pos] == 0 {
            if pos > 0 {
                self.accum[pos] = self.accum[pos - 1];
            } else {
                self.accum[0] = 1;
            }
        }

        if let Some(p) = self.next_impl(pos + 1, skip_first, lower_bound) {
            return Some(p);
        }

        for _ in self.pows[pos]..64 {
            let (r, overflow) = self.accum[pos].overflowing_mul(self.primes[pos]);
            if overflow {
                break;
            }

            self.pows[pos] += 1;
            self.accum[pos] = r;
            if let Some(p) = self.next_impl(pos + 1, skip_first, lower_bound) {
                return Some(p);
            }
        }

        if pos > 0 {
            self.pows[pos] = 0; // reset
        } else {
            self.done = true;
        }

        None
    }
}

/// Perform Pollard's rho algorithm with Brent's cycle detection.
fn pollard_brent_rho(n: u64) -> u64 {
    const M: u64 = 1000;

    if is_prime_u64(n) {
        return n;
    }

    if n.is_multiple_of(2) {
        return 2;
    }

    let field = Zp64::new(n);
    let mut rng = rand::rng();

    let mut c = 3;

    loop {
        let cf = field.to_element(c);
        let mut x = field.sample(&mut rng, (0, n.clamp(0, i64::MAX as u64) as i64));

        let mut y = x;
        let mut q = field.one();
        let mut ys = field.one();
        let mut r = 1;
        let mut g = 1;

        while g == 1 {
            x = y;

            for _ in 1..r {
                y = field.add(&field.mul(&y, &y), &cf);
            }

            let mut k = 0;
            while k < r && g == 1 {
                ys = y;
                for _ in 1..M.min(r - k) {
                    y = field.add(&field.mul(&y, &y), &cf);
                    field.mul_assign(&mut q, &field.sub(&x, &y));
                }

                g = gcd_unsigned(field.from_element(&q), n);
                k += M;
            }

            r *= 2;
        }

        if g == n {
            loop {
                ys = field.add(&field.mul(&ys, &ys), &cf);
                g = gcd_unsigned(field.from_element(&field.sub(&x, &ys)), n);
                if g > 1 {
                    if g == n {
                        // two sequences are repeating at the same time, increase constant
                        c += 1;
                        break;
                    }
                    return g;
                }
            }
        }

        if g != n {
            return g;
        }
    }
}

/// Factorize a 64-bit number into its prime factors.
pub fn factor(mut n: u64, out: &mut Vec<u64>) {
    if n < 2 {
        out.push(n);
        return;
    }

    while n.is_multiple_of(2) {
        out.push(2);
        n /= 2;
    }

    while n > 1 {
        let f = pollard_brent_rho(n);

        if f == n {
            out.push(n);
        } else {
            factor(f, out);
        }

        n /= f;
    }
}

/// Compute the Euler totient function of `n`.
pub fn totient(n: u64) -> u64 {
    if is_prime_u64(n) {
        return n - 1;
    }

    let mut factors = Vec::new();
    factor(n, &mut factors);
    factors.sort();
    factors.dedup();

    let mut t = n;
    for f in factors {
        t = t - t / f;
    }

    t
}

/// An iterator over the primitive roots of a finite field with odd (potentially composite) modulus `n`.
pub struct PrimitiveRootIterator {
    f: Zp64,
    totient: u64,
    totient_factors: Vec<u64>,
    current: u64,
}

impl PrimitiveRootIterator {
    pub fn new(n: u64) -> Self {
        let totient = totient(n);
        let mut factors = vec![];

        factor(totient, &mut factors);
        factors.sort();
        factors.dedup();

        Self::new_with_totient_factors(n, totient, factors)
    }

    pub fn new_with_totient_factors(n: u64, totient: u64, totient_factors: Vec<u64>) -> Self {
        let field = Zp64::new(n);
        PrimitiveRootIterator {
            totient,
            totient_factors,
            f: field,
            current: 2,
        }
    }
}

impl Iterator for PrimitiveRootIterator {
    type Item = u64;

    fn next(&mut self) -> Option<u64> {
        'next: for c in self.current..self.f.get_prime() {
            for &f in &self.totient_factors {
                let r = self.f.pow(&self.f.to_element(c), self.totient / f);
                if self.f.is_one(&r) || self.f.is_zero(&r) {
                    self.current = c + 1;
                    continue 'next;
                }
            }

            self.current = c + 1;
            return Some(c);
        }

        None
    }
}

/// Small primes used for the smooth prime iterator.
pub const SMOOTH_PRIME_BASE: [u64; 4] = [2, 3, 5, 7];
/// A list of smooth primes with their primitive roots and powers of the [SMOOTH_PRIME_BASE].
pub const SMOOTH_PRIMES: [(u64, u8, [u8; 4]); 323] = [
    (577, 5, [6, 2, 0, 0]),
    (641, 3, [7, 0, 1, 0]),
    (769, 11, [8, 1, 0, 0]),
    (1153, 5, [7, 2, 0, 0]),
    (1297, 10, [4, 4, 0, 0]),
    (1373, 2, [2, 0, 0, 3]),
    (1459, 3, [1, 6, 0, 0]),
    (1601, 3, [6, 0, 2, 0]),
    (2593, 7, [5, 4, 0, 0]),
    (2917, 5, [2, 6, 0, 0]),
    (3137, 3, [6, 0, 0, 2]),
    (3457, 7, [7, 3, 0, 0]),
    (3889, 11, [4, 5, 0, 0]),
    (4001, 3, [5, 0, 3, 0]),
    (10369, 13, [7, 4, 0, 0]),
    (12289, 11, [12, 1, 0, 0]),
    (16001, 3, [7, 0, 3, 0]),
    (17497, 5, [3, 7, 0, 0]),
    (18433, 5, [11, 2, 0, 0]),
    (25601, 3, [10, 0, 2, 0]),
    (39367, 3, [1, 9, 0, 0]),
    (40961, 3, [13, 0, 1, 0]),
    (50177, 3, [10, 0, 0, 2]),
    (52489, 7, [3, 8, 0, 0]),
    (62501, 2, [2, 0, 6, 0]),
    (65537, 3, [16, 0, 0, 0]),
    (114689, 3, [14, 0, 0, 1]),
    (139969, 13, [6, 7, 0, 0]),
    (147457, 10, [14, 2, 0, 0]),
    (160001, 3, [8, 0, 4, 0]),
    (163841, 3, [15, 0, 1, 0]),
    (209953, 10, [5, 8, 0, 0]),
    (268913, 3, [4, 0, 0, 5]),
    (331777, 5, [12, 4, 0, 0]),
    (470597, 2, [2, 0, 0, 6]),
    (472393, 5, [3, 10, 0, 0]),
    (614657, 3, [8, 0, 0, 4]),
    (629857, 5, [5, 9, 0, 0]),
    (746497, 5, [10, 6, 0, 0]),
    (786433, 10, [18, 1, 0, 0]),
    (839809, 7, [7, 8, 0, 0]),
    (995329, 7, [12, 5, 0, 0]),
    (1075649, 6, [6, 0, 0, 5]),
    (1179649, 19, [17, 2, 0, 0]),
    (1492993, 7, [11, 6, 0, 0]),
    (1769473, 5, [16, 3, 0, 0]),
    (1990657, 5, [13, 5, 0, 0]),
    (2654209, 11, [15, 4, 0, 0]),
    (3294173, 3, [2, 0, 0, 7]),
    (5038849, 29, [8, 9, 0, 0]),
    (5308417, 5, [16, 4, 0, 0]),
    (7340033, 3, [20, 0, 0, 1]),
    (8503057, 5, [4, 12, 0, 0]),
    (9834497, 3, [12, 0, 0, 4]),
    (11337409, 7, [6, 11, 0, 0]),
    (14155777, 7, [19, 3, 0, 0]),
    (16384001, 3, [17, 0, 3, 0]),
    (19131877, 5, [2, 14, 0, 0]),
    (26214401, 3, [20, 0, 2, 0]),
    (28311553, 5, [20, 3, 0, 0]),
    (40960001, 3, [16, 0, 4, 0]),
    (57395629, 10, [2, 15, 0, 0]),
    (62500001, 6, [5, 0, 9, 0]),
    (63700993, 5, [18, 5, 0, 0]),
    (71663617, 5, [15, 7, 0, 0]),
    (86093443, 2, [1, 16, 0, 0]),
    (102036673, 5, [6, 13, 0, 0]),
    (104857601, 3, [22, 0, 2, 0]),
    (113246209, 7, [22, 3, 0, 0]),
    (120932353, 5, [11, 10, 0, 0]),
    (167772161, 3, [25, 0, 1, 0]),
    (169869313, 5, [21, 4, 0, 0]),
    (210827009, 3, [8, 0, 0, 7]),
    (256000001, 3, [14, 0, 6, 0]),
    (258280327, 5, [1, 17, 0, 0]),
    (275365889, 3, [14, 0, 0, 5]),
    (359661569, 3, [20, 0, 0, 3]),
    (409600001, 7, [17, 0, 5, 0]),
    (469762049, 3, [26, 0, 0, 1]),
    (483729409, 23, [13, 10, 0, 0]),
    (655360001, 3, [20, 0, 4, 0]),
    (725594113, 5, [12, 11, 0, 0]),
    (1088391169, 11, [11, 12, 0, 0]),
    (1129900997, 2, [2, 0, 0, 10]),
    (1438646273, 3, [22, 0, 0, 3]),
    (1811939329, 13, [26, 3, 0, 0]),
    (1927561217, 3, [14, 0, 0, 6]),
    (2441406251, 2, [1, 0, 13, 0]),
    (2500000001, 3, [8, 0, 10, 0]),
    (2717908993, 5, [25, 4, 0, 0]),
    (3221225473, 5, [30, 1, 0, 0]),
    (3439853569, 7, [19, 8, 0, 0]),
    (4076863489, 7, [24, 5, 0, 0]),
    (4194304001, 3, [25, 0, 3, 0]),
    (6879707137, 5, [20, 8, 0, 0]),
    (7909306973, 2, [2, 0, 0, 11]),
    (10485760001, 3, [24, 0, 4, 0]),
    (10871635969, 7, [27, 4, 0, 0]),
    (11609505793, 5, [16, 11, 0, 0]),
    (16384000001, 3, [20, 0, 6, 0]),
    (18345885697, 10, [23, 7, 0, 0]),
    (29386561537, 5, [11, 15, 0, 0]),
    (52613349377, 3, [30, 0, 0, 2]),
    (69657034753, 5, [17, 12, 0, 0]),
    (77309411329, 7, [33, 2, 0, 0]),
    (97656250001, 3, [4, 0, 14, 0]),
    (123834728449, 7, [21, 10, 0, 0]),
    (126548911553, 5, [6, 0, 0, 11]),
    (206158430209, 22, [36, 1, 0, 0]),
    (215886856193, 3, [18, 0, 0, 7]),
    (251048476873, 7, [3, 22, 0, 0]),
    (347892350977, 10, [32, 4, 0, 0]),
    (390625000001, 3, [6, 0, 14, 0]),
    (409600000001, 7, [20, 0, 8, 0]),
    (863547424769, 3, [20, 0, 0, 7]),
    (880602513409, 7, [27, 8, 0, 0]),
    (1253826625537, 5, [18, 14, 0, 0]),
    (1410554953729, 14, [15, 16, 0, 0]),
    (1677721600001, 3, [29, 0, 5, 0]),
    (1761205026817, 5, [28, 8, 0, 0]),
    (2024782584833, 3, [10, 0, 0, 11]),
    (2348273369089, 11, [30, 7, 0, 0]),
    (2380311484417, 5, [11, 19, 0, 0]),
    (2684354560001, 3, [32, 0, 4, 0]),
    (2748779069441, 3, [39, 0, 1, 0]),
    (2783138807809, 11, [35, 4, 0, 0]),
    (4294967296001, 3, [35, 0, 3, 0]),
    (4511594708993, 5, [28, 0, 0, 5]),
    (4518872583697, 5, [4, 24, 0, 0]),
    (4628074479617, 3, [14, 0, 0, 10]),
    (5566277615617, 5, [36, 4, 0, 0]),
    (6044831973377, 3, [20, 0, 0, 8]),
    (6347497291777, 5, [14, 18, 0, 0]),
    (6597069766657, 5, [41, 1, 0, 0]),
    (13816758796289, 3, [24, 0, 0, 7]),
    (14281868906497, 5, [12, 20, 0, 0]),
    (15258789062501, 2, [2, 0, 18, 0]),
    (17832200896513, 19, [25, 12, 0, 0]),
    (22568879259649, 11, [19, 16, 0, 0]),
    (25048249270273, 5, [35, 6, 0, 0]),
    (29686813949953, 5, [40, 3, 0, 0]),
    (33853318889473, 5, [18, 17, 0, 0]),
    (39582418599937, 5, [42, 2, 0, 0]),
    (40000000000001, 3, [15, 0, 13, 0]),
    (42949672960001, 6, [36, 0, 4, 0]),
    (44530220924929, 7, [39, 4, 0, 0]),
    (56358560858113, 7, [33, 8, 0, 0]),
    (62500000000001, 13, [11, 0, 15, 0]),
    (72185515343873, 3, [32, 0, 0, 5]),
    (79164837199873, 5, [43, 2, 0, 0]),
    (84537841287169, 11, [32, 9, 0, 0]),
    (85691213438977, 53, [13, 21, 0, 0]),
    (90275517038593, 10, [21, 16, 0, 0]),
    (99214346656769, 3, [10, 0, 0, 13]),
    (104857600000001, 3, [28, 0, 8, 0]),
    (126806761930753, 10, [31, 10, 0, 0]),
    (142657607172097, 5, [28, 12, 0, 0]),
    (150289495621633, 5, [36, 7, 0, 0]),
    (167772160000001, 3, [31, 0, 7, 0]),
    (215504279044097, 3, [42, 0, 0, 2]),
    (256000000000001, 7, [20, 0, 12, 0]),
    (270826551115777, 5, [21, 17, 0, 0]),
    (288742061375489, 6, [34, 0, 0, 5]),
    (409600000000001, 3, [23, 0, 11, 0]),
    (411782264189299, 3, [1, 30, 0, 0]),
    (457019805007873, 10, [17, 20, 0, 0]),
    (474989023199233, 5, [44, 3, 0, 0]),
    (578415690713089, 7, [11, 24, 0, 0]),
    (655360000000001, 3, [26, 0, 10, 0]),
    (930522055948829, 2, [2, 0, 0, 17]),
    (1141260857376769, 41, [31, 12, 0, 0]),
    (1711891286065153, 5, [30, 13, 0, 0]),
    (1717986918400001, 3, [39, 0, 5, 0]),
    (1899956092796929, 7, [46, 3, 0, 0]),
    (1952152956156673, 5, [8, 27, 0, 0]),
    (2621440000000001, 3, [28, 0, 10, 0]),
    (3084883683803137, 7, [15, 23, 0, 0]),
    (3249918613389313, 5, [23, 18, 0, 0]),
    (3799912185593857, 5, [47, 3, 0, 0]),
    (3851755393646593, 5, [28, 15, 0, 0]),
    (4113178245070849, 13, [17, 22, 0, 0]),
    (4194304000000001, 3, [31, 0, 9, 0]),
    (4294967296000001, 3, [38, 0, 6, 0]),
    (6349718186033153, 3, [16, 0, 0, 13]),
    (7312316880125953, 5, [21, 20, 0, 0]),
    (7808611824626689, 7, [10, 27, 0, 0]),
    (7881299347898369, 6, [50, 0, 0, 1]),
    (12173449145352193, 5, [36, 11, 0, 0]),
    (15258789062500001, 3, [5, 0, 21, 0]),
    (19499511680335873, 5, [24, 19, 0, 0]),
    (22799473113563137, 5, [48, 4, 0, 0]),
    (24136479252938753, 3, [46, 0, 0, 3]),
    (25398872744132609, 3, [18, 0, 0, 13]),
    (30399297484750849, 11, [50, 3, 0, 0]),
    (31525197391593473, 3, [52, 0, 0, 1]),
    (34665798542819329, 11, [28, 17, 0, 0]),
    (38146972656250001, 3, [4, 0, 22, 0]),
    (39531097362172609, 11, [6, 31, 0, 0]),
    (40960000000000001, 3, [25, 0, 13, 0]),
    (51298814505517057, 7, [46, 6, 0, 0]),
    (57711166318706689, 7, [43, 8, 0, 0]),
    (58498535041007617, 7, [24, 20, 0, 0]),
    (59296646043258913, 5, [5, 32, 0, 0]),
    (59553411580724993, 3, [8, 0, 0, 17]),
    (61628086298345473, 5, [32, 15, 0, 0]),
    (73040694872113153, 5, [37, 12, 0, 0]),
    (92442129447518209, 11, [31, 16, 0, 0]),
    (97387593162817537, 5, [39, 11, 0, 0]),
    (112589990684262401, 3, [52, 0, 2, 0]),
    (115422332637413377, 10, [44, 8, 0, 0]),
    (136796838681378817, 5, [49, 5, 0, 0]),
    (167772160000000001, 3, [34, 0, 10, 0]),
    (180143985094819841, 6, [55, 0, 1, 0]),
    (182382322965970289, 3, [4, 0, 0, 19]),
    (281110025686560769, 7, [12, 29, 0, 0]),
    (390625000000000001, 3, [12, 0, 20, 0]),
    (409600000000000001, 3, [26, 0, 14, 0]),
    (666334875701477377, 7, [18, 26, 0, 0]),
    (687194767360000001, 3, [43, 0, 7, 0]),
    (882705526964617217, 5, [54, 0, 0, 2]),
    (1048576000000000001, 3, [32, 0, 12, 0]),
    (1276676260761792017, 3, [4, 0, 0, 20]),
    (1314732507698036737, 11, [38, 14, 0, 0]),
    (1332669751402954753, 7, [19, 26, 0, 0]),
    (1667495524260299777, 5, [10, 0, 0, 18]),
    (1945555039024054273, 5, [56, 3, 0, 0]),
    (2401514164751985937, 5, [4, 36, 0, 0]),
    (2844673747342852097, 3, [22, 0, 0, 14]),
    (3906250000000000001, 3, [13, 0, 21, 0]),
    (4803028329503971873, 5, [5, 36, 0, 0]),
    (6566248256706183169, 7, [53, 6, 0, 0]),
    (9404839736113102849, 11, [23, 4, 0, 12]),
    (9428312208179200001, 12, [43, 0, 5, 3]),
    (9437184000000000001, 22, [32, 2, 12, 0]),
    (9440732714731831297, 5, [24, 14, 0, 6]),
    (9655952453613281251, 2, [1, 4, 24, 0]),
    (9674195232406634497, 5, [21, 23, 0, 2]),
    (9720000000000000001, 19, [18, 5, 16, 0]),
    (9806581127812500001, 7, [5, 22, 10, 0]),
    (9897465023658196993, 5, [42, 8, 0, 3]),
    (10030613004288000001, 7, [27, 14, 6, 0]),
    (10155995666841600001, 11, [23, 18, 5, 0]),
    (10644187350545989633, 5, [46, 2, 0, 5]),
    (10672683797745543169, 13, [11, 17, 0, 9]),
    (10673396287786604161, 7, [7, 34, 1, 0]),
    (10696160921593577473, 10, [36, 3, 0, 8]),
    (10728836059570312501, 2, [2, 2, 25, 0]),
    (10765727022134826757, 14, [2, 28, 0, 6]),
    (10808639105689190401, 7, [57, 1, 2, 0]),
    (10851569165584000001, 3, [10, 0, 6, 14]),
    (10883911680000000001, 7, [21, 12, 10, 0]),
    (10937500000000000001, 3, [14, 0, 20, 1]),
    (10972313025465286657, 5, [22, 3, 0, 13]),
    (11007531417600000001, 14, [32, 8, 8, 0]),
    (11255594788757023489, 13, [8, 3, 0, 18]),
    (11298551057489880577, 5, [9, 13, 0, 12]),
    (11438396227480500001, 19, [5, 28, 6, 0]),
    (11465712868037492737, 10, [26, 20, 0, 2]),
    (11542233263741337601, 14, [46, 8, 2, 0]),
    (11568313814261760001, 7, [16, 24, 4, 0]),
    (11570186199040000001, 3, [19, 0, 7, 10]),
    (11664000000000000001, 7, [19, 6, 15, 0]),
    (11682510375976562501, 2, [2, 0, 24, 2]),
    (11704581886121607169, 17, [22, 19, 0, 4]),
    (11864270595822053977, 5, [3, 31, 0, 4]),
    (11914996070292187501, 7, [2, 27, 8, 0]),
    (11961120409626632833, 5, [7, 9, 0, 15]),
    (12050326889856000001, 31, [13, 23, 6, 0]),
    (12093235200000000001, 13, [22, 10, 11, 0]),
    (12150979463747690497, 11, [13, 7, 0, 14]),
    (12224183910392659969, 34, [39, 3, 0, 7]),
    (12610078956637388801, 3, [56, 0, 2, 1]),
    (12649106723253977089, 19, [16, 14, 0, 9]),
    (12666373951979520001, 14, [51, 2, 4, 0]),
    (12839184645488640001, 11, [32, 14, 4, 0]),
    (12914016300000000001, 7, [11, 17, 11, 0]),
    (12926347140213144163, 11, [1, 4, 0, 20]),
    (12961910018419826689, 17, [13, 23, 0, 5]),
    (13060694016000000001, 17, [22, 13, 9, 0]),
    (13084411621093750001, 3, [4, 0, 22, 3]),
    (13136816711425781251, 2, [1, 16, 16, 0]),
    (13238884522546875001, 14, [3, 25, 9, 0]),
    (13293172227840400001, 3, [7, 0, 5, 16]),
    (13326697514029547521, 17, [20, 26, 1, 0]),
    (13374150672384000001, 11, [29, 13, 6, 0]),
    (13612208361308569601, 6, [14, 0, 2, 16]),
    (14007897484541025409, 17, [7, 18, 0, 10]),
    (14187762048339843751, 7, [1, 19, 14, 0]),
    (14238281250000000001, 11, [10, 6, 19, 0]),
    (14245362154520607853, 11, [2, 7, 0, 18]),
    (14305114746093750001, 7, [4, 1, 25, 0]),
    (14331184359791394817, 10, [28, 3, 0, 11]),
    (14401160845923188737, 5, [18, 4, 0, 14]),
    (14456121969433116673, 10, [19, 14, 0, 8]),
    (14487921671576485889, 3, [44, 0, 0, 7]),
    (14676708556800000001, 13, [34, 7, 8, 0]),
    (14831542968750000001, 7, [7, 5, 21, 0]),
    (15000000000000000001, 17, [18, 1, 19, 0]),
    (15045919506432000001, 23, [26, 15, 6, 0]),
    (15064734743319840769, 13, [11, 12, 0, 12]),
    (15564440312192434177, 5, [59, 3, 0, 0]),
    (15690529804500000001, 7, [8, 22, 9, 0]),
    (15742161438447403009, 11, [15, 29, 0, 1]),
    (15853825200249372673, 5, [30, 16, 0, 3]),
    (15925248000000000001, 11, [28, 5, 12, 0]),
    (15966281025818984449, 23, [45, 3, 0, 5]),
    (16000000000000000001, 3, [22, 0, 18, 0]),
    (16210220612075905069, 10, [2, 39, 0, 0]),
    (16342547065297862977, 5, [6, 17, 0, 11]),
    (16529940864000000001, 23, [16, 17, 9, 0]),
    (16584334684125659137, 5, [23, 24, 0, 1]),
    (16601952574270844929, 13, [12, 15, 0, 10]),
    (16815125390625000001, 11, [6, 16, 14, 0]),
    (16945772188860000001, 7, [8, 25, 7, 0]),
    (17006112000000000001, 7, [17, 12, 12, 0]),
    (17085937500000000001, 7, [11, 7, 18, 0]),
    (17133181593402212353, 10, [24, 11, 0, 8]),
    (17556872829182410753, 5, [21, 20, 0, 4]),
    (17623416832000000001, 11, [29, 0, 9, 5]),
    (17631936921600000001, 11, [20, 16, 8, 0]),
    (17642599747293652993, 5, [11, 21, 0, 7]),
    (18078415936000000001, 3, [15, 0, 9, 10]),
    (18336275865588989953, 5, [38, 4, 0, 7]),
];

#[cfg(test)]
mod test {
    use super::{FiniteFieldCore, Zp};
    use crate::domains::{
        Ring, RingOps,
        finite_field::{PrimitiveRootIterator, Zp64},
    };

    #[test]
    fn primitive_root() {
        let roots: Vec<_> = PrimitiveRootIterator::new(23).collect();
        assert_eq!(roots, [5, 7, 10, 11, 14, 15, 17, 19, 20, 21]);
    }

    #[test]
    fn pow() {
        let field = Zp::new(31);

        let mut q = field.one();
        let x = field.to_element(3);
        for i in 0..100 {
            let r = field.pow(&x, i);
            assert_eq!(r, q);
            q = field.mul(q, x);
        }
    }

    #[test]
    fn non_prime() {
        let field = Zp::new_non_prime(27);
        let x = field.to_element(13);
        let y = field.to_element(5);
        let r = field.mul(&x, &y);
        assert_eq!(field.from_element(&r), 11);
    }

    #[test]
    fn factor() {
        let mut factors = Vec::new();
        for i in 18446744073709541426..18446744073709551426 {
            super::factor(i, &mut factors);
            let mut res = 1;
            for f in &factors {
                res *= f;
            }

            assert_eq!(res, i);
            factors.clear();
        }
    }

    #[test]
    fn discrete_log() {
        let field = Zp64::new(73);
        let base = field.to_element(5);
        let y = field.to_element(11);
        let log = field.discrete_log(&base, &y, 72, &[(2, 3), (3, 2)]);
        assert_eq!(field.from_element(&log), 55);
    }
}
