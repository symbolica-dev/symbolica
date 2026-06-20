#[cfg(feature = "gmp")]
pub(crate) use rug::Complete;
#[cfg(feature = "gmp")]
pub use rug::Integer as MultiPrecisionInteger;
#[cfg(feature = "gmp")]
pub(crate) use rug::ops::RemRounding;

#[cfg(feature = "gmp")]
pub(crate) fn pow_ref_u32(base: &MultiPrecisionInteger, e: u32) -> MultiPrecisionInteger {
    use rug::ops::Pow;

    base.pow(e).into()
}

#[cfg(feature = "gmp")]
pub(crate) fn probably_prime(value: &MultiPrecisionInteger, reps: u32) -> Option<bool> {
    Some(value.is_probably_prime(reps) != rug::integer::IsPrime::No)
}

#[cfg(feature = "gmp")]
pub fn from_lsf_bytes(bytes: &[u8]) -> MultiPrecisionInteger {
    MultiPrecisionInteger::from_digits(bytes, rug::integer::Order::Lsf)
}

#[cfg(feature = "gmp")]
pub fn write_lsf_bytes(value: &MultiPrecisionInteger, dest: &mut Vec<u8>) {
    let value = value.as_abs();
    let num_digits = value.significant_digits::<u8>();
    let old_len = dest.len();
    dest.resize(old_len + num_digits, 0);
    value.write_digits(&mut dest[old_len..], rug::integer::Order::Lsf);
}

#[cfg(feature = "gmp")]
pub fn lsf_byte_size(value: &MultiPrecisionInteger) -> usize {
    value.significant_digits::<u8>()
}

#[cfg(feature = "gmp")]
pub fn to_lsf_bytes(value: &MultiPrecisionInteger) -> Vec<u8> {
    let mut bytes = Vec::new();
    write_lsf_bytes(value, &mut bytes);
    bytes
}

#[cfg(feature = "gmp")]
pub fn from_digits_radix(digits: &[u8], radix: u32, is_negative: bool) -> MultiPrecisionInteger {
    let mut value = MultiPrecisionInteger::new();
    unsafe {
        value.assign_bytes_radix_unchecked(
            digits,
            i32::try_from(radix).expect("radix does not fit in i32"),
            is_negative,
        );
    }
    value
}

#[cfg(all(feature = "gmp", feature = "bincode"))]
pub(crate) fn to_be_bytes(value: &MultiPrecisionInteger) -> Vec<u8> {
    value.to_digits::<u8>(rug::integer::Order::MsfBe)
}

#[cfg(all(feature = "gmp", feature = "bincode"))]
pub(crate) fn from_be_bytes(bytes: &[u8]) -> Result<MultiPrecisionInteger, &'static str> {
    Ok(MultiPrecisionInteger::from_digits(
        bytes,
        rug::integer::Order::MsfBe,
    ))
}

#[cfg(feature = "gmp")]
pub(crate) struct BackendRandState(rug::rand::RandState<'static>);

#[cfg(feature = "gmp")]
impl BackendRandState {
    pub(crate) fn new(seed: u128) -> Self {
        let mut state = rug::rand::RandState::new();
        state.seed(&MultiPrecisionInteger::from(seed));
        Self(state)
    }

    pub(crate) fn below(&mut self, modulus: &MultiPrecisionInteger) -> MultiPrecisionInteger {
        modulus.clone().random_below(&mut self.0)
    }
}

#[cfg(feature = "no_gmp")]
mod malachite {
    use std::{
        fmt::{Debug, Display, Formatter, UpperHex},
        ops::{
            Add, AddAssign, BitAnd, BitAndAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem,
            RemAssign, Shl, Shr, Sub, SubAssign,
        },
        str::FromStr,
    };

    use malachite_base::num::{
        arithmetic::traits::{
            Abs, DivMod, ExtendedGcd, FloorRoot, Gcd, Mod, Pow as MalachitePow, UnsignedAbs,
        },
        logic::traits::SignificantBits,
    };
    use malachite_nz::integer::Integer as MalachiteInteger;

    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
    pub struct MultiPrecisionInteger(MalachiteInteger);

    #[allow(dead_code)]
    pub(crate) trait RemRounding {}

    impl RemRounding for MultiPrecisionInteger {}

    pub(crate) trait Complete {
        fn complete(self) -> Self;
    }

    impl<T> Complete for T {
        #[inline]
        fn complete(self) -> Self {
            self
        }
    }

    impl MultiPrecisionInteger {
        #[inline]
        pub fn factorial(n: u32) -> Self {
            let mut f = Self::from(1);
            for x in 2..=n {
                f *= Self::from(x);
            }
            f
        }

        #[inline]
        pub fn from_f64(f: f64) -> Option<Self> {
            f.is_finite()
                .then(|| f.trunc().to_string().parse::<Self>().ok())
                .flatten()
        }

        #[inline]
        pub fn to_i64(&self) -> Option<i64> {
            i64::try_from(&self.0).ok()
        }

        #[inline]
        pub fn to_i128(&self) -> Option<i128> {
            i128::try_from(&self.0).ok()
        }

        #[inline]
        pub fn to_u64(&self) -> Option<u64> {
            u64::try_from(&self.0).ok()
        }

        pub fn to_usize(&self) -> Option<usize> {
            self.to_u64().and_then(|x| usize::try_from(x).ok())
        }

        #[inline]
        pub fn to_u128(&self) -> Option<u128> {
            u128::try_from(&self.0).ok()
        }

        #[inline]
        pub fn mod_u(&self, modulus: u32) -> u32 {
            self.rem_euc(Self::from(modulus)).to_u64().unwrap() as u32
        }

        #[inline]
        pub fn rem_euc<T: Into<Self>>(&self, rhs: T) -> Self {
            Self(self.0.clone().mod_op(rhs.into().0.abs()))
        }

        #[inline]
        pub fn div_rem_euc<T: Into<Self>>(self, rhs: T) -> (Self, Self) {
            let rhs = rhs.into();
            let r = self.rem_euc(rhs.clone());
            let q = (self.0 - r.0.clone()) / rhs.0;
            (Self(q), r)
        }

        #[inline]
        pub fn div_rem_ref(&self, rhs: &Self) -> (Self, Self) {
            let (q, r) = self.0.clone().div_mod(rhs.0.clone());
            (Self(q), Self(r))
        }

        #[inline]
        pub fn root_ref(&self, e: u32) -> Self {
            Self(self.0.clone().floor_root(u64::from(e)))
        }

        #[inline]
        pub fn signum_ref(&self) -> i8 {
            if self.0 > 0 {
                1
            } else if self.0 < 0 {
                -1
            } else {
                0
            }
        }

        #[inline]
        pub fn gcd(&self, rhs: &Self) -> Self {
            Self(MalachiteInteger::from(Gcd::gcd(
                self.0.clone().unsigned_abs(),
                rhs.0.clone().unsigned_abs(),
            )))
        }

        #[inline]
        pub fn extended_gcd(self, rhs: Self, _scratch: Self) -> (Self, Self, Self) {
            let (g, s, t) = ExtendedGcd::extended_gcd(self.0, rhs.0);
            (Self(MalachiteInteger::from(g)), Self(s), Self(t))
        }

        #[inline]
        pub fn invert(&self, modulus: &Self) -> Result<Self, ()> {
            let (g, s, _) = ExtendedGcd::extended_gcd(self.0.clone(), modulus.0.clone());
            if g != 1u32 {
                return Err(());
            }
            Ok(Self(s.mod_op(modulus.0.clone())))
        }

        #[inline]
        pub fn significant_bits(&self) -> u64 {
            SignificantBits::significant_bits(&self.0)
        }

        #[inline]
        pub fn pow(self, e: u32) -> Self {
            Self(MalachitePow::pow(self.0, u64::from(e)))
        }

        #[inline]
        pub fn abs(self) -> Self {
            Self(self.0.abs())
        }

        #[inline]
        pub fn is_negative(&self) -> bool {
            self.0 < 0
        }

        #[inline]
        pub fn as_abs(&self) -> Self {
            Self(self.0.clone().abs())
        }

        #[inline]
        pub fn is_zero(&self) -> bool {
            self.0 == 0
        }

        fn rem_trunc(lhs: MalachiteInteger, rhs: MalachiteInteger) -> Self {
            let q = lhs.clone() / rhs.clone();
            Self(lhs - q * rhs)
        }
    }

    impl Display for MultiPrecisionInteger {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            Display::fmt(&self.0, f)
        }
    }

    impl Debug for MultiPrecisionInteger {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            Debug::fmt(&self.0, f)
        }
    }

    impl UpperHex for MultiPrecisionInteger {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            UpperHex::fmt(&self.0, f)
        }
    }

    impl FromStr for MultiPrecisionInteger {
        type Err = <MalachiteInteger as FromStr>::Err;

        fn from_str(s: &str) -> Result<Self, Self::Err> {
            MalachiteInteger::from_str(s).map(Self)
        }
    }

    macro_rules! impl_from_primitive {
        ($($t:ty),* $(,)?) => {
            $(
                impl From<$t> for MultiPrecisionInteger {
                    #[inline]
                    fn from(value: $t) -> Self {
                        Self(MalachiteInteger::from(value))
                    }
                }
            )*
        };
    }

    impl_from_primitive!(
        i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize
    );

    impl From<MalachiteInteger> for MultiPrecisionInteger {
        #[inline]
        fn from(value: MalachiteInteger) -> Self {
            Self(value)
        }
    }

    impl From<MultiPrecisionInteger> for MalachiteInteger {
        #[inline]
        fn from(value: MultiPrecisionInteger) -> Self {
            value.0
        }
    }

    macro_rules! impl_cmp_primitive {
        ($($t:ty),* $(,)?) => {
            $(
                impl PartialEq<$t> for MultiPrecisionInteger {
                    #[inline]
                    fn eq(&self, other: &$t) -> bool {
                        self.0 == MalachiteInteger::from(*other)
                    }
                }

                impl PartialEq<MultiPrecisionInteger> for $t {
                    #[inline]
                    fn eq(&self, other: &MultiPrecisionInteger) -> bool {
                        MalachiteInteger::from(*self) == other.0
                    }
                }

                impl PartialOrd<$t> for MultiPrecisionInteger {
                    #[inline]
                    fn partial_cmp(&self, other: &$t) -> Option<std::cmp::Ordering> {
                        self.0.partial_cmp(&MalachiteInteger::from(*other))
                    }
                }

                impl PartialOrd<MultiPrecisionInteger> for $t {
                    #[inline]
                    fn partial_cmp(&self, other: &MultiPrecisionInteger) -> Option<std::cmp::Ordering> {
                        MalachiteInteger::from(*self).partial_cmp(&other.0)
                    }
                }
            )*
        };
    }

    impl_cmp_primitive!(i32, i64, i128, u32, u64, u128);

    macro_rules! impl_bin_op {
        ($trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident) => {
            impl $trait for MultiPrecisionInteger {
                type Output = Self;

                #[inline]
                fn $method(self, rhs: Self) -> Self::Output {
                    Self(self.0.$method(rhs.0))
                }
            }

            impl<'a> $trait<&'a MultiPrecisionInteger> for MultiPrecisionInteger {
                type Output = Self;

                #[inline]
                fn $method(self, rhs: &'a MultiPrecisionInteger) -> Self::Output {
                    Self(self.0.$method(&rhs.0))
                }
            }

            impl $trait<MultiPrecisionInteger> for &MultiPrecisionInteger {
                type Output = MultiPrecisionInteger;

                #[inline]
                fn $method(self, rhs: MultiPrecisionInteger) -> Self::Output {
                    MultiPrecisionInteger((&self.0).$method(rhs.0))
                }
            }

            impl<'a> $trait<&'a MultiPrecisionInteger> for &MultiPrecisionInteger {
                type Output = MultiPrecisionInteger;

                #[inline]
                fn $method(self, rhs: &'a MultiPrecisionInteger) -> Self::Output {
                    MultiPrecisionInteger((&self.0).$method(&rhs.0))
                }
            }

            impl $assign_trait<MultiPrecisionInteger> for MultiPrecisionInteger {
                #[inline]
                fn $assign_method(&mut self, rhs: MultiPrecisionInteger) {
                    self.0.$assign_method(rhs.0);
                }
            }

            impl<'a> $assign_trait<&'a MultiPrecisionInteger> for MultiPrecisionInteger {
                #[inline]
                fn $assign_method(&mut self, rhs: &'a MultiPrecisionInteger) {
                    self.0.$assign_method(&rhs.0);
                }
            }
        };
    }

    impl_bin_op!(Add, add, AddAssign, add_assign);
    impl_bin_op!(Sub, sub, SubAssign, sub_assign);
    impl_bin_op!(Mul, mul, MulAssign, mul_assign);
    impl_bin_op!(Div, div, DivAssign, div_assign);

    macro_rules! impl_owned_primitive_ops {
        ($($t:ty),* $(,)?) => {
            $(
                impl Add<$t> for MultiPrecisionInteger {
                    type Output = Self;

                    #[inline]
                    fn add(self, rhs: $t) -> Self::Output {
                        self + Self::from(rhs)
                    }
                }

                impl Add<MultiPrecisionInteger> for $t {
                    type Output = MultiPrecisionInteger;

                    #[inline]
                    fn add(self, rhs: MultiPrecisionInteger) -> Self::Output {
                        MultiPrecisionInteger::from(self) + rhs
                    }
                }

                impl Sub<$t> for MultiPrecisionInteger {
                    type Output = Self;

                    #[inline]
                    fn sub(self, rhs: $t) -> Self::Output {
                        self - Self::from(rhs)
                    }
                }

                impl Sub<MultiPrecisionInteger> for $t {
                    type Output = MultiPrecisionInteger;

                    #[inline]
                    fn sub(self, rhs: MultiPrecisionInteger) -> Self::Output {
                        MultiPrecisionInteger::from(self) - rhs
                    }
                }

                impl Mul<$t> for MultiPrecisionInteger {
                    type Output = Self;

                    #[inline]
                    fn mul(self, rhs: $t) -> Self::Output {
                        self * Self::from(rhs)
                    }
                }

                impl Mul<MultiPrecisionInteger> for $t {
                    type Output = MultiPrecisionInteger;

                    #[inline]
                    fn mul(self, rhs: MultiPrecisionInteger) -> Self::Output {
                        MultiPrecisionInteger::from(self) * rhs
                    }
                }

                impl Div<$t> for MultiPrecisionInteger {
                    type Output = Self;

                    #[inline]
                    fn div(self, rhs: $t) -> Self::Output {
                        self / Self::from(rhs)
                    }
                }

                impl Div<MultiPrecisionInteger> for $t {
                    type Output = MultiPrecisionInteger;

                    #[inline]
                    fn div(self, rhs: MultiPrecisionInteger) -> Self::Output {
                        MultiPrecisionInteger::from(self) / rhs
                    }
                }

                impl BitAnd<$t> for MultiPrecisionInteger {
                    type Output = Self;

                    #[inline]
                    fn bitand(self, rhs: $t) -> Self::Output {
                        self & Self::from(rhs)
                    }
                }

                impl BitAnd<MultiPrecisionInteger> for $t {
                    type Output = MultiPrecisionInteger;

                    #[inline]
                    fn bitand(self, rhs: MultiPrecisionInteger) -> Self::Output {
                        MultiPrecisionInteger::from(self) & rhs
                    }
                }

                impl AddAssign<$t> for MultiPrecisionInteger {
                    #[inline]
                    fn add_assign(&mut self, rhs: $t) {
                        *self += Self::from(rhs);
                    }
                }

                impl SubAssign<$t> for MultiPrecisionInteger {
                    #[inline]
                    fn sub_assign(&mut self, rhs: $t) {
                        *self -= Self::from(rhs);
                    }
                }

                impl MulAssign<$t> for MultiPrecisionInteger {
                    #[inline]
                    fn mul_assign(&mut self, rhs: $t) {
                        *self *= Self::from(rhs);
                    }
                }

                impl DivAssign<$t> for MultiPrecisionInteger {
                    #[inline]
                    fn div_assign(&mut self, rhs: $t) {
                        *self /= Self::from(rhs);
                    }
                }

                impl BitAndAssign<$t> for MultiPrecisionInteger {
                    #[inline]
                    fn bitand_assign(&mut self, rhs: $t) {
                        *self &= Self::from(rhs);
                    }
                }
            )*
        };
    }

    impl_owned_primitive_ops!(i64, i128, u32, u64, u128);

    macro_rules! impl_owned_ref_primitive_ops {
        ($($t:ty),* $(,)?) => {
            $(
                impl<'a> Add<&'a $t> for MultiPrecisionInteger {
                    type Output = Self;

                    #[inline]
                    fn add(self, rhs: &'a $t) -> Self::Output {
                        self + *rhs
                    }
                }

                impl<'a> Sub<&'a $t> for MultiPrecisionInteger {
                    type Output = Self;

                    #[inline]
                    fn sub(self, rhs: &'a $t) -> Self::Output {
                        self - *rhs
                    }
                }

                impl<'a> Mul<&'a $t> for MultiPrecisionInteger {
                    type Output = Self;

                    #[inline]
                    fn mul(self, rhs: &'a $t) -> Self::Output {
                        self * *rhs
                    }
                }

                impl<'a> Div<&'a $t> for MultiPrecisionInteger {
                    type Output = Self;

                    #[inline]
                    fn div(self, rhs: &'a $t) -> Self::Output {
                        self / *rhs
                    }
                }
            )*
        };
    }

    impl_owned_ref_primitive_ops!(i64, i128);

    macro_rules! impl_ref_primitive_ops {
        ($($t:ty),* $(,)?) => {
            $(
                impl<'a> Add<&'a MultiPrecisionInteger> for $t {
                    type Output = MultiPrecisionInteger;

                    #[inline]
                    fn add(self, rhs: &'a MultiPrecisionInteger) -> Self::Output {
                        MultiPrecisionInteger::from(self) + rhs
                    }
                }

                impl Add<$t> for &MultiPrecisionInteger {
                    type Output = MultiPrecisionInteger;

                    #[inline]
                    fn add(self, rhs: $t) -> Self::Output {
                        self + &MultiPrecisionInteger::from(rhs)
                    }
                }

                impl<'a> Sub<&'a MultiPrecisionInteger> for $t {
                    type Output = MultiPrecisionInteger;

                    #[inline]
                    fn sub(self, rhs: &'a MultiPrecisionInteger) -> Self::Output {
                        MultiPrecisionInteger::from(self) - rhs
                    }
                }

                impl Sub<$t> for &MultiPrecisionInteger {
                    type Output = MultiPrecisionInteger;

                    #[inline]
                    fn sub(self, rhs: $t) -> Self::Output {
                        self - &MultiPrecisionInteger::from(rhs)
                    }
                }

                impl<'a> Mul<&'a MultiPrecisionInteger> for $t {
                    type Output = MultiPrecisionInteger;

                    #[inline]
                    fn mul(self, rhs: &'a MultiPrecisionInteger) -> Self::Output {
                        MultiPrecisionInteger::from(self) * rhs
                    }
                }

                impl Mul<$t> for &MultiPrecisionInteger {
                    type Output = MultiPrecisionInteger;

                    #[inline]
                    fn mul(self, rhs: $t) -> Self::Output {
                        self * &MultiPrecisionInteger::from(rhs)
                    }
                }

                impl<'a> Div<&'a MultiPrecisionInteger> for $t {
                    type Output = MultiPrecisionInteger;

                    #[inline]
                    fn div(self, rhs: &'a MultiPrecisionInteger) -> Self::Output {
                        MultiPrecisionInteger::from(self) / rhs
                    }
                }

                impl Div<$t> for &MultiPrecisionInteger {
                    type Output = MultiPrecisionInteger;

                    #[inline]
                    fn div(self, rhs: $t) -> Self::Output {
                        self / &MultiPrecisionInteger::from(rhs)
                    }
                }

                impl<'a> BitAnd<&'a MultiPrecisionInteger> for $t {
                    type Output = MultiPrecisionInteger;

                    #[inline]
                    fn bitand(self, rhs: &'a MultiPrecisionInteger) -> Self::Output {
                        MultiPrecisionInteger::from(self) & rhs
                    }
                }

                impl BitAnd<$t> for &MultiPrecisionInteger {
                    type Output = MultiPrecisionInteger;

                    #[inline]
                    fn bitand(self, rhs: $t) -> Self::Output {
                        self & &MultiPrecisionInteger::from(rhs)
                    }
                }
            )*
        };
    }

    impl_ref_primitive_ops!(i64, i128);

    macro_rules! impl_ref_ref_primitive_ops {
        ($($t:ty),* $(,)?) => {
            $(
                impl<'a, 'b> Add<&'a MultiPrecisionInteger> for &'b $t {
                    type Output = MultiPrecisionInteger;

                    #[inline]
                    fn add(self, rhs: &'a MultiPrecisionInteger) -> Self::Output {
                        MultiPrecisionInteger::from(*self) + rhs
                    }
                }

                impl<'a, 'b> Sub<&'a MultiPrecisionInteger> for &'b $t {
                    type Output = MultiPrecisionInteger;

                    #[inline]
                    fn sub(self, rhs: &'a MultiPrecisionInteger) -> Self::Output {
                        MultiPrecisionInteger::from(*self) - rhs
                    }
                }

                impl<'a, 'b> Mul<&'a MultiPrecisionInteger> for &'b $t {
                    type Output = MultiPrecisionInteger;

                    #[inline]
                    fn mul(self, rhs: &'a MultiPrecisionInteger) -> Self::Output {
                        MultiPrecisionInteger::from(*self) * rhs
                    }
                }

                impl<'a, 'b> Div<&'a MultiPrecisionInteger> for &'b $t {
                    type Output = MultiPrecisionInteger;

                    #[inline]
                    fn div(self, rhs: &'a MultiPrecisionInteger) -> Self::Output {
                        MultiPrecisionInteger::from(*self) / rhs
                    }
                }

                impl<'a, 'b> Mul<&'a $t> for &'b MultiPrecisionInteger {
                    type Output = MultiPrecisionInteger;

                    #[inline]
                    fn mul(self, rhs: &'a $t) -> Self::Output {
                        self * &MultiPrecisionInteger::from(*rhs)
                    }
                }
            )*
        };
    }

    impl_ref_ref_primitive_ops!(i64, i128);

    impl BitAnd for MultiPrecisionInteger {
        type Output = Self;

        #[inline]
        fn bitand(self, rhs: Self) -> Self::Output {
            Self(self.0 & rhs.0)
        }
    }

    impl<'a> BitAnd<&'a MultiPrecisionInteger> for MultiPrecisionInteger {
        type Output = Self;

        #[inline]
        fn bitand(self, rhs: &'a MultiPrecisionInteger) -> Self::Output {
            Self(self.0 & &rhs.0)
        }
    }

    impl BitAnd<MultiPrecisionInteger> for &MultiPrecisionInteger {
        type Output = MultiPrecisionInteger;

        #[inline]
        fn bitand(self, rhs: MultiPrecisionInteger) -> Self::Output {
            MultiPrecisionInteger((&self.0) & rhs.0)
        }
    }

    impl<'a> BitAnd<&'a MultiPrecisionInteger> for &MultiPrecisionInteger {
        type Output = MultiPrecisionInteger;

        #[inline]
        fn bitand(self, rhs: &'a MultiPrecisionInteger) -> Self::Output {
            MultiPrecisionInteger((&self.0) & &rhs.0)
        }
    }

    impl BitAndAssign<MultiPrecisionInteger> for MultiPrecisionInteger {
        #[inline]
        fn bitand_assign(&mut self, rhs: MultiPrecisionInteger) {
            self.0.bitand_assign(rhs.0);
        }
    }

    impl<'a> BitAndAssign<&'a MultiPrecisionInteger> for MultiPrecisionInteger {
        #[inline]
        fn bitand_assign(&mut self, rhs: &'a MultiPrecisionInteger) {
            self.0.bitand_assign(&rhs.0);
        }
    }

    impl Neg for MultiPrecisionInteger {
        type Output = Self;

        #[inline]
        fn neg(self) -> Self::Output {
            Self(-self.0)
        }
    }

    impl Neg for &MultiPrecisionInteger {
        type Output = MultiPrecisionInteger;

        #[inline]
        fn neg(self) -> Self::Output {
            MultiPrecisionInteger(-&self.0)
        }
    }

    impl Rem for MultiPrecisionInteger {
        type Output = Self;

        #[inline]
        fn rem(self, rhs: Self) -> Self::Output {
            Self::rem_trunc(self.0, rhs.0)
        }
    }

    impl<'a> Rem<&'a MultiPrecisionInteger> for MultiPrecisionInteger {
        type Output = Self;

        #[inline]
        fn rem(self, rhs: &'a MultiPrecisionInteger) -> Self::Output {
            Self::rem_trunc(self.0, rhs.0.clone())
        }
    }

    impl RemAssign for MultiPrecisionInteger {
        #[inline]
        fn rem_assign(&mut self, rhs: Self) {
            *self = self.clone() % rhs;
        }
    }

    impl<'a> RemAssign<&'a MultiPrecisionInteger> for MultiPrecisionInteger {
        #[inline]
        fn rem_assign(&mut self, rhs: &'a MultiPrecisionInteger) {
            *self = self.clone() % rhs;
        }
    }

    impl Shl<u32> for MultiPrecisionInteger {
        type Output = Self;

        #[inline]
        fn shl(self, rhs: u32) -> Self::Output {
            Self(self.0 << u64::from(rhs))
        }
    }

    impl Shl<usize> for MultiPrecisionInteger {
        type Output = Self;

        #[inline]
        fn shl(self, rhs: usize) -> Self::Output {
            Self(self.0 << u64::try_from(rhs).expect("Shift amount does not fit in u64"))
        }
    }

    impl Shl<usize> for &MultiPrecisionInteger {
        type Output = MultiPrecisionInteger;

        #[inline]
        fn shl(self, rhs: usize) -> Self::Output {
            MultiPrecisionInteger(
                &self.0 << u64::try_from(rhs).expect("Shift amount does not fit in u64"),
            )
        }
    }

    impl Shr<u32> for MultiPrecisionInteger {
        type Output = Self;

        #[inline]
        fn shr(self, rhs: u32) -> Self::Output {
            Self(self.0 >> u64::from(rhs))
        }
    }

    impl Shr<u32> for &MultiPrecisionInteger {
        type Output = MultiPrecisionInteger;

        #[inline]
        fn shr(self, rhs: u32) -> Self::Output {
            MultiPrecisionInteger(&self.0 >> u64::from(rhs))
        }
    }

    impl Shl<u64> for MultiPrecisionInteger {
        type Output = Self;

        #[inline]
        fn shl(self, rhs: u64) -> Self::Output {
            Self(self.0 << rhs)
        }
    }

    impl Shr<usize> for MultiPrecisionInteger {
        type Output = Self;

        #[inline]
        fn shr(self, rhs: usize) -> Self::Output {
            Self(self.0 >> u64::try_from(rhs).expect("Shift amount does not fit in u64"))
        }
    }

    impl Shr<usize> for &MultiPrecisionInteger {
        type Output = MultiPrecisionInteger;

        #[inline]
        fn shr(self, rhs: usize) -> Self::Output {
            MultiPrecisionInteger(
                &self.0 >> u64::try_from(rhs).expect("Shift amount does not fit in u64"),
            )
        }
    }

    #[cfg(feature = "bincode")]
    pub(crate) fn to_be_bytes(value: &MultiPrecisionInteger) -> Vec<u8> {
        value.to_string().into_bytes()
    }

    #[cfg(feature = "bincode")]
    pub(crate) fn from_be_bytes(bytes: &[u8]) -> Result<MultiPrecisionInteger, &'static str> {
        std::str::from_utf8(bytes)
            .ok()
            .and_then(|s| s.parse::<MultiPrecisionInteger>().ok())
            .ok_or("Failed to parse large integer")
    }

    pub(crate) struct BackendRandState {
        rng: rand::rngs::ThreadRng,
    }

    impl BackendRandState {
        pub(crate) fn new(_seed: u128) -> Self {
            Self { rng: rand::rng() }
        }

        pub(crate) fn below(&mut self, modulus: &MultiPrecisionInteger) -> MultiPrecisionInteger {
            use rand::Rng;

            if let Some(m) = modulus.to_u128() {
                MultiPrecisionInteger::from(self.rng.random_range(0..m))
            } else {
                MultiPrecisionInteger::from(self.rng.random::<u128>()).rem_euc(modulus.clone())
            }
        }
    }
}

#[cfg(feature = "no_gmp")]
pub use malachite::MultiPrecisionInteger;
#[cfg(feature = "no_gmp")]
pub(crate) use malachite::{BackendRandState, Complete, RemRounding};
#[cfg(feature = "no_gmp")]
pub(crate) fn pow_ref_u32(base: &MultiPrecisionInteger, e: u32) -> MultiPrecisionInteger {
    base.clone().pow(e)
}
#[cfg(feature = "no_gmp")]
pub(crate) fn probably_prime(_value: &MultiPrecisionInteger, _reps: u32) -> Option<bool> {
    None
}
#[cfg(feature = "no_gmp")]
pub fn from_lsf_bytes(bytes: &[u8]) -> MultiPrecisionInteger {
    let mut value = MultiPrecisionInteger::from(0u32);
    for &byte in bytes.iter().rev() {
        value = (value << 8usize) + u32::from(byte);
    }
    value
}
#[cfg(feature = "no_gmp")]
pub fn write_lsf_bytes(value: &MultiPrecisionInteger, dest: &mut Vec<u8>) {
    let mut value = if value.is_negative() {
        -value.clone()
    } else {
        value.clone()
    };

    while !value.is_zero() {
        dest.push((value.clone() & 0xffu32).to_u64().unwrap() as u8);
        value = value >> 8usize;
    }
}
#[cfg(feature = "no_gmp")]
pub fn lsf_byte_size(value: &MultiPrecisionInteger) -> usize {
    value
        .significant_bits()
        .div_ceil(u64::from(u8::BITS))
        .try_into()
        .expect("large integer byte length does not fit in usize")
}
#[cfg(feature = "no_gmp")]
pub fn to_lsf_bytes(value: &MultiPrecisionInteger) -> Vec<u8> {
    let mut bytes = Vec::new();
    write_lsf_bytes(value, &mut bytes);
    bytes
}
#[cfg(feature = "no_gmp")]
pub fn from_digits_radix(digits: &[u8], radix: u32, is_negative: bool) -> MultiPrecisionInteger {
    let mut value = MultiPrecisionInteger::from(0u32);
    for &digit in digits {
        value *= radix;
        value += u32::from(digit);
    }

    if is_negative { -value } else { value }
}
#[cfg(all(feature = "no_gmp", feature = "bincode"))]
pub(crate) use malachite::{from_be_bytes, to_be_bytes};
