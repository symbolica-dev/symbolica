mod implementation {
    #[cfg(all(feature = "gmp", target_pointer_width = "64", not(windows)))]
    use rug::integer::IntegerExt64;
    #[cfg(feature = "gmp")]
    use std::cell::RefCell;
    use std::{
        fmt::{Debug, Display, Formatter, UpperHex},
        ops::{
            Add, AddAssign, BitAnd, BitAndAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem,
            RemAssign, Shl, Shr, Sub, SubAssign,
        },
        str::FromStr,
    };

    #[cfg(feature = "integer-malachite")]
    use malachite_base::num::{
        arithmetic::traits::{
            Abs, ExtendedGcd, FloorRoot, Gcd, Mod, Pow as MalachitePow, UnsignedAbs,
        },
        logic::traits::SignificantBits,
    };
    /// Native integer type used by the selected arbitrary-precision backend.
    #[cfg(feature = "integer-malachite")]
    pub type RawMultiPrecisionInteger = malachite_nz::integer::Integer;
    /// Native integer type used by the selected arbitrary-precision backend.
    #[cfg(feature = "integer-gmp")]
    pub type RawMultiPrecisionInteger = rug::Integer;

    #[cfg(feature = "gmp")]
    const LARGE_INTEGER_CACHE_SIZE: usize = 32;
    #[cfg(feature = "gmp")]
    const LARGE_INTEGER_CACHE_MAX_BITS: usize = 1 << 20;

    #[cfg(feature = "gmp")]
    thread_local! {
        /// A bounded cache of cleared arbitrary-precision integers. Keeping the backing limb
        /// allocation alive avoids repeatedly entering the allocator for short-lived results.
        static LARGE_INTEGER_CACHE: RefCell<Vec<RawMultiPrecisionInteger>> =
            const { RefCell::new(Vec::new()) };
    }

    /// A backend-independent arbitrary-precision integer.
    ///
    /// Use [`Self::as_raw`], [`Self::to_raw`], or [`Self::into_raw`] when a
    /// backend-specific operation is required.
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    #[cfg_attr(feature = "serde", serde(transparent))]
    #[derive(PartialEq, Eq, PartialOrd, Ord, Hash)]
    #[repr(transparent)]
    pub struct MultiPrecisionInteger(RawMultiPrecisionInteger);

    impl Clone for MultiPrecisionInteger {
        #[inline]
        fn clone(&self) -> Self {
            #[cfg(feature = "gmp")]
            {
                use rug::Assign;

                let mut result = Self::default();
                result.0.assign(&self.0);
                result
            }
            #[cfg(feature = "no_gmp")]
            {
                Self(self.0.clone())
            }
        }
    }

    impl Default for MultiPrecisionInteger {
        #[inline]
        fn default() -> Self {
            #[cfg(feature = "gmp")]
            {
                let value = LARGE_INTEGER_CACHE
                    .try_with(|cache| cache.try_borrow_mut().ok()?.pop())
                    .ok()
                    .flatten()
                    .unwrap_or_default();
                Self(value)
            }
            #[cfg(feature = "no_gmp")]
            {
                Self(RawMultiPrecisionInteger::default())
            }
        }
    }

    #[cfg(feature = "gmp")]
    impl Drop for MultiPrecisionInteger {
        #[inline]
        fn drop(&mut self) {
            let mut value = std::mem::take(&mut self.0);

            let cacheable = value.capacity() > 0
                && usize::try_from(value.capacity()).unwrap_or(usize::MAX)
                    <= LARGE_INTEGER_CACHE_MAX_BITS;

            if !cacheable {
                return;
            }

            use rug::Assign;
            value.assign(0);

            let _ = LARGE_INTEGER_CACHE.try_with(|cache| {
                if let Ok(mut cache) = cache.try_borrow_mut()
                    && cache.len() < LARGE_INTEGER_CACHE_SIZE
                {
                    cache.push(value);
                }
            });
        }
    }

    /// Error returned when parsing a [`MultiPrecisionInteger`].
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub struct ParseMultiPrecisionIntegerError;

    impl Display for ParseMultiPrecisionIntegerError {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            f.write_str("invalid arbitrary-precision integer")
        }
    }

    impl std::error::Error for ParseMultiPrecisionIntegerError {}

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
        /// Wrap a value from the selected arbitrary-precision backend.
        #[inline]
        pub fn from_raw(value: RawMultiPrecisionInteger) -> Self {
            Self(value)
        }

        /// Borrow the value from the selected arbitrary-precision backend.
        #[inline]
        pub fn as_raw(&self) -> &RawMultiPrecisionInteger {
            &self.0
        }

        /// Clone the value from the selected arbitrary-precision backend.
        #[inline]
        pub fn to_raw(&self) -> RawMultiPrecisionInteger {
            self.0.clone()
        }

        /// Consume this wrapper and return the selected backend's value.
        #[inline]
        pub fn into_raw(self) -> RawMultiPrecisionInteger {
            let mut this = self;
            std::mem::take(&mut this.0)
        }

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
            #[cfg(feature = "integer-gmp")]
            return self.0.to_i64();
            #[cfg(feature = "integer-malachite")]
            return i64::try_from(&self.0).ok();
        }

        #[inline]
        pub fn to_i128(&self) -> Option<i128> {
            #[cfg(feature = "integer-gmp")]
            return self.0.to_i128();
            #[cfg(feature = "integer-malachite")]
            return i128::try_from(&self.0).ok();
        }

        #[inline]
        pub fn to_u64(&self) -> Option<u64> {
            #[cfg(feature = "integer-gmp")]
            return self.0.to_u64();
            #[cfg(feature = "integer-malachite")]
            return u64::try_from(&self.0).ok();
        }

        pub fn to_usize(&self) -> Option<usize> {
            self.to_u64().and_then(|x| usize::try_from(x).ok())
        }

        #[inline]
        pub fn to_u128(&self) -> Option<u128> {
            #[cfg(feature = "integer-gmp")]
            return self.0.to_u128();
            #[cfg(feature = "integer-malachite")]
            return u128::try_from(&self.0).ok();
        }

        #[inline]
        pub fn mod_u(&self, modulus: u32) -> u32 {
            #[cfg(feature = "gmp")]
            return self.0.mod_u(modulus);
            #[cfg(feature = "no_gmp")]
            return self.rem_euc(Self::from(modulus)).to_u64().unwrap() as u32;
        }

        #[inline]
        pub fn mod_u64(&self, modulus: u64) -> u64 {
            #[cfg(all(feature = "gmp", target_pointer_width = "64", not(windows)))]
            return self.0.mod_u64(modulus);
            #[cfg(any(
                feature = "no_gmp",
                all(feature = "gmp", any(not(target_pointer_width = "64"), windows))
            ))]
            return self.rem_euc(Self::from(modulus)).to_u64().unwrap();
        }

        #[inline]
        pub fn rem_euc<T: Into<Self>>(&self, rhs: T) -> Self {
            let rhs = rhs.into();
            let negative = rhs.is_negative();
            let modulus = rhs.into_raw();
            let modulus = if negative { -modulus } else { modulus };
            let mut remainder = self.0.clone() % modulus.clone();
            if remainder < 0 {
                remainder += modulus;
            }
            Self(remainder)
        }

        #[inline]
        pub fn div_rem_euc<T: Into<Self>>(self, rhs: T) -> (Self, Self) {
            #[cfg(feature = "gmp")]
            {
                let mut quotient = self.into_raw();
                let mut remainder = rhs.into().into_raw();
                quotient.div_rem_euc_mut(&mut remainder);
                return (Self(quotient), Self(remainder));
            }
            #[cfg(feature = "no_gmp")]
            {
                let rhs = rhs.into();
                let r = self.rem_euc(rhs.clone());
                let q = (self.into_raw() - r.0.clone()) / rhs.into_raw();
                return (Self(q), r);
            }
        }

        #[inline]
        pub fn div_rem_ref(&self, rhs: &Self) -> (Self, Self) {
            #[cfg(feature = "gmp")]
            {
                use rug::Assign;

                let mut quotient = Self::default();
                let mut remainder = Self::default();
                (&mut quotient.0, &mut remainder.0).assign(self.0.div_rem_ref(&rhs.0));
                return (quotient, remainder);
            }
            #[cfg(feature = "no_gmp")]
            {
                let q = self.0.clone() / rhs.0.clone();
                let r = self.0.clone() - q.clone() * rhs.0.clone();
                return (Self(q), Self(r));
            }
        }

        /// Divide in place when divisibility is guaranteed by the caller.
        #[inline]
        pub fn div_exact_owned(mut self, rhs: &Self) -> Self {
            #[cfg(feature = "gmp")]
            self.0.div_exact_mut(&rhs.0);
            #[cfg(feature = "no_gmp")]
            {
                self /= rhs;
            }
            self
        }

        #[inline]
        pub fn root_ref(&self, e: u32) -> Self {
            #[cfg(feature = "integer-gmp")]
            {
                return Self(self.0.clone().root(e));
            }
            #[cfg(feature = "integer-malachite")]
            {
                return Self(self.0.clone().floor_root(u64::from(e)));
            }
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
            #[cfg(feature = "integer-gmp")]
            {
                return Self(self.0.clone().gcd(&rhs.0));
            }
            #[cfg(feature = "integer-malachite")]
            {
                return Self(RawMultiPrecisionInteger::from(Gcd::gcd(
                    self.0.clone().unsigned_abs(),
                    rhs.0.clone().unsigned_abs(),
                )));
            }
        }

        #[inline]
        pub fn extended_gcd(self, rhs: Self, _scratch: Self) -> (Self, Self, Self) {
            #[cfg(feature = "integer-gmp")]
            {
                let (g, s, t) = self
                    .into_raw()
                    .extended_gcd(rhs.into_raw(), _scratch.into_raw());
                return (Self(g), Self(s), Self(t));
            }
            #[cfg(feature = "integer-malachite")]
            {
                let (g, s, t) = ExtendedGcd::extended_gcd(self.into_raw(), rhs.into_raw());
                return (Self(RawMultiPrecisionInteger::from(g)), Self(s), Self(t));
            }
        }

        #[inline]
        pub fn invert(&self, modulus: &Self) -> Result<Self, ()> {
            #[cfg(feature = "integer-gmp")]
            {
                return self.0.clone().invert(&modulus.0).map(Self).map_err(|_| ());
            }
            #[cfg(feature = "integer-malachite")]
            {
                let (g, s, _) = ExtendedGcd::extended_gcd(self.0.clone(), modulus.0.clone());
                if g != 1u32 {
                    return Err(());
                }
                return Ok(Self(s.mod_op(modulus.0.clone())));
            }
        }

        #[inline]
        pub fn significant_bits(&self) -> u64 {
            #[cfg(feature = "integer-gmp")]
            return u64::from(self.0.significant_bits());
            #[cfg(feature = "integer-malachite")]
            return SignificantBits::significant_bits(&self.0);
        }

        /// Add `b * c` in place. With GMP this maps to its fused add-multiply operation and does
        /// not materialize the product.
        #[inline(always)]
        pub fn add_mul_assign(&mut self, b: &Self, c: &Self) {
            self.0 += &b.0 * &c.0;
        }

        /// Subtract `b * c` in place without materializing the product.
        #[inline(always)]
        pub fn sub_mul_assign(&mut self, b: &Self, c: &Self) {
            self.0 -= &b.0 * &c.0;
        }

        #[inline(always)]
        pub fn add_i64_mul_assign(&mut self, b: i64, c: &Self) {
            #[cfg(feature = "gmp")]
            {
                self.0 += &c.0 * b;
            }
            #[cfg(feature = "no_gmp")]
            {
                self.0 += &c.0 * RawMultiPrecisionInteger::from(b);
            }
        }

        #[inline(always)]
        pub fn sub_i64_mul_assign(&mut self, b: i64, c: &Self) {
            #[cfg(feature = "gmp")]
            {
                self.0 -= &c.0 * b;
            }
            #[cfg(feature = "no_gmp")]
            {
                self.0 -= &c.0 * RawMultiPrecisionInteger::from(b);
            }
        }

        #[inline(always)]
        pub fn add_i128_mul_assign(&mut self, b: i128, c: &Self) {
            #[cfg(feature = "gmp")]
            {
                self.0 += &c.0 * b;
            }
            #[cfg(feature = "no_gmp")]
            {
                self.0 += &c.0 * RawMultiPrecisionInteger::from(b);
            }
        }

        #[inline(always)]
        pub fn sub_i128_mul_assign(&mut self, b: i128, c: &Self) {
            #[cfg(feature = "gmp")]
            {
                self.0 -= &c.0 * b;
            }
            #[cfg(feature = "no_gmp")]
            {
                self.0 -= &c.0 * RawMultiPrecisionInteger::from(b);
            }
        }

        #[inline]
        pub fn pow(self, e: u32) -> Self {
            #[cfg(feature = "integer-gmp")]
            {
                use rug::ops::Pow;
                return Self(self.into_raw().pow(e));
            }
            #[cfg(feature = "integer-malachite")]
            {
                return Self(MalachitePow::pow(self.into_raw(), u64::from(e)));
            }
        }

        #[inline]
        pub fn abs(self) -> Self {
            Self(self.into_raw().abs())
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

        fn rem_trunc(lhs: RawMultiPrecisionInteger, rhs: RawMultiPrecisionInteger) -> Self {
            #[cfg(feature = "gmp")]
            {
                let mut lhs = lhs;
                lhs %= rhs;
                return Self(lhs);
            }
            #[cfg(feature = "no_gmp")]
            {
                let q = lhs.clone() / rhs.clone();
                return Self(lhs - q * rhs);
            }
        }

        #[inline]
        fn shl_raw(value: RawMultiPrecisionInteger, rhs: u64) -> RawMultiPrecisionInteger {
            #[cfg(feature = "integer-gmp")]
            return value << usize::try_from(rhs).expect("shift amount does not fit in usize");
            #[cfg(feature = "integer-malachite")]
            return value << rhs;
        }

        #[inline]
        fn shr_raw(value: RawMultiPrecisionInteger, rhs: u64) -> RawMultiPrecisionInteger {
            #[cfg(feature = "integer-gmp")]
            return value >> usize::try_from(rhs).expect("shift amount does not fit in usize");
            #[cfg(feature = "integer-malachite")]
            return value >> rhs;
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
        type Err = ParseMultiPrecisionIntegerError;

        fn from_str(s: &str) -> Result<Self, Self::Err> {
            RawMultiPrecisionInteger::from_str(s)
                .map(Self)
                .map_err(|_| ParseMultiPrecisionIntegerError)
        }
    }

    macro_rules! impl_from_primitive {
        ($($t:ty),* $(,)?) => {
            $(
                impl From<$t> for MultiPrecisionInteger {
                    #[inline]
                    fn from(value: $t) -> Self {
                        #[cfg(feature = "gmp")]
                        {
                            use rug::Assign;

                            let mut result = Self::default();
                            result.0.assign(value);
                            result
                        }
                        #[cfg(feature = "no_gmp")]
                        {
                            Self(RawMultiPrecisionInteger::from(value))
                        }
                    }
                }
            )*
        };
    }

    impl_from_primitive!(
        i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize
    );

    impl From<RawMultiPrecisionInteger> for MultiPrecisionInteger {
        #[inline]
        fn from(value: RawMultiPrecisionInteger) -> Self {
            Self(value)
        }
    }

    impl From<MultiPrecisionInteger> for RawMultiPrecisionInteger {
        #[inline]
        fn from(value: MultiPrecisionInteger) -> Self {
            value.into_raw()
        }
    }

    macro_rules! impl_cmp_primitive {
        ($($t:ty),* $(,)?) => {
            $(
                impl PartialEq<$t> for MultiPrecisionInteger {
                    #[inline]
                    fn eq(&self, other: &$t) -> bool {
                        self.0 == RawMultiPrecisionInteger::from(*other)
                    }
                }

                impl PartialEq<MultiPrecisionInteger> for $t {
                    #[inline]
                    fn eq(&self, other: &MultiPrecisionInteger) -> bool {
                        RawMultiPrecisionInteger::from(*self) == other.0
                    }
                }

                impl PartialOrd<$t> for MultiPrecisionInteger {
                    #[inline]
                    fn partial_cmp(&self, other: &$t) -> Option<std::cmp::Ordering> {
                        self.0.partial_cmp(&RawMultiPrecisionInteger::from(*other))
                    }
                }

                impl PartialOrd<MultiPrecisionInteger> for $t {
                    #[inline]
                    fn partial_cmp(&self, other: &MultiPrecisionInteger) -> Option<std::cmp::Ordering> {
                        RawMultiPrecisionInteger::from(*self).partial_cmp(&other.0)
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
                    Self(self.into_raw().$method(rhs.into_raw()))
                }
            }

            impl<'a> $trait<&'a MultiPrecisionInteger> for MultiPrecisionInteger {
                type Output = Self;

                #[inline]
                fn $method(self, rhs: &'a MultiPrecisionInteger) -> Self::Output {
                    Self(self.into_raw().$method(&rhs.0))
                }
            }

            impl $trait<MultiPrecisionInteger> for &MultiPrecisionInteger {
                type Output = MultiPrecisionInteger;

                #[inline]
                fn $method(self, rhs: MultiPrecisionInteger) -> Self::Output {
                    MultiPrecisionInteger((&self.0).$method(rhs.into_raw()))
                }
            }

            impl<'a> $trait<&'a MultiPrecisionInteger> for &MultiPrecisionInteger {
                type Output = MultiPrecisionInteger;

                #[inline]
                fn $method(self, rhs: &'a MultiPrecisionInteger) -> Self::Output {
                    #[cfg(feature = "gmp")]
                    {
                        use rug::Assign;

                        let mut result = MultiPrecisionInteger::default();
                        result.0.assign((&self.0).$method(&rhs.0));
                        result
                    }
                    #[cfg(feature = "no_gmp")]
                    {
                        MultiPrecisionInteger(RawMultiPrecisionInteger::from(
                            (&self.0).$method(&rhs.0),
                        ))
                    }
                }
            }

            impl $assign_trait<MultiPrecisionInteger> for MultiPrecisionInteger {
                #[inline]
                fn $assign_method(&mut self, rhs: MultiPrecisionInteger) {
                    self.0.$assign_method(rhs.into_raw());
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
                        #[cfg(feature = "gmp")]
                        {
                            Self(self.into_raw() + rhs)
                        }
                        #[cfg(feature = "no_gmp")]
                        {
                            self + Self::from(rhs)
                        }
                    }
                }

                impl Add<MultiPrecisionInteger> for $t {
                    type Output = MultiPrecisionInteger;

                    #[inline]
                    fn add(self, rhs: MultiPrecisionInteger) -> Self::Output {
                        rhs + self
                    }
                }

                impl Sub<$t> for MultiPrecisionInteger {
                    type Output = Self;

                    #[inline]
                    fn sub(self, rhs: $t) -> Self::Output {
                        #[cfg(feature = "gmp")]
                        {
                            Self(self.into_raw() - rhs)
                        }
                        #[cfg(feature = "no_gmp")]
                        {
                            self - Self::from(rhs)
                        }
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
                        #[cfg(feature = "gmp")]
                        {
                            Self(self.into_raw() * rhs)
                        }
                        #[cfg(feature = "no_gmp")]
                        {
                            self * Self::from(rhs)
                        }
                    }
                }

                impl Mul<MultiPrecisionInteger> for $t {
                    type Output = MultiPrecisionInteger;

                    #[inline]
                    fn mul(self, rhs: MultiPrecisionInteger) -> Self::Output {
                        rhs * self
                    }
                }

                impl Div<$t> for MultiPrecisionInteger {
                    type Output = Self;

                    #[inline]
                    fn div(self, rhs: $t) -> Self::Output {
                        #[cfg(feature = "gmp")]
                        {
                            Self(self.into_raw() / rhs)
                        }
                        #[cfg(feature = "no_gmp")]
                        {
                            self / Self::from(rhs)
                        }
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
                        #[cfg(feature = "gmp")]
                        self.0.add_assign(rhs);
                        #[cfg(feature = "no_gmp")]
                        self.0.add_assign(RawMultiPrecisionInteger::from(rhs));
                    }
                }

                impl SubAssign<$t> for MultiPrecisionInteger {
                    #[inline]
                    fn sub_assign(&mut self, rhs: $t) {
                        #[cfg(feature = "gmp")]
                        self.0.sub_assign(rhs);
                        #[cfg(feature = "no_gmp")]
                        self.0.sub_assign(RawMultiPrecisionInteger::from(rhs));
                    }
                }

                impl MulAssign<$t> for MultiPrecisionInteger {
                    #[inline]
                    fn mul_assign(&mut self, rhs: $t) {
                        #[cfg(feature = "gmp")]
                        self.0.mul_assign(rhs);
                        #[cfg(feature = "no_gmp")]
                        self.0.mul_assign(RawMultiPrecisionInteger::from(rhs));
                    }
                }

                impl DivAssign<$t> for MultiPrecisionInteger {
                    #[inline]
                    fn div_assign(&mut self, rhs: $t) {
                        #[cfg(feature = "gmp")]
                        self.0.div_assign(rhs);
                        #[cfg(feature = "no_gmp")]
                        self.0.div_assign(RawMultiPrecisionInteger::from(rhs));
                    }
                }

                impl BitAndAssign<$t> for MultiPrecisionInteger {
                    #[inline]
                    fn bitand_assign(&mut self, rhs: $t) {
                        #[cfg(feature = "gmp")]
                        self.0.bitand_assign(rhs);
                        #[cfg(feature = "no_gmp")]
                        self.0
                            .bitand_assign(RawMultiPrecisionInteger::from(rhs));
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
            Self(self.into_raw() & rhs.into_raw())
        }
    }

    impl<'a> BitAnd<&'a MultiPrecisionInteger> for MultiPrecisionInteger {
        type Output = Self;

        #[inline]
        fn bitand(self, rhs: &'a MultiPrecisionInteger) -> Self::Output {
            Self(self.into_raw() & &rhs.0)
        }
    }

    impl BitAnd<MultiPrecisionInteger> for &MultiPrecisionInteger {
        type Output = MultiPrecisionInteger;

        #[inline]
        fn bitand(self, rhs: MultiPrecisionInteger) -> Self::Output {
            MultiPrecisionInteger((&self.0) & rhs.into_raw())
        }
    }

    impl<'a> BitAnd<&'a MultiPrecisionInteger> for &MultiPrecisionInteger {
        type Output = MultiPrecisionInteger;

        #[inline]
        fn bitand(self, rhs: &'a MultiPrecisionInteger) -> Self::Output {
            MultiPrecisionInteger(RawMultiPrecisionInteger::from((&self.0) & &rhs.0))
        }
    }

    impl BitAndAssign<MultiPrecisionInteger> for MultiPrecisionInteger {
        #[inline]
        fn bitand_assign(&mut self, rhs: MultiPrecisionInteger) {
            self.0.bitand_assign(rhs.into_raw());
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
            Self(-self.into_raw())
        }
    }

    impl Neg for &MultiPrecisionInteger {
        type Output = MultiPrecisionInteger;

        #[inline]
        fn neg(self) -> Self::Output {
            MultiPrecisionInteger(RawMultiPrecisionInteger::from(-&self.0))
        }
    }

    impl Rem for MultiPrecisionInteger {
        type Output = Self;

        #[inline]
        fn rem(self, rhs: Self) -> Self::Output {
            Self::rem_trunc(self.into_raw(), rhs.into_raw())
        }
    }

    impl<'a> Rem<&'a MultiPrecisionInteger> for MultiPrecisionInteger {
        type Output = Self;

        #[inline]
        fn rem(self, rhs: &'a MultiPrecisionInteger) -> Self::Output {
            Self::rem_trunc(self.into_raw(), rhs.0.clone())
        }
    }

    impl RemAssign for MultiPrecisionInteger {
        #[inline]
        fn rem_assign(&mut self, rhs: Self) {
            #[cfg(feature = "gmp")]
            {
                self.0 %= rhs.into_raw();
            }
            #[cfg(feature = "no_gmp")]
            {
                *self = self.clone() % rhs;
            }
        }
    }

    impl<'a> RemAssign<&'a MultiPrecisionInteger> for MultiPrecisionInteger {
        #[inline]
        fn rem_assign(&mut self, rhs: &'a MultiPrecisionInteger) {
            #[cfg(feature = "gmp")]
            {
                self.0 %= &rhs.0;
            }
            #[cfg(feature = "no_gmp")]
            {
                *self = self.clone() % rhs;
            }
        }
    }

    impl Shl<u32> for MultiPrecisionInteger {
        type Output = Self;

        #[inline]
        fn shl(self, rhs: u32) -> Self::Output {
            Self(MultiPrecisionInteger::shl_raw(
                self.into_raw(),
                u64::from(rhs),
            ))
        }
    }

    impl Shl<usize> for MultiPrecisionInteger {
        type Output = Self;

        #[inline]
        fn shl(self, rhs: usize) -> Self::Output {
            Self(MultiPrecisionInteger::shl_raw(
                self.into_raw(),
                u64::try_from(rhs).expect("shift amount does not fit in u64"),
            ))
        }
    }

    impl Shl<usize> for &MultiPrecisionInteger {
        type Output = MultiPrecisionInteger;

        #[inline]
        fn shl(self, rhs: usize) -> Self::Output {
            MultiPrecisionInteger(MultiPrecisionInteger::shl_raw(
                self.0.clone(),
                u64::try_from(rhs).expect("shift amount does not fit in u64"),
            ))
        }
    }

    impl Shr<u32> for MultiPrecisionInteger {
        type Output = Self;

        #[inline]
        fn shr(self, rhs: u32) -> Self::Output {
            Self(MultiPrecisionInteger::shr_raw(
                self.into_raw(),
                u64::from(rhs),
            ))
        }
    }

    impl Shr<u32> for &MultiPrecisionInteger {
        type Output = MultiPrecisionInteger;

        #[inline]
        fn shr(self, rhs: u32) -> Self::Output {
            MultiPrecisionInteger(MultiPrecisionInteger::shr_raw(
                self.0.clone(),
                u64::from(rhs),
            ))
        }
    }

    impl Shl<u64> for MultiPrecisionInteger {
        type Output = Self;

        #[inline]
        fn shl(self, rhs: u64) -> Self::Output {
            Self(MultiPrecisionInteger::shl_raw(self.into_raw(), rhs))
        }
    }

    impl Shr<usize> for MultiPrecisionInteger {
        type Output = Self;

        #[inline]
        fn shr(self, rhs: usize) -> Self::Output {
            Self(MultiPrecisionInteger::shr_raw(
                self.into_raw(),
                u64::try_from(rhs).expect("shift amount does not fit in u64"),
            ))
        }
    }

    impl Shr<usize> for &MultiPrecisionInteger {
        type Output = MultiPrecisionInteger;

        #[inline]
        fn shr(self, rhs: usize) -> Self::Output {
            MultiPrecisionInteger(MultiPrecisionInteger::shr_raw(
                self.0.clone(),
                u64::try_from(rhs).expect("shift amount does not fit in u64"),
            ))
        }
    }

    #[cfg(all(feature = "integer-malachite", feature = "bincode"))]
    pub(crate) fn to_be_bytes(value: &MultiPrecisionInteger) -> Vec<u8> {
        value.to_string().into_bytes()
    }

    #[cfg(all(feature = "integer-malachite", feature = "bincode"))]
    pub(crate) fn from_be_bytes(bytes: &[u8]) -> Result<MultiPrecisionInteger, &'static str> {
        std::str::from_utf8(bytes)
            .ok()
            .and_then(|s| s.parse::<MultiPrecisionInteger>().ok())
            .ok_or("Failed to parse large integer")
    }

    #[cfg(feature = "integer-malachite")]
    pub(crate) struct BackendRandState {
        rng: rand::rngs::ThreadRng,
    }

    #[cfg(feature = "integer-malachite")]
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

    #[cfg(all(feature = "integer-gmp", feature = "bincode"))]
    pub(crate) fn to_be_bytes(value: &MultiPrecisionInteger) -> Vec<u8> {
        value.0.to_digits::<u8>(rug::integer::Order::MsfBe)
    }

    #[cfg(all(feature = "integer-gmp", feature = "bincode"))]
    pub(crate) fn from_be_bytes(bytes: &[u8]) -> Result<MultiPrecisionInteger, &'static str> {
        Ok(MultiPrecisionInteger(rug::Integer::from_digits(
            bytes,
            rug::integer::Order::MsfBe,
        )))
    }

    #[cfg(feature = "integer-gmp")]
    pub(crate) struct BackendRandState(rug::rand::RandState<'static>);

    #[cfg(feature = "integer-gmp")]
    impl BackendRandState {
        pub(crate) fn new(seed: u128) -> Self {
            let mut state = rug::rand::RandState::new();
            state.seed(&rug::Integer::from(seed));
            Self(state)
        }

        pub(crate) fn below(&mut self, modulus: &MultiPrecisionInteger) -> MultiPrecisionInteger {
            MultiPrecisionInteger(modulus.0.clone().random_below(&mut self.0))
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn mod_u_matches_euclidean_remainder() {
            let large = (MultiPrecisionInteger::from(1) << 1024usize)
                + MultiPrecisionInteger::from(0xdead_beefu32);
            let values = [
                MultiPrecisionInteger::from(0),
                MultiPrecisionInteger::from(23),
                MultiPrecisionInteger::from(-23),
                large.clone(),
                -large,
            ];
            let moduli = [1, 2, 3, 17, 2_147_483_647, u32::MAX];

            for value in values {
                for modulus in moduli {
                    let expected = value
                        .rem_euc(MultiPrecisionInteger::from(modulus))
                        .to_u64()
                        .unwrap() as u32;
                    assert_eq!(value.mod_u(modulus), expected, "{value} mod {modulus}");
                }
            }
        }

        #[test]
        fn mod_u_returns_positive_representative_for_negative_input() {
            assert_eq!(MultiPrecisionInteger::from(-23).mod_u(10), 7);
        }

        #[test]
        fn mod_u64_matches_euclidean_remainder() {
            let large = (MultiPrecisionInteger::from(1) << 1024usize)
                + MultiPrecisionInteger::from(0xdead_beefu32);
            let values = [
                MultiPrecisionInteger::from(0),
                MultiPrecisionInteger::from(23),
                MultiPrecisionInteger::from(-23),
                large.clone(),
                -large,
            ];
            let moduli = [
                1,
                2,
                3,
                17,
                2_147_483_647,
                18_346_744_073_709_552_031,
                u64::MAX,
            ];

            for value in values {
                for modulus in moduli {
                    let expected = value
                        .rem_euc(MultiPrecisionInteger::from(modulus))
                        .to_u64()
                        .unwrap();
                    assert_eq!(value.mod_u64(modulus), expected, "{value} mod {modulus}");
                }
            }
        }

        #[test]
        fn mod_u64_returns_positive_representative_for_negative_input() {
            assert_eq!(MultiPrecisionInteger::from(-23).mod_u64(10), 7);
        }

        #[test]
        fn truncating_remainder_owned_and_in_place_agree() {
            let large = (MultiPrecisionInteger::from(1) << 521usize)
                + MultiPrecisionInteger::from(0xdead_beefu32);
            let values = [large.clone(), -large, MultiPrecisionInteger::from(-23)];
            let divisors = [
                (MultiPrecisionInteger::from(1) << 193usize) + MultiPrecisionInteger::from(17),
                MultiPrecisionInteger::from(-10),
            ];

            for value in values {
                for divisor in &divisors {
                    let quotient = value.clone() / divisor.clone();
                    let expected = value.clone() - quotient * divisor;
                    let owned = value.clone() % divisor.clone();
                    let borrowed = value.clone() % divisor;
                    let mut assigned_owned = value.clone();
                    assigned_owned %= divisor.clone();
                    let mut assigned_borrowed = value.clone();
                    assigned_borrowed %= divisor;

                    assert_eq!(owned, expected);
                    assert_eq!(borrowed, expected);
                    assert_eq!(assigned_owned, expected);
                    assert_eq!(assigned_borrowed, expected);
                }
            }
        }

        #[test]
        #[should_panic]
        fn mod_u_rejects_zero_modulus() {
            MultiPrecisionInteger::from(1).mod_u(0);
        }

        #[test]
        #[should_panic]
        fn mod_u64_rejects_zero_modulus() {
            MultiPrecisionInteger::from(1).mod_u64(0);
        }

        #[cfg(feature = "gmp")]
        #[test]
        fn large_integer_cache_is_bounded_and_reused() {
            LARGE_INTEGER_CACHE.with(|cache| cache.borrow_mut().clear());

            let values = (0..(LARGE_INTEGER_CACHE_SIZE + 8))
                .map(|shift| MultiPrecisionInteger::from(1) << (256 + shift))
                .collect::<Vec<_>>();
            drop(values);

            LARGE_INTEGER_CACHE
                .with(|cache| assert_eq!(cache.borrow().len(), LARGE_INTEGER_CACHE_SIZE));

            let recycled = MultiPrecisionInteger::default();
            assert!(recycled.as_raw().capacity() >= 256);
            LARGE_INTEGER_CACHE
                .with(|cache| assert_eq!(cache.borrow().len(), LARGE_INTEGER_CACHE_SIZE - 1));
        }
    }
}

pub(crate) use implementation::{BackendRandState, Complete, RemRounding};
pub use implementation::{
    MultiPrecisionInteger, ParseMultiPrecisionIntegerError, RawMultiPrecisionInteger,
};

pub(crate) fn pow_ref_u32(base: &MultiPrecisionInteger, e: u32) -> MultiPrecisionInteger {
    base.clone().pow(e)
}

pub(crate) fn probably_prime(value: &MultiPrecisionInteger, reps: u32) -> Option<bool> {
    #[cfg(feature = "integer-gmp")]
    return Some(value.as_raw().is_probably_prime(reps) != rug::integer::IsPrime::No);
    #[cfg(feature = "integer-malachite")]
    {
        let _ = (value, reps);
        return None;
    }
}
#[cfg(feature = "integer-gmp")]
pub fn from_lsf_bytes(bytes: &[u8]) -> MultiPrecisionInteger {
    MultiPrecisionInteger::from_raw(rug::Integer::from_digits(bytes, rug::integer::Order::Lsf))
}

#[cfg(feature = "integer-malachite")]
pub fn from_lsf_bytes(bytes: &[u8]) -> MultiPrecisionInteger {
    let mut value = MultiPrecisionInteger::from(0u32);
    for &byte in bytes.iter().rev() {
        value = (value << 8usize) + u32::from(byte);
    }
    value
}
#[cfg(feature = "integer-gmp")]
pub fn write_lsf_bytes(value: &MultiPrecisionInteger, dest: &mut Vec<u8>) {
    let value = value.as_raw().as_abs();
    let num_digits = value.significant_digits::<u8>();
    let old_len = dest.len();
    dest.resize(old_len + num_digits, 0);
    value.write_digits(&mut dest[old_len..], rug::integer::Order::Lsf);
}

#[cfg(feature = "integer-malachite")]
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
#[cfg(feature = "integer-gmp")]
pub fn lsf_byte_size(value: &MultiPrecisionInteger) -> usize {
    value.as_raw().significant_digits::<u8>()
}

#[cfg(feature = "integer-malachite")]
pub fn lsf_byte_size(value: &MultiPrecisionInteger) -> usize {
    value
        .significant_bits()
        .div_ceil(u64::from(u8::BITS))
        .try_into()
        .expect("large integer byte length does not fit in usize")
}
pub fn to_lsf_bytes(value: &MultiPrecisionInteger) -> Vec<u8> {
    let mut bytes = Vec::new();
    write_lsf_bytes(value, &mut bytes);
    bytes
}
#[cfg(feature = "integer-gmp")]
pub fn from_digits_radix(digits: &[u8], radix: u32, is_negative: bool) -> MultiPrecisionInteger {
    let mut value = rug::Integer::new();
    unsafe {
        value.assign_bytes_radix_unchecked(
            digits,
            i32::try_from(radix).expect("radix does not fit in i32"),
            is_negative,
        );
    }
    MultiPrecisionInteger::from_raw(value)
}

#[cfg(feature = "integer-malachite")]
pub fn from_digits_radix(digits: &[u8], radix: u32, is_negative: bool) -> MultiPrecisionInteger {
    let mut value = MultiPrecisionInteger::from(0u32);
    for &digit in digits {
        value *= radix;
        value += u32::from(digit);
    }

    if is_negative { -value } else { value }
}
#[cfg(feature = "bincode")]
pub(crate) use implementation::{from_be_bytes, to_be_bytes};
