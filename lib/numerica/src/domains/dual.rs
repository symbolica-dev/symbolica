//! A hyperdual number is a number that keep track of (higher order) derivatives in one or multiple
//! variables.
//! ```
//! use numerica::{
//! create_hyperdual_from_components, create_hyperdual_single_derivative,
//! domains::{float::FloatLike, rational::Rational},
//! };
//!
//! create_hyperdual_single_derivative!(SingleDual, 3);
//!
//! create_hyperdual_from_components!(
//! Dual,
//! [
//!     [0, 0, 0],
//!     [1, 0, 0],
//!     [0, 1, 0],
//!     [0, 0, 1],
//!     [1, 1, 0],
//!     [1, 0, 1],
//!     [0, 1, 1],
//!     [1, 1, 1],
//!     [2, 0, 0]
//! ]
//! );
//!
//! fn main() {
//! let x = Dual::<Rational>::new_variable(0, (1, 1).into());
//! let y = Dual::new_variable(1, (2, 1).into());
//! let z = Dual::new_variable(2, (3, 1).into());
//!
//! let t3 = x * y * z;
//!
//! println!("{}", t3.inv());
//! }
//! ```

use std::sync::Arc;

use crate::domains::{
    float::{Constructible, FloatLike, Real, SingleFloat},
    rational::Rational,
};

/// Get the size of a dual number with the given maximum depth per variable.
pub const fn get_dual_size<const N: usize>(x: &[usize; N]) -> usize {
    let mut c = 1;
    let mut i = 0;
    while i < x.len() {
        c *= x[i] + 1;
        i += 1;
    }
    c
}

/// Get the dual components for a single derivative in multiple variables.
pub const fn get_single_derivative_dual_components<const N: usize, const N_PLUS_ONE: usize>()
-> [[usize; N]; N_PLUS_ONE] {
    let mut res = [[0; N]; N_PLUS_ONE];

    let mut i = 1;
    while i < res.len() {
        res[i][i - 1] = 1;
        i += 1;
    }
    res
}

// Get the dual components for a given shape.
pub const fn get_dual_components<const N: usize, const K: usize>(
    x: &[usize; N],
) -> [[usize; N]; K] {
    let mut res = [[0; N]; K];
    let mut cur = [0; N];

    let mut i = 0;
    let mut k;

    let mut max_depth = 0;
    while i < x.len() {
        max_depth += x[i];
        i += 1;
    }
    i = 0;

    let mut d = 0;
    let mut depth_sum = 0;
    while d <= max_depth {
        k = 0;
        'done: loop {
            if depth_sum == d {
                res[i] = cur;
                i += 1;
            }

            while cur[k] + 1 > x[k] {
                depth_sum -= cur[k];
                cur[k] = 0;
                if k + 1 == N {
                    break 'done;
                }

                k += 1;
            }
            cur[k] += 1;
            depth_sum += 1;
            k = 0;
        }

        d += 1;
    }
    res
}

/// Get the index in `r` of the multiplication of `a` and `b` in `r`.
pub const fn get_multiplication_index<const N: usize, const C: usize>(
    r: &[[usize; N]; C],
    a: usize,
    b: usize,
) -> Option<usize> {
    let mut sum = r[a];
    let mut i = 0;
    while i < N {
        sum[i] += r[b][i];
        i += 1;
    }

    i = 0;

    // find the index of the sum of powers
    while i < r.len() {
        let mut j = 0;
        while j < N {
            if r[i][j] != sum[j] {
                break;
            }
            j += 1;
        }
        if j == N {
            break;
        }
        i += 1;
    }

    if i == r.len() { None } else { Some(i) }
}

/// Get the size of the multiplication table.
pub const fn get_mult_table_size<const N: usize, const C: usize>(r: &[[usize; N]; C]) -> usize {
    let mut i = 0;
    let mut ri = 0;
    while i < r.len() {
        let mut j = 1; // skip first entry
        while j < r.len() {
            if get_multiplication_index::<N, C>(r, i, j).is_some() {
                ri += 1;
            }
            j += 1;
        }
        i += 1;
    }

    ri
}

// Construct the multiplication table for a dual.
pub const fn get_mult_table<const N: usize, const C: usize, const T: usize>(
    r: &[[usize; N]; C],
) -> [(usize, usize, usize); T] {
    let mut res = [(0, 0, 0); T];

    let mut ri = 0;
    let mut i = 0;
    while i < r.len() {
        let mut j = 1; // skip first entry
        while j < r.len() {
            if let Some(index) = get_multiplication_index::<N, C>(r, i, j) {
                res[ri] = (i, j, index);
                ri += 1;
            }
            j += 1;
        }
        i += 1;
    }

    res
}

/// Create a new hyperdual number, with only single derivatives and with `$var` variables.
///
/// For example:
/// ```
/// # use numerica::{
/// # create_hyperdual_single_derivative,
/// # domains::{float::FloatLike, rational::Rational},
/// # };
/// create_hyperdual_single_derivative!(Dual, 3);
///
/// fn main() {
///     let x = Dual::<Rational>::new_variable(0, (1, 1).into());
///     let y = Dual::new_variable(1, (2, 1).into());
///     let z = Dual::new_variable(2, (3, 1).into());
///
///     let t3 = x * y * z;
///
///     println!("{}", t3.inv());
/// }
/// ```
#[macro_export]
macro_rules! create_hyperdual_single_derivative {
    ($t: ident, $var: expr) => {
        $crate::create_hyperdual_from_components!(
            $t,
            $crate::domains::dual::get_single_derivative_dual_components::<{ $var }, { $var + 1 }>(
            )
        );
    };
}

/// Create a new hyperdual number, with a given derivative depth per variable.
#[macro_export]
macro_rules! create_hyperdual_from_depths {
    ($t: ident, $var: expr) => {
        $crate::create_hyperdual_from_components!(
            $t,
            $crate::domains::dual::get_dual_components::<
                { $var.len() },
                { $crate::domains::dual::get_dual_size(&$var) },
            >(&$var)
        );
    };
}

/// Describes the shape of dual numbers.
pub trait DualNumberStructure {
    // Get the length of the dual number.
    fn get_len(&self) -> usize;
    /// Get the shape of the dual number.
    fn get_shape(&self) -> Vec<&[usize]>;
    /// Get the multiplication table of the dual number.
    fn get_multiplication_table(&self) -> &[(usize, usize, usize)];
    /// Get the maximum derivative depth of the dual number.
    fn get_max_depth(&self) -> usize;
}

/// Create a new hyperdual number, from a specification of the components.
/// The first components must be the real value and the single derivatives of the variables in order.
///
/// For example:
/// ```
/// # use numerica::{
/// # create_hyperdual_from_components,
/// # domains::{float::FloatLike, rational::Rational},
/// # };
/// create_hyperdual_from_components!(
///     Dual,
///     [
///         [0, 0, 0],
///         [1, 0, 0],
///         [0, 1, 0],
///         [0, 0, 1],
///         [1, 1, 0],
///         [1, 0, 1],
///         [0, 1, 1],
///         [1, 1, 1],
///         [2, 0, 0]
///     ]
/// );
///
/// fn main() {
///     let x = Dual::<Rational>::new_variable(0, (1, 1).into());
///     let y = Dual::new_variable(1, (2, 1).into());
///     let z = Dual::new_variable(2, (3, 1).into());
///
///     let t3 = x * y * z;
///
///     println!("{}", t3.inv());
/// }
/// ```
#[macro_export]
macro_rules! create_hyperdual_from_components {
    ($t: ident, $var: expr) => {
        /// A hyperdual with a given shape.
        /// The multiplication table is precomputed and (partially) unrolled
        /// for performance.
        #[derive(Clone, Debug)]
        pub struct $t<T> {
            pub values: [T; $var.len()],
        }

        impl<T> $t<T> {
            /// The shape of the hyperdual.
            pub const SHAPE: [[usize; { $var[0].len() }]; { $var.len() }] = $var;
            const MAX_POW: usize = {
                let mut max_pow = 0;
                let mut i = 0;
                while i < $var.len() {
                    let mut c = 0;
                    let mut j = 0;
                    while j < $var[i].len() {
                        c += $var[i][j];
                        j += 1;
                    }
                    if c > max_pow {
                        max_pow = c;
                    }
                    i += 1;
                }
                max_pow
            };
            const MULT_TABLE: [(usize, usize, usize); {
                $crate::domains::dual::get_mult_table_size(&$var)
            }] = $crate::domains::dual::get_mult_table(&$var);
        }

        impl<T> $crate::domains::dual::DualNumberStructure for $t<T> {
            fn get_len(&self) -> usize {
                self.values.len()
            }

            fn get_shape(&self) -> Vec<&[usize]> {
                Self::SHAPE.iter().map(|s| s.as_slice()).collect()
            }

            fn get_multiplication_table(&self) -> &[(usize, usize, usize)] {
                &Self::MULT_TABLE
            }

            fn get_max_depth(&self) -> usize {
                Self::MAX_POW
            }
        }

        impl<T: $crate::domains::float::FloatLike> $t<T> {
            /// Create a new dual variable for the variable `var`, i.e. `value + 1*ε_var`,
            /// inheriting the floating point settings from `self`.
            #[allow(dead_code)]
            pub fn variable(&self, var: usize, value: T) -> Self {
                let mut values = std::array::from_fn(|_| self.values[0].zero());
                values[0] = value;
                values[1 + var] = self.values[0].one();
                $t { values }
            }
        }

        impl<T: $crate::domains::float::Constructible> $t<T> {
            /// Create a new dual variable for the variable `var`, i.e. `value + 1*ε_var`.
            #[allow(dead_code)]
            pub fn new_variable(var: usize, value: T) -> Self {
                let mut values = std::array::from_fn(|_| T::new_zero());
                values[0] = value;
                values[1 + var] = T::new_one();
                $t { values }
            }
        }

        impl<T: for<'a> std::ops::MulAssign<&'a T>> std::ops::Mul<&T> for $t<T> {
            type Output = Self;

            fn mul(mut self, other: &T) -> Self::Output {
                for s in self.values.iter_mut() {
                    *s *= other;
                }

                self
            }
        }

        impl<T: $crate::domains::float::FloatLike> std::fmt::Display for $t<T> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                use std::fmt::Write;
                for (i, (v, s)) in self.values.iter().zip(Self::SHAPE).enumerate() {
                    if i > 0 && i < self.values.len() {
                        f.write_char('+')?;
                    }
                    f.write_char('(')?;
                    std::fmt::Display::fmt(v, f)?;
                    f.write_char(')')?;
                    if i > 0 {
                        f.write_char('*')?;
                    }
                    for (i, p) in s.iter().enumerate() {
                        if *p > 0 {
                            f.write_char('ε')?;
                            std::fmt::Display::fmt(&i, f)?;
                            if *p > 1 {
                                f.write_fmt(format_args!("^{}", p))?;
                            }
                        }
                    }
                }

                Ok(())
            }
        }

        impl<T: $crate::domains::float::FloatLike> std::fmt::LowerExp for $t<T> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                use std::fmt::Write;
                for (i, (v, s)) in self.values.iter().zip(Self::SHAPE).enumerate() {
                    if i > 0 && i < self.values.len() {
                        f.write_char('+')?;
                    }
                    f.write_char('(')?;
                    std::fmt::LowerExp::fmt(v, f)?;
                    f.write_char(')')?;
                    if i > 0 {
                        f.write_char('*')?;
                    }
                    for (i, p) in s.iter().enumerate() {
                        if *p > 0 {
                            f.write_char('ε')?;
                            std::fmt::Display::fmt(&i, f)?;
                            if *p > 1 {
                                f.write_fmt(format_args!("^{}", p))?;
                            }
                        }
                    }
                }

                Ok(())
            }
        }

        impl<T: $crate::domains::float::FloatLike> PartialEq for $t<T> {
            fn eq(&self, other: &Self) -> bool {
                // only compare the real part
                self.values[0].eq(&other.values[0])
            }
        }

        impl<T: $crate::domains::float::FloatLike + PartialOrd> PartialOrd for $t<T> {
            fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
                self.values[0].partial_cmp(&other.values[0])
            }
        }

        impl<T: $crate::domains::float::FloatLike> std::ops::Neg for $t<T> {
            type Output = Self;

            #[inline]
            fn neg(self) -> Self::Output {
                let mut res = self.clone();
                for (s, o) in res.values.iter_mut().zip(self.values.into_iter()) {
                    *s = -o;
                }
                res
            }
        }

        impl<T: $crate::domains::float::FloatLike> std::ops::Add<&T> for $t<T> {
            type Output = Self;

            #[inline]
            fn add(mut self, rhs: &T) -> Self::Output {
                self.values[0] += rhs;
                self
            }
        }

        impl<T: $crate::domains::float::FloatLike> std::ops::Add<T> for $t<T> {
            type Output = Self;

            #[inline]
            fn add(mut self, rhs: T) -> Self::Output {
                self.values[0] += rhs;
                self
            }
        }

        impl<T: $crate::domains::float::FloatLike> std::ops::Sub<&T> for $t<T> {
            type Output = Self;

            #[inline]
            fn sub(mut self, rhs: &T) -> Self::Output {
                self.values[0] -= rhs;
                self
            }
        }

        impl<T: $crate::domains::float::FloatLike> std::ops::Sub<T> for $t<T> {
            type Output = Self;

            #[inline]
            fn sub(mut self, rhs: T) -> Self::Output {
                self.values[0] -= rhs;
                self
            }
        }

        impl<T: $crate::domains::float::FloatLike> std::ops::Add<&$t<T>> for $t<T> {
            type Output = Self;

            #[inline]
            fn add(mut self, rhs: &Self) -> Self::Output {
                for (s, o) in self.values.iter_mut().zip(&rhs.values) {
                    *s += o;
                }

                self
            }
        }

        impl<T: $crate::domains::float::FloatLike> std::ops::Add<$t<T>> for $t<T> {
            type Output = Self;

            #[inline]
            fn add(mut self, rhs: Self) -> Self::Output {
                for (s, o) in self.values.iter_mut().zip(rhs.values.into_iter()) {
                    *s += o;
                }

                self
            }
        }

        impl<T: $crate::domains::float::FloatLike> std::ops::Sub<&$t<T>> for $t<T> {
            type Output = Self;

            #[inline]
            fn sub(mut self, rhs: &Self) -> Self::Output {
                for (s, o) in self.values.iter_mut().zip(&rhs.values) {
                    *s -= o;
                }

                self
            }
        }

        impl<T: $crate::domains::float::FloatLike> std::ops::Sub<$t<T>> for $t<T> {
            type Output = Self;

            #[inline]
            fn sub(mut self, rhs: Self) -> Self::Output {
                for (s, o) in self.values.iter_mut().zip(rhs.values.into_iter()) {
                    *s -= o;
                }

                self
            }
        }

        impl<T: $crate::domains::float::FloatLike> std::ops::Mul<&$t<T>> for $t<T> {
            type Output = Self;

            #[inline]
            fn mul(self, rhs: &Self) -> Self::Output {
                let mut res = self.clone();

                for s in &mut res.values {
                    *s *= rhs.values[0].clone();
                }

                // the compiler will (partially) unroll this
                for (si, oi, index) in Self::MULT_TABLE.iter() {
                    unsafe {
                        *res.values.get_unchecked_mut(*index) +=
                            self.values.get_unchecked(*si).clone() * rhs.values.get_unchecked(*oi);
                    }
                }

                res
            }
        }

        impl<T: $crate::domains::float::FloatLike> std::ops::Mul<$t<T>> for $t<T> {
            type Output = Self;

            #[inline]
            fn mul(self, rhs: Self) -> Self::Output {
                self * &rhs
            }
        }

        impl<T: $crate::domains::float::FloatLike> std::ops::Div<&$t<T>> for $t<T> {
            type Output = Self;

            #[inline]
            fn div(self, rhs: &Self) -> Self::Output {
                use $crate::domains::float::FloatLike;
                self * rhs.inv()
            }
        }

        impl<T: $crate::domains::float::FloatLike> std::ops::Div<$t<T>> for $t<T> {
            type Output = Self;

            #[inline]
            fn div(self, rhs: Self) -> Self::Output {
                use $crate::domains::float::FloatLike;
                self * rhs.inv()
            }
        }

        impl<T: $crate::domains::float::FloatLike> std::ops::AddAssign<&T> for $t<T> {
            #[inline]
            fn add_assign(&mut self, rhs: &T) {
                self.values[0] += rhs;
            }
        }

        impl<T: $crate::domains::float::FloatLike> std::ops::AddAssign<T> for $t<T> {
            #[inline]
            fn add_assign(&mut self, rhs: T) {
                self.values[0] += rhs;
            }
        }

        impl<T: $crate::domains::float::FloatLike> std::ops::SubAssign<&T> for $t<T> {
            #[inline]
            fn sub_assign(&mut self, rhs: &T) {
                self.values[0] -= rhs;
            }
        }

        impl<T: $crate::domains::float::FloatLike> std::ops::SubAssign<T> for $t<T> {
            #[inline]
            fn sub_assign(&mut self, rhs: T) {
                self.values[0] -= rhs;
            }
        }

        impl<T: $crate::domains::float::FloatLike> std::ops::AddAssign<&$t<T>> for $t<T> {
            #[inline]
            fn add_assign(&mut self, rhs: &$t<T>) {
                for (s, o) in self.values.iter_mut().zip(&rhs.values) {
                    *s += o;
                }
            }
        }

        impl<T: $crate::domains::float::FloatLike> std::ops::AddAssign<$t<T>> for $t<T> {
            #[inline]
            fn add_assign(&mut self, rhs: $t<T>) {
                self.add_assign(&rhs)
            }
        }

        impl<T: $crate::domains::float::FloatLike> std::ops::SubAssign<&$t<T>> for $t<T> {
            #[inline]
            fn sub_assign(&mut self, rhs: &$t<T>) {
                for (s, o) in self.values.iter_mut().zip(&rhs.values) {
                    *s -= o;
                }
            }
        }

        impl<T: $crate::domains::float::FloatLike> std::ops::SubAssign<$t<T>> for $t<T> {
            #[inline]
            fn sub_assign(&mut self, rhs: $t<T>) {
                self.sub_assign(&rhs)
            }
        }

        impl<T: $crate::domains::float::FloatLike> std::ops::MulAssign<&$t<T>> for $t<T> {
            #[inline]
            fn mul_assign(&mut self, rhs: &$t<T>) {
                *self = self.clone() * rhs.clone();
            }
        }

        impl<T: $crate::domains::float::FloatLike> std::ops::MulAssign<$t<T>> for $t<T> {
            #[inline]
            fn mul_assign(&mut self, rhs: $t<T>) {
                *self = self.clone() * rhs;
            }
        }

        impl<T: $crate::domains::float::FloatLike> std::ops::DivAssign<&$t<T>> for $t<T> {
            #[inline]
            fn div_assign(&mut self, rhs: &$t<T>) {
                use $crate::domains::float::FloatLike;
                *self = rhs.inv() * &*self;
            }
        }

        impl<T: $crate::domains::float::FloatLike> std::ops::DivAssign<$t<T>> for $t<T> {
            #[inline]
            fn div_assign(&mut self, rhs: $t<T>) {
                use $crate::domains::float::FloatLike;
                *self = rhs.inv() * &*self;
            }
        }

        impl<T: $crate::domains::float::FloatLike> $crate::domains::float::FloatLike for $t<T> {
            #[inline(always)]
            fn mul_add(&self, a: &Self, b: &Self) -> Self {
                self.clone() * a + b
            }

            #[inline(always)]
            fn neg(&self) -> Self {
                -self.clone()
            }

            #[inline(always)]
            fn zero(&self) -> Self {
                $t {
                    values: std::array::from_fn(|_| self.values[0].zero()),
                }
            }

            #[inline(always)]
            fn new_zero() -> Self {
                $t {
                    values: std::array::from_fn(|_| T::new_zero()),
                }
            }

            #[inline(always)]
            fn one(&self) -> Self {
                let mut res = self.zero();
                res.values[0] = res.values[0].one();
                res
            }

            #[inline]
            fn pow(&self, e: u64) -> Self {
                // TODO: use binary exponentiation
                let mut res = self.clone();
                for _ in 1..e {
                    res = res.clone() * self;
                }
                res
            }

            #[inline(always)]
            fn inv(&self) -> Self {
                let e = self.values[0].inv();
                let mut r = self.clone() * &e;
                r.values[0] = r.values[0].zero();

                let mut accum = r.one();
                let mut res = r.one();

                for i in 1..Self::MAX_POW + 1 {
                    accum *= &r;
                    if i % 2 == 0 {
                        res += accum.clone()
                    } else {
                        res -= accum.clone()
                    }
                }

                res * &e
            }

            #[inline(always)]
            fn from_usize(&self, a: usize) -> Self {
                let mut res = self.zero();
                res.values[0] = res.values[0].from_usize(a);
                res
            }

            #[inline(always)]
            fn from_i64(&self, a: i64) -> Self {
                let mut res = self.zero();
                res.values[0] = res.values[0].from_i64(a);
                res
            }

            #[inline(always)]
            fn get_precision(&self) -> u32 {
                self.values[0].get_precision()
            }

            #[inline(always)]
            fn get_epsilon(&self) -> f64 {
                self.values[0].get_epsilon()
            }

            #[inline(always)]
            fn fixed_precision(&self) -> bool {
                self.values[0].fixed_precision()
            }

            fn sample_unit<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Self {
                let mut res = self.zero();
                res.values[0] = self.values[0].sample_unit(rng);
                res
            }

            #[inline(always)]
            fn is_fully_zero(&self) -> bool {
                self.values.iter().all(|v| v.is_fully_zero())
            }
        }

        impl<T: $crate::domains::float::SingleFloat> $crate::domains::float::SingleFloat for $t<T> {
            #[inline(always)]
            fn is_zero(&self) -> bool {
                self.values[0].is_zero()
            }

            #[inline(always)]
            fn is_one(&self) -> bool {
                self.values[0].is_one()
            }

            #[inline(always)]
            fn is_finite(&self) -> bool {
                self.values[0].is_finite()
            }

            #[inline(always)]
            fn from_rational(&self, rat: &Rational) -> Self {
                let mut res = self.zero();
                res.values[0] = self.values[0].from_rational(rat);
                res
            }
        }

        impl<T: $crate::domains::float::Constructible> $crate::domains::float::Constructible
            for $t<T>
        {
            #[inline(always)]
            fn new_one() -> Self {
                let mut res = <Self as $crate::domains::float::FloatLike>::new_zero();
                res.values[0] = T::new_one();
                res
            }

            #[inline(always)]
            fn new_from_usize(a: usize) -> Self {
                let mut res = <Self as $crate::domains::float::FloatLike>::new_zero();
                res.values[0] = T::new_from_usize(a);
                res
            }

            #[inline(always)]
            fn new_from_i64(a: i64) -> Self {
                let mut res = <Self as $crate::domains::float::FloatLike>::new_zero();
                res.values[0] = T::new_from_i64(a);
                res
            }

            #[inline(always)]
            fn new_sample_unit<R: rand::Rng + ?Sized>(rng: &mut R) -> Self {
                let mut res = <Self as $crate::domains::float::FloatLike>::new_zero();
                res.values[0] = T::new_sample_unit(rng);
                res
            }
        }

        impl<T: $crate::domains::float::Real> $crate::domains::float::Real for $t<T> {
            #[inline(always)]
            fn pi(&self) -> Self {
                let mut res = self.zero();
                res.values[0] = self.values[0].pi();
                res
            }

            #[inline(always)]
            fn e(&self) -> Self {
                let mut res = self.zero();
                res.values[0] = self.values[0].e();
                res
            }

            #[inline(always)]
            fn euler(&self) -> Self {
                let mut res = self.zero();
                res.values[0] = self.values[0].euler();
                res
            }

            #[inline(always)]
            fn phi(&self) -> Self {
                let mut res = self.zero();
                res.values[0] = self.values[0].phi();
                res
            }

            #[inline(always)]
            fn i(&self) -> Option<Self> {
                let mut res = self.zero();
                res.values[0] = self.values[0].i()?;
                Some(res)
            }

            #[inline(always)]
            fn conj(&self) -> Self {
                // assume variable is real
                let mut r = self.clone();
                for x in &mut r.values {
                    *x = x.conj();
                }
                r
            }

            #[inline(always)]
            fn norm(&self) -> Self {
                let n = self.values[0].norm();
                if n == self.values[0] {
                    return self.clone();
                } else {
                    let scale = n / &self.values[0];
                    self.clone() * &scale
                }
            }

            #[inline(always)]
            fn sqrt(&self) -> Self {
                use $crate::domains::float::FloatLike;
                let e = self.values[0].sqrt();

                let norm = self.values[0].inv(); // TODO: check 0
                let mut r = self.clone() * &norm;
                r.values[0] = self.values[0].zero();

                let mut accum = self.one();
                let mut res = self.one();
                let mut num = self.values[0].one();

                let mut scale = 1;
                for p in 1..Self::MAX_POW + 1 {
                    scale *= p;
                    num = num.clone() * (num.from_usize(2).inv() - &num.from_usize(p as usize - 1));
                    accum *= &r;
                    res += accum.clone() * &num * &num.from_usize(scale).inv();
                }

                res * &e
            }

            #[inline(always)]
            fn log(&self) -> Self {
                let e = self.values[0].log();

                let norm = self.values[0].inv(); // TODO: check 0
                let mut r = self.clone() * &norm;
                r.values[0] = self.values[0].zero();

                let mut accum = r.clone();
                let mut res = self.one();
                res.values[0] = e.clone();

                let mut scale = -1;
                for p in 1..Self::MAX_POW + 1 {
                    scale *= -1;
                    res += accum.clone() * &e.from_i64(p as i64 * scale).inv();
                    accum *= &r;
                }

                res
            }

            #[inline(always)]
            fn exp(&self) -> Self {
                let e = self.values[0].exp();
                let mut res = self.one();

                let mut r = self.clone();
                r.values[0] = self.values[0].zero();
                let mut accum = self.one();
                let mut scale = 1;
                for p in 0..Self::MAX_POW {
                    scale *= p + 1;
                    accum *= &r; // TODO: many multiplications with 0
                    res += accum.clone() * &e.from_usize(scale).inv();
                }

                res * &e
            }

            #[inline(always)]
            fn sin(&self) -> Self {
                let s = self.values[0].sin();
                let c = self.values[0].cos();

                let mut p = self.clone();
                p.values[0] = self.values[0].zero();

                let mut e = self.one();
                e.values[0] = s.clone();
                let mut sp = p.clone();
                let mut scale = 1;
                for i in 1..Self::MAX_POW + 1 {
                    scale *= i;
                    let mut b = if i % 2 == 1 { c.clone() } else { s.clone() };

                    if i % 4 >= 2 {
                        b = b.neg();
                    }

                    let s = sp.clone() * &b * &e.from_usize(scale).inv();

                    sp *= &p;

                    e += s;
                }

                e
            }

            #[inline(always)]
            fn cos(&self) -> Self {
                let s = self.values[0].sin();
                let c = self.values[0].cos();

                let mut p = self.clone();
                p.values[0] = self.values[0].zero();

                let mut e = self.one();
                e.values[0] = c.clone();
                let mut sp = p.clone();
                let mut scale = 1;
                for i in 1..Self::MAX_POW + 1 {
                    scale *= i;
                    let mut b = if i % 2 == 1 { s.clone() } else { -c.clone() };

                    if i % 4 < 2 {
                        b = b.neg();
                    }

                    let s = sp.clone() * &b * &e.from_usize(scale).inv();

                    sp *= &p;

                    e += s;
                }

                e
            }

            #[inline(always)]
            fn tan(&self) -> Self {
                // TODO: improve
                self.sin() / self.cos()
            }

            #[inline(always)]
            fn asin(&self) -> Self {
                unimplemented!()
            }

            #[inline(always)]
            fn acos(&self) -> Self {
                unimplemented!()
            }

            #[inline(always)]
            fn atan2(&self, _x: &Self) -> Self {
                unimplemented!()
            }

            #[inline(always)]
            fn sinh(&self) -> Self {
                let s = self.values[0].sinh();
                let c = self.values[0].cosh();

                let mut p = self.clone();
                p.values[0] = self.values[0].zero();

                let mut e = self.one();
                e.values[0] = s.clone();
                let mut sp = p.clone();
                let mut scale = 1;
                for i in 1..Self::MAX_POW + 1 {
                    scale *= i;
                    let b = if i % 2 == 1 { &c } else { &s };

                    let s = sp.clone() * b * &e.from_usize(scale).inv();

                    sp *= &p;

                    e += s;
                }

                e
            }

            #[inline(always)]
            fn cosh(&self) -> Self {
                let s = self.values[0].sinh();
                let c = self.values[0].cosh();

                let mut p = self.clone();
                p.values[0] = self.values[0].zero();

                let mut e = self.one();
                e.values[0] = c.clone();
                let mut sp = p.clone();
                let mut scale = 1;
                for i in 1..Self::MAX_POW + 1 {
                    scale *= i;
                    let b = if i % 2 == 1 { &s } else { &c };

                    let s = sp.clone() * b * &e.from_usize(scale).inv();

                    sp *= &p;

                    e += s;
                }

                e
            }

            #[inline(always)]
            fn tanh(&self) -> Self {
                self.sinh() / self.cosh()
            }

            #[inline(always)]
            fn asinh(&self) -> Self {
                unimplemented!()
            }

            #[inline(always)]
            fn acosh(&self) -> Self {
                unimplemented!()
            }

            #[inline(always)]
            fn atanh(&self) -> Self {
                unimplemented!()
            }

            #[inline]
            fn powf(&self, e: &Self) -> Self {
                // TODO: improve
                (self.log() * e).exp()
            }
        }
    };
}

/// The shape of a dual number.
#[derive(Clone, Debug, PartialEq, Eq)]
struct DualShape {
    shape: Vec<Vec<usize>>,
    max_pow: usize,
    mult_table: Vec<(usize, usize, usize)>,
}

/// A hyperdual with a given shape determined at runtime.
/// Alternatively, you can use the macros [create_hyperdual_single_derivative] or [create_hyperdual_from_components] to create types
/// with precomputed multiplication tables for better performance.
///
/// # Example
/// ```
/// # use numerica::domains::dual::HyperDual;
/// # use numerica::domains::float::FloatLike;
/// # use numerica::domains::rational::Rational;
/// let dual = HyperDual::<Rational>::new_single_derivative(3);
/// let x = dual.variable(0, (1, 1).into());
/// let y = dual.variable(1, (2, 1).into());
/// let z = dual.variable(2, (3, 1).into());
///
/// let t3 = x * y * z;
/// let one = t3.clone() * t3.inv();
///
/// assert_eq!(one, t3.one());
/// ```
#[derive(Clone, Debug)]
pub struct HyperDual<T> {
    pub values: Vec<T>,
    shape: Arc<DualShape>,
}

impl<T> HyperDual<T> {
    /// Create a new dual with a given shape. If you already have a dual with the same shape,
    /// consider using methods such as [Dual::variable] to inherit the shape and floating point settings.
    pub fn from_values(shape: Vec<Vec<usize>>, values: Vec<T>) -> Self {
        if shape.iter().any(|s| s.len() != shape[0].len()) {
            panic!(
                "All shape vectors must have the same length: got {:?}",
                shape
            );
        }

        if shape.len() != values.len() {
            panic!(
                "Shape and values must have the same length: got {} and {}",
                shape.len(),
                values.len()
            );
        }

        let max_pow = shape
            .iter()
            .map(|s| s.iter().sum::<usize>())
            .max()
            .unwrap_or(0);

        let mut mult_table = vec![];

        let mut sum = vec![0; shape[0].len()];
        for (i, vi) in shape.iter().enumerate() {
            for (j, vj) in shape.iter().enumerate().skip(1) {
                for (s, (vii, vjj)) in sum.iter_mut().zip(vi.iter().zip(vj.iter())) {
                    *s = vii + vjj;
                }

                if let Some(p) = shape.iter().position(|s| s == &sum) {
                    mult_table.push((i, j, p));
                }
            }
        }

        HyperDual {
            values,
            shape: Arc::new(DualShape {
                shape,
                max_pow,
                mult_table,
            }),
        }
    }
}

impl<T: Default> HyperDual<T> {
    /// Create a new dual with a given shape. If you already have a dual with the same shape,
    /// consider using methods such as [Dual::variable] to inherit the shape and floating point settings.
    pub fn new(shape: Vec<Vec<usize>>) -> Self {
        let values = (0..shape.len()).map(|_| T::default()).collect();
        Self::from_values(shape, values)
    }

    /// Create a new dual number structure for single derivatives with `num_vars` variables,
    pub fn new_single_derivative(num_vars: usize) -> Self {
        let mut shape = vec![];
        shape.push(vec![0; num_vars]);
        for i in 0..num_vars {
            let mut s = vec![0; num_vars];
            s[i] = 1;
            shape.push(s);
        }
        Self::new(shape)
    }
}

impl<T> DualNumberStructure for HyperDual<T> {
    fn get_len(&self) -> usize {
        self.values.len()
    }

    fn get_shape(&self) -> Vec<&[usize]> {
        self.shape.shape.iter().map(|s| s.as_slice()).collect()
    }

    fn get_multiplication_table(&self) -> &[(usize, usize, usize)] {
        &self.shape.mult_table
    }

    fn get_max_depth(&self) -> usize {
        self.shape.max_pow
    }
}

impl<T: FloatLike> HyperDual<T> {
    /// Create a new dual variable for the variable `var`, i.e. `value + 1*ε_var`,
    /// inheriting the floating point settings from `self`.
    #[allow(dead_code)]
    pub fn variable(&self, var: usize, value: T) -> Self {
        let mut values = vec![self.values[0].zero(); self.values.len()];
        values[0] = value;
        values[1 + var] = self.values[0].one();
        HyperDual {
            values,
            shape: self.shape.clone(),
        }
    }
}

impl<T: for<'a> std::ops::MulAssign<&'a T>> std::ops::Mul<&T> for HyperDual<T> {
    type Output = Self;

    fn mul(mut self, other: &T) -> Self::Output {
        for s in self.values.iter_mut() {
            *s *= other;
        }

        self
    }
}

impl<T: FloatLike> std::fmt::Display for HyperDual<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use std::fmt::Write;
        for (i, (v, s)) in self.values.iter().zip(&self.shape.shape).enumerate() {
            if i > 0 && i < self.values.len() {
                f.write_char('+')?;
            }
            f.write_char('(')?;
            std::fmt::Display::fmt(v, f)?;
            f.write_char(')')?;
            if i > 0 {
                f.write_char('*')?;
            }
            for (i, p) in s.iter().enumerate() {
                if *p > 0 {
                    f.write_char('ε')?;
                    std::fmt::Display::fmt(&i, f)?;
                    if *p > 1 {
                        f.write_fmt(format_args!("^{}", p))?;
                    }
                }
            }
        }

        Ok(())
    }
}

impl<T: FloatLike> std::fmt::LowerExp for HyperDual<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use std::fmt::Write;
        for (i, (v, s)) in self.values.iter().zip(&self.shape.shape).enumerate() {
            if i > 0 && i < self.values.len() {
                f.write_char('+')?;
            }
            f.write_char('(')?;
            std::fmt::LowerExp::fmt(v, f)?;
            f.write_char(')')?;
            if i > 0 {
                f.write_char('*')?;
            }
            for (i, p) in s.iter().enumerate() {
                if *p > 0 {
                    f.write_char('ε')?;
                    std::fmt::Display::fmt(&i, f)?;
                    if *p > 1 {
                        f.write_fmt(format_args!("^{}", p))?;
                    }
                }
            }
        }

        Ok(())
    }
}

impl<T: FloatLike> PartialEq for HyperDual<T> {
    fn eq(&self, other: &Self) -> bool {
        // only compare the real part
        self.values[0].eq(&other.values[0])
    }
}

impl<T: FloatLike + PartialOrd> PartialOrd for HyperDual<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.values[0].partial_cmp(&other.values[0])
    }
}

impl<T: FloatLike> std::ops::Neg for HyperDual<T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        let mut res = self.clone();
        for (s, o) in res.values.iter_mut().zip(self.values.into_iter()) {
            *s = -o;
        }
        res
    }
}

impl<T: FloatLike> std::ops::Add<&HyperDual<T>> for HyperDual<T> {
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: &Self) -> Self::Output {
        if self.shape != rhs.shape {
            if self.shape.shape.is_empty() {
                return rhs.clone() + self;
            }

            if !rhs.shape.shape.is_empty() {
                panic!(
                    "Cannot add dual numbers with different shapes: {:?} vs {:?}",
                    self.shape.shape, rhs.shape.shape
                );
            }
        }

        for (s, o) in self.values.iter_mut().zip(&rhs.values) {
            *s += o;
        }

        self
    }
}

impl<T: FloatLike> std::ops::Add<HyperDual<T>> for HyperDual<T> {
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: Self) -> Self::Output {
        if self.shape != rhs.shape {
            if self.shape.shape.is_empty() {
                return rhs + self;
            }

            if !rhs.shape.shape.is_empty() {
                panic!(
                    "Cannot add dual numbers with different shapes: {:?} vs {:?}",
                    self.shape.shape, rhs.shape.shape
                );
            }
        }

        for (s, o) in self.values.iter_mut().zip(rhs.values.into_iter()) {
            *s += o;
        }

        self
    }
}

impl<T: FloatLike> std::ops::Sub<&HyperDual<T>> for HyperDual<T> {
    type Output = Self;

    #[inline]
    fn sub(mut self, rhs: &Self) -> Self::Output {
        if self.shape != rhs.shape {
            if self.shape.shape.is_empty() {
                return -rhs.clone() + self;
            }

            if !rhs.shape.shape.is_empty() {
                panic!(
                    "Cannot subtract dual numbers with different shapes: {:?} vs {:?}",
                    self.shape.shape, rhs.shape.shape
                );
            }
        }

        for (s, o) in self.values.iter_mut().zip(&rhs.values) {
            *s -= o;
        }

        self
    }
}

impl<T: FloatLike> std::ops::Sub<HyperDual<T>> for HyperDual<T> {
    type Output = Self;

    #[inline]
    fn sub(mut self, rhs: Self) -> Self::Output {
        if self.shape != rhs.shape {
            if self.shape.shape.is_empty() {
                return -rhs + self;
            }

            if !rhs.shape.shape.is_empty() {
                panic!(
                    "Cannot subtract dual numbers with different shapes: {:?} vs {:?}",
                    self.shape.shape, rhs.shape.shape
                );
            }
        }

        for (s, o) in self.values.iter_mut().zip(rhs.values.into_iter()) {
            *s -= o;
        }

        self
    }
}

impl<T: FloatLike> std::ops::Add<&T> for HyperDual<T> {
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: &T) -> Self::Output {
        self.values[0] += rhs;
        self
    }
}

impl<T: FloatLike> std::ops::Add<T> for HyperDual<T> {
    type Output = Self;

    #[inline]
    fn add(mut self, rhs: T) -> Self::Output {
        self.values[0] += rhs;
        self
    }
}

impl<T: FloatLike> std::ops::Sub<&T> for HyperDual<T> {
    type Output = Self;

    #[inline]
    fn sub(mut self, rhs: &T) -> Self::Output {
        self.values[0] -= rhs;
        self
    }
}

impl<T: FloatLike> std::ops::Sub<T> for HyperDual<T> {
    type Output = Self;

    #[inline]
    fn sub(mut self, rhs: T) -> Self::Output {
        self.values[0] -= rhs;
        self
    }
}

impl<T: FloatLike> std::ops::Mul<&HyperDual<T>> for HyperDual<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: &Self) -> Self::Output {
        if self.shape != rhs.shape {
            if self.shape.shape.is_empty() {
                // multiply a constant
                return rhs.clone() * &self.values[0];
            }

            if rhs.shape.shape.is_empty() {
                // multiply a constant
                return self * &rhs.values[0];
            }

            panic!(
                "Cannot multiply dual numbers with different shapes: {:?} vs {:?}",
                self.shape.shape, rhs.shape.shape
            );
        }

        let mut res = self.clone();

        for s in &mut res.values {
            *s *= rhs.values[0].clone();
        }

        // the compiler will (partially) unroll this
        for (si, oi, index) in self.shape.mult_table.iter() {
            unsafe {
                *res.values.get_unchecked_mut(*index) +=
                    self.values.get_unchecked(*si).clone() * rhs.values.get_unchecked(*oi);
            }
        }

        res
    }
}

impl<T: FloatLike> std::ops::Mul<HyperDual<T>> for HyperDual<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self * &rhs
    }
}

impl<T: FloatLike> std::ops::Div<&HyperDual<T>> for HyperDual<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: &Self) -> Self::Output {
        use FloatLike;
        self * rhs.inv()
    }
}

impl<T: FloatLike> std::ops::Mul<T> for HyperDual<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        self * &rhs
    }
}

impl<T: FloatLike> std::ops::Div<&T> for HyperDual<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: &T) -> Self::Output {
        self * rhs.inv()
    }
}

impl<T: FloatLike> std::ops::Div<HyperDual<T>> for HyperDual<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        use FloatLike;
        self * rhs.inv()
    }
}

impl<T: FloatLike> std::ops::AddAssign<&HyperDual<T>> for HyperDual<T> {
    #[inline]
    fn add_assign(&mut self, rhs: &HyperDual<T>) {
        for (s, o) in self.values.iter_mut().zip(&rhs.values) {
            *s += o;
        }
    }
}

impl<T: FloatLike> std::ops::AddAssign<HyperDual<T>> for HyperDual<T> {
    #[inline]
    fn add_assign(&mut self, rhs: HyperDual<T>) {
        self.add_assign(&rhs)
    }
}

impl<T: FloatLike> std::ops::SubAssign<&HyperDual<T>> for HyperDual<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: &HyperDual<T>) {
        for (s, o) in self.values.iter_mut().zip(&rhs.values) {
            *s -= o;
        }
    }
}

impl<T: FloatLike> std::ops::SubAssign<HyperDual<T>> for HyperDual<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: HyperDual<T>) {
        self.sub_assign(&rhs)
    }
}

impl<T: FloatLike> std::ops::AddAssign<&T> for HyperDual<T> {
    #[inline]
    fn add_assign(&mut self, rhs: &T) {
        self.values[0] += rhs;
    }
}

impl<T: FloatLike> std::ops::AddAssign<T> for HyperDual<T> {
    #[inline]
    fn add_assign(&mut self, rhs: T) {
        self.values[0] += rhs;
    }
}

impl<T: FloatLike> std::ops::SubAssign<&T> for HyperDual<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: &T) {
        self.values[0] -= rhs;
    }
}

impl<T: FloatLike> std::ops::SubAssign<T> for HyperDual<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: T) {
        self.values[0] -= rhs;
    }
}

impl<T: FloatLike> std::ops::MulAssign<&HyperDual<T>> for HyperDual<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &HyperDual<T>) {
        *self = self.clone() * rhs.clone();
    }
}

impl<T: FloatLike> std::ops::MulAssign<HyperDual<T>> for HyperDual<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: HyperDual<T>) {
        *self = self.clone() * rhs;
    }
}

impl<T: FloatLike> std::ops::DivAssign<&HyperDual<T>> for HyperDual<T> {
    #[inline]
    fn div_assign(&mut self, rhs: &HyperDual<T>) {
        use FloatLike;
        *self = rhs.inv() * &*self;
    }
}

impl<T: FloatLike> std::ops::DivAssign<HyperDual<T>> for HyperDual<T> {
    #[inline]
    fn div_assign(&mut self, rhs: HyperDual<T>) {
        use FloatLike;
        *self = rhs.inv() * &*self;
    }
}

impl<T: FloatLike> FloatLike for HyperDual<T> {
    #[inline(always)]
    fn mul_add(&self, a: &Self, b: &Self) -> Self {
        self.clone() * a + b
    }

    #[inline(always)]
    fn neg(&self) -> Self {
        -self.clone()
    }

    #[inline(always)]
    fn zero(&self) -> Self {
        HyperDual {
            values: vec![self.values[0].zero(); self.values.len()],
            shape: self.shape.clone(),
        }
    }

    #[inline(always)]
    fn new_zero() -> Self {
        HyperDual {
            values: vec![T::new_zero()],
            shape: Arc::new(DualShape {
                shape: vec![],
                max_pow: 0,
                mult_table: vec![],
            }),
        }
    }

    #[inline(always)]
    fn one(&self) -> Self {
        let mut res = self.zero();
        res.values[0] = res.values[0].one();
        res
    }

    #[inline]
    fn pow(&self, e: u64) -> Self {
        // TODO: use binary exponentiation
        let mut res = self.clone();
        for _ in 1..e {
            res = res.clone() * self;
        }
        res
    }

    #[inline(always)]
    fn inv(&self) -> Self {
        let e = self.values[0].inv();
        let mut r = self.clone() * &e;
        r.values[0] = r.values[0].zero();

        let mut accum = r.one();
        let mut res = r.one();

        for i in 1..self.shape.max_pow + 1 {
            accum *= &r;
            if i % 2 == 0 {
                res += accum.clone()
            } else {
                res -= accum.clone()
            }
        }

        res * &e
    }

    #[inline(always)]
    fn from_usize(&self, a: usize) -> Self {
        let mut res = self.zero();
        res.values[0] = res.values[0].from_usize(a);
        res
    }

    #[inline(always)]
    fn from_i64(&self, a: i64) -> Self {
        let mut res = self.zero();
        res.values[0] = res.values[0].from_i64(a);
        res
    }

    #[inline(always)]
    fn get_precision(&self) -> u32 {
        self.values[0].get_precision()
    }

    #[inline(always)]
    fn get_epsilon(&self) -> f64 {
        self.values[0].get_epsilon()
    }

    #[inline(always)]
    fn fixed_precision(&self) -> bool {
        self.values[0].fixed_precision()
    }

    fn sample_unit<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Self {
        let mut res = self.zero();
        res.values[0] = self.values[0].sample_unit(rng);
        res
    }

    #[inline(always)]
    fn is_fully_zero(&self) -> bool {
        self.values.iter().all(|v| v.is_fully_zero())
    }
}

impl<T: SingleFloat> SingleFloat for HyperDual<T> {
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.values[0].is_zero()
    }

    #[inline(always)]
    fn is_one(&self) -> bool {
        self.values[0].is_one()
    }

    #[inline(always)]
    fn is_finite(&self) -> bool {
        self.values[0].is_finite()
    }

    #[inline(always)]
    fn from_rational(&self, rat: &Rational) -> Self {
        let mut res = self.zero();
        res.values[0] = self.values[0].from_rational(rat);
        res
    }
}

impl<T: Constructible> Constructible for HyperDual<T> {
    #[inline(always)]
    fn new_one() -> Self {
        let mut res = <Self as FloatLike>::new_zero();
        res.values[0] = T::new_one();
        res
    }

    #[inline(always)]
    fn new_from_usize(a: usize) -> Self {
        let mut res = <Self as FloatLike>::new_zero();
        res.values[0] = T::new_from_usize(a);
        res
    }

    #[inline(always)]
    fn new_from_i64(a: i64) -> Self {
        let mut res = <Self as FloatLike>::new_zero();
        res.values[0] = T::new_from_i64(a);
        res
    }

    #[inline(always)]
    fn new_sample_unit<R: rand::Rng + ?Sized>(rng: &mut R) -> Self {
        let mut res = <Self as FloatLike>::new_zero();
        res.values[0] = T::new_sample_unit(rng);
        res
    }
}

impl<T: Real> Real for HyperDual<T> {
    #[inline(always)]
    fn pi(&self) -> Self {
        let mut res = self.zero();
        res.values[0] = self.values[0].pi();
        res
    }

    #[inline(always)]
    fn e(&self) -> Self {
        let mut res = self.zero();
        res.values[0] = self.values[0].e();
        res
    }

    #[inline(always)]
    fn euler(&self) -> Self {
        let mut res = self.zero();
        res.values[0] = self.values[0].euler();
        res
    }

    #[inline(always)]
    fn phi(&self) -> Self {
        let mut res = self.zero();
        res.values[0] = self.values[0].phi();
        res
    }

    #[inline(always)]
    fn i(&self) -> Option<Self> {
        let mut res = self.zero();
        res.values[0] = self.values[0].i()?;
        Some(res)
    }

    #[inline(always)]
    fn conj(&self) -> Self {
        // assume variable is real
        let mut r = self.clone();
        for x in &mut r.values {
            *x = x.conj();
        }
        r
    }

    #[inline(always)]
    fn norm(&self) -> Self {
        let n = self.values[0].norm();
        if n == self.values[0] {
            return self.clone();
        } else {
            let scale = n / &self.values[0];
            self.clone() * &scale
        }
    }

    #[inline(always)]
    fn sqrt(&self) -> Self {
        use FloatLike;
        let e = self.values[0].sqrt();

        let norm = self.values[0].inv(); // TODO: check 0
        let mut r = self.clone() * &norm;
        r.values[0] = self.values[0].zero();

        let mut accum = self.one();
        let mut res = self.one();
        let mut num = self.values[0].one();

        let mut scale = 1;
        for p in 1..self.shape.max_pow + 1 {
            scale *= p;
            num = num.clone() * (num.from_usize(2).inv() - &num.from_usize(p as usize - 1));
            accum *= &r;
            res += accum.clone() * &num * &num.from_usize(scale).inv();
        }

        res * &e
    }

    #[inline(always)]
    fn log(&self) -> Self {
        let e = self.values[0].log();

        let norm = self.values[0].inv(); // TODO: check 0
        let mut r = self.clone() * &norm;
        r.values[0] = self.values[0].zero();

        let mut accum = r.clone();
        let mut res = self.one();
        res.values[0] = e.clone();

        let mut scale = -1;
        for p in 1..self.shape.max_pow + 1 {
            scale *= -1;
            res += accum.clone() * &e.from_i64(p as i64 * scale).inv();
            accum *= &r;
        }

        res
    }

    #[inline(always)]
    fn exp(&self) -> Self {
        let e = self.values[0].exp();
        let mut res = self.one();

        let mut r = self.clone();
        r.values[0] = self.values[0].zero();
        let mut accum = self.one();
        let mut scale = 1;
        for p in 0..self.shape.max_pow {
            scale *= p + 1;
            accum *= &r; // TODO: many multiplications with 0
            res += accum.clone() * &e.from_usize(scale).inv();
        }

        res * &e
    }

    #[inline(always)]
    fn sin(&self) -> Self {
        let s = self.values[0].sin();
        let c = self.values[0].cos();

        let mut p = self.clone();
        p.values[0] = self.values[0].zero();

        let mut e = self.one();
        e.values[0] = s.clone();
        let mut sp = p.clone();
        let mut scale = 1;
        for i in 1..self.shape.max_pow + 1 {
            scale *= i;
            let mut b = if i % 2 == 1 { c.clone() } else { s.clone() };

            if i % 4 >= 2 {
                b = b.neg();
            }

            let s = sp.clone() * &b * &e.from_usize(scale).inv();

            sp *= &p;

            e += s;
        }

        e
    }

    #[inline(always)]
    fn cos(&self) -> Self {
        let s = self.values[0].sin();
        let c = self.values[0].cos();

        let mut p = self.clone();
        p.values[0] = self.values[0].zero();

        let mut e = self.one();
        e.values[0] = c.clone();
        let mut sp = p.clone();
        let mut scale = 1;
        for i in 1..self.shape.max_pow + 1 {
            scale *= i;
            let mut b = if i % 2 == 1 { s.clone() } else { -c.clone() };

            if i % 4 < 2 {
                b = b.neg();
            }

            let s = sp.clone() * &b * &e.from_usize(scale).inv();

            sp *= &p;

            e += s;
        }

        e
    }

    #[inline(always)]
    fn tan(&self) -> Self {
        // TODO: improve
        self.sin() / self.cos()
    }

    #[inline(always)]
    fn asin(&self) -> Self {
        unimplemented!()
    }

    #[inline(always)]
    fn acos(&self) -> Self {
        unimplemented!()
    }

    #[inline(always)]
    fn atan2(&self, _x: &Self) -> Self {
        unimplemented!()
    }

    #[inline(always)]
    fn sinh(&self) -> Self {
        let s = self.values[0].sinh();
        let c = self.values[0].cosh();

        let mut p = self.clone();
        p.values[0] = self.values[0].zero();

        let mut e = self.one();
        e.values[0] = s.clone();
        let mut sp = p.clone();
        let mut scale = 1;
        for i in 1..self.shape.max_pow + 1 {
            scale *= i;
            let b = if i % 2 == 1 { &c } else { &s };

            let s = sp.clone() * b * &e.from_usize(scale).inv();

            sp *= &p;

            e += s;
        }

        e
    }

    #[inline(always)]
    fn cosh(&self) -> Self {
        let s = self.values[0].sinh();
        let c = self.values[0].cosh();

        let mut p = self.clone();
        p.values[0] = self.values[0].zero();

        let mut e = self.one();
        e.values[0] = c.clone();
        let mut sp = p.clone();
        let mut scale = 1;
        for i in 1..self.shape.max_pow + 1 {
            scale *= i;
            let b = if i % 2 == 1 { &s } else { &c };

            let s = sp.clone() * b * &e.from_usize(scale).inv();

            sp *= &p;

            e += s;
        }

        e
    }

    #[inline(always)]
    fn tanh(&self) -> Self {
        self.sinh() / self.cosh()
    }

    #[inline(always)]
    fn asinh(&self) -> Self {
        unimplemented!()
    }

    #[inline(always)]
    fn acosh(&self) -> Self {
        unimplemented!()
    }

    #[inline(always)]
    fn atanh(&self) -> Self {
        unimplemented!()
    }

    #[inline]
    fn powf(&self, e: &Self) -> Self {
        // TODO: improve
        (self.log() * e).exp()
    }
}

#[cfg(test)]
mod test {
    use crate::domains::{
        dual::HyperDual,
        float::{FloatLike, Real},
        rational::Rational,
    };

    create_hyperdual_from_depths!(Dual, [1, 2, 3]);

    #[test]
    fn real_functions() {
        let x = Dual::<f64> {
            values: [
                0.2, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
                19., 20., 21., 22., 23., 24.,
            ],
        };

        fn approx_eq(a: Dual<f64>, b: &Dual<f64>) -> bool {
            for (a, b) in a.values.iter().zip(b.values.iter()) {
                if (a - b).abs() / (a + b) > 1e-10
                    && (*a != 0. || *b > 1e-10)
                    && (*b != 0. || *a > 1e-10)
                {
                    return false;
                }
            }

            true
        }

        assert!(approx_eq(x.sqrt() * x.sqrt(), &x));
        assert!(approx_eq(x.log().exp(), &x));
        assert!(approx_eq(x.exp().log(), &x));
        assert!(approx_eq(x.sin() * x.sin() + x.cos() * x.cos(), &x.one()));
        assert!(approx_eq((-x.clone()).norm(), &x.norm()));
        assert!(approx_eq(
            x.cosh() * x.cosh() - x.sinh() * x.sinh(),
            &x.one()
        ));
        assert_eq!(x.cosh() * x.cosh() - x.sinh() * x.sinh(), x.one());
    }

    #[test]
    fn dual_inv() {
        let x = Dual::<Rational>::new_variable(0, (1, 1).into());
        let y = Dual::new_variable(1, (2, 1).into());
        let z = Dual::new_variable(2, (3, 1).into());

        let t3 = x * y * z;
        let one = t3.clone() * t3.inv();

        assert_eq!(one, t3.one());
    }

    #[test]
    fn runtime_dual() {
        let x = HyperDual::<f64>::from_values(
            vec![
                vec![0, 0, 0],
                vec![1, 0, 0],
                vec![0, 1, 0],
                vec![0, 0, 1],
                vec![1, 1, 0],
                vec![0, 2, 0],
                vec![1, 0, 1],
                vec![0, 1, 1],
                vec![0, 0, 2],
                vec![1, 2, 0],
                vec![1, 1, 1],
                vec![0, 2, 1],
                vec![1, 0, 2],
                vec![0, 1, 2],
                vec![0, 0, 3],
                vec![1, 2, 1],
                vec![1, 1, 2],
                vec![0, 2, 2],
                vec![1, 0, 3],
                vec![0, 1, 3],
                vec![1, 2, 2],
                vec![1, 1, 3],
                vec![0, 2, 3],
                vec![1, 2, 3],
            ],
            vec![
                0.2, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.,
                19., 20., 21., 22., 23., 24.,
            ],
        );

        fn approx_eq(a: HyperDual<f64>, b: &HyperDual<f64>) -> bool {
            for (a, b) in a.values.iter().zip(b.values.iter()) {
                if (a - b).abs() / (a + b) > 1e-10
                    && (*a != 0. || *b > 1e-10)
                    && (*b != 0. || *a > 1e-10)
                {
                    return false;
                }
            }

            true
        }

        assert!(approx_eq(x.sqrt() * x.sqrt(), &x));
        assert!(approx_eq(x.log().exp(), &x));
        assert!(approx_eq(x.exp().log(), &x));
        assert!(approx_eq(x.sin() * x.sin() + x.cos() * x.cos(), &x.one()));
        assert!(approx_eq((-x.clone()).norm(), &x.norm()));
        assert!(approx_eq(
            x.cosh() * x.cosh() - x.sinh() * x.sinh(),
            &x.one()
        ));
        assert_eq!(x.cosh() * x.cosh() - x.sinh() * x.sinh(), x.one());
    }
}
