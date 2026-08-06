use std::{
    cmp::Ordering,
    fmt::{self, Debug, Display, Formatter, LowerExp, Write},
    hash::Hash,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use rand::Rng;

use super::{Complex, Float, FloatLike, Real, RealLike, RoundingDirection, SingleFloat};
use crate::domains::{InternalOrdering, integer::Integer, rational::Rational};

/// A closed real interval represented by a center and a nonnegative radius.
///
/// The ball represents all numbers in `[center - radius, center + radius]`.
/// Addition, subtraction, multiplication, division, and inversion use directed
/// rounding and return certified enclosures. Other operations are not
/// currently certifying.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct RealBall {
    pub center: Float,
    pub radius: Float,
}

/// A rectangular complex ball, consisting of real balls for its real and
/// imaginary components.
///
/// Construction from rational data, negation, conjugation, addition,
/// subtraction, multiplication, division, inversion, nonnegative integer
/// powers, and [`ComplexBall::is_disjoint`] are certifying. They enclose the
/// exact result using directed rounding. Inherited transcendental functions
/// are not currently certifying.
pub type ComplexBall = Complex<RealBall>;

impl ComplexBall {
    /// Certify that every value in this ball is real and strictly positive.
    pub fn is_strictly_positive(&self) -> bool {
        self.is_real() && self.re.is_strictly_positive()
    }

    /// Certify that every value in this ball is real and strictly negative.
    pub fn is_strictly_negative(&self) -> bool {
        self.is_real() && self.re.is_strictly_negative()
    }

    /// Return whether this ball contains zero.
    pub fn contains_zero(&self) -> bool {
        let zero = self.re.center.zero();
        self.re.contains(&zero) && self.im.contains(&zero)
    }

    /// Return whether this ball contains the real value `value`.
    pub fn contains(&self, value: &Float) -> bool {
        self.im.contains_zero() && self.re.contains(value)
    }

    /// Return whether this ball contains the complex value `value`.
    pub fn contains_complex(&self, value: &Complex<Float>) -> bool {
        self.re.contains(&value.re) && self.im.contains(&value.im)
    }

    /// Return whether this ball contains the whole of `other`.
    pub fn contains_ball(&self, other: &Self) -> bool {
        self.re.contains_ball(&other.re) && self.im.contains_ball(&other.im)
    }

    /// Return whether this ball intersects `other`.
    pub fn intersects(&self, other: &Self) -> bool {
        self.re.intersects(&other.re) && self.im.intersects(&other.im)
    }
}

impl RealBall {
    /// Construct a ball from its center and radius.
    ///
    /// A negative radius is converted to its absolute value.
    pub fn new(center: Float, radius: Float) -> Self {
        Self {
            center,
            radius: radius.norm(),
        }
    }

    /// Construct a ball containing exactly one floating-point number.
    pub fn exact(center: Float) -> Self {
        let radius = center.zero();
        Self { center, radius }
    }

    /// Enclose a mathematical value represented by a rounded float.
    ///
    /// The radius is one unit of relative precision, with an absolute scale
    /// of one for values whose magnitude is below one.
    fn from_rounded_value(center: Float, prec: u32) -> Self {
        let epsilon = Rational::from((Integer::one(), Integer::from(2).pow(prec as u64)));
        let epsilon = Float::from_rational_round(&epsilon, prec, RoundingDirection::Up);
        let magnitude = center.norm();
        let scale = if magnitude < center.one() {
            center.one()
        } else {
            magnitude
        };
        let radius = epsilon.mul_round(&scale, prec, RoundingDirection::Up);
        Self { center, radius }
    }

    /// Construct the smallest readily representable ball from two bounds.
    ///
    /// # Panics
    /// Panics if `lower > upper`.
    pub fn from_bounds(lower: Float, upper: Float) -> Self {
        assert!(lower <= upper, "lower interval bound exceeds upper bound");
        let prec = lower.get_precision().min(upper.get_precision());
        Self::from_outward_bounds(lower, upper, prec)
    }

    /// Construct a certified ball from exact rational bounds.
    pub fn from_rational_bounds(lower: &Rational, upper: &Rational, prec: u32) -> Self {
        assert!(lower <= upper, "lower interval bound exceeds upper bound");
        let lower = Float::from_rational_round(lower, prec, RoundingDirection::Down);
        let upper = Float::from_rational_round(upper, prec, RoundingDirection::Up);
        Self::from_outward_bounds(lower, upper, prec)
    }

    /// Construct a certified ball from an exact rational center and radius.
    pub fn from_rational_ball(center: &Rational, radius: &Rational, prec: u32) -> Self {
        assert!(!radius.is_negative(), "ball radius must be nonnegative");
        Self::from_rational_bounds(&(center - radius), &(center + radius), prec)
    }

    fn from_outward_bounds(lower: Float, upper: Float, prec: u32) -> Self {
        let midpoint_sum = lower.add_round(&upper, prec, RoundingDirection::Nearest);
        // Division by two only changes the binary exponent and is exact.
        let center = midpoint_sum.clone() / midpoint_sum.from_usize(2);
        let left_radius = center.sub_round(&lower, prec, RoundingDirection::Up);
        let right_radius = upper.sub_round(&center, prec, RoundingDirection::Up);
        let radius = if left_radius >= right_radius {
            left_radius
        } else {
            right_radius
        };
        Self { center, radius }
    }

    /// The lower endpoint of this ball.
    pub fn lower_bound(&self) -> Float {
        self.center
            .sub_round(&self.radius, self.get_precision(), RoundingDirection::Down)
    }

    /// The upper endpoint of this ball.
    pub fn upper_bound(&self) -> Float {
        self.center
            .add_round(&self.radius, self.get_precision(), RoundingDirection::Up)
    }

    /// The width of this interval.
    pub fn width(&self) -> Float {
        self.radius.clone() + &self.radius
    }

    /// Return whether this ball contains `value`.
    pub fn contains(&self, value: &Float) -> bool {
        self.lower_bound() <= *value && *value <= self.upper_bound()
    }

    /// Return whether this ball contains the whole of `other`.
    pub fn contains_ball(&self, other: &Self) -> bool {
        self.lower_bound() <= other.lower_bound() && self.upper_bound() >= other.upper_bound()
    }

    /// Return whether this ball intersects `other`.
    pub fn intersects(&self, other: &Self) -> bool {
        self.lower_bound() <= other.upper_bound() && other.lower_bound() <= self.upper_bound()
    }

    /// Certify that every value in this ball is positive.
    pub fn is_strictly_positive(&self) -> bool {
        self.lower_bound() > self.center.zero()
    }

    /// Certify that every value in this ball is negative.
    pub fn is_strictly_negative(&self) -> bool {
        self.upper_bound() < self.center.zero()
    }

    fn contains_zero(&self) -> bool {
        let zero = self.center.zero();
        self.contains(&zero)
    }

    fn whole(prec: u32) -> Self {
        Self {
            center: Float::new(prec),
            radius: Float::with_val(prec, f64::INFINITY),
        }
    }

    fn invalid(prec: u32) -> Self {
        Self {
            center: Float::with_val(prec, f64::NAN),
            radius: Float::with_val(prec, f64::NAN),
        }
    }

    fn monotone_increasing(&self, f: impl Fn(&Float) -> Float) -> Self {
        Self::from_bounds(f(&self.lower_bound()), f(&self.upper_bound()))
    }

    fn monotone_decreasing(&self, f: impl Fn(&Float) -> Float) -> Self {
        Self::from_bounds(f(&self.upper_bound()), f(&self.lower_bound()))
    }
}

impl Complex<RealBall> {
    /// Construct a certified rectangular complex ball from an exact rational
    /// center and a shared radius.
    pub fn from_rational_ball(center: &Complex<Rational>, radius: &Rational, prec: u32) -> Self {
        Self::new(
            RealBall::from_rational_ball(&center.re, radius, prec),
            RealBall::from_rational_ball(&center.im, radius, prec),
        )
    }

    /// Certify that these rectangular complex balls are disjoint.
    pub fn is_disjoint(&self, other: &Self) -> bool {
        !self.intersects(other)
    }
}

impl From<Float> for RealBall {
    fn from(value: Float) -> Self {
        Self::exact(value)
    }
}

impl From<f64> for RealBall {
    fn from(value: f64) -> Self {
        Self::exact(value.into())
    }
}

impl InternalOrdering for RealBall {
    fn internal_cmp(&self, other: &Self) -> Ordering {
        self.center
            .internal_cmp(&other.center)
            .then_with(|| self.radius.internal_cmp(&other.radius))
    }
}

impl PartialOrd for RealBall {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.center.partial_cmp(&other.center)
    }
}

impl Neg for RealBall {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self {
            center: -self.center,
            radius: self.radius,
        }
    }
}

impl Add<&RealBall> for RealBall {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        let prec = self.get_precision().min(rhs.get_precision());
        let lower = self
            .lower_bound()
            .add_round(&rhs.lower_bound(), prec, RoundingDirection::Down);
        let upper = self
            .upper_bound()
            .add_round(&rhs.upper_bound(), prec, RoundingDirection::Up);
        Self::from_outward_bounds(lower, upper, prec)
    }
}

impl Add for RealBall {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl Sub<&RealBall> for RealBall {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        let prec = self.get_precision().min(rhs.get_precision());
        let lower = self
            .lower_bound()
            .sub_round(&rhs.upper_bound(), prec, RoundingDirection::Down);
        let upper = self
            .upper_bound()
            .sub_round(&rhs.lower_bound(), prec, RoundingDirection::Up);
        Self::from_outward_bounds(lower, upper, prec)
    }
}

impl Sub for RealBall {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self - &rhs
    }
}

impl Mul<&RealBall> for RealBall {
    type Output = Self;

    fn mul(self, rhs: &Self) -> Self::Output {
        let prec = self.get_precision().min(rhs.get_precision());
        let self_bounds = [self.lower_bound(), self.upper_bound()];
        let rhs_bounds = [rhs.lower_bound(), rhs.upper_bound()];

        let mut lower = self_bounds[0].mul_round(&rhs_bounds[0], prec, RoundingDirection::Down);
        let mut upper = self_bounds[0].mul_round(&rhs_bounds[0], prec, RoundingDirection::Up);
        for a in &self_bounds {
            for b in &rhs_bounds {
                let candidate_lower = a.mul_round(b, prec, RoundingDirection::Down);
                let candidate_upper = a.mul_round(b, prec, RoundingDirection::Up);
                if candidate_lower < lower {
                    lower = candidate_lower;
                }
                if candidate_upper > upper {
                    upper = candidate_upper;
                }
            }
        }
        Self::from_outward_bounds(lower, upper, prec)
    }
}

impl Mul for RealBall {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self * &rhs
    }
}

impl Div<&RealBall> for RealBall {
    type Output = Self;

    fn div(self, rhs: &Self) -> Self::Output {
        if rhs.contains_zero() {
            return Self::whole(self.get_precision().min(rhs.get_precision()));
        }

        let prec = self.get_precision().min(rhs.get_precision());
        let self_bounds = [self.lower_bound(), self.upper_bound()];
        let rhs_bounds = [rhs.lower_bound(), rhs.upper_bound()];

        let mut lower = self_bounds[0].div_round(&rhs_bounds[0], prec, RoundingDirection::Down);
        let mut upper = self_bounds[0].div_round(&rhs_bounds[0], prec, RoundingDirection::Up);
        for a in &self_bounds {
            for b in &rhs_bounds {
                let candidate_lower = a.div_round(b, prec, RoundingDirection::Down);
                let candidate_upper = a.div_round(b, prec, RoundingDirection::Up);
                if candidate_lower < lower {
                    lower = candidate_lower;
                }
                if candidate_upper > upper {
                    upper = candidate_upper;
                }
            }
        }
        Self::from_outward_bounds(lower, upper, prec)
    }
}

impl Div for RealBall {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self / &rhs
    }
}

macro_rules! impl_ref_op {
    ($trait:ident, $method:ident) => {
        impl $trait<&RealBall> for &RealBall {
            type Output = RealBall;

            fn $method(self, rhs: &RealBall) -> Self::Output {
                self.clone().$method(rhs)
            }
        }

        impl $trait<RealBall> for &RealBall {
            type Output = RealBall;

            fn $method(self, rhs: RealBall) -> Self::Output {
                self.clone().$method(rhs)
            }
        }
    };
}

impl_ref_op!(Add, add);
impl_ref_op!(Sub, sub);
impl_ref_op!(Mul, mul);
impl_ref_op!(Div, div);

macro_rules! impl_assign {
    ($trait:ident, $method:ident, $op:tt) => {
        impl $trait<&RealBall> for RealBall {
            fn $method(&mut self, rhs: &RealBall) {
                *self = self.clone() $op rhs;
            }
        }

        impl $trait<RealBall> for RealBall {
            fn $method(&mut self, rhs: RealBall) {
                *self = self.clone() $op rhs;
            }
        }
    };
}

impl_assign!(AddAssign, add_assign, +);
impl_assign!(SubAssign, sub_assign, -);
impl_assign!(MulAssign, mul_assign, *);
impl_assign!(DivAssign, div_assign, /);

impl Display for RealBall {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_char('(')?;
        Display::fmt(&self.center, f)?;
        f.write_str(" +/- ")?;
        Display::fmt(&self.radius, f)?;
        f.write_char(')')
    }
}

impl Debug for RealBall {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("RealBall")
            .field("center", &self.center)
            .field("radius", &self.radius)
            .finish()
    }
}

impl LowerExp for RealBall {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.write_char('(')?;
        LowerExp::fmt(&self.center, f)?;
        f.write_str(" +/- ")?;
        LowerExp::fmt(&self.radius, f)?;
        f.write_char(')')
    }
}

impl FloatLike for RealBall {
    fn set_from(&mut self, other: &Self) {
        self.center.set_from(&other.center);
        self.radius.set_from(&other.radius);
    }

    fn mul_add(&self, a: &Self, b: &Self) -> Self {
        self.clone() * a + b
    }

    fn neg(&self) -> Self {
        -self.clone()
    }

    fn zero(&self) -> Self {
        Self::exact(self.center.zero())
    }

    fn new_zero() -> Self {
        Self::exact(Float::new(1))
    }

    fn one(&self) -> Self {
        Self::exact(self.center.one())
    }

    fn pow(&self, mut e: u64) -> Self {
        let mut base = self.clone();
        let mut result = self.one();
        while e != 0 {
            if e & 1 == 1 {
                result *= &base;
            }
            e >>= 1;
            if e != 0 {
                base = base.clone() * base;
            }
        }
        result
    }

    fn inv(&self) -> Self {
        if self.contains_zero() {
            return Self::whole(self.get_precision());
        }

        let prec = self.get_precision();
        let one = self.center.one();
        let lower = one.div_round(&self.upper_bound(), prec, RoundingDirection::Down);
        let upper = one.div_round(&self.lower_bound(), prec, RoundingDirection::Up);
        Self::from_outward_bounds(lower, upper, prec)
    }

    fn from_usize(&self, a: usize) -> Self {
        let value: Rational = Integer::from(a).into();
        Self::from_rational_bounds(&value, &value, self.get_precision())
    }

    fn from_i64(&self, a: i64) -> Self {
        let value: Rational = a.into();
        Self::from_rational_bounds(&value, &value, self.get_precision())
    }

    fn get_precision(&self) -> u32 {
        self.center.get_precision().min(self.radius.get_precision())
    }

    fn get_epsilon(&self) -> f64 {
        2.0f64.powi(-(self.get_precision() as i32))
    }

    fn fixed_precision(&self) -> bool {
        false
    }

    fn sample_unit<R: Rng + ?Sized>(&self, rng: &mut R) -> Self {
        Self::exact(self.center.sample_unit(rng))
    }

    fn is_fully_zero(&self) -> bool {
        self.center.is_fully_zero() && self.radius.is_fully_zero()
    }
}

impl SingleFloat for RealBall {
    fn is_zero(&self) -> bool {
        self.center.is_zero() && self.radius.is_zero()
    }

    fn is_one(&self) -> bool {
        self.center.is_one() && self.radius.is_zero()
    }

    fn is_finite(&self) -> bool {
        self.center.is_finite() && self.radius.is_finite()
    }

    fn from_rational(&self, rat: &Rational) -> Self {
        Self::from_rational_bounds(rat, rat, self.get_precision())
    }
}

impl RealLike for RealBall {
    fn to_usize_clamped(&self) -> usize {
        self.center.to_usize_clamped()
    }

    fn to_f64(&self) -> f64 {
        self.center.to_f64()
    }

    fn round_to_nearest_integer(&self) -> Integer {
        self.center.round_to_nearest_integer()
    }
}

impl Real for RealBall {
    fn pi(&self) -> Self {
        Self::from_rounded_value(self.center.pi(), self.get_precision())
    }

    fn e(&self) -> Self {
        Self::from_rounded_value(self.center.e(), self.get_precision())
    }

    fn euler(&self) -> Self {
        Self::from_rounded_value(self.center.euler(), self.get_precision())
    }

    fn phi(&self) -> Self {
        Self::from_rounded_value(self.center.phi(), self.get_precision())
    }

    fn i(&self) -> Option<Self> {
        None
    }

    fn conj(&self) -> Self {
        self.clone()
    }

    fn norm(&self) -> Self {
        let lower = self.lower_bound();
        let upper = self.upper_bound();
        let zero = self.center.zero();

        if lower >= zero {
            self.clone()
        } else if upper <= zero {
            -self.clone()
        } else {
            let bound = if lower.norm() >= upper.norm() {
                lower.norm()
            } else {
                upper.norm()
            };
            Self::from_bounds(self.center.zero(), bound)
        }
    }

    fn sqrt(&self) -> Self {
        if self.lower_bound() < self.center.zero() {
            return Self::invalid(self.get_precision());
        }
        self.monotone_increasing(Float::sqrt)
    }

    fn log(&self) -> Self {
        if self.lower_bound() <= self.center.zero() {
            return Self::invalid(self.get_precision());
        }
        self.monotone_increasing(Float::log)
    }

    fn exp(&self) -> Self {
        self.monotone_increasing(Float::exp)
    }

    fn sin(&self) -> Self {
        // Sine is 1-Lipschitz. Intersect that enclosure with its global range.
        let value = self.center.sin();
        let minus_one = self.center.from_i64(-1);
        let one = self.center.one();
        let lower = value.clone() - &self.radius;
        let upper = value + &self.radius;
        Self::from_bounds(
            if lower < minus_one { minus_one } else { lower },
            if upper > one { one } else { upper },
        )
    }

    fn cos(&self) -> Self {
        // Cosine is 1-Lipschitz. Intersect that enclosure with its global range.
        let value = self.center.cos();
        let minus_one = self.center.from_i64(-1);
        let one = self.center.one();
        let lower = value.clone() - &self.radius;
        let upper = value + &self.radius;
        Self::from_bounds(
            if lower < minus_one { minus_one } else { lower },
            if upper > one { one } else { upper },
        )
    }

    fn tan(&self) -> Self {
        let lower = self.lower_bound();
        let upper = self.upper_bound();
        let lower_f64 = lower.to_f64();
        let upper_f64 = upper.to_f64();
        let first_pole = ((lower_f64 - std::f64::consts::FRAC_PI_2) / std::f64::consts::PI).ceil()
            * std::f64::consts::PI
            + std::f64::consts::FRAC_PI_2;

        if !lower_f64.is_finite() || !upper_f64.is_finite() || first_pole <= upper_f64 {
            Self::whole(self.get_precision())
        } else {
            Self::from_bounds(lower.tan(), upper.tan())
        }
    }

    fn asin(&self) -> Self {
        let minus_one = self.center.from_i64(-1);
        let one = self.center.one();
        if self.lower_bound() < minus_one || self.upper_bound() > one {
            return Self::invalid(self.get_precision());
        }
        self.monotone_increasing(Float::asin)
    }

    fn acos(&self) -> Self {
        let minus_one = self.center.from_i64(-1);
        let one = self.center.one();
        if self.lower_bound() < minus_one || self.upper_bound() > one {
            return Self::invalid(self.get_precision());
        }
        self.monotone_decreasing(Float::acos)
    }

    fn atan2(&self, x: &Self) -> Self {
        if self.radius.is_zero() && x.radius.is_zero() {
            return Self::exact(self.center.atan2(&x.center));
        }

        let zero = self.center.zero();
        if x.contains(&zero) && self.contains(&zero) {
            let pi = self.center.pi();
            return Self::from_bounds(-pi.clone(), pi);
        }

        let ys = [self.lower_bound(), self.upper_bound()];
        let xs = [x.lower_bound(), x.upper_bound()];
        let mut angles = Vec::with_capacity(4);
        for y in &ys {
            for x in &xs {
                angles.push(y.atan2(x));
            }
        }

        // Crossing the negative real-axis branch cut requires the full
        // principal range as a single real interval.
        if x.upper_bound() < zero && self.contains(&zero) {
            let pi = self.center.pi();
            return Self::from_bounds(-pi.clone(), pi);
        }

        let mut lower = angles[0].clone();
        let mut upper = angles[0].clone();
        for angle in angles.into_iter().skip(1) {
            if angle < lower {
                lower = angle.clone();
            }
            if angle > upper {
                upper = angle;
            }
        }
        Self::from_bounds(lower, upper)
    }

    fn sinh(&self) -> Self {
        self.monotone_increasing(Float::sinh)
    }

    fn cosh(&self) -> Self {
        let lower = self.lower_bound();
        let upper = self.upper_bound();
        let zero = self.center.zero();
        if lower >= zero {
            Self::from_bounds(lower.cosh(), upper.cosh())
        } else if upper <= zero {
            Self::from_bounds(upper.cosh(), lower.cosh())
        } else {
            let left = lower.cosh();
            let right = upper.cosh();
            let high = if left >= right { left } else { right };
            Self::from_bounds(self.center.one(), high)
        }
    }

    fn tanh(&self) -> Self {
        self.monotone_increasing(Float::tanh)
    }

    fn asinh(&self) -> Self {
        self.monotone_increasing(Float::asinh)
    }

    fn acosh(&self) -> Self {
        if self.lower_bound() < self.center.one() {
            return Self::invalid(self.get_precision());
        }
        self.monotone_increasing(Float::acosh)
    }

    fn atanh(&self) -> Self {
        let minus_one = self.center.from_i64(-1);
        let one = self.center.one();
        if self.lower_bound() <= minus_one || self.upper_bound() >= one {
            return Self::invalid(self.get_precision());
        }
        self.monotone_increasing(Float::atanh)
    }

    fn powf(&self, e: &Self) -> Self {
        if self.radius.is_zero() && e.radius.is_zero() {
            return Self::exact(self.center.powf(&e.center));
        }
        if self.lower_bound() <= self.center.zero() {
            return Self::invalid(self.get_precision().min(e.get_precision()));
        }
        (e.clone() * &self.log()).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn float(value: f64) -> Float {
        Float::with_val(80, value)
    }

    fn assert_contains_rational(ball: &RealBall, value: &Rational) {
        let lower = ball.lower_bound().to_rational();
        let upper = ball.upper_bound().to_rational();
        assert!(
            lower <= *value && *value <= upper,
            "{value} is not contained in [{lower}, {upper}]"
        );
    }

    fn assert_encloses(ball: &RealBall, lower: &Rational, upper: &Rational) {
        let actual_lower = ball.lower_bound().to_rational();
        let actual_upper = ball.upper_bound().to_rational();
        assert!(actual_lower <= *lower, "{actual_lower} > {lower}");
        assert!(actual_upper >= *upper, "{actual_upper} < {upper}");
    }

    #[test]
    fn basic_interval_arithmetic() {
        let a = RealBall::new(float(2.0), float(0.25));
        let b = RealBall::new(float(3.0), float(0.5));

        let sum = a.clone() + &b;
        assert!(sum.contains(&float(4.25)));
        assert!(sum.contains(&float(5.75)));

        let product = a * b;
        assert!(product.contains(&float(4.375)));
        assert!(product.contains(&float(7.875)));
    }

    #[test]
    fn division_by_zero_containing_ball_is_unbounded() {
        let a = RealBall::exact(float(1.0));
        let b = RealBall::new(float(0.0), float(1.0));
        assert!(!(a / b).is_finite());
    }

    #[test]
    fn complex_ball_uses_real_ball_components() {
        let z = ComplexBall::new(RealBall::exact(float(1.0)), RealBall::exact(float(2.0)));
        let square = z.clone() * z;
        assert!(square.re.contains(&float(-3.0)));
        assert!(square.im.contains(&float(4.0)));
    }

    #[test]
    fn rational_conversion_and_arithmetic_are_certified() {
        let third: Rational = (1, 3).into();
        let seventh: Rational = (1, 7).into();
        let a = RealBall::from_rational_bounds(&third, &third, 5);
        let b = RealBall::from_rational_bounds(&seventh, &seventh, 5);

        assert_contains_rational(&a, &third);
        assert_contains_rational(&(a.clone() + &b), &(&third + &seventh));
        assert_contains_rational(&(a * b), &(&third * &seventh));
    }

    #[test]
    fn rounded_operations_enclose_exact_float_operands() {
        let a = RealBall::exact(Float::with_val(5, 1.34375));
        let b = RealBall::exact(Float::with_val(5, -1.71875));
        let a_exact = a.center.to_rational();
        let b_exact = b.center.to_rational();

        assert_contains_rational(&(a.clone() + &b), &(&a_exact + &b_exact));
        assert_contains_rational(&(a * b), &(&a_exact * &b_exact));
    }

    #[test]
    fn mathematical_constants_have_rounding_radii() {
        let low = RealBall::exact(Float::new(16));
        let high = Float::new(256);
        let constants = [
            (low.pi(), high.pi()),
            (low.e(), high.e()),
            (low.euler(), high.euler()),
            (low.phi(), high.phi()),
        ];

        for (ball, reference) in constants {
            assert!(!ball.radius.is_zero());
            assert!(ball.contains(&reference));
        }
    }

    #[test]
    fn complex_multiplication_is_certified() {
        let one_third: Rational = (1, 3).into();
        let one_fifth: Rational = (1, 5).into();
        let two_sevenths: Rational = (2, 7).into();
        let minus_three_elevenths: Rational = (-3, 11).into();

        let z = ComplexBall::new(
            RealBall::from_rational_bounds(&one_third, &one_third, 8),
            RealBall::from_rational_bounds(&one_fifth, &one_fifth, 8),
        );
        let w = ComplexBall::new(
            RealBall::from_rational_bounds(&two_sevenths, &two_sevenths, 8),
            RealBall::from_rational_bounds(&minus_three_elevenths, &minus_three_elevenths, 8),
        );
        let product = z * w;
        let expected_re = &one_third * &two_sevenths - &one_fifth * &minus_three_elevenths;
        let expected_im = &one_third * &minus_three_elevenths + &one_fifth * &two_sevenths;

        assert_contains_rational(&product.re, &expected_re);
        assert_contains_rational(&product.im, &expected_im);
    }

    #[test]
    fn directed_rounding_encloses_rational_interval_operations() {
        for prec in 2..=12 {
            for a_num in -4..=4 {
                for b_num in -4..=4 {
                    let a_lower: Rational = (a_num, 7).into();
                    let a_upper = &a_lower + &Rational::from((1, 13));
                    let b_lower: Rational = (b_num, 5).into();
                    let b_upper = &b_lower + &Rational::from((1, 11));
                    let a = RealBall::from_rational_bounds(&a_lower, &a_upper, prec);
                    let b = RealBall::from_rational_bounds(&b_lower, &b_upper, prec);

                    assert_encloses(
                        &(a.clone() + &b),
                        &(&a_lower + &b_lower),
                        &(&a_upper + &b_upper),
                    );

                    let products = [
                        &a_lower * &b_lower,
                        &a_lower * &b_upper,
                        &a_upper * &b_lower,
                        &a_upper * &b_upper,
                    ];
                    let product_lower = products.iter().min().unwrap();
                    let product_upper = products.iter().max().unwrap();
                    assert_encloses(&(a.clone() * &b), product_lower, product_upper);

                    if b_upper < Rational::zero() || b_lower > Rational::zero() {
                        let quotients = [
                            &a_lower / &b_lower,
                            &a_lower / &b_upper,
                            &a_upper / &b_lower,
                            &a_upper / &b_upper,
                        ];
                        let quotient_lower = quotients.iter().min().unwrap();
                        let quotient_upper = quotients.iter().max().unwrap();
                        assert_encloses(&(a / &b), quotient_lower, quotient_upper);

                        let inverse_lower = b_upper.inv();
                        let inverse_upper = b_lower.inv();
                        assert_encloses(&b.inv(), &inverse_lower, &inverse_upper);
                    }
                }
            }
        }
    }

    #[test]
    fn complex_division_and_inversion_are_certified() {
        let one_third: Rational = (1, 3).into();
        let two_fifths: Rational = (2, 5).into();
        let three_sevenths: Rational = (3, 7).into();
        let minus_one_eleventh: Rational = (-1, 11).into();
        let z = ComplexBall::new(
            RealBall::from_rational_bounds(&one_third, &one_third, 16),
            RealBall::from_rational_bounds(&two_fifths, &two_fifths, 16),
        );
        let w = ComplexBall::new(
            RealBall::from_rational_bounds(&three_sevenths, &three_sevenths, 16),
            RealBall::from_rational_bounds(&minus_one_eleventh, &minus_one_eleventh, 16),
        );

        let norm = &three_sevenths * &three_sevenths + &minus_one_eleventh * &minus_one_eleventh;
        let expected_re =
            (&one_third * &three_sevenths + &two_fifths * &minus_one_eleventh) / &norm;
        let expected_im =
            (&two_fifths * &three_sevenths - &one_third * &minus_one_eleventh) / &norm;
        let quotient = z / &w;
        assert_contains_rational(&quotient.re, &expected_re);
        assert_contains_rational(&quotient.im, &expected_im);

        let inverse = w.inv();
        assert_contains_rational(&inverse.re, &(&three_sevenths / &norm));
        let expected_inverse_im = minus_one_eleventh.neg() / &norm;
        assert_contains_rational(&inverse.im, &expected_inverse_im);
    }

    #[test]
    fn root_predicates_are_certifying() {
        let radius: Rational = (1, 100).into();
        let left_center = Complex::new(Rational::from((1, 3)), Rational::from((2, 5)));
        let right_center = Complex::new(Rational::from((2, 3)), Rational::from((2, 5)));
        let left = ComplexBall::from_rational_ball(&left_center, &radius, 8);
        let right = ComplexBall::from_rational_ball(&right_center, &radius, 8);

        assert!(left.re.is_strictly_positive());
        assert!(!left.re.is_strictly_negative());
        assert!(left.is_disjoint(&right));
    }
}
