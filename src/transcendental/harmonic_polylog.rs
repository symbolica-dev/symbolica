//! Harmonic polylogarithms at the two primitive sixth roots of unity.
//!
//! The kernels are `f_0(t)=1/t`, `f_1(t)=1/(1-t)` and
//! `f_-1(t)=1/(1+t)`. Trailing zeros are removed by shuffle identities.
//! After rescaling the endpoint to one, a Hölder convolution at one half
//! expresses every remaining GPL in geometrically convergent series.
//!
//! References: Remiddi--Vermaseren, hep-ph/9905237; Vollinga--Weinzierl,
//! hep-ph/0410259. The implementation does not require an external evaluator.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Mutex, OnceLock};

use crate::atom::{Atom, AtomView, EvaluationInfo, Symbol};
use crate::domains::float::{Complex, ComplexBall, Float, Real, RealBall, SingleFloat};
use crate::domains::integer::Integer;
use crate::domains::rational::Rational;
use crate::state::StateInitializer;
use crate::{function, get_symbol, symbol};

use super::SymbolCache;

const MAX_WEIGHT: usize = 6;
const MAX_WORDS: usize = 4096;
const MAX_BITS: u32 = 1_000_000;
const MAX_SERIES_WORK: u64 = 250_000_000;

static HPL: SymbolCache<Symbol> =
    SymbolCache::new(|| get_symbol!("hpl_at_sixth_root").expect("sixth-root HPL was initialized"));
static INDICES: SymbolCache<Symbol> =
    SymbolCache::new(|| get_symbol!("hpl_indices").expect("HPL index tag was initialized"));

#[cfg(not(doctest))]
crate::_inventory::submit! {
    StateInitializer::new("symbolica::sixth_root_hpl", || {
        // Explicit (empty) attributes register the inert word tag as a builtin
        // name, so parsing from a user namespace retains its exact identity.
        let _ = symbol!("hpl_indices";);
        let _ = symbol!(
            "hpl_at_sixth_root",
            eval = EvaluationInfo::constant(evaluate_constant).with_tags(2)
        );
    }, &["symbolica"])
}

/// The evaluatable constant `hpl_at_sixth_root(hpl_indices(a1,...,an), sign)`.
///
/// Indices are -1, 0, or 1, up to weight six. `sign` is +1 or -1 and selects
/// `exp(sign*i*pi/3)`. Evaluation follows the straight radial path from zero,
/// with the principal logarithm for trailing zeros. Both arguments are exact
/// symbolic tags; evaluation receives the requested precision explicitly.
pub fn hpl_at_sixth_root() -> Symbol {
    *HPL
}

/// Construct a sixth-root HPL constant, validating its word and endpoint.
pub fn sixth_root_hpl(indices: &[i8], sign: i8) -> Result<Atom, String> {
    validate_word(indices)?;
    validate_sign(sign)?;
    let word = INDICES.call(&indices.iter().map(|i| Atom::num(*i)).collect::<Vec<_>>());
    Ok(function!(hpl_at_sixth_root(), word, Atom::num(sign)))
}

fn validate_word(word: &[i8]) -> Result<(), String> {
    if word.len() > MAX_WEIGHT || word.iter().any(|i| !(-1..=1).contains(i)) {
        return Err("sixth-root HPL requires indices in {-1,0,1} and weight at most six".into());
    }
    Ok(())
}

fn validate_sign(sign: i8) -> Result<(), String> {
    if ![-1, 1].contains(&sign) {
        return Err("sixth-root HPL sign must be exactly -1 or +1".into());
    }
    Ok(())
}

type ConstantCache = BTreeMap<(Vec<i8>, u32), Complex<Float>>;

fn evaluate_constant(tags: &[AtomView], bits: u32) -> Result<Complex<Float>, String> {
    let [word, sign] = tags else {
        return Err("sixth-root HPL expects an index tag and a sign".into());
    };
    let AtomView::Fun(word) = word else {
        return Err("sixth-root HPL expects hpl_indices(...) as its first argument".into());
    };
    if word.get_symbol() != *INDICES {
        return Err("sixth-root HPL expects hpl_indices(...) as its first argument".into());
    }
    let indices = word
        .iter()
        .map(|i| i64::try_from(i).ok().and_then(|i| i8::try_from(i).ok()))
        .collect::<Option<Vec<_>>>()
        .ok_or("sixth-root HPL indices must be exact integers")?;
    let sign = i64::try_from(*sign)
        .ok()
        .and_then(|i| i8::try_from(i).ok())
        .ok_or("sixth-root HPL sign must be exactly -1 or +1")?;
    validate_word(&indices)?;
    validate_sign(sign)?;
    static CACHE: OnceLock<Mutex<ConstantCache>> = OnceLock::new();
    let cache = CACHE.get_or_init(Default::default);
    let key = (indices.clone(), bits);
    let cached = cache
        .lock()
        .map_err(|_| "HPL numerical cache is poisoned")?
        .get(&key)
        .cloned();
    let mut value = if let Some(value) = cached {
        value
    } else {
        let value = SixthRootHplEvaluator::new(&[indices])?
            .evaluate(1, bits)?
            .remove(0);
        let mut cache = cache
            .lock()
            .map_err(|_| "HPL numerical cache is poisoned")?;
        // Bound retained numerical data, including high-precision requests.
        if cache.len() >= 512
            || cache.keys().map(|(_, p)| u64::from(*p)).sum::<u64>() + u64::from(bits) > 64_000_000
        {
            cache.clear();
        }
        cache.insert(key, value.clone());
        value
    };
    if sign == -1 {
        value.im = -value.im;
    }
    Ok(value)
}

type Expansion = BTreeMap<(Vec<i8>, usize), Rational>;

fn remove_trailing_zeros(word: &[i8], memo: &mut BTreeMap<Vec<i8>, Expansion>) -> Expansion {
    if let Some(result) = memo.get(word) {
        return result.clone();
    }
    let trailing = word.iter().rev().take_while(|&&i| i == 0).count();
    let mut result = Expansion::new();
    if trailing == 0 {
        result.insert((word.to_vec(), 0), Rational::one());
    } else if trailing == word.len() {
        let factorial = (1..=word.len() as i64).product::<i64>();
        result.insert((vec![], word.len()), Rational::from((1, factorial)));
    } else {
        let prefix = &word[..word.len() - 1];
        for ((regular, logs), coefficient) in remove_trailing_zeros(prefix, memo) {
            result.insert((regular, logs + 1), coefficient);
        }
        // The desired word appears `trailing` times in the shuffle with [0].
        // Every other insertion has one fewer trailing zero, ensuring termination.
        for position in 0..word.len() - trailing {
            let mut inserted = prefix.to_vec();
            inserted.insert(position, 0);
            for (key, coefficient) in remove_trailing_zeros(&inserted, memo) {
                *result.entry(key).or_insert_with(Rational::zero) -= coefficient;
            }
        }
        for coefficient in result.values_mut() {
            *coefficient /= Rational::from(trailing as i64);
        }
        result.retain(|_, coefficient| !coefficient.is_zero());
    }
    memo.insert(word.to_vec(), result.clone());
    result
}

/// Poles after scaling the endpoint to one, and their Hölder complements.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Pole {
    Zero,
    Plus,
    Minus,
    One,
    ComplementPlus,
    ComplementMinus,
}

impl Pole {
    fn original(index: i8) -> Self {
        match index {
            0 => Self::Zero,
            1 => Self::Plus,
            -1 => Self::Minus,
            _ => unreachable!(),
        }
    }
    fn complement(self) -> Self {
        match self {
            Self::Zero => Self::One,
            Self::Plus => Self::ComplementPlus,
            Self::Minus => Self::ComplementMinus,
            _ => unreachable!(),
        }
    }

    /// Multiply fixed-point A+B*i*sqrt(3) by 1/(2*pole), rounding each
    /// component toward zero. All matrix row norms are at most one; each
    /// division contributes less than one fixed-point unit of error.
    fn multiply_half_inverse(self, value: Fixed) -> Fixed {
        let Fixed { a, b } = value;
        match self {
            Self::Plus => Fixed {
                a: (&a - &b * 3) / 4,
                b: (a + b) / 4,
            },
            Self::Minus => Fixed {
                a: (&b * 3 - &a) / 4,
                b: -(a + b) / 4,
            },
            Self::One => Fixed { a: a / 2, b: b / 2 },
            Self::ComplementPlus => Fixed {
                a: (&a + &b * 3) / 4,
                b: (b - a) / 4,
            },
            Self::ComplementMinus => Fixed {
                a: (&a - &b) / 4,
                b: (b * 3 + a) / 12,
            },
            Self::Zero => unreachable!(),
        }
    }
}

#[derive(Clone)]
struct Fixed {
    a: Integer,
    b: Integer,
}
impl Default for Fixed {
    fn default() -> Self {
        Self {
            a: Integer::zero(),
            b: Integer::zero(),
        }
    }
}
impl Fixed {
    fn sub(&self, rhs: &Self) -> Self {
        Self {
            a: &self.a - &rhs.a,
            b: &self.b - &rhs.b,
        }
    }
    fn div(&self, n: i64) -> Self {
        Self {
            a: &self.a / n,
            b: &self.b / n,
        }
    }
    fn add_assign(&mut self, rhs: &Self) {
        self.a += &rhs.a;
        self.b += &rhs.b;
    }
}

struct Node {
    pole: Pole,
    suffix: usize,
    weight: usize,
}

/// A reusable batch of sixth-root HPLs with a shared suffix-series graph.
///
/// Batch evaluation avoids recalculating common GPL suffixes. Output order
/// matches input order, including repeated words. Precision is in binary bits;
/// no precomputed numerical constants or external process are used.
pub struct SixthRootHplEvaluator {
    expansions: Vec<Expansion>,
    regular: BTreeMap<Vec<i8>, Vec<(usize, usize, i8)>>,
    nodes: Vec<Node>,
}

impl SixthRootHplEvaluator {
    /// Prepare up to 4096 words, each of weight at most six.
    pub fn new(words: &[Vec<i8>]) -> Result<Self, String> {
        if words.len() > MAX_WORDS {
            return Err("sixth-root HPL batch exceeds 4096 words".into());
        }
        for word in words {
            validate_word(word)?;
        }
        let mut memo = BTreeMap::new();
        let expansions = words
            .iter()
            .map(|w| remove_trailing_zeros(w, &mut memo))
            .collect::<Vec<_>>();
        let regular_words = expansions
            .iter()
            .flat_map(|e| e.keys().map(|(w, _)| w.clone()))
            .collect::<BTreeSet<_>>();
        let mut nodes = vec![];
        let mut indices = BTreeMap::from([(vec![], 0)]);
        fn intern(
            word: &[Pole],
            nodes: &mut Vec<Node>,
            indices: &mut BTreeMap<Vec<Pole>, usize>,
        ) -> usize {
            if let Some(&index) = indices.get(word) {
                return index;
            }
            let suffix = intern(&word[1..], nodes, indices);
            let index = nodes.len() + 1;
            nodes.push(Node {
                pole: word[0],
                suffix,
                weight: word.len(),
            });
            indices.insert(word.to_vec(), index);
            index
        }
        let mut regular = BTreeMap::new();
        for word in regular_words {
            // G(a1,...,an;1) = sum_j (-1)^j
            // G(1-aj,...,1-a1;1/2) G(a(j+1),...,an;1/2).
            // The right word has no trailing zero; every left pole is nonzero.
            let poles = word.iter().map(|&i| Pole::original(i)).collect::<Vec<_>>();
            let hpl_sign = if word.iter().filter(|&&i| i == 1).count() % 2 == 0 {
                1
            } else {
                -1
            };
            let terms = (0..=word.len())
                .map(|split| {
                    let left = poles[..split]
                        .iter()
                        .rev()
                        .map(|p| p.complement())
                        .collect::<Vec<_>>();
                    (
                        intern(&left, &mut nodes, &mut indices),
                        intern(&poles[split..], &mut nodes, &mut indices),
                        if split % 2 == 0 { hpl_sign } else { -hpl_sign },
                    )
                })
                .collect();
            regular.insert(word, terms);
        }
        Ok(Self {
            expansions,
            regular,
            nodes,
        })
    }

    /// Evaluate at `exp(sign*i*pi/3)`, rounding to the requested precision.
    ///
    /// Series truncation and fixed-point rounding errors are enclosed before
    /// Hölder products and trailing-zero shuffles are combined. Guard precision
    /// is increased when cancellation prevents the requested accuracy. Before
    /// final rounding, very small components have an absolute error bound of
    /// 2^(-bits-32); the returned floats are then rounded to `bits` precision.
    pub fn evaluate(&self, sign: i8, bits: u32) -> Result<Vec<Complex<Float>>, String> {
        validate_sign(sign)?;
        if !(2..=MAX_BITS).contains(&bits) {
            return Err(format!(
                "sixth-root HPL precision must be between 2 and {MAX_BITS} bits"
            ));
        }
        if self.expansions.is_empty() {
            return Ok(vec![]);
        }
        let mut guard = 64;
        for _ in 0..4 {
            let work = bits.checked_add(guard).ok_or("HPL precision overflow")?;
            let balls = self.evaluate_balls(work)?;
            if balls
                .iter()
                .all(|b| component_is_accurate(&b.re, bits) && component_is_accurate(&b.im, bits))
            {
                return Ok(balls
                    .into_iter()
                    .map(|ball| {
                        let mut value = Complex::new(ball.re.center, ball.im.center);
                        value.re.set_prec(bits);
                        value.im.set_prec(bits);
                        if sign == -1 {
                            value.im = -value.im;
                        }
                        value
                    })
                    .collect());
            }
            guard *= 2;
        }
        Err("sixth-root HPL could not enclose the requested accuracy after increasing guard precision".into())
    }

    fn evaluate_balls(&self, bits: u32) -> Result<Vec<ComplexBall>, String> {
        let weight = self.nodes.iter().map(|n| n.weight).max().unwrap_or(1);
        let mut n_terms = bits.max((4 * weight) as u32);
        // |coefficient_n| <= (n+1)^(weight-1). For n >= 4*weight the
        // successive majorant ratio is <3/4, giving a geometric tail bound.
        while n_terms + 1 < bits + 2 + (weight as u32 - 1) * ceil_log2(n_terms + 2) {
            n_terms += 1;
        }
        if u64::from(n_terms) * self.nodes.len() as u64 > MAX_SERIES_WORK {
            return Err(
                "sixth-root HPL series work limit exceeded; split the batch or reduce precision"
                    .into(),
            );
        }
        let scale = Integer::one() << bits;
        let mut previous = vec![Fixed::default(); self.nodes.len() + 1];
        let mut current = previous.clone();
        let mut states = previous.clone();
        let mut sums = previous.clone();
        previous[0].a = scale.clone();
        sums[0].a = scale.clone();
        for n in 1..=n_terms {
            for (offset, node) in self.nodes.iter().enumerate() {
                let index = offset + 1;
                // b_w(n) = [t^n] G(w;t) / 2^n. For a nonzero first pole a,
                // n*b_w(n) = (1/(2*a))*((n-1)*b_w(n-1)-b_suffix(n-1)).
                // A zero first pole instead divides b_suffix(n) by n.
                current[index] = if node.pole == Pole::Zero {
                    current[node.suffix].div(i64::from(n))
                } else {
                    states[index] = node
                        .pole
                        .multiply_half_inverse(states[index].sub(&previous[node.suffix]));
                    states[index].div(i64::from(n))
                };
                sums[index].add_assign(&current[index]);
            }
            std::mem::swap(&mut previous, &mut current);
            current[0] = Fixed::default();
        }
        let rational = |value: Integer| Rational::from((value, scale.clone()));
        let sqrt_integer =
            Integer::from((Integer::from(3) << (2 * bits)).to_multi_prec().root_ref(2));
        let sqrt3 = RealBall::from_rational_bounds(
            &rational(sqrt_integer.clone()),
            &rational(sqrt_integer + 1),
            bits,
        );
        let zero = RealBall::exact(Float::new(bits));
        let one = RealBall::exact(Float::with_val(bits, 1));
        let gpls = sums
            .into_iter()
            .enumerate()
            .map(|(index, value)| {
                // If suffix coefficient error <=2*(d-1) ulps, the n*a_n state
                // accumulates <=n*(2*d-1) ulps, and division by n adds <1.
                // Thus each coefficient has <=2*d ulps error, and the sum <=2*d*N.
                // The additional ulp encloses the exact analytic tail.
                let error = if index == 0 {
                    0
                } else {
                    2 * self.nodes[index - 1].weight as u64 * u64::from(n_terms) + 1
                };
                let radius = rational(Integer::from(error));
                Complex::new(
                    RealBall::from_rational_ball(&rational(value.a), &radius, bits),
                    RealBall::from_rational_ball(&rational(value.b), &radius, bits) * &sqrt3,
                )
            })
            .collect::<Vec<_>>();
        let mut masters = BTreeMap::new();
        for (word, terms) in &self.regular {
            let mut value = Complex::new(zero.clone(), zero.clone());
            for &(left, right, sign) in terms {
                let product = &gpls[left] * &gpls[right];
                if sign == 1 {
                    value += product;
                } else {
                    value -= product;
                }
            }
            masters.insert(word, value);
        }
        let log = Complex::new(
            zero.clone(),
            one.pi() / RealBall::exact(Float::with_val(bits, 3)),
        );
        let mut logs = vec![Complex::new(one, zero.clone())];
        for power in 1..=MAX_WEIGHT {
            logs.push(&logs[power - 1] * &log);
        }
        Ok(self
            .expansions
            .iter()
            .map(|expansion| {
                let mut value = Complex::new(zero.clone(), zero.clone());
                for ((word, power), coefficient) in expansion {
                    let scalar = RealBall::from_rational_ball(coefficient, &Rational::zero(), bits);
                    value += &masters[word] * &logs[*power] * Complex::new(scalar, zero.clone());
                }
                value
            })
            .collect())
    }
}

fn ceil_log2(value: u32) -> u32 {
    u32::BITS - (value - 1).leading_zeros()
}

fn component_is_accurate(value: &RealBall, bits: u32) -> bool {
    if !value.is_finite() {
        return false;
    }
    let unit = Rational::from((Integer::one(), Integer::one() << (bits + 8)));
    let scale = value.center.norm();
    let floor = Float::from_rational_round(
        &Rational::from((Integer::one(), Integer::one() << 24u32)),
        bits,
        crate::domains::float::RoundingDirection::Down,
    );
    let tolerance =
        Float::from_rational_round(&unit, bits, crate::domains::float::RoundingDirection::Down)
            * if scale < floor { floor } else { scale };
    value.radius <= tolerance
}

#[cfg(test)]
mod tests;
