//! Endpoint-one GPLs in the unit sixth-root alphabet used by vacuum periods.
//!
//! This is the ordinary G convention, with kernel 1/(t-a), on the real path
//! from zero to one. No HPL sign conversion is applied.

use super::*;
use crate::atom::AtomCore;

/// Exact letters admitted by the endpoint-one vacuum-period GPL evaluator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GplLetter {
    Zero,
    One,
    MinusOne,
    /// (1+i sqrt(3))/2, with positive real sqrt(3).
    R1,
    /// (-1+i sqrt(3))/2, with positive real sqrt(3).
    R2,
    /// (-1-i sqrt(3))/2, with positive real sqrt(3).
    R4,
    /// (1-i sqrt(3))/2, with positive real sqrt(3).
    R5,
}

static GPL: SymbolCache<Symbol> =
    SymbolCache::new(|| get_symbol!("gpl_at_one").expect("endpoint-one GPL was initialized"));
static LETTERS: SymbolCache<Symbol> =
    SymbolCache::new(|| get_symbol!("gpl_letters").expect("GPL letter tag was initialized"));

#[cfg(not(doctest))]
crate::_inventory::submit! {
    StateInitializer::new("symbolica::endpoint_one_gpl", || {
        let _ = symbol!("gpl_letters";);
        let _ = symbol!(
            "gpl_at_one",
            eval = EvaluationInfo::constant(|tags, bits| {
                if !(2..=MAX_BITS).contains(&bits) {
                    return Err(format!("endpoint-one GPL precision must be between 2 and {MAX_BITS} bits"));
                }
                let word = parse_tags(tags)?;
                let work = bits.checked_add(64).ok_or("GPL precision overflow")?;
                let mut value = EndpointOneGplEvaluator::new(&word)?.evaluate_ball(work)?;
                value.re.center.set_prec(bits);
                value.im.center.set_prec(bits);
                Ok(Complex::new(value.re.center, value.im.center))
            }).with_tags(1).register_tracked(|tags, arguments, bits| {
                if !arguments.is_empty() {
                    return Err("endpoint-one GPL expects one symbolic letter tag".into());
                }
                let word = parse_tags(tags)?;
                Ok(TrackedComplex::from_ball(
                    EndpointOneGplEvaluator::new(&word)?.evaluate_ball(bits)?,
                    ErrorQuality::Bounded,
                ))
            })
        );
    }, &["symbolica"])
}

/// The tagged constant `gpl_at_one(gpl_letters(a1,...,an))`.
///
/// Supports words through weight six over zero and the sixth roots of unity,
/// using the ordinary kernel 1/(t-a). Nonempty words must start with a letter
/// other than one and end with a nonzero letter. No analytic continuation or
/// trailing-zero regularization is performed.
pub fn gpl_at_one() -> Symbol {
    *GPL
}

impl GplLetter {
    fn atom(self) -> Atom {
        match self {
            Self::Zero => Atom::num(0),
            Self::One => Atom::num(1),
            Self::MinusOne => Atom::num(-1),
            Self::R1 | Self::R2 | Self::R4 | Self::R5 => {
                let real = if matches!(self, Self::R1 | Self::R5) {
                    1
                } else {
                    -1
                };
                let imaginary = if matches!(self, Self::R1 | Self::R2) {
                    1
                } else {
                    -1
                };
                (Atom::num(real)
                    + Atom::num(Complex::new(Rational::zero(), Rational::from(imaginary)))
                        * Atom::num(3).pow(Atom::num((1, 2))))
                    / 2
            }
        }
    }
}

/// Construct a supported endpoint-one GPL constant with exact letter tags.
pub fn endpoint_one_gpl(word: &[GplLetter]) -> Result<Atom, String> {
    validate(word)?;
    let letters = word.iter().map(|letter| letter.atom()).collect::<Vec<_>>();
    Ok(GPL.call(LETTERS.call(&letters)))
}

fn validate(word: &[GplLetter]) -> Result<(), String> {
    use GplLetter::*;
    if word.len() <= MAX_WEIGHT && word.first() != Some(&One) && word.last() != Some(&Zero) {
        Ok(())
    } else {
        Err(
            "endpoint-one GPL requires weight at most six, a first letter other than one and a nonzero final letter"
                .into(),
        )
    }
}

fn parse_tags(tags: &[AtomView]) -> Result<Vec<GplLetter>, String> {
    let [AtomView::Fun(letters)] = tags else {
        return Err("endpoint-one GPL requires gpl_letters(...) as its symbolic tag".into());
    };
    if letters.get_symbol() != *LETTERS {
        return Err("endpoint-one GPL requires gpl_letters(...) as its symbolic tag".into());
    }
    let alphabet = [
        GplLetter::Zero,
        GplLetter::One,
        GplLetter::MinusOne,
        GplLetter::R1,
        GplLetter::R2,
        GplLetter::R4,
        GplLetter::R5,
    ]
    .map(|letter| (letter.atom().expand(), letter));
    let word = letters
        .iter()
        .map(|letter| {
            let expanded = letter.expand();
            alphabet
                .iter()
                .find_map(|(atom, tag)| (atom == &expanded).then_some(*tag))
                .ok_or_else(|| format!("unsupported endpoint-one GPL letter: {letter}"))
        })
        .collect::<Result<Vec<_>, _>>()?;
    validate(&word)?;
    Ok(word)
}

/// A shared-suffix Hölder evaluation of one supported endpoint-one GPL.
pub struct EndpointOneGplEvaluator {
    nodes: Vec<Node>,
    terms: Vec<(usize, usize, i8)>,
}

impl EndpointOneGplEvaluator {
    pub fn new(word: &[GplLetter]) -> Result<Self, String> {
        validate(word)?;
        // Evaluate conjugated letters in the exact quadratic field
        // using the existing recurrence, then conjugate the complete result.
        let poles = word
            .iter()
            .map(|letter| match letter {
                GplLetter::Zero => Pole::Zero,
                GplLetter::One => Pole::One,
                GplLetter::MinusOne => Pole::NegativeOne,
                GplLetter::R1 => Pole::Plus,
                GplLetter::R2 => Pole::ConjugateMinus,
                GplLetter::R4 => Pole::Minus,
                GplLetter::R5 => Pole::ComplementPlus,
            })
            .collect::<Vec<_>>();
        let mut nodes = Vec::new();
        let mut indices = BTreeMap::from([(Vec::new(), 0)]);
        let terms = (0..=word.len())
            .map(|split| {
                let left = poles[..split]
                    .iter()
                    .rev()
                    .map(|p| p.complement())
                    .collect::<Vec<_>>();
                (
                    intern_poles(&left, &mut nodes, &mut indices),
                    intern_poles(&poles[split..], &mut nodes, &mut indices),
                    if split % 2 == 0 { 1 } else { -1 },
                )
            })
            .collect();
        Ok(Self { nodes, terms })
    }

    /// Evaluate once at the given working precision, preserving error radii.
    pub fn evaluate_ball(&self, bits: u32) -> Result<ComplexBall, String> {
        if !(2..=MAX_BITS).contains(&bits) {
            return Err(format!(
                "endpoint-one GPL precision must be between 2 and {MAX_BITS} bits"
            ));
        }
        let values = evaluate_gpl_nodes(&self.nodes, bits)?;
        let zero = RealBall::exact(Float::new(bits));
        let mut value = Complex::new(zero.clone(), zero);
        for &(left, right, sign) in &self.terms {
            let product = &values[left] * &values[right];
            if sign == 1 {
                value += product;
            } else {
                value -= product;
            }
        }
        Ok(value.conj())
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod alphabet_tests;
