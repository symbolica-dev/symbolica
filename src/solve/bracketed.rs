//! Prepared numerical scalar refinement. No symbolic work is performed here.
use numerica::domains::float::{Real, RealLike};

/// Coordinate tolerances are expressed in the callback's native number domain.
/// Use zero absolute tolerance when small nonzero roots require relative accuracy.
#[derive(Clone, Debug)]
pub struct BracketedRootOptions<N> {
    pub absolute_tolerance: N,
    pub relative_tolerance: N,
    pub max_iterations: usize,
    pub initial_guess: Option<N>,
    pub convergence: BracketedRootConvergence,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BracketedRootConvergence {
    /// Stop on the numerical bracket width or an evaluated numerical zero.
    #[default]
    Bracket,
    /// Also allow a small local Newton correction. This need not bound the
    /// distance to a root for a general function with rapidly varying derivative.
    NewtonOrBracket,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BracketedRootTermination {
    /// The evaluated value has zero numerical centre; this is not an exact proof.
    NumericalZero,
    /// A local heuristic, not a general root-distance guarantee.
    NewtonCorrection,
    BracketWidth,
}

/// An ordinary numerical sign bracket, not a certified interval enclosure.
/// Tracked domains retain their arithmetic uncertainty in `root`; neither this
/// bracket nor the stopping criteria certify that uncertainty or uniqueness.
#[derive(Clone, Debug)]
pub struct BracketedRoot<N> {
    pub root: N,
    pub lower: N,
    pub upper: N,
    pub lower_value: N,
    pub upper_value: N,
    pub residual: N,
    pub iterations: usize,
    pub evaluations: usize,
    pub termination: BracketedRootTermination,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BracketedRootError {
    InvalidInput(&'static str),
    NotBracketed,
    NonFinite,
    SingularRootDerivative,
    Stagnation,
    IterationLimit,
    CounterOverflow,
}

impl std::fmt::Display for BracketedRootError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(reason) => write!(f, "invalid bracketed root input: {reason}"),
            Self::NotBracketed => f.write_str("root endpoint values do not bracket zero"),
            Self::NonFinite => f.write_str("nonfinite bracketed root value or derivative"),
            Self::SingularRootDerivative => f.write_str("root has a zero numerical derivative"),
            Self::Stagnation => f.write_str("root refinement stagnated before reaching tolerance"),
            Self::IterationLimit => f.write_str("root refinement reached its iteration limit"),
            Self::CounterOverflow => f.write_str("root evaluation counter overflow"),
        }
    }
}
impl std::error::Error for BracketedRootError {}

/// Refine a simple real root from an already prepared `(f(x), f'(x))` callback.
/// The caller supplies a continuous function with a root in the sign bracket.
/// Newton steps are safeguarded by bisection; the callback is never constructed,
/// differentiated or optimized here. Endpoints may themselves be roots.
///
/// Every operation stays in `N`. Even a zero numerical centre receives a Newton
/// correction before return, so error-tracked coefficient inputs are not replaced
/// by an exact endpoint/initial-guess constant. The tracker remains heuristic;
/// iteration decisions use its centre comparisons, not certified interval signs.
/// A caller requiring certification must independently enclose endpoint signs.
pub fn nsolve_bracketed<N, F>(
    mut lower: N,
    mut upper: N,
    options: &BracketedRootOptions<N>,
    mut evaluate: F,
) -> Result<BracketedRoot<N>, BracketedRootError>
where
    N: RealLike + Real + PartialOrd,
    F: FnMut(&N) -> (N, N),
{
    use BracketedRootError as Error;
    let zero = lower.zero();
    if !lower.is_finite() || !upper.is_finite() || lower >= upper {
        return Err(Error::InvalidInput(
            "finite strictly ordered endpoints are required",
        ));
    }
    let abs = &options.absolute_tolerance;
    let rel = &options.relative_tolerance;
    if !abs.is_finite()
        || !rel.is_finite()
        || abs < &zero
        || rel < &zero
        || (abs.is_zero() && rel.is_zero())
        || options.max_iterations == 0
    {
        return Err(Error::InvalidInput(
            "nonnegative finite tolerances, one positive, and a nonzero iteration limit are required",
        ));
    }
    if let Some(initial) = &options.initial_guess {
        if !initial.is_finite() || initial < &lower || initial > &upper {
            return Err(Error::InvalidInput(
                "initial guess lies outside the finite bracket",
            ));
        }
    }
    let two = lower.from_usize(2);
    let mut evaluations = 0usize;
    let mut checked = |point: &N| {
        evaluations = evaluations.checked_add(1).ok_or(Error::CounterOverflow)?;
        let (value, derivative) = evaluate(point);
        if value.is_finite() && derivative.is_finite() {
            Ok((value, derivative))
        } else {
            Err(Error::NonFinite)
        }
    };
    let (mut lower_value, lower_derivative) = checked(&lower)?;
    let (mut upper_value, upper_derivative) = checked(&upper)?;
    let endpoint = if lower_value.is_zero() {
        Some((lower.clone(), lower_value.clone(), lower_derivative))
    } else if upper_value.is_zero() {
        Some((upper.clone(), upper_value.clone(), upper_derivative))
    } else {
        None
    };
    let (mut current, mut current_evaluation) = if let Some((point, value, derivative)) = endpoint {
        (point, Some((value, derivative)))
    } else {
        if (lower_value > zero) == (upper_value > zero) {
            return Err(Error::NotBracketed);
        }
        (
            options
                .initial_guess
                .clone()
                .unwrap_or_else(|| lower.clone() / &two + upper.clone() / &two),
            None,
        )
    };
    let mut previous_step = upper.clone() / &two - lower.clone() / &two;
    for iteration in 0..options.max_iterations {
        let (value, derivative) = match current_evaluation.take() {
            Some(values) => values,
            None => checked(&current)?,
        };
        let numerical_zero = value.is_zero();
        if numerical_zero && derivative.is_zero() {
            return Err(Error::SingularRootDerivative);
        }
        if !numerical_zero {
            if (value > zero) == (lower_value > zero) {
                lower = current.clone();
                lower_value = value.clone();
            } else {
                upper = current.clone();
                upper_value = value.clone();
            }
        }
        let tolerance = abs.clone() + rel.clone() * current.norm();
        if !tolerance.is_finite() {
            return Err(Error::NonFinite);
        }
        let newton = if derivative.is_zero() {
            None
        } else {
            let correction = value / derivative;
            let candidate = current.clone() - &correction;
            (correction.is_finite() && candidate.is_finite()).then_some((candidate, correction))
        };
        if let Some((candidate, correction)) = &newton {
            // A final rounded Newton correction may land just outside a narrow
            // sign bracket. Admit it only if the bracket enlarged to contain
            // it still meets the coordinate tolerance. Keeping the correction
            // preserves tracked coefficient uncertainty at the returned root.
            let enclosing_lower = if candidate < &lower {
                candidate
            } else {
                &lower
            };
            let enclosing_upper = if candidate > &upper {
                candidate
            } else {
                &upper
            };
            let enclosing_half_width =
                enclosing_upper.clone() / &two - enclosing_lower.clone() / &two;
            let termination = if numerical_zero {
                Some(BracketedRootTermination::NumericalZero)
            } else if options.convergence == BracketedRootConvergence::NewtonOrBracket
                && correction.norm() <= tolerance
                && !tolerance.is_zero()
                && candidate >= &lower
                && candidate <= &upper
            {
                Some(BracketedRootTermination::NewtonCorrection)
            } else if enclosing_half_width <= tolerance.clone() / &two && !tolerance.is_zero() {
                Some(BracketedRootTermination::BracketWidth)
            } else {
                None
            };
            if let Some(termination) = termination {
                let (residual, _) = checked(candidate)?;
                if candidate < &lower {
                    if !residual.is_zero() && (residual > zero) != (lower_value > zero) {
                        upper = lower;
                        upper_value = lower_value;
                    }
                    lower = candidate.clone();
                    lower_value = residual.clone();
                } else if candidate > &upper {
                    if !residual.is_zero() && (residual > zero) != (upper_value > zero) {
                        lower = upper;
                        lower_value = upper_value;
                    }
                    upper = candidate.clone();
                    upper_value = residual.clone();
                }
                return Ok(BracketedRoot {
                    root: candidate.clone(),
                    lower,
                    upper,
                    lower_value,
                    upper_value,
                    residual,
                    iterations: iteration + 1,
                    evaluations,
                    termination,
                });
            }
        }
        // Requiring successive Newton corrections to contract avoids a run of
        // arbitrarily poor in-bracket steps. Rejected steps use the safe midpoint.
        let next = match newton {
            Some((candidate, correction))
                if candidate > lower
                    && candidate < upper
                    && correction.norm() <= previous_step.clone() / &two =>
            {
                candidate
            }
            _ => lower.clone() / &two + upper.clone() / &two,
        };
        if next == current || next <= lower || next >= upper {
            return Err(Error::Stagnation);
        }
        previous_step = (next.clone() - &current).norm();
        current = next;
    }
    Err(Error::IterationLimit)
}
