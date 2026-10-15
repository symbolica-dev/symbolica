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

/// A finite native Newton proposal observed before further iteration.
///
/// `N` is borrowed in the observer and owned in a caller-stopped result. The
/// corrected point need not be inside the numerical bracket. Neither the
/// bracket, proposal nor any tracked errors certify a mathematical root.
#[derive(Clone, Debug)]
pub struct BracketedRootCandidate<N> {
    pub current: N,
    pub corrected: N,
    pub correction: N,
    pub lower: N,
    pub upper: N,
    pub lower_value: N,
    pub upper_value: N,
    pub tolerance: N,
    pub iterations: usize,
    pub evaluations: usize,
}
impl<N: Clone> BracketedRootCandidate<&N> {
    fn into_owned(self) -> BracketedRootCandidate<N> {
        BracketedRootCandidate {
            current: self.current.clone(),
            corrected: self.corrected.clone(),
            correction: self.correction.clone(),
            lower: self.lower.clone(),
            upper: self.upper.clone(),
            lower_value: self.lower_value.clone(),
            upper_value: self.upper_value.clone(),
            tolerance: self.tolerance.clone(),
            iterations: self.iterations,
            evaluations: self.evaluations,
        }
    }
}

/// Ordinary numerical convergence or an explicit caller interruption.
///
/// A caller can interrupt for any reason. Its payload is not verified by this
/// solver, and `CallerStopped` does not assert convergence or certification.
#[derive(Clone, Debug)]
pub enum ObservedBracketedRoot<N, C> {
    Converged(BracketedRoot<N>),
    CallerStopped {
        candidate: BracketedRootCandidate<N>,
        payload: C,
    },
}

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
    lower: N,
    upper: N,
    options: &BracketedRootOptions<N>,
    evaluate: F,
) -> Result<BracketedRoot<N>, BracketedRootError>
where
    N: RealLike + Real + PartialOrd,
    F: FnMut(&N) -> (N, N),
{
    match nsolve_bracketed_observed(lower, upper, options, evaluate, |_| {
        std::ops::ControlFlow::<std::convert::Infallible>::Continue(())
    })? {
        ObservedBracketedRoot::Converged(root) => Ok(root),
        ObservedBracketedRoot::CallerStopped { payload, .. } => match payload {},
    }
}

/// Refine using the same native iteration, with an optional caller-stop hook.
///
/// The observer sees a finite Newton proposal only when ordinary numerical
/// convergence has not already completed. Returning `Continue(())` leaves the
/// arithmetic and callback-evaluation sequence unchanged. A zero derivative
/// away from a root still falls back without an observation; a singular zero
/// remains an error. Convergence and errors already reached take precedence; a
/// caller stop can precede a later stagnation or iteration-limit error.
///
/// `Break(payload)` retains the actual native candidate, including any tracked
/// uncertainty, and returns `CallerStopped`. The caller must check whatever
/// condition its payload claims, including the actual proposed return value.
/// No candidate replacement, interval certification or uncertainty reset occurs.
/// The hook owns its work budget and may also stop for cancellation. No state
/// accumulates in the solver, and no observer is retained after return.
pub fn nsolve_bracketed_observed<N, F, O, C>(
    mut lower: N,
    mut upper: N,
    options: &BracketedRootOptions<N>,
    mut evaluate: F,
    mut observe: O,
) -> Result<ObservedBracketedRoot<N, C>, BracketedRootError>
where
    N: RealLike + Real + PartialOrd,
    F: FnMut(&N) -> (N, N),
    O: FnMut(BracketedRootCandidate<&N>) -> std::ops::ControlFlow<C>,
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
    if let Some(initial) = &options.initial_guess
        && (!initial.is_finite() || initial < &lower || initial > &upper)
    {
        return Err(Error::InvalidInput(
            "initial guess lies outside the finite bracket",
        ));
    }
    let two = lower.from_usize(2);
    let mut evaluations = 0usize;
    let mut checked = |point: &N, evaluations: &mut usize| {
        *evaluations = evaluations.checked_add(1).ok_or(Error::CounterOverflow)?;
        let (value, derivative) = evaluate(point);
        if value.is_finite() && derivative.is_finite() {
            Ok((value, derivative))
        } else {
            Err(Error::NonFinite)
        }
    };
    let (mut lower_value, lower_derivative) = checked(&lower, &mut evaluations)?;
    let (mut upper_value, upper_derivative) = checked(&upper, &mut evaluations)?;
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
            None => checked(&current, &mut evaluations)?,
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
                let (residual, _) = checked(candidate, &mut evaluations)?;
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
                return Ok(ObservedBracketedRoot::Converged(BracketedRoot {
                    root: candidate.clone(),
                    lower,
                    upper,
                    lower_value,
                    upper_value,
                    residual,
                    iterations: iteration + 1,
                    evaluations,
                    termination,
                }));
            }
            let observation = BracketedRootCandidate {
                current: &current,
                corrected: candidate,
                correction,
                lower: &lower,
                upper: &upper,
                lower_value: &lower_value,
                upper_value: &upper_value,
                tolerance: &tolerance,
                iterations: iteration + 1,
                evaluations,
            };
            if let std::ops::ControlFlow::Break(payload) = observe(observation.clone()) {
                return Ok(ObservedBracketedRoot::CallerStopped {
                    candidate: observation.into_owned(),
                    payload,
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

#[cfg(test)]
mod root_tests {
    use crate::solve::{BracketedRootConvergence, BracketedRootOptions, nsolve_bracketed};
    use numerica::domains::float::{DoubleFloat, ErrorPropagatingFloat, Float, Real, RealLike};

    fn options<N: RealLike>(one: &N, tolerance: N) -> BracketedRootOptions<N> {
        BracketedRootOptions {
            absolute_tolerance: one.zero(),
            relative_tolerance: tolerance,
            max_iterations: 256,
            initial_guess: None,
            convergence: BracketedRootConvergence::Bracket,
        }
    }

    fn quadratic<N: RealLike + Real + PartialOrd>(one: N, tolerance: N) {
        let two = one.from_usize(2);
        let settings = options(&one, tolerance.clone());
        let result = nsolve_bracketed(one.zero(), two.clone(), &settings, |x| {
            (x.clone() * x - &two, two.clone() * x)
        })
        .unwrap();
        assert!((result.root.clone() * &result.root - &two).norm() < tolerance * one.from_usize(8));
        assert!(result.lower <= result.root && result.root <= result.upper);
        assert!(result.lower_value <= one.zero() && result.upper_value >= one.zero());
        assert!(result.iterations < 64);
        assert!(result.evaluations >= result.iterations);
    }

    #[test]
    fn native_scalar_domains_keep_their_own_precision() {
        quadratic(1f64, 1e-14);
        quadratic(DoubleFloat::from(1.), DoubleFloat::from(1e-29));
        quadratic(Float::with_val(192, 1), Float::with_val(192, 1e-50));
    }

    #[test]
    fn center_zero_and_endpoint_roots_retain_coefficient_uncertainty() {
        tracked_control(1f64);
        tracked_control(DoubleFloat::from(1.));
        tracked_control(Float::with_val(192, 1));
    }

    fn tracked_control<N: RealLike + Real + PartialOrd>(unit: N) {
        let exact = |x| ErrorPropagatingFloat::new_with_accuracy(x, f64::INFINITY);
        for coefficient in [1, 2, 4] {
            let coefficient = ErrorPropagatingFloat::new(unit.from_usize(coefficient), 6.);
            let one = exact(unit.clone());
            let two = exact(unit.from_usize(2));
            let settings = options(
                &one,
                exact(unit.clone() / unit.from_i64(100_000_000_000_000)),
            );
            let result = nsolve_bracketed(exact(unit.zero()), one.clone(), &settings, |x| {
                (
                    coefficient.clone() * x * x - &one,
                    two.clone() * &coefficient * x,
                )
            })
            .unwrap();
            let expected = unit.clone() / coefficient.get_num().sqrt();
            assert!((result.root.get_num().clone() - &expected).norm().to_f64() < 1e-13);
            // Local dr/da = -r/(2a): uncertain coefficients must not produce
            // an allegedly exact endpoint or exact initial-guess root.
            let local_error = expected.to_f64() * coefficient.get_relative_error() / 2.;
            assert!(result.root.get_absolute_error() >= 0.99 * local_error);
            assert!(result.root.get_absolute_error().is_finite());
        }
    }
}

#[cfg(test)]
mod observer_tests {
    use crate::solve::{
        BracketedRootConvergence, BracketedRootOptions, ObservedBracketedRoot, nsolve_bracketed,
        nsolve_bracketed_observed,
    };
    use numerica::domains::float::{DoubleFloat, ErrorPropagatingFloat, Float, Real, RealLike};
    use std::{ops::ControlFlow, sync::Arc};

    fn options<N: RealLike>(one: &N, tolerance: N) -> BracketedRootOptions<N> {
        BracketedRootOptions {
            absolute_tolerance: one.zero(),
            relative_tolerance: tolerance,
            max_iterations: 1024,
            initial_guess: None,
            convergence: BracketedRootConvergence::Bracket,
        }
    }
    fn unchanged<N: RealLike + Real + PartialOrd + std::fmt::Debug>(one: N, tolerance: N) {
        let settings = options(&one, tolerance);
        let two = one.from_usize(2);
        let mut ordinary = vec![];
        let first = nsolve_bracketed(one.zero(), two.clone(), &settings, |x| {
            ordinary.push(format!("{x:?}"));
            (x.clone() * x - &two, two.clone() * x)
        })
        .unwrap();
        let mut observed = vec![];
        let mut count = 0;
        let second = nsolve_bracketed_observed(
            one.zero(),
            two.clone(),
            &settings,
            |x| {
                observed.push(format!("{x:?}"));
                (x.clone() * x - &two, two.clone() * x)
            },
            |c| {
                count += 1;
                assert!(c.corrected.is_finite());
                ControlFlow::<()>::Continue(())
            },
        )
        .unwrap();
        let ObservedBracketedRoot::Converged(second) = second else {
            panic!("always-continue interrupted")
        };
        assert_eq!(ordinary, observed);
        assert!(count > 0);
        assert_eq!(format!("{first:?}"), format!("{second:?}"));
    }
    #[test]
    fn always_continue_keeps_native_domains_and_sequence() {
        unchanged(1f64, 1e-14);
        unchanged(DoubleFloat::from(1.), DoubleFloat::from(1e-28));
        unchanged(Float::with_val(192, 1), Float::with_val(192, 1e-50));
        unchanged(
            ErrorPropagatingFloat::new_with_accuracy(1f64, f64::INFINITY),
            ErrorPropagatingFloat::new_with_accuracy(1e-14, f64::INFINITY),
        );
        unchanged(
            ErrorPropagatingFloat::new_with_accuracy(Float::with_val(192, 1), f64::INFINITY),
            ErrorPropagatingFloat::new_with_accuracy(Float::with_val(192, 1e-50), f64::INFINITY),
        );
    }
    #[test]
    fn caller_stop_retains_native_uncertainty_and_owned_payload() {
        type E = ErrorPropagatingFloat<f64>;
        let exact = |v| E::new_with_accuracy(v, f64::INFINITY);
        let coefficient = E::new(2., 8.);
        let settings = options(&exact(1.), exact(1e-14));
        let mut calls = 0;
        let owner = Arc::new(17);
        let weak = Arc::downgrade(&owner);
        let result = nsolve_bracketed_observed(
            exact(0.),
            exact(2.),
            &settings,
            |x| {
                calls += 1;
                (*x * x - coefficient, exact(2.) * x)
            },
            move |c| {
                assert_eq!(*owner, 17);
                assert!(c.corrected.get_absolute_error() > 0.);
                ControlFlow::Break(String::from("caller budget, not convergence"))
            },
        )
        .unwrap();
        let ObservedBracketedRoot::CallerStopped { candidate, payload } = result else {
            panic!("caller stop lost")
        };
        assert_eq!(payload, "caller budget, not convergence");
        assert!(weak.upgrade().is_none());
        assert_eq!(candidate.evaluations, calls);
        assert_eq!(calls, 3);
        assert_eq!(candidate.iterations, 1);
        assert!(candidate.corrected.get_absolute_error() >= coefficient.get_absolute_error() / 2.);
        assert_ne!(
            *candidate.corrected.get_num() * candidate.corrected.get_num(),
            2.
        );
    }
}
