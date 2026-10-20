//! Direct evaluation with one native function cache across an expression vector.
use std::{collections::HashMap as StdHashMap, hash::BuildHasher};

use super::{EvaluationDomain, EvaluationError};
use crate::{
    atom::{AtomCore, AtomView, KeyLookup},
    domains::float::Real,
};

/// Values and executed user functions from one direct numerical evaluation.
///
/// Cache keys borrow the input expressions. The cache is private and belongs
/// to this completed call only: it cannot be reused with another point, numeric
/// domain or precision. Inspecting it is optional and does not copy numeric
/// values or expressions.
#[derive(Debug)]
pub struct DirectEvaluation<'a, T> {
    values: Vec<T>,
    cache: ahash::HashMap<AtomView<'a>, T>,
}

impl<'a, T> DirectEvaluation<'a, T> {
    /// Numerical results in the same order as the input expressions.
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Take the numerical results and discard the private function cache.
    pub fn into_values(self) -> Vec<T> {
        self.values
    }

    /// Values of registered user-function calls that were actually evaluated.
    ///
    /// Each native function Atom appears once, including constant or tagged
    /// user functions. Iteration order is unspecified. This is not a trace of
    /// all arithmetic: built-in functions, variables, explicit function-value
    /// map overrides and calls in unselected `if` branches are not included.
    /// Entries preserve the original numeric domain, precision and tracking.
    pub fn function_values(&self) -> impl Iterator<Item = (AtomView<'a>, &T)> {
        self.cache
            .iter()
            .filter_map(|(atom, value)| atom.as_fun_view().is_some().then_some((*atom, value)))
    }
}

/// Evaluate expressions directly at one immutable point and precision, sharing
/// the native user-function cache across the complete output vector.
///
/// This uses the same evaluation rules as [`AtomCore::evaluate_with_prec`],
/// without constructing or optimizing an evaluator. Native `if` branches stay
/// lazy, and nested functions are evaluated in dependency order. Repeated
/// identical user-function calls are evaluated once across all outputs. As
/// with scalar direct evaluation, callback results must depend only on their
/// arguments and the selected evaluation domain for that call.
///
/// All variables and functions without evaluation hooks must occur in `map`.
/// Explicit custom-function map values take precedence over their hooks. Any
/// output error returns `Err`, never partial output/cache success. Nonfinite
/// numeric values retain the scalar API's behavior.
///
/// # Example
/// ```
/// use std::collections::HashMap;
/// use symbolica::{atom::AtomCore, evaluate::evaluate_multiple_with_prec, parse};
///
/// let expressions = [parse!("f(x) + 1"), parse!("2*f(x)")];
/// let point = HashMap::from([(parse!("f(x)"), 3.0)]);
/// let result = evaluate_multiple_with_prec(&expressions, &point, 53).unwrap();
/// assert_eq!(result.values(), &[4.0, 6.0]);
/// assert_eq!(result.function_values().count(), 0); // Explicit map override.
/// ```
pub fn evaluate_multiple_with_prec<'a, E, A, T>(
    expressions: &'a [E],
    map: &StdHashMap<A, T, impl BuildHasher>,
    binary_prec: u32,
) -> Result<DirectEvaluation<'a, T>, EvaluationError>
where
    E: AtomCore,
    A: AtomCore + KeyLookup,
    T: Real + EvaluationDomain,
{
    let mut cache = ahash::HashMap::default();
    let values = expressions
        .iter()
        .map(|expression| {
            expression
                .as_atom_view()
                .evaluate_impl(map, &mut cache, binary_prec)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(DirectEvaluation { values, cache })
}

#[cfg(test)]
mod direct_tests {
    use crate::{
        atom::{Atom, AtomCore, EvaluationInfo, Symbol},
        domains::float::Real,
        evaluate::evaluate_multiple_with_prec,
        function, symbol,
    };
    use std::{
        collections::HashMap,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };

    fn counted<T: Real>(calls: Arc<AtomicUsize>) -> impl Fn(&[T]) -> T + Clone {
        move |arguments: &[T]| {
            calls.fetch_add(1, Ordering::SeqCst);
            arguments[0].clone() * &arguments[0]
        }
    }

    fn square(name: &str) -> (Symbol, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let function = symbol!(
            name,
            eval = EvaluationInfo::new().register(counted::<f64>(calls.clone()))
        );
        (function, calls)
    }

    #[test]
    fn direct_vector_shares_native_nested_calls_without_compilation() {
        let (f, calls) = square("direct_vector::nested");
        let x = Atom::var(symbol!("direct_vector::x"));
        let inner = function!(f, &x);
        let outer = function!(f, &inner);
        let expressions = [outer.clone(), inner.clone(), &outer + &inner];
        let point = HashMap::from([(x, 2.)]);
        let individually = expressions
            .iter()
            .map(|expression| expression.evaluate_with_prec(&point, 53).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(calls.load(Ordering::SeqCst), 5);
        calls.store(0, Ordering::SeqCst);
        let result = evaluate_multiple_with_prec(&expressions, &point, 53).unwrap();
        assert_eq!(result.values(), individually);
        assert_eq!(result.values(), &[16., 4., 20.]);
        assert_eq!(calls.load(Ordering::SeqCst), 2);
        let functions = result
            .function_values()
            .map(|(key, value)| (key.to_owned(), *value))
            .collect::<HashMap<_, _>>();
        assert_eq!(functions, HashMap::from([(inner, 4.), (outer, 16.)]));
        assert_eq!(result.into_values(), vec![16., 4., 20.]);
    }

    #[test]
    fn native_conditional_never_executes_or_reports_unselected_hook() {
        let (f, calls) = square("direct_vector::selected");
        let invalid = symbol!(
            "direct_vector::invalid",
            eval = EvaluationInfo::new()
                .register(|_: &[f64]| -> f64 { panic!("unselected branch executed") })
        );
        let x = Atom::var(symbol!("direct_vector::lazy_x"));
        let condition = Atom::var(symbol!("direct_vector::condition"));
        let live = function!(f, &x);
        let dead = function!(invalid, &x);
        let expressions = [
            function!(Symbol::IF, &condition, dead.clone(), live.clone()),
            &live + 1,
        ];
        let point = HashMap::from([(condition, 0.), (x, 2.)]);
        let result = evaluate_multiple_with_prec(&expressions, &point, 53).unwrap();
        assert_eq!(result.values(), &[4., 5.]);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(result.function_values().count(), 1);
        assert!(
            result
                .function_values()
                .all(|(atom, _)| atom == live.as_view())
        );
    }
}
