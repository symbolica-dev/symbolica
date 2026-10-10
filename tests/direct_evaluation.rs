use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
use symbolica::{
    atom::{Atom, AtomCore, AtomView, EvalFn, EvaluationInfo, Symbol},
    domains::float::{
        Complex, DoubleFloat, ErrorPropagatingFloat, Float, FloatLike, Real, RealLike,
    },
    evaluate::{EvaluationDomain, evaluate_multiple_with_prec},
    function, symbol,
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
        eval = EvaluationInfo::new()
            .register(counted::<f64>(calls.clone()))
            .register(counted::<DoubleFloat>(calls.clone()))
            .register(counted::<Float>(calls.clone()))
            .register(counted::<ErrorPropagatingFloat<f64>>(calls.clone()))
            .register(counted::<ErrorPropagatingFloat<Float>>(calls.clone()))
            .register(counted::<Complex<f64>>(calls.clone()))
            .register(counted::<Complex<ErrorPropagatingFloat<f64>>>(
                calls.clone()
            ))
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

fn native_domain<T: Real + EvaluationDomain>(
    root: &Atom,
    x: &Atom,
    value: T,
    bits: u32,
    check: impl Fn(&[T]),
) {
    let expressions = [root.clone(), root * 2];
    let result =
        evaluate_multiple_with_prec(&expressions, &HashMap::from([(x.clone(), value)]), bits)
            .unwrap();
    check(result.values());
    assert_eq!(result.function_values().count(), 1);
}

#[test]
fn all_native_domains_preserve_values_precision_and_tracking() {
    let (f, calls) = square("direct_vector::domains");
    let x = Atom::var(symbol!("direct_vector::domain_x"));
    let root = function!(f, &x);
    native_domain(&root, &x, 2f64, 53, |v| assert_eq!(v, &[4., 8.]));
    native_domain(&root, &x, DoubleFloat::from(2.), 106, |v| {
        assert_eq!(v.iter().map(|v| v.to_f64()).collect::<Vec<_>>(), [4., 8.])
    });
    for bits in [128, 256] {
        native_domain(&root, &x, Float::with_val(bits, 2), bits, |v| {
            assert_eq!(v.iter().map(|v| v.to_f64()).collect::<Vec<_>>(), [4., 8.]);
            assert!(v.iter().all(|v| v.get_precision() == bits));
        });
    }
    native_domain(
        &root,
        &x,
        ErrorPropagatingFloat::new_with_accuracy(2., 10.),
        53,
        |v| assert!(v.iter().all(|v| v.get_absolute_error() > 0.)),
    );
    native_domain(
        &root,
        &x,
        ErrorPropagatingFloat::new_with_accuracy(Float::with_val(192, 2), 10.),
        192,
        |v| assert!(v.iter().all(|v| v.get_absolute_error() > 0.)),
    );
    native_domain(&root, &x, Complex::new(2., 0.), 53, |v| {
        assert_eq!(v, &[Complex::new(4., 0.), Complex::new(8., 0.)])
    });
    native_domain(
        &root,
        &x,
        Complex::new(
            ErrorPropagatingFloat::new_with_accuracy(2., 10.),
            ErrorPropagatingFloat::new_with_accuracy(0., 12.),
        ),
        53,
        |v| {
            assert!(
                v.iter()
                    .all(|v| v.re.get_absolute_error() > 0. && v.im.get_absolute_error() > 0.)
            )
        },
    );
    assert_eq!(calls.load(Ordering::SeqCst), 8);
}

#[test]
fn explicit_function_maps_take_precedence_without_becoming_executed_hooks() {
    let (f, calls) = square("direct_vector::overridden");
    let root = function!(f, Atom::var(symbol!("direct_vector::unbound")));
    let expressions = [root.clone(), &root + 1];
    let result =
        evaluate_multiple_with_prec(&expressions, &HashMap::from([(root, 7.)]), 53).unwrap();
    assert_eq!(result.values(), &[7., 8.]);
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert_eq!(result.function_values().count(), 0);
}

#[test]
fn requests_are_scoped_to_one_point_and_no_partial_success_is_returned() {
    let (f, calls) = square("direct_vector::points");
    let x = Atom::var(symbol!("direct_vector::point_x"));
    let root = function!(f, &x);
    for (input, expected) in [(2., 4.), (3., 9.)] {
        let expressions = [root.clone(), root.clone()];
        let result =
            evaluate_multiple_with_prec(&expressions, &HashMap::from([(x.clone(), input)]), 53)
                .unwrap();
        assert_eq!(result.values(), &[expected, expected]);
    }
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    let expressions = [root, Atom::var(symbol!("direct_vector::missing"))];
    assert!(evaluate_multiple_with_prec(&expressions, &HashMap::from([(x, 2.)]), 53).is_err());
    let empty: [Atom; 0] = [];
    let result = evaluate_multiple_with_prec(&empty, &HashMap::<Atom, f64>::new(), 53).unwrap();
    assert!(result.values().is_empty());
    assert_eq!(result.function_values().count(), 0);
}

#[test]
fn prior_native_cancellations_do_not_execute_removable_singularities() {
    let bad = symbol!(
        "direct_vector::cancelled",
        eval = EvaluationInfo::new()
            .register(|_: &[f64]| -> f64 { panic!("cancelled function executed") })
    );
    let x = Atom::var(symbol!("direct_vector::cancel_x"));
    let root = function!(bad, &x);
    let left = [&root / &x, &root / &x + 2];
    let right = [-&root / &x, -&root / &x + 3];
    let summed = left
        .iter()
        .zip(&right)
        .map(|(a, b)| a + b)
        .collect::<Vec<_>>();
    let result = evaluate_multiple_with_prec(&summed, &HashMap::from([(x, 0.)]), 53).unwrap();
    assert_eq!(result.values(), &[0., 5.]);
    assert_eq!(result.function_values().count(), 0);
}

#[test]
fn constant_and_tagged_hooks_keep_their_full_native_identity() {
    let constants = Arc::new(AtomicUsize::new(0));
    let observed = constants.clone();
    let constant = symbol!(
        "direct_vector::constant",
        eval = EvaluationInfo::constant(move |tags, bits| {
            assert_eq!(tags.len(), 1);
            assert_eq!(i64::try_from(tags[0]).unwrap(), 7);
            observed.fetch_add(1, Ordering::SeqCst);
            Ok(Float::with_val(bits, 5).into())
        })
        .with_tags(1)
    );
    let constant_atom = function!(constant, 7);
    let observed = constants.clone();
    let variable = symbol!(
        "direct_vector::constant_variable",
        eval = EvaluationInfo::constant(move |_, bits| {
            observed.fetch_add(1, Ordering::SeqCst);
            Ok(Float::with_val(bits, 5).into())
        })
    );
    let variable_atom = Atom::var(variable);
    let tagged_calls = Arc::new(AtomicUsize::new(0));
    let observed = tagged_calls.clone();
    let tagged = symbol!(
        "direct_vector::tagged",
        eval = EvaluationInfo::new().with_tags(1).register_tagged(
            move |tags: &[AtomView<'_>]| -> EvalFn<f64> {
                let offset = i64::try_from(tags[0]).unwrap() as f64;
                let observed = observed.clone();
                Box::new(move |arguments| {
                    observed.fetch_add(1, Ordering::SeqCst);
                    offset + arguments[0]
                })
            }
        )
    );
    let x = Atom::var(symbol!("direct_vector::tag_x"));
    let a = function!(tagged, 2, &x);
    let b = function!(tagged, 3, &x);
    let expressions = [
        constant_atom.clone(),
        constant_atom.clone(),
        variable_atom,
        a.clone(),
        b.clone(),
        &a + &b,
    ];
    let result = evaluate_multiple_with_prec(&expressions, &HashMap::from([(x, 1.)]), 53).unwrap();
    assert_eq!(result.values(), &[5., 5., 5., 3., 4., 7.]);
    assert_eq!(constants.load(Ordering::SeqCst), 2); // Variable and function are distinct.
    assert_eq!(tagged_calls.load(Ordering::SeqCst), 2);
    let executed = result
        .function_values()
        .map(|(atom, value)| (atom.to_owned(), *value))
        .collect::<HashMap<_, _>>();
    assert_eq!(
        executed,
        HashMap::from([(constant_atom, 5.), (a, 3.), (b, 4.)])
    );
}

#[test]
fn builtins_are_not_reported_and_nonfinite_semantics_remain_scalar() {
    let (f, calls) = square("direct_vector::builtin_argument");
    let x = Atom::var(symbol!("direct_vector::builtin_x"));
    let root = function!(f, &x);
    let expressions = [function!(Symbol::LOG, &root), root.clone()];
    let result = evaluate_multiple_with_prec(&expressions, &HashMap::from([(x, 2.)]), 53).unwrap();
    assert_eq!(result.values(), &[4f64.ln(), 4.]);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(result.function_values().count(), 1);
    let nonfinite = symbol!(
        "direct_vector::nonfinite",
        eval = EvaluationInfo::new().register(|_: &[f64]| f64::NAN)
    );
    let expressions = [function!(nonfinite, 0)];
    let result =
        evaluate_multiple_with_prec(&expressions, &HashMap::<Atom, f64>::new(), 53).unwrap();
    assert!(result.values()[0].is_nan());
    assert!(result.function_values().next().unwrap().1.is_nan());
}
