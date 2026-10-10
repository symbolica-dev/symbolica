use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use symbolica::{
    atom::{Atom, AtomCore, AtomView, EvalFn, EvaluationInfo, Symbol},
    domains::float::{Complex, Real},
    evaluate::OptimizationSettings,
    function, symbol,
};

fn settings() -> OptimizationSettings {
    OptimizationSettings::new()
        .direct_translation(true)
        .horner_iterations(0)
        .cpe_iterations(Some(0))
}

fn counted<T: Real>(calls: Arc<AtomicUsize>) -> impl Fn(&[T]) -> T + Clone {
    move |arguments: &[T]| {
        calls.fetch_add(1, Ordering::SeqCst);
        arguments[0].clone() * &arguments[0]
    }
}

fn square(name: &str) -> (Symbol, Arc<AtomicUsize>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let f = symbol!(
        name,
        eval = EvaluationInfo::new()
            .register(counted::<f64>(calls.clone()))
            .register(counted::<Complex<f64>>(calls.clone()))
    );
    (f, calls)
}

#[test]
fn direct_external_calls_share_across_vector_outputs_without_optimizer() {
    let (f, calls) = square("direct_external_cache::vector_f");
    let x = Atom::var(symbol!("direct_external_cache::vector_x"));
    let value = function!(f, &x);
    let exact = Atom::evaluator_multiple(
        &[value.clone(), &value + 1, 3 * &value],
        std::slice::from_ref(&x),
    )
    .optimization_settings(settings())
    .build()
    .unwrap();
    let mut real = exact.clone().map_coeff(&|c| c.re.to_f64());
    let mut output = [0.; 3];
    real.evaluate(&[2.], &mut output);
    assert_eq!(output, [4., 5., 12.]);
    assert_eq!(calls.swap(0, Ordering::SeqCst), 1);
    real.evaluate(&[3.], &mut output);
    assert_eq!(output, [9., 10., 27.]);
    assert_eq!(calls.swap(0, Ordering::SeqCst), 1);
    let mut complex = exact.map_coeff(&|c| Complex::new(c.re.to_f64(), c.im.to_f64()));
    let mut output = [Complex::new(0., 0.); 3];
    complex.evaluate(&[Complex::new(2., 1.)], &mut output);
    assert_eq!(output[0], Complex::new(3., 4.));
    assert_eq!(output[1], Complex::new(4., 4.));
    assert_eq!(output[2], Complex::new(9., 12.));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn aliases_share_the_callers_external_expression_cache() {
    let (f, calls) = square("direct_external_cache::alias_f");
    let x = Atom::var(symbol!("direct_external_cache::alias_x"));
    let a = Atom::var(symbol!("direct_external_cache::alias_a"));
    let b = Atom::var(symbol!("direct_external_cache::alias_b"));
    let value = function!(f, &x);
    let mut evaluator =
        Atom::evaluator_multiple(&[&a + &b, value.clone()], std::slice::from_ref(&x))
            .optimization_settings(settings())
            .add_aliases([(a, value.clone()), (b, &value + 1)])
            .unwrap()
            .build()
            .unwrap()
            .map_coeff(&|c| c.re.to_f64());
    let mut output = [0.; 2];
    evaluator.evaluate(&[2.], &mut output);
    assert_eq!(output, [9., 4.]);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn tags_and_evaluated_arguments_distinguish_external_calls() {
    let calls = Arc::new(AtomicUsize::new(0));
    let factory_calls = calls.clone();
    let f = symbol!(
        "direct_external_cache::tagged_f",
        eval = EvaluationInfo::new().with_tags(1).register_tagged(
            move |tags: &[AtomView<'_>]| -> EvalFn<f64> {
                let offset = if tags[0] == Atom::one().as_view() {
                    1.
                } else {
                    2.
                };
                let calls = factory_calls.clone();
                Box::new(move |arguments| {
                    calls.fetch_add(1, Ordering::SeqCst);
                    arguments[0] * arguments[0] + offset
                })
            }
        )
    );
    let x = Atom::var(symbol!("direct_external_cache::tagged_x"));
    let a = function!(f, 1, &x);
    let b = function!(f, 2, &x);
    let c = function!(f, 1, &x + 1);
    let mut evaluator =
        Atom::evaluator_multiple(&[a.clone(), b.clone(), c.clone(), a + b + c], &[x])
            .optimization_settings(settings())
            .build()
            .unwrap()
            .map_coeff(&|c| c.re.to_f64());
    let mut output = [0.; 4];
    evaluator.evaluate(&[2.], &mut output);
    assert_eq!(output, [5., 6., 10., 21.]);
    assert_eq!(calls.load(Ordering::SeqCst), 3);
}

#[test]
fn branch_local_cache_never_escapes_or_eagerly_executes_a_dead_branch() {
    let (f, calls) = square("direct_external_cache::branch_f");
    let x = Atom::var(symbol!("direct_external_cache::branch_x"));
    let condition = Atom::var(symbol!("direct_external_cache::branch_condition"));
    let value = function!(f, &x);
    let branch = function!(Symbol::IF, &condition, value.pow(2) + &value, Atom::Zero);
    let mut evaluator = Atom::evaluator_multiple(&[branch, value], &[condition.clone(), x.clone()])
        .optimization_settings(settings())
        .build()
        .unwrap()
        .map_coeff(&|c| c.re.to_f64());
    let mut output = [0.; 2];
    evaluator.evaluate(&[0., 2.], &mut output);
    assert_eq!(output, [0., 4.]);
    assert_eq!(calls.swap(0, Ordering::SeqCst), 1);
    evaluator.evaluate(&[1., 2.], &mut output);
    assert_eq!(output, [20., 4.]);
    // The branch cannot define the unconditional output's slot.
    assert_eq!(calls.swap(0, Ordering::SeqCst), 2);
    let dead = symbol!(
        "direct_external_cache::dead",
        eval = EvaluationInfo::new()
            .register(|_: &[f64]| -> f64 { panic!("unselected external function executed") })
    );
    let conditional = function!(
        Symbol::IF,
        &condition,
        function!(dead, &x),
        function!(f, &x)
    );
    let mut evaluator = conditional
        .evaluator(&[condition, x])
        .optimization_settings(settings())
        .build()
        .unwrap()
        .map_coeff(&|c| c.re.to_f64());
    assert_eq!(evaluator.evaluate_single(&[0., 3.]), 9.);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn distinct_inline_function_scopes_keep_their_argument_bindings() {
    let (f, calls) = square("direct_external_cache::scope_f");
    let g = symbol!("direct_external_cache::scope_g");
    let formal = symbol!("direct_external_cache::scope_formal");
    let x = Atom::var(symbol!("direct_external_cache::scope_x"));
    let y = Atom::var(symbol!("direct_external_cache::scope_y"));
    let value = function!(f, Atom::var(formal));
    let mut evaluator = Atom::evaluator_multiple(&[function!(g, &x), function!(g, &y)], &[x, y])
        .optimization_settings(settings())
        .add_function(g, vec![formal], value.pow(2) + &value)
        .unwrap()
        .build()
        .unwrap()
        .map_coeff(&|c| c.re.to_f64());
    let mut output = [0.; 2];
    evaluator.evaluate(&[2., 3.], &mut output);
    assert_eq!(output, [20., 90.]);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[test]
fn conditional_branches_can_reuse_an_already_evaluated_parent_call() {
    let (f, calls) = square("direct_external_cache::parent_f");
    let x = Atom::var(symbol!("direct_external_cache::parent_x"));
    let condition = Atom::var(symbol!("direct_external_cache::parent_condition"));
    let value = function!(f, &x);
    let conditional = function!(Symbol::IF, &condition, &value, function!(f, &x + 1));
    let mut evaluator = Atom::evaluator_multiple(&[value, conditional], &[condition, x])
        .optimization_settings(settings())
        .build()
        .unwrap()
        .map_coeff(&|c| c.re.to_f64());
    let mut output = [0.; 2];
    evaluator.evaluate(&[1., 2.], &mut output);
    assert_eq!(output, [4., 4.]);
    assert_eq!(calls.swap(0, Ordering::SeqCst), 1);
    evaluator.evaluate(&[0., 2.], &mut output);
    assert_eq!(output, [4., 9.]);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}
