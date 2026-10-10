#![cfg(feature = "native_code_generation")]

use std::{
    collections::BTreeSet,
    sync::{
        Arc, Barrier, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Duration,
};
use symbolica::{
    atom::{Atom, AtomCore, EvaluationInfo},
    domains::float::Complex,
    evaluate::{
        EvaluationDomain, FunctionMap, FunctionRegistrationOptions, InliningPolicy,
        JITCompilationSettings, JITCompiledNumber,
    },
    function, symbol,
};

static NEXT: AtomicUsize = AtomicUsize::new(1);
type Calls = Arc<Mutex<BTreeSet<usize>>>;
struct Workspace {
    id: usize,
    calls: Calls,
}
impl Workspace {
    fn new(calls: Calls) -> Self {
        Self {
            id: NEXT.fetch_add(1, Ordering::Relaxed),
            calls,
        }
    }
    fn call<T: Clone>(&self, args: &[T]) -> T {
        self.calls.lock().unwrap().insert(self.id);
        args[0].clone()
    }
}
impl Clone for Workspace {
    fn clone(&self) -> Self {
        Self::new(self.calls.clone())
    }
}
fn callback<T: Clone>(calls: Calls) -> impl Fn(&[T]) -> T + Clone + Send + Sync {
    let workspace = Workspace::new(calls);
    move |args| workspace.call(args)
}
fn settings(threads: bool) -> JITCompilationSettings {
    JITCompilationSettings::default()
        .optimization_level(2)
        .direct_translation(true)
        .with_option("use_threads", if threads { "true" } else { "false" })
}

fn cloned_domain<T>(value: T)
where
    T: JITCompiledNumber
        + EvaluationDomain
        + Clone
        + Default
        + PartialEq
        + std::fmt::Debug
        + Send
        + Sync
        + 'static,
{
    let calls = Calls::default();
    let f = symbol!(
        &format!("jit_clone_domain_{}", NEXT.fetch_add(1, Ordering::Relaxed)),
        eval = EvaluationInfo::new().register(callback::<T>(calls.clone()))
    );
    let x = Atom::var(symbol!("jit_clone_domain_x"));
    let exact = function!(f, &x).evaluator(&[x]).build().unwrap();
    for threads in [false, true] {
        let original = exact.jit_compile::<T>(settings(threads)).unwrap();
        let mut a = original.clone();
        let mut b = original.clone();
        drop(original);
        calls.lock().unwrap().clear();
        let mut output = [T::default()];
        a.evaluate(std::slice::from_ref(&value), &mut output);
        assert_eq!(output[0], value);
        b.evaluate(std::slice::from_ref(&value), &mut output);
        assert_eq!(output[0], value);
        assert_eq!(calls.lock().unwrap().len(), 2);
        // More than one SIMD-aligned chunk and a non-aligned tail, including
        // the public native packed-domain path for SIMD number types.
        calls.lock().unwrap().clear();
        let input = vec![value.clone(); 1031];
        let mut output = vec![T::default(); input.len()];
        a.batch_evaluate(&input, &mut output, input.len());
        assert!(output.iter().all(|v| *v == value));
        b.batch_evaluate(&input, &mut output, input.len());
        assert!(output.iter().all(|v| *v == value));
        assert_eq!(calls.lock().unwrap().len(), 2);
    }
}

#[test]
fn clone_scalar_real_callbacks() {
    cloned_domain(3.25f64);
}
#[test]
fn clone_scalar_complex_callbacks() {
    cloned_domain(Complex::new(3.25, -0.125));
}
#[test]
fn clone_simd_real_callbacks() {
    cloned_domain(wide::f64x4::from([1., 2., 3., 4.]));
}

#[test]
fn cloned_callbacks_run_on_independent_threads() {
    let calls = Calls::default();
    let callback = callback::<f64>(calls.clone());
    let f = symbol!(
        "jit_clone_parallel_f",
        eval = EvaluationInfo::new().register(move |a: &[f64]| {
            let value = callback(a);
            std::thread::sleep(Duration::from_millis(5));
            value
        })
    );
    let x = Atom::var(symbol!("jit_clone_parallel_x"));
    let original = function!(f, &x)
        .evaluator(&[x])
        .build()
        .unwrap()
        .jit_compile::<f64>(settings(false))
        .unwrap();
    let gate = Arc::new(Barrier::new(4));
    let handles = (0..4)
        .map(|_| {
            let mut evaluator = original.clone();
            let gate = gate.clone();
            std::thread::spawn(move || {
                gate.wait();
                let mut out = [0.];
                evaluator.evaluate(&[5.], &mut out);
                assert_eq!(out, [5.]);
            })
        })
        .collect::<Vec<_>>();
    drop(original);
    for handle in handles {
        handle.join().unwrap();
    }
    assert_eq!(calls.lock().unwrap().len(), 4);
}

#[test]
fn nested_compiled_callback_restores_outer_registry() {
    let calls = Calls::default();
    let leaf = symbol!(
        "jit_nested_leaf",
        eval = EvaluationInfo::new().register(callback::<f64>(calls.clone()))
    );
    let x = Atom::var(symbol!("jit_nested_x"));
    let inner = function!(leaf, &x)
        .evaluator(std::slice::from_ref(&x))
        .build()
        .unwrap()
        .jit_compile::<f64>(settings(false))
        .unwrap();
    let nested = symbol!(
        "jit_nested_compiled",
        eval = EvaluationInfo::new().register(f64::into_external_function(inner))
    );
    let expression = function!(nested, &x) + function!(leaf, &x) + function!(leaf, &x + 1);
    let original = expression
        .evaluator(&[x])
        .build()
        .unwrap()
        .jit_compile::<f64>(settings(false))
        .unwrap();
    let mut a = original.clone();
    let mut b = original.clone();
    drop(original);
    calls.lock().unwrap().clear();
    let mut output = [0.];
    a.evaluate(&[3.], &mut output);
    assert_eq!(output, [10.]);
    b.evaluate(&[3.], &mut output);
    assert_eq!(output, [10.]);
    assert_eq!(calls.lock().unwrap().len(), 4);
}

#[test]
fn non_inlined_body_uses_parent_registry() {
    let calls = Calls::default();
    let leaf = symbol!(
        "jit_body_leaf",
        eval = EvaluationInfo::new().register(callback::<f64>(calls.clone()))
    );
    let body = symbol!("jit_body_function");
    let x_symbol = symbol!("jit_body_x");
    let x = Atom::var(x_symbol);
    let mut functions = FunctionMap::new();
    functions
        .add_function_with_options(
            body,
            vec![x_symbol],
            function!(leaf, &x) + 1,
            FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
        )
        .unwrap();
    let original = function!(body, &x)
        .evaluator(&[x])
        .function_map(functions)
        .build()
        .unwrap()
        .jit_compile::<f64>(settings(true))
        .unwrap();
    let mut a = original.clone();
    let mut b = original.clone();
    drop(original);
    calls.lock().unwrap().clear();
    let mut output = vec![0.; 513];
    a.batch_evaluate(&vec![3.; 513], &mut output, 513);
    assert_eq!(output, vec![4.; 513]);
    b.batch_evaluate(&vec![3.; 513], &mut output, 513);
    assert_eq!(output, vec![4.; 513]);
    assert_eq!(calls.lock().unwrap().len(), 2);
}

#[cfg(feature = "bincode")]
#[test]
fn saved_evaluator_restores_independent_callback_contexts() {
    let calls = Calls::default();
    let f = symbol!(
        "jit_saved_leaf",
        eval = EvaluationInfo::new().register(callback::<f64>(calls.clone()))
    );
    let x = Atom::var(symbol!("jit_saved_x"));
    let original = function!(f, &x)
        .evaluator(&[x])
        .build()
        .unwrap()
        .jit_compile::<f64>(settings(false))
        .unwrap();
    let bytes = bincode::encode_to_vec(&original, bincode::config::standard()).unwrap();
    drop(original);
    let (mut restored, consumed): (symbolica::evaluate::JITCompiledEvaluator<f64>, _) =
        bincode::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
    assert_eq!(consumed, bytes.len());
    let mut cloned = restored.clone();
    calls.lock().unwrap().clear();
    let mut output = [0.];
    restored.evaluate(&[7.], &mut output);
    assert_eq!(output, [7.]);
    cloned.evaluate(&[8.], &mut output);
    assert_eq!(output, [8.]);
    assert_eq!(calls.lock().unwrap().len(), 2);
}
