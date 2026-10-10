use super::*;
use std::{
    collections::BTreeSet,
    sync::atomic::{AtomicUsize, Ordering},
};

static NEXT: AtomicUsize = AtomicUsize::new(1);
struct Workspace {
    id: usize,
    calls: Arc<Mutex<BTreeSet<usize>>>,
}
impl Clone for Workspace {
    fn clone(&self) -> Self {
        Self {
            id: NEXT.fetch_add(1, Ordering::Relaxed),
            calls: self.calls.clone(),
        }
    }
}
impl Workspace {
    fn call<T: Clone>(&self, args: &[T]) -> T {
        self.calls.lock().unwrap().insert(self.id);
        args[0].clone()
    }
}

#[test]
fn jit_context_complex_simd_callback_clones() {
    type T = Complex<wide::f64x4>;
    let calls = Arc::new(Mutex::new(BTreeSet::new()));
    let workspace = Workspace {
        id: NEXT.fetch_add(1, Ordering::Relaxed),
        calls: calls.clone(),
    };
    // The low-level JIT supports this domain even though the high-level exact
    // evaluator currently has no EvaluationDomain implementation for it.
    let symbol = crate::symbol!("jit_callback_tests::complex_simd_lowlevel");
    let mut external = ExternalFunctionContainer::new(symbol, vec![], vec![]);
    external.imp = Some(Box::new(move |args: &[T]| workspace.call(args)));
    let instructions = vec![Instruction::Fun(
        Slot::Out(0),
        Box::new((symbol, vec![], vec![Slot::Param(0)])),
        false,
    )];
    let original = T::jit_compile(
        instructions,
        vec![],
        1,
        &[external],
        JITCompilationSettings::default()
            .optimization_level(2)
            .direct_translation(true),
    )
    .unwrap();
    let mut first = original.clone();
    let mut second = original.clone();
    drop(original);
    calls.lock().unwrap().clear();
    let value = Complex::new(
        wide::f64x4::from([1., 2., 3., 4.]),
        wide::f64x4::from([0.5, 0.25, 0.125, 0.0625]),
    );
    let input = vec![value; 1031];
    let mut output = vec![T::default(); input.len()];
    first.batch_evaluate(&input, &mut output, input.len());
    assert_eq!(output, input);
    second.batch_evaluate(&input, &mut output, input.len());
    assert_eq!(output, input);
    assert_eq!(calls.lock().unwrap().len(), 2);
}

#[test]
fn jit_context_standalone_converter_needs_no_scope() {
    let f = crate::symbol!(
        "jit_callback_standalone",
        eval = EvaluationInfo::new().register(|args: &[f64]| args[0] + 2.)
    );
    let x = Atom::var(crate::symbol!("jit_callback_standalone_x"));
    let expression = crate::function!(f, &x)
        .evaluator(&[x])
        .build()
        .unwrap()
        .map_coeff(&|c| c.re.to_f64());
    let exported = expression.export_instructions_impl(&expression.external_fns, &[]);
    let constants = exported
        .constants
        .iter()
        .map(|v| v.to_complex_f64().unwrap())
        .collect();
    let settings = JITCompilationSettings::default();
    let mut config = Config::default();
    config
        .set_defuns(f64::convert_external_functions(&expression.external_fns, &settings).unwrap());
    let mut translator = translate_to_symjit(exported.instructions, constants, 1, config).unwrap();
    let applet = translator.compile().unwrap().seal().unwrap();
    let mut output = [0.];
    applet.evaluate(&[3.], &mut output);
    assert_eq!(output, [5.]);
}

#[test]
fn jit_context_intrinsics_need_no_callback_scope() {
    let x = crate::parse!("jit_callback_intrinsic_x");
    let expression = x.clone().sin() + x.clone().cos();
    let mut evaluator = expression
        .evaluator(&[x])
        .build()
        .unwrap()
        .jit_compile::<f64>(JITCompilationSettings::default())
        .unwrap();
    assert!(evaluator.callback_context.is_none());
    let mut output = [0.];
    evaluator.evaluate(&[0.], &mut output);
    assert_eq!(output, [1.]);
}
