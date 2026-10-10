//! The generated machine code owns only a registry identity and function index.
//! Every invocation borrows the numerical callbacks of the evaluator being run.
//! This keeps machine code shared while honoring callback Clone implementations.
use std::{cell::Cell, sync::Arc};

use rayon::prelude::*;
use symjit::Compiled;

use super::{Complex, ExternalFunctionContainer};

/// Opaque identity shared by one compiled application and its evaluator clones.
/// Used only by the internal JIT callback conversion hook.
#[doc(hidden)]
#[derive(Clone, Default)]
pub struct JITCallbackContext(Arc<()>);

// Constants and instruction bodies do not invoke callback implementations.
// Intrinsics with no user evaluation metadata also need no cloned workspace.
// Nested bodies share the complete topological registry, so any user callback
// they call is represented separately in this same table.
pub(super) fn context_for<T>(
    functions: &[ExternalFunctionContainer<T>],
) -> Option<JITCallbackContext> {
    functions
        .iter()
        .any(|f| {
            f.constant_index.is_none()
                && f.body.is_none()
                && (!f.symbol.is_builtin() || f.symbol.get_evaluation_info().is_some())
        })
        .then(JITCallbackContext::default)
}

struct Frame<T> {
    registry: *const (),
    functions: *const [ExternalFunctionContainer<T>],
    previous: *const Frame<T>,
}

macro_rules! domain {
    ($module:ident, $number:ty) => {
        pub(super) mod $module {
            use super::*;

            thread_local! {
                static CURRENT: Cell<*const Frame<$number>> = const { Cell::new(std::ptr::null()) };
            }

            pub fn with<R>(
                context: Option<&JITCallbackContext>,
                functions: &[ExternalFunctionContainer<$number>],
                run: impl FnOnce() -> R,
            ) -> R {
                let Some(context) = context else { return run() };
                // Frame and callbacks outlive the synchronous call. The only
                // exported pointer is thread-local and restored before return,
                // including unwinding. Recursive calls push stack-local frames;
                // no growing TLS allocation or callback lock is needed.
                let frame = Frame {
                    registry: Arc::as_ptr(&context.0),
                    functions,
                    previous: CURRENT.with(Cell::get),
                };
                struct Restore(*const Frame<$number>);
                impl Drop for Restore {
                    fn drop(&mut self) {
                        CURRENT.with(|current| current.set(self.0));
                    }
                }
                CURRENT.with(|current| current.set(&frame));
                let _restore = Restore(frame.previous);
                run()
            }

            pub fn call(context: &JITCallbackContext, index: usize, args: &[$number]) -> $number {
                // End the TLS access before invoking user code, which may
                // recursively invoke another evaluator of this numerical type.
                let mut frame = CURRENT.with(Cell::get);
                while !frame.is_null() {
                    // SAFETY: with() keeps every linked frame and its borrowed
                    // callback slice alive on this thread until the call ends.
                    let active = unsafe { &*frame };
                    if active.registry == Arc::as_ptr(&context.0) {
                        let functions = unsafe { &*active.functions };
                        return functions[index].imp.as_ref().unwrap()(args);
                    }
                    frame = active.previous;
                }
                panic!("JIT callback invoked outside its evaluator context");
            }

            pub fn batch<E: symjit::Element + Send + Sync>(
                code: &symjit::Applet,
                context: Option<&JITCallbackContext>,
                functions: &[ExternalFunctionContainer<$number>],
                args: &[E],
                out: &mut [E],
                rows: usize,
            ) {
                if context.is_none() {
                    code.evaluate_matrix(args, out, rows);
                    return;
                }
                if !code.use_threads || rows <= 1 {
                    with(context, functions, || code.evaluate_matrix(args, out, rows));
                    return;
                }
                // SymJIT's internal Rayon workers cannot inherit caller TLS.
                // Use the same caller-selected Rayon pool, scoping each chunk
                // and retaining SymJIT's own SIMD packing and scalar tail code.
                let mut local = code.clone();
                local.use_threads = false;
                if out.is_empty() {
                    with(context, functions, || {
                        local.evaluate_matrix(args, out, rows)
                    });
                    return;
                }
                let lanes = code.compiled_simd.as_ref().map_or(1, |c| c.count_lanes());
                let chunk_rows = 64 * lanes;
                let inputs = args.len() / rows;
                let outputs = out.len() / rows;
                assert_eq!(args.len(), rows * inputs);
                assert_eq!(out.len(), rows * outputs);
                out.par_chunks_mut(chunk_rows * outputs)
                    .enumerate()
                    .for_each(|(chunk, output)| {
                        let first = chunk * chunk_rows;
                        let count = output.len() / outputs;
                        let input = &args[first * inputs..(first + count) * inputs];
                        with(context, functions, || {
                            local.evaluate_matrix(input, output, count)
                        });
                    });
            }
            #[cfg(test)]
            mod tests {
                use super::*;

                #[test]
                fn context_restores_nested_frames_and_unwinds() {
                    let a = JITCallbackContext::default();
                    let b = JITCallbackContext::default();
                    assert!(CURRENT.with(Cell::get).is_null());
                    with(Some(&a), &[], || {
                        let original = CURRENT.with(Cell::get);
                        with(Some(&b), &[], || {
                            let nested = CURRENT.with(Cell::get);
                            assert_ne!(nested, original);
                            with(Some(&a), &[], || {
                                assert_ne!(CURRENT.with(Cell::get), nested)
                            });
                            assert_eq!(CURRENT.with(Cell::get), nested);
                        });
                        assert_eq!(CURRENT.with(Cell::get), original);
                        let failure = std::panic::catch_unwind(|| {
                            with(Some(&b), &[], || panic!("scope unwind control"));
                        });
                        assert!(failure.is_err());
                        assert_eq!(CURRENT.with(Cell::get), original);
                    });
                    assert!(CURRENT.with(Cell::get).is_null());
                }
            }
        }
    };
}

domain!(real, f64);
domain!(real_simd, wide::f64x4);
domain!(complex, Complex<f64>);
domain!(complex_simd, Complex<wide::f64x4>);
