//! Evaluation of expressions.
//!
//! The main entry point is through [AtomCore::evaluator].
use ahash::{AHasher, HashMap, HashMapExt, HashSet};
use dyn_clone::DynClone;
use rand::Rng;
#[cfg(feature = "native_code_generation")]
use self_cell::self_cell;
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, hash_map::Entry},
    hash::{Hash, Hasher},
    panic,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};
#[cfg(feature = "native_code_generation")]
use std::{
    os::raw::{c_ulong, c_void},
    path::{Path, PathBuf},
};
#[cfg(feature = "native_code_generation")]
use symjit::{Applet, Composer, Config, Defuns, Storage, Translator};

#[cfg(feature = "native_code_generation")]
mod backend;
mod domain;
mod dual;
mod evaluator;
#[cfg(feature = "native_code_generation")]
mod export;
mod external;
mod function_map;
mod instruction;
mod optimize;
mod tree;

#[cfg(feature = "native_code_generation")]
pub use backend::*;
pub use domain::*;
pub use dual::*;
pub use evaluator::*;
#[cfg(feature = "native_code_generation")]
pub use export::*;
pub use external::*;
pub use function_map::*;
pub use instruction::{
    ComplexPhase, ExportedInstructions, ExportedSubEvaluator, Instruction, InstructionList, Label,
    Slot, VectorInstruction,
};
pub use optimize::*;
pub use tree::*;

use function_map::Expr;
use instruction::Instr;

use crate::{
    LicenseManager, OperationCount,
    atom::{Atom, AtomCore, AtomView, EvaluationInfo, Indeterminate, KeyLookup, Symbol},
    coefficient::CoefficientView,
    combinatorics::unique_permutations,
    domains::{
        InternalOrdering,
        dual::DualNumberStructure,
        float::{
            Complex, Constructible, DoubleFloat, ErrorPropagatingFloat, F64, Float, FloatLike,
            Real, RealLike, SingleFloat,
        },
        integer::Integer,
        rational::Rational,
    },
    error, get_symbol,
    id::ConditionResult,
    info,
    numerical_integration::MonteCarloRng,
    state::State,
    utils::AbortCheck,
};

#[cfg(test)]
mod test {
    use ahash::HashMap;
    use numerica::domains::{dual::HyperDual, float::Real};

    use crate::{
        atom::{Atom, AtomCore, EvaluationInfo},
        create_hyperdual_from_components,
        domains::{
            float::{Complex, Float},
            rational::Rational,
        },
        evaluate::{
            CompileOptions, Dualizer, EvaluationError, ExportSettings, ExportedInstructions,
            FunctionMap, InlineASM, Instruction, JITCompilationSettings, OptimizationSettings,
            Slot,
        },
        id::ConditionResult,
        parse, symbol,
    };

    #[test]
    fn function_map_inconsistent_tag_count_returns_evaluation_error() {
        let mut fn_map = FunctionMap::new();
        let f = symbol!("symbolica::test::tag_count_mismatch");

        fn_map
            .add_tagged_function(f, vec![Atom::num(1)], vec![symbol!("x")], parse!("x"))
            .unwrap();

        assert_eq!(
            fn_map.add_tagged_function(
                f,
                vec![Atom::num(1), Atom::num(2)],
                vec![symbol!("x")],
                parse!("x"),
            ),
            Err(EvaluationError::InconsistentFunctionTagCount {
                function: f,
                expected: 1,
                actual: 2,
            })
        );
    }

    #[test]
    fn eval_fun() {
        let _ = symbol!(
            "symbolica::eval_fun::e",
            eval = EvaluationInfo::constant(|_tags, prec| { Ok(Float::new(prec).e().into()) })
        );

        let _ = symbol!(
            "symbolica::eval_fun::atanh",
            eval = EvaluationInfo::new()
                .register(|args: &[Complex<Float>]| args[0].atanh())
                .register(|args: &[f64]| args[0].atanh())
        );

        let a = parse!("symbolica::eval_fun::e*symbolica::eval_fun::atanh(x)");

        assert!(
            (parse!("symbolica::eval_fun::e*symbolica::eval_fun::atanh(0.1`32)").to_float(32)
                - parse!("2.7273975248950224505081204947890e-1`32"))
            .abs()
                < parse!("1e-30`32")
        );

        let r = a.evaluator(&[parse!("x")]).build().unwrap();

        let mut r_f64 = r.clone().map_coeff(&|x| x.re.to_f64());

        let mut res = [0.];
        r_f64.evaluate(&[0.1], &mut res);
        assert_eq!(res[0], 0.2727397524895022);

        let mut jit_compiled = r_f64
            .jit_compile(JITCompilationSettings::default())
            .unwrap();

        jit_compiled.evaluate(&[0.1], &mut res);
        assert_eq!(res[0], 0.2727397524895022);

        let mut r_wide = r_f64.map_coeff(&|x| (*x).into());

        let mut res = [wide::f64x4::new([0., 0., 0., 0.])];
        r_wide.evaluate(&[wide::f64x4::new([0.1, 0.2, 0.3, 0.4])], &mut res);
        assert_eq!(
            res[0].to_array(),
            [
                0.2727397524895022,
                0.5510842177223028,
                0.8413615156571546,
                1.1515971885913823
            ]
        );
    }

    #[test]
    fn evaluate() {
        let a = parse!("v1*cos(v1) + f1(1)^2");

        let mut const_map = HashMap::default();
        const_map.insert(parse!("v1"), 6.);
        const_map.insert(parse!("f1(1)"), 7.);

        let r = a.evaluate(&const_map).unwrap();
        assert_eq!(r, 54.761021719902196);
    }

    #[test]
    fn arb_prec() {
        let x = symbol!("v1");
        let a = parse!("128731/12893721893721 + v1");

        let mut const_map = HashMap::default();

        let v = Atom::var(x);
        const_map.insert(v.as_view(), Float::with_val(200, 6));

        let r = a.evaluate_with_prec(&const_map, 200).unwrap();

        assert_eq!(
            format!("{r}"),
            "6.00000000998400625211945786243908951675582851493871969158108"
        );
    }

    #[test]
    fn nested() {
        let e1 = parse!("x + pi + cos(x) + f(g(x+1),h(x*2)) + p(1,x)");
        let e2 = parse!("x + h(x*2) + cos(x)");
        let f = parse!("y^2 + z^2*y^2");
        let g = parse!("i(y+7)+x*i(y+7)*(y-1)");
        let h = parse!("y*(1+x*(1+x^2)) + y^2*(1+x*(1+x^2))^2 + 3*(1+x^2)");
        let i = parse!("y - 1");
        let p1 = parse!("3*z^3 + 4*z^2 + 6*z +8");

        let mut fn_map = FunctionMap::new();

        fn_map
            .add_tagged_function(symbol!("p"), vec![Atom::num(1)], vec![symbol!("z")], p1)
            .unwrap();
        fn_map
            .add_function(symbol!("f"), vec![symbol!("y"), symbol!("z")], f)
            .unwrap();
        fn_map
            .add_function(symbol!("g"), vec![symbol!("y")], g)
            .unwrap();
        fn_map
            .add_function(symbol!("h"), vec![symbol!("y")], h)
            .unwrap();
        fn_map
            .add_function(symbol!("i"), vec![symbol!("y")], i)
            .unwrap();

        let params = vec![parse!("x")];

        let evaluator = Atom::evaluator_multiple(&[e1, e2], &params)
            .function_map(fn_map)
            .optimization_settings(OptimizationSettings::default())
            .build()
            .unwrap();

        let mut e_f64 = evaluator.map_coeff(&|x| x.clone().to_real().unwrap().into());
        let mut res = [0., 0.];
        e_f64.evaluate(&[1.1], &mut res);
        assert!((res[0] - 1622709.2241624785).abs() / 1622709.2241624785 < 1e-10);
    }

    #[test]
    fn non_inlined_function_uses_sub_evaluator() {
        fn execute_exported(
            exported: &ExportedInstructions<f64>,
            params: &[f64],
            output_count: usize,
        ) -> Vec<f64> {
            fn read(
                slot: Slot,
                params: &[f64],
                constants: &[f64],
                temporaries: &[f64],
                outputs: &[f64],
            ) -> f64 {
                match slot {
                    Slot::Param(i) => params[i],
                    Slot::Const(i) => constants[i],
                    Slot::Temp(i) => temporaries[i],
                    Slot::Out(i) => outputs[i],
                }
            }

            fn write(slot: Slot, value: f64, temporaries: &mut [f64], outputs: &mut [f64]) {
                match slot {
                    Slot::Temp(i) => temporaries[i] = value,
                    Slot::Out(i) => outputs[i] = value,
                    Slot::Param(_) | Slot::Const(_) => panic!("cannot write to {slot}"),
                }
            }

            let mut temporaries = vec![0.; exported.temporary_count];
            let mut outputs = vec![0.; output_count];
            for instruction in &exported.instructions {
                let value = |slot| read(slot, params, &exported.constants, &temporaries, &outputs);
                match instruction {
                    Instruction::Add(out, args, _) => {
                        let result = args.iter().map(|arg| value(*arg)).sum();
                        write(*out, result, &mut temporaries, &mut outputs);
                    }
                    Instruction::Mul(out, args, _) => {
                        let result = args.iter().map(|arg| value(*arg)).product();
                        write(*out, result, &mut temporaries, &mut outputs);
                    }
                    Instruction::Pow(out, base, exponent, _) => {
                        let result = value(*base).powi((*exponent).try_into().unwrap());
                        write(*out, result, &mut temporaries, &mut outputs);
                    }
                    Instruction::Powf(out, base, exponent, _) => {
                        let result = value(*base).powf(value(*exponent));
                        write(*out, result, &mut temporaries, &mut outputs);
                    }
                    Instruction::Fun(out, function, _) => {
                        let (symbol, tags, args) = &**function;
                        let sub_evaluator = exported
                            .sub_evaluators
                            .iter()
                            .find(|sub| sub.symbol == *symbol && sub.tags == *tags)
                            .expect("test expression should only call exported sub-evaluators");
                        let arguments = args.iter().map(|arg| value(*arg)).collect::<Vec<_>>();
                        assert_eq!(arguments.len(), sub_evaluator.input_count);
                        let result = execute_exported(
                            &sub_evaluator.instructions,
                            &arguments,
                            sub_evaluator.output_count,
                        );
                        assert_eq!(result.len(), 1);
                        write(*out, result[0], &mut temporaries, &mut outputs);
                    }
                    Instruction::Assign(out, input) => {
                        let result = value(*input);
                        write(*out, result, &mut temporaries, &mut outputs);
                    }
                    Instruction::IfElse(_, _)
                    | Instruction::Goto(_)
                    | Instruction::Label(_)
                    | Instruction::Join(_, _, _, _) => {
                        panic!("control flow is not used by this test")
                    }
                }
            }
            outputs
        }

        let mut fn_map = FunctionMap::new();
        fn_map
            .add_function_no_inline(
                symbol!("symbolica::sub_eval::helper"),
                vec![symbol!("z")],
                parse!("z^2 + 2"),
            )
            .unwrap();
        fn_map
            .add_function_no_inline(
                symbol!("symbolica::sub_eval::large"),
                vec![symbol!("y")],
                parse!("symbolica::sub_eval::helper(y) + x + 5"),
            )
            .unwrap();
        fn_map
            .add_tagged_function_no_inline(
                symbol!("symbolica::sub_eval::tagged"),
                vec![Atom::num(1)],
                vec![symbol!("y")],
                parse!("3*y + 7"),
            )
            .unwrap();
        fn_map
            .add_function_no_inline(
                symbol!("symbolica::sub_eval::captured"),
                Vec::<crate::atom::Symbol>::new(),
                parse!("q + 1"),
            )
            .unwrap();
        fn_map
            .add_function(
                symbol!("symbolica::sub_eval::outer"),
                vec![symbol!("q")],
                parse!("symbolica::sub_eval::captured()"),
            )
            .unwrap();

        let expressions = [
            parse!("symbolica::sub_eval::large(x)"),
            parse!("symbolica::sub_eval::large(x + 1)"),
            parse!("symbolica::sub_eval::tagged(1, x)"),
            parse!("symbolica::sub_eval::outer(x)"),
        ];
        let evaluator = Atom::evaluator_multiple(&expressions, &[parse!("x")])
            .function_map(fn_map)
            // The legacy tree optimizer also falls back to direct translation when a function is
            // explicitly kept out of line.
            .direct_translation(false)
            .build()
            .unwrap();

        assert_eq!(evaluator.count_operations().function_calls, 4);
        assert_eq!(evaluator.external_fns.len(), 3);
        assert!(
            evaluator
                .external_fns
                .iter()
                .all(|external| external.sub_evaluator.is_some())
        );

        let exported = evaluator
            .clone()
            .map_coeff(&|c| c.re.to_f64())
            .export_instructions();
        assert_eq!(exported.input_count, 1);
        assert_eq!(exported.output_count, 4);
        assert_eq!(exported.sub_evaluators.len(), 3);
        assert_eq!(
            execute_exported(&exported, &[2.], exported.output_count),
            [13., 18., 13., 3.]
        );

        let large = exported
            .sub_evaluators
            .iter()
            .find(|sub| sub.symbol == symbol!("symbolica::sub_eval::large"))
            .unwrap();
        // Captured values, explicit arguments, and hoisted coefficients are all passed through the
        // function instruction in exactly the order described by `input_count`.
        assert_eq!(large.input_count, 4);
        assert_eq!(large.output_count, 1);
        assert_eq!(large.instructions.input_count, large.input_count);
        assert_eq!(large.instructions.output_count, large.output_count);
        assert!(large.instructions.constants.is_empty());
        assert_eq!(large.instructions.sub_evaluators.len(), 1);
        let helper = &large.instructions.sub_evaluators[0];
        assert_eq!(helper.symbol, symbol!("symbolica::sub_eval::helper"));
        let helper_argument_count = large
            .instructions
            .instructions
            .iter()
            .find_map(|instruction| match instruction {
                Instruction::Fun(_, function, _)
                    if function.0 == helper.symbol && function.1 == helper.tags =>
                {
                    Some(function.2.len())
                }
                _ => None,
            })
            .unwrap();
        assert_eq!(helper.input_count, helper_argument_count);
        assert_eq!(helper.instructions.input_count, helper.input_count);
        assert_eq!(helper.instructions.output_count, helper.output_count);
        assert!(helper.instructions.sub_evaluators.is_empty());

        let rational_constants = evaluator.get_constants().to_vec();
        let mut evaluator_f64 = evaluator.clone().map_coeff(&|c| c.re.to_f64());
        let mut out = [0.; 4];
        evaluator_f64.evaluate(&[2.], &mut out);
        assert_eq!(out, [13., 18., 13., 3.]);

        let mut remapped = evaluator_f64
            .set_coeff(&rational_constants)
            .map_coeff(&|c| c.re.to_f64());
        remapped.evaluate(&[2.], &mut out);
        assert_eq!(out, [13., 18., 13., 3.]);

        #[cfg(feature = "bincode")]
        {
            let bytes = bincode::encode_to_vec(&evaluator, bincode::config::standard()).unwrap();
            let (decoded, _) = bincode::decode_from_slice::<
                crate::evaluate::ExpressionEvaluator<Complex<Rational>>,
                _,
            >(&bytes, bincode::config::standard())
            .unwrap();
            let mut decoded = decoded.map_coeff(&|c| c.re.to_f64());
            decoded.evaluate(&[2.], &mut out);
            assert_eq!(out, [13., 18., 13., 3.]);
        }

        #[cfg(feature = "native_code_generation")]
        {
            let base = std::env::temp_dir()
                .join(format!("symbolica_sub_evaluator_{}", std::process::id()));
            let source = base.with_extension("cpp");
            let library = base.with_extension("so");
            let mut compiled = evaluator
                .map_coeff(&|c| c.re.to_f64())
                .export_cpp::<f64>(&source, "sub_evaluator", ExportSettings::default())
                .unwrap()
                .compile(&library, CompileOptions::default().compiler("c++"))
                .unwrap()
                .load()
                .unwrap();

            compiled.evaluate(&[2.], &mut out);
            assert_eq!(out, [13., 18., 13., 3.]);

            let _ = std::fs::remove_file(source);
            let _ = std::fs::remove_file(library);
        }
    }

    #[cfg(feature = "native_code_generation")]
    #[test]
    fn jit_compiles_nested_sub_evaluators() {
        let evaluator = parse!("symbolica::sub_eval::jit_f(x) + 1")
            .evaluator(&[parse!("x")])
            .add_function_no_inline(
                symbol!("symbolica::sub_eval::jit_g"),
                vec![symbol!("z")],
                parse!("z^2 + 2"),
            )
            .unwrap()
            .add_function_no_inline(
                symbol!("symbolica::sub_eval::jit_f"),
                vec![symbol!("y")],
                parse!("symbolica::sub_eval::jit_g(y)*y + 5"),
            )
            .unwrap()
            .build()
            .unwrap();

        assert!(
            evaluator
                .external_fns
                .iter()
                .all(|external| { external.sub_evaluator.is_some() && external.imp.is_none() })
        );

        let mut compiled = evaluator
            .jit_compile::<f64>(JITCompilationSettings::default())
            .unwrap();
        let mut out = [0.];
        compiled.evaluate(&[3.], &mut out);
        assert_eq!(out, [39.]);

        #[cfg(feature = "bincode")]
        {
            let bytes = bincode::encode_to_vec(&compiled, bincode::config::standard()).unwrap();
            let (mut decoded, _) = bincode::decode_from_slice::<
                crate::evaluate::JITCompiledEvaluator<f64>,
                _,
            >(&bytes, bincode::config::standard())
            .unwrap();
            decoded.evaluate(&[4.], &mut out);
            assert_eq!(out, [78.]);
        }

        let mut compiled = evaluator
            .jit_compile::<Complex<f64>>(JITCompilationSettings::default())
            .unwrap();
        let mut out = [Complex::new(0., 0.)];
        compiled.evaluate(&[Complex::new(3., 1.)], &mut out);
        assert_eq!(out, [Complex::new(30., 28.)]);
    }

    #[cfg(feature = "native_code_generation")]
    #[test]
    fn cpp_asm_export_includes_nested_sub_evaluators() {
        fn assert_asm_wrapper(source: &str, name: &str, number_type: &str) {
            let signature = format!("__attribute__((noinline)) {number_type} {name}(");
            let start = source
                .find(&signature)
                .unwrap_or_else(|| panic!("missing sub-evaluator wrapper '{signature}'"));
            let body = &source[start..];
            let end = body
                .find("\n}\n")
                .expect("sub-evaluator wrapper should have a function body");
            assert!(
                body[..end].contains("__asm__("),
                "sub-evaluator '{name}' should contain inline ASM"
            );
        }

        let f = symbol!("non_inlined_functions::f");
        let g = symbol!("non_inlined_functions::g");
        let evaluator = parse!("non_inlined_functions::f(x)")
            .evaluator(&[parse!("x")])
            .add_function_no_inline(g, vec![symbol!("z")], parse!("z*z + 2"))
            .unwrap()
            .add_function_no_inline(
                f,
                vec![symbol!("y")],
                parse!("non_inlined_functions::g(y) + y*y + 1"),
            )
            .unwrap()
            .build()
            .unwrap();

        let (f_index, f_external) = evaluator
            .external_fns
            .iter()
            .enumerate()
            .find(|(_, external)| external.symbol == f)
            .unwrap();
        let f_name = evaluator.external_cpp_name(f_index);
        let f_evaluator = f_external.sub_evaluator.as_ref().unwrap();
        let (g_index, _) = f_evaluator
            .external_fns
            .iter()
            .enumerate()
            .find(|(_, external)| external.symbol == g)
            .unwrap();
        let g_name = f_evaluator.external_cpp_name(g_index);
        assert_eq!(f_name, "f_0");
        assert_eq!(g_name, "g_0");

        let settings = ExportSettings::new().inline_asm(InlineASM::X64);
        let real_source = evaluator
            .export_cpp_str::<f64>("asm_sub_evaluator", settings.clone())
            .unwrap();
        assert_asm_wrapper(&real_source, &f_name, "double");
        assert_asm_wrapper(&real_source, &g_name, "double");

        let complex_source = evaluator
            .export_cpp_str::<Complex<f64>>("asm_sub_evaluator", settings.clone())
            .unwrap();
        assert_asm_wrapper(&complex_source, &f_name, "std::complex<double>");
        assert_asm_wrapper(&complex_source, &g_name, "std::complex<double>");

        let simd_source = evaluator
            .export_cpp_str::<wide::f64x4>("asm_sub_evaluator", settings.clone())
            .unwrap();
        assert_asm_wrapper(&simd_source, &f_name, "simd");
        assert_asm_wrapper(&simd_source, &g_name, "simd");

        let complex_simd_source = evaluator
            .export_cpp_str::<Complex<wide::f64x4>>("asm_sub_evaluator", settings)
            .unwrap();
        assert_asm_wrapper(&complex_simd_source, &f_name, "simd");
        assert_asm_wrapper(&complex_simd_source, &g_name, "simd");
    }

    #[test]
    fn exported_sub_evaluator_preserves_external_calls() {
        let external = symbol!(
            "symbolica::sub_eval::exported_external",
            eval = EvaluationInfo::new().register(|args: &[f64]| args[0] + 1.)
        );
        let wrapper = symbol!("symbolica::sub_eval::external_wrapper");
        let evaluator = parse!("symbolica::sub_eval::external_wrapper(x)")
            .evaluator(&[parse!("x")])
            .add_function_no_inline(
                wrapper,
                vec![symbol!("y")],
                parse!("symbolica::sub_eval::exported_external(y)"),
            )
            .unwrap()
            .build()
            .unwrap();

        let exported = evaluator.export_instructions();
        assert_eq!(exported.sub_evaluators.len(), 1);
        let wrapper = &exported.sub_evaluators[0].instructions;
        assert!(wrapper.sub_evaluators.is_empty());
        let (symbol, tags, _) = wrapper
            .instructions
            .iter()
            .find_map(|instruction| match instruction {
                Instruction::Fun(_, function, _) => Some(&**function),
                _ => None,
            })
            .unwrap();
        assert_eq!(*symbol, external);
        assert!(tags.is_empty());
    }

    #[test]
    fn zero_test() {
        let e = parse!(
            "(sin(v1)^2-sin(v1))(sin(v1)^2+sin(v1))^2 - (1/4 sin(2v1)^2-1/2 sin(2v1)cos(v1)-2 cos(v1)^2+1/2 sin(2v1)cos(v1)^3+3 cos(v1)^4-cos(v1)^6)"
        );
        assert_eq!(e.zero_test(10, f64::EPSILON), ConditionResult::Inconclusive);

        let e = parse!("x + (1+x)^2 + (x+2)*5");
        assert_eq!(e.zero_test(10, f64::EPSILON), ConditionResult::False);
    }

    #[test]
    fn branching() {
        let tests = vec![
            ("if(y, x*x + z*z + x*z*z, x * x + 3)", 25., 12.),
            ("if(y+1, x*x + z*z + x*z*z, x * x + 3)", 12., 25.),
            ("if(y, x*x + z*z + x*z*z, 3)", 25., 3.),
            ("if(x + z, if(y, 1 + x, 1+x+y), 0)", 4., 4.),
            ("if(y, x * z, 0) + x * z", 12., 6.),
            ("if(y, x + 1, 2)*if(y+1, x + 1, 3)", 12., 8.),
            ("if(y, if(z, x + 1, 3)*if(z-2, x + 1, 4), 2)", 16., 2.),
        ];

        for (input, true_res, false_res) in tests {
            let mut eval = parse!(input)
                .evaluator(&vec![
                    crate::parse!("x"),
                    crate::parse!("y"),
                    crate::parse!("z"),
                ])
                .build()
                .unwrap()
                .map_coeff(&|x| x.re.to_f64());

            let res = eval.evaluate_single(&[3., -1., 2.]);
            assert_eq!(res, true_res);
            let res = eval.evaluate_single(&[3., 0., 2.]);
            assert_eq!(res, false_res);
        }
    }

    #[test]
    fn vectorize_dual() {
        create_hyperdual_from_components!(
            Dual,
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
                [2, 0, 0]
            ]
        );

        let ev = parse!("sin(x+y)^2+cos(x+y)^2 - exp(sqrt(x)/sqrt(z)-1)")
            .evaluator(&[parse!("x"), parse!("y"), parse!("z")])
            .build()
            .unwrap();

        let dual = Dualizer::new(Dual::<Complex<Rational>>::new_zero(), vec![]);
        let vec_ev = ev.vectorize(&dual).unwrap();

        let mut vec_f = vec_ev.map_coeff(&|x| x.re.to_f64());
        let mut dest = vec![0.; 9];
        vec_f.evaluate(
            &[
                2.0, 1.0, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.,
                2.0, 1.0, 2., 3., 4., 5., 6., 7., 8.,
            ],
            &mut dest,
        );

        assert!(dest.iter().all(|x| x.abs() < 1e-10));
    }

    #[test]
    fn vectorize_dual_with_external() {
        let dual = Dualizer::new(
            HyperDual::from_values(
                vec![vec![0], vec![1]],
                vec![Complex::<Rational>::new_zero(); 2],
            ),
            vec![],
        );

        let _ = symbol!(
            "symbolica::vec::f",
            eval = EvaluationInfo::new().register(|args: &[f64]| args[0])
        );
        let _ = symbol!(
            "symbolica::vec::f_v",
            eval = EvaluationInfo::new().register_tagged(|tags| if tags[0] == 0 {
                Box::new(|args: &[f64]| args[0])
            } else {
                Box::new(|args: &[f64]| args[1])
            })
        );

        let ev = parse!("symbolica::vec::f(x + 1)")
            .evaluator(&[parse!("x")])
            .build()
            .unwrap();

        let mut vec_ev = ev.vectorize(&dual).unwrap().map_coeff(&|c| c.re.to_f64());

        let mut out = vec![0.; 2];
        vec_ev.evaluate(&[1., 2.], &mut out);
        assert_eq!(out, vec![2., 2.]);
    }

    #[test]
    fn constant_with_args() {
        let r = parse!("zeta(5/6)");
        let numerical = f64::try_from(r.to_float(53)).unwrap();
        let ev = r.evaluator(&[] as &[Atom]).build().unwrap();
        let ev2 = ev.map_coeff(&|c| c.re.to_f64());
        let exported = ev2.export_instructions();
        assert!(matches!(
            exported.instructions[0],
            Instruction::Assign(_, _)
        ));
        assert!((exported.constants[0] - numerical).abs() / numerical < f64::EPSILON);
    }

    #[test]
    fn export_cpp_includes_evaluation_info_snippet() {
        let _ = symbol!(
            "cpp_external",
            eval = EvaluationInfo::new()
                .with_cpp("inline double cpp_external(double x) { return x + 1.; }")
        );

        let ev = parse!("cpp_external(x)")
            .evaluator(&[parse!("x")])
            .build()
            .unwrap()
            .map_coeff(&|x| x.re.to_f64());

        let code = ev
            .export_cpp_str::<f64>("snippet_test", ExportSettings::default())
            .unwrap();

        assert!(code.contains("inline double cpp_external(double x)"));
        assert!(code.contains("cpp_external(params[0])"));
    }

    #[test]
    fn jit_compile() {
        use crate::parse;
        let eval = parse!("x^2 * cos(x)")
            .evaluator(&[parse!("x")])
            .build()
            .unwrap();

        let mut res = [0.; 1];
        let mut eval_re = eval.clone().map_coeff(&|x| x.re.to_f64());
        eval_re.evaluate(&[0.5], &mut res);

        let mut jit_eval_re = eval_re
            .jit_compile(
                JITCompilationSettings::new()
                    .direct_translation(true)
                    .optimization_level(2),
            )
            .unwrap();

        let mut jit_res = [0.; 1];
        jit_eval_re.evaluate(&[0.5], &mut jit_res);
        assert_eq!(res[0], jit_res[0]);

        let mut res = [Complex::new(0., 0.); 1];
        let mut eval_c = eval
            .clone()
            .map_coeff(&|x| Complex::new(x.re.to_f64(), x.im.to_f64()));
        eval_c.evaluate(&[Complex::new(0.5, 1.2)], &mut res);

        let mut jit_eval_c = eval
            .jit_compile::<Complex<f64>>(JITCompilationSettings::default())
            .unwrap();
        let mut jit_res = [Complex::new(0., 0.); 1];
        jit_eval_c.evaluate(&[Complex::new(0.5, 1.2)], &mut jit_res);
        assert_eq!(res[0], jit_res[0]);
    }
}
