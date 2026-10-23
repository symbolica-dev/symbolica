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
        atomic::{AtomicBool, Ordering},
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
    ComplexPhase, ExportedConstantFunction, ExportedInstructions, ExportedSubEvaluator,
    Instruction, InstructionList, Label, Slot, VectorInstruction,
};
pub use optimize::*;
pub use tree::*;

use function_map::Expr;
use instruction::Instr;

use crate::{
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
    error, get_symbol, info,
    license::LicenseManager,
    numerical_integration::MonteCarloRng,
    utils::AbortCheck,
};

/// The number of operations needed by an evaluator or expression tree.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct OperationCount {
    /// The number of additions.
    pub additions: usize,
    /// The number of multiplications.
    pub multiplications: usize,
    /// The number of inversions.
    pub inversions: usize,
    /// The number of function calls.
    pub function_calls: usize,
}

impl OperationCount {
    /// Create a new operation count.
    pub fn new(
        additions: usize,
        multiplications: usize,
        inversions: usize,
        function_calls: usize,
    ) -> Self {
        Self {
            additions,
            multiplications,
            inversions,
            function_calls,
        }
    }

    /// Add the cost of raising a value to an integer power.
    ///
    /// Negative powers count as one inversion plus the multiplications required for the absolute
    /// power. For example, `x^-3` counts as one inversion and two multiplications.
    pub fn add_integer_power(&mut self, exponent: i64) {
        if exponent < 0 {
            self.inversions += 1;
        }

        self.multiplications += exponent.unsigned_abs().saturating_sub(1) as usize;
    }

    /// Add one function call.
    pub fn add_function_call(&mut self) {
        self.function_calls += 1;
    }
}

impl std::ops::Add for OperationCount {
    type Output = OperationCount;

    fn add(self, rhs: Self) -> Self::Output {
        OperationCount {
            additions: self.additions + rhs.additions,
            multiplications: self.multiplications + rhs.multiplications,
            inversions: self.inversions + rhs.inversions,
            function_calls: self.function_calls + rhs.function_calls,
        }
    }
}

impl std::ops::AddAssign for OperationCount {
    fn add_assign(&mut self, rhs: Self) {
        self.additions += rhs.additions;
        self.multiplications += rhs.multiplications;
        self.inversions += rhs.inversions;
        self.function_calls += rhs.function_calls;
    }
}

impl std::fmt::Display for OperationCount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} +, {} ×, {} x⁻¹, {} f(·)",
            self.additions, self.multiplications, self.inversions, self.function_calls
        )
    }
}

#[cfg(test)]
mod test {
    use ahash::HashMap;
    use numerica::domains::{dual::HyperDual, float::Real};

    #[cfg(feature = "native_code_generation")]
    use crate::evaluate::{CompileOptions, ExportSettings, InlineASM, JITCompilationSettings};
    use crate::{
        atom::{Atom, AtomCore, EvaluationInfo},
        create_hyperdual_from_components,
        domains::{
            float::{Complex, Float},
            rational::Rational,
        },
        evaluate::{
            Dualizer, EvaluationError, ExportedInstructions, FunctionMap,
            FunctionRegistrationOptions, InliningPolicy, Instruction, OptimizationSettings, Slot,
        },
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

        #[cfg(feature = "native_code_generation")]
        {
            let mut jit_compiled = r_f64
                .jit_compile(JITCompilationSettings::default())
                .unwrap();

            jit_compiled.evaluate(&[0.1], &mut res);
            assert_eq!(res[0], 0.2727397524895022);
        }

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
    fn horner_scheme_uses_function_map_bodies() {
        let mut fn_map = FunctionMap::new();
        fn_map
            .add_function(symbol!("f"), vec![symbol!("x")], parse!("x^3+x^2+x+1"))
            .unwrap();

        let expr = parse!("exp(f(y))+log(f(y))");
        let params = vec![parse!("y")];
        let without_horner = expr
            .evaluator(&params)
            .function_map(fn_map.clone())
            .horner_iterations(0)
            .cpe_iterations(Some(0))
            .build()
            .unwrap();
        let with_horner = expr
            .evaluator(&params)
            .function_map(fn_map)
            .horner_iterations(1)
            .cpe_iterations(Some(0))
            .build()
            .unwrap();

        assert!(
            with_horner.count_operations().multiplications
                < without_horner.count_operations().multiplications
        );

        let mut evaluator = with_horner.map_coeff(&|x| x.re.to_f64());
        let value = evaluator.evaluate_single(&[2.]);
        let expected = 15f64.exp() + 15f64.ln();
        assert!((value - expected).abs() < 1e-10);
    }

    #[test]
    fn completed_conditionals_reuse_only_their_parent_scope() {
        let params = [parse!("c"), parse!("x"), parse!("y")];
        let branches = |graph: &ExportedInstructions<Complex<Rational>>| {
            graph
                .instructions
                .iter()
                .filter(|i| matches!(i, Instruction::IfElse(..)))
                .count()
        };
        let repeated = parse!("(if(c,1/x,sqrt(x+2))+x)*(if(c,1/x,sqrt(x+2))+y)");
        let exact = repeated
            .evaluator(&params)
            .horner_iterations(0)
            .build()
            .unwrap();
        assert_eq!(branches(&exact.export_instructions()), 1);
        let heads = [
            parse!("if(c,1/x,sqrt(x+2))+x"),
            parse!("if(c,1/x,sqrt(x+2))+y"),
        ];
        let multiple = Atom::evaluator_multiple(&heads, &params)
            .horner_iterations(0)
            .build()
            .unwrap();
        assert_eq!(branches(&multiple.export_instructions()), 1);
        let mut multiple = multiple.map_coeff(&|c| c.re.to_f64());
        let mut output = [0.; 2];
        multiple.evaluate(&[1., 2., 3.], &mut output);
        assert_eq!(output, [2.5, 3.5]);
        let expression =
            parse!("if(a,(if(c,1/x,sqrt(x+2))+x)*(if(c,1/x,sqrt(x+2))+y),if(c,1/x,sqrt(x+2))+y)");
        let exact = expression
            .evaluator(&[parse!("a"), parse!("c"), parse!("x"), parse!("y")])
            .horner_iterations(0)
            .build()
            .unwrap();
        // The inner expression is completed once in each sibling scope.
        assert_eq!(branches(&exact.export_instructions()), 3);
        let mut scalar = exact.clone().map_coeff(&|c| c.re.to_f64());
        let inputs = [
            [1., 0., 0., 2.],
            [0., 0., 0., 2.],
            [1., 1., -2., 3.],
            [0., 1., -2., 3.],
        ];
        let expected = inputs.map(|[a, c, x, y]| {
            let selected = if c != 0. { 1. / x } else { (x + 2.).sqrt() };
            if a != 0. {
                (selected + x) * (selected + y)
            } else {
                selected + y
            }
        });
        for index in [0, 3, 1, 2, 0, 1] {
            let actual = scalar.evaluate_single(&inputs[index]);
            assert!(actual.is_finite() && (actual - expected[index]).abs() < 1e-12);
        }
        #[cfg(feature = "bincode")]
        {
            let bytes = bincode::encode_to_vec(&exact, bincode::config::standard()).unwrap();
            let (decoded, _) = bincode::decode_from_slice::<
                crate::evaluate::ExpressionEvaluator<Complex<Rational>>,
                _,
            >(&bytes, bincode::config::standard())
            .unwrap();
            let mut decoded = decoded.map_coeff(&|c| c.re.to_f64());
            assert!((decoded.evaluate_single(&inputs[0]) - expected[0]).abs() < 1e-12);
        }
        #[cfg(feature = "native_code_generation")]
        {
            let mut jit = scalar
                .jit_compile(JITCompilationSettings::default())
                .unwrap();
            for (input, expected) in inputs.iter().zip(expected) {
                let mut output = [0.];
                jit.evaluate(input, &mut output);
                assert!((output[0] - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn completed_shared_conditionals_preserve_captures_and_calls() {
        let params = [parse!("c"), parse!("x"), parse!("y")];
        let exact = parse!("conditional_cache::f(c,x)+conditional_cache::f(c,x+1)")
            .evaluator(&params)
            .add_function_with_options(
                symbol!("conditional_cache::f"),
                vec![symbol!("b"), symbol!("z")],
                parse!("(if(b,1/z,z+y)+z)*(if(b,1/z,z+y)+y)"),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap()
            .horner_iterations(0)
            .build()
            .unwrap();
        let exported = exact.export_instructions();
        assert_eq!(exported.sub_evaluators.len(), 1);
        let helper = &exported.sub_evaluators[0];
        assert_eq!(helper.input_count, 3); // b,z,captured y
        assert_eq!(
            helper
                .instructions
                .instructions
                .iter()
                .filter(|i| matches!(i, Instruction::IfElse(..)))
                .count(),
            1
        );
        let mut scalar = exact.map_coeff(&|c| c.re.to_f64());
        for (c, x, y) in [(0., 0., 2.), (1., 2., 3.), (0., -1., 4.), (1., -3., -2.)] {
            let selected = |z| if c != 0. { 1. / z } else { z + y };
            let term = |z| (selected(z) + z) * (selected(z) + y);
            let actual = scalar.evaluate_single(&[c, x, y]);
            assert!(actual.is_finite() && (actual - term(x) - term(x + 1.)).abs() < 1e-12);
        }
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
            .add_function_with_options(
                symbol!("symbolica::sub_eval::helper"),
                vec![symbol!("z")],
                parse!("z^2 + 2"),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap();
        fn_map
            .add_function_with_options(
                symbol!("symbolica::sub_eval::large"),
                vec![symbol!("y")],
                parse!("symbolica::sub_eval::helper(y) + x + 5"),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap();
        fn_map
            .add_tagged_function_with_options(
                symbol!("symbolica::sub_eval::tagged"),
                vec![Atom::num(1)],
                vec![symbol!("y")],
                parse!("3*y + 7"),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap();
        fn_map
            .add_function_with_options(
                symbol!("symbolica::sub_eval::captured"),
                vec![symbol!("q")],
                parse!("q + 1"),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap();
        fn_map
            .add_function(
                symbol!("symbolica::sub_eval::outer"),
                vec![symbol!("q")],
                parse!("symbolica::sub_eval::captured(q)"),
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
        assert_eq!(evaluator.external_fns.len(), 4);
        assert!(
            evaluator
                .external_fns
                .iter()
                .all(|external| external.body.is_some())
        );

        let exported = evaluator
            .clone()
            .map_coeff(&|c| c.re.to_f64())
            .export_instructions();
        assert_eq!(exported.input_count, 1);
        assert_eq!(exported.output_count, 4);
        assert_eq!(exported.sub_evaluators.len(), 3);
        assert!(
            exported
                .sub_evaluators
                .iter()
                .find(|sub| sub.symbol == symbol!("symbolica::sub_eval::large"))
                .unwrap()
                .tags
                .is_empty()
        );
        assert_eq!(
            exported
                .sub_evaluators
                .iter()
                .find(|sub| sub.symbol == symbol!("symbolica::sub_eval::tagged"))
                .unwrap()
                .tags,
            ["1"]
        );
        assert_eq!(
            execute_exported(&exported, &[2.], exported.output_count),
            [13., 18., 13., 3.]
        );

        let large = exported
            .sub_evaluators
            .iter()
            .find(|sub| sub.symbol == symbol!("symbolica::sub_eval::large"))
            .unwrap();
        // Explicit arguments, captured values, and hoisted coefficients are all passed through the
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

    #[test]
    fn non_inlined_function_optimizes_instructions_and_respects_pair_budget() {
        let build = |rounds| {
            parse!("shared_body_opt::outer(x)+shared_body_opt::outer(x+1)")
                .evaluator(&[parse!("x"), parse!("b"), parse!("c"), parse!("d")])
                .add_function(
                    symbol!("shared_body_opt::inner"),
                    vec![symbol!("z")],
                    parse!("sin(z)"),
                )
                .unwrap()
                .add_function_with_options(
                    symbol!("shared_body_opt::outer"),
                    vec![symbol!("a")],
                    // Inlining inner produces the same instruction as sin(a),
                    // but in a different symbolic subexpression cache.
                    parse!("shared_body_opt::inner(a)*b*c+sin(a)*b*d"),
                    FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
                )
                .unwrap()
                .horner_iterations(0)
                .cpe_iterations(rounds)
                .build()
                .unwrap()
        };
        for rounds in [Some(0), Some(1), None] {
            let exact = build(rounds);
            assert_eq!(exact.external_fns.len(), 1);
            let body = exact.external_fns[0].body.as_ref().unwrap();
            assert_eq!(
                body.instructions
                    .iter()
                    .filter(|(i, _)| matches!(i, super::Instr::BuiltinFun(..)))
                    .count(),
                1,
                "instruction CSE must run even with CPE disabled"
            );
            let multiplications: usize = body
                .instructions
                .iter()
                .filter_map(|(i, _)| match i {
                    super::Instr::Mul(_, args) => Some(args.len() - 1),
                    _ => None,
                })
                .sum();
            assert_eq!(multiplications, if rounds == Some(0) { 4 } else { 3 });
            assert!(std::sync::Arc::ptr_eq(
                body,
                exact.clone().external_fns[0].body.as_ref().unwrap()
            ));
            let mut dual = exact
                .clone()
                .vectorize(&Dualizer::new(
                    crate::domains::dual::HyperDual::<Complex<Rational>>::new(vec![
                        vec![0],
                        vec![1],
                    ]),
                    vec![],
                ))
                .unwrap()
                .map_coeff(&|c| c.re.to_f64());
            let mut evaluator = exact.map_coeff(&|c| c.re.to_f64());
            for x in [0.2_f64, -0.7, 1.3] {
                let expected = (x.sin() + (x + 1.).sin()) * 2. * (3. + 5.);
                let actual = evaluator.evaluate_single(&[x, 2., 3., 5.]);
                assert!((actual - expected).abs() < 1e-12);
                let mut out = [0.; 2];
                dual.evaluate(&[x, 1., 2., 0., 3., 0., 5., 0.], &mut out);
                assert!((out[0] - expected).abs() < 1e-12);
                let derivative = (x.cos() + (x + 1.).cos()) * 2. * (3. + 5.);
                assert!((out[1] - derivative).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn non_inlined_function_reuses_temporary_stack_slots() {
        // Keep this regression on libtest's default stack even when a local Cargo
        // configuration raises RUST_MIN_STACK.
        std::thread::Builder::new()
            .name("non_inlined_function_temporary_stack_slots".into())
            .stack_size(2 * 1024 * 1024)
            .spawn(check_non_inlined_function_temporary_stack_slots)
            .unwrap()
            .join()
            .unwrap();
    }

    fn check_non_inlined_function_temporary_stack_slots() {
        let helper = symbol!("symbolica::shared_stack::chain");
        let argument = symbol!("symbolica::shared_stack::argument");
        let parent = symbol!("symbolica::shared_stack::parent");
        let captured = parse!("symbolica::shared_stack::captured");
        let mut body = Atom::var(argument);
        for _ in 0..64 {
            body = symbol!("sin").call((&body,)) + &captured;
        }
        let expression = parent.call((parse!("x"),)) + parent.call((parse!("x+1"),));
        let params = [parse!("x"), captured];
        let build = |policy| {
            expression
                .evaluator(&params)
                .add_function_with_options(
                    helper,
                    vec![argument],
                    body.clone(),
                    FunctionRegistrationOptions::new().inlining(policy),
                )
                .unwrap()
                .add_function_with_options(
                    parent,
                    vec![argument],
                    helper.call((symbol!("sin").call((Atom::var(argument),)),)),
                    FunctionRegistrationOptions::new().inlining(policy),
                )
                .unwrap()
                .horner_iterations(0)
                .build()
                .unwrap()
        };
        let exact = build(InliningPolicy::Never);
        assert_eq!(exact.external_fns.len(), 2);
        let shared = exact.external_fns[0].body.as_ref().unwrap();
        assert_eq!(exact.external_fns[0].symbol, helper);
        let parent_body = exact.external_fns[1].body.as_ref().unwrap();
        assert!(
            parent_body
                .instructions
                .iter()
                .any(|(instruction, _)| match instruction {
                    super::Instr::ExternalFun(target, _, args) => args.contains(target),
                    _ => false,
                })
        );
        assert_eq!(shared.param_count, 2);
        assert!(shared.instructions.len() >= 64);
        assert!(shared.stack_size <= shared.param_count + 3);
        assert!(shared.result_index < shared.stack_size);
        assert!(
            shared
                .instructions
                .iter()
                .any(|(instruction, _)| match instruction {
                    super::Instr::Add(target, args) => args.contains(target),
                    super::Instr::BuiltinFun(target, _, arg) => target == arg,
                    _ => false,
                })
        );
        let clone = exact.clone();
        assert!(std::sync::Arc::ptr_eq(
            shared,
            clone.external_fns[0].body.as_ref().unwrap()
        ));
        let exported = exact.export_instructions();
        assert!(
            exported.sub_evaluators[0].instructions.sub_evaluators[0]
                .instructions
                .temporary_count
                <= 3
        );
        let mut shared = exact.map_coeff(&|c| c.re.to_f64());
        let mut inline = build(InliningPolicy::Always).map_coeff(&|c| c.re.to_f64());
        for inputs in [[0.2, 0.1], [-0.3, 0.4], [1.5, -0.7]] {
            assert_eq!(
                shared.evaluate_single(&inputs),
                inline.evaluate_single(&inputs)
            );
        }
    }

    #[test]
    fn non_inlined_stack_reuse_preserves_nested_lazy_branches() {
        let build = |policy| {
            parse!("shared_stack_branch::outer(c,x)+shared_stack_branch::outer(c,x+1)")
                .evaluator(&[parse!("c"), parse!("x"), parse!("q")])
                .add_function_with_options(
                    symbol!("shared_stack_branch::inner"),
                    vec![symbol!("b"), symbol!("z")],
                    parse!("if(b,1/z+sqrt(z+2),sqrt(z+2)+q*z)+q"),
                    FunctionRegistrationOptions::new().inlining(policy),
                )
                .unwrap()
                .add_function_with_options(
                    symbol!("shared_stack_branch::outer"),
                    vec![symbol!("a"), symbol!("y")],
                    parse!("if(a,shared_stack_branch::inner(a,y)+q*y,if(q,shared_stack_branch::inner(a,y)*q,1/(y+2)))"),
                    FunctionRegistrationOptions::new().inlining(policy),
                )
                .unwrap()
                .horner_iterations(0)
                .build()
                .unwrap()
                .map_coeff(&|c| c.re.to_f64())
        };
        let mut shared = build(InliningPolicy::Never);
        let mut inline = build(InliningPolicy::Always);
        let inputs = [0., 0., 2., 1., 3., 4., 0., 0., 0., 1., 2., 1., 0., -1., 2.];
        for index in [0, 2, 1, 4, 3, 0] {
            let args = &inputs[3 * index..3 * index + 3];
            let expected = inline.evaluate_single(args);
            let result = shared.evaluate_single(args);
            assert!(result.is_finite());
            assert!((result - expected).abs() < 1e-12);
        }
        #[cfg(feature = "native_code_generation")]
        {
            let mut jit = shared
                .jit_compile(JITCompilationSettings::default())
                .unwrap();
            for args in inputs.as_chunks::<3>().0 {
                let mut result = [0.];
                jit.evaluate(args, &mut result);
                assert!((result[0] - inline.evaluate_single(args)).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn non_inlined_function_discovers_only_required_captures() {
        // Every level introduces a new argument and drops the arguments of its ancestors.
        // Rebuilding after capture pruning recursively duplicates the rest of this chain.
        let depth = 32;
        let functions = (0..depth)
            .map(|i| symbol!(format!("symbolica::sub_eval::deep_f{i}")))
            .collect::<Vec<_>>();
        let arguments = (0..depth)
            .map(|i| symbol!(format!("symbolica::sub_eval::deep_x{i}")))
            .collect::<Vec<_>>();
        let mut fn_map = FunctionMap::new();
        for i in 0..depth {
            let argument = Atom::var(arguments[i]);
            let body = if i + 1 == depth {
                argument + Atom::num(1)
            } else {
                functions[i + 1].call(&[argument][..])
            };
            fn_map
                .add_function_with_options(
                    functions[i],
                    vec![arguments[i]],
                    body,
                    FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
                )
                .unwrap();
        }

        let evaluator = functions[0]
            .call(&[parse!("x")][..])
            .evaluator(&[parse!("unused"), parse!("x")])
            .function_map(fn_map)
            .horner_iterations(0)
            .build()
            .unwrap();
        assert_eq!(evaluator.external_fns.len(), depth);
        for (external, function) in evaluator.external_fns.iter().zip(functions.iter().rev()) {
            assert_eq!(external.symbol, *function);
            let sub = external.body.as_ref().unwrap();
            assert_eq!(sub.param_count, 2);
        }
        let mut evaluator = evaluator.map_coeff(&|c| c.re.to_f64());
        assert_eq!(evaluator.evaluate_single(&[123., 4.]), 5.);
    }

    #[test]
    fn non_inlined_function_capture_inputs() {
        let function = symbol!("symbolica::sub_eval::compact_captures");
        let _ = symbol!(
            "symbolica::sub_eval::compact_external",
            eval = EvaluationInfo::new().register(|args: &[f64]| args[0] + 1.)
        );
        let params = [
            parse!("a"),
            parse!("b"),
            parse!("c"),
            parse!("d"),
            parse!("y"),
        ];
        let expression = function.call(&[parse!("c"), parse!("a")][..]);
        for (body, capture_count) in [
            ("b", 1),
            ("y", 0),
            ("7", 0),
            ("if(b, sin(y) + d*y^b + 1/y, d)", 2),
            ("symbolica::sub_eval::compact_external(y)", 0),
        ] {
            let build = |policy| {
                expression
                    .evaluator(&params)
                    .add_function_with_options(
                        function,
                        vec![symbol!("y"), symbol!("unused_arg")],
                        parse!(body),
                        FunctionRegistrationOptions::new().inlining(policy),
                    )
                    .unwrap()
                    .horner_iterations(0)
                    .build()
                    .unwrap()
                    .map_coeff(&|c| c.re.to_f64())
            };
            let mut evaluator = build(InliningPolicy::Never);
            let mut inlined = build(InliningPolicy::Always);
            let sub = evaluator
                .external_fns
                .iter()
                .find(|e| e.symbol == function)
                .unwrap()
                .body
                .as_ref()
                .unwrap();
            // Keep unused explicit arguments and discover only referenced globals.
            // The enclosing y must be shadowed by the explicit y.
            assert_eq!(
                sub.param_count,
                capture_count + 2 + evaluator.get_constants().len()
            );
            for inputs in [[11., 2., 3., 5., 100.], [11., 0., 3., 5., 200.]] {
                assert_eq!(
                    evaluator.evaluate_single(&inputs),
                    inlined.evaluate_single(&inputs),
                    "{body}"
                );
            }
        }
    }

    #[test]
    fn non_inlined_function_reuses_body_across_scopes() {
        let q = symbol!("symbolica::pure_scope::q");
        let g = symbol!("symbolica::pure_scope::g");
        let f = symbol!("symbolica::pure_scope::f");
        let expressions = [
            parse!("symbolica::pure_scope::g()"),
            parse!("symbolica::pure_scope::f(x) + x"),
        ];
        for policy in [InliningPolicy::Always, InliningPolicy::Never] {
            for reverse in [false, true] {
                let mut expressions = expressions.clone();
                if reverse {
                    expressions.reverse();
                }
                let evaluator = Atom::evaluator_multiple(&expressions, &[parse!("x")])
                    .add_function(q, Vec::<crate::atom::Symbol>::new(), Atom::num(1))
                    .unwrap()
                    .add_function_with_options(
                        f,
                        vec![q],
                        parse!("symbolica::pure_scope::g()"),
                        FunctionRegistrationOptions::new().inlining(policy),
                    )
                    .unwrap()
                    .add_function_with_options(
                        g,
                        Vec::<crate::atom::Symbol>::new(),
                        parse!("symbolica::pure_scope::q + 1"),
                        FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
                    )
                    .unwrap()
                    .build()
                    .unwrap();
                let sub = evaluator
                    .external_fns
                    .iter()
                    .find(|e| e.symbol == g)
                    .unwrap()
                    .body
                    .as_ref()
                    .unwrap();
                // q resolves to its registered definition in both calling scopes.
                assert_eq!(sub.param_count, 1);
                let mut evaluator = evaluator.map_coeff(&|c| c.re.to_f64());
                let expected = if reverse { [5., 2.] } else { [2., 5.] };
                let mut output = [0.; 2];
                evaluator.evaluate(&[3.], &mut output);
                assert_eq!(output, expected);
                #[cfg(feature = "native_code_generation")]
                {
                    let mut compiled = evaluator
                        .jit_compile(JITCompilationSettings::default())
                        .unwrap();
                    compiled.evaluate(&[3.], &mut output);
                    assert_eq!(output, expected);
                }
            }
        }
    }

    #[test]
    fn non_inlined_function_resolves_capture_alias_with_explicit_arguments() {
        let mut evaluator = parse!("symbolica::pure_alias::g(x)")
            .evaluator(&[parse!("x"), parse!("symbolica::pure_alias::y")])
            .add_aliases([(
                parse!("symbolica::pure_alias::q"),
                parse!("symbolica::pure_alias::y^2"),
            )])
            .unwrap()
            .add_function_with_options(
                symbol!("symbolica::pure_alias::g"),
                vec![symbol!("symbolica::pure_alias::y")],
                parse!("symbolica::pure_alias::q + 1"),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap()
            .build()
            .unwrap()
            .map_coeff(&|c| c.re.to_f64());
        assert_eq!(evaluator.evaluate_single(&[3., 100.]), 10.);
    }

    #[test]
    fn non_inlined_function_ignores_caller_function_parameters() {
        let q = crate::atom::Indeterminate::try_from(parse!("symbolica::pure_fun_capture::q(1)"))
            .unwrap();
        let expressions = [
            parse!("symbolica::pure_fun_capture::g()"),
            parse!("symbolica::pure_fun_capture::f(x)"),
        ];
        let mut evaluator = Atom::evaluator_multiple(&expressions, &[parse!("x")])
            .add_function(q.clone(), Vec::<crate::atom::Symbol>::new(), Atom::num(2))
            .unwrap()
            .add_function(
                symbol!("symbolica::pure_fun_capture::f"),
                vec![q],
                parse!("symbolica::pure_fun_capture::g()"),
            )
            .unwrap()
            .add_function_with_options(
                symbol!("symbolica::pure_fun_capture::g"),
                Vec::<crate::atom::Symbol>::new(),
                parse!("symbolica::pure_fun_capture::q(1) + 1"),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap()
            .build()
            .unwrap()
            .map_coeff(&|c| c.re.to_f64());
        let mut output = [0.; 2];
        evaluator.evaluate(&[5.], &mut output);
        assert_eq!(output, [3., 3.]);
    }

    #[test]
    fn non_inlined_function_reuses_body_for_repeated_nested_calls() {
        let depth = 24;
        let functions = (0..depth)
            .map(|i| symbol!(format!("symbolica::pure_repeated::f{i}")))
            .collect::<Vec<_>>();
        let y = symbol!("symbolica::pure_repeated::y");
        let mut fn_map = FunctionMap::new();
        for i in 0..depth {
            let argument = Atom::var(y);
            let body = if i + 1 == depth {
                argument
            } else {
                functions[i + 1].call((&argument,))
                    + functions[i + 1].call((argument + Atom::num(1),))
            };
            fn_map
                .add_function_with_options(
                    functions[i],
                    vec![y],
                    body,
                    FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
                )
                .unwrap();
        }
        let evaluator = functions[0]
            .call((parse!("x"),))
            .evaluator(&[parse!("x")])
            .function_map(fn_map)
            .horner_iterations(0)
            .build()
            .unwrap();
        // The expanded call tree has millions of nodes, but there are only `depth` bodies.
        // Inspect the stored bodies without executing that exponentially large call tree.
        assert_eq!(evaluator.external_fns.len(), depth);
        for (i, external) in evaluator.external_fns.iter().enumerate() {
            assert_eq!(external.symbol, functions[depth - i - 1]);
            let sub = external.body.as_ref().unwrap();
            assert_eq!(
                sub.instructions
                    .iter()
                    .filter(|(i, _)| matches!(i, super::Instr::ExternalFun(..)))
                    .count(),
                if i == 0 { 0 } else { 2 }
            );
            for (instruction, _) in &sub.instructions {
                if let super::Instr::ExternalFun(_, target, _) = instruction {
                    assert!(*target < i);
                }
            }
        }
    }

    #[test]
    fn non_inlined_function_hoists_external_constants_without_changing_call_indices() {
        let _ = symbol!(
            "symbolica::pure_external::h",
            eval = EvaluationInfo::new().register(|args: &[f64]| args[0] + 2.)
        );
        let mut evaluator =
            parse!("symbolica::pure_external::g(x) + symbolica::pure_external::g(x+1)")
                .evaluator(&[parse!("x")])
                .add_function_with_options(
                    symbol!("symbolica::pure_external::g"),
                    vec![symbol!("y")],
                    parse!("symbolica::pure_external::h(0) + symbolica::pure_external::h(y)"),
                    FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
                )
                .unwrap()
                .build()
                .unwrap()
                .map_coeff(&|c| c.re.to_f64());
        assert_eq!(evaluator.evaluate_single(&[3.]), 15.);
    }

    #[test]
    fn non_inlined_recursive_function_returns_error() {
        for body in [
            "symbolica::pure_recursive::f()",
            "symbolica::pure_recursive::q",
        ] {
            let result = parse!("symbolica::pure_recursive::f()")
                .evaluator(&Vec::<Atom>::new())
                .add_aliases([(
                    parse!("symbolica::pure_recursive::q"),
                    parse!("symbolica::pure_recursive::f()"),
                )])
                .unwrap()
                .add_function_with_options(
                    symbol!("symbolica::pure_recursive::f"),
                    Vec::<crate::atom::Symbol>::new(),
                    parse!(body),
                    FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
                )
                .unwrap()
                .build();
            assert!(matches!(
                result,
                Err(EvaluationError::EvaluatorConstructionFailed { .. })
            ));
        }
    }

    #[test]
    fn non_inlined_function_requires_declared_arguments_when_used_as_a_variable() {
        let function = symbol!("symbolica::pure_missing_arg::f");
        let result = Atom::var(function)
            .evaluator(&[parse!("y")])
            .add_function_with_options(
                function,
                vec![symbol!("y")],
                parse!("y + 1"),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap()
            .build();
        assert!(
            matches!(result, Err(EvaluationError::WrongNumberOfArguments {
            function: found, expected: 1, actual: 0,
        }) if found == function)
        );
    }

    #[test]
    fn non_inlined_alias_recursion_cannot_be_broken_by_caller_bindings() {
        let result = parse!("symbolica::pure_bound_capture::g()")
            .evaluator(&Vec::<Atom>::new())
            .add_aliases([(
                parse!("symbolica::pure_bound_capture::q"),
                parse!("symbolica::pure_bound_capture::f(2)"),
            )])
            .unwrap()
            .add_function(
                symbol!("symbolica::pure_bound_capture::f"),
                vec![symbol!("symbolica::pure_bound_capture::q")],
                parse!("symbolica::pure_bound_capture::g()"),
            )
            .unwrap()
            .add_function_with_options(
                symbol!("symbolica::pure_bound_capture::g"),
                Vec::<crate::atom::Symbol>::new(),
                parse!("symbolica::pure_bound_capture::q + 1"),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap()
            .build();
        assert!(matches!(
            result,
            Err(EvaluationError::EvaluatorConstructionFailed { .. })
        ));
    }

    #[test]
    fn function_registration_options_control_inlining() {
        let never = symbol!("symbolica::sub_eval::options_never");
        let always = symbol!("symbolica::sub_eval::options_always");
        let auto = symbol!("symbolica::sub_eval::options_auto");
        let evaluator = parse!(
            "symbolica::sub_eval::options_never(x) + symbolica::sub_eval::options_always(x) + symbolica::sub_eval::options_auto(x)"
        )
        .evaluator(&[parse!("x")])
        .add_function_with_options(
            never,
            vec![symbol!("y")],
            parse!("y^2 + 1"),
            FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
        )
        .unwrap()
        .add_function_with_options(
            always,
            vec![symbol!("y")],
            parse!("2*y"),
            FunctionRegistrationOptions::new().inlining(InliningPolicy::Always),
        )
        .unwrap()
        .add_function_with_options(
            auto,
            vec![symbol!("y")],
            parse!("y + 3"),
            FunctionRegistrationOptions::new().inlining(InliningPolicy::Auto),
        )
        .unwrap()
        .build()
        .unwrap();

        assert_eq!(evaluator.external_fns.len(), 1);
        assert_eq!(evaluator.external_fns[0].symbol, never);
        assert!(evaluator.external_fns[0].body.is_some());

        let mut evaluator = evaluator.map_coeff(&|coefficient| coefficient.re.to_f64());
        let mut out = [0.];
        evaluator.evaluate(&[2.], &mut out);
        assert_eq!(out, [14.]);
    }

    #[cfg(feature = "bincode")]
    #[test]
    fn function_registration_options_can_be_serialized() {
        let options = FunctionRegistrationOptions::new().inlining(InliningPolicy::Auto);
        let bytes = bincode::encode_to_vec(&options, bincode::config::standard()).unwrap();
        let (decoded, _) = bincode::decode_from_slice::<FunctionRegistrationOptions, _>(
            &bytes,
            bincode::config::standard(),
        )
        .unwrap();
        assert_eq!(decoded, options);
    }

    #[cfg(feature = "native_code_generation")]
    #[test]
    fn jit_compiles_non_inlined_function_registered_with_options() {
        let function = symbol!("symbolica::sub_eval::jit_options");
        let evaluator =
            parse!("symbolica::sub_eval::jit_options(x) + symbolica::sub_eval::jit_options(x + 1)")
                .evaluator(&[parse!("x")])
                .add_function_with_options(
                    function,
                    vec![symbol!("y")],
                    parse!("y^2 + 2"),
                    FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
                )
                .unwrap()
                .build()
                .unwrap()
                .map_coeff(&|coefficient| coefficient.re.to_f64());

        let mut compiled = evaluator
            .jit_compile(JITCompilationSettings::default())
            .unwrap();
        let mut out = [0.];
        compiled.evaluate(&[3.], &mut out);
        assert_eq!(out, [29.]);
        compiled.evaluate(&[5.], &mut out);
        assert_eq!(out, [65.]);
    }

    #[cfg(feature = "native_code_generation")]
    #[test]
    fn jit_compiles_nested_sub_evaluators() {
        let evaluator = parse!("symbolica::sub_eval::jit_f(x) + 1")
            .evaluator(&[parse!("x")])
            .add_function_with_options(
                symbol!("symbolica::sub_eval::jit_g"),
                vec![symbol!("z")],
                parse!("z^2 + 2"),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap()
            .add_function_with_options(
                symbol!("symbolica::sub_eval::jit_f"),
                vec![symbol!("y")],
                parse!("symbolica::sub_eval::jit_g(y)*y + 5"),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap()
            .build()
            .unwrap();

        assert!(
            evaluator
                .external_fns
                .iter()
                .all(|external| { external.body.is_some() && external.imp.is_none() })
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
    fn jit_compiles_repeated_nested_sub_evaluator_calls() {
        let mut fn_map = FunctionMap::new();
        fn_map
            .add_function_with_options(
                symbol!("f"),
                vec![symbol!("y")],
                parse!("3 + y^2 + g(y)"),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap();
        fn_map
            .add_function_with_options(
                symbol!("g"),
                vec![symbol!("z")],
                parse!("z"),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap();

        let evaluator = parse!("f(x) + f(2)")
            .evaluator(&[parse!("x")])
            .function_map(fn_map)
            .build()
            .unwrap();

        let mut evaluator = evaluator.map_coeff(&|x| x.to_real().unwrap().into());
        let mut out = [0.];
        evaluator.evaluate(&[5.], &mut out);
        assert_eq!(out, [42.]);

        let mut compiled = evaluator
            .jit_compile(JITCompilationSettings::default())
            .unwrap();
        compiled.evaluate(&[5.], &mut out);
        assert_eq!(out, [42.]);

        #[cfg(feature = "bincode")]
        {
            let bytes = bincode::encode_to_vec(&compiled, bincode::config::standard()).unwrap();
            let (mut decoded, _) = bincode::decode_from_slice::<
                crate::evaluate::JITCompiledEvaluator<f64>,
                _,
            >(&bytes, bincode::config::standard())
            .unwrap();
            decoded.evaluate(&[7.], &mut out);
            assert_eq!(out, [68.]);
        }
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
            .add_function_with_options(
                g,
                vec![symbol!("z")],
                parse!("z*z + 2"),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap()
            .add_function_with_options(
                f,
                vec![symbol!("y")],
                parse!("non_inlined_functions::g(y) + y*y + 1"),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap()
            .build()
            .unwrap();

        let (f_index, _) = evaluator
            .external_fns
            .iter()
            .enumerate()
            .find(|(_, external)| external.symbol == f)
            .unwrap();
        let f_name =
            super::ExpressionEvaluator::external_cpp_name(&evaluator.external_fns, f_index);
        let (g_index, _) = evaluator
            .external_fns
            .iter()
            .enumerate()
            .find(|(_, external)| external.symbol == g)
            .unwrap();
        let g_name =
            super::ExpressionEvaluator::external_cpp_name(&evaluator.external_fns, g_index);
        assert_eq!(f_name, "f_1");
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
            .add_function_with_options(
                wrapper,
                vec![symbol!("y")],
                parse!("symbolica::sub_eval::exported_external(y)"),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
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
                Instruction::Fun(_, function, _) => Some(function.as_ref()),
                _ => None,
            })
            .unwrap();
        assert_eq!(*symbol, external);
        assert!(tags.is_empty());
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

    #[cfg(feature = "native_code_generation")]
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

    #[cfg(feature = "native_code_generation")]
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
