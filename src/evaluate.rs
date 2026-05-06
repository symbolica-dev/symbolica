//! Evaluation of expressions.
//!
//! The main entry point is through [AtomCore::evaluator].
use ahash::{AHasher, HashMap, HashMapExt, HashSet};
use dyn_clone::DynClone;
use rand::Rng;
use self_cell::self_cell;
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, hash_map::Entry},
    hash::{Hash, Hasher},
    os::raw::{c_ulong, c_void},
    panic,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};
use symjit::{Applet, Composer, Config, Defuns, Storage, Translator};

mod backend;
mod domain;
mod dual;
mod evaluator;
mod export;
mod external;
mod function_map;
mod instruction;
mod optimize;
mod tree;

pub use backend::*;
pub use domain::*;
pub use dual::*;
pub use evaluator::*;
pub use export::*;
pub use external::*;
pub use function_map::*;
pub use instruction::{ComplexPhase, Instruction, InstructionList, Label, Slot, VectorInstruction};
pub use optimize::*;
pub use tree::*;

use function_map::Expr;
use instruction::Instr;

use crate::{
    LicenseManager,
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
            float::{Complex, Float, FloatLike},
            rational::Rational,
        },
        evaluate::{Dualizer, EvaluationFn, ExportSettings, FunctionMap, OptimizationSettings},
        id::ConditionResult,
        parse, symbol,
    };

    #[test]
    fn eval_fun() {
        let _ = symbol!(
            "e",
            eval = EvaluationInfo::constant(|_tags, prec| { Ok(Float::new(prec).e().into()) })
        );

        let _ = symbol!(
            "symbolica::eval_fun::atanh",
            eval = EvaluationInfo::new()
                .register(|args: &[Complex<Float>]| args[0].atanh())
                .register(|args: &[f64]| args[0].atanh())
        );

        let a = parse!("e*symbolica::eval_fun::atanh(x)");

        assert!(
            (parse!("e*symbolica::eval_fun::atanh(0.1`32)").to_float(32)
                - parse!("2.7273975248950224505081204947890e-1`32"))
            .abs()
                < parse!("1e-30`32")
        );

        let fn_map = FunctionMap::new();

        let r = a
            .evaluator(&fn_map, &[parse!("x")], OptimizationSettings::default())
            .unwrap();

        let mut r_f64 = r.clone().map_coeff(&|x| x.re.to_f64());

        let mut res = [0.];
        r_f64.evaluate(&[0.1], &mut res);
        assert_eq!(res[0], 0.2727397524895022);

        let mut jit_compiled = r_f64.jit_compile().unwrap();

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
        let x = symbol!("v1");
        let f = symbol!("f1");
        let g = symbol!("f2");
        let p0 = parse!("v2(0)");
        let a = parse!("v1*cos(v1) + f1(v1, 1)^2 + f2(f2(v1)) + v2(0)");

        let v = Atom::var(x);

        let mut const_map = HashMap::default();
        let mut fn_map: HashMap<_, EvaluationFn<_, _>> = HashMap::default();

        const_map.insert(v.as_view(), 6.);
        const_map.insert(p0.as_view(), 7.);

        fn_map.insert(
            f,
            EvaluationFn::new(Box::new(|args: &[f64], _, _, _| {
                args[0] * args[0] + args[1]
            })),
        );

        fn_map.insert(
            g,
            EvaluationFn::new(Box::new(move |args: &[f64], var_map, fn_map, cache| {
                fn_map.get(&f).unwrap().get()(&[args[0], 3.], var_map, fn_map, cache)
            })),
        );

        let r = a.evaluate(|x| x.into(), &const_map, &fn_map).unwrap();
        assert_eq!(r, 2905.761021719902);
    }

    #[test]
    fn arb_prec() {
        let x = symbol!("v1");
        let a = parse!("128731/12893721893721 + v1");

        let mut const_map = HashMap::default();

        let v = Atom::var(x);
        const_map.insert(v.as_view(), Float::with_val(200, 6));

        let r = a
            .evaluate(
                |r| r.to_multi_prec_float(200),
                &const_map,
                &HashMap::default(),
            )
            .unwrap();

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

        let evaluator =
            Atom::evaluator_multiple(&[e1, e2], &fn_map, &params, OptimizationSettings::default())
                .unwrap();

        let mut e_f64 = evaluator.map_coeff(&|x| x.clone().to_real().unwrap().into());
        let mut res = [0., 0.];
        e_f64.evaluate(&[1.1], &mut res);
        assert!((res[0] - 1622709.2241624785).abs() / 1622709.2241624785 < 1e-10);
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
                .evaluator(
                    &FunctionMap::new(),
                    &vec![crate::parse!("x"), crate::parse!("y"), crate::parse!("z")],
                    Default::default(),
                )
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
            .evaluator(
                &FunctionMap::new(),
                &[parse!("x"), parse!("y"), parse!("z")],
                OptimizationSettings::default(),
            )
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
            .evaluator(
                &FunctionMap::new(),
                &[parse!("x")],
                OptimizationSettings::default(),
            )
            .unwrap();

        let mut vec_ev = ev.vectorize(&dual).unwrap().map_coeff(&|c| c.re.to_f64());

        let mut out = vec![0.; 2];
        vec_ev.evaluate(&[1., 2.], &mut out);
        assert_eq!(out, vec![2., 2.]);
    }

    #[test]
    fn export_cpp_includes_evaluation_info_snippet() {
        let _ = symbol!(
            "cpp_external",
            eval = EvaluationInfo::new()
                .with_cpp("inline double cpp_external(double x) { return x + 1.; }")
        );

        let ev = parse!("cpp_external(x)")
            .evaluator(
                &FunctionMap::new(),
                &[parse!("x")],
                OptimizationSettings::default(),
            )
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
            .evaluator(
                &FunctionMap::new(),
                &[parse!("x")],
                OptimizationSettings::default(),
            )
            .unwrap();

        let mut res = [0.; 1];
        let mut eval_re = eval.clone().map_coeff(&|x| x.re.to_f64());
        eval_re.evaluate(&[0.5], &mut res);

        let mut jit_eval_re = eval_re.jit_compile().unwrap();

        let mut jit_res = [0.; 1];
        jit_eval_re.evaluate(&[0.5], &mut jit_res);
        assert_eq!(res[0], jit_res[0]);

        let mut res = [Complex::new(0., 0.); 1];
        let mut eval_c = eval
            .clone()
            .map_coeff(&|x| Complex::new(x.re.to_f64(), x.im.to_f64()));
        eval_c.evaluate(&[Complex::new(0.5, 1.2)], &mut res);

        let mut jit_eval_c = eval.jit_compile::<Complex<f64>>().unwrap();
        let mut jit_res = [Complex::new(0., 0.); 1];
        jit_eval_c.evaluate(&[Complex::new(0.5, 1.2)], &mut jit_res);
        assert_eq!(res[0], jit_res[0]);
    }
}
