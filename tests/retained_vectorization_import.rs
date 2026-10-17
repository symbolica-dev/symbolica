#![cfg(feature = "bincode")]

use std::{fs, process::Command};

use symbolica::{
    domains::dual::HyperDual,
    evaluate::{FunctionRegistrationOptions, InliningPolicy, ParameterRowEvaluator},
    prelude::*,
};

type Number = Complex<Rational>;

fn evaluator(power: i64) -> ExpressionEvaluator<Number> {
    let x = symbol!("retained_import::x");
    let f = symbol!(format!("retained_import::power_{power}"));
    function!(f, x)
        .evaluator(&[Atom::var(x)])
        .add_function_with_options(
            f,
            vec![x],
            Atom::var(x).pow(power),
            FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
        )
        .unwrap()
        .build()
        .unwrap()
        .vectorize(&Dualizer::new(
            HyperDual::<Number>::new(vec![vec![0], vec![1]]),
            vec![],
        ))
        .unwrap()
}

#[test]
fn imported_vectorized_bodies_keep_identity_when_merged_with_fresh_bodies() {
    const ROLE: &str = "SYMBOLICA_RETAINED_IMPORT_ROLE";
    const PAYLOAD: &str = "SYMBOLICA_RETAINED_IMPORT_PAYLOAD";
    let Ok(role) = std::env::var(ROLE) else {
        let path = std::env::temp_dir().join(format!(
            "symbolica-retained-vector-import-{}",
            std::process::id()
        ));
        for role in ["export", "import"] {
            let status = Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "imported_vectorized_bodies_keep_identity_when_merged_with_fresh_bodies",
                ])
                .env(ROLE, role)
                .env(PAYLOAD, &path)
                .status()
                .unwrap();
            assert!(status.success(), "fresh-process {role} failed");
        }
        fs::remove_file(path).unwrap();
        return;
    };
    let path = std::path::PathBuf::from(std::env::var_os(PAYLOAD).unwrap());
    if role == "export" {
        let bytes = bincode::encode_to_vec(evaluator(2), bincode::config::standard()).unwrap();
        fs::write(path, bytes).unwrap();
    } else {
        let bytes = fs::read(path).unwrap();
        let (mut imported, consumed): (ExpressionEvaluator<Number>, _) =
            bincode::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        assert_eq!(consumed, bytes.len());
        let fresh = evaluator(3);
        let old_functions = imported.export_instructions().sub_evaluators;
        let new_functions = fresh.export_instructions().sub_evaluators;
        assert!(old_functions.iter().all(|old| {
            new_functions
                .iter()
                .all(|new| old.symbol != new.symbol || old.tags != new.tags)
        }));
        imported.merge(fresh, Some(2)).unwrap();
        let mut numeric = imported.map_coeff(&|coefficient| coefficient.re.to_f64());
        let mut outputs = [0.; 4];
        numeric.evaluate(&[2., 1.], &mut outputs);
        assert_eq!(outputs, [4., 4., 8., 12.]);
    }
}

#[test]
fn imported_row_kernels_keep_identity_and_captured_constants() {
    const ROLE: &str = "SYMBOLICA_ROW_KERNEL_IMPORT_ROLE";
    const PAYLOAD: &str = "SYMBOLICA_ROW_KERNEL_IMPORT_PAYLOAD";
    let Ok(role) = std::env::var(ROLE) else {
        let path = std::env::temp_dir().join(format!(
            "symbolica-row-kernel-import-{}",
            std::process::id()
        ));
        for role in ["export", "import"] {
            let status = Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "imported_row_kernels_keep_identity_and_captured_constants",
                ])
                .env(ROLE, role)
                .env(PAYLOAD, &path)
                .status()
                .unwrap();
            assert!(status.success(), "fresh-process {role} failed");
        }
        fs::remove_file(path).unwrap();
        return;
    };
    let path = std::path::PathBuf::from(std::env::var_os(PAYLOAD).unwrap());
    let build = |power: i64| {
        let x = parse!("row_kernel_import::x");
        let c = parse!("row_kernel_import::c");
        let a = symbol!("row_kernel_import::a");
        let b = symbol!("row_kernel_import::b");
        let function = symbol!(format!("row_kernel_import::power_{power}"));
        function
            .call((&x, &c))
            .evaluator(&[x, c.clone()])
            .add_function_with_options(
                function,
                vec![a, b],
                Symbol::SQRT.call_args([Atom::var(a) + Atom::var(Symbol::PI)])
                    + Atom::var(b) * Atom::var(a).pow(power),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap()
            .build_parameter_rows(&[c], &[vec![1.into()], vec![2.into()]])
            .unwrap()
            .vectorize(&Dualizer::new(
                HyperDual::<Number>::new(vec![vec![0], vec![1]]),
                vec![],
            ))
            .unwrap()
    };
    if role == "export" {
        let bytes = bincode::encode_to_vec(build(1), bincode::config::standard()).unwrap();
        fs::write(path, bytes).unwrap();
    } else {
        // Register unrelated symbols before importing, then construct a fresh
        // private kernel: its name must not collide with an imported kernel.
        let _ = symbol!("row_kernel_import::unrelated");
        let bytes = fs::read(path).unwrap();
        let (imported, consumed): (ParameterRowEvaluator<Number>, _) =
            bincode::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        assert_eq!(consumed, bytes.len());
        let fresh = build(2);
        let (_, _, old_body) = imported.clone().into_parts();
        let (_, _, fresh_body) = fresh.clone().into_parts();
        let old_functions = old_body.export_instructions().sub_evaluators;
        let new_functions = fresh_body.export_instructions().sub_evaluators;
        assert!(!old_functions.is_empty());
        assert!(old_functions.iter().all(|old| {
            new_functions
                .iter()
                .all(|new| old.symbol != new.symbol || old.tags != new.tags)
        }));
        for (program, power) in [(imported, 1_i32), (fresh, 2)] {
            let mut numeric = program.map_coeff(&|value| value.re.to_f64());
            let mut output = [0.; 2];
            numeric
                .evaluate(&[4., 1.], &mut output, Rational::to_f64)
                .unwrap();
            let root = (4. + std::f64::consts::PI).sqrt();
            let expected = [
                2. * root + 3. * 4_f64.powi(power),
                1. / root + 3. * f64::from(power) * 4_f64.powi(power - 1),
            ];
            for (actual, expected) in output.into_iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-12);
            }
        }
    }
}
