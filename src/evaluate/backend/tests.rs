use super::*;
use crate::parse;

fn generic_cpp_compile_options() -> CompileOptions {
    CompileOptions::default()
        .compiler(std::env::var("SYMBOLICA_TEST_CXX").unwrap_or_else(|_| "g++".into()))
        .fast_math(false)
        .unsafe_math(false)
        .arg("-std=c++17")
}

fn check_restrict_macro_scope<F: CompiledNumber>(source: &mut String, function_name: &str) {
    // Concatenating generated functions must not leave helper macros visible to
    // later exports or to the caller's source, even with include_header(false).
    let macro_name = format!("{}_RESTRICT", F::construct_function_name(function_name));
    *source += &format!(
        "\n#ifdef {macro_name}\n#error Exported restrict macro escaped its scope\n#endif\n"
    );
}

#[test]
fn generic_cpp_nonoverlapping_real_buffers() {
    let base = std::env::temp_dir().join(format!("symbolica_generic_real_{}", std::process::id()));
    let source = base.with_extension("cpp");
    let library = base.with_extension("so");
    let expressions = [
        parse!("(x+y)*(x-y)+x*y"),
        parse!("x*y+x+y"),
        parse!("x"),
        parse!("y"),
    ];
    let evaluator = Atom::evaluator_multiple(&expressions, &[parse!("x"), parse!("y")])
        .build()
        .unwrap()
        .map_coeff(&|c| c.re.to_f64());
    let code = evaluator
        .export_cpp::<f64>(
            &source,
            "generic_real",
            ExportSettings::new().inline_asm(InlineASM::None),
        )
        .unwrap();
    let mut text = std::fs::read_to_string(&source).unwrap();
    check_restrict_macro_scope::<f64>(&mut text, "generic_real");
    text += &evaluator
        .export_cpp_str::<f64>(
            "generic_real_headerless",
            ExportSettings::new()
                .inline_asm(InlineASM::None)
                .include_header(false),
        )
        .unwrap();
    check_restrict_macro_scope::<f64>(&mut text, "generic_real_headerless");
    std::fs::write(&source, text).unwrap();
    code.compile(&library, generic_cpp_compile_options())
        .unwrap();

    let mut loaded = CompiledRealEvaluator::load(&library, "generic_real").unwrap();
    let mut headerless = loaded.load_new_function("generic_real_headerless").unwrap();
    for args in [[1.25, -0.75], [-2.5, 3.0], [0.125, 1.875]] {
        let [x, y] = args;
        let expected = [(x + y) * (x - y) + x * y, x * y + x + y, x, y];
        for compiled in [&mut loaded, &mut headerless] {
            let mut output = [f64::NAN; 4];
            compiled.evaluate(&args, &mut output);
            for (actual, expected) in output.into_iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-12);
            }
        }
    }
    std::fs::remove_file(source).unwrap();
    std::fs::remove_file(library).unwrap();
}

#[test]
fn generic_cpp_nonoverlapping_complex_buffers() {
    let base =
        std::env::temp_dir().join(format!("symbolica_generic_complex_{}", std::process::id()));
    let source = base.with_extension("cpp");
    let library = base.with_extension("so");
    let expressions = [
        parse!("(x+y)*(x-y)+x*y"),
        parse!("x*y+x+y"),
        parse!("x+y/3"),
        parse!("x"),
        parse!("y"),
    ];
    let evaluator = Atom::evaluator_multiple(&expressions, &[parse!("x"), parse!("y")])
        .build()
        .unwrap()
        .map_coeff(&|c| Complex::new(c.re.to_f64(), c.im.to_f64()));
    let code = evaluator
        .export_cpp::<Complex<f64>>(
            &source,
            "generic_complex",
            ExportSettings::new().inline_asm(InlineASM::None),
        )
        .unwrap();
    let mut text = std::fs::read_to_string(&source).unwrap();
    check_restrict_macro_scope::<Complex<f64>>(&mut text, "generic_complex");
    text += &evaluator
        .export_cpp_str::<Complex<f64>>(
            "generic_complex_headerless",
            ExportSettings::new()
                .inline_asm(InlineASM::None)
                .include_header(false),
        )
        .unwrap();
    check_restrict_macro_scope::<Complex<f64>>(&mut text, "generic_complex_headerless");
    std::fs::write(&source, text).unwrap();
    code.compile(&library, generic_cpp_compile_options())
        .unwrap();

    let mut loaded = CompiledComplexEvaluator::load(&library, "generic_complex").unwrap();
    let mut headerless = loaded
        .load_new_function("generic_complex_headerless")
        .unwrap();
    for args in [
        [Complex::new(1.25, 0.5), Complex::new(-0.75, 2.0)],
        [Complex::new(-2.5, 1.0), Complex::new(3.0, -0.125)],
    ] {
        let mut expected = [Complex::new(0., 0.); 5];
        evaluator.clone().evaluate(&args, &mut expected);
        for compiled in [&mut loaded, &mut headerless] {
            let mut output = [Complex::new(f64::NAN, f64::NAN); 5];
            compiled.evaluate(&args, &mut output);
            for (actual, expected) in output.into_iter().zip(expected) {
                assert!((actual.re - expected.re).abs() < 1e-12);
                assert!((actual.im - expected.im).abs() < 1e-12);
            }
        }
    }
    std::fs::remove_file(source).unwrap();
    std::fs::remove_file(library).unwrap();
}

#[test]
fn compiled_dimensions_survive_reload_and_function_switch() {
    let base = std::env::temp_dir().join(format!("symbolica_dimensions_{}", std::process::id()));
    let source = base.with_extension("cpp");
    let library = base.with_extension("so");
    let evaluator = parse!("x + y")
        .evaluator(&[parse!("x"), parse!("y")])
        .build()
        .unwrap()
        .map_coeff(&|c| c.re.to_f64());
    let code = evaluator
        .export_cpp::<f64>(&source, "sum", ExportSettings::default())
        .unwrap();
    let constant = parse!("7")
        .evaluator::<Atom>(&[])
        .build()
        .unwrap()
        .map_coeff(&|c| c.re.to_f64());
    let mut text = std::fs::read_to_string(&source).unwrap();
    text += &constant
        .export_cpp_str::<f64>("constant", ExportSettings::new().include_header(false))
        .unwrap();
    std::fs::write(&source, text).unwrap();
    code.compile(&library, CompileOptions::default()).unwrap();

    let mut loaded = CompiledRealEvaluator::load(&library, "sum").unwrap();
    assert_eq!((loaded.get_input_len(), loaded.get_output_len()), (2, 1));
    let mut output = [0.; 2];
    loaded
        .evaluate_batch(2, &[1., 2., 3., 4.], &mut output)
        .unwrap();
    assert_eq!(output, [3., 7.]);
    // Divisible batch lengths can still disagree with the compiled signature.
    assert!(loaded.evaluate_batch(2, &[1., 2.], &mut output).is_err());
    assert!(loaded.evaluate_batch(1, &[1., 2.], &mut []).is_err());
    assert!(loaded.evaluate_batch(usize::MAX, &[], &mut []).is_err());
    loaded.evaluate_batch(0, &[], &mut []).unwrap();
    for (input, output_len) in [(&[1.][..], 1), (&[1., 2.][..], 0)] {
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                loaded.evaluate(input, &mut output[..output_len]);
            }))
            .is_err()
        );
    }

    let mut constant = loaded.load_new_function("constant").unwrap().clone();
    assert_eq!(
        (constant.get_input_len(), constant.get_output_len()),
        (0, 1)
    );
    constant.evaluate_batch(2, &[], &mut output).unwrap();
    assert_eq!(output, [7., 7.]);
    std::fs::remove_file(source).unwrap();
    std::fs::remove_file(library).unwrap();
}
