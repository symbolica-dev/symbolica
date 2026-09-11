use super::*;
use crate::parse;

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
