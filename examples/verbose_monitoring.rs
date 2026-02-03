use symbolica::{
    atom::AtomCore,
    evaluate::{FunctionMap, OptimizationSettings, VerboseMode},
    parse,
};

fn main() {
    let expr = parse!("x^2 + x^3 + x^4 + x^5");
    let params = vec![parse!("x")];
    let fn_map = FunctionMap::new();

    println!("VerboseMode::None");
    let settings = OptimizationSettings::default();
    let _ = expr.evaluator(&fn_map, &params, settings).unwrap();

    println!("\nVerboseMode::Simple");
    let settings = OptimizationSettings {
        verbose: VerboseMode::Simple,
        horner_iterations: 50,
        cpe_iterations: Some(3),
        ..OptimizationSettings::default()
    };
    let _ = expr.evaluator(&fn_map, &params, settings).unwrap();

    println!("\nVerboseMode::Progress");
    let settings = OptimizationSettings {
        verbose: VerboseMode::Progress,
        horner_iterations: 50,
        cpe_iterations: Some(3),
        ..OptimizationSettings::default()
    };
    let _ = expr.evaluator(&fn_map, &params, settings).unwrap();

    println!("\nDone.");
}
