use symbolica::{
    atom::{Atom, AtomCore},
    evaluate::{EvaluatorMonitoring, FunctionMap, OptimizationSettings},
    parse, symbol,
};

fn main() {
    // Build a moderately complex expression to demonstrate monitoring.
    let e1 = parse!("x + cos(x) + f(g(x+1),h(x*2))");
    let e2 = parse!("x + h(x*2) + cos(x)");
    let f = parse!("y^2 + z^2*y^2");
    let g = parse!("y+7+x*(y+7)*(y-1)");
    let h = parse!("y*(1+x*(1+x^2)) + y^2*(1+x*(1+x^2))^2 + 3*(1+x^2)");

    let mut fn_map = FunctionMap::new();
    fn_map
        .add_function(
            symbol!("f"),
            "f".to_string(),
            vec![symbol!("y"), symbol!("z")],
            f,
        )
        .unwrap();
    fn_map
        .add_function(symbol!("g"), "g".to_string(), vec![symbol!("y")], g)
        .unwrap();
    fn_map
        .add_function(symbol!("h"), "h".to_string(), vec![symbol!("y")], h)
        .unwrap();

    let params = vec![parse!("x")];

    // --- Silent mode: no output ---
    println!("=== Silent mode ===");
    let _evaluator = Atom::evaluator_multiple(
        &[e1.as_view(), e2.as_view()],
        &fn_map,
        &params,
        OptimizationSettings {
            monitoring: EvaluatorMonitoring::Silent,
            ..OptimizationSettings::default()
        },
    )
    .unwrap();
    println!("(no output expected)\n");

    // --- Verbose mode: text-based logging via tracing ---
    println!("=== Verbose mode ===");
    let _evaluator = Atom::evaluator_multiple(
        &[e1.as_view(), e2.as_view()],
        &fn_map,
        &params,
        OptimizationSettings {
            monitoring: EvaluatorMonitoring::Verbose,
            horner_iterations: 100,
            ..OptimizationSettings::default()
        },
    )
    .unwrap();
    println!();

    // --- Progress mode: indicatif progress bars + memory stats ---
    println!("=== Progress mode ===");
    let evaluator = Atom::evaluator_multiple(
        &[e1.as_view(), e2.as_view()],
        &fn_map,
        &params,
        OptimizationSettings {
            monitoring: EvaluatorMonitoring::Progress,
            horner_iterations: 100,
            cpe_iterations: Some(5),
            ..OptimizationSettings::default()
        },
    )
    .unwrap();
    println!();

    // Evaluate to verify correctness
    let mut e_f64 = evaluator.map_coeff(&|x| x.to_real().unwrap().into());
    let mut out = vec![0., 0.];
    e_f64.evaluate(&[5.], &mut out);
    println!("Evaluation result: [{}, {}]", out[0], out[1]);
}
