use symbolica::{
    atom::{Atom, AtomCore},
    domains::{float::Complex, rational::Rational},
    evaluate::{FunctionMap, OptimizationSettings, VerboseMode},
    parse, symbol,
};

fn main() {
    let x = parse!("x");
    let y = parse!("y");
    let z = parse!("z");
    let params = vec![x.clone(), y.clone(), z.clone()];

    let mut fn_map = FunctionMap::new();
    fn_map.add_constant(
        symbol!("pi").into(),
        Complex::from(Rational::from((22, 7))),
    );

    let f_body = parse!(
        "x^2*y + x*y^2 + (x+y)^3 + (x+y)^4 + (x+y)^5 + (x+y)^6 + (x+y)^7 + (x+y)^8"
    );
    fn_map
        .add_function(
            symbol!("f"),
            "f".to_string(),
            vec![symbol!("x"), symbol!("y")],
            f_body,
        )
        .unwrap();

    let g_body = parse!("(x+1)^10 + (x+2)^10 + (x+3)^10 + (x+4)^10");
    fn_map
        .add_function(symbol!("g"), "g".to_string(), vec![symbol!("x")], g_body)
        .unwrap();

    let h_body = parse!("(x+y+z)^8 + (x+y)^6 + (y+z)^6 + (x+z)^6");
    fn_map
        .add_function(
            symbol!("h"),
            "h".to_string(),
            vec![symbol!("x"), symbol!("y"), symbol!("z")],
            h_body,
        )
        .unwrap();

    let e1 = parse!(
        "f(x,y) + f(x,y) + f(y,x) + g(x) + g(x) + h(x,y,z) + h(x,y,z) + pi"
    );
    let e2 = parse!(
        "f(x,y) * f(x,y) + (g(x) + g(x)) * (g(x) + g(x)) + (x+y+z)^12"
    );
    let e3 = parse!(
        "(x+y)^10 + (x+y)^10 + (x+y)^10 + (x+y)^10 + (x+y)^10 + (x+y)^10 + (x+y)^10"
    );

    println!("\n=== Case 1: VerboseMode::Progress (bounded CPE target) ===");
    let settings = OptimizationSettings {
        verbose: VerboseMode::Progress,
        horner_iterations: 200,
        n_cores: 4,
        cpe_iterations: Some(6),
        max_common_pair_cache_entries: 200_000,
        max_common_pair_distance: 500,
        ..OptimizationSettings::default()
    };

    let eval = Atom::evaluator_multiple(
        &[e1.as_view(), e2.as_view(), e3.as_view()],
        &fn_map,
        &params,
        settings,
    )
    .unwrap();

    let mut eval_f64 = eval.map_coeff(&|c| c.to_real().unwrap().to_f64());
    let mut out = vec![0.0f64; 3];
    eval_f64.evaluate(&[1.0, 2.0, 3.0], &mut out);
    println!("Outputs (x=1, y=2, z=3): {out:?}");


    println!("\n=== Case 2: VerboseMode::Simple ===");
    let settings = OptimizationSettings {
        verbose: VerboseMode::Simple,
        horner_iterations: 80,
        n_cores: 4,
        cpe_iterations: Some(3),
        ..OptimizationSettings::default()
    };
    let _ = Atom::evaluator_multiple(&[e1.as_view()], &fn_map, &params, settings).unwrap();


    println!("\n=== Case 3: VerboseMode::Progress (unbounded CPE target = None) ===");
    let small = parse!("(x+y+z)^10 + (x+y+z)^10 + (x+y+z)^10 + (x+y+z)^10");
    let settings = OptimizationSettings {
        verbose: VerboseMode::Progress,
        horner_iterations: 120,
        n_cores: 4,
        cpe_iterations: None,
        max_common_pair_cache_entries: 150_000,
        max_common_pair_distance: 300,
        ..OptimizationSettings::default()
    };
    let _ = small.evaluator(&fn_map, &params, settings).unwrap();

    println!("\n=== Case 4: VerboseMode::None ===");
    let settings = OptimizationSettings::default();
    let _ = e3.evaluator(&fn_map, &params, settings).unwrap();

    println!("\nDone.");
}
