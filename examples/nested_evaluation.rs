use symbolica::{
    atom::{Atom, AtomCore},
    domains::{float::Complex, rational::Rational},
    evaluate::{CompileOptions, ExportSettings, FunctionMap, InlineASM, OptimizationSettings},
    parse, symbol,
};

fn main() {
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

    let evaluator = Atom::evaluator_multiple(
        &[e1.as_view(), e2.as_view()],
        &fn_map,
        &params,
        OptimizationSettings::default(),
    )
    .unwrap();

    let mut e_f64 = evaluator.map_coeff(&|x| x.to_real().unwrap().into());
    let mut out = vec![0., 0.];
    e_f64.evaluate(&[5.], &mut out);
    println!("{}", out[0]);

    let mut compiled = e_f64
        .export_cpp::<f64>(
            "nested_evaluate.cpp",
            "nested",
            ExportSettings {
                include_header: true,
                inline_asm: InlineASM::default(),
                ..Default::default()
            },
        )
        .unwrap()
        .compile("nested", CompileOptions::default())
        .unwrap()
        .load()
        .unwrap();

    compiled.evaluate(&[5.], &mut out);
    println!("{}", out[0]);
}
