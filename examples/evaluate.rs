use ahash::HashMap;
use symbolica::atom::Atom;
use symbolica::atom::AtomCore;
use symbolica::evaluate::EvaluationFn;
use symbolica::{parse, symbol};

fn main() {
    let x = symbol!("x");
    let f = symbol!("f");
    let g = symbol!("g");
    let p0 = parse!("p(0)");
    let a = parse!("x*cos(x) + f(x, 1)^2 + g(g(x)) + p(0)");

    let mut const_map = HashMap::default();
    let mut fn_map: HashMap<_, _> = HashMap::default();

    // x = 6 and p(0) = 7
    const_map.insert(Atom::var(x), 6.);
    const_map.insert(p0, 7.);

    // f(x, y) = x^2 + y
    fn_map.insert(
        f,
        EvaluationFn::new(Box::new(|args: &[f64], _, _, _| {
            args[0] * args[0] + args[1]
        })),
    );

    // g(x) = f(x, 3)
    fn_map.insert(
        g,
        EvaluationFn::new(Box::new(move |args: &[f64], var_map, fn_map, cache| {
            fn_map.get(&f).unwrap().get()(&[args[0], 3.], var_map, fn_map, cache)
        })),
    );

    println!(
        "Result for x = 6.: {}",
        a.evaluate(|x| x.into(), &const_map, &fn_map).unwrap()
    );
}
