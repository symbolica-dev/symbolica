use symbolica::{
    atom::{Atom, AtomCore},
    function, parse, symbol,
};

fn main() {
    let input = parse!("x*(1+a)+x*5*y+f(5,x)+2+y^2+x^2 + x^3");
    let x = Atom::var(symbol!("x"));
    let key = symbol!("key");
    let coeff = symbol!("val");

    let r = input.coefficient_list::<i8>(std::slice::from_ref(&x));

    println!("> Coefficient list:");
    for (key, val) in r {
        println!("\t{key} {val}");
    }

    println!("> Collect in x:");
    let out = input.collect::<i8>(
        &x,
        Some(Box::new(|x, out| {
            out.set_from_view(&x);
        })),
        None,
    );
    println!("\t{out}");

    println!("> Collect in x with wrapping:");
    let out = input.collect::<i8>(
        &x,
        Some(Box::new(move |a, out| {
            out.set_from_view(&a);
            *out = function!(key, out);
        })),
        Some(Box::new(move |a, out| {
            out.set_from_view(&a);
            *out = function!(coeff, out);
        })),
    );
    println!("\t{out}");
}
