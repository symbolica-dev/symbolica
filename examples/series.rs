use symbolica::{
    atom::{Atom, AtomCore},
    parse, symbol,
};

fn main() {
    let x = symbol!("x");
    let a = parse!("(1-cos(x))/sin(x)");

    let out = a.series(x, Atom::num(0), 4.into(), true).unwrap();

    println!("{out}");
}
