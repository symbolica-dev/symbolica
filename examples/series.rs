use symbolica::prelude::*;

fn main() {
    let x = symbol!("x");
    let a = parse!("(1-cos(x))/sin(x)");

    let out = a.series(x, 0, 4).unwrap();

    println!("{out}");
}
