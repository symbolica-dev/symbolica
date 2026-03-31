use symbolica::{atom::AtomCore, parse};

fn main() {
    let expr = parse!(" f(1,2,x) + f(1,2,3)");
    let pat = parse!("f(1,2,y_)");
    let rhs = parse!("f(1,2,y_+1)");

    let out = expr.replace(pat).with(rhs);
    println!("{out}");
}
