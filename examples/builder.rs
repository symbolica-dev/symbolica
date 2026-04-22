use symbolica::{atom::AtomCore, function, symbol};

fn main() {
    let (x, y, f) = symbol!("x", "y", "f");

    let f = function!(f, x, y, 2);

    let xb = (-(y + x + 2) * y * 6).pow(5) / y * f / 4;

    println!("{xb}");
}
