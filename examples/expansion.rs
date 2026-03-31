use symbolica::{atom::AtomCore, parse};

fn main() {
    let input = parse!("(1+x)^3");

    let o = input.expand();

    println!("> Expansion of {input}: {o}");
}
