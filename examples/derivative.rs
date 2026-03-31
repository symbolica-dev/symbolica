use symbolica::{atom::AtomCore, parse, symbol};

fn main() {
    let x = symbol!("x");
    let inputs = [
        "(1+2*x)^(5+x)",
        "log(2*x) + exp(3*x) + sin(4*x) + cos(y*x)",
        "f(x^2,x)",
        "der(0,1,f(x,x^3))",
    ];

    for input in inputs {
        let input = parse!(input);

        let a = input.derivative(x);

        println!("d({input})/dx = {a}:");
    }
}
