use symbolica::{
    atom::AtomCore,
    domains::{integer::Z, rational_polynomial::RationalPolynomial},
    parse,
};

fn main() {
    let expr = parse!("(x*y^2*5+5)^2/(2*x+5)+(x+4)/(6*x^2+1)");
    let rat: RationalPolynomial<_, u8> = expr.to_rational_polynomial(&Z, &Z, None);
    println!("{rat}");
}
