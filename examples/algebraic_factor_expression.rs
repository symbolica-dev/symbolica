use symbolica::prelude::*;

fn main() {
    let expression = parse!("x^6 - 2");
    let extensions = [parse!("sqrt(2)"), parse!("root(x^24-7,0)")];

    let (context, polynomial) = expression
        .to_polynomial_in_algebraic_extension::<u16>(symbol!("x"), &extensions)
        .unwrap();
    println!("{polynomial}");

    let factors = polynomial.factor();
    let factor = factors[0].0.to_expression_with_context(&context).unwrap();
    println!("{factor}");
}
