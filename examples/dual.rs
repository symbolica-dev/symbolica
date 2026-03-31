use symbolica::{
    create_hyperdual_from_components,
    domains::{float::FloatLike, rational::Rational},
};

create_hyperdual_from_components!(
    Dual,
    [
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
        [2, 0, 0]
    ]
);

fn main() {
    let x = Dual::<Rational>::new_variable(0, (1, 1).into());
    let y = Dual::new_variable(1, (2, 1).into());
    let z = Dual::new_variable(2, (3, 1).into());

    let t3 = x * y * z;

    println!("{}", t3.inv());
}
