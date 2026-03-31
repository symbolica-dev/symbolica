use symbolica::domains::{
    Field, RingOps, Set,
    finite_field::{FiniteFieldCore, Zp},
    rational::{Q, Rational},
};

/// An arbitrarily complicated black-box function.
fn black_box(field: &Zp, eval: &[<Zp as Set>::Element]) -> <Zp as Set>::Element {
    // eval -(x^2+y/3)
    field.neg(&field.add(
        &field.mul(&eval[0], &eval[0]),
        &field.div(&eval[1], &field.to_element(3)),
    ))
}

fn main() {
    let r =
        Rational::rational_reconstruction::<_, Q>(black_box, &[(1, 2).into(), (3, 1).into()], None);

    assert_eq!(r, Ok((-5, 4).into()));
    println!("Reconstructed f(1/2,3)={}", r.unwrap());
}
