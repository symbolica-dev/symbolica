//! Run with `cargo run --example linear_combination`.

use symbolica::{linear_combination::LinearCombination, prelude::*};

fn main() {
    // Values need only an ordering: duplicate values combine and zero terms disappear.
    let a = LinearCombination::from_terms(
        vec![
            ('z', Rational::from(1)),
            ('x', (1, 2).into()),
            ('y', 2.into()),
            ('x', (1, 2).into()),
        ],
        Q,
    );
    assert_eq!(
        a.terms(),
        &[('x', 1.into()), ('y', 2.into()), ('z', 1.into())]
    );

    let b = LinearCombination::from_terms(vec![('y', (-2).into()), ('w', 3.into())], Q);
    let added = &a + &b;
    assert_eq!(
        added.terms(),
        &[('w', 3.into()), ('x', 1.into()), ('z', 1.into())]
    );

    // Merge several sorted combinations in one pass over their terms.
    let c = LinearCombination::from_terms(vec![('z', (-1).into())], Q);
    let combined = LinearCombination::sum_ref([&a, &b, &c], Q);
    assert_eq!(combined.terms(), &[('w', 3.into()), ('x', 1.into())]);

    // Replace x by 2*y - 3*w. Existing w terms cancel after substitution.
    let replacement = LinearCombination::from_terms(vec![('y', 2.into()), ('w', (-3).into())], Q);
    let substituted = combined.substitute(&'x', &replacement);
    assert_eq!(substituted.terms(), &[('y', 2.into())]);

    // The same container works over finite fields, with arithmetic in the chosen domain.
    let field = Zp::new(5);
    let a = LinearCombination::from_terms(
        vec![('y', field.to_element(4)), ('x', field.to_element(3))],
        field.clone(),
    );
    let b = LinearCombination::from_terms(
        vec![
            ('z', field.to_element(1)),
            ('y', field.to_element(2)),
            ('x', field.to_element(2)),
        ],
        field.clone(),
    );
    let combined = LinearCombination::sum([a, b], field.clone());
    assert_eq!(
        combined.terms(),
        &[('y', field.to_element(1)), ('z', field.to_element(1))]
    );
}
