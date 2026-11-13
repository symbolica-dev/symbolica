use numerica::{
    create_hyperdual_from_components,
    domains::dual::{DualNumber, HyperDual},
};

create_hyperdual_from_components!(StaticDual, [[0], [1]]);

// Borrowing components must not require arithmetic, cloning, or default values.
struct Component(u8);

#[test]
fn component_views_support_non_numeric_values_and_trait_objects() {
    let mut fixed = StaticDual {
        values: [Component(2), Component(3)],
    };
    let mut dynamic =
        HyperDual::from_values(vec![vec![0], vec![1]], vec![Component(5), Component(7)]);

    for dual in [&mut fixed as &mut dyn DualNumber<Component>, &mut dynamic] {
        assert_eq!(dual.values().len(), dual.get_len());
        dual.values_mut()[0].0 += 1;
    }

    assert_eq!(fixed.values[0].0, 3);
    assert_eq!(fixed.values[1].0, 3);
    assert_eq!(dynamic.values[0].0, 6);
    assert_eq!(dynamic.values[1].0, 7);
}
