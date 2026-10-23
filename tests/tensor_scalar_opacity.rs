use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use symbolica::{
    atom::{Atom, AtomCore, AtomView, Symbol},
    function, symbol,
};

#[test]
fn scalar_function_payload_is_not_a_tensor_index_domain() {
    let index = function!(symbol!("tensor_scalar_opacity::slot"), 710);
    let private = function!(symbol!("tensor_scalar_opacity::a"), &index)
        * function!(symbol!("tensor_scalar_opacity::b"), &index);
    let metadata = symbol!("tensor_scalar_opacity::metadata"; Scalar);
    for payload in [
        private.clone(),
        private.clone().pow(2),
        private.clone().pow(-1),
        private.pow((1, 2)),
    ] {
        let value = function!(metadata, payload);
        let canonical = value.canonize_tensors([(index.clone(), 0)]).unwrap();
        assert_eq!(canonical.canonical_form, value);
        assert!(canonical.external_indices.is_empty());
        assert!(canonical.dummy_indices.is_empty());
        assert_eq!(
            canonical
                .canonical_form
                .canonize_tensors([(index.clone(), 0)])
                .unwrap()
                .canonical_form,
            value
        );
    }
}

#[test]
fn tensor_metadata_scalar_payload_is_literal_and_never_replayed_as_changed() {
    let index = function!(symbol!("tensor_metadata_opacity::slot"), 711);
    let metadata = function!(
        symbol!("tensor_metadata_opacity::metadata"; Scalar),
        function!(symbol!("tensor_metadata_opacity::a"), &index)
            * function!(symbol!("tensor_metadata_opacity::b"), &index)
    );
    let altered = Arc::new(AtomicUsize::new(0));
    let observed = altered.clone();
    let expected = metadata.clone();
    let head = symbol!(
        "tensor_metadata_opacity::T",
        norm = move |value, output| {
            if !matches!(value, AtomView::Fun(fun) if fun.iter().next() == Some(expected.as_view()))
            {
                observed.fetch_add(1, Ordering::SeqCst);
                output.to_num(0);
            }
        }
    );
    let source = function!(head, &metadata, &index)
        * function!(symbol!("tensor_metadata_opacity::c"), &index);
    assert_eq!(altered.load(Ordering::SeqCst), 0);
    let canonical = source.canonize_tensors([(index.clone(), 0)]).unwrap();
    assert!(!canonical.canonical_form.is_zero());
    assert!(canonical.canonical_form.contains(&metadata));
    assert_eq!(canonical.dummy_indices, vec![(index.clone(), 0)]);
    assert_eq!(altered.load(Ordering::SeqCst), 0);
    assert_eq!(
        canonical
            .canonical_form
            .canonize_tensors([(index, 0)])
            .unwrap()
            .canonical_form,
        canonical.canonical_form
    );
    assert_eq!(altered.load(Ordering::SeqCst), 0);
}

#[test]
fn untagged_functions_keep_tensor_index_traversal() {
    let index = function!(symbol!("tensor_scalar_unmarked::slot"), 712);
    let product = function!(symbol!("tensor_scalar_unmarked::a"), &index)
        * function!(symbol!("tensor_scalar_unmarked::b"), &index);
    for head in [symbol!("tensor_scalar_unmarked::metadata"), Symbol::COS] {
        assert!(!head.is_scalar());
        let metadata = function!(head, &product);
        let outer = function!(symbol!("tensor_scalar_unmarked::T"), &metadata, &index)
            * function!(symbol!("tensor_scalar_unmarked::c"), &index);
        let error = outer.canonize_tensors([(index.clone(), 0)]).unwrap_err();
        assert!(
            error.to_string().contains("contracted more than once"),
            "{error}"
        );
    }
}

#[test]
fn scalar_variables_remain_available_as_actual_indices() {
    let index = Atom::var(symbol!("tensor_scalar_actual_index::i"; Scalar));
    let source = function!(symbol!("tensor_scalar_actual_index::a"), &index)
        * function!(symbol!("tensor_scalar_actual_index::b"), &index);
    let canonical = source.canonize_tensors([(index.clone(), 0)]).unwrap();
    assert_eq!(canonical.canonical_form, source);
    assert_eq!(canonical.dummy_indices, vec![(index, 0)]);
    assert!(canonical.external_indices.is_empty());
}

#[test]
fn opaque_scalar_coefficient_preserves_tensor_antisymmetry_cancellation() {
    let indices = [0, 1, 2, 3].map(|i| function!(symbol!("tensor_scalar_sign::slot"), i));
    let [i, j, k, l] = &indices;
    let antisymmetric = symbol!("tensor_scalar_sign::A"; Antisymmetric);
    let ordinary = symbol!("tensor_scalar_sign::B");
    let metadata = function!(
        symbol!("tensor_scalar_sign::metadata"; Scalar),
        function!(symbol!("tensor_scalar_sign::x"), i)
            * function!(symbol!("tensor_scalar_sign::y"), i)
    );
    let source = metadata
        * (function!(antisymmetric, i, j) * function!(ordinary, i, j)
            + function!(antisymmetric, l, k) * function!(ordinary, k, l));
    assert!(!source.is_zero());
    let canonical = source
        .canonize_tensors(indices.into_iter().map(|i| (i, 0)))
        .unwrap();
    assert!(canonical.canonical_form.is_zero());
}
