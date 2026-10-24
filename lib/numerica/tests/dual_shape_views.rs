use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    hint::black_box,
};

use numerica::domains::dual::{DualNumberStructure, HyperDual};

#[allow(
    clippy::suspicious_arithmetic_impl,
    clippy::suspicious_op_assign_impl,
    clippy::reversed_empty_ranges
)]
mod fixed {
    numerica::create_hyperdual_from_components!(
        MixedDual,
        [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1]]
    );
    numerica::create_hyperdual_single_derivative!(ConstantDual, 0);
}

const SHAPE: [[usize; 2]; 5] = [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1]];

fn dynamic() -> HyperDual<f64> {
    HyperDual::from_values(SHAPE.iter().map(|row| row.to_vec()).collect(), vec![0.; 5])
}

fn matches_layout(value: &dyn DualNumberStructure, expected: &[&[usize]]) -> bool {
    value.get_len() == expected.len()
        && expected
            .iter()
            .enumerate()
            .all(|(index, &row)| value.get_shape_component(index) == Some(row))
        && value.get_shape_component(expected.len()).is_none()
}

#[test]
fn borrowed_components_match_complete_static_and_dynamic_shapes() {
    let fixed = fixed::MixedDual { values: [0.; 5] };
    let dynamic = dynamic();
    let cloned = dynamic.clone();
    let expected = SHAPE.map(|row| row.to_vec());
    let expected: Vec<_> = expected.iter().map(Vec::as_slice).collect();
    for value in [&fixed as &dyn DualNumberStructure, &dynamic, &cloned] {
        assert!(matches_layout(value, &expected));
        for (index, row) in value.get_shape().into_iter().enumerate() {
            assert_eq!(value.get_shape_component(index), Some(row));
        }
        assert_eq!(value.get_shape_component(SHAPE.len()), None);
        assert_eq!(value.get_shape_component(usize::MAX), None);
    }
    let constant = fixed::ConstantDual { values: [1.] };
    let dynamic_constant = HyperDual::from_values(vec![vec![]], vec![1.]);
    for value in [&constant as &dyn DualNumberStructure, &dynamic_constant] {
        assert_eq!(value.get_shape_component(0), Some(&[][..]));
        assert_eq!(value.get_shape_component(1), None);
    }
}

#[test]
fn complete_validation_rejects_reordered_higher_mixed_and_malformed_layouts() {
    let value = dynamic();
    let expected = value.get_shape();
    let mut reordered = SHAPE;
    reordered.swap(3, 4);
    let mut other_derivative = SHAPE;
    other_derivative[3] = [0, 2];
    for shape in [reordered, other_derivative] {
        let other =
            HyperDual::from_values(shape.iter().map(|row| row.to_vec()).collect(), vec![0.; 5]);
        assert!(!matches_layout(&other, &expected));
    }
    let other_dimension =
        HyperDual::from_values((0..5).map(|order| vec![order]).collect(), vec![0.; 5]);
    assert!(!matches_layout(&other_dimension, &expected));

    let mut shortened = dynamic();
    shortened.values.truncate(4);
    assert_eq!(shortened.get_len(), 4);
    assert_eq!(shortened.get_shape_component(4), Some(&[1, 1][..]));
    assert!(!matches_layout(&shortened, &expected));
    // Matching just the shortened value count and shape prefix would accept
    // this invalid carrier; checking the next shape component rejects it.
    assert!(!matches_layout(&shortened, &expected[..4]));

    let mut extended = dynamic();
    extended.values.push(0.);
    assert_eq!(extended.get_len(), 6);
    assert_eq!(extended.get_shape_component(5), None);
    assert!(!matches_layout(&extended, &expected));
}

/// Deliberately implements only the original trait methods.
struct LegacyShape(Vec<Vec<usize>>);
impl DualNumberStructure for LegacyShape {
    fn get_len(&self) -> usize {
        self.0.len()
    }
    fn get_shape(&self) -> Vec<&[usize]> {
        self.0.iter().map(Vec::as_slice).collect()
    }
    fn get_multiplication_table(&self) -> &[(usize, usize, usize)] {
        &[]
    }
    fn get_max_depth(&self) -> usize {
        0
    }
}

#[test]
fn default_accessor_preserves_existing_implementations_and_trait_objects() {
    let legacy = LegacyShape(vec![vec![0, 0], vec![1, 0]]);
    let object: &dyn DualNumberStructure = &legacy;
    let row = object.get_shape_component(1).unwrap();
    assert_eq!(row, &[1, 0]);
    assert!(std::ptr::eq(row.as_ptr(), legacy.0[1].as_ptr()));
    assert_eq!(object.get_shape_component(2), None);
    assert_eq!(object.get_shape_component(usize::MAX), None);
}

struct CountingAllocator;
thread_local! {
    // Count only the measured test thread, independently of parallel tests.
    static ALLOCATIONS: Cell<Option<usize>> = const { Cell::new(None) };
}
fn record_allocation() {
    let _ = ALLOCATIONS.try_with(|count| {
        if let Some(value) = count.get() {
            count.set(Some(value + 1));
        }
    });
}

// The test allocator forwards every request unchanged to System. Counting uses
// constant-initialized, non-allocating TLS and does not affect other threads.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc(layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        record_allocation();
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        record_allocation();
        unsafe { System.realloc(pointer, layout, size) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) }
    }
}
#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn builtin_component_access_and_complete_validation_allocate_nothing() {
    let fixed = fixed::MixedDual { values: [0.; 5] };
    let dynamic = dynamic();
    let expected = SHAPE.map(|row| row.to_vec());
    let expected: Vec<_> = expected.iter().map(Vec::as_slice).collect();
    // Positive control: the old accessor materializes a vector of references.
    ALLOCATIONS.with(|count| count.set(Some(0)));
    let owned = black_box(&dynamic as &dyn DualNumberStructure).get_shape();
    black_box(&owned);
    let allocations = ALLOCATIONS.with(|count| count.replace(None)).unwrap();
    assert!(allocations > 0);

    ALLOCATIONS.with(|count| count.set(Some(0)));
    for _ in 0..1024 {
        for value in [&fixed as &dyn DualNumberStructure, &dynamic] {
            assert!(matches_layout(black_box(value), black_box(&expected)));
            assert_eq!(black_box(value).get_shape_component(usize::MAX), None);
        }
    }
    let allocations = ALLOCATIONS.with(|count| count.replace(None)).unwrap();
    assert_eq!(allocations, 0);
}
