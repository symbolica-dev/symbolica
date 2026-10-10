use numerica::{
    domains::rational::{Q, Rational},
    tensors::matrix::{Matrix, MatrixError},
};

fn matrix(entries: &[i64], dimension: u32) -> Matrix<Q> {
    Matrix::from_linear(
        entries.iter().copied().map(Rational::from).collect(),
        dimension,
        dimension,
        Q,
    )
    .unwrap()
}

#[test]
fn in_place_determinant_preserves_odd_row_swap_sign() {
    let mut m = matrix(&[0, 1, 1, 0], 2);
    assert_eq!(m.det().unwrap(), Rational::from(-1));
    assert_eq!(m.det_in_place().unwrap(), Rational::from(-1));
}

#[test]
fn in_place_determinant_preserves_even_row_swap_sign() {
    let mut m = matrix(&[0, 1, 0, 0, 0, 1, 1, 0, 0], 3);
    assert_eq!(m.det().unwrap(), Rational::from(1));
    assert_eq!(m.det_in_place().unwrap(), Rational::from(1));
}

#[test]
fn in_place_determinant_counts_a_late_row_swap() {
    let mut m = matrix(&[2, 3, 4, 0, 0, 5, 0, 7, 8], 3);
    assert_eq!(m.det().unwrap(), Rational::from(-70));
    assert_eq!(m.det_in_place().unwrap(), Rational::from(-70));
}

#[test]
fn in_place_determinant_preserves_singular_and_small_cases() {
    for (entries, dimension, expected) in [
        (vec![], 0, 1),
        (vec![-3], 1, -3),
        (vec![2, 3, 0, 5], 2, 10),
        (vec![0, 1, 2, 1, 0, 0, 2, 0, 0], 3, 0),
    ] {
        let mut m = matrix(&entries, dimension);
        assert_eq!(m.det().unwrap(), Rational::from(expected));
        assert_eq!(m.det_in_place().unwrap(), Rational::from(expected));
    }
    assert!(matches!(
        Matrix::new(2, 3, Q).det_in_place(),
        Err(MatrixError::NotSquare)
    ));
}

#[test]
fn row_reduction_and_solve_keep_their_public_contract() {
    let original = matrix(&[0, 2, 1, 3], 2);
    let mut reduced = original.clone();
    assert_eq!(reduced.partial_row_reduce(2), 2);
    assert_eq!(reduced, matrix(&[1, 3, 0, 2], 2));
    let rhs = Matrix::new_vec(vec![4.into(), 7.into()], Q);
    assert_eq!(original.solve(&rhs).unwrap().into_vec(), [1, 2]);
    let mut rectangular = Matrix::from_nested_vec(
        vec![
            vec![0.into(), 1.into(), 2.into()],
            vec![1.into(), 0.into(), 3.into()],
        ],
        Q,
    )
    .unwrap();
    assert_eq!(rectangular.partial_row_reduce(2), 2);
    assert_eq!(rectangular[(0, 0)], Rational::from(1));
    assert_eq!(rectangular[(1, 1)], Rational::from(1));
}
