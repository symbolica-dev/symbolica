use std::panic::{AssertUnwindSafe, catch_unwind};

use numerica::{
    create_hyperdual_single_derivative,
    domains::{
        Ring,
        dual::{DualNumberStructure, HyperDual},
        finite_field::{FiniteField, FiniteFieldCore, Mersenne32, Mersenne64, Z2, Zp, Zp64},
        float::{
            Complex, DoubleFloat, ErrorPropagatingFloat, F64, Float, FloatLike, Real, RealBall,
            RealLike, SingleFloat,
        },
        integer::{Integer, MultiPrecisionInteger, Z},
        rational::{Q, Rational},
    },
    numerical_integration::{ContinuousGrid, DiscreteGrid, Grid, MonteCarloRng, Sample},
    tensors::{
        matrix::{Matrix, MatrixError, Vector},
        sparse::SparseMatrix,
    },
};

#[test]
fn nan_preserves_precision_and_components() {
    assert!(0.0_f64.nan().unwrap().is_nan());
    assert!(F64(0.0).nan().unwrap().to_f64().is_nan());
    assert!(DoubleFloat::from(0.0).nan().unwrap().to_f64().is_nan());
    assert!(Rational::from(1).nan().is_none());
    for precision in [53, 100, 256] {
        let nan = Float::new(precision).nan().unwrap();
        assert_eq!(nan.prec(), precision);
        assert!(nan.to_f64().is_nan());
    }
    let complex = Complex::new(Float::new(80), Float::new(120)).nan().unwrap();
    assert_eq!((complex.re.prec(), complex.im.prec()), (80, 120));
    assert!(complex.re.to_f64().is_nan() && complex.im.to_f64().is_nan());
    assert!(
        Complex::new(Rational::from(1), Rational::from(0))
            .nan()
            .is_none()
    );
    let ball = RealBall::new(Float::new(80), Float::new(120))
        .nan()
        .unwrap();
    assert_eq!((ball.center.prec(), ball.radius.prec()), (80, 120));
    assert!(!ball.is_finite());
    let tracked = ErrorPropagatingFloat::new(Float::new(100), 20.0)
        .nan()
        .unwrap();
    assert!(tracked.get_num().to_f64().is_nan());
    assert!(tracked.get_absolute_error().is_nan());
    assert_eq!(tracked.get_num().prec(), 100);
    for v in wide::f64x4::splat(1.0).nan().unwrap().to_array() {
        assert!(v.is_nan());
    }
}

create_hyperdual_single_derivative!(TestDual, 2);

#[test]
fn dual_nan_preserves_shape() {
    let shape = vec![vec![0], vec![1]];
    let dual = HyperDual::from_values(shape, vec![Float::new(80), Float::new(120)]);
    let nan = dual.nan().unwrap();
    assert_eq!(nan.get_shape(), dual.get_shape());
    assert_eq!(
        nan.values.iter().map(Float::prec).collect::<Vec<_>>(),
        [80, 120]
    );
    assert!(nan.values.iter().all(|v| v.to_f64().is_nan()));
    let fixed = TestDual::<f64>::new_variable(0, 1.0).nan().unwrap();
    assert!(fixed.values.iter().all(|v| v.is_nan()));
    assert!(
        TestDual::<Rational>::new_variable(0, 1.into())
            .nan()
            .is_none()
    );
}

#[test]
fn parse_counts_significant_digits_independently_of_notation() {
    for s in [
        "123456789012345678901234567890",
        "-123456789012345678901234567890",
        "+123456789012345678901234567890",
        "1.23456789012345678901234567890e29",
        "-0.000123456789012345678901234567890E33",
        "  123456789012345678901234567890  ",
        "123456789012345678901234567890`",
    ] {
        let value = Float::parse(s, None).unwrap();
        assert_eq!(value.prec(), 100, "{s}");
        let expected: Rational = "123456789012345678901234567890"
            .parse::<Integer>()
            .unwrap()
            .into();
        let expected = if s.trim_start().starts_with('-') {
            -expected
        } else {
            expected
        };
        assert_eq!(value.try_to_rational(), Some(expected), "{s}");
    }
    for s in ["0", "-0.0000", "1e100", "1.25E-100", "NaN"] {
        assert_eq!(Float::parse(s, None).unwrap().prec(), 53, "{s}");
    }
    assert!(Float::parse("NaN", Some(80)).unwrap().to_f64().is_nan());
    assert_eq!(
        Float::parse("-Infinity", None).unwrap().to_f64(),
        f64::NEG_INFINITY
    );
    assert_eq!(Float::parse("1.25`40", None).unwrap().prec(), 133);
    assert_eq!(Float::parse("1.25`40", Some(80)).unwrap().prec(), 80);
    for s in [
        "", "-", "1e", "1.2.3", "1`0", "1`-1", "1`NaN", "1`inf", "1`1`2",
    ] {
        assert!(Float::parse(s, None).is_err(), "{s}");
        assert!(Float::parse(s, Some(80)).is_err(), "{s}");
    }
    assert!(Float::parse("1", Some(0)).is_err());
}

fn check_zero_powers<F: Ring>(field: F, period: u64) {
    assert_eq!(field.pow(&field.zero(), 0), field.one());
    for exponent in [1, 2, period, period.saturating_mul(2), u64::MAX] {
        assert_eq!(field.pow(&field.zero(), exponent), field.zero());
        assert_eq!(field.pow(&field.one(), exponent), field.one());
    }
}

#[test]
fn zero_powers_are_correct_in_all_finite_field_representations() {
    check_zero_powers(Zp::new(7), 6);
    check_zero_powers(Zp64::new(7), 6);
    check_zero_powers(Zp::new_non_prime(9), 8);
    check_zero_powers(
        FiniteField::<Mersenne32>::new(Mersenne32::new()),
        (1 << 31) - 2,
    );
    check_zero_powers(
        FiniteField::<Mersenne64>::new(Mersenne64::new()),
        (1 << 61) - 2,
    );
    check_zero_powers(Z2, 1);
    check_zero_powers(FiniteField::<Integer>::new_non_prime(7.into()), 6);
    check_zero_powers(
        FiniteField::<MultiPrecisionInteger>::new_non_prime(7.into()),
        6,
    );
}

#[test]
fn rectangular_matrix_indexing_checks_both_coordinates() {
    let mut matrix = Matrix::from_linear((1..=6).map(Integer::from).collect(), 2, 3, Z).unwrap();
    assert_eq!(matrix[0], [1, 2, 3]);
    assert_eq!(matrix[1], [4, 5, 6]);
    matrix[(1, 2)] = 9.into();
    assert_eq!(matrix[(1, 2)], 9);
    for index in [(0, 3), (2, 0), (u32::MAX, u32::MAX)] {
        assert!(catch_unwind(|| &matrix[index]).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| matrix[index] = 0.into())).is_err());
    }
    assert!(catch_unwind(|| &matrix[2]).is_err());
    assert!(Matrix::from_linear(Vec::new(), 65536, 65536, Z).is_err());
}

#[test]
fn empty_matrix_shapes_work_in_iteration_and_algebra() {
    let empty = Matrix::from_nested_vec(vec![], Q).unwrap();
    assert_eq!((empty.nrows(), empty.ncols()), (0, 0));
    assert_eq!(empty.det().unwrap(), Rational::from(1));
    assert_eq!(empty.inv().unwrap(), empty);
    let rows = Matrix::from_nested_vec(vec![vec![], vec![]], Q).unwrap();
    assert_eq!((rows.nrows(), rows.ncols()), (2, 0));
    assert_eq!(rows.row_iter().map(<[_]>::len).collect::<Vec<_>>(), [0, 0]);
    assert_eq!(rows.row_iter().rev().count(), 2);
    assert_eq!(rows[1].len(), 0);
    assert!(catch_unwind(|| &rows[2]).is_err());
    assert_eq!(rows.transpose().rank(), 0);
    assert_eq!((&rows * &Matrix::new(0, 3, Q)), Matrix::new(2, 3, Q));
    assert_eq!(rows.to_sparse().to_dense(), rows);
    let unconstrained = rows.transpose();
    assert_eq!(
        unconstrained.solve_any(&Matrix::new(0, 1, Q)).unwrap(),
        Matrix::new(2, 1, Q)
    );
}

#[test]
fn matrix_column_splitting_handles_every_boundary() {
    let matrix = Matrix::from_linear((1..=6).map(Integer::from).collect(), 2, 3, Z).unwrap();
    for split in 0..=3 {
        let (left, right) = matrix.split_col(split).unwrap();
        assert_eq!(left.ncols(), split as usize);
        assert_eq!(left.augment(&right).unwrap(), matrix);
    }
    assert!(matrix.split_col(4).is_err());
    let empty = Matrix::new(2, 0, Z);
    let (left, right) = empty.split_col(0).unwrap();
    assert_eq!(left.augment(&right).unwrap(), empty);
}

#[test]
fn matrix_and_vector_operations_reject_mixed_fields() {
    let f5 = Zp::new(5);
    let f7 = Zp::new(7);
    let a = Matrix::new_vec(vec![f5.one()], f5.clone());
    let b = Matrix::new_vec(vec![f7.one()], f7.clone());
    assert!(catch_unwind(|| &a + &b).is_err());
    assert!(catch_unwind(|| &a - &b).is_err());
    assert!(catch_unwind(|| &a * &b).is_err());
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let mut m = a.clone();
            m += &b;
        }))
        .is_err()
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let mut m = a.clone();
            m -= &b;
        }))
        .is_err()
    );
    assert!(matches!(a.augment(&b), Err(MatrixError::FieldMismatch)));
    assert!(matches!(a.solve(&b), Err(MatrixError::FieldMismatch)));
    let a = Vector::new(vec![f5.one()], f5);
    let b = Vector::new(vec![f7.one()], f7);
    assert!(catch_unwind(|| a.dot(&b)).is_err());
    assert!(catch_unwind(|| &a + &b).is_err());
}

#[test]
fn csr_validation_rejects_invalid_structure() {
    for (rows, cols) in [
        (vec![0, 99], vec![99]),
        (vec![1, 1], vec![0]),
        (vec![0, 0], vec![0]),
        (vec![0, 1], vec![1]),
        (vec![0], vec![0]),
    ] {
        assert!(SparseMatrix::try_from_csr(1, 1, vec![1.into()], rows, cols, Z).is_err());
    }
    assert!(SparseMatrix::try_from_csr(2, 2, vec![1.into()], vec![0, 2, 1], vec![0], Z).is_err());
    for cols in [vec![1, 0], vec![0, 0]] {
        assert!(
            SparseMatrix::try_from_csr(1, 2, vec![1.into(), 2.into()], vec![0, 2], cols, Z)
                .is_err()
        );
    }
    let valid =
        SparseMatrix::try_from_csr(2, 3, vec![2.into(), 3.into()], vec![0, 0, 2], vec![0, 2], Z)
            .unwrap();
    assert_eq!(valid.to_dense()[1], [2, 0, 3]);
    assert!(
        catch_unwind(|| SparseMatrix::from_csr(1, 1, vec![1.into()], vec![0, 99], vec![99], Z))
            .is_err()
    );
    assert!(catch_unwind(|| SparseMatrix::from_triplets(1, 1, vec![(1, 0, 1.into())], Z)).is_err());
}

#[test]
fn empty_grids_and_invalid_bin_evolution_return_errors() {
    assert!(DiscreteGrid::<f64>::new(vec![], 100.0, false).is_err());
    for (dims, bins) in [(0, 10), (1, 0), (0, 0)] {
        assert!(ContinuousGrid::<f64>::new(dims, bins, 100, None, false).is_err());
        assert!(
            ContinuousGrid::<f64>::new_with_min_probability_density(
                dims, bins, 100, None, false, 0.1
            )
            .is_err()
        );
    }
    for evolution in [vec![], vec![0], vec![10, 0]] {
        assert!(ContinuousGrid::<f64>::new(1, 10, 100, Some(evolution), false).is_err());
    }
    assert!(ContinuousGrid::<f64>::new(1, 1, 100, Some(vec![1, 2]), false).is_ok());
}

#[test]
fn sample_buffers_can_be_reused_across_grid_variants() {
    let mut rng = MonteCarloRng::new(0, 0);
    let mut sample = Sample::new();
    let mut discrete = DiscreteGrid::new(vec![None], 100.0, false).unwrap();
    let mut continuous = ContinuousGrid::new(1, 1, 1, None, false).unwrap();
    let mut uniform = Grid::Uniform(vec![2], continuous.clone());
    for _ in 0..3 {
        discrete.sample(&mut rng, &mut sample);
        continuous.sample(&mut rng, &mut sample);
        assert!(matches!(&sample, Sample::Continuous(w, xs) if *w == 1.0 && xs.len() == 1));
        uniform.sample(&mut rng, &mut sample);
        assert!(
            matches!(&sample, Sample::Uniform(w, ds, xs) if *w == 2.0 && ds.len() == 1 && xs.len() == 1)
        );
    }
}

#[test]
fn exact_zero_does_not_destroy_significant_digits() {
    let tiny = Float::parse("1e-60", Some(200)).unwrap();
    let result = tiny.clone() + Float::new(200);
    assert_eq!(result.prec(), tiny.prec());
    assert_eq!(result.to_rational(), tiny.to_rational());
}

#[test]
fn real_powers_handle_exact_dyadic_roots() {
    for (base, exponent, expected) in [
        (4, "0.5", "2"),
        (16, "0.25", "2"),
        (16, "-0.25", "0.5"),
        (4, "1.5", "8"),
    ] {
        let result = Float::with_val(200, base).powf(&Float::parse(exponent, Some(200)).unwrap());
        assert_eq!(
            result.to_rational(),
            Float::parse(expected, Some(200)).unwrap().to_rational()
        );
    }
}

#[test]
fn atan2_preserves_signed_zero_quadrants_and_infinities() {
    for y in [0.0_f64, -0.0, 1.0, -1.0, f64::INFINITY, f64::NEG_INFINITY] {
        for x in [0.0_f64, -0.0, 1.0, -1.0, f64::INFINITY, f64::NEG_INFINITY] {
            let angle = Float::with_val(160, y)
                .atan2(&Float::with_val(160, x))
                .to_f64();
            assert!(
                (angle - y.atan2(x)).abs() < 1e-15,
                "atan2({y}, {x}) = {angle}"
            );
            assert_eq!(angle.is_sign_negative(), y.is_sign_negative());
        }
    }
    assert!(
        Float::with_val(160, 1)
            .atan2(&Float::with_val(160, f64::NAN))
            .to_f64()
            .is_nan()
    );
}
