use numerica::domains::float::{
    Complex, ErrorPropagatingFloat, Float, FloatLike, Magnitude, Real, RealLike,
};

type Tracked<T> = ErrorPropagatingFloat<T>;

fn assert_same_f64(actual: f64, expected: f64) {
    if expected.is_nan() {
        assert!(actual.is_nan());
    } else {
        assert_eq!(actual.to_bits(), expected.to_bits());
    }
}

#[test]
fn square_roots_cover_both_branch_lips_and_real_axis_crossings() {
    for (re, im, re_error, im_error) in [
        (0.01, 0., 0.1, 0.),
        (-0.01, 0., 0.1, 0.),
        (-1., 0., 0.01, 1e-6),
        (-1., 0.01, 0.01, 0.02),
        (1e-100, 0., 2e-100, 0.),
    ] {
        let value = Complex::new(
            Tracked::new_with_absolute_error(re, re_error),
            Tracked::new_with_absolute_error(im, im_error),
        );
        let root = value.sqrt();
        let center = Complex::new(re, im).sqrt();
        assert_same_f64(*root.re.get_num(), center.re);
        assert_same_f64(*root.im.get_num(), center.im);
        assert!(root.re.get_error_magnitude().is_finite());
        assert!(root.im.get_error_magnitude().is_finite());
        assert!(!root.re.is_exact());
        assert!(!root.im.is_exact());
        for re_step in [-1., 0., 1.] {
            for im_step in [-1., 0., 1.] {
                let sample = Complex::new(re + re_step * re_error, im + im_step * im_error).sqrt();
                assert!((sample.re - root.re.get_num()).abs() <= root.re.get_absolute_error());
                assert!((sample.im - root.im.get_num()).abs() <= root.im.get_absolute_error());
            }
        }
    }
}

#[test]
fn square_root_branch_bounds_keep_the_full_exponent_range() {
    for precision in [128, 256, 512] {
        for text in ["1e-1000", "1e1000"] {
            let scale = Float::parse(text, Some(precision)).unwrap();
            let re_error = scale.absolute_magnitude().scale_pow2(-10);
            let im_error = scale.absolute_magnitude().scale_pow2(-20);
            let value = Complex::new(
                Tracked::new_with_error_magnitude(-scale.clone(), re_error),
                Tracked::new_with_error_magnitude(Float::with_val(precision, 0), im_error),
            );
            let root = value.sqrt();
            for component in [&root.re, &root.im] {
                assert!(component.get_error_magnitude().is_finite());
                assert!(!component.is_exact());
            }
            for sign in [-1, 1] {
                let sample_im = scale.clone() / (1 << 20) * sign;
                let sample = Complex::new(-scale.clone(), sample_im).sqrt();
                let re_difference = (sample.re - root.re.get_num()).absolute_magnitude();
                let im_difference = (sample.im - root.im.get_num()).absolute_magnitude();
                assert!(re_difference <= root.re.get_error_magnitude());
                assert!(im_difference <= root.im.get_error_magnitude());
            }
        }
    }
}

#[test]
fn square_root_preserves_resolved_real_axes_and_exact_zero() {
    for precision in [128, 256, 512] {
        let positive = Complex::new(
            Tracked::new_with_absolute_error(Float::with_val(precision, 4), 0.001),
            Tracked::<Float>::exact_zero(precision),
        )
        .sqrt();
        assert_eq!(positive.re.get_num().to_f64(), 2.);
        assert!(positive.im.is_exact_zero());
        assert!(!positive.re.is_exact());
        let exact = Complex::new(
            Tracked::<Float>::exact_zero(precision),
            Tracked::<Float>::exact_zero(precision),
        )
        .sqrt();
        assert!(exact.re.is_exact_zero());
        assert!(exact.im.is_exact_zero());
        for sign in [0., -0.] {
            let negative = Complex::new(
                Tracked::new_with_absolute_error(Float::with_val(precision, -4), 0.001),
                Tracked::new_with_absolute_error(Float::with_val(precision, sign), 0.),
            )
            .sqrt();
            assert!(negative.re.is_exact_zero());
            assert_eq!(negative.im.get_num().to_f64().abs(), 2.);
        }
    }
}

#[test]
fn finite_real_paths_retain_real_error_and_exact_imaginary_zero() {
    for precision in [128, 256, 512] {
        for error in [0.001, f64::INFINITY] {
            let re = Tracked::new_with_absolute_error(Float::with_val(precision, 2), error);
            let other =
                Tracked::new_with_absolute_error(Float::with_val(precision / 2, 1.5), 0.002);
            let x = Complex::from(re.clone());
            let y = Complex::from(other.clone());
            let mut mul_assign = x.clone();
            mul_assign *= &y;
            let mut owned_mul_assign = x.clone();
            owned_mul_assign *= y.clone();
            let mut div_assign = x.clone();
            div_assign /= &y;
            let mut owned_div_assign = x.clone();
            owned_div_assign /= y.clone();
            for (value, reference) in [
                (x.clone() * &y, re.clone() * &other),
                (mul_assign, re.clone() * &other),
                (owned_mul_assign, re.clone() * &other),
                (x.clone() / &y, re.clone() / &other),
                (div_assign, re.clone() / &other),
                (owned_div_assign, re.clone() / &other),
                (x.exp(), re.exp()),
                (x.sin(), re.sin()),
                (x.cos(), re.cos()),
            ] {
                assert_eq!(value.re.get_num(), reference.get_num());
                assert!(value.re.get_error_magnitude() >= reference.get_error_magnitude());
                assert!(value.im.is_exact_zero());
                assert!(value.re.get_error_magnitude() > Magnitude::ZERO);
            }
        }
    }
}

#[test]
fn real_shortcuts_preserve_signed_zeros_and_exceptional_multiplication() {
    for a in [
        0.,
        -0.,
        1.,
        -1.,
        f64::MIN_POSITIVE,
        f64::MAX,
        f64::INFINITY,
        f64::NAN,
    ] {
        for b in [0., -0., 2., -2., f64::MIN_POSITIVE, f64::MAX] {
            for x in [0., -0.] {
                for y in [0., -0.] {
                    let product = Complex::new(a, x) * Complex::new(b, y);
                    assert_same_f64(product.re, a * b - x * y);
                    assert_same_f64(product.im, a * y + x * b);
                    let mut assigned = Complex::new(a, x);
                    assigned *= &Complex::new(b, y);
                    assert_same_f64(assigned.re, product.re);
                    assert_same_f64(assigned.im, product.im);
                }
            }
        }
    }
    for re in [0., -0., 1., -1., 1000., -1000.] {
        for im in [0., -0.] {
            let z = Complex::new(re, im);
            let exponential = z.exp();
            assert_same_f64(exponential.re, re.exp());
            assert_same_f64(exponential.im, im);
            let sine = z.sin();
            assert_same_f64(sine.re, re.sin());
            assert_same_f64(sine.im, im * re.cos());
            let cosine = z.cos();
            assert_same_f64(cosine.re, re.cos());
            assert_same_f64(cosine.im, -im * re.sin());
        }
    }
}

#[test]
fn uncertain_imaginary_zero_does_not_take_real_shortcuts() {
    let x = Complex::new(
        Tracked::new_with_absolute_error(2., 0.001),
        Tracked::new_with_absolute_error(0., 0.002),
    );
    let y = Complex::from(Tracked::new_with_absolute_error(1.5, 0.001));
    for value in [x * y, x / y, x.exp(), x.sin(), x.cos()] {
        assert!(value.im.get_num().is_fully_zero());
        assert!(!value.im.is_exact());
    }
}
