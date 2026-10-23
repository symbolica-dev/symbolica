use symbolica::{
    domains::float::{FloatField, FloatLike},
    prelude::*,
};

#[test]
fn newton_roots_are_independent_of_equation_scale() {
    let tolerance = Float::with_val(256, 1e-60);
    for scale in ["1", "10^40", "10^-40", "-10^40", "10^-100"] {
        let expression = parse!(&format!("{scale}*(x^3-2)"));
        let root = expression
            .nsolve(
                symbol!("x"),
                Float::with_val(256, 1),
                tolerance.clone(),
                100,
            )
            .unwrap_or_else(|error| panic!("scale {scale}: {error:?}"));
        let residual = root.clone() * &root * &root - Float::with_val(256, 2);
        assert!(
            residual.norm() < Float::with_val(256, 1e-59),
            "scale {scale}"
        );
    }
}

#[test]
fn scaled_newton_roots_respect_iteration_limit() {
    for scale in ["1", "10^40", "10^-100"] {
        let expression = parse!(&format!("{scale}*(x^3-2)"));
        assert!(
            expression
                .nsolve(
                    symbol!("x"),
                    Float::with_val(256, 1),
                    Float::with_val(256, 1e-60),
                    1
                )
                .is_err()
        );
    }
}

#[test]
fn newton_system_roots_are_independent_of_equation_scale() {
    for (first, second) in [("1", "1"), ("10^40", "10^-40"), ("10^-100", "10^-100")] {
        let system = [
            parse!(&format!("{first}*(x^3-2)")),
            parse!(&format!("{second}*(y^2-x)")),
        ];
        let roots = Atom::nsolve_system(
            &system,
            &[symbol!("x").into(), symbol!("y").into()],
            &[Float::with_val(256, 1), Float::with_val(256, 1)],
            Float::with_val(256, 1e-60),
            100,
        )
        .unwrap_or_else(|error| panic!("scales {first}, {second}: {error:?}"));
        let residual = roots[0].clone() * &roots[0] * &roots[0] - Float::with_val(256, 2);
        assert!(residual.norm() < Float::with_val(256, 1e-59));
        assert!((roots[1].clone() * &roots[1] - &roots[0]).norm() < Float::with_val(256, 1e-59));
    }
}

#[test]
fn aberth_roots_are_independent_of_equation_scale() {
    let tolerance = Float::with_val(256, 1e-60);
    for scale in ["1", "10^40", "10^-40", "-10^40", "10^-100"] {
        let polynomial = parse!(&format!("{scale}*(x^3-2)"))
            .to_polynomial::<_, u16>(&Q, None)
            .to_univariate_from_univariate(0)
            .map_coeff(
                |c| Complex::from(c.to_multi_prec_float(256)),
                FloatField::from_rep(Complex::from(tolerance.clone())),
            );
        let roots = polynomial
            .roots(100, &tolerance)
            .unwrap_or_else(|_| panic!("scale {scale}: no convergence"));
        assert_eq!(roots.len(), 3);
        for root in &roots {
            let residual = root.clone() * root * root - Complex::from(Float::with_val(256, 2));
            assert!(
                residual.norm_squared() < Float::with_val(256, 1e-118),
                "scale {scale}"
            );
        }
        for i in 0..roots.len() {
            for j in 0..i {
                assert!((roots[i].clone() - &roots[j]).norm_squared() > Float::with_val(256, 1));
            }
        }
    }
}

#[test]
fn newton_accepts_accurate_initial_guesses() {
    for scale in ["1", "10^40", "10^-100"] {
        let expression = parse!(&format!("{scale}*(x^3-2)"));
        let root = expression
            .nsolve(
                symbol!("x"),
                Float::with_val(256, 1),
                Float::with_val(256, 1e-60),
                100,
            )
            .unwrap();
        assert!(
            expression
                .nsolve(symbol!("x"), root, Float::with_val(256, 1e-60), 2)
                .is_ok()
        );
        assert!(
            parse!(&format!("{scale}*(x-1)^2"))
                .nsolve(
                    symbol!("x"),
                    Float::with_val(256, 1),
                    Float::with_val(256, 1e-60),
                    1
                )
                .is_ok()
        );
    }
}

#[test]
fn small_aberth_steps_require_small_backward_error() {
    let tolerance = Float::with_val(256, 1e-60);
    let polynomial = parse!("x^3-10^-240")
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0)
        .map_coeff(
            |c| Complex::from(c.to_multi_prec_float(256)),
            FloatField::from_rep(Complex::from(tolerance.clone())),
        );
    let guesses = [(1e-70, 0.0), (0.0, 1e-70), (-1e-70, -1e-70)]
        .map(|(re, im)| Complex::new(Float::with_val(256, re), Float::with_val(256, im)))
        .to_vec();
    assert!(
        polynomial
            .roots_hot_start(1, &tolerance, guesses.clone())
            .is_err()
    );
    let roots = polynomial
        .roots_hot_start(200, &tolerance, guesses)
        .unwrap();
    let radius = Complex::from(
        parse!("10^-80")
            .to_polynomial::<_, u16>(&Q, None)
            .get_constant()
            .to_multi_prec_float(256),
    );
    for root in roots {
        let scaled = root / &radius;
        let residual = scaled.clone() * &scaled * &scaled - scaled.one();
        assert!(residual.norm().re < Float::with_val(256, 1e-59));
    }
}

#[test]
fn aberth_does_not_square_tiny_tolerances() {
    // Squaring this representable tolerance underflows in double precision.
    let tolerance = F64::from(1e-200);
    let polynomial = parse!("x-1")
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0)
        .map_coeff(
            |c| Complex::from(F64::from(c.to_f64())),
            FloatField::from_rep(Complex::from(tolerance)),
        );
    let roots = polynomial.roots(10, &tolerance).unwrap();
    assert_eq!(roots, vec![Complex::from(F64::from(1.0))]);
}

#[test]
fn newton_accepts_exact_roots_with_large_finite_coefficients() {
    for scale in ["1", "10^308"] {
        let expression = parse!(&format!("{scale}*(x-1)"));
        let root = expression
            .nsolve(symbol!("x"), F64::from(1.0), F64::from(1e-10), 10)
            .unwrap();
        assert_eq!(root, F64::from(1.0));
        let roots = Atom::nsolve_system(
            &[expression, parse!("y-1")],
            &[symbol!("x").into(), symbol!("y").into()],
            &[F64::from(1.0), F64::from(1.0)],
            F64::from(1e-10),
            10,
        )
        .unwrap();
        assert_eq!(roots, vec![F64::from(1.0); 2]);
    }
}

#[test]
fn aberth_backward_error_does_not_overflow() {
    // At these guesses the polynomial and derivative are finite, but the
    // unscaled sum of absolute monomial values exceeds f64::MAX.
    for (expression, guesses) in [
        ("10^308*(x-1)", vec![1.0]),
        ("6*10^307*(x^2-2)", vec![-2.0f64.sqrt(), 2.0f64.sqrt()]),
    ] {
        let polynomial = parse!(expression)
            .to_polynomial::<_, u16>(&Q, None)
            .to_univariate_from_univariate(0)
            .map_coeff(
                |c| Complex::from(F64::from(c.to_f64())),
                FloatField::from_rep(Complex::from(F64::from(1.0))),
            );
        let guesses = guesses
            .into_iter()
            .map(|x| Complex::from(F64::from(x)))
            .collect::<Vec<_>>();
        let roots = polynomial
            .roots_hot_start(10, &F64::from(1e-10), guesses.clone())
            .unwrap();
        for (root, expected) in roots.iter().zip(guesses) {
            assert!((root.clone() - expected).norm().re < F64::from(1e-10));
        }
    }
}

#[test]
fn aberth_preserves_small_coefficients_in_backward_error() {
    let polynomial = parse!("6*10^307*x^2+6*10^-309")
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0)
        .map_coeff(
            |c| Complex::from(F64::from(c.to_f64())),
            FloatField::from_rep(Complex::from(F64::from(1.0))),
        );
    let guesses = [-1e-308, 1e-308]
        .map(|im| Complex::new(F64::from(0.0), F64::from(im)))
        .to_vec();
    let roots = polynomial
        .roots_hot_start(10, &F64::from(1e-310), guesses)
        .unwrap();
    for root in roots {
        assert!(root.re.norm() < F64::from(1e-310));
        assert!((root.im.norm() / F64::from(1e-308) - F64::from(1.0)).norm() < F64::from(1e-10));
    }
}

#[test]
fn newton_convergence_does_not_depend_on_expansion() {
    let factored = parse!("(10^80*x-1)^2-2");
    let expanded = factored.expand();
    let roots = [factored, expanded].map(|expression| {
        expression
            .nsolve(
                symbol!("x"),
                Float::with_val(256, 2e-80),
                Float::with_val(256, 1e-60),
                1,
            )
            .unwrap()
    });
    assert!((roots[0].clone() - &roots[1]).norm() < Float::with_val(256, 1e-150));
}

fn wilkinson(degree: usize) -> symbolica::poly::univariate::UnivariatePolynomial<Q> {
    let factors = (1..=degree)
        .map(|k| format!("(x-{k})"))
        .collect::<Vec<_>>()
        .join("*");
    parse!(&factors)
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0)
}

#[test]
fn aberth_initial_guesses_follow_root_magnitudes() {
    let polynomial = parse!("(x-10^-6)*(x-1)*(x-10^6)")
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0)
        .map_coeff(
            |c| Complex::from(F64::from(c.to_f64())),
            FloatField::from_rep(Complex::from(F64::from(1.0))),
        );
    // Starting points spread up to the Cauchy bound need many more iterations here.
    let roots = polynomial.roots(10, &F64::from(1e-12)).unwrap();
    for (root, expected) in roots.iter().zip([1e-6, 1.0, 1e6]) {
        let expected = F64::from(expected);
        assert!((root.re.clone() / &expected - F64::from(1.0)).norm() < F64::from(1e-12));
        assert!(root.im.norm() < F64::from(1e-12) * expected);
    }
}

#[test]
fn aberth_initial_guesses_avoid_the_real_axis() {
    for (expression, expected) in [
        (
            "x^4+1",
            vec![
                (-0.5f64.sqrt(), -0.5f64.sqrt()),
                (-0.5f64.sqrt(), 0.5f64.sqrt()),
                (0.5f64.sqrt(), -0.5f64.sqrt()),
                (0.5f64.sqrt(), 0.5f64.sqrt()),
            ],
        ),
        ("(x^2+1)*(x-2)", vec![(0.0, -1.0), (0.0, 1.0), (2.0, 0.0)]),
    ] {
        let polynomial = parse!(expression)
            .to_polynomial::<_, u16>(&Q, None)
            .to_univariate_from_univariate(0)
            .map_coeff(
                |c| Complex::from(F64::from(c.to_f64())),
                FloatField::from_rep(Complex::from(F64::from(1.0))),
            );
        let roots = polynomial.roots(50, &F64::from(1e-12)).unwrap();
        assert_eq!(roots.len(), expected.len(), "{expression}");
        for (root, (re, im)) in roots.iter().zip(expected) {
            let distance = (root.clone() - Complex::new(F64::from(re), F64::from(im)))
                .norm()
                .re;
            assert!(distance < F64::from(1e-12), "{expression}: {root}");
        }
    }
}

#[test]
fn aberth_does_not_overflow_on_large_cauchy_bounds() {
    // The Cauchy bound is about 20!, and evaluating the polynomial at such a
    // starting point overflows in double precision.
    let roots = wilkinson(20)
        .approximate_roots::<F64>(100, &F64::from(1e-3))
        .unwrap();
    assert_eq!(roots.len(), 20);
    for (root, expected) in roots.iter().zip(1..=20) {
        assert!(root.0.is_finite());
        assert!(
            (root.0.re.clone() - F64::from(expected as f64)).norm() < F64::from(0.05),
            "{expected}: {}",
            root.0
        );
    }
}

#[test]
fn aberth_keeps_working_precision_on_ill_conditioned_polynomials() {
    // Evaluating this polynomial near its roots loses tens of bits to cancellation.
    // Iterates that inherit that tracked loss decay to a handful of bits within
    // ten iterations.
    let roots = wilkinson(20)
        .approximate_roots::<Float>(100, &Float::with_val(256, 1e-40))
        .unwrap();
    assert_eq!(roots.len(), 20);
    for ((root, multiplicity), expected) in roots.iter().zip(1..=20) {
        assert_eq!(*multiplicity, 1);
        assert!(root.re.get_precision() >= 200, "{expected}: {root}");
        assert!(
            (root.re.clone() - Float::with_val(256, expected)).norm() < Float::with_val(256, 1e-40),
            "{expected}: {root}"
        );
        assert!(
            root.im.norm() < Float::with_val(256, 1e-40),
            "{expected}: {root}"
        );
    }
}

#[test]
fn aberth_tolerance_is_relative_for_large_roots() {
    // An absolute tolerance of 1e-60 is below the spacing of 256-bit floats near 1e30.
    let tolerance = Float::with_val(256, 1e-60);
    let polynomial = parse!("(x-10^-30)*(x-1)*(x-10^30)*(x^2+10^20)")
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0)
        .map_coeff(
            |c| Complex::from(c.to_multi_prec_float(256)),
            FloatField::from_rep(Complex::from(tolerance.clone())),
        );
    let roots = polynomial.roots(100, &tolerance).unwrap();
    assert_eq!(roots.len(), 5);
    let exact = |expression: &str| {
        parse!(expression)
            .to_polynomial::<_, u16>(&Q, None)
            .get_constant()
            .to_multi_prec_float(256)
    };
    let zero = Float::with_val(256, 0);
    for expected in [
        Complex::new(exact("10^-30"), zero.clone()),
        Complex::new(exact("1"), zero.clone()),
        Complex::new(exact("10^30"), zero.clone()),
        Complex::new(zero.clone(), exact("10^10")),
        Complex::new(zero.clone(), exact("-10^10")),
    ] {
        let closest = roots
            .iter()
            .map(|root| (root.clone() - &expected).norm().re)
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();
        assert!(
            closest < Float::with_val(256, 1e-55) * expected.norm().re,
            "{expected}: {closest}"
        );
    }
}
