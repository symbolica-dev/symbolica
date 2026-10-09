use symbolica::{
    atom::{Atom, AtomCore},
    domains::{
        float::{Complex, DoubleFloat, Float, Real, RealLike},
        rational::Rational,
    },
    evaluate::{EvaluationDomain, ExpressionEvaluator},
    solve::{BracketedRootConvergence, BracketedRootOptions, nsolve_bracketed},
    symbol,
};

fn prepared_equation() -> ExpressionEvaluator<Complex<Rational>> {
    let x = symbol!("prepared_scalar_root::x");
    let a = symbol!("prepared_scalar_root::a");
    let b = symbol!("prepared_scalar_root::b");
    let c = symbol!("prepared_scalar_root::c");
    let value = Atom::var(a) * Atom::var(x).pow(2)
        + Atom::var(b) * Atom::var(x).pow(4)
        + Atom::var(c) * Atom::var(x).pow(6)
        - Atom::one();
    let derivative = value.derivative(x);
    Atom::evaluator_multiple(
        &[value, derivative],
        &[Atom::var(x), Atom::var(a), Atom::var(b), Atom::var(c)],
    )
    .build()
    .unwrap()
}

fn eager_control<N: RealLike + Real + PartialOrd + EvaluationDomain>(
    one: N,
    tolerance: N,
    bits: u32,
) {
    // Symbolic derivatives, optimization and coefficient mapping happen once.
    let mut program = prepared_equation().map_coeff_with_prec(
        &|coefficient| {
            assert_eq!(coefficient.im, Rational::from(0));
            one.from_rational(&coefficient.re)
        },
        bits,
    );
    let settings = BracketedRootOptions {
        absolute_tolerance: one.zero(),
        relative_tolerance: tolerance.clone(),
        max_iterations: 256,
        initial_guess: None,
        convergence: BracketedRootConvergence::Bracket,
    };
    let mut output = [one.zero(), one.zero()];
    for coefficients in [[1, 0, 0], [4, 0, 0], [1, 2, 3], [7, 5, 19]] {
        let parameters = coefficients.map(|a| one.from_usize(a));
        let answer = nsolve_bracketed(one.zero(), one.clone(), &settings, |x| {
            program.evaluate(
                &[
                    x.clone(),
                    parameters[0].clone(),
                    parameters[1].clone(),
                    parameters[2].clone(),
                ],
                &mut output,
            );
            (output[0].clone(), output[1].clone())
        })
        .unwrap();
        assert!(answer.residual.norm() < tolerance.clone() * one.from_usize(64));
        if coefficients[1] == 0 && coefficients[2] == 0 {
            let expected = one.clone() / parameters[0].sqrt();
            assert!((answer.root - expected).norm() <= tolerance);
        }
    }
}

#[test]
fn prepared_native_evaluator_retains_scalar_precision() {
    eager_control(1f64, 1e-14, 53);
    eager_control(DoubleFloat::from(1.), DoubleFloat::from(1e-29), 106);
    eager_control(Float::with_val(192, 1), Float::with_val(192, 1e-50), 192);
}

#[cfg(feature = "native_code_generation")]
#[test]
fn prepared_jit_evaluator_is_borrowed_across_refinements() {
    use symbolica::evaluate::JITCompilationSettings;

    let mut program = prepared_equation()
        .jit_compile::<f64>(
            JITCompilationSettings::default()
                .optimization_level(2)
                .direct_translation(true)
                .with_option("use_threads", "false"),
        )
        .unwrap();
    let mut settings = BracketedRootOptions {
        absolute_tolerance: 0.,
        relative_tolerance: 1e-14,
        max_iterations: 256,
        initial_guess: None,
        convergence: BracketedRootConvergence::Bracket,
    };
    let mut output = [0.; 2];
    for a in [1., 4., 16., 1e100, 1e200] {
        settings.initial_guess = (a > 1e20).then(|| 1. / f64::sqrt(a));
        let answer = nsolve_bracketed(0., 1., &settings, |x| {
            program.evaluate(&[*x, a, 0., 0.], &mut output);
            (output[0], output[1])
        })
        .unwrap();
        assert!((answer.root * a.sqrt() - 1.).abs() < 1e-13);
    }
}
