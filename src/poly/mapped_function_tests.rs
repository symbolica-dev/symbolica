//! Bulk-conversion gates. The false raw mode preserves the old
//! conversion at every recursive level, rather than just skipping the top call.
use super::*;
use crate::domains::{algebraic::AlgebraicExtension, integer::Z, rational::Q};

fn atom(source: &str) -> Atom {
    Atom::parse(source, "mapped_function_bulk_tests", Default::default()).unwrap()
}

fn variables(names: &[&str]) -> Arc<Vec<PolyVariable>> {
    Arc::new(
        names
            .iter()
            .map(|name| PolyVariable::try_from(atom(name)).unwrap())
            .collect(),
    )
}

// Exactly the unchanged public wrapper's normalization boundary, applied only
// after the forced old raw converter has finished the complete expression.
fn finish_reference<R: Ring, E: PositiveExponent>(
    mut value: FactorizedRationalPolynomial<R, E>,
) -> FactorizedRationalPolynomial<R, E> {
    value.numerator.reduce_rational_power_variable_basis();
    for (denominator, _) in &mut value.denominators {
        denominator.reduce_rational_power_variable_basis();
    }
    for _ in 0..2 {
        for (denominator, _) in &mut value.denominators {
            value.numerator.unify_variables(denominator);
        }
    }
    value
}

// Denominator factors represent a commutative product and their vector order
// is not canonical. Ignore only that order, keeping every field, map,
// polynomial, scalar, power and repeated factor exact.
fn same_factorized_contents<R: Ring, E: PositiveExponent>(
    left: &FactorizedRationalPolynomial<R, E>,
    right: &FactorizedRationalPolynomial<R, E>,
) -> bool {
    if left.numerator.ring() != right.numerator.ring()
        || left.numerator.variables() != right.numerator.variables()
        || left.numerator != right.numerator
        || left.numer_coeff != right.numer_coeff
        || left.denom_coeff != right.denom_coeff
        || left.denominators.len() != right.denominators.len()
    {
        return false;
    }
    let mut unmatched = right.denominators.iter().collect::<Vec<_>>();
    for (factor, exponent) in &left.denominators {
        let Some(position) = unmatched.iter().position(|(other, other_exponent)| {
            factor.ring() == other.ring()
                && factor.variables() == other.variables()
                && factor == other
                && exponent == other_exponent
        }) else {
            return false;
        };
        unmatched.swap_remove(position);
    }
    unmatched.is_empty()
}

#[test]
fn factorized_content_comparison_ignores_only_factor_order() {
    let vars = variables(&["x", "f(x)"]);
    let reference = atom("(f(x)+1)/((x-2)*(x-3))")
        .as_view()
        .to_factorized_rational_polynomial_mode::<_, _, u32, false>(&Q, &Z, &vars)
        .unwrap();
    assert_eq!(reference.denominators.len(), 2);
    let mut reordered = reference.clone();
    reordered.denominators.reverse();
    assert_ne!(reordered, reference);
    assert!(same_factorized_contents(&reordered, &reference));

    let mut changed = reference.clone();
    changed.numerator = changed.numerator.add_constant(Z.one());
    assert!(!same_factorized_contents(&changed, &reference));
    let mut changed = reference.clone();
    changed.numer_coeff = Z.zero();
    assert!(!same_factorized_contents(&changed, &reference));
    let mut changed = reference.clone();
    changed.denominators[0].1 += 1;
    assert!(!same_factorized_contents(&changed, &reference));
    let mut changed = reference.clone();
    changed.denominators.pop();
    assert!(!same_factorized_contents(&changed, &reference));
    let mut changed = reference.clone();
    changed.denominators[0] = changed.denominators[1].clone();
    assert!(!same_factorized_contents(&changed, &reference));
    let replacement = PolyVariable::try_from(atom("different_variable")).unwrap();
    let mut changed = reference.clone();
    changed.numerator.rename_variable(&vars[0], &replacement);
    assert!(!same_factorized_contents(&changed, &reference));
    let mut changed = reference.clone();
    changed.denominators[0]
        .0
        .rename_variable(&vars[0], &replacement);
    assert!(!same_factorized_contents(&changed, &reference));
}

fn assert_real_reference(input: &Atom, vars: &Arc<Vec<PolyVariable>>) {
    let old = input
        .as_view()
        .to_factorized_rational_polynomial_mode::<_, _, u32, false>(&Q, &Z, vars)
        .unwrap();
    let new = input
        .as_view()
        .to_factorized_rational_polynomial_impl::<_, _, u32>(&Q, &Z, vars)
        .unwrap();
    assert_eq!(new.get_variables(), old.get_variables());
    assert_eq!(new, old, "raw conversion changed: {input}");
    let finished = input
        .try_to_factorized_rational_polynomial::<_, _, u32>(&Q, &Z, Some(vars.clone()))
        .unwrap();
    let old = finish_reference(old);
    assert_eq!(finished.get_variables(), old.get_variables());
    assert_eq!(finished, old, "final normalization changed: {input}");
}

#[test]
fn mapped_function_bulk_preserves_exact_maps_and_coefficients() {
    let input =
        atom("2/3*f(h(x))*g(y)^2*x^3-7/11*f(h(x))^3+13*g(y)+123456789123456789123456789*f(h(x))");
    for names in [
        ["unused", "g(y)", "x", "f(h(x))", "unused_function(z)"],
        ["f(h(x))", "unused_function(z)", "unused", "x", "g(y)"],
    ] {
        let vars = variables(&names);
        assert!(
            input
                .as_view()
                .to_polynomial_expanded::<_, u32>(&Q, Some(&vars), false)
                .is_err()
        );
        let bulk = input
            .as_view()
            .to_polynomial_expanded_mode::<_, u32, true>(&Q, Some(&vars), false)
            .unwrap();
        assert_eq!(bulk.variables().as_ref(), vars.as_ref());
        assert_real_reference(&input, &vars);
    }
}

#[test]
fn mapped_function_bulk_preserves_complex_field_and_opaque_arguments() {
    let field = AlgebraicExtension::complex(Q);
    let input = atom("(2+3𝑖)*f(x)^2+(7-5𝑖)*g(1+𝑖)*x-11*f(x)");
    let vars = variables(&["unused", "g(1+𝑖)", "x", "f(x)"]);
    let bulk = input
        .as_view()
        .to_polynomial_expanded_mode::<_, u32, true>(&field, Some(&vars), false)
        .unwrap();
    assert_eq!(bulk.variables().as_ref(), vars.as_ref());
    let old = input
        .as_view()
        .to_factorized_rational_polynomial_mode::<_, _, u32, false>(&field, &field, &vars)
        .unwrap();
    let new = input
        .as_view()
        .to_factorized_rational_polynomial_impl::<_, _, u32>(&field, &field, &vars)
        .unwrap();
    assert_eq!(new.get_variables(), old.get_variables());
    assert_eq!(new, old);
    assert_eq!(
        input
            .try_to_factorized_rational_polynomial::<_, _, u32>(&field, &field, Some(vars.clone()),)
            .unwrap(),
        finish_reference(old),
    );
}

#[test]
fn mapped_function_bulk_preserves_recursive_fallback_and_root_boundary() {
    for source in [
        "(f(x)+g(x))/(1-x)+(f(x)-g(x))/(1-x)-2*f(x)/(1-x)",
        "f(x)*x^(1/2)/(1-x)+g(x)*x^(1/3)/(1+x)+f(x)*x^(5/6)/(1-x)",
        "f(x)*x^(a/2)/(1-x)+g(x)*x^(a/3)/(1+x)",
        "f(x)^(-2)+g(x)^(-1)+f(x)^(1/2)",
    ] {
        let input = atom(source);
        let raw = input
            .as_view()
            .to_factorized_rational_polynomial_mode::<_, _, u32, false>(
                &Q,
                &Z,
                &Arc::new(Vec::new()),
            )
            .unwrap();
        let mut map = raw.get_variables().to_vec();
        map.push(PolyVariable::try_from(atom("unused")).unwrap());
        assert_real_reference(&input, &Arc::new(map));
    }
}

#[test]
fn mapped_function_bulk_declines_unmapped_and_nonpolynomial_inputs() {
    let vars = variables(&["x", "f(x)"]);
    for source in [
        "g(x)+f(x)",
        "f(y)+f(x)",
        "y*f(x)",
        "f(x)^(-1)",
        "f(x)^(1/2)",
        "f(x)^x",
        "(x+f(x))^2",
    ] {
        let input = atom(source);
        assert!(
            input
                .as_view()
                .to_polynomial_expanded_mode::<_, u32, true>(&Q, Some(&vars), false)
                .is_err(),
            "unexpected admission: {source}"
        );
        assert_real_reference(&input, &vars);
    }
    // Unsupported coefficients retain the old conversion error/fallback; the
    // new parser does not silently approximate a floating coefficient in Q.
    assert!(
        atom("0.125*f(x)")
            .as_view()
            .to_polynomial_expanded_mode::<_, u32, true>(&Q, Some(&vars), false)
            .is_err()
    );
}

#[test]
fn mapped_function_bulk_checks_generic_exponents_without_truncation() {
    let vars = variables(&["x", "f(x)"]);
    assert!(
        atom("f(x)^255+x*f(x)")
            .as_view()
            .to_polynomial_expanded_mode::<_, u8, true>(&Q, Some(&vars), false)
            .is_ok()
    );
    for source in ["f(x)^256", "f(x)*x^256"] {
        assert!(
            atom(source)
                .as_view()
                .to_polynomial_expanded_mode::<_, u8, true>(&Q, Some(&vars), false)
                .is_err()
        );
    }
    for source in ["f(x)^65536", "f(x)*x^65536"] {
        assert!(
            atom(source)
                .as_view()
                .to_polynomial_expanded_mode::<_, u16, true>(&Q, Some(&vars), false)
                .is_err()
        );
    }
    for source in ["f(x)^2147483648", "f(x)^123456789123456789123456789"] {
        assert!(
            atom(source)
                .as_view()
                .to_polynomial_expanded_mode::<_, u32, true>(&Q, Some(&vars), false)
                .is_err()
        );
    }
    // Deliberately do not run the general converter on oversized inputs: this
    // optimization declines them and does not promise to repair old overflow.
}

#[test]
fn mapped_function_bulk_leaves_ordinary_algebraic_root_embedding_unchanged() {
    let root = atom("root(a^3-2,2)");
    assert!(matches!(root.as_view(), AtomView::Fun(_)));
    let field = AlgebraicExtension::from_polynomial_with_embedding(
        atom("a^3-2").to_polynomial::<_, u16>(&Q, None),
        2,
    );
    let coefficient = field.try_element_from_root(root.as_view()).unwrap();
    let vars = Arc::new(vec![PolyVariable::try_from(root.clone()).unwrap()]);
    let expected = MultivariatePolynomial::new(&field, None, vars.clone()).constant(coefficient);
    assert_eq!(
        root.to_polynomial::<_, u16>(&field, Some(vars.clone())),
        expected
    );
    // Factorized conversion has always treated this function as an indeterminate.
    let old = root
        .as_view()
        .to_factorized_rational_polynomial_mode::<_, _, u16, false>(&field, &field, &vars)
        .unwrap();
    let new = root
        .as_view()
        .to_factorized_rational_polynomial_impl::<_, _, u16>(&field, &field, &vars)
        .unwrap();
    assert_eq!(old, new);
    assert_eq!(new.numerator.exponents, vec![1]);
}

fn assert_power_reference<R, RO>(
    input: &Atom,
    vars: &Arc<Vec<PolyVariable>>,
    field: &R,
    out_field: &RO,
) where
    R: EuclideanDomain + ConvertToRing,
    RO: EuclideanDomain + PolynomialGCD<u32>,
    FactorizedRationalPolynomial<RO, u32>: FromNumeratorAndFactorizedDenominator<R, RO, u32>
        + FromNumeratorAndFactorizedDenominator<RO, RO, u32>,
    MultivariatePolynomial<RO, u32>: Factorize,
{
    let old = input
        .as_view()
        .to_factorized_rational_polynomial_mode::<_, _, u32, false>(field, out_field, vars)
        .unwrap();
    let candidate = input
        .as_view()
        .to_factorized_rational_polynomial_impl::<_, _, u32>(field, out_field, vars)
        .unwrap();
    assert!(same_factorized_contents(&candidate, &old), "raw: {input}");
    let finished = input
        .try_to_factorized_rational_polynomial::<_, _, u32>(field, out_field, Some(vars.clone()))
        .unwrap();
    assert!(
        same_factorized_contents(&finished, &finish_reference(old)),
        "finished: {input}"
    );
}

#[test]
fn mapped_power_bulk_preserves_primitive_powers_and_root_normalization() {
    for (source, names) in [
        (
            "mu^ep*f(x)+(1/m^2)^ep*g(x)*x",
            vec!["unused", "g(x)", "(1/m^2)^ep", "x", "f(x)", "mu^ep"],
        ),
        (
            "x^(1/2)*f(x)+x^(1/3)*g(x)",
            vec!["unused", "x^(1/3)", "f(x)", "x", "g(x)", "x^(1/2)"],
        ),
        (
            "x^(ep/2)*f(x)+x^(ep/3)*g(x)",
            vec!["unused", "x^(ep/3)", "f(x)", "g(x)", "x^(ep/2)"],
        ),
        (
            "f(x)^(1/2)+g(x)^ep",
            vec!["unused", "g(x)^ep", "f(x)^(1/2)"],
        ),
        (
            "2^(1/2)*f(x)+3^(1/2)*g(x)",
            vec!["unused", "2^(1/2)", "f(x)", "g(x)", "3^(1/2)"],
        ),
    ] {
        let input = atom(source);
        for reverse in [false, true] {
            let mut names = names.clone();
            if reverse {
                names.reverse();
            }
            let vars = variables(&names);
            let bulk = input
                .as_view()
                .to_polynomial_expanded_mode::<_, u32, true>(&Q, Some(&vars), false)
                .unwrap();
            assert_eq!(bulk.variables().as_ref(), vars.as_ref());
            assert!(
                input
                    .as_view()
                    .to_polynomial_expanded::<_, u32>(&Q, Some(&vars), false)
                    .is_err()
            );
            assert_power_reference(&input, &vars, &Q, &Z);
        }
    }
}

#[test]
fn mapped_power_bulk_preserves_positive_integer_content_and_complex_domain() {
    let field = AlgebraicExtension::complex(Q);
    for (source, names) in [
        (
            "(2+3𝑖)*mu^(3*ep)*f(x)+(7-5𝑖)*(1/m^2)^(3*ep)*g(1+𝑖)",
            vec![
                "unused",
                "mu^(3*ep)",
                "mu^ep",
                "(1/m^2)^ep",
                "f(x)",
                "g(1+𝑖)",
            ],
        ),
        (
            "𝑖*x^(2/3)*f(x)+(1+𝑖)*x^(1/2)*g(x)",
            vec!["unused", "x^(2/3)", "x^(1/3)", "x^(1/2)", "f(x)", "g(x)"],
        ),
        (
            "mu^((2+4𝑖)*ep)*f(x)+mu^(2/3+4𝑖/5)*g(x)",
            vec!["unused", "mu^((1+2𝑖)*ep)", "mu^(1/3+2𝑖/5)", "f(x)", "g(x)"],
        ),
        (
            "(x^(1/2))^3*f(x)+(mu^ep)^3*g(x)",
            vec!["unused", "x^(1/2)", "mu^ep", "f(x)", "g(x)"],
        ),
    ] {
        let input = atom(source);
        let vars = variables(&names);
        let bulk = input
            .as_view()
            .to_polynomial_expanded_mode::<_, u32, true>(&field, Some(&vars), false)
            .unwrap();
        assert_eq!(bulk.variables().as_ref(), vars.as_ref());
        assert_power_reference(&input, &vars, &field, &field);
    }
}

#[test]
fn mapped_power_bulk_keeps_legacy_decomposition_and_missing_map_fallbacks() {
    for source in [
        "mu^(a+ep)",
        "mu^(2*ep)",
        "mu^(-ep)",
        "mu^(2/3)",
        "mu^(-1/2)",
        "mu^(2*ep/3)",
        "mu^((2+4𝑖)*ep)",
        "mu^((-2+4𝑖)*ep)",
    ] {
        // Supplying the undecomposed whole Power must not override the legacy
        // Add/integer-content rewrite or invent its missing primitive variable.
        let input = atom(source);
        let vars = variables(&["unused", source]);
        assert!(
            input
                .as_view()
                .to_polynomial_expanded_mode::<_, u32, true>(&Q, Some(&vars), false)
                .is_err(),
            "incorrect whole-Power admission: {source}"
        );
        assert_power_reference(&input, &vars, &Q, &Z);
    }
    for (source, names) in [
        ("mu^(-ep)", vec!["mu^(-ep)", "mu^ep"]),
        ("mu^(-3*ep)", vec!["mu^(-3*ep)", "mu^ep"]),
        ("mu^(a+ep)", vec!["mu^(a+ep)", "mu^a", "mu^ep"]),
    ] {
        // Even complete primitive/component maps must not override the old
        // negative-content inversion or additive-exponent decomposition.
        let input = atom(source);
        let vars = variables(&names);
        assert!(
            input
                .as_view()
                .to_polynomial_expanded_mode::<_, u32, true>(&Q, Some(&vars), false)
                .is_err(),
            "incorrect decomposition admission: {source}"
        );
        assert_power_reference(&input, &vars, &Q, &Z);
    }
    for source in ["mu^ep*f(x)", "mu^(3*ep)*f(x)"] {
        let input = atom(source);
        let vars = variables(&["unused", "f(x)"]);
        assert!(
            input
                .as_view()
                .to_polynomial_expanded_mode::<_, u32, true>(&Q, Some(&vars), false)
                .is_err()
        );
        assert_power_reference(&input, &vars, &Q, &Z);
    }
    let vars = variables(&["mu^ep"]);
    assert!(
        atom("mu^(255*ep)")
            .as_view()
            .to_polynomial_expanded_mode::<_, u8, true>(&Q, Some(&vars), false)
            .is_ok()
    );
    for source in ["mu^(256*ep)", "mu^(2147483648*ep)"] {
        assert!(
            atom(source)
                .as_view()
                .to_polynomial_expanded_mode::<_, u8, true>(&Q, Some(&vars), false)
                .is_err()
        );
    }
}

#[test]
#[ignore = "bounded conversion microbenchmark; run explicitly in an isolated release test process"]
fn mapped_function_bulk_replay_benchmark() {
    let mut map = vec![PolyVariable::try_from(atom("x")).unwrap()];
    map.extend((0..24).map(|i| PolyVariable::try_from(atom(&format!("master({i})"))).unwrap()));
    map.push(PolyVariable::try_from(atom("unused")).unwrap());
    let vars = Arc::new(map);
    let input = Atom::add_many((0..3072).map(|i| {
        atom(&format!(
            "{}/{}*master({})*x^{}",
            1 + i % 17,
            1 + i % 13,
            i % 24,
            i / 24
        ))
    }));
    let mut old_times = Vec::new();
    let mut new_times = Vec::new();
    for old_first in [true, false, false, true] {
        let mut old_result = None;
        let mut new_result = None;
        for old_mode in [old_first, !old_first] {
            let start = std::time::Instant::now();
            if old_mode {
                let value = input
                    .as_view()
                    .to_factorized_rational_polynomial_mode::<_, _, u32, false>(&Q, &Z, &vars)
                    .unwrap();
                old_times.push(start.elapsed().as_secs_f64());
                old_result = Some(value);
            } else {
                let value = input
                    .as_view()
                    .to_factorized_rational_polynomial_impl::<_, _, u32>(&Q, &Z, &vars)
                    .unwrap();
                new_times.push(start.elapsed().as_secs_f64());
                new_result = Some(value);
            }
        }
        let old = old_result.unwrap();
        let new = new_result.unwrap();
        assert_eq!(old.get_variables(), new.get_variables());
        assert_eq!(old, new);
    }
    eprintln!(
        "fixed-map 3072-term raw conversion seconds: old={old_times:?}; candidate={new_times:?}; input construction/equality excluded"
    );
}

#[test]
#[ignore = "replay grouped numerators from saved d879601 capture; explicit test-only file required"]
fn mapped_power_bulk_replay_captured_numerators() {
    let path = std::env::var_os("SYMBOLICA_MAPPED_POWER_CAPTURE_FILE")
        .expect("set SYMBOLICA_MAPPED_POWER_CAPTURE_FILE to saved d879601 canonical .expr input");
    let path = std::path::PathBuf::from(path);
    assert_eq!(
        path.file_name().unwrap(),
        "d87960113bb4eaec6508b833002c7de6bdffe465848b74d19975c81421b27120.expr"
    );
    assert!(std::fs::metadata(&path).unwrap().len() < 16 * 1024 * 1024);
    let source = std::fs::read_to_string(&path).unwrap();
    let input = Atom::parse(&source, "vakint", Default::default())
        .unwrap()
        .expand();
    assert!(input.as_view().has_complex_coefficients());
    let AtomView::Add(sum) = input.as_view() else {
        panic!("the captured input must be an expanded sum");
    };
    assert!(sum.get_nargs() < 50_000);
    let field = AlgebraicExtension::complex(Q);
    let discovery = input
        .as_view()
        .to_factorized_rational_polynomial_mode::<_, _, u32, false>(
            &field,
            &field,
            &Arc::new(Vec::new()),
        )
        .unwrap();
    let vars = discovery.numerator.variables().clone();
    assert!(vars.iter().any(|v| matches!(v, PolyVariable::Power(_))));
    drop(discovery);

    // Reproduce the large numerator sums passed to conversion after grouping
    // terms by their exact negative-integer-power denominator factors. Keep
    // their complete original raw map; do not normalize roots per term.
    let mut groups = std::collections::BTreeMap::<Atom, Vec<Atom>>::new();
    for term in sum {
        let factors = match term {
            AtomView::Mul(product) => product.iter().collect::<Vec<_>>(),
            factor => vec![factor],
        };
        let (denominator, numerator): (Vec<_>, Vec<_>) = factors.into_iter().partition(|factor| {
            let AtomView::Pow(power) = factor else {
                return false;
            };
            Rational::try_from(power.get_base_exp().1)
                .is_ok_and(|exponent| exponent.is_integer() && exponent.is_negative())
        });
        groups
            .entry(Atom::mul_many(denominator))
            .or_default()
            .push(Atom::mul_many(numerator));
    }
    let mut seconds = [0.0; 2];
    let mut terms = 0;
    for numerators in groups.values() {
        let numerator = Atom::add_many(numerators);
        terms += numerator.as_view().nterms();
        // This assertion makes the saved 3ε scale factors an admission gate,
        // rather than merely comparing two conversions that both fell back.
        let bulk = numerator
            .as_view()
            .to_polynomial_expanded_mode::<_, u32, true>(&field, Some(&vars), false)
            .expect("the captured numerator must use the mapped-Power bulk path");
        assert_eq!(bulk.variables().as_ref(), vars.as_ref());
        let start = std::time::Instant::now();
        let old = numerator
            .as_view()
            .to_factorized_rational_polynomial_mode::<_, _, u32, false>(&field, &field, &vars)
            .unwrap();
        seconds[0] += start.elapsed().as_secs_f64();
        let start = std::time::Instant::now();
        let candidate = numerator
            .as_view()
            .to_factorized_rational_polynomial_impl::<_, _, u32>(&field, &field, &vars)
            .unwrap();
        seconds[1] += start.elapsed().as_secs_f64();
        assert!(same_factorized_contents(&candidate, &old));
        assert!(same_factorized_contents(
            &finish_reference(candidate),
            &finish_reference(old)
        ));
    }
    eprintln!(
        "mapped-Power captured numerator replay: groups={} terms={terms} seconds[old,candidate]={seconds:?}; parsing, grouping, map discovery, admission and equality excluded",
        groups.len()
    );
}

// The old raw converter discovers the original map independently. Check both
// the fixed-map and automatic-map conversions against it. A call to together()
// is not an oracle: it would also use the candidate converter.
fn replay_capture_in_field<R, RO>(input: &Atom, field: &R, out_field: &RO) -> (usize, [f64; 4])
where
    R: EuclideanDomain + ConvertToRing,
    RO: EuclideanDomain + PolynomialGCD<u32>,
    FactorizedRationalPolynomial<RO, u32>: FromNumeratorAndFactorizedDenominator<R, RO, u32>
        + FromNumeratorAndFactorizedDenominator<RO, RO, u32>,
    MultivariatePolynomial<RO, u32>: Factorize,
{
    let started = std::time::Instant::now();
    let discovery = input
        .as_view()
        .to_factorized_rational_polynomial_mode::<_, _, u32, false>(
            field,
            out_field,
            &Arc::new(Vec::new()),
        )
        .unwrap();
    let vars = discovery.numerator.variables().clone();
    let discovery_seconds = started.elapsed().as_secs_f64();

    let started = std::time::Instant::now();
    let reference = input
        .as_view()
        .to_factorized_rational_polynomial_mode::<_, _, u32, false>(field, out_field, &vars)
        .unwrap();
    let reference_seconds = started.elapsed().as_secs_f64();
    let started = std::time::Instant::now();
    let candidate = input
        .as_view()
        .to_factorized_rational_polynomial_impl::<_, _, u32>(field, out_field, &vars)
        .unwrap();
    let candidate_seconds = started.elapsed().as_secs_f64();
    let started = std::time::Instant::now();
    let automatic = input
        .as_view()
        .to_factorized_rational_polynomial_impl::<_, _, u32>(
            field,
            out_field,
            &Arc::new(Vec::new()),
        )
        .unwrap();
    let automatic_seconds = started.elapsed().as_secs_f64();
    assert_eq!(automatic.get_variables(), discovery.get_variables());

    assert_eq!(reference.get_variables(), vars.as_slice());
    assert_eq!(candidate.get_variables(), vars.as_slice());
    assert_eq!(
        candidate.numerator.variables(),
        reference.numerator.variables()
    );
    assert!(
        same_factorized_contents(&candidate, &reference),
        "raw captured-input conversion changed"
    );
    assert!(
        same_factorized_contents(&automatic, &discovery),
        "raw automatic-map conversion changed"
    );

    // The public wrapper's body is unchanged and finish_reference implements
    // that exact boundary. Apply it once to each existing raw result without
    // repeating those conversions through the public wrapper. This is
    // an exact content comparison, including final numerator/denominator maps.
    let reference = finish_reference(reference);
    let candidate = finish_reference(candidate);
    let discovery = finish_reference(discovery);
    let automatic = finish_reference(automatic);
    assert_eq!(candidate.get_variables(), reference.get_variables());
    assert!(
        same_factorized_contents(&candidate, &reference),
        "final captured-input conversion changed"
    );
    assert!(
        same_factorized_contents(&reference, &discovery),
        "old fixed-map and automatic-map results differ"
    );
    assert!(
        same_factorized_contents(&automatic, &discovery),
        "final automatic-map conversion changed"
    );
    (
        vars.len(),
        [
            discovery_seconds,
            reference_seconds,
            candidate_seconds,
            automatic_seconds,
        ],
    )
}

#[test]
#[ignore = "replay the complete saved81-input corpus; explicit test-only directory required"]
fn mapped_function_bulk_replay_captured_inputs() {
    let directory = std::env::var_os("SYMBOLICA_MAPPED_FUNCTION_CAPTURE_DIR")
        .expect("set SYMBOLICA_MAPPED_FUNCTION_CAPTURE_DIR to the saved81 canonical .expr files");
    let mut paths = std::fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "expr")
        })
        .collect::<Vec<_>>();
    paths.sort();
    assert_eq!(
        paths.len(),
        81,
        "this gate requires the complete saved corpus"
    );

    let complex_field = AlgebraicExtension::complex(Q);
    let mut counts = [0usize; 2];
    for path in paths {
        // Parse the complete canonical expression exactly once. Do not parse
        // terms independently, discover symbols separately, or normalize roots
        // before the full raw conversion has completed.
        let source = std::fs::read_to_string(&path).unwrap();
        let input = Atom::parse(&source, "vakint", Default::default()).unwrap();
        let complex = input.as_view().has_complex_coefficients();
        // Select the original domain before cancellation and retain it for map
        // discovery, both raw conversions and both completed values, even if
        // the final coefficients become real or the entire input cancels.
        let (variables, seconds) = if complex {
            counts[1] += 1;
            replay_capture_in_field(&input, &complex_field, &complex_field)
        } else {
            counts[0] += 1;
            replay_capture_in_field(&input, &Q, &Z)
        };
        eprintln!(
            "capture={} field={} variables={variables} seconds[old_auto,old_fixed,new_fixed,new_auto]={seconds:?} exact_raw_and_finished=true",
            path.display(),
            if complex { "Q(i)" } else { "Q/Z" },
        );
    }
    assert_eq!(counts.iter().sum::<usize>(), 81);
    eprintln!(
        "captured raw/finished replay passed: Q/Z={}, Q(i)={}",
        counts[0], counts[1]
    );
}
