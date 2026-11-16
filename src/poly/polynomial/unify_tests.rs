use std::sync::Arc;

use super::*;
use crate::{
    atom::AtomCore,
    domains::integer::{Integer, Z},
    parse, symbol,
};

type Poly = MultivariatePolynomial<IntegerRing, u16>;

fn poly(expression: &str, vars: &[&str]) -> Poly {
    let vars = Arc::new(vars.iter().map(|v| symbol!(*v).into()).collect::<Vec<_>>());
    parse!(expression).to_polynomial(&Z, Some(vars))
}

fn vars(p: &Poly) -> Vec<String> {
    p.get_vars_ref().iter().map(|v| v.to_string()).collect()
}

/// Unify `a` and `b` and check that both are valid and unchanged in value.
fn unify(a: &str, a_vars: &[&str], b: &str, b_vars: &[&str]) -> (Poly, Poly) {
    let mut p1 = poly(a, a_vars);
    let mut p2 = poly(b, b_vars);
    let (e1, e2) = (p1.to_expression(), p2.to_expression());
    p1.unify_variables(&mut p2);

    p1.check_consistency();
    p2.check_consistency();
    assert_eq!(p1.variables(), p2.variables());
    assert!(Arc::ptr_eq(p1.variables(), p2.variables()));
    assert_eq!(p1.to_expression(), e1);
    assert_eq!(p2.to_expression(), e2);
    (p1, p2)
}

/// The canonical (sorted) order of the given variables.
fn sorted(names: &[&str]) -> Vec<String> {
    let mut v = names
        .iter()
        .map(|v| PolyVariable::from(symbol!(*v)))
        .collect::<Vec<_>>();
    v.sort();
    v.iter().map(|v| v.to_string()).collect()
}

#[test]
fn unify_uses_map_of_strict_superset() {
    let (p1, _) = unify("b^2+b", &["b"], "a*b+c^2+a^3", &["a", "b", "c"]);
    assert_eq!(vars(&p1), ["a", "b", "c"]);

    let (p1, _) = unify("a*c+c^2+a", &["a", "c"], "a*b+c^2+b^3", &["c", "b", "a"]);
    assert_eq!(vars(&p1), ["c", "b", "a"]);

    let (p1, _) = unify("a*b+c^2+b^3+a", &["a", "b", "c"], "3*c^2*a", &["c", "a"]);
    assert_eq!(vars(&p1), ["a", "b", "c"]);
}

#[test]
fn unify_keeps_superset_untouched() {
    let mut p1 = poly("a*b+c^2+b^3+a", &["a", "b", "c"]);
    let exponents = p1.exponents.as_ptr();
    let mut p2 = poly("c*a+a^2+c^5+a*c^3", &["c", "a"]);
    let e2 = p2.to_expression();
    p2.unify_variables(&mut p1);

    assert_eq!(vars(&p1), ["a", "b", "c"]);
    assert_eq!(p1.exponents.as_ptr(), exponents);
    p2.check_consistency();
    assert_eq!(p2.to_expression(), e2);
}

#[test]
fn unify_merge_uses_sorted_union() {
    let (p1, _) = unify(
        "x^2*a+b+x*a*b+x^3",
        &["x", "a", "b"],
        "b^2+a*y+y^3",
        &["b", "y", "a"],
    );
    assert_eq!(vars(&p1), sorted(&["x", "a", "b", "y"]));

    // equal variables in a different order
    let (p1, _) = unify("a*c", &["a", "c"], "c*a+a^2+c^5+a*c^3", &["c", "a"]);
    assert_eq!(vars(&p1), sorted(&["a", "c"]));
}

#[test]
fn unify_merge_keeps_common_prefix() {
    let (p1, _) = unify("y*z+x", &["y", "x", "z"], "w*x+y^2", &["y", "x", "w"]);
    let mut expected = vec!["y".to_string(), "x".to_string()];
    expected.extend(sorted(&["z", "w"]));
    assert_eq!(vars(&p1), expected);

    let mut polys = vec![
        poly("y*z+x", &["y", "x", "z"]),
        poly("x", &["x"]),
        poly("w*x+y^2", &["y", "x", "w"]),
        poly("v*y", &["y", "x", "v"]),
    ];
    MultivariatePolynomial::unify_variables_list(&mut polys);
    let mut expected = vec!["y".to_string(), "x".to_string()];
    expected.extend(sorted(&["z", "w", "v"]));
    assert_eq!(vars(&polys[0]), expected);
}

#[test]
fn conversion_keeps_order_of_given_partial_map() {
    let map = Arc::new(vec![symbol!("y").into(), symbol!("x").into()]);
    for expression in ["(x+y)^2*z+(y-x)*w", "f(z)*x+(y+x)^2*w+(x*z+1)^2"] {
        let p: Poly = parse!(expression).to_polynomial(&Z, Some(map.clone()));
        assert_eq!(&p.get_vars_ref()[..2], map.as_slice());
    }
}

#[test]
fn unify_is_symmetric() {
    let cases: [(&str, &[&str], &str, &[&str]); 4] = [
        (
            "x^2*a+b+x*a*b",
            &["x", "a", "b"],
            "b^2+a*y+y^3",
            &["b", "y", "a"],
        ),
        ("a*c+c^2", &["c", "a"], "c*a+a^2+c^5", &["a", "c"]),
        ("a*b+c^2+b^3+a", &["b", "c", "a"], "3*c^2*a+a", &["c", "a"]),
        ("a+1", &["a"], "b+1", &["b"]),
    ];
    for (a, a_vars, b, b_vars) in cases {
        let (p1, p2) = unify(a, a_vars, b, b_vars);
        let (q2, q1) = unify(b, b_vars, a, a_vars);
        assert_eq!(p1.variables(), q1.variables());
        assert_eq!(p1.exponents, q1.exponents);
        assert_eq!(p2.exponents, q2.exponents);

        let s1 = poly(a, a_vars) + poly(b, b_vars);
        let s2 = poly(b, b_vars) + poly(a, a_vars);
        assert_eq!(s1.variables(), s2.variables());
        assert_eq!(s1.exponents, s2.exponents);
        assert_eq!(s1.coefficients, s2.coefficients);
    }
}

#[test]
fn unify_repeated_converges() {
    // with a rule where `self` always wins, `b` would alternate between the maps of `a` and `c`
    let mut a = poly("x+y^2", &["x", "y"]);
    let mut b = poly("x^2+y", &["y", "x"]);
    let mut c = poly("x*y^2+1", &["y", "x"]);
    for _ in 0..2 {
        a.unify_variables(&mut b);
        c.unify_variables(&mut b);
    }
    assert_eq!(a.variables(), b.variables());
    assert_eq!(b.variables(), c.variables());
}

#[test]
fn unify_list_yields_common_map() {
    let input: [(&str, &[&str]); 5] = [
        ("a", &["a"]),
        ("c*a+a^2+c^5+a*c^3", &["c", "a"]),
        ("b+d*c+c^2*b", &["d", "c", "b"]),
        ("e*a+b", &["e", "a", "b"]),
        ("2", &["b", "a"]),
    ];

    let check = |input: &[(&str, &[&str])]| {
        let mut polys = input.iter().map(|(e, v)| poly(e, v)).collect::<Vec<_>>();
        let expressions = polys.iter().map(|p| p.to_expression()).collect::<Vec<_>>();
        MultivariatePolynomial::unify_variables_list(&mut polys);
        for (p, e) in polys.iter().zip(&expressions) {
            p.check_consistency();
            assert_eq!(p.variables(), polys[0].variables());
            assert_eq!(&p.to_expression(), e);
        }
        vars(&polys[0])
    };

    // no polynomial has all variables
    let mut reversed = input.to_vec();
    reversed.reverse();
    assert_eq!(check(&input), sorted(&["a", "b", "c", "d", "e"]));
    assert_eq!(check(&reversed), sorted(&["a", "b", "c", "d", "e"]));

    // one polynomial has all variables
    let mut with_full = input.to_vec();
    with_full.push(("a+b+c+d+e", &["e", "d", "c", "b", "a"]));
    assert_eq!(check(&with_full), ["e", "d", "c", "b", "a"]);
    with_full.reverse();
    assert_eq!(check(&with_full), ["e", "d", "c", "b", "a"]);

    // two polynomials have all variables in a different order
    with_full.push(("a*b*c*d*e", &["a", "b", "c", "e", "d"]));
    assert_eq!(check(&with_full), sorted(&["a", "b", "c", "d", "e"]));

    // consistent with pairwise unification
    let (p1, _) = unify("x+y", &["y", "x"], "x*z", &["x", "z"]);
    assert_eq!(
        check(&[("x+y", &["y", "x"]), ("x*z", &["x", "z"])]),
        vars(&p1)
    );
}

#[test]
fn from_coefficient_list_sorts_merges_and_removes_zeros() {
    let vars = Arc::new(vec![symbol!("a").into(), symbol!("b").into()]);
    let coefficients = [3, 0, 5, -3, 1, 2, -2, 4]
        .into_iter()
        .map(Integer::from)
        .collect::<Vec<_>>();
    #[rustfmt::skip]
    let exponents: Vec<u16> = vec![
        1, 0,
        0, 7,
        2, 1,
        1, 0,
        0, 0,
        0, 3,
        0, 3,
        2, 1,
    ];
    let p = Poly::from_coefficient_list(coefficients, exponents, vars.clone(), &Z);
    p.check_consistency();
    assert_eq!(p.to_expression(), parse!("1+9*a^2*b"));

    let p = Poly::from_coefficient_list(
        vec![Integer::from(1), Integer::from(-1)],
        vec![1, 0, 1, 0],
        vars.clone(),
        &Z,
    );
    assert!(p.is_zero());

    let p = Poly::from_coefficient_list(vec![], vec![], vars, &Z);
    assert!(p.is_zero());
}

#[test]
fn arithmetic_with_different_variable_maps() {
    let expected = |e: &str| parse!(e).expand();

    let p1 = poly("a*b+c^2+b^3+a+1", &["a", "b", "c"]);
    let p2 = poly("d*c-c^2-a+c^5+7", &["d", "c", "a"]);
    let monomial = poly("-2*c*d^2", &["c", "d"]);

    let sum = p1.clone() + p2.clone();
    sum.check_consistency();
    assert_eq!(sum.to_expression(), expected("a*b+b^3+1+d*c+c^5+7"));

    let sum = &p1 + &p2;
    sum.check_consistency();
    assert_eq!(sum.to_expression(), expected("a*b+b^3+1+d*c+c^5+7"));

    let diff = p1.clone() - p1.clone();
    assert!(diff.is_zero());

    for (l, r) in [(&p1, &monomial), (&monomial, &p1), (&p1, &p2)] {
        let product = l * r;
        product.check_consistency();
        assert_eq!(
            product.to_expression(),
            (l.to_expression() * r.to_expression()).expand()
        );
    }
}
