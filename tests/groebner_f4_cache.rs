use std::sync::Arc;
use symbolica::{
    atom::{Atom, AtomCore},
    domains::rational::{Q, RationalField},
    poly::{PolyVariable, groebner::GroebnerBasis, polynomial::MultivariatePolynomial},
    symbol,
};

type Poly = MultivariatePolynomial<RationalField, u16>;
fn parse(text: &str, variables: &Arc<Vec<PolyVariable>>) -> Poly {
    Atom::parse(text, "f4_cache_regression", Default::default())
        .unwrap()
        .try_to_polynomial(&Q, variables.clone())
        .unwrap()
}
fn variables(order: &[&str]) -> Arc<Vec<PolyVariable>> {
    Arc::new(
        order
            .iter()
            .map(|s| PolyVariable::from(symbol!(*s)))
            .collect(),
    )
}
fn input(v: &Arc<Vec<PolyVariable>>) -> Vec<Poly> {
    [
        "f4_cache_regression::x^2-f4_cache_regression::x-f4_cache_regression::z",
        "f4_cache_regression::u-f4_cache_regression::x*f4_cache_regression::u-1",
        "2*f4_cache_regression::x*f4_cache_regression::v-f4_cache_regression::v-1",
    ]
    .iter()
    .map(|s| parse(s, v))
    .collect()
}

#[test]
fn f4_exact_multiples_preserve_nonzero_critical_pair() {
    let variables = variables(&[
        "f4_cache_regression::x",
        "f4_cache_regression::u",
        "f4_cache_regression::z",
        "f4_cache_regression::v",
    ]);
    let source = input(&variables);
    // Independently written reduced Lex basis, including the eliminant omitted
    // when two distinct cached multiples were replaced by the same matrix row.
    let expected = [
        "f4_cache_regression::x-1/2-2*f4_cache_regression::z*f4_cache_regression::v-f4_cache_regression::v/2",
        "f4_cache_regression::u*f4_cache_regression::z+2*f4_cache_regression::z*f4_cache_regression::v+f4_cache_regression::v/2+1/2",
        "f4_cache_regression::u*f4_cache_regression::v-f4_cache_regression::u-2*f4_cache_regression::v",
        "(f4_cache_regression::z+1/4)*f4_cache_regression::v^2-1/4",
    ].iter().map(|s| parse(s,&variables)).collect::<Vec<_>>();
    assert!(GroebnerBasis::is_groebner_basis(&expected));
    for permutation in [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ] {
        let input = permutation.map(|i| source[i].clone());
        let basis = GroebnerBasis::new(&input, false);
        assert!(GroebnerBasis::is_groebner_basis(&basis.system));
        assert!(source.iter().all(|f| f.reduce(&basis.system).is_zero()));
        assert!(expected.iter().all(|f| f.reduce(&basis.system).is_zero()));
        assert!(basis.system.iter().all(|f| f.reduce(&expected).is_zero()));
    }
}

#[test]
fn f4_input_membership_survives_all_lex_variable_orders() {
    let names = [
        "f4_cache_regression::x",
        "f4_cache_regression::u",
        "f4_cache_regression::z",
        "f4_cache_regression::v",
    ];
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                for d in 0..4 {
                    let indices = [a, b, c, d];
                    if indices
                        .iter()
                        .enumerate()
                        .any(|(i, v)| indices[..i].contains(v))
                    {
                        continue;
                    }
                    let v = variables(&indices.map(|i| names[i]));
                    let source = input(&v);
                    let basis = GroebnerBasis::new(&source, false);
                    assert!(GroebnerBasis::is_groebner_basis(&basis.system));
                    assert!(source.iter().all(|f| f.reduce(&basis.system).is_zero()));
                    // All these ideals have a point; no false unit ideal may pass merely by
                    // reducing every input to zero.
                    assert!(!parse("1", &v).reduce(&basis.system).is_zero());
                }
            }
        }
    }
}

#[test]
fn f4_same_cache_invariant_over_finite_fields_and_grevlex() {
    use symbolica::{domains::finite_field::Zp, poly::GrevLexOrder};
    let vars = variables(&[
        "f4_cache_regression::x",
        "f4_cache_regression::u",
        "f4_cache_regression::z",
        "f4_cache_regression::v",
    ]);
    let rational = input(&vars);
    let grev = rational
        .iter()
        .map(|f| f.reorder::<GrevLexOrder>())
        .collect::<Vec<_>>();
    let basis = GroebnerBasis::new(&grev, false);
    assert!(GroebnerBasis::is_groebner_basis(&basis.system));
    assert!(grev.iter().all(|f| f.reduce(&basis.system).is_zero()));
    for prime in [13, 101, 65521] {
        let field = Zp::new(prime);
        let polys = rational
            .iter()
            .map(|f| {
                f.to_expression()
                    .try_to_polynomial::<_, u16>(&field, vars.clone())
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let basis = GroebnerBasis::new(&polys, false);
        assert!(GroebnerBasis::is_groebner_basis(&basis.system));
        assert!(polys.iter().all(|f| f.reduce(&basis.system).is_zero()));
        let grev = polys
            .iter()
            .map(|f| f.reorder::<GrevLexOrder>())
            .collect::<Vec<_>>();
        let basis = GroebnerBasis::new(&grev, false);
        assert!(GroebnerBasis::is_groebner_basis(&basis.system));
        assert!(grev.iter().all(|f| f.reduce(&basis.system).is_zero()));
    }
}
