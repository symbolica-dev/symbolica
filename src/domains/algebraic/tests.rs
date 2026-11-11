//! Tests for algebraic extensions, elements, contexts, and roots.

use super::*;

#[test]
fn galois_prime_downgrade_checks_membership() {
    let base = Zp::new(5);
    let extension = base.upgrade(2);
    for value in [base.zero(), base.nth(2.into())] {
        let upgraded = base.upgrade_element(&value, &extension);
        assert_eq!(base.try_downgrade_element(&upgraded), Some(value));
        assert_eq!(base.downgrade_element(&upgraded), value);
    }
    assert!(base.try_downgrade_element(&extension.generator()).is_none());
    assert!(Zp::new(7).try_downgrade_element(&extension.one()).is_none());
}

#[test]
fn galois_extension_downgrade_preserves_representation() {
    let field = AlgebraicExtension::galois_field(Zp::new(5), 2, PolyVariable::Temporary(0));
    for value in [field.zero(), field.add(&field.generator(), &field.one())] {
        assert_eq!(field.try_downgrade_element(&value), Some(value));
    }
    let larger = field.upgrade(3);
    assert!(
        field
            .try_downgrade_element(&larger.pow(&larger.generator(), 2))
            .is_none()
    );
}

#[test]
fn contextual_conversion_of_new_rational_coefficients_with_zero_generator() {
    let x = symbol!("rational_context_zero_generator::x");
    let mut context = AlgebraicContext::new(AlgebraicExtension::trivial(Q));
    context.extend(Atom::zero().as_view()).unwrap();
    context.extend(Atom::one().as_view()).unwrap();
    let input = Atom::one() + Atom::var(x);
    let polynomial = context
        .to_polynomial::<u32>(input.as_view(), Arc::new(vec![PolyVariable::from(x)]))
        .unwrap();
    // Arithmetic creates a coefficient absent from the context's expression
    // cache. Its degree-zero generator term must not evaluate Atom::pow(0, 0).
    let doubled = &polynomial + &polynomial;
    let result = doubled.to_expression_with_context(&context).unwrap();
    assert!((result - Atom::num(2) * input).expand().is_zero());
}

#[test]
fn to_alg() {
    let a = crate::parse!("sqrt(2)+1");
    let ext = a.as_view().embedding_field().unwrap();
    let alg = a.as_view().to_algebraic(&ext).unwrap();
    let generator = ext.element_from_polynomial(ext.poly.one().mul_exp(&[1]));
    assert_eq!(ext.embedding.index(), Some(1));
    assert_eq!(alg, ext.add(&generator, &ext.one()));

    let b = crate::parse!("sqrt(2)+sqrt(3)");
    let ext = b.as_view().embedding_field().unwrap();
    let alg = b.as_view().to_algebraic(&ext).unwrap();
    let generator = ext.element_from_polynomial(ext.poly.one().mul_exp(&[1]));
    assert_eq!(ext.embedding.index(), Some(3));
    assert_eq!(alg, generator);

    let b = crate::parse!("sqrt(2)+sqrt(3)+sqrt(6)");
    let ext = b.as_view().embedding_field().unwrap();
    let alg = b.as_view().to_algebraic(&ext).unwrap();
    assert_eq!(ext.embedding.index(), Some(3));
    assert_eq!(ext.is_positive(&alg), Ok(true));

    let b = crate::parse!("sqrt(3+sqrt(2))+1");
    let ext = b.as_view().embedding_field().unwrap();
    let alg = b.as_view().to_algebraic(&ext).unwrap();
    let generator = ext.element_from_polynomial(ext.poly.one().mul_exp(&[1]));
    assert_eq!(ext.embedding.index(), Some(3));
    assert_eq!(alg, ext.add(&generator, &ext.one()));

    let b = crate::parse!("2^(2/3)");
    let ext = b.as_view().embedding_field().unwrap();
    let alg = b.as_view().to_algebraic(&ext).unwrap();
    assert_eq!(ext.embedding.index(), Some(0));
    assert_eq!(ext.pow(&alg, 3), ext.nth(4.into()));

    let b = crate::parse!("root(1-10*x^2+x^4,3)+1");
    let ext = b.as_view().embedding_field().unwrap();
    let alg = b.as_view().to_algebraic(&ext).unwrap();
    let generator = ext.element_from_polynomial(ext.poly.one().mul_exp(&[1]));
    assert_eq!(ext.embedding.index(), Some(3));
    assert_eq!(alg, ext.add(&generator, &ext.one()));
    let polynomial = b.as_view().try_to_polynomial::<_, u16>(&ext, None).unwrap();
    assert_eq!(polynomial.nvars(), 0);
    assert_eq!(polynomial.get_constant(), alg);

    let b = crate::parse!("root(x^3-2,1)+sqrt(3)");
    let ext = b.as_view().embedding_field().unwrap();
    let complex_cube_root = crate::parse!("root(x^3-2,1)")
        .as_view()
        .to_algebraic(&ext)
        .unwrap();
    let sqrt3 = crate::parse!("sqrt(3)")
        .as_view()
        .to_algebraic(&ext)
        .unwrap();
    let alg = b.as_view().to_algebraic(&ext).unwrap();
    assert_eq!(ext.pow(&complex_cube_root, 3), ext.nth(2.into()));
    assert_eq!(ext.mul(&sqrt3, &sqrt3), ext.nth(3.into()));
    assert_eq!(alg, ext.add(&complex_cube_root, &sqrt3));

    let b = crate::parse!("sqrt(2)+1𝑖");
    let ext = b.as_view().embedding_field().unwrap();
    let imaginary_unit = crate::parse!("1𝑖").as_view().to_algebraic(&ext).unwrap();
    let sqrt2 = crate::parse!("sqrt(2)")
        .as_view()
        .to_algebraic(&ext)
        .unwrap();
    let alg = b.as_view().to_algebraic(&ext).unwrap();
    assert_eq!(
        ext.mul(&imaginary_unit, &imaginary_unit),
        ext.neg(&ext.one())
    );
    assert_eq!(ext.mul(&sqrt2, &sqrt2), ext.nth(2.into()));
    assert_eq!(alg, ext.add(&sqrt2, &imaginary_unit));

    let b = crate::parse!("2+3𝑖");
    let ext = b.as_view().embedding_field().unwrap();
    assert_eq!(ext.embedding.index(), Some(1));
    let imaginary_unit = ext.imaginary_unit().unwrap();
    assert_eq!(
        b.as_view().to_algebraic(&ext).unwrap(),
        ext.add(
            &ext.nth(2.into()),
            &ext.mul(&ext.nth(3.into()), &imaginary_unit)
        )
    );
}

#[test]
fn algebraic_context_conversion() {
    let trivial = AlgebraicContext::from_atom(crate::parse!("x+1").as_view()).unwrap();
    assert!(trivial.is_trivial());

    let (trivial, polynomial) = crate::parse!("x^2+1")
        .to_polynomial_in_algebraic_extension::<u16>(symbol!("x"), &[])
        .unwrap();
    assert!(trivial.is_trivial());
    assert_eq!(polynomial.degree(0), 2);

    let expression = crate::parse!("x+sqrt(2)+sqrt(3)");
    let (context, polynomial) = expression
        .to_polynomial_in_algebraic_extension::<u16>(symbol!("x"), &[])
        .unwrap();
    assert_eq!(context.field().embedding.index(), Some(3));

    let sqrt2 = context
        .images()
        .get(&crate::parse!("sqrt(2)"))
        .unwrap()
        .clone();
    let sqrt3 = context
        .images()
        .get(&crate::parse!("sqrt(3)"))
        .unwrap()
        .clone();
    assert_eq!(
        context.field().mul(&sqrt2, &sqrt2),
        context.field().nth(2.into())
    );
    assert_eq!(
        context.field().mul(&sqrt3, &sqrt3),
        context.field().nth(3.into())
    );

    assert_eq!(
        polynomial.get_constant(),
        context.field().add(&sqrt2, &sqrt3)
    );
    assert_eq!(polynomial.coefficient(&[1]).unwrap(), context.field().one());

    let expression = crate::parse!("(x+sqrt(2))/(x-sqrt(3))");
    let (mut context, rational) = expression
        .to_rational_polynomial_in_algebraic_extension::<u16>(symbol!("x"))
        .unwrap();
    let sqrt2 = context
        .convert_atom(crate::parse!("sqrt(2)").as_view())
        .unwrap();
    let sqrt3 = context
        .convert_atom(crate::parse!("sqrt(3)").as_view())
        .unwrap();
    assert_eq!(rational.numerator.get_constant(), sqrt2);
    assert_eq!(
        rational.denominator.get_constant(),
        context.field().neg(&sqrt3)
    );
    assert_eq!(
        rational.numerator.coefficient(&[1]).unwrap(),
        context.field().one()
    );
    assert_eq!(
        rational.denominator.coefficient(&[1]).unwrap(),
        context.field().one()
    );
}

#[test]
fn algebraic_context_adjoins_high_degree_rational_root() {
    let generators = [crate::parse!("sqrt(2)"), crate::parse!("root(x^7-4,0)")];
    let (context, polynomial) = crate::parse!("x^2-2")
        .to_polynomial_in_algebraic_extension::<u16>(symbol!("x"), &generators)
        .unwrap();

    let sqrt_2 = context.image(&generators[0]).unwrap();
    assert_eq!(
        context.field().mul(sqrt_2, sqrt_2),
        context.field().nth(2.into())
    );
    let seventh_root = context.image(&generators[1]).unwrap();
    assert_eq!(
        context.field().pow(seventh_root, 7),
        context.field().nth(4.into())
    );
    assert_eq!(
        context
            .field()
            .root_index_of_element(
                seventh_root,
                &crate::parse!("x^7-4").to_polynomial::<_, u16>(&Q, None),
            )
            .unwrap(),
        0
    );
    assert_eq!(context.field().poly().degree(0), 14);
    assert_eq!(polynomial.factor().len(), 2);
}

#[test]
fn algebraic_context_adjoins_selected_rational_root() {
    let sqrt_2 = crate::parse!("sqrt(2)");
    let defining_polynomial = crate::parse!("y^4-7").to_polynomial::<_, u16>(&Q, None);
    let roots = [
        crate::parse!("root(y^4-7,0)"),
        crate::parse!("root(y^4-7,1)"),
        crate::parse!("root(y^4-7,2)"),
        crate::parse!("root(y^4-7,3)"),
    ];

    for (embedding, root) in roots.into_iter().enumerate() {
        let context = AlgebraicContext::from_generators(&[sqrt_2.clone(), root.clone()]).unwrap();
        let sqrt_2_image = context.image(&sqrt_2).unwrap();
        let root_image = context.image(&root).unwrap();

        assert_eq!(context.field().poly().degree(0), 8);
        assert_eq!(
            context.field().pow(sqrt_2_image, 2),
            context.field().nth(2.into())
        );
        assert_eq!(
            context.field().pow(root_image, 4),
            context.field().nth(7.into())
        );
        assert_eq!(
            context
                .field()
                .root_index_of_element(root_image, &defining_polynomial)
                .unwrap(),
            embedding
        );
    }
}

#[test]
fn algebraic_context_selects_factor_of_reducible_rational_root() {
    let sqrt_2 = crate::parse!("sqrt(2)");
    let root = crate::parse!("root((y^2-2)*(y^2-3),3)");
    let context = AlgebraicContext::from_generators(&[sqrt_2, root.clone()]).unwrap();
    let root_image = context.image(&root).unwrap();
    let defining_polynomial = crate::parse!("(y^2-2)*(y^2-3)").to_polynomial::<_, u16>(&Q, None);

    assert_eq!(context.field().poly().degree(0), 4);
    assert_eq!(
        context
            .field()
            .root_index_of_element(root_image, &defining_polynomial)
            .unwrap(),
        3
    );
}

#[test]
fn algebraic_display_hides_temporary_generators() {
    let (_, rational_polynomial) = crate::parse!("x^2-2")
        .to_polynomial_in_algebraic_extension::<u16>(symbol!("x"), &[])
        .unwrap();
    assert_eq!(rational_polynomial.to_string(), "-2+x^2");

    let (context, algebraic_polynomial) = crate::parse!("x+sqrt(2)")
        .to_polynomial_in_algebraic_extension::<u16>(symbol!("x"), &[])
        .unwrap();
    assert!(!algebraic_polynomial.to_string().contains("_TMP_"));
    assert!(!context.field().generator().to_string().contains("_TMP_"));
    assert!(algebraic_polynomial.to_string().contains('ξ'));
}

#[test]
fn algebraic_polynomial_to_expression() {
    let expression = crate::parse!("x+sqrt(2)");
    let (_, polynomial) = expression
        .to_polynomial_in_algebraic_extension::<u16>(symbol!("x"), &[])
        .unwrap();
    assert_eq!(polynomial.to_expression(), expression);

    let (context, polynomial) = crate::parse!("x^2-2")
        .to_polynomial_in_algebraic_extension::<u16>(symbol!("x"), &[crate::parse!("sqrt(2)")])
        .unwrap();
    let factors = polynomial
        .factor()
        .into_iter()
        .map(|(factor, _)| {
            let automatic = factor.to_expression();
            let with_context = factor.to_expression_with_context(&context).unwrap();
            assert_eq!(automatic, with_context);
            with_context
        })
        .collect::<Vec<_>>();
    assert!(factors.contains(&crate::parse!("x-sqrt(2)")));
    assert!(factors.contains(&crate::parse!("x+sqrt(2)")));

    let other_context = AlgebraicContext::from_generators(&[crate::parse!("sqrt(3)")]).unwrap();
    assert_eq!(
        polynomial
            .to_expression_with_context(&other_context)
            .unwrap_err(),
        "The polynomial uses a different coefficient field"
    );
}

#[test]
fn factor_with_large_unrelated_extension() {
    let lll_before = crate::poly::factor::LLL_RECOMBINATION_SUCCESSES.with(|count| count.get());
    let expression = crate::parse!("x^6-2");
    let extensions = [crate::parse!("sqrt(2)"), crate::parse!("root(x^24-7,0)")];
    let (context, polynomial) = expression
        .to_polynomial_in_algebraic_extension::<u16>(symbol!("x"), &extensions)
        .unwrap();

    let factors = polynomial
        .factor()
        .into_iter()
        .map(|(factor, exponent)| {
            assert_eq!(exponent, 1);
            factor.to_expression_with_context(&context).unwrap()
        })
        .collect::<Vec<_>>();

    assert_eq!(factors.len(), 2);
    assert!(factors.contains(&crate::parse!("x^3-sqrt(2)")));
    assert!(factors.contains(&crate::parse!("x^3+sqrt(2)")));
    let lll_after = crate::poly::factor::LLL_RECOMBINATION_SUCCESSES.with(|count| count.get());
    assert!(lll_after > lll_before);
}

#[test]
fn algebraic_context_preserves_integer_powers() {
    for expression in [
        crate::parse!("(1/2-1/2*13^(1/2))^2"),
        crate::parse!("root(-2+x^4,1)^2"),
        crate::parse!("root(1+4*x^3+x^6,1)^3"),
    ] {
        let context = AlgebraicContext::from_atom(expression.as_view()).unwrap();
        assert!(!context.is_trivial());
        assert!(context.images().contains_key(&expression));
    }
}

#[test]
fn factor_in_extension() {
    let factorization = crate::parse!("x^2+1")
        .factor_in_extension(&[Atom::i()])
        .unwrap();
    assert_eq!(factorization, crate::parse!("(x-1𝑖)*(x+1𝑖)"));

    let factorization = crate::parse!("x^2-2")
        .as_view()
        .factor_in_extension(&[crate::parse!("sqrt(2)")])
        .unwrap();
    assert_eq!(factorization.expand(), crate::parse!("x^2-2"));

    let factorization = crate::parse!("y^3-2")
        .as_view()
        .factor_in_extension(&[crate::parse!("root(x^3-2,2)")])
        .unwrap();
    assert_eq!(
        factorization,
        crate::parse!("(y-root(x^3-2,2))*(y*root(x^3-2,2)+y^2+root(x^3-2,2)^2)")
    );

    let factorization = crate::parse!("x^3-2")
        .as_view()
        .factor_in_extension(&[crate::parse!("root(x^3-2,2)")])
        .unwrap();
    assert_eq!(
        factorization,
        crate::parse!("(x-root(x^3-2,2))*(x*root(x^3-2,2)+x^2+root(x^3-2,2)^2)")
    );

    let factorization = crate::parse!("(x^2-2)/(x^2-3)")
        .as_view()
        .factor_in_extension(&[crate::parse!("sqrt(2)"), crate::parse!("sqrt(3)")])
        .unwrap();
    let mut context =
        AlgebraicContext::from_generators(&[crate::parse!("sqrt(2)"), crate::parse!("sqrt(3)")])
            .unwrap();
    let actual = context
        .to_rational_polynomial::<u16>(factorization.as_view(), None)
        .unwrap();
    let expected = context
        .to_rational_polynomial::<u16>(crate::parse!("(x^2-2)/(x^2-3)").as_view(), None)
        .unwrap();
    assert_eq!(actual, expected);

    let expression = crate::parse!("x^3+sqrt(2)*x^2-3*x-3*sqrt(2)");
    let factorization = expression.as_view().factor_in_extension(&[]).unwrap();
    assert_ne!(factorization, expression);
    assert_eq!(factorization.expand(), expression);

    let factorization = expression
        .as_view()
        .factor_in_extension(&[crate::parse!("sqrt(3)")])
        .unwrap();
    let mut context = AlgebraicContext::from_atom(expression.as_view()).unwrap();
    context
        .adjoin_generators(&[crate::parse!("sqrt(3)")])
        .unwrap();
    let actual = context
        .to_rational_polynomial::<u16>(factorization.as_view(), None)
        .unwrap();
    let expected = context
        .to_rational_polynomial::<u16>(expression.as_view(), None)
        .unwrap();
    assert_eq!(actual, expected);

    assert_eq!(
        crate::parse!("x^2-1")
            .as_view()
            .factor_in_extension(&[])
            .unwrap(),
        crate::parse!("(x-1)*(x+1)")
    );
    assert!(
        crate::parse!("x^2-2")
            .as_view()
            .factor_in_extension(&[crate::parse!("x")])
            .is_err()
    );
}

use std::cmp::Ordering;

use crate::atom::AtomCore;
use crate::domains::algebraic::{AlgebraicExtension, Root};
use crate::domains::finite_field::{PrimeIteratorU64, Z2, Zp};
use crate::domains::integer::{IntegerRing, Z};
use crate::domains::rational::Q;
use crate::domains::rational_polynomial::RationalPolynomialField;
use crate::domains::{RealEmbedding, Ring, RingOps};
use crate::{parse, symbol};

#[test]
fn simplify_parametric_root_struct() {
    let root =
        Root::<RationalPolynomialField<IntegerRing, u16>>::from_atom(parse!("-a+z^2").as_view(), 0)
            .unwrap();
    assert_eq!(root.simplify().unwrap(), parse!("-a^(1/2)"));
}

#[test]
fn positive_nested_radicals_use_real_first_embeddings() {
    for degree in [3, 4] {
        let expression = parse!(format!("(3+sqrt(2))^(1/{degree})"));
        let field = expression.as_view().embedding_field().unwrap();
        let value = expression.as_view().to_algebraic(&field).unwrap();
        let base = parse!("3+sqrt(2)").as_view().to_algebraic(&field).unwrap();
        assert!(field.is_positive_real(&value).unwrap());
        assert_eq!(field.pow(&value, degree), base);
    }
}

#[test]
fn quadratic_root_order_is_independent_of_leading_coefficient_sign() {
    for polynomial in ["x^2-2", "x^2+1"] {
        for scale in [-3, -1, 2] {
            for index in 0..2 {
                let root = Root::<Q>::from_atom(
                    parse!(format!("({scale})*({polynomial})")).as_view(),
                    index,
                )
                .unwrap();
                assert_eq!(
                    root.to_atom(),
                    parse!(format!("root({polynomial},{index})"))
                );
            }
        }
    }
}

#[test]
fn normalize_degree_twenty_extension_without_rediscovering_embeddings() {
    // Use a non-canonical presentation of Q(i) so that Root::simplify is
    // forced to collapse the degree-ten polynomial to degree twenty over Q.
    let gaussian = AlgebraicExtension::from_polynomial_with_embedding(
        parse!("u^2+1").to_polynomial(&Q, None),
        1,
    );
    assert_ne!(gaussian, AlgebraicExtension::complex(Q));

    let polynomial = parse!("x^10+(2+1i)*x^7-3").to_polynomial::<_, u16>(&gaussian, None);
    let simplified = Root::new(polynomial, 3)
        .unwrap()
        .simplify()
        .unwrap()
        .unwrap();

    assert_eq!(simplified.polynomial().degree(0), 20);
    assert_eq!(simplified.index(), 6);
}

#[test]
fn adjoin_and_convert() {
    let sqrt2 = AlgebraicExtension::from_polynomial_with_embedding(
        parse!("a^2-2").to_polynomial(&Q, None),
        1,
    );
    let sqrt3 = AlgebraicExtension::from_polynomial_with_embedding(
        parse!("b^2-3")
            .to_polynomial(&Q, None)
            .to_number_field(&sqrt2),
        1,
    );

    let (sqrt23, _, _) = sqrt2.adjoin_with_embedding(&sqrt3, Some(symbol!("gamma").into()));

    let poly = parse!("gamma").to_polynomial(&Q, None);
    let var = sqrt23.element_from_polynomial(poly);

    let var2 = sqrt23.mul(&var, &var);
    let e = sqrt23.element_to_atom(&var2);
    println!("{}", e);
    println!(
        "comparison: {}",
        (e - parse!("(sqrt(2) + sqrt(3))^2")).to_float(16)
    );
}

#[test]
fn adjoin_with_embedding() {
    for (sqrt2_embedding, sqrt3_embedding, expected_embedding) in
        [(0, 0, 0), (1, 0, 1), (0, 1, 2), (1, 1, 3)]
    {
        let sqrt2 = AlgebraicExtension::from_polynomial_with_embedding(
            parse!("a^2-2").to_polynomial(&Q, None),
            sqrt2_embedding,
        );
        let sqrt3 = AlgebraicExtension::from_polynomial_with_embedding(
            parse!("b^2-3")
                .to_polynomial(&Q, None)
                .to_number_field(&sqrt2),
            sqrt3_embedding,
        );

        let (sqrt23, r1, r2) = sqrt2.adjoin_with_embedding(&sqrt3, Some(symbol!("gamma").into()));

        assert_eq!(sqrt23.embedding.index(), Some(expected_embedding));
        assert_eq!(sqrt23.mul(&r1, &r1), sqrt23.nth(2.into()));
        assert_eq!(sqrt23.mul(&r2, &r2), sqrt23.nth(3.into()));
    }
}

#[test]
fn adjoin_with_all_embeddings_from_degree_one_extension() {
    let base = AlgebraicExtension::new(parse!("a-2").to_polynomial(&Q, None));
    let polynomial = parse!("b^3-3")
        .to_polynomial(&Q, None)
        .to_number_field(&base);
    let gamma = symbol!("gamma");
    let (extensions, old_generator, new_generator) =
        base.adjoin_with_all_embeddings(&polynomial, Some(gamma.into()));

    assert_eq!(extensions.len(), 3);
    for (embedding, extension) in extensions.iter().enumerate() {
        assert_eq!(extension.embedding().index(), Some(embedding));
        assert_eq!(
            extension.poly().get_vars_ref(),
            &[crate::poly::PolyVariable::from(gamma)]
        );
        assert_eq!(&old_generator, &extension.nth(2.into()));
        assert_eq!(extension.pow(&new_generator, 3), extension.nth(3.into()));
    }
}

#[test]
fn adjoin_with_complex_embedding() {
    for (sqrt2_embedding, i_embedding, expected_embedding) in
        [(0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3)]
    {
        let sqrt2 = AlgebraicExtension::from_polynomial_with_embedding(
            parse!("a^2-2").to_polynomial(&Q, None),
            sqrt2_embedding,
        );
        let i = AlgebraicExtension::from_polynomial_with_embedding(
            parse!("b^2+1")
                .to_polynomial(&Q, None)
                .to_number_field(&sqrt2),
            i_embedding,
        );

        let (extension, r1, r2) = sqrt2.adjoin_with_embedding(&i, Some(symbol!("gamma").into()));

        assert_eq!(extension.embedding.index(), Some(expected_embedding));
        assert_eq!(extension.mul(&r1, &r1), extension.nth(2.into()));
        assert_eq!(extension.mul(&r2, &r2), extension.neg(&extension.one()));
        let i = if i_embedding == 1 {
            r2
        } else {
            extension.neg(&r2)
        };
        assert_eq!(extension.imaginary_unit.get(), Some(&i));
    }
}

#[test]
fn algebraic_number_to_atom_complex() {
    let ring = AlgebraicExtension::complex(Q);

    let i = ring.generator();
    assert_eq!(ring.element_to_atom(&i), parse!("1𝑖"));

    let one_plus_i = ring.add(&ring.one(), &i);
    assert_eq!(ring.element_to_atom(&one_plus_i), parse!("1+1𝑖"));

    let ring = AlgebraicExtension::from_polynomial_with_embedding(
        parse!("a^2+1").to_polynomial(&Q, None),
        1,
    );
    let a = ring.element_from_polynomial(parse!("a").to_polynomial::<_, u16>(&Q, None));
    assert_eq!(ring.element_to_atom(&a), parse!("1𝑖"));
}

#[test]
fn algebraic_number_to_atom_binomial_root() {
    let ring = AlgebraicExtension::from_polynomial_with_embedding(
        parse!("a^3-2").to_polynomial(&Q, None),
        2,
    );

    let a_squared = ring.element_from_polynomial(parse!("a^2").to_polynomial::<_, u16>(&Q, None));
    assert_eq!(ring.element_to_atom(&a_squared), parse!("root(a^3-2,2)^2"));
}

#[test]
fn gcd_number_field() {
    let ring = parse!("a^3 + 3a^2 - 46*a + 1").to_polynomial(&Q, None);
    let ring = AlgebraicExtension::new(ring);

    let a = parse!("x^3-2x^2+(-2a^2+8a+2)x-a^2+11a-1")
        .to_polynomial::<_, u16>(&Q, None)
        .to_number_field(&ring);

    let b = parse!("x^3-2x^2-x+1")
        .to_polynomial(&Q, a.variables().clone())
        .to_number_field(&ring);

    let r = a.gcd(&b).from_number_field();

    let expected = parse!("-50/91+x-23/91*a-1/91*a^2").to_polynomial(&Q, a.variables().clone());
    assert_eq!(r, expected);
}

#[test]
fn galois() {
    for j in 1..10 {
        let _ = AlgebraicExtension::galois_field(Z2, j, symbol!("v1").into());
    }

    for i in PrimeIteratorU64::new(2).take(20) {
        for j in 1..10 {
            let _ = AlgebraicExtension::galois_field(Zp::new(i as u32), j, symbol!("v1").into());
        }
    }
}

fn check_extension_sampling<F>(field: &F, size: usize)
where
    F: UniformlySampleableRing,
{
    use rand::{SeedableRng, rngs::StdRng};

    for (nonzero, expected_count) in [(false, size), (true, size - 1)] {
        let mut rng = StdRng::seed_from_u64(42);
        let mut seen = HashSet::new();
        for _ in 0..1024 {
            let sample = if nonzero {
                field.sample_uniform_nonzero(&mut rng)
            } else {
                field.sample_uniform(&mut rng)
            };
            if nonzero {
                assert!(!field.is_zero(&sample));
            }
            seen.insert(sample);
        }
        assert_eq!(seen.len(), expected_count);
    }
}

#[test]
fn extension_sampling_nonzero_covers_the_whole_extension() {
    // Over GF(2), requiring every coefficient to be nonzero would produce
    // only 1 + a. Both 1 and a must also be reachable.
    let binary = AlgebraicExtension::new(parse!("a^2+a+1").to_polynomial::<_, u16>(&Z2, None));
    check_extension_sampling(&binary, 4);
    let odd = AlgebraicExtension::new(parse!("a^2+1").to_polynomial::<_, u16>(&Zp::new(3), None));
    check_extension_sampling(&odd, 9);
    check_extension_sampling(&AlgebraicExtension::trivial(Z2), 2);
}

#[test]
fn extension_sampling_nonzero_covers_nested_extensions_and_quotients() {
    let base = AlgebraicExtension::new(parse!("a^2+a+1").to_polynomial::<_, u16>(&Z2, None));
    let mut polynomial =
        MultivariatePolynomial::new(&base, None, Arc::new(vec![PolyVariable::Temporary(1)]));
    polynomial.append_monomial(base.generator(), &[0]);
    polynomial.append_monomial(base.one(), &[1]);
    polynomial.append_monomial(base.one(), &[2]);

    // b^2 + b + a defines GF(16) over GF(4).
    check_extension_sampling(&AlgebraicExtension::new(polynomial.clone()), 16);
    check_extension_sampling(&AlgebraicQuotient::new(polynomial.clone()), 16);

    let quotient = AlgebraicQuotient::new(base.poly().clone());
    check_extension_sampling(&quotient, 4);
    let polynomial = polynomial.map_coeff(|c| c.clone(), quotient);
    check_extension_sampling(&AlgebraicExtension::new(polynomial), 16);
}

#[test]
fn extension_sampling_preserves_explicit_coefficient_ranges() {
    use rand::{SeedableRng, rngs::StdRng};
    let field = AlgebraicExtension::new(parse!("a^2+a+1").to_polynomial::<_, u16>(&Z2, None));
    let mut rng = StdRng::seed_from_u64(43);
    for _ in 0..10 {
        assert_eq!(
            field.sample(&mut rng, &AlgebraicSamplingPolicy::new((1..=1).into())),
            field.add(&field.one(), &field.generator()),
        );
        assert_eq!(
            field.sample_nonzero(&mut rng, &AlgebraicSamplingPolicy::new((1..=1).into())),
            field.add(&field.one(), &field.generator()),
        );
        assert!(
            field.is_zero(&field.sample(&mut rng, &AlgebraicSamplingPolicy::new((0..=0).into())))
        );
    }

    let rational = AlgebraicExtension::new(parse!("a^2-2").to_polynomial::<_, u16>(&Q, None));
    assert_eq!(
        rational.sample(
            &mut rng,
            &AlgebraicSamplingPolicy::new(Integer::from(2)..=Integer::from(2)),
        ),
        rational.mul(
            &rational.nth(2.into()),
            &rational.add(&rational.one(), &rational.generator())
        ),
    );
}

#[test]
fn extension_sampling_degree_bounds() {
    use rand::{SeedableRng, rngs::StdRng};
    let field = AlgebraicExtension::galois_field(Z2, 3, PolyVariable::Temporary(0));
    let quotient = AlgebraicQuotient::new(field.poly().clone());
    let mut rng = StdRng::seed_from_u64(44);

    for (bound, expected) in [
        (Some(0), field.one()),
        (Some(1), field.add(&field.one(), &field.generator())),
        (
            Some(usize::MAX),
            field.add(
                &field.add(&field.one(), &field.generator()),
                &field.pow(&field.generator(), 2),
            ),
        ),
        (
            None,
            field.add(
                &field.add(&field.one(), &field.generator()),
                &field.pow(&field.generator(), 2),
            ),
        ),
    ] {
        let policy = AlgebraicSamplingPolicy {
            max_degree: bound,
            coefficient: (1..2).into(),
        };
        assert_eq!(field.sample(&mut rng, &policy), expected);
        assert_eq!(quotient.sample(&mut rng, &policy), expected);
    }

    let policy = field.sampling_policy(Some(0), field.poly.ring().uniform_sampling_policy());
    for _ in 0..32 {
        assert!(field.sample(&mut rng, &policy).poly().is_constant());
        assert!(field.is_one(&field.sample_nonzero(&mut rng, &policy)));
    }
}

#[test]
fn extension_sampling_nested_degree_bounds_are_local() {
    use rand::{SeedableRng, rngs::StdRng};
    let base = AlgebraicExtension::new(parse!("a^2+a+1").to_polynomial::<_, u16>(&Z2, None));
    let mut polynomial =
        MultivariatePolynomial::new(&base, None, Arc::new(vec![PolyVariable::Temporary(1)]));
    polynomial.append_monomial(base.generator(), &[0]);
    polynomial.append_monomial(base.one(), &[1]);
    polynomial.append_monomial(base.one(), &[2]);
    let field = AlgebraicExtension::new(polynomial);
    let mut rng = StdRng::seed_from_u64(45);
    let policy = AlgebraicSamplingPolicy {
        max_degree: Some(0),
        coefficient: AlgebraicSamplingPolicy::new((1..=1).into()),
    };
    assert_eq!(
        field.sample_nonzero(&mut rng, &policy),
        field.constant(base.add(&base.one(), &base.generator())),
    );
}

#[test]
fn extension_nth_preserves_integer_embedding() {
    let field = AlgebraicExtension::galois_field(Zp::new(3), 2, PolyVariable::Temporary(0));
    for n in [-10, -1, 0, 1, 2, 3, 4, 10] {
        assert_eq!(
            field.nth(n.into()),
            field.constant(field.poly.ring().nth(n.into()))
        );
        assert_eq!(
            field.nth((n + 1).into()),
            field.add(&field.nth(n.into()), &field.one())
        );
    }
}

#[test]
fn extension_sampling_nonzero_through_a_fraction_field() {
    use crate::domains::rational::FractionField;
    let base = FractionField::new(Z2);
    let polynomial = parse!("a^2+a+1")
        .to_polynomial::<_, u16>(&Z2, None)
        .map_coeff(|c| base.to_element_numerator(*c), base.clone());
    check_extension_sampling(&AlgebraicExtension::new(polynomial), 4);
}

#[test]
fn norm() {
    let a = parse!("z^4+z^3+(2+a-a^2)z^2+(1+a^2-2a^3)z-2").to_polynomial::<_, u8>(&Q, None);
    let f = parse!("a^4-3").to_polynomial::<_, u16>(&Q, None);
    let f = AlgebraicExtension::new(f);
    let norm = a.to_number_field(&f).norm();

    let res = parse!("16-32*z-64*z^2-64*z^3-52*z^4-40*z^5-132*z^6-24*z^7-50*z^8+120*z^9+66*z^10+92*z^11+47*z^12+32*z^13+14*z^14+4*z^15+z^16")
    .to_polynomial::<_, u8>(&Q, a.variables().clone());

    assert_eq!(norm, res);
}

#[test]
fn extend() {
    let a = parse!("x^2-2").to_polynomial(&Q, None);
    let ae = AlgebraicExtension::new(a);

    let b = parse!("y^2-3").to_polynomial(&Q, None).to_number_field(&ae);

    let (c, rep1, rep2) = ae.adjoin(&b, None);

    let rf = parse!("1-10*y^2+y^4").to_polynomial(&Q, None);

    assert_eq!(c.poly.as_ref(), &rf);

    let r1 = parse!("-9/2y+1/2y^3").to_polynomial::<_, u16>(&Q, None);
    assert_eq!(rep1.poly, r1);

    let r2 = parse!("11/2*y-1/2*y^3").to_polynomial::<_, u16>(&Q, None);
    assert_eq!(rep2.poly, r2);
}

#[test]
fn simplify() {
    let poly =
        AlgebraicExtension::new(parse!("13-16v1+28v1^2+2v1^3+11v1^4+v1^6").to_polynomial(&Q, None));

    let a = poly.element_from_polynomial(
        parse!("-295/1882 -2693/1882v1 -237/1882v1^2 -385/941v1^3 -9/1882v1^4  -33/941v1^5")
            .to_polynomial::<_, u16>(&Q, None),
    );

    let r = poly.simplify(&a);
    let res = parse!("1+v1+v1^2").to_polynomial(&Q, None);
    assert_eq!(*r.poly, res);
}

#[test]
fn simplify_preserves_the_selected_embedding() {
    for (source_embedding, expected_embedding) in [(0, 1), (1, 1), (2, 0), (3, 0)] {
        let field = AlgebraicExtension::from_polynomial_with_embedding(
            parse!("x^4-2").to_polynomial(&Q, None),
            source_embedding,
        );
        let generator = field.generator();
        let squared = field.mul(&generator, &generator);
        let simplified = field.simplify(&squared);

        assert_eq!(simplified.poly(), &parse!("x^2-2").to_polynomial(&Q, None));
        assert_eq!(simplified.embedding().index(), Some(expected_embedding));
    }
}

#[test]
fn certified_ball_determines_real_sign() {
    let field = AlgebraicExtension::from_polynomial_with_embedding(
        parse!("x^2-2").to_polynomial(&Q, None),
        1,
    );
    let generator = field.generator();
    let negative_generator = field.neg(&generator);

    assert_eq!(field.is_positive_real(&generator), Ok(true));
    assert_eq!(field.is_positive_real(&negative_generator), Ok(false));
    assert_eq!(field.has_positive_real_part(&generator), Ok(true));
    assert_eq!(field.try_sign(&generator), Ok(Ordering::Greater));
    assert_eq!(
        field.try_cmp(&negative_generator, &generator),
        Ok(Ordering::Less)
    );

    let complex = AlgebraicExtension::complex(Q);
    assert!(complex.try_sign(&complex.generator()).is_err());
    assert!(
        complex
            .try_cmp(&complex.generator(), &complex.generator())
            .is_err()
    );
}

#[test]
fn try_div() {
    let extension = AlgebraicExtension::new(parse!("v1^3-2v1+3").to_polynomial(&Z, None));

    let f1 = extension.element_from_polynomial(parse!("v1^2-2").to_polynomial(&Z, None));
    let f2 = extension.element_from_polynomial(parse!("v1-5").to_polynomial(&Z, None));
    let prod = extension.mul(&f1, &f2);

    assert_eq!(extension.try_div(&prod, &f2).unwrap(), f1);
    assert_eq!(extension.try_div(&prod, &f1).unwrap(), f2);
    assert!(extension.try_div(&f2, &f1).is_none());
}

#[test]
fn univariate_divisibility_clears_number_field_denominators() {
    let field = AlgebraicExtension::new(parse!("a^3+a/2-2").to_polynomial(&Q, None));
    let factor_1 = parse!("x^2+a*x+1")
        .to_polynomial::<_, u16>(&Q, None)
        .to_number_field(&field);
    let factor_2 = parse!("2*x+a/3")
        .to_polynomial::<_, u16>(&Q, Some(factor_1.variables().clone()))
        .to_number_field(&field);
    let not_a_factor = parse!("x+a+1")
        .to_polynomial::<_, u16>(&Q, Some(factor_1.variables().clone()))
        .to_number_field(&field);
    let product = &factor_1 * &factor_2;

    assert!(<AlgebraicExtension<Q> as crate::poly::gcd::PolynomialGCD<
        u16,
    >>::divides_exact(&product, &factor_1));
    assert!(<AlgebraicExtension<Q> as crate::poly::gcd::PolynomialGCD<
        u16,
    >>::divides_exact(&product, &factor_2));
    assert!(
        !<AlgebraicExtension<Q> as crate::poly::gcd::PolynomialGCD<u16>>::divides_exact(
            &product,
            &not_a_factor,
        )
    );
}

#[test]
fn polynomial_power_variable_print() {
    let reciprocal = MultivariatePolynomial::<_, u16>::new(
        &Q,
        None,
        Arc::new(vec![PolyVariable::Power(parse!("x^-1"))]),
    )
    .monomial(1.into(), vec![1]);
    assert_eq!(reciprocal.to_string(), "1/x");

    let radical = parse!("sqrt(2)");
    let mut modulus = MultivariatePolynomial::<_, u16>::new(
        &Q,
        None,
        Arc::new(vec![PolyVariable::Power(radical)]),
    );
    let linear = modulus.monomial(1.into(), vec![1]);
    assert_eq!(linear.to_string(), "2^(1/2)");
    let mixed = &linear + &modulus.monomial(1.into(), vec![2]);
    let mixed_text = mixed.to_string();
    assert!(mixed_text.contains("(2^(1/2))^2"), "{mixed_text}");
    assert!(
        mixed_text.contains("2^(1/2)+") || mixed_text.ends_with("+2^(1/2)"),
        "{mixed_text}"
    );
    modulus.append_monomial((-2).into(), &[0]);
    modulus.append_monomial(1.into(), &[2]);
    let field = AlgebraicExtension::new(modulus);
    let text = field.to_string();
    assert!(text.contains("(2^(1/2))^2"), "{text}");
}

// Count entry into the two containment/discovery paths, per test thread. A warm
// coefficient must skip both, including AlgebraicContext's field discovery.
thread_local! {
    pub(super) static ROOT_DISCOVERIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[test]
fn imaginary_unit_cache_composite_coefficients_and_clones() {
    use crate::coefficient::{Coefficient, CoefficientView};
    use std::hash::{DefaultHasher, Hash, Hasher};

    let field = AlgebraicExtension::from_polynomial_with_embedding(
        parse!("t^4-2*t^2+9").to_polynomial(&Q, None),
        3, // sqrt(2) + i
    );
    let clone = field.clone();
    let independent =
        AlgebraicExtension::from_polynomial_with_embedding(field.poly.as_ref().clone(), 3);
    let hash = |field: &AlgebraicExtension<Q>| {
        let mut hasher = DefaultHasher::new();
        field.hash(&mut hasher);
        hasher.finish()
    };
    let before = hash(&field);
    assert!(clone.imaginary_unit.get().is_none());
    ROOT_DISCOVERIES.set(0);
    let i = field.imaginary_unit().unwrap();
    assert_eq!(ROOT_DISCOVERIES.get(), 1);
    assert_eq!(field.pow(&i, 2), field.nth((-1).into()));
    assert!(Arc::ptr_eq(&field.imaginary_unit, &clone.imaginary_unit));
    assert_eq!(clone.imaginary_unit.get(), Some(&i));
    assert_eq!(field, independent);
    assert_eq!(hash(&field), before);
    assert_eq!(hash(&independent), before);
    assert!(independent.imaginary_unit.get().is_none());

    let mut context = AlgebraicContext::new(clone.clone());
    ROOT_DISCOVERIES.set(0);
    for (text, large) in [
        ("2+3i", false),
        ("-7/3-5/2i", false),
        (
            "123456789012345678901234567890+987654321098765432109876543210i",
            true,
        ),
        (
            "-234567890123456789012345678901/7-876543210987654321098765432109/11i",
            true,
        ),
    ] {
        let atom = parse!(text);
        let AtomView::Num(number) = atom.as_view() else {
            panic!("expected number")
        };
        let view = number.get_coeff_view();
        assert_eq!(matches!(view, CoefficientView::Large(..)), large);
        let Coefficient::Complex(value) = view.to_owned() else {
            panic!("expected complex")
        };
        let expected = field.add(
            &field.constant(value.re),
            &field.mul(&field.constant(value.im), &i),
        );
        assert_eq!(
            clone.try_element_from_coefficient_view(view).unwrap(),
            expected
        );
        assert_eq!(
            clone.try_element_from_coefficient(view.to_owned()).unwrap(),
            expected
        );
        context.extend(atom.as_view()).unwrap();
        assert_eq!(context.image(&atom), Some(&expected));
    }
    assert_eq!(clone.imaginary_unit().unwrap(), i);
    assert_eq!(ROOT_DISCOVERIES.get(), 0);
    assert!(Arc::ptr_eq(
        &context.field.imaginary_unit,
        &field.imaginary_unit
    ));
}

#[test]
fn imaginary_unit_cache_conjugates_and_embedding_changes() {
    for index in 0..4 {
        let field = AlgebraicExtension::from_polynomial_with_embedding(
            parse!("t^4-2*t^2+9").to_polynomial(&Q, None),
            index,
        );
        let i = field.imaginary_unit().unwrap();
        assert_eq!(
            field
                .root_index_of_element(
                    &i,
                    &AlgebraicExtension::imaginary_unit_defining_polynomial()
                )
                .unwrap(),
            1
        );
        assert_eq!(field.clone().imaginary_unit().unwrap(), i);
    }

    // Detach even an empty cache, before either conjugate is warmed.
    let positive = AlgebraicExtension::complex(Q);
    let mut negative = positive.clone();
    negative.set_embedding(AlgebraicEmbedding::Indexed(0));
    assert!(!Arc::ptr_eq(
        &positive.imaginary_unit,
        &negative.imaginary_unit
    ));
    assert_eq!(positive.imaginary_unit().unwrap(), positive.generator());
    assert_eq!(
        negative.imaginary_unit().unwrap(),
        negative.neg(&negative.generator())
    );
    let mut changed = positive.clone();
    changed.set_embedding(AlgebraicEmbedding::Isolated(Arc::new(
        positive
            .poly
            .to_univariate_from_univariate(0)
            .root(0)
            .unwrap(),
    )));
    assert!(changed.imaginary_unit.get().is_none());
    assert_eq!(
        changed.imaginary_unit().unwrap(),
        changed.neg(&changed.generator())
    );
    assert!(matches!(
        changed.embedding(),
        AlgebraicEmbedding::Isolated(_)
    ));
    assert_eq!(positive.imaginary_unit().unwrap(), positive.generator());
}

#[test]
fn imaginary_unit_cache_seed_and_transport_isolated_extensions() {
    let real = AlgebraicExtension::from_polynomial_with_embedding(
        parse!("t^2-2").to_polynomial(&Q, None),
        1,
    );
    assert!(real.imaginary_unit().is_err());
    assert!(real.imaginary_unit.get().is_none());
    let polynomial = AlgebraicExtension::imaginary_unit_defining_polynomial();
    let (composite, _, i) = real
        .with_adjoined_rational_root_data(&polynomial, 1)
        .unwrap();
    assert_eq!(composite.imaginary_unit.get(), Some(&i));
    assert!(matches!(
        composite.embedding(),
        AlgebraicEmbedding::Isolated(_)
    ));
    ROOT_DISCOVERIES.set(0);
    assert_eq!(composite.imaginary_unit().unwrap(), i);
    assert_eq!(
        composite
            .with_adjoined_rational_root_data(&polynomial, 1)
            .unwrap()
            .2,
        i
    );
    assert_eq!(ROOT_DISCOVERIES.get(), 0);

    let (larger, old_generator, _) = composite
        .with_adjoined_rational_root_data(&parse!("u^2-3").to_polynomial(&Q, None), 1)
        .unwrap();
    let transported = AlgebraicContext::transport_element(&i, &larger, &old_generator);
    assert_eq!(larger.imaginary_unit.get(), Some(&transported));
    assert_eq!(larger.pow(&transported, 2), larger.nth((-1).into()));
    assert!(matches!(
        larger.embedding(),
        AlgebraicEmbedding::Isolated(_)
    ));
    ROOT_DISCOVERIES.set(0);
    assert_eq!(larger.imaginary_unit().unwrap(), transported);
    assert_eq!(ROOT_DISCOVERIES.get(), 0);

    // Coprime binomial adjunction uses a product primitive element.
    let gaussian = AlgebraicExtension::complex(Q);
    let i = gaussian.imaginary_unit().unwrap();
    let (product, old_generator, _) = gaussian
        .with_adjoined_rational_root_data(&parse!("u^3-2").to_polynomial(&Q, None), 0)
        .unwrap();
    assert_eq!(
        product.imaginary_unit.get(),
        Some(&AlgebraicContext::transport_element(
            &i,
            &product,
            &old_generator
        ))
    );
    assert!(matches!(
        product.embedding(),
        AlgebraicEmbedding::Isolated(_)
    ));
    assert!(real.imaginary_unit().is_err());
}

#[test]
fn imaginary_unit_cache_indexed_adjunctions() {
    let base = AlgebraicExtension::trivial(Q);
    let polynomial = parse!("u^2+1").to_polynomial(&base, None);
    let (fields, _, generator) = base.adjoin_with_all_embeddings(&polynomial, None);
    assert!(!Arc::ptr_eq(
        &fields[0].imaginary_unit,
        &fields[1].imaginary_unit
    ));
    assert_eq!(
        fields[0].imaginary_unit.get(),
        Some(&fields[0].neg(&generator))
    );
    assert_eq!(fields[1].imaginary_unit.get(), Some(&generator));

    let gaussian = &fields[0];
    let i = gaussian.imaginary_unit().unwrap();
    let polynomial = parse!("v^2-2").to_polynomial(gaussian, None);
    let (extensions, old_generator, _) =
        gaussian.adjoin_with_all_embeddings(&polynomial, Some(symbol!("cache_gamma").into()));
    assert!(!Arc::ptr_eq(
        &extensions[0].imaginary_unit,
        &extensions[1].imaginary_unit
    ));
    for field in extensions {
        let expected = AlgebraicContext::transport_element(&i, &field, &old_generator);
        assert_eq!(field.imaginary_unit.get(), Some(&expected));
        assert_eq!(
            field
                .root_index_of_element(
                    &expected,
                    &AlgebraicExtension::imaginary_unit_defining_polynomial()
                )
                .unwrap(),
            1
        );
    }
}

#[test]
fn imaginary_unit_cache_renames_with_context() {
    let mut context = AlgebraicContext::from_atom(parse!("sqrt(2)+1i").as_view()).unwrap();
    let original = context.field.clone();
    let original_i = original.imaginary_unit().unwrap();
    let variable = symbol!("imaginary_cache_renamed").into();
    let mut expected = original_i.clone();
    expected
        .poly
        .rename_variable(&original.poly.get_vars_ref()[0], &variable);
    context.rename_generator(variable);
    assert_eq!(context.field.imaginary_unit.get(), Some(&expected));
    assert!(!Arc::ptr_eq(
        &original.imaginary_unit,
        &context.field.imaginary_unit
    ));
    ROOT_DISCOVERIES.set(0);
    context.extend(parse!("3+4i").as_view()).unwrap();
    assert_eq!(context.field.imaginary_unit().unwrap(), expected);
    assert_eq!(original.imaginary_unit().unwrap(), original_i);
    assert_eq!(ROOT_DISCOVERIES.get(), 0);
}

#[test]
fn imaginary_unit_cache_real_fields_reject_complex_coefficients() {
    for polynomial in ["t", "t^2-2", "t^2+2"] {
        let field = AlgebraicExtension::new(parse!(polynomial).to_polynomial(&Q, None));
        assert!(parse!("2+3i").as_view().to_algebraic(&field).is_err());
        assert!(field.imaginary_unit.get().is_none());
        assert_eq!(
            parse!("2").as_view().to_algebraic(&field).unwrap(),
            field.nth(2.into())
        );
    }
}

#[test]
fn imaginary_unit_cache_concurrent_clones() {
    let field = AlgebraicExtension::from_polynomial_with_embedding(
        parse!("t^4-2*t^2+9").to_polynomial(&Q, None),
        3,
    );
    let barrier = std::sync::Barrier::new(4);
    let images = std::thread::scope(|scope| {
        (0..4)
            .map(|_| {
                let field = field.clone();
                let barrier = &barrier;
                scope.spawn(move || {
                    barrier.wait();
                    field.imaginary_unit().unwrap()
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|thread| thread.join().unwrap())
            .collect::<Vec<_>>()
    });
    for image in images {
        assert_eq!(field.imaginary_unit.get(), Some(&image));
    }
}
