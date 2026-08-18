//! Tests for algebraic extensions, elements, contexts, and roots.

use super::*;

#[test]
fn to_alg() {
    let a = crate::parse!("sqrt(2)+1");
    let ext = a.as_view().embedding_field().unwrap();
    let alg = a.as_view().to_algebraic(&ext).unwrap();
    let generator = ext.element_from_polynomial(ext.poly.one().mul_exp(&[1]));
    assert_eq!(ext.embedding, 1);
    assert_eq!(alg, ext.add(&generator, &ext.one()));

    let b = crate::parse!("sqrt(2)+sqrt(3)");
    let ext = b.as_view().embedding_field().unwrap();
    let alg = b.as_view().to_algebraic(&ext).unwrap();
    let generator = ext.element_from_polynomial(ext.poly.one().mul_exp(&[1]));
    assert_eq!(ext.embedding, 3);
    assert_eq!(alg, generator);

    let b = crate::parse!("sqrt(2)+sqrt(3)+sqrt(6)");
    let ext = b.as_view().embedding_field().unwrap();
    let alg = b.as_view().to_algebraic(&ext).unwrap();
    assert_eq!(ext.embedding, 3);
    assert_eq!(ext.is_positive(&alg), Ok(true));

    let b = crate::parse!("sqrt(3+sqrt(2))+1");
    let ext = b.as_view().embedding_field().unwrap();
    let alg = b.as_view().to_algebraic(&ext).unwrap();
    let generator = ext.element_from_polynomial(ext.poly.one().mul_exp(&[1]));
    assert_eq!(ext.embedding, 3);
    assert_eq!(alg, ext.add(&generator, &ext.one()));

    let b = crate::parse!("2^(2/3)");
    let ext = b.as_view().embedding_field().unwrap();
    let alg = b.as_view().to_algebraic(&ext).unwrap();
    assert_eq!(ext.embedding, 2);
    assert_eq!(ext.pow(&alg, 3), ext.nth(4.into()));

    let b = crate::parse!("root(1-10*x^2+x^4,3)+1");
    let ext = b.as_view().embedding_field().unwrap();
    let alg = b.as_view().to_algebraic(&ext).unwrap();
    let generator = ext.element_from_polynomial(ext.poly.one().mul_exp(&[1]));
    assert_eq!(ext.embedding, 3);
    assert_eq!(alg, ext.add(&generator, &ext.one()));
    let polynomial = b.as_view().try_to_polynomial::<_, u16>(&ext, None).unwrap();
    assert_eq!(polynomial.nvars(), 0);
    assert_eq!(polynomial.get_constant(), alg);

    let b = crate::parse!("root(x^3-2,0)+sqrt(3)");
    let ext = b.as_view().embedding_field().unwrap();
    let complex_cube_root = crate::parse!("root(x^3-2,0)")
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
    assert_eq!(ext.embedding, 1);
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

    let expression = crate::parse!("x+sqrt(2)+sqrt(3)");
    let (context, polynomial) = expression
        .as_view()
        .to_polynomial_in_algebraic_extension::<u16>(symbol!("x"))
        .unwrap()
        .unwrap();
    assert_eq!(context.field().embedding, 3);

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
        .as_view()
        .to_rational_polynomial_in_algebraic_extension::<u16>(symbol!("x"))
        .unwrap()
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

        assert_eq!(sqrt23.embedding, expected_embedding);
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
        assert_eq!(extension.embedding(), embedding);
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

        assert_eq!(extension.embedding, expected_embedding);
        assert_eq!(extension.mul(&r1, &r1), extension.nth(2.into()));
        assert_eq!(extension.mul(&r2, &r2), extension.neg(&extension.one()));
    }
}

#[test]
fn algebraic_number_to_atom_complex() {
    let ring = AlgebraicExtension::complex(Q);

    let i = ring.element_from_polynomial(parse!("𝑖").to_polynomial::<_, u16>(&Q, None));
    assert_eq!(ring.element_to_atom(&i), parse!("1𝑖"));

    let one_plus_i = ring.element_from_polynomial(parse!("1+𝑖").to_polynomial::<_, u16>(&Q, None));
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
    for (source_embedding, expected_embedding) in [(0, 1), (1, 0), (2, 0), (3, 1)] {
        let field = AlgebraicExtension::from_polynomial_with_embedding(
            parse!("x^4-2").to_polynomial(&Q, None),
            source_embedding,
        );
        let generator = field.generator();
        let squared = field.mul(&generator, &generator);
        let simplified = field.simplify(&squared);

        assert_eq!(simplified.poly(), &parse!("x^2-2").to_polynomial(&Q, None));
        assert_eq!(simplified.embedding(), expected_embedding);
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
