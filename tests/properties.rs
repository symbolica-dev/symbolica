use symbolica::{
    atom::{Atom, AtomCore, Symbol},
    coefficient::Coefficient,
    domains::{
        finite_field::{FiniteFieldCore, Zp64},
        float::Complex,
        rational::Rational,
    },
    function,
    id::ConditionResult::{self, False, Inconclusive, True},
    parse, symbol,
};

#[test]
fn properties_distinguish_false_from_unknown() {
    let x = parse!("property_unknown");
    assert_eq!(x.is_real(), Inconclusive);
    assert_eq!(x.is_integer(), Inconclusive);
    assert_eq!(x.is_scalar(), Inconclusive);
    assert_eq!(x.is_positive(), Inconclusive);
    assert_eq!(x.is_nonnegative(), Inconclusive);
    assert_eq!(!x.is_real(), Inconclusive);

    let i = Atom::num(Complex::new(Rational::from(0), Rational::from(1)));
    assert_eq!(i.is_real(), False);
    assert_eq!(i.is_integer(), False);
    assert_eq!(i.is_positive(), False);
    assert_eq!(i.is_nonnegative(), False);
    assert_eq!(i.is_scalar(), True);
    assert_eq!(Atom::num((1, 2)).is_integer(), False);
    assert_eq!(Atom::num(-2).is_integer(), True);
    assert_eq!(Atom::num(-2).is_positive(), False);
    assert_eq!(Atom::num(-2).is_nonnegative(), False);
    assert_eq!(Atom::num(2).is_positive(), True);
    assert_eq!(Atom::num(0).is_positive(), False);
    assert_eq!(Atom::num(0).is_nonnegative(), True);

    let real = Atom::var(symbol!("property_real"; Real));
    assert_eq!(real.is_real(), True);
    assert_eq!(real.is_integer(), Inconclusive);
    assert_eq!(real.is_positive(), Inconclusive);
    assert_eq!((&real + &i).is_real(), False);
    assert_eq!((&x + &i).is_real(), Inconclusive);
    assert_eq!((&i * &x).is_real(), Inconclusive);
}

#[test]
fn positivity_distinguishes_squares_from_strict_bounds() {
    let r = Atom::var(symbol!("property_sign_real"; Real));
    let p = Atom::var(symbol!("property_sign_positive"; Positive));
    let square = r.pow(2);
    assert_eq!(square.is_nonnegative(), True);
    assert_eq!(square.is_positive(), Inconclusive);
    assert_eq!((-&square).is_positive(), False);
    assert_eq!((-&square).is_nonnegative(), Inconclusive);
    assert_eq!((&square + 1).is_positive(), True);
    assert_eq!((&square * &p).is_positive(), Inconclusive);
    assert_eq!((&square * &p).is_nonnegative(), True);
    assert_eq!((-&p).is_positive(), False);
    assert_eq!((-&p).is_nonnegative(), False);
    assert_eq!(((-&p) * (-&p)).is_positive(), True);
    assert_eq!(p.pow(-2).is_positive(), True);
    assert_eq!(r.abs().is_nonnegative(), True);
    assert_eq!(r.abs().is_positive(), Inconclusive);
    assert!(
        !r.abs()
            .as_view()
            .has_attributes_of(symbol!("property_positive_restriction"; Positive))
    );
    assert_eq!(square.pow(&r).is_real(), True);
    assert_eq!(square.pow(&r).is_nonnegative(), True);
    assert_eq!(square.pow(&r).is_positive(), Inconclusive);
}

#[test]
fn integer_powers_require_nonnegative_integer_exponents() {
    let n = Atom::var(symbol!("property_integer_base"; Integer));
    let m = Atom::var(symbol!("property_integer_exponent"; Integer));
    assert_eq!(n.pow(2).is_integer(), True);
    assert_eq!(n.pow(-1).is_integer(), Inconclusive);
    assert_eq!(n.pow(&m).is_integer(), Inconclusive);
    assert_eq!(n.pow(m.pow(2)).is_integer(), True);
    assert_eq!((n + (1, 2)).is_integer(), False);
}

#[test]
fn functions_and_branches_do_not_negate_sufficient_conditions() {
    let x = parse!("property_function_unknown");
    let r = Atom::var(symbol!("property_function_real"; Real));
    let real_fun = symbol!("property_real_fun"; Real);
    assert_eq!(function!(real_fun, &x).is_real(), True);
    assert_eq!(function!(Symbol::SIN, &r).is_real(), True);
    assert_eq!(function!(Symbol::SIN, &x).is_real(), Inconclusive);
    assert_eq!(function!(Symbol::LOG, &r).is_real(), Inconclusive);
    assert_eq!(function!(Symbol::CONJ, &x).is_real(), Inconclusive);
    assert_eq!(function!(Symbol::ABS, &x).is_real(), True);
    assert_eq!(function!(Symbol::ABS, &x).is_positive(), Inconclusive);

    let i = Atom::num(Complex::new(Rational::from(0), Rational::from(1)));
    assert_eq!(function!(Symbol::IF, &x, 1, &r).is_real(), True);
    assert_eq!(function!(Symbol::IF, &x, 1, &i).is_real(), Inconclusive);
    assert_eq!(function!(Symbol::IF, &x, &i, &i + 1).is_real(), False);
    assert_eq!(function!(Symbol::IF, &x, 0, 1).is_positive(), Inconclusive);
    assert_eq!(function!(Symbol::IF, &x, 0, 1).is_nonnegative(), True);
}

#[test]
fn normalization_still_uses_proven_weak_inequalities() {
    let r = Atom::var(symbol!("property_normalize_real"; Real));
    let square = r.pow(2);
    assert_eq!(square.abs(), square);
    assert_eq!(square.pow((1, 2)), r.abs());
    assert_eq!(r.abs().pow(2), square);
    assert_eq!(r.pow(3).conj(), r.pow(3));
    let x = parse!("property_normalize_unknown");
    assert_ne!(x.pow((1, 2)).conj(), x.pow((1, 2)));
}

#[test]
fn structural_checks_and_truth_conversions_remain_explicit() {
    let x = parse!("property_structural_unknown");
    assert!(!x.is_zero());
    assert!(!x.is_one());
    assert!(!x.is_constant());
    assert!(x.pow(-1).is_finite()); // No explicit infinity; does not exclude poles.
    assert_eq!(Option::<bool>::from(True), Some(true));
    assert_eq!(Option::<bool>::from(False), Some(false));
    assert_eq!(Option::<bool>::from(Inconclusive), None);
    assert_eq!(True & Inconclusive, Inconclusive);
    assert_eq!(False & Inconclusive, False);
    assert_eq!(True | Inconclusive, True);
    assert_eq!(False | Inconclusive, Inconclusive);
    assert_eq!(ConditionResult::from(false), False);
}

#[test]
fn coefficient_domains_are_not_confused_with_real_or_integer_membership() {
    let field = Zp64::new(17);
    let element = field.to_element(5);
    let finite_field = Atom::num(Coefficient::from_finite_field(field, element));
    let parameter = symbol!("property_coefficient_parameter");
    let polynomial = (Atom::var(parameter) + 1).set_coefficient_ring(parameter);
    for atom in [
        finite_field,
        polynomial,
        Atom::num(Coefficient::Indeterminate),
        Atom::num(Coefficient::positive_infinity()),
    ] {
        assert_eq!(atom.is_real(), Inconclusive);
        assert_eq!(atom.is_integer(), Inconclusive);
        assert_eq!(atom.is_positive(), Inconclusive);
    }
    assert_eq!(Atom::num(2).to_float(20).is_integer(), True);
    assert_eq!(Atom::num((1, 2)).to_float(20).is_integer(), False);
    assert_eq!(Atom::num(-2).to_float(20).is_nonnegative(), False);
}
