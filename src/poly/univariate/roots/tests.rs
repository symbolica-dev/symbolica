//! Tests for certified root isolation and caching.

use super::{CachedRoot, RootCache, RootLocation, RootMultiset, UnivariatePolynomial, root_cache};
use std::{
    cmp::Ordering,
    sync::{
        Arc, Barrier, RwLock,
        atomic::{AtomicUsize, Ordering as AtomicOrdering},
    },
    time::Duration,
};

use crate::{
    atom::AtomCore,
    domains::{
        float::{Complex, F64, Float, FloatField},
        integer::Integer,
        rational::{Q, Rational},
    },
    parse,
    poly::{PolyVariable, univariate::IsolatedRoot},
};

#[test]
fn isolate() {
    let p =
    parse!("-13559717115*x^6+624134407779*x^7+-13046815434285*x^8+163110612017313*x^9+-1347733455544188*x^10+7635969738026784*x^11+-29444295941654904*x^12+71604709665043392*x^13+-77045857071990336*x^14+-99619711608972096*x^15+375578692434494208*x^16+66256662107418624*x^17+-1548072112541055488*x^18+800263217632600064*x^19+4816054475648851968*x^20+-4271696436901249024*x^21+-12066471810013724672*x^22+10894783995791278080*x^23+28270081588804452352*x^24+-17402041731641245696*x^25+-56047633173904883712*x^26+8535267319469834240*x^27+82086860869945262080*x^28+30788799964221800448*x^29+-66898313364436418560*x^30+-66318040948916879360*x^31+44159548067414016*x^32+31084367995645984768*x^33+20957883496015069184*x^34+6860635897973440512*x^35+1254041389990150144*x^36+123004564822556672*x^37+5066549580791808*x^38")
    .to_polynomial::<_, u32>(&Q, None)
    .to_univariate_from_univariate(0);

    let roots = p.isolate_real_root_intervals();

    assert_eq!(
        roots,
        vec![
            ((-7, 1).into(), (-7, 2).into(), 6),
            ((-1, 1).into(), (-1, 1).into(), 3),
            ((0, 1).into(), (0, 1).into(), 6),
            ((1, 8).into(), (3, 16).into(), 3),
            ((15, 64).into(), (9, 32).into(), 1),
            ((3, 4).into(), (1, 1).into(), 1),
        ],
    );

    let ref_roots: Vec<_> = roots
        .into_iter()
        .map(|x| {
            let r = p.refine_root_interval((x.0, x.1), &(1, 1000).into());
            (r.0, r.1, x.2)
        })
        .collect();

    assert_eq!(
        ref_roots,
        vec![
            ((-3955, 1024).into(), (-987, 256).into(), 6),
            ((-1, 1).into(), (-1, 1).into(), 3),
            ((0, 1).into(), (0, 1).into(), 6),
            ((723, 4096).into(), (181, 1024).into(), 3),
            ((1023, 4096).into(), (2049, 8192).into(), 1),
            ((995, 1024).into(), (249, 256).into(), 1),
        ],
    );
}

#[test]
fn complex_roots() {
    let p = parse!("x^10+9x^7+4x^3+2x+1")
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0);
    let pc = p.approximate_roots::<F64>(10000, &1e-8.into()).unwrap();
    assert!(pc[0].0.re < 2f64.into());
    assert!(pc[9].0.re > 1f64.into());
}

fn assert_pairwise_isolated(roots: &[IsolatedRoot]) {
    for i in 0..roots.len() {
        for j in i + 1..roots.len() {
            assert!(roots[i].enclosure().is_disjoint(roots[j].enclosure()));
        }
    }
}

fn is_real(root: &IsolatedRoot) -> bool {
    matches!(root.location, Some(RootLocation::Real | RootLocation::Zero))
}

fn is_imaginary(root: &IsolatedRoot) -> bool {
    matches!(
        root.location,
        Some(RootLocation::Imaginary | RootLocation::Zero)
    )
}

fn assert_complex_roots_canonical(roots: &[IsolatedRoot]) {
    for pair in roots.windows(2) {
        assert!(
            UnivariatePolynomial::<Q>::cmp_complex_roots_canonical(&pair[0], &pair[1])
                != Ordering::Greater
        );
    }
}

fn without_multiplicities(roots: Vec<(IsolatedRoot, usize)>) -> Vec<IsolatedRoot> {
    roots.into_iter().map(|(root, _)| root).collect()
}

fn refine_roots(
    roots: Vec<(IsolatedRoot, usize)>,
    tolerance: &Rational,
) -> Vec<(IsolatedRoot, usize)> {
    roots
        .into_iter()
        .map(|(root, multiplicity)| (root.refined(tolerance), multiplicity))
        .collect()
}

fn resolve_locations(roots: Vec<IsolatedRoot>) -> Vec<IsolatedRoot> {
    roots
        .into_iter()
        .map(|mut root| {
            root.classify_location();
            root
        })
        .collect()
}

#[test]
fn aberth_handles_very_close_roots_without_non_finite_iterates() {
    let p = parse!("(x^2+1)*((x-1/1000000)^2+1)*((x+1/1000000)^2+1)")
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0);

    let prec = 256;
    let tolerance = Float::with_val(prec, 1e-40);
    let field = FloatField::from_rep(Complex::from(tolerance.clone()));
    let p_float = p.map_coeff(|c| c.to_multi_prec_float(prec).into(), field);

    let roots = match p_float.roots(10000, &tolerance) {
        Ok(roots) | Err(roots) => roots,
    };

    assert_eq!(roots.len(), 6);
    assert!(roots.iter().all(|r| r.re.is_finite() && r.im.is_finite()));
}

#[test]
fn complex_root_isolation_refines_across_factors() {
    let p = parse!("(x^2+1)*((x-1/100)^2+1)")
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0);
    let roots = without_multiplicities(p.isolate_roots());

    assert_eq!(roots.len(), 4);
    assert_pairwise_isolated(&roots);
    assert_complex_roots_canonical(&roots);
}

#[test]
fn complex_root_isolation_marks_real_roots() {
    let p = parse!("(x+1)*(x-2)*(x^2+1)")
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0);
    let roots = resolve_locations(without_multiplicities(p.isolate_roots()));

    assert_eq!(roots.len(), 4);
    assert_eq!(roots.iter().filter(|root| is_real(root)).count(), 2);
    assert_eq!(roots.iter().filter(|root| is_imaginary(root)).count(), 2);
    assert!(roots.iter().all(|root| root.location.is_some()));
    assert_pairwise_isolated(&roots);
    assert_complex_roots_canonical(&roots);
}

#[test]
fn real_root_isolation_returns_root_objects_with_multiplicity() {
    let p = parse!("(x+1)^2*(x-2)^3*(x^2+1)^4")
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0);
    let mut roots = p.isolate_real_roots();

    assert_eq!(roots.len(), 2);
    assert_eq!(roots[0].1, 2);
    assert_eq!(roots[1].1, 3);
    assert!(!roots[0].0.is_positive_real());
    assert!(roots[1].0.is_positive_real());
    assert!(roots.iter().all(|(root, _)| is_real(root)));
}

#[test]
fn isolated_root_to_atom_uses_the_canonical_root_variable() {
    let x_polynomial = parse!("x^5+x+1")
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0);
    let y_polynomial = parse!("y^5+y+1")
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0);

    assert_eq!(
        x_polynomial.root(0).unwrap().to_atom(),
        y_polynomial.root(0).unwrap().to_atom()
    );
}

#[test]
fn exact_complex_isolated_root_to_atom_is_exact() {
    let field = FloatField::from_rep(Complex::from(Rational::one()));
    let mut polynomial =
        UnivariatePolynomial::new(&field, None, Arc::new(PolyVariable::Temporary(0)));
    polynomial.coefficients = vec![
        Complex::new(Rational::from(-1), Rational::from(-1)),
        Complex::new(Rational::one(), Rational::zero()),
    ];

    assert_eq!(polynomial.root(0).unwrap().to_atom(), parse!("1+1𝑖"));
}

#[test]
fn complex_root_isolation_marks_imaginary_roots_from_common_axis_part() {
    let p = parse!("x^3+x")
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0);
    let roots = resolve_locations(without_multiplicities(p.isolate_roots()));

    assert_eq!(roots.len(), 3);
    assert_eq!(roots.iter().filter(|root| is_imaginary(root)).count(), 3);
    assert_eq!(roots[1].location, Some(RootLocation::Zero));
    assert_pairwise_isolated(&roots);
    assert_complex_roots_canonical(&roots);
}

#[test]
fn complex_root_isolation_handles_axis_binomial() {
    let p = parse!("x^4-2")
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0);
    let tolerance = Rational::from((1, 1 << 16));
    let roots = resolve_locations(without_multiplicities(refine_roots(
        p.isolate_roots(),
        &tolerance,
    )));

    assert_eq!(roots.len(), 4);
    assert_eq!(roots.iter().filter(|root| is_real(root)).count(), 2);
    assert_eq!(roots.iter().filter(|root| is_imaginary(root)).count(), 2);
    assert_pairwise_isolated(&roots);
    assert!(is_real(&roots[0]));
    assert!(is_imaginary(&roots[1]) && roots[1].enclosure().center().im.is_negative());
    assert!(is_imaginary(&roots[2]) && roots[2].enclosure().center().im > Rational::zero());
    assert!(is_real(&roots[3]));
}

#[test]
fn complex_root_reisolation_honors_target_radius_for_axis_roots() {
    let p = parse!("x^4-2")
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0);
    let target_radius = Rational::from((Integer::from(1), Integer::from(1) << 32u32));
    let roots = p.isolate_square_free_roots(Some(&target_radius));

    assert!(
        roots
            .iter()
            .all(|root| root.enclosure().radius() <= &target_radius)
    );
}

#[test]
fn complex_root_isolation_handles_non_axis_binomial() {
    let p = parse!("x^8-2")
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0);
    let tolerance = Rational::from((1, 1 << 16));
    let roots = resolve_locations(without_multiplicities(refine_roots(
        p.isolate_roots(),
        &tolerance,
    )));

    assert_eq!(roots.len(), 8);
    assert_eq!(roots.iter().filter(|root| is_real(root)).count(), 2);
    assert_eq!(roots.iter().filter(|root| is_imaginary(root)).count(), 2);
    assert_pairwise_isolated(&roots);
    assert!(is_real(&roots[0]));
    assert!(roots[1].enclosure().center().im.is_negative());
    assert!(roots[2].enclosure().center().im > Rational::zero());
    assert!(is_imaginary(&roots[3]) && roots[3].enclosure().center().im.is_negative());
    assert!(is_imaginary(&roots[4]) && roots[4].enclosure().center().im > Rational::zero());
    assert!(roots[5].enclosure().center().im.is_negative());
    assert!(roots[6].enclosure().center().im > Rational::zero());
    assert!(is_real(&roots[7]));
    assert!(
        roots
            .iter()
            .enumerate()
            .all(|(index, root)| root.index() == index)
    );
}

fn complex_root_contains(root: &IsolatedRoot, re: Rational, im: Rational) -> bool {
    (&root.enclosure.center.re - &re).abs() <= root.enclosure.radius
        && (&root.enclosure.center.im - &im).abs() <= root.enclosure.radius
}

#[test]
fn complex_root_isolation_handles_exact_complex_coefficients() {
    let field = FloatField::from_rep(Complex::from(Rational::one()));
    let mut p = UnivariatePolynomial::new(&field, None, Arc::new(PolyVariable::Temporary(0)));
    p.coefficients = vec![
        Complex::new(Rational::from(3), Rational::one()),
        Complex::new(Rational::from(-3), Rational::zero()),
        Complex::new(Rational::one(), Rational::zero()),
    ];

    let roots = without_multiplicities(p.isolate_roots());

    assert_eq!(roots.len(), 2);
    assert!(complex_root_contains(
        &roots[0],
        Rational::one(),
        Rational::one()
    ));
    assert!(complex_root_contains(
        &roots[1],
        Rational::from(2),
        Rational::from(-1)
    ));
    assert_pairwise_isolated(&roots);
    assert_complex_roots_canonical(&roots);
}

#[test]
fn exact_complex_root_location_is_resolved_and_cached() {
    let field = FloatField::from_rep(Complex::from(Rational::one()));
    let mut p = UnivariatePolynomial::new(&field, None, Arc::new(PolyVariable::Temporary(0)));
    p.coefficients = vec![
        Complex::new(Rational::one(), Rational::from(-1)),
        Complex::new(Rational::zero(), Rational::from(3)),
        Complex::new(Rational::from(-2), Rational::from(-2)),
        Complex::new(Rational::one(), Rational::zero()),
    ];

    let roots = without_multiplicities(p.isolate_roots());
    assert!(roots.iter().all(|root| root.location.is_none()));

    let mut first = roots[0].clone();
    let first_location = first.classify_location();
    assert_eq!(first_location, RootLocation::Imaginary);
    assert!(is_imaginary(&first));

    let cached_real = p.root(1).unwrap();
    assert_eq!(cached_real.location, Some(RootLocation::Real));
    assert!(is_real(&cached_real));

    let cached_complex = p.root(2).unwrap();
    assert_eq!(cached_complex.location, Some(RootLocation::Complex));
    assert!(!is_real(&cached_complex));
    assert!(!is_imaginary(&cached_complex));
}

#[test]
fn complex_root_canonical_sort_handles_equal_real_parts() {
    let field = FloatField::from_rep(Complex::from(Rational::one()));
    let mut p = UnivariatePolynomial::new(&field, None, Arc::new(PolyVariable::Temporary(0)));
    p.coefficients = vec![
        Complex::new(Rational::from(-1), Rational::from(3)),
        Complex::new(Rational::from(-2), Rational::from(-3)),
        Complex::new(Rational::one(), Rational::zero()),
    ];

    let roots = without_multiplicities(p.isolate_roots());

    assert_eq!(roots.len(), 2);
    assert!(complex_root_contains(
        &roots[0],
        Rational::one(),
        Rational::one()
    ));
    assert!(complex_root_contains(
        &roots[1],
        Rational::one(),
        Rational::from(2)
    ));
    assert_pairwise_isolated(&roots);
    assert_complex_roots_canonical(&roots);
}

#[test]
fn complex_root_isolation_maps_rational_complex_coefficients_to_q() {
    let p = parse!("x^2+1")
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0);
    let field = FloatField::from_rep(Complex::from(Rational::one()));
    let p_complex = p.map_coeff(|c| Complex::from(c.clone()), field);

    let roots = resolve_locations(without_multiplicities(p_complex.isolate_roots()));

    assert_eq!(roots.len(), 2);
    assert_eq!(roots.iter().filter(|root| is_imaginary(root)).count(), 2);
    assert_complex_roots_canonical(&roots);
}

#[test]
fn complex_root_isolation_handles_very_close_conjugate_pairs() {
    let p = parse!("(x^2+1)*((x-1/10000)^2+1)*((x+1/10000)^2+1)")
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0);
    let roots = without_multiplicities(p.isolate_roots());

    assert_eq!(roots.len(), 6);
    assert_pairwise_isolated(&roots);
    assert_complex_roots_canonical(&roots);
}

#[test]
fn complex_root_isolation_handles_mixed_close_clusters() {
    let p = parse!(
        "((x-2)^2+(1/10)^2)*\
         ((x-2-1/100)^2+(1/10)^2)*\
         ((x+3)^2+(1/5)^2)*\
         (x^2+2*x+2)"
    )
    .to_polynomial::<_, u16>(&Q, None)
    .to_univariate_from_univariate(0);
    let refine = Rational::from((1, 1000));
    let roots = without_multiplicities(refine_roots(p.isolate_roots(), &refine));

    assert_eq!(roots.len(), 8);
    assert_pairwise_isolated(&roots);
    assert_complex_roots_canonical(&roots);
    assert!(
        roots
            .iter()
            .all(|root| root.enclosure().radius() <= &refine)
    );
}

#[test]
fn targeted_complex_root_refinement_is_retained_in_cache() {
    let p = parse!("x^3-7919")
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0);
    let tolerance = Rational::from((Integer::from(1), Integer::from(1) << 80u32));

    let original = p.root(0).unwrap();
    let defining_polynomial = original
        .defining_polynomial()
        .try_map_to_rational()
        .unwrap();
    root_cache()
        .rational
        .roots
        .write()
        .unwrap()
        .remove(defining_polynomial.coefficients());
    let refined = original.clone().refined(&tolerance);
    assert!(refined.enclosure().radius() <= &tolerance);

    let cached = p.root(0).unwrap();
    assert_eq!(cached.enclosure().center(), refined.enclosure().center());
    assert_eq!(cached.enclosure().radius(), refined.enclosure().radius());

    let refreshed = original.refined(&Rational::one());
    assert_eq!(refreshed.enclosure().center(), refined.enclosure().center());
    assert_eq!(refreshed.enclosure().radius(), refined.enclosure().radius());
}

#[test]
fn isolated_root_determines_real_sign() {
    let p = parse!("x*(x-2)*(x+3)")
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0);
    let mut roots = without_multiplicities(p.isolate_roots());

    assert!(!roots[0].is_positive_real());
    assert_eq!(roots[1].classify_location(), RootLocation::Zero);
    assert!(!roots[1].is_positive_real());
    assert!(roots[2].is_positive_real());

    let mut non_real = parse!("x^2+1")
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0)
        .root(0)
        .unwrap();
    assert!(!non_real.is_positive_real());
}

#[test]
fn all_root_refinement_is_retained_in_cache() {
    let p = parse!("x^4+43*x+103")
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0);
    let tolerance = Rational::from((Integer::from(1), Integer::from(1) << 72u32));

    p.isolate_roots();
    let refined = refine_roots(p.isolate_roots(), &tolerance);
    assert!(
        refined
            .iter()
            .all(|(root, _)| root.enclosure.radius <= tolerance)
    );

    let cached = p.isolate_roots();
    assert!(cached.iter().zip(refined).all(
        |((cached, cached_multiplicity), (refined, refined_multiplicity))| {
            cached.enclosure().center() == refined.enclosure().center()
                && cached.enclosure().radius() == refined.enclosure().radius()
                && cached_multiplicity == &refined_multiplicity
        }
    ));
}

#[test]
fn complex_root_multiplicity_expands_canonical_indices() {
    let p = parse!("(x+2)^2*(x-1)^3")
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0);
    let roots = p.isolate_roots();

    assert_eq!(roots.len(), 2);
    assert_eq!(roots[0].1, 2);
    assert_eq!(roots[1].1, 3);
    assert_eq!(roots[0].0.defining_polynomial().degree(), 1);
    assert_eq!(roots[0].0.index(), 0);
    assert_eq!(roots[1].0.index(), 0);
    assert!(
        root_cache()
            .rational
            .root_multisets
            .read()
            .unwrap()
            .contains_key(p.coefficients())
    );
    assert!(
        !root_cache()
            .rational
            .roots
            .read()
            .unwrap()
            .contains_key(p.coefficients())
    );
    assert_eq!(
        p.root(0).unwrap().enclosure().center(),
        p.root(1).unwrap().enclosure().center()
    );
    assert_eq!(
        p.root(2).unwrap().enclosure().center(),
        p.root(4).unwrap().enclosure().center()
    );
}

#[test]
fn complex_root_cache_ignores_variable_name() {
    let p_x = parse!("x^2+1")
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0);
    let p_y = parse!("y^2+1")
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0);

    let cache = RootCache::new();
    let mut calls = 0;
    let entry = cache.rational.root_multiset_slot(&p_x);
    entry.get_or_init(|| {
        calls += 1;
        RootMultiset::new()
    });
    let entry = cache.rational.root_multiset_slot(&p_y);
    entry.get_or_init(|| {
        calls += 1;
        RootMultiset::new()
    });

    assert_eq!(calls, 1);
}

#[test]
fn concurrent_root_cache_miss_computes_once() {
    let polynomial = Arc::new(
        parse!("x^3-2")
            .to_polynomial::<_, u16>(&Q, None)
            .to_univariate_from_univariate(0),
    );
    let cache = Arc::new(RootCache::new());
    let calls = Arc::new(AtomicUsize::new(0));
    let start = Arc::new(Barrier::new(8));

    let threads = (0..8)
        .map(|_| {
            let polynomial = polynomial.clone();
            let cache = cache.clone();
            let calls = calls.clone();
            let start = start.clone();
            std::thread::spawn(move || {
                start.wait();
                let entry = cache.rational.root_multiset_slot(polynomial.as_ref());
                entry.get_or_init(|| {
                    calls.fetch_add(1, AtomicOrdering::Relaxed);
                    std::thread::sleep(Duration::from_millis(25));
                    RootMultiset::new()
                });
            })
        })
        .collect::<Vec<_>>();

    for thread in threads {
        thread.join().unwrap();
    }
    assert_eq!(calls.load(AtomicOrdering::Relaxed), 1);
}

#[test]
fn refining_one_polynomial_does_not_lock_other_entries() {
    let p_a = parse!("x^2-2")
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0);
    let p_b = parse!("x^2-3")
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0);
    let cache = Arc::new(RootCache::new());
    let entry_a = cache.rational.root_slot(&p_a);
    entry_a.get_or_init(|| RwLock::new(Vec::<CachedRoot>::new()));
    let entry_b = cache.rational.root_slot(&p_b);
    entry_b.get_or_init(|| RwLock::new(Vec::<CachedRoot>::new()));

    let roots_a = entry_a.get().unwrap();
    let _refining_a = roots_a.write().unwrap();

    let (finished_tx, finished_rx) = std::sync::mpsc::channel();
    let worker = {
        let cache = cache.clone();
        std::thread::spawn(move || {
            let entry_b = cache.rational.root_slot(&p_b);
            let _roots_b = entry_b.get().unwrap().read().unwrap();
            finished_tx.send(()).unwrap();
        })
    };

    finished_rx
        .recv_timeout(Duration::from_millis(250))
        .expect("an unrelated cache entry was blocked");
    worker.join().unwrap();
}

#[test]
fn complex_root_cache_ignores_variable_name_for_complex_coefficients() {
    let field = FloatField::from_rep(Complex::from(Rational::one()));
    let mut p_x = UnivariatePolynomial::new(&field, None, Arc::new(PolyVariable::Temporary(0)));
    p_x.coefficients = vec![
        Complex::new(Rational::from(3), Rational::one()),
        Complex::new(Rational::from(-3), Rational::zero()),
        Complex::new(Rational::one(), Rational::zero()),
    ];
    let mut p_y = UnivariatePolynomial::new(&field, None, Arc::new(PolyVariable::Temporary(1)));
    p_y.coefficients = p_x.coefficients.clone();

    let cache = RootCache::new();
    let mut calls = 0;
    let entry = cache.complex.root_multiset_slot(&p_x);
    entry.get_or_init(|| {
        calls += 1;
        RootMultiset::new()
    });
    let entry = cache.complex.root_multiset_slot(&p_y);
    entry.get_or_init(|| {
        calls += 1;
        RootMultiset::new()
    });

    assert_eq!(calls, 1);
}
