//! Controlled external consumer of the public typed GPL and tagged constant APIs.
use std::collections::HashMap;

use symbolica::atom::{Atom, AtomCore};
use symbolica::domains::float::{Complex, Float, Real};
use symbolica::parse;
use symbolica::transcendental::{
    SixthRootGplEvaluator, SixthRootGplLetter, gpl_at_one_with_sixth_root_letters, sixth_root_gpl,
};

fn main() {
    use SixthRootGplLetter::{MinusRoot as M, One as O, Zero as Z};
    let words = vec![vec![Z, O, O, M], vec![Z, Z, M, O]];
    let parameters = HashMap::<Atom, Complex<Float>>::new();
    let values = SixthRootGplEvaluator::new(&words)
        .unwrap()
        .evaluate(1, 1536)
        .unwrap();
    let tolerance = parse!("1e-385")
        .evaluate_with_prec(&parameters, 1600)
        .unwrap()
        .re;
    for (index, (word, batch)) in words.iter().zip(values).enumerate() {
        let expression = sixth_root_gpl(word, 1).unwrap();
        assert!(expression.contains_symbol(gpl_at_one_with_sixth_root_letters()));
        let canonical = expression.to_canonical_string();
        let parsed = parse!(&canonical, default_namespace = "external_gpl_consumer");
        let tagged = parsed.evaluate_with_prec(&parameters, 1536).unwrap();
        let difference =
            (tagged.re.clone() - &batch.re).norm() + (tagged.im.clone() - &batch.im).norm();
        assert!(difference < tolerance);
        println!(
            "GPL_PUBLIC_API_CASE\t{index}\t{canonical}\t{}\t{}",
            tagged.re, tagged.im
        );
    }
    assert!(sixth_root_gpl(&[Z, M], 0).is_err());
    assert!(sixth_root_gpl(&[O, Z], 1).is_err());
    assert!(SixthRootGplEvaluator::new(&[vec![Z; 7]]).is_err());
    println!("GPL_PUBLIC_API_COMPLETE\t2\t385");
}
