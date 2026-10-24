use super::*;
use crate::atom::AtomCore;
use crate::parse;
use std::collections::HashMap;

fn evaluate(word: &[i8], bits: u32) -> Complex<Float> {
    SixthRootHplEvaluator::new(&[word.to_vec()])
        .unwrap()
        .evaluate(1, bits)
        .unwrap()
        .remove(0)
}

fn close(left: &Complex<Float>, right: &Complex<Float>, bits: u32) {
    let delta = left - right;
    let difference = delta.re.norm() + delta.im.norm();
    let tolerance = Float::from_rational_round(
        &Rational::from((Integer::one(), Integer::one() << (bits - 8))),
        bits,
        crate::domains::float::RoundingDirection::Up,
    );
    assert!(
        difference < tolerance,
        "difference={difference}, tolerance={tolerance}, left={left}, right={right}"
    );
}

#[test]
fn low_weight_values_and_trailing_zeros() {
    let bits = 160;
    let zero = Float::new(bits);
    let one = Float::with_val(bits, 1);
    let log = Complex::new(zero.clone(), one.pi() / 3);
    close(
        &evaluate(&[], bits),
        &Complex::new(one.clone(), zero.clone()),
        bits,
    );
    close(&evaluate(&[0], bits), &log, bits);
    close(&evaluate(&[1], bits), &log, bits);
    let minus_one = parse!("log((3+sqrt(3)*𝑖)/2)")
        .evaluate_with_prec(&HashMap::<Atom, Complex<Float>>::new(), bits)
        .unwrap();
    close(&evaluate(&[-1], bits), &minus_one, bits);
    close(
        &evaluate(&[0, 0], bits),
        &(&log * &log / Complex::new(Float::with_val(bits, 2), zero.clone())),
        bits,
    );
    close(
        &evaluate(&[1, 1, 1], bits),
        &(&log * &log * &log / Complex::new(Float::with_val(bits, 6), zero)),
        bits,
    );
    let expected = parse!("polylog(2,(1+sqrt(3)*𝑖)/2)")
        .evaluate_with_prec(&HashMap::<Atom, Complex<Float>>::new(), bits)
        .unwrap();
    close(&evaluate(&[0, 1], bits), &expected, bits);
    close(
        &(evaluate(&[1, 0], bits) + evaluate(&[0, 1], bits)),
        &(&log * &log),
        bits,
    );
}

#[test]
fn shuffle_and_conjugation() {
    let bits = 128;
    let batch = SixthRootHplEvaluator::new(&[vec![-1, 0], vec![1, -1], vec![0, 1, 0, 0]]).unwrap();
    let plus = batch.evaluate(1, bits).unwrap();
    let minus = batch.evaluate(-1, bits).unwrap();
    for (p, m) in plus.iter().zip(&minus) {
        assert_eq!(p.re, m.re);
        assert_eq!(p.im, -m.im.clone());
    }
    let product = evaluate(&[-1], bits) * evaluate(&[0, 1], bits);
    let shuffled =
        evaluate(&[-1, 0, 1], bits) + evaluate(&[0, -1, 1], bits) + evaluate(&[0, 1, -1], bits);
    close(&product, &shuffled, bits);
}

#[test]
fn exact_tagged_constant_evaluates_and_reports_invalid_inputs() {
    assert!(crate::state::State::is_builtin("hpl_indices"));
    let expression = sixth_root_hpl(&[1, 0], 1).unwrap();
    let map = HashMap::<Atom, Complex<Float>>::new();
    close(
        &expression.evaluate_with_prec(&map, 128).unwrap(),
        &evaluate(&[1, 0], 128),
        128,
    );
    assert!(
        parse!("hpl_at_sixth_root(hpl_indices(2),1)")
            .evaluate_with_prec(&map, 128)
            .unwrap_err()
            .to_string()
            .contains("indices")
    );
    assert!(
        parse!("hpl_at_sixth_root(hpl_indices(1,0),0)")
            .evaluate_with_prec(&map, 128)
            .unwrap_err()
            .to_string()
            .contains("sign")
    );
    assert!(
        parse!("hpl_at_sixth_root(x,1)")
            .evaluate_with_prec(&map, 128)
            .is_err()
    );
    assert!(SixthRootHplEvaluator::new(&[vec![1; 7]]).is_err());
    assert!(
        SixthRootHplEvaluator::new(&[vec![1]])
            .unwrap()
            .evaluate(1, 0)
            .is_err()
    );
}

#[test]
fn precision_is_recomputed_and_batch_matches_individual_values() {
    let words = vec![
        vec![-1, 1, 0, -1, 1, 0],
        vec![0, 0, 1, 0, 0, 0],
        vec![1, -1, 1, -1, 1, -1],
    ];
    let batch = SixthRootHplEvaluator::new(&words).unwrap();
    let low = batch.evaluate(1, 128).unwrap();
    let high = batch.evaluate(1, 800).unwrap();
    for ((word, lo), hi) in words.iter().zip(&low).zip(&high) {
        close(lo, hi, 128);
        close(hi, &evaluate(word, 800), 800);
        assert_eq!(hi.re.prec(), 800);
    }
}

#[test]
fn fixed_point_enclosures_include_higher_precision_recomputations() {
    let evaluator =
        SixthRootHplEvaluator::new(&[vec![-1, 1, 0, -1, 1, 0], vec![0, 0, 1, 0, 0, 0]]).unwrap();
    let lower = evaluator.evaluate_balls(100).unwrap();
    let upper = evaluator.evaluate_balls(250).unwrap();
    for (a, b) in lower.iter().zip(&upper) {
        assert!(a.re.contains(&b.re.center));
        assert!(a.im.contains(&b.im.center));
    }
}

#[test]
fn difficult_weight_six_values_match_an_independent_evaluator() {
    use std::io::Write;
    use std::process::{Command, Stdio};
    // GiNaC is an optional development oracle, never a runtime dependency.
    if Command::new("ginsh").arg("--help").output().is_err() {
        return;
    }
    let words = vec![
        vec![-1, 1, 0, -1, 1, 0],
        vec![1, -1, 1, -1, 1, -1],
        vec![0, 0, 1, 0, 0, 0],
    ];
    let mut script = "Digits=220:\n".to_owned();
    for word in &words {
        let indices = word.iter().map(i8::to_string).collect::<Vec<_>>().join(",");
        script += &format!("evalf(H({{{indices}}},(1+I*sqrt(3))/2));\n");
    }
    script += "quit;\n";
    let mut child = Command::new("ginsh")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(script.as_bytes())
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    let reference = String::from_utf8(output.stdout).unwrap();
    let values = SixthRootHplEvaluator::new(&words)
        .unwrap()
        .evaluate(1, 665)
        .unwrap();
    let map = HashMap::<Atom, Complex<Float>>::new();
    let lines = reference
        .lines()
        .filter(|line| !line.trim().is_empty())
        .collect::<Vec<_>>();
    assert_eq!(lines.len(), words.len());
    for (actual, line) in values.iter().zip(lines) {
        let reference = parse!(&line.replace("I", "𝑖"))
            .evaluate_with_prec(&map, 720)
            .unwrap();
        close(actual, &reference, 665);
    }
}

#[test]
fn frozen_independent_weight_six_references() {
    // GiNaC 1.8.10, Digits=220, H(word, (1+I*sqrt(3))/2).
    // Keep a self-contained oracle regression when ginsh is not installed.
    let words = vec![
        vec![-1, 1, 0, -1, 1, 0],
        vec![1, -1, 1, -1, 1, -1],
        vec![0, 0, 1, 0, 0, 0],
    ];
    let reference = [
        "0.02879157001790452155518851226410347247675503697654049707591932893328041231866354647578949369956043757365264290045256283414905565204448550941076174656027143259812871582616998084275237500668088873560326614646084536851734093856563146165+0.02619725222854897909320165397208761537394853436604039861444652500418230234534757969114079586267812658660366918553226470776600336714618603818166822290373649913564697628412098531631945132734027531367004857600237617389050717630479943408*𝑖",
        "5.22480634776896255795043571624735682008474536881722816588822353098071184649624441275713299830603925012960195089682768090350163229440402999804496471868931461533943728252540156255409220869497472890866803879524255968854282543541609311E-4+7.23961297110217523432123041636949817079593304897736777298616057672930953841654068514792880060126708002643138395651136345901471445967125398503573484466725052031765488803161840934313042363296602919660856178252725271486471426767168395E-4*𝑖",
        "-9.57906468226792858313758478208614046957142118926638879081991192803564678954895208452714279098839114978336568149778704174309319072821562737000462391006503445585455491301497214436194691128051361069737509730857270403948617325944787555-4.34687894887752985959840445813243050871880769506038019234358245944481624828622712456170772631998406065232874606705432880183190915244215021742028142525319507635562830445926767038238301833619104782756949847623381123590561103457221213*𝑖",
    ];
    let values = SixthRootHplEvaluator::new(&words)
        .unwrap()
        .evaluate(1, 665)
        .unwrap();
    let map = HashMap::<Atom, Complex<Float>>::new();
    for (actual, expected) in values.iter().zip(reference) {
        close(
            actual,
            &parse!(expected).evaluate_with_prec(&map, 720).unwrap(),
            665,
        );
    }
}
