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
    let lower = evaluator.evaluate_balls(1, 100).unwrap();
    let upper = evaluator.evaluate_balls(1, 250).unwrap();
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

#[cfg(feature = "float-mpfr")]
#[test]
fn trilog_fractional_transform_matches_sixth_root_hpls() {
    // For z=exp(sign*i*pi/3), (1-z)/(1+z)=-sign*i/sqrt(3).
    // This HPL identity follows by differentiating in z and matching the
    // z=0 value zeta(3). Check both conjugate endpoints against the independent
    // native Li_3 series, whose argument lies strictly inside the unit disk.
    let bits = 720;
    let reference_bits = 840;
    let ordinary_map = HashMap::<Atom, Complex<Float>>::new();
    let tracked_map = HashMap::<Atom, TrackedComplex>::new();
    let mut ordinary_values = Vec::new();
    let mut tracked_values = Vec::new();
    for sign in [1, -1] {
        let h = |word: &[i8]| sixth_root_hpl(word, sign).unwrap();
        let mut identity = parse!("zeta(3)") - parse!("zeta(2)") * (h(&[1]) + h(&[-1]));
        for a in [-1, 1] {
            for b in [-1, 1] {
                identity -= parse!("log(2)") * h(&[a, b]);
                identity -= h(&[a, b, 0]);
                identity += h(&[a, b, -1]);
            }
        }
        let direct = if sign == 1 {
            parse!("polylog(3,-𝑖/sqrt(3))")
        } else {
            parse!("polylog(3,𝑖/sqrt(3))")
        };
        let reference = direct
            .evaluate_with_prec(&ordinary_map, reference_bits)
            .unwrap();
        let reference_ball = direct
            .evaluate_tracked(&tracked_map, reference_bits)
            .unwrap();
        let ordinary = identity.evaluate_with_prec(&ordinary_map, bits).unwrap();
        let tracked = identity.evaluate_tracked(&tracked_map, bits).unwrap();
        close(&ordinary, &reference, bits);
        assert!(reference_ball.ball.contains_complex(&reference));
        assert!(tracked.ball.contains_ball(&reference_ball.ball));
        assert_eq!(tracked.error_quality, ErrorQuality::Bounded);
        assert!(tracked.precision().real.significant_digits().unwrap() >= 200);
        assert!(tracked.precision().imaginary.significant_digits().unwrap() >= 200);
        ordinary_values.push(ordinary);
        tracked_values.push(tracked);
    }
    close(&ordinary_values[0].conj(), &ordinary_values[1], bits);
    assert!(
        tracked_values[0]
            .ball
            .conj()
            .intersects(&tracked_values[1].ball)
    );
}
#[test]
fn unit_alphabet_pole_matrices_and_convergence_bounds_are_exact() {
    // Define the poles independently in Q(i sqrt(3)). Multiplying each
    // computed matrix column by its pole must give half the corresponding
    // input column. This also checks conjugate/complement signs exactly.
    let poles = [
        (Pole::Plus, (1, 2), (-1, 2)),
        (Pole::Minus, (-1, 2), (1, 2)),
        (Pole::One, (1, 1), (0, 1)),
        (Pole::NegativeOne, (-1, 1), (0, 1)),
        (Pole::Two, (2, 1), (0, 1)),
        (Pole::ConjugateMinus, (-1, 2), (-1, 2)),
        (Pole::ComplementPlus, (1, 2), (1, 2)),
        (Pole::ComplementMinus, (3, 2), (-1, 2)),
        (Pole::ComplementConjugateMinus, (3, 2), (1, 2)),
    ];
    for (pole, real, imaginary) in poles {
        let real = Rational::from(real);
        let imaginary = Rational::from(imaginary);
        assert!(
            real.clone() * &real + imaginary.clone() * &imaginary * Rational::from(3)
                >= Rational::one()
        );
        assert_eq!(pole.complement().complement(), pole);
        let mut columns = Vec::new();
        for (a, b) in [(12, 0), (0, 12)] {
            let output = pole.multiply_half_inverse(Fixed {
                a: Integer::from(a),
                b: Integer::from(b),
            });
            let x = Rational::from(output.a.clone());
            let y = Rational::from(output.b.clone());
            assert_eq!(
                &x * &real - &y * &imaginary * Rational::from(3),
                Rational::from((a, 2))
            );
            assert_eq!(&x * &imaginary + &y * &real, Rational::from((b, 2)));
            columns.push((output.a.abs(), output.b.abs()));
        }
        assert!(&columns[0].0 + &columns[1].0 <= Integer::from(12));
        assert!(&columns[0].1 + &columns[1].1 <= Integer::from(12));
    }
}
