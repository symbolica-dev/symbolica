use super::*;
use std::collections::HashMap;

#[test]
fn endpoint_one_gpl_words_conjugation_and_error_enclosures() {
    use GplLetter::*;
    for word in [[Zero, Zero, R4, One], [Zero, One, One, R4]] {
        let evaluator = EndpointOneGplEvaluator::new(&word).unwrap();
        let low = evaluator.evaluate_ball(160).unwrap();
        let high = evaluator.evaluate_ball(256).unwrap();
        assert!(low.contains_ball(&high));
        let expression = endpoint_one_gpl(&word).unwrap();
        let value = expression
            .evaluate_tracked(&HashMap::<Atom, TrackedComplex>::new(), 160)
            .unwrap();
        assert!(value.ball.contains_ball(&high));
        assert!(value.precision().real.significant_digits().unwrap() >= 40);
        if word[1] == Zero {
            assert!(low.re.is_strictly_negative());
            assert!(low.im.is_strictly_positive());
        } else {
            assert!(low.re.is_strictly_positive());
            assert!(low.im.is_strictly_negative());
        }
    }
    assert!(endpoint_one_gpl(&[One, One, One, R4]).is_err());
}

// Independent tanh-sinh quadratures at 230 and 270 decimal digits,
// reproducible with tools/validate_endpoint_one_gpl.py (development-only mpmath):
// A = 1/2 integral_0^1 log(t)^2 log(1-t)/(t-r4) dt;
// B = integral_0^1 Li2(t)/t log(1-(1-t)/r4) dt
//   = -integral_0^1 Li3(t)/(t+r4-1) dt.
// Both B forms agree independently; reference values retain 210 digits.
#[test]
fn endpoint_one_gpl_agrees_with_independent_quadrature() {
    use GplLetter::*;
    let references = [
        (
            [Zero, Zero, R4, One],
            "-0.0855980801943223880649734710357238830706309959523641162550092597448480034073261999506417643318392090471527657984567619861954223738247186601796758283523971431750457394333575458324659428899756804097743598229526669",
            "0.0946225372783897470526754572830559678371266820138213587236956728231893906277528796952168463823808171643815820577230491626318541180192642396258721858154367615850493447848788723628046642323748215935023516515995159",
        ),
        (
            [Zero, One, One, R4],
            "0.308894286447851572071760224094619543514528349357323793865566597637323653651257774122493592946542362785622334307130647419012915894747908424235103348470220050599136515904577272226721968866572539777404443213191323",
            "-0.348907504838312019520792264262531598973439203751174590542666808596893897771671516954893206164430070369728651018955840995447945054148807841741674560495023703091306745309007825320443318172902847335874792147879539",
        ),
    ];
    let bits = 840;
    let tolerance = Float::parse("1e-205", Some(bits)).unwrap();
    for (word, real, imaginary) in references {
        let value = EndpointOneGplEvaluator::new(&word)
            .unwrap()
            .evaluate_ball(bits)
            .unwrap();
        for (computed, reference) in [(&value.re, real), (&value.im, imaginary)] {
            let reference = Float::parse(reference, Some(bits)).unwrap();
            assert!((computed.center.clone() - reference).norm() < tolerance);
            assert!(computed.radius < tolerance);
        }
    }
}

#[test]
#[ignore = "explicit twenty-thousand-digit GPL acceptance"]
fn endpoint_one_gpl_twenty_thousand_digits() {
    use GplLetter::*;
    let bits = Float::decimal_digits_to_bits(20_000.0).unwrap() + 128;
    for word in [[Zero, Zero, R4, One], [Zero, One, One, R4]] {
        let evaluator = EndpointOneGplEvaluator::new(&word).unwrap();
        let value = evaluator.evaluate_ball(bits).unwrap();
        let reference = evaluator.evaluate_ball(bits + 128).unwrap();
        assert!(value.contains_ball(&reference));
        let precision = TrackedComplex::from_ball(value, ErrorQuality::Bounded).precision();
        assert!(precision.real.significant_digits().unwrap() >= 20_000);
        assert!(precision.imaginary.significant_digits().unwrap() >= 20_000);
    }
}
