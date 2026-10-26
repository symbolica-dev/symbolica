use symbolica::{
    atom::{Atom, AtomCore, EvaluationInfo},
    evaluate::Instruction,
    parse, symbol,
};

#[test]
fn external_expression_tags_survive_both_evaluator_construction_paths() {
    let tags = vec![
        Atom::num(2),
        parse!("evaluator_tags::a::label"),
        parse!("evaluator_tags::b::label + 1"),
    ];
    let expected = tags.clone();
    let function = symbol!(
        "evaluator_tags::external",
        eval = EvaluationInfo::new()
            .with_tags(3)
            .register_tagged(move |actual| {
                assert_eq!(
                    actual.iter().map(|tag| tag.to_owned()).collect::<Vec<_>>(),
                    expected
                );
                Box::new(|args: &[f64]| args[0] + 5.)
            })
    );
    let x = symbol!("evaluator_tags::x");
    let expression = function.call((tags[0].clone(), tags[1].clone(), tags[2].clone(), x));

    for direct in [false, true] {
        let mut evaluator = expression
            .evaluator(&[x.to_atom()])
            .direct_translation(direct)
            .build()
            .unwrap()
            .map_coeff(&|coefficient| coefficient.re.to_f64());
        assert_eq!(evaluator.evaluate_single(&[3.]), 8.);
        let exported = evaluator.export_instructions();
        let call = exported
            .instructions
            .iter()
            .find_map(|instruction| match instruction {
                Instruction::Fun(_, call, _) if call.0 == function => Some(call),
                _ => None,
            })
            .unwrap();
        assert_eq!(call.1, tags);
        assert_eq!(call.2.len(), 1);
    }
}
