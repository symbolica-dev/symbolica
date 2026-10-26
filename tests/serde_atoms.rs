#![cfg(feature = "serde")]

use std::process::Command;

use symbolica::{
    atom::{Atom, AtomCore, AtomView, Symbol, UserData},
    coefficient::Coefficient,
    domains::{
        finite_field::{FiniteFieldCore, Zp64},
        float::{Complex, Float},
    },
    evaluate::{
        Expression, ExpressionEvaluator, FunctionMap, FunctionRegistrationOptions, InliningPolicy,
        Instruction, Slot,
    },
    symbol,
};

fn finite_field_atom(prime: u64) -> Atom {
    let field = Zp64::new(prime);
    let value = field.to_element(5);
    Atom::num(Coefficient::from_finite_field(field, value))
}

fn tagged_atom() -> Atom {
    let owner = symbol!(
        "serde_atoms::owner"; Symmetric;
        data = UserData::Atom(finite_field_atom(17)),
        aliases = ["serde_atoms::owner_alias"]
    );
    let x = symbol!("serde_atoms::tag_x");
    symbol!("serde_atoms::unused_between");
    let y = symbol!("serde_atoms::tag_y");
    owner.call((x, y))
}

fn assert_tag_metadata(tag: &Atom) {
    let owner = tag.get_symbol().unwrap();
    assert!(owner.is_symmetric());
    assert_eq!(owner.get_data(), &UserData::Atom(finite_field_atom(17)));
    assert_eq!(symbol!("serde_atoms::owner_alias"), owner);
    let x = symbol!("serde_atoms::tag_x");
    let y = symbol!("serde_atoms::tag_y");
    assert_eq!(tag, &owner.call((y, x)));
}

// Each payload is decoded in a fresh process so one roundtrip cannot accidentally
// register the metadata needed by another, and no unsafe global reset is needed.
#[test]
fn serde_imports_atom_state_in_a_fresh_process() {
    const PAYLOAD: &str = "SYMBOLICA_SERDE_ATOMS_PAYLOAD";
    const KIND: &str = "SYMBOLICA_SERDE_ATOMS_KIND";
    if let Ok(json) = std::env::var(PAYLOAD) {
        let kind = std::env::var(KIND).unwrap();
        if kind == "conflict" {
            symbol!("serde_atoms::owner"; Antisymmetric);
            let bytes: Vec<u8> = serde_json::from_str(&json).unwrap();
            let expected = Atom::import(&mut bytes.as_slice(), None).unwrap_err();
            let actual = serde_json::from_str::<Atom>(&json).unwrap_err();
            assert!(expected.to_string().contains("Symbol conflict"));
            assert!(actual.to_string().contains(&expected.to_string()));
            return;
        }

        // Register referenced symbols in the opposite order, and another field
        // first, to exercise remapping both symbol and coefficient resource IDs.
        symbol!("serde_atoms::tag_y");
        symbol!("serde_atoms::receiver_padding");
        symbol!("serde_atoms::tag_x");
        let _padding = finite_field_atom(19);
        match kind.as_str() {
            "atom" => {
                let tag: Atom = serde_json::from_str(&json).unwrap();
                assert_tag_metadata(&tag);
            }
            "instruction" => {
                let instruction: Instruction = serde_json::from_str(&json).unwrap();
                let Instruction::Fun(out, call, real) = instruction else {
                    panic!("expected a function instruction");
                };
                assert_eq!(out, Slot::Out(0));
                assert!(real);
                assert_eq!(call.0, symbol!("serde_atoms::function"));
                assert_eq!(call.2, vec![Slot::Param(0)]);
                assert_eq!(call.1.len(), 1);
                assert_tag_metadata(&call.1[0]);
            }
            "expression" => {
                let expression: Expression<i64> = serde_json::from_str(&json).unwrap();
                let Expression::Fun(hash, function, tags, args) = expression else {
                    panic!("expected a function expression");
                };
                assert_eq!(hash, 7);
                assert_eq!(function, symbol!("serde_atoms::function"));
                assert_eq!(args, vec![Expression::Parameter(11, 0)]);
                assert_eq!(tags.len(), 1);
                assert_tag_metadata(&tags[0]);
            }
            "evaluator" => {
                let mut evaluator: ExpressionEvaluator<f64> = serde_json::from_str(&json).unwrap();
                assert_eq!(evaluator.evaluate_single(&[3.]), 10.);
                let exported = evaluator.export_instructions();
                assert_eq!(exported.sub_evaluators.len(), 1);
                let body = &exported.sub_evaluators[0];
                assert_eq!(body.tags.len(), 1);
                assert_tag_metadata(&body.tags[0]);
            }
            _ => panic!("unexpected payload kind: {kind}"),
        }
        return;
    }

    let tag = tagged_atom();
    let function = symbol!("serde_atoms::function");
    let x = symbol!("serde_atoms::input");
    let instruction = Instruction::Fun(
        Slot::Out(0),
        Box::new((function, vec![tag.clone()], vec![Slot::Param(0)])),
        true,
    );
    let expression = Expression::<i64>::Fun(
        7,
        function,
        vec![tag.clone()],
        vec![Expression::Parameter(11, 0)],
    );
    let mut functions = FunctionMap::new();
    functions
        .add_tagged_function_with_options(
            function,
            vec![tag.clone()],
            vec![x],
            x.pow(2) + 1,
            FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
        )
        .unwrap();
    let evaluator = function
        .call((tag.clone(), x))
        .evaluator(&[x.to_atom()])
        .function_map(functions)
        .horner_iterations(0)
        .build()
        .unwrap()
        .map_coeff(&|c| c.re.to_f64());
    let atom_json = serde_json::to_string(&tag).unwrap();
    let payloads = [
        ("atom", atom_json.clone()),
        ("instruction", serde_json::to_string(&instruction).unwrap()),
        ("expression", serde_json::to_string(&expression).unwrap()),
        ("evaluator", serde_json::to_string(&evaluator).unwrap()),
        ("conflict", atom_json),
    ];
    for (kind, json) in payloads {
        let output = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "serde_imports_atom_state_in_a_fresh_process",
                "--nocapture",
            ])
            .env(PAYLOAD, json)
            .env(KIND, kind)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{kind} roundtrip failed:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
}

#[test]
fn serde_atom_preserves_floating_coefficient_precision() {
    let atom = Atom::num(Complex::new(
        Float::with_val(173, 1.25),
        Float::with_val(211, 0),
    ));
    let json = serde_json::to_string(&atom).unwrap();
    let decoded: Atom = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, atom);
    let AtomView::Num(number) = decoded.as_view() else {
        panic!("expected a number");
    };
    let Coefficient::Float(value) = number.get_coeff_view().to_owned() else {
        panic!("expected a floating coefficient");
    };
    assert_eq!(value.re.prec(), 173);
    assert_eq!(value.im.prec(), 211);
}

#[test]
fn serde_atom_and_symbol_reject_invalid_export_data() {
    assert!(serde_json::from_str::<Atom>("[]").is_err());
    assert!(serde_json::from_str::<Symbol>("[]").is_err());
    assert!(serde_json::from_str::<Atom>("\"x + 1\"").is_err());

    let atom = Atom::num(42);
    let mut atom_bytes = Vec::new();
    atom.export(&mut atom_bytes).unwrap();
    let truncated = serde_json::to_string(&atom_bytes[..atom_bytes.len() - 1]).unwrap();
    assert!(serde_json::from_str::<Atom>(&truncated).is_err());
    atom_bytes.push(0);
    assert!(serde_json::from_str::<Atom>(&serde_json::to_string(&atom_bytes).unwrap()).is_err());

    let symbol = symbol!("serde_atoms::invalid_bytes_symbol");
    let mut symbol_bytes = Vec::new();
    symbol.export(&mut symbol_bytes).unwrap();
    symbol_bytes.push(0);
    assert!(
        serde_json::from_str::<Symbol>(&serde_json::to_string(&symbol_bytes).unwrap()).is_err()
    );
}
