use super::{ExportedInstructions, ExportedSubEvaluator, Instruction, Slot};
use crate::{
    atom::{Atom, AtomCore, EvaluationInfo, Symbol},
    coefficient::Coefficient,
    domains::{
        dual::HyperDual,
        finite_field::{FiniteFieldCore, Zp64},
        float::{Complex, Float, RealLike},
        rational::Rational,
    },
    evaluate::{
        Dualizer, ExpressionEvaluator, FunctionMap, FunctionRegistrationOptions, InliningPolicy,
    },
    parse, symbol,
};
use std::sync::Arc;

fn expression_tags() -> Vec<Atom> {
    let field = Zp64::new(17);
    let value = field.to_element(5);
    vec![
        Atom::num(2),
        parse!("symbolica::import_test::tag"),
        parse!("symbolica::import_test::tag + 1"),
        // This coefficient's domain cannot be preserved by parsing its printed value.
        Atom::num(Coefficient::from_finite_field(field, value)),
    ]
}

fn instructions(
    input_count: usize,
    output_count: usize,
    temporary_count: usize,
    constants: Vec<f64>,
    instructions: Vec<Instruction>,
) -> ExportedInstructions<f64> {
    ExportedInstructions {
        input_count,
        output_count,
        temporary_count,
        constants,
        instructions,
        constant_functions: vec![],
        sub_evaluators: vec![],
    }
}

#[test]
fn import_arithmetic_and_output_dependencies() {
    use Slot::{Const, Out, Param, Temp};

    let exported = instructions(
        2,
        3,
        4,
        vec![2.],
        vec![
            Instruction::Add(Temp(0), vec![Param(0), Const(0)], 2),
            Instruction::Mul(Temp(1), vec![Temp(0), Param(1)], 2),
            Instruction::Pow(Out(0), Temp(1), -1, true),
            Instruction::Powf(Temp(2), Param(0), Param(1), true),
            Instruction::Fun(
                Temp(3),
                Box::new((Symbol::SIN, vec![], vec![Param(0)])),
                true,
            ),
            Instruction::Add(Out(1), vec![Temp(2), Temp(3), Out(0)], 3),
            Instruction::Assign(Out(2), Out(1)),
        ],
    );
    let mut evaluator = ExpressionEvaluator::from_instructions(&exported).unwrap();
    let mut output = [0.; 3];
    for [x, y] in [[2.0_f64, 3.0_f64], [4.0, 0.5]] {
        evaluator.evaluate(&[x, y], &mut output);
        let inverse = 1. / ((x + 2.) * y);
        let expected = x.powf(y) + x.sin() + inverse;
        assert_eq!(output, [inverse, expected, expected]);
    }

    let mut reimported =
        ExpressionEvaluator::from_instructions(&evaluator.export_instructions()).unwrap();
    let mut restored_output = [0.; 3];
    reimported.evaluate(&[4., 0.5], &mut restored_output);
    assert_eq!(restored_output, output);
}

#[test]
fn import_repeated_parameter_constant_and_expression_outputs() {
    let expressions = [
        parse!("x"),
        parse!("7/2"),
        parse!("x + y"),
        parse!("x + y"),
        parse!("x"),
    ];
    let exact = Atom::evaluator_multiple(&expressions, &[parse!("x"), parse!("y")])
        .build()
        .unwrap();
    let imported = ExpressionEvaluator::from_instructions(&exact.export_instructions()).unwrap();
    let mut evaluator = imported.map_coeff(&|c| c.re.to_f64());
    let mut output = [0.; 5];
    evaluator.evaluate(&[2., 3.], &mut output);
    assert_eq!(output, [2., 3.5, 5., 5., 2.]);
    evaluator.evaluate(&[-2., 7.], &mut output);
    assert_eq!(output, [-2., 3.5, 5., 5., -2.]);
}

#[test]
fn imported_assignments_can_be_vectorized() {
    let expressions = [parse!("x"), parse!("x"), parse!("7/2"), parse!("x^2")];
    let exported = Atom::evaluator_multiple(&expressions, &[parse!("x")])
        .build()
        .unwrap()
        .export_instructions();
    for unary_multiplications in [false, true] {
        let mut exported = exported.clone();
        if unary_multiplications {
            for instruction in &mut exported.instructions {
                if let Instruction::Assign(output, input) = instruction {
                    *instruction = Instruction::Mul(*output, vec![*input], 1);
                }
            }
        }
        let mut evaluator = ExpressionEvaluator::from_instructions(&exported)
            .unwrap()
            .vectorize(&Dualizer::new(
                HyperDual::<f64>::new(vec![vec![0], vec![1]]),
                vec![],
            ))
            .unwrap()
            .map_coeff(&|c| c.re.to_f64());
        let mut output = [0.; 8];
        evaluator.evaluate(&[2., 1.], &mut output);
        assert_eq!(output, [2., 1., 2., 1., 3.5, 0., 4., 4.]);
    }
}

#[test]
fn import_preserves_lazy_control_flow_and_branch_local_values() {
    use Slot::{Const, Out, Param, Temp};

    let positive = symbol!(
        "symbolica::import_test::positive",
        eval = EvaluationInfo::new().register(|args: &[f64]| {
            assert!(args[0] > 0., "inactive branch must not execute");
            args[0] * 2.
        })
    );
    let exported = instructions(
        1,
        2,
        2,
        vec![5.],
        vec![
            Instruction::IfElse(Param(0), 7),
            Instruction::Fun(Temp(0), Box::new((positive, vec![], vec![Param(0)])), true),
            Instruction::Goto(9),
            Instruction::Label(7),
            Instruction::Assign(Temp(1), Const(0)),
            Instruction::Label(9),
            Instruction::Join(Out(0), Param(0), Temp(0), Temp(1)),
            Instruction::Add(Out(1), vec![Out(0), Const(0)], 2),
        ],
    );
    let mut evaluator = ExpressionEvaluator::from_instructions(&exported).unwrap();
    let mut output = [0.; 2];
    for (input, expected) in [(0., [5., 10.]), (2., [4., 9.]), (0., [5., 10.])] {
        evaluator.evaluate(&[input], &mut output);
        assert_eq!(output, expected);
    }
}

#[test]
fn import_preserves_expression_tags_for_registered_functions() {
    let tags = expression_tags();
    let expected = tags.clone();
    let tagged = symbol!(
        "symbolica::import_test::tagged",
        eval = EvaluationInfo::new()
            .with_tags(4)
            .register_tagged(move |tags| {
                assert_eq!(
                    tags[1..]
                        .iter()
                        .map(|tag| tag.to_owned())
                        .collect::<Vec<_>>(),
                    expected[1..],
                );
                let scale = if tags[0] == 2 { 3. } else { 11. };
                Box::new(move |args: &[f64]| scale * args[0])
            })
    );
    let mut arguments = tags.clone();
    arguments.push(parse!("x"));
    let first = tagged.call(&arguments);
    arguments[0] = Atom::num(3);
    let expressions = [first, tagged.call(&arguments)];
    for direct in [false, true] {
        let original = Atom::evaluator_multiple(&expressions, &[parse!("x")])
            .direct_translation(direct)
            .build()
            .unwrap()
            .map_coeff(&|c| c.re.to_f64());
        let mut exported = original.export_instructions();
        let mut imported = ExpressionEvaluator::from_instructions(&exported).unwrap();
        let mut output = [0.; 2];
        imported.evaluate(&[2.], &mut output);
        assert_eq!(output, [6., 22.]);
        assert!(
            imported
                .export_instructions()
                .instructions
                .iter()
                .any(|instruction| {
                    matches!(instruction, Instruction::Fun(_, call, _) if call.1 == tags)
                })
        );

        let call = exported
            .instructions
            .iter_mut()
            .find_map(|instruction| {
                if let Instruction::Fun(_, call, _) = instruction {
                    Some(call)
                } else {
                    None
                }
            })
            .unwrap();
        call.1.pop();
        assert!(ExpressionEvaluator::from_instructions(&exported).is_err());
    }
}

#[test]
fn import_dispatches_retained_bodies_by_expression_tags() {
    let function = symbol!("symbolica::import_test::tagged_body");
    let x = symbol!("x");
    let y = symbol!("y");
    let first_tags = expression_tags();
    let mut second_tags = first_tags.clone();
    second_tags[0] = Atom::num(3);
    let mut functions = FunctionMap::new();
    for (tags, body) in [
        (first_tags.clone(), parse!("y^2")),
        (second_tags.clone(), parse!("y + 7")),
    ] {
        functions
            .add_tagged_function_with_options(
                function,
                tags,
                vec![y],
                body,
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap();
    }
    let expressions = [first_tags.clone(), second_tags.clone()].map(|mut tags| {
        tags.push(x.to_atom());
        function.call(&tags)
    });
    let original = Atom::evaluator_multiple(&expressions, &[x.to_atom()])
        .function_map(functions)
        .build()
        .unwrap();
    let exported = original.export_instructions();
    assert_eq!(exported.sub_evaluators.len(), 2);
    let imported = ExpressionEvaluator::from_instructions(&exported).unwrap();
    let reexported = imported.export_instructions();
    assert!(
        reexported
            .sub_evaluators
            .iter()
            .any(|body| body.tags == first_tags)
    );
    assert!(
        reexported
            .sub_evaluators
            .iter()
            .any(|body| body.tags == second_tags)
    );
    let mut restored = ExpressionEvaluator::from_instructions(&reexported)
        .unwrap()
        .map_coeff(&|c| c.re.to_f64());
    let mut output = [0.; 2];
    restored.evaluate(&[3.], &mut output);
    assert_eq!(output, [9., 10.]);
}

#[test]
fn import_nested_shared_function_bodies_with_captures_and_branches() {
    let _constant = symbol!(
        "symbolica::import_test::nested_constant",
        eval = EvaluationInfo::constant(|_, precision| Ok(Float::with_val(precision, 2).into()))
    );
    let mut functions = FunctionMap::new();
    for (name, body) in [
        (
            symbol!("symbolica::import_test::helper"),
            parse!("if(z, z^2 + x + symbolica::import_test::nested_constant, x + 2)"),
        ),
        (
            symbol!("symbolica::import_test::left"),
            parse!("symbolica::import_test::helper(z) + 3"),
        ),
        (
            symbol!("symbolica::import_test::right"),
            parse!("symbolica::import_test::helper(z + 1) + 4"),
        ),
    ] {
        functions
            .add_function_with_options(
                name,
                vec![symbol!("z")],
                body,
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap();
    }
    let expressions = [
        parse!("symbolica::import_test::left(x)"),
        parse!("symbolica::import_test::right(x)"),
        parse!("symbolica::import_test::helper(x + 2)"),
    ];
    let exact = Atom::evaluator_multiple(&expressions, &[parse!("x")])
        .function_map(functions)
        .build()
        .unwrap();
    let exported = exact.export_instructions();
    assert_eq!(exported.constant_functions.len(), 1);
    let helper = exported
        .sub_evaluators
        .iter()
        .find(|sub| sub.symbol == symbol!("symbolica::import_test::helper"))
        .unwrap();
    for wrapper in exported
        .sub_evaluators
        .iter()
        .filter(|sub| sub.symbol != helper.symbol)
    {
        assert!(Arc::ptr_eq(
            &helper.instructions,
            &wrapper.instructions.sub_evaluators[0].instructions,
        ));
    }
    let mut imported = ExpressionEvaluator::from_instructions(&exported)
        .unwrap()
        .map_coeff(&|c| c.re.to_f64());
    let mut output = [0.; 3];
    for (input, expected) in [
        (2., [11., 17., 20.]),
        (0., [5., 7., 6.]),
        (-1., [5., 5., 2.]),
    ] {
        imported.evaluate(&[input], &mut output);
        assert_eq!(output, expected);
    }
}

#[test]
fn import_retains_external_constants_and_fixed_arguments_at_target_precision() {
    let tags = expression_tags();
    let expected = tags.clone();
    let constant = symbol!(
        "symbolica::import_test::precision",
        // Encode the requested precision in the value to detect eager resolution
        // or loss of the definition during an exact or numeric round trip.
        eval = EvaluationInfo::constant(move |tags, precision| {
            assert_eq!(
                tags.iter().map(|tag| tag.to_owned()).collect::<Vec<_>>(),
                expected,
            );
            Ok(Float::with_val(precision, precision).into())
        })
        .with_tags(4)
    );
    let expected_complex = tags.clone();
    let expected_real = tags.clone();
    let square = symbol!(
        "symbolica::import_test::square",
        eval = EvaluationInfo::new()
            .with_tags(4)
            .register_tagged(move |tags| {
                assert_eq!(
                    tags.iter().map(|tag| tag.to_owned()).collect::<Vec<_>>(),
                    expected_complex,
                );
                Box::new(|args: &[Complex<Float>]| args[0].clone() * &args[0])
            })
            .register_tagged(move |tags| {
                assert_eq!(
                    tags.iter().map(|tag| tag.to_owned()).collect::<Vec<_>>(),
                    expected_real,
                );
                Box::new(|args: &[f64]| args[0] * args[0])
            })
    );
    let mut fixed_arguments = tags.clone();
    fixed_arguments.push(parse!("3/2"));
    let exact = (constant.call(&tags) + square.call(&fixed_arguments) + 1)
        .evaluator::<Atom>(&[])
        .build()
        .unwrap();
    let exported = exact.export_instructions();
    assert_eq!(exported.constant_functions.len(), 2);
    assert!(
        exported
            .constant_functions
            .iter()
            .all(|constant| constant.tags == tags)
    );
    assert!(exported.constant_functions.iter().any(|constant| {
        constant.symbol == symbol!("symbolica::import_test::square")
            && constant.fixed_args == vec![Complex::new(Rational::from((3, 2)), Rational::from(0))]
    }));
    let imported = ExpressionEvaluator::from_instructions(&exported).unwrap();
    assert!(
        imported
            .export_instructions()
            .constant_functions
            .iter()
            .all(|constant| constant.tags == tags)
    );
    let mut invalid = exported.clone();
    invalid.constant_functions[0].tags.pop();
    assert!(ExpressionEvaluator::from_instructions(&invalid).is_err());
    for precision in [64, 128] {
        let mut mapped = imported.clone().map_coeff_with_prec(
            &|c| {
                Complex::new(
                    c.re.to_multi_prec_float(precision),
                    c.im.to_multi_prec_float(precision),
                )
            },
            precision,
        );
        let value = mapped.evaluate_single(&[]);
        assert_eq!(value.re.to_f64(), f64::from(precision) + 3.25);
        assert_eq!(value.re.prec(), precision);

        let mut numeric =
            ExpressionEvaluator::from_instructions(&mapped.export_instructions()).unwrap();
        assert_eq!(numeric.evaluate_single(&[]), value);
    }

    let binary64 = imported.map_coeff(&|c| c.re.to_f64());
    let mut numeric =
        ExpressionEvaluator::from_instructions(&binary64.export_instructions()).unwrap();
    assert_eq!(numeric.evaluate_single(&[]), 53. + 3.25);
    let mut remapped = numeric.map_coeff_with_prec(
        &|c| Complex::new(Float::with_val(128, *c), Float::new(128)),
        128,
    );
    assert_eq!(remapped.evaluate_single(&[]).re.to_f64(), 128. + 3.25);
}

#[test]
fn import_rejects_malformed_slots_arithmetic_and_control_flow() {
    use Slot::{Const, Out, Param, Temp};

    let baseline = instructions(
        1,
        1,
        1,
        vec![2.],
        vec![Instruction::Add(Out(0), vec![Param(0), Const(0)], 2)],
    );
    assert_eq!(
        ExpressionEvaluator::from_instructions(&baseline)
            .unwrap()
            .evaluate_single(&[3.]),
        5.
    );
    let invalid = [
        vec![Instruction::Assign(Param(0), Const(0))],
        vec![Instruction::Assign(Const(0), Param(0))],
        vec![Instruction::Assign(Temp(1), Param(0))],
        vec![Instruction::Assign(Out(1), Param(0))],
        vec![Instruction::Assign(Out(0), Param(1))],
        vec![Instruction::Assign(Out(0), Const(1))],
        vec![Instruction::Assign(Out(0), Temp(1))],
        vec![Instruction::Assign(Out(0), Out(1))],
        vec![Instruction::Assign(Out(0), Temp(0))],
        vec![Instruction::Add(Out(0), vec![], 0)],
        vec![Instruction::Mul(Out(0), vec![], 0)],
        vec![Instruction::Add(Out(0), vec![Param(0)], 2)],
        vec![Instruction::Mul(Out(0), vec![Param(0)], 2)],
        vec![Instruction::Goto(99)],
        vec![Instruction::Label(4), Instruction::Label(4)],
        vec![Instruction::Label(4), Instruction::Goto(4)],
        vec![
            Instruction::IfElse(Param(0), 1),
            Instruction::Assign(Out(0), Param(0)),
            Instruction::Label(1),
        ],
        vec![Instruction::Join(Out(0), Param(0), Temp(0), Const(0))],
        vec![Instruction::Fun(
            Out(0),
            Box::new((Symbol::SIN, vec![], vec![])),
            true,
        )],
    ];
    for instructions in invalid {
        let mut malformed = baseline.clone();
        malformed.instructions = instructions;
        assert!(
            ExpressionEvaluator::from_instructions(&malformed).is_err(),
            "accepted malformed stream: {:?}",
            malformed.instructions
        );
    }
}

#[test]
fn import_rejects_inconsistent_sub_evaluator_signatures() {
    use Slot::{Out, Param};

    let function = symbol!("symbolica::import_test::identity_body");
    let body = Arc::new(instructions(
        1,
        1,
        0,
        vec![],
        vec![Instruction::Assign(Out(0), Param(0))],
    ));
    let mut exported = instructions(
        1,
        1,
        0,
        vec![],
        vec![Instruction::Fun(
            Out(0),
            Box::new((function, vec![], vec![Param(0)])),
            true,
        )],
    );
    exported.sub_evaluators.push(ExportedSubEvaluator {
        symbol: function,
        tags: vec![],
        input_count: 1,
        output_count: 1,
        instructions: body,
    });
    assert_eq!(
        ExpressionEvaluator::from_instructions(&exported)
            .unwrap()
            .evaluate_single(&[7.]),
        7.
    );

    let mut bad_metadata = exported.clone();
    bad_metadata.sub_evaluators[0].input_count = 2;
    assert!(ExpressionEvaluator::from_instructions(&bad_metadata).is_err());
    let mut bad_arity = exported.clone();
    bad_arity.instructions = vec![Instruction::Fun(
        Out(0),
        Box::new((function, vec![], vec![])),
        true,
    )];
    assert!(ExpressionEvaluator::from_instructions(&bad_arity).is_err());
    exported
        .sub_evaluators
        .push(exported.sub_evaluators[0].clone());
    assert!(ExpressionEvaluator::from_instructions(&exported).is_err());
}

#[test]
fn import_rejects_invalid_constant_function_slots() {
    let exact = Atom::var(Symbol::PI)
        .evaluator::<Atom>(&[])
        .build()
        .unwrap();
    let exported = exact.export_instructions();
    assert_eq!(exported.constant_functions.len(), 1);
    let mut imported = ExpressionEvaluator::from_instructions(&exported)
        .unwrap()
        .map_coeff(&|c| c.re.to_f64());
    assert_eq!(imported.evaluate_single(&[]), std::f64::consts::PI);

    let mut out_of_bounds = exported.clone();
    out_of_bounds.constant_functions[0].index = exported.constants.len();
    assert!(ExpressionEvaluator::from_instructions(&out_of_bounds).is_err());

    let mut duplicate = exported;
    duplicate
        .constant_functions
        .push(duplicate.constant_functions[0].clone());
    assert!(ExpressionEvaluator::from_instructions(&duplicate).is_err());
}
