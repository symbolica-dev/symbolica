use super::*;

fn evaluator(
    params: usize,
    instructions: Vec<Instr>,
    results: Vec<usize>,
) -> ExpressionEvaluator<f64> {
    ExpressionEvaluator {
        stack: vec![0.; params + instructions.len()],
        param_count: params,
        reserved_indices: params,
        instructions: instructions
            .into_iter()
            .map(|instruction| (instruction, ComplexPhase::Any))
            .collect(),
        result_indices: results,
        external_fns: vec![],
        settings: OptimizationSettings::default(),
    }
}

#[test]
fn dependent_duplicates_collapse_in_one_pass() {
    let depth = 128;
    let copies = 4;
    let mut instructions = vec![];
    let mut results = vec![];
    for _ in 0..copies {
        let mut input = 0;
        for _ in 0..depth {
            let output = instructions.len() + 1;
            instructions.push(Instr::BuiltinFun(output, Symbol::SIN, input));
            input = output;
        }
        results.push(input);
    }
    let mut original = evaluator(1, instructions, results);
    let mut optimized = original.clone();
    assert_eq!(optimized.remove_common_instructions(), (copies - 1) * depth);
    assert_eq!(optimized.instructions.len(), depth);
    assert_eq!(optimized.remove_common_instructions(), 0);
    assert_eq!(optimized.result_indices, vec![depth; copies]);
    for point in [-0.7_f64, 0., 0.3, 1.] {
        let expected = (0..depth).fold(point, |value, _| value.sin());
        let mut before = vec![0.; copies];
        let mut after = vec![0.; copies];
        original.evaluate(&[point], &mut before);
        optimized.evaluate(&[point], &mut after);
        assert_eq!(after, before);
        assert!(after.iter().all(|value| (value - expected).abs() < 1e-14));
    }
}

#[test]
fn remapped_arithmetic_and_function_keys_preserve_operand_order() {
    let mut instructions = vec![];
    let mut results = vec![];
    for _ in 0..3 {
        let start = instructions.len() + 2;
        instructions.extend([
            Instr::Pow(start, 0, 2),
            Instr::BuiltinFun(start + 1, Symbol::SIN, start),
            Instr::Mul(start + 2, vec![0, start + 1]),
            Instr::Add(start + 3, vec![1, start + 2]),
            Instr::Powf(start + 4, start + 3, 1),
            Instr::ExternalFun(start + 5, 0, vec![start + 4, 0]),
        ]);
        results.push(start + 5);
    }
    let end = instructions.len() + 2;
    // Different argument order and a different callee must remain distinct.
    instructions.push(Instr::ExternalFun(end, 0, vec![0, 6]));
    instructions.push(Instr::ExternalFun(end + 1, 1, vec![6, 0]));
    results.extend([end, end + 1]);
    let mut original = evaluator(2, instructions, results);
    for (name, factor) in [("cse_test::f", 2.), ("cse_test::g", 3.)] {
        let mut external = ExternalFunctionContainer::new(crate::symbol!(name), vec![], vec![]);
        external.imp = Some(Box::new(move |args: &[f64]| args[0] + factor * args[1]));
        original.external_fns.push(external);
    }
    let mut optimized = original.clone();
    assert_eq!(optimized.remove_common_instructions(), 12);
    assert_eq!(optimized.instructions.len(), 8);
    assert_eq!(optimized.remove_common_instructions(), 0);
    for point in [[0.2, 2.], [0.7, 3.]] {
        let mut before = [0.; 5];
        let mut after = [0.; 5];
        original.evaluate(&point, &mut before);
        optimized.evaluate(&point, &mut after);
        assert_eq!(after, before);
        assert_eq!(after[0], after[1]);
        assert_eq!(after[1], after[2]);
        assert_ne!(after[0], after[3]);
        assert_ne!(after[0], after[4]);
    }

    // Renaming may reverse the order of originally sorted Add/Mul operands.
    let mut original = evaluator(
        2,
        vec![
            Instr::Pow(2, 0, 2),
            Instr::Pow(3, 1, 2),
            Instr::Pow(4, 0, 2),
            Instr::Add(5, vec![3, 4]),
            Instr::Add(6, vec![2, 3]),
            Instr::Mul(7, vec![3, 4]),
            Instr::Mul(8, vec![2, 3]),
        ],
        vec![5, 6, 7, 8],
    );
    assert_eq!(original.remove_common_instructions(), 3);
    let mut output = [0.; 4];
    original.evaluate(&[2., 3.], &mut output);
    assert_eq!(output, [13., 13., 36., 36.]);
}

#[test]
fn remapped_keys_reuse_parent_but_not_sibling_or_child_branches() {
    let instructions = vec![
        Instr::BuiltinFun(3, Symbol::SIN, 1),
        Instr::IfElse(0, Label(20)),
        Instr::BuiltinFun(5, Symbol::SIN, 1),
        Instr::BuiltinFun(6, Symbol::COS, 5),
        Instr::BuiltinFun(7, Symbol::SIN, 2),
        Instr::BuiltinFun(8, Symbol::COS, 7),
        Instr::Goto(Label(30)),
        Instr::Label(Label(20)),
        Instr::BuiltinFun(11, Symbol::SIN, 2),
        Instr::BuiltinFun(12, Symbol::COS, 11),
        Instr::BuiltinFun(13, Symbol::SIN, 1),
        Instr::BuiltinFun(14, Symbol::COS, 13),
        Instr::Label(Label(30)),
        Instr::Join(16, 0, 8, 12),
        Instr::Join(17, 0, 6, 14),
        Instr::BuiltinFun(18, Symbol::COS, 3),
    ];
    let mut original = evaluator(3, instructions, vec![16, 17, 18]);
    original.fix_labels();
    let mut optimized = original.clone();
    assert_eq!(optimized.remove_common_instructions(), 2);
    assert_eq!(optimized.remove_common_instructions(), 0);
    for condition in [0., 1.] {
        for (x, y) in [(0.2_f64, 0.7_f64), (-0.5, 1.1)] {
            let mut before = [0.; 3];
            let mut after = [0.; 3];
            original.evaluate(&[condition, x, y], &mut before);
            optimized.evaluate(&[condition, x, y], &mut after);
            assert_eq!(after, before);
            assert_eq!(after, [y.sin().cos(), x.sin().cos(), x.sin().cos()]);
        }
    }
}
