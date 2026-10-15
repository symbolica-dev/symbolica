use super::*;
mod optimize;

/// Assemble scalar evaluator programs using their existing input/output slots.
///
/// Appended evaluators are inlined as instructions; no symbolic expressions are
/// reconstructed. Constants, external callbacks and control-flow labels retain
/// their native ownership. Slots belong to this composer and must refer to an
/// input or a result returned by an earlier append.
pub struct EvaluatorComposer<T> {
    parameters: usize,
    instructions: InstructionList<T>,
    external_functions: Vec<ExternalFunctionContainer<T>>,
    next_label: usize,
}

impl<T: Default + Clone> EvaluatorComposer<T> {
    /// Start a program with `parameters` scalar inputs.
    pub fn new(parameters: usize) -> Self {
        Self {
            parameters,
            instructions: InstructionList {
                instructions: Vec::new(),
                constants: Vec::new(),
                unknown_constants: Vec::new(),
                dim: 1,
            },
            external_functions: Vec::new(),
            next_label: 0,
        }
    }

    fn valid(&self, slot: Slot) -> bool {
        match slot {
            Slot::Param(index) => index < self.parameters,
            Slot::Const(index) => index < self.instructions.constants.len(),
            Slot::Temp(index) => index < self.instructions.instructions.len(),
            Slot::Out(_) => false,
        }
    }

    /// Append an evaluator with the supplied input bindings, returning all outputs.
    ///
    /// Out-of-range slots, incorrect input counts and non-inlined function bodies
    /// are rejected before changing the composer. Compile registered functions
    /// with the default inlining policy before composing them.
    pub fn append(
        &mut self,
        evaluator: &ExpressionEvaluator<T>,
        inputs: &[Slot],
    ) -> Result<Vec<Slot>, String> {
        if evaluator.param_count != inputs.len() {
            return Err(format!(
                "Expected {} inputs, received {}",
                evaluator.param_count,
                inputs.len()
            ));
        }
        if let Some(slot) = inputs.iter().find(|slot| !self.valid(**slot)) {
            return Err(format!("Invalid evaluator composition input slot {slot}"));
        }
        if let Some(function) = evaluator
            .external_fns
            .iter()
            .find(|function| function.body.is_some())
        {
            return Err(format!(
                "Sub-evaluator '{}' must be inlined before composition",
                function.symbol
            ));
        }
        let mut evaluator = evaluator.clone();
        evaluator.undo_stack_optimization();
        Ok(evaluator.inline_vector_components(
            inputs,
            &mut self.instructions,
            &mut self.external_functions,
            &mut self.next_label,
        ))
    }

    /// Finish the program with selected/reordered outputs and native optimization.
    /// Straight-line programs discard instructions and callback constants that do
    /// not contribute to an output. Programs with control flow are kept intact.
    /// Literal and callback constants are deduplicated before native CSE/CPE.
    pub fn finish(
        mut self,
        outputs: &[Slot],
        settings: OptimizationSettings,
    ) -> Result<ExpressionEvaluator<T>, String>
    where
        T: Eq + Hash,
    {
        if let Some(slot) = outputs.iter().find(|slot| !self.valid(**slot)) {
            return Err(format!("Invalid evaluator composition output slot {slot}"));
        }
        let rounds = settings.cpe_iterations;
        let mut outputs = outputs.to_vec();
        self.prune_unused(&mut outputs);
        self.deduplicate_constants(&mut outputs);
        Ok(ExpressionEvaluator::from_instruction_list(
            self.parameters,
            self.instructions,
            self.external_functions,
            outputs,
            settings,
            rounds,
            true,
        ))
    }
}

impl<T: Default + Clone> ExpressionEvaluator<T> {
    /// Lower scalar instruction storage shared by native vectorization/composition.
    pub(super) fn from_instruction_list(
        param_count: usize,
        instructions: InstructionList<T>,
        external_fns: Vec<ExternalFunctionContainer<T>>,
        outputs: Vec<Slot>,
        settings: OptimizationSettings,
        cpe_rounds: Option<usize>,
        eliminate_common_instructions: bool,
    ) -> Self {
        let reserved_indices = param_count + instructions.constants.len();
        let index = |slot| match slot {
            Slot::Param(index) => index,
            Slot::Const(index) => param_count + index,
            Slot::Temp(index) => reserved_indices + index,
            Slot::Out(_) => unreachable!("composition validates all slots"),
        };
        let mut lowered = Vec::with_capacity(instructions.instructions.len());
        for instruction in instructions.instructions {
            let out = reserved_indices + lowered.len();
            let instruction = match instruction {
                VectorInstruction::Add(a, b) | VectorInstruction::Mul(a, b) => {
                    let mut inputs = vec![index(a), index(b)];
                    inputs.sort_unstable();
                    if matches!(instruction, VectorInstruction::Add(..)) {
                        Instr::Add(out, inputs)
                    } else {
                        Instr::Mul(out, inputs)
                    }
                }
                VectorInstruction::Assign(a) => Instr::Add(out, vec![index(a)]),
                VectorInstruction::Pow(a, power) => Instr::Pow(out, index(a), power),
                VectorInstruction::Powf(a, b) => Instr::Powf(out, index(a), index(b)),
                VectorInstruction::BuiltinFun(symbol, a) => {
                    Instr::BuiltinFun(out, symbol, index(a))
                }
                VectorInstruction::ExternalFun(function, args) => {
                    Instr::ExternalFun(out, function, args.into_iter().map(index).collect())
                }
                VectorInstruction::IfElse(condition, label) => {
                    Instr::IfElse(index(condition), label)
                }
                VectorInstruction::Goto(label) => Instr::Goto(label),
                VectorInstruction::Label(label) => Instr::Label(label),
                VectorInstruction::Join(condition, a, b) => {
                    Instr::Join(out, index(condition), index(a), index(b))
                }
            };
            lowered.push((instruction, ComplexPhase::Any));
        }
        let mut stack = vec![T::default(); param_count];
        stack.extend(instructions.constants);
        stack.resize(reserved_indices + lowered.len(), T::default());
        let mut evaluator = Self {
            stack,
            param_count,
            reserved_indices,
            instructions: lowered,
            result_indices: outputs.into_iter().map(index).collect(),
            external_fns,
            settings,
        };
        if eliminate_common_instructions {
            loop {
                if evaluator.settings.abort_level > 0 || evaluator.remove_common_instructions() == 0
                {
                    evaluator.settings.abort_level = 0;
                    break;
                }
            }
        }
        for _ in 0..cpe_rounds.unwrap_or(usize::MAX) {
            if (eliminate_common_instructions && evaluator.settings.abort_level > 0)
                || evaluator.remove_common_pairs() == 0
            {
                if eliminate_common_instructions {
                    evaluator.settings.abort_level = 0;
                }
                break;
            }
        }
        evaluator.optimize_stack();
        evaluator.fix_labels();
        evaluator
    }
}

#[cfg(test)]
mod composition_tests {
    use crate::{
        evaluate::{EvaluatorComposer, FunctionRegistrationOptions, InliningPolicy, Slot},
        prelude::*,
    };
    #[test]
    fn composition_preserves_native_control_flow_callbacks_and_admission() {
        let settings = OptimizationSettings::new()
            .horner_iterations(0)
            .cores(1)
            .cpe_iterations(Some(1000));
        let p = Atom::evaluator_multiple(
            &[
                parse!("x+y+x*y"),
                parse!("if(x,if(y,x+y,x-7),y+11)"),
                parse!("gamma(x+1)+gamma(1/2)"),
            ],
            &[parse!("x"), parse!("y")],
        )
        .optimization_settings(settings.clone())
        .build()
        .unwrap();
        let dependent = parse!("a+b*c")
            .evaluator(&[parse!("a"), parse!("b"), parse!("c")])
            .optimization_settings(settings.clone())
            .build()
            .unwrap();
        let constant = Atom::evaluator_multiple(
            &[
                parse!("-123456789012345678901234567890123456789/7"),
                parse!("gamma(1/2)"),
            ],
            &[] as &[Atom],
        )
        .optimization_settings(settings.clone())
        .build()
        .unwrap();
        let body = parse!("f(x)")
            .evaluator(&[parse!("x")])
            .add_function_with_options(
                symbol!("f"),
                vec![symbol!("x")],
                parse!("x^2+1"),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap()
            .build()
            .unwrap();
        let mut c = EvaluatorComposer::new(2);
        let a = c.append(&p, &[Slot::Param(0), Slot::Param(1)]).unwrap();
        // Rejected work after valid work must preserve its labels, constants and outputs.
        assert!(c.append(&p, &[]).is_err());
        for bad in [
            Slot::Out(0),
            Slot::Temp(usize::MAX),
            Slot::Const(usize::MAX),
            Slot::Param(2),
        ] {
            assert!(c.append(&p, &[bad, Slot::Param(0)]).is_err());
        }
        assert!(c.append(&body, &[Slot::Param(0)]).is_err());
        let b = c.append(&p, &[Slot::Param(1), Slot::Param(0)]).unwrap();
        let k = c.append(&constant, &[]).unwrap();
        let d = c.append(&dependent, &[a[1], b[1], a[2]]).unwrap();
        let selected = vec![
            d[0],
            a[0],
            b[0],
            a[1],
            b[1],
            a[2],
            b[2],
            k[0],
            k[1],
            Slot::Param(1),
            d[0],
        ];
        let exact = c.finish(&selected, settings.clone()).unwrap();
        let mut e = exact.clone().map_coeff(&|q| q.re.to_f64());
        let mut first = p.clone().map_coeff(&|q| q.re.to_f64());
        let mut second = p.clone().map_coeff(&|q| q.re.to_f64());
        let mut fixed = constant.map_coeff(&|q| q.re.to_f64());
        let mut fixed_out = [0.; 2];
        fixed.evaluate(&[], &mut fixed_out);
        #[cfg(feature = "native_code_generation")]
        let mut jit = exact
            .jit_compile::<f64>(
                JITCompilationSettings::default()
                    .optimization_level(2)
                    .with_option("use_threads", "false"),
            )
            .unwrap();
        for (x, y) in [
            (0., 0.),
            (2., 0.),
            (0., 3.),
            (2., 3.),
            (0., 0.),
            (2., 0.),
            (0., 3.),
        ] {
            let mut av = [0.; 3];
            let mut bv = [0.; 3];
            first.evaluate(&[x, y], &mut av);
            second.evaluate(&[y, x], &mut bv);
            let expected = [
                av[1] + bv[1] * av[2],
                av[0],
                bv[0],
                av[1],
                bv[1],
                av[2],
                bv[2],
                fixed_out[0],
                fixed_out[1],
                y,
                av[1] + bv[1] * av[2],
            ];
            let mut actual = [0.; 11];
            let mut j = [0.; 11];
            e.evaluate(&[x, y], &mut actual);
            #[cfg(feature = "native_code_generation")]
            jit.evaluate(&[x, y], &mut j);
            #[cfg(not(feature = "native_code_generation"))]
            j.copy_from_slice(&actual);
            for (i, ((a, b), v)) in actual.into_iter().zip(j).zip(expected).enumerate() {
                assert!(
                    (a - v).abs() <= 1e-12 * v.abs().max(1.),
                    "eager {i}: {a} {v}"
                );
                assert!((b - v).abs() <= 1e-12 * v.abs().max(1.), "jit {i}: {b} {v}");
            }
        }
        assert!(
            EvaluatorComposer::<Complex<Rational>>::new(0)
                .finish(&[Slot::Param(0)], settings.clone())
                .is_err()
        );
        let mut empty = EvaluatorComposer::<Complex<Rational>>::new(0)
            .finish(&[], settings)
            .unwrap()
            .map_coeff(&|q| q.re.to_f64());
        empty.evaluate(&[], &mut []);
        println!(
            "PASS: appended nested branch programs repeatedly toggle both conditions; reordered/fanout/repeated outputs; dynamic gamma and precision-aware gamma constants; huge negative exact rational; zero-input constants; failed append atomicity after valid work (arity, all slot kinds, non-inlined function); empty output and invalid finish; eager and SymJIT O2 agree with independent native program calls."
        );
    }

    #[test]
    fn selected_outputs_prune_dead_jets_and_callbacks_before_optimization() {
        use crate::{domains::dual::HyperDual, evaluate::Dualizer};
        let settings = OptimizationSettings::new()
            .horner_iterations(0)
            .cores(1)
            .cpe_iterations(Some(0));
        let original = Atom::evaluator_multiple(
            &[parse!("x+x^2+x^3+x^4"), parse!("gamma(x)+gamma(1/2)")],
            &[parse!("x")],
        )
        .optimization_settings(settings.clone())
        .build()
        .unwrap();
        let shape = (0..5).map(|i| vec![i]).collect::<Vec<_>>();
        let lowered = original
            .vectorize(&Dualizer::new(
                HyperDual::<Complex<Rational>>::new(shape),
                vec![],
            ))
            .unwrap();
        let original_instructions = lowered.export_instructions().instructions.len();
        let seed = Atom::evaluator_multiple(
            &[
                parse!("x"),
                Atom::num(1),
                Atom::num(0),
                Atom::num(0),
                Atom::num(0),
            ],
            &[parse!("x")],
        )
        .optimization_settings(settings.clone())
        .build()
        .unwrap();
        let mut composed = EvaluatorComposer::new(1);
        let input = composed.append(&seed, &[Slot::Param(0)]).unwrap();
        let output = composed.append(&lowered, &input).unwrap();
        let exact = composed
            .finish(&[output[0], output[1]], settings.clone())
            .unwrap();
        let exported = exact.export_instructions();
        assert!(exported.instructions.len() < original_instructions);
        assert!(exported.constant_functions.is_empty());
        let mut evaluator = exact.map_coeff(&|c| c.re.to_f64());
        for x in [0., 0.25, 1., 2.] {
            let mut values = [0.; 2];
            evaluator.evaluate(&[x], &mut values);
            assert_eq!(
                values,
                [
                    x + x * x + x * x * x + x * x * x * x,
                    1. + 2. * x + 3. * x * x + 4. * x * x * x
                ]
            );
        }

        // A dead callback-only owner contributes neither a constant nor a callback.
        let callback = parse!("gamma(1/2)")
            .evaluator(&[] as &[Atom])
            .optimization_settings(settings.clone())
            .build()
            .unwrap();
        let mut unused = EvaluatorComposer::new(1);
        unused.append(&callback, &[]).unwrap();
        let passthrough = unused.finish(&[Slot::Param(0)], settings).unwrap();
        let exported = passthrough.export_instructions();
        assert!(exported.instructions.iter().all(|instruction| matches!(
            instruction,
            crate::evaluate::Instruction::Assign(Slot::Out(0), Slot::Param(0))
        )));
        assert!(exported.constants.is_empty());
        assert!(exported.constant_functions.is_empty());
    }
}
