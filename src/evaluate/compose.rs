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
