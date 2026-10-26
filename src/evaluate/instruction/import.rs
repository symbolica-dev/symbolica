use super::*;

impl<T: FloatLike + EvaluationDomain> ExpressionEvaluator<T> {
    /// Load an evaluator from its structured instruction representation.
    ///
    /// Constants retain their exported values and precision. Registered constant
    /// definitions are retained as well, so a later [`Self::map_coeff_with_prec`]
    /// resolves them at the requested precision. External callbacks are looked up
    /// in their symbols' registered evaluation information, just as when loading
    /// a serialized evaluator; an unavailable callback remains unresolved.
    /// Function tags retain their original symbolic expressions.
    ///
    /// Invalid slots, instructions, control flow, and function signatures return
    /// an error. Jumps must be forward, and every read and output must be defined
    /// on the path that uses it. Function bodies must have one output and no local
    /// constants, as produced by [`Self::export_instructions`]; their constants
    /// are passed through the exported hidden parameters.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::prelude::*;
    /// use symbolica::evaluate::ExpressionEvaluator;
    ///
    /// let evaluator = parse!("x^2 + 1").evaluator(&[parse!("x")]).build().unwrap();
    /// let restored = ExpressionEvaluator::from_instructions(&evaluator.export_instructions())
    ///     .unwrap();
    /// let mut numerical = restored.map_coeff(&|c| c.re.to_f64());
    /// assert_eq!(numerical.evaluate_single(&[3.]), 10.);
    /// ```
    pub fn from_instructions(exported: &ExportedInstructions<T>) -> Result<Self, String> {
        let mut importer = Importer::default();
        let program = importer.program(exported, false)?;
        let mut stack = Vec::new();
        stack
            .try_reserve_exact(program.stack_size)
            .map_err(|e| format!("Cannot allocate evaluator storage: {e}"))?;
        stack.resize_with(exported.input_count, T::new_zero);
        stack.extend(exported.constants.iter().cloned());
        stack.resize_with(program.stack_size, T::new_zero);
        Ok(Self {
            stack,
            param_count: exported.input_count,
            reserved_indices: program.reserved,
            instructions: program.instructions,
            result_indices: (program.outputs..program.stack_size).collect(),
            external_fns: importer.external,
            settings: OptimizationSettings::default(),
        })
    }
}

type Signature = (Symbol, Vec<Atom>);

struct Importer<T> {
    external: Vec<ExternalFunctionContainer<T>>,
    callbacks: HashMap<Signature, usize>,
    bodies: HashMap<(Symbol, Vec<Atom>, usize), usize>,
    active_bodies: HashSet<usize>,
}

impl<T> Default for Importer<T> {
    fn default() -> Self {
        Self {
            external: Vec::new(),
            callbacks: HashMap::default(),
            bodies: HashMap::default(),
            active_bodies: HashSet::default(),
        }
    }
}

struct ImportedProgram {
    instructions: Vec<(Instr, ComplexPhase)>,
    reserved: usize,
    outputs: usize,
    stack_size: usize,
}

impl<T: EvaluationDomain> Importer<T> {
    fn container(
        symbol: Symbol,
        tags: Vec<Atom>,
        fixed_args: Vec<Complex<Rational>>,
    ) -> Result<ExternalFunctionContainer<T>, String> {
        if symbol.get_ascii_name().is_none() {
            return Err(format!("Function {symbol} has no ASCII export name"));
        }
        Ok(ExternalFunctionContainer::new(symbol, tags, fixed_args))
    }

    fn check_tags(symbol: Symbol, tags: &[Atom]) -> Result<(), String> {
        if let Some(info) = symbol.get_evaluation_info()
            && info.get_tag_count() != tags.len()
        {
            return Err(format!(
                "Function {symbol} expects {} tags, got {}",
                info.get_tag_count(),
                tags.len()
            ));
        }
        Ok(())
    }

    fn callback(&mut self, signature: Signature) -> Result<usize, String> {
        if let Some(index) = self.callbacks.get(&signature) {
            return Ok(*index);
        }
        Self::check_tags(signature.0, &signature.1)?;
        let mut external = Self::container(signature.0, signature.1.clone(), vec![])?;
        external.imp = external.fetch_impl_for::<T>();
        let index = self.external.len();
        self.external.push(external);
        self.callbacks.insert(signature, index);
        Ok(index)
    }

    fn body(&mut self, sub: &ExportedSubEvaluator<T>, tags: Vec<Atom>) -> Result<usize, String> {
        if sub.input_count != sub.instructions.input_count
            || sub.output_count != sub.instructions.output_count
        {
            return Err(format!(
                "Function {} has inconsistent input/output counts",
                sub.symbol
            ));
        }
        if sub.output_count != 1 {
            return Err(format!(
                "Function {} must have exactly one output",
                sub.symbol
            ));
        }
        let pointer = Arc::as_ptr(&sub.instructions) as usize;
        let key = (sub.symbol, tags.clone(), pointer);
        if let Some(index) = self.bodies.get(&key) {
            return Ok(*index);
        }
        if !self.active_bodies.insert(pointer) {
            return Err(format!("Function {} has a recursive body", sub.symbol));
        }
        let program = self.program(&sub.instructions, true)?;
        self.active_bodies.remove(&pointer);
        let mut external = Self::container(sub.symbol, tags, vec![])?;
        external.body = Some(Arc::new(FunctionBody {
            instructions: program.instructions,
            param_count: sub.input_count,
            result_index: program.outputs,
            stack_size: program.stack_size,
        }));
        // Every callee was imported recursively before this caller, as required
        // by the interpreter's disjoint mutable function storage.
        let index = self.external.len();
        self.external.push(external);
        self.bodies.insert(key, index);
        Ok(index)
    }

    fn program(
        &mut self,
        exported: &ExportedInstructions<T>,
        is_body: bool,
    ) -> Result<ImportedProgram, String> {
        if is_body && (!exported.constants.is_empty() || !exported.constant_functions.is_empty()) {
            return Err("Function bodies must pass constants as hidden parameters".into());
        }
        let reserved = exported.input_count.checked_add(exported.constants.len());
        let outputs = reserved.and_then(|n| n.checked_add(exported.temporary_count));
        let stack_size = outputs.and_then(|n| n.checked_add(exported.output_count));
        let (Some(reserved), Some(outputs), Some(stack_size)) = (reserved, outputs, stack_size)
        else {
            return Err("Evaluator storage size overflows usize".into());
        };
        // Validate allocation limits before constructing index sets or bodies.
        if stack_size > isize::MAX as usize / std::mem::size_of::<T>().max(1) {
            return Err("Evaluator storage is too large".into());
        }
        let slot = |slot: Slot| -> Result<usize, String> {
            match slot {
                Slot::Param(i) if i < exported.input_count => Ok(i),
                Slot::Const(i) if i < exported.constants.len() => Ok(exported.input_count + i),
                Slot::Temp(i) if i < exported.temporary_count => Ok(reserved + i),
                Slot::Out(i) if i < exported.output_count => Ok(outputs + i),
                _ => Err(format!("Slot {slot} is out of bounds")),
            }
        };
        let destination = |output: Slot| -> Result<usize, String> {
            if !matches!(output, Slot::Temp(_) | Slot::Out(_)) {
                return Err(format!("Cannot write to immutable slot {output}"));
            }
            slot(output)
        };

        let mut constant_slots = HashSet::default();
        for constant in &exported.constant_functions {
            if constant.index >= exported.constants.len() {
                return Err(format!(
                    "Constant function slot {} is out of bounds",
                    constant.index
                ));
            }
            if !constant_slots.insert(constant.index) {
                return Err(format!(
                    "Constant slot {} has multiple definitions",
                    constant.index
                ));
            }
            let tags = constant.tags.clone();
            Self::check_tags(constant.symbol, &tags)?;
            let mut external = Self::container(constant.symbol, tags, constant.fixed_args.clone())?;
            external.constant_index = Some(constant.index);
            external.imp = external.fetch_impl_for::<T>();
            self.external.push(external);
        }

        let mut local_functions = HashMap::default();
        for sub in &exported.sub_evaluators {
            let tags = sub.tags.clone();
            let signature = (sub.symbol, tags.clone());
            if local_functions.contains_key(&signature) {
                return Err(format!(
                    "Function {} has multiple local definitions",
                    sub.symbol
                ));
            }
            let index = self.body(sub, tags)?;
            local_functions.insert(signature, (index, sub.input_count));
        }

        let mut labels = HashMap::default();
        for (position, instruction) in exported.instructions.iter().enumerate() {
            if let Instruction::Label(label) = instruction
                && labels.insert(*label, position).is_some()
            {
                return Err(format!("Label {label} is defined more than once"));
            }
        }
        let mut instructions = Vec::with_capacity(exported.instructions.len());
        for (position, instruction) in exported.instructions.iter().enumerate() {
            let target = |label: usize| -> Result<Label, String> {
                let target = *labels
                    .get(&label)
                    .ok_or_else(|| format!("Unknown label {label}"))?;
                if target <= position {
                    return Err(format!("Jump to label {label} must be forward"));
                }
                Ok(Label(target))
            };
            let real_phase = |real| {
                if real {
                    ComplexPhase::Real
                } else {
                    ComplexPhase::Any
                }
            };
            let lowered = match instruction {
                Instruction::Add(output, args, real) | Instruction::Mul(output, args, real) => {
                    if args.is_empty() || *real > args.len() {
                        return Err(format!("Invalid argument count at instruction {position}"));
                    }
                    let args = args
                        .iter()
                        .copied()
                        .map(slot)
                        .collect::<Result<Vec<_>, _>>()?;
                    let phase = if *real == args.len() {
                        ComplexPhase::Real
                    } else if *real == 0 {
                        ComplexPhase::Any
                    } else {
                        ComplexPhase::PartialReal(*real)
                    };
                    let output = destination(*output)?;
                    (
                        if matches!(instruction, Instruction::Add(..)) {
                            Instr::Add(output, args)
                        } else {
                            Instr::Mul(output, args)
                        },
                        phase,
                    )
                }
                Instruction::Pow(output, base, exponent, real) => (
                    Instr::Pow(destination(*output)?, slot(*base)?, *exponent),
                    real_phase(*real),
                ),
                Instruction::Powf(output, base, exponent, real) => (
                    Instr::Powf(destination(*output)?, slot(*base)?, slot(*exponent)?),
                    real_phase(*real),
                ),
                Instruction::Assign(output, input) => (
                    Instr::Add(destination(*output)?, vec![slot(*input)?]),
                    ComplexPhase::Any,
                ),
                Instruction::Fun(output, function, real) => {
                    let (symbol, tags, arguments) = &**function;
                    let args = arguments
                        .iter()
                        .copied()
                        .map(slot)
                        .collect::<Result<Vec<_>, _>>()?;
                    let output = destination(*output)?;
                    let signature = (*symbol, tags.clone());
                    let instruction =
                        if let Some((index, input_count)) = local_functions.get(&signature) {
                            if args.len() != *input_count {
                                return Err(format!(
                                    "Function {symbol} expects {input_count} inputs, got {}",
                                    args.len()
                                ));
                            }
                            Instr::ExternalFun(output, *index, args)
                        } else if symbol.is_fixed_builtin() {
                            if !matches!(
                                symbol.get_id(),
                                Symbol::EXP_ID
                                    | Symbol::LOG_ID
                                    | Symbol::SIN_ID
                                    | Symbol::COS_ID
                                    | Symbol::SQRT_ID
                                    | Symbol::ABS_ID
                                    | Symbol::CONJ_ID
                            ) || !signature.1.is_empty()
                                || args.len() != 1
                            {
                                return Err(format!(
                                    "Invalid built-in function instruction for {symbol}"
                                ));
                            }
                            Instr::BuiltinFun(output, *symbol, args[0])
                        } else {
                            Instr::ExternalFun(output, self.callback(signature)?, args)
                        };
                    (instruction, real_phase(*real))
                }
                Instruction::IfElse(condition, label) => (
                    Instr::IfElse(slot(*condition)?, target(*label)?),
                    ComplexPhase::Any,
                ),
                Instruction::Goto(label) => (Instr::Goto(target(*label)?), ComplexPhase::Any),
                Instruction::Label(_) => (Instr::Label(Label(position)), ComplexPhase::Any),
                Instruction::Join(output, condition, yes, no) => (
                    Instr::Join(
                        destination(*output)?,
                        slot(*condition)?,
                        slot(*yes)?,
                        slot(*no)?,
                    ),
                    ComplexPhase::Any,
                ),
            };
            instructions.push(lowered);
        }
        validate_initialization(&instructions, reserved, outputs, stack_size)?;
        Ok(ImportedProgram {
            instructions,
            reserved,
            outputs,
            stack_size,
        })
    }
}

// Definite initialization for a forward control-flow graph. Conditional sets
// describe slots initialized on every path where a condition is true / false.
// This permits Join to read a branch-local result without reading the untaken
// branch. None means that truth value is unreachable on the incoming paths.
#[derive(Clone, Default)]
struct Initialized {
    slots: HashSet<usize>,
    conditions: HashMap<usize, [Option<HashSet<usize>>; 2]>,
}

impl Initialized {
    fn write(&mut self, output: usize) {
        self.conditions.remove(&output);
        self.slots.insert(output);
        for alternatives in self.conditions.values_mut() {
            for slots in alternatives.iter_mut().flatten() {
                slots.insert(output);
            }
        }
    }

    fn branch(mut self, condition: usize, truth: bool) -> Option<Self> {
        let selected = usize::from(truth);
        if let Some(alternatives) = self.conditions.get(&condition) {
            self.slots = alternatives[selected].clone()?;
        }
        for alternatives in self.conditions.values_mut() {
            for slots in alternatives.iter_mut().flatten() {
                slots.extend(self.slots.iter().copied());
            }
        }
        let mut alternatives = [None, None];
        alternatives[selected] = Some(self.slots.clone());
        self.conditions.insert(condition, alternatives);
        Some(self)
    }

    fn merge(&mut self, other: Self) {
        let keys: HashSet<_> = self
            .conditions
            .keys()
            .chain(other.conditions.keys())
            .copied()
            .collect();
        for condition in keys {
            let ours = self
                .conditions
                .entry(condition)
                .or_insert_with(|| [Some(self.slots.clone()), Some(self.slots.clone())]);
            for truth in 0..2 {
                let theirs = other
                    .conditions
                    .get(&condition)
                    .map(|alternatives| alternatives[truth].as_ref())
                    .unwrap_or(Some(&other.slots));
                match (&mut ours[truth], theirs) {
                    (Some(ours), Some(theirs)) => ours.retain(|slot| theirs.contains(slot)),
                    (ours @ None, Some(theirs)) => *ours = Some(theirs.clone()),
                    _ => {}
                }
            }
        }
        self.slots.retain(|slot| other.slots.contains(slot));
    }
}

fn validate_initialization(
    instructions: &[(Instr, ComplexPhase)],
    reserved: usize,
    outputs: usize,
    stack_size: usize,
) -> Result<(), String> {
    let mut incoming = vec![None; instructions.len() + 1];
    incoming[0] = Some(Initialized::default());
    let propagate = |incoming: &mut [Option<Initialized>], index: usize, state: Initialized| {
        if let Some(current) = &mut incoming[index] {
            current.merge(state);
        } else {
            incoming[index] = Some(state);
        }
    };
    for (position, (instruction, _)) in instructions.iter().enumerate() {
        let Some(mut state) = incoming[position].take() else {
            continue;
        };
        let read = |slot: usize| -> Result<(), String> {
            if slot < reserved || state.slots.contains(&slot) {
                Ok(())
            } else {
                Err(format!(
                    "Instruction {position} reads an uninitialized slot"
                ))
            }
        };
        let output = match instruction {
            Instr::Add(output, args)
            | Instr::Mul(output, args)
            | Instr::ExternalFun(output, _, args) => {
                for arg in args {
                    read(*arg)?;
                }
                Some(*output)
            }
            Instr::Pow(output, base, _) | Instr::BuiltinFun(output, _, base) => {
                read(*base)?;
                Some(*output)
            }
            Instr::Powf(output, base, exponent) => {
                read(*base)?;
                read(*exponent)?;
                Some(*output)
            }
            Instr::Join(output, condition, yes, no) => {
                read(*condition)?;
                for (truth, value) in [(1, *yes), (0, *no)] {
                    let initialized = state
                        .conditions
                        .get(condition)
                        .map(|alternatives| alternatives[truth].as_ref())
                        .unwrap_or(Some(&state.slots));
                    if value >= reserved && initialized.is_some_and(|slots| !slots.contains(&value))
                    {
                        return Err(format!(
                            "Join at instruction {position} reads an uninitialized branch value"
                        ));
                    }
                }
                Some(*output)
            }
            Instr::IfElse(condition, label) => {
                read(*condition)?;
                if let Some(branch) = state.clone().branch(*condition, false) {
                    propagate(&mut incoming, label.0, branch);
                }
                if let Some(branch) = state.branch(*condition, true) {
                    propagate(&mut incoming, position + 1, branch);
                }
                continue;
            }
            Instr::Goto(label) => {
                propagate(&mut incoming, label.0, state);
                continue;
            }
            Instr::Label(_) => None,
        };
        if let Some(output) = output {
            state.write(output);
        }
        propagate(&mut incoming, position + 1, state);
    }
    if let Some(state) = incoming.last().unwrap() {
        for output in outputs..stack_size {
            if !state.slots.contains(&output) {
                return Err(format!(
                    "Output {} is not initialized on every path",
                    output - outputs
                ));
            }
        }
    }
    Ok(())
}
