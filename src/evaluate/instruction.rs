use super::*;

/// A slot in a list that contains a numerical value.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Slot {
    /// An entry in the list of parameters.
    Param(usize),
    /// An entry in the list of constants.
    Const(usize),
    /// An entry in the list of temporary storage.
    Temp(usize),
    /// An entry in the list of results.
    Out(usize),
}

impl std::fmt::Display for Slot {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Slot::Param(i) => write!(f, "p{i}"),
            Slot::Const(i) => write!(f, "c{i}"),
            Slot::Temp(i) => write!(f, "t{i}"),
            Slot::Out(i) => write!(f, "o{i}"),
        }
    }
}

impl Slot {
    /// Return a slot in the same storage area, offset by `index` entries.
    pub fn index(&self, index: usize) -> Slot {
        match self {
            Slot::Param(i) => Slot::Param(*i + index),
            Slot::Const(i) => Slot::Const(*i + index),
            Slot::Temp(i) => Slot::Temp(*i + index),
            Slot::Out(i) => Slot::Out(*i + index),
        }
    }
}

/// An evaluation instruction. Function tags remain symbolic expressions.
/// Binary decoding uses a [`crate::state::StateMap`] to remap their symbol identifiers.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "bincode",
    derive(bincode_trait_derive::Encode),
    derive(bincode_trait_derive::Decode),
    derive(bincode_trait_derive::BorrowDecodeFromDecode),
    trait_decode(trait = crate::state::HasStateMap)
)]
#[derive(Debug, Clone)]
pub enum Instruction {
    /// `Add(o, [i0,...,i_n])` means `o = i0 + ... + i_n`, where the first
    /// `n_real` arguments are real.
    Add(Slot, Vec<Slot>, usize),
    /// `Mul(o, [i0,...,i_n], n_real)` means `o = i0 * ... * i_n`, where the first
    /// `n_real` arguments are real.
    Mul(Slot, Vec<Slot>, usize),
    /// `Pow(o, b, e, is_real)` means `o = b^e`. The `is_real` flag indicates
    /// whether the exponentiation is expected to yield a real number.
    Pow(Slot, Slot, i64, bool),
    /// `Powf(o, b, e, is_real)` means `o = b^e`. The `is_real` flag indicates
    /// whether the exponentiation is expected to yield a real number.
    Powf(Slot, Slot, Slot, bool),
    /// A function that has a known evaluator or is external, given a symbol name, tags, and arguments.
    /// `Fun(o, (s, t, a), is_real)` means `o = s(t, a)`.
    /// The `is_real` flag indicates whether the function is expected to yield a real number.
    Fun(Slot, Box<(Symbol, Vec<Atom>, Vec<Slot>)>, bool),
    /// `Assign(o, v)` means `o = v`.
    Assign(Slot, Slot),
    /// `IfElse(cond, label)` means jump to `label` if `cond` is zero.
    IfElse(Slot, usize),
    /// Unconditional jump to `label`.
    Goto(usize),
    /// A position in the instruction list to jump to.
    Label(usize),
    /// `Join(o, cond, t, f)` means `o = cond ? t : f`.
    Join(Slot, Slot, Slot, Slot),
}

impl std::fmt::Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Instruction::Add(o, a, _) => {
                write!(
                    f,
                    "{} = {}",
                    o,
                    a.iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join("+")
                )
            }
            Instruction::Mul(o, a, _) => {
                write!(
                    f,
                    "{} = {}",
                    o,
                    a.iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join("*")
                )
            }
            Instruction::Pow(o, b, e, _) => {
                write!(f, "{o} = {b}^{e}")
            }
            Instruction::Powf(o, b, e, _) => {
                write!(f, "{o} = {b}^{e}")
            }
            Instruction::Fun(o, b, _) => {
                let (name, tags, args) = &**b;
                let mut values = tags
                    .iter()
                    .map(|x| x.to_canonical_string())
                    .collect::<Vec<_>>();
                values.extend(args.iter().map(|x| x.to_string()));
                write!(
                    f,
                    "{} = {}({})",
                    o,
                    name.get_ascii_name()
                        .unwrap_or_else(|| name.get_name().replace("::", "_")),
                    values.join(", ")
                )
            }
            Instruction::Assign(o, v) => {
                write!(f, "{} = {}", o, v)
            }
            Instruction::IfElse(cond, label) => {
                write!(f, "if {} == 0 goto L{}", cond, label)
            }
            Instruction::Goto(label) => {
                write!(f, "goto L{}", label)
            }
            Instruction::Label(label) => {
                write!(f, "L{}:", label)
            }
            Instruction::Join(o, cond, a, b) => {
                write!(f, "{} = {} ? {} : {}", o, cond, a, b)
            }
        }
    }
}

/// Instructions and data exported from an [`ExpressionEvaluator`].
///
/// This is the structured representation used when an evaluator is translated to another runtime.
/// Function symbols and tags retain their symbolic identity in the current Symbolica state.
/// Binary instruction decoding uses a [`crate::state::StateMap`]; serde includes the required
/// state in each tag's atom export and reports conflicting symbol definitions when importing.
/// Parameters are referenced by [`Slot::Param`], constants by [`Slot::Const`], temporaries by
/// [`Slot::Temp`], and outputs by [`Slot::Out`]. Calls to functions defined as non-inlined in a
/// [`FunctionMap`] can be resolved through [`ExportedInstructions::sub_evaluators`].
#[derive(Debug, Clone)]
pub struct ExportedInstructions<T> {
    /// The number of values expected in the parameter list.
    pub input_count: usize,
    /// The number of values produced in the output list.
    pub output_count: usize,
    /// The linear instruction stream.
    pub instructions: Vec<Instruction>,
    /// The number of temporary storage slots required to execute `instructions`.
    pub temporary_count: usize,
    /// Constant values referenced by [`Slot::Const`]. Slots listed in
    /// [`Self::constant_functions`] may contain rational placeholders in an exact
    /// evaluator; resolve their definitions at the target precision before use.
    pub constants: Vec<T>,
    /// Symbolic definitions of constant slots evaluated by registered functions.
    ///
    /// In exact evaluators these slots may contain placeholders, not
    /// the value of the registered constant. A consumer must resolve these
    /// definitions at its target precision before using the constant slots.
    pub constant_functions: Vec<ExportedConstantFunction>,
    /// Locally defined evaluators called by [`Instruction::Fun`] instructions in this stream.
    ///
    /// A call resolves to a sub-evaluator when its symbol and tags match the corresponding fields
    /// of an entry in this list. Calls without a matching entry remain ordinary external function
    /// calls. Sub-evaluators may recursively contain their own sub-evaluators.
    pub sub_evaluators: Vec<ExportedSubEvaluator<T>>,
}

/// An exact registered constant or fixed-argument function in a constant slot.
#[derive(Debug, Clone)]
pub struct ExportedConstantFunction {
    /// Index in [`ExportedInstructions::constants`], not in temporary storage.
    pub index: usize,
    /// Registered function or constant symbol.
    pub symbol: Symbol,
    /// Symbolic function tags, in their original order.
    pub tags: Vec<Atom>,
    /// Exact numerical arguments; empty for a registered constant such as pi.
    pub fixed_args: Vec<Complex<Rational>>,
}

/// A non-inlined function body exported alongside an instruction stream.
///
/// The `symbol` and `tags` fields form the call signature and match the corresponding values in an
/// [`Instruction::Fun`] instruction. Parameters are passed in the order of that instruction's
/// argument slots.
#[derive(Debug, Clone)]
pub struct ExportedSubEvaluator<T> {
    /// The function symbol used by the calling [`Instruction::Fun`].
    pub symbol: Symbol,
    /// The function tags used by the calling [`Instruction::Fun`].
    pub tags: Vec<Atom>,
    /// The number of parameter values expected by this evaluator.
    pub input_count: usize,
    /// The number of values produced by this evaluator.
    pub output_count: usize,
    /// The exported function body.
    pub instructions: Arc<ExportedInstructions<T>>,
}

impl<T: Default + Clone> ExpressionEvaluator<T> {
    /// Import scalar constants and shared function bodies, remapping their indices.
    /// The caller chooses how to store literals; registered constant placeholders
    /// always get distinct slots before any later literal can be interned.
    pub(super) fn import_context(
        &self,
        ins: &mut InstructionList<T>,
        external_functions: &mut Vec<ExternalFunctionContainer<T>>,
        mut import_scalar_constant: impl FnMut(&mut InstructionList<T>, T) -> Slot,
    ) -> (Vec<Slot>, Vec<usize>) {
        let unknown: HashSet<_> = self
            .external_fns
            .iter()
            .filter_map(|external| external.constant_index)
            .collect();
        let mut slots = Vec::new();
        let mut constants = Vec::new();
        for (index, value) in self.stack[self.param_count..self.reserved_indices]
            .iter()
            .enumerate()
        {
            let slot = if unknown.contains(&index) {
                let slot = ins.push_scalar_constant(value.clone());
                *ins.unknown_constants.last_mut().unwrap() = true;
                slot
            } else {
                import_scalar_constant(ins, value.clone())
            };
            let Slot::Const(index) = slot else {
                panic!("Imported scalar constants must use constant slots");
            };
            constants.push(index);
            slots.push(slot);
        }
        let mut functions = Vec::new();
        for external in &self.external_fns {
            let mut external = external.clone();
            if let Some(index) = external.constant_index.as_mut() {
                *index = constants[*index];
            }
            if let Some(body) = &mut external.body {
                for (instruction, _) in &mut Arc::make_mut(body).instructions {
                    if let Instr::ExternalFun(_, function, _) = instruction {
                        *function = functions[*function];
                    }
                }
            }
            let index = external_functions
                .iter()
                .position(|f| {
                    f == &external
                        && f.constant_index == external.constant_index
                        && f.body == external.body
                })
                .unwrap_or_else(|| {
                    external_functions.push(external);
                    external_functions.len() - 1
                });
            functions.push(index);
        }
        (slots, functions)
    }

    /// Inline this evaluator's scalar instructions into an instruction list.
    /// Each output updates the source-slot map after its inputs are read, so reused
    /// stack slots become distinct temporaries without modifying the source.
    /// Shared function bodies are imported without inlining, and the caller chooses
    /// how scalar literals are stored in the destination's vector-sized groups.
    pub(super) fn inline_instructions(
        &self,
        args: &[Slot],
        ins: &mut InstructionList<T>,
        external_functions: &mut Vec<ExternalFunctionContainer<T>>,
        next_label: &mut usize,
        import_scalar_constant: impl FnMut(&mut InstructionList<T>, T) -> Slot,
    ) -> Vec<Slot> {
        let (constants, functions) =
            self.import_context(ins, external_functions, import_scalar_constant);
        let mut slots = args.to_vec();
        slots.extend(constants);
        slots.resize(self.stack.len(), Slot::Temp(0));
        let mut labels = HashMap::default();
        for (instr, _) in &self.instructions {
            if let Instr::Label(label) = instr {
                labels.insert(*label, Label(*next_label));
                *next_label += 1;
            }
        }
        for (instr, _) in &self.instructions {
            let (output, instr) = match instr {
                Instr::Add(output, args) | Instr::Mul(output, args) => {
                    let mut value = slots[args[0]];
                    for arg in &args[1..] {
                        value = ins.add(if matches!(instr, Instr::Add(..)) {
                            VectorInstruction::Add(value, slots[*arg])
                        } else {
                            VectorInstruction::Mul(value, slots[*arg])
                        });
                    }
                    slots[*output] = value;
                    continue;
                }
                Instr::Pow(o, a, p) => (*o, VectorInstruction::Pow(slots[*a], *p)),
                Instr::Powf(o, a, b) => (*o, VectorInstruction::Powf(slots[*a], slots[*b])),
                Instr::BuiltinFun(o, f, a) => (*o, VectorInstruction::BuiltinFun(*f, slots[*a])),
                Instr::ExternalFun(o, f, args) => (
                    *o,
                    VectorInstruction::ExternalFun(
                        functions[*f],
                        args.iter().map(|a| slots[*a]).collect(),
                    ),
                ),
                Instr::Join(o, c, a, b) => {
                    (*o, VectorInstruction::Join(slots[*c], slots[*a], slots[*b]))
                }
                Instr::IfElse(c, l) => {
                    ins.add(VectorInstruction::IfElse(slots[*c], labels[l]));
                    continue;
                }
                Instr::Goto(l) => {
                    ins.add(VectorInstruction::Goto(labels[l]));
                    continue;
                }
                Instr::Label(l) => {
                    ins.add(VectorInstruction::Label(labels[l]));
                    continue;
                }
            };
            slots[output] = ins.add(instr);
        }
        self.result_indices.iter().map(|i| slots[*i]).collect()
    }

    /// Lower an instruction list to native evaluator storage.
    /// The caller must optimize the stack and fix labels before evaluation, and
    /// may run instruction optimizations before those final steps.
    pub(super) fn from_instruction_list(
        param_count: usize,
        instructions: InstructionList<T>,
        external_fns: Vec<ExternalFunctionContainer<T>>,
        outputs: Vec<Slot>,
        settings: OptimizationSettings,
    ) -> Self {
        let reserved_indices = param_count + instructions.constants.len();
        let index = |slot| match slot {
            Slot::Param(index) => index,
            Slot::Const(index) => param_count + index,
            Slot::Temp(index) => reserved_indices + index,
            Slot::Out(_) => unreachable!("instruction lists do not use output slots"),
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
        Self {
            stack,
            param_count,
            reserved_indices,
            instructions: lowered,
            result_indices: outputs.into_iter().map(index).collect(),
            external_fns,
            settings,
        }
    }
}

impl<T: Clone> ExpressionEvaluator<T> {
    /// Export the instruction stream, temporary storage size, constants, and sub-evaluators.
    ///
    /// The returned [`ExportedInstructions`] value contains all data needed to execute the evaluator
    /// outside Symbolica:
    /// - [`ExportedInstructions::input_count`] is the number of input parameters.
    /// - [`ExportedInstructions::output_count`] is the number of output values.
    /// - [`ExportedInstructions::instructions`] is the linear instruction list.
    /// - [`ExportedInstructions::temporary_count`] is the number of temporary slots required.
    /// - [`ExportedInstructions::constants`] contains the values addressed by [`Slot::Const`].
    /// - [`ExportedInstructions::constant_functions`] identifies registered constants and
    ///   fixed-argument functions whose slots must be resolved at the target precision.
    /// - [`ExportedInstructions::sub_evaluators`] contains bodies for non-inlined functions. A
    ///   function instruction without a matching sub-evaluator is an ordinary external call.
    ///
    /// This function can be used to create an evaluator in a different language.
    pub fn export_instructions(&self) -> ExportedInstructions<T> {
        // Callees precede callers, so each body can reuse already-exported bodies.
        let mut bodies = Vec::with_capacity(self.external_fns.len());
        for external in &self.external_fns {
            let body = external.body.as_ref().map(|body| {
                let instructions = Arc::new(
                    body.program()
                        .export_instructions(&self.external_fns, &bodies),
                );
                ExportedSubEvaluator {
                    symbol: external.symbol,
                    tags: external.tags.clone(),
                    input_count: instructions.input_count,
                    output_count: instructions.output_count,
                    instructions,
                }
            });
            bodies.push(body);
        }
        self.export_instructions_impl(&self.external_fns, &bodies)
    }

    pub(super) fn export_instructions_impl(
        &self,
        external_fns: &[ExternalFunctionContainer<T>],
        bodies: &[Option<ExportedSubEvaluator<T>>],
    ) -> ExportedInstructions<T> {
        let mut exported = self.program().export_instructions(external_fns, bodies);
        exported.constant_functions = self
            .external_fns
            .iter()
            .filter_map(|external| {
                external
                    .constant_index
                    .map(|index| ExportedConstantFunction {
                        index,
                        symbol: external.symbol,
                        tags: external.tags.clone(),
                        fixed_args: external.fixed_args.clone(),
                    })
            })
            .collect();
        exported
    }
}

impl<T: Clone> ProgramView<'_, T> {
    pub(super) fn export_instructions(
        &self,
        external_fns: &[ExternalFunctionContainer<T>],
        bodies: &[Option<ExportedSubEvaluator<T>>],
    ) -> ExportedInstructions<T> {
        let mut instr = vec![];
        let constants: Vec<_> = self.constants.to_vec();
        let mut calls = self
            .instructions
            .iter()
            .filter_map(|(instruction, _)| {
                if !bodies.is_empty()
                    && let Instr::ExternalFun(_, index, _) = instruction
                {
                    Some(*index)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        calls.sort_unstable();
        calls.dedup();
        let sub_evaluators = calls
            .into_iter()
            .filter_map(|key| bodies[key].clone())
            .collect();

        macro_rules! get_slot {
            ($i:expr) => {
                if $i < self.param_count {
                    Slot::Param($i)
                } else if $i < self.reserved_indices {
                    Slot::Const($i - self.param_count)
                } else {
                    if self.result_indices.contains(&$i) {
                        Slot::Out(self.result_indices.iter().position(|x| *x == $i).unwrap())
                    } else {
                        Slot::Temp($i - self.reserved_indices)
                    }
                }
            };
        }

        for (i, sc) in self.instructions {
            match i {
                Instr::Add(o, a) => {
                    let n_real_args = match sc {
                        ComplexPhase::Real => a.len(),
                        ComplexPhase::PartialReal(n) => *n,
                        _ => 0,
                    };

                    instr.push(Instruction::Add(
                        get_slot!(*o),
                        a.iter().map(|x| get_slot!(*x)).collect(),
                        n_real_args,
                    ));
                }
                Instr::Mul(o, a) => {
                    let n_real_args = match sc {
                        ComplexPhase::Real => a.len(),
                        ComplexPhase::PartialReal(n) => *n,
                        _ => 0,
                    };

                    instr.push(Instruction::Mul(
                        get_slot!(*o),
                        a.iter().map(|x| get_slot!(*x)).collect(),
                        n_real_args,
                    ));
                }
                Instr::Pow(o, b, e) => {
                    instr.push(Instruction::Pow(
                        get_slot!(*o),
                        get_slot!(*b),
                        *e,
                        *sc == ComplexPhase::Real,
                    ));
                }
                Instr::Powf(o, b, e) => {
                    instr.push(Instruction::Powf(
                        get_slot!(*o),
                        get_slot!(*b),
                        get_slot!(*e),
                        *sc == ComplexPhase::Real,
                    ));
                }
                Instr::BuiltinFun(o, s, a) => {
                    instr.push(Instruction::Fun(
                        get_slot!(*o),
                        Box::new((*s, vec![], vec![get_slot!(*a)])),
                        *sc == ComplexPhase::Real,
                    ));
                }
                Instr::ExternalFun(o, f, a) => {
                    instr.push(Instruction::Fun(
                        get_slot!(*o),
                        Box::new((
                            external_fns[*f].symbol,
                            external_fns[*f].tags.clone(),
                            a.iter().map(|x| get_slot!(*x)).collect(),
                        )),
                        *sc == ComplexPhase::Real,
                    ));
                }
                Instr::IfElse(cond, label) => {
                    instr.push(Instruction::IfElse(get_slot!(*cond), label.0));
                }
                Instr::Goto(label) => {
                    instr.push(Instruction::Goto(label.0));
                }
                Instr::Label(label) => {
                    instr.push(Instruction::Label(label.0));
                }
                Instr::Join(o, cond, a, b) => {
                    instr.push(Instruction::Join(
                        get_slot!(*o),
                        get_slot!(*cond),
                        get_slot!(*a),
                        get_slot!(*b),
                    ));
                }
            }
        }

        for (out, i) in self.result_indices.iter().enumerate() {
            if get_slot!(*i) != Slot::Out(out) {
                instr.push(Instruction::Assign(Slot::Out(out), get_slot!(*i)));
            }
        }

        ExportedInstructions {
            input_count: self.param_count,
            output_count: self.result_indices.len(),
            instructions: instr,
            temporary_count: self.stack_size - self.reserved_indices,
            constants,
            constant_functions: Vec::new(),
            sub_evaluators,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// An operation on vector-valued slots. Its position in an instruction list
/// identifies the temporary slot that receives its result.
pub enum VectorInstruction {
    /// Add the two input slots componentwise.
    Add(Slot, Slot),
    /// Copy the input slot.
    Assign(Slot),
    /// Multiply the two input slots componentwise.
    Mul(Slot, Slot),
    /// Raise the input to an integer power.
    Pow(Slot, i64),
    /// Raise the first input to the power given by the second input.
    Powf(Slot, Slot),
    /// Apply the named built-in function to the input.
    BuiltinFun(Symbol, Slot),
    /// Call an external function by its index with the given argument slots.
    ExternalFun(usize, Vec<Slot>),
    /// Jump to the label when the condition is zero.
    IfElse(Slot, Label),
    /// Jump unconditionally to the label.
    Goto(Label),
    /// Mark a jump destination.
    Label(Label),
    /// Select the second or third slot according to whether the first is nonzero.
    Join(Slot, Slot, Slot),
}

/// A sequence of vector operations with shared constant storage.
/// Each constant occupies a fixed number of components determined by the list.
pub struct InstructionList<T> {
    pub(super) instructions: Vec<VectorInstruction>,
    pub(super) constants: Vec<T>,
    pub(super) unknown_constants: Vec<bool>,
    pub(super) dim: usize,
}

impl<T> InstructionList<T> {
    /// Append an instruction and return the temporary slot containing its result.
    pub fn add(&mut self, instr: VectorInstruction) -> Slot {
        self.instructions.push(instr);
        Slot::Temp(self.instructions.len() - 1)
    }
}

impl<T: Default> InstructionList<T> {
    /// Append a scalar in its own vector-sized group without interning it.
    pub(super) fn push_scalar_constant(&mut self, value: T) -> Slot {
        let slot = Slot::Const(self.constants.len());
        self.constants.push(value);
        self.constants.extend((1..self.dim).map(|_| T::default()));
        self.unknown_constants.push(false);
        slot
    }
}

impl<T: PartialEq + Clone + std::fmt::Debug> InstructionList<T> {
    /// Store a constant vector, reusing an equal existing constant when possible.
    ///
    /// # Panics
    ///
    /// Panics if the vector length differs from the list's component count.
    pub fn add_constant(&mut self, value: Vec<T>) -> Slot {
        assert_eq!(value.len(), self.dim);
        if let Some(c) = self
            .constants
            .chunks(self.dim)
            .zip(&self.unknown_constants)
            .position(|(x, u)| x == value && !u)
        {
            Slot::Const(c * self.dim)
        } else {
            self.constants.extend(value);
            self.unknown_constants.push(false);
            Slot::Const(self.constants.len() - self.dim)
        }
    }

    /// Store a vector with `value` in every component, reusing existing storage
    /// when an equal constant is already present.
    pub fn add_repeated_constant(&mut self, value: T) -> Slot {
        if let Some(c) = self
            .constants
            .chunks(self.dim)
            .zip(&self.unknown_constants)
            .position(|(x, u)| x.iter().all(|x| *x == value) && !u)
        {
            Slot::Const(c * self.dim)
        } else {
            for _ in 0..self.dim {
                self.constants.push(value.clone());
            }
            self.unknown_constants.push(false);
            Slot::Const(self.constants.len() - self.dim)
        }
    }
}

impl<T: SingleFloat> InstructionList<T> {
    /// Return whether `slot` refers to a known constant component equal to zero.
    pub fn is_zero(&self, slot: &Slot) -> bool {
        match slot {
            Slot::Const(c) => {
                self.constants[*c].is_zero() && !self.unknown_constants[*c / self.dim]
            }
            _ => false,
        }
    }

    /// Return whether `slot` refers to a known constant component equal to one.
    pub fn is_one(&self, slot: &Slot) -> bool {
        match slot {
            Slot::Const(c) => self.constants[*c].is_one() && !self.unknown_constants[*c / self.dim],
            _ => false,
        }
    }

    /// Store a constant vector with `value` in its first component and zeros
    /// in all remaining components.
    pub fn add_constant_in_first_component(&mut self, value: T) -> Slot {
        let mut v = vec![value.clone()];
        v.extend((1..self.dim).map(|_| value.zero()));
        self.add_constant(v)
    }
}

/// A label in the instruction list.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Label(pub(super) usize);

/// An evaluation instruction.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[derive(Debug, Clone, PartialEq)]
pub(super) enum Instr {
    Add(usize, Vec<usize>),
    Mul(usize, Vec<usize>),
    Pow(usize, usize, i64),
    Powf(usize, usize, usize),
    BuiltinFun(usize, Symbol, usize),
    ExternalFun(usize, usize, Vec<usize>),
    IfElse(usize, Label),
    Goto(Label),
    Label(Label),
    Join(usize, usize, usize, usize),
}

/// The phase of an operation in a complex evaluator.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[derive(Debug, Copy, Clone, PartialEq, Default, Hash)]
pub enum ComplexPhase {
    /// The result is real.
    Real,
    /// The result is purely imaginary.
    Imag,
    /// The first `usize` inputs are real; remaining inputs may be complex.
    PartialReal(usize),
    #[default]
    /// No real or imaginary restriction is known.
    Any,
}

#[cfg(all(test, feature = "bincode"))]
mod tag_serialization_tests {
    use super::*;
    use crate::state::StateMap;

    fn check_remapping<C: bincode::config::Config>(config: C) {
        let source_x = crate::symbol!("symbolica::instruction_tags::source::x");
        let source_y = crate::symbol!("symbolica::instruction_tags::source::y");
        let source_function = crate::symbol!("symbolica::instruction_tags::source::f");
        // Reverse the variable creation order to exercise normalization after remapping.
        let target_y = crate::symbol!("symbolica::instruction_tags::target::y");
        let target_x = crate::symbol!("symbolica::instruction_tags::target::x");
        let target_function = crate::symbol!("symbolica::instruction_tags::target::f");
        let function = crate::symbol!("symbolica::instruction_tags::evaluate");
        let source_tags = vec![
            crate::parse!("2/3"),
            source_x.to_atom(),
            source_function.call((source_x.to_atom() + source_y,)),
        ];
        let expected_tags = vec![
            crate::parse!("2/3"),
            target_x.to_atom(),
            target_function.call((target_x.to_atom() + target_y,)),
        ];
        assert_ne!(source_tags, expected_tags);

        let state_map = || {
            let mut map = StateMap::default();
            map.symbols.insert(source_x.get_id(), target_x);
            map.symbols.insert(source_y.get_id(), target_y);
            map.symbols
                .insert(source_function.get_id(), target_function);
            map
        };
        let instruction = Instruction::Fun(
            Slot::Out(0),
            Box::new((function, source_tags.clone(), vec![Slot::Param(0)])),
            true,
        );
        let tree = |tags: Vec<Atom>| -> Expression<i64> {
            Expression::Fun(
                0,
                function,
                tags.clone(),
                vec![Expression::Fun(
                    0,
                    function,
                    tags,
                    vec![Expression::Const(0, Box::new(3))],
                )],
            )
        };
        let bytes = bincode::encode_to_vec((instruction, tree(source_tags)), config).unwrap();
        let expected_tree = tree(expected_tags.clone());
        let check =
            |((instruction, expression), consumed): ((Instruction, Expression<i64>), usize)| {
                assert_eq!(consumed, bytes.len());
                let Instruction::Fun(output, call, is_real) = instruction else {
                    panic!("Expected a function instruction");
                };
                assert_eq!(output, Slot::Out(0));
                assert_eq!(call.0, function);
                assert_eq!(call.1, expected_tags);
                assert_eq!(call.2, [Slot::Param(0)]);
                assert!(is_real);
                assert_eq!(expression, expected_tree);
            };
        check(bincode::decode_from_slice_with_context(&bytes, config, state_map()).unwrap());
        check(bincode::borrow_decode_from_slice_with_context(&bytes, config, state_map()).unwrap());
    }

    #[test]
    fn binary_instruction_and_expression_tags_follow_the_state_map() {
        check_remapping(bincode::config::standard());
        check_remapping(
            bincode::config::standard()
                .with_fixed_int_encoding()
                .with_big_endian(),
        );
    }
}
