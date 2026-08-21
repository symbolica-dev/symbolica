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
    pub fn index(&self, index: usize) -> Slot {
        match self {
            Slot::Param(i) => Slot::Param(*i + index),
            Slot::Const(i) => Slot::Const(*i + index),
            Slot::Temp(i) => Slot::Temp(*i + index),
            Slot::Out(i) => Slot::Out(*i + index),
        }
    }
}

/// An evaluation instruction.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
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
    Fun(Slot, Box<(Symbol, Vec<String>, Vec<Slot>)>, bool),
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
                let mut values = tags.iter().map(|x| x.to_string()).collect::<Vec<_>>();
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
/// This is the portable representation used when an evaluator is translated to another runtime.
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
    /// Constant values referenced by [`Slot::Const`].
    pub constants: Vec<T>,
    /// Locally defined evaluators called by [`Instruction::Fun`] instructions in this stream.
    ///
    /// A call resolves to a sub-evaluator when its symbol and tags match the corresponding fields
    /// of an entry in this list. Calls without a matching entry remain ordinary external function
    /// calls. Sub-evaluators may recursively contain their own sub-evaluators.
    pub sub_evaluators: Vec<ExportedSubEvaluator<T>>,
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
    pub tags: Vec<String>,
    /// The number of parameter values expected by this evaluator.
    pub input_count: usize,
    /// The number of values produced by this evaluator.
    pub output_count: usize,
    /// The recursively exported function body.
    pub instructions: ExportedInstructions<T>,
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
    /// - [`ExportedInstructions::sub_evaluators`] contains bodies for non-inlined functions. A
    ///   function instruction without a matching sub-evaluator is an ordinary external call.
    ///
    /// This function can be used to create an evaluator in a different language.
    pub fn export_instructions(&self) -> ExportedInstructions<T> {
        let mut instr = vec![];
        let constants: Vec<_> = self.stack[self.param_count..self.reserved_indices].to_vec();
        let sub_evaluators = self
            .external_fns
            .iter()
            .filter_map(|external| {
                external.sub_evaluator.as_ref().map(|evaluator| {
                    let instructions = evaluator.export_instructions();
                    ExportedSubEvaluator {
                        symbol: external.symbol,
                        tags: external
                            .tags
                            .iter()
                            .map(|tag| tag.to_canonical_string())
                            .collect(),
                        input_count: instructions.input_count,
                        output_count: instructions.output_count,
                        instructions,
                    }
                })
            })
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

        for (i, sc) in &self.instructions {
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
                            self.external_fns[*f].symbol,
                            self.external_fns[*f]
                                .tags
                                .iter()
                                .map(|x| x.to_canonical_string())
                                .collect(),
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
            temporary_count: self.stack.len() - self.reserved_indices,
            constants,
            sub_evaluators,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum VectorInstruction {
    Add(Slot, Slot),
    Assign(Slot),
    Mul(Slot, Slot),
    Pow(Slot, i64),
    Powf(Slot, Slot),
    BuiltinFun(Symbol, Slot),
    ExternalFun(usize, Vec<Slot>),
    IfElse(Slot, Label),
    Goto(Label),
    Label(Label),
    Join(Slot, Slot, Slot),
}

pub struct InstructionList<T> {
    pub(super) instructions: Vec<VectorInstruction>,
    pub(super) constants: Vec<T>,
    pub(super) unknown_constants: Vec<bool>,
    pub(super) dim: usize,
}

impl<T> InstructionList<T> {
    pub fn add(&mut self, instr: VectorInstruction) -> Slot {
        self.instructions.push(instr);
        Slot::Temp(self.instructions.len() - 1)
    }
}

impl<T: PartialEq + Clone + std::fmt::Debug> InstructionList<T> {
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
    pub fn is_zero(&self, slot: &Slot) -> bool {
        match slot {
            Slot::Const(c) => {
                self.constants[*c].is_zero() && !self.unknown_constants[*c / self.dim]
            }
            _ => false,
        }
    }

    pub fn is_one(&self, slot: &Slot) -> bool {
        match slot {
            Slot::Const(c) => self.constants[*c].is_one() && !self.unknown_constants[*c / self.dim],
            _ => false,
        }
    }

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
    Real,
    Imag,
    PartialReal(usize),
    #[default]
    Any,
}
