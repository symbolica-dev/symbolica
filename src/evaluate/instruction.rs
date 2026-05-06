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

impl<T: Clone> ExpressionEvaluator<T> {
    /// Export the instructions, the size of the temporary storage, and the list of constants.
    /// This function can be used to create an evaluator in a different language.
    pub fn export_instructions(&self) -> (Vec<Instruction>, usize, Vec<T>) {
        let mut instr = vec![];
        let constants: Vec<_> = self.stack[self.param_count..self.reserved_indices].to_vec();

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

        (instr, self.stack.len() - self.reserved_indices, constants)
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
        if let Some(c) = self.constants.chunks(self.dim).position(|x| x == &value) {
            Slot::Const(c * self.dim)
        } else {
            self.constants.extend(value);
            Slot::Const(self.constants.len() - self.dim)
        }
    }

    pub fn add_repeated_constant(&mut self, value: T) -> Slot {
        if let Some(c) = self
            .constants
            .chunks(self.dim)
            .position(|x| x.iter().all(|x| *x == value))
        {
            Slot::Const(c * self.dim)
        } else {
            for _ in 0..self.dim {
                self.constants.push(value.clone());
            }
            Slot::Const(self.constants.len() - self.dim)
        }
    }
}

impl<T: SingleFloat> InstructionList<T> {
    pub fn is_zero(&self, slot: &Slot) -> bool {
        match slot {
            Slot::Const(c) => self.constants[*c].is_zero(),
            _ => false,
        }
    }

    pub fn is_one(&self, slot: &Slot) -> bool {
        match slot {
            Slot::Const(c) => self.constants[*c].is_one(),
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
