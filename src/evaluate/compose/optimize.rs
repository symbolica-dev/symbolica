use super::*;

impl VectorInstruction {
    fn map_inputs(&mut self, mut f: impl FnMut(Slot) -> Slot) {
        match self {
            Self::Assign(a) | Self::Pow(a, _) | Self::BuiltinFun(_, a) | Self::IfElse(a, _) => {
                *a = f(*a)
            }
            Self::Add(a, b) | Self::Mul(a, b) | Self::Powf(a, b) => {
                *a = f(*a);
                *b = f(*b);
            }
            Self::ExternalFun(_, args) => args.iter_mut().for_each(|a| *a = f(*a)),
            Self::Join(c, a, b) => {
                *c = f(*c);
                *a = f(*a);
                *b = f(*b);
            }
            Self::Goto(_) | Self::Label(_) => {}
        }
    }
    fn has_control_flow(&self) -> bool {
        matches!(
            self,
            Self::IfElse(..) | Self::Goto(..) | Self::Label(..) | Self::Join(..)
        )
    }
}

impl<T: Default + Clone> EvaluatorComposer<T> {
    /// A backwards dependency slice of straight-line, single-assignment storage.
    /// Branches deliberately retain the complete program; this is not a CFG pass.
    pub(super) fn prune_unused(&mut self, outputs: &mut [Slot]) {
        if self
            .instructions
            .instructions
            .iter()
            .any(VectorInstruction::has_control_flow)
        {
            return;
        }
        let mut used = vec![false; self.instructions.instructions.len()];
        let mut constants = vec![false; self.instructions.constants.len()];
        let mut functions = vec![false; self.external_functions.len()];
        let mark = |slot: Slot, used: &mut [bool], constants: &mut [bool]| match slot {
            Slot::Temp(i) => used[i] = true,
            Slot::Const(i) => constants[i] = true,
            _ => {}
        };
        for slot in outputs.iter() {
            mark(*slot, &mut used, &mut constants);
        }
        for (i, instruction) in self.instructions.instructions.iter_mut().enumerate().rev() {
            if !used[i] {
                continue;
            }
            instruction.map_inputs(|slot| {
                mark(slot, &mut used, &mut constants);
                slot
            });
            if let VectorInstruction::ExternalFun(index, _) = instruction {
                functions[*index] = true;
            }
        }
        for (index, function) in self.external_functions.iter().enumerate() {
            if let Some(constant) = function.constant_index {
                functions[index] |= constants[constant];
            }
        }
        let mut constant_map = vec![0; constants.len()];
        let mut new_constants = Vec::new();
        let mut new_unknown = Vec::new();
        for (old, (value, unknown)) in std::mem::take(&mut self.instructions.constants)
            .into_iter()
            .zip(std::mem::take(&mut self.instructions.unknown_constants))
            .enumerate()
        {
            if constants[old] {
                constant_map[old] = new_constants.len();
                new_constants.push(value);
                new_unknown.push(unknown);
            }
        }
        self.instructions.constants = new_constants;
        self.instructions.unknown_constants = new_unknown;
        let mut function_map = vec![0; functions.len()];
        let mut new_functions = Vec::new();
        for (old, mut function) in std::mem::take(&mut self.external_functions)
            .into_iter()
            .enumerate()
        {
            if functions[old] {
                function_map[old] = new_functions.len();
                if let Some(index) = &mut function.constant_index {
                    *index = constant_map[*index];
                }
                new_functions.push(function);
            }
        }
        self.external_functions = new_functions;
        let mut temporary_map = vec![0; used.len()];
        let mut new_instructions = Vec::new();
        let remap = |slot: Slot, temporary_map: &[usize]| match slot {
            Slot::Temp(i) => Slot::Temp(temporary_map[i]),
            Slot::Const(i) => Slot::Const(constant_map[i]),
            _ => slot,
        };
        for (old, mut instruction) in std::mem::take(&mut self.instructions.instructions)
            .into_iter()
            .enumerate()
        {
            if !used[old] {
                continue;
            }
            temporary_map[old] = new_instructions.len();
            instruction.map_inputs(|slot| remap(slot, &temporary_map));
            if let VectorInstruction::ExternalFun(index, _) = &mut instruction {
                *index = function_map[*index];
            }
            new_instructions.push(instruction);
        }
        for slot in outputs {
            *slot = remap(*slot, &temporary_map);
        }
        self.instructions.instructions = new_instructions;
    }
}

impl<T: Default + Clone + Eq + Hash> EvaluatorComposer<T> {
    /// Use the same literal-vs-callback distinction as native evaluator merging.
    pub(super) fn deduplicate_constants(&mut self, outputs: &mut [Slot]) {
        #[derive(PartialEq, Eq, Hash)]
        enum Constant<T> {
            Literal(T),
            Function(Symbol, Vec<Atom>, Vec<Complex<Rational>>),
        }
        let mut callbacks = HashMap::default();
        for function in &self.external_functions {
            if let Some(index) = function.constant_index {
                callbacks.insert(index, function);
            }
        }
        let mut known = HashMap::default();
        let mut remap = Vec::new();
        let mut values = Vec::new();
        let mut unknown = Vec::new();
        for (old, value) in self.instructions.constants.iter().enumerate() {
            let key = if let Some(function) = callbacks.get(&old) {
                Constant::Function(
                    function.symbol,
                    function.tags.clone(),
                    function.fixed_args.clone(),
                )
            } else {
                Constant::Literal(value.clone())
            };
            let next = *known.entry(key).or_insert_with(|| {
                let index = values.len();
                values.push(value.clone());
                unknown.push(self.instructions.unknown_constants[old]);
                index
            });
            remap.push(next);
        }
        self.instructions.constants = values;
        self.instructions.unknown_constants = unknown;
        let mut functions = Vec::new();
        let mut function_map = Vec::new();
        for mut function in std::mem::take(&mut self.external_functions) {
            if let Some(index) = &mut function.constant_index {
                *index = remap[*index];
            }
            let next = functions
                .iter()
                .position(|existing: &ExternalFunctionContainer<T>| {
                    existing == &function && existing.constant_index == function.constant_index
                })
                .unwrap_or_else(|| {
                    functions.push(function);
                    functions.len() - 1
                });
            function_map.push(next);
        }
        self.external_functions = functions;
        let slot = |slot| match slot {
            Slot::Const(i) => Slot::Const(remap[i]),
            _ => slot,
        };
        for instruction in &mut self.instructions.instructions {
            instruction.map_inputs(slot);
            if let VectorInstruction::ExternalFun(index, _) = instruction {
                *index = function_map[*index];
            }
        }
        outputs
            .iter_mut()
            .for_each(|output| *output = slot(*output));
    }
}
