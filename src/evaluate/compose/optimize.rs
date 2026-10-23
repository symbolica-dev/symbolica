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
}

impl<T: Default + Clone> EvaluatorComposer<T> {
    /// Slice data dependencies and the structured branches guarding live values.
    /// Keeping a live branch's labels preserves lazy evaluation and the
    /// branch structure expected by native CSE/CPE. Completely dead branches vanish.
    pub(super) fn prune_unused(&mut self, outputs: &mut [Slot]) {
        let instructions = &self.instructions.instructions;
        let mut labels = HashMap::default();
        for (i, instruction) in instructions.iter().enumerate() {
            if let VectorInstruction::Label(label) = instruction {
                labels.insert(*label, i);
            }
        }
        let mut branches = Vec::new();
        let mut starts = HashMap::default();
        for (i, instruction) in instructions.iter().enumerate() {
            if let VectorInstruction::IfElse(condition, otherwise) = instruction {
                let otherwise = labels[otherwise];
                let VectorInstruction::Goto(end) = instructions[otherwise - 1] else {
                    unreachable!("Native evaluators use structured if/else branches");
                };
                let end = labels[&end];
                let controls = [i, otherwise - 1, otherwise, end];
                // A vectorized branch can produce several consecutive joins.
                let mut last = end;
                while let Some(VectorInstruction::Join(c, _, _)) = instructions.get(last + 1) {
                    if c != condition {
                        break;
                    }
                    last += 1;
                }
                starts.insert(i, branches.len());
                branches.push((last, controls));
            }
        }
        let mut owners = vec![None; instructions.len()];
        let mut active: Vec<usize> = Vec::new();
        let mut parents = vec![None; branches.len()];
        for (i, owner) in owners.iter_mut().enumerate() {
            while active.last().is_some_and(|b| branches[*b].0 < i) {
                active.pop();
            }
            if let Some(branch) = starts.get(&i) {
                parents[*branch] = active.last().copied();
                active.push(*branch);
            }
            *owner = active.last().copied();
        }
        let mut live_branches = vec![false; branches.len()];
        let mut used = vec![false; instructions.len()];
        let mut constants = vec![false; self.instructions.constants.len()];
        let mut functions = vec![false; self.external_functions.len()];
        let mut pending = outputs.to_vec();
        while let Some(slot) = pending.pop() {
            match slot {
                Slot::Const(i) => constants[i] = true,
                Slot::Temp(i) if !used[i] => {
                    used[i] = true;
                    let mut branch = owners[i];
                    while let Some(b) = branch {
                        if live_branches[b] {
                            break;
                        }
                        live_branches[b] = true;
                        pending.extend(branches[b].1.iter().map(|i| Slot::Temp(*i)));
                        branch = parents[b];
                    }
                    let instruction = &mut self.instructions.instructions[i];
                    instruction.map_inputs(|slot| {
                        pending.push(slot);
                        slot
                    });
                    if let VectorInstruction::ExternalFun(index, _) = instruction {
                        functions[*index] = true;
                    }
                }
                _ => {}
            }
        }
        // Callees precede callers. Retain transitive body dependencies, as well as
        // callbacks supplying live constant slots, before compacting the table.
        for (index, function) in self.external_functions.iter().enumerate().rev() {
            if let Some(constant) = function.constant_index {
                functions[index] |= constants[constant];
            }
            if functions[index]
                && let Some(body) = &function.body
            {
                for (instruction, _) in &body.instructions {
                    if let Instr::ExternalFun(_, callee, _) = instruction {
                        functions[*callee] = true;
                    }
                }
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
                if let Some(body) = &mut function.body {
                    for (instruction, _) in &mut Arc::make_mut(body).instructions {
                        if let Instr::ExternalFun(_, callee, _) = instruction {
                            *callee = function_map[*callee];
                        }
                    }
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
    /// Intern live constants before optimization and remap the selected outputs.
    pub(super) fn deduplicate_constants(&mut self, outputs: &mut [Slot]) {
        #[derive(PartialEq, Eq, Hash)]
        enum Constant<'a, T> {
            Literal(&'a T),
            Function(Symbol, &'a [Atom], &'a [Complex<Rational>]),
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
                Constant::Function(function.symbol, &function.tags, &function.fixed_args)
            } else {
                Constant::Literal(value)
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
            if let Some(body) = &mut function.body {
                for (instruction, _) in &mut Arc::make_mut(body).instructions {
                    if let Instr::ExternalFun(_, callee, _) = instruction {
                        *callee = function_map[*callee];
                    }
                }
            }
            let next = functions
                .iter()
                .position(|existing: &ExternalFunctionContainer<T>| {
                    existing == &function
                        && existing.constant_index == function.constant_index
                        && existing.body == function.body
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

impl<T: Default + Clone> ExpressionEvaluator<T> {
    /// Restore native n-ary operations after scalar inlining. Only single-use
    /// intermediates in the same basic block are absorbed, so shared work stays shared
    /// and branch-local computations are never moved into an unconditional path.
    pub(super) fn flatten_composition_operations(&mut self) {
        let mut uses = vec![0; self.reserved_indices + self.instructions.len()];
        let mut blocks = vec![0; self.instructions.len()];
        let mut block = 0;
        for (i, (instruction, _)) in self.instructions.iter().enumerate() {
            match instruction {
                Instr::Add(_, args) | Instr::Mul(_, args) | Instr::ExternalFun(_, _, args) => {
                    for arg in args {
                        uses[*arg] += 1;
                    }
                }
                Instr::Pow(_, a, _) | Instr::BuiltinFun(_, _, a) => uses[*a] += 1,
                Instr::Powf(_, a, b) => {
                    uses[*a] += 1;
                    uses[*b] += 1;
                }
                Instr::IfElse(c, _) => {
                    uses[*c] += 1;
                    block += 1;
                }
                Instr::Join(_, c, a, b) => {
                    uses[*c] += 1;
                    uses[*a] += 1;
                    uses[*b] += 1;
                    block += 1;
                }
                Instr::Goto(_) | Instr::Label(_) => block += 1,
            }
            blocks[i] = block;
        }
        for output in &self.result_indices {
            uses[*output] += 1;
        }
        let mut absorbed = vec![false; self.instructions.len()];
        for (i, (instruction, _)) in self.instructions.iter().enumerate() {
            let (Instr::Add(_, args) | Instr::Mul(_, args)) = instruction else {
                continue;
            };
            for arg in args {
                if *arg < self.reserved_indices || uses[*arg] != 1 {
                    continue;
                }
                let child = *arg - self.reserved_indices;
                if blocks[child] == blocks[i]
                    && matches!(
                        (instruction, &self.instructions[child].0),
                        (Instr::Add(..), Instr::Add(..)) | (Instr::Mul(..), Instr::Mul(..))
                    )
                {
                    absorbed[child] = true;
                }
            }
        }
        let mut rename: Vec<_> = (0..self.reserved_indices).collect();
        let mut kept = 0;
        for absorb in &absorbed {
            rename.push(self.reserved_indices + kept);
            if !absorb {
                kept += 1;
            }
        }
        let mut instructions = Vec::with_capacity(kept);
        for (i, (instruction, phase)) in self.instructions.iter().enumerate() {
            if absorbed[i] {
                continue;
            }
            let out = self.reserved_indices + instructions.len();
            let instruction = match instruction {
                Instr::Add(_, args) | Instr::Mul(_, args) => {
                    let mut pending = args.clone();
                    let mut flattened = Vec::new();
                    while let Some(arg) = pending.pop() {
                        if arg >= self.reserved_indices && absorbed[arg - self.reserved_indices] {
                            match &self.instructions[arg - self.reserved_indices].0 {
                                Instr::Add(_, args) | Instr::Mul(_, args) => pending.extend(args),
                                _ => unreachable!(),
                            }
                        } else {
                            flattened.push(rename[arg]);
                        }
                    }
                    flattened.sort_unstable();
                    if matches!(instruction, Instr::Add(..)) {
                        Instr::Add(out, flattened)
                    } else {
                        Instr::Mul(out, flattened)
                    }
                }
                Instr::Pow(_, a, p) => Instr::Pow(out, rename[*a], *p),
                Instr::Powf(_, a, b) => Instr::Powf(out, rename[*a], rename[*b]),
                Instr::BuiltinFun(_, f, a) => Instr::BuiltinFun(out, *f, rename[*a]),
                Instr::ExternalFun(_, f, args) => {
                    Instr::ExternalFun(out, *f, args.iter().map(|a| rename[*a]).collect())
                }
                Instr::Join(_, c, a, b) => Instr::Join(out, rename[*c], rename[*a], rename[*b]),
                Instr::IfElse(c, label) => Instr::IfElse(rename[*c], *label),
                Instr::Goto(label) => Instr::Goto(*label),
                Instr::Label(label) => Instr::Label(*label),
            };
            instructions.push((instruction, *phase));
        }
        for output in &mut self.result_indices {
            *output = rename[*output];
        }
        self.instructions = instructions;
    }
}
