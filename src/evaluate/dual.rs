use super::*;

impl<T: Default + Clone> ExpressionEvaluator<T> {
    /// Inline generated scalar instructions without dualizing them again.
    fn inline_vector_components(
        &self,
        args: &[Slot],
        ins: &mut InstructionList<T>,
        external_functions: &mut Vec<ExternalFunctionContainer<T>>,
        next_label: &mut usize,
    ) -> Vec<Slot> {
        let mut slots = args.to_vec();
        let mut constants = Vec::new();
        for value in &self.stack[self.param_count..self.reserved_indices] {
            constants.push(ins.constants.len());
            slots.push(Slot::Const(ins.constants.len()));
            // InstructionList tracks constants in vector-sized groups.
            ins.constants.push(value.clone());
            ins.constants.extend((1..ins.dim).map(|_| T::default()));
            ins.unknown_constants.push(false);
        }
        slots.resize(self.stack.len(), Slot::Temp(0));
        let mut functions = Vec::new();
        for external in &self.external_fns {
            let mut external = external.clone();
            if let Some(index) = external.constant_index.as_mut() {
                *index = constants[*index];
                ins.unknown_constants[*index / ins.dim] = true;
            }
            let index = external_functions
                .iter()
                .position(|f| f == &external && f.constant_index == external.constant_index)
                .unwrap_or_else(|| {
                    external_functions.push(external);
                    external_functions.len() - 1
                });
            functions.push(index);
        }
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

    /// Redefine every operation to take `n` components in and
    /// yield `n` components. This can be used to define efficient
    /// evaluation over dual numbers.
    ///
    /// Non built-in functions can be rewritten to functions with the suffix `_v`
    /// that take the vector index as an additional tag. If no such function exists,
    /// the vectorizer may generate the components automatically. [Dualizer] does
    /// this using the function's symbolic derivatives.
    /// The input to the functions
    /// is the flattened vector of all components of all parameters,
    /// followed by all previously computed output components.
    ///
    /// # Example
    ///
    /// Create a dual number and evaluate an expression over it:
    /// ```
    /// use ahash::HashMap;
    /// use symbolica::prelude::*;
    /// use symbolica::create_hyperdual_single_derivative;
    ///
    /// create_hyperdual_single_derivative!(Dual2, 2);
    ///
    /// let ev = parse!("sin(x+y)^2+cos(x+y)^2 - 1")
    ///     .evaluator(&[parse!("x"), parse!("y")])
    ///     .build()
    ///    .unwrap();
    ///
    /// let dualizer = Dualizer::new(Dual2::<Complex<Rational>>::new_zero(), vec![]);
    /// let vec_ev = ev.vectorize(&dualizer).unwrap();
    ///
    /// let mut vec_f = vec_ev.map_coeff(&|x| x.re.to_f64());
    /// let mut dest = vec![0.; 3];
    /// vec_f.evaluate(&[2.0, 1.0, 0., 3.0, 1.0, 1.], &mut dest);
    ///
    /// assert!(dest.iter().all(|x| x.abs() < 1e-10));
    /// ```
    pub fn vectorize<V: Vectorize<T>>(mut self, v: &V) -> Result<ExpressionEvaluator<T>, String> {
        let mut new_external_fns = vec![];
        let mut external_fn_index_map = HashMap::default();
        let mut unknown_constants = vec![false; self.reserved_indices - self.param_count];
        for external_fn in &self.external_fns {
            if external_fn.sub_evaluator.is_some() {
                return Err(format!(
                    "Sub-evaluator '{}' cannot be vectorized",
                    external_fn.symbol.get_name()
                ));
            }

            if let Some(c_index) = external_fn.constant_index {
                unknown_constants[c_index] = true;
                let d = if v.duplicate_constants() {
                    v.get_dimension()
                } else {
                    1
                };
                for i in 0..d {
                    let mut e = external_fn.clone();
                    e.constant_index = Some(c_index * v.get_dimension() + i);
                    new_external_fns.push(e);
                }
                continue;
            }

            let Some(s) = get_symbol!(format!("{}_v", external_fn.symbol.get_name())) else {
                continue;
            };

            for i in 0..v.get_dimension() {
                let mut tags = external_fn.tags.clone();
                tags.push(i.into());
                new_external_fns.push(ExternalFunctionContainer::new(s, tags, vec![]));
                external_fn_index_map.insert((external_fn.clone(), i), new_external_fns.len() - 1);
            }
        }

        self.undo_stack_optimization();

        // unfold every instruction to a single operation
        let mut new_instr = vec![];
        for (x, c) in &mut self.instructions {
            match x {
                Instr::Add(o, a) => {
                    new_instr.push((Instr::Add(*o, vec![a[0], a[1]]), *c));
                    for x in a.iter().skip(2) {
                        new_instr.push((Instr::Add(*o, vec![*o, *x]), *c));
                    }
                }
                Instr::Mul(o, a) => {
                    new_instr.push((Instr::Mul(*o, vec![a[0], a[1]]), *c));
                    for x in a.iter().skip(2) {
                        new_instr.push((Instr::Mul(*o, vec![*o, *x]), *c));
                    }
                }
                _ => new_instr.push((x.clone(), *c)),
            }
        }

        self.instructions = new_instr;

        let mut constants = vec![];
        for c in &self.stack[self.param_count..self.reserved_indices] {
            if v.duplicate_constants() {
                for _ in 0..v.get_dimension() {
                    constants.push(c.clone());
                }
            } else {
                constants.push(c.clone());
                for _ in 1..v.get_dimension() {
                    constants.push(T::default());
                }
            }
        }
        let old_constants_num = constants.len();

        let mut slot_map = HashMap::default();
        for x in 0..self.reserved_indices {
            slot_map.insert(x, x * v.get_dimension()); // set the start of the vector
        }

        self.param_count *= v.get_dimension();
        self.reserved_indices *= v.get_dimension();
        macro_rules! get_slot {
            ($i:expr) => {
                if $i < self.param_count {
                    Slot::Param($i)
                } else if $i < self.reserved_indices {
                    Slot::Const($i - self.param_count)
                } else {
                    Slot::Temp($i - self.reserved_indices)
                }
            };
        }

        macro_rules! from_slot {
            ($i:expr) => {
                match $i {
                    Slot::Param(x) => x,
                    Slot::Const(x) => x + self.param_count,
                    Slot::Temp(x) => x + self.reserved_indices,
                    Slot::Out(_) => unreachable!(),
                }
            };
        }

        let mut ins = InstructionList {
            instructions: vec![],
            constants,
            unknown_constants,
            dim: v.get_dimension(),
        };

        let mut generated_functions = HashMap::default();
        let mut next_label = self
            .instructions
            .iter()
            .filter_map(|(i, _)| match i {
                Instr::Label(l) => Some(l.0 + 1),
                _ => None,
            })
            .max()
            .unwrap_or(0);

        for (i, _sc) in self.instructions.drain(..) {
            let (o, instr) = match i {
                Instr::Add(o, a) => (
                    o,
                    VectorInstruction::Add(get_slot!(slot_map[&a[0]]), get_slot!(slot_map[&a[1]])),
                ),
                Instr::Mul(o, a) => (
                    o,
                    VectorInstruction::Mul(get_slot!(slot_map[&a[0]]), get_slot!(slot_map[&a[1]])),
                ),
                Instr::Pow(o, a, e) => (o, VectorInstruction::Pow(get_slot!(slot_map[&a]), e)),
                Instr::Powf(o, b, e) => (
                    o,
                    VectorInstruction::Powf(get_slot!(slot_map[&b]), get_slot!(slot_map[&e])),
                ),
                Instr::BuiltinFun(o, f, a) => {
                    (o, VectorInstruction::BuiltinFun(f, get_slot!(slot_map[&a])))
                }
                Instr::ExternalFun(o, f, a) => {
                    if !external_fn_index_map.contains_key(&(self.external_fns[f].clone(), 0)) {
                        let external = &self.external_fns[f];
                        let generated = match generated_functions.entry((f, a.len())) {
                            Entry::Occupied(entry) => entry.into_mut(),
                            Entry::Vacant(entry) => {
                                let mut evaluator = v.vectorize_function(external.symbol, &external.tags, a.len())?
                                    .ok_or_else(|| format!(
                                        "To vectorize the function '{0}', the symbol '{0}_v' must be defined that takes the vector index as an additional tag",
                                        external.symbol.get_name()
                                    ))?;
                                if evaluator.param_count != a.len() * v.get_dimension()
                                    || evaluator.result_indices.len() != v.get_dimension()
                                {
                                    return Err(format!(
                                        "Generated vector function '{}' has incompatible input/output dimensions",
                                        external.symbol
                                    ));
                                }
                                evaluator.undo_stack_optimization();
                                entry.insert(evaluator)
                            }
                        };
                        let args = a
                            .iter()
                            .flat_map(|x| {
                                let slot = get_slot!(slot_map[x]);
                                (0..v.get_dimension()).map(move |k| slot.index(k))
                            })
                            .collect::<Vec<_>>();
                        let results = generated.inline_vector_components(
                            &args,
                            &mut ins,
                            &mut new_external_fns,
                            &mut next_label,
                        );
                        // Keep the components contiguous, as required by slot_map.
                        for result in results {
                            ins.add(VectorInstruction::Assign(result));
                        }
                        slot_map.insert(
                            o,
                            ins.instructions.len() + self.reserved_indices - v.get_dimension(),
                        );
                        continue;
                    }
                    let mut results = vec![];
                    for j in 0..v.get_dimension() {
                        let Some(index) =
                            external_fn_index_map.get(&(self.external_fns[f].clone(), j))
                        else {
                            return Err(format!(
                                "No external function mapping found for function '{}' with index {}",
                                self.external_fns[f], j
                            ));
                        };

                        // call with flattened arguments and pass all previously computed components
                        let r = ins.add(VectorInstruction::ExternalFun(
                            *index,
                            a.iter()
                                .map(|x| get_slot!(slot_map[x]))
                                .flat_map(|x| (0..v.get_dimension()).map(move |k| x.index(k)))
                                .chain(results.iter().cloned())
                                .collect(),
                        ));
                        results.push(r);
                    }

                    slot_map.insert(
                        o,
                        ins.instructions.len() + self.reserved_indices - v.get_dimension(),
                    );
                    continue;
                }
                Instr::Goto(l) => {
                    ins.instructions.push(VectorInstruction::Goto(l));
                    continue;
                }
                Instr::Label(l) => {
                    ins.instructions.push(VectorInstruction::Label(l));
                    continue;
                }
                Instr::IfElse(c, l) => {
                    ins.instructions
                        .push(VectorInstruction::IfElse(get_slot!(slot_map[&c]), l));
                    continue;
                }
                Instr::Join(o, c, t, f) => (
                    o,
                    VectorInstruction::Join(
                        get_slot!(slot_map[&c]),
                        get_slot!(slot_map[&t]),
                        get_slot!(slot_map[&f]),
                    ),
                ),
            };

            let r = v.map_instruction(&instr, &mut ins);
            assert_eq!(r.len(), v.get_dimension());
            for ii in r {
                ins.add(ii);
            }

            slot_map.insert(
                o,
                ins.instructions.len() + self.reserved_indices - v.get_dimension(),
            );
        }

        self.stack.clear();
        self.stack.resize(self.param_count, T::default());
        self.stack.extend(ins.constants);

        let stack_shift = self.stack.len() - old_constants_num - self.param_count;

        let mut new_result_indices = vec![];
        for x in 0..self.result_indices.len() {
            let mut p = slot_map[&self.result_indices[x]];
            if p >= self.reserved_indices {
                p += stack_shift;
            }

            for i in 0..v.get_dimension() {
                new_result_indices.push(p + i);
            }
        }

        self.reserved_indices += stack_shift;
        self.result_indices = new_result_indices;

        for i in ins.instructions {
            let out = self.instructions.len() + self.reserved_indices;
            match i {
                VectorInstruction::Add(slot, slot1) => {
                    let mut s1 = from_slot!(slot);
                    let mut s2 = from_slot!(slot1);
                    if s1 > s2 {
                        (s1, s2) = (s2, s1);
                    }

                    self.instructions
                        .push((Instr::Add(out, vec![s1, s2]), ComplexPhase::Any));
                }
                VectorInstruction::Assign(slot) => {
                    self.instructions
                        .push((Instr::Add(out, vec![from_slot!(slot)]), ComplexPhase::Any));
                }
                VectorInstruction::Mul(slot, slot1) => {
                    let mut s1 = from_slot!(slot);
                    let mut s2 = from_slot!(slot1);
                    if s1 > s2 {
                        (s1, s2) = (s2, s1);
                    }

                    self.instructions
                        .push((Instr::Mul(out, vec![s1, s2]), ComplexPhase::Any));
                }
                VectorInstruction::Pow(slot, e) => {
                    self.instructions
                        .push((Instr::Pow(out, from_slot!(slot), e), ComplexPhase::Any));
                }
                VectorInstruction::Powf(slot, slot1) => {
                    self.instructions.push((
                        Instr::Powf(out, from_slot!(slot), from_slot!(slot1)),
                        ComplexPhase::Any,
                    ));
                }
                VectorInstruction::BuiltinFun(builtin_symbol, slot) => {
                    self.instructions.push((
                        Instr::BuiltinFun(out, builtin_symbol, from_slot!(slot)),
                        ComplexPhase::Any,
                    ));
                }
                VectorInstruction::ExternalFun(f, args) => {
                    self.instructions.push((
                        Instr::ExternalFun(out, f, args.iter().map(|x| from_slot!(*x)).collect()),
                        ComplexPhase::Any,
                    ));
                }
                VectorInstruction::IfElse(cond, label) => {
                    self.instructions
                        .push((Instr::IfElse(from_slot!(cond), label), ComplexPhase::Any));
                }
                VectorInstruction::Goto(label) => {
                    self.instructions
                        .push((Instr::Goto(label), ComplexPhase::Any));
                }
                VectorInstruction::Label(label) => {
                    self.instructions
                        .push((Instr::Label(label), ComplexPhase::Any));
                }
                VectorInstruction::Join(cond, t, f) => {
                    self.instructions.push((
                        Instr::Join(out, from_slot!(cond), from_slot!(t), from_slot!(f)),
                        ComplexPhase::Any,
                    ));
                }
            }
        }

        self.stack.resize(
            self.reserved_indices + self.instructions.len(),
            T::default(),
        );
        self.external_fns = new_external_fns;

        self.remove_common_pairs();
        self.optimize_stack();
        self.fix_labels();

        Ok(self)
    }
}

/// A trait to define how to vectorize coefficients and instructions.
/// Every slot is mapped to `n` slots and every instruction is mapped to `n` instructions, where `n` is the dimension.
pub trait Vectorize<T> {
    /// Return `true` if the constants are duplicated across the vector dimension, and
    /// `false` if each constant only appears in the first slot and has zeros for all other slots.
    fn duplicate_constants(&self) -> bool;

    /// Map an instruction applied to a vector of slots (components accessible with [Slot::index])
    /// to a vector of instructions of [Vectorize::get_dimension] length.
    fn map_instruction(
        &self,
        instr: &VectorInstruction,
        instr_addr: &mut InstructionList<T>,
    ) -> Vec<VectorInstruction>;

    /// Get the dimension of the vectorization.
    fn get_dimension(&self) -> usize;

    /// Generate a scalar evaluator whose inputs and outputs are flattened vector
    /// components. Called for external functions without an explicit `_v` symbol.
    /// Return `None` if automatic vectorization is not supported.
    fn vectorize_function(
        &self,
        _symbol: Symbol,
        _tags: &[Atom],
        _argument_count: usize,
    ) -> Result<Option<ExpressionEvaluator<T>>, String> {
        Ok(None)
    }
}

/// A dualizer that maps coefficients and instructions to dual number components.
///
/// You can specify which components of the dual numbers are always zero
/// by providing a list of `(component index, dual index)` pairs in the constructor.
pub struct Dualizer<T: DualNumberStructure> {
    dual: T,
    zero_components: HashSet<(usize, usize)>, // component, index
}

impl<T: DualNumberStructure> Dualizer<T> {
    /// Create a new dualizer for the given dual number structure.
    /// You can specify which components are always zero
    /// by providing a list of `(component index, dual index)` pairs.
    pub fn new(dual: T, zero_components_per_parameter: Vec<(usize, usize)>) -> Self {
        Self {
            dual,
            zero_components: zero_components_per_parameter.into_iter().collect(),
        }
    }
}

impl<T: DualNumberStructure> Vectorize<Complex<Rational>> for Dualizer<T> {
    fn duplicate_constants(&self) -> bool {
        false
    }

    fn vectorize_function(
        &self,
        symbol: Symbol,
        tags: &[Atom],
        argument_count: usize,
    ) -> Result<Option<ExpressionEvaluator<Complex<Rational>>>, String> {
        let shape = self.dual.get_shape();
        let dim = shape.len();
        let indices = shape
            .iter()
            .enumerate()
            .map(|(i, s)| (*s, i))
            .collect::<HashMap<_, _>>();
        let mut parameters = Vec::with_capacity(argument_count * dim);
        let mut index = 0usize;
        let parameter_symbol = crate::symbol!("symbolica::evaluate::dual_arg");
        while parameters.len() < argument_count * dim {
            let parameter = crate::function!(parameter_symbol, index);
            index += 1;
            // Tags are constants and must not capture an internal parameter.
            if !tags.iter().any(|tag| tag.contains(&parameter)) {
                parameters.push(parameter);
            }
        }
        let variables = parameters
            .iter()
            .cloned()
            .map(|p| Indeterminate::try_from(p).unwrap())
            .collect::<Vec<_>>();
        let mut function = crate::atom::FunctionBuilder::new(symbol);
        for tag in tags {
            function = function.add_arg(tag);
        }
        for args in parameters.chunks(dim) {
            function = function.add_arg(&args[0]);
        }
        let mut components = vec![Atom::new(); dim];
        components[0] = function.finish();

        // Components are Taylor coefficients. For a component alpha, choose
        // beta = alpha - e_i and apply the total derivative D_i to c_beta:
        // D_i = sum_(arg,gamma) (gamma_i + 1) a_(arg,gamma+e_i) d/da_(arg,gamma).
        // Dividing by alpha_i restores the Taylor factorial convention.
        let mut order = (1..dim).collect::<Vec<_>>();
        order.sort_by_key(|i| shape[*i].iter().sum::<usize>());
        for component in order {
            let alpha = shape[component];
            let axis = alpha
                .iter()
                .position(|p| *p > 0)
                .ok_or("Dual shape has a non-leading scalar component")?;
            let mut beta = alpha.to_vec();
            beta[axis] -= 1;
            let parent = &components[*indices
                .get(beta.as_slice())
                .ok_or("Dual shape is not ancestor-closed")?];
            let mut derivative = Atom::new();
            for (j, gamma) in shape.iter().enumerate() {
                let mut successor = gamma.to_vec();
                successor[axis] += 1;
                let Some(&next) = indices.get(successor.as_slice()) else {
                    continue;
                };
                for argument in 0..argument_count {
                    derivative += parent.derivative(&variables[argument * dim + j])
                        * &parameters[argument * dim + next]
                        * Atom::num(successor[axis]);
                }
            }
            components[component] = derivative / Atom::num(alpha[axis]);
            if components[component].contains_symbol(Symbol::DERIVATIVE) {
                return Err(format!(
                    "Cannot automatically dualize '{}': no explicit derivative for component {:?}: {}",
                    symbol.get_name(),
                    alpha,
                    components[component]
                ));
            }
        }
        let evaluator = Atom::evaluator_multiple(&components, &parameters)
            .build()
            .map_err(|e| format!("Cannot automatically dualize '{}': {e}", symbol.get_name()))?;
        Ok(Some(evaluator))
    }

    fn map_instruction(
        &self,
        i: &VectorInstruction,
        instrs: &mut InstructionList<Complex<Rational>>,
    ) -> Vec<VectorInstruction> {
        fn is_zero<T: DualNumberStructure>(
            a: &Slot,
            dualizer: &Dualizer<T>,
            instrs: &InstructionList<Complex<Rational>>,
        ) -> bool {
            if instrs.is_zero(a) {
                return true;
            }

            if let Slot::Param(x) = a {
                dualizer
                    .zero_components
                    .contains(&(*x / dualizer.get_dimension(), *x % dualizer.get_dimension()))
            } else {
                false
            }
        }

        fn scalar_add<T: DualNumberStructure>(
            a: &Slot,
            b: &Slot,
            dualizer: &Dualizer<T>,
            instrs: &mut InstructionList<Complex<Rational>>,
        ) -> Slot {
            if is_zero(a, dualizer, instrs) {
                *b
            } else if instrs.is_zero(b) {
                *a
            } else {
                instrs.add(VectorInstruction::Add(*a, *b))
            }
        }

        fn scalar_yield_add<T: DualNumberStructure>(
            a: &Slot,
            b: &Slot,
            dualizer: &Dualizer<T>,
            instrs: &mut InstructionList<Complex<Rational>>,
        ) -> VectorInstruction {
            if is_zero(a, dualizer, instrs) {
                VectorInstruction::Assign(*b)
            } else if is_zero(b, dualizer, instrs) {
                VectorInstruction::Assign(*a)
            } else {
                VectorInstruction::Add(*a, *b)
            }
        }

        fn scalar_mul<T: DualNumberStructure>(
            a: &Slot,
            b: &Slot,
            dualizer: &Dualizer<T>,
            instrs: &mut InstructionList<Complex<Rational>>,
        ) -> Slot {
            if is_zero(a, dualizer, instrs) || instrs.is_one(b) {
                *a
            } else if is_zero(b, dualizer, instrs) || instrs.is_one(a) {
                *b
            } else {
                instrs.add(VectorInstruction::Mul(*a, *b))
            }
        }

        fn scalar_yield_mul<T: DualNumberStructure>(
            a: &Slot,
            b: &Slot,
            dualizer: &Dualizer<T>,
            instrs: &mut InstructionList<Complex<Rational>>,
        ) -> VectorInstruction {
            if is_zero(a, dualizer, instrs) || instrs.is_one(b) {
                VectorInstruction::Assign(*a)
            } else if is_zero(b, dualizer, instrs) || instrs.is_one(a) {
                VectorInstruction::Assign(*b)
            } else {
                VectorInstruction::Mul(*a, *b)
            }
        }

        fn rescale<T: DualNumberStructure>(
            a: &[Slot],
            c: &Slot,
            dualizer: &Dualizer<T>,
            instrs: &mut InstructionList<Complex<Rational>>,
        ) -> Vec<Slot> {
            a.iter()
                .map(|x| scalar_mul(x, c, dualizer, instrs))
                .collect()
        }

        fn add<T: DualNumberStructure>(
            a: &[Slot],
            b: &[Slot],
            dualizer: &Dualizer<T>,
            instrs: &mut InstructionList<Complex<Rational>>,
        ) -> Vec<Slot> {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| scalar_add(x, y, dualizer, instrs))
                .collect()
        }

        fn mul<T: DualNumberStructure>(
            a: &[Slot],
            b: &[Slot],
            table: &[(usize, usize, usize)],
            dualizer: &Dualizer<T>,
            instrs: &mut InstructionList<Complex<Rational>>,
        ) -> Vec<Slot> {
            let mut current_index = vec![];
            for j in 0..a.len() {
                current_index.push(scalar_mul(&a[j], &b[0], dualizer, instrs));
            }

            for (si, oi, index) in table.iter() {
                let tmp = scalar_mul(&a[*si], &b[*oi], dualizer, instrs);
                current_index[*index] = scalar_add(&current_index[*index], &tmp, dualizer, instrs);
            }
            current_index
        }

        let mult_table = self.dual.get_multiplication_table();

        match i {
            VectorInstruction::Add(a, b) => (0..self.dual.get_len())
                .map(|j| scalar_yield_add(&a.index(j), &b.index(j), self, instrs))
                .collect(),
            VectorInstruction::Mul(a, b) => {
                let mut current_index = vec![];
                for j in 0..self.dual.get_len() {
                    current_index.push(scalar_mul(&a.index(j), b, self, instrs));
                }

                for (si, oi, index) in self.dual.get_multiplication_table().iter() {
                    let tmp = scalar_mul(&a.index(*si), &b.index(*oi), self, instrs);
                    current_index[*index] = scalar_add(&current_index[*index], &tmp, self, instrs);
                }

                current_index
                    .iter()
                    .map(|x| VectorInstruction::Assign(*x))
                    .collect()
            }
            VectorInstruction::BuiltinFun(f, a) => match f.get_id() {
                Symbol::SQRT_ID => {
                    let e = instrs.add(VectorInstruction::BuiltinFun(*f, *a));
                    let norm = instrs.add(VectorInstruction::Pow(*a, -1)); // TODO: check 0?

                    let zero = instrs.add_repeated_constant(Complex::new_zero());
                    let mut r = vec![zero];
                    r.extend(
                        (1..self.dual.get_len())
                            .map(|j| scalar_mul(&a.index(j), &norm, self, instrs)),
                    );

                    let one =
                        instrs.add_constant_in_first_component(Complex::from(Rational::one()));

                    let mut accum = (0..self.dual.get_len())
                        .map(|j| one.index(j))
                        .collect::<Vec<_>>();
                    let mut res = (0..self.dual.get_len())
                        .map(|j| one.index(j))
                        .collect::<Vec<_>>();
                    let mut num = Complex::from(Rational::one());

                    let mut scale = 1;
                    for p in 1..self.dual.get_max_depth() + 1 {
                        scale *= p;
                        num = num.clone() * (num.from_usize(2).inv() - &num.from_usize(p - 1));
                        accum = mul(&accum, &r, mult_table, self, instrs);

                        let c = instrs
                            .add_constant_in_first_component(&num * &num.from_usize(scale).inv());

                        res = add(&res, &rescale(&accum, &c, self, instrs), self, instrs);
                    }

                    res.iter()
                        .map(|x| scalar_yield_mul(x, &e, self, instrs))
                        .collect()
                }
                Symbol::EXP_ID => {
                    let e = instrs.add(VectorInstruction::BuiltinFun(*f, *a));

                    let one =
                        instrs.add_constant_in_first_component(Complex::from(Rational::one()));

                    let mut accum = (0..self.dual.get_len())
                        .map(|j| one.index(j))
                        .collect::<Vec<_>>();
                    let mut res = (0..self.dual.get_len())
                        .map(|j| one.index(j))
                        .collect::<Vec<_>>();

                    let zero = instrs.add_repeated_constant(Complex::new_zero());
                    let mut r = vec![zero];
                    r.extend((1..self.dual.get_len()).map(|j| a.index(j)));
                    let mut scale = Complex::from(Rational::one());
                    for p in 0..self.dual.get_max_depth() {
                        scale *= Rational::from(p + 1);
                        accum = mul(&accum, &r, mult_table, self, instrs);

                        let c = instrs.add_constant_in_first_component(scale.inv());

                        res = add(&res, &rescale(&accum, &c, self, instrs), self, instrs);
                    }

                    res.iter()
                        .map(|x| scalar_yield_mul(x, &e, self, instrs))
                        .collect()
                }
                Symbol::LOG_ID => {
                    let e = instrs.add(VectorInstruction::BuiltinFun(*f, *a));

                    let norm = instrs.add(VectorInstruction::Pow(*a, -1)); // TODO: check 0?

                    let zero = instrs.add_repeated_constant(Complex::new_zero());
                    let mut r = vec![zero];
                    r.extend(
                        (1..self.dual.get_len())
                            .map(|j| scalar_mul(&a.index(j), &norm, self, instrs)),
                    );

                    let mut accum = r.clone();

                    let mut res = (0..self.dual.get_len()).map(|_| zero).collect::<Vec<_>>();
                    res[0] = e;

                    let mut scale = Complex::from(Rational::from(-1));
                    for p in 1..self.dual.get_max_depth() + 1 {
                        scale *= Rational::from(-1);

                        let c = instrs
                            .add_constant_in_first_component((&scale * Rational::from(p)).inv());

                        res = add(&res, &rescale(&accum, &c, self, instrs), self, instrs);
                        accum = mul(&accum, &r, mult_table, self, instrs);
                    }

                    res.iter().map(|x| VectorInstruction::Assign(*x)).collect()
                }
                Symbol::SIN_ID => {
                    let s = instrs.add(VectorInstruction::BuiltinFun(*f, *a));
                    let c = instrs.add(VectorInstruction::BuiltinFun(Symbol::COS, *a));

                    let zero = instrs.add_repeated_constant(Complex::new_zero());
                    let mut p = vec![zero];
                    p.extend((1..self.dual.get_len()).map(|j| a.index(j)));

                    let mut e = (0..self.dual.get_len()).map(|_| zero).collect::<Vec<_>>();
                    e[0] = s;

                    let mut sp = p.clone();
                    let mut scale = Complex::from(Rational::one());
                    for i in 1..self.dual.get_max_depth() + 1 {
                        scale *= Rational::from(i);
                        let b = if i % 2 == 1 { c } else { s };

                        let sc = instrs.add_constant_in_first_component(if i % 4 >= 2 {
                            -scale.inv()
                        } else {
                            scale.inv()
                        });

                        let s = rescale(&sp, &scalar_mul(&b, &sc, self, instrs), self, instrs);

                        sp = mul(&sp, &p, mult_table, self, instrs);

                        e = add(&e, &s, self, instrs);
                    }

                    e.iter().map(|x| VectorInstruction::Assign(*x)).collect()
                }
                Symbol::COS_ID => {
                    let s = instrs.add(VectorInstruction::BuiltinFun(Symbol::SIN, *a));
                    let c = instrs.add(VectorInstruction::BuiltinFun(*f, *a));

                    let zero = instrs.add_repeated_constant(Complex::new_zero());
                    let mut p = vec![zero];
                    p.extend((1..self.dual.get_len()).map(|j| a.index(j)));

                    let mut e = (0..self.dual.get_len()).map(|_| zero).collect::<Vec<_>>();
                    e[0] = c;

                    let mut sp = p.clone();
                    let mut scale = Complex::from(Rational::one());
                    for i in 1..self.dual.get_max_depth() + 1 {
                        scale *= Rational::from(i);
                        let b = if i % 2 == 1 { s } else { c };

                        let sc =
                            instrs.add_constant_in_first_component(if (i % 2 == 0) ^ (i % 4 < 2) {
                                -scale.inv()
                            } else {
                                scale.inv()
                            });

                        let s = rescale(&sp, &scalar_mul(&b, &sc, self, instrs), self, instrs);

                        sp = mul(&sp, &p, mult_table, self, instrs);

                        e = add(&e, &s, self, instrs);
                    }

                    e.iter().map(|x| VectorInstruction::Assign(*x)).collect()
                }
                Symbol::ABS_ID => {
                    let n = instrs.add(VectorInstruction::BuiltinFun(Symbol::ABS, *a));

                    let inv_val = instrs.add(VectorInstruction::Pow(*a, -1));
                    let scale = instrs.add(VectorInstruction::Mul(n, inv_val));

                    (0..self.dual.get_len())
                        .map(|j| scalar_yield_mul(&a.index(j), &scale, self, instrs))
                        .collect()
                }
                Symbol::CONJ_ID => {
                    // assume variables are real
                    (0..self.dual.get_len())
                        .map(|j| VectorInstruction::BuiltinFun(*f, a.index(j)))
                        .collect()
                }
                _ => unimplemented!(
                    "Vectorization not implemented for built-in function {}",
                    f.get_name()
                ),
            },
            VectorInstruction::Pow(a, b) => {
                assert_eq!(*b, -1); // only b = -1 is used in practice
                let a_inv = instrs.add(VectorInstruction::Pow(*a, -1));

                let zero = instrs.add_repeated_constant(Complex::new_zero());
                let mut r = vec![zero];
                r.extend(
                    (1..self.dual.get_len()).map(|j| scalar_mul(&a.index(j), &a_inv, self, instrs)),
                );

                let one = instrs.add_constant_in_first_component(Complex::from(Rational::one()));
                let neg_one =
                    instrs.add_constant_in_first_component(Complex::from(Rational::from(-1)));

                let mut accum = (0..self.dual.get_len())
                    .map(|j| one.index(j))
                    .collect::<Vec<_>>();
                let mut res = (0..self.dual.get_len())
                    .map(|j| one.index(j))
                    .collect::<Vec<_>>();

                for i in 1..self.dual.get_max_depth() + 1 {
                    accum = mul(&accum, &r, mult_table, self, instrs);
                    if i % 2 == 0 {
                        res = add(&res, &accum, self, instrs);
                    } else {
                        res = add(&res, &rescale(&accum, &neg_one, self, instrs), self, instrs);
                    }
                }

                res.iter()
                    .map(|x| scalar_yield_mul(x, &a_inv, self, instrs))
                    .collect()
            }
            VectorInstruction::Powf(b, e) => {
                let input = VectorInstruction::BuiltinFun(Symbol::LOG, *b);
                let log: Vec<_> = self
                    .map_instruction(&input, instrs)
                    .into_iter()
                    .map(|x| instrs.add(x))
                    .collect();
                let e = (0..self.dual.get_len())
                    .map(|j| e.index(j))
                    .collect::<Vec<_>>();
                let r = mul(&log, &e, mult_table, self, instrs);

                // exp needs adjacent slots
                let adjacent: Vec<_> = r
                    .into_iter()
                    .map(|x| instrs.add(VectorInstruction::Assign(x)))
                    .collect();
                let exp_in = VectorInstruction::BuiltinFun(Symbol::EXP, adjacent[0]);
                self.map_instruction(&exp_in, instrs)
            }
            VectorInstruction::Join(c, a, b) => (0..self.dual.get_len())
                .map(|j| VectorInstruction::Join(c.index(0), a.index(j), b.index(j)))
                .collect(),
            VectorInstruction::Assign(a) => (0..self.dual.get_len())
                .map(|j| VectorInstruction::Assign(a.index(j)))
                .collect(),
            VectorInstruction::ExternalFun(_, _)
            | VectorInstruction::Goto(_)
            | VectorInstruction::Label(_)
            | VectorInstruction::IfElse(..) => {
                unreachable!(
                    "Instruction {:?} should not appear inside vectorized instructions",
                    i
                )
            }
        }
    }

    fn get_dimension(&self) -> usize {
        self.dual.get_len()
    }
}

#[cfg(test)]
mod test {
    use numerica::domains::{dual::HyperDual, float::Complex, rational::Rational};

    use crate::{
        atom::{Atom, AtomCore, EvaluationInfo},
        evaluate::Dualizer,
        function, parse, symbol,
    };

    fn dualize(
        expression: &str,
        parameters: &[Atom],
        shape: Vec<Vec<usize>>,
    ) -> super::ExpressionEvaluator<Complex<Rational>> {
        parse!(expression)
            .evaluator(parameters)
            .build()
            .unwrap()
            .vectorize(&Dualizer::new(
                HyperDual::<Complex<Rational>>::new(shape),
                vec![],
            ))
            .unwrap()
    }

    #[test]
    fn automatic_tanh_higher_derivatives() {
        let evaluator = dualize(
            "tanh(x)",
            &[parse!("x")],
            vec![vec![0], vec![1], vec![2], vec![3]],
        );
        let mut real = evaluator.clone().map_coeff(&|r| r.re.to_f64());
        let x = 0.4_f64;
        let value = x.tanh();
        let slope = 1. - value * value;
        let mut out = [0.; 4];
        real.evaluate(&[x, 2., 3., 4.], &mut out);
        let expected = [
            value,
            2. * slope,
            slope * (3. - 4. * value),
            4. * slope - 12. * value * slope + 8. / 3. * slope * (3. * value * value - 1.),
        ];
        for (actual, expected) in out.into_iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-12, "{actual} != {expected}");
        }
        let mut complex = evaluator.map_coeff(&|r| Complex::new(r.re.to_f64(), r.im.to_f64()));
        let z = Complex::new(0.4, 0.2);
        let mut out = [Complex::new(0., 0.); 4];
        complex.evaluate(
            &[
                z,
                Complex::new(1., 0.),
                Complex::new(0., 0.),
                Complex::new(0., 0.),
            ],
            &mut out,
        );
        use crate::domains::float::Real;
        let expected = z.tanh();
        assert!((out[0] - expected).norm().re < 1e-12);
        assert!(
            (out[1] - (Complex::new(1., 0.) - expected * expected))
                .norm()
                .re
                < 1e-12
        );
    }

    #[test]
    fn automatic_mixed_derivatives_and_multiple_arguments() {
        // Nonstandard component order and nonzero higher input coefficients.
        let shape = vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1], vec![2, 0]];
        let expressions = [
            "atan(x,y)",
            "tanh(x*y)+sinh(x+y)",
            "polygamma(2,x)",
            "polylog(3,x)",
            "bessel_j(2,x)",
            "gamma(x)",
            "erf(x)",
        ];
        let x = symbol!("x");
        let y = symbol!("y");
        for expression in expressions {
            let expr = parse!(expression);
            let mut expected = Atom::evaluator_multiple(
                &[
                    expr.clone(),
                    expr.derivative(y),
                    expr.derivative(x),
                    expr.derivative(x).derivative(y),
                    expr.derivative(x).derivative(x) / 2,
                ],
                &[Atom::from(x), Atom::from(y)],
            )
            .build()
            .unwrap()
            .map_coeff(&|r| r.re.to_f64());
            let mut actual = dualize(expression, &[x.into(), y.into()], shape.clone())
                .map_coeff(&|r| r.re.to_f64());
            let mut a = [0.; 5];
            let mut b = [0.; 5];
            actual.evaluate(&[0.4, 0., 1., 0., 0., 0.7, 1., 0., 0., 0.], &mut a);
            expected.evaluate(&[0.4, 0.7], &mut b);
            for (a, b) in a.into_iter().zip(b) {
                assert!(
                    (a - b).abs() < 1e-10 * b.abs().max(1.),
                    "{expression}: {a} != {b}"
                );
            }
        }
    }

    #[test]
    fn automatic_custom_function_reuses_scalar_output() {
        let f = symbol!(
            "auto_dual_exp",
            der = |f, i, out| {
                if i == 0 {
                    **out = f.to_owned();
                }
            },
            eval = EvaluationInfo::new().register(|args: &[f64]| args[0].exp())
        );
        let x = parse!("x");
        let expression = function!(f, &x);
        let evaluator = expression
            .evaluator(&[x])
            .build()
            .unwrap()
            .vectorize(&Dualizer::new(
                HyperDual::<Complex<Rational>>::new(vec![vec![0], vec![1], vec![2]]),
                vec![],
            ))
            .unwrap();
        assert_eq!(evaluator.count_operations().function_calls, 1);
        let mut evaluator = evaluator.map_coeff(&|r| r.re.to_f64());
        let mut out = [0.; 3];
        evaluator.evaluate(&[0.3, 2., 3.], &mut out);
        assert!((out[0] - 0.3_f64.exp()).abs() < 1e-12);
        assert!((out[1] - 2. * out[0]).abs() < 1e-12);
        assert!((out[2] - 5. * out[0]).abs() < 1e-12);
    }

    #[test]
    fn missing_derivatives_report_an_error() {
        let f = symbol!(
            "auto_dual_missing",
            eval = EvaluationInfo::new().register(|args: &[f64]| args[0])
        );
        let x = parse!("x");
        let error = function!(f, &x)
            .evaluator(&[x])
            .build()
            .unwrap()
            .vectorize(&Dualizer::new(
                HyperDual::<Complex<Rational>>::new(vec![vec![0], vec![1]]),
                vec![],
            ))
            .unwrap_err();
        assert!(error.contains("no explicit derivative"), "{error}");
        assert!(error.contains("auto_dual_missing"), "{error}");
    }

    #[test]
    fn explicit_vector_function_still_overrides_generation() {
        let f = symbol!(
            "auto_dual_override",
            eval = EvaluationInfo::new().register(|args: &[f64]| args[0])
        );
        symbol!(
            "auto_dual_override_v",
            eval = EvaluationInfo::new().with_tags(1).register_tagged(|tags| {
                let scalar = tags[0] == 0;
                Box::new(move |args: &[f64]| if scalar { args[0] } else { 7. * args[1] })
            })
        );
        let x = parse!("x");
        let mut evaluator = function!(f, &x)
            .evaluator(&[x])
            .build()
            .unwrap()
            .vectorize(&Dualizer::new(
                HyperDual::<Complex<Rational>>::new(vec![vec![0], vec![1]]),
                vec![],
            ))
            .unwrap()
            .map_coeff(&|r| r.re.to_f64());
        let mut out = [0.; 2];
        evaluator.evaluate(&[3., 2.], &mut out);
        assert_eq!(out, [3., 14.]);
    }

    #[test]
    fn generated_derivatives_preserve_branches_and_labels() {
        let f = symbol!(
            "auto_dual_branch",
            der = |f, i, out| {
                if i == 0 {
                    let x = f.as_fun_view().unwrap().iter().next().unwrap();
                    **out = function!(crate::atom::Symbol::IF, x, x.to_owned() * 2, 0);
                }
            },
            eval = EvaluationInfo::new().register(|args: &[f64]| args[0] * args[0])
        );
        let x = parse!("x");
        let c = parse!("c");
        let expr = function!(
            crate::atom::Symbol::IF,
            &c,
            function!(f, &x),
            function!(f, &x + 1)
        );
        let mut evaluator = expr
            .evaluator(&[c, x])
            .build()
            .unwrap()
            .vectorize(&Dualizer::new(
                HyperDual::<Complex<Rational>>::new(vec![vec![0], vec![1]]),
                vec![],
            ))
            .unwrap()
            .map_coeff(&|r| r.re.to_f64());
        for (c, x, expected) in [
            (1., 0.5, [0.25, 2.]),
            (0., 0.5, [2.25, 6.]),
            (1., 0., [0., 0.]),
        ] {
            let mut out = [0.; 2];
            evaluator.evaluate(&[c, 0., x, 2.], &mut out);
            assert_eq!(out, expected);
        }
    }

    #[test]
    fn unknown_constant() {
        let expr = parse!("2*pi*x^2");

        let mut evaluator = expr.evaluator(&[parse!("x")]).build().unwrap();
        let dual = HyperDual::<Complex<Rational>>::new(vec![vec![0], vec![1]]);
        let dualizer = Dualizer::new(dual, vec![]);
        evaluator = evaluator.vectorize(&dualizer).unwrap();

        let mut evaluator = evaluator.map_coeff(&|r| r.re.to_f64());

        let mut out = vec![0.; 2];
        evaluator.evaluate(&[2., 1.], &mut out);
        assert_eq!(out, [25.132741228718345, 25.132741228718345]);
    }
}
