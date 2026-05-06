use super::*;

/// Settings for operation realness of complex evaluators, used in [ExpressionEvaluator::set_real_params].
#[derive(Clone, Debug)]
pub struct ComplexEvaluatorSettings {
    /// Whether sqrt with real arguments yields real results.
    pub sqrt_real: bool,
    /// Whether log with real arguments yields real results.
    pub log_real: bool,
    /// Whether powf with real arguments yields real results.
    pub powf_real: bool,
    /// Report on the number of converted operations.
    pub verbose: bool,
}

impl ComplexEvaluatorSettings {
    /// Create complex evaluator settings, used for [ExpressionEvaluator::set_real_params].
    pub fn new(sqrt_real: bool, log_real: bool, powf_real: bool, verbose: bool) -> Self {
        ComplexEvaluatorSettings {
            sqrt_real,
            log_real,
            powf_real,
            verbose,
        }
    }

    /// Set that all square roots with real arguments yield real results.
    pub fn sqrt_real(mut self) -> Self {
        self.sqrt_real = true;
        self
    }

    /// Set that all logarithms with real arguments yield real results.
    pub fn log_real(mut self) -> Self {
        self.log_real = true;
        self
    }

    /// Set that all powf with real arguments yield real results.
    pub fn powf_real(mut self) -> Self {
        self.powf_real = true;
        self
    }

    /// Set verbose reporting.
    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }
}

impl Default for ComplexEvaluatorSettings {
    /// Create default complex evaluator settings.
    fn default() -> Self {
        ComplexEvaluatorSettings {
            sqrt_real: false,
            log_real: false,
            powf_real: false,
            verbose: false,
        }
    }
}

impl<T: Default + PartialEq> ExpressionEvaluator<Complex<T>> {
    /// Set which parameters are fully real. This allows for more optimal
    /// assembly output that uses real arithmetic instead of complex arithmetic
    /// where possible.
    ///
    /// You can also set if all encountered sqrt, log, and powf operations with real
    /// arguments are expected to yield real results.
    ///
    /// Must be called after all optimization functions and merging are performed
    /// on the evaluator, or the registration will be lost.
    pub fn set_real_params(
        &mut self,
        real_params: &[usize],
        settings: ComplexEvaluatorSettings,
    ) -> Result<(), String> {
        let mut subcomponents = vec![ComplexPhase::Any; self.stack.len()];

        for i in real_params {
            if *i >= self.param_count {
                return Err(format!(
                    "Real parameter index {} out of bounds (parameter count {})",
                    i, self.param_count
                ));
            }

            subcomponents[*i] = ComplexPhase::Real;
        }

        for (s, c) in subcomponents
            .iter_mut()
            .zip(self.stack.iter())
            .skip(self.param_count)
            .take(self.reserved_indices - self.param_count)
        {
            if c.im == T::default() {
                *s = ComplexPhase::Real;
            } else if c.re == T::default() {
                *s = ComplexPhase::Imag;
            }
        }

        let mut div_components = 0;
        let mut mul_components = 0;

        for (instr, sc) in &mut self.instructions {
            let is_add = matches!(instr, Instr::Add(_, _));
            match instr {
                Instr::Add(r, args) | Instr::Mul(r, args) => {
                    let real_parts = args
                        .iter()
                        .filter(|x| subcomponents[**x] == ComplexPhase::Real)
                        .count();

                    if real_parts > 0 && real_parts != args.len() {
                        args.sort_by_key(|x| !matches!(subcomponents[*x], ComplexPhase::Real)); // sort real components first
                    }

                    if !is_add && real_parts > 1 {
                        mul_components += real_parts - 1;
                    }

                    if real_parts == args.len() {
                        *sc = ComplexPhase::Real;
                    } else if args.iter().all(|x| subcomponents[*x] == ComplexPhase::Imag) {
                        *sc = ComplexPhase::Imag;
                    } else if real_parts > 0 {
                        *sc = ComplexPhase::PartialReal(real_parts);
                    } else {
                        *sc = ComplexPhase::Any;
                    }

                    subcomponents[*r] = *sc;
                }
                Instr::Pow(r, b, _) => {
                    if subcomponents[*b] == ComplexPhase::Real {
                        *sc = ComplexPhase::Real;
                        div_components += 1;
                    } else {
                        *sc = ComplexPhase::Any;
                    }
                    subcomponents[*r] = *sc;
                }
                Instr::BuiltinFun(r, s, a) => {
                    if s.is_real() {
                        *sc = ComplexPhase::Real;
                        subcomponents[*r] = *sc;
                        continue;
                    }

                    if subcomponents[*a] != ComplexPhase::Real {
                        subcomponents[*r] = ComplexPhase::Any;
                        *sc = ComplexPhase::Any;
                        continue;
                    }

                    match s.get_id() {
                        Symbol::EXP_ID | Symbol::CONJ_ID | Symbol::SIN_ID | Symbol::COS_ID => {
                            *sc = ComplexPhase::Real;
                        }
                        Symbol::SQRT_ID if settings.sqrt_real => {
                            *sc = ComplexPhase::Real;
                        }
                        Symbol::LOG_ID if settings.log_real => {
                            *sc = ComplexPhase::Real;
                        }
                        _ => {
                            *sc = ComplexPhase::Any;
                        }
                    }

                    subcomponents[*r] = *sc;
                }
                Instr::Join(r, _, t, f) => {
                    if subcomponents[*t] == subcomponents[*f] {
                        *sc = subcomponents[*t];
                    } else {
                        *sc = ComplexPhase::Any;
                    }
                    subcomponents[*r] = *sc;
                }
                Instr::Powf(r, b, e) => {
                    if settings.powf_real
                        && subcomponents[*b] == ComplexPhase::Real
                        && subcomponents[*e] == ComplexPhase::Real
                    {
                        *sc = ComplexPhase::Real;
                    } else {
                        *sc = ComplexPhase::Any;
                    }
                    subcomponents[*r] = *sc;
                }
                Instr::ExternalFun(r, ..) => {
                    *sc = ComplexPhase::Any;
                    subcomponents[*r] = *sc;
                }
                Instr::IfElse(..) | Instr::Goto(..) | Instr::Label(..) => {
                    *sc = ComplexPhase::Any;
                }
            }
        }

        if settings.verbose {
            info!(
                "Changed {} mul ops and {} div ops from complex to double",
                mul_components, div_components
            );
        }

        Ok(())
    }
}

impl<T: Default + Clone + Eq + Hash> ExpressionEvaluator<T> {
    /// Merge evaluator `other` into `self`. The parameters must be the same, and
    /// the outputs will be concatenated.
    ///
    /// The optional `cpe_rounds` parameter can be used to limit the number of common
    /// pair elimination rounds after the merge.
    pub fn merge(&mut self, mut other: Self, cpe_rounds: Option<usize>) -> Result<(), String> {
        if self.param_count != other.param_count {
            return Err(format!(
                "Parameter count is different: {} vs {}",
                self.param_count, other.param_count
            ));
        }
        if self.external_fns != other.external_fns {
            return Err(format!(
                "External functions do not match: {:?} vs {:?}",
                self.external_fns, other.external_fns
            ));
        }

        let mut constants = HashMap::default();

        for (i, c) in self.stack[self.param_count..self.reserved_indices]
            .iter()
            .enumerate()
        {
            constants.insert(c.clone(), i);
        }

        let old_len = self.stack.len() - self.reserved_indices;

        self.stack.truncate(self.reserved_indices);

        for c in &other.stack[self.param_count..other.reserved_indices] {
            if constants.get(c).is_none() {
                let i = constants.len();
                constants.insert(c.clone(), i);
                self.stack.push(c.clone());
            }
        }

        let new_reserved_indices = self.stack.len();
        let mut delta = new_reserved_indices - self.reserved_indices;

        // shift stack indices
        if delta > 0 {
            for (i, _) in &mut self.instructions {
                match i {
                    Instr::Add(r, a) | Instr::Mul(r, a) | Instr::ExternalFun(r, _, a) => {
                        *r += delta;
                        for aa in a {
                            if *aa >= self.reserved_indices {
                                *aa += delta;
                            }
                        }
                    }
                    Instr::Pow(r, b, _) | Instr::BuiltinFun(r, _, b) => {
                        *r += delta;
                        if *b >= self.reserved_indices {
                            *b += delta;
                        }
                    }
                    Instr::Powf(r, b, e) => {
                        *r += delta;
                        if *b >= self.reserved_indices {
                            *b += delta;
                        }
                        if *e >= self.reserved_indices {
                            *e += delta;
                        }
                    }
                    Instr::IfElse(c, _) => {
                        if *c >= self.reserved_indices {
                            *c += delta;
                        }
                    }
                    Instr::Join(r, c, t, f) => {
                        *r += delta;
                        if *c >= self.reserved_indices {
                            *c += delta;
                        }
                        if *t >= self.reserved_indices {
                            *t += delta;
                        }
                        if *f >= self.reserved_indices {
                            *f += delta;
                        }
                    }
                    Instr::Goto(..) | Instr::Label(..) => {}
                }
            }

            for x in &mut self.result_indices {
                if *x >= self.reserved_indices {
                    *x += delta;
                }
            }
        }

        delta = old_len + new_reserved_indices - other.reserved_indices;
        for (i, _) in &mut other.instructions {
            match i {
                Instr::Add(r, a) | Instr::Mul(r, a) | Instr::ExternalFun(r, _, a) => {
                    *r += delta;
                    for aa in a {
                        if *aa >= other.reserved_indices {
                            *aa += delta;
                        } else if *aa >= other.param_count {
                            *aa = self.param_count + constants[&other.stack[*aa]];
                        }
                    }
                }
                Instr::Pow(r, b, _) | Instr::BuiltinFun(r, _, b) => {
                    *r += delta;
                    if *b >= other.reserved_indices {
                        *b += delta;
                    } else if *b >= other.param_count {
                        *b = self.param_count + constants[&other.stack[*b]];
                    }
                }
                Instr::Powf(r, b, e) => {
                    *r += delta;
                    if *b >= other.reserved_indices {
                        *b += delta;
                    } else if *b >= other.param_count {
                        *b = self.param_count + constants[&other.stack[*b]];
                    }
                    if *e >= other.reserved_indices {
                        *e += delta;
                    } else if *e >= other.param_count {
                        *e = self.param_count + constants[&other.stack[*e]];
                    }
                }
                Instr::IfElse(c, l) => {
                    if *c >= other.reserved_indices {
                        *c += delta;
                    } else if *c >= other.param_count {
                        *c = self.param_count + constants[&other.stack[*c]];
                    }

                    l.0 += self.instructions.len();
                }
                Instr::Join(r, c, t, f) => {
                    *r += delta;
                    if *c >= other.reserved_indices {
                        *c += delta;
                    } else if *c >= other.param_count {
                        *c = self.param_count + constants[&other.stack[*c]];
                    }
                    if *t >= other.reserved_indices {
                        *t += delta;
                    } else if *t >= other.param_count {
                        *t = self.param_count + constants[&other.stack[*t]];
                    }
                    if *f >= other.reserved_indices {
                        *f += delta;
                    } else if *f >= other.param_count {
                        *f = self.param_count + constants[&other.stack[*f]];
                    }
                }
                Instr::Goto(l) | Instr::Label(l) => {
                    l.0 += self.instructions.len();
                }
            }
        }

        for x in &mut other.result_indices {
            if *x >= other.reserved_indices {
                *x += delta;
            } else if *x >= other.param_count {
                *x = self.param_count + constants[&other.stack[*x]];
            }
        }

        self.instructions.append(&mut other.instructions);
        self.result_indices.append(&mut other.result_indices);
        self.reserved_indices = new_reserved_indices;

        self.undo_stack_optimization();

        loop {
            if self.settings.abort_level > 0 || self.remove_common_instructions() == 0 {
                self.settings.abort_level = 0;
                break;
            }
        }

        for _ in 0..cpe_rounds.unwrap_or(usize::MAX) {
            if self.settings.abort_level > 0 || self.remove_common_pairs() == 0 {
                self.settings.abort_level = 0;
                break;
            }
        }

        self.optimize_stack();

        Ok(())
    }
}

impl<T> ExpressionEvaluator<T> {
    pub fn optimize_stack(&mut self) {
        let mut last_use: Vec<usize> = vec![0; self.stack.len()];

        for (i, (x, _)) in self.instructions.iter().enumerate() {
            match x {
                Instr::Add(_, a) | Instr::Mul(_, a) | Instr::ExternalFun(_, _, a) => {
                    for v in a {
                        last_use[*v] = i;
                    }
                }
                Instr::Pow(_, b, _) | Instr::BuiltinFun(_, _, b) => {
                    last_use[*b] = i;
                }
                Instr::Powf(_, a, b) => {
                    last_use[*a] = i;
                    last_use[*b] = i;
                }
                Instr::Join(_, c, a, b) => {
                    last_use[*c] = i;
                    last_use[*a] = i;
                    last_use[*b] = i;
                }
                Instr::IfElse(c, _) => {
                    last_use[*c] = i;
                }
                Instr::Goto(..) | Instr::Label(..) => {}
            };
        }

        // prevent init slots from being overwritten
        for i in 0..self.reserved_indices {
            last_use[i] = self.instructions.len();
        }

        // prevent the output slots from being overwritten
        for i in &self.result_indices {
            last_use[*i] = self.instructions.len();
        }

        let mut rename_map: Vec<_> = (0..self.stack.len()).collect(); // identity map

        let mut free_indices = BinaryHeap::<Reverse<(usize, usize)>>::new();

        let mut max_reg = self.reserved_indices;
        for (i, (x, _)) in self.instructions.iter_mut().enumerate() {
            let cur_reg = match x {
                Instr::Add(r, _)
                | Instr::Mul(r, _)
                | Instr::Pow(r, _, _)
                | Instr::Powf(r, _, _)
                | Instr::BuiltinFun(r, _, _)
                | Instr::ExternalFun(r, _, _)
                | Instr::Join(r, _, _, _) => *r,
                Instr::IfElse(c, _) => {
                    *c = rename_map[*c];
                    continue;
                }
                Instr::Goto(..) | Instr::Label(..) => continue,
            };

            let new_reg = if let Some(Reverse((last_pos, _))) = free_indices.peek()
                 // <= is ok because we store intermediate results in temp values
                && *last_pos <= i
            {
                free_indices.pop().unwrap().0.1
            } else {
                max_reg += 1;
                max_reg - 1
            };

            free_indices.push(Reverse((last_use[cur_reg], new_reg)));
            rename_map[cur_reg] = new_reg;

            match x {
                Instr::Add(r, a) | Instr::Mul(r, a) | Instr::ExternalFun(r, _, a) => {
                    *r = new_reg;
                    for v in a {
                        *v = rename_map[*v];
                    }
                }
                Instr::Pow(r, b, _) | Instr::BuiltinFun(r, _, b) => {
                    *r = new_reg;
                    *b = rename_map[*b];
                }
                Instr::Powf(r, a, b) => {
                    *r = new_reg;
                    *a = rename_map[*a];
                    *b = rename_map[*b];
                }
                Instr::Join(r, c, a, b) => {
                    *r = new_reg;
                    *c = rename_map[*c];
                    *a = rename_map[*a];
                    *b = rename_map[*b];
                }
                Instr::IfElse(_, _) | Instr::Goto(..) | Instr::Label(..) => {
                    unreachable!()
                }
            };
        }

        self.stack.truncate(max_reg + 1);

        for i in &mut self.result_indices {
            *i = rename_map[*i];
        }
    }
}

impl<T: Default> ExpressionEvaluator<T> {
    pub(super) fn undo_stack_optimization(&mut self) {
        // undo the stack optimization
        let mut unfold = HashMap::default();
        for (index, (i, _c)) in &mut self.instructions.iter_mut().enumerate() {
            match i {
                Instr::Add(r, a) | Instr::Mul(r, a) | Instr::ExternalFun(r, _, a) => {
                    for aa in a {
                        if *aa >= self.reserved_indices {
                            *aa = unfold[aa];
                        }
                    }

                    unfold.insert(*r, index + self.reserved_indices);
                    *r = index + self.reserved_indices;
                }
                Instr::Pow(r, b, _) | Instr::BuiltinFun(r, _, b) => {
                    if *b >= self.reserved_indices {
                        *b = unfold[b];
                    }
                    unfold.insert(*r, index + self.reserved_indices);
                    *r = index + self.reserved_indices;
                }
                Instr::Powf(r, b, e) => {
                    if *b >= self.reserved_indices {
                        *b = unfold[b];
                    }
                    if *e >= self.reserved_indices {
                        *e = unfold[e];
                    }
                    unfold.insert(*r, index + self.reserved_indices);
                    *r = index + self.reserved_indices;
                }
                Instr::IfElse(r, _) => {
                    if *r >= self.reserved_indices {
                        *r = unfold[r];
                    }
                }
                Instr::Join(r, c, t, f) => {
                    if *c >= self.reserved_indices {
                        *c = unfold[c];
                    }
                    if *t >= self.reserved_indices {
                        *t = unfold[t];
                    }
                    if *f >= self.reserved_indices {
                        *f = unfold[f];
                    }
                    unfold.insert(*r, index + self.reserved_indices);
                    *r = index + self.reserved_indices;
                }
                Instr::Goto(..) | Instr::Label(..) => {}
            }
        }

        for i in &mut self.result_indices {
            if *i >= self.reserved_indices {
                *i = unfold[i];
            }
        }

        for _ in 0..self.instructions.len() {
            self.stack.push(T::default());
        }
    }
}
