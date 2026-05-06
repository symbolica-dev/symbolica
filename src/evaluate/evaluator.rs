use super::*;

#[derive(Clone)]
pub struct ExpressionEvaluator<T> {
    pub(super) stack: Vec<T>,
    pub(super) param_count: usize,
    pub(super) reserved_indices: usize,
    pub(super) instructions: Vec<(Instr, ComplexPhase)>,
    pub(super) result_indices: Vec<usize>,
    pub(super) external_fns: Vec<ExternalFunctionContainer<T>>,
    pub(super) settings: OptimizationSettings,
}

#[cfg(feature = "serde")]
impl<T: serde::Serialize> serde::Serialize for ExpressionEvaluator<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (
            &self.stack,
            &self.param_count,
            &self.reserved_indices,
            &self.instructions,
            &self.result_indices,
            &self.external_fns,
            &self.settings,
        )
            .serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: serde::Deserialize<'de> + EvaluationDomain> serde::Deserialize<'de>
    for ExpressionEvaluator<T>
{
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (
            stack,
            param_count,
            reserved_indices,
            instructions,
            result_indices,
            external_fns,
            settings,
        ): (
            Vec<T>,
            usize,
            usize,
            Vec<(Instr, ComplexPhase)>,
            Vec<usize>,
            Vec<ExternalFunctionContainer<T>>,
            OptimizationSettings,
        ) = serde::Deserialize::deserialize(deserializer)?;

        Ok(Self {
            stack,
            param_count,
            reserved_indices,
            instructions,
            result_indices,
            external_fns,
            settings,
        })
    }
}

#[cfg(feature = "bincode")]
impl<T: bincode::Encode> bincode::Encode for ExpressionEvaluator<T> {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> core::result::Result<(), bincode::error::EncodeError> {
        bincode::Encode::encode(&self.stack, encoder)?;
        bincode::Encode::encode(&self.param_count, encoder)?;
        bincode::Encode::encode(&self.reserved_indices, encoder)?;
        bincode::Encode::encode(&self.instructions, encoder)?;
        bincode::Encode::encode(&self.result_indices, encoder)?;
        bincode::Encode::encode(&self.external_fns, encoder)?;
        bincode::Encode::encode(&self.settings, encoder)
    }
}

#[cfg(feature = "bincode")]
impl<Context, T: bincode::Decode<Context> + EvaluationDomain> bincode::Decode<Context>
    for ExpressionEvaluator<T>
{
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        Ok(Self {
            stack: bincode::Decode::decode(decoder)?,
            param_count: bincode::Decode::decode(decoder)?,
            reserved_indices: bincode::Decode::decode(decoder)?,
            instructions: bincode::Decode::decode(decoder)?,
            result_indices: bincode::Decode::decode(decoder)?,
            external_fns: bincode::Decode::decode(decoder)?,
            settings: bincode::Decode::decode(decoder)?,
        })
    }
}

#[cfg(feature = "bincode")]
impl<'de, Context, T: bincode::BorrowDecode<'de, Context> + EvaluationDomain>
    bincode::BorrowDecode<'de, Context> for ExpressionEvaluator<T>
{
    fn borrow_decode<D: bincode::de::BorrowDecoder<'de, Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        Ok(Self {
            stack: bincode::BorrowDecode::borrow_decode(decoder)?,
            param_count: bincode::BorrowDecode::borrow_decode(decoder)?,
            reserved_indices: bincode::BorrowDecode::borrow_decode(decoder)?,
            instructions: bincode::BorrowDecode::borrow_decode(decoder)?,
            result_indices: bincode::BorrowDecode::borrow_decode(decoder)?,
            external_fns: bincode::BorrowDecode::borrow_decode(decoder)?,
            settings: bincode::BorrowDecode::borrow_decode(decoder)?,
        })
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for ExpressionEvaluator<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExpressionEvaluator")
            .field("stack", &self.stack)
            .field("param_count", &self.param_count)
            .field("reserved_indices", &self.reserved_indices)
            .field("instructions", &self.instructions)
            .field("result_indices", &self.result_indices)
            .field("external_fns", &self.external_fns)
            .finish()
    }
}

impl<T: SingleFloat> ExpressionEvaluator<Complex<T>> {
    /// Check if the expression evaluator is real, i.e., all coefficients are real.
    pub fn is_real(&self) -> bool {
        self.stack.iter().all(|x| x.is_real())
    }
}

impl<T: Real> ExpressionEvaluator<T> {
    /// Evaluate the expression evaluator which yields a single result.
    pub fn evaluate_single(&mut self, params: &[T]) -> T {
        if self.result_indices.len() != 1 {
            panic!(
                "Evaluator does not return a single result but {} results",
                self.result_indices.len()
            );
        }

        let mut res = T::new_zero();
        self.evaluate(params, std::slice::from_mut(&mut res));
        res
    }

    /// Evaluate the expression evaluator and write the results in `out`.
    #[inline]
    pub fn evaluate(&mut self, params: &[T], out: &mut [T]) {
        self.evaluate_impl(params, out);
    }

    #[cold]
    #[inline(never)]
    fn evaluate_impl_no_ops(
        stack: &mut [T],
        instr: &Instr,
        external_fns: &mut [ExternalFunctionContainer<T>],
    ) -> Option<usize> {
        match instr {
            Instr::Powf(r, b, e) => {
                stack[*r] = stack[*b].powf(&stack[*e]);
            }
            Instr::BuiltinFun(r, s, arg) => match s.get_id() {
                Symbol::EXP_ID => stack[*r] = stack[*arg].exp(),
                Symbol::LOG_ID => stack[*r] = stack[*arg].log(),
                Symbol::SIN_ID => stack[*r] = stack[*arg].sin(),
                Symbol::COS_ID => stack[*r] = stack[*arg].cos(),
                Symbol::SQRT_ID => stack[*r] = stack[*arg].sqrt(),
                Symbol::ABS_ID => stack[*r] = stack[*arg].norm(),
                Symbol::CONJ_ID => stack[*r] = stack[*arg].conj(),
                _ => unreachable!(),
            },
            Instr::ExternalFun(r, s, args) => {
                let external = &mut external_fns[*s];
                let Some(f) = external.imp.as_ref() else {
                    panic!(
                        "External function '{external}' does not have an implementation for {}",
                        std::any::type_name::<T>()
                    );
                };

                if external.cache.len() < args.len() {
                    external.cache.resize(args.len(), T::new_zero());
                }

                for (dst, src) in external.cache.iter_mut().zip(args) {
                    dst.set_from(&stack[*src]);
                }

                stack[*r] = (f)(&external.cache[..args.len()]);
            }
            Instr::IfElse(n, label) => {
                // jump to else block
                if stack[*n].is_fully_zero() {
                    return Some(label.0);
                }
            }
            Instr::Goto(label) => {
                return Some(label.0);
            }
            Instr::Label(_) => {}
            Instr::Join(r, c, a, b) => {
                if !stack[*c].is_fully_zero() {
                    stack[*r] = stack[*a].clone();
                } else {
                    stack[*r] = stack[*b].clone();
                }
            }
            Instr::Add(..) | Instr::Mul(..) | Instr::Pow(..) => {
                unreachable!()
            }
        }

        None
    }

    /// Evaluate the expression evaluator and write the results in `out`.
    fn evaluate_impl(&mut self, params: &[T], out: &mut [T]) {
        if self.param_count != params.len() {
            panic!(
                "Parameter count mismatch: expected {}, got {}",
                self.param_count,
                params.len()
            );
        }

        for (t, p) in self.stack.iter_mut().zip(params) {
            t.set_from(p);
        }

        let mut tmp = T::new_zero();
        let mut i = 0;
        let (stack, external_fns) = (&mut self.stack, &mut self.external_fns);
        while i < self.instructions.len() {
            let (instr, _) = unsafe { &self.instructions.get_unchecked(i) };
            match instr {
                Instr::Add(r, v) => unsafe {
                    match v.len() {
                        2 => {
                            tmp.set_from(stack.get_unchecked(*v.get_unchecked(0)));
                            tmp += stack.get_unchecked(*v.get_unchecked(1));
                        }
                        3 => {
                            tmp.set_from(stack.get_unchecked(*v.get_unchecked(0)));
                            tmp += stack.get_unchecked(*v.get_unchecked(1));
                            tmp += stack.get_unchecked(*v.get_unchecked(2));
                        }
                        _ => {
                            tmp.set_from(stack.get_unchecked(*v.get_unchecked(0)));
                            for x in v.get_unchecked(1..) {
                                tmp += stack.get_unchecked(*x);
                            }
                        }
                    }

                    std::mem::swap(stack.get_unchecked_mut(*r), &mut tmp);
                },
                Instr::Mul(r, v) => unsafe {
                    match v.len() {
                        2 => {
                            tmp.set_from(stack.get_unchecked(*v.get_unchecked(0)));
                            tmp *= stack.get_unchecked(*v.get_unchecked(1));
                        }
                        3 => {
                            tmp.set_from(stack.get_unchecked(*v.get_unchecked(0)));
                            tmp *= stack.get_unchecked(*v.get_unchecked(1));
                            tmp *= stack.get_unchecked(*v.get_unchecked(2));
                        }
                        _ => {
                            tmp.set_from(stack.get_unchecked(*v.get_unchecked(0)));
                            for x in v.get_unchecked(1..) {
                                tmp *= stack.get_unchecked(*x);
                            }
                        }
                    }

                    std::mem::swap(stack.get_unchecked_mut(*r), &mut tmp);
                },
                Instr::Pow(r, b, e) => {
                    if *e == -1 {
                        stack[*r] = stack[*b].inv();
                    } else if *e >= 0 {
                        stack[*r] = stack[*b].pow(*e as u64);
                    } else {
                        stack[*r] = stack[*b].pow(e.unsigned_abs()).inv();
                    }
                }
                _ => {
                    if let Some(idx) = Self::evaluate_impl_no_ops(stack, instr, external_fns) {
                        i = idx;
                        continue;
                    }
                }
            }

            i += 1;
        }

        for (o, i) in out.iter_mut().zip(&self.result_indices) {
            o.set_from(&stack[*i]);
        }
    }
}

impl ExpressionEvaluator<Complex<Rational>> {}

impl<T: Default> ExpressionEvaluator<T> {
    /// Map the coefficients to a different type.
    pub fn map_coeff_with_prec<T2: EvaluationDomain, F: Fn(&T) -> T2>(
        self,
        f: &F,
        binary_prec: u32,
    ) -> ExpressionEvaluator<T2> {
        let mut stack: Vec<_> = self.stack.iter().map(f).collect();

        let mut external_fns = self
            .external_fns
            .iter()
            .map(|x| x.map())
            .collect::<Vec<_>>();
        for external in &mut external_fns {
            if let Some(i) = external.constant_index {
                let Some(eval) = external.symbol.get_evaluation_info() else {
                    panic!(
                        "Symbol '{}' does not have evaluation info",
                        external.symbol.get_name()
                    );
                };
                let tags = external.tag_views();
                let c = eval.evaluate_constant(&tags, binary_prec).unwrap();
                stack[self.param_count + i] = T2::try_from_complex_float(c).unwrap();
            }
        }

        ExpressionEvaluator {
            stack,
            param_count: self.param_count,
            reserved_indices: self.reserved_indices,
            instructions: self.instructions,
            result_indices: self.result_indices,
            external_fns,
            settings: self.settings.clone(),
        }
    }

    /// Map the coefficients to a different type.
    pub fn map_coeff<T2: EvaluationDomain, F: Fn(&T) -> T2>(
        self,
        f: &F,
    ) -> ExpressionEvaluator<T2> {
        if self.external_fns.iter().any(|x| x.constant_index.is_some())
            && T2::FIXED_PRECISION.is_none()
        {
            panic!(
                "Cannot evaluate constants since the target precision is not specified. Use map_coeff_with_prec."
            );
        }

        self.map_coeff_with_prec(f, T2::FIXED_PRECISION.unwrap_or(53))
    }

    #[allow(dead_code)]
    pub(crate) fn set_coeff<T2: Default + Clone + EvaluationDomain>(
        self,
        coeffs: &[T2],
    ) -> ExpressionEvaluator<T2> {
        if coeffs.len() != self.reserved_indices - self.param_count {
            panic!(
                "Wrong number of coefficients: {} vs {}",
                coeffs.len(),
                self.reserved_indices - self.param_count
            )
        }

        let mut stack = vec![T2::default(); self.stack.len()];
        for (s, coeff) in stack.iter_mut().skip(self.param_count).zip(coeffs) {
            *s = coeff.clone();
        }

        ExpressionEvaluator {
            stack,
            param_count: self.param_count,
            reserved_indices: self.reserved_indices,
            instructions: self.instructions,
            result_indices: self.result_indices,
            external_fns: self.external_fns.into_iter().map(|x| x.map()).collect(),
            settings: self.settings,
        }
    }

    pub fn get_input_len(&self) -> usize {
        self.param_count
    }

    pub fn get_output_len(&self) -> usize {
        self.result_indices.len()
    }

    pub fn get_constants(&self) -> &[T] {
        &self.stack[self.param_count..self.reserved_indices]
    }

    /// Return the total number of additions and multiplications.
    pub fn count_operations(&self) -> (usize, usize) {
        let mut add_count = 0;
        let mut mul_count = 0;

        for (instr, _) in &self.instructions {
            match instr {
                Instr::Add(_, s) => add_count += s.len() - 1,
                Instr::Mul(_, s) => mul_count += s.len() - 1,
                _ => {}
            }
        }

        (add_count, mul_count)
    }

    /// Remove common instructions and return the number of removed instructions.
    pub(super) fn remove_common_instructions(&mut self) -> usize {
        #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
        enum CSE<'a> {
            Add(&'a [usize]),
            Mul(&'a [usize]),
            Pow(usize, i64),
            Powf(usize, usize),
            BuiltinFun(u32, usize),
            ExternalFun(u32, &'a [usize]),
        }

        let mut common_instr = HashMap::with_capacity(self.instructions.len());
        let mut new_instr = Vec::with_capacity(self.instructions.len());
        let mut i = 0;

        let mut rename_map: Vec<_> = (0..self.reserved_indices).collect();
        let mut removed = 0;

        let mut dag_nodes = vec![0]; // store index to parent node
        let mut current_node = 0;

        for (instr, phase) in &self.instructions {
            let new_pos = new_instr.len() + self.reserved_indices;

            let key = match &self.instructions[i].0 {
                Instr::Add(_, a) => Some(CSE::Add(a)),
                Instr::Mul(_, a) => Some(CSE::Mul(a)),
                Instr::Pow(_, b, e) => Some(CSE::Pow(*b, *e)),
                Instr::Powf(_, b, e) => Some(CSE::Powf(*b, *e)),
                Instr::BuiltinFun(_, s, a) => Some(CSE::BuiltinFun(s.get_id(), *a)),
                Instr::ExternalFun(_, s, a) => Some(CSE::ExternalFun(*s as u32, a)),
                _ => None,
            };

            if let Some(key) = key {
                match common_instr.entry(key) {
                    Entry::Occupied(mut o) => {
                        let (old_pos, branch) = o.get_mut();

                        let mut cur = current_node;
                        while cur > *branch {
                            cur = dag_nodes[cur];
                        }

                        if cur == *branch {
                            removed += 1;
                            rename_map.push(*old_pos);
                            i += 1;
                            continue;
                        } else {
                            // the previous occurrence was in a non-parent branch
                            // that cannot be reused, so treat this occurrence
                            // as the first
                            *old_pos = new_pos;
                            *branch = current_node;
                        }
                    }
                    Entry::Vacant(v) => {
                        v.insert((new_pos, current_node));
                    }
                }
            }

            let mut s = instr.clone();

            match &mut s {
                Instr::Add(p, a) | Instr::Mul(p, a) => {
                    let mut last = 0;
                    let mut sort = false;
                    for x in &mut *a {
                        *x = rename_map[*x];
                        if *x < last {
                            sort = true;
                        } else {
                            last = *x;
                        }
                    }

                    if sort {
                        a.sort_unstable();
                    }
                    *p = new_pos;
                }
                Instr::Pow(p, b, _) | Instr::BuiltinFun(p, _, b) => {
                    *b = rename_map[*b];
                    *p = new_pos;
                }
                Instr::Powf(p, a, b) => {
                    *a = rename_map[*a];
                    *b = rename_map[*b];
                    *p = new_pos;
                }
                Instr::ExternalFun(p, _, a) => {
                    *p = new_pos;
                    for x in a {
                        *x = rename_map[*x];
                    }
                }
                Instr::Join(p, a, b, c) => {
                    current_node = dag_nodes[current_node];
                    *a = rename_map[*a];
                    *b = rename_map[*b];
                    *c = rename_map[*c];
                    *p = new_pos;
                }
                Instr::IfElse(c, _) => {
                    dag_nodes.push(current_node); // enter if block
                    current_node = dag_nodes.len() - 1;
                    *c = rename_map[*c];
                }
                Instr::Goto(_) => {
                    let parent = dag_nodes[current_node];
                    dag_nodes.push(parent); // enter else block (included goto and labels)
                    current_node = dag_nodes.len() - 1;
                }
                _ => {}
            }

            new_instr.push((s, *phase));
            rename_map.push(new_pos);
            i += 1;
        }

        for x in &mut self.result_indices {
            *x = rename_map[*x];
        }

        self.instructions = new_instr;

        self.fix_labels();

        removed
    }

    /// Set the labels to the their instruction position.
    pub(super) fn fix_labels(&mut self) {
        let mut label_map: HashMap<usize, usize> = HashMap::default();
        for (i, (x, _)) in self.instructions.iter_mut().enumerate().rev() {
            match x {
                Instr::Label(l) => {
                    label_map.insert(l.0, i);
                    l.0 = i;
                }
                Instr::Goto(l) => {
                    l.0 = label_map[&l.0];
                }
                Instr::IfElse(_, l) => {
                    l.0 = label_map[&l.0];
                }
                _ => {}
            }
        }
    }

    /// Remove common pairs of instructions. Assumes that the arguments
    /// of the instructions are sorted.
    pub(super) fn remove_common_pairs(&mut self) -> usize {
        let mut affected_lines = vec![false; self.instructions.len()];

        // store the global branch a line belongs to
        let mut branch_id = vec![0; self.instructions.len()];
        let mut dag_nodes = vec![0]; // store index to parent node
        let mut current_node = 0;

        let mut common_ops_simple: HashMap<_, u32> = HashMap::default();

        if self.instructions.len() > u32::MAX as usize / 2 {
            // the extension is easy, but it will cost more memory.
            // will only be added when a user runs into it.
            error!(
                "Too many instructions to find common pairs. Reach out to Symbolica devs to extend the limit."
            );
            return 0;
        }

        for (p, (i, _)) in self.instructions.iter().enumerate() {
            if common_ops_simple.len() > self.settings.max_common_pair_cache_entries {
                break;
            }

            if p % 10000 == 0 {
                if let Some(abort_check) = &self.settings.abort_check {
                    if abort_check() {
                        self.settings.abort_level = 1;
                        break;
                    }
                }
            }

            match i {
                Instr::Add(_, a) | Instr::Mul(_, a) => {
                    let is_add = matches!(i, Instr::Add(_, _));
                    'add_loop: for (li, l) in a.iter().enumerate() {
                        for r in &a[li + 1..] {
                            let mut key = (*l as u64) << 32 | (*r as u64) << 1;
                            if !is_add {
                                key |= 1;
                            }

                            if common_ops_simple.len() > self.settings.max_common_pair_cache_entries
                            {
                                break 'add_loop;
                            }

                            common_ops_simple
                                .entry(key)
                                .and_modify(|x| *x += 1)
                                .or_insert(1);
                        }
                    }
                }
                Instr::IfElse(_, _) => {
                    branch_id[p] = current_node;
                    dag_nodes.push(current_node); // enter if block
                    current_node = dag_nodes.len() - 1;
                    continue;
                }
                Instr::Goto(_) => {
                    let parent = dag_nodes[current_node];
                    dag_nodes.push(parent); // enter else block (included goto and labels)
                    current_node = dag_nodes.len() - 1;
                }
                Instr::Join(..) => {
                    current_node = dag_nodes[current_node];
                }
                _ => {}
            }
            branch_id[p] = current_node;
        }

        common_ops_simple.retain(|_, v| *v > 1);

        let mut common_ops_2: HashMap<_, Vec<usize>> = HashMap::default();

        for (p, (i, _)) in self.instructions.iter().enumerate() {
            match i {
                Instr::Add(_, a) | Instr::Mul(_, a) => {
                    let is_add = matches!(i, Instr::Add(_, _));
                    for (li, l) in a.iter().enumerate() {
                        for r in &a[li + 1..] {
                            let mut key = (*l as u64) << 32 | (*r as u64) << 1;
                            if !is_add {
                                key |= 1;
                            }

                            if *common_ops_simple.get(&key).unwrap_or(&0) > 1 {
                                common_ops_2.entry(key).or_default().push(p);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        drop(common_ops_simple); // clear the memory

        if common_ops_2.is_empty() {
            return 0;
        }

        let mut to_remove: Vec<_> = common_ops_2.into_iter().collect();
        to_remove.retain_mut(|(_, v)| {
            let keep = v.len() > 1;
            v.dedup();
            keep
        });

        // sort in other direction since we pop
        to_remove.sort_by(|a, b| a.1.len().cmp(&b.1.len()).then_with(|| a.cmp(b)));

        let total_remove = to_remove.len();

        let old_len = self.instructions.len();

        let mut new_symb_branch = vec![];

        while let Some((key, lines)) = to_remove.pop() {
            let l = (key >> 32) as usize;
            let r = ((key >> 1) & 0x7FFFFFFF) as usize;
            let is_add = key & 1 == 0;

            if lines.iter().any(|x| affected_lines[*x]) {
                continue;
            }

            let new_idx = self.stack.len();
            let new_op = if is_add {
                Instr::Add(new_idx, vec![l, r])
            } else {
                Instr::Mul(new_idx, vec![l, r])
            };

            self.stack.push(T::default());
            self.instructions.push((new_op, ComplexPhase::Any));

            let mut branch = branch_id[lines[0]];
            for &line in &lines {
                affected_lines[line] = true;

                let mut new_branch = branch_id[line];
                // find common root
                while branch != new_branch {
                    if branch > new_branch {
                        branch = dag_nodes[branch];
                    } else {
                        new_branch = dag_nodes[new_branch];
                    }
                }

                if let Instr::Add(_, a) | Instr::Mul(_, a) = &mut self.instructions[line].0 {
                    if l == r {
                        let count = a.iter().filter(|x| **x == l).count();
                        let pairs = count / 2;
                        if pairs > 0 {
                            a.retain(|x| *x != l);

                            if count % 2 == 1 {
                                a.push(l);
                            }

                            a.extend(std::iter::repeat_n(new_idx, pairs));
                            a.sort_unstable();
                        }
                    } else {
                        let mut idx1_count = 0;
                        let mut idx2_count = 0;
                        for v in &*a {
                            if *v == l {
                                idx1_count += 1;
                            }
                            if *v == r {
                                idx2_count += 1;
                            }
                        }

                        let pair_count = idx1_count.min(idx2_count);

                        if pair_count > 0 {
                            a.retain(|x| *x != l && *x != r);

                            // add back removed indices in cases such as idx1*idx2*idx2
                            if idx1_count > pair_count {
                                a.extend(std::iter::repeat_n(l, idx1_count - pair_count));
                            }
                            if idx2_count > pair_count {
                                a.extend(std::iter::repeat_n(r, idx2_count - pair_count));
                            }

                            a.extend(std::iter::repeat_n(new_idx, pair_count));
                            a.sort_unstable();
                        }
                    }
                }
            }

            new_symb_branch.push((lines[0], branch));
        }

        // detect the earliest point and latest point for an instruction placement
        // earliest point: after last dependency
        // latest point: before first usage in the correct usage zone
        let mut placement_bounds = vec![];
        for ((i, _), (first_usage, branch)) in
            self.instructions.drain(old_len..).zip(new_symb_branch)
        {
            let deps = match &i {
                Instr::BuiltinFun(_, _, a) => std::slice::from_ref(a),
                Instr::Add(_, a) | Instr::Mul(_, a) | Instr::ExternalFun(_, _, a) => a.as_slice(),
                _ => unreachable!(),
            };

            let mut last_dep = deps[0];
            for v in deps {
                last_dep = last_dep.max(*v);
            }

            let ins = if last_dep < self.reserved_indices {
                0
            } else {
                last_dep + 1 - self.reserved_indices
            };

            let mut latest_pos = ins;
            for j in (ins..first_usage + 1).rev() {
                if branch_id[j] == branch {
                    latest_pos = j;
                    break;
                }
            }

            placement_bounds.push((ins, latest_pos, i));
        }

        placement_bounds.sort_by_key(|x| x.1);

        let mut new_instr = vec![];
        let mut i = 0;
        let mut j = 0;

        let mut sub_rename = HashMap::default();
        let mut rename_map: Vec<_> = (0..self.reserved_indices).collect();

        macro_rules! rename {
            ($i:expr) => {
                if $i >= self.reserved_indices + self.instructions.len() {
                    sub_rename[&$i]
                } else {
                    rename_map[$i]
                }
            };
        }

        while i < self.instructions.len() {
            let new_pos = new_instr.len() + self.reserved_indices;

            if j < placement_bounds.len() && i == placement_bounds[j].1 {
                let (o, a) = match &placement_bounds[j].2 {
                    Instr::Add(o, a) => (*o, a.as_slice()),
                    Instr::Mul(o, a) => (*o, a.as_slice()),
                    _ => unreachable!(),
                };

                let mut new_a = a.iter().map(|x| rename!(*x)).collect::<Vec<_>>();
                new_a.sort();

                match placement_bounds[j].2 {
                    Instr::Add(_, _) => {
                        new_instr.push((Instr::Add(new_pos, new_a), ComplexPhase::Any));
                    }
                    Instr::Mul(_, _) => {
                        new_instr.push((Instr::Mul(new_pos, new_a), ComplexPhase::Any));
                    }
                    _ => unreachable!(),
                }

                sub_rename.insert(o, new_pos);

                j += 1;
            } else {
                let (mut s, sc) = self.instructions[i].clone();

                match &mut s {
                    Instr::Add(p, a) | Instr::Mul(p, a) => {
                        for x in &mut *a {
                            *x = rename!(*x);
                        }
                        a.sort();

                        // remove assignments
                        if a.len() == 1 {
                            rename_map.push(a[0]);
                            i += 1;
                            continue;
                        }

                        *p = new_pos;
                    }
                    Instr::Pow(p, b, _) | Instr::BuiltinFun(p, _, b) => {
                        *b = rename!(*b);
                        *p = new_pos;
                    }
                    Instr::Powf(p, a, b) => {
                        *a = rename!(*a);
                        *b = rename!(*b);
                        *p = new_pos;
                    }
                    Instr::ExternalFun(p, _, a) => {
                        *p = new_pos;
                        for x in a {
                            *x = rename!(*x);
                        }
                    }
                    Instr::Join(p, a, b, c) => {
                        *a = rename!(*a);
                        *b = rename!(*b);
                        *c = rename!(*c);
                        *p = new_pos;
                    }
                    Instr::IfElse(c, _) => {
                        *c = rename!(*c);
                    }
                    Instr::Goto(_) | Instr::Label(_) => {}
                }

                new_instr.push((s, sc));
                rename_map.push(new_pos);
                i += 1;
            }
        }

        for x in &mut self.result_indices {
            *x = rename!(*x);
        }

        assert!(j == placement_bounds.len());

        self.instructions = new_instr;
        self.fix_labels();

        total_remove
    }
}
