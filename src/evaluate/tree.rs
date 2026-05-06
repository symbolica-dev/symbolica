use super::*;

#[derive(Debug, Clone)]
struct SplitExpression<T> {
    tree: Vec<Expression<T>>,
    subexpressions: Vec<Expression<T>>,
}

/// A tree representation of multiple expressions, including function definitions.
#[derive(Debug, Clone)]
pub struct EvalTree<T> {
    functions: Vec<(String, Vec<Indeterminate>, SplitExpression<T>)>,
    external_functions: Vec<ExternalFunctionContainer<T>>,
    expressions: SplitExpression<T>,
    param_count: usize,
}

fn register_constant_external_container<T>(
    external_functions: &mut Vec<ExternalFunctionContainer<T>>,
    symbol: Symbol,
    tags: Vec<Atom>,
    constants: &mut Vec<Complex<Rational>>,
) -> usize {
    let index = if let Some(index) = external_functions
        .iter()
        .position(|x| x.symbol == symbol && x.tags == tags)
    {
        index
    } else {
        external_functions.push(ExternalFunctionContainer::new(symbol, tags));
        external_functions.len() - 1
    };

    if let Some(constant_index) = external_functions[index].constant_index {
        return constant_index;
    }

    let constant_index = constants.len();
    constants.push(Complex::default());
    external_functions[index].constant_index = Some(constant_index);
    constant_index
}

impl<'a> AtomView<'a> {
    pub(crate) fn to_evaluator(
        expressions: &[Self],
        fn_map: &FunctionMap,
        params: &[Atom],
        settings: OptimizationSettings,
    ) -> Result<ExpressionEvaluator<Complex<Rational>>, String> {
        if settings.verbose {
            let mut cse = HashSet::default();
            let (mut n_add, mut n_mul) = (0, 0);
            for e in expressions {
                let (add, mul) = e.count_operations_with_subexpressions(&mut cse);
                n_add += add;
                n_mul += mul;
            }
            info!(
                "Initial ops: {} additions and {} multiplications",
                n_add, n_mul
            );
        }

        if settings.horner_iterations == 0 {
            return Self::linearize_multiple(expressions, fn_map, params, settings);
        }

        let v = match &settings.hot_start {
            Some(_) => {
                return Err(
                    "Hot start not supported before the deprecation of Expression".to_owned(),
                );
            }
            None => {
                // start with an occurence order Horner scheme
                let mut v = HashMap::default();

                for t in expressions {
                    t.count_indeterminates(true, &mut v);
                }

                let mut v: Vec<_> = v.into_iter().collect();
                v.retain(|(_, vv)| *vv > 1);
                v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                v.truncate(settings.max_horner_scheme_variables);
                v.into_iter()
                    .map(|(k, _)| Indeterminate::try_from(k.to_owned()).unwrap())
                    .collect::<Vec<_>>()
            }
        };

        let scheme = if settings.horner_iterations > 1 {
            Self::optimize_horner_scheme_multiple(expressions, &v, &settings)
        } else {
            v
        };

        let hornered_expressions = expressions
            .iter()
            .map(|x| x.horner_scheme(Some(&scheme), true))
            .collect::<Vec<_>>();

        if settings.horner_iterations == 1 && settings.verbose {
            let mut cse = HashSet::default();
            let (mut n_add, mut n_mul) = (0, 0);
            for e in expressions {
                let (add, mul) = e.count_operations_with_subexpressions(&mut cse);
                n_add += add;
                n_mul += mul;
            }
            info!(
                "Horner scheme ops: {} additions and {} multiplications",
                n_add, n_mul
            );
        }

        let mut f = fn_map.clone();
        for Expr { body, .. } in f.tagged_fn_map.values_mut() {
            *body = body.as_view().horner_scheme(Some(&scheme), true);
        }

        let mut e = Self::linearize_multiple(&hornered_expressions, fn_map, params, settings)?;

        drop(f);
        drop(hornered_expressions);

        loop {
            let r = e.remove_common_instructions();

            if r == 0 || e.settings.abort_level > 0 {
                e.settings.abort_level = 0;
                break;
            }

            if e.settings.verbose {
                let (add_count, mul_count) = e.count_operations();
                info!(
                    "Removed {} common instructions: {} + and {} ×",
                    r, add_count, mul_count
                );
            }
        }

        for _ in 0..e.settings.cpe_iterations.unwrap_or(usize::MAX) {
            let r = e.remove_common_pairs();
            if r == 0 || e.settings.abort_level > 0 {
                e.settings.abort_level = 0;
                break;
            }

            if e.settings.verbose {
                let (add_count, mul_count) = e.count_operations();
                info!(
                    "Removed {} common pairs: {} + and {} ×",
                    r, add_count, mul_count
                );
            }
        }

        e.optimize_stack();
        Ok(e)
    }

    pub fn optimize_horner_scheme_multiple(
        expressions: &[Self],
        vars: &[Indeterminate],
        settings: &OptimizationSettings,
    ) -> Vec<Indeterminate> {
        if vars.is_empty() {
            return vars.to_vec();
        }

        let horner: Vec<_> = expressions
            .iter()
            .map(|x| x.horner_scheme(Some(&vars), true))
            .collect();
        let mut subexpr = HashSet::default();
        let mut best_ops = (0, 0);
        for h in &horner {
            let ops = h
                .as_view()
                .count_operations_with_subexpressions(&mut subexpr);
            best_ops = (best_ops.0 + ops.0, best_ops.1 + ops.1);
        }

        if settings.verbose {
            info!(
                "Initial Horner scheme ops: {} additions and {} multiplications",
                best_ops.0, best_ops.1
            );
        }

        let best_mul = Arc::new(AtomicUsize::new(best_ops.1));
        let best_add = Arc::new(AtomicUsize::new(best_ops.0));
        let best_scheme = Arc::new(Mutex::new(vars.to_vec()));

        let n_iterations = settings.horner_iterations.max(1) - 1;

        let permutations = if vars.len() < 10
            && Integer::factorial(vars.len() as u32) <= settings.horner_iterations.max(1)
        {
            let v: Vec<_> = (0..vars.len()).collect();
            Some(unique_permutations(&v).1)
        } else {
            None
        };
        let p_ref = &permutations;

        let n_cores = if LicenseManager::is_licensed() {
            settings.n_cores
        } else {
            1
        }
        .min(n_iterations);

        std::thread::scope(|s| {
            let abort = Arc::new(AtomicBool::new(false));

            for i in 0..n_cores {
                let mut rng = MonteCarloRng::new(0, i);

                let mut cvars = vars.to_vec();
                let best_scheme = best_scheme.clone();
                let best_mul = best_mul.clone();
                let best_add = best_add.clone();
                let mut last_mul = usize::MAX;
                let mut last_add = usize::MAX;
                let abort = abort.clone();

                let mut op = move || {
                    for j in 0..n_iterations / n_cores {
                        if abort.load(Ordering::Relaxed) {
                            return;
                        }

                        if i == n_cores - 1
                            && let Some(a) = &settings.abort_check
                            && a()
                        {
                            abort.store(true, Ordering::Relaxed);

                            if settings.verbose {
                                info!(
                                    "Aborting Horner optimization at step {}/{}.",
                                    j,
                                    settings.horner_iterations / n_cores
                                );
                            }

                            return;
                        }

                        // try a random swap
                        let mut t1 = 0;
                        let mut t2 = 0;

                        if let Some(p) = p_ref {
                            if j >= p.len() / n_cores {
                                break;
                            }

                            let perm = &p[i * (p.len() / n_cores) + j];
                            cvars = perm.iter().map(|x| vars[*x].clone()).collect();
                        } else {
                            t1 = rng.random_range(0..cvars.len());
                            t2 = rng.random_range(0..cvars.len() - 1);

                            cvars.swap(t1, t2);
                        }

                        let horner: Vec<_> = expressions
                            .iter()
                            .map(|x| x.horner_scheme(Some(&cvars), true))
                            .collect();
                        let mut subexpr = HashSet::default();
                        let mut cur_ops = (0, 0);

                        for h in &horner {
                            let ops = h
                                .as_view()
                                .count_operations_with_subexpressions(&mut subexpr);
                            cur_ops = (cur_ops.0 + ops.0, cur_ops.1 + ops.1);
                        }

                        // prefer fewer multiplications
                        if cur_ops.1 <= last_mul || cur_ops.1 == last_mul && cur_ops.0 <= last_add {
                            if settings.verbose {
                                info!(
                                    "Accept move at step {}/{}: {} + and {} ×",
                                    j,
                                    settings.horner_iterations / n_cores,
                                    cur_ops.0,
                                    cur_ops.1
                                );
                            }

                            last_add = cur_ops.0;
                            last_mul = cur_ops.1;

                            if cur_ops.1 <= best_mul.load(Ordering::Relaxed)
                                || cur_ops.1 == best_mul.load(Ordering::Relaxed)
                                    && cur_ops.0 <= best_add.load(Ordering::Relaxed)
                            {
                                let mut best_scheme = best_scheme.lock().unwrap();

                                // check again if it is the best now that we have locked
                                let best_mul_l = best_mul.load(Ordering::Relaxed);
                                let best_add_l = best_add.load(Ordering::Relaxed);
                                if cur_ops.1 <= best_mul_l
                                    || cur_ops.1 == best_mul_l && cur_ops.0 <= best_add_l
                                {
                                    if cur_ops.0 == best_add_l && cur_ops.1 == best_mul_l {
                                        if *best_scheme < cvars {
                                            // on a draw, accept the lexicographical minimum
                                            // to get a deterministic scheme
                                            *best_scheme = cvars.clone();
                                        }
                                    } else {
                                        best_mul.store(cur_ops.1, Ordering::Relaxed);
                                        best_add.store(cur_ops.0, Ordering::Relaxed);
                                        *best_scheme = cvars.clone();
                                    }
                                }
                            }
                        } else {
                            cvars.swap(t1, t2);
                        }
                    }
                };

                if i + 1 < n_cores {
                    s.spawn(op);
                } else {
                    // execute in the main thread and do the abort check on the main thread
                    // this helps with catching ctrl-c
                    op()
                }
            }
        });

        if settings.verbose {
            info!(
                "Final scheme: {} + and {} ×",
                best_add.load(Ordering::Relaxed),
                best_mul.load(Ordering::Relaxed)
            );
        }

        Arc::try_unwrap(best_scheme).unwrap().into_inner().unwrap()
    }

    pub(crate) fn linearize_multiple<T: AtomCore>(
        expressions: &[T],
        fn_map: &FunctionMap,
        params: &[Atom],
        settings: OptimizationSettings,
    ) -> Result<ExpressionEvaluator<Complex<Rational>>, String> {
        let mut constants = Vec::new();
        let mut constant_map = HashMap::new();
        let mut instr = Vec::new();

        // we can only safely remove entries that don't depend on any of the function arguments
        let mut subexpression: HashMap<AtomView, Slot> = HashMap::default();

        let mut external_functions = vec![];

        let mut result_indices = vec![];
        let mut arg_stack = vec![];
        for expr in expressions {
            let res = expr.as_atom_view().linearize_impl(
                fn_map,
                params,
                &mut constants,
                &mut constant_map,
                &mut external_functions,
                &mut instr,
                &mut subexpression,
                &mut arg_stack,
                0,
            )?;
            result_indices.push(res);
        }

        let reserved_indices = params.len() + constants.len();

        let mut stack = vec![Complex::default(); params.len() + constants.len() + instr.len()];
        for (s, c) in stack.iter_mut().skip(params.len()).zip(constants) {
            *s = c;
        }

        macro_rules! slot_map {
            ($s: expr) => {
                match $s {
                    Slot::Param(i) => i,
                    Slot::Const(i) => params.len() + i,
                    Slot::Temp(i) => reserved_indices + i,
                    Slot::Out(_) => unreachable!(),
                }
            };
        }

        let mut instructions = vec![];
        for i in instr {
            match i {
                Instruction::Add(o, args, _) => {
                    instructions.push((
                        Instr::Add(
                            slot_map!(o),
                            args.clone().into_iter().map(|x| slot_map!(x)).collect(),
                        ),
                        ComplexPhase::default(),
                    ));
                }
                Instruction::Mul(o, args, _) => {
                    instructions.push((
                        Instr::Mul(
                            slot_map!(o),
                            args.clone().into_iter().map(|x| slot_map!(x)).collect(),
                        ),
                        ComplexPhase::default(),
                    ));
                }
                Instruction::Pow(o, base, exp, _) => {
                    instructions.push((
                        Instr::Pow(slot_map!(o), slot_map!(base), exp),
                        ComplexPhase::default(),
                    ));
                }
                Instruction::Powf(o, base, exp, _) => {
                    instructions.push((
                        Instr::Powf(slot_map!(o), slot_map!(base), slot_map!(exp)),
                        ComplexPhase::default(),
                    ));
                }
                Instruction::Fun(o, b, _) => {
                    let (sym, tags, args) = &*b;

                    if sym.is_fixed_builtin() {
                        if args.len() != 1 {
                            return Err(format!(
                                "Builtin function {} must have exactly one argument",
                                sym.get_name()
                            ));
                        }
                        instructions.push((
                            Instr::BuiltinFun(slot_map!(o), sym.clone(), slot_map!(args[0])),
                            ComplexPhase::default(),
                        ));
                        continue;
                    }

                    let tags = tags.iter().map(|x| crate::parse!(x)).collect::<Vec<_>>();
                    let index = if let Some(index) = external_functions
                        .iter()
                        .position(|x| x.symbol == *sym && x.tags == tags)
                    {
                        index
                    } else {
                        external_functions.push(ExternalFunctionContainer::new(*sym, tags));
                        external_functions.len() - 1
                    };

                    instructions.push((
                        Instr::ExternalFun(
                            slot_map!(o),
                            index,
                            args.clone().into_iter().map(|x| slot_map!(x)).collect(),
                        ),
                        ComplexPhase::default(),
                    ));
                }
                Instruction::Assign(_, _) => {
                    unimplemented!("Assign should not occur in input")
                }
                Instruction::IfElse(cond, l) => {
                    instructions.push((
                        Instr::IfElse(slot_map!(cond), Label(l)),
                        ComplexPhase::default(),
                    ));
                }
                Instruction::Goto(label) => {
                    instructions.push((Instr::Goto(Label(label)), ComplexPhase::default()));
                }
                Instruction::Label(label) => {
                    instructions.push((Instr::Label(Label(label)), ComplexPhase::default()));
                }
                Instruction::Join(o, cond, t, f) => {
                    instructions.push((
                        Instr::Join(slot_map!(o), slot_map!(cond), slot_map!(t), slot_map!(f)),
                        ComplexPhase::default(),
                    ));
                }
            }
        }

        Ok(ExpressionEvaluator {
            stack,
            param_count: params.len(),
            reserved_indices,
            instructions: instructions,
            result_indices: result_indices.iter().map(|s| slot_map!(*s)).collect(),
            external_fns: external_functions,
            settings: settings.clone(),
        })
    }

    // Yields the stack index that contains the output.
    fn linearize_impl(
        &self,
        fn_map: &'a FunctionMap,
        params: &[Atom],
        constants: &mut Vec<Complex<Rational>>,
        constant_map: &mut HashMap<Complex<Rational>, usize>,
        external_functions: &mut Vec<ExternalFunctionContainer<Complex<Rational>>>,
        instr: &mut Vec<Instruction>,
        subexpressions: &mut HashMap<AtomView<'a>, Slot>,
        args: &mut Vec<(AtomView<'a>, Slot)>,
        arg_start: usize,
    ) -> Result<Slot, String> {
        if matches!(*self, AtomView::Var(_) | AtomView::Fun(_)) {
            if let Some(p) = args.iter().skip(arg_start).find(|s| *self == s.0) {
                return Ok(p.1);
            }

            if let Some(p) = params.iter().position(|a| a.as_view() == *self) {
                return Ok(Slot::Param(p));
            }
        }

        if let Some(s) = subexpressions.get(self) {
            return Ok(*s);
        }

        let res = match self {
            AtomView::Num(n) => {
                let c = match n.get_coeff_view() {
                    CoefficientView::Natural(n, d, ni, di) => {
                        Complex::new(Rational::from((n, d)), Rational::from((ni, di)))
                    }
                    CoefficientView::Large(l, i) => Complex::new(l.to_rat(), i.to_rat()),
                    CoefficientView::Float(r, i) => {
                        // TODO: converting back to rational is slow
                        Complex::new(r.to_float().to_rational(), i.to_float().to_rational())
                    }
                    CoefficientView::Indeterminate => Err("Cannot convert indeterminate")?,
                    CoefficientView::Infinity(_) => Err("Cannot convert infinity")?,
                    CoefficientView::FiniteField(_, _) => {
                        Err("Finite field not yet supported for evaluation".to_string())?
                    }
                    CoefficientView::RationalPolynomial(_) => Err(
                        "Rational polynomial coefficient not yet supported for evaluation"
                            .to_string(),
                    )?,
                };

                if let Some(&i) = constant_map.get(&c) {
                    return Ok(Slot::Const(i));
                }

                let i = constants.len();
                constants.push(c.clone());
                constant_map.insert(c, i);
                Slot::Const(i)
            }
            AtomView::Var(v) => {
                let s = v.get_symbol();
                if s.get_evaluation_info().is_some() {
                    let i = register_constant_external_container(
                        external_functions,
                        s,
                        vec![],
                        constants,
                    );
                    return Ok(Slot::Const(i));
                }

                Err(format!(
                    "Variable {} not in constant map",
                    v.get_symbol().get_name()
                ))?
            }
            AtomView::Fun(f) => {
                let name = f.get_symbol();
                if [
                    Symbol::EXP_ID,
                    Symbol::LOG_ID,
                    Symbol::SIN_ID,
                    Symbol::COS_ID,
                    Symbol::SQRT_ID,
                    Symbol::ABS_ID,
                    Symbol::CONJ_ID,
                ]
                .contains(&name.get_id())
                {
                    assert!(f.get_nargs() == 1);
                    let arg = f.iter().next().unwrap();
                    let arg_eval = arg.linearize_impl(
                        fn_map,
                        params,
                        constants,
                        constant_map,
                        external_functions,
                        instr,
                        subexpressions,
                        args,
                        arg_start,
                    )?;

                    let temp = Slot::Temp(instr.len());
                    let c = Instruction::Fun(temp, Box::new((name, vec![], vec![arg_eval])), false);
                    instr.push(c);

                    subexpressions.insert(*self, temp);
                    return Ok(temp);
                }

                let fun = if name == Symbol::IF {
                    if f.get_nargs() != 3 {
                        return Err(format!(
                            "Condition function called with wrong number of arguments: {} vs 3",
                            f.get_nargs(),
                        ));
                    }

                    let mut arg_iter = f.iter();
                    let cond = arg_iter.next().unwrap();
                    let then_branch = arg_iter.next().unwrap();
                    let else_branch = arg_iter.next().unwrap();

                    let instr_len = instr.len();
                    let subexpression_len = subexpressions.len();
                    let cond = cond.linearize_impl(
                        fn_map,
                        params,
                        constants,
                        constant_map,
                        external_functions,
                        instr,
                        subexpressions,
                        args,
                        arg_start,
                    )?;

                    // try to resolve the condition if it is fully numeric
                    fn resolve(
                        cond: Slot,
                        instr: &[Instruction],
                        constants: &[Complex<Rational>],
                    ) -> Option<Complex<Rational>> {
                        let i = match cond {
                            Slot::Param(_) => {
                                return None;
                            }
                            Slot::Const(i) => {
                                // TODO: check that this constant is not an external function evaluation slot!
                                return Some(constants[i].clone());
                            }
                            Slot::Temp(t) => t,
                            Slot::Out(_) => {
                                unreachable!()
                            }
                        };

                        match &instr[i] {
                            Instruction::Add(_, args, _) => {
                                let mut res = Complex::default();
                                for x in args {
                                    match resolve(*x, instr, constants) {
                                        Some(v) => res += v,
                                        None => return None,
                                    }
                                }

                                Some(res)
                            }
                            Instruction::Mul(_, args, _) => {
                                let mut res = Complex::new(Rational::one(), Rational::zero());
                                for x in args {
                                    match resolve(*x, instr, constants) {
                                        Some(v) => res *= v,
                                        None => return None,
                                    }
                                }

                                Some(res)
                            }
                            Instruction::Pow(_, base, exp, _) => {
                                if let Some(base_val) = resolve(*base, instr, constants) {
                                    if *exp == -1 {
                                        Some(base_val.inv())
                                    } else if *exp < 0 {
                                        Some(base_val.pow(exp.unsigned_abs()).inv())
                                    } else {
                                        Some(base_val.pow(exp.unsigned_abs()))
                                    }
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        }
                    }

                    if let Some(cond_res) = resolve(cond, instr, constants) {
                        // remove dead code
                        instr.truncate(instr_len);
                        if subexpression_len != subexpressions.len() {
                            // remove subexpressions that are created as part of the conditions
                            subexpressions.retain(|_, &mut v| {
                                if let Slot::Temp(v) = v {
                                    v < instr_len
                                } else {
                                    true
                                }
                            });
                        }

                        let res = if !cond_res.is_zero() {
                            then_branch.linearize_impl(
                                fn_map,
                                params,
                                constants,
                                constant_map,
                                external_functions,
                                instr,
                                subexpressions,
                                args,
                                arg_start,
                            )?
                        } else {
                            else_branch.linearize_impl(
                                fn_map,
                                params,
                                constants,
                                constant_map,
                                external_functions,
                                instr,
                                subexpressions,
                                args,
                                arg_start,
                            )?
                        };

                        subexpressions.insert(*self, res);
                        return Ok(res);
                    }

                    let if_instr_pos = instr.len();
                    instr.push(Instruction::IfElse(cond, 0));

                    let mut sub_expr_pos_child = subexpressions.clone(); // TODO: prevent clone?
                    let then_branch = then_branch.linearize_impl(
                        fn_map,
                        params,
                        constants,
                        constant_map,
                        external_functions,
                        instr,
                        &mut sub_expr_pos_child,
                        args,
                        arg_start,
                    )?;

                    let label_end_pos = instr.len();
                    instr.push(Instruction::Goto(0));
                    instr[if_instr_pos] = Instruction::IfElse(cond, instr.len());
                    instr.push(Instruction::Label(instr.len()));

                    sub_expr_pos_child.clone_from(&subexpressions);
                    let else_branch = else_branch.linearize_impl(
                        fn_map,
                        params,
                        constants,
                        constant_map,
                        external_functions,
                        instr,
                        &mut sub_expr_pos_child,
                        args,
                        arg_start,
                    )?;

                    instr[label_end_pos] = Instruction::Goto(instr.len());
                    instr.push(Instruction::Label(instr.len()));

                    let temp = Slot::Temp(instr.len());
                    instr.push(Instruction::Join(temp, cond, then_branch, else_branch));
                    return Ok(temp);
                } else if let Some(fun) = fn_map.get(*self) {
                    fun
                } else if let Some(eval_info) = name.get_evaluation_info() {
                    let tags = f
                        .iter()
                        .take(eval_info.get_tag_count())
                        .map(|x| x.to_owned())
                        .collect::<Vec<_>>();

                    // check if it a constant external function
                    if f.get_nargs() == eval_info.get_tag_count() {
                        let i = register_constant_external_container(
                            external_functions,
                            name,
                            tags,
                            constants,
                        );
                        return Ok(Slot::Const(i));
                    };

                    let eval_args = f
                        .iter()
                        .skip(eval_info.get_tag_count())
                        .map(|arg| {
                            arg.linearize_impl(
                                fn_map,
                                params,
                                constants,
                                constant_map,
                                external_functions,
                                instr,
                                subexpressions,
                                args,
                                arg_start,
                            )
                        })
                        .collect::<Result<_, _>>()?;

                    let temp = Slot::Temp(instr.len());
                    instr.push(Instruction::Fun(
                        temp,
                        Box::new((
                            name,
                            tags.iter().map(|x| x.to_canonical_string()).collect(),
                            eval_args,
                        )),
                        false,
                    ));
                    return Ok(temp);
                } else {
                    return Err(format!("Undefined function {}", self.to_plain_string()));
                };

                {
                    let Expr {
                        tag_len,
                        args: arg_spec,
                        body: e,
                        ..
                    } = fun;

                    if f.get_nargs() != arg_spec.len() + tag_len {
                        return Err(format!(
                            "Function {} called with wrong number of arguments: {} vs {}",
                            f.get_symbol().get_name(),
                            f.get_nargs(),
                            arg_spec.len() + tag_len
                        ));
                    }

                    let old_arg_stack_len = args.len();

                    let mut arg_shadowed = false;
                    for (eval_arg, arg_spec) in f.iter().skip(*tag_len).zip(arg_spec) {
                        let slot = eval_arg.linearize_impl(
                            fn_map,
                            params,
                            constants,
                            constant_map,
                            external_functions,
                            instr,
                            subexpressions,
                            args,
                            arg_start,
                        )?;

                        if args.iter().any(|(a, _)| *a == arg_spec.as_view()) {
                            arg_shadowed = true;
                        }

                        args.push((arg_spec.as_view(), slot));
                    }

                    // inline function call
                    // we have to use a new subexpression list as the function has arguments that may be different per call
                    // this means that not all subexpressions will be shared across calls
                    let mut sub_expr_pos_child = HashMap::default();
                    let r = e.as_view().linearize_impl(
                        fn_map,
                        params,
                        constants,
                        constant_map,
                        external_functions,
                        instr,
                        if old_arg_stack_len == args.len() {
                            subexpressions
                        } else {
                            // we can only inherit the subexpressions if the new function argument symbols
                            // have not been used earlier
                            if !arg_shadowed {
                                sub_expr_pos_child.clone_from(subexpressions);
                            }

                            &mut sub_expr_pos_child
                        },
                        args,
                        old_arg_stack_len,
                    )?;

                    args.truncate(old_arg_stack_len);

                    r
                }
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();
                let b_eval = b.linearize_impl(
                    fn_map,
                    params,
                    constants,
                    constant_map,
                    external_functions,
                    instr,
                    subexpressions,
                    args,
                    arg_start,
                )?;

                if let AtomView::Num(n) = e
                    && let CoefficientView::Natural(num, den, num_i, _den_i) = n.get_coeff_view()
                    && den == 1
                    && num_i == 0
                {
                    let new_base = if num.unsigned_abs() > 1 {
                        let temp = Slot::Temp(instr.len());
                        instr.push(Instruction::Mul(
                            temp,
                            vec![b_eval; num.unsigned_abs() as usize],
                            0,
                        ));
                        temp
                    } else {
                        b_eval
                    };

                    let res = if num > 0 {
                        new_base
                    } else {
                        let temp = Slot::Temp(instr.len());
                        instr.push(Instruction::Pow(temp, new_base, -1, false));
                        temp
                    };

                    subexpressions.insert(*self, res);
                    return Ok(res);
                }

                let e_eval = e.linearize_impl(
                    fn_map,
                    params,
                    constants,
                    constant_map,
                    external_functions,
                    instr,
                    subexpressions,
                    args,
                    arg_start,
                )?;

                let temp = Slot::Temp(instr.len());
                instr.push(Instruction::Powf(temp, b_eval, e_eval, false));
                temp
            }
            AtomView::Mul(m) => {
                let mut muls = vec![];
                for arg in m.iter() {
                    let a = arg.linearize_impl(
                        fn_map,
                        params,
                        constants,
                        constant_map,
                        external_functions,
                        instr,
                        subexpressions,
                        args,
                        arg_start,
                    )?;
                    muls.push(a);
                }

                muls.sort();

                let temp = Slot::Temp(instr.len());
                instr.push(Instruction::Mul(temp, muls, 0));
                temp
            }
            AtomView::Add(a) => {
                let mut adds = vec![];
                for arg in a.iter() {
                    adds.push(arg.linearize_impl(
                        fn_map,
                        params,
                        constants,
                        constant_map,
                        external_functions,
                        instr,
                        subexpressions,
                        args,
                        arg_start,
                    )?);
                }

                adds.sort();

                let temp = Slot::Temp(instr.len());
                instr.push(Instruction::Add(temp, adds, 0));
                temp
            }
        };

        subexpressions.insert(*self, res);
        Ok(res)
    }
}

/// A hash of an expression, used for common subexpression elimination.
pub type ExpressionHash = u64;

/// A tree representation of an expression.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expression<T> {
    Const(ExpressionHash, Box<T>),
    Parameter(ExpressionHash, usize),
    Eval(ExpressionHash, u32, Vec<Expression<T>>),
    Add(ExpressionHash, Vec<Expression<T>>),
    Mul(ExpressionHash, Vec<Expression<T>>),
    Pow(ExpressionHash, Box<(Expression<T>, i64)>),
    Powf(ExpressionHash, Box<(Expression<T>, Expression<T>)>),
    ReadArg(ExpressionHash, usize), // read nth function argument
    BuiltinFun(ExpressionHash, Symbol, Box<Expression<T>>),
    Fun(ExpressionHash, Symbol, Vec<String>, Vec<Expression<T>>),
    IfElse(
        ExpressionHash,
        Box<(Expression<T>, Expression<T>, Expression<T>)>,
    ),
    SubExpression(ExpressionHash, usize),
}

impl<T> Expression<T> {
    fn get_hash(&self) -> ExpressionHash {
        match self {
            Expression::Const(h, _) => *h,
            Expression::Parameter(h, _) => *h,
            Expression::Eval(h, _, _) => *h,
            Expression::Add(h, _) => *h,
            Expression::Mul(h, _) => *h,
            Expression::Pow(h, _) => *h,
            Expression::Powf(h, _) => *h,
            Expression::ReadArg(h, _) => *h,
            Expression::BuiltinFun(h, _, _) => *h,
            Expression::SubExpression(h, _) => *h,
            Expression::Fun(h, _, _, _) => *h,
            Expression::IfElse(h, _) => *h,
        }
    }
}

impl<T: Eq + InternalOrdering> PartialOrd for Expression<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Eq + InternalOrdering> Ord for Expression<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Expression::Const(_, a), Expression::Const(_, b)) => a.internal_cmp(b),
            (Expression::Parameter(_, a), Expression::Parameter(_, b)) => a.cmp(b),
            (Expression::Eval(_, a, b), Expression::Eval(_, c, d)) => {
                a.cmp(c).then_with(|| b.cmp(d))
            }
            (Expression::Add(_, a), Expression::Add(_, b)) => a.cmp(b),
            (Expression::Mul(_, a), Expression::Mul(_, b)) => a.cmp(b),
            (Expression::Pow(_, p1), Expression::Pow(_, p2)) => p1.cmp(p2),
            (Expression::Powf(_, p1), Expression::Powf(_, p2)) => p1.cmp(p2),
            (Expression::ReadArg(_, r1), Expression::ReadArg(_, r2)) => r1.cmp(r2),
            (Expression::BuiltinFun(_, a, b), Expression::BuiltinFun(_, c, d)) => {
                a.cmp(c).then_with(|| b.cmp(d))
            }
            (Expression::SubExpression(_, s1), Expression::SubExpression(_, s2)) => s1.cmp(s2),
            (Expression::Fun(_, a, b, c), Expression::Fun(_, d, e, f)) => {
                a.cmp(d).then_with(|| b.cmp(e)).then_with(|| c.cmp(f))
            }
            (Expression::IfElse(_, a), Expression::IfElse(_, b)) => a.cmp(b),
            (Expression::Const(_, _), _) => std::cmp::Ordering::Less,
            (_, Expression::Const(_, _)) => std::cmp::Ordering::Greater,
            (Expression::Parameter(_, _), _) => std::cmp::Ordering::Less,
            (_, Expression::Parameter(_, _)) => std::cmp::Ordering::Greater,
            (Expression::Eval(_, _, _), _) => std::cmp::Ordering::Less,
            (_, Expression::Eval(_, _, _)) => std::cmp::Ordering::Greater,
            (Expression::Add(_, _), _) => std::cmp::Ordering::Less,
            (_, Expression::Add(_, _)) => std::cmp::Ordering::Greater,
            (Expression::Mul(_, _), _) => std::cmp::Ordering::Less,
            (_, Expression::Mul(_, _)) => std::cmp::Ordering::Greater,
            (Expression::Pow(_, _), _) => std::cmp::Ordering::Less,
            (_, Expression::Pow(_, _)) => std::cmp::Ordering::Greater,
            (Expression::Powf(_, _), _) => std::cmp::Ordering::Less,
            (_, Expression::Powf(_, _)) => std::cmp::Ordering::Greater,
            (Expression::ReadArg(_, _), _) => std::cmp::Ordering::Less,
            (_, Expression::ReadArg(_, _)) => std::cmp::Ordering::Greater,
            (Expression::BuiltinFun(_, _, _), _) => std::cmp::Ordering::Less,
            (_, Expression::BuiltinFun(_, _, _)) => std::cmp::Ordering::Greater,
            (Expression::Fun(_, _, _, _), _) => std::cmp::Ordering::Less,
            (_, Expression::Fun(_, _, _, _)) => std::cmp::Ordering::Greater,
            (Expression::IfElse(_, _), _) => std::cmp::Ordering::Greater,
            (_, Expression::IfElse(_, _)) => std::cmp::Ordering::Less, // sort last so that parent common subexpressions can be used inside both branches
        }
    }
}

impl<T: Eq + Hash> Hash for Expression<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.get_hash())
    }
}

impl<T: Eq + Hash + Clone + InternalOrdering> Expression<T> {
    fn find_subexpression<'a>(&'a self, subexp: &mut HashMap<&'a Expression<T>, usize>) -> bool {
        if matches!(
            self,
            Expression::Const(_, _) | Expression::Parameter(_, _) | Expression::ReadArg(_, _)
        ) {
            return true;
        }

        if let Some(i) = subexp.get_mut(self) {
            *i += 1;
            return true;
        }

        subexp.insert(self, 1);

        match self {
            Expression::Const(_, _) | Expression::Parameter(_, _) | Expression::ReadArg(_, _) => {}
            Expression::Eval(_, _, ae) => {
                for arg in ae {
                    arg.find_subexpression(subexp);
                }
            }
            Expression::Add(_, a) | Expression::Mul(_, a) | Expression::Fun(_, _, _, a) => {
                for arg in a {
                    arg.find_subexpression(subexp);
                }
            }
            Expression::Pow(_, p) => {
                p.0.find_subexpression(subexp);
            }
            Expression::Powf(_, p) => {
                p.0.find_subexpression(subexp);
                p.1.find_subexpression(subexp);
            }
            Expression::BuiltinFun(_, _, a) => {
                a.find_subexpression(subexp);
            }
            Expression::SubExpression(_, _) => {}
            Expression::IfElse(_, b) => {
                b.0.find_subexpression(subexp);
                b.1.find_subexpression(subexp);
                b.2.find_subexpression(subexp);
            }
        }

        false
    }

    fn replace_subexpression(&mut self, subexp: &HashMap<&Expression<T>, usize>, skip_root: bool) {
        if !skip_root && let Some(i) = subexp.get(self) {
            *self = Expression::SubExpression(self.get_hash(), *i); // TODO: do not recyle hash?
            return;
        }

        match self {
            Expression::Const(_, _) | Expression::Parameter(_, _) | Expression::ReadArg(_, _) => {}
            Expression::Eval(_, _, ae) => {
                for arg in &mut *ae {
                    arg.replace_subexpression(subexp, false);
                }
            }
            Expression::Add(_, a) | Expression::Mul(_, a) | Expression::Fun(_, _, _, a) => {
                for arg in a {
                    arg.replace_subexpression(subexp, false);
                }
            }
            Expression::Pow(_, p) => {
                p.0.replace_subexpression(subexp, false);
            }
            Expression::Powf(_, p) => {
                p.0.replace_subexpression(subexp, false);
                p.1.replace_subexpression(subexp, false);
            }
            Expression::BuiltinFun(_, _, a) => {
                a.replace_subexpression(subexp, false);
            }
            Expression::SubExpression(_, _) => {}
            Expression::IfElse(_, b) => {
                b.0.replace_subexpression(subexp, false);
                b.1.replace_subexpression(subexp, false);
                b.2.replace_subexpression(subexp, false);
            }
        }
    }

    // Count the number of additions and multiplications in the expression, counting
    // subexpressions only once.
    pub fn count_operations_with_subexpression<'a>(
        &'a self,
        sub_expr: &mut HashMap<&'a Self, usize>,
    ) -> (usize, usize) {
        if matches!(
            self,
            Expression::Const(_, _) | Expression::Parameter(_, _) | Expression::ReadArg(_, _)
        ) {
            return (0, 0);
        }

        if sub_expr.contains_key(self) {
            return (0, 0);
        }

        sub_expr.insert(self, 1);

        match self {
            Expression::Const(_, _) => (0, 0),
            Expression::Parameter(_, _) => (0, 0),
            Expression::Eval(_, _, args) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in args {
                    let (a, m) = arg.count_operations_with_subexpression(sub_expr);
                    add += a;
                    mul += m;
                }
                (add, mul)
            }
            Expression::Add(_, a) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in a {
                    let (a, m) = arg.count_operations_with_subexpression(sub_expr);
                    add += a;
                    mul += m;
                }
                (add + a.len() - 1, mul)
            }
            Expression::Mul(_, m) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in m {
                    let (a, m) = arg.count_operations_with_subexpression(sub_expr);
                    add += a;
                    mul += m;
                }
                (add, mul + m.len() - 1)
            }
            Expression::Pow(_, p) => {
                let (a, m) = p.0.count_operations_with_subexpression(sub_expr);
                (a, m + p.1.unsigned_abs() as usize - 1)
            }
            Expression::Powf(_, p) => {
                let (a, m) = p.0.count_operations_with_subexpression(sub_expr);
                let (a2, m2) = p.1.count_operations_with_subexpression(sub_expr);
                (a + a2, m + m2 + 1) // not clear how to count this
            }
            Expression::ReadArg(_, _) => (0, 0),
            Expression::BuiltinFun(_, _, b) => b.count_operations_with_subexpression(sub_expr), // not clear how to count this, third arg?
            Expression::SubExpression(_, _) => (0, 0),
            Expression::Fun(_, _, _, a) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in a {
                    let (a, m) = arg.count_operations_with_subexpression(sub_expr);
                    add += a;
                    mul += m;
                }
                (add + a.len() - 1, mul)
            }
            Expression::IfElse(_, b) => {
                let (a1, m1) = b.0.count_operations_with_subexpression(sub_expr);
                let (a2, m2) = b.1.count_operations_with_subexpression(sub_expr);
                let (a3, m3) = b.2.count_operations_with_subexpression(sub_expr);
                (a1 + a2 + a3, m1 + m2 + m3)
            }
        }
    }
}

impl<T: std::hash::Hash + Clone> Expression<T> {
    fn rehashed(mut self, partial: bool) -> Self {
        self.rehash(partial);
        self
    }

    fn rehash(&mut self, partial: bool) -> ExpressionHash {
        match self {
            Expression::Const(h, c) => {
                if partial && *h != 0 {
                    return *h;
                }

                let mut hasher = AHasher::default();
                hasher.write_u8(0);
                c.hash(&mut hasher);
                *h = hasher.finish();
                *h
            }
            Expression::Parameter(h, p) => {
                if partial && *h != 0 {
                    return *h;
                }

                let mut hasher = AHasher::default();
                hasher.write_u8(1);
                hasher.write_usize(*p);
                *h = hasher.finish();
                *h
            }
            Expression::Eval(h, i, v) => {
                if partial && *h != 0 {
                    return *h;
                }

                let mut hasher = AHasher::default();
                hasher.write_u8(2);
                hasher.write_u32(*i);
                for x in v {
                    hasher.write_u64(x.rehash(partial));
                }
                *h = hasher.finish();
                *h
            }
            Expression::Add(h, v) => {
                if partial && *h != 0 {
                    return *h;
                }

                let mut hasher = AHasher::default();
                hasher.write_u8(3);

                // do an additive hash
                let mut arg_sum = 0u64;
                for x in v {
                    arg_sum = arg_sum.wrapping_add(x.rehash(partial));
                }
                hasher.write_u64(arg_sum);
                *h = hasher.finish();
                *h
            }
            Expression::Mul(h, v) => {
                if partial && *h != 0 {
                    return *h;
                }

                let mut hasher = AHasher::default();
                hasher.write_u8(4);

                // do an additive hash
                let mut arg_sum = 0u64;
                for x in v {
                    arg_sum = arg_sum.wrapping_add(x.rehash(partial));
                }
                hasher.write_u64(arg_sum);
                *h = hasher.finish();
                *h
            }
            Expression::Pow(h, p) => {
                if partial && *h != 0 {
                    return *h;
                }

                let mut hasher = AHasher::default();
                hasher.write_u8(5);
                let hb = p.0.rehash(partial);
                hasher.write_u64(hb);
                hasher.write_i64(p.1);
                *h = hasher.finish();
                *h
            }
            Expression::Powf(h, p) => {
                if partial && *h != 0 {
                    return *h;
                }

                let mut hasher = AHasher::default();
                hasher.write_u8(6);
                let hb = p.0.rehash(partial);
                let he = p.1.rehash(partial);
                hasher.write_u64(hb);
                hasher.write_u64(he);
                *h = hasher.finish();
                *h
            }
            Expression::ReadArg(h, i) => {
                if partial && *h != 0 {
                    return *h;
                }

                let mut hasher = AHasher::default();
                hasher.write_u8(7);
                hasher.write_usize(*i);
                *h = hasher.finish();
                *h
            }
            Expression::BuiltinFun(h, s, a) => {
                if partial && *h != 0 {
                    return *h;
                }

                let mut hasher = AHasher::default();
                hasher.write_u8(8);
                s.hash(&mut hasher);
                let ha = a.rehash(partial);
                hasher.write_u64(ha);
                *h = hasher.finish();
                *h
            }
            Expression::SubExpression(h, i) => {
                if partial && *h != 0 {
                    return *h;
                }

                let mut hasher = AHasher::default();
                hasher.write_u8(9);
                hasher.write_usize(*i);
                *h = hasher.finish();
                *h
            }
            Expression::Fun(h, s, tags, a) => {
                if partial && *h != 0 {
                    return *h;
                }

                let mut hasher = AHasher::default();
                hasher.write_u8(10);
                s.hash(&mut hasher);
                tags.hash(&mut hasher);
                for x in a {
                    hasher.write_u64(x.rehash(partial));
                }
                *h = hasher.finish();
                *h
            }
            Expression::IfElse(h, b) => {
                if partial && *h != 0 {
                    return *h;
                }

                let mut hasher = AHasher::default();
                hasher.write_u8(11);
                hasher.write_u64(b.0.rehash(partial));
                hasher.write_u64(b.1.rehash(partial));
                hasher.write_u64(b.2.rehash(partial));
                *h = hasher.finish();
                *h
            }
        }
    }
}

impl<T> ExpressionEvaluator<T> {
    pub(super) fn export_external_cpps(&self) -> String {
        let mut seen = HashSet::default();
        let mut res = String::new();

        for external in &self.external_fns {
            if external.constant_index.is_some() {
                continue;
            }

            let Some(snippet) = external.cpp() else {
                continue;
            };

            if seen.insert(snippet.to_owned()) {
                res += snippet;
                if !snippet.ends_with('\n') {
                    res += "\n";
                }
                res += "\n";
            }
        }

        res
    }
}

impl<T: Clone + PartialEq> SplitExpression<T> {
    pub fn map_coeff<T2, F: Fn(&T) -> T2>(&self, f: &F) -> SplitExpression<T2> {
        SplitExpression {
            tree: self.tree.iter().map(|x| x.map_coeff(f)).collect(),
            subexpressions: self.subexpressions.iter().map(|x| x.map_coeff(f)).collect(),
        }
    }
}

impl<T: Clone + PartialEq> Expression<T> {
    /// Map the coefficients.
    ///
    /// Note that no rehashing is performed.
    pub fn map_coeff<T2, F: Fn(&T) -> T2>(&self, f: &F) -> Expression<T2> {
        match self {
            Expression::Const(h, c) => Expression::Const(*h, Box::new(f(c))),
            Expression::Parameter(h, p) => Expression::Parameter(*h, *p),
            Expression::Eval(h, id, e_args) => {
                Expression::Eval(*h, *id, e_args.iter().map(|x| x.map_coeff(f)).collect())
            }
            Expression::Add(h, a) => {
                let new_args = a.iter().map(|x| x.map_coeff(f)).collect();
                Expression::Add(*h, new_args)
            }
            Expression::Mul(h, m) => {
                let new_args = m.iter().map(|x| x.map_coeff(f)).collect();
                Expression::Mul(*h, new_args)
            }
            Expression::Pow(h, p) => {
                let (b, e) = &**p;
                Expression::Pow(*h, Box::new((b.map_coeff(f), *e)))
            }
            Expression::Powf(h, p) => {
                let (b, e) = &**p;
                Expression::Powf(*h, Box::new((b.map_coeff(f), e.map_coeff(f))))
            }
            Expression::ReadArg(h, s) => Expression::ReadArg(*h, *s),
            Expression::BuiltinFun(h, s, a) => {
                Expression::BuiltinFun(*h, *s, Box::new(a.map_coeff(f)))
            }
            Expression::SubExpression(h, i) => Expression::SubExpression(*h, *i),
            Expression::Fun(h, s, tags, a) => {
                let new_args = a.iter().map(|x| x.map_coeff(f)).collect();
                Expression::Fun(*h, *s, tags.clone(), new_args)
            }
            Expression::IfElse(h, b) => {
                let (cond, then_expr, else_expr) = &**b;
                Expression::IfElse(
                    *h,
                    Box::new((
                        cond.map_coeff(f),
                        then_expr.map_coeff(f),
                        else_expr.map_coeff(f),
                    )),
                )
            }
        }
    }

    fn strip_constants(&mut self, stack: &mut Vec<T>, param_len: usize) {
        match self {
            Expression::Const(_, t) => {
                if let Some(p) = stack.iter().skip(param_len).position(|x| x == &**t) {
                    *self = Expression::Parameter(0, param_len + p);
                } else {
                    stack.push(t.as_ref().clone());
                    *self = Expression::Parameter(0, stack.len() - 1);
                }
            }
            Expression::Parameter(_, _) => {}
            Expression::Eval(_, _, e_args) => {
                for a in e_args {
                    a.strip_constants(stack, param_len);
                }
            }
            Expression::Add(_, a) | Expression::Mul(_, a) => {
                for arg in a {
                    arg.strip_constants(stack, param_len);
                }
            }
            Expression::Pow(_, p) => {
                p.0.strip_constants(stack, param_len);
            }
            Expression::Powf(_, p) => {
                p.0.strip_constants(stack, param_len);
                p.1.strip_constants(stack, param_len);
            }
            Expression::ReadArg(_, _) => {}
            Expression::BuiltinFun(_, _, a) => {
                a.strip_constants(stack, param_len);
            }
            Expression::SubExpression(_, _) => {}
            Expression::Fun(_, _, _, a) => {
                for arg in a {
                    arg.strip_constants(stack, param_len);
                }
            }
            Expression::IfElse(_, b) => {
                b.0.strip_constants(stack, param_len);
                b.1.strip_constants(stack, param_len);
                b.2.strip_constants(stack, param_len);
            }
        }
    }
}

impl<T: Clone + PartialEq> EvalTree<T> {
    pub fn map_coeff<T2: EvaluationDomain, F: Fn(&T) -> T2>(&self, f: &F) -> EvalTree<T2> {
        EvalTree {
            expressions: SplitExpression {
                tree: self
                    .expressions
                    .tree
                    .iter()
                    .map(|x| x.map_coeff(f))
                    .collect(),
                subexpressions: self
                    .expressions
                    .subexpressions
                    .iter()
                    .map(|x| x.map_coeff(f))
                    .collect(),
            },
            functions: self
                .functions
                .iter()
                .map(|(s, a, e)| (s.clone(), a.clone(), e.map_coeff(f)))
                .collect(),
            external_functions: self.external_functions.iter().map(|x| x.map()).collect(),
            param_count: self.param_count,
        }
    }
}

impl EvalTree<Complex<Rational>> {
    /// Create a linear version of the tree that can be evaluated more efficiently.
    pub fn linearize(
        &mut self,
        settings: &OptimizationSettings,
    ) -> ExpressionEvaluator<Complex<Rational>> {
        let mut stack = vec![Complex::<Rational>::default(); self.param_count];

        // strip every constant and move them into the stack after the params
        self.strip_constants(&mut stack); // FIXME
        let reserved_indices = stack.len();

        let mut sub_expr_pos = HashMap::default();
        let mut instructions = vec![];

        let mut result_indices = vec![];

        for t in &self.expressions.tree {
            let result_index = self.linearize_impl(
                t,
                &self.expressions.subexpressions,
                &mut stack,
                &mut instructions,
                &mut sub_expr_pos,
                &[],
                false,
                reserved_indices,
            );
            result_indices.push(result_index);
        }

        let mut e = ExpressionEvaluator {
            stack,
            param_count: self.param_count,
            reserved_indices,
            instructions,
            result_indices,
            external_fns: self.external_functions.clone(),
            settings: settings.clone(),
        };

        loop {
            let r = e.remove_common_instructions();

            if r == 0 || e.settings.abort_level > 0 {
                e.settings.abort_level = 0;
                break;
            }

            if settings.verbose {
                let (add_count, mul_count) = e.count_operations();
                info!(
                    "Removed {} common instructions: {} + and {} ×",
                    r, add_count, mul_count
                );
            }
        }

        for _ in 0..settings.cpe_iterations.unwrap_or(usize::MAX) {
            let r = e.remove_common_pairs();
            if r == 0 || e.settings.abort_level > 0 {
                e.settings.abort_level = 0;
                break;
            }

            if settings.verbose {
                let (add_count, mul_count) = e.count_operations();
                info!(
                    "Removed {} common pairs: {} + and {} ×",
                    r, add_count, mul_count
                );
            }
        }

        e.optimize_stack();
        e
    }

    fn strip_constants(&mut self, stack: &mut Vec<Complex<Rational>>) {
        for t in &mut self.expressions.tree {
            t.strip_constants(stack, self.param_count);
        }

        for e in &mut self.expressions.subexpressions {
            e.strip_constants(stack, self.param_count);
        }

        for (_, _, e) in &mut self.functions {
            for t in &mut e.tree {
                t.strip_constants(stack, self.param_count);
            }

            for e in &mut e.subexpressions {
                e.strip_constants(stack, self.param_count);
            }
        }
    }

    // Yields the stack index that contains the output.
    fn linearize_impl(
        &self,
        tree: &Expression<Complex<Rational>>,
        subexpressions: &[Expression<Complex<Rational>>],
        stack: &mut Vec<Complex<Rational>>,
        instr: &mut Vec<(Instr, ComplexPhase)>,
        sub_expr_pos: &mut HashMap<usize, usize>,
        args: &[usize],
        in_branch: bool,
        reserved_indices: usize,
    ) -> usize {
        match tree {
            Expression::Const(_, t) => {
                unreachable!(
                    "Constants should have been stripped from the expression tree. Found constant {}",
                    t
                );
            }
            Expression::Parameter(_, i) => *i,
            Expression::Eval(_, id, e_args) => {
                // inline the function
                let new_args: Vec<_> = e_args
                    .iter()
                    .map(|x| {
                        self.linearize_impl(
                            x,
                            subexpressions,
                            stack,
                            instr,
                            sub_expr_pos,
                            args,
                            in_branch,
                            reserved_indices,
                        )
                    })
                    .collect();

                let mut sub_expr_pos = HashMap::default();
                let func = &self.functions[*id as usize].2;
                self.linearize_impl(
                    &func.tree[0],
                    &func.subexpressions,
                    stack,
                    instr,
                    &mut sub_expr_pos,
                    &new_args,
                    in_branch,
                    reserved_indices,
                )
            }
            Expression::Add(_, a) => {
                let mut args: Vec<_> = a
                    .iter()
                    .map(|x| {
                        self.linearize_impl(
                            x,
                            subexpressions,
                            stack,
                            instr,
                            sub_expr_pos,
                            args,
                            in_branch,
                            reserved_indices,
                        )
                    })
                    .collect();
                args.sort();

                stack.push(Complex::default());
                let res = stack.len() - 1;

                let add = Instr::Add(res, args);
                instr.push((add, ComplexPhase::Any));

                res
            }
            Expression::Mul(_, m) => {
                let mut args: Vec<_> = m
                    .iter()
                    .map(|x| {
                        self.linearize_impl(
                            x,
                            subexpressions,
                            stack,
                            instr,
                            sub_expr_pos,
                            args,
                            in_branch,
                            reserved_indices,
                        )
                    })
                    .collect();
                args.sort();

                stack.push(Complex::default());
                let res = stack.len() - 1;

                let mul = Instr::Mul(res, args);
                instr.push((mul, ComplexPhase::Any));

                res
            }
            Expression::Pow(_, p) => {
                let b = self.linearize_impl(
                    &p.0,
                    subexpressions,
                    stack,
                    instr,
                    sub_expr_pos,
                    args,
                    in_branch,
                    reserved_indices,
                );
                stack.push(Complex::default());
                let mut res = stack.len() - 1;

                if p.1 > 1 {
                    instr.push((Instr::Mul(res, vec![b; p.1 as usize]), ComplexPhase::Any));
                } else if p.1 < -1 {
                    instr.push((Instr::Mul(res, vec![b; -p.1 as usize]), ComplexPhase::Any));
                    stack.push(Complex::default());
                    res += 1;
                    instr.push((Instr::Pow(res, res - 1, -1), ComplexPhase::Any));
                } else {
                    instr.push((Instr::Pow(res, b, p.1), ComplexPhase::Any));
                }
                res
            }
            Expression::Powf(_, p) => {
                let b = self.linearize_impl(
                    &p.0,
                    subexpressions,
                    stack,
                    instr,
                    sub_expr_pos,
                    args,
                    in_branch,
                    reserved_indices,
                );
                let e = self.linearize_impl(
                    &p.1,
                    subexpressions,
                    stack,
                    instr,
                    sub_expr_pos,
                    args,
                    in_branch,
                    reserved_indices,
                );
                stack.push(Complex::default());
                let res = stack.len() - 1;

                instr.push((Instr::Powf(res, b, e), ComplexPhase::Any));
                res
            }
            Expression::ReadArg(_, a) => args[*a],
            Expression::BuiltinFun(_, s, v) => {
                let arg = self.linearize_impl(
                    v,
                    subexpressions,
                    stack,
                    instr,
                    sub_expr_pos,
                    args,
                    in_branch,
                    reserved_indices,
                );
                stack.push(Complex::default());
                let c = Instr::BuiltinFun(stack.len() - 1, *s, arg);
                instr.push((c, ComplexPhase::Any));
                stack.len() - 1
            }
            Expression::SubExpression(_, id) => {
                if sub_expr_pos.contains_key(id) {
                    *sub_expr_pos.get(id).unwrap()
                } else {
                    let res = self.linearize_impl(
                        &subexpressions[*id],
                        subexpressions,
                        stack,
                        instr,
                        sub_expr_pos,
                        args,
                        in_branch,
                        reserved_indices,
                    );

                    // only register the subexpression as computed when it is not
                    // computed in a branch, as the sub expression may not be computed
                    // in the other branch
                    if !in_branch {
                        sub_expr_pos.insert(*id, res);
                    }

                    res
                }
            }
            Expression::Fun(_, s, tags, v) => {
                let args: Vec<_> = v
                    .iter()
                    .map(|x| {
                        self.linearize_impl(
                            x,
                            subexpressions,
                            stack,
                            instr,
                            sub_expr_pos,
                            args,
                            in_branch,
                            reserved_indices,
                        )
                    })
                    .collect();

                stack.push(Complex::default());
                let res = stack.len() - 1;

                let tag_atoms = tags.iter().map(|x| crate::parse!(x)).collect::<Vec<_>>();
                let index = self
                    .external_functions
                    .iter()
                    .position(|x| x.symbol == *s && x.tags == tag_atoms)
                    .expect("missing external function container");

                let f = Instr::ExternalFun(res, index, args);
                instr.push((f, ComplexPhase::Any));

                res
            }
            Expression::IfElse(_, b) => {
                let instr_len = instr.len();
                let stack_len = stack.len();
                let subexpression_len = sub_expr_pos.len();
                let cond = self.linearize_impl(
                    &b.0,
                    subexpressions,
                    stack,
                    instr,
                    sub_expr_pos,
                    args,
                    in_branch,
                    reserved_indices,
                );

                // try to resolve the condition if it is fully numeric
                fn resolve(
                    instr: &[(Instr, ComplexPhase)],
                    stack: &[Complex<Rational>],
                    cond: usize,
                    param_count: usize,
                    reserved_indices: usize,
                ) -> Option<Complex<Rational>> {
                    if cond < param_count {
                        return None;
                    }
                    if cond < reserved_indices {
                        return Some(stack[cond].clone());
                    }

                    match &instr[cond - reserved_indices].0 {
                        Instr::Add(_, args) => {
                            let mut res = Complex::default();
                            for x in args {
                                match resolve(instr, stack, *x, param_count, reserved_indices) {
                                    Some(v) => res += v,
                                    None => return None,
                                }
                            }

                            Some(res)
                        }
                        Instr::Mul(_, args) => {
                            let mut res = Complex::new(Rational::one(), Rational::zero());
                            for x in args {
                                match resolve(instr, stack, *x, param_count, reserved_indices) {
                                    Some(v) => res *= v,
                                    None => return None,
                                }
                            }

                            Some(res)
                        }
                        Instr::Pow(_, base, exp) => {
                            if let Some(base_val) =
                                resolve(instr, stack, *base, param_count, reserved_indices)
                            {
                                if *exp == -1 {
                                    Some(base_val.inv())
                                } else if *exp < 0 {
                                    Some(base_val.pow(exp.unsigned_abs()).inv())
                                } else {
                                    Some(base_val.pow(exp.unsigned_abs()))
                                }
                            } else {
                                None
                            }
                        }
                        _ => None,
                    }
                }

                if let Some(cond_res) =
                    resolve(instr, stack, cond, self.param_count, reserved_indices)
                {
                    // remove dead code
                    instr.truncate(instr_len);
                    stack.truncate(stack_len);
                    if subexpression_len != sub_expr_pos.len() {
                        // remove subexpressions that are created as part of the conditions
                        sub_expr_pos.retain(|_, &mut v| v < reserved_indices + instr_len);
                    }

                    return if !cond_res.is_zero() {
                        self.linearize_impl(
                            &b.1,
                            subexpressions,
                            stack,
                            instr,
                            sub_expr_pos,
                            args,
                            in_branch,
                            reserved_indices,
                        )
                    } else {
                        self.linearize_impl(
                            &b.2,
                            subexpressions,
                            stack,
                            instr,
                            sub_expr_pos,
                            args,
                            in_branch,
                            reserved_indices,
                        )
                    };
                }

                let label_else = Label(instr.len());
                stack.push(Complex::default());
                instr.push((Instr::IfElse(cond, label_else), ComplexPhase::Any));

                let then_branch = self.linearize_impl(
                    &b.1,
                    subexpressions,
                    stack,
                    instr,
                    sub_expr_pos,
                    args,
                    true,
                    reserved_indices,
                );

                let label_end = Label(instr.len());
                stack.push(Complex::default());
                instr.push((Instr::Goto(label_end), ComplexPhase::Any));
                stack.push(Complex::default());
                instr.push((Instr::Label(label_else), ComplexPhase::Any));

                let else_branch = self.linearize_impl(
                    &b.2,
                    subexpressions,
                    stack,
                    instr,
                    sub_expr_pos,
                    args,
                    true,
                    reserved_indices,
                );

                stack.push(Complex::default());
                instr.push((Instr::Label(label_end), ComplexPhase::Any));

                stack.push(Complex::default());
                let res = stack.len() - 1;

                instr.push((
                    Instr::Join(res, cond, then_branch, else_branch),
                    ComplexPhase::Any,
                ));

                res
            }
        }
    }

    /// Find a near-optimal Horner scheme that minimizes the number of multiplications
    /// and additions, using `iterations` iterations of the optimization algorithm
    /// and `n_cores` cores. Optionally, a starting scheme can be provided.
    pub fn optimize(
        mut self,
        settings: &OptimizationSettings,
    ) -> ExpressionEvaluator<Complex<Rational>> {
        if settings.verbose {
            let (n_add, n_mul) = self.count_operations();
            info!(
                "Initial ops: {} additions and {} multiplications",
                n_add, n_mul
            );
        }

        if settings.horner_iterations > 0 {
            let _ = self.optimize_horner_scheme(settings);
        }

        self.common_subexpression_elimination();
        self.linearize(settings)
    }

    /// Write the expressions in a Horner scheme where the variables
    /// are sorted by their occurrence count.
    pub fn horner_scheme(&mut self) {
        for t in &mut self.expressions.tree {
            t.occurrence_order_horner_scheme();
        }

        for e in &mut self.expressions.subexpressions {
            e.occurrence_order_horner_scheme();
        }

        for (_, _, e) in &mut self.functions {
            for t in &mut e.tree {
                t.occurrence_order_horner_scheme();
            }

            for e in &mut e.subexpressions {
                e.occurrence_order_horner_scheme();
            }
        }
    }

    /// Find a near-optimal Horner scheme that minimizes the number of multiplications
    /// and additions, using `iterations` iterations of the optimization algorithm
    /// and `n_cores` cores. Optionally, a starting scheme can be provided.
    pub fn optimize_horner_scheme(
        &mut self,
        settings: &OptimizationSettings,
    ) -> Vec<Expression<Complex<Rational>>> {
        let v = match &settings.hot_start {
            Some(a) => a.clone(),
            None => {
                let mut v = HashMap::default();

                for t in &mut self.expressions.tree {
                    t.find_all_variables(&mut v);
                }

                for e in &mut self.expressions.subexpressions {
                    e.find_all_variables(&mut v);
                }

                let mut v: Vec<_> = v.into_iter().collect();
                v.retain(|(_, vv)| *vv > 1);
                v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                v.truncate(settings.max_horner_scheme_variables);
                v.into_iter().map(|(k, _)| k).collect::<Vec<_>>()
            }
        };

        let scheme =
            Expression::optimize_horner_scheme_multiple(&self.expressions.tree, &v, settings);
        for e in &mut self.expressions.tree {
            e.apply_horner_scheme(&scheme);
        }

        for e in &mut self.expressions.subexpressions {
            e.apply_horner_scheme(&scheme);
        }

        for (name, _, e) in &mut self.functions {
            let mut v = HashMap::default();

            for t in &mut e.tree {
                t.find_all_variables(&mut v);
            }

            for e in &mut e.subexpressions {
                e.find_all_variables(&mut v);
            }

            let mut v: Vec<_> = v.into_iter().collect();
            v.retain(|(_, vv)| *vv > 1);
            v.sort_by_key(|k| Reverse(k.1));
            v.truncate(settings.max_horner_scheme_variables);
            let v = v.into_iter().map(|(k, _)| k).collect::<Vec<_>>();

            if settings.verbose {
                info!(
                    "Optimizing Horner scheme for function {} with {} variables",
                    name,
                    v.len()
                );
            }

            let scheme = Expression::optimize_horner_scheme_multiple(&e.tree, &v, settings);

            for t in &mut e.tree {
                t.apply_horner_scheme(&scheme);
            }

            for e in &mut e.subexpressions {
                e.apply_horner_scheme(&scheme);
            }
        }

        scheme
    }
}

impl Expression<Complex<Rational>> {
    pub fn apply_horner_scheme(&mut self, scheme: &[Expression<Complex<Rational>>]) {
        if scheme.is_empty() {
            return;
        }

        let a = match self {
            Expression::Add(_, a) => a,
            Expression::Eval(_, _, a) => {
                for arg in a {
                    arg.apply_horner_scheme(scheme);
                }
                return;
            }
            Expression::Mul(_, m) => {
                for a in m {
                    a.apply_horner_scheme(scheme);
                }
                return;
            }
            Expression::Pow(_, b) => {
                b.0.apply_horner_scheme(scheme);
                return;
            }
            Expression::Powf(_, b) => {
                b.0.apply_horner_scheme(scheme);
                b.1.apply_horner_scheme(scheme);
                return;
            }
            Expression::BuiltinFun(_, _, b) => {
                b.apply_horner_scheme(scheme);
                return;
            }
            _ => {
                return;
            }
        };

        a.sort();

        let mut max_pow: Option<i64> = None;
        for x in &*a {
            if let Expression::Mul(_, m) = x {
                let mut pow_counter = 0;
                for y in m {
                    if let Expression::Pow(_, p) = y {
                        if p.0 == scheme[0] && p.1 > 0 {
                            pow_counter += p.1;
                        }
                    } else if y == &scheme[0] {
                        pow_counter += 1; // support x*x*x^3 in term
                    }
                }

                if pow_counter > 0 && (max_pow.is_none() || pow_counter < max_pow.unwrap()) {
                    max_pow = Some(pow_counter);
                }
            } else if let Expression::Pow(_, p) = x {
                if p.0 == scheme[0] && p.1 > 0 && (max_pow.is_none() || p.1 < max_pow.unwrap()) {
                    max_pow = Some(p.1);
                }
            } else if x == &scheme[0] {
                max_pow = Some(1);
            }
        }

        // TODO: jump to next variable if the current variable only appears in one factor?
        // this will improve the scheme but may hide common subexpressions?

        let Some(max_pow) = max_pow else {
            return self.apply_horner_scheme(&scheme[1..]);
        };

        // extract GCD and phase of integer coefficients
        // keep rational coefficients untouched to avoid numerical precision issues
        let mut gcd = Complex::new(Rational::zero(), Rational::zero());
        for x in &*a {
            let mut num = None;

            if let Expression::Mul(_, m) = x {
                for y in m {
                    if let Expression::Const(_, c) = y
                        && c.re.is_integer()
                        && c.im.is_integer()
                    {
                        num = Some(c);
                    }
                }
            } else if let Expression::Const(_, c) = x
                && c.re.is_integer()
                && c.im.is_integer()
            {
                num = Some(c);
            }

            if let Some(n) = num {
                if n.im.is_zero() && gcd.im.is_zero() {
                    gcd = Complex::new(gcd.re.gcd(&n.re), Rational::zero());
                } else if n.re.is_zero() && gcd.re.is_zero() {
                    gcd = Complex::new(Rational::zero(), gcd.im.gcd(&n.im));
                } else {
                    gcd = Complex::new_one();
                }
            } else {
                gcd = Complex::new_one();
            }
        }

        if !gcd.is_zero() && !gcd.is_one() {
            for x in &mut *a {
                match x {
                    Expression::Mul(_, m) => {
                        for y in m {
                            if let Expression::Const(_, c) = y {
                                **c /= &gcd;
                                break;
                            }
                        }
                    }
                    Expression::Const(_, c) => {
                        **c /= &gcd;
                    }
                    _ => {
                        unreachable!()
                    }
                }
            }
        }

        let mut contains = vec![];
        let mut rest = vec![];

        for mut x in a.drain(..) {
            let mut found = false;
            if let Expression::Mul(_, m) = &mut x {
                let mut pow_counter = 0;

                m.retain(|y| {
                    if let Expression::Pow(_, p) = y {
                        if p.0 == scheme[0] && p.1 > 0 {
                            pow_counter += p.1;
                            false
                        } else {
                            true
                        }
                    } else if y == &scheme[0] {
                        pow_counter += 1;
                        false
                    } else {
                        true
                    }
                });

                if pow_counter > max_pow {
                    if pow_counter > max_pow + 1 {
                        m.push(
                            Expression::Pow(
                                0,
                                Box::new((scheme[0].clone(), pow_counter - max_pow)),
                            )
                            .rehashed(true),
                        );
                    } else {
                        m.push(scheme[0].clone());
                    }

                    m.sort();
                }

                if m.is_empty() {
                    x = Expression::Const(0, Box::new(Complex::new_one())).rehashed(true);
                } else if m.len() == 1 {
                    x = m.pop().unwrap();
                }

                found = pow_counter > 0;
            } else if let Expression::Pow(_, p) = &mut x {
                if p.0 == scheme[0] && p.1 > 0 {
                    if p.1 > max_pow + 1 {
                        p.1 -= max_pow;
                    } else if p.1 - max_pow == 1 {
                        x = scheme[0].clone();
                    } else {
                        x = Expression::Const(0, Box::new(Complex::new_one())).rehashed(true);
                    }
                    found = true;
                }
            } else if x == scheme[0] {
                found = true;
                x = Expression::Const(0, Box::new(Complex::new_one())).rehashed(true);
            }

            if found {
                contains.push(x);
            } else {
                rest.push(x);
            }
        }

        let extracted = if max_pow == 1 {
            scheme[0].clone()
        } else {
            Expression::Pow(0, Box::new((scheme[0].clone(), max_pow))).rehashed(true)
        };

        let mut contains = if contains.len() == 1 {
            contains.pop().unwrap()
        } else {
            Expression::Add(0, contains).rehashed(true)
        };

        contains.apply_horner_scheme(scheme); // keep trying with same variable

        let mut v = vec![];
        if let Expression::Mul(_, a) = contains {
            v.extend(a);
        } else {
            v.push(contains);
        }

        v.push(extracted);
        v.retain(|x| {
            if let Expression::Const(_, y) = x
                && y.is_one()
            {
                false
            } else {
                true
            }
        });
        v.sort();

        let mut c = if v.len() == 1 {
            v.pop().unwrap()
        } else {
            Expression::Mul(0, v).rehashed(true)
        };

        if rest.is_empty() {
            if !gcd.is_zero() && !gcd.is_one() {
                if let Expression::Mul(_, v) = &mut c {
                    v.push(Expression::Const(0, Box::new(gcd)));
                    v.sort();
                    *self = c.rehashed(true);
                } else {
                    *self = Expression::Mul(0, vec![Expression::Const(0, Box::new(gcd)), c])
                        .rehashed(true);
                }
            } else {
                *self = c.rehashed(true);
            }
        } else {
            let mut r = if rest.len() == 1 {
                rest.pop().unwrap()
            } else {
                Expression::Add(0, rest).rehashed(true)
            };

            r.apply_horner_scheme(&scheme[1..]);

            a.clear();
            a.push(c);

            if let Expression::Add(_, aa) = r {
                a.extend(aa);
            } else {
                a.push(r);
            }

            a.sort();

            if !gcd.is_zero() && !gcd.is_one() {
                *self = Expression::Mul(
                    0,
                    vec![
                        Expression::Const(0, Box::new(gcd)),
                        Expression::Add(0, std::mem::take(a)),
                    ],
                )
                .rehashed(true);
            }
        }
    }

    /// Apply a simple occurrence-order Horner scheme to every addition.
    pub fn occurrence_order_horner_scheme(&mut self) {
        match self {
            Expression::Const(_, _) | Expression::Parameter(_, _) | Expression::ReadArg(_, _) => {}
            Expression::Eval(_, _, ae) => {
                for arg in ae {
                    arg.occurrence_order_horner_scheme();
                }
            }
            Expression::Add(_, a) => {
                for arg in &mut *a {
                    arg.occurrence_order_horner_scheme();
                }

                let mut occurrence = HashMap::default();

                for arg in &*a {
                    match arg {
                        Expression::Mul(_, m) => {
                            for aa in m {
                                if let Expression::Pow(_, p) = aa
                                    && !matches!(p.0, Expression::Const(_, _))
                                {
                                    occurrence
                                        .entry(p.0.clone())
                                        .and_modify(|x| *x += 1)
                                        .or_insert(1);
                                } else if !matches!(aa, Expression::Const(_, _)) {
                                    occurrence
                                        .entry(aa.clone())
                                        .and_modify(|x| *x += 1)
                                        .or_insert(1);
                                }
                            }
                        }
                        x => {
                            if let Expression::Pow(_, p) = x
                                && !matches!(p.0, Expression::Const(_, _))
                            {
                                occurrence
                                    .entry(p.0.clone())
                                    .and_modify(|x| *x += 1)
                                    .or_insert(1);
                            } else if !matches!(x, Expression::Const(_, _)) {
                                occurrence
                                    .entry(x.clone())
                                    .and_modify(|x| *x += 1)
                                    .or_insert(1);
                            }
                        }
                    }
                }

                occurrence.retain(|_, v| *v > 1);
                let mut order: Vec<_> = occurrence.into_iter().collect();
                order.sort_by_key(|k| Reverse(k.1)); // occurrence order
                let scheme = order.into_iter().map(|(k, _)| k).collect::<Vec<_>>();

                self.apply_horner_scheme(&scheme);
            }
            Expression::Mul(_, a) => {
                for arg in a {
                    arg.occurrence_order_horner_scheme();
                }
            }
            Expression::Pow(_, p) => {
                p.0.occurrence_order_horner_scheme();
            }
            Expression::Powf(_, p) => {
                p.0.occurrence_order_horner_scheme();
                p.1.occurrence_order_horner_scheme();
            }
            Expression::BuiltinFun(_, _, a) => {
                a.occurrence_order_horner_scheme();
            }
            Expression::SubExpression(_, _) => {}
            Expression::Fun(_, _, _, a) => {
                for arg in a {
                    arg.occurrence_order_horner_scheme();
                }
            }
            Expression::IfElse(_, b) => {
                b.0.occurrence_order_horner_scheme();
                b.1.occurrence_order_horner_scheme();
                b.2.occurrence_order_horner_scheme();
            }
        }
    }

    pub fn optimize_horner_scheme(
        &self,
        vars: &[Self],
        settings: &OptimizationSettings,
    ) -> Vec<Self> {
        Self::optimize_horner_scheme_multiple(std::slice::from_ref(self), vars, settings)
    }

    pub fn optimize_horner_scheme_multiple(
        expressions: &[Self],
        vars: &[Self],
        settings: &OptimizationSettings,
    ) -> Vec<Self> {
        if vars.is_empty() {
            return vars.to_vec();
        }

        let horner: Vec<_> = expressions
            .iter()
            .map(|x| {
                let mut h = x.clone();
                h.apply_horner_scheme(vars);
                h.rehashed(true)
            })
            .collect();
        let mut subexpr = HashMap::default();
        let mut best_ops = (0, 0);
        for h in &horner {
            let ops = h.count_operations_with_subexpression(&mut subexpr);
            best_ops = (best_ops.0 + ops.0, best_ops.1 + ops.1);
        }

        if settings.verbose {
            info!(
                "Initial Horner scheme ops: {} additions and {} multiplications",
                best_ops.0, best_ops.1
            );
        }

        let best_mul = Arc::new(AtomicUsize::new(best_ops.1));
        let best_add = Arc::new(AtomicUsize::new(best_ops.0));
        let best_scheme = Arc::new(Mutex::new(vars.to_vec()));

        let n_iterations = settings.horner_iterations.max(1) - 1;

        let permutations = if vars.len() < 10
            && Integer::factorial(vars.len() as u32) <= settings.horner_iterations.max(1)
        {
            let v: Vec<_> = (0..vars.len()).collect();
            Some(unique_permutations(&v).1)
        } else {
            None
        };
        let p_ref = &permutations;

        let n_cores = if LicenseManager::is_licensed() {
            settings.n_cores
        } else {
            1
        }
        .min(n_iterations);

        std::thread::scope(|s| {
            let abort = Arc::new(AtomicBool::new(false));

            for i in 0..n_cores {
                let mut rng = MonteCarloRng::new(0, i);

                let mut cvars = vars.to_vec();
                let best_scheme = best_scheme.clone();
                let best_mul = best_mul.clone();
                let best_add = best_add.clone();
                let mut last_mul = usize::MAX;
                let mut last_add = usize::MAX;
                let abort = abort.clone();

                let mut op = move || {
                    for j in 0..n_iterations / n_cores {
                        if abort.load(Ordering::Relaxed) {
                            return;
                        }

                        if i == n_cores - 1
                            && let Some(a) = &settings.abort_check
                            && a()
                        {
                            abort.store(true, Ordering::Relaxed);

                            if settings.verbose {
                                info!(
                                    "Aborting Horner optimization at step {}/{}.",
                                    j,
                                    settings.horner_iterations / n_cores
                                );
                            }

                            return;
                        }

                        // try a random swap
                        let mut t1 = 0;
                        let mut t2 = 0;

                        if let Some(p) = p_ref {
                            if j >= p.len() / n_cores {
                                break;
                            }

                            let perm = &p[i * (p.len() / n_cores) + j];
                            cvars = perm.iter().map(|x| vars[*x].clone()).collect();
                        } else {
                            t1 = rng.random_range(0..cvars.len());
                            t2 = rng.random_range(0..cvars.len() - 1);

                            cvars.swap(t1, t2);
                        }

                        let horner: Vec<_> = expressions
                            .iter()
                            .map(|x| {
                                let mut h = x.clone();
                                h.apply_horner_scheme(&cvars);
                                h.rehash(true);
                                h
                            })
                            .collect();
                        let mut subexpr = HashMap::default();
                        let mut cur_ops = (0, 0);

                        for h in &horner {
                            let ops = h.count_operations_with_subexpression(&mut subexpr);
                            cur_ops = (cur_ops.0 + ops.0, cur_ops.1 + ops.1);
                        }

                        // prefer fewer multiplications
                        if cur_ops.1 <= last_mul || cur_ops.1 == last_mul && cur_ops.0 <= last_add {
                            if settings.verbose {
                                info!(
                                    "Accept move at step {}/{}: {} + and {} ×",
                                    j,
                                    settings.horner_iterations / n_cores,
                                    cur_ops.0,
                                    cur_ops.1
                                );
                            }

                            last_add = cur_ops.0;
                            last_mul = cur_ops.1;

                            if cur_ops.1 <= best_mul.load(Ordering::Relaxed)
                                || cur_ops.1 == best_mul.load(Ordering::Relaxed)
                                    && cur_ops.0 <= best_add.load(Ordering::Relaxed)
                            {
                                let mut best_scheme = best_scheme.lock().unwrap();

                                // check again if it is the best now that we have locked
                                let best_mul_l = best_mul.load(Ordering::Relaxed);
                                let best_add_l = best_add.load(Ordering::Relaxed);
                                if cur_ops.1 <= best_mul_l
                                    || cur_ops.1 == best_mul_l && cur_ops.0 <= best_add_l
                                {
                                    if cur_ops.0 == best_add_l && cur_ops.1 == best_mul_l {
                                        if *best_scheme < cvars {
                                            // on a draw, accept the lexicographical minimum
                                            // to get a deterministic scheme
                                            *best_scheme = cvars.clone();
                                        }
                                    } else {
                                        best_mul.store(cur_ops.1, Ordering::Relaxed);
                                        best_add.store(cur_ops.0, Ordering::Relaxed);
                                        *best_scheme = cvars.clone();
                                    }
                                }
                            }
                        } else {
                            cvars.swap(t1, t2);
                        }
                    }
                };

                if i + 1 < n_cores {
                    s.spawn(op);
                } else {
                    // execute in the main thread and do the abort check on the main thread
                    // this helps with catching ctrl-c
                    op()
                }
            }
        });

        if settings.verbose {
            info!(
                "Final scheme: {} + and {} ×",
                best_add.load(Ordering::Relaxed),
                best_mul.load(Ordering::Relaxed)
            );
        }

        Arc::try_unwrap(best_scheme).unwrap().into_inner().unwrap()
    }

    fn find_all_variables(&self, vars: &mut HashMap<Expression<Complex<Rational>>, usize>) {
        match self {
            Expression::Const(_, _) | Expression::Parameter(_, _) | Expression::ReadArg(_, _) => {}
            Expression::Eval(_, _, ae) => {
                for arg in ae {
                    arg.find_all_variables(vars);
                }
            }
            Expression::Add(_, a) => {
                for arg in a {
                    arg.find_all_variables(vars);
                }

                for arg in a {
                    match arg {
                        Expression::Mul(_, m) => {
                            for aa in m {
                                if let Expression::Pow(_, p) = aa
                                    && !matches!(p.0, Expression::Const(_, _))
                                {
                                    vars.entry(p.0.clone()).and_modify(|x| *x += 1).or_insert(1);
                                } else if !matches!(aa, Expression::Const(_, _)) {
                                    vars.entry(aa.clone()).and_modify(|x| *x += 1).or_insert(1);
                                }
                            }
                        }
                        x => {
                            if let Expression::Pow(_, p) = x
                                && !matches!(p.0, Expression::Const(_, _))
                            {
                                vars.entry(p.0.clone()).and_modify(|x| *x += 1).or_insert(1);
                            } else if !matches!(x, Expression::Const(_, _)) {
                                vars.entry(x.clone()).and_modify(|x| *x += 1).or_insert(1);
                            }
                        }
                    }
                }
            }
            Expression::Mul(_, a) => {
                for arg in a {
                    arg.find_all_variables(vars);
                }
            }
            Expression::Pow(_, p) => {
                p.0.find_all_variables(vars);
            }
            Expression::Powf(_, p) => {
                p.0.find_all_variables(vars);
                p.1.find_all_variables(vars);
            }
            Expression::BuiltinFun(_, _, a) => {
                a.find_all_variables(vars);
            }
            Expression::SubExpression(_, _) => {}
            Expression::Fun(_, _, _, a) => {
                for arg in a {
                    arg.find_all_variables(vars);
                }
            }
            Expression::IfElse(_, b) => {
                b.0.find_all_variables(vars);
                b.1.find_all_variables(vars);
                b.2.find_all_variables(vars);
            }
        }
    }
}

impl<T: Clone + Default + std::fmt::Debug + Eq + std::hash::Hash + InternalOrdering> EvalTree<T> {
    pub fn common_subexpression_elimination(&mut self) {
        self.expressions.common_subexpression_elimination();

        for (_, _, e) in &mut self.functions {
            e.common_subexpression_elimination();
        }
    }

    pub fn count_operations(&self) -> (usize, usize) {
        let mut add = 0;
        let mut mul = 0;
        for e in &self.functions {
            let (ea, em) = e.2.count_operations();
            add += ea;
            mul += em;
        }

        let (ea, em) = self.expressions.count_operations();
        (add + ea, mul + em)
    }
}

impl<T: Clone + Default + std::fmt::Debug + Eq + std::hash::Hash + InternalOrdering>
    SplitExpression<T>
{
    /// Eliminate common subexpressions in the expression, also checking for subexpressions
    /// up to length `max_subexpr_len`.
    pub fn common_subexpression_elimination(&mut self) {
        let mut subexpression_count = HashMap::default();

        for t in &mut self.tree {
            t.rehash(true);
            t.find_subexpression(&mut subexpression_count);
        }

        subexpression_count.retain(|_, v| *v > 1);

        let mut v: Vec<_> = subexpression_count
            .iter()
            .map(|(k, v)| (*v, (*k).clone()))
            .collect();
        v.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        // assign a unique index to every subexpression
        let mut h = HashMap::default();
        for (index, (_, e)) in v.iter().enumerate() {
            h.insert(&*e, self.subexpressions.len() + index);
        }

        for t in &mut self.tree {
            t.replace_subexpression(&h, false);
        }

        // replace subexpressions in subexpressions and
        // sort them based on their dependencies
        for (_, x) in &v {
            let mut he = x.clone();
            he.replace_subexpression(&h, true);
            self.subexpressions.push(he);
        }

        let mut dep_tree = vec![];
        for (i, s) in self.subexpressions.iter().enumerate() {
            let mut deps = vec![];
            s.get_dependent_subexpressions(&mut deps);
            dep_tree.push((i, deps.clone()));
        }

        let mut rename = HashMap::default();
        let mut new_subs = vec![];
        let mut i = 0;
        while !dep_tree.is_empty() {
            if dep_tree[i].1.iter().all(|x| rename.contains_key(x)) {
                rename.insert(dep_tree[i].0, new_subs.len());
                new_subs.push(self.subexpressions[dep_tree[i].0].clone());
                dep_tree.swap_remove(i);
                if i == dep_tree.len() {
                    i = 0;
                }
            } else {
                i = (i + 1) % dep_tree.len();
            }
        }

        for x in &mut new_subs {
            x.rename_subexpression(&rename);
        }
        for t in &mut self.tree {
            t.rename_subexpression(&rename);
        }

        self.subexpressions = new_subs;
    }
}

impl<T: Clone + Default + std::fmt::Debug + Eq + std::hash::Hash + InternalOrdering> Expression<T> {
    fn rename_subexpression(&mut self, subexp: &HashMap<usize, usize>) {
        match self {
            Expression::Const(_, _) | Expression::Parameter(_, _) | Expression::ReadArg(_, _) => {}
            Expression::Eval(_, _, ae) => {
                for arg in &mut *ae {
                    arg.rename_subexpression(subexp);
                }
            }
            Expression::Add(_, a) | Expression::Mul(_, a) => {
                for arg in &mut *a {
                    arg.rename_subexpression(subexp);
                }

                a.sort();
            }
            Expression::Pow(_, p) => {
                p.0.rename_subexpression(subexp);
            }
            Expression::Powf(_, p) => {
                p.0.rename_subexpression(subexp);
                p.1.rename_subexpression(subexp);
            }
            Expression::BuiltinFun(_, _, a) => {
                a.rename_subexpression(subexp);
            }
            Expression::SubExpression(h, i) => {
                *self = Expression::SubExpression(*h, *subexp.get(i).unwrap());
            }
            Expression::Fun(_, _, _, a) => {
                for arg in a {
                    arg.rename_subexpression(subexp);
                }
            }
            Expression::IfElse(_, b) => {
                b.0.rename_subexpression(subexp);
                b.1.rename_subexpression(subexp);
                b.2.rename_subexpression(subexp);
            }
        }
    }

    fn get_dependent_subexpressions(&self, dep: &mut Vec<usize>) {
        match self {
            Expression::Const(_, _) | Expression::Parameter(_, _) | Expression::ReadArg(_, _) => {}
            Expression::Eval(_, _, ae) => {
                for arg in ae {
                    arg.get_dependent_subexpressions(dep);
                }
            }
            Expression::Add(_, a) | Expression::Mul(_, a) => {
                for arg in a {
                    arg.get_dependent_subexpressions(dep);
                }
            }
            Expression::Pow(_, p) => {
                p.0.get_dependent_subexpressions(dep);
            }
            Expression::Powf(_, p) => {
                p.0.get_dependent_subexpressions(dep);
                p.1.get_dependent_subexpressions(dep);
            }
            Expression::BuiltinFun(_, _, a) => {
                a.get_dependent_subexpressions(dep);
            }
            Expression::SubExpression(_, i) => {
                dep.push(*i);
            }
            Expression::Fun(_, _, _, a) => {
                for arg in a {
                    arg.get_dependent_subexpressions(dep);
                }
            }
            Expression::IfElse(_, b) => {
                b.0.get_dependent_subexpressions(dep);
                b.1.get_dependent_subexpressions(dep);
                b.2.get_dependent_subexpressions(dep);
            }
        }
    }
}

impl<T: Clone + Default + std::fmt::Debug + Eq + std::hash::Hash + InternalOrdering>
    SplitExpression<T>
{
    pub fn count_operations(&self) -> (usize, usize) {
        let mut add = 0;
        let mut mul = 0;
        for e in &self.subexpressions {
            let (ea, em) = e.count_operations();
            add += ea;
            mul += em;
        }

        for e in &self.tree {
            let (ea, em) = e.count_operations();
            add += ea;
            mul += em;
        }

        (add, mul)
    }
}

impl<T: Clone + Default + std::fmt::Debug + Eq + std::hash::Hash + InternalOrdering> Expression<T> {
    // Count the number of additions and multiplications in the expression.
    pub fn count_operations(&self) -> (usize, usize) {
        match self {
            Expression::Const(_, _) => (0, 0),
            Expression::Parameter(_, _) => (0, 0),
            Expression::Eval(_, _, args) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in args {
                    let (a, m) = arg.count_operations();
                    add += a;
                    mul += m;
                }
                (add, mul)
            }
            Expression::Add(_, a) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in a {
                    let (a, m) = arg.count_operations();
                    add += a;
                    mul += m;
                }
                (add + a.len() - 1, mul)
            }
            Expression::Mul(_, m) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in m {
                    let (a, m) = arg.count_operations();
                    add += a;
                    mul += m;
                }
                (add, mul + m.len() - 1)
            }
            Expression::Pow(_, p) => {
                let (a, m) = p.0.count_operations();
                (a, m + p.1.unsigned_abs() as usize - 1)
            }
            Expression::Powf(_, p) => {
                let (a, m) = p.0.count_operations();
                let (a2, m2) = p.1.count_operations();
                (a + a2, m + m2 + 1) // not clear how to count this
            }
            Expression::ReadArg(_, _) => (0, 0),
            Expression::BuiltinFun(_, _, b) => b.count_operations(), // not clear how to count this, third arg?
            Expression::SubExpression(_, _) => (0, 0),
            Expression::Fun(_, _, _, args) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in args {
                    let (a, m) = arg.count_operations();
                    add += a;
                    mul += m;
                }
                (add, mul)
            }
            Expression::IfElse(_, b) => {
                let (a1, m1) = b.0.count_operations();
                let (a2, m2) = b.1.count_operations();
                let (a3, m3) = b.2.count_operations();
                (a1 + a2 + a3, m1 + m2 + m3)
            }
        }
    }
}

impl<T: Real> EvalTree<T> {
    /// Evaluate the evaluation tree. Consider converting to a linear form for repeated evaluation.
    pub fn evaluate(&mut self, params: &[T], out: &mut [T]) {
        for (o, e) in out.iter_mut().zip(&self.expressions.tree) {
            *o = self.evaluate_impl(e, &self.expressions.subexpressions, params, &[])
        }
    }

    fn evaluate_impl(
        &self,
        expr: &Expression<T>,
        subexpressions: &[Expression<T>],
        params: &[T],
        args: &[T],
    ) -> T {
        match expr {
            Expression::Const(_, c) => c.as_ref().clone(),
            Expression::Parameter(_, p) => params[*p].clone(),
            Expression::Eval(_, f, e_args) => {
                let mut arg_buf = vec![T::new_zero(); e_args.len()];
                for (b, a) in arg_buf.iter_mut().zip(e_args.iter()) {
                    *b = self.evaluate_impl(a, subexpressions, params, args);
                }

                let func = &self.functions[*f as usize].2;
                self.evaluate_impl(&func.tree[0], &func.subexpressions, params, &arg_buf)
            }
            Expression::Add(_, a) => {
                let mut r = self.evaluate_impl(&a[0], subexpressions, params, args);
                for arg in &a[1..] {
                    r += self.evaluate_impl(arg, subexpressions, params, args);
                }
                r
            }
            Expression::Mul(_, m) => {
                let mut r = self.evaluate_impl(&m[0], subexpressions, params, args);
                for arg in &m[1..] {
                    r *= self.evaluate_impl(arg, subexpressions, params, args);
                }
                r
            }
            Expression::Pow(_, p) => {
                let (b, e) = &**p;
                let b_eval = self.evaluate_impl(b, subexpressions, params, args);

                if *e == -1 {
                    b_eval.inv()
                } else if *e >= 0 {
                    b_eval.pow(*e as u64)
                } else {
                    b_eval.pow(e.unsigned_abs()).inv()
                }
            }
            Expression::Powf(_, p) => {
                let (b, e) = &**p;
                let b_eval = self.evaluate_impl(b, subexpressions, params, args);
                let e_eval = self.evaluate_impl(e, subexpressions, params, args);
                b_eval.powf(&e_eval)
            }
            Expression::ReadArg(_, i) => args[*i].clone(),
            Expression::BuiltinFun(_, s, a) => {
                let arg = self.evaluate_impl(a, subexpressions, params, args);
                match s.get_id() {
                    Symbol::EXP_ID => arg.exp(),
                    Symbol::LOG_ID => arg.log(),
                    Symbol::SIN_ID => arg.sin(),
                    Symbol::COS_ID => arg.cos(),
                    Symbol::SQRT_ID => arg.sqrt(),
                    Symbol::ABS_ID => arg.norm(),
                    Symbol::CONJ_ID => arg.conj(),
                    _ => unreachable!(),
                }
            }
            Expression::SubExpression(_, s) => {
                // TODO: cache
                self.evaluate_impl(&subexpressions[*s], subexpressions, params, args)
            }
            Expression::Fun(_, name, _, _args) => {
                unimplemented!(
                    "External function calls not implemented for EvalTree: {}",
                    name
                );
            }
            Expression::IfElse(_, b) => {
                let cond = self.evaluate_impl(&b.0, subexpressions, params, args);
                if !cond.is_fully_zero() {
                    self.evaluate_impl(&b.1, subexpressions, params, args)
                } else {
                    self.evaluate_impl(&b.2, subexpressions, params, args)
                }
            }
        }
    }
}

/// Represents exported code that can be compiled with [Self::compile].
impl<'a> AtomView<'a> {
    /// Convert nested expressions to a tree.
    pub fn to_evaluation_tree(
        &self,
        fn_map: &FunctionMap,
        params: &[Atom],
    ) -> Result<EvalTree<Complex<Rational>>, String> {
        Self::to_eval_tree_multiple(std::slice::from_ref(self), fn_map, params)
    }

    /// Convert nested expressions to a tree.
    pub fn to_eval_tree_multiple<A: AtomCore>(
        exprs: &[A],
        fn_map: &FunctionMap,
        params: &[Atom],
    ) -> Result<EvalTree<Complex<Rational>>, String> {
        let mut funcs = vec![];
        let mut func_id_to_index = HashMap::default();
        let mut external_functions = vec![];

        let tree = exprs
            .iter()
            .map(|t| {
                t.as_atom_view()
                    .to_eval_tree_impl(
                        fn_map,
                        params,
                        &[],
                        &mut func_id_to_index,
                        &mut funcs,
                        &mut external_functions,
                    )
                    .map(|x| x.rehashed(true))
            })
            .collect::<Result<_, _>>()?;

        Ok(EvalTree {
            expressions: SplitExpression {
                tree,
                subexpressions: vec![],
            },
            functions: funcs,
            external_functions,
            param_count: params.len(),
        })
    }

    fn to_eval_tree_impl(
        &self,
        fn_map: &FunctionMap,
        params: &[Atom],
        args: &[Indeterminate],
        fn_id_map: &mut HashMap<usize, usize>,
        funcs: &mut Vec<(
            String,
            Vec<Indeterminate>,
            SplitExpression<Complex<Rational>>,
        )>,
        external_functions: &mut Vec<ExternalFunctionContainer<Complex<Rational>>>,
    ) -> Result<Expression<Complex<Rational>>, String> {
        if matches!(self, AtomView::Var(_) | AtomView::Fun(_)) {
            if let Some(p) = args.iter().position(|s| *self == s.as_view()) {
                return Ok(Expression::ReadArg(0, p));
            }

            if let Some(p) = params.iter().position(|a| a.as_view() == *self) {
                return Ok(Expression::Parameter(0, p));
            }
        }

        match self {
            AtomView::Num(n) => match n.get_coeff_view() {
                CoefficientView::Natural(n, d, ni, di) => Ok(Expression::Const(
                    0,
                    Box::new(Complex::new(
                        Rational::from((n, d)),
                        Rational::from((ni, di)),
                    )),
                )),
                CoefficientView::Large(l, i) => Ok(Expression::Const(
                    0,
                    Box::new(Complex::new(l.to_rat(), i.to_rat())),
                )),
                CoefficientView::Float(r, i) => {
                    // TODO: converting back to rational is slow
                    Ok(Expression::Const(
                        0,
                        Box::new(Complex::new(
                            r.to_float().to_rational(),
                            i.to_float().to_rational(),
                        )),
                    ))
                }
                CoefficientView::Indeterminate => {
                    panic!("Cannot convert indeterminate")
                }
                CoefficientView::Infinity(_) => {
                    panic!("Cannot convert infinity")
                }
                CoefficientView::FiniteField(_, _) => {
                    Err("Finite field not yet supported for evaluation".to_string())
                }
                CoefficientView::RationalPolynomial(_) => Err(
                    "Rational polynomial coefficient not yet supported for evaluation".to_string(),
                ),
            },
            AtomView::Var(v) => Err(format!(
                "Variable {} not in constant map",
                v.get_symbol().get_name()
            )),
            AtomView::Fun(f) => {
                let name = f.get_symbol();
                if [
                    Symbol::EXP_ID,
                    Symbol::LOG_ID,
                    Symbol::SIN_ID,
                    Symbol::COS_ID,
                    Symbol::SQRT_ID,
                    Symbol::ABS_ID,
                    Symbol::CONJ_ID,
                ]
                .contains(&name.get_id())
                {
                    assert!(f.get_nargs() == 1);
                    let arg = f.iter().next().unwrap();
                    let arg_eval = arg.to_eval_tree_impl(
                        fn_map,
                        params,
                        args,
                        fn_id_map,
                        funcs,
                        external_functions,
                    )?;

                    return Ok(Expression::BuiltinFun(
                        0,
                        f.get_symbol(),
                        Box::new(arg_eval),
                    ));
                }

                if name == Symbol::IF {
                    if f.get_nargs() != 3 {
                        return Err(format!(
                            "Condition function called with wrong number of arguments: {} vs 3",
                            f.get_nargs(),
                        ));
                    }

                    let mut arg_iter = f.iter();
                    let cond_eval = arg_iter.next().unwrap().to_eval_tree_impl(
                        fn_map,
                        params,
                        args,
                        fn_id_map,
                        funcs,
                        external_functions,
                    )?;

                    if let Expression::Const(0, c) = &cond_eval {
                        if !c.is_zero() {
                            return arg_iter.next().unwrap().to_eval_tree_impl(
                                fn_map,
                                params,
                                args,
                                fn_id_map,
                                funcs,
                                external_functions,
                            );
                        }

                        let _ = arg_iter.next().unwrap();
                        return arg_iter.next().unwrap().to_eval_tree_impl(
                            fn_map,
                            params,
                            args,
                            fn_id_map,
                            funcs,
                            external_functions,
                        );
                    }

                    let t_eval = arg_iter.next().unwrap().to_eval_tree_impl(
                        fn_map,
                        params,
                        args,
                        fn_id_map,
                        funcs,
                        external_functions,
                    )?;
                    let f_eval = arg_iter.next().unwrap().to_eval_tree_impl(
                        fn_map,
                        params,
                        args,
                        fn_id_map,
                        funcs,
                        external_functions,
                    )?;

                    return Ok(Expression::IfElse(0, Box::new((cond_eval, t_eval, f_eval))));
                }

                if let Some(Expr {
                    id,
                    tag_len,
                    args: arg_spec,
                    body: e,
                }) = fn_map.get(*self)
                {
                    return {
                        if f.get_nargs() != arg_spec.len() + tag_len {
                            return Err(format!(
                                "Function {} called with wrong number of arguments: {} vs {}",
                                f.get_symbol().get_name(),
                                f.get_nargs(),
                                arg_spec.len() + tag_len
                            ));
                        }

                        let eval_args = f
                            .iter()
                            .skip(*tag_len)
                            .map(|arg| {
                                arg.to_eval_tree_impl(
                                    fn_map,
                                    params,
                                    args,
                                    fn_id_map,
                                    funcs,
                                    external_functions,
                                )
                            })
                            .collect::<Result<_, _>>()?;

                        if let Some(pos) = fn_id_map.get(id) {
                            Ok(Expression::Eval(0, *pos as u32, eval_args))
                        } else {
                            let r = e.as_view().to_eval_tree_impl(
                                fn_map,
                                params,
                                arg_spec,
                                fn_id_map,
                                funcs,
                                external_functions,
                            )?;
                            funcs.push((
                                name.get_name().to_owned(),
                                arg_spec.clone(),
                                SplitExpression {
                                    tree: vec![r.clone()],
                                    subexpressions: vec![],
                                },
                            ));
                            fn_id_map.insert(*id, funcs.len() - 1);
                            Ok(Expression::Eval(0, funcs.len() as u32 - 1, eval_args))
                        }
                    };
                }

                if let Some(eval_info) = name.get_evaluation_info() {
                    let tags = f
                        .iter()
                        .take(eval_info.get_tag_count())
                        .map(|x| x.to_canonical_string())
                        .collect::<Vec<_>>();
                    let tag_atoms = tags.iter().map(|x| crate::parse!(x)).collect::<Vec<_>>();

                    if external_functions
                        .iter()
                        .all(|x| x.symbol != name || x.tags != tag_atoms)
                    {
                        external_functions.push(ExternalFunctionContainer::new(name, tag_atoms));
                    }

                    let eval_args = f
                        .iter()
                        .skip(eval_info.get_tag_count())
                        .map(|arg| {
                            arg.to_eval_tree_impl(
                                fn_map,
                                params,
                                args,
                                fn_id_map,
                                funcs,
                                external_functions,
                            )
                        })
                        .collect::<Result<_, _>>()?;
                    return Ok(Expression::Fun(0, name, tags, eval_args));
                }

                Err(format!("Undefined function {}", self.to_plain_string()))
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();
                let b_eval = b.to_eval_tree_impl(
                    fn_map,
                    params,
                    args,
                    fn_id_map,
                    funcs,
                    external_functions,
                )?;

                if let AtomView::Num(n) = e
                    && let CoefficientView::Natural(num, den, num_i, _den_i) = n.get_coeff_view()
                    && den == 1
                    && num_i == 0
                {
                    return Ok(Expression::Pow(0, Box::new((b_eval.clone(), num))));
                }

                let e_eval = e.to_eval_tree_impl(
                    fn_map,
                    params,
                    args,
                    fn_id_map,
                    funcs,
                    external_functions,
                )?;
                Ok(Expression::Powf(0, Box::new((b_eval, e_eval))))
            }
            AtomView::Mul(m) => {
                let mut muls = vec![];
                for arg in m.iter() {
                    let a = arg.to_eval_tree_impl(
                        fn_map,
                        params,
                        args,
                        fn_id_map,
                        funcs,
                        external_functions,
                    )?;
                    if let Expression::Mul(0, m) = a {
                        muls.extend(m);
                    } else {
                        muls.push(a);
                    }
                }

                muls.sort();

                Ok(Expression::Mul(0, muls))
            }
            AtomView::Add(a) => {
                let mut adds = vec![];
                for arg in a.iter() {
                    adds.push(arg.to_eval_tree_impl(
                        fn_map,
                        params,
                        args,
                        fn_id_map,
                        funcs,
                        external_functions,
                    )?);
                }

                adds.sort();

                Ok(Expression::Add(0, adds))
            }
        }
    }

    /// Evaluate an expression using a constant map and a function map.
    /// The constant map can map any literal expression to a value, for example
    /// a variable or a function with fixed arguments.
    ///
    /// All variables and all user functions in the expression must occur in the map.
    pub(crate) fn evaluate<A: AtomCore + KeyLookup, T: Real, F: Fn(&Rational) -> T + Copy>(
        &self,
        coeff_map: F,
        const_map: &HashMap<A, T>,
        function_map: &HashMap<Symbol, EvaluationFn<A, T>>,
    ) -> Result<T, String> {
        let mut cache = HashMap::default();
        self.evaluate_impl(coeff_map, const_map, function_map, &mut cache)
    }

    fn evaluate_impl<A: AtomCore + KeyLookup, T: Real, F: Fn(&Rational) -> T + Copy>(
        &self,
        coeff_map: F,
        const_map: &HashMap<A, T>,
        function_map: &HashMap<Symbol, EvaluationFn<A, T>>,
        cache: &mut HashMap<AtomView<'a>, T>,
    ) -> Result<T, String> {
        if let Some(c) = const_map.get(self.get_data()) {
            return Ok(c.clone());
        }

        match self {
            AtomView::Num(n) => match n.get_coeff_view() {
                CoefficientView::Natural(n, d, ni, di) => {
                    if ni == 0 {
                        Ok(coeff_map(&Rational::from_int_unchecked(n, d)))
                    } else {
                        let num = coeff_map(&Rational::from_int_unchecked(n, d));
                        Ok(coeff_map(&Rational::from_int_unchecked(ni, di))
                            * num.i().ok_or_else(|| {
                                "Numerical type does not support imaginary unit".to_string()
                            })?
                            + num)
                    }
                }
                CoefficientView::Large(l, i) => {
                    if i.is_zero() {
                        Ok(coeff_map(&l.to_rat()))
                    } else {
                        let num = coeff_map(&l.to_rat());
                        Ok(coeff_map(&i.to_rat())
                            * num.i().ok_or_else(|| {
                                "Numerical type does not support imaginary unit".to_string()
                            })?
                            + num)
                    }
                }
                CoefficientView::Float(r, i) => {
                    // TODO: converting back to rational is slow
                    let rm = coeff_map(&r.to_float().to_rational());
                    if i.is_zero() {
                        Ok(rm)
                    } else {
                        Ok(coeff_map(&i.to_float().to_rational())
                            * rm.i().ok_or_else(|| {
                                "Numerical type does not support imaginary unit".to_string()
                            })?
                            + rm)
                    }
                }
                CoefficientView::Indeterminate => Err("Cannot evaluate indeterminate".to_string()),
                CoefficientView::Infinity(_) => Err("Cannot evaluate infinity".to_string()),
                CoefficientView::FiniteField(_, _) => {
                    Err("Finite field not yet supported for evaluation".to_string())
                }
                CoefficientView::RationalPolynomial(_) => Err(
                    "Rational polynomial coefficient not yet supported for evaluation".to_string(),
                ),
            },
            AtomView::Var(v) => {
                let s = v.get_symbol();
                match s.get_id() {
                    Symbol::E_ID => Ok(coeff_map(&1.into()).e()),
                    Symbol::PI_ID => Ok(coeff_map(&1.into()).pi()),
                    _ => {
                        if let Some(fun) = function_map.get(&s) {
                            if let Some(eval) = cache.get(self) {
                                return Ok(eval.clone());
                            }

                            let eval = fun.get()(&[], const_map, function_map, cache);
                            cache.insert(*self, eval.clone());
                            Ok(eval)
                        } else {
                            Err(format!(
                                "Variable {} not in constant map or function map",
                                v.get_symbol().get_name()
                            ))
                        }
                    }
                }
            }
            AtomView::Fun(f) => {
                let name = f.get_symbol();
                if [
                    Symbol::EXP_ID,
                    Symbol::LOG_ID,
                    Symbol::SIN_ID,
                    Symbol::COS_ID,
                    Symbol::SQRT_ID,
                    Symbol::ABS_ID,
                    Symbol::CONJ_ID,
                ]
                .contains(&name.get_id())
                {
                    assert!(f.get_nargs() == 1);
                    let arg = f.iter().next().unwrap();
                    let arg_eval = arg.evaluate_impl(coeff_map, const_map, function_map, cache)?;

                    return Ok(match f.get_symbol_id() {
                        Symbol::EXP_ID => arg_eval.exp(),
                        Symbol::LOG_ID => arg_eval.log(),
                        Symbol::SIN_ID => arg_eval.sin(),
                        Symbol::COS_ID => arg_eval.cos(),
                        Symbol::SQRT_ID => arg_eval.sqrt(),
                        Symbol::ABS_ID => arg_eval.norm(),
                        Symbol::CONJ_ID => arg_eval.conj(),
                        _ => unreachable!(),
                    });
                }

                if name == Symbol::IF {
                    if f.get_nargs() != 3 {
                        return Err(format!(
                            "Condition function called with wrong number of arguments: {} vs 3",
                            f.get_nargs(),
                        ));
                    }

                    let mut arg_iter = f.iter();

                    let cond_eval = arg_iter.next().unwrap().evaluate_impl(
                        coeff_map,
                        const_map,
                        function_map,
                        cache,
                    )?;

                    if !cond_eval.is_fully_zero() {
                        let t_eval = arg_iter.next().unwrap().evaluate_impl(
                            coeff_map,
                            const_map,
                            function_map,
                            cache,
                        )?;
                        return Ok(t_eval);
                    } else {
                        let _ = arg_iter.next().unwrap();
                        let f_eval = arg_iter.next().unwrap().evaluate_impl(
                            coeff_map,
                            const_map,
                            function_map,
                            cache,
                        )?;
                        return Ok(f_eval);
                    }
                }

                if let Some(eval) = cache.get(self) {
                    return Ok(eval.clone());
                }

                let mut args = Vec::with_capacity(f.get_nargs());
                for arg in f {
                    args.push(arg.evaluate_impl(coeff_map, const_map, function_map, cache)?);
                }

                let Some(fun) = function_map.get(&f.get_symbol()) else {
                    Err(format!("Missing function {}", f.get_symbol().get_name()))?
                };
                let eval = fun.get()(&args, const_map, function_map, cache);

                cache.insert(*self, eval.clone());
                Ok(eval)
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();
                let b_eval = b.evaluate_impl(coeff_map, const_map, function_map, cache)?;

                if let AtomView::Num(n) = e
                    && let CoefficientView::Natural(num, den, ni, _di) = n.get_coeff_view()
                    && den == 1
                    && ni == 0
                {
                    if num == -1 {
                        return Ok(b_eval.inv());
                    } else if num >= 0 {
                        return Ok(b_eval.pow(num as u64));
                    } else {
                        return Ok(b_eval.pow(num.unsigned_abs()).inv());
                    }
                }

                let e_eval = e.evaluate_impl(coeff_map, const_map, function_map, cache)?;
                Ok(b_eval.powf(&e_eval))
            }
            AtomView::Mul(m) => {
                let mut it = m.iter();
                let mut r =
                    it.next()
                        .unwrap()
                        .evaluate_impl(coeff_map, const_map, function_map, cache)?;
                for arg in it {
                    r *= arg.evaluate_impl(coeff_map, const_map, function_map, cache)?;
                }
                Ok(r)
            }
            AtomView::Add(a) => {
                let mut it = a.iter();
                let mut r =
                    it.next()
                        .unwrap()
                        .evaluate_impl(coeff_map, const_map, function_map, cache)?;
                for arg in it {
                    r += arg.evaluate_impl(coeff_map, const_map, function_map, cache)?;
                }
                Ok(r)
            }
        }
    }

    /// Check if the expression could be 0, using (potentially) numerical sampling with
    /// a given tolerance and number of iterations.
    pub fn zero_test(&self, iterations: usize, tolerance: f64) -> ConditionResult {
        match self {
            AtomView::Num(num_view) => {
                if num_view.is_zero() {
                    ConditionResult::True
                } else {
                    ConditionResult::False
                }
            }
            AtomView::Var(_) => ConditionResult::False,
            AtomView::Fun(_) => ConditionResult::False,
            AtomView::Pow(p) => p.get_base().zero_test(iterations, tolerance),
            AtomView::Mul(mul_view) => {
                let mut is_zero = ConditionResult::False;
                for arg in mul_view {
                    match arg.zero_test(iterations, tolerance) {
                        ConditionResult::True => return ConditionResult::True,
                        ConditionResult::False => {}
                        ConditionResult::Inconclusive => {
                            is_zero = ConditionResult::Inconclusive;
                        }
                    }
                }

                is_zero
            }
            AtomView::Add(_) => self.zero_test_impl(iterations, tolerance),
        }
    }

    fn zero_test_impl(&self, iterations: usize, tolerance: f64) -> ConditionResult {
        // collect all variables and functions and fill in random variables

        let mut rng = MonteCarloRng::new(0, 0);

        if !self.is_real() {
            let mut vars: HashMap<_, _> = self
                .get_all_indeterminates(true)
                .into_iter()
                .filter_map(|x| {
                    let s = x.get_symbol().unwrap();
                    if !State::is_fixed_builtin(s) || s == Symbol::DERIVATIVE {
                        Some((x, Complex::new(0f64.into(), 0f64.into())))
                    } else {
                        None
                    }
                })
                .collect();

            for _ in 0..iterations {
                for x in vars.values_mut() {
                    *x = x.sample_unit(&mut rng);
                }

                let r = self
                    .evaluate(
                        |x| {
                            Complex::new(
                                ErrorPropagatingFloat::new(
                                    0f64.from_rational(x),
                                    -0f64.get_epsilon().log10(),
                                ),
                                ErrorPropagatingFloat::new(
                                    0f64.zero(),
                                    -0f64.get_epsilon().log10(),
                                ),
                            )
                        },
                        &vars,
                        &HashMap::default(),
                    )
                    .unwrap();

                let res_re = r.re.get_num().to_f64();
                let res_im = r.im.get_num().to_f64();
                if res_re.is_finite()
                    && (res_re - r.re.get_absolute_error() > 0.
                        || res_re + r.re.get_absolute_error() < 0.)
                    || res_im.is_finite()
                        && (res_im - r.im.get_absolute_error() > 0.
                            || res_im + r.im.get_absolute_error() < 0.)
                {
                    return ConditionResult::False;
                }

                if vars.is_empty() && r.re.get_absolute_error() < tolerance {
                    return ConditionResult::True;
                }
            }

            ConditionResult::Inconclusive
        } else {
            let mut vars: HashMap<_, ErrorPropagatingFloat<f64>> = self
                .get_all_indeterminates(true)
                .into_iter()
                .filter_map(|x| {
                    let s = x.get_symbol().unwrap();
                    if !State::is_fixed_builtin(s) || s == Symbol::DERIVATIVE {
                        Some((x, 0f64.into()))
                    } else {
                        None
                    }
                })
                .collect();

            for _ in 0..iterations {
                for x in vars.values_mut() {
                    *x = x.sample_unit(&mut rng);
                }

                let r = self
                    .evaluate(
                        |x| {
                            ErrorPropagatingFloat::new(
                                0f64.from_rational(x),
                                -0f64.get_epsilon().log10(),
                            )
                        },
                        &vars,
                        &HashMap::default(),
                    )
                    .unwrap();

                let res = r.get_num().to_f64();

                // trust the error when the relative error is less than 20%
                if res != 0.
                    && res.is_finite()
                    && r.get_absolute_error() / res.abs() < 0.2
                    && (res - r.get_absolute_error() > 0. || res + r.get_absolute_error() < 0.)
                {
                    return ConditionResult::False;
                }

                if vars.is_empty() && r.get_absolute_error() < tolerance {
                    return ConditionResult::True;
                }
            }

            ConditionResult::Inconclusive
        }
    }
}
