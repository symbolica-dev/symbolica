use super::*;

/// An expression evaluator that applies each instruction across a batch.
///
/// Inputs and outputs are row-major: a row contains all parameters or results
/// for one sample. Internally, lanes of each stack slot are contiguous. Shared
/// functions receive the same batch, and branches evaluate only their active
/// lanes. Storage is retained between calls and grows to the largest batch seen.
///
/// Map exact coefficients to the desired numeric domain and precision before
/// calling [`ExpressionEvaluator::into_batch_evaluator`]. Dual numbers can be
/// represented by flattening their components with [`Dualizer`] first.
#[derive(Clone)]
pub struct BatchExpressionEvaluator<T> {
    evaluator: ExpressionEvaluator<T>,
    workspace: BatchWorkspace<T>,
    functions: Vec<BatchWorkspace<T>>,
    lanes: Vec<usize>,
    capacity: usize,
}

#[derive(Clone)]
struct BatchWorkspace<T> {
    stack: Vec<T>,
    active: Vec<usize>,
    // Forward instruction index at which each lane resumes after a branch.
    resume: Vec<usize>,
}

impl<T> Default for BatchWorkspace<T> {
    fn default() -> Self {
        Self {
            stack: Vec::new(),
            active: Vec::new(),
            resume: Vec::new(),
        }
    }
}

impl<T> ExpressionEvaluator<T> {
    /// Convert to an interpreter that carries batches through instructions and
    /// shared function calls, with reusable storage and lazy per-lane branches.
    ///
    /// ```
    /// use symbolica::prelude::*;
    /// let mut evaluator = parse!("x^2+y")
    ///     .evaluator(&[parse!("x"), parse!("y")]).build().unwrap()
    ///     .map_coeff(&|c| c.re.to_f64()).into_batch_evaluator();
    /// let mut output = [0.; 3];
    /// evaluator.evaluate_batch(&[2., 1., 3., 2., 4., 3.], &mut output, 3).unwrap();
    /// assert_eq!(output, [5., 11., 19.]);
    /// ```
    pub fn into_batch_evaluator(self) -> BatchExpressionEvaluator<T> {
        let functions = (0..self.external_fns.len())
            .map(|_| BatchWorkspace::default())
            .collect();
        BatchExpressionEvaluator {
            evaluator: self,
            workspace: BatchWorkspace::default(),
            functions,
            lanes: Vec::new(),
            capacity: 0,
        }
    }
}

impl<T> BatchExpressionEvaluator<T> {
    /// Number of parameters in each input row.
    pub fn get_input_len(&self) -> usize {
        self.evaluator.param_count
    }

    /// Number of results in each output row.
    pub fn get_output_len(&self) -> usize {
        self.evaluator.result_indices.len()
    }

    /// Return the original scalar evaluator, discarding batch workspaces.
    pub fn into_expression_evaluator(self) -> ExpressionEvaluator<T> {
        self.evaluator
    }

    fn size_overflow() -> EvaluationError {
        EvaluationError::EvaluationFailed {
            expression: Atom::num(0),
            reason: "Batch dimensions exceed the addressable buffer size".into(),
        }
    }
}

impl<T: Real> BatchWorkspace<T> {
    fn reserve(&mut self, slots: usize, lanes: usize) -> Result<(), EvaluationError> {
        let size = slots
            .checked_mul(lanes)
            .ok_or_else(BatchExpressionEvaluator::<T>::size_overflow)?;
        self.stack.resize_with(size, T::new_zero);
        self.resume.resize(lanes, 0);
        if self.active.capacity() < lanes {
            self.active.reserve(lanes - self.active.len());
        }
        Ok(())
    }
}

impl<T: Real> BatchExpressionEvaluator<T> {
    /// Preallocate reusable root and shared-function workspaces for `rows`.
    pub fn reserve(&mut self, rows: usize) -> Result<(), EvaluationError> {
        if rows <= self.capacity {
            return Ok(());
        }
        self.workspace.reserve(self.evaluator.stack.len(), rows)?;
        for (workspace, function) in self
            .functions
            .iter_mut()
            .zip(&mut self.evaluator.external_fns)
        {
            if let Some(body) = &function.body {
                workspace.reserve(body.stack_size, rows)?;
            }
        }
        self.lanes = (0..rows).collect();
        self.capacity = rows;
        Ok(())
    }

    /// Evaluate `rows` samples. Input and output lengths must be exactly
    /// `rows * get_input_len()` and `rows * get_output_len()`, respectively.
    /// Empty batches are supported, including evaluators with no parameters.
    /// Shape errors leave the output buffer unchanged.
    pub fn evaluate_batch(
        &mut self,
        parameters: &[T],
        out: &mut [T],
        rows: usize,
    ) -> Result<(), EvaluationError> {
        let inputs = rows
            .checked_mul(self.get_input_len())
            .ok_or_else(Self::size_overflow)?;
        let outputs = rows
            .checked_mul(self.get_output_len())
            .ok_or_else(Self::size_overflow)?;
        if parameters.len() != inputs {
            return Err(EvaluationError::InvalidParameterCount {
                expected: inputs,
                actual: parameters.len(),
            });
        }
        if out.len() != outputs {
            return Err(EvaluationError::InvalidOutputCount {
                expected: outputs,
                actual: out.len(),
            });
        }
        if rows == 0 {
            return Ok(());
        }
        // Avoid packing overhead and batch allocations for the scalar case.
        if rows == 1 {
            return self.evaluator.try_evaluate(parameters, out);
        }
        self.reserve(rows)?;
        let stride = self.capacity;
        let evaluator = &mut self.evaluator;
        for parameter in 0..evaluator.param_count {
            for lane in 0..rows {
                self.workspace.stack[parameter * stride + lane]
                    .set_from(&parameters[lane * evaluator.param_count + parameter]);
            }
        }
        for constant in evaluator.param_count..evaluator.reserved_indices {
            for lane in 0..rows {
                self.workspace.stack[constant * stride + lane].set_from(&evaluator.stack[constant]);
            }
        }
        Self::evaluate_program(
            &evaluator.instructions,
            &mut self.workspace,
            &mut evaluator.external_fns,
            &mut self.functions,
            &self.lanes[..rows],
            stride,
        )?;
        for (column, slot) in evaluator.result_indices.iter().enumerate() {
            for lane in 0..rows {
                out[lane * evaluator.result_indices.len() + column]
                    .set_from(&self.workspace.stack[slot * stride + lane]);
            }
        }
        Ok(())
    }

    fn evaluate_program(
        instructions: &[(Instr, ComplexPhase)],
        workspace: &mut BatchWorkspace<T>,
        external_fns: &mut [ExternalFunctionContainer<T>],
        functions: &mut [BatchWorkspace<T>],
        lanes: &[usize],
        stride: usize,
    ) -> Result<(), EvaluationError> {
        let BatchWorkspace {
            stack,
            active,
            resume,
        } = workspace;
        active.clear();
        active.extend_from_slice(lanes);
        for lane in lanes {
            resume[*lane] = 0;
        }
        let mut temporary = T::new_zero();
        for (pc, (instruction, _)) in instructions.iter().enumerate() {
            match instruction {
                Instr::Label(_) => {
                    active.clear();
                    active.extend(lanes.iter().copied().filter(|lane| resume[*lane] <= pc));
                }
                Instr::Goto(label) => {
                    debug_assert!(label.0 > pc, "Batch programs have forward branches");
                    for lane in active.iter() {
                        resume[*lane] = label.0;
                    }
                    active.clear();
                }
                Instr::IfElse(condition, label) => {
                    debug_assert!(label.0 > pc, "Batch programs have forward branches");
                    active.retain(|lane| {
                        if stack[condition * stride + lane].is_fully_zero() {
                            resume[*lane] = label.0;
                            false
                        } else {
                            true
                        }
                    });
                }
                Instr::Add(out, args) | Instr::Mul(out, args) => {
                    for lane in active.iter().copied() {
                        temporary.set_from(&stack[args[0] * stride + lane]);
                        for arg in &args[1..] {
                            if matches!(instruction, Instr::Add(..)) {
                                temporary += &stack[arg * stride + lane];
                            } else {
                                temporary *= &stack[arg * stride + lane];
                            }
                        }
                        std::mem::swap(&mut stack[out * stride + lane], &mut temporary);
                    }
                }
                Instr::Pow(out, base, exponent) => {
                    for lane in active.iter().copied() {
                        let value = &stack[base * stride + lane];
                        temporary = if *exponent == -1 {
                            value.inv()
                        } else if *exponent >= 0 {
                            value.pow(*exponent as u64)
                        } else {
                            value.pow(exponent.unsigned_abs()).inv()
                        };
                        std::mem::swap(&mut stack[out * stride + lane], &mut temporary);
                    }
                }
                Instr::Powf(out, base, exponent) => {
                    for lane in active.iter().copied() {
                        temporary =
                            stack[base * stride + lane].powf(&stack[exponent * stride + lane]);
                        std::mem::swap(&mut stack[out * stride + lane], &mut temporary);
                    }
                }
                Instr::BuiltinFun(out, symbol, arg) => {
                    for lane in active.iter().copied() {
                        let value = &stack[arg * stride + lane];
                        temporary = match symbol.get_id() {
                            Symbol::EXP_ID => value.exp(),
                            Symbol::LOG_ID => value.log(),
                            Symbol::SIN_ID => value.sin(),
                            Symbol::COS_ID => value.cos(),
                            Symbol::SQRT_ID => value.sqrt(),
                            Symbol::ABS_ID => value.norm(),
                            Symbol::CONJ_ID => value.conj(),
                            _ => unreachable!("Unsupported built-in evaluator instruction"),
                        };
                        std::mem::swap(&mut stack[out * stride + lane], &mut temporary);
                    }
                }
                Instr::Join(out, condition, then_value, else_value) => {
                    for lane in active.iter().copied() {
                        let value = if stack[condition * stride + lane].is_fully_zero() {
                            else_value
                        } else {
                            then_value
                        };
                        temporary.set_from(&stack[value * stride + lane]);
                        std::mem::swap(&mut stack[out * stride + lane], &mut temporary);
                    }
                }
                Instr::ExternalFun(out, index, args) => {
                    if active.is_empty() {
                        continue;
                    }
                    let (callees, current) = external_fns.split_at_mut(*index);
                    let (callee_workspaces, current_workspace) = functions.split_at_mut(*index);
                    let external = &mut current[0];
                    if let Some(body) = &external.body {
                        let workspace = &mut current_workspace[0];
                        debug_assert_eq!(args.len(), body.param_count);
                        for (parameter, arg) in args.iter().enumerate() {
                            for lane in active.iter().copied() {
                                workspace.stack[parameter * stride + lane]
                                    .set_from(&stack[arg * stride + lane]);
                            }
                        }
                        Self::evaluate_program(
                            &body.instructions,
                            workspace,
                            callees,
                            callee_workspaces,
                            active,
                            stride,
                        )?;
                        for lane in active.iter().copied() {
                            stack[out * stride + lane]
                                .set_from(&workspace.stack[body.result_index * stride + lane]);
                        }
                    } else if let Some(function) = &external.imp {
                        if external.stack.len() < args.len() {
                            external.stack.resize_with(args.len(), T::new_zero);
                        }
                        for lane in active.iter().copied() {
                            for (value, arg) in external.stack.iter_mut().zip(args) {
                                value.set_from(&stack[arg * stride + lane]);
                            }
                            stack[out * stride + lane] = function(&external.stack[..args.len()]);
                        }
                    } else {
                        return Err(EvaluationError::EvaluationFailed {
                            expression: external.symbol.call(&external.tags),
                            reason: format!(
                                "No function implementation for {}",
                                std::any::type_name::<T>()
                            ),
                        });
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_polynomial_and_buffer_shapes() {
        let exact = Atom::evaluator_multiple(
            &[crate::parse!("x^2+2*x*y+y"), crate::parse!("x-y")],
            &[crate::parse!("x"), crate::parse!("y")],
        )
        .build()
        .unwrap();
        let mut scalar = exact.map_coeff(&|c| c.re.to_f64());
        let mut batch = scalar.clone().into_batch_evaluator();
        for rows in [0, 1, 8, 3, 33, 2] {
            let parameters = (0..rows)
                .flat_map(|i| [i as f64 / 7., (rows - i) as f64 / 3.])
                .collect::<Vec<_>>();
            let mut actual = vec![0.; rows * 2];
            batch
                .evaluate_batch(&parameters, &mut actual, rows)
                .unwrap();
            for (input, result) in parameters.chunks_exact(2).zip(actual.chunks_exact(2)) {
                let mut expected = [0.; 2];
                scalar.evaluate(input, &mut expected);
                assert_eq!(result, expected);
            }
        }
        let mut out = [42.; 4];
        assert!(matches!(
            batch.evaluate_batch(&[1.; 3], &mut out, 2),
            Err(EvaluationError::InvalidParameterCount { .. })
        ));
        assert_eq!(out, [42.; 4]);
        assert!(matches!(
            batch.evaluate_batch(&[1.; 4], &mut out[..3], 2),
            Err(EvaluationError::InvalidOutputCount { .. })
        ));
        assert_eq!(out, [42.; 4]);
    }

    #[test]
    fn batch_constant_outputs_and_oversized_dimensions() {
        let mut evaluator = crate::parse!("pi")
            .evaluator::<Atom>(&[])
            .build()
            .unwrap()
            .map_coeff(&|c| c.re.to_f64())
            .into_batch_evaluator();
        let mut out = [0.; 5];
        evaluator.evaluate_batch(&[], &mut out, 5).unwrap();
        assert_eq!(out, [std::f64::consts::PI; 5]);
        let mut evaluator = crate::parse!("x+y")
            .evaluator(&[crate::parse!("x"), crate::parse!("y")])
            .build()
            .unwrap()
            .map_coeff(&|c| c.re.to_f64())
            .into_batch_evaluator();
        assert!(evaluator.evaluate_batch(&[], &mut [], usize::MAX).is_err());
    }

    #[test]
    fn batch_complex_precision_domains_and_flattened_duals() {
        let exact = crate::parse!("batch_precision::f(c,x)")
            .evaluator(&[crate::parse!("c"), crate::parse!("x")])
            .add_function_with_options(
                crate::symbol!("batch_precision::f"),
                vec![crate::symbol!("a"), crate::symbol!("z")],
                crate::parse!("if(a,log(1+z)+pi,sqrt(1+z))"),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap()
            .build()
            .unwrap()
            .vectorize(&Dualizer::new(
                crate::domains::dual::HyperDual::<Complex<Rational>>::new(vec![
                    vec![0],
                    vec![1],
                    vec![2],
                ]),
                vec![],
            ))
            .unwrap();
        fn check<T: Real>(mut scalar: ExpressionEvaluator<T>, input: Vec<T>) {
            let mut batch = scalar.clone().into_batch_evaluator();
            let mut result = vec![T::new_zero(); 9];
            batch.evaluate_batch(&input, &mut result, 3).unwrap();
            for (input, actual) in input.chunks_exact(6).zip(result.chunks_exact(3)) {
                let mut expected = vec![T::new_zero(); 3];
                scalar.evaluate(input, &mut expected);
                assert_eq!(actual, expected);
            }
        }
        let values = [
            (0, 0),
            (0, 0),
            (0, 0),
            (3, 4),
            (1, 0),
            (0, 0),
            (1, 0),
            (0, 0),
            (0, 0),
            (2, 1),
            (1, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (0, 0),
            (1, 2),
            (2, 1),
            (1, 0),
        ];
        let to_f64 = |c: &Complex<Rational>| Complex::new(c.re.to_f64(), c.im.to_f64());
        let to_double = |c: &Complex<Rational>| {
            Complex::new(DoubleFloat::from(&c.re), DoubleFloat::from(&c.im))
        };
        let to_float = |c: &Complex<Rational>| {
            Complex::new(c.re.to_multi_prec_float(192), c.im.to_multi_prec_float(192))
        };
        let exact_inputs =
            values.map(|(re, im)| Complex::new(Rational::from(re), Rational::from(im)));
        check(
            exact.clone().map_coeff(&to_f64),
            exact_inputs.iter().map(to_f64).collect(),
        );
        check(
            exact.clone().map_coeff(&to_double),
            exact_inputs.iter().map(to_double).collect(),
        );
        check(
            exact.map_coeff_with_prec(&to_float, 192),
            exact_inputs.iter().map(to_float).collect(),
        );
    }

    #[test]
    fn batch_calls_are_instruction_major_and_skip_inactive_lanes() {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let f_calls = calls.clone();
        crate::symbol!(
            "batch_order::f",
            eval = EvaluationInfo::new().register(move |args: &[f64]| {
                assert_ne!(args[0], 0., "inactive lane reached external callback");
                f_calls.lock().unwrap().push(('f', args[0]));
                1. / args[0]
            })
        );
        let g_calls = calls.clone();
        crate::symbol!(
            "batch_order::g",
            eval = EvaluationInfo::new().register(move |args: &[f64]| {
                g_calls.lock().unwrap().push(('g', args[0]));
                args[0] * args[0]
            })
        );
        let mut evaluator = crate::parse!("batch_order::body(c,x)")
            .evaluator(&[crate::parse!("c"), crate::parse!("x")])
            .add_function_with_options(
                crate::symbol!("batch_order::body"),
                vec![crate::symbol!("a"), crate::symbol!("z")],
                crate::parse!("if(a,batch_order::f(z)+batch_order::g(z),z)"),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap()
            .build()
            .unwrap()
            .map_coeff(&|c| c.re.to_f64())
            .into_batch_evaluator();
        let mut out = [0.; 4];
        evaluator
            .evaluate_batch(&[1., 1., 0., 0., 1., 2., 1., 3.], &mut out, 4)
            .unwrap();
        assert_eq!(out, [2., 0., 4.5, 9. + 1. / 3.]);
        let calls = calls.lock().unwrap();
        assert_eq!(calls.len(), 6);
        for group in calls.chunks_exact(3) {
            assert!(group.iter().all(|(function, _)| *function == group[0].0));
            assert_eq!(
                group.iter().map(|(_, value)| *value).collect::<Vec<_>>(),
                [1., 2., 3.]
            );
        }
        assert_ne!(calls[0].0, calls[3].0);
    }

    #[test]
    fn batch_shared_functions_keep_inactive_branches_lazy() {
        let exact = crate::parse!("batch_shared::outer(c,x)+batch_shared::outer(c,x+1)")
            .evaluator(&[crate::parse!("c"), crate::parse!("x"), crate::parse!("q")])
            .add_function_with_options(
                crate::symbol!("batch_shared::inner"),
                vec![crate::symbol!("a"), crate::symbol!("y")],
                crate::parse!("if(a,1/y,sqrt(y+2))"),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap()
            .add_function_with_options(
                crate::symbol!("batch_shared::outer"),
                vec![crate::symbol!("b"), crate::symbol!("z")],
                crate::parse!("if(b,batch_shared::inner(b,z)+q,batch_shared::inner(b,z)*q)"),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap()
            .build()
            .unwrap();
        let mut scalar = exact.map_coeff(&|c| c.re.to_f64());
        let mut batch = scalar.clone().into_batch_evaluator();
        let inputs = [0., 0., 2., 1., -4., 3., 0., 0., 4., 1., 2., 1., 0., -1., 2.];
        for rows in [5, 2, 3, 5] {
            let mut actual = vec![0.; rows];
            batch
                .evaluate_batch(&inputs[..3 * rows], &mut actual, rows)
                .unwrap();
            for (parameters, result) in inputs.chunks_exact(3).zip(actual) {
                assert_eq!(result, scalar.evaluate_single(parameters));
                assert!(result.is_finite());
            }
        }
    }
}
