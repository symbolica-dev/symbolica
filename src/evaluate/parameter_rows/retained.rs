//! Invariant scalar arguments extracted from retained native function bodies.

use super::*;
use crate::atom::AliasedAtom;

type Number = Complex<Rational>;

struct Variant {
    lifted: ExpressionEvaluator<Number>,
    physical: Vec<usize>,
    arguments: Vec<usize>,
    function: usize,
}

struct Splitter {
    functions: Vec<ExternalFunctionContainer<Number>>,
    variants: HashMap<(usize, Vec<bool>), Option<Arc<Variant>>>,
    settings: OptimizationSettings,
}

/// Lift invariant instructions to the original call site, leaving a retained
/// kernel for the varying work. The row planner can then share the lifted
/// scalars without looking through a function on every residue row.
pub(super) fn split_calls(
    evaluator: &mut ExpressionEvaluator<Number>,
    columns: &[usize],
    rows: &[Vec<Rational>],
) -> Result<(), EvaluationError> {
    if rows.len() < 2
        || evaluator
            .external_fns
            .iter()
            .all(|function| function.body.is_none())
    {
        return Ok(());
    }
    let mut varying = vec![false; evaluator.param_count];
    for (column, position) in columns.iter().enumerate() {
        varying[*position] = rows.iter().any(|row| row[column] != rows[0][column]);
    }
    let mut splitter = Splitter {
        functions: evaluator.external_fns.clone(),
        variants: HashMap::default(),
        settings: evaluator.settings.clone(),
    };
    if splitter.rewrite(evaluator, &varying)? {
        evaluator.external_fns = splitter.functions;
        evaluator.remove_common_instructions();
        evaluator.undo_stack_optimization();
    }
    Ok(())
}

impl Splitter {
    fn variant(
        &mut self,
        function: usize,
        varying: Vec<bool>,
    ) -> Result<Option<Arc<Variant>>, EvaluationError> {
        let Some(original) = self.functions[function].body.clone() else {
            return Ok(None);
        };
        if varying.len() != original.param_count {
            return Err(EvaluationError::WrongNumberOfArguments {
                function: self.functions[function].symbol,
                expected: original.param_count,
                actual: varying.len(),
            });
        }
        if !varying.contains(&true) {
            // The existing planner already shares the entire invariant call.
            return Ok(None);
        }
        let key = (function, varying.clone());
        if let Some(variant) = self.variants.get(&key) {
            return Ok(variant.clone());
        }
        // Native retained functions are acyclic. A pending entry also prevents
        // repeated expansion if an imported body has an unsupported cycle.
        self.variants.insert(key.clone(), None);
        let mut source =
            ExpressionEvaluator::from_retained_body(&original, &self.functions, &self.settings);
        source.undo_stack_optimization();
        self.rewrite(&mut source, &varying)?;
        // New nested kernels use the same registry, while constants in every
        // retained scope continue to arrive through explicit captured inputs.
        source.external_fns = self.functions.clone();
        for function in &mut source.external_fns {
            function.constant_index = None;
        }
        let columns = varying
            .iter()
            .enumerate()
            .filter_map(|(index, varying)| varying.then_some(index))
            .collect::<Vec<_>>();
        // A varying callee argument is an unknown runtime value, not a dummy
        // exact coefficient. In particular it cannot certify a lazy branch.
        let analysis = RowAnalysis::with_bindings(&source, &columns, 1, |_, _| None);
        let candidates = source
            .instructions
            .iter()
            .filter_map(|(instruction, _)| {
                let target = output(instruction)?;
                (!matches!(instruction, Instr::Join(..))
                    && analysis.dependencies[target].is_empty()
                    && analysis.active[target][0] == Some(true)
                    && analysis.safe[target][0])
                    .then_some(target)
            })
            .collect::<Vec<_>>();
        if candidates.is_empty() {
            return Ok(None);
        }
        let all_outputs = source
            .instructions
            .iter()
            .filter_map(|(instruction, _)| output(instruction))
            .collect();
        let mut kernel = body_with_shared_inputs(&source, &candidates, &all_outputs);
        let mut arguments = kernel.retain_output_component(0);
        let selected = arguments
            .iter()
            .copied()
            .filter(|argument| *argument >= source.param_count)
            .map(|argument| argument - source.param_count)
            .collect::<Vec<_>>();
        if selected.is_empty() {
            return Ok(None);
        }
        let physical = (0..source.param_count)
            .filter(|index| !varying[*index])
            .collect::<Vec<_>>();
        let ports = selected
            .iter()
            .map(|index| candidates[*index])
            .collect::<Vec<_>>();
        let mut prelude = Prelude::new(&source, &physical, &columns, &[], &analysis, &ports);
        let outputs = ports
            .iter()
            .map(|port| prelude.copy(*port, 0, &[]))
            .collect();
        let lifted = prelude.finish(outputs);
        debug_assert_eq!(lifted.param_count, lifted.reserved_indices);
        for argument in &mut arguments {
            if *argument >= source.param_count {
                let port = *argument - source.param_count;
                *argument =
                    source.param_count + selected.iter().position(|index| *index == port).unwrap();
            }
        }
        let function = self.functions.len();
        self.functions.push(ExternalFunctionContainer::with_body(
            AliasedAtom::fresh_alias_symbol(),
            Vec::new(),
            Arc::new(FunctionBody {
                instructions: kernel.instructions,
                param_count: kernel.param_count,
                result_index: kernel.result_indices[0],
                stack_size: kernel.stack.len(),
            }),
        ));
        let variant = Arc::new(Variant {
            lifted,
            physical,
            arguments,
            function,
        });
        self.variants.insert(key, Some(variant.clone()));
        Ok(Some(variant))
    }

    // Both the root row program and newly opened retained bodies arrive in SSA.
    fn rewrite(
        &mut self,
        evaluator: &mut ExpressionEvaluator<Number>,
        varying: &[bool],
    ) -> Result<bool, EvaluationError> {
        let columns = varying
            .iter()
            .enumerate()
            .filter_map(|(index, varying)| varying.then_some(index))
            .collect::<Vec<_>>();
        let analysis = RowAnalysis::with_bindings(evaluator, &columns, 1, |_, _| None);
        let mut slots =
            (0..evaluator.reserved_indices + evaluator.instructions.len()).collect::<Vec<_>>();
        let mut instructions = Vec::new();
        let mut changed = false;
        for (instruction, phase) in &evaluator.instructions {
            if let Instr::ExternalFun(old_output, function, arguments) = instruction {
                let varying = arguments
                    .iter()
                    .map(|argument| !analysis.dependencies[*argument].is_empty())
                    .collect();
                if let Some(variant) = self.variant(*function, varying)? {
                    let mut arguments = arguments
                        .iter()
                        .map(|argument| slots[*argument])
                        .collect::<Vec<_>>();
                    let lifted = &variant.lifted;
                    let mut lifted_slots = vec![0; lifted.stack.len()];
                    for (input, original) in variant.physical.iter().enumerate() {
                        lifted_slots[input] = arguments[*original];
                    }
                    for (instruction, phase) in &lifted.instructions {
                        let old = output(instruction)
                            .expect("lifted scalars have no control-flow instructions");
                        let target = evaluator.reserved_indices + instructions.len();
                        instructions.push((
                            remap(instruction, target, |input| lifted_slots[input]),
                            *phase,
                        ));
                        lifted_slots[old] = target;
                    }
                    arguments.extend(
                        lifted
                            .result_indices
                            .iter()
                            .map(|output| lifted_slots[*output]),
                    );
                    let target = evaluator.reserved_indices + instructions.len();
                    instructions.push((
                        Instr::ExternalFun(
                            target,
                            variant.function,
                            variant
                                .arguments
                                .iter()
                                .map(|argument| arguments[*argument])
                                .collect(),
                        ),
                        *phase,
                    ));
                    slots[*old_output] = target;
                    changed = true;
                    continue;
                }
            }
            let target = evaluator.reserved_indices + instructions.len();
            instructions.push((remap(instruction, target, |input| slots[input]), *phase));
            if let Some(old) = output(instruction) {
                slots[old] = target;
            }
        }
        if changed {
            evaluator.instructions = instructions;
            evaluator
                .result_indices
                .iter_mut()
                .for_each(|output| *output = slots[*output]);
            evaluator.stack.resize(
                evaluator.reserved_indices + evaluator.instructions.len(),
                Number::default(),
            );
            evaluator.fix_labels();
        }
        Ok(changed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse, symbol};

    #[test]
    fn retained_body_shares_physical_root_without_inlining_the_numerator() {
        let x = parse!("retained_row_scalar::x");
        let c = parse!("retained_row_scalar::c");
        let head = symbol!("retained_row_scalar::f");
        let original_body = parse!("sqrt(retained_row_scalar::a)+retained_row_scalar::b");
        let mut functions = FunctionMap::new();
        functions
            .add_function_with_options(
                head,
                vec![
                    symbol!("retained_row_scalar::a"),
                    symbol!("retained_row_scalar::b"),
                ],
                original_body.clone(),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap();
        for count in [3, 96] {
            let rows = (1..=count)
                .map(|value| vec![value.into()])
                .collect::<Vec<_>>();
            let staged = head
                .call((&x, &c))
                .evaluator(&[x.clone(), c.clone()])
                .function_map(functions.clone())
                .build_parameter_rows(std::slice::from_ref(&c), &rows)
                .unwrap();
            assert_eq!(
                staged
                    .prelude
                    .instructions
                    .iter()
                    .filter(|(instruction, _)| matches!(instruction, Instr::BuiltinFun(_, f, _) if *f == Symbol::SQRT))
                    .count(),
                1,
                "the physical square root belongs to the once-per-sample prelude"
            );
            let retained = staged
                .body
                .external_fns
                .iter()
                .filter_map(|function| function.body.as_ref())
                .collect::<Vec<_>>();
            assert_eq!(
                retained.len(),
                1,
                "one retained kernel, independent of row count"
            );
            assert_eq!(
                retained[0].param_count, 2,
                "unused original arguments are removed"
            );
            assert!(retained[0].instructions.iter().all(|(instruction, _)| {
                !matches!(instruction, Instr::BuiltinFun(_, f, _) if *f == Symbol::SQRT)
            }));
            let mut numeric = staged.map_coeff(&|value| value.re.to_f64());
            for value in [4_f64, 9.] {
                let mut output = [0.];
                numeric
                    .evaluate(&[value], &mut output, Rational::to_f64)
                    .unwrap();
                let expected = count as f64 * value.sqrt() + (count * (count + 1) / 2) as f64;
                assert_eq!(output, [expected]);
            }
        }
        assert_eq!(
            functions.get(head.call((&x, &c)).as_view()).unwrap().body,
            original_body
        );
    }

    #[test]
    fn nested_retained_sharing_preserves_captures_variance_masks_and_guards() {
        let params = [
            parse!("retained_row_nested::x"),
            parse!("retained_row_nested::c"),
            parse!("retained_row_nested::gate"),
            parse!("retained_row_nested::z"),
        ];
        let inner = symbol!("retained_row_nested::inner");
        let outer = symbol!("retained_row_nested::outer");
        let rows = vec![vec![1.into()], vec![2.into()], vec![3.into()]];
        for policy in [InliningPolicy::Always, InliningPolicy::Never] {
            for guarded in [false, true] {
                let mut functions = FunctionMap::new();
                functions
                    .add_function_with_options(
                        inner,
                        vec![symbol!("retained_row_nested::a"), symbol!("retained_row_nested::b")],
                        parse!("sqrt(retained_row_nested::a+retained_row_nested::z+pi)+if(retained_row_nested::a,1/retained_row_nested::a,0)+retained_row_nested::b"),
                        FunctionRegistrationOptions::new().inlining(policy),
                    )
                    .unwrap();
                functions
                    .add_function_with_options(
                        outer,
                        vec![symbol!("retained_row_nested::u"), symbol!("retained_row_nested::v")],
                        parse!("retained_row_nested::inner(retained_row_nested::u,retained_row_nested::v)+retained_row_nested::inner(retained_row_nested::u,retained_row_nested::v+1)"),
                        FunctionRegistrationOptions::new().inlining(policy),
                    )
                    .unwrap();
                // These two call sites have opposite varying-argument masks;
                // nested kernels must not reuse the other site's hidden ABI.
                let sum =
                    outer.call((&params[0], &params[1])) + outer.call((&params[1], &params[0]));
                let root = if guarded {
                    Symbol::IF.call_args([params[2].clone(), sum, Atom::Zero])
                } else {
                    sum
                };
                let source = root
                    .evaluator(&params)
                    .function_map(functions)
                    .build()
                    .unwrap();
                let staged =
                    ParameterRowEvaluator::new(source.clone(), vec![1], rows.clone()).unwrap();
                if guarded {
                    assert_eq!(
                        staged.prelude.get_output_len(),
                        0,
                        "an unknown physical guard must retain its singular work"
                    );
                } else {
                    assert!(staged.prelude.instructions.iter().any(|(instruction, _)|
                        matches!(instruction, Instr::BuiltinFun(_, f, _) if *f == Symbol::SQRT)));
                }
                let mut oracle = source.map_coeff(&|value| value.re.to_f64());
                let mut numeric = staged.clone().map_coeff(&|value| value.re.to_f64());
                for (x, gate) in [(0., 1.), (2., 1.), (-5., 0.)] {
                    if !guarded && gate == 0. {
                        continue;
                    }
                    let expected = if gate == 0. {
                        0.
                    } else {
                        (1..=3)
                            .map(|c| {
                                let c = f64::from(c);
                                2. * (x + 0.5 + std::f64::consts::PI).sqrt()
                                    + 2. * (c + 0.5 + std::f64::consts::PI).sqrt()
                                    + 2. * c
                                    + 2. * x
                                    + 2.
                                    + if x == 0. { 0. } else { 2. / x }
                                    + 2. / c
                            })
                            .sum::<f64>()
                    };
                    let original = (1..=3)
                        .map(|c| oracle.evaluate_single(&[x, f64::from(c), gate, 0.5]))
                        .sum::<f64>();
                    let mut output = [0.];
                    numeric
                        .evaluate(&[x, gate, 0.5], &mut output, Rational::to_f64)
                        .unwrap();
                    assert!((original - expected).abs() < 1e-12);
                    assert!(
                        (output[0] - expected).abs() < 1e-12,
                        "policy={policy:?}, guarded={guarded}, x={x}, gate={gate}: {output:?} versus {expected}"
                    );
                    #[cfg(feature = "native_code_generation")]
                    for direct in [false, true] {
                        let (plan, prelude, body) = staged
                            .clone()
                            .map_coeff(&|value| value.re.to_f64())
                            .into_parts();
                        let settings = JITCompilationSettings::new()
                            .optimization_level(2)
                            .direct_translation(direct);
                        let mut prelude = (prelude.get_output_len() != 0)
                            .then(|| prelude.jit_compile(settings.clone()).unwrap());
                        let mut body = body.jit_compile(settings).unwrap();
                        plan.evaluate_with(
                            &[x, gate, 0.5],
                            &mut output,
                            &mut ParameterRowWorkspace::default(),
                            Rational::to_f64,
                            |input, output| {
                                prelude.as_mut().unwrap().evaluate(input, output);
                                Ok(())
                            },
                            |input, output| {
                                body.evaluate(input, output);
                                Ok(())
                            },
                        )
                        .unwrap();
                        assert!((output[0] - expected).abs() < 1e-12);
                    }
                }
            }
        }
    }

    #[test]
    fn retained_shared_scalars_preserve_mixed_jets_and_program_roundtrips() {
        let params = [
            parse!("retained_row_jets::x"),
            parse!("retained_row_jets::c"),
        ];
        let function = symbol!("retained_row_jets::f");
        let staged = function
            .call((&params[0], &params[1]))
            .evaluator(&params)
            .add_function_with_options(
                function,
                vec![
                    symbol!("retained_row_jets::a"),
                    symbol!("retained_row_jets::b"),
                ],
                parse!("sqrt(retained_row_jets::a)+retained_row_jets::b*retained_row_jets::a^2"),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap()
            .build_parameter_rows(
                &params[1..],
                &[vec![1.into()], vec![2.into()], vec![3.into()]],
            )
            .unwrap();
        let shape = crate::domains::dual::HyperDual::<Number>::new(vec![
            vec![0, 0],
            vec![1, 0],
            vec![0, 1],
            vec![1, 1],
        ]);
        let mut numeric = staged
            .vectorize(&Dualizer::new(shape, vec![]))
            .unwrap()
            .map_coeff(&|value| value.re.to_f64());
        let mut output = [0.; 4];
        numeric
            .evaluate(&[4., 1., 2., 0.], &mut output, Rational::to_f64)
            .unwrap();
        assert_eq!(output, [102., 48.75, 97.5, 23.8125]);
        #[cfg(all(feature = "serde", feature = "bincode"))]
        {
            let json = serde_json::to_vec(&numeric).unwrap();
            let mut restored: ParameterRowEvaluator<f64> = serde_json::from_slice(&json).unwrap();
            restored
                .evaluate(&[4., 1., 2., 0.], &mut output, Rational::to_f64)
                .unwrap();
            assert_eq!(output, [102., 48.75, 97.5, 23.8125]);
            let bytes = bincode::encode_to_vec(&numeric, bincode::config::standard()).unwrap();
            let (mut restored, _): (ParameterRowEvaluator<f64>, _) =
                bincode::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
            restored
                .evaluate(&[4., 1., 2., 0.], &mut output, Rational::to_f64)
                .unwrap();
            assert_eq!(output, [102., 48.75, 97.5, 23.8125]);
        }
        #[cfg(feature = "native_code_generation")]
        for direct in [false, true] {
            let (plan, prelude, body) = numeric.clone().into_parts();
            let settings = JITCompilationSettings::new()
                .optimization_level(2)
                .direct_translation(direct);
            let mut prelude = prelude.jit_compile(settings.clone()).unwrap();
            let mut body = body.jit_compile(settings).unwrap();
            plan.evaluate_with(
                &[4., 1., 2., 0.],
                &mut output,
                &mut ParameterRowWorkspace::default(),
                Rational::to_f64,
                |input, output| {
                    prelude.evaluate(input, output);
                    Ok(())
                },
                |input, output| {
                    body.evaluate(input, output);
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(output, [102., 48.75, 97.5, 23.8125]);
        }
    }
}
