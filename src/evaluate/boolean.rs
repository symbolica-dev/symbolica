//! Conservative proofs of native branch values, independent of precision.
//!
//! Only signed Boolean values are certified. General rational arithmetic can
//! round or underflow after domain mapping, so it must remain a runtime test.

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) struct NativeBoolean(i8);

impl NativeBoolean {
    pub(super) fn from_exact(value: &Complex<Rational>) -> Option<Self> {
        if !value.im.is_zero() {
            return None;
        }
        if value.re.is_zero() {
            Some(Self(0))
        } else if value.re.is_one() {
            Some(Self(1))
        } else if value.re == -Rational::one() {
            Some(Self(-1))
        } else {
            None
        }
    }

    pub(super) fn from_truth(value: bool) -> Self {
        Self(i8::from(value))
    }

    pub(super) fn is_zero(self) -> bool {
        self.0 == 0
    }
}

/// Interpret only operations proved exact in every supported floating domain.
/// Unknown inputs may pass through an unselected branch, but an executed
/// unknown operation stops the proof. Retained bodies keep their native scope.
pub(super) struct BooleanEvaluation<'a> {
    functions: &'a [ExternalFunctionContainer<Complex<Rational>>],
    calls: HashMap<(usize, Vec<Option<NativeBoolean>>), Option<NativeBoolean>>,
}

impl<'a> BooleanEvaluation<'a> {
    pub(super) fn new(functions: &'a [ExternalFunctionContainer<Complex<Rational>>]) -> Self {
        Self {
            functions,
            calls: HashMap::default(),
        }
    }

    pub(super) fn value(
        &mut self,
        instruction: &Instr,
        values: &[Option<NativeBoolean>],
    ) -> Option<NativeBoolean> {
        match instruction {
            Instr::Add(_, arguments) if arguments.len() <= 2 => {
                let mut result = 0;
                for argument in arguments {
                    result += values[*argument]?.0;
                }
                (-1..=1).contains(&result).then_some(NativeBoolean(result))
            }
            Instr::Mul(_, arguments) => {
                let mut result = 1;
                for argument in arguments {
                    // Do not infer zero from a zero times an unknown value.
                    result *= values[*argument]?.0;
                }
                Some(NativeBoolean(result))
            }
            Instr::Pow(_, argument, power) => {
                let value = values[*argument]?.0;
                if *power == 0 {
                    Some(NativeBoolean(1))
                } else if value == 0 {
                    (*power > 0).then_some(NativeBoolean(0))
                } else {
                    Some(NativeBoolean(if *power % 2 == 0 { 1 } else { value }))
                }
            }
            Instr::Join(_, condition, yes, no) | Instr::Select(_, condition, yes, no) => {
                values[if values[*condition]?.is_zero() {
                    *no
                } else {
                    *yes
                }]
            }
            Instr::ExternalFun(_, index, arguments) => {
                let functions = self.functions;
                let body = functions[*index].body.as_ref()?;
                let arguments = arguments
                    .iter()
                    .map(|argument| values[*argument])
                    .collect::<Vec<_>>();
                let key = (*index, arguments.clone());
                if let Some(value) = self.calls.get(&key) {
                    return *value;
                }
                self.calls.insert(key.clone(), None);
                let mut locals = vec![None; body.stack_size];
                locals[..arguments.len()].copy_from_slice(&arguments);
                let result = self.evaluate(&body.instructions, locals, body.result_index);
                self.calls.insert(key, result);
                result
            }
            _ => None,
        }
    }

    fn evaluate(
        &mut self,
        instructions: &[(Instr, ComplexPhase)],
        mut values: Vec<Option<NativeBoolean>>,
        output: usize,
    ) -> Option<NativeBoolean> {
        let mut position = 0;
        while position < instructions.len() {
            let instruction = &instructions[position].0;
            match instruction {
                Instr::IfElse(condition, label) => {
                    if values[*condition]?.is_zero() {
                        position = label.0;
                        continue;
                    }
                }
                Instr::Goto(label) => {
                    position = label.0;
                    continue;
                }
                Instr::Label(_) => {}
                Instr::Add(output, _)
                | Instr::Mul(output, _)
                | Instr::Pow(output, _, _)
                | Instr::Powf(output, _, _)
                | Instr::Div(output, _, _)
                | Instr::BuiltinFun(output, _, _)
                | Instr::ExternalFun(output, _, _)
                | Instr::Join(output, _, _, _)
                | Instr::Select(output, _, _, _) => {
                    values[*output] = Some(self.value(instruction, &values)?);
                }
            }
            position += 1;
        }
        values[output]
    }

    pub(super) fn program(
        &mut self,
        program: &ExpressionEvaluator<Complex<Rational>>,
    ) -> Option<NativeBoolean> {
        if program.result_indices.len() != 1 {
            return None;
        }
        let mut values = vec![None; program.stack.len()];
        for (index, value) in values
            .iter_mut()
            .enumerate()
            .take(program.reserved_indices)
            .skip(program.param_count)
        {
            if !program
                .external_fns
                .iter()
                .any(|function| function.constant_index == Some(index - program.param_count))
            {
                *value = NativeBoolean::from_exact(&program.stack[index]);
            }
        }
        self.evaluate(&program.instructions, values, program.result_indices[0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse, symbol};

    #[test]
    fn zero_inverse_in_a_native_condition_is_not_exactly_folded() {
        let a = symbol!("boolean_inverse::a");
        let b = symbol!("boolean_inverse::b");
        let f = symbol!("boolean_inverse::f");
        let body = Symbol::IF.call_args([
            parse!("boolean_inverse::a/boolean_inverse::b"),
            Atom::one(),
            Atom::Zero,
        ]);
        for policy in [InliningPolicy::Always, InliningPolicy::Never] {
            let mut program = f
                .call((0, 0))
                .evaluator(&[] as &[Atom])
                .add_function_with_options(
                    f,
                    vec![a, b],
                    body.clone(),
                    FunctionRegistrationOptions::new().inlining(policy),
                )
                .unwrap()
                .build()
                .unwrap()
                .map_coeff(&|value| value.re.to_f64());
            assert_eq!(program.evaluate_single(&[]), 1.);
        }
    }
}
