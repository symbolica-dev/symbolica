//! Repeated-row evaluation using the native evaluator instruction graph.
//!
//! The shared program computes invariant values and repeated calls once per
//! physical input. Its results become ordinary inputs of the unchanged row
//! program. Extraction never moves work out of an unknown dynamic branch.

use std::{collections::BTreeSet, ops::AddAssign};

use super::boolean::{BooleanEvaluation, NativeBoolean};
use super::*;

mod retained;

/// Scheduling data for a shared prelude and an exact parameter-row program.
///
/// It contains no backend-specific evaluator. All cached values belong to one
/// invocation of [`Self::evaluate_with`], including after model/input changes.
#[cfg_attr(feature = "bincode", derive(bincode::Encode))]
#[derive(Clone, Debug)]
pub struct ParameterRowPlan {
    physical_positions: Vec<usize>,
    bound_positions: Vec<usize>,
    rows: Vec<Vec<Rational>>,
    gathers: Vec<Vec<(usize, Option<usize>)>>,
    original_inputs: usize,
    shared_outputs: usize,
    prelude_values: Vec<Rational>,
    outputs: usize,
    components: usize,
    duplicate_constants: bool,
    bound_writes: Vec<Vec<usize>>,
    ports: usize,
}

/// Reusable intermediate storage for [`ParameterRowPlan`].
#[derive(Clone, Debug)]
pub struct ParameterRowWorkspace<T> {
    input: Vec<T>,
    shared_input: Vec<T>,
    shared: Vec<T>,
    output: Vec<T>,
    mapped_output: Vec<T>,
}

impl<T> Default for ParameterRowWorkspace<T> {
    fn default() -> Self {
        Self {
            input: Vec::new(),
            shared_input: Vec::new(),
            shared: Vec::new(),
            output: Vec::new(),
            mapped_output: Vec::new(),
        }
    }
}

impl ParameterRowPlan {
    fn program_bound_writes<T>(&self, body: &ExpressionEvaluator<T>) -> Vec<Vec<usize>> {
        let read_inputs = body
            .instructions
            .iter()
            .flat_map(|(instruction, _)| instruction.input_indices())
            .chain(body.result_indices.iter().copied())
            .filter(|index| *index < body.param_count)
            .map(|index| index / self.components)
            .collect::<HashSet<_>>();
        let live_columns = self
            .bound_positions
            .iter()
            .enumerate()
            .filter_map(|(column, position)| read_inputs.contains(position).then_some(column))
            .collect::<Vec<_>>();
        self.rows
            .iter()
            .enumerate()
            .map(|(index, row)| {
                live_columns
                    .iter()
                    .copied()
                    .filter(|column| index == 0 || row[*column] != self.rows[index - 1][*column])
                    .collect()
            })
            .collect()
    }

    fn validate(&self) -> Result<(), EvaluationError> {
        let mut positions = self
            .physical_positions
            .iter()
            .chain(&self.bound_positions)
            .copied()
            .collect::<Vec<_>>();
        positions.sort_unstable();
        let valid = self.components != 0
            && positions.len() == self.original_inputs
            && positions
                .iter()
                .enumerate()
                .all(|(index, position)| *position == index)
            && self
                .rows
                .iter()
                .all(|row| row.len() == self.bound_positions.len())
            && self.gathers.len() == self.rows.len()
            && self.bound_writes.len() == self.rows.len()
            && self.bound_writes.iter().all(|writes| {
                writes
                    .iter()
                    .all(|column| *column < self.bound_positions.len())
                    && writes.iter().collect::<HashSet<_>>().len() == writes.len()
            })
            && self.gathers.iter().all(|row| {
                row.iter().all(|(port, output)| {
                    *port < self.ports && output.is_none_or(|output| output < self.shared_outputs)
                }) && row
                    .iter()
                    .map(|(port, _)| port)
                    .collect::<HashSet<_>>()
                    .len()
                    == row.len()
            })
            && [
                self.physical_positions.len(),
                self.shared_outputs,
                self.outputs,
            ]
            .iter()
            .all(|count| count.checked_mul(self.components).is_some())
            && self
                .original_inputs
                .checked_add(self.ports)
                .and_then(|count| count.checked_mul(self.components))
                .is_some()
            && self
                .physical_positions
                .len()
                .checked_add(self.prelude_values.len())
                .and_then(|count| count.checked_mul(self.components))
                .is_some();
        if !valid {
            return Err(EvaluationError::EvaluatorConstructionFailed {
                expression: Atom::Zero,
                reason: "invalid parameter-row scheduling metadata".into(),
            });
        }
        Ok(())
    }

    /// Validate the interface and derived coefficient-write schedule against
    /// both native programs. Call this after restoring the plan and programs
    /// separately, before compiling or executing backend-specific programs.
    pub fn validate_programs<T>(
        &self,
        prelude: &ExpressionEvaluator<T>,
        body: &ExpressionEvaluator<T>,
    ) -> Result<(), EvaluationError> {
        self.validate()?;
        let valid = (self.physical_positions.len() + self.prelude_values.len()) * self.components
            == prelude.param_count
            && self.shared_outputs * self.components == prelude.result_indices.len()
            && (self.original_inputs + self.ports) * self.components == body.param_count
            && self.outputs * self.components == body.result_indices.len();
        if !valid {
            return Err(EvaluationError::EvaluatorConstructionFailed {
                expression: Atom::Zero,
                reason: "inconsistent parameter-row program interfaces".into(),
            });
        }
        if self.bound_writes != self.program_bound_writes(body) {
            return Err(EvaluationError::EvaluatorConstructionFailed {
                expression: Atom::Zero,
                reason: "parameter-row coefficient writes disagree with the program and rows"
                    .into(),
            });
        }
        Ok(())
    }

    /// Number of physical input components accepted by the complete plan.
    pub fn get_input_len(&self) -> usize {
        self.physical_positions.len() * self.components
    }

    /// Number of summed output components.
    pub fn get_output_len(&self) -> usize {
        self.outputs * self.components
    }

    /// The exact rows, in execution order.
    pub fn rows(&self) -> &[Vec<Rational>] {
        &self.rows
    }

    /// Run backend callbacks using the library's binding and reduction plan.
    ///
    /// `coefficient` converts exact values directly to the requested numerical
    /// precision. The prelude runs once; each row invokes the row program once.
    /// Inactive gathers are reset to zero, including all derivative components.
    pub fn evaluate_with<T: Clone + AddAssign<T>>(
        &self,
        physical: &[T],
        output: &mut [T],
        workspace: &mut ParameterRowWorkspace<T>,
        coefficient: impl Fn(&Rational) -> T,
        prelude: impl FnMut(&[T], &mut [T]) -> Result<(), EvaluationError>,
        body: impl FnMut(&[T], &mut [T]) -> Result<(), EvaluationError>,
    ) -> Result<(), EvaluationError> {
        self.evaluate_with_input_mapping(
            physical,
            output,
            workspace,
            coefficient,
            T::clone_from,
            prelude,
            body,
        )
    }

    /// Run the native row schedule with application-owned input/output types.
    ///
    /// Physical inputs are mapped directly into the reusable native workspace.
    /// Reduction uses native values and is mapped to outputs only on success.
    /// The additional reduction buffer is reused across calls; it never caches
    /// values across physical inputs or changes in numerical precision.
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_mapped_with<I, O, T: Clone + AddAssign<T>>(
        &self,
        physical: &[I],
        output: &mut [O],
        workspace: &mut ParameterRowWorkspace<T>,
        coefficient: impl Fn(&Rational) -> T,
        set_input: impl FnMut(&mut T, &I),
        mut set_output: impl FnMut(&mut O, &T),
        prelude: impl FnMut(&[T], &mut [T]) -> Result<(), EvaluationError>,
        body: impl FnMut(&[T], &mut [T]) -> Result<(), EvaluationError>,
    ) -> Result<(), EvaluationError> {
        if physical.len() != self.get_input_len() {
            return Err(EvaluationError::InvalidParameterCount {
                expected: self.get_input_len(),
                actual: physical.len(),
            });
        }
        if output.len() != self.get_output_len() {
            return Err(EvaluationError::InvalidOutputCount {
                expected: self.get_output_len(),
                actual: output.len(),
            });
        }
        let mut native_output = std::mem::take(&mut workspace.mapped_output);
        native_output.resize_with(output.len(), || coefficient(&Rational::zero()));
        let result = self.evaluate_with_input_mapping(
            physical,
            &mut native_output,
            workspace,
            coefficient,
            set_input,
            prelude,
            body,
        );
        if result.is_ok() {
            for (destination, source) in output.iter_mut().zip(&native_output) {
                set_output(destination, source);
            }
        }
        workspace.mapped_output = native_output;
        result
    }

    #[allow(clippy::too_many_arguments)]
    fn evaluate_with_input_mapping<I, T: Clone + AddAssign<T>>(
        &self,
        physical: &[I],
        output: &mut [T],
        workspace: &mut ParameterRowWorkspace<T>,
        coefficient: impl Fn(&Rational) -> T,
        mut set_input: impl FnMut(&mut T, &I),
        mut prelude: impl FnMut(&[T], &mut [T]) -> Result<(), EvaluationError>,
        mut body: impl FnMut(&[T], &mut [T]) -> Result<(), EvaluationError>,
    ) -> Result<(), EvaluationError> {
        if physical.len() != self.get_input_len() {
            return Err(EvaluationError::InvalidParameterCount {
                expected: self.get_input_len(),
                actual: physical.len(),
            });
        }
        if output.len() != self.get_output_len() {
            return Err(EvaluationError::InvalidOutputCount {
                expected: self.get_output_len(),
                actual: output.len(),
            });
        }
        let zero = coefficient(&Rational::zero());
        output.fill(zero.clone());
        if self.rows.is_empty() {
            return Ok(());
        }
        workspace.input.resize(
            (self.original_inputs + self.ports) * self.components,
            zero.clone(),
        );
        workspace.input.fill(zero.clone());
        workspace
            .shared
            .resize(self.shared_outputs * self.components, zero.clone());
        workspace.output.resize(output.len(), zero.clone());
        for (source, destination) in self.physical_positions.iter().enumerate() {
            for (destination, source) in workspace.input
                [destination * self.components..(destination + 1) * self.components]
                .iter_mut()
                .zip(&physical[source * self.components..(source + 1) * self.components])
            {
                set_input(destination, source);
            }
        }
        if self.shared_outputs != 0 {
            workspace.shared_input.clear();
            for position in &self.physical_positions {
                workspace.shared_input.extend_from_slice(
                    &workspace.input[position * self.components..(position + 1) * self.components],
                );
            }
            for value in &self.prelude_values {
                let value = coefficient(value);
                workspace.shared_input.push(value.clone());
                workspace.shared_input.extend((1..self.components).map(|_| {
                    if self.duplicate_constants {
                        value.clone()
                    } else {
                        zero.clone()
                    }
                }));
            }
            prelude(&workspace.shared_input, &mut workspace.shared)?;
        }
        for (row_index, (row, gathers)) in self.rows.iter().zip(&self.gathers).enumerate() {
            for column in &self.bound_writes[row_index] {
                let position = self.bound_positions[*column];
                let value = coefficient(&row[*column]);
                let target = &mut workspace.input
                    [position * self.components..(position + 1) * self.components];
                if self.duplicate_constants {
                    target.fill(value);
                } else {
                    target[0] = value;
                }
            }
            for (port, source) in gathers {
                let start = (self.original_inputs + port) * self.components;
                let target = &mut workspace.input[start..start + self.components];
                if let Some(source) = source {
                    target.clone_from_slice(
                        &workspace.shared[source * self.components..(source + 1) * self.components],
                    );
                } else {
                    target.fill(zero.clone());
                }
            }
            body(&workspace.input, &mut workspace.output)?;
            if row_index == 0 {
                output.clone_from_slice(&workspace.output);
            } else {
                for (total, value) in output.iter_mut().zip(&workspace.output) {
                    *total += value.clone();
                }
            }
        }
        Ok(())
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for ParameterRowPlan {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (
            &self.physical_positions,
            &self.bound_positions,
            &self.rows,
            &self.gathers,
            self.original_inputs,
            self.shared_outputs,
            &self.prelude_values,
            self.outputs,
            self.components,
            self.duplicate_constants,
            &self.bound_writes,
            self.ports,
        )
            .serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for ParameterRowPlan {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (
            physical_positions,
            bound_positions,
            rows,
            gathers,
            original_inputs,
            shared_outputs,
            prelude_values,
            outputs,
            components,
            duplicate_constants,
            bound_writes,
            ports,
        ) = serde::Deserialize::deserialize(deserializer)?;
        let result = Self {
            physical_positions,
            bound_positions,
            rows,
            gathers,
            original_inputs,
            shared_outputs,
            prelude_values,
            outputs,
            components,
            duplicate_constants,
            bound_writes,
            ports,
        };
        result.validate().map_err(serde::de::Error::custom)?;
        Ok(result)
    }
}

#[cfg(feature = "bincode")]
impl<Context> bincode::Decode<Context> for ParameterRowPlan {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let (
            physical_positions,
            bound_positions,
            rows,
            gathers,
            original_inputs,
            shared_outputs,
            prelude_values,
            outputs,
            components,
            duplicate_constants,
            bound_writes,
            ports,
        ) = bincode::Decode::decode(decoder)?;
        let result = Self {
            physical_positions,
            bound_positions,
            rows,
            gathers,
            original_inputs,
            shared_outputs,
            prelude_values,
            outputs,
            components,
            duplicate_constants,
            bound_writes,
            ports,
        };
        result
            .validate()
            .map_err(|error| bincode::error::DecodeError::OtherString(error.to_string()))?;
        Ok(result)
    }
}

#[cfg(feature = "bincode")]
bincode::impl_borrow_decode!(ParameterRowPlan);

/// Ordinary shared and row evaluators together with their library-owned plan.
#[derive(Clone)]
pub struct ParameterRowEvaluator<T> {
    plan: ParameterRowPlan,
    prelude: ExpressionEvaluator<T>,
    body: ExpressionEvaluator<T>,
    workspace: ParameterRowWorkspace<T>,
}

impl<T> ParameterRowEvaluator<T> {
    /// Restore a schedule and its programs, validating their shared interface.
    pub fn from_parts(
        plan: ParameterRowPlan,
        prelude: ExpressionEvaluator<T>,
        body: ExpressionEvaluator<T>,
    ) -> Result<Self, EvaluationError> {
        plan.validate_programs(&prelude, &body)?;
        Ok(Self {
            plan,
            prelude,
            body,
            workspace: ParameterRowWorkspace::default(),
        })
    }

    /// Borrow the execution plan.
    pub fn plan(&self) -> &ParameterRowPlan {
        &self.plan
    }

    /// Separate the backend-independent schedule and its ordinary programs.
    pub fn into_parts(
        self,
    ) -> (
        ParameterRowPlan,
        ExpressionEvaluator<T>,
        ExpressionEvaluator<T>,
    ) {
        (self.plan, self.prelude, self.body)
    }
}

impl<T: Default + Clone + PartialEq> ParameterRowEvaluator<T> {
    /// Vectorize both programs and their schedule.
    ///
    /// Input assumptions refer to the plan's physical inputs. Exact row values
    /// have zero derivative components. Shared ports carry the prelude's full
    /// vectors, including any components proved to be identically zero.
    pub fn vectorize<V: Vectorize<T> + ?Sized>(mut self, vectorizer: &V) -> Result<Self, String> {
        if self.plan.components != 1 {
            return Err("A parameter-row plan is already vectorized".into());
        }
        let dim = vectorizer.get_dimension();
        let instructions = InstructionList {
            instructions: Vec::new(),
            constants: Vec::new(),
            unknown_constants: Vec::new(),
            dim,
        };
        let mut zeros = Vec::new();
        for (physical, original) in self.plan.physical_positions.iter().enumerate() {
            for component in 0..dim {
                if vectorizer.is_zero(&Slot::Param(physical * dim + component), &instructions) {
                    zeros.push((*original, component));
                }
            }
        }
        if !vectorizer.duplicate_constants() {
            zeros.extend(
                self.plan
                    .bound_positions
                    .iter()
                    .flat_map(|parameter| (1..dim).map(move |component| (*parameter, component))),
            );
        }
        let mut prelude_zeros = Vec::new();
        for physical in 0..self.plan.physical_positions.len() {
            for component in 0..dim {
                if vectorizer.is_zero(&Slot::Param(physical * dim + component), &instructions) {
                    prelude_zeros.push((physical, component));
                }
            }
        }
        if !vectorizer.duplicate_constants() {
            let physical_count = self.plan.physical_positions.len();
            prelude_zeros.extend((0..self.plan.prelude_values.len()).flat_map(|parameter| {
                (1..dim).map(move |component| (parameter + physical_count, component))
            }));
        }
        let prelude_vectorizer = vectorizer.rebind_parameters(prelude_zeros)?;
        self.prelude = self.prelude.vectorize(prelude_vectorizer.as_ref())?;
        for port in 0..self.plan.ports {
            for component in 0..dim {
                if self
                    .plan
                    .gathers
                    .iter()
                    .flatten()
                    .filter(|(index, _)| *index == port)
                    .all(|(_, source)| {
                        source.is_none_or(|output| {
                            let slot = self.prelude.result_indices[output * dim + component];
                            slot >= self.prelude.param_count
                                && slot < self.prelude.reserved_indices
                                && self.prelude.stack[slot] == T::default()
                                && !self.prelude.external_fns.iter().any(|function| {
                                    function.constant_index == Some(slot - self.prelude.param_count)
                                })
                        })
                    })
                {
                    zeros.push((self.plan.original_inputs + port, component));
                }
            }
        }
        let rebound = vectorizer.rebind_parameters(zeros)?;
        self.body = self.body.vectorize(rebound.as_ref())?;
        self.plan.components = dim;
        self.plan.duplicate_constants = vectorizer.duplicate_constants();
        self.plan.bound_writes = self.plan.program_bound_writes(&self.body);
        self.workspace = ParameterRowWorkspace::default();
        Ok(self)
    }
}

#[cfg(feature = "serde")]
impl<T: serde::Serialize> serde::Serialize for ParameterRowEvaluator<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (&self.plan, &self.prelude, &self.body).serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: serde::Deserialize<'de> + EvaluationDomain> serde::Deserialize<'de>
    for ParameterRowEvaluator<T>
{
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (plan, prelude, body) = serde::Deserialize::deserialize(deserializer)?;
        Self::from_parts(plan, prelude, body).map_err(serde::de::Error::custom)
    }
}

#[cfg(feature = "bincode")]
impl<T: bincode::Encode> bincode::Encode for ParameterRowEvaluator<T> {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        bincode::Encode::encode(&(&self.plan, &self.prelude, &self.body), encoder)
    }
}

#[cfg(feature = "bincode")]
impl<Context, T: bincode::Decode<Context> + EvaluationDomain> bincode::Decode<Context>
    for ParameterRowEvaluator<T>
{
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let (plan, prelude, body) = bincode::Decode::decode(decoder)?;
        Self::from_parts(plan, prelude, body)
            .map_err(|error| bincode::error::DecodeError::OtherString(error.to_string()))
    }
}

impl<T: Default> ParameterRowEvaluator<T> {
    /// Map both programs to another numerical domain at a specified precision.
    pub fn map_coeff_with_prec<U: EvaluationDomain>(
        self,
        map: &impl Fn(&T) -> U,
        binary_prec: u32,
    ) -> ParameterRowEvaluator<U> {
        ParameterRowEvaluator {
            plan: self.plan,
            prelude: self.prelude.map_coeff_with_prec(map, binary_prec),
            body: self.body.map_coeff_with_prec(map, binary_prec),
            workspace: ParameterRowWorkspace::default(),
        }
    }

    /// Map both programs to a fixed-precision numerical domain.
    pub fn map_coeff<U: EvaluationDomain>(
        self,
        map: &impl Fn(&T) -> U,
    ) -> ParameterRowEvaluator<U> {
        ParameterRowEvaluator {
            plan: self.plan,
            prelude: self.prelude.map_coeff(map),
            body: self.body.map_coeff(map),
            workspace: ParameterRowWorkspace::default(),
        }
    }
}

impl<T: Real> ParameterRowEvaluator<T> {
    /// Evaluate the ordered row sum with mapped application storage.
    pub fn evaluate_mapped<I, O>(
        &mut self,
        input: &[I],
        output: &mut [O],
        coefficient: impl Fn(&Rational) -> T,
        set_input: impl FnMut(&mut T, &I),
        set_output: impl FnMut(&mut O, &T),
    ) -> Result<(), EvaluationError> {
        self.plan.evaluate_mapped_with(
            input,
            output,
            &mut self.workspace,
            coefficient,
            set_input,
            set_output,
            |input, output| self.prelude.try_evaluate(input, output),
            |input, output| self.body.try_evaluate(input, output),
        )
    }

    /// Evaluate all rows and sum their outputs in row order.
    ///
    /// The converter must use the same precision as the physical inputs. It is
    /// also used for exact zero, so arbitrary-precision buffers retain that precision.
    pub fn evaluate(
        &mut self,
        input: &[T],
        output: &mut [T],
        coefficient: impl Fn(&Rational) -> T,
    ) -> Result<(), EvaluationError> {
        self.plan.evaluate_with(
            input,
            output,
            &mut self.workspace,
            coefficient,
            |input, output| self.prelude.try_evaluate(input, output),
            |input, output| self.body.try_evaluate(input, output),
        )
    }
}

impl ParameterRowEvaluator<Complex<Rational>> {
    /// Schedule exact rows for an existing native evaluator.
    ///
    /// `bound_positions` are distinct indices in the evaluator's input list.
    /// Other inputs remain physical inputs in their original relative order.
    /// This entry point reuses an already prepared native program, including
    /// resolved function scopes, instead of reconstructing symbolic functions.
    /// Safe row-invariant instructions inside retained functions are extracted
    /// into shared scalar arguments; their varying kernels remain retained and
    /// are reused for all rows with the same argument-dependency pattern.
    pub fn new(
        mut evaluator: ExpressionEvaluator<Complex<Rational>>,
        bound_positions: Vec<usize>,
        rows: Vec<Vec<Rational>>,
    ) -> Result<Self, EvaluationError> {
        if bound_positions
            .iter()
            .any(|position| *position >= evaluator.param_count)
            || bound_positions
                .iter()
                .copied()
                .collect::<HashSet<_>>()
                .len()
                != bound_positions.len()
        {
            return Err(EvaluationError::EvaluatorConstructionFailed {
                expression: Atom::Zero,
                reason: "parameter-row input indices must be distinct and in bounds".into(),
            });
        }
        for row in &rows {
            if row.len() != bound_positions.len() {
                return Err(EvaluationError::InvalidParameterCount {
                    expected: bound_positions.len(),
                    actual: row.len(),
                });
            }
        }
        evaluator.undo_stack_optimization();
        retained::split_calls(&mut evaluator, &bound_positions, &rows)?;
        let physical_positions = (0..evaluator.param_count)
            .filter(|position| !bound_positions.contains(position))
            .collect::<Vec<_>>();
        let (prelude, body, gathers, prelude_values) =
            partition(&evaluator, &physical_positions, &bound_positions, &rows)?;
        let ports = gathers.first().map_or(0, Vec::len);
        let mut previous = vec![None; ports];
        let gathers = gathers
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .enumerate()
                    .filter_map(|(port, output)| {
                        if previous[port] == output {
                            None
                        } else {
                            previous[port] = output;
                            Some((port, output))
                        }
                    })
                    .collect()
            })
            .collect();
        let mut plan = ParameterRowPlan {
            physical_positions,
            bound_positions,
            rows,
            gathers,
            original_inputs: evaluator.param_count,
            shared_outputs: prelude.get_output_len(),
            prelude_values,
            outputs: evaluator.get_output_len(),
            components: 1,
            duplicate_constants: false,
            bound_writes: Vec::new(),
            ports,
        };
        plan.bound_writes = plan.program_bound_writes(&body);
        Self::from_parts(plan, prelude, body)
    }
}

fn output(instruction: &Instr) -> Option<usize> {
    match instruction {
        Instr::Add(output, _)
        | Instr::Mul(output, _)
        | Instr::Pow(output, _, _)
        | Instr::Powf(output, _, _)
        | Instr::Div(output, _, _)
        | Instr::Select(output, _, _, _)
        | Instr::BuiltinFun(output, _, _)
        | Instr::ExternalFun(output, _, _)
        | Instr::Join(output, _, _, _) => Some(*output),
        _ => None,
    }
}

fn remap(instruction: &Instr, target: usize, mut map: impl FnMut(usize) -> usize) -> Instr {
    match instruction {
        Instr::Add(_, args) => Instr::Add(target, args.iter().map(|input| map(*input)).collect()),
        Instr::Mul(_, args) => Instr::Mul(target, args.iter().map(|input| map(*input)).collect()),
        Instr::Pow(_, input, power) => Instr::Pow(target, map(*input), *power),
        Instr::Powf(_, base, exponent) => Instr::Powf(target, map(*base), map(*exponent)),
        Instr::Div(_, numerator, denominator) => {
            Instr::Div(target, map(*numerator), map(*denominator))
        }
        Instr::Select(_, condition, yes, no) => {
            Instr::Select(target, map(*condition), map(*yes), map(*no))
        }
        Instr::BuiltinFun(_, function, input) => Instr::BuiltinFun(target, *function, map(*input)),
        Instr::ExternalFun(_, function, args) => Instr::ExternalFun(
            target,
            *function,
            args.iter().map(|input| map(*input)).collect(),
        ),
        Instr::Join(_, condition, yes, no) => {
            Instr::Join(target, map(*condition), map(*yes), map(*no))
        }
        Instr::IfElse(condition, label) => Instr::IfElse(map(*condition), *label),
        Instr::Goto(label) => Instr::Goto(*label),
        Instr::Label(label) => Instr::Label(*label),
    }
}

fn and(left: Option<bool>, right: Option<bool>) -> Option<bool> {
    match (left, right) {
        (Some(false), _) | (_, Some(false)) => Some(false),
        (Some(true), Some(true)) => Some(true),
        _ => None,
    }
}

struct RowAnalysis {
    dependencies: Vec<BTreeSet<usize>>,
    active: Vec<Vec<Option<bool>>>,
    safe: Vec<Vec<bool>>,
    /// Selected scalar data are independent of coefficient inputs. A proven
    /// selector may choose different physical-only values for different rows.
    invariant: Vec<Vec<bool>>,
    conditions: HashMap<usize, Vec<Option<bool>>>,
}

impl RowAnalysis {
    fn new(
        evaluator: &ExpressionEvaluator<Complex<Rational>>,
        columns: &[usize],
        rows: &[Vec<Rational>],
    ) -> Self {
        Self::with_bindings(evaluator, columns, rows.len(), |row, column| {
            Some(&rows[row][column])
        })
    }

    fn with_bindings<'a>(
        evaluator: &ExpressionEvaluator<Complex<Rational>>,
        columns: &[usize],
        row_count: usize,
        binding: impl Fn(usize, usize) -> Option<&'a Rational>,
    ) -> Self {
        let slots = evaluator.reserved_indices + evaluator.instructions.len();
        let mut result = Self {
            dependencies: vec![BTreeSet::new(); slots],
            active: vec![vec![Some(true); row_count]; slots],
            safe: vec![vec![true; row_count]; slots],
            invariant: vec![vec![true; row_count]; slots],
            conditions: HashMap::default(),
        };
        for (column, slot) in columns.iter().enumerate() {
            result.dependencies[*slot].insert(column);
            result.invariant[*slot].fill(false);
        }
        for (instruction, _) in &evaluator.instructions {
            if let Some(target) = output(instruction) {
                result.dependencies[target] = instruction
                    .input_indices()
                    .into_iter()
                    .flat_map(|input| result.dependencies[input].iter().copied())
                    .collect();
            }
        }
        // Only branch predicates need value proofs. In particular, do not
        // interpret retained numerator bodies merely to classify their data.
        let mut proof_inputs = HashSet::default();
        let mut pending = evaluator
            .instructions
            .iter()
            .filter_map(|(instruction, _)| match instruction {
                Instr::IfElse(condition, _) => Some(*condition),
                _ => None,
            })
            .collect::<Vec<_>>();
        while let Some(slot) = pending.pop() {
            if !proof_inputs.insert(slot) || slot < evaluator.reserved_indices {
                continue;
            }
            pending.extend(
                evaluator.instructions[slot - evaluator.reserved_indices]
                    .0
                    .input_indices(),
            );
        }
        let mut booleans = BooleanEvaluation::new(&evaluator.external_fns);
        for row_index in 0..row_count {
            let mut known = vec![None; slots];
            for (column, slot) in columns.iter().enumerate() {
                if proof_inputs.contains(slot) {
                    known[*slot] = binding(row_index, column).and_then(|value| {
                        NativeBoolean::from_exact(&Complex::new(value.clone(), Rational::zero()))
                    });
                }
            }
            for (slot, value) in known
                .iter_mut()
                .enumerate()
                .take(evaluator.reserved_indices)
                .skip(evaluator.param_count)
            {
                if proof_inputs.contains(&slot)
                    && !evaluator.external_fns.iter().any(|function| {
                        function.constant_index == Some(slot - evaluator.param_count)
                    })
                {
                    *value = NativeBoolean::from_exact(&evaluator.stack[slot]);
                }
            }
            let mut reach = Some(true);
            let mut branches = Vec::<(Option<bool>, Option<bool>, Option<Label>)>::new();
            for (instruction, _) in &evaluator.instructions {
                match instruction {
                    Instr::IfElse(condition, _) => {
                        let selected = known[*condition].as_ref().map(|value| !value.is_zero());
                        result
                            .conditions
                            .entry(*condition)
                            .or_insert_with(|| vec![None; row_count])[row_index] = selected;
                        branches.push((reach, selected, None));
                        reach = and(reach, selected);
                    }
                    Instr::Goto(label) => {
                        let (parent, selected, exit) = branches.last_mut().unwrap();
                        *exit = Some(*label);
                        reach = and(*parent, selected.map(|value| !value));
                    }
                    Instr::Join(target, condition, yes, no) => {
                        let selected = known[*condition].as_ref().map(|value| !value.is_zero());
                        result
                            .conditions
                            .entry(*condition)
                            .or_insert_with(|| vec![None; row_count])[row_index] = selected;
                        known[*target] = selected
                            .and_then(|yes_branch| known[if yes_branch { *yes } else { *no }]);
                        result.active[*target][row_index] = reach;
                        result.invariant[*target][row_index] = selected.is_some_and(|yes_branch| {
                            result.invariant[if yes_branch { *yes } else { *no }][row_index]
                        });
                        result.safe[*target][row_index] = reach == Some(true)
                            && selected.is_some_and(|yes_branch| {
                                result.safe[if yes_branch { *yes } else { *no }][row_index]
                            });
                    }
                    Instr::Label(label) => {
                        // One conditional may produce several phi values after
                        // native vectorization. Restore its scope at the exit
                        // label once, rather than popping it for every Join.
                        if branches
                            .last()
                            .is_some_and(|(_, _, exit)| *exit == Some(*label))
                        {
                            let (parent, _, _) = branches.pop().unwrap();
                            reach = parent;
                        }
                    }
                    _ => {
                        let target = output(instruction).unwrap();
                        let args = instruction.input_indices();
                        result.active[target][row_index] = reach;
                        result.invariant[target][row_index] =
                            args.iter().all(|input| result.invariant[*input][row_index]);
                        result.safe[target][row_index] = reach == Some(true)
                            && args.iter().all(|input| result.safe[*input][row_index]);
                        if reach == Some(false) {
                            continue;
                        }
                        if proof_inputs.contains(&target) {
                            known[target] = booleans.value(instruction, &known);
                        }
                    }
                }
            }
        }
        result
    }
}

// The single construction caller consumes both programs and their matching
// gather/coefficient tables together; a separate wrapper adds no shared API.
#[allow(clippy::type_complexity)]
fn partition(
    evaluator: &ExpressionEvaluator<Complex<Rational>>,
    physical: &[usize],
    columns: &[usize],
    rows: &[Vec<Rational>],
) -> Result<
    (
        ExpressionEvaluator<Complex<Rational>>,
        ExpressionEvaluator<Complex<Rational>>,
        Vec<Vec<Option<usize>>>,
        Vec<Rational>,
    ),
    EvaluationError,
> {
    let analysis = RowAnalysis::new(evaluator, columns, rows);
    let mut candidates = HashSet::default();
    for (instruction, _) in &evaluator.instructions {
        let Some(target) = output(instruction) else {
            continue;
        };
        if !analysis.active[target]
            .iter()
            .zip(&analysis.safe[target])
            .all(|(active, safe)| *active == Some(false) || *safe)
        {
            continue;
        }
        // A physical-only scalar may occur in separate coefficient-selected
        // leaves, each visited by only one row. Move those leaves together so
        // native CSE can share their common factors across the complete sum.
        // Unknown physical branches failed the safety check above; their
        // potentially singular inputs must remain inside their runtime guard.
        if rows.len() > 1
            && analysis.active[target]
                .iter()
                .zip(&analysis.invariant[target])
                .all(|(active, invariant)| *active == Some(false) || *invariant)
            && analysis.active[target].contains(&Some(true))
        {
            candidates.insert(target);
            continue;
        }
        if matches!(instruction, Instr::Join(..)) {
            continue;
        }
        let costly = matches!(
            instruction,
            Instr::BuiltinFun(..)
                | Instr::ExternalFun(..)
                | Instr::Powf(..)
                | Instr::Div(..)
                | Instr::Pow(_, _, ..=-1)
        );
        if !analysis.dependencies[target].is_empty() && !costly {
            continue;
        }
        let mut bindings = HashSet::default();
        if rows
            .iter()
            .zip(&analysis.active[target])
            .any(|(row, active)| {
                *active == Some(true)
                    && !bindings.insert(
                        analysis.dependencies[target]
                            .iter()
                            .map(|column| row[*column].clone())
                            .collect::<Vec<_>>(),
                    )
            })
        {
            candidates.insert(target);
        }
    }
    let mut used = HashSet::default();
    let mut ports = BTreeSet::new();
    let mut pending = evaluator.result_indices.clone();
    pending.extend(evaluator.instructions.iter().filter_map(
        |(instruction, _)| match instruction {
            Instr::IfElse(condition, _) => Some(*condition),
            _ => None,
        },
    ));
    while let Some(slot) = pending.pop() {
        if slot < evaluator.reserved_indices || !used.insert(slot) {
            continue;
        }
        if candidates.contains(&slot) {
            ports.insert(slot);
        } else {
            pending.extend(
                evaluator.instructions[slot - evaluator.reserved_indices]
                    .0
                    .input_indices(),
            );
        }
    }
    let ports = ports.into_iter().collect::<Vec<_>>();
    let mut prelude = Prelude::new(evaluator, physical, columns, rows, &analysis, &ports);
    let mut gathers = vec![Vec::with_capacity(ports.len()); rows.len()];
    let mut output_indices = HashMap::default();
    let mut shared_outputs = Vec::new();
    for port in &ports {
        for (row_index, row) in rows.iter().enumerate() {
            let gather = if analysis.active[*port][row_index] == Some(true) {
                let slot = prelude.copy(*port, row_index, row);
                Some(*output_indices.entry(slot).or_insert_with(|| {
                    let index = shared_outputs.len();
                    shared_outputs.push(slot);
                    index
                }))
            } else {
                None
            };
            gathers[row_index].push(gather);
        }
    }
    let prelude_values = prelude.values.clone();
    let mut prelude = prelude.finish(shared_outputs);
    prelude.remove_common_instructions();
    prelude.optimize_stack();
    prelude.prune_unused_external_functions();

    let mut body = body_with_shared_inputs(evaluator, &ports, &used);
    body.prune_output_dependencies(false);
    body.prune_unused_external_functions();
    Ok((prelude, body, gathers, prelude_values))
}

fn body_with_shared_inputs(
    evaluator: &ExpressionEvaluator<Complex<Rational>>,
    ports: &[usize],
    used: &HashSet<usize>,
) -> ExpressionEvaluator<Complex<Rational>> {
    let port_map = ports
        .iter()
        .enumerate()
        .map(|(index, slot)| (*slot, index))
        .collect::<HashMap<_, _>>();
    let map = |slot: usize| {
        if let Some(port) = port_map.get(&slot) {
            evaluator.param_count + port
        } else if slot < evaluator.param_count {
            slot
        } else {
            slot + ports.len()
        }
    };
    let mut body = evaluator.clone();
    body.instructions = evaluator
        .instructions
        .iter()
        .filter_map(|(instruction, phase)| {
            let target = output(instruction);
            if target
                .is_some_and(|target| !used.contains(&target) || port_map.contains_key(&target))
            {
                return None;
            }
            Some((remap(instruction, target.map_or(0, map), map), *phase))
        })
        .collect();
    body.param_count += ports.len();
    body.reserved_indices += ports.len();
    body.stack.splice(
        evaluator.param_count..evaluator.param_count,
        std::iter::repeat_n(Complex::default(), ports.len()),
    );
    body.result_indices = evaluator
        .result_indices
        .iter()
        .map(|slot| map(*slot))
        .collect();
    body.fix_labels();
    body
}

struct Prelude<'a> {
    source: &'a ExpressionEvaluator<Complex<Rational>>,
    analysis: &'a RowAnalysis,
    physical: HashMap<usize, usize>,
    columns: HashMap<usize, usize>,
    constants: Vec<Complex<Rational>>,
    row_values: HashMap<(usize, Rational), usize>,
    values: Vec<Rational>,
    instructions: Vec<(Instr, ComplexPhase)>,
    cache: HashMap<(usize, Vec<Rational>), usize>,
}

impl<'a> Prelude<'a> {
    fn new(
        source: &'a ExpressionEvaluator<Complex<Rational>>,
        physical: &[usize],
        columns: &[usize],
        rows: &[Vec<Rational>],
        analysis: &'a RowAnalysis,
        ports: &[usize],
    ) -> Self {
        let constants = source.stack[source.param_count..source.reserved_indices].to_vec();
        let mut row_values = HashMap::default();
        let mut values = Vec::new();
        for port in ports {
            for (row_index, row) in rows.iter().enumerate() {
                if analysis.active[*port][row_index] != Some(true) {
                    continue;
                }
                for column in &analysis.dependencies[*port] {
                    let value = &row[*column];
                    row_values
                        .entry((*column, value.clone()))
                        .or_insert_with(|| {
                            let index = values.len();
                            values.push(value.clone());
                            index
                        });
                }
            }
        }
        Self {
            source,
            analysis,
            physical: physical
                .iter()
                .enumerate()
                .map(|(index, slot)| (*slot, index))
                .collect(),
            columns: columns
                .iter()
                .enumerate()
                .map(|(index, slot)| (*slot, index))
                .collect(),
            constants,
            row_values,
            values,
            instructions: Vec::new(),
            cache: HashMap::default(),
        }
    }

    fn copy(&mut self, slot: usize, row_index: usize, row: &[Rational]) -> usize {
        if let Some(input) = self.physical.get(&slot) {
            return *input;
        }
        if let Some(column) = self.columns.get(&slot) {
            return self.physical.len() + self.row_values[&(*column, row[*column].clone())];
        }
        if slot < self.source.reserved_indices {
            return self.physical.len() + self.values.len() + slot - self.source.param_count;
        }
        let values = self.analysis.dependencies[slot]
            .iter()
            .map(|column| row[*column].clone())
            .collect();
        let key = (slot, values);
        if let Some(mapped) = self.cache.get(&key) {
            return *mapped;
        }
        let (instruction, phase) =
            self.source.instructions[slot - self.source.reserved_indices].clone();
        if let Instr::Join(_, condition, yes, no) = instruction {
            let branch = self.analysis.conditions[&condition][row_index]
                .expect("only statically selected branch values enter the prelude");
            let result = self.copy(if branch { yes } else { no }, row_index, row);
            self.cache.insert(key, result);
            return result;
        }
        let mut mapped = HashMap::default();
        for input in instruction.input_indices() {
            mapped.insert(input, self.copy(input, row_index, row));
        }
        let target = self.physical.len()
            + self.values.len()
            + self.constants.len()
            + self.instructions.len();
        self.instructions
            .push((remap(&instruction, target, |input| mapped[&input]), phase));
        self.cache.insert(key, target);
        target
    }

    fn finish(self, outputs: Vec<usize>) -> ExpressionEvaluator<Complex<Rational>> {
        let param_count = self.physical.len() + self.values.len();
        let reserved_indices = param_count + self.constants.len();
        let mut stack = vec![Complex::default(); param_count];
        stack.extend(self.constants);
        stack.resize(
            reserved_indices + self.instructions.len(),
            Complex::default(),
        );
        ExpressionEvaluator {
            stack,
            param_count,
            reserved_indices,
            instructions: self.instructions,
            result_indices: outputs,
            external_fns: self.source.external_fns.clone(),
            settings: self.source.settings.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse, symbol};

    #[test]
    fn selected_scalar_leaves_share_factors_without_exposing_physical_poles() {
        let params = [
            parse!("selected_scalar::x"),
            parse!("selected_scalar::c"),
            parse!("selected_scalar::s"),
            parse!("selected_scalar::g"),
        ];
        let numerator = symbol!("selected_scalar::n");
        let selected = Symbol::IF.call_args([
            params[2].clone(),
            parse!("1/(selected_scalar::x+1)+1/(selected_scalar::x+2)"),
            parse!("2/(selected_scalar::x+1)+1/(selected_scalar::x+3)"),
        ]);
        let term = numerator.call(&params[0] + &params[1]) * selected;
        let rows = vec![vec![1.into(), 0.into()], vec![2.into(), 1.into()]];
        for guarded in [false, true] {
            let expression = if guarded {
                Symbol::IF.call_args([params[3].clone(), term.clone(), Atom::Zero])
            } else {
                term.clone()
            };
            let source = expression
                .evaluator(&params)
                .add_function_with_options(
                    numerator,
                    vec![symbol!("selected_scalar::u")],
                    parse!("selected_scalar::u^2+1"),
                    FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
                )
                .unwrap()
                .build()
                .unwrap();
            let source_body = source.external_fns[0].body.clone().unwrap();
            let staged =
                ParameterRowEvaluator::new(source.clone(), vec![1, 2], rows.clone()).unwrap();
            let inversions = |program: &ExpressionEvaluator<Complex<Rational>>| {
                program
                    .instructions
                    .iter()
                    .filter(|(instruction, _)| matches!(instruction, Instr::Pow(_, _, -1)))
                    .count()
            };
            if guarded {
                assert_eq!(inversions(&staged.prelude), 0);
                assert!(inversions(&staged.body) > 0);
            } else {
                assert_eq!(inversions(&staged.prelude), 3);
                assert_eq!(inversions(&staged.body), 0);
            }
            assert!(staged.prelude.external_fns.is_empty());
            assert_eq!(staged.body.external_fns.len(), 1);
            assert!(Arc::ptr_eq(
                &source_body,
                staged.body.external_fns[0].body.as_ref().unwrap()
            ));
            let mut oracle = source.clone().map_coeff(&|value| value.re.to_f64());
            let mut numeric = staged.clone().map_coeff(&|value| value.re.to_f64());
            for (x, gate) in [(0.25, 1.), (2., 1.), (7., 1.), (-1., 0.)] {
                if !guarded && gate == 0. {
                    continue;
                }
                let expected = rows
                    .iter()
                    .map(|row| oracle.evaluate_single(&[x, row[0].to_f64(), row[1].to_f64(), gate]))
                    .sum::<f64>();
                let mut output = [0.];
                numeric
                    .evaluate(&[x, gate], &mut output, Rational::to_f64)
                    .unwrap();
                assert!(expected.is_finite());
                assert!((output[0] - expected).abs() < 1e-12);
                #[cfg(feature = "native_code_generation")]
                for direct in [false, true] {
                    let (plan, prelude, body) = staged
                        .clone()
                        .map_coeff(&|value| value.re.to_f64())
                        .into_parts();
                    let settings = JITCompilationSettings::new()
                        .optimization_level(2)
                        .direct_translation(direct);
                    let mut body = body.jit_compile(settings.clone()).unwrap();
                    let mut prelude = (prelude.get_output_len() != 0)
                        .then(|| prelude.jit_compile(settings).unwrap());
                    plan.evaluate_with(
                        &[x, gate],
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
            if !guarded {
                let never_selected_rows = vec![vec![1.into(), 0.into()], vec![2.into(), 0.into()]];
                let staged =
                    ParameterRowEvaluator::new(source, vec![1, 2], never_selected_rows).unwrap();
                // The x+2 denominator belongs only to the unused positive arm.
                assert_eq!(inversions(&staged.prelude), 2);
                let mut numeric = staged.map_coeff(&|value| value.re.to_f64());
                let mut output = [0.];
                numeric
                    .evaluate(&[-2., 1.], &mut output, Rational::to_f64)
                    .unwrap();
                assert_eq!(output, [-3.]);
            }
        }
    }

    #[test]
    fn stage_registries_keep_only_reachable_calls_and_preserve_shared_leaf_bodies() {
        let params = [parse!("stage_registry::x"), parse!("stage_registry::c")];
        let inner = symbol!("stage_registry::inner");
        let outer = symbol!("stage_registry::outer");
        let exact = parse!(
            "stage_registry::outer(stage_registry::x+stage_registry::c)+sqrt(stage_registry::x+pi)"
        )
        .evaluator(&params)
        .add_function_with_options(
            inner,
            vec![symbol!("stage_registry::u")],
            parse!("1/stage_registry::u"),
            FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
        )
        .unwrap()
        .add_function_with_options(
            outer,
            vec![symbol!("stage_registry::v")],
            parse!("if(stage_registry::v,stage_registry::inner(stage_registry::v),0)"),
            FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
        )
        .unwrap()
        .build()
        .unwrap();
        let leaf = exact
            .external_fns
            .iter()
            .find(|function| function.symbol == inner)
            .unwrap()
            .body
            .clone()
            .unwrap();
        let staged = ParameterRowEvaluator::new(
            exact,
            vec![1],
            vec![vec![1.into()], vec![2.into()], vec![3.into()]],
        )
        .unwrap();
        assert!(staged.prelude.get_output_len() > 0);
        assert!(
            staged
                .prelude
                .external_fns
                .iter()
                .all(|function| function.body.is_none())
        );
        assert!(staged.prelude.external_fns.iter().any(|function| {
            function.symbol == Symbol::PI && function.constant_index.is_some()
        }));
        assert!(
            staged
                .body
                .external_fns
                .iter()
                .all(|function| function.constant_index.is_none())
        );
        assert_eq!(staged.body.external_fns.len(), 2);
        let restored_leaf = staged
            .body
            .external_fns
            .iter()
            .find(|function| function.symbol == inner)
            .unwrap()
            .body
            .as_ref()
            .unwrap();
        assert!(
            Arc::ptr_eq(&leaf, restored_leaf),
            "unchanged leaf bodies must remain shared"
        );
        let mut eager = staged.clone().map_coeff(&|value| value.re.to_f64());
        for x in [-2.0_f64, 3.0] {
            let expected = (1..=3)
                .map(|row| {
                    let argument = x + f64::from(row);
                    if argument == 0. { 0. } else { 1. / argument }
                })
                .sum::<f64>()
                + 3. * (x + std::f64::consts::PI).sqrt();
            let mut output = [0.];
            eager.evaluate(&[x], &mut output, Rational::to_f64).unwrap();
            assert!((output[0] - expected).abs() < 1e-12);
            #[cfg(feature = "native_code_generation")]
            for direct in [false, true] {
                let (plan, prelude, body) = staged
                    .clone()
                    .map_coeff(&|value| value.re.to_f64())
                    .into_parts();
                let cpp = prelude
                    .export_cpp_str::<f64>(
                        "prelude",
                        ExportSettings::new().inline_asm(InlineASM::None),
                    )
                    .unwrap();
                for symbol in [inner, outer] {
                    assert!(
                        !cpp.contains(&format!("{}_", symbol.get_stripped_ascii_name().unwrap()))
                    );
                }
                let settings = JITCompilationSettings::new()
                    .optimization_level(2)
                    .direct_translation(direct);
                let mut prelude = prelude.jit_compile(settings.clone()).unwrap();
                let mut body = body.jit_compile(settings).unwrap();
                plan.evaluate_with(
                    &[x],
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
                assert!((output[0] - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn stage_registry_keeps_external_constants_referenced_only_by_outputs() {
        let parameter = parse!("stage_constant::c");
        let staged = Atom::var(Symbol::PI)
            .evaluator(std::slice::from_ref(&parameter))
            .build_parameter_rows(&[parameter], &[vec![1.into()], vec![2.into()]])
            .unwrap();
        assert_eq!(staged.prelude.get_output_len(), 0);
        assert!(staged.prelude.external_fns.is_empty());
        assert!(staged.body.instructions.is_empty());
        assert_eq!(staged.body.external_fns.len(), 1);
        assert_eq!(staged.body.external_fns[0].symbol, Symbol::PI);
        assert!(staged.body.external_fns[0].constant_index.is_some());
        let mut numeric = staged.map_coeff(&|value| value.re.to_f64());
        let mut output = [0.];
        numeric
            .evaluate(&[], &mut output, Rational::to_f64)
            .unwrap();
        assert_eq!(output, [2. * std::f64::consts::PI]);
    }

    #[test]
    fn row_branch_selection_uses_the_runtime_coefficient_precision() {
        let params = [parse!("staged_rounding::x"), parse!("staged_rounding::q")];
        let near_one = Rational::new(18_014_398_509_481_985_i64, 18_014_398_509_481_984_i64);
        let tiny = Rational::from((1, 10)).pow(400);
        for (offset, value) in [(1, near_one), (0, tiny)] {
            let root =
                Symbol::IF.call_args([&params[1] - offset, params[0].pow(2), &params[0] + 3]);
            let rows = vec![
                vec![Rational::from(offset)],
                vec![Rational::from(offset)],
                vec![value],
            ];
            let program = root.evaluator(&params).build().unwrap();
            let staged = root
                .evaluator(&params)
                .build_parameter_rows(&params[1..], &rows)
                .unwrap();
            let mut baseline64 = program.clone().map_coeff(&|value| value.re.to_f64());
            let mut staged64 = staged.clone().map_coeff(&|value| value.re.to_f64());
            let expected = rows
                .iter()
                .map(|row| baseline64.evaluate_single(&[2., row[0].to_f64()]))
                .sum::<f64>();
            assert_eq!(
                expected, 15.,
                "the final native guard rounds to zero in binary64"
            );
            let mut output = [0.];
            staged64
                .evaluate(&[2.], &mut output, Rational::to_f64)
                .unwrap();
            assert_eq!(output, [expected]);

            let prototype = Float::new(128);
            let mut baseline128 =
                program.map_coeff_with_prec(&|value| prototype.from_rational(&value.re), 128);
            let mut staged128 =
                staged.map_coeff_with_prec(&|value| prototype.from_rational(&value.re), 128);
            // Interface metadata must also be available for variable-precision
            // coefficients, which deliberately do not implement Default.
            assert_eq!(baseline128.get_input_len(), 2);
            assert_eq!(baseline128.get_output_len(), 1);
            assert!(!baseline128.get_constants().is_empty());
            let point = prototype.from_rational(&Rational::from(2));
            let expected = rows.iter().fold(prototype.zero(), |sum, row| {
                sum + baseline128
                    .evaluate_single(&[point.clone(), prototype.from_rational(&row[0])])
            });
            assert_eq!(
                expected.to_f64(),
                14.,
                "the final native guard is nonzero at 128 bits"
            );
            let mut output = [prototype.zero()];
            staged128
                .evaluate(&[point], &mut output, |value| {
                    prototype.from_rational(value)
                })
                .unwrap();
            assert_eq!(output[0], expected);
        }
    }

    #[test]
    fn native_boolean_functions_still_allow_guarded_reuse() {
        let params = [parse!("staged_boolean::x"), parse!("staged_boolean::q")];
        let root = Symbol::IF.call_args([
            parse!("staged_boolean::not(staged_boolean::q)"),
            params[0].pow(2),
            &params[0] + 3,
        ]);
        let staged = root
            .evaluator(&params)
            .add_function_with_options(
                symbol!("staged_boolean::not"),
                vec![symbol!("staged_boolean::a")],
                parse!("1-staged_boolean::a"),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap()
            .build_parameter_rows(
                &params[1..],
                &[
                    vec![0.into()],
                    vec![1.into()],
                    vec![0.into()],
                    vec![1.into()],
                ],
            )
            .unwrap();
        assert!(
            staged.plan.ports >= 2,
            "certified Boolean guards retain shared branch computations"
        );
        let mut staged = staged.map_coeff(&|value| value.re.to_f64());
        let mut output = [0.];
        staged
            .evaluate(&[2.], &mut output, Rational::to_f64)
            .unwrap();
        assert_eq!(output, [18.]);
    }

    #[test]
    fn empty_row_tables_produce_zero_without_visiting_inactive_sources() {
        let parameters = [parse!("staged_empty::x"), parse!("staged_empty::c")];
        let roots = [
            parse!("staged_empty::unregistered(staged_empty::x)"),
            parse!("1/staged_empty::c"),
        ];
        let mut evaluator = Atom::evaluator_multiple(&roots, &parameters)
            .build_parameter_rows(&parameters[1..], &[])
            .unwrap()
            .map_coeff(&|value| value.re.to_f64());
        assert_eq!(evaluator.plan.get_input_len(), 1);
        assert_eq!(evaluator.plan.get_output_len(), 2);
        let mut output = [f64::NAN; 2];
        evaluator
            .evaluate(&[0.], &mut output, Rational::to_f64)
            .unwrap();
        assert_eq!(output, [0.; 2]);
    }

    #[test]
    fn row_schedule_writes_only_live_changed_inputs_and_gathers() {
        let params = [
            parse!("staged_writes::x"),
            parse!("staged_writes::c"),
            parse!("staged_writes::unused"),
        ];
        let root = parse!("staged_writes::x^2+staged_writes::c^2");
        let staged = root
            .evaluator(&params)
            .build_parameter_rows(
                &params[1..],
                &[
                    vec![1.into(), Rational::from((1, 3))],
                    vec![1.into(), Rational::from((2, 3))],
                    vec![2.into(), Rational::from((4, 3))],
                ],
            )
            .unwrap()
            .map_coeff(&|value| value.re.to_f64());
        let (plan, mut prelude, mut body) = staged.into_parts();
        assert_eq!(plan.bound_writes, vec![vec![0], vec![], vec![0]]);
        assert_eq!(
            plan.gathers.iter().map(Vec::len).collect::<Vec<_>>(),
            vec![1, 0, 0]
        );
        assert!(plan.prelude_values.is_empty());
        let conversions = std::cell::Cell::new(0);
        let mut workspace = ParameterRowWorkspace::default();
        let mut output = [0.];
        for (input, expected) in [(3., 33.), (4., 54.)] {
            conversions.set(0);
            plan.evaluate_with(
                &[input],
                &mut output,
                &mut workspace,
                |value| {
                    conversions.set(conversions.get() + 1);
                    value.to_f64()
                },
                |input, output| prelude.try_evaluate(input, output),
                |input, output| body.try_evaluate(input, output),
            )
            .unwrap();
            assert_eq!(output, [expected]);
            assert_eq!(
                conversions.get(),
                3,
                "one zero and two live coefficient writes per sample"
            );
        }
    }

    #[test]
    fn native_rows_share_calls_and_refresh_each_physical_input() {
        let params = [parse!("staged::x"), parse!("staged::c")];
        let root =
            parse!("staged::f(staged::x)+staged::f(staged::x+staged::c)+staged::x/(staged::c+3)");
        let rows = vec![
            vec![1.into()],
            vec![2.into()],
            vec![1.into()],
            vec![2.into()],
        ];
        let make = || {
            root.evaluator(&params)
                .add_function_with_options(
                    symbol!("staged::f"),
                    vec![symbol!("staged::t")],
                    parse!("staged::t^2+1"),
                    FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
                )
                .unwrap()
        };
        let mut baseline = make()
            .build()
            .unwrap()
            .map_coeff(&|value| value.re.to_f64());
        let staged = make().build_parameter_rows(&params[1..], &rows).unwrap();
        assert_eq!(staged.prelude.count_operations().function_calls, 3);
        assert_eq!(staged.body.count_operations().function_calls, 0);
        assert_eq!(staged.prelude.export_instructions().sub_evaluators.len(), 1);
        let mut staged = staged.map_coeff(&|value| value.re.to_f64());
        for x in [4., 7., -2.] {
            let expected = rows
                .iter()
                .map(|row| baseline.evaluate_single(&[x, row[0].to_f64()]))
                .sum::<f64>();
            let mut output = [0.];
            staged
                .evaluate(&[x], &mut output, Rational::to_f64)
                .unwrap();
            assert!((output[0] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn row_guards_and_physical_guards_keep_poles_lazy() {
        let params = [
            parse!("staged_guard::x"),
            parse!("staged_guard::p"),
            parse!("staged_guard::c"),
            parse!("staged_guard::s"),
        ];
        let selected_pole = Symbol::IF.call_args([
            params[3].clone(),
            parse!("1/(staged_guard::x-staged_guard::c)"),
            Atom::Zero,
        ]);
        let physical = Symbol::IF.call_args([
            params[1].clone(),
            Symbol::SQRT.call_args([parse!("staged_guard::x-staged_guard::c")]),
            Atom::Zero,
        ]);
        let root = selected_pole + physical;
        let rows = vec![
            vec![0.into(), 0.into()],
            vec![1.into(), 1.into()],
            vec![0.into(), 0.into()],
            vec![1.into(), 1.into()],
        ];
        let staged = root
            .evaluator(&params)
            .build_parameter_rows(&params[2..], &rows)
            .unwrap();
        assert!(
            staged
                .plan
                .gathers
                .iter()
                .any(|row| row.iter().any(|(_, output)| output.is_none()))
        );
        let mut staged = staged.map_coeff(&|value| value.re.to_f64());
        let mut output = [0.];
        staged
            .evaluate(&[0., 0.], &mut output, Rational::to_f64)
            .unwrap();
        assert_eq!(output, [-2.]);
        staged
            .evaluate(&[4., 1.], &mut output, Rational::to_f64)
            .unwrap();
        let expected = 2. / 3. + 4. + 2. * 3.0_f64.sqrt();
        assert!((output[0] - expected).abs() < 1e-12);
    }

    #[test]
    fn equal_exact_values_preserve_distinct_native_argument_roles() {
        let params = [
            parse!("staged_roles::x"),
            parse!("staged_roles::a"),
            parse!("staged_roles::b"),
        ];
        let root = parse!(
            "staged_roles::f(staged_roles::a*staged_roles::x)-staged_roles::f(staged_roles::b*staged_roles::x)"
        );
        let make = || {
            root.evaluator(&params)
                .add_function_with_options(
                    symbol!("staged_roles::f"),
                    vec![symbol!("staged_roles::t")],
                    parse!("staged_roles::t^2"),
                    FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
                )
                .unwrap()
        };
        let mut baseline = make()
            .build()
            .unwrap()
            .map_coeff(&|value| value.re.to_f64());
        let staged = make()
            .build_parameter_rows(
                &params[1..],
                &[vec![1.into(), 1.into()], vec![1.into(), 1.into()]],
            )
            .unwrap();
        assert_eq!(
            staged.plan.prelude_values,
            vec![Rational::one(), Rational::one()]
        );
        let mut staged = staged.map_coeff(&|value| value.re.to_f64());
        let mut output = [0.];
        staged
            .evaluate(&[3.], &mut output, Rational::to_f64)
            .unwrap();
        assert_eq!(output, [0.]);
        assert!(baseline.evaluate_single(&[1e200, 1., 1.]).is_nan());
        staged
            .evaluate(&[1e200], &mut output, Rational::to_f64)
            .unwrap();
        assert!(output[0].is_nan());
    }

    #[test]
    fn vectorized_rows_propagate_exact_zero_seeds() {
        let params = [
            parse!("staged_dual::x"),
            parse!("staged_dual::m"),
            parse!("staged_dual::c"),
        ];
        let root = parse!("(staged_dual::x+staged_dual::c)^2")
            * Symbol::SQRT.call_args([params[1].clone()]);
        let staged = root
            .evaluator(&params)
            .build_parameter_rows(&params[2..], &[vec![1.into()], vec![2.into()]])
            .unwrap();
        let vectorizer = Dualizer::new(
            crate::domains::dual::HyperDual::<Complex<Rational>>::new(vec![
                vec![0],
                vec![1],
                vec![2],
            ]),
            vec![(1, 1), (1, 2)],
        );
        let mut staged = staged
            .vectorize(&vectorizer)
            .unwrap()
            .map_coeff(&|value| value.re.to_f64());
        let mut output = [0.; 3];
        staged
            .evaluate(&[3., 1., 0., 4., 0., 0.], &mut output, Rational::to_f64)
            .unwrap();
        assert_eq!(output, [82., 36., 4.]);
        staged
            .evaluate(&[3., 1., 0., 0., 0., 0.], &mut output, Rational::to_f64)
            .unwrap();
        assert_eq!(output, [0.; 3]);
    }

    #[cfg(all(feature = "serde", feature = "bincode"))]
    #[test]
    fn restored_coefficient_writes_must_match_live_inputs_and_rows() {
        let params = [
            parse!("staged_write_reload::x"),
            parse!("staged_write_reload::c"),
        ];
        let staged = (params[0].clone() + &params[1])
            .evaluator(&params)
            .build_parameter_rows(&params[1..], &[vec![1.into()], vec![2.into()]])
            .unwrap();
        let vectorizer = Dualizer::new(
            crate::domains::dual::HyperDual::<Complex<Rational>>::new(vec![
                vec![0],
                vec![1],
                vec![2],
            ]),
            vec![],
        );
        for staged in [staged.clone(), staged.vectorize(&vectorizer).unwrap()] {
            let mut staged = staged.map_coeff(&|value| value.re.to_f64());
            let components = staged.plan.components;
            let mut physical = vec![0.; components];
            physical[0] = 3.;
            let mut output = vec![0.; components];
            staged
                .evaluate(&physical, &mut output, Rational::to_f64)
                .unwrap();
            assert_eq!(output[0], 9.);
            let json = serde_json::to_vec(&staged).unwrap();
            let _: ParameterRowEvaluator<f64> = serde_json::from_slice(&json).unwrap();

            // The same dimensions and individually valid indices must not
            // allow a schedule to skip or delay a required coefficient write.
            for writes in [vec![vec![], vec![]], vec![vec![0], vec![]]] {
                let mut invalid = staged.clone();
                invalid.plan.bound_writes = writes;
                assert!(invalid.plan.validate().is_ok());
                assert!(
                    invalid
                        .plan
                        .validate_programs(&invalid.prelude, &invalid.body)
                        .is_err()
                );
                let json = serde_json::to_vec(&invalid).unwrap();
                assert!(serde_json::from_slice::<ParameterRowEvaluator<f64>>(&json).is_err());
                let bytes = bincode::encode_to_vec(&invalid, bincode::config::standard()).unwrap();
                assert!(
                    bincode::decode_from_slice::<ParameterRowEvaluator<f64>, _>(
                        &bytes,
                        bincode::config::standard()
                    )
                    .is_err()
                );
            }
        }
    }

    #[cfg(all(feature = "serde", feature = "bincode"))]
    #[test]
    fn row_program_roundtrips_validate_scheduling_metadata() {
        let params = [parse!("staged_reload::x"), parse!("staged_reload::c")];
        let root = parse!("(staged_reload::x+1)^2/(staged_reload::c+2)");
        let mut original = root
            .evaluator(&params)
            .build_parameter_rows(
                &params[1..],
                &[vec![1.into()], vec![2.into()], vec![1.into()]],
            )
            .unwrap()
            .map_coeff(&|value| value.re.to_f64());
        let mut expected = [0.];
        original
            .evaluate(&[2.], &mut expected, Rational::to_f64)
            .unwrap();
        let json = serde_json::to_vec(&original).unwrap();
        let mut decoded: ParameterRowEvaluator<f64> = serde_json::from_slice(&json).unwrap();
        let mut output = [0.];
        decoded
            .evaluate(&[2.], &mut output, Rational::to_f64)
            .unwrap();
        assert_eq!(output, expected);
        let bytes = bincode::encode_to_vec(&original, bincode::config::standard()).unwrap();
        let (mut decoded, read): (ParameterRowEvaluator<f64>, _) =
            bincode::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        assert_eq!(read, bytes.len());
        decoded
            .evaluate(&[3.], &mut output, Rational::to_f64)
            .unwrap();
        original
            .evaluate(&[3.], &mut expected, Rational::to_f64)
            .unwrap();
        assert_eq!(output, expected);
        let mut malformed = original.plan.clone();
        malformed.components = 0;
        assert!(
            serde_json::from_slice::<ParameterRowPlan>(&serde_json::to_vec(&malformed).unwrap())
                .is_err()
        );
        malformed.components = 1;
        malformed.gathers[0][0].1 = Some(malformed.shared_outputs);
        let bytes = bincode::encode_to_vec(&malformed, bincode::config::standard()).unwrap();
        assert!(
            bincode::decode_from_slice::<ParameterRowPlan, _>(&bytes, bincode::config::standard())
                .is_err()
        );
    }

    #[cfg(feature = "native_code_generation")]
    #[test]
    fn ordinary_stage_programs_support_direct_and_indirect_jit() {
        let params = [parse!("staged_jit::x"), parse!("staged_jit::c")];
        let root = parse!("(staged_jit::x+staged_jit::c)^2+sqrt(staged_jit::x)");
        for direct in [false, true] {
            let staged = root
                .evaluator(&params)
                .build_parameter_rows(
                    &params[1..],
                    &[vec![1.into()], vec![2.into()], vec![1.into()]],
                )
                .unwrap()
                .map_coeff(&|value| value.re.to_f64());
            let (plan, prelude, body) = staged.into_parts();
            let settings = JITCompilationSettings::new()
                .direct_translation(direct)
                .optimization_level(2);
            let mut prelude = prelude.jit_compile(settings.clone()).unwrap();
            let mut body = body.jit_compile(settings).unwrap();
            let mut output = [0.];
            plan.evaluate_with(
                &[4.],
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
            assert_eq!(output, [92.]);
        }
    }
    #[test]
    fn already_vectorized_branches_can_be_scheduled_over_exact_rows() {
        let params = [
            parse!("staged_existing_dual::x"),
            parse!("staged_existing_dual::c"),
        ];
        let root = parse!(
            "if(staged_existing_dual::c,staged_existing_dual::x^2,if(staged_existing_dual::x,1/staged_existing_dual::x,0))"
        );
        let program = root
            .evaluator(&params)
            .build()
            .unwrap()
            .vectorize(&Dualizer::new(
                crate::domains::dual::HyperDual::<Complex<Rational>>::new(vec![
                    vec![0],
                    vec![1],
                    vec![2],
                ]),
                vec![(1, 1), (1, 2)],
            ))
            .unwrap();
        let staged = ParameterRowEvaluator::new(
            program.clone(),
            vec![3, 4, 5],
            vec![
                vec![0.into(), 0.into(), 0.into()],
                vec![1.into(), 0.into(), 0.into()],
            ],
        )
        .unwrap();
        let mut numeric = staged.clone().map_coeff(&|value| value.re.to_f64());
        let mut oracle = program.map_coeff(&|value| value.re.to_f64());
        for (x, expected) in [
            (2., [4.5, 3.75, 1.125]),
            (-2., [3.5, -4.25, 0.875]),
            (0., [0., 0., 1.]),
        ] {
            let mut original = [0.; 3];
            let mut row = [0.; 3];
            for c in [0., 1.] {
                oracle.evaluate(&[x, 1., 0., c, 0., 0.], &mut row);
                for (sum, value) in original.iter_mut().zip(row) {
                    *sum += value;
                }
            }
            assert_eq!(original, expected);
            let mut actual = [0.; 3];
            numeric
                .evaluate(&[x, 1., 0.], &mut actual, Rational::to_f64)
                .unwrap();
            assert_eq!(actual, expected);
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
                    &[x, 1., 0.],
                    &mut actual,
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
                assert_eq!(actual, expected);
            }
        }
    }

    #[test]
    fn selected_scalar_results_drop_empty_control_scopes_without_inlining() {
        let params = [
            parse!("row_selected::x"),
            parse!("row_selected::c"),
            parse!("row_selected::s"),
            parse!("row_selected::t"),
            parse!("row_selected::gate"),
        ];
        let function = symbol!("row_selected::n");
        let expression = parse!(
            "row_selected::n(row_selected::x+row_selected::c)*if(row_selected::s,if(row_selected::t,1/(row_selected::x+1),1/(row_selected::x+2)),1/(row_selected::x+3))"
        );
        let rows = vec![
            vec![1.into(), 0.into(), 0.into()],
            vec![2.into(), 1.into(), 0.into()],
            vec![3.into(), 1.into(), 1.into()],
        ];
        for policy in [InliningPolicy::Always, InliningPolicy::Never] {
            for guarded in [false, true] {
                let root = if guarded {
                    Symbol::IF.call_args([params[4].clone(), expression.clone(), Atom::Zero])
                } else {
                    expression.clone()
                };
                let source = root
                    .evaluator(&params)
                    .add_function_with_options(
                        function,
                        vec![symbol!("row_selected::a")],
                        parse!("row_selected::a^2+1"),
                        FunctionRegistrationOptions::new().inlining(policy),
                    )
                    .unwrap()
                    .build()
                    .unwrap();
                let staged =
                    ParameterRowEvaluator::new(source.clone(), vec![1, 2, 3], rows.clone())
                        .unwrap();
                let controls = staged
                    .body
                    .instructions
                    .iter()
                    .filter(|(instruction, _)| {
                        matches!(
                            instruction,
                            Instr::IfElse(..)
                                | Instr::Goto(..)
                                | Instr::Label(..)
                                | Instr::Join(..)
                        )
                    })
                    .count();
                eprintln!(
                    "selected scalar carrier inventory: policy={policy:?}, guarded={guarded}, body_instructions={}, control_instructions={controls}, body_operations={:?}, prelude_instructions={}",
                    staged.body.instructions.len(),
                    staged.body.count_operations(),
                    staged.prelude.instructions.len()
                );
                if guarded {
                    assert!(controls > 0, "the physical guard must remain lazy");
                } else {
                    assert_eq!(
                        controls, 0,
                        "selected scalar carriers need no repeated native branch calls"
                    );
                }
                let bodies = staged
                    .body
                    .external_fns
                    .iter()
                    .filter(|function| function.body.is_some())
                    .count();
                assert_eq!(bodies, usize::from(policy == InliningPolicy::Never));
                let mut numeric = staged.clone().map_coeff(&|value| value.re.to_f64());
                let mut actual = [0.];
                numeric
                    .evaluate(&[2., 1.], &mut actual, Rational::to_f64)
                    .unwrap();
                let expected = 10. / 5. + 17. / 4. + 26. / 3.;
                assert!((actual[0] - expected).abs() < 1e-12);
                if guarded {
                    numeric
                        .evaluate(&[-1., 0.], &mut actual, Rational::to_f64)
                        .unwrap();
                    assert_eq!(actual, [0.]);
                }
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
                    for (x, gate, expected) in [(2., 1., expected), (-1., 0., 0.)] {
                        if !guarded && gate == 0. {
                            continue;
                        }
                        plan.evaluate_with(
                            &[x, gate],
                            &mut actual,
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
                        assert!((actual[0] - expected).abs() < 1e-12);
                    }
                }
                let mut inactive =
                    ParameterRowEvaluator::new(source, vec![1, 2, 3], rows[..2].to_vec())
                        .unwrap()
                        .map_coeff(&|value| value.re.to_f64());
                inactive
                    .evaluate(&[-1., 1.], &mut actual, Rational::to_f64)
                    .unwrap();
                assert_eq!(
                    actual,
                    [2.5],
                    "the never-selected x+1 pole must not be evaluated"
                );
            }
        }
    }
}
