use super::*;
mod optimize;

/// An opaque slot belonging to one [`EvaluatorComposer`].
///
/// Handles remain valid when inputs are added or internal storage is rewritten.
/// A handle from another composer is rejected, even if its index happens to match.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ComposerSlot {
    owner: usize,
    index: usize,
}

/// An invalid binding or incompatible function definition in a composition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ComposeError {
    /// The evaluator's input count differs from the number of bindings.
    ArityMismatch { expected: usize, actual: usize },
    /// A binding or selected output belongs to another composer.
    ForeignSlot,
    /// Two evaluators define different functions with the same exported name.
    ConflictingFunction { symbol: Symbol },
}

impl std::fmt::Display for ComposeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ArityMismatch { expected, actual } => {
                write!(f, "Expected {expected} bindings, received {actual}")
            }
            Self::ForeignSlot => write!(f, "Slot belongs to another evaluator composer"),
            Self::ConflictingFunction { symbol } => {
                write!(f, "Conflicting definitions for function '{symbol}'")
            }
        }
    }
}

impl std::error::Error for ComposeError {}

/// Connect existing scalar evaluators without reconstructing symbolic expressions.
///
/// Inputs are ordered by creation. All declared inputs, including unused ones,
/// remain in the finished evaluator's signature. Applications may bind any input,
/// constant or earlier result; shared function bodies remain shared. Finishing
/// automatically interns equal constants before native CSE/CPE. Compose exact
/// coefficients before mapping to a numerical domain, or use [`F64`] when
/// composing floating-point evaluators: finishing requires `Eq + Hash`.
///
/// ```
/// use symbolica::prelude::*;
///
/// let map = parse!("x+y").evaluator(&[parse!("x"), parse!("y")]).build().unwrap();
/// let function = parse!("z^2").evaluator(&[parse!("z")]).build().unwrap();
/// let (mut composer, inputs) = EvaluatorComposer::new_with_inputs(2);
/// let mapped = composer.add(&map, &inputs).unwrap();
/// let outputs = composer.add(&function, &mapped).unwrap();
/// let mut evaluator = composer.finish(&outputs, OptimizationSettings::default())
///     .unwrap().map_coeff(&|c| c.re.to_f64());
/// assert_eq!(evaluator.evaluate_single(&[2., 3.]), 25.);
/// ```
pub struct EvaluatorComposer<T> {
    owner: usize,
    parameters: usize,
    values: Vec<Slot>,
    instructions: InstructionList<T>,
    external_functions: Vec<ExternalFunctionContainer<T>>,
    next_label: usize,
}

impl<T: Default + Clone> Default for EvaluatorComposer<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Default + Clone> EvaluatorComposer<T> {
    /// Start a program with no inputs. Use [`Self::add_input`] to declare inputs.
    pub fn new() -> Self {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT_OWNER: AtomicUsize = AtomicUsize::new(0);
        // fetch_update supports the crate's MSRV; its newer name is not available there.
        #[allow(deprecated)]
        let owner = NEXT_OWNER
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_add(1))
            .expect("Evaluator composer identities exhausted");
        Self {
            owner,
            parameters: 0,
            values: Vec::new(),
            instructions: InstructionList {
                instructions: Vec::new(),
                constants: Vec::new(),
                unknown_constants: Vec::new(),
                dim: 1,
            },
            external_functions: Vec::new(),
            next_label: 0,
        }
    }

    /// Start a program and return its inputs in parameter order.
    pub fn new_with_inputs(count: usize) -> (Self, Vec<ComposerSlot>) {
        let mut composer = Self::new();
        let inputs = (0..count).map(|_| composer.add_input()).collect();
        (composer, inputs)
    }

    fn value(&mut self, slot: Slot) -> ComposerSlot {
        let value = ComposerSlot {
            owner: self.owner,
            index: self.values.len(),
        };
        self.values.push(slot);
        value
    }

    fn slots(&self, values: &[ComposerSlot]) -> Result<Vec<Slot>, ComposeError> {
        values
            .iter()
            .map(|value| {
                if value.owner != self.owner {
                    return Err(ComposeError::ForeignSlot);
                }
                Ok(self.values[value.index])
            })
            .collect()
    }

    /// Declare the next input. Its position is the number of inputs already created.
    pub fn add_input(&mut self) -> ComposerSlot {
        let slot = Slot::Param(self.parameters);
        self.parameters += 1;
        self.value(slot)
    }

    /// Create a literal constant. Equal constants are interned by [`Self::finish`].
    pub fn constant(&mut self, value: T) -> ComposerSlot {
        let slot = Slot::Const(self.instructions.constants.len());
        self.instructions.constants.push(value);
        self.instructions.unknown_constants.push(false);
        self.value(slot)
    }

    /// Add an evaluator bound to existing values, returning its outputs in order.
    ///
    /// The evaluator is borrowed and retains its original instructions and function
    /// bodies. Invalid bindings and conflicting function definitions leave the
    /// composer unchanged. Function bodies and their transitive callees are imported
    /// with remapped indices, preserving shared calls rather than inlining bodies.
    pub fn add(
        &mut self,
        evaluator: &ExpressionEvaluator<T>,
        inputs: &[ComposerSlot],
    ) -> Result<Vec<ComposerSlot>, ComposeError> {
        if evaluator.param_count != inputs.len() {
            return Err(ComposeError::ArityMismatch {
                expected: evaluator.param_count,
                actual: inputs.len(),
            });
        }
        let inputs = self.slots(inputs)?;
        let checkpoint = (
            self.instructions.instructions.len(),
            self.instructions.constants.len(),
            self.external_functions.len(),
            self.next_label,
        );
        let outputs = evaluator.inline_instructions(
            &inputs,
            &mut self.instructions,
            &mut self.external_functions,
            &mut self.next_label,
            InstructionList::push_scalar_constant,
        );
        // Exporters identify callable functions by name. Reject incompatible
        // definitions instead of silently calling the first evaluator's body.
        // Identical definitions were already interned when importing instructions.
        let mut definitions = HashSet::default();
        let conflict = self.external_functions.iter().find_map(|function| {
            (function.constant_index.is_none() && !definitions.insert(&function.export_name))
                .then_some(function.symbol)
        });
        if let Some(symbol) = conflict {
            self.instructions.instructions.truncate(checkpoint.0);
            self.instructions.constants.truncate(checkpoint.1);
            self.instructions.unknown_constants.truncate(checkpoint.1);
            self.external_functions.truncate(checkpoint.2);
            self.next_label = checkpoint.3;
            return Err(ComposeError::ConflictingFunction { symbol });
        }
        Ok(outputs.into_iter().map(|slot| self.value(slot)).collect())
    }

    /// Select/reorder outputs, prune unused work, and optimize the combined program.
    ///
    /// Dead instructions, branch regions, constants and transitive function bodies
    /// are discarded. Live branches retain lazy evaluation. Native CSE runs to a
    /// fixed point, then CPE runs up to `settings.cpe_iterations` (or to a fixed
    /// point for `None`), followed by stack optimization. Symbolic Horner search
    /// is not repeated. Literal and callback constants are automatically interned
    /// before CSE/CPE, enabling sharing across evaluator boundaries.
    pub fn finish(
        mut self,
        outputs: &[ComposerSlot],
        settings: OptimizationSettings,
    ) -> Result<ExpressionEvaluator<T>, ComposeError>
    where
        T: Eq + Hash,
    {
        let mut outputs = self.slots(outputs)?;
        self.prune_unused(&mut outputs);
        self.deduplicate_constants(&mut outputs);
        let mut evaluator = ExpressionEvaluator::from_instruction_list(
            self.parameters,
            self.instructions,
            self.external_functions,
            outputs,
            settings,
        );
        evaluator.flatten_composition_operations();
        evaluator.optimize_instructions();
        evaluator.optimize_stack();
        evaluator.fix_labels();
        Ok(evaluator)
    }
}

#[cfg(test)]
#[path = "compose/tests.rs"]
mod composition_tests;
