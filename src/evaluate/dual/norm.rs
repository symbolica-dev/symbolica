//! Stable real-parameter derivatives of a real or complex modulus.
//!
//! This is the instruction form of Numerica's phase-frame norm recurrence.
//! Direct division retains subnormal ratios; projections avoid both overflowing
//! parallel sums and underflowing halves. Value selections do not change control
//! flow or evaluate external functions from an inactive physical branch.

use super::*;

impl<T: DualNumberStructure> Dualizer<T> {
    pub(super) fn norm_components(
        &self,
        input: Slot,
        instructions: &mut InstructionList<Complex<Rational>>,
    ) -> Vec<VectorInstruction> {
        let zero = instructions.add_repeated_constant(Complex::new_zero());
        let norm = instructions.add(VectorInstruction::BuiltinFun(Symbol::ABS, input));
        let dimension = self.get_dimension();
        if (1..dimension).all(|index| self.is_zero(&input.index(index), instructions)) {
            return std::iter::once(VectorInstruction::Assign(norm))
                .chain((1..dimension).map(|_| VectorInstruction::Assign(zero)))
                .collect();
        }
        let one = instructions.add_constant_in_first_component(Complex::from(Rational::one()));
        let two = instructions.add_constant_in_first_component(Complex::from(Rational::from(2)));
        let minus_one =
            instructions.add_constant_in_first_component(Complex::from(Rational::from(-1)));
        let safe_norm = instructions.add(VectorInstruction::Select(norm, norm, one));
        let phase = instructions.add(VectorInstruction::Div(input, safe_norm));
        let phase_norm = instructions.add(VectorInstruction::BuiltinFun(Symbol::ABS, phase));
        let safe_phase_norm =
            instructions.add(VectorInstruction::Select(phase_norm, phase_norm, one));
        let unit_phase = instructions.add(VectorInstruction::Div(phase, safe_phase_norm));
        let unit_phase = instructions.add(VectorInstruction::BuiltinFun(Symbol::CONJ, unit_phase));
        let root_norm = instructions.add(VectorInstruction::BuiltinFun(Symbol::SQRT, safe_norm));
        let twice_phase_norm = scalar_mul(&safe_phase_norm, &two, self, instructions);
        let root_phase = instructions.add(VectorInstruction::BuiltinFun(
            Symbol::SQRT,
            twice_phase_norm,
        ));
        let scale = scalar_mul(&root_norm, &root_phase, self, instructions);

        let mut parallel = vec![zero; dimension];
        let mut transverse = vec![zero; dimension];
        let mut activity = norm;
        for index in 1..dimension {
            let value = input.index(index);
            if self.is_zero(&value, instructions) {
                continue;
            }
            let magnitude = instructions.add(VectorInstruction::BuiltinFun(Symbol::ABS, value));
            activity = scalar_add(&activity, &magnitude, self, instructions);
            let rotated = scalar_mul(&value, &unit_phase, self, instructions);
            let conjugate = instructions.add(VectorInstruction::BuiltinFun(Symbol::CONJ, rotated));
            let half = instructions.add(VectorInstruction::Div(rotated, two));
            let conjugate_half =
                instructions.add(VectorInstruction::BuiltinFun(Symbol::CONJ, half));
            for (negative, destination) in [
                (false, &mut parallel[index]),
                (true, &mut transverse[index]),
            ] {
                let (conjugate, conjugate_half) = if negative {
                    (
                        scalar_mul(&conjugate, &minus_one, self, instructions),
                        scalar_mul(&conjugate_half, &minus_one, self, instructions),
                    )
                } else {
                    (conjugate, conjugate_half)
                };
                let sum = scalar_add(&rotated, &conjugate, self, instructions);
                let projection = instructions.add(VectorInstruction::Div(sum, two));
                let scaled_projection = scalar_add(&half, &conjugate_half, self, instructions);
                let negative_sum = scalar_mul(&sum, &minus_one, self, instructions);
                let nonfinite = scalar_add(&sum, &negative_sum, self, instructions);
                *destination = instructions.add(VectorInstruction::Select(
                    nonfinite,
                    scaled_projection,
                    projection,
                ));
            }
        }

        let mut correction = vec![zero; dimension];
        let shape = self.dual.get_shape();
        let mut order = (1..dimension).collect::<Vec<_>>();
        order.sort_by_key(|index| shape[*index].iter().sum::<usize>());
        for index in order {
            let mut coefficient = zero;
            for &(left, right, target) in self.dual.get_multiplication_table() {
                if target != index || left == 0 || right == 0 {
                    continue;
                }
                let conjugate = instructions.add(VectorInstruction::BuiltinFun(
                    Symbol::CONJ,
                    transverse[right],
                ));
                let product = self.norm_scaled_product(
                    transverse[left],
                    conjugate,
                    scale,
                    zero,
                    minus_one,
                    instructions,
                );
                coefficient = scalar_add(&coefficient, &product, self, instructions);
                let product = self.norm_scaled_product(
                    parallel[left],
                    correction[right],
                    scale,
                    zero,
                    minus_one,
                    instructions,
                );
                let product = scalar_mul(&product, &two, self, instructions);
                let product = scalar_mul(&product, &minus_one, self, instructions);
                coefficient = scalar_add(&coefficient, &product, self, instructions);
                let product = self.norm_scaled_product(
                    correction[left],
                    correction[right],
                    scale,
                    zero,
                    minus_one,
                    instructions,
                );
                let product = scalar_mul(&product, &minus_one, self, instructions);
                coefficient = scalar_add(&coefficient, &product, self, instructions);
            }
            correction[index] = coefficient;
        }
        // A fully zero jet has zero norm. A zero primal with a nonzero jet is
        // a cusp, and must keep invalid derivatives rather than inventing a sign.
        let cusp = instructions.add(VectorInstruction::Div(norm, norm));
        let at_zero = instructions.add(VectorInstruction::Select(activity, cusp, zero));
        let mut result = vec![VectorInstruction::Assign(norm)];
        for index in 1..dimension {
            let component = scalar_add(&parallel[index], &correction[index], self, instructions);
            result.push(VectorInstruction::Select(norm, component, at_zero));
        }
        result
    }

    /// Multiply derivative factors over the common scale, retaining finite
    /// results when one normalized factor alone would overflow or underflow.
    fn norm_scaled_product(
        &self,
        left: Slot,
        right: Slot,
        scale: Slot,
        zero: Slot,
        minus_one: Slot,
        instructions: &mut InstructionList<Complex<Rational>>,
    ) -> Slot {
        if self.is_zero(&left, instructions) || self.is_zero(&right, instructions) {
            return zero;
        }
        let scaled_left = instructions.add(VectorInstruction::Div(left, scale));
        let scaled_right = instructions.add(VectorInstruction::Div(right, scale));
        let normalized = scalar_mul(&scaled_left, &scaled_right, self, instructions);
        let product = scalar_mul(&left, &right, self, instructions);
        let product = instructions.add(VectorInstruction::Div(product, scale));
        let fallback = instructions.add(VectorInstruction::Div(product, scale));
        let mut result = normalized;
        for (original, scaled) in [(left, scaled_left), (right, scaled_right)] {
            let underflow = instructions.add(VectorInstruction::Select(original, fallback, result));
            result = instructions.add(VectorInstruction::Select(scaled, result, underflow));
            let negative = scalar_mul(&scaled, &minus_one, self, instructions);
            let nonfinite = scalar_add(&scaled, &negative, self, instructions);
            result = instructions.add(VectorInstruction::Select(nonfinite, fallback, result));
        }
        // These are derivative convolution terms. Their exact zero factors
        // contribute nothing even when a different coefficient exceeds range.
        result = instructions.add(VectorInstruction::Select(left, result, zero));
        instructions.add(VectorInstruction::Select(right, result, zero))
    }
}
