//! Uncertainty estimates for the numerical special-function implementations.
//!
//! Symmetric, extra-precision probes retain nonlinear changes at stationary
//! points. Probes use at least a resolvable local step, scaling their response
//! by the actual error with wide-exponent magnitudes. Mixed-coordinate probes
//! account for simultaneous argument and order perturbations. These inflated
//! numerical estimates are not certified interval enclosures: sampling cannot
//! bound every interior extremum or the numerical backend's approximation error.

use super::*;
use crate::domains::float::Magnitude;

type Tracked = Complex<ErrorPropagatingFloat<Float>>;

#[derive(Clone, Copy)]
pub(super) enum SpecialFunction {
    Gamma,
    Erf,
    Zeta,
    Polygamma(u32),
    Polylog,
    BesselJ,
    BesselY,
    BesselI,
    BesselK,
}

impl SpecialFunction {
    fn value(
        self,
        order: Option<&Complex<Float>>,
        argument: &Complex<Float>,
        precision: u32,
    ) -> Option<Complex<Float>> {
        match self {
            Self::Gamma => Some(gamma_numeric_eval(argument, precision)),
            Self::Erf => Some(erf_numeric_eval(argument, precision)),
            Self::Zeta => Some(zeta_numeric_eval(argument, precision)),
            Self::Polygamma(n) => Some(polygamma_numeric_eval(n, argument, precision)),
            Self::Polylog => polylog_numeric_eval(order?, argument, precision),
            Self::BesselJ => bessel_j_numeric_eval(order?, argument, precision),
            Self::BesselY => bessel_y_numeric_eval(order?, argument, precision),
            Self::BesselI => bessel_i_numeric_eval(order?, argument, precision),
            Self::BesselK => bessel_k_numeric_eval(order?, argument, precision),
        }
    }
}

fn center(value: &Tracked, precision: u32) -> Complex<Float> {
    let mut re = value.re.get_num().clone();
    let mut im = value.im.get_num().clone();
    re.set_prec(precision);
    im.set_prec(precision);
    Complex::new(re, im)
}

fn roundoff(value: &Float) -> Magnitude {
    if value.is_zero() {
        return Magnitude::ZERO;
    }
    value
        .absolute_magnitude()
        .scale_pow2(-(value.prec() as i64))
        .max(value.minimum_roundoff())
}

fn finish(value: Complex<Float>, errors: [Magnitude; 2]) -> Tracked {
    let include_roundoff = |error: Magnitude, value: &Float| {
        if error.is_infinite() {
            error
        } else {
            error + roundoff(value)
        }
    };
    let real_error = include_roundoff(errors[0], &value.re);
    let imaginary_error = include_roundoff(errors[1], &value.im);
    Complex::new(
        ErrorPropagatingFloat::new_with_error_magnitude(value.re, real_error),
        ErrorPropagatingFloat::new_with_error_magnitude(value.im, imaginary_error),
    )
}

fn magnitude_float(value: Magnitude, precision: u32) -> Float {
    let significand = Float::with_val(precision, value.significand());
    let exponent = i64::from(value.exponent()) - 31;
    if exponent < 0 {
        significand * Float::with_val(precision, 0.5).pow(exponent.unsigned_abs())
    } else {
        significand * Float::with_val(precision, 2).pow(exponent.unsigned_abs())
    }
}

fn reaches_zero(value: &Float, error: Magnitude) -> bool {
    value.absolute_magnitude_lower() <= error
}

fn shifted(value: &Float, shift: i64) -> Float {
    value.clone() - Float::with_val(value.prec(), shift)
}

fn nearest_gamma_pole(value: &Float) -> Float {
    if *value > Float::new(value.prec()) {
        Float::new(value.prec())
    } else {
        // Backends differ in integer conversion rounding. Check adjacent
        // integers too, so truncation toward zero cannot miss a negative pole.
        let mut nearest =
            Rational::from(value.round_to_nearest_integer()).to_multi_prec_float(value.prec());
        let base = nearest.clone();
        let mut distance = (value.clone() - &nearest).norm();
        for offset in [-1, 1] {
            let candidate = base.clone() + Float::with_val(value.prec(), offset);
            if candidate > Float::new(value.prec()) {
                continue;
            }
            let candidate_distance = (value.clone() - &candidate).norm();
            if candidate_distance < distance {
                nearest = candidate;
                distance = candidate_distance;
            }
        }
        nearest
    }
}

fn on_cut(argument: &Tracked, endpoint: i64, positive: bool) -> bool {
    let real_error = argument.re.get_error_magnitude();
    let imaginary_error = argument.im.get_error_magnitude();
    if !reaches_zero(argument.im.get_num(), imaginary_error) {
        return false;
    }
    let offset = shifted(argument.re.get_num(), endpoint);
    let zero = Float::new(offset.prec());
    let outside_ray = if positive {
        offset < zero
    } else {
        offset > zero
    };
    if outside_ray && !reaches_zero(&offset, real_error) {
        return false;
    }
    // Uncertainty transverse to a cut can cross its two lips. Purely real
    // perturbations on one lip are safe unless they reach its branch point.
    !imaginary_error.is_zero() || (!real_error.is_zero() && reaches_zero(&offset, real_error))
}

fn singular_neighborhood(
    function: SpecialFunction,
    order: Option<&Tracked>,
    argument: &Tracked,
) -> bool {
    let real_error = argument.re.get_error_magnitude();
    let imaginary_error = argument.im.get_error_magnitude();
    match function {
        SpecialFunction::Gamma | SpecialFunction::Polygamma(_) => {
            reaches_zero(argument.im.get_num(), imaginary_error)
                && reaches_zero(
                    &(argument.re.get_num().clone() - nearest_gamma_pole(argument.re.get_num())),
                    real_error,
                )
        }
        SpecialFunction::Zeta => {
            reaches_zero(argument.im.get_num(), imaginary_error)
                && reaches_zero(&shifted(argument.re.get_num(), 1), real_error)
        }
        SpecialFunction::Polylog => {
            on_cut(argument, 1, true)
                || (argument.im.is_exact_zero()
                    && shifted(argument.re.get_num(), 1).is_zero()
                    && order.is_some_and(|order| {
                        reaches_zero(order.im.get_num(), order.im.get_error_magnitude())
                            && reaches_zero(
                                &shifted(order.re.get_num(), 1),
                                order.re.get_error_magnitude(),
                            )
                    }))
        }
        SpecialFunction::BesselY | SpecialFunction::BesselK => on_cut(argument, 0, false),
        SpecialFunction::BesselJ | SpecialFunction::BesselI => {
            let integer_order = order.is_some_and(|n| {
                n.re.is_exact()
                    && n.im.is_exact_zero()
                    && n.re
                        .get_num()
                        .try_to_rational()
                        .is_some_and(|r| r.is_integer())
            });
            !integer_order && on_cut(argument, 0, false)
        }
        SpecialFunction::Erf => false,
    }
}

fn real_neighborhood(
    function: SpecialFunction,
    order: Option<&Tracked>,
    argument: &Tracked,
) -> bool {
    if !argument.im.is_exact_zero() || order.is_some_and(|order| !order.im.is_exact_zero()) {
        return false;
    }
    match function {
        SpecialFunction::Gamma
        | SpecialFunction::Erf
        | SpecialFunction::Zeta
        | SpecialFunction::Polygamma(_) => true,
        SpecialFunction::Polylog => {
            let distance = Float::with_val(argument.re.get_num().prec(), 1) - argument.re.get_num();
            (argument.re.is_exact() && distance.is_zero())
                || (distance > Float::new(distance.prec())
                    && !reaches_zero(&distance, argument.re.get_error_magnitude()))
        }
        SpecialFunction::BesselJ | SpecialFunction::BesselI => {
            order.is_some_and(|order| {
                order.re.is_exact()
                    && order
                        .re
                        .get_num()
                        .try_to_rational()
                        .is_some_and(|r| r.is_integer())
            }) || positive_real_neighborhood(argument)
        }
        SpecialFunction::BesselY | SpecialFunction::BesselK => positive_real_neighborhood(argument),
    }
}

fn positive_real_neighborhood(argument: &Tracked) -> bool {
    *argument.re.get_num() > Float::new(argument.re.get_num().prec())
        && !reaches_zero(argument.re.get_num(), argument.re.get_error_magnitude())
}

fn broad_oscillatory_neighborhood(
    function: SpecialFunction,
    order: Option<&Tracked>,
    argument: &Tracked,
) -> bool {
    let radius = argument.re.get_error_magnitude() + argument.im.get_error_magnitude();
    let center =
        argument.re.get_num().absolute_magnitude() + argument.im.get_num().absolute_magnitude();
    let limit = Magnitude::ONE.scale_pow2(-1);
    let order_size = order.map_or(Magnitude::ZERO, |order| {
        order.re.get_num().absolute_magnitude() + order.im.get_num().absolute_magnitude()
    });
    let order_radius = order.map_or(Magnitude::ZERO, |order| {
        order.re.get_error_magnitude() + order.im.get_error_magnitude()
    });
    let mut rate = Magnitude::ONE + center + order_size;
    if matches!(function, SpecialFunction::Erf) {
        if argument.im.is_exact_zero() {
            return false;
        }
        // exp(-z^2), the derivative of erf, can oscillate arbitrarily rapidly
        // far from the origin. Limit its possible phase change, rather than
        // only the absolute size of the uncertainty rectangle.
        return radius * (center.scale_pow2(1) + radius) > limit;
    }
    if matches!(
        function,
        SpecialFunction::Gamma | SpecialFunction::Polygamma(_)
    ) {
        if argument.im.is_exact_zero() {
            return false;
        }
        let distance = (argument.re.get_num().clone() - nearest_gamma_pole(argument.re.get_num()))
            .absolute_magnitude_lower()
            .max(argument.im.get_num().absolute_magnitude_lower());
        let degree = match function {
            SpecialFunction::Polygamma(n) => Magnitude::from_f64(f64::from(n) + 1.),
            _ => Magnitude::ONE,
        };
        // Stay local both near the pole lattice and where gamma's complex
        // phase varies rapidly. The magnitude term deliberately overestimates
        // its logarithmic growth; no uncertified phase reduction is needed.
        rate += degree / distance;
    }
    if matches!(
        function,
        SpecialFunction::BesselJ
            | SpecialFunction::BesselY
            | SpecialFunction::BesselI
            | SpecialFunction::BesselK
    ) {
        // The z^order factor changes phase on the scale |order|/|z|,
        // even when the absolute argument perturbation is very small.
        if !order_size.is_zero() {
            let distance = argument
                .re
                .get_num()
                .absolute_magnitude_lower()
                .max(argument.im.get_num().absolute_magnitude_lower());
            rate += order_size / distance;
        }
        // Samples separated by a period can all lie close to zeros while
        // missing large interior extrema. Wide Bessel neighborhoods require
        // a dedicated analytic or interval bound instead.
        let radius_limit = Magnitude::ONE.scale_pow2(-2);
        if radius > radius_limit || order_radius > radius_limit {
            return true;
        }
    }
    (radius + order_radius) * rate > limit
}

fn smaller_nonzero(a: Magnitude, b: Magnitude) -> Magnitude {
    if !b.is_zero() && b < a { b } else { a }
}

fn probe_rate(function: SpecialFunction, order: Option<&Tracked>, argument: &Tracked) -> Magnitude {
    let argument_size =
        argument.re.get_num().absolute_magnitude() + argument.im.get_num().absolute_magnitude();
    let order_size = order.map_or(Magnitude::ZERO, |order| {
        order.re.get_num().absolute_magnitude() + order.im.get_num().absolute_magnitude()
    });
    let mut rate = Magnitude::ONE + argument_size.scale_pow2(1) + order_size;
    match function {
        SpecialFunction::Gamma | SpecialFunction::Polygamma(_) => {
            let distance = (argument.re.get_num().clone()
                - nearest_gamma_pole(argument.re.get_num()))
            .absolute_magnitude_lower()
            .max(argument.im.get_num().absolute_magnitude_lower());
            let degree = match function {
                SpecialFunction::Polygamma(n) => Magnitude::from_f64(f64::from(n) + 1.),
                _ => Magnitude::ONE,
            };
            rate += degree / distance;
        }
        SpecialFunction::BesselJ
        | SpecialFunction::BesselY
        | SpecialFunction::BesselI
        | SpecialFunction::BesselK
            if !order_size.is_zero() =>
        {
            let distance = argument
                .re
                .get_num()
                .absolute_magnitude_lower()
                .max(argument.im.get_num().absolute_magnitude_lower());
            rate += order_size / distance;
        }
        _ => {}
    }
    rate
}

fn probe_scale(function: SpecialFunction, order_axis: bool, value: &Complex<Float>) -> Magnitude {
    let magnitude = value
        .re
        .absolute_magnitude()
        .max(value.im.absolute_magnitude());
    let mut scale = if magnitude.is_zero() {
        Magnitude::ONE
    } else {
        magnitude
    };
    if !order_axis {
        let displacement = match function {
            SpecialFunction::Gamma | SpecialFunction::Polygamma(_) => {
                Some(value.re.clone() - nearest_gamma_pole(&value.re))
            }
            SpecialFunction::Zeta | SpecialFunction::Polylog => Some(shifted(&value.re, 1)),
            _ => None,
        };
        if let Some(displacement) = displacement {
            let distance = displacement.absolute_magnitude() + value.im.absolute_magnitude();
            scale = smaller_nonzero(scale, distance);
        }
    }
    scale
}

struct Axis {
    order: bool,
    imaginary: bool,
    step: Float,
    ratio: Magnitude,
}

impl Axis {
    fn perturb(
        &self,
        order: &mut Option<Complex<Float>>,
        argument: &mut Complex<Float>,
        sign: bool,
    ) {
        let value = if self.order {
            order.as_mut().unwrap()
        } else {
            argument
        };
        let component = if self.imaginary {
            &mut value.im
        } else {
            &mut value.re
        };
        if sign {
            *component += &self.step;
        } else {
            *component -= &self.step;
        }
    }
}

fn variation(value: &Complex<Float>, origin: &Complex<Float>, ratio: Magnitude) -> [Magnitude; 2] {
    let component = |value: &Float, origin: &Float| {
        // Include cancellation and evaluation roundoff before scaling. The
        // ratio never passes through f64, even for errors below its range.
        let difference = value.clone() - origin;
        (difference.absolute_magnitude() + roundoff(value) + roundoff(origin)) * ratio
    };
    [
        component(&value.re, &origin.re),
        component(&value.im, &origin.im),
    ]
}

pub(super) fn evaluate(
    function: SpecialFunction,
    order: Option<&Tracked>,
    argument: &Tracked,
) -> Option<Tracked> {
    let precision = argument
        .re
        .get_num()
        .prec()
        .max(argument.im.get_num().prec())
        .max(order.map_or(0, |order| {
            order.re.get_num().prec().max(order.im.get_num().prec())
        }));
    let argument_center = center(argument, precision);
    let order_center = order.map(|value| center(value, precision));
    let value = function.value(order_center.as_ref(), &argument_center, precision)?;
    let real_output = real_neighborhood(function, order, argument);
    let finish_result = |mut value: Complex<Float>, mut errors: [Magnitude; 2]| {
        if real_output {
            // Integer Bessel powers and complex series may leave an imaginary
            // rounding residual on a provably real domain. Enforce that exact
            // identity while retaining the residual in the numerical estimate.
            if !errors[0].is_infinite() {
                errors[0] += value.im.absolute_magnitude();
            }
            value.im = Float::new(value.im.prec());
            errors[1] = Magnitude::ZERO;
        }
        finish(value, errors)
    };
    let unresolved = |value, error| finish_result(value, [error; 2]);
    let inputs = [Some(argument), order];
    let errors = inputs
        .iter()
        .flatten()
        .flat_map(|input| {
            [
                input.re.get_error_magnitude(),
                input.im.get_error_magnitude(),
            ]
        })
        .collect::<Vec<_>>();
    let exact_input = errors.iter().all(|error| error.is_zero());
    if errors.iter().any(|error| error.is_nan()) {
        return Some(unresolved(value, Magnitude::NAN));
    }
    if !finite(&value)
        || errors.iter().any(|error| error.is_infinite())
        || (!exact_input && singular_neighborhood(function, order, argument))
        || broad_oscillatory_neighborhood(function, order, argument)
    {
        return Some(unresolved(value, Magnitude::INFINITY));
    }
    let work = precision.checked_add(64)?;
    let work_argument = center(argument, work);
    let work_order = order.map(|value| center(value, work));
    let Some(origin) = function
        .value(work_order.as_ref(), &work_argument, work)
        .filter(finite)
    else {
        return Some(unresolved(value, Magnitude::INFINITY));
    };

    let mut axes = Vec::new();
    // The resolvable probe floor must also stay inside the local phase regime;
    // a tiny input error does not make a broad numerical probe trustworthy.
    let phase_scale = Magnitude::ONE / probe_rate(function, order, argument);
    for (order_axis, input) in [(false, Some(argument)), (true, order)] {
        let Some(input) = input else {
            continue;
        };
        let input_center = if order_axis {
            work_order.as_ref().unwrap()
        } else {
            &work_argument
        };
        let floor = smaller_nonzero(probe_scale(function, order_axis, input_center), phase_scale)
            .scale_pow2(-i64::from((precision / 4).clamp(8, 32)));
        for (imaginary, component) in [(false, &input.re), (true, &input.im)] {
            let error = component.get_error_magnitude();
            if error.is_zero() {
                continue;
            }
            let step_size = error.max(floor);
            let step = magnitude_float(step_size, work);
            if !step.is_finite() || step.is_zero() {
                return Some(unresolved(value, Magnitude::INFINITY));
            }
            let central_component = if imaginary {
                &input_center.im
            } else {
                &input_center.re
            };
            if central_component.clone() + &step == *central_component
                || central_component.clone() - &step == *central_component
            {
                return Some(unresolved(value, Magnitude::INFINITY));
            }
            axes.push(Axis {
                order: order_axis,
                imaginary,
                step,
                ratio: error / step_size,
            });
        }
    }

    let mut estimates = [Magnitude::ZERO; 2];
    for axis in &axes {
        let mut largest = [Magnitude::ZERO; 2];
        for sign in [false, true] {
            let mut perturbed_order = work_order.clone();
            let mut perturbed_argument = work_argument.clone();
            axis.perturb(&mut perturbed_order, &mut perturbed_argument, sign);
            let Some(sample) = function
                .value(perturbed_order.as_ref(), &perturbed_argument, work)
                .filter(finite)
            else {
                return Some(unresolved(value, Magnitude::INFINITY));
            };
            let response = variation(&sample, &origin, axis.ratio);
            for component in 0..2 {
                largest[component] = largest[component].max(response[component]);
            }
        }
        for component in 0..2 {
            estimates[component] += largest[component];
        }
    }

    if axes.len() > 1 {
        let ratio = axes
            .iter()
            .fold(Magnitude::ZERO, |ratio, axis| ratio.max(axis.ratio));
        for corner in 0..(1usize << axes.len()) {
            let mut perturbed_order = work_order.clone();
            let mut perturbed_argument = work_argument.clone();
            for (index, axis) in axes.iter().enumerate() {
                axis.perturb(
                    &mut perturbed_order,
                    &mut perturbed_argument,
                    corner & (1 << index) != 0,
                );
            }
            let Some(sample) = function
                .value(perturbed_order.as_ref(), &perturbed_argument, work)
                .filter(finite)
            else {
                return Some(unresolved(value, Magnitude::INFINITY));
            };
            let response = variation(&sample, &origin, ratio);
            for component in 0..2 {
                estimates[component] = estimates[component].max(response[component]);
            }
        }
    }
    let correction = variation(&origin, &value, Magnitude::ONE);
    for component in 0..2 {
        estimates[component] = estimates[component].scale_pow2(1) + correction[component];
    }
    Some(finish_result(value, estimates))
}
