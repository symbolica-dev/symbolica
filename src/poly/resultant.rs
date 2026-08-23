use rand::Rng;

use crate::domains::finite_field::{FiniteFieldCore, PrimeIteratorU64, ToFiniteField, Zp};
use crate::domains::integer::{Integer, IntegerRing, Z};
use crate::domains::rational::{Q, Rational, RationalField};
use crate::domains::{EuclideanDomain, Field, Ring};

use super::PositiveExponent;
use super::polynomial::{MultivariatePolynomial, PolynomialRing};
use super::univariate::UnivariatePolynomial;

const RESULTANT_CRT_PRIME_BLOCK: u64 = 1 << 30;
const RESULTANT_CRT_VERIFICATION_PRIMES: usize = 2;

impl<F: Ring> UnivariatePolynomial<F> {
    /// Handle resultants for zero, constant, linear, and two quadratic inputs
    /// with division-free straight-line formulas.
    fn resultant_small(&self, other: &Self) -> Option<F::Element> {
        if self.is_zero() || other.is_zero() {
            return Some(self.ring.zero());
        }

        if self.is_constant() {
            return Some(self.ring.pow(&self.get_constant(), other.degree() as u64));
        }

        if other.is_constant() {
            return Some(self.ring.pow(&other.get_constant(), self.degree() as u64));
        }

        if other.degree() == 1 {
            return Some(self.resultant_with_linear_second(other));
        }

        if self.degree() == 1 {
            let mut r = other.resultant_with_linear_second(self);
            if other.degree() % 2 == 1 {
                r = self.ring.neg(&r);
            }
            return Some(r);
        }

        if self.degree() == 2 && other.degree() == 2 {
            // Res(a0 + a1*x + a2*x^2, b0 + b1*x + b2*x^2)
            //   = p^2 - q*r,
            // p = a2*b0 - a0*b2, q = a2*b1 - a1*b2,
            // r = a1*b0 - a0*b1.
            let p = self.ring.sub(
                &self.ring.mul(&self.coefficients[2], &other.coefficients[0]),
                &self.ring.mul(&self.coefficients[0], &other.coefficients[2]),
            );
            let q = self.ring.sub(
                &self.ring.mul(&self.coefficients[2], &other.coefficients[1]),
                &self.ring.mul(&self.coefficients[1], &other.coefficients[2]),
            );
            let r = self.ring.sub(
                &self.ring.mul(&self.coefficients[1], &other.coefficients[0]),
                &self.ring.mul(&self.coefficients[0], &other.coefficients[1]),
            );

            return Some(
                self.ring
                    .sub(&self.ring.mul(&p, &p), &self.ring.mul(&q, &r)),
            );
        }

        None
    }

    /// Compute `Res(self, b0 + b1*x)` by homogeneous Horner evaluation.
    fn resultant_with_linear_second(&self, linear: &Self) -> F::Element {
        debug_assert_eq!(linear.degree(), 1);

        let neg_b0 = self.ring.neg(&linear.coefficients[0]);
        let b1 = &linear.coefficients[1];
        let mut b1_power = b1.clone();
        let mut result = self.lcoeff();

        for coefficient in self.coefficients[..self.degree()].iter().rev() {
            result = self.ring.add(
                &self.ring.mul(&result, &neg_b0),
                &self.ring.mul(coefficient, &b1_power),
            );
            self.ring.mul_assign(&mut b1_power, b1);
        }

        result
    }
}

impl<F: EuclideanDomain> UnivariatePolynomial<F> {
    /// Compute the resultant using Brown's polynomial remainder sequence algorithm.
    pub fn resultant_prs(&self, other: &Self) -> F::Element {
        if let Some(resultant) = self.resultant_small(other) {
            return resultant;
        }

        self.resultant_prs_with_linear_subresultant(other).0
    }

    /// Compute the resultant using the Lazard-Ducos subresultant algorithm.
    ///
    /// Compared with a classical subresultant PRS, the Ducos recurrence avoids
    /// pseudo-division after the initial remainder. This is particularly useful
    /// when the coefficients are multivariate polynomials.
    pub fn resultant_ducos(&self, other: &Self) -> F::Element {
        if let Some(resultant) = self.resultant_small(other) {
            return resultant;
        }

        if self.degree() < other.degree() {
            let mut resultant = other.resultant_ducos(self);
            if self.degree() % 2 == 1 && other.degree() % 2 == 1 {
                resultant = self.ring.neg(&resultant);
            }
            return resultant;
        }

        let mut a = other.clone();
        let mut b = self.pseudo_remainder_with_negative_divisor(other);
        let mut nominal_leading = self
            .ring
            .pow(&other.lcoeff(), (self.degree() - other.degree()) as u64);

        loop {
            if b.is_zero() {
                return self.ring.zero();
            }

            let degree_difference = a.degree() - b.degree();
            let c = if degree_difference > 1 {
                b.lazard_scale(&nominal_leading, degree_difference - 1)
            } else {
                b.clone()
            };

            if b.is_constant() {
                return c.get_constant();
            }

            let next = if degree_difference == 1 {
                a.ducos_next_subresultant_adjacent(&b, &nominal_leading)
            } else {
                a.ducos_next_subresultant(&b, &c, &nominal_leading)
            };
            a = c;
            nominal_leading = a.lcoeff();
            b = next;
        }
    }

    /// Compute the canonical pseudo-remainder `prem(self, -divisor)`.
    fn pseudo_remainder_with_negative_divisor(&self, divisor: &Self) -> Self {
        debug_assert!(!self.is_zero());
        debug_assert!(!divisor.is_zero());
        debug_assert!(self.degree() >= divisor.degree());

        let divisor_degree = divisor.degree();
        let negative_leading = self.ring.neg(&divisor.lcoeff());
        let mut remaining_scalings = self.degree() - divisor_degree + 1;
        let mut remainder = self.clone();

        while !remainder.is_zero() && remainder.degree() >= divisor_degree {
            let remainder_degree = remainder.degree();
            let shift = remainder_degree - divisor_degree;
            let remainder_leading = remainder.lcoeff();
            let mut next = self.zero();
            next.coefficients.reserve(remainder_degree);

            // One fraction-free pseudo-division step with `-divisor`:
            //
            //   next = lc(remainder) * x^shift * divisor
            //          - lc(divisor) * remainder.
            //
            // The leading terms cancel, so construct only the coefficients
            // below `remainder_degree`.
            for degree in 0..remainder_degree {
                let mut coefficient = if self.ring.is_zero(&remainder.coefficients[degree]) {
                    self.ring.zero()
                } else {
                    self.ring
                        .mul(&negative_leading, &remainder.coefficients[degree])
                };

                if degree >= shift {
                    let divisor_coefficient = &divisor.coefficients[degree - shift];
                    if !self.ring.is_zero(divisor_coefficient) {
                        let correction = self.ring.mul(&remainder_leading, divisor_coefficient);
                        coefficient = self.ring.add(&coefficient, &correction);
                    }
                }

                next.coefficients.push(coefficient);
            }

            next.truncate();
            remainder = next;
            remaining_scalings -= 1;
        }

        // A degree drop can skip pseudo-division steps. Apply the corresponding
        // unused powers of `-lc(divisor)` in one multiplication.
        if remaining_scalings > 0 && !remainder.is_zero() {
            let scale = self.ring.pow(&negative_leading, remaining_scalings as u64);
            remainder = remainder.mul_coeff(&scale);
        }

        remainder
    }

    /// Compute `lc(self)^exponent * self / denominator^exponent` using
    /// Lazard's dichotomic exact-division recurrence.
    fn lazard_scale(&self, denominator: &F::Element, exponent: usize) -> Self {
        debug_assert!(exponent > 0);

        let leading = self.lcoeff();
        let mut factor = leading.clone();
        let mut power_of_two = 1usize << (usize::BITS - 1 - exponent.leading_zeros());
        let mut remaining = exponent - power_of_two;

        while power_of_two > 1 {
            power_of_two >>= 1;
            factor = self.exact_div_element(self.ring.mul(&factor, &factor), denominator);

            if remaining >= power_of_two {
                factor = self.exact_div_element(self.ring.mul(&factor, &leading), denominator);
                remaining -= power_of_two;
            }
        }

        self.clone().mul_coeff(&factor).div_coeff(denominator)
    }

    /// Compute the Ducos recurrence for the common adjacent-degree case
    /// `degree(self) = degree(previous) + 1`.
    ///
    /// Writing `e = degree(previous)`, this evaluates the two fused formulas
    ///
    ///   D = (lc(previous) * self - coeff(self, e) * previous) / lc(self)
    ///   next = (lc(previous) * (D - x * previous)
    ///           + coeff(previous, e - 1) * previous) / nominal_leading
    ///
    /// coefficient by coefficient below degree `e`, where all higher terms
    /// cancel. This avoids constructing the general sequence of `H_j`
    /// polynomials when the degree drop is one.
    fn ducos_next_subresultant_adjacent(
        &self,
        previous: &Self,
        nominal_leading: &F::Element,
    ) -> Self {
        let d = self.degree();
        let e = previous.degree();
        debug_assert_eq!(d, e + 1);
        debug_assert!(e > 0);

        let self_leading = self.lcoeff();
        let self_at_e = &self.coefficients[e];
        let previous_leading = previous.lcoeff();
        let previous_at_e_minus_one = &previous.coefficients[e - 1];
        let mut next = self.zero();
        next.coefficients.reserve(e);

        for degree in 0..e {
            let self_coefficient = &self.coefficients[degree];
            let previous_coefficient = &previous.coefficients[degree];

            let mut d_coefficient = if self.ring.is_zero(self_coefficient) {
                self.ring.zero()
            } else {
                self.ring.mul(&previous_leading, self_coefficient)
            };

            if !self.ring.is_zero(self_at_e) && !self.ring.is_zero(previous_coefficient) {
                let correction = self.ring.mul(self_at_e, previous_coefficient);
                d_coefficient = self.ring.sub(&d_coefficient, &correction);
            }

            if !self.ring.is_zero(&d_coefficient) {
                d_coefficient = self.exact_div_element(d_coefficient, &self_leading);
            }

            if degree > 0 && !self.ring.is_zero(&previous.coefficients[degree - 1]) {
                d_coefficient = self
                    .ring
                    .sub(&d_coefficient, &previous.coefficients[degree - 1]);
            }

            let mut coefficient = if self.ring.is_zero(&d_coefficient) {
                self.ring.zero()
            } else {
                self.ring.mul(&previous_leading, &d_coefficient)
            };

            if !self.ring.is_zero(previous_at_e_minus_one)
                && !self.ring.is_zero(previous_coefficient)
            {
                let correction = self.ring.mul(previous_at_e_minus_one, previous_coefficient);
                coefficient = self.ring.add(&coefficient, &correction);
            }

            if !self.ring.is_zero(&coefficient) {
                coefficient = self.exact_div_element(coefficient, nominal_leading);
            }

            next.coefficients.push(coefficient);
        }

        next.truncate();
        next
    }

    /// Ducos' division-minimizing recurrence for the next nonzero
    /// subresultant. Here `self = S_d`, `previous = S_(d-1)`, and
    /// `regular = S_e`, where `e = degree(previous)`.
    fn ducos_next_subresultant(
        &self,
        previous: &Self,
        regular: &Self,
        nominal_leading: &F::Element,
    ) -> Self {
        let d = self.degree();
        let e = previous.degree();
        debug_assert!(d > e && e > 0);
        debug_assert_eq!(regular.degree(), e);

        let regular_leading = regular.lcoeff();
        let previous_leading = previous.lcoeff();
        let mut sum = self.zero();

        // H_j = lc(S_e) * x^j for 0 <= j < e.
        sum.coefficients.resize(e, self.ring.zero());
        for j in 0..e {
            if !self.ring.is_zero(&self.coefficients[j]) {
                sum.coefficients[j] = self.ring.mul(&self.coefficients[j], &regular_leading);
            }
        }
        sum.truncate();

        // H_e = lc(S_e) * x^e - S_e.
        let mut h = self.monomial(regular_leading, e) - regular.clone();
        if !self.ring.is_zero(&self.coefficients[e]) {
            sum = sum + h.clone().mul_coeff(&self.coefficients[e]);
        }

        // H_j = x*H_(j-1) - coeff(x*H_(j-1), e) * S_(d-1) / lc(S_(d-1)).
        for j in e + 1..d {
            let mut next_h = h.mul_exp(1);
            let correction = next_h
                .coefficients
                .get(e)
                .cloned()
                .unwrap_or_else(|| self.ring.zero());

            if !self.ring.is_zero(&correction) {
                let correction = previous
                    .clone()
                    .mul_coeff(&correction)
                    .div_coeff(&previous_leading);
                next_h = next_h - correction;
            }

            h = next_h;
            if !self.ring.is_zero(&self.coefficients[j]) {
                sum = sum + h.clone().mul_coeff(&self.coefficients[j]);
            }
        }

        // D = sum(coeff(S_d, j) * H_j, j=0..d-1) / lc(S_d).
        let d_poly = sum.div_coeff(&self.lcoeff());
        let shifted_h = h.mul_exp(1);
        let correction = shifted_h
            .coefficients
            .get(e)
            .cloned()
            .unwrap_or_else(|| self.ring.zero());

        let mut next = (shifted_h + d_poly).mul_coeff(&previous_leading);
        if !self.ring.is_zero(&correction) {
            next = next - previous.clone().mul_coeff(&correction);
        }
        next = next.div_coeff(nominal_leading);

        if (d - e + 1) % 2 == 1 {
            next = -next;
        }

        next
    }

    fn exact_div_element(&self, numerator: F::Element, denominator: &F::Element) -> F::Element {
        let (quotient, remainder) = self.ring.quot_rem_owned(numerator, denominator);
        debug_assert!(self.ring.is_zero(&remainder));
        quotient
    }

    /// Compute the resultant and retain the last linear member of Brown's
    /// polynomial remainder sequence, when one occurs.
    ///
    /// The linear subresultant is useful when the resultant constructs a
    /// primitive extension: its two coefficients directly recover the old
    /// generator, avoiding a second polynomial GCD over the new field.
    pub(crate) fn resultant_prs_with_linear_subresultant(
        &self,
        other: &Self,
    ) -> (F::Element, Option<(F::Element, F::Element)>) {
        if self.is_zero() || other.is_zero() {
            return (self.ring.zero(), None);
        }

        if self.degree() < other.degree() {
            let (resultant, linear) = other.resultant_prs_with_linear_subresultant(self);
            if self.degree() % 2 == 1 && other.degree() % 2 == 1 {
                return (self.ring.neg(&resultant), linear);
            }
            return (resultant, linear);
        }

        if other.is_constant() {
            return (
                self.ring.pow(&other.get_constant(), self.degree() as u64),
                None,
            );
        }

        let mut a = self.clone();
        let mut a_new = other.clone();

        let mut deg = a.degree() as u64 - a_new.degree() as u64;
        let mut neg_lc = self.ring.one(); //unused
        let mut init = false;
        let mut beta = self.ring.pow(&self.ring.neg(&self.ring.one()), deg + 1);
        let mut psi = self.ring.neg(&self.ring.one());
        let mut linear_subresultant =
            (a_new.degree() == 1).then(|| (a_new.get_constant(), a_new.lcoeff()));

        let mut lcs = vec![(a.lcoeff(), a.degree() as u64)];
        while !a_new.is_constant() {
            if init {
                psi = if deg == 0 {
                    // can only happen on the first iteration
                    psi
                } else if deg == 1 {
                    neg_lc.clone()
                } else {
                    let a = self.ring.pow(&neg_lc, deg);
                    let psi_old = self.ring.pow(&psi, deg - 1);
                    let (q, r) = self.ring.quot_rem(&a, &psi_old);
                    debug_assert!(self.ring.is_zero(&r));
                    q
                };

                deg = a.degree() as u64 - a_new.degree() as u64;
                beta = self.ring.mul(&neg_lc, &self.ring.pow(&psi, deg));
            } else {
                init = true;
            }

            neg_lc = self.ring.neg(a_new.coefficients.last().unwrap());

            let (_, mut r) = a
                .mul_coeff(&self.ring.pow(&neg_lc, deg + 1))
                .quot_rem(&a_new);
            if (deg + 1) % 2 == 1 {
                r = -r;
            }

            lcs.push((a_new.lcoeff(), a_new.degree() as u64));

            (a, a_new) = (a_new, r.div_coeff(&beta));
            if a_new.degree() == 1 {
                linear_subresultant = Some((a_new.get_constant(), a_new.lcoeff()));
            }
        }

        lcs.push((a_new.lcoeff(), 0));

        if a_new.is_zero() {
            return (self.ring.zero(), linear_subresultant);
        }

        // compute the resultant from the PRS, using the fundamental theorem
        let mut rho = self.ring.one();
        let mut den = self.ring.one();
        for k in 1..lcs.len() {
            let mut deg = lcs[k - 1].1 as i64 - lcs[k].1 as i64;
            for l in k..lcs.len() - 1 {
                deg *= 1 - (lcs[l].1 as i64 - lcs[l + 1].1 as i64);
            }

            if deg > 0 {
                self.ring
                    .mul_assign(&mut rho, &self.ring.pow(&lcs[k].0, deg as u64));
            } else if deg < 0 {
                self.ring
                    .mul_assign(&mut den, &self.ring.pow(&lcs[k].0, (-deg) as u64));
            }
        }

        (self.ring.quot_rem(&rho, &den).0, linear_subresultant)
    }

    /// Compute the resultant using a primitive polynomial remainder sequence.
    pub fn resultant_primitive(&self, other: &Self) -> F::Element {
        if let Some(resultant) = self.resultant_small(other) {
            return resultant;
        }

        if self.degree() < other.degree() {
            if self.degree() % 2 == 1 && other.degree() % 2 == 1 {
                return self.ring.neg(&other.resultant_primitive(self));
            } else {
                return other.resultant_primitive(self);
            }
        }

        let mut a = self.clone();
        let mut a_new = other.clone();

        let mut v = vec![a.degree()];
        let mut c = vec![a.lcoeff()];
        let mut ab = vec![(self.ring.one(), self.ring.one())];

        while !a_new.is_constant() {
            let n = a.degree() as u64 + 1 - a_new.degree() as u64;
            let alpha = self.ring.pow(&a_new.lcoeff(), n);

            let (_, mut r) = a.clone().mul_coeff(&alpha).quot_rem(&a_new);

            let beta = r.content();
            r = r.div_coeff(&beta);

            (a, a_new) = (a_new, r);

            v.push(a.degree());
            c.push(a.lcoeff());
            ab.push((alpha, beta));
        }

        let r = a_new.lcoeff();
        if self.ring.is_zero(&r) {
            return r;
        }

        let mut sign = 0;
        for w in v.windows(2) {
            sign += w[0] * w[1];
        }

        let mut res = self.ring.pow(&r, *v.last().unwrap() as u64);
        if sign % 2 == 1 {
            res = self.ring.neg(&res);
        };

        v.push(0);
        let mut num = self.ring.one();
        let mut den = self.ring.one();
        for i in 1..c.len() {
            self.ring.mul_assign(
                &mut res,
                &self.ring.pow(&c[i], v[i - 1] as u64 - v[i + 1] as u64),
            );

            self.ring
                .mul_assign(&mut num, &self.ring.pow(&ab[i].1, v[i] as u64));
            self.ring
                .mul_assign(&mut den, &self.ring.pow(&ab[i].0, v[i] as u64));
        }

        let (q, r) = self.ring.quot_rem(&self.ring.mul(&num, &res), &den);
        assert!(self.ring.is_zero(&r));
        q
    }
}

impl<E: PositiveExponent> UnivariatePolynomial<PolynomialRing<IntegerRing, E>> {
    /// Compute the resultant modulo a sequence of word-sized primes and
    /// reconstruct its integer coefficients with the Chinese Remainder Theorem.
    ///
    /// Reconstruction stops early when the symmetric representative is stable
    /// and agrees with two fresh modular images. This is a Monte Carlo check;
    /// absent early termination, the Sylvester 1-norm bound provides a
    /// deterministic stopping point.
    pub fn resultant_ducos_crt(&self, other: &Self) -> MultivariatePolynomial<IntegerRing, E> {
        if let Some(resultant) = self.resultant_small(other) {
            return resultant;
        }

        let coefficient_bound = self.resultant_coefficient_bound(other);
        let deterministic_modulus = &coefficient_bound * 2;
        let mut rng = rand::rng();
        let mut primes = PrimeIteratorU64::new(
            rng.random_range(RESULTANT_CRT_PRIME_BLOCK..RESULTANT_CRT_PRIME_BLOCK * 3 / 2),
        );
        let mut verification_prime_iterators: [PrimeIteratorU64;
            RESULTANT_CRT_VERIFICATION_PRIMES] = std::array::from_fn(|index| {
            let block_start = (index as u64 + 2) * RESULTANT_CRT_PRIME_BLOCK;
            PrimeIteratorU64::new(
                rng.random_range(block_start..block_start + RESULTANT_CRT_PRIME_BLOCK / 2),
            )
        });
        let mut modulus = Integer::one();
        let mut reconstruction = self.coefficients[0].zero();
        let mut accumulated_primes = 0usize;

        loop {
            let (prime, image) = self.next_ducos_modular_image(other, &mut primes);
            let previous = reconstruction.clone();

            if accumulated_primes == 0 {
                reconstruction = image;
                modulus = prime;
            } else {
                reconstruction = reconstruction.chinese_remainder(&image, &modulus, &prime);
                modulus *= &prime;
            }
            accumulated_primes += 1;

            if modulus > deterministic_modulus {
                return reconstruction;
            }

            if accumulated_primes < 2 || reconstruction != previous {
                continue;
            }

            let mut verification_images = Vec::with_capacity(RESULTANT_CRT_VERIFICATION_PRIMES);
            let mut verified = true;
            for verification_primes in &mut verification_prime_iterators {
                let (prime, image) = self.next_ducos_modular_image(other, verification_primes);
                if !Self::congruent_modulo(&reconstruction, &image, &prime) {
                    verified = false;
                }
                verification_images.push((prime, image));

                if !verified {
                    break;
                }
            }

            if verified {
                return reconstruction;
            }

            // A failed check is still a valid image. Fold it into the CRT so
            // that none of the modular resultant work is discarded.
            for (prime, image) in verification_images {
                reconstruction = reconstruction.chinese_remainder(&image, &modulus, &prime);
                modulus *= &prime;
                accumulated_primes += 1;
            }

            if modulus > deterministic_modulus {
                return reconstruction;
            }
        }
    }

    /// Bound the 1-norm of the resultant by the product of the Sylvester row
    /// norms: `||self||_1^degree(other) * ||other||_1^degree(self)`.
    fn resultant_coefficient_bound(&self, other: &Self) -> Integer {
        let self_norm = self.coefficient_l1_norm();
        let other_norm = other.coefficient_l1_norm();
        &self_norm.pow(other.degree() as u64) * &other_norm.pow(self.degree() as u64)
    }

    fn coefficient_l1_norm(&self) -> Integer {
        let mut norm = Integer::zero();
        for polynomial in &self.coefficients {
            for coefficient in &polynomial.coefficients {
                norm += coefficient.abs();
            }
        }
        norm
    }

    fn next_ducos_modular_image(
        &self,
        other: &Self,
        primes: &mut PrimeIteratorU64,
    ) -> (Integer, MultivariatePolynomial<IntegerRing, E>) {
        loop {
            let prime = primes
                .next()
                .and_then(|p| u32::try_from(p).ok())
                .expect("Ran out of word-sized primes for resultant reconstruction");
            let field = Zp::new(prime);
            let polynomial_ring = PolynomialRing::new(field.clone());
            let self_mod = self.map_coeff(
                |polynomial| {
                    polynomial.map_coeff(
                        |coefficient| coefficient.to_finite_field(&field),
                        field.clone(),
                    )
                },
                polynomial_ring.clone(),
            );
            let other_mod = other.map_coeff(
                |polynomial| {
                    polynomial.map_coeff(
                        |coefficient| coefficient.to_finite_field(&field),
                        field.clone(),
                    )
                },
                polynomial_ring,
            );

            // A vanished outer leading coefficient changes the Sylvester
            // matrix dimensions, so such a prime cannot be used.
            if self_mod.degree() != self.degree() || other_mod.degree() != other.degree() {
                continue;
            }

            let image = self_mod
                .resultant_ducos(&other_mod)
                .map_coeff(|coefficient| field.to_symmetric_integer(coefficient), Z);
            return (prime.into(), image);
        }
    }

    fn congruent_modulo(
        reconstruction: &MultivariatePolynomial<IntegerRing, E>,
        image: &MultivariatePolynomial<IntegerRing, E>,
        prime: &Integer,
    ) -> bool {
        reconstruction.map_coeff(|c| c.clone().symmetric_mod(prime), Z) == *image
    }
}

impl<E: PositiveExponent> UnivariatePolynomial<PolynomialRing<RationalField, E>> {
    /// Clear rational denominators, compute the integer resultant with modular
    /// Ducos images and CRT reconstruction, and restore the overall scale.
    pub fn resultant_ducos_crt(&self, other: &Self) -> MultivariatePolynomial<RationalField, E> {
        if let Some(resultant) = self.resultant_small(other) {
            return resultant;
        }

        let self_denominator = self.coefficient_denominator_lcm();
        let other_denominator = other.coefficient_denominator_lcm();
        let integer_ring = PolynomialRing::new(Z);
        let self_integer = self.map_coeff(
            |polynomial| Self::clear_coefficient_denominators(polynomial, &self_denominator),
            integer_ring.clone(),
        );
        let other_integer = other.map_coeff(
            |polynomial| Self::clear_coefficient_denominators(polynomial, &other_denominator),
            integer_ring,
        );

        let scale = &self_denominator.pow(other.degree() as u64)
            * &other_denominator.pow(self.degree() as u64);
        self_integer.resultant_ducos_crt(&other_integer).map_coeff(
            |coefficient| Rational::from((coefficient.clone(), scale.clone())),
            Q,
        )
    }

    fn coefficient_denominator_lcm(&self) -> Integer {
        let mut denominator = Integer::one();
        for polynomial in &self.coefficients {
            for coefficient in &polynomial.coefficients {
                let gcd = Z.gcd(&denominator, coefficient.denominator_ref());
                denominator = Z.quot_rem(&denominator, &gcd).0 * coefficient.denominator_ref();
            }
        }
        denominator
    }

    fn clear_coefficient_denominators(
        polynomial: &MultivariatePolynomial<RationalField, E>,
        common_denominator: &Integer,
    ) -> MultivariatePolynomial<IntegerRing, E> {
        polynomial.map_coeff(
            |coefficient| {
                coefficient.numerator_ref()
                    * &Z.quot_rem(common_denominator, coefficient.denominator_ref())
                        .0
            },
            Z,
        )
    }
}

impl<F: Field> UnivariatePolynomial<F> {
    /// Compute the resultant of the two polynomials.
    pub fn resultant(&self, other: &Self) -> F::Element {
        if let Some(resultant) = self.resultant_small(other) {
            return resultant;
        }

        let mut a = self.clone();
        let mut a_new = other.clone();

        let mut v = vec![a.degree()];
        let mut c = vec![a.lcoeff()];

        while !a_new.is_constant() {
            let (_, r) = a.quot_rem(&a_new);
            (a, a_new) = (a_new, r);

            v.push(a.degree());
            c.push(a.lcoeff());
        }

        let r = a_new.lcoeff();
        if self.ring.is_zero(&r) {
            return r;
        }

        let mut sign = 0;
        for w in v.windows(2) {
            sign += w[0] * w[1];
        }

        let mut res = self.ring.pow(&r, *v.last().unwrap() as u64);
        if sign % 2 == 1 {
            res = self.ring.neg(&res);
        };

        v.push(0);
        for i in 1..c.len() {
            self.ring.mul_assign(
                &mut res,
                &self.ring.pow(&c[i], v[i - 1] as u64 - v[i + 1] as u64),
            );
        }

        res
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use crate::atom::AtomCore;
    use crate::domains::integer::Z;
    use crate::domains::rational::Q;
    use crate::domains::rational_polynomial::{
        FromNumeratorAndDenominator, RationalPolynomial, RationalPolynomialField,
    };
    use crate::poly::polynomial::MultivariatePolynomial;
    use crate::{parse, symbol};

    #[test]
    fn resultant() {
        let a = parse!("9v1^6-27v1^4-27v1^3+72v1^2+18v1-451")
            .to_polynomial::<_, u8>(&Q, None)
            .to_univariate_from_univariate(0);
        let b = parse!("3v1^4-4v1^2-9v1+21")
            .to_polynomial::<_, u8>(&Q, None)
            .to_univariate_from_univariate(0);
        let r = a.resultant(&b);
        assert_eq!(r, 11149673028381u64);
    }

    #[test]
    fn res_methods() {
        let (x, y, z) = symbol!("v1", "v2", "v3");
        let vars = Arc::new(vec![x.into(), y.into(), z.into()]);
        let a = parse!("2v1^2").to_polynomial::<_, u8>(&Z, Some(vars.clone()));

        let aa = a.to_univariate(0).map_coeff(
            |x| RationalPolynomial::from_num_den(x.clone(), x.one(), x.ring(), false),
            RationalPolynomialField::from_poly(&a),
        );
        let b = parse!("v3^2+v1^5")
            .to_polynomial::<_, u8>(&Z, Some(vars.clone()))
            .to_univariate(0)
            .map_coeff(
                |x| RationalPolynomial::from_num_den(x.clone(), x.one(), x.ring(), false),
                RationalPolynomialField::from_poly(&a),
            );

        let prs = aa.resultant_prs(&b);
        let ducos = aa.resultant_ducos(&b);
        let prim = aa.resultant_primitive(&b);
        let res = aa.resultant(&b);
        assert_eq!(prs, ducos);
        assert_eq!(prs, prim);
        assert_eq!(prim, res);
    }

    #[test]
    fn resultant_small_formulas() {
        let (x, y, z) = symbol!("x", "y", "z");
        let vars = Arc::new(vec![x.into(), y.into(), z.into()]);

        let zero = parse!("0").to_polynomial::<_, u8>(&Z, Some(vars.clone()));
        let constant = parse!("7").to_polynomial::<_, u8>(&Z, Some(vars.clone()));
        let linear = parse!("3*x+y").to_polynomial::<_, u8>(&Z, Some(vars.clone()));
        let high = parse!("2*x^7+z*x^2+y+1").to_polynomial::<_, u8>(&Z, Some(vars.clone()));

        let zero = zero.to_univariate(0);
        let constant = constant.to_univariate(0);
        assert!(zero.resultant_prs(&constant).is_zero());
        assert!(constant.resultant_prs(&zero).is_zero());
        assert!(zero.resultant_ducos(&constant).is_zero());
        assert!(constant.resultant_ducos(&zero).is_zero());
        assert!(zero.resultant_primitive(&constant).is_zero());
        assert!(constant.resultant_primitive(&zero).is_zero());
        assert!(zero.resultant_prs(&zero).is_zero());
        assert!(
            zero.resultant_prs_with_linear_subresultant(&constant)
                .0
                .is_zero()
        );
        assert!(
            constant
                .resultant_prs_with_linear_subresultant(&zero)
                .0
                .is_zero()
        );

        let q_vars = Arc::new(vec![x.into()]);
        let q_zero = parse!("0")
            .to_polynomial::<_, u8>(&Q, Some(q_vars.clone()))
            .to_univariate_from_univariate(0);
        let q_constant = parse!("7")
            .to_polynomial::<_, u8>(&Q, Some(q_vars))
            .to_univariate_from_univariate(0);
        assert_eq!(q_zero.resultant(&q_constant), 0u64);
        assert_eq!(q_constant.resultant(&q_zero), 0u64);

        let expected =
            parse!("-2*y^7+243*z*y^2+2187*y+2187").to_polynomial::<_, u8>(&Z, Some(vars.clone()));
        let high = high.to_univariate(0);
        let linear = linear.to_univariate(0);
        assert_eq!(high.resultant_prs(&linear), expected);
        assert_eq!(high.resultant_ducos(&linear), expected);
        assert_eq!(high.resultant_primitive(&linear), expected);

        let reverse = linear.resultant_ducos(&high);
        assert_eq!(reverse, -expected);

        let quadratic_a = parse!("x^2-y").to_polynomial::<_, u8>(&Z, Some(vars.clone()));
        let quadratic_b = parse!("x^2-z").to_polynomial::<_, u8>(&Z, Some(vars.clone()));
        let expected = parse!("(y-z)^2").to_polynomial::<_, u8>(&Z, Some(vars.clone()));
        let quadratic_a = quadratic_a.to_univariate(0);
        let quadratic_b = quadratic_b.to_univariate(0);
        assert_eq!(quadratic_a.resultant_prs(&quadratic_b), expected);
        assert_eq!(quadratic_a.resultant_ducos(&quadratic_b), expected);
        assert_eq!(quadratic_a.resultant_primitive(&quadratic_b), expected);
    }

    #[test]
    fn resultant_ducos_matches_brown() {
        let (x, y, z) = symbol!("x", "y", "z");
        let vars = Arc::new(vec![x.into(), y.into(), z.into()]);
        let cases = [
            ("x^8+(y+1)*x^5+(y^2+z)*x^2+3", "2*x^5+(y-1)*x^4+z*x+1"),
            (
                "(y+1)*x^6+z*x^5+(y*z+2)*x^3+y^2+1",
                "(z+1)*x^6+y*x^4+(z^2-1)*x^2+3",
            ),
            ("x^12+y*x^4+z*x+1", "x^7+(y+z)*x^2+2"),
            ("3*x^9+(y^2+1)*x^8+z*x^3+y", "(y+2)*x^7+z*x^6+(y*z-1)*x+4"),
            ("(x^3+y*x+1)*(x^4+z*x+2)", "(x^3+y*x+1)*(x^2+y+z)"),
        ];

        for (a, b) in cases {
            let a = parse!(a).to_polynomial::<_, u16>(&Z, Some(vars.clone()));
            let b = parse!(b).to_polynomial::<_, u16>(&Z, Some(vars.clone()));
            let a = a.to_univariate(0);
            let b = b.to_univariate(0);

            assert_eq!(a.resultant_ducos(&b), a.resultant_prs(&b));
            assert_eq!(a.resultant_ducos_crt(&b), a.resultant_prs(&b));
            assert_eq!(b.resultant_ducos(&a), b.resultant_prs(&a));
        }
    }

    #[test]
    fn resultant_ducos_crt_with_rational_coefficients() {
        let (x, y, z) = symbol!("x", "y", "z");
        let vars = Arc::new(vec![x.into(), y.into(), z.into()]);
        let a = parse!("(y/2+z/3)*x^6+(y*z/5+7/11)*x^3+y/13+1")
            .to_polynomial::<_, u16>(&Q, Some(vars.clone()))
            .to_univariate(0);
        let b = parse!("(z/7+2/3)*x^4+(y/17-z/19)*x^2+5/23")
            .to_polynomial::<_, u16>(&Q, Some(vars))
            .to_univariate(0);

        assert_eq!(a.resultant_ducos_crt(&b), a.resultant_ducos(&b));
    }

    #[test]
    fn resultant_prs_large() {
        let system = [
        "-272853213601 + 114339252960*v2 - 4841413740*v2^2 + 296664007680*v4 - 25123011840*v2*v4 -
    32592015360*v4^2 - 4907531205*v5 + 6155208630*v5^2 - 3860046090*v5^3 + 1210353435*v5^4 -
    151807041*v5^5 + 312814245280*v6 - 97612876080*v2*v6 + 1518070410*v2^2*v6 -
    253265840640*v4*v6 + 7877554560*v2*v4*v6 + 10219530240*v4^2*v6 - 146051082720*v6^2 +
    29048482440*v2*v6^2 + 75369035520*v4*v6^2 + 35852138640*v6^3 - 3036140820*v2*v6^3 -
    7877554560*v4*v6^3 - 4841413740*v6^4 + 303614082*v6^5",
        "-121828703201 - 1128406464*v1 + 303614082*v1^2 + 24547177584*v2 - 303614082*v2^2 -
    2927757312*v3 + 1575510912*v1*v3 + 2043906048*v3^2 + 123022775808*v4 - 6600113280*v2*v4 -
    15080712192*v4^2 + 1480577211*v5 + 146055798*v5^2 - 1347744906*v5^3 + 816475707*v5^4 -
    151807041*v5^5 + 171636450272*v6 - 32717479104*v2*v6 + 303614082*v2^2*v6 -
    135541762560*v4*v6 + 3151021824*v2*v4*v6 + 6131718144*v4^2*v6 - 95005523376*v6^2 +
    13441077468*v2*v6^2 + 49947954048*v4*v6^2 + 26959925088*v6^3 - 1821684492*v2*v6^3 -
    6302043648*v4*v6^3 - 4176745074*v6^4 + 303614082*v6^5",
    ];

        let mut system = system
            .iter()
            .map(|s| parse!(s).to_polynomial::<_, u16>(&Z, None))
            .collect::<Vec<_>>();
        MultivariatePolynomial::unify_variables_list(&mut system);

        let var = 0;
        let a = system[0].to_univariate(var);
        let b = system[1].to_univariate(var);

        let r = a.resultant_prs(&b);
        let r_ducos = a.resultant_ducos(&b);
        assert_eq!(r, r_ducos);

        let res = "-351386377558921617913117495604303232443676-13790107017999999428952788718610765086720*v3+9827971852963339087984510765471845089280*v3^2-280524240668642743539521434896511795200*v3^3+97918838723960202933606538595952230400*v3^4-5314937079854166446575553985297899043840*v1+7575728303325907213654727048384547256320*v1*v3-324356153273118172217571659099091763200*v1*v3^2+150958209699438646189310080335426355200*v1*v3^3+1459905975120096702631379691615772127520*v1^2-125012267407347628875522410277774950400*v1^2*v3+87272714982487967328194890193918361600*v1^2*v3^2-16060603798860632876369198542630809600*v1^3+22424239266333713827383409285937356800*v1^3*v3+2160668887641529717742672248905422400*v1^4+1354294280612676780452085325751267257668192*v6+39734613593311178569710537839540105625600*v6*v3-27865256722167384570188475202759258521600*v6*v3^2+175921981436267483236649035443575193600*v6*v3^3-61406729369263178110905795390681907200*v6*v3^4+15314382322422016740409269792322749043200*v6*v1-21479468723337358939520282968793595110400*v6*v1*v3+203409791035684277492375447231633817600*v6*v1*v3^2-94668707777614066254313101227301273600*v6*v1*v3^3-4139272618559803545636721197111265724400*v6*v1^2+78397523628336648616853036953858867200*v6*v1^2*v3-54730346683933132053274761647033548800*v6*v1^2*v3^2+10071904077251583329248480441988812800*v6*v1^3-14062658522955040874799765145418342400*v6*v1^3*v3-1354995743097230500957269037449163200*v6*v1^4-2367896305717791221956898715505617501445056*v6^2-46292687892477418837577147366397047930880*v6^2*v3+32337290781902786933956629706880358481920*v6^2*v3^2-27580988615008037626084806404289331200*v6^2*v3^3+9627326214672616907218281480742502400*v6^2*v3^4-17841973458559005176982858880798862223360*v6^2*v1+24926661644383398261591568732386942996480*v6^2*v1*v3-31890518086103043505160557404959539200*v6^2*v1*v3^2+14842127914286951065294850616144691200*v6^2*v1*v3^3+4803575421053050706660875224470400473280*v6^2*v1^2-12291137179018881350947298166494822400*v6^2*v1^2*v3+8580605200447143584623585512458649600*v6^2*v1^2*v3^2-1579069707026731284670312611667737600*v6^2*v1^3+2204738836226002171049115721951180800*v6^2*v1^3*v3+212435773282192917522961671125504400*v6^2*v1^4+2494961992653296325028324212640453124889600*v6^3+28849190833763405479699812009600299827200*v6^3*v3-20140001148098981183941378195381341388800*v6^3*v3^2+11118958967179645861967635878700115558400*v6^3*v1-15524584218326297995954812358939783987200*v6^3*v1*v3-2991716750406630342970458631670687539200*v6^3*v1^2-1772535279912235533967029650358928182132480*v6^4-10505535408790438577935793180820583219200*v6^4*v3+7334053021231060894030648069629463756800*v6^4*v3^2-4049008438804648201912753621774599782400*v6^4*v1+5653332537198942772481957887006044979200*v6^4*v1*v3+1089444291022712930113710634475123251200*v6^4*v1^2+898690122182833295589205662503794559796864*v6^5+2251068354128239337582288282696747581440*v6^5*v3-1571500549108393499821597480373201141760*v6^5*v3^2+867599261486925578026506942289371463680*v6^5*v1-1211365006604386656112481391121009213440*v6^5*v1*v3-233440131481053678521676101413944483840*v6^5*v1^2-334972523815164547408286541369481963062528*v6^6-263982621934964082348273502963276185600*v6^6*v3+184289754935729642394077728483796582400*v6^6*v3^2-101743302204100740071730412600429363200*v6^6*v1+142056686096291599345434915706259865600*v6^6*v1*v3+27375507216472860290526520214227161600*v6^6*v1^2+92887008215136370072837287898141019689344*v6^7+13110546324286806774343784710927810560*v6^7*v3-9152645547143619823598491213289226240*v6^7*v3^2+5053023062485540110945000357336760320*v6^7*v1-7055164275923206947357170310243778560*v6^7*v1*v3-1359588949006034672146954695203228160*v6^7*v1^2-19087257764871901513773677515006244966400*v6^8+2841567690972631378229632354823913000960*v6^9-291195946735118945124677882620928308224*v6^10+18431616670849378149970607435511871488*v6^11-543835579602413868858781878081291264*v6^12+15597477966285265118914372191523231041000*v5-245447710139298529858941866813755392000*v5*v3+171350288210453690656242435322810368000*v5*v3^2-94599638282854641716467177834468224000*v5*v1+132082513828891386547520210561332992000*v5*v1*v3+25453401102442610949261707243590212000*v5*v1^2-45411984028105044019699380521763730985280*v5*v6+182919978541731072291801887104756531200*v5*v6*v3-127698852944227352354654147601433804800*v5*v6*v3^2+70500408396292184112465310654958246400*v5*v6*v1-98434532477841917440045905442771891200*v5*v6*v1*v3-18969154696250786173342179694700833200*v5*v6*v1^2+55003612820966057503024025353699032766080*v5*v6^2-33223980977628423890341541470219468800*v5*v6^2*v3+23194099927778333659295038384870195200*v5*v6^2*v3^2-12805076001794288374402469108313753600*v5*v6^2*v1+17878785360995798862373258755004108800*v5*v6^2*v1*v3+3445390928941898739103180072578916800*v5*v6^2*v1^2-36212231181655357463687198453157452313600*v5*v6^3+14069264595301729477604555666412809299200*v5*v6^4-3237479989051682809580844453154845667200*v5*v6^5+409563107761162882925994444729920136960*v5*v6^6-22013254475726077349297074842098808960*v5*v6^7-10053352936060966782937950767819809690860*v5^2+32932755432170760569169811376981606400*v5^2*v3-22990791528119210208665717376383385600*v5^2*v3^2+12692832822815813969367531468211660800*v5^2*v1-17722068469591891202513157144295526400*v5^2*v1*v3-3415190277994270700484306324681950400*v5^2*v1^2+29707896766397290357555785282575595878880*v5^2*v6-57019534921064997757748321171863142400*v5^2*v6*v3+39806090416592545604465809119979929600*v5^2*v6*v3^2-21976279084160467885798832118322252800*v5^2*v6*v1+30683861362790087236775727863317862400*v5^2*v6*v1*v3+5913035783454339727920322556993546400*v5^2*v6*v1^2-37302062378389232891674984124380580041920*v5^2*v6^2+14641076363022695273709831834334003200*v5^2*v6^2*v3-10221128781732825002401203356044492800*v5^2*v6^2*v3^2+5642914848248330470075664352816230400*v5^2*v6^2*v1-7878786769252385939350927586950963200*v5^2*v6^2*v1*v3-1518307866991345207062418337068675200*v5^2*v6^2*v1^2+25681302707666851532860725046888387123200*v5^2*v6^3-10458741720579739914368041771228643980800*v5^2*v6^4+2520479886656951321446241477235442417920*v5^2*v6^5-333083286426653512111002147322570191360*v5^2*v6^6+18640148890139899619012458968935427840*v5^2*v6^7+414911263816261790113586737614444084960*v5^3+151752777978897395607235689418029465600*v5^3*v3-105940618589041578065428688839001702400*v5^3*v3^2+58488049846033371223622088629865523200*v5^3*v1-81662560162386216425434614313397145600*v5^3*v1*v3-15737055864626510456984795466644241600*v5^3*v1^2-1585672703184495673894588405081928144640*v5^3*v6-72360704332631397795065899642766131200*v5^3*v6*v3+50515963402025692800329024278912204800*v5^3*v6*v3^2-27889021461535017900181648820649446400*v5^3*v6*v1+38939388455728138200253622881661491200*v5^3*v6*v1*v3+7503944650322609965673875242820183200*v5^3*v6*v1^2+3527447631833347104006026820522109140480*v5^3*v6^2+7769123963168496709778228887319347200*v5^3*v6^2*v3-5423728049759139212486688091147468800*v5^3*v6^2*v3^2+2994349860804524773560359050320998400*v5^3*v6^2*v1-4180790371689336476291822070259507200*v5^3*v6^2*v1*v3-805673144544299216785403211456259200*v5^3*v6^2*v1^2-3727514692990837419389364030442284902400*v5^3*v6^3+2051485173833671116475035035491521907200*v5^3*v6^4-614821883114260826429743802606140235520*v5^3*v6^5+95621327057944630975369889554673978880*v5^3*v6^6-6064521488024361377257398718961660160*v5^3*v6^7+1627462659741295232635314702377463845640*v5^4-101642857058676788342485563311434137600*v5^4*v3+70958220965491342805131430991001190400*v5^4*v3^2-39174851158031678840332977526281907200*v5^4*v1+54696961994232910078955478055563417600*v5^4*v1*v3+10540560384305300379798711916957533600*v5^4*v1^2-4785395055990532927423173887678038866240*v5^4*v6+56591005231715981715543689963314790400*v5^4*v6*v3-39506928180631911763681443936653721600*v5^4*v6*v3^2+21811116599723867952865797173360908800*v5^4*v6*v1-30453257139237098651171113034503910400*v5^4*v6*v1*v3-5868596427873815885902766574357524400*v5^4*v6*v1^2+5114113403851989582240098526003298972800*v5^4*v6^2-7751167590372267518615413258149888000*v5^4*v6^2*v3+5411192468750450909222080953802752000*v5^4*v6^2*v3^2-2987429175455978106133023859911936000*v5^4*v6^2*v1+4171127527995139242525354068556288000*v5^4*v6^2*v1*v3+803811034040729958194990106961368000*v5^4*v6^2*v1^2-2712349074575972935281868605613212595200*v5^4*v6^3+770070834029855766768147977571966240000*v5^4*v6^4-110058118810184370995065087641038839680*v5^4*v6^5+5531213039802125716963085591624820480*v5^4*v6^6+137796177264125135690569732621948800*v5^4*v6^7-373744237967747318097655035394806229392*v5^5+19528752507418994002306184430398177280*v5^5*v3-13633280052349109020477902338202501120*v5^5*v3^2+7526706695567737271722175249215964160*v5^5*v1-10508986707019104869951716385697761280*v5^5*v1*v3-2025169313331806667646945345160506080*v5^5*v1^2+1215536820450847752474367554142920011904*v5^5*v6-11349923971616534580829712270862336000*v5^5*v6*v3+7923531829241731688503761396639744000*v5^5*v6*v3^2-4374449864060539369694784937728192000*v5^5*v6*v1+6107722451707168176554982743243136000*v5^5*v6*v1*v3+1177009014131068867356949799479146000*v5^5*v6*v1^2-1413634176130610954389489058318893777152*v5^5*v6^2+1638818290535850846792973088865976320*v5^5*v6^2*v3-1144080693392952477949811401661153280*v5^5*v6^2*v3^2+631627882810692513868125044667095040*v5^5*v6^2*v1-881895534490400868419646288780472320*v5^5*v6^2*v1*v3-169948618625754334018369336900403520*v5^5*v6^2*v1^2+826980745012980525654275093992819891200*v5^5*v6^3-273850667214219945816540968220232769280*v5^5*v6^4+51787378668025078969009698679958550144*v5^5*v6^5-5148065182587715069399685210756007168*v5^5*v6^6+203938342350905200822043204280484224*v5^5*v6^7+8396010529199487178908025111075470600*v5^6+12821686359815483795642849669657085120*v5^6*v6-3486741933219625604409869636436712320*v5^6*v6^2-36785147870484641738753696782202499360*v5^7+12905301212905230238157985715741313280*v5^7*v6-916923342881865840823092794954641920*v5^7*v6^2+20230288344237735053451621915739901460*v5^8-9430829758183594873828448918573310240*v5^8*v6+1082631046775166944466178953329743680*v5^8*v6^2-4939771095356636533922616686506369560*v5^9+2523408013736821291155145004510228160*v5^9*v6-321524413616291983277996042784547200*v5^9*v6^2+474541792488342833563034227065152484*v5^10-254004286756870666789616873799792288*v5^10*v6+33989723725150866803673867380080704*v5^10*v6^2+340054807036742119278407339941916972482560*v4-1168465430599048731112810012610311926251520*v4*v6+1803251539160284741968818546040707238297600*v4*v6^2-1645584197198932009018040684997750258401280*v4*v6^3+983225719045982886492157957057230189711360*v4*v6^4-401872324263573241960647370747903148544000*v4*v6^5+113789782501275194186732116722172129443840*v4*v6^6-22040058529475774799832268879481077760000*v4*v6^7+2794927291759340041335283859047296860160*v4*v6^8-209557447006475795544473788404267909120*v4*v6^9+7055164275923206947357170310243778560*v4*v6^10-4148924309292177805612699194701222707200*v4*v5+7776797270776577512297113724354201190400*v4*v5*v6-5792702378959601123657932251494387712000*v4*v5*v6^2+2140476011255620382325631426462128537600*v4*v5*v6^3-392197080479229244438225955932453171200*v4*v5*v6^4+28509414494560868456216818014736281600*v4*v5*v6^5+5203735574366460298565080345896448819200*v4*v5^2-9753949119279097218813329078003574374400*v4*v5^2*v6+7265423322762889544926898078145503232000*v4*v5^2*v6^2-2684664827676540818510113992511822233600*v4*v5^2*v6^3+491908202634965493024215605745788723200*v4*v5^2*v6^4-35757570721991597724746517510008217600*v4*v5^2*v6^5-3263359597484051373676406318613027225600*v4*v5^3+6116883345988586391459206370951394099200*v4*v5^3*v6-4556282422749608697666020828667518976000*v4*v5^3*v6^2+1683603366509017123472444368185380044800*v4*v5^3*v6^3-308484805042266495625355549366003097600*v4*v5^3*v6^4+22424239266333713827383409285937356800*v4*v5^3*v6^5+1023256822939914413779890116853237350400*v4*v5^4-1918005794928624546474496912925437132800*v4*v5^4*v6+1428664827472334930624091276785577984000*v4*v5^4*v6^2-527909530176556216682037640871686963200*v4*v5^4*v6^3+96728286326812375746933519716458598400*v4*v5^4*v6^4-7031329261477520437399882572709171200*v4*v5^4*v6^5-128340686267040112914765879062948413440*v4*v5^5+240563438685963078710360629756749742080*v4*v5^5*v6-179188469886360652315563990647682662400*v4*v5^5*v6^2+66212381750957898363509805804245483520*v4*v5^5*v6^3-12132022352854433568259458405115146240*v4*v5^5*v6^4+881895534490400868419646288780472320*v4*v5^5*v6^5-49164550414950667847480579268710887587840*v4^2+123028102530480819459927412746971678638080*v4^2*v6-134877064351018867955458522266590727045120*v4^2*v6^2+84554433522522008212768621789075249397760*v4^2*v6^3-33138726988704698123377646193993999974400*v4^2*v6^4+8312765414115754783491341757137477959680*v4^2*v6^5-1303272543738486978767710360229180866560*v4^2*v6^6+116758072925724015046986158180338237440*v4^2*v6^7-4576322773571809911799245606644613120*v4^2*v6^8+599844478451881128522317955860417740800*v4^2*v5-564260483967447502254044856783952281600*v4^2*v5*v6+176929134803352182910166607635646054400*v4^2*v5*v6^2-18492593185661103863491990063612723200*v4^2*v5*v6^3-752347311956596669672059809045269708800*v4^2*v5^2+707716539213408731640666430542584217600*v4^2*v5^2*v6-221911118227933246361903880763352678400*v4^2*v5^2*v6^2+23194099927778333659295038384870195200*v4^2*v5^2*v6^3+471811026142272487760444287028389478400*v4^2*v5^3-443822236455866492723807761526705356800*v4^2*v5^3*v6+139164599566670001955770230309221171200*v4^2*v5^3*v6^2-14545452497081327888032481698986393600*v4^2*v5^3*v6^3-147940745485288830907935920508901785600*v4^2*v5^4+139164599566670001955770230309221171200*v4^2*v5^4*v6-43636357491243983664097445096959180800*v4^2*v5^4*v6^2+4560862223661094337772896803919462400*v4^2*v5^4*v6^3+18555279942222666927436030707896156160*v4^2*v5^5-17454542996497593465638978038783672320*v4^2*v5^5*v6+5473034668393313205327476164703354880*v4^2*v5^5*v6^2-572040346696476238974905700830576640*v4^2*v5^5*v6^3";
        let res = parse!(res).to_polynomial::<_, u16>(&Z, system[0].variables().clone());

        assert_eq!(r, res);
    }
}
