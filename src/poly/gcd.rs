//! Compute the greatest common divisor (GCD) of multivariate polynomials with coefficients that implement [PolynomialGCD].

use ahash::{HashMap, HashSet, HashSetExt};
use rand;
use smallvec::{SmallVec, smallvec};
use std::borrow::Cow;
use std::cmp::{Ordering, max, min};
use std::mem;
use std::ops::Add;
use tracing::{debug, instrument};

use crate::GLOBAL_SETTINGS;
use crate::domains::algebraic_number::{AlgebraicExtension, GaloisField};
use crate::domains::finite_field::{
    FiniteField, FiniteFieldCore, FiniteFieldElement, FiniteFieldWorkspace, PrimeIteratorU64,
    SMOOTH_PRIME_BASE, SMOOTH_PRIMES, ToFiniteField, Zp, Zp64,
};
use crate::domains::float::{C, Complex, F64, Float, FloatField, SingleFloat};
use crate::domains::integer::{FromFiniteField, Integer, IntegerRing, SMALL_PRIMES, Z};
use crate::domains::rational::{Q, Rational, RationalField};
use crate::domains::{EuclideanDomain, Field, InternalOrdering, Ring, RingOps, SelfRing, Set};
use crate::poly::INLINED_EXPONENTS;
use crate::poly::factor::Factorize;
use crate::tensors::matrix::{Matrix, MatrixError};

use super::PositiveExponent;
use super::polynomial::MultivariatePolynomial;

/// The maximum power of a variable that is cached
pub(crate) const POW_CACHE_SIZE: usize = 1000;
pub(crate) const INITIAL_POW_MAP_SIZE: usize = 1000;

/// The upper bound of the range to be sampled during the computation of multiple gcds
pub(crate) const MAX_RNG_PREFACTOR: u32 = 50000;

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
enum GCDError {
    BadOriginalImage,
    BadCurrentImage,
}

impl<R: Ring, E: PositiveExponent> MultivariatePolynomial<R, E> {
    /// Evaluation of the exponents by filling in the variables
    #[inline(always)]
    fn evaluate_exponents(
        &self,
        r: &[(usize, R::Element)],
        cache: &mut [Vec<R::Element>],
    ) -> Vec<R::Element> {
        let mut eval = vec![self.ring.one(); self.nterms()];
        for (c, t) in eval.iter_mut().zip(self) {
            // evaluate each exponent
            for (n, v) in r {
                let exp = t.exponents[*n].to_u32() as usize;
                if exp > 0 {
                    if exp < cache[*n].len() {
                        if self.ring.is_zero(&cache[*n][exp]) {
                            cache[*n][exp] = self.ring.pow(v, exp as u64);
                        }

                        self.ring.mul_assign(c, &cache[*n][exp]);
                    } else {
                        self.ring.mul_assign(c, &self.ring.pow(v, exp as u64));
                    }
                }
            }
        }
        eval
    }

    /// Evaluate a polynomial using the evaluation of the exponent of every monomial.
    #[inline(always)]
    fn evaluate_using_exponents(
        &self,
        exp_evals: &[R::Element],
        main_var: usize,
        out: &mut MultivariatePolynomial<R, E>,
    ) {
        out.clear();
        let mut c = self.ring.zero();
        let mut new_exp = vec![E::zero(); self.nvars()];
        for (aa, e) in self.into_iter().zip(exp_evals) {
            if aa.exponents[main_var] != new_exp[main_var] {
                if !self.ring.is_zero(&c) {
                    out.coefficients.push(c);
                    out.exponents.extend_from_slice(&new_exp);

                    c = self.ring.zero();
                }

                new_exp[main_var] = aa.exponents[main_var];
            }

            self.ring.add_mul_assign(&mut c, aa.coefficient, e);
        }

        if !self.ring.is_zero(&c) {
            out.coefficients.push(c);
            out.exponents.extend_from_slice(&new_exp);
        }
    }
}

impl<F: Field, E: PositiveExponent> MultivariatePolynomial<F, E> {
    /// Compute the univariate GCD using Euclid's algorithm. The result is normalized to 1.
    pub fn univariate_gcd(&self, b: &Self) -> Self {
        if self.is_zero() {
            return b.clone();
        }
        if b.is_zero() {
            return self.clone();
        }

        let mut c = self.clone();
        let mut d = b.clone();
        if self.ldegree_max() < b.ldegree_max() {
            mem::swap(&mut c, &mut d);
        }

        // TODO: there exists an efficient algorithm for univariate poly
        // division in a finite field using FFT
        let mut r = c.quot_rem_univariate(&mut d).1;
        while !r.is_zero() {
            c = d;
            d = r;
            r = c.quot_rem_univariate(&mut d).1;
        }

        // normalize the gcd
        if let Some(l) = d.coefficients.last()
            && !d.ring.is_one(l)
        {
            let i = d.ring.inv(l);
            for x in &mut d.coefficients {
                d.ring.mul_assign(x, &i);
            }
        }

        d
    }

    /// Replace all variables except `v` in the polynomial by elements from
    /// a finite field of size `p`.
    pub fn sample_polynomial(
        &self,
        v: usize,
        r: &[(usize, F::Element)],
        cache: &mut [Vec<F::Element>],
        tm: &mut HashMap<E, F::Element>,
    ) -> Self {
        for mv in self.into_iter() {
            let mut c = mv.coefficient.clone();
            for (n, vv) in r {
                let exp = mv.exponents[*n].to_u32() as usize;
                if exp > 0 {
                    if exp < cache[*n].len() {
                        if self.ring.is_zero(&cache[*n][exp]) {
                            cache[*n][exp] = self.ring.pow(vv, exp as u64);
                        }

                        self.ring.mul_assign(&mut c, &cache[*n][exp]);
                    } else {
                        self.ring.mul_assign(&mut c, &self.ring.pow(vv, exp as u64));
                    }
                }
            }

            tm.entry(mv.exponents[v])
                .and_modify(|e| self.ring.add_assign(e, &c))
                .or_insert(c);
        }

        let mut res = self.zero();
        let mut e = vec![E::zero(); self.nvars()];
        for (k, c) in tm.drain() {
            if !self.ring.is_zero(&c) {
                e[v] = k;
                res.append_monomial(c, &e);
                e[v] = E::zero();
            }
        }

        res
    }

    /// Find the upper bound of a variable `var` in the gcd.
    /// This is done by computing the univariate gcd by
    /// substituting all variables except `var`. This
    /// upper bound could be too tight due to an unfortunate
    /// sample point, but this is rare.
    fn get_gcd_var_bound(ap: &Self, bp: &Self, vars: &[usize], var: usize) -> E {
        let mut rng = rand::rng();

        // store a table for variables raised to a certain power
        let mut cache = (0..ap.nvars())
            .map(|i| {
                vec![
                    ap.ring.zero();
                    min(
                        max(ap.degree(i), bp.degree(i)).to_u32() as usize + 1,
                        POW_CACHE_SIZE
                    )
                ]
            })
            .collect::<Vec<_>>();

        // store a power map for the univariate polynomials that will be sampled
        // the sampling_polynomial routine will set the power to 0 after use
        let mut tm = HashMap::with_capacity_and_hasher(INITIAL_POW_MAP_SIZE, Default::default());

        // generate random numbers for all non-leading variables
        // TODO: apply a Horner scheme to speed up the substitution?

        let mut fail_count = 0;
        let (_, a1, b1) = loop {
            for v in &mut cache {
                for vi in v {
                    *vi = ap.ring.zero();
                }
            }

            let r: Vec<_> = vars
                .iter()
                .map(|i| (*i, ap.ring.sample(&mut rng, (1, MAX_RNG_PREFACTOR as i64))))
                .collect();

            let a1 = ap.sample_polynomial(var, &r, &mut cache, &mut tm);
            let b1 = bp.sample_polynomial(var, &r, &mut cache, &mut tm);

            if a1.ldegree(var) == ap.degree(var) && b1.ldegree(var) == bp.degree(var) {
                break (r, a1, b1);
            }

            if let Some(size) = ap.ring.size()
                && fail_count * 2 > size
            {
                debug!("Field is too small to find a good sample point");
                // TODO: upgrade to larger field?
                return ap.degree(var).min(bp.degree(var));
            }

            debug!(
                "Degree error during sampling: trying again: a={}, a1={}, bp={}, b1={}",
                ap, a1, bp, b1
            );
            fail_count += 1;
        };

        let g1 = a1.univariate_gcd(&b1);
        g1.ldegree_max()
    }

    fn solve_vandermonde(
        &self,
        main_var: usize,
        shape: &[(MultivariatePolynomial<F, E>, E)],
        row_sample_values: Vec<Vec<F::Element>>,
        samples: Vec<Vec<F::Element>>,
    ) -> MultivariatePolynomial<F, E> {
        let mut gp = self.zero();

        // solve the transposed Vandermonde system
        for (((c, ex), sample), rhs) in shape.iter().zip(&row_sample_values).zip(&samples) {
            if c.nterms() == 1 {
                let coeff = self.ring.div(&rhs[0], &sample[0]);
                let mut ee: SmallVec<[E; INLINED_EXPONENTS]> = c.exponents(0).into();
                ee[main_var] = *ex;
                gp.append_monomial(coeff, &ee);
                continue;
            }

            // construct the master polynomial (1-s1)*(1-s2)*... efficiently
            let mut master = vec![self.ring.zero(); sample.len() + 1];
            master[0] = self.ring.one();

            for (i, x) in sample.iter().take(c.nterms()).enumerate() {
                let first = &mut master[0];
                let mut old_last = first.clone();
                self.ring.mul_assign(first, &self.ring.neg(x));
                for m in &mut master[1..=i] {
                    let ov = m.clone();
                    self.ring.mul_assign(m, &self.ring.neg(x));
                    self.ring.add_assign(m, &old_last);
                    old_last = ov;
                }
                master[i + 1] = self.ring.one();
            }

            for (i, s) in sample.iter().take(c.nterms()).enumerate() {
                let mut norm = self.ring.one();

                // sample master/(1-s_i) by using the factorized form
                for (j, l) in sample.iter().enumerate() {
                    if j != i {
                        self.ring.mul_assign(&mut norm, &self.ring.sub(s, l))
                    }
                }

                // divide out 1-s_i
                let mut coeff = self.ring.zero();
                let mut last_q = self.ring.zero();
                for (m, rhs) in master.iter().skip(1).zip(rhs).rev() {
                    last_q = self.ring.add(m, &self.ring.mul(s, &last_q));
                    self.ring.add_mul_assign(&mut coeff, &last_q, rhs);
                }
                self.ring.div_assign(&mut coeff, &norm);

                // divide by the Vandermonde row since the Vandermonde matrices should start with a 1
                self.ring.div_assign(&mut coeff, s);

                let mut ee: SmallVec<[E; INLINED_EXPONENTS]> = c.exponents(i).into();
                ee[main_var] = *ex;

                gp.append_monomial(coeff, &ee);
            }
        }

        gp
    }

    fn solve_vandermonde_matrix(
        &self,
        matrix: Vec<Vec<F::Element>>,
        rhs: Vec<F::Element>,
    ) -> Vec<F::Element> {
        let m = Matrix::from_nested_vec(matrix, self.ring.clone()).unwrap();
        m.solve(&Matrix::new_vec(rhs, self.ring.clone()))
            .unwrap()
            .into_vec()
    }

    /// Perform Newton interpolation in the variable `x`, by providing
    /// a list of sample points `a` and their evaluations `u`.
    pub fn newton_interpolation(
        a: &[F::Element],
        u: &[MultivariatePolynomial<F, E>],
        x: usize, // the variable index to extend the polynomial by
    ) -> MultivariatePolynomial<F, E> {
        let field = &u[0].ring;

        // compute inverses
        let mut gammas = Vec::with_capacity(a.len());
        for k in 1..a.len() {
            let mut pr = field.sub(&a[k], &a[0]);
            for i in 1..k {
                u[0].ring.mul_assign(&mut pr, &field.sub(&a[k], &a[i]));
            }
            gammas.push(u[0].ring.inv(&pr));
        }

        // compute Newton coefficients
        let mut v = vec![u[0].clone()];
        for k in 1..a.len() {
            let mut tmp = v[k - 1].clone();
            for j in (0..k - 1).rev() {
                tmp = tmp.mul_coeff(field.sub(&a[k], &a[j])).add(v[j].clone());
            }

            let mut r = u[k].clone() - tmp;
            r = r.mul_coeff(gammas[k - 1].clone());
            v.push(r);
        }

        // convert to standard form
        let mut e = vec![E::zero(); u[0].nvars()];
        e[x] = E::one();
        let xp = u[0].monomial(field.one(), e);
        let mut u = v[v.len() - 1].clone();
        for k in (0..v.len() - 1).rev() {
            // TODO: prevent cloning
            u = u * &(xp.clone() - v[0].constant(a[k].clone())) + v[k].clone();
        }
        u
    }

    #[instrument(level = "trace", fields(%a, %b))]
    fn construct_new_image_single_scale(
        a: &MultivariatePolynomial<F, E>,
        b: &MultivariatePolynomial<F, E>,
        a_ldegree: E,
        b_ldegree: E,
        bounds: &mut [E],
        single_scale: usize,
        vars: &[usize],
        main_var: usize,
        shape: &[(MultivariatePolynomial<F, E>, E)],
    ) -> Result<MultivariatePolynomial<F, E>, GCDError> {
        if vars.is_empty() {
            // return gcd divided by the single scale factor
            let g = a.univariate_gcd(b);

            if g.ldegree(main_var) < bounds[main_var] {
                // original image and form and degree bounds are unlucky
                // change the bound and try a new prime
                debug!("Unlucky degree bound: {} vs {}", g, bounds[main_var]);
                bounds[main_var] = g.ldegree(main_var);
                return Err(GCDError::BadOriginalImage);
            }

            if g.ldegree(main_var) > bounds[main_var] {
                return Err(GCDError::BadCurrentImage);
            }

            // check if all the monomials of the image appear in the shape
            // if not, the original shape is bad
            for m in g.into_iter() {
                if shape.iter().all(|(_, pow)| *pow != m.exponents[main_var]) {
                    debug!("Bad shape: terms missing");
                    return Err(GCDError::BadOriginalImage);
                }
            }

            // construct the scaling coefficient
            let (_, d) = &shape[single_scale];
            for t in &g {
                if t.exponents[main_var] == *d {
                    let scale_factor = a.ring.neg(&a.ring.inv(t.coefficient)); // TODO: why -1?
                    return Ok(g.mul_coeff(scale_factor));
                }
            }

            // the scaling term is missing, so the assumed form is wrong
            debug!("Bad original image");
            return Err(GCDError::BadOriginalImage);
        }

        let mut rng = rand::rng();

        let mut failure_count = 0;

        // store a table for variables raised to a certain power
        let mut cache = (0..a.nvars())
            .map(|i| {
                vec![
                    a.ring.zero();
                    min(
                        max(a.degree(i), b.degree(i)).to_u32() as usize + 1,
                        POW_CACHE_SIZE
                    )
                ]
            })
            .collect::<Vec<_>>();

        // find a set of sample points that yield unique coefficients for every coefficient of a term in the shape
        let (row_sample_values, samples) = 'find_root_sample: loop {
            for v in &mut cache {
                for vi in v {
                    *vi = a.ring.zero();
                }
            }

            let r_orig: SmallVec<[_; INLINED_EXPONENTS]> = vars
                .iter()
                .map(|i| (*i, a.ring.sample(&mut rng, (1, MAX_RNG_PREFACTOR as i64))))
                .collect();

            let mut row_sample_values = Vec::with_capacity(shape.len()); // coefficients for the linear system
            let mut samples_needed = 0;
            for (c, _) in shape.iter() {
                samples_needed = samples_needed.max(c.nterms());
                let mut row = Vec::with_capacity(c.nterms());
                let mut seen = HashSet::new();

                for t in c {
                    // evaluate each exponent
                    let mut c = a.ring.one();
                    for (n, v) in &r_orig {
                        let exp = t.exponents[*n].to_u32() as usize;
                        if exp > 0 {
                            if exp < cache[*n].len() {
                                if a.ring.is_zero(&cache[*n][exp]) {
                                    cache[*n][exp] = a.ring.pow(v, exp as u64);
                                }

                                a.ring.mul_assign(&mut c, &cache[*n][exp]);
                            } else {
                                a.ring.mul_assign(&mut c, &a.ring.pow(v, exp as u64));
                            }
                        }
                    }
                    row.push(c.clone());

                    // check if each element is unique
                    if !seen.insert(c.clone()) {
                        debug!("Duplicate element: restarting");
                        continue 'find_root_sample;
                    }
                }

                row_sample_values.push(row);
            }

            let mut samples = vec![Vec::with_capacity(samples_needed); shape.len()];
            let mut r = r_orig.clone();

            let a_eval = a.evaluate_exponents(&r_orig, &mut cache);
            let b_eval = b.evaluate_exponents(&r_orig, &mut cache);

            let mut a_current = Cow::Borrowed(&a_eval);
            let mut b_current = Cow::Borrowed(&b_eval);

            let mut a_poly = a.zero_with_capacity(a.degree(main_var).to_u32() as usize + 1);
            let mut b_poly = b.zero_with_capacity(b.degree(main_var).to_u32() as usize + 1);

            for sample_index in 0..samples_needed {
                // sample at r^i
                if sample_index > 0 {
                    for (c, rr) in r.iter_mut().zip(&r_orig) {
                        *c = (c.0, a.ring.mul(&c.1, &rr.1));
                    }

                    for (c, e) in a_current.to_mut().iter_mut().zip(&a_eval) {
                        a.ring.mul_assign(c, e);
                    }
                    for (c, e) in b_current.to_mut().iter_mut().zip(&b_eval) {
                        b.ring.mul_assign(c, e);
                    }
                }

                // now construct the univariate polynomials from the current evaluated monomials
                a.evaluate_using_exponents(&a_current, main_var, &mut a_poly);
                b.evaluate_using_exponents(&b_current, main_var, &mut b_poly);

                if a_poly.ldegree(main_var) != a_ldegree || b_poly.ldegree(main_var) != b_ldegree {
                    continue 'find_root_sample;
                }

                let g = a_poly.univariate_gcd(&b_poly);
                debug!(
                    "GCD of sample at point {:?} in main var {}: {}",
                    r, main_var, g
                );

                if g.ldegree(main_var) < bounds[main_var] {
                    // original image and form and degree bounds are unlucky
                    // change the bound and try a new prime

                    debug!("Unlucky degree bound: {} vs {}", g, bounds[main_var]);
                    bounds[main_var] = g.ldegree(main_var);
                    return Err(GCDError::BadOriginalImage);
                }

                if g.ldegree(main_var) > bounds[main_var] {
                    failure_count += 1;
                    if failure_count > 2 {
                        // p is likely unlucky
                        debug!(
                            "Bad current image: gcd({},{}) mod {} under {:?} = {}",
                            a, b, a.ring, r, g
                        );
                        return Err(GCDError::BadCurrentImage);
                    }
                    debug!("Degree too high");
                    continue 'find_root_sample;
                }

                // construct the scaling coefficient
                let mut scale_factor = a.ring.one();
                let mut coeff = a.ring.one();
                let (c, d) = &shape[single_scale];
                for (n, v) in r.iter() {
                    // TODO: can be taken from row?
                    a.ring.mul_assign(
                        &mut coeff,
                        &a.ring.pow(v, c.exponents(0)[*n].to_u32() as u64),
                    );
                }

                let mut found = false;
                for t in &g {
                    if t.exponents[main_var] == *d {
                        scale_factor = g.ring.div(&coeff, t.coefficient);
                        found = true;
                        break;
                    }
                }

                if !found {
                    // the scaling term is missing, so the assumed form is wrong
                    debug!("Bad original image");
                    return Err(GCDError::BadOriginalImage);
                }

                // check if all the monomials of the image appear in the shape
                // if not, the original shape is bad
                for m in g.into_iter() {
                    if shape.iter().all(|(_, pow)| *pow != m.exponents[main_var]) {
                        debug!("Bad shape: terms missing");
                        return Err(GCDError::BadOriginalImage);
                    }
                }

                // construct the right-hand side
                'rhs: for (i, (rhs, (shape_part, exp))) in samples.iter_mut().zip(shape).enumerate()
                {
                    // we may not need all terms
                    if rhs.len() == shape_part.nterms() {
                        continue;
                    }

                    // find the associated term in the sample, trying the usual place first
                    if i < g.nterms() && g.exponents(i)[main_var] == *exp {
                        rhs.push(a.ring.neg(&a.ring.mul(&g.coefficients[i], &scale_factor)));
                    } else {
                        // find the matching term if it exists
                        for m in g.into_iter() {
                            if m.exponents[main_var] == *exp {
                                rhs.push(a.ring.neg(&a.ring.mul(m.coefficient, &scale_factor)));
                                continue 'rhs;
                            }
                        }

                        rhs.push(a.ring.zero());
                    }
                }
            }

            break (row_sample_values, samples);
        };

        Ok(a.solve_vandermonde(main_var, shape, row_sample_values, samples))
    }

    /// Construct an image in the case where no monomial in the main variable is a single term.
    /// Using Javadi's method to solve the normalization problem, we first determine the coefficients of a single monomial using
    /// Gaussian elimination. Then, we are back in the single term case and we use a Vandermonde
    /// matrix to solve for every coefficient.
    #[instrument(level = "trace", fields(%a, %b))]
    fn construct_new_image_multiple_scales(
        a: &MultivariatePolynomial<F, E>,
        b: &MultivariatePolynomial<F, E>,
        a_ldegree: E,
        b_ldegree: E,
        bounds: &mut [E],
        vars: &[usize],
        main_var: usize,
        shape: &[(MultivariatePolynomial<F, E>, E)],
    ) -> Result<MultivariatePolynomial<F, E>, GCDError> {
        let mut rng = rand::rng();

        let mut failure_count = 0;

        // store a table for variables raised to a certain power
        let mut cache = (0..a.nvars())
            .map(|i| {
                vec![
                    a.ring.zero();
                    min(
                        max(a.degree(i), b.degree(i)).to_u32() as usize + 1,
                        POW_CACHE_SIZE
                    )
                ]
            })
            .collect::<Vec<_>>();

        // sort the shape based on the number of terms in the coefficient
        let mut shape_map: Vec<_> = (0..shape.len()).collect();
        shape_map.sort_unstable_by_key(|i| shape[*i].0.nterms());

        let mut scaling_var_relations: Vec<Vec<F::Element>> = vec![];

        let max_terms = shape[*shape_map.last().unwrap()].0.nterms();

        // find a set of sample points that yield unique coefficients for every coefficient of a term in the shape
        let (row_sample_values, samples) = 'find_root_sample: loop {
            for v in &mut cache {
                for vi in v {
                    *vi = a.ring.zero();
                }
            }

            let r_orig: SmallVec<[_; INLINED_EXPONENTS]> = vars
                .iter()
                .map(|i| (*i, a.ring.sample(&mut rng, (1, MAX_RNG_PREFACTOR as i64))))
                .collect();

            let mut row_sample_values = Vec::with_capacity(shape.len()); // coefficients for the linear system

            let max_samples_needed = 2 * max_terms - 1;
            for (c, _) in shape.iter() {
                let mut row = Vec::with_capacity(c.nterms());
                let mut seen = HashSet::new();

                for t in c {
                    // evaluate each exponent
                    let mut c = a.ring.one();
                    for (n, v) in &r_orig {
                        let exp = t.exponents[*n].to_u32() as usize;
                        if exp > 0 {
                            if exp < cache[*n].len() {
                                if a.ring.is_zero(&cache[*n][exp]) {
                                    cache[*n][exp] = a.ring.pow(v, exp as u64);
                                }

                                a.ring.mul_assign(&mut c, &cache[*n][exp]);
                            } else {
                                a.ring.mul_assign(&mut c, &a.ring.pow(v, exp as u64));
                            }
                        }
                    }
                    row.push(c.clone());

                    // check if each element is unique
                    if !seen.insert(c) {
                        debug!("Duplicate element: restarting");
                        continue 'find_root_sample;
                    }
                }

                row_sample_values.push(row);
            }

            let mut samples = vec![Vec::with_capacity(max_samples_needed); shape.len()];
            let mut r = r_orig.clone();

            let a_eval = a.evaluate_exponents(&r_orig, &mut cache);
            let b_eval = b.evaluate_exponents(&r_orig, &mut cache);

            let mut a_current = Cow::Borrowed(&a_eval);
            let mut b_current = Cow::Borrowed(&b_eval);

            let mut a_poly = a.zero_with_capacity(a.degree(main_var).to_u32() as usize + 1);
            let mut b_poly = b.zero_with_capacity(b.degree(main_var).to_u32() as usize + 1);

            let mut second_index = 1;
            let mut solved_coeff = None;
            for sample_index in 0..max_samples_needed {
                if solved_coeff.is_some() && sample_index >= max_terms {
                    // we have enough samples
                    break;
                }

                // sample at r^i
                if sample_index > 0 {
                    for (c, rr) in r.iter_mut().zip(&r_orig) {
                        *c = (c.0, a.ring.mul(&c.1, &rr.1));
                    }

                    for (c, e) in a_current.to_mut().iter_mut().zip(&a_eval) {
                        a.ring.mul_assign(c, e);
                    }
                    for (c, e) in b_current.to_mut().iter_mut().zip(&b_eval) {
                        b.ring.mul_assign(c, e);
                    }
                }

                // now construct the univariate polynomials from the current evaluated monomials
                a.evaluate_using_exponents(&a_current, main_var, &mut a_poly);
                b.evaluate_using_exponents(&b_current, main_var, &mut b_poly);

                if a_poly.ldegree(main_var) != a_ldegree || b_poly.ldegree(main_var) != b_ldegree {
                    continue 'find_root_sample;
                }

                let mut g = a_poly.univariate_gcd(&b_poly);
                debug!(
                    "GCD of sample at point {:?} in main var {}: {}",
                    r, main_var, g
                );

                if g.ldegree(main_var) < bounds[main_var] {
                    // original image and form and degree bounds are unlucky
                    // change the bound and try a new prime

                    debug!("Unlucky degree bound: {} vs {}", g, bounds[main_var]);
                    bounds[main_var] = g.ldegree(main_var);
                    return Err(GCDError::BadOriginalImage);
                }

                if g.ldegree(main_var) > bounds[main_var] {
                    failure_count += 1;
                    if failure_count > 2 {
                        // p is likely unlucky
                        debug!(
                            "Bad current image: gcd({},{}) mod {} under {:?} = {}",
                            a, b, a.ring, r, g
                        );
                        return Err(GCDError::BadCurrentImage);
                    }
                    debug!("Degree too high");
                    continue 'find_root_sample;
                }

                // check if all the monomials of the image appear in the shape
                // if not, the original shape is bad
                for m in g.into_iter() {
                    if shape.iter().all(|(_, pow)| *pow != m.exponents[main_var]) {
                        debug!("Bad shape: terms missing");
                        return Err(GCDError::BadOriginalImage);
                    }
                }

                // set the coefficient of the scaling term in the gcd to 1
                let (_, d) = &shape[shape_map[0]];
                let mut found = false;
                for t in &g {
                    if t.exponents[main_var] == *d {
                        let scale_factor = g.ring.inv(t.coefficient);
                        g = g.mul_coeff(scale_factor);
                        found = true;
                        break;
                    }
                }

                if !found {
                    // the scaling term is missing, so the sample point is bad
                    debug!("Bad sample point: scaling term missing");
                    // TODO: check if this happen a number of times in a row
                    // as the prime may be too small to generate n samples that
                    // all contain the scaling term
                    continue 'find_root_sample;
                }

                // construct the right-hand side
                'rhs: for (i, (rhs, (shape_part, exp))) in samples.iter_mut().zip(shape).enumerate()
                {
                    // we may not need all terms
                    if solved_coeff.is_some() && rhs.len() == shape_part.nterms() {
                        continue;
                    }

                    // find the associated term in the sample, trying the usual place first
                    if i < g.nterms() && g.exponents(i)[main_var] == *exp {
                        rhs.push(g.coefficients[i].clone());
                    } else {
                        // find the matching term if it exists
                        for m in g.into_iter() {
                            if m.exponents[main_var] == *exp {
                                rhs.push(m.coefficient.clone());
                                continue 'rhs;
                            }
                        }

                        rhs.push(a.ring.zero());
                    }
                }

                // see if we have collected enough samples to solve for the scaling factor
                while solved_coeff.is_none() {
                    // try to solve the system!
                    let vars_scale = shape[shape_map[0]].0.nterms() - 1;
                    let vars_second = shape[shape_map[second_index]].0.nterms();
                    let samples_needed = vars_scale + vars_second;
                    let rows = samples_needed + scaling_var_relations.len();

                    if sample_index + 1 < samples_needed {
                        break; // obtain more samples
                    }

                    let mut gfm = Vec::with_capacity(rows * samples_needed);
                    let mut new_rhs = Vec::with_capacity(rows);

                    for sample_index in 0..samples_needed {
                        let rhs_sec = &samples[shape_map[second_index]][sample_index];
                        let row_eval_sec = &row_sample_values[shape_map[second_index]];
                        let row_eval_first = &row_sample_values[shape_map[0]];

                        // assume first constant is 1, which will form the rhs of our equation
                        let actual_rhs = a.ring.mul(
                            rhs_sec,
                            &a.ring.pow(&row_eval_first[0], sample_index as u64 + 1),
                        );

                        for aa in row_eval_sec {
                            gfm.push(a.ring.pow(aa, sample_index as u64 + 1));
                        }

                        // place the scaling term variables at the end
                        for aa in &row_eval_first[1..] {
                            gfm.push(
                                a.ring.neg(
                                    &a.ring
                                        .mul(rhs_sec, &a.ring.pow(aa, sample_index as u64 + 1)),
                                ),
                            );
                        }

                        new_rhs.push(actual_rhs);
                    }

                    // add extra relations between the scaling term variables coming from previous tries
                    // that yielded underdetermined systems
                    for extra_relations in &scaling_var_relations {
                        for _ in 0..vars_second {
                            gfm.push(a.ring.zero());
                        }

                        for v in &extra_relations[..vars_scale] {
                            gfm.push(v.clone());
                        }
                        new_rhs.push(extra_relations.last().unwrap().clone());
                    }

                    let m = Matrix::from_linear(
                        gfm,
                        rows as u32,
                        samples_needed as u32,
                        a.ring.clone(),
                    )
                    .unwrap();
                    let rhs = Matrix::new_vec(new_rhs, a.ring.clone());

                    match m.solve(&rhs) {
                        Ok(r) => {
                            debug!(
                                "Solved with {} and {} term",
                                shape[shape_map[0]].0, shape[shape_map[second_index]].0
                            );

                            let mut r = r.into_vec();
                            r.drain(0..vars_second);
                            solved_coeff = Some(r);
                        }
                        Err(MatrixError::Underdetermined {
                            row_reduced_augmented_matrix,
                            ..
                        }) => {
                            // extract relations between the variables in the scaling term from the row reduced augmented matrix

                            debug!(
                                "Underdetermined system {} and {} term; row reduction={}, rhs={}",
                                shape[shape_map[0]].0,
                                shape[shape_map[second_index]].0,
                                row_reduced_augmented_matrix,
                                rhs
                            );

                            for x in row_reduced_augmented_matrix.row_iter() {
                                if x[..vars_second].iter().all(|x| a.ring.is_zero(x))
                                    && x.iter().any(|y| !a.ring.is_zero(y))
                                {
                                    scaling_var_relations.push(x[vars_second..].to_vec());
                                }
                            }

                            second_index += 1;
                            if second_index == shape.len() {
                                // the system remains underdetermined, that means the shape is bad
                                debug!(
                                    "Could not determine monomial scaling due to a bad shape\na={}\nb={}\na_ldegree={}, b_ldegree={}\nbounds={:?}, vars={:?}, main_var={},\nmat={}\nrhs={},\nshape=",
                                    a,
                                    b,
                                    a_ldegree,
                                    b_ldegree,
                                    bounds,
                                    vars,
                                    main_var,
                                    row_reduced_augmented_matrix,
                                    rhs
                                );
                                for s in shape {
                                    debug!("\t({}, {})", s.0, s.1);
                                }

                                return Err(GCDError::BadOriginalImage);
                            }
                        }
                        Err(MatrixError::Inconsistent) => {
                            debug!("Inconsistent system: bad shape");
                            return Err(GCDError::BadOriginalImage);
                        }
                        Err(
                            MatrixError::NotSquare
                            | MatrixError::ShapeMismatch
                            | MatrixError::RightHandSideIsNotVector
                            | MatrixError::Singular
                            | MatrixError::ResultNotInDomain,
                        ) => {
                            unreachable!()
                        }
                    }
                }
            }

            if let Some(r) = solved_coeff {
                // evaluate the scaling term for every sample
                let mut lcoeff_cache = Vec::with_capacity(max_terms);
                for sample_index in 0..max_terms {
                    let row_eval_first = &row_sample_values[shape_map[0]];
                    let mut scaling_factor =
                        a.ring.pow(&row_eval_first[0], sample_index as u64 + 1); // coeff eval is 1
                    for (exp_eval, coeff_eval) in
                        row_sample_values[shape_map[0]][1..].iter().zip(&r)
                    {
                        a.ring.add_mul_assign(
                            &mut scaling_factor,
                            coeff_eval,
                            &a.ring.pow(exp_eval, sample_index as u64 + 1),
                        );
                    }

                    debug!(
                        "Scaling fac {}: {}",
                        sample_index,
                        a.ring.printer(&scaling_factor)
                    );
                    lcoeff_cache.push(scaling_factor);
                }

                for ((c, _), rhs) in shape.iter().zip(&mut samples) {
                    rhs.truncate(c.nterms()); // drop unneeded samples
                    for (r, scale) in rhs.iter_mut().zip(&lcoeff_cache) {
                        a.ring.mul_assign(r, scale);
                    }
                }
            } else {
                debug!(
                    "Could not solve the system with just 2 terms: a={}, b={}",
                    a, b
                );
            }

            break (row_sample_values, samples);
        };

        Ok(a.solve_vandermonde(main_var, shape, row_sample_values, samples))
    }
}

impl<F: Field + PolynomialGCD<E>, E: PositiveExponent> MultivariatePolynomial<F, E> {
    /// Compute the gcd shape of two polynomials in a finite field by filling in random
    /// numbers.
    #[instrument(level = "debug", skip_all)]
    fn gcd_shape_modular(
        a: &Self,
        b: &Self,
        vars: &[usize],         // variables
        bounds: &mut [E],       // degree bounds
        tight_bounds: &mut [E], // tighter degree bounds
    ) -> Option<Self> {
        let lastvar = *vars.last().unwrap();

        // if we are in the univariate case, return the univariate gcd
        // TODO: this is a modification of the algorithm!
        if vars.len() == 1 {
            let gg = a.univariate_gcd(b);
            if gg.degree(vars[0]) > bounds[vars[0]] {
                debug!(
                    "Unexpectedly high GCD bound: {} vs {}",
                    gg.degree(vars[0]),
                    bounds[vars[0]]
                );
                return None;
            }
            bounds[vars[0]] = gg.degree(vars[0]); // update degree bound
            return Some(gg);
        }

        // the gcd of the content in the last variable should be 1
        let c = a.multivariate_content_gcd(b, lastvar);
        if !c.is_one() {
            debug!("Content in last variable is not 1, but {}", c);
            // TODO: we assume that a content of -1 is also allowed
            // like in the special case gcd_(-x0*x1,-x0-x0*x1)
            if c.nterms() != 1 || c.coefficients[0] != a.ring.neg(&a.ring.one()) {
                return None;
            }
        }

        let gamma = a
            .lcoeff_last_varorder(vars)
            .univariate_gcd(&b.lcoeff_last_varorder(vars));

        let mut rng = rand::rng();

        let mut failure_count = 0;

        'newfirstnum: loop {
            // if we had two failures, it may be that the tight degree bound
            // was too tight due to an unfortunate prime/evaluation, so we relax it
            if failure_count == 2 {
                debug!(
                    "Changing tight bound for x{} from {} to {}",
                    lastvar, tight_bounds[lastvar], bounds[lastvar]
                );
                tight_bounds[lastvar] = bounds[lastvar];
            }
            failure_count += 1;

            if let Some(size) = a.ring.size()
                && failure_count * 2 > size
            {
                debug!("Cannot find unique sampling points: prime field is likely too small");
                return None;
            }

            let mut sample_fail_count = 0i64;
            let v = loop {
                let r = a.ring.sample(&mut rng, (1, MAX_RNG_PREFACTOR as i64));
                if !gamma.replace(lastvar, &r).is_zero() {
                    break r;
                }

                sample_fail_count += 1;
                if let Some(size) = a.ring.size()
                    && sample_fail_count * 2 > size
                {
                    debug!("Cannot find unique sampling points: prime field is likely too small");
                    continue 'newfirstnum;
                }
            };

            debug!("Chosen variable: {}", a.ring.printer(&v));
            let av = a.replace(lastvar, &v);
            let bv = b.replace(lastvar, &v);

            // performance dense reconstruction
            let mut gv = if vars.len() > 2 {
                MultivariatePolynomial::gcd_shape_modular(
                    &av,
                    &bv,
                    &vars[..vars.len() - 1],
                    bounds,
                    tight_bounds,
                )?
            } else {
                let gg = av.univariate_gcd(&bv);
                if gg.degree(vars[0]) > bounds[vars[0]] {
                    debug!(
                        "Unexpectedly high GCD bound: {} vs {}",
                        gg.degree(vars[0]),
                        bounds[vars[0]]
                    );
                    return None;
                }
                bounds[vars[0]] = gg.degree(vars[0]); // update degree bound
                gg
            };

            debug!(
                "GCD shape suggestion for sample point {} and gamma {}: {}",
                a.ring.printer(&v),
                gamma,
                gv
            );

            // construct a new assumed form
            let gfu = gv.to_univariate_polynomial_list(vars[0]);

            // find a coefficient of x1 in gg that is a monomial (single scaling)
            let mut single_scale = None;
            let mut nx = 0; // count the minimal number of samples needed
            for (i, (c, _e)) in gfu.iter().enumerate() {
                if c.nterms() > nx {
                    nx = c.nterms();
                }
                if c.nterms() == 1 {
                    single_scale = Some(i);
                }
            }

            // In the case of multiple scaling, each sample adds an
            // additional unknown, except for the first
            if single_scale.is_none() {
                let mut nx1 = (gv.nterms() - 1) / (gfu.len() - 1);
                if (gv.nterms() - 1) % (gfu.len() - 1) != 0 {
                    nx1 += 1;
                }
                if nx < nx1 {
                    nx = nx1;
                }
                debug!("Multiple scaling case: sample {} times", nx);
            }

            let mut lc = gv.lcoeff_varorder(vars);

            let mut gseq = vec![
                gv.clone().mul_coeff(
                    gamma
                        .ring
                        .div(&gamma.replace(lastvar, &v).coefficients[0], &lc),
                ),
            ];
            let mut vseq = vec![v];

            // sparse reconstruction

            'newnum: loop {
                if gseq.len()
                    == (tight_bounds[lastvar].to_u32() + gamma.ldegree_max().to_u32() + 1) as usize
                {
                    break;
                }

                let v = loop {
                    let v = a.ring.sample(&mut rng, (1, MAX_RNG_PREFACTOR as i64));
                    if !gamma.replace(lastvar, &v).is_zero() {
                        // we need unique sampling points
                        if !vseq.contains(&v) {
                            break v;
                        }
                    }

                    sample_fail_count += 1;
                    if let Some(size) = a.ring.size()
                        && sample_fail_count * 2 > size
                    {
                        debug!(
                            "Cannot find unique sampling points: prime field is likely too small"
                        );
                        continue 'newfirstnum;
                    }
                };

                debug!("Chosen sample: {}", a.ring.printer(&v));

                let av = a.replace(lastvar, &v);
                let bv = b.replace(lastvar, &v);

                let rec = if let Some(single_scale) = single_scale {
                    Self::construct_new_image_single_scale(
                        &av,
                        &bv,
                        av.degree(vars[0]),
                        bv.degree(vars[0]),
                        bounds,
                        single_scale,
                        &vars[1..vars.len() - 1],
                        vars[0],
                        &gfu,
                    )
                } else {
                    Self::construct_new_image_multiple_scales(
                        &av,
                        &bv,
                        // NOTE: different from paper where they use a.degree(..)
                        // it could be that the degree in av is lower than that of a
                        // which means the sampling will never terminate
                        av.degree(vars[0]),
                        bv.degree(vars[0]),
                        bounds,
                        &vars[1..vars.len() - 1],
                        vars[0],
                        &gfu,
                    )
                };

                match rec {
                    Ok(r) => {
                        gv = r;
                    }
                    Err(GCDError::BadOriginalImage) => {
                        debug!("Bad original image");
                        continue 'newfirstnum;
                    }
                    Err(GCDError::BadCurrentImage) => {
                        debug!("Bad current image");
                        sample_fail_count += 1;

                        if let Some(size) = a.ring.size()
                            && sample_fail_count * 2 > size
                        {
                            debug!("Too many bad current images: prime field is likely too small");
                            continue 'newfirstnum;
                        }

                        continue 'newnum;
                    }
                }

                lc = gv.lcoeff_varorder(vars);

                gseq.push(
                    gv.clone().mul_coeff(
                        gamma
                            .ring
                            .div(&gamma.replace(lastvar, &v).coefficients[0], &lc),
                    ),
                );
                vseq.push(v);
            }

            // use interpolation to construct x_n dependence
            let mut gc = Self::newton_interpolation(&vseq, &gseq, lastvar);
            debug!("Interpolated: {}", gc);

            // remove content in x_n (wrt all other variables)
            let cont = gc.multivariate_content(lastvar);
            if !cont.is_one() {
                debug!("Removing content in x{}: {}", lastvar, cont);
                gc = gc.try_div(&cont).unwrap();
            }

            // do a probabilistic division test
            let (g1, a1, b1) = loop {
                // store a table for variables raised to a certain power
                let mut cache = (0..a.nvars())
                    .map(|i| {
                        vec![
                            a.ring.zero();
                            min(
                                max(a.degree(i), b.degree(i)).to_u32() as usize + 1,
                                POW_CACHE_SIZE
                            )
                        ]
                    })
                    .collect::<Vec<_>>();

                let r: Vec<_> = vars
                    .iter()
                    .skip(1)
                    .map(|i| (*i, a.ring.sample(&mut rng, (1, MAX_RNG_PREFACTOR as i64))))
                    .collect();

                let g1 = gc.replace_except(vars[0], &r, &mut cache);

                if g1.ldegree(vars[0]) == gc.degree(vars[0]) {
                    let a1 = a.replace_except(vars[0], &r, &mut cache);
                    let b1 = b.replace_except(vars[0], &r, &mut cache);
                    break (g1, a1, b1);
                }
            };

            if g1.is_one() || (a1.try_div(&g1).is_some() && b1.try_div(&g1).is_some()) {
                return Some(gc);
            }

            // if the gcd is bad, we had a bad number
            debug!(
                "Division test failed: gcd may be bad or probabilistic division test is unlucky: a1 {} b1 {} g1 {}",
                a1, b1, g1
            );
        }
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent> MultivariatePolynomial<R, E> {
    /// Get the content of a multivariate polynomial viewed as a
    /// univariate polynomial in `x`.
    pub fn univariate_content(&self, x: usize) -> MultivariatePolynomial<R, E> {
        let a = self.to_univariate_polynomial_list(x);

        let mut f = Vec::with_capacity(a.len());
        for (c, _) in a {
            f.push(c);
        }

        PolynomialGCD::gcd_multiple(f)
    }

    /// Get the content of a multivariate polynomial viewed as a
    /// multivariate polynomial in all variables except `x`.
    pub fn multivariate_content(&self, x: usize) -> MultivariatePolynomial<R, E> {
        let af = self.to_multivariate_polynomial_list(&[x], false);
        PolynomialGCD::gcd_multiple(af.into_values().collect())
    }

    /// Compute the gcd of the univariate content in `x`.
    pub fn univariate_content_gcd(
        &self,
        b: &MultivariatePolynomial<R, E>,
        x: usize,
    ) -> MultivariatePolynomial<R, E> {
        let af = self.to_univariate_polynomial_list(x);
        let bf = b.to_univariate_polynomial_list(x);

        let mut f = Vec::with_capacity(af.len() + bf.len());
        for (c, _) in af.into_iter().chain(bf.into_iter()) {
            f.push(c);
        }

        PolynomialGCD::gcd_multiple(f)
    }

    /// Get the GCD of the contents of a polynomial and another one,
    /// viewed as a multivariate polynomial in all variables except `x`.
    pub fn multivariate_content_gcd(
        &self,
        b: &MultivariatePolynomial<R, E>,
        x: usize,
    ) -> MultivariatePolynomial<R, E> {
        let af = self.to_multivariate_polynomial_list(&[x], false);
        let bf = b.to_multivariate_polynomial_list(&[x], false);

        let f = af.into_values().chain(bf.into_values()).collect();

        PolynomialGCD::gcd_multiple(f)
    }

    /// Apply a GCD repeatedly to a list of polynomials.
    #[inline(always)]
    pub fn repeated_gcd(mut f: Vec<MultivariatePolynomial<R, E>>) -> MultivariatePolynomial<R, E> {
        if f.len() == 1 {
            return f.swap_remove(0);
        }

        if f.len() == 2 {
            return f[0].gcd(&f[1]);
        }

        f.sort_unstable_by_key(|p| p.nterms());

        let mut gcd = f.pop().unwrap();
        for p in f {
            if R::one_is_gcd_unit() && gcd.is_one() {
                return gcd;
            }

            gcd = gcd.gcd(&p);
        }
        gcd
    }

    /// Compute a standard GCD-free basis. The input should not
    /// contain 0 or units.
    pub fn gcd_free_basis(mut polys: Vec<Self>) -> Vec<Self> {
        let mut i = 0;
        while i + 1 < polys.len() {
            if polys[i].is_one() {
                i += 1;
                continue;
            }

            let mut j = i + 1;
            while j < polys.len() {
                if polys[j].is_one() {
                    j += 1;
                    continue;
                }

                let g = polys[i].gcd(&polys[j]);
                if !g.is_one() {
                    polys[i] = &polys[i] / &g;
                    polys[j] = &polys[j] / &g;
                    polys.push(g);
                }

                j += 1;
            }

            i += 1;
        }

        polys.retain(|p| !p.is_one());
        polys
    }

    /// Compute the GCD for simple cases.
    #[inline(always)]
    fn simple_gcd(&self, b: &MultivariatePolynomial<R, E>) -> Option<MultivariatePolynomial<R, E>> {
        if self == b {
            return Some(self.clone());
        }

        if self.is_zero() {
            return Some(b.clone());
        }
        if b.is_zero() {
            return Some(self.clone());
        }

        if self.is_one() {
            return Some(self.clone());
        }

        if b.is_one() {
            return Some(b.clone());
        }

        if self.is_constant() {
            let mut gcd = self.coefficients[0].clone();
            for c in &b.coefficients {
                gcd = self.ring.gcd(&gcd, c);
                if R::one_is_gcd_unit() && self.ring.is_one(&gcd) {
                    break;
                }
            }
            return Some(self.constant(gcd));
        }

        if b.is_constant() {
            let mut gcd = b.coefficients[0].clone();
            for c in &self.coefficients {
                gcd = self.ring.gcd(&gcd, c);
                if R::one_is_gcd_unit() && self.ring.is_one(&gcd) {
                    break;
                }
            }
            return Some(self.constant(gcd));
        }

        None
    }

    /// Compute the gcd of two multivariate polynomials.
    #[instrument(skip_all)]
    pub fn gcd(&self, b: &MultivariatePolynomial<R, E>) -> MultivariatePolynomial<R, E> {
        debug!("gcd of {} and {}", self, b);

        if let Some(g) = self.simple_gcd(b) {
            debug!("Simple {} ", g);
            return g;
        }

        // a and b are only copied when needed
        let mut a = Cow::Borrowed(self);
        let mut b = Cow::Borrowed(b);

        if self.variables != b.variables {
            a.to_mut().unify_variables(b.to_mut());
        }

        // determine the maximum shared power of every variable
        let mut shared_degree: SmallVec<[E; INLINED_EXPONENTS]> = a.exponents(0).into();
        for p in [&a, &b] {
            for e in p.exponents_iter() {
                for (md, v) in shared_degree.iter_mut().zip(e) {
                    *md = (*md).min(*v);
                }
            }
        }

        // divide out the common factors
        if shared_degree.iter().any(|d| *d != E::zero()) {
            let aa = a.to_mut();
            for e in aa.exponents_iter_mut() {
                for (v, d) in e.iter_mut().zip(&shared_degree) {
                    *v = *v - *d;
                }
            }

            let bb = b.to_mut();
            for e in bb.exponents_iter_mut() {
                for (v, d) in e.iter_mut().zip(&shared_degree) {
                    *v = *v - *d;
                }
            }
        };

        // remove superfluous shifts: all variables should occur with exponent 1
        for v in 0..a.nvars() {
            let exp = a.degree_bounds(v).0;
            if exp > E::zero() {
                let pp = a.to_mut();
                for e in pp.exponents_iter_mut() {
                    e[v] = e[v] - exp;
                }
            }

            let exp = b.degree_bounds(v).0;
            if exp > E::zero() {
                let pp = b.to_mut();
                for e in pp.exponents_iter_mut() {
                    e[v] = e[v] - exp;
                }
            }
        }

        let mut base_degree: SmallVec<[Option<E>; INLINED_EXPONENTS]> = smallvec![None; a.nvars()];

        if let Some(g) = MultivariatePolynomial::simple_gcd(&a, &b) {
            return rescale_gcd(g, &shared_degree, &base_degree, &a.constant(a.ring.one()));
        }

        // check if the polynomial are functions of x^n, n > 1
        for p in [&a, &b] {
            for t in p.into_iter() {
                for (md, v) in base_degree.iter_mut().zip(t.exponents) {
                    if !v.is_zero() {
                        if let Some(mm) = md.as_mut() {
                            if *mm != E::one() {
                                *mm = mm.gcd(v);
                            }
                        } else {
                            *md = Some(*v);
                        }
                    }
                }
            }
        }

        // rename x^base_deg to x
        if base_degree
            .iter()
            .any(|d| d.is_some() && d.unwrap() > E::one())
        {
            let aa = a.to_mut();
            for e in aa.exponents_iter_mut() {
                for (v, d) in e.iter_mut().zip(&base_degree) {
                    if let Some(d) = d {
                        *v = *v / *d;
                    }
                }
            }

            let bb = b.to_mut();
            for e in bb.exponents_iter_mut() {
                for (v, d) in e.iter_mut().zip(&base_degree) {
                    if let Some(d) = d {
                        *v = *v / *d;
                    }
                }
            }
        }

        /// Undo simplifications made to the input polynomials and normalize the gcd.
        #[inline(always)]
        fn rescale_gcd<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent>(
            mut g: MultivariatePolynomial<R, E>,
            shared_degree: &[E],
            base_degree: &[Option<E>],
            content: &MultivariatePolynomial<R, E>,
        ) -> MultivariatePolynomial<R, E> {
            if !content.is_one() {
                g = g * content;
            }

            if shared_degree.iter().any(|d| *d > E::from_u32(0))
                || base_degree
                    .iter()
                    .any(|d| d.map(|bd| bd > E::one()).unwrap_or(false))
            {
                for e in g.exponents_iter_mut() {
                    for ((v, d), s) in e.iter_mut().zip(base_degree).zip(shared_degree) {
                        if let Some(d) = d {
                            *v = *v * *d;
                        }

                        *v += *s;
                    }
                }
            }

            PolynomialGCD::normalize(g)
        }

        if let Some(gcd) = PolynomialGCD::heuristic_gcd(&a, &b) {
            debug!("Heuristic gcd succeeded: {}", gcd.0);
            return rescale_gcd(
                gcd.0,
                &shared_degree,
                &base_degree,
                &a.constant(a.ring.one()),
            );
        }

        // store which variables appear in which expression
        let mut scratch: SmallVec<[i32; INLINED_EXPONENTS]> = smallvec![0i32; a.nvars()];
        for (p, inc) in [(&a, 1), (&b, 2)] {
            for t in p.into_iter() {
                for (e, ee) in scratch.iter_mut().zip(t.exponents) {
                    if !ee.is_zero() {
                        *e |= inc;
                    }
                }
            }
        }

        if a == b {
            debug!("Equal {} ", a);
            return rescale_gcd(a.into_owned(), &shared_degree, &base_degree, &b.one());
        }

        // compute the gcd efficiently if some variables do not occur in both
        // polynomials
        if scratch.iter().any(|x| *x > 0 && *x < 3) {
            let inca: SmallVec<[_; INLINED_EXPONENTS]> = scratch
                .iter()
                .enumerate()
                .filter_map(|(i, v)| if *v == 1 || *v == 3 { Some(i) } else { None })
                .collect();

            let incb: SmallVec<[_; INLINED_EXPONENTS]> = scratch
                .iter()
                .enumerate()
                .filter_map(|(i, v)| if *v == 2 || *v == 3 { Some(i) } else { None })
                .collect();

            // extract the variables of b in the coefficient of a and vice versa
            let a1 = a.to_multivariate_polynomial_list(&incb, false);
            let b1 = b.to_multivariate_polynomial_list(&inca, false);

            let f = a1.into_values().chain(b1.into_values()).collect();

            return rescale_gcd(
                PolynomialGCD::gcd_multiple(f),
                &shared_degree,
                &base_degree,
                &a.one(),
            );
        }

        // try if b divides a or vice versa, doing a heuristical length check first
        if a.nterms() >= b.nterms() && a.try_div(&b).is_some() {
            return rescale_gcd(b.into_owned(), &shared_degree, &base_degree, &a.one());
        }
        if a.nterms() <= b.nterms() && b.try_div(&a).is_some() {
            return rescale_gcd(a.into_owned(), &shared_degree, &base_degree, &b.one());
        }

        // check if the polynomial is linear in a variable and compute the gcd using the univariate content
        for (p1, p2) in [(&a, &b), (&b, &a)] {
            if let Some(var) = (0..p1.nvars()).find(|v| p1.degree(*v) == E::one()) {
                let mut cont = p1.univariate_content(var);

                let p1_prim = p1.as_ref() / &cont;

                if !cont.is_one() || !R::one_is_gcd_unit() {
                    let cont_p2 = p2.univariate_content(var);
                    cont = cont.gcd(&cont_p2);
                }

                if p2.try_div(&p1_prim).is_some() {
                    return rescale_gcd(p1_prim, &shared_degree, &base_degree, &cont);
                } else {
                    return rescale_gcd(
                        cont,
                        &shared_degree,
                        &base_degree,
                        &p1.constant(p1.ring.one()),
                    );
                }
            }
        }

        let mut vars: SmallVec<[_; INLINED_EXPONENTS]> = scratch
            .iter()
            .enumerate()
            .filter_map(|(i, v)| if *v == 3 { Some(i) } else { None })
            .collect();

        // find upper bounds for all variables
        let mut bounds = R::get_gcd_var_bounds(&a, &b, &vars);

        // if all bounds are 0, the gcd is a constant
        if bounds.iter().all(|x| x.is_zero()) {
            return rescale_gcd(
                a.constant(a.ring.gcd(&a.content(), &b.content())),
                &shared_degree,
                &base_degree,
                &a.one(),
            );
        }

        // if some variables do not appear in the gcd, split the polynomials in these variables
        if bounds.iter().any(|x| x.is_zero()) {
            let zero_bound: SmallVec<[_; INLINED_EXPONENTS]> = bounds
                .iter()
                .enumerate()
                .filter_map(|(i, v)| {
                    if *v == E::zero() && a.degree(i) > E::zero() {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect();

            if !zero_bound.is_empty() {
                let a1 = a.to_multivariate_polynomial_list(&zero_bound, true);
                let b1 = b.to_multivariate_polynomial_list(&zero_bound, true);

                let f = a1.into_values().chain(b1.into_values()).collect();

                return rescale_gcd(
                    PolynomialGCD::gcd_multiple(f),
                    &shared_degree,
                    &base_degree,
                    &a.one(),
                );
            }
        }

        // Determine a good variable ordering based on the estimated degree (decreasing) in the gcd.
        // If it is different from the input, make a copy and rearrange so that the
        // polynomials do not have to be sorted after filling in variables.
        vars.sort_by(|&i, &j| bounds[j].cmp(&bounds[i]));

        // strip the gcd of the univariate contents wrt the new first variable
        let content = if vars.len() > 1 {
            debug!("Starting univariate content computation in {}", vars[0]);
            let content = a.univariate_content_gcd(&b, vars[0]);
            debug!("GCD of content: {}", content);

            if !content.is_one() {
                a = Cow::Owned(a.as_ref() / &content);
                b = Cow::Owned(b.as_ref() / &content);
            }

            // TODO: update GCD degree bounds?

            // even if variables got removed, benchmarks show that it is not
            // worth it do restart the gcd computation
            content
        } else {
            // get the integer content for univariate polynomials
            let uca = a.content();
            let ucb = b.content();
            let content = a.ring.gcd(&a.content(), &b.content());
            let p = a.zero_with_capacity(1);

            if !a.ring.is_one(&uca) {
                a = Cow::Owned(a.into_owned().div_coeff(&uca));
            }
            if !a.ring.is_one(&ucb) {
                b = Cow::Owned(b.into_owned().div_coeff(&ucb));
            }

            p.add_constant(content)
        };

        let rearrange = vars.len() > 1 && vars.windows(2).any(|s| s[0] > s[1]);
        if rearrange {
            debug!("Rearranging variables with map: {:?}", vars);
            a = Cow::Owned(a.rearrange_impl(&vars, false, false));
            b = Cow::Owned(b.rearrange_impl(&vars, false, false));

            let mut newbounds: SmallVec<[_; INLINED_EXPONENTS]> =
                smallvec![E::zero(); bounds.len()];
            for x in 0..vars.len() {
                newbounds[x] = bounds[vars[x]];
            }
            bounds = newbounds;
        }

        let mut g = PolynomialGCD::gcd(
            &a,
            &b,
            &if rearrange {
                Cow::Owned((0..vars.len()).collect::<SmallVec<[usize; INLINED_EXPONENTS]>>())
            } else {
                Cow::Borrowed(&vars)
            },
            &mut bounds,
        );

        if rearrange {
            g = g.rearrange_impl(&vars, true, false);
        }

        rescale_gcd(g, &shared_degree, &base_degree, &content)
    }
}

/// An error that can occur during the heuristic GCD algorithm.
#[derive(Debug)]
pub enum HeuristicGCDError {
    MaxSizeExceeded,
    BadReconstruction,
}

impl<E: PositiveExponent> MultivariatePolynomial<IntegerRing, E> {
    /// Perform a heuristic GCD algorithm.
    #[instrument(level = "debug", skip_all)]
    pub fn heuristic_gcd(&self, b: &Self) -> Result<(Self, Self, Self), HeuristicGCDError> {
        fn interpolate<E: PositiveExponent>(
            mut gamma: MultivariatePolynomial<IntegerRing, E>,
            var: usize,
            xi: &Integer,
        ) -> MultivariatePolynomial<IntegerRing, E> {
            let mut g = gamma.zero();
            let mut i = 0;
            let xi_half = xi / &Integer::Single(2);
            while !gamma.is_zero() {
                // create xi-adic representation using the symmetric modulus
                let mut g_i = gamma.zero_with_capacity(gamma.nterms());
                for m in &gamma {
                    let mut c = Z.quot_rem(m.coefficient, xi).1;

                    if c > xi_half {
                        c -= xi;
                    }

                    if !c.is_zero() {
                        g_i.append_monomial(c, m.exponents);
                    }
                }

                for c in &mut g_i.coefficients {
                    *c = Z.quot_rem(c, xi).1;

                    if *c > xi_half {
                        *c -= xi;
                    }
                }

                // multiply with var^i
                let mut g_i_2 = g_i.clone();
                let nvars = g_i_2.nvars();
                for x in g_i_2.exponents.chunks_mut(nvars) {
                    x[var] = E::from_u32(i);
                }

                g = g.add(g_i_2);

                gamma = (gamma - g_i).div_coeff(xi);
                i += 1;
            }
            g
        }

        debug!("a={}; b={}", self, b);

        // do integer GCD
        let content_gcd = self.ring.gcd(&self.content(), &b.content());

        debug!("content={}", content_gcd);

        let mut a = Cow::Borrowed(self);
        let mut b = Cow::Borrowed(b);

        if !a.ring.is_one(&content_gcd) {
            a = Cow::Owned(a.into_owned().div_coeff(&content_gcd));
            b = Cow::Owned(b.into_owned().div_coeff(&content_gcd));
        }

        debug!("a_red={}; b_red={}", a, b);

        if let Some(var) =
            (0..a.nvars()).find(|x| a.degree(*x) > E::zero() && b.degree(*x) > E::zero())
        {
            let max_a = a
                .coefficients
                .iter()
                .max_by(|x1, x2| x1.abs_cmp(x2))
                .unwrap_or(&Integer::Single(0));

            let max_b = b
                .coefficients
                .iter()
                .max_by(|x1, x2| x1.abs_cmp(x2))
                .unwrap_or(&Integer::Single(0));

            let min = if max_a.abs_cmp(max_b) == Ordering::Greater {
                max_b.abs()
            } else {
                max_a.abs()
            };

            let mut xi = &(&min * &Integer::Single(2)) + &Integer::Single(29);

            for retry in 0..6 {
                debug!("round {}, xi={}", retry, xi);
                match &xi * &Integer::Single(a.degree(var).max(b.degree(var)).to_u32() as i64) {
                    Integer::Single(_) => {}
                    Integer::Double(_) => {}
                    Integer::Large(r) => {
                        if r.as_limbs().len() > 4 {
                            debug!("big num {}", r);
                            return Err(HeuristicGCDError::MaxSizeExceeded);
                        }
                    }
                }

                let aa = a.replace(var, &xi);
                let bb = b.replace(var, &xi);

                let (gamma, co_fac_p, co_fac_q) = match aa.heuristic_gcd(&bb) {
                    Ok(x) => x,
                    Err(HeuristicGCDError::MaxSizeExceeded) => {
                        return Err(HeuristicGCDError::MaxSizeExceeded);
                    }
                    Err(HeuristicGCDError::BadReconstruction) => {
                        xi = Z
                            .quot_rem(&(&xi * &Integer::Single(73794)), &Integer::Single(27011))
                            .0;
                        continue;
                    }
                };

                debug!("gamma={}", gamma);

                let g = interpolate(gamma, var, &xi);
                let g_cont = g.content();

                let gc = g.div_coeff(&g_cont);

                if let Some(q) = a.try_div(&gc)
                    && let Some(q1) = b.try_div(&gc)
                {
                    debug!("match {} {}", q, q1);
                    return Ok((gc.mul_coeff(content_gcd), q, q1));
                }

                debug!("co_fac_p {}", co_fac_p);

                if !co_fac_p.is_zero() {
                    let a_co_fac = interpolate(co_fac_p, var, &xi);

                    if let Some(q) = a.try_div(&a_co_fac)
                        && let Some(q1) = b.try_div(&q)
                    {
                        return Ok((q.mul_coeff(content_gcd), a_co_fac, q1));
                    }
                }

                if !co_fac_q.is_zero() {
                    let b_co_fac = interpolate(co_fac_q, var, &xi);
                    debug!("cofac b {}", b_co_fac);

                    if let Some(q) = b.try_div(&b_co_fac)
                        && let Some(q1) = a.try_div(&q)
                    {
                        return Ok((q.mul_coeff(content_gcd), q1, b_co_fac));
                    }
                }

                xi = Z
                    .quot_rem(&(&xi * &Integer::Single(73794)), &Integer::Single(27011))
                    .0;
            }

            Err(HeuristicGCDError::BadReconstruction)
        } else {
            Ok((self.constant(content_gcd), a.into_owned(), b.into_owned()))
        }
    }

    /// Compute the gcd of multiple polynomials efficiently.
    /// `gcd(f0,f1,f2,...)=gcd(f0,f1+k2*f(2)+k3*f(3))`
    /// with high likelihood.
    pub fn gcd_multiple(
        mut f: Vec<MultivariatePolynomial<IntegerRing, E>>,
    ) -> MultivariatePolynomial<IntegerRing, E> {
        assert!(!f.is_empty());

        let mut prime_index = 1; // skip prime 2
        let mut loop_counter = 0;
        loop {
            if f.len() == 1 {
                return f.swap_remove(0);
            }

            if f.len() == 2 {
                return f[0].gcd(&f[1]);
            }

            // check if any entry is a number, as the gcd is then the gcd of the contents
            if let Some(n) = f.iter().find(|x| x.is_constant()) {
                let mut gcd = n.content();
                for x in f.iter() {
                    if x.ring.is_one(&gcd) {
                        break;
                    }

                    gcd = x.ring.gcd(&gcd, &x.content());
                }
                return n.constant(gcd);
            }

            // take the smallest element
            let index_smallest = f
                .iter()
                .enumerate()
                .min_by_key(|(_, v)| v.nterms())
                .unwrap()
                .0;

            let a = f.swap_remove(index_smallest);

            // add all other polynomials
            let term_bound = f.iter().map(|x| x.nterms()).sum();
            let mut b = a.zero_with_capacity(term_bound);

            // prevent sampling f[i] and f[i+prime_len] with the same
            // prefactor every iteration
            let num_primes = if f.len().is_multiple_of(SMALL_PRIMES.len()) {
                SMALL_PRIMES.len() - 1
            } else {
                SMALL_PRIMES.len()
            };

            for p in f.iter() {
                let k = Integer::Single(SMALL_PRIMES[prime_index % num_primes]);
                prime_index += 1;
                b = b + p.clone().mul_coeff(k);
            }

            let mut gcd = a.gcd(&b);
            if gcd.is_one() {
                return gcd;
            }

            // remove the content from the gcd before the division test as the odds
            // of an unlucky content are high
            let content = gcd.content();
            gcd = gcd.div_coeff(&content);
            let mut content_gcd = content;

            let old_length = f.len();

            f.retain(|x| {
                if x.try_div(&gcd).is_some() {
                    content_gcd = gcd.ring.gcd(&content_gcd, &x.content());
                    false
                } else {
                    true
                }
            });

            gcd = gcd.mul_coeff(content_gcd);

            if f.is_empty() {
                return gcd;
            }

            debug!(
                "Multiply GCD not found in one try, current estimate: {}",
                gcd
            );

            f.push(gcd);

            if f.len() == old_length + 1 && loop_counter > 5 {
                debug!("Multiple GCD failed");
                return MultivariatePolynomial::repeated_gcd(f);
            }

            loop_counter += 1;
        }
    }

    /// Compute the gcd of two multivariate polynomials using Zippel's algorithm.
    /// TODO: provide a parallel implementation?
    #[instrument(level = "debug", skip_all)]
    fn gcd_zippel<UField: FiniteFieldWorkspace + 'static>(
        &self,
        b: &Self,
        vars: &[usize], // variables
        bounds: &mut [E],
        tight_bounds: &mut [E],
    ) -> Self
    where
        FiniteField<UField>: FiniteFieldCore<UField>,
        <FiniteField<UField> as Set>::Element: Copy,
        Integer: ToFiniteField<UField> + FromFiniteField<UField>,
    {
        debug!("Zippel gcd of {} and {}", self, b);
        #[cfg(debug_assertions)]
        {
            self.check_consistency();
            b.check_consistency();
        }

        // compute scaling factor in Z
        let gamma = self
            .ring
            .gcd(&self.lcoeff_varorder(vars), &b.lcoeff_varorder(vars));
        debug!("gamma {}", gamma);

        let mut primes =
            PrimeIteratorU64::new(UField::get_large_prime().to_u64().unwrap_or(1 << 63));

        'newfirstprime: loop {
            let Some(p) = primes.next() else {
                panic!("Ran out of primes for gcd reconstruction.\ngcd({self},{b})");
            };
            let Some(p) = UField::try_from_integer(p.into()) else {
                panic!("Ran out of primes for gcd reconstruction.\ngcd({self},{b})");
            };

            let mut finite_field = FiniteField::<UField>::new(p.clone());
            let mut gammap = gamma.to_finite_field(&finite_field);

            if finite_field.is_zero(&gammap) {
                continue 'newfirstprime;
            }

            let ap = self.map_coeff(|c| c.to_finite_field(&finite_field), finite_field.clone());
            let bp = b.map_coeff(|c| c.to_finite_field(&finite_field), finite_field.clone());

            debug!("New first image: gcd({},{}) mod {}", ap, bp, p);

            // calculate modular gcd image
            let mut gp = match MultivariatePolynomial::gcd_shape_modular(
                &ap,
                &bp,
                vars,
                bounds,
                tight_bounds,
            ) {
                Some(x) => x,
                None => {
                    debug!("Modular GCD failed: getting new prime");
                    continue 'newfirstprime;
                }
            };

            debug!("GCD suggestion: {}", gp);

            bounds[vars[0]] = gp.degree(vars[0]);

            // construct a new assumed form
            // we have to find the proper normalization
            let gfu = gp.to_univariate_polynomial_list(vars[0]);

            // find a coefficient of x1 in gf that is a monomial (single scaling)
            let mut single_scale = None;
            let mut nx = 0; // count the minimal number of samples needed
            for (i, (c, _e)) in gfu.iter().enumerate() {
                if c.nterms() > nx {
                    nx = c.nterms();
                }
                if c.nterms() == 1 {
                    single_scale = Some(i);
                }
            }

            // In the case of multiple scaling, each sample adds an
            // additional unknown, except for the first
            if single_scale.is_none() {
                let mut nx1 = (gp.nterms() - 1) / (gfu.len() - 1);
                if (gp.nterms() - 1) % (gfu.len() - 1) != 0 {
                    nx1 += 1;
                }
                if nx < nx1 {
                    nx = nx1;
                }
                debug!("Multiple scaling case: sample {} times", nx);
            }

            let gpc = gp.lcoeff_varorder(vars);
            let lcoeff_factor = gp.ring.div(&gammap, &gpc);

            // construct the gcd suggestion in Z
            let mut gm = self.zero_with_capacity(gp.nterms());
            gm.exponents.clone_from(&gp.exponents);
            gm.coefficients = gp
                .coefficients
                .iter()
                .map(|x| {
                    gp.ring
                        .to_symmetric_integer(&gp.ring.mul(x, &lcoeff_factor))
                })
                .collect();

            let mut m = Integer::from_prime(&finite_field); // size of finite field

            debug!("GCD suggestion with gamma: {} mod {} ", gm, p);

            let mut old_gm = self.zero();

            // add new primes until we can reconstruct the full gcd
            'newprime: loop {
                if gm == old_gm {
                    // divide by integer content
                    let gmc = gm.content();
                    let gc = gm.clone().div_coeff(&gmc);

                    debug!("Final suggested gcd: {}", gc);
                    if gc.is_one() || (self.try_div(&gc).is_some() && b.try_div(&gc).is_some()) {
                        return gc;
                    }

                    // if it does not divide, we need more primes
                    debug!("Does not divide: more primes needed");
                }

                old_gm = gm.clone();

                loop {
                    let Some(p) = primes.next() else {
                        panic!(
                            "Ran out of primes for gcd images.\ngcd({self},{b})\nAttempt: {gm}\n vars: {vars:?}, bounds: {bounds:?}; {tight_bounds:?}"
                        );
                    };
                    let Some(p) = UField::try_from_integer(p.into()) else {
                        panic!(
                            "Ran out of primes for gcd images.\ngcd({self},{b})\nAttempt: {gm}\n vars: {vars:?}, bounds: {bounds:?}; {tight_bounds:?}"
                        );
                    };

                    finite_field = FiniteField::<UField>::new(p.clone());

                    gammap = gamma.to_finite_field(&finite_field);

                    if !finite_field.is_zero(&gammap) {
                        break;
                    }
                }

                let ap = self.map_coeff(|c| c.to_finite_field(&finite_field), finite_field.clone());
                let bp = b.map_coeff(|c| c.to_finite_field(&finite_field), finite_field.clone());
                debug!("New image: gcd({},{})", ap, bp);

                // for the univariate case, we don't need to construct an image
                if vars.len() == 1 {
                    gp = ap.univariate_gcd(&bp);
                    if gp.degree(vars[0]) < bounds[vars[0]] {
                        // original image and variable bound unlucky: restart
                        debug!("Unlucky original image: restart");
                        continue 'newfirstprime;
                    }

                    if gp.degree(vars[0]) > bounds[vars[0]] {
                        // prime is probably unlucky
                        debug!("Unlucky current image: try new one");
                        continue 'newprime;
                    }

                    for m in gp.into_iter() {
                        if gfu.iter().all(|(_, pow)| *pow != m.exponents[vars[0]]) {
                            debug!("Bad shape: terms missing");
                            continue 'newfirstprime;
                        }
                    }
                } else {
                    let rec = if let Some(single_scale) = single_scale {
                        MultivariatePolynomial::construct_new_image_single_scale(
                            &ap,
                            &bp,
                            ap.degree(vars[0]),
                            bp.degree(vars[0]),
                            bounds,
                            single_scale,
                            &vars[1..],
                            vars[0],
                            &gfu,
                        )
                    } else {
                        MultivariatePolynomial::construct_new_image_multiple_scales(
                            &ap,
                            &bp,
                            // NOTE: different from paper where they use a.degree(..)
                            // it could be that the degree in ap is lower than that of a
                            // which means the sampling will never terminate
                            ap.degree(vars[0]),
                            bp.degree(vars[0]),
                            bounds,
                            &vars[1..],
                            vars[0],
                            &gfu,
                        )
                    };

                    match rec {
                        Ok(r) => {
                            gp = r;
                        }
                        Err(GCDError::BadOriginalImage) => continue 'newfirstprime,
                        Err(GCDError::BadCurrentImage) => continue 'newprime,
                    }
                }

                // scale the new image
                let gpc = gp.lcoeff_varorder(vars);
                gp = gp.mul_coeff(ap.ring.div(&gammap, &gpc));
                debug!("gp: {} mod {}", gp, gp.ring.get_prime());

                // use chinese remainder theorem to merge coefficients and map back to Z
                // terms could be missing in gp, but not in gm (TODO: check this?)
                let mut gpi = 0;
                for t in 0..gm.nterms() {
                    let gpc = if gm.exponents(t) == gp.exponents(gpi) {
                        gpi += 1;
                        gp.coefficients[gpi - 1]
                    } else {
                        ap.ring.zero()
                    };

                    let gmc = &mut gm.coefficients[t];
                    let coeff = if gmc.is_negative() {
                        self.ring.add(&*gmc, &m)
                    } else {
                        gmc.clone()
                    };

                    *gmc = Integer::chinese_remainder(
                        coeff,
                        Integer::from_finite_field(&gp.ring, gpc),
                        m.clone(),
                        Integer::from_prime(&gp.ring),
                    );
                }

                self.ring.mul_assign(&mut m, &Integer::from_prime(&gp.ring));

                debug!("gm: {} from ring {}", gm, m);
            }
        }
    }

    /// Compute the gcd of two multivariate polynomials using Hu-Monagan's algorithm.
    /// The polynomials must be primitive in the first variable, which must also be present.
    #[instrument(level = "debug", skip_all)]
    pub fn gcd_hu_monagan(&self, b: &Self, bounds: &[E]) -> Self {
        debug!(
            "Hu-Monagan gcd of {} and {} with bounds {:?}",
            self, b, bounds
        );
        assert!(bounds[0] > E::zero());

        let lc_a = self.univariate_lcoeff(0);
        let lc_b = b.univariate_lcoeff(0);
        let gamma = if lc_a.nterms() < lc_b.nterms() {
            // TODO: in paper they write it the other way around, but in the text they say smaller
            lc_a
        } else {
            lc_b
        };

        let largest_coeff = self
            .coefficients
            .iter()
            .chain(&b.coefficients)
            .max_by(|a, b| a.abs_cmp(b))
            .unwrap()
            .abs()
            * 2i64;

        let mut r: Vec<_> = (0..self.nvars())
            .map(|i| {
                self.degree(i)
                    .max(b.degree(i))
                    .max(bounds[i] + gamma.degree(i))
                    .to_u32()
            })
            .collect();

        let mut delta = 1;
        let mut d_0 = bounds[0];
        let mut h = self.map_exp(|e| e.to_u32()).zero();
        let mut m = Integer::one(); // CRA new modulus
        let mut ms: Vec<u64> = vec![]; // CRA primes
        let mut missing_terms = true;
        let mut sigma: Vec<HashSet<u64>> = vec![];
        let mut tau = vec![];

        let mut smooth_prime_index = 0;
        let mut rng = rand::rng();

        'kronecker_prime: loop {
            for rr in &mut r {
                *rr += 1;
            }

            let mut powers = vec![r[1]];
            for (i, r) in r.iter().skip(1).enumerate().skip(1) {
                powers.push(powers[i - 1] * r);
            }
            let ri_prod = *powers.last().unwrap() as u64;

            let kr_a = self.map_exp(|e| e.to_u32()).kronecker_map(&powers, 1);
            let kr_b = b.map_exp(|e| e.to_u32()).kronecker_map(&powers, 1);
            let kr_gamma = gamma.map_exp(|e| e.to_u32()).kronecker_map(&powers, 1);

            if kr_a.degree(0) < self.degree(0).to_u32() || kr_b.degree(0) < b.degree(0).to_u32() {
                debug!("Bad Kronecker image");
                continue 'kronecker_prime;
            }

            delta += 1;

            let mut re_init = true;
            'new_image: loop {
                if re_init {
                    h = kr_a.zero();
                    m = Integer::one();
                    missing_terms = true;
                    sigma = vec![HashSet::new(); d_0.to_u32() as usize + 1];
                    tau = vec![0; d_0.to_u32() as usize + 1];

                    re_init = false;
                }

                // TODO: check degree bound d_0 again?
                // TODO: store current images of kra and krb, only recompute h mod p!
                ms.retain(|p| {
                    let p = Zp64::new(*p);
                    let kr_a_p = kr_a.map_coeff(|c| c.to_finite_field(&p), p.clone());
                    let kr_b_p = kr_b.map_coeff(|c| c.to_finite_field(&p), p.clone());
                    let kr_gamma_p = kr_gamma.map_coeff(|c| c.to_finite_field(&p), p.clone());
                    let h = h.map_coeff(|c| c.to_finite_field(&p), p.clone());

                    loop {
                        let beta = p.sample(&mut rng, (0, i64::MAX));
                        if kr_gamma_p.replace(1, &beta).is_zero() {
                            continue;
                        }

                        let h_b = h.replace(1, &beta);
                        if kr_a_p.replace(1, &beta).try_div(&h_b).is_none()
                            || kr_b_p.replace(1, &beta).try_div(&h_b).is_none()
                        {
                            debug!("Bad Kronecker image for prime {}", p);

                            m /= p.get_prime();
                            break false; // keep prime
                        }

                        break true;
                    }
                });

                let prime_bound = ri_prod * 2u64.pow(delta);

                let (p, totient_primes, alpha, kr_a_p, kr_b_p, kr_gamma_p) = loop {
                    let Some((p, alpha, fs)) = SMOOTH_PRIMES.get(smooth_prime_index) else {
                        panic!(
                            "Ran out of smooth primes for Hu-Monagan GCD.\ngcd({},{})",
                            self, b
                        );
                    };

                    smooth_prime_index += 1;

                    if *p < prime_bound
                        || largest_coeff < 1i64 << 32 && *p < largest_coeff.to_u64().unwrap()
                    {
                        continue;
                    }

                    let field = Zp64::new(*p);

                    let kr_a_p = kr_a.map_coeff(|c| c.to_finite_field(&field), field.clone());
                    let kr_b_p = kr_b.map_coeff(|c| c.to_finite_field(&field), field.clone());
                    let kr_gamma_p =
                        kr_gamma.map_coeff(|c| c.to_finite_field(&field), field.clone());

                    if kr_a_p.degree(0) < kr_a.degree(0).to_u32()
                        || kr_b_p.degree(0) < kr_b.degree(0).to_u32()
                    {
                        debug!("Bad prime {}", p);
                        continue;
                    }

                    let mut totient_primes = vec![];
                    for (f, prime) in fs.iter().zip(&SMOOTH_PRIME_BASE) {
                        if *f > 0 {
                            totient_primes.push((*prime, *f as u32));
                        }
                    }

                    let alpha = field.to_element(*alpha as u64);
                    break (field, totient_primes, alpha, kr_a_p, kr_b_p, kr_gamma_p);
                };

                let s = p.sample(&mut rng, (0, i64::MAX));
                let shift = p.from_element(&s);

                let mut gs = vec![];

                let mut sample_points = vec![];
                let mut coeffs = vec![];
                let mut bma_polys = vec![None; d_0.to_u32() as usize + 1];

                let kr_ap_exponents_alpha: Vec<_> = kr_a_p
                    .exponents
                    .chunks(kr_a_p.nvars())
                    .map(|x| p.pow(&alpha, x[1].to_u32() as u64))
                    .collect(); // will be updated live

                let mut kr_ap_exponents_s: Vec<_> = kr_ap_exponents_alpha
                    .iter()
                    .map(|x| p.pow(x, shift))
                    .collect();

                let kr_bp_exponents_alpha: Vec<_> = kr_b_p
                    .exponents
                    .chunks(kr_b_p.nvars())
                    .map(|x| p.pow(&alpha, x[1].to_u32() as u64))
                    .collect();

                let mut kr_bp_exponents_s: Vec<_> = kr_bp_exponents_alpha
                    .iter()
                    .map(|x| p.pow(x, shift))
                    .collect();

                let kr_gamma_p_exponents_alpha: Vec<_> = kr_gamma_p
                    .exponents
                    .chunks(kr_gamma_p.nvars())
                    .map(|x| p.pow(&alpha, x[1].to_u32() as u64))
                    .collect();

                let mut kr_gamma_p_exponents_s: Vec<_> = kr_gamma_p_exponents_alpha
                    .iter()
                    .map(|x| p.pow(x, shift))
                    .collect();

                fn eval(
                    p: &Zp64,
                    poly: &MultivariatePolynomial<Zp64, u32>,
                    alpha: &[FiniteFieldElement<u64>],
                    s: &mut [FiniteFieldElement<u64>],
                ) -> MultivariatePolynomial<Zp64, u32> {
                    let mut res = poly.zero();
                    let mut exp = vec![0; poly.nvars()];
                    for (((c, ee), aa), ss) in poly
                        .coefficients
                        .iter()
                        .zip(poly.exponents.chunks(poly.nvars()))
                        .zip(alpha.iter())
                        .zip(s.iter_mut())
                    {
                        exp[0] = ee[0];

                        let c = p.mul(c, ss);
                        *ss = p.mul(&*ss, aa);

                        res.append_monomial(c, &exp);
                    }

                    res
                }

                'new_sample: loop {
                    for _ in 0..2 {
                        let sample_point = p.pow(&alpha, gs.len() as u64 + p.from_element(&s));

                        let a_j = eval(&p, &kr_a_p, &kr_ap_exponents_alpha, &mut kr_ap_exponents_s);
                        let b_j = eval(&p, &kr_b_p, &kr_bp_exponents_alpha, &mut kr_bp_exponents_s);

                        if a_j.degree(0) < kr_a_p.degree(0) || b_j.degree(0) < kr_b_p.degree(0) {
                            debug!("Bad Kronecker image, trying new prime");
                            continue 'kronecker_prime;
                        }

                        let kr_gamma_p_e = eval(
                            &p,
                            &kr_gamma_p,
                            &kr_gamma_p_exponents_alpha,
                            &mut kr_gamma_p_exponents_s,
                        );

                        let g = a_j.univariate_gcd(&b_j) * &kr_gamma_p_e;
                        gs.push(g);
                        sample_points.push(sample_point);
                    }

                    let d_min = gs[gs.len() - 2].degree(0).min(gs[gs.len() - 1].degree(0));
                    let d_max = gs[gs.len() - 2].degree(0).max(gs[gs.len() - 1].degree(0));
                    if d_min < d_0.to_u32() {
                        debug!("Unlucky sample in dimension {}", d_min);
                        d_0 = E::from_u32(d_min);
                        re_init = true;
                        continue 'new_image;
                    }
                    if d_max > d_0.to_u32() {
                        debug!("Bad Kronecker image, trying new prime");
                        continue 'kronecker_prime;
                    }

                    if missing_terms
                        && (gs.len() < kr_gamma.nterms() * 2
                            || gs.len() < tau.iter().max().unwrap() * 2)
                        || !missing_terms && gs.len() <= *tau.iter().max().unwrap()
                    {
                        continue 'new_sample;
                    }

                    coeffs.clear(); // FIXME: do update instead
                    let mut ee = vec![0; self.nvars()];
                    for i in 0..=d_0.to_u32() {
                        ee[0] = i;
                        coeffs.push(
                            gs.iter()
                                .map(|x| x.coefficient(&ee).unwrap_or(p.zero()))
                                .collect::<Vec<_>>(),
                        );
                    }

                    // determine all monomials through linear recurrence relations
                    if missing_terms {
                        for i in 0..=d_0.to_u32() {
                            if coeffs[i as usize].iter().all(|x| p.is_zero(x)) {
                                continue;
                            }

                            let (r, s) = p.find_linear_recurrence_relation(&coeffs[i as usize]);

                            if s < 2 || p.is_zero(&r[0]) {
                                continue 'new_sample;
                            }

                            let mut bma_poly = kr_a_p.zero();
                            let mut exp2 = vec![0u32; self.nvars()];
                            for (j, cs) in r.iter().rev().enumerate() {
                                if !p.is_zero(cs) {
                                    exp2[0] = j as u32;
                                    bma_poly.append_monomial(p.neg(cs), &exp2);
                                }
                            }
                            exp2[0] = r.len() as u32;
                            bma_poly.append_monomial(p.one(), &exp2);

                            tau[i as usize] = bma_poly.degree(0).to_u32() as usize;

                            if let Some(bma_poly_old) = bma_polys[i as usize].as_ref() {
                                if bma_poly_old != &bma_poly {
                                    debug!("bma_poly changed for {}, trying new sample", i);
                                    sigma[i as usize].clear();
                                } else {
                                    continue;
                                }
                            }

                            debug!("bma_poly for {}: {}; s= {}", i, bma_poly, s);

                            let mut fs = bma_poly.factor();
                            bma_polys[i as usize] = Some(bma_poly);
                            fs.retain(|(f, _)| !f.is_constant());

                            if fs.len() != tau[i as usize] {
                                debug!("Stabilized too early"); // TODO: don't remove all the other terms
                                continue 'new_sample; // TODO: clear sigma on fail?
                            }

                            for (f, _ee) in fs {
                                assert_eq!(f.degree(0), 1);

                                let m = p.neg(&f.get_constant());

                                let e =
                                    p.discrete_log(&alpha, &m, p.get_prime() - 1, &totient_primes);

                                let ee = p.from_element(&e);

                                if ee >= ri_prod {
                                    debug!("Unlucky evaluations");
                                    continue 'kronecker_prime;
                                }

                                sigma[i as usize].insert(ee);
                            }
                        }
                    }

                    break;
                }

                // solve system to find coefficients
                let mut h_p = kr_a_p.zero();
                for i in 0..=d_0.to_u32() {
                    if coeffs[i as usize].iter().all(|x| p.is_zero(x)) {
                        continue;
                    }

                    let monomials: Vec<_> = sigma[i as usize].iter().cloned().collect();
                    let mat = sample_points
                        .iter()
                        .map(|x| monomials.iter().map(|e| p.pow(x, *e)).collect())
                        .collect::<Vec<_>>();

                    let sol = gs[0].solve_vandermonde_matrix(mat, coeffs[i as usize].clone());

                    let mut exp = vec![0; self.nvars()];
                    let mut h_p_e = kr_a_p.zero();
                    for (coeff, e) in sol.into_iter().zip(&monomials) {
                        assert!(*e <= u32::MAX as u64);
                        exp[1] = *e as u32; // FIXME: we should support u64 exponents!
                        h_p_e.append_monomial(coeff, &exp);
                    }

                    if !missing_terms {
                        let aa = h_p_e.replace(
                            1,
                            &p.pow(&alpha, p.from_element(&s) as u64 + tau[i as usize] as u64),
                        );
                        if aa.get_constant() != coeffs[i as usize][tau[i as usize]] {
                            debug!("Missing terms in h_p_e: {}", h_p_e);

                            missing_terms = true;
                            delta += 1;
                            continue 'new_image;
                        }
                    }

                    exp[0] = i;
                    exp[1] = 0;
                    h_p = h_p + h_p_e.mul_exp(&exp);
                }

                missing_terms = false;
                // TODO: increase tau and sigma or is this already done correctly during the run?

                let hz = h_p.map_coeff(|c| p.to_symmetric_integer(c), Z);

                debug!("h_p: {}", h_p);
                debug!("hz: {}", hz);

                // chinese remainder
                ms.push(p.get_prime());
                if m == 1 {
                    h = hz;
                    m = p.get_prime().into();
                } else {
                    // FIXME: assumes that there are no missing terms in gm
                    let mut gpi = 0;
                    for t in 0..h.nterms() {
                        let gpc = if h.exponents(t) == hz.exponents(gpi) {
                            gpi += 1;
                            hz.coefficients[gpi - 1].clone()
                        } else {
                            0.into()
                        };

                        let gmc = &mut h.coefficients[t];
                        let coeff = if gmc.is_negative() {
                            self.ring.add(&*gmc, &m)
                        } else {
                            gmc.clone()
                        };

                        *gmc =
                            Integer::chinese_remainder(coeff, gpc, m.clone(), p.get_prime().into());
                    }

                    m *= p.get_prime();
                }

                debug!("h_cra = {}", h);

                // division test
                let hm = h.kronecker_inv_map(&powers, 1).map_exp(|e| E::from_u32(*e));
                debug!("h_inv = {}", hm);
                let content = hm.univariate_content(0); // triggers another hu gcd run!
                let hc = hm / &content;

                debug!("hc {}", hc);

                if self.try_div(&hc).is_some() && b.try_div(&hc).is_some() {
                    debug!("Found GCD: {}", hc);
                    return hc;
                } else {
                    debug!("Non-division of {}, trying new image", hc);
                }
            }
        }
    }

    /// Compute the gcd of two multivariate polynomials using Hu-Monagan's algorithm.
    /// The polynomials must be primitive in the first variable, which must also be present.
    #[instrument(level = "debug", skip_all)]
    pub fn gcd_hu_monagan_bivariate(&self, b: &Self, bounds: &[E]) -> Self {
        debug!(
            "Hu-Monagan gcd of {} and {} with bounds {:?}",
            self, b, bounds
        );
        assert!(bounds[0] > E::zero());

        let start_exp = 2;

        // TODO: make primitive in x0,x1?

        let lc_a = self.bivariate_lcoeff();
        let lc_b = b.bivariate_lcoeff();
        let gamma = if lc_a.nterms() < lc_b.nterms() {
            lc_a
        } else {
            lc_b
        };

        let largest_coeff = self
            .coefficients
            .iter()
            .chain(&b.coefficients)
            .max_by(|a, b| a.abs_cmp(b))
            .unwrap()
            .abs()
            * 2i64;

        let mut r: Vec<_> = (0..self.nvars())
            .map(|i| {
                self.degree(i)
                    .max(b.degree(i))
                    .max(bounds[i] + gamma.degree(i))
                    .to_u32()
            })
            .collect();

        let mut delta = 1;
        let mut d_0_1 = (bounds[0], bounds[1]);
        let mut h = self.map_exp(|e| e.to_u32()).zero();
        let mut m = Integer::one(); // CRA new modulus
        let mut ms: Vec<u64> = vec![]; // CRA primes
        let mut missing_terms = true;
        let mut sigma: HashMap<(u32, u32), HashSet<u64>> = HashMap::default();
        let mut tau = HashMap::default();

        let mut smooth_prime_index = 0;
        let mut rng = rand::rng();

        'kronecker_prime: loop {
            for rr in &mut r {
                *rr += 1;
            }

            let mut powers = vec![r[start_exp]];
            for (i, r) in r.iter().skip(start_exp).enumerate().skip(1) {
                powers.push(powers[i - 1] * r);
            }
            let ri_prod = *powers.last().unwrap() as u64;

            let kr_a = self
                .map_exp(|e| e.to_u32())
                .kronecker_map(&powers, start_exp);
            let kr_b = b.map_exp(|e| e.to_u32()).kronecker_map(&powers, start_exp);
            let kr_gamma = gamma
                .map_exp(|e| e.to_u32())
                .kronecker_map(&powers, start_exp);

            for i in 0..start_exp {
                if kr_a.degree(i) < self.degree(i).to_u32() || kr_b.degree(i) < b.degree(i).to_u32()
                {
                    debug!("Bad Kronecker image");
                    continue 'kronecker_prime;
                }
            }

            delta += 1;

            let mut re_init = true;
            'new_image: loop {
                if re_init {
                    h = kr_a.zero();
                    m = Integer::one();
                    missing_terms = true;
                    sigma.clear();
                    tau.clear();

                    re_init = false;
                }

                // TODO: check degree bound d_0 again?
                // TODO: store current images of kra and krb, only recompute h mod p!
                ms.retain(|p| {
                    let p = Zp64::new(*p);
                    let kr_a_p = kr_a.map_coeff(|c| c.to_finite_field(&p), p.clone());
                    let kr_b_p = kr_b.map_coeff(|c| c.to_finite_field(&p), p.clone());
                    let kr_gamma_p = kr_gamma.map_coeff(|c| c.to_finite_field(&p), p.clone());
                    let h = h.map_coeff(|c| c.to_finite_field(&p), p.clone());

                    loop {
                        let beta = p.sample(&mut rng, (0, i64::MAX));
                        if kr_gamma_p.replace(start_exp, &beta).is_zero() {
                            continue;
                        }

                        let h_b = h.replace(start_exp, &beta);
                        if kr_a_p.replace(start_exp, &beta).try_div(&h_b).is_none()
                            || kr_b_p.replace(start_exp, &beta).try_div(&h_b).is_none()
                        {
                            debug!("Bad Kronecker image for prime {}", p);

                            m /= p.get_prime();
                            break false; // keep prime
                        }

                        break true;
                    }
                });

                let prime_bound = ri_prod * 2u64.pow(delta);

                let (p, totient_primes, alpha, kr_a_p, kr_b_p, kr_gamma_p) = 'new_prime: loop {
                    let Some((p, alpha, fs)) = SMOOTH_PRIMES.get(smooth_prime_index) else {
                        panic!(
                            "Ran out of smooth primes for Hu-Monagan GCD.\ngcd({},{})",
                            self, b
                        );
                    };

                    smooth_prime_index += 1;

                    if *p < prime_bound
                        || largest_coeff < 1i64 << 32 && *p < largest_coeff.to_u64().unwrap()
                    {
                        continue;
                    }

                    let field = Zp64::new(*p);

                    let kr_a_p = kr_a.map_coeff(|c| c.to_finite_field(&field), field.clone());
                    let kr_b_p = kr_b.map_coeff(|c| c.to_finite_field(&field), field.clone());
                    let kr_gamma_p =
                        kr_gamma.map_coeff(|c| c.to_finite_field(&field), field.clone());

                    for i in 0..start_exp {
                        if kr_a_p.degree(i) < kr_a.degree(i).to_u32()
                            || kr_b_p.degree(i) < kr_b.degree(i).to_u32()
                        {
                            debug!("Bad prime {}", p);
                            continue 'new_prime;
                        }
                    }

                    let mut totient_primes = vec![];
                    for (f, prime) in fs.iter().zip(&SMOOTH_PRIME_BASE) {
                        if *f > 0 {
                            totient_primes.push((*prime, *f as u32));
                        }
                    }

                    let alpha = field.to_element(*alpha as u64);
                    break (field, totient_primes, alpha, kr_a_p, kr_b_p, kr_gamma_p);
                };

                let s = p.sample(&mut rng, (0, i64::MAX));
                let shift = p.from_element(&s);

                let mut gs = vec![];

                let mut sample_points = vec![];
                let mut coeffs: HashMap<(u32, u32), Vec<(usize, FiniteFieldElement<u64>)>> =
                    HashMap::default();
                let mut bma_polys = HashMap::default();

                let kr_ap_exponents_alpha: Vec<_> = kr_a_p
                    .exponents
                    .chunks(kr_a_p.nvars())
                    .map(|x| p.pow(&alpha, x[start_exp].to_u32() as u64))
                    .collect(); // will be updated live

                let mut kr_ap_exponents_s: Vec<_> = kr_ap_exponents_alpha
                    .iter()
                    .map(|x| p.pow(x, shift))
                    .collect();

                let kr_bp_exponents_alpha: Vec<_> = kr_b_p
                    .exponents
                    .chunks(kr_b_p.nvars())
                    .map(|x| p.pow(&alpha, x[start_exp].to_u32() as u64))
                    .collect();

                let mut kr_bp_exponents_s: Vec<_> = kr_bp_exponents_alpha
                    .iter()
                    .map(|x| p.pow(x, shift))
                    .collect();

                let kr_gamma_p_exponents_alpha: Vec<_> = kr_gamma_p
                    .exponents
                    .chunks(kr_gamma_p.nvars())
                    .map(|x| p.pow(&alpha, x[start_exp].to_u32() as u64))
                    .collect();

                let mut kr_gamma_p_exponents_s: Vec<_> = kr_gamma_p_exponents_alpha
                    .iter()
                    .map(|x| p.pow(x, shift))
                    .collect();

                fn eval(
                    p: &Zp64,
                    poly: &MultivariatePolynomial<Zp64, u32>,
                    alpha: &[FiniteFieldElement<u64>],
                    s: &mut [FiniteFieldElement<u64>],
                ) -> MultivariatePolynomial<Zp64, u32> {
                    let mut res = poly.zero();
                    let mut exp = vec![0; poly.nvars()];
                    for (((c, ee), aa), ss) in poly
                        .coefficients
                        .iter()
                        .zip(poly.exponents.chunks(poly.nvars()))
                        .zip(alpha.iter())
                        .zip(s.iter_mut())
                    {
                        exp[0] = ee[0];
                        exp[1] = ee[1];

                        let c = p.mul(c, ss);
                        *ss = p.mul(&*ss, aa);

                        res.append_monomial(c, &exp);
                    }

                    res
                }

                'new_sample: loop {
                    for _ in 0..2 {
                        let sample_point = p.pow(&alpha, gs.len() as u64 + p.from_element(&s));

                        let a_j = eval(&p, &kr_a_p, &kr_ap_exponents_alpha, &mut kr_ap_exponents_s);
                        let b_j = eval(&p, &kr_b_p, &kr_bp_exponents_alpha, &mut kr_bp_exponents_s);

                        for i in 0..start_exp {
                            if a_j.degree(i) < kr_a_p.degree(i) || b_j.degree(i) < kr_b_p.degree(i)
                            {
                                debug!("Bad Kronecker image, trying new prime");
                                continue 'kronecker_prime;
                            }
                        }

                        let kr_gamma_p_e = eval(
                            &p,
                            &kr_gamma_p,
                            &kr_gamma_p_exponents_alpha,
                            &mut kr_gamma_p_exponents_s,
                        );

                        let g = a_j.gcd(&b_j) * &kr_gamma_p_e; // FIXME: bad normalization?
                        gs.push(g);
                        sample_points.push(sample_point);
                    }

                    let d_min = gs[gs.len() - 2].degree(0).min(gs[gs.len() - 1].degree(0));
                    let d_max = gs[gs.len() - 2].degree(0).max(gs[gs.len() - 1].degree(0));
                    if d_min < d_0_1.0.to_u32() {
                        debug!("Unlucky sample: dim x0 is {} vs {}", d_min, d_0_1.0);
                        d_0_1.0 = E::from_u32(d_min);
                        re_init = true;
                        continue 'new_image;
                    }
                    if d_max > d_0_1.0.to_u32() {
                        debug!("Bad Kronecker image, trying new prime");
                        continue 'kronecker_prime;
                    }

                    let d_min = gs[gs.len() - 2].degree(1).min(gs[gs.len() - 1].degree(1));
                    let d_max = gs[gs.len() - 2].degree(1).max(gs[gs.len() - 1].degree(1));
                    if d_min < d_0_1.1.to_u32() {
                        debug!("Unlucky sample: dim x1 is {} vs {}", d_min, d_0_1.1);
                        d_0_1.1 = E::from_u32(d_min);
                        re_init = true;
                        continue 'new_image;
                    }
                    if d_max > d_0_1.1.to_u32() {
                        debug!(
                            "Bad Kronecker image for var 1, trying new prime: {} vs {}",
                            d_max, d_0_1.1
                        );
                        continue 'kronecker_prime;
                    }

                    if missing_terms
                        && (gs.len() < kr_gamma.nterms() * 2
                            || gs.len() < tau.values().max().unwrap_or(&0) * 2)
                        || !missing_terms && gs.len() <= *tau.values().max().unwrap_or(&0)
                    {
                        continue 'new_sample;
                    }

                    coeffs.clear(); // FIXME: do update instead

                    for (i, hh) in gs.iter().enumerate() {
                        for m in hh {
                            if !p.is_zero(m.coefficient) {
                                coeffs
                                    .entry((m.exponents[0], m.exponents[1]))
                                    .or_default()
                                    .push((i, *m.coefficient));
                            }
                        }
                    }

                    // determine all monomials through linear recurrence relations
                    let mut row = vec![p.zero(); gs.len()];
                    if missing_terms {
                        for (e_x_y, l) in &coeffs {
                            for pp in l {
                                row[pp.0] = pp.1;
                            }

                            let (r, s) = p.find_linear_recurrence_relation(&row);

                            for pp in l {
                                row[pp.0] = p.zero();
                            }

                            if s < 2 || p.is_zero(&r[0]) {
                                continue 'new_sample;
                            }

                            let mut bma_poly = kr_a_p.zero();
                            let mut exp2 = vec![0u32; self.nvars()];
                            for (j, cs) in r.iter().rev().enumerate() {
                                if !p.is_zero(cs) {
                                    exp2[0] = j as u32;
                                    bma_poly.append_monomial(p.neg(cs), &exp2);
                                }
                            }
                            exp2[0] = r.len() as u32;
                            bma_poly.append_monomial(p.one(), &exp2);

                            tau.insert(*e_x_y, bma_poly.degree(0).to_u32() as usize);

                            if let Some(bma_poly_old) = bma_polys.get(e_x_y) {
                                if bma_poly_old != &bma_poly {
                                    debug!("bma_poly changed for {:?}, trying new sample", e_x_y);
                                    sigma.remove(e_x_y);
                                } else {
                                    continue;
                                }
                            }

                            debug!("bma_poly for {:?}: {}; s= {}", e_x_y, bma_poly, s);

                            let mut fs = bma_poly.factor();
                            bma_polys.insert(*e_x_y, bma_poly);
                            fs.retain(|(f, _)| !f.is_constant());

                            if fs.len() != tau[e_x_y] {
                                debug!("Stabilized too early"); // TODO: don't remove all the other terms
                                continue 'new_sample; // TODO: clear sigma on fail?
                            }

                            for (f, _ee) in fs {
                                assert_eq!(f.degree(0), 1);

                                let m = p.neg(&f.get_constant());

                                let e =
                                    p.discrete_log(&alpha, &m, p.get_prime() - 1, &totient_primes);

                                let ee = p.from_element(&e);

                                if ee >= ri_prod {
                                    debug!("Unlucky evaluations");
                                    continue 'kronecker_prime;
                                }

                                sigma.entry(*e_x_y).or_default().insert(ee);
                            }
                        }
                    }

                    break;
                }

                // solve system to find coefficients
                let mut h_p = kr_a_p.zero();
                for (e_x_y, l) in &coeffs {
                    let mut row = vec![p.zero(); gs.len()];
                    for pp in l {
                        row[pp.0] = pp.1;
                    }

                    let monomials: Vec<_> = sigma[e_x_y].iter().cloned().collect();
                    let mat = sample_points
                        .iter()
                        .map(|x| monomials.iter().map(|e| p.pow(x, *e)).collect())
                        .collect::<Vec<_>>();

                    let sol = gs[0].solve_vandermonde_matrix(mat, row);

                    let mut exp = vec![0; self.nvars()];
                    let mut h_p_e = kr_a_p.zero();
                    for (coeff, e) in sol.into_iter().zip(&monomials) {
                        assert!(*e <= u32::MAX as u64);
                        exp[2] = *e as u32; // FIXME: we should support u64 exponents!
                        h_p_e.append_monomial(coeff, &exp);
                    }

                    if !missing_terms {
                        let max = tau[e_x_y];
                        let aa = h_p_e.replace(
                            start_exp,
                            &p.pow(&alpha, p.from_element(&s) as u64 + max as u64),
                        );

                        if aa.get_constant()
                            != coeffs[e_x_y]
                                .iter()
                                .find(|(i, _)| *i == max)
                                .map(|(_, v)| *v)
                                .unwrap_or(p.zero())
                        {
                            debug!("Missing terms in h_p2 for {:?}: {}; {}", e_x_y, h_p_e, aa);

                            missing_terms = true;
                            delta += 1;
                            continue 'new_image;
                        }
                    }

                    exp[0] = e_x_y.0;
                    exp[1] = e_x_y.1;
                    exp[2] = 0;

                    h_p = h_p + h_p_e.mul_exp(&exp);
                }

                missing_terms = false;
                // TODO: increase tau and sigma or is this already done correctly during the run?

                let hz = h_p.map_coeff(|c| p.to_symmetric_integer(c), Z);

                debug!("h_p: {}", h_p);
                debug!("hz: {}", hz);

                // chinese remainder
                ms.push(p.get_prime());
                if m == 1 {
                    h = hz;
                    m = p.get_prime().into();
                } else {
                    // FIXME: assumes that there are no missing terms in gm
                    let mut gpi = 0;
                    for t in 0..h.nterms() {
                        let gpc = if h.exponents(t) == hz.exponents(gpi) {
                            gpi += 1;
                            hz.coefficients[gpi - 1].clone()
                        } else {
                            0.into()
                        };

                        let gmc = &mut h.coefficients[t];
                        let coeff = if gmc.is_negative() {
                            self.ring.add(&*gmc, &m)
                        } else {
                            gmc.clone()
                        };

                        *gmc =
                            Integer::chinese_remainder(coeff, gpc, m.clone(), p.get_prime().into());
                    }

                    m *= p.get_prime();
                }

                debug!("h_cra = {}", h);

                // division test
                let hm = h
                    .kronecker_inv_map(&powers, start_exp)
                    .map_exp(|e| E::from_u32(*e));
                debug!("h_inv = {}", hm);
                let content = hm.univariate_content(0); // TODO: bivariate content?
                let hc = hm / &content;

                debug!("hc {}", hc);

                if self.try_div(&hc).is_some() && b.try_div(&hc).is_some() {
                    debug!("Found GCD: {}", hc);
                    return hc;
                } else {
                    debug!("Non-division of {}, trying new image", hc);
                }
            }
        }
    }
}

/// Polynomial GCD functions for a certain coefficient type `Self`.
pub trait PolynomialGCD<E: PositiveExponent>: Ring {
    fn heuristic_gcd(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
    ) -> Option<(
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
    )>;
    fn gcd_multiple(f: Vec<MultivariatePolynomial<Self, E>>) -> MultivariatePolynomial<Self, E>;
    fn gcd(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
        vars: &[usize],
        bounds: &mut [E],
    ) -> MultivariatePolynomial<Self, E>;
    fn get_gcd_var_bounds(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
        vars: &[usize],
    ) -> SmallVec<[E; INLINED_EXPONENTS]>;
    fn normalize(a: MultivariatePolynomial<Self, E>) -> MultivariatePolynomial<Self, E>;
}

impl<E: PositiveExponent> PolynomialGCD<E> for IntegerRing {
    fn heuristic_gcd(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
    ) -> Option<(
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
    )> {
        // estimate if the heuristic gcd will overflow
        let mut max_deg_a = 0;
        let mut contains_a: SmallVec<[bool; INLINED_EXPONENTS]> = smallvec![false; a.nvars()];
        for t in a {
            let mut deg = 1;
            for (var, e) in t.exponents.iter().enumerate() {
                let v = e.to_u32() as usize;
                if v > 0 {
                    contains_a[var] = true;
                    deg *= v + 1;
                }
            }

            if deg > max_deg_a {
                max_deg_a = deg;
            }
        }

        let mut max_deg_b = 0;
        let mut contains_b: SmallVec<[bool; INLINED_EXPONENTS]> = smallvec![false; b.nvars()];
        for t in b {
            let mut deg = 1;
            for (var, e) in t.exponents.iter().enumerate() {
                let v = e.to_u32() as usize;
                if v > 0 {
                    contains_b[var] = true;
                    deg *= v + 1;
                }
            }

            if deg > max_deg_b {
                max_deg_b = deg;
            }
        }

        let num_shared_vars = contains_a
            .iter()
            .zip(&contains_b)
            .filter(|(a, b)| **a && **b)
            .count();

        if max_deg_a < 20 || max_deg_b < 20 || num_shared_vars < 3 && max_deg_a.min(max_deg_b) < 150
        {
            a.heuristic_gcd(b).ok()
        } else {
            None
        }
    }

    fn gcd_multiple(f: Vec<MultivariatePolynomial<Self, E>>) -> MultivariatePolynomial<Self, E> {
        MultivariatePolynomial::gcd_multiple(f)
    }

    fn gcd(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
        vars: &[usize],
        bounds: &mut [E],
    ) -> MultivariatePolynomial<Self, E> {
        if GLOBAL_SETTINGS
            .use_hu_monagan_poly_gcd
            .load(std::sync::atomic::Ordering::Relaxed)
            && vars[0] == 0
            && bounds[0] > E::zero()
            && bounds.iter().filter(|x| **x > E::zero()).count() > 2
        {
            // TODO: choose better ordering
            return a.gcd_hu_monagan_bivariate(b, bounds);
        }

        let mut tight_bounds: SmallVec<[E; INLINED_EXPONENTS]> = bounds.iter().cloned().collect();
        if a.coefficients
            .iter()
            .any(|x| !matches!(x, Integer::Single(_)))
            || b.coefficients
                .iter()
                .any(|x| !matches!(x, Integer::Single(_)))
        {
            MultivariatePolynomial::gcd_zippel::<u64>(a, b, vars, bounds, &mut tight_bounds)
        } else {
            MultivariatePolynomial::gcd_zippel::<u32>(a, b, vars, bounds, &mut tight_bounds)
        }
    }

    fn get_gcd_var_bounds(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
        vars: &[usize],
    ) -> SmallVec<[E; INLINED_EXPONENTS]> {
        let mut bounds: SmallVec<[_; INLINED_EXPONENTS]> =
            (0..a.nvars()).map(|_| E::zero()).collect();

        let mut primes = PrimeIteratorU64::new(u32::get_large_prime() as u64);

        let mut f = Zp::new(primes.next().unwrap() as u32);
        let mut ap = a.map_coeff(|c| c.to_finite_field(&f), f.clone());
        let mut bp = b.map_coeff(|c| c.to_finite_field(&f), f.clone());

        for var in vars.iter() {
            if a.degree(*var) == E::zero() || b.degree(*var) == E::zero() {
                continue;
            }

            while ap.degree(*var) != a.degree(*var) || bp.degree(*var) != b.degree(*var) {
                debug!("Variable bounds failed due to bad prime");

                let Some(p) = u32::try_from_integer(primes.next().unwrap().into()) else {
                    panic!("Ran out of primes for gcd var bound detection.\ngcd({a},{b})");
                };

                f = Zp::new(p);
                ap = a.map_coeff(|c| c.to_finite_field(&f), f.clone());
                bp = b.map_coeff(|c| c.to_finite_field(&f), f.clone());
            }

            let vvars: SmallVec<[usize; INLINED_EXPONENTS]> =
                vars.iter().filter(|i| *i != var).cloned().collect();
            bounds[*var] = MultivariatePolynomial::get_gcd_var_bound(&ap, &bp, &vvars, *var);

            // evaluate at every other variable at one, if they are present
            /*if loose_bounds
                .iter()
                .enumerate()
                .all(|(v, b)| *b == E::zero() || v == *var)
            {
                continue;
            }

            let mut a1 = a.zero();
            let mut exp = vec![E::zero(); a.nvars()];
            for m in a {
                exp[*var] = m.exponents[*var];
                a1.append_monomial(m.coefficient.clone(), &exp);
            }

            let mut b1 = b.zero();
            for m in b {
                exp[*var] = m.exponents[*var];
                b1.append_monomial(m.coefficient.clone(), &exp);
            }

            if a1.degree(*var) == a.degree(*var) && b1.degree(*var) == b.degree(*var) {
                let bound = a1.gcd(&b1).degree(*var);
                if bound < tight_bounds[*var] {
                    tight_bounds[*var] = bound;
                }
            }*/
        }

        bounds
    }

    fn normalize(a: MultivariatePolynomial<Self, E>) -> MultivariatePolynomial<Self, E> {
        if a.lcoeff().is_negative() { -a } else { a }
    }
}

impl<E: PositiveExponent> PolynomialGCD<E> for RationalField {
    fn heuristic_gcd(
        _a: &MultivariatePolynomial<Self, E>,
        _b: &MultivariatePolynomial<Self, E>,
    ) -> Option<(
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
    )> {
        // TODO: restructure
        None
    }

    fn gcd_multiple(f: Vec<MultivariatePolynomial<Self, E>>) -> MultivariatePolynomial<Self, E> {
        MultivariatePolynomial::repeated_gcd(f)
    }

    fn gcd(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
        vars: &[usize],
        bounds: &mut [E],
    ) -> MultivariatePolynomial<Self, E> {
        // remove the content so that the polynomials have integer coefficients
        let content = a.ring.gcd(&a.content(), &b.content());

        let a_int = a.map_coeff(|c| a.ring.div(c, &content).numerator(), Z);
        let b_int = b.map_coeff(|c| b.ring.div(c, &content).numerator(), Z);

        PolynomialGCD::gcd(&a_int, &b_int, vars, bounds).map_coeff(|c| c.to_rational(), Q)
    }

    fn get_gcd_var_bounds(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
        vars: &[usize],
    ) -> SmallVec<[E; INLINED_EXPONENTS]> {
        // remove the content so that the polynomials have integer coefficients
        let content = a.ring.gcd(&a.content(), &b.content());

        let a_int = a.map_coeff(|c| a.ring.div(c, &content).numerator(), Z);
        let b_int = b.map_coeff(|c| b.ring.div(c, &content).numerator(), Z);

        PolynomialGCD::get_gcd_var_bounds(&a_int, &b_int, vars)
    }

    fn normalize(a: MultivariatePolynomial<Self, E>) -> MultivariatePolynomial<Self, E> {
        if a.lcoeff().is_negative() { -a } else { a }
    }
}

impl<UField: FiniteFieldWorkspace, F: GaloisField<Base = FiniteField<UField>>, E: PositiveExponent>
    PolynomialGCD<E> for F
where
    FiniteField<UField>: FiniteFieldCore<UField>,
    <FiniteField<UField> as Set>::Element: Copy,
{
    fn heuristic_gcd(
        _a: &MultivariatePolynomial<Self, E>,
        _b: &MultivariatePolynomial<Self, E>,
    ) -> Option<(
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
    )> {
        None
    }

    fn gcd(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
        vars: &[usize],
        bounds: &mut [E],
    ) -> MultivariatePolynomial<Self, E> {
        assert!(!a.is_zero() || !b.is_zero());
        match MultivariatePolynomial::gcd_shape_modular(
            a,
            b,
            vars,
            bounds,
            &mut bounds
                .iter()
                .cloned()
                .collect::<SmallVec<[E; INLINED_EXPONENTS]>>(),
        ) {
            Some(x) => x,
            None => {
                // upgrade to a Galois field that is large enough
                // TODO: start at a better bound?
                // TODO: run with Zp[var]/m_i instead and use CRT
                let field = a.ring.upgrade(a.ring.get_extension_degree() as usize + 1);
                let ag = a.map_coeff(|c| a.ring.upgrade_element(c, &field), field.clone());
                let bg = b.map_coeff(|c| a.ring.upgrade_element(c, &field), field.clone());
                let g = PolynomialGCD::gcd(&ag, &bg, vars, bounds);

                // workaround for ICE https://github.com/rust-lang/rust/issues/146965
                // inline the following call: g.map_coeff(|c| a.ring.downgrade_element(c), a.ring.clone())
                let mut coefficients = Vec::with_capacity(g.coefficients.len());
                let mut exponents = Vec::with_capacity(g.exponents.len());

                for m in g.into_iter() {
                    let nc = a.ring.downgrade_element(m.coefficient);
                    if !a.ring.is_zero(&nc) {
                        coefficients.push(nc);
                        exponents.extend(m.exponents);
                    }
                }

                MultivariatePolynomial {
                    coefficients,
                    exponents,
                    ring: a.ring.clone(),
                    variables: g.variables.clone(),
                    _phantom: std::marker::PhantomData,
                }
            }
        }
    }

    fn get_gcd_var_bounds(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
        vars: &[usize],
    ) -> SmallVec<[E; INLINED_EXPONENTS]> {
        let mut tight_bounds: SmallVec<[_; INLINED_EXPONENTS]> =
            (0..a.nvars()).map(|_| E::zero()).collect();
        for var in vars {
            let vvars: SmallVec<[usize; INLINED_EXPONENTS]> =
                vars.iter().filter(|i| *i != var).cloned().collect();
            tight_bounds[*var] = MultivariatePolynomial::get_gcd_var_bound(a, b, &vvars, *var);
        }
        tight_bounds
    }

    fn gcd_multiple(f: Vec<MultivariatePolynomial<Self, E>>) -> MultivariatePolynomial<Self, E> {
        MultivariatePolynomial::repeated_gcd(f)
    }

    fn normalize(a: MultivariatePolynomial<Self, E>) -> MultivariatePolynomial<Self, E> {
        a.make_monic()
    }
}

impl<E: PositiveExponent> PolynomialGCD<E> for C {
    fn heuristic_gcd(
        _a: &MultivariatePolynomial<Self, E>,
        _b: &MultivariatePolynomial<Self, E>,
    ) -> Option<(
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
    )> {
        None
    }

    fn gcd_multiple(f: Vec<MultivariatePolynomial<Self, E>>) -> MultivariatePolynomial<Self, E> {
        MultivariatePolynomial::repeated_gcd(f)
    }

    fn gcd(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
        vars: &[usize],
        bounds: &mut [E],
    ) -> MultivariatePolynomial<Self, E> {
        PolynomialGCD::gcd(&a.to_extension(), &b.to_extension(), vars, bounds).to_complex()
    }

    fn get_gcd_var_bounds(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
        vars: &[usize],
    ) -> SmallVec<[E; INLINED_EXPONENTS]> {
        PolynomialGCD::get_gcd_var_bounds(&a.to_extension(), &b.to_extension(), vars)
    }

    fn normalize(a: MultivariatePolynomial<Self, E>) -> MultivariatePolynomial<Self, E> {
        a.make_monic()
    }
}

impl<E: PositiveExponent> PolynomialGCD<E> for AlgebraicExtension<RationalField> {
    fn heuristic_gcd(
        _a: &MultivariatePolynomial<Self, E>,
        _b: &MultivariatePolynomial<Self, E>,
    ) -> Option<(
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
    )> {
        None
    }

    fn gcd_multiple(f: Vec<MultivariatePolynomial<Self, E>>) -> MultivariatePolynomial<Self, E> {
        MultivariatePolynomial::repeated_gcd(f)
    }

    fn gcd(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
        vars: &[usize],
        bounds: &mut [E],
    ) -> MultivariatePolynomial<Self, E> {
        let content = a.ring.poly().content().inv();
        let a_integer =
            AlgebraicExtension::new(a.ring.poly().map_coeff(|c| (c * &content).numerator(), Z));
        let a_lcoeff = a_integer.poly().lcoeff();

        debug!("Zippel gcd of {} and {} % {}", a, b, a_integer);
        #[cfg(debug_assertions)]
        {
            a.check_consistency();
            b.check_consistency();
        }

        let mut primes = PrimeIteratorU64::new(u32::get_large_prime() as u64);

        let mut tight_bounds: SmallVec<[E; INLINED_EXPONENTS]> = bounds.iter().cloned().collect();

        'newfirstprime: loop {
            let Some(p) = u32::try_from_integer(primes.next().unwrap().into()) else {
                panic!("Ran out of primes for gcd reconstruction.\ngcd({a},{b})");
            };

            let mut finite_field = Zp::new(p);
            let mut algebraic_field_ff = a.ring.to_finite_field(&finite_field);

            let a_lcoeff_p = a_lcoeff.to_finite_field(&finite_field);

            if finite_field.is_zero(&a_lcoeff_p) {
                continue 'newfirstprime;
            }

            let ap = a.map_coeff(
                |c| c.to_finite_field(&finite_field),
                algebraic_field_ff.clone(),
            );
            let bp = b.map_coeff(
                |c| c.to_finite_field(&finite_field),
                algebraic_field_ff.clone(),
            );

            debug!("New first image: gcd({},{}) mod {}", ap, bp, p);

            // calculate modular gcd image
            let mut gp = match MultivariatePolynomial::gcd_shape_modular(
                &ap,
                &bp,
                vars,
                bounds,
                &mut tight_bounds,
            ) {
                Some(x) => x,
                None => {
                    debug!("Modular GCD failed: getting new prime");
                    continue 'newfirstprime;
                }
            };

            debug!("GCD suggestion: {}", gp);

            bounds[vars[0]] = gp.degree(vars[0]);

            // construct a new assumed form
            // we have to find the proper normalization
            let gfu = gp.to_univariate_polynomial_list(vars[0]);

            // find a coefficient of x1 in gf that is a monomial (single scaling)
            let mut single_scale = None;
            let mut nx = 0; // count the minimal number of samples needed
            for (i, (c, _e)) in gfu.iter().enumerate() {
                if c.nterms() > nx {
                    nx = c.nterms();
                }
                if c.nterms() == 1 {
                    single_scale = Some(i);
                }
            }

            // In the case of multiple scaling, each sample adds an
            // additional unknown, except for the first
            if single_scale.is_none() {
                let mut nx1 = (gp.nterms() - 1) / (gfu.len() - 1);
                if (gp.nterms() - 1) % (gfu.len() - 1) != 0 {
                    nx1 += 1;
                }
                if nx < nx1 {
                    nx = nx1;
                }
                debug!("Multiple scaling case: sample {} times", nx);
            }

            let gpc = gp.lcoeff_varorder(vars);
            let lcoeff_factor = gp.ring.inv(&gpc);

            // construct the gcd suggestion in Z
            // contrary to the integer case, we do not know the leading coefficient in Z
            // as it cannot easily be predicted from the two input polynomials
            // we use rational reconstruction to recover it
            let mut gm: MultivariatePolynomial<AlgebraicExtension<IntegerRing>, E> =
                MultivariatePolynomial::new(&a_integer, gp.nterms().into(), a.variables.clone());
            gm.exponents.clone_from(&gp.exponents);
            gm.coefficients = gp
                .coefficients
                .iter()
                .map(|x| {
                    a_integer.to_element(
                        gp.ring
                            .mul(x, &lcoeff_factor)
                            .poly
                            .map_coeff(|c| finite_field.to_symmetric_integer(c), Z),
                    )
                })
                .collect();

            let mut m = Integer::from_prime(&finite_field); // size of finite field

            debug!("GCD suggestion with gamma: {} mod {} ", gm, p);

            // add new primes until we can reconstruct the full gcd
            'newprime: loop {
                loop {
                    let Some(p) = u32::try_from_integer(primes.next().unwrap().into()) else {
                        panic!(
                            "Ran out of primes for gcd images.\ngcd({a},{b})\nAttempt: {gm}\n vars: {vars:?}, bounds: {bounds:?}; {tight_bounds:?}"
                        );
                    };

                    finite_field = Zp::new(p);
                    algebraic_field_ff = a.ring.to_finite_field(&finite_field);

                    let a_lcoeff_p = a_lcoeff.to_finite_field(&finite_field);

                    if !finite_field.is_zero(&a_lcoeff_p) {
                        break;
                    }
                }

                let ap = a.map_coeff(
                    |c| c.to_finite_field(&finite_field),
                    algebraic_field_ff.clone(),
                );
                let bp = b.map_coeff(
                    |c| c.to_finite_field(&finite_field),
                    algebraic_field_ff.clone(),
                );
                debug!("New image: gcd({},{})", ap, bp);

                // for the univariate case, we don't need to construct an image
                if vars.len() == 1 {
                    gp = ap.univariate_gcd(&bp);
                    if gp.degree(vars[0]) < bounds[vars[0]] {
                        // original image and variable bound unlucky: restart
                        debug!("Unlucky original image: restart");
                        continue 'newfirstprime;
                    }

                    if gp.degree(vars[0]) > bounds[vars[0]] {
                        // prime is probably unlucky
                        debug!("Unlucky current image: try new one");
                        continue 'newprime;
                    }

                    for m in gp.into_iter() {
                        if gfu.iter().all(|(_, pow)| *pow != m.exponents[vars[0]]) {
                            debug!("Bad shape: terms missing");
                            continue 'newfirstprime;
                        }
                    }
                } else {
                    let rec = if let Some(single_scale) = single_scale {
                        MultivariatePolynomial::construct_new_image_single_scale(
                            &ap,
                            &bp,
                            ap.degree(vars[0]),
                            bp.degree(vars[0]),
                            bounds,
                            single_scale,
                            &vars[1..],
                            vars[0],
                            &gfu,
                        )
                    } else {
                        MultivariatePolynomial::construct_new_image_multiple_scales(
                            &ap,
                            &bp,
                            // NOTE: different from paper where they use a.degree(..)
                            // it could be that the degree in ap is lower than that of a
                            // which means the sampling will never terminate
                            ap.degree(vars[0]),
                            bp.degree(vars[0]),
                            bounds,
                            &vars[1..],
                            vars[0],
                            &gfu,
                        )
                    };

                    match rec {
                        Ok(r) => {
                            gp = r;
                        }
                        Err(GCDError::BadOriginalImage) => continue 'newfirstprime,
                        Err(GCDError::BadCurrentImage) => continue 'newprime,
                    }
                }

                // scale the new image
                let gpc = gp.lcoeff_varorder(vars);
                gp = gp.mul_coeff(ap.ring.inv(&gpc));
                debug!("gp: {} mod {}", gp, gp.ring);

                // use chinese remainder theorem to merge coefficients and map back to Z
                // terms could be missing in gp, but not in gm (TODO: check this?)
                let mut gpi = 0;
                for t in 0..gm.nterms() {
                    let gpc = if gm.exponents(t) == gp.exponents(gpi) {
                        gpi += 1;
                        gp.coefficients[gpi - 1].clone()
                    } else {
                        ap.ring.zero()
                    };

                    let gmc_a = &mut gm.coefficients[t];

                    // apply CRT to each integer coefficient in the algebraic number ring
                    let mut gpc_pos = 0;
                    let mut gmc_pos = 0;
                    for i in 0..a.ring.poly().degree(0) {
                        let gpc =
                            if gpc_pos < gpc.poly.nterms() && i == gpc.poly.exponents(gpc_pos)[0] {
                                gpc_pos += 1;
                                Integer::from_finite_field(
                                    &finite_field,
                                    gpc.poly.coefficients[gpc_pos - 1],
                                )
                            } else {
                                Integer::zero()
                            };

                        let gpm = if gmc_pos < gmc_a.poly.nterms()
                            && i == gmc_a.poly.exponents(gmc_pos)[0]
                        {
                            gmc_pos += 1;
                            let r = &gmc_a.poly.coefficients[gmc_pos - 1];
                            if r.is_negative() { r + &m } else { r.clone() }
                        } else {
                            Integer::zero()
                        };

                        let absent = gpm.is_zero();

                        let res = Integer::chinese_remainder(
                            gpm,
                            gpc,
                            m.clone(),
                            Integer::from_prime(&finite_field),
                        );

                        if absent {
                            if !res.is_zero() {
                                gmc_a.poly.append_monomial(res, &[i]);
                                gmc_pos += 1;
                            }
                        } else {
                            assert!(!res.is_zero());
                            gmc_a.poly.coefficients[gmc_pos - 1] = res;
                        }
                    }
                }

                m *= &Integer::from_prime(&finite_field);

                debug!("gm: {} from ring {}", gm, m);

                // do rational reconstruction
                // TODO: don't try every iteration?
                let mut gc = a.zero();

                for c in &gm.coefficients {
                    let mut nc = a.ring.poly().zero();

                    for aa in &c.poly.coefficients {
                        match Rational::maximal_quotient_reconstruction(aa, &m, None) {
                            Ok(x) => nc.coefficients.push(x),
                            Err(e) => {
                                debug!("Bad rational reconstruction: {}", e);
                                // more samples!
                                continue 'newprime;
                            }
                        }
                    }

                    nc.exponents.clone_from(&c.poly.exponents);
                    gc.coefficients.push(a.ring.to_element(nc));
                }

                gc.exponents.clone_from(&gm.exponents);

                debug!("Final suggested gcd: {}", gc);
                if gc.is_one() || (a.try_div(&gc).is_some() && b.try_div(&gc).is_some()) {
                    return gc;
                }

                // if it does not divide, we need more primes
                debug!("Does not divide: more primes needed");
            }
        }
    }

    fn get_gcd_var_bounds(
        a: &MultivariatePolynomial<Self, E>,
        b: &MultivariatePolynomial<Self, E>,
        vars: &[usize],
    ) -> SmallVec<[E; INLINED_EXPONENTS]> {
        let mut bounds: SmallVec<[_; INLINED_EXPONENTS]> =
            (0..a.nvars()).map(|_| E::zero()).collect();
        let mut primes = PrimeIteratorU64::new(u32::get_large_prime() as u64);

        let mut f = Zp::new(primes.next().unwrap() as u32);
        let mut algebraic_field_ff = a.ring.to_finite_field(&f);
        let mut ap = a.map_coeff(|c| c.to_finite_field(&f), algebraic_field_ff.clone());
        let mut bp = b.map_coeff(|c| c.to_finite_field(&f), algebraic_field_ff.clone());

        for var in vars.iter() {
            if a.degree(*var) == E::zero() || b.degree(*var) == E::zero() {
                continue;
            }

            while ap.degree(*var) != a.degree(*var) || bp.degree(*var) != b.degree(*var) {
                debug!("Variable bounds failed due to bad prime");

                let Some(p) = u32::try_from_integer(primes.next().unwrap().into()) else {
                    panic!("Ran out of primes for gcd var bound detection.\ngcd({a},{b})");
                };

                f = Zp::new(p);
                algebraic_field_ff = a.ring.to_finite_field(&f);
                ap = a.map_coeff(|c| c.to_finite_field(&f), algebraic_field_ff.clone());
                bp = b.map_coeff(|c| c.to_finite_field(&f), algebraic_field_ff.clone());
            }

            let vvars: SmallVec<[usize; INLINED_EXPONENTS]> =
                vars.iter().filter(|i| *i != var).cloned().collect();
            bounds[*var] = MultivariatePolynomial::get_gcd_var_bound(&ap, &bp, &vvars, *var);
        }

        bounds
    }

    fn normalize(a: MultivariatePolynomial<Self, E>) -> MultivariatePolynomial<Self, E> {
        a.make_monic()
    }
}

pub trait InexactCoefficient {}

impl InexactCoefficient for Float {}
impl InexactCoefficient for F64 {}
impl InexactCoefficient for Complex<Float> {}
impl InexactCoefficient for Complex<F64> {}

/// Polynomial GCD functions for floating point coefficient return 1 (for now).
impl<
    T: InexactCoefficient + SelfRing + SingleFloat + std::hash::Hash + Eq + InternalOrdering,
    E: PositiveExponent,
> PolynomialGCD<E> for FloatField<T>
{
    fn heuristic_gcd(
        _a: &MultivariatePolynomial<Self, E>,
        _b: &MultivariatePolynomial<Self, E>,
    ) -> Option<(
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
        MultivariatePolynomial<Self, E>,
    )> {
        None
    }

    fn gcd_multiple(f: Vec<MultivariatePolynomial<Self, E>>) -> MultivariatePolynomial<Self, E> {
        f[0].one()
    }

    /// Returns 1 (for now).
    fn gcd(
        a: &MultivariatePolynomial<Self, E>,
        _b: &MultivariatePolynomial<Self, E>,
        _vars: &[usize],
        _bounds: &mut [E],
    ) -> MultivariatePolynomial<Self, E> {
        a.one()
    }

    fn get_gcd_var_bounds(
        a: &MultivariatePolynomial<Self, E>,
        _b: &MultivariatePolynomial<Self, E>,
        _vars: &[usize],
    ) -> SmallVec<[E; INLINED_EXPONENTS]> {
        (0..a.nvars()).map(|_| E::zero()).collect()
    }

    fn normalize(a: MultivariatePolynomial<Self, E>) -> MultivariatePolynomial<Self, E> {
        a.one()
    }
}
