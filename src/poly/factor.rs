//! Factorization methods for multivariate polynomials
//! that implement [Factorize].

use std::{cmp::Reverse, sync::Arc};

use ahash::{HashMap, HashSet, HashSetExt};
use numerica::domains::float::C;
use rand::{Rng, rng};
use tracing::debug;

use crate::{
    combinatorics::CombinationIterator,
    domains::{
        EuclideanDomain, Field, InternalOrdering, Ring, RingOps, Set,
        algebraic_number::{AlgebraicExtension, GaloisField},
        finite_field::{
            FiniteField, FiniteFieldCore, FiniteFieldWorkspace, PrimeIteratorU64, ToFiniteField,
            Zp, Zp64,
        },
        integer::{Integer, IntegerRing, Z, gcd_unsigned},
        rational::{Q, RationalField},
    },
    poly::PolyVariable,
};

use super::{LexOrder, PositiveExponent, gcd::PolynomialGCD, polynomial::MultivariatePolynomial};

/// A polynomial that can be factorized.
pub trait Factorize: Sized {
    /// Perform a square-free factorization.
    /// The output is `a_1^e1*...*a_n^e_n`
    /// where each `a_i` is relative prime.
    fn square_free_factorization(&self) -> Vec<(Self, usize)>;
    /// Factor a polynomial over its coefficient ring.
    fn factor(&self) -> Vec<(Self, usize)>;
    fn is_irreducible(&self) -> bool;
}

impl<F: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent>
    MultivariatePolynomial<F, E, LexOrder>
{
    /// Find factors that do not contain all variables.
    pub fn factor_separable(&self) -> Vec<Self> {
        let mut stripped = self.clone();

        let mut factors = vec![];
        for x in 0..self.nvars() {
            if self.degree(x) == E::zero() {
                continue;
            }

            let c = stripped.to_univariate_polynomial_list(x);
            let cs = c.into_iter().map(|x| x.0).collect();

            let gcd = PolynomialGCD::gcd_multiple(cs);

            if !gcd.is_constant() {
                stripped = stripped / &gcd;
                let mut fs = gcd.factor_separable();
                factors.append(&mut fs);
            }
        }

        factors.push(stripped);
        factors
    }

    /// Perform a square free factorization using Yun's algorithm.
    ///
    /// The characteristic of the ring must be 0 and all variables
    /// must occur in every factor.
    fn square_free_factorization_0_char(&self) -> Vec<(Self, usize)> {
        if self.is_constant() {
            if self.is_one() {
                return vec![];
            } else {
                return vec![(self.clone(), 1)];
            }
        }

        // any variable can be selected
        // select the one with the lowest degree
        let lowest_rank_var = (0..self.nvars())
            .filter_map(|x| {
                let d = self.degree(x);
                if d > E::zero() { Some((x, d)) } else { None }
            })
            .min_by_key(|a| a.1)
            .unwrap()
            .0;

        let b = self.derivative(lowest_rank_var);
        let c = self.gcd(&b);

        if c.is_one() {
            return vec![(self.clone(), 1)];
        }

        let mut factors = vec![];

        let mut w = self / &c;
        let mut y = &b / &c;

        let mut i = 1;
        while !w.is_constant() {
            let z = y - w.derivative(lowest_rank_var);
            let g = w.gcd(&z);
            w = w / &g;
            y = z / &g;

            if !g.is_one() {
                factors.push((g, i));
            }
            i += 1
        }

        factors
    }

    /// Use Newton's polygon method to test if a bivariate polynomial is irreducible.
    /// If this method returns `false`, the test is inconclusive.
    ///
    /// The polynomial must have overall factors of single variables removed.
    fn bivariate_irreducibility_test(&self) -> bool {
        /// Compute the convex hull via the Monotone chain algorithm.
        fn convex_hull(mut points: Vec<(isize, isize)>) -> Vec<(isize, isize)> {
            points.sort();
            if points.len() < 2 {
                return points;
            }

            // Cross product of o-a and o-b vectors, positive means ccw turn, negative means cw turn and 0 means collinear.
            fn cross(o: &(isize, isize), a: &(isize, isize), b: &(isize, isize)) -> isize {
                (a.0 - o.0) * (b.1 - o.1) - (a.1 - o.1) * (b.0 - o.0)
            }

            let mut lower = vec![];
            let mut upper = vec![];

            for (t, rev) in [(&mut lower, false), (&mut upper, true)] {
                for i in 0..points.len() {
                    let p = if rev {
                        points[points.len() - 1 - i]
                    } else {
                        points[i]
                    };
                    while t.len() >= 2 && cross(&t[t.len() - 2], &t[t.len() - 1], &p) <= 0 {
                        t.pop();
                    }
                    t.push(p);
                }
            }

            lower.pop();
            upper.pop();
            lower.extend(upper);
            lower
        }

        let vars: Vec<_> = (0..self.nvars())
            .filter(|v| self.degree(*v) > E::zero())
            .collect();

        if vars.len() != 2 {
            return false;
        }

        let points = self
            .exponents
            .chunks(self.nvars())
            .map(|e| (e[vars[0]].to_u32() as isize, e[vars[1]].to_u32() as isize))
            .collect();

        let hull = convex_hull(points);

        match hull.len() {
            2 => {
                let x_deg = hull[0].0.abs_diff(hull[1].0);
                let y_deg = hull[0].1.abs_diff(hull[1].1);
                gcd_unsigned(x_deg as u64, y_deg as u64) == 1
            }
            3 => {
                // the hull has the form (n, 0), (0, m), (u, v)
                let (mut n, mut m, mut u, mut v) = (-1, -1, -1, -1);
                for (x, y) in hull {
                    if x != 0 && y == 0 {
                        n = x;
                    } else if y != 0 && x == 0 {
                        m = y;
                    } else {
                        u = x;
                        v = y;
                    }
                }

                n != -1
                    && m != -1
                    && u != -1
                    && v != -1
                    && gcd_unsigned(
                        gcd_unsigned(gcd_unsigned(n as u64, m as u64), u as u64),
                        v as u64,
                    ) == 1
            }
            _ => false,
        }
    }
}

impl<R: EuclideanDomain, E: PositiveExponent> MultivariatePolynomial<R, E, LexOrder> {
    /// Check if a parse lift is possible.
    fn sparse_lift_possible(&self, factors: &[Self], order: &[usize]) -> bool {
        // check if all bivariate monomials occur in the product of factors
        let mut all_monomials = HashSet::with_capacity(self.nterms());
        for e in self.exponents.chunks(self.nvars()) {
            all_monomials.insert((e[order[0]], e[order[1]]));
        }

        let mut total = factors[0].clone();
        for f in &factors[1..] {
            total = &total * f;
        }

        let mut all_monomials_in_factors = HashSet::with_capacity(self.nterms());
        for e in total.exponents.chunks(total.nvars()) {
            all_monomials_in_factors.insert((e[order[0]], e[order[1]]));
        }

        all_monomials == all_monomials_in_factors
    }

    /// Try to sparsely lift a bivariate factorization to a multivariate factorization.
    /// Based on an algorithm by Lucks.
    fn sparse_lifting(
        &self,
        factors: &[Self],
        true_lcoeffs: &[Self],
        order: &[usize],
    ) -> Option<Vec<Self>> {
        let variables: usize = factors.iter().map(|f| f.nterms()).sum();

        // TODO: refine limit
        if variables > 1000 {
            return None;
        }

        // create a unique variable for every monomial
        let vm = self.variables.as_ref();
        let vars: Arc<Vec<_>> = Arc::new(
            (0..self.nvars() + variables)
                .map(|i| {
                    if i < self.nvars() {
                        vm[i].clone()
                    } else {
                        PolyVariable::Temporary(i)
                    }
                })
                .collect(),
        );

        // attach the proper lcoeff
        let mut factors_with_true_lcoeff = Vec::with_capacity(factors.len());
        for (b, l) in factors.iter().zip(true_lcoeffs) {
            let b_one_coeff = MultivariatePolynomial {
                exponents: b.exponents.clone(),
                coefficients: vec![b.ring.one(); b.nterms()],
                ring: b.ring.clone(),
                variables: b.variables.clone(),
                _phantom: b._phantom,
            };

            let mut bs = b_one_coeff.to_univariate_polynomial_list(order[0]);
            bs.last_mut().unwrap().0 = l.clone();

            let mut fixed_fac = self.zero();
            let mut exp = vec![E::zero(); self.nvars()];
            for (p, e) in bs {
                exp[order[0]] = e;
                fixed_fac = fixed_fac + p.mul_exp(&exp);
            }

            factors_with_true_lcoeff.push(fixed_fac);
        }

        let mut factors_grown = Vec::with_capacity(factors.len());
        let mut index = 0;
        for f in &factors_with_true_lcoeff {
            let mut exponents = vec![E::zero(); (f.nvars() + variables) * f.nterms()];

            let d = f.degree(order[0]);

            for (ge, e) in exponents
                .chunks_mut(f.nvars() + variables)
                .zip(f.exponents.chunks(f.nvars()))
            {
                ge[..f.nvars()].copy_from_slice(e);

                if d != e[order[0]] {
                    ge[f.nvars() + index] = E::one();
                    index += 1;
                }
            }

            factors_grown.push(MultivariatePolynomial {
                exponents,
                coefficients: f.coefficients.clone(),
                ring: f.ring.clone(),
                variables: vars.clone(),
                _phantom: f._phantom,
            });
        }

        let mut total = factors_grown[0].one();
        for f in &factors_grown {
            debug!("Factor {}", f);
            total = &total * f;
        }

        debug!("Total {}", total);

        // find linear relations
        let mut system = HashMap::default();
        let mut exp = vec![E::zero(); total.nvars()];
        for t in self {
            let p = system
                .entry((t.exponents[order[0]], t.exponents[order[1]]))
                .or_insert_with(|| total.zero());

            exp[..self.nvars()].copy_from_slice(t.exponents);
            exp[order[0]] = E::zero();
            exp[order[1]] = E::zero();

            p.append_monomial(t.coefficient.clone(), &exp);
        }

        for t in &total {
            let p = system
                .entry((t.exponents[order[0]], t.exponents[order[1]]))
                .or_insert_with(|| total.zero());

            exp.copy_from_slice(t.exponents);
            exp[order[0]] = E::zero();
            exp[order[1]] = E::zero();

            p.append_monomial(self.ring.neg(t.coefficient), &exp);
        }

        for (k, v) in &system {
            debug!("({},{}) = {}", k.0, k.1, v);
        }

        let mut system: Vec<_> = system.into_values().collect();
        system.retain(|x| !x.is_zero());

        let mut solve_map = vec![self.zero(); variables];
        'bigloop: while !system.is_empty() {
            // solve all equations with a single variable

            let mut has_sol = false;
            'next: for (r, v) in system.iter().enumerate() {
                let mut used_variables = vec![false; variables];

                for e in v.exponents.chunks(v.nvars()) {
                    for (vv, ee) in e[self.nvars()..].iter().enumerate() {
                        if *ee == E::one() {
                            used_variables[vv] = true;
                        } else if *ee > E::one() {
                            continue 'next;
                        }
                    }
                }

                match used_variables.iter().filter(|x| **x).count() {
                    0 => {
                        // this is a non-zero empty equation, therefore we have an inconsistency
                        debug!("Inconsistent system");
                        return None;
                    }
                    1 => {
                        // solve the equation
                        let mut coeff = self.zero();
                        let mut rhs = self.zero();
                        let var = used_variables.iter().position(|x| *x).unwrap();
                        for (e, c) in v.exponents.chunks(v.nvars()).zip(&v.coefficients) {
                            if e[self.nvars() + var] == E::one() {
                                coeff.append_monomial(self.ring.neg(c), &e[..self.nvars()]);
                            } else {
                                rhs.append_monomial(c.clone(), &e[..self.nvars()]);
                            }
                        }

                        system.remove(r);

                        let (sol, r) = rhs.quot_rem(&coeff, false);
                        if !r.is_zero() {
                            debug!("Inconsistency in sol {}; rest = {}", sol, r);
                            return None;
                        }

                        if !solve_map[var].is_zero() {
                            if sol != solve_map[var] {
                                debug!("Inconsistency in sol: {} vs {}", sol, solve_map[var]);
                                return None;
                            }
                            continue 'bigloop;
                        } else {
                            debug!("Sol x{} = {}", self.nvars() + var, sol);
                            solve_map[var] = sol;
                            has_sol = true;
                            break;
                        }
                    }
                    _ => {
                        continue 'next;
                    }
                }
            }

            if !has_sol {
                debug!("No solution found: {}", system.len());
                return None;
            }

            // we have a new solution so fill it in all equations
            for row in system.iter_mut() {
                let mut new_row = row.zero();

                for t in &*row {
                    if !t.exponents[self.nvars()..]
                        .iter()
                        .zip(&solve_map)
                        .any(|(x, s)| *x > E::zero() && !s.is_zero())
                    {
                        new_row.append_monomial(t.coefficient.clone(), t.exponents);
                        continue;
                    }

                    let mut exp = t.exponents.to_vec();
                    let mut buffer = row.one();
                    for (exp, sol) in exp[self.nvars()..].iter_mut().zip(&solve_map) {
                        if *exp > E::zero() && !sol.is_zero() {
                            // upgrade sol
                            let mut sol_larger = sol.clone();
                            buffer.unify_variables(&mut sol_larger);

                            buffer = &buffer * &sol_larger.pow(exp.to_u32() as usize);
                            *exp = E::zero();
                        }
                    }
                    new_row = new_row + (&buffer * &row.monomial(t.coefficient.clone(), exp));
                }
                debug!("new eq {}", new_row);
                *row = new_row;
            }

            system.retain(|x| !x.is_zero());
        }

        // construct the solution
        let mut factors = vec![];
        for f in factors_grown {
            let mut new_factor = self.zero();

            for t in &f {
                if !t.exponents[self.nvars()..]
                    .iter()
                    .zip(&solve_map)
                    .any(|(x, s)| *x > E::zero() && !s.is_zero())
                {
                    new_factor.append_monomial(t.coefficient.clone(), &t.exponents[..self.nvars()]);
                    continue;
                }

                let mut buffer = self.one();
                for (exp, sol) in t.exponents[self.nvars()..].iter().zip(&solve_map) {
                    if *exp > E::zero() && !sol.is_zero() {
                        buffer = &buffer * &sol.pow(exp.to_u32() as usize);
                    }
                }
                new_factor = new_factor
                    + (&buffer
                        * &self
                            .monomial(t.coefficient.clone(), t.exponents[..self.nvars()].to_vec()));
            }
            factors.push(new_factor);
        }

        Some(factors)
    }
}

impl<E: PositiveExponent> Factorize for MultivariatePolynomial<IntegerRing, E, LexOrder> {
    fn square_free_factorization(&self) -> Vec<(Self, usize)> {
        if self.is_zero() {
            return vec![];
        }

        let mut c = self.content();
        let stripped = self.clone().div_coeff(&c);

        let mut factors = vec![];

        let fs = stripped.factor_separable();

        for mut f in fs {
            // make sure f is primitive
            if f.lcoeff().is_negative() {
                c = -c;
                f = -f;
            }

            let mut nf = f.square_free_factorization_0_char();
            factors.append(&mut nf);
        }

        if !c.is_one() {
            factors.insert(0, (self.constant(c), 1));
        }

        if factors.is_empty() {
            factors.push((self.one(), 1))
        }

        factors
    }

    fn factor(&self) -> Vec<(Self, usize)> {
        let sf = self.square_free_factorization();

        let mut factors = vec![];
        let mut degrees = vec![0; self.nvars()];
        for (f, p) in sf {
            debug!("SFF {} {}", f, p);

            let mut var_count = 0;
            for (v, d) in degrees.iter_mut().enumerate() {
                *d = f.degree(v).to_u32() as usize;
                if *d > 0 {
                    var_count += 1;
                }
            }

            match var_count {
                0 | 1 => factors.extend(f.factor_reconstruct().into_iter().map(|ff| (ff, p))),
                2 => {
                    let mut order: Vec<_> = degrees
                        .iter()
                        .enumerate()
                        .filter(|(_, d)| **d > 0)
                        .collect();
                    order.sort_by_key(|o| Reverse(o.1));
                    let order: Vec<_> = order.into_iter().map(|(v, _)| v).collect();

                    factors.extend(
                        f.bivariate_factor_reconstruct(order[0], order[1])
                            .into_iter()
                            .map(|ff| (ff, p)),
                    )
                }
                _ => {
                    // select the variable with the smallest leading coefficient and the highest degree to be first
                    let mut lcoeff_length = vec![0; self.nvars()];
                    for x in f.exponents_iter() {
                        for ((lc, e), d) in lcoeff_length.iter_mut().zip(x).zip(&degrees) {
                            if e.to_i32() as usize == *d {
                                *lc += 1;
                            }
                        }
                    }

                    let first = (0..self.nvars())
                        .min_by(|a, b| {
                            lcoeff_length[*a]
                                .cmp(&lcoeff_length[*b])
                                .then_with(|| degrees[*b].cmp(&degrees[*a]))
                        })
                        .unwrap();

                    // TODO: find better order
                    let mut order: Vec<_> = degrees
                        .iter()
                        .enumerate()
                        .filter(|(_, d)| **d > 0)
                        .collect();
                    order.sort_by_key(|o| {
                        if o.0 == first {
                            Reverse(&usize::MAX)
                        } else {
                            Reverse(o.1)
                        }
                    });

                    let mut order: Vec<_> = order.into_iter().map(|(v, _)| v).collect();

                    factors.extend(
                        f.multivariate_factorization(&mut order, 10, None)
                            .into_iter()
                            .map(|ff| (ff, p)),
                    )
                }
            }
        }

        factors
    }

    fn is_irreducible(&self) -> bool {
        let mut sf = self.square_free_factorization();
        if sf.len() > 1 {
            return false;
        }

        let (f, _) = sf.pop().unwrap();

        let mut degrees = vec![0; self.nvars()];
        let mut var_count = 0;
        for (v, d) in degrees.iter_mut().enumerate() {
            *d = f.degree(v).to_u32() as usize;
            if *d > 0 {
                var_count += 1;
            }
        }

        match var_count {
            0 | 1 => f.factor_reconstruct().len() == 1,
            2 => {
                let mut order: Vec<_> = degrees
                    .iter()
                    .enumerate()
                    .filter(|(_, d)| **d > 0)
                    .collect();
                order.sort_by_key(|o| Reverse(o.1));
                let order: Vec<_> = order.into_iter().map(|(v, _)| v).collect();

                f.bivariate_factor_reconstruct(order[0], order[1]).len() == 1
            }
            _ => {
                // TODO: find better order
                let mut order: Vec<_> = degrees
                    .iter()
                    .enumerate()
                    .filter(|(_, d)| **d > 0)
                    .collect();
                order.sort_by_key(|o| Reverse(o.1));

                let mut order: Vec<_> = order.into_iter().map(|(v, _)| v).collect();

                f.multivariate_factorization(&mut order, 10, None).len() == 1
            }
        }
    }
}

impl<E: PositiveExponent> Factorize for MultivariatePolynomial<RationalField, E, LexOrder> {
    fn square_free_factorization(&self) -> Vec<(Self, usize)> {
        let c = self.content();

        let stripped = self.map_coeff(
            |coeff| {
                let coeff = self.ring.div(coeff, &c);
                debug_assert!(coeff.is_integer());
                coeff.numerator()
            },
            Z,
        );

        let fs = stripped.square_free_factorization();

        let mut factors: Vec<_> = fs
            .into_iter()
            .map(|(f, e)| (f.map_coeff(|coeff| coeff.into(), Q), e))
            .collect();

        if !c.is_one() {
            factors.push((self.constant(c), 1));
        }

        factors
    }

    fn factor(&self) -> Vec<(Self, usize)> {
        let c = self.content();

        let stripped = self.map_coeff(
            |coeff| {
                let coeff = self.ring.div(coeff, &c);
                debug_assert!(coeff.is_integer());
                coeff.numerator()
            },
            Z,
        );

        let mut factors: Vec<_> = stripped
            .factor()
            .into_iter()
            .map(|(ff, p)| (ff.map_coeff(|coeff| coeff.into(), Q), p))
            .collect();

        if !c.is_one() {
            factors.push((self.constant(c), 1));
        }

        factors
    }

    fn is_irreducible(&self) -> bool {
        let c = self.content();

        let stripped = self.map_coeff(
            |coeff| {
                let coeff = self.ring.div(coeff, &c);
                debug_assert!(coeff.is_integer());
                coeff.numerator()
            },
            Z,
        );

        stripped.is_irreducible()
    }
}

impl<E: PositiveExponent> Factorize
    for MultivariatePolynomial<AlgebraicExtension<RationalField>, E, LexOrder>
{
    fn square_free_factorization(&self) -> Vec<(Self, usize)> {
        if self.is_zero() {
            return vec![];
        }

        let c = self.content();
        let stripped = self.clone().div_coeff(&c);

        let mut factors = vec![];

        if !self.ring.is_one(&c) {
            factors.push((self.constant(c), 1));
        }

        let fs = stripped.factor_separable();

        for f in fs {
            let mut nf = f.square_free_factorization_0_char();
            factors.append(&mut nf);
        }

        if factors.is_empty() {
            factors.push((self.one(), 1))
        }

        factors
    }

    /// Perform Trager's algorithm for factorization.
    fn factor(&self) -> Vec<(Self, usize)> {
        let sf = self.square_free_factorization();

        let mut constant = self.ring.one();
        let mut full_factors = vec![];
        for (f, p) in &sf {
            if f.is_constant() {
                self.ring
                    .mul_assign(&mut constant, self.ring.pow(&f.get_constant(), *p as u64));
                continue;
            }

            let (v, s, g, n) = f.norm_impl();

            let factors = n.factor();

            if factors.len() == 1 {
                full_factors.push((f.clone(), *p));
                continue;
            }

            let mut g_f = g.to_number_field(&self.ring);

            for (f, b) in factors {
                debug!("Rational factor {}", f);
                let alpha_poly = g.variable(&self.get_vars_ref()[v]).unwrap()
                    + g.variable(&self.ring.poly().variables[0]).unwrap()
                        * &g.constant((s as u64).into());

                let f = f.to_number_field(&self.ring);

                let gcd = f.gcd(&g_f);

                g_f = g_f / &gcd;

                let g = MultivariatePolynomial::from_number_field(&gcd)
                    .replace_with_poly(v, &alpha_poly)
                    .to_number_field(&self.ring);

                let lc = g.lcoeff();
                self.ring
                    .mul_assign(&mut constant, &self.ring.pow(&lc, (b * p) as u64));

                full_factors.push((g.mul_coeff(self.ring.inv(&lc)), b * p));
            }
        }

        if !self.ring.is_one(&constant) || full_factors.is_empty() {
            full_factors.push((self.constant(constant), 1));
        }

        full_factors
    }

    fn is_irreducible(&self) -> bool {
        // TODO: improve
        self.factor().len() == 1
    }
}

impl<E: PositiveExponent> Factorize for MultivariatePolynomial<C, E, LexOrder> {
    fn square_free_factorization(&self) -> Vec<(Self, usize)> {
        if self.is_zero() {
            return vec![];
        }

        let c = self.content();
        let stripped = self.clone().div_coeff(&c);

        let mut factors = vec![];

        if !self.ring.is_one(&c) {
            factors.push((self.constant(c), 1));
        }

        let fs = stripped.factor_separable();

        for f in fs {
            let mut nf = f.square_free_factorization_0_char();
            factors.append(&mut nf);
        }

        if factors.is_empty() {
            factors.push((self.one(), 1))
        }

        factors
    }

    /// Perform Trager's algorithm for factorization.
    fn factor(&self) -> Vec<(Self, usize)> {
        self.to_extension()
            .factor()
            .into_iter()
            .map(|(f, p)| (f.to_complex(), p))
            .collect()
    }

    fn is_irreducible(&self) -> bool {
        // TODO: improve
        self.factor().len() == 1
    }
}

impl<
    UField: FiniteFieldWorkspace,
    F: GaloisField<Base = FiniteField<UField>> + PolynomialGCD<E>,
    E: PositiveExponent,
> Factorize for MultivariatePolynomial<F, E, LexOrder>
where
    FiniteField<UField>: Field + FiniteFieldCore<UField> + PolynomialGCD<u16>,
    <FiniteField<UField> as Set>::Element: Copy,
    AlgebraicExtension<<F as GaloisField>::Base>: PolynomialGCD<E>,
{
    fn square_free_factorization(&self) -> Vec<(Self, usize)> {
        let c = self.lcoeff();
        let stripped = self.clone().make_monic();

        let mut factors = vec![];
        let fs = stripped.factor_separable();

        for f in fs {
            let mut nf = f.square_free_factorization_bernardin();
            factors.append(&mut nf);
        }

        if factors.is_empty() || !self.ring.is_one(&c) {
            factors.push((self.constant(c), 1))
        }

        factors
    }

    fn factor(&self) -> Vec<(Self, usize)> {
        let sf = self.square_free_factorization();

        let mut factors = vec![];
        let mut degrees = vec![0; self.nvars()];
        for (f, p) in sf {
            debug!("SFF {} {}", f, p);

            let mut var_count = 0;
            for v in 0..self.nvars() {
                degrees[v] = f.degree(v).to_u32() as usize;
                if degrees[v] > 0 {
                    var_count += 1;
                }
            }

            match var_count {
                0 => {
                    factors.push((f, p));
                }
                1 => {
                    for (d2, f2) in f.distinct_degree_factorization() {
                        debug!("DDF {} {}", f2, d2);
                        for f3 in f2.equal_degree_factorization(d2) {
                            debug!("EDF {} {}", f3, p);
                            factors.push((f3, p));
                        }
                    }
                }
                2 => {
                    let mut order: Vec<_> = degrees
                        .iter()
                        .enumerate()
                        .filter(|(_, d)| **d > 0)
                        .collect();
                    order.sort_by_key(|o| Reverse(o.1));
                    let order: Vec<_> = order.into_iter().map(|(v, _)| v).collect();

                    factors.extend(
                        f.bivariate_factorization(order[0], order[1])
                            .into_iter()
                            .map(|ff| (ff, p)),
                    )
                }
                _ => {
                    // TODO: find better order
                    let mut order: Vec<_> = degrees
                        .iter()
                        .enumerate()
                        .filter(|(_, d)| **d > 0)
                        .collect();
                    order.sort_by_key(|o| Reverse(o.1));

                    let mut order: Vec<_> = order.into_iter().map(|(v, _)| v).collect();

                    factors.extend(
                        f.multivariate_factorization(&mut order, 10, None)
                            .into_iter()
                            .map(|ff| (ff, p)),
                    )
                }
            }
        }

        factors
    }

    fn is_irreducible(&self) -> bool {
        let mut sf = self.square_free_factorization();
        if sf.len() > 1 {
            return false;
        }

        let (f, p) = sf.pop().unwrap();

        let mut degrees = vec![0; self.nvars()];
        debug!("SFF {} {}", f, p);

        let mut var_count = 0;
        for v in 0..self.nvars() {
            degrees[v] = f.degree(v).to_u32() as usize;
            if degrees[v] > 0 {
                var_count += 1;
            }
        }

        match var_count {
            0 => true,
            1 => {
                let mut d = f.distinct_degree_factorization();
                if d.len() > 1 {
                    return false;
                }

                let (d2, f2) = d.pop().unwrap();

                f2.equal_degree_factorization(d2).len() == 1
            }
            2 => {
                let mut order: Vec<_> = degrees
                    .iter()
                    .enumerate()
                    .filter(|(_, d)| **d > 0)
                    .collect();
                order.sort_by_key(|o| Reverse(o.1));
                let order: Vec<_> = order.into_iter().map(|(v, _)| v).collect();

                f.bivariate_factorization(order[0], order[1]).len() == 1
            }
            _ => {
                // TODO: find better order
                let mut order: Vec<_> = degrees
                    .iter()
                    .enumerate()
                    .filter(|(_, d)| **d > 0)
                    .collect();
                order.sort_by_key(|o| Reverse(o.1));

                let mut order: Vec<_> = order.into_iter().map(|(v, _)| v).collect();

                f.multivariate_factorization(&mut order, 10, None).len() == 1
            }
        }
    }
}

impl<
    UField: FiniteFieldWorkspace,
    F: GaloisField<Base = FiniteField<UField>> + PolynomialGCD<E>,
    E: PositiveExponent,
> MultivariatePolynomial<F, E, LexOrder>
where
    FiniteField<UField>: Field + FiniteFieldCore<UField> + PolynomialGCD<u16>,
    <FiniteField<UField> as Set>::Element: Copy,
    AlgebraicExtension<<F as GaloisField>::Base>: PolynomialGCD<E>,
{
    /// Bernardin's algorithm for square free factorization.
    fn square_free_factorization_bernardin(&self) -> Vec<(Self, usize)> {
        if self.is_constant() {
            if self.is_one() {
                return vec![];
            } else {
                return vec![(self.clone(), 1)];
            }
        }

        let mut f = self.clone();

        let mut h = HashMap::default();
        let mut hr;
        for var in 0..self.nvars() {
            if f.degree(var) > E::zero() {
                (f, hr) = f.square_free_factorization_ff_yun(var);

                for (part, pow) in hr {
                    h.entry(pow)
                        .and_modify(|f| {
                            *f = &*f * &part;
                        })
                        .or_insert(part);
                }
            }
        }

        // take the pth root
        // the coefficients remain unchanged, since x^1/p = x
        // since the derivative in every var is 0, all powers are divisible by p
        let p = self.ring.characteristic().to_u64().unwrap() as usize;
        let mut b = f.clone();
        for es in b.exponents_iter_mut() {
            for e in es {
                if e.is_zero() {
                    continue;
                }

                if p < u32::MAX as usize {
                    debug_assert_eq!(e.to_u32() as usize % p, 0);
                    *e = *e / E::from_u32(p as u32);
                } else {
                    // at the moment exponents are limited to 32-bits
                    // so only the case where e = 0 is supported
                    assert!(*e == E::zero());
                }
            }
        }

        let mut factors = vec![];
        let sub_factors = b.square_free_factorization_bernardin();

        for (mut k, n) in sub_factors {
            for (powh, hi) in &mut h {
                if *powh < p {
                    let g = k.gcd(hi);
                    if !g.is_constant() {
                        k = k / &g;
                        *hi = &*hi / &g;
                        factors.push((g, n * p + *powh));
                    }
                }
            }

            if !k.is_constant() {
                factors.push((k, n * p));
            }
        }

        for (powh, hi) in h {
            if !hi.is_constant() {
                factors.push((hi, powh));
            }
        }

        factors
    }

    /// A modified version of Yun's square free factorization algorithm.
    fn square_free_factorization_ff_yun(&self, var: usize) -> (Self, Vec<(Self, usize)>) {
        let b = self.derivative(var);
        let mut c = self.gcd(&b);
        let mut w = self / &c;
        let mut v = &b / &c;

        let mut factors = vec![];

        let mut i = 1;
        while !w.is_constant() && i < self.ring.characteristic().to_u64().unwrap() as usize {
            let z = v - w.derivative(var);
            let g = w.gcd(&z);
            w = w / &g;
            v = z / &g;
            c = c / &w;

            if !g.is_one() {
                factors.push((g, i));
            }
            i += 1
        }

        (c, factors)
    }

    /// Perform distinct degree factorization on a monic, univariate and square-free polynomial.
    pub fn distinct_degree_factorization(&self) -> Vec<(usize, Self)> {
        let Some(var) = self.last_exponents().iter().position(|x| *x > E::zero()) else {
            return vec![(0, self.clone())]; // constant polynomial
        };

        let mut e = self.last_exponents().to_vec();
        e[var] = E::one();
        let x = self.monomial(self.ring.one(), e);

        let mut factors = vec![];
        let mut h = x.clone();
        let mut f = self.clone();
        let mut i: usize = 0;
        while !f.is_one() {
            i += 1;

            h = h.exp_mod_univariate(self.ring.size().unwrap(), &mut f);

            let mut g = f.gcd(&(&h - &x));

            if !g.is_one() {
                f = f.quot_rem_univariate(&mut g).0;
                factors.push((i, g));
            }

            if f.last_exponents()[var] < E::from_u32(2 * (i as u32 + 1)) {
                // f cannot be split more
                if !f.is_constant() {
                    factors.push((f.last_exponents()[var].to_u32() as usize, f));
                }
                break;
            }
        }

        factors
    }

    /// Perform Cantor-Zassenhaus's probabilistic algorithm for
    /// finding irreducible factors of degree `d`.
    pub fn equal_degree_factorization(&self, d: usize) -> Vec<Self> {
        let s = self.clone().make_monic();

        let Some(var) = self.last_exponents().iter().position(|x| *x > E::zero()) else {
            if d == 1 {
                return vec![s];
            } else {
                panic!("Degree mismatch for {self}: {d}");
            }
        };

        let n = self.degree(var).to_u32() as usize;

        if n == d {
            return vec![s];
        }

        let mut rng = rng();
        let mut random_poly = self.zero_with_capacity(d);
        let mut exp = vec![E::zero(); self.nvars()];

        let characteristic = self.ring.characteristic();

        // compute inverse
        let mut s_rev = s.clone();
        s_rev.reverse();
        let s_inv = s_rev.inverse_univariate(var, E::from_u32(n as u32 + 1));

        let factor = loop {
            // generate a random non-constant polynomial
            random_poly.clear();

            for i in 0..n {
                let r = self
                    .ring
                    .sample(&mut rng, (0, characteristic.to_i64().unwrap_or(i64::MAX)));
                if !self.ring.is_zero(&r) {
                    exp[var] = E::from_u32(i as u32);
                    random_poly.append_monomial(r, &exp);
                }
            }

            if random_poly.degree(var) == E::zero() {
                continue;
            }
            *random_poly.coefficients.last_mut().unwrap() = self.ring.one();

            let g = random_poly.gcd(&s);

            if !g.is_one() {
                break g;
            }

            let b = if self.ring.characteristic() == 2 {
                let max = self.ring.get_extension_degree() as usize * d;

                let mut b = random_poly.clone();
                let mut vcur = b.clone();

                for _ in 1..max {
                    vcur = (&vcur * &vcur).rem(&s);
                    b = b + vcur.clone();
                }

                b
            } else {
                // TODO: use Frobenius map and modular composition to prevent computing large exponent poly^(p^d)
                let p = self.ring.size().unwrap();
                random_poly.exp_mod_univariate_fast(
                    var,
                    &(&p.pow(d as u64) - &1i64.into()) / &2i64.into(),
                    &s,
                    &s_inv,
                ) - self.one()
            };

            if b.is_constant() {
                continue;
            }

            let g = b.gcd(&s);

            if !g.is_one() && g != s {
                break g;
            }
        };

        let mut factors = factor.equal_degree_factorization(d);
        factors.extend((self / &factor).equal_degree_factorization(d));
        factors
    }

    /// Perform distinct and equal degree factorization on a square-free univariate polynomial.
    fn factor_distinct_equal_degree(&self) -> Vec<Self> {
        let mut factors = vec![];
        for (d2, f2) in self.distinct_degree_factorization() {
            debug!("DDF {} {}", f2, d2);
            for f3 in f2.equal_degree_factorization(d2) {
                debug!("EDF {}", f3);
                factors.push(f3);
            }
        }
        factors
    }

    /// Bernardin's algorithm based on
    /// "A new bivariate Hensel lifting algorithm for n factors"
    /// by Garrett Paluck. The formulation of the algorithm in other sources contain serious errors.
    // TODO: merge with an almost similar method for the integer case. A modification that needs
    // to be made here is to make the lcoeff_y=0 monic
    fn bivariate_hensel_lift_bernardin(
        &self,
        interpolation_var: usize,
        lcoeff: &Self,
        univariate_factors: &[Self],
        iterations: usize,
    ) -> Vec<Self> {
        let y_poly = self.to_univariate_polynomial_list(interpolation_var);

        // add the leading coefficient as a first factor
        let mut factors = vec![lcoeff.replace(interpolation_var, &self.ring.zero())];
        factors.extend_from_slice(univariate_factors);

        // extract coefficients in y
        let mut u: Vec<_> = factors
            .iter()
            .map(|f| {
                let mut dense = vec![self.zero(); iterations + 1];
                dense[0] = f.clone();
                dense
            })
            .collect();

        // update the first polynomial as it may contain y, since it's lcoeff
        let y_lcoeff = lcoeff.to_univariate_polynomial_list(interpolation_var);
        for (p, e) in y_lcoeff {
            u[0][e.to_u32() as usize] = p;
        }

        let mut p = u.clone();
        let mut cur_p = p[0][0].clone();
        for x in &mut p.iter_mut().skip(1) {
            cur_p = cur_p * &x[0];
            x[0] = cur_p.clone();
        }

        let delta = Self::diophantine_univariate(&mut factors, &self.one());

        for k in 1..iterations {
            // extract the coefficient required to compute the error in y^k
            // computed using a convolution
            p[0][k] = u[0][k].clone();
            for i in 1..factors.len() {
                for j in 0..k {
                    p[i][k] = &p[i][k] + &(&p[i - 1][k - j] * &u[i][j]);
                }
            }

            // find the kth power of y in f
            // since we compute the error per power of y, we cannot stop on a 0 error
            let e = if let Some((v, _)) = y_poly.iter().find(|e| e.1.to_u32() as usize == k) {
                v - &p.last().unwrap()[k]
            } else {
                -p.last().unwrap()[k].clone()
            };

            if e.is_zero() {
                continue;
            }

            for ((dp, f), d) in u.iter_mut().zip(factors.iter_mut()).zip(&delta) {
                dp[k] = &dp[k] + &(d * &e).quot_rem_univariate(f).1;
            }

            // update the coefficients with the new y^k contributions
            // note that the lcoeff[k] contribution is not new
            let mut t = self.zero();
            for i in 1..factors.len() {
                t = &u[i][0] * &t + &u[i][k] * &p[i - 1][0];
                p[i][k] = &p[i][k] + &t;
            }
        }

        // convert dense polynomials to multivariate polynomials
        u.into_iter()
            .map(|ts| {
                let mut new_poly = self.zero_with_capacity(ts.len());
                for (i, mut f) in ts.into_iter().enumerate() {
                    for x in f.exponents_iter_mut() {
                        x[interpolation_var] = E::from_u32(i as u32);
                    }
                    new_poly = new_poly + f;
                }
                new_poly
            })
            .collect()
    }

    /// Compute the bivariate factorization of a square-free polynomial.
    fn bivariate_factorization(&self, main_var: usize, interpolation_var: usize) -> Vec<Self> {
        assert!(main_var != interpolation_var);

        if self.bivariate_irreducibility_test() {
            return vec![self.clone()];
        }

        // check for problems arising from canceling terms in the derivative
        let der = self.derivative(main_var);
        if der.is_zero() {
            return self.bivariate_factorization(interpolation_var, main_var);
        }

        let g = self.gcd(&der);
        if !g.is_constant() {
            let mut factors = g.bivariate_factorization(main_var, interpolation_var);
            factors.extend((self / &g).bivariate_factorization(main_var, interpolation_var));
            return factors;
        }

        let mut sample_point = self.ring.zero();
        let mut uni_f = self.replace(interpolation_var, &sample_point);

        let mut i = 0;
        let mut rng = rng();
        loop {
            if self.ring.size() == Some(i.into()) {
                let field = self
                    .ring
                    .upgrade(self.ring.get_extension_degree().to_u64().unwrap() as usize + 1);

                debug!(
                    "Upgrading to Galois field with exponent {}",
                    field.get_extension_degree()
                );

                let s_l = self.map_coeff(|c| self.ring.upgrade_element(c, &field), field.clone());

                let facs = s_l.bivariate_factorization(main_var, interpolation_var);

                return facs
                    .into_iter()
                    .map(|f| f.map_coeff(|c| self.ring.downgrade_element(c), self.ring.clone()))
                    .collect();
            }

            if self.degree(main_var) == uni_f.degree(main_var)
                && uni_f.gcd(&uni_f.derivative(main_var)).is_constant()
            {
                break;
            }

            // TODO: sample simple points first
            sample_point = self.ring.sample(&mut rng, (0, i));
            uni_f = self.replace(interpolation_var, &sample_point);
            i += 1;
        }

        let mut d = self.degree(interpolation_var).to_u32();

        let shifted_poly = if !self.ring.is_zero(&sample_point) {
            self.shift_var_cached(interpolation_var, &sample_point)
        } else {
            self.clone()
        };

        let fs = uni_f.factor_distinct_equal_degree();

        let mut lcoeff = shifted_poly.lcoeff_last_varorder(&[main_var, interpolation_var]);
        let mut lc_d = lcoeff.degree(interpolation_var).to_u32();

        let iter = (d + lc_d + 1) as usize;
        let mut factors =
            shifted_poly.bivariate_hensel_lift_bernardin(interpolation_var, &lcoeff, &fs, iter);

        factors.swap_remove(0); // remove the lcoeff

        let mut rec_factors = vec![];
        // factor recombination
        let mut s = 1;

        let mut rest = shifted_poly;
        'len: while 2 * s <= factors.len() {
            let mut fs = CombinationIterator::new(factors.len(), s);
            while let Some(cs) = fs.next() {
                // TODO: multiply in the leading coefficient here,
                // then we can skip the Pade approximation and reduce the
                // number of iterations in the Hensel lifting to d + 1, like in the integer case?
                let mut g = rest.constant(rest.lcoeff());
                for (i, f) in factors.iter().enumerate() {
                    if cs.contains(&i) {
                        g = (&g * f).mod_var(interpolation_var, E::from_u32(iter as u32 + 1));
                    }
                }

                let y_polys: Vec<_> = g
                    .to_univariate_polynomial_list(main_var)
                    .into_iter()
                    .map(|(x, _)| x)
                    .collect();

                let mut g_lcoeff = Self::lcoeff_reconstruct(&y_polys, d, lc_d);
                g = (&g * &g_lcoeff)
                    .mod_var(interpolation_var, E::from_u32(d + 1))
                    .make_monic();

                let (h, r) = rest.quot_rem(&g, true);

                if r.is_zero() {
                    rec_factors.push(g);

                    for i in cs.iter().rev() {
                        factors.remove(*i);
                    }

                    rest = h;
                    lcoeff = lcoeff.quot_rem_univariate(&mut g_lcoeff).0;
                    lc_d = lcoeff.degree(interpolation_var).to_u32();
                    d = rest.degree(interpolation_var).to_u32();

                    continue 'len;
                }
            }

            s += 1;
        }

        rec_factors.push(rest);

        if !self.ring.is_zero(&sample_point) {
            for x in &mut rec_factors {
                // shift the polynomial to y - sample
                *x = x.shift_var_cached(interpolation_var, &self.ring.neg(&sample_point));
            }
        }

        rec_factors
    }

    /// Reconstruct the leading coefficient using a Pade approximation with numerator degree `deg_n` and
    /// denominator degree `deg_d`. The resulting denominator should be a factor of the leading coefficient.
    fn lcoeff_reconstruct(coeffs: &[Self], deg_n: u32, deg_d: u32) -> Self {
        let mut lcoeff = coeffs[0].constant(coeffs[0].ring.one());
        for x in coeffs {
            let d = x.rational_approximant_univariate(deg_n, deg_d).unwrap().1;
            if !d.is_one() {
                let g = d.gcd(&lcoeff);
                lcoeff = lcoeff * &(d / &g);
            }
        }
        lcoeff
    }

    /// Sort the bivariate factors based on their univariate image so that they are
    /// aligned between the different vars.
    fn canonical_sort(
        biv_polys: &[Self],
        replace_var: usize,
        sample_points: &[(usize, <F as Set>::Element)],
    ) -> Vec<(Self, <F as Set>::Element, Self)> {
        let mut univariate_factors = biv_polys
            .iter()
            .map(|f| {
                let mut u = f.clone();
                for (v, p) in sample_points {
                    if *v == replace_var {
                        u = u.replace(*v, p);
                    }
                }

                (f.clone(), u.lcoeff(), u.make_monic())
            })
            .collect::<Vec<_>>();
        univariate_factors.sort_by(|(_, _, a), (_, _, b)| {
            a.exponents
                .cmp(&b.exponents)
                .then(a.coefficients.internal_cmp(&b.coefficients))
        });

        univariate_factors
    }

    /// Precompute the leading coefficients of the polynomial factors, using an
    /// adapted version of Kaltofen's algorithm that has modifications of Martin Lee and Stanislav Poslavsky.
    fn lcoeff_precomputation(
        &self,
        bivariate_factors: &[Self],
        sample_points: &[(usize, <F as Set>::Element)],
        order: &[usize],
    ) -> Result<(Vec<Self>, Vec<Self>), usize> {
        let lcoeff = self.univariate_lcoeff(order[0]);
        let sqf = lcoeff.square_free_factorization();

        let mut lcoeff_square_free = self.one();
        for (f, _) in &sqf {
            lcoeff_square_free = lcoeff_square_free * f;
        }

        let sorted_main_factors = Self::canonical_sort(bivariate_factors, order[1], sample_points);

        let mut true_lcoeffs: Vec<_> = sorted_main_factors
            .iter()
            .map(|(_, u, _)| self.constant(u.clone()))
            .collect();

        let main_bivariate_factors: Vec<_> =
            sorted_main_factors.into_iter().map(|(f, _, _)| f).collect();

        let mut lcoeff_left = lcoeff.clone();
        for f in &true_lcoeffs {
            lcoeff_left = lcoeff_left / f;
        }

        // TODO: smarter ordering
        for (i, &var) in order[1..].iter().enumerate() {
            if lcoeff_left.is_one() {
                break;
            }

            if lcoeff_left.degree(var).is_zero() {
                continue;
            }

            // only construct factors that depend on var
            let c = lcoeff_square_free.univariate_content(var);
            // make sure that the content removal does not change the unit
            let mut c_eval = c.clone();
            for (v, p) in sample_points {
                c_eval = c_eval.replace(*v, p);
            }

            let lcoeff_square_free_pp = &lcoeff_square_free / &c * &c_eval;
            debug!("Content-free lcsqf {}", lcoeff_square_free_pp);

            // check if the evaluated leading coefficient remains square free
            let mut poly_eval = lcoeff_square_free_pp.clone();
            for (v, p) in sample_points {
                if *v != var {
                    poly_eval = poly_eval.replace(*v, p);
                }
            }
            let sqf = poly_eval.square_free_factorization();
            if sqf.len() != 1 || sqf[0].1 != 1 {
                debug!("Polynomial is not square free: {}", poly_eval);
                return Err(main_bivariate_factors.len());
            }

            let bivariate_factors = if var == order[1] {
                main_bivariate_factors.to_vec()
            } else {
                let mut poly_eval = self.clone();
                for (v, p) in sample_points {
                    if *v != var {
                        poly_eval = poly_eval.replace(*v, p);
                    }
                }

                if poly_eval.degree(order[0]) != self.degree(order[0])
                    || poly_eval.degree(var) != self.degree(var)
                    || poly_eval.univariate_lcoeff(order[0]).degree(var) != lcoeff.degree(var)
                {
                    debug!("Bad sample for reconstructing lcoeff: degrees do not match");
                    return Err(main_bivariate_factors.len());
                }

                let bivariate_factors: Vec<_> =
                    poly_eval.factor().into_iter().map(|(f, _)| f).collect();

                if bivariate_factors.len() != main_bivariate_factors.len() {
                    return Err(bivariate_factors.len().min(main_bivariate_factors.len()));
                }

                Self::canonical_sort(&bivariate_factors, var, sample_points)
                    .into_iter()
                    .map(|(f, _, _)| f)
                    .collect()
            };

            let square_free_lc_biv_factors: Vec<_> = bivariate_factors
                .iter()
                .map(|f| f.univariate_lcoeff(order[0]).square_free_factorization())
                .collect();

            let basis = Self::gcd_free_basis(
                square_free_lc_biv_factors
                    .iter()
                    .flatten()
                    .map(|x| x.0.clone())
                    .filter(|x| !x.is_constant())
                    .collect(),
            );

            if basis.is_empty() {
                continue;
            }

            let lifted = if basis.len() == 1 {
                vec![lcoeff_square_free_pp.clone()]
            } else {
                let mut new_order = order.to_vec();
                new_order.swap(1, i + 1);
                new_order.remove(0);

                lcoeff_square_free_pp.multivariate_hensel_lift_with_auto_lcoeff_fixing(
                    &basis,
                    sample_points,
                    &new_order,
                )
            };

            for (l, fac) in true_lcoeffs.iter_mut().zip(&square_free_lc_biv_factors) {
                let mut contrib = self.one();
                for (full, b) in lifted.iter().zip(&basis) {
                    // check if a GCD-free basis element is a factor of the leading coefficient of this bivariate factor
                    if let Some((_, m)) = fac.iter().find(|(f, _)| f == b || f.try_div(b).is_some())
                    {
                        for _ in 0..*m {
                            contrib = &contrib * full;
                        }
                    }
                }

                let g = contrib.gcd(l);
                let mut new = contrib / &g;

                // make sure the new part keeps the desired image coeff intact
                let mut b_lc_eval = new.clone();
                for (v, p) in sample_points {
                    b_lc_eval = b_lc_eval.replace(*v, p);
                }

                new = new / &b_lc_eval;

                *l = &*l * &new;
                lcoeff_left = &lcoeff_left / &new;
            }
        }

        if !lcoeff_left.is_one() {
            panic!(
                "Could not reconstruct leading coefficient of {self}: order={order:?}, samples={sample_points:?} Rest = {lcoeff_left}"
            );
        }

        Ok((main_bivariate_factors, true_lcoeffs))
    }

    fn multivariate_hensel_lift_with_auto_lcoeff_fixing(
        &self,
        factors: &[Self],
        sample_points: &[(usize, <F as Set>::Element)],
        order: &[usize],
    ) -> Vec<Self> {
        let lcoeff = self.univariate_lcoeff(order[0]);

        if lcoeff.is_constant() {
            // the factors should be properly normalized
            let (mut uni, delta) =
                Self::univariate_diophantine_field(factors, order, sample_points);
            return self.multivariate_hensel_lifting(
                factors,
                &mut uni,
                &delta,
                sample_points,
                None,
                order,
                1,
            );
        }

        // repeat the leading coefficient for every factor so that the leading coefficient is known
        let padded_lcoeffs = vec![lcoeff.clone(); factors.len()];

        let mut self_adjusted = self.clone();
        for _ in 1..factors.len() {
            self_adjusted = &self_adjusted * &lcoeff;
        }

        // set the proper lc
        let mut lc_var_eval = lcoeff.clone();
        for (v, p) in sample_points {
            if *v != order[0] {
                lc_var_eval = lc_var_eval.replace(*v, p);
            }
        }

        let adjusted_factors: Vec<_> = factors
            .iter()
            .map(|f| f.clone().make_monic() * &lc_var_eval)
            .collect();

        let (mut uni, delta) =
            Self::univariate_diophantine_field(&adjusted_factors, order, sample_points);
        self_adjusted
            .multivariate_hensel_lifting(
                &adjusted_factors,
                &mut uni,
                &delta,
                sample_points,
                Some(&padded_lcoeffs),
                order,
                1,
            )
            .into_iter()
            .map(|f| {
                let c = f.univariate_content(order[0]);
                f / &c
            })
            .collect()
    }

    fn univariate_diophantine_field(
        factors: &[Self],
        order: &[usize],
        sample_points: &[(usize, <F as Set>::Element)],
    ) -> (Vec<Self>, Vec<Self>) {
        // produce univariate factors and univariate delta
        let mut univariate_factors = factors.to_vec();
        for f in &mut univariate_factors {
            for (v, s) in sample_points {
                if order[0] != *v {
                    *f = f.replace(*v, s);
                }
            }
        }

        let univariate_deltas = Self::diophantine_univariate(
            &mut univariate_factors,
            &factors[0].constant(factors[0].ring.one()),
        );

        (univariate_factors, univariate_deltas)
    }

    /// Perform multivariate factorization on a square-free polynomial.
    fn multivariate_factorization(
        &self,
        order: &mut [usize],
        mut coefficient_upper_bound: u64,
        max_bivariate_factors: Option<usize>,
    ) -> Vec<Self> {
        if let Some(m) = max_bivariate_factors
            && m == 1
        {
            return vec![self.clone()];
        }

        // check for problems arising from canceling terms in the derivative
        let der = self.derivative(order[0]);
        if der.is_zero() {
            let mut new_order = order.to_vec();
            let v = new_order.remove(0);
            new_order.push(v);
            return self.multivariate_factorization(
                &mut new_order,
                coefficient_upper_bound,
                max_bivariate_factors,
            );
        }

        let g = self.gcd(&der);
        if !g.is_constant() {
            let mut factors =
                g.multivariate_factorization(order, coefficient_upper_bound, max_bivariate_factors);
            factors.extend((self / &g).multivariate_factorization(
                order,
                coefficient_upper_bound,
                max_bivariate_factors,
            ));
            return factors;
        }

        // select a suitable evaluation point
        let mut sample_points: Vec<_> = order[1..].iter().map(|i| (*i, self.ring.zero())).collect();
        let mut uni_f;
        let mut biv_f;
        let mut rng = rng();
        let degree = self.degree(order[0]);

        let uni_lcoeff = self.univariate_lcoeff(order[0]);

        let mut content_fail_count = 0;
        let mut sample_fail = Integer::zero();
        'new_sample: loop {
            sample_fail += &1.into();

            if &sample_fail * &2.into() > self.ring.size().unwrap() {
                // the field is too small, upgrade
                let field = self
                    .ring
                    .upgrade(self.ring.get_extension_degree().to_u64().unwrap() as usize + 1);

                debug!(
                    "Upgrading to Galois field with exponent {}",
                    field.get_extension_degree()
                );

                let s_l = self.map_coeff(|c| self.ring.upgrade_element(c, &field), field.clone());

                let facs = s_l.multivariate_factorization(
                    order,
                    coefficient_upper_bound,
                    max_bivariate_factors,
                );

                return facs
                    .into_iter()
                    .map(|f| f.map_coeff(|c| self.ring.downgrade_element(c), self.ring.clone()))
                    .collect();
            }

            for s in &mut sample_points {
                s.1 = self
                    .ring
                    .nth(rng.random_range(0..=coefficient_upper_bound).into());
            }

            biv_f = self.clone();
            for ((v, s), rem_var) in sample_points[1..].iter().zip(&order[1..]).rev() {
                biv_f = biv_f.replace(*v, s);
                if biv_f.degree(*rem_var) != self.degree(*rem_var) {
                    coefficient_upper_bound += 10;
                    continue 'new_sample;
                }
            }

            // requirement for leading coefficient precomputation
            if biv_f.univariate_lcoeff(order[0]).degree(order[1]) != uni_lcoeff.degree(order[1]) {
                debug!(
                    "Degree of x{} in leading coefficient of bivariate image is wrong",
                    order[1]
                );
                coefficient_upper_bound += 10;
                continue 'new_sample;
            }

            let biv_df = biv_f.derivative(order[0]);

            uni_f = biv_f.replace(sample_points[0].0, &sample_points[0].1);
            let uni_df = uni_f.derivative(order[0]);

            if degree == biv_f.degree(order[0])
                && degree == uni_f.degree(order[0])
                && biv_f.gcd(&biv_df).is_constant()
                && uni_f.gcd(&uni_df).is_constant()
            {
                if !biv_f.univariate_content(order[0]).is_one() {
                    content_fail_count += 1;

                    debug!("Univariate content is not one");
                    if content_fail_count == 4 {
                        // it is likely that we will always find content for this variable ordering, so change the
                        // second variable
                        // TODO: is this guaranteed to work or should we also change the first variable?
                        let sec_var = order[1];
                        order.copy_within(2..order.len(), 1);
                        order[order.len() - 1] = sec_var;

                        for ((vs, _), v) in sample_points.iter_mut().zip(&order[1..]) {
                            *vs = *v;
                        }

                        debug!("Changed the second variable to {}", order[1]);
                        content_fail_count = 0;
                    }
                } else {
                    break;
                }
            }

            coefficient_upper_bound += 10;
        }

        for (v, s) in &sample_points {
            debug!("Sample point {}={}", v, self.ring.printer(s));
        }

        let bivariate_factors = biv_f.bivariate_factorization(order[0], order[1]);

        if bivariate_factors.len() == 1 {
            // the polynomial is irreducible
            return vec![self.clone()];
        }

        if let Some(max) = max_bivariate_factors
            && bivariate_factors.len() > max
        {
            return self.multivariate_factorization(
                order,
                coefficient_upper_bound,
                max_bivariate_factors,
            );
        }

        let (sorted_biv_factors, true_lcoeffs) =
            match self.lcoeff_precomputation(&bivariate_factors, &sample_points, order) {
                Ok((sorted_biv_factors, true_lcoeffs)) => (sorted_biv_factors, true_lcoeffs),
                Err(max_biv) => {
                    // the leading coefficient computation failed because the bivaraite factorization was wrong
                    // try again with other sample points and a better bound
                    return self.multivariate_factorization(
                        order,
                        coefficient_upper_bound + 10,
                        Some(max_biv),
                    );
                }
            };

        for (b, l) in sorted_biv_factors.iter().zip(&true_lcoeffs) {
            debug!("Bivariate factor {} with true lcoeff {}", b, l);
        }

        let factorization =
            if let Some(f) = self.sparse_lifting(&sorted_biv_factors, &true_lcoeffs, order) {
                f
            } else {
                let (mut uni, delta) = MultivariatePolynomial::univariate_diophantine_field(
                    &sorted_biv_factors,
                    order,
                    &sample_points,
                );

                self.multivariate_hensel_lifting(
                    &sorted_biv_factors,
                    &mut uni,
                    &delta,
                    &sample_points,
                    Some(&true_lcoeffs),
                    order,
                    2,
                )
            };

        // test the factorization
        let mut test = self.one();
        for f in &factorization {
            debug!("Factor = {}", f);
            test = &test * f;
        }

        if &test == self {
            factorization
        } else {
            debug!(
                "No immediate factorization of {} for sample points {:?}",
                self, sample_points
            );

            // the bivariate factorization has too many factors, try again with other sample points
            self.multivariate_factorization(
                order,
                coefficient_upper_bound + 10,
                Some(max_bivariate_factors.unwrap_or(bivariate_factors.len()) - 1),
            )
        }
    }
}

impl<F: Field, E: PositiveExponent> MultivariatePolynomial<F, E, LexOrder> {
    fn multivariate_diophantine(
        univariate_deltas: &[Self],
        univariate_factors: &mut [Self],
        prods: &[Self],
        error: &Self,
        order: &[usize],
        sample_points: &[(usize, F::Element)],
        degrees: &[usize],
        mod_vars: &[MultivariatePolynomial<F, E, LexOrder>],
    ) -> Vec<Self> {
        if order.len() == 1 {
            return univariate_deltas
                .iter()
                .zip(univariate_factors)
                .map(|(d, f)| (d * error).quot_rem_univariate(f).1)
                .collect();
        }

        let last_var = *order.last().unwrap();

        let shift = &sample_points.iter().find(|s| s.0 == last_var).unwrap().1;

        let prods_mod = prods
            .iter()
            .map(|f| f.replace(last_var, shift))
            .collect::<Vec<_>>();
        let error_mod = error.replace(last_var, shift);

        debug!("dioph e[x{}^0] = {}", last_var, error_mod);

        let mut deltas = Self::multivariate_diophantine(
            univariate_deltas,
            univariate_factors,
            &prods_mod,
            &error_mod,
            &order[..order.len() - 1],
            sample_points,
            &degrees[..order.len() - 1],
            &mod_vars[..order.len() - 1],
        );

        let mut exp = vec![E::zero(); error.nvars()];
        exp[last_var] = E::one();
        let var_pow = error
            .monomial(error.ring.one(), exp)
            .shift_var(last_var, &error.ring.neg(shift));
        let mut cur_exponent;
        let mut next_exponent = var_pow.clone();

        for j in 1..=*degrees.last().unwrap() {
            cur_exponent = next_exponent.clone();
            next_exponent = &next_exponent * &var_pow;

            let mut e = error.clone();
            for (d, p) in deltas.iter().zip(prods) {
                debug!("delta {} p {}", d, p);
                e = &e - &(d * p);

                for m in mod_vars {
                    // TODO: faster implementation for univariate divisor possible?
                    e = e.quot_rem(m, false).1;
                }

                // TODO: mod with (x-shift)^(j+1)?
                // then we cannot break on 0 error
            }

            debug!("dioph  e at x{}^{} = {}", last_var, j, e);

            if e.is_zero() {
                break;
            }

            // take the jth power of var - shift in the error
            let mut e_mod = e
                .shift_var(last_var, shift)
                .mod_var(last_var, E::from_u32(j as u32 + 1));
            for e in e_mod.exponents_iter_mut() {
                debug_assert_eq!(e[last_var], E::from_u32(j as u32));
                e[last_var] = E::zero();
            }

            debug!("dioph  e[x{}^{}] = {}", last_var, j, e_mod);

            if e_mod.is_zero() {
                continue;
            }

            let mut new_deltas = Self::multivariate_diophantine(
                univariate_deltas,
                univariate_factors,
                &prods_mod,
                &e_mod,
                &order[..order.len() - 1],
                sample_points,
                &degrees[..order.len() - 1],
                &mod_vars[..order.len() - 1],
            );

            for (d, nd) in deltas.iter_mut().zip(&mut new_deltas) {
                debug!("dioph  d[x{}^{}] = {}", last_var, j, nd);

                // multiply (y-s)^j and mod wrt all vars
                let nd = &*nd * &cur_exponent;
                *d = &*d + &nd;

                for m in mod_vars {
                    e = e.quot_rem(m, false).1;
                }
            }
        }

        deltas
    }

    fn multivariate_hensel_lifting(
        &self,
        factors: &[Self],
        univariate_factors: &mut [Self],
        univariate_deltas: &[Self],
        sample_points: &[(usize, F::Element)],
        true_lcoeffs: Option<&[Self]>,
        order: &[usize],
        start_index: usize,
    ) -> Vec<Self> {
        debug!("Hensel lift {} with order {:?}", self, order);

        let mut degrees: Vec<_> = order
            .iter()
            .map(|v| self.degree(*v).to_u32() as usize)
            .collect();

        let mut reconstructed_factors = factors.to_vec();
        for v in start_index..order.len() {
            // replace the lcoeff in x0 with the proper lcoeff
            let mut factors_with_true_lcoeff = vec![];

            if let Some(true_lcoeffs) = true_lcoeffs {
                for (b, l) in reconstructed_factors.iter().zip(true_lcoeffs) {
                    let mut lcoeff = l.clone();
                    for j in v + 1..order.len() {
                        for s in sample_points {
                            if s.0 == order[j] {
                                lcoeff = lcoeff.replace(s.0, &s.1);
                            }
                        }
                    }

                    let mut bs = b.to_univariate_polynomial_list(order[0]);
                    bs.last_mut().unwrap().0 = lcoeff;

                    let mut fixed_fac = self.zero();
                    let mut exp = vec![E::zero(); self.nvars()];
                    for (p, e) in bs {
                        exp[order[0]] = e;
                        fixed_fac = fixed_fac + p.mul_exp(&exp);
                    }

                    factors_with_true_lcoeff.push(fixed_fac);
                }
            } else {
                factors_with_true_lcoeff.extend(reconstructed_factors);
            }

            let mut f = self.clone();
            for j in v + 1..order.len() {
                for s in sample_points {
                    if s.0 == order[j] {
                        f = f.replace(s.0, &s.1);
                    }
                }
            }

            // shift the polynomial such that the evaluation point is now at 0
            // so that we can use a convolution for fast error computation
            let shift = &sample_points.iter().find(|s| s.0 == order[v]).unwrap().1;
            f = f.shift_var(order[v], shift);

            for f in &mut factors_with_true_lcoeff {
                *f = f.shift_var(order[v], shift);
            }

            let mut factor_products = vec![];
            for i in 0..factors_with_true_lcoeff.len() {
                // we cannot compute prod_i = prod_k(f_k)/f_i as the leading coefficient may
                // not have an inverse in the almost-field p^k
                let mut tot = self.one();
                for (j, b) in factors_with_true_lcoeff.iter().enumerate() {
                    if i != j {
                        tot = tot * b;
                    }
                }

                factor_products.push(tot);
            }

            reconstructed_factors = f.multivariate_hensel_step(
                univariate_deltas,
                univariate_factors,
                sample_points,
                &mut factors_with_true_lcoeff,
                &factor_products,
                &order[..=v],
                &mut degrees[..=v],
            );

            for f in &mut reconstructed_factors {
                *f = f.shift_var(order[v], &self.ring.neg(shift));
            }

            for f in &reconstructed_factors {
                debug!("Reconstructed factor {}", f);
            }
        }

        reconstructed_factors
    }

    fn multivariate_hensel_step(
        &self,
        univariate_deltas: &[Self],
        univariate_factors: &mut [Self],
        sample_points: &[(usize, F::Element)],
        factors: &mut [Self],
        prods: &[Self],
        order: &[usize],
        degrees: &mut [usize],
    ) -> Vec<Self> {
        let last_var = *order.last().unwrap();
        let last_degree = *degrees.last().unwrap();
        let y_poly = self.to_univariate_polynomial_list(last_var);

        // extract coefficients in last_var
        let mut u: Vec<_> = factors
            .iter()
            .map(|f| {
                let mut dense = vec![self.zero(); last_degree + 1];

                for (p, e) in f.to_univariate_polynomial_list(last_var) {
                    dense[e.to_u32() as usize] = p;
                }

                dense
            })
            .collect();

        // TODO: do entire initialization here?
        // the loop below cannot be cut short anyway, so it's not extra work to do it all here
        let mut p = u.clone();
        let mut cur_p = p[0][0].clone();
        for x in &mut p.iter_mut().skip(1) {
            for j in &mut *x {
                *j = &*j * &cur_p;
            }

            cur_p = x[0].clone();
        }

        let prod_mod = prods
            .iter()
            .map(|f| f.replace(last_var, &self.ring.zero()))
            .collect::<Vec<_>>();

        debug!("in shift {}", self);
        debug!("deg {:?}", degrees);

        // create the polynomials (x_i-shift_i)^deg used for modding during Hensel lifting
        let mut mod_vars = Vec::with_capacity(order.len() - 2);
        let mut exp = vec![E::zero(); self.nvars()];
        for r in order[1..order.len() - 1]
            .iter()
            .zip(&degrees[1..order.len() - 1])
        {
            let shift = &sample_points.iter().find(|s| s.0 == *r.0).unwrap().1;
            exp[*r.0] = E::one();
            let var_pow = self
                .monomial(self.ring.one(), exp.clone())
                .shift_var(*r.0, &self.ring.neg(shift))
                .pow(r.1 + 1);
            exp[*r.0] = E::zero();
            mod_vars.push(var_pow);
        }

        for k in 1..=last_degree {
            // extract the coefficient required to compute the error in y^k
            // computed using a convolution
            for i in 1..factors.len() {
                for j in 0..k {
                    p[i][k] = &p[i][k] + &(&p[i - 1][k - j] * &u[i][j]);
                }
            }

            // find the kth power of y in f
            // since we compute the error per power of y, we cannot stop on a 0 error
            let e = if let Some((v, _)) = y_poly.iter().find(|e| e.1.to_u32() as usize == k) {
                v - &p.last().unwrap()[k]
            } else {
                -p.last().unwrap()[k].clone()
            };

            debug!("hensel e[x{}^{}] = {}", last_var, k, e);

            if e.is_zero() {
                continue;
            }

            let new_delta = Self::multivariate_diophantine(
                univariate_deltas,
                univariate_factors,
                &prod_mod,
                &e,
                &order[..order.len() - 1],
                sample_points,
                &degrees[..order.len() - 1],
                &mod_vars,
            );

            // update the coefficients with the new y^k contributions
            let mut t = self.zero();

            for (i, (du, d)) in u.iter_mut().zip(&new_delta).enumerate() {
                debug!("hensel d[x{}^{}] = {}", last_var, k, d);
                du[k] = &du[k] + d;

                if i > 0 {
                    t = &du[0] * &t + d * &p[i - 1][0];
                } else {
                    t = &t + d;
                }

                p[i][k] = &p[i][k] + &t;
            }
        }

        // convert dense polynomials to multivariate polynomials
        u.into_iter()
            .map(|ts| {
                let mut new_poly = self.zero_with_capacity(ts.len());
                for (i, mut f) in ts.into_iter().enumerate() {
                    for x in f.exponents_iter_mut() {
                        debug_assert_eq!(x[last_var], E::zero());
                        x[last_var] = E::from_u32(i as u32);
                    }
                    new_poly = new_poly + f;
                }
                new_poly
            })
            .collect()
    }
}

impl<E: PositiveExponent> MultivariatePolynomial<IntegerRing, E, LexOrder> {
    /// Hensel lift a solution of `self = u * w mod p` to `self = u * w mod max_p`
    /// where `max_p` is a power of `p`.
    ///
    /// If the lifting is successful, i.e. the error is 0 at some stage,
    /// it will return `Ok((u,w))` where `u` and `w` are the true factors over
    /// the integers. If a true factorization is not possible, it returns
    /// `Err((u,w))` where `u` and `w` are monic.
    pub fn hensel_lift<UField: FiniteFieldWorkspace>(
        &self,
        mut u: MultivariatePolynomial<FiniteField<UField>, E, LexOrder>,
        mut w: MultivariatePolynomial<FiniteField<UField>, E, LexOrder>,
        gamma: Option<Integer>,
        max_p: &Integer,
    ) -> Result<(Self, Self), (Self, Self)>
    where
        FiniteField<UField>: Field + PolynomialGCD<E> + FiniteFieldCore<UField>,
        Integer: ToFiniteField<UField>,
    {
        let lcoeff = self.lcoeff(); // lcoeff % p != 0
        let mut gamma = gamma.unwrap_or(lcoeff.clone());
        let lcoeff_p = lcoeff.to_finite_field(&u.ring);
        let gamma_p = gamma.to_finite_field(&u.ring);
        let field = u.ring.clone();
        let p = field.get_prime().to_integer();

        let a = self.clone().mul_coeff(gamma.clone());

        u = u.make_monic().mul_coeff(gamma_p.clone());
        w = w.make_monic().mul_coeff(lcoeff_p.clone());

        let (_, s, t) = u.eea_univariate(&w);

        debug_assert!((&s * &u + &t * &w).is_one());

        let mut u_i = u.map_coeff(|c| field.to_symmetric_integer(c), Z);
        let mut w_i = w.map_coeff(|c| field.to_symmetric_integer(c), Z);

        // only replace the leading coefficient
        *u_i.coefficients.last_mut().unwrap() = gamma.clone();
        *w_i.coefficients.last_mut().unwrap() = lcoeff;

        let mut e = &a - &(&u_i * &w_i);

        let mut m = p.clone();

        while !e.is_zero() && &m <= max_p {
            let e_p = e.map_coeff(|c| (c / &m).to_finite_field(&field), field.clone());
            let (q, r) = (&e_p * &s).quot_rem_univariate(&mut w);
            let tau = &e_p * &t + q * &u;

            u_i = u_i
                + tau
                    .map_coeff(|c| field.to_symmetric_integer(c), Z)
                    .mul_coeff(m.clone());
            w_i = w_i
                + r.map_coeff(|c| field.to_symmetric_integer(c), Z)
                    .mul_coeff(m.clone());
            e = &a - &(&u_i * &w_i);

            m = &m * &p;
        }

        if e.is_zero() {
            let content = u_i.content();
            if !content.is_one() {
                u_i = u_i.div_coeff(&content);
                gamma = &gamma / &content;
            }

            if !gamma.is_one() {
                w_i = w_i.div_coeff(&gamma); // true division is possible in this case
            }

            Ok((u_i, w_i))
        } else {
            if !u_i.lcoeff().is_one() {
                let inv = u_i.lcoeff().mod_inverse(&m);
                u_i = u_i.map_coeff(|c| (c * &inv).symmetric_mod(&m), Z);
            }

            if !w_i.lcoeff().is_one() {
                let inv = w_i.lcoeff().mod_inverse(&m);
                w_i = w_i.map_coeff(|c| (c * &inv).symmetric_mod(&m), Z);
            }

            Err((u_i, w_i))
        }
    }

    /// Lift multiple factors by creating a binary tree and lifting each product.
    fn multi_factor_hensel_lift(
        &self,
        hs: &[MultivariatePolynomial<Zp, E, LexOrder>],
        max_p: &Integer,
    ) -> Vec<Self> {
        if hs.len() == 1 {
            if self.lcoeff().is_one() {
                return vec![self.clone()];
            } else {
                let inv = self.lcoeff().mod_inverse(max_p);
                let r = self.map_coeff(|c| (c * &inv).symmetric_mod(max_p), Z);
                return vec![r];
            }
        }

        let (gs, hs) = hs.split_at(hs.len() / 2);

        let mut g = gs[0].one();
        for x in gs {
            g = g * x;
        }

        let mut h = hs[0].one();
        for x in hs {
            h = h * x;
        }

        let (g_i, h_i) = self.hensel_lift(g, h, None, max_p).unwrap_or_else(|e| e);

        let mut factors = g_i.multi_factor_hensel_lift(gs, max_p);
        factors.extend(h_i.multi_factor_hensel_lift(hs, max_p));
        factors
    }

    /// Factor a square-free univariate polynomial over the integers by Hensel lifting factors computed over
    /// a finite field image of the polynomial.
    fn factor_reconstruct(&self) -> Vec<Self> {
        let Some(var) = self.last_exponents().iter().position(|x| *x > E::zero()) else {
            return vec![self.clone()]; // constant polynomial
        };
        let d = self.degree(var).to_u32();

        if d == 1 {
            return vec![self.clone()];
        }

        // select a suitable prime
        // we try small primes first as the distinct and equal degree algorithms
        // scale as log(p)
        let mut field;
        let mut f_p;
        let mut pi = PrimeIteratorU64::new(101);
        loop {
            let p = pi.next().unwrap();
            if p > u32::MAX as u64 {
                panic!("Ran out of primes during factorization of {self}");
            }
            let p = p as u32;

            if (&self.lcoeff() % &Integer::Single(p as i64)).is_zero() {
                continue;
            }

            field = Zp::new(p);
            f_p = self.map_coeff(|f| f.to_finite_field(&field), field.clone());
            let df_p = f_p.derivative(var);

            // check is f_p remains square-free
            if f_p.gcd(&df_p).is_one() {
                break;
            }
        }

        let hs: Vec<_> = f_p.factor_distinct_equal_degree();

        if hs.len() == 1 {
            // the polynomial is irreducible
            return vec![self.clone()];
        }

        let bound = self.coefficient_bound();
        let p: Integer = (field.get_prime() as i64).into();
        let mut max_p = p.clone();
        while max_p < bound {
            max_p = &max_p * &p;
        }

        let mut factors = self.multi_factor_hensel_lift(&hs, &max_p);

        #[cfg(debug_assertions)]
        for (h, h_p) in factors.iter().zip(&hs) {
            let hh_p = h
                .map_coeff(|c| c.to_finite_field(&field), field.clone())
                .make_monic();
            if &hh_p != h_p {
                panic!("Mismatch of lifted factor: {hh_p} vs {h_p} in {self}");
            }
        }

        let mut rec_factors = vec![];
        // factor recombination
        let mut s = 1;

        let mut rest = self.clone();
        'len: while 2 * s <= factors.len() {
            let mut fs = CombinationIterator::new(factors.len(), s);
            while let Some(cs) = fs.next() {
                // check if the constant term matches
                if rest.exponents[..rest.nvars()]
                    .iter()
                    .all(|e| *e == E::zero())
                {
                    let mut g1 = rest.lcoeff();
                    let mut h1 = rest.lcoeff();
                    for (i, f) in factors.iter().enumerate() {
                        if f.exponents[..rest.nvars()].iter().all(|x| *x == E::zero()) {
                            if cs.contains(&i) {
                                g1 = (&g1 * &f.coefficients[0]).symmetric_mod(&max_p);
                            } else {
                                h1 = (&h1 * &f.coefficients[0]).symmetric_mod(&max_p);
                            }
                        }
                    }

                    // TODO: improve check
                    // for monic factors we can do &g1 * &h1 != &rest.lcoeff() * &rest.coefficients[0]
                    if (&g1 * &h1).abs() > bound {
                        continue;
                    }
                }

                let mut g = rest.constant(rest.lcoeff());
                for (i, f) in factors.iter().enumerate() {
                    if cs.contains(&i) {
                        g = (&g * f).map_coeff(|i| i.clone().symmetric_mod(&max_p), Z);
                    }
                }
                let c = g.content();
                g = g.div_coeff(&c);

                let (h, r) = rest.quot_rem(&g, true);

                if r.is_zero() {
                    // should always happen happen when |g1|_1 * |h1|_1 <= bound
                    rec_factors.push(g);

                    for i in cs.iter().rev() {
                        factors.remove(*i);
                    }

                    let c = h.content();
                    rest = h.div_coeff(&c);

                    continue 'len;
                }
            }

            s += 1;
        }

        rec_factors.push(rest);
        rec_factors
    }

    /// Lift a solution of `poly ≡ lcoeff * univariate_factors mod y mod p^k`
    /// to `mod y^iterations mod p^k`.
    ///
    /// Univariate factors must be monic and `lcoeff_y=0` should be as well.
    fn bivariate_hensel_lift_bernardin(
        poly: &MultivariatePolynomial<FiniteField<Integer>, E, LexOrder>,
        interpolation_var: usize,
        lcoeff: &MultivariatePolynomial<FiniteField<Integer>, E, LexOrder>,
        univariate_factors: &[MultivariatePolynomial<FiniteField<Integer>, E, LexOrder>],
        iterations: usize,
        p: u32,
        k: usize,
    ) -> Vec<MultivariatePolynomial<FiniteField<Integer>, E, LexOrder>> {
        let finite_field = Zp::new(p);

        // add the leading coefficient as a first factor
        let mut factors = vec![lcoeff.replace(interpolation_var, &poly.ring.zero())];

        for f in univariate_factors {
            factors.push(f.clone());
        }

        let delta = Self::lift_diophantine_univariate(
            &mut factors,
            &poly.constant(poly.ring.one()),
            finite_field.get_prime(),
            k,
        );

        let y_poly = poly.to_univariate_polynomial_list(interpolation_var);

        // extract coefficients in y
        let mut u: Vec<_> = factors
            .iter()
            .map(|f| {
                let mut dense = vec![poly.zero(); iterations + 1];
                dense[0] = f.clone();
                dense
            })
            .collect();

        // update the first polynomial as it may contain y, since it's lcoeff
        let y_lcoeff = lcoeff.to_univariate_polynomial_list(interpolation_var);
        for (p, e) in y_lcoeff {
            u[0][e.to_u32() as usize] = p;
        }

        let mut p = u.clone();
        let mut cur_p = p[0][0].clone();
        for x in &mut p.iter_mut().skip(1) {
            cur_p = cur_p * &x[0];
            x[0] = cur_p.clone();
        }

        for k in 1..iterations {
            // extract the coefficient required to compute the error in y^k
            // computed using a convolution
            p[0][k] = u[0][k].clone();
            for i in 1..factors.len() {
                for j in 0..k {
                    p[i][k] = &p[i][k] + &(&p[i - 1][k - j] * &u[i][j]);
                }
            }

            // find the kth power of y in f
            // since we compute the error per power of y, we cannot stop on a 0 error
            let e = if let Some((v, _)) = y_poly.iter().find(|e| e.1.to_u32() as usize == k) {
                v - &p.last().unwrap()[k]
            } else {
                -p.last().unwrap()[k].clone()
            };

            if e.is_zero() {
                continue;
            }

            for ((dp, f), d) in u.iter_mut().zip(factors.iter()).zip(&delta) {
                dp[k] = &dp[k] + &(d * &e).quot_rem_univariate_monic(f).1;
            }

            // update the coefficients with the new y^k contributions
            // note that the lcoeff[k] contribution is not new
            let mut t = poly.zero();
            for i in 1..factors.len() {
                t = &u[i][0] * &t + &u[i][k] * &p[i - 1][0];
                p[i][k] = &p[i][k] + &t;
            }
        }

        // convert dense polynomials to multivariate polynomials
        u.into_iter()
            .map(|ts| {
                let mut new_poly = poly.zero_with_capacity(ts.len());
                for (i, mut f) in ts.into_iter().enumerate() {
                    for x in f.exponents_iter_mut() {
                        x[interpolation_var] = E::from_u32(i as u32);
                    }
                    new_poly = new_poly + f;
                }

                new_poly
            })
            .collect()
    }

    /// Factor a square-free bivariate polynomial over the integers.
    fn bivariate_factor_reconstruct(&self, main_var: usize, interpolation_var: usize) -> Vec<Self> {
        if self.bivariate_irreducibility_test() {
            return vec![self.clone()];
        }

        let d2 = self.degree(interpolation_var).to_u32();

        // select a suitable evaluation point, as small as possible as to not change the coefficient bound
        let mut sample_point;
        let mut uni_f;
        let mut i = 0u64;
        loop {
            sample_point = i.into();
            uni_f = self.replace(interpolation_var, &sample_point);

            if self.degree(main_var) == uni_f.degree(main_var)
                && uni_f.gcd(&uni_f.derivative(main_var)).is_constant()
            {
                break;
            }

            i += 1;
        }

        // factor the univariate polynomial
        let mut uni_fs: Vec<_> = uni_f
            .factor()
            .into_iter()
            .map(|(f, p)| {
                debug_assert_eq!(p, 1);
                f
            })
            .collect();

        // strip potential content
        uni_fs.retain_mut(|f| !f.is_constant());

        // select a suitable prime
        // we try small primes first as the distinct and equal degree algorithms
        // scale as log(p)
        let mut pi = PrimeIteratorU64::new(101);
        let mut field;
        'new_prime: loop {
            i += 1;

            let p = pi.next().unwrap();
            if p > u32::MAX as u64 {
                panic!("Ran out of primes during factorization of {self}");
            }
            let p = p as u32;

            if (&uni_f.lcoeff() % &Integer::Single(p as i64)).is_zero() {
                continue;
            }

            field = Zp::new(p);

            // make sure the factors stay coprime
            let fs_p: Vec<_> = uni_fs
                .iter()
                .map(|f| f.map_coeff(|c| c.to_finite_field(&field), field.clone()))
                .collect();

            for (j, f) in fs_p.iter().enumerate() {
                for g in &fs_p[j + 1..] {
                    if !f.gcd(g).is_one() {
                        continue 'new_prime;
                    }
                }
            }

            break;
        }

        let shifted_poly = if !sample_point.is_zero() {
            self.shift_var(interpolation_var, &sample_point)
        } else {
            self.clone()
        };

        // TODO: if bound is less than u64, we may also use Zp64 for the computation
        let bound = shifted_poly.coefficient_bound();

        let p = field.get_prime().to_integer();
        let mut max_p = p.clone();
        let mut k = 1;
        while &max_p * 2 < bound {
            max_p = &max_p * &p;
            k += 1;
        }

        let mod_field = FiniteField::<Integer>::new(max_p.clone());

        // make all factors monic, this is possible since the lcoeff is invertible mod p^k
        let uni_fs_mod: Vec<_> = uni_fs
            .iter()
            .map(|f| {
                let f1 = f.map_coeff(|c| mod_field.to_element(c.clone()), mod_field.clone());
                f1.make_monic()
            })
            .collect();

        let mut f_mod =
            shifted_poly.map_coeff(|c| c.clone().symmetric_mod(&max_p), mod_field.clone());

        // make sure the lcoeff is monic at y=0
        let inv_coeff = mod_field.inv(&uni_f.lcoeff());
        let f_mod_monic = f_mod.clone().mul_coeff(inv_coeff);
        let lcoeff_monic = f_mod_monic.lcoeff_last_varorder(&[main_var, interpolation_var]);

        let mut factors = Self::bivariate_hensel_lift_bernardin(
            &f_mod_monic,
            interpolation_var,
            &lcoeff_monic,
            &uni_fs_mod,
            (d2 + 1) as usize,
            field.get_prime(),
            k,
        );

        factors.swap_remove(0); // remove the lcoeff

        let mut rec_factors = vec![];
        // factor recombination
        let mut s = 1;

        let mut lcoeff = f_mod.lcoeff_last_varorder(&[main_var, interpolation_var]);
        let mut rest = shifted_poly;
        'len: while 2 * s <= factors.len() {
            let mut fs = CombinationIterator::new(factors.len(), s);
            while let Some(cs) = fs.next() {
                let mut g = lcoeff.clone();
                for (i, f) in factors.iter().enumerate() {
                    if cs.contains(&i) {
                        g = (&g * f).mod_var(interpolation_var, E::from_u32(d2 + 1));
                    }
                }

                // convert to integer
                let mut g_int = g.map_coeff(|c| mod_field.to_symmetric_integer(c), Z);

                let content = g_int.univariate_content(main_var);
                g_int = &g_int / &content;

                let (h, r) = rest.quot_rem(&g_int, true);

                if r.is_zero() {
                    rec_factors.push(g_int);

                    for i in cs.iter().rev() {
                        factors.remove(*i);
                    }

                    rest = h;
                    f_mod = rest.map_coeff(|c| mod_field.to_element(c.clone()), mod_field.clone());
                    lcoeff = f_mod.lcoeff_last_varorder(&[main_var, interpolation_var]);

                    continue 'len;
                }
            }

            s += 1;
        }

        rec_factors.push(rest);

        if !sample_point.is_zero() {
            for x in &mut rec_factors {
                // shift the polynomial to y - sample
                *x = x.shift_var(interpolation_var, &self.ring.neg(&sample_point));
            }
        }

        rec_factors
    }

    /// Solve a Diophantine equation over the ring `Z_p^k` using Newton iteration.
    /// All factors must be monic.
    fn lift_diophantine_univariate(
        factors: &mut [MultivariatePolynomial<FiniteField<Integer>, E, LexOrder>],
        rhs: &MultivariatePolynomial<FiniteField<Integer>, E, LexOrder>,
        p: u32,
        k: usize,
    ) -> Vec<MultivariatePolynomial<FiniteField<Integer>, E, LexOrder>> {
        let field = Zp::new(p);
        let prime: Integer = (p as u64).into();

        let mut f_p: Vec<_> = factors
            .iter()
            .map(|f| f.map_coeff(|c| c.to_finite_field(&field), field.clone()))
            .collect();
        let rhs_p = rhs.map_coeff(|c| c.to_finite_field(&field), field.clone());

        // TODO: recycle from finite field computation that must have happened earlier
        let mut delta =
            MultivariatePolynomial::<Zp, E, LexOrder>::diophantine_univariate(&mut f_p, &rhs_p);

        let mut deltas: Vec<_> = delta
            .iter()
            .map(|s| {
                s.map_coeff(
                    |c| field.to_symmetric_integer(c).to_finite_field(&rhs.ring),
                    rhs.ring.clone(),
                )
            })
            .collect();

        if k == 1 {
            return deltas;
        }

        let mut tot = rhs.constant(rhs.ring.one());
        for f in factors.iter() {
            tot = &tot * f;
        }

        let pi = factors
            .iter_mut()
            .map(|f| tot.quot_rem_univariate(f).0)
            .collect::<Vec<_>>();

        let mut m = prime.clone();

        for _ in 1..k {
            // TODO: is breaking on e=0 safe?
            let mut e = rhs.clone();
            for (dd, pp) in deltas.iter_mut().zip(&pi) {
                e = &e - &(&*dd * pp);
            }

            let e_m = e.map_coeff(|c| (c / &m).to_finite_field(&field), field.clone());

            for ((p, d_m), d) in f_p.iter_mut().zip(&mut delta).zip(deltas.iter_mut()) {
                let new_delta = (&e_m * &*d_m).quot_rem_univariate(p).1;

                *d = &*d
                    + &new_delta.map_coeff(
                        |c| (&field.to_symmetric_integer(c) * &m).to_finite_field(&rhs.ring),
                        rhs.ring.clone(),
                    );
            }

            m = &m * &prime;
        }

        deltas
    }

    /// Compute the Gelfond bound for the coefficients magnitude of every factor of this polynomial
    fn coefficient_bound(&self) -> Integer {
        let max_norm = self.coefficients.iter().map(|x| x.abs()).max().unwrap();

        let mut bound = Integer::one();
        let mut total_degree = 0;
        let mut non_zero_vars = 0;
        for v in 0..self.nvars() {
            let d = self.degree(v).to_u32() as u64;
            if d > 0 {
                non_zero_vars += 1;
                total_degree += d;
                bound *= &Integer::from(d + 1);
            }
        }

        // move the 2^n into the sqrt to prevent precision loss when converting the sqrt
        // to an integer
        bound = &bound * &Integer::Single(2).pow((total_degree * 2).saturating_sub(non_zero_vars));

        bound = &match bound {
            Integer::Single(b) => Integer::Single((b as f64).sqrt() as i64),
            Integer::Double(b) => Integer::from(rug::Integer::from(b).sqrt()),
            Integer::Large(b) => Integer::from(b.sqrt()),
        } + &1i64.into();

        &bound * &(&max_norm * &self.lcoeff().abs())
    }

    /// Sort the bivariate factors based on their univariate image so that they are
    /// aligned between the different vars.
    // TODO: merge with the implementation for finite fields as the implementation
    // is almost identical
    fn canonical_sort(
        biv_polys: &[Self],
        replace_var: usize,
        sample_points: &[(usize, Integer)],
    ) -> Vec<(Self, Integer, Self)> {
        let mut univariate_factors = biv_polys
            .iter()
            .map(|f| {
                let mut u = f.clone();
                for (v, p) in sample_points {
                    if *v == replace_var {
                        u = u.replace(*v, p);
                    }
                }

                // make sure the representative is unique
                let mut uni = u.clone().make_primitive();
                if uni.lcoeff().is_negative() {
                    uni = -uni;
                }

                (f.clone(), u.lcoeff(), uni)
            })
            .collect::<Vec<_>>();

        univariate_factors.sort_by(|(_, _, a), (_, _, b)| {
            a.exponents
                .cmp(&b.exponents)
                .then(a.coefficients.partial_cmp(&b.coefficients).unwrap())
        });

        univariate_factors
    }

    /// Precompute the leading coefficients of the polynomial factors, using an
    /// adapted version of Kaltofen's algorithm that has modifications of Martin Lee and Stanislav Poslavsky.
    ///
    // TODO: merge with the implementation for finite fields as the implementation
    // is almost identical
    fn lcoeff_precomputation(
        &self,
        bivariate_factors: &[Self],
        sample_points: &[(usize, Integer)],
        order: &[usize],
        bound: Integer,
        p: u32,
        k: usize,
    ) -> Result<(Vec<Self>, Vec<Self>), usize> {
        let lcoeff = self.univariate_lcoeff(order[0]);
        let sqf = lcoeff.square_free_factorization();

        let mut lcoeff_square_free = self.one();
        for (f, _) in &sqf {
            lcoeff_square_free = &lcoeff_square_free * f;
        }

        let sorted_main_factors = Self::canonical_sort(bivariate_factors, order[1], sample_points);

        let mut true_lcoeffs: Vec<_> = bivariate_factors.iter().map(|_| self.one()).collect();

        let mut lcoeff_left = lcoeff.clone();

        let main_bivariate_factors: Vec<_> =
            sorted_main_factors.into_iter().map(|(f, _, _)| f).collect();

        // TODO: smarter ordering
        for (i, &var) in order[1..].iter().enumerate() {
            if lcoeff_left.is_one() {
                break;
            }

            if lcoeff_left.degree(var).is_zero() {
                continue;
            }

            // only construct factors that depend on var and remove integer content and unit
            let c = lcoeff_square_free.univariate_content(var);
            let mut lcoeff_square_free_pp = &lcoeff_square_free / &c;

            // check if the evaluated leading coefficient remains square free
            let mut poly_eval = lcoeff_square_free_pp.clone();
            for (v, p) in sample_points {
                if *v != var {
                    poly_eval = poly_eval.replace(*v, p);
                }
            }

            if poly_eval.lcoeff().is_negative() {
                lcoeff_square_free_pp = -lcoeff_square_free_pp;
                poly_eval = -poly_eval;
            }
            debug!("Content-free lcsqf {}", lcoeff_square_free_pp);

            let sqf = poly_eval.square_free_factorization();
            if sqf.len() != 1 || sqf[0].1 != 1 {
                debug!("Polynomial is not square free: {}", poly_eval);
                return Err(main_bivariate_factors.len());
            }

            let bivariate_factors = if var == order[1] {
                main_bivariate_factors.to_vec()
            } else {
                let mut poly_eval = self.clone();
                for (v, p) in sample_points {
                    if *v != var {
                        poly_eval = poly_eval.replace(*v, p);
                    }
                }

                if poly_eval.degree(order[0]) != self.degree(order[0])
                    || poly_eval.degree(var) != self.degree(var)
                    || poly_eval.univariate_lcoeff(order[0]).degree(var) != lcoeff.degree(var)
                {
                    debug!("Bad sample for reconstructing lcoeff: degrees do not match");
                    return Err(main_bivariate_factors.len());
                }

                let bivariate_factors: Vec<_> = poly_eval
                    .factor()
                    .into_iter()
                    .map(|(f, _)| f)
                    // remove spurious content caused by particular evaluation point
                    .filter(|f| !f.is_constant())
                    .collect();

                if bivariate_factors.len() != main_bivariate_factors.len() {
                    return Err(bivariate_factors.len().min(main_bivariate_factors.len()));
                }

                Self::canonical_sort(&bivariate_factors, var, sample_points)
                    .into_iter()
                    .map(|(f, _, _)| f)
                    .collect()
            };

            let square_free_lc_biv_factors: Vec<_> = bivariate_factors
                .iter()
                .map(|f| {
                    let mut sff = f.univariate_lcoeff(order[0]).square_free_factorization();
                    // make sure every bivariate factor has positive lcoeff such that the product
                    // of the basis elements equals the evaluated lcoeff_square_free_pp
                    for (b, _) in &mut sff {
                        if b.lcoeff().is_negative() {
                            *b = -b.clone();
                        }
                    }
                    sff
                })
                .collect();

            let basis = Self::gcd_free_basis(
                square_free_lc_biv_factors
                    .iter()
                    .flatten()
                    .map(|x| x.0.clone())
                    .filter(|x| !x.is_constant())
                    .collect(),
            );

            if basis.is_empty() {
                continue;
            }

            let lifted = if basis.len() == 1 {
                vec![lcoeff_square_free_pp.clone()]
            } else {
                let mut new_order = order.to_vec();
                new_order.swap(1, i + 1);
                new_order.remove(0);

                lcoeff_square_free_pp.multivariate_hensel_lift_with_auto_lcoeff_fixing(
                    &basis,
                    sample_points,
                    &new_order,
                    bound.clone(),
                    p,
                    k,
                )
            };

            for (l, fac) in true_lcoeffs.iter_mut().zip(&square_free_lc_biv_factors) {
                let mut contrib = self.one();
                for (full, b) in lifted.iter().zip(&basis) {
                    // check if a GCD-free basis element is a factor of the leading coefficient of this bivariate factor
                    if let Some((_, m)) = fac.iter().find(|(f, _)| f == b || f.try_div(b).is_some())
                    {
                        for _ in 0..*m {
                            contrib = &contrib * full;
                        }
                    }
                }

                let g = contrib.gcd(l);
                let new = (contrib / &g).make_primitive();

                *l = (&*l * &new).make_primitive();

                let (q, r) = lcoeff_left.quot_rem(&new, true);
                if !r.is_zero() {
                    panic!(
                        "Problem with bivariate factor scaling in factorization of {self}: order={order:?}, samples={sample_points:?}"
                    );
                }

                lcoeff_left = q;
            }
        }

        if !lcoeff_left.is_constant() {
            panic!(
                "Could not reconstruct leading coefficient of {self}: order={order:?}, samples={sample_points:?} Rest = {lcoeff_left}"
            );
        }

        // rescale the leading coefficient factors to recover the missing content and sign
        for (f, b) in true_lcoeffs.iter_mut().zip(&main_bivariate_factors) {
            let mut b_eval = b.clone();
            for (v, p) in sample_points {
                b_eval = b_eval.replace(*v, p);
            }

            let b_lc = b_eval.lcoeff();

            let mut f_eval = f.clone();
            for (v, p) in sample_points {
                f_eval = f_eval.replace(*v, p);
            }
            let f_lc = f_eval.lcoeff();

            let (q, r) = Z.quot_rem(&b_lc, &f_lc);
            assert!(
                r.is_zero(),
                "Problem with bivariate factor scaling in factorization of {self}: order={order:?}, samples={sample_points:?}"
            );

            lcoeff_left = lcoeff_left.div_coeff(&q);
            *f = f.clone().mul_coeff(q);
        }

        if !lcoeff_left.is_one() {
            panic!(
                "Could not distribute content of {self}: order={order:?}, samples={sample_points:?} Rest = {lcoeff_left}"
            );
        }

        Ok((main_bivariate_factors, true_lcoeffs))
    }

    fn multivariate_hensel_lift_with_auto_lcoeff_fixing(
        &self,
        factors: &[Self],
        sample_points: &[(usize, Integer)],
        order: &[usize],
        bound: Integer,
        p: u32,
        k: usize,
    ) -> Vec<Self> {
        let modulus = FiniteField::<Integer>::new(bound);
        let ff = self.map_coeff(|c| modulus.to_element(c.clone()), modulus.clone());
        let factors_ff: Vec<_> = factors
            .iter()
            .map(|f| f.map_coeff(|c| modulus.to_element(c.clone()), modulus.clone()))
            .collect();
        let lcoeff = ff.univariate_lcoeff(order[0]);

        if lcoeff.is_constant() {
            // the factors should be properly normalized
            let (mut uni, delta) = MultivariatePolynomial::get_univariate_factors_and_deltas(
                &factors_ff,
                order,
                sample_points,
                p,
                k,
            );
            let h = ff.multivariate_hensel_lifting(
                &factors_ff,
                &mut uni,
                &delta,
                sample_points,
                None,
                order,
                1,
            );

            return h
                .into_iter()
                .map(|f| f.map_coeff(|c| modulus.to_symmetric_integer(c), Z))
                .collect();
        }

        // repeat the leading coefficient for every factor so that the leading coefficient is known
        let padded_lcoeffs = vec![lcoeff.clone(); factors.len()];

        let mut self_adjusted = ff;
        for _ in 1..factors_ff.len() {
            self_adjusted = &self_adjusted * &lcoeff;
        }

        // set the proper lc
        let mut lc_var_eval = lcoeff.clone();
        for (v, p) in sample_points {
            if *v != order[0] {
                lc_var_eval = lc_var_eval.replace(*v, p);
            }
        }

        let adjusted_factors: Vec<_> = factors_ff
            .into_iter()
            .map(|f| f.make_monic() * &lc_var_eval)
            .collect();

        let (mut uni, delta) = MultivariatePolynomial::get_univariate_factors_and_deltas(
            &adjusted_factors,
            order,
            sample_points,
            p,
            k,
        );
        let h = self_adjusted.multivariate_hensel_lifting(
            &adjusted_factors,
            &mut uni,
            &delta,
            sample_points,
            Some(&padded_lcoeffs),
            order,
            1,
        );

        h.into_iter()
            .map(|f| {
                let f_i = f.map_coeff(|c| modulus.to_symmetric_integer(c), Z);
                let c = f_i.univariate_content(order[0]);
                f_i / &c
            })
            .collect()
    }

    fn find_sample(
        &self,
        order: &mut [usize],
        mut coefficient_upper_bound: i64,
        max_factors_num: Option<usize>,
    ) -> (Vec<Self>, Vec<(usize, Integer)>, i64, Self) {
        debug!("Find sample for {} with order {:?}", self, order);

        // select a suitable evaluation point, as small as possible as to not change the coefficient bound
        let mut cur_sample_points: Vec<_> =
            order[1..].iter().map(|i| (*i, Integer::zero())).collect();
        let mut cur_uni_f;
        let mut cur_biv_f;
        let mut rng = rng();
        let degree = self.degree(order[0]);
        let mut bivariate_factors: Vec<_>;

        let uni_lcoeff = self.univariate_lcoeff(order[0]);

        let mut content_fail_count = 0;
        'new_sample: loop {
            for s in &mut cur_sample_points {
                s.1 = Integer::Single(rng.random_range(0..=coefficient_upper_bound));
                debug!("Sample x{} {}", s.0, s.1);
            }

            cur_biv_f = self.clone();
            for ((v, s), rem_var) in cur_sample_points[1..].iter().zip(&order[1..]).rev() {
                cur_biv_f = cur_biv_f.replace(*v, s);
                if cur_biv_f.degree(*rem_var) != self.degree(*rem_var) {
                    coefficient_upper_bound += 10;
                    continue 'new_sample;
                }
            }

            // requirement for leading coefficient precomputation
            if cur_biv_f.univariate_lcoeff(order[0]).degree(order[1]) != uni_lcoeff.degree(order[1])
            {
                debug!(
                    "Degree of x{} in leading coefficient of bivariate image is wrong",
                    order[1]
                );
                coefficient_upper_bound += 10;
                continue 'new_sample;
            }

            let biv_df = cur_biv_f.derivative(order[0]);

            cur_uni_f = cur_biv_f.replace(cur_sample_points[0].0, &cur_sample_points[0].1);
            let uni_df = cur_uni_f.derivative(order[0]);

            if degree == cur_biv_f.degree(order[0])
                && degree == cur_uni_f.degree(order[0])
                && cur_biv_f.gcd(&biv_df).is_constant()
                && cur_uni_f.gcd(&uni_df).is_constant()
            {
                if !cur_biv_f.univariate_content(order[0]).is_one() {
                    content_fail_count += 1;
                    coefficient_upper_bound += 10;

                    debug!("Univariate content is not one");
                    if content_fail_count == 4 {
                        // it is likely that we will always find content for this variable ordering, so change the
                        // second variable
                        // TODO: is this guaranteed to work or should we also change the first variable?
                        let sec_var = order[1];
                        order.copy_within(2..order.len(), 1);
                        order[order.len() - 1] = sec_var;

                        for ((vs, _), v) in cur_sample_points.iter_mut().zip(&order[1..]) {
                            *vs = *v;
                        }

                        debug!("Changed the second variable to {}", order[1]);
                        content_fail_count = 0;
                    }

                    continue;
                }

                bivariate_factors = cur_biv_f.factor().into_iter().map(|f| f.0).collect();

                // absorb unit in another factor
                let mut has_minus = 0;
                bivariate_factors.retain(|f| {
                    if f.is_constant() && f.lcoeff() == -1 {
                        has_minus += 1;
                        false
                    } else {
                        true
                    }
                });

                if has_minus % 2 == 1 {
                    bivariate_factors[0] = -bivariate_factors[0].clone();
                }

                if bivariate_factors.len() <= max_factors_num.unwrap_or(bivariate_factors.len()) {
                    break;
                }
                debug!(
                    "Number of factors is too large: {} vs {}",
                    bivariate_factors.len(),
                    max_factors_num.unwrap_or(bivariate_factors.len())
                );
            }

            coefficient_upper_bound += 10;
            debug!("Growing bound {}", coefficient_upper_bound);
        }

        (
            bivariate_factors,
            cur_sample_points,
            coefficient_upper_bound,
            cur_uni_f,
        )
    }

    /// Perform multivariate factorization on a square-free polynomial.
    fn multivariate_factorization(
        &self,
        order: &mut [usize],
        mut coefficient_upper_bound: i64,
        mut max_bivariate_factors: Option<usize>,
    ) -> Vec<Self> {
        if let Some(m) = max_bivariate_factors
            && m == 1
        {
            return vec![self.clone()];
        }

        let mut sparse_fail = 0;
        let (bivariate_factors, sample_points, uni_f) = loop {
            // find a sample point with a small shift
            let (bivariate_factors, sample_points, coeff_b, uni_f) = self.find_sample(
                order,
                coefficient_upper_bound.max(10),
                max_bivariate_factors,
            );

            coefficient_upper_bound = coeff_b;

            if bivariate_factors.len() == 1 {
                // the polynomial is irreducible
                return vec![self.clone()];
            }

            if let Some(max) = max_bivariate_factors {
                if bivariate_factors.len() < max {
                    debug!("Updating bivariate bound to {}", bivariate_factors.len());
                    max_bivariate_factors = Some(bivariate_factors.len());
                }
            } else {
                debug!("Updating bivariate bound to {}", bivariate_factors.len());
                max_bivariate_factors = Some(bivariate_factors.len());
            }

            if sparse_fail == 5 || self.sparse_lift_possible(&bivariate_factors, order) {
                break (bivariate_factors, sample_points, uni_f);
            } else {
                debug!("Bad sample for sparse lifting {:?}", sample_points);
                sparse_fail += 1;
                coefficient_upper_bound += 10;
            }
        };

        if let Some(max) = max_bivariate_factors
            && bivariate_factors.len() > max
        {
            return self.multivariate_factorization(
                order,
                coefficient_upper_bound,
                max_bivariate_factors,
            );
        }

        for (v, s) in &sample_points {
            debug!("Sample point x{} = {}", v, s);
        }

        // select a suitable prime
        // we start small as we do not want to overshoot the coefficient bound too much
        // however, the sparse lifting algorithm requires divisions, which means we
        // can get unlucky with numbers being a multiple of the prime
        let mut prime_iter = PrimeIteratorU64::new(1 << 22);
        let mut field;
        let mut p;
        'new_prime: loop {
            p = prime_iter.next().unwrap();

            if p > u32::MAX as u64 {
                panic!("Ran out of primes during factorization of {self}");
            }

            if (&uni_f.lcoeff() % &p.into()).is_zero() {
                continue;
            }

            field = Zp::new(p as u32);

            // make sure the bivariate factors stay coprime
            let fs_p: Vec<_> = bivariate_factors
                .iter()
                .map(|f| f.map_coeff(|c| c.to_finite_field(&field), field.clone()))
                .collect();

            for (j, f) in fs_p.iter().enumerate() {
                for g in &fs_p[j + 1..] {
                    if !f.gcd(g).is_one() {
                        continue 'new_prime;
                    }
                }
            }

            break;
        }

        // TODO: modify bound by taking the shifts into account?
        let bound = self.coefficient_bound();

        let p_int = field.get_prime().to_integer();
        let mut max_p = p_int.clone();
        let mut k = 1;
        while &max_p * 2 < bound {
            max_p = &max_p * &p_int;
            k += 1;
        }

        let (sorted_biv_factors, true_lcoeffs) = match self.lcoeff_precomputation(
            &bivariate_factors,
            &sample_points,
            order,
            max_p.clone(),
            p as u32,
            k,
        ) {
            Ok((sorted_biv_factors, true_lcoeffs)) => (sorted_biv_factors, true_lcoeffs),
            Err(max_biv) => {
                // the leading coefficient computation failed because the bivariate factorization was wrong
                // try again with other sample points and a better bound
                return self.multivariate_factorization(
                    order,
                    coefficient_upper_bound + 10,
                    Some(max_biv),
                );
            }
        };

        for (b, l) in sorted_biv_factors.iter().zip(&true_lcoeffs) {
            debug!("Bivariate factor {} with true lcoeff {}", b, l);
        }

        if let Some(factorization) = self.sparse_lifting(&sorted_biv_factors, &true_lcoeffs, order)
        {
            // test the factorization
            let mut test = self.one();
            for f in &factorization {
                debug!("Factor = {}", f);
                test = &test * f;
            }

            if &test == self {
                return factorization;
            }
        }

        let field_mod = FiniteField::<Integer>::new(max_p.clone());
        let sorted_biv_factors_ff: Vec<_> = sorted_biv_factors
            .iter()
            .map(|f| f.map_coeff(|c| c.to_finite_field(&field_mod), field_mod.clone()))
            .collect();

        let (mut uni, delta) = MultivariatePolynomial::get_univariate_factors_and_deltas(
            &sorted_biv_factors_ff,
            order,
            &sample_points,
            p as u32,
            k,
        );

        // perform the Hensel lifting in a performance-optimized finite field if possible
        let factorization = if max_p < u64::MAX {
            let prime = match max_p {
                Integer::Single(b) => b as u64,
                Integer::Double(b) => b as u64,
                Integer::Large(b) => b.to_u64().unwrap(),
            };
            let small_field_mod = Zp64::new(prime);

            let poly_ff = self.map_coeff(
                |c| c.to_finite_field(&small_field_mod),
                small_field_mod.clone(),
            );

            let true_lcoeffs_ff: Vec<_> = true_lcoeffs
                .into_iter()
                .map(|f| {
                    f.map_coeff(
                        |c| c.to_finite_field(&small_field_mod),
                        small_field_mod.clone(),
                    )
                })
                .collect();

            let sorted_biv_factors_ff: Vec<_> = sorted_biv_factors
                .into_iter()
                .map(|f| {
                    f.map_coeff(
                        |c| c.to_finite_field(&small_field_mod),
                        small_field_mod.clone(),
                    )
                })
                .collect();

            let mut uni_f: Vec<_> = uni
                .into_iter()
                .map(|f| {
                    f.map_coeff(
                        |c| field_mod.from_element(c).to_finite_field(&small_field_mod),
                        small_field_mod.clone(),
                    )
                })
                .collect();

            let delta_f: Vec<_> = delta
                .into_iter()
                .map(|f| {
                    f.map_coeff(
                        |c| field_mod.from_element(c).to_finite_field(&small_field_mod),
                        small_field_mod.clone(),
                    )
                })
                .collect();

            let sample_points: Vec<_> = sample_points
                .iter()
                .map(|(v, p)| (*v, p.to_finite_field(&small_field_mod)))
                .collect();

            let factorization_ff = poly_ff.multivariate_hensel_lifting(
                &sorted_biv_factors_ff,
                &mut uni_f,
                &delta_f,
                &sample_points,
                Some(&true_lcoeffs_ff),
                order,
                2,
            );

            factorization_ff
                .into_iter()
                .map(|f| f.map_coeff(|c| small_field_mod.to_symmetric_integer(c), Z))
                .collect()
        } else {
            let field_mod = FiniteField::<Integer>::new(max_p.clone());

            let poly_ff = self.map_coeff(|c| field_mod.to_element(c.clone()), field_mod.clone());

            let true_lcoeffs_ff: Vec<_> = true_lcoeffs
                .into_iter()
                .map(|f| f.map_coeff(|c| field_mod.to_element(c.clone()), field_mod.clone()))
                .collect();

            let factorization_ff = poly_ff.multivariate_hensel_lifting(
                &sorted_biv_factors_ff,
                &mut uni,
                &delta,
                &sample_points,
                Some(&true_lcoeffs_ff),
                order,
                2,
            );

            factorization_ff
                .into_iter()
                .map(|f| f.map_coeff(|c| field_mod.to_symmetric_integer(c), Z))
                .collect()
        };

        // test the factorization
        let mut test = self.one();
        for f in &factorization {
            debug!("Factor = {}", f);
            test = &test * f;
        }

        if &test == self {
            factorization
        } else {
            let new_bound = max_bivariate_factors.unwrap_or(bivariate_factors.len()) - 1;
            debug!(
                "No immediate factorization of {} for sample points {:?}, new bound of bivariate factors: {}",
                self, sample_points, new_bound
            );

            // the bivariate factorization has too many factors, try again with other sample points
            self.multivariate_factorization(order, coefficient_upper_bound + 10, Some(new_bound))
        }
    }
}

impl<E: PositiveExponent> MultivariatePolynomial<FiniteField<Integer>, E, LexOrder> {
    /// Compute a univariate diophantine equation in `Z_p^k` by Newton iteration.
    fn get_univariate_factors_and_deltas(
        factors: &[Self],
        order: &[usize],
        sample_points: &[(usize, Integer)],
        p: u32,
        k: usize,
    ) -> (Vec<Self>, Vec<Self>) {
        // produce univariate factors and univariate delta
        let mut univariate_factors = factors.to_vec();
        for f in &mut univariate_factors {
            for (v, s) in sample_points {
                if order[0] != *v {
                    *f = f.replace(*v, s);
                }
            }
        }

        let univariate_deltas = MultivariatePolynomial::lift_diophantine_univariate(
            &mut univariate_factors,
            &factors[0].constant(factors[0].ring.one()),
            p,
            k,
        );

        (univariate_factors, univariate_deltas)
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use crate::{
        atom::AtomCore,
        domains::{
            InternalOrdering,
            algebraic_number::AlgebraicExtension,
            finite_field::{Z2, Zp},
            integer::Z,
            rational::Q,
        },
        parse,
        poly::factor::Factorize,
        symbol,
    };

    #[test]
    fn factor_ff_square_free() {
        let field = Zp::new(3);
        let poly = parse!("(1+v1)*(1+v1^2)^2*(v1^4+1)^3").to_polynomial::<_, u8>(&field, None);

        let res = [("1+v1^4", 3), ("1+v1^2", 2), ("1+v1", 1)];

        let mut res = res
            .iter()
            .map(|(f, p)| {
                (
                    parse!(f)
                        .expand()
                        .to_polynomial(&field, poly.variables.clone()),
                    *p,
                )
            })
            .collect::<Vec<_>>();
        res.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut r = poly.square_free_factorization();
        r.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));

        assert_eq!(r, res);
    }

    #[test]
    fn factor_ff_bivariate() {
        let field = Zp::new(997);
        let poly = parse!("((v2+1)*v1^2+v1*v2+1)*((v2^2+2)*v1^2+v2+1)")
            .to_polynomial::<_, u8>(&field, None);

        let res = [("1+2*v1^2+v2+v2^2*v1^2", 1), ("1+v1^2+v2*v1+v2*v1^2", 1)];

        let mut res = res
            .iter()
            .map(|(f, p)| {
                (
                    parse!(f)
                        .expand()
                        .to_polynomial(&field, poly.variables.clone()),
                    *p,
                )
            })
            .collect::<Vec<_>>();

        res.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut r = poly.factor();
        r.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        assert_eq!(r, res);
    }

    #[test]
    fn factor_square_free() {
        let poly =
            parse!("3*(2*v1^2+v2)(v1^3+v2)^2(1+4*v2)^2(1+v1)").to_polynomial::<_, u8>(&Z, None);

        let res = [
            ("3", 1),
            ("1+4*v2", 2),
            ("1+v1", 1),
            ("v2+2*v1^2", 1),
            ("v2+v1^3", 2),
        ];

        let mut res = res
            .iter()
            .map(|(f, p)| {
                (
                    parse!(f).expand().to_polynomial(&Z, poly.variables.clone()),
                    *p,
                )
            })
            .collect::<Vec<_>>();

        res.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut r = poly.square_free_factorization();
        r.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        assert_eq!(r, res);
    }

    #[test]
    fn factor_univariate_1() {
        let poly = parse!("2*(4 + 3*v1)*(3 + 2*v1 + 3*v1^2)*(3 + 8*v1^2)*(4 + v1 + v1^16)")
            .to_polynomial::<_, u8>(&Z, None);

        let res = [
            ("2", 1),
            ("4+3*v1", 1),
            ("3+2*v1+3*v1^2", 1),
            ("3+8*v1^2", 1),
            ("4+v1+v1^16", 1),
        ];

        let mut res = res
            .iter()
            .map(|(f, p)| {
                (
                    parse!(f).expand().to_polynomial(&Z, poly.variables.clone()),
                    *p,
                )
            })
            .collect::<Vec<_>>();

        res.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut r = poly.factor();
        r.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        assert_eq!(r, res);
    }

    #[test]
    fn factor_univariate_2() {
        let poly =
            parse!("(v1+1)(v1+2)(v1+3)^3(v1+4)(v1+5)(v1^2+6)(v1^3+7)(v1+8)^2(v1^4+9)(v1^5+v1+10)")
                .to_polynomial::<_, u8>(&Z, None);

        let res = [
            ("5+v1", 1),
            ("1+v1", 1),
            ("4+v1", 1),
            ("2+v1", 1),
            ("7+v1^3", 1),
            ("10+v1+v1^5", 1),
            ("6+v1^2", 1),
            ("9+v1^4", 1),
            ("8+v1", 2),
            ("3+v1", 3),
        ];

        let mut res = res
            .iter()
            .map(|(f, p)| {
                (
                    parse!(f).expand().to_polynomial(&Z, poly.variables.clone()),
                    *p,
                )
            })
            .collect::<Vec<_>>();

        res.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut r = poly.factor();
        r.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        assert_eq!(r, res);
    }

    #[test]
    fn factor_bivariate() {
        let input = "(v1^2+v2+v1+1)(3*v1+v2^2+4)*(6*v1*(v2+1)+v2+5)*(7*v1*v2+4)";
        let poly = parse!(input).to_polynomial::<_, u8>(&Z, None);

        let res = [
            ("(1+v2+v1+v1^2)", 1),
            ("(5+v2+6*v1+6*v1*v2)", 1),
            ("(4+v2^2+3*v1)", 1),
            ("(4+7*v1*v2)", 1),
        ];

        let mut res = res
            .iter()
            .map(|(f, p)| {
                (
                    parse!(f).expand().to_polynomial(&Z, poly.variables.clone()),
                    *p,
                )
            })
            .collect::<Vec<_>>();

        res.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut r = poly.factor();
        r.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        assert_eq!(r, res);
    }

    #[test]
    fn factor_multivariate() {
        let input = "(v1*(2+2*v2+2*v3)+1)*(v1*(4+v3^2)+v2+3)*(v1*(v4+v4^2+4+v2)+v4+5)";
        let poly = parse!(input).to_polynomial::<_, u8>(&Z, None);

        let res = [
            ("5+v4+4*v1+v1*v4+v1*v4^2+v1*v2", 1),
            ("1+2*v1+2*v1*v3+2*v1*v2 ", 1),
            ("3+v2+4*v1+v1*v3^2", 1),
        ];

        let mut res = res
            .iter()
            .map(|(f, p)| {
                (
                    parse!(f).expand().to_polynomial(&Z, poly.variables.clone()),
                    *p,
                )
            })
            .collect::<Vec<_>>();

        res.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut r = poly.factor();
        r.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        assert_eq!(r, res);
    }

    #[test]
    fn factor_overall_minus() {
        let poly = parse!("-v1*v3^2-v1*v2*v3^2").to_polynomial::<_, u8>(
            &Z,
            Some(Arc::new(vec![
                symbol!("v1").into(),
                symbol!("v2").into(),
                symbol!("v3").into(),
            ])),
        );

        let res = [("-1", 1), ("v3", 2), ("1+v2", 1), ("v1", 1)];

        let mut res = res
            .iter()
            .map(|(f, p)| {
                (
                    parse!(f).expand().to_polynomial(&Z, poly.variables.clone()),
                    *p,
                )
            })
            .collect::<Vec<_>>();

        res.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut r = poly.factor();
        r.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        assert_eq!(r, res);
    }

    #[test]
    fn factor_multivariate_2() {
        let poly = parse!("v2^2*v3-v1*v2*v3+v1*v2*v3^2+v1*v2^2-v1^2*v3^2+v1^2*v2*v3")
            .to_polynomial::<_, u8>(
                &Z,
                Some(Arc::new(vec![
                    symbol!("v1").into(),
                    symbol!("v2").into(),
                    symbol!("v3").into(),
                ])),
            );

        let res = [("v2+v1*v3", 1), ("v2*v3-v1*v3+v1*v2", 1)];

        let mut res = res
            .iter()
            .map(|(f, p)| {
                (
                    parse!(f).expand().to_polynomial(&Z, poly.variables.clone()),
                    *p,
                )
            })
            .collect::<Vec<_>>();

        res.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        let mut r = poly.factor();
        r.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));
        assert_eq!(r, res);
    }

    #[test]
    fn galois_upgrade() {
        let a =
            parse!("x^7(y^5+y^4+y^3+y^2)+x^5(y^3+y)+x^4(y^4+y)+x^3(y^2+y)+x^2y+x*y^2+x*y+x+y+1")
                .to_polynomial::<_, u8>(&Z2, None);

        assert_eq!(a.factor().len(), 2)
    }

    #[test]
    fn algebraic_extension() {
        let a = parse!("z^4+z^3+(2+a-a^2)z^2+(1+a^2-2a^3)z-2").to_polynomial::<_, u8>(&Q, None);
        let f = parse!("a^4-3").to_polynomial::<_, u16>(&Q, None);
        let f = AlgebraicExtension::new(f);

        let mut factors = a.to_number_field(&f).factor();

        let f1 = parse!("(1-a^2)+(1-a)*z+z^2")
            .to_polynomial::<_, u8>(&Q, a.get_vars().clone())
            .to_number_field(&f);
        let f2 = parse!("(1+a^2)+(a)*z+z^2")
            .to_polynomial::<_, u8>(&Q, a.get_vars().clone())
            .to_number_field(&f);

        factors.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.cmp(&b.1)));

        assert_eq!(factors, vec![(f1, 1), (f2, 1)])
    }
}
