//! Compute Groebner bases for polynomial ideals.
//!
//! # Examples
//! ```
//! use symbolica::{
//!   atom::{Atom, AtomCore},
//!   domains::finite_field::Zp,
//!   parse,
//!   poly::{groebner::GroebnerBasis, polynomial::MultivariatePolynomial, GrevLexOrder},
//! };
//!
//! let polys = [
//!     "v1 v2 v3 v4 - 1",
//!     "v1 v2 v3 + v1 v2 v4 + v1 v3 v4 + v2 v3 v4",
//!     "v1 v2 + v2 v3 + v1 v4 + v3 v4",
//!     "v1 + v2 + v3 + v4",
//! ];
//!
//! let ideal: Vec<MultivariatePolynomial<_, u16>> = polys
//! .iter()
//! .map(|x| {
//!     let a = parse!(x);
//!     a.to_polynomial(&Zp::new(13), None)
//! })
//! .collect();
//!
//! // compute the Groebner basis with lex ordering
//! let gb = GroebnerBasis::new(&ideal, false);
//!
//! // verify the result is correct
//! let res = [
//!     "v4+v3+v2+v1",
//!     "v4^2+2*v2*v4+v2^2",
//!     "11*v4^2+v3*v4+v3^2*v4^4-v2*v4+v2*v3",
//!     "-v4+v4^5-v2+v2*v4^4",
//!     "-v4-v3+v3^2*v4^3+v3^3*v4^2",
//!     "1-v4^4-v3^2*v4^2+v3^2*v4^6",
//! ];
//!
//! let res: Vec<MultivariatePolynomial<_, u16>> = res
//! .iter()
//! .map(|x| {
//!     let a = parse!(x);
//!     a.to_polynomial(&Zp::new(13), ideal[0].variables.clone())
//! })
//! .collect();
//!
//! assert_eq!(gb.system, res);
//! ```

use std::{cmp::Ordering, rc::Rc};

use ahash::HashMap;

use crate::domains::{
    Field, Ring, RingOps, Set,
    algebraic_number::AlgebraicExtension,
    finite_field::{FiniteField, FiniteFieldCore, Mersenne64, Z2, Zp, Zp64},
    rational::RationalField,
};

use super::{Exponent, MonomialOrder, polynomial::MultivariatePolynomial};

#[derive(Debug)]
pub struct CriticalPair<R: Field, E: Exponent, O: MonomialOrder> {
    lcm_diff_first: Vec<E>,
    poly_first: Rc<MultivariatePolynomial<R, E, O>>,
    index_first: usize,
    lcm_diff_sec: Vec<E>,
    poly_sec: Rc<MultivariatePolynomial<R, E, O>>,
    index_sec: usize,
    lcm: Vec<E>,
    degree: E,
    disjoint: bool,
}

impl<R: Field, E: Exponent, O: MonomialOrder> CriticalPair<R, E, O> {
    fn new(
        f1: Rc<MultivariatePolynomial<R, E, O>>,
        f2: Rc<MultivariatePolynomial<R, E, O>>,
        index1: usize,
        index2: usize,
    ) -> CriticalPair<R, E, O> {
        // determine the lcm of leading monomials
        let lcm: Vec<E> = f1
            .max_exp()
            .iter()
            .zip(f2.max_exp())
            .map(|(e1, e2)| *e1.max(e2))
            .collect();

        let lcm_diff_first: Vec<E> = lcm
            .iter()
            .zip(f1.max_exp())
            .map(|(e1, e2)| *e1 - *e2)
            .collect();

        let lcm_diff_sec: Vec<E> = lcm
            .iter()
            .zip(f2.max_exp())
            .map(|(e1, e2)| *e1 - *e2)
            .collect();

        CriticalPair {
            disjoint: lcm_diff_first == f2.max_exp(),
            degree: lcm.iter().cloned().sum::<E>(),
            lcm_diff_first,
            poly_first: f1,
            index_first: index1,
            lcm_diff_sec,
            poly_sec: f2,
            index_sec: index2,
            lcm,
        }
    }
}

/// A position of a monomial in the reduction matrix.
pub struct MonomialData {
    present: bool,
    column: usize,
}

pub struct GroebnerBasis<R: Field, E: Exponent, O: MonomialOrder> {
    pub system: Vec<MultivariatePolynomial<R, E, O>>,
    pub print_stats: bool,
}

impl<R: Field + Echelonize, E: Exponent, O: MonomialOrder> GroebnerBasis<R, E, O> {
    /// Construct a Groebner basis for a polynomial ideal.
    ///
    /// Progress can be monitored with `print_stats`.
    pub fn new(
        ideal: &[MultivariatePolynomial<R, E, O>],
        print_stats: bool,
    ) -> GroebnerBasis<R, E, O> {
        let mut ideal = ideal.to_vec();
        MultivariatePolynomial::unify_variables_list(&mut ideal);

        let mut b = GroebnerBasis {
            system: ideal,
            print_stats,
        };

        b.f4();
        b.reduce_basis()
    }

    #[inline]
    fn simplify(
        tab: &mut Vec<(Vec<E>, Rc<MultivariatePolynomial<R, E, O>>)>,
        lcm: &[E],
    ) -> Rc<MultivariatePolynomial<R, E, O>> {
        for (m, f) in tab.iter().rev() {
            if m == lcm {
                return f.clone();
            }

            if lcm.iter().zip(m).all(|(el, em)| *el >= *em) {
                let diff: Vec<_> = lcm.iter().zip(m).map(|(el, em)| *el - *em).collect();
                let a = Rc::new((**f).clone().mul_exp(&diff));
                tab.push((lcm.to_vec(), a.clone()));
                return a;
            }
        }
        panic!("Unknown polynomial associated with exponent map {lcm:?}");
    }

    /// The F4 algorithm for computing a Groebner basis.
    ///
    /// Adapted from [A new efficient algorithm for computing Gröbner bases (F4)](https://doi.org/10.1016/S0022-4049(99)00005-5) by Jean-Charles Faugére.
    ///
    fn f4(&mut self) {
        let nvars = self.system[0].nvars();
        let field = self.system[0].ring.clone();

        let mut simplifications = vec![];
        let mut basis = vec![];
        let mut critical_pairs = vec![];

        for (i, f) in self.system.drain(..).enumerate() {
            let poly = Rc::new(f.clone().make_monic());
            simplifications.push(vec![(vec![E::zero(); nvars], poly.clone())]);
            Self::update(&mut basis, &mut critical_pairs, poly, i);
        }

        let mut matrix = vec![];

        let mut all_monomials: HashMap<Vec<E>, MonomialData> = HashMap::default();
        let mut current_monomials = vec![];
        let mut sorted_monomial_indices = vec![];
        let mut exp = vec![E::zero(); nvars];
        let mut new_polys = vec![];
        let mut selected_polys = vec![];

        let mut buffer = vec![];
        let mut pivots: Vec<Option<usize>> = vec![];

        let mut iter_count = 1;
        while !critical_pairs.is_empty() {
            // select the critical pairs with the lowest lcm degree
            let lowest_lcm_deg = critical_pairs.iter().map(|x| x.degree).min().unwrap();

            if self.print_stats {
                println!(
                    "Iteration {}:\n\tDegree={}, Basis length={}, Critical pairs={}",
                    iter_count,
                    lowest_lcm_deg,
                    basis.len(),
                    critical_pairs.len(),
                );
            }
            iter_count += 1;

            selected_polys.clear();
            let mut i = critical_pairs.len() - 1;

            let mut l_tmp = vec![];
            loop {
                if critical_pairs[i].degree == lowest_lcm_deg {
                    let pair = critical_pairs.swap_remove(i);

                    let e = [
                        (pair.index_first, pair.lcm_diff_first),
                        (pair.index_sec, pair.lcm_diff_sec),
                    ];
                    for poly_info in e {
                        if !l_tmp.contains(&poly_info) {
                            let new_f1 =
                                Self::simplify(&mut simplifications[poly_info.0], &poly_info.1);
                            selected_polys.push(new_f1);
                            l_tmp.push(poly_info);
                        }
                    }
                }

                if i == 0 {
                    break;
                }

                i -= 1;
            }

            // symbolic preprocessing

            for x in all_monomials.values_mut() {
                x.present = false;
            }

            // flag all head monomials as done
            for p in &selected_polys {
                if let Some(m) = all_monomials.get_mut(p.max_exp()) {
                    m.present = true;
                } else {
                    all_monomials.insert(
                        p.max_exp().to_vec(),
                        MonomialData {
                            present: true,
                            column: 0,
                        },
                    );
                }
            }

            new_polys.clear();
            let mut i = 0;
            while i < selected_polys.len() {
                for monom in selected_polys[i].exponents_iter() {
                    if let Some(m) = all_monomials.get_mut(monom) {
                        if m.present {
                            continue;
                        }
                        m.present = true;
                    } else {
                        all_monomials.insert(
                            monom.to_vec(),
                            MonomialData {
                                present: true,
                                column: 0,
                            },
                        );
                    }

                    // search for a reducer and select the smallest for better performance
                    if let Some((index, g)) = basis
                        .iter()
                        .filter(|g| monom.iter().zip(g.1.max_exp()).all(|(pe, ge)| *pe >= *ge))
                        .min_by_key(|g| g.1.nterms())
                    {
                        for ((e, pe), ge) in exp.iter_mut().zip(monom).zip(g.max_exp()) {
                            *e = *pe - *ge;
                        }

                        let pp = Self::simplify(&mut simplifications[*index], &exp);
                        new_polys.push(pp);
                    }
                }

                i += 1;

                selected_polys.append(&mut new_polys);
            }

            // construct a matrix that is sparse in the columns

            current_monomials.clear();
            sorted_monomial_indices.clear();

            for (k, v) in &all_monomials {
                if v.present {
                    current_monomials.extend_from_slice(k);
                }
            }

            for i in 0..(current_monomials.len() / nvars) {
                sorted_monomial_indices.push(i);
            }

            if self.print_stats {
                println!(
                    "\tMonomials in use={}/{}",
                    sorted_monomial_indices.len(),
                    all_monomials.len()
                );
                println!(
                    "\tMatrix shape={}x{}, density={:.2}%",
                    selected_polys.len(),
                    sorted_monomial_indices.len(),
                    selected_polys.iter().map(|i| i.nterms()).sum::<usize>() as f64
                        / (sorted_monomial_indices.len() as f64 * selected_polys.len() as f64)
                        * 100.
                );
            }

            // sort monomials in descending order
            sorted_monomial_indices.sort_unstable_by(|e1, e2| {
                O::cmp(
                    &current_monomials[*e2 * nvars..(*e2 + 1) * nvars],
                    &current_monomials[*e1 * nvars..(*e1 + 1) * nvars],
                )
            });

            for (column, index) in sorted_monomial_indices.iter().enumerate() {
                all_monomials
                    .get_mut(&current_monomials[index * nvars..(index + 1) * nvars])
                    .unwrap()
                    .column = column;
            }

            R::echelonize(
                &mut matrix,
                &mut selected_polys,
                &all_monomials,
                &sorted_monomial_indices,
                &field,
                &mut buffer,
                &mut pivots,
                self.print_stats,
            );

            // construct new polynomials
            for m in &matrix {
                let lmi = sorted_monomial_indices[m[0].1];
                let lm = &current_monomials[lmi * nvars..(lmi + 1) * nvars];

                // create the new polynomial in the proper order
                let mut poly = selected_polys[0].zero_with_capacity(m.len());
                for (coeff, col) in m.iter().rev() {
                    let index = sorted_monomial_indices[*col];
                    let exp = &current_monomials[index * nvars..(index + 1) * nvars];
                    poly.append_monomial(field.from_larger(coeff), exp);
                }

                let poly = Rc::new(poly);

                if selected_polys.iter().all(|p| p.max_exp() != lm) {
                    let new_index = simplifications.len();
                    simplifications.push(vec![(vec![E::zero(); nvars], poly.clone())]);

                    Self::update(&mut basis, &mut critical_pairs, poly, new_index);
                } else {
                    // update entries in the tab with simpler polynomials
                    let mut diff = vec![E::zero(); nvars];
                    'bf: for (g_ind, g) in &basis {
                        if poly
                            .last_exponents()
                            .iter()
                            .zip(g.last_exponents())
                            .all(|(pi, gi)| *pi >= *gi)
                        {
                            for ((d, pi), gi) in diff
                                .iter_mut()
                                .zip(poly.last_exponents())
                                .zip(g.last_exponents())
                            {
                                *d = *pi - *gi;
                            }

                            for (diff_e, p) in &mut simplifications[*g_ind] {
                                if diff == *diff_e {
                                    *p = poly.clone();
                                    continue 'bf;
                                }
                            }

                            // new polynomial
                            simplifications[*g_ind].push((diff.clone(), poly.clone()));
                        }
                    }
                }
            }
        }

        self.system = basis.into_iter().map(|x| (*x.1).clone()).collect();
    }
}

impl<R: Field, E: Exponent, O: MonomialOrder> MultivariatePolynomial<R, E, O> {
    /// Completely reduce the polynomial w.r.t the polynomials `gs`.
    /// For example reducing `f=y^2+x` by `g=[x]` yields `y^2`.
    pub fn reduce(
        &self,
        gs: &[MultivariatePolynomial<R, E, O>],
    ) -> MultivariatePolynomial<R, E, O> {
        if gs.iter().any(|x| self.variables != x.variables) {
            let mut sys: Vec<_> = gs.to_vec();
            sys.push(self.clone());
            Self::unify_variables_list(&mut sys);
            return sys.last().unwrap().reduce(&sys[..gs.len()]);
        }

        let mut q = self.zero_with_capacity(self.nterms());
        let mut r = self.clone();

        let mut rest_coeff = vec![];
        let mut rest_exponents = vec![];

        let mut monom = vec![E::zero(); self.nvars()];

        'term: while !r.is_zero() {
            // find a divisor that has the least amount of terms
            while let Some(g) = gs
                .iter()
                .filter(|g| {
                    r.max_exp()
                        .iter()
                        .zip(g.max_exp())
                        .all(|(h1, h2)| *h1 >= *h2)
                })
                .min_by_key(|g| g.nterms())
            {
                for ((e, e1), e2) in monom.iter_mut().zip(r.max_exp()).zip(g.max_exp()) {
                    *e = *e1 - *e2;
                }

                let ratio = g.ring.div(r.max_coeff(), g.max_coeff());
                r = r - g.clone().mul_exp(&monom).mul_coeff(ratio);

                if r.is_zero() {
                    break 'term;
                }
            }

            // strip leading monomial that is not reducible
            rest_exponents.extend_from_slice(r.exponents(r.nterms() - 1));
            rest_coeff.push(r.coefficients.pop().unwrap());
        }

        // append in sorted order
        while let Some(c) = rest_coeff.pop() {
            let l = rest_coeff.len();
            q.append_monomial(c, &rest_exponents[l * self.nvars()..(l + 1) * self.nvars()]);
        }

        q
    }
}

impl<R: Field, E: Exponent, O: MonomialOrder> GroebnerBasis<R, E, O> {
    /// Add a new polynomial to the basis, updating and filtering the existing
    /// basis and critical pairs, based on Gebauer and Moeller's redundant pair criteria.
    ///
    /// Adapted from "A Computational Approach to Commutative Algebra" by Thomas Becker Volker Weispfenning.
    fn update(
        basis: &mut Vec<(usize, Rc<MultivariatePolynomial<R, E, O>>)>,
        critical_pairs: &mut Vec<CriticalPair<R, E, O>>,
        f: Rc<MultivariatePolynomial<R, E, O>>,
        index: usize,
    ) {
        let mut new_pairs: Vec<_> = basis
            .iter()
            .map(|b| (CriticalPair::new(b.1.clone(), f.clone(), b.0, index), true))
            .collect();

        for i in 0..new_pairs.len() {
            new_pairs[i].1 = false;
            new_pairs[i].1 = new_pairs[i].0.disjoint
                || new_pairs.iter().all(|p2| {
                    !p2.1
                        || new_pairs[i]
                            .0
                            .lcm
                            .iter()
                            .zip(&p2.0.lcm)
                            .any(|(e1, e2)| *e1 < *e2)
                });
        }

        new_pairs.retain(|p| p.1 && !p.0.disjoint);

        critical_pairs.retain(|p| {
            p.lcm.iter().zip(f.max_exp()).any(|(e1, e2)| *e1 < *e2)
                || p.poly_first
                    .max_exp()
                    .iter()
                    .zip(f.max_exp())
                    .zip(&p.lcm)
                    .all(|((e1, e2), ecm)| e1.max(e2) == ecm)
                || p.poly_sec
                    .max_exp()
                    .iter()
                    .zip(f.max_exp())
                    .zip(&p.lcm)
                    .all(|((e1, e2), ecm)| e1.max(e2) == ecm)
        });

        critical_pairs.extend(new_pairs.into_iter().map(|np| np.0));

        basis.retain(|b| {
            b.1.max_exp()
                .iter()
                .zip(f.max_exp())
                .any(|(e1, e2)| *e1 < *e2)
        });

        basis.push((index, f));
    }

    pub fn reduce_basis(mut self) -> Self {
        // filter lead-reducible polynomials
        let mut res = vec![true; self.system.len()];
        'l1: for (i, p1) in self.system.iter().enumerate() {
            for (j, p2) in self.system.iter().enumerate() {
                if i != j
                    && res[j]
                    && p1
                        .max_exp()
                        .iter()
                        .zip(p2.max_exp())
                        .all(|(h1, h2)| *h1 >= *h2)
                {
                    res[i] = false;
                    continue 'l1;
                }
            }
        }

        let mut lead_reduced = vec![];
        for (i, p) in self.system.drain(..).enumerate() {
            if res[i] {
                lead_reduced.push(p);
            }
        }

        let mut basis = vec![];
        for i in 0..lead_reduced.len() {
            lead_reduced.swap(0, i);
            let h = lead_reduced[0].reduce(&lead_reduced[1..]);
            if !h.is_zero() {
                let i = h.ring.inv(h.max_coeff());
                basis.push(h.mul_coeff(i));
            }
        }

        basis.sort_by(|p1, p2| p2.max_exp().cmp(p1.max_exp()));

        GroebnerBasis {
            system: basis,
            print_stats: self.print_stats,
        }
    }

    pub fn is_groebner_basis(system: &[MultivariatePolynomial<R, E, O>]) -> bool {
        for (i, p1) in system.iter().enumerate() {
            for p2 in &system[i + 1..] {
                let lcm: Vec<E> = p1
                    .max_exp()
                    .iter()
                    .zip(p2.max_exp())
                    .map(|(e1, e2)| *e1.max(e2))
                    .collect();

                // construct s-polynomial
                let extra_factor_f1: Vec<E> = lcm
                    .iter()
                    .zip(p1.max_exp())
                    .map(|(e1, e2)| *e1 - *e2)
                    .collect();

                let extra_factor_f2: Vec<E> = lcm
                    .iter()
                    .zip(p2.max_exp())
                    .map(|(e1, e2)| *e1 - *e2)
                    .collect();
                let new_f1 = p1
                    .clone()
                    .mul_exp(&extra_factor_f1)
                    .mul_coeff(p1.ring.div(p2.max_coeff(), p1.max_coeff()));
                let new_f2 = p2
                    .clone()
                    .mul_exp(&extra_factor_f2)
                    .mul_coeff(p1.ring.div(p1.max_coeff(), p2.max_coeff()));

                let s = new_f1 - new_f2;

                if !s.reduce(system).is_zero() {
                    return false;
                }
            }
        }
        true
    }
}

/// Echelonize a matrix with entries in the field.
pub trait Echelonize: Field {
    type LargerField;

    fn from_larger(&self, element: &Self::LargerField) -> <Self as Set>::Element;
    fn echelonize<E: Exponent, O: MonomialOrder>(
        matrix: &mut Vec<Vec<(Self::LargerField, usize)>>,
        selected_polys: &mut Vec<Rc<MultivariatePolynomial<Self, E, O>>>,
        all_monomials: &HashMap<Vec<E>, MonomialData>,
        sorted_monomial_indices: &[usize],
        field: &Self,
        buffer: &mut Vec<Self::LargerField>,
        pivots: &mut Vec<Option<usize>>,
        print_stats: bool,
    );
}

impl Echelonize for Zp {
    type LargerField = i64;

    /// Specialized 32-bit finite field echelonization based on
    /// "A Compact Parallel Implementation of F4" by Monagan and Pearce.
    fn echelonize<E: Exponent, O: MonomialOrder>(
        matrix: &mut Vec<Vec<(i64, usize)>>,
        selected_polys: &mut Vec<Rc<MultivariatePolynomial<Zp, E, O>>>,
        all_monomials: &HashMap<Vec<E>, MonomialData>,
        sorted_monomial_indices: &[usize],
        field: &Zp,
        buffer: &mut Vec<i64>,
        pivots: &mut Vec<Option<usize>>,
        print_stats: bool,
    ) {
        fn u32_inv(coeff: u32, prime: u32) -> u32 {
            // extended Euclidean algorithm: a x + b p = gcd(x, p) = 1 or a x = 1 (mod p)
            let mut u1: u32 = 1;
            let mut u3 = coeff;
            let mut v1: u32 = 0;
            let mut v3 = prime;
            let mut even_iter: bool = true;

            while v3 != 0 {
                let q = u3 / v3;
                let t3 = u3 % v3;
                let t1 = u1 + q * v1;
                u1 = v1;
                v1 = t1;
                u3 = v3;
                v3 = t3;
                even_iter = !even_iter;
            }

            if even_iter { u1 } else { prime - u1 }
        }

        matrix.resize(selected_polys.len(), vec![]);
        for (row, p) in matrix.iter_mut().zip(selected_polys) {
            row.clear();

            for (coeff, exp) in p.coefficients.iter().zip(p.exponents_iter()).rev() {
                row.push((
                    field.from_element(coeff) as i64,
                    all_monomials.get(exp).unwrap().column,
                ));
            }
        }

        // sort the matrix rows to sort the shortest and most reduced pivots on top
        matrix.sort_unstable_by(|r1, r2| {
            r1[0]
                .1
                .cmp(&r2[0].1)
                .then(r1.len().cmp(&r2.len()))
                .then_with(|| {
                    for ((_, i1), (_, i2)) in r1.iter().zip(r2) {
                        match i1.cmp(i2) {
                            Ordering::Equal => {}
                            x => {
                                return x.reverse();
                            }
                        }
                    }

                    Ordering::Equal
                })
        });

        // row-reduce the sparse matrix
        for p in &mut *pivots {
            *p = None;
        }

        buffer.resize(sorted_monomial_indices.len(), 0);
        pivots.resize(sorted_monomial_indices.len(), None);

        let p = field.get_prime() as i64;
        let p2 = p * p;

        let mut pc = 0;
        for r in 0..matrix.len() {
            // identify all pivots
            if let Some((coeff, col)) = matrix[r].first_mut()
                && pivots[*col].is_none() {
                    pivots[*col] = Some(r);
                    pc += 1;

                    if *coeff != 1 {
                        let inv_pivot = u32_inv(*coeff as u32, field.get_prime());

                        for (coeff, _) in &mut matrix[r] {
                            *coeff *= inv_pivot as i64;
                            *coeff %= field.get_prime() as i64;
                        }
                    }
                }
        }

        if print_stats {
            println!("\tPivots={}, rows to reduce={}", pc, matrix.len() - pc);
        }

        for r in 0..matrix.len() {
            if matrix[r].is_empty() {
                continue;
            }

            if let Some((coeff, col)) = matrix[r].first_mut()
                && pivots[*col].is_none() {
                    pivots[*col] = Some(r);
                    pc += 1;

                    if *coeff != 1 {
                        let inv_pivot = u32_inv(*coeff as u32, field.get_prime());

                        for (coeff, _) in &mut matrix[r] {
                            *coeff *= inv_pivot as i64;
                            *coeff %= field.get_prime() as i64;
                        }
                    }
                }

            // do not reduce pivots
            if pivots.contains(&Some(r)) {
                continue;
            }

            // copy row into the buffer
            for (coeff, col) in &*matrix[r] {
                buffer[*col] = *coeff;
            }

            for i in 0..buffer.len() {
                if buffer[i] != 0 {
                    buffer[i] %= p;
                }

                if buffer[i] == 0 {
                    continue;
                }

                let Some(pivot_index) = pivots[i] else {
                    // keep on reducing this new pivot
                    continue;
                };

                let pivot = &matrix[pivot_index];
                let c = buffer[i];

                buffer[i] = 0;

                let mut t;
                let mut m;
                for (coeff, col) in pivot.iter().skip(1) {
                    t = buffer[*col];
                    m = *coeff * c;

                    if t >= m {
                        t -= m;
                    } else {
                        t += p2 - m;
                    }

                    buffer[*col] = t;
                }
            }

            matrix[r].clear();

            for (col, coeff) in buffer.iter_mut().enumerate() {
                if *coeff != 0 {
                    matrix[r].push((*coeff, col));
                    *coeff = 0;
                }
            }

            if let Some((coeff, col)) = matrix[r].first() {
                pivots[*col] = Some(r);

                if *coeff != 1 {
                    let inv_pivot = u32_inv(*coeff as u32, field.get_prime());

                    for (coeff, _) in &mut matrix[r] {
                        *coeff *= inv_pivot as i64;
                        *coeff %= field.get_prime() as i64;
                    }
                }
            }
        }

        // TODO: do back substitution
        matrix.retain(|r| !r.is_empty());
    }

    fn from_larger(&self, element: &i64) -> <Self as Set>::Element {
        self.to_element(*element as u32)
    }
}

macro_rules! echelonize_impl {
    ($f: ty) => {
        impl Echelonize for $f {
            type LargerField = Self::Element;

            #[inline(never)]
            fn echelonize<E: Exponent, O: MonomialOrder>(
                matrix: &mut Vec<Vec<(Self::Element, usize)>>,
                selected_polys: &mut Vec<Rc<MultivariatePolynomial<Self, E, O>>>,
                all_monomials: &HashMap<Vec<E>, MonomialData>,
                sorted_monomial_indices: &[usize],
                field: &Self,
                buffer: &mut Vec<Self::Element>,
                pivots: &mut Vec<Option<usize>>,
                print_stats: bool,
            ) {
                matrix.resize(selected_polys.len(), vec![]);
                for (row, p) in matrix.iter_mut().zip(selected_polys) {
                    row.clear();

                    for (coeff, exp) in p.coefficients.iter().zip(p.exponents_iter()).rev() {
                        row.push((coeff.clone(), all_monomials.get(exp).unwrap().column));
                    }
                }

                // sort the matrix rows to sort the shortest and most reduced pivots on top
                matrix.sort_unstable_by(|r1, r2| {
                    r1[0]
                        .1
                        .cmp(&r2[0].1)
                        .then(r1.len().cmp(&r2.len()))
                        .then_with(|| {
                            for ((_, i1), (_, i2)) in r1.iter().zip(r2) {
                                match i1.cmp(i2) {
                                    Ordering::Equal => {}
                                    x => {
                                        return x.reverse();
                                    }
                                }
                            }

                            Ordering::Equal
                        })
                });

                for p in &mut *pivots {
                    *p = None;
                }

                buffer.resize(sorted_monomial_indices.len(), field.zero());
                pivots.resize(sorted_monomial_indices.len(), None);

                let mut pc = 0;
                for r in 0..matrix.len() {
                    // identify all pivots
                    if let Some((coeff, col)) = matrix[r].first_mut() {
                        if pivots[*col].is_none() {
                            pivots[*col] = Some(r);
                            pc += 1;

                            if field.is_one(coeff) {
                                let inv_pivot = field.inv(coeff);

                                for (coeff, _) in &mut matrix[r] {
                                    field.mul_assign(coeff, &inv_pivot);
                                }
                            }
                        }
                    }
                }

                if print_stats {
                    println!("\tPivots={}, rows to reduce={}", pc, matrix.len() - pc);
                }

                for r in 0..matrix.len() {
                    if matrix[r].is_empty() {
                        continue;
                    }

                    // do not reduce pivots
                    if pivots.iter().any(|c| *c == Some(r)) {
                        continue;
                    }

                    // copy row into the buffer
                    for (coeff, col) in &*matrix[r] {
                        buffer[*col] = coeff.clone();
                    }

                    for i in 0..buffer.len() {
                        if field.is_zero(&buffer[i]) {
                            continue;
                        }

                        let Some(pivot_index) = pivots[i] else {
                            continue;
                        };

                        let pivot: &Vec<(Self::Element, usize)> = &matrix[pivot_index];
                        let c = buffer[i].clone();

                        buffer[i] = field.zero();

                        for (coeff, col) in pivot.iter().skip(1) {
                            field.sub_mul_assign(&mut buffer[*col], coeff, &c);
                        }
                    }

                    matrix[r].clear();

                    for (col, coeff) in buffer.iter_mut().enumerate() {
                        if !field.is_zero(coeff) {
                            matrix[r].push((coeff.clone(), col));
                            *coeff = field.zero();
                        }
                    }

                    if let Some((coeff, col)) = matrix[r].first() {
                        pivots[*col] = Some(r);
                        let inv_pivot = field.inv(coeff);

                        for (coeff, _) in &mut matrix[r] {
                            field.mul_assign(coeff, &inv_pivot);
                        }
                    }
                }

                matrix.retain(|r| !r.is_empty());
            }

            fn from_larger(&self, element: &Self::LargerField) -> <Self as Set>::Element {
                element.clone()
            }
        }
    };
}

echelonize_impl!(Zp64);
echelonize_impl!(FiniteField<Mersenne64>);
echelonize_impl!(RationalField);
echelonize_impl!(Z2);
echelonize_impl!(AlgebraicExtension<Zp>);
echelonize_impl!(AlgebraicExtension<Zp64>);
echelonize_impl!(AlgebraicExtension<Z2>);
echelonize_impl!(AlgebraicExtension<RationalField>);

#[cfg(test)]
mod test {
    use crate::{
        atom::AtomCore,
        domains::finite_field::Zp,
        parse,
        poly::{GrevLexOrder, groebner::GroebnerBasis, polynomial::MultivariatePolynomial},
    };

    #[test]
    fn cyclic4() {
        let polys = [
            "v1 v2 v3 v4 - 1",
            "v1 v2 v3 + v1 v2 v4 + v1 v3 v4 + v2 v3 v4",
            "v1 v2 + v2 v3 + v1 v4 + v3 v4",
            "v1 + v2 + v3 + v4",
        ];

        let ideal: Vec<MultivariatePolynomial<_, u16>> = polys
            .iter()
            .map(|x| {
                let a = parse!(x).expand();
                a.to_polynomial(&Zp::new(13), None)
            })
            .collect();

        // compute the Groebner basis with lex ordering
        let gb = GroebnerBasis::new(&ideal, false);

        let res = [
            "v4+v3+v2+v1",
            "v4^2+2*v2*v4+v2^2",
            "11*v4^2+v3*v4+v3^2*v4^4-v2*v4+v2*v3",
            "-v4+v4^5-v2+v2*v4^4",
            "-v4-v3+v3^2*v4^3+v3^3*v4^2",
            "1-v4^4-v3^2*v4^2+v3^2*v4^6",
        ];

        let res: Vec<MultivariatePolynomial<_, u16>> = res
            .iter()
            .map(|x| {
                let a = parse!(x).expand();
                a.to_polynomial(&Zp::new(13), ideal[0].variables.clone())
            })
            .collect();

        assert_eq!(gb.system, res);

        // compute the Groebner basis with grevlex ordering by converting the polynomials
        let grevlex_ideal: Vec<_> = ideal.iter().map(|p| p.reorder::<GrevLexOrder>()).collect();
        let gb = GroebnerBasis::new(&grevlex_ideal, false);

        let res = [
            "v4+v3+v2+v1",
            "v4^2+2*v2*v4+v2^2",
            "-v4^3-v2*v4^2+v3^2*v4+v2*v3^2",
            "-1-v4^4+v3*v4^3-v2*v4^3+v3^2*v4^2+v2*v3*v4^2",
            "-v4-v2+v4^5+v2*v4^4",
            "-v4-v3+v3^2*v4^3+v3^3*v4^2",
            "11*v4^2+v3*v4-v2*v4+v2*v3+v3^2*v4^4",
        ];

        let res: Vec<MultivariatePolynomial<_, u16, _>> = res
            .iter()
            .map(|x| {
                let a = parse!(x).expand();
                a.to_polynomial(&Zp::new(13), ideal[0].variables.clone())
                    .reorder::<GrevLexOrder>()
            })
            .collect();

        assert_eq!(gb.system, res);
    }
}
