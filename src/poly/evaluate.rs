//! Efficient evaluation of polynomials.
//!
//! Prefer using [AtomCore::evaluator].
use std::{
    cmp::Reverse,
    hash::{Hash, Hasher},
};

use ahash::{AHasher, HashMap, HashSet, HashSetExt};
use rand::{Rng, rng};

use crate::{
    atom::{Atom, AtomView, KeyLookup},
    domains::{Ring, RingOps, float::Real},
    evaluate::EvaluationFn,
};
use crate::{
    atom::{AtomCore, Symbol},
    coefficient::CoefficientView,
    domains::{
        EuclideanDomain,
        float::FloatLike,
        rational::{Q, Rational, RationalField},
    },
    state::Workspace,
};

use super::{PositiveExponent, polynomial::MultivariatePolynomial};

/// A borrowed version of a Horner node, suitable as a key in a
/// hashmap. It uses precomputed hashes for the complete node
/// `var^pow*content+rest` and for its children `var^pow*content` and `var^pow`.
#[derive(Debug, Clone, Copy)]
pub struct BorrowedHornerNode<'a, R: Ring>
where
    R::Element: Hash + Eq,
{
    pub var: usize,
    pub pow: usize,
    pub gcd: R::Element, // only used for counting number of operations
    pub content: Option<&'a HornerScheme<R>>,
    pub rest: Option<&'a HornerScheme<R>>,
    pub hash: (u64, u64, u64),
}

impl<R: Ring> PartialEq for BorrowedHornerNode<'_, R>
where
    R::Element: Hash + Eq,
{
    fn eq(&self, other: &Self) -> bool {
        // hash and gcd is skipped
        self.var == other.var
            && self.pow == other.pow
            && self.content == other.content
            && self.rest == other.rest
    }
}

impl<R: Ring> Eq for BorrowedHornerNode<'_, R> where R::Element: Hash + Eq {}

impl Hash for BorrowedHornerNode<'_, RationalField> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let hash = if self.content.is_some() {
            if self.rest.is_some() {
                self.hash.2
            } else {
                self.hash.1
            }
        } else {
            self.hash.0
        };

        state.write_u64(hash);
    }
}

impl<'a, R: Ring> From<&'a HornerNode<R>> for BorrowedHornerNode<'a, R>
where
    R::Element: Hash + Eq,
{
    fn from(n: &'a HornerNode<R>) -> Self {
        BorrowedHornerNode {
            var: n.var,
            pow: n.pow,
            gcd: n.gcd.clone(),
            content: n.content_rest.0.as_ref(),
            rest: n.content_rest.1.as_ref(),
            hash: n.hash,
        }
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum BorrowedHornerScheme<'a, R: Ring>
where
    R::Element: Hash + Eq,
{
    Node(BorrowedHornerNode<'a, R>),
    Leaf(&'a R::Element),
}

impl<'a, R: Ring> From<&'a HornerScheme<R>> for BorrowedHornerScheme<'a, R>
where
    R::Element: Hash + Eq,
{
    fn from(value: &'a HornerScheme<R>) -> Self {
        match value {
            HornerScheme::Node(n) => BorrowedHornerScheme::Node(BorrowedHornerNode::from(n)),
            HornerScheme::Leaf(_, l) => BorrowedHornerScheme::Leaf(l),
        }
    }
}

impl<'a> BorrowedHornerScheme<'a, RationalField> {
    /// Determine the number of operations required to evaluate the Horner scheme.
    /// Common subexpressions are only counted once.
    pub fn op_count_cse(&self) -> usize {
        let mut h = HashSet::default();
        self.op_count_cse_impl(&mut h)
    }

    fn op_count_cse_impl(&self, set: &mut HashSet<BorrowedHornerNode<'a, RationalField>>) -> usize {
        match self {
            BorrowedHornerScheme::Node(n) => {
                let gcd_op = if n.gcd.abs() != Rational::one() { 1 } else { 0 };

                // check if n = var^pow*a+b is seen before
                if set.contains(n) {
                    return gcd_op;
                }

                // check if var^pow*a is seen before
                let mut b = n.clone();
                b.rest = None;

                if set.contains(&b) {
                    set.insert(n.clone());
                    gcd_op
                        + 1
                        + n.rest
                            .map(|x| BorrowedHornerScheme::from(x).op_count_cse_impl(set))
                            .unwrap() // can fail now because of gcd?
                } else {
                    // add var^pow to seen list
                    let instr = if n.pow > 1 {
                        let mut c = b.clone();
                        c.content = None;

                        if set.contains(&c) {
                            0
                        } else {
                            set.insert(c.clone());
                            n.pow - 1
                        }
                    } else {
                        0
                    };

                    set.insert(b.clone());
                    set.insert(n.clone());

                    gcd_op
                        + instr
                        + n.content
                            .map(|x| 1 + BorrowedHornerScheme::from(x).op_count_cse_impl(set))
                            .unwrap_or(0)
                        + n.rest
                            .map(|x| 1 + BorrowedHornerScheme::from(x).op_count_cse_impl(set))
                            .unwrap_or(0)
                }
            }
            BorrowedHornerScheme::Leaf(_) => 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HornerNode<R: Ring>
where
    R::Element: Hash + Eq,
{
    pub var: usize,
    pub pow: usize,
    pub gcd: R::Element,
    pub content_rest: Box<(Option<HornerScheme<R>>, Option<HornerScheme<R>>)>,
    pub hash: (u64, u64, u64),
}

impl<R: Ring> PartialEq for HornerNode<R>
where
    R::Element: Hash + Eq,
{
    fn eq(&self, other: &Self) -> bool {
        // hash is skipped, but the gcd is included
        self.var == other.var
            && self.pow == other.pow
            && self.gcd == other.gcd
            && self.content_rest == other.content_rest
    }
}

/// A polynomial written in a Horner scheme, i.e.:
/// `x^2*(x*(y*(z + 1) + y) + 5) + z^2`.
#[derive(Debug, Clone)]
pub enum HornerScheme<R: Ring>
where
    R::Element: Hash + Eq,
{
    Node(HornerNode<R>),
    Leaf(u64, R::Element), // hash and number
}

impl<R: Ring> PartialEq for HornerScheme<R>
where
    R::Element: Hash + Eq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Node(l0), Self::Node(r0)) => l0 == r0,
            (Self::Leaf(_, l1), Self::Leaf(_, r1)) => l1 == r1,
            _ => false,
        }
    }
}

impl HornerScheme<RationalField> {
    /// Evaluate a polynomial written in a Horner scheme. For faster
    /// evaluation, convert the Horner scheme into an `InstructionList`.
    pub fn evaluate(&self, samples: &[Rational]) -> Rational {
        match self {
            HornerScheme::Node(n) => {
                let e = match &n.content_rest.0 {
                    Some(s) => match &n.content_rest.1 {
                        Some(s1) => Q.add(
                            Q.mul(Q.pow(&samples[n.var], n.pow as u64), s.evaluate(samples)),
                            s1.evaluate(samples),
                        ),
                        None => Q.mul(Q.pow(&samples[n.var], n.pow as u64), s.evaluate(samples)),
                    },
                    None => match &n.content_rest.1 {
                        Some(s1) => {
                            Q.add(Q.pow(&samples[n.var], n.pow as u64), s1.evaluate(samples))
                        }
                        None => Q.pow(&samples[n.var], n.pow as u64),
                    },
                };
                &e * &n.gcd
            }
            HornerScheme::Leaf(_, l) => l.clone(),
        }
    }

    /// Clean up the Horner scheme and save the allocations in `boxes`.
    fn cleanup(
        &mut self,
        boxes: &mut Vec<
            Box<(
                Option<HornerScheme<RationalField>>,
                Option<HornerScheme<RationalField>>,
            )>,
        >,
    ) {
        let private = std::mem::replace(self, HornerScheme::Leaf(0, Rational::zero()));
        match private {
            HornerScheme::Node(mut n) => {
                if let Some(c) = &mut n.content_rest.0 {
                    c.cleanup(boxes);
                }
                if let Some(c) = &mut n.content_rest.1 {
                    c.cleanup(boxes);
                }

                *n.content_rest = (None, None);

                boxes.push(n.content_rest);
            }
            HornerScheme::Leaf(_, _) => {}
        }
    }
}

impl<R: Ring> std::fmt::Display for HornerScheme<R>
where
    R::Element: Hash + Eq,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HornerScheme::Leaf(_, l) => f.write_fmt(format_args!("{l:?}")),
            HornerScheme::Node(n) => {
                f.write_fmt(format_args!("+{:?}*(", n.gcd))?;

                if n.pow == 1 {
                    f.write_fmt(format_args!("x{}", n.var))?;
                } else {
                    f.write_fmt(format_args!("x{}^{}", n.var, n.pow))?;
                }
                if let Some(s) = &n.content_rest.0 {
                    f.write_fmt(format_args!("*("))?;
                    s.fmt(f)?;
                    f.write_fmt(format_args!(")"))?;
                }

                if let Some(s) = &n.content_rest.1 {
                    if let HornerScheme::Leaf(_, _) = &s {
                        f.write_str("+")?;
                        s.fmt(f)?;
                    } else {
                        s.fmt(f)?;
                    }
                }

                f.write_str(")")?;

                Ok(())
            }
        }
    }
}

impl<E: PositiveExponent> MultivariatePolynomial<RationalField, E> {
    /// Write the polynomial in a Horner scheme with the variable ordering
    /// defined in `order`.
    pub fn to_horner_scheme(&self, order: &[usize]) -> HornerScheme<RationalField> {
        let mut indices: Vec<_> = (0..self.nterms()).collect();
        let mut power_sub = vec![E::zero(); self.nvars()];
        let mut horner_boxes = vec![];

        self.to_horner_scheme_impl(order, &mut indices, 0, &mut power_sub, &mut horner_boxes)
    }

    /// Create a Horner scheme using the variable order specified in `order`,
    /// using the terms from `self` indexed by `indices[index_start..]`.
    /// The exponents of each term should be reduced by `power_sub`
    /// and allocations from previous Horner scheme constructions can
    /// be provided using `boxes`.
    ///
    /// This function is optimized, as it is called many times during
    /// the Horner scheme optimization.
    fn to_horner_scheme_impl(
        &self,
        order: &[usize],
        indices: &mut Vec<usize>,
        index_start: usize,
        power_sub: &mut [E],
        boxes: &mut Vec<
            Box<(
                Option<HornerScheme<RationalField>>,
                Option<HornerScheme<RationalField>>,
            )>,
        >,
    ) -> HornerScheme<RationalField> {
        if order.is_empty() {
            debug_assert!(indices.len() <= index_start + 1);

            let num = if indices.len() == index_start {
                self.ring.zero()
            } else {
                self.coefficients[indices[index_start]].clone()
            };

            let mut h = AHasher::default();
            h.write_u8(1);
            num.hash(&mut h);
            return HornerScheme::Leaf(h.finish(), num);
        }

        // find the lowest non-zero power of the next variable in the ordering
        let var = order[0];
        let mut min_pow = E::zero();

        for i in &indices[index_start..] {
            let e = self.exponents(*i)[var];
            if e > power_sub[var] {
                let p = e - power_sub[var];
                if min_pow == E::zero() || p < min_pow {
                    min_pow = p;
                }
            }
        }

        if min_pow == E::zero() {
            return self.to_horner_scheme_impl(&order[1..], indices, index_start, power_sub, boxes);
        }

        let new_index_start = indices.len();

        // create the branch for terms that do not contain var^min_pow
        for t in index_start..new_index_start {
            if self.exponents(indices[t])[var] <= power_sub[var] {
                indices.push(indices[t]);
            }
        }

        let mut rest =
            self.to_horner_scheme_impl(&order[1..], indices, new_index_start, power_sub, boxes);

        // create the branch for terms that do contain var^min_pow and lower the power
        indices.truncate(new_index_start);
        for t in index_start..new_index_start {
            if self.exponents(indices[t])[var] > power_sub[var] {
                indices.push(indices[t]);
            }
        }

        power_sub[var] += min_pow;

        let mut content =
            self.to_horner_scheme_impl(order, indices, new_index_start, power_sub, boxes);

        power_sub[var] = power_sub[var] - min_pow;

        indices.truncate(new_index_start);

        // compute the gcd of both branches
        // normalize such that the first branch is positive
        let mut gcd = match &content {
            HornerScheme::Node(n) => n.gcd.clone(),
            HornerScheme::Leaf(_, l) => l.clone(),
        };

        let gcd_norm = match &content {
            HornerScheme::Node(n) => {
                if n.gcd.is_negative() {
                    (-1, 1).into()
                } else {
                    (1, 1).into()
                }
            }
            HornerScheme::Leaf(_, l) => {
                if l.is_negative() {
                    (-1, 1).into()
                } else {
                    (1, 1).into()
                }
            }
        };

        gcd = Q.gcd(
            &gcd,
            match &rest {
                HornerScheme::Node(n) => &n.gcd,
                HornerScheme::Leaf(_, l) => l,
            },
        );

        gcd *= &gcd_norm;

        if !gcd.is_one() {
            for s in [&mut content, &mut rest] {
                match s {
                    HornerScheme::Node(n) => n.gcd = &n.gcd / &gcd,
                    HornerScheme::Leaf(n, l) => {
                        *l = &*l / &gcd;

                        // overwrite the hash of the number
                        let mut h = AHasher::default();
                        h.write_u8(1);
                        l.hash(&mut h);
                        *n = h.finish();
                    }
                };
            }
        }

        // compute the hash of the node and its components
        let mut h = AHasher::default();
        h.write_u8(0);
        var.hash(&mut h);
        (min_pow.to_i32() as usize).hash(&mut h);

        let pow_hash = h.finish(); // hash var^pow

        match &content {
            HornerScheme::Node(n) => h.write_u64(n.hash.2),
            HornerScheme::Leaf(n, _) => h.write_u64(*n),
        }

        let pow_content_hash = h.finish(); // hash var^pow*content

        match &rest {
            HornerScheme::Node(n) => h.write_u64(n.hash.2),
            HornerScheme::Leaf(n, _) => h.write_u64(*n),
        }

        let full_hash = h.finish(); // hash var^pow*content+rest

        let children = (
            if let HornerScheme::Leaf(_, r) = &content {
                if r.is_one() { None } else { Some(content) }
            } else {
                Some(content)
            },
            if let HornerScheme::Leaf(_, r) = &rest {
                if r.is_zero() { None } else { Some(rest) }
            } else {
                Some(rest)
            },
        );

        // recycle the box if it is available
        let boxed_children = if let Some(mut b) = boxes.pop() {
            *b = children;
            b
        } else {
            Box::new(children)
        };

        HornerScheme::Node(HornerNode {
            var,
            pow: min_pow.to_i32() as usize,
            gcd,
            hash: (pow_hash, pow_content_hash, full_hash),
            content_rest: boxed_children,
        })
    }

    /// Find the Horner scheme that minimizes the number of operations,
    /// taking common subexpressions into account.
    pub fn optimize_horner_scheme(
        &self,
        num_tries: usize,
    ) -> (HornerScheme<RationalField>, usize, Vec<usize>) {
        let (mut hs, op_count, scheme) =
            HornerScheme::optimize_multiple(std::slice::from_ref(&self), num_tries);
        (hs.pop().unwrap(), op_count, scheme)
    }

    /// Optimize an expression for evaluation, given `num_iter` tries.
    pub fn optimize(&self, num_iter: usize) -> InstructionListOutput<Rational> {
        let (h, _ops, _scheme) = self.optimize_horner_scheme(num_iter);
        let mut i = h.to_instr(self.nvars());
        i.fuse_operations();
        while i.common_pair_elimination() {
            i.fuse_operations();
        }

        i.to_output(self.variables.as_ref().to_vec(), true)
    }
}

impl HornerScheme<RationalField> {
    pub fn optimize_multiple<E: PositiveExponent>(
        polys: &[&MultivariatePolynomial<RationalField, E>],
        num_tries: usize,
    ) -> (Vec<HornerScheme<RationalField>>, usize, Vec<usize>) {
        if polys.is_empty() {
            return (vec![], 0, vec![]);
        }

        assert!(
            polys
                .windows(2)
                .all(|r| r[0].variables == r[1].variables && r[0].nvars() == r[1].nvars()),
            "Variable maps of all polynomials must be the same"
        );

        // the starting scheme is the descending order of occurrence of variables
        let mut occurrence: Vec<_> = (0..polys[0].nvars()).map(|x| (x, 0)).collect();
        for es in polys[0].exponents.chunks(polys[0].nvars()) {
            for ((_, o), e) in occurrence.iter_mut().zip(es) {
                if *e > E::zero() {
                    *o += 1;
                }
            }
        }
        occurrence.sort_by_key(|e| Reverse(e.1));

        let mut scheme: Vec<_> = occurrence.into_iter().map(|(v, _)| v).collect();

        let mut indices: Vec<_> = vec![];
        let mut power_sub = vec![E::zero(); polys[0].nvars()];

        let mut horner_boxes = vec![];

        let mut best = Vec::with_capacity(polys.len());
        let mut best_score = 0;
        for x in polys {
            indices.clear();
            indices.extend(0..x.nterms());

            let h = x.to_horner_scheme_impl(
                &scheme,
                &mut indices,
                0,
                &mut power_sub,
                &mut horner_boxes,
            );
            best_score += BorrowedHornerScheme::from(&h).op_count_cse();
            best.push(h);
        }

        let mut best_scheme = scheme.clone();
        let mut rng = rng();

        let mut new_best = Vec::with_capacity(polys.len());

        // TODO: for few variables, test all permutations
        for i in 0..num_tries {
            let a = rng.random_range(0..polys[0].nvars());
            let b = rng.random_range(0..polys[0].nvars());
            scheme.swap(a, b);

            let mut new_oc = 0;

            // use the same hash set for all polynomials
            let mut hash_set = HashSet::with_capacity(best_score * 2);
            for x in polys {
                indices.clear();
                indices.extend(0..x.nterms());

                let h = x.to_horner_scheme_impl(
                    &scheme,
                    &mut indices,
                    0,
                    &mut power_sub,
                    &mut horner_boxes,
                );
                new_best.push(h);
            }

            for x in &new_best {
                new_oc += BorrowedHornerScheme::from(x).op_count_cse_impl(&mut hash_set);
            }

            if new_oc <= best_score {
                // accept move
                for mut x in best.drain(..) {
                    x.cleanup(&mut horner_boxes);
                }

                for x in new_best.drain(..) {
                    best.push(x);
                }

                best_score = new_oc;
                best_scheme.copy_from_slice(&scheme);
            } else {
                for mut x in new_best.drain(..) {
                    x.cleanup(&mut horner_boxes);
                }
            }

            // see if we reject the change
            if new_oc as f64 / best_score as f64 > 1. + 0.5 * (1. - i as f64 / num_tries as f64) {
                //if new_oc as f64 > best_score as f64 {
                scheme.swap(a, b);
            }
        }

        (best, best_score, best_scheme)
    }

    fn get_constants(&self, constants: &mut HashMap<Rational, usize>, shift: usize) {
        match self {
            HornerScheme::Node(n) => {
                if n.gcd != Rational::one() && !constants.contains_key(&n.gcd) {
                    constants.insert(n.gcd.clone(), constants.len() + shift);
                }
                if let Some(content) = n.content_rest.0.as_ref() {
                    content.get_constants(constants, shift);
                }
                if let Some(content) = n.content_rest.1.as_ref() {
                    content.get_constants(constants, shift);
                }
            }
            HornerScheme::Leaf(_, l) => {
                if !constants.contains_key(l) {
                    constants.insert(l.clone(), constants.len() + shift);
                }
            }
        }
    }

    /// Convert the Horner schemes to a list of instructions, suitable for numerical evaluation.
    pub fn to_instr_multiple(
        schemes: &[HornerScheme<RationalField>],
        nvars: usize,
    ) -> InstructionList {
        let mut seen: HashMap<BorrowedHornerNode<'_, RationalField>, usize> = HashMap::default();

        // collect all unique constants
        let mut constant_map: HashMap<Rational, usize> = HashMap::default();

        for s in schemes {
            s.get_constants(&mut constant_map, nvars);
        }

        let mut constants: Vec<_> = constant_map.iter().collect();
        constants.sort_by_key(|(_, c)| *c);

        let mut instr: Vec<_> = (0..nvars)
            .map(|i| Instruction::Init(Variable::Var(i, None)))
            .collect();

        for x in constants {
            instr.push(Instruction::Init(Variable::Constant(x.0.clone())));
        }

        for s in schemes {
            let v = s.to_instr_rec(&mut instr, &mut seen, &mut constant_map);
            instr.push(Instruction::Yield(v));
        }

        InstructionList { instr }
    }

    /// Convert the Horner scheme to a list of instructions, suitable for numerical evaluation.
    pub fn to_instr(&self, nvars: usize) -> InstructionList {
        Self::to_instr_multiple(std::slice::from_ref(self), nvars)
    }

    fn to_instr_rec<'a>(
        &'a self,
        instr: &mut Vec<Instruction<Rational>>,
        seen: &mut HashMap<BorrowedHornerNode<'a, RationalField>, usize>,
        constant_seen: &mut HashMap<Rational, usize>,
    ) -> usize {
        match self {
            HornerScheme::Node(n) => {
                let nb = BorrowedHornerNode::from(n);

                // check if n = var^pow*content+rest is seen before
                if let Some(v) = seen.get(&nb) {
                    return if n.gcd != Rational::one() {
                        let gcd = *constant_seen.get(&n.gcd).unwrap();
                        instr.push(Instruction::Mul(vec![gcd, *v]));
                        instr.len() - 1
                    } else {
                        *v
                    };
                }

                // check if var^pow*content is seen before
                let mut b = BorrowedHornerNode::from(n);
                b.rest = None;

                let v1 = if let Some(v) = seen.get(&b) {
                    *v
                } else {
                    // check if var^pow is seen before
                    let vp = if n.pow > 1 {
                        let mut c = b.clone();
                        c.content = None;
                        if let Some(v) = seen.get(&c) {
                            *v
                        } else {
                            /// Binary exponentiation
                            fn bin_exp(
                                var: usize,
                                p: usize,
                                instr: &mut Vec<Instruction<Rational>>,
                                seen: &mut HashMap<BorrowedHornerNode<'_, RationalField>, usize>,
                            ) -> usize {
                                if p == 1 {
                                    return var;
                                }

                                // create a borrowed node with the proper power
                                let mut h = AHasher::default();
                                h.write_u8(0);
                                var.hash(&mut h);
                                p.hash(&mut h);
                                let hash = h.finish();
                                let a = BorrowedHornerNode {
                                    var,
                                    pow: p,
                                    gcd: Rational::one(),
                                    content: None,
                                    rest: None,
                                    hash: (hash, hash, hash),
                                };

                                if let Some(a) = seen.get(&a) {
                                    return *a;
                                }

                                if p.is_multiple_of(2) {
                                    let p_half = bin_exp(var, p / 2, instr, seen);
                                    instr.push(Instruction::Mul(vec![p_half, p_half]));
                                } else {
                                    let p_minone = bin_exp(var, p - 1, instr, seen);
                                    instr.push(Instruction::Mul(vec![var, p_minone]));
                                }

                                seen.insert(a, instr.len() - 1);

                                instr.len() - 1
                            }

                            bin_exp(n.var, n.pow, instr, seen)
                        }
                    } else {
                        n.var
                    };

                    if let Some(s) = &n.content_rest.0 {
                        let v1 = s.to_instr_rec(instr, seen, constant_seen);

                        if v1 < vp {
                            instr.push(Instruction::Mul(vec![v1, vp]));
                        } else {
                            instr.push(Instruction::Mul(vec![vp, v1]));
                        }
                        seen.insert(b, instr.len() - 1);
                        instr.len() - 1
                    } else {
                        vp
                    }
                };

                let vr = if let Some(s) = &n.content_rest.1 {
                    let v2 = s.to_instr_rec(instr, seen, constant_seen);

                    if v1 < v2 {
                        instr.push(Instruction::Add(vec![v1, v2]));
                    } else {
                        instr.push(Instruction::Add(vec![v2, v1]));
                    }

                    seen.insert(nb, instr.len() - 1);
                    instr.len() - 1
                } else {
                    v1
                };

                if n.gcd != Rational::one() {
                    let gcd = *constant_seen.get(&n.gcd).unwrap();
                    instr.push(Instruction::Mul(vec![gcd, vr]));
                    instr.len() - 1
                } else {
                    vr
                }
            }
            HornerScheme::Leaf(_, l) => *constant_seen.get(l).unwrap(),
        }
    }
}

// An arithmetical instruction that is part of an `InstructionList`.
#[derive(Debug, Clone)]
pub enum Instruction<N: FloatLike> {
    Init(Variable<N>),
    Add(Vec<usize>),
    Mul(Vec<usize>),
    Yield(usize),
    Empty,
}

// An variable that is part of an `InstructionList`,
// which may refer to another instruction in the instruction list.
#[derive(Debug, Clone, PartialEq)]
pub enum Variable<N: FloatLike> {
    Var(usize, Option<usize>), // var or var[index]
    Constant(N),
}

impl Variable<Rational> {
    fn to_pretty_string(
        &self,
        var_map: &[super::PolyVariable],
        mode: InstructionSetMode,
    ) -> String {
        match self {
            Variable::Var(v, index) => {
                // convert f(0) to f[0]
                if let super::PolyVariable::Function(_, f) = &var_map[*v]
                    && let AtomView::Fun(f) = f.as_view()
                    && f.get_nargs() == 1
                    && let Some(a) = f.iter().next()
                    && let AtomView::Num(n) = a
                    && let CoefficientView::Natural(n, d, ni, _di) = n.get_coeff_view()
                    && d == 1
                    && ni == 0
                    && n >= 0
                {
                    return format!("{}[{}]", f.get_symbol(), a);
                }

                let mut s = var_map[*v].to_string();

                if let Some(index) = index {
                    s.push_str(&format!("[{index}]"));
                }

                s
            }
            Variable::Constant(c) => match mode {
                InstructionSetMode::Plain => format!("{c}"),
                InstructionSetMode::CPP(_) => {
                    if c.is_integer() {
                        format!("T({})", c.numerator_ref())
                    } else {
                        format!("T({})/T({})", c.numerator_ref(), c.denominator_ref())
                    }
                }
            },
        }
    }
}

impl<N: FloatLike> std::fmt::Display for Variable<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Variable::Var(v, index) => {
                if let Some(index) = index {
                    f.write_fmt(format_args!("x[{v}][{index}]"))
                } else {
                    f.write_fmt(format_args!("x{v}"))
                }
            }
            Variable::Constant(c) => <N as std::fmt::Display>::fmt(c, f),
        }
    }
}

impl<N: FloatLike> Variable<N> {
    pub fn convert<NO: FloatLike, F: Fn(&N) -> NO>(&self, coeff_map: F) -> Variable<NO> {
        match self {
            Variable::Var(v, index) => Variable::Var(*v, *index),
            Variable::Constant(c) => Variable::Constant(coeff_map(c)),
        }
    }
}

/// A list of instructions suitable for fast numerical evaluation.
pub struct InstructionList {
    instr: Vec<Instruction<Rational>>,
}

impl InstructionList {
    /// Evaluate the instructions and yield the result.
    /// For a more efficient evaluation, call `to_output()` first.
    pub fn evaluate(&self, samples: &[Rational]) -> Rational {
        let mut eval: Vec<Rational> = vec![Rational::zero(); self.instr.len()];

        for (reg, x) in self.instr.iter().enumerate() {
            match x {
                Instruction::Add(a) => {
                    let mut r = Rational::zero();
                    for x in a {
                        r += &eval[*x];
                    }
                    eval[reg] = r;
                }
                Instruction::Mul(m) => {
                    let mut r = Rational::one();
                    for x in m {
                        r *= &eval[*x];
                    }
                    eval[reg] = r;
                }
                Instruction::Init(i) => match i {
                    Variable::Var(v, _index) => eval[reg] = samples[*v].clone(),
                    Variable::Constant(c) => eval[reg] = c.clone(),
                },
                Instruction::Yield(y) => return eval[*y].clone(),
                Instruction::Empty => {}
            }
        }
        unreachable!()
    }

    /// Return the number of arithmetical operations required for evaluation.
    pub fn op_count(&self) -> usize {
        let mut sum = 0;
        // TODO: detect -1 in init and discount it to stay in line with the FORM counting
        for x in &self.instr {
            sum += match x {
                Instruction::Add(a) => a.len() - 1,
                Instruction::Mul(m) => m.len() - 1,
                Instruction::Yield(_) => 0,
                Instruction::Empty => 0,
                Instruction::Init(_) => 0,
            };
        }
        sum
    }

    /// Fuse `Z1=a+b`, `Z2=Z1+c` to `Z1=a+b+c` if `Z1` is only used in `Z2`
    pub fn fuse_operations(&mut self) {
        let mut use_count: Vec<usize> = vec![0; self.instr.len()];

        for (i, x) in self.instr.iter().enumerate() {
            match x {
                Instruction::Add(a) => {
                    for &v in a {
                        if let Instruction::Mul(_) = self.instr[v] {
                            use_count[v] = 2; // different type, so disable
                        } else {
                            use_count[v] += 1;
                        }
                    }
                }
                Instruction::Mul(m) => {
                    for &v in m {
                        if let Instruction::Add(_) = self.instr[v] {
                            use_count[v] = 2;
                        } else {
                            use_count[v] += 1;
                        }
                    }
                }
                Instruction::Yield(v) => {
                    use_count[*v] = 2; // always different type
                }
                Instruction::Empty => {}
                Instruction::Init(_) => {
                    use_count[i] = 2;
                }
            };
        }

        for i in 0..self.instr.len() {
            // we could be in chain of single use -> single use -> etc so work from the start
            if let Instruction::Add(a) | Instruction::Mul(a) = &self.instr[i]
                && a.iter().any(|v| use_count[*v] == 1)
            {
                let mut instr = std::mem::replace(&mut self.instr[i], Instruction::Empty);

                if let Instruction::Add(a) | Instruction::Mul(a) = &mut instr {
                    let mut new_a = Vec::with_capacity(a.len());
                    for v in a.drain(..) {
                        if use_count[v] == 1 {
                            if let Instruction::Add(aa) | Instruction::Mul(aa) = &self.instr[v] {
                                for x in aa {
                                    new_a.push(*x);
                                }
                                self.instr[v] = Instruction::Empty;
                            } else {
                                unreachable!()
                            }
                        } else {
                            new_a.push(v);
                        }
                    }
                    new_a.sort();
                    *a = new_a;
                }

                self.instr[i] = instr;
            }
        }

        self.remove_empty_ops();
    }

    /// Remove empty instructions from the list.
    fn remove_empty_ops(&mut self) {
        // now remove the old labels and renumber all
        let mut num_removed_entries = 0;
        let mut new_instr = Vec::with_capacity(self.instr.len());
        let mut cum_step = vec![0; self.instr.len()];
        for (i, mut x) in std::mem::take(&mut self.instr).into_iter().enumerate() {
            cum_step[i] = num_removed_entries;
            if let Instruction::Empty = x {
                num_removed_entries += 1;
                continue;
            }

            match &mut x {
                Instruction::Add(a) | Instruction::Mul(a) => {
                    for v in a {
                        *v -= cum_step[*v];
                    }
                }
                Instruction::Yield(v) => {
                    *v -= cum_step[*v];
                }
                Instruction::Empty => {}
                Instruction::Init(_) => {}
            };
            new_instr.push(x);
        }

        self.instr = new_instr;
    }

    /// Find and extract pairs of variables that appear in more than one instruction.
    /// This reduces the number of operations. Returns `true` iff an extraction could be performed.
    ///
    /// This function can be called multiple times such that common subexpressions that
    /// are larger than pairs can also be extracted.
    pub fn common_pair_elimination(&mut self) -> bool {
        let mut pair_count = HashMap::default();
        let mut last_init = 0;

        let mut d = vec![];
        let mut rep = vec![];
        for (i, x) in self.instr.iter().enumerate() {
            match x {
                Instruction::Add(m) | Instruction::Mul(m) => {
                    d.clone_from(m);
                    d.dedup();
                    rep.clear();
                    rep.resize(d.len(), 0);

                    for (c, v) in rep.iter_mut().zip(&d) {
                        for v2 in m {
                            if v == v2 {
                                *c += 1;
                            }
                        }
                    }

                    for i in 0..d.len() - 1 {
                        if rep[i] > 2 {
                            *pair_count
                                .entry((matches!(x, Instruction::Add(_)), d[i], d[i]))
                                .or_insert(0) += rep[i] / 2;
                        }

                        for j in i + 1..d.len() {
                            *pair_count
                                .entry((matches!(x, Instruction::Add(_)), d[i], d[j]))
                                .or_insert(0) += rep[i].min(rep[j]);
                        }
                    }
                }
                Instruction::Yield(_) | Instruction::Empty => {}
                Instruction::Init(_) => {
                    last_init = i + 1;
                }
            };
        }

        let mut v: Vec<_> = pair_count.into_iter().collect();
        v.sort_by_key(|k| Reverse(k.1));

        if v.is_empty() || v[0].1 < 2 {
            return false;
        }

        // TODO: instead of only doing the first, do all non-overlapping replacements

        let (is_add, idx1, idx2) = v[0].0;

        let insert_index = last_init.max(idx1 + 1).max(idx2 + 1);
        self.instr.insert(
            insert_index,
            if is_add {
                Instruction::Add(vec![idx1, idx2])
            } else {
                Instruction::Mul(vec![idx1, idx2])
            },
        );

        // substitute the pair in all instructions
        for x in self.instr.iter_mut().skip(insert_index + 1) {
            let same_op = is_add == matches!(x, Instruction::Add(_));

            match x {
                Instruction::Add(a) | Instruction::Mul(a) => {
                    for v in &mut *a {
                        if *v >= insert_index {
                            *v += 1;
                        }
                    }

                    if !same_op {
                        continue;
                    }

                    if idx1 == idx2 {
                        let count = a.iter().filter(|x| *x == &idx1).count();
                        let pairs = count / 2;
                        if pairs > 0 {
                            a.retain(|x| x != &idx1);

                            if count % 2 == 1 {
                                a.push(idx1);
                            }

                            a.extend(std::iter::repeat_n(insert_index, pairs));
                            a.sort();
                        }
                    } else {
                        let mut idx1_count = 0;
                        let mut idx2_count = 0;
                        for v in &*a {
                            if *v == idx1 {
                                idx1_count += 1;
                            }
                            if *v == idx2 {
                                idx2_count += 1;
                            }
                        }

                        let pair_count = idx1_count.min(idx2_count);

                        if pair_count > 0 {
                            a.retain(|x| x != &idx1 && x != &idx2);

                            // add back removed indices in cases such as idx1*idx2*idx2
                            if idx1_count > pair_count {
                                a.extend(std::iter::repeat_n(idx1, idx1_count - pair_count));
                            }
                            if idx2_count > pair_count {
                                a.extend(std::iter::repeat_n(idx2, idx2_count - pair_count));
                            }

                            // TODO: Z2=Z1 can be detected here with a.is_empty() && pair_count == 1
                            a.extend(std::iter::repeat_n(insert_index, pair_count));
                            a.sort();
                        }
                    }
                }
                Instruction::Yield(v) => {
                    if *v >= insert_index {
                        *v += 1;
                    }
                }
                Instruction::Empty | Instruction::Init(_) => {}
            };
        }

        // remove trivial relabeling instructions, e.g. Z2=Z1, that could have been created
        let mut map: Vec<_> = (0..self.instr.len()).collect();
        for (i, x) in self.instr.iter_mut().enumerate() {
            match x {
                Instruction::Add(a) | Instruction::Mul(a) => {
                    if a.len() == 1 {
                        map[i] = a[0];
                        *x = Instruction::Empty;
                        continue;
                    }

                    for v in a {
                        *v = map[*v];
                    }
                }
                Instruction::Yield(v) => {
                    *v = map[*v];
                }
                Instruction::Empty => {}
                Instruction::Init(_) => {}
            };
        }

        self.remove_empty_ops();

        true
    }

    /// Convert the instruction list into its output format, where
    /// intermediate registers can be recycled. This vastly improves
    /// the computational memory needed for evaluations.
    pub fn to_output(
        self,
        input_map: Vec<super::PolyVariable>,
        recycle_registers: bool,
    ) -> InstructionListOutput<Rational> {
        if !recycle_registers {
            return InstructionListOutput {
                instr: self.instr.into_iter().enumerate().collect(),
                input_map,
            };
        }

        let mut last_use: Vec<usize> = (0..self.instr.len()).collect();

        for (i, x) in self.instr.iter().enumerate() {
            match x {
                Instruction::Add(a) | Instruction::Mul(a) => {
                    for v in a {
                        last_use[*v] = i;
                    }
                }
                Instruction::Yield(v) => {
                    last_use[*v] = i;
                }
                Instruction::Empty | Instruction::Init(_) => {}
            };
        }

        // prevent init slots from being overwritten
        for (i, x) in self.instr.iter().enumerate() {
            if let Instruction::Init(_) = x {
                last_use[i] = self.instr.len();
            }
        }

        let mut rename_map: Vec<_> = (0..self.instr.len()).collect();

        let mut output = Vec::with_capacity(self.instr.len());

        for (i, mut x) in self.instr.into_iter().enumerate() {
            let cur_last_use = last_use[i];
            // find first free variable
            let reg = if let Some((new_v, lu)) =
                last_use[..i].iter_mut().enumerate().find(|(_, r)| **r <= i)
            {
                *lu = cur_last_use; // set the last use to the current variable last use
                last_use[i] = 0; // make the current index available
                rename_map[i] = new_v; // set the rename map so that every occurrence on the rhs is replaced
                new_v
            } else {
                i
            };

            match &mut x {
                Instruction::Add(a) | Instruction::Mul(a) => {
                    for v in a {
                        *v = rename_map[*v];
                    }
                }
                Instruction::Yield(v) => {
                    *v = rename_map[*v];
                }
                Instruction::Empty | Instruction::Init(_) => {}
            };

            output.push((reg, x));
        }

        InstructionListOutput {
            instr: output,
            input_map,
        }
    }
}

/// A list of instructions suitable for fast numerical evaluation.
pub struct InstructionListOutput<N: FloatLike> {
    instr: Vec<(usize, Instruction<N>)>,
    input_map: Vec<super::PolyVariable>,
}

/// An efficient structure that performs a range of operations.
/// `Add(reg,index,len)` means: `eval[reg] = eval[indices[index]] +...+ eval[indices[index_pos]]`.
#[derive(Clone, Copy)]
enum InstructionRange {
    Add(usize, usize, usize), // reg, index, len
    Mul(usize, usize, usize),
    Out(usize),
}
/// A fast polynomial evaluator that evaluates polynomials written
/// in the form:
/// ```text
/// Z0 = x
/// Z1 = y
/// Z2 = 2.
/// Z3 = 5.
/// Z4 = Z0*Z2
/// Z5 = Z3+Z1+Z0
/// Z4 = Z4*Z3
/// ```
/// where `Z0,Z1` is the range of sample points that get overwritten
/// at every call, `Z2,Z3` is a range of numerical constants
/// that are set only once, and `Z4,Z5` are instructions that
/// get overwritten at every call. These instructions only use
/// indices in the `Z` array and their evaluation can therefore
/// be done efficiently.
#[derive(Clone)]
pub struct InstructionEvaluator<N: FloatLike> {
    input_map: Vec<super::PolyVariable>,
    instr: Vec<InstructionRange>,
    indices: Vec<usize>,
    eval: Vec<N>, // evaluation buffer
    out: Vec<N>,  // output buffer
}

impl<N: FloatLike> InstructionEvaluator<N> {
    pub fn output_len(&self) -> usize {
        let mut len = 0;
        for x in &self.instr {
            if let InstructionRange::Out(pos) = x {
                len = len.max(*pos + 1);
            }
        }
        len
    }

    /// Evaluate the converted polynomials at a given sample point and
    /// write the values in `out`.
    ///
    /// The user must ensure that `samples` has the
    /// same length as the number of variables in the
    /// polynomials (including non-occurring ones).
    pub fn evaluate_with_input(&mut self, samples: &[N]) -> &[N] {
        // write the sample point into the evaluation buffer
        // all constant numbers are still in the evaluation buffer
        self.eval[..samples.len()].clone_from_slice(samples);

        self.evaluate_impl()
    }

    fn evaluate_impl(&mut self) -> &[N] {
        macro_rules! get_eval {
            ($i:expr) => {
                unsafe { self.eval.get_unchecked(*self.indices.get_unchecked($i)) }
            };
        }

        let mut out_counter = 0;

        for x in &self.instr {
            match x {
                InstructionRange::Add(reg, pos, len) => {
                    // unroll the loop for additional performance
                    *unsafe { self.eval.get_unchecked_mut(*reg) } = match len {
                        2 => get_eval!(*pos).clone() + get_eval!(*pos + 1),
                        3 => get_eval!(*pos).clone() + get_eval!(*pos + 1) + get_eval!(*pos + 2),
                        4 => {
                            get_eval!(*pos).clone()
                                + get_eval!(*pos + 1)
                                + get_eval!(*pos + 2)
                                + get_eval!(*pos + 3)
                        }
                        _ => {
                            let mut tmp = unsafe {
                                self.eval
                                    .get_unchecked(*self.indices.get_unchecked(*pos))
                                    .clone()
                            };
                            for aa in unsafe { self.indices.get_unchecked(*pos + 1..(pos + len)) } {
                                tmp += unsafe { self.eval.get_unchecked(*aa) };
                            }
                            tmp
                        }
                    };
                }
                InstructionRange::Mul(reg, pos, len) => {
                    *unsafe { self.eval.get_unchecked_mut(*reg) } = match len {
                        2 => get_eval!(*pos).clone() * get_eval!(*pos + 1),
                        3 => get_eval!(*pos).clone() * get_eval!(*pos + 1) * get_eval!(*pos + 2),
                        4 => {
                            get_eval!(*pos).clone()
                                * get_eval!(*pos + 1)
                                * get_eval!(*pos + 2)
                                * get_eval!(*pos + 3)
                        }
                        _ => {
                            let mut tmp = unsafe {
                                self.eval
                                    .get_unchecked(*self.indices.get_unchecked(*pos))
                                    .clone()
                            };
                            for aa in unsafe { self.indices.get_unchecked(*pos + 1..(pos + len)) } {
                                tmp *= unsafe { self.eval.get_unchecked(*aa) };
                            }
                            tmp
                        }
                    };
                }
                InstructionRange::Out(pos) => {
                    unsafe {
                        *self.out.get_unchecked_mut(out_counter) =
                            self.eval.get_unchecked(*pos).clone()
                    };
                    out_counter += 1;
                }
            }
        }

        &self.out
    }
}

impl<N: Real + for<'b> From<&'b Rational>> InstructionEvaluator<N> {
    /// Evaluate all instructions, using a constant map and a function map for the input variables.
    /// The constant map can map any literal expression to a value, for example
    /// a variable or a function with fixed arguments.
    ///
    /// All variables and all user functions in the expression must occur in the map.
    pub fn evaluate<A: AtomCore + KeyLookup, F: Fn(&Rational) -> N + Copy>(
        &mut self,
        coeff_map: F,
        const_map: &HashMap<A, N>,
        function_map: &HashMap<Symbol, EvaluationFn<A, N>>,
    ) -> &[N] {
        Workspace::get_local().with(|ws| {
            for (input, expr) in self.eval.iter_mut().zip(&self.input_map) {
                match expr {
                    super::PolyVariable::Symbol(s) => {
                        *input = const_map
                            .get(ws.new_var(*s).as_view().get_data())
                            .expect("Variable not found")
                            .clone();
                    }
                    super::PolyVariable::Function(_, o) | super::PolyVariable::Power(o) => {
                        *input = o.evaluate(coeff_map, const_map, function_map).unwrap();
                    }
                    super::PolyVariable::Temporary(_) => panic!("Temporary variable in input"),
                }
            }
        });

        self.evaluate_impl()
    }
}

impl<N: FloatLike> InstructionListOutput<N> {
    /// Convert all numbers in the instruction list from the field `N` to the field `NO`.
    pub fn convert<'a, NO: FloatLike + for<'b> From<&'b N>>(&'a self) -> InstructionListOutput<NO> {
        self.convert_with_map(|x| x.into())
    }

    /// Convert all numbers in the instruction list from the field `N` to the field `NO`,
    /// using a custom map function.
    pub fn convert_with_map<NO: FloatLike, F: Fn(&N) -> NO + Copy>(
        &self,
        coeff_map: F,
    ) -> InstructionListOutput<NO> {
        let mut instr = Vec::with_capacity(self.instr.len());

        for (reg, inst) in &self.instr {
            let new_instr = match inst {
                Instruction::Add(a) => Instruction::Add(a.clone()),
                Instruction::Mul(a) => Instruction::Mul(a.clone()),
                Instruction::Yield(y) => Instruction::Yield(*y),
                Instruction::Empty => unreachable!("No empty slots allowed in output"),
                Instruction::Init(v) => Instruction::Init(v.convert(coeff_map)),
            };
            instr.push((*reg, new_instr));
        }
        InstructionListOutput {
            instr,
            input_map: self.input_map.clone(),
        }
    }

    /// Create a fast numerical evaluator.
    pub fn evaluator(&self) -> InstructionEvaluator<N> {
        let mut eval = vec![N::new_zero(); self.instr.len()];

        let mut out_counter = 0;
        let mut simple_instr = vec![];
        let mut indices: Vec<usize> = vec![];
        for (reg, ins) in &self.instr {
            match ins {
                Instruction::Init(x) => {
                    // fill in all constants into the buffer
                    // these entries will never be overwritten
                    if let Variable::Constant(c) = x {
                        eval[*reg] = c.clone();
                    }
                }
                Instruction::Add(a) => {
                    let len = indices.len();
                    indices.extend(a);
                    simple_instr.push(InstructionRange::Add(*reg, len, indices.len() - len));
                }
                Instruction::Mul(a) => {
                    let len = indices.len();
                    indices.extend(a);
                    simple_instr.push(InstructionRange::Mul(*reg, len, indices.len() - len));
                }
                Instruction::Yield(i) => {
                    simple_instr.push(InstructionRange::Out(*i));
                    out_counter += 1;
                }
                Instruction::Empty => {}
            }
        }

        InstructionEvaluator {
            input_map: self.input_map.clone(),
            instr: simple_instr,
            indices,
            eval,
            out: vec![N::new_zero(); out_counter],
        }
    }
}

impl std::fmt::Display for InstructionList {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut out_counter = 0;
        for (reg, x) in self.instr.iter().enumerate() {
            match x {
                Instruction::Add(a) => f.write_fmt(format_args!(
                    "Z{} = {};\n",
                    reg,
                    a.iter()
                        .map(|x| format!("Z{x}"))
                        .collect::<Vec<_>>()
                        .join("+")
                ))?,
                Instruction::Mul(m) => f.write_fmt(format_args!(
                    "Z{} = {};\n",
                    reg,
                    m.iter()
                        .map(|x| format!("Z{x}"))
                        .collect::<Vec<_>>()
                        .join("*")
                ))?,
                Instruction::Yield(y) => {
                    f.write_fmt(format_args!("OUT{out_counter} = Z{y};\n"))?;
                    out_counter += 1;
                }
                Instruction::Empty => f.write_fmt(format_args!("Z{reg} = NOP;\n"))?,
                Instruction::Init(i) => f.write_fmt(format_args!("Z{reg} = {i};\n"))?,
            }
        }

        Ok(())
    }
}

pub struct CPPPrinter {}

impl CPPPrinter {
    pub fn format_number(num: &Rational) -> String {
        if num.is_integer() {
            format!("T({})", num.numerator_ref())
        } else {
            format!("T({})/T({})", num.numerator_ref(), num.denominator_ref())
        }
    }
}

#[derive(Clone, Copy)]
pub struct InstructionSetModeCPPSettings {
    pub write_header_and_test: bool,
    pub always_pass_output_array: bool,
}

#[derive(Clone, Copy)]
pub enum InstructionSetMode {
    Plain,
    CPP(InstructionSetModeCPPSettings),
}

pub struct InstructionSetPrinter<'a> {
    pub instr: &'a InstructionListOutput<Rational>,
    pub mode: InstructionSetMode,
    pub name: String, // function name
}

impl std::fmt::Display for InstructionSetPrinter<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let has_only_one_return_value = self
            .instr
            .instr
            .iter()
            .filter(|x| matches!(x.1, Instruction::Yield(_)))
            .count()
            == 1;
        let use_return_value = has_only_one_return_value
            && match self.mode {
                InstructionSetMode::Plain => false,
                InstructionSetMode::CPP(s) => !s.always_pass_output_array,
            };

        if let InstructionSetMode::CPP(s) = self.mode {
            if s.write_header_and_test {
                f.write_str("#include <iostream>\n")?;
                f.write_str("\n")?;
            }

            f.write_str("template<typename T>\n")?;

            let mut seen_arrays = vec![];

            f.write_fmt(format_args!(
                "{} {}({}{}) {{\n",
                if use_return_value { "T" } else { "void" },
                self.name,
                self.instr
                    .input_map
                    .iter()
                    .filter_map(|x| if let super::PolyVariable::Function(x, _) = x {
                        if !seen_arrays.contains(x) {
                            seen_arrays.push(*x);

                            Some(format!("T* {}", super::PolyVariable::Symbol(*x)))
                        } else {
                            None
                        }
                    } else if let super::PolyVariable::Symbol(i) = x {
                        if [Symbol::E, Symbol::PI].contains(i) {
                            None
                        } else {
                            Some(format!("T {x}"))
                        }
                    } else {
                        Some(format!("T {x}"))
                    })
                    .collect::<Vec<_>>()
                    .join(","),
                if use_return_value { "" } else { ", T* out" }
            ))?;

            let max_register = self.instr.instr.iter().map(|r| r.0).max().unwrap_or(0);
            f.write_fmt(format_args!(
                "\tT {};\n",
                (0..=max_register)
                    .map(|x| format!("Z{x}"))
                    .collect::<Vec<_>>()
                    .join(","),
            ))?;
        }

        let mut out_counter = 0;

        for (reg, x) in &self.instr.instr {
            match x {
                Instruction::Add(a) => f.write_fmt(format_args!(
                    "\tZ{} = {};\n",
                    reg,
                    a.iter()
                        .map(|x| format!("Z{x}"))
                        .collect::<Vec<_>>()
                        .join("+")
                ))?,
                Instruction::Mul(m) => f.write_fmt(format_args!(
                    "\tZ{} = {};\n",
                    reg,
                    m.iter()
                        .map(|x| format!("Z{x}"))
                        .collect::<Vec<_>>()
                        .join("*")
                ))?,
                Instruction::Yield(y) => {
                    match self.mode {
                        InstructionSetMode::Plain => {
                            f.write_fmt(format_args!("\tOUT{out_counter} = Z{y};\n"))?
                        }
                        InstructionSetMode::CPP(_) => {
                            if use_return_value {
                                f.write_fmt(format_args!("\treturn Z{y};\n"))?
                            } else {
                                f.write_fmt(format_args!("\tout[{out_counter}] = Z{y};\n"))?
                            }
                        }
                    }
                    out_counter += 1;
                }
                Instruction::Empty => f.write_fmt(format_args!("\tZ{reg} = NOP;\n"))?,
                Instruction::Init(x) => f.write_fmt(format_args!(
                    "\tZ{} = {};\n",
                    reg,
                    x.to_pretty_string(&self.instr.input_map, self.mode)
                ))?,
            }
        }

        if let InstructionSetMode::CPP(s) = self.mode {
            f.write_str("}\n")?;

            if s.write_header_and_test {
                let points: Vec<_> = (0..self.instr.input_map.len())
                    .map(|i| ((i + 1) as f64 / (self.instr.input_map.len() + 2) as f64).to_string())
                    .collect();

                if use_return_value {
                    f.write_fmt(format_args!(
                        "\nint main() {{\n\tstd::cout << evaluate<double>({}) << std::endl;\n}}",
                        points.join(",")
                    ))?;
                } else {
                    f.write_fmt(format_args!(
                        "\nint main() {{\n\tdouble out[{}];\n\tevaluate({}, out);\n\tstd::cout << {} << std::endl;\n}}",
                        out_counter,
                        points.join(","),
                        (0..out_counter)
                            .map(|i| format!("out[{i}]"))
                            .collect::<Vec<_>>()
                            .join(" << \", \" << ")
                    ))?;
                }
            }
        }

        Ok(())
    }
}

/// A computational graph with efficient output evaluation for a nesting of variable identifications (`x_n = x_{n-1} + 2*x_{n-2}`, etc).
pub struct ExpressionEvaluator {
    operations: Vec<(
        super::PolyVariable,
        usize,
        InstructionListOutput<Rational>,
        Vec<super::PolyVariable>,
    )>,
    input: Vec<super::PolyVariable>,
}

impl ExpressionEvaluator {
    /// Create a computational graph with efficient output evaluation for a nesting of variable identifications (`x_n = x_{n-1} + 2*x_{n-2}`, etc).
    /// Every level provides a list of independent vectors whose expressions only depend on variables defined in previous levels.
    /// In these expressions, the references to previous vectors are represented using functions with the vector's name whose single argument is an index into the output array of the evaluation of that vector.
    /// For example:
    /// ```text
    /// x_0 = (p1, p2)
    /// x_1 = x_0[0] * x_0[1] + 2
    /// x_2 = x_1 + 2*x_0
    /// ```
    /// can be represented by:
    /// ```
    /// # use symbolica::{atom::Atom, parse, symbol};
    /// vec![
    ///       vec![(symbol!("x0"), vec![parse!("p1"), parse!("p2")])],
    ///       vec![(symbol!("x1"), vec![parse!("x0(0) * x0(1) + 2")])],
    ///       vec![(symbol!("x2"), vec![parse!("x1(0) * 2 * x0(1)")])]
    /// ];
    /// ```
    ///
    /// Each expression will be converted to a polynomial and optimized by writing it in a near-optimal Horner scheme and by performing
    /// common subexpression elimination. The number of optimization iterations can be set using `n_iter`.
    ///
    pub fn new(levels: Vec<Vec<(Symbol, Vec<Atom>)>>, n_iter: usize) -> ExpressionEvaluator {
        let mut overall_ops = vec![]; // the main function that calls all levels

        for l in levels {
            for (id, joint) in l {
                let mut polys: Vec<MultivariatePolynomial<_, u16>> =
                    joint.iter().map(|a| a.to_polynomial(&Q, None)).collect();

                // fuse the variable maps
                MultivariatePolynomial::unify_variables_list(&mut polys);

                let var_map = polys[0].variables.clone();

                let poly_ref = polys.iter().collect::<Vec<_>>();

                let (h, _score, _scheme) = HornerScheme::optimize_multiple(&poly_ref, n_iter);

                // TODO: support giving output names and multiple destinations?
                let mut i = HornerScheme::to_instr_multiple(&h, var_map.len());

                i.fuse_operations();

                for _ in 0..20_000 {
                    if !i.common_pair_elimination() {
                        break;
                    }
                    i.fuse_operations();
                }

                let o = i.to_output(var_map.as_ref().to_vec(), true);

                let mut seen_arrays = vec![];
                let call_args = var_map
                    .iter()
                    .filter_map(|x| {
                        if let super::PolyVariable::Function(x, _) = x {
                            if !seen_arrays.contains(x) {
                                seen_arrays.push(*x);

                                Some(super::PolyVariable::Symbol(*x))
                            } else {
                                None
                            }
                        } else if let super::PolyVariable::Symbol(i) = x {
                            if [Symbol::E, Symbol::PI].contains(i) {
                                None
                            } else {
                                Some(x.clone())
                            }
                        } else {
                            panic!("Expression contains non-array functions")
                        }
                    })
                    .collect::<Vec<_>>();

                overall_ops.push((super::PolyVariable::Symbol(id), h.len(), o, call_args));
            }
        }

        let internal: HashSet<_> = overall_ops.iter().map(|x| &x.0).collect();
        let mut external = HashSet::new();
        for (_, _, _, args) in &overall_ops {
            for arg in args {
                if !internal.contains(arg) {
                    external.insert(arg);
                }
            }
        }
        let mut input = external.into_iter().cloned().collect::<Vec<_>>();
        input.sort_by_cached_key(|f| f.to_string());

        ExpressionEvaluator {
            operations: overall_ops,
            input,
        }
    }

    /// Get the list of input variables that have to be provided in this order to the generated evaluation function.
    pub fn get_input(&self) -> &[super::PolyVariable] {
        &self.input
    }
}

impl std::fmt::Display for ExpressionEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(
            "#include <cmath>
#include <complex>
#include <iostream>

using namespace std::complex_literals;

auto 𝑖 = 1i;\n",
        )?;

        for (id, _, o, _) in self.operations.iter() {
            f.write_fmt(format_args!(
                "{}\n",
                InstructionSetPrinter {
                    instr: o,
                    name: id.to_string(),
                    mode: InstructionSetMode::CPP(InstructionSetModeCPPSettings {
                        write_header_and_test: false,
                        always_pass_output_array: true,
                    },),
                }
            ))?;
        }

        let last = self.operations.last().unwrap().0.clone();

        f.write_str("template<typename T>\n")?;
        f.write_fmt(format_args!(
            "void evaluate({}, T* {}_res) {{\n",
            self.input
                .iter()
                .map(|x| format!("T* {x}"))
                .collect::<Vec<_>>()
                .join(", "),
            last
        ))?;

        for (id, out_len, _, args) in &self.operations {
            let name = id.to_string();

            if *id != last {
                f.write_fmt(format_args!("\tT {name}_res[{out_len}];\n"))?;
            }

            let mut f_args: Vec<_> = args
                .iter()
                .map(|x| {
                    if self.operations.iter().any(|(name, _, _, _)| x == name) {
                        x.to_string() + "_res"
                    } else {
                        x.to_string()
                    }
                })
                .collect();
            f_args.push(format!("{name}_res"));

            f.write_fmt(format_args!("\t{}({});\n", name, f_args.join(",")))?;
        }

        f.write_str("}")
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use crate::{
        atom::AtomCore,
        domains::{float::Complex, rational::Q},
        parse,
        poly::{
            evaluate::{BorrowedHornerScheme, InstructionSetPrinter},
            polynomial::MultivariatePolynomial,
        },
        symbol,
    };

    use wide::f64x4;

    const RES_53: &str = "-a5^3*b0^5+a4*a5^2*b0^4*b1-a4^2*a5*b0^4*b2+a4^3*b0^4*b3-a3*a5^2*
b0^3*b1^2+2*a3*a5^2*b0^4*b2+a3*a4*a5*b0^3*b1*b2-3*a3*a4*a5*b0^4*
b3-a3*a4^2*b0^3*b1*b3-a3^2*a5*b0^3*b2^2+2*a3^2*a5*b0^3*b1*b3+a3^2
*a4*b0^3*b2*b3-a3^3*b0^3*b3^2+a2*a5^2*b0^2*b1^3-3*a2*a5^2*b0^3*b1
*b2+3*a2*a5^2*b0^4*b3-a2*a4*a5*b0^2*b1^2*b2+2*a2*a4*a5*b0^3*b2^2+
a2*a4*a5*b0^3*b1*b3+a2*a4^2*b0^2*b1^2*b3-2*a2*a4^2*b0^3*b2*b3+a2*
a3*a5*b0^2*b1*b2^2-2*a2*a3*a5*b0^2*b1^2*b3-a2*a3*a5*b0^3*b2*b3-a2
*a3*a4*b0^2*b1*b2*b3+3*a2*a3*a4*b0^3*b3^2+a2*a3^2*b0^2*b1*b3^2-
a2^2*a5*b0^2*b2^3+3*a2^2*a5*b0^2*b1*b2*b3-3*a2^2*a5*b0^3*b3^2+
a2^2*a4*b0^2*b2^2*b3-2*a2^2*a4*b0^2*b1*b3^2-a2^2*a3*b0^2*b2*b3^2+
a2^3*b0^2*b3^3-a1*a5^2*b0*b1^4+4*a1*a5^2*b0^2*b1^2*b2-2*a1*a5^2*
b0^3*b2^2-4*a1*a5^2*b0^3*b1*b3+a1*a4*a5*b0*b1^3*b2-3*a1*a4*a5*
b0^2*b1*b2^2-a1*a4*a5*b0^2*b1^2*b3+5*a1*a4*a5*b0^3*b2*b3-a1*a4^2*
b0*b1^3*b3+3*a1*a4^2*b0^2*b1*b2*b3-3*a1*a4^2*b0^3*b3^2-a1*a3*a5*
b0*b1^2*b2^2+2*a1*a3*a5*b0*b1^3*b3+2*a1*a3*a5*b0^2*b2^3-4*a1*a3*
a5*b0^2*b1*b2*b3+3*a1*a3*a5*b0^3*b3^2+a1*a3*a4*b0*b1^2*b2*b3-2*a1
*a3*a4*b0^2*b2^2*b3-a1*a3*a4*b0^2*b1*b3^2-a1*a3^2*b0*b1^2*b3^2+2*
a1*a3^2*b0^2*b2*b3^2+a1*a2*a5*b0*b1*b2^3-3*a1*a2*a5*b0*b1^2*b2*b3
-a1*a2*a5*b0^2*b2^2*b3+5*a1*a2*a5*b0^2*b1*b3^2-a1*a2*a4*b0*b1*
b2^2*b3+2*a1*a2*a4*b0*b1^2*b3^2+a1*a2*a4*b0^2*b2*b3^2+a1*a2*a3*b0
*b1*b2*b3^2-3*a1*a2*a3*b0^2*b3^3-a1*a2^2*b0*b1*b3^3-a1^2*a5*b0*
b2^4+4*a1^2*a5*b0*b1*b2^2*b3-2*a1^2*a5*b0*b1^2*b3^2-4*a1^2*a5*
b0^2*b2*b3^2+a1^2*a4*b0*b2^3*b3-3*a1^2*a4*b0*b1*b2*b3^2+3*a1^2*a4
*b0^2*b3^3-a1^2*a3*b0*b2^2*b3^2+2*a1^2*a3*b0*b1*b3^3+a1^2*a2*b0*
b2*b3^3-a1^3*b0*b3^4+a0*a5^2*b1^5-5*a0*a5^2*b0*b1^3*b2+5*a0*a5^2*
b0^2*b1*b2^2+5*a0*a5^2*b0^2*b1^2*b3-5*a0*a5^2*b0^3*b2*b3-a0*a4*a5
*b1^4*b2+4*a0*a4*a5*b0*b1^2*b2^2+a0*a4*a5*b0*b1^3*b3-2*a0*a4*a5*
b0^2*b2^3-7*a0*a4*a5*b0^2*b1*b2*b3+3*a0*a4*a5*b0^3*b3^2+a0*a4^2*
b1^4*b3-4*a0*a4^2*b0*b1^2*b2*b3+2*a0*a4^2*b0^2*b2^2*b3+4*a0*a4^2*
b0^2*b1*b3^2+a0*a3*a5*b1^3*b2^2-2*a0*a3*a5*b1^4*b3-3*a0*a3*a5*b0*
b1*b2^3+6*a0*a3*a5*b0*b1^2*b2*b3+3*a0*a3*a5*b0^2*b2^2*b3-7*a0*a3*
a5*b0^2*b1*b3^2-a0*a3*a4*b1^3*b2*b3+3*a0*a3*a4*b0*b1*b2^2*b3+a0*
a3*a4*b0*b1^2*b3^2-5*a0*a3*a4*b0^2*b2*b3^2+a0*a3^2*b1^3*b3^2-3*a0
*a3^2*b0*b1*b2*b3^2+3*a0*a3^2*b0^2*b3^3-a0*a2*a5*b1^2*b2^3+3*a0*
a2*a5*b1^3*b2*b3+2*a0*a2*a5*b0*b2^4-6*a0*a2*a5*b0*b1*b2^2*b3-3*a0
*a2*a5*b0*b1^2*b3^2+7*a0*a2*a5*b0^2*b2*b3^2+a0*a2*a4*b1^2*b2^2*b3
-2*a0*a2*a4*b1^3*b3^2-2*a0*a2*a4*b0*b2^3*b3+4*a0*a2*a4*b0*b1*b2*
b3^2-3*a0*a2*a4*b0^2*b3^3-a0*a2*a3*b1^2*b2*b3^2+2*a0*a2*a3*b0*
b2^2*b3^2+a0*a2*a3*b0*b1*b3^3+a0*a2^2*b1^2*b3^3-2*a0*a2^2*b0*b2*
b3^3+a0*a1*a5*b1*b2^4-4*a0*a1*a5*b1^2*b2^2*b3+2*a0*a1*a5*b1^3*
b3^2-a0*a1*a5*b0*b2^3*b3+7*a0*a1*a5*b0*b1*b2*b3^2-3*a0*a1*a5*b0^2
*b3^3-a0*a1*a4*b1*b2^3*b3+3*a0*a1*a4*b1^2*b2*b3^2+a0*a1*a4*b0*
b2^2*b3^2-5*a0*a1*a4*b0*b1*b3^3+a0*a1*a3*b1*b2^2*b3^2-2*a0*a1*a3*
b1^2*b3^3-a0*a1*a3*b0*b2*b3^3-a0*a1*a2*b1*b2*b3^3+3*a0*a1*a2*b0*
b3^4+a0*a1^2*b1*b3^4-a0^2*a5*b2^5+5*a0^2*a5*b1*b2^3*b3-5*a0^2*a5*
b1^2*b2*b3^2-5*a0^2*a5*b0*b2^2*b3^2+5*a0^2*a5*b0*b1*b3^3+a0^2*a4*
b2^4*b3-4*a0^2*a4*b1*b2^2*b3^2+2*a0^2*a4*b1^2*b3^3+4*a0^2*a4*b0*
b2*b3^3-a0^2*a3*b2^3*b3^2+3*a0^2*a3*b1*b2*b3^3-3*a0^2*a3*b0*b3^4+
a0^2*a2*b2^2*b3^3-2*a0^2*a2*b1*b3^4-a0^2*a1*b2*b3^4+a0^3*b3^5";

    #[test]
    fn res_53() {
        let vars = ["a0", "a1", "a2", "a3", "a4", "a5", "b0", "b1", "b2", "b3"];
        let vars = vars.iter().map(|x| symbol!(x).into()).collect::<Vec<_>>();

        let poly: MultivariatePolynomial<_, u8> =
            parse!(RES_53).to_polynomial(&Q, Some(Arc::new(vars)));

        let (h, _ops, scheme) = poly.optimize_horner_scheme(1000);
        let mut i = h.to_instr(poly.nvars());

        println!(
            "Number of operations={}, with scheme={:?}",
            BorrowedHornerScheme::from(&h).op_count_cse(),
            scheme,
        );

        i.fuse_operations();

        for _ in 0..100_000 {
            if !i.common_pair_elimination() {
                break;
            }
            i.fuse_operations();
        }

        let o = i.to_output(poly.variables.as_ref().to_vec(), true);
        let o_f64 = o.convert::<f64>();

        let _ = format!(
            "{}",
            InstructionSetPrinter {
                name: "sigma".to_string(),
                instr: &o,
                mode: crate::poly::evaluate::InstructionSetMode::CPP(
                    crate::poly::evaluate::InstructionSetModeCPPSettings {
                        write_header_and_test: true,
                        always_pass_output_array: false,
                    }
                )
            }
        );

        let mut evaluator = o_f64.evaluator();

        let res = evaluator
            .evaluate_with_input(&(0..poly.nvars()).map(|x| x as f64 + 1.).collect::<Vec<_>>())[0];

        assert_eq!(res, -1167748.);

        // evaluate with simd
        let o_f64x4 = o.convert::<f64x4>();
        let mut evaluator = o_f64x4.evaluator();

        let res = evaluator.evaluate_with_input(
            &(0..poly.nvars())
                .map(|x| f64x4::new([x as f64 + 1., x as f64 + 2., x as f64 + 3., x as f64 + 4.]))
                .collect::<Vec<_>>(),
        )[0];

        assert_eq!(
            res,
            f64x4::new([-1167748., -3589814., -8821476., -18822982.])
        );

        // evaluate with complex numbers
        let mut complex_evaluator = o.convert::<Complex<f64>>().evaluator();
        let res = complex_evaluator.evaluate_with_input(
            &(0..poly.nvars())
                .map(|x| Complex::new(x as f64 + 0.1, x as f64 + 2.))
                .collect::<Vec<_>>(),
        )[0];
        assert!(
            (res.re - -9246939.447600078).abs() < 1e-6
                && (res.im - -21536192.22926005).abs() < 1e-6
        );
    }
}
