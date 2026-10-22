use std::{cmp::Ordering, collections::BinaryHeap, ops::DerefMut};

use smallvec::SmallVec;

use crate::{
    atom::{
        AddView, Atom, AtomCore, AtomView, Fun, InlineVar, MulView, PowView, Symbol,
        representation::{
            FunView, InlineNum, ListIterator, ListSlice, compare_expanded_function_data,
            compare_expanded_sum_contents, compare_expanded_term_keys,
        },
    },
    coefficient::{Coefficient, CoefficientView},
    domains::{
        float::{Complex, Float, Real},
        integer::Z,
        rational::{Q, Rational},
    },
    poly::PolyVariable,
    state::{RecycledAtom, State, Workspace},
    utils::minimum_rotation_by,
    warn,
};

impl<'a> AtomView<'a> {
    /// Get a slice that can be used to compare terms.
    /// The coefficients are not included in the slice.
    #[inline(always)]
    pub(crate) fn get_term_cmp_slice(&self) -> &'a [u8] {
        if let AtomView::Mul(m1) = self {
            let mut it1 = m1.to_slice();

            if m1.has_coefficient() {
                it1 = it1.fast_forward(1);
            }

            it1.get_data()
        } else if let AtomView::Num(_) = self {
            &[]
        } else {
            self.get_data()
        }
    }
}

impl AtomView<'_> {
    /// Compare two atoms semantically in an ordering that sorts by type.
    ///
    /// This is not the ordering used for sorting terms or factors.
    #[inline]
    pub fn cmp(&self, other: &AtomView<'_>) -> Ordering {
        if self.get_data() == other.get_data() {
            // Compare representation bytes before structural comparison
            return Ordering::Equal;
        }

        self.cmp_structural(other)
    }

    /// Compare structure after the caller has checked representation equality.
    pub(crate) fn cmp_structural(&self, other: &AtomView<'_>) -> Ordering {
        // Borrow ordinary views directly. Materializing both resolved views
        // here copies their data and ownership pointers on every comparison.
        let resolved_a;
        let a = if matches!(self, AtomView::Alias(_)) {
            resolved_a = resolve_alias_loop(*self);
            &resolved_a
        } else {
            self
        };
        let resolved_b;
        let b = if matches!(other, AtomView::Alias(_)) {
            resolved_b = resolve_alias_loop(*other);
            &resolved_b
        } else {
            other
        };

        match (a, b) {
            // comparisons on floating points may yield Equal even if the binary data is not the same
            // for example when the precision is different. To stay consistent with the PartialEq implementation,
            // tie break on binary data comparison
            (AtomView::Num(n1), AtomView::Num(n2)) => n1
                .get_coeff_view()
                .cmp(&n2.get_coeff_view())
                .then_with(|| a.get_data().cmp(b.get_data())),
            (AtomView::Num(_), _) => Ordering::Less,
            (_, AtomView::Num(_)) => Ordering::Greater,
            (AtomView::Var(v1), AtomView::Var(v2)) => v1.get_symbol_id().cmp(&v2.get_symbol_id()),
            (AtomView::Var(_), _) => Ordering::Less,
            (_, AtomView::Var(_)) => Ordering::Greater,
            (AtomView::Alias(_), AtomView::Alias(_)) => a.get_data().cmp(b.get_data()),
            (AtomView::Alias(_), _) => Ordering::Less,
            (_, AtomView::Alias(_)) => Ordering::Greater,
            (AtomView::Pow(p1), AtomView::Pow(p2)) => {
                let (b1, e1) = p1.get_base_exp();
                let (b2, e2) = p2.get_base_exp();
                b1.cmp(&b2).then_with(|| e1.cmp(&e2))
            }
            (_, AtomView::Pow(_)) => Ordering::Greater,
            (AtomView::Pow(_), _) => Ordering::Less,
            (AtomView::Mul(m1), AtomView::Mul(m2)) => {
                let it1 = m1.to_slice();
                let it2 = m2.to_slice();

                let len_cmp = it1.len().cmp(&it2.len());
                if len_cmp != Ordering::Equal {
                    return len_cmp;
                }

                for (t1, t2) in it1.iter().zip(it2.iter()) {
                    let argcmp = t1.cmp(&t2);
                    if argcmp != Ordering::Equal {
                        return argcmp;
                    }
                }

                // unreachable!(
                //     "Expression is equal but not binary equal: {};\n{:?}\n{:?}",
                //     self, m1, m2
                // );
                Ordering::Equal
            }
            (AtomView::Mul(_), _) => Ordering::Less,
            (_, AtomView::Mul(_)) => Ordering::Greater,
            (AtomView::Add(a1), AtomView::Add(a2)) => {
                if a1.has_alias() || a2.has_alias() {
                    return compare_expanded_sum_contents(
                        a1.as_view().get_data(),
                        a2.as_view().get_data(),
                    );
                }
                let it1 = a1.to_slice();
                let it2 = a2.to_slice();

                let len_cmp = it1.len().cmp(&it2.len());
                if len_cmp != Ordering::Equal {
                    return len_cmp;
                }

                for (t1, t2) in it1.iter().zip(it2.iter()) {
                    let argcmp = t1.cmp(&t2);
                    if argcmp != Ordering::Equal {
                        return argcmp;
                    }
                }

                // unreachable!(
                //     "Expression is equal but not binary equal: {};\n{:?}\n{:?}",
                //     self, a1, a2
                // );
                Ordering::Equal
            }
            (AtomView::Add(_), _) => Ordering::Less,
            (_, AtomView::Add(_)) => Ordering::Greater,

            (AtomView::Fun(f1), AtomView::Fun(f2)) => {
                let name_comp = f1.get_symbol_id().cmp(&f2.get_symbol_id());
                if name_comp != Ordering::Equal {
                    return name_comp;
                }

                if !cfg!(feature = "full_fn_cmp") {
                    return if f1.has_alias() || f2.has_alias() {
                        compare_expanded_function_data(
                            f1.as_view().get_data(),
                            f2.as_view().get_data(),
                        )
                    } else {
                        f1.fast_cmp(*f2)
                    };
                }

                let len_cmp = f1.get_nargs().cmp(&f2.get_nargs());
                if len_cmp != Ordering::Equal {
                    return len_cmp;
                }

                for (arg1, arg2) in f1.iter().zip(f2.iter()) {
                    let argcmp = arg1.cmp(&arg2);
                    if argcmp != Ordering::Equal {
                        return argcmp;
                    }
                }

                // unreachable!(
                //     "Expression is equal but not binary equal: {};\n{:?}\n{:?}",
                //     self, f1, f2
                // );
                Ordering::Equal
            }
        }
    }

    /// Compare factors semantically in a term. `x` and `x^2` are placed next to each other by sorting a power based on the base only.
    pub(crate) fn cmp_factors(&self, other: &AtomView<'_>) -> Ordering {
        let resolved_a;
        let a = if matches!(self, AtomView::Alias(_)) {
            resolved_a = resolve_alias_loop(*self);
            &resolved_a
        } else {
            self
        };
        let resolved_b;
        let b = if matches!(other, AtomView::Alias(_)) {
            resolved_b = resolve_alias_loop(*other);
            &resolved_b
        } else {
            other
        };

        match (a, b) {
            (AtomView::Num(_), AtomView::Num(_)) => Ordering::Equal,
            (AtomView::Num(_), _) => Ordering::Less,
            (_, AtomView::Num(_)) => Ordering::Greater,
            (AtomView::Var(v1), AtomView::Var(v2)) => v1.get_symbol_id().cmp(&v2.get_symbol_id()),
            (AtomView::Alias(_), AtomView::Alias(_)) => a.get_data().cmp(b.get_data()),
            (AtomView::Alias(_), _) => Ordering::Less,
            (_, AtomView::Alias(_)) => Ordering::Greater,
            (AtomView::Pow(p1), AtomView::Pow(p2)) => {
                // TODO: inline partial_cmp call by creating an inlined version
                p1.get_base().cmp(&p2.get_base())
            }
            (_, AtomView::Pow(p2)) => {
                let base = p2.get_base();
                a.cmp(&base).then(Ordering::Less) // sort x^2*x -> x*x^2
            }
            (AtomView::Pow(p1), _) => {
                let base = p1.get_base();
                base.cmp(other).then(Ordering::Greater)
            }
            (AtomView::Var(_), _) => Ordering::Less,
            (_, AtomView::Var(_)) => Ordering::Greater,

            (AtomView::Mul(_), _) | (_, AtomView::Mul(_)) => {
                unreachable!("Cannot have a submul in a factor");
            }
            (AtomView::Add(a1), AtomView::Add(a2)) => {
                if a1.has_alias() || a2.has_alias() {
                    return compare_expanded_sum_contents(
                        a1.as_view().get_data(),
                        a2.as_view().get_data(),
                    );
                }
                let it1 = a1.to_slice();
                let it2 = a2.to_slice();

                let len_cmp = it1.len().cmp(&it2.len());
                if len_cmp != Ordering::Equal {
                    return len_cmp;
                }

                for (t1, t2) in it1.iter().zip(it2.iter()) {
                    let argcmp = t1.cmp(&t2);
                    if argcmp != Ordering::Equal {
                        return argcmp;
                    }
                }

                Ordering::Equal
            }
            (AtomView::Add(_), _) => Ordering::Less,
            (_, AtomView::Add(_)) => Ordering::Greater,

            (AtomView::Fun(f1), AtomView::Fun(f2)) => {
                let name_comp = f1.get_symbol_id().cmp(&f2.get_symbol_id());
                if name_comp != Ordering::Equal {
                    return name_comp;
                }
                if !cfg!(feature = "full_fn_cmp") {
                    return if f1.has_alias() || f2.has_alias() {
                        compare_expanded_function_data(
                            f1.as_view().get_data(),
                            f2.as_view().get_data(),
                        )
                    } else {
                        f1.fast_cmp(*f2)
                    };
                }

                if f1.as_view().get_data() == f2.as_view().get_data() {
                    return Ordering::Equal;
                }

                let len_cmp = f1.get_nargs().cmp(&f2.get_nargs());
                if len_cmp != Ordering::Equal {
                    return len_cmp;
                }

                for (arg1, arg2) in f1.iter().zip(f2.iter()) {
                    let argcmp = arg1.cmp(&arg2);
                    if argcmp != Ordering::Equal {
                        return argcmp;
                    }
                }

                Ordering::Equal
            }
        }
    }

    /// Compare terms semantically in an expression. `x` and `2*x` are placed next to each other.
    #[inline]
    pub(crate) fn cmp_terms(&self, other: &AtomView<'_>, no_aliases: bool) -> Ordering {
        debug_assert!(!matches!(self, AtomView::Add(_)));
        debug_assert!(!matches!(other, AtomView::Add(_)));

        if no_aliases {
            return self.get_term_cmp_slice().cmp(other.get_term_cmp_slice());
        }

        let a = resolve_alias(*self);
        let b = resolve_alias(*other);

        // Match ordinary sums' encoded monomial order after expansion. Root
        // aliases were borrowed above, so alias-free bodies need no allocation.
        compare_expanded_term_keys(a.get_data(), b.get_data())
    }

    /// Simplify logs in the argument of the exponential function.
    fn simplify_exp_log(&self, ws: &Workspace, out: &mut Atom) -> bool {
        if let AtomView::Fun(f) = self
            && f.get_symbol_id() == Symbol::LOG_ID
            && f.get_nargs() == 1
        {
            out.set_from_view(&f.iter().next().unwrap());
            return true;
        }

        if let AtomView::Mul(m) = self {
            let mut found_index = None;
            for (i, a) in m.iter().enumerate() {
                if a.simplify_exp_log(ws, out) {
                    if found_index.is_some() {
                        return false;
                    }

                    found_index = Some(i);
                }
            }

            if let Some(i) = found_index {
                let mut mm = ws.new_atom();
                let mmm = mm.to_mul();
                for (j, a) in m.iter().enumerate() {
                    if j != i {
                        mmm.extend(a);
                    }
                }

                let mut b = ws.new_atom();
                b.to_pow(out.as_view(), mm.as_view());

                b.as_view().normalize(ws, out);
                return true;
            } else {
                return false;
            }
        }

        if let AtomView::Add(a) = self {
            // all terms must simplify
            let mut mm = ws.new_atom();
            let m = mm.to_mul();

            let mut aa = ws.new_atom();
            let aaa = aa.to_add();

            let mut changed = false;
            for a in a {
                if a.simplify_exp_log(ws, out) {
                    changed = true;
                    m.extend(out.as_view());
                } else {
                    aaa.extend(a);
                }
            }

            if changed {
                let mut new_exp = ws.new_atom();
                // TODO: change to e^()
                new_exp.to_fun(Symbol::EXP).add_arg(aa.as_view());

                m.extend(new_exp.as_view());

                mm.as_view().normalize(ws, out);
            }

            return changed;
        }

        false
    }
}

#[inline(always)]
fn resolve_alias(a: AtomView<'_>) -> AtomView<'_> {
    if let AtomView::Alias(_) = a {
        resolve_alias_loop(a)
    } else {
        a
    }
}

#[cold]
fn resolve_alias_loop(mut a: AtomView<'_>) -> AtomView<'_> {
    while let AtomView::Alias(alias) = a
        && !alias.is_opaque()
    {
        a = alias.get_body();
    }
    a
}

#[inline]
fn semantic_eq(a: &AtomView<'_>, b: &AtomView<'_>) -> bool {
    // Normalized alias-free expressions have a canonical byte representation.
    // Alias bodies can also contain unnormalized atoms from the raw builders.
    a.get_data() == b.get_data()
        || ((a.has_alias() || b.has_alias() || a.needs_normalization() || b.needs_normalization())
            && a.cmp(b).is_eq())
}

#[inline]
fn semantic_eq_slices(a: ListSlice<'_>, b: ListSlice<'_>, needs_semantic_cmp: bool) -> bool {
    a.len() == b.len()
        && (a.get_data() == b.get_data()
            || (needs_semantic_cmp && a.iter().zip(b.iter()).all(|(a, b)| semantic_eq(&a, &b))))
}

#[inline]
fn non_coefficient_factors<'a>(m: crate::atom::representation::MulView<'a>) -> ListSlice<'a> {
    let slice = m.to_slice();
    if m.has_coefficient() {
        slice.fast_forward(1)
    } else {
        slice
    }
}

impl Atom {
    /// Merge two factors if possible. If this function returns `true`, `self`
    /// will have been updated by the merge from `other` and `other` should be discarded.
    /// If the function return `false`, no merge was possible and no modifications were made.
    fn merge_factors(
        &mut self,
        other: &mut Self,
        helper: &mut Self,
        workspace: &Workspace,
    ) -> bool {
        if let Atom::Pow(p1) = self {
            // x^a * x^b = x^(a + b)
            if let Atom::Pow(p2) = other {
                let (base2, exp2) = p2.to_pow_view().get_base_exp();

                let (base1, exp1) = p1.to_pow_view().get_base_exp();

                if !semantic_eq(&base1, &base2) {
                    return false;
                }

                if let AtomView::Num(n) = &exp1
                    && let AtomView::Num(n2) = &exp2
                {
                    let new_exp = helper.to_num(n.get_coeff_view() + n2.get_coeff_view());

                    if new_exp.to_num_view().is_zero() {
                        self.to_num(1);
                    } else if new_exp.to_num_view().is_one() {
                        self.set_from_view(&base2);
                    } else {
                        p1.set_from_base_and_exp(base2, helper.as_view());
                    }

                    self.set_normalized(true);
                    return true;
                }

                let new_exp = helper.to_add();
                new_exp.extend(exp1);
                new_exp.extend(exp2);
                let mut helper2 = workspace.new_atom();
                helper.as_view().normalize(workspace, &mut helper2);

                if let AtomView::Num(n) = helper2.as_view() {
                    if n.is_zero() {
                        self.to_num(1);
                        return true;
                    } else if n.is_one() {
                        self.set_from_view(&base2);
                        return true;
                    }
                }

                p1.set_from_base_and_exp(base2, helper2.as_view());
                p1.set_normalized(true);
                return true;
            }

            // x^n * x = x^(n+1), unless x is a number
            let pv = p1.to_pow_view();
            let (base, exp) = pv.get_base_exp();

            if !matches!(other, Atom::Num(_)) && semantic_eq(&base, &other.as_view()) {
                if let AtomView::Num(n) = &exp {
                    let new_exp = n.get_coeff_view() + 1;

                    if new_exp.is_zero() {
                        self.to_num(1);
                    } else if new_exp == 1.into() {
                        self.set_from_view(&other.as_view());
                    } else {
                        let num = helper.to_num(new_exp);
                        self.to_pow(other.as_view(), AtomView::Num(num.to_num_view()));
                    }
                } else {
                    other.to_num(1);

                    let new_exp = helper.to_add();
                    new_exp.extend(other.as_view());
                    new_exp.extend(exp);
                    let mut helper2 = workspace.new_atom();
                    helper.as_view().normalize(workspace, &mut helper2);
                    other.to_pow(base, helper2.as_view());
                    std::mem::swap(self, other);
                }

                self.set_normalized(true);
                return true;
            }

            return false;
        }

        // x * x^n = x^(n+1), unless x is a number
        if let Atom::Pow(p) = other {
            let pv = p.to_pow_view();
            let (base, exp) = pv.get_base_exp();

            if !matches!(self, Atom::Num(_)) && semantic_eq(&base, &self.as_view()) {
                // The base borrowed from `other` survives mutations of `self`.
                // Copy only when preserving an alias held by `self` instead.
                let mut semantic_base;
                let base = if self.has_alias() {
                    semantic_base = workspace.new_atom();
                    semantic_base.set_from_view(&self.as_view());
                    semantic_base.as_view()
                } else {
                    base
                };

                if let AtomView::Num(n) = &exp {
                    let new_exp = n.get_coeff_view() + 1;

                    if new_exp.is_zero() {
                        self.to_num(1);
                    } else if new_exp == 1.into() {
                        self.set_from_view(&base);
                    } else {
                        let num = helper.to_num(new_exp);
                        self.to_pow(base, AtomView::Num(num.to_num_view()));
                    }
                } else {
                    self.to_num(1);

                    let new_exp = helper.to_add();
                    new_exp.extend(self.as_view());
                    new_exp.extend(exp);
                    let mut helper2 = workspace.new_atom();
                    helper.as_view().normalize(workspace, &mut helper2);
                    self.to_pow(base, helper2.as_view());
                }

                self.set_normalized(true);
                return true;
            } else {
                return false;
            }
        }

        // simplify num1 * num2
        if let Atom::Num(n1) = self {
            if let Atom::Num(n2) = other {
                n1.mul(&n2.to_num_view());
                return true;
            } else {
                return false;
            }
        }

        // x * x => x^2
        if semantic_eq(&self.as_view(), &other.as_view()) {
            // add powers
            let exp = other.to_num(2);
            helper.to_pow(self.as_view(), AtomView::Num(exp.to_num_view()));
            helper.set_normalized(true);

            // overwrite self with the new power view
            std::mem::swap(self, helper);

            return true;
        }

        false
    }

    /// Merge two terms if possible. If this function returns `true`, `self`
    /// will have been updated by the merge from `other` and `other` should be discarded.
    /// If the function return `false`, no merge was possible and no modifications were made.
    pub(crate) fn merge_terms(&mut self, other: AtomView, helper: &mut Self) -> bool {
        let resolved_other;
        let b = if matches!(other, AtomView::Alias(_)) {
            resolved_other = resolve_alias(other);
            &resolved_other
        } else {
            &other
        };

        // Callers normalize ordinary terms, but transparent aliases can still
        // contain raw bodies. Normalize those before copying their children
        // into a parent that will be marked normalized.
        if matches!(other, AtomView::Alias(_)) && b.needs_normalization() {
            return self.merge_terms_with_normalized_alias_bodies(other, helper);
        }

        if let Atom::Num(n1) = self {
            if let AtomView::Num(n2) = b {
                n1.add(n2);
                return true;
            } else {
                return false;
            }
        }

        // Ordinary products can expose their typed view directly. Only an
        // actual alias needs a resolved AtomView, and numbers above need none.
        let resolved_self;
        let self_mul = match self {
            Atom::Mul(m) => Some(m.to_mul_view()),
            Atom::Alias(a) => {
                resolved_self = resolve_alias(a.as_view());
                if resolved_self.needs_normalization() {
                    return self.merge_terms_with_normalized_alias_bodies(other, helper);
                }
                if let AtomView::Mul(m) = resolved_self {
                    Some(m)
                } else {
                    None
                }
            }
            _ => None,
        };

        // compare the non-coefficient part of terms and add the coefficients if they are the same
        if let Some(m) = self_mul {
            let mut slice = m.to_slice();

            let has_coeff = m.has_coefficient();
            let mut coeff = *b; // stub, should not be of type num
            if has_coeff {
                (coeff, slice) = slice.pop_first();
            }

            if let AtomView::Mul(m2) = b {
                let mut slice2 = m2.to_slice();

                let mut coeff2 = *b; // stub, should not be of type num
                if m2.has_coefficient() {
                    (coeff2, slice2) = slice2.pop_first();
                }

                if semantic_eq_slices(
                    slice,
                    slice2,
                    m.has_alias() || m2.has_alias() || !m.is_normalized() || !m2.is_normalized(),
                ) {
                    // TODO: not correct for finite fields!
                    let num = if let AtomView::Num(n) = coeff {
                        n.get_coeff_view()
                    } else {
                        CoefficientView::Natural(1, 1, 0, 1)
                    };

                    let new_coeff = if let AtomView::Num(n) = coeff2 {
                        num + n.get_coeff_view()
                    } else {
                        num + 1
                    };

                    let len = slice.len();

                    // strip coefficient if it is 1 (can be float)
                    if new_coeff.is_one() {
                        if len == 1 {
                            // downgrade
                            self.set_from_view(&slice2.get(0));
                        } else {
                            // remove coefficient
                            let m = self.to_mul();
                            for a in slice2.iter() {
                                m.extend(a); // TODO: modifying self instead
                            }
                            m.set_has_coefficient(false);
                            m.set_normalized(true);
                        }

                        return true;
                    }

                    if new_coeff.is_zero() {
                        self.to_num(new_coeff);
                        return true;
                    }

                    let on = helper.to_num(new_coeff);

                    if has_coeff && let Atom::Mul(m) = self {
                        m.replace_first(on.to_num_view().as_view());
                        m.set_has_coefficient(true);
                        m.set_normalized(true);
                    } else {
                        let m = self.to_mul();
                        m.extend(on.to_num_view().as_view());

                        for a in slice2.iter() {
                            m.extend(a);
                        }

                        m.set_has_coefficient(true);
                        m.set_normalized(true);
                    }

                    return true;
                }
            } else {
                if slice.len() != 1 || !semantic_eq(b, &slice.get(0)) {
                    return false;
                }

                let new_coeff = if let AtomView::Num(n) = coeff {
                    n.get_coeff_view() + 1
                } else {
                    return false;
                };

                assert!(new_coeff != 1.into());
                if new_coeff.is_zero() {
                    self.to_num(new_coeff);
                    return true;
                }

                let on = helper.to_num(new_coeff);

                if let Atom::Mul(m) = self {
                    m.replace_first(on.to_num_view().as_view());
                    m.set_normalized(true);
                } else {
                    let m = self.to_mul();
                    m.extend(on.to_num_view().as_view());
                    m.extend(*b);
                    m.set_has_coefficient(true);
                    m.set_normalized(true);
                }

                return true;
            }
        } else if let AtomView::Mul(m) = b {
            if !m.has_coefficient() {
                return false;
            }

            let slice = non_coefficient_factors(*m);

            if slice.len() != 1 {
                return false; // no match
            }

            let coeff = m.to_slice().get(0);
            let term = slice.get(0);

            if semantic_eq(&self.as_view(), &term) {
                let (new_coeff, has_num) = if let AtomView::Num(n) = &coeff {
                    (n.get_coeff_view() + 1, true)
                } else {
                    return false; // last elem is not a coefficient
                };

                assert!(new_coeff != 1.into());
                if new_coeff.is_zero() {
                    self.to_num(new_coeff);
                    return true;
                }

                let on = helper.to_num(new_coeff);

                b.clone_into(self);

                if let Atom::Mul(m) = self {
                    if has_num {
                        m.replace_first(on.to_num_view().as_view());
                    } else {
                        let m = self.to_mul();
                        m.extend(on.to_num_view().as_view());
                        m.extend(term);

                        m.set_has_coefficient(true);
                        m.set_normalized(true);
                        return true;
                    }

                    m.set_has_coefficient(true);
                }

                self.set_normalized(true);
                return true;
            }
        } else if semantic_eq(&self.as_view(), b) {
            let mul = helper.to_mul();
            self.to_num((2, 1));
            mul.extend(self.as_view());
            mul.extend(*b);
            mul.set_has_coefficient(true);
            mul.set_normalized(true);

            std::mem::swap(self, helper);
            return true;
        };

        false
    }

    #[cold]
    fn merge_terms_with_normalized_alias_bodies(
        &mut self,
        other: AtomView,
        helper: &mut Self,
    ) -> bool {
        Workspace::get_local().with(|ws| {
            let mut left = ws.new_atom();
            let mut right = ws.new_atom();
            resolve_alias(self.as_view()).normalize(ws, &mut left);
            resolve_alias(other).normalize(ws, &mut right);
            if left.merge_terms(right.as_view(), helper) {
                self.set_from_view(&left.as_view());
                true
            } else {
                false
            }
        })
    }
}

impl AtomView<'_> {
    #[inline(always)]
    /// Return whether the expression is marked as requiring normalization.
    /// This reads the stored flag rather than checking algebraic equivalence.
    pub fn needs_normalization(&self) -> bool {
        match self {
            AtomView::Num(_) | AtomView::Var(_) | AtomView::Alias(_) => false,
            AtomView::Fun(f) => !f.is_normalized(),
            AtomView::Pow(p) => !p.is_normalized(),
            AtomView::Mul(m) => !m.is_normalized(),
            AtomView::Add(a) => !a.is_normalized(),
        }
    }

    /// Normalize an atom.
    pub fn normalize(&self, workspace: &Workspace, out: &mut Atom) {
        if !self.needs_normalization() {
            out.set_from_view(self);
            return;
        }

        let mut cleanup_alias_handles =
            self.has_alias() && !matches!(self, AtomView::Add(_) | AtomView::Mul(_));
        // Keep each kind's normalization in a separate frame to avoid debug stack cost.
        match self {
            AtomView::Mul(t) => Self::normalize_mul(t, workspace, out, &mut cleanup_alias_handles),
            AtomView::Fun(f) => Self::normalize_fun(f, workspace, out),
            AtomView::Pow(p) => Self::normalize_pow(p, workspace, out),
            AtomView::Add(a) if a.has_alias() => {
                Self::normalize_add_alias(a, workspace, out, &mut cleanup_alias_handles)
            }
            AtomView::Add(a) => Self::normalize_add(a, workspace, out),
            AtomView::Num(_) | AtomView::Var(_) | AtomView::Alias(_) => unreachable!(),
        }
        if cleanup_alias_handles {
            out.refresh_alias_flags_from_tree();
            if out.has_alias() {
                out.refresh_alias_handles_from_tree();
            } else {
                out.clear_alias_handles();
            }
        }
    }

    #[cfg_attr(debug_assertions, inline(never))]
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn normalize_mul(
        t: &MulView<'_>,
        workspace: &Workspace,
        out: &mut Atom,
        cleanup_alias_handles: &mut bool,
    ) {
        let mut atom_test_buf: SmallVec<[_; 20]> = SmallVec::new();

        for a in t.iter() {
            let a = if matches!(a, AtomView::Alias(_)) {
                let resolved = resolve_alias(a);
                if resolved.needs_normalization()
                    || matches!(
                        resolved,
                        AtomView::Num(_) | AtomView::Mul(_) | AtomView::Pow(_)
                    )
                {
                    *cleanup_alias_handles = true;
                    resolved
                } else {
                    a
                }
            } else {
                a
            };
            // Borrow canonical products while flattening them. Their children
            // need owned buffers for merging, but copying the parent is redundant.
            let mut handle;
            let mul = if let AtomView::Mul(mul) = a
                && !a.needs_normalization()
            {
                mul
            } else {
                handle = workspace.new_atom();
                if a.needs_normalization() {
                    let had_alias = a.has_alias();
                    a.normalize(workspace, &mut handle);
                    if had_alias && !handle.has_alias() {
                        *cleanup_alias_handles = true;
                    }
                } else {
                    handle.set_from_view(&a);
                }

                let Atom::Mul(mul) = handle.deref_mut() else {
                    if let AtomView::Num(n) = handle.as_view()
                        && n.is_one()
                    {
                        continue;
                    }
                    atom_test_buf.push(handle);
                    continue;
                };
                mul.to_mul_view()
            };

            for c in mul.iter() {
                if let AtomView::Num(n) = c
                    && n.is_one()
                {
                    continue;
                }
                let mut handle = workspace.new_atom();
                handle.set_from_view(&c);
                atom_test_buf.push(handle);
            }
        }

        atom_test_buf.sort_by(|a, b| a.as_view().cmp_factors(&b.as_view()));

        let mut second_pass = false;
        if !atom_test_buf.is_empty() {
            let out_mul = out.to_mul();

            // Drain in sorted order without reversing the owned buffers.
            let mut factors = atom_test_buf.drain(..);
            let mut last_buf = factors.next().unwrap();

            let mut tmp = workspace.new_atom();
            let mut cur_len = 0;

            for mut cur_buf in factors {
                let had_alias = last_buf.has_alias() || cur_buf.has_alias();
                let merged = last_buf.merge_factors(&mut cur_buf, &mut tmp, workspace);
                if merged && had_alias {
                    *cleanup_alias_handles = true;
                }
                if !merged {
                    // we are done merging
                    {
                        let v = last_buf.as_view();
                        if let AtomView::Num(n) = v {
                            if matches!(
                                n.get_coeff_view(),
                                CoefficientView::Indeterminate | CoefficientView::Infinity(None)
                            ) {
                                out.set_from_view(&v);
                                return;
                            }

                            if n.is_zero() {
                                out.set_from_view(&v);
                                return;
                            }

                            if !n.is_one() {
                                if cur_len > 0 {
                                    // two factors created a number, e.g. 2^(1/2)*2^(1/2) = 2
                                    // we need a second pass to normalize the multiplication
                                    second_pass = true;
                                }

                                out_mul.set_has_coefficient(true);
                                out_mul.extend(v);
                                cur_len += 1;
                            }
                        } else {
                            out_mul.extend(v);
                            cur_len += 1;
                        }
                    }
                    last_buf = cur_buf;
                } else if let AtomView::Mul(_) = last_buf.as_view() {
                    // a sub-multiplication was created during the merge of two factors, e.g. sqrt(x*y)*sqrt(x*y) = x*y
                    // we need a second pass to normalize the multiplication
                    second_pass = true;
                }
            }

            if cur_len == 0 {
                std::mem::swap(out, &mut last_buf);
            } else {
                if second_pass {
                    out_mul.extend(last_buf.as_view());
                    out.as_view().normalize(workspace, &mut tmp);
                    std::mem::swap(out, &mut tmp);
                    return;
                }

                let v = last_buf.as_view();
                if let AtomView::Num(n) = v {
                    if matches!(
                        n.get_coeff_view(),
                        CoefficientView::Indeterminate | CoefficientView::Infinity(None)
                    ) {
                        out.set_from_view(&v);
                        return;
                    }

                    if n.is_zero() {
                        out.set_from_view(&v);
                        return;
                    }

                    if !n.is_one() {
                        out_mul.extend(v);
                        out_mul.set_has_coefficient(true);

                        if cur_len > 0 {
                            // number created during the merge of two factors, e.g. 2^(1/2)*2^(1/2) = 2
                            out.as_view().normalize(workspace, &mut tmp);
                            std::mem::swap(out, &mut tmp);
                            return;
                        }

                        out_mul.set_normalized(true);
                    } else if cur_len == 1 {
                        // downgrade
                        last_buf.set_from_view(&out_mul.to_mul_view().to_slice().get(0));
                        std::mem::swap(out, &mut last_buf);
                    } else {
                        out_mul.set_normalized(true);
                    }
                } else {
                    out_mul.extend(v);
                    out_mul.set_normalized(true);
                }
            }
        } else {
            out.to_num(1);
        }
    }

    #[cfg_attr(debug_assertions, inline(never))]
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn normalize_fun(f: &FunView<'_>, workspace: &Workspace, out: &mut Atom) {
        let id = f.get_symbol();

        if f.has_alias() && Self::normalize_fun_alias_arguments(f, id, workspace, out) {
            return;
        }

        if id.get_id() == Symbol::EXP_ID && f.get_nargs() == 1 {
            let mut h = workspace.new_atom();
            h.to_pow(
                InlineVar::new(Symbol::E).as_view(),
                f.iter().next().unwrap(),
            );
            h.as_view().normalize(workspace, out);
            return;
        }

        if id.get_id() == Symbol::SQRT_ID && f.get_nargs() == 1 {
            let mut h = workspace.new_atom();
            let exp = workspace.new_num((1, 2));
            h.to_pow(f.iter().next().unwrap(), exp.as_view());
            h.as_view().normalize(workspace, out);
            return;
        }

        let out_f = out.to_fun(id);

        if f.iter().all(|a| {
            !a.needs_normalization()
                && !matches!(a, AtomView::Fun(fa) if fa.get_symbol_id() == Symbol::ARG_ID)
        }) {
            // Builders commonly receive canonical arguments. Preserve
            // their encoding instead of rebuilding both headers per arg.
            out_f.set_from_view(f);
        } else {
            let mut handle = workspace.new_atom();
            for a in f {
                if a.needs_normalization() {
                    a.normalize(workspace, &mut handle);
                    Self::add_normalized_arg(out_f, handle.as_view());
                } else {
                    Self::add_normalized_arg(out_f, a);
                }
            }
        }

        out_f.set_normalized(true);

        if Self::normalize_builtin(id, workspace, out) {
            return;
        }
        Self::normalize_fun_attributes(id, workspace, out);
    }

    // Ordinary functions keep compressed arguments. Only expose bodies when a
    // normalization rule needs their shape, an alias hides arg(...) splicing,
    // or a direct alias contains a body that still needs normalization.
    // This is a normalization boundary; equality and hashing never call it.
    #[inline(never)]
    fn normalize_fun_alias_arguments(
        f: &FunView<'_>,
        id: Symbol,
        workspace: &Workspace,
        out: &mut Atom,
    ) -> bool {
        let data = id.get_global_data();
        let inspect_arguments = id.is_flat()
            || id.is_linear()
            || matches!(
                id.get_id(),
                Symbol::EXP_ID
                    | Symbol::SQRT_ID
                    | Symbol::SIN_ID
                    | Symbol::COS_ID
                    | Symbol::LOG_ID
                    | Symbol::ABS_ID
                    | Symbol::IF_ID
                    | Symbol::CONJ_ID
            )
            || id == Symbol::COEFF
            || data.custom_normalization.is_some()
            || data.custom_evaluation.is_some();
        let mut rebuilt: Option<RecycledAtom> = None;
        for (index, arg) in f.iter().enumerate() {
            let inspect_body = if let AtomView::Alias(alias) = arg
                && !alias.is_opaque()
            {
                let body = resolve_alias_loop(arg);
                body.needs_normalization()
                    || matches!(body, AtomView::Fun(inner) if inner.get_symbol_id() == Symbol::ARG_ID)
            } else {
                false
            };
            let expanded = if arg.has_alias()
                && !matches!(arg, AtomView::Alias(alias) if alias.is_opaque())
                && (inspect_arguments || inspect_body)
            {
                Some(arg.inline_aliases(false))
            } else {
                None
            };
            let replacement = expanded
                .as_ref()
                .map(|atom| atom.as_view())
                // Transparent aliases are already equal to their bodies, so
                // test the encoding when deciding whether rebuilding changed it.
                .filter(|view| view.get_data() != arg.get_data());
            if replacement.is_some() && rebuilt.is_none() {
                let mut handle = workspace.new_atom();
                let function = handle.to_fun(id);
                for previous in f.iter().take(index) {
                    function.add_arg(previous);
                }
                rebuilt = Some(handle);
            }
            if let Some(handle) = &mut rebuilt {
                let Atom::Fun(function) = handle.deref_mut() else {
                    unreachable!()
                };
                function.add_arg(replacement.unwrap_or(arg));
            }
        }
        if let Some(rebuilt) = rebuilt {
            // No callback has run for this outer function yet. The rebuilt
            // function invokes it once through the ordinary normalization path.
            rebuilt.as_view().normalize(workspace, out);
            true
        } else {
            // In particular, opaque-only arguments cannot recurse indefinitely.
            false
        }
    }

    #[cfg_attr(debug_assertions, inline(never))]
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn normalize_pow(p: &PowView<'_>, workspace: &Workspace, out: &mut Atom) {
        let (base, exp) = p.get_base_exp();

        // Inspect transparent bodies for numerical and structural power rules.
        // Remember only whether the original alias can be retained, so ordinary
        // operands do not need an extra saved copy of their views.
        let base_is_alias = matches!(base, AtomView::Alias(_));
        let base = if base_is_alias {
            resolve_alias_loop(base)
        } else {
            base
        };
        let preserve_base_alias = base_is_alias && !base.needs_normalization();
        let exp_is_alias = matches!(exp, AtomView::Alias(_));
        let exp = if exp_is_alias {
            resolve_alias_loop(exp)
        } else {
            exp
        };
        let preserve_exp_alias = exp_is_alias && !exp.needs_normalization();

        let mut base_handle;
        let base = if base.needs_normalization() {
            base_handle = workspace.new_atom();
            base.normalize(workspace, &mut base_handle);
            base_handle.as_view()
        } else {
            base
        };

        let mut exp_handle;
        let exp = if exp.needs_normalization() {
            exp_handle = workspace.new_atom();
            exp.normalize(workspace, &mut exp_handle);
            exp_handle.as_view()
        } else {
            exp
        };

        if base == InlineVar::new(Symbol::E).as_view() {
            // simplify logs inside exp
            if exp.contains_symbol(Symbol::LOG) {
                let mut buffer = workspace.new_atom();
                if exp.simplify_exp_log(workspace, &mut buffer) {
                    out.set_from_view(&buffer.as_view());
                    return;
                }
            }
        }

        'pow_simplify: {
            if let AtomView::Num(e) = exp {
                let exp_num = e.get_coeff_view();
                if exp_num == CoefficientView::Natural(1, 1, 0, 1) {
                    // remove power of 1
                    out.set_from_view(&if preserve_base_alias {
                        p.get_base()
                    } else {
                        base
                    });
                    break 'pow_simplify;
                } else if let AtomView::Num(n) = base {
                    // simplify a number raised to a numerical power
                    let (prefactor, new_base_num, new_exp_num) = n.get_coeff_view().pow(&exp_num);

                    if matches!(&new_base_num, Coefficient::Complex(c) if c.re.is_one() && c.im.is_zero())
                    {
                        out.to_num(prefactor);
                        break 'pow_simplify;
                    }

                    if !prefactor.is_one() {
                        let mut mul_h = workspace.new_atom();
                        let m = mul_h.to_mul();
                        let mut number = workspace.new_num(prefactor);
                        m.extend(number.as_view());
                        number.to_num(new_base_num);
                        let exponent = workspace.new_num(new_exp_num);
                        out.to_pow(number.as_view(), exponent.as_view());
                        m.extend(out.as_view());
                        mul_h.as_view().normalize(workspace, out);
                        break 'pow_simplify;
                    }

                    if new_exp_num.is_one() {
                        out.to_num(new_base_num);
                        break 'pow_simplify;
                    }

                    let number = workspace.new_num(new_base_num);
                    let exponent = workspace.new_num(new_exp_num);
                    out.to_pow(number.as_view(), exponent.as_view());
                    break 'pow_simplify;
                } else if let AtomView::Pow(p_base) = base {
                    let exp_is_integer = exp_num.is_integer();
                    if exp_is_integer || base.is_nonnegative().is_true() {
                        let (p_base_base, p_base_exp) = p_base.get_base_exp();
                        let mut mul_h = workspace.new_atom();
                        let mul = mul_h.to_mul();

                        mul.extend(p_base_exp);
                        mul.extend(exp);
                        let mut exp_h = workspace.new_atom();
                        mul.as_view().normalize(workspace, &mut exp_h);

                        if exp_is_integer || p_base_base.is_nonnegative().is_true() {
                            mul_h.to_pow(p_base_base, exp_h.as_view());
                        } else {
                            // the base-base is real but not positive, so add abs
                            let mut new_base = workspace.new_atom();
                            let abs_fun = new_base.to_fun(Symbol::ABS);
                            abs_fun.add_arg(p_base_base);
                            mul_h.to_pow(new_base.as_view(), exp_h.as_view());
                        }

                        mul_h.as_view().normalize(workspace, out);
                        break 'pow_simplify;
                    }
                } else if let AtomView::Mul(m) = base {
                    // rewrite (x*y)^2 as x^2*y^2
                    if exp_num.is_integer() {
                        let mut mul_h = workspace.new_atom();
                        let mul = mul_h.to_mul();
                        for arg in m {
                            let mut pow_h = workspace.new_atom();
                            pow_h.to_pow(arg, exp);
                            mul.extend(pow_h.as_view());
                        }

                        mul_h.as_view().normalize(workspace, out);
                        break 'pow_simplify;
                    }
                } else if exp_num == CoefficientView::Natural(0, 1, 0, 1) {
                    // x^0 = 1
                    out.to_num(1);
                    break 'pow_simplify;
                } else if let CoefficientView::Natural(n, 1, 0, 1) = exp_num
                    && n % 2 == 0
                    && let AtomView::Fun(f) = base
                {
                    let s = f.get_symbol_id();
                    if s == Symbol::ABS_ID && f.get_nargs() == 1 {
                        let abs_arg = f.iter().next().unwrap();
                        if abs_arg.is_real().is_true() {
                            let mut pow_h = workspace.new_atom();
                            pow_h.to_pow(abs_arg, workspace.new_num(n).as_view());
                            pow_h.as_view().normalize(workspace, out);
                            out.set_from_view(&pow_h.as_view());
                            break 'pow_simplify;
                        }
                    }
                }
            } else if let AtomView::Pow(p_base) = base
                && exp.is_integer().is_true()
            {
                // rewrite (x^y)^z as x^(z*y) if z is integer
                let (p_base_base, p_base_exp) = p_base.get_base_exp();

                let mut mul_h = workspace.new_atom();
                let mul = mul_h.to_mul();
                mul.extend(p_base_exp);
                mul.extend(exp);
                let mut exp_h = workspace.new_atom();
                mul.as_view().normalize(workspace, &mut exp_h);

                mul_h.to_pow(p_base_base, exp_h.as_view());
                mul_h.as_view().normalize(workspace, out);
                break 'pow_simplify;
            } else if base.is_one() {
                out.to_num(Coefficient::one()); // 1^x = 1 for any non-infinite x
                return;
            }

            // No simplification changed these operands. Preserve compressed
            // canonical aliases rather than copying their potentially large bodies.
            out.to_pow(
                if preserve_base_alias {
                    p.get_base()
                } else {
                    base
                },
                if preserve_exp_alias { p.get_exp() } else { exp },
            );
        }

        out.set_normalized(true);
    }

    #[cfg_attr(debug_assertions, inline(never))]
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn normalize_add(a: &AddView<'_>, workspace: &Workspace, out: &mut Atom) {
        let mut new_sum = workspace.new_atom();
        let ns = new_sum.to_add();

        let mut atom_sort_buf: SmallVec<[(AtomView<'_>, &[u8]); 20]> =
            SmallVec::with_capacity(a.get_nargs());

        let mut norm_arg = None;
        let mut staged_prefix = None;
        for x in a {
            if x.needs_normalization() {
                let norm_arg = norm_arg.get_or_insert_with(|| workspace.new_atom());
                x.normalize(workspace, norm_arg);
                let r = norm_arg.as_view();
                if let AtomView::Num(n) = r {
                    if matches!(n.get_coeff_view(), CoefficientView::Indeterminate) {
                        out.set_from_view(&r);
                        return;
                    } else if n.is_zero() {
                        continue;
                    }
                }

                let count = if let AtomView::Add(sum) = r {
                    sum.get_nargs()
                } else {
                    1
                };
                ns.extend(r);
                // Preserve the input order without borrowing ns while
                // it can still reallocate. The original dirty view marks
                // each slot to replace once the buffer is complete.
                if atom_sort_buf.is_empty() {
                    // Delay these slots until an input view is borrowed.
                    // If every term changes, just read ns in one pass.
                    staged_prefix = Some(x);
                } else {
                    atom_sort_buf.resize(atom_sort_buf.len() + count, (x, &[]));
                }
            } else {
                if let AtomView::Num(n) = x {
                    if matches!(n.get_coeff_view(), CoefficientView::Indeterminate) {
                        out.set_from_view(&x);
                        return;
                    } else if n.is_zero() {
                        continue;
                    }
                }
                if let Some(marker) = staged_prefix.take() {
                    atom_sort_buf.resize(ns.get_nargs(), (marker, &[]));
                }
                if let AtomView::Add(sum) = x {
                    for term in sum {
                        atom_sort_buf.push((term, term.get_term_cmp_slice()));
                    }
                } else {
                    atom_sort_buf.push((x, x.get_term_cmp_slice()));
                }
            }
        }

        if staged_prefix.is_some() {
            for term in ns.to_add_view().iter() {
                atom_sort_buf.push((term, term.get_term_cmp_slice()));
            }
        } else if ns.get_nargs() != 0 {
            let mut normalized = ns.to_add_view().iter();
            for entry in &mut atom_sort_buf {
                if entry.0.needs_normalization() {
                    let term = normalized.next().unwrap();
                    *entry = (term, term.get_term_cmp_slice());
                }
            }
            debug_assert!(normalized.next().is_none());
        }

        // A normalization hook can introduce an alias into an initially
        // alias-free sum. Reuse the normalized children in the semantic path.
        if ns.to_add_view().has_alias() {
            let mut aliased_sum = workspace.new_atom();
            let add = aliased_sum.to_add();
            for (term, _) in &atom_sort_buf {
                add.extend(*term);
            }
            aliased_sum.as_view().normalize(workspace, out);
            return;
        }

        atom_sort_buf.sort_unstable_by(|a, b| a.1.cmp(b.1));

        if atom_sort_buf.is_empty() {
            out.to_num(Coefficient::zero());
            return;
        }
        let out_add = out.to_add();

        let mut helper = workspace.new_atom();
        let mut helper2 = workspace.new_atom();
        let mut second_pass = false;

        for b in atom_sort_buf.chunk_by(|x, y| x.1 == y.1) {
            if b.len() == 1 {
                let first = b[0].0;
                if let AtomView::Num(n) = first
                    && matches!(
                        n.get_coeff_view(),
                        CoefficientView::Indeterminate | CoefficientView::Infinity(_)
                    )
                {
                    out.set_from_view(&first);
                    return;
                }
                out_add.extend(first);
                continue;
            }

            fn coefficient(cur: AtomView<'_>) -> CoefficientView<'_> {
                match cur {
                    AtomView::Num(n) => n.get_coeff_view(),
                    AtomView::Mul(m) => {
                        if let Some(AtomView::Num(n)) = m.get_coefficient() {
                            n.get_coeff_view()
                        } else {
                            CoefficientView::Natural(1, 1, 0, 1)
                        }
                    }
                    _ => CoefficientView::Natural(1, 1, 0, 1),
                }
            }
            // Most term coefficients are small integers. Avoid general
            // rational arithmetic until a different domain or overflow
            // requires it. Restarting preserves the original addition
            // order and does not introduce integer zero into a field.
            let integer_sum = b.iter().try_fold(0i64, |sum, (cur, _)| {
                if let CoefficientView::Natural(n, 1, 0, 1) = coefficient(*cur) {
                    sum.checked_add(n)
                } else {
                    None
                }
            });
            let coeff = if let Some(sum) = integer_sum {
                Coefficient::from(sum)
            } else {
                let mut coeff = coefficient(b[0].0).to_owned();
                for (cur, _) in &b[1..] {
                    coeff = coeff + coefficient(*cur).to_owned();
                }
                coeff
            };

            let first = b[0].0;
            if coeff.is_zero() {
                continue;
            }

            if matches!(first, AtomView::Num(_)) {
                if matches!(coeff, Coefficient::Indeterminate | Coefficient::Infinity(_)) {
                    out.to_num(coeff);
                    return;
                }
                helper.to_num(coeff);
                out_add.extend(helper.as_view());
                continue;
            }
            if matches!(coeff, Coefficient::Indeterminate) {
                out.to_num(coeff);
                return;
            }

            let has_coeff = !coeff.is_one();
            if !has_coeff {
                let single = match first {
                    AtomView::Mul(m) => {
                        let factors = m.to_slice().fast_forward(usize::from(m.has_coefficient()));
                        (factors.len() == 1).then(|| factors.get(0))
                    }
                    _ => Some(first),
                };
                if let Some(term) = single {
                    // Removing the coefficient can expose a sum:
                    // (x+y)/2 + (x+y)/2 = x+y. Flatten and re-sort it.
                    second_pass |= matches!(term, AtomView::Add(_));
                    out_add.extend(term);
                    continue;
                }
            }

            let m = helper.to_mul();
            if has_coeff {
                m.extend(helper2.to_num(coeff).as_view());
            }
            if let AtomView::Mul(mm) = first {
                for x in mm.iter().skip(usize::from(mm.has_coefficient())) {
                    m.extend(x);
                }
            } else {
                m.extend(first);
            }
            m.set_has_coefficient(has_coeff);
            m.set_normalized(true);
            out_add.extend(m.as_view());
        }

        if second_pass {
            out.as_view().normalize(workspace, &mut helper);
            std::mem::swap(out, &mut helper);
        } else {
            match out_add.get_nargs() {
                0 => {
                    out.to_num(Coefficient::zero());
                }
                1 => {
                    helper.set_from_view(&out_add.to_add_view().to_slice().get(0));
                    std::mem::swap(out, &mut helper);
                }
                _ => out_add.set_normalized(true),
            }
        }
    }

    // Keep alias-specific traversal and temporary buffers out of the ordinary
    // normalization frame. This path is only selected for sums with aliases.
    #[inline(never)]
    fn normalize_add_alias(
        a: &AddView<'_>,
        workspace: &Workspace,
        out: &mut Atom,
        cleanup_alias_handles: &mut bool,
    ) {
        let mut new_sum = workspace.new_atom();
        let ns = new_sum.to_add();

        let mut atom_sort_buf: SmallVec<[_; 20]> = SmallVec::new();

        let mut norm_arg = workspace.new_atom();
        for a in a {
            let resolved_a = resolve_alias(a);
            let a = if matches!(a, AtomView::Alias(_))
                && (resolved_a.needs_normalization()
                    || matches!(resolved_a, AtomView::Num(_) | AtomView::Add(_)))
            {
                *cleanup_alias_handles = true;
                resolved_a
            } else {
                a
            };

            let r = if a.needs_normalization() {
                // TODO: if a is a nested addition, prevent a sort
                let had_alias = a.has_alias();
                a.normalize(workspace, &mut norm_arg);
                if had_alias && !norm_arg.has_alias() {
                    *cleanup_alias_handles = true;
                }
                norm_arg.as_view()
            } else {
                a
            };

            if let AtomView::Add(new_add) = r {
                for c in new_add.iter() {
                    if let AtomView::Num(n) = c
                        && n.is_zero()
                    {
                        continue;
                    }

                    ns.extend(c);
                }
            } else {
                if let AtomView::Num(n) = r {
                    let coeff = n.get_coeff_view();
                    if matches!(coeff, CoefficientView::Indeterminate) {
                        out.set_from_view(&r);
                        return;
                    }

                    if n.is_zero() {
                        continue;
                    }
                }

                ns.extend(r); // TODO: prevent copy?
            }
        }

        let has_alias = ns.to_add_view().has_alias();

        for x in ns.to_add_view().iter() {
            if !has_alias {
                atom_sort_buf.push((x, x.get_term_cmp_slice()));
            } else {
                atom_sort_buf.push((x, &[]));
            }
        }

        if has_alias {
            atom_sort_buf.sort_unstable_by(|a, b| a.0.cmp_terms(&b.0, false))
        } else {
            atom_sort_buf.sort_unstable_by(|a, b| a.1.cmp(b.1));
        }

        if atom_sort_buf.is_empty() {
            out.to_num(Coefficient::zero());
            return;
        }
        let out_add = out.to_add();

        let mut last_buf = workspace.new_atom();
        last_buf.set_from_view(&atom_sort_buf[0].0);

        let mut helper = workspace.new_atom();
        let mut cur_len = 0;

        for (cur, _) in atom_sort_buf.iter().skip(1) {
            let last_had_alias = last_buf.has_alias();
            let cur_had_alias = cur.has_alias();
            let merged = last_buf.merge_terms(*cur, &mut helper);
            if merged && (last_had_alias || cur_had_alias) {
                *cleanup_alias_handles = true;
            }

            if !merged {
                // we are done merging
                let v = last_buf.as_view();
                if let AtomView::Num(n) = v {
                    let coeff = n.get_coeff_view();
                    if matches!(
                        coeff,
                        CoefficientView::Indeterminate | CoefficientView::Infinity(_)
                    ) {
                        out.set_from_view(&v);
                        return;
                    }

                    if !n.is_zero() {
                        out_add.extend(v);
                        cur_len += 1;
                    }
                } else {
                    out_add.extend(v);
                    cur_len += 1;
                }

                // TODO: prevent this copy, as it occurs on every non-merge
                cur.clone_into(&mut last_buf);
            }
        }

        if cur_len == 0 {
            std::mem::swap(out, &mut last_buf);
        } else {
            let v = last_buf.as_view();
            if let AtomView::Num(n) = v {
                let coeff = n.get_coeff_view();
                if matches!(
                    coeff,
                    CoefficientView::Indeterminate | CoefficientView::Infinity(_)
                ) {
                    out.set_from_view(&v);
                    return;
                }

                if !n.is_zero() {
                    out_add.extend(v);
                    out_add.set_normalized(true);
                } else if cur_len == 1 {
                    // downgrade
                    last_buf.set_from_view(&out_add.to_add_view().to_slice().get(0));
                    std::mem::swap(out, &mut last_buf);
                } else {
                    out_add.set_normalized(true);
                }
            } else {
                out_add.extend(v);
                out_add.set_normalized(true);
            }
        }
    }

    /// Add an argument `a` to `f` and flatten nested `arg`s.
    #[inline(always)]
    fn add_normalized_arg(f: &mut Fun, a: AtomView) {
        if let AtomView::Fun(fa) = a
            && fa.get_symbol_id() == Symbol::ARG_ID
        {
            // flatten f(arg(...)) = f(...)
            for aa in fa.iter() {
                f.add_arg(aa);
            }

            return;
        }

        f.add_arg(a);
    }

    #[cfg_attr(debug_assertions, inline(never))]
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn normalize_builtin(id: Symbol, workspace: &Workspace, out: &mut Atom) -> bool {
        let Atom::Fun(out_f) = out else {
            unreachable!()
        };

        if [
            Symbol::COS_ID,
            Symbol::SIN_ID,
            Symbol::LOG_ID,
            Symbol::ABS_ID,
        ]
        .contains(&id.get_id())
            && out_f.to_fun_view().get_nargs() == 1
        {
            let arg = out_f.to_fun_view().iter().next().unwrap();
            if let AtomView::Num(n) = arg {
                if n.is_zero() && id != Symbol::LOG || n.is_one() && id == Symbol::LOG {
                    if id == Symbol::COS {
                        out.to_num(Coefficient::one());
                        return true;
                    } else if id == Symbol::SIN || id == Symbol::LOG || id == Symbol::ABS {
                        out.to_num(Coefficient::zero());
                        return true;
                    }
                }

                if n.is_zero() && id == Symbol::LOG {
                    warn!("Created infinity by log(0)");
                    out.to_num(Coefficient::Infinity(Some(Rational::new(-1, 1).into())));
                    return true;
                }

                if let CoefficientView::Float(r, i) = n.get_coeff_view() {
                    match id.get_id() {
                        Symbol::COS_ID => {
                            let r = if i.is_zero() {
                                r.to_float().cos().into()
                            } else {
                                Complex::new(r.to_float(), i.to_float()).cos()
                            };
                            out.to_num(Coefficient::Float(r));
                            return true;
                        }
                        Symbol::SIN_ID => {
                            let r = if i.is_zero() {
                                r.to_float().sin().into()
                            } else {
                                Complex::new(r.to_float(), i.to_float()).sin()
                            };
                            out.to_num(Coefficient::Float(r));
                            return true;
                        }
                        Symbol::LOG_ID => {
                            let r = if i.is_zero() {
                                r.to_float().log().into()
                            } else {
                                Complex::new(r.to_float(), i.to_float()).log()
                            };
                            out.to_num(Coefficient::Float(r));
                            return true;
                        }
                        Symbol::ABS_ID => {
                            let r = if i.is_zero() {
                                r.to_float().norm().into()
                            } else {
                                Complex::new(r.to_float(), i.to_float()).norm()
                            };
                            out.to_num(Coefficient::Float(r));
                            return true;
                        }
                        _ => {}
                    }
                }

                if id.get_id() == Symbol::ABS_ID
                    && let Coefficient::Complex(c) = n.get_coeff_view().to_owned()
                {
                    if c.is_real() {
                        out.to_num(c.re.abs());
                    } else {
                        let r = c.norm_squared();
                        let mut buffer = workspace.new_atom();
                        buffer.to_num(r);
                        *out = buffer.pow((1, 2));
                    }
                    return true;
                }
            } else if id.get_id() == Symbol::ABS_ID && arg.is_nonnegative().is_true() {
                let mut buffer = workspace.new_atom();
                buffer.set_from_view(&arg);
                out.set_from_view(&buffer.as_view());
                return true;
            }
        }

        if id == Symbol::IF && out_f.to_fun_view().get_nargs() == 3 {
            let mut iter = out_f.to_fun_view().iter();

            if let AtomView::Num(x) = iter.next().unwrap() {
                let mut buffer = workspace.new_atom();
                if x.is_zero() {
                    iter.next();
                }

                buffer.set_from_view(&iter.next().unwrap());
                out.set_from_view(&buffer.as_view());
                return true;
            }
        }

        if id == Symbol::CONJ && out_f.to_fun_view().get_nargs() == 1 {
            let arg = out_f.to_fun_view().iter().next().unwrap();

            match arg {
                AtomView::Alias(a) if !a.is_opaque() => {
                    let mut inner = workspace.new_atom();
                    inner.to_fun(Symbol::CONJ).add_arg(a.get_body());
                    inner.as_view().normalize(workspace, out);
                }
                AtomView::Alias(_) => {}
                AtomView::Num(n) => {
                    let conj_coeff = n.get_coeff_view().to_owned().conjugate();
                    out.to_num(conj_coeff);
                }
                AtomView::Var(v) => {
                    let s = v.get_symbol();
                    if s.is_real() {
                        out.to_var(s);
                    }
                }
                AtomView::Fun(ff) => {
                    let s = ff.get_symbol();
                    if s == Symbol::CONJ {
                        // conj(conj(a)) = a
                        let inner_arg = ff.iter().next().unwrap();
                        let mut inner = workspace.new_atom();
                        inner.set_from_view(&inner_arg);
                        out.set_from_view(&inner.as_view());
                    } else if s.is_real() {
                        let mut inner = workspace.new_atom();
                        inner.set_from_view(&arg);
                        out.set_from_view(&inner.as_view());
                    }
                }
                AtomView::Pow(p) => {
                    if arg.is_real().is_true() {
                        let mut inner = workspace.new_atom();
                        inner.set_from_view(&arg);
                        out.set_from_view(&inner.as_view());
                    } else {
                        let (b, e) = p.get_base_exp();
                        if e.is_integer().is_true() {
                            let mut new_base = workspace.new_atom();
                            let nb = new_base.to_fun(Symbol::CONJ);
                            nb.add_arg(b);
                            let mut new_pow = workspace.new_atom();
                            new_pow.to_pow(new_base.as_view(), e);
                            new_pow.as_view().normalize(workspace, out);
                        } else if b.is_nonnegative().is_true() {
                            let mut new_exp = workspace.new_atom();
                            let ne = new_exp.to_fun(Symbol::CONJ);
                            ne.add_arg(e);
                            let mut new_pow = workspace.new_atom();
                            new_pow.to_pow(b, new_exp.as_view());
                            new_pow.as_view().normalize(workspace, out);
                        }
                    }
                }
                AtomView::Mul(m) => {
                    let mut new_mul = workspace.new_atom();
                    let nm = new_mul.to_mul();

                    let mut conj_a = workspace.new_atom();
                    for aa in m {
                        conj_a.to_fun(Symbol::CONJ).add_arg(aa);
                        nm.extend(conj_a.as_view());
                    }

                    new_mul.as_view().normalize(workspace, out);
                }
                AtomView::Add(a) => {
                    let mut new_add = workspace.new_atom();
                    let na = new_add.to_add();

                    let mut conj_a = workspace.new_atom();
                    for aa in a {
                        conj_a.to_fun(Symbol::CONJ).add_arg(aa);
                        na.extend(conj_a.as_view());
                    }

                    new_add.as_view().normalize(workspace, out);
                }
            }

            return true;
        }

        // simplify log(exp(real)) = real
        if id == Symbol::LOG && out_f.to_fun_view().get_nargs() == 1 {
            let arg = out_f.to_fun_view().iter().next().unwrap();

            if arg == InlineVar::new(Symbol::E).as_view() {
                out.set_from_view(&InlineNum::one().as_view());
                return true;
            }

            if let AtomView::Pow(p) = arg {
                let (b, e) = p.get_base_exp();
                // TODO: support comparison with symbol?
                if b == InlineVar::new(Symbol::E).as_view() {
                    if e.is_real().is_true() {
                        let mut buffer = workspace.new_atom();
                        buffer.set_from_view(&e);
                        out.set_from_view(&buffer.as_view());
                        return true;
                    }
                }
            }
        }

        // try to turn the argument into a number
        if id == Symbol::COEFF && out_f.to_fun_view().get_nargs() == 1 {
            let arg = out_f.to_fun_view().iter().next().unwrap();
            if let AtomView::Num(_) = arg {
                let mut buffer = workspace.new_atom();
                buffer.set_from_view(&arg);
                out.set_from_view(&buffer.as_view());
                return true;
            } else {
                let r = arg.to_rational_polynomial(&Q, &Z, None);

                // disallow wildcards as variables
                if r.numerator.get_vars_ref().iter().all(|v| {
                    if let PolyVariable::Symbol(v) = v {
                        v.get_wildcard_level() == 0
                    } else {
                        false
                    }
                }) {
                    out.to_num(Coefficient::RationalPolynomial(r));
                    return true;
                }
            }
        }

        false
    }

    #[cfg_attr(debug_assertions, inline(never))]
    #[cfg_attr(not(debug_assertions), inline(always))]
    fn normalize_fun_attributes(id: Symbol, workspace: &Workspace, out: &mut Atom) {
        let Atom::Fun(out_f) = out else {
            unreachable!()
        };

        /// Take Cartesian product of arguments
        #[inline(always)]
        fn cartesian_product<'b>(
            workspace: &Workspace,
            list: &[Vec<AtomView<'b>>],
            fun_name: Symbol,
            cur: &mut Vec<AtomView<'b>>,
            acc: &mut Vec<RecycledAtom>,
        ) {
            if list.is_empty() {
                let mut h = workspace.new_atom();
                let f = h.to_fun(fun_name);
                for a in cur.iter() {
                    AtomView::add_normalized_arg(f, *a);
                }
                acc.push(h);
                return;
            }

            for a in &list[0] {
                cur.push(*a);
                cartesian_product(workspace, &list[1..], fun_name, cur, acc);
                cur.pop();
            }
        }

        if id.is_flat()
            && out_f.to_fun_view().iter().any(|a| {
                if let AtomView::Fun(inner_f) = a {
                    inner_f.get_symbol() == id
                } else {
                    false
                }
            })
        {
            let mut flat_h = workspace.new_atom();
            let flat_f = flat_h.to_fun(id);
            for a in out_f.to_fun_view().iter() {
                if let AtomView::Fun(inner_f) = a
                    && inner_f.get_symbol() == id
                {
                    for b in inner_f.iter() {
                        flat_f.add_arg(b);
                    }
                } else {
                    flat_f.add_arg(a);
                }
            }

            flat_h.as_view().normalize(workspace, out);
            return;
        }

        if id.is_linear() {
            // linearize sums
            if out_f
                .to_fun_view()
                .iter()
                .any(|a| matches!(a, AtomView::Add(_)))
            {
                let mut arg_buf = Vec::with_capacity(out_f.to_fun_view().get_nargs());

                for a in out_f.to_fun_view().iter() {
                    let mut vec = vec![];
                    if let AtomView::Add(aa) = a {
                        for a in aa.iter() {
                            vec.push(a);
                        }
                    } else {
                        vec.push(a);
                    }
                    arg_buf.push(vec);
                }

                let mut acc = Vec::new();
                cartesian_product(workspace, &arg_buf, id, &mut vec![], &mut acc);

                let mut add_h = workspace.new_atom();
                let add = add_h.to_add();

                let mut h = workspace.new_atom();
                for a in acc {
                    a.as_view().normalize(workspace, &mut h);
                    add.extend(h.as_view());
                }

                add_h.as_view().normalize(workspace, out);
                return;
            }

            // linearize products
            if out_f.to_fun_view().iter().any(|a| {
                if let AtomView::Mul(m) = a {
                    m.has_coefficient() || m.iter().any(|a| a.is_scalar().is_true())
                } else {
                    false
                }
            }) {
                let mut new_term = workspace.new_atom();
                let t = new_term.to_mul();
                let mut new_fun = workspace.new_atom();
                let nf = new_fun.to_fun(id);
                let mut coeff: Coefficient = 1.into();
                for a in out_f.to_fun_view().iter() {
                    if let AtomView::Mul(m) = a {
                        let mut stripped = workspace.new_atom();
                        let mul = stripped.to_mul();

                        for a in m {
                            if let AtomView::Num(n) = a {
                                coeff = coeff * n.get_coeff_view().to_owned();
                            } else if a.is_scalar().is_true() {
                                t.extend(a);
                            } else {
                                mul.extend(a);
                            }
                        }

                        nf.add_arg(stripped.as_view());
                    } else {
                        nf.add_arg(a);
                    }
                }

                t.extend(new_fun.as_view());
                t.extend(workspace.new_num(coeff).as_view());
                t.as_view().normalize(workspace, out);
                return;
            }

            for a in out_f.to_fun_view() {
                if let AtomView::Num(n) = a
                    && n.is_zero()
                {
                    out.to_num(Coefficient::zero());
                    return;
                }
            }
        }

        let sort_antisymmetric =
            id.is_antisymmetric() && !out_f.to_fun_view().iter().any(|a| a.contains_wildcard());

        if id.is_symmetric() || sort_antisymmetric {
            let mut arg_buf: SmallVec<[(usize, _); 20]> = SmallVec::new();

            for (i, a) in out_f.to_fun_view().iter().enumerate() {
                let mut handle = workspace.new_atom();
                handle.set_from_view(&a);
                arg_buf.push((i, handle));
            }

            arg_buf.sort_by(|a, b| a.1.as_view().cmp(&b.1.as_view()));

            if id.is_antisymmetric() {
                if arg_buf
                    .windows(2)
                    .any(|w| w[0].1.as_view().cmp(&w[1].1.as_view()).is_eq())
                {
                    out.to_num(Coefficient::zero());
                    return;
                }

                // find the number of swaps needed to sort the arguments
                let mut order: SmallVec<[usize; 20]> = (0..arg_buf.len())
                    .map(|i| arg_buf.iter().position(|(j, _)| *j == i).unwrap())
                    .collect();
                let mut swaps = 0;
                for i in 0..order.len() {
                    let pos = order[i..].iter().position(|&x| x == i).unwrap();
                    order.copy_within(i..i + pos, i + 1);
                    swaps += pos;
                }

                if swaps % 2 == 1 {
                    let mut handle = workspace.new_atom();
                    let out_f = handle.to_fun(id);

                    for (_, a) in arg_buf {
                        out_f.add_arg(a.as_view());
                    }

                    out_f.set_normalized(true);

                    if let Some(f) = State::get_normalization_function(id) {
                        let mut fs = workspace.new_atom();
                        let mut setter = fs.deref_mut().into();
                        f(handle.as_view(), &mut setter);
                        if setter.is_set() {
                            std::mem::swap(&mut handle, &mut fs);
                        }
                        debug_assert!(!handle.as_view().needs_normalization());
                    }

                    let m = out.to_mul();
                    m.extend(InlineNum::new(-1, 1).as_view());
                    m.extend(handle.as_view());
                    m.set_has_coefficient(true);
                    m.set_normalized(true);

                    return;
                }
            }

            let out_f = out.to_fun(id);
            for (_, a) in arg_buf {
                out_f.add_arg(a.as_view());
            }

            out_f.set_normalized(true);
        } else if id.is_cyclesymmetric() {
            let mut args: SmallVec<[_; 20]> = SmallVec::new();
            for a in out_f.to_fun_view().iter() {
                args.push(a);
            }

            let best_shift = minimum_rotation_by(&args, AtomView::cmp);

            let mut f = workspace.new_atom();
            let ff = f.to_fun(id);
            for arg in args[best_shift..].iter().chain(&args[..best_shift]) {
                ff.add_arg(*arg);
            }

            drop(args);

            ff.set_normalized(true);
            std::mem::swap(ff, out_f);
        }

        let data = id.get_global_data();
        if let Some(n) = &data.custom_normalization {
            let mut fs = workspace.new_atom();
            let mut setter = fs.deref_mut().into();
            n(out.as_view(), &mut setter);
            if setter.is_set() {
                std::mem::swap(out, fs.deref_mut());
            }
            debug_assert!(!out.as_view().needs_normalization());
        }

        if let Some(e) = &data.custom_evaluation
            && e.get_tag_count() == 0
            && let Some(e) = e.get_evaluator::<Complex<Float>>(&[])
            && let AtomView::Fun(f) = out.as_view()
            && f.iter().all(|arg| {
                if let AtomView::Num(n) = arg
                    && n.get_coeff_view().is_float()
                {
                    true
                } else {
                    false
                }
            })
        {
            let args = f
                .iter()
                .map(|x| Complex::<Float>::try_from(x).unwrap())
                .collect::<Vec<_>>();
            let result = (e)(&args);
            out.to_num(result);
        }
    }

    /// Add two atoms and normalize the result.
    pub(crate) fn add_normalized(&self, rhs: AtomView, ws: &Workspace, out: &mut Atom) {
        if self.has_alias() || rhs.has_alias() {
            let mut e = ws.new_atom();
            let a = e.to_add();
            a.extend(*self);
            a.extend(rhs);
            e.as_view().normalize(ws, out);
            return;
        }

        // write (a+b)+(a+b) as 2*(a+b)
        if self.get_data() == rhs.get_data() {
            let mut a = ws.new_atom();
            let m = a.to_mul();
            m.extend(InlineNum::new(2, 1).as_view());
            m.extend(*self);
            a.as_view().normalize(ws, out);
            return;
        }

        if !matches!(self, AtomView::Add(_)) {
            if matches!(rhs, AtomView::Add(_)) {
                rhs.add_normalized(*self, ws, out);
            } else {
                self.add_normalized_pair(rhs, ws, out);
            }
            return;
        }

        let a = out.to_add();
        a.grow_capacity(self.get_byte_size() + rhs.get_byte_size());

        // The alias-bearing case returned before constructing the output.
        let no_aliases = true;

        let mut helper = ws.new_atom();
        let mut b = ws.new_atom();
        let mut second_pass = false;
        if let AtomView::Add(a1) = self
            && let AtomView::Add(a2) = rhs
        {
            let mut s = a1.iter();
            let mut t = a2.iter();

            let mut curs = s.next();
            let mut curst = t.next();
            while curs.is_some() || curst.is_some() {
                if let Some(ss) = curs {
                    if let Some(tt) = curst {
                        match ss.cmp_terms(&tt, no_aliases) {
                            Ordering::Less => {
                                a.extend(ss);
                                curs = s.next();
                            }
                            Ordering::Greater => {
                                a.extend(tt);
                                curst = t.next();
                            }
                            Ordering::Equal => {
                                b.set_from_view(&ss);
                                if b.merge_terms(tt, &mut helper) {
                                    second_pass |= matches!(b.as_view(), AtomView::Add(_));
                                    if let AtomView::Num(n) = b.as_view() {
                                        if !n.is_zero() {
                                            a.extend(b.as_view());
                                        }
                                    } else {
                                        a.extend(b.as_view());
                                    }
                                } else {
                                    unreachable!(
                                        "Equal terms do not merge:\n{}\n{}\n{:#?}\n{:#?}",
                                        ss, tt, ss, tt
                                    );
                                }

                                curst = t.next();
                                curs = s.next();
                            }
                        }
                    } else {
                        a.extend(ss);
                        curs = s.next();
                    }
                } else if let Some(tt) = curst {
                    a.extend(tt);
                    curst = t.next();
                }
            }

            if a.get_nargs() == 0 {
                out.to_num(Coefficient::zero());
            } else if a.get_nargs() == 1 {
                let mut b = ws.new_atom();
                b.set_from_view(&a.to_add_view().iter().next().unwrap());
                std::mem::swap(out, &mut b);
            } else if second_pass {
                out.as_view().normalize(ws, &mut helper);
                std::mem::swap(out, &mut helper);
            } else {
                a.set_normalized(true);
            }
            return;
        }

        if let AtomView::Add(a1) = self {
            if rhs.is_zero() {
                self.clone_into(out);
                return;
            }

            if let AtomView::Num(n) = rhs {
                let r = n.get_coeff_view();
                if r == CoefficientView::Indeterminate || r == CoefficientView::Infinity(None) {
                    rhs.clone_into(out);
                    return;
                }
            }

            if a1.get_nargs() < 50 {
                let mut found = false;
                for x in a1.iter() {
                    if found {
                        a.extend(x);
                        continue;
                    }

                    match x.cmp_terms(&rhs, no_aliases) {
                        Ordering::Less => {
                            a.extend(x);
                        }
                        Ordering::Equal => {
                            found = true;
                            b.set_from_view(&x);
                            if b.merge_terms(rhs, &mut helper) {
                                second_pass |= matches!(b.as_view(), AtomView::Add(_));
                                if let AtomView::Num(n) = b.as_view() {
                                    if !n.is_zero() {
                                        a.extend(b.as_view());
                                    }
                                } else {
                                    a.extend(b.as_view());
                                }
                            } else {
                                unreachable!(
                                    "Equal terms do not merge:\n{}\n{}\n{:#?}\n{:#?}",
                                    x, rhs, x, rhs
                                );
                            }
                        }
                        Ordering::Greater => {
                            found = true;
                            a.extend(rhs);
                            a.extend(x);
                        }
                    }
                }

                if !found {
                    a.extend(rhs);
                }
            } else {
                let v: Vec<_> = a1.iter().collect();
                match v.binary_search_by(|a| a.cmp_terms(&rhs, no_aliases)) {
                    Ok(p) => {
                        for x in v.iter().take(p) {
                            a.extend(*x);
                        }

                        b.set_from_view(&v[p]);
                        if b.merge_terms(rhs, &mut helper) {
                            second_pass |= matches!(b.as_view(), AtomView::Add(_));
                            if let AtomView::Num(n) = b.as_view() {
                                if !n.is_zero() {
                                    a.extend(b.as_view());
                                }
                            } else {
                                a.extend(b.as_view());
                            }
                        } else {
                            unreachable!(
                                "Equal terms do not merge:\n{}\n{}\n{:#?}\n{:#?}",
                                b, rhs, b, rhs
                            );
                        }

                        for x in v.iter().skip(p + 1) {
                            a.extend(*x);
                        }
                    }
                    Err(p) => {
                        for x in v.iter().take(p) {
                            a.extend(*x);
                        }

                        a.extend(rhs);

                        for x in v.iter().skip(p) {
                            a.extend(*x);
                        }
                    }
                }
            }

            if a.get_nargs() == 1 {
                let mut b = ws.new_atom();
                b.set_from_view(&a.to_add_view().iter().next().unwrap());
                std::mem::swap(out, &mut b);
            } else if second_pass {
                out.as_view().normalize(ws, &mut helper);
                std::mem::swap(out, &mut helper);
            } else {
                a.set_normalized(true);
            }
        }
    }

    /// Add two terms after excluding sums and aliases.
    fn add_normalized_pair(&self, rhs: AtomView, ws: &Workspace, out: &mut Atom) {
        if !self.needs_normalization()
            && !rhs.needs_normalization()
            && !matches!(self, AtomView::Num(_))
            && !matches!(rhs, AtomView::Num(_))
        {
            let order = self.get_term_cmp_slice().cmp(rhs.get_term_cmp_slice());
            if order != Ordering::Equal {
                // Canonical terms with distinct monomials need neither staging nor sorting.
                let a = out.to_add();
                a.grow_capacity(self.get_byte_size() + rhs.get_byte_size());
                if order == Ordering::Less {
                    a.extend(*self);
                    a.extend(rhs);
                } else {
                    a.extend(rhs);
                    a.extend(*self);
                }
                a.set_normalized(true);
                return;
            }
        }

        let mut sum = ws.new_atom();
        let a = sum.to_add();
        a.extend(*self);
        a.extend(rhs);
        sum.as_view().normalize(ws, out);
    }

    /// Add normalized atoms and normalize the result using an n-way merge over their terms.
    pub(crate) fn add_normalized_slice<'a, T: AtomCore>(
        args: &[T],
        ws: &Workspace,
        out: &mut Atom,
    ) {
        if args.iter().any(|arg| arg.as_atom_view().has_alias()) {
            let mut e = ws.new_atom();
            let a = e.to_add();
            for arg in args {
                a.extend(arg.as_atom_view());
            }
            e.as_view().normalize(ws, out);
            return;
        }

        struct MergeCursor<'a> {
            atom: AtomView<'a>,
            term: &'a [u8],
            rest: Option<ListIterator<'a>>,
            index: usize,
        }

        impl Eq for MergeCursor<'_> {}

        impl PartialEq for MergeCursor<'_> {
            fn eq(&self, other: &Self) -> bool {
                self.index == other.index && self.term == other.term
            }
        }

        impl Ord for MergeCursor<'_> {
            fn cmp(&self, other: &Self) -> Ordering {
                other
                    .term
                    .cmp(self.term)
                    .then_with(|| other.index.cmp(&self.index))
            }
        }

        impl PartialOrd for MergeCursor<'_> {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        let mut heap = BinaryHeap::with_capacity(args.len());
        let mut byte_size = 0;

        for (index, arg) in args.iter().enumerate() {
            let view = arg.as_atom_view();
            byte_size += view.get_byte_size();

            match view {
                AtomView::Add(a) => {
                    let mut iter = a.iter();
                    if let Some(term) = iter.next() {
                        heap.push(MergeCursor {
                            atom: term,
                            term: term.get_term_cmp_slice(),
                            rest: Some(iter),
                            index,
                        });
                    }
                }
                AtomView::Num(n) if n.is_zero() => {}
                term => {
                    heap.push(MergeCursor {
                        atom: term,
                        term: term.get_term_cmp_slice(),
                        rest: None,
                        index,
                    });
                }
            }
        }

        let Some(mut cursor) = heap.pop() else {
            out.to_num(Coefficient::zero());
            return;
        };

        let out_add = out.to_add();
        out_add.grow_capacity(byte_size);

        let mut last_buf = ws.new_atom();
        last_buf.set_from_view(&cursor.atom);

        if let Some(rest) = cursor.rest.as_mut()
            && let Some(term) = rest.next()
        {
            cursor.atom = term;
            cursor.term = term.get_term_cmp_slice();
            heap.push(cursor);
        }

        let mut helper = ws.new_atom();
        let mut cur_len = 0;
        let mut second_pass = false;

        while let Some(mut cursor) = heap.pop() {
            let cur = cursor.atom;

            if !last_buf.merge_terms(cur, &mut helper) {
                let v = last_buf.as_view();
                if let AtomView::Num(n) = v {
                    let coeff = n.get_coeff_view();
                    if matches!(
                        coeff,
                        CoefficientView::Indeterminate | CoefficientView::Infinity(_)
                    ) {
                        out.set_from_view(&v);
                        return;
                    }

                    if !n.is_zero() {
                        out_add.extend(v);
                        cur_len += 1;
                    }
                } else {
                    out_add.extend(v);
                    cur_len += 1;
                }

                last_buf.set_from_view(&cur);
            } else if let AtomView::Add(_) = last_buf.as_view() {
                second_pass = true;
            }

            if let Some(rest) = cursor.rest.as_mut()
                && let Some(term) = rest.next()
            {
                cursor.atom = term;
                cursor.term = term.get_term_cmp_slice();
                heap.push(cursor);
            }
        }

        if cur_len == 0 {
            std::mem::swap(out, &mut last_buf);
        } else {
            if second_pass {
                out_add.extend(last_buf.as_view());
                out.as_view().normalize(ws, &mut helper);
                std::mem::swap(out, &mut helper);
                return;
            }

            let v = last_buf.as_view();
            if let AtomView::Num(n) = v {
                let coeff = n.get_coeff_view();
                if matches!(
                    coeff,
                    CoefficientView::Indeterminate | CoefficientView::Infinity(_)
                ) {
                    out.set_from_view(&v);
                    return;
                }

                if !n.is_zero() {
                    out_add.extend(v);
                    out_add.set_normalized(true);
                } else if cur_len == 1 {
                    last_buf.set_from_view(&out_add.to_add_view().to_slice().get(0));
                    std::mem::swap(out, &mut last_buf);
                } else {
                    out_add.set_normalized(true);
                }
            } else {
                out_add.extend(v);
                out_add.set_normalized(true);
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        atom::{Atom, AtomCore, AtomView},
        parse,
        printer::PrintOptions,
        state::Workspace,
    };

    // Build the reference one level at a time, so only the expression under test
    // has an entire chain of dirty children to normalize recursively.
    fn nested(kind: usize, depth: usize) -> (Atom, Atom) {
        let f = crate::symbol!("stack_function");
        let x = Atom::var(crate::symbol!("stack_x"));
        let y = Atom::var(crate::symbol!("stack_y"));
        let wrap = |inner: &Atom, level: usize| {
            let mut outer = Atom::new();
            match if kind == 4 { level % 4 } else { kind } {
                0 => {
                    outer.to_fun(f).add_arg(inner.as_view());
                }
                1 => {
                    let sum = outer.to_add();
                    sum.extend(inner.as_view());
                    sum.extend(x.as_view());
                }
                2 => {
                    let mul = outer.to_mul();
                    mul.extend(inner.as_view());
                    mul.extend(x.as_view());
                }
                3 => {
                    outer.to_pow(x.as_view(), inner.as_view());
                }
                _ => unreachable!(),
            }
            outer
        };
        let mut dirty = y.clone();
        let mut expected = y;
        Workspace::get_local().with(|ws| {
            for level in 0..depth {
                dirty = wrap(&dirty, level);
                wrap(&expected, level)
                    .as_view()
                    .normalize(ws, &mut expected);
            }
        });
        (dirty, expected)
    }

    #[test]
    fn normalize_with_two_mb_stack() {
        // Explicit size also exercises the regression when the test runner itself
        // supplies a larger stack (e.g. via RUST_MIN_STACK).
        for kind in 0..5 {
            let (dirty, expected) = nested(kind, 32);
            std::thread::Builder::new()
                .name(format!("normalize-kind-{kind}"))
                .stack_size(2_000_000)
                .spawn(move || {
                    Workspace::get_local().with(|ws| {
                        let mut out = Atom::new();
                        dirty.as_view().normalize(ws, &mut out);
                        assert_eq!(out, expected);
                        assert!(!out.as_view().needs_normalization());
                    });
                })
                .unwrap()
                .join()
                .unwrap();
        }
    }

    #[test]
    fn function_argument_copy_fast_path() {
        use crate::atom::{Fun, Mul, Symbol, representation::RawAtom};

        let reference = parse!("packed_normalize_fast(x,x^2)");
        let AtomView::Fun(reference_fun) = reference.as_view() else {
            unreachable!()
        };
        let head = reference_fun.get_symbol();
        let x = parse!("x");
        let square = parse!("x^2");
        let normalize = |f: &Fun| {
            Workspace::get_local().with(|ws| {
                let mut out = Atom::new();
                f.as_view().normalize(ws, &mut out);
                out
            })
        };

        let mut f = Fun::new_into(head, RawAtom::new());
        f.add_args(&[x.as_view(), square.as_view()]);
        assert_eq!(normalize(&f), reference);

        let mut raw_square = Mul::new();
        raw_square.extend(x.as_view());
        raw_square.extend(x.as_view());
        f.set_from_symbol(head);
        f.add_args(&[x.as_view(), raw_square.as_view()]);
        assert_eq!(normalize(&f), reference);

        let mut arg = Fun::new_into(Symbol::ARG, RawAtom::new());
        arg.add_args(&[x.as_view(), square.as_view()]);
        arg.set_normalized(true);
        f.set_from_symbol(head);
        f.add_arg(arg.as_view());
        assert_eq!(normalize(&f), reference);
    }

    #[test]
    fn cyclic_normalization_preserves_structural_order() {
        use crate::atom::{Fun, representation::RawAtom};

        let head = crate::symbol!("normalize_cycle"; Cyclesymmetric);
        for input in [
            "cycle_input()",
            "cycle_input(x)",
            "cycle_input(x,x,x)",
            "cycle_input(y,x,y,x)",
            "cycle_input(x,x,y,x,x,x,y)",
            "cycle_input(f(x),x^2)",
            "cycle_input(f(x),x^2,0,x,x*y,x+y)",
            "cycle_input(x+y,x*y,x,0,x^2,f(x))",
        ] {
            let input = parse!(input);
            let AtomView::Fun(input) = input.as_view() else {
                unreachable!()
            };
            let args: Vec<_> = input.iter().collect();
            // Exhaustively compare rotations with the original type-aware order.
            let expected = (0..args.len())
                .map(|shift| {
                    let mut rotated = args.clone();
                    rotated.rotate_left(shift);
                    rotated
                })
                .min_by(|left, right| {
                    left.iter()
                        .zip(right)
                        .map(|(a, b)| a.cmp(b))
                        .find(|order| !order.is_eq())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or_default();
            if args.len() == 2 {
                // Encoded bytes put functions before powers; normalization must not.
                assert!(matches!(expected[0], AtomView::Pow(_)));
                assert!(expected[0].get_data().cmp(expected[1].get_data()).is_gt());
            }
            Workspace::get_local().with(|ws| {
                for shift in 0..args.len().max(1) {
                    let mut f = Fun::new_into(head, RawAtom::new());
                    for arg in args[shift..].iter().chain(&args[..shift]) {
                        f.add_arg(*arg);
                    }
                    let mut out = Atom::new();
                    f.as_view().normalize(ws, &mut out);
                    let AtomView::Fun(normalized) = out.as_view() else {
                        unreachable!()
                    };
                    assert_eq!(normalized.iter().collect::<Vec<_>>(), expected);
                    assert!(!out.as_view().needs_normalization());
                }
            });
        }
    }

    #[test]
    fn variable_power_normalizes_exponent() {
        let x = parse!("stack_power_x");
        let one = Atom::num(1);
        Workspace::get_local().with(|ws| {
            for (n, expected) in [
                (0, Atom::num(1)),
                (1, x.clone()),
                (2, parse!("stack_power_x^2")),
            ] {
                let mut exp = Atom::new();
                let sum = exp.to_add();
                sum.extend(Atom::num(n - 1).as_view());
                sum.extend(one.as_view());
                let mut power = Atom::new();
                power.to_pow(x.as_view(), exp.as_view());
                let mut out = Atom::new();
                power.as_view().normalize(ws, &mut out);
                assert_eq!(out, expected);
                assert!(!out.as_view().needs_normalization());
            }
        });
    }

    #[test]
    fn power_checks_normalized_base() {
        use crate::atom::Symbol;
        let x = parse!("canonical_power_x");
        let e = Atom::var(Symbol::E);
        Workspace::get_local().with(|ws| {
            let mut base = Atom::new();
            let sum = base.to_add();
            sum.extend(Atom::num(2).as_view());
            sum.extend(Atom::num(-1).as_view());
            let mut power = Atom::new();
            power.to_pow(base.as_view(), x.as_view());
            let mut out = Atom::num(42);
            power.as_view().normalize(ws, &mut out);
            assert_eq!(out, Atom::num(1));

            let sum = base.to_add();
            sum.extend(e.as_view());
            sum.extend(Atom::num(0).as_view());
            let log = parse!("log(canonical_power_x)");
            power.to_pow(base.as_view(), log.as_view());
            power.as_view().normalize(ws, &mut out);
            assert_eq!(out, x);
            assert!(!out.as_view().needs_normalization());
        });
    }

    #[test]
    fn power_checks_normalized_log_argument() {
        use crate::atom::Symbol;
        let x = parse!("canonical_power_x");
        let e = Atom::var(Symbol::E);
        let mut argument = Atom::new();
        let sum = argument.to_add();
        sum.extend(x.as_view());
        sum.extend(Atom::num(0).as_view());
        let mut exponent = Atom::new();
        exponent.to_fun(Symbol::LOG).add_arg(argument.as_view());
        let mut power = Atom::new();
        power.to_pow(e.as_view(), exponent.as_view());
        Workspace::get_local().with(|ws| {
            let mut out = Atom::num(42);
            power.as_view().normalize(ws, &mut out);
            assert_eq!(out, x);
            assert!(!out.as_view().needs_normalization());
        });
    }

    #[test]
    fn power_retains_unsimplifiable_logs() {
        use crate::atom::Symbol;
        let e = Atom::var(Symbol::E);
        for exponent in [
            parse!("hold(log(canonical_power_x))"),
            parse!("log(canonical_power_x)*log(canonical_power_y)"),
        ] {
            let mut power = Atom::new();
            power.to_pow(e.as_view(), exponent.as_view());
            let mut expected = power.clone();
            expected.set_normalized(true);
            Workspace::get_local().with(|ws| {
                let mut out = Atom::num(42);
                power.as_view().normalize(ws, &mut out);
                assert_eq!(out, expected);
                assert!(!out.as_view().needs_normalization());
            });
        }
    }

    #[test]
    fn power_normalizes_both_operands() {
        let mut base = Atom::new();
        let sum = base.to_add();
        sum.extend(Atom::num(3).as_view());
        sum.extend(Atom::num(5).as_view());
        let mut exp = Atom::new();
        let sum = exp.to_add();
        sum.extend(Atom::num((1, 4)).as_view());
        sum.extend(Atom::num((1, 4)).as_view());
        let mut power = Atom::new();
        power.to_pow(base.as_view(), exp.as_view());
        Workspace::get_local().with(|ws| {
            let mut out = Atom::new();
            power.as_view().normalize(ws, &mut out);
            assert_eq!(out, parse!("2*2^(1/2)"));
            assert!(!out.as_view().needs_normalization());
        });
    }

    fn assert_same_expression_key(actual: Atom, expected: Atom) {
        use std::hash::{Hash, Hasher};

        assert!(!actual.as_view().needs_normalization());
        assert_eq!(actual, expected);
        let mut left = std::hash::DefaultHasher::new();
        let mut right = std::hash::DefaultHasher::new();
        actual.hash(&mut left);
        expected.hash(&mut right);
        assert_eq!(left.finish(), right.finish());
        assert_eq!(Ord::cmp(&actual, &expected), std::cmp::Ordering::Equal);
    }

    #[test]
    fn transparent_aliases_follow_function_attributes_and_argument_splicing() {
        use crate::atom::Symbol;

        let flat = crate::symbol!("transparent_function_flat"; Flat);
        let linear = crate::symbol!("transparent_function_linear"; Linear);
        let ordinary = crate::symbol!("transparent_function_ordinary");
        let x = parse!("transparent_function_x");
        let y = parse!("transparent_function_y");
        let z = parse!("transparent_function_z");
        let nested = crate::function!(flat, &x, &y);
        assert_same_expression_key(
            crate::function!(flat, nested.alias(false).alias(false), &z),
            crate::function!(flat, &x, &y, &z),
        );
        let sum = &x + &y;
        assert_same_expression_key(
            crate::function!(linear, sum.alias(false)),
            crate::function!(linear, &x) + crate::function!(linear, &y),
        );
        let product = &x * 3;
        assert_same_expression_key(
            crate::function!(linear, product.alias(false)),
            crate::function!(linear, &x) * 3,
        );
        assert_same_expression_key(
            crate::function!(linear, Atom::num(0).alias(false)),
            Atom::num(0),
        );
        let arguments = crate::function!(Symbol::ARG, &x, &y);
        assert_same_expression_key(
            crate::function!(ordinary, arguments.alias(false).alias(false), &z),
            crate::function!(ordinary, &x, &y, &z),
        );
    }

    #[test]
    fn transparent_aliases_follow_builtin_function_rules() {
        use crate::atom::Symbol;

        for (function, argument) in [
            (Symbol::SIN, Atom::num(0)),
            (Symbol::COS, Atom::num(0)),
            (Symbol::LOG, Atom::num(1)),
            (Symbol::ABS, Atom::num(-2)),
            (Symbol::EXP, Atom::num(0)),
            (Symbol::SQRT, Atom::num(4)),
            (Symbol::COEFF, Atom::num(3)),
        ] {
            assert_same_expression_key(
                crate::function!(function, argument.alias(false)),
                crate::function!(function, &argument),
            );
        }
        let x = parse!("transparent_if_x");
        let y = parse!("transparent_if_y");
        for condition in [Atom::num(0), Atom::num(1)] {
            assert_same_expression_key(
                crate::function!(Symbol::IF, condition.alias(false), &x, &y),
                crate::function!(Symbol::IF, &condition, &x, &y),
            );
        }
    }

    #[test]
    fn function_alias_normalization_preserves_opacity_and_ordinary_compression() {
        use crate::atom::Symbol;

        let ordinary = crate::symbol!("retained_alias_function");
        let body = parse!("retained_alias_body(x+y)");
        let alias = body.alias(false);
        let compressed = crate::function!(ordinary, &alias);
        let AtomView::Fun(function) = compressed.as_view() else {
            unreachable!()
        };
        assert!(matches!(
            function.iter().next().unwrap(),
            AtomView::Alias(_)
        ));
        assert_same_expression_key(compressed, crate::function!(ordinary, &body));

        let flat = crate::symbol!("opaque_alias_flat"; Flat);
        let linear = crate::symbol!("opaque_alias_linear"; Linear);
        let x = parse!("opaque_function_x");
        let opaque_flat = crate::function!(flat, &x).alias(true);
        let opaque_zero = Atom::num(0).alias(true);
        let opaque_arg = crate::function!(Symbol::ARG, &x, &x).alias(true);
        for expression in [
            crate::function!(flat, &opaque_flat),
            crate::function!(linear, &opaque_zero),
            crate::function!(Symbol::SIN, &opaque_zero),
            crate::function!(ordinary, &opaque_arg),
        ] {
            let AtomView::Fun(function) = expression.as_view() else {
                unreachable!()
            };
            assert_eq!(function.get_nargs(), 1);
            assert!(
                matches!(function.iter().next().unwrap(), AtomView::Alias(alias) if alias.is_opaque())
            );
            assert!(!expression.as_view().needs_normalization());
        }
    }

    #[test]
    fn ordinary_functions_normalize_dirty_alias_argument_bodies() {
        let head = crate::symbol!("dirty_argument_outer");
        let x = parse!("dirty_argument_x");
        let mut dirty = Atom::new();
        let product = dirty.to_mul();
        product.extend(x.as_view());
        product.extend(x.as_view());
        let alias = dirty.alias(false).alias(false);
        assert_same_expression_key(
            crate::function!(head, &alias),
            crate::function!(head, x.pow(2)),
        );
        let opaque = dirty.alias(true);
        let retained = crate::function!(head, &opaque);
        let AtomView::Fun(function) = retained.as_view() else {
            unreachable!()
        };
        assert_eq!(
            function.iter().next().unwrap().get_data(),
            opaque.as_view().get_data()
        );
    }

    #[test]
    fn function_alias_normalization_invokes_outer_callback_once() {
        use crate::atom::SymbolBuilder;
        use std::sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        };

        let calls = Arc::new(AtomicUsize::new(0));
        let callback_calls = calls.clone();
        let function = SymbolBuilder::new(crate::wrap_symbol!("alias_shape_callback"))
            .with_normalization_function(move |view, out| {
                callback_calls.fetch_add(1, Ordering::Relaxed);
                let AtomView::Fun(function) = view else {
                    unreachable!()
                };
                if let Some(argument @ AtomView::Num(_)) = function.iter().next() {
                    out.set_from_view(&argument);
                }
            })
            .build()
            .unwrap();
        let result = crate::function!(function, Atom::num(2).alias(false));
        assert_eq!(result, Atom::num(2));
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        let opaque = crate::function!(function, Atom::num(2).alias(true));
        assert!(matches!(opaque.as_view(), AtomView::Fun(_)));
        assert_eq!(calls.load(Ordering::Relaxed), 2);

        let x = parse!("alias_callback_x");
        let plain = crate::function!(function, &x);
        let before_comparison = calls.load(Ordering::Relaxed);
        let mut raw = Atom::new();
        raw.to_fun(function).add_arg(x.alias(false).as_view());
        raw.set_normalized(true);
        assert_same_expression_key(raw, plain);
        assert_eq!(calls.load(Ordering::Relaxed), before_comparison);
    }

    #[test]
    fn equivalent_nested_alias_sums_merge_in_arithmetic() {
        let terms = [
            parse!("nested_sum_f(x)"),
            parse!("nested_sum_g(x,y,z,w)"),
            parse!("nested_sum_h(x,y)"),
            parse!("nested_sum_a(x,y,z,w,v,u)"),
        ];
        let plain = Atom::add_many(&terms);
        let aliases: Vec<_> = terms.iter().map(|term| term.alias(false)).collect();
        let aliased = Atom::add_many(&aliases);
        assert_same_expression_key(aliased.clone(), plain.clone());
        let AtomView::Add(plain_sum) = plain.as_view() else {
            unreachable!()
        };
        let AtomView::Add(alias_sum) = aliased.as_view() else {
            unreachable!()
        };
        assert_eq!(
            alias_sum
                .iter()
                .map(|term| super::resolve_alias(term).get_data())
                .collect::<Vec<_>>(),
            plain_sum
                .iter()
                .map(|term| term.get_data())
                .collect::<Vec<_>>()
        );

        // Simulate a previously normalized sum with a different stored order.
        // Its semantic key is canonical, so enclosing expressions must also
        // compare equally when arithmetic searches for mergeable terms/factors.
        let mut legacy = Atom::new();
        let add = legacy.to_add();
        for term in alias_sum.iter().collect::<Vec<_>>().into_iter().rev() {
            add.extend(term);
        }
        legacy.set_normalized(true);
        assert_same_expression_key(legacy.clone(), plain.clone());
        let head = crate::symbol!("nested_sum_outer");
        let expected = crate::function!(head, &plain);
        for sum in [&aliased, &legacy] {
            let wrapped = crate::function!(head, sum);
            assert_same_expression_key(wrapped.clone(), expected.clone());
            assert_same_expression_key(&wrapped + &expected, &expected * 2);
            assert_same_expression_key(&expected + &wrapped, &expected * 2);
            assert_same_expression_key(&wrapped * &expected, expected.pow(2));
            assert_same_expression_key(&expected * &wrapped, expected.pow(2));
            assert_same_expression_key(sum * &plain, plain.pow(2));
        }

        let opaque = plain.alias(true);
        let opaque_sum = &opaque + &terms[0];
        let different = &plain + Atom::num(1);
        let values = [
            plain,
            aliased,
            legacy,
            opaque_sum,
            different,
            expected,
            Atom::num(2),
            parse!("nested_sum_x^2"),
        ];
        for a in &values {
            for b in &values {
                let order = Ord::cmp(a, b);
                assert_eq!(order, Ord::cmp(b, a).reverse());
                assert_eq!(order.is_eq(), a == b);
                assert_eq!(a.as_view().cmp(&b.as_view()).is_eq(), a == b);
                for c in &values {
                    if order.is_le() && Ord::cmp(b, c).is_le() {
                        assert!(Ord::cmp(a, c).is_le());
                    }
                }
            }
        }
    }

    #[test]
    fn numeric_aliases_follow_normal_sum_and_product_rules() {
        let x = parse!("numeric_alias_x");
        for n in [-1, 0, 1, 2] {
            let number = Atom::num(n);
            let alias = number.alias(false).alias(false);
            assert_same_expression_key(&alias * &x, &number * &x);
            assert_same_expression_key(&x * &alias, &x * &number);
            assert_same_expression_key(&alias + &x, &number + &x);
            assert_same_expression_key(&x + &alias, &x + &number);
            assert_same_expression_key(&alias * 3, &number * 3);
            assert_same_expression_key(&alias + 3, &number + 3);
            assert_same_expression_key(&alias + &alias, &number + &number);
        }
        let two = Atom::num(2).alias(false);
        let three = Atom::num(3).alias(false);
        assert_same_expression_key(&two * &x + &three * &x, &x * 5);
        let zero = Atom::num(0).alias(true);
        assert_ne!(&zero * &x, Atom::num(0));
        assert_ne!(&zero + &x, x);
    }

    #[test]
    fn power_aliases_follow_structural_rules_and_keep_safe_compression() {
        let x = parse!("power_alias_x");
        let y = parse!("power_alias_y");
        for body in [
            Atom::num(2),
            x.pow(2),
            &x * &y,
            &x + &y,
            parse!("power_alias_f(x)"),
        ] {
            let alias = body.alias(false);
            for exponent in [
                Atom::num(-1),
                Atom::num(0),
                Atom::num(1),
                Atom::num(2),
                Atom::num((1, 2)),
            ] {
                let exponent_alias = exponent.alias(false);
                assert_same_expression_key(alias.pow(&exponent), body.pow(&exponent));
                assert_same_expression_key(body.pow(&exponent_alias), body.pow(&exponent));
                assert_same_expression_key(alias.pow(&exponent_alias), body.pow(&exponent));
            }
        }
        let power = x.pow(2);
        assert_same_expression_key(power.alias(false) * &x, x.pow(3));
        assert_same_expression_key(&x * power.alias(false), x.pow(3));

        let large_base = parse!("power_alias_large(a,b,c,d,e,f,g)");
        let retained = large_base.alias(false).pow(2);
        assert!(retained.as_view().has_alias());
        assert_same_expression_key(retained, large_base.pow(2));
        let opaque = Atom::num(2).alias(true);
        assert_ne!(opaque.pow(2), Atom::num(4));
        assert_ne!(x.pow(&opaque), x.pow(2));
    }

    #[test]
    fn sums_and_products_normalize_dirty_direct_alias_bodies() {
        let x = parse!("dirty_direct_alias_x");
        let y = parse!("dirty_direct_alias_y");
        let head = crate::symbol!("dirty_direct_alias_f");
        let mut product = Atom::new();
        let mul = product.to_mul();
        mul.extend(x.as_view());
        mul.extend(x.as_view());
        let mut sum = Atom::new();
        let add = sum.to_add();
        add.extend(x.as_view());
        add.extend(x.as_view());
        let mut power = Atom::new();
        power.to_pow(x.as_view(), Atom::num(1).as_view());
        let mut function = Atom::new();
        function.to_fun(head).add_arg(product.as_view());

        Workspace::get_local().with(|ws| {
            for body in [product, sum, power, function] {
                assert!(body.as_view().needs_normalization());
                let mut canonical = Atom::new();
                body.as_view().normalize(ws, &mut canonical);
                let alias = body.alias(false).alias(false);
                assert_same_expression_key(&alias + &y, &canonical + &y);
                assert_same_expression_key(&y + &alias, &y + &canonical);
                assert_same_expression_key(&alias * &y, &canonical * &y);
                assert_same_expression_key(&y * &alias, &y * &canonical);
                let opaque = body.alias(true);
                for retained in [&opaque + &y, &opaque * &y] {
                    assert!(retained.as_view().has_alias());
                    assert!(!retained.as_view().needs_normalization());
                }
            }
        });

        // Canonical bodies that need no structural rule stay compressed.
        let body = crate::function!(head, &x);
        let alias = body.alias(false);
        for retained in [&alias + &y, &alias * &y] {
            assert!(retained.as_view().has_alias());
        }
        assert_same_expression_key(&alias + &y, &body + &y);
        assert_same_expression_key(&alias * &y, &body * &y);
    }

    #[test]
    fn power_aliases_normalize_dirty_operand_bodies() {
        let x = parse!("dirty_power_alias_x");
        let mut base = Atom::new();
        let product = base.to_mul();
        product.extend(x.as_view());
        product.extend(x.as_view());
        let mut exponent = Atom::new();
        let sum = exponent.to_add();
        sum.extend(Atom::num(1).as_view());
        sum.extend(Atom::num(1).as_view());
        assert_same_expression_key(base.alias(false).pow(exponent.alias(false)), x.pow(4));
    }

    #[test]
    fn product_normalization_borrows_nested_canonical_products() {
        let x = parse!("borrow_product_x");
        let y = parse!("borrow_product_y");
        let product = &x * &y;
        // Mul::extend normally flattens products before normalization. Lists
        // use the same header layout, so retain this nested product in a raw
        // dirty Mul to exercise normalization's canonical-parent borrowing.
        let mut list = crate::atom::Add::new();
        list.extend(product.as_view());
        list.extend(x.pow(-1).as_view());
        let mut data = list.into_raw();
        data[0] = (data[0] & !0b111) | 4; // Replace Add's kind with Mul's kind.
        let dirty = unsafe { Atom::from_raw(data) };
        let AtomView::Mul(outer) = dirty.as_view() else {
            unreachable!()
        };
        assert!(matches!(outer.iter().next(), Some(AtomView::Mul(_))));
        Workspace::get_local().with(|ws| {
            let mut out = Atom::num(42);
            dirty.as_view().normalize(ws, &mut out);
            assert_eq!(out, y);
            assert!(!out.as_view().needs_normalization());
        });
    }

    #[test]
    fn product_normalization_moves_finished_buffers() {
        let x = parse!("move_product_x");
        let y = parse!("move_product_y");
        let root_two = parse!("2^(1/2)");
        let root_product = parse!("(move_product_x*move_product_y)^(1/2)");
        let mut dirty_product = Atom::new();
        let mul = dirty_product.to_mul();
        mul.extend(x.as_view());
        mul.extend(y.as_view());
        let cases = [
            (Vec::new(), Atom::num(1)),
            (vec![x.clone()], x.clone()),
            (vec![x.clone(), Atom::num(1)], x.clone()),
            (vec![x.clone(), x.pow(-1)], Atom::num(1)),
            (vec![x.clone(), y.clone(), y.pow(-1)], x.clone()),
            (vec![Atom::num(0), x.clone()], Atom::num(0)),
            (vec![root_two.clone(), root_two.clone()], Atom::num(2)),
            (vec![x.clone(), root_two.clone(), root_two], &x * 2),
            (vec![root_product.clone(), root_product], &x * &y),
            (vec![&x * &y, x.pow(-1)], y.clone()),
            (vec![dirty_product, x.pow(-1)], y.clone()),
            (vec![x.clone(); 24], x.pow(24)),
        ];
        Workspace::get_local().with(|ws| {
            let mut out = parse!("move_old_output(a,b,c)");
            for (factors, expected) in cases {
                // Exercise both inline and heap factor storage, including
                // second passes and downgrades to a single surviving factor.
                for shift in 0..factors.len().max(1) {
                    let mut dirty = Atom::new();
                    let mul = dirty.to_mul();
                    for factor in factors[shift..].iter().chain(&factors[..shift]) {
                        mul.extend(factor.as_view());
                    }
                    dirty.as_view().normalize(ws, &mut out);
                    assert_eq!(out, expected);
                    assert!(!out.as_view().needs_normalization());
                }
            }
        });
    }

    #[test]
    fn product_normalization_moves_alias_ownership() {
        use std::sync::Arc;

        for opaque in [false, true] {
            // Recycled top-level Alias atoms retain their handles. A fresh
            // thread lets us check final release after its workspace is dropped.
            let weak = std::thread::spawn(move || {
                let handle = crate::alias::register_alias_atom(parse!("move_alias_body(x)"));
                let weak = Arc::downgrade(&handle);
                let alias = if opaque {
                    handle.to_opaque_atom()
                } else {
                    handle.to_atom()
                };
                let y = parse!("move_alias_y");
                let mut canonical = Atom::new();
                let mul = canonical.to_mul();
                mul.extend(y.as_view());
                mul.extend(alias.as_view());
                canonical.set_normalized(true);
                // The transparent parent alias prevents Mul::extend from flattening
                // the product before normalize_mul can borrow its canonical body.
                let parent_handle = crate::alias::register_alias_atom(canonical);
                let parent_weak = Arc::downgrade(&parent_handle);
                let parent = parent_handle.to_atom();
                let mut dirty = Atom::new();
                let mul = dirty.to_mul();
                mul.extend(parent.as_view());
                mul.extend(y.pow(-1).as_view());
                drop(parent);
                drop(parent_handle);
                drop(alias);
                drop(handle);

                Workspace::get_local().with(|ws| {
                    let old = crate::alias::register_alias_atom(parse!("move_alias_old(x)"));
                    let old_weak = Arc::downgrade(&old);
                    let mut out =
                        crate::function!(crate::symbol!("move_alias_parent"), old.to_atom());
                    drop(old);
                    dirty.as_view().normalize(ws, &mut out);
                    drop(dirty);
                    assert!(parent_weak.upgrade().is_none());
                    assert!(old_weak.upgrade().is_none());
                    assert!(weak.upgrade().is_some());
                    assert!(out.as_view().has_alias());
                    assert!(!out.as_view().needs_normalization());
                    let AtomView::Alias(result) = out.as_view() else {
                        panic!("the surviving child alias should remain an alias");
                    };
                    assert_eq!(result.is_opaque(), opaque);
                    assert_eq!(result.get_body(), parse!("move_alias_body(x)").as_view());
                    let owners = weak.strong_count();
                    drop(out);
                    assert_eq!(weak.strong_count(), owners - 1);
                });
                weak
            })
            .join()
            .unwrap();
            assert!(weak.upgrade().is_none());
        }
    }

    #[test]
    fn pow_apart() {
        let res = parse!("v1*(v1*v2*v3)^-5");
        let refr = parse!("v1^-4*v2^-5*v3^-5");
        assert_eq!(res, refr);
    }

    #[test]
    fn pow_simplify() {
        assert_eq!(parse!("1^(1/2)"), parse!("1"));
        assert_eq!(
            format!("{}", parse!("(v1^2)^v2").printer(PrintOptions::file())),
            "(symbolica::v1^2)^symbolica::v2"
        );
        assert_eq!(parse!("(v1^v2)^2"), parse!("v1^(2*v2)"));
        assert_eq!(parse!("(v1^(1/2))^2"), parse!("v1"));
    }

    #[test]
    fn merge_nested_terms_and_factors_with_and_without_aliases() {
        let body = parse!("normalization_fast_f(x)*(y+z)^2");
        let x = parse!("x");
        let x_alias = x.alias(false);
        let aliased = body.replace_map(|atom, _, out| {
            if atom == x.as_view() {
                out.set_from_view(&x_alias.as_view());
            }
        });
        assert!(aliased.as_view().has_alias());

        for left in [&body, &aliased] {
            for right in [&body, &aliased] {
                let sum = left * 2 + right * 3;
                assert_eq!(sum.inline_aliases(false), &body * 5);

                let product = left * right;
                assert_eq!(product.inline_aliases(false), body.pow(2));
            }
        }
    }

    #[test]
    fn merge_aliases_of_unnormalized_bodies() {
        let x = parse!("raw_alias_merge_x");
        let y = parse!("raw_alias_merge_y");
        let two = Atom::num(2);

        let mut raw_power = Atom::new();
        raw_power.to_pow(x.as_view(), two.as_view());
        let power = x.pow(2);

        let mut raw_product = Atom::new();
        let mul = raw_product.to_mul();
        mul.extend(power.as_view());
        mul.extend(y.as_view());
        let product = &power * &y;

        let mut raw_nested_product = Atom::new();
        let mul = raw_nested_product.to_mul();
        mul.extend(raw_power.as_view());
        mul.extend(y.as_view());

        for (raw, normalized) in [
            (&raw_power, &power),
            (&raw_product, &product),
            (&raw_nested_product, &product),
        ] {
            assert!(raw.as_view().needs_normalization());
            let alias = raw.alias(false);
            for (left, right) in [(normalized, &alias), (&alias, normalized)] {
                let sum = left + right;
                assert_eq!(sum.nterms(), 1);
                assert!(sum.as_view().cmp(&((normalized * 2).as_view())).is_eq());
            }
        }
    }

    #[test]
    fn merging_raw_aliases_keeps_followup_operations_canonical() {
        let x = parse!("raw_alias_followup_x");
        let two = Atom::num(2);
        let y = parse!("raw_alias_followup_y");
        let mut raw_power = Atom::new();
        raw_power.to_pow(x.as_view(), two.as_view());
        let power = x.pow(2);
        let f = crate::symbol!("raw_alias_followup_f");
        let mut raw_function = Atom::new();
        raw_function.to_fun(f).add_arg(raw_power.as_view());
        let mut raw_product = Atom::new();
        raw_product.to_mul().extend(raw_power.as_view());
        if let Atom::Mul(m) = &mut raw_product {
            m.extend(y.as_view());
        }

        for (raw, canonical) in [
            (raw_power, power.clone()),
            (raw_function, crate::function!(f, &power)),
            (raw_product, &power * &y),
        ] {
            let alias = raw.alias(false);
            let expected = &canonical * 2;
            for (mut merged, other) in [
                (canonical.clone(), alias.as_view()),
                (alias.clone(), canonical.as_view()),
            ] {
                let mut helper = Atom::new();
                assert!(merged.merge_terms(other, &mut helper));
                assert_eq!(merged, expected);
                let left = crate::function!(f, merged);
                let right = crate::function!(f, &expected);
                assert_eq!(&left * &right, right.pow(2));
            }

            let mut unchanged = alias.clone();
            assert!(!unchanged.merge_terms(two.as_view(), &mut Atom::new()));
            assert_eq!(unchanged, alias);
            let opaque = raw.alias(true);
            let mut unchanged = canonical.clone();
            assert!(!unchanged.merge_terms(opaque.as_view(), &mut Atom::new()));
            assert_eq!(unchanged, canonical);

            // A normalized function can still contain a raw aliased argument.
            // Retaining that alias must retain the semantic comparison path.
            let nested = crate::function!(f, alias);
            let canonical_nested = crate::function!(f, canonical);
            assert_eq!(
                (&nested * &canonical_nested).inline_aliases(false),
                canonical_nested.pow(2)
            );
        }
    }

    #[test]
    fn merging_aliased_products_preserves_factors_and_normalizes_raw_bodies() {
        let x = parse!("raw_alias_coefficient_x");
        let y = parse!("raw_alias_coefficient_y");
        let two = Atom::num(2);
        let mut raw_power = Atom::new();
        raw_power.to_pow(x.as_view(), two.as_view());
        let mut raw_product = Atom::new();
        let m = raw_product.to_mul();
        m.extend(two.as_view());
        m.extend(raw_power.as_view());
        m.extend(y.as_view());
        let term = x.pow(2) * &y;
        let canonical_product = &term * 2;

        for body in [raw_product, canonical_product] {
            let alias = body.alias(false);
            for coefficient in [-2, -1, 1, 3] {
                let other = &term * coefficient;
                let expected = &term * (coefficient + 2);
                for (mut merged, other) in [
                    (other.clone(), alias.as_view()),
                    (alias.clone(), other.as_view()),
                ] {
                    assert!(merged.merge_terms(other, &mut Atom::new()));
                    assert_eq!(merged, expected);
                }
            }
        }

        // This also exercises the resolved-Mul / ordinary-factor branch.
        let alias = (&x * 2).alias(false);
        for (mut merged, other) in [(alias.clone(), x.as_view()), (x.clone(), alias.as_view())] {
            assert!(merged.merge_terms(other, &mut Atom::new()));
            assert_eq!(merged, &x * 3);
        }
    }

    #[test]
    fn linear_symmetric() {
        let res = parse!("fsl1(v2+2*v3,v1+3*v2-v3)");
        let refr = parse!("fsl1(v1,v2)+2*fsl1(v1,v3)+3*fsl1(v2,v2)+5*fsl1(v2,v3)-2*fsl1(v3,v3)");
        assert_eq!(res, refr);
    }

    #[test]
    fn coeff_flag() {
        let a = parse!("-v1*v2");

        if let Atom::Mul(m) = &a {
            assert!(m.to_mul_view().has_coefficient());
            assert!(m.to_mul_view().is_normalized());
        }

        let b = a * &parse!("-v3");

        if let Atom::Mul(m) = &b {
            assert!(!m.to_mul_view().has_coefficient());
            assert!(m.to_mul_view().is_normalized());
        } else {
            panic!("Expected a Mul");
        }

        let c = &b * 2;

        if let Atom::Mul(m) = &c {
            assert!(m.to_mul_view().has_coefficient());
            assert!(m.to_mul_view().is_normalized());
        } else {
            panic!("Expected a Mul");
        }
    }

    #[test]
    fn add_normalized() {
        let a = parse!("v1 + v2 + v3");
        let b = parse!("1 + v2 + v4 + v5");
        assert_eq!(a + b, parse!("v1+2*v2+v3+v4+v5+1"));

        let a = parse!("v1 + v2 + v3");
        let b = parse!("v4");
        assert_eq!(a + b, parse!("v1+v2+v3+v4"));

        let a = parse!("v2 + v3 + v4");
        let b = parse!("v1");
        assert_eq!(a + b, parse!("v1+v2+v3+v4"));

        let half_sum = parse!("(v1 + v2) / 2");
        assert_eq!(parse!("v1") + &half_sum + &half_sum, parse!("2*v1+v2"));

        let a = parse!("v1 + (v1 + v2) / 2");
        let b = parse!("v2 + (v1 + v2) / 2");
        assert_eq!(a + b, parse!("2*v1+2*v2"));
    }

    #[test]
    fn pairwise_add_matches_general_normalization() {
        use crate::{
            coefficient::Coefficient,
            domains::finite_field::{FiniteFieldCore, Zp64},
        };

        let x = parse!("pairwise_add_x");
        let y = parse!("pairwise_add_y");
        let field = Zp64::new(17);
        let finite = |n| {
            Atom::num(Coefficient::from_finite_field(
                field.clone(),
                field.to_element(n),
            ))
        };
        let infinity = Atom::num(Coefficient::positive_infinity());
        let negative_infinity = Atom::num(-Coefficient::positive_infinity());
        let mut cases = vec![
            (&x * &y, 2 * x.pow(2) * y.pow(2)),
            (x.clone(), &x * 2),
            (&x * 2, &x * -2),
            (Atom::num(0), x.clone()),
            (Atom::num(0), Atom::num(0)),
            (Atom::num(2), Atom::num(3)),
            (Atom::num((1, 2)), Atom::num((1, 3))),
            (Atom::num(2).to_float(20), Atom::num(3).to_float(30)),
            (Atom::num(2).to_float(20), Atom::num(-2).to_float(20)),
            (Atom::num(0).to_float(20), Atom::num(3)),
            (finite(2), finite(3)),
            (finite(2), finite(15)),
            (finite(0), Atom::num(3)),
            (finite(3), Atom::num(0)),
            (&x * finite(2), &x * finite(15)),
            (infinity.clone(), negative_infinity.clone()),
            (&x * &infinity, &x * &negative_infinity),
            ((&x + &y) / 3, 2 * (&x + &y) / 3),
            (x.pow(3), &x * &y),
            (
                crate::function!(crate::symbol!("pairwise_add_f"), &x),
                crate::function!(crate::symbol!("pairwise_add_f"), &y),
            ),
            (crate::function!(crate::symbol!("pairwise_add_f"), &x), y),
        ];
        for number in [
            Atom::num(Coefficient::Indeterminate),
            infinity,
            negative_infinity,
            Atom::num(Coefficient::complex_infinity()),
        ] {
            cases.push((number.clone(), x.clone()));
            cases.push((number, Atom::num(3)));
        }

        Workspace::get_local().with(|ws| {
            for (left, right) in cases {
                for (left, right) in [(&left, &right), (&right, &left)] {
                    let mut sum = Atom::new();
                    let a = sum.to_add();
                    a.extend(left.as_view());
                    a.extend(right.as_view());
                    let mut expected = Atom::new();
                    sum.as_view().normalize(ws, &mut expected);
                    let mut actual = Atom::new();
                    left.as_view()
                        .add_normalized(right.as_view(), ws, &mut actual);
                    assert_eq!(actual, expected, "adding {left} and {right}");
                    assert!(!actual.as_view().needs_normalization());
                }
            }
        });

        // Keep the preexisting identical-input rule, which retains a factored sum.
        let sum = parse!("pairwise_add_x+pairwise_add_y");
        assert_eq!(&sum + &sum, &sum * 2);
    }

    #[test]
    fn pairwise_add_preserves_dirty_and_aliased_inputs() {
        let x = parse!("pairwise_dirty_x");
        let y = parse!("pairwise_dirty_y");
        let two = Atom::num(2);
        let mut raw_power = Atom::new();
        raw_power.to_pow(x.as_view(), two.as_view());
        let canonical_power = x.pow(2);
        let mut raw_function = Atom::new();
        raw_function
            .to_fun(crate::symbol!("pairwise_dirty_f"))
            .add_arg(raw_power.as_view());
        let canonical_function =
            crate::function!(crate::symbol!("pairwise_dirty_f"), &canonical_power);
        let alias = raw_power.alias(false);
        let opaque = raw_power.alias(true);
        let unrelated_handle =
            crate::alias::register_alias_atom(parse!("pairwise_unrelated_alias"));
        let cases = [
            (raw_power.clone(), y.clone()),
            (raw_power, canonical_power.clone()),
            (raw_function, canonical_function),
            (alias, canonical_power.clone()),
            (opaque, canonical_power),
            (x, y),
        ];
        Workspace::get_local().with(|ws| {
            for (left, right) in cases {
                for (left, right) in [(&left, &right), (&right, &left)] {
                    let mut sum = Atom::new();
                    let a = sum.to_add();
                    a.extend(left.as_view());
                    a.extend(right.as_view());
                    let mut expected = Atom::new();
                    sum.as_view().normalize(ws, &mut expected);

                    let handle =
                        crate::alias::register_alias_atom(parse!("pairwise_reused_output"));
                    let token = handle.token();
                    let mut actual = handle.to_atom();
                    drop(handle);
                    left.as_view()
                        .add_normalized(right.as_view(), ws, &mut actual);
                    assert_eq!(actual, expected, "adding {left} and {right}");
                    assert!(!actual.as_view().needs_normalization());
                    assert!(crate::alias::get_alias(token).is_none());
                }
            }
        });
        assert!(crate::alias::get_alias(unrelated_handle.token()).is_some());
    }

    #[test]
    fn add_normalized_slice() {
        let args = [
            parse!("v1 + v3 + 1"),
            parse!("v2 + v3 + 2"),
            parse!("-v1 + v4 - 3"),
            parse!("v3"),
            parse!("0"),
        ];
        let views: Vec<_> = args.iter().map(|x| x.as_view()).collect();

        let mut out = Atom::new();
        Workspace::get_local().with(|ws| AtomView::add_normalized_slice(&views, ws, &mut out));

        assert_eq!(out, parse!("v2 + 3*v3 + v4"));

        let args = [
            parse!("v1"),
            parse!("(v1 + v2) / 2"),
            parse!("(v1 + v2) / 2"),
        ];
        let views: Vec<_> = args.iter().map(|x| x.as_view()).collect();
        Workspace::get_local().with(|ws| AtomView::add_normalized_slice(&views, ws, &mut out));

        assert_eq!(out, parse!("2*v1 + v2"));
    }
}
