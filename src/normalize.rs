use std::{cmp::Ordering, ops::DerefMut};

use smallvec::SmallVec;

use crate::{
    atom::{Atom, AtomCore, AtomView, Fun, Symbol, representation::InlineNum},
    coefficient::{Coefficient, CoefficientView},
    domains::{
        float::{Complex, FloatLike, Real},
        integer::Z,
        rational::{Q, Rational},
    },
    poly::PolyVariable,
    state::{RecycledAtom, State, Workspace},
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
        } else {
            self.get_data()
        }
    }
}

impl AtomView<'_> {
    /// Compare two atoms in an ordering that sorts by type.
    ///
    /// This is not the ordering used for sorting terms or factors.
    pub fn cmp(&self, other: &AtomView<'_>) -> Ordering {
        if self == other {
            // equality comparison is a fast check
            return Ordering::Equal;
        }

        match (&self, other) {
            // comparisons on floating points may yield Equal even if the binary data is not the same
            // for example when the precision is different. To stay consistent with the PartialEq implementation,
            // tie break on binary data comparison
            (AtomView::Num(n1), AtomView::Num(n2)) => n1
                .get_coeff_view()
                .cmp(&n2.get_coeff_view())
                .then_with(|| self.get_data().cmp(other.get_data())),
            (AtomView::Num(_), _) => Ordering::Less,
            (_, AtomView::Num(_)) => Ordering::Greater,
            (AtomView::Var(v1), AtomView::Var(v2)) => v1.get_symbol_id().cmp(&v2.get_symbol_id()),
            (AtomView::Var(_), _) => Ordering::Less,
            (_, AtomView::Var(_)) => Ordering::Greater,
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

                unreachable!(
                    "Expression is equal but not binary equal: {};\n{:?}\n{:?}",
                    self, m1, m2
                );
            }
            (AtomView::Mul(_), _) => Ordering::Less,
            (_, AtomView::Mul(_)) => Ordering::Greater,
            (AtomView::Add(a1), AtomView::Add(a2)) => {
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

                unreachable!(
                    "Expression is equal but not binary equal: {};\n{:?}\n{:?}",
                    self, a1, a2
                );
            }
            (AtomView::Add(_), _) => Ordering::Less,
            (_, AtomView::Add(_)) => Ordering::Greater,

            (AtomView::Fun(f1), AtomView::Fun(f2)) => {
                let name_comp = f1.get_symbol_id().cmp(&f2.get_symbol_id());
                if name_comp != Ordering::Equal {
                    return name_comp;
                }

                if cfg!(feature = "full_fn_cmp") {
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

                    unreachable!(
                        "Expression is equal but not binary equal: {};\n{:?}\n{:?}",
                        self, f1, f2
                    );
                } else {
                    f1.fast_cmp(*f2)
                }
            }
        }
    }

    /// Compare factors in a term. `x` and `x^2` are placed next to each other by sorting a power based on the base only.
    pub(crate) fn cmp_factors(&self, other: &AtomView<'_>) -> Ordering {
        match (&self, other) {
            (AtomView::Num(_), AtomView::Num(_)) => Ordering::Equal,
            (AtomView::Num(_), _) => Ordering::Less,
            (_, AtomView::Num(_)) => Ordering::Greater,

            (AtomView::Var(v1), AtomView::Var(v2)) => v1.get_symbol_id().cmp(&v2.get_symbol_id()),
            (AtomView::Pow(p1), AtomView::Pow(p2)) => {
                // TODO: inline partial_cmp call by creating an inlined version
                p1.get_base().cmp(&p2.get_base())
            }
            (_, AtomView::Pow(p2)) => {
                let base = p2.get_base();
                self.cmp(&base).then(Ordering::Less) // sort x^2*x -> x*x^2
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

                if cfg!(feature = "full_fn_cmp") {
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
                } else {
                    f1.fast_cmp(*f2)
                }
            }
        }
    }

    /// Compare terms in an expression. `x` and `2*x` are placed next to each other.
    #[inline]
    pub(crate) fn cmp_terms(&self, other: &AtomView<'_>) -> Ordering {
        debug_assert!(!matches!(self, AtomView::Add(_)));
        debug_assert!(!matches!(other, AtomView::Add(_)));
        self.get_term_cmp_slice().cmp(other.get_term_cmp_slice())
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

                if base1 != base2 {
                    return false;
                }

                if let AtomView::Num(n) = &exp1
                    && let AtomView::Num(n2) = &exp2
                {
                    let new_exp = helper.to_num(n.get_coeff_view() + n2.get_coeff_view());

                    if new_exp.to_num_view().is_zero() {
                        self.to_num(1.into());
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
                        self.to_num(1.into());
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

            if !matches!(other, Atom::Num(_)) && other.as_view() == base {
                if let AtomView::Num(n) = &exp {
                    let new_exp = n.get_coeff_view() + 1;

                    if new_exp.is_zero() {
                        self.to_num(1.into());
                    } else if new_exp == 1.into() {
                        self.set_from_view(&other.as_view());
                    } else {
                        let num = helper.to_num(new_exp);
                        self.to_pow(other.as_view(), AtomView::Num(num.to_num_view()));
                    }
                } else {
                    other.to_num(1.into());

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

            if !matches!(self, Atom::Num(_)) && self.as_view() == base {
                if let AtomView::Num(n) = &exp {
                    let new_exp = n.get_coeff_view() + 1;

                    if new_exp.is_zero() {
                        self.to_num(1.into());
                    } else if new_exp == 1.into() {
                        self.set_from_view(&base);
                    } else {
                        let num = helper.to_num(new_exp);
                        self.to_pow(base, AtomView::Num(num.to_num_view()));
                    }
                } else {
                    self.to_num(1.into());

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
        if self.as_view() == other.as_view() {
            // add powers
            let exp = other.to_num(2.into());
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
        if let Atom::Num(n1) = self {
            if let AtomView::Num(n2) = other {
                n1.add(&n2);
                return true;
            } else {
                return false;
            }
        }

        // compare the non-coefficient part of terms and add the coefficients if they are the same
        if let Atom::Mul(m) = self {
            let mv = m.to_mul_view();
            let mut slice = m.to_mul_view().to_slice();

            let has_coeff = mv.has_coefficient();
            let mut coeff = other; // stub, should not be of type num
            if has_coeff {
                (coeff, slice) = slice.pop_first();
            }

            if let AtomView::Mul(m2) = other {
                let mut slice2 = m2.to_slice();

                let mut coeff2 = other; // stub, should not be of type num
                if m2.has_coefficient() {
                    (coeff2, slice2) = slice2.pop_first();
                }

                if slice.eq(&slice2) {
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

                    if has_coeff {
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
                if slice.len() != 1 || other != slice.get(0) {
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

                m.replace_first(on.to_num_view().as_view());
                m.set_normalized(true);

                return true;
            }
        } else if let AtomView::Mul(m) = other {
            let slice = m.to_slice();

            if slice.len() != 2 {
                return false; // no match
            }

            let (coeff, slice) = slice.pop_first();
            let (term, _) = slice.pop_first();

            if self.as_view() == term {
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

                other.clone_into(self);

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
        } else if self.as_view() == other {
            let mul = helper.to_mul();
            self.to_num((2, 1).into());
            mul.extend(self.as_view());
            mul.extend(other);
            mul.set_has_coefficient(true);
            mul.set_normalized(true);

            std::mem::swap(self, helper);
            return true;
        };

        false
    }
}

impl AtomView<'_> {
    #[inline(always)]
    pub fn needs_normalization(&self) -> bool {
        match self {
            AtomView::Num(_) | AtomView::Var(_) => false,
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

        match self {
            AtomView::Mul(t) => {
                let mut atom_test_buf: SmallVec<[_; 20]> = SmallVec::new();

                for a in t.iter() {
                    let mut handle = workspace.new_atom();

                    if a.needs_normalization() {
                        a.normalize(workspace, &mut handle);
                    } else {
                        handle.set_from_view(&a);
                    }

                    if let Atom::Mul(mul) = handle.deref_mut() {
                        for c in mul.to_mul_view().iter() {
                            // TODO: remove this copy
                            let mut handle = workspace.new_atom();
                            handle.set_from_view(&c);

                            if let AtomView::Num(n) = c
                                && n.is_one()
                            {
                                continue;
                            }

                            atom_test_buf.push(handle);
                        }
                    } else {
                        if let AtomView::Num(n) = handle.as_view()
                            && n.is_one()
                        {
                            continue;
                        }

                        atom_test_buf.push(handle);
                    }
                }

                atom_test_buf.sort_by(|a, b| a.as_view().cmp_factors(&b.as_view()));

                let mut second_pass = false;
                if !atom_test_buf.is_empty() {
                    let out_mul = out.to_mul();

                    atom_test_buf.reverse();
                    let mut last_buf = atom_test_buf.pop().unwrap();

                    let mut tmp = workspace.new_atom();
                    let mut cur_len = 0;

                    while let Some(mut cur_buf) = atom_test_buf.pop() {
                        if !last_buf.merge_factors(&mut cur_buf, &mut tmp, workspace) {
                            // we are done merging
                            {
                                let v = last_buf.as_view();
                                if let AtomView::Num(n) = v {
                                    if matches!(
                                        n.get_coeff_view(),
                                        CoefficientView::Indeterminate
                                            | CoefficientView::Infinity(None)
                                    ) {
                                        out.set_from_view(&v);
                                        return;
                                    }

                                    if n.is_zero() {
                                        out.set_from_view(&v);
                                        return;
                                    }

                                    if !n.is_one() {
                                        out_mul.set_has_coefficient(true);
                                        out_mul.extend(v);
                                        cur_len += 1;
                                    } else {
                                        out_mul.set_has_coefficient(false);
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
                        out.set_from_view(&last_buf.as_view());
                    } else {
                        if second_pass {
                            out_mul.extend(last_buf.as_view());
                            out.as_view().normalize(workspace, &mut tmp);
                            out.set_from_view(&tmp.as_view());
                            return;
                        }

                        // TODO: should not be reached anymore!
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
                                out_mul.set_normalized(true);
                            } else if cur_len == 1 {
                                // downgrade
                                last_buf.set_from_view(&out_mul.to_mul_view().to_slice().get(0));
                                out.set_from_view(&last_buf.as_view());
                            } else {
                                out_mul.set_normalized(true);
                            }
                        } else {
                            out_mul.extend(v);
                            out_mul.set_normalized(true);
                        }
                    }
                } else {
                    out.to_num(1.into());
                }
            }
            AtomView::Num(n) => {
                let normalized_num = n.get_coeff_view().normalize();
                out.to_num(normalized_num);
            }
            AtomView::Var(_) => {
                self.clone_into(out);
            }
            AtomView::Fun(f) => {
                let id = f.get_symbol();
                let out_f = out.to_fun(id);

                /// Add an argument `a` to `f` and flatten nested `arg`s.
                #[inline(always)]
                fn add_arg(f: &mut Fun, a: AtomView) {
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
                            add_arg(f, *a);
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

                let mut handle = workspace.new_atom();
                for a in f {
                    if a.needs_normalization() {
                        a.normalize(workspace, &mut handle);
                        add_arg(out_f, handle.as_view());
                    } else {
                        add_arg(out_f, a);
                    }
                }

                out_f.set_normalized(true);

                if [
                    Symbol::COS_ID,
                    Symbol::SIN_ID,
                    Symbol::EXP_ID,
                    Symbol::LOG_ID,
                    Symbol::SQRT_ID,
                    Symbol::ABS_ID,
                ]
                .contains(&id.get_id())
                    && out_f.to_fun_view().get_nargs() == 1
                {
                    let arg = out_f.to_fun_view().iter().next().unwrap();
                    if let AtomView::Num(n) = arg {
                        if n.is_zero() && id != Symbol::LOG || n.is_one() && id == Symbol::LOG {
                            if id == Symbol::COS || id == Symbol::EXP {
                                out.to_num(Coefficient::one());
                                return;
                            } else if id == Symbol::SIN
                                || id == Symbol::LOG
                                || id == Symbol::SQRT
                                || id == Symbol::ABS
                            {
                                out.to_num(Coefficient::zero());
                                return;
                            }
                        }

                        if id == Symbol::SQRT && n.is_one() {
                            out.to_num(Coefficient::one());
                            return;
                        }

                        if n.is_zero() && id == Symbol::LOG {
                            warn!("Created infinity by log(0)");
                            out.to_num(Coefficient::Infinity(Some(Rational::new(-1, 1).into())));
                            return;
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
                                    return;
                                }
                                Symbol::SIN_ID => {
                                    let r = if i.is_zero() {
                                        r.to_float().sin().into()
                                    } else {
                                        Complex::new(r.to_float(), i.to_float()).sin()
                                    };
                                    out.to_num(Coefficient::Float(r));
                                    return;
                                }
                                Symbol::EXP_ID => {
                                    let r = if i.is_zero() {
                                        r.to_float().exp().into()
                                    } else {
                                        Complex::new(r.to_float(), i.to_float()).exp()
                                    };
                                    out.to_num(Coefficient::Float(r));
                                    return;
                                }
                                Symbol::LOG_ID => {
                                    let r = if i.is_zero() {
                                        r.to_float().log().into()
                                    } else {
                                        Complex::new(r.to_float(), i.to_float()).log()
                                    };
                                    out.to_num(Coefficient::Float(r));
                                    return;
                                }
                                Symbol::SQRT_ID => {
                                    let r = if i.is_zero() {
                                        let r = r.to_float();
                                        Complex::new(r.sqrt(), r.zero())
                                    } else {
                                        Complex::new(r.to_float(), i.to_float()).sqrt()
                                    };
                                    out.to_num(Coefficient::Float(r));
                                    return;
                                }
                                Symbol::ABS_ID => {
                                    let r = if i.is_zero() {
                                        r.to_float().norm().into()
                                    } else {
                                        Complex::new(r.to_float(), i.to_float()).norm().into()
                                    };
                                    out.to_num(Coefficient::Float(r));
                                    return;
                                }
                                _ => {}
                            }
                        }

                        if id.get_id() == Symbol::ABS_ID {
                            if let Coefficient::Complex(c) = n.get_coeff_view().to_owned() {
                                if c.is_real() {
                                    out.to_num(c.re.abs().into());
                                } else {
                                    let r = c.norm_squared();
                                    let mut buffer = workspace.new_atom();
                                    buffer.to_num(r.into());
                                    *out = buffer.pow((1, 2));
                                }
                                return;
                            }
                        }
                    } else if id.get_id() == Symbol::ABS_ID && arg.is_positive() {
                        let mut buffer = workspace.new_atom();
                        buffer.set_from_view(&arg);
                        out.set_from_view(&buffer.as_view());
                        return;
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
                        return;
                    }
                }

                if id == Symbol::CONJ && out_f.to_fun_view().get_nargs() == 1 {
                    let arg = out_f.to_fun_view().iter().next().unwrap();

                    match arg {
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
                            } else if s == Symbol::EXP && ff.get_nargs() == 1 {
                                // conj(exp(a)) = exp(conj(a))
                                let exp_arg = ff.iter().next().unwrap();
                                let mut conj_fun = workspace.new_atom();
                                conj_fun.to_fun(Symbol::CONJ).add_arg(exp_arg);

                                let mut new_exp = workspace.new_atom();
                                new_exp.to_fun(Symbol::EXP).add_arg(conj_fun.as_view());
                                new_exp.as_view().normalize(workspace, out);
                            }
                        }
                        AtomView::Pow(p) => {
                            if arg.is_real() {
                                let mut inner = workspace.new_atom();
                                inner.set_from_view(&arg);
                                out.set_from_view(&inner.as_view());
                            } else {
                                let (b, e) = p.get_base_exp();
                                if e.is_integer() {
                                    let mut new_base = workspace.new_atom();
                                    let nb = new_base.to_fun(Symbol::CONJ);
                                    nb.add_arg(b);
                                    let mut new_pow = workspace.new_atom();
                                    new_pow.to_pow(new_base.as_view(), e);
                                    new_pow.as_view().normalize(workspace, out);
                                } else if b.is_positive() {
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

                    return;
                }

                // simplify log(exp(real)) = real
                if id == Symbol::LOG && out_f.to_fun_view().get_nargs() == 1 {
                    let arg = out_f.to_fun_view().iter().next().unwrap();

                    if let AtomView::Fun(f2) = arg
                        && f2.get_symbol() == Symbol::EXP
                        && f2.get_nargs() == 1
                    {
                        let exp_arg = f2.iter().next().unwrap();
                        if exp_arg.is_real() {
                            let mut buffer = workspace.new_atom();
                            buffer.set_from_view(&exp_arg);
                            out.set_from_view(&buffer.as_view());
                            return;
                        }
                    }
                }

                if id == Symbol::EXP && out_f.to_fun_view().get_nargs() == 1 {
                    let arg = out_f.to_fun_view().iter().next().unwrap();
                    // simplify logs inside exp
                    if arg.contains_symbol(Symbol::LOG) {
                        let mut buffer = workspace.new_atom();
                        if arg.simplify_exp_log(workspace, &mut buffer) {
                            out.set_from_view(&buffer.as_view());
                        }
                        return;
                    }
                }

                // try to turn the argument into a number
                if id == Symbol::COEFF && out_f.to_fun_view().get_nargs() == 1 {
                    let arg = out_f.to_fun_view().iter().next().unwrap();
                    if let AtomView::Num(_) = arg {
                        let mut buffer = workspace.new_atom();
                        buffer.set_from_view(&arg);
                        out.set_from_view(&buffer.as_view());
                        return;
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
                            return;
                        }
                    }
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
                            m.has_coefficient() || m.iter().any(|a| a.is_scalar())
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
                                    } else if a.is_scalar() {
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

                if id.is_symmetric() || id.is_antisymmetric() {
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
                            .any(|w| w[0].1.as_view() == w[1].1.as_view())
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
                            m.extend(handle.as_view());
                            handle.to_num((-1).into());
                            m.extend(handle.as_view());
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

                    let mut best_shift = 0;
                    'shift: for shift in 1..args.len() {
                        for i in 0..args.len() {
                            match args[(i + best_shift) % args.len()]
                                .cmp(&args[(i + shift) % args.len()])
                            {
                                std::cmp::Ordering::Equal => {}
                                std::cmp::Ordering::Less => {
                                    continue 'shift;
                                }
                                std::cmp::Ordering::Greater => break,
                            }
                        }

                        best_shift = shift;
                    }

                    let mut f = workspace.new_atom();
                    let ff = f.to_fun(id);
                    for arg in args[best_shift..].iter().chain(&args[..best_shift]) {
                        ff.add_arg(*arg);
                    }

                    drop(args);

                    ff.set_normalized(true);
                    std::mem::swap(ff, out_f);
                }

                if let Some(f) = State::get_normalization_function(id) {
                    let mut fs = workspace.new_atom();
                    let mut setter = fs.deref_mut().into();
                    f(out.as_view(), &mut setter);
                    if setter.is_set() {
                        std::mem::swap(out, fs.deref_mut());
                    }
                    debug_assert!(!out.as_view().needs_normalization());
                }
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();

                let mut base_handle = workspace.new_atom();
                let mut exp_handle = workspace.new_atom();

                if base.needs_normalization() {
                    base.normalize(workspace, &mut base_handle);
                } else {
                    // TODO: prevent copy
                    base_handle.set_from_view(&base);
                };

                if exp.needs_normalization() {
                    exp.normalize(workspace, &mut exp_handle);
                } else {
                    // TODO: prevent copy
                    exp_handle.set_from_view(&exp);
                };

                'pow_simplify: {
                    if let AtomView::Num(e) = exp_handle.as_view() {
                        let exp_num = e.get_coeff_view();
                        if exp_num == CoefficientView::Natural(1, 1, 0, 1) {
                            // remove power of 1
                            out.set_from_view(&base_handle.as_view());
                            break 'pow_simplify;
                        } else if let AtomView::Num(n) = base_handle.as_view() {
                            // simplify a number raised to a numerical power
                            let (prefactor, new_base_num, new_exp_num) =
                                n.get_coeff_view().pow(&exp_num);

                            if !prefactor.is_one() {
                                let mut mul_h = workspace.new_atom();
                                let m = mul_h.to_mul();
                                base_handle.to_num(prefactor);
                                m.extend(base_handle.as_view());
                                base_handle.to_num(new_base_num);
                                exp_handle.to_num(new_exp_num);
                                out.to_pow(base_handle.as_view(), exp_handle.as_view());
                                m.extend(out.as_view());
                                mul_h.as_view().normalize(workspace, out);
                                break 'pow_simplify;
                            }

                            if new_exp_num.is_one() {
                                out.to_num(new_base_num);
                                break 'pow_simplify;
                            }

                            base_handle.to_num(new_base_num);
                            exp_handle.to_num(new_exp_num);
                        } else if let AtomView::Pow(p_base) = base_handle.as_view() {
                            let exp_is_integer = exp_num.is_integer();
                            if exp_is_integer || base_handle.is_positive() {
                                let (p_base_base, p_base_exp) = p_base.get_base_exp();
                                let mut mul_h = workspace.new_atom();
                                let mul = mul_h.to_mul();

                                mul.extend(p_base_exp);
                                mul.extend(exp_handle.as_view());
                                let mut exp_h = workspace.new_atom();
                                mul.as_view().normalize(workspace, &mut exp_h);

                                if exp_is_integer || p_base_base.is_positive() {
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
                        } else if let AtomView::Mul(m) = base_handle.as_view() {
                            // rewrite (x*y)^2 as x^2*y^2
                            if exp_num.is_integer() {
                                let mut mul_h = workspace.new_atom();
                                let mul = mul_h.to_mul();
                                for arg in m {
                                    let mut pow_h = workspace.new_atom();
                                    pow_h.to_pow(arg, exp_handle.as_view());
                                    mul.extend(pow_h.as_view());
                                }

                                mul_h.as_view().normalize(workspace, out);
                                break 'pow_simplify;
                            }
                        } else if exp_num == CoefficientView::Natural(0, 1, 0, 1) {
                            // x^0 = 1
                            out.to_num(1.into());
                            break 'pow_simplify;
                        } else if let CoefficientView::Natural(n, 1, 0, 1) = exp_num
                            && n % 2 == 0
                            && let AtomView::Fun(f) = base_handle.as_view()
                        {
                            let s = f.get_symbol_id();
                            if s == Symbol::SQRT_ID && f.get_nargs() == 1 {
                                let sqrt_arg = f.iter().next().unwrap();
                                if n == 2 {
                                    out.set_from_view(&sqrt_arg);
                                } else {
                                    let mut pow_h = workspace.new_atom();
                                    pow_h.to_pow(sqrt_arg, workspace.new_num(n / 2).as_view());
                                    pow_h.as_view().normalize(workspace, out);
                                    out.set_from_view(&pow_h.as_view());
                                }

                                break 'pow_simplify;
                            } else if s == Symbol::ABS_ID && f.get_nargs() == 1 {
                                let abs_arg = f.iter().next().unwrap();
                                if abs_arg.is_real() {
                                    let mut pow_h = workspace.new_atom();
                                    pow_h.to_pow(abs_arg, workspace.new_num(n).as_view());
                                    pow_h.as_view().normalize(workspace, out);
                                    out.set_from_view(&pow_h.as_view());
                                    break 'pow_simplify;
                                }
                            }
                        }
                    } else if let AtomView::Pow(p_base) = base_handle.as_view()
                        && exp_handle.is_integer()
                    {
                        // rewrite (x^y)^z as x^(z*y) if z is integer
                        let (p_base_base, p_base_exp) = p_base.get_base_exp();

                        let mut mul_h = workspace.new_atom();
                        let mul = mul_h.to_mul();
                        mul.extend(p_base_exp);
                        mul.extend(exp_handle.as_view());
                        let mut exp_h = workspace.new_atom();
                        mul.as_view().normalize(workspace, &mut exp_h);

                        mul_h.to_pow(p_base_base, exp_h.as_view());
                        mul_h.as_view().normalize(workspace, out);
                        break 'pow_simplify;
                    }
                    out.to_pow(base_handle.as_view(), exp_handle.as_view());
                }

                out.set_normalized(true);
            }
            AtomView::Add(a) => {
                let mut new_sum = workspace.new_atom();
                let ns = new_sum.to_add();

                let mut atom_sort_buf: SmallVec<[_; 20]> = SmallVec::new();

                let mut norm_arg = workspace.new_atom();
                for a in a {
                    let r = if a.needs_normalization() {
                        // TODO: if a is a nested addition, prevent a sort
                        a.normalize(workspace, &mut norm_arg);
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

                for x in ns.to_add_view().iter() {
                    atom_sort_buf.push((x, x.get_term_cmp_slice()));
                }

                atom_sort_buf.sort_unstable_by(|a, b| a.1.cmp(&b.1));

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
                    if !last_buf.merge_terms(*cur, &mut helper) {
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
                    out.set_from_view(&last_buf.as_view());
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
                            out.set_from_view(&last_buf.as_view());
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
    }

    /// Add two atoms and normalize the result.
    pub(crate) fn add_normalized(&self, rhs: AtomView, ws: &Workspace, out: &mut Atom) {
        // write (a+b)+(a+b) as 2*(a+b)
        if *self == rhs {
            let mut a = ws.new_atom();
            let m = a.to_mul();
            m.extend(InlineNum::new(2, 1).as_view());
            m.extend(*self);
            a.as_view().normalize(ws, out);
            return;
        }

        let a = out.to_add();
        a.grow_capacity(self.get_byte_size() + rhs.get_byte_size());

        let mut helper = ws.new_atom();
        let mut b = ws.new_atom();
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
                        match ss.cmp_terms(&tt) {
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
                out.set_from_view(&b.as_view());
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

                    match x.cmp_terms(&rhs) {
                        Ordering::Less => {
                            a.extend(x);
                        }
                        Ordering::Equal => {
                            found = true;
                            b.set_from_view(&x);
                            if b.merge_terms(rhs, &mut helper) {
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
                match v.binary_search_by(|a| a.cmp_terms(&rhs)) {
                    Ok(p) => {
                        for x in v.iter().take(p) {
                            a.extend(*x);
                        }

                        b.set_from_view(&v[p]);
                        if b.merge_terms(rhs, &mut helper) {
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
                out.set_from_view(&b.as_view());
            } else {
                a.set_normalized(true);
            }
        } else if let AtomView::Add(_) = rhs {
            rhs.add_normalized(*self, ws, out);
        } else {
            let mut e = ws.new_atom();
            let a = e.to_add();
            a.extend(*self);
            a.extend(rhs);
            e.as_view().normalize(ws, out);
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        atom::{Atom, AtomCore},
        parse,
        printer::PrintOptions,
    };

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
    }
}
