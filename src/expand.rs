use std::{ops::DerefMut, sync::Arc};

use smallvec::SmallVec;

use crate::{
    atom::{Atom, AtomView},
    coefficient::CoefficientView,
    combinatorics::CombinationWithReplacementIterator,
    domains::{integer::Integer, rational::Q},
    poly::{Exponent, PolyVariable},
    state::{RecycledAtom, Workspace},
};

impl AtomView<'_> {
    /// Expand an expression. The function [expand_via_poly] may be faster.
    pub(crate) fn expand(&self) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut a = ws.new_atom();
            self.expand_with_ws_into(ws, None, &mut a);
            a.into_inner()
        })
    }

    /// Expand an expression. The function [expand_via_poly] may be faster.
    pub(crate) fn expand_in(&self, var: AtomView) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut a = ws.new_atom();
            self.expand_with_ws_into(ws, Some(var), &mut a);
            a.into_inner()
        })
    }

    /// Expand an expression, returning `true` iff the expression changed.
    pub(crate) fn expand_into(&self, var: Option<AtomView>, out: &mut Atom) -> bool {
        Workspace::get_local().with(|ws| self.expand_with_ws_into(ws, var, out))
    }

    /// Expand an expression, returning `true` iff the expression changed.
    pub(crate) fn expand_with_ws_into(
        &self,
        workspace: &Workspace,
        var: Option<AtomView>,
        out: &mut Atom,
    ) -> bool {
        let changed = self.expand_no_norm(workspace, var, out);

        if changed {
            let mut a = workspace.new_atom();
            out.as_view().normalize(workspace, &mut a);
            std::mem::swap(out, &mut a);
        }

        changed
    }

    /// Check if the expression is expanded, optionally in only the variable or function `var`.
    pub(crate) fn is_expanded(&self, var: Option<AtomView>) -> bool {
        match self {
            AtomView::Num(_) | AtomView::Var(_) | AtomView::Fun(_) => true,
            AtomView::Pow(pow_view) => {
                let (base, exp) = pow_view.get_base_exp();
                if !base.is_expanded(var) || !exp.is_expanded(var) {
                    return false;
                }

                if let AtomView::Num(n) = exp
                    && let CoefficientView::Natural(n, 1, 0, 1) = n.get_coeff_view()
                    && n.unsigned_abs() <= u32::MAX as u64
                    && matches!(base, AtomView::Add(_) | AtomView::Mul(_))
                {
                    return var.map(|s| !base.contains(s)).unwrap_or(false);
                }

                true
            }
            AtomView::Mul(mul_view) => {
                for arg in mul_view {
                    if !arg.is_expanded(var) {
                        return false;
                    }

                    if matches!(arg, AtomView::Add(_)) {
                        return var.map(|s| !arg.contains(s)).unwrap_or(false);
                    }
                }

                true
            }
            AtomView::Add(add_view) => {
                for arg in add_view {
                    if !arg.is_expanded(var) {
                        return false;
                    }
                }

                true
            }
        }
    }

    /// Expand the expression by converting it to a polynomial, optionally
    /// only in the indeterminate `var`. The parameter `E` should be a numerical type
    /// that fits the largest exponent in the expanded expression. Often,
    /// `u8` or `u16` is sufficient.
    pub(crate) fn expand_via_poly<E: Exponent>(&self, var: Option<AtomView>) -> Atom {
        let var_map = var.map(|v| Arc::new(vec![v.to_owned().try_into().unwrap()]));

        let mut out = Atom::new();
        Workspace::get_local().with(|ws| {
            self.expand_via_poly_impl::<E>(ws, var, &var_map, &mut out);
        });
        out
    }

    fn expand_via_poly_impl<E: Exponent>(
        &self,
        ws: &Workspace,
        var: Option<AtomView>,
        var_map: &Option<Arc<Vec<PolyVariable>>>,
        out: &mut Atom,
    ) {
        if self.is_expanded(var) {
            out.set_from_view(self);
            return;
        }

        if let Some(v) = var
            && !self.contains(v)
        {
            out.set_from_view(self);
            return;
        }

        match self {
            AtomView::Num(_) | AtomView::Var(_) | AtomView::Fun(_) => unreachable!(),
            AtomView::Pow(_) => {
                if let Some(v) = var_map {
                    *out = self.to_polynomial_in_vars::<E>(v).flatten(true);
                } else {
                    if let Ok(p) = self.try_to_polynomial::<_, E>(&Q, None) {
                        *out = p.to_expression();
                    } else {
                        self.expand_into(var, out);
                    }
                }
            }
            AtomView::Mul(_) => {
                if let Some(v) = var_map {
                    *out = self.to_polynomial_in_vars::<E>(v).flatten(true);
                } else {
                    if let Ok(p) = self.try_to_polynomial::<_, E>(&Q, None) {
                        *out = p.to_expression();
                    } else {
                        self.expand_into(var, out);
                    }
                }
            }
            AtomView::Add(add_view) => {
                let mut t = ws.new_atom();

                let add = out.to_add();

                for arg in add_view {
                    arg.expand_via_poly_impl::<E>(ws, var, var_map, &mut t);
                    add.extend(t.as_view());
                }

                add.as_view().normalize(ws, &mut t);
                std::mem::swap(out, &mut t);
            }
        }
    }

    /// Expand an expression, but do not normalize the result.
    fn expand_no_norm(&self, workspace: &Workspace, var: Option<AtomView>, out: &mut Atom) -> bool {
        if let Some(s) = var
            && !self.contains(s)
        {
            out.set_from_view(self);
            return false;
        }

        match self {
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();

                let mut new_base = workspace.new_atom();
                let mut changed = base.expand_with_ws_into(workspace, var, &mut new_base);

                let mut new_exp = workspace.new_atom();
                changed |= exp.expand_with_ws_into(workspace, var, &mut new_exp);

                let (negative, num) = 'get_num: {
                    if let AtomView::Num(n) = new_exp.as_view()
                        && let CoefficientView::Natural(n, 1, 0, 1) = n.get_coeff_view()
                        && n.unsigned_abs() <= u32::MAX as u64
                    {
                        break 'get_num (n < 0, n.unsigned_abs() as u32);
                    }

                    let mut pow_h = workspace.new_atom();
                    let pow = pow_h.to_pow(new_base.as_view(), new_exp.as_view());
                    pow.set_normalized(!changed);
                    pow_h.as_view().normalize(workspace, out);
                    return changed;
                };

                if let AtomView::Add(a) = new_base.as_view() {
                    // expand (a+b+c+..)^n
                    let mut args: SmallVec<[AtomView; 10]> = SmallVec::with_capacity(a.get_nargs());
                    for arg in a {
                        args.push(arg);
                    }

                    let mut add_h = workspace.new_atom();
                    let add = add_h.to_add();

                    let mut ci = CombinationWithReplacementIterator::new(args.len(), num);

                    while let Some(new_term) = ci.next() {
                        let mut hh = workspace.new_atom();
                        let p = hh.to_mul();

                        let mut hhh = workspace.new_atom();
                        for (a, pow) in args.iter().zip(new_term) {
                            if *pow != 0 {
                                let mut new_exp_h = workspace.new_atom();
                                new_exp_h.to_num((*pow as i64).into());
                                hhh.to_pow(*a, new_exp_h.as_view());
                                p.extend(hhh.as_view());
                            }
                        }

                        let mut normalized_child = workspace.new_atom();
                        hh.as_view().normalize(workspace, &mut normalized_child);

                        let mut expanded_child = workspace.new_atom();
                        normalized_child.as_view().expand_with_ws_into(
                            workspace,
                            var,
                            &mut expanded_child,
                        );

                        let coeff_f = Integer::multinom(new_term);
                        if coeff_f != Integer::one() {
                            let mut coeff_h = workspace.new_atom();
                            coeff_h.to_num(coeff_f.into());

                            if let Atom::Mul(m) = expanded_child.deref_mut() {
                                m.extend(coeff_h.as_view());
                                add.extend(expanded_child.as_view());
                            } else {
                                let mut mul_h = workspace.new_atom();
                                let mul = mul_h.to_mul();
                                mul.extend(expanded_child.as_view());
                                mul.extend(coeff_h.as_view());
                                add.extend(mul_h.as_view());
                            }
                        } else {
                            add.extend(expanded_child.as_view());
                        }
                    }

                    if negative {
                        let mut num_h = workspace.new_atom();
                        num_h.to_num((-1i64).into());

                        let mut pow_h = workspace.new_atom();
                        pow_h.to_pow(add_h.as_view(), num_h.as_view());

                        pow_h.as_view().normalize(workspace, out);
                    } else {
                        add_h.as_view().normalize(workspace, out);
                    }

                    true
                } else if let AtomView::Mul(m) = new_base.as_view() {
                    let mut mul_h = workspace.new_atom();
                    let mul = mul_h.to_mul();

                    let mut exp_h = workspace.new_atom();
                    if negative {
                        exp_h.to_num((-(num as i64)).into());
                    } else {
                        exp_h.to_num((num as i64).into());
                    }

                    for arg in m {
                        let mut pow_h = workspace.new_atom();
                        pow_h.to_pow(arg, exp_h.as_view());
                        mul.extend(pow_h.as_view());
                    }

                    mul_h.as_view().normalize(workspace, out);

                    true
                } else {
                    let mut pow_h = workspace.new_atom();
                    let pow = pow_h.to_pow(new_base.as_view(), new_exp.as_view());
                    pow.set_normalized(!changed);
                    pow_h.as_view().normalize(workspace, out);
                    changed
                }
            }
            AtomView::Mul(m) => {
                let mut changed = false;

                let mut sum: SmallVec<[RecycledAtom; 10]> = SmallVec::new();
                let mut new_sum: SmallVec<[RecycledAtom; 10]> = SmallVec::new();

                for arg in m {
                    let mut new_arg = workspace.new_atom();
                    changed |= arg.expand_with_ws_into(workspace, var, &mut new_arg);

                    // expand (1+x)*y
                    if let AtomView::Add(a) = new_arg.as_view() {
                        changed = true;

                        for child in a {
                            for s in &sum {
                                let mut b = workspace.new_atom();
                                b.set_from_view(&s.as_view());

                                if let Atom::Mul(m) = b.deref_mut() {
                                    m.extend(child);
                                    new_sum.push(b);
                                } else {
                                    let mut mul_h = workspace.new_atom();
                                    let mul = mul_h.to_mul();
                                    mul.extend(b.as_view());
                                    mul.extend(child);
                                    new_sum.push(mul_h);
                                }
                            }

                            if sum.is_empty() {
                                let mut b = workspace.new_atom();
                                b.set_from_view(&child);
                                new_sum.push(b);
                            }
                        }

                        std::mem::swap(&mut sum, &mut new_sum);
                        new_sum.clear();
                    } else if sum.is_empty() {
                        sum.push(new_arg);
                    } else {
                        for summand in &mut sum {
                            if let Atom::Mul(m) = summand.deref_mut() {
                                m.extend(new_arg.as_view());
                            } else {
                                let mut mul_h = workspace.new_atom();
                                let mul = mul_h.to_mul();
                                mul.extend(summand.as_view());
                                mul.extend(new_arg.as_view());
                                *summand = mul_h;
                            }
                        }
                    }
                }

                if !changed {
                    out.set_from_view(self);
                    return false;
                }

                debug_assert!(!sum.is_empty());

                if sum.len() == 1 {
                    sum[0].as_view().normalize(workspace, out);
                } else {
                    let add = out.to_add();
                    for x in sum {
                        add.extend(x.as_view());
                    }
                }

                changed
            }
            AtomView::Add(a) => {
                let mut changed = false;

                let add = out.to_add();

                let mut new_arg = workspace.new_atom();
                for arg in a {
                    changed |= arg.expand_no_norm(workspace, var, &mut new_arg);
                    add.extend(new_arg.as_view());
                }

                add.set_normalized(!changed);
                changed
            }
            _ => {
                out.set_from_view(self);
                false
            }
        }
    }

    /// Distribute numbers in the expression, for example:
    /// `2*(x+y)` -> `2*x+2*y`.
    pub(crate) fn expand_num(&self) -> Atom {
        let mut a = Atom::new();
        Workspace::get_local().with(|ws| {
            self.expand_num_impl(ws, &mut a);
        });
        a
    }

    pub(crate) fn expand_num_into(&self, out: &mut Atom) {
        Workspace::get_local().with(|ws| {
            self.expand_with_ws_into(ws, None, out);
        })
    }

    pub(crate) fn expand_num_impl(&self, ws: &Workspace, out: &mut Atom) -> bool {
        match self {
            AtomView::Num(_) | AtomView::Var(_) | AtomView::Fun(_) => {
                out.set_from_view(self);
                false
            }
            AtomView::Pow(pow_view) => {
                let (base, exp) = pow_view.get_base_exp();
                let mut new_base = ws.new_atom();
                let mut changed = base.expand_num_impl(ws, &mut new_base);

                let mut new_exp = ws.new_atom();
                changed |= exp.expand_num_impl(ws, &mut new_exp);

                let mut pow_h = ws.new_atom();
                pow_h.to_pow(new_base.as_view(), new_exp.as_view());
                pow_h.as_view().normalize(ws, out);

                changed
            }
            AtomView::Mul(mul_view) => {
                let mut changed = false;

                // propagate to all arguments
                let mut new_mul = ws.new_atom();
                let m = new_mul.to_mul();
                let mut new_arg = ws.new_atom();
                for arg in mul_view {
                    changed |= arg.expand_num_impl(ws, &mut new_arg);
                    m.extend(new_arg.as_view());
                }

                if changed {
                    new_mul.as_view().normalize(ws, &mut new_arg);
                    new_arg.as_view().expand_num_impl(ws, out);
                    return true;
                }

                if !mul_view.has_coefficient()
                    || !mul_view.iter().any(|a| matches!(a, AtomView::Add(_)))
                {
                    out.set_from_view(self);
                    return false;
                }

                let mut args: Vec<_> = mul_view.iter().collect();
                let mut sum = None;
                let mut num = None;

                args.retain(|a| {
                    if let AtomView::Add(_) = a {
                        if sum.is_none() {
                            sum = Some(*a);
                            false
                        } else {
                            true
                        }
                    } else if let AtomView::Num(_) = a {
                        if num.is_none() {
                            num = Some(*a);
                            false
                        } else {
                            true
                        }
                    } else {
                        true
                    }
                });

                let mut add = ws.new_atom();
                let add_view = add.to_add();
                let n = num.unwrap();

                let mut m = ws.new_atom();
                if let AtomView::Add(sum) = sum.unwrap() {
                    for a in sum.iter() {
                        let mm = m.to_mul();
                        mm.extend(a);
                        mm.extend(n);
                        add_view.extend(m.as_view());
                    }
                }

                add_view.as_view().normalize(ws, &mut m);
                let m2 = add.to_mul();
                for a in args {
                    m2.extend(a);
                }
                m2.extend(m.as_view());

                m2.as_view().normalize(ws, out);

                true
            }
            AtomView::Add(add_view) => {
                let mut changed = false;

                let mut new = ws.new_atom();
                let add = new.to_add();

                let mut new_arg = ws.new_atom();
                for arg in add_view {
                    changed |= arg.expand_num_impl(ws, &mut new_arg);
                    add.extend(new_arg.as_view());
                }

                if !changed {
                    out.set_from_view(self);
                    return false;
                }

                new.as_view().normalize(ws, out);
                true
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::atom::AtomCore;
    use crate::{parse, symbol};

    #[test]
    fn expand_num() {
        let exp = parse!("5+2*v3*(v1-v2)*(v4+v5)").expand_num();
        let res = parse!("5+v3*(v4+v5)*(2*v1-2*v2)");
        assert_eq!(exp, res);
    }

    #[test]
    fn exponent() {
        let exp = parse!("(1+v1+v2)^4").expand();
        let res = parse!(
            "4*v1+4*v2+6*v1^2+4*v1^3+v1^4+6*v2^2+4*v2^3+v2^4+12*v1*v2+12*v1*v2^2+4*v1*v2^3+12*v1^2*v2+6*v1^2*v2^2+4*v1^3*v2+1"
        );
        assert_eq!(exp, res);
    }

    #[test]
    fn association() {
        let exp = parse!("(1+v1)*(2+v2)*(3+v1)").expand();
        let res = parse!("8*v1+3*v2+2*v1^2+4*v1*v2+v1^2*v2+6");
        assert_eq!(exp, res);
    }

    #[test]
    fn mul_pow() {
        let exp = parse!("(v1*v2*2)^3*2").expand();
        let res = parse!("v1^3*v2^3*16");
        assert_eq!(exp, res);
    }

    #[test]
    fn mul_pow_neg() {
        let exp = parse!("(v1*v2*2)^-3").expand();
        let res = parse!("8^-1*v1^-3*v2^-3");
        assert_eq!(exp, res);
    }

    #[test]
    fn expand_in_var() {
        let exp = parse!("(1+v1)^2+(1+v2)^100").expand_in_symbol(symbol!("v1"));
        let res = parse!("1+2*v1+v1^2+(v2+1)^100");
        assert_eq!(exp, res);
    }

    #[test]
    fn expand_with_poly() {
        let exp = parse!("(1+v1)^2+(1+v2)^100").expand_in_symbol(symbol!("v1"));
        let res = parse!("1+2*v1+v1^2+(v2+1)^100");
        assert_eq!(exp, res);
    }
}
