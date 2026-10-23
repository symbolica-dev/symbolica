use std::{ops::DerefMut, sync::Arc};

use smallvec::SmallVec;

use crate::{
    atom::{Atom, AtomView, ListIterator},
    coefficient::CoefficientView,
    combinatorics::CombinationWithReplacementIterator,
    domains::{integer::Integer, rational::Q},
    poly::{Exponent, PolyVariable},
    state::Workspace,
    utils::Settable,
};

// Borrow only the alias chain needed for a structural decision. Unchanged
// subexpressions retain their original compressed representation.
#[inline]
fn expansion_body(view: AtomView<'_>) -> AtomView<'_> {
    if matches!(view, AtomView::Alias(_)) {
        expansion_alias_body(view)
    } else {
        view
    }
}

#[cold]
#[inline(never)]
fn expansion_alias_body(mut view: AtomView<'_>) -> AtomView<'_> {
    while let AtomView::Alias(alias) = view {
        if alias.is_opaque() {
            break;
        }
        view = alias.get_body();
    }
    view
}

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
        let mut set = Settable::from(&mut *out);
        self.expand_with_ws_into_settable(workspace, var, &mut set);
        let changed = set.is_set();
        if !changed {
            out.set_from_view(self);
        }

        changed
    }

    /// Expand and normalize, leaving `out` unset if no expansion was needed.
    fn expand_with_ws_into_settable(
        &self,
        workspace: &Workspace,
        var: Option<AtomView>,
        out: &mut Settable<'_, Atom>,
    ) {
        self.expand_no_norm(workspace, var, out);
        if let Some(value) = out.get() {
            let mut a = workspace.new_atom();
            value.as_view().normalize(workspace, &mut a);
            std::mem::swap(&mut **out, &mut a);
        }
    }

    /// Check if the expression is expanded, optionally in only the variable or function `var`.
    pub(crate) fn is_expanded(&self, var: Option<AtomView>) -> bool {
        match self {
            AtomView::Alias(a) => a.is_opaque() || a.get_body().is_expanded(var),
            AtomView::Num(_) | AtomView::Var(_) | AtomView::Fun(_) => true,
            AtomView::Pow(pow_view) => {
                let (base, exp) = pow_view.get_base_exp();
                if !base.is_expanded(var) || !exp.is_expanded(var) {
                    return false;
                }

                let base = expansion_body(base);
                let exp = expansion_body(exp);
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

                    let body = expansion_body(arg);
                    if matches!(body, AtomView::Add(_)) && var.is_none_or(|s| body.contains(s)) {
                        return false;
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

        // Polynomial conversion does not yet resolve aliased exponents or
        // preserve opaque aliases throughout its recursive conversion paths.
        // Native expansion handles both while retaining unchanged aliases.
        if self.has_alias() {
            self.expand_with_ws_into(ws, var, out);
            return;
        }

        match self {
            AtomView::Alias(a) => {
                a.get_body()
                    .expand_via_poly_impl::<E>(ws, var, var_map, out);
            }
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

    /// Expand an expression without final normalization, leaving `out` unset
    /// if no expansion was needed.
    fn expand_no_norm(
        &self,
        workspace: &Workspace,
        var: Option<AtomView>,
        out: &mut Settable<'_, Atom>,
    ) {
        if let Some(s) = var
            && !self.contains_literally_or_as_symbol(s)
        {
            return;
        }

        match self {
            AtomView::Alias(a) => {
                if !a.is_opaque() {
                    a.get_body().expand_no_norm(workspace, var, out);
                }
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();

                let mut new_base = workspace.new_atom();
                let mut base_set = Settable::from(new_base.deref_mut());
                base.expand_with_ws_into_settable(workspace, var, &mut base_set);

                let mut new_exp = workspace.new_atom();
                let mut exp_set = Settable::from(new_exp.deref_mut());
                exp.expand_with_ws_into_settable(workspace, var, &mut exp_set);

                let changed = base_set.is_set() || exp_set.is_set();
                let base = base_set.as_view_or(base);
                let exp = exp_set.as_view_or(exp);
                let structural_base = expansion_body(base);
                let structural_exp = expansion_body(exp);

                let (negative, num) = 'get_num: {
                    if let AtomView::Num(n) = structural_exp
                        && let CoefficientView::Natural(n, 1, 0, 1) = n.get_coeff_view()
                        && n.unsigned_abs() <= u32::MAX as u64
                    {
                        break 'get_num (n < 0, n.unsigned_abs() as u32);
                    }

                    if changed {
                        let mut pow_h = workspace.new_atom();
                        pow_h.to_pow(base, exp);
                        pow_h.as_view().normalize(workspace, out);
                    }
                    return;
                };

                if let AtomView::Add(a) = structural_base {
                    // expand (a+b+c+..)^n
                    let mut rest_buffer = workspace.new_atom();
                    let mut args: SmallVec<[AtomView; 10]> = SmallVec::with_capacity(a.get_nargs());
                    let mut rest_args: SmallVec<[AtomView; 10]> = SmallVec::new();
                    for arg in a {
                        if let Some(s) = var
                            && !arg.contains_literally_or_as_symbol(s)
                        {
                            rest_args.push(arg);
                        } else {
                            args.push(arg);
                        }
                    }

                    if rest_args.len() == 1 {
                        args.push(rest_args[0]);
                    } else if rest_args.len() > 1 {
                        let add = rest_buffer.to_add();
                        for arg in rest_args {
                            add.extend(arg);
                        }
                        add.set_normalized(true); // ordered subset of terms is normalized
                        args.push(rest_buffer.as_view());
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
                                new_exp_h.to_num(*pow as i64);
                                hhh.to_pow(*a, new_exp_h.as_view());
                                p.extend(hhh.as_view());
                            }
                        }

                        let mut normalized_child = workspace.new_atom();
                        hh.as_view().normalize(workspace, &mut normalized_child);

                        let mut expanded_child = workspace.new_atom();
                        let mut child_set = Settable::from(expanded_child.deref_mut());
                        normalized_child.as_view().expand_with_ws_into_settable(
                            workspace,
                            var,
                            &mut child_set,
                        );
                        // The normalized child is already owned, so reuse it
                        // directly when recursive expansion leaves it unchanged.
                        let child = if child_set.is_set() {
                            expanded_child.deref_mut()
                        } else {
                            normalized_child.deref_mut()
                        };

                        let coeff_f = Integer::multinom(new_term);
                        if coeff_f != Integer::one() {
                            let mut coeff_h = workspace.new_atom();
                            coeff_h.to_num(coeff_f);

                            if let Atom::Mul(m) = child {
                                m.extend(coeff_h.as_view());
                                add.extend(child.as_view());
                            } else {
                                let mut mul_h = workspace.new_atom();
                                let mul = mul_h.to_mul();
                                mul.extend(child.as_view());
                                mul.extend(coeff_h.as_view());
                                add.extend(mul_h.as_view());
                            }
                        } else {
                            add.extend(child.as_view());
                        }
                    }

                    if negative {
                        let mut num_h = workspace.new_atom();
                        num_h.to_num(-1i64);

                        let mut pow_h = workspace.new_atom();
                        pow_h.to_pow(add_h.as_view(), num_h.as_view());

                        pow_h.as_view().normalize(workspace, out);
                    } else {
                        add_h.as_view().normalize(workspace, out);
                    }
                } else if let AtomView::Mul(m) = structural_base {
                    let mut mul_h = workspace.new_atom();
                    let mul = mul_h.to_mul();

                    let mut exp_h = workspace.new_atom();
                    if negative {
                        exp_h.to_num(-(num as i64));
                    } else {
                        exp_h.to_num(num as i64);
                    }

                    for arg in m {
                        let mut pow_h = workspace.new_atom();
                        pow_h.to_pow(arg, exp_h.as_view());
                        mul.extend(pow_h.as_view());
                    }

                    mul_h.as_view().normalize(workspace, out);
                } else if changed {
                    let mut pow_h = workspace.new_atom();
                    pow_h.to_pow(base, exp);
                    pow_h.as_view().normalize(workspace, out);
                }
            }
            AtomView::Mul(m) => {
                let mut changed = false;
                let mut sum = workspace.new_atom();
                let mut new_sum = workspace.new_atom();
                let mut new_arg = workspace.new_atom();
                let mut term = workspace.new_atom();
                // An unrelated sum in expand_in() must remain a single factor.
                let mut expand_sum = false;

                for (factor_index, arg) in m.iter().enumerate() {
                    let mut arg_set = Settable::from(new_arg.deref_mut());
                    arg.expand_with_ws_into_settable(workspace, var, &mut arg_set);
                    let original_arg = arg_set.as_view_or(arg);
                    let body = expansion_body(original_arg);
                    let expand_arg = matches!(body, AtomView::Add(_))
                        && var.is_none_or(|s| body.contains_literally_or_as_symbol(s));
                    let borrowed_sum = expand_arg && matches!(original_arg, AtomView::Alias(_));
                    let arg = if expand_arg { body } else { original_arg };

                    if !changed {
                        if !arg_set.is_set() && !expand_arg {
                            continue;
                        }
                        changed = true;

                        if factor_index == 0 {
                            if arg_set.is_set() && !borrowed_sum {
                                std::mem::swap(&mut sum, &mut new_arg);
                            } else {
                                sum.set_from_view(&arg);
                            }
                            expand_sum = expand_arg;
                            continue;
                        }

                        // Materialize the unchanged prefix only when a factor
                        // changes or needs distribution.
                        let prefix = sum.to_mul();
                        for factor in m.iter().take(factor_index) {
                            prefix.extend(factor);
                        }
                    }

                    if expand_arg || expand_sum {
                        let terms = match sum.as_view() {
                            AtomView::Add(a) if expand_sum => a.iter(),
                            a => ListIterator::from_one(a),
                        };
                        let args = match arg {
                            AtomView::Add(a) if expand_arg => a.iter(),
                            a => ListIterator::from_one(a),
                        };

                        let add = new_sum.to_add();
                        for child in args {
                            for s in terms {
                                let mul = term.to_mul();
                                mul.extend(s);
                                mul.extend(child);
                                add.extend(term.as_view());
                            }
                        }

                        // Fuse terms before the next factor. The final product
                        // is normalized by the caller.
                        if expand_arg && factor_index + 1 < m.get_nargs() {
                            new_sum.as_view().normalize(workspace, &mut sum);
                        } else {
                            std::mem::swap(&mut sum, &mut new_sum);
                        }
                        expand_sum = matches!(sum.as_view(), AtomView::Add(_));
                    } else if let Atom::Mul(m) = sum.deref_mut() {
                        m.extend(arg);
                    } else {
                        let mul = new_sum.to_mul();
                        mul.extend(sum.as_view());
                        mul.extend(arg);
                        std::mem::swap(&mut sum, &mut new_sum);
                    }
                }

                if changed {
                    std::mem::swap(&mut **out, &mut sum);
                }
            }
            AtomView::Add(a) => {
                let mut add = None;
                let mut new_arg = workspace.new_atom();
                for (i, arg) in a.iter().enumerate() {
                    let mut arg_set = Settable::from(new_arg.deref_mut());
                    arg.expand_no_norm(workspace, var, &mut arg_set);

                    if add.is_none() && arg_set.is_set() {
                        let new_add = out.to_add();
                        for child in a.iter().take(i) {
                            new_add.extend(child);
                        }
                        new_add.extend(arg_set.as_view());
                        add = Some(new_add);
                    } else if let Some(add) = &mut add {
                        add.extend(arg_set.as_view_or(arg));
                    }
                }
            }
            _ => {}
        }
    }

    /// Distribute numbers in the expression, for example:
    /// `2*(x+y)` -> `2*x+2*y`.
    pub(crate) fn expand_num(&self) -> Atom {
        let mut a = Atom::new();
        self.expand_num_into(&mut a);
        a
    }

    pub(crate) fn expand_num_into(&self, out: &mut Atom) {
        Workspace::get_local().with(|ws| {
            let mut set = Settable::from(&mut *out);
            self.expand_num_impl(ws, &mut set);
            if !set.is_set() {
                out.set_from_view(self);
            }
        })
    }

    /// Distribute numbers, leaving `out` unset if no expansion was needed.
    fn expand_num_impl(&self, ws: &Workspace, out: &mut Settable<'_, Atom>) {
        match self {
            AtomView::Alias(a) => {
                if !a.is_opaque() {
                    a.get_body().expand_num_impl(ws, out);
                }
            }
            AtomView::Num(_) | AtomView::Var(_) | AtomView::Fun(_) => {}
            AtomView::Pow(pow_view) => {
                let (base, exp) = pow_view.get_base_exp();
                let mut new_base = ws.new_atom();
                let mut base_set = Settable::from(new_base.deref_mut());
                base.expand_num_impl(ws, &mut base_set);

                let mut new_exp = ws.new_atom();
                let mut exp_set = Settable::from(new_exp.deref_mut());
                exp.expand_num_impl(ws, &mut exp_set);

                if base_set.is_set() || exp_set.is_set() {
                    let mut pow_h = ws.new_atom();
                    pow_h.to_pow(base_set.as_view_or(base), exp_set.as_view_or(exp));
                    pow_h.as_view().normalize(ws, out);
                }
            }
            AtomView::Mul(mul_view) => {
                // Rebuild only after a child changes.
                let mut new_mul = ws.new_atom();
                let mut mul = None;
                let mut new_arg = ws.new_atom();
                for (i, arg) in mul_view.iter().enumerate() {
                    let mut set = Settable::from(new_arg.deref_mut());
                    arg.expand_num_impl(ws, &mut set);
                    if mul.is_none() && set.is_set() {
                        let m = new_mul.to_mul();
                        for child in mul_view.iter().take(i) {
                            m.extend(child);
                        }
                        m.extend(set.as_view());
                        mul = Some(m);
                    } else if let Some(m) = &mut mul {
                        m.extend(set.as_view_or(arg));
                    }
                }

                if mul.is_some() {
                    new_mul.as_view().normalize(ws, &mut new_arg);
                    new_arg.as_view().expand_num_impl(ws, out);
                    // A child changed even if the normalized product needs
                    // no further expansion. Transfer that result to the caller.
                    if !out.is_set() {
                        std::mem::swap(&mut **out, &mut new_arg);
                    }
                    return;
                }

                if !mul_view.has_coefficient()
                    || !mul_view
                        .iter()
                        .any(|a| matches!(expansion_body(a), AtomView::Add(_)))
                {
                    return;
                }

                let mut args: Vec<_> = mul_view.iter().collect();
                let mut sum = None;
                let mut num = None;

                args.retain(|a| {
                    let body = expansion_body(*a);
                    if let AtomView::Add(_) = body {
                        if sum.is_none() {
                            sum = Some(body);
                            false
                        } else {
                            true
                        }
                    } else if let AtomView::Num(_) = body {
                        if num.is_none() {
                            num = Some(body);
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
            }
            AtomView::Add(add_view) => {
                let mut new = ws.new_atom();
                let mut add = None;

                let mut new_arg = ws.new_atom();
                for (i, arg) in add_view.iter().enumerate() {
                    let mut set = Settable::from(new_arg.deref_mut());
                    arg.expand_num_impl(ws, &mut set);
                    if add.is_none() && set.is_set() {
                        let a = new.to_add();
                        for child in add_view.iter().take(i) {
                            a.extend(child);
                        }
                        a.extend(set.as_view());
                        add = Some(a);
                    } else if let Some(a) = &mut add {
                        a.extend(set.as_view_or(arg));
                    }
                }

                if add.is_some() {
                    new.as_view().normalize(ws, out);
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::atom::{Atom, AtomCore, AtomView};
    use crate::{parse, symbol};

    #[test]
    fn expansion_respects_alias_opacity() {
        for body in [parse!("(x+y)^2"), parse!("2*(x+y)")] {
            let transparent = body.alias(false);
            let opaque = body.alias(true);
            assert_eq!(transparent.expand(), body.expand());
            assert_eq!(opaque.expand(), opaque);
            assert_eq!(opaque.expand_num(), opaque);
            assert_eq!(opaque.as_view().expand_via_poly::<u16>(None), opaque);
        }
        let body = parse!("2*(x+y)");
        assert_eq!(body.alias(false).expand_num(), body.expand_num());
    }

    #[test]
    fn polynomial_expansion_resolves_transparent_integer_exponents() {
        let x = parse!("x");
        let expected = parse!("x^2+2*x*y+y^2");
        for exponent in [
            Atom::num(2).alias(false),
            Atom::num(2).alias(false).alias(false),
        ] {
            let input = parse!("x+y").pow(exponent);
            let AtomView::Pow(power) = input.as_view() else {
                panic!("expected a compressed power");
            };
            assert!(matches!(power.get_exp(), AtomView::Alias(_)));
            for var in [None, Some(x.as_view())] {
                let result = input.as_view().expand_via_poly::<u16>(var);
                assert_eq!(result.as_view().get_data(), expected.as_view().get_data());
            }
        }
    }

    #[test]
    fn polynomial_expansion_preserves_opaque_aliases_in_mixed_products() {
        let x = parse!("x");
        let y = parse!("y");
        let opaque = parse!("f(z)").alias(true);
        let input = parse!("x+y").alias(false).alias(false) * &opaque;
        let expected = &x * &opaque + &y * &opaque;
        for var in [None, Some(x.as_view())] {
            let result = input.as_view().expand_via_poly::<u16>(var);
            assert_eq!(result, expected);
            assert_ne!(result, parse!("x*f(z)+y*f(z)"));
            assert!(result.is_expanded(var));
        }
    }

    fn assert_alias_expands_like_plain(input: &Atom, plain: &Atom) {
        let expected = plain.expand();
        let mut result = parse!("stale_expansion_output");
        assert!(input.expand_into::<Atom>(None, &mut result));
        assert_eq!(result, expected);
        assert_eq!(
            result.inline_aliases(false).as_view().get_data(),
            expected.as_view().get_data()
        );
    }

    #[test]
    fn expansion_resolves_aliased_power_shapes_and_exponents() {
        for base in [parse!("x+y+1"), parse!("x*y*z")] {
            let base_alias = base.alias(false).alias(false);
            for exponent in [2, 3, -2] {
                let number = Atom::num(exponent);
                let exponent_alias = number.alias(false).alias(false);
                for exp in [&number, &exponent_alias] {
                    // Keep the compressed power even when normalization would
                    // otherwise distribute a product base or expose its exponent.
                    let mut input = Atom::new();
                    input.to_pow(base_alias.as_view(), exp.as_view());
                    assert!(!input.is_expanded::<Atom>(None));
                    let mut plain = Atom::new();
                    plain.to_pow(base.as_view(), number.as_view());
                    assert_alias_expands_like_plain(&input, &plain);
                }
            }
        }
    }

    #[test]
    fn expansion_distributes_aliased_sum_factors_at_every_position() {
        let a = parse!("x+y");
        let b = parse!("z+w");
        let z = parse!("u");
        let aliases = [a.alias(false), b.alias(false).alias(false)];
        for order in [[0, 1, 2], [2, 0, 1], [0, 2, 1]] {
            let factors = [&aliases[0], &aliases[1], &z];
            let plain_factors = [&a, &b, &z];
            let mut input = Atom::new();
            let mul = input.to_mul();
            for index in order {
                mul.extend(factors[index].as_view());
            }
            let plain = Atom::mul_many(order.map(|index| plain_factors[index]));
            assert!(!input.is_expanded::<Atom>(None));
            assert_alias_expands_like_plain(&input, &plain);
        }
        let a = a.alias(false);
        let input = (a.pow(2) + &z) * (a.pow(2) + parse!("v"));
        let plain = (parse!("x+y").pow(2) + &z) * (parse!("x+y").pow(2) + parse!("v"));
        assert_alias_expands_like_plain(&input, &plain);
    }

    #[test]
    fn selective_expansion_checks_all_product_factors() {
        let x = parse!("x");
        let unrelated = parse!("z+w");
        let relevant = parse!("x+y");
        let mut input = Atom::new();
        let mul = input.to_mul();
        mul.extend(unrelated.as_view());
        mul.extend(relevant.as_view());
        assert!(!input.is_expanded(Some(&x)));
        assert!(input.expand_in(&x).is_expanded(Some(&x)));
    }

    #[test]
    fn selective_expansion_keeps_unrelated_sum_aliases() {
        let x = parse!("x");
        let relevant = parse!("x+y");
        let unrelated = parse!("z+w");
        let alias = unrelated.alias(false);
        // An unrelated sum first must not hide a later expandable factor.
        let mut input = Atom::new();
        let mul = input.to_mul();
        mul.extend(alias.as_view());
        mul.extend(relevant.alias(false).as_view());
        assert!(!input.is_expanded(Some(&x)));
        let result = input.expand_in(&x);
        assert_eq!(
            result.inline_aliases(false),
            ((&unrelated) * (&relevant)).expand_in(&x)
        );
        assert!(
            result
                .get_aliases()
                .iter()
                .any(|body| body.as_ref() == &unrelated)
        );
        assert!(result.is_expanded(Some(&x)));
    }

    #[test]
    fn unchanged_and_opaque_alias_expansion_preserves_storage() {
        for body in [parse!("x*y"), parse!("x+y"), parse!("f(x+y)")] {
            let alias = body.alias(false);
            let mut result = Atom::new();
            assert!(!alias.expand_into::<Atom>(None, &mut result));
            assert_eq!(result.as_view().get_data(), alias.as_view().get_data());
            assert_eq!(
                alias.expand_num().as_view().get_data(),
                alias.as_view().get_data()
            );
        }
        let opaque = parse!("x+y").alias(true);
        for input in [opaque.pow(3), &opaque * parse!("z"), parse!("2") * &opaque] {
            assert!(input.is_expanded::<Atom>(None));
            assert_eq!(
                input.expand().as_view().get_data(),
                input.as_view().get_data()
            );
            assert_eq!(
                input.expand_num().as_view().get_data(),
                input.as_view().get_data()
            );
        }
        let mut input = Atom::new();
        input.to_pow(
            parse!("x+y").alias(false).as_view(),
            Atom::num(2).alias(true).as_view(),
        );
        assert!(input.is_expanded::<Atom>(None));
        assert_eq!(
            input.expand().as_view().get_data(),
            input.as_view().get_data()
        );
    }

    #[test]
    fn expand_num_distributes_coefficients_over_aliased_sums() {
        let a = parse!("x+y");
        let b = parse!("z+w");
        for input in [
            parse!("2*u") * a.alias(false),
            parse!("-3") * a.alias(false) * b.alias(false),
        ] {
            let plain = input.inline_aliases(false);
            let result = input.expand_num();
            assert_eq!(result.inline_aliases(false), plain.expand_num());
            assert_ne!(result.as_view().get_data(), input.as_view().get_data());
        }
    }

    #[test]
    fn expand_num() {
        let exp = parse!("5+2*v3*(v1-v2)*(v4+v5)").expand_num();
        let res = parse!("5+v3*(v4+v5)*(2*v1-2*v2)");
        assert_eq!(exp, res);
    }

    #[test]
    fn expand_num_sparse_and_collapsed_children() {
        let mut out = parse!("stale+output");
        for (input, expected) in [
            ("2*x*(y+z)", "x*(2*y+2*z)"),
            ("a+2*b*(x+y)+z^2", "a+b*(2*x+2*y)+z^2"),
            ("(2*(x+y))^z", "(2*x+2*y)^z"),
            ("a^(2*(x+y))", "a^(2*x+2*y)"),
            ("b*(c+2*(x+y))", "b*(c+2*x+2*y)"),
            ("2*(c+3*(x+y))", "2*c+6*x+6*y"),
            ("b*(2*(x+y)-2*x-2*y)", "0"),
            ("b*(1+2*(x+y)-2*x-2*y)", "b"),
            ("-3*(x+y)", "-3*x-3*y"),
        ] {
            let input = parse!(input);
            input.as_view().expand_num_into(&mut out);
            assert_eq!(out, parse!(expected), "{input}");
            assert_eq!(out.expand_num(), out, "{input}");
        }
        for expr in ["7", "x", "f(2*(x+y))", "x^y", "x*y", "x+y", "(x+y)*(z+w)"] {
            let input = parse!(expr);
            input.as_view().expand_num_into(&mut out);
            assert_eq!(out, input, "{expr}");
        }
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
        let exp = parse!("(1+v1)^2+(1+v2)^100").expand_in(symbol!("v1"));
        let res = parse!("1+2*v1+v1^2+(v2+1)^100");
        assert_eq!(exp, res);
    }

    #[test]
    fn expand_with_poly() {
        let exp = parse!("(1+v1)^2+(1+v2)^100").expand_via_poly::<u16, _>(parse!("v1"));
        let res = parse!("1+2*v1+v1^2+(v2+1)^100");
        assert_eq!(exp, res);
    }
}
