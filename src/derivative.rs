use std::{
    cell::RefCell,
    ops::{Add, DerefMut, Div, Mul, Sub},
    sync::Arc,
};

use ahash::{HashMap, HashSet};

use numerica::printer::PrintOptions;

use crate::{
    atom::{
        Atom, AtomCore, AtomView, FunctionBuilder, Indeterminate, InlineVar, Symbol,
        representation::{AddView, FunView},
    },
    coefficient::{Coefficient, CoefficientView},
    domains::{Ring, atom::AtomField, integer::Integer, rational::Rational},
    error,
    id::Pattern,
    poly::{
        PolyVariable,
        series::{Series, SeriesDepth, SeriesError},
    },
    state::Workspace,
};

/// The Taylor coefficients `f^(k)(c)/k!` of a function at a point that have been computed,
/// the last derivative `f^(k)(y)` they were computed from, and the values at the point
/// of the functions of `y` that occur in the derivatives.
struct TaylorCoefficients {
    derivative: Atom,
    coefficients: Vec<Atom>,
    values: HashMap<Atom, Atom>,
}

/// The caches of a call of [AtomView::series].
struct SeriesCache {
    /// The variable of the expansion.
    x: Indeterminate,
    /// The point of the expansion.
    expansion_point: Atom,
    /// The expression that is expanded.
    expression: Atom,
    /// The valuation bounds of the terms of the expression, if it is a sum. The driver
    /// needs them to choose the order of the expansion, and the expansion of the sum
    /// to choose the orders of its terms.
    term_bounds: Option<Vec<Option<(Rational, bool)>>>,
    hash_state: ahash::RandomState,
    /// The Taylor coefficients of functions, keyed by the function of the variable `y`
    /// and the point. They are shared by functions that occur several times and by
    /// the expansions to a higher order.
    taylor: HashMap<(Atom, Atom), TaylorCoefficients>,
    /// The hashes of the subexpressions with `x` that occur more than once.
    repeated: HashSet<u64>,
    /// The expansions of the repeated subexpressions, keyed by their hash and the
    /// relative order of the expansion, which determine the expansion.
    expansions: HashMap<(u64, Rational), Vec<(Atom, Series<AtomField>)>>,
}

impl SeriesCache {
    /// Create the caches for the expansion of `e` in `x` around `expansion_point`.
    fn new(e: AtomView, x: &Indeterminate, expansion_point: AtomView) -> Self {
        let mut cache = SeriesCache {
            x: x.clone(),
            expansion_point: expansion_point.to_owned(),
            expression: e.to_owned(),
            term_bounds: None,
            hash_state: crate::utils::thread_local_hash_state(),
            taylor: HashMap::default(),
            repeated: HashSet::default(),
            expansions: HashMap::default(),
        };
        cache.find_repeated(e, &mut HashSet::default());
        cache
    }

    /// Record the hashes of the subexpressions of `e` with `x` that were `seen` before.
    fn find_repeated(&mut self, e: AtomView, seen: &mut HashSet<u64>) {
        if matches!(e, AtomView::Num(_) | AtomView::Var(_)) || !e.contains_indeterminate(&self.x) {
            return;
        }

        let hash = self.hash_state.hash_one(e);
        if !seen.insert(hash) {
            // the subexpressions were visited at the first occurrence
            self.repeated.insert(hash);
            return;
        }

        match e {
            AtomView::Fun(f) => f.iter().for_each(|a| self.find_repeated(a, seen)),
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();
                self.find_repeated(base, seen);
                self.find_repeated(exp, seen);
            }
            AtomView::Mul(m) => m.iter().for_each(|a| self.find_repeated(a, seen)),
            AtomView::Add(a) => a.iter().for_each(|a| self.find_repeated(a, seen)),
            AtomView::Num(_) | AtomView::Var(_) => {}
        }
    }

    /// Get the valuation bounds of the terms of `e`, if it is the expression that is
    /// expanded and the bounds were found.
    fn term_bounds(
        &self,
        e: AtomView,
        x: &Indeterminate,
        expansion_point: AtomView,
    ) -> Option<Vec<Option<(Rational, bool)>>> {
        if *x != self.x || expansion_point != self.expansion_point || e != self.expression.as_view()
        {
            return None;
        }
        self.term_bounds.clone()
    }

    /// Get the key of the expansion of `e` with the settings `info`, if `e` is repeated.
    fn key(
        &self,
        e: AtomView,
        x: &Indeterminate,
        expansion_point: AtomView,
        info: &Series<AtomField>,
    ) -> Option<(u64, Rational)> {
        if self.repeated.is_empty() || *x != self.x || expansion_point != self.expansion_point {
            return None;
        }

        let hash = self.hash_state.hash_one(e);
        self.repeated
            .contains(&hash)
            .then(|| (hash, info.relative_order()))
    }
}

thread_local! {
    /// The caches of the call of [AtomView::series] in progress.
    static SERIES_CACHE: RefCell<Option<SeriesCache>> = const { RefCell::new(None) };
}

/// Enables [SERIES_CACHE] until it is dropped. A previous cache, of an outer call of
/// [AtomView::series] that may expand in another variable, is restored afterwards.
struct SeriesCacheScope {
    previous: Option<SeriesCache>,
}

impl SeriesCacheScope {
    fn new(e: AtomView, x: &Indeterminate, expansion_point: AtomView) -> Self {
        let cache = SeriesCache::new(e, x, expansion_point);
        let previous = SERIES_CACHE.with_borrow_mut(|c| c.replace(cache));
        Self { previous }
    }
}

impl Drop for SeriesCacheScope {
    fn drop(&mut self) {
        let previous = self.previous.take();
        SERIES_CACHE.with_borrow_mut(|c| *c = previous);
    }
}

impl AtomView<'_> {
    /// Take a derivative of the expression with respect to `x`.
    pub(crate) fn derivative(&self, x: &Indeterminate) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut out = ws.new_atom();
            self.derivative_with_ws_into(x, ws, &mut out);
            out.into_inner()
        })
    }

    /// Take a derivative of the expression with respect to `x` and
    /// write the result in `out`.
    /// Returns `true` if the derivative is non-zero.
    pub(crate) fn derivative_into(&self, x: &Indeterminate, out: &mut Atom) -> bool {
        Workspace::get_local().with(|ws| self.derivative_with_ws_into(x, ws, out))
    }

    /// Take a derivative of the expression with respect to `x` and
    /// write the result in `out`.
    /// Returns `true` if the derivative is non-zero.
    pub(crate) fn derivative_with_ws_into(
        &self,
        x: &Indeterminate,
        workspace: &Workspace,
        out: &mut Atom,
    ) -> bool {
        if x == self {
            out.to_num(1);
            return true;
        }

        match self {
            AtomView::Num(_) | AtomView::Var(_) => {
                out.to_num(Coefficient::zero());
                false
            }
            AtomView::Fun(f_orig) => {
                // detect if the function to derive is the derivative function itself.
                // der stores: der(depth_1, ..., depth_n, f, arg_1, ..., arg_n).
                let mut der_function_call = workspace.new_atom();
                let (f, is_der, arg_count) = if f_orig.get_symbol() == Symbol::DERIVATIVE {
                    if f_orig.get_nargs() < 3 || f_orig.get_nargs() % 2 == 0 {
                        error!(
                            "Derivative function {} must contain n depths, a function symbol, and n arguments",
                            self.printer(PrintOptions::file())
                        );
                        out.to_num(Coefficient::Indeterminate);
                        return false;
                    }

                    let der_depth_count = (f_orig.get_nargs() - 1) / 2;
                    let function_symbol = match f_orig.iter().nth(der_depth_count).unwrap() {
                        AtomView::Var(v) => v.get_symbol(),
                        _ => {
                            error!(
                                "Derivative function {} argument must be a function symbol",
                                self.printer(PrintOptions::file())
                            );
                            out.to_num(Coefficient::Indeterminate);
                            return false;
                        }
                    };

                    let function_call = der_function_call.to_fun(function_symbol);
                    for arg in f_orig.iter().skip(der_depth_count + 1) {
                        function_call.add_arg(arg);
                    }

                    (
                        match der_function_call.as_view() {
                            AtomView::Fun(f) => f,
                            _ => unreachable!(),
                        },
                        true,
                        der_depth_count,
                    )
                } else {
                    (*f_orig, false, f_orig.get_nargs())
                };

                // take derivative of all the arguments and store it in a list
                let mut args_der = Vec::with_capacity(f.get_nargs());
                for (i, arg) in f.iter().enumerate() {
                    let mut arg_der = workspace.new_atom();
                    if arg.derivative_with_ws_into(x, workspace, &mut arg_der) {
                        args_der.push((i, arg_der));
                    }
                }

                if args_der.is_empty() {
                    out.to_num(Coefficient::zero());
                    return false;
                }

                // derive special functions
                if f.get_nargs() == 1
                    && [Symbol::LOG, Symbol::SIN, Symbol::COS].contains(&f.get_symbol())
                {
                    let mut fn_der = workspace.new_atom();
                    match f.get_symbol_id() {
                        Symbol::LOG_ID => {
                            let mut n = workspace.new_atom();
                            n.to_num(-1);

                            fn_der.to_pow(f.iter().next().unwrap(), n.as_view());
                        }
                        Symbol::SIN_ID => {
                            let p = fn_der.to_fun(Symbol::COS);
                            p.add_arg(f.iter().next().unwrap());
                        }
                        Symbol::COS_ID => {
                            let mut n = workspace.new_atom();
                            n.to_num(-1);

                            let mut sin = workspace.new_atom();
                            let sin_fun = sin.to_fun(Symbol::SIN);
                            sin_fun.add_arg(f.iter().next().unwrap());

                            let m = fn_der.to_mul();
                            m.extend(sin.as_view());
                            m.extend(n.as_view());
                        }
                        _ => unreachable!(),
                    }

                    let (_, mut arg_der) = args_der.pop().unwrap_or((0, Atom::Zero.into()));
                    if let Atom::Mul(m) = arg_der.deref_mut() {
                        m.extend(fn_der.as_view());
                        arg_der.as_view().normalize(workspace, out);
                    } else {
                        let mut mul = workspace.new_atom();
                        let m = mul.to_mul();
                        m.extend(fn_der.as_view());
                        m.extend(arg_der.as_view());
                        mul.as_view().normalize(workspace, out);
                    }

                    return true;
                }

                if f.get_nargs() == 3 && f.get_symbol_id() == Symbol::IF_ID {
                    let mut fn_der = workspace.new_atom();
                    let der = fn_der.to_fun(Symbol::IF);
                    der.add_arg(f.iter().next().unwrap());

                    if let Some((_, d)) = args_der.iter().find(|(i, _)| *i == 1) {
                        der.add_arg(d.as_view());
                    } else {
                        der.add_arg(Atom::Zero.as_view());
                    };

                    if let Some((_, d)) = args_der.iter().find(|(i, _)| *i == 2) {
                        der.add_arg(d.as_view());
                    } else {
                        der.add_arg(Atom::Zero.as_view());
                    };

                    fn_der.as_view().normalize(workspace, out);
                    return true;
                }

                // create a derivative function that tags which index was derived
                let mut add = workspace.new_atom();
                let a = add.to_add();
                let mut fn_der = workspace.new_atom();
                let mut n = workspace.new_atom();
                let mut mul = workspace.new_atom();
                for (index, arg_der) in args_der {
                    if let Some(custom_der) = &f.get_symbol().get_global_data().custom_derivative {
                        let mut setter = fn_der.deref_mut().into();
                        custom_der(*self, index, &mut setter);
                        if let Some(value) = setter.get() {
                            let m = mul.to_mul();
                            m.extend(value.as_view());
                            m.extend(arg_der.as_view());
                            a.extend(m.as_view());
                            continue;
                        }
                    }

                    let p = fn_der.to_fun(Symbol::DERIVATIVE);

                    if is_der {
                        for (i, x_orig) in f_orig.iter().take(arg_count).enumerate() {
                            n.set_from_view(&x_orig);
                            *n += if i == index { 1 } else { 0 };
                            p.add_arg(n.as_view());
                        }
                    } else {
                        for i in 0..f.get_nargs() {
                            n.to_num((if i == index { 1 } else { 0 }, 1));
                            p.add_arg(n.as_view());
                        }
                    }

                    let function_symbol = InlineVar::new(f.get_symbol());
                    p.add_arg(function_symbol.as_view());
                    for arg in f.iter() {
                        p.add_arg(arg);
                    }

                    let m = mul.to_mul();
                    m.extend(fn_der.as_view());
                    m.extend(arg_der.as_view());
                    mul.as_view().normalize(workspace, out);

                    a.extend(mul.as_view());
                }

                add.as_view().normalize(workspace, out);
                true
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();

                let mut exp_der = workspace.new_atom();
                let exp_der_non_zero = exp.derivative_with_ws_into(x, workspace, &mut exp_der);

                let mut base_der = workspace.new_atom();
                let base_der_non_zero = base.derivative_with_ws_into(x, workspace, &mut base_der);

                if !exp_der_non_zero && !base_der_non_zero {
                    out.to_num(0);
                    return false;
                }

                let mut exp_der_contrib = workspace.new_atom();

                if exp_der_non_zero {
                    // create log(base)
                    let mut log_base = workspace.new_atom();
                    let lb = log_base.to_fun(Symbol::LOG);
                    lb.add_arg(base);

                    if let Atom::Mul(m) = exp_der.deref_mut() {
                        m.extend(*self);
                        m.extend(log_base.as_view());
                        exp_der.as_view().normalize(workspace, &mut exp_der_contrib);
                    } else {
                        let mut mul = workspace.new_atom();
                        let m = mul.to_mul();
                        m.extend(*self);
                        m.extend(exp_der.as_view());
                        m.extend(log_base.as_view());
                        mul.as_view().normalize(workspace, &mut exp_der_contrib);
                    }

                    if !base_der_non_zero {
                        out.set_from_view(&exp_der_contrib.as_view());
                        return true;
                    }
                }

                let mut mul_h = workspace.new_atom();
                let mul = mul_h.to_mul();
                mul.extend(base_der.as_view());

                let mut new_exp = workspace.new_atom();
                if let AtomView::Num(n) = exp {
                    mul.extend(exp);

                    let res = n.get_coeff_view() + -1;
                    new_exp.to_num(res);
                } else {
                    mul.extend(exp);

                    let ao = new_exp.to_add();
                    ao.extend(exp);

                    let mut min_one = workspace.new_atom();
                    min_one.to_num(-1);

                    ao.extend(min_one.as_view());
                }

                let mut pow_h = workspace.new_atom();
                pow_h.to_pow(base, new_exp.as_view());

                mul.extend(pow_h.as_view());

                if exp_der_non_zero {
                    let mut add = workspace.new_atom();
                    let a = add.to_add();

                    a.extend(mul_h.as_view());
                    a.extend(exp_der_contrib.as_view());

                    add.as_view().normalize(workspace, out);
                } else {
                    mul_h.as_view().normalize(workspace, out);
                }

                true
            }
            AtomView::Mul(args) => {
                let mut add_h = workspace.new_atom();
                let add = add_h.to_add();
                let mut mul_h = workspace.new_atom();
                let mut non_zero = false;
                for arg in args.iter() {
                    let mut arg_der = workspace.new_atom();
                    if arg.derivative_with_ws_into(x, workspace, &mut arg_der) {
                        if let Atom::Mul(mm) = arg_der.deref_mut() {
                            for other_arg in args.iter() {
                                if other_arg != arg {
                                    mm.extend(other_arg);
                                }
                            }

                            add.extend(arg_der.as_view());
                        } else {
                            let mm = mul_h.to_mul();
                            mm.extend(arg_der.as_view());
                            for other_arg in args.iter() {
                                if other_arg != arg {
                                    mm.extend(other_arg);
                                }
                            }
                            add.extend(mul_h.as_view());
                        }

                        non_zero = true;
                    }
                }

                if non_zero {
                    add_h.as_view().normalize(workspace, out);
                    true
                } else {
                    out.to_num(0);
                    false
                }
            }
            AtomView::Add(args) => {
                let mut add_h = workspace.new_atom();
                let add = add_h.to_add();
                let mut arg_der = workspace.new_atom();
                let mut non_zero = false;
                for arg in args.iter() {
                    if arg.derivative_with_ws_into(x, workspace, &mut arg_der) {
                        add.extend(arg_der.as_view());
                        non_zero = true;
                    }
                }

                if non_zero {
                    add_h.as_view().normalize(workspace, out);
                    true
                } else {
                    out.to_num(0);
                    false
                }
            }
        }
    }

    /// Series expand in `x` around `expansion_point` to depth `depth`.
    pub(crate) fn series(
        &self,
        x: &Indeterminate,
        expansion_point: AtomView,
        depth: SeriesDepth,
    ) -> Result<Series<AtomField>, SeriesError> {
        self.series_with_pole_order(x, expansion_point, depth, None)
    }

    /// Series expand in `x` around `expansion_point` to depth `depth`.
    ///
    /// Every subexpression is expanded to the same relative order, which the
    /// expansion of a sum maintains by expanding its terms deeper when they cancel.
    /// For an absolute depth, the relative order that the result needs then follows
    /// from its valuation: a pole of order `p` needs `p` more terms and a zero fewer.
    /// The valuation is `-pole_order` when it is given. Otherwise it is bounded from
    /// the structure of the expression, or found with a cheap expansion to relative
    /// order one if the expression may have a pole. If the result still falls short,
    /// for example because a cancellation in a product was missed, the expansion is
    /// repeated deeper. The pole order is ignored for a relative depth.
    pub(crate) fn series_with_pole_order(
        &self,
        x: &Indeterminate,
        expansion_point: AtomView,
        depth: SeriesDepth,
        pole_order: Option<Rational>,
    ) -> Result<Series<AtomField>, SeriesError> {
        let order = depth.order();
        if !depth.is_absolute() && (order.is_negative() || order.is_zero()) {
            return Err(SeriesError::NonPositiveRelativeDepth { depth });
        }

        let _cache = SeriesCacheScope::new(*self, x, expansion_point);

        // Preserve unresolved symbolic coefficients during expansion. Negative
        // exponents below retry only when the trailing coefficient is proven zero.
        let field = AtomField {
            statistical_zero_test: false,
            ..Default::default()
        };
        let variable: Arc<PolyVariable> = Arc::new(x.clone().into());
        let info = |relative_order: &Rational| {
            Series::new(
                &field,
                None,
                variable.clone(),
                expansion_point.to_owned(),
                relative_order.clone(),
            )
        };

        // Keep the relative order positive and on the grid of the requested depth.
        let denominator = Rational::from(order.denominator());
        let grid_step = Rational::from((1.into(), order.denominator()));
        let on_grid = |relative_order: Rational| {
            let units = (relative_order * &denominator).ceil().max(Integer::one());
            Rational::from(units) / &denominator
        };

        // The absolute order that the result must reach, or its relative order.
        let target = if depth.is_absolute() {
            order + &grid_step
        } else {
            order.clone()
        };

        // The relative order to start from when the valuation is unknown. An expansion
        // to a lower order may fail where this one succeeds, for example when the
        // leading terms of the argument of `exp` cancel, so it is also the fallback.
        let fallback_order = order.clone().max(Rational::one()) + &grid_step;

        let mut relative_order = if !depth.is_absolute() {
            target.clone()
        } else if let Some(pole_order) = pole_order {
            on_grid(&target + &pole_order)
        } else if let Some((valuation, _)) = self
            .root_valuation_bound(x, expansion_point)
            .filter(|(valuation, exact)| *exact || !valuation.is_negative())
        {
            on_grid(&target - &valuation)
        } else {
            // The expression may have a pole. Find its valuation with a cheap expansion.
            let probe_order = Rational::one();
            match self.series_impl(x, expansion_point, &info(&probe_order)) {
                Ok(mut probe) if probe.absolute_order() >= target => {
                    probe.truncate_absolute_order(target);
                    return Ok(probe);
                }
                Ok(probe) if !probe.is_zero() => {
                    // The relative order of the probe shows whether the expansion loses
                    // relative order to cancellations that it does not compensate.
                    let loss = (&probe_order - &probe.relative_order()).max(Rational::zero());
                    on_grid(&target - &probe.get_trailing_exponent() + &loss)
                }
                // The leading terms cancel, so that the valuation is unknown.
                _ => fallback_order.clone(),
            }
        };

        let mut retried = false;
        let mut zero_tested = false;
        loop {
            let mut series = match self.series_impl(x, expansion_point, &info(&relative_order)) {
                Ok(series) => series,
                // An error may be caused by a too low order, as for the probe.
                Err(_) if relative_order < fallback_order => {
                    relative_order = fallback_order.clone();
                    retried = true;
                    continue;
                }
                Err(e) => return Err(e),
            };
            if !depth.is_absolute() && !series.is_zero() && series.relative_order() >= *order {
                series.truncate_relative_order(order.clone());
                break Ok(series);
            } else if depth.is_absolute() && series.absolute_order() > *order {
                series.truncate_absolute_order(target);
                break Ok(series);
            }

            let shortfall = if depth.is_absolute() {
                &target - &series.absolute_order()
            } else if !series.is_zero() {
                &target - &series.relative_order()
            } else {
                // An empty truncation alone does not prove the input zero.
                // Check the exact stored coefficients before retrying a
                // relative expansion whose leading term may not exist. The test
                // is expensive and does not depend on the order, so it is done
                // once, when an expansion to the fallback order is still empty.
                if !zero_tested && relative_order >= fallback_order {
                    zero_tested = true;
                    if self
                        .check_zero_with(
                            &crate::atom::ZeroTestSettings::new()
                                .treat_floats_as_exact(true)
                                .numerical_samples(0),
                        )
                        .is_true()
                    {
                        return Ok(info(&target));
                    }
                }

                // An expansion that is still empty this deep is most likely zero.
                if relative_order >= 1000 {
                    series.truncate_relative_order(order.clone());
                    break Ok(series);
                }
                relative_order.clone()
            };

            // Expanding deeper by the shortfall normally suffices. If it did not,
            // for example when the expansion was empty, at least double the order.
            let next = on_grid(&relative_order + &shortfall);
            relative_order = if retried {
                next.max(&relative_order * &2.into())
            } else {
                next
            };
            retried = true;
        }
    }

    /// Find a lower bound on the valuation of the series of `self` in `x` around
    /// `expansion_point` from the structure of the expression, and whether it is
    /// exact. Return `None` when the expression may have a pole that the structure
    /// does not bound, such as the inverse of a sum whose leading terms may cancel
    /// or a function with a custom series expansion.
    fn valuation_bound(
        &self,
        x: &Indeterminate,
        expansion_point: AtomView,
    ) -> Option<(Rational, bool)> {
        match self.valuation_bound_impl(x, expansion_point) {
            ValuationBound::Constant => Some((Rational::zero(), true)),
            ValuationBound::Bound(valuation, exact) => Some((valuation, exact)),
            ValuationBound::Unknown => None,
        }
    }

    /// Find [Self::valuation_bound] of the expression that [Self::series_with_pole_order]
    /// expands. If it is a sum, the bounds of its terms are kept for [Self::term_infos].
    fn root_valuation_bound(
        &self,
        x: &Indeterminate,
        expansion_point: AtomView,
    ) -> Option<(Rational, bool)> {
        let AtomView::Add(args) = self else {
            return self.valuation_bound(x, expansion_point);
        };

        let bounds = args
            .iter()
            .map(|arg| arg.valuation_bound_impl(x, expansion_point))
            .collect::<Vec<_>>();
        let bound = sum_valuation_bound(bounds.iter().cloned());
        SERIES_CACHE.with_borrow_mut(|cache| {
            if let Some(cache) = cache {
                cache.term_bounds = Some(
                    bounds
                        .into_iter()
                        .map(ValuationBound::into_option)
                        .collect(),
                );
            }
        });
        bound.into_option()
    }

    /// Bound the valuation as [Self::valuation_bound] does, in a single pass over the
    /// expression that also finds the subexpressions without `x`.
    fn valuation_bound_impl(&self, x: &Indeterminate, expansion_point: AtomView) -> ValuationBound {
        if *x == *self {
            let valuation = if expansion_point.is_zero() {
                Rational::one()
            } else {
                Rational::zero()
            };
            return ValuationBound::Bound(valuation, true);
        }

        match self {
            AtomView::Num(_) | AtomView::Var(_) => ValuationBound::Constant,
            AtomView::Mul(args) => {
                let mut valuation = Rational::zero();
                let mut exact = true;
                let mut constant = true;
                for arg in args {
                    match arg.valuation_bound_impl(x, expansion_point) {
                        ValuationBound::Constant => {}
                        ValuationBound::Bound(v, e) => {
                            valuation += &v;
                            exact &= e;
                            constant = false;
                        }
                        ValuationBound::Unknown => return ValuationBound::Unknown,
                    }
                }
                if constant {
                    ValuationBound::Constant
                } else {
                    ValuationBound::Bound(valuation, exact)
                }
            }
            AtomView::Add(args) => sum_valuation_bound(
                args.iter()
                    .map(|arg| arg.valuation_bound_impl(x, expansion_point)),
            ),
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();
                let base_bound = base.valuation_bound_impl(x, expansion_point);
                if let AtomView::Num(n) = exp {
                    let (v, exact) = match base_bound {
                        ValuationBound::Constant => return ValuationBound::Constant,
                        ValuationBound::Bound(v, exact) => (v, exact),
                        ValuationBound::Unknown => return ValuationBound::Unknown,
                    };
                    let CoefficientView::Natural(n, d, 0, _) = n.get_coeff_view() else {
                        return ValuationBound::Unknown;
                    };
                    // a negative power of a base that may vanish faster may have a pole
                    if n > 0 || exact {
                        ValuationBound::Bound(v * &Rational::from((n, d)), exact)
                    } else {
                        ValuationBound::Unknown
                    }
                } else {
                    match (base_bound, exp.valuation_bound_impl(x, expansion_point)) {
                        (ValuationBound::Constant, ValuationBound::Constant) => {
                            ValuationBound::Constant
                        }
                        // the exponential of a series without poles is nonzero
                        (ValuationBound::Constant, ValuationBound::Bound(v, _))
                            if !v.is_negative() && !exp.contains_symbol(Symbol::LOG) =>
                        {
                            ValuationBound::Bound(Rational::zero(), true)
                        }
                        _ => ValuationBound::Unknown,
                    }
                }
            }
            AtomView::Fun(f) => {
                // A function of arguments without poles has none.
                let mut constant =
                    !matches!(x, Indeterminate::Symbol(s, _) if *s == f.get_symbol());
                let mut pole = false;
                let mut first_argument = None;
                for arg in f {
                    let (v, e) = match arg.valuation_bound_impl(x, expansion_point) {
                        ValuationBound::Constant => (Rational::zero(), true),
                        ValuationBound::Bound(v, e) => {
                            constant = false;
                            pole |= v.is_negative();
                            (v, e)
                        }
                        ValuationBound::Unknown => {
                            constant = false;
                            pole = true;
                            (Rational::zero(), false)
                        }
                    };
                    if first_argument.is_none() {
                        first_argument = Some((v, e));
                    }
                }

                if constant {
                    return ValuationBound::Constant;
                }

                if f.get_symbol_id() == Symbol::LOG_ID {
                    // log(1+y) = y + ... if y vanishes, and the logarithm has no poles
                    if let Some(AtomView::Add(args)) = f.iter().next()
                        && args.iter().any(|arg| arg.is_one())
                        && let Some(Some((v, exact))) = lowest_valuation_bound(
                            args.iter()
                                .filter(|arg| !arg.is_one())
                                .map(|arg| arg.valuation_bound(x, expansion_point)),
                        )
                        && v > 0
                    {
                        return ValuationBound::Bound(v, exact);
                    }
                    return ValuationBound::Bound(Rational::zero(), false);
                }

                if pole || f.get_symbol().get_series_function().is_some() {
                    return ValuationBound::Unknown;
                }

                // sin(x) = x + ..., cos(x) = 1 + ...
                match (f.get_symbol_id(), first_argument) {
                    (Symbol::SIN_ID, Some((v, exact))) if v > 0 => ValuationBound::Bound(v, exact),
                    (Symbol::COS_ID, Some((v, _))) if v > 0 => {
                        ValuationBound::Bound(Rational::zero(), true)
                    }
                    _ => ValuationBound::Bound(Rational::zero(), false),
                }
            }
        }
    }

    /// Series expand in `x` around `expansion_point` to the relative order of `info`.
    ///
    /// Unless the result is empty, its relative order is at least that of `info`:
    /// sums whose leading terms cancel expand their terms deeper, and functions
    /// whose arguments lose relative order are expanded again. Higher orders that
    /// the result cannot use are not computed where this is cheap to avoid.
    pub(crate) fn series_impl(
        &self,
        x: &Indeterminate,
        expansion_point: AtomView,
        info: &Series<AtomField>,
    ) -> Result<Series<AtomField>, SeriesError> {
        if !self.contains_indeterminate(x) {
            return Ok(info.constant(self.to_owned()));
        }

        if *x == *self {
            // The constant term is exact, which keeps the relative order of differences
            // such as `x^2-p^2` that cancel it.
            let variable = info.monomial(info.get_field().one(), Rational::one());
            if expansion_point.is_zero() {
                return Ok(variable);
            }
            return Ok(Series::sum([
                info.constant_inf_prec(expansion_point.to_owned()),
                variable,
            ]));
        }

        // A subexpression that occurs more than once is expanded once per order.
        let cached = SERIES_CACHE.with_borrow(|cache| {
            let cache = cache.as_ref()?;
            let key = cache.key(*self, x, expansion_point, info)?;
            let series = cache
                .expansions
                .get(&key)
                .and_then(|e| e.iter().find(|(a, _)| a.as_view() == *self))
                .map(|(_, s)| s.clone());
            Some((key, series))
        });

        match cached {
            Some((_, Some(series))) => Ok(series),
            Some((key, None)) => {
                let series = self.series_expand(x, expansion_point, info)?;
                SERIES_CACHE.with_borrow_mut(|cache| {
                    if let Some(cache) = cache {
                        let e = cache.expansions.entry(key).or_default();
                        e.push((self.to_owned(), series.clone()));
                    }
                });
                Ok(series)
            }
            None => self.series_expand(x, expansion_point, info),
        }
    }

    /// Series expand an expression that depends on `x` and is not `x`, see
    /// [Self::series_impl].
    fn series_expand(
        &self,
        x: &Indeterminate,
        expansion_point: AtomView,
        info: &Series<AtomField>,
    ) -> Result<Series<AtomField>, SeriesError> {
        match self {
            AtomView::Num(_) | AtomView::Var(_) => Ok(info.constant(self.to_owned())),
            AtomView::Fun(_) => {
                let series = self.series_function(x, expansion_point, info)?;
                let shortfall = info.relative_order() - series.relative_order();
                if series.is_zero() || shortfall <= 0 {
                    return Ok(series);
                }

                // A function such as `log(1+x)` whose constant term vanishes has a
                // lower relative order than its arguments. Expand them deeper.
                let mut series = self.series_function(
                    x,
                    expansion_point,
                    &with_relative_order(info, &(info.relative_order() + &shortfall)),
                )?;
                limit_relative_order(&mut series, info);
                Ok(series)
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();

                // a power of `x` is a monomial at expansion point zero
                if *x == base
                    && expansion_point.is_zero()
                    && let AtomView::Num(n) = exp
                    && let CoefficientView::Natural(n, d, 0, _) = n.get_coeff_view()
                {
                    return Ok(info.monomial(info.get_field().one(), (n, d).into()));
                }

                let mut base_series = base.series_impl(x, expansion_point, info)?;

                if let AtomView::Num(n) = exp {
                    if let CoefficientView::Natural(n, d, ni, _) = n.get_coeff_view() {
                        if ni != 0 {
                            return Err(SeriesError::UnsupportedComplexExponent {
                                expression: self.to_owned(),
                            });
                        }

                        if n < 0 {
                            // in case of 1/0, grow the expansion depth of the base series
                            // it could be that the base series is exactly zero,
                            // to prevent an infinite loop, we stop the loop at ep^-1000.
                            // Start from at least one, as doubling zero does not grow.
                            let mut current_depth = info.relative_order().max(Rational::one());

                            // Remove exact cancellations before choosing the leading power.
                            // Unresolved function coefficients must survive.
                            base_series.normalize_trailing_coefficient();
                            while base_series.is_zero() && current_depth < 1000 {
                                current_depth = &current_depth * &2.into();
                                let info = Series::new(
                                    info.get_field(),
                                    None,
                                    info.get_variable().clone(),
                                    info.get_expansion_point().clone(),
                                    &current_depth
                                        + &(1.into(), current_depth.denominator()).into(),
                                );

                                base_series = base.series_impl(x, expansion_point, &info)?;
                                base_series.normalize_trailing_coefficient();
                            }
                        }

                        // The power has the relative order of its base. Only a positive
                        // integer power, which may still cancel against other terms, keeps
                        // the extra terms of a base that is known further.
                        if n < 0 || d != 1 {
                            limit_relative_order(&mut base_series, info);
                        }

                        base_series.rpow((n, d).into())
                    } else {
                        Err(SeriesError::UnsupportedLargeExponent {
                            expression: self.to_owned(),
                        })
                    }
                } else {
                    // c^e = exp(log(c)*e) for a base c that does not depend on x,
                    // without expanding the logarithm of the constant series, and
                    // b^c = exp(log(b)*c) for an exponent c that does not depend on x
                    let mut exponent = if !base.contains_indeterminate(x) && !base.is_zero() {
                        let e = exp.series_impl(x, expansion_point, info)?;
                        if matches!(base, AtomView::Var(v) if v.get_symbol() == Symbol::E) {
                            e
                        } else {
                            e.mul_coeff(&FunctionBuilder::new(Symbol::LOG).add_arg(base).finish())
                        }
                    } else if !exp.contains_indeterminate(x) {
                        base_series.log()?.mul_coeff(&exp.to_owned())
                    } else {
                        &base_series.log()? * &exp.series_impl(x, expansion_point, info)?
                    };

                    // The exponential of a series without poles is nonzero, so its relative
                    // order is the absolute order of its argument without the constant term.
                    if !exponent.get_trailing_exponent().is_negative() {
                        exponent.truncate_absolute_order(info.absolute_order());
                    }
                    exponent.exp()
                }
            }
            AtomView::Mul(args) => {
                // At expansion point zero, a power of `x` only shifts the exponents
                // of the product of the other factors.
                let monomial_exponent = |arg: AtomView| -> Option<Rational> {
                    if *x == arg {
                        return Some(Rational::one());
                    }
                    if let AtomView::Pow(p) = arg
                        && let (base, AtomView::Num(n)) = p.get_base_exp()
                        && *x == base
                        && let CoefficientView::Natural(n, d, 0, _) = n.get_coeff_view()
                    {
                        return Some((n, d).into());
                    }
                    None
                };
                let shift_monomials = expansion_point.is_zero()
                    && args.iter().any(|arg| {
                        arg.contains_indeterminate(x) && monomial_exponent(arg).is_none()
                    });

                // A factor without `x` is multiplied into the coefficients of the
                // product of the factors before it, which gives the same coefficients
                // as a product with a constant series. Leading factors without `x`
                // are multiplied into the first series.
                let mut series: Option<Series<AtomField>> = None;
                let mut leading = vec![];
                let mut shift = Rational::zero();
                for arg in args.iter() {
                    if !arg.contains_indeterminate(x) {
                        match series {
                            Some(s) => series = Some(s.mul_coeff(&arg.to_owned())),
                            None => leading.push(arg),
                        }
                        continue;
                    }

                    if shift_monomials && let Some(e) = monomial_exponent(arg) {
                        shift += &e;
                        continue;
                    }

                    let mut s = arg.series_impl(x, expansion_point, info)?;
                    if !leading.is_empty() {
                        s = s.mul_coeff(&Atom::product(leading.drain(..)));
                    }
                    series = Some(match series {
                        Some(prev) => &prev * &s,
                        None => s,
                    });
                }

                // A product with `x` that is not `x` has a factor with `x` that is
                // expanded as a series, as a monomial is only shifted if there is one.
                let series = series.unwrap();
                Ok(if shift.is_zero() {
                    series
                } else {
                    series.mul_monomial(&shift)
                })
            }
            AtomView::Add(args) => {
                // The sum is needed up to its valuation plus the relative order, so a term
                // with a higher valuation than the lowest one is needed to a lower relative
                // order. Terms without `x` are exact, so that they do not limit the order.
                let term_infos = self.term_infos(x, expansion_point, info);
                let mut shared = SharedFactors::new(*args, x);
                let mut terms = Vec::with_capacity(args.get_nargs());
                let mut exact = Vec::with_capacity(args.get_nargs());
                for (i, (arg, term_info)) in args.iter().zip(&term_infos).enumerate() {
                    let term_info = term_info.as_ref().unwrap_or(info);
                    if !arg.contains_indeterminate(x) {
                        exact.push(true);
                        terms.push(info.constant_inf_prec(arg.to_owned()));
                        continue;
                    }
                    exact.push(false);
                    terms.push(shared.expand(i, arg, x, expansion_point, term_info)?);
                }

                // If the leading terms cancel, the sum is known to a lower relative order
                // than its terms. Then expand the terms that limit its absolute order
                // deeper. One retry normally suffices, as the valuation is then known.
                let relative_order = info.relative_order();
                for retry in 0.. {
                    // Without a cancellation of the lowest terms, or a term that is known
                    // to a lower relative order, the sum keeps the relative order.
                    let lowest = terms
                        .iter()
                        .filter(|t| !t.is_zero())
                        .map(|t| t.get_trailing_exponent())
                        .min();
                    let may_fall_short = lowest.is_some_and(|lowest| {
                        let required = &lowest + &relative_order;
                        terms
                            .iter()
                            .filter(|t| !t.is_zero() && t.get_trailing_exponent() == lowest)
                            .count()
                            > 1
                            || terms
                                .iter()
                                .zip(&exact)
                                .any(|(t, e)| !e && t.absolute_order() < required)
                    });
                    if !may_fall_short || retry == 2 {
                        let mut sum = Series::sum(terms);
                        if retry > 0 {
                            limit_relative_order(&mut sum, info);
                        }
                        return Ok(sum);
                    }

                    let mut sum = Series::sum(terms.clone());
                    if !sum.is_zero() && sum.relative_order() >= relative_order {
                        if retry > 0 {
                            limit_relative_order(&mut sum, info);
                        }
                        return Ok(sum);
                    }

                    // If all terms cancel, the valuation is at least the absolute order.
                    let required = if sum.is_zero() {
                        sum.absolute_order() + &relative_order
                    } else {
                        sum.get_trailing_exponent() + &relative_order
                    };
                    for (i, ((term, arg), exact)) in
                        terms.iter_mut().zip(args).zip(&exact).enumerate()
                    {
                        if *exact || term.absolute_order() >= required {
                            continue;
                        }

                        // Without a known valuation, assume that the absolute order of the
                        // term grows with its relative order.
                        let term_order = if term.is_zero() {
                            &relative_order + &required - &term.absolute_order()
                        } else {
                            &required - &term.get_trailing_exponent()
                        };
                        *term = shared.expand(
                            i,
                            arg,
                            x,
                            expansion_point,
                            &with_relative_order(info, &term_order),
                        )?;
                    }
                }
                unreachable!()
            }
        }
    }

    /// For the terms of the sum `self`, return the expansion settings for the relative
    /// order that each term needs, or `None` if it needs the relative order of `info`.
    ///
    /// The valuation of a term is bounded from below by its structure, or found with an
    /// expansion to relative order one if its structure allows a pole. A term needs the
    /// relative order of the sum plus the difference between the lowest valuation and
    /// its own. If the lowest terms cancel, the expansion of the sum expands the terms
    /// that fall short deeper.
    fn term_infos(
        &self,
        x: &Indeterminate,
        expansion_point: AtomView,
        info: &Series<AtomField>,
    ) -> Vec<Option<Series<AtomField>>> {
        let AtomView::Add(args) = self else {
            unreachable!("not a sum");
        };

        let relative_order = info.relative_order();
        if relative_order <= 1 {
            return vec![None; args.get_nargs()];
        }

        // The exact valuation of every term, or a lower bound if it is unknown.
        let bounds = SERIES_CACHE.with_borrow(|cache| {
            cache
                .as_ref()
                .and_then(|cache| cache.term_bounds(*self, x, expansion_point))
        });
        let probe_info = with_relative_order(info, &Rational::one());
        let valuations = args
            .iter()
            .enumerate()
            .map(|(i, arg)| {
                let bound = match &bounds {
                    Some(bounds) => bounds[i].clone(),
                    None => arg.valuation_bound(x, expansion_point),
                };
                if let Some((v, true)) = bound {
                    return Some((v, true));
                }
                match arg.series_impl(x, expansion_point, &probe_info) {
                    Ok(s) if !s.is_zero() => Some((s.get_trailing_exponent(), true)),
                    _ => bound,
                }
            })
            .collect::<Vec<_>>();

        // The valuation of the sum, unless its lowest terms cancel
        let Some(lowest) = valuations
            .iter()
            .flatten()
            .filter(|(_, exact)| *exact)
            .map(|(v, _)| v)
            .min()
            .cloned()
        else {
            return vec![None; args.get_nargs()];
        };

        valuations
            .iter()
            .map(|v| {
                let (v, _) = v.as_ref().filter(|(v, _)| *v > lowest)?;
                // Keep a positive relative order, so that the term still bounds the order
                // of the sum.
                let term_order = (&relative_order + &lowest - v)
                    .max(Rational::from((1, info.get_ramification() as i64)));
                Some(with_relative_order(info, &term_order))
            })
            .collect()
    }

    /// Expand a function `f` with a single argument that depends on `x`, around the constant
    /// term `c` of the series of that argument: `f(c + p) = sum_k f^(k)(c)/k! p^k`.
    ///
    /// The derivatives `f^(k)(y)` are taken with respect to a fresh variable `y`, so that
    /// they do not contain the derivatives of the argument, and only as many as the powers
    /// of `p` below the order of the argument require. The coefficients `f^(k)(c)/k!` are
    /// shared within a call of [AtomView::series].
    ///
    /// The derivatives are not collected in `f`, as derivatives of the collected products
    /// nest and grow exponentially (`gamma`), and they are not expanded, as that grows the
    /// powers of sums exponentially (`atan`).
    ///
    /// Returns `None` when the function has several arguments that depend on `x`, when the
    /// constant term is unknown or depends on `x`, for `if`, `conj` and `abs`, and when
    /// `f` of the fresh variable does not normalize to a call of `f`.
    fn function_taylor_series(
        f: FunView,
        args_series: &[Series<AtomField>],
        x: &Indeterminate,
    ) -> Option<Series<AtomField>> {
        // conj and abs are not holomorphic: the old path lets their normalization act on the
        // expanded argument (conj(x) = x for a real x), which a fresh variable cannot do.
        if [Symbol::IF, Symbol::CONJ, Symbol::ABS].contains(&f.get_symbol()) {
            return None;
        }

        let mut dependent = f
            .iter()
            .enumerate()
            .filter(|(_, a)| a.contains_indeterminate(x));
        let (index, _) = dependent.next()?;
        if dependent.next().is_some() {
            return None;
        }

        let arg = &args_series[index];
        if arg
            .terms()
            .any(|(_, c)| c.as_view().contains_indeterminate(x))
        {
            return None;
        }
        let constant = arg.coefficient(Rational::zero())?;

        let y = Indeterminate::from(crate::symbol!("symbolica::series_taylor_variable"));
        if f.iter().any(|a| a.contains_indeterminate(&y)) {
            return None;
        }
        let y_atom = Atom::from(y.clone());

        let mut f_y = FunctionBuilder::new(f.get_symbol());
        for (i, a) in f.iter().enumerate() {
            f_y = if i == index {
                f_y.add_arg(&y_atom)
            } else {
                f_y.add_arg(a)
            };
        }
        let key = (f_y.finish(), constant);

        // The normalization of f must not depend on the form of the argument.
        if !matches!(key.0.as_view(), AtomView::Fun(g) if g.get_symbol() == f.get_symbol()) {
            return None;
        }

        // Take the entry out of the cache while it is extended, so that the cache is not
        // borrowed while the derivatives are normalized.
        let mut taylor = SERIES_CACHE
            .with_borrow_mut(|cache| cache.as_mut().and_then(|cache| cache.taylor.remove(&key)))
            .unwrap_or_else(|| TaylorCoefficients {
                derivative: key.0.clone(),
                coefficients: vec![],
                values: HashMap::default(),
            });

        let series = arg.compose_taylor(|k| {
            while taylor.coefficients.len() <= k {
                let i = taylor.coefficients.len();
                if i > 0 && !taylor.derivative.is_zero() {
                    // Distribute numbers over sums, such as the 1/2 in the derivative of
                    // bessel_j, as nested sums double in size with every derivative.
                    taylor.derivative = taylor.derivative.as_view().derivative(&y).expand_num();
                }

                // Substitute y = c. The derivatives repeat the same functions of y, such as
                // polygamma(k, y), in many terms, so these are evaluated only once.
                let values = &mut taylor.values;
                let value = taylor.derivative.replace_map(|term, _, out| {
                    if term == y_atom.as_view() {
                        **out = key.1.clone();
                    } else if matches!(term, AtomView::Fun(_)) && term.contains_indeterminate(&y) {
                        **out = values
                            .entry(term.to_owned())
                            .or_insert_with(|| {
                                term.replace(Pattern::Literal(y_atom.clone()))
                                    .with(Pattern::Literal(key.1.clone()))
                            })
                            .clone();
                    }
                });
                taylor.coefficients.push(if i > 1 {
                    value / Atom::num(Integer::factorial(i as u32))
                } else {
                    value
                });
            }
            taylor.coefficients[k].clone()
        });

        SERIES_CACHE.with_borrow_mut(|cache| {
            if let Some(cache) = cache {
                cache.taylor.insert(key, taylor);
            }
        });

        Some(series)
    }

    /// Expand `self`, an argument of a function with a custom series whose expansion is
    /// `series`, deeper until its terms beyond the constant term are known to the
    /// relative order of `info`. The custom series may subtract the constant term, such
    /// as the pole of `gamma` at `-k`, and expand the truncated rest as if it were exact.
    /// Return whether the argument was expanded again.
    fn expand_beyond_constant_term(
        &self,
        x: &Indeterminate,
        expansion_point: AtomView,
        info: &Series<AtomField>,
        series: &mut Series<AtomField>,
    ) -> Result<bool, SeriesError> {
        if !self.contains_indeterminate(x)
            || series.is_zero()
            || !series.get_trailing_exponent().is_zero()
        {
            return Ok(false);
        }

        let relative_order = info.relative_order();
        let mut depth = relative_order.clone().max(Rational::one());
        let mut expanded = false;
        loop {
            // the lowest power of the terms beyond the constant term
            let rest = series
                .terms()
                .find(|(e, c)| *e > 0 && !c.is_zero())
                .map(|(e, _)| e);
            depth = match rest {
                Some(v) if series.absolute_order() >= &v + &relative_order => break,
                _ if depth >= 1000 => break,
                Some(v) => (&v + &relative_order).max(&depth + &Rational::one()),
                None => &depth * &2.into(),
            };
            *series = self.series_impl(x, expansion_point, &with_relative_order(info, &depth))?;
            expanded = true;
        }
        Ok(expanded)
    }

    /// Series expand a function, see [Self::series_impl].
    fn series_function(
        &self,
        x: &Indeterminate,
        expansion_point: AtomView,
        info: &Series<AtomField>,
    ) -> Result<Series<AtomField>, SeriesError> {
        let AtomView::Fun(f) = self else {
            unreachable!("not a function");
        };

        let symbol = f.get_symbol();

        // The logarithm and functions with a custom series expansion, which may have
        // poles, need the leading term of their arguments. Expand an argument whose
        // terms all cancel deeper, as for the base of a negative power.
        let needs_leading_term =
            f.get_symbol_id() == Symbol::LOG_ID || symbol.get_series_function().is_some();

        // log(1+y) = y + ... needs y up to its valuation plus the relative order.
        let deeper_info;
        let info = if f.get_symbol_id() == Symbol::LOG_ID
            && let Some((valuation, _)) = self.valuation_bound(x, expansion_point)
            && valuation > 0
        {
            deeper_info = with_relative_order(info, &(info.relative_order() + &valuation));
            &deeper_info
        } else {
            info
        };

        let mut args_series = Vec::with_capacity(f.get_nargs());
        for arg in f {
            let mut arg_series = arg.series_impl(x, expansion_point, info)?;
            if needs_leading_term && arg_series.is_zero() && arg.contains_indeterminate(x) {
                // Start from at least one, as doubling zero does not grow.
                let mut current_depth = info.relative_order().max(Rational::one());
                while arg_series.is_zero() && current_depth < 1000 {
                    current_depth = &current_depth * &2.into();
                    arg_series = arg.series_impl(
                        x,
                        expansion_point,
                        &with_relative_order(info, &current_depth),
                    )?;
                    arg_series.normalize_trailing_coefficient();
                }
                limit_relative_order(&mut arg_series, info);
            }
            args_series.push(arg_series);
        }

        if args_series.is_empty() {
            return Ok(info.constant(f.to_owned().into()));
        }

        if symbol == Symbol::IF && f.iter().skip(1).any(|arg| arg.contains_indeterminate(x)) {
            return Err(SeriesError::NonConstantIf {
                expression: self.to_owned(),
            });
        }

        // An argument without terms that is known up to a negative order may or may not
        // have a pole, so it is not filled in: f(0) is not the value of the function.
        if !f.get_symbol().is_fixed_builtin()
            && args_series
                .iter()
                .any(|x| !x.is_zero() && x.get_trailing_exponent().is_negative())
        {
            // fill in the expanded arguments, perhaps the leading negative exponent will be popped out,
            // in which case we can proceed
            let mut f_eval = FunctionBuilder::new(f.get_symbol());
            for c in &args_series {
                f_eval = f_eval.add_arg(c.to_atom());
            }
            let a = f_eval.finish();

            if !matches!(a, Atom::Fun(_)) {
                return a.as_view().series_impl(x, expansion_point, info);
            }
        }

        match f.get_symbol_id() {
            Symbol::COS_ID | Symbol::SIN_ID => {
                let mut arg = args_series.swap_remove(0);
                // If the function does not vanish at the constant term of the argument,
                // the result is nonzero and the argument is needed only up to the
                // relative order of the expansion.
                let valuation = arg.get_trailing_exponent();
                if !valuation.is_negative() {
                    let constant = if valuation.is_zero() {
                        arg.get_trailing_coefficient()
                    } else {
                        Atom::Zero
                    };
                    if !FunctionBuilder::new(symbol)
                        .add_arg(&constant)
                        .finish()
                        .is_zero()
                    {
                        arg.truncate_absolute_order(info.absolute_order());
                    }
                }

                if f.get_symbol_id() == Symbol::COS_ID {
                    arg.cos()
                } else {
                    arg.sin()
                }
            }
            Symbol::LOG_ID => args_series[0].log(),
            _ => {
                if let Some(custom_series) = symbol.get_series_function()
                    && let Some((singular, regularized)) = {
                        // A custom series may subtract the constant term of an argument,
                        // such as the pole of gamma at -k, and treat the truncated rest as
                        // exact, so the rest must be known to the relative order.
                        let mut parts = custom_series(&args_series);
                        let mut expanded = false;
                        if parts.is_some() {
                            for (arg, arg_series) in f.iter().zip(&mut args_series) {
                                expanded |= arg.expand_beyond_constant_term(
                                    x,
                                    expansion_point,
                                    info,
                                    arg_series,
                                )?;
                            }
                        }
                        if expanded {
                            parts = custom_series(&args_series);
                        }
                        parts
                    }
                {
                    // Both parts keep the relative order of the expansion, and so does
                    // their product.
                    let singular_series =
                        singular.as_view().series_impl(x, expansion_point, info)?;
                    let regularized_series =
                        regularized
                            .as_view()
                            .series_impl(x, expansion_point, info)?;
                    return Ok(&singular_series * &regularized_series);
                }

                // TODO: also check for log(x)
                if args_series
                    .iter()
                    .any(|x| x.get_trailing_exponent().is_negative())
                {
                    return Err(SeriesError::FunctionArgumentPole {
                        expression: self.to_owned(),
                    });
                }

                if let Some(series) = Self::function_taylor_series(*f, &args_series, x) {
                    return Ok(series);
                }

                let mut f_eval = FunctionBuilder::new(f.get_symbol());
                for c in &args_series {
                    f_eval = f_eval.add_arg(c.to_atom());
                }
                let a = f_eval.finish();

                let constant = a
                    .replace(Pattern::Literal(Atom::from(x.clone())))
                    .with(Pattern::Literal(expansion_point.to_owned()));

                // The Taylor expansion is known up to the lowest absolute order of the
                // arguments that depend on `x`. It is needed up to the relative order of
                // the expansion beyond its valuation, which is found on the way. Search
                // for the valuation only up to the relative order.
                let argument_order = f
                    .iter()
                    .zip(&args_series)
                    .filter(|(arg, _)| arg.contains_indeterminate(x))
                    .map(|(_, s)| s.absolute_order())
                    .min()
                    .unwrap();
                let order_for_valuation = |valuation: Option<u32>| {
                    let v = Rational::from(valuation.unwrap_or(0));
                    argument_order.clone().min(info.absolute_order() + &v)
                };
                let mut valuation = (!constant.is_zero()).then_some(0);
                let mut order = order_for_valuation(valuation);

                let mut terms = vec![];
                if !constant.is_zero() {
                    terms.push(info.constant(constant));
                }

                // TODO: do not allow series expansion in IF if it is not a constant!

                let mut d = a;
                let mut i = 1u32;
                // Keep searching for the valuation up to the order of the arguments, as
                // otherwise atan(x) at relative order one would come out empty.
                while order > i || (valuation.is_none() && argument_order > i) {
                    // Collect repeated calls before differentiating again, to prevent exponential growth.
                    d = d.as_view().derivative(x).collect_symbol::<i32>(symbol);

                    if d.is_zero() {
                        break;
                    }

                    let rep = d
                        .replace(Pattern::Literal(Atom::from(x.clone())))
                        .with(Pattern::Literal(expansion_point.to_owned()));

                    if !rep.is_zero() {
                        if valuation.is_none() {
                            valuation = Some(i);
                            order = order_for_valuation(valuation);
                        }

                        terms.push(
                            info.monomial(info.get_field().one(), i.into())
                                .mul_coeff(&rep)
                                .div_coeff(&Atom::num(Integer::factorial(i))),
                        );
                    }
                    i += 1;
                }

                // An empty series that carries the order of the result.
                terms.push(Series::new(
                    info.get_field(),
                    None,
                    info.get_variable(),
                    info.get_expansion_point(),
                    order.clone(),
                ));
                let result = Series::sum(terms);
                Ok(result)
            }
        }
    }
}

/// The terms of a sum that differ only in their factors without `x` share the
/// expansion of their factors with `x`: `c_1*g + c_2*g` expands `g` once. Each term
/// multiplies it by its own factors without `x`, so that the coefficients of terms that
/// cancel, such as `a*y + b*y - a*y - b*y`, stay syntactically equal.
struct SharedFactors {
    /// For every term, its group of terms with the same factors with `x` and the
    /// product of its factors without `x`, if it shares its factors with another term.
    groups: Vec<Option<(usize, Option<Atom>)>>,
    /// The product of the factors with `x` of every group.
    factors: Vec<Atom>,
    /// The expansions of the factors with `x`, per group and relative order.
    expansions: HashMap<(usize, Rational), Series<AtomField>>,
}

impl SharedFactors {
    /// Find the terms of the sum `args` that share their factors with `x`.
    fn new(args: AddView, x: &Indeterminate) -> Self {
        // the factors with `x` and the factors without `x` of every term
        let mut members: HashMap<Vec<AtomView>, Vec<usize>> = HashMap::default();
        let mut constants = Vec::with_capacity(args.get_nargs());
        for (i, arg) in args.iter().enumerate() {
            let (factors, constant) = match arg {
                AtomView::Mul(m) => m.iter().partition(|a| a.contains_indeterminate(x)),
                _ if arg.contains_indeterminate(x) => (vec![arg], vec![]),
                _ => (vec![], vec![]),
            };
            if !factors.is_empty() {
                members.entry(factors).or_default().push(i);
            }
            constants.push(constant);
        }

        let mut groups = vec![None; args.get_nargs()];
        let mut factors = vec![];
        for (shared, terms) in members {
            if terms.len() < 2 {
                continue;
            }

            for i in terms {
                let constant = std::mem::take(&mut constants[i]);
                let constant = (!constant.is_empty()).then(|| Atom::product(constant));
                groups[i] = Some((factors.len(), constant));
            }
            factors.push(Atom::product(shared));
        }

        SharedFactors {
            groups,
            factors,
            expansions: HashMap::default(),
        }
    }

    /// Series expand the term `arg` with index `i` of the sum to the relative order
    /// of `info`.
    fn expand(
        &mut self,
        i: usize,
        arg: AtomView,
        x: &Indeterminate,
        expansion_point: AtomView,
        info: &Series<AtomField>,
    ) -> Result<Series<AtomField>, SeriesError> {
        let Some((group, constant)) = &self.groups[i] else {
            return arg.series_impl(x, expansion_point, info);
        };

        let key = (*group, info.relative_order());
        let series = match self.expansions.get(&key) {
            Some(series) => series.clone(),
            None => {
                let series =
                    self.factors[*group]
                        .as_view()
                        .series_impl(x, expansion_point, info)?;
                self.expansions.insert(key, series.clone());
                series
            }
        };

        Ok(match constant {
            Some(constant) => series.mul_coeff(constant),
            None => series,
        })
    }
}

/// A lower bound on the valuation of a series, found from the structure of its expression.
#[derive(Clone)]
enum ValuationBound {
    /// The expression does not depend on the variable.
    Constant,
    /// The lowest exponent is at least the valuation, and equal to it if it is exact.
    Bound(Rational, bool),
    /// The expression may have a pole that its structure does not bound.
    Unknown,
}

impl ValuationBound {
    /// Get the bound as `(valuation, exact)`, or `None` if it is unknown.
    fn into_option(self) -> Option<(Rational, bool)> {
        match self {
            ValuationBound::Constant => Some((Rational::zero(), true)),
            ValuationBound::Bound(v, exact) => Some((v, exact)),
            ValuationBound::Unknown => None,
        }
    }
}

/// Combine the valuation bounds of the terms of a sum into a bound for the sum.
fn sum_valuation_bound(bounds: impl Iterator<Item = ValuationBound>) -> ValuationBound {
    let mut constant = true;
    let lowest = lowest_valuation_bound(bounds.map(|bound| {
        constant &= matches!(bound, ValuationBound::Constant);
        bound.into_option()
    }));
    match lowest {
        _ if constant => ValuationBound::Constant,
        Some(Some((v, exact))) => ValuationBound::Bound(v, exact),
        _ => ValuationBound::Unknown,
    }
}

/// Combine valuation bounds of terms into a bound for their sum: the lowest bound,
/// which is exact if a single exact term has it. Return `None` if a bound is `None`,
/// and `Some(None)` for an empty sum.
fn lowest_valuation_bound(
    bounds: impl Iterator<Item = Option<(Rational, bool)>>,
) -> Option<Option<(Rational, bool)>> {
    let mut lowest: Option<(Rational, bool)> = None;
    for bound in bounds {
        let (v, e) = bound?;
        match &mut lowest {
            Some((l, exact)) if v == *l => *exact = false,
            Some((l, _)) if v > *l => {}
            _ => lowest = Some((v, e)),
        }
    }
    Some(lowest)
}

/// Return a copy of `info` with at least the relative order `relative_order`,
/// rounded up to the grid of `info`.
fn with_relative_order(info: &Series<AtomField>, relative_order: &Rational) -> Series<AtomField> {
    let ramification = Integer::from(info.get_ramification() as i64);
    let units = (relative_order * &Rational::from(&ramification)).ceil();
    Series::new(
        info.get_field(),
        None,
        info.get_variable(),
        info.get_expansion_point(),
        Rational::from((units, ramification)),
    )
}

/// Truncate a nonempty `series` to the relative order of `info`, if it is known further.
fn limit_relative_order(series: &mut Series<AtomField>, info: &Series<AtomField>) {
    if !series.is_zero() && series.relative_order() > info.relative_order() {
        series.truncate_relative_order(info.relative_order());
    }
}

impl Mul<&Atom> for Series<AtomField> {
    type Output = Result<Series<AtomField>, SeriesError>;

    fn mul(self, rhs: &Atom) -> Result<Series<AtomField>, SeriesError> {
        (&self) * rhs
    }
}

impl Mul<&Series<AtomField>> for &Atom {
    type Output = Result<Series<AtomField>, SeriesError>;

    fn mul(self, rhs: &Series<AtomField>) -> Result<Series<AtomField>, SeriesError> {
        rhs * self
    }
}

impl Mul<&Series<AtomField>> for Atom {
    type Output = Result<Series<AtomField>, SeriesError>;

    fn mul(self, rhs: &Series<AtomField>) -> Result<Series<AtomField>, SeriesError> {
        rhs * &self
    }
}

impl Mul<&Atom> for &Series<AtomField> {
    type Output = Result<Series<AtomField>, SeriesError>;

    fn mul(self, rhs: &Atom) -> Result<Series<AtomField>, SeriesError> {
        let x = match self.get_variable().as_ref() {
            PolyVariable::Symbol(x) => Indeterminate::from(*x),
            PolyVariable::Function(_, x) => Indeterminate::try_from(x.clone()).map_err(|_| {
                SeriesError::NonIndeterminateSeriesVariable {
                    variable: self.get_variable().as_ref().clone(),
                }
            })?,
            PolyVariable::Power(_) | PolyVariable::Temporary(_) => {
                return Err(SeriesError::NonIndeterminateSeriesVariable {
                    variable: self.get_variable().as_ref().clone(),
                });
            }
        };

        let expansion_point = self.get_expansion_point();
        let mut current_depth = self.relative_order();

        if current_depth.is_zero() {
            current_depth = (2, 1).into();
        }

        loop {
            let info = Series::new(
                self.get_field(),
                None,
                self.get_variable().clone(),
                expansion_point.to_owned(),
                current_depth.clone(),
            );

            let series = rhs
                .as_view()
                .series_impl(&x, expansion_point.as_view(), &info)?
                * self;
            if series.relative_order() >= self.relative_order() {
                return Ok(series);
            } else {
                // increase the expansion depth
                current_depth = &current_depth * &2.into();
            }
        }
    }
}

impl Add<&Atom> for Series<AtomField> {
    type Output = Result<Series<AtomField>, SeriesError>;

    fn add(self, rhs: &Atom) -> Result<Series<AtomField>, SeriesError> {
        (&self) + rhs
    }
}

impl Add<&Series<AtomField>> for &Atom {
    type Output = Result<Series<AtomField>, SeriesError>;

    fn add(self, rhs: &Series<AtomField>) -> Result<Series<AtomField>, SeriesError> {
        rhs + self
    }
}

impl Add<&Series<AtomField>> for Atom {
    type Output = Result<Series<AtomField>, SeriesError>;

    fn add(self, rhs: &Series<AtomField>) -> Result<Series<AtomField>, SeriesError> {
        rhs + &self
    }
}

impl Add<&Atom> for &Series<AtomField> {
    type Output = Result<Series<AtomField>, SeriesError>;

    fn add(self, rhs: &Atom) -> Result<Series<AtomField>, SeriesError> {
        let x = match self.get_variable().as_ref() {
            PolyVariable::Symbol(x) => Indeterminate::from(*x),
            PolyVariable::Function(_, x) => Indeterminate::try_from(x.clone()).map_err(|_| {
                SeriesError::NonIndeterminateSeriesVariable {
                    variable: self.get_variable().as_ref().clone(),
                }
            })?,
            PolyVariable::Power(_) | PolyVariable::Temporary(_) => {
                return Err(SeriesError::NonIndeterminateSeriesVariable {
                    variable: self.get_variable().as_ref().clone(),
                });
            }
        };

        let expansion_point = self.get_expansion_point();
        let mut current_depth = self.relative_order();

        if current_depth.is_zero() {
            current_depth = (2, 1).into();
        }

        loop {
            let info = Series::new(
                self.get_field(),
                None,
                self.get_variable().clone(),
                expansion_point.to_owned(),
                current_depth.clone(),
            );

            let series = rhs
                .as_view()
                .series_impl(&x, expansion_point.as_view(), &info)?
                + self.clone();
            if series.absolute_order() >= self.absolute_order() {
                return Ok(series);
            } else {
                // increase the expansion depth
                current_depth = &current_depth * &2.into();
            }
        }
    }
}

impl Div<&Atom> for Series<AtomField> {
    type Output = Result<Series<AtomField>, SeriesError>;

    fn div(self, rhs: &Atom) -> Result<Series<AtomField>, SeriesError> {
        (&self) / rhs
    }
}

impl Div<&Series<AtomField>> for &Atom {
    type Output = Result<Series<AtomField>, SeriesError>;

    fn div(self, rhs: &Series<AtomField>) -> Result<Series<AtomField>, SeriesError> {
        rhs.rpow((-1, 1).into())? * self
    }
}

impl Div<&Series<AtomField>> for Atom {
    type Output = Result<Series<AtomField>, SeriesError>;

    fn div(self, rhs: &Series<AtomField>) -> Result<Series<AtomField>, SeriesError> {
        rhs.rpow((-1, 1).into())? * &self
    }
}

impl Div<&Atom> for &Series<AtomField> {
    type Output = Result<Series<AtomField>, SeriesError>;

    fn div(self, rhs: &Atom) -> Result<Series<AtomField>, SeriesError> {
        let x = match self.get_variable().as_ref() {
            PolyVariable::Symbol(x) => Indeterminate::from(*x),
            PolyVariable::Function(_, x) => Indeterminate::try_from(x.clone()).map_err(|_| {
                SeriesError::NonIndeterminateSeriesVariable {
                    variable: self.get_variable().as_ref().clone(),
                }
            })?,
            PolyVariable::Power(_) | PolyVariable::Temporary(_) => {
                return Err(SeriesError::NonIndeterminateSeriesVariable {
                    variable: self.get_variable().as_ref().clone(),
                });
            }
        };

        let expansion_point = self.get_expansion_point();
        let mut current_depth = self.relative_order();

        if current_depth.is_zero() {
            current_depth = (2, 1).into();
        }

        loop {
            let info = Series::new(
                self.get_field(),
                None,
                self.get_variable().clone(),
                expansion_point.to_owned(),
                current_depth.clone(),
            );

            let series = self
                / &rhs
                    .as_view()
                    .series_impl(&x, expansion_point.as_view(), &info)?;
            if series.relative_order() >= self.relative_order() {
                return Ok(series);
            } else {
                // increase the expansion depth
                current_depth = &current_depth * &2.into();
            }
        }
    }
}

impl Sub<&Atom> for Series<AtomField> {
    type Output = Result<Series<AtomField>, SeriesError>;

    fn sub(self, rhs: &Atom) -> Result<Series<AtomField>, SeriesError> {
        (&self) + &(-rhs)
    }
}

impl Sub<&Series<AtomField>> for &Atom {
    type Output = Result<Series<AtomField>, SeriesError>;

    fn sub(self, rhs: &Series<AtomField>) -> Result<Series<AtomField>, SeriesError> {
        -rhs.clone() + self
    }
}

impl Sub<&Series<AtomField>> for Atom {
    type Output = Result<Series<AtomField>, SeriesError>;

    fn sub(self, rhs: &Series<AtomField>) -> Result<Series<AtomField>, SeriesError> {
        -rhs.clone() + &self
    }
}

impl Sub<&Atom> for &Series<AtomField> {
    type Output = Result<Series<AtomField>, SeriesError>;

    fn sub(self, rhs: &Atom) -> Result<Series<AtomField>, SeriesError> {
        self + &(-rhs)
    }
}

#[cfg(test)]
mod test {
    use crate::{
        atom::{Atom, AtomCore, AtomView},
        parse,
        poly::series::{Series, SeriesDepth},
        symbol,
    };

    #[test]
    fn derivative() {
        let v1 = symbol!("v1");
        let inputs = [
            "(1+2*v1)^(5+v1)",
            "log(2*v1) + exp(3*v1) + sin(4*v1) + cos(y*v1)",
            "f(v1^2,v1)",
            "der(0,1,f,v1,v1^3)",
        ];
        let r = inputs.map(|input| parse!(input).derivative(v1));

        let res = [
            "(2*v1+1)^(v1+5)*log(2*v1+1)+2*(v1+5)*(2*v1+1)^(v1+4)",
            "2*(2*v1)^-1+3*exp(3*v1)+4*cos(4*v1)-y*sin(v1*y)",
            "der(0,1,f,v1^2,v1)+2*v1*der(1,0,f,v1^2,v1)",
            "der(1,1,f,v1,v1^3)+3*v1^2*der(0,2,f,v1,v1^3)",
        ];
        let res = res.map(|input| parse!(input));

        assert_eq!(r, res);
    }

    #[test]
    fn series() {
        let v1 = symbol!("v1");

        let input = parse!("exp(v1^2+1)*log(v1+3)/v1/(v1+1)");
        let t = input
            .series(v1, Atom::num(0).as_view(), SeriesDepth::absolute(2))
            .unwrap()
            .to_atom();

        let res = parse!(
            "1/3*exp(1)+v1*(-7/18*exp(1)+2*exp(1)*log(3))+v1^2*(119/162*exp(1)-2*exp(1)*log(3))-exp(1)*log(3)+v1^-1*exp(1)*log(3)"
        );
        assert_eq!(t, res);
    }

    #[test]
    fn series_shift() {
        let v1 = symbol!("v1");
        let input = parse!("1/(v1+1)");
        let t = input
            .series(v1, Atom::num(-1).as_view(), SeriesDepth::absolute(5))
            .unwrap()
            .to_atom();

        let res = parse!("1/(v1+1)");
        assert_eq!(t, res);
    }

    #[test]
    fn series_spurious_pole() {
        let v1 = symbol!("v1");
        let input = parse!("(1-cos(v1))/sin(v1)");
        let t = input
            .series(v1, Atom::num(0).as_view(), SeriesDepth::absolute(5))
            .unwrap()
            .to_atom();

        let res = parse!("1/2*v1+1/24*v1^3+1/240*v1^5");
        assert_eq!(t, res);
    }

    #[test]
    fn series_logx() {
        let v1 = symbol!("v1");
        let input = parse!("log(v1)*(1+v1)");
        let t = input
            .series(v1, Atom::num(0).as_view(), SeriesDepth::absolute(4))
            .unwrap()
            .to_atom();

        let res = parse!("log(v1)+v1*log(v1)");
        assert_eq!(t, res);
    }

    #[test]
    fn series_sqrt() {
        let v1 = symbol!("v1");
        let input = parse!("(v1^3+v1+1)^(1/2)");
        let t = input
            .series(v1, Atom::num(0).as_view(), SeriesDepth::absolute(4))
            .unwrap()
            .to_atom();

        let res = parse!("1+1/2*v1-1/8*v1^2+9/16*v1^3-37/128*v1^4");
        assert_eq!(t, res);
    }

    #[test]
    fn series_fractions() {
        let v1 = symbol!("v1");
        let input = parse!("1/v1^5");

        let t = input
            .series(v1, Atom::num(0).as_view(), SeriesDepth::absolute(3))
            .unwrap();

        let t2 = t.rpow((1, 3).into()).unwrap();

        assert_eq!(t2.absolute_order(), (22, 3));
    }

    #[test]
    fn series_zero() {
        let v1 = symbol!("v1");

        let input = parse!("1/v1^2+1/v1+v1");
        let t = input
            .series(v1, Atom::num(0).as_view(), SeriesDepth::absolute(0))
            .unwrap();

        assert_eq!(t.to_atom().expand(), parse!("v1^-2+v1^-1"));
    }

    #[test]
    fn series_poles() {
        let v1 = symbol!("v1");
        let input = parse!("1/(v1^10+v1^20)");

        let t = input
            .series(v1, Atom::num(0).as_view(), SeriesDepth::absolute(-1))
            .unwrap()
            .to_atom();

        assert_eq!(t, parse!("v1^-10"))
    }

    #[test]
    fn series_user_function() {
        let v1 = symbol!("v1");

        let input = parse!("f(exp(v1),sin(v1))");
        let t = input
            .series(v1, Atom::num(0).as_view(), SeriesDepth::absolute(2))
            .unwrap()
            .to_atom();

        let res = parse!(
            "f(1,0)+v1*(der(0,1,f,1,0)+der(1,0,f,1,0))+1/2*v1^2*(der(0,2,f,1,0)+der(1,0,f,1,0)+2*der(1,1,f,1,0)+der(2,0,f,1,0))"
        );
        assert_eq!(t, res);
    }

    #[test]
    fn series_derivative_keeps_function_unevaluated() {
        let v1 = symbol!("v1");
        let _ = symbol!(
            "series_derivative_keeps_function_unevaluated::f",
            norm = |input, out| {
                if let AtomView::Fun(f) = input
                    && f.get_nargs() == 1
                    && f.iter().next().unwrap() == Atom::num(2).as_view()
                {
                    out.to_num(1);
                }
            }
        );

        let input = parse!("series_derivative_keeps_function_unevaluated::f(v1)");
        let t = input
            .series(v1, Atom::num(2).as_view(), SeriesDepth::absolute(2))
            .unwrap()
            .to_atom();

        let res = parse!(
            "1+(-2+v1)*der(1,series_derivative_keeps_function_unevaluated::f,2)+1/2*(-2+v1)^2*der(2,series_derivative_keeps_function_unevaluated::f,2)"
        );
        assert_eq!(t, res);
    }

    #[test]
    fn series_function_taylor_coefficients() {
        let v1 = symbol!("v1");

        // the derivatives of gamma are taken at the constant term of the argument
        let t = parse!("gamma(1-2*v1+v1^2)")
            .series(v1, Atom::num(0).as_view(), SeriesDepth::absolute(2))
            .unwrap();
        assert_eq!(t.absolute_order(), (3, 1));
        let g = Atom::var(crate::transcendental::euler_gamma());
        let pi = Atom::var(crate::atom::Symbol::PI);
        let res = Atom::num(1) + Atom::num(2) * &g * v1 - Atom::var(v1).pow(Atom::num(2)) * &g
            + Atom::num(2)
                * Atom::var(v1).pow(Atom::num(2))
                * (g.clone().pow(Atom::num(2)) + pi.pow(Atom::num(2)) / 6);
        assert_eq!((t.to_atom() - res).expand(), Atom::num(0));

        // the numbers in the derivatives of bessel_j are distributed
        let t = parse!("bessel_j(0,v1)")
            .series(v1, Atom::num(0).as_view(), SeriesDepth::absolute(10))
            .unwrap()
            .to_atom();
        assert_eq!(
            t,
            parse!("1-1/4*v1^2+1/64*v1^4-1/2304*v1^6+1/147456*v1^8-1/14745600*v1^10")
        );
    }

    #[test]
    fn series_function_of_puiseux_series() {
        let v1 = symbol!("v1");
        let t = parse!("atan(v1^(1/2)+v1)")
            .series(v1, Atom::num(0).as_view(), SeriesDepth::absolute(2))
            .unwrap()
            .to_atom();
        assert_eq!(t, parse!("v1^(1/2)+v1-1/3*v1^(3/2)-v1^2-4/5*v1^(5/2)"));
    }

    #[test]
    fn series_conj_and_abs_of_real_variable() {
        // conj and abs simplify on the expanded argument, which a Taylor expansion
        // in a fresh, non-real variable cannot do
        let x = symbol!("series_conj_real_x"; Real);
        let t = parse!("atan(series_conj_real_x)*conj(atan(series_conj_real_x))")
            .series(x, 0, 6)
            .unwrap();
        assert_eq!(
            t.to_atom(),
            parse!("series_conj_real_x^2-2/3*series_conj_real_x^4+23/45*series_conj_real_x^6")
        );

        let t = parse!("conj(sin(series_conj_real_x))")
            .series(x, 0, 5)
            .unwrap();
        assert_eq!(
            t.to_atom(),
            parse!("series_conj_real_x-1/6*series_conj_real_x^3+1/120*series_conj_real_x^5")
        );

        let t = parse!("abs(cosh(series_conj_real_x))")
            .series(x, 0, 4)
            .unwrap();
        assert_eq!(
            t.to_atom(),
            parse!("1+1/2*series_conj_real_x^2+1/24*series_conj_real_x^4")
        );
    }

    #[test]
    fn series_function_of_argument_known_to_negative_order() {
        // At a low order the argument cancels to O(v1^-1). That is not a pole, so
        // f(0) = 0 is not the value of the function.
        let v1 = symbol!("v1");
        let _ = symbol!(
            "series_negative_order_argument::f",
            norm = |input, out| {
                if let AtomView::Fun(f) = input
                    && f.get_nargs() == 1
                    && f.iter().next().unwrap().is_zero()
                {
                    **out = Atom::Zero;
                }
            }
        );

        let t = parse!("series_negative_order_argument::f((atan(v1)-v1)/v1^3)")
            .series(v1, 0, 0)
            .unwrap();
        assert_eq!(
            t.to_atom(),
            parse!("series_negative_order_argument::f(-1/3)")
        );
        assert_eq!(t.absolute_order(), 1);
    }

    #[test]
    fn series_function_with_argument_dependent_normalization() {
        // f(y) for a symbol y normalizes to y^2, so that a Taylor expansion in a fresh
        // variable would not expand f
        let v1 = symbol!("v1");
        let _ = symbol!(
            "series_argument_norm::f",
            norm = |input, out| {
                if let AtomView::Fun(f) = input
                    && let [AtomView::Var(_)] = f.iter().collect::<Vec<_>>().as_slice()
                {
                    **out = f.iter().next().unwrap().to_owned().pow(Atom::num(2));
                }
            }
        );

        let t = parse!("series_argument_norm::f(1+v1)")
            .series(v1, 0, 2)
            .unwrap();
        assert_eq!(
            t.to_atom(),
            parse!(
                "series_argument_norm::f(1)+v1*der(1,series_argument_norm::f,1)+1/2*v1^2*der(2,series_argument_norm::f,1)"
            )
        );
    }

    #[test]
    fn series_repeated_subexpressions() {
        let v1 = symbol!("v1");
        let terms = [
            "a*exp(v1+v1^2)/(1+v1)",
            "b*exp(v1+v1^2)/(1+v1)^2",
            "exp(v1+v1^2)*sin(v1)/v1^2",
        ]
        .map(|t| parse!(t));

        let whole = Atom::sum(&terms)
            .series(v1, Atom::num(0).as_view(), SeriesDepth::absolute(5))
            .unwrap();
        let parts = Series::sum(terms.map(|t| {
            t.series(v1, Atom::num(0).as_view(), SeriesDepth::absolute(5))
                .unwrap()
        }));

        assert_eq!((whole.to_atom() - parts.to_atom()).expand(), Atom::Zero);
        assert_eq!(whole.absolute_order(), parts.absolute_order());
    }

    #[test]
    fn series_grouped_terms() {
        let v1 = symbol!("v1");
        let t = parse!("a*exp(v1)*sin(v1)+b*sin(v1)*exp(v1)+c*v1^2+d*v1^2+e")
            .series(v1, Atom::num(0).as_view(), SeriesDepth::absolute(4))
            .unwrap();

        let res = parse!("(a+b)*(v1+v1^2+1/3*v1^3)+(c+d)*v1^2+e");
        assert!((t.to_atom() - res).expand().is_zero());
        assert_eq!(t.absolute_order(), (5, 1));
    }

    #[test]
    fn series_grouped_terms_that_cancel() {
        // The terms with exp(y*v1) and with 1+y*v1 cancel in their first orders. Each
        // term keeps its own constant factor, so that a*y+b*y-a*y-b*y normalizes to 0
        // and the leading terms are found.
        let v1 = symbol!("v1");
        let t = parse!("a*exp(y*v1)+b*exp(y*v1)-a*(1+y*v1)-b*(1+y*v1)")
            .series(v1, 0, SeriesDepth::relative(3))
            .unwrap();
        let res = parse!("(a+b)*(y^2/2*v1^2+y^3/6*v1^3+y^4/24*v1^4)");
        assert!((t.to_atom() - res).expand().is_zero());
        assert_eq!(t.absolute_order(), 5);

        let t = parse!("a*exp(y*v1)+b*exp(y*v1)-a*exp(y*v1+v1^9)-b*exp(y*v1+v1^9)")
            .series(v1, 0, SeriesDepth::relative(2))
            .unwrap();
        let res = parse!("-(a+b)*(v1^9+y*v1^10)");
        assert!((t.to_atom() - res).expand().is_zero());
        assert_eq!(t.absolute_order(), 11);

        // a function of such a sum needs its leading term
        let t = parse!(
            "log(a*exp(y*v1)+b*exp(y*v1)-a*(1+y*v1)-b*(1+y*v1))-log(v1^2*(a/2*y^2+b/2*y^2))"
        )
        .series(v1, 0, 2)
        .unwrap();
        assert_eq!(t.absolute_order(), 3);
        let res = parse!("y/3*v1+y^2/36*v1^2");
        assert!((t.to_atom() - res).together().is_zero());
        let a = parse!("a*exp(y*v1)/v1^2+b*exp(y*v1)/v1^2-a*(1+y*v1)/v1^2-b*(1+y*v1)/v1^2");
        let t = crate::function!(crate::atom::Symbol::SIN, a)
            .series(v1, 0, 2)
            .unwrap();
        assert_eq!(t.absolute_order(), 3);
    }

    #[test]
    fn series_linear_function() {
        // lin(1+v1) normalizes to lin(1)+lin(v1), so that the Taylor coefficients must
        // be taken at the point of the argument of lin(v1).
        let v1 = symbol!("v1");
        let _ = symbol!("series_linear_function::lin"; Linear);
        let t = parse!("series_linear_function::lin(1+v1)")
            .series(v1, 1, 2)
            .unwrap();
        let res = parse!(
            "2*series_linear_function::lin(1)+(v1-1)*der(1,series_linear_function::lin,1)+1/2*(v1-1)^2*der(2,series_linear_function::lin,1)"
        );
        assert!((t.to_atom() - res).expand().is_zero());
    }

    #[test]
    fn series_custom_function() {
        let v1 = symbol!("v1");
        let _ = symbol!(
            "series_custom_function::f",
            series = |args| {
                let [arg] = args else { unreachable!() };
                let point = arg.coefficient(0.into())?;
                let delta = arg.to_atom() - &point;
                Some((Atom::num(1), Atom::num(1) / (Atom::num(1) - &delta)))
            }
        );

        let t = parse!("series_custom_function::f(v1)")
            .series(v1, Atom::num(0).as_view(), SeriesDepth::absolute(3))
            .unwrap()
            .to_atom();

        assert_eq!(t, parse!("1+v1+v1^2+v1^3"));
    }

    #[test]
    fn series_exp_log() {
        let v1 = symbol!("v1");

        let input = parse!("1+2*log(v1^4)");
        let t = input
            .series(v1, Atom::num(0).as_view(), SeriesDepth::absolute(4))
            .unwrap()
            .exp()
            .unwrap();

        assert_eq!(t.to_atom().expand(), parse!("v1^8*exp(1)"));
    }

    #[test]
    fn series_sub_atom() {
        let v1 = symbol!("v1");

        let input = parse!("1/(1-v1)");
        let t = input
            .series(v1, Atom::num(0).as_view(), SeriesDepth::absolute(4))
            .unwrap();

        let r = (t - &parse!("1/v1+1")).unwrap();

        assert_eq!(r.absolute_order(), (5, 1));
        assert_eq!(r.to_atom(), parse!("-1*v1^-1+v1+v1^2+v1^3+v1^4"));
    }

    #[test]
    fn series_div_atom() {
        let v1 = symbol!("v1");

        let input = parse!("v1");
        let t = input
            .series(v1, Atom::num(0).as_view(), SeriesDepth::absolute(4))
            .unwrap();

        let r = ((t / &parse!("exp(v1)-1")).unwrap() * &parse!("v1")).unwrap();

        assert_eq!(r.relative_order(), (4, 1));
        assert_eq!(r.to_atom(), parse!("v1+-1/2*v1^2+1/12*v1^3"));
    }

    #[test]
    fn series_relative_order() {
        let v1 = symbol!("v1");

        let input = parse!("exp(v1)/v1-1/6*v1^2");
        let t = input
            .series(v1, Atom::num(0).as_view(), SeriesDepth::relative(4))
            .unwrap();

        assert_eq!(t.relative_order(), (4, 1));
        assert_eq!(t.to_atom(), parse!("v1^-1+1+1/2*v1"));
    }

    #[test]
    fn series_relative_noncanonical_exact_zero() {
        let x = symbol!("relative_exact_zero_x");
        let depth = SeriesDepth::relative(4);
        let expected = Atom::Zero.series(x, 0, depth.clone()).unwrap();
        for text in [
            "0.125*(1/(relative_exact_zero_x*(relative_exact_zero_x-1))-1/(relative_exact_zero_x-1)+1/relative_exact_zero_x)",
            "1/8*(1/(relative_exact_zero_x*(relative_exact_zero_x-1))-1/(relative_exact_zero_x-1)+1/relative_exact_zero_x)",
        ] {
            let input = parse!(text).expand();
            assert!(!input.is_zero());
            let result = input.series(x, 0, depth.clone()).unwrap();
            assert!(result.to_atom().is_zero());
            assert_eq!(result.relative_order(), expected.relative_order());
            assert_eq!(result.absolute_order(), expected.absolute_order());
        }
    }

    #[test]
    fn series_relative_zero_proof_preserves_nonzero_and_unresolved_terms() {
        let x = symbol!("relative_zero_control_x");
        let zero = parse!(
            "0.125*(1/(relative_zero_control_x*(relative_zero_control_x-1))-1/(relative_zero_control_x-1)+1/relative_zero_control_x)"
        )
        .expand();
        let exact = crate::atom::ZeroTestSettings::new()
            .treat_floats_as_exact(true)
            .numerical_samples(0);
        for remainder in [
            parse!("relative_zero_control_x^12"),
            parse!("relative_zero_control_a"),
            parse!("relative_zero_control_f(0)"),
        ] {
            let input = &zero + &remainder;
            let result = input
                .series(x, 0, SeriesDepth::relative(1))
                .unwrap()
                .to_atom();
            // A floating-point unit coefficient can retain its precision.
            assert!(
                result.check_equal_with(&remainder, &exact).is_true(),
                "expected {remainder}, got {result}"
            );
        }
        let tiny = &zero + parse!("1e-1000");
        assert!(!tiny.check_zero_with(&exact).is_true());
    }

    #[test]
    fn series_truncate() {
        let v1 = symbol!("v1");

        let input = parse!("v1^10");
        let t = input
            .series(v1, Atom::num(0).as_view(), SeriesDepth::absolute(4))
            .unwrap();
        assert_eq!(t.absolute_order(), (10, 1));
        assert_eq!(t.relative_order(), (0, 1));
        assert_eq!(t.to_atom(), parse!("0"));

        let r = (&t * &input).unwrap();
        assert_eq!(r.absolute_order(), (20, 1));
    }

    #[test]
    fn series_empty() {
        let v1 = symbol!("v1");

        let input = parse!("v1");
        let t = input
            .series(v1, Atom::num(0).as_view(), SeriesDepth::absolute(4))
            .unwrap();

        let r = &t - &t;

        let t2 = parse!("v1^6")
            .series(v1, Atom::num(0).as_view(), SeriesDepth::relative(4))
            .unwrap();

        let x = (&r + &parse!("v1^6")).unwrap();
        assert_eq!(r.absolute_order(), (5, 1));

        let c = x.cos().unwrap();
        assert_eq!(c.absolute_order(), (10, 1));
        assert_eq!(c.relative_order(), (10, 1));

        let s = x.sin().unwrap();
        assert_eq!(s.absolute_order(), (5, 1));
        assert_eq!(s.relative_order(), (5, 1));

        let e = x.exp().unwrap();
        assert_eq!(e.absolute_order(), (5, 1));
        assert_eq!(e.relative_order(), (5, 1));

        let add = &r + &t2;
        assert_eq!(add.absolute_order(), (5, 1));
        let mul = &r * &t2;
        assert_eq!(mul.absolute_order(), (11, 1));
    }

    #[test]
    fn series_pole_order_does_not_change_the_result() {
        let x = symbol!("series_pole_order_x");
        for (text, depth) in [
            (
                "1/(exp(series_pole_order_x)-1)^3*sin(series_pole_order_x)^2/(1-cos(series_pole_order_x))",
                6,
            ),
            (
                "1/(series_pole_order_x^5*(exp(series_pole_order_x)-1-series_pole_order_x-series_pole_order_x^2/2-series_pole_order_x^3/6))",
                2,
            ),
            (
                "gamma(series_pole_order_x)*gamma(-series_pole_order_x)/series_pole_order_x",
                1,
            ),
            (
                "(series_pole_order_x-sin(series_pole_order_x))/series_pole_order_x^3",
                5,
            ),
        ] {
            let input = parse!(text);
            let expected = input.series(x, 0, depth).unwrap();
            for pole_order in [-2, 0, 1, 3, 10] {
                let result = input
                    .series_with_pole_order(x, 0, depth, pole_order)
                    .unwrap();
                assert_eq!(result.to_atom(), expected.to_atom(), "{text}, {pole_order}");
                assert_eq!(result.absolute_order(), expected.absolute_order());
            }
        }
    }

    #[test]
    fn series_cancelling_function_argument_keeps_its_precision() {
        // The leading terms of the argument cancel, so that it must be expanded deeper
        // before the custom series of gamma takes its inverse.
        let x = symbol!("series_cancelling_argument_x");
        for (text, equivalent) in [
            (
                "gamma(exp(series_cancelling_argument_x)-1-series_cancelling_argument_x)",
                "gamma(exp(series_cancelling_argument_x)-series_cancelling_argument_x)/(exp(series_cancelling_argument_x)-1-series_cancelling_argument_x)",
            ),
            (
                "gamma(2*series_cancelling_argument_x^2+1-cos(series_cancelling_argument_x))",
                "gamma(2*series_cancelling_argument_x^2+2-cos(series_cancelling_argument_x))/(2*series_cancelling_argument_x^2+1-cos(series_cancelling_argument_x))",
            ),
        ] {
            let result = parse!(text).series(x, 0, 3).unwrap();
            let expected = parse!(equivalent).series(x, 0, 3).unwrap();
            assert_eq!(result.absolute_order(), expected.absolute_order(), "{text}");
            assert_eq!(
                (result.to_atom() - expected.to_atom()).expand(),
                Atom::Zero,
                "{text}"
            );
        }
    }

    #[test]
    fn series_function_with_shifted_pole_keeps_its_precision() {
        // The custom series subtracts the pole from the truncated argument, so the rest
        // must be known to the relative order. The equivalent forms have no pole in a
        // function argument: gamma(a) = gamma(a+2)/(a*(a+1)), tan(pi/2+y) = -cos(y)/sin(y).
        let v1 = symbol!("v1");
        for (text, equivalent, depth) in [
            (
                "gamma(exp(v1)-2)",
                "gamma(exp(v1))/((exp(v1)-2)*(exp(v1)-1))",
                SeriesDepth::absolute(-1),
            ),
            (
                "gamma(exp(v1)-2)",
                "gamma(exp(v1))/((exp(v1)-2)*(exp(v1)-1))",
                SeriesDepth::absolute(2),
            ),
            (
                "gamma(cos(v1)-2)",
                "gamma(cos(v1))/((cos(v1)-2)*(cos(v1)-1))",
                SeriesDepth::absolute(3),
            ),
            (
                "gamma(cos(v1)-2)",
                "gamma(cos(v1))/((cos(v1)-2)*(cos(v1)-1))",
                SeriesDepth::relative(2),
            ),
            (
                "tan(pi/2*cos(v1))",
                "-cos(pi/2*(cos(v1)-1))/sin(pi/2*(cos(v1)-1))",
                SeriesDepth::absolute(2),
            ),
        ] {
            let result = parse!(text).series(v1, 0, depth.clone()).unwrap();
            let expected = parse!(equivalent).series(v1, 0, depth).unwrap();
            assert_eq!(result.absolute_order(), expected.absolute_order(), "{text}");
            assert_eq!(
                (result.to_atom() - expected.to_atom()).expand(),
                Atom::Zero,
                "{text}"
            );
        }
    }

    #[test]
    fn series_to_low_depth_with_cancellations() {
        // An expansion to relative order one is empty or fails for these, so the
        // order to start from is not known from it.
        let v1 = symbol!("v1");
        let y = parse!("y");
        for (text, point, depth, expected) in [
            ("(1-cos(v1))/(exp(v1)-1)^3", Atom::Zero, -1, "1/2*v1^-1"),
            ("(exp(v1)-1)/v1^3", Atom::Zero, -2, "v1^-2"),
            ("atan(v1)/v1^3", Atom::Zero, -2, "v1^-2"),
            ("(v1-sin(v1))/v1^5", Atom::Zero, -1, "1/6*v1^-2"),
            ("(v1-sin(v1))/v1^3+(1-cos(v1))^2/v1^4", Atom::Zero, -2, "0"),
            ("log(v1/y)/(v1-y)^2", y.clone(), -1, "1/(y*(v1-y))"),
            (
                "exp(1/v1^2-1/sin(v1)^2)",
                Atom::Zero,
                2,
                "exp(-1/3)-1/15*exp(-1/3)*v1^2",
            ),
        ] {
            let result = parse!(text).series(v1, point, depth).unwrap();
            assert!(result.absolute_order() > depth, "{text}");
            assert_eq!(
                (result.to_atom() - parse!(expected)).expand(),
                Atom::Zero,
                "{text}"
            );
        }
    }

    #[test]
    fn series_function_of_vanishing_puiseux_argument() {
        let x = symbol!("series_vanishing_puiseux_x");
        let exp = parse!("exp(series_vanishing_puiseux_x^(3/2))-1")
            .series(x, 0, 0)
            .unwrap();
        assert!(exp.is_zero());
        assert_eq!(exp.absolute_order(), (3, 2));

        let cos = parse!("1-cos(series_vanishing_puiseux_x^(3/2))")
            .series(x, 0, 1)
            .unwrap();
        // the cancelling sum is expanded deeper, up to the leading term x^3/2
        assert!(cos.is_zero());
        assert_eq!(cos.absolute_order(), 3);

        let quotient =
            parse!("(exp(series_vanishing_puiseux_x^(3/2))-1)/series_vanishing_puiseux_x^(3/2)")
                .series(x, 0, 2)
                .unwrap();
        assert_eq!(
            quotient.to_atom(),
            parse!("1+series_vanishing_puiseux_x^(3/2)/2")
        );
    }

    #[test]
    fn series_relative_depth_finds_the_leading_term() {
        let x = symbol!("series_relative_leading_x");
        let t = parse!("log(exp(series_relative_leading_x^3))")
            .series(x, 0, SeriesDepth::relative(1))
            .unwrap();
        assert_eq!(t.to_atom(), parse!("series_relative_leading_x^3"));
        assert_eq!(t.absolute_order(), 4);

        let t = parse!("log(1+series_relative_leading_x^2)")
            .series(x, 0, SeriesDepth::relative(3))
            .unwrap();
        assert_eq!(
            t.to_atom(),
            parse!("series_relative_leading_x^2-series_relative_leading_x^4/2")
        );
        assert_eq!(t.relative_order(), 3);
    }

    #[test]
    fn series_sum_with_different_valuations() {
        let x = symbol!("series_valuations_x");
        let input = parse!(
            "gamma(1+series_valuations_x)/series_valuations_x+exp(series_valuations_x)/series_valuations_x^2+series_valuations_x^3*atan(series_valuations_x)"
        );
        let result = input.series(x, 0, 4).unwrap();
        assert_eq!(result.absolute_order(), 5);
        let gamma = parse!("gamma(1+series_valuations_x)")
            .series(x, 0, 5)
            .unwrap()
            .to_atom();
        let expected = (gamma / parse!("series_valuations_x")
            + parse!(
                "series_valuations_x^-2+series_valuations_x^-1+1/2+series_valuations_x/6+series_valuations_x^2/24+series_valuations_x^3/120+series_valuations_x^4/720+series_valuations_x^4"
            ))
        .expand();
        let mut expected = expected.series(x, 0, 4).unwrap();
        expected.truncate_absolute_order(5.into());
        assert_eq!((result.to_atom() - expected.to_atom()).expand(), Atom::Zero);
    }
}
