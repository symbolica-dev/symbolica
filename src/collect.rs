use ahash::{HashMap, HashSet};
use numerica::domains::{
    float::{Complex, FloatField},
    rational::Rational,
};

use crate::{
    atom::{Add, Atom, AtomCore, AtomOrView, AtomView, Indeterminate, Symbol},
    coefficient::{Coefficient, CoefficientView},
    domains::{algebraic_number::AlgebraicExtension, float::FloatLike, integer::Z, rational::Q},
    poly::{Exponent, factor::Factorize, polynomial::MultivariatePolynomial},
    state::Workspace,
};
use std::{ops::Div, sync::Arc};

impl<'a> AtomView<'a> {
    /// Collect terms involving the same power of `x`, where `x` is an indeterminate, e.g.
    ///
    /// ```math
    /// collect(x + x * y + x^2, x) = x * (1+y) + x^2
    /// ```
    ///
    /// Both the *key* (the quantity collected in) and its coefficient can be mapped using
    /// `key_map` and `coeff_map` respectively.
    pub(crate) fn collect<E: Exponent, T: AtomCore>(
        &self,
        x: T,
        key_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
        coeff_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
    ) -> Atom {
        self.collect_multiple::<E, T>(std::slice::from_ref(&x), key_map, coeff_map)
    }

    pub(crate) fn collect_symbol<E: Exponent>(
        &self,
        x: Symbol,
        key_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
        coeff_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
    ) -> Atom {
        let vars: Vec<_> = self
            .get_all_indeterminates(false)
            .into_iter()
            .filter(|v| v.get_symbol().unwrap() == x)
            .collect();

        self.collect_multiple::<E, AtomView>(&vars, key_map, coeff_map)
    }

    pub(crate) fn collect_multiple<E: Exponent, T: AtomCore>(
        &self,
        xs: &[T],
        key_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
        coeff_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
    ) -> Atom {
        let mut out = Atom::new();
        Workspace::get_local()
            .with(|ws| self.collect_multiple_impl::<E, T>(xs, ws, key_map, coeff_map, &mut out));
        out
    }

    pub(crate) fn collect_multiple_impl<E: Exponent, T: AtomCore>(
        &self,
        xs: &[T],
        ws: &Workspace,
        key_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
        coeff_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
        out: &mut Atom,
    ) {
        let r = self.coefficient_list::<E, T>(xs);

        let mut add_h = Atom::new();
        let add = add_h.to_add();

        fn map_key_coeff(
            key: AtomView,
            coeff: Atom,
            workspace: &Workspace,
            key_map: &Option<Box<dyn Fn(AtomView, &mut Atom)>>,
            coeff_map: &Option<Box<dyn Fn(AtomView, &mut Atom)>>,
            add: &mut Add,
        ) {
            let mut mul_h = workspace.new_atom();
            let mul = mul_h.to_mul();

            if let Some(key_map) = &key_map {
                let mut handle = workspace.new_atom();
                key_map(key, &mut handle);
                mul.extend(handle.as_view());
            } else {
                mul.extend(key);
            }

            if let Some(coeff_map) = &coeff_map {
                let mut handle = workspace.new_atom();
                coeff_map(coeff.as_view(), &mut handle);
                mul.extend(handle.as_view());
            } else {
                mul.extend(coeff.as_view());
            }

            add.extend(mul_h.as_view());
        }

        for (key, coeff) in r {
            map_key_coeff(key.as_view(), coeff, ws, &key_map, &coeff_map, add);
        }

        add_h.as_view().normalize(ws, out);
    }

    /// Collect terms involving the same powers of `x` in `xs`, where `x` is an indeterminate.
    /// Return the list of key-coefficient pairs.
    pub(crate) fn coefficient_list<E: Exponent, T: AtomCore>(&self, xs: &[T]) -> Vec<(Atom, Atom)> {
        let vars = xs
            .iter()
            .map(|x| x.as_atom_view().to_owned().try_into().unwrap())
            .collect::<Vec<_>>();

        let p = self.to_polynomial_in_vars::<E>(&Arc::new(vars));

        let mut coeffs = vec![];
        for t in p.into_iter() {
            let mut key = Atom::num(1);

            for (p, v) in t.exponents.iter().zip(xs) {
                let mut pow = Atom::new();
                pow.to_pow(v.as_atom_view(), Atom::num(p.to_i32() as i64).as_view());
                key *= pow;
            }

            coeffs.push((key, t.coefficient.clone()));
        }

        coeffs
    }

    /// Collect terms involving the literal occurrence of `x`.
    pub fn coefficient_with_ws(&self, x: AtomView<'_>, workspace: &Workspace) -> Atom {
        let mut coeffs = workspace.new_atom();
        let coeff_add = coeffs.to_add();

        match self {
            AtomView::Add(a) => {
                for arg in a {
                    arg.collect_factor(x, workspace, coeff_add)
                }
            }
            _ => self.collect_factor(x, workspace, coeff_add),
        }

        let mut rest_norm = Atom::new();
        coeffs.as_view().normalize(workspace, &mut rest_norm);
        rest_norm
    }

    fn collect_factor(&self, x: AtomView<'_>, workspace: &Workspace, coeff: &mut Add) {
        match self {
            AtomView::Add(_) => {}
            AtomView::Mul(m) => {
                if m.iter().any(|a| a == x) {
                    let mut collected = workspace.new_atom();
                    let mul = collected.to_mul();

                    // we could have a double match if x*x(..)
                    // we then only collect on the first hit
                    let mut bracket = None;

                    for a in m {
                        if bracket.is_none() && a == x {
                            bracket = Some(a);
                        } else {
                            mul.extend(a);
                        }
                    }

                    coeff.extend(collected.as_view());
                }

                if let AtomView::Mul(y) = x {
                    // check if all factors occur
                    for xx in y.iter() {
                        if !m.iter().any(|a| a == xx) {
                            return;
                        }
                    }

                    let mut collected = workspace.new_atom();
                    let mul = collected.to_mul();

                    for xx in m {
                        if !y.iter().any(|a| a == xx) {
                            mul.extend(xx);
                        }
                    }

                    coeff.extend(collected.as_view());
                }
            }
            _ => {
                if *self == x {
                    // add the coefficient 1
                    let collected = workspace.new_num(1);
                    coeff.extend(collected.as_view());
                }
            }
        }
    }

    /// Write the expression over a common denominator.
    pub fn together(&self) -> Atom {
        let mut out = Atom::new();
        self.together_into(&mut out);
        out
    }

    /// Write the expression over a common denominator.
    pub fn together_into(&self, out: &mut Atom) {
        if self.has_complex_coefficients() {
            let f = AlgebraicExtension::new_complex(Q);
            let f2 = FloatField::from_rep(Complex::new(Rational::zero(), Rational::one()));
            if let Ok(p) = self.try_to_rational_polynomial::<_, _, u32>(&f, &f, None) {
                let num = p.numerator.map_coeff(
                    |c| {
                        Complex::new(
                            c.poly.get_constant(),
                            c.poly.coefficient(&[1]).unwrap_or(Rational::zero()),
                        )
                    },
                    f2.clone(),
                );

                let den = p.denominator.map_coeff(
                    |c| {
                        Complex::new(
                            c.poly.get_constant(),
                            c.poly.coefficient(&[1]).unwrap_or(Rational::zero()),
                        )
                    },
                    f2.clone(),
                );

                *out = num.to_expression() / den.to_expression();
                return;
            } else {
                out.set_from_view(self);
            }
        } else if let Ok(poly) = self.try_to_rational_polynomial::<_, _, u32>(&Q, &Z, None) {
            poly.to_expression_into(out);
            return;
        }

        // expression contains floats or cannot be converted to rational polynomial for other reasons
        // do a cross multiplication
        if let AtomView::Add(a) = self {
            let mut terms = vec![];
            let mut all_dens = HashSet::default();

            for arg in a {
                let mut numerators = vec![];
                let mut denominators = vec![];
                if let AtomView::Mul(m) = arg {
                    for aa in m {
                        if let AtomView::Pow(p) = aa {
                            let (b, e) = p.get_base_exp();
                            if let AtomView::Num(n) = e
                                && let CoefficientView::Natural(n, d, ni, _di) = n.get_coeff_view()
                                && ni == 0
                                && n < 0
                                && d == 1
                            {
                                all_dens.insert(b);
                                denominators.push(b);
                                continue;
                            }
                        }

                        numerators.push(aa);
                    }
                } else if let AtomView::Pow(p) = arg {
                    let (b, e) = p.get_base_exp();
                    if let AtomView::Num(n) = e
                        && let CoefficientView::Natural(n, d, ni, _di) = n.get_coeff_view()
                        && ni == 0
                        && n < 0
                        && d == 1
                    {
                        all_dens.insert(b);
                        denominators.push(b);
                    } else {
                        numerators.push(arg);
                    }
                } else {
                    numerators.push(arg);
                }

                terms.push((numerators, denominators));
            }

            let mut numerator = Atom::new();

            for (num, den) in terms {
                let mut term = if num.is_empty() {
                    // empty when input is just a denominator
                    Atom::num(1)
                } else {
                    num[0].to_owned()
                };
                for n in num.iter().skip(1) {
                    term *= *n;
                }
                for d in &all_dens {
                    if !den.iter().any(|x| x == d) {
                        // multiply numerator by missing denominator
                        term *= *d;
                    }
                }

                numerator += term;
            }

            let mut denominator = Atom::num(1);
            for d in all_dens {
                denominator *= d;
            }

            *out = numerator / denominator;
        } else {
            out.set_from_view(self);
        }
    }

    /// Write the expression as a sum of terms with minimal denominators.
    pub fn apart(&self, x: &Indeterminate) -> Atom {
        let mut out = Atom::new();

        Workspace::get_local().with(|ws| {
            self.apart_with_ws_into(x, ws, &mut out);
        });

        out
    }

    /// Write the expression as a sum of terms with minimal denominators.
    pub fn apart_with_ws_into(&self, x: &Indeterminate, ws: &Workspace, out: &mut Atom) {
        if self.has_complex_coefficients() {
            let f = AlgebraicExtension::new_complex(Q);
            let f2 = FloatField::from_rep(Complex::new(Rational::zero(), Rational::one()));
            if let Ok(poly) = self.try_to_rational_polynomial::<_, _, u32>(&f, &f, None) {
                if let Some(v) = poly.get_variables().iter().position(|v| v == x) {
                    let mut a = ws.new_atom();
                    let add = a.to_add();

                    let mut numa = ws.new_atom();
                    let mut dena = ws.new_atom();
                    for x in poly.apart(v) {
                        let num = x.numerator.map_coeff(
                            |c| {
                                Complex::new(
                                    c.poly.get_constant(),
                                    c.poly.coefficient(&[1]).unwrap_or(Rational::zero()),
                                )
                            },
                            f2.clone(),
                        );

                        let den = x.denominator.map_coeff(
                            |c| {
                                Complex::new(
                                    c.poly.get_constant(),
                                    c.poly.coefficient(&[1]).unwrap_or(Rational::zero()),
                                )
                            },
                            f2.clone(),
                        );

                        num.to_expression_into(&mut numa);
                        den.to_expression_into(&mut dena);
                        add.extend(numa.as_view().div(dena.as_view()).as_view());
                    }

                    add.as_view().normalize(ws, out);
                } else {
                    out.set_from_view(self);
                }
            } else {
                out.set_from_view(self);
            }
        } else if let Ok(poly) = self.try_to_rational_polynomial::<_, _, u32>(&Q, &Z, None) {
            if let Some(v) = poly.get_variables().iter().position(|v| v == x) {
                let mut a = ws.new_atom();
                let add = a.to_add();

                let mut a = ws.new_atom();
                for x in poly.apart(v) {
                    x.to_expression_into(&mut a);
                    add.extend(a.as_view());
                }

                add.as_view().normalize(ws, out);
            } else {
                out.set_from_view(self);
            }
        } else {
            out.set_from_view(self);
        }
    }

    /// Write the expression as a sum of terms with minimal denominators in all variables.
    pub fn apart_multivariate(&self) -> Atom {
        let mut out = Atom::new();

        Workspace::get_local().with(|ws| {
            self.apart_multivariate_with_ws_into(ws, &mut out);
        });

        out
    }

    /// Write the expression as a sum of terms with minimal denominators in all variables.
    pub fn apart_multivariate_with_ws_into(&self, ws: &Workspace, out: &mut Atom) {
        if self.has_complex_coefficients() {
            let f = AlgebraicExtension::new_complex(Q);
            let f2 = FloatField::from_rep(Complex::new(Rational::zero(), Rational::one()));
            if let Ok(poly) = self.try_to_rational_polynomial::<_, _, u32>(&f, &f, None) {
                let mut a = ws.new_atom();
                let add = a.to_add();

                let mut numa = ws.new_atom();
                let mut dena = ws.new_atom();
                for x in poly.apart_multivariate() {
                    let num = x.numerator.map_coeff(
                        |c| {
                            Complex::new(
                                c.poly.get_constant(),
                                c.poly.coefficient(&[1]).unwrap_or(Rational::zero()),
                            )
                        },
                        f2.clone(),
                    );

                    let den = x.denominator.map_coeff(
                        |c| {
                            Complex::new(
                                c.poly.get_constant(),
                                c.poly.coefficient(&[1]).unwrap_or(Rational::zero()),
                            )
                        },
                        f2.clone(),
                    );

                    num.to_expression_into(&mut numa);
                    den.to_expression_into(&mut dena);
                    add.extend(numa.as_view().div(dena.as_view()).as_view());
                }

                add.as_view().normalize(ws, out);
            } else {
                out.set_from_view(self);
            }
        } else if let Ok(poly) = self.try_to_rational_polynomial::<_, _, u32>(&Q, &Z, None) {
            let mut a = ws.new_atom();
            let add = a.to_add();

            let mut a = ws.new_atom();
            for x in poly.apart_multivariate() {
                x.to_expression_into(&mut a);
                add.extend(a.as_view());
            }

            add.as_view().normalize(ws, out);
        } else {
            out.set_from_view(self);
        }
    }

    /// Cancel all common factors between numerators and denominators.
    /// Any non-canceling parts of the expression will not be rewritten.
    pub(crate) fn cancel(&self) -> Atom {
        let mut out = Atom::new();
        self.cancel_into(&mut out);
        out
    }

    /// Cancel all common factors between numerators and denominators.
    /// Any non-canceling parts of the expression will not be rewritten.
    pub(crate) fn cancel_into(&self, out: &mut Atom) {
        Workspace::get_local().with(|ws| {
            self.cancel_with_ws_into(ws, out);
        });
    }

    fn cancel_with_ws_into(&self, ws: &Workspace, out: &mut Atom) -> bool {
        match self {
            AtomView::Num(_) | AtomView::Var(_) | AtomView::Fun(_) | AtomView::Pow(_) => {
                out.set_from_view(self);
                false
            }
            AtomView::Mul(m) => {
                // split between numerator, denominator and rest
                // any numerator or denominator part that does not cancel will be kept as is
                let mut numerators = vec![];
                let mut denominators = vec![];
                let mut num_changed = vec![];
                let mut den_changed = vec![];
                let mut rest = vec![];

                for a in m {
                    if let AtomView::Pow(p) = a {
                        let (b, e) = p.get_base_exp();
                        if let AtomView::Num(n) = e
                            && let CoefficientView::Natural(n, d, ni, _di) = n.get_coeff_view()
                            && ni == 0
                        {
                            if n < 0 && d == 1 {
                                denominators.push(
                                    b.to_polynomial::<_, u16>(&Q, None)
                                        .pow(n.unsigned_abs() as usize),
                                );
                                den_changed.push((a, false));
                                continue;
                            } else if n > 0 && d == 1 {
                                numerators.push(a.to_polynomial::<_, u16>(&Q, None));
                                num_changed.push((a, false));
                                continue;
                            }
                        }

                        rest.push(a);
                    } else {
                        numerators.push(a.to_polynomial(&Q, None));
                        num_changed.push((a, false));
                    }
                }

                if numerators.is_empty() || denominators.is_empty() {
                    out.set_from_view(self);
                    return false;
                }

                MultivariatePolynomial::unify_variables_list(&mut numerators);
                MultivariatePolynomial::unify_variables_list(&mut denominators);
                numerators[0].unify_variables(&mut denominators[0]);
                MultivariatePolynomial::unify_variables_list(&mut numerators);
                MultivariatePolynomial::unify_variables_list(&mut denominators);

                let mut changed = false;
                for (d, ds) in denominators.iter_mut().zip(&mut den_changed) {
                    for (n, ns) in numerators.iter_mut().zip(&mut num_changed) {
                        let g = n.gcd(d);
                        if !g.is_one() {
                            changed = true;
                            ds.1 = true;
                            ns.1 = true;
                            *n = &*n / &g;
                            *d = &*d / &g;
                        }
                    }
                }

                if !changed {
                    out.set_from_view(self);
                    return false;
                }

                let mut mul = ws.new_atom();
                let mul_view = mul.to_mul();

                let mut tmp = ws.new_atom();
                for (n, (orig, changed)) in numerators.iter().zip(num_changed) {
                    if changed {
                        n.to_expression_into(&mut tmp);
                        mul_view.extend(tmp.as_view());
                    } else {
                        mul_view.extend(orig);
                    }
                }

                for (d, (orig, changed)) in denominators.iter().zip(den_changed) {
                    if changed {
                        d.to_expression_into(&mut tmp);

                        let mut pow = ws.new_atom();
                        let exp = ws.new_num(-1);
                        pow.to_pow(tmp.as_view(), exp.as_view());

                        mul_view.extend(pow.as_view());
                    } else {
                        mul_view.extend(orig);
                    }
                }

                for r in rest {
                    mul_view.extend(r);
                }

                mul_view.as_view().normalize(ws, out);
                true
            }
            AtomView::Add(a) => {
                let mut add = ws.new_atom();
                let add_view = add.to_add();

                let mut changed = false;
                let mut tmp = ws.new_atom();
                for arg in a {
                    if arg.cancel_with_ws_into(ws, &mut tmp) {
                        changed = true;
                        add_view.extend(tmp.as_view());
                    } else {
                        add_view.extend(arg);
                    }
                }

                if changed {
                    add_view.as_view().normalize(ws, out);
                    true
                } else {
                    out.set_from_view(self);
                    false
                }
            }
        }
    }

    /// Factor the expression over the rationals.
    pub fn factor(&self) -> Atom {
        if self.has_complex_coefficients() {
            let f = AlgebraicExtension::new_complex(Q);
            let f2 = FloatField::from_rep(Complex::new(Rational::zero(), Rational::one()));
            let Ok(r) = self.try_to_rational_polynomial::<_, _, u32>(&f, &f, None) else {
                return self.to_owned();
            };

            let f_n = r.numerator.factor();
            let f_d = r.denominator.factor();

            if f_n.is_empty() {
                return Atom::num(0);
            }

            let mut out = Atom::new();
            let mul = out.to_mul();

            let mut pow = Atom::new();
            for (k, v) in f_n {
                let k = k.map_coeff(
                    |c| {
                        Complex::new(
                            c.poly.get_constant(),
                            c.poly.coefficient(&[1]).unwrap_or(Rational::zero()),
                        )
                    },
                    f2.clone(),
                );
                if v > 1 {
                    let exp = Atom::num(v as i64);
                    pow.to_pow(k.to_expression().as_view(), exp.as_view());
                    mul.extend(pow.as_view());
                } else {
                    mul.extend(k.to_expression().as_view());
                }
            }

            for (k, v) in f_d {
                let k = k.map_coeff(
                    |c| {
                        Complex::new(
                            c.poly.get_constant(),
                            c.poly.coefficient(&[1]).unwrap_or(Rational::zero()),
                        )
                    },
                    f2.clone(),
                );

                let exp = Atom::num(-(v as i64));
                pow.to_pow(k.to_expression().as_view(), exp.as_view());
                mul.extend(pow.as_view());
            }

            Workspace::get_local().with(|ws| {
                out.as_view().normalize(ws, &mut pow);
            });

            pow
        } else {
            let Ok(r) = self.try_to_rational_polynomial::<_, _, u32>(&Q, &Z, None) else {
                return self.to_owned();
            };

            let f_n = r.numerator.factor();
            let f_d = r.denominator.factor();

            if f_n.is_empty() {
                return Atom::num(0);
            }

            let mut out = Atom::new();
            let mul = out.to_mul();

            let mut pow = Atom::new();
            for (k, v) in f_n {
                if v > 1 {
                    let exp = Atom::num(v as i64);
                    pow.to_pow(k.to_expression().as_view(), exp.as_view());
                    mul.extend(pow.as_view());
                } else {
                    mul.extend(k.to_expression().as_view());
                }
            }

            for (k, v) in f_d {
                let exp = Atom::num(-(v as i64));
                pow.to_pow(k.to_expression().as_view(), exp.as_view());
                mul.extend(pow.as_view());
            }

            Workspace::get_local().with(|ws| {
                out.as_view().normalize(ws, &mut pow);
            });

            pow
        }
    }

    /// Collect numerical factors by removing the numerical content from additions.
    /// For example, `-2*x + 4*x^2 + 6*x^3` will be transformed into `-2*(x - 2*x^2 - 3*x^3)`.
    ///
    /// The first argument of the addition is normalized to a positive quantity.
    pub fn collect_num(&self) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut coeff = Atom::new();
            self.collect_num_impl(ws, &mut coeff);
            coeff
        })
    }

    fn collect_num_impl(&self, ws: &Workspace, out: &mut Atom) -> bool {
        fn get_num(a: AtomView) -> Option<Coefficient> {
            match a {
                AtomView::Num(n) => Some(n.get_coeff_view().to_owned()),
                AtomView::Add(add) => {
                    // perform GCD of all arguments
                    // make sure the first argument is positive
                    let mut is_negative = false;
                    let mut gcd: Option<Coefficient> = None;
                    for arg in add.iter() {
                        if let Some(num) = get_num(arg) {
                            if let Some(g) = gcd {
                                gcd = Some(g.gcd(&num));
                            } else {
                                is_negative = num.is_negative();
                                gcd = Some(num);
                            }
                        }
                    }

                    if let Some(g) = gcd {
                        if is_negative && !g.is_negative() {
                            Some(-g)
                        } else {
                            Some(g)
                        }
                    } else {
                        None
                    }
                }
                AtomView::Mul(mul) => {
                    if mul.has_coefficient() {
                        for aa in mul.iter() {
                            if let AtomView::Num(n) = aa {
                                return Some(n.get_coeff_view().to_owned());
                            }
                        }

                        unreachable!()
                    } else {
                        None
                    }
                }
                AtomView::Pow(p) => {
                    let (b, e) = p.get_base_exp();
                    if let Ok(e) = i64::try_from(e)
                        && let Some(n) = get_num(b)
                        && let Coefficient::Complex(r) = n
                    {
                        if e < 0 {
                            return Some(r.pow((-e) as u64).inv().into());
                        } else {
                            return Some(r.pow(e as u64).into());
                        }
                    }

                    None
                }
                AtomView::Var(_) | AtomView::Fun(_) => None,
            }
        }

        match self {
            AtomView::Add(a) => {
                let mut r = ws.new_atom();
                let ra = r.to_add();
                let mut na = ws.new_atom();
                let mut changed = false;
                for arg in a {
                    changed |= arg.collect_num_impl(ws, &mut na);
                    ra.extend(na.as_view());
                }

                if !changed {
                    out.set_from_view(self);
                } else {
                    r.as_view().normalize(ws, out);
                }

                if let AtomView::Add(aa) = out.as_view()
                    && let Some(n) = get_num(out.as_view())
                {
                    let v = ws.new_num(n);
                    // divide every term by n
                    let ra = r.to_add();
                    let mut div = ws.new_atom();
                    for arg in aa.iter() {
                        arg.div_with_ws_into(ws, v.as_view(), &mut div);
                        ra.extend(div.as_view());
                    }

                    let m = div.to_mul();
                    m.extend(r.as_view());
                    m.extend(v.as_view());
                    m.as_view().normalize(ws, out);
                    changed = true;
                }

                changed
            }
            AtomView::Mul(m) => {
                let mut r = ws.new_atom();
                let ra = r.to_mul();
                let mut na = ws.new_atom();
                let mut changed = false;
                for arg in m {
                    changed |= arg.collect_num_impl(ws, &mut na);
                    ra.extend(na.as_view());
                }

                if !changed {
                    out.set_from_view(self);
                } else {
                    r.as_view().normalize(ws, out);
                }

                changed
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();

                let mut changed = false;
                let mut nb = ws.new_atom();
                changed |= b.collect_num_impl(ws, &mut nb);
                let mut ne = ws.new_atom();
                changed |= e.collect_num_impl(ws, &mut ne);

                if !changed {
                    out.set_from_view(self);
                } else {
                    let mut np = ws.new_atom();
                    np.to_pow(nb.as_view(), ne.as_view());
                    np.as_view().normalize(ws, out);
                }

                changed
            }
            _ => {
                out.set_from_view(self);
                false
            }
        }
    }

    pub(crate) fn collect_factors(&self) -> Atom {
        let mut factors = HashMap::default();
        Workspace::get_local().with(|ws| {
            self.collect_factors_impl(ws, &mut factors);

            if factors.len() == 1 {
                let (f, p) = factors.into_iter().next().unwrap();
                if p == 1 {
                    f.into_owned()
                } else {
                    let mut res = Atom::new();
                    let mut pow = ws.new_atom();
                    let exp = ws.new_num(p as i64);
                    pow.to_pow(f.as_view(), exp.as_view());
                    pow.as_view().normalize(ws, &mut res);
                    res
                }
            } else {
                let mut res = Atom::new();
                let mut mul = ws.new_atom();
                let mul_view = mul.to_mul();
                for (a, n) in factors {
                    let mut pow = ws.new_atom();
                    let exp = ws.new_num(n as i64);
                    pow.to_pow(a.as_view(), exp.as_view());
                    mul_view.extend(pow.as_view());
                }
                mul.as_view().normalize(ws, &mut res);
                res
            }
        })
    }

    fn collect_factors_impl(&self, ws: &Workspace, factors: &mut HashMap<AtomOrView<'a>, isize>) {
        match self {
            AtomView::Num(_) | AtomView::Var(_) | AtomView::Fun(_) => {
                *factors.entry(self.into()).or_insert(0) += 1;
            }
            AtomView::Add(a) => {
                let mut subfactors = Vec::with_capacity(a.get_nargs());

                for arg in a {
                    let mut h = HashMap::default();
                    arg.collect_factors_impl(ws, &mut h);
                    subfactors.push(h);
                }

                let mut first = true;
                for f in &subfactors {
                    for (k, v) in f {
                        if let Some(p) = factors.get_mut(k) {
                            *p = (*p).min(*v);
                        } else if first {
                            factors.insert(k.clone(), *v);
                        } else {
                            factors.insert(k.clone(), 0.min(*v));
                        }
                    }

                    first = false;
                }

                for (ff, p) in &mut *factors {
                    for f in &subfactors {
                        if !f.contains_key(ff) {
                            *p = 0.min(*p)
                        }
                    }

                    for f in &mut subfactors {
                        if let Some(v) = f.get_mut(ff) {
                            *v -= *p;
                        } else {
                            f.insert(ff.clone(), -*p);
                        }
                    }
                }

                factors.retain(|_, p| *p != 0);

                // construct the sum factor
                let mut sum = ws.new_atom();
                let mut mm = ws.new_atom();
                let a = sum.to_add();
                for f in subfactors {
                    let m = mm.to_mul();
                    for (k, v) in f {
                        if v == 0 {
                            if m.get_nargs() == 0 {
                                m.extend(ws.new_num(1).as_view());
                            }
                        } else if v == 1 {
                            m.extend(k.as_view());
                        } else {
                            let mut pow = ws.new_atom();
                            let exp = ws.new_num(v as i64);
                            pow.to_pow(k.as_view(), exp.as_view());
                            m.extend(pow.as_view());
                        }
                    }

                    a.extend(mm.as_view());
                }

                let mut out = Atom::new();
                sum.as_view().normalize(ws, &mut out);

                *factors.entry(out.into()).or_insert(0) += 1;
            }
            AtomView::Mul(m) => {
                let mut new_factors = HashMap::default();
                for arg in m {
                    arg.collect_factors_impl(ws, &mut new_factors);

                    // merge factors
                    for (k, v) in new_factors.drain() {
                        *factors.entry(k).or_insert(0) += v;
                    }
                }
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();

                let mut new_factors = HashMap::default();
                b.collect_factors_impl(ws, &mut new_factors);

                if let Ok(n) = i64::try_from(e) {
                    for (f, p) in new_factors {
                        *factors.entry(f).or_insert(0) += n as isize * p;
                    }
                } else {
                    // TODO: extract number from sum in exponent, e.g. x^(a+2)?
                    let mut pow = ws.new_atom();
                    let mut prod = ws.new_atom();
                    let p = prod.to_mul();
                    for (k, v) in new_factors {
                        if v == 1 {
                            p.extend(k.as_view());
                        } else {
                            let mut pow = ws.new_atom();
                            let exp = ws.new_num(v as i64);
                            pow.to_pow(k.as_view(), exp.as_view());
                            p.extend(pow.as_view());
                        }
                    }

                    pow.to_pow(prod.as_view(), e);

                    let mut out = Atom::new();
                    pow.as_view().normalize(ws, &mut out);

                    *factors.entry(out.into()).or_insert(0) += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{
        atom::{Atom, AtomCore, representation::InlineVar},
        function, parse, symbol,
    };

    #[test]
    fn collect_factors() {
        let input = parse!("v1*(v1+v2*v1+v1^2+v2*(v1+v1^2))");
        let r = input.collect_factors();
        let res = parse!("v1^2*(1+v1+v2+v2*(1+v1))");
        assert_eq!(r, res);
    }

    #[test]
    fn collect_symbol() {
        let input = parse!("f1 + v1*f1 + f1(5,3)*v1 + f1(5,3)*v2 + f1(5,3)*f1(7,5)");
        let x = symbol!("f1");

        let r = input.collect_symbol::<i8>(x, None, None);
        let res = parse!("f1*(v1+1)+(v1+v2)*f1(5,3)+f1(5,3)*f1(7,5)");
        assert_eq!(r, res);
    }

    #[test]
    fn collect_num() {
        let input = parse!("2*v1+4*v1^2+6*v1^3");
        let out = input.collect_num();
        let ref_out = parse!("2*(v1+2v1^2+3v1^3)");
        assert_eq!(out, ref_out);

        let input = parse!("(-3*v1+3*v2)(2*v3+2*v4)");
        let out = input.collect_num();
        let ref_out = parse!("-6*(v4+v3)*(v1-v2)");
        assert_eq!(out, ref_out);

        let input = parse!("v1+v2+2*(v1+v2)");
        let out = input.expand_num().collect_num();
        let ref_out = parse!("3*(v1+v2)");
        assert_eq!(out, ref_out);
    }

    #[test]
    fn coefficient_list() {
        let input = parse!("v1*(1+v3)+v1*5*v2+f1(5,v1)+2+v2^2+v1^2+v1^3");
        let x = symbol!("v1");

        let r = input.coefficient_list::<i8>(&[InlineVar::new(x)]);

        let res = vec![
            (parse!("1"), parse!("v2^2+f1(5,v1)+2")),
            (parse!("v1"), parse!("v3+5*v2+1")),
            (parse!("v1^2"), parse!("1")),
            (parse!("v1^3"), parse!("1")),
        ];

        assert_eq!(r, res);
    }

    #[test]
    fn collect() {
        let input = parse!("v1*(1+v3)+v1*5*v2+f1(5,v1)+2+v2^2+v1^2+v1^3");
        let x = symbol!("v1");

        let out = input.collect::<i8>(InlineVar::new(x), None, None);

        let ref_out = parse!("v1^2+v1^3+v2^2+f1(5,v1)+v1*(5*v2+v3+1)+2");
        assert_eq!(out, ref_out)
    }

    #[test]
    fn collect_nested() {
        let input = parse!("(1+v1)^2*v1+(1+v2)^100");
        let x = symbol!("v1");

        let out = input.collect::<i8>(InlineVar::new(x), None, None);

        let ref_out = parse!("v1+2*v1^2+v1^3+(v2+1)^100");
        assert_eq!(out, ref_out)
    }

    #[test]
    fn collect_wrap() {
        let input = parse!("v1*(1+v3)+v1*5*v2+f1(5,v1)+2+v2^2+v1^2+v1^3");
        let x = symbol!("v1");
        let key = symbol!("f3");
        let coeff = symbol!("f4");
        println!("> Collect in x with wrapping:");
        let out = input.collect::<i8>(
            InlineVar::new(x),
            Some(Box::new(move |a, out| {
                out.set_from_view(&a);
                *out = function!(key, out);
            })),
            Some(Box::new(move |a, out| {
                out.set_from_view(&a);
                *out = function!(coeff, out);
            })),
        );

        let ref_out =
            parse!("f3(1)*f4(v2^2+f1(5,v1)+2)+f3(v1)*f4(5*v2+v3+1)+f3(v1^2)*f4(1)+f3(v1^3)*f4(1)");

        assert_eq!(out, ref_out);
    }

    #[test]
    fn together() {
        let input = parse!("v1^2/2+v1^3/v4*v2+v3/(1+v4)");
        let out = input.together();

        let ref_out = parse!("(2*v4+2*v4^2)^-1*(2*v3*v4+v1^2*v4+v1^2*v4^2+2*v1^3*v2+2*v1^3*v2*v4)");

        assert_eq!(out, ref_out);
    }

    #[test]
    fn apart() {
        let input = parse!("(2*v4+2*v4^2)^-1*(2*v3*v4+v1^2*v4+v1^2*v4^2+2*v1^3*v2+2*v1^3*v2*v4)");
        let out = input.apart(symbol!("v4"));

        let ref_out = parse!("1/2*v1^2+v3*(v4+1)^-1+v1^3*v2*v4^-1");

        assert_eq!(out, ref_out);
    }

    #[test]
    fn cancel() {
        let input = parse!("1/(v1+1)^2 + (v1^2 - 1)*(v2+1)^10/(v1 - 1)+ 5 + (v1+1)/(v1^2+2v1+1)");
        let out = input.cancel();

        let ref_out = parse!("(v1+1)^-2+(v1+1)^-1+(v1+1)*(v2+1)^10+5");

        assert_eq!(out, ref_out);
    }

    #[test]
    fn factor() {
        let input = parse!("(6 + v1)/(7776 + 6480*v1 + 2160*v1^2 + 360*v1^3 + 30*v1^4 + v1^5)");
        let out = input.factor();

        let ref_out = parse!("(v1+6)^-4");

        assert_eq!(out, ref_out);
    }

    #[test]
    fn coefficient_list_multiple() {
        let input = parse!(
            "(v1+v2+v3)^2+v1+v1^2+ v2 + 5*v1*v2^2 + v3 + v2*(v4+1)^10 + v1*v5(1,2,3)^2 + v5(1,2)"
        );

        let out = input.as_view().coefficient_list::<i16, _>(&[
            Atom::var(symbol!("v1")),
            Atom::var(symbol!("v2")),
            parse!("v5(1,2,3)"),
        ]);

        assert_eq!(out.len(), 8);
    }
}
