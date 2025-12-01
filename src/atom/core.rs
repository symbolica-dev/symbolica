//! Provide the basic operations on general expressions.
//!
//! See [AtomCore] for the possible operations.

use ahash::{HashMap, HashSet};
use rayon::ThreadPool;

use crate::{
    atom::{
        AddView, FunctionBuilder, Indeterminate, KeyLookup, MulView, NumView, VarView,
        representation::FunView,
    },
    coefficient::{Coefficient, CoefficientView, ConvertToRing},
    domains::{
        EuclideanDomain, InternalOrdering,
        atom::AtomField,
        factorized_rational_polynomial::{
            FactorizedRationalPolynomial, FromNumeratorAndFactorizedDenominator,
        },
        float::{Complex, Real, SingleFloat},
        integer::Z,
        rational::Rational,
        rational_polynomial::{
            FromNumeratorAndDenominator, RationalPolynomial, RationalPolynomialField,
        },
    },
    evaluate::{EvalTree, EvaluationFn, ExpressionEvaluator, FunctionMap, OptimizationSettings},
    id::{
        BorrowReplacement, Condition, ConditionResult, Context, MatchSettings, Pattern,
        PatternAtomTreeIterator, PatternRestriction, ReplaceBuilder,
    },
    poly::{
        Exponent, PolyVariable, PositiveExponent, factor::Factorize, gcd::PolynomialGCD,
        polynomial::MultivariatePolynomial, series::Series,
    },
    printer::{AtomPrinter, CanonicalOrderingSettings, PrintOptions, PrintState},
    solve::SolveError,
    state::Workspace,
    tensors::{CanonicalTensor, matrix::Matrix},
    utils::{BorrowedOrOwned, Settable},
};
use std::sync::Arc;

use super::{
    Atom, AtomOrView, AtomView, ListSlice, Symbol,
    representation::{InlineNum, InlineVar},
};

/// All core features of expressions, such as expansion and
/// pattern matching that leave the expression unchanged.
///
///
/// This trait is sealed, such that new methods can be added
/// without breaking existing implementations.
pub trait AtomCore: private::Sealed {
    /// Take a view of the atom.
    fn as_atom_view(&self) -> AtomView<'_>;

    /// Get a function view if the atom is a function.
    fn as_fun_view(&self) -> Option<FunView<'_>> {
        match self.as_atom_view() {
            AtomView::Fun(f) => Some(f),
            _ => None,
        }
    }

    /// Get a variable view if the atom is a variable.
    fn as_var_view(&self) -> Option<VarView<'_>> {
        match self.as_atom_view() {
            AtomView::Var(v) => Some(v),
            _ => None,
        }
    }

    /// Get a numerical view if the atom is a number.
    fn as_num_view(&self) -> Option<NumView<'_>> {
        match self.as_atom_view() {
            AtomView::Num(n) => Some(n),
            _ => None,
        }
    }

    /// Get a multiplication view if the atom is a multiplication.
    fn as_mul_view(&self) -> Option<MulView<'_>> {
        match self.as_atom_view() {
            AtomView::Mul(m) => Some(m),
            _ => None,
        }
    }

    /// Get an addition view if the atom is an addition.
    fn as_add_view(&self) -> Option<AddView<'_>> {
        match self.as_atom_view() {
            AtomView::Add(a) => Some(a),
            _ => None,
        }
    }

    /// Export the atom and state to a binary stream. It can be loaded
    /// with [Atom::import].
    fn export<W: std::io::Write>(&self, dest: W) -> Result<(), std::io::Error> {
        self.as_atom_view().export(dest)
    }

    /// Get the symbol of a variable or function.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// let atom = parse!("f(x)");
    /// assert_eq!(atom.get_symbol(), Some(symbol!("f")));
    /// ```
    #[inline(always)]
    fn get_symbol(&self) -> Option<Symbol> {
        match self.as_atom_view() {
            AtomView::Var(v) => Some(v.get_symbol()),
            AtomView::Fun(f) => Some(f.get_symbol()),
            _ => None,
        }
    }

    /// Collect terms involving the same power of `x`, where `x` is a variable or function, e.g.
    ///
    /// ```math
    /// collect(x + x * y + x^2, x) = x * (1+y) + x^2
    /// ```
    ///
    /// Use [collect_symbol](AtomCore::collect_symbol) to collect using the name of a function only.
    ///
    /// Both the *key* (the quantity collected in) and its coefficient can be mapped using
    /// `key_map` and `coeff_map` respectively.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("x + x * y + x^2");
    /// let x = parse!("x");
    /// let collected = expr.collect::<u8>(x, None, None);
    /// assert_eq!(collected, parse!("x * (1 + y) + x^2"));
    /// ```
    fn collect<E: Exponent>(
        &self,
        x: impl AtomCore,
        key_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
        coeff_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
    ) -> Atom {
        self.as_atom_view().collect::<E, _>(x, key_map, coeff_map)
    }

    /// Collect terms involving the same power of variables or functions with the name `x`, e.g.
    ///
    /// ```math
    /// collect_symbol(f(1,2) + x*f*(1,2), f) = (1+x)*f(1,2)
    /// ```
    ///
    ///
    /// Both the *key* (the quantity collected in) and its coefficient can be mapped using
    /// `key_map` and `coeff_map` respectively.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse, symbol};
    /// let expr = parse!("f(1,2) + x*f(1,2)");
    /// let collected = expr.collect_symbol::<u8>(symbol!("f"), None, None);
    /// assert_eq!(collected, parse!("(1+x)*f(1,2)"));
    /// ```
    fn collect_symbol<E: Exponent>(
        &self,
        x: Symbol,
        key_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
        coeff_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
    ) -> Atom {
        self.as_atom_view()
            .collect_symbol::<E>(x, key_map, coeff_map)
    }

    /// Collect terms involving the same power of `x`, where `x` is a variable or function, e.g.
    ///
    /// ```math
    /// collect(x + x * y + x^2, x) = x * (1+y) + x^2
    /// ```
    ///
    /// Both the *key* (the quantity collected in) and its coefficient can be mapped using
    /// `key_map` and `coeff_map` respectively.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("x + x * y + x^2 + z + z^2");
    /// let x = parse!("x");
    /// let z = parse!("z");
    /// let collected = expr.collect_multiple::<u8>(&[x, z], None, None);
    /// assert_eq!(collected, parse!("x * (1 + y) + x^2 + z + z^2"));
    /// ```
    fn collect_multiple<E: Exponent>(
        &self,
        xs: &[impl AtomCore],
        key_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
        coeff_map: Option<Box<dyn Fn(AtomView, &mut Atom)>>,
    ) -> Atom {
        self.as_atom_view()
            .collect_multiple::<E, _>(xs, key_map, coeff_map)
    }

    /// Collect common factors from (nested) sums.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("x*(x+y*x+x^2+y*(x+x^2))");
    /// let collected = expr.collect_factors();
    /// assert_eq!(collected, parse!("x^2*(1+x+y+y*(1+x))"));
    /// ```
    fn collect_factors(&self) -> Atom {
        self.as_atom_view().collect_factors()
    }

    /// Collect terms involving the same power of `x` in `xs`, where `xs` is a list of indeterminates.
    /// Return the list of key-coefficient pairs
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("x + x * y + x^2 + z + z^2");
    /// let x = parse!("x");
    /// let z = parse!("z");
    /// let coeff_list = expr.coefficient_list::<u8>(&[x, z]);
    /// assert_eq!(coeff_list.len(), 4);
    /// ```
    fn coefficient_list<E: Exponent>(&self, xs: &[impl AtomCore]) -> Vec<(Atom, Atom)> {
        self.as_atom_view().coefficient_list::<E, _>(xs)
    }

    /// Collect terms involving the literal occurrence of `x`.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("x + x * y + x^2");
    /// let x = parse!("x");
    /// let coeff = expr.coefficient(x);
    /// let r = parse!("1+y");
    /// assert_eq!(coeff, coeff);
    /// ```
    fn coefficient<T: AtomCore>(&self, x: T) -> Atom {
        Workspace::get_local().with(|ws| {
            self.as_atom_view()
                .coefficient_with_ws(x.as_atom_view(), ws)
        })
    }

    /// Write the expression over a common denominator.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("1/x + 1/y");
    /// let together = expr.together();
    /// let r = parse!("(x + y) / (x * y)");
    /// assert_eq!(together, r);
    /// ```
    fn together(&self) -> Atom {
        self.as_atom_view().together()
    }

    /// Write the expression as a sum of terms with minimal denominators in `x`.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// let expr = parse!("(x + y) / (x * y)");
    /// let apart = expr.apart(symbol!("x"));
    /// let r = parse!("1 / y + 1 / x");
    /// assert_eq!(apart, r);
    /// ```
    fn apart<'a, V: Into<BorrowedOrOwned<'a, Indeterminate>>>(&self, x: V) -> Atom {
        self.as_atom_view().apart(x.into().borrow())
    }

    /// Write the expression as a sum of terms with minimal denominators in all variables.
    /// This method computes a Groebner basis and may therefore be slow for large inputs.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// let expr = parse!("(2y-x)/(y*(x+y)*(y-x))");
    /// let apart = expr.apart_multivariate();
    /// let r = parse!("3/(2*y*x+2*y^2)+1/(2*y^2-2*x*y)");
    /// assert_eq!(apart, r);
    /// ```
    fn apart_multivariate(&self) -> Atom {
        self.as_atom_view().apart_multivariate()
    }

    /// Cancel all common factors between numerators and denominators.
    /// Any non-canceling parts of the expression will not be rewritten.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("(x^2 - 1) / (x - 1)");
    /// let canceled = expr.cancel();
    /// let r = parse!("x+1");
    /// assert_eq!(canceled, r);
    /// ```
    fn cancel(&self) -> Atom {
        self.as_atom_view().cancel()
    }

    /// Factor the expression over the rationals.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("x^2 - 1");
    /// let factored = expr.factor();
    /// let r = parse!("(x - 1) * (x + 1)");
    /// assert_eq!(factored, r);
    /// ```
    fn factor(&self) -> Atom {
        self.as_atom_view().factor()
    }

    /// Collect numerical factors by removing the numerical content from additions.
    /// For example, `-2*x + 4*x^2 + 6*x^3` will be transformed into `-2*(x - 2*x^2 - 3*x^3)`.
    ///
    /// The first argument of the addition is normalized to a positive quantity.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("-2*x + 4*x^2 + 6*x^3");
    /// let collected_num = expr.collect_num();
    /// let r = parse!("-2 * (x - 2 * x^2 - 3 * x^3)");
    /// assert_eq!(collected_num, r);
    /// ```
    fn collect_num(&self) -> Atom {
        self.as_atom_view().collect_num()
    }

    /// Expand an expression. The function [AtomCore::expand_via_poly] may be faster.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("(x + 1)^2");
    /// let expanded = expr.expand();
    /// let r = parse!("x^2 + 2 * x + 1");
    /// assert_eq!(expanded, r);
    /// ```
    fn expand(&self) -> Atom {
        self.as_atom_view().expand()
    }

    /// Expand the expression by converting it to a polynomial, optionally
    /// only in the indeterminate `var`. The parameter `E` should be a numerical type
    /// that fits the largest exponent in the expanded expression. Often,
    /// `u8` or `u16` is sufficient.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse};
    /// let expr = parse!("(x + 1)^2");
    /// let expanded = expr.expand_via_poly::<u8, Atom>(None);
    /// let r = parse!("x^2 + 2 * x + 1");
    /// assert_eq!(expanded, r);
    /// ```
    fn expand_via_poly<E: Exponent, T: AtomCore>(&self, var: impl Into<Option<T>>) -> Atom {
        self.as_atom_view()
            .expand_via_poly::<E>(var.into().as_ref().map(|x| x.as_atom_view()))
    }

    /// Expand an expression in the variable `var`. The function [AtomCore::expand_via_poly] may be faster.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("(x + 1)^2");
    /// let x = parse!("x");
    /// let expanded = expr.expand_in(x);
    /// let r = parse!("x^2 + 2 * x + 1");
    /// assert_eq!(expanded, r);
    /// ```
    fn expand_in<T: AtomCore>(&self, var: T) -> Atom {
        self.as_atom_view().expand_in(var.as_atom_view())
    }

    /// Expand an expression in the variable `var`.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// let expr = parse!("(x + 1)^2");
    /// let expanded = expr.expand_in_symbol(symbol!("x"));
    /// let r = parse!("x^2 + 2 * x + 1");
    /// assert_eq!(expanded, r);
    /// ```
    fn expand_in_symbol(&self, var: Symbol) -> Atom {
        self.as_atom_view()
            .expand_in(InlineVar::from(var).as_view())
    }

    /// Expand an expression, returning `true` iff the expression changed.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse};
    /// let expr = parse!("(x + 1)^2");
    /// let mut out = Atom::new();
    /// let changed = expr.expand_into::<Atom>(None, &mut out);
    /// let r = parse!("x^2 + 2 * x + 1");
    /// assert!(changed);
    /// assert_eq!(out, r);
    /// ```
    fn expand_into<T: AtomCore>(&self, var: impl Into<Option<T>>, out: &mut Atom) -> bool {
        self.as_atom_view()
            .expand_into(var.into().as_ref().map(|x| x.as_atom_view()), out)
    }

    /// Distribute numbers in the expression, for example:
    /// `2*(x+y)` -> `2*x+2*y`.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("2*(x+y)");
    /// let expanded_num = expr.expand_num();
    /// let r = parse!("2 * x + 2 * y");
    /// assert_eq!(expanded_num, r);
    /// ```
    fn expand_num(&self) -> Atom {
        self.as_atom_view().expand_num()
    }

    /// Check if the expression is expanded, optionally in only the variable or function `var`.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse};
    /// let expr = parse!("x^2 + 2*x + 1");
    /// let is_expanded = expr.is_expanded::<Atom>(None);
    /// assert!(is_expanded);
    /// ```
    fn is_expanded<T: AtomCore>(&self, var: Option<T>) -> bool {
        self.as_atom_view()
            .is_expanded(var.as_ref().map(|x| x.as_atom_view()))
    }

    /// Take a derivative of the expression with respect to `x`.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// let expr = parse!("x^2 + 2*x + 1");
    /// let derivative = expr.derivative(symbol!("x"));
    /// let r = parse!("2 * x + 2");
    /// assert_eq!(derivative, r);
    /// ```
    fn derivative<'a, V: Into<BorrowedOrOwned<'a, Indeterminate>>>(&self, x: V) -> Atom {
        self.as_atom_view().derivative(x.into().borrow())
    }

    /// Take a derivative of the expression with respect to `x` and
    /// write the result in `out`.
    /// Returns `true` if the derivative is non-zero.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// let expr = parse!("x^2 + 2*x + 1");
    /// let mut out = Atom::new();
    /// let non_zero = expr.derivative_into(symbol!("x"), &mut out);
    /// assert!(non_zero);
    /// assert_eq!(out, parse!("2 * x + 2"));
    /// ```
    fn derivative_into<'a, V: Into<BorrowedOrOwned<'a, Indeterminate>>>(
        &self,
        x: V,
        out: &mut Atom,
    ) -> bool {
        self.as_atom_view().derivative_into(x.into().borrow(), out)
    }

    /// Series expand in `x` around `expansion_point` to depth `depth`.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// let expr = parse!("exp(x)");
    /// let series = expr
    ///     .series(symbol!("x"), Atom::num(0), (4, 1).into(), true)
    ///     .unwrap();
    /// assert_eq!(
    ///     series.to_atom(),
    ///     parse!("1 + x + x^2 / 2 + x^3 / 6 + x^4 / 24")
    /// );
    /// ```
    fn series<'a, T: AtomCore, V: Into<BorrowedOrOwned<'a, Indeterminate>>>(
        &self,
        x: V,
        expansion_point: T,
        depth: Rational,
        depth_is_absolute: bool,
    ) -> Result<Series<AtomField>, String> {
        self.as_atom_view().series(
            x.into().borrow(),
            expansion_point.as_atom_view(),
            depth,
            depth_is_absolute,
        )
    }

    /// Find the root of a function in `x` numerically over the reals using Newton's method.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// let expr = parse!("x^2 - 2");
    /// let root = expr.nsolve(symbol!("x"), 1.0, 1e-7, 100).unwrap();
    /// assert!((root - 1.414213562373095).abs() < 1e-7);
    /// ```
    fn nsolve<
        'a,
        N: SingleFloat + Real + PartialOrd,
        V: Into<BorrowedOrOwned<'a, Indeterminate>>,
    >(
        &self,
        x: V,
        init: N,
        prec: N,
        max_iterations: usize,
    ) -> Result<N, String> {
        self.as_atom_view()
            .nsolve(x.into().borrow(), init, prec, max_iterations)
    }

    /// Solve a non-linear system numerically over the reals using Newton's method.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// use symbolica::domains::float::F64;
    /// let expr1 = parse!("x^2 + y^2 - 1");
    /// let expr2 = parse!("x^2 - y");
    /// let system = &[expr1, expr2];
    /// let vars = &[symbol!("x").into(), symbol!("y").into()];
    /// let init = &[F64::from(0.5), F64::from(0.5)];
    /// let roots = Atom::nsolve_system(system, vars, init, 1e-7.into(), 100).unwrap();
    /// assert!((roots[0].into_inner() - 0.786151377757424).abs() < 1e-7);
    /// assert!((roots[1].into_inner() - 0.6180339887498941).abs() < 1e-7);
    /// ```
    fn nsolve_system<
        N: SingleFloat + Real + PartialOrd + InternalOrdering + Eq + std::hash::Hash,
        T: AtomCore,
    >(
        system: &[T],
        vars: &[Indeterminate],
        init: &[N],
        prec: N,
        max_iterations: usize,
    ) -> Result<Vec<N>, String> {
        AtomView::nsolve_system(system, vars, init, prec, max_iterations)
    }

    /// Solve a system that is linear in `vars`, if possible.
    /// Each expression in `system` is understood to yield 0.
    ///
    /// If the system is underdetermined, a partial solution is returned
    /// where each bound variable is a linear combination of the free
    /// variables. The free variables are chosen such that they have the
    /// highest index in the `vars` list.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse};
    /// let expr1 = parse!("2*x + y - 1");
    /// let expr2 = parse!("x + y + 1");
    /// let system = &[expr1, expr2];
    /// let vars = &[parse!("x"), parse!("y")];
    /// let solution = Atom::solve_linear_system::<u8, _, _>(system, vars).unwrap();
    /// assert_eq!(solution, [Atom::num(2), Atom::num(-3)]);
    /// ```
    ///
    /// Underdetermined system example:
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, solve::SolveError, symbol};
    /// let (v1, v2, v3) = symbol!("v1", "v2", "v3");
    /// let eqs = ["v1 + v2 - 3", "2*v1 + 2*v2 - 6", "v1 + v3 - 5"];
    /// let system: Vec<_> = eqs.iter().map(|e| parse!(e)).collect();
    ///
    /// let sol = Atom::solve_linear_system::<u8, _, Atom>(
    ///     &system,
    ///     &[v1.into(), v2.into(), v3.into()],
    /// );
    ///
    /// assert_eq!(
    ///     sol,
    ///     Err(SolveError::Underdetermined {
    ///         rank: 2,
    ///         partial_solution: vec![parse!("-v3+5"), parse!("v3-2"), parse!("v3"),],
    ///     })
    /// );
    /// ```
    fn solve_linear_system<E: PositiveExponent, T1: AtomCore, T2: AtomCore>(
        system: &[T1],
        vars: &[T2],
    ) -> Result<Vec<Atom>, SolveError> {
        AtomView::solve_linear_system::<E, T1, T2>(system, vars)
    }

    /// Convert a system of linear equations to a matrix representation, returning the matrix
    /// and the right-hand side.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse};
    /// use symbolica::domains::Ring;
    /// let expr1 = parse!("2*x + y - 1");
    /// let expr2 = parse!("x - y + 1");
    /// let system = &[expr1, expr2];
    /// let vars = &[parse!("x"), parse!("y")];
    /// let (matrix, rhs) = Atom::system_to_matrix::<u8, _, _>(system, vars).unwrap();
    /// let one = matrix.field().one();
    /// assert_eq!(
    ///     matrix.into_vec(),
    ///     [&one + &one, one.clone(), one.clone(), -one.clone()]
    /// );
    /// assert_eq!(rhs.into_vec(), [one.clone(), -one]);
    /// ```
    fn system_to_matrix<E: PositiveExponent, T1: AtomCore, T2: AtomCore>(
        system: &[T1],
        vars: &[T2],
    ) -> Result<
        (
            Matrix<RationalPolynomialField<Z, E>>,
            Matrix<RationalPolynomialField<Z, E>>,
        ),
        String,
    > {
        AtomView::system_to_matrix::<E, T1, T2>(system, vars)
    }

    /// Evaluate a (nested) expression a single time.
    /// For repeated evaluations, use [Self::evaluator()] and convert
    /// to an optimized version or generate a compiled version of your expression.
    ///
    /// All variables and all user functions in the expression must occur in the map.
    ///
    /// # Example
    ///
    /// ```
    /// use ahash::HashMap;
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("x + y");
    /// let x = parse!("x");
    /// let y = parse!("y");
    /// let mut const_map = HashMap::default();
    /// const_map.insert(x.clone(), 1.0);
    /// const_map.insert(y.clone(), 2.0);
    /// let result = expr
    ///     .evaluate(|r| r.to_f64(), &const_map, &HashMap::default())
    ///     .unwrap();
    /// assert_eq!(result, 3.0);
    /// ```
    fn evaluate<A: AtomCore + KeyLookup, T: Real, F: Fn(&Rational) -> T + Copy>(
        &self,
        coeff_map: F,
        const_map: &HashMap<A, T>,
        function_map: &HashMap<Symbol, EvaluationFn<A, T>>,
    ) -> Result<T, String> {
        self.as_atom_view()
            .evaluate(coeff_map, const_map, function_map)
    }

    /// Convert nested expressions to a tree suitable for repeated evaluations with
    /// different values for `params`.
    /// All variables and all user functions in the expression must occur in the map.
    ///
    /// Consider using [AtomCore::evaluator] instead.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// use symbolica::evaluate::{FunctionMap, OptimizationSettings};
    /// let expr = parse!("x + y");
    /// let x = parse!("x");
    /// let y = parse!("y");
    /// let fn_map = FunctionMap::new();
    /// let params = vec![x.clone(), y.clone()];
    /// let mut tree = expr.to_evaluation_tree(&fn_map, &params).unwrap();
    /// tree.common_subexpression_elimination();
    /// let e = tree.optimize(&OptimizationSettings::default());
    /// let mut e = e.map_coeff(&|c| c.to_real().unwrap().to_f64());
    /// let r = e.evaluate_single(&[0.5, 0.3]);
    /// assert_eq!(r, 0.8);
    /// ```
    fn to_evaluation_tree(
        &self,
        fn_map: &FunctionMap<Complex<Rational>>,
        params: &[Atom],
    ) -> Result<EvalTree<Complex<Rational>>, String> {
        self.as_atom_view().to_evaluation_tree(fn_map, params)
    }

    /// Create an efficient evaluator for a (nested) expression.
    /// All free parameters must appear in `params` and all other variables
    /// and user functions in the expression must occur in the function map.
    /// The function map may have nested expressions.
    ///
    /// # Examples
    ///
    /// A simple evaluation without nested expressions:
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// use symbolica::evaluate::{FunctionMap, OptimizationSettings};
    /// let fn_map = FunctionMap::new();
    /// let params = vec![parse!("x"), parse!("y")];
    /// let optimization_settings = OptimizationSettings::default();
    /// let mut evaluator = parse!("x + y")
    ///     .evaluator(&fn_map, &params, optimization_settings)
    ///     .unwrap()
    ///     .map_coeff(&|x| x.to_real().unwrap().to_f64());
    /// assert_eq!(evaluator.evaluate_single(&[1.0, 2.0]), 3.0);
    /// ```
    ///
    /// An evaluation with a nested function `f(x) = x^2 + 1`:
    /// ```rust
    /// use symbolica::{atom::AtomCore, parse, symbol};
    /// use symbolica::evaluate::{FunctionMap, OptimizationSettings};
    /// let mut fn_map = FunctionMap::new();
    /// fn_map.add_function(symbol!("f"), "f".to_string(), vec![symbol!("x")], parse!("x^2 + 1")).unwrap();
    ///
    /// let optimization_settings = OptimizationSettings::default();
    /// let mut evaluator = parse!("f(x)")
    ///     .evaluator(&fn_map, &vec![parse!("x")], optimization_settings)
    ///     .unwrap().map_coeff(&|x| x.re.to_f64());
    /// assert_eq!(evaluator.evaluate_single(&[2.0]), 5.0);
    /// ```
    ///
    /// An evaluation with externally defined functions:
    /// ```rust
    /// use ahash::HashMap;
    /// use symbolica::{atom::AtomCore, evaluate::{FunctionMap, OptimizationSettings}, parse, symbol};
    ///
    /// let mut ext: HashMap<String, Box<dyn Fn(&[f64]) -> f64 + Send + Sync>> = HashMap::default();
    /// ext.insert("f".to_string(), Box::new(|a| a[0] * a[0] + a[1]));
    ///
    /// let mut f = FunctionMap::new();
    /// f.add_external_function(symbol!("f"), "f".to_string()).unwrap();
    ///
    /// let params = vec![parse!("x"), parse!("y")];
    /// let optimization_settings = OptimizationSettings::default();
    /// let evaluator = parse!("f(x,y)").evaluator(&f, &params, optimization_settings).unwrap().map_coeff(&|x| x.re.to_f64());
    ///
    /// let mut ev = evaluator.with_external_functions(ext).unwrap();
    /// assert_eq!(ev.evaluate_single(&[2.0, 3.0]), 7.0);
    /// ```
    fn evaluator(
        &self,
        fn_map: &FunctionMap<Complex<Rational>>,
        params: &[Atom],
        optimization_settings: OptimizationSettings,
    ) -> Result<ExpressionEvaluator<Complex<Rational>>, String> {
        let mut tree = self.to_evaluation_tree(fn_map, params)?;
        Ok(tree.optimize(&optimization_settings))
    }

    /// Convert nested expressions to a tree suitable for repeated evaluations with
    /// different values for `params`.
    /// All variables and all user functions in the expression must occur in the map.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse};
    /// use symbolica::evaluate::{FunctionMap, OptimizationSettings};
    /// let expr1 = parse!("x + y");
    /// let expr2 = parse!("x - y");
    /// let x = parse!("x");
    /// let y = parse!("y");
    /// let fn_map = FunctionMap::new();
    /// let params = vec![x.clone(), y.clone()];
    /// let evaluator = Atom::evaluator_multiple(
    ///     &[expr1, expr2],
    ///     &fn_map,
    ///     &params,
    ///     OptimizationSettings::default(),
    /// )
    /// .unwrap();
    /// let mut evaluator = evaluator.map_coeff(&|c| c.to_real().unwrap().to_f64());
    /// let mut out = vec![0., 0.];
    /// evaluator.evaluate(&[1.0, 2.0], &mut out);
    /// assert_eq!(out, &[3.0, -1.0]);
    /// ```
    fn evaluator_multiple<A: AtomCore>(
        exprs: &[A],
        fn_map: &FunctionMap<Complex<Rational>>,
        params: &[Atom],
        optimization_settings: OptimizationSettings,
    ) -> Result<ExpressionEvaluator<Complex<Rational>>, String> {
        let mut tree = AtomView::to_eval_tree_multiple(exprs, fn_map, params)?;
        Ok(tree.optimize(&optimization_settings))
    }

    /// Check if the expression could be 0, using (potentially) numerical sampling with
    /// a given tolerance and number of iterations.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// use symbolica::id::ConditionResult;
    /// let expr = parse!("(x+1)^2 - x^2 - 2x - 1");
    /// let result = expr.zero_test(100, 1e-7);
    /// assert_eq!(result, ConditionResult::Inconclusive);
    /// ```
    fn zero_test(&self, iterations: usize, tolerance: f64) -> ConditionResult {
        self.as_atom_view().zero_test(iterations, tolerance)
    }

    /// Set the coefficient ring to the multivariate rational polynomial with `vars` variables.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// let expr = parse!("x*y + x^2*y + y/(1+x)");
    /// let vars = Arc::new(vec![symbol!("x").into()]);
    /// let result = expr.set_coefficient_ring(&vars);
    /// let r = result.set_coefficient_ring(&Arc::new(vec![]));
    /// assert_eq!(r, parse!("y*(x+1)^-1*(x+2*x^2+x^3+1)"));
    /// ```
    fn set_coefficient_ring(&self, vars: &Arc<Vec<PolyVariable>>) -> Atom {
        self.as_atom_view().set_coefficient_ring(vars)
    }

    /// Convert all coefficients and built-in functions to floats with a given precision `decimal_prec`.
    /// The precision of floating point coefficients in the input will be truncated to `decimal_prec`.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("cos(1/3) + 1/2");
    /// let result = expr.to_float(2);
    /// assert_eq!(result.to_string(), "1.4");
    /// ```
    fn to_float(&self, decimal_prec: u32) -> Atom {
        let mut a = Atom::new();
        self.as_atom_view().to_float_into(decimal_prec, &mut a);
        a
    }

    /// Convert all coefficients and built-in functions to floats with a given precision `decimal_prec`.
    /// The precision of floating point coefficients in the input will be truncated to `decimal_prec`.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse};
    /// let expr = parse!("1/3");
    /// let mut out = Atom::new();
    /// expr.to_float_into(2, &mut out);
    /// assert_eq!(out.to_string(), "3.3e-1");
    /// ```
    fn to_float_into(&self, decimal_prec: u32, out: &mut Atom) {
        self.as_atom_view().to_float_into(decimal_prec, out);
    }

    /// Map all coefficients using a given function.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// use symbolica::coefficient::{Coefficient, CoefficientView};
    /// use symbolica::domains::{float::Complex, rational::Rational};
    /// let expr = parse!("0.33*x + 3");
    /// let out = expr.map_coefficient(|c| match c {
    ///     CoefficientView::Natural(r, d, ri, di) => {
    ///         Coefficient::Float(Complex::new(Rational::from((r, d)).to_multi_prec_float(53),
    ///             Rational::from((ri, di)).to_multi_prec_float(53)))
    ///     }
    ///     _ => c.to_owned(),
    /// });
    /// assert_eq!(
    ///     out,
    ///     parse!("3.30000000000000e-1*x+3.00000000000000")
    /// );
    /// ```
    fn map_coefficient<F: Fn(CoefficientView) -> Coefficient + Copy>(&self, f: F) -> Atom {
        self.as_atom_view().map_coefficient(f)
    }

    /// Map all coefficients using a given function.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse};
    /// use symbolica::coefficient::{Coefficient, CoefficientView};
    /// use symbolica::domains::{float::Complex, rational::Rational};
    /// let expr = parse!("0.33*x + 3");
    /// let mut out = Atom::new();
    /// expr.map_coefficient_into(|c| match c {
    ///     CoefficientView::Natural(r, d, ri, di) => {
    ///         Coefficient::Float(Complex::new(Rational::from((r, d)).to_multi_prec_float(53),
    ///             Rational::from((ri, di)).to_multi_prec_float(53)))
    ///     }
    ///     _ => c.to_owned(),
    /// }, &mut out);
    /// assert_eq!(
    ///     out,
    ///     parse!("3.30000000000000e-1*x+3.00000000000000")
    /// );
    /// ```
    fn map_coefficient_into<F: Fn(CoefficientView) -> Coefficient + Copy>(
        &self,
        f: F,
        out: &mut Atom,
    ) {
        self.as_atom_view().map_coefficient_into(f, out);
    }

    /// Map all floating point and rational coefficients to the best rational approximation
    /// in the interval `[self*(1-relative_error),self*(1+relative_error)]`.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse};
    /// let expr = parse!("0.333");
    /// let result = expr.rationalize(&(1, 100).into());
    /// assert_eq!(result, Atom::num((1, 3)));
    /// ```
    fn rationalize(&self, relative_error: &Rational) -> Atom {
        self.as_atom_view().rationalize_coefficients(relative_error)
    }

    /// Convert the atom to a polynomial, optionally in the variable ordering
    /// specified by `var_map`. If new variables are encountered, they are
    /// added to the variable map. Similarly, non-polynomial parts are automatically
    /// defined as a new independent variable in the polynomial.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// use symbolica::domains::rational::Q;
    /// let expr = parse!("x^2 + 2*x + 1");
    /// let poly = expr.to_polynomial::<_,u8>(&Q, None);
    /// assert_eq!(poly.to_expression(), parse!("x^2 + 2 * x + 1"));
    /// ```
    ///
    /// With explicit variable ordering:
    ///
    /// ```
    /// # use std::sync::Arc;
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// use symbolica::domains::rational::Q;
    /// let expr = parse!("x^2 + 2*x + 1");
    /// let var_map = Arc::new(vec![symbol!("x").into()]);
    /// let poly = expr.to_polynomial::<_,u8>(&Q, Some(var_map));
    /// assert_eq!(poly.to_expression(), parse!("x^2 + 2 * x + 1"));
    /// ```
    fn to_polynomial<R: EuclideanDomain + ConvertToRing, E: Exponent>(
        &self,
        field: &R,
        var_map: impl Into<Option<Arc<Vec<PolyVariable>>>>,
    ) -> MultivariatePolynomial<R, E> {
        self.try_to_polynomial(field, var_map.into()).unwrap()
    }

    /// Convert the atom to a polynomial, optionally in the variable ordering
    /// specified by `var_map`. If new variables are encountered, they are
    /// added to the variable map. Similarly, non-polynomial parts are automatically
    /// defined as a new independent variable in the polynomial.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// use symbolica::domains::rational::Q;
    /// let expr = parse!("x^2 + 2*x + 1");
    /// let poly = expr.to_polynomial::<_,u8>(&Q, None);
    /// assert_eq!(poly.to_expression(), parse!("x^2 + 2 * x + 1"));
    /// ```
    ///
    /// With explicit variable ordering:
    ///
    /// ```
    /// # use std::sync::Arc;
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// use symbolica::domains::rational::Q;
    /// let expr = parse!("x^2 + 2*x + 1");
    /// let var_map = Arc::new(vec![symbol!("x").into()]);
    /// let poly = expr.to_polynomial::<_,u8>(&Q, Some(var_map));
    /// assert_eq!(poly.to_expression(), parse!("x^2 + 2 * x + 1"));
    /// ```
    fn try_to_polynomial<R: EuclideanDomain + ConvertToRing, E: Exponent>(
        &self,
        field: &R,
        var_map: impl Into<Option<Arc<Vec<PolyVariable>>>>,
    ) -> Result<MultivariatePolynomial<R, E>, String> {
        self.as_atom_view().try_to_polynomial(field, var_map.into())
    }

    /// Convert the atom to a polynomial in specific variables.
    /// All other parts will be collected into the coefficient, which
    /// is a general expression.
    ///
    /// This routine does not perform expansions.
    ///
    /// # Example
    ///
    /// ```
    /// use std::sync::Arc;
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// let expr = parse!("x^2 + y*x + x + 1");
    /// let var_map = Arc::new(vec![symbol!("x").into()]);
    /// let poly = expr.to_polynomial_in_vars::<u8>(&var_map);
    /// assert_eq!(
    ///     poly.flatten(false),
    ///     parse!("x^2 + (1+y)*x + 1")
    /// );
    /// ```
    fn to_polynomial_in_vars<E: Exponent>(
        &self,
        var_map: &Arc<Vec<PolyVariable>>,
    ) -> MultivariatePolynomial<AtomField, E> {
        self.as_atom_view().to_polynomial_in_vars(var_map)
    }

    /// Convert the atom to a rational polynomial, optionally in the variable ordering
    /// specified by `var_map`. If new variables are encountered, they are
    /// added to the variable map. Similarly, non-rational polynomial parts are automatically
    /// defined as a new independent variable in the rational polynomial.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};;
    /// use symbolica::domains::integer::Z;
    /// use symbolica::domains::rational::Q;
    /// let expr = parse!("(x^2 + 2*x + 1) / (x + 1)");
    /// let rat_poly = expr.to_rational_polynomial::<_, _, u8>(&Q, &Z, None);
    /// assert_eq!(rat_poly.to_expression(), parse!("1+x"));
    /// ```
    fn to_rational_polynomial<
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + PolynomialGCD<E>,
        E: PositiveExponent,
    >(
        &self,
        field: &R,
        out_field: &RO,
        var_map: impl Into<Option<Arc<Vec<PolyVariable>>>>,
    ) -> RationalPolynomial<RO, E>
    where
        RationalPolynomial<RO, E>:
            FromNumeratorAndDenominator<R, RO, E> + FromNumeratorAndDenominator<RO, RO, E>,
    {
        self.try_to_rational_polynomial(field, out_field, var_map)
            .unwrap()
    }

    /// Convert the atom to a rational polynomial, optionally in the variable ordering
    /// specified by `var_map`. If new variables are encountered, they are
    /// added to the variable map. Similarly, non-rational polynomial parts are automatically
    /// defined as a new independent variable in the rational polynomial.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};;
    /// use symbolica::domains::integer::Z;
    /// use symbolica::domains::rational::Q;
    /// let expr = parse!("(x^2 + 2*x + 1) / (x + 1)");
    /// let rat_poly = expr.to_rational_polynomial::<_, _, u8>(&Q, &Z, None);
    /// assert_eq!(rat_poly.to_expression(), parse!("1+x"));
    /// ```
    fn try_to_rational_polynomial<
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + PolynomialGCD<E>,
        E: PositiveExponent,
    >(
        &self,
        field: &R,
        out_field: &RO,
        var_map: impl Into<Option<Arc<Vec<PolyVariable>>>>,
    ) -> Result<RationalPolynomial<RO, E>, String>
    where
        RationalPolynomial<RO, E>:
            FromNumeratorAndDenominator<R, RO, E> + FromNumeratorAndDenominator<RO, RO, E>,
    {
        self.as_atom_view()
            .try_to_rational_polynomial(field, out_field, var_map.into())
    }

    /// Convert the atom to a rational polynomial with factorized denominators, optionally in the variable ordering
    /// specified by `var_map`. If new variables are encountered, they are
    /// added to the variable map. Similarly, non-rational polynomial parts are automatically
    /// defined as a new independent variable in the rational polynomial.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// use symbolica::domains::integer::Z;
    /// use symbolica::domains::rational::Q;
    /// let expr = parse!("(x^2 + 2*x + 1) / (x + 1)");
    /// let fact_rat_poly = expr.to_factorized_rational_polynomial::<_, _, u8>(&Q, &Z, None);
    /// assert_eq!(
    ///     fact_rat_poly.numerator.to_expression(),
    ///     parse!("x+1")
    /// );
    /// ```
    fn to_factorized_rational_polynomial<
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + PolynomialGCD<E>,
        E: PositiveExponent,
    >(
        &self,
        field: &R,
        out_field: &RO,
        var_map: impl Into<Option<Arc<Vec<PolyVariable>>>>,
    ) -> FactorizedRationalPolynomial<RO, E>
    where
        FactorizedRationalPolynomial<RO, E>: FromNumeratorAndFactorizedDenominator<R, RO, E>
            + FromNumeratorAndFactorizedDenominator<RO, RO, E>,
        MultivariatePolynomial<RO, E>: Factorize,
    {
        self.try_to_factorized_rational_polynomial(field, out_field, var_map.into())
            .unwrap()
    }

    /// Convert the atom to a rational polynomial with factorized denominators, optionally in the variable ordering
    /// specified by `var_map`. If new variables are encountered, they are
    /// added to the variable map. Similarly, non-rational polynomial parts are automatically
    /// defined as a new independent variable in the rational polynomial.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// use symbolica::domains::integer::Z;
    /// use symbolica::domains::rational::Q;
    /// let expr = parse!("(x^2 + 2*x + 1) / (x + 1)");
    /// let fact_rat_poly = expr.to_factorized_rational_polynomial::<_, _, u8>(&Q, &Z, None);
    /// assert_eq!(
    ///     fact_rat_poly.numerator.to_expression(),
    ///     parse!("x+1")
    /// );
    /// ```
    fn try_to_factorized_rational_polynomial<
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + PolynomialGCD<E>,
        E: PositiveExponent,
    >(
        &self,
        field: &R,
        out_field: &RO,
        var_map: impl Into<Option<Arc<Vec<PolyVariable>>>>,
    ) -> Result<FactorizedRationalPolynomial<RO, E>, String>
    where
        FactorizedRationalPolynomial<RO, E>: FromNumeratorAndFactorizedDenominator<R, RO, E>
            + FromNumeratorAndFactorizedDenominator<RO, RO, E>,
        MultivariatePolynomial<RO, E>: Factorize,
    {
        self.as_atom_view()
            .try_to_factorized_rational_polynomial(field, out_field, var_map.into())
    }

    /// Format the atom. See [AtomCore::printer] for more convenient printing.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// use symbolica::printer::{PrintOptions, PrintState};
    /// let expr = parse!("x + y");
    /// let mut output = String::new();
    /// expr.format(&mut output, &PrintOptions::file_no_namespace(), PrintState::default()).unwrap();
    /// assert_eq!(output, "x+y");
    /// ```
    fn format<W: std::fmt::Write>(
        &self,
        fmt: &mut W,
        opts: &PrintOptions,
        print_state: PrintState,
    ) -> Result<bool, std::fmt::Error> {
        self.as_atom_view().format(fmt, opts, print_state)
    }

    /// Construct a printer for the atom with special options.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// use symbolica::printer::PrintOptions;
    /// let expr = parse!("x^2");
    /// let opts = PrintOptions {
    ///     double_star_for_exponentiation: true,
    ///     hide_all_namespaces: true,
    ///    ..Default::default()
    /// };
    /// let printer = expr.printer(opts);
    /// assert_eq!(printer.to_string(), "x**2");
    /// ```
    fn printer(&self, opts: PrintOptions) -> AtomPrinter<'_> {
        AtomPrinter::new_with_options(self.as_atom_view(), opts)
    }

    /// Print the atom in a form that is independent of any implementation details, such
    /// as the definition order of the symbols. Use [AtomCore::to_canonical_string] for a fully
    /// canonical representation.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, printer::CanonicalOrderingSettings, symbol};
    /// let (y, x) = symbol!("canon::y", "canon::x");
    /// let expr = x.to_atom() + y;
    /// let canonical_str = expr.to_canonically_ordered_string(CanonicalOrderingSettings {
    ///   include_namespace: false,
    ///   include_attributes: false,
    ///   ..Default::default()
    /// });
    /// assert_eq!(canonical_str, "x+y");
    /// ```
    fn to_canonically_ordered_string(&self, settings: CanonicalOrderingSettings) -> String {
        self.as_atom_view().to_canonically_ordered_string(settings)
    }

    /// Print the atom in a form that is unique and independent of any implementation details.
    /// The resulting string can be parsed back to the same expression.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("x + y");
    /// let canonical_str = expr.to_canonical_string();
    /// assert_eq!(canonical_str, "symbolica::{}::x+symbolica::{}::y");
    /// ```
    fn to_canonical_string(&self) -> String {
        self.as_atom_view().to_canonical_string()
    }

    /// Map the function `f` over all terms.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("x + y");
    /// let result = expr.map_terms_single_core(|term| term.expand());
    /// assert_eq!(result, parse!("x + y"));
    /// ```
    fn map_terms_single_core(&self, f: impl Fn(AtomView) -> Atom) -> Atom {
        self.as_atom_view().map_terms_single_core(f)
    }

    /// Map the function `f` over all terms, using parallel execution with `n_cores` cores.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("x + y");
    /// let result = expr.map_terms(|term| term.expand(), 4);
    /// assert_eq!(result, parse!("x + y"));
    /// ```
    fn map_terms(&self, f: impl Fn(AtomView) -> Atom + Send + Sync, n_cores: usize) -> Atom {
        self.as_atom_view().map_terms(f, n_cores)
    }

    /// Map the function `f` over all terms, using parallel execution with `n_cores` cores.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("x + y");
    /// let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    /// let result = expr.map_terms_with_pool(|term| term.expand(), &pool);
    /// assert_eq!(result, parse!("x + y"));
    /// ```
    fn map_terms_with_pool(
        &self,
        f: impl Fn(AtomView) -> Atom + Send + Sync,
        p: &ThreadPool,
    ) -> Atom {
        self.as_atom_view().map_terms_with_pool(f, p)
    }

    /// Canonize (products of) tensors in the expression by relabeling repeated indices.
    /// The tensors must be written as functions, with its indices as the arguments.
    /// Subexpressions, constants and open indices are supported.
    ///
    /// If the contracted indices are distinguishable (for example in their dimension),
    /// you can provide a group marker as the second element in the tuple of the index
    /// specification.
    /// This makes sure that an index will not be renamed to an index from a different group.
    ///
    /// Returns the canonical expression, as well as the external indices and ordered dummy indices
    /// appearing in the canonical expression.
    ///
    /// Example
    /// -------
    /// ```
    /// # use symbolica::{atom::{Atom, AtomCore}, symbol, parse};
    /// #
    /// # fn main() {
    /// let _ = symbol!("fs"; Symmetric);
    /// let _ = symbol!("fc"; Cyclesymmetric);
    /// let a = parse!("fs(mu2,mu3)*fc(mu4,mu2,k1,mu4,k1,mu3)");
    ///
    /// let mu1 = parse!("mu1");
    /// let mu2 = parse!("mu2");
    /// let mu3 = parse!("mu3");
    /// let mu4 = parse!("mu4");
    ///
    /// let r = a.canonize_tensors([(mu1, 0), (mu2, 0), (mu3, 0), (mu4, 0)]).unwrap();
    /// println!("{}", r.canonical_form);
    /// # }
    /// ```
    /// yields `fs(mu1,mu2)*fc(mu1,k1,mu3,k1,mu2,mu3)`.
    fn canonize_tensors<I, T: AtomCore, G: Ord + std::hash::Hash>(
        &self,
        indices: I,
    ) -> Result<CanonicalTensor<T, G>, String>
    where
        I: IntoIterator<Item = (T, G)>,
    {
        self.as_atom_view().canonize_tensors(indices)
    }

    fn to_pattern(&self) -> Pattern {
        Pattern::from_view(self.as_atom_view(), true)
    }

    /// Get all symbols in the expression, optionally including function symbols.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// let expr = parse!("x + y");
    /// let symbols = expr.get_all_symbols(true);
    /// assert!(symbols.contains(&symbol!("x")));
    /// assert!(symbols.contains(&symbol!("y")));
    /// ```
    fn get_all_symbols(&self, include_function_symbols: bool) -> HashSet<Symbol> {
        self.as_atom_view()
            .get_all_symbols(include_function_symbols)
    }

    /// Get all variables and functions in the expression.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// let expr = parse!("x + f(x)");
    /// let indeterminates = expr.get_all_indeterminates(true);
    /// assert!(indeterminates.contains(&Atom::var(symbol!("x")).as_view()));
    /// assert!(indeterminates.contains(&parse!("f(x)").as_view()));
    /// ```
    fn get_all_indeterminates(&self, enter_functions: bool) -> HashSet<AtomView<'_>> {
        self.as_atom_view().get_all_indeterminates(enter_functions)
    }

    /// Returns true iff `self` contains the symbol `s`.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// let expr = parse!("x + y");
    /// let contains_x = expr.contains_symbol(symbol!("x"));
    /// assert!(contains_x);
    /// ```
    fn contains_symbol(&self, s: Symbol) -> bool {
        self.as_atom_view().contains_symbol(s)
    }

    /// Returns true iff `self` contains `a` literally.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("x + y");
    /// let x = parse!("x");
    /// let contains_x = expr.contains(x);
    /// assert!(contains_x);
    /// ```
    fn contains<T: AtomCore>(&self, s: T) -> bool {
        self.as_atom_view().contains(s.as_atom_view())
    }

    /// Returns true iff `self` is scalar, i.e. contains only numbers and symbols with the `Scalar` attribute.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// let _ = symbol!("x_scalar"; Scalar);
    /// let expr = parse!("3*2^x_scalar + (1+x_scalar)^2");
    /// assert!(expr.is_scalar());
    /// ```
    fn is_scalar(&self) -> bool {
        self.as_atom_view().is_scalar()
    }

    /// Returns true iff an expression is real. Symbols must have the `Real` attribute.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// let _ = symbol!("x_real"; Real);
    /// let expr = parse!("3*2^x_real + (1+x_real)^2 + (1/2)^x_real");
    /// assert!(expr.is_real());
    /// ```
    fn is_real(&self) -> bool {
        self.as_atom_view().is_real()
    }

    /// Returns true iff an expression only consists of integer numbers and symbols with the `Integer` attribute.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// let _ = symbol!("x_integer"; Integer);
    /// let expr = parse!("3*2^x_integer + (1+x_integer)^2");
    /// assert!(expr.is_integer());
    /// ```
    fn is_integer(&self) -> bool {
        self.as_atom_view().is_integer()
    }

    /// Returns true iff an expression is positive. Symbols must have the `Positive` attribute.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// let _ = symbol!("x_p"; Positive);
    /// let expr = parse!("3*2^x_p + (1+x_p)^2 + (1/2)^x_p");
    /// assert!(expr.is_positive());
    /// ```
    fn is_positive(&self) -> bool {
        self.as_atom_view().is_positive()
    }

    /// Returns true iff an expression has no explicit infinities and is not indeterminate.
    ///
    /// # Example
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// let expr = parse!("3x + x^2 + log(0)");
    /// assert!(!expr.is_finite());
    /// ```
    fn is_finite(&self) -> bool {
        self.as_atom_view().is_finite()
    }

    /// Returns true iff an expression is constant, i.e. contains no user-defined symbols or functions.
    ///
    /// # Example
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// let expr = parse!("cos(2 + exp(3) ) + 1/3");
    /// assert!(expr.is_constant());
    /// ```
    fn is_constant(&self) -> bool {
        self.as_atom_view().is_constant()
    }

    /// Check if the expression can be considered a polynomial in some variables, including
    /// redefinitions. For example `f(x)+y` is considered a polynomial in `f(x)` and `y`, whereas
    /// `f(x)+x` is not a polynomial.
    ///
    /// Rational powers or powers in variables are not rewritten, e.g. `x^(2y)` is not considered
    /// polynomial in `x^y`.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("f(x) + y");
    /// let is_poly = expr.is_polynomial(true, false);
    /// assert!(is_poly.is_some());
    /// ```
    fn is_polynomial(
        &self,
        allow_not_expanded: bool,
        allow_negative_powers: bool,
    ) -> Option<HashSet<AtomView<'_>>> {
        self.as_atom_view()
            .is_polynomial(allow_not_expanded, allow_negative_powers)
    }

    /// Exponentiate the atom.
    fn exp(&self) -> Atom {
        FunctionBuilder::new(Symbol::EXP)
            .add_arg(self.as_atom_view())
            .finish()
    }

    /// Take the logarithm of the atom.
    fn log(&self) -> Atom {
        FunctionBuilder::new(Symbol::LOG)
            .add_arg(self.as_atom_view())
            .finish()
    }

    /// Take the sine the atom.
    fn sin(&self) -> Atom {
        FunctionBuilder::new(Symbol::SIN)
            .add_arg(self.as_atom_view())
            .finish()
    }

    /// Take the cosine the atom.
    fn cos(&self) -> Atom {
        FunctionBuilder::new(Symbol::COS)
            .add_arg(self.as_atom_view())
            .finish()
    }

    /// Take the square root of the atom.
    fn sqrt(&self) -> Atom {
        FunctionBuilder::new(Symbol::SQRT)
            .add_arg(self.as_atom_view())
            .finish()
    }

    /// Take the complex conjugate of the atom.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("x+2 + 3^x + (5+2i) * (test::{real}::real) + (-2)^x");
    /// let result = expr.conj();
    /// assert_eq!(result, parse!("(5-2𝑖)*test::real+3^conj(x)+conj(x)+conj((-2)^x)+2"));
    /// ```
    fn conj(&self) -> Atom {
        FunctionBuilder::new(Symbol::CONJ)
            .add_arg(self.as_atom_view())
            .finish()
    }

    /// Replace all occurrences of the pattern. The right-hand side is
    /// either another pattern, or a function that maps the matched wildcards to a new expression.
    ///
    /// # Examples
    ///
    /// Replace all occurrences of `x` with `z`:
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// use symbolica::id::Pattern;
    /// let expr = parse!("x + y");
    /// let pattern = parse!("x");
    /// let replacement = parse!("z");
    /// let result = expr.replace(pattern).with(replacement);
    /// assert_eq!(result, parse!("z + y"));
    /// ```
    ///
    /// Set a condition `x_ > 1` (conditions can be chained with `&` and `|`):
    /// ```
    /// use symbolica::id::Pattern;
    /// use symbolica::{atom::AtomCore, parse, symbol};
    /// let expr = parse!("f(1) + f(2) + f(3)");
    /// let out = expr
    ///     .replace(parse!("f(x_)"))
    ///     .when(symbol!("x_").filter(|x| x.to_atom() > 1))
    ///     .with(parse!("f(x_ - 1)"));
    /// assert_eq!(out, parse!("2*f(1) + f(2)"));
    /// ```
    ///
    /// Use a map as a right-hand side:
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, function, parse, printer::PrintOptions, symbol};
    /// let (f, x_) = symbol!("f", "x_");
    /// let a = function!(f, 1) * function!(f, 3);
    /// let p = function!(f, x_);
    ///
    /// let r = a.replace(p).with_map(move |m| {
    ///     function!(
    ///         f,
    ///         parse!(&format!(
    ///             "p{}",
    ///             m.get(x_)
    ///                 .unwrap()
    ///                 .to_atom()
    ///                 .printer(PrintOptions::file()),
    ///         ))
    ///     )
    /// });
    /// let res = parse!("f(p1)*f(p3)");
    /// assert_eq!(r, res);
    /// ```
    ///
    /// Access the match stack to filter for an ascending order of `x`, `y`, `z`:
    /// ```
    /// use symbolica::id::{Condition, ConditionResult, Pattern};
    /// use symbolica::{atom::AtomCore, parse, symbol};
    /// let expr = parse!("f(1, 2, 3)");
    /// let out = expr
    ///     .replace(parse!("f(x_,y_,z_)"))
    ///     .when(Condition::match_stack(|m| {
    ///         if let Some(x) = m.get(symbol!("x_")) {
    ///             if let Some(y) = m.get(symbol!("y_")) {
    ///                 if x.to_atom() > y.to_atom() {
    ///                     return ConditionResult::False;
    ///                 }
    ///                 if let Some(z) = m.get(symbol!("z_")) {
    ///                     if y.to_atom() > z.to_atom() {
    ///                         return ConditionResult::False;
    ///                     }
    ///                 }
    ///                 return ConditionResult::True;
    ///             }
    ///         }
    ///         ConditionResult::Inconclusive
    ///     }))
    ///     .with(parse!("1"));
    /// assert_eq!(out, parse!("1"));
    /// ```
    fn replace<'b, P: Into<BorrowedOrOwned<'b, Pattern>>>(
        &self,
        pattern: P,
    ) -> ReplaceBuilder<'_, 'b> {
        self.as_atom_view().replace(pattern)
    }

    /// Replace all occurrences of the patterns, where replacements are tested in the order that they are given.
    /// To repeatedly replace multiple patterns, wrap the call in [Atom::replace_map].
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// use symbolica::id::{Pattern, Replacement};
    /// let expr = parse!("x + y");
    /// let pattern1 = parse!("x").to_pattern();
    /// let replacement1 = parse!("y").to_pattern();
    /// let pattern2 = parse!("y").to_pattern();
    /// let replacement2 = parse!("x").to_pattern();
    /// let result = expr.replace_multiple(&[
    ///     Replacement::new(pattern1, replacement1),
    ///     Replacement::new(pattern2, replacement2),
    /// ]);
    /// assert_eq!(result, parse!("x + y"));
    /// ```
    fn replace_multiple<T: BorrowReplacement>(&self, replacements: &[T]) -> Atom {
        self.as_atom_view().replace_multiple(replacements)
    }

    /// Replace all occurrences of the patterns, where replacements are tested in the order that they are given.
    /// Returns `true` iff a match was found.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse};
    /// use symbolica::id::{Pattern, Replacement};
    /// let expr = parse!("x + y");
    /// let pattern1 = parse!("x").to_pattern();
    /// let replacement1 = parse!("y").to_pattern();
    /// let pattern2 = parse!("y").to_pattern();
    /// let replacement2 = parse!("x").to_pattern();
    /// let mut out = Atom::new();
    /// let replacements = [
    ///     Replacement::new(pattern1, replacement1),
    ///     Replacement::new(pattern2, replacement2),
    /// ];
    /// let changed = expr.replace_multiple_into(&replacements, &mut out);
    /// assert!(changed);
    /// assert_eq!(out, parse!("x + y"));
    /// ```
    fn replace_multiple_into<T: BorrowReplacement>(
        &self,
        replacements: &[T],
        out: &mut Atom,
    ) -> bool {
        self.as_atom_view().replace_multiple_into(replacements, out)
    }

    /// Replace part of an expression by calling the map `m` on each subexpression.
    /// The function `m`  must return `true` if the expression was replaced and must write the new expression to `out`.
    /// A [Context] object is passed to the function, which contains information about the current position in the expression.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, symbol};
    /// let (x, y, z) = symbol!("x", "y", "z");
    /// let expr = Atom::var(x) + y;
    /// let result = expr.replace_map(|term, _ctx, out| {
    ///     if term.get_symbol() == Some(x) {
    ///         **out = Atom::from(z);
    ///     }
    /// });
    /// assert_eq!(result, Atom::var(y) + z);
    /// ```
    fn replace_map<F: FnMut(AtomView, &Context, &mut Settable<'_, Atom>)>(&self, m: F) -> Atom {
        self.as_atom_view().replace_map(m)
    }

    /// Call the function `v` for every subexpression. If `v` returns `true`, the
    /// subexpressions of the current expression will be visited.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{AtomCore, AtomView}, parse};
    /// let mut has_complex_coefficient = false;
    /// let expr = parse!("3*f(x,f(4+2𝑖))");
    /// let result = expr.visitor(&mut |a| {
    ///     if let AtomView::Num(n) = a {
    ///         if !n.get_coeff_view().is_real() {
    ///             has_complex_coefficient = true;
    ///         }
    ///     }
    ///     !has_complex_coefficient // early abort when found
    /// });
    /// assert!(has_complex_coefficient);
    /// ```
    fn visitor<F: FnMut(AtomView) -> bool>(&self, v: &mut F) {
        self.as_atom_view().visitor(v)
    }

    /// Return an iterator over matched expressions.
    /// Alternatively, use [ReplaceBuilder::match_iter].
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
    /// use symbolica::id::Pattern;
    /// let expr = parse!("f(1) + f(2)");
    /// let pattern = parse!("f(x_)").to_pattern();
    /// let mut iter = expr.pattern_match(&pattern, None, None);
    /// let result = iter.next().unwrap();
    /// assert_eq!(
    ///     result.get(&symbol!("x_")).unwrap(),
    ///     &Atom::num(1)
    /// );
    /// ```
    fn pattern_match<
        'a: 'b,
        'b,
        C: Into<Option<&'b Condition<PatternRestriction>>>,
        S: Into<Option<&'b MatchSettings>>,
    >(
        &'a self,
        pattern: &'b Pattern,
        conditions: C,
        settings: S,
    ) -> PatternAtomTreeIterator<'a, 'b> {
        PatternAtomTreeIterator::new(
            pattern,
            self.as_atom_view(),
            conditions.into(),
            settings.into(),
        )
    }

    /// Return an iterator over all terms in the expression.
    ///
    /// # Example
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("x + y + z");
    /// let mut iter = expr.terms();
    /// assert_eq!(iter.next().unwrap(), parse!("x").as_view());
    /// assert_eq!(iter.next().unwrap(), parse!("y").as_view());
    /// assert_eq!(iter.next().unwrap(), parse!("z").as_view());
    /// assert_eq!(iter.next(), None);
    /// ```
    fn terms(&self) -> impl Iterator<Item = AtomView<'_>> {
        let s = self.as_atom_view();
        match self.as_atom_view() {
            AtomView::Add(a) => a.to_slice().iter(),
            _ => ListSlice::from_one(s).iter(),
        }
    }

    /// Return an iterator over the children of the atom.
    ///
    /// # Example
    /// ```
    /// use symbolica::{atom::AtomCore, parse};
    /// let expr = parse!("f(x,y)");
    /// let mut iter = expr.children();
    /// assert_eq!(iter.next().unwrap(), parse!("x").as_view());
    /// assert_eq!(iter.next().unwrap(), parse!("y").as_view());
    /// assert_eq!(iter.next(), None);
    /// ```
    /// return `x, y`
    fn children(&self) -> impl Iterator<Item = AtomView<'_>> {
        match self.as_atom_view() {
            AtomView::Add(a) => a.to_slice().iter(),
            AtomView::Mul(a) => a.to_slice().iter(),
            AtomView::Pow(a) => a.to_slice().iter(),
            AtomView::Fun(a) => a.to_slice().iter(),
            AtomView::Num(_) | AtomView::Var(_) => ListSlice::empty().iter(),
        }
    }
}

impl AtomCore for InlineVar {
    fn as_atom_view(&self) -> AtomView<'_> {
        self.as_view()
    }
}

impl AtomCore for InlineNum {
    fn as_atom_view(&self) -> AtomView<'_> {
        self.as_view()
    }
}

impl AtomCore for Indeterminate {
    fn as_atom_view(&self) -> AtomView<'_> {
        self.as_view()
    }
}

impl<'a> AtomCore for AtomView<'a> {
    fn as_atom_view(&self) -> AtomView<'a> {
        *self
    }
}

impl<T: AsRef<Atom>> AtomCore for T {
    fn as_atom_view(&self) -> AtomView<'_> {
        self.as_ref().as_view()
    }
}

impl AtomCore for AtomOrView<'_> {
    fn as_atom_view(&self) -> AtomView<'_> {
        self.as_view()
    }
}

mod private {
    use crate::atom::{AtomView, Indeterminate, InlineNum, InlineVar};

    pub trait Sealed {}

    impl Sealed for InlineVar {}
    impl Sealed for InlineNum {}
    impl Sealed for Indeterminate {}
    impl<'a> Sealed for AtomView<'a> {}
    impl<T: AsRef<super::Atom>> Sealed for T {}
    impl Sealed for super::AtomOrView<'_> {}
}
