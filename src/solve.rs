//! Solve systems of equations.
//!
//! See [AtomCore::solve_linear_system] and [AtomCore::nsolve_system].

use std::{ops::Neg, sync::Arc};

use ahash::HashSet;
use numerica::domains::{Field, float::Complex, rational::Rational};

use crate::{
    atom::{Atom, AtomCore, AtomView, Indeterminate},
    coefficient::{Coefficient, ConvertToRing},
    domains::{
        InternalOrdering, SelfRing,
        float::{FloatField, Real, SingleFloat},
        integer::Z,
        rational::Q,
        rational_polynomial::{RationalPolynomial, RationalPolynomialField},
    },
    evaluate::{FunctionMap, OptimizationSettings},
    poly::{PolyVariable, PositiveExponent},
    tensors::matrix::{Matrix, MatrixError},
};

/// Errors that can occur when solving a system.
/// Underdetermined systems return a partial solution.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SolveError {
    /// The system was underdetermined. The partial solution is returned.
    Underdetermined {
        /// Rank of the system.
        rank: u32,
        /// Partial solution found, that may contain free variables.
        partial_solution: Vec<Atom>,
    },
    Other(String),
}

impl std::error::Error for SolveError {}

impl std::fmt::Display for SolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolveError::Underdetermined {
                rank,
                partial_solution,
            } => write!(
                f,
                "Underdetermined system of rank {}/{}. Partial solution: {:?}",
                rank,
                partial_solution.len(),
                partial_solution
            ),
            SolveError::Other(e) => f.write_str(e),
        }
    }
}

impl AtomView<'_> {
    /// Find the root of a function in `x` numerically over the reals using Newton's method.
    pub(crate) fn nsolve<N: SingleFloat + Real + PartialOrd>(
        &self,
        x: &Indeterminate,
        init: N,
        prec: N,
        max_iterations: usize,
    ) -> Result<N, String> {
        if self.has_complex_coefficients() {
            return Err("Complex coefficients are not supported".to_owned());
        }

        let v = x.clone().into();
        let f = self
            .to_evaluation_tree(&FunctionMap::new(), std::slice::from_ref(&v))?
            .optimize(&OptimizationSettings {
                horner_iterations: 0,
                n_cores: 0,
                cpe_iterations: None,
                hot_start: None,
                abort_check: None,
                verbose: false,
            });
        let df = self
            .derivative(x)
            .to_evaluation_tree(&FunctionMap::new(), std::slice::from_ref(&v))?
            .optimize(&OptimizationSettings {
                horner_iterations: 0,
                n_cores: 0,
                cpe_iterations: None,
                hot_start: None,
                abort_check: None,
                verbose: false,
            });

        let mut f_e = f.map_coeff(&|x| init.from_rational(x.to_real().unwrap()));
        let mut df_e = df.map_coeff(&|x| init.from_rational(x.to_real().unwrap()));

        let mut cur = init.clone();

        for _ in 0..max_iterations {
            let df_val = df_e.evaluate_single(std::slice::from_ref(&cur));
            let f_val = f_e.evaluate_single(std::slice::from_ref(&cur));

            if !df_val.is_finite() || df_val.is_zero() {
                return Err("Derivative is zero".to_owned());
            }

            cur -= f_val.clone() / df_val;
            if f_val.norm() < prec {
                return Ok(cur);
            }
        }

        Err("Did not converge".to_owned())
    }

    /// Solve a non-linear system numerically over the reals using Newton's method.
    pub(crate) fn nsolve_system<
        N: SelfRing + SingleFloat + Real + PartialOrd + InternalOrdering + Eq + std::hash::Hash,
        T: AtomCore,
    >(
        system: &[T],
        vars: &[Indeterminate],
        init: &[N],
        prec: N,
        max_iterations: usize,
    ) -> Result<Vec<N>, String> {
        let system = system.iter().map(|v| v.as_atom_view()).collect::<Vec<_>>();
        AtomView::nsolve_system_impl(&system, vars, init, prec, max_iterations)
    }

    fn nsolve_system_impl<
        N: SelfRing + SingleFloat + Real + PartialOrd + InternalOrdering + Eq + std::hash::Hash,
    >(
        system: &[AtomView],
        vars: &[Indeterminate],
        init: &[N],
        prec: N,
        max_iterations: usize,
    ) -> Result<Vec<N>, String> {
        if system.len() != vars.len() {
            Err("System must have same number of equations as there are unknowns".to_owned())?;
        }

        if vars.len() != init.len() {
            Err("Initial values must be provided for all unknowns".to_owned())?;
        }

        if system.is_empty() {
            return Ok(vec![]);
        }

        if system.iter().any(|a| a.has_complex_coefficients()) {
            return Err("Complex coefficients are not supported".to_owned());
        }

        if system.len() == 1 {
            return Ok(vec![system[0].nsolve(
                &vars[0],
                init[0].clone(),
                prec,
                max_iterations,
            )?]);
        }

        let avars = vars.iter().map(|v| v.clone().into()).collect::<Vec<_>>();

        let mut fs = system
            .iter()
            .map(|a| {
                Ok(a.to_evaluation_tree(&FunctionMap::new(), &avars)?
                    .optimize(&OptimizationSettings {
                        horner_iterations: 0,
                        n_cores: 0,
                        cpe_iterations: None,
                        hot_start: None,
                        abort_check: None,
                        verbose: false,
                    })
                    .map_coeff(&|x| init[0].from_rational(x.to_real().unwrap())))
            })
            .collect::<Result<Vec<_>, String>>()?;

        let mut jacobian = Vec::with_capacity(vars.len() * system.len());
        for a in system {
            let mut row = Vec::with_capacity(vars.len());
            for v in vars {
                let deriv = a.derivative(v);

                let a = deriv
                    .to_evaluation_tree(&FunctionMap::new(), &avars)?
                    .optimize(&OptimizationSettings {
                        horner_iterations: 0,
                        n_cores: 0,
                        cpe_iterations: None,
                        hot_start: None,
                        abort_check: None,
                        verbose: false,
                    })
                    .map_coeff(&|x| init[0].from_rational(x.to_real().unwrap()));

                row.push(a);
            }
            jacobian.extend_from_slice(&row);
        }

        let field = FloatField::from_rep(init[0].clone());
        let mut cur = init.to_vec();

        for _ in 0..max_iterations {
            let f = fs
                .iter_mut()
                .map(|a| a.evaluate_single(&cur))
                .collect::<Vec<_>>();
            let f = Matrix::new_vec(f, field.clone());

            let df = jacobian
                .iter_mut()
                .map(|a| a.evaluate_single(&cur))
                .collect::<Vec<_>>();

            let df = Matrix::from_linear(df, system.len() as u32, vars.len() as u32, field.clone())
                .unwrap();

            let Ok(i) = df.inv() else {
                return Err("Could not invert Jacobian".to_owned());
            };

            let mut ci = Matrix::new_vec(cur.to_vec(), field.clone());

            ci -= &(&i * &f);

            cur = ci.into_vec();

            if f.into_iter().all(|x| x.norm() < prec) {
                return Ok(cur);
            }
        }

        Err("Did not converge".to_owned())
    }

    /// Solve a system that is linear in `vars`, if possible.
    /// Each expression in `system` is understood to yield 0.
    pub(crate) fn solve_linear_system<E: PositiveExponent, T1: AtomCore, T2: AtomCore>(
        system: &[T1],
        vars: &[T2],
    ) -> Result<Vec<Atom>, SolveError> {
        let system: Vec<_> = system.iter().map(|v| v.as_atom_view()).collect();

        let vars: Vec<_> = vars
            .iter()
            .map(|v| v.as_atom_view().to_owned().try_into())
            .collect::<Result<Vec<_>, _>>()
            .map_err(SolveError::Other)?;

        AtomView::solve_linear_system_impl::<E>(&system, &vars)
    }

    /// Convert a system of linear equations to a matrix representation, returning the matrix
    /// and the right-hand side.
    pub(crate) fn system_to_matrix<E: PositiveExponent, T1: AtomCore, T2: AtomCore>(
        system: &[T1],
        vars: &[T2],
    ) -> Result<
        (
            Matrix<RationalPolynomialField<Z, E>>,
            Matrix<RationalPolynomialField<Z, E>>,
        ),
        String,
    > {
        let system: Vec<_> = system.iter().map(|v| v.as_atom_view()).collect();

        let vars: Vec<_> = vars
            .iter()
            .map(|v| v.as_atom_view().to_owned().try_into())
            .collect::<Result<Vec<_>, _>>()?;
        let params = Self::get_parameters(&system, &vars);

        AtomView::system_to_matrix_impl::<E>(&system, &vars, params)
    }

    fn system_to_matrix_impl<E: PositiveExponent>(
        system: &[AtomView],
        vars: &[PolyVariable],
        params: HashSet<AtomView>,
    ) -> Result<
        (
            Matrix<RationalPolynomialField<Z, E>>,
            Matrix<RationalPolynomialField<Z, E>>,
        ),
        String,
    > {
        let mut mat = Vec::with_capacity(system.len() * vars.len());
        let mut row = vec![RationalPolynomial::<_, E>::new(&Z, Arc::new(vec![])); vars.len()];
        let mut rhs = vec![RationalPolynomial::<_, E>::new(&Z, Arc::new(vec![])); system.len()];

        let params = Arc::new(
            params
                .iter()
                .map(|x| Ok(x.to_owned().try_into()?))
                .collect::<Result<Vec<_>, String>>()?,
        );

        for (si, a) in system.iter().enumerate() {
            let rat: RationalPolynomial<Z, E> = a.try_to_rational_polynomial(&Q, &Z, None)?;

            let poly = rat.to_polynomial(vars, true).map_err(|e| e.to_owned())?;

            for e in &mut row {
                *e = RationalPolynomial::<_, E>::new(&Z, params.clone());
            }

            // get linear coefficients
            'next_monomial: for e in poly.into_iter() {
                if e.exponents.iter().cloned().sum::<E>() > E::one() {
                    Err("Not a linear system")?;
                }

                for (rv, p) in row.iter_mut().zip(e.exponents) {
                    if !p.is_zero() {
                        *rv = e.coefficient.clone();
                        continue 'next_monomial;
                    }
                }

                // constant term
                rhs[si] = e.coefficient.clone().neg();
            }

            mat.extend_from_slice(&row);
        }

        let Some((first, rest)) = mat.split_first_mut() else {
            return Err("Empty system".to_owned());
        };

        for _ in 0..2 {
            for x in &mut *rest {
                first.unify_variables(x);
            }
            for x in &mut rhs {
                first.unify_variables(x);
            }
        }

        let field = RationalPolynomialField::new(Z);

        let m = Matrix::from_linear(mat, system.len() as u32, vars.len() as u32, field.clone())
            .unwrap();
        let b = Matrix::new_vec(rhs, field);

        Ok((m, b))
    }

    /// Get all parameters in the system that are not free variables.
    fn get_parameters<'a>(system: &[AtomView<'a>], vars: &[PolyVariable]) -> HashSet<AtomView<'a>> {
        let mut all_params = HashSet::default();
        for s in system {
            all_params.extend(s.get_all_indeterminates(false));
        }

        let v: Vec<_> = vars.iter().map(|x| x.to_atom()).collect();
        let mut all_vars = HashSet::default();
        for x in &v {
            all_vars.insert(x.as_view());
        }

        all_params
            .into_iter()
            .filter(|x| !all_vars.contains(x))
            .collect()
    }

    fn solve_linear_system_without_parameters<T: Field + ConvertToRing>(
        system: &[AtomView],
        vars: &[PolyVariable],
        field: T,
    ) -> Result<Vec<Atom>, SolveError>
    where
        T::Element: Into<Coefficient>,
    {
        let mut mat = vec![field.zero(); system.len() * vars.len()];
        let mut rhs = vec![field.zero(); system.len()];

        let vars = Arc::new(vars.to_vec());
        for (row, s) in system.iter().enumerate() {
            let poly = s
                .try_to_polynomial::<_, u8>(&field, Some(vars.clone()))
                .map_err(|e| SolveError::Other(e))?;

            for e in &poly {
                let mut found = false;
                for j in 0..vars.len() {
                    if e.exponents[j] != 0 {
                        if found {
                            return Err(SolveError::Other("Not a linear system".to_owned()));
                        }
                        mat[row * vars.len() + j] = e.coefficient.clone();
                        found = true;
                    }
                }

                if !found {
                    rhs[row] = field.neg(e.coefficient);
                }
            }
        }

        let m = Matrix::from_linear(mat, system.len() as u32, vars.len() as u32, field.clone())
            .map_err(|e| SolveError::Other(e))?;
        let rhs = Matrix::new_vec(rhs, field.clone());

        match m.solve(&rhs) {
            Ok(sol) => Ok(sol.into_vec().into_iter().map(|s| Atom::num(s)).collect()),
            Err(MatrixError::Underdetermined {
                rank,
                row_reduced_augmented_matrix,
            }) => {
                let mut sols = Vec::with_capacity(vars.len());

                let mut var_index = 0;
                for r in row_reduced_augmented_matrix.row_iter() {
                    while var_index < vars.len() as u32 && field.is_zero(&r[var_index as usize]) {
                        sols.push(vars[var_index as usize].to_atom());
                        var_index += 1;
                    }

                    if var_index >= vars.len() as u32 {
                        break;
                    }

                    if field.is_one(&r[var_index as usize]) {
                        let mut sol = Atom::num(r.last().unwrap().clone());

                        for (var, coeff) in vars.iter().zip(r).skip((var_index + 1) as usize) {
                            if !field.is_zero(coeff) {
                                sol -= Atom::num(coeff.clone()) * var.to_atom();
                            }
                        }

                        sols.push(sol);
                        var_index += 1;
                    }
                }

                for i in var_index as usize..vars.len() {
                    sols.push(vars[i].to_atom());
                }

                Err(SolveError::Underdetermined {
                    rank,
                    partial_solution: sols,
                })
            }
            Err(e) => Err(SolveError::Other(format!("Could not solve {e:?}"))),
        }
    }

    fn solve_linear_system_impl<E: PositiveExponent>(
        system: &[AtomView],
        vars: &[PolyVariable],
    ) -> Result<Vec<Atom>, SolveError> {
        let params = Self::get_parameters(system, vars);
        if params.is_empty() {
            if system.iter().any(|a| a.has_complex_coefficients()) {
                let f: FloatField<Complex<Rational>> = FloatField::from_rep(Complex::new_zero());
                return Self::solve_linear_system_without_parameters(system, vars, f);
            } else {
                return Self::solve_linear_system_without_parameters::<Q>(system, vars, Q);
            }
        }

        let (m, b) = Self::system_to_matrix_impl::<E>(system, vars, params)
            .map_err(|e| SolveError::Other(e.to_string()))?;

        match m.solve(&b) {
            Ok(sol) => Ok(sol
                .into_vec()
                .into_iter()
                .map(|s| s.to_expression())
                .collect()),
            Err(MatrixError::Underdetermined {
                rank,
                row_reduced_augmented_matrix,
            }) => {
                let mut sols = Vec::with_capacity(vars.len());

                let mut var_index = 0;
                for r in row_reduced_augmented_matrix.row_iter() {
                    while var_index < vars.len() as u32 && r[var_index as usize].is_zero() {
                        sols.push(vars[var_index as usize].to_atom());
                        var_index += 1;
                    }

                    if var_index >= vars.len() as u32 {
                        break;
                    }

                    if r[var_index as usize].is_one() {
                        let mut sol = r.last().unwrap().to_expression();

                        for (var, coeff) in vars.iter().zip(r).skip((var_index + 1) as usize) {
                            if !coeff.is_zero() {
                                sol -= coeff.to_expression() * var.to_atom();
                            }
                        }

                        sols.push(sol);
                        var_index += 1;
                    }
                }

                for i in var_index as usize..vars.len() {
                    sols.push(vars[i].to_atom());
                }

                Err(SolveError::Underdetermined {
                    rank,
                    partial_solution: sols,
                })
            }
            Err(e) => Err(SolveError::Other(format!("Could not solve {e:?}"))),
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use crate::{
        atom::{AtomCore, AtomView, representation::InlineVar},
        domains::{
            float::{F64, Real},
            integer::Z,
            rational::Q,
            rational_polynomial::{RationalPolynomial, RationalPolynomialField},
        },
        parse,
        poly::PolyVariable,
        solve::SolveError,
        symbol,
        tensors::matrix::Matrix,
    };

    #[test]
    fn underdetermined() {
        let v0 = symbol!("v0").into();
        let v1 = symbol!("v1").into();
        let v2 = symbol!("v2").into();
        let v3 = symbol!("v3").into();
        let v4 = symbol!("v4").into();
        let eqs = ["v1 + v2 - 3", "2*v1 + 2*v2 - 6", "v1 + v3 - 5"];

        let system: Vec<_> = eqs.iter().map(|e| parse!(e)).collect();
        let vars = [v0, v1, v2, v3, v4];

        let sol = AtomView::solve_linear_system::<u8, _, InlineVar>(&system, &vars);

        assert_eq!(
            sol,
            Err(SolveError::Underdetermined {
                rank: 2,
                partial_solution: vec![
                    parse!("v0"),
                    parse!("-v3+5"),
                    parse!("v3-2"),
                    parse!("v3"),
                    parse!("v4"),
                ],
            })
        );
    }

    #[test]
    fn solve() {
        let x = symbol!("v1").into();
        let y = symbol!("v2").into();
        let z = symbol!("v3").into();
        let eqs = [
            "v4*v1 + f1(v4)*v2 + v3 - 1",
            "v1 + v4*v2 + v3/v4 - 2",
            "(v4-1)v1 + v4*v3",
        ];

        let system: Vec<_> = eqs.iter().map(|e| parse!(e)).collect();

        let sol = AtomView::solve_linear_system::<u8, _, InlineVar>(&system, &[x, y, z]).unwrap();

        let res = [
            "(v4^3-2*v4^2*f1(v4))*(v4^2-v4^3+v4^4-f1(v4)+v4*f1(v4)-v4^2*f1(v4))^-1",
            "(v4^2-f1(v4))^-1*(2*v4-1)",
            "(v4^2-v4^3-2*v4*f1(v4)+2*v4^2*f1(v4))*(v4^2-v4^3+v4^4-f1(v4)+v4*f1(v4)-v4^2*f1(v4))^-1",
        ];
        let res = res.iter().map(|x| parse!(x)).collect::<Vec<_>>();

        assert_eq!(sol, res);
    }

    #[test]
    fn solve_from_matrix() {
        let system = [
            ["v4", "v4+1", "v4^2+5"],
            ["1", "v4", "v4+1"],
            ["v4-1", "-1", "v4"],
        ];
        let rhs = ["1", "2", "-1"];

        let var_map = Arc::new(vec![PolyVariable::Symbol(symbol!("v4"))]);

        let system_rat: Vec<RationalPolynomial<_, u8>> = system
            .iter()
            .flatten()
            .map(|s| parse!(s).to_rational_polynomial(&Q, &Z, Some(var_map.clone())))
            .collect();

        let rhs_rat: Vec<RationalPolynomial<_, u8>> = rhs
            .iter()
            .map(|s| parse!(s).to_rational_polynomial(&Q, &Z, Some(var_map.clone())))
            .collect();

        let field = RationalPolynomialField::from_poly(&rhs_rat[0].numerator);
        let m = Matrix::from_linear(
            system_rat,
            system.len() as u32,
            system.len() as u32,
            field.clone(),
        )
        .unwrap();
        let b = Matrix::new_vec(rhs_rat, field);

        let sol = m.solve(&b).unwrap();

        let res = [
            "(10-2*v4+4*v4^2-v4^3)/(6-4*v4+5*v4^2-3*v4^3+v4^4)",
            "(-4+10*v4-5*v4^2+2*v4^3)/(6-4*v4+5*v4^2-3*v4^3+v4^4)",
            "(2-4*v4)/(6-4*v4+5*v4^2-3*v4^3+v4^4)",
        ];

        let res = res
            .iter()
            .map(|x| parse!(x).to_rational_polynomial(&Z, &Z, m[(0, 0)].get_variables().clone()))
            .collect::<Vec<_>>();

        assert_eq!(sol.into_vec(), res);
    }

    #[test]
    fn find_root() {
        let x = symbol!("x");
        let a = parse!("x^2 - 2");
        let a = a.as_view();

        let root = a.nsolve(&x.into(), 1.0, 1e-10, 1000).unwrap();
        assert!((root - 2f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn solve_system_newton() {
        let a = parse!("5x^2+x*y^2+sin(2y)^2 - 2");
        let b = parse!("exp(2x-y)+4y - 3");

        let r = AtomView::nsolve_system(
            &[a.as_view(), b.as_view()],
            &[symbol!("x").into(), symbol!("y").into()],
            &[F64::from(1.), F64::from(1.)],
            F64::from(1e-10),
            100,
        )
        .unwrap();

        assert!((r[0] - F64::from(5.672_973_499_396_123e-1)).norm() < 1e-10.into());
        assert!((r[1] - F64::from(-3.0944227920271083e-1)).norm() < 1e-10.into());
    }
}
