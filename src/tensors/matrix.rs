//! Linear algebra methods using matrices and vectors.
//!
//! # Examples
//!
//! Solve an underdetermined linear system:
//!
//! ```
//! use numerica::domains::rational::Q;
//! use numerica::tensors::matrix::Matrix;
//! let m = vec![
//!     vec![1.into(), 1.into(), 1.into()],
//!     vec![1.into(), 1.into(), 2.into()],
//! ];
//! let rhs = Matrix::new_vec(vec![3.into(), 2.into()], Q);
//!
//! let mat = Matrix::from_nested_vec(m, Q).unwrap();
//! let r = mat.solve_any(&rhs).unwrap();
//! assert_eq!(r.into_vec(), [4, 0, -1]);
//! ```

use std::{
    fmt::Display,
    ops::{Add, AddAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign},
    slice::Chunks,
};

use colored::{Color, Colorize};

use crate::{
    domains::{
        Derivable, EuclideanDomain, Field, InternalOrdering, Ring, SelfRing,
        integer::Z,
        rational::{Q, Rational},
    },
    printer::{PrintOptions, PrintState},
    tensors::sparse::SparseMatrix,
};

/// An n-dimensional vector.
///
/// # Examples
///
/// ```
/// use numerica::domains::rational::Q;
/// use numerica::tensors::matrix::Vector;
/// let v1 = Vector::new(vec![(3,1).into(), (1,1).into()], Q);
/// let v2 = Vector::new(vec![(2,1).into(), (2,1).into()], Q);
/// let b = Vector::orthogonalize(&[v1, v2]);
///
/// let r = Vector::new(vec![(-2,5).into(), (6,5).into()], Q);
/// assert_eq!(b[1], r);
/// ```
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Vector<F: Ring> {
    pub(crate) data: Vec<F::Element>,
    pub(crate) field: F,
}

impl<F: Ring> Vector<F> {
    /// Create a new vector from a list of scalars.
    pub fn new(data: Vec<F::Element>, field: F) -> Vector<F> {
        Vector { data, field }
    }

    /// Create a new zero vector from an existing one.
    pub fn new_zero(&self) -> Vector<F> {
        Vector {
            data: vec![self.field.zero(); self.data.len()],
            field: self.field.clone(),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Create a column vector. This operation is very cheap.
    pub fn into_matrix(self) -> Matrix<F> {
        Matrix {
            nrows: 1,
            ncols: self.data.len() as u32,
            data: self.data,
            field: self.field,
        }
    }

    /// Yield the vector as a list of scalars.
    pub fn into_vec(self) -> Vec<F::Element> {
        self.data
    }

    /// Apply a function `f` to each entry of the vector.
    pub fn map<G: Ring>(&self, f: impl Fn(&F::Element) -> G::Element, field: G) -> Vector<G> {
        Vector {
            data: self.data.iter().map(f).collect(),
            field,
        }
    }

    pub fn norm_squared(&self) -> F::Element {
        let mut res = self.field.zero();
        for e in &self.data {
            self.field.add_mul_assign(&mut res, e, e);
        }
        res
    }

    /// Take the Euclidean scalar product of two column or row vectors.
    pub fn dot(&self, rhs: &Self) -> F::Element {
        if self.data.len() != rhs.data.len() {
            panic!(
                "Vectors do not have equal dimension: {} vs {}",
                self.data.len(),
                rhs.data.len()
            );
        }

        let mut res = self.field.zero();
        for (e1, e2) in self.data.iter().zip(&rhs.data) {
            self.field.add_mul_assign(&mut res, e1, e2);
        }

        res
    }

    /// Compute the Euclidean cross product in three dimensions.
    pub fn cross_product(&self, rhs: &Self) -> Vector<F> {
        if self.data.len() != rhs.data.len() {
            panic!(
                "Vectors do not have equal dimension: {} vs {}",
                self.data.len(),
                rhs.data.len()
            );
        }
        if self.data.len() != 3 {
            panic!(
                "Vectors must be three-dimensional instead of {}",
                self.data.len(),
            );
        }

        Vector {
            data: vec![
                self.field.sub(
                    &self.field.mul(&self.data[1], &rhs.data[2]),
                    &self.field.mul(&self.data[2], &rhs.data[1]),
                ),
                self.field.sub(
                    &self.field.mul(&self.data[2], &rhs.data[0]),
                    &self.field.mul(&self.data[0], &rhs.data[2]),
                ),
                self.field.sub(
                    &self.field.mul(&self.data[0], &rhs.data[1]),
                    &self.field.mul(&self.data[1], &rhs.data[0]),
                ),
            ],
            field: self.field.clone(),
        }
    }
}

impl<F: Ring> SelfRing for Vector<F> {
    fn is_one(&self) -> bool {
        self.data.iter().all(|e| self.field.is_one(e))
    }

    fn is_zero(&self) -> bool {
        self.data.iter().all(|x| self.field.is_zero(x))
    }

    fn format<W: std::fmt::Write>(
        &self,
        opts: &PrintOptions,
        mut state: PrintState,
        f: &mut W,
    ) -> Result<bool, std::fmt::Error> {
        if state.in_sum {
            f.write_char('+')?;
        }
        state.in_sum = false;
        state.in_product = false;
        state.in_exp = false;
        state.in_exp_base = false;

        if opts.mode.is_latex() {
            f.write_str("\\begin{pvector}")?;

            for (ri, r) in self.data.iter().enumerate() {
                self.field.format(r, opts, PrintState::new(), f)?;

                if ri + 1 < self.data.len() {
                    f.write_str(" & ")?;
                }
            }

            f.write_str("\\end{pvector}")?;
            Ok(false)
        } else {
            f.write_char('{')?;
            for (ri, r) in self.data.iter().enumerate() {
                self.field.format(r, opts, PrintState::new(), f)?;

                if ri + 1 < self.data.len() {
                    f.write_char(',')?;
                }
            }
            f.write_char('}')?;
            Ok(false)
        }
    }
}

impl<F: Ring> InternalOrdering for Vector<F> {
    fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.data.internal_cmp(&other.data)
    }
}

impl<F: Derivable> Vector<F> {
    /// Compute the derivative in the variable `x`.
    pub fn derivative(&self, x: &F::Variable) -> Vector<F> {
        self.map(|e| self.field.derivative(e, x), self.field.clone())
    }

    /// Compute the gradient in the variables `x`.
    pub fn grad(&self, x: &[F::Variable]) -> Vector<F> {
        if self.len() != x.len() {
            panic!(
                "The number of variables ({}) does not match the number of entries ({})",
                x.len(),
                self.len()
            );
        }

        Vector::new(
            self.data
                .iter()
                .zip(x)
                .map(|(e, v)| self.field.derivative(e, v))
                .collect(),
            self.field.clone(),
        )
    }

    /// Compute the curl of the vector in the variables `x`.
    pub fn curl(&self, x: &[F::Variable]) -> Vector<F> {
        if self.len() != 3 || x.len() != 3 {
            panic!("Vector and variable list must be three-dimensional");
        }

        Vector {
            data: vec![
                self.field.sub(
                    &self.field.derivative(&self.data[2], &x[1]),
                    &self.field.derivative(&self.data[1], &x[2]),
                ),
                self.field.sub(
                    &self.field.derivative(&self.data[0], &x[2]),
                    &self.field.derivative(&self.data[2], &x[0]),
                ),
                self.field.sub(
                    &self.field.derivative(&self.data[1], &x[0]),
                    &self.field.derivative(&self.data[0], &x[1]),
                ),
            ],
            field: self.field.clone(),
        }
    }

    /// Compute the Jacobian matrix of the vector of functions in the variables `x`.
    pub fn jacobian(&self, x: &[F::Variable]) -> Matrix<F> {
        let mut jacobian = Vec::with_capacity(self.data.len());

        for e in &self.data {
            for v in x {
                jacobian.push(self.field.derivative(e, v));
            }
        }

        Matrix {
            data: jacobian,
            ncols: x.len() as u32,
            nrows: self.len() as u32,
            field: self.field.clone(),
        }
    }
}

impl<F: Derivable> Matrix<F> {
    /// Compute the derivative in the variable `x`.
    pub fn derivative(&self, x: &F::Variable) -> Matrix<F> {
        self.map(|e| self.field.derivative(e, x), self.field.clone())
    }
}

impl<F: Ring> Mul<F::Element> for Vector<F> {
    type Output = Vector<F>;

    fn mul(mut self, rhs: F::Element) -> Self::Output {
        for x in &mut self.data {
            self.field.mul_assign(x, &rhs);
        }
        self
    }
}

impl<F: Ring> Mul<F::Element> for &Vector<F> {
    type Output = Vector<F>;

    fn mul(self, rhs: F::Element) -> Self::Output {
        self.clone() * rhs
    }
}

impl<F: EuclideanDomain> Matrix<F> {
    /// Write the first `max_col` columns of the matrix in (non-reduced) echelon form.
    /// Returns the matrix rank.
    pub fn partial_row_reduce_fraction_free(&mut self, max_col: u32) -> u32 {
        let mut i = 0;
        for j in 0..max_col.min(self.ncols) {
            if self.field.is_zero(&self[(i, j)]) {
                // Select a non-zero pivot.
                for k in i + 1..self.nrows {
                    if !self.field.is_zero(&self[(k, j)]) {
                        // Swap i-th row and k-th row.
                        self.swap_rows(i, k, j);
                        break;
                    }
                }

                // zero column found
                if self.field.is_zero(&self[(i, j)]) {
                    continue;
                }
            }

            // strip content from pivot row to prevent number growth
            let mut g = self[(i, j)].clone();
            for l in j + 1..self.ncols {
                if F::one_is_gcd_unit() && self.field.is_one(&g) {
                    break;
                }
                g = self.field.gcd(&g, &self[(i, l)]);
            }
            if !self.field.is_one(&g) {
                for l in j..self.ncols {
                    self[(i, l)] = self.field.try_div(&self[(i, l)], &g).unwrap();
                }
            }

            let x = self[(i, j)].clone();
            for k in i + 1..self.nrows {
                if !self.field.is_zero(&self[(k, j)]) {
                    let g = self.field.gcd(&x, &self[(k, j)]);
                    let scale_pivot = self.field.try_div(&self[(k, j)], &g).unwrap();
                    let scale_row = self.field.try_div(&x, &g).unwrap();

                    self[(k, j)] = self.field.zero();
                    for l in j + 1..self.ncols {
                        self[(k, l)] = self.field.sub(
                            &self.field.mul(&self[(k, l)], &scale_row),
                            &self.field.mul(&self[(i, l)], &scale_pivot),
                        );
                    }
                }
            }

            i += 1;
            if i >= self.nrows {
                break;
            }
        }

        i
    }

    /// Create a row-reduced (but not necessarily normalized) matrix from a matrix in echelon form.
    pub fn back_substitution_fraction_free(&mut self, mut max_col: u32) {
        max_col = max_col.min(self.ncols);
        for i in (0..self.nrows).rev() {
            if let Some(j) = (0..max_col).find(|&j| !self.field.is_zero(&self[(i, j)])) {
                // strip content from pivot row to prevent number growth
                let mut g = self[(i, j)].clone();
                for l in j + 1..self.ncols {
                    if F::one_is_gcd_unit() && self.field.is_one(&g) {
                        break;
                    }
                    g = self.field.gcd(&g, &self[(i, l)]);
                }
                if !self.field.is_one(&g) {
                    for l in j..self.ncols {
                        self[(i, l)] = self.field.try_div(&self[(i, l)], &g).unwrap();
                    }
                }

                for k in 0..i {
                    if !self.field.is_zero(&self[(k, j)]) {
                        let g = self.field.gcd(&self[(i, j)], &self[(k, j)]);

                        let scale_pivot = self.field.try_div(&self[(k, j)], &g).unwrap();
                        let scale_row = self.field.try_div(&self[(i, j)], &g).unwrap();

                        if !self.field.is_one(&scale_row) {
                            for l in 0..self.ncols {
                                if !self.field.is_zero(&self[(k, l)]) {
                                    self[(k, l)] = self.field.mul(&self[(k, l)], &scale_row);
                                }
                            }
                        }

                        self[(k, j)] = self.field.zero();
                        for l in j + 1..self.ncols {
                            self[(k, l)] = self
                                .field
                                .sub(&self[(k, l)], &self.field.mul(&self[(i, l)], &scale_pivot));
                        }
                    }
                }
            }
        }
    }

    /// Solve `A * x = b` for `x`, where `A` is `self` if `x` exists in the domain `F`.
    pub fn solve_fraction_free(&self, b: &Matrix<F>) -> Result<Matrix<F>, MatrixError<F>> {
        if self.nrows != b.nrows {
            return Err(MatrixError::ShapeMismatch);
        }
        if b.ncols != 1 {
            return Err(MatrixError::RightHandSideIsNotVector);
        }

        let (neqs, nvars) = (self.nrows, self.ncols);

        let mut m = self.augment(b)?;

        let rank = m.partial_row_reduce_fraction_free(nvars);

        for k in rank..neqs {
            if !self.field.is_zero(&m[(k, nvars)]) {
                return Err(MatrixError::Inconsistent);
            }
        }

        m.back_substitution_fraction_free(nvars);

        if rank < nvars {
            return Err(MatrixError::Underdetermined {
                rank,
                row_reduced_augmented_matrix: m,
            });
        }

        // now divide by the pivot, if it fails the result is not in the domain
        for x in 0..self.nrows {
            if let Some(q) = self.field.try_div(&m[(x, nvars)], &m[(x, x)]) {
                m[(x, nvars)] = q;
            } else {
                return Err(MatrixError::ResultNotInDomain);
            }
        }

        let result = Matrix {
            nrows: nvars,
            ncols: 1,
            data: (0..nvars).map(|i| m[(i, nvars)].clone()).collect(),
            field: m.field,
        };

        Ok(result)
    }
}

impl<F: Field> Vector<F> {
    /// Project the vector onto the `target` vector.
    pub fn project(&self, target: &Self) -> Self {
        target.clone() * self.field.div(&self.dot(target), &target.norm_squared())
    }

    /// Use the Gram–Schmidt method to create an orthogonal basis.
    pub fn orthogonalize(system: &[Self]) -> Vec<Vector<F>> {
        let mut res = vec![];

        for s in system {
            let mut new_vec = s.clone();
            for x in &res {
                new_vec -= &new_vec.project(x);
            }

            res.push(new_vec);
        }

        res
    }

    /// Apply the Gram-Schmidt method in-place and compute its factors.
    pub fn gram_schmidt(system: &mut [Self], factor: &mut [F::Element]) {
        for s in 0..system.len() {
            let mut res = system[s].clone();
            for j in 0..s {
                factor[s * system.len() + j] = system[0]
                    .field
                    .div(&system[s].dot(&system[j]), &factor[j * system.len() + j]);
                res -= &(system[j].clone() * factor[s * system.len() + j].clone());
            }

            factor[s * system.len() + s] = res.norm_squared();
            system[s] = res;
        }
    }
}

impl Vector<Z> {
    /// Apply the LLL lattice basis reduction algorithm.
    /// `delta` should be a constant between `1/4` and `1`. The most commonly used value is `3/4`.
    pub fn basis_reduction(system: &[Self], delta: Rational) -> Vec<Vector<Z>> {
        let mut res = system.to_vec();

        let mut mus = vec![0.into(); system.len() * system.len()];
        let mut b: Vec<Vector<Q>> = system
            .iter()
            .map(|v| v.map(|e| e.into(), Q))
            .collect::<Vec<_>>();

        Vector::gram_schmidt(&mut b, &mut mus);

        let mut k = 1;
        while k < system.len() {
            for j in (0..k).rev() {
                if mus[k * system.len() + j].abs() < (1, 2) {
                    continue;
                }

                let m = &res[j] * mus[k * system.len() + j].round_to_nearest_integer();
                res[k] -= &m;

                // reset b
                for (bb, rr) in b.iter_mut().zip(res.iter()) {
                    *bb = rr.map(|e| e.into(), Q);
                }

                Vector::gram_schmidt(&mut b, &mut mus); // TODO: do partial GS
            }

            if mus[k * system.len() + k]
                > &mus[(k - 1) * system.len() + (k - 1)]
                    * &(&delta - &(&mus[k * system.len() + k - 1] * &mus[k * system.len() + k - 1]))
            {
                k += 1;
            } else {
                res.swap(k, k - 1);

                for (bb, rr) in b.iter_mut().zip(res.iter()) {
                    *bb = rr.map(|e| e.into(), Q);
                }
                Vector::gram_schmidt(&mut b, &mut mus);

                k = 1.max(k - 1);
            }
        }

        res
    }
}

impl<F: Ring> Index<u32> for Vector<F> {
    type Output = F::Element;

    /// Get the `i`th entry of the vector.
    #[inline]
    fn index(&self, index: u32) -> &Self::Output {
        &self.data[index as usize]
    }
}

impl<F: Ring> IndexMut<u32> for Vector<F> {
    /// Get the `i`th entry of the vector.
    #[inline]
    fn index_mut(&mut self, index: u32) -> &mut F::Element {
        &mut self.data[index as usize]
    }
}

impl<F: Ring> Index<usize> for Vector<F> {
    type Output = F::Element;

    /// Get the `i`th entry of the vector.
    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<F: Ring> IndexMut<usize> for Vector<F> {
    /// Get the `i`th entry of the vector.
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut F::Element {
        &mut self.data[index]
    }
}

impl<F: Ring> Display for Vector<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.format(&PrintOptions::from_fmt(f), PrintState::from_fmt(f), f)
            .map(|_| ())
    }
}

impl<F: Ring> Add<&Vector<F>> for &Vector<F> {
    type Output = Vector<F>;

    /// Add two vectors.
    fn add(self, rhs: &Vector<F>) -> Self::Output {
        if self.data.len() != rhs.data.len() {
            panic!(
                "Cannot add vectors of different dimensions: {}  vs {}",
                self.data.len(),
                rhs.data.len()
            );
        }

        let mut m = self.new_zero();
        for (c, (a, b)) in m.data.iter_mut().zip(self.data.iter().zip(rhs.data.iter())) {
            *c = self.field.add(a, b);
        }

        m
    }
}

impl<F: Ring> AddAssign<&Vector<F>> for Vector<F> {
    ///Add two vectors in place.
    fn add_assign(&mut self, rhs: &Vector<F>) {
        if self.data.len() != rhs.data.len() {
            panic!(
                "Cannot add vectors of different dimensions: {}  vs {}",
                self.data.len(),
                rhs.data.len()
            );
        }

        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            self.field.add_assign(a, b);
        }
    }
}

impl<F: Ring> Sub<&Vector<F>> for &Vector<F> {
    type Output = Vector<F>;

    /// Subtract two vectors.
    fn sub(self, rhs: &Vector<F>) -> Self::Output {
        if self.data.len() != rhs.data.len() {
            panic!(
                "Cannot subtract vectors of different dimensions: {}  vs {}",
                self.data.len(),
                rhs.data.len()
            );
        }

        let mut m = self.new_zero();
        for (c, (a, b)) in m.data.iter_mut().zip(self.data.iter().zip(rhs.data.iter())) {
            *c = self.field.sub(a, b);
        }

        m
    }
}

impl<F: Ring> SubAssign<&Vector<F>> for Vector<F> {
    fn sub_assign(&mut self, rhs: &Vector<F>) {
        if self.data.len() != rhs.data.len() {
            panic!(
                "Cannot subtract vectors of different dimensions: {}  vs {}",
                self.data.len(),
                rhs.data.len()
            );
        }

        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            self.field.sub_assign(a, b);
        }
    }
}

impl<F: Ring> Neg for Vector<F> {
    type Output = Vector<F>;

    /// Negate each entry of the vector.
    fn neg(mut self) -> Self::Output {
        for e in &mut self.data {
            *e = self.field.neg(&*e);
        }

        self
    }
}

/// A matrix with entries that are elements of a ring `F`.
/// A vector can be represented as a matrix with one row or one column.
///
/// # Examples
///
/// ```    
/// # use numerica::domains::rational::Q;
/// # use numerica::tensors::matrix::Matrix;    
/// let a = Matrix::from_linear(vec![3.into(), 2.into(), 15.into(), 4.into()], 2, 2, Q).unwrap();
/// let inv = a.inv().unwrap();
/// assert_eq!(&a * &inv, Matrix::identity(2, Q));
/// ```
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct Matrix<F: Ring> {
    pub(crate) data: Vec<F::Element>,
    pub(crate) nrows: u32,
    pub(crate) ncols: u32,
    pub(crate) field: F,
}

impl<F: Ring> Matrix<F> {
    /// Create a new zeroed matrix with `nrows` rows and `ncols` columns.
    pub fn new(nrows: u32, ncols: u32, field: F) -> Matrix<F> {
        Matrix {
            data: (0..nrows as usize * ncols as usize)
                .map(|_| field.zero())
                .collect(),
            nrows,
            ncols,
            field,
        }
    }

    /// Create a new square matrix with `nrows` rows and ones on the main diagonal and zeroes elsewhere.
    pub fn identity(nrows: u32, field: F) -> Matrix<F> {
        Matrix {
            data: (0..nrows as usize * nrows as usize)
                .map(|i| {
                    if i % nrows as usize == i / nrows as usize {
                        field.one()
                    } else {
                        field.zero()
                    }
                })
                .collect(),
            nrows,
            ncols: nrows,
            field,
        }
    }

    /// Create a new matrix with the scalars `diag` on the main diagonal and zeroes elsewhere.
    pub fn eye(diag: &[F::Element], field: F) -> Matrix<F> {
        let mut m = Matrix::new(diag.len() as u32, diag.len() as u32, field);
        for (i, e) in diag.iter().enumerate() {
            m[(i as u32, i as u32)] = e.clone();
        }
        m
    }

    /// Create a new column vector from a list of scalars.
    pub fn new_vec(data: Vec<F::Element>, field: F) -> Matrix<F> {
        Matrix {
            nrows: data.len() as u32,
            ncols: 1,
            data,
            field,
        }
    }

    /// Convert a linear representation of a matrix to a `Matrix`.
    pub fn from_linear(
        data: Vec<F::Element>,
        nrows: u32,
        ncols: u32,
        field: F,
    ) -> Result<Matrix<F>, String> {
        if data.len() == (nrows * ncols) as usize {
            Ok(Matrix {
                data,
                nrows,
                ncols,
                field,
            })
        } else {
            Err(format!(
                "Data length does not match matrix dimensions: {} vs ({},{})",
                data.len(),
                nrows,
                ncols
            ))
        }
    }

    /// Create a new matrix from a 2-dimensional vector of scalars.
    pub fn from_nested_vec(matrix: Vec<Vec<F::Element>>, field: F) -> Result<Matrix<F>, String> {
        let mut data = vec![];

        let cols = matrix.first().map(|r| r.len()).unwrap_or(0);

        for d in matrix {
            if d.len() != cols {
                return Err("Matrix is not rectangular".to_string());
            }

            data.extend(d);
        }

        Ok(Matrix {
            nrows: (data.len() / cols) as u32,
            ncols: cols as u32,
            data,
            field,
        })
    }

    /// Return the number of rows.
    pub fn nrows(&self) -> usize {
        self.nrows as usize
    }

    /// Return the number of columns.
    pub fn ncols(&self) -> usize {
        self.ncols as usize
    }

    /// Return the field of the matrix entries.
    pub fn field(&self) -> &F {
        &self.field
    }

    /// Return an iterator over the rows of the matrix.
    pub fn row_iter(&self) -> Chunks<'_, F::Element> {
        self.data.chunks(self.ncols as usize)
    }

    /// Return true iff every entry in the matrix is zero.
    pub fn is_zero(&self) -> bool {
        self.data.iter().all(|e| self.field.is_zero(e))
    }

    /// Return true iff every non- main diagonal entry in the matrix is zero.
    pub fn is_diagonal(&self) -> bool {
        self.data
            .iter()
            .enumerate()
            .all(|(i, e)| i as u32 % self.ncols == i as u32 / self.ncols || self.field.is_zero(e))
    }

    /// Transpose the matrix.
    pub fn transpose(&self) -> Matrix<F> {
        let mut m = Matrix::new(self.ncols, self.nrows, self.field.clone());
        for i in 0..self.nrows {
            for j in 0..self.ncols {
                m[(j, i)] = self[(i, j)].clone();
            }
        }
        m
    }

    /// Return the underlying vector of scalars.
    pub fn into_vec(self) -> Vec<F::Element> {
        self.data
    }

    /// Transpose the matrix in-place.
    pub fn into_transposed(mut self) -> Matrix<F> {
        if self.nrows == self.ncols {
            for i in 0..self.nrows {
                for j in 0..i {
                    self.data
                        .swap((i * self.ncols + j) as usize, (j * self.ncols + i) as usize);
                }
            }

            (self.nrows, self.ncols) = (self.ncols, self.nrows);
            self
        } else {
            let mut m = Matrix::new(self.ncols, self.nrows, self.field.clone());
            for i in 0..self.nrows {
                for j in 0..self.ncols {
                    m[(j, i)] = std::mem::replace(&mut self[(i, j)], m.field.zero());
                }
            }
            m
        }
    }

    // Swap the i-th row with the j-th row, starting at column `start`.
    pub fn swap_rows(&mut self, mut i: u32, mut j: u32, start: u32) {
        if i == j {
            return;
        }

        if i > j {
            (i, j) = (j, i);
        }

        let (a, b) = self.data.split_at_mut((j * self.ncols) as usize);

        a[(i * self.ncols + start) as usize..((i + 1) * self.ncols) as usize]
            .swap_with_slice(&mut b[start as usize..self.ncols as usize]);
    }

    // Swap the i-th column with the j-th column.
    pub fn swap_cols(&mut self, i: u32, j: u32) {
        if i == j {
            return;
        }

        for k in 0..self.nrows {
            self.data
                .swap((k * self.ncols + i) as usize, (k * self.ncols + j) as usize);
        }
    }

    /// Multiply the scalar `e` to each entry of the matrix.
    pub fn mul_scalar(&self, e: &F::Element) -> Matrix<F> {
        Matrix {
            data: self.data.iter().map(|ee| self.field.mul(ee, e)).collect(),
            nrows: self.nrows,
            ncols: self.ncols,
            field: self.field.clone(),
        }
    }

    /// Return a row-first iterator over the entries of the matrix.
    pub fn iter(&self) -> std::slice::Iter<'_, F::Element> {
        self.data.iter()
    }

    /// Apply a function `f` to each entry of the matrix.
    pub fn map<G: Ring>(&self, f: impl Fn(&F::Element) -> G::Element, field: G) -> Matrix<G> {
        Matrix {
            data: self.data.iter().map(f).collect(),
            nrows: self.nrows,
            ncols: self.ncols,
            field,
        }
    }

    /// Get the squared Euclidean norm of the matrix.
    pub fn norm_squared(&self) -> F::Element {
        let mut norm = self.field.zero();
        for d in &self.data {
            self.field.add_mul_assign(&mut norm, d, d);
        }
        norm
    }

    /// Convert a matrix that is a row or column vector into a [Vector].
    /// Will panic if the input is not vector-like.
    pub fn into_vector(self) -> Vector<F> {
        if self.nrows != 1 && self.ncols != 1 {
            panic!("The matrix is not a vector");
        }

        Vector {
            data: self.data,
            field: self.field,
        }
    }

    /// Convert the matrix into a sparse matrix in CSR format.
    pub fn to_sparse(&self) -> SparseMatrix<F> {
        let mut values = vec![];
        let mut col_idcs = vec![];
        let mut row_idcs = vec![0];

        for row in self.row_iter() {
            for (col, val) in row.iter().enumerate() {
                if !self.field.is_zero(val) {
                    values.push(val.clone());
                    col_idcs.push(col as u32);
                }
            }
            row_idcs.push(values.len());
        }

        SparseMatrix::from_csr(
            self.nrows,
            self.ncols,
            values,
            row_idcs,
            col_idcs,
            self.field.clone(),
        )
    }

    /// Augment the matrix with another matrix, e.g. create `[A B]` from matrix `A` and `B`.
    ///
    /// Returns an error when the matrices do not have the same number of rows.
    pub fn augment(&self, matrix: &Matrix<F>) -> Result<Matrix<F>, MatrixError<F>> {
        if self.nrows != matrix.nrows {
            return Err(MatrixError::ShapeMismatch);
        }

        let mut m = Matrix::new(self.nrows, self.ncols + matrix.ncols, self.field.clone());

        for (r, (r1, r2)) in self.row_iter().zip(matrix.row_iter()).enumerate() {
            m.data[r * m.ncols as usize..r * m.ncols as usize + self.ncols as usize]
                .clone_from_slice(r1);
            m.data[r * m.ncols as usize + self.ncols as usize
                ..r * m.ncols as usize + m.ncols as usize]
                .clone_from_slice(r2);
        }

        Ok(m)
    }

    /// Split the matrix into two matrices at the `index`-th column.
    pub fn split_col(&self, index: u32) -> Result<(Matrix<F>, Matrix<F>), MatrixError<F>> {
        if index == 0 || index >= self.ncols - 1 {
            return Err(MatrixError::ShapeMismatch);
        }

        let mut m1 = Matrix::new(self.nrows, index, self.field.clone());
        let mut m2 = Matrix::new(self.nrows, self.ncols - index, self.field.clone());

        // chunks could be 0!
        for (r, (r1, r2)) in self.row_iter().zip(
            m1.data
                .chunks_mut(index as usize)
                .zip(m2.data.chunks_mut(self.ncols as usize - index as usize)),
        ) {
            r1.clone_from_slice(&r[..index as usize]);
            r2.clone_from_slice(&r[index as usize..]);
        }

        Ok((m1, m2))
    }

    /// Compute the determinant of the matrix.
    pub fn det(&self) -> Result<F::Element, MatrixError<F>> {
        if self.nrows != self.ncols {
            Err(MatrixError::NotSquare)?;
        }

        let f = &self.field;
        match self.nrows {
            0 => Err(MatrixError::Singular),
            1 => Ok(self.data[0].clone()),
            2 => Ok(f.sub(
                &f.mul(&self.data[0], &self.data[3]),
                &f.mul(&self.data[1], &self.data[2]),
            )),
            3 => {
                let m0 = f.mul(
                    &self.data[0],
                    &f.sub(
                        &f.mul(&self.data[4], &self.data[8]),
                        &f.mul(&self.data[5], &self.data[7]),
                    ),
                );
                let m1 = f.mul(
                    &self.data[1],
                    &f.sub(
                        &f.mul(&self.data[5], &self.data[6]),
                        &f.mul(&self.data[3], &self.data[8]),
                    ),
                );
                let m2 = f.mul(
                    &self.data[2],
                    &f.sub(
                        &f.mul(&self.data[3], &self.data[7]),
                        &f.mul(&self.data[4], &self.data[6]),
                    ),
                );

                Ok(f.add(&f.add(&m0, &m1), &m2))
            }
            _ => {
                // Bareiss fraction-free determinant algorithm
                let mut m = self.clone();
                let mut flip_sign = false;

                for k in 0..self.nrows - 1 {
                    if self.field.is_zero(&m[(k, k)]) {
                        let mut pivot_row = k;
                        while pivot_row < self.nrows && self.field.is_zero(&m[(pivot_row, k)]) {
                            pivot_row += 1;
                        }

                        if pivot_row == self.nrows {
                            return Ok(self.field.zero());
                        }

                        m.swap_rows(k, pivot_row, k);
                        flip_sign = !flip_sign;
                    }

                    for i in k + 1..self.nrows {
                        for j in k + 1..self.ncols {
                            let val = f.sub(
                                &f.mul(&m[(i, j)], &m[(k, k)]),
                                &f.mul(&m[(i, k)], &m[(k, j)]),
                            );

                            if k > 0 {
                                m[(i, j)] =
                                    f.try_div(&val, &m[(k - 1, k - 1)]).unwrap_or_else(|| {
                                        panic!(
                                            "Inexact division in determinant calculation of {}",
                                            self
                                        )
                                    });
                            } else {
                                m[(i, j)] = val;
                            }
                        }
                    }
                }

                let det = m[(self.nrows - 1, self.nrows - 1)].clone();
                if flip_sign { Ok(f.neg(&det)) } else { Ok(det) }
            }
        }
    }
}

impl<F: Ring> SelfRing for Matrix<F> {
    fn is_one(&self) -> bool {
        self.data.iter().enumerate().all(|(i, e)| {
            i as u32 % self.ncols == i as u32 / self.ncols && self.field.is_one(e)
                || self.field.is_zero(e)
        })
    }

    fn is_zero(&self) -> bool {
        self.data.iter().all(|x| self.field.is_zero(x))
    }

    fn format<W: std::fmt::Write>(
        &self,
        opts: &PrintOptions,
        mut state: PrintState,
        f: &mut W,
    ) -> Result<bool, std::fmt::Error> {
        if state.in_sum {
            f.write_char('+')?;
        }
        state.in_sum = false;
        state.in_product = false;
        state.in_exp = false;

        if opts.mode.is_latex() {
            f.write_str("\\begin{pmatrix}")?;

            for (ri, r) in self.row_iter().enumerate() {
                for (ci, c) in r.iter().enumerate() {
                    self.field.format(c, opts, state, f)?;

                    if ci + 1 < self.ncols as usize {
                        f.write_str(" & ")?;
                    }
                }
                if ri + 1 < self.nrows as usize {
                    f.write_str(r" \\ ")?;
                }
            }

            f.write_str("\\end{pmatrix}")?;
            Ok(false)
        } else if opts.pretty_matrix {
            f.write_char('[')?;
            for (ri, r) in self.row_iter().enumerate() {
                if ri > 0 {
                    f.write_char(' ')?;
                }
                f.write_char('[')?;
                for (ci, c) in r.iter().enumerate() {
                    self.field.format(c, opts, state, f)?;

                    if ci + 1 < self.ncols as usize {
                        write!(f, "{}", ", ".bold().color(Color::BrightMagenta))?;
                    }
                }
                f.write_char(']')?;

                if ri + 1 < self.nrows as usize {
                    f.write_str(",\n")?;
                }
            }
            f.write_char(']')?;
            Ok(false)
        } else {
            f.write_char('{')?;
            for (ri, r) in self.row_iter().enumerate() {
                f.write_char('{')?;
                for (ci, c) in r.iter().enumerate() {
                    self.field.format(c, opts, state, f)?;

                    if ci + 1 < self.ncols as usize {
                        f.write_char(',')?;
                    }
                }
                f.write_char('}')?;
                if ri + 1 < self.nrows as usize {
                    f.write_char(',')?;
                }
            }
            f.write_char('}')?;
            Ok(false)
        }
    }
}

impl<F: Ring> InternalOrdering for Matrix<F> {
    fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.nrows, self.ncols)
            .cmp(&(other.nrows, other.ncols))
            .then_with(|| self.data.internal_cmp(&other.data))
    }
}

impl<F: Ring> Index<u32> for Matrix<F> {
    type Output = [F::Element];

    /// Get the `index`th row of the matrix.
    #[inline]
    fn index(&self, index: u32) -> &Self::Output {
        &self.data[index as usize * self.nrows as usize..(index as usize + 1) * self.nrows as usize]
    }
}

impl<F: Ring> Index<(u32, u32)> for Matrix<F> {
    type Output = F::Element;

    /// Get the `i`th row and `j`th column of the matrix, where `index=(i,j)`.
    #[inline]
    fn index(&self, index: (u32, u32)) -> &Self::Output {
        &self.data[(index.0 * self.ncols + index.1) as usize]
    }
}

impl<F: Ring> IndexMut<(u32, u32)> for Matrix<F> {
    /// Get the `i`th row and `j`th column of the matrix, where `index=(i,j)`.
    #[inline]
    fn index_mut(&mut self, index: (u32, u32)) -> &mut F::Element {
        &mut self.data[(index.0 * self.ncols + index.1) as usize]
    }
}

impl<F: Ring> Display for Matrix<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.format(&PrintOptions::from_fmt(f), PrintState::from_fmt(f), f)
            .map(|_| ())
    }
}

impl<F: Ring> Add<&Matrix<F>> for &Matrix<F> {
    type Output = Matrix<F>;

    /// Add two matrices.
    fn add(self, rhs: &Matrix<F>) -> Self::Output {
        if self.nrows != rhs.nrows || self.ncols != rhs.ncols {
            panic!(
                "Cannot add matrices of different dimensions: ({},{}) vs ({},{})",
                self.nrows, self.ncols, rhs.nrows, rhs.ncols
            );
        }

        let mut m = Matrix::new(self.nrows, self.ncols, self.field.clone());
        for (c, (a, b)) in m.data.iter_mut().zip(self.data.iter().zip(rhs.data.iter())) {
            *c = self.field.add(a, b);
        }

        m
    }
}

impl<F: Ring> AddAssign<&Matrix<F>> for Matrix<F> {
    ///Add two matrices in place.
    fn add_assign(&mut self, rhs: &Matrix<F>) {
        if self.nrows != rhs.nrows || self.ncols != rhs.ncols {
            panic!(
                "Cannot add matrices of different dimensions: ({},{}) vs ({},{})",
                self.nrows, self.ncols, rhs.nrows, rhs.ncols
            );
        }

        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            self.field.add_assign(a, b);
        }
    }
}

impl<F: Ring> Sub<&Matrix<F>> for &Matrix<F> {
    type Output = Matrix<F>;

    /// Subtract two matrices.
    fn sub(self, rhs: &Matrix<F>) -> Self::Output {
        if self.nrows != rhs.nrows || self.ncols != rhs.ncols {
            panic!(
                "Cannot add matrices of different dimensions: ({},{}) vs ({},{})",
                self.nrows, self.ncols, rhs.nrows, rhs.ncols
            );
        }

        let mut m = Matrix::new(self.nrows, self.ncols, self.field.clone());
        for (c, (a, b)) in m.data.iter_mut().zip(self.data.iter().zip(rhs.data.iter())) {
            *c = self.field.sub(a, b);
        }

        m
    }
}

impl<F: Ring> SubAssign<&Matrix<F>> for Matrix<F> {
    ///Add two matrices in place.
    fn sub_assign(&mut self, rhs: &Matrix<F>) {
        if self.nrows != rhs.nrows || self.ncols != rhs.ncols {
            panic!(
                "Cannot add matrices of different dimensions: ({},{}) vs ({},{})",
                self.nrows, self.ncols, rhs.nrows, rhs.ncols
            );
        }

        for (a, b) in self.data.iter_mut().zip(rhs.data.iter()) {
            self.field.sub_assign(a, b);
        }
    }
}

impl<F: Ring> Mul<&Matrix<F>> for &Matrix<F> {
    type Output = Matrix<F>;

    /// Multiply two matrices.
    fn mul(self, rhs: &Matrix<F>) -> Self::Output {
        if self.ncols != rhs.nrows {
            panic!(
                "Cannot multiply matrices because of a dimension mismatch: ({},{}) vs ({},{})",
                self.nrows, self.ncols, rhs.nrows, rhs.ncols
            );
        }

        let mut m = Matrix::new(self.nrows, rhs.ncols, self.field.clone());

        for i in 0..self.nrows {
            for j in 0..rhs.ncols {
                let sum = &mut m[(i, j)];
                for k in 0..self.ncols {
                    self.field.add_mul_assign(sum, &self[(i, k)], &rhs[(k, j)]);
                }
            }
        }

        m
    }
}

impl<F: Ring> MulAssign<&Matrix<F>> for Matrix<F> {
    ///Multiply two matrices in place.
    fn mul_assign(&mut self, rhs: &Matrix<F>) {
        *self = &*self * rhs;
    }
}

impl<F: Ring> Neg for Matrix<F> {
    type Output = Matrix<F>;

    /// Negate each entry of the matrix.
    fn neg(mut self) -> Self::Output {
        for e in &mut self.data {
            *e = self.field.neg(&*e);
        }

        self
    }
}

impl<'a, F: Ring> IntoIterator for &'a Matrix<F> {
    type Item = &'a F::Element;
    type IntoIter = std::slice::Iter<'a, F::Element>;

    /// Create a row-major iterator over the matrix.
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

/// Errors that can occur when performing matrix operations.
#[derive(Debug)]
pub enum MatrixError<F: Ring> {
    Underdetermined {
        rank: u32,
        row_reduced_augmented_matrix: Matrix<F>,
    },
    Inconsistent,
    NotSquare,
    Singular,
    ShapeMismatch,
    RightHandSideIsNotVector,
    ResultNotInDomain,
}

impl<F: Ring> std::fmt::Display for MatrixError<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MatrixError::Underdetermined {
                rank,
                row_reduced_augmented_matrix,
            } => {
                writeln!(f, "The system is underdetermined with rank {rank}")?;
                writeln!(
                    f,
                    "\nRow reduced augmented matrix:\n{row_reduced_augmented_matrix}"
                )
            }
            MatrixError::Inconsistent => write!(f, "The system is inconsistent"),
            MatrixError::NotSquare => write!(f, "The matrix is not square"),
            MatrixError::Singular => write!(f, "The matrix is singular"),
            MatrixError::ShapeMismatch => write!(f, "The shape of the matrix is not compatible"),
            MatrixError::RightHandSideIsNotVector => {
                write!(f, "The right-hand side is not a vector")
            }
            MatrixError::ResultNotInDomain => write!(
                f,
                "The result does not belong to the same domain as the matrix."
            ),
        }
    }
}

impl<F: EuclideanDomain> Matrix<F> {
    /// Get the content of the matrix, i.e. the gcd of all entries.
    pub fn content(&self) -> F::Element {
        let mut gcd = self.field.zero();
        for e in &self.data {
            gcd = self.field.gcd(&gcd, e);
        }

        gcd
    }

    /// Divide each entry in the matrix by the scalar `e`.
    pub fn div_scalar(&self, e: &F::Element) -> Matrix<F> {
        Matrix {
            data: self
                .data
                .iter()
                .map(|ee| {
                    let (q, r) = self.field.quot_rem(ee, e);
                    assert_eq!(r, self.field.zero());
                    q
                })
                .collect(),
            nrows: self.nrows,
            ncols: self.ncols,
            field: self.field.clone(),
        }
    }

    /// Construct the same matrix, but with the content removed.
    pub fn primitive_part(&self) -> Matrix<F> {
        let c = self.content();
        if self.field.is_one(&c) {
            self.clone()
        } else {
            self.div_scalar(&c)
        }
    }
}

impl<F: Field> Matrix<F> {
    /// Compute the inverse of a square matrix, if it exists.
    /// Otherwise, this function returns `MatrixError::Singular`.
    pub fn inv(&self) -> Result<Self, MatrixError<F>> {
        if self.nrows != self.ncols {
            Err(MatrixError::NotSquare)?;
        }

        // hardcode options for 2x2 and 3x3 matrices
        let f = &self.field;
        if self.nrows == 2 {
            let d = self.field.sub(
                &f.mul(&self.data[0], &self.data[3]),
                &f.mul(&self.data[1], &self.data[2]),
            );

            if self.field.is_zero(&d) {
                return Err(MatrixError::Singular);
            }

            let d_inv = self.field.inv(&d);
            return Ok(Matrix {
                nrows: 2,
                ncols: 2,
                data: vec![
                    f.mul(&self.data[3], &d_inv),
                    f.mul(&f.neg(&self.data[1]), &d_inv),
                    f.mul(&f.neg(&self.data[2]), &d_inv),
                    f.mul(&self.data[0], &d_inv),
                ],
                field: f.clone(),
            });
        }

        if self.nrows == 3 {
            #[inline(always)]
            fn sub_mul_mul<F: Field>(
                f: &F,
                r: &[F::Element],
                a: usize,
                b: usize,
                c: usize,
                d: usize,
            ) -> F::Element {
                f.sub(&f.mul(&r[a], &r[b]), &f.mul(&r[c], &r[d]))
            }

            let mut m = Matrix {
                nrows: 3,
                ncols: 3,
                data: vec![
                    sub_mul_mul(f, &self.data, 4, 8, 5, 7),
                    sub_mul_mul(f, &self.data, 2, 7, 1, 8),
                    sub_mul_mul(f, &self.data, 1, 5, 2, 4),
                    sub_mul_mul(f, &self.data, 5, 6, 3, 8),
                    sub_mul_mul(f, &self.data, 0, 8, 2, 6),
                    sub_mul_mul(f, &self.data, 2, 3, 0, 5),
                    sub_mul_mul(f, &self.data, 3, 7, 4, 6),
                    sub_mul_mul(f, &self.data, 1, 6, 0, 7),
                    sub_mul_mul(f, &self.data, 0, 4, 1, 3),
                ],
                field: f.clone(),
            };

            let d = f.add(
                &f.mul(&self.data[0], &m.data[0]),
                &f.add(
                    &f.mul(&self.data[1], &m.data[3]),
                    &f.mul(&self.data[2], &m.data[6]),
                ),
            );

            if self.field.is_zero(&d) {
                return Err(MatrixError::Singular);
            }

            let d_inv = self.field.inv(&d);
            for e in &mut m.data {
                *e = f.mul(&*e, &d_inv);
            }
            return Ok(m);
        }

        // use Gaussian elimination with an augmented matrix to find the inverse
        let mut m = Matrix::new(self.nrows, self.nrows * 2, self.field.clone());
        for r in 0..self.nrows {
            for c in 0..self.ncols {
                m[(r, c)] = self[(r, c)].clone();
            }
            m[(r, self.nrows + r)] = self.field.one();
        }

        let rank = m.row_reduce(m.ncols);

        if rank < self.nrows as usize {
            return Err(MatrixError::Singular);
        }

        for r in 0..self.nrows {
            for c in 0..self.ncols {
                m.data[r as usize * self.ncols as usize + c as usize] = std::mem::replace(
                    &mut m.data
                        [r as usize * 2 * self.ncols as usize + self.ncols as usize + c as usize],
                    self.field.zero(),
                );
            }
        }

        m.ncols = self.ncols;
        m.data.truncate(m.nrows as usize * m.ncols as usize);

        Ok(m)
    }

    /// Compute the determinant of the matrix in-place.
    pub fn det_in_place(&mut self) -> Result<F::Element, MatrixError<F>> {
        if self.nrows != self.ncols {
            Err(MatrixError::NotSquare)?;
        }

        if self.nrows != self.partial_row_reduce(self.nrows) {
            return Ok(self.field.zero());
        }

        let mut det = self.field.one();
        for x in 0..self.nrows {
            self.field
                .mul_assign(&mut det, &self.data[(x + self.nrows * x) as usize]);
        }

        Ok(det)
    }

    /// Write the first `max_col` columns of the matrix in (non-reduced) echelon form.
    /// Returns the matrix rank.
    pub fn partial_row_reduce(&mut self, max_col: u32) -> u32 {
        let zero = self.field.zero();

        let mut i = 0;
        for j in 0..max_col.min(self.ncols) {
            if self.field.is_zero(&self[(i, j)]) {
                // Select a non-zero pivot.
                for k in i + 1..self.nrows {
                    if !self.field.is_zero(&self[(k, j)]) {
                        self.swap_rows(i, k, j);
                        break;
                    }
                }

                // zero column found
                if self.field.is_zero(&self[(i, j)]) {
                    continue;
                }
            }
            let x = self[(i, j)].clone();
            let inv_x = self.field.inv(&x);
            for k in i + 1..self.nrows {
                if !self.field.is_zero(&self[(k, j)]) {
                    let s = self.field.mul(&self[(k, j)], &inv_x);
                    self[(k, j)] = self.field.zero();
                    for l in j + 1..self.ncols {
                        let mut e = std::mem::replace(&mut self[(k, l)], zero.clone());
                        self.field.sub_mul_assign(&mut e, &self[(i, l)], &s);
                        self[(k, l)] = e;
                    }
                }
            }

            i += 1;
            if i >= self.nrows {
                break;
            }
        }

        i
    }

    /// Create a row-reduced matrix from a matrix in echelon form.
    pub fn back_substitution(&mut self, mut max_col: u32) {
        max_col = max_col.min(self.ncols);
        let field = self.field.clone();
        for i in (0..self.nrows).rev() {
            if let Some(j) = (0..max_col).find(|&j| !self.field.is_zero(&self[(i, j)])) {
                if !field.is_one(&self[(i, j)]) {
                    let inv_x = field.inv(&self[(i, j)]);

                    for k in j..self.ncols {
                        field.mul_assign(&mut self[(i, k)], &inv_x);
                    }
                }

                for k in 0..i {
                    if !self.field.is_zero(&self[(k, j)]) {
                        let scale = std::mem::replace(&mut self[(k, j)], field.zero());
                        for l in j + 1..self.ncols {
                            let mut e = std::mem::replace(&mut self[(k, l)], field.zero());
                            field.sub_mul_assign(&mut e, &self[(i, l)], &scale);
                            self[(k, l)] = e;
                        }
                    }
                }
            }
        }
    }

    /// Solve `A * x = b` for `x`, where `A` is `self`.
    pub fn solve(&self, b: &Matrix<F>) -> Result<Matrix<F>, MatrixError<F>> {
        if self.nrows != b.nrows {
            return Err(MatrixError::ShapeMismatch);
        }
        if b.ncols != 1 {
            return Err(MatrixError::RightHandSideIsNotVector);
        }

        let (neqs, nvars) = (self.nrows, self.ncols);

        let mut m = self.augment(b)?;

        let rank = m.partial_row_reduce(nvars);

        for k in rank..neqs {
            if !self.field.is_zero(&m[(k, nvars)]) {
                return Err(MatrixError::Inconsistent);
            }
        }

        m.back_substitution(nvars);

        if rank < nvars {
            return Err(MatrixError::Underdetermined {
                rank,
                row_reduced_augmented_matrix: m,
            });
        }

        let result = Matrix {
            nrows: nvars,
            ncols: 1,
            data: (0..nvars).map(|i| m[(i, nvars)].clone()).collect(),
            field: m.field,
        };

        Ok(result)
    }

    /// Solve `A * x = b` for `x`, where `A` is `self` and return any solution if the
    /// system is underdetermined.
    pub fn solve_any(&self, b: &Matrix<F>) -> Result<Matrix<F>, MatrixError<F>> {
        match self.solve(b) {
            Ok(x) => Ok(x),
            Err(MatrixError::Underdetermined {
                row_reduced_augmented_matrix,
                ..
            }) => {
                let mut x = Matrix::new(self.ncols, 1, self.field.clone());
                for r in row_reduced_augmented_matrix.row_iter() {
                    for (i, e) in r.iter().enumerate().take(self.ncols as usize) {
                        if !self.field.is_zero(e) {
                            x.data[i] = r.last().unwrap().clone();
                            break;
                        }
                    }
                }

                debug_assert_eq!(&(self * &x), b);

                Ok(x)
            }
            Err(e) => Err(e),
        }
    }

    /// Row-reduce the first `max_col` columns of the matrix in-place using Gaussian elimination and return the rank.
    pub fn row_reduce(&mut self, max_col: u32) -> usize {
        let rank = self.partial_row_reduce(max_col) as usize;
        self.back_substitution(max_col);
        rank
    }

    /// Get the rank of the matrix.
    pub fn rank(&self) -> usize {
        self.clone().partial_row_reduce(self.ncols) as usize
    }
}

#[cfg(test)]
mod test {
    use crate::{
        domains::{integer::Z, rational::Q},
        tensors::matrix::{Matrix, Vector},
    };

    #[test]
    fn basics() {
        let a = Matrix::from_linear(
            vec![1.into(), 2.into(), 3.into(), 4.into(), 5.into(), 6.into()],
            2,
            3,
            Z,
        )
        .unwrap();

        assert_eq!(a.transpose().data, vec![1, 4, 2, 5, 3, 6]);

        assert_eq!(a.clone().into_transposed().data, vec![1, 4, 2, 5, 3, 6]);

        assert_eq!((-a.clone()).data, vec![-1, -2, -3, -4, -5, -6]);

        assert_eq!((&a - &a).data, vec![0, 0, 0, 0, 0, 0]);

        let b = Matrix::from_nested_vec(
            vec![
                vec![7.into(), 8.into()],
                vec![9.into(), 10.into()],
                vec![11.into(), 12.into()],
            ],
            Z,
        )
        .unwrap();

        let c = &a * &b;

        assert_eq!(c.data, vec![58, 64, 139, 154]);
        assert_eq!(&c[1], &[139, 154]);
        assert_eq!(c[(0, 1)], 64);

        let mut d = a.clone();
        d += &a;

        assert_eq!(d.data, vec![2, 4, 6, 8, 10, 12]);

        let c_m = c.map(|x| x * 2u64, Z);
        assert_eq!(c_m.data, vec![116, 128, 278, 308]);
    }

    #[test]
    fn solve() {
        let a = Matrix::from_linear(
            vec![
                1.into(),
                2.into(),
                3.into(),
                4.into(),
                5.into(),
                16.into(),
                7.into(),
                8.into(),
                9.into(),
            ],
            3,
            3,
            Q,
        )
        .unwrap();

        assert_eq!(
            a.inv().unwrap().data,
            vec![
                (-83, 60),
                (1, 10),
                (17, 60),
                (19, 15),
                (-1, 5),
                (-1, 15),
                (-1, 20),
                (1, 10),
                (-1, 20)
            ]
        );
        assert_eq!(a.det().unwrap(), 60);

        let b = Matrix::from_linear(vec![1.into(), 2.into(), 3.into()], 3, 1, Q).unwrap();

        let r = a.solve(&b).unwrap();
        assert_eq!(r.data, vec![(-1, 3), (2, 3), (0, 1)]);
    }

    #[test]
    fn solve_any() {
        let m = vec![
            vec![1.into(), 1.into(), 1.into()],
            vec![1.into(), 1.into(), 2.into()],
        ];
        let rhs = Matrix::new_vec(vec![3.into(), 2.into()], Q);

        let mat = Matrix::from_nested_vec(m, Q).unwrap();
        let r = mat.solve_any(&rhs).unwrap();
        assert_eq!(r.data, vec![4, 0, -1]);
    }

    #[test]
    fn row_reduce() {
        let mut a = Matrix::from_linear(
            vec![
                1.into(),
                2.into(),
                3.into(),
                4.into(),
                5.into(),
                6.into(),
                7.into(),
                8.into(),
                9.into(),
            ],
            3,
            3,
            Q,
        )
        .unwrap();

        assert_eq!(a.row_reduce(a.ncols() as u32), 2);

        assert_eq!(a.data, vec![1, 0, (-1), 0, 1, 2, 0, 0, 0]);
    }

    #[test]
    fn orthogonalize() {
        let a = vec![
            Vector::new(vec![1.into(), 2.into(), 3.into()], Q),
            Vector::new(vec![4.into(), 5.into(), 6.into()], Q),
            Vector::new(vec![7.into(), 8.into(), 9.into()], Q),
        ];

        let res = Vector::orthogonalize(&a);
        for x in &res {
            for y in &res {
                if x != y {
                    assert_eq!(x.dot(y), (0, 1));
                }
            }
        }
    }

    #[test]
    fn inverse() {
        let a =
            Matrix::from_linear(vec![3.into(), 2.into(), 15.into(), 4.into()], 2, 2, Q).unwrap();

        let inv = a.inv().unwrap();
        assert_eq!(&a * &inv, Matrix::identity(2, Q));

        let a = Matrix::from_linear(
            vec![
                3.into(),
                2.into(),
                15.into(),
                4.into(),
                9.into(),
                6.into(),
                7.into(),
                8.into(),
                17.into(),
            ],
            3,
            3,
            Q,
        )
        .unwrap();

        let inv = a.inv().unwrap();
        assert_eq!(&a * &inv, Matrix::identity(3, Q));

        let a = Matrix::from_linear(
            vec![
                3.into(),
                2.into(),
                15.into(),
                4.into(),
                9.into(),
                6.into(),
                7.into(),
                8.into(),
                17.into(),
                45.into(),
                23.into(),
                12.into(),
                13.into(),
                14.into(),
                15.into(),
                16.into(),
            ],
            4,
            4,
            Q,
        )
        .unwrap();

        let inv = a.inv().unwrap();
        assert_eq!(&a * &inv, Matrix::identity(4, Q));
    }

    #[test]
    fn basis_reduction() {
        let v1 = Vector::new(vec![1.into(), 0.into(), 0.into(), 31416.into()], Z);
        let v2 = Vector::new(vec![0.into(), 1.into(), 0.into(), 27183.into()], Z);
        let v3 = Vector::new(vec![0.into(), 0.into(), 1.into(), (-320177).into()], Z);

        let basis = Vector::basis_reduction(&[v1, v2, v3], (3, 4).into());

        // 32.0177 = 5 * pi + 6 * e
        assert_eq!(basis[0].data, &[5, 6, 1, 1]);
    }

    #[test]
    fn split_augment() {
        let a = Matrix::from_linear(
            vec![1.into(), 1.into(), 1.into(), 1.into(), 1.into(), 2.into()],
            2,
            3,
            Q,
        )
        .unwrap();

        let b = Matrix::from_linear(vec![5.into(), 6.into(), 7.into(), 8.into()], 2, 2, Q).unwrap();

        let c = a.augment(&b).unwrap();

        let (d, e) = c.split_col(3).unwrap();

        assert_eq!(a, d);
        assert_eq!(b, e);
    }

    #[test]
    fn solve_fraction_free() {
        let a = Matrix::from_linear(
            vec![
                1.into(),
                (-2).into(),
                (3).into(),
                2.into(),
                (1).into(),
                1.into(),
                (-3).into(),
                2.into(),
                (-2).into(),
            ],
            3,
            3,
            Z,
        )
        .unwrap();

        let rhs = Matrix::from_linear(vec![7.into(), 4.into(), (-10).into()], 3, 1, Z).unwrap();

        let r = a.solve_fraction_free(&rhs).unwrap();
        assert_eq!(r.data, [2, -1, 1]);
    }
}
