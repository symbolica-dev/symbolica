//! Sparse linear algebra methods using matrices and vectors.
//!
//! # Example
//!
//! ```rust
//! use numerica::tensors::sparse::{SparseMatrix};
//! use numerica::domains::{
//! 	integer::{IntegerRing, Integer},
//!     rational::{FractionField}
//! };
//! let r = IntegerRing::new();
//!
//! let mat = SparseMatrix::from_csr(2,3,vec![Integer::new(10),Integer::new(7),Integer::new(13)],vec![0,2,3],vec![0,2,1],r);
//! assert_eq!(mat.fmt_mma(), "{{{1,1}->10,{1,3}->7,{2,2}->13},{2,3}}");

use std::{
    collections::HashSet,
    fmt::Display,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    sync::Mutex,
};

use itertools::Itertools;

use rand::Rng;

use rayon::prelude::*;

use crate::{
    domains::{
        Field, InternalOrdering, Ring, RingPrinter, SelfRing,
        integer::{Integer, IntegerRing},
        rational::{Fraction, FractionField},
    },
    printer::{PrintOptions, PrintState},
    tensors::matrix::Matrix,
};

/// A sparse vector in a compressed format (compressed sparse row style).
///
/// We keep the column indices sorted at all times.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct SparseVector<F: Ring> {
    /// The non-zero entries of the vector sorted by index.
    pub(crate) values: Vec<F::Element>,
    /// The indices corresponding to the entries of `values`.
    /// Must have the same length a `values`.
    pub(crate) idcs: Vec<u32>,
    /// The size/length of the vector.
    pub(crate) len: u32,
    /// The ring/field of the elements of the matrix
    pub(crate) field: F,
}

impl<F: Ring> SparseVector<F> {
    /// Create a new zeroed sparse vector over the ring/field `F` of length `len`.
    pub fn new(len: u32, field: F) -> SparseVector<F> {
        SparseVector {
            values: Vec::new(),
            idcs: Vec::new(),
            len: len,
            field: field,
        }
    }

    /// Create a new sparse vector from CSR-like data.
    ///
    /// The values vector should optimally not contain any zero elements. We do not check this, so it will be carried along.
    /// * `len` - The length of the vector.
    /// * `values` - Sorted (by index) non-zero entries of the vector..
    /// * `idcs` - The sorted indices/positions of the values.
    /// * `field` - the field of the vector entries.
    ///
    pub fn from_csr(
        len: u32,
        values: Vec<F::Element>,
        idcs: Vec<u32>,
        field: F,
    ) -> SparseVector<F> {
        SparseVector {
            values,
            idcs,
            len,
            field,
        }
    }

    /// Create a new sparse vector from CSR-like data.
    ///
    /// The values vector should optimally not contain any zero elements. We do not check this, so it will be carried along.
    /// * `len` - The length of the vector.
    /// * `values` - Sorted (by index) non-zero entries of the vector..
    /// * `idcs` - The sorted indices/positions of the values.
    /// * `field` - the field of the vector entries.
    ///
    pub fn from_csr_slices(
        len: u32,
        values: &[F::Element],
        idcs: &[u32],
        field: F,
    ) -> SparseVector<F> {
        SparseVector {
            values: values.to_vec(),
            idcs: idcs.to_vec(),
            len,
            field,
        }
    }

    /// Create a new sparse vector from (pos, value) pairs.
    ///
    /// The values should optimally not contain any zero elements. We do not check this, so it will be carried along.
    /// * `len` - The length of the vector.
    /// * `pairs` - (pos, vaule) pairs of the non-zero entries.
    /// * `field` - the field of the vector entries.
    ///
    pub fn from_pairs(len: u32, pairs: Vec<(u32, F::Element)>, field: F) -> SparseVector<F> {
        let n = pairs.len();
        let mut idcs = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);

        for (i, v) in pairs {
            idcs.push(i);
            values.push(v);
        }

        SparseVector {
            values,
            idcs,
            len,
            field,
        }
    }

    /// Create a new sparse vector from (pos, value) pairs.
    ///
    /// The values should optimally not contain any zero elements. We do not check this, so it will be carried along.
    /// * `len` - The length of the vector.
    /// * `pairs` - (pos, vaule) pairs of the non-zero entries.
    /// * `field` - the field of the vector entries.
    ///
    pub fn from_pairs_slices(len: u32, pairs: &[(u32, F::Element)], field: F) -> SparseVector<F> {
        let n = pairs.len();
        let mut idcs = Vec::with_capacity(n);
        let mut values = Vec::with_capacity(n);

        for (i, v) in pairs {
            idcs.push(*i);
            values.push(v.clone());
        }

        SparseVector {
            values,
            idcs,
            len,
            field,
        }
    }

    /// Return the length of the vector .
    pub fn len(&self) -> u32 {
        self.len
    }

    /// Format in Mathematica form.
    ///
    /// Simply apply `SparseArray@@` to the output in MMA.
    pub fn fmt_mma(&self) -> String {
        let vals = self
            .idcs
            .iter()
            .zip(self.values.iter())
            .map(|(idx, val)| {
                //format each element as {idx,col}->val
                let val_printer = RingPrinter::new(&self.field, &val);
                format!("{{{},{}}}->{}", idx + 1, 1, val_printer)
            })
            .join(",");

        format!("{{{{{}}},{{{},{}}}}}", vals, self.len, 1)
    }
}

/// An error enum for some functions of SparseMatrix.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub enum SparseMatrixError<F: Ring> {
    /// Shape of matrix and vector don't match
    ShapeMismatch,
    /// Fields of arguments don't match
    FieldMismatch,
    /// No solution exists
    Inconsistent,
    /// The matrix is not square
    NotSquare,
    /// System is underdetermined (infinite solutions)
    Underdetermined {
        rank: usize,
        row_reduced_augmented_matrix: SparseMatrix<F>,
    },
    /// Singular, non-invertible
    Singular,
}

impl<F: Ring> std::error::Error for SparseMatrixError<F> {}

impl<F: Ring> std::fmt::Display for SparseMatrixError<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SparseMatrixError::ShapeMismatch => {
                write!(f, "The shape of the matrix is not compatible")
            }
            SparseMatrixError::FieldMismatch => write!(f, "The fields of the arguments differ"),
            SparseMatrixError::Inconsistent => write!(f, "The system is inconsistent"),
            SparseMatrixError::NotSquare => write!(f, "The matrix is not square"),
            SparseMatrixError::Underdetermined {
                rank,
                row_reduced_augmented_matrix,
            } => {
                writeln!(f, "The system is underdetermined with rank {rank}")?;
                writeln!(
                    f,
                    "\nRow reduced augmented matrix:\n{row_reduced_augmented_matrix}"
                )
            }
            SparseMatrixError::Singular => write!(f, "The matrix is singular"),
        }
    }
}

/// A sparse matrix in compressed sparse row (CSR) format.
///
/// We keep the column indices (of each row) sorted at all times.
#[derive(Clone, Hash, PartialEq, Eq, Debug)]
pub struct SparseMatrix<F: Ring> {
    /// The non-zero entries of the matrix sorted by row and column
    pub(crate) values: Vec<F::Element>,
    /// Indices where new rows start within `values`, including an after-end index.
    /// Has length nrows + 1.
    pub(crate) row_ptrs: Vec<usize>,
    /// The column indices corresponding to the entries of `values`.
    /// Must have the same length a `values`.
    pub(crate) col_idcs: Vec<u32>,
    /// Number of rows
    pub(crate) nrows: u32,
    /// Number of columns
    pub(crate) ncols: u32,
    /// The ring/field of the elements of the matrix
    pub(crate) field: F,
}

pub struct SparseMatrixIterator<'a, F: Ring> {
    matrix: &'a SparseMatrix<F>,
    row: u32,
    pos: u32,
}

impl<'a, F: Ring> Iterator for SparseMatrixIterator<'a, F> {
    type Item = (u32, u32, &'a F::Element);

    /// Iterate in row-major order over the matrix.
    fn next(&mut self) -> Option<Self::Item> {
        while self.row < self.matrix.nrows {
            let row_start = self.matrix.row_ptrs[self.row as usize];
            let row_end = self.matrix.row_ptrs[(self.row + 1) as usize];

            if (self.pos as usize) < (row_end - row_start) {
                let idx = row_start + (self.pos as usize);
                let col = self.matrix.col_idcs[idx];
                let val = &self.matrix.values[idx];
                self.pos += 1;
                return Some((self.row, col, val));
            } else {
                //move to next row
                self.row += 1;
                self.pos = 0;
            }
        }
        None
    }
}

impl<'a, F: Ring> IntoIterator for &'a SparseMatrix<F> {
    type Item = (u32, u32, &'a F::Element);
    type IntoIter = SparseMatrixIterator<'a, F>;

    /// Create a row-major iterator over the matrix.
    fn into_iter(self) -> Self::IntoIter {
        SparseMatrixIterator {
            matrix: self,
            row: 0,
            pos: 0,
        }
    }
}

pub struct SparseMatrixRowIterator<'a, F: Ring> {
    /// The matrix we are iterating over.
    matrix: &'a SparseMatrix<F>,
    /// The next row
    row: u32,
    /// After the end row (needed for DoubleEndedIterator)
    end_row: u32,
}

impl<'a, F: Ring> Iterator for SparseMatrixRowIterator<'a, F> {
    /// (row_idx, col_idcs, values)
    type Item = (u32, &'a [u32], &'a [F::Element]);

    /// Iterate over the rows of the matrix.
    fn next(&mut self) -> Option<Self::Item> {
        if self.row < self.end_row {
            let row = self.row;
            
            let row_start = self.matrix.row_ptrs[row as usize];
            let row_end = self.matrix.row_ptrs[(row + 1) as usize];
            
            //move to next row
            self.row += 1;

            Some((
                row,
                &self.matrix.col_idcs[row_start..row_end],
                &self.matrix.values[row_start..row_end]
            ))
        } else {
            None
        }
    }
}

impl<'a, F: Ring> DoubleEndedIterator for SparseMatrixRowIterator<'a, F> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.row < self.end_row {
            self.end_row -= 1;

            let row = self.end_row;

            let row_start = self.matrix.row_ptrs[row as usize];
            let row_end = self.matrix.row_ptrs[(row + 1) as usize];

            Some((
                row,
                &self.matrix.col_idcs[row_start..row_end],
                &self.matrix.values[row_start..row_end],
            ))
        } else {
            None
        }
    }
}

impl<F: Ring> Display for SparseMatrix<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.format(&PrintOptions::from_fmt(f), PrintState::from_fmt(f), f)
            .map(|_| ())
    }
}

impl<F: Ring> InternalOrdering for SparseMatrix<F> {
    fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.nrows, self.ncols)
            .cmp(&(other.nrows, other.ncols))
            .then_with(|| {
                for (a, b) in self.into_iter().zip(other.into_iter()) {
                    if (a.0, a.1) != (b.0, b.1) {
                        return (a.0, a.1).cmp(&(b.0, b.1));
                    }

                    let ord = a.2.internal_cmp(b.2);
                    if ord != std::cmp::Ordering::Equal {
                        return ord;
                    }
                }
                std::cmp::Ordering::Equal
            })
    }
}

impl<F: Ring> SelfRing for SparseMatrix<F> {
    fn is_one(&self) -> bool {
        self.values.iter().enumerate().all(|(i, e)| {
            i as u32 % self.ncols == i as u32 / self.ncols && self.field.is_one(e)
                || self.field.is_zero(e)
        })
    }

    fn is_zero(&self) -> bool {
        self.values.is_empty()
    }

    fn format<W: std::fmt::Write>(
        &self,
        opts: &PrintOptions,
        state: PrintState,
        f: &mut W,
    ) -> Result<bool, std::fmt::Error> {
        self.to_dense().format(opts, state, f)
    }
}

impl<F: Ring> SparseMatrix<F> {
    /// Create a new zeroed sparse matrix over the ring/field `F` with `nrows` rows and `ncols` columns
    ///
    /// * `nrows` - number of rows
    /// * `ncols` - number of columns
    /// * `field` - the field of the matrix entries
    pub fn new(nrows: u32, ncols: u32, field: F) -> SparseMatrix<F> {
        SparseMatrix {
            values: Vec::new(),
            row_ptrs: vec![0; (nrows + 1) as usize],
            col_idcs: Vec::new(),
            nrows,
            ncols,
            field,
        }
    }

    /// Create a new sparse matrix over the ring/field `F` from explicit CSR data
    ///
    /// The values vector should optimally not contain any zero elements. We do not check this, so it will be carried along.
    /// * `nrows` - number of rows
    /// * `ncols` - number of columns
    /// * `values` - non-zero entries sorted by row and column
    /// * `row_ptrs` - indices where new rows start within `values`, including an after-end index
    /// * `col_idcs` - column indices corresponding to entries of `values`
    /// * `field` - the field of the matrix entries
    pub fn from_csr(
        nrows: u32,
        ncols: u32,
        values: Vec<F::Element>,
        row_ptrs: Vec<usize>,
        col_idcs: Vec<u32>,
        field: F,
    ) -> SparseMatrix<F> {
        assert!(values.len() == col_idcs.len());
        assert!(row_ptrs.len() == ((nrows + 1) as usize));
        SparseMatrix {
            values,
            row_ptrs,
            col_idcs,
            nrows,
            ncols,
            field,
        }
    }

    /// Create a new sparse matrix over the ring/field `F` from explicit CSR data
    ///
    /// The values vector should optimally not contain any zero elements. We do not check this, so it will be carried along.
    /// * `nrows` - number of rows
    /// * `ncols` - number of columns
    /// * `values` - non-zero entries sorted by row and column
    /// * `row_ptrs` - indices where new rows start within `values`, including an after-end index
    /// * `col_idcs` - column indices corresponding to entries of `values`
    /// * `field` - the field of the matrix entries
    pub fn from_csr_slices(
        nrows: u32,
        ncols: u32,
        values: &[F::Element],
        row_ptrs: &[usize],
        col_idcs: &[u32],
        field: F,
    ) -> SparseMatrix<F> {
        assert!(values.len() == col_idcs.len());
        assert!(row_ptrs.len() == ((nrows + 1) as usize));
        SparseMatrix {
            values: values.to_vec(),
            row_ptrs: row_ptrs.to_vec(),
            col_idcs: col_idcs.to_vec(),
            nrows,
            ncols,
            field,
        }
    }

    /// Create a sparse matrix from ordered triplets of (row, column, entry)
    ///
    /// # Arguments
    /// * `nrows` - number of rows
    /// * `ncols` - number of columns
    /// * `triplets` - ordered(!) triplets of (row, column, entry). Row and column indices are 0-indexed
    /// * `field` - the ring/field of the matrix entries
    ///
    /// # Example
    /// ```rust
    /// use numerica::tensors::sparse::{SparseMatrix};
    /// use numerica::domains::integer::{IntegerRing, Integer};
    /// let r = IntegerRing::new();
    ///
    /// let mat = SparseMatrix::from_triplets(4,3, vec![(0,0,Integer::new(15)),(0,2,Integer::new(-23)),(2,1,Integer::new(-7)),(2,2,Integer::new(2)),(3,0,Integer::new(-1))], r);
    /// println!("{}", mat.fmt_mma());
    /// assert_eq!(mat.fmt_mma(), "{{{1,1}->15,{1,3}->-23,{3,2}->-7,{3,3}->2,{4,1}->-1},{4,3}}");
    /// ```
    ///
    pub fn from_triplets(
        nrows: u32,
        ncols: u32,
        triplets: Vec<(u32, u32, F::Element)>,
        field: F,
    ) -> SparseMatrix<F> {
        debug_assert!(triplets.is_sorted_by_key(|&(row, col, _)| (row, col)));
        let mut ret = SparseMatrix {
            values: Vec::with_capacity(triplets.len()),
            row_ptrs: Vec::with_capacity((nrows + 1) as usize),
            col_idcs: Vec::with_capacity(triplets.len()),
            nrows,
            ncols,
            field,
        };
        ret.row_ptrs.push(0);
        let mut current_row: u32 = 0;
        for (row, col, el) in triplets {
            while current_row < row {
                //start new row/insert empty rows
                ret.row_ptrs.push(ret.values.len());
                current_row += 1;
            }
            ret.values.push(el);
            ret.col_idcs.push(col);
        }
        //finish up the row_ptrs
        while current_row < ret.nrows {
            ret.row_ptrs.push(ret.values.len());
            current_row += 1;
        }
        debug_assert!(ret.row_ptrs.len() == (ret.nrows + 1) as usize);

        ret
    }

    /// Create a new square matrix with `nrows` rows and ones on the main diagonal and zeroes elsewhere.
    pub fn identity(nrows: u32, field: F) -> SparseMatrix<F> {
        SparseMatrix {
            values: vec![field.one(); nrows as usize],
            col_idcs: (0..nrows).collect(),
            row_ptrs: (0..(nrows + 1) as usize).collect(),
            nrows: nrows,
            ncols: nrows,
            field: field,
        }
    }

    /// Return the number of rows.
    pub fn nrows(&self) -> u32 {
        self.nrows as u32
    }

    /// Return the number of columns.
    pub fn ncols(&self) -> u32 {
        self.ncols as u32
    }

    /// Return the field of the matrix entries.
    pub fn field(&self) -> &F {
        &self.field
    }

    /// Return the number of non-zero entries.
    pub fn nvalues(&self) -> usize {
        self.values.len()
    }

    /// Access the values vector.
    pub fn values(&self) -> &Vec<F::Element> {
        &self.values
    }

    /// Access the column indices vector.
    pub fn col_idcs(&self) -> &Vec<u32> {
        &self.col_idcs
    }

    /// Access the row pointer vector.
    pub fn row_ptrs(&self) -> &Vec<usize> {
        &self.row_ptrs
    }

    /// Multiply the scalar `e` to each entry of the matrix
    pub fn mul_scalar(&self, el: &F::Element) -> SparseMatrix<F> {
        let mut ret = SparseMatrix {
            values: self
                .values
                .iter()
                .map(|ell| self.field.mul(ell, el))
                .collect(),
            row_ptrs: self.row_ptrs.clone(),
            col_idcs: self.col_idcs.clone(),
            nrows: self.nrows,
            ncols: self.ncols,
            field: self.field.clone(),
        };
        ret.erase_zeroes();

        ret
    }

    /// Add a new row to the matrix
    ///
    /// # Parameters
    /// * `values` - the values of the non-zero entries of the row
    /// * `col_idcs` - the column indices of the non-zero entries of the row
    pub fn add_row(&mut self, values: Vec<F::Element>, col_idcs: Vec<u32>) -> () {
        self.row_ptrs
            .push(self.row_ptrs.last().unwrap() + values.len());
        self.values.extend(values);
        self.col_idcs.extend(col_idcs);
        self.nrows += 1;
    }

    /// Add empty columns to the matrix.
    ///
    /// * `col_pos` - Ordered(!) positions where the new columns should be inserted. Each entry must NOT account for previously inserted columns.
    pub fn add_cols(&mut self, col_pos: &Vec<u32>) -> () {
        debug_assert!(col_pos.is_sorted());

        //update ncols
        self.ncols += col_pos.len() as u32;

        //update col_idcs, row by row
        for pair in self.row_ptrs.windows(2) {
            let mut col_pos_it: u32 = 0;
            for pos in pair[0]..pair[1] {
                //advance iterator as long *it <= col_idx at pos
                while (col_pos_it as usize) < col_pos.len()
                    && col_pos[col_pos_it as usize] <= self.col_idcs[pos]
                {
                    col_pos_it += 1;
                }
                //shift the current col_idx by the number of new columns that are inserted before it
                self.col_idcs[pos] += col_pos_it;
            }
        }
    }

    /// Append a column to the right of the matrix
    pub fn append_col(&mut self, col: SparseVector<F>) -> () {
        debug_assert_eq!(col.values.len(), col.idcs.len());
        assert_eq!(col.len(), self.nrows);

        let old_values = std::mem::take(&mut self.values);
        let old_col_idcs = std::mem::take(&mut self.col_idcs);

        self.values = Vec::with_capacity(old_values.len() + col.values.len());
        self.col_idcs = Vec::with_capacity(old_col_idcs.len() + col.idcs.len());

        let mut old_values_iter = old_values.into_iter();
        let mut old_col_idcs_iter = old_col_idcs.into_iter();
        let mut col_iter = col.idcs.into_iter().zip(col.values.into_iter());

        let mut current_col = col_iter.next();

        for row in 0..self.nrows as usize {
            let row_start = self.row_ptrs[row];
            let row_end = self.row_ptrs[row + 1];
            let row_len = row_end - row_start;

            // update row_ptrs
            self.row_ptrs[row] = self.values.len();

            // move old values
            self.values.extend(old_values_iter.by_ref().take(row_len));
            self.col_idcs
                .extend(old_col_idcs_iter.by_ref().take(row_len));

            // move value from new column
            if current_col
                .as_ref()
                .map_or(false, |&(idx, _)| idx as usize == row)
            {
                let (_, val) = current_col.take().unwrap();
                self.values.push(val);
                self.col_idcs.push(self.ncols);
                current_col = col_iter.next();
            }
        }
        //update after-the-end
        self.row_ptrs[self.nrows as usize] = self.values.len();

        self.ncols += 1;
    }

    /// For a square matrix, it will append the identity matrix to the right.
    ///
    /// I.e. A -> (A|I).
    pub fn append_identity(&mut self) -> () {
        assert_eq!(self.ncols, self.nrows);
        let n = self.ncols;
        let old_values = std::mem::take(&mut self.values);
        let old_col_idcs = std::mem::take(&mut self.col_idcs);

        self.values = Vec::with_capacity(old_values.len() + (n as usize));
        self.col_idcs = Vec::with_capacity(old_col_idcs.len() + (n as usize));

        let mut old_values_iter = old_values.into_iter();
        let mut old_col_idcs_iter = old_col_idcs.into_iter();

        for row in 0..n as usize {
            let start = self.row_ptrs[row];
            let end = self.row_ptrs[row + 1];
            let row_len = end - start;

            //update row_ptrs
            self.row_ptrs[row] = self.values.len();

            //move old values
            self.values.extend(old_values_iter.by_ref().take(row_len));
            self.col_idcs
                .extend(old_col_idcs_iter.by_ref().take(row_len));

            //push the new 1 in this row
            self.values.push(self.field.one());
            self.col_idcs.push(n + (row as u32));
        }
        //update after-the-end
        self.row_ptrs[n as usize] = self.values.len();

        self.ncols += n;
    }

    /// Returns the row-reversed right half of the matrix.
    ///
    /// The number of rows needs to be divisible by two.
    /// E.g. 2n x n matrix (A|B) -> rev(B), where rev means that the row ordering has been reversed
    pub(crate) fn take_rev_right_half(&mut self) -> () {
        assert!(self.ncols % 2 == 0);
        let n = self.ncols / 2;
        let mut old_values = std::mem::take(&mut self.values);
        let old_col_idcs = std::mem::take(&mut self.col_idcs);
        let old_row_ptrs = std::mem::take(&mut self.row_ptrs);

        //very conservative allocation
        self.values = Vec::new();
        self.col_idcs = Vec::new();
        self.row_ptrs = Vec::new();

        for row in (0..self.nrows as usize).rev() {
            let start = old_row_ptrs[row];
            let end = old_row_ptrs[row + 1];

            //update row_ptrs
            self.row_ptrs.push(self.values.len());

            for px in start..end {
                let col_idx = old_col_idcs[px];
                if col_idx >= n {
                    self.values
                        .push(std::mem::replace(&mut old_values[px], self.field.zero()));
                    self.col_idcs.push(col_idx - n);
                }
            }
        }
        //set after-the-end
        self.row_ptrs.push(self.values.len());
        self.ncols = n;
    }

    /// Return the number of non-zero entries in the given row.
    pub fn row_weight(&self, row: u32) -> u32 {
        (self.row_ptrs[(row + 1) as usize] - self.row_ptrs[row as usize]) as u32
    }

    /// Extract the last column of the matrix.
    pub fn last_column(self) -> SparseVector<F> {
        let mut values = self.values;
        let mut ret = SparseVector::new(self.nrows, self.field.clone());
        for row in 0..self.nrows as usize {
            let start = self.row_ptrs[row];
            let end = self.row_ptrs[row + 1];
            if end > start && self.col_idcs[end - 1] + 1 == self.ncols {
                //last entry of this column is non zero
                ret.idcs.push(row as u32);
                ret.values
                    .push(std::mem::replace(&mut values[end - 1], self.field.zero()));
            }
        }
        ret
    }

    /// Extract the reversed last column of the matrix.
    pub fn last_column_rev(self) -> SparseVector<F> {
        let mut values = self.values;
        let mut ret = SparseVector::new(self.nrows, self.field.clone());
        for row in (0..self.nrows as usize).rev() {
            let start = self.row_ptrs[row];
            let end = self.row_ptrs[row + 1];
            if end > start && self.col_idcs[end - 1] + 1 == self.ncols {
                //last entry of this column is non zero
                ret.idcs.push(self.nrows - (row as u32) - 1);
                ret.values
                    .push(std::mem::replace(&mut values[end - 1], self.field.zero()));
            }
        }
        ret
    }

    /// Extract the last column of the matrix, sorted by the corresponding pivot column.
    ///
    /// # Arguments
    /// * `pivots` - The pivot positions for each column, i.e. there is a pivot on column `j` and `row pivots[j]`.
    pub fn last_column_by_pivot(self, pivots: &Vec<Option<u32>>) -> SparseVector<F> {
        let mut values = self.values;
        let mut ret = SparseVector::new(self.nrows, self.field.clone());
        for (col, pivot) in pivots.iter().enumerate() {
            if let Some(row) = pivot {
                let start = self.row_ptrs[*row as usize];
                let end = self.row_ptrs[(row + 1) as usize];
                if end > start && self.col_idcs[end - 1] + 1 == self.ncols {
                    //last entry of this column is non zero
                    ret.idcs.push(col as u32);
                    ret.values
                        .push(std::mem::replace(&mut values[end - 1], self.field.zero()));
                }
            }
        }
        ret
    }

    /// Get the last row of the matrix
    ///
    /// Returns (row_idx, col_idcs, values)
    pub fn last_row(&self) -> Option<(u32, &[u32], &[F::Element])> {
        if self.nrows == 0 {
            None
        } else {
            let row_start = self.row_ptrs[self.row_ptrs.len() - 2] as usize;
            let row_end = self.row_ptrs[self.row_ptrs.len() - 1] as usize;

            Some((self.nrows - 1, &self.col_idcs[row_start..row_end], &self.values[row_start..row_end]))
        }
    }

    /// Format in Mathematica form
    ///
    /// Simply apply `SparseArray@@` to the output in MMA.
    ///
    /// # Example
    /// ```rust
    /// use numerica::tensors::sparse::{SparseMatrix};
    /// use numerica::domains::integer::{IntegerRing, Integer};
    /// let r = IntegerRing::new();
    ///
    /// let mat = SparseMatrix::from_csr(2,3,vec![Integer::new(10),Integer::new(7),Integer::new(13)],vec![0,2,3],vec![0,2,1],r);
    /// assert_eq!(mat.fmt_mma(), "{{{1,1}->10,{1,3}->7,{2,2}->13},{2,3}}");
    ///  ```
    pub fn fmt_mma(&self) -> String {
        let vals = self
            .row_ptrs
            .windows(2)
            .enumerate() //iterate over rows (with index)
            .flat_map(|(idx, pair)| {
                //iterate over row entries
                (pair[0]..pair[1]).map(move |i| {
                    //format each element as {idx,col}->val
                    let val = RingPrinter::new(&self.field, &self.values[i]);
                    format!("{{{},{}}}->{}", idx + 1, self.col_idcs[i] + 1, val)
                })
            })
            .join(",");
        //format as {vals, {nrows, ncols}}
        format!("{{{{{}}},{{{},{}}}}}", vals, self.nrows, self.ncols)
    }

    /// Erase zeroes from the values vector.
    fn erase_zeroes(&mut self) -> () {
        let mut pos: usize = 0;
        //iterate over rows
        for row in 0..(self.nrows as usize) {
            let start = self.row_ptrs[row];
            let end = self.row_ptrs[row + 1];

            //record new start of row
            self.row_ptrs[row] = pos;

            for i in start..end {
                if !self.field.is_zero(&self.values[i]) {
                    //move it to correct position
                    self.values[pos] = self.values[i].clone();
                    self.col_idcs[pos] = self.col_idcs[i];
                    pos += 1;
                }
            }
        }
        //write after-the-end pos for last row
        self.row_ptrs[self.nrows as usize] = pos;

        //shrink the vector to their actual values
        self.values.truncate(pos);
        self.col_idcs.truncate(pos);
    }

    /// Convert the sparse matrix to a dense matrix.
    pub fn to_dense(&self) -> Matrix<F> {
        let mut mat = Matrix::new(self.nrows, self.ncols, self.field.clone());

        for row in 0..self.nrows as usize {
            let start = self.row_ptrs[row];
            let end = self.row_ptrs[row + 1];

            for i in start..end {
                let col = self.col_idcs[i] as usize;
                mat[(row as u32, col as u32)] = self.values[i].clone();
            }
        }

        mat
    }

    /// Append another SparseMatrix at the bottom of self
    ///
    /// Note that the fields need to be exactly the same
    pub fn append(&mut self, mut other: SparseMatrix<F>) {
        debug_assert_eq!(self.field, other.field);
        debug_assert_eq!(self.ncols, other.ncols);

        //get shift for the row indices
        let shift = self.values.len();

        //simply append values and col indices
        self.values.append(&mut other.values);
        self.col_idcs.append(&mut other.col_idcs);

        self.row_ptrs.reserve(other.row_ptrs.len() - 1);
        for &row_idx in &other.row_ptrs[1..] {
            self.row_ptrs.push(row_idx + shift);
        }
        self.nrows += other.nrows;
    }

    /// Sorts the rows of the matrix by their pivot column.
    ///
    /// The matrix must be forward solved, e.g. with a [`SparseRowReducer`], which also provides the `pivots` object.
    /// # Arguments
    /// * `pivots` - The pivot positions for each column, i.e. there is a pivot on column `j` and row `pivots[j]`.
    pub fn sort_rows_by_pivot(&mut self, pivots: &Vec<Option<u32>>) {
        let mut new_values = Vec::with_capacity(self.values.len());
        let mut new_col_idcs = Vec::with_capacity(self.col_idcs.len());
        let mut new_row_ptrs = Vec::with_capacity(self.row_ptrs.len());
        new_row_ptrs.push(0);

        for pivot in pivots {
            if let Some(row) = pivot {
                let start = self.row_ptrs[*row as usize];
                let end = self.row_ptrs[(row + 1) as usize];
                for px in start..end {
                    new_values.push(std::mem::replace(&mut self.values[px], self.field.zero()));
                    new_col_idcs.push(self.col_idcs[px]);
                }
                new_row_ptrs.push(new_values.len());
            }
        }
        self.values = new_values;
        self.col_idcs = new_col_idcs;
        self.row_ptrs = new_row_ptrs;
    }

    /// Get an iterator over the rows of the matrix.
    ///
    /// It iterates over tuples of the form: (row_idx, col_idcs, values)
    pub fn row_iter(&self) -> SparseMatrixRowIterator<'_, F> {
        SparseMatrixRowIterator {
            matrix: self,
            row: 0,
            end_row: self.nrows(),
        }
    }
}

impl<F: Field> SparseMatrix<F> {
    /// Solve the linear system `A * x = b`, where `A` is `self` using a [`SparseRowReducer`]
    pub fn solve(mut self, b: SparseVector<F>) -> Result<SparseVector<F>, SparseMatrixError<F>> {
        if self.nrows() != b.len() {
            return Err(SparseMatrixError::ShapeMismatch);
        }
        if self.field != b.field {
            return Err(SparseMatrixError::FieldMismatch);
        }

        let nvars = self.ncols;
        // append b as the last column
        self.append_col(b);

        //perform forward solving
        let sparse_row_reducer = SparseRowReducer::from_matrix_checked(&self, LuLMode::None);

        if sparse_row_reducer.is_none() {
            return Err(SparseMatrixError::Inconsistent);
        }

        let mut sparse_row_reducer = sparse_row_reducer.unwrap();

        //check for underdeterminedness
        // rank < nvars
        if sparse_row_reducer.u.nrows() < nvars {
            return Err(SparseMatrixError::Underdetermined {
                rank: sparse_row_reducer.u.nrows() as usize,
                row_reduced_augmented_matrix: sparse_row_reducer.u,
            });
        }

        //go to actual rref form
        sparse_row_reducer.back_substitute();

        //solution is the reversed last column of U
        Ok(sparse_row_reducer.u.last_column_rev())
    }

    /// Compute the determinant of the matrix.
    pub fn det(&self) -> Result<F::Element, SparseMatrixError<F>> {
        if self.nrows != self.ncols {
            Err(SparseMatrixError::NotSquare)?;
        }

        let sparse_row_reducer = SparseRowReducer::from_matrix_check_dependent(self, LuLMode::Full);

        if sparse_row_reducer.is_none() {
            //has a vanishing row, must be zero
            return Ok(self.field.zero());
        }

        let sparse_row_reducer = sparse_row_reducer.unwrap();

        //take the product of all the diagonal entries of L (are guaranteed to be last in the row)
        let mut det = self.field.one();
        for row in 0..sparse_row_reducer.l.nrows {
            //use "pivot-last" property
            self.field.mul_assign(
                &mut det,
                &sparse_row_reducer.l.values[sparse_row_reducer.l.row_ptrs[(row + 1) as usize] - 1],
            );
        }
        //we also need the sign of the permutation of rows that would sort them (according to pivot)
        let mut inversions = 0;
        for i in 0..sparse_row_reducer.pivots.len() {
            for j in i + 1..sparse_row_reducer.pivots.len() {
                if sparse_row_reducer.pivots[i].unwrap() > sparse_row_reducer.pivots[j].unwrap() {
                    inversions += 1;
                }
            }
        }
        if inversions % 2 == 1 {
            det = self.field.neg(det);
        }

        Ok(det)
    }

    /// Compute the inverse of the matrix.
    pub fn inv(&self) -> Result<Self, SparseMatrixError<F>> {
        if self.nrows != self.ncols {
            Err(SparseMatrixError::NotSquare)?;
        }
        //need a copy of self
        let mut mat = self.clone();
        mat.append_identity();

        let sparse_row_reducer = SparseRowReducer::from_matrix_check_dependent(&mat, LuLMode::Full);

        if sparse_row_reducer.is_none() {
            //not invertible
            return Err(SparseMatrixError::Singular)?;
        }

        let mut sparse_row_reducer = sparse_row_reducer.unwrap();

        sparse_row_reducer.back_substitute();

        //extract the right half of the matrix
        sparse_row_reducer.u.take_rev_right_half();

        Ok(sparse_row_reducer.u)
    }
}

impl<F: Field + Sync + Send> SparseMatrix<F>
where
    F::Element: Sync + Send,
{
    /// Solve the linear system `A * x = b`, where `A` is `self` using the a [`SparseRowReducer`]
    /// The back substitution uses parallelized code.
    pub fn solve_parallel(
        mut self,
        b: SparseVector<F>,
    ) -> Result<SparseVector<F>, SparseMatrixError<F>> {
        if self.nrows() != b.len() {
            return Err(SparseMatrixError::ShapeMismatch);
        }
        if self.field != b.field {
            return Err(SparseMatrixError::FieldMismatch);
        }

        let nvars = self.ncols;
        // append b as the last column
        self.append_col(b);

        //perform forward solving
        let sparse_row_reducer = SparseRowReducer::from_matrix_checked(&self, LuLMode::None);

        if sparse_row_reducer.is_none() {
            return Err(SparseMatrixError::Inconsistent);
        }

        let mut sparse_row_reducer = sparse_row_reducer.unwrap();

        //check for underdeterminedness
        // rank < nvars
        if sparse_row_reducer.u.nrows() < nvars {
            return Err(SparseMatrixError::Underdetermined {
                rank: sparse_row_reducer.u.nrows() as usize,
                row_reduced_augmented_matrix: sparse_row_reducer.u,
            });
        }

        //go to actual rref form
        sparse_row_reducer.back_substitute_parallel();

        //solution is the reversed last column of U
        Ok(sparse_row_reducer.u.last_column_by_pivot(&sparse_row_reducer.pivots))
    }
}

impl SparseMatrix<FractionField<IntegerRing>> {
    /// Generate a random SparseMatrix with the given dimensions and number of entries
    pub fn random(nrows: u32, ncols: u32, nentries: usize) -> Self {
        assert!((nrows as usize) * (ncols as usize) > nentries);
        //idea: generate random entry triplets and use the from_triplets constructor
        let mut rng = rand::rng();

        //generate nentries unique coordinates
        let mut pairs: HashSet<(u32, u32)> = HashSet::with_capacity(nentries);
        while pairs.len() < nentries {
            pairs.insert((rng.random_range(0..nrows), rng.random_range(0..ncols)));
        }

        let f = FractionField::new(IntegerRing::new());

        let mut triplets: Vec<(u32, u32, Fraction<IntegerRing>)> = pairs
            .into_iter()
            .enumerate()
            .map(|(_, (a, b))| {
                (
                    a,
                    b,
                    f.to_element(
                        Integer::new(rng.random::<i64>()),
                        Integer::new(rng.random::<i64>()),
                        true,
                    ),
                )
            })
            .collect();

        triplets.sort();

        SparseMatrix::from_triplets(nrows, ncols, triplets, f)
    }
}

impl<F: Ring> Neg for SparseMatrix<F> {
    type Output = SparseMatrix<F>;

    /// Negate each entry of the matrix
    fn neg(mut self) -> Self::Output {
        for val in &mut self.values {
            *val = self.field.neg(&*val);
        }

        self
    }
}

impl<F: Ring> Add<&SparseMatrix<F>> for &SparseMatrix<F> {
    type Output = SparseMatrix<F>;

    /// Add two sparse matrices
    fn add(self, rhs: &SparseMatrix<F>) -> Self::Output {
        if self.nrows != rhs.nrows || self.ncols != rhs.ncols {
            panic!(
                "Cannot add sparse matrices of different dimensions: ({},{}) vs ({},{})",
                self.nrows, self.ncols, rhs.nrows, rhs.ncols
            );
        }
        if self.field != rhs.field {
            panic!(
                "Cannot add sparse matrices over different fields: {} vs {}",
                self.field, rhs.field
            );
        }

        if self.values.is_empty() {
            return rhs.clone();
        }
        if rhs.values.is_empty() {
            return self.clone();
        }

        let mut values = Vec::new();
        let mut col_idcs = Vec::new();
        let mut row_ptrs = Vec::with_capacity((self.nrows + 1) as usize);
        row_ptrs.push(0);

        //iterate through both matrices simultaneously
        for row in 0..self.nrows as usize {
            let mut lhs_idx = self.row_ptrs[row];
            let mut rhs_idx = rhs.row_ptrs[row];
            let lhs_end = self.row_ptrs[row + 1];
            let rhs_end = rhs.row_ptrs[row + 1];

            //iterate through row entries
            while lhs_idx < lhs_end || rhs_idx < rhs_end {
                match (lhs_idx < lhs_end, rhs_idx < rhs_end) {
                    (true, true) => {
                        let lhs_col = self.col_idcs[lhs_idx];
                        let rhs_col = rhs.col_idcs[rhs_idx];

                        if lhs_col == rhs_col {
                            let sum = self.field.add(&self.values[lhs_idx], &rhs.values[rhs_idx]);

                            if !self.field.is_zero(&sum) {
                                col_idcs.push(lhs_col);
                                values.push(sum)
                            }
                            lhs_idx += 1;
                            rhs_idx += 1;
                        } else if lhs_col < rhs_col {
                            col_idcs.push(lhs_col);
                            values.push(self.values[lhs_idx].clone());
                            lhs_idx += 1;
                        } else {
                            col_idcs.push(rhs_col);
                            values.push(rhs.values[rhs_idx].clone());
                            rhs_idx += 1;
                        }
                    }
                    (true, false) => {
                        col_idcs.push(self.col_idcs[lhs_idx]);
                        values.push(self.values[lhs_idx].clone());
                        lhs_idx += 1;
                    }
                    (false, true) => {
                        col_idcs.push(rhs.col_idcs[rhs_idx]);
                        values.push(rhs.values[rhs_idx].clone());
                        rhs_idx += 1;
                    }
                    (false, false) => unreachable!(),
                }
            }

            row_ptrs.push(values.len());
        }

        SparseMatrix {
            values,
            row_ptrs,
            col_idcs,
            nrows: self.nrows,
            ncols: self.ncols,
            field: self.field.clone(),
        }
    }
}

impl<F: Ring> AddAssign<&SparseMatrix<F>> for SparseMatrix<F> {
    /// Add two sparse matrices in place
    fn add_assign(&mut self, rhs: &SparseMatrix<F>) {
        *self = &*self + rhs;
    }
}

impl<F: Ring> Sub<&SparseMatrix<F>> for &SparseMatrix<F> {
    type Output = SparseMatrix<F>;

    /// Add two sparse matrices
    fn sub(self, rhs: &SparseMatrix<F>) -> Self::Output {
        if self.nrows != rhs.nrows || self.ncols != rhs.ncols {
            panic!(
                "Cannot subtract sparse matrices of different dimensions: ({},{}) vs ({},{})",
                self.nrows, self.ncols, rhs.nrows, rhs.ncols
            );
        }
        if self.field != rhs.field {
            panic!(
                "Cannot subtract sparse matrices over different fields: {} vs {}",
                self.field, rhs.field
            );
        }

        if self.values.is_empty() {
            return rhs.clone().neg();
        }

        if rhs.values.is_empty() {
            return self.clone();
        }

        let mut values = Vec::new();
        let mut col_idcs = Vec::new();
        let mut row_ptrs = Vec::with_capacity((self.nrows + 1) as usize);
        row_ptrs.push(0);

        //iterate through both matrices simultaneously
        for row in 0..self.nrows as usize {
            let mut lhs_idx = self.row_ptrs[row];
            let mut rhs_idx = rhs.row_ptrs[row];
            let lhs_end = self.row_ptrs[row + 1];
            let rhs_end = rhs.row_ptrs[row + 1];

            //iterate through row entries
            while lhs_idx < lhs_end || rhs_idx < rhs_end {
                match (lhs_idx < lhs_end, rhs_idx < rhs_end) {
                    (true, true) => {
                        let lhs_col = self.col_idcs[lhs_idx];
                        let rhs_col = rhs.col_idcs[rhs_idx];

                        if lhs_col == rhs_col {
                            let sum = self.field.sub(&self.values[lhs_idx], &rhs.values[rhs_idx]);

                            if !self.field.is_zero(&sum) {
                                col_idcs.push(lhs_col);
                                values.push(sum)
                            }
                            lhs_idx += 1;
                            rhs_idx += 1;
                        } else if lhs_col < rhs_col {
                            col_idcs.push(lhs_col);
                            values.push(self.values[lhs_idx].clone());
                            lhs_idx += 1;
                        } else {
                            col_idcs.push(rhs_col);
                            values.push(self.field.neg(&rhs.values[rhs_idx]));
                            rhs_idx += 1;
                        }
                    }
                    (true, false) => {
                        col_idcs.push(self.col_idcs[lhs_idx]);
                        values.push(self.values[lhs_idx].clone());
                        lhs_idx += 1;
                    }
                    (false, true) => {
                        col_idcs.push(rhs.col_idcs[rhs_idx]);
                        values.push(self.field.neg(&rhs.values[rhs_idx]));
                        rhs_idx += 1;
                    }
                    (false, false) => unreachable!(),
                }
            }

            row_ptrs.push(values.len());
        }

        SparseMatrix {
            values,
            row_ptrs,
            col_idcs,
            nrows: self.nrows,
            ncols: self.ncols,
            field: self.field.clone(),
        }
    }
}

impl<F: Ring> SubAssign<&SparseMatrix<F>> for SparseMatrix<F> {
    /// Subtract two sparse matrices in place
    fn sub_assign(&mut self, rhs: &SparseMatrix<F>) {
        *self = &*self - rhs;
    }
}

impl<F: Ring> Mul<&SparseMatrix<F>> for &SparseMatrix<F> {
    type Output = SparseMatrix<F>;

    /// Multiply two sparse matrices.
    fn mul(self, rhs: &SparseMatrix<F>) -> Self::Output {
        if self.ncols != rhs.nrows {
            panic!(
                "Cannot multiply sparse matrices of non-matching shapes: ({},{}) * ({},{})",
                self.nrows, self.ncols, rhs.nrows, rhs.ncols
            );
        }
        if self.field != rhs.field {
            panic!(
                "Cannot multiply sparse matrices over different fields: {} vs {}",
                self.field, rhs.field
            );
        }

        let mut values = Vec::new();
        let mut col_idcs = Vec::new();
        let mut row_ptrs = Vec::with_capacity((self.nrows + 1) as usize);
        row_ptrs.push(0);

        // temporary dense vector to accumulate the output for the row we are working on
        let mut row_acc = vec![self.field.zero(); rhs.ncols as usize];

        //handle row by row in self/LHS
        for row in 0..self.nrows as usize {
            //reset row_acc
            for val in &mut row_acc {
                *val = self.field.zero();
            }

            //iterate over the row elements
            for lhs_idx in self.row_ptrs[row]..self.row_ptrs[row + 1] {
                let lhs_col = self.col_idcs[lhs_idx];
                let lhs_val = &self.values[lhs_idx];

                //iterate over the corresponding row in RHS
                for rhs_idx in rhs.row_ptrs[lhs_col as usize]..rhs.row_ptrs[(lhs_col + 1) as usize]
                {
                    let rhs_col = rhs.col_idcs[rhs_idx];
                    let rhs_val = &rhs.values[rhs_idx];

                    row_acc[rhs_col as usize] = self.field.add(
                        &row_acc[rhs_col as usize],
                        &self.field.mul(lhs_val, rhs_val),
                    );
                }
            }

            // push non-zero entries into a new row of the return matrix
            for (col, val) in row_acc.iter().enumerate() {
                if !self.field.is_zero(&val) {
                    col_idcs.push(col as u32);
                    values.push(val.clone());
                }
            }

            row_ptrs.push(values.len());
        }

        SparseMatrix {
            values,
            col_idcs,
            row_ptrs,
            nrows: self.nrows,
            ncols: rhs.ncols,
            field: self.field.clone(),
        }
    }
}

impl<F: Ring> MulAssign<&SparseMatrix<F>> for SparseMatrix<F> {
    /// Multiply two sparse matrices in place
    fn mul_assign(&mut self, rhs: &SparseMatrix<F>) {
        *self = &*self * rhs;
    }
}

/// An option for [`SparseRowReducer`] of how to handle the L matrix in the LU decomposition.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LuLMode {
    /// Construct the full L matrix in the process
    Full,
    /// Construct only the pattern of the L matrix (i.e. don't record the values in the CSR format)
    Pattern,
    /// Don't construct L at all
    None,
}

/// Scratch data used internally by the SparseForwardSolver algorithm
#[derive(Clone, Debug)]
struct Scratch<F: Field> {
    /// Stores the row that we are currently working on in dense form.
    dense_row: Vec<F::Element>,

	/// Remembers which columns we have touched in the current forward solving step.
    touched: Vec<bool>,
}

impl<F: Field> Scratch<F> {
    /// Constructs a new scratch with the given number of columns.
    pub fn new(ncols: u32, field: &F) -> Self {
        Self {
            dense_row: vec![field.zero(); ncols as usize],
            touched: vec![false; ncols as usize],
        }
    }
}

/// Computes the RREF form of a sparse matrix.
///
/// The struct implements two main algorithms:
/// * For a given matrix `A` it computes `L*U = A`, where `U` is upper triangular up to row permutations and `L` is lower triangular.
///   This step is also called forward solving or elimination and we implement a step-by-step/forward-looking algorithm.
///   For this step, one may either call `addRow()` to provide row-by-row, or submit a whole sparse matrix via `addMatrix()`
///   (or provide it during construction).
/// * For a sparse matrix `U` in upper triangular form (up to row permutations) it computes the RREF by performing a back substitution.
///
/// # Type parameters
/// * `F` - the field of the matrix entries
#[derive(Debug, Clone)]
pub struct SparseRowReducer<F: Field> {
    /// The U output matrix
    pub(crate) u: SparseMatrix<F>,

    /// The L output matrix
    l: SparseMatrix<F>,

    /// The pivot positions of U for each column.
    /// I.e. there is a pivot on column j and row pivots[j]. No pivot present if None.
    pivots: Vec<Option<u32>>,

    /// Whether to keep the L matrix, just record the pattern or don't record anything at all
    mode: LuLMode,

    /// Collection of some internal scratch data
    scratch: Scratch<F>,
}

impl<F: Field> SparseRowReducer<F> {
    /// Construct an empty reducer object. 
    ///
    /// New row(s) can be added with `add_row()` or `add_matrix()`.
    pub fn new(ncols: u32, field: F, mode: LuLMode) -> Self {
        Self {
            u: SparseMatrix::new(0, ncols, field.clone()),
            scratch: Scratch::new(ncols, &field),
            l: SparseMatrix::new(0, 0, field),
            pivots: vec![None; ncols as usize],
            mode: mode,
        }
    }

    /// Construct a new row reducer that immediately forward solves the given matrix.
    pub fn from_matrix(mat: &SparseMatrix<F>, mode: LuLMode) -> Self {
        let mut ret = Self::new(mat.ncols(), mat.field().clone(), mode);

        ret.add_matrix(mat);
        ret
    }

    /// Construct a new row reducer that immediately forward solves the given matrix.
    ///
    /// This version also performs back substitution on every new row during the forward solving.
    pub fn from_matrix_with_back_subs(mat: &SparseMatrix<F>, mode: LuLMode) -> Self {
        let mut ret = Self::new(mat.ncols(), mat.field().clone(), mode);

        ret.add_matrix_with_back_subs(mat);
        ret
    }
    
    /// Construct a new row reducer that immediately forward solves the given matrix and checks for consistency at each step.
    ///
    /// Checking for consistency means that we return None whenever a new row in `U` is all zero except the last entry.
    /// The idea is that we decompose the matrix `(A|b)` for solving the system `A * x = b`, which becomes unsolvable in this case.
    pub fn from_matrix_checked(mat: &SparseMatrix<F>, mode: LuLMode) -> Option<Self> {
        let mut ret = Self {
            u: SparseMatrix::new(0, mat.ncols(), mat.field().clone()),
            l: SparseMatrix::new(0, 0, mat.field().clone()),
            pivots: vec![None; mat.ncols() as usize],
            mode: mode,
            scratch: Scratch::new(mat.ncols(), mat.field()),
        };

        for row in mat.row_ptrs.windows(2) {
            if let Some(_) = ret.forward_solve_row(&mat.values[row[0]..row[1]], &mat.col_idcs[row[0]..row[1]]) {
                //check last, just added, row for inconsistency
                let start = ret.u.row_ptrs[(ret.u.nrows - 1) as usize];
                let end = ret.u.row_ptrs[ret.u.nrows as usize];
                if end - start == 1
                    && ret.u.col_idcs[start as usize] + 1 == ret.u.ncols
                    && ret.u.field.is_zero(&ret.u.values[start as usize])
                {
                    //row has only one entry and it's on the last column
                    return None;
                }
            }
        }

        Some(ret)
    }

    /// Construct a new row reducer that immediately forward solves the given matrix and checks for consistency at each step.
    ///
    /// Checking for consistency means that we return None whenever a new row in `U` is all zero except the last entry.
    /// The idea is that we decompose the matrix `(A|b)` for solving the system `A * x = b`, which becomes unsolvable in this case.
    /// This version also performs back substitution on every new row during the forward solving.
    pub fn from_matrix_checked_with_back_subs(mat: &SparseMatrix<F>, mode: LuLMode) -> Option<Self> {
        let mut ret = Self {
            u: SparseMatrix::new(0, mat.ncols(), mat.field().clone()),
            l: SparseMatrix::new(0, 0, mat.field().clone()),
            pivots: vec![None; mat.ncols() as usize],
            mode: mode,
            scratch: Scratch::new(mat.ncols(), mat.field()),
        };

        for row in mat.row_ptrs.windows(2) {
            if let Some(_) = ret.forward_solve_row_with_back_subs(&mat.values[row[0]..row[1]], &mat.col_idcs[row[0]..row[1]]) {
                //check last, just added, row for inconsistency
                let start = ret.u.row_ptrs[(ret.u.nrows - 1) as usize];
                let end = ret.u.row_ptrs[ret.u.nrows as usize];
                if end - start == 1
                    && ret.u.col_idcs[start as usize] + 1 == ret.u.ncols
                    && ret.u.field.is_zero(&ret.u.values[start as usize])
                {
                    //row has only one entry and it's on the last column
                    return None;
                }
            }
        }

        Some(ret)
    }

    /// Construct a new row reducer that immediately forward solves the given matrix and stops when a linearly dependent row is found.
    ///
    /// Checking for consistency means that we return None whenever a new row in `U` is all zero except the last entry.
    /// The idea is that we decompose the matrix `(A|b)` for solving the system `A * x = b`, which becomes unsolvable in this case.
    pub fn from_matrix_check_dependent(mat: &SparseMatrix<F>, mode: LuLMode) -> Option<Self> {
        let mut ret = Self {
            u: SparseMatrix::new(0, mat.ncols(), mat.field().clone()),
            l: SparseMatrix::new(0, 0, mat.field().clone()),
            pivots: vec![None; mat.ncols() as usize],
            mode: mode,
            scratch: Scratch::new(mat.ncols(), mat.field()),
        };

        for row in mat.row_ptrs.windows(2) {
            if ret
                .forward_solve_row(
                    &mat.values[row[0]..row[1]],
                    &mat.col_idcs[row[0]..row[1]],
                )
                .is_none()
            {
                return None;
            }
        }

        Some(ret)
    }

    /// Construct a new row reducer that immediately forward solves the given matrix and stops when a linearly dependent row is found.
    ///
    /// Checking for consistency means that we return None whenever a new row in `U` is all zero except the last entry.
    /// The idea is that we decompose the matrix `(A|b)` for solving the system `A * x = b`, which becomes unsolvable in this case.
    /// This version also performs back substitution on every new row during the forward solving.
    pub fn from_matrix_check_dependent_with_back_subs(mat: &SparseMatrix<F>, mode: LuLMode) -> Option<Self> {
        let mut ret = Self {
            u: SparseMatrix::new(0, mat.ncols(), mat.field().clone()),
            l: SparseMatrix::new(0, 0, mat.field().clone()),
            pivots: vec![None; mat.ncols() as usize],
            mode: mode,
            scratch: Scratch::new(mat.ncols(), mat.field()),
        };

        for row in mat.row_ptrs.windows(2) {
            if ret
                .forward_solve_row_with_back_subs(
                    &mat.values[row[0]..row[1]],
                    &mat.col_idcs[row[0]..row[1]],
                )
                .is_none()
            {
                return None;
            }
        }

        Some(ret)
    }

    /// Construct a reducer from an upper triangular matrix (up to row permutations).
    ///
    /// One may still add more rows to this matrix and perform more forward solving.
    /// Note that the correct pivots need to be provided too.
    /// # Arguments
    /// * `u` -- A sparse matrix in upper triangular form (up to row permutations).
    /// * `pivots` -- The pivots of `u`: I.e. there is a pivot on column j and row pivots[j] or no pivot present if None.
    pub fn from_upper_triangular_matrix(u: SparseMatrix<F>, pivots: Vec<Option<u32>>) -> Self {
        Self {
            l: SparseMatrix::new(0, 0, u.field().clone()),
            scratch: Scratch::new(u.ncols(), u.field()),
            u,
            pivots,
            mode: LuLMode::None,
        }
    }

    /// Return the U matrix.
    pub fn u(&self) -> &SparseMatrix<F> {
        &self.u
    }

    /// Return the L matrix.
    pub fn l(&self) -> &SparseMatrix<F> {
        &self.l
    }

    /// Return the pivot positions of U for each column.
    /// I.e. there is a pivot on column `j` and row `pivots()[j]` and no pivot present if `None`.
    pub fn pivots(&self) -> &Vec<Option<u32>> {
        &self.pivots
    }

    /// Adds a new row to the system and processes it in the next forward solving step.
    ///
    /// # Parameters
    /// * `values` - the values of the non-zero entries of the row
    /// * `col_idcs` - the column indices of the non-zero entries of the row
    ///
    /// # Return
    /// If the new row is linearly independent from the rows of the current system it returns the pivot column index
    /// of the new row after the row reduction step, otherwise None.
    /// Note linearly dependent rows are not added to the system.
    pub fn add_row(&mut self, values: &[F::Element], col_idcs: &[u32]) -> Option<u32> {
        assert_eq!(values.len(), col_idcs.len());

        //run next forward solving step
        self.forward_solve_row(values, col_idcs)
    }

    /// Adds a new row to the system and processes it in the next forward solving step.
    ///
    /// This version also performs as much back substitution as possible on the new row.
    ///
    /// # Parameters
    /// * `values` - the values of the non-zero entries of the row
    /// * `col_idcs` - the column indices of the non-zero entries of the row
    ///
    /// # Return
    /// If the new row is linearly independent from the rows of the current system it returns the pivot column index
    /// of the new row after the row reduction step, otherwise None.
    /// Note linearly dependent rows are not added to the system.
    pub fn add_row_with_back_subs(&mut self, values: &[F::Element], col_idcs: &[u32]) -> Option<u32> {
        assert_eq!(values.len(), col_idcs.len());

        //run next forward solving step
        self.forward_solve_row_with_back_subs(values, col_idcs)
    }

    /// Adds all the rows from the matrix to the system and processes them in the next forward solving step.
    ///
    /// # Parameters
    /// * `mat` - The matrix whose row are to be added to the system through forward solving.
    pub fn add_matrix(&mut self, mat: &SparseMatrix<F>) {
        assert_eq!(mat.ncols(), self.u.ncols());

        for row in mat.row_ptrs.windows(2) {
            self.forward_solve_row(&mat.values[row[0]..row[1]], &mat.col_idcs[row[0]..row[1]]);
        }
    }

    /// Adds all the rows from the matrix to the system and processes them in the next forward solving step.
    ///
    /// This version also performs back substitution on every new row during the forward solving.
    ///
    /// # Parameters
    /// * `mat` - The matrix whose row are to be added to the system through forward solving.
    pub fn add_matrix_with_back_subs(&mut self, mat: &SparseMatrix<F>) {
        assert_eq!(mat.ncols(), self.u.ncols());

        for row in mat.row_ptrs.windows(2) {
            self.forward_solve_row_with_back_subs(&mat.values[row[0]..row[1]], &mat.col_idcs[row[0]..row[1]]);
        }
    }

    /// Adds empty columns to the U matrix and updates the pivots accordingly.
    ///
    /// * `col_pos` - Ordered(!) positions where the new columns should be inserted. Each entry must NOT account for previously inserted columns.
    pub fn add_cols(&mut self, col_pos: &Vec<u32>) -> () {
        //update U
        self.u.add_cols(col_pos);
        
        //update pivots
        let mut new_pivots = Vec::with_capacity(self.pivots.len() + col_pos.len());
        let mut pivots_idx: usize = 0;
        let mut col_pos_idx: usize = 0;

        while pivots_idx < self.pivots.len() || col_pos_idx < col_pos.len() {
            if col_pos_idx < col_pos.len() && pivots_idx == col_pos[col_pos_idx] as usize {
                new_pivots.push(None);
                col_pos_idx += 1;
            } else {
                assert!(pivots_idx < self.pivots.len());
                new_pivots.push(self.pivots[pivots_idx]);
                pivots_idx += 1;
            }
        }
        self.pivots = new_pivots;

        //update scratch
        self.scratch
            .dense_row
            .resize(self.u.ncols() as usize, self.u.field().zero());
        self.scratch.touched.resize(self.u.ncols() as usize, false);
    }

    /// Applies backsubstitution to the U matrix to bring it into reversed RREF form (i.e. U will be in lower right triangular form).
    ///
    /// We do not keep track of the L matrix, so LuLMode will be set to `None` and the L matrix will be emptied.
    pub fn back_substitute(&mut self) -> () {
        //clear L
        self.clear_l();

        let ncols = self.u.ncols;
        let mut new_u = SparseMatrix::new(0, ncols, self.u.field.clone());
        let mut new_pivots = vec![None; ncols as usize];

        let rows: Vec<_> = self.pivots.iter().rev()
            .filter_map(|&r| r)
            .collect();


        for row in rows {
            let new_row = new_u.nrows;
            let pivot_col = self.back_substitute_row(
                row,
                &mut new_u,
                &new_pivots,
            );
            new_pivots[pivot_col] = Some(new_row);
        }

        self.u = new_u;
        self.pivots = new_pivots;
    }

    /// Clears the L matrix (and sets LuLMode to None).
    fn clear_l(&mut self) {
        match self.mode {
            LuLMode::Full => {
                self.l.values.clear();
                self.l.col_idcs.clear();
                self.l.row_ptrs.clear();
                self.l.row_ptrs.push(0);
                self.l.nrows = 0;
                self.l.ncols = 0;
                self.mode = LuLMode::None;
            },
            LuLMode::Pattern => {
                self.l.col_idcs.clear();
                self.l.row_ptrs.clear();
                self.l.row_ptrs.push(0);
                self.l.nrows = 0;
                self.l.ncols = 0;
                self.mode = LuLMode::None;
            },
            LuLMode::None => ()
        }
    }

    /// Apply forward solving step to the given row.
    ///
    /// # Return
    /// The pivot column of the new row in U if it was a linearly independent row.
    fn forward_solve_row(&mut self, values: &[F::Element], col_idcs: &[u32]) -> Option<u32> {
        if self.u.nrows == self.u.ncols || col_idcs.is_empty() {
            //full rank reached or empty row
            return None;
        }

        //check whether we can directly copy the row without doing anything
        let mut pivot_col = col_idcs[0] as usize;
        if self.pivots[pivot_col].is_none() {
            //yes, we can, no pivot present on the leading column
            self.pivots[pivot_col] = Some(self.u.nrows());

            //copy the whole row into u (and normalize)
            let leading_coeff = &values[0];
            let leading_coeff_inv = self.u.field.inv(&leading_coeff);
            self.u.nrows += 1;
            self.u.col_idcs.extend_from_slice(&col_idcs);
            if self.u.field.is_one(&leading_coeff_inv) {
                self.u.values.extend_from_slice(&values);
            } else {
                self.u.values.extend(
                    values
                        .iter()
                        .map(|val| self.u.field.mul(val, &leading_coeff_inv)),
                );
            }
            self.u.row_ptrs.push(self.u.values.len());

            //also compute L if wanted
            match self.mode {
                LuLMode::Full => {
                    //put a 1 on the diagonal
                    self.l.col_idcs.push(self.u.nrows - 1);
                    self.l.values.push(leading_coeff.clone());
                    //finish the row
                    self.l.row_ptrs.push(self.l.col_idcs.len());
                    //update nrows/ncols
                    self.l.nrows += 1;
                    self.l.ncols += 1;
                }
                LuLMode::Pattern => {
                    //put an entry on the diagonal
                    self.l.col_idcs.push(self.u.nrows - 1);
                    //finish the row
                    self.l.row_ptrs.push(self.l.col_idcs.len());
                    //update nrows/ncols
                    self.l.nrows += 1;
                    self.l.ncols += 1;
                }
                LuLMode::None => (), //nothing to be done
            }

            return Some(pivot_col as u32);
        }

        //prepare the scratch (copy sparse row into dense representation)
        for (val, col_idx) in values.iter().zip(col_idcs.iter()) {
            self.scratch.dense_row[*col_idx as usize] = val.clone();
            self.scratch.touched[*col_idx as usize] = true;
        }

        let n = self.u.ncols as usize;
        
        //gaussian reduction until we find a pivot (or dense_row is zero)
        while pivot_col < n {
            if self.scratch.touched[pivot_col] {
                if self.u.field.is_zero(&self.scratch.dense_row[pivot_col]) {
                    //accidental zero, remove from touched
                    self.scratch.touched[pivot_col] = false;
                    //step
                    pivot_col += 1;
                    continue;
                }

                if let Some(pivot_row) = self.pivots[pivot_col] {
                    //subtract the pivot row
                    let start = self.u.row_ptrs[pivot_row as usize];
                    let end = self.u.row_ptrs[pivot_row as usize + 1];
                    let pivot_val = self.scratch.dense_row[pivot_col].clone();
                    Self::scatter_with_touched(
                        &mut self.scratch.dense_row,
                        &self.u.field.neg(pivot_val.clone()),
                        &self.u.values[start..end],
                        &self.u.col_idcs[start..end],
                        &self.u.field,
                        &mut self.scratch.touched
                    );

                    debug_assert!(self.u.field.is_zero(&self.scratch.dense_row[pivot_col]));
                    self.scratch.touched[pivot_col] = false;

                    //also update L if wanted
                    match self.mode {
                        LuLMode::Full => {
                            self.l.col_idcs.push(pivot_row);
                            self.l.values.push(pivot_val);
                        }
                        LuLMode::Pattern => {
                            self.l.col_idcs.push(pivot_row);
                        }
                        LuLMode::None => (), //nothing to be done
                    }
                } else {
                    //found a pivot, we are done
                    break;
                }
            }

            //step
            pivot_col += 1;
        }

        if pivot_col < n {
            self.pivots[pivot_col] = Some(self.u.nrows());

            let leading_coeff = &self.scratch.dense_row[pivot_col];
            match self.mode {
                LuLMode::Full => {
                    self.l.col_idcs.push(self.u.nrows);
                    self.l.values.push(leading_coeff.clone());
                    //finish the row
                    self.l.row_ptrs.push(self.l.col_idcs.len());
                    self.l.nrows += 1;
                    self.l.ncols += 1;
                }
                LuLMode::Pattern => {
                    self.l.col_idcs.push(self.u.nrows);
                    //finish the row
                    self.l.row_ptrs.push(self.l.col_idcs.len());
                    self.l.nrows += 1;
                    self.l.ncols += 1;
                }
                LuLMode::None => (), //notheng to be done
            }

            self.u.nrows += 1;
            
            //normalize and copy the dense row into U
            let leading_coeff_inv = self.u.field.inv(&leading_coeff);
            for i in pivot_col..n {
                if self.scratch.touched[i] {
                    //need to check for accidental zeroes
                    let val = &mut self.scratch.dense_row[i];
                    if !self.u.field.is_zero(val) {
	                    self.u.col_idcs.push(i as u32);
    	                if self.u.field.is_one(&leading_coeff_inv) {
        	                self.u.values.push(val.clone());
            	        } else {
                	        self.u.values.push(self.u.field.mul(&leading_coeff_inv, val));
                    	}
                        //clean up scratch                      
                    	*val = self.u.field.zero();
                    }

                    //clean up scratch
                    self.scratch.touched[i] = false;
                }
            }

            //finish the row
            self.u.row_ptrs.push(self.u.col_idcs.len());

            return Some(pivot_col as u32);
        }

        //else: need to finish L
        match self.mode {
            LuLMode::Full => {
                //finish the row
                self.l.row_ptrs.push(self.l.col_idcs.len());
                self.l.nrows += 1;
            }
            LuLMode::Pattern => {
                //finish the row
                self.l.row_ptrs.push(self.l.col_idcs.len());
                self.l.nrows += 1;
            }
            LuLMode::None => (), //nothing to be done
        }

        //row is linearly dependent
        None
    }
    
    /// Apply forward solving step to the given row.
    ///
    /// This version also performs as much back substitution as possible on the current row.
    ///
    /// # Return
    /// The pivot column of the new row in U if it was a linearly independent row.
    fn forward_solve_row_with_back_subs(&mut self, values: &[F::Element], col_idcs: &[u32]) -> Option<u32> {
        if self.u.nrows == self.u.ncols || col_idcs.is_empty() {
            //full rank reached or empty row
            return None;
        }

        //check whether we can directly copy the row without doing anything
        let mut pivot_col = col_idcs[0] as usize;
        let mut direct = true;
        //need to check whether the row contains ANY entry on a pivot column of previous rows
        for col_idx in col_idcs {
            if self.pivots[*col_idx as usize].is_some() {
                direct = false;
                break;
            }
        }
        if direct {
            //yes, we can directly copy it without doing anything
            self.pivots[pivot_col] = Some(self.u.nrows());

            //copy the whole row into u (and normalize)
            let leading_coeff = &values[0];
            let leading_coeff_inv = self.u.field.inv(&leading_coeff);
            self.u.nrows += 1;
            self.u.col_idcs.extend_from_slice(&col_idcs);
            if self.u.field.is_one(&leading_coeff_inv) {
                self.u.values.extend_from_slice(&values);
            } else {
                self.u.values.extend(
                    values
                        .iter()
                        .map(|val| self.u.field.mul(val, &leading_coeff_inv)),
                );
            }
            self.u.row_ptrs.push(self.u.values.len());

            //also compute L if wanted
            match self.mode {
                LuLMode::Full => {
                    //put a 1 on the diagonal
                    self.l.col_idcs.push(self.u.nrows - 1);
                    self.l.values.push(leading_coeff.clone());
                    //finish the row
                    self.l.row_ptrs.push(self.l.col_idcs.len());
                    //update nrows/ncols
                    self.l.nrows += 1;
                    self.l.ncols += 1;
                }
                LuLMode::Pattern => {
                    //put an entry on the diagonal
                    self.l.col_idcs.push(self.u.nrows - 1);
                    //finish the row
                    self.l.row_ptrs.push(self.l.col_idcs.len());
                    //update nrows/ncols
                    self.l.nrows += 1;
                    self.l.ncols += 1;
                }
                LuLMode::None => (), //nothing to be done
            }

            return Some(pivot_col as u32);
        }

        //prepare the scratch (copy sparse row into dense representation)
        for (val, col_idx) in values.iter().zip(col_idcs.iter()) {
            self.scratch.dense_row[*col_idx as usize] = val.clone();
            self.scratch.touched[*col_idx as usize] = true;
        }

        let n = self.u.ncols as usize;
        
        //gaussian reduction until we find a pivot (or dense_row is zero)
        pivot_col = n;
        for col in col_idcs[0] as usize..n {
            if self.scratch.touched[col] {
                if self.u.field.is_zero(&self.scratch.dense_row[col]) {
                    //accidental zero, remove from touched
                    self.scratch.touched[col] = false;
                    continue;
                }

                if let Some(pivot_row) = self.pivots[col] {
                    //subtract the pivot row
                    let start = self.u.row_ptrs[pivot_row as usize];
                    let end = self.u.row_ptrs[pivot_row as usize + 1];
                    let pivot_val = self.scratch.dense_row[col].clone();
                    Self::scatter_with_touched(
                        &mut self.scratch.dense_row,
                        &self.u.field.neg(pivot_val.clone()),
                        &self.u.values[start..end],
                        &self.u.col_idcs[start..end],
                        &self.u.field,
                        &mut self.scratch.touched
                    );

                    debug_assert!(self.u.field.is_zero(&self.scratch.dense_row[col]));
                    self.scratch.touched[col] = false;

                    //also update L if wanted
                    match self.mode {
                        LuLMode::Full => {
                            self.l.col_idcs.push(pivot_row);
                            self.l.values.push(pivot_val);
                        }
                        LuLMode::Pattern => {
                            self.l.col_idcs.push(pivot_row);
                        }
                        LuLMode::None => (), //nothing to be done
                    }
                } else if pivot_col == n{
                    //found a pivot, but we still continue
                    pivot_col = col;
                }
            }
        }

        if pivot_col < n {
            self.pivots[pivot_col] = Some(self.u.nrows());

            let leading_coeff = &self.scratch.dense_row[pivot_col];
            match self.mode {
                LuLMode::Full => {
                    self.l.col_idcs.push(self.u.nrows);
                    self.l.values.push(leading_coeff.clone());
                    //finish the row
                    self.l.row_ptrs.push(self.l.col_idcs.len());
                    self.l.nrows += 1;
                    self.l.ncols += 1;
                }
                LuLMode::Pattern => {
                    self.l.col_idcs.push(self.u.nrows);
                    //finish the row
                    self.l.row_ptrs.push(self.l.col_idcs.len());
                    self.l.nrows += 1;
                    self.l.ncols += 1;
                }
                LuLMode::None => (), //notheng to be done
            }

            self.u.nrows += 1;
            
            //normalize and copy the dense row into U
            let leading_coeff_inv = self.u.field.inv(&leading_coeff);
            for i in pivot_col..n {
                if self.scratch.touched[i] {
                    //need to check for accidental zeroes
                    let val = &mut self.scratch.dense_row[i];
                    if !self.u.field.is_zero(val) {
	                    self.u.col_idcs.push(i as u32);
    	                if self.u.field.is_one(&leading_coeff_inv) {
        	                self.u.values.push(val.clone());
            	        } else {
                	        self.u.values.push(self.u.field.mul(&leading_coeff_inv, val));
                    	}
                        //clean up scratch                      
                    	*val = self.u.field.zero();
                    }

                    //clean up scratch
                    self.scratch.touched[i] = false;
                }
            }

            //finish the row
            self.u.row_ptrs.push(self.u.col_idcs.len());

            return Some(pivot_col as u32);
        }

        //else: need to finish L
        match self.mode {
            LuLMode::Full => {
                //finish the row
                self.l.row_ptrs.push(self.l.col_idcs.len());
                self.l.nrows += 1;
            }
            LuLMode::Pattern => {
                //finish the row
                self.l.row_ptrs.push(self.l.col_idcs.len());
                self.l.nrows += 1;
            }
            LuLMode::None => (), //nothing to be done
        }

        //row is linearly dependent
        None
    }

    /// Apply a back substitution step to the given row.
    ///
    /// Output will be written into new_u.
    /// Helper of back_substitute().
    fn back_substitute_row(
        &mut self,
        row: u32,
        new_u: &mut SparseMatrix<F>,
        new_pivots: &[Option<u32>],
    ) -> usize {
        let n = self.scratch.dense_row.len();
        let row_begin = self.u.row_ptrs[row as usize];
        let row_end = self.u.row_ptrs[row as usize + 1];

        // scatter input row into dense scratch
        for px in row_begin..row_end {
            let j = self.u.col_idcs[px] as usize;
            self.scratch.dense_row[j] = self.u.values[px].clone();
            self.scratch.touched[j] = true;
        }

        // subtract already back-substituted pivot rows
        for px in row_begin..row_end {
            let j = self.u.col_idcs[px] as usize;
            if !self.scratch.touched[j] {
                continue;
            }
            if self.u.field.is_zero(&self.scratch.dense_row[j]) {
                self.scratch.touched[j] = false;
                continue;
            }
            let new_pivot_row = match new_pivots[j] {
                Some(r) => r,
                None => continue,
            };
            let coeff = self.scratch.dense_row[j].clone();
            let neg_coeff = self.u.field.neg(coeff);

            let qx_begin = new_u.row_ptrs[new_pivot_row as usize];
            let qx_end = new_u.row_ptrs[new_pivot_row as usize + 1];
            Self::scatter_with_touched(
                &mut self.scratch.dense_row,
                &neg_coeff,
                &new_u.values[qx_begin..qx_end],
                &new_u.col_idcs[qx_begin..qx_end],
                &self.u.field,
                &mut self.scratch.touched,
            );
            // the entry on the pivot col better be zero!
            self.scratch.touched[j] = false;
            debug_assert!(self.u.field.is_zero(&self.scratch.dense_row[j]));
        }

        let pivot_col = self.u.col_idcs[row_begin] as usize;
        debug_assert!(self.scratch.touched[pivot_col]);
        debug_assert!(self.u.field.is_one(&self.scratch.dense_row[pivot_col]));

        // gather into new_u
        new_u.nrows += 1;
        for i in pivot_col..n {
            if self.scratch.touched[i] {
                //need to check for accidental zeroes
                let val = &mut self.scratch.dense_row[i];
                if !self.u.field.is_zero(val) {
	                new_u.col_idcs.push(i as u32);
    	            new_u.values.push(val.clone());

                    //clean up scratch
            	    *val = self.u.field.zero();
                }
                //clean up scratch
                self.scratch.touched[i] = false;
            }
        }
        new_u.row_ptrs.push(new_u.col_idcs.len());

        pivot_col
    }

    /// Compute x = x + beta * sparse_row, writing it directly into x.
    ///
    /// # Arguments
    /// * `x` - the vector x into which we scatter.
    /// * `beta` - the scalar which we multiply into the sparse row
    /// * `values` - he non-zero values of the sparse row on the RHS of the equation.
    /// * `col_idcs` - he column indices of the non-zero values of the sparse row on the RHS of the equation.
    /// * `field` - the field to be used for the arithmetics.
    #[allow(dead_code)]
    fn scatter(
        x: &mut Vec<F::Element>,
        beta: &F::Element,
        values: &[F::Element],
        col_idcs: &[u32],
        field: &F,
    ) {
        for (val, col) in values.iter().zip(col_idcs.iter()) {
            x[*col as usize] = field.add(&field.mul(beta, &val), &x[*col as usize]);
        }
    }

    /// Compute x = x + beta * sparse_row, writing it directly into x.
    ///
    /// This version also records the touched entries of x in `touched`.
    /// # Arguments
    /// * `x` - the vector x into which we scatter.
    /// * `beta` - the scalar which we multiply into the sparse row
    /// * `values` - he non-zero values of the sparse row on the RHS of the equation.
    /// * `col_idcs` - he column indices of the non-zero values of the sparse row on the RHS of the equation.
    /// * `field` - the field to be used for the arithmetics.
    fn scatter_with_touched(
        x: &mut Vec<F::Element>,
        beta: &F::Element,
        values: &[F::Element],
        col_idcs: &[u32],
        field: &F,
        touched: &mut Vec<bool>,
    ) {
        for (val, col) in values.iter().zip(col_idcs.iter()) {
            x[*col as usize] = field.add(&field.mul(beta, &val), &x[*col as usize]);
            touched[*col as usize] = true;
        }
    }

    /// Computes the level of each row in U.
    /// The level of row i is defined as 1+max(level[j]), where the max is over all rows j with U[i,j] != 0.
    /// U must be in upper triangular form (up to row permutations) and the pivots must have been correctly computed.
    fn compute_levels(&self) -> Vec<u32> {
        let mut ret = vec![0; self.u.nrows as usize];

        for pivot in self.pivots.iter().rev() {
            if let Some(row) = pivot {
                let mut max: u32 = 0;
                for j in self.u.row_ptrs[*row as usize]..self.u.row_ptrs[(*row as usize) + 1] {
                    let col_idx = self.u.col_idcs[j];
                    let pivot_row = self.pivots[col_idx as usize];

                    if let Some(row2) = pivot_row {
                        max = max.max(ret[row2 as usize] + 1);
                    }
                }
                ret[*row as usize] = max;
            }
        }

        ret
    }
}

impl<F: Field + Sync + Send> SparseRowReducer<F>
where
    F::Element: Sync + Send,
{
    /// Applies backsubstitution to the U matrix to bring it into RREF form (up to row permutation).
    ///
    /// We do not keep track of the L matrix, so LuLMode will be set to `None` and the L matrix will be emptied.
    /// This version employs a parallel algorithm, which though in total does more work than the serial version.
    /// Note that the output of this version might not be the same as of back_substitution() as rows might be permuted.
    pub fn back_substitute_parallel(&mut self) {
        self.clear_l();

        if self.u.nrows == 0 {
            return;
        }

        // compute level sets
        let levels = self.compute_levels();
        let max_level = *levels.iter().max().unwrap() as usize;
        let mut level_sets = vec![Vec::<u32>::new(); max_level + 1];
        for row in 0..self.u.nrows {
            level_sets[levels[row as usize] as usize].push(row);
        }
        drop(levels);

        let n_cols = self.u.ncols;
        let mut new_u = SparseMatrix::new(0, n_cols, self.u.field.clone());
        let mut new_pivots = vec![None; n_cols as usize];

        // pool of scratch objects to reuse across levels
        let scratch_pool: Mutex<Vec<Scratch<F>>> = Mutex::new(vec![]);

        // process levels from highest (rightmost pivots) to lowest
        for level in level_sets.iter() {
            if level.is_empty() {
                continue;
            }

            // parallel: each thread gets a local mat and scratch, processes its rows
            let results: Vec<(SparseMatrix<F>, Vec<(u32, u32)>)> = level
                .par_iter()
                .fold(
                    || {
                        let scratch = scratch_pool
                            .lock()
                            .unwrap()
                            .pop()
                            .unwrap_or_else(|| Scratch::new(n_cols, &self.u.field));
                        (
                            SparseMatrix::new(0, n_cols, self.u.field.clone()),
                            Vec::<(u32, u32)>::new(), // (pivot_col, local_row)
                            scratch,
                        )
                    },
                    |(mut local_mat, mut local_pivots, mut scratch), &row| {
                        let local_row = local_mat.nrows;
                        let pivot_col = self.back_substitute_row_with_scratch(
                            row,
                            &new_u,
                            &new_pivots,
                            &mut local_mat,
                            &mut scratch,
                        );
                        local_pivots.push((pivot_col as u32, local_row));
                        (local_mat, local_pivots, scratch)
                    },
                )
                .map(|(mat, pivots, scratch)| {
                    // return scratch to pool
                    scratch_pool.lock().unwrap().push(scratch);
                    (mat, pivots)
                })
                .collect();

            // serial: fuse local results into new_u and new_pivots
            let mut n_new_entries = 0;
            for (mat, _) in &results {
                n_new_entries += mat.values.len();
            }
            new_u.values.reserve(n_new_entries);
            new_u.col_idcs.reserve(n_new_entries);

            for (local_mat, local_pivots) in results {
                let n_entries_before = new_u.values.len();
                let n_rows_before = new_u.nrows;

                new_u.values.extend(local_mat.values);
                new_u.col_idcs.extend(local_mat.col_idcs);
                new_u.nrows += local_mat.nrows;
                new_u.row_ptrs.reserve(new_u.nrows as usize + 1);
                // skip first row_ptr (always 0)
                for i in 1..local_mat.row_ptrs.len() {
                    new_u.row_ptrs.push(local_mat.row_ptrs[i] + n_entries_before);
                }
                for (pivot_col, local_row) in local_pivots {
                    debug_assert!(new_pivots[pivot_col as usize].is_none());
                    new_pivots[pivot_col as usize] = Some(local_row + n_rows_before);
                }
            }
        }

        self.u = new_u;
        self.pivots = new_pivots;
    }

    /// Same as back_substitute_row but takes an explicit scratch instead of using self's scratch and takes a separate ouput matrix.
    /// This is needed for the parallel version where each thread has its own scratch.
    fn back_substitute_row_with_scratch(
        &self,
        row: u32,
        new_u: &SparseMatrix<F>,
        new_pivots: &[Option<u32>],
        out_mat: &mut SparseMatrix<F>,
        scratch: &mut Scratch<F>,
    ) -> usize {
        let n = scratch.dense_row.len();
        let row_begin = self.u.row_ptrs[row as usize];
        let row_end = self.u.row_ptrs[row as usize + 1];

        // scatter input row into dense scratch
        for px in row_begin..row_end {
            let j = self.u.col_idcs[px] as usize;
            scratch.dense_row[j] = self.u.values[px].clone();
            scratch.touched[j] = true;
        }

        // subtract already back-substituted pivot rows
        for px in row_begin..row_end {
            let j = self.u.col_idcs[px] as usize;
            if !scratch.touched[j] {
                continue;
            }
            if self.u.field.is_zero(&scratch.dense_row[j]) {
                scratch.touched[j] = false;
                continue;
            }
            let new_pivot_row = match new_pivots[j] {
                Some(r) => r,
                None => continue,
            };
            let coeff = scratch.dense_row[j].clone();
            let neg_coeff = self.u.field.neg(coeff);

            let qx_begin = new_u.row_ptrs[new_pivot_row as usize];
            let qx_end = new_u.row_ptrs[new_pivot_row as usize + 1];
            Self::scatter_with_touched(
                &mut scratch.dense_row,
                &neg_coeff,
                &new_u.values[qx_begin..qx_end],
                &new_u.col_idcs[qx_begin..qx_end],
                &self.u.field,
                &mut scratch.touched,
            );
            scratch.touched[j] = false;
            debug_assert!(self.u.field.is_zero(&scratch.dense_row[j]));
        }

        let pivot_col = self.u.col_idcs[row_begin] as usize;
        debug_assert!(scratch.touched[pivot_col]);
        debug_assert!(self.u.field.is_one(&scratch.dense_row[pivot_col]));

        // gather into out_mat
        out_mat.nrows += 1;
        for i in pivot_col..n {
            if scratch.touched[i] {
                //need to check for accidental zeroes
                let val = &mut scratch.dense_row[i];
                if !self.u.field.is_zero(val) {
	                out_mat.col_idcs.push(i as u32);
    	            out_mat.values.push(val.clone());
        	        //clean up scratch
            	    *val = self.u.field.zero();
                }
                //clean up scratch
                scratch.touched[i] = false;
            }
        }
        out_mat.row_ptrs.push(out_mat.col_idcs.len());

        pivot_col
    }
}

#[cfg(test)]
mod tests {
    use crate::domains::{Set, rational::Q};

    use crate::tensors::{
        matrix::Matrix,
        sparse::{SparseRowReducer, LuLMode, SparseMatrix, SparseVector}
    };


    #[test]
    fn dense_to_sparse() {
        let a = Matrix::from_linear(
            vec![
                3.into(),
                0.into(),
                0.into(),
                15.into(),
                0.into(),
                4.into(),
                0.into(),
                7.into(),
                0.into(),
            ],
            3,
            3,
            Q,
        )
            .unwrap();

        let b = a.to_sparse().to_dense();
        assert_eq!(a, b);
    }

    #[test]
    #[should_panic(expected = "Cannot add sparse matrices of different dimensions")]
    fn add_sparse_matrices_with_different_column_counts_panics() {
        let lhs = SparseMatrix::from_triplets(2, 2, vec![(0, 0, 1.into())], Q);
        let rhs = SparseMatrix::from_triplets(2, 3, vec![(0, 2, 1.into())], Q);

        let _ = &lhs + &rhs;
    }

    #[test]
    #[should_panic(expected = "Cannot subtract sparse matrices of different dimensions")]
    fn subtract_sparse_matrices_with_different_column_counts_panics() {
        let lhs = SparseMatrix::from_triplets(2, 2, vec![(0, 0, 1.into())], Q);
        let rhs = SparseMatrix::from_triplets(2, 3, vec![(0, 2, 1.into())], Q);

        let _ = &lhs - &rhs;
    }

    #[test]
    fn row_by_row_rref() {
        let mut sparse_row_reducer = SparseRowReducer::new(6, Q, LuLMode::Full);
        let mut sparse_row_reducer_with_back_subs = SparseRowReducer::new(6, Q, LuLMode::Full);
        let mut mat = SparseMatrix::new(0, 6, Q);

        //mat=
        //[
        //  [0,3,7,0,0,13],
        //  [1,0,0,3,-7,11],
        //  [-2,0,0,14,-27,0],
        //  [0,23,18,6,0,0]
        //]

        let values: Vec<<Q as Set>::Element> = vec![3.into(), 7.into(), 13.into()];
        let col_idcs: Vec<u32> = vec![1,2,5];
        sparse_row_reducer.add_row(&values, &col_idcs);
        sparse_row_reducer_with_back_subs.add_row_with_back_subs(&values, &col_idcs);
        mat.add_row(values, col_idcs);

        let values: Vec<<Q as Set>::Element> = vec![1.into(), 3.into(), (-7).into(), 11.into()];
        let col_idcs: Vec<u32> = vec![0,3,4,5];
        sparse_row_reducer.add_row(&values, &col_idcs);
        sparse_row_reducer_with_back_subs.add_row_with_back_subs(&values, &col_idcs);
        mat.add_row(values, col_idcs);

        let values: Vec<<Q as Set>::Element> = vec![(-2).into(), 14.into(), (-27).into()];
        let col_idcs: Vec<u32> = vec![0,3,4];
        sparse_row_reducer.add_row(&values, &col_idcs);
        sparse_row_reducer_with_back_subs.add_row_with_back_subs(&values, &col_idcs);
        mat.add_row(values, col_idcs);

        let values: Vec<<Q as Set>::Element> = vec![23.into(), 18.into(), 6.into()];
        let col_idcs: Vec<u32> = vec![1,2,3];
        sparse_row_reducer.add_row(&values, &col_idcs);
        sparse_row_reducer_with_back_subs.add_row_with_back_subs(&values, &col_idcs);
        mat.add_row(values, col_idcs);

        //check L.U == A (also checking multiplication and subtraction)
        assert_eq!(&(sparse_row_reducer.l() * sparse_row_reducer.u()), &mat);
        assert_eq!(&(sparse_row_reducer_with_back_subs.l() * sparse_row_reducer_with_back_subs.u()), &mat);
        assert_eq!(
            &(sparse_row_reducer.l() * sparse_row_reducer.u()) - &mat,
            SparseMatrix::new(mat.nrows(), mat.ncols(), Q)
        );
        
        //check Us explicitly
        assert_eq!(
            sparse_row_reducer.u().fmt_mma(),
            "{{{1,2}->1,{1,3}->7/3,{1,6}->13/3,{2,1}->1,{2,4}->3,{2,5}->-7,{2,6}->11,{3,4}->1,{3,5}->-41/20,{3,6}->11/10,{4,3}->1,{4,4}->-18/107,{4,6}->299/107},{4,6}}"
        );

        assert_eq!(
            sparse_row_reducer_with_back_subs.u().fmt_mma(),
            "{{{1,2}->1,{1,3}->7/3,{1,6}->13/3,{2,1}->1,{2,4}->3,{2,5}->-7,{2,6}->11,{3,4}->1,{3,5}->-41/20,{3,6}->11/10,{4,3}->1,{4,5}->-369/1070,{4,6}->1594/535},{4,6}}"
        );

        //check rref
        sparse_row_reducer.back_substitute();
        sparse_row_reducer_with_back_subs.back_substitute();
        
        assert_eq!(sparse_row_reducer.u(), sparse_row_reducer_with_back_subs.u());
        assert_eq!(
            sparse_row_reducer.u().fmt_mma(),
            "{{{1,4}->1,{1,5}->-41/20,{1,6}->11/10,{2,3}->1,{2,5}->-369/1070,{2,6}->1594/535,{3,2}->1,{3,5}->861/1070,{3,6}->-1401/535,{4,1}->1,{4,5}->-17/20,{4,6}->77/10},{4,6}}"
        );
    }

    #[test]
    fn all_at_once_rref() {
        
        let mut mat = SparseMatrix::new(0, 6, Q);

        //mat=
        //[
        //  [0,3,7,0,0,13],
        //  [1,0,0,3,-7,11],
        //  [-2,0,0,14,-27,0],
        //  [0,23,18,6,0,0]
        //]

        let values: Vec<<Q as Set>::Element> = vec![3.into(), 7.into(), 13.into()];
        let col_idcs: Vec<u32> = vec![1,2,5];
        mat.add_row(values, col_idcs);

        let values: Vec<<Q as Set>::Element> = vec![1.into(), 3.into(), (-7).into(), 11.into()];
        let col_idcs: Vec<u32> = vec![0,3,4,5];
        mat.add_row(values, col_idcs);

        let values: Vec<<Q as Set>::Element> = vec![(-2).into(), 14.into(), (-27).into()];
        let col_idcs: Vec<u32> = vec![0,3,4];
        mat.add_row(values, col_idcs);

        let values: Vec<<Q as Set>::Element> = vec![23.into(), 18.into(), 6.into()];
        let col_idcs: Vec<u32> = vec![1,2,3];
        mat.add_row(values, col_idcs);

        let mut sparse_row_reducer = SparseRowReducer::from_matrix(&mat, LuLMode::Full);
        let mut sparse_row_reducer_with_back_subs = SparseRowReducer::from_matrix_with_back_subs(&mat, LuLMode::Full);

        //check L.U == A (also checking multiplication and subtraction)
        assert_eq!(&(sparse_row_reducer.l() * sparse_row_reducer.u()), &mat);
        assert_eq!(
            &(sparse_row_reducer.l() * sparse_row_reducer.u()) - &mat,
            SparseMatrix::new(mat.nrows(), mat.ncols(), Q)
        );

        assert_eq!(&(sparse_row_reducer_with_back_subs.l() * sparse_row_reducer_with_back_subs.u()), &mat);
        assert_eq!(
            &(sparse_row_reducer_with_back_subs.l() * sparse_row_reducer_with_back_subs.u()) - &mat,
            SparseMatrix::new(mat.nrows(), mat.ncols(), Q)
        );

        //check Us explicitly
        assert_eq!(
            sparse_row_reducer.u().fmt_mma(),
            "{{{1,2}->1,{1,3}->7/3,{1,6}->13/3,{2,1}->1,{2,4}->3,{2,5}->-7,{2,6}->11,{3,4}->1,{3,5}->-41/20,{3,6}->11/10,{4,3}->1,{4,4}->-18/107,{4,6}->299/107},{4,6}}"
        );

        assert_eq!(
            sparse_row_reducer_with_back_subs.u().fmt_mma(),
            "{{{1,2}->1,{1,3}->7/3,{1,6}->13/3,{2,1}->1,{2,4}->3,{2,5}->-7,{2,6}->11,{3,4}->1,{3,5}->-41/20,{3,6}->11/10,{4,3}->1,{4,5}->-369/1070,{4,6}->1594/535},{4,6}}"
        );


        //check rref
        sparse_row_reducer.back_substitute();
        sparse_row_reducer_with_back_subs.back_substitute();
        
        assert_eq!(sparse_row_reducer.u(), sparse_row_reducer_with_back_subs.u());
        assert_eq!(
            sparse_row_reducer.u().fmt_mma(),
            "{{{1,4}->1,{1,5}->-41/20,{1,6}->11/10,{2,3}->1,{2,5}->-369/1070,{2,6}->1594/535,{3,2}->1,{3,5}->861/1070,{3,6}->-1401/535,{4,1}->1,{4,5}->-17/20,{4,6}->77/10},{4,6}}"
        );
    }

    #[test]
    fn solve() {
        // sparse 5x5 matrix triplets
        let triplets = vec![
            // row, col, entry
            (0, 0, 1.into()),
            (0, 2, 2.into()),
            (1, 1, 1.into()),
            (1, 3, 3.into()),
            (2, 2, 1.into()),
            (2, 4, 4.into()),
            (3, 3, 1.into()),
            (4, 0, 2.into()),
            (4, 4, 1.into()),
        ];

        //sparse vector pairs
        let pairs = vec![
            (0, 3.into()),
            (1, 5.into()),
            (2, 7.into()),
            (3, 2.into()),
            (4, 8.into()),
        ];

        let mat = SparseMatrix::from_triplets(5, 5, triplets, Q);
        let mat2 = mat.clone();

        let b = SparseVector::from_pairs(5, pairs, Q);
        let b2 = b.clone();

        let res = mat.solve(b);
        let res2 = mat2.solve_parallel(b2);


        //check serial vs. parallel result
        assert_eq!(res, res2);

        match res {
            Ok(value) => assert_eq!(
                value.fmt_mma(),
                "{{{1,1}->53/17,{2,1}->-1,{3,1}->-1/17,{4,1}->2,{5,1}->30/17},{5,1}}"
            ),
            Err(_) => assert!(false),
        }
    }

    #[test]
    fn sparse_det_random() {
        //compare sparse to dense algorithm
        let mat = SparseMatrix::<Q>::random(10, 10, 80);
        let mat2 = mat.to_dense();

        let det1 = mat.det();
        let det2 = mat2.det();

        assert_eq!(det1.unwrap(), det2.unwrap());
    }

    #[test]
    fn sparse_inv() {
        // sparse 5x5 matrix triplets
        let triplets = vec![
            // row, col, entry
            (0, 0, 1.into()),
            (0, 2, 2.into()),
            (1, 1, 1.into()),
            (1, 3, 3.into()),
            (2, 2, 1.into()),
            (2, 4, 4.into()),
            (3, 3, 1.into()),
            (4, 0, 2.into()),
            (4, 4, 1.into()),
        ];

        let mat = SparseMatrix::from_triplets(5, 5, triplets, Q);

        let inv = mat.inv().unwrap();

        assert_eq!(&mat * &inv, SparseMatrix::identity(5, Q));
    }

    #[test]
    fn row_iter() {
        // sparse 5x5 matrix triplets
        let triplets = vec![
            // row, col, entry
            (0, 0, 1.into()),
            (0, 2, 2.into()),
            (1, 1, 1.into()),
            (1, 3, 3.into()),
            (2, 2, 1.into()),
            (2, 4, 4.into()),
            (3, 3, 1.into()),
            (4, 0, 2.into()),
            (4, 4, 1.into()),
        ];

        let mat = SparseMatrix::from_triplets(5, 5, triplets, Q);

        let mut count_entries = 0;
        let mut count_rows = 0;
        for (idx, row) in mat.row_iter().enumerate() {
            count_rows += 1;
            assert_eq!(idx, row.0 as usize);
            assert_eq!(row.1.len(), row.2.len());
            count_entries += row.1.len();
        }

        assert_eq!(count_entries, 9);
        assert_eq!(count_rows, 5);
        
    }
}
