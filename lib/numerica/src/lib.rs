//! Numerica is an open-source mathematics library for Rust, that provides high-performance number types, such as error-tracking floats and finite field elements.
//!
//! It provides
//! - Abstractions over rings, Euclidean domains, fields and floats
//! - High-performance Integer with automatic up-and-downgrading to arbitrary precision types
//! - Rational numbers with reconstruction algorithms
//! - Fast finite field arithmetic
//! - Error-tracking floating point types
//! - Generic dual numbers for automatic differentiation
//! - Matrix operations and linear system solving
//! - Numerical integration using Vegas algorithm with discrete layer support
//!
//! For operations on symbols, check out the sister project [Symbolica](https://symbolica.io).
//!
//! # Example
//! Solve a linear system over the rationals:
//!
//! ```rust
//! # use numerica::tensors::matrix::Matrix;
//! # use numerica::domains::rational::Q;
//! let a = Matrix::from_linear(
//!     vec![
//!         1.into(), 2.into(), 3.into(),
//!         4.into(), 5.into(), 16.into(),
//!         7.into(), 8.into(), 9.into(),
//!     ],
//!     3, 3, Q,
//! )
//! .unwrap();
//!
//! let b = Matrix::from_linear(vec![1.into(), 2.into(), 3.into()], 3, 1, Q).unwrap();
//!
//! let r = a.solve(&b).unwrap();
//! assert_eq!(r.into_vec(), [(-1, 3), (2, 3), (0, 1)]);
//! ```
//! Solution: $(-1/3, 2/3, 0)$.
pub mod combinatorics;
pub mod domains;
pub mod numerical_integration;
pub mod printer;
pub mod tensors;
pub mod utils;
