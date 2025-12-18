//! Defines core algebraic traits and data structures.
//!
//! The core trait is [Ring], which has two binary operations, addition and multiplication.
//! Each ring has an associated element type, that should not be confused with the ring type itself.
//! For example:
//! - The ring of integers [Z](type@integer::Z) has elements of type [Integer](integer::Integer).
//! - The ring of rational numbers [Q](type@rational::Q) has elements of type [Rational](rational::Rational).
//! - The ring of finite fields [FiniteField](finite_field::FiniteField) has elements of type [FiniteField](finite_field::FiniteFieldElement).
//! - The ring of polynomials [PolynomialRing](super::poly::polynomial::PolynomialRing) has elements of type [MultivariatePolynomial](super::poly::polynomial::MultivariatePolynomial).
//!
//! In general, the ring elements do not implement operations such as addition or multiplication,
//! but rather the ring itself does. Most Symbolica structures are generic over the ring type.
//!
//! An extension of the ring trait is the [`EuclideanDomain`] trait, which adds the ability to compute remainders, quotients, and gcds.
//! Another extension is the [`Field`] trait, which adds the ability to divide and invert elements.
pub mod algebraic_number;
pub mod atom;
pub mod factorized_rational_polynomial;
pub mod rational_polynomial;

pub use numerica::domains::*;
