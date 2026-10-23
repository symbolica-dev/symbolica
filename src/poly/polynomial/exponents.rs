//! Exponent rows, including the empty row of a constant with no variables.

/// Iterator over the exponent rows of a polynomial.
///
/// A nonzero polynomial without variables has one empty row; a zero polynomial
/// has no rows. Nonzero-width rows retain the underlying slice chunk iterator.
#[derive(Clone, Debug)]
pub struct ExponentsIterator<I: Iterator> {
    chunks: I,
    constant: Option<I::Item>,
}

impl<I: Iterator> ExponentsIterator<I> {
    pub(super) fn new(chunks: I, constant: Option<I::Item>) -> Self {
        Self { chunks, constant }
    }
}

impl<I: Iterator> Iterator for ExponentsIterator<I> {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.constant.take().or_else(|| self.chunks.next())
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let extra = usize::from(self.constant.is_some());
        let (lower, upper) = self.chunks.size_hint();
        (
            lower.saturating_add(extra),
            upper.and_then(|n| n.checked_add(extra)),
        )
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        if let Some(row) = self.constant.take() {
            if n == 0 {
                Some(row)
            } else {
                self.chunks.nth(n - 1)
            }
        } else {
            self.chunks.nth(n)
        }
    }

    #[inline]
    fn count(self) -> usize {
        usize::from(self.constant.is_some()) + self.chunks.count()
    }

    #[inline]
    fn last(self) -> Option<Self::Item> {
        self.chunks.last().or(self.constant)
    }

    #[inline]
    fn fold<B, F: FnMut(B, Self::Item) -> B>(self, init: B, mut f: F) -> B {
        let init = if let Some(row) = self.constant {
            f(init, row)
        } else {
            init
        };
        self.chunks.fold(init, f)
    }
}

impl<I: DoubleEndedIterator + ExactSizeIterator> DoubleEndedIterator for ExponentsIterator<I> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        self.chunks.next_back().or_else(|| self.constant.take())
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        let len = self.chunks.len();
        let row = self.chunks.nth_back(n);
        if n < len {
            row
        } else {
            self.constant.take().filter(|_| n == len)
        }
    }
}

impl<I: ExactSizeIterator> ExactSizeIterator for ExponentsIterator<I> {
    #[inline]
    fn len(&self) -> usize {
        self.chunks.len() + usize::from(self.constant.is_some())
    }
}

impl<I: std::iter::FusedIterator> std::iter::FusedIterator for ExponentsIterator<I> {}

#[cfg(test)]
mod tests {
    use super::super::MultivariatePolynomial;
    use crate::{
        domains::rational::{Q, Rational},
        poly::PolyVariable,
        symbol,
    };
    use std::{iter::FusedIterator, sync::Arc};

    type Poly = MultivariatePolynomial<Q, u32>;

    fn traits<I: DoubleEndedIterator + ExactSizeIterator + FusedIterator>(_: &I) {}

    #[test]
    fn exponent_iteration_zero_variable_constants() {
        let one = Poly::new_one(&Q);
        for mut p in [
            Poly::new_zero(&Q),
            one.clone(),
            one.constant(Rational::from(-3)),
        ] {
            let count = usize::from(!p.is_zero());
            let rows = p.exponents_iter();
            traits(&rows);
            assert_eq!(rows.size_hint(), (count, Some(count)));
            assert_eq!(rows.clone().collect::<Vec<_>>(), vec![&[] as &[u32]; count]);
            assert_eq!(p.coefficients.iter().zip(rows).count(), count);
            let mut rows = p.exponents_iter_mut();
            traits(&rows);
            assert_eq!(rows.len(), count);
            assert_eq!(rows.next_back().map(|r| r.len()), (count == 1).then_some(0));
            assert_eq!(rows.next(), None);
            assert_eq!(rows.next_back(), None);
        }
    }

    #[test]
    fn exponent_iteration_retains_row_order_and_mutates_storage() {
        let variables = Arc::new(vec![
            PolyVariable::from(symbol!("exponent_rows_x")),
            PolyVariable::from(symbol!("exponent_rows_y")),
        ]);
        let mut p = Poly::new(&Q, None, variables);
        for (coefficient, exponents) in [(2, [0, 0]), (3, [1, 0]), (5, [2, 1])] {
            p.append_monomial(Rational::from(coefficient), &exponents);
        }
        let mut rows = p.exponents_iter();
        assert_eq!(rows.nth(1), Some(&[1, 0][..]));
        assert_eq!(rows.next_back(), Some(&[2, 1][..]));
        assert_eq!(rows.len(), 0);
        let mut rows = p.exponents_iter_mut();
        traits(&rows);
        rows.next().unwrap()[0] = 1;
        rows.next_back().unwrap()[0] = 3;
        rows.next().unwrap()[0] = 2;
        assert_eq!(rows.next(), None);
        assert_eq!(p.exponents, [1, 0, 2, 0, 3, 1]);
    }
}
