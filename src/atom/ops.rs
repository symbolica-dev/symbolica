use crate::atom::{Atom, AtomView, InlineVar, Symbol};
use crate::coefficient::Coefficient;
use crate::state::Workspace;

// Unary negation
impl std::ops::Neg for &Atom {
    type Output = Atom;
    fn neg(self) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.as_view().neg_with_ws_into(ws, &mut t);
            t.into_inner()
        })
    }
}

impl std::ops::Neg for Atom {
    type Output = Atom;
    fn neg(mut self) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.as_view().neg_with_ws_into(ws, &mut t);
            std::mem::swap(&mut self, &mut t);
        });
        self
    }
}

impl std::ops::Neg for AtomView<'_> {
    type Output = Atom;
    fn neg(self) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.neg_with_ws_into(ws, &mut t);
            t.into_inner()
        })
    }
}

/// Implements all arithmetic operations for Atom types with consistent patterns
/// This generates implementations for:
/// - Binary operations (+, -, *, /)
/// - Assignment operations (+=, -=, *=, /=)
macro_rules! impl_atom_arithmetic {
    () => {
        // Binary operations (+, -, *, /)
        impl_binary_ops!(Add, add, add_with_ws_into);
        impl_binary_ops!(Sub, sub, sub_with_ws_into);
        impl_binary_ops!(Mul, mul, mul_with_ws_into);
        impl_binary_ops!(Div, div, div_with_ws_into);

        // Assignment operations (+=, -=, *=, /=)
        impl_assign_ops!(AddAssign, add_assign, Add, add);
        impl_assign_ops!(SubAssign, sub_assign, Sub, sub);
        impl_assign_ops!(MulAssign, mul_assign, Mul, mul);
        impl_assign_ops!(DivAssign, div_assign, Div, div);
    };
}

/// Implements binary operations for various combinations of Atom types
macro_rules! impl_binary_ops {
    ($op_trait:ident, $op_method:ident, $op_ws_fn:ident) => {
        // Atom op Atom
        impl std::ops::$op_trait<Atom> for Atom {
            type Output = Atom;
            fn $op_method(mut self, rhs: Atom) -> Atom {
                Workspace::get_local().with(|ws| {
                    let mut t = ws.new_atom();
                    self.as_view().$op_ws_fn(ws, rhs.as_view(), &mut t);
                    std::mem::swap(&mut self, &mut t);
                });
                self
            }
        }

        // &Atom op Atom
        impl std::ops::$op_trait<Atom> for &Atom {
            type Output = Atom;
            fn $op_method(self, mut rhs: Atom) -> Atom {
                Workspace::get_local().with(|ws| {
                    let mut t = ws.new_atom();
                    self.as_view().$op_ws_fn(ws, rhs.as_view(), &mut t);
                    std::mem::swap(&mut rhs, &mut t);
                });
                rhs
            }
        }

        // Atom op &Atom
        impl std::ops::$op_trait<&Atom> for Atom {
            type Output = Atom;
            fn $op_method(mut self, rhs: &Atom) -> Atom {
                Workspace::get_local().with(|ws| {
                    let mut t = ws.new_atom();
                    self.as_view().$op_ws_fn(ws, rhs.as_view(), &mut t);
                    std::mem::swap(&mut self, &mut t);
                });
                self
            }
        }

        // &Atom op &Atom
        impl std::ops::$op_trait<&Atom> for &Atom {
            type Output = Atom;
            fn $op_method(self, rhs: &Atom) -> Atom {
                Workspace::get_local().with(|ws| {
                    let mut t = ws.new_atom();
                    self.as_view().$op_ws_fn(ws, rhs.as_view(), &mut t);
                    t.into_inner()
                })
            }
        }

        // AtomView op AtomView
        impl<'a, 'b> std::ops::$op_trait<AtomView<'b>> for AtomView<'a> {
            type Output = Atom;
            fn $op_method(self, rhs: AtomView<'b>) -> Atom {
                Workspace::get_local().with(|ws| {
                    let mut t = ws.new_atom();
                    self.$op_ws_fn(ws, rhs, &mut t);
                    t.into_inner()
                })
            }
        }

        // Atom op AtomView
        impl<'a> std::ops::$op_trait<AtomView<'a>> for Atom {
            type Output = Atom;
            fn $op_method(mut self, rhs: AtomView<'a>) -> Atom {
                Workspace::get_local().with(|ws| {
                    let mut t = ws.new_atom();
                    self.as_view().$op_ws_fn(ws, rhs, &mut t);
                    std::mem::swap(&mut self, &mut t);
                });
                self
            }
        }

        // Atom op Symbol
        impl std::ops::$op_trait<Symbol> for Atom {
            type Output = Atom;
            fn $op_method(mut self, rhs: Symbol) -> Atom {
                let v = InlineVar::new(rhs);
                Workspace::get_local().with(|ws| {
                    let mut t = ws.new_atom();
                    self.as_view().$op_ws_fn(ws, v.as_view(), &mut t);
                    std::mem::swap(&mut self, &mut t);
                });
                self
            }
        }

        // Symbol op Symbol
        impl std::ops::$op_trait<Symbol> for Symbol {
            type Output = Atom;
            fn $op_method(self, rhs: Symbol) -> Atom {
                let s = InlineVar::new(self);
                let r = InlineVar::new(rhs);
                Workspace::get_local().with(|ws| {
                    let mut t = ws.new_atom();
                    s.as_view().$op_ws_fn(ws, r.as_view(), &mut t);
                    t.into_inner()
                })
            }
        }

        // Atom op Numeric
        impl<T: Into<Coefficient>> std::ops::$op_trait<T> for Atom {
            type Output = Atom;
            fn $op_method(mut self, rhs: T) -> Atom {
                Workspace::get_local().with(|ws| {
                    let n = ws.new_num(rhs);
                    let mut t = ws.new_atom();
                    self.as_view().$op_ws_fn(ws, n.as_view(), &mut t);
                    std::mem::swap(&mut self, &mut t);
                });
                self
            }
        }

        // &Atom op Numeric
        impl<T: Into<Coefficient>> std::ops::$op_trait<T> for &Atom {
            type Output = Atom;
            fn $op_method(self, rhs: T) -> Atom {
                Workspace::get_local().with(|ws| {
                    let n = ws.new_num(rhs);
                    let mut t = ws.new_atom();
                    self.as_view().$op_ws_fn(ws, n.as_view(), &mut t);
                    t.into_inner()
                })
            }
        }
    };
}

/// Implements assignment operations for various combinations of Atom types
macro_rules! impl_assign_ops {
    ($assign_trait:ident, $assign_method:ident, $op_trait:ident, $op_method:ident) => {
        // Atom op= Atom
        impl std::ops::$assign_trait<Atom> for Atom {
            fn $assign_method(&mut self, rhs: Atom) {
                *self = std::ops::$op_trait::$op_method(std::mem::take(self), rhs);
            }
        }

        // Atom op= &Atom
        impl std::ops::$assign_trait<&Atom> for Atom {
            fn $assign_method(&mut self, rhs: &Atom) {
                *self = std::ops::$op_trait::$op_method(std::mem::take(self), rhs);
            }
        }

        // Atom op= AtomView
        impl<'a> std::ops::$assign_trait<AtomView<'a>> for Atom {
            fn $assign_method(&mut self, rhs: AtomView<'a>) {
                *self = std::ops::$op_trait::$op_method(std::mem::take(self), rhs);
            }
        }

        // Atom op= Symbol
        impl std::ops::$assign_trait<Symbol> for Atom {
            fn $assign_method(&mut self, rhs: Symbol) {
                *self = std::ops::$op_trait::$op_method(std::mem::take(self), rhs);
            }
        }

        // Atom op= Numeric
        impl<T: Into<Coefficient>> std::ops::$assign_trait<T> for Atom {
            fn $assign_method(&mut self, rhs: T) {
                *self = std::ops::$op_trait::$op_method(std::mem::take(self), rhs);
            }
        }
    };
}

impl_atom_arithmetic!();
