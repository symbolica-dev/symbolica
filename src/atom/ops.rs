use crate::atom::{AliasedAtom, Atom, AtomView, InlineVar, Symbol};
use crate::coefficient::Coefficient;
use crate::domains::{float::Complex, integer::Integer, rational::Rational};
use crate::state::Workspace;
use numerica::domains::float::Float;

macro_rules! atom_op_views {
    ($lhs:expr, $op_ws_fn:ident, $rhs:expr) => {{
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            $lhs.$op_ws_fn(ws, $rhs, &mut t);
            t.into_inner()
        })
    }};
}

macro_rules! atom_op_num_rhs {
    ($lhs:expr, $op_ws_fn:ident, $rhs:expr) => {{
        Workspace::get_local().with(|ws| {
            let n = ws.new_num($rhs);
            let mut t = ws.new_atom();
            $lhs.$op_ws_fn(ws, n.as_view(), &mut t);
            t.into_inner()
        })
    }};
}

macro_rules! atom_op_num_lhs {
    ($lhs:expr, $op_ws_fn:ident, $rhs:expr) => {{
        Workspace::get_local().with(|ws| {
            let n = ws.new_num($lhs);
            let mut t = ws.new_atom();
            n.as_view().$op_ws_fn(ws, $rhs, &mut t);
            t.into_inner()
        })
    }};
}

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
    fn neg(self) -> Atom {
        -self.as_view()
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

macro_rules! impl_binary_ops {
    ($op_trait:ident, $op_method:ident, $op_ws_fn:ident) => {
        impl std::ops::$op_trait<Atom> for Atom {
            type Output = Atom;
            fn $op_method(self, rhs: Atom) -> Atom {
                atom_op_views!(self.as_view(), $op_ws_fn, rhs.as_view())
            }
        }

        impl std::ops::$op_trait<Atom> for &Atom {
            type Output = Atom;
            fn $op_method(self, rhs: Atom) -> Atom {
                atom_op_views!(self.as_view(), $op_ws_fn, rhs.as_view())
            }
        }

        impl std::ops::$op_trait<&Atom> for Atom {
            type Output = Atom;
            fn $op_method(self, rhs: &Atom) -> Atom {
                atom_op_views!(self.as_view(), $op_ws_fn, rhs.as_view())
            }
        }

        impl std::ops::$op_trait<&Atom> for &Atom {
            type Output = Atom;
            fn $op_method(self, rhs: &Atom) -> Atom {
                atom_op_views!(self.as_view(), $op_ws_fn, rhs.as_view())
            }
        }

        impl<'a, 'b> std::ops::$op_trait<AtomView<'b>> for AtomView<'a> {
            type Output = Atom;
            fn $op_method(self, rhs: AtomView<'b>) -> Atom {
                atom_op_views!(self, $op_ws_fn, rhs)
            }
        }

        impl<'a> std::ops::$op_trait<AtomView<'a>> for Atom {
            type Output = Atom;
            fn $op_method(self, rhs: AtomView<'a>) -> Atom {
                atom_op_views!(self.as_view(), $op_ws_fn, rhs)
            }
        }

        impl std::ops::$op_trait<Symbol> for Atom {
            type Output = Atom;
            fn $op_method(self, rhs: Symbol) -> Atom {
                let rhs = InlineVar::new(rhs);
                atom_op_views!(self.as_view(), $op_ws_fn, rhs.as_view())
            }
        }

        impl std::ops::$op_trait<Symbol> for &Atom {
            type Output = Atom;
            fn $op_method(self, rhs: Symbol) -> Atom {
                let rhs = InlineVar::new(rhs);
                atom_op_views!(self.as_view(), $op_ws_fn, rhs.as_view())
            }
        }

        impl std::ops::$op_trait<Symbol> for AtomView<'_> {
            type Output = Atom;
            fn $op_method(self, rhs: Symbol) -> Atom {
                let rhs = InlineVar::new(rhs);
                atom_op_views!(self, $op_ws_fn, rhs.as_view())
            }
        }

        impl std::ops::$op_trait<Symbol> for Symbol {
            type Output = Atom;
            fn $op_method(self, rhs: Symbol) -> Atom {
                let lhs = InlineVar::new(self);
                let rhs = InlineVar::new(rhs);
                atom_op_views!(lhs.as_view(), $op_ws_fn, rhs.as_view())
            }
        }

        impl std::ops::$op_trait<Atom> for Symbol {
            type Output = Atom;
            fn $op_method(self, rhs: Atom) -> Atom {
                let lhs = InlineVar::new(self);
                atom_op_views!(lhs.as_view(), $op_ws_fn, rhs.as_view())
            }
        }

        impl std::ops::$op_trait<&Atom> for Symbol {
            type Output = Atom;
            fn $op_method(self, rhs: &Atom) -> Atom {
                let lhs = InlineVar::new(self);
                atom_op_views!(lhs.as_view(), $op_ws_fn, rhs.as_view())
            }
        }

        impl std::ops::$op_trait<AtomView<'_>> for Symbol {
            type Output = Atom;
            fn $op_method(self, rhs: AtomView<'_>) -> Atom {
                let lhs = InlineVar::new(self);
                atom_op_views!(lhs.as_view(), $op_ws_fn, rhs)
            }
        }

        impl std::ops::$op_trait<Atom> for AliasedAtom {
            type Output = AliasedAtom;
            fn $op_method(self, rhs: Atom) -> AliasedAtom {
                let AliasedAtom { root, aliases } = self;
                AliasedAtom {
                    root: atom_op_views!(root.as_view(), $op_ws_fn, rhs.as_view()),
                    aliases,
                }
            }
        }

        impl std::ops::$op_trait<Atom> for &AliasedAtom {
            type Output = AliasedAtom;
            fn $op_method(self, rhs: Atom) -> AliasedAtom {
                AliasedAtom {
                    root: atom_op_views!(self.root.as_view(), $op_ws_fn, rhs.as_view()),
                    aliases: self.aliases.clone(),
                }
            }
        }

        impl std::ops::$op_trait<&Atom> for AliasedAtom {
            type Output = AliasedAtom;
            fn $op_method(self, rhs: &Atom) -> AliasedAtom {
                let AliasedAtom { root, aliases } = self;
                AliasedAtom {
                    root: atom_op_views!(root.as_view(), $op_ws_fn, rhs.as_view()),
                    aliases,
                }
            }
        }

        impl std::ops::$op_trait<&Atom> for &AliasedAtom {
            type Output = AliasedAtom;
            fn $op_method(self, rhs: &Atom) -> AliasedAtom {
                AliasedAtom {
                    root: atom_op_views!(self.root.as_view(), $op_ws_fn, rhs.as_view()),
                    aliases: self.aliases.clone(),
                }
            }
        }

        impl std::ops::$op_trait<AtomView<'_>> for AliasedAtom {
            type Output = AliasedAtom;
            fn $op_method(self, rhs: AtomView<'_>) -> AliasedAtom {
                let AliasedAtom { root, aliases } = self;
                AliasedAtom {
                    root: atom_op_views!(root.as_view(), $op_ws_fn, rhs),
                    aliases,
                }
            }
        }

        impl std::ops::$op_trait<AtomView<'_>> for &AliasedAtom {
            type Output = AliasedAtom;
            fn $op_method(self, rhs: AtomView<'_>) -> AliasedAtom {
                AliasedAtom {
                    root: atom_op_views!(self.root.as_view(), $op_ws_fn, rhs),
                    aliases: self.aliases.clone(),
                }
            }
        }

        impl std::ops::$op_trait<Symbol> for AliasedAtom {
            type Output = AliasedAtom;
            fn $op_method(self, rhs: Symbol) -> AliasedAtom {
                let rhs = InlineVar::new(rhs);
                let AliasedAtom { root, aliases } = self;
                AliasedAtom {
                    root: atom_op_views!(root.as_view(), $op_ws_fn, rhs.as_view()),
                    aliases,
                }
            }
        }

        impl std::ops::$op_trait<Symbol> for &AliasedAtom {
            type Output = AliasedAtom;
            fn $op_method(self, rhs: Symbol) -> AliasedAtom {
                let rhs = InlineVar::new(rhs);
                AliasedAtom {
                    root: atom_op_views!(self.root.as_view(), $op_ws_fn, rhs.as_view()),
                    aliases: self.aliases.clone(),
                }
            }
        }

        impl std::ops::$op_trait<AliasedAtom> for Atom {
            type Output = AliasedAtom;
            fn $op_method(self, rhs: AliasedAtom) -> AliasedAtom {
                let AliasedAtom { root, aliases } = rhs;
                AliasedAtom {
                    root: atom_op_views!(self.as_view(), $op_ws_fn, root.as_view()),
                    aliases,
                }
            }
        }

        impl std::ops::$op_trait<&AliasedAtom> for Atom {
            type Output = AliasedAtom;
            fn $op_method(self, rhs: &AliasedAtom) -> AliasedAtom {
                AliasedAtom {
                    root: atom_op_views!(self.as_view(), $op_ws_fn, rhs.root.as_view()),
                    aliases: rhs.aliases.clone(),
                }
            }
        }

        impl std::ops::$op_trait<AliasedAtom> for &Atom {
            type Output = AliasedAtom;
            fn $op_method(self, rhs: AliasedAtom) -> AliasedAtom {
                let AliasedAtom { root, aliases } = rhs;
                AliasedAtom {
                    root: atom_op_views!(self.as_view(), $op_ws_fn, root.as_view()),
                    aliases,
                }
            }
        }

        impl std::ops::$op_trait<&AliasedAtom> for &Atom {
            type Output = AliasedAtom;
            fn $op_method(self, rhs: &AliasedAtom) -> AliasedAtom {
                AliasedAtom {
                    root: atom_op_views!(self.as_view(), $op_ws_fn, rhs.root.as_view()),
                    aliases: rhs.aliases.clone(),
                }
            }
        }

        impl std::ops::$op_trait<AliasedAtom> for AtomView<'_> {
            type Output = AliasedAtom;
            fn $op_method(self, rhs: AliasedAtom) -> AliasedAtom {
                let AliasedAtom { root, aliases } = rhs;
                AliasedAtom {
                    root: atom_op_views!(self, $op_ws_fn, root.as_view()),
                    aliases,
                }
            }
        }

        impl std::ops::$op_trait<&AliasedAtom> for AtomView<'_> {
            type Output = AliasedAtom;
            fn $op_method(self, rhs: &AliasedAtom) -> AliasedAtom {
                AliasedAtom {
                    root: atom_op_views!(self, $op_ws_fn, rhs.root.as_view()),
                    aliases: rhs.aliases.clone(),
                }
            }
        }

        impl std::ops::$op_trait<AliasedAtom> for Symbol {
            type Output = AliasedAtom;
            fn $op_method(self, rhs: AliasedAtom) -> AliasedAtom {
                let lhs = InlineVar::new(self);
                let AliasedAtom { root, aliases } = rhs;
                AliasedAtom {
                    root: atom_op_views!(lhs.as_view(), $op_ws_fn, root.as_view()),
                    aliases,
                }
            }
        }

        impl std::ops::$op_trait<&AliasedAtom> for Symbol {
            type Output = AliasedAtom;
            fn $op_method(self, rhs: &AliasedAtom) -> AliasedAtom {
                let lhs = InlineVar::new(self);
                AliasedAtom {
                    root: atom_op_views!(lhs.as_view(), $op_ws_fn, rhs.root.as_view()),
                    aliases: rhs.aliases.clone(),
                }
            }
        }

        impl<T: Into<Coefficient>> std::ops::$op_trait<T> for Symbol {
            type Output = Atom;
            fn $op_method(self, rhs: T) -> Atom {
                let lhs = InlineVar::new(self);
                atom_op_num_rhs!(lhs.as_view(), $op_ws_fn, rhs)
            }
        }

        impl<T: Into<Coefficient>> std::ops::$op_trait<T> for Atom {
            type Output = Atom;
            fn $op_method(self, rhs: T) -> Atom {
                atom_op_num_rhs!(self.as_view(), $op_ws_fn, rhs)
            }
        }

        impl<T: Into<Coefficient>> std::ops::$op_trait<T> for &Atom {
            type Output = Atom;
            fn $op_method(self, rhs: T) -> Atom {
                atom_op_num_rhs!(self.as_view(), $op_ws_fn, rhs)
            }
        }

        impl<T: Into<Coefficient>> std::ops::$op_trait<T> for AliasedAtom {
            type Output = AliasedAtom;
            fn $op_method(self, rhs: T) -> AliasedAtom {
                let AliasedAtom { root, aliases } = self;
                AliasedAtom {
                    root: atom_op_num_rhs!(root.as_view(), $op_ws_fn, rhs),
                    aliases,
                }
            }
        }

        impl<T: Into<Coefficient>> std::ops::$op_trait<T> for &AliasedAtom {
            type Output = AliasedAtom;
            fn $op_method(self, rhs: T) -> AliasedAtom {
                AliasedAtom {
                    root: atom_op_num_rhs!(self.root.as_view(), $op_ws_fn, rhs),
                    aliases: self.aliases.clone(),
                }
            }
        }
    };
}

macro_rules! impl_numeric_lhs_symbol_ops {
    ($op_trait:ident, $op_method:ident, $op_ws_fn:ident, $($ty:ty),+ $(,)?) => {
        $(
            impl std::ops::$op_trait<Symbol> for $ty {
                type Output = Atom;
                fn $op_method(self, rhs: Symbol) -> Atom {
                    let rhs = InlineVar::new(rhs);
                    atom_op_num_lhs!(self, $op_ws_fn, rhs.as_view())
                }
            }
        )+
    };
}

macro_rules! impl_numeric_lhs_atom_ops {
    ($op_trait:ident, $op_method:ident, $op_ws_fn:ident, $($ty:ty),+ $(,)?) => {
        $(
            impl std::ops::$op_trait<Atom> for $ty {
                type Output = Atom;
                fn $op_method(self, rhs: Atom) -> Atom {
                    atom_op_num_lhs!(self, $op_ws_fn, rhs.as_view())
                }
            }

            impl std::ops::$op_trait<&Atom> for $ty {
                type Output = Atom;
                fn $op_method(self, rhs: &Atom) -> Atom {
                    atom_op_num_lhs!(self, $op_ws_fn, rhs.as_view())
                }
            }

            impl std::ops::$op_trait<AtomView<'_>> for $ty {
                type Output = Atom;
                fn $op_method(self, rhs: AtomView<'_>) -> Atom {
                    atom_op_num_lhs!(self, $op_ws_fn, rhs)
                }
            }
        )+
    };
}

macro_rules! impl_numeric_lhs_aliased_atom_ops {
    ($op_trait:ident, $op_method:ident, $op_ws_fn:ident, $($ty:ty),+ $(,)?) => {
        $(
            impl std::ops::$op_trait<AliasedAtom> for $ty {
                type Output = AliasedAtom;
                fn $op_method(self, rhs: AliasedAtom) -> AliasedAtom {
                    let AliasedAtom { root, aliases } = rhs;
                    AliasedAtom {
                        root: atom_op_num_lhs!(self, $op_ws_fn, root.as_view()),
                        aliases,
                    }
                }
            }

            impl std::ops::$op_trait<&AliasedAtom> for $ty {
                type Output = AliasedAtom;
                fn $op_method(self, rhs: &AliasedAtom) -> AliasedAtom {
                    AliasedAtom {
                        root: atom_op_num_lhs!(self, $op_ws_fn, rhs.root.as_view()),
                        aliases: rhs.aliases.clone(),
                    }
                }
            }
        )+
    };
}

macro_rules! impl_domain_lhs_atom_ops {
    ($op_trait:ident, $op_method:ident, $op_ws_fn:ident, $($ty:ty),+ $(,)?) => {
        $(
            impl std::ops::$op_trait<Atom> for $ty {
                type Output = Atom;
                fn $op_method(self, rhs: Atom) -> Atom {
                    atom_op_num_lhs!(self, $op_ws_fn, rhs.as_view())
                }
            }

            impl std::ops::$op_trait<AtomView<'_>> for $ty {
                type Output = Atom;
                fn $op_method(self, rhs: AtomView<'_>) -> Atom {
                    atom_op_num_lhs!(self, $op_ws_fn, rhs)
                }
            }
        )+
    };
}

macro_rules! impl_numeric_lhs_aliased_atom_ops_for_all_ops {
    ($($ty:ty),+ $(,)?) => {
        impl_numeric_lhs_aliased_atom_ops!(Add, add, add_with_ws_into, $($ty),+);
        impl_numeric_lhs_aliased_atom_ops!(Sub, sub, sub_with_ws_into, $($ty),+);
        impl_numeric_lhs_aliased_atom_ops!(Mul, mul, mul_with_ws_into, $($ty),+);
        impl_numeric_lhs_aliased_atom_ops!(Div, div, div_with_ws_into, $($ty),+);
    };
}

macro_rules! impl_numeric_lhs_symbol_ops_for_all_ops {
    ($($ty:ty),+ $(,)?) => {
        impl_numeric_lhs_symbol_ops!(Add, add, add_with_ws_into, $($ty),+);
        impl_numeric_lhs_symbol_ops!(Sub, sub, sub_with_ws_into, $($ty),+);
        impl_numeric_lhs_symbol_ops!(Mul, mul, mul_with_ws_into, $($ty),+);
        impl_numeric_lhs_symbol_ops!(Div, div, div_with_ws_into, $($ty),+);
    };
}

macro_rules! impl_numeric_lhs_atom_ops_for_all_ops {
    ($($ty:ty),+ $(,)?) => {
        impl_numeric_lhs_atom_ops!(Add, add, add_with_ws_into, $($ty),+);
        impl_numeric_lhs_atom_ops!(Sub, sub, sub_with_ws_into, $($ty),+);
        impl_numeric_lhs_atom_ops!(Mul, mul, mul_with_ws_into, $($ty),+);
        impl_numeric_lhs_atom_ops!(Div, div, div_with_ws_into, $($ty),+);
    };
}

macro_rules! impl_domain_lhs_atom_ops_for_all_ops {
    ($($ty:ty),+ $(,)?) => {
        impl_domain_lhs_atom_ops!(Add, add, add_with_ws_into, $($ty),+);
        impl_domain_lhs_atom_ops!(Sub, sub, sub_with_ws_into, $($ty),+);
        impl_domain_lhs_atom_ops!(Mul, mul, mul_with_ws_into, $($ty),+);
        impl_domain_lhs_atom_ops!(Div, div, div_with_ws_into, $($ty),+);
    };
}

impl_binary_ops!(Add, add, add_with_ws_into);
impl_binary_ops!(Sub, sub, sub_with_ws_into);
impl_binary_ops!(Mul, mul, mul_with_ws_into);
impl_binary_ops!(Div, div, div_with_ws_into);

impl_numeric_lhs_symbol_ops_for_all_ops!(
    Coefficient,
    Rational,
    Float,
    Complex<Rational>,
    Complex<Float>,
    i8,
    i16,
    i32,
    i64,
    i128,
    isize,
    u8,
    u16,
    u32,
    u64,
    u128,
    usize,
    f64,
);

impl_numeric_lhs_atom_ops_for_all_ops!(
    i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f64,
);

impl_numeric_lhs_aliased_atom_ops_for_all_ops!(
    i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, f64,
);

impl_domain_lhs_atom_ops_for_all_ops!(Integer, Rational);

macro_rules! impl_assign_ops {
    ($assign_trait:ident, $assign_method:ident, $op_trait:ident, $op_method:ident) => {
        impl std::ops::$assign_trait<Atom> for Atom {
            fn $assign_method(&mut self, rhs: Atom) {
                *self = std::ops::$op_trait::$op_method(std::mem::take(self), rhs);
            }
        }

        impl std::ops::$assign_trait<&Atom> for Atom {
            fn $assign_method(&mut self, rhs: &Atom) {
                *self = std::ops::$op_trait::$op_method(std::mem::take(self), rhs);
            }
        }

        impl<'a> std::ops::$assign_trait<AtomView<'a>> for Atom {
            fn $assign_method(&mut self, rhs: AtomView<'a>) {
                *self = std::ops::$op_trait::$op_method(std::mem::take(self), rhs);
            }
        }

        impl std::ops::$assign_trait<Symbol> for Atom {
            fn $assign_method(&mut self, rhs: Symbol) {
                *self = std::ops::$op_trait::$op_method(std::mem::take(self), rhs);
            }
        }

        impl<T: Into<Coefficient>> std::ops::$assign_trait<T> for Atom {
            fn $assign_method(&mut self, rhs: T) {
                *self = std::ops::$op_trait::$op_method(std::mem::take(self), rhs);
            }
        }
    };
}

impl_assign_ops!(AddAssign, add_assign, Add, add);
impl_assign_ops!(SubAssign, sub_assign, Sub, sub);
impl_assign_ops!(MulAssign, mul_assign, Mul, mul);
impl_assign_ops!(DivAssign, div_assign, Div, div);
