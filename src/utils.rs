//! Utility traits and structures.

use std::ops::{Deref, DerefMut};

use dyn_clone::DynClone;

/// A wrapper around a mutable reference that tracks if the value
/// has been mutably accessed.
#[derive(Debug)]
pub struct Settable<'a, T> {
    value: &'a mut T,
    is_set: bool,
}

impl<T> Deref for Settable<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        if !self.is_set {
            panic!("Attempted to deref a Settable that was not set");
        }
        self.value
    }
}

impl<T> DerefMut for Settable<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.is_set = true;
        self.value
    }
}

impl<'a, T> From<&'a mut T> for Settable<'a, T> {
    fn from(value: &'a mut T) -> Self {
        Self {
            value,
            is_set: false,
        }
    }
}

impl<T> Settable<'_, T> {
    /// Check if the value has been set.
    pub fn is_set(&self) -> bool {
        self.is_set
    }
}

/// An enum that contains either an owned value of type `T` or a reference to a value of type `T`.
///
/// Use `Into<BorrowedOrOwned<'b, T>>>` to a accept both owned and borrowed values as arguments.
#[derive(Debug, Clone)]
pub enum BorrowedOrOwned<'a, T> {
    Owned(T),
    Borrowed(&'a T),
}

impl<'a, T> From<&'a T> for BorrowedOrOwned<'a, T> {
    fn from(v: &'a T) -> Self {
        Self::Borrowed(v)
    }
}

impl<'a, T> From<&'a mut T> for BorrowedOrOwned<'a, T> {
    fn from(v: &'a mut T) -> Self {
        Self::Borrowed(v)
    }
}

impl<T> From<T> for BorrowedOrOwned<'_, T> {
    fn from(v: T) -> Self {
        Self::Owned(v)
    }
}

impl<T> BorrowedOrOwned<'_, T> {
    /// Get a reference to the value.
    pub fn borrow(&self) -> &T {
        match self {
            BorrowedOrOwned::Owned(t) => t,
            BorrowedOrOwned::Borrowed(t) => t,
        }
    }
}

impl<T: Clone> BorrowedOrOwned<'_, T> {
    /// Yield an owned value.
    pub fn yield_owned(self) -> T {
        match self {
            BorrowedOrOwned::Owned(t) => t,
            BorrowedOrOwned::Borrowed(t) => t.clone(),
        }
    }

    /// Create an owned version of `Self`.
    pub fn into_owned(self) -> BorrowedOrOwned<'static, T> {
        match self {
            BorrowedOrOwned::Owned(t) => BorrowedOrOwned::Owned(t),
            BorrowedOrOwned::Borrowed(t) => BorrowedOrOwned::Owned(t.clone()),
        }
    }
}

impl<T> Deref for BorrowedOrOwned<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.borrow()
    }
}

/// A cloneable function that checks for abort.
pub trait AbortCheck: Fn() -> bool + DynClone + Send + Sync {}
dyn_clone::clone_trait_object!(AbortCheck);
impl<T: Clone + Send + Sync + Fn() -> bool> AbortCheck for T {}
