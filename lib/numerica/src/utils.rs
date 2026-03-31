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
        &self.value
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

/// A cloneable function that checks for abort.
pub trait AbortCheck: Fn() -> bool + DynClone + Send + Sync {}
dyn_clone::clone_trait_object!(AbortCheck);
impl<T: Clone + Send + Sync + Fn() -> bool> AbortCheck for T {}
