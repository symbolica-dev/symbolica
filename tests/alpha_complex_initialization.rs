//! This standalone test target gives the initializer a fresh global State.
//! Its crate name sorts before symbolica::special_functions, while initialize!
//! supplies only its default dependency on the base symbolica initializer.

use std::sync::atomic::{AtomicBool, Ordering};

use symbolica::{
    domains::{Ring, RingOps, algebraic::AlgebraicExtension, rational::Q},
    get_symbol, initialize,
};

static INITIALIZER_RAN: AtomicBool = AtomicBool::new(false);

initialize!(|| {
    assert!(
        get_symbol!("symbolica::root").is_none(),
        "callback must precede special-function registration"
    );
    let field = AlgebraicExtension::complex(Q);
    let imaginary = field.generator();
    assert_eq!(field.mul(&imaginary, &imaginary), field.neg(&field.one()));
    assert_eq!(field.embedding().index(), Some(1));
    INITIALIZER_RAN.store(true, Ordering::SeqCst);
});

#[test]
fn complex_constructor_is_reentrant_before_special_functions_initialize() {
    // Enter the generator cache before State initialization. Its initializer
    // must allow the callback above to enter the same cache before it completes.
    let field = AlgebraicExtension::complex(Q);
    assert!(INITIALIZER_RAN.load(Ordering::SeqCst));
    let imaginary = field.generator();
    assert_eq!(field.mul(&imaginary, &imaginary), field.neg(&field.one()));
    assert!(get_symbol!("symbolica::root").is_some());
}
