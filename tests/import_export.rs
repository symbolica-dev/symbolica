use std::io::Cursor;

use smartstring::SmartString;
use symbolica::{atom::Atom, parse, state::State, symbol};

fn conflict() {
    symbol!("x", "y");
    symbol!("f"; Symmetric);

    let a = parse!("f(x, y)*x^2");

    let mut a_export = vec![];
    a.as_view().write(&mut a_export).unwrap();

    let mut state_export = vec![];
    State::export(&mut state_export).unwrap();

    // reset the state and create a conflict
    unsafe { State::reset() };

    symbol!("y");
    symbol!("x");
    symbol!("f");

    let state_map = State::import(
        &mut Cursor::new(&state_export),
        Some(Box::new(|old_name| SmartString::from(old_name) + "1")),
    )
    .unwrap();

    let a_rec = Atom::import_with_map(&mut Cursor::new(&a_export), &state_map).unwrap();

    let r = parse!("x^2*f1(y, x)");
    assert_eq!(a_rec, r);
}

#[test]
fn rational_rename() {
    symbol!("x");

    let a = parse!("x^2*coeff(x)");

    let mut a_export = vec![];
    a.as_view().write(&mut a_export).unwrap();

    let mut state_export = vec![];
    State::export(&mut state_export).unwrap();

    // reset the state and create a conflict
    unsafe { State::reset() };

    symbol!("y");

    let state_map = State::import(&mut Cursor::new(&state_export), None).unwrap();

    let a_rec = Atom::import_with_map(&mut Cursor::new(&a_export), &state_map).unwrap();

    let r = parse!("x^2*coeff(x)");
    assert_eq!(a_rec, r);

    unsafe { State::reset() };
    conflict();
}
