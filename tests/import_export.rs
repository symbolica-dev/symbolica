use std::io::Cursor;

use smartstring::SmartString;
use symbolica::{
    atom::{Atom, AtomCore},
    parse,
    state::State,
    symbol,
};

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

fn partial_state_export() {
    unsafe { State::reset() };

    // Interleave unused symbols so that the exported symbol IDs are sparse.
    symbol!("partial_export_unused_before");
    symbol!("partial_export_x");
    symbol!("partial_export_unused_between");
    symbol!("partial_export_f"; Symmetric);
    symbol!("partial_export_y");

    let a = parse!("partial_export_f(partial_export_y, partial_export_x)*partial_export_x^2");

    symbol!("partial_export_unused_after");

    let mut export = vec![];
    a.export(&mut export).unwrap();
    drop(a);

    unsafe { State::reset() };

    let a_rec = Atom::import(&mut export.as_slice(), None).unwrap();
    let expected =
        parse!("partial_export_f(partial_export_x, partial_export_y)*partial_export_x^2");
    assert_eq!(a_rec, expected);
    assert!(symbol!("partial_export_f").is_symmetric());

    let imported_names: Vec<_> = State::symbol_iter().map(|(_, name)| name).collect();
    assert!(
        !imported_names
            .iter()
            .any(|name| name.ends_with("partial_export_unused_before"))
    );
    assert!(
        !imported_names
            .iter()
            .any(|name| name.ends_with("partial_export_unused_between"))
    );
    assert!(
        !imported_names
            .iter()
            .any(|name| name.ends_with("partial_export_unused_after"))
    );
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
    partial_state_export();
}
