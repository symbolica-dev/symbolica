use std::io::Cursor;

use smartstring::SmartString;
use symbolica::{
    atom::{Atom, AtomCore, UserData},
    coefficient::Coefficient,
    domains::{
        finite_field::{FiniteFieldCore, Zp64},
        integer::Z,
        rational::Q,
    },
    parse,
    state::State,
    symbol,
};

fn finite_field_atom(prime: u64) -> Atom {
    let field = Zp64::new(prime);
    let element = field.to_element(5);
    Atom::num(Coefficient::from_finite_field(field, element))
}

fn user_data_resources() {
    unsafe { State::reset() };

    let owner = symbol!(
        "user_data_finite_field_owner",
        data = UserData::Atom(finite_field_atom(17))
    );
    let mut export = vec![];
    owner.to_atom().export(&mut export).unwrap();

    unsafe { State::reset() };
    let _padding = finite_field_atom(19);

    let imported = Atom::import(&mut export.as_slice(), None).unwrap();
    assert_eq!(
        imported.get_symbol().unwrap().get_data(),
        &UserData::Atom(finite_field_atom(17))
    );

    unsafe { State::reset() };

    let owner = symbol!(
        "user_data_finite_field_owner",
        data = UserData::Atom(finite_field_atom(17))
    );
    let mut state_export = vec![];
    State::export(&mut state_export).unwrap();
    let mut atom_export = vec![];
    owner.to_atom().as_view().write(&mut atom_export).unwrap();

    unsafe { State::reset() };
    let _padding = finite_field_atom(19);

    let state_map = State::import(&mut state_export.as_slice(), None).unwrap();
    let imported = Atom::import_with_map(&mut atom_export.as_slice(), &state_map).unwrap();
    assert_eq!(
        imported.get_symbol().unwrap().get_data(),
        &UserData::Atom(finite_field_atom(17))
    );

    unsafe { State::reset() };

    let polynomial = parse!("1 + user_data_coefficient_parameter")
        .to_rational_polynomial::<_, _, u16>(&Q, &Z, None);
    let owner = symbol!(
        "user_data_coefficient_owner",
        data = UserData::Atom(Atom::num(Coefficient::RationalPolynomial(polynomial)))
    );
    let mut export = vec![];
    owner.to_atom().export(&mut export).unwrap();

    unsafe { State::reset() };
    let _padding =
        parse!("1 + user_data_padding_parameter").to_rational_polynomial::<_, _, u16>(&Q, &Z, None);

    let imported = Atom::import(&mut export.as_slice(), None).unwrap();
    let expected = parse!("1 + user_data_coefficient_parameter")
        .to_rational_polynomial::<_, _, u16>(&Q, &Z, None);
    assert_eq!(
        imported.get_symbol().unwrap().get_data(),
        &UserData::Atom(Atom::num(Coefficient::RationalPolynomial(expected)))
    );
}

fn finite_field_offset() {
    unsafe { State::reset() };

    let source_field = Zp64::new(17);
    let source_element = source_field.to_element(5);
    let source = Atom::num(Coefficient::from_finite_field(source_field, source_element));
    let mut export = vec![];
    source.export(&mut export).unwrap();

    unsafe { State::reset() };
    let padding_field = Zp64::new(19);
    let padding_element = padding_field.to_element(5);
    let _padding = Atom::num(Coefficient::from_finite_field(
        padding_field,
        padding_element,
    ));

    let imported = Atom::import(&mut export.as_slice(), None).unwrap();
    let expected_field = Zp64::new(17);
    let expected_element = expected_field.to_element(5);
    let expected = Atom::num(Coefficient::from_finite_field(
        expected_field,
        expected_element,
    ));
    assert_eq!(imported, expected);
}

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

// All registry-reset scenarios stay in this integration test's single thread.
// The callback schedules registration exactly between a count and its payload,
// without relying on a probabilistic race between OS threads.
fn export_registry_snapshot() {
    use std::io::{self, Write};

    #[derive(Clone, Copy)]
    enum Registration {
        Symbol,
        FiniteField,
        Polynomial,
    }

    struct RegisteringWriter {
        bytes: Vec<u8>,
        offset: usize,
        registration: Registration,
        triggered: bool,
    }
    impl Write for RegisteringWriter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            let offset = self.bytes.len();
            self.bytes.extend_from_slice(bytes);
            if offset == self.offset {
                assert!(!self.triggered);
                self.triggered = true;
                match self.registration {
                    Registration::Symbol => {
                        symbol!("snapshot_registered_in_writer");
                    }
                    Registration::FiniteField => {
                        let _ = finite_field_atom(7);
                    }
                    Registration::Polynomial => {
                        let parameter = symbol!("snapshot_polynomial_parameter");
                        let _ = (parameter.to_atom() + 1).set_coefficient_ring(parameter);
                    }
                }
            }
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    for full_state in [false, true] {
        for registration in [
            Registration::Symbol,
            Registration::FiniteField,
            Registration::Polynomial,
        ] {
            unsafe { State::reset() };
            let atom = Atom::num(1);
            let mut control = Vec::new();
            if full_state {
                State::export(&mut control).unwrap();
            } else {
                atom.export(&mut control).unwrap();
            }
            let offset = if full_state {
                match registration {
                    Registration::Symbol => 7, // symbol count follows magic/version/flags
                    Registration::FiniteField => control.len() - 16,
                    Registration::Polynomial => control.len() - 8,
                }
            } else {
                match registration {
                    // New coefficient-variable symbols must not enter a partial
                    // export after its symbol-dependency closure was collected.
                    Registration::Symbol => 0,
                    Registration::FiniteField => 15,
                    Registration::Polynomial => 23,
                }
            };
            let mut writer = RegisteringWriter {
                bytes: Vec::new(),
                offset,
                registration: if !full_state && matches!(registration, Registration::Symbol) {
                    Registration::Polynomial
                } else {
                    registration
                },
                triggered: false,
            };
            if full_state {
                State::export(&mut writer).unwrap();
            } else {
                atom.export(&mut writer).unwrap();
            }
            assert!(writer.triggered);
            assert_eq!(writer.bytes, control);

            // Import against different identifiers; the serialized export must
            // be self-contained, independent of the writer's new registrations.
            unsafe { State::reset() };
            symbol!("snapshot_import_padding");
            let _ = finite_field_atom(19);
            let padding = symbol!("snapshot_import_polynomial_padding");
            let _ = (padding.to_atom() + 1).set_coefficient_ring(padding);
            let mut source = writer.bytes.as_slice();
            let imported = if full_state {
                let map = State::import(&mut source, None).unwrap();
                let mut atom_bytes = Vec::new();
                atom.as_view().write(&mut atom_bytes).unwrap();
                Atom::import_with_map(&mut atom_bytes.as_slice(), &map).unwrap()
            } else {
                Atom::import(&mut source, None).unwrap()
            };
            assert!(source.is_empty());
            assert_eq!(imported, Atom::num(1));
        }
    }
}

fn export_inside_symbol_generator() {
    use std::sync::{Arc, Mutex};
    use symbolica::{atom::SymbolBuilder, wrap_symbol};

    unsafe { State::reset() };
    let original = parse!("snapshot_existing^2 + 1");
    let mut expected_atom = Vec::new();
    original.export(&mut expected_atom).unwrap();
    let mut expected_state = Vec::new();
    State::export(&mut expected_state).unwrap();
    let output = Arc::new(Mutex::new(None));
    let callback_output = output.clone();
    let callback_atom = original.clone();
    let builder =
        SymbolBuilder::new(wrap_symbol!("snapshot_generated")).with_generator(move |_, builder| {
            let mut atom = Vec::new();
            callback_atom.export(&mut atom).unwrap();
            let mut state = Vec::new();
            State::export(&mut state).unwrap();
            *callback_output.lock().unwrap() = Some((atom, state));
            builder
        });
    SymbolBuilder::build_group(vec![builder]).unwrap();
    let (atom, state) = output.lock().unwrap().take().unwrap();
    assert_eq!(atom, expected_atom);
    assert_eq!(state, expected_state);
    let mut source = atom.as_slice();
    assert_eq!(Atom::import(&mut source, None).unwrap(), original);
    assert!(source.is_empty());
    let mut source = state.as_slice();
    State::import(&mut source, None).unwrap();
    assert!(source.is_empty());
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
    finite_field_offset();
    user_data_resources();
    export_registry_snapshot();
    export_inside_symbol_generator();
}
