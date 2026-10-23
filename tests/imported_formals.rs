#![cfg(feature = "bincode")]

use std::{fs, process::Command};

use symbolica::{
    evaluate::{FunctionMap, FunctionRegistrationOptions, InliningPolicy},
    prelude::*,
    state::State,
};

#[cfg(feature = "serde")]
#[test]
fn symbols_deserialize_from_json_arrays() {
    let symbol = symbol!("imported_formals_json");
    let encoded = serde_json::to_string(&symbol).unwrap();
    let decoded: Symbol = serde_json::from_str(&encoded).unwrap();
    assert_eq!(symbol, decoded);
}

#[test]
fn imported_formals_rebuild_inline_symbol_identity() {
    const ROLE: &str = "SYMBOLICA_FORMAL_IMPORT_ROLE";
    const DIRECTORY: &str = "SYMBOLICA_FORMAL_IMPORT_DIRECTORY";
    let Ok(role) = std::env::var(ROLE) else {
        let directory =
            std::env::temp_dir().join(format!("symbolica-formal-import-{}", std::process::id()));
        fs::create_dir(&directory).unwrap();
        for role in ["export", "import"] {
            let status = Command::new(std::env::current_exe().unwrap())
                .args(["--exact", "imported_formals_rebuild_inline_symbol_identity"])
                .env(ROLE, role)
                .env(DIRECTORY, &directory)
                .status()
                .unwrap();
            assert!(status.success(), "fresh-process {role} failed");
        }
        fs::remove_dir_all(directory).unwrap();
        return;
    };
    let directory = std::path::PathBuf::from(std::env::var_os(DIRECTORY).unwrap());
    if role == "export" {
        let mut functions = FunctionMap::new();
        functions
            .add_aliases([(parse!("q"), parse!("y + 1"))])
            .unwrap();
        functions
            .add_tagged_function_with_options(
                symbol!("f"),
                vec![Atom::num(7)],
                vec![symbol!("z")],
                parse!("z^2 + q"),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap();
        let formals = vec![
            Indeterminate::from(symbol!("x")),
            Indeterminate::from(symbol!("z")),
            Indeterminate::try_from(parse!("g(x)")).unwrap(),
        ];
        let expression = parse!("f(7,x) + f(7,x+1)");
        let payload = bincode::encode_to_vec(
            (formals, functions, expression),
            bincode::config::standard(),
        )
        .unwrap();
        let mut state = Vec::new();
        State::export(&mut state).unwrap();
        fs::write(directory.join("state"), state).unwrap();
        fs::write(directory.join("payload"), payload).unwrap();
    } else {
        // Assign different IDs before importing, without resetting global state.
        for i in 0..100 {
            let _ = parse!(&format!("padding_{i}"));
        }
        let state = fs::read(directory.join("state")).unwrap();
        let mapping = State::import(&mut state.as_slice(), None).unwrap();
        let payload = fs::read(directory.join("payload")).unwrap();
        let ((formals, functions, expression), consumed): (
            (Vec<Indeterminate>, FunctionMap, Atom),
            _,
        ) = bincode::decode_from_slice_with_context(&payload, bincode::config::standard(), mapping)
            .unwrap();
        assert_eq!(consumed, payload.len());
        for (formal, expected) in formals
            .iter()
            .zip([parse!("x"), parse!("z"), parse!("g(x)")])
        {
            assert_eq!(formal.as_view(), expected.as_view());
            assert_eq!(
                formal.get_symbol(),
                expected.as_view().get_symbol().unwrap()
            );
        }
        let mut evaluator = expression
            .evaluator(&[parse!("x"), parse!("y")])
            .function_map(functions)
            .build()
            .unwrap()
            .map_coeff(&|coefficient| coefficient.re.to_f64());
        assert_eq!(evaluator.evaluate_single(&[2., 3.]), 21.);
    }
}
