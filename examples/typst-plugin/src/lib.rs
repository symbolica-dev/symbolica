//! Minimal Typst plugin that proves Symbolica links against the raw WASM ABI.
//!
//! From this example's directory, run:
//! `cargo build --release --target wasm32-unknown-unknown`.

use std::str;

use symbolica::{
    atom::{Atom, AtomCore},
    parser::ParseSettings,
    printer::PrintOptions,
};
use wasm_minimal_protocol::*;

initiate_protocol!();

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
#[unsafe(no_mangle)]
unsafe extern "Rust" fn __getrandom_v03_custom(
    dest: *mut u8,
    len: usize,
) -> Result<(), getrandom::Error> {
    if dest.is_null() && len != 0 {
        return Err(getrandom::Error::new_custom(1));
    }

    let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ len as u64;
    for index in 0..len {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        unsafe { dest.add(index).write((state >> 56) as u8) };
    }

    Ok(())
}

/// Parse and expand an expression, returning Symbolica syntax.
#[wasm_func]
pub fn expand(expression: &[u8]) -> Result<Vec<u8>, String> {
    let expression =
        str::from_utf8(expression).map_err(|error| format!("Expression is not UTF-8: {error}"))?;
    let atom = Atom::parse(expression, "typst", ParseSettings::default())?;
    Ok(atom
        .expand()
        .printer(PrintOptions::file_no_namespace())
        .to_string()
        .into_bytes())
}

/// Parse an expression and return its Typst representation.
#[wasm_func]
pub fn to_typst(expression: &[u8]) -> Result<Vec<u8>, String> {
    let expression =
        str::from_utf8(expression).map_err(|error| format!("Expression is not UTF-8: {error}"))?;
    let atom = Atom::parse(expression, "typst", ParseSettings::default())?;
    Ok(atom.printer(PrintOptions::typst()).to_string().into_bytes())
}
