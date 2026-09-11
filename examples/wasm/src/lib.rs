//! Minimal browser-facing WASM example.
//!
//! From this example's directory, run: `wasm-pack build --target web`.

use std::collections::HashMap;

use symbolica::{
    atom::{Atom, AtomCore, Indeterminate},
    parser::ParseSettings,
    printer::PrintOptions,
};
use tinyjson::JsonValue;
use wasm_bindgen::prelude::*;

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

#[derive(Debug)]
struct PlaygroundRequest {
    expression: String,
    operation: String,
    variable: String,
    point: String,
    depth: i64,
    formats: Vec<String>,
}

/// Evaluate a small Symbolica playground request.
///
/// The input and output are JSON strings so JavaScript only needs one stable
/// entry point.
#[wasm_bindgen]
pub fn evaluate_json(request_json: &str) -> String {
    match evaluate_json_impl(request_json) {
        Ok(response) => response,
        Err(error) => error_response(error),
    }
}

fn evaluate_json_impl(request_json: &str) -> Result<String, String> {
    let request = parse_request(request_json)?;
    let expr = Atom::parse(&request.expression, "playground", ParseSettings::default())?;

    let result = match request.operation.as_str() {
        "simplify" => expr.cancel(),
        "expand" => expr.expand(),
        "factor" => expr.factor(),
        "derivative" => expr.derivative(parse_indeterminate(&request.variable)?),
        "series" => {
            let variable = parse_indeterminate(&request.variable)?;
            let point = Atom::parse(&request.point, "playground", ParseSettings::default())?;
            expr.series(variable, point, request.depth)
                .map_err(|error| error.to_string())?
                .to_atom()
        }
        "latex" | "typst" => expr,
        operation => return Err(format!("Unknown operation '{operation}'")),
    };

    let primary_format = match request.operation.as_str() {
        "latex" => "latex",
        "typst" => "typst",
        _ => "symbolica",
    };
    let mut outputs = HashMap::default();
    let mut formats = request.formats;
    if !formats.iter().any(|format| format == primary_format) {
        formats.push(primary_format.to_owned());
    }
    for format in formats {
        outputs.insert(
            format.clone(),
            JsonValue::from(format_atom(&result, &format)?),
        );
    }

    let result = outputs
        .get(primary_format)
        .and_then(JsonValue::get::<String>)
        .cloned()
        .unwrap_or_else(|| format_atom(&result, primary_format).unwrap_or_default());

    let mut response = HashMap::default();
    response.insert("ok".to_owned(), JsonValue::from(true));
    response.insert("result".to_owned(), JsonValue::from(result));
    response.insert("elapsed_ms".to_owned(), JsonValue::from(0.0));
    response.insert("outputs".to_owned(), JsonValue::from(outputs));
    response.insert(
        "diagnostics".to_owned(),
        JsonValue::from(Vec::<JsonValue>::new()),
    );

    JsonValue::from(response)
        .stringify()
        .map_err(|error| format!("Could not serialize response: {error}"))
}

fn parse_request(request_json: &str) -> Result<PlaygroundRequest, String> {
    let value: JsonValue = request_json
        .parse()
        .map_err(|error| format!("Could not parse request JSON: {error}"))?;
    let object = value
        .get::<HashMap<String, JsonValue>>()
        .ok_or_else(|| "Request JSON must be an object".to_owned())?;

    Ok(PlaygroundRequest {
        expression: string_field(object, "expression")?,
        operation: optional_string_field(object, "operation").unwrap_or_else(|| "simplify".into()),
        variable: optional_string_field(object, "variable").unwrap_or_else(|| "x".into()),
        point: optional_string_field(object, "point").unwrap_or_else(|| "0".into()),
        depth: optional_i64_field(object, "depth").unwrap_or(4),
        formats: optional_string_array_field(object, "formats")
            .unwrap_or_else(|| vec!["symbolica".into(), "latex".into(), "typst".into()]),
    })
}

fn string_field(object: &HashMap<String, JsonValue>, key: &str) -> Result<String, String> {
    optional_string_field(object, key).ok_or_else(|| format!("Missing string field '{key}'"))
}

fn optional_string_field(object: &HashMap<String, JsonValue>, key: &str) -> Option<String> {
    object.get(key)?.get::<String>().cloned()
}

fn optional_i64_field(object: &HashMap<String, JsonValue>, key: &str) -> Option<i64> {
    let number = *object.get(key)?.get::<f64>()?;
    number.is_finite().then_some(number.trunc() as i64)
}

fn optional_string_array_field(
    object: &HashMap<String, JsonValue>,
    key: &str,
) -> Option<Vec<String>> {
    let values = object.get(key)?.get::<Vec<JsonValue>>()?;
    Some(
        values
            .iter()
            .filter_map(|value| value.get::<String>().cloned())
            .collect(),
    )
}

fn parse_indeterminate(input: &str) -> Result<Indeterminate, String> {
    Atom::parse(input, "playground", ParseSettings::default())?.try_into()
}

fn format_atom(atom: &Atom, format: &str) -> Result<String, String> {
    let options = match format {
        "symbolica" => PrintOptions::file_no_namespace(),
        "latex" => PrintOptions::latex(),
        "typst" => PrintOptions::typst(),
        "mathematica" => PrintOptions::mathematica(),
        other => return Err(format!("Unknown output format '{other}'")),
    };
    Ok(atom.printer(options).to_string())
}

fn error_response(error: String) -> String {
    let mut response = HashMap::default();
    response.insert("ok".to_owned(), JsonValue::from(false));
    response.insert("error".to_owned(), JsonValue::from(error.clone()));
    response.insert("result".to_owned(), JsonValue::from(String::new()));
    response.insert("elapsed_ms".to_owned(), JsonValue::from(0.0));
    response.insert(
        "diagnostics".to_owned(),
        JsonValue::from(vec![JsonValue::from(error)]),
    );

    JsonValue::from(response)
        .stringify()
        .unwrap_or_else(|_| "{\"ok\":false,\"error\":\"Could not serialize error\"}".to_owned())
}
