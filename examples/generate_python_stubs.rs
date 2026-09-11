//! Generate binding metadata for review without overwriting the shipped stub.
//! cargo run --features python_stubgen --example generate_python_stubs -- /tmp/symbolica-stubs

#[cfg(feature = "python_stubgen")]
fn main() {
    let output = std::env::args_os()
        .nth(1)
        .expect("provide an output directory");
    let mut info = symbolica::api::python::stub_info().expect("gather Python binding metadata");
    info.python_root = output.into();
    info.generate().expect("write generated Python stubs");
}

#[cfg(not(feature = "python_stubgen"))]
fn main() {
    eprintln!("Enable --features python_stubgen to generate the Python stubs");
    std::process::exit(1);
}
