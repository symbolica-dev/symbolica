# Symbolica Typst plugin example

This standalone crate exports two functions through Typst's minimal WebAssembly
plugin protocol:

- `expand` parses and expands a Symbolica expression.
- `to_typst` converts a Symbolica expression to Typst math syntax.

Build the plugin and compile the example document from this directory:

```sh
rustup target add wasm32-unknown-unknown
cargo build --release --target wasm32-unknown-unknown
typst compile example.typ
```

The plugin is loaded directly from
`target/wasm32-unknown-unknown/release/symbolica_typst_plugin.wasm` by
[`example.typ`](example.typ).
