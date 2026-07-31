# Symbolica browser WebAssembly example

This standalone crate exposes a small JSON-based Symbolica API with
`wasm-bindgen`. Build it with `wasm-pack`, then serve this directory over HTTP:

```sh
rustup target add wasm32-unknown-unknown
wasm-pack build --target web
python -m http.server 8000
```

Open <http://localhost:8000>. [`index.html`](index.html) imports the generated
`pkg/symbolica_wasm.js` module and calls `evaluate_json`.
