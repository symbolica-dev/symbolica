#let symbolica = plugin("target/wasm32-unknown-unknown/release/symbolica_typst_plugin.wasm")

#let expand(expression) = str(symbolica.expand(bytes(expression)))
#let symbolica-math(expression) = {
  let source = str(symbolica.to_typst(bytes(expression)))
  eval(source, mode: "math")
}

= Symbolica Typst plugin

This document calls Symbolica from a Typst WebAssembly plugin. For example,
expanding `"(x+1)^4"` gives:

#let expanded = expand("(x+1)^4")
#block(inset: 8pt, fill: luma(240), radius: 3pt, raw(expanded))

The second exported function converts Symbolica syntax directly into Typst
math content:

$ #symbolica-math(expanded) $

It also handles expressions that were not expanded first:

$ #symbolica-math("sqrt(2)+x^2/(x+1)") $
