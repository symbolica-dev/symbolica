<h1 align="center">
  <br>
  <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://symbolica.io/logo_dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="https://symbolica.io/logo.svg">
  <img src="https://symbolica.io/logo.svg" alt="logo" width="200">
</picture>
  <br>
</h1>

<p align="center">
<a href="https://symbolica.io"><img alt="Symbolica website" src="https://img.shields.io/static/v1?label=symbolica&message=website&color=orange&style=flat-square"></a>
  <a href="https://zulip.symbolica.io"><img alt="Zulip Chat" src="https://img.shields.io/static/v1?label=zulip&message=discussions&color=blue&style=flat-square"></a>
    <a href="https://github.com/symbolica-dev/symbolica"><img alt="Symbolica repository" src="https://img.shields.io/static/v1?label=github&message=development&color=green&style=flat-square&logo=github"></a>
    <a href="https://app.codecov.io/gh/symbolica-dev/symbolica"><img alt="Codecov" src="https://img.shields.io/codecov/c/github/symbolica-dev/symbolica?token=rhCESyNPk7&style=flat-square"></a>
</p>

# Symbolica ⊆ Modern Computer Algebra

Symbolica is a high-performance computer algebra library for Python and Rust. It
is built for large expressions, symbolic rewrites, exact polynomial arithmetic,
and optimized numerical evaluators.

Trusted by CERN and research groups at ETH Zurich, the University of Zurich, the
University of Bern, and Karlsruhe Institute of Technology.

Try the live [Jupyter Notebook demo](https://colab.research.google.com/drive/1VAtND2kddgBwNt1Tjsai8vnbVIbgg-7D?usp=sharing),
read the [documentation](https://symbolica.io/docs/), or see
[symbolica.io](https://symbolica.io) for licensing and support.

There is also a static browser playground in
[`docs/playground`](docs/playground/) for experimenting with the WASM build.
The website showcase sketch in [`docs/showcase`](docs/showcase/) demonstrates a
guided live-demo layout with syntax-highlighted Python examples.
The Pyodide browser session in [`docs/pyodide`](docs/pyodide/) runs the Python
API in browser Python.

## Why Symbolica?

- Native Python and Rust APIs for the same symbolic core
- Optimized numerical evaluators, with JIT, C++, SIMD, ASM, and CUDA code generation
- Fast multivariate polynomial arithmetic for large symbolic workloads
- Pattern matching and rewrites for domain-specific algebra
- Mixed exact and numerical computation with error propagation
- Streaming tools for expressions too large to keep in memory

# Installation

Visit the [Get Started](https://symbolica.io/docs/get_started.html) page for detailed installation instructions.

## Python

Symbolica can be installed from PyPI using `pip`:

```sh
pip install symbolica
```

### Pyodide

To build a browser-compatible Python wheel for Pyodide, install
`pyodide-build` and run:

```sh
pip install pyodide-build
rustup toolchain install nightly-2025-06-15 --profile minimal --target wasm32-unknown-emscripten
tools/build_pyodide_wheel.sh
```

This produces a `dist/symbolica-*-wasm32.whl` wheel that can be installed in
Pyodide with `micropip.install()`. The Pyodide build uses Symbolica's WASM
feature set, disables GMP and native code generation, and omits the Python APIs
that load generated C++/CUDA/JIT shared libraries. The helper defaults to the
non-release `pyodide-debug` profile for a quick smoke test; set
`SYMBOLICA_PYODIDE_PROFILE=release` for an optimized wheel.

To run the local Node/Pyodide smoke test, point it at a Pyodide checkout and the
wheel built for that runtime:

```sh
PYODIDE_ROOT=/path/to/pyodide-root \
SYMBOLICA_PYODIDE_WHEEL=target/wheels/symbolica-*-wasm32.whl \
node --experimental-wasm-stack-switching tools/smoke_pyodide_symbolica.mjs
```

To serve the browser session locally from this repository:

```sh
python3 -m http.server 8765
```

Then open <http://localhost:8765/docs/pyodide/>.
The page discovers the newest `symbolica-*-wasm32.whl` in `target/wheels/` and
runs a parse-and-expand example when the Pyodide session is ready.

### JupyterLite

To build a JupyterLite site with a starter Symbolica notebook, install the
JupyterLite tooling and run:

```sh
pip install 'jupyterlite-core==0.6.4' 'jupyterlite-pyodide-kernel==0.6.1' jupyterlab notebook
tools/build_jupyterlite.sh
```

Then open
<http://localhost:8765/docs/jupyterlite/_output/lab/index.html?path=Symbolica.ipynb>.
The helper bundles Pyodide 0.27.7 and the local Symbolica wheel into the
JupyterLite output so the notebook can run `pip install symbolica` in the
browser.

## Rust

If you want to use Symbolica as a library in Rust, simply include it:

```sh
cargo add symbolica
```

# Example

Here is one compact workflow that combines symbolic manipulation, series
expansion, replacement, solving a parameterized linear system, and numerical
evaluation. Check the [guide](https://symbolica.io/docs/) for a complete
overview.

### Pendulum calibration

Start with a pendulum whose restoring torque is controlled by the scale `κ`:

```python
from symbolica import *

θ, κ = S("θ", "κ")

V = κ*(1 - θ.cos())
τ = -V.derivative(θ)

τ
```

```math
-\kappa \sin\!\left(\theta\right)
```

Expand the torque to get a small-angle model:

```python
τ_small = τ.series(θ, 0, 3)
τ_small
```

```math
-\kappa\theta+\frac{1}{6}\kappa\theta^3+\mathcal{O}\!\left(\theta^4\right)
```

Suppose the scale `κ` and a sensor offset `τ_0` are unknown. Each pair
`(θ_i, τ_i)` is one sensor reading: at angle `θ_i`, the measured torque is
`τ_i`. Convert the truncated series back to an expression, evaluate it at two
measurement angles using `replace`, and solve the resulting linear system:

```python
τ0, τ1, τ2, θ1, θ2 = S("τ_0", "τ_1", "τ_2", "θ_1", "θ_2")

τ_model = τ_small.to_expression() + τ0

κ_fit, τ0_fit = Expression.solve_linear_system([
    τ_model.replace(θ, θ1) - τ1,
    τ_model.replace(θ, θ2) - τ2,
], [κ, τ0])

κ_fit
```

```math
\frac{6\tau_1-6\tau_2}{-6\theta_1+6\theta_2+\theta_1^3-\theta_2^3}
```

Finally, plug in measured values:

```python
κ_fit.evaluate({
    θ1: 0.10,
    θ2: 0.20,
    τ1: -0.4697,
    τ2: -0.9545,
}).real
```

```text
4.905227655986509
```

## Development

Follow the development of Symbolica and the open-source spin-off projects [numerica](https://github.com/symbolica-dev/numerica) and [graphica](https://github.com/symbolica-dev/graphica) on [Zulip](https://zulip.symbolica.io)!
