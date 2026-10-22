//! Inline C++ forwards for the numeric implementations supplied by the standard library.

use super::*;

fn wrapper(name: &str, arguments: &str, body: &str) -> String {
    format!(
        "template<typename T>\n\
         #ifdef __CUDACC__\n\
         __host__ __device__\n\
         #endif\n\
         inline T {name}({arguments}) {{ {body} }}\n"
    )
}

fn unavailable(name: &str, reason: &str) -> String {
    wrapper(
        name,
        "T x",
        &format!("static_assert(sizeof(T) == 0, \"{reason}\"); return x;"),
    )
}

fn special(name: &str, source: String) -> String {
    format!(
        "#if defined(__cpp_lib_math_special_functions) && __cpp_lib_math_special_functions >= 201603L\n\
         {source}\
         #else\n\
         {}\
         #endif\n",
        unavailable(
            name,
            "This forward requires C++17 standard-library mathematical special functions"
        )
    )
}

pub(super) fn unary(symbol: &str, builtin: &str, argument: &str, reciprocal: bool) -> String {
    let name = format!("symbolica_{symbol}");
    let call = format!("{builtin}({argument})");
    let value = if reciprocal {
        format!("T(1) / {call}")
    } else {
        call
    };
    let source = wrapper(
        &name,
        "T x",
        &format!("using std::{builtin}; return {value};"),
    );
    if builtin == "riemann_zeta" {
        special(&name, source)
    } else {
        source
    }
}

pub(super) fn atan() -> String {
    let mut source = String::from("#include <complex>\n");
    source += &unary("atan", "atan", "x", false);
    source += &wrapper(
        "symbolica_atan",
        "T x, T y",
        "using std::atan2; return atan2(y, x);",
    );
    // C++ has no complex atan2. Match Symbolica's real-axis handling and its
    // logarithmic continuation for non-real arguments.
    for namespace in ["std", "cuda::std"] {
        if namespace == "cuda::std" {
            source += "#ifdef __CUDACC__\n#include <cuda/std/complex>\n";
        }
        source += &format!(
            "template<typename T>\n\
             #ifdef __CUDACC__\n__host__ __device__\n#endif\n\
             inline {namespace}::complex<T> symbolica_atan({namespace}::complex<T> x, {namespace}::complex<T> y) {{\n\
             using std::atan2; using {namespace}::log;\n\
             if (x.imag() == T(0) && y.imag() == T(0)) return {{atan2(y.real(), x.real()), T(0)}};\n\
             const {namespace}::complex<T> i(0, 1), one(1, 0), r = y / x;\n\
             return (log(one + i * r) - log(one - i * r)) / (T(2) * i);\n\
             }}\n"
        );
        if namespace == "cuda::std" {
            source += "#endif\n";
        }
    }
    source
}

pub(super) fn bessel(builtin: &str, name: &str, tags: &[AtomView]) -> String {
    let Ok(order) = complex_float_tag("Bessel C++ forward", tags, 53) else {
        return unavailable(name, "A Bessel C++ forward requires a numeric order");
    };
    let nu = order.re.to_f64();
    if !order.im.is_zero() || !nu.is_finite() {
        return unavailable(
            name,
            "A standard C++ Bessel function requires a finite real order",
        );
    }

    // The standard functions only accept nonnegative orders. Use the order
    // reflection identities before forwarding, including exact integer parity.
    let magnitude = format!("T({:e})", nu.abs());
    let call = format!("{builtin}({magnitude}, x)");
    let mut imports = format!("using std::{builtin};");
    let value = if nu >= 0.0 || builtin == "cyl_bessel_k" {
        call
    } else if nu.fract() == 0.0 {
        if matches!(builtin, "cyl_bessel_j" | "cyl_neumann") && nu % 2.0 != 0.0 {
            format!("-{call}")
        } else {
            call
        }
    } else {
        let angle = std::f64::consts::PI * nu;
        let cos = angle.cos();
        let sin = angle.sin();
        match builtin {
            "cyl_bessel_j" => {
                imports += " using std::cyl_neumann;";
                format!("T({cos:e}) * {call} + T({sin:e}) * cyl_neumann({magnitude}, x)")
            }
            "cyl_neumann" => {
                imports += " using std::cyl_bessel_j;";
                format!("T({cos:e}) * {call} - T({sin:e}) * cyl_bessel_j({magnitude}, x)")
            }
            "cyl_bessel_i" => {
                imports += " using std::cyl_bessel_k;";
                let coefficient = 2.0 * sin / std::f64::consts::PI;
                format!("{call} - T({coefficient:e}) * cyl_bessel_k({magnitude}, x)")
            }
            _ => unreachable!(),
        }
    };
    special(
        name,
        wrapper(name, "T x", &format!("{imports} return {value};")),
    )
}
