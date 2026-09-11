#![cfg(feature = "native_code_generation")]

use std::{path::PathBuf, process::Command};

use symbolica::{
    atom::{Atom, AtomCore, EvaluationInfo},
    domains::float::Complex,
    evaluate::{ExportNumber, ExportSettings, InlineASM},
    parse, symbol,
};

struct TestFiles(PathBuf);

impl TestFiles {
    fn new(name: &str) -> Self {
        let directory = std::env::temp_dir().join(format!(
            "symbolica_cpp_transcendentals_{}_{name}",
            std::process::id()
        ));
        std::fs::create_dir(&directory).unwrap();
        Self(directory)
    }

    fn source(&self) -> PathBuf {
        self.0.join("evaluator.cpp")
    }

    fn executable(&self) -> PathBuf {
        self.0
            .join(format!("evaluator{}", std::env::consts::EXE_SUFFIX))
    }
}

impl Drop for TestFiles {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn compiler() -> String {
    std::env::var("SYMBOLICA_TEST_CXX").unwrap_or_else(|_| "g++".into())
}

fn run_cpp(files: &TestFiles, code: &str, cases: &str) {
    // A standalone executable also exercises compilers whose runtime cannot
    // safely be unloaded from a Rust test thread (notably GCC on macOS).
    std::fs::write(
        files.source(),
        format!("#include <vector>\n#include <algorithm>\n{code}\nint main() {{\n{cases}\n}}"),
    )
    .unwrap();
    let output = Command::new(compiler())
        .args(["-std=c++17", "-O2"])
        .arg(files.source())
        .arg("-o")
        .arg(files.executable())
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = Command::new(files.executable()).output().unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn cpp_case<T: ExportNumber>(kind: &str, function: &str, inputs: &[T], expected: &[T]) -> String {
    let values = |values: &[T]| {
        values
            .iter()
            .map(|x| format!("{kind}({})", x.export()))
            .collect::<Vec<_>>()
            .join(", ")
    };
    format!(
        "{{\n\
         std::vector<{kind}> params = {{{}}}, expected = {{{}}};\n\
         std::vector<{kind}> result(expected.size()), buffer({function}_get_buffer_len());\n\
         {function}(params.data(), buffer.data(), result.data());\n\
         for (size_t i = 0; i < result.size(); ++i) {{\n\
         if (!(std::abs(result[i] - expected[i]) < 1e-10 * std::max(1., std::abs(expected[i])))) {{\n\
         std::cerr << \"Output \" << i << \": expected \" << expected[i] << \", got \" << result[i] << std::endl; return 1;\n\
         }}\n}}\n}}\n",
        values(inputs),
        values(expected)
    )
}

fn geometric_expressions() -> Vec<Atom> {
    [
        "tan(x)",
        "cot(x)",
        "sec(x)",
        "csc(x)",
        "asin(x)",
        "acos(x)",
        "atan(x)",
        "atan(x,y)",
        "acot(x)",
        "asec(x+2)",
        "acsc(x+2)",
        "sinh(x)",
        "cosh(x)",
        "tanh(x)",
        "coth(x)",
        "sech(x)",
        "csch(x)",
        "asinh(x)",
        "acosh(x+2)",
        "atanh(x)",
        "acoth(x+2)",
        "asech(x)",
        "acsch(x)",
        // These already use dedicated evaluator instructions.
        "sin(x)",
        "cos(x)",
        "exp(x)",
        "log(x)",
        "sqrt(x)",
    ]
    .iter()
    .map(|expression| parse!(expression))
    .collect()
}

fn check_real(name: &str, expressions: &[Atom], parameters: &[Atom], inputs: &[f64]) {
    let evaluator = Atom::evaluator_multiple(expressions, parameters)
        .build()
        .unwrap()
        .map_coeff(&|c| c.re.to_f64());
    let mut expected = vec![0.; expressions.len()];
    evaluator.clone().evaluate(inputs, &mut expected);

    for (index, asm) in [InlineASM::None, InlineASM::default()]
        .into_iter()
        .enumerate()
    {
        let files = TestFiles::new(&format!("{name}_{index}"));
        let code = evaluator
            .export_cpp_str::<f64>("forward_test", ExportSettings::default().inline_asm(asm))
            .unwrap();
        run_cpp(
            &files,
            &code,
            &cpp_case("double", "forward_test_realf64", inputs, &expected),
        );
    }
}

#[test]
fn real_transcendental_forwards() {
    let mut expressions = geometric_expressions();
    expressions.extend([parse!("gamma(x)"), parse!("erf(x)")]);
    check_real(
        "real",
        &expressions,
        &[parse!("x"), parse!("y")],
        &[0.25, -1.],
    );
}

#[test]
fn complex_transcendental_forwards() {
    let expressions = geometric_expressions();
    let evaluator = Atom::evaluator_multiple(&expressions, &[parse!("x"), parse!("y")])
        .build()
        .unwrap()
        .map_coeff(&|c| Complex::new(c.re.to_f64(), c.im.to_f64()));
    for (index, asm) in [InlineASM::None, InlineASM::default()]
        .into_iter()
        .enumerate()
    {
        let files = TestFiles::new(&format!("complex_{index}"));
        let code = evaluator
            .export_cpp_str::<Complex<f64>>(
                "forward_test",
                ExportSettings::default().inline_asm(asm),
            )
            .unwrap();
        let mut cases = String::new();
        for inputs in [
            [Complex::new(0.25, 0.1), Complex::new(-1., 0.2)],
            [Complex::new(0.25, 0.), Complex::new(-1., 0.)],
        ] {
            let mut expected = vec![Complex::new(0., 0.); expressions.len()];
            evaluator.clone().evaluate(&inputs, &mut expected);
            cases += &cpp_case(
                "std::complex<double>",
                "forward_test_complexf64",
                &inputs,
                &expected,
            );
        }
        run_cpp(&files, &code, &cases);
    }
}

#[test]
fn cpp17_special_function_forwards() {
    // libc++ does not provide the optional C++17 mathematical special functions.
    // Set SYMBOLICA_TEST_CXX to a compiler using libstdc++ to exercise these.
    let macros = Command::new(compiler())
        .args([
            "-std=c++17",
            "-dM",
            "-E",
            "-x",
            "c++",
            "-include",
            "cmath",
            "-",
        ])
        .stdin(std::process::Stdio::null())
        .output()
        .unwrap();
    assert!(macros.status.success());
    if !String::from_utf8_lossy(&macros.stdout).contains("__cpp_lib_math_special_functions") {
        eprintln!(
            "Skipping C++17 special functions: the selected standard library does not provide them"
        );
        return;
    }

    let mut expressions = vec![parse!("zeta(x+2)")];
    for function in ["bessel_j", "bessel_y", "bessel_i", "bessel_k"] {
        for order in ["0", "2", "1/2", "-1", "-2", "-1/2"] {
            expressions.push(parse!(&format!("{function}({order},x)")));
        }
    }
    check_real("special", &expressions, &[parse!("x")], &[1.25]);
}

#[test]
fn generated_snippets_use_the_exported_name_and_tags() {
    let _ = symbol!(
        "cpp_tagged_forward",
        eval = EvaluationInfo::new()
            .with_tags(1)
            .with_cpp_generator(|name, tags| {
                let tag = f64::try_from(tags[0]).unwrap();
                format!("inline double {name}(double x) {{ return x + {tag:e}; }}")
            })
            .register_tagged(|tags| {
                let tag = f64::try_from(tags[0]).unwrap();
                Box::new(move |args: &[f64]| args[0] + tag)
            })
    );
    check_real(
        "tagged",
        &[
            parse!("cpp_tagged_forward(1/2,x)"),
            parse!("cpp_tagged_forward(-1,x)"),
            parse!("cpp_tagged_forward(1/2,x)+cpp_tagged_forward(1/2,x+1)"),
        ],
        &[parse!("x")],
        &[0.25],
    );
}
