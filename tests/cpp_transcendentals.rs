#![cfg(feature = "native_code_generation")]

use std::{path::PathBuf, process::Command};

use symbolica::{
    atom::{Atom, AtomCore, EvaluationInfo},
    domains::float::Complex,
    evaluate::{ComplexEvaluatorSettings, ExportNumber, ExportSettings, InlineASM},
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
    run_cpp_with_args(files, code, cases, &[]);
}

fn run_cpp_with_args(files: &TestFiles, code: &str, cases: &str, args: &[String]) {
    // A standalone executable also exercises compilers whose runtime cannot
    // safely be unloaded from a Rust test thread (notably GCC on macOS).
    std::fs::write(
        files.source(),
        format!("#include <vector>\n#include <algorithm>\n{code}\nint main() {{\n{cases}\n}}"),
    )
    .unwrap();
    let output = Command::new(compiler())
        .args(["-std=c++17", "-O2"])
        .args(args)
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

fn avx2_compile_args() -> Option<Vec<String>> {
    let mut args = vec!["-mavx2".to_owned()];
    if cfg!(target_arch = "x86_64") {
        #[cfg(target_arch = "x86_64")]
        if !std::is_x86_feature_detected!("avx2") {
            return None;
        }
    } else if cfg!(all(target_arch = "aarch64", target_os = "macos")) {
        // Rosetta can run the exported x86 kernels even when Rust runs on ARM.
        if !Command::new("/usr/bin/arch")
            .args(["-x86_64", "/usr/bin/true"])
            .status()
            .is_ok_and(|s| s.success())
        {
            return None;
        }
        args.extend(["-arch".to_owned(), "x86_64".to_owned()]);
    } else {
        return None;
    }
    let include = std::env::var_os("SYMBOLICA_TEST_XSIMD_INCLUDE");
    if let Some(include) = &include {
        args.extend(["-I".to_owned(), include.to_string_lossy().into_owned()]);
    }
    let output = Command::new(compiler())
        .args(&args)
        .args([
            "-std=c++17",
            "-fsyntax-only",
            "-x",
            "c++",
            "-include",
            "xsimd/xsimd.hpp",
            "-",
        ])
        .stdin(std::process::Stdio::null())
        .output()
        .unwrap();
    if !output.status.success() {
        assert!(
            include.is_none(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        eprintln!("Skipping SIMD execution: xsimd is unavailable for the selected compiler");
        return None;
    }
    Some(args)
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
fn reused_registers_preserve_repeated_operands() {
    let parameters = [
        parse!("x"),
        parse!("y"),
        parse!("z"),
        parse!("w"),
        parse!("a"),
    ];
    for (case, expression) in [
        "(x+y)*(z+w)+(x+y)*z+(x+y)*w+(x+y)*a",
        "(x+y)*(z+w)+(x+y)*z+(x+y)*w+(x+y)*z+(x+y)*w+(x+y)*a",
        "a*(x+y)*(x*z+y*z)",
        "a*(x+y)*(x*z+y*z)*(x*w+y*w)",
    ]
    .into_iter()
    .enumerate()
    {
        let evaluator = parse!(expression)
            .evaluator(&parameters)
            .direct_translation(false)
            .build()
            .unwrap()
            .map_coeff(&|c| c.re.to_f64());
        for (index, asm) in [InlineASM::None, InlineASM::default()]
            .into_iter()
            .enumerate()
        {
            let files = TestFiles::new(&format!("repeated_operands_{case}_{index}"));
            let code = evaluator
                .export_cpp_str::<f64>("repeated_operands", ExportSettings::new().inline_asm(asm))
                .unwrap();
            let mut cases = String::new();
            for (inputs, expected) in [
                ([1., 2., 3., 4., 5.], [57., 78., 135., 1620.][case]),
                ([2., -1., -3., 4., -2.], [0., 1., 6., 24.][case]),
            ] {
                let mut result = [0.];
                evaluator.clone().evaluate(&inputs, &mut result);
                assert_eq!(result, [expected]);
                cases += &cpp_case("double", "repeated_operands_realf64", &inputs, &[expected]);
            }
            run_cpp(&files, &code, &cases);
        }
    }
}

#[test]
fn avx2_reciprocal_reuses_its_input_register() {
    let Some(args) = avx2_compile_args() else {
        return;
    };
    let evaluator = parse!("1/(x+y)+z")
        .evaluator(&[parse!("x"), parse!("y"), parse!("z")])
        .direct_translation(false)
        .build()
        .unwrap();
    let code = evaluator
        .export_cpp_str::<wide::f64x4>(
            "reciprocal",
            ExportSettings::new().inline_asm(InlineASM::AVX2),
        )
        .unwrap();
    // Each lane has a different denominator; scalar division cannot pass this.
    let cases = r#"
        double x[] = {1, 2, 3, 4}, y[] = {2, 3, 4, 5}, z[] = {-1, 0, 1, 2};
        simd params[] = {simd::load_unaligned(x), simd::load_unaligned(y), simd::load_unaligned(z)};
        std::vector<simd> buffer(reciprocal_simd_realf64_get_buffer_len());
        simd result[1];
        reciprocal_simd_realf64(params, buffer.data(), result);
        double actual[4]; result[0].store_unaligned(actual);
        for (int i = 0; i < 4; ++i)
            if (std::abs(actual[i] - (1/(x[i]+y[i]) + z[i])) > 1e-12) return 1;
    "#;
    run_cpp_with_args(&TestFiles::new("avx2_reciprocal"), &code, cases, &args);
}

#[test]
fn complex_simd_multiplication_with_real_parameters() {
    let Some(args) = avx2_compile_args() else {
        return;
    };
    let parameters = ["x", "y", "z", "w", "a", "b", "c", "d"].map(|p| parse!(p));
    // Exercise both preloaded operands and the serial path for long products.
    let expressions = [
        parse!("x*y*z"),
        parse!("x*y*z*w*a*b"),
        parse!("x*y*z*w*a*b*c*d"),
    ];
    for real_count in [3, 8] {
        let mut evaluator = Atom::evaluator_multiple(&expressions, &parameters)
            .direct_translation(false)
            .build()
            .unwrap();
        evaluator
            .set_real_params(
                &(0..real_count).collect::<Vec<_>>(),
                ComplexEvaluatorSettings::default(),
            )
            .unwrap();
        let code = evaluator
            .export_cpp_str::<Complex<wide::f64x4>>(
                "product",
                ExportSettings::new().inline_asm(InlineASM::AVX2),
            )
            .unwrap();
        let cases = format!(
            r#"
            using C = std::complex<double>;
            C scalar[8][4]; simd params[8];
            for (int p = 0; p < 8; ++p) {{
                for (int lane = 0; lane < 4; ++lane)
                    scalar[p][lane] = C(p+lane+2, p < {real_count} ? 0 : lane+1);
                params[p] = simd::load_unaligned(scalar[p]);
            }}
            std::vector<simd> buffer(product_simd_complexf64_get_buffer_len());
            simd result[3]; product_simd_complexf64(params, buffer.data(), result);
            const int lengths[] = {{3,6,8}};
            for (int output = 0; output < 3; ++output) {{
                C actual[4]; result[output].store_unaligned(actual);
                for (int lane = 0; lane < 4; ++lane) {{
                    C expected(1);
                    for (int p = 0; p < lengths[output]; ++p) expected *= scalar[p][lane];
                    if (std::abs(actual[lane]-expected) > 1e-12*std::abs(expected)) return 1;
                }}
            }}
        "#
        );
        run_cpp_with_args(
            &TestFiles::new(&format!("complex_simd_product_{real_count}")),
            &code,
            &cases,
            &args,
        );
    }
}

#[test]
fn complex_simd_accepts_complex_coefficients() {
    let args = avx2_compile_args();
    let expressions = [
        parse!("1i*x"),
        parse!("(1/3+2/7*1i)*x"),
        parse!("1i"),
        parse!("3/5"),
    ];
    for direct in [true, false] {
        let evaluator = Atom::evaluator_multiple(&expressions, &[parse!("x")])
            .direct_translation(direct)
            .build()
            .unwrap();
        // Export must work even on machines without SIMD execution dependencies.
        for (index, asm) in [InlineASM::None, InlineASM::AVX2].into_iter().enumerate() {
            let code = evaluator
                .export_cpp_str::<Complex<wide::f64x4>>(
                    "constants",
                    ExportSettings::new().inline_asm(asm),
                )
                .unwrap();
            if let Some(args) = &args {
                let cases = r#"
                    using C = std::complex<double>;
                    C input[] = {C(2,3), C(-1,2), C(0,-4), C(5,0)};
                    simd params[] = {simd::load_unaligned(input)}, result[4];
                    std::vector<simd> buffer(constants_simd_complexf64_get_buffer_len());
                    constants_simd_complexf64(params, buffer.data(), result);
                    for (int output = 0; output < 4; ++output) {
                        C actual[4]; result[output].store_unaligned(actual);
                        for (int lane = 0; lane < 4; ++lane) {
                            C expected[] = {C(0,1)*input[lane], C(1./3.,2./7.)*input[lane], C(0,1), C(3./5.,0)};
                            if (std::abs(actual[lane]-expected[output]) > 1e-12) return 1;
                        }
                    }
                "#;
                run_cpp_with_args(
                    &TestFiles::new(&format!("complex_simd_constants_{direct}_{index}")),
                    &code,
                    cases,
                    args,
                );
            }
        }
    }
}

const COMPLEX_RECIPROCAL_CASES: &str = r#"
    using C = std::complex<double>;
    C inputs[] = {C(1e200,1e200), C(1e-200,1e-200), C(1e308,1e308), C(1e-308,1e-308),
                  C(1e200,0), C(0,1e200), C(1e200,1e100), C(1e-200,1e-100), C(-1e200,1e200), C(3,4),
                  C(INFINITY,1), C(1,INFINITY), C(NAN,1), C(0,0)};
    C expected[] = {C(5e-201,-5e-201), C(5e199,-5e199), C(5e-309,-5e-309), C(5e307,-5e307),
                    C(1e-200,0), C(0,-1e-200), C(1e-200,-1e-300), C(1,-1e100), C(-5e-201,-5e-201), C(.12,-.16),
                    C(NAN,0), C(0,NAN), C(NAN,NAN), C(NAN,NAN)};
    constexpr int count = sizeof(inputs)/sizeof(inputs[0]);
    auto close = [](double a, double b) { return std::isnan(b) ? std::isnan(a) : b == 0 ? a == 0 : std::isfinite(a) && std::abs((a-b)/b) < 2e-12; };
"#;

#[test]
fn complex_reciprocals_avoid_norm_overflow_and_underflow() {
    use symbolica::evaluate::{FunctionRegistrationOptions, InliningPolicy};

    let evaluator = Atom::evaluator_multiple(
        &[parse!("1/x"), parse!("complex_inverse_regression(x)")],
        &[parse!("x")],
    )
    .add_function_with_options(
        symbol!("complex_inverse_regression"),
        vec![symbol!("x")],
        parse!("1/x"),
        FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
    )
    .unwrap()
    .build()
    .unwrap();
    let avx2_args = avx2_compile_args();
    let mut targets = vec![(InlineASM::default(), Vec::new())];
    if InlineASM::default() != InlineASM::X64
        && let Some(args) = &avx2_args
    {
        targets.push((InlineASM::X64, args.clone()));
    }
    for (index, (asm, args)) in targets.into_iter().enumerate() {
        let code = evaluator
            .export_cpp_str::<Complex<f64>>("inverse", ExportSettings::new().inline_asm(asm))
            .unwrap();
        let cases = COMPLEX_RECIPROCAL_CASES.to_owned()
            + r#"
            std::vector<C> buffer(inverse_complexf64_get_buffer_len());
            for (int i = 0; i < count; ++i) {
                C result[2]; inverse_complexf64(&inputs[i], buffer.data(), result);
                for (const auto& value : result) if (!close(value.real(),expected[i].real()) || !close(value.imag(),expected[i].imag())) {
                    std::cerr << inputs[i] << ": " << value << " expected " << expected[i] << std::endl; return 1;
                }
            }
        "#;
        run_cpp_with_args(
            &TestFiles::new(&format!("complex_inverse_{index}")),
            &code,
            &cases,
            &args,
        );
    }
    if let Some(args) = avx2_args {
        let code = evaluator
            .export_cpp_str::<Complex<wide::f64x4>>(
                "inverse",
                ExportSettings::new().inline_asm(InlineASM::AVX2),
            )
            .unwrap();
        let cases = COMPLEX_RECIPROCAL_CASES.to_owned()
            + r#"
            std::vector<simd> buffer(inverse_simd_complexf64_get_buffer_len());
            for (int start = 0; start < count; ++start) {
                C lane_inputs[4]; for (int i = 0; i < 4; ++i) lane_inputs[i] = inputs[(start+i)%count];
                simd params[] = {simd::load_unaligned(lane_inputs)}, result[2];
                inverse_simd_complexf64(params, buffer.data(), result);
                for (const auto& value : result) {
                    C actual[4]; value.store_unaligned(actual);
                    for (int i = 0; i < 4; ++i) {
                        C e = expected[(start+i)%count];
                        if (!close(actual[i].real(),e.real()) || !close(actual[i].imag(),e.imag())) return 1;
                    }
                }
            }
        "#;
        run_cpp_with_args(
            &TestFiles::new("complex_simd_inverse"),
            &code,
            &cases,
            &args,
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
fn automatically_dualized_transcendentals_export() {
    use symbolica::{
        domains::{dual::HyperDual, rational::Rational},
        evaluate::Dualizer,
    };
    let expressions = geometric_expressions();
    let evaluator = Atom::evaluator_multiple(&expressions, &[parse!("x"), parse!("y")])
        .build()
        .unwrap()
        .vectorize(&Dualizer::new(
            HyperDual::<Complex<Rational>>::new(vec![vec![0], vec![1], vec![2]]),
            vec![],
        ))
        .unwrap();
    let mut real = evaluator.clone().map_coeff(&|c| c.re.to_f64());
    let inputs = [0.25, 1., 0., -1., 0.5, 0.];
    let mut expected = vec![0.; expressions.len() * 3];
    real.evaluate(&inputs, &mut expected);
    let code = real
        .export_cpp_str::<f64>(
            "dual_test",
            ExportSettings::default().inline_asm(InlineASM::None),
        )
        .unwrap();
    run_cpp(
        &TestFiles::new("dual_real"),
        &code,
        &cpp_case("double", "dual_test_realf64", &inputs, &expected),
    );

    let mut complex = evaluator.map_coeff(&|c| Complex::new(c.re.to_f64(), c.im.to_f64()));
    let inputs = [
        Complex::new(0.25, 0.1),
        Complex::new(1., 0.2),
        Complex::new(0., 0.),
        Complex::new(-1., 0.2),
        Complex::new(0.5, -0.1),
        Complex::new(0., 0.),
    ];
    let mut expected = vec![Complex::new(0., 0.); expressions.len() * 3];
    complex.evaluate(&inputs, &mut expected);
    let code = complex
        .export_cpp_str::<Complex<f64>>(
            "dual_test",
            ExportSettings::default().inline_asm(InlineASM::None),
        )
        .unwrap();
    run_cpp(
        &TestFiles::new("dual_complex"),
        &code,
        &cpp_case(
            "std::complex<double>",
            "dual_test_complexf64",
            &inputs,
            &expected,
        ),
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
fn complex_constants_compile_in_both_translation_modes() {
    let expressions = [
        parse!("1i*x"),
        parse!("(1/3+2/7*1i)*x"),
        parse!("2/5-3/11*1i"),
        parse!("1i"),
        parse!("3/5"),
    ];
    let inputs = [Complex::new(2., 3.)];
    let expected = [
        Complex::new(-3., 2.),
        Complex::new(-4. / 21., 11. / 7.),
        Complex::new(2. / 5., -3. / 11.),
        Complex::new(0., 1.),
        Complex::new(3. / 5., 0.),
    ];

    for direct in [true, false] {
        let evaluator = Atom::evaluator_multiple(&expressions, &[parse!("x")])
            .direct_translation(direct)
            .build()
            .unwrap();
        let floating = evaluator
            .clone()
            .map_coeff(&|c| Complex::new(c.re.to_f64(), c.im.to_f64()));
        let mut interpreted = vec![Complex::new(0., 0.); expressions.len()];
        floating.clone().evaluate(&inputs, &mut interpreted);
        for (actual, expected) in interpreted.iter().zip(&expected) {
            assert!((actual.re - expected.re).abs() < 1e-14);
            assert!((actual.im - expected.im).abs() < 1e-14);
        }

        // Exercise both Python's floating constants and the exact rational
        // exporter: unwrapped "1/3" would silently perform integer division.
        let settings = ExportSettings::default().inline_asm(InlineASM::None);
        for (kind, code) in [
            (
                "rational",
                evaluator
                    .export_cpp_str::<Complex<f64>>("constants", settings.clone())
                    .unwrap(),
            ),
            (
                "floating",
                floating
                    .export_cpp_str::<Complex<f64>>("constants", settings)
                    .unwrap(),
            ),
        ] {
            run_cpp(
                &TestFiles::new(&format!("complex_constants_{direct}_{kind}")),
                &code,
                &cpp_case(
                    "std::complex<double>",
                    "constants_complexf64",
                    &inputs,
                    &expected,
                ),
            );
        }
    }
}

#[test]
fn complex_constants_preserve_custom_wrapper_precision() {
    use symbolica::domains::rational::Rational;

    // This real part differs from one beyond double precision. Component
    // construction must retain the requested wrapper and exact divisions.
    let constant = Complex::new(
        Rational::from((1_152_921_504_606_846_977_i64, 1_152_921_504_606_846_976_i64)),
        Rational::from((1, 7)),
    );
    let exported = constant.export_wrapped_with("PreciseComplex");
    let code = "#include <complex>\n#include <limits>\n#include <cmath>\n\
                using PreciseComplex = std::complex<long double>;\n";
    let cases = format!(
        "const PreciseComplex value = {exported};\n\
         if (!(std::abs(value.imag() - 1.L/7.L) < 1e-18L)) return 1;\n\
         if (std::numeric_limits<long double>::digits > 60 &&\n\
             value.real() - 1.L != std::ldexp(1.L, -60)) return 2;\n"
    );
    run_cpp(&TestFiles::new("complex_custom_wrapper"), code, &cases);
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
