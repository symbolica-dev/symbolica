#![cfg(all(feature = "native_code_generation", feature = "serde"))]

use symbolica::{
    atom::{Atom, AtomCore},
    domains::float::Complex,
    evaluate::{JITCompilationSettings, JITCompiledEvaluator},
    parse,
};

#[test]
fn real_jit_serde_roundtrip() {
    let exact = Atom::evaluator_multiple(
        &[parse!("x^2+y"), parse!("x-y^2")],
        &[parse!("x"), parse!("y")],
    )
    .build()
    .unwrap();
    let jit = exact
        .jit_compile::<f64>(JITCompilationSettings::default())
        .unwrap();
    let bytes = bincode::serde::encode_to_vec(&jit, bincode::config::standard()).unwrap();
    drop(jit);
    drop(exact);
    let (mut restored, used): (JITCompiledEvaluator<f64>, _) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
    assert_eq!(used, bytes.len());
    for (inputs, expected) in [([2., 3.], [7., -7.]), ([-1., 2.], [3., -5.])] {
        let mut outputs = [0.; 2];
        restored.evaluate(&inputs, &mut outputs);
        assert_eq!(outputs, expected);
    }
}

#[test]
fn complex_jit_serde_roundtrip() {
    let exact = Atom::evaluator_multiple(
        &[parse!("x^2+y"), parse!("x-y^2")],
        &[parse!("x"), parse!("y")],
    )
    .build()
    .unwrap();
    let jit = exact
        .jit_compile::<Complex<f64>>(JITCompilationSettings::default())
        .unwrap();
    let bytes = bincode::serde::encode_to_vec(&jit, bincode::config::standard()).unwrap();
    drop(jit);
    drop(exact);
    let (mut restored, used): (JITCompiledEvaluator<Complex<f64>>, _) =
        bincode::serde::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
    assert_eq!(used, bytes.len());
    for (x, y) in [
        (Complex::new(2., 1.), Complex::new(3., -2.)),
        (Complex::new(-1., 2.), Complex::new(2., 3.)),
    ] {
        let mut outputs = [Complex::new(0., 0.); 2];
        restored.evaluate(&[x, y], &mut outputs);
        assert_eq!(outputs, [x * x + y, x - y * y]);
    }
}
