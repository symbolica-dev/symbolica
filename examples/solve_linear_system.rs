use std::sync::Arc;

use symbolica::{
    atom::{AtomCore, AtomView, representation::InlineVar},
    domains::{
        integer::Z,
        rational::Q,
        rational_polynomial::{RationalPolynomial, RationalPolynomialField},
    },
    parse,
    poly::PolyVariable,
    symbol,
    tensors::matrix::Matrix,
};

fn solve() {
    let x = symbol!("x").into();
    let y = symbol!("y").into();
    let z = symbol!("z").into();
    let eqs = ["c*x + f(c)*y + z - 1", "x + c*y + z/c - 2", "(c-1)x + c*z"];

    let system: Vec<_> = eqs.iter().map(|e| parse!(e)).collect();

    let sol = AtomView::solve_linear_system::<u8, _, InlineVar>(&system, &[x, y, z]).unwrap();

    for (v, s) in ["x", "y", "z"].iter().zip(&sol) {
        println!("{v} = {s}");
    }
}

fn solve_from_matrix() {
    let system = [["c", "c+1", "c^2+5"], ["1", "c", "c+1"], ["c-1", "-1", "c"]];
    let rhs = ["1", "2", "-1"];

    println!("Solve linear system in x\u{20D7}:");
    for (r, v) in system.iter().zip(&rhs) {
        println!("\t ({}).x\u{20D7} = {}", r.join(","), v);
    }

    let var_map = Arc::new(vec![PolyVariable::Symbol(symbol!("c"))]);

    let system_rat: Vec<RationalPolynomial<_, u8>> = system
        .iter()
        .flatten()
        .map(|s| parse!(s).to_rational_polynomial(&Q, &Z, Some(var_map.clone())))
        .collect();

    let rhs_rat: Vec<RationalPolynomial<_, u8>> = rhs
        .iter()
        .map(|s| parse!(s).to_rational_polynomial(&Q, &Z, Some(var_map.clone())))
        .collect();

    let field = RationalPolynomialField::from_poly(&rhs_rat[0].numerator);
    let m = Matrix::from_linear(
        system_rat,
        system.len() as u32,
        system.len() as u32,
        field.clone(),
    )
    .unwrap();
    let b = Matrix::new_vec(rhs_rat, field);

    match m.solve(&b) {
        Ok(sol) => {
            println!(
                "x\u{20D7} = {{{}}}",
                sol.row_iter()
                    .flatten()
                    .map(|r| format!("{r}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
        Err(e) => panic!("Could not solve {e:?}"),
    }
}

fn main() {
    solve();
    solve_from_matrix();
}
