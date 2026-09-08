//! Matched factored black boxes for the external FireFly/FIRE7 comparison.
use std::{sync::Arc, time::Instant};
use symbolica::{
    domains::finite_field::{FiniteFieldElement, Zp64},
    poly::{
        PolyVariable,
        reconstruction::{
            ReconstructionMethod::*, ReconstructionOptions, reconstruct_rational_function,
            reconstruct_rational_function_over_q,
        },
    },
    prelude::*,
};

fn oracle(f: &Zp64, x: &[FiniteFieldElement<u64>], large: bool) -> Option<FiniteFieldElement<u64>> {
    let c = |n: i64| f.nth(n.into());
    let (n, d) = if large {
        (
            f.add(
                &f.mul(
                    &f.pow(&f.add(&x[1], &c(13)), 30),
                    &f.pow(&f.add(&f.pow(&x[0], 2), &c(9)), 7),
                ),
                &c(1),
            ),
            f.mul(
                &f.pow(&f.sub(&x[1], &c(4)), 29),
                &f.pow(&f.sub(&f.pow(&x[0], 2), &c(1)), 5),
            ),
        )
    } else {
        let xy = f.mul(&x[0], &x[1]);
        (
            f.add(&xy, &c(2)),
            f.add(&f.sub(&xy, &f.mul(&c(2), &x[0])), &c(4)),
        )
    };
    (!f.is_zero(&d)).then(|| f.div(&n, &d))
}

fn main() {
    let repeats = std::env::var("RECONSTRUCTION_REPEATS")
        .map(|s| s.parse::<u64>().unwrap())
        .unwrap_or(9);
    let mode = std::env::args().nth(1).unwrap_or("ff".into());
    assert!(mode == "ff" || mode == "q");
    let field = Zp64::new(9_223_372_036_854_775_783); // FireFly's first prime.
    println!("case,method,mode,seed,elapsed_us,probes");
    for (name, num, den, names, large) in [
        ("paper_eq3", "x*y+2", "x*y-2*x+4", ["x", "y"], false),
        (
            "paper_eq28_y_d",
            "(d+13)^30*(y^2+9)^7+1",
            "(d-4)^29*(y^2-1)^5",
            ["y", "d"],
            true,
        ),
    ] {
        let vars: Arc<Vec<PolyVariable>> =
            Arc::new(names.iter().map(|s| symbol!(*s).into()).collect());
        let n: MultivariatePolynomial<_, u16> = parse!(num).to_polynomial(&Z, vars.clone());
        let d: MultivariatePolynomial<_, u16> = parse!(den).to_polynomial(&Z, vars.clone());
        for seed in 0..=repeats {
            let mut methods = Vec::from(if seed % 2 == 0 {
                [CuytLee, BalancedZippel]
            } else {
                [BalancedZippel, CuytLee]
            });
            if std::env::var_os("RECONSTRUCTION_SEPARATED").is_some() {
                methods.push(BalancedZippelSeparated);
                methods.rotate_left((seed % 3) as usize);
            }
            for method in methods {
                let opts = ReconstructionOptions {
                    seed,
                    max_degree: 64,
                    degree_race: std::env::var_os("RECONSTRUCTION_DEGREE_RACE").is_some(),
                    ..Default::default()
                };
                let start = Instant::now();
                let (elapsed, probes) = if mode == "q" {
                    let (r, s) = reconstruct_rational_function_over_q(
                        vars.clone(),
                        |f, x| oracle(f, x, large),
                        method,
                        &opts,
                        12,
                    )
                    .unwrap();
                    let elapsed = start.elapsed().as_secs_f64() * 1e6;
                    assert_eq!(&r.numerator * &d, &r.denominator * &n);
                    (elapsed, s.probes)
                } else {
                    let (r, s) = reconstruct_rational_function(
                        field.clone(),
                        vars.clone(),
                        |f, x| oracle(f, x, large),
                        method,
                        &opts,
                    )
                    .unwrap();
                    let elapsed = start.elapsed().as_secs_f64() * 1e6;
                    use symbolica::domains::finite_field::ToFiniteField;
                    assert_eq!(
                        &r.numerator * &d.map_coeff(|c| c.to_finite_field(&field), field.clone()),
                        &r.denominator * &n.map_coeff(|c| c.to_finite_field(&field), field.clone())
                    );
                    (elapsed, s.probes)
                };
                if seed > 0 {
                    println!("{name},{method:?},{mode},{seed},{elapsed:.3},{probes}");
                }
            }
        }
    }
}
