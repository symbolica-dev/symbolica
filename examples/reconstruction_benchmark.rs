//! Reproducible comparison of the two in-tree reconstruction implementations.
//! Run `cargo run --release --example reconstruction_benchmark -- [case-filter]`.
//! Optional environment: RECONSTRUCTION_REPEATS (default 5), PROBE_WORK (default 0).
//! PROBE_WORK adds that many dependent finite-field multiplications per oracle
//! call, solely as a controlled model of expensive probes; it is not an IBP solve.
use std::{hint::black_box, sync::Arc, time::Instant};
use symbolica::{
    domains::finite_field::Zp64,
    poly::{
        PolyVariable,
        reconstruction::{
            ReconstructionMethod::*, ReconstructionOptions, reconstruct_rational_function,
        },
    },
    prelude::*,
};

fn main() {
    let repeats = std::env::var("RECONSTRUCTION_REPEATS")
        .map(|s| s.parse::<u64>().unwrap())
        .unwrap_or(5);
    let probe_work = std::env::var("PROBE_WORK")
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(0);
    let filter = std::env::args().nth(1).unwrap_or_default();
    let field = Zp64::new(2_305_843_009_213_693_951);
    let cases = [
        ("paper_eq3", "x*y+2", "x*y-2*x+4", vec!["x", "y"]),
        ("firefly_shift", "3*x+7*y", "x+y+4*x*y", vec!["x", "y"]),
        (
            "sparse3",
            "x^8*y^2+3*y^5*z+7*z^9+11",
            "x^3*z^5+2*y^7+5",
            vec!["x", "y", "z"],
        ),
        (
            "sparse4",
            "x^4*y^2+3*y^5*z+7*z^3*w^2+11",
            "x^3*w^2+2*y*z^3+5",
            vec!["x", "y", "z", "w"],
        ),
        (
            "homogeneous_shift",
            "x^4+y^4+z^4",
            "x*y+y*z+x*z",
            vec!["x", "y", "z"],
        ),
        (
            "dense_total3",
            "(x+2*y+3*z+5)^4+7*x*y",
            "(2*x+3*y+z+7)^3+11*z",
            vec!["x", "y", "z"],
        ),
        (
            "dense_box3",
            "(x+2)^3*(y+3)^3*(z+5)^3+1",
            "(x-3)^2*(y-5)^2*(z-7)^2",
            vec!["x", "y", "z"],
        ),
        (
            "paper_eq28_y_d",
            "(d+13)^30*(y^2+9)^7+1",
            "(d-4)^29*(y^2-1)^5",
            vec!["y", "d"],
        ),
        (
            "paper_eq28_d_y",
            "(d+13)^30*(y^2+9)^7+1",
            "(d-4)^29*(y^2-1)^5",
            vec!["d", "y"],
        ),
    ];
    println!(
        "case,method,seed,probe_work,elapsed_us,probes,poles,attempts,thiele,linear_solves,num_terms,den_terms"
    );
    for (name, num, den, names) in cases {
        if !name.contains(&filter) {
            continue;
        }
        let vars: Arc<Vec<PolyVariable>> =
            Arc::new(names.iter().map(|s| symbol!(*s).into()).collect());
        let n: MultivariatePolynomial<_, u16> = parse!(num).to_polynomial(&field, vars.clone());
        let d: MultivariatePolynomial<_, u16> = parse!(den).to_polynomial(&field, vars.clone());
        // One warm-up per method, then identical seeds in alternating method order.
        for run in 0..=repeats {
            let methods = if run % 2 == 0 {
                [CuytLee, BalancedZippel]
            } else {
                [BalancedZippel, CuytLee]
            };
            for method in methods {
                let options = ReconstructionOptions {
                    seed: run,
                    max_degree: 64,
                    ..Default::default()
                };
                let start = Instant::now();
                let (result, stats) = reconstruct_rational_function(
                    field.clone(),
                    vars.clone(),
                    |f, p| {
                        let mut cost = p[0];
                        for _ in 0..probe_work {
                            cost = f.mul(&cost, black_box(&p[0]));
                        }
                        black_box(cost);
                        let dv = d.replace_all(p);
                        (!f.is_zero(&dv)).then(|| f.div(&n.replace_all(p), &dv))
                    },
                    method,
                    &options,
                )
                .unwrap_or_else(|e| panic!("{name}/{method:?}: {e}"));
                let elapsed = start.elapsed().as_secs_f64() * 1e6;
                // Exact identity check outside the timed region; never just compare probes.
                assert_eq!(
                    &result.numerator * &d,
                    &result.denominator * &n,
                    "{name}/{method:?}"
                );
                if run > 0 {
                    println!(
                        "{name},{method:?},{run},{probe_work},{elapsed:.3},{},{},{},{},{},{},{}",
                        stats.probes,
                        stats.poles,
                        stats.attempts,
                        stats.univariate_interpolations,
                        stats.linear_solves,
                        result.numerator.nterms(),
                        result.denominator.nterms()
                    );
                }
            }
        }
    }
}
