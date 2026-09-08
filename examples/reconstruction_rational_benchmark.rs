//! End-to-end Q reconstruction, including modular images, CRT and a fresh prime.
use std::{sync::Arc, time::Instant};
use symbolica::{
    domains::finite_field::{ToFiniteField, Zp64},
    poly::{
        PolyVariable,
        reconstruction::{
            ReconstructionMethod::*, ReconstructionOptions, reconstruct_rational_function_over_q,
        },
    },
    prelude::*,
};

type ModularImage = (
    Zp64,
    MultivariatePolynomial<Zp64>,
    MultivariatePolynomial<Zp64>,
);

fn main() {
    let repeats = std::env::var("RECONSTRUCTION_REPEATS")
        .map(|s| s.parse::<u64>().unwrap())
        .unwrap_or(5);
    println!("case,method,seed,elapsed_us,probes,primes,images,support_resets");
    for (name, num, den, names) in [
        ("paper_eq3", "x*y+2", "x*y-2*x+4", ["x", "y"]),
        (
            "paper_eq28_y_d",
            "(d+13)^30*(y^2+9)^7+1",
            "(d-4)^29*(y^2-1)^5",
            ["y", "d"],
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
                let mut image: Option<ModularImage> = None;
                let start = Instant::now();
                let (r, stats) = reconstruct_rational_function_over_q(
                    vars.clone(),
                    |f, point| {
                        if image.as_ref().is_none_or(|(old, _, _)| old != f) {
                            image = Some((
                                f.clone(),
                                n.map_coeff(|c| c.to_finite_field(f), f.clone()),
                                d.map_coeff(|c| c.to_finite_field(f), f.clone()),
                            ));
                        }
                        let (_, n, d) = image.as_ref().unwrap();
                        let dv = d.replace_all(point);
                        (!f.is_zero(&dv)).then(|| f.div(&n.replace_all(point), &dv))
                    },
                    method,
                    &ReconstructionOptions {
                        seed,
                        max_degree: 64,
                        ..Default::default()
                    },
                    12,
                )
                .unwrap();
                let elapsed = start.elapsed().as_secs_f64() * 1e6;
                assert_eq!(&r.numerator * &d, &r.denominator * &n, "{name}/{method:?}");
                if seed > 0 {
                    println!(
                        "{name},{method:?},{seed},{elapsed:.3},{},{},{},{}",
                        stats.probes, stats.primes, stats.successful_images, stats.support_resets
                    );
                }
            }
        }
    }
}
