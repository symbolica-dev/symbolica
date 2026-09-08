use std::sync::Arc;
use symbolica::{
    poly::reconstruction::{
        ReconstructionMethod, ReconstructionOptions, reconstruct_rational_function_over_q,
    },
    prelude::*,
};

fn main() {
    // Smirnov–Zeng Eq. (3), available to the reconstruction only as an oracle.
    let variables = Arc::new(vec![symbol!("x").into(), symbol!("y").into()]);
    let (result, stats) = reconstruct_rational_function_over_q(
        variables,
        |field, point| {
            let xy = field.mul(&point[0], &point[1]);
            let numerator = field.add(&xy, &field.to_element(2));
            let denominator = field.add(
                &field.sub(&xy, &field.mul(&field.to_element(2), &point[0])),
                &field.to_element(4),
            );
            (!field.is_zero(&denominator)).then(|| field.div(&numerator, &denominator))
        },
        ReconstructionMethod::BalancedZippel,
        &ReconstructionOptions::default(),
        12,
    )
    .unwrap();
    println!("f(x,y) = {result}");
    println!("{} probes across {} primes", stats.probes, stats.primes);
}
