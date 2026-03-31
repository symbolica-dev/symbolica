use std::f64::consts::PI;

use symbolica::numerical_integration::{ContinuousGrid, DiscreteGrid, Grid, MonteCarloRng, Sample};

fn main() {
    // Integrate x*pi + x^2 using multi-channeling:
    // x*pi and x^2 will have their own Vegas grid
    let fs = [|x: f64| (x * PI).sin(), |x: f64| x * x];

    let mut grid = DiscreteGrid::new(
        vec![
            Some(Grid::Continuous(ContinuousGrid::new(
                1, 10, 1000, None, false,
            ))),
            Some(Grid::Continuous(ContinuousGrid::new(
                1, 10, 1000, None, false,
            ))),
        ],
        0.01,
        false,
    );

    let mut rng = MonteCarloRng::new(0, 0);

    let mut sample = Sample::new();
    for iteration in 1..20 {
        // sample 10_000 times per iteration
        for _ in 0..10_000 {
            grid.sample(&mut rng, &mut sample);

            if let Sample::Discrete(_weight, i, cont_sample) = &sample {
                if let Sample::Continuous(_cont_weight, xs) = cont_sample.as_ref().unwrap().as_ref()
                {
                    grid.add_training_sample(&sample, fs[*i](xs[0])).unwrap();
                }
            }
        }

        grid.update(1.5, 1.5);

        println!(
            "Integral at iteration {:2}: {:.6} ± {:.6} (χ² = {:.3})",
            iteration,
            grid.accumulator.avg,
            grid.accumulator.err,
            grid.accumulator.chi_sq / grid.accumulator.cur_iter as f64
        );
    }
}
