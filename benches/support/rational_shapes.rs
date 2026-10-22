//! Deterministic sparse, dense, and degree-boundary rational polynomial fixtures.
use std::sync::Arc;
use symbolica::domains::rational::RationalField;
use symbolica::prelude::*;

type P = MultivariatePolynomial<IntegerRing, u16>;
fn polynomial(vars: Arc<Vec<PolyVariable>>, shape: &str, bits: u32, variant: usize) -> P {
    let mut output = MultivariatePolynomial::new(&Z, None, vars.clone());
    let mut index = 0;
    let mut append = |powers: &[u16]| {
        let value = (Integer::from(((index * 17 + variant * 13) % 23 + 1) as i64)
            << bits.saturating_sub(5))
            + Integer::from((index * 3 + variant * 5 + 1) as i64);
        if shape == "sparse8_div128" && variant == 0 {
            // Multiplication by one monomial preserves every product collision.
            let mut shifted = powers.to_vec();
            shifted[0] += 1;
            output.append_monomial(value, &shifted);
        } else {
            output.append_monomial(value, powers);
        }
        index += 1;
    };
    if shape.starts_with("sparse") {
        // The boundary cases have the same support up to one monomial shift,
        // giving maximum dividend degrees 127 versus 128 with equal collisions.
        let high_degree = match shape {
            "sparse8_div127" | "sparse8_div128" => {
                if variant == 0 {
                    64
                } else {
                    63
                }
            }
            _ => 128 - variant as u16,
        };
        append(&vec![0; 8]);
        for variable in 0..8 {
            for degree in [1, high_degree] {
                let mut powers = vec![0; 8];
                powers[variable] = degree;
                append(&powers);
            }
        }
    } else {
        let degree = if shape == "dense3_d12" { 12 } else { 20 } - variant as u16;
        for x in 0..=degree {
            for y in 0..=degree - x {
                for z in 0..=degree - x - y {
                    append(&[x, y, z]);
                }
            }
        }
    }
    output
}

pub const SHAPES: [&str; 5] = [
    "sparse8_d128",
    "dense3_d12",
    "dense3_d20",
    "sparse8_div127",
    "sparse8_div128",
];
pub fn inputs(
    vars: Arc<Vec<PolyVariable>>,
    shape: &str,
    bits: u32,
    rational: bool,
) -> [MultivariatePolynomial<RationalField, u16>; 2] {
    let source = [
        polynomial(vars.clone(), shape, bits, 0),
        polynomial(vars.clone(), shape, bits, 1),
    ];
    let mut q = Vec::new();
    for p in source {
        let mut qp = MultivariatePolynomial::new(&Q, None, vars.clone());
        for (i, (c, e)) in p.coefficients.iter().zip(p.exponents_iter()).enumerate() {
            let denominator = if rational { [1, 3, 5, 7, 11][i % 5] } else { 1 };
            qp.append_monomial(Q.to_element(c.clone(), Integer::from(denominator), true), e);
        }
        q.push(qp);
    }

    q.try_into().unwrap()
}
