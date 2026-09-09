//! Reuse learned support and rational coefficient hypotheses in a
//! later prime. A completely known homogeneous component fixes the line scale.
use super::*;
use crate::domains::{finite_field::ToFiniteField, rational::Rational};
use std::collections::BTreeMap;

type Component = ((usize, u16), Vec<Vec<u16>>);

pub(super) fn reconstruct<F>(
    field: Zp64,
    variables: Arc<Vec<PolyVariable>>,
    black_box: &mut F,
    options: &ReconstructionOptions,
    support: &[(usize, Vec<u16>)],
    frozen: &[Option<Rational>],
) -> Option<Result<(Fraction, ReconstructionStats)>>
where
    F: FnMut(&Zp64, &[Element]) -> Option<Element>,
{
    let template = Polynomial::new(&field, None, variables);
    let mut known = [template.zero(), template.zero()];
    let mut groups = BTreeMap::<(usize, u16), Vec<Vec<u16>>>::new();
    for ((side, ex), c) in support.iter().zip(frozen) {
        let degree: u64 = ex.iter().map(|e| *e as u64).sum();
        let Ok(degree) = u16::try_from(degree) else {
            return None;
        };
        let group = groups.entry((*side, degree)).or_default();
        if let Some(c) = c {
            if field.is_zero(&c.denominator_ref().to_finite_field(&field)) {
                return None;
            }
            known[*side].append_monomial(c.to_finite_field(&field), ex);
        } else {
            group.push(ex.clone());
        }
    }
    // A partially known coefficient of t cannot normalize an individual line.
    // Wait until one whole homogeneous component is known and nonzero here.
    if !groups.iter().any(|(&(side, degree), ex)| {
        ex.is_empty() && !homogeneous_part(&known[side], degree).is_zero()
    }) {
        return None;
    }
    let mut components: Vec<_> = groups
        .into_iter()
        .filter(|(_, ex)| !ex.is_empty())
        .collect();
    components.sort_by_key(|((side, degree), ex)| (ex.len(), *degree, 1 - *side));
    let mut ctx = Context {
        field,
        template,
        black_box,
        options,
        rng: StdRng::seed_from_u64(options.seed),
        cache: HashMap::new(),
        stats: Default::default(),
        monomial_factors: None,
        balanced_pilot: None,
    };
    // A rejected coefficient hypothesis is unlikely to improve with another
    // direction. Fall back immediately; ordinary reconstruction owns retries.
    ctx.stats.attempts = 1;
    Some(
        reconstruct_components(&mut ctx, &known, &components).and_then(|r| {
            if ctx.verify(&r)? {
                let r = Fraction::from_num_den(r.numerator, r.denominator, &ctx.field, true);
                Ok((r, ctx.stats))
            } else {
                unlucky()
            }
        }),
    )
}

fn reconstruct_components<F>(
    ctx: &mut Context<'_, F>,
    initial: &[Polynomial; 2],
    components: &[Component],
) -> Result<Fraction>
where
    F: FnMut(&Zp64, &[Element]) -> Option<Element>,
{
    let f = ctx.field.clone();
    let mut known = initial.clone();
    let mut powers = [Vec::new(), Vec::new()];
    for ((side, degree), _) in components {
        powers[*side].push(*degree);
    }
    let mut anchors = ctx.point();
    anchors[0] = f.one();
    let shift = vec![f.zero(); anchors.len()];
    let mut lines = Vec::<Fraction>::new();
    // Check the first line early when substantial interpolation remains. This
    // rejects wrong fully known components before paying for every direction.
    let check_first_line = components.iter().map(|(_, ex)| ex.len()).sum::<usize>()
        > components.len() + ctx.options.verification_points;
    for ((side, degree), exponents) in components {
        let nodes: Vec<_> = exponents
            .iter()
            .map(|ex| {
                ex.iter()
                    .zip(&anchors)
                    .fold(f.one(), |v, (e, a)| f.mul(&v, &f.pow(a, *e as u64)))
            })
            .collect();
        distinct_nonzero(&f, &nodes)?;
        let correction = homogeneous_part(&known[*side], *degree);
        let mut values = Vec::with_capacity(exponents.len());
        for i in 0..exponents.len() {
            let direction: Vec<_> = anchors.iter().map(|a| f.pow(a, i as u64 + 1)).collect();
            if i == lines.len() {
                let fixed = [
                    homogeneous_line(&known[0], &direction),
                    homogeneous_line(&known[1], &direction),
                ];
                let line =
                    ctx.line_solve(&shift, &direction, &powers[0], &powers[1], Some(&fixed))?;
                if lines.is_empty() && check_first_line {
                    verify_line(ctx, &direction, &line)?;
                }
                lines.push(line);
            }
            let line = &lines[i];
            let p = if *side == 0 {
                &line.numerator
            } else {
                &line.denominator
            };
            values.push(f.sub(
                &coefficient(p, 0, *degree),
                &correction.replace_all(&direction),
            ));
        }
        let coefficients = ctx
            .template
            .solve_shifted_transposed_vandermonde(&nodes, &values);
        for (ex, c) in exponents.iter().zip(coefficients) {
            known[*side].append_monomial(c, ex);
        }
        powers[*side].retain(|d| *d != *degree);
    }
    let [numerator, denominator] = known;
    Ok(Fraction {
        numerator,
        denominator,
    })
}

fn verify_line<F>(ctx: &mut Context<'_, F>, direction: &[Element], line: &Fraction) -> Result<()>
where
    F: FnMut(&Zp64, &[Element]) -> Option<Element>,
{
    let f = ctx.field.clone();
    let mut checked = 0;
    for _ in 0..ctx.options.verification_points * 16 {
        let t = ctx.random();
        let point: Vec<_> = direction.iter().map(|x| f.mul(x, &t)).collect();
        if ctx.cache.contains_key(&point) {
            continue;
        }
        if let Some(y) = ctx.probe(&point)? {
            // direction[0] = 1, and the line is stored as a polynomial in x_0.
            if value(line, &point) != Some(y) {
                return unlucky();
            }
            checked += 1;
            if checked == ctx.options.verification_points {
                return Ok(());
            }
        }
    }
    unlucky()
}
