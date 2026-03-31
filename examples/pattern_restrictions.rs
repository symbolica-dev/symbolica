use symbolica::{
    atom::{AtomCore, AtomView},
    coefficient::CoefficientView,
    domains::finite_field,
    id::{Match, MatchSettings, WildcardRestriction},
    parse, symbol,
};
fn main() {
    let expr = parse!("f(1,2,3,4,5,6,7)");
    let pat_expr = parse!("f(x__,y__,z__,w__)");

    let pattern = pat_expr.to_pattern();

    let x = symbol!("x__");
    let y = symbol!("y__");
    let z = symbol!("z__");
    let w = symbol!("w__");

    let conditions = x.restrict(WildcardRestriction::Length(0, Some(2)))
        & y.restrict(WildcardRestriction::Length(0, Some(4)))
        & y.restrict(WildcardRestriction::cmp(x, |y, x| {
            let len_x = match x {
                Match::Multiple(_, s) => s.len(),
                _ => 1,
            };
            let len_y = match y {
                Match::Multiple(_, s) => s.len(),
                _ => 1,
            };
            len_x >= len_y
        }))
        & z.restrict(WildcardRestriction::filter(|x: &Match| {
            if let Match::Single(AtomView::Num(num)) = x {
                if let CoefficientView::Natural(x, y, n, _) = num.get_coeff_view() {
                    n == 0 && y == 1 && x > 0 && finite_field::is_prime_u64(x as u64)
                } else {
                    false
                }
            } else {
                false
            }
        }))
        & w.restrict(WildcardRestriction::Length(0, None));
    let settings = MatchSettings::default();

    println!(
        "> Matching pattern {pat_expr} : 0 <= len(x) <= 2, 0 <= len(y) <= 4, len(x) >= len(y) & is_prime(z) to {expr}:"
    );

    let mut it = expr.pattern_match(&pattern, Some(&conditions), Some(&settings));
    while let Some(m) = it.next_detailed() {
        println!("\tMatch at location {:?} - {:?}:", m.position, m.used_flags);
        for (id, v) in m.match_stack {
            print!("\t\t{id} = ");
            match v {
                Match::Single(s) => {
                    print!("{s}")
                }
                Match::Multiple(slice_type, mm) => {
                    print!("{slice_type:?} ");
                    for vv in mm {
                        print!("{vv}");
                        print!(", ")
                    }
                }
                Match::FunctionName(f) => {
                    print!("Fn {f}")
                }
            }
            println!();
        }
    }
}
