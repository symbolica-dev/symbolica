use symbolica::{atom::AtomCore, id::Match, parse, symbol};

fn main() {
    let expr = parse!("f(z)*f(f(x))*f(y)");
    let pat_expr = parse!("f(x_)");

    let pattern = pat_expr.to_pattern();

    println!("> Matching pattern {pat_expr} to {expr}:");

    for x in expr.pattern_match(&pattern, None, None) {
        println!("\t x_ = {}", x.get(&symbol!("x_")).unwrap());
    }

    println!("> Matching pattern {pat_expr} to {expr}:");

    // use next_detailed for detailed information
    let mut it = expr.pattern_match(&pattern, None, None);
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
