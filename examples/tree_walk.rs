use symbolica::{
    atom::Atom,
    id::{AtomTreeIterator, MatchSettings},
    parse,
};

fn main() {
    let expr: Atom = parse!("f(z)*f(f(x),z)*f(y)");

    println!("> Tree walk of {expr}:");

    for (loc, view) in AtomTreeIterator::new(
        expr.as_view(),
        MatchSettings {
            level_range: (1, Some(2)),
            level_is_tree_depth: false,
            ..Default::default()
        },
    ) {
        println!("\tAtom at location {loc:?}: {view}");
    }
}
