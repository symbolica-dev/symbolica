use ahash::HashMap;
use symbolica::prelude::*;

fn main() {
    let a = parse!("v1*cos(v1) + f1(1)^2");

    let mut const_map = HashMap::default();
    const_map.insert(parse!("v1"), 6.);
    const_map.insert(parse!("f1(1)"), 7.);

    let r = a.evaluate(&const_map).unwrap();
    assert_eq!(r, 54.761021719902196);

    let expr = parse!("2x");
    let x = parse!("x");
    let mut const_map = HashMap::default();
    const_map.insert(x.clone(), Float::with_val(200, 3));
    let result = expr.evaluate_with_prec(&const_map, 200).unwrap();
    assert_eq!(result, Float::with_val(200, 6));
}
