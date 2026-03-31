use brotli::CompressorWriter;
use symbolica::{
    atom::AtomCore,
    parse,
    streaming::{TermStreamer, TermStreamerConfig},
};

fn main() {
    let input = parse!("x+ f(x) + 2*f(y) + 7*f(z)");
    let pattern = parse!("f(x_)").to_pattern();
    let rhs = parse!("f(x) + x").to_pattern();

    let mut stream = TermStreamer::<CompressorWriter<_>>::new(TermStreamerConfig {
        n_cores: 4,
        path: ".".to_owned(),
        max_mem_bytes: 40,
    });
    stream.push(input);

    // map every term in the expression
    stream = stream.map(|x| x.replace(&pattern).with(&rhs).expand());

    let res = stream.to_expression();
    println!("\t+ {res}");
}
