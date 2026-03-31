use std::{
    io::{self, BufRead, Write},
    sync::Arc,
};

use smartstring::{LazyCompact, SmartString};
use symbolica::{
    domains::{SelfRing, integer::Z, rational::Q, rational_polynomial::RationalPolynomial},
    parser::{ParseSettings, Token},
    printer::{PrintOptions, PrintState},
    symbol,
};

fn main() {
    let mut buffer = String::with_capacity(2048);
    let mut stdin = io::stdin().lock();
    let mut stdout = io::stdout().lock();

    // read the number of variables
    let _ = stdin.read_line(&mut buffer).unwrap();
    let mut num_vars_and_var_names = buffer.split(' ');
    let num_vars = num_vars_and_var_names
        .next()
        .expect("Expected number of variables")
        .parse::<usize>()
        .expect("Number of vars should be a non-negative integer");

    let mut var_names: Vec<SmartString<LazyCompact>> = vec![];
    for _ in 0..num_vars {
        var_names.push(
            num_vars_and_var_names
                .next()
                .expect("Expected variable")
                .trim_end()
                .into(),
        );
    }

    let vars: Arc<Vec<_>> = Arc::new(var_names.iter().map(|v| symbol!(v).into()).collect());

    let print_opt = PrintOptions::file();

    buffer.clear();
    while let Ok(n) = stdin.read_line(&mut buffer) {
        if n == 0 || buffer.starts_with('\n') || buffer.starts_with("&q") {
            break;
        }

        let r: RationalPolynomial<_, u16> = Token::parse(&buffer, ParseSettings::polynomial())
            .unwrap()
            .to_rational_polynomial(&Q, &Z, &vars, &var_names)
            .unwrap();

        buffer.clear();
        r.format(&print_opt, PrintState::new(), &mut buffer)
            .unwrap();
        writeln!(stdout, "{buffer}").unwrap();

        buffer.clear();
    }
}
