use wolfram_library_link::{self as wll};

use std::borrow::BorrowMut;
use std::sync::{Arc, RwLock};

use smartstring::{LazyCompact, SmartString};

use crate::domains::SelfRing;
use crate::domains::finite_field::{Zp, Zp64};
use crate::domains::integer::Z;
use crate::domains::rational::Q;
use crate::parser::{ParseSettings, Token};
use crate::poly::PolyVariable;
use crate::{
    domains::rational_polynomial::RationalPolynomial, printer::PrintOptions, printer::PrintState,
    symbol,
};
use once_cell::sync::Lazy;

static STATE: Lazy<RwLock<LocalState>> = Lazy::new(|| {
    RwLock::new(LocalState {
        buffer: String::with_capacity(2048),
        var_map: Arc::new(vec![]),
        var_name_map: vec![],
        input_has_rational_numbers: false,
        exp_fits_in_u8: true,
    })
});

struct LocalState {
    buffer: String,
    var_map: Arc<Vec<PolyVariable>>,
    var_name_map: Vec<SmartString<LazyCompact>>,
    input_has_rational_numbers: bool,
    exp_fits_in_u8: bool,
}

#[wll::export(name = "SymbolicaSetOptions")]
fn set_options(input_has_rational_numbers: bool, exp_fits_in_u8: bool) {
    let mut symbolica = STATE.write().unwrap();
    symbolica.input_has_rational_numbers = input_has_rational_numbers;
    symbolica.exp_fits_in_u8 = exp_fits_in_u8;
}

#[wll::export(name = "SymbolicaSetPolyVariables")]
fn set_vars(vars: String) {
    let mut symbolica = STATE.write().unwrap();

    let mut var_map = vec![];
    for var in vars.split(',') {
        let v = symbol!(var);
        var_map.push(v.into());
        symbolica.var_name_map.push(var.into());
    }

    symbolica.var_map = Arc::new(var_map);
}

/// Simplify a rational polynomial.
#[wll::export(name = "SymbolicaSimplify")]
fn simplify(input: String, prime: i64, explicit_rational_polynomial: bool) -> String {
    let mut symbolica = STATE.write().unwrap();
    let symbolica: &mut LocalState = symbolica.borrow_mut();

    let token = Token::parse(&input, ParseSettings::polynomial()).unwrap();

    macro_rules! to_rational {
        ($in_field: expr, $exp_size: ty) => {
            if prime == 0 {
                let r: RationalPolynomial<_, $exp_size> = token
                    .to_rational_polynomial(
                        &$in_field,
                        &Z,
                        &symbolica.var_map,
                        &symbolica.var_name_map,
                    )
                    .unwrap();

                r.format_string(
                    &PrintOptions {
                        explicit_rational_polynomial,
                        ..PrintOptions::mathematica()
                    },
                    PrintState::default(),
                )
            } else {
                if prime >= 0 && prime <= u32::MAX as i64 {
                    let field = Zp::new(prime as u32);
                    let rf: RationalPolynomial<_, $exp_size> = token
                        .to_rational_polynomial(
                            &field,
                            &field,
                            &symbolica.var_map,
                            &symbolica.var_name_map,
                        )
                        .unwrap();

                    symbolica.buffer.clear();
                    rf.format_string(
                        &PrintOptions {
                            explicit_rational_polynomial,
                            ..PrintOptions::mathematica()
                        },
                        PrintState::default(),
                    )
                } else {
                    let field = Zp64::new(prime as u64);
                    let rf: RationalPolynomial<_, $exp_size> = token
                        .to_rational_polynomial(
                            &field,
                            &field,
                            &symbolica.var_map,
                            &symbolica.var_name_map,
                        )
                        .unwrap();

                    symbolica.buffer.clear();
                    rf.format_string(
                        &PrintOptions {
                            explicit_rational_polynomial,
                            ..PrintOptions::mathematica()
                        },
                        PrintState::default(),
                    )
                }
            }
        };
    }

    match (
        symbolica.input_has_rational_numbers,
        symbolica.exp_fits_in_u8,
    ) {
        (false, true) => to_rational!(Z, u8),
        (true, true) => to_rational!(Q, u8),
        (false, false) => to_rational!(Z, u16),
        (true, false) => to_rational!(Q, u16),
    }
}
