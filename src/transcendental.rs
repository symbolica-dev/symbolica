use std::sync::LazyLock;

use rug::float::Constant;

use crate::{
    atom::{Atom, AtomCore, AtomView, EvaluationInfo, Symbol},
    coefficient::{Coefficient, CoefficientView},
    domains::{
        float::{Complex, Float, Real, RealLike},
        integer::Integer,
        rational::Rational,
    },
    function, get_symbol,
    state::{State, StateInitializer},
    symbol,
    utils::Settable,
};

static SPECIALS: LazyLock<SpecialSymbols> = LazyLock::new(|| SpecialSymbols {
    euler_gamma: get_symbol!("euler_gamma").expect("Euler gamma not defined"),
    gamma: get_symbol!("gamma").expect("gamma not defined"),
    polygamma: get_symbol!("polygamma").expect("polygamma not defined"),
    polylog: get_symbol!("polylog").expect("polylog not defined"),
});
static GEOMETRICS: LazyLock<GeometricSymbols> = LazyLock::new(|| GeometricSymbols {
    tan: get_symbol!("tan").expect("tan not defined"),
    cot: get_symbol!("cot").expect("cot not defined"),
    sec: get_symbol!("sec").expect("sec not defined"),
    csc: get_symbol!("csc").expect("csc not defined"),
    asin: get_symbol!("asin").expect("asin not defined"),
    acos: get_symbol!("acos").expect("acos not defined"),
    atan: get_symbol!("atan").expect("atan not defined"),
    acot: get_symbol!("acot").expect("acot not defined"),
    asec: get_symbol!("asec").expect("asec not defined"),
    acsc: get_symbol!("acsc").expect("acsc not defined"),
    sinh: get_symbol!("sinh").expect("sinh not defined"),
    cosh: get_symbol!("cosh").expect("cosh not defined"),
    tanh: get_symbol!("tanh").expect("tanh not defined"),
    coth: get_symbol!("coth").expect("coth not defined"),
    sech: get_symbol!("sech").expect("sech not defined"),
    csch: get_symbol!("csch").expect("csch not defined"),
    asinh: get_symbol!("asinh").expect("asinh not defined"),
    acosh: get_symbol!("acosh").expect("acosh not defined"),
    atanh: get_symbol!("atanh").expect("atanh not defined"),
    acoth: get_symbol!("acoth").expect("acoth not defined"),
    asech: get_symbol!("asech").expect("asech not defined"),
    acsch: get_symbol!("acsch").expect("acsch not defined"),
});
static BESSELS: LazyLock<BesselSymbols> = LazyLock::new(|| BesselSymbols {
    bessel_j: get_symbol!("bessel_j").expect("bessel_j not defined"),
    bessel_y: get_symbol!("bessel_y").expect("bessel_y not defined"),
    bessel_i: get_symbol!("bessel_i").expect("bessel_i not defined"),
    bessel_k: get_symbol!("bessel_k").expect("bessel_k not defined"),
});

struct SpecialSymbols {
    euler_gamma: Symbol,
    gamma: Symbol,
    polygamma: Symbol,
    polylog: Symbol,
}

struct GeometricSymbols {
    tan: Symbol,
    cot: Symbol,
    sec: Symbol,
    csc: Symbol,
    asin: Symbol,
    acos: Symbol,
    atan: Symbol,
    acot: Symbol,
    asec: Symbol,
    acsc: Symbol,
    sinh: Symbol,
    cosh: Symbol,
    tanh: Symbol,
    coth: Symbol,
    sech: Symbol,
    csch: Symbol,
    asinh: Symbol,
    acosh: Symbol,
    atanh: Symbol,
    acoth: Symbol,
    asech: Symbol,
    acsch: Symbol,
}

struct BesselSymbols {
    bessel_j: Symbol,
    bessel_y: Symbol,
    bessel_i: Symbol,
    bessel_k: Symbol,
}

impl SpecialSymbols {
    fn new() -> Self {
        let euler_gamma = symbol!(
            "γ",
            aliases = ["euler_gamma"],
            der = |_x, _i, out| {
                out.to_num(Coefficient::zero());
            },
            eval = EvaluationInfo::constant(|_tags, prec| {
                Ok(Complex::new(
                    Float::with_val(prec, Constant::Euler),
                    Float::new(prec),
                ))
            })
        );

        let mut symbols = crate::symbol_group!(
            "gamma";;
            |symbols, b| {
                let gamma_symbol = symbols[0];
                let polygamma_symbol = symbols[1];

                b.with_normalization_function(|x, out| {
                    if let Some([arg]) = function_arguments::<1>(x) {
                        if maybe_eval_unary_float_in_norm(arg, out, gamma_numeric_eval) {
                            return;
                        }

                        if let Ok(rat) = Rational::try_from(arg)
                            && let Some(exact) = gamma_exact_rational(&rat)
                        {
                            **out = exact;
                        }
                    }
                })
                .with_derivative_function(move |x, i, out| {
                    if i == 0
                        && let Some([arg]) = function_arguments::<1>(x)
                    {
                        **out = function!(gamma_symbol, arg) * function!(polygamma_symbol, 0, arg);
                    }
                })
                .with_series_function(
                    move |args| {
                        let [arg] = args else { return None; };
                        let point = arg.coefficient(0.into());
                        let Ok(pole) = Integer::try_from(&point) else {
                            return None;
                        };

                        if pole > 0 {
                            return None;
                        }

                        let arg = arg.to_atom();
                        let shift = (-pole).to_i64().unwrap();
                        let mut regularized = function!(gamma_symbol, arg.to_owned() + shift + 1);
                        for k in 0..shift {
                            regularized = regularized / (arg.to_owned() + k);
                        }

                        Some((Atom::num(1) / (arg + shift), regularized))
                    },
                )
                .with_evaluation_info(
                    EvaluationInfo::new()
                    .register(|args: &[Complex<Float>]| {
                        unary_eval_complex_float(args, gamma_numeric_eval)
                    })
                    .register(|args: &[f64]| unary_eval_f64(args, gamma_numeric_eval))
                    .register(|args: &[Complex<f64>]| {
                        unary_eval_complex_f64(args, gamma_numeric_eval)
                    }),
                )
            },
            "polygamma";;
            |symbols, b| {
                let polygamma_symbol = symbols[1];

                b.with_normalization_function(move |x, out| {
                    if let Some([n, z]) = function_arguments::<2>(x) {
                        if let Ok(order) = u32::try_from(n) {
                            if maybe_eval_polygamma_float_in_norm(order, z, out) {
                                return;
                            }

                            if let Ok(rat) = Rational::try_from(z)
                                && let Some(exact) = polygamma_exact_rational(order, &rat)
                            {
                                **out = exact;
                                return;
                            }

                            if let Ok(rat) = Rational::try_from(z)
                                && rat.denominator() == 1
                                && rat.numerator() <= 0
                            {
                                out.to_num(Coefficient::complex_infinity());
                            }
                        }
                    }
                })
                .with_derivative_function(move |x, i, out| {
                    if i == 1
                        && let Some([n, z]) = function_arguments::<2>(x)
                        && let Ok(order) = u32::try_from(n)
                    {
                        **out = function!(polygamma_symbol, Atom::num(order + 1), z);
                    }
                })
                .with_series_function(move |args| {
                    let [n, arg] = args else { return None; };
                    let order = u32::try_from(n.coefficient(0.into())).ok()?;
                    let point = arg.coefficient(0.into());
                    let Ok(pole) = Integer::try_from(&point) else {
                        return None;
                    };

                    if pole > 0 {
                        return None;
                    }

                    let arg = arg.to_atom();
                    let shift = (-pole).to_i64().unwrap();
                    let pole = arg.to_owned() + shift;
                    let pole_power = pole.pow(Atom::num(order + 1));

                    let sign = if order % 2 == 0 {
                        Atom::num(Integer::factorial(order))
                    } else {
                        -Atom::num(Integer::factorial(order))
                    };

                    let mut recurrence_sum = Atom::new();
                    for k in 0..=shift {
                        recurrence_sum +=
                            Atom::num(1) / (arg.to_owned() + k).pow(Atom::num(order + 1));
                    }

                    let regularized = pole_power.clone()
                        * (function!(polygamma_symbol, Atom::num(order), arg + shift + 1)
                            - sign * recurrence_sum);

                    Some((Atom::num(1) / pole_power, regularized))
                })
                .with_evaluation_info(
                    EvaluationInfo::new().with_tags(1)
                    .register_tagged(|tags| {
                        let order = nonnegative_integer_tag("polygamma", tags).ok();
                        Box::new(move |args: &[Complex<Float>]| {
                            let Some(order) = order else {
                                return Complex::new(
                                    Float::with_val(53, rug::float::Special::Nan),
                                    Float::with_val(53, rug::float::Special::Nan),
                                );
                            };
                            unary_eval_complex_float(args, |arg, prec| {
                                polygamma_numeric_eval(order, arg, prec)
                            })
                        })
                    })
                    .register_tagged(|tags| {
                        let order = nonnegative_integer_tag("polygamma", tags).ok();
                        Box::new(move |args: &[f64]| polygamma_eval_f64(order, args))
                    })
                    .register_tagged(|tags| {
                        let order = nonnegative_integer_tag("polygamma", tags).ok();
                        Box::new(move |args: &[Complex<f64>]| {
                            polygamma_eval_complex_f64(order, args)
                        })
                    }),
                )
            },
            "polylog";;
            |symbols, b| {
                let polylog_symbol = symbols[2];

                b.with_normalization_function(|x, out| {
                    if let Some([s, z]) = function_arguments::<2>(x) {
                        if let Some(exact) = polylog_exact(s, z) {
                            **out = exact;
                            return;
                        }

                        if maybe_eval_binary_float_in_norm(s, z, out, polylog_numeric_eval) {
                            return;
                        }
                    }
                })
                .with_derivative_function(move |x, i, out| {
                    if i == 1
                        && let Some([s, z]) = function_arguments::<2>(x)
                        && let Some(order) = atom_to_integer(s)
                    {
                        if order == 1 {
                            **out = Atom::num(1) / (Atom::num(1) - z);
                        } else {
                            **out = function!(polylog_symbol, Atom::num(order - 1), z) / z;
                        }
                    }
                })
                .with_evaluation_info(
                    EvaluationInfo::new().with_tags(1)
                    .register_tagged(|tags| {
                        let tags = tags.iter().map(|x| x.to_owned()).collect::<Vec<_>>();
                        Box::new(move |args: &[Complex<Float>]| {
                            let prec = args
                                .first()
                                .map(|x| x.re.prec().max(x.im.prec()))
                                .unwrap_or(53);
                            let tag_views = tags.iter().map(|x| x.as_view()).collect::<Vec<_>>();
                            let Ok(order) = complex_float_tag("polylog", &tag_views, prec) else {
                                return Complex::new(
                                    Float::with_val(53, rug::float::Special::Nan),
                                    Float::with_val(53, rug::float::Special::Nan),
                                );
                            };
                            let [arg] = args else {
                                return Complex::new(
                                    Float::with_val(53, rug::float::Special::Nan),
                                    Float::with_val(53, rug::float::Special::Nan),
                                );
                            };
                            polylog_numeric_eval(&order, arg, prec).unwrap_or_else(|| {
                                Complex::new(
                                    Float::with_val(53, rug::float::Special::Nan),
                                    Float::with_val(53, rug::float::Special::Nan),
                                )
                            })
                        })
                    })
                    .register_tagged(|tags| {
                        let order = complex_float_tag("polylog", tags, 53).ok();
                        Box::new(move |args: &[f64]| polylog_eval_f64(order.clone(), args))
                    })
                    .register_tagged(|tags| {
                        let order = complex_float_tag("polylog", tags, 53).ok();
                        Box::new(move |args: &[Complex<f64>]| {
                            polylog_eval_complex_f64(order.clone(), args)
                        })
                    }),
                )
            }
        );

        let polylog = symbols.pop().unwrap();
        let polygamma = symbols.pop().unwrap();
        let gamma = symbols.pop().unwrap();

        Self {
            euler_gamma,
            gamma,
            polygamma,
            polylog,
        }
    }
}

impl GeometricSymbols {
    fn new() -> Self {
        let mut symbols = crate::symbol_group!(
            "tan";;
            |symbols, b| {
                let tan = symbols[0];
                let sec = symbols[2];
                b.with_normalization_function(|x, out| {
                    if let Some([arg]) = function_arguments::<1>(x) {
                        if arg.is_zero() {
                            out.to_num(Coefficient::zero());
                            return;
                        }
                        let _ = maybe_eval_unary_float_in_norm(arg, out, tan_numeric_eval);
                    }
                })
                .with_derivative_function(move |x, i, out| {
                    if i == 0 && let Some([arg]) = function_arguments::<1>(x) {
                        **out = function!(sec, arg).pow(Atom::num(2));
                    }
                })
                .with_series_function(move |args| {
                    let [arg] = args else { return None; };
                    let point = arg.coefficient(0.into());
                    if !is_tan_pole(point.as_view()) {
                        return None;
                    }
                    let delta = arg.to_atom() - &point;
                    Some((Atom::num(1) / delta.clone(), -delta.clone() / function!(tan, delta)))
                })
                .with_evaluation_info(
                    EvaluationInfo::new().register(|args: &[Complex<Float>]| {
                        unary_eval_complex_float(args, tan_numeric_eval)
                    })
                    .register(|args: &[f64]| unary_eval_real_f64(args, f64::tan))
                    .register(|args: &[Complex<f64>]| unary_eval_complex_real_f64(args, |z| z.tan())),
                )
            },
            "cot";;
            |symbols, b| {
                let tan = symbols[0];
                let csc = symbols[3];
                b.with_normalization_function(|x, out| {
                    if let Some([arg]) = function_arguments::<1>(x) {
                        if arg.is_zero() {
                            out.to_num(Coefficient::complex_infinity());
                            return;
                        }
                        let _ = maybe_eval_unary_float_in_norm(arg, out, cot_numeric_eval);
                    }
                })
                .with_derivative_function(move |x, i, out| {
                    if i == 0 && let Some([arg]) = function_arguments::<1>(x) {
                        **out = -function!(csc, arg).pow(Atom::num(2));
                    }
                })
                .with_series_function(move |args| {
                    let [arg] = args else { return None; };
                    let point = arg.coefficient(0.into());
                    if !is_cot_csc_pole(point.as_view()) {
                        return None;
                    }
                    let delta = arg.to_atom() - &point;
                    Some((Atom::num(1) / delta.clone(), delta.clone() / function!(tan, delta)))
                })
                .with_evaluation_info(
                    EvaluationInfo::new().register(|args: &[Complex<Float>]| {
                        unary_eval_complex_float(args, cot_numeric_eval)
                    })
                    .register(|args: &[f64]| unary_eval_real_f64(args, cot_eval_f64))
                    .register(|args: &[Complex<f64>]| unary_eval_complex_real_f64(args, |z| Complex::new(1.0, 0.0) / z.tan())),
                )
            },
            "sec";;
            |symbols, b| {
                let tan = symbols[0];
                let sec = symbols[2];
                b.with_normalization_function(|x, out| {
                    if let Some([arg]) = function_arguments::<1>(x) {
                        if arg.is_zero() {
                            out.to_num(Coefficient::one());
                            return;
                        }
                        let _ = maybe_eval_unary_float_in_norm(arg, out, sec_numeric_eval);
                    }
                })
                .with_derivative_function(move |x, i, out| {
                    if i == 0 && let Some([arg]) = function_arguments::<1>(x) {
                        **out = function!(sec, arg) * function!(tan, arg);
                    }
                })
                .with_series_function(move |args| {
                    let [arg] = args else { return None; };
                    let point = arg.coefficient(0.into());
                    if !is_tan_pole(point.as_view()) {
                        return None;
                    }
                    let delta = arg.to_atom() - &point;
                    Some((
                        Atom::num(1) / delta.clone(),
                        -delta.clone() / function!(State::SIN, delta),
                    ))
                })
                .with_evaluation_info(
                    EvaluationInfo::new().register(|args: &[Complex<Float>]| {
                        unary_eval_complex_float(args, sec_numeric_eval)
                    })
                    .register(|args: &[f64]| unary_eval_real_f64(args, sec_eval_f64))
                    .register(|args: &[Complex<f64>]| unary_eval_complex_real_f64(args, |z| Complex::new(1.0, 0.0) / z.cos())),
                )
            },
            "csc";;
            |symbols, b| {
                let cot = symbols[1];
                let csc = symbols[3];
                b.with_normalization_function(|x, out| {
                    if let Some([arg]) = function_arguments::<1>(x) {
                        if arg.is_zero() {
                            out.to_num(Coefficient::complex_infinity());
                            return;
                        }
                        let _ = maybe_eval_unary_float_in_norm(arg, out, csc_numeric_eval);
                    }
                })
                .with_derivative_function(move |x, i, out| {
                    if i == 0 && let Some([arg]) = function_arguments::<1>(x) {
                        **out = -function!(csc, arg) * function!(cot, arg);
                    }
                })
                .with_series_function(move |args| {
                    let [arg] = args else { return None; };
                    let point = arg.coefficient(0.into());
                    if !is_cot_csc_pole(point.as_view()) {
                        return None;
                    }
                    let delta = arg.to_atom() - &point;
                    Some((
                        Atom::num(1) / delta.clone(),
                        delta.clone() / function!(State::SIN, delta),
                    ))
                })
                .with_evaluation_info(
                    EvaluationInfo::new().register(|args: &[Complex<Float>]| {
                        unary_eval_complex_float(args, csc_numeric_eval)
                    })
                    .register(|args: &[f64]| unary_eval_real_f64(args, csc_eval_f64))
                    .register(|args: &[Complex<f64>]| unary_eval_complex_real_f64(args, |z| Complex::new(1.0, 0.0) / z.sin())),
                )
            },
            "sinh";;
            |symbols, b| {
                let cosh = symbols[5];
                b.with_normalization_function(|x, out| {
                    if let Some([arg]) = function_arguments::<1>(x) {
                        if arg.is_zero() {
                            out.to_num(Coefficient::zero());
                            return;
                        }
                        let _ = maybe_eval_unary_float_in_norm(arg, out, sinh_numeric_eval);
                    }
                })
                .with_derivative_function(move |x, i, out| {
                    if i == 0 && let Some([arg]) = function_arguments::<1>(x) {
                        **out = function!(cosh, arg);
                    }
                })
                .with_evaluation_info(
                    EvaluationInfo::new().register(|args: &[Complex<Float>]| {
                        unary_eval_complex_float(args, sinh_numeric_eval)
                    })
                    .register(|args: &[f64]| unary_eval_real_f64(args, f64::sinh))
                    .register(|args: &[Complex<f64>]| unary_eval_complex_real_f64(args, |z| z.sinh())),
                )
            },
            "cosh";;
            |symbols, b| {
                let sinh = symbols[4];
                b.with_normalization_function(|x, out| {
                    if let Some([arg]) = function_arguments::<1>(x) {
                        if arg.is_zero() {
                            out.to_num(Coefficient::one());
                            return;
                        }
                        let _ = maybe_eval_unary_float_in_norm(arg, out, cosh_numeric_eval);
                    }
                })
                .with_derivative_function(move |x, i, out| {
                    if i == 0 && let Some([arg]) = function_arguments::<1>(x) {
                        **out = function!(sinh, arg);
                    }
                })
                .with_evaluation_info(
                    EvaluationInfo::new().register(|args: &[Complex<Float>]| {
                        unary_eval_complex_float(args, cosh_numeric_eval)
                    })
                    .register(|args: &[f64]| unary_eval_real_f64(args, f64::cosh))
                    .register(|args: &[Complex<f64>]| unary_eval_complex_real_f64(args, |z| z.cosh())),
                )
            },
            "tanh";;
            |symbols, b| {
                let sech = symbols[8];
                b.with_normalization_function(|x, out| {
                    if let Some([arg]) = function_arguments::<1>(x) {
                        if arg.is_zero() {
                            out.to_num(Coefficient::zero());
                            return;
                        }
                        let _ = maybe_eval_unary_float_in_norm(arg, out, tanh_numeric_eval);
                    }
                })
                .with_derivative_function(move |x, i, out| {
                    if i == 0 && let Some([arg]) = function_arguments::<1>(x) {
                        **out = function!(sech, arg).pow(Atom::num(2));
                    }
                })
                .with_evaluation_info(
                    EvaluationInfo::new().register(|args: &[Complex<Float>]| {
                        unary_eval_complex_float(args, tanh_numeric_eval)
                    })
                    .register(|args: &[f64]| unary_eval_real_f64(args, f64::tanh))
                    .register(|args: &[Complex<f64>]| unary_eval_complex_real_f64(args, |z| z.tanh())),
                )
            },
            "coth";;
            |symbols, b| {
                let csch = symbols[9];
                b.with_normalization_function(|x, out| {
                    if let Some([arg]) = function_arguments::<1>(x) {
                        if arg.is_zero() {
                            out.to_num(Coefficient::complex_infinity());
                            return;
                        }
                        let _ = maybe_eval_unary_float_in_norm(arg, out, coth_numeric_eval);
                    }
                })
                .with_derivative_function(move |x, i, out| {
                    if i == 0 && let Some([arg]) = function_arguments::<1>(x) {
                        **out = -function!(csch, arg).pow(Atom::num(2));
                    }
                })
                .with_evaluation_info(
                    EvaluationInfo::new().register(|args: &[Complex<Float>]| {
                        unary_eval_complex_float(args, coth_numeric_eval)
                    })
                    .register(|args: &[f64]| unary_eval_real_f64(args, coth_eval_f64))
                    .register(|args: &[Complex<f64>]| unary_eval_complex_real_f64(args, |z| Complex::new(1.0, 0.0) / z.tanh())),
                )
            },
            "sech";;
            |symbols, b| {
                let sech = symbols[8];
                let tanh = symbols[6];
                b.with_normalization_function(|x, out| {
                    if let Some([arg]) = function_arguments::<1>(x) {
                        if arg.is_zero() {
                            out.to_num(Coefficient::one());
                            return;
                        }
                        let _ = maybe_eval_unary_float_in_norm(arg, out, sech_numeric_eval);
                    }
                })
                .with_derivative_function(move |x, i, out| {
                    if i == 0 && let Some([arg]) = function_arguments::<1>(x) {
                        **out = -function!(sech, arg) * function!(tanh, arg);
                    }
                })
                .with_evaluation_info(
                    EvaluationInfo::new().register(|args: &[Complex<Float>]| {
                        unary_eval_complex_float(args, sech_numeric_eval)
                    })
                    .register(|args: &[f64]| unary_eval_real_f64(args, sech_eval_f64))
                    .register(|args: &[Complex<f64>]| unary_eval_complex_real_f64(args, |z| Complex::new(1.0, 0.0) / z.cosh())),
                )
            },
            "csch";;
            |symbols, b| {
                let csch = symbols[9];
                let coth = symbols[7];
                b.with_normalization_function(|x, out| {
                    if let Some([arg]) = function_arguments::<1>(x) {
                        if arg.is_zero() {
                            out.to_num(Coefficient::complex_infinity());
                            return;
                        }
                        let _ = maybe_eval_unary_float_in_norm(arg, out, csch_numeric_eval);
                    }
                })
                .with_derivative_function(move |x, i, out| {
                    if i == 0 && let Some([arg]) = function_arguments::<1>(x) {
                        **out = -function!(csch, arg) * function!(coth, arg);
                    }
                })
                .with_evaluation_info(
                    EvaluationInfo::new().register(|args: &[Complex<Float>]| {
                        unary_eval_complex_float(args, csch_numeric_eval)
                    })
                    .register(|args: &[f64]| unary_eval_real_f64(args, csch_eval_f64))
                    .register(|args: &[Complex<f64>]| unary_eval_complex_real_f64(args, |z| Complex::new(1.0, 0.0) / z.sinh())),
                )
            }
        );

        let csch = symbols.pop().unwrap();
        let sech = symbols.pop().unwrap();
        let coth = symbols.pop().unwrap();
        let tanh = symbols.pop().unwrap();
        let cosh = symbols.pop().unwrap();
        let sinh = symbols.pop().unwrap();
        let csc = symbols.pop().unwrap();
        let sec = symbols.pop().unwrap();
        let cot = symbols.pop().unwrap();
        let tan = symbols.pop().unwrap();

        let asin = symbol!(
            "asin",
            norm = |x, out| {
                if let Some([arg]) = function_arguments::<1>(x) {
                    if arg.is_zero() {
                        out.to_num(Coefficient::zero());
                        return;
                    }
                    let _ = maybe_eval_unary_float_in_norm(arg, out, asin_numeric_eval);
                }
            },
            der = move |x, i, out| {
                if i == 0
                    && let Some([arg]) = function_arguments::<1>(x)
                {
                    let arg = arg.to_owned();
                    **out = Atom::num(1)
                        / function!(State::SQRT, Atom::num(1) - arg.clone().pow(Atom::num(2)));
                }
            },
            eval = EvaluationInfo::new()
                .register(|args: &[Complex<Float>]| {
                    unary_eval_complex_float(args, asin_numeric_eval)
                })
                .register(|args: &[f64]| unary_eval_real_f64(args, f64::asin))
                .register(|args: &[Complex<f64>]| {
                    unary_eval_complex_real_f64(args, |z| z.asin())
                })
        );

        let acos = symbol!(
            "acos",
            norm = |x, out| {
                if let Some([arg]) = function_arguments::<1>(x) {
                    if arg.is_one() {
                        out.to_num(Coefficient::zero());
                        return;
                    }
                    let _ = maybe_eval_unary_float_in_norm(arg, out, acos_numeric_eval);
                }
            },
            der = move |x, i, out| {
                if i == 0
                    && let Some([arg]) = function_arguments::<1>(x)
                {
                    let arg = arg.to_owned();
                    **out = -Atom::num(1)
                        / function!(State::SQRT, Atom::num(1) - arg.clone().pow(Atom::num(2)));
                }
            },
            eval = EvaluationInfo::new()
                .register(|args: &[Complex<Float>]| {
                    unary_eval_complex_float(args, acos_numeric_eval)
                })
                .register(|args: &[f64]| unary_eval_real_f64(args, f64::acos))
                .register(|args: &[Complex<f64>]| {
                    unary_eval_complex_real_f64(args, |z| z.acos())
                })
        );

        let atan = symbol!(
            "atan",
            norm = |x, out| {
                if let Some([arg]) = function_arguments::<1>(x) {
                    if arg.is_zero() {
                        out.to_num(Coefficient::zero());
                        return;
                    }
                    let _ = maybe_eval_unary_float_in_norm(arg, out, atan_numeric_eval);
                }
            },
            der = move |x, i, out| {
                if i == 0
                    && let Some([arg]) = function_arguments::<1>(x)
                {
                    let arg = arg.to_owned();
                    **out = Atom::num(1) / (Atom::num(1) + arg.pow(Atom::num(2)));
                }
            },
            eval = EvaluationInfo::new()
                .register(|args: &[Complex<Float>]| {
                    unary_eval_complex_float(args, atan_numeric_eval)
                })
                .register(|args: &[f64]| unary_eval_real_f64(args, f64::atan))
                .register(|args: &[Complex<f64>]| {
                    unary_eval_complex_real_f64(args, atan_eval_complex_f64)
                })
        );

        let acot = symbol!(
            "acot",
            norm = |x, out| {
                if let Some([arg]) = function_arguments::<1>(x) {
                    let _ = maybe_eval_unary_float_in_norm(arg, out, acot_numeric_eval);
                }
            },
            der = move |x, i, out| {
                if i == 0
                    && let Some([arg]) = function_arguments::<1>(x)
                {
                    let arg = arg.to_owned();
                    **out = -Atom::num(1) / (Atom::num(1) + arg.pow(Atom::num(2)));
                }
            },
            eval = EvaluationInfo::new()
                .register(|args: &[Complex<Float>]| {
                    unary_eval_complex_float(args, acot_numeric_eval)
                })
                .register(|args: &[f64]| unary_eval_real_f64(args, acot_eval_f64))
                .register(|args: &[Complex<f64>]| {
                    unary_eval_complex_real_f64(args, |z| {
                        atan_eval_complex_f64(Complex::new(1.0, 0.0) / z)
                    })
                })
        );

        let asec = symbol!(
            "asec",
            norm = |x, out| {
                if let Some([arg]) = function_arguments::<1>(x) {
                    if arg.is_one() {
                        out.to_num(Coefficient::zero());
                        return;
                    }
                    let _ = maybe_eval_unary_float_in_norm(arg, out, asec_numeric_eval);
                }
            },
            der = move |x, i, out| {
                if i == 0
                    && let Some([arg]) = function_arguments::<1>(x)
                {
                    let inv = Atom::num(1) / arg;
                    **out = Atom::num(1)
                        / (arg.pow(Atom::num(2))
                            * function!(State::SQRT, Atom::num(1) - inv.pow(Atom::num(2))));
                }
            },
            eval = EvaluationInfo::new()
                .register(|args: &[Complex<Float>]| {
                    unary_eval_complex_float(args, asec_numeric_eval)
                })
                .register(|args: &[f64]| unary_eval_real_f64(args, asec_eval_f64))
                .register(|args: &[Complex<f64>]| {
                    unary_eval_complex_real_f64(args, |z| (Complex::new(1.0, 0.0) / z).acos())
                })
        );

        let acsc = symbol!(
            "acsc",
            norm = |x, out| {
                if let Some([arg]) = function_arguments::<1>(x) {
                    let _ = maybe_eval_unary_float_in_norm(arg, out, acsc_numeric_eval);
                }
            },
            der = move |x, i, out| {
                if i == 0
                    && let Some([arg]) = function_arguments::<1>(x)
                {
                    let inv = Atom::num(1) / arg;
                    **out = -Atom::num(1)
                        / (arg.pow(Atom::num(2))
                            * function!(State::SQRT, Atom::num(1) - inv.pow(Atom::num(2))));
                }
            },
            eval = EvaluationInfo::new()
                .register(|args: &[Complex<Float>]| {
                    unary_eval_complex_float(args, acsc_numeric_eval)
                })
                .register(|args: &[f64]| unary_eval_real_f64(args, acsc_eval_f64))
                .register(|args: &[Complex<f64>]| {
                    unary_eval_complex_real_f64(args, |z| (Complex::new(1.0, 0.0) / z).asin())
                })
        );

        let asinh = symbol!(
            "asinh",
            norm = |x, out| {
                if let Some([arg]) = function_arguments::<1>(x) {
                    if arg.is_zero() {
                        out.to_num(Coefficient::zero());
                        return;
                    }
                    let _ = maybe_eval_unary_float_in_norm(arg, out, asinh_numeric_eval);
                }
            },
            der = move |x, i, out| {
                if i == 0
                    && let Some([arg]) = function_arguments::<1>(x)
                {
                    let arg = arg.to_owned();
                    **out =
                        Atom::num(1) / function!(State::SQRT, Atom::num(1) + arg.pow(Atom::num(2)));
                }
            },
            eval = EvaluationInfo::new()
                .register(|args: &[Complex<Float>]| {
                    unary_eval_complex_float(args, asinh_numeric_eval)
                })
                .register(|args: &[f64]| unary_eval_real_f64(args, f64::asinh))
                .register(|args: &[Complex<f64>]| {
                    unary_eval_complex_real_f64(args, |z| z.asinh())
                })
        );

        let acosh = symbol!(
            "acosh",
            norm = |x, out| {
                if let Some([arg]) = function_arguments::<1>(x) {
                    if arg.is_one() {
                        out.to_num(Coefficient::zero());
                        return;
                    }
                    let _ = maybe_eval_unary_float_in_norm(arg, out, acosh_numeric_eval);
                }
            },
            der = move |x, i, out| {
                if i == 0
                    && let Some([arg]) = function_arguments::<1>(x)
                {
                    let arg = arg.to_owned();
                    **out = Atom::num(1)
                        / (function!(State::SQRT, arg.clone() - Atom::num(1))
                            * function!(State::SQRT, arg + Atom::num(1)));
                }
            },
            eval = EvaluationInfo::new()
                .register(|args: &[Complex<Float>]| {
                    unary_eval_complex_float(args, acosh_numeric_eval)
                })
                .register(|args: &[f64]| unary_eval_real_f64(args, f64::acosh))
                .register(|args: &[Complex<f64>]| {
                    unary_eval_complex_real_f64(args, |z| z.acosh())
                })
        );

        let atanh = symbol!(
            "atanh",
            norm = |x, out| {
                if let Some([arg]) = function_arguments::<1>(x) {
                    if arg.is_zero() {
                        out.to_num(Coefficient::zero());
                        return;
                    }
                    let _ = maybe_eval_unary_float_in_norm(arg, out, atanh_numeric_eval);
                }
            },
            der = move |x, i, out| {
                if i == 0
                    && let Some([arg]) = function_arguments::<1>(x)
                {
                    let arg = arg.to_owned();
                    **out = Atom::num(1) / (Atom::num(1) - arg.pow(Atom::num(2)));
                }
            },
            eval = EvaluationInfo::new()
                .register(|args: &[Complex<Float>]| {
                    unary_eval_complex_float(args, atanh_numeric_eval)
                })
                .register(|args: &[f64]| unary_eval_real_f64(args, f64::atanh))
                .register(|args: &[Complex<f64>]| {
                    unary_eval_complex_real_f64(args, |z| z.atanh())
                })
        );

        let acoth = symbol!(
            "acoth",
            norm = |x, out| {
                if let Some([arg]) = function_arguments::<1>(x) {
                    let _ = maybe_eval_unary_float_in_norm(arg, out, acoth_numeric_eval);
                }
            },
            der = move |x, i, out| {
                if i == 0
                    && let Some([arg]) = function_arguments::<1>(x)
                {
                    let arg = arg.to_owned();
                    **out = Atom::num(1) / (Atom::num(1) - arg.pow(Atom::num(2)));
                }
            },
            eval = EvaluationInfo::new()
                .register(|args: &[Complex<Float>]| {
                    unary_eval_complex_float(args, acoth_numeric_eval)
                })
                .register(|args: &[f64]| unary_eval_real_f64(args, acoth_eval_f64))
                .register(|args: &[Complex<f64>]| {
                    unary_eval_complex_real_f64(args, |z| (Complex::new(1.0, 0.0) / z).atanh())
                })
        );

        let asech = symbol!(
            "asech",
            norm = |x, out| {
                if let Some([arg]) = function_arguments::<1>(x) {
                    if arg.is_one() {
                        out.to_num(Coefficient::zero());
                        return;
                    }
                    let _ = maybe_eval_unary_float_in_norm(arg, out, asech_numeric_eval);
                }
            },
            der = move |x, i, out| {
                if i == 0
                    && let Some([arg]) = function_arguments::<1>(x)
                {
                    let arg = arg.to_owned();
                    let inv = Atom::num(1) / arg.clone();
                    **out = -Atom::num(1)
                        / (arg.clone().pow(Atom::num(2))
                            * function!(State::SQRT, inv.clone() - Atom::num(1))
                            * function!(State::SQRT, inv + Atom::num(1)));
                }
            },
            eval = EvaluationInfo::new()
                .register(|args: &[Complex<Float>]| {
                    unary_eval_complex_float(args, asech_numeric_eval)
                })
                .register(|args: &[f64]| unary_eval_real_f64(args, asech_eval_f64))
                .register(|args: &[Complex<f64>]| {
                    unary_eval_complex_real_f64(args, |z| (Complex::new(1.0, 0.0) / z).acosh())
                })
        );

        let acsch = symbol!(
            "acsch",
            norm = |x, out| {
                if let Some([arg]) = function_arguments::<1>(x) {
                    let _ = maybe_eval_unary_float_in_norm(arg, out, acsch_numeric_eval);
                }
            },
            der = move |x, i, out| {
                if i == 0
                    && let Some([arg]) = function_arguments::<1>(x)
                {
                    let arg = arg.to_owned();
                    let inv = Atom::num(1) / arg.clone();
                    **out = -Atom::num(1)
                        / (arg.clone().pow(Atom::num(2))
                            * function!(State::SQRT, Atom::num(1) + inv.pow(Atom::num(2))));
                }
            },
            eval = EvaluationInfo::new()
                .register(|args: &[Complex<Float>]| {
                    unary_eval_complex_float(args, acsch_numeric_eval)
                })
                .register(|args: &[f64]| unary_eval_real_f64(args, acsch_eval_f64))
                .register(|args: &[Complex<f64>]| {
                    unary_eval_complex_real_f64(args, |z| (Complex::new(1.0, 0.0) / z).asinh())
                })
        );

        Self {
            tan,
            cot,
            sec,
            csc,
            asin,
            acos,
            atan,
            acot,
            asec,
            acsc,
            sinh,
            cosh,
            tanh,
            coth,
            sech,
            csch,
            asinh,
            acosh,
            atanh,
            acoth,
            asech,
            acsch,
        }
    }
}

impl BesselSymbols {
    fn new() -> Self {
        let bessel_j = symbol!(
            "bessel_j",
            norm = |x, out| {
                if let Some([nu, z]) = function_arguments::<2>(x) {
                    if z.is_zero()
                        && let Some(n) = atom_to_integer(nu)
                    {
                        if n == 0 {
                            out.to_num(Coefficient::one());
                        } else {
                            out.to_num(Coefficient::zero());
                        }
                        return;
                    }
                    let _ = maybe_eval_binary_float_in_norm(nu, z, out, bessel_j_numeric_eval);
                }
            },
            der = move |x, i, out| {
                if i == 1
                    && let Some([nu, z]) = function_arguments::<2>(x)
                {
                    let symbol = x.as_fun_view().unwrap().get_symbol();
                    let nu = nu.to_owned();
                    **out = (function!(symbol, nu.clone() - Atom::num(1), z)
                        - function!(symbol, nu + Atom::num(1), z))
                        / Atom::num(2);
                }
            },
            eval = EvaluationInfo::new()
                .with_tags(1)
                .register_tagged(|tags| {
                    let tags = tags.iter().map(|x| x.to_owned()).collect::<Vec<_>>();
                    Box::new(move |args: &[Complex<Float>]| {
                        tagged_unary_eval_complex_float(
                            "bessel_j",
                            &tags,
                            args,
                            bessel_j_numeric_eval,
                        )
                    })
                })
                .register_tagged(|tags| {
                    let order = complex_float_tag("bessel_j", tags, 53).ok();
                    Box::new(move |args: &[f64]| {
                        tagged_unary_eval_real_f64(order.clone(), args, bessel_j_numeric_eval)
                    })
                })
                .register_tagged(|tags| {
                    let order = complex_float_tag("bessel_j", tags, 53).ok();
                    Box::new(move |args: &[Complex<f64>]| {
                        tagged_unary_eval_complex_f64(order.clone(), args, bessel_j_numeric_eval)
                    })
                })
        );

        let bessel_y = symbol!(
            "bessel_y",
            norm = |x, out| {
                if let Some([nu, z]) = function_arguments::<2>(x) {
                    if z.is_zero() {
                        out.to_num(Coefficient::complex_infinity());
                        return;
                    }
                    let _ = maybe_eval_binary_float_in_norm(nu, z, out, bessel_y_numeric_eval);
                }
            },
            der = move |x, i, out| {
                if i == 1
                    && let Some([nu, z]) = function_arguments::<2>(x)
                {
                    let symbol = x.as_fun_view().unwrap().get_symbol();
                    let nu = nu.to_owned();
                    **out = (function!(symbol, nu.clone() - Atom::num(1), z)
                        - function!(symbol, nu + Atom::num(1), z))
                        / Atom::num(2);
                }
            },
            eval = EvaluationInfo::new()
                .with_tags(1)
                .register_tagged(|tags| {
                    let tags = tags.iter().map(|x| x.to_owned()).collect::<Vec<_>>();
                    Box::new(move |args: &[Complex<Float>]| {
                        tagged_unary_eval_complex_float(
                            "bessel_y",
                            &tags,
                            args,
                            bessel_y_numeric_eval,
                        )
                    })
                })
                .register_tagged(|tags| {
                    let order = complex_float_tag("bessel_y", tags, 53).ok();
                    Box::new(move |args: &[f64]| {
                        tagged_unary_eval_real_f64(order.clone(), args, bessel_y_numeric_eval)
                    })
                })
                .register_tagged(|tags| {
                    let order = complex_float_tag("bessel_y", tags, 53).ok();
                    Box::new(move |args: &[Complex<f64>]| {
                        tagged_unary_eval_complex_f64(order.clone(), args, bessel_y_numeric_eval)
                    })
                })
        );

        let bessel_i = symbol!(
            "bessel_i",
            norm = |x, out| {
                if let Some([nu, z]) = function_arguments::<2>(x) {
                    if z.is_zero()
                        && let Some(n) = atom_to_integer(nu)
                        && n >= 0
                    {
                        if n == 0 {
                            out.to_num(Coefficient::one());
                        } else {
                            out.to_num(Coefficient::zero());
                        }
                        return;
                    }
                    let _ = maybe_eval_binary_float_in_norm(nu, z, out, bessel_i_numeric_eval);
                }
            },
            der = move |x, i, out| {
                if i == 1
                    && let Some([nu, z]) = function_arguments::<2>(x)
                {
                    let symbol = x.as_fun_view().unwrap().get_symbol();
                    let nu = nu.to_owned();
                    **out = (function!(symbol, nu.clone() - Atom::num(1), z)
                        + function!(symbol, nu + Atom::num(1), z))
                        / Atom::num(2);
                }
            },
            eval = EvaluationInfo::new()
                .with_tags(1)
                .register_tagged(|tags| {
                    let tags = tags.iter().map(|x| x.to_owned()).collect::<Vec<_>>();
                    Box::new(move |args: &[Complex<Float>]| {
                        tagged_unary_eval_complex_float(
                            "bessel_i",
                            &tags,
                            args,
                            bessel_i_numeric_eval,
                        )
                    })
                })
                .register_tagged(|tags| {
                    let order = complex_float_tag("bessel_i", tags, 53).ok();
                    Box::new(move |args: &[f64]| {
                        tagged_unary_eval_real_f64(order.clone(), args, bessel_i_numeric_eval)
                    })
                })
                .register_tagged(|tags| {
                    let order = complex_float_tag("bessel_i", tags, 53).ok();
                    Box::new(move |args: &[Complex<f64>]| {
                        tagged_unary_eval_complex_f64(order.clone(), args, bessel_i_numeric_eval)
                    })
                })
        );

        let bessel_k = symbol!(
            "bessel_k",
            norm = |x, out| {
                if let Some([nu, z]) = function_arguments::<2>(x) {
                    if z.is_zero() {
                        out.to_num(Coefficient::complex_infinity());
                        return;
                    }
                    let _ = maybe_eval_binary_float_in_norm(nu, z, out, bessel_k_numeric_eval);
                }
            },
            der = move |x, i, out| {
                if i == 1
                    && let Some([nu, z]) = function_arguments::<2>(x)
                {
                    let symbol = x.as_fun_view().unwrap().get_symbol();
                    let nu = nu.to_owned();
                    **out = -(function!(symbol, nu.clone() - Atom::num(1), z)
                        + function!(symbol, nu + Atom::num(1), z))
                        / Atom::num(2);
                }
            },
            eval = EvaluationInfo::new()
                .with_tags(1)
                .register_tagged(|tags| {
                    let tags = tags.iter().map(|x| x.to_owned()).collect::<Vec<_>>();
                    Box::new(move |args: &[Complex<Float>]| {
                        tagged_unary_eval_complex_float(
                            "bessel_k",
                            &tags,
                            args,
                            bessel_k_numeric_eval,
                        )
                    })
                })
                .register_tagged(|tags| {
                    let order = complex_float_tag("bessel_k", tags, 53).ok();
                    Box::new(move |args: &[f64]| {
                        tagged_unary_eval_real_f64(order.clone(), args, bessel_k_numeric_eval)
                    })
                })
                .register_tagged(|tags| {
                    let order = complex_float_tag("bessel_k", tags, 53).ok();
                    Box::new(move |args: &[Complex<f64>]| {
                        tagged_unary_eval_complex_f64(order.clone(), args, bessel_k_numeric_eval)
                    })
                })
        );

        Self {
            bessel_j,
            bessel_y,
            bessel_i,
            bessel_k,
        }
    }
}

#[cfg(not(doctest))]
crate::_inventory::submit! {
    StateInitializer::new("symbolica::special_functions", || {
        let _ = SpecialSymbols::new();
        let _ = GeometricSymbols::new();
        let _ = BesselSymbols::new();
    }, &["symbolica"])
}

/// Return the built-in Euler gamma function symbol `gamma`.
///
/// `gamma(z)` is meromorphic with simple poles at the non-positive integers and no branch cuts.
pub fn gamma() -> Symbol {
    SPECIALS.gamma
}

/// Return the built-in Euler-Mascheroni constant symbol `euler_gamma`.
///
/// This is the constant `gamma_E`, approximately `0.57721`, used in expansions of `gamma`,
/// `polygamma`, and related special functions.
pub fn euler_gamma() -> Symbol {
    SPECIALS.euler_gamma
}

/// Return the built-in polygamma function symbol `polygamma`.
///
/// The first argument is the non-negative integer order. For fixed order, `polygamma(n, z)` is
/// meromorphic in `z` with poles at the non-positive integers and no branch cuts.
pub fn polygamma() -> Symbol {
    SPECIALS.polygamma
}

/// Return the built-in polylogarithm symbol `polylog`.
///
/// `polylog(s, z)` uses the principal branch in `z`; for generic `s` it has the standard branch cut
/// along the real interval `[1, +infinity)`.
pub fn polylog() -> Symbol {
    SPECIALS.polylog
}

/// Return the built-in tangent function symbol `tan`.
///
/// `tan(z)` is meromorphic with simple poles at `pi/2 + k pi`.
pub fn tan() -> Symbol {
    GEOMETRICS.tan
}

/// Return the built-in cotangent function symbol `cot`.
///
/// `cot(z)` is meromorphic with simple poles at `k pi`.
pub fn cot() -> Symbol {
    GEOMETRICS.cot
}

/// Return the built-in secant function symbol `sec`.
///
/// `sec(z)` is meromorphic with simple poles at `pi/2 + k pi`.
pub fn sec() -> Symbol {
    GEOMETRICS.sec
}

/// Return the built-in cosecant function symbol `csc`.
///
/// `csc(z)` is meromorphic with simple poles at `k pi`.
pub fn csc() -> Symbol {
    GEOMETRICS.csc
}

/// Return the built-in inverse sine symbol `asin`.
///
/// `asin(z)` uses the principal branch with branch cuts on `(-infinity, -1]` and `[1, +infinity)`.
pub fn asin() -> Symbol {
    GEOMETRICS.asin
}

/// Return the built-in inverse cosine symbol `acos`.
///
/// `acos(z)` uses the principal branch with branch cuts on `(-infinity, -1]` and `[1, +infinity)`.
pub fn acos() -> Symbol {
    GEOMETRICS.acos
}

/// Return the built-in inverse tangent symbol `atan`.
///
/// `atan(z)` uses the principal branch with branch cuts on the imaginary axis:
/// `(-i infinity, -i]` and `[i, i infinity)`.
pub fn atan() -> Symbol {
    GEOMETRICS.atan
}

/// Return the built-in inverse cotangent symbol `acot`.
///
/// `acot(z)` uses the principal branch with branch cuts on the imaginary axis:
/// `(-i infinity, -i]` and `[i, i infinity)`.
pub fn acot() -> Symbol {
    GEOMETRICS.acot
}

/// Return the built-in inverse secant symbol `asec`.
///
/// `asec(z)` uses the principal branch with branch cuts on `[-1, 1]`.
pub fn asec() -> Symbol {
    GEOMETRICS.asec
}

/// Return the built-in inverse cosecant symbol `acsc`.
///
/// `acsc(z)` uses the principal branch with branch cuts on `[-1, 1]`.
pub fn acsc() -> Symbol {
    GEOMETRICS.acsc
}

/// Return the built-in hyperbolic sine symbol `sinh`.
///
/// `sinh(z)` is entire.
pub fn sinh() -> Symbol {
    GEOMETRICS.sinh
}

/// Return the built-in hyperbolic cosine symbol `cosh`.
///
/// `cosh(z)` is entire.
pub fn cosh() -> Symbol {
    GEOMETRICS.cosh
}

/// Return the built-in hyperbolic tangent symbol `tanh`.
///
/// `tanh(z)` is meromorphic with simple poles at `i (pi/2 + k pi)`.
pub fn tanh() -> Symbol {
    GEOMETRICS.tanh
}

/// Return the built-in hyperbolic cotangent symbol `coth`.
///
/// `coth(z)` is meromorphic with simple poles at `i k pi`.
pub fn coth() -> Symbol {
    GEOMETRICS.coth
}

/// Return the built-in hyperbolic secant symbol `sech`.
///
/// `sech(z)` is meromorphic with simple poles at `i (pi/2 + k pi)`.
pub fn sech() -> Symbol {
    GEOMETRICS.sech
}

/// Return the built-in hyperbolic cosecant symbol `csch`.
///
/// `csch(z)` is meromorphic with simple poles at `i k pi`.
pub fn csch() -> Symbol {
    GEOMETRICS.csch
}

/// Return the built-in inverse hyperbolic sine symbol `asinh`.
///
/// `asinh(z)` uses the principal branch with branch cuts on the imaginary axis:
/// `(-i infinity, -i]` and `[i, i infinity)`.
pub fn asinh() -> Symbol {
    GEOMETRICS.asinh
}

/// Return the built-in inverse hyperbolic cosine symbol `acosh`.
///
/// `acosh(z)` uses the principal branch with branch cut `(-infinity, 1]`.
pub fn acosh() -> Symbol {
    GEOMETRICS.acosh
}

/// Return the built-in inverse hyperbolic tangent symbol `atanh`.
///
/// `atanh(z)` uses the principal branch with branch cuts on `(-infinity, -1]` and `[1, +infinity)`.
pub fn atanh() -> Symbol {
    GEOMETRICS.atanh
}

/// Return the built-in inverse hyperbolic cotangent symbol `acoth`.
///
/// `acoth(z)` uses the principal branch with branch cut `[-1, 1]`.
pub fn acoth() -> Symbol {
    GEOMETRICS.acoth
}

/// Return the built-in inverse hyperbolic secant symbol `asech`.
///
/// `asech(z)` uses the principal branch with branch cuts on `(-infinity, 0]` and `[1, +infinity)`.
pub fn asech() -> Symbol {
    GEOMETRICS.asech
}

/// Return the built-in inverse hyperbolic cosecant symbol `acsch`.
///
/// `acsch(z)` uses the principal branch with branch cuts on the imaginary interval `[-i, i]`.
pub fn acsch() -> Symbol {
    GEOMETRICS.acsch
}

/// Return the built-in cylindrical Bessel function of the first kind symbol `bessel_j`.
///
/// For fixed order `nu`, `bessel_j(nu, z)` is entire in `z`.
pub fn bessel_j() -> Symbol {
    BESSELS.bessel_j
}

/// Return the built-in cylindrical Bessel function of the second kind symbol `bessel_y`.
///
/// For fixed order `nu`, `bessel_y(nu, z)` uses the principal branch in `z` with branch cut
/// on `(-infinity, 0]`.
pub fn bessel_y() -> Symbol {
    BESSELS.bessel_y
}

/// Return the built-in modified Bessel function of the first kind symbol `bessel_i`.
///
/// For fixed order `nu`, `bessel_i(nu, z)` is entire in `z`.
pub fn bessel_i() -> Symbol {
    BESSELS.bessel_i
}

/// Return the built-in modified Bessel function of the second kind symbol `bessel_k`.
///
/// For fixed order `nu`, `bessel_k(nu, z)` uses the principal branch in `z` with branch cut
/// on `(-infinity, 0]`.
pub fn bessel_k() -> Symbol {
    BESSELS.bessel_k
}

fn unary_eval_complex_float<F>(args: &[Complex<Float>], evaluator: F) -> Complex<Float>
where
    F: FnOnce(&Complex<Float>, u32) -> Complex<Float>,
{
    let prec = args
        .first()
        .map(|x| x.re.prec().max(x.im.prec()))
        .unwrap_or(53);
    let [arg] = args else {
        return Complex::new(
            Float::with_val(53, rug::float::Special::Nan),
            Float::with_val(53, rug::float::Special::Nan),
        );
    };
    evaluator(arg, prec)
}

fn tagged_unary_eval_to_float(
    name: &str,
    tags: &[AtomView],
    args: &[Complex<Float>],
    prec: u32,
    evaluator: fn(&Complex<Float>, &Complex<Float>, u32) -> Option<Complex<Float>>,
) -> Result<Complex<Float>, String> {
    let order = complex_float_tag(name, tags, prec)?;
    let [arg] = args else {
        return Err(format!(
            "{name} expects exactly one argument after its order tag, got {}",
            args.len()
        ));
    };
    evaluator(&order, arg, prec).ok_or_else(|| format!("{name} numeric evaluation failed"))
}

fn tagged_unary_eval_complex_float(
    name: &str,
    tags: &[Atom],
    args: &[Complex<Float>],
    evaluator: fn(&Complex<Float>, &Complex<Float>, u32) -> Option<Complex<Float>>,
) -> Complex<Float> {
    let prec = args
        .first()
        .map(|x| x.re.prec().max(x.im.prec()))
        .unwrap_or(53);
    let tag_views = tags.iter().map(|x| x.as_view()).collect::<Vec<_>>();
    tagged_unary_eval_to_float(name, &tag_views, args, prec, evaluator).unwrap_or_else(|_| {
        Complex::new(
            Float::with_val(53, rug::float::Special::Nan),
            Float::with_val(53, rug::float::Special::Nan),
        )
    })
}

fn nonnegative_integer_tag(name: &str, tags: &[AtomView]) -> Result<u32, String> {
    let [tag] = tags else {
        return Err(format!(
            "{name} expects exactly one order tag, got {}",
            tags.len()
        ));
    };

    let Some(order) = atom_to_integer(*tag) else {
        return Err(format!("{name} order tag must be an integer"));
    };

    if order < 0 || order > u32::MAX as i64 {
        return Err(format!("{name} order tag must be a nonnegative integer"));
    }

    Ok(order as u32)
}

fn complex_float_tag(name: &str, tags: &[AtomView], prec: u32) -> Result<Complex<Float>, String> {
    let [tag] = tags else {
        return Err(format!(
            "{name} expects exactly one order tag, got {}",
            tags.len()
        ));
    };

    if let Ok(value) = Complex::<Float>::try_from(*tag) {
        return Ok(value);
    }

    if let Ok(value) = Complex::<Rational>::try_from(*tag) {
        return Ok(Complex::new(
            value.re.to_multi_prec_float(prec),
            value.im.to_multi_prec_float(prec),
        ));
    }

    Err(format!("{name} order tag must be numeric"))
}

fn unary_eval_real_f64(args: &[f64], evaluator: fn(f64) -> f64) -> f64 {
    let [arg] = args else {
        return f64::NAN;
    };
    evaluator(*arg)
}

fn unary_eval_f64(args: &[f64], evaluator: fn(&Complex<Float>, u32) -> Complex<Float>) -> f64 {
    let [arg] = args else {
        return f64::NAN;
    };
    evaluator(&Complex::new(Float::with_val(53, *arg), Float::new(53)), 53)
        .re
        .to_f64()
}

fn unary_eval_complex_f64(
    args: &[Complex<f64>],
    evaluator: fn(&Complex<Float>, u32) -> Complex<Float>,
) -> Complex<f64> {
    let [arg] = args else {
        return Complex::new(f64::NAN, f64::NAN);
    };
    evaluator(&complex_f64_to_float(arg, 53), 53).to_f64()
}

fn unary_eval_complex_real_f64<F>(args: &[Complex<f64>], evaluator: F) -> Complex<f64>
where
    F: FnOnce(Complex<f64>) -> Complex<f64>,
{
    let [arg] = args else {
        return Complex::new(f64::NAN, f64::NAN);
    };
    evaluator(arg.clone())
}

fn tagged_unary_eval_real_f64(
    order: Option<Complex<Float>>,
    args: &[f64],
    evaluator: fn(&Complex<Float>, &Complex<Float>, u32) -> Option<Complex<Float>>,
) -> f64 {
    let Some(order) = order else {
        return f64::NAN;
    };
    let [arg] = args else {
        return f64::NAN;
    };
    evaluator(
        &order,
        &Complex::new(Float::with_val(53, *arg), Float::new(53)),
        53,
    )
    .map(|z| z.re.to_f64())
    .unwrap_or(f64::NAN)
}

fn tagged_unary_eval_complex_f64(
    order: Option<Complex<Float>>,
    args: &[Complex<f64>],
    evaluator: fn(&Complex<Float>, &Complex<Float>, u32) -> Option<Complex<Float>>,
) -> Complex<f64> {
    let Some(order) = order else {
        return Complex::new(f64::NAN, f64::NAN);
    };
    let [arg] = args else {
        return Complex::new(f64::NAN, f64::NAN);
    };
    evaluator(&order, &complex_f64_to_float(arg, 53), 53)
        .map(|z| z.to_f64())
        .unwrap_or_else(|| Complex::new(f64::NAN, f64::NAN))
}

fn tan_numeric_eval(z: &Complex<Float>, _binary_prec: u32) -> Complex<Float> {
    z.tan()
}

fn cot_numeric_eval(z: &Complex<Float>, binary_prec: u32) -> Complex<Float> {
    complex_one(binary_prec) / z.clone().tan()
}

fn sec_numeric_eval(z: &Complex<Float>, binary_prec: u32) -> Complex<Float> {
    complex_one(binary_prec) / z.clone().cos()
}

fn csc_numeric_eval(z: &Complex<Float>, binary_prec: u32) -> Complex<Float> {
    complex_one(binary_prec) / z.clone().sin()
}

fn asin_numeric_eval(z: &Complex<Float>, _binary_prec: u32) -> Complex<Float> {
    z.asin()
}

fn acos_numeric_eval(z: &Complex<Float>, _binary_prec: u32) -> Complex<Float> {
    z.acos()
}

fn atan_numeric_eval(z: &Complex<Float>, binary_prec: u32) -> Complex<Float> {
    let i = complex_i(binary_prec);
    let one = complex_one(binary_prec);
    let two = Complex::new(Float::with_val(binary_prec, 2), Float::new(binary_prec));

    (i.clone() / two) * ((one.clone() - i.clone() * z.clone()).log() - (one + i * z.clone()).log())
}

fn acot_numeric_eval(z: &Complex<Float>, binary_prec: u32) -> Complex<Float> {
    atan_numeric_eval(&(complex_one(binary_prec) / z.clone()), binary_prec)
}

fn asec_numeric_eval(z: &Complex<Float>, binary_prec: u32) -> Complex<Float> {
    acos_numeric_eval(&(complex_one(binary_prec) / z.clone()), binary_prec)
}

fn acsc_numeric_eval(z: &Complex<Float>, binary_prec: u32) -> Complex<Float> {
    asin_numeric_eval(&(complex_one(binary_prec) / z.clone()), binary_prec)
}

fn sinh_numeric_eval(z: &Complex<Float>, _binary_prec: u32) -> Complex<Float> {
    z.clone().sinh()
}

fn cosh_numeric_eval(z: &Complex<Float>, _binary_prec: u32) -> Complex<Float> {
    z.clone().cosh()
}

fn tanh_numeric_eval(z: &Complex<Float>, _binary_prec: u32) -> Complex<Float> {
    z.clone().tanh()
}

fn coth_numeric_eval(z: &Complex<Float>, binary_prec: u32) -> Complex<Float> {
    complex_one(binary_prec) / z.clone().tanh()
}

fn sech_numeric_eval(z: &Complex<Float>, binary_prec: u32) -> Complex<Float> {
    complex_one(binary_prec) / z.clone().cosh()
}

fn csch_numeric_eval(z: &Complex<Float>, binary_prec: u32) -> Complex<Float> {
    complex_one(binary_prec) / z.clone().sinh()
}

fn asinh_numeric_eval(z: &Complex<Float>, _binary_prec: u32) -> Complex<Float> {
    z.clone().asinh()
}

fn acosh_numeric_eval(z: &Complex<Float>, _binary_prec: u32) -> Complex<Float> {
    z.clone().acosh()
}

fn atanh_numeric_eval(z: &Complex<Float>, _binary_prec: u32) -> Complex<Float> {
    z.clone().atanh()
}

fn acoth_numeric_eval(z: &Complex<Float>, binary_prec: u32) -> Complex<Float> {
    atanh_numeric_eval(&(complex_one(binary_prec) / z.clone()), binary_prec)
}

fn asech_numeric_eval(z: &Complex<Float>, binary_prec: u32) -> Complex<Float> {
    acosh_numeric_eval(&(complex_one(binary_prec) / z.clone()), binary_prec)
}

fn acsch_numeric_eval(z: &Complex<Float>, binary_prec: u32) -> Complex<Float> {
    asinh_numeric_eval(&(complex_one(binary_prec) / z.clone()), binary_prec)
}

fn complex_one(prec: u32) -> Complex<Float> {
    Complex::new(Float::with_val(prec, 1), Float::new(prec))
}

fn complex_i(prec: u32) -> Complex<Float> {
    Complex::new(Float::new(prec), Float::with_val(prec, 1))
}

fn complex_i_f64() -> Complex<f64> {
    Complex::new(0.0, 1.0)
}

fn atan_eval_complex_f64(z: Complex<f64>) -> Complex<f64> {
    let i = complex_i_f64();
    let one = Complex::new(1.0, 0.0);
    let two = Complex::new(2.0, 0.0);
    (i / two) * ((one - i * z.clone()).log() - (one + i * z).log())
}

fn cot_eval_f64(x: f64) -> f64 {
    1.0 / x.tan()
}

fn sec_eval_f64(x: f64) -> f64 {
    1.0 / x.cos()
}

fn csc_eval_f64(x: f64) -> f64 {
    1.0 / x.sin()
}

fn acot_eval_f64(x: f64) -> f64 {
    (1.0 / x).atan()
}

fn asec_eval_f64(x: f64) -> f64 {
    (1.0 / x).acos()
}

fn acsc_eval_f64(x: f64) -> f64 {
    (1.0 / x).asin()
}

fn coth_eval_f64(x: f64) -> f64 {
    1.0 / x.tanh()
}

fn sech_eval_f64(x: f64) -> f64 {
    1.0 / x.cosh()
}

fn csch_eval_f64(x: f64) -> f64 {
    1.0 / x.sinh()
}

fn acoth_eval_f64(x: f64) -> f64 {
    (1.0 / x).atanh()
}

fn asech_eval_f64(x: f64) -> f64 {
    (1.0 / x).acosh()
}

fn acsch_eval_f64(x: f64) -> f64 {
    (1.0 / x).asinh()
}

fn complex_float_is_zero(value: &Complex<Float>) -> bool {
    value.norm().re.to_f64() == 0.0
}

fn bessel_j_numeric_eval(
    order: &Complex<Float>,
    z: &Complex<Float>,
    binary_prec: u32,
) -> Option<Complex<Float>> {
    if complex_float_is_zero(z) {
        if let Some(n) = complex_float_to_integer(order)
            && n >= 0
        {
            return Some(if n == 0 {
                complex_one(binary_prec)
            } else {
                Complex::new(Float::new(binary_prec), Float::new(binary_prec))
            });
        }
    }

    if let Some(n) = complex_float_to_integer(order)
        && n < 0
    {
        let positive = Complex::new(Float::with_val(binary_prec, -n), Float::new(binary_prec));
        let value = bessel_j_numeric_eval(&positive, z, binary_prec)?;
        return Some(if n % 2 == 0 { value } else { -value });
    }

    bessel_series_eval(order, z, binary_prec, true)
}

fn bessel_i_numeric_eval(
    order: &Complex<Float>,
    z: &Complex<Float>,
    binary_prec: u32,
) -> Option<Complex<Float>> {
    if complex_float_is_zero(z) {
        if let Some(n) = complex_float_to_integer(order)
            && n >= 0
        {
            return Some(if n == 0 {
                complex_one(binary_prec)
            } else {
                Complex::new(Float::new(binary_prec), Float::new(binary_prec))
            });
        }
    }

    if let Some(n) = complex_float_to_integer(order)
        && n < 0
    {
        let positive = Complex::new(Float::with_val(binary_prec, -n), Float::new(binary_prec));
        return bessel_i_numeric_eval(&positive, z, binary_prec);
    }

    bessel_series_eval(order, z, binary_prec, false)
}

fn bessel_y_numeric_eval(
    order: &Complex<Float>,
    z: &Complex<Float>,
    binary_prec: u32,
) -> Option<Complex<Float>> {
    if complex_float_is_zero(z) {
        return Some(Complex::new(
            Float::with_val(binary_prec, f64::INFINITY),
            Float::new(binary_prec),
        ));
    }

    if let Some(n) = complex_float_to_integer(order)
        && n < 0
    {
        let positive = Complex::new(Float::with_val(binary_prec, -n), Float::new(binary_prec));
        let value = bessel_y_numeric_eval(&positive, z, binary_prec)?;
        return Some(if n % 2 == 0 { value } else { -value });
    }

    let order = bessel_regularized_order(order, binary_prec);
    let pi_order = complex_pi(binary_prec) * order.clone();
    let j_pos = bessel_j_numeric_eval(&order, z, binary_prec)?;
    let j_neg = bessel_j_numeric_eval(&(-order.clone()), z, binary_prec)?;
    Some((j_pos * pi_order.clone().cos() - j_neg) / pi_order.sin())
}

fn bessel_k_numeric_eval(
    order: &Complex<Float>,
    z: &Complex<Float>,
    binary_prec: u32,
) -> Option<Complex<Float>> {
    if complex_float_is_zero(z) {
        return Some(Complex::new(
            Float::with_val(binary_prec, f64::INFINITY),
            Float::new(binary_prec),
        ));
    }

    let order = if let Some(n) = complex_float_to_integer(order) {
        Complex::new(
            Float::with_val(binary_prec, n.abs()),
            Float::new(binary_prec),
        )
    } else {
        order.clone()
    };
    let order = bessel_regularized_order(&order, binary_prec);
    let pi_order = complex_pi(binary_prec) * order.clone();
    let i_neg = bessel_i_numeric_eval(&(-order.clone()), z, binary_prec)?;
    let i_pos = bessel_i_numeric_eval(&order, z, binary_prec)?;
    let pref = Complex::new(
        Float::with_val(binary_prec, Constant::Pi) / Float::with_val(binary_prec, 2),
        Float::new(binary_prec),
    );
    Some(pref * (i_neg - i_pos) / pi_order.sin())
}

fn bessel_series_eval(
    order: &Complex<Float>,
    z: &Complex<Float>,
    binary_prec: u32,
    alternating: bool,
) -> Option<Complex<Float>> {
    let zero = Float::new(binary_prec);
    let z_half = z.clone() / Complex::new(Float::with_val(binary_prec, 2), zero.clone());
    let one = complex_one(binary_prec);
    let gamma = gamma_numeric_eval(&(order.clone() + one), binary_prec);
    let mut term = z_half.clone().powf(order) / gamma;
    let mut sum = term.clone();
    let z_half_sq = z_half.clone() * z_half;
    let threshold = 2f64.powi(-(binary_prec.min(900) as i32));

    for k in 1..(16 * binary_prec.max(16)) {
        let kf = Float::with_val(binary_prec, k);
        let denom = Complex::new(kf.clone(), zero.clone())
            * (order.clone() + Complex::new(kf, zero.clone()));
        let factor = if alternating {
            -z_half_sq.clone()
        } else {
            z_half_sq.clone()
        };
        term = term * factor / denom;
        let term_size = term.norm().re.to_f64().abs();
        sum += term.clone();
        if k > 16 && (term_size == 0.0 || term_size < threshold) {
            return Some(sum);
        }
    }

    Some(sum)
}

fn bessel_regularized_order(order: &Complex<Float>, binary_prec: u32) -> Complex<Float> {
    if let Some(n) = complex_float_to_integer(order) {
        let eps = bessel_order_epsilon(binary_prec);
        Complex::new(
            Float::with_val(binary_prec, n) + eps,
            Float::new(binary_prec),
        )
    } else {
        order.clone()
    }
}

fn bessel_order_epsilon(binary_prec: u32) -> Float {
    let eps = 2f64.powi(-((binary_prec.min(200) as i32) / 3));
    Float::with_val(binary_prec, eps.max(1e-8))
}

fn complex_pi(prec: u32) -> Complex<Float> {
    Complex::new(Float::with_val(prec, Constant::Pi), Float::new(prec))
}

fn polygamma_eval_f64(order: Option<u32>, args: &[f64]) -> f64 {
    let Some(order) = order else {
        return f64::NAN;
    };
    let [arg] = args else {
        return f64::NAN;
    };
    polygamma_numeric_eval(
        order,
        &Complex::new(Float::with_val(53, *arg), Float::new(53)),
        53,
    )
    .re
    .to_f64()
}

fn polygamma_eval_complex_f64(order: Option<u32>, args: &[Complex<f64>]) -> Complex<f64> {
    let Some(order) = order else {
        return Complex::new(f64::NAN, f64::NAN);
    };
    let [arg] = args else {
        return Complex::new(f64::NAN, f64::NAN);
    };
    polygamma_numeric_eval(order, &complex_f64_to_float(arg, 53), 53).to_f64()
}

fn polylog_eval_f64(order: Option<Complex<Float>>, args: &[f64]) -> f64 {
    let Some(order) = order else {
        return f64::NAN;
    };
    let [arg] = args else {
        return f64::NAN;
    };
    let arg = Complex::new(Float::with_val(53, *arg), Float::new(53));
    polylog_numeric_eval(&order, &arg, 53)
        .map(|x| x.re.to_f64())
        .unwrap_or(f64::NAN)
}

fn polylog_eval_complex_f64(order: Option<Complex<Float>>, args: &[Complex<f64>]) -> Complex<f64> {
    let Some(order) = order else {
        return Complex::new(f64::NAN, f64::NAN);
    };
    let [arg] = args else {
        return Complex::new(f64::NAN, f64::NAN);
    };
    polylog_numeric_eval(&order, &complex_f64_to_float(arg, 53), 53)
        .map(|x| x.to_f64())
        .unwrap_or_else(|| Complex::new(f64::NAN, f64::NAN))
}

fn complex_f64_to_float(value: &Complex<f64>, prec: u32) -> Complex<Float> {
    Complex::new(
        Float::with_val(prec, value.re),
        Float::with_val(prec, value.im),
    )
}

fn complex_float_to_integer(value: &Complex<Float>) -> Option<i64> {
    if value.im.to_f64().abs() > 1e-12 {
        return None;
    }
    let re = value.re.to_f64();
    if !re.is_finite() {
        return None;
    }
    let rounded = re.round();
    if (re - rounded).abs() > 1e-12 || rounded < i64::MIN as f64 || rounded > i64::MAX as f64 {
        return None;
    }
    Some(rounded as i64)
}

fn function_arguments<'a, const N: usize>(view: AtomView<'a>) -> Option<[AtomView<'a>; N]> {
    let AtomView::Fun(fun) = view else {
        return None;
    };

    fun.iter().try_into().ok()
}

fn gamma_exact_rational(rat: &Rational) -> Option<Atom> {
    let denominator = rat.denominator();

    if denominator == 1 {
        let numerator = rat.numerator();
        if numerator <= 0 {
            return Some(Atom::num(Coefficient::complex_infinity()));
        }

        let n = numerator.to_i64()?;
        if n > u32::MAX as i64 {
            return None;
        }

        return Some(Atom::num(Integer::factorial((n - 1) as u32)));
    }

    if denominator == 2 {
        let numerator = rat.numerator().to_i64()?;
        if numerator % 2 == 0 {
            return None;
        }

        let prefactor = gamma_half_integer_prefactor(numerator);
        return Some(Atom::num(prefactor) * Atom::from(State::PI).pow(Atom::num((1, 2))));
    }

    None
}

fn polygamma_order_zero_exact_rational(rat: &Rational) -> Option<Atom> {
    let denominator = rat.denominator();

    if denominator == 1 {
        let numerator = rat.numerator();
        if numerator <= 0 {
            return Some(Atom::num(Coefficient::complex_infinity()));
        }

        let n = numerator.to_i64()?;
        if n > u32::MAX as i64 {
            return None;
        }

        return Some(Atom::num(harmonic_rational((n - 1) as u32)) - Atom::from(euler_gamma()));
    }

    if denominator == 2 {
        let numerator = rat.numerator().to_i64()?;
        if numerator % 2 == 0 {
            return None;
        }

        return Some(
            polygamma_order_zero_half_integer_base()
                + Atom::num(polygamma_order_zero_half_integer_shift(numerator)),
        );
    }

    None
}

fn polygamma_exact_rational(order: u32, rat: &Rational) -> Option<Atom> {
    if order == 0 {
        return polygamma_order_zero_exact_rational(rat);
    }

    if *rat != Rational::one() {
        return None;
    }

    match order {
        1 => Some(Atom::from(State::PI).pow(2) / 6),
        3 => Some(Atom::from(State::PI).pow(4) / 15),
        _ => None,
    }
}

fn gamma_half_integer_prefactor(numerator: i64) -> Rational {
    let one = Rational::from((1, 1));
    let target = Rational::from((numerator, 2));
    let mut current = Rational::from((1, 2));
    let mut prefactor = Rational::from((1, 1));

    while current < target {
        prefactor *= &current;
        current += &one;
    }

    while current > target {
        current -= &one;
        prefactor /= &current;
    }

    prefactor
}

fn harmonic_rational(n: u32) -> Rational {
    let mut sum = Rational::from((0, 1));
    let one = Rational::from((1, 1));

    for k in 1..=n {
        sum += &(one.clone() / Rational::from((k as i64, 1)));
    }

    sum
}

fn polygamma_order_zero_half_integer_base() -> Atom {
    -Atom::from(euler_gamma()) - Atom::num(2) * function!(State::LOG, Atom::num(2))
}

fn polygamma_order_zero_half_integer_shift(numerator: i64) -> Rational {
    let one = Rational::from((1, 1));
    let target = Rational::from((numerator, 2));
    let mut current = Rational::from((1, 2));
    let mut correction = Rational::from((0, 1));

    while current < target {
        correction += &(one.clone() / current.clone());
        current += &one;
    }

    while current > target {
        current -= &one;
        correction -= &(one.clone() / current.clone());
    }

    correction
}

fn is_tan_pole(point: AtomView) -> bool {
    let Some(ratio) = rational_multiple_of_pi(point) else {
        return false;
    };

    ratio.denominator() == 2 && ratio.numerator().to_i64().is_some_and(|n| n % 2 != 0)
}

fn is_cot_csc_pole(point: AtomView) -> bool {
    rational_multiple_of_pi(point).is_some_and(|r| r.denominator() == 1)
}

fn rational_multiple_of_pi(point: AtomView) -> Option<Rational> {
    match point {
        AtomView::Num(_) => {
            let r = Rational::try_from(point).ok()?;
            if r.is_zero() {
                Some(Rational::zero())
            } else {
                None
            }
        }
        AtomView::Var(v) => {
            if v.get_symbol() == State::PI {
                Some(Rational::one())
            } else {
                None
            }
        }
        AtomView::Mul(m) => {
            let mut coeff = Rational::one();
            let mut has_pi = false;

            for factor in m {
                if let Ok(r) = Rational::try_from(factor) {
                    coeff *= &r;
                    continue;
                }

                if let AtomView::Var(v) = factor
                    && v.get_symbol() == State::PI
                    && !has_pi
                {
                    has_pi = true;
                    continue;
                }

                return None;
            }

            if has_pi { Some(coeff) } else { None }
        }
        AtomView::Add(a) => {
            let mut sum = Rational::zero();
            for term in a {
                sum += &rational_multiple_of_pi(term)?;
            }
            Some(sum)
        }
        _ => None,
    }
}

fn atom_float_precision(value: AtomView) -> Option<u32> {
    let AtomView::Num(number) = value else {
        return None;
    };

    match number.get_coeff_view() {
        CoefficientView::Float(r, i) => Some(r.to_float().prec().max(i.to_float().prec())),
        _ => None,
    }
}

fn atom_to_integer(value: AtomView) -> Option<i64> {
    let rat = Rational::try_from(value).ok()?;
    if !rat.is_integer() {
        return None;
    }
    rat.numerator().to_i64()
}

fn maybe_eval_unary_float_in_norm(
    arg: AtomView,
    out: &mut Settable<Atom>,
    evaluator: fn(&Complex<Float>, u32) -> Complex<Float>,
) -> bool {
    let Some(prec) = atom_float_precision(arg) else {
        return false;
    };
    let Some(z) = atom_to_complex_float(arg, prec) else {
        return false;
    };

    out.to_num(evaluator(&z, prec).into());
    true
}

fn maybe_eval_polygamma_float_in_norm(order: u32, arg: AtomView, out: &mut Settable<Atom>) -> bool {
    let Some(prec) = atom_float_precision(arg) else {
        return false;
    };
    let Some(z) = atom_to_complex_float(arg, prec) else {
        return false;
    };

    out.to_num(polygamma_numeric_eval(order, &z, prec).into());
    true
}

fn maybe_eval_binary_float_in_norm(
    lhs: AtomView,
    rhs: AtomView,
    out: &mut Settable<Atom>,
    evaluator: fn(&Complex<Float>, &Complex<Float>, u32) -> Option<Complex<Float>>,
) -> bool {
    let precision = atom_float_precision(lhs).or_else(|| atom_float_precision(rhs));
    let Some(prec) = precision else {
        return false;
    };
    let Some(lhs_float) = atom_to_complex_float(lhs, prec) else {
        return false;
    };
    let Some(rhs_float) = atom_to_complex_float(rhs, prec) else {
        return false;
    };
    let Some(value) = evaluator(&lhs_float, &rhs_float, prec) else {
        return false;
    };
    out.to_num(value.into());
    true
}

fn atom_to_complex_float(value: AtomView, binary_prec: u32) -> Option<Complex<Float>> {
    let AtomView::Num(number) = value else {
        return None;
    };

    match number.get_coeff_view() {
        CoefficientView::Natural(nr, dr, ni, di) => Some(Complex::new(
            Float::with_val(binary_prec, nr) / Float::with_val(binary_prec, dr),
            Float::with_val(binary_prec, ni) / Float::with_val(binary_prec, di),
        )),
        CoefficientView::Large(r, i) => Some(Complex::new(
            r.to_rat().to_multi_prec_float(binary_prec),
            i.to_rat().to_multi_prec_float(binary_prec),
        )),
        CoefficientView::Float(r, i) => {
            let mut re = r.to_float();
            let mut im = i.to_float();
            if re.prec() > binary_prec {
                re.set_prec(binary_prec);
            }
            if im.prec() > binary_prec {
                im.set_prec(binary_prec);
            }
            Some(Complex::new(re, im))
        }
        _ => None,
    }
}

fn gamma_numeric_eval(z: &Complex<Float>, binary_prec: u32) -> Complex<Float> {
    if z.im.to_f64() == 0.0 {
        Complex::new(
            z.re.clone().into_inner().gamma().into(),
            Float::new(binary_prec),
        )
    } else {
        gamma_complex_spouge(z, binary_prec)
    }
}

fn polygamma_order_zero_numeric_eval(z: &Complex<Float>, binary_prec: u32) -> Complex<Float> {
    if z.im.to_f64() == 0.0 {
        Complex::new(
            z.re.clone().into_inner().digamma().into(),
            Float::new(binary_prec),
        )
    } else {
        polygamma_order_zero_complex(z, binary_prec)
    }
}

fn polygamma_numeric_eval(order: u32, z: &Complex<Float>, binary_prec: u32) -> Complex<Float> {
    if order == 0 {
        return polygamma_order_zero_numeric_eval(z, binary_prec);
    }

    let zero = Float::new(binary_prec);
    let one = Float::with_val(binary_prec, 1);
    let exponent = Complex::new(Float::with_val(binary_prec, order + 1), zero.clone());
    let factorial = Float::with_val(binary_prec, Integer::factorial(order).to_multi_prec());
    let sign = if order % 2 == 0 {
        -factorial
    } else {
        factorial
    };
    let prefactor = Complex::new(sign, zero.clone());

    let mut shifted = z.clone();
    let mut correction = Complex::new(zero.clone(), zero.clone());

    while shifted.re.to_f64() < 8.0 {
        let denom = shifted.clone().powf(&exponent);
        correction += prefactor.clone() / denom;
        shifted += Complex::new(one.clone(), zero.clone());
    }

    let threshold = 2f64.powi(-(binary_prec.min(900) as i32));
    let mut sum = Complex::new(zero.clone(), zero.clone());
    for n in 0..(8 * binary_prec.max(16)) {
        let shift = Complex::new(Float::with_val(binary_prec, n), zero.clone());
        let term = prefactor.clone() / (shifted.clone() + shift).powf(&exponent);
        let term_size = term.norm().re.to_f64().abs();
        sum += term;
        if n > 16 && (term_size == 0.0 || term_size < threshold) {
            break;
        }
    }

    correction + sum
}

fn polylog_exact(s: AtomView, z: AtomView) -> Option<Atom> {
    if z.is_zero() {
        return Some(Atom::new());
    }

    if let Some(order) = atom_to_integer(s) {
        if order == 0 {
            return Some(z.to_owned() / (Atom::num(1) - z));
        }

        if order == 1 {
            return Some(-function!(State::LOG, Atom::num(1) - z));
        }
    }

    None
}

fn polylog_numeric_eval(
    s: &Complex<Float>,
    z: &Complex<Float>,
    binary_prec: u32,
) -> Option<Complex<Float>> {
    let zero = Float::new(binary_prec);
    let one = Float::with_val(binary_prec, 1);

    if z.norm().re.to_f64() == 0.0 {
        return Some(Complex::new(zero.clone(), zero.clone()));
    }

    if s.im.to_f64() == 0.0 && s.re.to_f64() == 0.0 {
        return Some(z.clone() / (Complex::new(one.clone(), zero.clone()) - z.clone()));
    }

    if s.im.to_f64() == 0.0 && s.re.to_f64() == 1.0 {
        return Some(-(Complex::new(one.clone(), zero.clone()) - z.clone()).log());
    }

    if s.im.to_f64() == 0.0 && s.re.to_f64() == 2.0 && z.im.to_f64() == 0.0 {
        return Some(Complex::new(
            z.re.clone().into_inner().li2().into(),
            zero.clone(),
        ));
    }

    if z.norm().re.to_f64() >= 0.95 {
        return None;
    }

    let threshold = 2f64.powi(-(binary_prec.min(900) as i32));
    let mut z_pow = z.clone();
    let mut sum = Complex::new(zero.clone(), zero.clone());

    for k in 1..(12 * binary_prec.max(16)) {
        let base = Complex::new(Float::with_val(binary_prec, k), zero.clone());
        let denom = base.powf(s);
        let term = z_pow.clone() / denom;
        let term_size = term.norm().re.to_f64().abs();
        sum += term;
        if k > 16 && (term_size == 0.0 || term_size < threshold) {
            return Some(sum);
        }
        z_pow *= z.clone();
    }

    Some(sum)
}

fn gamma_complex_spouge(z: &Complex<Float>, binary_prec: u32) -> Complex<Float> {
    let zero = Float::new(binary_prec);
    let one = Float::with_val(binary_prec, 1);
    let half = Float::with_val(binary_prec, 1) / Float::with_val(binary_prec, 2);
    let pi = Float::with_val(binary_prec, Constant::Pi);

    if z.re.to_f64() < 0.5 {
        let numerator = Complex::new(pi.clone(), zero.clone());
        let pi_z = Complex::new(pi, zero.clone()) * z.clone();
        let reflected = Complex::new(one.clone(), zero.clone()) - z.clone();
        return numerator / (pi_z.sin() * gamma_complex_spouge(&reflected, binary_prec));
    }

    let a = spouge_parameter(binary_prec);
    let mut sum = Complex::new(
        (Float::with_val(binary_prec, 2) * Float::with_val(binary_prec, Constant::Pi)).sqrt(),
        zero.clone(),
    );

    for k in 1..a {
        let coeff = spouge_coefficient(a, k, binary_prec);
        let denom = z.clone() + Complex::new(Float::with_val(binary_prec, k - 1), zero.clone());
        sum += Complex::new(coeff, zero.clone()) / denom;
    }

    let t = z.clone() + Complex::new(Float::with_val(binary_prec, a - 1), zero.clone());
    let exponent = z.clone() - Complex::new(half, zero);
    t.powf(&exponent) * (-t).exp() * sum
}

fn polygamma_order_zero_complex(z: &Complex<Float>, binary_prec: u32) -> Complex<Float> {
    let zero = Float::new(binary_prec);
    let one = Float::with_val(binary_prec, 1);
    let pi = Float::with_val(binary_prec, Constant::Pi);

    if z.re.to_f64() < 0.5 {
        let pi_z = Complex::new(pi.clone(), zero.clone()) * z.clone();
        let reflected = Complex::new(one.clone(), zero.clone()) - z.clone();
        return polygamma_order_zero_complex(&reflected, binary_prec)
            - Complex::new(pi, zero.clone()) / pi_z.tan();
    }

    let mut shifted = z.clone();
    let mut correction = Complex::new(zero.clone(), zero.clone());
    while shifted.re.to_f64() < 8.0 {
        correction -= Complex::new(one.clone(), zero.clone()) / shifted.clone();
        shifted += Complex::new(one.clone(), zero.clone());
    }

    let mut sum = Complex::new(-Float::with_val(binary_prec, Constant::Euler), zero.clone());
    let threshold = 2f64.powi(-(binary_prec.min(900) as i32));

    for n in 0..(8 * binary_prec.max(16)) {
        let n1 = Float::with_val(binary_prec, n + 1);
        let term = Complex::new(one.clone() / &n1, zero.clone())
            - Complex::new(one.clone(), zero.clone())
                / (shifted.clone() + Complex::new(n1, zero.clone()));
        let term_size = term.norm().re.to_f64().abs();
        sum += term;
        if n > 16 && (term_size == 0.0 || term_size < threshold) {
            break;
        }
    }

    correction + sum
}

fn spouge_parameter(binary_prec: u32) -> u32 {
    ((binary_prec as f64 / (2.0 * std::f64::consts::PI).log2()).ceil() as u32).max(12) + 2
}

fn spouge_coefficient(a: u32, k: u32, binary_prec: u32) -> Float {
    let half = Float::with_val(binary_prec, 1) / Float::with_val(binary_prec, 2);
    let a_minus_k = Float::with_val(binary_prec, a - k);
    let exponent = Float::with_val(binary_prec, k) - half;
    let factorial = Float::with_val(binary_prec, Integer::factorial(k - 1).to_multi_prec());
    let mut coeff = a_minus_k.powf(&exponent) * a_minus_k.exp() / factorial;

    if k % 2 == 0 {
        coeff = -coeff;
    }

    coeff
}

#[cfg(test)]
mod tests {
    use std::{
        io::Write,
        process::{Command, Stdio},
    };

    use crate::{
        atom::{Atom, AtomCore},
        coefficient::Coefficient,
        domains::float::{Complex, Float, Real, RealLike},
        evaluate::{FunctionMap, OptimizationSettings},
        parse, symbol,
    };

    #[test]
    fn gamma_exact_normalization() {
        assert_eq!(parse!("gamma(5)"), Atom::num(24));
        assert_eq!(parse!("gamma(1/2)"), parse!("pi^(1/2)"));
        assert_eq!(parse!("gamma(-5/2)"), parse!("-8/15*pi^(1/2)"));
        assert_eq!(parse!("gamma(0.3)"), parse!("2.991568987687591"));
        assert_eq!(
            parse!("gamma(0)"),
            Atom::num(Coefficient::complex_infinity())
        );
    }

    #[test]
    fn gamma_derivative() {
        assert_eq!(
            parse!("gamma(x)").derivative(symbol!("x")),
            parse!("gamma(x)*polygamma(0,x)")
        );
    }

    #[test]
    fn gamma_laurent_series() {
        let x = symbol!("x");
        assert_eq!(
            parse!("gamma(x)")
                .series(x, Atom::num(0).as_view(), 3.into(), true)
                .unwrap()
                .to_atom(),
            parse!(
                "x^-1-euler_gamma+1/2*(euler_gamma^2+1/6*pi^2)*x+1/6*(-euler_gamma^3-1/2*euler_gamma*pi^2+polygamma(2,1))*x^2+1/24*(euler_gamma^4+euler_gamma^2*pi^2-4*euler_gamma*polygamma(2,1)+3/20*pi^4)*x^3"
            )
        );
        assert_eq!(
            parse!("gamma(x)")
                .series(x, Atom::num(0).as_view(), 0.into(), true)
                .unwrap()
                .to_atom(),
            parse!("x^-1-euler_gamma")
        );

        assert_eq!(
            parse!("gamma(x-1)")
                .series(x, Atom::num(1).as_view(), 0.into(), true)
                .unwrap()
                .to_atom(),
            parse!("(x-1)^-1-euler_gamma")
        );
    }

    #[test]
    fn gamma_matches_ginac() {
        if Command::new("ginsh").arg("--help").output().is_err() {
            return;
        }

        let real_expected = ginsh_gamma("1.25", 50);
        let real_actual = Complex::<Float>::try_from(parse!("gamma(5/4)").to_float(50)).unwrap();
        assert_close_complex(&real_actual, &real_expected, "1e-40");

        let complex_expected = ginsh_gamma("0.5+0.25*I", 50);
        let complex_actual =
            Complex::<Float>::try_from(parse!("gamma(1/2+1i/4)").to_float(50)).unwrap();
        assert_close_complex(&complex_actual, &complex_expected, "1e-36");
    }

    #[test]
    fn polygamma_order_zero_exact_normalization() {
        assert_eq!(parse!("polygamma(0,1)"), parse!("-euler_gamma"));
        assert_eq!(parse!("polygamma(0,1/2)"), parse!("-euler_gamma-2*log(2)"));
        assert_eq!(
            parse!("polygamma(0,5/2)"),
            parse!("8/3-euler_gamma-2*log(2)")
        );
        assert_eq!(
            parse!("polygamma(0,0)"),
            Atom::num(Coefficient::complex_infinity())
        );
    }

    #[test]
    fn polygamma_order_zero_numeric_recurrence() {
        let lhs = Complex::<Float>::try_from(
            (parse!("polygamma(0,3/2+1i/4)") - parse!("polygamma(0,1/2+1i/4)")).to_float(50),
        )
        .unwrap();
        let rhs = Complex::<Float>::try_from(parse!("1/(1/2+1i/4)").to_float(50)).unwrap();

        assert_close_complex(&lhs, &rhs, "1e-30");
    }

    #[test]
    fn special_float_inputs_normalize_immediately() {
        assert!(Complex::<Float>::try_from(parse!("gamma(1.25)")).is_ok());
        assert!(Complex::<Float>::try_from(parse!("polygamma(0,1.25)")).is_ok());
        assert!(Complex::<Float>::try_from(parse!("polygamma(1,1.25)")).is_ok());
    }

    #[test]
    fn polygamma_basic_normalization() {
        assert_eq!(
            parse!("polygamma(1,0)"),
            Atom::num(Coefficient::complex_infinity())
        );
        assert_eq!(
            parse!("polygamma(0,x)").derivative(symbol!("x")),
            parse!("polygamma(1,x)")
        );
        assert_eq!(
            parse!("polygamma(3,x)").derivative(symbol!("x")),
            parse!("polygamma(4,x)")
        );
    }

    #[test]
    fn polygamma_numeric_recurrence() {
        let lhs = Complex::<Float>::try_from(
            (parse!("polygamma(1,3/2+1i/4)") - parse!("polygamma(1,1/2+1i/4)")).to_float(50),
        )
        .unwrap();
        let rhs = Complex::<Float>::try_from(parse!("-1/(1/2+1i/4)^2").to_float(50)).unwrap();

        assert_close_complex(&lhs, &rhs, "1e-28");
    }

    #[test]
    fn polygamma_laurent_series() {
        let x = symbol!("x");
        assert_eq!(
            parse!("polygamma(0,x)")
                .series(x, Atom::num(0).as_view(), 2.into(), true)
                .unwrap()
                .to_atom(),
            parse!("-x^-1-euler_gamma+1/6*pi^2*x+1/2*polygamma(2,1)*x^2")
        );
        assert_eq!(
            parse!("polygamma(1,x)")
                .series(x, Atom::num(0).as_view(), 2.into(), true)
                .unwrap()
                .to_atom(),
            parse!("x^-2+1/6*pi^2+polygamma(2,1)*x+1/30*pi^4*x^2")
        );
    }

    #[test]
    fn polylog_basic_normalization() {
        assert_eq!(parse!("polylog(s,0)"), Atom::new());
        assert_eq!(parse!("polylog(0,x)"), parse!("x/(1-x)"));
        assert_eq!(parse!("polylog(1,x)"), parse!("-log(1-x)"));
        assert_eq!(
            parse!("polylog(3,x)").derivative(symbol!("x")),
            parse!("polylog(2,x)/x")
        );
    }

    #[test]
    fn polylog_float_normalization() {
        assert!(Complex::<Float>::try_from(parse!("polylog(2,0.5)")).is_ok());
    }

    #[test]
    fn polylog_matches_ginac_for_dilog() {
        if Command::new("ginsh").arg("--help").output().is_err() {
            return;
        }

        let expected = ginsh_polylog("2", "0.5", 50);
        let actual = Complex::<Float>::try_from(parse!("polylog(2,1/2)").to_float(50)).unwrap();
        assert_close_complex(&actual, &expected, "1e-40");
    }

    #[test]
    fn special_functions_register_eval_info() {
        let mut evaluator =
            parse!("gamma(x)+polygamma(0,x)+polygamma(1,x)+polylog(2,x)+euler_gamma")
                .evaluator(
                    &FunctionMap::new(),
                    &[parse!("x")],
                    OptimizationSettings::default(),
                )
                .unwrap()
                .map_coeff(&|x| x.re.to_f64());

        let mut out = [0.0];
        evaluator.evaluate(&[0.25], &mut out);

        assert!(out[0].is_finite());
    }

    #[test]
    fn geometric_float_inputs_normalize_immediately() {
        assert!(Complex::<Float>::try_from(parse!("tan(0.25)")).is_ok());
        assert!(Complex::<Float>::try_from(parse!("atan(0.25)")).is_ok());
        assert!(Complex::<Float>::try_from(parse!("sinh(0.25)")).is_ok());
        assert!(Complex::<Float>::try_from(parse!("atanh(0.25)")).is_ok());
        assert!(Complex::<Float>::try_from(parse!("sec(0.25)")).is_ok());
        assert!(Complex::<Float>::try_from(parse!("coth(1.25)")).is_ok());
    }

    #[test]
    fn geometric_derivatives() {
        assert_eq!(
            parse!("tan(x)").derivative(symbol!("x")),
            parse!("sec(x)^2")
        );
        assert_eq!(
            parse!("atan(x)").derivative(symbol!("x")),
            parse!("1/(1+x^2)")
        );
        assert_eq!(
            parse!("sech(x)").derivative(symbol!("x")),
            parse!("-sech(x)*tanh(x)")
        );
        assert_eq!(
            parse!("atanh(x)").derivative(symbol!("x")),
            parse!("1/(1-x^2)")
        );
    }

    #[test]
    fn tan_laurent_series() {
        let x = symbol!("x");
        assert_eq!(
            parse!("tan(x)")
                .series(x, parse!("pi/2").as_view(), 0.into(), true)
                .unwrap()
                .to_atom(),
            parse!("-(x-pi/2)^-1")
        );
        assert_eq!(
            parse!("sec(x)")
                .series(x, parse!("pi/2").as_view(), 0.into(), true)
                .unwrap()
                .to_atom(),
            parse!("-(x-pi/2)^-1")
        );
        assert_eq!(
            parse!("cot(x)")
                .series(x, Atom::num(0).as_view(), 0.into(), true)
                .unwrap()
                .to_atom(),
            parse!("x^-1")
        );
        assert_eq!(
            parse!("csc(x)")
                .series(x, Atom::num(0).as_view(), 0.into(), true)
                .unwrap()
                .to_atom(),
            parse!("x^-1")
        );
    }

    #[test]
    fn geometric_functions_register_eval_info() {
        let mut evaluator = parse!("tan(x)+atan(x)+sinh(x)+atanh(x/2)+sec(x)+coth(x+2)")
            .evaluator(
                &FunctionMap::new(),
                &[parse!("x")],
                OptimizationSettings::default(),
            )
            .unwrap()
            .map_coeff(&|x| x.re.to_f64());

        let mut out = [0.0];
        evaluator.evaluate(&[0.25], &mut out);

        assert!(out[0].is_finite());
    }

    #[test]
    fn bessel_float_inputs_normalize_immediately() {
        assert!(Complex::<Float>::try_from(parse!("bessel_j(1/2,2.0)")).is_ok());
        assert!(Complex::<Float>::try_from(parse!("bessel_y(1/2,2.0)")).is_ok());
        assert!(Complex::<Float>::try_from(parse!("bessel_i(1/2,2.0)")).is_ok());
        assert!(Complex::<Float>::try_from(parse!("bessel_k(1/2,2.0)")).is_ok());
    }

    #[test]
    fn bessel_known_half_integer_values() {
        let j = Complex::<Float>::try_from(parse!("bessel_j(1/2,2)").to_float(80)).unwrap();
        let y = Complex::<Float>::try_from(parse!("bessel_y(1/2,2)").to_float(80)).unwrap();
        let i = Complex::<Float>::try_from(parse!("bessel_i(1/2,2)").to_float(80)).unwrap();
        let k = Complex::<Float>::try_from(parse!("bessel_k(1/2,2)").to_float(80)).unwrap();

        let j_expected =
            Complex::<Float>::try_from(parse!("pi^(-1/2)*sin(2)").to_float(80)).unwrap();
        let y_expected =
            Complex::<Float>::try_from(parse!("-pi^(-1/2)*cos(2)").to_float(80)).unwrap();
        let i_expected =
            Complex::<Float>::try_from(parse!("pi^(-1/2)*sinh(2)").to_float(80)).unwrap();
        let k_expected =
            Complex::<Float>::try_from(parse!("1/2*pi^(1/2)*exp(-2)").to_float(80)).unwrap();

        assert_close_complex(&j, &j_expected, "1e-20");
        assert_close_complex(&y, &y_expected, "1e-20");
        assert_close_complex(&i, &i_expected, "1e-20");
        assert_close_complex(&k, &k_expected, "1e-18");
    }

    #[test]
    fn bessel_derivatives() {
        assert_eq!(
            parse!("bessel_j(nu,x)").derivative(symbol!("x")),
            parse!("(bessel_j(nu-1,x)-bessel_j(nu+1,x))/2")
        );
        assert_eq!(
            parse!("bessel_i(nu,x)").derivative(symbol!("x")),
            parse!("(bessel_i(nu-1,x)+bessel_i(nu+1,x))/2")
        );
    }

    #[test]
    fn bessel_functions_register_eval_info() {
        let mut evaluator =
            parse!("bessel_j(1/2,x)+bessel_y(1/2,x)+bessel_i(1/2,x)+bessel_k(1/2,x)")
                .evaluator(
                    &FunctionMap::new(),
                    &[parse!("x")],
                    OptimizationSettings::default(),
                )
                .unwrap()
                .map_coeff(&|x| x.re.to_f64());

        let mut out = [0.0];
        evaluator.evaluate(&[2.0], &mut out);

        assert!(out[0].is_finite());
    }

    fn ginsh_gamma(argument: &str, digits: u32) -> Complex<Float> {
        let script = format!("Digits={digits}:\nevalf(tgamma({argument}));\nquit;\n");

        let mut child = Command::new("ginsh")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("ginsh must be available");
        child
            .stdin
            .as_mut()
            .unwrap()
            .write_all(script.as_bytes())
            .expect("ginsh stdin must be writable");

        let output = child.wait_with_output().expect("ginsh must complete");
        assert!(
            output.status.success(),
            "ginsh failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let stdout = String::from_utf8(output.stdout).unwrap();
        let value = stdout
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty())
            .unwrap();

        parse_ginsh_complex(value)
    }

    fn ginsh_polylog(order: &str, argument: &str, digits: u32) -> Complex<Float> {
        let script = format!("Digits={digits}:\nevalf(Li({order},{argument}));\nquit;\n");

        let mut child = Command::new("ginsh")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("ginsh must be available");
        child
            .stdin
            .as_mut()
            .unwrap()
            .write_all(script.as_bytes())
            .expect("ginsh stdin must be writable");

        let output = child.wait_with_output().expect("ginsh must complete");
        assert!(
            output.status.success(),
            "ginsh failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );

        let stdout = String::from_utf8(output.stdout).unwrap();
        let value = stdout
            .lines()
            .map(str::trim)
            .find(|line| !line.is_empty())
            .unwrap();

        parse_ginsh_complex(value)
    }

    fn assert_close_complex(actual: &Complex<Float>, expected: &Complex<Float>, tolerance: &str) {
        let tol = Float::parse(tolerance, Some(256)).unwrap();
        let diff = (actual.clone() - expected.clone()).norm().re;
        let scale = expected.norm().re;
        let limit = if scale.to_f64() == 0.0 {
            tol
        } else {
            tol * scale
        };

        assert!(
            diff <= limit,
            "difference too large: actual={}, expected={}, diff={}, limit={}",
            actual,
            expected,
            diff,
            limit
        );
    }

    fn parse_ginsh_complex(value: &str) -> Complex<Float> {
        let value = value.trim();

        if !value.contains("*I") {
            return Complex::new(Float::parse(value, Some(256)).unwrap(), Float::new(256));
        }

        let imag_marker = value
            .rfind("*I")
            .expect("GiNaC complex output must contain *I");
        let core = &value[..imag_marker];
        let split = core
            .char_indices()
            .skip(1)
            .filter(|(_, ch)| *ch == '+' || *ch == '-')
            .filter(|(idx, _)| !matches!(core.as_bytes()[idx - 1], b'e' | b'E'))
            .map(|(idx, _)| idx)
            .last()
            .expect("GiNaC complex output must contain a real and imaginary part");

        let re = Float::parse(&core[..split], Some(256)).unwrap();
        let im = Float::parse(&core[split..], Some(256)).unwrap();
        Complex::new(re, im)
    }
}
