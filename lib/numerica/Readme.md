# Numerica

<p align="center">
<a href="https://symbolica.io"><img alt="Symbolica website" src="https://img.shields.io/static/v1?label=symbolica&message=website&color=orange&style=flat-square"></a>
  <a href="https://zulip.symbolica.io"><img alt="Zulip Chat" src="https://img.shields.io/static/v1?label=zulip&message=discussions&color=blue&style=flat-square"></a>
    <a href="https://github.com/symbolica-dev/numerica"><img alt="Numerica repository" src="https://img.shields.io/static/v1?label=github&message=development&color=green&style=flat-square&logo=github"></a>
    <a href="https://app.codecov.io/gh/symbolica-dev/numerica"><img alt="Codecov" src="https://img.shields.io/codecov/c/github/symbolica-dev/numerica?token=W5GTATIVZI&style=flat-square"></a>
</p>

Numerica is an open-source mathematics library for Rust. Compute exactly with
integers and fractions, track numerical errors, and differentiate calculations
automatically.

- Fast integers that grow to arbitrary precision when needed
- Exact rational numbers and finite-field arithmetic
- Arbitrary-precision, complex, and SIMD floating-point numbers
- Error tracking and interval arithmetic
- Automatic differentiation, including higher derivatives
- Matrix operations, linear systems, and integer relations
- Adaptive Monte Carlo integration

For symbolic expressions, explore [Symbolica](https://symbolica.io).

## Get started

```sh
cargo add numerica
```

Browse the [API documentation](https://docs.rs/numerica) for the full set of
operations and the [examples](examples) for more complete applications.

## Migrating to 3.1

Version 3.1.0 renames the numeric traits:

| Previous name | New name |
| --- | --- |
| `FloatLike` | `Numeric` |
| `SingleFloat` | `Scalar` |
| `RealLike` | `RealScalar` |
| `Real` | `ElementaryFunctions` |

The previous names remain aliases, so existing imports and trait bounds still work.
However, **3.1.0 includes a breaking change for custom trait implementations**:
`conj()` moves from `Real` (`ElementaryFunctions`) to `FloatLike` (`Numeric`). Move
its implementation into your `Numeric` implementation. Types implementing only
`Numeric` must now implement `conj()` too; real types can return `self.clone()`.
For dual numbers, conjugate every derivative component as well as the primal value.
Replace qualified calls such as `Real::conj(&value)` with `Numeric::conj(&value)`,
and bring `Numeric` into scope for method calls.

`Numeric::norm_squared()` returns the same numeric type, including for exact
complex values. The existing inherent `Complex<T>::norm_squared()` still returns
`T`; use `Numeric::norm_squared(&value)` to request the trait method explicitly.
`Numeric::is_self_conjugate()` tests whether conjugation is known to leave a value
unchanged; `false` can mean unknown. It does not assert that the value is a scalar.

## Examples

### Exact solutions

Solve a linear system without rounding errors:

```rust
use numerica::{domains::rational::Q, tensors::matrix::Matrix};

let a = Matrix::from_linear(
    vec![
        1.into(), 2.into(), 3.into(),
        4.into(), 5.into(), 16.into(),
        7.into(), 8.into(), 9.into(),
    ],
    3, 3, Q,
).unwrap();

let b = Matrix::new_vec(vec![1.into(), 2.into(), 3.into()], Q);
let solution = a.solve(&b).unwrap();

assert_eq!(solution.into_vec(), [(-1, 3), (2, 3), (0, 1)]);
```

The solution is exactly $(-1/3, 2/3, 0)$.

### See when digits are lost

Subtracting nearly equal numbers can erase much of their accuracy.
Numerica tracks an estimate of how many digits remain:

```rust
use numerica::domains::float::{ErrorPropagatingFloat, Float, Numeric, ElementaryFunctions};

let a = ErrorPropagatingFloat::new(Float::parse("1e-50", Some(200)).unwrap(), 60.);
let result = (a.exp() - a.one()) / a;

println!("{result}");
println!("About {:.0} reliable digits remain", result.get_precision().unwrap());
```

The result is close to one, but only about ten of the original sixty digits
remain reliable.

### Differentiate with dual numbers

Dual numbers carry derivatives alongside ordinary values. Here, one calculation
finds the value and all three first derivatives of $1/(xyz)$:

```rust
use numerica::{
    create_hyperdual_single_derivative,
    domains::{float::Numeric, rational::Rational},
};

create_hyperdual_single_derivative!(Dual, 3);

fn main() {
    let x = Dual::<Rational>::new_variable(0, 1.into());
    let y = Dual::new_variable(1, 2.into());
    let z = Dual::new_variable(2, 3.into());

    println!("{}", (x * y * z).inv());
}
```

At $(1, 2, 3)$, the value is $1/6$ and the derivatives are
$(-1/6, -1/12, -1/18)$.

## Development

Numerica is MIT licensed. Join the development and discussions on
[Zulip](https://zulip.symbolica.io)!
