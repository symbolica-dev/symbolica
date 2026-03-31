# Numerica

<p align="center">
<a href="https://symbolica.io"><img alt="Symbolica website" src="https://img.shields.io/static/v1?label=symbolica&message=website&color=orange&style=flat-square"></a>
  <a href="https://zulip.symbolica.io"><img alt="Zulip Chat" src="https://img.shields.io/static/v1?label=zulip&message=discussions&color=blue&style=flat-square"></a>
    <a href="https://github.com/symbolica-dev/numerica"><img alt="Numerica repository" src="https://img.shields.io/static/v1?label=github&message=development&color=green&style=flat-square&logo=github"></a>
    <a href="https://app.codecov.io/gh/symbolica-dev/numerica"><img alt="Codecov" src="https://img.shields.io/codecov/c/github/symbolica-dev/numerica?token=W5GTATIVZI&style=flat-square"></a>
</p>

Numerica is an open-source mathematics library for Rust, that provides high-performance number types, such as error-tracking floats and finite field elements.

It provides
- Abstractions over rings, Euclidean domains, fields and floats
- High-performance Integer with automatic up-and-downgrading to arbitrary precision types
- Rational numbers with reconstruction algorithms
- Fast finite field arithmetic
- Error-tracking floating point types
- Generic dual numbers for automatic (higher-order) differentiation
- Matrix operations and linear system solving
- Numerical integration using Vegas algorithm with discrete layer support

For operations on symbols, check out the parent project [Symbolica](https://symbolica.io).


# Examples


### Solving a linear system

Solve a linear system over the rationals:

```rust
let a = Matrix::from_linear(
    vec![
        1.into(), 2.into(), 3.into(),
        4.into(), 5.into(), 16.into(),
        7.into(), 8.into(), 9.into(),
    ],
    3, 3, Q,
)
.unwrap();

let b = Matrix::new_vec(vec![1.into(), 2.into(), 3.into()], Q);

let r = a.solve(&b).unwrap();
assert_eq!(r.into_vec(), [(-1, 3), (2, 3), (0, 1)]);
```
Solution: $(-1/3, 2/3, 0)$.

Solve over the finite field $\mathbb{Z}_7$:

```rust
let z_7 = Zp::new(7);

let a = Matrix::from_linear(
    vec![
        z_7.to_element(5), z_7.to_element(8),
        z_7.to_element(2), z_7.to_element(1),
    ],
    2, 2, z_7,
)
.unwrap();

let r = a.inv();
assert_eq!(r.into_vec(), [z_7.to_element(4), z_7.to_element(3), z_7.to_element(2), z_7.to_element(0)]);
```

### Error-tracking floating points

Wrap `f64` and `Float` in an `ErrorPropagatingFloat` to propagate errors through
your computation. For example, a number with 60 accurate digits only has 10 remaining after the following operations:

```rust
let a = ErrorPropagatingFloat::new(Float::with_val(200, 1e-50), 60.);
let r = (a.exp() - a.one()) / a; // large cancellation
assert_eq!(format!("{r}"), "1.000000000");
assert_eq!(r.get_precision(), Some(10.205999132796238));
```

### Automatic differentiation with dual numbers

Create a dual number that fits your needs (supports multiple variables and higher-order differentiation).
Here, we create a simple dual number in three variables:

```rust
create_hyperdual_single_derivative!(Dual, 3);

fn main() {
    let x = Dual::<Rational>::new_variable(0, (1, 1).into());
    let y = Dual::new_variable(1, (2, 1).into());
    let z = Dual::new_variable(2, (3, 1).into());

    let t3 = x * y * z;

    println!("{}", t3.inv());
}
```
It yields `(1/6)+(-1/6)*ε0+(-1/12)*ε1+(-1/18)*ε2`.

The multiplication table is computed and unrolled at compile time for maximal performance.

### Solve integer relations

Solve 

$$
-32.0177 c_1 + 3.1416 c_2 + 2.7183 c_3 = 0
$$

over the integers using PSLQ:

```rust
let result = Integer::solve_integer_relation(
    &[F64::from(-32.0177), F64::from(3.1416), F64::from(2.7183)],
    F64::from(1e-4),
    1,
    Some(Integer::from(100000u64)),
    None,
)
.unwrap();

assert_eq!(result, &[1, 5, 6]);
```

Or via LLL basis reduction:

```rust
let v1 = Vector::new(vec![1.into(), 0.into(), 0.into(), 31416.into()], Z);
let v2 = Vector::new(vec![0.into(), 1.into(), 0.into(), 27183.into()], Z);
let v3 = Vector::new(vec![0.into(), 0.into(), 1.into(), (-320177).into()], Z);

let basis = Vector::basis_reduction(&[v1, v2, v3], (3, 4).into());

assert_eq!(basis[0].into_vec(), [5, 6, 1, 1]);
```

### Numerical integration

Integrate $sin(x \pi) + y$ using the Vegas algorithm and using random numbers suitable for Monte Carlo integration:

```rust
let f = |x: &[f64]| (x[0] * std::f64::consts::PI).sin() + x[1];
let mut grid = Grid::Continuous(ContinuousGrid::new(2, 128, 100, None, false));
let mut rng = MonteCarloRng::new(0, 0);
let mut sample = Sample::new();
for iteration in 1..20 {
     for _ in 0..10_000 {
         grid.sample(&mut rng, &mut sample);
    
         if let Sample::Continuous(_cont_weight, xs) = &sample {
             grid.add_training_sample(&sample, f(xs)).unwrap();
         }
     }
    
     grid.update(1.5, 1.5);
    
     println!(
         "Integral at iteration {}: {}",
         iteration,
         grid.get_statistics().format_uncertainty()
     );
}
```




## Development

Follow the development and discussions on [Zulip](https://reform.zulipchat.com)!