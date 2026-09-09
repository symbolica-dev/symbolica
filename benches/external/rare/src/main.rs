//! Upstream scaling reconstruction on the shared cached-power Q oracle.
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256StarStar;
use rare::{
    algebra::{poly::flat::FlatPoly, rat::Rat},
    rec::{
        primes::LARGE_PRIMES,
        rat::thiele_multivar::{Rec, Status},
    },
    traits::{One, Zero},
    Z64,
};
use rug::{integer::IntegerExt64, Integer};
use seq_macro::seq;
use std::{fmt::Write as _, fs, ops::ControlFlow::*, path::Path, process::Command, time::Instant};

struct Term {
    coefficient: Integer,
    powers: Vec<usize>,
}
struct Input {
    names: Vec<String>,
    terms: [Vec<Term>; 2],
    degrees: Vec<usize>,
}
impl Input {
    fn read(path: &Path) -> Self {
        let source = fs::read_to_string(path).expect("read Q oracle");
        let mut words = source.split_whitespace();
        let n: usize = words.next().unwrap().parse().unwrap();
        assert!(
            (1..=8).contains(&n),
            "adapter supports one to eight variables"
        );
        assert_eq!(words.next(), Some("0"), "expected integer Q oracle");
        let counts: [usize; 2] = std::array::from_fn(|_| words.next().unwrap().parse().unwrap());
        assert!(counts[1] > 0, "empty denominator");
        let names = (0..n).map(|_| words.next().unwrap().to_owned()).collect();
        let mut degrees = vec![0; n];
        let terms = counts.map(|count| {
            (0..count)
                .map(|_| {
                    let coefficient = words.next().unwrap().parse().unwrap();
                    let powers = (0..n)
                        .map(|i| {
                            let e: usize = words.next().unwrap().parse().unwrap();
                            assert!(e <= 65535, "oracle degree exceeds exported format");
                            degrees[i] = degrees[i].max(e);
                            e
                        })
                        .collect();
                    Term {
                        coefficient,
                        powers,
                    }
                })
                .collect()
        });
        assert!(words.next().is_none(), "trailing oracle data");
        Self {
            names,
            terms,
            degrees,
        }
    }
}
struct Image<const P: u64, const N: usize> {
    terms: [Vec<(Z64<P>, [usize; N])>; 2],
    powers: [Vec<Z64<P>>; N],
}
impl<const P: u64, const N: usize> Image<P, N> {
    fn new(input: &Input) -> Self {
        assert_eq!(input.names.len(), N);
        Self {
            terms: std::array::from_fn(|side| {
                input.terms[side]
                    .iter()
                    .map(|t| {
                        (
                            Z64::from(t.coefficient.mod_u64(P)),
                            std::array::from_fn(|i| t.powers[i]),
                        )
                    })
                    .collect()
            }),
            powers: std::array::from_fn(|i| vec![Z64::one(); input.degrees[i] + 1]),
        }
    }
    fn evaluate(&mut self, point: &[Z64<P>; N]) -> Option<Z64<P>> {
        for (powers, x) in self.powers.iter_mut().zip(point) {
            for i in 1..powers.len() {
                powers[i] = powers[i - 1] * *x;
            }
        }
        let eval = |terms: &[(Z64<P>, [usize; N])]| {
            terms.iter().fold(Z64::zero(), |sum, (c, ex)| {
                sum + ex
                    .iter()
                    .zip(&self.powers)
                    .fold(*c, |v, (&e, powers)| if e == 0 { v } else { v * powers[e] })
            })
        };
        Some(eval(&self.terms[0]) * eval(&self.terms[1]).try_inv()?)
    }
}
#[derive(Debug)]
enum Stop {
    Time,
    Probes,
    Primes,
    Reconstruction(String),
    Identity(String),
}
impl Stop {
    fn status(&self) -> &'static str {
        match self {
            Self::Time => "time_limit",
            Self::Probes => "probe_limit",
            Self::Primes => "prime_limit",
            Self::Reconstruction(_) => "reconstruction_error",
            Self::Identity(_) => "identity_check_failed",
        }
    }
}
struct Stats {
    start: Instant,
    timeout: f64,
    cap: usize,
    max_primes: usize,
    probes: usize,
    by_prime: Vec<(u64, usize)>,
}
impl Stats {
    fn probe<const P: u64, const N: usize>(
        &mut self,
        image: &mut Image<P, N>,
        point: &[Z64<P>; N],
    ) -> Result<Option<Z64<P>>, Stop> {
        if self.start.elapsed().as_secs_f64() >= self.timeout {
            return Err(Stop::Time);
        }
        if self.probes >= self.cap {
            return Err(Stop::Probes);
        }
        if self.by_prime.last().map(|p| p.0) != Some(P) {
            self.by_prime.push((P, 0));
        }
        self.probes += 1;
        self.by_prime.last_mut().unwrap().1 += 1;
        Ok(image.evaluate(point))
    }
}
fn reconstruct<const N: usize>(
    input: &Input,
    stats: &mut Stats,
    seed: u64,
) -> Result<Rat<FlatPoly<Integer, N>>, Stop> {
    if stats.max_primes == 0 {
        return Err(Stop::Primes);
    }
    let mut rng = Xoshiro256StarStar::seed_from_u64(seed);
    const FIRST: u64 = LARGE_PRIMES[0];
    // Same extra-point setting and sampling driver as the authors' scaling-rec.
    // The library receives no source coefficients, powers, or degree bounds.
    let mut rec: Rec<FIRST, N> = Rec::with_random_shift(1, &mut rng);
    let mut image = Image::<FIRST, N>::new(input);
    let (mut point, mut value) = loop {
        let point = std::array::from_fn(|_| rng.gen());
        if let Some(value) = stats.probe(&mut image, &point)? {
            break (point, value);
        }
    };
    let first_scaling_prime = loop {
        match rec
            .add_pt(point, value)
            .map_err(|e| Stop::Reconstruction(e.to_string()))?
        {
            Continue(Status::Varying(n)) => loop {
                point[n] += Z64::one();
                if let Some(v) = stats.probe(&mut image, &point)? {
                    value = v;
                    break;
                }
            },
            Continue(Status::Scaling) => break 0,
            Continue(Status::NextMod) => break 1,
            Break(result) => return Ok(result),
        }
    };
    // Native prime order; upstream's demonstration uses the first ten.
    seq! { M in 0..16 {{
        if M >= first_scaling_prime && M < stats.max_primes {
            const P: u64 = LARGE_PRIMES[M];
            let mut image = Image::<P, N>::new(input);
            loop {
                let t: Z64<P> = rng.gen();
                let point = rec.to_args(t).ok_or_else(|| Stop::Reconstruction("missing scaling map".into()))?;
                // Poles count as probes but do not advance reconstruction.
                let Some(value) = stats.probe(&mut image, &point)? else { continue; };
                match rec.add_pt(point, value).map_err(|e| Stop::Reconstruction(e.to_string()))? {
                    Continue(Status::Scaling) => {}, Continue(Status::NextMod) => break,
                    Continue(Status::Varying(_)) => return Err(Stop::Reconstruction("unexpected varying state".into())),
                    Break(result) => return Ok(result),
                }
            }
        }
    }}}
    Err(Stop::Primes)
}
fn serialize<const N: usize>(r: &Rat<FlatPoly<Integer, N>>, names: &[String]) -> String {
    let poly = |p: &FlatPoly<Integer, N>| {
        if p.is_empty() {
            return "0".to_owned();
        }
        let mut text = String::new();
        for (i, t) in p.terms().iter().enumerate() {
            if i != 0 {
                text.push('+');
            }
            write!(text, "({})", t.coeff).unwrap();
            for (name, e) in names.iter().zip(t.powers) {
                if e != 0 {
                    write!(text, "*{name}^{e}").unwrap();
                }
            }
        }
        text
    };
    format!("({})/({})", poly(r.num()), poly(r.den()))
}
fn verify(input: &Path, result: &Path) -> Result<(), Stop> {
    let checker = std::env::var_os("RARE_Q_CHECKER")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            std::env::current_exe()
                .unwrap()
                .parent()
                .unwrap()
                .join("check-q-result")
        });
    let mut command = if let Some(loader) = std::env::var_os("EXTERNAL_LOADER") {
        let mut command = Command::new(loader);
        command
            .arg("--library-path")
            .arg(std::env::var_os("EXTERNAL_LIBRARY_PATH").expect("loader library path"));
        command.arg(checker);
        command
    } else {
        Command::new(checker)
    };
    let checked = command
        .arg(input)
        .arg(result)
        .output()
        .map_err(|e| Stop::Identity(e.to_string()))?;
    eprint!(
        "{}{}",
        String::from_utf8_lossy(&checked.stdout),
        String::from_utf8_lossy(&checked.stderr)
    );
    if checked.status.success() {
        Ok(())
    } else {
        Err(Stop::Identity(checked.status.to_string()))
    }
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(
        args.len(),
        5,
        "usage: rare-q-stress ORACLE_FILE CASE SEED RESULT_FILE"
    );
    let input = Input::read(Path::new(&args[1]));
    let seed: u64 = args[3].parse().expect("seed");
    let mut stats = Stats {
        start: Instant::now(),
        timeout: std::env::var("BENCH_TIMEOUT")
            .unwrap_or_else(|_| "180".into())
            .parse()
            .unwrap(),
        cap: std::env::var("MAX_TOTAL_PROBES")
            .unwrap_or_else(|_| "2000000".into())
            .parse()
            .unwrap(),
        max_primes: std::env::var("MAX_PRIMES")
            .unwrap_or_else(|_| "16".into())
            .parse()
            .unwrap(),
        probes: 0,
        by_prime: Vec::new(),
    };
    assert!(stats.timeout.is_finite() && stats.timeout >= 0.0);
    assert!(
        stats.max_primes <= 16,
        "adapter supports at most sixteen native primes"
    );
    // End timing before serialization and independent exact identity checking.
    let (candidate, elapsed) = match input.names.len() {
        1 => {
            let r = reconstruct::<1>(&input, &mut stats, seed);
            let t = stats.start.elapsed();
            (r.map(|r| serialize(&r, &input.names)), t)
        }
        2 => {
            let r = reconstruct::<2>(&input, &mut stats, seed);
            let t = stats.start.elapsed();
            (r.map(|r| serialize(&r, &input.names)), t)
        }
        3 => {
            let r = reconstruct::<3>(&input, &mut stats, seed);
            let t = stats.start.elapsed();
            (r.map(|r| serialize(&r, &input.names)), t)
        }
        4 => {
            let r = reconstruct::<4>(&input, &mut stats, seed);
            let t = stats.start.elapsed();
            (r.map(|r| serialize(&r, &input.names)), t)
        }
        5 => {
            let r = reconstruct::<5>(&input, &mut stats, seed);
            let t = stats.start.elapsed();
            (r.map(|r| serialize(&r, &input.names)), t)
        }
        6 => {
            let r = reconstruct::<6>(&input, &mut stats, seed);
            let t = stats.start.elapsed();
            (r.map(|r| serialize(&r, &input.names)), t)
        }
        7 => {
            let r = reconstruct::<7>(&input, &mut stats, seed);
            let t = stats.start.elapsed();
            (r.map(|r| serialize(&r, &input.names)), t)
        }
        8 => {
            let r = reconstruct::<8>(&input, &mut stats, seed);
            let t = stats.start.elapsed();
            (r.map(|r| serialize(&r, &input.names)), t)
        }
        _ => unreachable!(),
    };
    let outcome = candidate.and_then(|text| {
        fs::write(&args[4], text).map_err(|e| Stop::Identity(e.to_string()))?;
        verify(Path::new(&args[1]), Path::new(&args[4]))
    });
    let status = match &outcome {
        Ok(()) => "ok",
        Err(e) => e.status(),
    };
    if let Err(Stop::Reconstruction(message) | Stop::Identity(message)) = &outcome {
        eprintln!("{message}");
    }
    let distribution = stats
        .by_prime
        .iter()
        .map(|(p, n)| format!("{p}:{n}"))
        .collect::<Vec<_>>()
        .join(";");
    println!("case,method,seed,status,elapsed_us,probes,primes,probes_by_prime");
    println!(
        "{},Rare_scaling,{seed},{status},{:.3},{},{},{distribution}",
        args[2],
        elapsed.as_secs_f64() * 1e6,
        stats.probes,
        stats.by_prime.len()
    );
}
